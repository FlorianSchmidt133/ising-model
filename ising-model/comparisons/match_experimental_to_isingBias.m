function out = match_experimental_to_isingBias(varargin)
%MATCH_EXPERIMENTAL_TO_ISINGBIAS  Find which bias value of double_pulse_bias10
%   best matches each experimental condition's fraction-active trajectory.
%
%   For each condition (Naive, Beginner, Expert, ...) the script:
%     1. Builds the trial-mean fraction-active trace from BinarisedData in
%        ExperimentalData.mat (same masking as
%        analyze_ExperimentalData_FractionActive.m / plot_FractionActive_BeforeDuring.m).
%     2. Loads PerturbationResults_double_pulse_bias10_*.mat (or HDF5
%        equivalent), averages over (sims, sizes, reps) per bias value to
%        get a simulated fraction-active trace.
%     3. Aligns both traces to stim onset (experimental frame 81; Ising
%        frame pre_stim_frames+1) and clips to MatchWindow seconds.
%     4. Scores every (condition, bias_value) pair via RMSE on the
%        clipped traces.
%     5. Reports the argmin bias value per condition, draws an overlay
%        figure per condition, and writes a CSV summary.
%
%   Name-Value Parameters:
%     'ExperimentalDataFile'
%         path to ExperimentalData.mat (BinarisedData + RecordingMetadata
%         + TimingInfo). Default
%         'Fig. 5 Ising Models\ExperimentalData.mat'.
%     'PerturbationResultsPath'
%         folder containing PerturbationResults_<rawMode>_*.mat.
%         Default mba_p('IsingModelData_39x78_100K\IsingPerturbations').
%     'Mode'
%         raw Ising perturbation mode. Default 'double_pulse_bias10'.
%     'StimulusSize'
%         which Ising stimulus_size to use (px). Default 4.
%     'TargetDurationFrames'
%         which stimulus_duration entry to use. Default = the largest
%         duration that produces a stim period <= MatchWindow(2).
%     'MatchWindow'
%         [tStart tEnd] in seconds, both relative to stim onset. Default
%         [-1, 3] (matches plot_FractionActive_BeforeDuring.m).
%     'SamplingRate'  Default 10 (Hz).
%     'Conditions'    Default {'Naive','Beginner','Expert'}.
%     'OutputDir'     where to write overlay PNGs + summary CSV. Default
%         '<PerturbationResultsPath>\..\PerturbationAnalysis\BiasMatchExperiment\'.
%
%   Returns a struct with:
%     out.scores              [nConditions x nBiasValues] RMSE table
%     out.bestBiasPerCondition.<cond>  struct with .bias_value, .rmse, .png
%     out.expTraces.<cond>    [1 x nFrames] experimental trace
%     out.simTraces           [nBiasValues x nFrames] simulated traces
%     out.timeAxis            [1 x nFrames] seconds (relative to stim onset)
%
%   Example:
%     out = match_experimental_to_isingBias();
%     out = match_experimental_to_isingBias('StimulusSize', 6, ...
%               'MatchWindow', [-1.5, 4]);

    p = inputParser;
    addParameter(p, 'ExperimentalDataFile', ...
        'Fig. 5 Ising Models\ExperimentalData.mat', @ischar);
    addParameter(p, 'PerturbationResultsPath', ...
        mba_p('IsingModelData_39x78_100K\IsingPerturbations'), @ischar);
    addParameter(p, 'Mode',                'double_pulse_bias10', @ischar);
    addParameter(p, 'StimulusSize',        [], @(x) isempty(x) || isnumeric(x));
    % StimulusSizes (vector). When StimulusSize is empty (default), the
    % matcher loops over StimulusSizes and writes outputs to OutputDir/size_<N>/.
    % If StimulusSize is set explicitly (scalar), single-size mode and outputs
    % go to OutputDir/size_<N>/ for just that one size.
    addParameter(p, 'StimulusSizes',       [2, 3, 4], @isnumeric);
    % TargetStimDurSec: target physical stim duration in seconds.
    % Picks the model dur from `info.stimulusDurations` whose physical
    % length (`dur * secPerFrame`) is closest to this target. Default 2.0
    % matches the experimental 2 s stim. Has priority over
    % TargetDurationFrames; set to [] to fall back to TargetDurationFrames
    % or (if that's also []) to the legacy `largest-fits` picker.
    addParameter(p, 'TargetStimDurSec',     2.0, @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
    % TargetDurationFrames: legacy frame-count target. Kept for backward
    % compat; set to [] (the default) to use TargetStimDurSec instead.
    addParameter(p, 'TargetDurationFrames', [],  @(x) isnumeric(x) || isempty(x));
    % MatchWindow controls the data slice the matcher operates on. Widened
    % from [-1, 3] to [-1, 6] so the post-stim relaxation period (where
    % runaway sims show late ringing at t≈4-5s) is included in the full
    % window's RMSE — required for Option 2/3 joint optimization to catch
    % the Beginner-style runaway.
    addParameter(p, 'MatchWindow',         [-1, 6], @(x) isnumeric(x) && numel(x) == 2);
    % Baseline window for per-sim selection — accepts an Nx2 matrix where
    % each row is a [start, end] interval. Default unions the full pre-stim
    % period [-8, -0.1] s with a post-stim relaxation window [5, 10] s.
    % Both regions are bias-invariant (bias is zero outside stim), so the
    % sim's intrinsic dynamics (J/h field) drive both. Including the
    % post-stim relaxation in the match catches sims that ring after stim
    % offset (the Beginner-style runaway issue).
    addParameter(p, 'BaselineWindow', [-8, -0.1; 5, 10], ...
        @(x) isnumeric(x) && size(x, 2) == 2 && size(x, 1) >= 1);
    % Onset window is relative to stim onset (t=0). Default [0, 0.3] s captures
    % the rising edge as the stim region first pulses to +1 (or +bias).
    addParameter(p, 'OnsetWindow',         [0, 0.3], @(x) isnumeric(x) && numel(x) == 2);
    % Offset window is relative to stim END (t=stimEndSec). Captures the
    % second peak amplitude (the double-pulse hallmark). Slightly wider
    % than the original [0, 0.3] to cover the full peak + immediate decay.
    addParameter(p, 'OffsetWindow',        [0, 0.5], @(x) isnumeric(x) && numel(x) == 2);
    % TailWindow is relative to stim END. Captures late post-stim
    % relaxation (the runaway region) WITHOUT including the immediate
    % decay (which belongs to OffsetWindow). Used in the joint-opt
    % combined score with weight WindowWeights.tail.
    addParameter(p, 'TailWindow',          [1, 4], @(x) isnumeric(x) && numel(x) == 2);
    % StimWindow is in seconds relative to stim ONSET (t=0). Default
    % [0, 2.3] covers the stim duration plus the leading edge of the
    % offset peak (which sits around 2.0-2.3 s for the canonical 2 s stim).
    % Adds a 6th rated window 'stim' alongside full/onset/offset; downstream
    % windows can match on this for "during-stim only" RMSE without baseline
    % or tail contamination.
    addParameter(p, 'StimWindow',          [0, 2.4], @(x) isnumeric(x) && numel(x) == 2);
    % StimPeriodWindow: stim duration only, no post-offset extension.
    % Default [0, 2] = the canonical 2 s stim interval. Differs from
    % StimWindow which extends to 2.5 s to cover the second peak.
    addParameter(p, 'StimPeriodWindow',    [0, 2], @(x) isnumeric(x) && numel(x) == 2);
    % WindowWeights: combined-score weights for the joint-opt step.
    %   onset/offset → peak-amplitude match priority (default 1.0 each)
    %   tail        → runaway penalty (default 1.0; tune up to 1.5/2.0
    %                 if runaway returns; tune down to 0.5 if peaks
    %                 still under-shoot)
    addParameter(p, 'WindowWeights', struct('onset', 1.0, 'offset', 1.0, 'tail', 1.0), ...
        @(x) isstruct(x) && all(isfield(x, {'onset', 'offset', 'tail'})));
    % FoldBaselineWindow: pre-stim window for the 'fold' scoring criterion.
    % Kept narrow ([-1, -0.1]) and pre-stim only so a runaway tail in the
    % model doesn't inflate the baseline denominator and artificially
    % lower the model fold ratio.
    addParameter(p, 'FoldBaselineWindow', [-1, -0.1], @(x) isnumeric(x) && numel(x) == 2);
    addParameter(p, 'SamplingRate',        10, @isnumeric);
    addParameter(p, 'Conditions',          {'Naive', 'Beginner', 'Expert', 'NoSpout'}, @iscell);
    % BaselineAlign: when true (default), add a per-condition DC offset to
    % the experimental trace equal to (model baseline - exp baseline) so
    % both start at the same baseline level. Removes the systematic
    % baseline gap that otherwise inflates RMSE and depresses the data
    % fold ratio. The shifted exp trace propagates through ALL matching
    % modes (RMSE windows, fold variants, overlay rendering, summary CSV).
    % Pass false to recover the unshifted comparison.
    addParameter(p, 'BaselineAlign',       true, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    % OverrideBestSimIdx: optional struct for one-off "force sim X for cond Y"
    % runs. Each field name is a condition; value is a 1-indexed sim number
    % (MATLAB convention; HDF5 sim_<i> is i = simIdx-1). When set, the
    % matcher's per-sim baseline argmin is overridden AFTER the natural
    % selection runs — so subsequent scoring, overlay rendering, and
    % MatchingMode joint-opts use the forced sim. Leave empty (default)
    % for the canonical automated pick.
    addParameter(p, 'OverrideBestSimIdx', struct(), @isstruct);
    % TimeStretch: scale the model time axis per condition to align peaks
    % with the data. 'auto' (default) grid-searches the optimal scale per
    % cond by minimizing full-window RMSE at the per-sim baseline-best sim.
    % A scalar value (e.g. 1.10) skips the grid search and applies the
    % value uniformly across all conds. Stretched axis flows through all
    % downstream scoring + overlay sites.
    addParameter(p, 'TimeStretch',         'auto', @(x) ischar(x) || (isnumeric(x) && isscalar(x)));
    % TimeShift: additive shift (seconds) of model time axis per condition,
    % applied AFTER TimeStretch:  simAxis_eff = simTimeAxis*stretch - shift
    % Compensates for intrinsic model latency (e.g. Ising blob nucleation
    % lag) that pushes the model peak ~0.5-1.0 s after the data peak.
    % 'auto' jointly grid-searches stretch×shift in
    %   stretch ∈ [0.85, 1.20]  ×  shift ∈ [-1.5, +0.5] s
    % by minimising full-window RMSE at the per-sim baseline-best sim.
    % A scalar applies uniformly. 0 disables.
    addParameter(p, 'TimeShift',           'auto', @(x) ischar(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'OutputDir',           '', @ischar);
    % If true, also load the fully-clamped variant (double_pulse10) and
    % append it as the upper-boundary entry of the bias sweep, labelled
    % 'clamped' (numeric Inf in the bias-values vector). Provides the
    % "essentially infinite bias" reference point.
    addParameter(p, 'IncludeClamped',      true, @islogical);
    addParameter(p, 'ClampedMode',         'double_pulse10', @ischar);
    % Skip: per-condition list of ORIGINAL Grid40 recording indices to drop,
    % matching the canonical Figure4.m / plot_FractionActive_BeforeDuring set,
    % plus Naive rec 14 (211-frame outlier). Applied via
    % RecordingMetadata.<cond>.recording_indices on top of whatever was
    % already filtered out during Figure5_dataAggregation.
    %
    % Naive additions 2026-05-08: drop recIdx 12 (flat trace, fold ~1.02x/1.07x —
    % no detectable onset or offset peak) and recIdx 15 (only 10 trials, weak
    % onset 1.27x). These were diluting the cond-mean fold ratio and shifting
    % matcher picks toward unphysiologically high biases.
    defaultSkip = struct( ...
        'Naive',    [1 9 10 12 14 15 16], ...
        'Beginner', [1 6 7 11 12], ...
        'Expert',   [1 4 11:17], ...
        'NoSpout',  [1 4 9 10 11 14]);
    addParameter(p, 'Skip', defaultSkip, @isstruct);
    % MatchingMode controls how a "best bias" is selected:
    %   'perCondition' — each condition's own argmin RMSE per window (default).
    %   'global'       — single bias minimising sum-RMSE across non-Naive
    %                    conditions; applied to all conds.
    %   'expert'       — Expert's best bias propagated to every condition
    %                    (incl. Naive). "Force everyone through Expert's lens".
    % Sim selection (per-cond baseline RMSE) is bias-invariant and unchanged
    % across modes — only bias_value in bestBiasPerCondition is overridden.
    addParameter(p, 'MatchingMode', 'perCondition', @(x) ischar(x) && ...
        any(strcmpi(x, {'perCondition', 'global', 'expert'})));
    parse(p, varargin{:});
    opts = p.Results;

    if isempty(opts.OutputDir)
        opts.OutputDir = fullfile(opts.PerturbationResultsPath, '..', ...
            'PerturbationAnalysis', 'BiasMatchExperiment');
    end
    if ~exist(opts.OutputDir, 'dir')
        mkdir(opts.OutputDir);
    end

    %% --- 1. Experimental traces -----------------------------------------
    fprintf('Loading experimental data: %s\n', opts.ExperimentalDataFile);
    if ~exist(opts.ExperimentalDataFile, 'file')
        error('ExperimentalData.mat not found: %s', opts.ExperimentalDataFile);
    end
    SE = load(opts.ExperimentalDataFile, 'BinarisedData', 'RecordingMetadata', 'TimingInfo');
    BinarisedData     = SE.BinarisedData;
    TimingInfo        = SE.TimingInfo;
    if isfield(SE, 'RecordingMetadata')
        RecordingMetadata = SE.RecordingMetadata;
    else
        RecordingMetadata = struct();
        warning('RecordingMetadata not in file; Skip will be ignored.');
    end

    if isstruct(TimingInfo) && ~isempty(fieldnames(TimingInfo))
        prestimFrames = double(TimingInfo.trial_structure.prestim_frames(:)');
        totalFrames   = double(TimingInfo.trial_structure.total_frames);
    else
        prestimFrames = 1:80;
        totalFrames   = 185;
    end
    expStimOnsetFrame = prestimFrames(end) + 1;
    expTimeAxis = ((1:totalFrames) - expStimOnsetFrame) / opts.SamplingRate;

    expWinMask = expTimeAxis >= opts.MatchWindow(1) & expTimeAxis <= opts.MatchWindow(2);
    expWinTime = expTimeAxis(expWinMask);
    fprintf('Experimental window: %.2fs to %.2fs (%d frames)\n', ...
        expWinTime(1), expWinTime(end), numel(expWinTime));

    expTraces = struct();
    for c = 1:numel(opts.Conditions)
        cond = opts.Conditions{c};
        if ~isfield(BinarisedData, cond) || isempty(BinarisedData.(cond))
            warning('Condition %s missing from BinarisedData; skipping', cond);
            continue;
        end
        bin = BinarisedData.(cond);                  % [gY, gX, T, nTrials]
        [gY, gX, T, nTrials] = size(bin);
        recMask = any(bin > 0, [3 4]);               % active pixels in any trial
        recMask = reshape(recMask, gY, gX);
        nValid = nnz(recMask);
        if nValid == 0
            warning('No active pixels for %s; skipping', cond);
            continue;
        end

        % Per-recording traces (so the target matches Fig6c row 1 thick line:
        % equally-weighted across recordings rather than trial-pooled).
        recTraces = [];
        nKept = 0; nDropped = 0;
        if isfield(RecordingMetadata, cond)
            meta = RecordingMetadata.(cond);
            nTpr = double(meta.nTrials_per_recording(:)');
            if isfield(meta, 'recording_indices') && ~isempty(meta.recording_indices)
                recIdxList = double(meta.recording_indices(:)');
            else
                recIdxList = 1:numel(nTpr);
            end
            if isfield(opts.Skip, cond) && ~isempty(opts.Skip.(cond))
                skipSet = opts.Skip.(cond);
            else
                skipSet = [];
            end
            cursor = 0;
            for r = 1:numel(nTpr)
                n = nTpr(r);
                if n <= 0, cursor = cursor + max(n, 0); continue; end
                if cursor + n > nTrials
                    n = nTrials - cursor;
                    if n <= 0, break; end
                end
                if ismember(recIdxList(r), skipSet)
                    cursor = cursor + n;
                    nDropped = nDropped + 1;
                    continue;
                end
                recBin = bin(:, :, :, cursor + (1:n));
                cursor = cursor + n;
                recFlat = reshape(recBin, gY*gX, T, n);
                recFlat = recFlat(recMask(:), :, :);
                recTrace = reshape(mean(mean(recFlat, 1), 3), 1, []);
                recTraces = [recTraces; recTrace]; %#ok<AGROW>
                nKept = nKept + 1;
            end
        end

        if isempty(recTraces)
            % Fallback: pooled trial mean over all trials (no metadata case)
            binFlat = reshape(bin, gY*gX, T, nTrials);
            binFlat = binFlat(recMask(:), :, :);
            trace = reshape(mean(mean(binFlat, 1), 3), 1, []);
            fprintf('  %s: no Skip/metadata applied; pooled %d trials, %d active px, baseline=%.4f, peak=%.4f\n', ...
                cond, nTrials, nValid, mean(trace(prestimFrames)), max(trace));
        else
            trace = mean(recTraces, 1, 'omitnan');
            fprintf('  %s: kept %d / dropped %d recs (Skip=[%s]), %d active px, baseline=%.4f, peak=%.4f\n', ...
                cond, nKept, nDropped, num2str(skipSet), nValid, ...
                mean(trace(prestimFrames)), max(trace));
        end
        expTraces.(cond) = trace;
    end

    %% --- 2. Locate perturbation files (once, before size loop) ----------
    pattern = sprintf('PerturbationResults_%s_*.mat', opts.Mode);
    perturbFiles = dir(fullfile(opts.PerturbationResultsPath, pattern));
    if isempty(perturbFiles)
        error(['No PerturbationResults file matching %s found in %s. ' ...
               'Run run_ising_perturbations.py for mode ''%s'' first.'], ...
               pattern, opts.PerturbationResultsPath, opts.Mode);
    end
    [~, idx] = max([perturbFiles.datenum]);
    perturbFile = fullfile(perturbFiles(idx).folder, perturbFiles(idx).name);
    fprintf('Will load Ising results: %s\n', perturbFile);

    clampedFile = '';
    if opts.IncludeClamped
        clampedPattern = sprintf('PerturbationResults_%s_*.mat', opts.ClampedMode);
        clampedFiles = dir(fullfile(opts.PerturbationResultsPath, clampedPattern));
        if isempty(clampedFiles)
            warning(['IncludeClamped=true but no aggregate matching %s found ' ...
                     'in %s; proceeding without the clamped reference.'], ...
                    clampedPattern, opts.PerturbationResultsPath);
        else
            [~, ci] = max([clampedFiles.datenum]);
            clampedFile = fullfile(clampedFiles(ci).folder, clampedFiles(ci).name);
            fprintf('Will append clamped reference: %s\n', clampedFile);
        end
    end

    %% --- 3. Determine sizes to process ----------------------------------
    if isempty(opts.StimulusSize)
        sizesToProcess = opts.StimulusSizes(:)';
    else
        sizesToProcess = opts.StimulusSize(:)';
    end
    fprintf('Sizes to process: %s\n', mat2str(sizesToProcess));

    %% --- 4. Loop over sizes ---------------------------------------------
    out = struct();
    out.bySize = struct();
    out.opts   = opts;
    expCtx = struct();
    expCtx.expTraces     = expTraces;
    expCtx.expTimeAxis   = expTimeAxis;
    expCtx.expWinMask    = expWinMask;
    expCtx.expWinTime    = expWinTime;
    expCtx.prestimFrames = prestimFrames;
    for szi = 1:numel(sizesToProcess)
        sz = sizesToProcess(szi);
        opts_sz = opts;
        opts_sz.StimulusSize = sz;
        opts_sz.OutputDir = fullfile(opts.OutputDir, sprintf('size_%d', sz));
        if ~exist(opts_sz.OutputDir, 'dir'), mkdir(opts_sz.OutputDir); end
        fprintf('\n========== STIMULUS SIZE %d (%d/%d) ==========\n', ...
            sz, szi, numel(sizesToProcess));
        sizeOut = processOneSize(opts_sz, perturbFile, clampedFile, expCtx);
        out.bySize.(sprintf('size_%d', sz)) = sizeOut;
    end
end


%% =====================================================================
function sizeOut = processOneSize(opts, perturbFile, clampedFile, expCtx)
%PROCESSONESIZE  Run the full match for one stimulus size.
%   Lazy-loads the Ising aggregate's activity_crop slice for the chosen
%   size, optionally appends clamped, scores per (cond, bias, window) using
%   the per-condition best-baseline-matching sim, plots overlays + summary,
%   writes CSV. Returns a struct of results.

    sizeOut = struct();
    expTraces   = expCtx.expTraces;
    expWinMask  = expCtx.expWinMask;
    expWinTime  = expCtx.expWinTime;
    expTimeAxis = expCtx.expTimeAxis;

    fprintf('Loading Ising results for size=%d ...\n', opts.StimulusSize);
    [Sim, info] = loadIsingPerturbation(perturbFile, opts);
    if isempty(Sim) || ~isfield(Sim, opts.Conditions{1})
        warning('No Sim data for size=%d; skipping', opts.StimulusSize);
        return;
    end
    nBiasValues = numel(info.stimulusBiasValues);
    fprintf('  bias values: %s\n', mat2str(info.stimulusBiasValues));

    if ~isempty(clampedFile)
        fprintf('  appending clamped reference\n');
        Sim = appendClampedToSim(Sim, info, clampedFile, opts);
        info.stimulusBiasValues = [info.stimulusBiasValues, Inf];
        nBiasValues = numel(info.stimulusBiasValues);
    end

    simTimeAxis = ((1:info.totalFrames) - info.stimOnsetFrame) * info.secPerFrame;
    sizeIdxLocal = pickSizeIdx(info.stimulusSizes, opts.StimulusSize);
    if ~isempty(opts.TargetStimDurSec)
        durSecAvail = info.stimulusDurations * info.secPerFrame;
        [~, durIdxLocal] = min(abs(durSecAvail - opts.TargetStimDurSec));
        fprintf('  dur-pick: TargetStimDurSec=%.3fs -> dur=%d frames (=%.3fs)\n', ...
            opts.TargetStimDurSec, info.stimulusDurations(durIdxLocal), durSecAvail(durIdxLocal));
    elseif ~isempty(opts.TargetDurationFrames)
        [~, durIdxLocal] = min(abs(info.stimulusDurations - opts.TargetDurationFrames));
    else
        durIdxLocal = pickDurationIdx(info.stimulusDurations, opts.MatchWindow, info.secPerFrame);
    end
    stimEndSec = info.stimulusDurations(min(durIdxLocal, numel(info.stimulusDurations))) * info.secPerFrame;
    info.stimEndSec = stimEndSec;
    fprintf('  size_idx=%d, dur=%d frames (=%.2fs)\n', ...
        sizeIdxLocal, info.stimulusDurations(durIdxLocal), stimEndSec);

    %% Build per-(cond, sim, bias) trace
    nCond = numel(opts.Conditions);
    nSims = NaN;
    for c = 1:nCond
        cond = opts.Conditions{c};
        if isfield(Sim, cond) && isfield(Sim.(cond), 'activity')
            nSims = size(Sim.(cond).activity, 1);
            break;
        end
    end
    if isnan(nSims)
        warning('No Sim activity for size=%d; skipping', opts.StimulusSize);
        return;
    end
    nFrames = info.totalFrames;

    simTracesCondSim = nan(nCond, nSims, nBiasValues, nFrames);
    for c = 1:nCond
        cond = opts.Conditions{c};
        if ~isfield(Sim, cond), continue; end
        actData = Sim.(cond).activity;     % [nSims, 1, nBiasValues, nReps, nFrames]
        if ndims(actData) < 5, continue; end
        for s = 1:nSims
            for b = 1:nBiasValues
                slice = squeeze(actData(s, sizeIdxLocal, b, :, :));   % [nReps, nFrames]
                if isvector(slice), slice = reshape(slice, [], nFrames); end
                simTracesCondSim(c, s, b, :) = mean(slice, 1, 'omitnan');
            end
        end
    end

    %% --- BaselineAlign: shift experimental trace to model baseline -----
    %  When enabled (default), compute one DC offset per condition equal to
    %  (mean model baseline) - (mean exp baseline) and add it to expTraces.
    %  Every site downstream of expTraces (RMSE scoring, fold computation,
    %  overlay) reads the shifted trace transparently — no further plumbing.
    blRow = opts.BaselineWindow(1, :);   % use pre-stim row only (post-stim couples to runaway dynamics)
    simBaselineMask = simTimeAxis >= blRow(1) & simTimeAxis <= blRow(2);
    expBaselineMask = expTimeAxis >= blRow(1) & expTimeAxis <= blRow(2);
    dcShift = zeros(nCond, 1);
    if opts.BaselineAlign
        fprintf('BaselineAlign=true: shifting experimental traces to match model baseline.\n');
        for c = 1:nCond
            cond = opts.Conditions{c};
            if ~isfield(expTraces, cond), continue; end
            simSlab = simTracesCondSim(c, :, :, simBaselineMask);
            simBaselineMean = mean(simSlab(:), 'omitnan');
            expBaseline = mean(expTraces.(cond)(expBaselineMask), 'omitnan');
            if isnan(simBaselineMean) || isnan(expBaseline)
                fprintf('  %s: insufficient baseline data; skipping shift\n', cond);
                continue;
            end
            dcShift(c) = simBaselineMean - expBaseline;
            expTraces.(cond) = expTraces.(cond) + dcShift(c);
            fprintf('  %s: shift +%.4f (sim baseline=%.4f, exp baseline=%.4f)\n', ...
                cond, dcShift(c), simBaselineMean, expBaseline);
        end
    end

    %% Per-sim baseline selection per condition
    %  IMPORTANT: baseline RMSE is scored on the FULL experimental time axis
    %  (not just the MatchWindow slice) so that BaselineWindow intervals
    %  outside MatchWindow (e.g. [-8, -1] or [5, 10] s) actually contribute.
    %  The bias-rated windows below still use expWinMask/expWinTime as
    %  before — only the baseline path is widened.
    bestSimIdx = nan(nCond, 1);
    bestBaselineRmse = nan(nCond, 1);
    perSimBaselineRmse = nan(nCond, nSims);
    for c = 1:nCond
        cond = opts.Conditions{c};
        if ~isfield(expTraces, cond), continue; end
        expFull = expTraces.(cond)(:);            % full trial trace
        ttFull  = expTimeAxis(:);                 % full trial time axis
        for s = 1:nSims
            simBiasMean = squeeze(mean(simTracesCondSim(c, s, :, :), 3, 'omitnan'));
            simSliceFull = interp1(simTimeAxis, simBiasMean(:)', ttFull, 'linear', NaN);
            simSliceFull = simSliceFull(:);
            n = min(numel(expFull), numel(simSliceFull));
            if n < 2, continue; end
            es = expFull(1:n); ss = simSliceFull(1:n);
            tt = ttFull(1:n);
            % Union mask over all rows of BaselineWindow (Nx2)
            inWin = false(size(tt));
            for k = 1:size(opts.BaselineWindow, 1)
                inWin = inWin | ...
                    (tt >= opts.BaselineWindow(k, 1) & tt <= opts.BaselineWindow(k, 2));
            end
            mask = inWin & ~isnan(es) & ~isnan(ss);
            if nnz(mask) < 2, continue; end
            perSimBaselineRmse(c, s) = sqrt(mean((es(mask) - ss(mask)) .^ 2));
        end
        [v, sBest] = min(perSimBaselineRmse(c, :));
        if ~isnan(v)
            bestSimIdx(c) = sBest;
            bestBaselineRmse(c) = v;
        end
        % OverrideBestSimIdx: force-pick a specific sim for this cond,
        % bypassing the per-sim baseline argmin. Used for one-off override
        % runs (e.g. visually-preferred Beginner sim_2 for full_naive).
        if isfield(opts.OverrideBestSimIdx, cond)
            forced = opts.OverrideBestSimIdx.(cond);
            if isnumeric(forced) && isscalar(forced) && forced >= 1 && forced <= nSims
                origSim = bestSimIdx(c);
                bestSimIdx(c) = forced;
                if isfinite(perSimBaselineRmse(c, forced))
                    bestBaselineRmse(c) = perSimBaselineRmse(c, forced);
                end
                fprintf('  %-10s OVERRIDE: forced sim_%d (was sim_%d auto)\n', ...
                    cond, forced - 1, origSim - 1);
            else
                warning('OverrideBestSimIdx.%s = %s: invalid; ignoring', cond, mat2str(forced));
            end
        end
        if isnan(bestSimIdx(c))
            fprintf('  %-10s no usable sim for baseline match\n', cond);
        else
            fprintf('  %-10s best baseline sim: sim_%d (RMSE=%.4f)\n', ...
                cond, bestSimIdx(c)-1, bestBaselineRmse(c));
        end
    end

    %% Build per-condition simTraces using chosen sim
    simTraces = nan(nCond, nBiasValues, nFrames);
    for c = 1:nCond
        if isnan(bestSimIdx(c)), continue; end
        simTraces(c, :, :) = squeeze(simTracesCondSim(c, bestSimIdx(c), :, :));
    end

    %% Per-cond TimeStretch + TimeShift fit
    %  Align model time axis to data:  simAxis_eff = simTimeAxis*stretch - shift
    %  'auto' jointly grid-searches  stretch∈[0.50,2.00]×0.05  ×  shift∈[-1.5,+1.5]×0.05 s
    %  minimising full-window RMSE between data and the cond's HIGHEST-BIAS
    %  sim trace at bestSimIdx(c) — the high-bias trace has the clearest
    %  peak, anchoring the alignment to the actual stim response rather
    %  than baseline noise. Scalar values applied uniformly.
    %  Aligned axis is re-used for every (sim, bias, window) score below.
    timeStretch = ones(nCond, 1);
    timeShift   = zeros(nCond, 1);
    stretchAuto = (ischar(opts.TimeStretch) || isstring(opts.TimeStretch)) && strcmpi(string(opts.TimeStretch), 'auto');
    shiftAuto   = (ischar(opts.TimeShift)   || isstring(opts.TimeShift))   && strcmpi(string(opts.TimeShift),   'auto');
    if stretchAuto && ~shiftAuto && isnumeric(opts.TimeShift)
        timeShift = ones(nCond, 1) * opts.TimeShift;
    elseif shiftAuto && ~stretchAuto && isnumeric(opts.TimeStretch)
        timeStretch = ones(nCond, 1) * opts.TimeStretch;
    elseif ~stretchAuto && isnumeric(opts.TimeStretch)
        timeStretch = ones(nCond, 1) * opts.TimeStretch;
    end
    if ~shiftAuto && isnumeric(opts.TimeShift)
        timeShift = ones(nCond, 1) * opts.TimeShift;
    end
    if stretchAuto || shiftAuto
        if stretchAuto
            stretchGrid = 0.50:0.05:2.00;
        else
            stretchGrid = unique(timeStretch(:))';
        end
        if shiftAuto
            shiftGrid = -1.5:0.05:1.5;
        else
            shiftGrid = unique(timeShift(:))';
        end
        stretchModeStr = 'fixed'; if stretchAuto, stretchModeStr = 'auto'; end
        shiftModeStr   = 'fixed'; if shiftAuto,   shiftModeStr   = 'auto'; end
        fprintf('\nTime alignment fit: stretch∈[%.2f,%.2f] (%s)  shift∈[%.2f,%.2f]s (%s)\n', ...
            min(stretchGrid), max(stretchGrid), stretchModeStr, ...
            min(shiftGrid),   max(shiftGrid),   shiftModeStr);
        %  Time-alignment reference: use the HIGHEST-BIAS trace (clearest peak)
        %  rather than mean-across-biases. The bias-mean trace at the
        %  baseline-best sim is dominated by near-zero biases that have no
        %  visible peak, causing the shift optimizer to align noise instead
        %  of the actual stim response. Picking the strongest-driven trace
        %  guarantees a well-defined peak to anchor the alignment to.
        %  (Find the highest finite bias index; clamped=Inf is excluded
        %  because its trace can saturate and be flat.)
        finiteBiasMask = isfinite(info.stimulusBiasValues);
        if any(finiteBiasMask)
            [~, refBiasIdx] = max(info.stimulusBiasValues .* finiteBiasMask);
        else
            refBiasIdx = 1;
        end
        for c = 1:nCond
            cond = opts.Conditions{c};
            if ~isfield(expTraces, cond) || isnan(bestSimIdx(c)), continue; end
            es = expTraces.(cond)(expWinMask); es = es(:);
            simRefTrace = squeeze(simTracesCondSim(c, bestSimIdx(c), refBiasIdx, :));
            bestRmseTS = Inf; bestTS = 1.0; bestSh = 0.0;
            for ts = stretchGrid
                for sh = shiftGrid
                    simAxis_eff = simTimeAxis * ts - sh;
                    simSlice = interp1(simAxis_eff, simRefTrace(:)', expWinTime, 'linear', NaN);
                    simSlice = simSlice(:);
                    mk = ~isnan(es) & ~isnan(simSlice);
                    if nnz(mk) < 5, continue; end
                    rmse = sqrt(mean((es(mk) - simSlice(mk)).^2));
                    if rmse < bestRmseTS
                        bestRmseTS = rmse; bestTS = ts; bestSh = sh;
                    end
                end
            end
            timeStretch(c) = bestTS;
            timeShift(c)   = bestSh;
            fprintf('  %-10s stretch=%.3f shift=%+.2fs (RMSE=%.4f at sim_%d bias=%.2f)\n', ...
                cond, bestTS, bestSh, bestRmseTS, bestSimIdx(c)-1, info.stimulusBiasValues(refBiasIdx));
        end
    end
    if ~stretchAuto && isnumeric(opts.TimeStretch) && opts.TimeStretch ~= 1.0
        fprintf('\nTimeStretch=%.3f (uniform across conds)\n', opts.TimeStretch);
    end
    if ~shiftAuto && isnumeric(opts.TimeShift) && opts.TimeShift ~= 0.0
        fprintf('TimeShift=%+.3fs (uniform across conds)\n', opts.TimeShift);
    end

    %% Score 8 windows: baseline, full, onset, offset, tail, stim, BothPeaks, StimPeriod
    %  BothPeaks: union of [0, 0.3] (onset peak) and stimEndSec + [0, 0.3]
    %  (offset peak's leading edge — narrower than the standalone offset
    %  window's [0, 0.5] which extends through the peak's decay). Captures
    %  peak-rise-only RMSE, excluding the during-stim plateau and tail.
    windowDefs = struct( ...
        'baseline',   opts.BaselineWindow, ...
        'full',       opts.MatchWindow, ...
        'onset',      opts.OnsetWindow, ...
        'offset',     stimEndSec + opts.OffsetWindow, ...
        'tail',       stimEndSec + opts.TailWindow, ...
        'stim',       opts.StimWindow, ...
        'BothPeaks',  [0, 0.4; stimEndSec + 0, stimEndSec + 0.4], ...
        'StimPeriod', opts.StimPeriodWindow);
    winNames = fieldnames(windowDefs);
    nWin = numel(winNames);

    % Score per (cond, sim, bias, window) — 4D table used for joint
    % (sim, bias) optimization in the MatchingMode post-process. The legacy
    % 3D `scores(c, b, w)` is then derived as scores4D(c, bestSimIdx(c), b, w)
    % so existing per-best-sim consumers (CSV summary, log lines) keep
    % working unchanged.
    scores4D = nan(nCond, nSims, nBiasValues, nWin);
    for c = 1:nCond
        cond = opts.Conditions{c};
        if ~isfield(expTraces, cond), continue; end
        expSlice = expTraces.(cond)(expWinMask);
        expSlice = expSlice(:);
        simTimeAxisC = simTimeAxis * timeStretch(c) - timeShift(c);   % cond-specific stretched+shifted axis
        for s = 1:nSims
            for b = 1:nBiasValues
                simBiasMean = squeeze(simTracesCondSim(c, s, b, :));
                simSlice = interp1(simTimeAxisC, simBiasMean(:)', expWinTime, 'linear', NaN);
                simSlice = simSlice(:);
                n = min(numel(expSlice), numel(simSlice));
                if n < 2, continue; end
                es = expSlice(1:n); ss = simSlice(1:n);
                tt = expWinTime(:); tt = tt(1:n);
                for w = 1:nWin
                    wlim = windowDefs.(winNames{w});
                    if size(wlim, 1) >= 2
                        inWin = false(size(tt));
                        for k = 1:size(wlim, 1)
                            inWin = inWin | (tt >= wlim(k, 1) & tt <= wlim(k, 2));
                        end
                    else
                        inWin = tt >= wlim(1) & tt <= wlim(2);
                    end
                    mask = inWin & ~isnan(es) & ~isnan(ss);
                    if nnz(mask) < 2, continue; end
                    scores4D(c, s, b, w) = sqrt(mean((es(mask) - ss(mask)) .^ 2));
                end
            end
        end
    end
    % Derive legacy 3D `scores` from the 4D table at bestSimIdx (baseline-best sim)
    scores = nan(nCond, nBiasValues, nWin);
    for c = 1:nCond
        if isnan(bestSimIdx(c)), continue; end
        scores(c, :, :) = squeeze(scores4D(c, bestSimIdx(c), :, :));
    end

    %% Pick best bias per (cond, window) — baseline window EXCLUDED.
    %  Bias only fires during stim, so pre-stim activity is bias-invariant
    %  by construction. We score baseline RMSE per bias only as a diagnostic
    %  (CSV rows present, IsBest always 0). The baseline window's actual
    %  utility is upstream: choosing the best-fitting sim out of the top-10
    %  parameter matches per condition (handled in the per-sim selection
    %  step above).
    biasRatedWindows = {'full', 'onset', 'offset', 'stim', 'BothPeaks', 'StimPeriod'};
    bestBiasPerCondition = struct();
    summaryRows = {};
    for c = 1:nCond
        cond = opts.Conditions{c};
        if all(isnan(scores(c, :, :)), 'all'), continue; end
        bestBiasPerCondition.(cond) = struct();
        if ~isnan(bestSimIdx(c))
            bestBiasPerCondition.(cond).bestSimIdx = bestSimIdx(c);
            bestBiasPerCondition.(cond).baselineRmse = bestBaselineRmse(c);
        end
        for w = 1:nWin
            wn = winNames{w};
            scoreVec = squeeze(scores(c, :, w));
            if all(isnan(scoreVec)), continue; end
            isRated = any(strcmp(wn, biasRatedWindows));
            if isRated
                [bestRmse, bIdx] = min(scoreVec);
                bestVal = info.stimulusBiasValues(bIdx);
                bestBiasPerCondition.(cond).(wn).bias_value = bestVal;
                bestBiasPerCondition.(cond).(wn).rmse       = bestRmse;
            else
                bIdx = NaN;   % no winner for un-rated windows (e.g. baseline)
            end
            for b = 1:nBiasValues
                simIdxOut = -1;
                if ~isnan(bestSimIdx(c)), simIdxOut = bestSimIdx(c) - 1; end
                isBest = isRated && (b == bIdx);
                summaryRows(end+1, :) = {cond, opts.StimulusSize, simIdxOut, ...
                    info.stimulusBiasValues(b), wn, scores(c, b, w), ...
                    isBest}; %#ok<AGROW>
            end
        end
        if all(isfield(bestBiasPerCondition.(cond), {'full','onset','offset'}))
            simIdxLog = NaN;
            if ~isnan(bestSimIdx(c)), simIdxLog = bestSimIdx(c) - 1; end
            extraLog = '';
            extraKeysLog  = {'stim', 'BothPeaks', 'StimPeriod'};
            for el = 1:numel(extraKeysLog)
                ek = extraKeysLog{el};
                if isfield(bestBiasPerCondition.(cond), ek)
                    extraLog = sprintf('%s | %s=%-7s (%.4f)', extraLog, ek, ...
                        biasLabel(bestBiasPerCondition.(cond).(ek).bias_value), ...
                        bestBiasPerCondition.(cond).(ek).rmse);
                end
            end
            fprintf('Best bias for %-10s [sim=%d] : full=%-7s (%.4f) | on=%-7s (%.4f) | off=%-7s (%.4f)%s\n', ...
                cond, simIdxLog, ...
                biasLabel(bestBiasPerCondition.(cond).full.bias_value),   bestBiasPerCondition.(cond).full.rmse, ...
                biasLabel(bestBiasPerCondition.(cond).onset.bias_value),  bestBiasPerCondition.(cond).onset.rmse, ...
                biasLabel(bestBiasPerCondition.(cond).offset.bias_value), bestBiasPerCondition.(cond).offset.rmse, ...
                extraLog);
        end
    end

    %% Build combined score: w_onset × onset_rmse + w_offset × offset_rmse + w_tail × tail_rmse
    %  Drives the joint (sim, bias) optimization for all 3 MatchingModes.
    %  Decouples peak-amplitude match (onset/offset) from runaway penalty
    %  (tail) — neither extreme dominates as long as weights are balanced.
    onsetIdx  = find(strcmp(winNames, 'onset'),  1);
    offsetIdx = find(strcmp(winNames, 'offset'), 1);
    tailIdx   = find(strcmp(winNames, 'tail'),   1);
    fullIdx   = find(strcmp(winNames, 'full'),   1);   %#ok<NASGU> (kept for diagnostics)
    w_on  = opts.WindowWeights.onset;
    w_off = opts.WindowWeights.offset;
    w_t   = opts.WindowWeights.tail;
    fprintf('Combined-score weights: onset=%.2f, offset=%.2f, tail=%.2f\n', w_on, w_off, w_t);
    combinedScore = w_on * scores4D(:, :, :, onsetIdx) + ...
                    w_off * scores4D(:, :, :, offsetIdx) + ...
                    w_t  * scores4D(:, :, :, tailIdx);
    % NaN propagation: any window NaN → combined NaN. Set Inf in those slots
    % so argmin skips them.
    combinedScore(isnan(combinedScore)) = Inf;

    %% MatchingMode override: pick joint (sim, bias) per the combined score.
    switch lower(opts.MatchingMode)
        case 'percondition'
            % Joint (sim, bias) per cond minimising combinedScore.
            for c = 1:nCond
                cond = opts.Conditions{c};
                if ~isfield(bestBiasPerCondition, cond), continue; end
                slice2D = squeeze(combinedScore(c, :, :));   % [nSims x nBias]
                if all(isinf(slice2D), 'all'), continue; end
                [~, idx] = min(slice2D(:));
                [sBest, bBest] = ind2sub(size(slice2D), idx);
                bestSimIdx(c) = sBest;
                fprintf('MatchingMode=perCondition  %-10s -> sim=%d, bias=%-7s (combined=%.4f, on=%.4f, off=%.4f, tail=%.4f)\n', ...
                    cond, sBest-1, biasLabel(info.stimulusBiasValues(bBest)), ...
                    combinedScore(c, sBest, bBest), ...
                    scores4D(c, sBest, bBest, onsetIdx), ...
                    scores4D(c, sBest, bBest, offsetIdx), ...
                    scores4D(c, sBest, bBest, tailIdx));
                bestBiasPerCondition.(cond).bestSimIdx = sBest;
                for w = 1:nWin
                    wn = winNames{w};
                    if ~any(strcmp(wn, biasRatedWindows)), continue; end
                    sliceWin = squeeze(scores4D(c, sBest, :, w));
                    if all(isnan(sliceWin)), continue; end
                    sliceWin(isnan(sliceWin)) = Inf;
                    [bestRmseW, bWin] = min(sliceWin);
                    bestBiasPerCondition.(cond).(wn).bias_value   = info.stimulusBiasValues(bWin);
                    bestBiasPerCondition.(cond).(wn).rmse         = bestRmseW;
                    bestBiasPerCondition.(cond).(wn).simIdx       = sBest;
                    bestBiasPerCondition.(cond).(wn).matchingMode = 'perCondition';
                end
            end
        case 'global'
            condIsNaive = cellfun(@(c) strcmp(c, 'Naive'), opts.Conditions);
            condIdxs = find(~condIsNaive);
            % minOverSim(c, b) = min_s combinedScore(c, s, b)
            minOverSim_combined = squeeze(min(combinedScore, [], 2, 'omitnan'));
            sumRmse = squeeze(sum(minOverSim_combined(condIdxs, :), 1, 'omitnan'));
            hasAny  = squeeze(sum(~isinf(minOverSim_combined(condIdxs, :)), 1));
            sumRmse(hasAny == 0) = Inf;
            [~, bGlobal] = min(sumRmse);
            globalBiasVal = info.stimulusBiasValues(bGlobal);
            fprintf('MatchingMode=global  bias=%-7s (sum-combined=%.4f over %d non-Naive conds)\n', ...
                biasLabel(globalBiasVal), sumRmse(bGlobal), numel(condIdxs));
            for c = 1:nCond
                cond = opts.Conditions{c};
                if ~isfield(bestBiasPerCondition, cond), continue; end
                sliceSimAtB = squeeze(combinedScore(c, :, bGlobal));
                if all(isinf(sliceSimAtB))
                    fprintf('  %-10s no sim usable at bias=%s\n', cond, biasLabel(globalBiasVal));
                    continue;
                end
                [~, sBest] = min(sliceSimAtB);
                bestSimIdx(c) = sBest;
                bestBiasPerCondition.(cond).bestSimIdx = sBest;
                fprintf('  %-10s -> sim=%d (combined=%.4f, on=%.4f, off=%.4f, tail=%.4f at bias=%s)\n', ...
                    cond, sBest-1, combinedScore(c, sBest, bGlobal), ...
                    scores4D(c, sBest, bGlobal, onsetIdx), ...
                    scores4D(c, sBest, bGlobal, offsetIdx), ...
                    scores4D(c, sBest, bGlobal, tailIdx), ...
                    biasLabel(globalBiasVal));
                for w = 1:nWin
                    wn = winNames{w};
                    if ~any(strcmp(wn, biasRatedWindows)), continue; end
                    bestBiasPerCondition.(cond).(wn).bias_value   = globalBiasVal;
                    bestBiasPerCondition.(cond).(wn).rmse         = scores4D(c, sBest, bGlobal, w);
                    bestBiasPerCondition.(cond).(wn).simIdx       = sBest;
                    bestBiasPerCondition.(cond).(wn).matchingMode = 'global';
                end
            end
        case 'expert'
            expertIdx = find(strcmp(opts.Conditions, 'Expert'), 1);
            if isempty(expertIdx)
                warning('MatchingMode=expert but Expert missing; falling back to perCondition');
            else
                slice2D = squeeze(combinedScore(expertIdx, :, :));
                if all(isinf(slice2D), 'all')
                    warning('MatchingMode=expert: no Expert scores; falling back');
                else
                    [~, idx] = min(slice2D(:));
                    [sExp, bExp] = ind2sub(size(slice2D), idx);
                    expertBias = info.stimulusBiasValues(bExp);
                    fprintf('MatchingMode=expert  Expert sim=%d, bias=%-7s (combined=%.4f)\n', ...
                        sExp-1, biasLabel(expertBias), combinedScore(expertIdx, sExp, bExp));
                    for c = 1:nCond
                        cond = opts.Conditions{c};
                        if ~isfield(bestBiasPerCondition, cond), continue; end
                        if c == expertIdx
                            sBest = sExp;
                        else
                            sliceSimAtB = squeeze(combinedScore(c, :, bExp));
                            if all(isinf(sliceSimAtB)), continue; end
                            [~, sBest] = min(sliceSimAtB);
                        end
                        bestSimIdx(c) = sBest;
                        bestBiasPerCondition.(cond).bestSimIdx = sBest;
                        fprintf('  %-10s -> sim=%d (combined=%.4f, on=%.4f, off=%.4f, tail=%.4f at Expert''s bias=%s)\n', ...
                            cond, sBest-1, combinedScore(c, sBest, bExp), ...
                            scores4D(c, sBest, bExp, onsetIdx), ...
                            scores4D(c, sBest, bExp, offsetIdx), ...
                            scores4D(c, sBest, bExp, tailIdx), ...
                            biasLabel(expertBias));
                        for w = 1:nWin
                            wn = winNames{w};
                            if ~any(strcmp(wn, biasRatedWindows)), continue; end
                            bestBiasPerCondition.(cond).(wn).bias_value   = expertBias;
                            bestBiasPerCondition.(cond).(wn).rmse         = scores4D(c, sBest, bExp, w);
                            bestBiasPerCondition.(cond).(wn).simIdx       = sBest;
                            bestBiasPerCondition.(cond).(wn).matchingMode = 'expert';
                        end
                    end
                end
            end
        otherwise
            warning('Unknown MatchingMode "%s"; keeping defaults', opts.MatchingMode);
    end

    % --- Add fold-change windows: match by fold change between baseline ---
    % and the bigger peak (max of onset/offset peaks). Two flavours, both
    % shared-bias across conditions (per-cond bias is biologically wrong:
    % all conditions get the same physical stimulus):
    %   fold_global  - bias minimising sum |dataFold - modelFold| over
    %                  non-Naive conds; per-cond sim picked at that bias.
    %   fold_nospout - bias anchored on NoSpout's joint argmin |fold diff|;
    %                  per-cond sim picked at that bias. NoSpout is the
    %                  un-rewarded control, so its fold is the "pure
    %                  sensory" response anchor.
    % Both are mode-independent (same picks across all MatchingModes).
    fprintf('\nFold-change windows: matching on max(onset, offset) / baseline\n');
    dataFold = nan(nCond, 1);
    modelFold = nan(nCond, nSims, nBiasValues);
    for c = 1:nCond
        cond = opts.Conditions{c};
        if ~isfield(expTraces, cond), continue; end
        ttFull = expTimeAxis(:);
        es = expTraces.(cond)(:);
        n = min(numel(ttFull), numel(es));
        ttFull = ttFull(1:n); es = es(1:n);
        bMaskExp = (ttFull >= opts.FoldBaselineWindow(1) & ttFull <= opts.FoldBaselineWindow(2)) & ~isnan(es);
        onMaskExp  = (ttFull >= opts.OnsetWindow(1) & ttFull <= opts.OnsetWindow(2)) & ~isnan(es);
        offMaskExp = (ttFull >= stimEndSec + opts.OffsetWindow(1) & ttFull <= stimEndSec + opts.OffsetWindow(2)) & ~isnan(es);
        if nnz(bMaskExp) < 2 || nnz(onMaskExp) < 1 || nnz(offMaskExp) < 1
            fprintf('  fold   %-10s window mask too small; skipping\n', cond);
            continue;
        end
        baselineExp = mean(es(bMaskExp));
        if baselineExp <= 0
            fprintf('  fold   %-10s baseline <=0 (%.4f); skipping\n', cond, baselineExp);
            continue;
        end
        peakExp = max(max(es(onMaskExp)), max(es(offMaskExp)));
        dataFold(c) = peakExp / baselineExp;
    end
    for c = 1:nCond
        if isnan(dataFold(c)), continue; end
        simTimeAxisC = simTimeAxis * timeStretch(c) - timeShift(c);   % cond-specific stretched+shifted axis
        for s = 1:nSims
            for b = 1:nBiasValues
                simBiasMean = squeeze(simTracesCondSim(c, s, b, :));
                if all(isnan(simBiasMean)), continue; end
                ttSim = simTimeAxisC(:);
                ss = simBiasMean(:);
                bMaskSim = (ttSim >= opts.FoldBaselineWindow(1) & ttSim <= opts.FoldBaselineWindow(2)) & ~isnan(ss);
                onMaskSim  = (ttSim >= opts.OnsetWindow(1) & ttSim <= opts.OnsetWindow(2)) & ~isnan(ss);
                offMaskSim = (ttSim >= stimEndSec + opts.OffsetWindow(1) & ttSim <= stimEndSec + opts.OffsetWindow(2)) & ~isnan(ss);
                if nnz(bMaskSim) < 2 || nnz(onMaskSim) < 1 || nnz(offMaskSim) < 1, continue; end
                baselineSim = mean(ss(bMaskSim));
                if baselineSim <= 0, continue; end
                peakSim = max(max(ss(onMaskSim)), max(ss(offMaskSim)));
                modelFold(c, s, b) = peakSim / baselineSim;
            end
        end
    end
    foldDiff = abs(reshape(dataFold, [], 1, 1) - modelFold);   % [nCond x nSims x nBiasValues]
    foldDiff(isnan(foldDiff)) = Inf;

    % fold_global: shared bias from min sum |fold diff| over non-Naive conds
    condIsNaive = cellfun(@(c) strcmp(c, 'Naive'), opts.Conditions);
    nonNaiveIdxs = find(~condIsNaive);
    minOverSim_fold = squeeze(min(foldDiff, [], 2));   % [nCond x nBias]
    if isempty(nonNaiveIdxs)
        fprintf('fold_global: no non-Naive conds; skipping\n');
    else
        sumFold = squeeze(sum(minOverSim_fold(nonNaiveIdxs, :), 1));
        hasAny  = squeeze(sum(~isinf(minOverSim_fold(nonNaiveIdxs, :)), 1));
        sumFold(hasAny == 0) = Inf;
        if all(isinf(sumFold))
            fprintf('fold_global: no usable bias; skipping\n');
        else
            [~, bGlobal] = min(sumFold);
            biasGlobal = info.stimulusBiasValues(bGlobal);
            fprintf('fold_global: bias=%-7s (sum-foldDiff=%.4f over %d non-Naive conds)\n', ...
                biasLabel(biasGlobal), sumFold(bGlobal), numel(nonNaiveIdxs));
            for c = 1:nCond
                cond = opts.Conditions{c};
                if ~isfield(bestBiasPerCondition, cond), continue; end
                if isnan(dataFold(c)), continue; end
                sliceSim = squeeze(foldDiff(c, :, bGlobal));
                if all(isinf(sliceSim))
                    fprintf('  fold_global  %-10s no usable sim at bias=%s\n', cond, biasLabel(biasGlobal));
                    continue;
                end
                [~, sBest] = min(sliceSim);
                bestBiasPerCondition.(cond).fold_global.bias_value   = biasGlobal;
                bestBiasPerCondition.(cond).fold_global.simIdx       = sBest;
                bestBiasPerCondition.(cond).fold_global.rmse         = sliceSim(sBest);
                bestBiasPerCondition.(cond).fold_global.dataFold     = dataFold(c);
                bestBiasPerCondition.(cond).fold_global.modelFold    = modelFold(c, sBest, bGlobal);
                bestBiasPerCondition.(cond).fold_global.matchingMode = opts.MatchingMode;
                fprintf('  fold_global  %-10s -> sim=%d bias=%-7s (data=%.2f, model=%.2f, diff=%.4f)\n', ...
                    cond, sBest-1, biasLabel(biasGlobal), ...
                    dataFold(c), modelFold(c, sBest, bGlobal), sliceSim(sBest));
            end
        end
    end

    % fold_nospout: shared bias = NoSpout's joint argmin |fold diff|
    nsIdx = find(strcmp(opts.Conditions, 'NoSpout'), 1);
    if isempty(nsIdx)
        fprintf('fold_nospout: NoSpout not in Conditions; skipping\n');
    elseif isnan(dataFold(nsIdx))
        fprintf('fold_nospout: dataFold(NoSpout) is NaN; skipping\n');
    else
        slice2D_NS = squeeze(foldDiff(nsIdx, :, :));
        if all(isinf(slice2D_NS), 'all')
            fprintf('fold_nospout: NoSpout has no usable (sim, bias); skipping\n');
        else
            [~, idx] = min(slice2D_NS(:));
            [sNS, bNS] = ind2sub(size(slice2D_NS), idx);
            biasNS = info.stimulusBiasValues(bNS);
            fprintf('fold_nospout: NoSpout sim=%d bias=%-7s (foldDiff=%.4f)\n', ...
                sNS-1, biasLabel(biasNS), slice2D_NS(sNS, bNS));
            for c = 1:nCond
                cond = opts.Conditions{c};
                if ~isfield(bestBiasPerCondition, cond), continue; end
                if isnan(dataFold(c)), continue; end
                if c == nsIdx
                    sBest = sNS;
                else
                    sliceSim = squeeze(foldDiff(c, :, bNS));
                    if all(isinf(sliceSim))
                        fprintf('  fold_nospout %-10s no usable sim at NoSpout bias=%s\n', cond, biasLabel(biasNS));
                        continue;
                    end
                    [~, sBest] = min(sliceSim);
                end
                bestBiasPerCondition.(cond).fold_nospout.bias_value   = biasNS;
                bestBiasPerCondition.(cond).fold_nospout.simIdx       = sBest;
                bestBiasPerCondition.(cond).fold_nospout.rmse         = foldDiff(c, sBest, bNS);
                bestBiasPerCondition.(cond).fold_nospout.dataFold     = dataFold(c);
                bestBiasPerCondition.(cond).fold_nospout.modelFold    = modelFold(c, sBest, bNS);
                bestBiasPerCondition.(cond).fold_nospout.matchingMode = opts.MatchingMode;
                fprintf('  fold_nospout %-10s -> sim=%d bias=%-7s (data=%.2f, model=%.2f, diff=%.4f)\n', ...
                    cond, sBest-1, biasLabel(biasNS), ...
                    dataFold(c), modelFold(c, sBest, bNS), foldDiff(c, sBest, bNS));
            end
        end
    end

    % fold_naive: shared bias = Naive's joint argmin |fold diff|. Naive is
    % the pre-training condition, so its fold is the cleanest "untrained
    % sensory baseline" anchor. Symmetric to fold_nospout but using the
    % learning-naive control instead of the un-rewarded control.
    naiveIdx = find(strcmp(opts.Conditions, 'Naive'), 1);
    if isempty(naiveIdx)
        fprintf('fold_naive: Naive not in Conditions; skipping\n');
    elseif isnan(dataFold(naiveIdx))
        fprintf('fold_naive: dataFold(Naive) is NaN; skipping\n');
    else
        slice2D_NV = squeeze(foldDiff(naiveIdx, :, :));
        if all(isinf(slice2D_NV), 'all')
            fprintf('fold_naive: Naive has no usable (sim, bias); skipping\n');
        else
            [~, idx] = min(slice2D_NV(:));
            [sNV, bNV] = ind2sub(size(slice2D_NV), idx);
            biasNV = info.stimulusBiasValues(bNV);
            fprintf('fold_naive: Naive sim=%d bias=%-7s (foldDiff=%.4f)\n', ...
                sNV-1, biasLabel(biasNV), slice2D_NV(sNV, bNV));
            for c = 1:nCond
                cond = opts.Conditions{c};
                if ~isfield(bestBiasPerCondition, cond), continue; end
                if isnan(dataFold(c)), continue; end
                if c == naiveIdx
                    sBest = sNV;
                else
                    sliceSim = squeeze(foldDiff(c, :, bNV));
                    if all(isinf(sliceSim))
                        fprintf('  fold_naive   %-10s no usable sim at Naive bias=%s\n', cond, biasLabel(biasNV));
                        continue;
                    end
                    [~, sBest] = min(sliceSim);
                end
                bestBiasPerCondition.(cond).fold_naive.bias_value   = biasNV;
                bestBiasPerCondition.(cond).fold_naive.simIdx       = sBest;
                bestBiasPerCondition.(cond).fold_naive.rmse         = foldDiff(c, sBest, bNV);
                bestBiasPerCondition.(cond).fold_naive.dataFold     = dataFold(c);
                bestBiasPerCondition.(cond).fold_naive.modelFold    = modelFold(c, sBest, bNV);
                bestBiasPerCondition.(cond).fold_naive.matchingMode = opts.MatchingMode;
                fprintf('  fold_naive   %-10s -> sim=%d bias=%-7s (data=%.2f, model=%.2f, diff=%.4f)\n', ...
                    cond, sBest-1, biasLabel(biasNV), ...
                    dataFold(c), modelFold(c, sBest, bNV), foldDiff(c, sBest, bNV));
            end
        end
    end

    % --- Naive-anchored RMSE windows: for each rated window, anchor bias
    %     on Naive's per-window argmin AT NAIVE's BASELINE-BEST SIM (NOT
    %     joint argmin over (sim, bias) — that mode would land at suspicious
    %     low biases because some non-canonical Naive sim happens to fit
    %     well at low bias). Propagate the anchor bias to all conds; each
    %     non-Naive cond picks its own sim by argmin RMSE for the same
    %     window at that bias. Mode-independent.
    naiveIdxR = find(strcmp(opts.Conditions, 'Naive'), 1);
    % Compute Naive's BASELINE-BEST sim from perSimBaselineRmse directly —
    % bestSimIdx may have been overwritten by the MatchingMode switch above
    % to the joint-opt sim, which is NOT what we want here (we want the
    % canonical "Best bias for Naive [sim=7]"-style anchor).
    sNV_anchor = NaN;
    if ~isempty(naiveIdxR) && size(perSimBaselineRmse, 1) >= naiveIdxR
        baselineRow = perSimBaselineRmse(naiveIdxR, :);
        baselineRow(isnan(baselineRow)) = Inf;
        if ~all(isinf(baselineRow))
            [~, sNV_anchor] = min(baselineRow);
        end
    end
    if isempty(naiveIdxR) || isnan(sNV_anchor)
        fprintf('Naive-anchored RMSE windows: Naive missing or has no baseline-best sim; skipping.\n');
    else
        fprintf('\nNaive-anchored RMSE windows: anchor on Naive''s baseline-best sim=%d.\n', sNV_anchor - 1);
        for wi = 1:nWin
            wn = winNames{wi};
            if ~any(strcmp(wn, biasRatedWindows)), continue; end
            sliceBias_NVR = squeeze(scores4D(naiveIdxR, sNV_anchor, :, wi));
            sliceBias_NVR(isnan(sliceBias_NVR)) = Inf;
            if all(isinf(sliceBias_NVR))
                fprintf('  %s_naive: Naive''s baseline sim has no usable bias; skipping.\n', wn);
                continue;
            end
            [~, bNVR] = min(sliceBias_NVR);
            biasNVR = info.stimulusBiasValues(bNVR);
            wnNaive = sprintf('%s_naive', wn);
            fprintf('  %s_naive: anchor bias=%-7s (rmse=%.4f at Naive sim=%d)\n', ...
                wn, biasLabel(biasNVR), sliceBias_NVR(bNVR), sNV_anchor - 1);
            for c = 1:nCond
                cond = opts.Conditions{c};
                if ~isfield(bestBiasPerCondition, cond), continue; end
                if c == naiveIdxR
                    sBestR = sNV_anchor;
                else
                    sliceSimR = squeeze(scores4D(c, :, bNVR, wi));
                    if all(isnan(sliceSimR)), continue; end
                    sliceSimR(isnan(sliceSimR)) = Inf;
                    if all(isinf(sliceSimR))
                        fprintf('    %-10s no usable sim at Naive''s bias=%s\n', cond, biasLabel(biasNVR));
                        continue;
                    end
                    [~, sBestR] = min(sliceSimR);
                end
                bestBiasPerCondition.(cond).(wnNaive).bias_value   = biasNVR;
                bestBiasPerCondition.(cond).(wnNaive).simIdx       = sBestR;
                bestBiasPerCondition.(cond).(wnNaive).rmse         = scores4D(c, sBestR, bNVR, wi);
                bestBiasPerCondition.(cond).(wnNaive).matchingMode = opts.MatchingMode;
            end
        end
    end

    % --- Add 'bias6' window: bias forced to 6.0 for every condition. -----
    % Per-cond sim is the one minimising the combined score AT bias=6 (so
    % each cond gets the sim that's best-matched at the physiological bias,
    % independent of which (sim, bias) the joint-opt chose for full/onset/
    % offset). The bias6 entry is written for ALL MatchingModes so downstream
    % figs can render it as a 4th window inside every mode folder.
    if ~isempty(info.stimulusBiasValues)
        biasVec = info.stimulusBiasValues(:)';
        finiteBias = biasVec(isfinite(biasVec));
        if isempty(finiteBias)
            fprintf('bias6 window: no finite bias values available; skipping.\n');
        else
            [~, bias6_idx_local] = min(abs(finiteBias - 6.0));
            bias6_idx = find(biasVec == finiteBias(bias6_idx_local), 1);
            bias6_val = info.stimulusBiasValues(bias6_idx);
            fprintf('bias6 window: bias_idx=%d (bias=%.2f, mode=%s)\n', ...
                bias6_idx, bias6_val, opts.MatchingMode);
            for c = 1:nCond
                cond = opts.Conditions{c};
                if ~isfield(bestBiasPerCondition, cond), continue; end
                sliceSimAtB6 = squeeze(combinedScore(c, :, bias6_idx));
                if all(isinf(sliceSimAtB6))
                    fprintf('  bias6  %-10s no sim usable at bias=%.2f\n', cond, bias6_val);
                    continue;
                end
                [~, sBest6] = min(sliceSimAtB6);
                bestBiasPerCondition.(cond).bias6.bias_value   = bias6_val;
                bestBiasPerCondition.(cond).bias6.rmse         = scores4D(c, sBest6, bias6_idx, fullIdx);
                bestBiasPerCondition.(cond).bias6.simIdx       = sBest6;
                bestBiasPerCondition.(cond).bias6.matchingMode = opts.MatchingMode;
                fprintf('  bias6  %-10s -> sim=%d (combined=%.4f, on=%.4f, off=%.4f, tail=%.4f at bias=%.2f)\n', ...
                    cond, sBest6-1, combinedScore(c, sBest6, bias6_idx), ...
                    scores4D(c, sBest6, bias6_idx, onsetIdx), ...
                    scores4D(c, sBest6, bias6_idx, offsetIdx), ...
                    scores4D(c, sBest6, bias6_idx, tailIdx), ...
                    bias6_val);
            end
        end
    end

    % Re-apply OverrideBestSimIdx after MatchingMode/post-process blocks may
    % have overwritten bestSimIdx (e.g. global mode at line 644). The user
    % override takes precedence for downstream overlay rendering. We also
    % rebuild `scores(c, :, :)` from scores4D at the overridden sim so the
    % overlay's window-winner highlighting reflects the forced sim's
    % per-bias RMSE table.
    for c = 1:nCond
        cond = opts.Conditions{c};
        if isfield(opts.OverrideBestSimIdx, cond)
            forced = opts.OverrideBestSimIdx.(cond);
            if isnumeric(forced) && isscalar(forced) && forced >= 1 && forced <= nSims
                if bestSimIdx(c) ~= forced
                    fprintf('  %-10s OVERRIDE re-applied post-MatchingMode: sim_%d (was sim_%d)\n', ...
                        cond, forced - 1, bestSimIdx(c) - 1);
                    bestSimIdx(c) = forced;
                end
                scores(c, :, :) = squeeze(scores4D(c, forced, :, :));
            end
        end
    end

    % Final OverrideBestSimIdx propagation: force the override into every
    % per-window simIdx in bestBiasPerCondition.<cond>. Without this,
    % per-window blocks (joint-opt, fold_*, bias6, _naive) keep their own
    % argmin sims and Fig6c/g/h read those — making the override invisible
    % to downstream consumers.
    overrideFields = fieldnames(opts.OverrideBestSimIdx);
    for ofi = 1:numel(overrideFields)
        condOv = overrideFields{ofi};
        forcedOv = opts.OverrideBestSimIdx.(condOv);
        if ~isnumeric(forcedOv) || ~isscalar(forcedOv) || forcedOv < 1 || forcedOv > nSims
            continue;
        end
        if ~isfield(bestBiasPerCondition, condOv), continue; end
        wnFieldsOv = fieldnames(bestBiasPerCondition.(condOv));
        nForced = 0;
        for wfi = 1:numel(wnFieldsOv)
            wnOv = wnFieldsOv{wfi};
            vOv = bestBiasPerCondition.(condOv).(wnOv);
            if isstruct(vOv) && isfield(vOv, 'simIdx')
                bestBiasPerCondition.(condOv).(wnOv).simIdx = forcedOv;
                nForced = nForced + 1;
            end
        end
        fprintf('  OverrideBestSimIdx final-propagate: %s simIdx -> %d across %d windows\n', ...
            condOv, forcedOv, nForced);
    end

    % Rebuild per-cond simTraces after joint-optimization may have moved
    % bestSimIdx (downstream plot overlays use simTraces).
    simTraces = nan(nCond, nBiasValues, nFrames);
    for c = 1:nCond
        if isnan(bestSimIdx(c)), continue; end
        simTraces(c, :, :) = squeeze(simTracesCondSim(c, bestSimIdx(c), :, :));
    end

    %% Plot overlays per condition
    cmap = parula(nBiasValues);
    winShade = struct( ...
        'baseline', [0.40 0.40 0.40], ...
        'onset',    [0.20 0.55 0.85], ...
        'offset',   [0.85 0.45 0.20]);
    for c = 1:nCond
        cond = opts.Conditions{c};
        if ~isfield(expTraces, cond) || all(isnan(scores(c, :, :)), 'all'), continue; end
        figName = sprintf('BiasMatch_%s_size%d', cond, opts.StimulusSize);
        fig = figure('Name', figName, 'Color', 'w'); %#ok<NASGU>
        % Two-tile layout: trace plot ~70% width, scores heatmap ~30%.
        % Using 10-column tiledlayout with span [1 7] / [1 3].
        tl = tiledlayout(1, 10, 'TileSpacing', 'compact', 'Padding', 'compact');

        % --- Compute winners and modelFold (used by both tiles) ---
        % winNames is {baseline, full, onset, offset, tail, stim, BothPeaks,
        % StimPeriod} so the rated windows we display are at indices
        % [1 2 3 4 6 7 8] (skipping tail=5 which is internal to combined-score).
        scoreCols = [1 2 3 4 6 7 8];
        scoreLabels = {'b', 'f', 'on', 'off', 'stim', 'bp', 'sp'};
        winnerIdx = nan(1, numel(scoreCols));
        for kk = 1:numel(scoreCols)
            [~, winnerIdx(kk)] = min(squeeze(scores(c, :, scoreCols(kk))));
        end
        haveModelFold = exist('modelFold', 'var') && ~isnan(bestSimIdx(c));

        % --- LEFT TILE: trace plot ----------------------------------------
        axTrace = nexttile(tl, [1 7]);
        hold(axTrace, 'on');

        yLimsTmp = [-0.005, 0.15];
        for w = {'baseline', 'onset', 'offset'}
            wn = w{1};
            wlim = windowDefs.(wn);
            % Iterate rows so BaselineWindow's Nx2 intervals each get their own
            % patch (was using linear indexing which made baseline span the
            % whole plot for Nx2 inputs).
            for kw = 1:size(wlim, 1)
                x0 = wlim(kw, 1); x1 = wlim(kw, 2);
                patch(axTrace, [x0 x1 x1 x0], ...
                      [yLimsTmp(1) yLimsTmp(1) yLimsTmp(2) yLimsTmp(2)], ...
                      winShade.(wn), 'FaceAlpha', 0.10, 'EdgeColor', 'none', ...
                      'HandleVisibility', 'off');
            end
        end

        % Experimental label: bias=N/A; show data fold + DC shift if any.
        dataFoldStr = '';
        if exist('dataFold', 'var') && c <= numel(dataFold) && ~isnan(dataFold(c))
            dataFoldStr = sprintf(', dataFold=%.2f', dataFold(c));
        end
        if opts.BaselineAlign && abs(dcShift(c)) > 0
            dataFoldStr = sprintf('%s, DC+%.4f', dataFoldStr, dcShift(c));
        end
        if abs(timeStretch(c) - 1.0) > 1e-3
            dataFoldStr = sprintf('%s, t×%.2f', dataFoldStr, timeStretch(c));
        end
        if abs(timeShift(c)) > 1e-3
            dataFoldStr = sprintf('%s, shift%+.2fs', dataFoldStr, timeShift(c));
        end
        plot(axTrace, expWinTime, expTraces.(cond)(expWinMask), 'k-', 'LineWidth', 2.5, ...
             'DisplayName', sprintf('%s (experiment%s)', cond, dataFoldStr));

        % Per-bias trace + winner marking. Legend label is now a SHORT
        % `bias=X.XX [winner-tags]` string — all per-window scores moved to
        % the right-tile heatmap.
        fullWinnerIdx       = winnerIdx(strcmp(scoreLabels, 'f'));
        onsetWinnerIdx      = winnerIdx(strcmp(scoreLabels, 'on'));
        offsetWinnerIdx     = winnerIdx(strcmp(scoreLabels, 'off'));
        stimWinnerIdx       = winnerIdx(strcmp(scoreLabels, 'stim'));
        bothPeaksWinnerIdx  = winnerIdx(strcmp(scoreLabels, 'bp'));
        stimPeriodWinnerIdx = winnerIdx(strcmp(scoreLabels, 'sp'));
        simTimeAxisC = simTimeAxis * timeStretch(c) - timeShift(c);   % cond-specific stretched+shifted axis
        for b = 1:nBiasValues
            wins = {};
            if b == fullWinnerIdx,       wins{end+1} = 'full';       end %#ok<AGROW>
            if b == onsetWinnerIdx,      wins{end+1} = 'on';         end %#ok<AGROW>
            if b == offsetWinnerIdx,     wins{end+1} = 'off';        end %#ok<AGROW>
            if b == stimWinnerIdx,       wins{end+1} = 'stim';       end %#ok<AGROW>
            if b == bothPeaksWinnerIdx,  wins{end+1} = 'bp';         end %#ok<AGROW>
            if b == stimPeriodWinnerIdx, wins{end+1} = 'sp';         end %#ok<AGROW>
            isAnyWinner = ~isempty(wins);
            lw = 1.0 + 1.0 * numel(wins);
            ls = '--'; if isAnyWinner, ls = '-'; end
            winTag = '';
            if isAnyWinner, winTag = sprintf(' [%s-win]', strjoin(wins, '+')); end
            % Fade non-winner traces to 55% alpha so winners pop visually.
            if isAnyWinner
                plotColor = cmap(b, :);
            else
                plotColor = [cmap(b, :), 0.55];
            end
            simSlice = interp1(simTimeAxisC, squeeze(simTraces(c, b, :))', expWinTime, 'linear', NaN);
            plot(axTrace, expWinTime, simSlice, ls, 'Color', plotColor, 'LineWidth', lw, ...
                 'DisplayName', sprintf('bias=%s%s', biasLabel(info.stimulusBiasValues(b)), winTag));
        end
        xline(axTrace, 0, 'k:', 'stim onset');
        xline(axTrace, stimEndSec, 'k:', 'stim end');
        xlim(axTrace, opts.MatchWindow);
        xlabel(axTrace, 'Time from stim onset (s)');
        ylabel(axTrace, 'Fraction active');
        grid(axTrace, 'on'); box(axTrace, 'on');
        legend(axTrace, 'Location', 'southoutside', 'Interpreter', 'none', ...
            'FontSize', 7, 'NumColumns', 4);

        % --- RIGHT TILE: scores heatmap -----------------------------------
        axScores = nexttile(tl, [1 3]);
        % Build [nBiasValues x 8] matrix: 7 RMSE columns + 1 fold column.
        nCols = numel(scoreCols) + 1;
        M = nan(nBiasValues, nCols);
        for kk = 1:numel(scoreCols)
            M(:, kk) = scores(c, :, scoreCols(kk));
        end
        if haveModelFold
            M(:, end) = squeeze(modelFold(c, bestSimIdx(c), :));
        end
        % Per-column min-max normalisation for colour mapping (each col gets
        % its own dynamic range so RMSE columns and fold are comparable).
        Mnorm = nan(size(M));
        for kk = 1:nCols
            colVals = M(:, kk);
            finite = isfinite(colVals);
            if any(finite)
                lo = min(colVals(finite)); hi = max(colVals(finite));
                if hi > lo
                    Mnorm(:, kk) = (colVals - lo) / (hi - lo);
                else
                    Mnorm(:, kk) = 0.5;
                end
            end
        end
        imagesc(axScores, Mnorm); colormap(axScores, parula);
        clim(axScores, [0 1]);
        set(axScores, 'YDir', 'normal');
        % Bias labels (short) on Y axis — tinted with the corresponding
        % parula color so reader can visually map heatmap row to trace.
        biasYTickLabels = arrayfun(@(b) biasLabel(info.stimulusBiasValues(b)), ...
            1:nBiasValues, 'UniformOutput', false);
        yticks(axScores, 1:nBiasValues);
        yticklabels(axScores, repmat({''}, nBiasValues, 1));   % blank built-in labels; we render as text() below
        xticks(axScores, 1:nCols);
        xticklabels(axScores, [scoreLabels, {'fold'}]);
        title(axScores, 'scores', 'FontSize', 9);
        set(axScores, 'FontSize', 7);
        % Cell text + winner outlines
        hold(axScores, 'on');
        % Render Y-tick bias labels as text(), tinted with the parula color
        % for that bias to give an instant heatmap-row ↔ trace-color lookup.
        for bb = 1:nBiasValues
            text(axScores, 0.35, bb, biasYTickLabels{bb}, ...
                'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
                'FontSize', 7, 'Color', cmap(bb, :), 'FontWeight', 'bold');
        end
        for kk = 1:nCols
            for bb = 1:nBiasValues
                v = M(bb, kk);
                if ~isfinite(v), continue; end
                if kk == nCols
                    txt = sprintf('%.2f', v);   % fold ratio
                else
                    txt = sprintf('%.3f', v);   % RMSE
                end
                % White on dark cells, black on light cells (by Mnorm)
                cellColor = 'k';
                if Mnorm(bb, kk) > 0.5, cellColor = 'w'; end
                text(axScores, kk, bb, txt, ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'FontSize', 6, 'Color', cellColor);
            end
            % Winner outline for RMSE columns (skip fold)
            if kk <= numel(winnerIdx) && ~isnan(winnerIdx(kk))
                wb = winnerIdx(kk);
                rectangle(axScores, 'Position', [kk-0.5, wb-0.5, 1, 1], ...
                    'EdgeColor', 'r', 'LineWidth', 1.5);
            end
        end
        simIdxStr = '';
        if ~isnan(bestSimIdx(c))
            simIdxStr = sprintf(' [sim=%d]', bestSimIdx(c)-1);
        end
        % Append fold picks (fold_global / fold_nospout / fold_naive) to
        % the title so the bias choice for each downstream window is
        % visible alongside the RMSE-window picks.
        foldKeys  = {'fold_global', 'fold_nospout', 'fold_naive'};
        foldNames = {'fG', 'fNS', 'fNV'};   % short labels: title overflows full names
        foldParts = {};
        for k = 1:numel(foldKeys)
            fk = foldKeys{k};
            if isfield(bestBiasPerCondition.(cond), fk) && ...
               isstruct(bestBiasPerCondition.(cond).(fk)) && ...
               isfield(bestBiasPerCondition.(cond).(fk), 'bias_value')
                foldParts{end+1} = sprintf('%s=%s', foldNames{k}, ...
                    biasLabel(bestBiasPerCondition.(cond).(fk).bias_value)); %#ok<AGROW>
            end
        end
        % Two-line title: line 1 has cond/size/sim + RMSE-window picks,
        % line 2 has fold picks. Font size 10 + short fold labels keep
        % the title within the axes width (axes are narrowed by the
        % eastoutside legend, so a wide title would overflow off the
        % left edge of the saved PNG).
        if all(isfield(bestBiasPerCondition.(cond), {'full','onset','offset'}))
            extraPicks = '';
            extraKeys  = {'stim', 'BothPeaks', 'StimPeriod'};
            extraLabels = {'stim', 'bp', 'sp'};
            for k = 1:numel(extraKeys)
                ek = extraKeys{k};
                if isfield(bestBiasPerCondition.(cond), ek) && ...
                   isstruct(bestBiasPerCondition.(cond).(ek)) && ...
                   isfield(bestBiasPerCondition.(cond).(ek), 'bias_value')
                    extraPicks = sprintf('%s %s=%s', extraPicks, extraLabels{k}, ...
                        biasLabel(bestBiasPerCondition.(cond).(ek).bias_value));
                end
            end
            line1 = sprintf('%s (size=%d%s) : f=%s on=%s off=%s%s', ...
                cond, opts.StimulusSize, simIdxStr, ...
                biasLabel(bestBiasPerCondition.(cond).full.bias_value), ...
                biasLabel(bestBiasPerCondition.(cond).onset.bias_value), ...
                biasLabel(bestBiasPerCondition.(cond).offset.bias_value), ...
                extraPicks);
        else
            line1 = sprintf('%s (size=%d%s)', cond, opts.StimulusSize, simIdxStr);
        end
        % Title placed on the tile layout (spans both tiles).
        if isempty(foldParts)
            title(tl, line1, 'FontSize', 10);
        else
            line2 = strjoin(foldParts, '  ');
            title(tl, {line1, line2}, 'FontSize', 10);
        end
        % Wider figure: trace + legend + heatmap need horizontal room.
        set(gcf, 'Units', 'pixels', 'Position', [50 50 1700 900]);
        pngPath = fullfile(opts.OutputDir, [figName '.png']);
        exportgraphics(gcf, pngPath, 'Resolution', 200);
        close(gcf);
        fprintf('  Saved overlay: %s\n', pngPath);

        % --- Curves-only variant: trace plot alone, no heatmap, no legend.
        %     Paper-figure-friendly. Reuses the same plotting logic but in
        %     a single full-width axes.
        figCurves = figure('Name', [figName '_curves'], 'Color', 'w'); %#ok<NASGU>
        set(gcf, 'Units', 'pixels', 'Position', [50 50 1400 700]);
        axC = gca; hold(axC, 'on');
        for w = {'baseline', 'onset', 'offset'}
            wn = w{1};
            wlim = windowDefs.(wn);
            for kw = 1:size(wlim, 1)
                x0 = wlim(kw, 1); x1 = wlim(kw, 2);
                patch(axC, [x0 x1 x1 x0], ...
                      [yLimsTmp(1) yLimsTmp(1) yLimsTmp(2) yLimsTmp(2)], ...
                      winShade.(wn), 'FaceAlpha', 0.10, 'EdgeColor', 'none', ...
                      'HandleVisibility', 'off');
            end
        end
        plot(axC, expWinTime, expTraces.(cond)(expWinMask), 'k-', 'LineWidth', 2.5);
        for b = 1:nBiasValues
            wins = {};
            if b == fullWinnerIdx,       wins{end+1} = 'full';   end %#ok<AGROW>
            if b == onsetWinnerIdx,      wins{end+1} = 'on';     end %#ok<AGROW>
            if b == offsetWinnerIdx,     wins{end+1} = 'off';    end %#ok<AGROW>
            if b == stimWinnerIdx,       wins{end+1} = 'stim';   end %#ok<AGROW>
            if b == bothPeaksWinnerIdx,  wins{end+1} = 'bp';     end %#ok<AGROW>
            if b == stimPeriodWinnerIdx, wins{end+1} = 'sp';     end %#ok<AGROW>
            isAnyWinner = ~isempty(wins);
            lw = 1.0 + 1.0 * numel(wins);
            ls = '--'; if isAnyWinner, ls = '-'; end
            if isAnyWinner
                plotColor = cmap(b, :);
            else
                plotColor = [cmap(b, :), 0.55];
            end
            simSlice = interp1(simTimeAxisC, squeeze(simTraces(c, b, :))', expWinTime, 'linear', NaN);
            plot(axC, expWinTime, simSlice, ls, 'Color', plotColor, 'LineWidth', lw);
        end
        xline(axC, 0, 'k:', 'stim onset');
        xline(axC, stimEndSec, 'k:', 'stim end');
        xlim(axC, opts.MatchWindow);
        xlabel(axC, 'Time from stim onset (s)');
        ylabel(axC, 'Fraction active');
        grid(axC, 'on'); box(axC, 'on');
        title(axC, sprintf('%s (size=%d%s)', cond, opts.StimulusSize, simIdxStr), 'FontSize', 11);
        curvesPath = fullfile(opts.OutputDir, [figName '_curves.png']);
        exportgraphics(gcf, curvesPath, 'Resolution', 200);
        close(gcf);
        fprintf('  Saved curves: %s\n', curvesPath);
    end

    %% Summary plot + CSV — only the rated windows (no baseline)
    figure('Name', 'BiasMatch_Summary', 'Color', 'w');
    plotConds = fieldnames(bestBiasPerCondition);
    nC = numel(plotConds);
    plotWinNames = biasRatedWindows;     % {'full','onset','offset'}
    nPlotWin = numel(plotWinNames);
    bestMatrix = nan(nC, nPlotWin);
    isClampedMatrix = false(nC, nPlotWin);
    for i = 1:nC
        for w = 1:nPlotWin
            wn = plotWinNames{w};
            if isfield(bestBiasPerCondition.(plotConds{i}), wn) && isstruct(bestBiasPerCondition.(plotConds{i}).(wn))
                v = bestBiasPerCondition.(plotConds{i}).(wn).bias_value;
                bestMatrix(i, w) = v;
                isClampedMatrix(i, w) = isinf(v);
            end
        end
    end
    finiteMax = max(bestMatrix(~isinf(bestMatrix)), [], 'omitnan');
    if isempty(finiteMax) || ~isfinite(finiteMax), finiteMax = 1; end
    clampedHeight = finiteMax * 1.25;
    bestMatrixForPlot = bestMatrix;
    bestMatrixForPlot(isClampedMatrix) = clampedHeight;
    bar(bestMatrixForPlot, 'grouped');
    set(gca, 'XTick', 1:nC, 'XTickLabel', plotConds);
    ylabel('Best-matching bias value');
    legend(plotWinNames, 'Location', 'best');
    title(sprintf('%s — best bias per (cond, window), size=%d, stimEnd=%.2fs (∞ = clamped)', ...
        opts.Mode, opts.StimulusSize, stimEndSec));
    yline(clampedHeight, ':r', 'clamped (∞)', 'LabelHorizontalAlignment', 'left');
    grid on;
    nGroups = nPlotWin;
    if nGroups > 0
        groupOffsets = linspace(-0.3, 0.3, nGroups);
        for i = 1:nC
            for w = 1:nPlotWin
                if isClampedMatrix(i, w)
                    text(i + groupOffsets(w), clampedHeight + 0.03 * clampedHeight, '∞', ...
                        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', 'r');
                end
            end
        end
    end
    summaryPng = fullfile(opts.OutputDir, 'BiasMatch_Summary.png');
    exportgraphics(gcf, summaryPng, 'Resolution', 200);
    close(gcf);
    fprintf('Saved summary: %s\n', summaryPng);

    if ~isempty(summaryRows)
        T = cell2table(summaryRows, ...
            'VariableNames', {'Condition', 'Size', 'BestBaselineSim', ...
                              'BiasValue', 'Window', 'RMSE', 'IsBest'});
        csvPath = fullfile(opts.OutputDir, 'BiasMatch_Summary.csv');
        writetable(T, csvPath);
        fprintf('Saved CSV: %s\n', csvPath);
    end

    sizeOut.scores                 = scores;
    sizeOut.scores4D               = scores4D;
    sizeOut.simTraces              = simTraces;
    sizeOut.simTracesCondSim       = simTracesCondSim;
    sizeOut.bestSimIdx             = bestSimIdx;
    sizeOut.bestBaselineRmse       = bestBaselineRmse;
    sizeOut.perSimBaselineRmse     = perSimBaselineRmse;
    sizeOut.bestBiasPerCondition   = bestBiasPerCondition;
    sizeOut.matchingMode           = opts.MatchingMode;
    sizeOut.baselineAlign          = opts.BaselineAlign;
    sizeOut.dcShift                = dcShift;
    sizeOut.timeStretch            = timeStretch;
    sizeOut.timeShift              = timeShift;
    sizeOut.chosenDurFrames        = info.stimulusDurations(durIdxLocal);
    sizeOut.chosenDurSec           = stimEndSec;
    sizeOut.windowDefs             = windowDefs;
    sizeOut.simTimeAxis            = simTimeAxis;
    sizeOut.info                   = info;
end


%% =====================================================================
function [Sim, info] = loadIsingPerturbation(perturbFile, opts)
%LOADISINGPERTURBATION  Load a per-mode PerturbationResults_*.mat file and
%   return a per-condition activity struct plus axis info.
    Sim = struct();
    info = struct();

    % For files >2 GB, force the HDF5 path. load() of a v7.3 .mat decompresses
    % everything into the workspace, which can OOM even on 32 GB nodes; the
    % HDF5 path lazy-reads only the (size, dur) slice we actually need.
    fileBytes = 0;
    try
        d = dir(perturbFile);
        if ~isempty(d), fileBytes = d.bytes; end
    catch
    end
    forceHDF5 = fileBytes > 2e9;

    useHDF5 = forceHDF5;
    if ~forceHDF5
        try
            Results = load(perturbFile);
        catch ME
            fprintf('  load() failed (%s); falling back to direct HDF5 read.\n', ME.identifier);
            useHDF5 = true;
        end
    end

    if useHDF5
        % --- Direct HDF5 read, lazy: only the chosen (size, dur) slice ---
        info.stimulusBiasValues = double(h5read(perturbFile, '/stimulus_bias_values'));
        info.stimulusBiasValues = info.stimulusBiasValues(:)';
        info.stimulusSizes      = double(h5read(perturbFile, '/stimulus_sizes'));
        info.stimulusSizes      = info.stimulusSizes(:)';
        info.stimulusDurations  = double(h5read(perturbFile, '/stimulus_durations'));
        info.stimulusDurations  = info.stimulusDurations(:)';
        info.preStimFrames      = double(h5read(perturbFile, '/pre_stim_frames'));
        info.postStimFrames     = double(h5read(perturbFile, '/post_stim_frames'));
        info.globalMeanSF       = double(h5read(perturbFile, '/global_mean_sf'));
        info.secPerFrame        = info.globalMeanSF / opts.SamplingRate;

        % Pick (size, dur) NOW so we only read what we need. After picking,
        % collapse info.stimulusSizes/Durations to the single chosen entry so
        % the caller's pickSizeIdx/pickDurationIdx return 1 and downstream
        % slicing `actData(:, sizeIdx, b, :, :)` accesses the correct slot.
        [~, sizeIdx0] = min(abs(info.stimulusSizes - opts.StimulusSize));
        chosenSize = info.stimulusSizes(sizeIdx0);
        if ~isempty(opts.TargetStimDurSec)
            durSecAvail = info.stimulusDurations * info.secPerFrame;
            [~, durIdx0] = min(abs(durSecAvail - opts.TargetStimDurSec));
        elseif ~isempty(opts.TargetDurationFrames)
            [~, durIdx0] = min(abs(info.stimulusDurations - opts.TargetDurationFrames));
        else
            durSec = info.stimulusDurations * info.secPerFrame;
            fits = find(durSec <= opts.MatchWindow(2));
            if isempty(fits)
                durIdx0 = 1;
            else
                durIdx0 = fits(end);
            end
        end
        chosenDur = info.stimulusDurations(durIdx0);

        nBiases = numel(info.stimulusBiasValues);

        % Probe across ALL biases on sim_0 to determine nFrames and the MAX
        % rep count. Different biases can be stored with different rep
        % counts (e.g. 50 vs 150); we allocate the activity buffer to fit
        % the largest, then NaN-pad shorter leaves below.
        nReps = NaN;
        nFrames = NaN;
        probedAny = false;
        for c = 1:numel(opts.Conditions)
            condProbe = opts.Conditions{c};
            for bp = 1:nBiases
                probeBk = sprintf('bias_%s', strrep(sprintf('%.2f', info.stimulusBiasValues(bp)), '.', 'p'));
                probePath = sprintf('/experiments/%s/sim_0/%s/size_%d/dur_%d/%s/activity_crop', ...
                    condProbe, opts.Mode, chosenSize, chosenDur, probeBk);
                try
                    dInfo = h5info(perturbFile, probePath);
                    dims = dInfo.Dataspace.Size;   % HDF5 dim order: [reps, frames] (Python writer)
                    % h5info returns dims in Python/HDF5 order [reps, frames];
                    % h5read transposes to MATLAB. Detect by largest=frames.
                    if numel(dims) ~= 2, continue; end
                    if dims(1) >= dims(2)
                        thisFrames = dims(1);
                        thisReps   = dims(2);
                    else
                        thisFrames = dims(2);
                        thisReps   = dims(1);
                    end
                    if isnan(nFrames)
                        nFrames = thisFrames;
                    elseif thisFrames ~= nFrames
                        warning('Leaf %s has %d frames (expected %d); skipping for probe', ...
                            probePath, thisFrames, nFrames);
                        continue;
                    end
                    if isnan(nReps) || thisReps > nReps
                        nReps = thisReps;
                    end
                    probedAny = true;
                catch
                    continue;
                end
            end
            if probedAny, break; end   % one cond's sim_0 is enough
        end
        if isnan(nFrames)
            error(['Could not probe activity dataset at %s for mode %s ' ...
                   'size %d dur %d — file may be missing this combo.'], ...
                  perturbFile, opts.Mode, chosenSize, chosenDur);
        end
        fprintf('Probe: nFrames=%d, max nReps=%d (across %d biases)\n', nFrames, nReps, nBiases);
        info.totalFrames = nFrames;
        info.stimOnsetFrame = info.preStimFrames + 1;

        % For each condition, read activity per (sim, bias) and assemble a 5-D
        % array with size dim = singleton (since we lazily loaded only one).
        for c = 1:numel(opts.Conditions)
            cond = opts.Conditions{c};
            % Determine sim count via h5info
            try
                gInfo = h5info(perturbFile, sprintf('/experiments/%s', cond));
                simNames = {gInfo.Groups.Name};
            catch
                fprintf('  %s: condition not in HDF5 file; skipping\n', cond);
                continue;
            end
            simKeys = {};
            for k = 1:numel(simNames)
                tok = regexp(simNames{k}, 'sim_(\d+)$', 'tokens', 'once');
                if ~isempty(tok)
                    simKeys{end+1} = sprintf('sim_%s', tok{1}); %#ok<AGROW>
                end
            end
            simKeys = unique(simKeys, 'stable');
            nSims = numel(simKeys);
            if nSims == 0
                fprintf('  %s: no sim_* groups; skipping\n', cond);
                continue;
            end

            activity = nan(nSims, 1, nBiases, nReps, nFrames);
            nShortPadded = 0;
            for sIdx = 1:nSims
                for b = 1:nBiases
                    bk = sprintf('bias_%s', strrep(sprintf('%.2f', info.stimulusBiasValues(b)), '.', 'p'));
                    leafPath = sprintf('/experiments/%s/%s/%s/size_%d/dur_%d/%s/activity_crop', ...
                        cond, simKeys{sIdx}, opts.Mode, chosenSize, chosenDur, bk);
                    try
                        act = h5read(perturbFile, leafPath);
                        % h5read returns [nFrames, actReps]; transpose to [actReps, nFrames]
                        if size(act, 1) == nFrames
                            act = act';
                        end
                        if size(act, 2) ~= nFrames
                            continue;
                        end
                        actReps = size(act, 1);
                        if actReps == nReps
                            activity(sIdx, 1, b, :, :) = act;
                        elseif actReps < nReps
                            % NaN-pad to the buffer's nReps so omitnan-mean
                            % downstream uses only the valid reps.
                            activity(sIdx, 1, b, 1:actReps, :) = act;
                            nShortPadded = nShortPadded + 1;
                        else
                            % More reps than probe found: truncate. Defensive.
                            activity(sIdx, 1, b, :, :) = act(1:nReps, :);
                        end
                    catch
                        % Missing leaf — leave NaN
                    end
                end
            end
            Sim.(cond).activity = activity;
            fprintf('  %s: loaded HDF5 activity [%dsims x 1size x %dbias x %dreps x %dframes]; %d leaves NaN-padded\n', ...
                cond, nSims, nBiases, nReps, nFrames, nShortPadded);
        end

        % Collapse the size/duration lists to the chosen ones so downstream
        % pickSizeIdx/pickDurationIdx return 1 and slice correctly.
        info.stimulusSizes     = chosenSize;
        info.stimulusDurations = chosenDur;
        return;
    else
        info.stimulusBiasValues = double(Results.stimulus_bias_values(:)');
        info.stimulusSizes      = double(Results.stimulus_sizes(:)');
        info.stimulusDurations  = double(Results.stimulus_durations(:)');
        info.preStimFrames      = double(Results.pre_stim_frames);
        info.postStimFrames     = double(Results.post_stim_frames);
        info.globalMeanSF       = double(Results.global_mean_sf);
        Mode = opts.Mode;
        modeKey = strrep(Mode, '_', '');

        if ~isfield(Results, 'experiments')
            error('Results.experiments missing in %s', perturbFile);
        end
        expsField = Results.experiments;

        % Determine total frames per duration (must be probed lazily; use
        % the first available condition + sim + size to read activity shape)
        info.totalFrames = NaN;
        for c = 1:numel(opts.Conditions)
            cond = opts.Conditions{c};
            if ~isfield(expsField, cond), continue; end
            simKeys = fieldnames(expsField.(cond));
            if isempty(simKeys), continue; end
            simBlock = expsField.(cond).(simKeys{1});
            if ~isfield(simBlock, Mode), continue; end
            modeBlock = simBlock.(Mode);
            sizeKeys = fieldnames(modeBlock);
            if isempty(sizeKeys), continue; end
            firstSizeBlock = modeBlock.(sizeKeys{1});
            durKeys = fieldnames(firstSizeBlock);
            if isempty(durKeys), continue; end
            biasField = firstSizeBlock.(durKeys{1});
            if isstruct(biasField)
                biasKeys = fieldnames(biasField);
                if isempty(biasKeys), continue; end
                probe = biasField.(biasKeys{1});
            else
                probe = biasField;
            end
            if isfield(probe, 'activity')
                info.totalFrames = size(probe.activity, 2);
                break;
            end
        end
        if isnan(info.totalFrames)
            error('Could not infer totalFrames from %s for mode %s', ...
                perturbFile, Mode);
        end

        info.stimOnsetFrame = info.preStimFrames + 1;
        info.secPerFrame    = info.globalMeanSF / opts.SamplingRate;

        % Stitch a 5D array per condition: [nSims, nSizes, nBiasValues, nReps, nFrames]
        % We restrict to the requested stimulusSize/duration only — but to
        % keep the matrix layout symmetric, we still allocate over all sizes
        % and fill only the relevant slot. Let downstream caller pick sizeIdx.
        nSizes  = numel(info.stimulusSizes);
        nDurs   = numel(info.stimulusDurations);
        nBiases = numel(info.stimulusBiasValues);
        for c = 1:numel(opts.Conditions)
            cond = opts.Conditions{c};
            if ~isfield(expsField, cond), continue; end
            simKeys = fieldnames(expsField.(cond));
            nSims = numel(simKeys);
            % Pre-allocate with the MAX nReps across all biases (different
            % biases may be stored with different rep counts; we NaN-pad
            % short ones below).
            nRepsProbe = NaN;
            for sIdx = 1:nSims
                if ~isfield(expsField.(cond).(simKeys{sIdx}), Mode), continue; end
                modeBlock = expsField.(cond).(simKeys{sIdx}).(Mode);
                sizeKeys = fieldnames(modeBlock);
                if isempty(sizeKeys), continue; end
                durKeys = fieldnames(modeBlock.(sizeKeys{1}));
                if isempty(durKeys), continue; end
                biasField = modeBlock.(sizeKeys{1}).(durKeys{1});
                if isstruct(biasField)
                    biasKeys = fieldnames(biasField);
                    for bk = 1:numel(biasKeys)
                        probeEntry = biasField.(biasKeys{bk});
                        if isfield(probeEntry, 'activity')
                            r = size(probeEntry.activity, 1);
                            if isnan(nRepsProbe) || r > nRepsProbe
                                nRepsProbe = r;
                            end
                        end
                    end
                else
                    probeEntry = biasField;
                    if isfield(probeEntry, 'activity')
                        r = size(probeEntry.activity, 1);
                        if isnan(nRepsProbe) || r > nRepsProbe
                            nRepsProbe = r;
                        end
                    end
                end
                if ~isnan(nRepsProbe)
                    break;
                end
            end
            if isnan(nRepsProbe)
                warning('No replicate count for %s; skipping', cond);
                continue;
            end

            activity = nan(nSims, nSizes, nBiases, nRepsProbe, info.totalFrames);
            nShortPadded = 0;
            for sIdx = 1:nSims
                if ~isfield(expsField.(cond).(simKeys{sIdx}), Mode), continue; end
                modeBlock = expsField.(cond).(simKeys{sIdx}).(Mode);
                for s = 1:nSizes
                    sizeKey = sprintf('size_%d', info.stimulusSizes(s));
                    if ~isfield(modeBlock, sizeKey), continue; end
                    sizeBlock = modeBlock.(sizeKey);
                    for d = 1:nDurs
                        durKey = sprintf('dur_%d', info.stimulusDurations(d));
                        if ~isfield(sizeBlock, durKey), continue; end
                        durBlock = sizeBlock.(durKey);
                        for b = 1:nBiases
                            % Filename biasKey = 'bias_<v>p<vv>'
                            bk = sprintf('bias_%s', strrep(sprintf('%.2f', info.stimulusBiasValues(b)), '.', 'p'));
                            if ~isfield(durBlock, bk), continue; end
                            entry = durBlock.(bk);
                            if ~isfield(entry, 'activity'), continue; end
                            act = entry.activity;
                            actReps = size(act, 1);
                            if actReps == nRepsProbe
                                activity(sIdx, s, b, :, :) = act;
                            elseif actReps < nRepsProbe
                                activity(sIdx, s, b, 1:actReps, :) = act;
                                nShortPadded = nShortPadded + 1;
                            else
                                activity(sIdx, s, b, :, :) = act(1:nRepsProbe, :, :);
                            end
                        end
                    end
                end
            end
            Sim.(cond).activity = activity;
            if nShortPadded > 0
                fprintf('  %s: %d leaves NaN-padded to nReps=%d\n', cond, nShortPadded, nRepsProbe);
            end
        end
    end
end


%% =====================================================================
function idx = pickSizeIdx(stimulusSizes, requested)
    [~, idx] = min(abs(stimulusSizes - requested));
    if abs(stimulusSizes(idx) - requested) > 1e-6
        warning('Requested stimulus_size=%g not in stimulusSizes [%s]; using %g (idx %d)', ...
            requested, mat2str(stimulusSizes), stimulusSizes(idx), idx);
    end
end


%% =====================================================================
function idx = pickDurationIdx(durations, matchWindow, secPerFrame)
%   Pick the largest duration that fits inside MatchWindow seconds. If
%   none fit, pick the smallest available.
    durSec = durations * secPerFrame;
    fits = find(durSec <= matchWindow(2));
    if isempty(fits)
        idx = 1;
    else
        idx = fits(end);
    end
end


%% =====================================================================
function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end


%% =====================================================================
function lbl = biasLabel(value)
%BIASLABEL  Pretty-format a bias value (Inf → 'clamped').
    if isinf(value)
        lbl = 'clamped';
    else
        lbl = sprintf('%.2f', value);
    end
end


%% =====================================================================
function Sim = appendClampedToSim(Sim, info, clampedFile, opts)
%APPENDCLAMPEDTOSIM  Append the fully-clamped activity to each cond's 5-D
%   bias array as the (nBiasValues+1)th slot. The clamped aggregate's leaf
%   path lacks a bias level: /experiments/<cond>/<sim>/<mode>/size_<s>/dur_<d>/activity_crop.
%   We pick the SAME (size, dur) the bias-mode pass already chose so the
%   appended slice aligns with the rest of the simTraces matrix.
    chosenSize = info.stimulusSizes(end);     % HDF5 path collapses to scalar
    chosenDur  = info.stimulusDurations(end); % same
    nFrames = info.totalFrames;

    % Probe one leaf to figure out nReps and orientation
    nReps = NaN;
    for c = 1:numel(opts.Conditions)
        condProbe = opts.Conditions{c};
        probePath = sprintf('/experiments/%s/sim_0/%s/size_%d/dur_%d/activity_crop', ...
            condProbe, opts.ClampedMode, chosenSize, chosenDur);
        try
            probe = h5read(clampedFile, probePath);
            if size(probe, 1) >= size(probe, 2)
                nFramesProbe = size(probe, 1);
                nReps        = size(probe, 2);
            else
                nFramesProbe = size(probe, 2);
                nReps        = size(probe, 1);
            end
            if nFramesProbe ~= nFrames
                warning(['Clamped totalFrames (%d) differs from bias-mode (%d); ' ...
                         'truncating to bias-mode length.'], nFramesProbe, nFrames);
            end
            break;
        catch
            continue;
        end
    end
    if isnan(nReps)
        warning('Could not probe clamped leaf at %s — clamped will be omitted.', clampedFile);
        return;
    end

    for c = 1:numel(opts.Conditions)
        cond = opts.Conditions{c};
        if ~isfield(Sim, cond) || ~isfield(Sim.(cond), 'activity'), continue; end
        existing = Sim.(cond).activity;   % [nSims, 1, nBias, nReps, nFrames]
        [nSims, nSize, nBias, nRepsExisting, nFramesExisting] = size(existing);

        % Clamped slice: [nSims, 1, 1, nReps, nFrames]
        clampedSlice = nan(nSims, nSize, 1, nRepsExisting, nFramesExisting);
        try
            simInfo = h5info(clampedFile, sprintf('/experiments/%s', cond));
            simNames = {simInfo.Groups.Name};
        catch
            fprintf('  %s: clamped condition group missing; skipping.\n', cond);
            continue;
        end
        simKeys = {};
        for k = 1:numel(simNames)
            tok = regexp(simNames{k}, 'sim_(\d+)$', 'tokens', 'once');
            if ~isempty(tok)
                simKeys{end+1} = sprintf('sim_%s', tok{1}); %#ok<AGROW>
            end
        end
        simKeys = unique(simKeys, 'stable');

        for sIdx = 1:min(nSims, numel(simKeys))
            leafPath = sprintf('/experiments/%s/%s/%s/size_%d/dur_%d/activity_crop', ...
                cond, simKeys{sIdx}, opts.ClampedMode, chosenSize, chosenDur);
            try
                act = h5read(clampedFile, leafPath);
                % Normalise to (nReps, nFrames). Heuristic: frames > reps
                % across both clamped (150) and bias (50) sweeps; if dim 1
                % is larger we transpose. Robust to differing rep counts
                % between aggregates (clamped has more reps than bias).
                if size(act, 1) > size(act, 2)
                    act = act';
                end
                % Truncate (or pad-skip) to fit the existing slot.
                rUse = min(size(act, 1), nRepsExisting);
                fUse = min(size(act, 2), nFramesExisting);
                if rUse > 0 && fUse > 0
                    clampedSlice(sIdx, 1, 1, 1:rUse, 1:fUse) = act(1:rUse, 1:fUse);
                end
            catch
                % Missing leaf — leave NaN
            end
        end

        Sim.(cond).activity = cat(3, existing, clampedSlice);
        fprintf('  %s: appended clamped slice [%dsims x 1size x 1bias x %dreps x %dframes]\n', ...
            cond, nSims, nRepsExisting, nFramesExisting);
    end
end
