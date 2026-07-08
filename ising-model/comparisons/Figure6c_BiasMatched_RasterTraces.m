function out = Figure6c_BiasMatched_RasterTraces(varargin)
%FIGURE6C_BIASMATCHED_RASTERTRACES  Generate Figure 6c/d Model panels from
%   matcher's best-(sim, bias) selection per condition.
%
%   For each requested (StimulusSize, Window) combination, produces a
%   1×nCond figure: top row imagesc [nReps × frames] sorted by pre-stim
%   activity, bottom row mean+SEM trace. Mirrors the data-side raster
%   layout from plot_FractionActive_BeforeDuring.m, but each condition's
%   model trace comes from THE specific (sim, bias_value) chosen by
%   match_experimental_to_isingBias.m for that (window, condition).
%
%   This replaces the older clamped-only Figure 6c rendering: instead of a
%   single fixed mode (e.g. double_pulse10), each condition uses its own
%   matched local-bias variant of double_pulse_bias10 (or clamped, when
%   that won the bias rating).
%
%   Inputs (Name-Value):
%     'MatcherOutputDir' (str)  Folder containing per-size matcher outputs.
%        Looks for `<dir>/matcher_output_size<N>.mat` first (per-size files
%        from the parallel-by-size SLURM runs), falls back to a single
%        `matcher_output.mat` whose `out.bySize.size_<N>` contains all sizes.
%        Default: $PerturbationResultsPath/../PerturbationAnalysis/BiasMatchExperiment.
%     'PerturbationResultsPath' (str)  Where the dp_bias10 + dp10 aggregates
%        live. Default: mba_p('IsingModelData_39x78_100K\IsingPerturbations').
%     'ExperimentalDataFile' (str)  Optional. If provided, the experimental
%        trace is overlaid in the bottom row for visual comparison.
%        Default: 'Fig. 5 Ising Models\ExperimentalData.mat'.
%     'OutputDir' (str)  Where panels land. Default:
%        'Fig. 6 Ising Model\c_BiasMatched_RasterTraces'.
%     'Sizes' (vec)  Stim sizes to process. Default [2, 3, 4].
%     'Windows' (cell)  Windows to process. Default {'full','onset','offset','stim','bias6','fold_global','fold_nospout','fold_naive'}.
%     'Mode' (str)  Bias-mode aggregate name. Default 'double_pulse_bias10'.
%     'ClampedMode' (str)  Clamped aggregate name (used when best bias is
%        Inf). Default 'double_pulse10'.
%     'Conditions' (cell)  Default {'Naive','Beginner','Expert','NoSpout'}.
%     'SamplingRate' (num)  Hz. Default 10.
%     'MatchWindow' ([t0 t1])  X-axis range for plots in seconds rel to stim
%        onset. Default [-1, 3].

    p = inputParser;
    addParameter(p, 'MatcherOutputDir', '', @ischar);
    addParameter(p, 'PerturbationResultsPath', mba_p('IsingModelData_39x78_100K\IsingPerturbations'), @ischar);
    addParameter(p, 'ExperimentalDataFile', 'Fig. 5 Ising Models\ExperimentalData.mat', @ischar);
    addParameter(p, 'OutputDir', 'Fig. 6 Ising Model\c_BiasMatched_RasterTraces', @ischar);
    addParameter(p, 'Sizes', [2, 3, 4], @isnumeric);
    addParameter(p, 'Windows', {'full', 'onset', 'offset', 'stim', 'BothPeaks', 'StimPeriod', 'full_naive', 'onset_naive', 'offset_naive', 'stim_naive', 'BothPeaks_naive', 'StimPeriod_naive', 'bias6', 'fold_global', 'fold_nospout', 'fold_naive'}, @iscell);
    addParameter(p, 'Mode', 'double_pulse_bias10', @ischar);
    addParameter(p, 'ClampedMode', 'double_pulse10', @ischar);
    addParameter(p, 'Conditions', {'Naive', 'Beginner', 'Expert', 'NoSpout'}, @iscell);
    addParameter(p, 'SamplingRate', 10, @isnumeric);
    addParameter(p, 'MatchWindow', [-1, 3], @(x) isnumeric(x) && numel(x) == 2);
    % Skip: per-condition list of ORIGINAL Grid40 recording indices to drop,
    % matching the canonical Figure4.m / plot_FractionActive_BeforeDuring set.
    % Applied via RecordingMetadata.<cond>.recording_indices on top of the
    % aggregation skip already baked into ExperimentalData.mat.
    defaultSkip = struct( ...
        'Naive',    [1 9 10 14 16], ...
        'Beginner', [1 6 7 11 12], ...
        'Expert',   [1 4 11:17], ...
        'NoSpout',  [1 4 9 10 11 14]);
    addParameter(p, 'Skip', defaultSkip, @isstruct);
    % Reps-per-dot in the Fig 6d swarm. With 150 reps and BatchSize=10 we
    % get 15 dots per condition, matching the paper figure's visual density.
    % Set to 1 for per-replicate dots.
    addParameter(p, 'SwarmBatchSize', 10, @(x) isnumeric(x) && x >= 1);
    % MatchingMode selects which matcher output to read and which subfolder
    % to write to. Default 'perCondition' (current behavior). Other options:
    % 'global' (single bias for all conds), 'expert' (Expert's bias for all).
    % Setting this changes both the matcher input directory
    % (BiasMatchExperiment_<mode>/) and the per-mode output suffix on
    % MatcherOutputDir/OutputDir if those are left at their defaults.
    addParameter(p, 'MatchingMode', 'perCondition', @(x) ischar(x) && ...
        any(strcmpi(x, {'perCondition', 'global', 'expert'})));
    % DisplayWindow controls the rendered xlim for Fig6c rows 1/2/3
    % (separate from MatchWindow which is the matcher's RMSE scoring window).
    % Empty default → auto-fill to the full experimental trial extent
    % (mirrors plot_FractionActive_BeforeDuring.m which uses no xlim and
    % so shows the full ~[-8, +10] s range).
    % RenderSqueezeFactor: scalar (default 1.0). When != 1, rescales
    % secPerFrame at render time so the model time axis is compressed
    % (or expanded). Set to (data_stim_sec) / (model_stim_sec) to make
    % the model stim period visually align with the experimental stim
    % period. Affects only display + downstream metric windows; does NOT
    % re-load H5 data with a different dur.
    addParameter(p, 'RenderSqueezeFactor', 1.0, @(x) isnumeric(x) && isscalar(x) && x > 0);
    % StimEndDisplaySec: scalar (default []). When non-empty, overrides the
    % computed stim_end position for plotting + window placement. Used to
    % align the offset bias pulse's RISE (sim frame chosenDur - pulse_width)
    % with the data stim end at 2.0 s when squeezing a long-window sim.
    addParameter(p, 'StimEndDisplaySec', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
    addParameter(p, 'DisplayWindow', [], @(x) isempty(x) || ...
        (isnumeric(x) && numel(x) == 2));
    parse(p, varargin{:});
    opts = p.Results;

    % MatchingMode-aware path resolution: if the caller didn't override
    % MatcherOutputDir/OutputDir, append the mode suffix so the three modes
    % don't collide on disk.
    if isempty(opts.MatcherOutputDir)
        opts.MatcherOutputDir = fullfile(opts.PerturbationResultsPath, '..', ...
            'PerturbationAnalysis', sprintf('BiasMatchExperiment_%s', opts.MatchingMode));
    end
    % If OutputDir is the legacy default, append the mode suffix to it as well.
    legacyOutDir = 'Fig. 6 Ising Model\c_BiasMatched_RasterTraces';
    if strcmp(opts.OutputDir, legacyOutDir)
        opts.OutputDir = sprintf('%s_%s', legacyOutDir, opts.MatchingMode);
    end
    if ~exist(opts.OutputDir, 'dir'), mkdir(opts.OutputDir); end
    fprintf('MatchingMode    : %s\n', opts.MatchingMode);
    fprintf('Matcher input   : %s\n', opts.MatcherOutputDir);
    fprintf('Output dir      : %s\n', opts.OutputDir);

    % Load the paper's standard fraction-active colormap (matches Fig 6b
    % data panel rendered by plot_FractionActive_BeforeDuring.m).
    rasterCmap = loadEntropyColourmap();

    %% Locate the perturbation aggregates (once)
    biasFile = locateAggregate(opts.PerturbationResultsPath, opts.Mode);
    clampedFile = locateAggregate(opts.PerturbationResultsPath, opts.ClampedMode);
    fprintf('Bias aggregate:    %s\n', biasFile);
    fprintf('Clamped aggregate: %s\n', clampedFile);

    %% Read aggregate-level metadata once (frames, sec/frame, conditions)
    [stimulusSizes, stimulusDurations, stimulusBiasValues, preStimFrames, ...
     postStimFrames, globalMeanSF] = loadAggregateMetadata(biasFile);
    secPerFrame = globalMeanSF / opts.SamplingRate;
    fprintf('Aggregate dims: sizes=%s, durs=%s, biases=%s, secPerFrame=%.4f\n', ...
        mat2str(stimulusSizes), mat2str(stimulusDurations), ...
        mat2str(stimulusBiasValues), secPerFrame);

    %% Load experimental data: per-condition trial-mean trace AND per-recording
    %  spaghetti traces (mirrors Fig 6b bottom row).
    expTraces = struct();
    if ~isempty(opts.ExperimentalDataFile) && exist(opts.ExperimentalDataFile, 'file')
        fprintf('Loading experimental data: %s\n', opts.ExperimentalDataFile);
        SE = load(opts.ExperimentalDataFile, 'BinarisedData', 'RecordingMetadata', 'TimingInfo');
        if isfield(SE, 'TimingInfo') && ~isempty(fieldnames(SE.TimingInfo))
            expPrestim = double(SE.TimingInfo.trial_structure.prestim_frames(:)');
            expTotalFrames = double(SE.TimingInfo.trial_structure.total_frames);
        else
            expPrestim = 1:80; expTotalFrames = 185;
        end
        expStimOnset = expPrestim(end) + 1;
        expTimeAxis = ((1:expTotalFrames) - expStimOnset) / opts.SamplingRate;
        % Auto-resolve DisplayWindow to the full experimental trial extent
        % when caller didn't specify (mirrors plot_FractionActive_BeforeDuring
        % default of no xlim → full data range).
        if isempty(opts.DisplayWindow)
            opts.DisplayWindow = [expTimeAxis(1), expTimeAxis(end)];
            fprintf('DisplayWindow auto-set to [%.2f, %.2f] s (full trial extent)\n', ...
                opts.DisplayWindow(1), opts.DisplayWindow(2));
        end
        for c = 1:numel(opts.Conditions)
            cond = opts.Conditions{c};
            if ~isfield(SE.BinarisedData, cond) || isempty(SE.BinarisedData.(cond))
                fprintf('  %s: no BinarisedData; skipping\n', cond);
                continue;
            end
            bin = SE.BinarisedData.(cond);
            [gY, gX, T, nTr] = size(bin);
            recMask = any(bin > 0, [3 4]);
            recMask = reshape(recMask, gY, gX);
            hasRM = isfield(SE, 'RecordingMetadata') && ...
                    isstruct(SE.RecordingMetadata) && ...
                    isfield(SE.RecordingMetadata, cond);
            fprintf('  %s: BinarisedData [%s], activePx=%d/%d, RecordingMetadata=%d\n', ...
                cond, mat2str(size(bin)), nnz(recMask), gY*gX, hasRM);
            if nnz(recMask) == 0, continue; end
            binFlat = reshape(bin, gY*gX, T, nTr);
            binFlat = binFlat(recMask(:), :, :);

            % Pooled trial-mean (kept for model-vs-data overlay).
            % reshape forces a [1 × T] row regardless of how mean/squeeze
            % collapses singleton dims.
            poolMean = reshape(mean(mean(binFlat, 1), 3), 1, []);

            % Per-recording traces: split trials by RecordingMetadata
            recTraces = [];
            recLabels = {};
            keepTrialMask = false(nTr, 1);  % which trials survive the Figure4 skip
            if hasRM
                try
                    meta = SE.RecordingMetadata.(cond);
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
                    if isfield(meta, 'animal_names') && ~isempty(meta.animal_names)
                        storedNames = meta.animal_names;
                        if isstring(storedNames), storedNames = cellstr(storedNames); end
                        if iscell(storedNames) && size(storedNames, 1) > 1
                            storedNames = storedNames(:)';
                        end
                    else
                        storedNames = arrayfun(@(i) sprintf('rec%d', i), 1:numel(nTpr), ...
                            'UniformOutput', false);
                    end
                    cursor = 0;
                    nKept = 0; nSkipped = 0;
                    for r = 1:numel(nTpr)
                        n = nTpr(r);
                        if n <= 0, cursor = cursor + max(n, 0); continue; end
                        if cursor + n > nTr
                            warning('%s rec %d: trial range %d:%d exceeds nTr=%d; truncating', ...
                                cond, r, cursor+1, cursor+n, nTr);
                            n = nTr - cursor;
                            if n <= 0, break; end
                        end
                        origIdx = recIdxList(r);
                        if ismember(origIdx, skipSet)
                            cursor = cursor + n;
                            nSkipped = nSkipped + 1;
                            continue;
                        end
                        trialIdx = cursor + (1:n);
                        keepTrialMask(trialIdx) = true;
                        recBin = bin(:, :, :, trialIdx);
                        cursor = cursor + n;
                        % Use the global active-pixel mask (same as pooled),
                        % so per-rec traces are directly comparable.
                        recFlat = reshape(recBin, gY*gX, T, n);
                        recFlat = recFlat(recMask(:), :, :);
                        % Force [1 × T] row regardless of squeeze behavior
                        recTrace = reshape(mean(mean(recFlat, 1), 3), 1, []);
                        recTraces = [recTraces; recTrace]; %#ok<AGROW>
                        if r <= numel(storedNames)
                            recLabels{end+1, 1} = char(storedNames{r}); %#ok<AGROW>
                        else
                            recLabels{end+1, 1} = sprintf('rec%d', r); %#ok<AGROW>
                        end
                        nKept = nKept + 1;
                    end
                    fprintf('    %s: kept %d / dropped %d recs (Skip=[%s])\n', ...
                        cond, nKept, nSkipped, num2str(skipSet));
                catch ME
                    warning('%s: per-rec extraction failed (%s); falling back to pooled mean', ...
                        cond, ME.message);
                    recTraces = [];
                    recLabels = {};
                end
            end

            % Recompute poolMean over only the kept trials (so it's consistent
            % with the Figure4 skip applied above).
            if any(keepTrialMask)
                poolMean = reshape(mean(mean(binFlat(:, :, keepTrialMask), 1), 3), 1, []);
            end

            % Fallback: if per-rec failed/empty, plot the pooled mean as a
            % single "recording" so the row is not empty.
            if isempty(recTraces)
                fprintf('    %s: per-rec empty — using pooled mean as 1 trace\n', cond);
                recTraces = poolMean;
                recLabels = {'pooled'};
            end

            fprintf('    recTraces: [%s], poolMean: [%s]\n', ...
                mat2str(size(recTraces)), mat2str(size(poolMean)));

            expTraces.(cond) = struct( ...
                'time',       expTimeAxis, ...
                'mean',       poolMean, ...
                'recTraces',  recTraces, ...
                'recLabels',  {recLabels});
        end
    end

    % Final fallback for DisplayWindow if it never got auto-set above (e.g.
    % ExperimentalDataFile missing). Use [-8, 10] as a sensible full-trial
    % default at 10 Hz with 80 pre-stim + 105 post-stim frames.
    if isempty(opts.DisplayWindow)
        opts.DisplayWindow = [-8, 10];
        fprintf('DisplayWindow fallback to [%.2f, %.2f] s\n', ...
            opts.DisplayWindow(1), opts.DisplayWindow(2));
    end

    % Per-condition colors (paper convention, matching Figure4.m / Figure6.m)
    condColors = struct( ...
        'Naive',    [0.3373, 0.7059, 0.9137], ...
        'Beginner', [0.8431, 0.2549, 0.6078], ...
        'Expert',   [0,      0.6196, 0.4510], ...
        'NoSpout',  [0.8353, 0.3686, 0]);

    %% Loop sizes × windows
    out = struct();
    out.bySize = struct();
    for sz = opts.Sizes(:)'
        sizeKey = sprintf('size_%d', sz);
        matcherOut = loadMatcherForSize(opts.MatcherOutputDir, sz);
        if isempty(matcherOut)
            warning('No matcher output for size=%d in %s; skipping', sz, opts.MatcherOutputDir);
            continue;
        end
        [~, sizeIdxLocal] = min(abs(stimulusSizes - sz));
        % Apply optional render-time squeeze: rescales secPerFrame (and thus
        % every downstream physical time) by RenderSqueezeFactor. Lets us
        % visualize dur=25 model traces compressed to fit the 2s data stim.
        secPerFrameEff = secPerFrame * opts.RenderSqueezeFactor;
        % Prefer the matcher's chosen dur (single source of truth across the
        % matcher and the figure). Falls back to the legacy "largest-fits"
        % picker for older matcher outputs that don't save chosenDurFrames.
        chosenDur = [];
        if isfield(matcherOut, 'chosenDurFrames') && ~isempty(matcherOut.chosenDurFrames)
            chosenDur = matcherOut.chosenDurFrames;
        elseif isfield(matcherOut, 'sizeOut') && isfield(matcherOut.sizeOut, 'chosenDurFrames')
            chosenDur = matcherOut.sizeOut.chosenDurFrames;
        end
        if isempty(chosenDur)
            durSec = stimulusDurations * secPerFrameEff;
            fits = find(durSec <= opts.MatchWindow(2));
            if isempty(fits), durIdxLocal = 1; else, durIdxLocal = fits(end); end
            chosenDur = stimulusDurations(durIdxLocal);
        end
        stimEndSec = chosenDur * secPerFrameEff;
        if ~isempty(opts.StimEndDisplaySec)
            stimEndSec = opts.StimEndDisplaySec;
        end
        if opts.RenderSqueezeFactor ~= 1
            fprintf('\n=== size=%d, dur=%d frames (%.2fs) [squeeze=%.3f] ===\n', ...
                sz, chosenDur, stimEndSec, opts.RenderSqueezeFactor);
        else
            fprintf('\n=== size=%d, dur=%d frames (%.2fs) ===\n', sz, chosenDur, stimEndSec);
        end

        for wi = 1:numel(opts.Windows)
            wn = opts.Windows{wi};
            outDirSW = fullfile(opts.OutputDir, sizeKey, sprintf('window_%s', wn));
            if ~exist(outDirSW, 'dir'), mkdir(outDirSW); end
            fprintf('  window=%s → %s\n', wn, outDirSW);

            % --- Pull per-condition (sim, bias) selection from matcher ---
            sel = struct();
            for c = 1:numel(opts.Conditions)
                cond = opts.Conditions{c};
                bbpc = matcherOut.bestBiasPerCondition;
                if ~isfield(bbpc, cond) || ~isfield(bbpc.(cond), wn), continue; end
                bestBias = bbpc.(cond).(wn).bias_value;
                % Prefer per-window sim (set by joint-opt for full/onset/
                % offset, and by bias6 post-process at fixed bias=6) so
                % each window's sim is appropriate to its bias choice.
                if isfield(bbpc.(cond).(wn), 'simIdx') && ~isnan(bbpc.(cond).(wn).simIdx)
                    bestSim = bbpc.(cond).(wn).simIdx;
                else
                    bestSim = matcherOut.bestSimIdx(c);   % 1-indexed fallback
                end
                if isnan(bestSim), continue; end
                sel.(cond) = struct('sim', bestSim, 'bias', bestBias, ...
                    'rmse', bbpc.(cond).(wn).rmse);
            end

            % --- Read activity_crop slice per condition (one leaf per cond) ---
            simRaster = struct();
            nFrames = NaN;
            for c = 1:numel(opts.Conditions)
                cond = opts.Conditions{c};
                if ~isfield(sel, cond), continue; end
                sIdx = sel.(cond).sim - 1;   % HDF5 sim_<i> 0-indexed
                bv = sel.(cond).bias;
                if isinf(bv)
                    leafPath = sprintf('/experiments/%s/sim_%d/%s/size_%d/dur_%d/activity_crop', ...
                        cond, sIdx, opts.ClampedMode, sz, chosenDur);
                    file = clampedFile;
                else
                    bk = sprintf('bias_%s', strrep(sprintf('%.2f', bv), '.', 'p'));
                    leafPath = sprintf('/experiments/%s/sim_%d/%s/size_%d/dur_%d/%s/activity_crop', ...
                        cond, sIdx, opts.Mode, sz, chosenDur, bk);
                    file = biasFile;
                end
                try
                    act = h5read(file, leafPath);
                    if size(act, 1) > size(act, 2), act = act'; end   % normalise to [reps, frames]
                    if isnan(nFrames), nFrames = size(act, 2); end
                    simRaster.(cond) = act;
                catch ME
                    warning('Failed to read %s: %s', leafPath, ME.message);
                end
            end
            if isempty(fieldnames(simRaster))
                warning('No conditions loaded for size=%d window=%s; skipping figure', sz, wn);
                continue;
            end

            % Time axis (in seconds rel to stim onset)
            stimOnsetFrame = preStimFrames + 1;
            simTimeAxis = ((1:nFrames) - stimOnsetFrame) * secPerFrameEff;

            % --- 3-row figure rendered in 6 variants:
            %   1. Default      (all reps, full DisplayWindow)
            %   2. 30-rep       (raster subsampled to first 30 reps; DisplayWindow)
            %   3. 30-rep zoom  (subsampled, xlim [-2, 5] for closer inspection)
            %   4-6. _sharedY versions of 1-3: row 1 and row 3 share the same
            %        y-axis (max of the two), and the black "exp mean" dashed
            %        overlay is removed from row 3 (the model panel).
            % Rows 1 (data spaghetti) and 3 (mean ± SEM) always use ALL data
            % regardless of variant — only Row 2 raster subsamples.
            variants(1) = struct('suffix', '',                       'repCap', Inf, 'dispWin', opts.DisplayWindow, 'sharedY', false, 'noExpInRow3', false);
            variants(2) = struct('suffix', '_30reps',                'repCap', 30,  'dispWin', opts.DisplayWindow, 'sharedY', false, 'noExpInRow3', false);
            variants(3) = struct('suffix', '_30reps_zoom',           'repCap', 30,  'dispWin', [-2, 5],            'sharedY', false, 'noExpInRow3', false);
            variants(4) = struct('suffix', '_sharedY',               'repCap', Inf, 'dispWin', opts.DisplayWindow, 'sharedY', true,  'noExpInRow3', true);
            variants(5) = struct('suffix', '_30reps_sharedY',        'repCap', 30,  'dispWin', opts.DisplayWindow, 'sharedY', true,  'noExpInRow3', true);
            variants(6) = struct('suffix', '_30reps_zoom_sharedY',   'repCap', 30,  'dispWin', [-2, 5],            'sharedY', true,  'noExpInRow3', true);
            pngPath = '';   % canonical (default variant) path saved to out struct

            for vi = 1:numel(variants)
                var = variants(vi);
                dispWin = var.dispWin;
                repCap  = var.repCap;
                tWinMask = simTimeAxis >= dispWin(1) & simTimeAxis <= dispWin(2);

                figName = sprintf('Fig6c_BiasMatched%s', var.suffix);
                fig = figure('Name', figName, 'Color', 'w'); %#ok<NASGU>
                nC = numel(opts.Conditions);
                tl = tiledlayout(3, nC, ...
                    'TileSpacing', 'compact', 'Padding', 'compact');

                % Compute color limits from the (possibly subsampled) raster
                % so contrast is appropriate for each variant.
                allVals = [];
                for c = 1:nC
                    cond = opts.Conditions{c};
                    if isfield(simRaster, cond)
                        nReps = size(simRaster.(cond), 1);
                        rowsKeep = 1:min(nReps, repCap);
                        v = simRaster.(cond)(rowsKeep, tWinMask);
                        allVals = [allVals; v(:)]; %#ok<AGROW>
                    end
                end
                if ~isempty(allVals)
                    clims = [prctile(allVals, 5), prctile(allVals, 95)];
                    if clims(1) >= clims(2), clims = [0, max(allVals(:))]; end
                else
                    clims = [0 1];
                end

                % --- Row 1: experimental per-recording spaghetti (all data) ---
                yMaxExp = 0;
                for c = 1:nC
                    cond = opts.Conditions{c};
                    nexttile(c);
                    hold on;
                    if isfield(condColors, cond)
                        col = condColors.(cond);
                    else
                        col = [0.5 0.5 0.5];
                    end
                    if isfield(expTraces, cond) && ~isempty(expTraces.(cond).recTraces)
                        et = expTraces.(cond).time;
                        rtAll = expTraces.(cond).recTraces;
                        nCommon = min(numel(et), size(rtAll, 2));
                        et = et(1:nCommon);
                        rtAll = rtAll(:, 1:nCommon);
                        keep = et >= dispWin(1) & et <= dispWin(2);
                        rt = rtAll(:, keep);
                        for r = 1:size(rt, 1)
                            plot(et(keep), rt(r, :), 'Color', [col, 0.45], 'LineWidth', 0.9, ...
                                'HandleVisibility', 'off');
                        end
                        muExp = mean(rt, 1, 'omitnan');
                        plot(et(keep), muExp, 'Color', col, 'LineWidth', 2.0, ...
                            'DisplayName', sprintf('mean (n=%d rec)', size(rt, 1)));
                        yMaxExp = max(yMaxExp, max(rt, [], 'all'));
                    end
                    xline(0, 'k:', 'LineWidth', 1);
                    xline(stimEndSec, 'k:', 'LineWidth', 1);
                    xlim(dispWin);
                    if c == 1, ylabel('Data: per-rec'); end
                    title(cond, 'Color', col, 'FontWeight', 'bold');
                    grid on; box on;
                    set(gca, 'XTickLabel', []);
                end
                if isempty(yMaxExp) || ~isfinite(yMaxExp), yMaxExp = 0; end
                row1Top = max(yMaxExp * 1.1, 0.05);
                for c = 1:nC
                    ax = nexttile(c);
                    set(ax, 'YLim', [0, row1Top]);
                end

                % --- Row 2: model rasters (subsampled to repCap) -------------
                for c = 1:nC
                    cond = opts.Conditions{c};
                    nexttile(nC + c);
                    if ~isfield(simRaster, cond)
                        title(sprintf('%s (no data)', cond));
                        axis off; continue;
                    end
                    nReps = size(simRaster.(cond), 1);
                    rowsKeep = 1:min(nReps, repCap);
                    act = simRaster.(cond)(rowsKeep, tWinMask);
                    tWinTime = simTimeAxis(tWinMask);
                    imagesc(tWinTime, 1:size(act, 1), act);
                    colormap(gca, rasterCmap);
                    clim(clims);
                    set(gca, 'YDir', 'normal');
                    hold on;
                    xline(0, 'k--', 'LineWidth', 1.2);
                    xline(stimEndSec, 'k--', 'LineWidth', 1.2);
                    xlim(dispWin);
                    if c == 1, ylabel('Model: replicates'); end
                    bv = sel.(cond).bias;
                    if isinf(bv), bvStr = 'clamped'; else, bvStr = sprintf('%.2f', bv); end
                    text(0.02, 0.98, sprintf('bias=%s, sim=%d', bvStr, sel.(cond).sim - 1), ...
                        'Units', 'normalized', 'VerticalAlignment', 'top', ...
                        'FontSize', 8, 'BackgroundColor', [1 1 1 0.7]);
                    set(gca, 'XTickLabel', []);
                end

                % --- Row 3: model mean ± SEM (always all reps) + exp overlay -
                yMaxL = 0;
                for c = 1:nC
                    cond = opts.Conditions{c};
                    nexttile(2*nC + c);
                    hold on;
                    if isfield(condColors, cond)
                        col = condColors.(cond);
                    else
                        col = [0.2 0.2 0.6];
                    end
                    if isfield(simRaster, cond)
                        act = simRaster.(cond)(:, tWinMask);   % all reps
                        tWinTime = simTimeAxis(tWinMask);
                        mu = mean(act, 1, 'omitnan');
                        sem = std(act, 0, 1, 'omitnan') / sqrt(size(act, 1));
                        fill([tWinTime, fliplr(tWinTime)], ...
                             [mu+sem, fliplr(mu-sem)], ...
                             col, 'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
                             'HandleVisibility', 'off');
                        plot(tWinTime, mu, 'Color', col, 'LineWidth', 2, ...
                            'DisplayName', 'model mean');
                        yMaxL = max(yMaxL, max(mu + sem));
                    end
                    if isfield(expTraces, cond) && ~isempty(expTraces.(cond).recTraces) && ~var.noExpInRow3
                        et = expTraces.(cond).time;
                        rt = expTraces.(cond).recTraces;
                        nC2 = min(numel(et), size(rt, 2));
                        et = et(1:nC2); rt = rt(:, 1:nC2);
                        em = mean(rt, 1, 'omitnan');
                        keep = et >= dispWin(1) & et <= dispWin(2);
                        plot(et(keep), em(keep), 'k--', 'LineWidth', 1.5, ...
                            'DisplayName', 'exp mean');
                        if any(keep)
                            yMaxL = max(yMaxL, max(em(keep)));
                        end
                    end
                    xline(0, 'k:', 'LineWidth', 1);
                    xline(stimEndSec, 'k:', 'LineWidth', 1);
                    xlim(dispWin);
                    xlabel('Time from stim onset (s)');
                    if c == 1, ylabel('Model: mean ± SEM'); end
                    grid on; box on;
                end
                if isempty(yMaxL) || ~isfinite(yMaxL), yMaxL = 0; end
                row3Top = max(yMaxL * 1.1, 0.05);
                if var.sharedY
                    % Force rows 1 and 3 to share the same y-axis (max of
                    % both) so amplitudes are visually comparable.
                    sharedTop = max(row1Top, row3Top);
                    row1Top = sharedTop;
                    row3Top = sharedTop;
                    for c = 1:nC
                        ax = nexttile(c);
                        set(ax, 'YLim', [0, row1Top]);
                    end
                end
                for c = 1:nC
                    ax = nexttile(2*nC + c);
                    set(ax, 'YLim', [0, row3Top]);
                end

                title(tl, sprintf('Figure 6c — bias-matched model (size=%d, window=%s%s)', ...
                    sz, wn, var.suffix), 'FontSize', 12, 'FontWeight', 'bold');

                varPath = fullfile(outDirSW, [figName '.png']);
                exportgraphics(gcf, varPath, 'Resolution', 300);
                try, exportgraphics(gcf, fullfile(outDirSW, [figName '.pdf']), 'ContentType', 'vector'); catch, end
                try, print(gcf, fullfile(outDirSW, [figName '.svg']), '-dsvg'); catch, end
                close(gcf);
                fprintf('    saved: %s\n', varPath);
                if vi == 1, pngPath = varPath; end
            end

            % Restore canonical tWinMask for downstream blocks (Fig6d swarms etc.)
            tWinMask = simTimeAxis >= opts.DisplayWindow(1) & simTimeAxis <= opts.DisplayWindow(2);

            % --- All-sims overview: 10 rows × nC cols, mean ± SEM per sim --
            %  Same metric as the row-3 panel above, but expanded to one row
            %  per simulation seed so we can see how the picked bias affects
            %  all top-N sims of each condition (not just the matcher's
            %  baseline-RMSE-best one).
            allSimsPngPath = '';
            if isfield(matcherOut, 'simTracesCondSim') && ~isempty(matcherOut.simTracesCondSim)
                nSimsTotal = size(matcherOut.simTracesCondSim, 2);
            else
                nSimsTotal = 10;
            end
            allSimsFigName = 'Fig6c_AllSims';
            figure('Name', allSimsFigName, 'Color', 'w');
            tlAll = tiledlayout(nSimsTotal, nC, ...
                'TileSpacing', 'compact', 'Padding', 'compact');
            yMaxAll = 0;
            for sIdx = 0:(nSimsTotal-1)
                for c = 1:nC
                    cond = opts.Conditions{c};
                    nexttile;
                    hold on;
                    if isfield(condColors, cond)
                        col = condColors.(cond);
                    else
                        col = [0.2 0.2 0.6];
                    end
                    if isfield(sel, cond)
                        bv = sel.(cond).bias;
                        if isinf(bv)
                            leafPath = sprintf('/experiments/%s/sim_%d/%s/size_%d/dur_%d/activity_crop', ...
                                cond, sIdx, opts.ClampedMode, sz, chosenDur);
                            file = clampedFile;
                            bvStr = 'clamped';
                        else
                            bk = sprintf('bias_%s', strrep(sprintf('%.2f', bv), '.', 'p'));
                            leafPath = sprintf('/experiments/%s/sim_%d/%s/size_%d/dur_%d/%s/activity_crop', ...
                                cond, sIdx, opts.Mode, sz, chosenDur, bk);
                            file = biasFile;
                            bvStr = sprintf('%.2f', bv);
                        end
                        try
                            act = h5read(file, leafPath);
                            if size(act, 1) > size(act, 2), act = act'; end
                            actW = act(:, tWinMask);
                            tWinTime = simTimeAxis(tWinMask);
                            mu = mean(actW, 1, 'omitnan');
                            sem = std(actW, 0, 1, 'omitnan') / sqrt(size(actW, 1));
                            fill([tWinTime, fliplr(tWinTime)], [mu+sem, fliplr(mu-sem)], ...
                                col, 'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
                                'HandleVisibility', 'off');
                            plot(tWinTime, mu, 'Color', col, 'LineWidth', 1.5);
                            yMaxAll = max(yMaxAll, max(mu+sem));
                            % Experimental mean overlay (per-rec averaged, like row 3)
                            if isfield(expTraces, cond) && ~isempty(expTraces.(cond).recTraces)
                                et = expTraces.(cond).time;
                                rt = expTraces.(cond).recTraces;
                                em = mean(rt, 1, 'omitnan');
                                nC2 = min(numel(et), numel(em));
                                et = et(1:nC2); em = em(1:nC2);
                                keep = et >= opts.DisplayWindow(1) & et <= opts.DisplayWindow(2);
                                plot(et(keep), em(keep), 'k--', 'LineWidth', 1.0);
                                if any(keep), yMaxAll = max(yMaxAll, max(em(keep))); end
                            end
                        catch ME
                            text(0.5, 0.5, '(no data)', 'Units', 'normalized', ...
                                'HorizontalAlignment', 'center', 'FontSize', 7);
                        end
                    else
                        bvStr = '?';
                    end
                    xline(0, 'k:', 'LineWidth', 0.5);
                    xline(stimEndSec, 'k:', 'LineWidth', 0.5);
                    xlim(opts.DisplayWindow);
                    if c == 1
                        ylabel(sprintf('sim %d', sIdx), 'FontSize', 8);
                    end
                    if sIdx == 0
                        title(sprintf('%s (b=%s)', cond, bvStr), ...
                            'Color', col, 'FontSize', 9);
                    end
                    if sIdx == nSimsTotal - 1
                        xlabel('Time (s)', 'FontSize', 8);
                    else
                        set(gca, 'XTickLabel', []);
                    end
                    set(gca, 'FontSize', 7);
                    grid on; box on;
                    hold off;
                end
            end
            % Sync ylim across all subplots
            yTopAll = max(yMaxAll * 1.1, 0.05);
            for ti = 1:(nSimsTotal * nC)
                ax = nexttile(ti);
                set(ax, 'YLim', [0, yTopAll]);
            end
            title(tlAll, sprintf( ...
                'Figure 6c — all sims (size=%d, window=%s, mode=%s)', ...
                sz, wn, opts.MatchingMode), 'FontSize', 12, 'FontWeight', 'bold');
            allSimsPngPath = fullfile(outDirSW, [allSimsFigName '.png']);
            % Bigger size for the dense 10×nC grid (need legible per-panel text)
            set(gcf, 'Position', [50, 50, 360*nC, 200*nSimsTotal]);
            exportgraphics(gcf, allSimsPngPath, 'Resolution', 200);
            try, exportgraphics(gcf, strrep(allSimsPngPath, '.png', '.pdf'), 'ContentType', 'vector'); catch, end
            try, print(gcf, strrep(allSimsPngPath, '.png', '.svg'), '-dsvg'); catch, end
            close(gcf);
            fprintf('    saved: %s\n', allSimsPngPath);

            % --- Fig 6 Expert vs NoSpout: three variants of the 2x3 + 2-col Fig6c sister ---
            try
                % Variant 1: 2-cond raw (Exp + NoSp) — primary fold figure
                [simRasterFB, selFB, extremeReps, extremeSim] = Figure6_ExpertVsNoSpout( ...
                    expTraces, simRaster, sel, sz, wn, outDirSW, ...
                    matcherOut, condColors, opts, simTimeAxis, stimEndSec, ...
                    biasFile, clampedFile, chosenDur, nSimsTotal, ...
                    'TargetConds', {'Expert', 'NoSpout'}, 'NormalizeToNaiveBaseline', false, ...
                    'SaveTiles', true, 'OutputStem', 'Fig6_ExpertVsNoSpout');

                % Variant 1b: 2-cond beta-isolated pair (Exp + NoSp share
                % every Ising param except beta). Saves under stem
                % Fig6_ExpertVsNoSpout_BetaIsolated. Skips silently if no
                % matched-except-beta pair exists in the top-10.
                try
                    Figure6_ExpertVsNoSpout( ...
                        expTraces, simRaster, sel, sz, wn, outDirSW, ...
                        matcherOut, condColors, opts, simTimeAxis, stimEndSec, ...
                        biasFile, clampedFile, chosenDur, nSimsTotal, ...
                        'TargetConds', {'Expert', 'NoSpout'}, ...
                        'NormalizeToNaiveBaseline', false, ...
                        'SaveTiles', true, ...
                        'BetaIsolationMode', true);
                catch MEbi
                    warning('Figure6_ExpertVsNoSpout(BetaIsolationMode) failed: %s', MEbi.message);
                end

                % Variant 2: 4-cond raw — all conds, own-baseline metrics
                Figure6_ExpertVsNoSpout( ...
                    expTraces, simRaster, sel, sz, wn, outDirSW, ...
                    matcherOut, condColors, opts, simTimeAxis, stimEndSec, ...
                    biasFile, clampedFile, chosenDur, nSimsTotal, ...
                    'TargetConds', opts.Conditions, 'NormalizeToNaiveBaseline', false, ...
                    'SaveTiles', true, 'OutputStem', 'Fig6_AllConds');

                % Variant 3: 4-cond Naive-baseline-normalized
                Figure6_ExpertVsNoSpout( ...
                    expTraces, simRaster, sel, sz, wn, outDirSW, ...
                    matcherOut, condColors, opts, simTimeAxis, stimEndSec, ...
                    biasFile, clampedFile, chosenDur, nSimsTotal, ...
                    'TargetConds', opts.Conditions, 'NormalizeToNaiveBaseline', true, ...
                    'SaveTiles', true, 'OutputStem', 'Fig6_AllConds_NaiveNorm');

                % Fold-best Fig6c-style 2-col (Exp + NoSp only)
                fbSimIdx = struct('Expert',  selFB.Expert.sim  - 1, ...
                                  'NoSpout', selFB.NoSpout.sim - 1);
                fbBias   = struct('Expert',  selFB.Expert.bias, ...
                                  'NoSpout', selFB.NoSpout.bias);
                Figure6c_ExpertVsNoSpout_2col(expTraces, simRasterFB, fbSimIdx, fbBias, ...
                    sz, wn, outDirSW, condColors, opts, simTimeAxis, stimEndSec, ...
                    rasterCmap, 'FoldBest');

                % Extreme-Δ Fig6c-style 2-col
                if isfield(extremeSim, 'Expert') && isfield(extremeSim, 'NoSpout')
                    xtSimIdx = struct('Expert',  extremeSim.Expert.simIdx, ...
                                      'NoSpout', extremeSim.NoSpout.simIdx);
                    Figure6c_ExpertVsNoSpout_2col(expTraces, extremeReps, xtSimIdx, fbBias, ...
                        sz, wn, outDirSW, condColors, opts, simTimeAxis, stimEndSec, ...
                        rasterCmap, 'ExtremeDelta');
                end
            catch ME
                warning('Figure6_ExpertVsNoSpout(+_2col) failed: %s', ME.message);
            end

            % --- Fig 6d (Model) — fold-change-vs-Naive swarm panel --------
            swarmPngPath = '';
            swarmStruct = struct();
            naiveIdx = find(strcmp(opts.Conditions, 'Naive'), 1);
            if isempty(naiveIdx) || ~isfield(simRaster, 'Naive')
                warning('Fig 6d swarm needs Naive present and loaded; skipping for size=%d window=%s', sz, wn);
            else
                stimMaskFull = simTimeAxis >= 0 & simTimeAxis < stimEndSec;
                if nnz(stimMaskFull) == 0
                    warning('Fig 6d swarm: empty stim mask; skipping');
                else
                    naivePerRep = mean(simRaster.Naive(:, stimMaskFull), 2, 'omitnan');
                    naiveMean = mean(naivePerRep, 'omitnan');
                    if ~isfinite(naiveMean) || naiveMean <= 0
                        warning('Fig 6d swarm: bad Naive mean (%g); skipping', naiveMean);
                    else
                        allFold = [];
                        allConds = {};
                        condOrder = {};
                        colorMap = [];
                        for c = 1:nC
                            cond = opts.Conditions{c};
                            if strcmp(cond, 'Naive') || ~isfield(simRaster, cond), continue; end
                            valsPerRep = mean(simRaster.(cond)(:, stimMaskFull), 2, 'omitnan');
                            % Optional batching for visual density
                            bs = max(1, round(opts.SwarmBatchSize));
                            if bs > 1 && numel(valsPerRep) >= bs
                                nB = floor(numel(valsPerRep) / bs);
                                batched = zeros(nB, 1);
                                for b = 1:nB
                                    idx = (b-1)*bs + (1:bs);
                                    batched(b) = mean(valsPerRep(idx), 'omitnan');
                                end
                                vals = batched;
                            else
                                vals = valsPerRep;
                            end
                            fold = vals / naiveMean;
                            allFold = [allFold; fold]; %#ok<AGROW>
                            allConds = [allConds; repmat({cond}, numel(fold), 1)]; %#ok<AGROW>
                            condOrder{end+1} = cond; %#ok<AGROW>
                            if isfield(condColors, cond)
                                colorMap = [colorMap; condColors.(cond)]; %#ok<AGROW>
                            else
                                colorMap = [colorMap; 0.5 0.5 0.5]; %#ok<AGROW>
                            end
                        end
                        if isempty(allFold)
                            warning('Fig 6d swarm: no non-Naive conditions loaded; skipping');
                        else
                            swarmFigName = 'Fig6d_Model';
                            figure('Name', swarmFigName, 'Color', 'w');
                            ax = gca;
                            hold(ax, 'on');
                            for ci = 1:numel(condOrder)
                                m = strcmp(allConds, condOrder{ci});
                                xVals = ci * ones(nnz(m), 1);
                                swarmchart(ax, xVals, allFold(m), 36, ...
                                    colorMap(ci, :), 'filled', ...
                                    'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'none', ...
                                    'XJitter', 'density', 'XJitterWidth', 0.45);
                                % Median tick mark
                                medVal = median(allFold(m), 'omitnan');
                                plot(ax, [ci - 0.25, ci + 0.25], [medVal, medVal], ...
                                    'k-', 'LineWidth', 2);
                            end
                            yline(ax, 1, '--', 'Color', [0.5 0.5 0.5], ...
                                'LineWidth', 1, 'Label', 'Naive mean', ...
                                'LabelHorizontalAlignment', 'left');
                            set(ax, 'XTick', 1:numel(condOrder), ...
                                    'XTickLabel', condOrder, ...
                                    'XLim', [0.5, numel(condOrder) + 0.5]);
                            ylabel(ax, 'Fold Change vs Naive (stim)');
                            title(ax, sprintf('Fig 6d Model — size=%d, window=%s', sz, wn));
                            grid(ax, 'on'); box(ax, 'on');
                            % Significance
                            pvals = computePairwiseStats(allFold, allConds, condOrder);
                            annotatePairwiseStats(ax, pvals, allFold, allConds, condOrder);
                            hold(ax, 'off');
                            swarmPngPath = fullfile(outDirSW, [swarmFigName '.png']);
                            exportgraphics(gcf, swarmPngPath, 'Resolution', 300);
                            try, exportgraphics(gcf, fullfile(outDirSW, [swarmFigName '.pdf']), 'ContentType', 'vector'); catch, end
                            try, print(gcf, fullfile(outDirSW, [swarmFigName '.svg']), '-dsvg'); catch, end
                            close(gcf);
                            fprintf('    saved: %s\n', swarmPngPath);
                            printPvalsTable(sprintf('Fig6d Model size=%d window=%s', sz, wn), pvals);
                            swarmStruct = struct('foldValues', allFold, ...
                                'groups', {allConds}, 'condOrder', {condOrder}, ...
                                'pvals', pvals, 'naiveMean', naiveMean, ...
                                'batchSize', bs, 'png', swarmPngPath);
                        end
                    end
                end
            end

            % --- Fig 6d (Model) — Top-10-sims swarm ----------------------
            % One dot per sim (10 per condition). For each sim s of cond c,
            % pick its own best bias (argmin RMSE vs cond c's experimental
            % trace within the current matcher window), then take the mean
            % of that sim's full-stim window from simTracesCondSim. Normalize
            % by Naive's mean across its 10 sims.
            top10PngPath = '';
            top10Struct = struct();
            if ~isfield(matcherOut, 'simTracesCondSim') || ~isfield(matcherOut, 'info')
                warning('Top-10-sims swarm: matcher missing simTracesCondSim/info; skipping size=%d window=%s', sz, wn);
            else
                SCS = matcherOut.simTracesCondSim;     % [nCondM x nSims x nBias x nFrames]
                biasVals10 = matcherOut.info.stimulusBiasValues;
                if isfield(matcherOut, 'simTimeAxis')
                    staM = matcherOut.simTimeAxis;
                else
                    staM = simTimeAxis;
                end
                nC_M = size(SCS, 1);
                nSims = size(SCS, 2);
                nBias_M = size(SCS, 3);
                if isfield(matcherOut, 'windowDefs') && isfield(matcherOut.windowDefs, wn)
                    wlim = matcherOut.windowDefs.(wn);
                elseif strcmp(wn, 'full'),    wlim = opts.MatchWindow;
                elseif strcmp(wn, 'onset'),   wlim = [0, 0.3];
                elseif strcmp(wn, 'offset'),  wlim = [stimEndSec, stimEndSec + 0.3];
                else,                         wlim = opts.MatchWindow;
                end

                stimMaskM = staM >= 0 & staM < stimEndSec;
                perSimStimMean = nan(nC_M, nSims);
                perSimBestBias = nan(nC_M, nSims);
                for c2 = 1:min(nC_M, numel(opts.Conditions))
                    cond_c = opts.Conditions{c2};
                    if ~isfield(expTraces, cond_c) || isempty(expTraces.(cond_c).recTraces)
                        continue;
                    end
                    et = expTraces.(cond_c).time;
                    rt = expTraces.(cond_c).recTraces;
                    em = mean(rt, 1, 'omitnan');
                    nCe = min(numel(et), numel(em));
                    et = et(1:nCe); em = em(1:nCe);
                    for s = 1:nSims
                        bestRMSE = Inf; bestB = NaN;
                        for b = 1:nBias_M
                            tr = squeeze(SCS(c2, s, b, :))';
                            interpTr = interp1(staM, tr, et, 'linear', NaN);
                            mask = et >= wlim(1) & et <= wlim(2) & ...
                                   ~isnan(em) & ~isnan(interpTr);
                            if nnz(mask) < 2, continue; end
                            r = sqrt(mean((em(mask) - interpTr(mask)).^2));
                            if r < bestRMSE
                                bestRMSE = r; bestB = b;
                            end
                        end
                        if isnan(bestB), continue; end
                        tr = squeeze(SCS(c2, s, bestB, :))';
                        if nnz(stimMaskM) > 0
                            perSimStimMean(c2, s) = mean(tr(stimMaskM), 'omitnan');
                            perSimBestBias(c2, s) = biasVals10(bestB);
                        end
                    end
                end

                naiveIdx10 = find(strcmp(opts.Conditions, 'Naive'), 1);
                if isempty(naiveIdx10) || all(isnan(perSimStimMean(naiveIdx10, :)))
                    warning('Top-10-sims swarm: Naive missing/empty; skipping');
                else
                    naiveRef = mean(perSimStimMean(naiveIdx10, :), 'omitnan');
                    if ~isfinite(naiveRef) || naiveRef <= 0
                        warning('Top-10-sims swarm: bad Naive ref %.4g; skipping', naiveRef);
                    else
                        allFold10 = []; allConds10 = {}; condOrder10 = {}; colorMap10 = [];
                        biasPerSim10 = struct();
                        for c2 = 1:min(nC_M, numel(opts.Conditions))
                            cond_c = opts.Conditions{c2};
                            if strcmp(cond_c, 'Naive'), continue; end
                            vals = perSimStimMean(c2, :)';
                            biases = perSimBestBias(c2, :)';
                            keep = ~isnan(vals);
                            vals = vals(keep);
                            biases = biases(keep);
                            if isempty(vals), continue; end
                            fold = vals / naiveRef;
                            allFold10 = [allFold10; fold]; %#ok<AGROW>
                            allConds10 = [allConds10; repmat({cond_c}, numel(fold), 1)]; %#ok<AGROW>
                            condOrder10{end+1} = cond_c; %#ok<AGROW>
                            if isfield(condColors, cond_c)
                                colorMap10 = [colorMap10; condColors.(cond_c)]; %#ok<AGROW>
                            else
                                colorMap10 = [colorMap10; 0.5 0.5 0.5]; %#ok<AGROW>
                            end
                            biasPerSim10.(cond_c) = biases;
                        end
                        if isempty(allFold10)
                            warning('Top-10-sims swarm: empty dataset; skipping');
                        else
                            figName10 = 'Fig6d_Model_Top10Sims';
                            figure('Name', figName10, 'Color', 'w');
                            ax = gca;
                            hold(ax, 'on');
                            for ci = 1:numel(condOrder10)
                                m = strcmp(allConds10, condOrder10{ci});
                                swarmchart(ax, ci * ones(nnz(m), 1), allFold10(m), 60, ...
                                    colorMap10(ci, :), 'filled', ...
                                    'MarkerFaceAlpha', 0.75, 'MarkerEdgeColor', 'none', ...
                                    'XJitter', 'density', 'XJitterWidth', 0.45);
                                medVal = median(allFold10(m), 'omitnan');
                                plot(ax, [ci - 0.25, ci + 0.25], [medVal, medVal], ...
                                    'k-', 'LineWidth', 2);
                            end
                            yline(ax, 1, '--', 'Color', [0.5 0.5 0.5], ...
                                'LineWidth', 1, 'Label', 'Naive mean', ...
                                'LabelHorizontalAlignment', 'left');
                            set(ax, 'XTick', 1:numel(condOrder10), ...
                                    'XTickLabel', condOrder10, ...
                                    'XLim', [0.5, numel(condOrder10) + 0.5]);
                            ylabel(ax, 'Fold Change vs Naive (stim)');
                            title(ax, sprintf('Fig 6d Model — top 10 sims (size=%d, window=%s)', sz, wn));
                            grid(ax, 'on'); box(ax, 'on');
                            pvals10 = computePairwiseStats(allFold10, allConds10, condOrder10);
                            annotatePairwiseStats(ax, pvals10, allFold10, allConds10, condOrder10);
                            hold(ax, 'off');
                            top10PngPath = fullfile(outDirSW, [figName10 '.png']);
                            exportgraphics(gcf, top10PngPath, 'Resolution', 300);
                            try, exportgraphics(gcf, fullfile(outDirSW, [figName10 '.pdf']), 'ContentType', 'vector'); catch, end
                            try, print(gcf, fullfile(outDirSW, [figName10 '.svg']), '-dsvg'); catch, end
                            close(gcf);
                            fprintf('    saved: %s\n', top10PngPath);
                            printPvalsTable(sprintf('Fig6d Top10 sims size=%d window=%s', sz, wn), pvals10);
                            top10Struct = struct('foldValues', allFold10, ...
                                'groups', {allConds10}, 'condOrder', {condOrder10}, ...
                                'pvals', pvals10, 'naiveRef', naiveRef, ...
                                'biasPerSim', biasPerSim10, 'png', top10PngPath);
                        end
                    end
                end
            end

            out.bySize.(sizeKey).(sprintf('window_%s', wn)) = struct( ...
                'selection', sel, 'pngPath', pngPath, 'swarm', swarmStruct, ...
                'top10swarm', top10Struct);
        end
    end
    out.opts = opts;
end


%% ====================================================================
function cmap = loadEntropyColourmap()
%LOADENTROPYCOLOURMAP  Locate and load the paper's standard fraction-active
%   colormap. Tries a few likely paths (Windows-local repo, Linux cluster
%   repo, addpath search) before falling back to a built-in red ramp.
    candidates = {
        fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', '..', ...
            'Helper Functions and Packages', 'EntropyColourMap.mat'), ...
        'EntropyColourMap.mat', ...
        'EntropyColourMap.mat'
    };
    for k = 1:numel(candidates)
        f = char(candidates{k});
        if startsWith(f, '~'), f = strrep(f, '~', char(java.lang.System.getProperty('user.home'))); end
        if exist(f, 'file')
            S = load(f);
            if isfield(S, 'EntropyColourmap')
                cmap = S.EntropyColourmap;
                return;
            end
        end
    end
    % Fallback: red ramp from white to dark red (similar visual feel)
    n = 256;
    cmap = [linspace(1, 0.55, n)', linspace(1, 0.0, n)', linspace(1, 0.0, n)'];
end


%% ====================================================================
function file = locateAggregate(perturbDir, mode)
    pat = sprintf('PerturbationResults_%s_*.mat', mode);
    files = dir(fullfile(perturbDir, pat));
    if isempty(files)
        error('No aggregate matching %s in %s', pat, perturbDir);
    end
    [~, idx] = max([files.datenum]);
    file = fullfile(files(idx).folder, files(idx).name);
end


%% ====================================================================
function [sizes, durs, biases, preStim, postStim, gMeanSF] = loadAggregateMetadata(file)
    sizes  = double(h5read(file, '/stimulus_sizes'));      sizes  = sizes(:)';
    durs   = double(h5read(file, '/stimulus_durations'));  durs   = durs(:)';
    biases = double(h5read(file, '/stimulus_bias_values')); biases = biases(:)';
    preStim  = double(h5read(file, '/pre_stim_frames'));
    postStim = double(h5read(file, '/post_stim_frames'));
    gMeanSF  = double(h5read(file, '/global_mean_sf'));
end


%% ====================================================================
function mo = loadMatcherForSize(matcherOutputDir, sz)
%LOADMATCHERFORSIZE  Try per-size file first, fall back to all-sizes file.
    mo = [];
    perSize = fullfile(matcherOutputDir, sprintf('matcher_output_size%d.mat', sz));
    allSize = fullfile(matcherOutputDir, 'matcher_output.mat');
    candidates = {perSize, allSize};
    for k = 1:numel(candidates)
        f = candidates{k};
        if ~exist(f, 'file'), continue; end
        S = load(f, 'out');
        if ~isfield(S, 'out'), continue; end
        if isfield(S.out, 'bySize') && isfield(S.out.bySize, sprintf('size_%d', sz))
            mo = S.out.bySize.(sprintf('size_%d', sz));
            return;
        end
        % Older single-size matcher_output structure
        if isfield(S.out, 'bestBiasPerCondition')
            mo = S.out;
            return;
        end
    end
end


%% ====================================================================
function pvals = computePairwiseStats(allValues, allGroups, groupLabels)
    nG = length(groupLabels);
    pvals = struct();
    for i = 1:nG-1
        for j = i+1:nG
            g1 = allValues(strcmp(allGroups, groupLabels{i}));
            g2 = allValues(strcmp(allGroups, groupLabels{j}));
            g1 = g1(~isnan(g1));
            g2 = g2(~isnan(g2));
            key = sprintf('%s_vs_%s', matlab.lang.makeValidName(groupLabels{i}), ...
                                      matlab.lang.makeValidName(groupLabels{j}));
            if numel(g1) > 1 && numel(g2) > 1
                pvals.(key) = ranksum(g1, g2);
            else
                pvals.(key) = NaN;
            end
        end
    end
end


%% ====================================================================
function annotatePairwiseStats(ax, pvals, allValues, allGroups, groupLabels)
    nG = length(groupLabels);
    validVals = allValues(~isnan(allValues));
    if isempty(validVals), return; end
    yMax = max(validVals);
    yMin = min(validVals);
    yRange = yMax - yMin;
    if yRange == 0, yRange = abs(yMax) + 1; end
    yStep = 0.08 * yRange;
    pairIdx = 0;
    for i = 1:nG-1
        for j = i+1:nG
            pairIdx = pairIdx + 1;
            key = sprintf('%s_vs_%s', matlab.lang.makeValidName(groupLabels{i}), ...
                                      matlab.lang.makeValidName(groupLabels{j}));
            if ~isfield(pvals, key) || isnan(pvals.(key)), continue; end
            pVal = pvals.(key);
            yPos = yMax + yStep * pairIdx;
            xMid = (i + j) / 2;
            plot(ax, [i, j], [yPos, yPos] - yStep*0.2, 'k-', 'LineWidth', 1);
            text(ax, xMid, yPos, sprintf('%s (p=%.3g)', pToStars(pVal), pVal), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, ...
                'FontWeight', 'bold', 'BackgroundColor', 'white', 'EdgeColor', 'none');
        end
    end
    % Bump the axis limit to fit the brackets
    yMaxNew = yMax + yStep * (pairIdx + 1);
    yLimNow = ylim(ax);
    if yMaxNew > yLimNow(2)
        ylim(ax, [yLimNow(1), yMaxNew]);
    end
end


%% ====================================================================
function s = pToStars(p)
    if p < 0.001, s = '***';
    elseif p < 0.01, s = '**';
    elseif p < 0.05, s = '*';
    else, s = 'n.s.';
    end
end


%% ====================================================================
function printPvalsTable(label, pvals)
    fprintf('\n=== Pairwise p-values: %s ===\n', label);
    fields = fieldnames(pvals);
    for fi = 1:numel(fields)
        p = pvals.(fields{fi});
        if isnan(p)
            fprintf('  %s: NaN\n', fields{fi});
        else
            fprintf('  %s: p=%.4g %s\n', fields{fi}, p, pToStars(p));
        end
    end
end
