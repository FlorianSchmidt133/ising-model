function Fig6c_AverageVariants_local(varargin)
%FIG6C_AVERAGEVARIANTS_LOCAL  Three condition-average Fig 6c variants from
%   matcher mats + local ExperimentalData.mat (no HDF5 / cluster needed).
%
%   Variant 1: Fig6c_AverageOverlay        — 1 panel, 4 cond means overlaid
%   Variant 2: Fig6c_AverageTop10_4Panel   — 4 panels, each = mean over top-N
%                                            sims for that cond + exp overlay
%   Variant 3: Fig6c_DataModelOverlay      — 1x2 panels: experimental overlay
%                                            (left) + model overlay (right)
%
%   Data source: matcher_output_size{N}.mat (saves simTracesCondSim,
%   simTimeAxis, bestBiasPerCondition, info.stimulusBiasValues).
%   Each entry simTracesCondSim(c, s, b, t) is already the rep-mean for
%   (cond c, sim s, bias b, frame t), so pooled top-N = mean across s.

    p = inputParser;
    addParameter(p, 'MatcherDir', ...
        'Paper/K0_25-50-25_perCondition_double_pulse_bias10/_matcher_mats');
    addParameter(p, 'OutBase', ...
        'Paper/K0_25-50-25_perCondition_double_pulse_bias10/Fig6_perCondition/Fig6c');
    addParameter(p, 'ExpDataFile', 'Paper/Fig. 5 Ising Models/ExperimentalData.mat');
    % CorrectRec.mat: contains the authoritative Rec.<cond>.AnimalName per
    % recording (16 rows for Naive, 12 for Beginner, 17 for Expert, 14 for
    % NoSpout). recording_indices in ExperimentalData.mat map into this.
    addParameter(p, 'CorrectRecFile', mba_p('CorrectRec.mat'));
    addParameter(p, 'Sizes', [2 3 4]);
    addParameter(p, 'Window', 'full_naive');
    addParameter(p, 'Conditions', {'Naive', 'Beginner', 'Expert', 'NoSpout'});
    addParameter(p, 'DisplayWindow', [-8, 10]);  % full trial extent at 10 Hz
    addParameter(p, 'SamplingRate', 10);
    addParameter(p, 'TopN', 10);
    % BiasOverride: struct mapping 'size_<N>' -> bias_value. Forces this bias
    % for ALL conditions at that size, overriding bestBiasPerCondition from
    % the matcher. Defaults to the values shown in the existing AllSims
    % panels (the user's preferred result), so the averaged figures preserve
    % the stim transients seen there. Pass an empty struct() to disable.
    defaultBiasOverride = struct( ...
        'size_2', 1.75, ...
        'size_3', 1.25, ...
        'size_4', 1.00);
    addParameter(p, 'BiasOverride', defaultBiasOverride, @(x) isstruct(x));
    % DurSliceFile: optional path to a pre-extracted .mat with dur=25 (or
    % other) traces. When provided, BYPASSES the matcher mats entirely and
    % reads `size_N_traces` arrays of shape [4 conds, nSims, nFrames]. The
    % bias is baked into the extraction (one bias per size).
    % Expected fields: time_axis, conditions (cellstr), stim_end_sec,
    % size_2_traces / size_3_traces / size_4_traces, size_N_bias_value.
    addParameter(p, 'DurSliceFile', '', @ischar);
    parse(p, varargin{:});
    opts = p.Results;

    condColors = struct( ...
        'Naive',    [0.3373, 0.7059, 0.9137], ...
        'Beginner', [0.8431, 0.2549, 0.6078], ...
        'Expert',   [0,      0.6196, 0.4510], ...
        'NoSpout',  [0.8353, 0.3686, 0]);

    % Per-condition recording skip (mirrors Figure4.m / Fig6c default)
    skipSet = struct( ...
        'Naive',    [1 9 10 14 16], ...
        'Beginner', [1 6 7 11 12], ...
        'Expert',   [1 4 11:17], ...
        'NoSpout',  [1 4 9 10 11 14]);

    expTraces = loadExperimentalTraces(opts.ExpDataFile, opts.Conditions, ...
        skipSet, opts.SamplingRate, opts.CorrectRecFile);

    % If DurSliceFile is provided, load once for use across all sizes.
    durSlice = [];
    if ~isempty(opts.DurSliceFile) && exist(opts.DurSliceFile, 'file')
        durSlice = load(opts.DurSliceFile);
        fprintf('Using pre-extracted dur slices from: %s\n', opts.DurSliceFile);
        fprintf('  time_axis range: %.2f to %.2f s, n=%d\n', ...
            durSlice.time_axis(1), durSlice.time_axis(end), numel(durSlice.time_axis));
        fprintf('  stim_end_sec: %.2f\n', durSlice.stim_end_sec);
    end

    for sz = opts.Sizes(:)'
        outDir = fullfile(opts.OutBase, sprintf('size_%d', sz), ...
            sprintf('window_%s', opts.Window));
        if ~exist(outDir, 'dir'), mkdir(outDir); end
        matcherConds = {'Naive', 'Beginner', 'Expert', 'NoSpout'};
        condMu   = struct(); condSem = struct(); condBias = struct();
        condStimPerSim = struct(); condBasePerSim = struct();   % for swarm figs
        condStimMaxPerSim = struct();                            % max-delta variant

        if ~isempty(durSlice)
            % --- Branch A: read from pre-extracted dur slice file ---
            fldTraces = sprintf('size_%d_traces', sz);
            fldBias   = sprintf('size_%d_bias_value', sz);
            if ~isfield(durSlice, fldTraces)
                warning('Dur slice file missing field %s; skipping', fldTraces);
                continue;
            end
            traces = durSlice.(fldTraces);   % [4 conds, nSims, nFrames]
            bv = durSlice.(fldBias);
            simTimeAxis = durSlice.time_axis(:)';
            stimEndSec  = durSlice.stim_end_sec;
            nSimsAvail  = size(traces, 2);
            nTop        = min(opts.TopN, nSimsAvail);
            fprintf('  size=%d: dur-sliced bias=%.2f, nSims=%d, nTop=%d, stim_end=%.2fs\n', ...
                sz, bv, nSimsAvail, nTop, stimEndSec);
            for c = 1:numel(opts.Conditions)
                cond = opts.Conditions{c};
                cIdx = find(strcmp(matcherConds, cond), 1);
                if isempty(cIdx), continue; end
                perSimMu = squeeze(traces(cIdx, 1:nTop, :));   % [nTop x nFrames]
                if size(perSimMu, 2) == 1, perSimMu = perSimMu'; end
                if all(isnan(perSimMu(:)))
                    fprintf('    cond=%s: all-NaN; skipping\n', cond); continue;
                end
                condMu.(cond)   = mean(perSimMu, 1, 'omitnan');
                condSem.(cond)  = std(perSimMu, 0, 1, 'omitnan') / sqrt(size(perSimMu, 1));
                condBias.(cond) = bv;
                % --- Per-sim stim/baseline means for swarm figures ---
                %  stim:   [0, stimEndSec)
                %  base:   [-5, 0)
                stimMaskRaw = simTimeAxis >= 0 & simTimeAxis < stimEndSec;
                baseMaskRaw = simTimeAxis >= -5 & simTimeAxis < 0;
                condStimPerSim.(cond) = mean(perSimMu(:, stimMaskRaw), 2, 'omitnan');
                condBasePerSim.(cond) = mean(perSimMu(:, baseMaskRaw), 2, 'omitnan');
                condStimMaxPerSim.(cond) = max(perSimMu(:, stimMaskRaw), [], 2, 'omitnan');
            end
            secPerFrame = simTimeAxis(2) - simTimeAxis(1);
        else
            % --- Branch B: read from matcher mat (legacy path) ---
            matFile = fullfile(opts.MatcherDir, sprintf('matcher_output_size%d.mat', sz));
            if ~exist(matFile, 'file')
                warning('matcher mat not found: %s', matFile); continue;
            end
            S = load(matFile, 'out');
            if isfield(S.out, 'bySize') && isfield(S.out.bySize, sprintf('size_%d', sz))
                mo = S.out.bySize.(sprintf('size_%d', sz));
            else
                mo = S.out;
            end

            simTracesCondSim = mo.simTracesCondSim;
            simTimeAxis      = mo.simTimeAxis(:)';
            biasValues       = double(mo.info.stimulusBiasValues(:));
            bbpc             = mo.bestBiasPerCondition;
            nSimsAvail = size(simTracesCondSim, 2);
            nTop       = min(opts.TopN, nSimsAvail);
            sizeKey = sprintf('size_%d', sz);
            useOverride = isfield(opts.BiasOverride, sizeKey);
            if useOverride
                overrideBias = opts.BiasOverride.(sizeKey);
                fprintf('  size=%d: BIAS OVERRIDE = %.2f (all conditions)\n', sz, overrideBias);
            end
            for c = 1:numel(opts.Conditions)
                cond = opts.Conditions{c};
                if useOverride
                    bv = overrideBias;
                else
                    if ~isfield(bbpc, cond) || ~isfield(bbpc.(cond), opts.Window), continue; end
                    bv = bbpc.(cond).(opts.Window).bias_value;
                    if isinf(bv) || isnan(bv), continue; end
                end
                [~, bIdx] = min(abs(biasValues - bv));
                cIdx = find(strcmp(matcherConds, cond), 1);
                if isempty(cIdx), continue; end
                perSimMu = squeeze(simTracesCondSim(cIdx, 1:nTop, bIdx, :));
                if size(perSimMu, 2) == 1, perSimMu = perSimMu'; end
                if all(isnan(perSimMu(:))), continue; end
                condMu.(cond)   = mean(perSimMu, 1, 'omitnan');
                condSem.(cond)  = std(perSimMu, 0, 1, 'omitnan') / sqrt(size(perSimMu, 1));
                condBias.(cond) = biasValues(bIdx);
            end
            secPerFrame = simTimeAxis(2) - simTimeAxis(1);
            if isfield(mo, 'info') && isfield(mo.info, 'stimulusDurations') && ~isempty(mo.info.stimulusDurations)
                stimEndSec = double(mo.info.stimulusDurations(1)) * secPerFrame;
            else
                stimEndSec = 5.0;
            end
        end

        % --- Apply dispMask to BOTH tWin and the per-cond traces ---
        dispMask = simTimeAxis >= opts.DisplayWindow(1) & ...
                   simTimeAxis <= opts.DisplayWindow(2);
        tWin = simTimeAxis(dispMask);
        condNames_loc = fieldnames(condMu);
        for k = 1:numel(condNames_loc)
            cn = condNames_loc{k};
            condMu.(cn)  = condMu.(cn)(dispMask);
            condSem.(cn) = condSem.(cn)(dispMask);
        end
        fprintf('  size=%d: stimEndSec=%.2f (secPerFrame=%.4f)\n', sz, stimEndSec, secPerFrame);

        renderAverageOverlay(condMu, condSem, condBias, tWin, dispMask, ...
            opts.DisplayWindow, stimEndSec, opts.Conditions, condColors, ...
            sz, opts.Window, nTop, outDir);

        renderAverageTop10_4Panel(condMu, condSem, condBias, tWin, dispMask, ...
            opts.DisplayWindow, stimEndSec, opts.Conditions, condColors, ...
            sz, opts.Window, nTop, expTraces, outDir);

        renderDataModelOverlay(condMu, condSem, tWin, dispMask, ...
            opts.DisplayWindow, stimEndSec, opts.Conditions, condColors, ...
            sz, opts.Window, nTop, expTraces, outDir);

        % --- Swarm-style figures (only from dur-slice branch which has raw per-sim) ---
        if ~isempty(fieldnames(condStimPerSim))
            renderDeltaStimBaseline(condStimPerSim, condBasePerSim, opts.Conditions, ...
                condColors, condBias, sz, opts.Window, outDir);
            % Experimental parallel: per-recording delta from expTraces
            renderDeltaStimBaselineExp(expTraces, opts.Conditions, condColors, ...
                condBias, sz, opts.Window, stimEndSec, outDir);
            % Experimental per-animal: average recordings within animal first
            renderDeltaStimBaselineExpPerAnimal(expTraces, opts.Conditions, condColors, ...
                condBias, sz, opts.Window, stimEndSec, outDir);
            % --- AllSims-style: 10x4 grid + per-row figures, no exp overlay ---
            fldRaw = sprintf('size_%d_traces_raw', sz);
            if isfield(durSlice, fldRaw)
                rawTraces = durSlice.(fldRaw);  % [4 conds x 10 sims x 50 reps x nFrames]
                renderAllSimsNoExp(rawTraces, opts.Conditions, condColors, ...
                    condBias, simTimeAxis, stimEndSec, opts.DisplayWindow, ...
                    sz, opts.Window, outDir);
            end
            % --- MAX-delta variants (peak amplitude instead of mean) ---
            renderDeltaStimBaselineMax(condStimMaxPerSim, condBasePerSim, opts.Conditions, ...
                condColors, condBias, sz, opts.Window, outDir);
            renderDeltaStimBaselineMaxExp(expTraces, opts.Conditions, condColors, ...
                condBias, sz, opts.Window, stimEndSec, outDir);
            renderDeltaStimBaselineMaxExpPerAnimal(expTraces, opts.Conditions, condColors, ...
                condBias, sz, opts.Window, stimEndSec, outDir);
        end
    end
end

%% =====================================================================
function renderAverageOverlay(condMu, condSem, condBias, tWin, ~, ...
        dispWin, stimEndSec, conditions, condColors, sz, wn, nTop, outDir)
    fig = figure('Name', 'Fig6c_AverageOverlay', 'Color', 'w');
    hold on; yMax = 0;
    for c = 1:numel(conditions)
        cond = conditions{c};
        if ~isfield(condMu, cond), continue; end
        mu  = condMu.(cond)(:)';  sem = condSem.(cond)(:)';
        % Clip to dispMask length defensively
        nU = min(numel(mu), numel(tWin));
        mu = mu(1:nU); sem = sem(1:nU);
        col = condColors.(cond);
        fill([tWin(1:nU), fliplr(tWin(1:nU))], ...
             [mu + sem, fliplr(mu - sem)], col, ...
             'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(tWin(1:nU), mu, 'Color', col, 'LineWidth', 1.8, ...
             'DisplayName', sprintf('%s (b=%.2f)', cond, condBias.(cond)));
        yMax = max(yMax, max(mu + sem));
    end
    xline(0, 'k:', 'HandleVisibility', 'off'); xline(stimEndSec, 'k:', 'HandleVisibility', 'off');
    xlim(dispWin); ylim([0, max(yMax * 1.1, 0.05)]);
    xlabel('Time from stim onset (s)');
    ylabel('Fraction active (FOV crop)');
    legend('Location', 'best', 'Box', 'off');
    grid on; box on;
    title(sprintf('Fig 6c — model mean (top-%d sim pool)', nTop), ...
        'FontSize', 11, 'Interpreter', 'none');
    set(fig, 'Position', [50, 50, 700, 500]);
    saveAllFormats(fullfile(outDir, 'Fig6c_AverageOverlay'));
    close(fig);
end

%% =====================================================================
function renderAverageTop10_4Panel(condMu, condSem, condBias, tWin, ~, ...
        dispWin, stimEndSec, conditions, condColors, sz, wn, nTop, ...
        expTraces, outDir)
    nC = numel(conditions);
    fig = figure('Name', 'Fig6c_AverageTop10_4Panel', 'Color', 'w');
    tl = tiledlayout(1, nC, 'TileSpacing', 'compact', 'Padding', 'compact');
    yMax = 0; axes_ = gobjects(1, nC);
    for c = 1:nC
        cond = conditions{c};
        ax = nexttile; axes_(c) = ax; hold(ax, 'on');
        if isfield(condMu, cond)
            mu  = condMu.(cond)(:)';  sem = condSem.(cond)(:)';
            nU = min(numel(mu), numel(tWin));
            mu = mu(1:nU); sem = sem(1:nU);
            col = condColors.(cond);
            fill([tWin(1:nU), fliplr(tWin(1:nU))], [mu+sem, fliplr(mu-sem)], ...
                col, 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(tWin(1:nU), mu, 'Color', col, 'LineWidth', 1.8);
            yMax = max(yMax, max(mu + sem));
        end
        if isfield(expTraces, cond) && ~isempty(expTraces.(cond).recTraces)
            et = expTraces.(cond).time;
            rt = expTraces.(cond).recTraces;
            em = mean(rt, 1, 'omitnan');
            nU = min(numel(et), numel(em)); et = et(1:nU); em = em(1:nU);
            keep = et >= dispWin(1) & et <= dispWin(2);
            plot(et(keep), em(keep), 'k--', 'LineWidth', 1.2);
            if any(keep), yMax = max(yMax, max(em(keep))); end
        end
        xline(ax, 0, 'k:', 'HandleVisibility', 'off'); xline(ax, stimEndSec, 'k:', 'HandleVisibility', 'off');
        xlim(ax, dispWin);
        xlabel(ax, 'Time (s)');
        if c == 1, ylabel(ax, 'Fraction active (FOV crop)'); end
        if isfield(condBias, cond)
            title(ax, sprintf('%s (b=%.2f)', cond, condBias.(cond)), ...
                'Color', condColors.(cond), 'FontSize', 10);
        else
            title(ax, cond, 'Color', condColors.(cond), 'FontSize', 10);
        end
        grid(ax, 'on'); box(ax, 'on');
    end
    yTop = max(yMax * 1.1, 0.05);
    for c = 1:nC, set(axes_(c), 'YLim', [0, yTop]); end
    title(tl, sprintf('Fig 6c — model mean (top-%d sims) per condition', nTop), ...
        'FontSize', 12, 'Interpreter', 'none');
    set(fig, 'Position', [50, 50, 320*nC, 420]);
    saveAllFormats(fullfile(outDir, 'Fig6c_AverageTop10_4Panel'));
    close(fig);
end

%% =====================================================================
function renderDataModelOverlay(condMu, condSem, tWin, ~, dispWin, ...
        stimEndSec, conditions, condColors, sz, wn, nTop, expTraces, outDir)
    fig = figure('Name', 'Fig6c_DataModelOverlay', 'Color', 'w');
    tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    % LEFT: experimental overlay
    axL = nexttile; hold(axL, 'on'); yMaxL = 0;
    for c = 1:numel(conditions)
        cond = conditions{c};
        if ~isfield(expTraces, cond) || isempty(expTraces.(cond).recTraces), continue; end
        et = expTraces.(cond).time;
        rt = expTraces.(cond).recTraces;
        em  = mean(rt, 1, 'omitnan');
        esem = std(rt, 0, 1, 'omitnan') / sqrt(size(rt, 1));
        nU = min(numel(et), numel(em)); et = et(1:nU); em = em(1:nU); esem = esem(1:nU);
        keep = et >= dispWin(1) & et <= dispWin(2);
        col = condColors.(cond);
        etK = et(keep); emK = em(keep); esemK = esem(keep);
        fill([etK, fliplr(etK)], [emK+esemK, fliplr(emK-esemK)], col, ...
            'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(etK, emK, 'Color', col, 'LineWidth', 1.8, 'DisplayName', cond);
        if any(keep), yMaxL = max(yMaxL, max(emK + esemK)); end
    end
    xline(axL, 0, 'k:', 'HandleVisibility', 'off'); xline(axL, stimEndSec, 'k:', 'HandleVisibility', 'off');
    xlim(axL, dispWin);
    xlabel(axL, 'Time (s)'); ylabel(axL, 'Fraction active');
    title(axL, 'Experimental — condition means ± SEM (per-rec)', 'Interpreter', 'none');
    legend(axL, 'Location', 'best', 'Box', 'off');
    grid(axL, 'on'); box(axL, 'on');

    % RIGHT: model overlay (top-N pooled)
    axR = nexttile; hold(axR, 'on'); yMaxR = 0;
    for c = 1:numel(conditions)
        cond = conditions{c};
        if ~isfield(condMu, cond), continue; end
        mu  = condMu.(cond)(:)';  sem = condSem.(cond)(:)';
        nU = min(numel(mu), numel(tWin)); mu = mu(1:nU); sem = sem(1:nU);
        col = condColors.(cond);
        fill([tWin(1:nU), fliplr(tWin(1:nU))], [mu+sem, fliplr(mu-sem)], col, ...
            'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(tWin(1:nU), mu, 'Color', col, 'LineWidth', 1.8, 'DisplayName', cond);
        yMaxR = max(yMaxR, max(mu + sem));
    end
    xline(axR, 0, 'k:', 'HandleVisibility', 'off'); xline(axR, stimEndSec, 'k:', 'HandleVisibility', 'off');
    xlim(axR, dispWin);
    xlabel(axR, 'Time (s)');
    title(axR, sprintf('Model — top-%d sim pool', nTop), 'Interpreter', 'none');
    grid(axR, 'on'); box(axR, 'on');

    % Shared Y across both
    yTop = max([yMaxL, yMaxR]) * 1.1;
    if yTop <= 0 || ~isfinite(yTop), yTop = 0.05; end
    set([axL, axR], 'YLim', [0, yTop]);

    title(tl, 'Fig 6c — data vs model condition means', ...
        'FontSize', 12, 'Interpreter', 'none');
    set(fig, 'Position', [50, 50, 1100, 480]);
    saveAllFormats(fullfile(outDir, 'Fig6c_DataModelOverlay'));
    close(fig);
end

%% =====================================================================
function expTraces = loadExperimentalTraces(file, conditions, skipSet, fs, correctRecFile)
%LOADEXPERIMENTALTRACES  Load per-recording trial-mean traces for each
%   condition from ExperimentalData.mat, applying the Figure4 Skip list.
%   Uses CorrectRec.mat's Rec.<cond>.AnimalName as the authoritative source
%   for per-recording animal IDs (more reliable than RecordingMetadata's
%   embedded animal_names).
    expTraces = struct();
    if ~exist(file, 'file')
        warning('ExperimentalData.mat not found at %s', file);
        return;
    end
    SE = load(file, 'BinarisedData', 'RecordingMetadata', 'TimingInfo');

    % Load CorrectRec for authoritative animal IDs
    correctRec = [];
    if nargin >= 5 && ~isempty(correctRecFile) && exist(correctRecFile, 'file')
        CR = load(correctRecFile, 'Rec');
        if isfield(CR, 'Rec')
            correctRec = CR.Rec;
            fprintf('Loaded CorrectRec.mat: %s\n', strjoin(fieldnames(correctRec), ', '));
        end
    end
    if isfield(SE, 'TimingInfo') && ~isempty(fieldnames(SE.TimingInfo))
        expPrestim = double(SE.TimingInfo.trial_structure.prestim_frames(:)');
        expTotal   = double(SE.TimingInfo.trial_structure.total_frames);
    else
        expPrestim = 1:80; expTotal = 185;
    end
    expStimOnset = expPrestim(end) + 1;
    timeAxis = ((1:expTotal) - expStimOnset) / fs;
    for c = 1:numel(conditions)
        cond = conditions{c};
        if ~isfield(SE.BinarisedData, cond) || isempty(SE.BinarisedData.(cond))
            continue;
        end
        bin = SE.BinarisedData.(cond);
        [gY, gX, T, nTr] = size(bin);
        recMask = any(bin > 0, [3 4]);
        recMask = reshape(recMask, gY, gX);
        if nnz(recMask) == 0, continue; end
        binFlat = reshape(bin, gY*gX, T, nTr);
        binFlat = binFlat(recMask(:), :, :);

        recTraces = [];
        recAnimals = {};   % aligned with recTraces rows (one entry per kept rec)
        if isfield(SE, 'RecordingMetadata') && isfield(SE.RecordingMetadata, cond)
            meta = SE.RecordingMetadata.(cond);
            nTpr = double(meta.nTrials_per_recording(:)');
            if isfield(meta, 'recording_indices') && ~isempty(meta.recording_indices)
                recIdxList = double(meta.recording_indices(:)');
            else
                recIdxList = 1:numel(nTpr);
            end
            % --- Resolve animal names per recording ---
            % Prefer CorrectRec.Rec.<cond>.AnimalName indexed via recording_indices.
            % Fall back to embedded animal_names if CorrectRec unavailable.
            storedNames = cell(1, numel(nTpr));
            usedSource = '';
            if ~isempty(correctRec) && isfield(correctRec, cond) ...
                    && istable(correctRec.(cond)) ...
                    && ismember('AnimalName', correctRec.(cond).Properties.VariableNames)
                recTbl = correctRec.(cond);
                an = recTbl.AnimalName;
                for r = 1:numel(nTpr)
                    if r <= numel(recIdxList)
                        idx = recIdxList(r);
                        if idx >= 1 && idx <= height(recTbl)
                            val = an{idx};
                            if iscell(val), val = val{1}; end
                            storedNames{r} = char(val);
                            continue;
                        end
                    end
                    storedNames{r} = sprintf('rec%d', r);
                end
                usedSource = sprintf('CorrectRec.Rec.%s.AnimalName', cond);
            elseif isfield(meta, 'animal_names') && ~isempty(meta.animal_names)
                an = meta.animal_names;
                if isstring(an), an = cellstr(an); end
                if iscell(an) && size(an, 1) > 1, an = an(:)'; end
                storedNames = an;
                usedSource = 'RecordingMetadata.animal_names (fallback)';
            else
                storedNames = arrayfun(@(i) sprintf('rec%d', i), 1:numel(nTpr), ...
                    'UniformOutput', false);
                usedSource = 'synthetic rec## (no source)';
            end
            fprintf('  %s: animal-name source = %s\n', cond, usedSource);
            sk = [];
            if isfield(skipSet, cond), sk = skipSet.(cond); end
            cursor = 0;
            for r = 1:numel(nTpr)
                n = nTpr(r);
                if n <= 0, continue; end
                if cursor + n > nTr, n = nTr - cursor; end
                if n <= 0, break; end
                origIdx = recIdxList(r);
                if ismember(origIdx, sk)
                    cursor = cursor + n; continue;
                end
                trialIdx = cursor + (1:n);
                recBin = bin(:, :, :, trialIdx);
                cursor = cursor + n;
                recFlat = reshape(recBin, gY*gX, T, n);
                recFlat = recFlat(recMask(:), :, :);
                recTrace = reshape(mean(mean(recFlat, 1), 3), 1, []);
                recTraces = [recTraces; recTrace]; %#ok<AGROW>
                if r <= numel(storedNames)
                    recAnimals{end+1, 1} = char(storedNames{r}); %#ok<AGROW>
                else
                    recAnimals{end+1, 1} = sprintf('rec%d', r); %#ok<AGROW>
                end
            end
        end
        if isempty(recTraces)
            poolMean = reshape(mean(mean(binFlat, 1), 3), 1, []);
            recTraces = poolMean;
            recAnimals = {'pooled'};
        end
        expTraces.(cond) = struct('time', timeAxis, ...
            'recTraces', recTraces, ...
            'recAnimals', {recAnimals});
    end
end

%% =====================================================================
function saveAllFormats(stem)
    exportgraphics(gcf, [stem '.png'], 'Resolution', 200);
    try, exportgraphics(gcf, [stem '.pdf'], 'ContentType', 'vector'); catch, end
    try, print(gcf, [stem '.svg'], '-dsvg'); catch, end
    fprintf('    saved: %s\n', [stem '.png']);
end

%% =====================================================================
function renderDeltaStimBaseline(condStimPerSim, condBasePerSim, conditions, ...
        condColors, condBias, sz, wn, outDir)
    perCondData = struct();
    for c = 1:numel(conditions)
        cond = conditions{c};
        if ~isfield(condStimPerSim, cond), continue; end
        perCondData.(cond) = condStimPerSim.(cond) - condBasePerSim.(cond);
    end
    renderSwarm(perCondData, conditions, condColors, condBias, ...
        '\Delta fraction active (stim - baseline)', ...
        'Fig 6c — Delta stim-baseline (model)', ...
        fullfile(outDir, 'Fig6c_DeltaStimBaseline'), false);
end

%% =====================================================================
function renderDeltaStimBaselineExp(expTraces, conditions, condColors, ...
        condBias, sz, wn, stimEndSec, outDir)
    % Per-recording delta (stim - baseline) for the experimental traces.
    %  Stim window:    [0, stimEndSec)         — matches model
    %  Baseline window:[-5, 0)                 — matches model
    perCondData = struct();
    for c = 1:numel(conditions)
        cond = conditions{c};
        if ~isfield(expTraces, cond) || isempty(expTraces.(cond).recTraces)
            continue;
        end
        et = expTraces.(cond).time(:)';
        rt = expTraces.(cond).recTraces;   % [nRec x nFrames]
        nU = min(numel(et), size(rt, 2));
        et = et(1:nU); rt = rt(:, 1:nU);
        stimMaskExp = et >= 0 & et < stimEndSec;
        baseMaskExp = et >= -5 & et < 0;
        if nnz(stimMaskExp) == 0 || nnz(baseMaskExp) == 0
            warning('exp %s: empty stim or baseline mask; skipping', cond);
            continue;
        end
        stimVals = mean(rt(:, stimMaskExp), 2, 'omitnan');
        baseVals = mean(rt(:, baseMaskExp), 2, 'omitnan');
        perCondData.(cond) = stimVals - baseVals;
    end
    if isempty(fieldnames(perCondData))
        warning('No experimental data available for delta plot; skipping');
        return;
    end
    renderSwarm(perCondData, conditions, condColors, condBias, ...
        '\Delta fraction active (stim - baseline)', ...
        'Fig 6c — Delta stim-baseline (experimental, per-rec)', ...
        fullfile(outDir, 'Fig6c_DeltaStimBaseline_Exp'), false);
end

%% =====================================================================
function renderDeltaStimBaselineExpPerAnimal(expTraces, conditions, condColors, ...
        condBias, sz, wn, stimEndSec, outDir)
    % Per-animal delta: group recordings by animal_name, average within
    % animal first, then plot one dot per animal.
    perCondData = struct();
    for c = 1:numel(conditions)
        cond = conditions{c};
        if ~isfield(expTraces, cond) || isempty(expTraces.(cond).recTraces)
            continue;
        end
        et = expTraces.(cond).time(:)';
        rt = expTraces.(cond).recTraces;
        animals = expTraces.(cond).recAnimals;
        if ~iscell(animals) || numel(animals) ~= size(rt, 1)
            warning('exp %s: missing per-rec animal names; skipping', cond); continue;
        end
        nU = min(numel(et), size(rt, 2));
        et = et(1:nU); rt = rt(:, 1:nU);
        stimMaskExp = et >= 0 & et < stimEndSec;
        baseMaskExp = et >= -5 & et < 0;
        if nnz(stimMaskExp) == 0 || nnz(baseMaskExp) == 0, continue; end
        % Per-rec delta
        recDelta = mean(rt(:, stimMaskExp), 2, 'omitnan') ...
                 - mean(rt(:, baseMaskExp), 2, 'omitnan');
        % Group by animal: average within animal
        [uniqAnimals, ~, animalIdx] = unique(animals);
        animalDelta = nan(numel(uniqAnimals), 1);
        for a = 1:numel(uniqAnimals)
            animalDelta(a) = mean(recDelta(animalIdx == a), 'omitnan');
        end
        perCondData.(cond) = animalDelta;
        fprintf('    exp per-animal %s: %d recs -> %d animals\n', ...
            cond, size(rt, 1), numel(uniqAnimals));
    end
    if isempty(fieldnames(perCondData))
        warning('No experimental data for per-animal delta plot; skipping');
        return;
    end
    renderSwarm(perCondData, conditions, condColors, condBias, ...
        '\Delta fraction active (stim - baseline)', ...
        'Fig 6c — Delta stim-baseline (experimental, per-animal)', ...
        fullfile(outDir, 'Fig6c_DeltaStimBaseline_Exp_PerAnimal'), false);
end

%% =====================================================================
function renderSwarm(perCondData, conditions, condColors, condBias, ...
        yLabelStr, titleStr, saveStem, drawUnityLine)
    if nargin < 8, drawUnityLine = false; end
    allVals = [];
    allConds = {};
    condOrder = {};
    for c = 1:numel(conditions)
        cond = conditions{c};
        if ~isfield(perCondData, cond), continue; end
        vals = perCondData.(cond)(:);
        vals = vals(isfinite(vals));
        if isempty(vals), continue; end
        allVals = [allVals; vals];
        allConds = [allConds; repmat({cond}, numel(vals), 1)];
        condOrder{end+1} = cond; %#ok<AGROW>
    end
    if isempty(allVals)
        warning('empty swarm dataset; skipping %s', saveStem);
        return;
    end

    fig = figure('Name', saveStem, 'Color', 'w');
    ax = gca; hold(ax, 'on');
    if drawUnityLine
        yline(ax, 1.0, 'k--', 'Alpha', 0.4, 'HandleVisibility', 'off');
    else
        yline(ax, 0.0, 'k--', 'Alpha', 0.4, 'HandleVisibility', 'off');
    end
    for ci = 1:numel(condOrder)
        m = strcmp(allConds, condOrder{ci});
        xVals = ci * ones(nnz(m), 1);
        col = condColors.(condOrder{ci});
        swarmchart(ax, xVals, allVals(m), 50, col, 'filled', ...
            'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', 'none', ...
            'XJitter', 'density', 'XJitterWidth', 0.45);
        medVal = median(allVals(m), 'omitnan');
        plot(ax, [ci - 0.25, ci + 0.25], [medVal, medVal], ...
            'k-', 'LineWidth', 2);
    end
    xLabelStrs = cell(1, numel(condOrder));
    for ci = 1:numel(condOrder)
        cb = NaN;
        if isfield(condBias, condOrder{ci}), cb = condBias.(condOrder{ci}); end
        xLabelStrs{ci} = sprintf('%s (b=%.2f)', condOrder{ci}, cb);
    end
    set(ax, 'XTick', 1:numel(condOrder), ...
            'XTickLabel', xLabelStrs, ...
            'XLim', [0.5, numel(condOrder) + 0.5], ...
            'FontSize', 10);
    ylabel(ax, yLabelStr, 'Interpreter', 'tex');
    title(ax, titleStr, 'Interpreter', 'none');
    grid(ax, 'on'); box(ax, 'on');
    annotatePairwise(ax, allVals, allConds, condOrder);
    hold(ax, 'off');
    set(fig, 'Position', [50, 50, 700, 500]);
    saveAllFormats(saveStem);
    close(fig);
end

%% =====================================================================
function annotatePairwise(ax, allVals, allConds, condOrder)
    nG = numel(condOrder);
    if nG < 2, return; end
    pairs = nchoosek(1:nG, 2);
    pvals = nan(size(pairs, 1), 1);
    for k = 1:size(pairs, 1)
        v1 = allVals(strcmp(allConds, condOrder{pairs(k, 1)}));
        v2 = allVals(strcmp(allConds, condOrder{pairs(k, 2)}));
        if numel(v1) >= 3 && numel(v2) >= 3
            try
                pvals(k) = ranksum(v1, v2);
            catch, end
        end
    end
    % Holm-Bonferroni
    [psort, ord] = sort(pvals);
    nValid = sum(~isnan(psort));
    pAdj = nan(size(psort));
    for k = 1:nValid
        pAdj(k) = min(1, psort(k) * (nValid - k + 1));
    end
    for k = 2:nValid, pAdj(k) = max(pAdj(k-1), pAdj(k)); end
    pvalsAdj = nan(size(pvals));
    pvalsAdj(ord) = pAdj;
    % Annotate
    yLim = get(ax, 'YLim');
    yMax = max(allVals);
    yRange = yLim(2) - yLim(1);
    if yRange <= 0, yRange = max(1e-6, yMax * 0.1); end
    yStep = 0.06 * yRange;
    yOffset = yMax + 0.04 * yRange;
    drawn = 0;
    for k = 1:size(pairs, 1)
        if isnan(pvalsAdj(k)) || pvalsAdj(k) >= 0.05, continue; end
        x1 = pairs(k, 1); x2 = pairs(k, 2);
        yLine = yOffset + drawn * yStep;
        plot(ax, [x1, x2], [yLine, yLine], 'k-', 'LineWidth', 1);
        text(ax, (x1+x2)/2, yLine + 0.015*yRange, pStars(pvalsAdj(k)), ...
            'HorizontalAlignment', 'center', 'FontSize', 11);
        drawn = drawn + 1;
    end
    if drawn > 0
        set(ax, 'YLim', [yLim(1), yOffset + drawn * yStep + 0.08 * yRange]);
    end
end

%% =====================================================================
function s = pStars(p)
    if p < 1e-4, s = '****';
    elseif p < 1e-3, s = '***';
    elseif p < 1e-2, s = '**';
    elseif p < 0.05, s = '*';
    else, s = 'n.s.'; end
end

%% =====================================================================
function renderDeltaStimBaselineMax(condStimMaxPerSim, condBasePerSim, conditions, ...
        condColors, condBias, sz, wn, outDir)
    % Per-sim max-stim minus mean-baseline (peak amplitude variant)
    perCondData = struct();
    for c = 1:numel(conditions)
        cond = conditions{c};
        if ~isfield(condStimMaxPerSim, cond), continue; end
        perCondData.(cond) = condStimMaxPerSim.(cond) - condBasePerSim.(cond);
    end
    renderSwarm(perCondData, conditions, condColors, condBias, ...
        '\Delta_{max} fraction active (max stim - mean baseline)', ...
        'Fig 6c — Max-Delta stim-baseline (model)', ...
        fullfile(outDir, 'Fig6c_DeltaStimBaselineMax'), false);
end

%% =====================================================================
function renderDeltaStimBaselineMaxExp(expTraces, conditions, condColors, ...
        condBias, sz, wn, stimEndSec, outDir)
    perCondData = struct();
    for c = 1:numel(conditions)
        cond = conditions{c};
        if ~isfield(expTraces, cond) || isempty(expTraces.(cond).recTraces), continue; end
        et = expTraces.(cond).time(:)';
        rt = expTraces.(cond).recTraces;
        nU = min(numel(et), size(rt, 2));
        et = et(1:nU); rt = rt(:, 1:nU);
        stimMaskExp = et >= 0 & et < stimEndSec;
        baseMaskExp = et >= -5 & et < 0;
        if nnz(stimMaskExp) == 0 || nnz(baseMaskExp) == 0, continue; end
        stimMax = max(rt(:, stimMaskExp), [], 2, 'omitnan');
        baseMean = mean(rt(:, baseMaskExp), 2, 'omitnan');
        perCondData.(cond) = stimMax - baseMean;
    end
    if isempty(fieldnames(perCondData))
        warning('No exp data for max-delta plot; skipping'); return;
    end
    renderSwarm(perCondData, conditions, condColors, condBias, ...
        '\Delta_{max} fraction active (max stim - mean baseline)', ...
        'Fig 6c — Max-Delta stim-baseline (experimental, per-rec)', ...
        fullfile(outDir, 'Fig6c_DeltaStimBaselineMax_Exp'), false);
end

%% =====================================================================
function renderDeltaStimBaselineMaxExpPerAnimal(expTraces, conditions, condColors, ...
        condBias, sz, wn, stimEndSec, outDir)
    perCondData = struct();
    for c = 1:numel(conditions)
        cond = conditions{c};
        if ~isfield(expTraces, cond) || isempty(expTraces.(cond).recTraces), continue; end
        et = expTraces.(cond).time(:)';
        rt = expTraces.(cond).recTraces;
        animals = expTraces.(cond).recAnimals;
        if ~iscell(animals) || numel(animals) ~= size(rt, 1)
            warning('exp %s: missing per-rec animal names; skipping', cond); continue;
        end
        nU = min(numel(et), size(rt, 2));
        et = et(1:nU); rt = rt(:, 1:nU);
        stimMaskExp = et >= 0 & et < stimEndSec;
        baseMaskExp = et >= -5 & et < 0;
        if nnz(stimMaskExp) == 0 || nnz(baseMaskExp) == 0, continue; end
        recDelta = max(rt(:, stimMaskExp), [], 2, 'omitnan') ...
                 - mean(rt(:, baseMaskExp), 2, 'omitnan');
        [uniqAnimals, ~, animalIdx] = unique(animals);
        animalDelta = nan(numel(uniqAnimals), 1);
        for a = 1:numel(uniqAnimals)
            animalDelta(a) = mean(recDelta(animalIdx == a), 'omitnan');
        end
        perCondData.(cond) = animalDelta;
        fprintf('    exp per-animal MAX %s: %d recs -> %d animals\n', ...
            cond, size(rt, 1), numel(uniqAnimals));
    end
    if isempty(fieldnames(perCondData))
        warning('No exp data for per-animal max-delta plot; skipping'); return;
    end
    renderSwarm(perCondData, conditions, condColors, condBias, ...
        '\Delta_{max} fraction active (max stim - mean baseline)', ...
        'Fig 6c — Max-Delta stim-baseline (experimental, per-animal)', ...
        fullfile(outDir, 'Fig6c_DeltaStimBaselineMax_Exp_PerAnimal'), false);
end

%% =====================================================================
function renderAllSimsNoExp(rawTraces, conditions, condColors, condBias, ...
        simTimeAxis, stimEndSec, dispWin, sz, wn, outDirParent)
    %RENDERALLSIMSNOEXP  AllSims-style 10x4 grid + 10 per-row figures
    %   without experimental overlay. Also copies the existing
    %   Fig6c_AllSims.{png,pdf,svg} into the same subfolder.
    %
    %   rawTraces: [4 conds x nSims x nReps x nFrames]
    nC = numel(conditions);
    nSims = size(rawTraces, 2);

    % Subfolder
    subDir = fullfile(outDirParent, 'Fig6c_AllSims');
    if ~exist(subDir, 'dir'), mkdir(subDir); end

    % Copy existing originals if present
    for ext = {'.png', '.pdf', '.svg', '.fig'}
        src = fullfile(outDirParent, ['Fig6c_AllSims' ext{1}]);
        if exist(src, 'file')
            copyfile(src, fullfile(subDir, ['Fig6c_AllSims_WithExp' ext{1}]));
        end
    end

    dispMask = simTimeAxis >= dispWin(1) & simTimeAxis <= dispWin(2);
    tWin = simTimeAxis(dispMask);

    % Precompute per (cond, sim) mean + SEM over raw reps
    muAll = nan(nC, nSims, nnz(dispMask));
    semAll = nan(nC, nSims, nnz(dispMask));
    yMaxGlobal = 0;
    for c = 1:nC
        for s = 1:nSims
            reps = squeeze(rawTraces(c, s, :, :));      % [nReps x nFrames]
            if ndims(reps) > 2, continue; end
            reps = reps(:, dispMask);
            mu = mean(reps, 1, 'omitnan');
            sem = std(reps, 0, 1, 'omitnan') / sqrt(sum(any(isfinite(reps), 2)));
            muAll(c, s, :) = mu;
            semAll(c, s, :) = sem;
            yMaxGlobal = max(yMaxGlobal, max(mu + sem));
        end
    end
    yTop = max(yMaxGlobal * 1.1, 0.05);

    % --- 10x4 combined figure (no exp overlay) ---
    figName = 'Fig6c_AllSims_Model';
    fig = figure('Name', figName, 'Color', 'w');
    tl = tiledlayout(nSims, nC, 'TileSpacing', 'compact', 'Padding', 'compact');
    for s = 1:nSims
        for c = 1:nC
            cond = conditions{c};
            ax = nexttile;
            hold(ax, 'on');
            col = condColors.(cond);
            mu  = squeeze(muAll(c, s, :))';
            sem = squeeze(semAll(c, s, :))';
            if all(isnan(mu))
                text(ax, 0.5, 0.5, '(no data)', 'Units', 'normalized', ...
                    'HorizontalAlignment', 'center', 'FontSize', 7);
            else
                fill(ax, [tWin, fliplr(tWin)], [mu+sem, fliplr(mu-sem)], ...
                    col, 'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
                    'HandleVisibility', 'off');
                plot(ax, tWin, mu, 'Color', col, 'LineWidth', 1.5);
            end
            xline(ax, 0, 'k:', 'LineWidth', 0.5, 'HandleVisibility', 'off');
            xline(ax, stimEndSec, 'k:', 'LineWidth', 0.5, 'HandleVisibility', 'off');
            xlim(ax, dispWin); ylim(ax, [0, yTop]);
            grid(ax, 'on'); box(ax, 'on');
            set(ax, 'FontSize', 7);
            if c == 1
                ylabel(ax, sprintf('sim %d', s-1), 'FontSize', 8);
            end
            if s == 1
                cb = NaN;
                if isfield(condBias, cond), cb = condBias.(cond); end
                title(ax, sprintf('%s (b=%.2f)', cond, cb), ...
                    'Color', col, 'FontSize', 9);
            end
            if s == nSims
                xlabel(ax, 'Time (s)', 'FontSize', 8);
            else
                set(ax, 'XTickLabel', []);
            end
            hold(ax, 'off');
        end
    end
    title(tl, 'Fig 6c — all sims', ...
        'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');
    set(fig, 'Position', [50, 50, 360*nC, 200*nSims]);
    saveAllFormats(fullfile(subDir, figName));
    close(fig);

    % --- Per-sim row figures (one figure per sim, 1x4 layout) ---
    for s = 1:nSims
        figName = sprintf('Fig6c_AllSims_Model_sim%d', s-1);
        fig = figure('Name', figName, 'Color', 'w');
        tl = tiledlayout(1, nC, 'TileSpacing', 'compact', 'Padding', 'compact');
        for c = 1:nC
            cond = conditions{c};
            ax = nexttile;
            hold(ax, 'on');
            col = condColors.(cond);
            mu  = squeeze(muAll(c, s, :))';
            sem = squeeze(semAll(c, s, :))';
            if all(isnan(mu))
                text(ax, 0.5, 0.5, '(no data)', 'Units', 'normalized', ...
                    'HorizontalAlignment', 'center');
            else
                fill(ax, [tWin, fliplr(tWin)], [mu+sem, fliplr(mu-sem)], ...
                    col, 'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
                    'HandleVisibility', 'off');
                plot(ax, tWin, mu, 'Color', col, 'LineWidth', 1.8);
            end
            xline(ax, 0, 'k:', 'LineWidth', 0.5, 'HandleVisibility', 'off');
            xline(ax, stimEndSec, 'k:', 'LineWidth', 0.5, 'HandleVisibility', 'off');
            xlim(ax, dispWin); ylim(ax, [0, yTop]);
            grid(ax, 'on'); box(ax, 'on');
            cb = NaN;
            if isfield(condBias, cond), cb = condBias.(cond); end
            title(ax, sprintf('%s (b=%.2f)', cond, cb), ...
                'Color', col, 'FontSize', 11);
            xlabel(ax, 'Time (s)');
            if c == 1, ylabel(ax, sprintf('sim %d activity', s-1)); end
        end
        title(tl, sprintf('Fig 6c — sim %d', s-1), ...
            'FontSize', 12, 'Interpreter', 'none');
        set(fig, 'Position', [50, 50, 300*nC, 400]);
        saveAllFormats(fullfile(subDir, figName));
        close(fig);
    end
end
