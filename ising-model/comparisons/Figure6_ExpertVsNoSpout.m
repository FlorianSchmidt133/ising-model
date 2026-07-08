function [simRasterFB, selFB, extremeRepsOut, extremeSimOut] = Figure6_ExpertVsNoSpout( ...
    expTraces, simRaster, sel, sz, wn, outDirSW, ...
    matcherOut, condColors, opts, simTimeAxis, stimEndSec, ...
    biasFile, clampedFile, chosenDur, nSimsTotal, varargin)
% Figure6_ExpertVsNoSpout
%   2x3 figure: trace tiles + boxplot quantification tiles.
%
%   Per-cond rendering: TargetConds controls how many cond columns/boxes
%   are shown. Default is {Expert, NoSpout} (2-cond mode, 4 boxes per
%   quant tile). Passing all 4 conds gives a 4-cond mode with 8 boxes per
%   quant tile and full pairwise brackets.
%
%   Optional name-value args (varargin):
%     'TargetConds'               cell array of cond names (default {'Expert','NoSpout'})
%     'NormalizeToNaiveBaseline'  logical (default false). When true, metric
%                                 baseline reference is Naive's baseline mean
%                                 (per-source: Naive data baseline for data;
%                                 Naive model baseline for model) rather than
%                                 each row's own baseline.
%     'SaveTiles'                 logical (default false). When true, also
%                                 save each of the 6 tiles as a standalone
%                                 image in outDirSW/<OutputStem>_tiles/.
%     'OutputStem'                file stem (default 'Fig6_ExpertVsNoSpout').
%
%   Composite figure file: outDirSW/<OutputStem>.{png,pdf,svg}
%   Stats CSV: outDirSW/<OutputStem>_stats.csv
%   Tiles dir (if SaveTiles): outDirSW/<OutputStem>_tiles/*.png

    % --- Parse name-value args ------------------------------------------
    p = inputParser; p.KeepUnmatched = true;
    addParameter(p, 'TargetConds',              {'Expert', 'NoSpout'}, @iscell);
    addParameter(p, 'NormalizeToNaiveBaseline', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'SaveTiles',                false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'OutputStem',               'Fig6_ExpertVsNoSpout', @ischar);
    addParameter(p, 'BetaIsolationMode',        false, @(x) islogical(x) || isnumeric(x));
    parse(p, varargin{:});
    userOpts = p.Results;
    targetConds  = userOpts.TargetConds;
    saveTiles    = logical(userOpts.SaveTiles);
    normToNaive  = logical(userOpts.NormalizeToNaiveBaseline);
    outputStem   = userOpts.OutputStem;
    betaIso      = logical(userOpts.BetaIsolationMode);
    nC           = numel(targetConds);

    % In BetaIsolationMode, rewrite the default outputStem so the new figure
    % does not overwrite the unconstrained version. If the caller already
    % passed a non-default stem, leave it alone.
    if betaIso && strcmp(outputStem, 'Fig6_ExpertVsNoSpout')
        outputStem = 'Fig6_ExpertVsNoSpout_BetaIsolated';
    end

    % --- Default outputs (in case of early return) ----------------------
    simRasterFB     = simRaster;
    selFB           = sel;
    extremeRepsOut  = struct();
    extremeSimOut   = struct();

    % --- Presence check -------------------------------------------------
    missing = {};
    for ci = 1:nC
        cond = targetConds{ci};
        if ~(isfield(expTraces, cond) && isfield(sel, cond) && isfield(simRaster, cond))
            missing{end+1} = cond; %#ok<AGROW>
        end
    end
    if ~isempty(missing)
        warning('Figure6_ExpertVsNoSpout: conds [%s] missing for size=%d window=%s stem=%s; skipping', ...
            strjoin(missing, ','), sz, wn, outputStem);
        return;
    end
    if normToNaive && ~ismember('Naive', targetConds) ...
            && ~(isfield(expTraces, 'Naive') && isfield(simRaster, 'Naive'))
        warning('Figure6_ExpertVsNoSpout: NormalizeToNaiveBaseline=true but Naive not available; skipping');
        return;
    end

    if nargin < 15 || isempty(nSimsTotal), nSimsTotal = 10; end

    % --- Window parameters ----------------------------------------------
    onsetWin    = [0, 0.3];
    offsetWin   = [0, 0.4];
    baselineWin = [-1, -0.1];
    if isfield(matcherOut, 'opts')
        if isfield(matcherOut.opts, 'OnsetWindow'),        onsetWin    = matcherOut.opts.OnsetWindow;        end
        if isfield(matcherOut.opts, 'OffsetWindow'),       offsetWin   = matcherOut.opts.OffsetWindow;       end
        if isfield(matcherOut.opts, 'FoldBaselineWindow'), baselineWin = matcherOut.opts.FoldBaselineWindow; end
    end

    % --- Time alignment from matcher ------------------------------------
    timeStretch = ones(numel(opts.Conditions), 1);
    timeShift   = zeros(numel(opts.Conditions), 1);
    if isfield(matcherOut, 'sizeOut')
        if isfield(matcherOut.sizeOut, 'timeStretch'), timeStretch = matcherOut.sizeOut.timeStretch; end
        if isfield(matcherOut.sizeOut, 'timeShift'),   timeShift   = matcherOut.sizeOut.timeShift;   end
    end

    % --- Naive baseline scalars (only used in normToNaive branch) -------
    naiveBaseData  = NaN;
    naiveBaseModel = NaN;
    if normToNaive && isfield(expTraces, 'Naive') && isfield(simRaster, 'Naive')
        tD_naive = expTraces.Naive.time(:)';
        bMaskD = tD_naive >= baselineWin(1) & tD_naive <= baselineWin(2);
        if any(bMaskD)
            tmp = expTraces.Naive.recTraces(:, bMaskD);
            naiveBaseData = mean(tmp(:), 'omitnan');
        end
        cIdxNaive = find(strcmp(opts.Conditions, 'Naive'), 1);
        if isempty(cIdxNaive)
            tM_naive = simTimeAxis(:)';
        else
            tM_naive = simTimeAxis(:)' * timeStretch(cIdxNaive) - timeShift(cIdxNaive);
        end
        bMaskM = tM_naive >= baselineWin(1) & tM_naive <= baselineWin(2);
        if any(bMaskM)
            tmp = simRaster.Naive(:, bMaskM);
            naiveBaseModel = mean(tmp(:), 'omitnan');
        end
    end

    % --- Per-cond DATA fold (target for fold-best sim picking; uses cond's own baseline) ---
    foldData = struct();
    for ci = 1:nC
        cond = targetConds{ci};
        recT = expTraces.(cond).recTraces;
        tD   = expTraces.(cond).time(:)';
        foldData.(cond) = computeFoldPerRow(recT, tD, baselineWin, onsetWin, offsetWin, stimEndSec);
    end

    % --- BetaIsolationMode override -------------------------------------
    %  Force the (Expert, NoSpout) sims to the matched-except-beta pair from
    %  the top-10 lists, BEFORE the fold-best / extreme-Δ search loops.
    %  If no such pair exists, exit early with a warning.
    betaPair = struct();    % populated only in iso mode
    if betaIso
        betaPair = findMatchedBetaPair(biasFile, opts.Mode, nSimsTotal, targetConds);
        if isempty(fieldnames(betaPair))
            warning(['Figure6_ExpertVsNoSpout: BetaIsolationMode requested but no ', ...
                     'matched-except-beta pair found in top-10; skipping.']);
            return;
        end
        fprintf('    [%s] forced pair (matched except beta):\n', outputStem);
        for ci = 1:nC
            cond = targetConds{ci};
            if ~isfield(betaPair, cond), continue; end
            cIdx = find(strcmp(opts.Conditions, cond), 1);
            if isnan(cIdx), continue; end
            sIdx = betaPair.(cond).simIdx;
            bv   = sel.(cond).bias;
            try
                reps = loadActivityCropReps(biasFile, clampedFile, ...
                    cond, sIdx, bv, opts.Mode, opts.ClampedMode, sz, chosenDur);
            catch ME
                warning('Figure6_ExpertVsNoSpout: failed to load forced sim %d for %s: %s', ...
                        sIdx, cond, ME.message);
                return;
            end
            if isempty(reps)
                warning('Figure6_ExpertVsNoSpout: empty reps for forced sim %d (%s)', sIdx, cond);
                return;
            end
            simRaster.(cond) = reps;
            sel.(cond).sim   = sIdx + 1;
            fprintf('      %s: sim_%d (beta=%.4f, bias=%.4f)\n', ...
                    cond, sIdx, betaPair.(cond).beta, bv);
        end
        if isfield(betaPair, 'shared')
            fprintf('      shared: c=%g, decay=%g, rad=%g, h_b=%g\n', ...
                    betaPair.shared.c, betaPair.shared.decay_const, ...
                    betaPair.shared.inhibition_range, betaPair.shared.bias);
        end
    end

    % --- FOLD-BEST sim per cond -----------------------------------------
    foldBestSim = struct();
    for ci = 1:nC
        cond  = targetConds{ci};
        cIdx  = find(strcmp(opts.Conditions, cond), 1);
        if isnan(cIdx), continue; end
        targetFold = mean(foldData.(cond), 'omitnan');
        if ~isfinite(targetFold), continue; end
        bv = sel.(cond).bias;
        if betaIso
            % Skip the search; record the forced pick's stats for reporting.
            if ~isfield(betaPair, cond), continue; end
            reps = simRaster.(cond);
            if isempty(reps), continue; end
            tM = simTimeAxis(:)' * timeStretch(cIdx) - timeShift(cIdx);
            mu = mean(computeFoldPerRow(reps, tM, baselineWin, onsetWin, offsetWin, stimEndSec), 'omitnan');
            foldBestSim.(cond) = struct('simIdx', betaPair.(cond).simIdx, ...
                'meanFold', mu, 'targetFold', targetFold, 'err', abs(mu - targetFold));
            continue;
        end
        bestErr = Inf; bestSim = NaN; bestReps = []; bestMean = NaN;
        for sIdx = 0:(nSimsTotal-1)
            try
                reps = loadActivityCropReps(biasFile, clampedFile, ...
                    cond, sIdx, bv, opts.Mode, opts.ClampedMode, sz, chosenDur);
            catch, continue; end
            if isempty(reps), continue; end
            tM = simTimeAxis(:)' * timeStretch(cIdx) - timeShift(cIdx);
            mu = mean(computeFoldPerRow(reps, tM, baselineWin, onsetWin, offsetWin, stimEndSec), 'omitnan');
            if ~isfinite(mu), continue; end
            err = abs(mu - targetFold);
            if err < bestErr
                bestErr = err; bestSim = sIdx; bestReps = reps; bestMean = mu;
            end
        end
        if isnan(bestSim), continue; end
        fprintf('    [%s] fold-best sim for %s: sim_%d (model=%.3f, target=%.3f)\n', ...
            outputStem, cond, bestSim, bestMean, targetFold);
        simRaster.(cond)   = bestReps;
        sel.(cond).sim     = bestSim + 1;
        foldBestSim.(cond) = struct('simIdx', bestSim, 'meanFold', bestMean, ...
            'targetFold', targetFold, 'err', bestErr);
    end

    % --- EXTREME-Δ sim per cond -----------------------------------------
    %  Active/responsive conds (Beginner, Expert) use argmax (strongest);
    %  baseline-like conds (Naive, NoSpout) use argmin (weakest available).
    %  In BetaIsolationMode, the extreme-Δ branch reuses the forced pair
    %  (no per-cond argmax/argmin search), so the right-column tile and
    %  boxplot branch reflect the same beta-isolated pair.
    extremeSim  = struct();
    extremeReps = struct();
    argmaxConds = {'Beginner', 'Expert'};
    for ci = 1:nC
        cond = targetConds{ci};
        cIdx = find(strcmp(opts.Conditions, cond), 1);
        if isnan(cIdx), continue; end
        if betaIso
            if ~isfield(betaPair, cond), continue; end
            tag = 'forced';
            extremeSim.(cond)  = struct('simIdx', betaPair.(cond).simIdx, ...
                                        'delta', NaN, 'direction', tag);
            extremeReps.(cond) = simRaster.(cond);
            continue;
        end
        pickArgmax = ismember(cond, argmaxConds);
        bestVal = -Inf; if ~pickArgmax, bestVal = Inf; end
        bestSim = NaN; bestRepsOut = [];
        bv = sel.(cond).bias;
        for sIdx = 0:(nSimsTotal-1)
            try
                reps = loadActivityCropReps(biasFile, clampedFile, ...
                    cond, sIdx, bv, opts.Mode, opts.ClampedMode, sz, chosenDur);
            catch, continue; end
            if isempty(reps), continue; end
            tM = simTimeAxis(:)' * timeStretch(cIdx) - timeShift(cIdx);
            d = computeMetricPerRow(reps, tM, 'delta_base', baselineWin, onsetWin, offsetWin, stimEndSec, NaN);
            mu = mean(d, 'omitnan');
            if ~isfinite(mu), continue; end
            better = (pickArgmax && mu > bestVal) || (~pickArgmax && mu < bestVal);
            if better
                bestVal = mu; bestSim = sIdx; bestRepsOut = reps;
            end
        end
        if isnan(bestSim), continue; end
        tag = 'argmin'; if pickArgmax, tag = 'argmax'; end
        fprintf('    [%s] extreme-Δ sim for %s (%s): sim_%d (Δ=%.4f)\n', outputStem, cond, tag, bestSim, bestVal);
        extremeSim.(cond)  = struct('simIdx', bestSim, 'delta', bestVal, 'direction', tag);
        extremeReps.(cond) = bestRepsOut;
    end

    % --- Compute all per-cond traces + metrics (post-selection) ---------
    foldModel        = struct();  stimMeanData     = struct();  stimMeanModel    = struct();
    deltaBaseData    = struct();  deltaBaseModel   = struct();
    foldModelXT      = struct();  stimMeanModelXT  = struct();  deltaBaseModelXT = struct();
    traceData        = struct();  traceModel       = struct();  traceExtreme     = struct();
    for ci = 1:nC
        cond = targetConds{ci};
        cIdx = find(strcmp(opts.Conditions, cond), 1);

        % DATA
        recT = expTraces.(cond).recTraces;
        tD   = expTraces.(cond).time(:)';
        traceData.(cond).t  = tD;
        traceData.(cond).mu = mean(recT, 1, 'omitnan');
        traceData.(cond).sd = std(recT,  0, 1, 'omitnan');
        baseDataOverride  = NaN; if normToNaive, baseDataOverride  = naiveBaseData;  end
        foldData.(cond)         = computeFoldPerRow(  recT, tD, baselineWin, onsetWin, offsetWin, stimEndSec, baseDataOverride);
        stimMeanData.(cond)     = computeMetricPerRow(recT, tD, 'stim_mean',  baselineWin, onsetWin, offsetWin, stimEndSec, baseDataOverride);
        deltaBaseData.(cond)    = computeMetricPerRow(recT, tD, 'delta_base', baselineWin, onsetWin, offsetWin, stimEndSec, baseDataOverride);

        % MODEL — FOLD-BEST sim (in simRaster after the selection above)
        reps = simRaster.(cond);
        if isnan(cIdx)
            tM = simTimeAxis(:)';
        else
            tM = simTimeAxis(:)' * timeStretch(cIdx) - timeShift(cIdx);
        end
        traceModel.(cond).t  = tM;
        traceModel.(cond).mu = mean(reps, 1, 'omitnan');
        traceModel.(cond).sd = std(reps,  0, 1, 'omitnan');
        baseModelOverride = NaN; if normToNaive, baseModelOverride = naiveBaseModel; end
        foldModel.(cond)        = computeFoldPerRow(  reps, tM, baselineWin, onsetWin, offsetWin, stimEndSec, baseModelOverride);
        stimMeanModel.(cond)    = computeMetricPerRow(reps, tM, 'stim_mean',  baselineWin, onsetWin, offsetWin, stimEndSec, baseModelOverride);
        deltaBaseModel.(cond)   = computeMetricPerRow(reps, tM, 'delta_base', baselineWin, onsetWin, offsetWin, stimEndSec, baseModelOverride);

        % MODEL — EXTREME-Δ sim (in extremeReps; fall back to fold-best if absent)
        if isfield(extremeReps, cond) && ~isempty(extremeReps.(cond))
            repsX = extremeReps.(cond);
            traceExtreme.(cond).t  = tM;
            traceExtreme.(cond).mu = mean(repsX, 1, 'omitnan');
            traceExtreme.(cond).sd = std(repsX,  0, 1, 'omitnan');
            foldModelXT.(cond)      = computeFoldPerRow(  repsX, tM, baselineWin, onsetWin, offsetWin, stimEndSec, baseModelOverride);
            stimMeanModelXT.(cond)  = computeMetricPerRow(repsX, tM, 'stim_mean',  baselineWin, onsetWin, offsetWin, stimEndSec, baseModelOverride);
            deltaBaseModelXT.(cond) = computeMetricPerRow(repsX, tM, 'delta_base', baselineWin, onsetWin, offsetWin, stimEndSec, baseModelOverride);
        else
            traceExtreme.(cond)       = traceModel.(cond);
            foldModelXT.(cond)        = foldModel.(cond);
            stimMeanModelXT.(cond)    = stimMeanModel.(cond);
            deltaBaseModelXT.(cond)   = deltaBaseModel.(cond);
        end
    end

    % --- Shared Y across all three trace panels -------------------------
    yMaxTrace = 0;
    for ci = 1:nC
        cond = targetConds{ci};
        yMaxTrace = max([yMaxTrace, ...
            max(traceData.(cond).mu    + traceData.(cond).sd,    [], 'omitnan'), ...
            max(traceModel.(cond).mu   + traceModel.(cond).sd,   [], 'omitnan'), ...
            max(traceExtreme.(cond).mu + traceExtreme.(cond).sd, [], 'omitnan')]);
    end
    if ~isfinite(yMaxTrace) || yMaxTrace <= 0, yMaxTrace = 0.05; end
    yLimTraces = [0, yMaxTrace * 1.10];

    % --- Build per-tile metadata for the composite + tile-saving --------
    %  Two metric struct arrays:
    %    metricsFB: model side from FOLD-BEST sims (matches tile 2 traces)
    %    metricsXT: model side from EXTREME-Δ sims (matches tile 3 traces)
    titleFold     = ifelse(normToNaive, 'Fold quant (Naive-baseline-normalized)', 'Fold quantification');
    ylabFold      = ifelse(normToNaive, 'Peak / Naive baseline', 'Fold change (peak / baseline)');
    titleStim     = ifelse(normToNaive, 'Stim mean (Naive-baseline-subtracted)', 'Stim-period mean');
    ylabStim      = ifelse(normToNaive, 'Stim mean − Naive baseline', 'Stim-period mean (fraction active)');
    titleDelta    = ifelse(normToNaive, '\Delta to Naive baseline', '\Delta to baseline');
    ylabDelta     = ifelse(normToNaive, 'Peak − Naive baseline', '\Delta to baseline (peak − baseline)');
    yrefStim      = ifelse(normToNaive, 0, NaN);

    metricsFB(1).name = 'fold';       metricsFB(1).ttl = titleFold;  metricsFB(1).ylab = ylabFold;  metricsFB(1).yref = 1;
    metricsFB(1).data = collectGroupVals(targetConds, foldData, foldModel);
    metricsFB(2).name = 'stim_mean';  metricsFB(2).ttl = titleStim;  metricsFB(2).ylab = ylabStim;  metricsFB(2).yref = yrefStim;
    metricsFB(2).data = collectGroupVals(targetConds, stimMeanData, stimMeanModel);
    metricsFB(3).name = 'delta_base'; metricsFB(3).ttl = titleDelta; metricsFB(3).ylab = ylabDelta; metricsFB(3).yref = 0;
    metricsFB(3).data = collectGroupVals(targetConds, deltaBaseData, deltaBaseModel);

    metricsXT(1).name = 'fold';       metricsXT(1).ttl = titleFold;  metricsXT(1).ylab = ylabFold;  metricsXT(1).yref = 1;
    metricsXT(1).data = collectGroupVals(targetConds, foldData, foldModelXT);
    metricsXT(2).name = 'stim_mean';  metricsXT(2).ttl = titleStim;  metricsXT(2).ylab = ylabStim;  metricsXT(2).yref = yrefStim;
    metricsXT(2).data = collectGroupVals(targetConds, stimMeanData, stimMeanModelXT);
    metricsXT(3).name = 'delta_base'; metricsXT(3).ttl = titleDelta; metricsXT(3).ylab = ylabDelta; metricsXT(3).yref = 0;
    metricsXT(3).data = collectGroupVals(targetConds, deltaBaseData, deltaBaseModelXT);

    % --- Traces row (single figure, shows both sim selections) ----------
    sumTitle = buildSummaryTitle(outputStem, sz, wn, sel, targetConds, foldBestSim, extremeSim, normToNaive);
    if betaIso && isfield(betaPair, 'shared')
        betaParts = cell(1, nC);
        for ci = 1:nC
            cond = targetConds{ci};
            if isfield(betaPair, cond)
                betaParts{ci} = sprintf('\\beta_{%s}=%.3f', cond(1:min(3,end)), betaPair.(cond).beta);
            end
        end
        betaParts = betaParts(~cellfun('isempty', betaParts));
        sumTitle = sprintf('%s\nbeta-isolated: %s | shared: c=%g, decay=%g, rad=%g, h_b=%g', ...
            sumTitle, strjoin(betaParts, ', '), ...
            betaPair.shared.c, betaPair.shared.decay_const, ...
            betaPair.shared.inhibition_range, betaPair.shared.bias);
    end

    f1 = figure('Name', [outputStem '_traces'], 'Color', 'w');
    tl1 = tiledlayout(f1, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax_data    = nexttile(tl1, 1); renderTraceTile(ax_data,    targetConds, traceData,    condColors, '-',  stimEndSec, opts.DisplayWindow, yLimTraces, 'Data (mean ± std across recs)', sel, false);
    ax_modelFB = nexttile(tl1, 2); renderTraceTile(ax_modelFB, targetConds, traceModel,   condColors, '--', stimEndSec, opts.DisplayWindow, yLimTraces, 'Model — fold-best sim (mean ± std)', sel, true);
    ax_modelXT = nexttile(tl1, 3); renderTraceTile(ax_modelXT, targetConds, traceExtreme, condColors, '--', stimEndSec, opts.DisplayWindow, yLimTraces, 'Model — extreme-Δ sims', sel, true, extremeSim);
    title(tl1, sumTitle, 'FontWeight', 'normal');
    set(f1, 'Position', [50, 50, 1500, 450]);
    saveStem1 = fullfile(outDirSW, [outputStem '_traces']);
    exportgraphics(f1, [saveStem1 '.png'], 'Resolution', 250);
    try, exportgraphics(f1, [saveStem1 '.pdf'], 'ContentType', 'vector'); catch, end
    try, print(f1, [saveStem1 '.svg'], '-dsvg'); catch, end
    close(f1);
    fprintf('    saved: %s.png\n', saveStem1);

    % --- Boxplots row, rendered TWICE (one per sim-selection strategy) ---
    %   Each goes to its own subfolder, with its own stats CSV and tile splits.
    pTable = struct();
    selBranches = struct('foldbest', metricsFB, 'extremedelta', metricsXT);
    branchNames = fieldnames(selBranches);
    for bi = 1:numel(branchNames)
        branch = branchNames{bi};
        ms = selBranches.(branch);
        branchDir = fullfile(outDirSW, [outputStem '_' branch]);
        if ~exist(branchDir, 'dir'), mkdir(branchDir); end

        f2 = figure('Name', [outputStem '_boxplots_' branch], 'Color', 'w');
        tl2 = tiledlayout(f2, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        pBranch = struct();
        for mi = 1:numel(ms)
            ax = nexttile(tl2, mi);
            pBranch.(ms(mi).name) = renderBoxTile(ax, targetConds, ms(mi), condColors);
        end
        title(tl2, sprintf('%s | sim-selection: %s', sumTitle, branch), 'FontWeight', 'normal');
        set(f2, 'Position', [50, 50, 1500, 500]);
        saveStem2 = fullfile(branchDir, [outputStem '_boxplots']);
        exportgraphics(f2, [saveStem2 '.png'], 'Resolution', 250);
        try, exportgraphics(f2, [saveStem2 '.pdf'], 'ContentType', 'vector'); catch, end
        try, print(f2, [saveStem2 '.svg'], '-dsvg'); catch, end
        close(f2);
        fprintf('    saved: %s.png\n', saveStem2);

        pTable.(branch) = pBranch;

        % Per-branch boxplot tile splits
        if saveTiles
            tileDirB = fullfile(branchDir, [outputStem '_tiles_boxplots']);
            if ~exist(tileDirB, 'dir'), mkdir(tileDirB); end
            tileSpecsB = { ...
                'tile_4_fold',        @(ax) renderBoxTile(ax, targetConds, ms(1), condColors); ...
                'tile_5_stim_mean',   @(ax) renderBoxTile(ax, targetConds, ms(2), condColors); ...
                'tile_6_delta_base',  @(ax) renderBoxTile(ax, targetConds, ms(3), condColors)};
            for ti = 1:size(tileSpecsB, 1)
                tname = tileSpecsB{ti, 1};
                renderFn = tileSpecsB{ti, 2};
                tf = figure('Name', tname, 'Color', 'w', 'Position', [50, 50, 700, 500]);
                ax = axes(tf);
                renderFn(ax);
                ts = fullfile(tileDirB, tname);
                exportgraphics(tf, [ts '.png'], 'Resolution', 250);
                try, exportgraphics(tf, [ts '.pdf'], 'ContentType', 'vector'); catch, end
                try, print(tf, [ts '.svg'], '-dsvg'); catch, end
                close(tf);
            end
        end

        % Per-branch stats CSV
        writeStatsCsv(branchDir, outputStem, targetConds, ms, pBranch, sel, ...
            foldBestSim, extremeSim, baselineWin, onsetWin, offsetWin, stimEndSec, ...
            normToNaive, naiveBaseData, naiveBaseModel, branch);
    end

    % --- Traces tile splits (single set, top-level) ---------------------
    if saveTiles
        tileDir = fullfile(outDirSW, [outputStem '_tiles']);
        if ~exist(tileDir, 'dir'), mkdir(tileDir); end
        tileSpecs = {
            'tile_1_data_traces',           @(ax) renderTraceTile(ax, targetConds, traceData,    condColors, '-',  stimEndSec, opts.DisplayWindow, yLimTraces, 'Data (mean ± std across recs)', sel, false);
            'tile_2_model_foldbest',        @(ax) renderTraceTile(ax, targetConds, traceModel,   condColors, '--', stimEndSec, opts.DisplayWindow, yLimTraces, 'Model — fold-best sim',         sel, true);
            'tile_3_model_extremedelta',    @(ax) renderTraceTile(ax, targetConds, traceExtreme, condColors, '--', stimEndSec, opts.DisplayWindow, yLimTraces, 'Model — extreme-Δ sims',  sel, true, extremeSim);
        };
        for ti = 1:size(tileSpecs, 1)
            tname = tileSpecs{ti, 1};
            renderFn = tileSpecs{ti, 2};
            tf = figure('Name', tname, 'Color', 'w', 'Position', [50, 50, 700, 500]);
            ax = axes(tf);
            renderFn(ax);
            ts = fullfile(tileDir, tname);
            exportgraphics(tf, [ts '.png'], 'Resolution', 250);
            try, exportgraphics(tf, [ts '.pdf'], 'ContentType', 'vector'); catch, end
            try, print(tf, [ts '.svg'], '-dsvg'); catch, end
            close(tf);
        end
        fprintf('    saved traces tiles: %s/*.png\n', tileDir);
    end

    % --- Returns --------------------------------------------------------
    simRasterFB    = simRaster;
    selFB          = sel;
    extremeRepsOut = extremeReps;
    extremeSimOut  = extremeSim;
end


% ===================== Rendering helpers ==============================

function renderTraceTile(ax, targetConds, traces, condColors, lineStyle, stimEndSec, dispWin, yLim, ttl, sel, isModel, extremeSim)
    hold(ax, 'on');
    for ci = 1:numel(targetConds)
        cond = targetConds{ci};
        if ~isfield(traces, cond), continue; end
        col = condColors.(cond);
        td  = traces.(cond);
        if isModel
            if nargin >= 12 && isfield(extremeSim, cond)
                tag = sprintf('sim_%d %s', extremeSim.(cond).simIdx, extremeSim.(cond).direction);
            else
                tag = sprintf('sim_%d, b=%.2f, n=%d', sel.(cond).sim - 1, sel.(cond).bias, numel(td.mu));
            end
            displayName = sprintf('%s model (%s)', cond, tag);
        else
            n = NaN;
            if isfield(td, 'n'), n = td.n; end
            displayName = sprintf('%s data', cond);
        end
        plotMeanSD(ax, td.t, td.mu, td.sd, col, lineStyle, displayName);
    end
    xline(ax, 0,          'k:', 'stim onset');
    xline(ax, stimEndSec, 'k:', 'stim end');
    xlim(ax, dispWin);
    ylim(ax, yLim);
    xlabel(ax, 'Time from stim onset (s)');
    ylabel(ax, 'Fraction active');
    title(ax, ttl);
    grid(ax, 'on'); box(ax, 'on');
    legend(ax, 'Location', 'northeast', 'Box', 'off', 'FontSize', 8);
end

function pOut = renderBoxTile(ax, targetConds, metric, condColors)
% Renders 2N boxes (N conds, data+model), with dashed mean lines per box
% and N^2 ranksum brackets:
%   - N within-cond (data vs model)
%   - C(N,2) cross-cond pairs on the data side (positions 2k-1 ↔ 2j-1)
%   - C(N,2) cross-cond pairs on the model side (positions 2k ↔ 2j)
% Returns a struct of p-values, keyed by "cond1_source_vs_cond2_source".
    hold(ax, 'on');
    nC = numel(targetConds);
    nGrp = 2 * nC;
    grpCols   = zeros(nGrp, 3);
    grpAlphas = zeros(1, nGrp);
    grpLabels = cell(1, nGrp);
    grpVals   = metric.data;     % cell{1..2N} from collectGroupVals
    for ci = 1:nC
        col = condColors.(targetConds{ci});
        grpCols(2*ci - 1, :) = col;   grpAlphas(2*ci - 1) = 0.55;   grpLabels{2*ci - 1} = sprintf('%s data',  targetConds{ci});
        grpCols(2*ci    , :) = col;   grpAlphas(2*ci    ) = 0.30;   grpLabels{2*ci    } = sprintf('%s model', targetConds{ci});
    end

    allY = [];
    for gi = 1:nGrp
        y = grpVals{gi}; y = y(~isnan(y));
        if isempty(y), continue; end
        allY = [allY; y]; %#ok<AGROW>
        bc = boxchart(ax, gi * ones(numel(y), 1), y, 'BoxWidth', 0.55, 'MarkerStyle', 'none');
        bc.BoxFaceColor     = grpCols(gi, :);
        bc.BoxFaceAlpha     = grpAlphas(gi);
        bc.WhiskerLineColor = grpCols(gi, :) * 0.7;
        bc.BoxEdgeColor     = grpCols(gi, :) * 0.7;
        mu = mean(y, 'omitnan');
        plot(ax, [gi - 0.275, gi + 0.275], [mu, mu], '--', ...
            'Color', grpCols(gi, :) * 0.5, 'LineWidth', 1.4, 'HandleVisibility', 'off');
    end
    if ~isnan(metric.yref)
        yline(ax, metric.yref, '--', 'Color', [0.5 0.5 0.5], ...
            'Label', sprintf('ref=%g', metric.yref), 'LabelHorizontalAlignment', 'left');
    end
    set(ax, 'XTick', 1:nGrp, 'XTickLabel', grpLabels, 'XTickLabelRotation', 35, ...
            'XLim', [0.3, nGrp + 0.7]);
    ylabel(ax, metric.ylab);
    title(ax, metric.ttl); grid(ax, 'on'); box(ax, 'on');

    % Stats
    pOut = struct();
    if isempty(allY), return; end
    yBase = max(allY) * 1.05;
    if yBase <= 0, yBase = 0.01; end
    bracketY = yBase;
    pairOrder = {};   % {x1, x2, p, label}
    % 1) within-cond data vs model
    for ci = 1:nC
        a = grpVals{2*ci - 1}; b = grpVals{2*ci};
        p_ = ranksumSafe(a, b);
        key = sprintf('%s_data_vs_%s_model', targetConds{ci}, targetConds{ci});
        pOut.(key) = p_;
        pairOrder(end+1, :) = {2*ci - 1, 2*ci, p_, key}; %#ok<AGROW>
    end
    % 2) cross-cond data
    for ci = 1:nC
        for cj = ci+1:nC
            a = grpVals{2*ci - 1}; b = grpVals{2*cj - 1};
            p_ = ranksumSafe(a, b);
            key = sprintf('%s_data_vs_%s_data', targetConds{ci}, targetConds{cj});
            pOut.(key) = p_;
            pairOrder(end+1, :) = {2*ci - 1, 2*cj - 1, p_, key}; %#ok<AGROW>
        end
    end
    % 3) cross-cond model
    for ci = 1:nC
        for cj = ci+1:nC
            a = grpVals{2*ci}; b = grpVals{2*cj};
            p_ = ranksumSafe(a, b);
            key = sprintf('%s_model_vs_%s_model', targetConds{ci}, targetConds{cj});
            pOut.(key) = p_;
            pairOrder(end+1, :) = {2*ci, 2*cj, p_, key}; %#ok<AGROW>
        end
    end
    % Render: stack brackets at increasing yBase fractions
    nPairs = size(pairOrder, 1);
    stepFrac = 0.08;
    for k = 1:nPairs
        plotSigBracket(ax, pairOrder{k, 1}, pairOrder{k, 2}, bracketY * (1 + (k-1) * stepFrac), pairOrder{k, 3});
    end
    ax.YLim = [min(0, min(allY)), bracketY * (1 + (nPairs + 1) * stepFrac)];
end

function gv = collectGroupVals(targetConds, dataStruct, modelStruct)
% Interleave data, model per cond into a 1×(2*nC) cell array.
    nC = numel(targetConds);
    gv = cell(1, 2*nC);
    for ci = 1:nC
        cond = targetConds{ci};
        gv{2*ci - 1} = dataStruct.(cond)(:);
        gv{2*ci    } = modelStruct.(cond)(:);
    end
end

function txt = buildSummaryTitle(stem, sz, wn, sel, targetConds, foldBestSim, extremeSim, normToNaive)
    nC = numel(targetConds);
    biasParts = cell(1, nC);
    for ci = 1:nC
        cond = targetConds{ci};
        biasParts{ci} = sprintf('%s b=%.2f', cond(1:min(3, end)), sel.(cond).bias);
    end
    fbParts = {};
    for ci = 1:nC
        cond = targetConds{ci};
        if isfield(foldBestSim, cond)
            fbParts{end+1} = sprintf('%s=%d', cond(1:min(3, end)), sel.(cond).sim - 1); %#ok<AGROW>
        end
    end
    xtParts = {};
    for ci = 1:nC
        cond = targetConds{ci};
        if isfield(extremeSim, cond)
            xtParts{end+1} = sprintf('%s=%d(%s)', cond(1:min(3, end)), extremeSim.(cond).simIdx, extremeSim.(cond).direction(4)); %#ok<AGROW>
        end
    end
    normTag = ''; if normToNaive, normTag = ' [Naive-baseline-normalized]'; end
    txt = sprintf('%s — size=%d, window=%s%s | %s | fold-best: %s | extreme-Δ: %s', ...
        stem, sz, wn, normTag, strjoin(biasParts, ', '), strjoin(fbParts, ' '), strjoin(xtParts, ' '));
end

function writeStatsCsv(outDirSW, stem, targetConds, metrics, pTable, sel, ...
        foldBestSim, extremeSim, baselineWin, onsetWin, offsetWin, stimEndSec, ...
        normToNaive, naiveBaseData, naiveBaseModel, branchTag)
    if nargin < 16, branchTag = ''; end
    statsCsv = fullfile(outDirSW, [stem '_stats.csv']);
    fid = fopen(statsCsv, 'w');
    if fid <= 0, return; end
    if ~isempty(branchTag)
        fprintf(fid, '# sim_selection,%s\n', branchTag);
    end
    fprintf(fid, 'metric,condition,source,n,mean,std,median,iqr_low,iqr_high,bias,sim_idx\n');
    nC = numel(targetConds);
    for mi = 1:numel(metrics)
        m = metrics(mi);
        for ci = 1:nC
            cond = targetConds{ci};
            for si = 1:2
                src = 'data'; if si == 2, src = 'model'; end
                y = m.data{2*ci - 2 + si}; y = y(~isnan(y));
                simIdxStr = '';
                if si == 2
                    if strcmp(branchTag, 'extremedelta') && isfield(extremeSim, cond)
                        simIdxStr = sprintf('%d', extremeSim.(cond).simIdx);
                    else
                        simIdxStr = sprintf('%d', sel.(cond).sim - 1);
                    end
                end
                if isempty(y)
                    fprintf(fid, '%s,%s,%s,0,NaN,NaN,NaN,NaN,NaN,%.4f,%s\n', m.name, cond, src, sel.(cond).bias, simIdxStr);
                    continue;
                end
                q = quantile(y, [0.25, 0.75]);
                fprintf(fid, '%s,%s,%s,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.4f,%s\n', ...
                    m.name, cond, src, numel(y), mean(y), std(y), median(y), q(1), q(2), sel.(cond).bias, simIdxStr);
            end
        end
    end
    fprintf(fid, '\np_value_test,metric,comparison,p\n');
    for mi = 1:numel(metrics)
        m = metrics(mi);
        pt = pTable.(m.name);
        keys = fieldnames(pt);
        for ki = 1:numel(keys)
            fprintf(fid, 'ranksum,%s,%s,%.6g\n', m.name, keys{ki}, pt.(keys{ki}));
        end
    end
    fprintf(fid, '\nwindow,start,end\n');
    fprintf(fid, 'baseline,%.3f,%.3f\n', baselineWin(1), baselineWin(2));
    fprintf(fid, 'onset,%.3f,%.3f\n',    onsetWin(1),    onsetWin(2));
    fprintf(fid, 'offset_rel_stimend,%.3f,%.3f\n', offsetWin(1), offsetWin(2));
    fprintf(fid, 'stim,%.3f,%.3f\n', 0, stimEndSec);
    if normToNaive
        fprintf(fid, '\nnaive_baseline,data,%.6g\n', naiveBaseData);
        fprintf(fid, 'naive_baseline,model,%.6g\n', naiveBaseModel);
    end
    fprintf(fid, '\nsim_selection_method,cond,sim_idx,detail\n');
    for ci = 1:nC
        cond = targetConds{ci};
        if isfield(foldBestSim, cond)
            fb = foldBestSim.(cond);
            fprintf(fid, 'fold_best,%s,%d,target=%.4f model_mean=%.4f err=%.4f\n', cond, fb.simIdx, fb.targetFold, fb.meanFold, fb.err);
        end
        if isfield(extremeSim, cond)
            xs = extremeSim.(cond);
            fprintf(fid, 'extreme_%s,%s,%d,delta_mean=%.4f\n', xs.direction, cond, xs.simIdx, xs.delta);
        end
    end
    fclose(fid);
    fprintf('    saved: %s\n', statsCsv);
end


% ===================== Low-level helpers ==============================

function reps = loadActivityCropReps(biasFile, clampedFile, cond, sIdx, biasVal, mode, clampedMode, sz, chosenDur)
    if isinf(biasVal)
        leafPath = sprintf('/experiments/%s/sim_%d/%s/size_%d/dur_%d/activity_crop', ...
            cond, sIdx, clampedMode, sz, chosenDur);
        file = clampedFile;
    else
        bk = sprintf('bias_%s', strrep(sprintf('%.2f', biasVal), '.', 'p'));
        leafPath = sprintf('/experiments/%s/sim_%d/%s/size_%d/dur_%d/%s/activity_crop', ...
            cond, sIdx, mode, sz, chosenDur, bk);
        file = biasFile;
    end
    reps = h5read(file, leafPath);
    if size(reps, 1) > size(reps, 2), reps = reps'; end
end

function fold = computeFoldPerRow(rows, t, baselineWin, onsetWin, offsetWin, stimEndSec, baselineOverride)
% Per-row fold = max(onset_peak, offset_peak) / baseline.
% If baselineOverride is finite, use that scalar as the denominator
% instead of each row's own baseline mean.
    if nargin < 7, baselineOverride = NaN; end
    n = size(rows, 1);
    fold = nan(n, 1);
    bMask  = t >= baselineWin(1)             & t <= baselineWin(2);
    oMask  = t >= onsetWin(1)                & t <= onsetWin(2);
    fMask  = t >= stimEndSec + offsetWin(1)  & t <= stimEndSec + offsetWin(2);
    useOverride = isfinite(baselineOverride);
    for r = 1:n
        tr   = rows(r, :);
        if useOverride
            base = baselineOverride;
        else
            okB  = bMask & ~isnan(tr);
            if nnz(okB) < 2, continue; end
            base = mean(tr(okB));
        end
        if ~isfinite(base) || base <= 0, continue; end
        okO = oMask & ~isnan(tr);
        okF = fMask & ~isnan(tr);
        if ~any(okO) && ~any(okF), continue; end
        peak = -inf;
        if any(okO), peak = max(peak, max(tr(okO))); end
        if any(okF), peak = max(peak, max(tr(okF))); end
        if ~isfinite(peak), continue; end
        fold(r) = peak / base;
    end
end

function vals = computeMetricPerRow(rows, t, metric, baselineWin, onsetWin, offsetWin, stimEndSec, baselineOverride)
% Same baselineOverride semantics as computeFoldPerRow:
%   delta_base: peak - baselineOverride (if finite) else peak - own_baseline
%   stim_mean : mean(stim window) - baselineOverride (if finite) else raw mean
    if nargin < 8, baselineOverride = NaN; end
    n = size(rows, 1);
    vals = nan(n, 1);
    bMask = t >= baselineWin(1)            & t <= baselineWin(2);
    oMask = t >= onsetWin(1)               & t <= onsetWin(2);
    fMask = t >= stimEndSec + offsetWin(1) & t <= stimEndSec + offsetWin(2);
    sMask = t >= 0                         & t <= stimEndSec;
    useOverride = isfinite(baselineOverride);
    for r = 1:n
        tr = rows(r, :);
        okB = bMask & ~isnan(tr);
        okO = oMask & ~isnan(tr);
        okF = fMask & ~isnan(tr);
        okS = sMask & ~isnan(tr);
        switch metric
            case 'onset_peak'
                if ~any(okO), continue; end
                vals(r) = max(tr(okO));
            case 'stim_mean'
                if nnz(okS) < 2, continue; end
                m = mean(tr(okS));
                if useOverride
                    vals(r) = m - baselineOverride;
                else
                    vals(r) = m;
                end
            case 'delta_base'
                if useOverride
                    base = baselineOverride;
                else
                    if nnz(okB) < 2, continue; end
                    base = mean(tr(okB));
                end
                if ~any(okO) && ~any(okF), continue; end
                peak = -inf;
                if any(okO), peak = max(peak, max(tr(okO))); end
                if any(okF), peak = max(peak, max(tr(okF))); end
                if ~isfinite(peak) || ~isfinite(base), continue; end
                vals(r) = peak - base;
            otherwise
                error('Unknown metric: %s', metric);
        end
    end
end

function plotMeanSD(ax, t, mu, sd, col, lineStyle, displayName)
    fill(ax, [t, fliplr(t)], [mu + sd, fliplr(mu - sd)], col, ...
        'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(ax, t, mu, lineStyle, 'Color', col, 'LineWidth', 1.8, ...
        'DisplayName', displayName);
end

function p = ranksumSafe(a, b)
    a = a(~isnan(a)); b = b(~isnan(b));
    if isempty(a) || isempty(b), p = NaN; return; end
    try, p = ranksum(a, b); catch, p = NaN; end
end

function plotSigBracket(ax, x1, x2, y, p)
    line(ax, [x1, x1, x2, x2], [y*0.97, y, y, y*0.97], 'Color', 'k', 'LineWidth', 0.6);
    if isnan(p),       lbl = 'n.s.';
    elseif p < 0.001,  lbl = '***';
    elseif p < 0.01,   lbl = '**';
    elseif p < 0.05,   lbl = '*';
    else,              lbl = 'n.s.';
    end
    if isnan(p), txt = lbl; else, txt = sprintf('%s p=%.2g', lbl, p); end
    text(ax, (x1+x2)/2, y*1.03, txt, ...
        'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', 'k');
end

function out = ifelse(cond, a, b)
    if cond, out = a; else, out = b; end
end

function pair = findMatchedBetaPair(biasFile, mode, nSimsTotal, targetConds)
% findMatchedBetaPair  Locate one (Expert, NoSpout) top-10 pair that shares
%   every Ising sim parameter except beta.
%
%   Reads /best_matches/<cond>/sim_<i>/{beta,c,decay_const,inhibition_range,bias}
%   from the perturbation HDF5 file (`biasFile`). Returns:
%     struct(  'Expert',  struct('simIdx', e0, 'beta', be), ...
%              'NoSpout', struct('simIdx', n0, 'beta', bn), ...
%              'shared',  struct('c', .., 'decay_const', .., 'inhibition_range', .., 'bias', ..) )
%   or struct() if no matched-except-beta pair exists.
%
%   The mode arg is accepted for forward-compatibility (currently unused —
%   /best_matches/<cond>/sim_<i>/ is mode-invariant in run_ising_perturbations.py).
%
%   Requires exactly 2 conditions in targetConds. Matches the largest-beta-gap
%   pair when multiple exist (most isolated beta effect).

    pair = struct();
    if nargin < 4, targetConds = {'Expert', 'NoSpout'}; end
    if numel(targetConds) ~= 2
        warning('findMatchedBetaPair: requires exactly 2 target conds, got %d', numel(targetConds));
        return;
    end
    cA = targetConds{1};
    cB = targetConds{2};

    paramKeys = {'beta', 'c', 'decay_const', 'inhibition_range', 'bias'};
    nParam    = numel(paramKeys);

    Pa = readBestMatchParams(biasFile, cA, nSimsTotal, paramKeys);
    Pb = readBestMatchParams(biasFile, cB, nSimsTotal, paramKeys);

    if all(isnan(Pa(:))) || all(isnan(Pb(:)))
        warning('findMatchedBetaPair: could not read /best_matches/%s or /best_matches/%s from %s', ...
                cA, cB, biasFile);
        return;
    end

    % Scan all 100 cross-pairs for matched-except-beta. Match keys = idx 2:5.
    matchIdx = 2:nParam;
    bestGap = -Inf;
    bestI = NaN; bestJ = NaN;
    fprintf('    [findMatchedBetaPair] scanning %dx%d top-10 cross-pairs (%s vs %s)\n', ...
            nSimsTotal, nSimsTotal, cA, cB);
    nFound = 0;
    for i = 1:nSimsTotal
        for j = 1:nSimsTotal
            if any(isnan(Pa(i, :))) || any(isnan(Pb(j, :))), continue; end
            if isequal(Pa(i, matchIdx), Pb(j, matchIdx)) && Pa(i, 1) ~= Pb(j, 1)
                gap = abs(Pa(i, 1) - Pb(j, 1));
                nFound = nFound + 1;
                fprintf('      pair (%s.sim_%d, %s.sim_%d): beta=%.3f vs %.3f (|Δ|=%.3f), shared c=%g d=%g r=%g h_b=%g\n', ...
                        cA, i-1, cB, j-1, Pa(i,1), Pb(j,1), gap, ...
                        Pa(i,2), Pa(i,3), Pa(i,4), Pa(i,5));
                if gap > bestGap
                    bestGap = gap; bestI = i; bestJ = j;
                end
            end
        end
    end
    fprintf('    [findMatchedBetaPair] %d matched-except-beta pair(s) found\n', nFound);
    if isnan(bestI), return; end

    pair.(cA) = struct('simIdx', bestI - 1, 'beta', Pa(bestI, 1));
    pair.(cB) = struct('simIdx', bestJ - 1, 'beta', Pb(bestJ, 1));
    pair.shared = struct('c',                Pa(bestI, 2), ...
                         'decay_const',      Pa(bestI, 3), ...
                         'inhibition_range', Pa(bestI, 4), ...
                         'bias',             Pa(bestI, 5));
end

function P = readBestMatchParams(biasFile, condName, nSimsTotal, paramKeys)
% Read /best_matches/<condName>/sim_<i>/<paramKeys{k}> for i=0..nSimsTotal-1.
%   Returns nSimsTotal x numel(paramKeys) double matrix, NaN where missing.
    nParam = numel(paramKeys);
    P = NaN(nSimsTotal, nParam);
    for i = 0:(nSimsTotal-1)
        for kk = 1:nParam
            ds = sprintf('/best_matches/%s/sim_%d/%s', condName, i, paramKeys{kk});
            try
                v = h5read(biasFile, ds);
                P(i+1, kk) = double(v(1));
            catch
                P(i+1, kk) = NaN;
            end
        end
    end
end
