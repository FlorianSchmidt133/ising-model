function Figure6_AllConds_ManualSims(expTraces, sel, sz, wn, outDirSW, ...
    matcherOut, condColors, opts, simTimeAxis, stimEndSec, ...
    biasFile, clampedFile, chosenDur, manualSims, varargin)
% Figure6_AllConds_ManualSims
%   Compact variant of Figure6_ExpertVsNoSpout that uses HARDCODED sim
%   indices per condition (no fold-best / extreme-Δ / ordering selection).
%
%   Renders:
%     - <stem>_traces.{png,pdf,svg}    1×2 layout: Data | Model (manual sims)
%     - <stem>_boxplots.{png,pdf,svg}  1×3 layout: fold | stim_mean | delta_base
%                                       (8 boxes per panel, 4 conds × data/model)
%     - <stem>_stats.csv               per-metric per-cond means/stds/medians + p-values
%
%   `manualSims` is a struct mapping cond → 0-indexed sim number:
%     manualSims = struct('Naive', 8, 'Beginner', 8, 'Expert', 3, 'NoSpout', 3);
%
%   Optional name-value args:
%     'NormalizeToNaiveBaseline'  (default false)
%     'OutputStem'                (default 'Fig6_AllConds_Manual')

    p = inputParser; p.KeepUnmatched = true;
    addParameter(p, 'NormalizeToNaiveBaseline', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'OutputStem', 'Fig6_AllConds_Manual', @(s) ischar(s) || isstring(s));
    addParameter(p, 'TargetConds', {'Naive','Beginner','Expert','NoSpout'}, @iscell);
    parse(p, varargin{:});
    normToNaive = logical(p.Results.NormalizeToNaiveBaseline);
    outputStem  = char(p.Results.OutputStem);
    if exist('outDirSW', 'var') && (isstring(outDirSW) || iscell(outDirSW))
        outDirSW = char(outDirSW);
    end

    targetConds = p.Results.TargetConds;
    nC = numel(targetConds);

    missing = {};
    for ci = 1:nC
        cond = targetConds{ci};
        if ~(isfield(expTraces, cond) && isfield(sel, cond)) || ...
                ~isfield(manualSims, cond)
            missing{end+1} = cond; %#ok<AGROW>
        end
    end
    if ~isempty(missing)
        warning('Figure6_AllConds_ManualSims: missing data/sim for [%s]; skipping', strjoin(missing, ','));
        return;
    end

    if ~exist(outDirSW, 'dir'), mkdir(outDirSW); end

    onsetWin    = [0, 0.3];
    offsetWin   = [0, 0.4];
    baselineWin = [-1, -0.1];
    if isfield(matcherOut, 'opts')
        if isfield(matcherOut.opts, 'OnsetWindow'),        onsetWin    = matcherOut.opts.OnsetWindow;        end
        if isfield(matcherOut.opts, 'OffsetWindow'),       offsetWin   = matcherOut.opts.OffsetWindow;       end
        if isfield(matcherOut.opts, 'FoldBaselineWindow'), baselineWin = matcherOut.opts.FoldBaselineWindow; end
    end

    timeStretch = ones(numel(opts.Conditions), 1);
    timeShift   = zeros(numel(opts.Conditions), 1);
    if isfield(matcherOut, 'sizeOut')
        if isfield(matcherOut.sizeOut, 'timeStretch'), timeStretch = matcherOut.sizeOut.timeStretch; end
        if isfield(matcherOut.sizeOut, 'timeShift'),   timeShift   = matcherOut.sizeOut.timeShift;   end
    end

    % Load reps for the manual sim picks
    repsByCond = struct();
    for ci = 1:nC
        cond = targetConds{ci};
        sIdx = manualSims.(cond);
        bv   = sel.(cond).bias;
        try
            reps = loadActivityCropReps(biasFile, clampedFile, ...
                cond, sIdx, bv, opts.Mode, opts.ClampedMode, sz, chosenDur);
        catch ME
            warning('Failed to load %s sim_%d: %s', cond, sIdx, ME.message);
            return;
        end
        repsByCond.(cond) = reps;
        fprintf('    [%s] manual %s: sim_%d (bias=%.2f, n=%d)\n', ...
            outputStem, cond, sIdx, bv, size(reps, 1));
    end

    % Naive-baseline scalars (only used in normToNaive branch)
    naiveBaseData  = NaN;
    naiveBaseModel = NaN;
    if normToNaive
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
            tmp = repsByCond.Naive(:, bMaskM);
            naiveBaseModel = mean(tmp(:), 'omitnan');
        end
    end

    % Compute per-cond traces + per-row metrics
    foldData    = struct(); stimMeanData    = struct(); deltaBaseData    = struct();
    foldModel   = struct(); stimMeanModel   = struct(); deltaBaseModel   = struct();
    traceData   = struct(); traceModel      = struct();
    for ci = 1:nC
        cond = targetConds{ci};
        cIdx = find(strcmp(opts.Conditions, cond), 1);

        recT = expTraces.(cond).recTraces;
        tD   = expTraces.(cond).time(:)';
        traceData.(cond).t  = tD;
        traceData.(cond).mu = mean(recT, 1, 'omitnan');
        traceData.(cond).sd = std(recT,  0, 1, 'omitnan');

        baseDataOverride = NaN; if normToNaive, baseDataOverride = naiveBaseData; end
        foldData.(cond)       = computeFoldPerRow(  recT, tD, baselineWin, onsetWin, offsetWin, stimEndSec, baseDataOverride);
        stimMeanData.(cond)   = computeMetricPerRow(recT, tD, 'stim_mean',  baselineWin, onsetWin, offsetWin, stimEndSec, baseDataOverride);
        deltaBaseData.(cond)  = computeMetricPerRow(recT, tD, 'delta_base', baselineWin, onsetWin, offsetWin, stimEndSec, baseDataOverride);

        reps = repsByCond.(cond);
        if isnan(cIdx)
            tM = simTimeAxis(:)';
        else
            tM = simTimeAxis(:)' * timeStretch(cIdx) - timeShift(cIdx);
        end
        traceModel.(cond).t  = tM;
        traceModel.(cond).mu = mean(reps, 1, 'omitnan');
        traceModel.(cond).sd = std(reps,  0, 1, 'omitnan');

        baseModelOverride = NaN; if normToNaive, baseModelOverride = naiveBaseModel; end
        foldModel.(cond)      = computeFoldPerRow(  reps, tM, baselineWin, onsetWin, offsetWin, stimEndSec, baseModelOverride);
        stimMeanModel.(cond)  = computeMetricPerRow(reps, tM, 'stim_mean',  baselineWin, onsetWin, offsetWin, stimEndSec, baseModelOverride);
        deltaBaseModel.(cond) = computeMetricPerRow(reps, tM, 'delta_base', baselineWin, onsetWin, offsetWin, stimEndSec, baseModelOverride);
    end

    % Shared Y for traces
    yMax = 0;
    for ci = 1:nC
        cond = targetConds{ci};
        yMax = max([yMax, ...
            max(traceData.(cond).mu  + traceData.(cond).sd,  [], 'omitnan'), ...
            max(traceModel.(cond).mu + traceModel.(cond).sd, [], 'omitnan')]);
    end
    if ~isfinite(yMax) || yMax <= 0, yMax = 0.05; end
    yLimTraces = [0, yMax * 1.10];

    summaryStr = sprintf('%s — size=%d, window=%s | manual sims: %s', outputStem, sz, wn, ...
        strjoin(arrayfun(@(ci) sprintf('%s=%d', targetConds{ci}(1:min(3,end)), manualSims.(targetConds{ci})), ...
            1:nC, 'UniformOutput', false), ', '));
    if normToNaive
        summaryStr = [summaryStr, ' [Naive-baseline-normalized]'];
    end

    % --- Traces 1×2 figure ---
    f1 = figure('Name', [outputStem '_traces'], 'Color', 'w');
    tl1 = tiledlayout(f1, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax_data  = nexttile(tl1, 1); renderTraceTile(ax_data,  targetConds, traceData,  condColors, '-',  stimEndSec, opts.DisplayWindow, yLimTraces, 'Data (mean ± std across recs)', false, struct());
    ax_model = nexttile(tl1, 2); renderTraceTile(ax_model, targetConds, traceModel, condColors, '--', stimEndSec, opts.DisplayWindow, yLimTraces, 'Model (manual sims, mean ± std)', true, manualSims);
    title(tl1, summaryStr, 'FontWeight', 'normal');
    set(f1, 'Position', [50, 50, 1100, 450]);
    saveStem1 = fullfile(outDirSW, [outputStem '_traces']);
    savePngRobust(f1, [saveStem1 '.png']);
    try, exportgraphics(f1, [saveStem1 '.pdf'], 'ContentType', 'vector'); catch ME, fprintf(2, '    pdf fail: %s\n', ME.message); end
    try, print(f1, [saveStem1 '.svg'], '-dsvg'); catch ME, fprintf(2, '    svg fail: %s\n', ME.message); end
    close(f1);
    fprintf('    saved: %s.png\n', saveStem1);

    % --- Boxplots 1×3 figure ---
    titleFold  = ifelse(normToNaive, 'Fold quant (Naive-baseline-normalized)', 'Fold quantification');
    ylabFold   = ifelse(normToNaive, 'Peak / Naive baseline',                  'Fold change (peak / baseline)');
    titleStim  = ifelse(normToNaive, 'Stim mean (Naive-baseline-subtracted)',  'Stim-period mean');
    ylabStim   = ifelse(normToNaive, 'Stim mean − Naive baseline',             'Stim-period mean (fraction active)');
    titleDelta = ifelse(normToNaive, '\Delta to Naive baseline',                '\Delta to baseline');
    ylabDelta  = ifelse(normToNaive, 'Peak − Naive baseline',                   '\Delta to baseline (peak − baseline)');
    yrefStim   = ifelse(normToNaive, 0, NaN);

    metrics(1).name='fold';      metrics(1).ttl=titleFold;  metrics(1).ylab=ylabFold;  metrics(1).yref=1;
    metrics(1).data = collectGroupVals(targetConds, foldData, foldModel);
    metrics(2).name='stim_mean'; metrics(2).ttl=titleStim;  metrics(2).ylab=ylabStim;  metrics(2).yref=yrefStim;
    metrics(2).data = collectGroupVals(targetConds, stimMeanData, stimMeanModel);
    metrics(3).name='delta_base';metrics(3).ttl=titleDelta; metrics(3).ylab=ylabDelta; metrics(3).yref=0;
    metrics(3).data = collectGroupVals(targetConds, deltaBaseData, deltaBaseModel);

    f2 = figure('Name', [outputStem '_boxplots'], 'Color', 'w');
    tl2 = tiledlayout(f2, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    pTable = struct();
    for mi = 1:numel(metrics)
        ax = nexttile(tl2, mi);
        pTable.(metrics(mi).name) = renderBoxTile(ax, targetConds, metrics(mi), condColors);
    end
    title(tl2, summaryStr, 'FontWeight', 'normal');
    set(f2, 'Position', [50, 50, 1500, 500]);
    saveStem2 = fullfile(outDirSW, [outputStem '_boxplots']);
    savePngRobust(f2, [saveStem2 '.png']);
    try, exportgraphics(f2, [saveStem2 '.pdf'], 'ContentType', 'vector'); catch ME, fprintf(2, '    pdf fail: %s\n', ME.message); end
    try, print(f2, [saveStem2 '.svg'], '-dsvg'); catch ME, fprintf(2, '    svg fail: %s\n', ME.message); end
    close(f2);
    fprintf('    saved: %s.png\n', saveStem2);

    % --- Stats CSV ---
    writeStatsCsv(outDirSW, outputStem, targetConds, metrics, pTable, sel, ...
        manualSims, baselineWin, onsetWin, offsetWin, stimEndSec, ...
        normToNaive, naiveBaseData, naiveBaseModel);
end


% ===================== Helpers ========================================

function gv = collectGroupVals(targetConds, dataStruct, modelStruct)
    nC = numel(targetConds);
    gv = cell(1, 2*nC);
    for ci = 1:nC
        cond = targetConds{ci};
        gv{2*ci - 1} = dataStruct.(cond)(:);
        gv{2*ci    } = modelStruct.(cond)(:);
    end
end

function renderTraceTile(ax, targetConds, traces, condColors, lineStyle, stimEndSec, dispWin, yLim, ttl, isModel, manualSims)
    hold(ax, 'on');
    for ci = 1:numel(targetConds)
        cond = targetConds{ci};
        if ~isfield(traces, cond), continue; end
        col = condColors.(cond);
        td  = traces.(cond);
        if isModel
            displayName = sprintf('%s model (sim_%d)', cond, manualSims.(cond));
        else
            displayName = sprintf('%s data', cond);
        end
        fill(ax, [td.t, fliplr(td.t)], [td.mu + td.sd, fliplr(td.mu - td.sd)], col, ...
            'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(ax, td.t, td.mu, lineStyle, 'Color', col, 'LineWidth', 1.8, 'DisplayName', displayName);
    end
    xline(ax, 0,          'k:', 'stim onset');
    xline(ax, stimEndSec, 'k:', 'stim end');
    xlim(ax, dispWin); ylim(ax, yLim);
    xlabel(ax, 'Time from stim onset (s)'); ylabel(ax, 'Fraction active');
    title(ax, ttl); grid(ax, 'on'); box(ax, 'on');
    legend(ax, 'Location', 'northeast', 'Box', 'off', 'FontSize', 8);
end

function pOut = renderBoxTile(ax, targetConds, metric, condColors)
    hold(ax, 'on');
    nC = numel(targetConds);
    nGrp = 2 * nC;
    grpCols = zeros(nGrp, 3); grpAlphas = zeros(1, nGrp); grpLabels = cell(1, nGrp);
    for ci = 1:nC
        col = condColors.(targetConds{ci});
        grpCols(2*ci - 1, :) = col; grpAlphas(2*ci - 1) = 0.55; grpLabels{2*ci - 1} = sprintf('%s data',  targetConds{ci});
        grpCols(2*ci    , :) = col; grpAlphas(2*ci    ) = 0.30; grpLabels{2*ci    } = sprintf('%s model', targetConds{ci});
    end
    grpVals = metric.data;
    allY = [];
    for gi = 1:nGrp
        y = grpVals{gi}; y = y(~isnan(y));
        if isempty(y), continue; end
        allY = [allY; y]; %#ok<AGROW>
        bc = boxchart(ax, gi * ones(numel(y), 1), y, 'BoxWidth', 0.55, 'MarkerStyle', 'none');
        bc.BoxFaceColor    = grpCols(gi, :); bc.BoxFaceAlpha = grpAlphas(gi);
        bc.WhiskerLineColor = grpCols(gi, :) * 0.7; bc.BoxEdgeColor = grpCols(gi, :) * 0.7;
        mu = mean(y, 'omitnan');
        plot(ax, [gi - 0.275, gi + 0.275], [mu, mu], '--', ...
            'Color', grpCols(gi, :) * 0.5, 'LineWidth', 1.4, 'HandleVisibility', 'off');
    end
    if ~isnan(metric.yref)
        yline(ax, metric.yref, '--', 'Color', [0.5 0.5 0.5], 'Label', sprintf('ref=%g', metric.yref), 'LabelHorizontalAlignment', 'left');
    end
    set(ax, 'XTick', 1:nGrp, 'XTickLabel', grpLabels, 'XTickLabelRotation', 35, 'XLim', [0.3, nGrp + 0.7]);
    ylabel(ax, metric.ylab); title(ax, metric.ttl); grid(ax, 'on'); box(ax, 'on');

    pOut = struct();
    if isempty(allY), return; end
    yBase = max(allY) * 1.05; if yBase <= 0, yBase = 0.01; end
    pairOrder = {};
    for ci = 1:nC
        a = grpVals{2*ci - 1}; b = grpVals{2*ci};
        p_ = ranksumSafe(a, b);
        key = sprintf('%s_data_vs_%s_model', targetConds{ci}, targetConds{ci});
        pOut.(key) = p_;
        pairOrder(end+1, :) = {2*ci - 1, 2*ci, p_, key}; %#ok<AGROW>
    end
    for ci = 1:nC
        for cj = ci+1:nC
            a = grpVals{2*ci - 1}; b = grpVals{2*cj - 1};
            p_ = ranksumSafe(a, b);
            key = sprintf('%s_data_vs_%s_data', targetConds{ci}, targetConds{cj});
            pOut.(key) = p_;
            pairOrder(end+1, :) = {2*ci - 1, 2*cj - 1, p_, key}; %#ok<AGROW>
        end
    end
    for ci = 1:nC
        for cj = ci+1:nC
            a = grpVals{2*ci}; b = grpVals{2*cj};
            p_ = ranksumSafe(a, b);
            key = sprintf('%s_model_vs_%s_model', targetConds{ci}, targetConds{cj});
            pOut.(key) = p_;
            pairOrder(end+1, :) = {2*ci, 2*cj, p_, key}; %#ok<AGROW>
        end
    end
    nPairs = size(pairOrder, 1);
    stepFrac = 0.08;
    for k = 1:nPairs
        plotSigBracket(ax, pairOrder{k, 1}, pairOrder{k, 2}, yBase * (1 + (k-1) * stepFrac), pairOrder{k, 3});
    end
    ax.YLim = [min(0, min(allY)), yBase * (1 + (nPairs + 1) * stepFrac)];
end

function writeStatsCsv(outDirSW, stem, targetConds, metrics, pTable, sel, ...
        manualSims, baselineWin, onsetWin, offsetWin, stimEndSec, normToNaive, naiveBaseData, naiveBaseModel)
    statsCsv = fullfile(outDirSW, [stem '_stats.csv']);
    fid = fopen(statsCsv, 'w');
    if fid <= 0, return; end
    fprintf(fid, '# sim_selection,manual\n');
    fprintf(fid, '# manual_sims,');
    for ci = 1:numel(targetConds)
        cond = targetConds{ci};
        fprintf(fid, '%s=%d ', cond, manualSims.(cond));
    end
    fprintf(fid, '\n');
    fprintf(fid, 'metric,condition,source,n,mean,std,median,iqr_low,iqr_high,bias,sim_idx\n');
    for mi = 1:numel(metrics)
        m = metrics(mi);
        for ci = 1:numel(targetConds)
            cond = targetConds{ci};
            for si = 1:2
                src = 'data'; if si == 2, src = 'model'; end
                y = m.data{2*ci - 2 + si}; y = y(~isnan(y));
                simIdxStr = '';
                if si == 2, simIdxStr = sprintf('%d', manualSims.(cond)); end
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
        m = metrics(mi); pt = pTable.(m.name); keys = fieldnames(pt);
        for ki = 1:numel(keys)
            fprintf(fid, 'ranksum,%s,%s,%.6g\n', m.name, keys{ki}, pt.(keys{ki}));
        end
    end
    fprintf(fid, '\nwindow,start,end\nbaseline,%.3f,%.3f\nonset,%.3f,%.3f\noffset_rel_stimend,%.3f,%.3f\nstim,%.3f,%.3f\n', ...
        baselineWin(1), baselineWin(2), onsetWin(1), onsetWin(2), offsetWin(1), offsetWin(2), 0, stimEndSec);
    if normToNaive
        fprintf(fid, '\nnaive_baseline,data,%.6g\nnaive_baseline,model,%.6g\n', naiveBaseData, naiveBaseModel);
    end
    fclose(fid);
    fprintf('    saved: %s\n', statsCsv);
end

function reps = loadActivityCropReps(biasFile, clampedFile, cond, sIdx, biasVal, mode, clampedMode, sz, chosenDur)
    if isinf(biasVal)
        leafPath = sprintf('/experiments/%s/sim_%d/%s/size_%d/dur_%d/activity_crop', cond, sIdx, clampedMode, sz, chosenDur);
        file = clampedFile;
    else
        bk = sprintf('bias_%s', strrep(sprintf('%.2f', biasVal), '.', 'p'));
        leafPath = sprintf('/experiments/%s/sim_%d/%s/size_%d/dur_%d/%s/activity_crop', cond, sIdx, mode, sz, chosenDur, bk);
        file = biasFile;
    end
    reps = h5read(file, leafPath);
    if size(reps, 1) > size(reps, 2), reps = reps'; end
end

function fold = computeFoldPerRow(rows, t, baselineWin, onsetWin, offsetWin, stimEndSec, baselineOverride)
    if nargin < 7, baselineOverride = NaN; end
    n = size(rows, 1); fold = nan(n, 1);
    bMask  = t >= baselineWin(1)             & t <= baselineWin(2);
    oMask  = t >= onsetWin(1)                & t <= onsetWin(2);
    fMask  = t >= stimEndSec + offsetWin(1)  & t <= stimEndSec + offsetWin(2);
    useOverride = isfinite(baselineOverride);
    for r = 1:n
        tr = rows(r, :);
        if useOverride
            base = baselineOverride;
        else
            okB = bMask & ~isnan(tr);
            if nnz(okB) < 2, continue; end
            base = mean(tr(okB));
        end
        if ~isfinite(base) || base <= 0, continue; end
        okO = oMask & ~isnan(tr); okF = fMask & ~isnan(tr);
        if ~any(okO) && ~any(okF), continue; end
        peak = -inf;
        if any(okO), peak = max(peak, max(tr(okO))); end
        if any(okF), peak = max(peak, max(tr(okF))); end
        if ~isfinite(peak), continue; end
        fold(r) = peak / base;
    end
end

function vals = computeMetricPerRow(rows, t, metric, baselineWin, onsetWin, offsetWin, stimEndSec, baselineOverride)
    if nargin < 8, baselineOverride = NaN; end
    n = size(rows, 1); vals = nan(n, 1);
    bMask = t >= baselineWin(1)            & t <= baselineWin(2);
    oMask = t >= onsetWin(1)               & t <= onsetWin(2);
    fMask = t >= stimEndSec + offsetWin(1) & t <= stimEndSec + offsetWin(2);
    sMask = t >= 0                         & t <= stimEndSec;
    useOverride = isfinite(baselineOverride);
    for r = 1:n
        tr = rows(r, :);
        okB = bMask & ~isnan(tr); okO = oMask & ~isnan(tr); okF = fMask & ~isnan(tr); okS = sMask & ~isnan(tr);
        switch metric
            case 'stim_mean'
                if nnz(okS) < 2, continue; end
                m = mean(tr(okS));
                if useOverride, vals(r) = m - baselineOverride; else, vals(r) = m; end
            case 'delta_base'
                if useOverride, base = baselineOverride;
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
    text(ax, (x1+x2)/2, y*1.03, txt, 'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', 'k');
end

function out = ifelse(cond, a, b)
    if cond, out = a; else, out = b; end
end

function savePngRobust(fig, pngPath)
    success = false;
    try
        exportgraphics(fig, pngPath, 'Resolution', 250);
        success = true;
    catch ME
        fprintf(2, '    exportgraphics(%s) failed: %s\n', pngPath, ME.message);
    end
    if ~success
        try
            print(fig, pngPath, '-dpng', '-r250');
            success = isfile(pngPath);
            if success, fprintf('    fallback: print -dpng OK\n'); end
        catch ME
            fprintf(2, '    print -dpng failed: %s\n', ME.message);
        end
    end
    if ~success
        try
            saveas(fig, pngPath);
            success = isfile(pngPath);
            if success, fprintf('    fallback: saveas OK\n'); end
        catch ME
            fprintf(2, '    saveas failed: %s\n', ME.message);
        end
    end
    if ~success
        fprintf(2, '    ALL PNG SAVE METHODS FAILED for %s\n', pngPath);
    end
end
