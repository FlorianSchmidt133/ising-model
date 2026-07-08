function out = Figure6g_BiasMatched_ConnectedSwarm(varargin)
%FIGURE6G_BIASMATCHED_CONNECTEDSWARM  Largest connected group (per-rep)
%   swarm panel for Figure 6g (Model side), using the bias-matched pipeline.
%
%   For each condition the matcher picks a "best bias" value (per size,
%   window). For that (sim, bias) we read the `activity_crop` HDF5
%   leaf from the dp_bias10 (or clamped) aggregate, take per-rep mean over
%   the stim window, and plot a swarm column per condition. Output PNGs
%   land in `<OutputDir>/size_<N>/window_<wn>/`, mirroring Fig6c.
%
%   Inputs (Name-Value):
%     'MatcherOutputDir' (str)  Folder containing per-size matcher outputs.
%        Looks for `<dir>/matcher_output_size<N>.mat` first, falls back to
%        `matcher_output.mat`.
%        Default: $PerturbationResultsPath/../PerturbationAnalysis/BiasMatchExperiment.
%     'PerturbationResultsPath' (str)  Where the dp_bias10 + dp10 aggregates
%        live. Default: mba_p('IsingModelData_39x78_100K\IsingPerturbations').
%     'OutputDir' (str)  Where panels land. Default:
%        'Fig. 6 Ising Model\g_BiasMatched_ConnectedSwarm'.
%     'Sizes' (vec)  Stim sizes to process. Default [2, 3, 4].
%     'Windows' (cell)  Windows to process. Default {'full','onset','offset','stim','bias6','fold_global','fold_nospout','fold_naive'}.
%     'Mode' (str)  Bias-mode aggregate name. Default 'double_pulse_bias10'.
%     'ClampedMode' (str)  Clamped aggregate name (used when best bias=Inf).
%        Default 'double_pulse10'.
%     'Conditions' (cell)  Default {'Naive','Beginner','Expert','NoSpout'}.
%     'SamplingRate' (num)  Hz. Default 10.
%     'MatchWindow' ([t0 t1])  X-axis range in seconds rel to stim onset.
%        Default [-1, 3]. Used to define the stim integration window for
%        each replicate (stimMask = simTimeAxis >= 0 & simTimeAxis < stimEndSec).
%     'BatchSize' (num)  Reps per dot. Default 1 = one dot per replicate.
%        Set higher (e.g. 10) to reduce visual density.
%     'YLabel' (str)  Y-axis label. Default 'Fraction active (FOV crop)'.

    p = inputParser;
    addParameter(p, 'MatcherOutputDir', '', @ischar);
    addParameter(p, 'PerturbationResultsPath', mba_p('IsingModelData_39x78_100K\IsingPerturbations'), @ischar);
    addParameter(p, 'OutputDir', 'Fig. 6 Ising Model\g_BiasMatched_ConnectedSwarm', @ischar);
    addParameter(p, 'Sizes', [2, 3, 4], @isnumeric);
    addParameter(p, 'Windows', {'full', 'onset', 'offset', 'stim', 'BothPeaks', 'StimPeriod', 'full_naive', 'onset_naive', 'offset_naive', 'stim_naive', 'BothPeaks_naive', 'StimPeriod_naive', 'bias6', 'fold_global', 'fold_nospout', 'fold_naive'}, @iscell);
    addParameter(p, 'Mode', 'double_pulse_bias10', @ischar);
    addParameter(p, 'ClampedMode', 'double_pulse10', @ischar);
    addParameter(p, 'Conditions', {'Naive', 'Beginner', 'Expert', 'NoSpout'}, @iscell);
    addParameter(p, 'SamplingRate', 10, @isnumeric);
    addParameter(p, 'MatchWindow', [-1, 3], @(x) isnumeric(x) && numel(x) == 2);
    addParameter(p, 'BatchSize', 1, @(x) isnumeric(x) && x >= 1);
    % NOTE: previously read `stimulus_blob_area` (largest connected component
    % on the full 39×78 lattice). That metric isn't available in FOV-crop
    % form in the aggregate. Switched to `activity_crop` (fraction of active
    % pixels in the 13×26 FOV per frame, bounded [0, 1]) so values reflect
    % the per-rec fraction-active metric.
    addParameter(p, 'YLabel', 'Fraction active (FOV crop)', @ischar);
    addParameter(p, 'MatchingMode', 'perCondition', @(x) ischar(x) && ...
        any(strcmpi(x, {'perCondition', 'global', 'expert'})));
    parse(p, varargin{:});
    opts = p.Results;

    if isempty(opts.MatcherOutputDir)
        opts.MatcherOutputDir = fullfile(opts.PerturbationResultsPath, '..', ...
            'PerturbationAnalysis', sprintf('BiasMatchExperiment_%s', opts.MatchingMode));
    end
    legacyOutDir = 'Fig. 6 Ising Model\g_BiasMatched_ConnectedSwarm';
    if strcmp(opts.OutputDir, legacyOutDir)
        opts.OutputDir = sprintf('%s_%s', legacyOutDir, opts.MatchingMode);
    end
    if ~exist(opts.OutputDir, 'dir'), mkdir(opts.OutputDir); end
    fprintf('MatchingMode    : %s\n', opts.MatchingMode);
    fprintf('Matcher input   : %s\n', opts.MatcherOutputDir);
    fprintf('Output dir      : %s\n', opts.OutputDir);

    %% Locate the perturbation aggregates (once)
    biasFile = locateAggregate(opts.PerturbationResultsPath, opts.Mode);
    clampedFile = locateAggregate(opts.PerturbationResultsPath, opts.ClampedMode);
    fprintf('Bias aggregate:    %s\n', biasFile);
    fprintf('Clamped aggregate: %s\n', clampedFile);

    %% Read aggregate-level metadata once
    [stimulusSizes, stimulusDurations, stimulusBiasValues, preStimFrames, ...
     postStimFrames, globalMeanSF] = loadAggregateMetadata(biasFile);
    secPerFrame = globalMeanSF / opts.SamplingRate;
    fprintf('Aggregate dims: sizes=%s, durs=%s, biases=%s, secPerFrame=%.4f\n', ...
        mat2str(stimulusSizes), mat2str(stimulusDurations), ...
        mat2str(stimulusBiasValues), secPerFrame);

    % Per-condition colors (paper convention)
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
        [~, sizeIdxLocal] = min(abs(stimulusSizes - sz)); %#ok<ASGLU>
        % pick duration that fits MatchWindow (mirror matcher's rule)
        durSec = stimulusDurations * secPerFrame;
        fits = find(durSec <= opts.MatchWindow(2));
        if isempty(fits), durIdxLocal = 1; else, durIdxLocal = fits(end); end
        chosenDur = stimulusDurations(durIdxLocal);
        stimEndSec = chosenDur * secPerFrame;
        fprintf('\n=== size=%d, dur=%d frames (%.2fs) ===\n', sz, chosenDur, stimEndSec);

        for wi = 1:numel(opts.Windows)
            wn = opts.Windows{wi};
            outDirSW = fullfile(opts.OutputDir, sizeKey, sprintf('window_%s', wn));
            if ~exist(outDirSW, 'dir'), mkdir(outDirSW); end
            fprintf('  window=%s → %s\n', wn, outDirSW);

            % --- Per-condition (sim, bias) selection from matcher ---
            sel = struct();
            for c = 1:numel(opts.Conditions)
                cond = opts.Conditions{c};
                bbpc = matcherOut.bestBiasPerCondition;
                if ~isfield(bbpc, cond) || ~isfield(bbpc.(cond), wn), continue; end
                bestBias = bbpc.(cond).(wn).bias_value;
                % Prefer per-window sim (set by joint-opt and bias6 post-
                % process) so each window's sim matches its bias choice.
                if isfield(bbpc.(cond).(wn), 'simIdx') && ~isnan(bbpc.(cond).(wn).simIdx)
                    bestSim = bbpc.(cond).(wn).simIdx;
                else
                    bestSim = matcherOut.bestSimIdx(c);
                end
                if isnan(bestSim), continue; end
                sel.(cond) = struct('sim', bestSim, 'bias', bestBias, ...
                    'rmse', bbpc.(cond).(wn).rmse);
            end

            % --- Two metrics: render BOTH per (size, window) ---------------
            %   1. BlobArea  — `stimulus_blob_area` (largest connected blob,
            %                  full 39×78 lattice; values in pixels^2 up to
            %                  the lattice size).
            %   2. FracActive — `activity_crop`     (mean fraction of active
            %                  pixels inside the 13×26 FOV crop, in [0, 1]).
            metricSpecs = { ...
                struct('leaf', 'stimulus_blob_area', ...
                       'tag',  'BlobArea', ...
                       'ylabel', 'Largest connected component (per rep)'); ...
                struct('leaf', 'activity_crop', ...
                       'tag',  'FracActive', ...
                       'ylabel', 'Fraction active (FOV crop, per rep)') ...
            };
            perWindowOut = struct();
            for mi = 1:numel(metricSpecs)
                spec = metricSpecs{mi};
                leafName = spec.leaf;
                metricTag = spec.tag;

                % --- Read per-cond slice ---
                blobRaster = struct();
                nFrames = NaN;
                for c = 1:numel(opts.Conditions)
                    cond = opts.Conditions{c};
                    if ~isfield(sel, cond), continue; end
                    sIdx = sel.(cond).sim - 1;   % HDF5 sim_<i> 0-indexed
                    bv = sel.(cond).bias;
                    if isinf(bv)
                        leafPath = sprintf('/experiments/%s/sim_%d/%s/size_%d/dur_%d/%s', ...
                            cond, sIdx, opts.ClampedMode, sz, chosenDur, leafName);
                        file = clampedFile;
                    else
                        bk = sprintf('bias_%s', strrep(sprintf('%.2f', bv), '.', 'p'));
                        leafPath = sprintf('/experiments/%s/sim_%d/%s/size_%d/dur_%d/%s/%s', ...
                            cond, sIdx, opts.Mode, sz, chosenDur, bk, leafName);
                        file = biasFile;
                    end
                    try
                        act = h5read(file, leafPath);
                        if size(act, 1) > size(act, 2), act = act'; end
                        if isnan(nFrames), nFrames = size(act, 2); end
                        blobRaster.(cond) = act;
                    catch ME
                        warning('Failed to read %s: %s', leafPath, ME.message);
                    end
                end
                if isempty(fieldnames(blobRaster))
                    warning('No conditions loaded for size=%d window=%s metric=%s; skipping', ...
                        sz, wn, metricTag);
                    continue;
                end

                stimOnsetFrame = preStimFrames + 1;
                simTimeAxis = ((1:nFrames) - stimOnsetFrame) * secPerFrame;
                stimMask = simTimeAxis >= 0 & simTimeAxis < stimEndSec;

                allVals = [];
                allConds = {};
                condOrder = {};
                colorMap = [];
                biasLabels = struct();
                for c = 1:numel(opts.Conditions)
                    cond = opts.Conditions{c};
                    if ~isfield(blobRaster, cond), continue; end
                    act = blobRaster.(cond)(:, stimMask);
                    perRep = mean(act, 2, 'omitnan');
                    bs = max(1, round(opts.BatchSize));
                    if bs > 1 && numel(perRep) >= bs
                        nB = floor(numel(perRep) / bs);
                        batched = zeros(nB, 1);
                        for b = 1:nB
                            idx = (b-1)*bs + (1:bs);
                            batched(b) = mean(perRep(idx), 'omitnan');
                        end
                        vals = batched;
                    else
                        vals = perRep;
                    end
                    allVals = [allVals; vals]; %#ok<AGROW>
                    allConds = [allConds; repmat({cond}, numel(vals), 1)]; %#ok<AGROW>
                    condOrder{end+1} = cond; %#ok<AGROW>
                    if isfield(condColors, cond)
                        colorMap = [colorMap; condColors.(cond)]; %#ok<AGROW>
                    else
                        colorMap = [colorMap; 0.5 0.5 0.5]; %#ok<AGROW>
                    end
                    bv = sel.(cond).bias;
                    if isinf(bv), biasLabels.(cond) = 'clamped';
                    else, biasLabels.(cond) = sprintf('%.2f', bv); end
                end
                if isempty(allVals)
                    warning('Empty swarm dataset for size=%d window=%s metric=%s; skipping', ...
                        sz, wn, metricTag);
                    continue;
                end

                figName = sprintf('Fig6g_Model_%s', metricTag);
                figure('Name', figName, 'Color', 'w');
                ax = gca;
                hold(ax, 'on');
                for ci = 1:numel(condOrder)
                    m = strcmp(allConds, condOrder{ci});
                    xVals = ci * ones(nnz(m), 1);
                    swarmchart(ax, xVals, allVals(m), 30, ...
                        colorMap(ci, :), 'filled', ...
                        'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'none', ...
                        'XJitter', 'density', 'XJitterWidth', 0.45);
                    medVal = median(allVals(m), 'omitnan');
                    plot(ax, [ci - 0.25, ci + 0.25], [medVal, medVal], ...
                        'k-', 'LineWidth', 2);
                end
                xLabelStrs = cell(1, numel(condOrder));
                for ci = 1:numel(condOrder)
                    xLabelStrs{ci} = sprintf('%s (b=%s)', condOrder{ci}, biasLabels.(condOrder{ci}));
                end
                set(ax, 'XTick', 1:numel(condOrder), ...
                        'XTickLabel', xLabelStrs, ...
                        'XLim', [0.5, numel(condOrder) + 0.5]);
                ylabel(ax, spec.ylabel);
                title(ax, sprintf('Fig 6g Model — %s — size=%d, window=%s', ...
                    metricTag, sz, wn));
                grid(ax, 'on'); box(ax, 'on');
                pvals = computePairwiseStats(allVals, allConds, condOrder);
                annotatePairwiseStats(ax, pvals, allVals, allConds, condOrder);
                hold(ax, 'off');
                pngPath = fullfile(outDirSW, [figName '.png']);
                exportgraphics(gcf, pngPath, 'Resolution', 300);
                try, exportgraphics(gcf, fullfile(outDirSW, [figName '.pdf']), 'ContentType', 'vector'); catch, end
                try, print(gcf, fullfile(outDirSW, [figName '.svg']), '-dsvg'); catch, end
                close(gcf);
                fprintf('    saved: %s\n', pngPath);
                printPvalsTable(sprintf('Fig6g Model %s size=%d window=%s', ...
                    metricTag, sz, wn), pvals);

                perWindowOut.(metricTag) = struct( ...
                    'selection', sel, 'pngPath', pngPath, ...
                    'allVals', allVals, 'allConds', {allConds}, ...
                    'condOrder', {condOrder}, 'pvals', pvals, ...
                    'biasLabels', biasLabels);
            end
            out.bySize.(sizeKey).(sprintf('window_%s', wn)) = perWindowOut;
        end
    end
    out.opts = opts;
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
