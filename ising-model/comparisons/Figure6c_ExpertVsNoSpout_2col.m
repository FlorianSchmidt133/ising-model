function Figure6c_ExpertVsNoSpout_2col(expTraces, simRasterIn, simIdxByCond, ...
    biasByCond, sz, wn, outDirSW, condColors, opts, simTimeAxis, stimEndSec, ...
    rasterCmap, variantTag)
% Figure6c_ExpertVsNoSpout_2col
%   3-row x 2-column Fig6c-style figure focused on Expert + NoSpout, using
%   sim picks supplied by the caller (typically from Figure6_ExpertVsNoSpout
%   — either the FOLD-BEST or the EXTREME-DELTA selection).
%
%     Row 1: per-recording experimental traces (faint) + bold mean
%     Row 2: model raster — imagesc over replicates of the chosen sim
%     Row 3: model mean +/- SEM (with dashed exp mean overlay in default)
%
%   Saves two sub-variants per call:
%     Fig6c_ExpertVsNoSpout_<variantTag>.{png,pdf,svg}
%       — default, full DisplayWindow, dashed exp mean in row 3
%     Fig6c_ExpertVsNoSpout_<variantTag>_sharedY.{png,pdf,svg}
%       — rows 1 and 3 share max Y; no exp overlay in row 3

    targetConds = {'Expert', 'NoSpout'};
    nC = numel(targetConds);

    haveBoth = all(cellfun(@(c) isfield(simRasterIn, c) && ~isempty(simRasterIn.(c)), targetConds));
    if ~haveBoth
        warning('Figure6c_ExpertVsNoSpout_2col(%s): missing Expert or NoSpout reps; skipping', variantTag);
        return;
    end

    % Two sub-variants (mirrors Figure6c's _sharedY pattern)
    variants(1) = struct('suffix', '',         'sharedY', false, 'noExpInRow3', false);
    variants(2) = struct('suffix', '_sharedY', 'sharedY', true,  'noExpInRow3', true);

    dispWin  = opts.DisplayWindow;
    tWinMask = simTimeAxis >= dispWin(1) & simTimeAxis <= dispWin(2);
    tWinTime = simTimeAxis(tWinMask);

    % Clim from cond-pooled 5/95 percentiles
    allVals = [];
    for ci = 1:nC
        v = simRasterIn.(targetConds{ci})(:, tWinMask);
        allVals = [allVals; v(:)]; %#ok<AGROW>
    end
    if ~isempty(allVals)
        clims = [prctile(allVals, 5), prctile(allVals, 95)];
        if clims(1) >= clims(2), clims = [0, max(allVals(:))]; end
    else
        clims = [0, 1];
    end

    for vi = 1:numel(variants)
        var = variants(vi);
        figName = sprintf('Fig6c_ExpertVsNoSpout_%s%s', variantTag, var.suffix);
        fig = figure('Name', figName, 'Color', 'w'); %#ok<NASGU>
        tl  = tiledlayout(3, nC, 'TileSpacing', 'compact', 'Padding', 'compact');

        % --- Row 1: per-recording data spaghetti + bold mean ---------------
        yMaxExp = 0;
        for c = 1:nC
            cond = targetConds{c};
            nexttile(c); hold on;
            col = condColors.(cond);
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

        % --- Row 2: model raster -------------------------------------------
        for c = 1:nC
            cond = targetConds{c};
            nexttile(nC + c);
            act = simRasterIn.(cond)(:, tWinMask);
            imagesc(tWinTime, 1:size(act, 1), act);
            colormap(gca, rasterCmap);
            clim(clims);
            set(gca, 'YDir', 'normal');
            hold on;
            xline(0, 'k--', 'LineWidth', 1.2);
            xline(stimEndSec, 'k--', 'LineWidth', 1.2);
            xlim(dispWin);
            if c == 1, ylabel('Model: replicates'); end
            bv = biasByCond.(cond);
            if isinf(bv), bvStr = 'clamped'; else, bvStr = sprintf('%.2f', bv); end
            extraTag = variantTag;
            if strcmp(variantTag, 'ExtremeDelta')
                if strcmp(cond, 'Expert'),  extraTag = 'argmax Δ';
                else,                        extraTag = 'argmin Δ'; end
            end
            text(0.02, 0.98, sprintf('sim=%d, bias=%s (%s)', ...
                    simIdxByCond.(cond), bvStr, extraTag), ...
                'Units', 'normalized', 'VerticalAlignment', 'top', ...
                'FontSize', 8, 'BackgroundColor', [1 1 1 0.7]);
            set(gca, 'XTickLabel', []);
        end

        % --- Row 3: model mean +/- SEM (+ dashed exp mean unless sharedY) --
        yMaxL = 0;
        for c = 1:nC
            cond = targetConds{c};
            nexttile(2*nC + c); hold on;
            col = condColors.(cond);
            act = simRasterIn.(cond)(:, tWinMask);
            mu  = mean(act, 1, 'omitnan');
            sem = std(act, 0, 1, 'omitnan') / sqrt(size(act, 1));
            fill([tWinTime, fliplr(tWinTime)], ...
                 [mu+sem, fliplr(mu-sem)], col, ...
                 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(tWinTime, mu, 'Color', col, 'LineWidth', 2, 'DisplayName', 'model mean');
            yMaxL = max(yMaxL, max(mu + sem));
            if ~var.noExpInRow3 && isfield(expTraces, cond) && ~isempty(expTraces.(cond).recTraces)
                et = expTraces.(cond).time;
                rt = expTraces.(cond).recTraces;
                nC2 = min(numel(et), size(rt, 2));
                et = et(1:nC2); rt = rt(:, 1:nC2);
                em = mean(rt, 1, 'omitnan');
                keep = et >= dispWin(1) & et <= dispWin(2);
                plot(et(keep), em(keep), 'k--', 'LineWidth', 1.5, 'DisplayName', 'exp mean');
                if any(keep), yMaxL = max(yMaxL, max(em(keep))); end
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

        title(tl, sprintf('Figure 6c (Expert vs NoSpout) — %s sims — size=%d, window=%s%s', ...
            variantTag, sz, wn, var.suffix), 'FontSize', 12, 'FontWeight', 'bold');

        saveStem = fullfile(outDirSW, figName);
        exportgraphics(gcf, [saveStem '.png'], 'Resolution', 300);
        try, exportgraphics(gcf, [saveStem '.pdf'], 'ContentType', 'vector'); catch, end
        try, print(gcf, [saveStem '.svg'], '-dsvg'); catch, end
        close(gcf);
        fprintf('    saved: %s.png\n', saveStem);
    end
end
