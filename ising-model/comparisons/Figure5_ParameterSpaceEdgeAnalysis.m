%% =========================================================================
%% Figure 5: Parameter Space Edge Analysis
%% =========================================================================
% This script analyzes, for each grid mode, how close the best-matching
% Ising simulations are to the edges of the parameter space, and quantifies
% match quality using Wasserstein distances for both Moran's I and Activity.
%
% Grid Modes Analyzed:
%   'downsample': Ising 32x32 resized to 13x26
%   'upsample':   Experimental 13x26 resized to 32x32
%   'normalize':  Native resolutions, normalized comparison
%   'subselect':  Center crop 13x26 from Ising 32x32
%
% Ising Parameter Grid (5D, 4 values each = 1024 combinations):
%   beta:             [0.5, 0.6, 0.7, 0.8]      Inverse temperature
%   c:                [2, 4, 6, 8]              Coupling strength
%   decay_const:      [2, 4, 6, 8]              Decay constant
%   inhibition_range: [1, 4, 9, 13]             Inhibition range
%   bias:             [-1, -0.8, -0.6, -0.4]    Bias term
%
% Key Outputs:
%   - Edge proximity metrics per condition per grid mode (Top 3 matches)
%   - Wasserstein distances for Moran's I and Activity (separately)
%   - Publication-quality visualizations
%
% Author: Generated for Florian Schmidt
% Date: 2025-01-XX

%% =========================================================================
%% SECTION 1: Configuration
%% =========================================================================

fprintf('\n=== Figure 5: Parameter Space Edge Analysis ===\n\n');

% -------------------------------------------------------------------------
% Grid Modes to Analyze
% -------------------------------------------------------------------------
config.gridModes = {'downsample', 'upsample', 'normalize', 'subselect'};

% -------------------------------------------------------------------------
% Experimental Conditions
% -------------------------------------------------------------------------
config.conditions = {'Naive', 'Beginner', 'Expert', 'NoSpout'};

% -------------------------------------------------------------------------
% Ising Parameter Grid (boundaries for edge detection)
% -------------------------------------------------------------------------
config.isingParams.beta_values = [0.5, 0.6, 0.7, 0.8];
config.isingParams.c_values = [2, 4, 6, 8];
config.isingParams.decay_const_values = [2, 4, 6, 8];
config.isingParams.inhibition_range_values = [1, 4, 9, 13];
config.isingParams.bias_values = [-1, -0.8, -0.6, -0.4];

% Parameter names for iteration
config.paramNames = {'beta', 'c', 'decay_const', 'inhibition_range', 'bias'};
config.paramLabels = {'Beta', 'Coupling (c)', 'Decay Const', 'Inhib Range', 'Bias'};

% -------------------------------------------------------------------------
% Analysis Parameters
% -------------------------------------------------------------------------
config.nTopMatches = 3;  % Number of top matches to analyze
config.edgeWarningThreshold = 0.7;  % Edge score threshold for warnings

% -------------------------------------------------------------------------
% Data Paths
% -------------------------------------------------------------------------
config.cachePath = 'Fig. 5 Model\IsingModels\IsingComparison\Data';
config.experimentalDataPath = 'Fig. 5 Model\IsingModels\Data\DataForAbir.mat';
config.outputPath = 'Fig. 5 Model\IsingModels\IsingComparison\EdgeAnalysis';

% -------------------------------------------------------------------------
% Condition Colors (for plotting)
% -------------------------------------------------------------------------
config.colors.Naive = [0.3373, 0.7059, 0.9137];     % Light blue
config.colors.Beginner = [0.8431, 0.2549, 0.6078];  % Magenta/pink
config.colors.Expert = [0, 0.6196, 0.4510];         % Teal/green
config.colors.NoSpout = [0.8353, 0.3686, 0];        % Orange

fprintf('Configuration:\n');
fprintf('  Grid modes: %s\n', strjoin(config.gridModes, ', '));
fprintf('  Conditions: %s\n', strjoin(config.conditions, ', '));
fprintf('  Top matches to analyze: %d\n', config.nTopMatches);
fprintf('  Cache path: %s\n', config.cachePath);
fprintf('  Output path: %s\n', config.outputPath);

% Create output directory
if ~exist(config.outputPath, 'dir')
    mkdir(config.outputPath);
    fprintf('\nCreated output directory: %s\n', config.outputPath);
end

%% =========================================================================
%% SECTION 2: Load Experimental Data
%% =========================================================================

fprintf('\n--- Section 2: Loading Experimental Data ---\n');

if exist(config.experimentalDataPath, 'file')
    expData = load(config.experimentalDataPath);
    fprintf('Loaded experimental data from: %s\n', config.experimentalDataPath);

    % Extract relevant fields
    BinarisedData_Exp = expData.BinarisedData;
    MoransI_Exp = expData.MoransI;

    % Compute experimental statistics for each condition
    ExpStats = struct();

    for c = 1:length(config.conditions)
        condition = config.conditions{c};

        if ~isfield(MoransI_Exp, condition) || isempty(MoransI_Exp.(condition))
            fprintf('  %s: No data available\n', condition);
            continue;
        end

        moransI = MoransI_Exp.(condition);  % [nTrials x nTimepoints]

        % Store Moran's I statistics
        ExpStats.(condition).MoransI_all = moransI(:);
        ExpStats.(condition).MoransI_mean = mean(moransI(:), 'omitnan');
        ExpStats.(condition).MoransI_std = std(moransI(:), 'omitnan');

        % Compute Activity (sparsity) from BinarisedData
        if isfield(BinarisedData_Exp, condition) && ~isempty(BinarisedData_Exp.(condition))
            binData = BinarisedData_Exp.(condition);
            % Compute mean activity per frame (mean across spatial dimensions)
            if ndims(binData) == 4
                activity = squeeze(mean(binData, [1 2]));  % [trials x timepoints]
                ExpStats.(condition).Activity_all = activity(:);
            else
                activity = squeeze(mean(binData, [1 2]));  % [frames]
                ExpStats.(condition).Activity_all = activity(:);
            end
            ExpStats.(condition).Activity_mean = mean(ExpStats.(condition).Activity_all, 'omitnan');
            ExpStats.(condition).Activity_std = std(ExpStats.(condition).Activity_all, 'omitnan');
        end

        fprintf('  %s: MoransI mean=%.4f, Activity mean=%.4f, n=%d\n', ...
            condition, ExpStats.(condition).MoransI_mean, ...
            ExpStats.(condition).Activity_mean, numel(moransI));
    end
else
    error('Experimental data file not found: %s', config.experimentalDataPath);
end

%% =========================================================================
%% SECTION 3: Loop Through Grid Modes and Compute Metrics
%% =========================================================================

fprintf('\n--- Section 3: Analyzing All Grid Modes ---\n');

% Initialize results structure
EdgeAnalysis = struct();

for g = 1:length(config.gridModes)
    gridMode = config.gridModes{g};

    fprintf('\n=== Grid Mode: %s ===\n', upper(gridMode));

    % Load cached Ising data for this grid mode
    cachePath = fullfile(config.cachePath, sprintf('IsingData_cache_%s.mat', gridMode));

    if ~exist(cachePath, 'file')
        warning('Cache file not found: %s. Skipping grid mode.', cachePath);
        continue;
    end

    cacheData = load(cachePath);
    IsingData = cacheData.IsingData;
    fprintf('Loaded %d simulations from cache\n', length(IsingData.simIDs));

    % Initialize structure for this grid mode
    EdgeAnalysis.(gridMode) = struct();

    for c = 1:length(config.conditions)
        condition = config.conditions{c};

        if ~isfield(ExpStats, condition)
            continue;
        end

        fprintf('\n  Analyzing %s...\n', condition);

        % -----------------------------------------------------------------
        % Compute Wasserstein distances for both metrics
        % -----------------------------------------------------------------
        nSims = length(IsingData.simIDs);
        wasserstein_moransI = zeros(nSims, 1);
        wasserstein_activity = zeros(nSims, 1);

        exp_moransI = ExpStats.(condition).MoransI_all;
        exp_activity = ExpStats.(condition).Activity_all;

        for s = 1:nSims
            % Moran's I distance
            ising_moransI = IsingData.MoransI_all{s}(:);
            wasserstein_moransI(s) = wasserstein_1d(exp_moransI, ising_moransI);

            % Activity distance
            ising_activity = IsingData.Activity_all{s}(:);
            wasserstein_activity(s) = wasserstein_1d(exp_activity, ising_activity);
        end

        % -----------------------------------------------------------------
        % Find Top 3 matches (using combined metric for ranking)
        % -----------------------------------------------------------------
        % Normalize distances for combined ranking
        wm_norm = (wasserstein_moransI - nanmean(wasserstein_moransI)) / nanstd(wasserstein_moransI);
        wa_norm = (wasserstein_activity - nanmean(wasserstein_activity)) / nanstd(wasserstein_activity);
        wm_norm(isnan(wm_norm)) = 0;
        wa_norm(isnan(wa_norm)) = 0;

        combined_dist = 0.5 * wm_norm + 0.5 * wa_norm;

        % Handle NaN by setting to Inf for sorting
        combined_for_sort = combined_dist;
        combined_for_sort(isnan(combined_for_sort)) = Inf;

        [~, rank_order] = sort(combined_for_sort, 'ascend');
        top3_idx = rank_order(1:config.nTopMatches);

        % -----------------------------------------------------------------
        % Extract parameters and compute edge metrics
        % -----------------------------------------------------------------
        top3_simIDs = IsingData.simIDs(top3_idx);
        top3_params = struct();
        top3_params.beta = IsingData.params.beta(top3_idx);
        top3_params.c = IsingData.params.c(top3_idx);
        top3_params.decay_const = IsingData.params.decay_const(top3_idx);
        top3_params.inhibition_range = IsingData.params.inhibition_range(top3_idx);
        top3_params.bias = IsingData.params.bias(top3_idx);

        % Compute edge proximity for each simulation
        normalized_position = zeros(config.nTopMatches, 5);
        is_at_edge = false(config.nTopMatches, 5);
        edge_count = zeros(config.nTopMatches, 1);
        edge_score = zeros(config.nTopMatches, 1);

        for m = 1:config.nTopMatches
            for p = 1:length(config.paramNames)
                paramName = config.paramNames{p};
                paramValues = config.isingParams.([paramName '_values']);
                paramMin = min(paramValues);
                paramMax = max(paramValues);

                value = top3_params.(paramName)(m);

                % Normalized position: 0 = min edge, 1 = max edge
                normalized_position(m, p) = (value - paramMin) / (paramMax - paramMin);

                % Is at edge (min or max)?
                is_at_edge(m, p) = (value == paramMin) || (value == paramMax);
            end

            % Edge count: how many parameters at boundary
            edge_count(m) = sum(is_at_edge(m, :));

            % Edge score: mean of |normalized_position - 0.5| * 2 (0=center, 1=edge)
            edge_score(m) = mean(abs(normalized_position(m, :) - 0.5)) * 2;
        end

        % -----------------------------------------------------------------
        % Store results
        % -----------------------------------------------------------------
        EdgeAnalysis.(gridMode).(condition).top3_simIDs = top3_simIDs;
        EdgeAnalysis.(gridMode).(condition).top3_params = top3_params;
        EdgeAnalysis.(gridMode).(condition).wasserstein_moransI = wasserstein_moransI(top3_idx);
        EdgeAnalysis.(gridMode).(condition).wasserstein_activity = wasserstein_activity(top3_idx);
        EdgeAnalysis.(gridMode).(condition).normalized_position = normalized_position;
        EdgeAnalysis.(gridMode).(condition).is_at_edge = is_at_edge;
        EdgeAnalysis.(gridMode).(condition).edge_count = edge_count;
        EdgeAnalysis.(gridMode).(condition).edge_score = edge_score;
        EdgeAnalysis.(gridMode).(condition).mean_edge_score = mean(edge_score);
        EdgeAnalysis.(gridMode).(condition).total_edge_hits = sum(is_at_edge(:));

        % Store all distances for later analysis
        EdgeAnalysis.(gridMode).(condition).all_wasserstein_moransI = wasserstein_moransI;
        EdgeAnalysis.(gridMode).(condition).all_wasserstein_activity = wasserstein_activity;

        % Report
        fprintf('    Top match: sim_%d (W_I=%.4f, W_A=%.4f, edge_score=%.2f)\n', ...
            top3_simIDs(1), wasserstein_moransI(top3_idx(1)), ...
            wasserstein_activity(top3_idx(1)), edge_score(1));
        fprintf('    Edge hits in top 3: %d/15, mean edge score: %.2f\n', ...
            sum(is_at_edge(:)), mean(edge_score));
    end
end

%% =========================================================================
%% SECTION 4: Publication-Quality Visualizations
%% =========================================================================

fprintf('\n--- Section 4: Creating Publication-Quality Visualizations ---\n');

% -------------------------------------------------------------------------
% Figure 1: Master Summary Table
% -------------------------------------------------------------------------
figure('Name', 'Master Summary Table');

nModes = length(config.gridModes);
nConds = length(config.conditions);

% Create data for table visualization
tableData_simID = cell(nModes, nConds);
tableData_WI = zeros(nModes, nConds);
tableData_WA = zeros(nModes, nConds);
tableData_edge = zeros(nModes, nConds);

for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(EdgeAnalysis, gridMode)
        continue;
    end
    for c = 1:nConds
        condition = config.conditions{c};
        if isfield(EdgeAnalysis.(gridMode), condition)
            res = EdgeAnalysis.(gridMode).(condition);
            tableData_simID{g, c} = sprintf('sim_%d', res.top3_simIDs(1));
            tableData_WI(g, c) = res.wasserstein_moransI(1);
            tableData_WA(g, c) = res.wasserstein_activity(1);
            tableData_edge(g, c) = res.edge_score(1);
        end
    end
end

% Create heatmap for Wasserstein Moran's I (best match)
subplot(2, 2, 1);
imagesc(tableData_WI);
colorbar;
colormap(gca, flipud(hot));
set(gca, 'XTick', 1:nConds, 'XTickLabel', config.conditions);
set(gca, 'YTick', 1:nModes, 'YTickLabel', config.gridModes);
xtickangle(45);
title('Best Match: Wasserstein (Moran''s I)');
xlabel('Condition');
ylabel('Grid Mode');

% Add text annotations
for g = 1:nModes
    for c = 1:nConds
        text(c, g, sprintf('%.3f', tableData_WI(g, c)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'w');
    end
end

% Create heatmap for Wasserstein Activity
subplot(2, 2, 2);
imagesc(tableData_WA);
colorbar;
colormap(gca, flipud(hot));
set(gca, 'XTick', 1:nConds, 'XTickLabel', config.conditions);
set(gca, 'YTick', 1:nModes, 'YTickLabel', config.gridModes);
xtickangle(45);
title('Best Match: Wasserstein (Activity)');
xlabel('Condition');
ylabel('Grid Mode');

for g = 1:nModes
    for c = 1:nConds
        text(c, g, sprintf('%.3f', tableData_WA(g, c)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'w');
    end
end

% Create heatmap for Edge Score
subplot(2, 2, 3);
imagesc(tableData_edge);
colorbar;
caxis([0 1]);
colormap(gca, parula);
set(gca, 'XTick', 1:nConds, 'XTickLabel', config.conditions);
set(gca, 'YTick', 1:nModes, 'YTickLabel', config.gridModes);
xtickangle(45);
title('Best Match: Edge Score (0=center, 1=boundary)');
xlabel('Condition');
ylabel('Grid Mode');

for g = 1:nModes
    for c = 1:nConds
        if tableData_edge(g, c) > config.edgeWarningThreshold
            color = 'r';
        else
            color = 'k';
        end
        text(c, g, sprintf('%.2f', tableData_edge(g, c)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', color, 'FontWeight', 'bold');
    end
end

% Summary text panel
subplot(2, 2, 4);
axis off;
% Find best grid mode (lowest mean Wasserstein across conditions)
meanWI_perMode = mean(tableData_WI, 2, 'omitnan');
[bestWI, bestModeIdx] = min(meanWI_perMode);
bestMode = config.gridModes{bestModeIdx};

summaryText = {
    'SUMMARY';
    '';
    sprintf('Best grid mode: %s', upper(bestMode));
    sprintf('  Mean W(I): %.4f', bestWI);
    '';
    'Edge score interpretation:';
    '  0.0 = center of parameter space';
    '  1.0 = at parameter boundary';
    sprintf('  Warning threshold: %.1f', config.edgeWarningThreshold);
    '';
    'Red values indicate boundary risk';
};
text(0.1, 0.9, summaryText, 'VerticalAlignment', 'top', 'FontSize', 10, 'FontName', 'FixedWidth');

sgtitle('Master Summary: Grid Mode x Condition Analysis');
saveMyFig('Fig1_MasterSummary', config.outputPath, gcf);

% -------------------------------------------------------------------------
% Figure 2: Wasserstein Distance Comparison (Bar Charts)
% -------------------------------------------------------------------------
figure('Name', 'Wasserstein Distance Comparison');

for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(EdgeAnalysis, gridMode)
        continue;
    end

    subplot(2, 2, g);

    % Collect data for bar chart
    WI_means = zeros(1, nConds);
    WI_stds = zeros(1, nConds);
    WA_means = zeros(1, nConds);
    WA_stds = zeros(1, nConds);

    for c = 1:nConds
        condition = config.conditions{c};
        if isfield(EdgeAnalysis.(gridMode), condition)
            WI_means(c) = mean(EdgeAnalysis.(gridMode).(condition).wasserstein_moransI);
            WI_stds(c) = std(EdgeAnalysis.(gridMode).(condition).wasserstein_moransI);
            WA_means(c) = mean(EdgeAnalysis.(gridMode).(condition).wasserstein_activity);
            WA_stds(c) = std(EdgeAnalysis.(gridMode).(condition).wasserstein_activity);
        end
    end

    % Grouped bar chart
    x = 1:nConds;
    width = 0.35;

    hold on;
    b1 = bar(x - width/2, WI_means, width, 'FaceColor', [0.2 0.4 0.8]);
    b2 = bar(x + width/2, WA_means, width, 'FaceColor', [0.8 0.4 0.2]);

    % Error bars
    errorbar(x - width/2, WI_means, WI_stds, 'k', 'LineStyle', 'none', 'LineWidth', 1);
    errorbar(x + width/2, WA_means, WA_stds, 'k', 'LineStyle', 'none', 'LineWidth', 1);

    hold off;

    set(gca, 'XTick', x, 'XTickLabel', config.conditions);
    xtickangle(45);
    ylabel('Wasserstein Distance');
    title(sprintf('%s', upper(gridMode)));
    legend({'Moran''s I', 'Activity'}, 'Location', 'best');
end

sgtitle('Wasserstein Distances: Moran''s I vs Activity (Top 3 matches, mean +/- std)');
saveMyFig('Fig2_WassersteinComparison', config.outputPath, gcf);

% -------------------------------------------------------------------------
% Figure 3: Edge Proximity Heatmap
% -------------------------------------------------------------------------
figure('Name', 'Edge Proximity Heatmap');

for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(EdgeAnalysis, gridMode)
        continue;
    end

    subplot(2, 2, g);

    % Collect mean normalized positions [conditions x parameters]
    heatmapData = zeros(nConds, 5);
    edgeHitCounts = zeros(nConds, 5);

    for c = 1:nConds
        condition = config.conditions{c};
        if isfield(EdgeAnalysis.(gridMode), condition)
            heatmapData(c, :) = mean(EdgeAnalysis.(gridMode).(condition).normalized_position, 1);
            edgeHitCounts(c, :) = sum(EdgeAnalysis.(gridMode).(condition).is_at_edge, 1);
        end
    end

    imagesc(heatmapData);
    colorbar;
    caxis([0 1]);
    colormap(gca, parula);

    set(gca, 'XTick', 1:5, 'XTickLabel', config.paramLabels);
    set(gca, 'YTick', 1:nConds, 'YTickLabel', config.conditions);
    xtickangle(45);

    title(sprintf('%s', upper(gridMode)));
    xlabel('Parameter');
    ylabel('Condition');

    % Add annotations for edge hits
    for c = 1:nConds
        for p = 1:5
            if edgeHitCounts(c, p) > 0
                text(p, c, sprintf('%d', edgeHitCounts(c, p)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 8, ...
                    'Color', 'r', 'FontWeight', 'bold');
            end
        end
    end
end

sgtitle('Edge Proximity: Mean Normalized Position (0=min, 1=max). Red numbers = edge hits in top 3');
saveMyFig('Fig3_EdgeProximityHeatmap', config.outputPath, gcf);

% -------------------------------------------------------------------------
% Figure 4: Parameter Position Distributions (Strip Plot)
% -------------------------------------------------------------------------
figure('Name', 'Parameter Position Distributions');

for p = 1:5
    subplot(2, 3, p);
    hold on;

    paramName = config.paramNames{p};
    paramValues = config.isingParams.([paramName '_values']);

    xOffset = 0;
    xTicks = [];
    xLabels = {};

    for g = 1:nModes
        gridMode = config.gridModes{g};
        if ~isfield(EdgeAnalysis, gridMode)
            continue;
        end

        for c = 1:nConds
            condition = config.conditions{c};
            if ~isfield(EdgeAnalysis.(gridMode), condition)
                continue;
            end

            xOffset = xOffset + 1;
            xTicks(end+1) = xOffset;
            xLabels{end+1} = sprintf('%s\n%s', config.conditions{c}(1:3), gridMode(1:3));

            % Get top 3 parameter values
            values = EdgeAnalysis.(gridMode).(condition).top3_params.(paramName);

            % Plot with jitter
            jitter = 0.1 * (rand(size(values)) - 0.5);
            scatter(xOffset + jitter, values, 60, config.colors.(condition), 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        end

        xOffset = xOffset + 0.5;  % Gap between grid modes
    end

    hold off;

    % Add boundary lines
    yline(min(paramValues), 'r--', 'LineWidth', 1.5);
    yline(max(paramValues), 'r--', 'LineWidth', 1.5);

    set(gca, 'XTick', xTicks, 'XTickLabel', xLabels);
    xtickangle(90);
    ylabel(config.paramLabels{p});
    title(config.paramLabels{p});
    ylim([min(paramValues) - 0.1*range(paramValues), max(paramValues) + 0.1*range(paramValues)]);
end

% Legend in 6th subplot
subplot(2, 3, 6);
axis off;
hold on;
for c = 1:nConds
    scatter(0.2, 1 - 0.2*c, 100, config.colors.(config.conditions{c}), 'filled', ...
        'MarkerEdgeColor', 'k');
    text(0.35, 1 - 0.2*c, config.conditions{c}, 'FontSize', 10);
end
text(0.1, 0.1, 'Red dashed lines = parameter boundaries', 'FontSize', 9);
hold off;
xlim([0 1]);
ylim([0 1]);
title('Legend');

sgtitle('Parameter Values of Top 3 Matches (red lines = grid boundaries)');
saveMyFig('Fig4_ParameterPositions', config.outputPath, gcf);

% -------------------------------------------------------------------------
% Figure 5: Match Quality vs Edge Proximity
% -------------------------------------------------------------------------
figure('Name', 'Match Quality vs Edge Proximity');

all_edge_scores = [];
all_wasserstein = [];
all_condition_colors = [];
all_gridmode_markers = {'o', 's', 'd', '^'};
legendEntries = {};

subplot(1, 2, 1);
hold on;

for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(EdgeAnalysis, gridMode)
        continue;
    end

    for c = 1:nConds
        condition = config.conditions{c};
        if ~isfield(EdgeAnalysis.(gridMode), condition)
            continue;
        end

        edge_scores = EdgeAnalysis.(gridMode).(condition).edge_score;
        wasserstein_I = EdgeAnalysis.(gridMode).(condition).wasserstein_moransI;

        scatter(edge_scores, wasserstein_I, 80, config.colors.(condition), ...
            all_gridmode_markers{g}, 'filled', 'MarkerEdgeColor', 'k');

        all_edge_scores = [all_edge_scores; edge_scores];
        all_wasserstein = [all_wasserstein; wasserstein_I];
    end
end

% Correlation line
if length(all_edge_scores) > 3
    [r, p] = corr(all_edge_scores, all_wasserstein, 'rows', 'complete');
    coeffs = polyfit(all_edge_scores, all_wasserstein, 1);
    x_fit = linspace(0, 1, 100);
    y_fit = polyval(coeffs, x_fit);
    plot(x_fit, y_fit, 'k--', 'LineWidth', 1.5);
    text(0.05, max(all_wasserstein)*0.95, sprintf('r=%.3f, p=%.3f', r, p), 'FontSize', 10);
end

hold off;
xlabel('Edge Score (0=center, 1=boundary)');
ylabel('Wasserstein Distance (Moran''s I)');
title('Moran''s I Match Quality vs Edge Proximity');
xlim([0 1]);

subplot(1, 2, 2);
hold on;

all_edge_scores = [];
all_wasserstein = [];

for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(EdgeAnalysis, gridMode)
        continue;
    end

    for c = 1:nConds
        condition = config.conditions{c};
        if ~isfield(EdgeAnalysis.(gridMode), condition)
            continue;
        end

        edge_scores = EdgeAnalysis.(gridMode).(condition).edge_score;
        wasserstein_A = EdgeAnalysis.(gridMode).(condition).wasserstein_activity;

        scatter(edge_scores, wasserstein_A, 80, config.colors.(condition), ...
            all_gridmode_markers{g}, 'filled', 'MarkerEdgeColor', 'k');

        all_edge_scores = [all_edge_scores; edge_scores];
        all_wasserstein = [all_wasserstein; wasserstein_A];
    end
end

% Correlation line
if length(all_edge_scores) > 3
    [r, p] = corr(all_edge_scores, all_wasserstein, 'rows', 'complete');
    coeffs = polyfit(all_edge_scores, all_wasserstein, 1);
    x_fit = linspace(0, 1, 100);
    y_fit = polyval(coeffs, x_fit);
    plot(x_fit, y_fit, 'k--', 'LineWidth', 1.5);
    text(0.05, max(all_wasserstein)*0.95, sprintf('r=%.3f, p=%.3f', r, p), 'FontSize', 10);
end

hold off;
xlabel('Edge Score (0=center, 1=boundary)');
ylabel('Wasserstein Distance (Activity)');
title('Activity Match Quality vs Edge Proximity');
xlim([0 1]);

sgtitle('Does Edge Proximity Correlate with Match Quality?');
saveMyFig('Fig5_EdgeVsMatchQuality', config.outputPath, gcf);

% -------------------------------------------------------------------------
% Figure 6: Grid Mode Comparison
% -------------------------------------------------------------------------
figure('Name', 'Grid Mode Comparison');

% Mean Wasserstein across conditions for each grid mode
meanWI = zeros(nModes, 1);
meanWA = zeros(nModes, 1);
meanEdge = zeros(nModes, 1);

for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(EdgeAnalysis, gridMode)
        continue;
    end

    WI_all = [];
    WA_all = [];
    edge_all = [];

    for c = 1:nConds
        condition = config.conditions{c};
        if isfield(EdgeAnalysis.(gridMode), condition)
            WI_all = [WI_all; EdgeAnalysis.(gridMode).(condition).wasserstein_moransI];
            WA_all = [WA_all; EdgeAnalysis.(gridMode).(condition).wasserstein_activity];
            edge_all = [edge_all; EdgeAnalysis.(gridMode).(condition).edge_score];
        end
    end

    meanWI(g) = mean(WI_all);
    meanWA(g) = mean(WA_all);
    meanEdge(g) = mean(edge_all);
end

subplot(1, 3, 1);
bar(meanWI);
set(gca, 'XTickLabel', config.gridModes);
xtickangle(45);
ylabel('Mean Wasserstein Distance');
title('Moran''s I');
[~, bestIdx] = min(meanWI);
hold on;
bar(bestIdx, meanWI(bestIdx), 'FaceColor', [0.2 0.8 0.2]);
hold off;

subplot(1, 3, 2);
bar(meanWA);
set(gca, 'XTickLabel', config.gridModes);
xtickangle(45);
ylabel('Mean Wasserstein Distance');
title('Activity');
[~, bestIdx] = min(meanWA);
hold on;
bar(bestIdx, meanWA(bestIdx), 'FaceColor', [0.2 0.8 0.2]);
hold off;

subplot(1, 3, 3);
bar(meanEdge);
set(gca, 'XTickLabel', config.gridModes);
xtickangle(45);
ylabel('Mean Edge Score');
title('Edge Score');
[~, bestIdx] = min(meanEdge);
hold on;
bar(bestIdx, meanEdge(bestIdx), 'FaceColor', [0.2 0.8 0.2]);
hold off;

sgtitle('Grid Mode Comparison (Green = Best, lower is better)');
saveMyFig('Fig6_GridModeComparison', config.outputPath, gcf);

% -------------------------------------------------------------------------
% Figure 7: Boundary Hit Summary (Actionable Recommendations)
% -------------------------------------------------------------------------
fprintf('\n--- Creating Parameter Range Extension Figures ---\n');

figure('Name', 'Boundary Hit Summary');

% Aggregate boundary hits across all grid modes and conditions
minHits = zeros(5, 1);  % Hits at MIN boundary per parameter (aggregated)
maxHits = zeros(5, 1);  % Hits at MAX boundary per parameter (aggregated)

% NEW: Per-condition boundary hits [parameters x conditions]
minHits_perCond = zeros(5, nConds);
maxHits_perCond = zeros(5, nConds);

for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(EdgeAnalysis, gridMode)
        continue;
    end

    for c = 1:nConds
        condition = config.conditions{c};
        if ~isfield(EdgeAnalysis.(gridMode), condition)
            continue;
        end

        for p = 1:5
            paramName = config.paramNames{p};
            paramValues = config.isingParams.([paramName '_values']);
            paramMin = min(paramValues);
            paramMax = max(paramValues);

            % Check each of the top 3 matches
            for m = 1:config.nTopMatches
                value = EdgeAnalysis.(gridMode).(condition).top3_params.(paramName)(m);
                if value == paramMin
                    minHits(p) = minHits(p) + 1;
                    minHits_perCond(p, c) = minHits_perCond(p, c) + 1;
                elseif value == paramMax
                    maxHits(p) = maxHits(p) + 1;
                    maxHits_perCond(p, c) = maxHits_perCond(p, c) + 1;
                end
            end
        end
    end
end

% Total possible hits per condition = nModes * nTopMatches
hitsPerCondMax = nModes * config.nTopMatches;

% Create horizontal bar chart (aggregated)
subplot(2, 2, 1);
hold on;

y = 1:5;
barh(y, -minHits, 'FaceColor', [0.8 0.2 0.2], 'DisplayName', 'MIN boundary');
barh(y, maxHits, 'FaceColor', [0.2 0.2 0.8], 'DisplayName', 'MAX boundary');

xline(0, 'k-', 'LineWidth', 1);

for p = 1:5
    if minHits(p) > 0 && maxHits(p) > 0
        recommendation = 'EXTEND BOTH';
        recColor = [0.8 0 0.8];
    elseif minHits(p) > maxHits(p)
        recommendation = 'EXTEND LOWER';
        recColor = [0.8 0.2 0.2];
    elseif maxHits(p) > minHits(p)
        recommendation = 'EXTEND UPPER';
        recColor = [0.2 0.2 0.8];
    else
        recommendation = 'OK';
        recColor = [0.2 0.7 0.2];
    end
    text(max(maxHits) + 2, p, recommendation, 'FontSize', 9, 'FontWeight', 'bold', ...
        'Color', recColor, 'HorizontalAlignment', 'left');
end

hold off;
set(gca, 'YTick', y, 'YTickLabel', config.paramLabels);
xlabel('Boundary Hits');
title('AGGREGATED (All Conditions)');
legend('Location', 'southwest');
xlim([-max([minHits; maxHits])-5, max([minHits; maxHits])+15]);

% NEW: Per-condition heatmap for MIN hits
subplot(2, 2, 2);
imagesc(minHits_perCond);
colorbar;
colormap(gca, flipud(hot));
caxis([0 hitsPerCondMax]);
set(gca, 'XTick', 1:nConds, 'XTickLabel', config.conditions);
set(gca, 'YTick', 1:5, 'YTickLabel', config.paramLabels);
xtickangle(45);
title('MIN Boundary Hits (per condition)');
xlabel('Condition');
ylabel('Parameter');
% Add text annotations
for p = 1:5
    for c = 1:nConds
        if minHits_perCond(p, c) > 0
            text(c, p, sprintf('%d', minHits_perCond(p, c)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, ...
                'Color', 'w', 'FontWeight', 'bold');
        end
    end
end

% NEW: Per-condition heatmap for MAX hits
subplot(2, 2, 3);
imagesc(maxHits_perCond);
colorbar;
colormap(gca, flipud(cool));
caxis([0 hitsPerCondMax]);
set(gca, 'XTick', 1:nConds, 'XTickLabel', config.conditions);
set(gca, 'YTick', 1:5, 'YTickLabel', config.paramLabels);
xtickangle(45);
title('MAX Boundary Hits (per condition)');
xlabel('Condition');
ylabel('Parameter');
for p = 1:5
    for c = 1:nConds
        if maxHits_perCond(p, c) > 0
            text(c, p, sprintf('%d', maxHits_perCond(p, c)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, ...
                'Color', 'k', 'FontWeight', 'bold');
        end
    end
end

% Summary text panel
subplot(2, 2, 4);
axis off;

summaryText = {'AGGREGATED SUMMARY'; ''};
for p = 1:5
    paramName = config.paramNames{p};
    paramValues = config.isingParams.([paramName '_values']);

    if minHits(p) > 0 && maxHits(p) > 0
        rec = 'EXTEND BOTH';
    elseif minHits(p) > maxHits(p)
        rec = sprintf('EXTEND LOWER (<%.2f)', min(paramValues));
    elseif maxHits(p) > minHits(p)
        rec = sprintf('EXTEND UPPER (>%.2f)', max(paramValues));
    else
        rec = 'OK';
    end

    summaryText{end+1} = sprintf('%s: %s', config.paramLabels{p}, rec);
end

text(0.05, 0.95, summaryText, 'VerticalAlignment', 'top', 'FontSize', 10, 'FontName', 'FixedWidth');

sgtitle('Boundary Hit Summary (Aggregated + Per-Condition)');
saveMyFig('Fig7_BoundaryHitSummary', config.outputPath, gcf);

% Store per-condition hits for later use
EdgeAnalysis.BoundaryHits.minHits_perCond = minHits_perCond;
EdgeAnalysis.BoundaryHits.maxHits_perCond = maxHits_perCond;

% -------------------------------------------------------------------------
% Pre-compute: Trend Analysis for ALL Grid Modes
% -------------------------------------------------------------------------
fprintf('Computing trend analysis for all grid modes...\n');

% Store trend analysis for each grid mode
TrendAnalysis_AllModes = struct();

for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(EdgeAnalysis, gridMode)
        continue;
    end

    % Load Ising params for this grid mode
    cachePath_trend = fullfile(config.cachePath, sprintf('IsingData_cache_%s.mat', gridMode));
    cacheData_trend = load(cachePath_trend);
    IsingParams_mode = cacheData_trend.IsingData.params;

    % Initialize trend analysis for this mode
    TrendAnalysis_AllModes.(gridMode).slopeAtMin = zeros(5, nConds);
    TrendAnalysis_AllModes.(gridMode).slopeAtMax = zeros(5, nConds);
    TrendAnalysis_AllModes.(gridMode).minIsAtEdge = false(5, nConds);
    TrendAnalysis_AllModes.(gridMode).recommendExtendLow = false(5, nConds);
    TrendAnalysis_AllModes.(gridMode).recommendExtendHigh = false(5, nConds);

    for p = 1:5
        paramName = config.paramNames{p};
        paramValues = config.isingParams.([paramName '_values']);
        nParamVals = length(paramValues);

        for c = 1:nConds
            condition = config.conditions{c};
            if ~isfield(EdgeAnalysis.(gridMode), condition)
                continue;
            end

            % Get all Wasserstein distances
            all_W = EdgeAnalysis.(gridMode).(condition).all_wasserstein_moransI;
            all_param_vals = IsingParams_mode.(paramName);

            % Compute mean Wasserstein for each parameter value
            meanW = zeros(1, nParamVals);
            for v = 1:nParamVals
                mask = all_param_vals == paramValues(v);
                meanW(v) = mean(all_W(mask), 'omitnan');
            end

            % Compute slopes at edges
            if nParamVals >= 2
                slopeMin = (meanW(2) - meanW(1)) / (paramValues(2) - paramValues(1));
                slopeMax = (meanW(end) - meanW(end-1)) / (paramValues(end) - paramValues(end-1));

                TrendAnalysis_AllModes.(gridMode).slopeAtMin(p, c) = slopeMin;
                TrendAnalysis_AllModes.(gridMode).slopeAtMax(p, c) = slopeMax;

                [~, minIdx] = min(meanW);
                TrendAnalysis_AllModes.(gridMode).minIsAtEdge(p, c) = (minIdx == 1) || (minIdx == nParamVals);

                % Recommend extension based on slope direction
                TrendAnalysis_AllModes.(gridMode).recommendExtendLow(p, c) = (slopeMin > 0) || (minIdx == 1);
                TrendAnalysis_AllModes.(gridMode).recommendExtendHigh(p, c) = (slopeMax < 0) || (minIdx == nParamVals);
            end
        end
    end

    fprintf('  %s: trend analysis complete\n', gridMode);
end

% Store in EdgeAnalysis
EdgeAnalysis.TrendAnalysis_AllModes = TrendAnalysis_AllModes;

% -------------------------------------------------------------------------
% Figure 8a-d: Match Quality vs Parameter Value (1D Sensitivity + Trend Analysis)
% One figure per grid mode
% -------------------------------------------------------------------------
fprintf('\nCreating per-grid-mode parameter sensitivity figures...\n');

for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(EdgeAnalysis, gridMode) || ~isfield(TrendAnalysis_AllModes, gridMode)
        continue;
    end

    figure('Name', sprintf('Parameter Sensitivity: %s', upper(gridMode)));

    % Load Ising params for this mode
    cachePath_fig8 = fullfile(config.cachePath, sprintf('IsingData_cache_%s.mat', gridMode));
    cacheData_fig8 = load(cachePath_fig8);
    IsingParams_mode = cacheData_fig8.IsingData.params;

    % Get pre-computed trend analysis for this mode
    trendAnalysis = TrendAnalysis_AllModes.(gridMode);

    for p = 1:5
        subplot(2, 3, p);
        hold on;

        paramName = config.paramNames{p};
        paramValues = config.isingParams.([paramName '_values']);
        nParamVals = length(paramValues);

        % For each condition, plot Wasserstein vs parameter value using ALL simulations
        for c = 1:nConds
            condition = config.conditions{c};
            if ~isfield(EdgeAnalysis.(gridMode), condition)
                continue;
            end

            % Get all Wasserstein distances and corresponding parameter values
            all_W = EdgeAnalysis.(gridMode).(condition).all_wasserstein_moransI;
            all_param_vals = IsingParams_mode.(paramName);

            % Compute mean Wasserstein for each parameter value
            meanW = zeros(1, nParamVals);
            semW = zeros(1, nParamVals);

            for v = 1:nParamVals
                mask = all_param_vals == paramValues(v);
                meanW(v) = mean(all_W(mask), 'omitnan');
                semW(v) = std(all_W(mask), 'omitnan') / sqrt(sum(mask));
            end

            % Plot with error bars
            errorbar(paramValues, meanW, semW, 'o-', 'Color', config.colors.(condition), ...
                'MarkerFaceColor', config.colors.(condition), 'LineWidth', 1.5, ...
                'MarkerSize', 8, 'DisplayName', condition);
        end

        hold off;

        % Add boundary markers
        xline(min(paramValues), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
        xline(max(paramValues), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');

        xlabel(config.paramLabels{p});
        ylabel('Mean Wasserstein (I)');
        title(config.paramLabels{p});

        if p == 1
            legend('Location', 'best');
        end

        % Add per-condition extension indicators
        yRange = ylim;
        yTop = yRange(2);
        yStep = (yRange(2) - yRange(1)) * 0.08;

        for c = 1:nConds
            condition = config.conditions{c};
            condColor = config.colors.(condition);

            % Add small arrows/markers at edges if extension recommended for this condition
            if trendAnalysis.recommendExtendLow(p, c)
                % Small left arrow at MIN edge
                text(paramValues(1), yTop - c*yStep, sprintf('%s<', condition(1)), ...
                    'Color', condColor, 'FontWeight', 'bold', 'FontSize', 8, ...
                    'HorizontalAlignment', 'center');
            end
            if trendAnalysis.recommendExtendHigh(p, c)
                % Small right arrow at MAX edge
                text(paramValues(end), yTop - c*yStep, sprintf('>%s', condition(1)), ...
                    'Color', condColor, 'FontWeight', 'bold', 'FontSize', 8, ...
                    'HorizontalAlignment', 'center');
            end
        end
    end

    % Legend and info in 6th subplot
    subplot(2, 3, 6);
    axis off;
    hold on;
    for c = 1:nConds
        plot(NaN, NaN, 'o-', 'Color', config.colors.(config.conditions{c}), ...
            'MarkerFaceColor', config.colors.(config.conditions{c}), 'LineWidth', 1.5);
    end
    hold off;
    legend(config.conditions, 'Location', 'north');

    infoText = {
        'Red dashed lines = boundaries';
        '';
        'Letters at edges indicate';
        'per-condition extension need, e.g.:';
        '  E< = Expert needs LOWER END';
        '  >E = Expert needs HIGHER END';
        '';
        'Based on slope at edge';
        '(declining toward boundary)';
    };
    text(0.5, 0.35, infoText, 'HorizontalAlignment', 'center', 'FontSize', 8);

    sgtitle(sprintf('%s Mode - Match Quality vs Parameter Value', upper(gridMode)));
    saveMyFig(sprintf('Fig8%s_ParameterSensitivity_%s', char('a' + g - 1), gridMode), config.outputPath, gcf);

    fprintf('  Created Figure 8%s for %s\n', char('a' + g - 1), gridMode);
end

% Store trend analysis in EdgeAnalysis (keep all modes)
EdgeAnalysis.TrendAnalysis = TrendAnalysis_AllModes;
EdgeAnalysis.TrendAnalysis_paramNames = config.paramNames;
EdgeAnalysis.TrendAnalysis_conditions = config.conditions;

% -------------------------------------------------------------------------
% Figure 9: Parameter Extension Priority Matrix
% -------------------------------------------------------------------------
figure('Name', 'Parameter Extension Priority Matrix');

% Compute extension pressure: +1 if at MAX, -1 if at MIN, 0 if interior
% Aggregated across top 3 matches and all grid modes
extensionPressure = zeros(5, nConds);

for p = 1:5
    paramName = config.paramNames{p};
    paramValues = config.isingParams.([paramName '_values']);
    paramMin = min(paramValues);
    paramMax = max(paramValues);

    for c = 1:nConds
        condition = config.conditions{c};
        pressure = 0;
        count = 0;

        for g = 1:nModes
            gridMode = config.gridModes{g};
            if ~isfield(EdgeAnalysis, gridMode) || ~isfield(EdgeAnalysis.(gridMode), condition)
                continue;
            end

            for m = 1:config.nTopMatches
                value = EdgeAnalysis.(gridMode).(condition).top3_params.(paramName)(m);
                if value == paramMax
                    pressure = pressure + 1;
                elseif value == paramMin
                    pressure = pressure - 1;
                end
                count = count + 1;
            end
        end

        % Normalize by count
        if count > 0
            extensionPressure(p, c) = pressure / count;
        end
    end
end

% Create heatmap
imagesc(extensionPressure);
colorbar;
caxis([-1 1]);

% Red-White-Blue colormap
n = 64;
red = [ones(n/2, 1); linspace(1, 0, n/2)'];
blue = [linspace(0, 1, n/2)'; ones(n/2, 1)];
green = [linspace(0, 1, n/2)'; linspace(1, 0, n/2)'];
rwb_cmap = [red, green, blue];
colormap(gca, rwb_cmap);

set(gca, 'XTick', 1:nConds, 'XTickLabel', config.conditions);
set(gca, 'YTick', 1:5, 'YTickLabel', config.paramLabels);
xtickangle(45);

xlabel('Condition');
ylabel('Parameter');
title('Extension Pressure: Red = EXTEND LOWER, Blue = EXTEND UPPER, White = OK');

% Add text annotations
for p = 1:5
    for c = 1:nConds
        val = extensionPressure(p, c);
        if abs(val) > 0.3
            textColor = 'w';
        else
            textColor = 'k';
        end
        text(c, p, sprintf('%.2f', val), 'HorizontalAlignment', 'center', ...
            'FontSize', 9, 'Color', textColor, 'FontWeight', 'bold');
    end
end

sgtitle('Parameter Extension Priority Matrix');
saveMyFig('Fig9_ExtensionPriorityMatrix', config.outputPath, gcf);

% -------------------------------------------------------------------------
% Figure 10a-d: Separate Recommendations for Each Grid Mode
% -------------------------------------------------------------------------
fprintf('\nCreating per-grid-mode recommendation figures...\n');

% Store per-mode boundary hits [parameters x conditions x modes]
minHits_perMode = zeros(5, nConds, nModes);
maxHits_perMode = zeros(5, nConds, nModes);

% Compute per-mode boundary hits
for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(EdgeAnalysis, gridMode)
        continue;
    end

    for c = 1:nConds
        condition = config.conditions{c};
        if ~isfield(EdgeAnalysis.(gridMode), condition)
            continue;
        end

        for p = 1:5
            paramName = config.paramNames{p};
            paramValues = config.isingParams.([paramName '_values']);
            paramMin = min(paramValues);
            paramMax = max(paramValues);

            for m = 1:config.nTopMatches
                value = EdgeAnalysis.(gridMode).(condition).top3_params.(paramName)(m);
                if value == paramMin
                    minHits_perMode(p, c, g) = minHits_perMode(p, c, g) + 1;
                elseif value == paramMax
                    maxHits_perMode(p, c, g) = maxHits_perMode(p, c, g) + 1;
                end
            end
        end
    end
end

% Store recommendation matrices for each mode
RecommendedRanges_AllModes = struct();

% Red-White-Blue colormap (shared)
n = 64;
red = [ones(n/2, 1); linspace(1, 0, n/2)'];
blue = [linspace(0, 1, n/2)'; ones(n/2, 1)];
green = [linspace(0, 1, n/2)'; linspace(1, 0, n/2)'];
rwb_cmap = [red, green, blue];

% Create a figure for each grid mode
for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(EdgeAnalysis, gridMode) || ~isfield(TrendAnalysis_AllModes, gridMode)
        continue;
    end

    figure('Name', sprintf('Recommendations: %s', upper(gridMode)));

    % Get trend analysis for this mode
    trendMode = TrendAnalysis_AllModes.(gridMode);

    % Compute recommendation matrix for this mode
    % Uses ONLY this mode's boundary hits and trend analysis
    recommendMatrix_mode = zeros(5, nConds);

    for p = 1:5
        for c = 1:nConds
            score = 0;

            % Evidence from boundary hits (this mode only)
            if minHits_perMode(p, c, g) > 0
                score = score - 1;
            end
            if maxHits_perMode(p, c, g) > 0
                score = score + 1;
            end

            % Evidence from trend analysis (this mode only)
            if trendMode.recommendExtendLow(p, c)
                score = score - 1;
            end
            if trendMode.recommendExtendHigh(p, c)
                score = score + 1;
            end

            recommendMatrix_mode(p, c) = score;
        end
    end

    % LEFT PANEL: Recommendation heatmap
    subplot(1, 2, 1);

    imagesc(recommendMatrix_mode);
    colorbar;
    caxis([-2 2]);
    colormap(gca, rwb_cmap);

    set(gca, 'XTick', 1:nConds, 'XTickLabel', config.conditions);
    set(gca, 'YTick', 1:5, 'YTickLabel', config.paramLabels);
    xtickangle(45);
    xlabel('Condition');
    ylabel('Parameter');
    title({'Per-Condition Recommendations'; '(Red=EXTEND LOWER, Blue=EXTEND UPPER)'});

    % Add text annotations
    for p = 1:5
        for c = 1:nConds
            val = recommendMatrix_mode(p, c);
            if val <= -2
                label = 'LOWER!';
                textColor = 'w';
            elseif val == -1
                label = 'lower';
                textColor = 'w';
            elseif val >= 2
                label = 'UPPER!';
                textColor = 'w';
            elseif val == 1
                label = 'upper';
                textColor = 'k';
            else
                label = 'ok';
                textColor = 'k';
            end
            text(c, p, label, 'HorizontalAlignment', 'center', ...
                'FontSize', 9, 'Color', textColor, 'FontWeight', 'bold');
        end
    end

    % RIGHT PANEL: Summary text
    subplot(1, 2, 2);
    axis off;

    recommendationText = {sprintf('GRID MODE: %s', upper(gridMode)); ''; ...
        '----------------------------------------'; ''};

    % Store results for this mode
    RecommendedRanges_AllModes.(gridMode) = struct();

    for p = 1:5
        paramName = config.paramNames{p};
        paramValues = config.isingParams.([paramName '_values']);
        currentMin = min(paramValues);
        currentMax = max(paramValues);
        currentStep = paramValues(2) - paramValues(1);

        nExtendLow = sum(recommendMatrix_mode(p, :) < 0);
        nExtendHigh = sum(recommendMatrix_mode(p, :) > 0);

        extendLow = nExtendLow > 0;
        extendHigh = nExtendHigh > 0;

        % Calculate new range
        if extendLow && extendHigh
            newMin = currentMin - 2 * abs(currentStep);
            newMax = currentMax + 2 * abs(currentStep);
            priority = 'HIGH';
        elseif extendLow
            newMin = currentMin - 2 * abs(currentStep);
            newMax = currentMax;
            priority = 'MEDIUM';
        elseif extendHigh
            newMin = currentMin;
            newMax = currentMax + 2 * abs(currentStep);
            priority = 'MEDIUM';
        else
            newMin = currentMin;
            newMax = currentMax;
            priority = 'LOW';
        end

        % Store for this mode
        RecommendedRanges_AllModes.(gridMode).(paramName).current = [currentMin, currentMax];
        RecommendedRanges_AllModes.(gridMode).(paramName).recommended = [newMin, newMax];
        RecommendedRanges_AllModes.(gridMode).(paramName).priority = priority;
        RecommendedRanges_AllModes.(gridMode).(paramName).conditionsNeedLow = config.conditions(recommendMatrix_mode(p, :) < 0);
        RecommendedRanges_AllModes.(gridMode).(paramName).conditionsNeedHigh = config.conditions(recommendMatrix_mode(p, :) > 0);

        % Format for display
        recommendationText{end+1} = sprintf('%s:', config.paramLabels{p});

        if extendLow || extendHigh
            if nExtendLow > 0
                condList = strjoin(RecommendedRanges_AllModes.(gridMode).(paramName).conditionsNeedLow, ',');
                recommendationText{end+1} = sprintf('  LOWER: %s', condList);
            end
            if nExtendHigh > 0
                condList = strjoin(RecommendedRanges_AllModes.(gridMode).(paramName).conditionsNeedHigh, ',');
                recommendationText{end+1} = sprintf('  UPPER: %s', condList);
            end
        else
            recommendationText{end+1} = '  -> OK';
        end
    end

    text(0.05, 0.95, recommendationText, 'VerticalAlignment', 'top', 'FontSize', 9, ...
        'FontName', 'FixedWidth', 'Interpreter', 'none');

    % Store the matrix
    RecommendedRanges_AllModes.(gridMode).recommendMatrix = recommendMatrix_mode;

    sgtitle(sprintf('%s Mode - Parameter Extension Recommendations', upper(gridMode)));
    saveMyFig(sprintf('Fig10%s_Recommendations_%s', char('a' + g - 1), gridMode), config.outputPath, gcf);

    fprintf('  Created Figure 10%s for %s\n', char('a' + g - 1), gridMode);
end

% Store all results
EdgeAnalysis.RecommendedRanges_AllModes = RecommendedRanges_AllModes;
EdgeAnalysis.minHits_perMode = minHits_perMode;
EdgeAnalysis.maxHits_perMode = maxHits_perMode;

% Also keep aggregated version for backward compatibility
RecommendedRanges = struct();
RecommendedRanges.conditions = config.conditions;
RecommendedRanges.paramNames = config.paramNames;

% Use best mode's recommendations as the primary
bestGridMode = config.gridModes{bestModeIdx};
recommendMatrix = RecommendedRanges_AllModes.(bestGridMode).recommendMatrix;

EdgeAnalysis.RecommendedRanges = RecommendedRanges;
EdgeAnalysis.BoundaryHits.minHits = minHits;
EdgeAnalysis.BoundaryHits.maxHits = maxHits;
EdgeAnalysis.BoundaryHits.paramNames = config.paramNames;
EdgeAnalysis.BoundaryHits.paramLabels = config.paramLabels;

fprintf('\nAll figures saved to: %s\n', config.outputPath);

%% =========================================================================
%% SECTION 5: Console Summary Report
%% =========================================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('  PARAMETER SPACE EDGE ANALYSIS COMPLETE\n');
fprintf('========================================\n');
fprintf('\n');

% Find best grid mode
[~, bestModeIdx] = min(meanWI);
fprintf('RECOMMENDATION:\n');
fprintf('  Best grid mode: %s\n', upper(config.gridModes{bestModeIdx}));
fprintf('  Mean Wasserstein(I): %.4f\n', meanWI(bestModeIdx));
fprintf('  Mean Wasserstein(A): %.4f\n', meanWA(bestModeIdx));
fprintf('  Mean Edge Score: %.2f\n', meanEdge(bestModeIdx));
fprintf('\n');

% Check for boundary warnings
fprintf('BOUNDARY WARNINGS (edge score > %.1f):\n', config.edgeWarningThreshold);
warningFound = false;

for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(EdgeAnalysis, gridMode)
        continue;
    end

    for c = 1:nConds
        condition = config.conditions{c};
        if ~isfield(EdgeAnalysis.(gridMode), condition)
            continue;
        end

        if EdgeAnalysis.(gridMode).(condition).mean_edge_score > config.edgeWarningThreshold
            fprintf('  WARNING: %s/%s has edge score %.2f\n', ...
                gridMode, condition, EdgeAnalysis.(gridMode).(condition).mean_edge_score);
            warningFound = true;

            % Show which parameters are at edge
            for m = 1:config.nTopMatches
                edgeParams = config.paramNames(EdgeAnalysis.(gridMode).(condition).is_at_edge(m, :));
                if ~isempty(edgeParams)
                    fprintf('    sim_%d: at edge for %s\n', ...
                        EdgeAnalysis.(gridMode).(condition).top3_simIDs(m), ...
                        strjoin(edgeParams, ', '));
                end
            end
        end
    end
end

if ~warningFound
    fprintf('  None - all best matches are away from parameter boundaries\n');
end

fprintf('\n');

% Detailed results per grid mode
fprintf('DETAILED RESULTS:\n');
for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(EdgeAnalysis, gridMode)
        continue;
    end

    fprintf('\n  %n', upper(gridMode));
    fprintf('  %10s  %8s  %8s  %8s  %10s  %8s\n', ...
        'Condition', 'W(I)', 'W(A)', 'EdgeScr', 'EdgeHits', 'BestSim');
    fprintf('  %s\n', repmat('-', 1, 60));

    for c = 1:nConds
        condition = config.conditions{c};
        if ~isfield(EdgeAnalysis.(gridMode), condition)
            continue;
        end

        res = EdgeAnalysis.(gridMode).(condition);
        fprintf('  %10s  %8.4f  %8.4f  %8.2f  %10d  sim_%d\n', ...
            condition, res.wasserstein_moransI(1), res.wasserstein_activity(1), ...
            res.mean_edge_score, res.total_edge_hits, res.top3_simIDs(1));
    end
end

fprintf('\n');

% -------------------------------------------------------------------------
% PER-MODE EXTENSION RECOMMENDATIONS
% -------------------------------------------------------------------------
fprintf('PARAMETER EXTENSION RECOMMENDATIONS BY GRID MODE:\n');
fprintf('================================================================================\n');
fprintf('(Based on boundary hits + trend analysis for EACH mode separately)\n');
fprintf('Scores: LOWER!/lower = extend below min, UPPER!/upper = extend above max, ok = sufficient\n\n');

for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(RecommendedRanges_AllModes, gridMode)
        continue;
    end

    recMatrix = RecommendedRanges_AllModes.(gridMode).recommendMatrix;

    fprintf('--- %s ---\n', upper(gridMode));
    fprintf('  %15s  ', 'Parameter');
    for c = 1:nConds
        fprintf('%10s  ', config.conditions{c});
    end
    fprintf('\n');
    fprintf('  %s\n', repmat('-', 1, 15 + 12*nConds));

    for p = 1:5
        fprintf('  %15s  ', config.paramLabels{p});
        for c = 1:nConds
            val = recMatrix(p, c);
            if val <= -2
                label = 'LOWER!';
            elseif val == -1
                label = 'lower';
            elseif val >= 2
                label = 'UPPER!';
            elseif val == 1
                label = 'upper';
            else
                label = 'ok';
            end
            fprintf('%10s  ', label);
        end
        fprintf('\n');
    end
    fprintf('\n');
end

fprintf('================================================================================\n');
fprintf('\n');

% Show per-mode detailed breakdown
fprintf('DETAILED PER-MODE BREAKDOWN:\n');
for g = 1:nModes
    gridMode = config.gridModes{g};
    if ~isfield(RecommendedRanges_AllModes, gridMode)
        continue;
    end

    recMatrix = RecommendedRanges_AllModes.(gridMode).recommendMatrix;
    hasRecommendations = false;

    for p = 1:5
        paramName = config.paramNames{p};
        condLow = config.conditions(recMatrix(p, :) < 0);
        condHigh = config.conditions(recMatrix(p, :) > 0);

        if ~isempty(condLow) || ~isempty(condHigh)
            if ~hasRecommendations
                fprintf('\n  %n', upper(gridMode));
                hasRecommendations = true;
            end
            fprintf('    %n', config.paramLabels{p});
            if ~isempty(condLow)
                fprintf('      EXTEND LOWER END for: %s\n', strjoin(condLow, ', '));
            end
            if ~isempty(condHigh)
                fprintf('      EXTEND UPPER END for: %s\n', strjoin(condHigh, ', '));
            end
        end
    end
end
fprintf('\n');

% Copy-paste ready config (using best mode)
fprintf('================================================================================\n');
fprintf('COPY-PASTE FOR NEW SIMULATION CONFIG (based on %s mode):\n', upper(bestGridMode));
fprintf('--------------------------------------\n');
for p = 1:5
    paramName = config.paramNames{p};
    if isfield(RecommendedRanges_AllModes.(bestGridMode), paramName)
        newRange = RecommendedRanges_AllModes.(bestGridMode).(paramName).recommended;
    else
        paramValues = config.isingParams.([paramName '_values']);
        newRange = [min(paramValues), max(paramValues)];
    end
    currentStep = config.isingParams.([paramName '_values'])(2) - config.isingParams.([paramName '_values'])(1);
    newValues = newRange(1):currentStep:newRange(2);
    fprintf('%s_values = %s;\n', paramName, mat2str(newValues, 3));
end
fprintf('\n');
fprintf('========================================\n');

%% =========================================================================
%% SECTION 6: Save Results
%% =========================================================================

fprintf('\n--- Section 6: Saving Results ---\n');

Results = struct();
Results.config = config;
Results.ExpStats = ExpStats;
Results.EdgeAnalysis = EdgeAnalysis;
Results.timestamp = datetime('now');

% Summary metrics for quick access
Results.Summary.meanWI_perMode = meanWI;
Results.Summary.meanWA_perMode = meanWA;
Results.Summary.meanEdge_perMode = meanEdge;
Results.Summary.bestGridMode = config.gridModes{bestModeIdx};

resultsPath = fullfile(config.outputPath, 'EdgeAnalysis_Results.mat');
save(resultsPath, 'Results', '-v7.3');
fprintf('Results saved to: %s\n', resultsPath);

fprintf('\n=== Analysis Complete ===\n');

%% =========================================================================
%% Helper Functions
%% =========================================================================

function d = wasserstein_1d(x, y)
    % Compute 1D Wasserstein (Earth Mover's) distance between two samples
    % Uses efficient quantile-based approach for large arrays

    % Remove NaN values
    x = x(:);
    y = y(:);
    x = x(~isnan(x));
    y = y(~isnan(y));

    % Handle empty inputs
    if isempty(x) || isempty(y)
        d = NaN;
        return;
    end

    % Sort both samples
    x = sort(x);
    y = sort(y);

    % Use quantile matching for efficiency with large arrays
    % Sample at most 1000 quantiles to avoid memory issues
    n = min(1000, min(length(x), length(y)));
    q = linspace(0, 1, n);

    x_quantiles = quantile(x, q);
    y_quantiles = quantile(y, q);

    % Wasserstein distance = mean absolute difference between quantile functions
    d = mean(abs(x_quantiles - y_quantiles));
end
