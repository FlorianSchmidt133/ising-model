%% =========================================================================
%% Figure 5: Threshold for Binarisation Determination
%% =========================================================================
% This script analyzes activity data distributions to determine optimal
% binarization thresholds for Figure 5 Ising models.
%
% Analysis includes:
% - Raw activity value distributions (ActivityData and Grid40)
% - Mean activity per neuron/grid cell distributions
% - Condition-wise comparisons (Naive, Beginner, Expert, NoSpout)
% - Animal-level comparisons
% - Aggregated distributions across all conditions
%
% Plot types: Histogram, CDF, Violin, KDE density

%% =========================================================================
%% SECTION 1: Setup and Parameters
%% =========================================================================



% -------------------------------------------------------------------------
% Per-section RUN flags — toggle which parts of the script execute.
% Sections not in this struct (Section 1 setup + Section 2 ActivityData)
% always run; everything else is gated.
%
% Quick recipes:
%
%   * Full run (default):
%       Leave defaults as-is. Everything runs EXCEPT section3_stimlocked
%       (the expensive Plots 3.7–3.9) — flip that one to true when you
%       need to regenerate those plots.
%
%   * Regenerate only Section 3 cheap plots (Plots 3.1–3.6) + save:
%       RUN.section3 = true; all others false; RUN.save_figures = true.
%
%   * Regenerate the expensive stim-locked/Zipf plots (3.7–3.9):
%       RUN.section3 = true; RUN.section3_stimlocked = true; rest false.
%
%   * Regenerate Section 8 plots only, using cached data (Step 1 skipped):
%       Load('EntropyPreservation_Results.mat') into the workspace first.
%       Then: RUN.section8 = true; RUN.section8_compute = false;
%             RUN.section8_plots = true; rest false; RUN.save_figures = true.
%
%   * Dependency: Section 8 needs AnimalStats_VarExplained from Section 7
%     Step 1. The dependency is auto-resolved below (Section 7 is forced
%     on whenever Section 8 is on AND section8_compute is true).
% -------------------------------------------------------------------------
RUN                        = struct();
RUN.section3               = 0;   % Grid40 Raw distribution (Plots 3.1–3.6)
RUN.section3_stimlocked    = 0;  % Plots 3.7 (PSTH), 3.8 (SNR), 3.9 (Zipf) — expensive
RUN.section4               = 0;   % Variance/Sigma per-grid-cell, per-timepoint, per-trial
RUN.section5               = 0;   % Animal-level CDF plots
RUN.section6               = 0;   % Visual Threshold Comparison (Expert Rec 5 Trial 1)
RUN.section7               = 0;   % Variance Explained (R², VarRet, NormRecon)
RUN.section7_abs_vs_sigma  = 0;   % 2σ-vs-2 dF/F comparison plots (Step 3b/3c)
RUN.section8               = 0;   % Raw vs Binarized Entropy Comparison (requires section7)
RUN.section8_compute       = 0;   % Section 8 data generation (Step 1 per-frame loop — SLOW)
RUN.section8_plots         = 0;   % Section 8 visualisations (Step 4: heatmaps, traces, etc.)
RUN.section9               = 1;   % NMF vs threshold-2 comparison (requires Grid40 + NMF file)
RUN.save_figures           = 1;   % saveMyFig + write EntropyPreservation_Results.mat

% Auto-resolve dependencies
% section7_abs_vs_sigma (Step 3b/3c) reads Results_VarExplained (built in
% Section 7 Step 2) for the 2σ comparison panel, so Section 7 must also run.
if RUN.section7_abs_vs_sigma && ~RUN.section7
    warning('section7_abs_vs_sigma reads Results_VarExplained — forcing section7=true');
    RUN.section7 = true;
end
% section8_compute reads AnimalStats_VarExplained (built in Section 7 Step 1).
if RUN.section8 && RUN.section8_compute && ~RUN.section7
    warning('Section 8 compute requires Section 7 Step 1 stats — forcing section7=true');
    RUN.section7 = true;
end

fprintf('\n=== Figure 5: Threshold for Binarisation Determination ===\n');

% Load data structures
% load(mba_p('RawData3.mat'),'ActivityData');
% load(mba_p("Grid40.mat"))
% load(mba_p('RawData3.mat'),'Rec');
% load(mba_p('RawData3.mat'),'params');

% Define Skip arrays (recordings to exclude from analysis)
Skip = [];
Skip.Naive = [1 9 10 16];
Skip.Beginner = [1 6 7 11];
Skip.Expert = [1 4 12 13 14];
Skip.NoSpout = [1 4 9 10 11 13 14];
Skip.ExpertRandom = [1];
Skip.ExpertAll = [1,4,5,13, 20,21,22,23,24,25,26];

% Filter flag: Remove empty/dead grid cells (values between 0 and 0.2)
Remove_Low_Values = true;  % Set to false to include all values
DEAD_CELL_THRESHOLD = 0.2; % Values below this are considered dead/empty cells

% GMM fitting flag: Fit Gaussian Mixture Models to mirrored distributions
fitGMM = false;  % Set to true to fit and visualize GMM components


if Remove_Low_Values
    fprintf('Filter enabled: Removing values < %.2f (dead/empty cells)\n', DEAD_CELL_THRESHOLD);
else
    fprintf('Filter disabled: Including all values\n');
end

% Define conditions to analyze
conditions = {'Naive', 'Beginner', 'Expert', 'NoSpout'};

fprintf('\nRUN flags:\n');
disp(RUN);

% Set up condition colors from params
conditionColors = struct();
conditionColors.Naive = params.Naive;
conditionColors.Beginner = params.Beginner;
conditionColors.Expert = params.Expert;
conditionColors.NoSpout = params.NoSpout;

% Define Individual condition names for Grid40
params.NaiveIndividual = params.Naive;
params.BeginnerIndividual = params.Beginner;
params.ExpertIndividual = params.Expert;
params.NoSpoutIndividual = params.NoSpout;

fprintf('Conditions to analyze: %s\n', strjoin(conditions, ', '));
fprintf('Position: P1 (Position1)\n');

% Define standardized bin evaluation points for consistent comparisons across all plots
kde_xi_5 = linspace(0, 5, 201);   % For full activity range plots (0-5)
kde_xi_2 = linspace(0, 2, 201);   % For mean activity range plots (0-2)
kde_xi_3 = linspace(0, 3, 201);   % For sigma/variance range plots (0-3)
histEdges_5 = linspace(0, 5, 201); % For histograms in 0-5 range
histEdges_2 = linspace(0, 2, 201); % For histograms in 0-2 range
fprintf('Bin definitions: kde_xi_5 (0-5), kde_xi_2 (0-2), kde_xi_3 (0-3), histEdges_5, histEdges_2\n');

% Define timeframe for Section 4 variance analysis
TimeFrameSelection = 1:80;  % Frames to include (e.g., 1:80 for pre-stimulus only)
% Options: 1:80 (pre-stim), 81:100 (stim), 1:85 (all frames)
fprintf('Section 4 timeframe selection: frames %d to %d\n', TimeFrameSelection(1), TimeFrameSelection(end));

%% =========================================================================
%% INTERPRETATION GUIDE: How These Plots Help Determine Threshold
%% =========================================================================
%
% PURPOSE OF BINARIZATION:
% Convert continuous neural activity into binary states (active/inactive)
% for Ising model analysis. Threshold determines what counts as "active".
%
% KEY QUESTIONS TO ASK:
%
% 1. HISTOGRAM (0-10 range):
%    - Is distribution unimodal (one peak) or bimodal (two peaks)?
%    - Where is most data concentrated? (indicates "baseline" activity)
%    - Bimodal = natural threshold between peaks
%    - Unimodal = need statistical criterion (e.g., 2σ above mean)
%
% 2. CDF PLOTS (Most important for threshold selection):
%    - Y-value at any X shows proportion of data BELOW that threshold
%    - Example: CDF(x=1.5σ) = 0.95 means 5% of data ABOVE 1.5σ
%    - Choose threshold based on desired sparsity (typically 1-10% active)
%    - Steeper slope = more data concentrated in that range
%
% 3. THRESHOLD REFERENCE LINES:
%    - Dashed lines (--): Standard deviation multiples (σ)
%      * 1.0σ ≈ 84th percentile (16% active)
%      * 1.5σ ≈ 93rd percentile (7% active)
%      * 2.0σ ≈ 97.7th percentile (2.3% active)
%    - Dotted lines (:): Direct percentiles for comparison
%    - Solid line: Mean (0σ baseline)
%
% 4. DENSITY (KDE) PLOTS:
%    - Shows smoothed probability density
%    - Peak location = most common activity level
%    - Helpful for identifying multimodal distributions
%
% 5. CONDITION COMPARISON:
%    - Do different conditions have similar distributions?
%    - If YES: Global threshold appropriate
%    - If NO: Consider condition-specific or per-animal normalization (BIN_AN)
%
% 6. ANIMAL COMPARISON:
%    - High variability between animals?
%    - If YES: Per-animal normalization (BIN_AN) recommended
%    - If NO: Global threshold across animals is appropriate
%
% 7. PROPORTION ACTIVE vs THRESHOLD:
%    - Directly shows sparsity level at each threshold
%    - Use to select threshold that gives desired % active cells
%    - Typical target: 2-10% active for sparse neural codes
%
% RECOMMENDED WORKFLOW:
% 1. Check aggregated histogram to see distribution shape (unimodal/bimodal)
% 2. Check aggregated CDF to see overall distribution
% 3. Identify desired sparsity level (e.g., 5% active)
% 4. Find corresponding threshold from CDF (e.g., 95th percentile)
% 5. Check "Proportion Active" plots to verify sparsity
% 6. Verify consistency across conditions and animals
% 7. Use threshold lookup table in Section 7 for exact values
% 8. Test sensitivity with Figure5_ThresholdSensitivity_BIN_AN.m
%
% NOTE: Plots focus on 0-10 range as this contains >99% of neural activity data.
% Outliers beyond 10 are excluded from visualization for better resolution.

% %% =========================================================================
% %% SECTION 2: ActivityData Raw Activity Distribution Analysis
% %% =========================================================================
% 
% fprintf('\n--- Section 2: ActivityData Raw Activity Distribution ---\n');
% 
% % Initialize data collection structure
% ActivityRawData = struct();
% 
% % Collect raw activity data from ActivityData.Position1
% for c = 1:length(conditions)
%     condition = conditions{c};
%     fprintf('Processing condition: %s\n', condition);
% 
%     nRecs = length(ActivityData.(condition));
%     allActivityValues = [];
% 
%     for r = 1:nRecs
%         % Skip if in Skip list
%         if ismember(r, Skip.(condition))
%             continue;
%         end
% 
%         % Check if Position1 exists
%         if isfield(ActivityData.(condition)(r), 'Position1') && ...
%            ~isempty(ActivityData.(condition)(r).Position1)
% 
%             % Get Position1 data: [neurons × timepoints × trials]
%             data_P1 = ActivityData.(condition)(r).Position1;
% 
%             % Flatten to 1D array (all values)
%             allActivityValues = [allActivityValues; data_P1(:)];
%         end
%     end
% 
%     % Remove NaN values
%     allActivityValues = allActivityValues(~isnan(allActivityValues));
% 
%     % Remove low values (dead/empty cells) if flag is enabled
%     if Remove_Low_Values
%         allActivityValues = allActivityValues(allActivityValues >= DEAD_CELL_THRESHOLD);
%     end
% 
%     % Store in structure
%     ActivityRawData.(condition) = allActivityValues;
% 
%     fprintf('  %s: %d activity values collected\n', condition, length(allActivityValues));
% end
% 
% %% Plot 2.1: Histogram by condition
% fprintf('Creating Plot 2.1: Histogram by condition\n');
% 
% % Define histogram edges for different xlim ranges with high resolution
% histEdges_5 = linspace(0, 5, 201);  % 200 bins in 0-5 range (for raw data)
% histEdges_2 = linspace(0, 2, 201);  % 200 bins in 0-2 range (for mean data)
% 
% figure('Name', 'ActivityData Raw - Histogram by Condition');
% tiledlayout(2, 2);
% 
% for c = 1:length(conditions)
%     condition = conditions{c};
%     data = ActivityRawData.(condition);
% 
%     if isempty(data)
%         continue;
%     end
% 
%     % Filter data to 0-5 range for better visualization
%     data_filtered = data(data >= 0 & data <= 5);
% 
%     nexttile;
%     histogram(data_filtered, histEdges_5, 'FaceColor', conditionColors.(condition), ...
%              'FaceAlpha', 0.7, 'EdgeColor', 'none');
%     xlabel('Activity Value');
%     ylabel('Frequency');
%     title(sprintf('%s (n=%d, %.1f%% shown)', condition, length(data), 100*length(data_filtered)/length(data)));
%     xlim([0 5]);
%     grid on;
%     box on;
% end
% 
% sgtitle('ActivityData Raw Activity Distribution - Histogram by Condition (0-5 range)', 'FontWeight', 'bold');
% 
% %% Plot 2.2: CDF by condition
% fprintf('Creating Plot 2.2: CDF by condition\n');
% 
% figure('Name', 'ActivityData Raw - CDF by Condition');
% hold on;
% 
% for c = 1:length(conditions)
%     condition = conditions{c};
%     data = ActivityRawData.(condition);
% 
%     if isempty(data)
%         continue;
%     end
% 
%     % Compute CDF using ksdensity
%     [f, xi] = ksdensity(data, 'Function', 'cdf', 'NumPoints', 200);
% 
%     plot(xi, f, 'Color', conditionColors.(condition), 'LineWidth', 2, ...
%          'DisplayName', sprintf('%s (n=%d)', condition, length(data)));
% end
% 
% % Add sigma threshold lines (calculated from aggregated data)
% meanVal_temp = mean(allConditionsData);
% stdVal_temp = std(allConditionsData);
% thresholds_sigma = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
% for thresh = thresholds_sigma
%     threshValue = meanVal_temp + thresh * stdVal_temp;
%     if threshValue >= 0 && threshValue <= 5
%         xline(threshValue, '--', sprintf('%.1fσ', thresh), ...
%               'LineWidth', 1.0, 'Color', [0.7 0.2 0.2], 'Alpha', 0.5, ...
%               'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
%     end
% end
% 
% xlabel('Activity Value');
% ylabel('Cumulative Probability');
% title('ActivityData Raw Activity - CDF by Condition (0-5 range)');
% legend('Location', 'best');
% xlim([0 5]);
% grid on;
% box on;
% hold off;
% 
% %% Plot 2.3: KDE density plot by condition
% fprintf('Creating Plot 2.3: KDE density plot by condition\n');
% 
% figure('Name', 'ActivityData Raw - Density by Condition');
% hold on;
% 
% for c = 1:length(conditions)
%     condition = conditions{c};
%     data = ActivityRawData.(condition);
% 
%     if isempty(data)
%         continue;
%     end
% 
%     % Compute KDE
%     [f, xi] = ksdensity(data, 'NumPoints', 200);
% 
%     plot(xi, f, 'Color', conditionColors.(condition), 'LineWidth', 2, ...
%          'DisplayName', sprintf('%s', condition));
% end
% 
% xlabel('Activity Value');
% ylabel('Density');
% title('ActivityData Raw Activity - Density (KDE) by Condition (0-5 range)');
% legend('Location', 'best');
% xlim([0 5]);
% grid on;
% box on;
% hold off;
% 
% %% Plot 2.4: Histogram aggregated
% fprintf('Creating Plot 2.4: Histogram aggregated\n');
% 
% % Combine all conditions
% allConditionsData = [];
% for c = 1:length(conditions)
%     condition = conditions{c};
%     allConditionsData = [allConditionsData; ActivityRawData.(condition)];
% end
% 
% % Calculate statistics for full dataset
% medianVal = median(allConditionsData);
% meanVal = mean(allConditionsData);
% stdVal = std(allConditionsData);
% p90 = prctile(allConditionsData, 90);
% p95 = prctile(allConditionsData, 95);
% p97_5 = prctile(allConditionsData, 97.5);
% p99 = prctile(allConditionsData, 99);
% 
% % Filter data for visualization
% allConditionsData_filtered = allConditionsData(allConditionsData >= 0 & allConditionsData <= 5);
% 
% figure('Name', 'ActivityData Raw - Histogram Aggregated');
% histogram(allConditionsData_filtered, histEdges_5, 'FaceColor', [0.5 0.5 0.5], ...
%          'FaceAlpha', 0.7, 'EdgeColor', 'none');
% xlabel('Activity Value');
% ylabel('Frequency');
% title(sprintf('ActivityData Raw Activity - All Conditions (n=%d, %.1f%% shown)', ...
%     length(allConditionsData), 100*length(allConditionsData_filtered)/length(allConditionsData)));
% xlim([0 5]);
% grid on;
% box on;
% 
% % Add statistics text
% text(0.65, 0.95, sprintf('Median: %.4f', medianVal), 'Units', 'normalized', 'FontSize', 9);
% text(0.65, 0.90, sprintf('Mean: %.4f', meanVal), 'Units', 'normalized', 'FontSize', 9);
% text(0.65, 0.85, sprintf('Std: %.4f', stdVal), 'Units', 'normalized', 'FontSize', 9);
% text(0.65, 0.80, sprintf('90th %%ile: %.4f', p90), 'Units', 'normalized', 'FontSize', 9);
% text(0.65, 0.75, sprintf('95th %%ile: %.4f', p95), 'Units', 'normalized', 'FontSize', 9);
% text(0.65, 0.70, sprintf('99th %%ile: %.4f', p99), 'Units', 'normalized', 'FontSize', 9);
% 
% %% Plot 2.5: CDF aggregated with enhanced threshold lines
% fprintf('Creating Plot 2.5: CDF aggregated\n');
% 
% figure('Name', 'ActivityData Raw - CDF Aggregated');
% [f, xi] = ksdensity(allConditionsData, 'Function', 'cdf', 'NumPoints', 200);
% plot(xi, f, 'Color', [0.2 0.2 0.2], 'LineWidth', 2.5);
% xlabel('Activity Value');
% ylabel('Cumulative Probability');
% title('ActivityData Raw Activity - CDF All Conditions (with threshold markers)');
% xlim([0 5]);
% grid on;
% box on;
% 
% % Add enhanced threshold reference lines
% hold on;
% 
% % Mean line
% xline(meanVal, '-', '0σ (mean)', 'LineWidth', 1.5, 'Color', [0.3 0.3 0.3], ...
%      'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
% 
% % Sigma-based thresholds
% thresholds_sigma = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
% for thresh = thresholds_sigma
%     threshValue = meanVal + thresh * stdVal;
%     if threshValue >= 0 && threshValue <= 5
%         xline(threshValue, '--', sprintf('%.1fσ', thresh), 'LineWidth', 1.2, 'Color', [0.7 0.2 0.2], ...
%              'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
%     end
% end
% 
% % Percentile-based thresholds
% percentiles = [90, 95, 97.5, 99];
% percentile_values = [p90, p95, p97_5, p99];
% for i = 1:length(percentiles)
%     pVal = percentile_values(i);
%     if pVal >= 0 && pVal <= 5
%         xline(pVal, ':', sprintf('P%.1f', percentiles(i)), 'LineWidth', 1.2, 'Color', [0.2 0.2 0.7], ...
%              'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'left');
%     end
% end
% 
% hold off;
% 
% fprintf('Section 2 complete: ActivityData raw activity distribution plots created\n');

%% =========================================================================
%% SECTION 3: Grid40 Raw Activity Distribution Analysis
%% =========================================================================
if RUN.section3

fprintf('\n--- Section 3: Grid40 Raw Activity Distribution ---\n');

% Initialize data collection structure
Grid40RawData = struct();

% Collect raw activity data from Grid40
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        Grid40RawData.(condition) = [];
        continue;
    end

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);
    allGridValues = [];

    for r = 1:nRecsGrid
        % Skip if in Skip list
        if ismember(r, Skip.(condition))
            continue;
        end

        % Check if P1 exists
        if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
            gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

            % Dereference cell array
            if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
                gridData_P1 = gridData_P1_cell{:};
            else
                gridData_P1 = gridData_P1_cell;
            end

            if ~isempty(gridData_P1)
                % Grid40 data: [gridY × gridX × timepoints × trials]
                allGridValues = [allGridValues; gridData_P1(:)];
            end
        end
    end

    % Remove NaN values
    allGridValues = allGridValues(~isnan(allGridValues));

    % Remove low values (dead/empty cells) if flag is enabled
    if Remove_Low_Values
        allGridValues = allGridValues(allGridValues >= DEAD_CELL_THRESHOLD);
    end

    % Store in structure
    Grid40RawData.(condition) = allGridValues;

    fprintf('  %s: %d grid values collected\n', condition, length(allGridValues));
end

% Aggregate across all conditions once (used by Plot 3.2 CDF reference lines,
% Plot 3.4 Histogram Aggregated, Plot 3.5 CDF Aggregated, and Plot 3.6 sweep).
allGrid40Data = [];
for c = 1:length(conditions)
    condition = conditions{c};
    allGrid40Data = [allGrid40Data; Grid40RawData.(condition)];
end

%% Plot 3.1: Histogram by condition
fprintf('Creating Plot 3.1: Histogram by condition (Grid40 Raw)\n');

figure('Name', 'Grid40 Raw - Histogram by Condition');
tiledlayout(2, 2);

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40RawData.(condition);

    if isempty(data)
        continue;
    end

    data_filtered = data(data >= 0 & data <= 5);

    nexttile;
    histogram(data_filtered, histEdges_5, 'FaceColor', conditionColors.(condition), ...
             'FaceAlpha', 0.7, 'EdgeColor', 'none');
    % Absolute threshold marker (used by Ising pipeline: Figure5_dataAggregation.m:41)
    hold on;
    pct_above_2 = 100 * mean(data >= 2.0);
    xline(2.0, '-', sprintf('2.0 dF/F (%.1f%%)', pct_above_2), ...
          'LineWidth', 1.8, 'Color', [0 0.5 0], ...
          'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'right');
    hold off;
    xlabel('Grid Activity Value');
    ylabel('Frequency');
    title(sprintf('%s (n=%d, %.1f%% shown)', condition, length(data), 100*length(data_filtered)/length(data)));
    xlim([0 5]);
    grid on;
    box on;
end

sgtitle('Grid40 Raw Activity Distribution - Histogram by Condition (0-5 range)', 'FontWeight', 'bold');

%% Plot 3.2: CDF by condition
fprintf('Creating Plot 3.2: CDF by condition (Grid40 Raw)\n');

figure('Name', 'Grid40 Raw - CDF by Condition');
hold on;

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40RawData.(condition);

    if isempty(data)
        continue;
    end

    [f, xi] = ksdensity(data, kde_xi_5, 'Function', 'cdf');

    plot(xi, f, 'Color', conditionColors.(condition), 'LineWidth', 2, ...
         'DisplayName', sprintf('%s (n=%d)', condition, length(data)));
end

% Add sigma threshold lines (calculated from aggregated data)
meanVal_temp = mean(allGrid40Data);
stdVal_temp = std(allGrid40Data);
thresholds_sigma = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
for thresh = thresholds_sigma
    threshValue = meanVal_temp + thresh * stdVal_temp;
    if threshValue >= 0 && threshValue <= 5
        xline(threshValue, '--', sprintf('%.1fσ', thresh), ...
              'LineWidth', 1.0, 'Color', [0.7 0.2 0.2], 'Alpha', 0.5, ...
              'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
    end
end

% Absolute threshold marker (Ising pipeline uses raw_activity_threshold = 2.0)
cdf_at_2 = interp1(kde_xi_5, ksdensity(allGrid40Data, kde_xi_5, 'Function', 'cdf'), 2.0);
xline(2.0, '-', sprintf('2.0 dF/F (top %.1f%%)', 100*(1 - cdf_at_2)), ...
      'LineWidth', 2.0, 'Color', [0 0.5 0], ...
      'LabelVerticalAlignment', 'middle', 'LabelHorizontalAlignment', 'right', ...
      'HandleVisibility', 'off');

xlabel('Grid Activity Value');
ylabel('Cumulative Probability');
title('Grid40 Raw Activity - CDF by Condition (0-5 range)');
legend('Location', 'best');
xlim([0 5]);
grid on;
box on;
hold off;

%% Plot 3.3: KDE density plot by condition
fprintf('Creating Plot 3.3: KDE density plot by condition (Grid40 Raw)\n');

figure('Name', 'Grid40 Raw - Density by Condition');
hold on;

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40RawData.(condition);

    if isempty(data)
        continue;
    end

    [f, xi] = ksdensity(data, kde_xi_5);

    plot(xi, f, 'Color', conditionColors.(condition), 'LineWidth', 2, ...
         'DisplayName', sprintf('%s', condition));
end

% Absolute threshold marker (Ising pipeline uses raw_activity_threshold = 2.0)
xline(2.0, '-', '2.0 dF/F', 'LineWidth', 2.0, 'Color', [0 0.5 0], ...
      'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'right', ...
      'HandleVisibility', 'off');

xlabel('Grid Activity Value');
ylabel('Density');
title('Grid40 Raw Activity - Density (KDE) by Condition (0-5 range)');
legend('Location', 'best');
xlim([0 5]);
grid on;
box on;
hold off;

%% Plot 3.4: Histogram aggregated
fprintf('Creating Plot 3.4: Histogram aggregated (Grid40 Raw)\n');

% allGrid40Data was already aggregated above (before Plot 3.1)

% Calculate statistics
medianVal = median(allGrid40Data);
meanVal = mean(allGrid40Data);
stdVal = std(allGrid40Data);
p90 = prctile(allGrid40Data, 90);
p95 = prctile(allGrid40Data, 95);
p97_5 = prctile(allGrid40Data, 97.5);
p99 = prctile(allGrid40Data, 99);

% Filter for visualization
allGrid40Data_filtered = allGrid40Data(allGrid40Data >= 0 & allGrid40Data <= 5);

figure('Name', 'Grid40 Raw - Histogram Aggregated');
histogram(allGrid40Data_filtered, histEdges_5, 'FaceColor', [0.5 0.5 0.5], ...
         'FaceAlpha', 0.7, 'EdgeColor', 'none');
xlabel('Grid Activity Value');
ylabel('Frequency');
title(sprintf('Grid40 Raw Activity - All Conditions (n=%d, %.1f%% shown)', ...
    length(allGrid40Data), 100*length(allGrid40Data_filtered)/length(allGrid40Data)));
xlim([0 5]);
grid on;
box on;

% Absolute threshold marker (Ising pipeline uses raw_activity_threshold = 2.0)
pct_above_2_agg = 100 * mean(allGrid40Data >= 2.0);
hold on;
xline(2.0, '-', sprintf('2.0 dF/F (%.2f%% above)', pct_above_2_agg), ...
      'LineWidth', 2.0, 'Color', [0 0.5 0], ...
      'LabelVerticalAlignment', 'middle', 'LabelHorizontalAlignment', 'right');
hold off;

% Add statistics
text(0.65, 0.95, sprintf('Median: %.4f', medianVal), 'Units', 'normalized', 'FontSize', 9);
text(0.65, 0.90, sprintf('Mean: %.4f', meanVal), 'Units', 'normalized', 'FontSize', 9);
text(0.65, 0.85, sprintf('Std: %.4f', stdVal), 'Units', 'normalized', 'FontSize', 9);
text(0.65, 0.80, sprintf('90th %%ile: %.4f', p90), 'Units', 'normalized', 'FontSize', 9);
text(0.65, 0.75, sprintf('95th %%ile: %.4f', p95), 'Units', 'normalized', 'FontSize', 9);
text(0.65, 0.70, sprintf('99th %%ile: %.4f', p99), 'Units', 'normalized', 'FontSize', 9);
text(0.65, 0.65, sprintf('%% ≥ 2.0: %.2f%%', pct_above_2_agg), ...
     'Units', 'normalized', 'FontSize', 9, 'FontWeight', 'bold', 'Color', [0 0.5 0]);

%% Plot 3.5: CDF aggregated with enhanced threshold lines
fprintf('Creating Plot 3.5: CDF aggregated (Grid40 Raw)\n');

figure('Name', 'Grid40 Raw - CDF Aggregated');
[f, xi] = ksdensity(allGrid40Data, kde_xi_5, 'Function', 'cdf');
plot(xi, f, 'Color', [0.2 0.2 0.2], 'LineWidth', 2.5);
xlabel('Grid Activity Value');
ylabel('Cumulative Probability');
title('Grid40 Raw Activity - CDF All Conditions (with threshold markers)');
xlim([0 5]);
grid on;
box on;

% Add enhanced threshold reference lines
hold on;

% Mean line
xline(meanVal, '-', '0σ (mean)', 'LineWidth', 1.5, 'Color', [0.3 0.3 0.3], ...
     'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');

% Sigma-based thresholds
thresholds_sigma = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
for thresh = thresholds_sigma
    threshValue = meanVal + thresh * stdVal;
    if threshValue >= 0 && threshValue <= 5
        xline(threshValue, '--', sprintf('%.1fσ', thresh), 'LineWidth', 1.2, 'Color', [0.7 0.2 0.2], ...
             'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
    end
end

% Percentile-based thresholds
percentiles = [90, 95, 97.5, 99];
percentile_values = [p90, p95, p97_5, p99];
for i = 1:length(percentiles)
    pVal = percentile_values(i);
    if pVal >= 0 && pVal <= 5
        xline(pVal, ':', sprintf('P%.1f', percentiles(i)), 'LineWidth', 1.2, 'Color', [0.2 0.2 0.7], ...
             'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'left');
    end
end

% Absolute threshold marker used by Ising pipeline (Figure5_dataAggregation.m:41)
% This is the production binarisation cutoff — raw_activity_threshold = 2.0 dF/F
cdf_at_2_agg = interp1(xi, f, 2.0);
pct_above_2_cdf = 100 * (1 - cdf_at_2_agg);
xline(2.0, '-', sprintf('2.0 dF/F (top %.2f%%)', pct_above_2_cdf), ...
      'LineWidth', 2.5, 'Color', [0 0.5 0], ...
      'LabelVerticalAlignment', 'middle', 'LabelHorizontalAlignment', 'right');
% Horizontal guide showing the corresponding cumulative probability
yline(cdf_at_2_agg, ':', sprintf('CDF(2.0) = %.4f', cdf_at_2_agg), ...
      'LineWidth', 1.2, 'Color', [0 0.5 0], ...
      'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'right');

hold off;

%% Plot 3.6: Fraction-active vs absolute threshold (by condition)
% This is the direct visual justification for the absolute threshold = 2.0 dF/F
% used by the Ising pipeline (Figure5_dataAggregation.m:41). Unlike sigma-based
% plots, this sweeps absolute dF/F cutoffs so the user can read off the fraction
% of grid-frame samples that remain "active" at each threshold, per condition.
fprintf('Creating Plot 3.6: Fraction-active vs absolute threshold (Grid40 Raw)\n');

abs_thresholds = 0.5:0.05:4.0;
fracActive = struct();
for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40RawData.(condition);
    if isempty(data)
        fracActive.(condition) = nan(size(abs_thresholds));
        continue;
    end
    fa = zeros(size(abs_thresholds));
    for ti = 1:length(abs_thresholds)
        fa(ti) = mean(data >= abs_thresholds(ti));
    end
    fracActive.(condition) = fa;
end

figure('Name', 'Grid40 Raw - Fraction Active vs Absolute Threshold');
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if all(isnan(fracActive.(condition)))
        continue;
    end
    plot(abs_thresholds, 100*fracActive.(condition), ...
         'Color', conditionColors.(condition), 'LineWidth', 2.2, ...
         'DisplayName', condition);
end

% Mark the production threshold
xline(2.0, '-', '2.0 dF/F (Ising cutoff)', 'LineWidth', 2.0, 'Color', [0 0.5 0], ...
      'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'right', ...
      'HandleVisibility', 'off');
% Reference sigma-equivalent markers for context (using aggregated stats)
for thresh = [1.0, 1.5, 2.0, 2.5]
    threshValue = meanVal + thresh * stdVal;
    if threshValue >= abs_thresholds(1) && threshValue <= abs_thresholds(end)
        xline(threshValue, '--', sprintf('%.1fσ', thresh), ...
              'LineWidth', 0.9, 'Color', [0.7 0.2 0.2], 'Alpha', 0.5, ...
              'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left', ...
              'HandleVisibility', 'off');
    end
end

xlabel('Absolute threshold (dF/F)');
ylabel('Fraction of grid-frame samples ≥ threshold (%)');
title('Grid40 Raw - Fraction Active vs Absolute Threshold (justification for cutoff = 2.0)');
legend('Location', 'northeast');
xlim([abs_thresholds(1), abs_thresholds(end)]);
set(gca, 'YScale', 'log');  % log scale makes the plateau above 2.0 visible
grid on;
box on;
hold off;

% Print the sparsity at the Ising cutoff for each condition
fprintf('\n  Fraction active at absolute threshold = 2.0 dF/n');
for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40RawData.(condition);
    if ~isempty(data)
        fprintf('    %-12s: %.3f%% (n=%d samples ≥ 2.0)\n', ...
                condition, 100*mean(data >= 2.0), sum(data >= 2.0));
    end
end

end  % RUN.section3

%% -------------------------------------------------------------------------
%% Plots 3.7 / 3.8 / 3.9 — Trial-structured justification for absolute = 2.0
%% -------------------------------------------------------------------------
% Independent of Section 3: runs whenever RUN.section3_stimlocked is true,
% even if RUN.section3 is false. Builds its own Grid40Trial data cache.
if RUN.section3_stimlocked
% These three plots reuse Grid40 data (per-trial shape, not flattened) to
% build stim-locked physiological and Ising-specific justifications for the
% absolute 2.0 dF/F cutoff used by Figure5_dataAggregation.m.
%
%   3.7 — Stim-locked fraction-active PSTH across absolute thresholds
%         (shows 2.0 dF/F gives a clear stim-locked bump above flat baseline)
%   3.8 — Stim-vs-baseline SNR of fraction-active vs absolute threshold
%         (objective optimum — if the SNR curve peaks near 2.0, that IS the
%          justification)
%   3.9 — Binary pattern rank-frequency (Zipf-like) across thresholds
%         (Ising-community gold standard — at the right threshold, the
%          rank-frequency curve should look near-Zipf on log-log axes)

fprintf('\n--- Section 3 (cont.): Trial-structured analyses for absolute threshold ---\n');

% Diagnostic: confirm Grid40 is in workspace
if ~exist('Grid40', 'var')
    error(['Grid40 variable is not in the workspace. Load it first:\n' ...
           '    load(mba_p(''RawData3.mat''), ''Grid40'')']);
end
fprintf('  Grid40 fields: %s\n', strjoin(fieldnames(Grid40), ', '));

% Stim timing (matches Figure5_dataAggregation.m conventions):
PRE_STIM_FRAMES  = 1:80;
STIM_FRAMES      = 81:100;
POST_STIM_FRAMES = 101:185;
stim_onset = STIM_FRAMES(1);

% Thresholds to sweep
psth_thresholds       = [1.0, 1.5, 2.0, 2.5, 3.0];    % absolute dF/F
snr_thresholds_dense  = 0.5:0.1:4.0;

% -- Extract trial-structured Grid40 data once, reuse for all three plots --
Grid40Trial = struct();
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    if ~isfield(Grid40, conditionIndividual)
        fprintf('  %s: Grid40.%s field not found — skipping\n', condition, conditionIndividual);
        Grid40Trial.(condition) = [];
        continue;
    end

    trialsCell = {};                  % Collect reshaped [cells × time × trials] blocks
    target_nCells = NaN;              % Set from first valid recording
    target_nT     = NaN;
    nRecsGrid     = length(Grid40.(conditionIndividual).AllNeurons);
    nRec_skipped_in_skiplist = 0;
    nRec_no_P1 = 0;
    nRec_empty = 0;
    nRec_shape_mismatch = 0;
    nRec_added = 0;

    fprintf('  %s: scanning %d recordings (Skip list: %s)\n', ...
            condition, nRecsGrid, mat2str(Skip.(condition)));

    for r = 1:nRecsGrid
        if ismember(r, Skip.(condition))
            nRec_skipped_in_skiplist = nRec_skipped_in_skiplist + 1;
            continue;
        end
        if ~isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
            nRec_no_P1 = nRec_no_P1 + 1;
            continue;
        end
        gridData_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;
        if iscell(gridData_cell) && ~isempty(gridData_cell)
            gridData_r = gridData_cell{1};  % First element only (safe for 1-element cell)
        else
            gridData_r = gridData_cell;
        end
        if isempty(gridData_r)
            nRec_empty = nRec_empty + 1;
            continue;
        end

        % Get dimensions robustly (size() may return <4 elements for
        % arrays with trailing singleton dimensions).
        sz_r = size(gridData_r);
        if numel(sz_r) < 4
            sz_r = [sz_r, ones(1, 4 - numel(sz_r))];  %#ok<AGROW>  % pad with trailing 1s
        end
        gY = sz_r(1); gX = sz_r(2); nT_r = sz_r(3); nTr_r = sz_r(4);
        nCells_r = gY * gX;

        % Set target shape from first valid recording
        if isnan(target_nCells)
            target_nCells = nCells_r;
            target_nT     = nT_r;
            fprintf('    Rec %d: setting target shape [nCells=%d, nT=%d, first nTr=%d]\n', ...
                    r, nCells_r, nT_r, nTr_r);
        end

        % Skip if this recording doesn't match the first valid shape
        if nCells_r ~= target_nCells || nT_r ~= target_nT
            fprintf(['    Warning: %s Rec %d has shape [%d × %d × %d × %d] ' ...
                     '(nCells=%d, nT=%d) — expected nCells=%d, nT=%d; skipping\n'], ...
                    condition, r, gY, gX, nT_r, nTr_r, ...
                    nCells_r, nT_r, target_nCells, target_nT);
            nRec_shape_mismatch = nRec_shape_mismatch + 1;
            continue;
        end

        % Reshape and store in cell array (defer cat to end)
        reshaped = reshape(gridData_r, [nCells_r, nT_r, nTr_r]);
        trialsCell{end+1} = reshaped; %#ok<AGROW>
        nRec_added = nRec_added + 1;
    end

    fprintf('    Summary: %d added, %d in Skip list, %d missing P1, %d empty, %d shape mismatch\n', ...
            nRec_added, nRec_skipped_in_skiplist, nRec_no_P1, nRec_empty, nRec_shape_mismatch);

    % Concatenate all recordings at the end. Use pre-allocation + slice
    % assignment instead of `cat(3, trialsCell{:})` because `cat` can
    % throw spurious "Index in position 1 exceeds array bounds" errors
    % on some MATLAB versions when the cs-list expansion hits a
    % shadowed built-in. This manual fill is equivalent and always works.
    if isempty(trialsCell)
        allTrialsConcat = [];
    else
        % Count total trials across all blocks
        totalTrials = 0;
        for blk = 1:length(trialsCell)
            sz_blk = size(trialsCell{blk});
            if numel(sz_blk) < 3
                totalTrials = totalTrials + 1;
            else
                totalTrials = totalTrials + sz_blk(3);
            end
        end

        if totalTrials > 0
            allTrialsConcat = zeros(target_nCells, target_nT, totalTrials, 'like', trialsCell{1});
            startIdx = 1;
            for blk = 1:length(trialsCell)
                sz_blk = size(trialsCell{blk});
                if numel(sz_blk) < 3
                    nThisBlock = 1;
                else
                    nThisBlock = sz_blk(3);
                end
                allTrialsConcat(:, :, startIdx:startIdx + nThisBlock - 1) = trialsCell{blk};
                startIdx = startIdx + nThisBlock;
            end
        else
            allTrialsConcat = [];
        end
    end

    Grid40Trial.(condition) = allTrialsConcat;
    if ~isempty(allTrialsConcat)
        fprintf('  %s: %d grid cells × %d timepoints × %d trials (across recordings)\n', ...
                condition, size(allTrialsConcat, 1), size(allTrialsConcat, 2), size(allTrialsConcat, 3));
    else
        fprintf('  %s: no valid trials collected\n', condition);
    end
end

%% Plot 3.7: Stim-locked fraction-active PSTH across absolute thresholds
fprintf('Creating Plot 3.7: Fraction-active PSTH across absolute thresholds\n');

figure('Name', 'Grid40 Raw - Fraction Active PSTH across Absolute Thresholds');
tiledlayout(2, 2);

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40Trial.(condition);
    if isempty(data)
        nexttile;
        text(0.5, 0.5, sprintf('%s: no data', condition), ...
             'HorizontalAlignment', 'center', 'Units', 'normalized');
        axis off;
        continue;
    end

    [nCells_c, nT_c, nTr_c] = size(data);

    nexttile;
    hold on;
    cmap = parula(length(psth_thresholds) + 1);
    legend_handles = gobjects(length(psth_thresholds), 1);
    for ti = 1:length(psth_thresholds)
        thresh_v = psth_thresholds(ti);
        bin = data > thresh_v;  % [cells × time × trials]
        % Fraction active = mean over grid cells, then mean over trials
        frac_per_trial = squeeze(nanmean(bin, 1));  % [time × trials]
        if size(frac_per_trial, 2) == 1
            frac_time = frac_per_trial(:)' * 100;
        else
            frac_time = nanmean(frac_per_trial, 2)' * 100;  % [1 × time]
        end

        if thresh_v == 2.0
            legend_handles(ti) = plot(1:nT_c, frac_time, 'Color', [0 0.5 0], ...
                                       'LineWidth', 3.2, ...
                                       'DisplayName', sprintf('%.1f dF/F (Ising)', thresh_v));
        else
            legend_handles(ti) = plot(1:nT_c, frac_time, 'Color', cmap(ti, :), ...
                                       'LineWidth', 1.6, ...
                                       'DisplayName', sprintf('%.1f dF/F', thresh_v));
        end
    end

    % Shade the stimulus period
    yl = ylim;
    patch([STIM_FRAMES(1) STIM_FRAMES(end) STIM_FRAMES(end) STIM_FRAMES(1)], ...
          [yl(1) yl(1) yl(2) yl(2)], [0.6 0.6 0.6], 'FaceAlpha', 0.12, ...
          'EdgeColor', 'none', 'HandleVisibility', 'off');
    xline(stim_onset, ':', 'stim', 'LineWidth', 1.0, 'HandleVisibility', 'off');

    xlabel('Timepoint (frames)');
    ylabel('% grid cells active');
    title(sprintf('%s (n=%d trials)', condition, nTr_c));
    legend(legend_handles, 'Location', 'northeast', 'FontSize', 7);
    grid on; box on; hold off;
end

sgtitle('Fraction-Active PSTH across absolute thresholds (stim period shaded)', 'FontWeight', 'bold');

%% Plot 3.8: Stim-vs-baseline SNR of fraction-active vs absolute threshold
fprintf('Creating Plot 3.8: Stim-vs-baseline SNR vs absolute threshold\n');

figure('Name', 'Grid40 Raw - Stim-vs-Baseline SNR vs Absolute Threshold');
hold on;

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40Trial.(condition);
    if isempty(data)
        continue;
    end

    [~, ~, nTr_c] = size(data);

    snr_curve = nan(size(snr_thresholds_dense));
    for ti = 1:length(snr_thresholds_dense)
        thresh_v = snr_thresholds_dense(ti);
        bin = data > thresh_v;  % [cells × time × trials]

        % Per-trial fraction active in baseline and stim windows
        % baseline_frac: [nTr × 1]
        baseline_frac = squeeze(nanmean(nanmean(bin(:, PRE_STIM_FRAMES, :), 1), 2));
        stim_frac     = squeeze(nanmean(nanmean(bin(:, STIM_FRAMES, :), 1), 2));

        mb = nanmean(baseline_frac);
        sb = nanstd(baseline_frac);
        ms = nanmean(stim_frac);

        if sb > 0
            snr_curve(ti) = (ms - mb) / sb;
        end
    end

    plot(snr_thresholds_dense, snr_curve, 'Color', conditionColors.(condition), ...
         'LineWidth', 2.2, 'DisplayName', condition);

    % Mark the per-condition optimum
    [peak_snr, peak_idx] = max(snr_curve);
    if ~isnan(peak_snr)
        plot(snr_thresholds_dense(peak_idx), peak_snr, 'o', ...
             'Color', conditionColors.(condition), ...
             'MarkerSize', 11, 'LineWidth', 2.2, 'HandleVisibility', 'off');
        fprintf('    %s: SNR peaks at %.2f dF/F (SNR = %.2f)\n', ...
                condition, snr_thresholds_dense(peak_idx), peak_snr);
    end
end

% Mark the Ising cutoff
xline(2.0, '-', '2.0 dF/F (Ising)', 'LineWidth', 2.0, 'Color', [0 0.5 0], ...
      'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'right', ...
      'HandleVisibility', 'off');

xlabel('Absolute threshold (dF/F)');
ylabel('(Stim − Baseline) / Baseline std');
title('Stim-vs-Baseline SNR of fraction-active (per-condition optima marked with circles)');
legend('Location', 'best');
grid on; box on; hold off;

%% Plot 3.9: Binary pattern rank-frequency (Zipf) across absolute thresholds
fprintf('Creating Plot 3.9: Binary pattern rank-frequency (Zipf) across thresholds\n');

figure('Name', 'Grid40 Raw - Pattern Rank-Frequency (Zipf) across Absolute Thresholds');
tiledlayout(2, 2);

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40Trial.(condition);
    if isempty(data)
        nexttile;
        text(0.5, 0.5, sprintf('%s: no data', condition), ...
             'HorizontalAlignment', 'center', 'Units', 'normalized');
        axis off;
        continue;
    end

    [nCells_c, nT_c, nTr_c] = size(data);
    nFrames_c = nT_c * nTr_c;

    nexttile;
    hold on;
    cmap = parula(length(psth_thresholds) + 1);
    legend_handles_z = gobjects(length(psth_thresholds), 1);

    for ti = 1:length(psth_thresholds)
        thresh_v = psth_thresholds(ti);

        bin = data > thresh_v;                          % [cells × time × trials] logical
        patterns = reshape(bin, [nCells_c, nFrames_c]); % [cells × frames]

        % Replace NaN-originating rows (should be none since bin is logical)
        % Each column = one 338-bit pattern. unique by rows after transpose.
        [~, ~, ic] = unique(patterns', 'rows');
        counts = accumarray(ic, 1);                      % [nUnique × 1]
        counts = sort(counts, 'descend');
        counts = counts(counts > 0);

        if thresh_v == 2.0
            legend_handles_z(ti) = plot(1:length(counts), counts, ...
                                         'Color', [0 0.5 0], 'LineWidth', 3, ...
                                         'DisplayName', sprintf('%.1f dF/F (Ising, %d patterns)', ...
                                                                 thresh_v, length(counts)));
        else
            legend_handles_z(ti) = plot(1:length(counts), counts, ...
                                         'Color', cmap(ti, :), 'LineWidth', 1.6, ...
                                         'DisplayName', sprintf('%.1f dF/F (%d patterns)', ...
                                                                 thresh_v, length(counts)));
        end
    end

    set(gca, 'XScale', 'log', 'YScale', 'log');
    xlabel('Pattern rank');
    ylabel('Frequency');
    title(sprintf('%s (n=%d frames)', condition, nFrames_c));
    legend(legend_handles_z, 'Location', 'southwest', 'FontSize', 7);
    grid on; box on; hold off;
end

sgtitle('Binary pattern rank-frequency distribution across absolute thresholds (log-log)', ...
        'FontWeight', 'bold');

% Free large intermediates (prevents memory pressure on later sections)
clear Grid40Trial trialsCell

end  % RUN.section3_stimlocked

%% =========================================================================
%% SECTION 4: Grid40 Variance/Sigma Analysis - Multiple Variance Types
%% =========================================================================
if RUN.section4

fprintf('\n--- Section 4: Grid40 Variance/Sigma Analysis ---\n');

% -------------------------------------------------------------------------
% SECTION 4.1: Per-Grid-Cell Sigma (variance across time AND trials)
% -------------------------------------------------------------------------

fprintf('\n--- Section 4.1: Per-Grid-Cell Sigma (variance across time and trials) ---\n');

% Initialize data collection structure
Grid40SigmaPerCell = struct();

% Collect per-grid-cell sigma values
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        Grid40SigmaPerCell.(condition) = [];
        continue;
    end

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);
    allSigmaValues = [];

    for r = 1:nRecsGrid
        % Skip if in Skip list
        if ismember(r, Skip.(condition))
            continue;
        end

        % Check if P1 exists
        if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
            gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

            % Dereference cell array
            if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
                gridData_P1 = gridData_P1_cell{:};
            else
                gridData_P1 = gridData_P1_cell;
            end

            if ~isempty(gridData_P1)
                % Grid40 data: [gridY × gridX × timepoints × trials]
                [gridY, gridX, nTimepoints, nTrials] = size(gridData_P1);

                % Subset timeframes for analysis
                gridData_P1 = gridData_P1(:, :, TimeFrameSelection, :);
                nTimepoints_selected = length(TimeFrameSelection);

                % Reshape to [gridCells × (timepoints*trials)]
                gridData_reshaped = reshape(gridData_P1, [gridY*gridX, nTimepoints_selected*nTrials]);

                % Calculate mean activity per grid cell (for filtering)
                meanActivity = mean(gridData_reshaped, 2, 'omitnan');

                % Calculate variance for each grid cell across all time and trials
                variances = var(gridData_reshaped, 0, 2, 'omitnan');

                % Convert to sigma (standard deviation)
                sigmas = sqrt(variances);

                % Filter based on Remove_Low_Values flag
                if Remove_Low_Values
                    % Only keep grid cells where mean activity >= threshold
                    validIdx = (meanActivity >= DEAD_CELL_THRESHOLD) & ~isnan(sigmas);
                    sigmas = sigmas(validIdx);
                else
                    % Remove only NaNs
                    sigmas = sigmas(~isnan(sigmas));
                end

                allSigmaValues = [allSigmaValues; sigmas];
            end
        end
    end

    % Store in structure
    Grid40SigmaPerCell.(condition) = allSigmaValues;

    fprintf('  %s: %d per-grid-cell sigma values collected\n', condition, length(allSigmaValues));
end

% Plot 4.1: CDF by condition - Per-Grid-Cell Sigma
fprintf('Creating Plot 4.1: CDF by condition (Per-Grid-Cell Sigma)\n');

figure('Name', 'Grid40 Per-Grid-Cell Sigma - CDF by Condition');
hold on;

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40SigmaPerCell.(condition);

    if isempty(data)
        continue;
    end

    [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

    plot(xi, f, 'Color', conditionColors.(condition), 'LineWidth', 2, ...
         'DisplayName', sprintf('%s (n=%d)', condition, length(data)));
end

% Add 98th percentile vertical lines for each condition
for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40SigmaPerCell.(condition);

    if ~isempty(data)
        [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

        % Find 98th percentile using direct index search
        p98_idx = find(f >= 0.98, 1, 'first');
        if ~isempty(p98_idx)
            p98_sigma = xi(p98_idx);
        else
            p98_sigma = NaN;
        end

        if ~isnan(p98_sigma) && p98_sigma >= 0 && p98_sigma <= 3
            xline(p98_sigma, '--', 'Color', conditionColors.(condition), ...
                  'LineWidth', 1.5, 'Alpha', 0.7, 'HandleVisibility', 'off');
        end
    end
end

xlabel('Sigma (Standard Deviation)');
ylabel('Cumulative Probability');
title('Grid40 Per-Grid-Cell Sigma - CDF by Condition');
legend('Location', 'best');
xlim([0 3]);
grid on;
box on;
hold off;

% -------------------------------------------------------------------------
% SECTION 4.2: Per-Grid-Cell Per-Timepoint Sigma (variance across trials)
% -------------------------------------------------------------------------

fprintf('\n--- Section 4.2: Per-Grid-Cell Per-Timepoint Sigma (variance across trials) ---\n');

% Initialize data collection structure
Grid40SigmaPerTimepoint = struct();

% Collect per-grid-cell per-timepoint sigma values
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        Grid40SigmaPerTimepoint.(condition) = [];
        continue;
    end

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);
    allSigmaValues = [];

    for r = 1:nRecsGrid
        % Skip if in Skip list
        if ismember(r, Skip.(condition))
            continue;
        end

        % Check if P1 exists
        if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
            gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

            % Dereference cell array
            if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
                gridData_P1 = gridData_P1_cell{:};
            else
                gridData_P1 = gridData_P1_cell;
            end

            if ~isempty(gridData_P1)
                % Grid40 data: [gridY × gridX × timepoints × trials]
                [gridY, gridX, nTimepoints, nTrials] = size(gridData_P1);

                % Calculate mean activity per grid cell (for filtering)
                gridData_forMean = reshape(gridData_P1, [gridY*gridX, nTimepoints*nTrials]);
                meanActivity = mean(gridData_forMean, 2, 'omitnan');

                % Reshape to [gridCells × timepoints × trials]
                gridData_reshaped = reshape(gridData_P1, [gridY*gridX, nTimepoints, nTrials]);

                % Subset timeframes for analysis
                gridData_reshaped = gridData_reshaped(:, TimeFrameSelection, :);

                % Calculate variance across trials for each grid cell at each timepoint
                % Result: [gridCells × timepoints_selected × 1]
                variances = var(gridData_reshaped, 0, 3, 'omitnan');

                % Convert to sigma and remove singleton dimension
                sigmas = sqrt(variances);
                sigmas = squeeze(sigmas);  % Remove singleton dimension: [gridCells × timepoints]

                % Filter based on Remove_Low_Values flag
                if Remove_Low_Values
                    % Expand meanActivity to match dimensions (after timeframe subsetting)
                    meanActivity_expanded = repmat(meanActivity, 1, size(sigmas, 2));
                    validIdx = (meanActivity_expanded >= DEAD_CELL_THRESHOLD) & ~isnan(sigmas);
                    sigmas_filtered = sigmas(validIdx);
                else
                    % Remove only NaNs
                    sigmas_filtered = sigmas(~isnan(sigmas(:)));
                end

                allSigmaValues = [allSigmaValues; sigmas_filtered(:)];
            end
        end
    end

    % Store in structure
    Grid40SigmaPerTimepoint.(condition) = allSigmaValues;

    fprintf('  %s: %d per-timepoint sigma values collected\n', condition, length(allSigmaValues));
end

% Plot 4.2: CDF by condition - Per-Timepoint Sigma
fprintf('Creating Plot 4.2: CDF by condition (Per-Timepoint Sigma)\n');

figure('Name', 'Grid40 Per-Timepoint Sigma - CDF by Condition');
hold on;

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40SigmaPerTimepoint.(condition);

    if isempty(data)
        continue;
    end

    [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

    plot(xi, f, 'Color', conditionColors.(condition), 'LineWidth', 2, ...
         'DisplayName', sprintf('%s (n=%d)', condition, length(data)));
end

% Add 98th percentile vertical lines for each condition
for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40SigmaPerTimepoint.(condition);

    if ~isempty(data)
        [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

        % Find 98th percentile using direct index search
        p98_idx = find(f >= 0.98, 1, 'first');
        if ~isempty(p98_idx)
            p98_sigma = xi(p98_idx);
        else
            p98_sigma = NaN;
        end

        if ~isnan(p98_sigma) && p98_sigma >= 0 && p98_sigma <= 3
            xline(p98_sigma, '--', 'Color', conditionColors.(condition), ...
                  'LineWidth', 1.5, 'Alpha', 0.7, 'HandleVisibility', 'off');
        end
    end
end

xlabel('Sigma (Standard Deviation)');
ylabel('Cumulative Probability');
title('Grid40 Per-Grid-Cell Per-Timepoint Sigma - CDF by Condition');
legend('Location', 'best');
xlim([0 3]);
grid on;
box on;
hold off;

% -------------------------------------------------------------------------
% SECTION 4.3: Per-Grid-Cell Per-Trial Sigma (variance across time)
% -------------------------------------------------------------------------

fprintf('\n--- Section 4.3: Per-Grid-Cell Per-Trial Sigma (variance across time) ---\n');

% Initialize data collection structure
Grid40SigmaPerTrial = struct();

% Collect per-grid-cell per-trial sigma values
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        Grid40SigmaPerTrial.(condition) = [];
        continue;
    end

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);
    allSigmaValues = [];

    for r = 1:nRecsGrid
        % Skip if in Skip list
        if ismember(r, Skip.(condition))
            continue;
        end

        % Check if P1 exists
        if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
            gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

            % Dereference cell array
            if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
                gridData_P1 = gridData_P1_cell{:};
            else
                gridData_P1 = gridData_P1_cell;
            end

            if ~isempty(gridData_P1)
                % Grid40 data: [gridY × gridX × timepoints × trials]
                [gridY, gridX, nTimepoints, nTrials] = size(gridData_P1);

                % Calculate mean activity per grid cell (for filtering)
                gridData_forMean = reshape(gridData_P1, [gridY*gridX, nTimepoints*nTrials]);
                meanActivity = mean(gridData_forMean, 2, 'omitnan');

                % Reshape to [gridCells × timepoints × trials]
                gridData_reshaped = reshape(gridData_P1, [gridY*gridX, nTimepoints, nTrials]);

                % Subset timeframes for analysis
                gridData_reshaped = gridData_reshaped(:, TimeFrameSelection, :);

                % Calculate variance across time for each grid cell in each trial
                % Result: [gridCells × 1 × trials]
                variances = var(gridData_reshaped, 0, 2, 'omitnan');

                % Convert to sigma and remove singleton dimension
                sigmas = sqrt(variances);
                sigmas = squeeze(sigmas);  % Remove singleton dimension: [gridCells × trials]

                % Filter based on Remove_Low_Values flag
                if Remove_Low_Values
                    % Expand meanActivity to match dimensions
                    meanActivity_expanded = repmat(meanActivity, 1, nTrials);
                    validIdx = (meanActivity_expanded >= DEAD_CELL_THRESHOLD) & ~isnan(sigmas);
                    sigmas_filtered = sigmas(validIdx);
                else
                    % Remove only NaNs
                    sigmas_filtered = sigmas(~isnan(sigmas(:)));
                end

                allSigmaValues = [allSigmaValues; sigmas_filtered(:)];
            end
        end
    end

    % Store in structure
    Grid40SigmaPerTrial.(condition) = allSigmaValues;

    fprintf('  %s: %d per-trial sigma values collected\n', condition, length(allSigmaValues));
end

% Plot 4.3: CDF by condition - Per-Trial Sigma
fprintf('Creating Plot 4.3: CDF by condition (Per-Trial Sigma)\n');

figure('Name', 'Grid40 Per-Trial Sigma - CDF by Condition');
hold on;

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40SigmaPerTrial.(condition);

    if isempty(data)
        continue;
    end

    [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

    plot(xi, f, 'Color', conditionColors.(condition), 'LineWidth', 2, ...
         'DisplayName', sprintf('%s (n=%d)', condition, length(data)));
end

% Add 98th percentile vertical lines for each condition
for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40SigmaPerTrial.(condition);

    if ~isempty(data)
        [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

        % Find 98th percentile using direct index search
        p98_idx = find(f >= 0.98, 1, 'first');
        if ~isempty(p98_idx)
            p98_sigma = xi(p98_idx);
        else
            p98_sigma = NaN;
        end

        if ~isnan(p98_sigma) && p98_sigma >= 0 && p98_sigma <= 3
            xline(p98_sigma, '--', 'Color', conditionColors.(condition), ...
                  'LineWidth', 1.5, 'Alpha', 0.7, 'HandleVisibility', 'off');
        end
    end
end

xlabel('Sigma (Standard Deviation)');
ylabel('Cumulative Probability');
title('Grid40 Per-Grid-Cell Per-Trial Sigma - CDF by Condition');
legend('Location', 'best');
xlim([0 3]);
grid on;
box on;
hold off;

% -------------------------------------------------------------------------
% SECTION 4.4: Whole-Recording Sigma (overall variance)
% -------------------------------------------------------------------------

fprintf('\n--- Section 4.4: Whole-Recording Sigma (overall variance) ---\n');

% Initialize data collection structure
Grid40SigmaWholeRecording = struct();

% Collect whole-recording sigma values (one value per recording)
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        Grid40SigmaWholeRecording.(condition) = [];
        continue;
    end

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);
    allSigmaValues = [];

    for r = 1:nRecsGrid
        % Skip if in Skip list
        if ismember(r, Skip.(condition))
            continue;
        end

        % Check if P1 exists
        if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
            gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

            % Dereference cell array
            if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
                gridData_P1 = gridData_P1_cell{:};
            else
                gridData_P1 = gridData_P1_cell;
            end

            if ~isempty(gridData_P1)
                % Grid40 data: [gridY × gridX × timepoints × trials]
                % Subset timeframes for analysis
                gridData_P1 = gridData_P1(:, :, TimeFrameSelection, :);

                % Flatten entire recording to 1D
                gridData_flat = gridData_P1(:);

                % Filter based on Remove_Low_Values flag
                if Remove_Low_Values
                    gridData_flat = gridData_flat(gridData_flat >= DEAD_CELL_THRESHOLD);
                end

                % Remove NaNs
                gridData_flat = gridData_flat(~isnan(gridData_flat));

                % Calculate overall variance for this recording
                if ~isempty(gridData_flat)
                    variance_rec = var(gridData_flat, 0, 'omitnan');
                    sigma_rec = sqrt(variance_rec);

                    allSigmaValues = [allSigmaValues; sigma_rec];
                end
            end
        end
    end

    % Store in structure
    Grid40SigmaWholeRecording.(condition) = allSigmaValues;

    fprintf('  %s: %d whole-recording sigma values collected\n', condition, length(allSigmaValues));
end

% Plot 4.4: CDF by condition - Whole-Recording Sigma
fprintf('Creating Plot 4.4: CDF by condition (Whole-Recording Sigma)\n');

figure('Name', 'Grid40 Whole-Recording Sigma - CDF by Condition');
hold on;

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40SigmaWholeRecording.(condition);

    if isempty(data)
        continue;
    end

    [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

    plot(xi, f, 'Color', conditionColors.(condition), 'LineWidth', 2, ...
         'DisplayName', sprintf('%s (n=%d)', condition, length(data)));
end

% Add 98th percentile vertical lines for each condition
for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40SigmaWholeRecording.(condition);

    if ~isempty(data)
        [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

        % Find 98th percentile using direct index search
        p98_idx = find(f >= 0.98, 1, 'first');
        if ~isempty(p98_idx)
            p98_sigma = xi(p98_idx);
        else
            p98_sigma = NaN;
        end

        if ~isnan(p98_sigma) && p98_sigma >= 0 && p98_sigma <= 3
            xline(p98_sigma, '--', 'Color', conditionColors.(condition), ...
                  'LineWidth', 1.5, 'Alpha', 0.7, 'HandleVisibility', 'off');
        end
    end
end

xlabel('Sigma (Standard Deviation)');
ylabel('Cumulative Probability');
title('Grid40 Whole-Recording Sigma - CDF by Condition');
legend('Location', 'best');
xlim([0 3]);
grid on;
box on;
hold off;

% -------------------------------------------------------------------------
% SECTION 4.5: Per-Trial Sigma by Animal (separated by individual animals)
% -------------------------------------------------------------------------

fprintf('\n--- Section 4.5: Per-Trial Sigma by Animal ---\n');

% Initialize data collection structure
Grid40SigmaPerTrial_ByAnimal = struct();

% Collect per-trial sigma values grouped by animal
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        Grid40SigmaPerTrial_ByAnimal.(condition) = struct();
        continue;
    end

    % Initialize condition structure
    Grid40SigmaPerTrial_ByAnimal.(condition) = struct();

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);

    for r = 1:nRecsGrid
        % Skip if in Skip list
        if ismember(r, Skip.(condition))
            continue;
        end

        % Get animal ID from Rec structure (using table column access)
        if isfield(Rec, condition) && height(Rec.(condition)) >= r && ...
           ismember('AnimalID', Rec.(condition).Properties.VariableNames)
            % Access AnimalID column with curly braces for cell extraction
            animalID = Rec.(condition).AnimalID{r};
        else
            % Fallback: use condition_Rec# format
            animalID = sprintf('%s_Rec%d', condition, r);
            fprintf('  Warning: No AnimalID for %s recording %d, using %s\n', condition, r, animalID);
        end

        % Make animalID a valid field name
        animalID_field = matlab.lang.makeValidName(animalID);

        % Check if P1 exists
        if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
            gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

            % Dereference cell array
            if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
                gridData_P1 = gridData_P1_cell{:};
            else
                gridData_P1 = gridData_P1_cell;
            end

            if ~isempty(gridData_P1)
                % Grid40 data: [gridY × gridX × timepoints × trials]
                [gridY, gridX, nTimepoints, nTrials] = size(gridData_P1);

                % Calculate mean activity per grid cell (for filtering)
                gridData_forMean = reshape(gridData_P1, [gridY*gridX, nTimepoints*nTrials]);
                meanActivity = mean(gridData_forMean, 2, 'omitnan');

                % Reshape to [gridCells × timepoints × trials]
                gridData_reshaped = reshape(gridData_P1, [gridY*gridX, nTimepoints, nTrials]);

                % Subset timeframes for analysis
                gridData_reshaped = gridData_reshaped(:, TimeFrameSelection, :);

                % Calculate variance across time for each grid cell in each trial
                % Result: [gridCells × 1 × trials]
                variances = var(gridData_reshaped, 0, 2, 'omitnan');

                % Convert to sigma and remove singleton dimension
                sigmas = sqrt(variances);
                sigmas = squeeze(sigmas);  % Remove singleton dimension: [gridCells × trials]

                % Filter based on Remove_Low_Values flag
                if Remove_Low_Values
                    % Expand meanActivity to match dimensions
                    meanActivity_expanded = repmat(meanActivity, 1, nTrials);
                    validIdx = (meanActivity_expanded >= DEAD_CELL_THRESHOLD) & ~isnan(sigmas);
                    sigmas_filtered = sigmas(validIdx);
                else
                    % Remove only NaNs
                    sigmas_filtered = sigmas(~isnan(sigmas(:)));
                end

                % Store or append to animal's data
                if isfield(Grid40SigmaPerTrial_ByAnimal.(condition), animalID_field)
                    Grid40SigmaPerTrial_ByAnimal.(condition).(animalID_field) = ...
                        [Grid40SigmaPerTrial_ByAnimal.(condition).(animalID_field); sigmas_filtered(:)];
                else
                    Grid40SigmaPerTrial_ByAnimal.(condition).(animalID_field) = sigmas_filtered(:);
                end
            end
        end
    end

    % Print summary
    if isstruct(Grid40SigmaPerTrial_ByAnimal.(condition))
        animalFields = fieldnames(Grid40SigmaPerTrial_ByAnimal.(condition));
        fprintf('  %s: %d animals found\n', condition, length(animalFields));
        for a = 1:length(animalFields)
            animalField = animalFields{a};
            nSamples = length(Grid40SigmaPerTrial_ByAnimal.(condition).(animalField));
            fprintf('    %s: %d sigma values\n', animalField, nSamples);
        end
    end
end

% Plot 4.5: CDF by animal - Per-Trial Sigma with subplots per condition
fprintf('Creating Plot 4.5: CDF by animal (Per-Trial Sigma) - Subplots per condition\n');

figure('Name', 'Grid40 Per-Trial Sigma by Animal - CDF by Condition');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for c = 1:length(conditions)
    condition = conditions{c};

    nexttile;
    hold on;

    % Check if we have data for this condition
    if ~isfield(Grid40SigmaPerTrial_ByAnimal, condition) || ...
       ~isstruct(Grid40SigmaPerTrial_ByAnimal.(condition))
        title(sprintf('%s (no data)', condition));
        xlabel('Sigma (Standard Deviation)');
        ylabel('Cumulative Probability');
        xlim([0 3]);
        grid on;
        box on;
        hold off;
        continue;
    end

    % Get animal fields for this condition
    animalFields = fieldnames(Grid40SigmaPerTrial_ByAnimal.(condition));
    nAnimals = length(animalFields);

    if nAnimals == 0
        title(sprintf('%s (no animals)', condition));
        xlabel('Sigma (Standard Deviation)');
        ylabel('Cumulative Probability');
        xlim([0 3]);
        grid on;
        box on;
        hold off;
        continue;
    end

    % Generate distinct colors for each animal
    colorOrder = lines(nAnimals);

    % Plot CDF for each animal
    for a = 1:nAnimals
        animalField = animalFields{a};
        data = Grid40SigmaPerTrial_ByAnimal.(condition).(animalField);

        if isempty(data)
            continue;
        end

        [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

        plot(xi, f, 'Color', colorOrder(a, :), 'LineWidth', 1.5, ...
             'DisplayName', sprintf('%s (n=%d)', animalField, length(data)));
    end

    xlabel('Sigma (Standard Deviation)');
    ylabel('Cumulative Probability');
    title(sprintf('%s (n=%d animals)', condition, nAnimals));
    legend('Location', 'best', 'FontSize', 8);
    xlim([0 3]);
    grid on;
    box on;
    hold off;
end

sgtitle('Grid40 Per-Trial Sigma by Animal - CDF by Condition', 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% SECTION 4.6: Per-Trial Sigma Aggregated by Animal (across all conditions)
% -------------------------------------------------------------------------

fprintf('\n--- Section 4.6: Per-Trial Sigma Aggregated by Animal (all conditions) ---\n');

% Initialize data collection structure
Grid40SigmaPerTrial_AllAnimals = struct();

% Collect per-trial sigma values aggregated by animal across all conditions
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        continue;
    end

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);

    for r = 1:nRecsGrid
        % Skip if in Skip list
        if ismember(r, Skip.(condition))
            continue;
        end

        % Get animal ID from Rec structure (using table column access)
        if isfield(Rec, condition) && height(Rec.(condition)) >= r && ...
           ismember('AnimalID', Rec.(condition).Properties.VariableNames)
            % Access AnimalID column with curly braces for cell extraction
            animalID = Rec.(condition).AnimalID{r};
        else
            % Fallback: use condition_Rec# format
            animalID = sprintf('%s_Rec%d', condition, r);
            fprintf('  Warning: No AnimalID for %s recording %d, using %s\n', condition, r, animalID);
        end

        % Make animalID a valid field name
        animalID_field = matlab.lang.makeValidName(animalID);

        % Check if P1 exists
        if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
            gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

            % Dereference cell array
            if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
                gridData_P1 = gridData_P1_cell{:};
            else
                gridData_P1 = gridData_P1_cell;
            end

            if ~isempty(gridData_P1)
                % Grid40 data: [gridY × gridX × timepoints × trials]
                [gridY, gridX, nTimepoints, nTrials] = size(gridData_P1);

                % Calculate mean activity per grid cell (for filtering)
                gridData_forMean = reshape(gridData_P1, [gridY*gridX, nTimepoints*nTrials]);
                meanActivity = mean(gridData_forMean, 2, 'omitnan');

                % Reshape to [gridCells × timepoints × trials]
                gridData_reshaped = reshape(gridData_P1, [gridY*gridX, nTimepoints, nTrials]);

                % Subset timeframes for analysis
                gridData_reshaped = gridData_reshaped(:, TimeFrameSelection, :);

                % Calculate variance across time for each grid cell in each trial
                % Result: [gridCells × 1 × trials]
                variances = var(gridData_reshaped, 0, 2, 'omitnan');

                % Convert to sigma and remove singleton dimension
                sigmas = sqrt(variances);
                sigmas = squeeze(sigmas);  % Remove singleton dimension: [gridCells × trials]

                % Filter based on Remove_Low_Values flag
                if Remove_Low_Values
                    % Expand meanActivity to match dimensions
                    meanActivity_expanded = repmat(meanActivity, 1, nTrials);
                    validIdx = (meanActivity_expanded >= DEAD_CELL_THRESHOLD) & ~isnan(sigmas);
                    sigmas_filtered = sigmas(validIdx);
                else
                    % Remove only NaNs
                    sigmas_filtered = sigmas(~isnan(sigmas(:)));
                end

                % Append to animal's data (aggregating across all conditions)
                if isfield(Grid40SigmaPerTrial_AllAnimals, animalID_field)
                    Grid40SigmaPerTrial_AllAnimals.(animalID_field) = ...
                        [Grid40SigmaPerTrial_AllAnimals.(animalID_field); sigmas_filtered(:)];
                else
                    Grid40SigmaPerTrial_AllAnimals.(animalID_field) = sigmas_filtered(:);
                end
            end
        end
    end
end

% Print summary
animalFields = fieldnames(Grid40SigmaPerTrial_AllAnimals);
nAnimalsTotal = length(animalFields);
fprintf('\nTotal animals found across all conditions: %d\n', nAnimalsTotal);
for a = 1:nAnimalsTotal
    animalField = animalFields{a};
    nSamples = length(Grid40SigmaPerTrial_AllAnimals.(animalField));
    fprintf('  %s: %d sigma values\n', animalField, nSamples);
end

% Plot 4.6: CDF by animal - All Conditions Aggregated
fprintf('Creating Plot 4.6: CDF by animal (Per-Trial Sigma) - All conditions aggregated\n');

figure('Name', 'Grid40 Per-Trial Sigma by Animal - All Conditions Aggregated');
hold on;

if nAnimalsTotal == 0
    title('No animals found');
    xlabel('Sigma (Standard Deviation)');
    ylabel('Cumulative Probability');
    xlim([0 3]);
    grid on;
    box on;
else
    % Generate distinct colors for each animal
    colorOrder = lines(nAnimalsTotal);

    % Plot CDF for each animal
    for a = 1:nAnimalsTotal
        animalField = animalFields{a};
        data = Grid40SigmaPerTrial_AllAnimals.(animalField);

        if isempty(data)
            continue;
        end

        [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

        plot(xi, f, 'Color', colorOrder(a, :), 'LineWidth', 2, ...
             'DisplayName', sprintf('%s (n=%d)', animalField, length(data)));
    end

    xlabel('Sigma (Standard Deviation)');
    ylabel('Cumulative Probability');
    title(sprintf('Grid40 Per-Trial Sigma by Animal - All Conditions Aggregated (n=%d animals)', nAnimalsTotal));
    legend('Location', 'best', 'FontSize', 9);
    xlim([0 3]);
    grid on;
    box on;
end

hold off;

% -------------------------------------------------------------------------
% SECTION 4.7: Per-Timepoint Sigma by Animal (separated by individual animals)
% -------------------------------------------------------------------------

fprintf('\n--- Section 4.7: Per-Timepoint Sigma by Animal ---\n');

% Initialize data collection structure
Grid40SigmaPerTimepoint_ByAnimal = struct();

% Collect per-timepoint sigma values grouped by animal
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        Grid40SigmaPerTimepoint_ByAnimal.(condition) = struct();
        continue;
    end

    % Initialize condition structure
    Grid40SigmaPerTimepoint_ByAnimal.(condition) = struct();

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);

    for r = 1:nRecsGrid
        % Skip if in Skip list
        if ismember(r, Skip.(condition))
            continue;
        end

        % Get animal ID from Rec structure (using table column access)
        if isfield(Rec, condition) && height(Rec.(condition)) >= r && ...
           ismember('AnimalID', Rec.(condition).Properties.VariableNames)
            % Access AnimalID column with curly braces for cell extraction
            animalID = Rec.(condition).AnimalID{r};
        else
            % Fallback: use condition_Rec# format
            animalID = sprintf('%s_Rec%d', condition, r);
            fprintf('  Warning: No AnimalID for %s recording %d, using %s\n', condition, r, animalID);
        end

        % Make animalID a valid field name
        animalID_field = matlab.lang.makeValidName(animalID);

        % Check if P1 exists
        if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
            gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

            % Dereference cell array
            if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
                gridData_P1 = gridData_P1_cell{:};
            else
                gridData_P1 = gridData_P1_cell;
            end

            if ~isempty(gridData_P1)
                % Grid40 data: [gridY × gridX × timepoints × trials]
                [gridY, gridX, nTimepoints, nTrials] = size(gridData_P1);

                % Calculate mean activity per grid cell (for filtering)
                gridData_forMean = reshape(gridData_P1, [gridY*gridX, nTimepoints*nTrials]);
                meanActivity = mean(gridData_forMean, 2, 'omitnan');

                % Reshape to [gridCells × timepoints × trials]
                gridData_reshaped = reshape(gridData_P1, [gridY*gridX, nTimepoints, nTrials]);

                % Subset timeframes for analysis
                gridData_reshaped = gridData_reshaped(:, TimeFrameSelection, :);
                nTimepoints_selected = length(TimeFrameSelection);

                % Calculate variance across TRIALS for each grid cell at each timepoint
                % Result: [gridCells × timepoints_selected × 1]
                variances = var(gridData_reshaped, 0, 3, 'omitnan');

                % Convert to sigma and remove singleton dimension
                sigmas = sqrt(variances);
                sigmas = squeeze(sigmas);  % Remove singleton dimension: [gridCells × timepoints_selected]

                % Filter based on Remove_Low_Values flag
                if Remove_Low_Values
                    % Expand meanActivity to match dimensions
                    meanActivity_expanded = repmat(meanActivity, 1, nTimepoints_selected);
                    validIdx = (meanActivity_expanded >= DEAD_CELL_THRESHOLD) & ~isnan(sigmas);
                    sigmas_filtered = sigmas(validIdx);
                else
                    % Remove only NaNs
                    sigmas_filtered = sigmas(~isnan(sigmas(:)));
                end

                % Store or append to animal's data
                if isfield(Grid40SigmaPerTimepoint_ByAnimal.(condition), animalID_field)
                    Grid40SigmaPerTimepoint_ByAnimal.(condition).(animalID_field) = ...
                        [Grid40SigmaPerTimepoint_ByAnimal.(condition).(animalID_field); sigmas_filtered(:)];
                else
                    Grid40SigmaPerTimepoint_ByAnimal.(condition).(animalID_field) = sigmas_filtered(:);
                end
            end
        end
    end

    % Print summary
    if isstruct(Grid40SigmaPerTimepoint_ByAnimal.(condition))
        animalFields = fieldnames(Grid40SigmaPerTimepoint_ByAnimal.(condition));
        fprintf('  %s: %d animals found\n', condition, length(animalFields));
        for a = 1:length(animalFields)
            animalField = animalFields{a};
            nSamples = length(Grid40SigmaPerTimepoint_ByAnimal.(condition).(animalField));
            fprintf('    %s: %d sigma values\n', animalField, nSamples);
        end
    end
end

% Plot 4.7: CDF by animal - Per-Timepoint Sigma with subplots per condition
fprintf('Creating Plot 4.7: CDF by animal (Per-Timepoint Sigma) - Subplots per condition\n');

figure('Name', 'Grid40 Per-Timepoint Sigma by Animal - CDF by Condition');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for c = 1:length(conditions)
    condition = conditions{c};

    nexttile;
    hold on;

    % Check if we have data for this condition
    if ~isfield(Grid40SigmaPerTimepoint_ByAnimal, condition) || ...
       ~isstruct(Grid40SigmaPerTimepoint_ByAnimal.(condition))
        title(sprintf('%s (no data)', condition));
        xlabel('Sigma (Standard Deviation)');
        ylabel('Cumulative Probability');
        xlim([0 3]);
        grid on;
        box on;
        hold off;
        continue;
    end

    % Get animal fields for this condition
    animalFields = fieldnames(Grid40SigmaPerTimepoint_ByAnimal.(condition));
    nAnimals = length(animalFields);

    if nAnimals == 0
        title(sprintf('%s (no animals)', condition));
        xlabel('Sigma (Standard Deviation)');
        ylabel('Cumulative Probability');
        xlim([0 3]);
        grid on;
        box on;
        hold off;
        continue;
    end

    % Generate distinct colors for each animal
    colorOrder = lines(nAnimals);

    % Plot CDF for each animal
    for a = 1:nAnimals
        animalField = animalFields{a};
        data = Grid40SigmaPerTimepoint_ByAnimal.(condition).(animalField);

        if isempty(data)
            continue;
        end

        [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

        plot(xi, f, 'Color', colorOrder(a, :), 'LineWidth', 1.5, ...
             'DisplayName', sprintf('%s (n=%d)', animalField, length(data)));
    end

    xlabel('Sigma (Standard Deviation)');
    ylabel('Cumulative Probability');
    title(sprintf('%s (n=%d animals)', condition, nAnimals));
    legend('Location', 'best', 'FontSize', 8);
    xlim([0 3]);
    grid on;
    box on;
    hold off;
end

sgtitle('Grid40 Per-Timepoint Sigma by Animal - CDF by Condition', 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% SECTION 4.8: Per-Timepoint Sigma Aggregated by Animal (across all conditions)
% -------------------------------------------------------------------------

fprintf('\n--- Section 4.8: Per-Timepoint Sigma Aggregated by Animal (all conditions) ---\n');

% Initialize data collection structure
Grid40SigmaPerTimepoint_AllAnimals = struct();

% Collect per-timepoint sigma values aggregated by animal across all conditions
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        continue;
    end

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);

    for r = 1:nRecsGrid
        % Skip if in Skip list
        if ismember(r, Skip.(condition))
            continue;
        end

        % Get animal ID from Rec structure (using table column access)
        if isfield(Rec, condition) && height(Rec.(condition)) >= r && ...
           ismember('AnimalID', Rec.(condition).Properties.VariableNames)
            % Access AnimalID column with curly braces for cell extraction
            animalID = Rec.(condition).AnimalID{r};
        else
            % Fallback: use condition_Rec# format
            animalID = sprintf('%s_Rec%d', condition, r);
            fprintf('  Warning: No AnimalID for %s recording %d, using %s\n', condition, r, animalID);
        end

        % Make animalID a valid field name
        animalID_field = matlab.lang.makeValidName(animalID);

        % Check if P1 exists
        if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
            gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

            % Dereference cell array
            if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
                gridData_P1 = gridData_P1_cell{:};
            else
                gridData_P1 = gridData_P1_cell;
            end

            if ~isempty(gridData_P1)
                % Grid40 data: [gridY × gridX × timepoints × trials]
                [gridY, gridX, nTimepoints, nTrials] = size(gridData_P1);

                % Calculate mean activity per grid cell (for filtering)
                gridData_forMean = reshape(gridData_P1, [gridY*gridX, nTimepoints*nTrials]);
                meanActivity = mean(gridData_forMean, 2, 'omitnan');

                % Reshape to [gridCells × timepoints × trials]
                gridData_reshaped = reshape(gridData_P1, [gridY*gridX, nTimepoints, nTrials]);

                % Subset timeframes for analysis
                gridData_reshaped = gridData_reshaped(:, TimeFrameSelection, :);
                nTimepoints_selected = length(TimeFrameSelection);

                % Calculate variance across TRIALS for each grid cell at each timepoint
                % Result: [gridCells × timepoints_selected × 1]
                variances = var(gridData_reshaped, 0, 3, 'omitnan');

                % Convert to sigma and remove singleton dimension
                sigmas = sqrt(variances);
                sigmas = squeeze(sigmas);  % Remove singleton dimension: [gridCells × timepoints_selected]

                % Filter based on Remove_Low_Values flag
                if Remove_Low_Values
                    % Expand meanActivity to match dimensions
                    meanActivity_expanded = repmat(meanActivity, 1, nTimepoints_selected);
                    validIdx = (meanActivity_expanded >= DEAD_CELL_THRESHOLD) & ~isnan(sigmas);
                    sigmas_filtered = sigmas(validIdx);
                else
                    % Remove only NaNs
                    sigmas_filtered = sigmas(~isnan(sigmas(:)));
                end

                % Append to animal's data (aggregating across all conditions)
                if isfield(Grid40SigmaPerTimepoint_AllAnimals, animalID_field)
                    Grid40SigmaPerTimepoint_AllAnimals.(animalID_field) = ...
                        [Grid40SigmaPerTimepoint_AllAnimals.(animalID_field); sigmas_filtered(:)];
                else
                    Grid40SigmaPerTimepoint_AllAnimals.(animalID_field) = sigmas_filtered(:);
                end
            end
        end
    end
end

% Print summary
animalFields = fieldnames(Grid40SigmaPerTimepoint_AllAnimals);
nAnimalsTotal = length(animalFields);
fprintf('\nTotal animals found across all conditions: %d\n', nAnimalsTotal);
for a = 1:nAnimalsTotal
    animalField = animalFields{a};
    nSamples = length(Grid40SigmaPerTimepoint_AllAnimals.(animalField));
    fprintf('  %s: %d sigma values\n', animalField, nSamples);
end

% Plot 4.8: CDF by animal - All Conditions Aggregated (Per-Timepoint)
fprintf('Creating Plot 4.8: CDF by animal (Per-Timepoint Sigma) - All conditions aggregated\n');

figure('Name', 'Grid40 Per-Timepoint Sigma by Animal - All Conditions Aggregated');
hold on;

if nAnimalsTotal == 0
    title('No animals found');
    xlabel('Sigma (Standard Deviation)');
    ylabel('Cumulative Probability');
    xlim([0 3]);
    grid on;
    box on;
else
    % Generate distinct colors for each animal
    colorOrder = lines(nAnimalsTotal);

    % Plot CDF for each animal
    for a = 1:nAnimalsTotal
        animalField = animalFields{a};
        data = Grid40SigmaPerTimepoint_AllAnimals.(animalField);

        if isempty(data)
            continue;
        end

        [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

        plot(xi, f, 'Color', colorOrder(a, :), 'LineWidth', 2, ...
             'DisplayName', sprintf('%s (n=%d)', animalField, length(data)));
    end

    xlabel('Sigma (Standard Deviation)');
    ylabel('Cumulative Probability');
    title(sprintf('Grid40 Per-Timepoint Sigma by Animal - All Conditions Aggregated (n=%d animals)', nAnimalsTotal));
    legend('Location', 'best', 'FontSize', 9);
    xlim([0 3]);
    grid on;
    box on;
end

hold off;

% -------------------------------------------------------------------------
% SECTION 4.9: Per-Timepoint Sigma - Expert vs NoSpout (by Animal, no individual ID)
% -------------------------------------------------------------------------

fprintf('\n--- Section 4.9: Per-Timepoint Sigma - Expert vs NoSpout ---\n');

% This section reuses data from Section 4.7
% Plot Expert vs NoSpout animals with thin lines in condition colors

% Plot 4.9: Expert vs NoSpout comparison (thin lines per animal)
fprintf('Creating Plot 4.9: Expert vs NoSpout (Per-Timepoint Sigma) - Thin lines per animal\n');

figure('Name', 'Grid40 Per-Timepoint Sigma - Expert vs NoSpout');
hold on;

% Plot Expert animals (thin lines in Expert color)
if isfield(Grid40SigmaPerTimepoint_ByAnimal, 'Expert') && ...
   isstruct(Grid40SigmaPerTimepoint_ByAnimal.Expert)

    expertAnimals = fieldnames(Grid40SigmaPerTimepoint_ByAnimal.Expert);
    nExpertAnimals = length(expertAnimals);

    fprintf('  Plotting %d Expert animals\n', nExpertAnimals);

    for a = 1:nExpertAnimals
        animalField = expertAnimals{a};
        data = Grid40SigmaPerTimepoint_ByAnimal.Expert.(animalField);

        if ~isempty(data)
            [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');
            plot(xi, f, 'Color', conditionColors.Expert, 'LineWidth', 0.5, ...
                 'HandleVisibility', 'off');
        end
    end
end

% Plot NoSpout animals (thin lines in NoSpout color)
if isfield(Grid40SigmaPerTimepoint_ByAnimal, 'NoSpout') && ...
   isstruct(Grid40SigmaPerTimepoint_ByAnimal.NoSpout)

    noSpoutAnimals = fieldnames(Grid40SigmaPerTimepoint_ByAnimal.NoSpout);
    nNoSpoutAnimals = length(noSpoutAnimals);

    fprintf('  Plotting %d NoSpout animals\n', nNoSpoutAnimals);

    for a = 1:nNoSpoutAnimals
        animalField = noSpoutAnimals{a};
        data = Grid40SigmaPerTimepoint_ByAnimal.NoSpout.(animalField);

        if ~isempty(data)
            [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');
            plot(xi, f, 'Color', conditionColors.NoSpout, 'LineWidth', 0.5, ...
                 'HandleVisibility', 'off');
        end
    end
end

% Add legend with dummy plots for conditions only (thicker lines)
plot(NaN, NaN, 'Color', conditionColors.Expert, 'LineWidth', 2, ...
     'DisplayName', 'Expert');
plot(NaN, NaN, 'Color', conditionColors.NoSpout, 'LineWidth', 2, ...
     'DisplayName', 'NoSpout');

xlabel('Sigma (Standard Deviation)');
ylabel('Cumulative Probability');
title('Grid40 Per-Timepoint Sigma - Expert vs NoSpout (by Animal)');
legend('Location', 'best', 'FontSize', 10);
xlim([0 3]);
grid on;
box on;
hold off;

fprintf('\nSection 4 complete: Grid40 variance/sigma analysis plots created\n');

fprintf('Section 3 complete: Grid40 raw activity distribution plots created\n');

end  % RUN.section4

%% =========================================================================
%% SECTION 3.X: Gaussian Mixture Model Fitting to Mirrored Distributions
%% =========================================================================

if fitGMM
    fprintf('\n--- Section 3.X: GMM Fitting to Mirrored Activity Distributions ---\n');

    % Initialize storage for GMM results
    GMM_Results = struct();

    % Prepare conditions list including "All"
    gmm_conditions = [conditions, {'AllConditions'}];

    for gmm_c = 1:length(gmm_conditions)
        gmm_condition = gmm_conditions{gmm_c};

        fprintf('\nFitting GMM for: %s\n', gmm_condition);

        % Collect all Grid40 activity data for this condition
        all_activity_data = [];

        if strcmp(gmm_condition, 'AllConditions')
            % Aggregate across all conditions
            for c = 1:length(conditions)
                cond = conditions{c};
                condIndividual = [cond 'Individual'];

                if ~isfield(Grid40, condIndividual)
                    continue;
                end

                nRecs = length(Grid40.(condIndividual).AllNeurons);

                for r = 1:nRecs
                    if ismember(r, Skip.(cond))
                        continue;
                    end

                    if ~isfield(Grid40.(condIndividual).AllNeurons(r), 'P1') || ...
                       isempty(Grid40.(condIndividual).AllNeurons(r).P1)
                        continue;
                    end

                    gridData_cell = Grid40.(condIndividual).AllNeurons(r).P1;
                    if iscell(gridData_cell) && ~isempty(gridData_cell)
                        gridData = gridData_cell{:};
                    else
                        gridData = gridData_cell;
                    end

                    activity_vec = gridData(:);
                    validIdx = ~isnan(activity_vec);
                    if Remove_Low_Values
                        validIdx = validIdx & (activity_vec >= DEAD_CELL_THRESHOLD);
                    end
                    all_activity_data = [all_activity_data; activity_vec(validIdx)];
                end
            end
        else
            % Single condition
            condIndividual = [gmm_condition 'Individual'];

            if isfield(Grid40, condIndividual)
                nRecs = length(Grid40.(condIndividual).AllNeurons);

                for r = 1:nRecs
                    if ismember(r, Skip.(gmm_condition))
                        continue;
                    end

                    if ~isfield(Grid40.(condIndividual).AllNeurons(r), 'P1') || ...
                       isempty(Grid40.(condIndividual).AllNeurons(r).P1)
                        continue;
                    end

                    gridData_cell = Grid40.(condIndividual).AllNeurons(r).P1;
                    if iscell(gridData_cell) && ~isempty(gridData_cell)
                        gridData = gridData_cell{:};
                    else
                        gridData = gridData_cell;
                    end

                    activity_vec = gridData(:);
                    validIdx = ~isnan(activity_vec);
                    if Remove_Low_Values
                        validIdx = validIdx & (activity_vec >= DEAD_CELL_THRESHOLD);
                    end
                    all_activity_data = [all_activity_data; activity_vec(validIdx)];
                end
            end
        end

        if isempty(all_activity_data)
            fprintf('  Warning: No activity data found for %s. Skipping.\n', gmm_condition);
            continue;
        end

        fprintf('  Collected %d activity values\n', length(all_activity_data));

        % Mirror the data around 0: combined = [original, -original]
        mirrored_data = [-all_activity_data; all_activity_data];
        fprintf('  Mirrored data size: %d (original: %d, mirrored: %d)\n', ...
                length(mirrored_data), length(all_activity_data), length(all_activity_data));

        % Sample data for fitting if dataset is too large
        % Use ~50,000 samples for fitting to ensure convergence
        max_fit_samples = 50000;
        if length(mirrored_data) > max_fit_samples
            sample_idx = randsample(length(mirrored_data), max_fit_samples);
            mirrored_data_fit = mirrored_data(sample_idx);
            fprintf('  Sampling %d points for fitting (from %.0f total)\n', max_fit_samples, length(mirrored_data));
        else
            mirrored_data_fit = mirrored_data;
        end

        % Fit Gaussian Mixture Model with 3 components
        try
            % Set fitting options to improve convergence
            options = statset('MaxIter', 500, 'TolFun', 1e-6);
            gmdist = fitgmdist(mirrored_data_fit, 3, 'Options', options);

            % Store GMM results
            GMM_Results.(gmm_condition).gmdist = gmdist;
            GMM_Results.(gmm_condition).means = gmdist.mu(:)';  % Ensure row vector

            % Extract sigmas from 3D covariance array (Sigma is d×d×k for k components)
            sigmas = zeros(1, gmdist.NumComponents);
            for k = 1:gmdist.NumComponents
                sigmas(k) = sqrt(gmdist.Sigma(1, 1, k));
            end
            GMM_Results.(gmm_condition).sigmas = sigmas;

            GMM_Results.(gmm_condition).weights = gmdist.ComponentProportion(:)';  % Ensure row vector

            fprintf('  GMM fit complete.\n');

            % Create visualization: Histogram + Gaussian curves
            fig = figure('Name', sprintf('GMM Fit - %s', gmm_condition));
            hold on;

            % Plot histogram
            bin_edges = linspace(min(mirrored_data), max(mirrored_data), 50);
            histogram(mirrored_data, bin_edges, 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k', 'FaceAlpha', 0.5);

            % Create x-values for Gaussian curves
            x_range = linspace(min(mirrored_data), max(mirrored_data), 1000);

            % Plot individual Gaussian components
            colors = [1 0 0; 0 1 0; 0 0 1];  % RGB for 3 components
            component_labels = {};

            for k = 1:3
                % Gaussian PDF: weight * (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))
                mu = gmdist.mu(k);
                sigma = sqrt(gmdist.Sigma(1, 1, k));  % Extract from 3D covariance array
                weight = gmdist.ComponentProportion(k);

                pdf_k = weight * normpdf(x_range, mu, sigma);
                plot(x_range, pdf_k, 'Color', colors(k,:), 'LineWidth', 2);

                component_labels{k} = sprintf('Component %d: μ=%.3f, σ=%.3f, w=%.3f', ...
                                             k, mu, sigma, weight);
            end

            % Formatting
            xlabel('Activity Value', 'FontSize', 12);
            ylabel('Probability Density', 'FontSize', 12);
            title(sprintf('GMM Fit (3 Gaussians) - %s\nOriginal + Mirrored Data', gmm_condition), 'FontSize', 13);
            legend([{'Data (Histogram)'}, component_labels], 'Location', 'best', 'FontSize', 10);
            grid on;
            hold off;

        catch ME
            fprintf('  Error fitting GMM for %s: %s\n', gmm_condition, ME.message);
        end
    end

    fprintf('\nSection 3.X complete: GMM fitting finished\n');
end

%% =========================================================================
%% SECTION 5: Animal-Level Comparison
%% =========================================================================
if RUN.section5

fprintf('\n--- Section 5: Animal-Level Comparison ---\n');

% Collect animal-specific data
AnimalData_ActivityRaw = struct();
AnimalData_Grid40Raw = struct();

% Get unique animals across all conditions
allAnimals = {};
for c = 1:length(conditions)
    condition = conditions{c};
    if isfield(Rec, condition)
        animalIDs = Rec.(condition).AnimalID;
        allAnimals = [allAnimals; animalIDs];
    end
end
uniqueAnimals = unique(allAnimals);

fprintf('Found %d unique animals across all conditions\n', length(uniqueAnimals));

% Define animal colors (rainbow colormap)
animalColorMap = lines(length(uniqueAnimals));

% Collect data for each animal
for animalIdx = 1:length(uniqueAnimals)
    currentAnimal = uniqueAnimals{animalIdx};
    animalField = ['Animal_' currentAnimal];

    % Initialize storage
    AnimalData_ActivityRaw.(animalField) = [];
    AnimalData_Grid40Raw.(animalField) = [];

    % Loop through conditions to find recordings for this animal
    for c = 1:length(conditions)
        condition = conditions{c};
        conditionIndividual = [condition 'Individual'];

        if ~isfield(Rec, condition)
            continue;
        end

        animalIDs = Rec.(condition).AnimalID;
        animalRecordings = find(strcmp(animalIDs, currentAnimal));

        % Process each recording for this animal
        for recIdx = 1:length(animalRecordings)
            r = animalRecordings(recIdx);

            % Skip if in Skip list
            if ismember(r, Skip.(condition))
                continue;
            end

            % ActivityData raw
            % Guard against r exceeding ActivityData length (Rec table can
            % have more entries than ActivityData for some conditions).
            if isfield(ActivityData, condition) && r <= length(ActivityData.(condition)) && ...
               isfield(ActivityData.(condition)(r), 'Position1') && ...
               ~isempty(ActivityData.(condition)(r).Position1)
                data_P1 = ActivityData.(condition)(r).Position1;
                AnimalData_ActivityRaw.(animalField) = [AnimalData_ActivityRaw.(animalField); data_P1(:)];
            end

            % Grid40 data
            if isfield(Grid40, conditionIndividual)
                nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);

                if r <= nRecsGrid && isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
                    gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

                    if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
                        gridData_P1 = gridData_P1_cell{:};
                    else
                        gridData_P1 = gridData_P1_cell;
                    end

                    if ~isempty(gridData_P1)
                        % Grid40 raw
                        AnimalData_Grid40Raw.(animalField) = [AnimalData_Grid40Raw.(animalField); gridData_P1(:)];
                    end
                end
            end
        end
    end

    % Remove NaN values
    AnimalData_ActivityRaw.(animalField) = AnimalData_ActivityRaw.(animalField)(~isnan(AnimalData_ActivityRaw.(animalField)));
    AnimalData_Grid40Raw.(animalField) = AnimalData_Grid40Raw.(animalField)(~isnan(AnimalData_Grid40Raw.(animalField)));

    % Remove low values (dead/empty cells) if flag is enabled
    if Remove_Low_Values
        AnimalData_ActivityRaw.(animalField) = AnimalData_ActivityRaw.(animalField)(AnimalData_ActivityRaw.(animalField) >= DEAD_CELL_THRESHOLD);
        AnimalData_Grid40Raw.(animalField) = AnimalData_Grid40Raw.(animalField)(AnimalData_Grid40Raw.(animalField) >= DEAD_CELL_THRESHOLD);
    end

    fprintf('  Animal %s: %d raw values collected\n', ...
        currentAnimal, ...
        length(AnimalData_ActivityRaw.(animalField)));
end

%% Plot 5.1: ActivityData raw activity by animal
fprintf('Creating Plot 5.1: ActivityData raw activity by animal\n');

figure('Name', 'ActivityData Raw - CDF by Animal');
hold on;

for animalIdx = 1:length(uniqueAnimals)
    currentAnimal = uniqueAnimals{animalIdx};
    animalField = ['Animal_' currentAnimal];
    data = AnimalData_ActivityRaw.(animalField);

    if isempty(data)
        continue;
    end

    [f, xi] = ksdensity(data, kde_xi_5, 'Function', 'cdf');
    plot(xi, f, 'Color', animalColorMap(animalIdx, :), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('Animal %s (n=%d)', currentAnimal, length(data)));
end

xlabel('Activity Value');
ylabel('Cumulative Probability');
title('ActivityData Raw Activity - CDF by Animal (0-5 range)');
legend('Location', 'best', 'NumColumns', 2);
xlim([0 5]);
grid on;
box on;
hold off;

%% Plot 5.2: Grid40 raw activity by animal
fprintf('Creating Plot 5.2: Grid40 raw activity by animal\n');

figure('Name', 'Grid40 Raw - CDF by Animal');
hold on;

for animalIdx = 1:length(uniqueAnimals)
    currentAnimal = uniqueAnimals{animalIdx};
    animalField = ['Animal_' currentAnimal];
    data = AnimalData_Grid40Raw.(animalField);

    if isempty(data)
        continue;
    end

    [f, xi] = ksdensity(data, kde_xi_5, 'Function', 'cdf');
    plot(xi, f, 'Color', animalColorMap(animalIdx, :), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('Animal %s (n=%d)', currentAnimal, length(data)));
end

xlabel('Grid Activity Value');
ylabel('Cumulative Probability');
title('Grid40 Raw Activity - CDF by Animal (0-5 range)');
legend('Location', 'best', 'NumColumns', 2);
xlim([0 5]);
grid on;
box on;
hold off;

fprintf('Section 5 complete: Animal-level comparison plots created\n');

end  % RUN.section5

%% =========================================================================
%% SECTION 6: Visual Threshold Comparison
%% =========================================================================
if RUN.section6

fprintf('\n--- Section 6: Visual Threshold Comparison ---\n');

% This section creates a visual demonstration of how different sigma thresholds
% affect binarization using real data from Expert condition, Recording 5, Trial 1

% Check if Expert condition and recording 5 exist
if ~isfield(Grid40, 'ExpertIndividual')
    fprintf('Warning: No ExpertIndividual data found in Grid40. Skipping Section 7.\n');
else
    % Determine which recording to use
    if length(Grid40.ExpertIndividual.AllNeurons) < 5
        fprintf('Warning: Recording 5 not found in Expert condition. Using recording 1 instead.\n');
        recIdx = 1;
    else
        recIdx = 5;
    end

    % Check if P1 data exists
    if ~isfield(Grid40.ExpertIndividual.AllNeurons(recIdx), 'P1') || ...
       isempty(Grid40.ExpertIndividual.AllNeurons(recIdx).P1)
        fprintf('Warning: No P1 data for Expert recording %d. Skipping Section 7.\n', recIdx);
    else
        fprintf('Using Expert condition, Recording %d, Trial 1, P1\n', recIdx);

        % Extract Grid40 data
        gridData_cell = Grid40.ExpertIndividual.AllNeurons(recIdx).P1;
        if iscell(gridData_cell) && ~isempty(gridData_cell)
            gridData = gridData_cell{:};  % [gridY × gridX × timepoints × trials]
        else
            gridData = gridData_cell;
        end

        % Get dimensions
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
        nGridCells = gridY * gridX;

        fprintf('  Data dimensions: %d×%d grid, %d timepoints, %d trials\n', ...
                gridY, gridX, nTimepoints, nTrials);

        % Extract trial 1
        if nTrials < 1
            fprintf('Warning: No trials found. Skipping Section 7.\n');
        else
            dataTrial1 = gridData(:, :, :, 1);  % [gridY × gridX × timepoints]

            % Reshape to [gridCells × timepoints]
            dataFlat = reshape(dataTrial1, [nGridCells, nTimepoints]);

            % Remove NaN values for threshold calculation
            validData = dataFlat(~isnan(dataFlat));

            % Remove low values (dead/empty cells) if flag is enabled
            if Remove_Low_Values
                validData = validData(validData >= DEAD_CELL_THRESHOLD);
            end

            % Calculate mean and std
            meanVal = mean(validData);
            stdVal = std(validData);

            fprintf('  Data statistics: mean=%.4f, std=%.4f\n', meanVal, stdVal);

            % Define SIGMA-based thresholds (relative: mean + k·σ)
            thresh_1p0 = meanVal + 1.0 * stdVal;
            thresh_1p5 = meanVal + 1.5 * stdVal;
            thresh_2p0 = meanVal + 2.0 * stdVal;

            % Define ABSOLUTE thresholds (dF/F, matching the Ising pipeline)
            abs_thresh_1 = 1.0;
            abs_thresh_1p5 = 1.5;
            abs_thresh_2 = 2.0;  % Production cutoff (Figure5_dataAggregation.m:41)

            fprintf('  Sigma thresholds: 1.0σ=%.4f, 1.5σ=%.4f, 2.0σ=%.4f\n', ...
                    thresh_1p0, thresh_1p5, thresh_2p0);
            fprintf('  Absolute thresholds: 1.0, 1.5, 2.0 dF/F\n');

            % Sort grid cells by strength per frame (strongest on top)
            dataSorted = zeros(size(dataFlat));
            dataBin_1p0 = zeros(size(dataFlat));  % sigma-based
            dataBin_1p5 = zeros(size(dataFlat));
            dataBin_2p0 = zeros(size(dataFlat));
            dataBin_abs1 = zeros(size(dataFlat));  % absolute-based
            dataBin_abs1p5 = zeros(size(dataFlat));
            dataBin_abs2 = zeros(size(dataFlat));

            for t = 1:nTimepoints
                % Sort this timeframe
                frameData = dataFlat(:, t);
                [sortedVals, sortIdx] = sort(frameData, 'descend', 'MissingPlacement', 'last');

                % Store sorted raw values
                dataSorted(:, t) = sortedVals;

                % Sigma-based binarised versions
                dataBin_1p0(:, t) = sortedVals > thresh_1p0;
                dataBin_1p5(:, t) = sortedVals > thresh_1p5;
                dataBin_2p0(:, t) = sortedVals > thresh_2p0;

                % Absolute-threshold binarised versions
                dataBin_abs1(:, t) = sortedVals > abs_thresh_1;
                dataBin_abs1p5(:, t) = sortedVals > abs_thresh_1p5;
                dataBin_abs2(:, t) = sortedVals > abs_thresh_2;
            end

            % Calculate sparsity for each threshold
            pct_1p0 = 100 * sum(dataBin_1p0(:)) / numel(dataBin_1p0);
            pct_1p5 = 100 * sum(dataBin_1p5(:)) / numel(dataBin_1p5);
            pct_2p0 = 100 * sum(dataBin_2p0(:)) / numel(dataBin_2p0);
            pct_abs1 = 100 * sum(dataBin_abs1(:)) / numel(dataBin_abs1);
            pct_abs1p5 = 100 * sum(dataBin_abs1p5(:)) / numel(dataBin_abs1p5);
            pct_abs2 = 100 * sum(dataBin_abs2(:)) / numel(dataBin_abs2);

            fprintf('  Sparsity (sigma): 1.0σ=%.1f%%, 1.5σ=%.1f%%, 2.0σ=%.1f%% active\n', ...
                    pct_1p0, pct_1p5, pct_2p0);
            fprintf('  Sparsity (absolute): 1.0=%.1f%%, 1.5=%.1f%%, 2.0=%.1f%% active\n', ...
                    pct_abs1, pct_abs1p5, pct_abs2);

            % Create 2x4 panel figure: raw + 3 absolute-threshold + 3 sigma-threshold panels
            figure('Name', 'Visual Threshold Comparison - Expert Rec5 Trial1');
            tiledlayout(2, 4);

            % Tile 1 (row 1, col 1): Raw data (sorted)
            nexttile;
            imagesc(dataSorted);
            colormap(gca, 'hot');
            cb1 = colorbar;
            cb1.Label.String = 'Activity';
            xlabel('Timeframe');
            ylabel('Grid Cells (sorted by strength)');
            title('Raw Data (sorted per frame)');
            clim([0, prctile(dataSorted(:), 99)]);

            % Tile 2 (row 1, col 2): Absolute 1.0 dF/F
            nexttile;
            imagesc(dataBin_abs1);
            colormap(gca, 'gray');
            cb = colorbar; cb.Ticks = [0 1]; cb.TickLabels = {'Inactive', 'Active'};
            xlabel('Timeframe');
            ylabel('Grid Cells (sorted)');
            title(sprintf('Absolute 1.0 dF/F\n%.1f%% active', pct_abs1));

            % Tile 3 (row 1, col 3): Absolute 1.5 dF/F
            nexttile;
            imagesc(dataBin_abs1p5);
            colormap(gca, 'gray');
            cb = colorbar; cb.Ticks = [0 1]; cb.TickLabels = {'Inactive', 'Active'};
            xlabel('Timeframe');
            ylabel('Grid Cells (sorted)');
            title(sprintf('Absolute 1.5 dF/F\n%.1f%% active', pct_abs1p5));

            % Tile 4 (row 1, col 4): Absolute 2.0 dF/F — ISING PRODUCTION CUTOFF
            nexttile;
            imagesc(dataBin_abs2);
            colormap(gca, 'gray');
            cb = colorbar; cb.Ticks = [0 1]; cb.TickLabels = {'Inactive', 'Active'};
            xlabel('Timeframe');
            ylabel('Grid Cells (sorted)');
            title(sprintf('Absolute 2.0 dF/F (Ising)\n%.1f%% active', pct_abs2), ...
                  'Color', [0 0.5 0], 'FontWeight', 'bold');
            % Highlight this panel with a green border
            ax = gca;
            ax.XColor = [0 0.5 0]; ax.YColor = [0 0.5 0]; ax.LineWidth = 2.0;

            % Tile 5 (row 2, col 1): Summary text comparing absolute vs sigma sparsity
            nexttile;
            axis off;
            summary_text = {
                sprintf('Trial statistics:');
                sprintf('  mean = %.3f, σ = %.3f', meanVal, stdVal);
                '';
                sprintf('Absolute → Sigma equivalent:');
                sprintf('  1.0 dF/F ≈ %.2fσ', (abs_thresh_1 - meanVal)/stdVal);
                sprintf('  1.5 dF/F ≈ %.2fσ', (abs_thresh_1p5 - meanVal)/stdVal);
                sprintf('  2.0 dF/F ≈ %.2fσ', (abs_thresh_2 - meanVal)/stdVal);
                '';
                sprintf('Sparsity match at 2.0 dF/F:');
                sprintf('  Absolute: %.1f%%', pct_abs2);
                sprintf('  2.0σ:     %.1f%%', pct_2p0);
                };
            text(0.05, 0.95, summary_text, 'Units', 'normalized', ...
                 'VerticalAlignment', 'top', 'FontName', 'monospaced', 'FontSize', 9);

            % Tile 6 (row 2, col 2): 1.0σ threshold
            nexttile;
            imagesc(dataBin_1p0);
            colormap(gca, 'gray');
            cb = colorbar; cb.Ticks = [0 1]; cb.TickLabels = {'Inactive', 'Active'};
            xlabel('Timeframe');
            ylabel('Grid Cells (sorted)');
            title(sprintf('1.0σ (%.3f)\n%.1f%% active', thresh_1p0, pct_1p0));

            % Tile 7 (row 2, col 3): 1.5σ threshold
            nexttile;
            imagesc(dataBin_1p5);
            colormap(gca, 'gray');
            cb = colorbar; cb.Ticks = [0 1]; cb.TickLabels = {'Inactive', 'Active'};
            xlabel('Timeframe');
            ylabel('Grid Cells (sorted)');
            title(sprintf('1.5σ (%.3f)\n%.1f%% active', thresh_1p5, pct_1p5));

            % Tile 8 (row 2, col 4): 2.0σ threshold
            nexttile;
            imagesc(dataBin_2p0);
            colormap(gca, 'gray');
            cb = colorbar; cb.Ticks = [0 1]; cb.TickLabels = {'Inactive', 'Active'};
            xlabel('Timeframe');
            ylabel('Grid Cells (sorted)');
            title(sprintf('2.0σ (%.3f)\n%.1f%% active', thresh_2p0, pct_2p0));

            sgtitle(sprintf(['Visual Threshold Comparison: Expert Recording %d, Trial 1 (P1)  |  ' ...
                             'Top row: absolute dF/F cutoffs  |  Bottom row: mean + k·σ'], recIdx), ...
                    'FontWeight', 'bold');

            fprintf('Section 6 complete: Visual threshold comparison created\n');
        end
    end
end

end  % RUN.section6

%% =========================================================================
%% SECTION 7: Variance Explained by Binarization
%% =========================================================================
if RUN.section7
%
% Purpose: Quantify how much information is retained when binarizing Grid40
% raw data at different sigma thresholds (1.0σ, 1.5σ, 2.0σ).
%
% Three complementary metrics:
% 1. R² (Coefficient of Determination): Regression quality (binary predicting raw)
% 2. Variance Retention Ratio: Fraction of variance remaining after binarization
% 3. Normalized Reconstruction Error: Reconstruction quality
%
% Normalization: BIN_AN (per-animal statistics)
% Position: P1 only
% Data source: Grid40 raw values

fprintf('\n--- Section 7: Variance Explained by Binarization ---\n');

% Define thresholds to test
thresholds = [1.0, 1.5, 2.0];
nThresholds = length(thresholds);

% Define metrics
metrics = {'R2', 'VarRetention', 'NormReconError'};
metricNames = {'R² (Coefficient of Determination)', ...
               'Variance Retention Ratio', ...
               'Normalized Reconstruction Error'};

% Initialize results structure
% Results(threshold_idx).Condition.Metric.recordingMedians = [nRecs × 1]
Results_VarExplained = struct();

fprintf('Testing thresholds: %.1f, %.1f, %.1f σ\n', thresholds);
fprintf('Metrics: R², Variance Retention, Normalized Reconstruction Error\n');

% Step 1: Calculate Per-Animal Statistics for BIN_AN Normalization

fprintf('\n=== Step 1: Calculating per-animal statistics (BIN_AN) ===\n');

AnimalStats_VarExplained = struct();

for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No %s data in Grid40. Skipping.\n', conditionIndividual);
        continue;
    end

    % Get animal IDs
    if ~isfield(Rec, condition)
        fprintf('  Warning: No recording metadata for %s. Skipping.\n', condition);
        continue;
    end

    nRecs = length(Grid40.(conditionIndividual).AllNeurons);
    animalIDs = Rec.(condition).AnimalID;
    uniqueAnimalIDs = unique(animalIDs);

    fprintf('  Condition: %s (%d recordings, %d animals)\n', ...
            condition, nRecs, length(uniqueAnimalIDs));

    % Store per-animal statistics
    AnimalStats_VarExplained.(condition) = struct();

    for a = 1:length(uniqueAnimalIDs)
        currentAnimalID = uniqueAnimalIDs{a};
        validFieldName = ['Animal_' currentAnimalID];

        % Find all recordings for this animal
        animalRecordings = find(strcmp(animalIDs, currentAnimalID));

        % Collect all Grid40 P1 data for this animal
        allGridData_P1 = [];

        for rec_idx = 1:length(animalRecordings)
            r = animalRecordings(rec_idx);

            % Skip if out of bounds or in Skip list
            if r > nRecs || ismember(r, Skip.(condition))
                continue;
            end

            % Extract Grid40 P1 data
            if ~isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1') || ...
               isempty(Grid40.(conditionIndividual).AllNeurons(r).P1)
                continue;
            end

            gridData_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

            % Handle cell vs non-cell
            if iscell(gridData_cell) && ~isempty(gridData_cell)
                gridData = gridData_cell{:};  % [gridY × gridX × timepoints × trials]
            else
                gridData = gridData_cell;
            end

            % Get dimensions
            [gridY, gridX, nTimepoints, nTrials] = size(gridData);
            nGridCells = gridY * gridX;

            % Reshape: [gridY × gridX × timepoints × trials] → [gridCells × timepoints × trials]
            gridData_reshaped = reshape(gridData, [nGridCells, nTimepoints, nTrials]);

            % Flatten all data for this recording
            gridData_flat = gridData_reshaped(:);

            % Append to animal data (removing NaNs)
            validData = gridData_flat(~isnan(gridData_flat));

            % Remove low values (dead/empty cells) if flag is enabled
            if Remove_Low_Values
                validData = validData(validData >= DEAD_CELL_THRESHOLD);
            end

            allGridData_P1 = [allGridData_P1; validData(:)];
        end

        % Calculate mean and std for this animal
        if ~isempty(allGridData_P1)
            AnimalStats_VarExplained.(condition).(validFieldName).mean = mean(allGridData_P1);
            AnimalStats_VarExplained.(condition).(validFieldName).std = std(allGridData_P1);

            fprintf('    Animal %s: mean=%.4f, std=%.4f (n=%d data points)\n', ...
                    currentAnimalID, ...
                    AnimalStats_VarExplained.(condition).(validFieldName).mean, ...
                    AnimalStats_VarExplained.(condition).(validFieldName).std, ...
                    length(allGridData_P1));
        else
            fprintf('    Animal %s: No valid data found.\n', currentAnimalID);
        end
    end
end

% Step 2: Compute Variance Explained Metrics for Each Threshold

fprintf('\n=== Step 2: Computing variance explained metrics ===\n');

for t_idx = 1:nThresholds
    current_threshold = thresholds(t_idx);
    fprintf('\n--- Processing Threshold = %.1f σ ---\n', current_threshold);

    % Initialize results for this threshold
    Results_VarExplained(t_idx).threshold = current_threshold;

    for c = 1:length(conditions)
        condition = conditions{c};
        conditionIndividual = [condition 'Individual'];

        % Check if Grid40 data exists
        if ~isfield(Grid40, conditionIndividual)
            continue;
        end

        % Check if animal stats exist
        if ~isfield(AnimalStats_VarExplained, condition)
            continue;
        end

        nRecs = length(Grid40.(conditionIndividual).AllNeurons);
        animalIDs = Rec.(condition).AnimalID;

        % Initialize metric storage for this condition
        R2_values = [];
        VarRetention_values = [];
        NormReconError_values = [];

        fprintf('  Condition: %s\n', condition);

        for r = 1:nRecs
            % Skip if in Skip list
            if ismember(r, Skip.(condition))
                continue;
            end

            % Get animal ID for this recording
            if r > length(animalIDs)
                continue;
            end
            currentAnimalID = animalIDs{r};
            validFieldName = ['Animal_' currentAnimalID];

            % Check if animal stats exist
            if ~isfield(AnimalStats_VarExplained.(condition), validFieldName)
                fprintf('    Recording %d: No animal stats for %s. Skipping.\n', r, currentAnimalID);
                continue;
            end

            % Get per-animal mean and std
            animal_mean = AnimalStats_VarExplained.(condition).(validFieldName).mean;
            animal_std = AnimalStats_VarExplained.(condition).(validFieldName).std;

            % Extract Grid40 P1 data
            if ~isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1') || ...
               isempty(Grid40.(conditionIndividual).AllNeurons(r).P1)
                continue;
            end

            gridData_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

            % Handle cell vs non-cell
            if iscell(gridData_cell) && ~isempty(gridData_cell)
                gridData = gridData_cell{:};
            else
                gridData = gridData_cell;
            end

            % Get dimensions
            [gridY, gridX, nTimepoints, nTrials] = size(gridData);
            nGridCells = gridY * gridX;

            % Reshape: [gridY × gridX × timepoints × trials] → [gridCells × timepoints × trials]
            gridData_reshaped = reshape(gridData, [nGridCells, nTimepoints, nTrials]);

            % Flatten for this recording
            gridData_flat = gridData_reshaped(:);

            % Remove NaNs
            validIdx = ~isnan(gridData_flat);
            data_raw = gridData_flat(validIdx);

            % Remove low values (dead/empty cells) if flag is enabled
            if Remove_Low_Values
                data_raw = data_raw(data_raw >= DEAD_CELL_THRESHOLD);
            end

            % Skip if no valid data
            if isempty(data_raw)
                continue;
            end

            % Calculate binarization threshold for this animal
            thresh_value = animal_mean + current_threshold * animal_std;

            % Binarize data (convert to double for calculations)
            data_binary = double(data_raw > thresh_value);

            % ===== Metric 1: R² (Coefficient of Determination) =====
            % R² = 1 - SS_residual / SS_total
            % Treating binary as predictor of raw
            SS_total = sum((data_raw - mean(data_raw)).^2);
            SS_residual = sum((data_raw - data_binary).^2);

            if SS_total > 0
                R2 = 1 - (SS_residual / SS_total);
            else
                R2 = NaN;
            end

            % ===== Metric 2: Variance Retention Ratio =====
            % var(binary) / var(raw)
            var_raw = var(data_raw);
            var_binary = var(data_binary);

            if var_raw > 0
                VarRetention = var_binary / var_raw;
            else
                VarRetention = NaN;
            end

            % ===== Metric 3: Normalized Reconstruction Error =====
            % 1 - MSE(raw, binary) / var(raw)
            MSE = mean((data_raw - data_binary).^2);

            if var_raw > 0
                NormReconError = 1 - (MSE / var_raw);
            else
                NormReconError = NaN;
            end

            % Store values
            R2_values = [R2_values; R2];
            VarRetention_values = [VarRetention_values; VarRetention];
            NormReconError_values = [NormReconError_values; NormReconError];

            fprintf('    Recording %d (Animal %s): R²=%.4f, VarRet=%.4f, NormRecon=%.4f\n', ...
                    r, currentAnimalID, R2, VarRetention, NormReconError);
        end

        % Store results for this condition
        Results_VarExplained(t_idx).(condition).R2.recordingMedians = R2_values;
        Results_VarExplained(t_idx).(condition).VarRetention.recordingMedians = VarRetention_values;
        Results_VarExplained(t_idx).(condition).NormReconError.recordingMedians = NormReconError_values;

        % Calculate summary statistics
        if ~isempty(R2_values)
            Results_VarExplained(t_idx).(condition).R2.mean = mean(R2_values);
            Results_VarExplained(t_idx).(condition).R2.std = std(R2_values);
        end
        if ~isempty(VarRetention_values)
            Results_VarExplained(t_idx).(condition).VarRetention.mean = mean(VarRetention_values);
            Results_VarExplained(t_idx).(condition).VarRetention.std = std(VarRetention_values);
        end
        if ~isempty(NormReconError_values)
            Results_VarExplained(t_idx).(condition).NormReconError.mean = mean(NormReconError_values);
            Results_VarExplained(t_idx).(condition).NormReconError.std = std(NormReconError_values);
        end

        fprintf('  Condition %s summary: R²=%.4f±%.4f, VarRet=%.4f±%.4f, NormRecon=%.4f±%.4f\n', ...
                condition, ...
                Results_VarExplained(t_idx).(condition).R2.mean, ...
                Results_VarExplained(t_idx).(condition).R2.std, ...
                Results_VarExplained(t_idx).(condition).VarRetention.mean, ...
                Results_VarExplained(t_idx).(condition).VarRetention.std, ...
                Results_VarExplained(t_idx).(condition).NormReconError.mean, ...
                Results_VarExplained(t_idx).(condition).NormReconError.std);
    end
end

% Step 3: Visualization with daboxplot

fprintf('\n=== Step 3: Creating daboxplot visualizations ===\n');

% Prepare color matrix from params
color_matrix = zeros(length(conditions), 3);
for c = 1:length(conditions)
    if isfield(params, conditions{c})
        color_matrix(c, :) = params.(conditions{c});
    else
        color_matrix(c, :) = [0.5 0.5 0.5];
    end
end

% Create one figure per metric
for m = 1:length(metrics)
    metric = metrics{m};
    metricName = metricNames{m};

    fprintf('  Creating figure for %s...\n', metricName);

    figure('Name', sprintf('Variance Explained - %s', metricName), 'Color', 'w');

    % Create 3 subplots (one per threshold)
    for t_idx = 1:nThresholds
        current_threshold = thresholds(t_idx);

        subplot(1, 3, t_idx);
        hold on;

        % Collect data for all conditions
        data_matrix = [];
        group_idx = [];

        for c = 1:length(conditions)
            condition = conditions{c};

            if isfield(Results_VarExplained(t_idx), condition) && ...
               isfield(Results_VarExplained(t_idx).(condition), metric) && ...
               isfield(Results_VarExplained(t_idx).(condition).(metric), 'recordingMedians')

                recording_values = Results_VarExplained(t_idx).(condition).(metric).recordingMedians;

                if ~isempty(recording_values)
                    % Add to data matrix
                    data_matrix = [data_matrix; recording_values(:)];
                    % Assign group index
                    group_idx = [group_idx; c * ones(length(recording_values), 1)];
                end
            end
        end

        % Create daboxplot if data exists
        if ~isempty(data_matrix)
            daboxplot(data_matrix, 'groups', group_idx, 'colors', color_matrix, ...
                'xtlabels', conditions, 'scatter', 1, 'outliers', 1, ...
                'mean', 1, 'boxalpha', 0.5, 'whiskers', 0);
        end

        % Format plot
        ylabel(metricName, 'FontSize', 10);
        title(sprintf('Threshold = %.1fσ', current_threshold), 'FontSize', 11);
        grid on;
        box on;

        % Set y-axis limits based on metric
        if strcmp(metric, 'R2')
            ylim([-0.1, 1.0]);
        elseif strcmp(metric, 'VarRetention')
            ylim([0, 0.5]);
        elseif strcmp(metric, 'NormReconError')
            ylim([-0.1, 1.0]);
        end

        hold off;
    end

    % Add overall title
    sgtitle(sprintf('%s Across Conditions and Thresholds', metricName), ...
            'FontSize', 14, 'FontWeight', 'bold');
end

% Step 3b: Compute the same three metrics at ABSOLUTE 2.0 dF/F
% ---------------------------------------------------------------
% Parallel to Step 2, but binarises by `data_raw > 2.0` directly (no
% per-animal normalisation). This gives a side-by-side comparison to the
% σ-normalised 2.0σ column already computed above — it is the version the
% Ising pipeline actually uses (Figure5_dataAggregation.m:41).
if RUN.section7_abs_vs_sigma

fprintf('\n=== Step 3b: Computing variance metrics at ABSOLUTE 2.0 dF/F ===\n');

Results_VarExplained_ABS = struct();
Results_VarExplained_ABS.threshold_absolute = 2.0;

for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    if ~isfield(Grid40, conditionIndividual)
        continue;
    end
    if ~isfield(AnimalStats_VarExplained, condition)
        continue;
    end

    nRecs = length(Grid40.(conditionIndividual).AllNeurons);

    R2_values_ABS = [];
    VarRetention_values_ABS = [];
    NormReconError_values_ABS = [];

    fprintf('  Condition: %s\n', condition);

    for r = 1:nRecs
        if ismember(r, Skip.(condition))
            continue;
        end

        if ~isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1') || ...
           isempty(Grid40.(conditionIndividual).AllNeurons(r).P1)
            continue;
        end

        gridData_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;
        if iscell(gridData_cell) && ~isempty(gridData_cell)
            gridData = gridData_cell{:};
        else
            gridData = gridData_cell;
        end

        gridData_flat = gridData(:);
        validIdx = ~isnan(gridData_flat);
        data_raw = gridData_flat(validIdx);

        if Remove_Low_Values
            data_raw = data_raw(data_raw >= DEAD_CELL_THRESHOLD);
        end

        if isempty(data_raw)
            continue;
        end

        % Absolute binarisation at 2.0 dF/F (matches Ising production)
        data_binary = double(data_raw > 2.0);

        % R² (coefficient of determination)
        SS_total = sum((data_raw - mean(data_raw)).^2);
        SS_residual = sum((data_raw - data_binary).^2);
        if SS_total > 0
            R2_abs = 1 - (SS_residual / SS_total);
        else
            R2_abs = NaN;
        end

        % Variance Retention Ratio
        var_raw = var(data_raw);
        var_binary = var(data_binary);
        if var_raw > 0
            VarRet_abs = var_binary / var_raw;
        else
            VarRet_abs = NaN;
        end

        % Normalized Reconstruction Error
        MSE = mean((data_raw - data_binary).^2);
        if var_raw > 0
            NormRecon_abs = 1 - (MSE / var_raw);
        else
            NormRecon_abs = NaN;
        end

        R2_values_ABS = [R2_values_ABS; R2_abs];
        VarRetention_values_ABS = [VarRetention_values_ABS; VarRet_abs];
        NormReconError_values_ABS = [NormReconError_values_ABS; NormRecon_abs];
    end

    Results_VarExplained_ABS.(condition).R2.recordingMedians = R2_values_ABS;
    Results_VarExplained_ABS.(condition).VarRetention.recordingMedians = VarRetention_values_ABS;
    Results_VarExplained_ABS.(condition).NormReconError.recordingMedians = NormReconError_values_ABS;

    if ~isempty(R2_values_ABS)
        Results_VarExplained_ABS.(condition).R2.mean = mean(R2_values_ABS);
        Results_VarExplained_ABS.(condition).R2.std = std(R2_values_ABS);
        Results_VarExplained_ABS.(condition).VarRetention.mean = mean(VarRetention_values_ABS);
        Results_VarExplained_ABS.(condition).VarRetention.std = std(VarRetention_values_ABS);
        Results_VarExplained_ABS.(condition).NormReconError.mean = mean(NormReconError_values_ABS);
        Results_VarExplained_ABS.(condition).NormReconError.std = std(NormReconError_values_ABS);

        fprintf('    %s: R²=%.4f±%.4f, VarRet=%.4f±%.4f, NormRecon=%.4f±%.4f (ABS 2.0 dF/F)\n', ...
                condition, ...
                Results_VarExplained_ABS.(condition).R2.mean, ...
                Results_VarExplained_ABS.(condition).R2.std, ...
                Results_VarExplained_ABS.(condition).VarRetention.mean, ...
                Results_VarExplained_ABS.(condition).VarRetention.std, ...
                Results_VarExplained_ABS.(condition).NormReconError.mean, ...
                Results_VarExplained_ABS.(condition).NormReconError.std);
    end
end

% Step 3c: Side-by-side comparison plots: 2σ (relative) vs 2 dF/F (absolute)
% ---------------------------------------------------------------
% One figure per metric. Each figure has 2 panels: left = 2σ (from the
% existing t_idx=3 slot, BIN_AN normalised), right = 2 dF/F absolute.

fprintf('\n=== Step 3c: Creating 2σ vs 2 dF/F comparison figures ===\n');

% Find index of 2.0σ in thresholds
t_idx_2sigma = find(abs(thresholds - 2.0) < eps, 1);
if isempty(t_idx_2sigma)
    warning('Could not find 2.0σ in thresholds vector; skipping comparison figures');
else
    comparison_labels = {'2σ (mean + 2·σ per animal)', '2 dF/F (absolute, Ising)'};

    for m = 1:length(metrics)
        metric = metrics{m};
        metricName = metricNames{m};

        fprintf('  Creating 2σ vs 2 dF/F comparison for %s...\n', metricName);

        figure('Name', sprintf('Variance Explained - %s - 2sigma vs 2dFF', metricName), 'Color', 'w');

        % Panel 1: 2σ (reuse existing Results_VarExplained at t_idx_2sigma)
        subplot(1, 2, 1);
        hold on;
        data_matrix_sigma = [];
        group_idx_sigma = [];
        for c = 1:length(conditions)
            condition = conditions{c};
            if isfield(Results_VarExplained(t_idx_2sigma), condition) && ...
               isfield(Results_VarExplained(t_idx_2sigma).(condition), metric) && ...
               isfield(Results_VarExplained(t_idx_2sigma).(condition).(metric), 'recordingMedians')
                vals = Results_VarExplained(t_idx_2sigma).(condition).(metric).recordingMedians;
                if ~isempty(vals)
                    data_matrix_sigma = [data_matrix_sigma; vals(:)];
                    group_idx_sigma = [group_idx_sigma; c * ones(length(vals), 1)];
                end
            end
        end
        if ~isempty(data_matrix_sigma)
            daboxplot(data_matrix_sigma, 'groups', group_idx_sigma, 'colors', color_matrix, ...
                'xtlabels', conditions, 'scatter', 1, 'outliers', 1, ...
                'mean', 1, 'boxalpha', 0.5, 'whiskers', 0);
        end
        ylabel(metricName, 'FontSize', 10);
        title(comparison_labels{1}, 'FontSize', 11);
        grid on;
        box on;
        if strcmp(metric, 'R2')
            ylim([-0.1, 1.0]);
        elseif strcmp(metric, 'VarRetention')
            ylim([0, 0.5]);
        elseif strcmp(metric, 'NormReconError')
            ylim([-0.1, 1.0]);
        end
        hold off;

        % Panel 2: 2 dF/F absolute (from Results_VarExplained_ABS)
        subplot(1, 2, 2);
        hold on;
        data_matrix_abs = [];
        group_idx_abs = [];
        for c = 1:length(conditions)
            condition = conditions{c};
            if isfield(Results_VarExplained_ABS, condition) && ...
               isfield(Results_VarExplained_ABS.(condition), metric) && ...
               isfield(Results_VarExplained_ABS.(condition).(metric), 'recordingMedians')
                vals = Results_VarExplained_ABS.(condition).(metric).recordingMedians;
                if ~isempty(vals)
                    data_matrix_abs = [data_matrix_abs; vals(:)];
                    group_idx_abs = [group_idx_abs; c * ones(length(vals), 1)];
                end
            end
        end
        if ~isempty(data_matrix_abs)
            daboxplot(data_matrix_abs, 'groups', group_idx_abs, 'colors', color_matrix, ...
                'xtlabels', conditions, 'scatter', 1, 'outliers', 1, ...
                'mean', 1, 'boxalpha', 0.5, 'whiskers', 0);
        end
        ylabel(metricName, 'FontSize', 10);
        title(comparison_labels{2}, 'FontSize', 11, 'Color', [0 0.5 0], 'FontWeight', 'bold');
        grid on;
        box on;
        % Green border on the ABS panel for emphasis
        ax_abs = gca;
        ax_abs.XColor = [0 0.5 0];
        ax_abs.YColor = [0 0.5 0];
        ax_abs.LineWidth = 1.8;
        if strcmp(metric, 'R2')
            ylim([-0.1, 1.0]);
        elseif strcmp(metric, 'VarRetention')
            ylim([0, 0.5]);
        elseif strcmp(metric, 'NormReconError')
            ylim([-0.1, 1.0]);
        end
        hold off;

        sgtitle(sprintf('%s: 2σ (relative) vs 2 dF/F (absolute, Ising cutoff)', metricName), ...
                'FontSize', 14, 'FontWeight', 'bold');
    end
end

end  % RUN.section7_abs_vs_sigma

% Step 4: Console Output Summary

fprintf('\n=== Step 4: Summary Tables ===\n');

for t_idx = 1:nThresholds
    current_threshold = thresholds(t_idx);

    fprintf('\n========================================================================\n');
    fprintf('=== VARIANCE EXPLAINED AT THRESHOLD %.1fσ ===\n', current_threshold);
    fprintf('========================================================================\n');
    fprintf('%-15s | %-20s | %-20s | %-20s\n', ...
            'Condition', 'R²', 'Var Retention', 'Norm Recon Error');
    fprintf('------------------------------------------------------------------------\n');

    for c = 1:length(conditions)
        condition = conditions{c};

        % Get values
        if isfield(Results_VarExplained(t_idx), condition)
            R2_mean = Results_VarExplained(t_idx).(condition).R2.mean;
            R2_std = Results_VarExplained(t_idx).(condition).R2.std;
            VarRet_mean = Results_VarExplained(t_idx).(condition).VarRetention.mean;
            VarRet_std = Results_VarExplained(t_idx).(condition).VarRetention.std;
            NormRecon_mean = Results_VarExplained(t_idx).(condition).NormReconError.mean;
            NormRecon_std = Results_VarExplained(t_idx).(condition).NormReconError.std;

            fprintf('%-15s | %.4f ± %.4f      | %.4f ± %.4f      | %.4f ± %.4f\n', ...
                    condition, R2_mean, R2_std, VarRet_mean, VarRet_std, ...
                    NormRecon_mean, NormRecon_std);
        else
            fprintf('%-15s | No data\n', condition);
        end
    end
    fprintf('========================================================================\n');
end

fprintf('\nSection 7 complete: Variance explained analysis finished\n');

end  % RUN.section7

%% =========================================================================
%% SECTION 8: Raw vs Binarized Entropy Comparison
%% =========================================================================
if RUN.section8
%
% Purpose: Compare entropy from raw neural activity data (ActivityData)
% against entropy computed from binarized spatial grid data (Grid40).
%
% Key Questions:
% 1. How well does spatial grid binarization preserve raw activity entropy patterns?
% 2. Does BIN (recording-level) or BIN_AN (per-animal) normalization better preserve entropy?
% 3. What threshold (1.0σ, 1.5σ, 2.0σ) maintains the strongest correlation?
%
% Data sources:
% - Raw entropy: Pre-computed Entropy variable (from ActivityData)
% - Binarized entropy: Grid40 data with BIN and BIN_AN binarization
%
% Analysis:
% 1. Load raw activity entropy (Entropy variable)
% 2. Compute entropy from binarized Grid40 (BIN: recording-level + BIN_AN: per-animal)
% 3. Frame-to-frame correlation: Compare temporal dynamics between raw and binarized
% 4. Distribution comparison: Compare entropy value distributions
% 5. Visualize preservation across conditions and thresholds
%
% Position: P1 only
% Conditions: Naive, Beginner, Expert, NoSpout

fprintf('\n--- Section 8: Raw vs Binarized Entropy Comparison ---\n');

% Define thresholds — these are needed both during compute (for data
% generation) and during save (saved alongside the .mat). Always define
% them at Section 8 entry so they exist regardless of which sub-flags
% are on.
entropy_thresholds = [1.0, 1.5, 2.0];
nEntropyThresholds = length(entropy_thresholds);

fprintf('Testing entropy comparison at thresholds: %.1f, %.1f, %.1f (units vary per method)\n', entropy_thresholds);
fprintf('Normalization approaches:\n');
fprintf('  BIN    — recording-level z-score (mean + k·σ)\n');
fprintf('  BIN_AN — per-animal z-score (mean + k·σ)\n');
fprintf('  ABS    — absolute dF/F cutoff (Ising production: Figure5_dataAggregation.m:41)\n');

if ~RUN.section8_compute && ~RUN.section8_plots
    fprintf('\n(Section 8 entered but both section8_compute=0 and section8_plots=0\n');
    fprintf(' — nothing to do. Set one of them to 1 to generate or plot entropy data.)\n');
end

% Step 1: Load Raw Entropy and Compute Binarized Grid Entropy
if RUN.section8_compute

fprintf('\n=== Step 1: Computing raw and binarized entropy comparisons ===\n');

% Initialize results structure (fresh)
EntropyPreservation_Results = struct();

% Load raw activity entropy (Entropy variable)
if ~exist('Entropy', 'var')
    error('Section 8 compute requires Entropy variable (raw activity entropy). Please load it first.');
end

fprintf('Loaded Entropy variable (raw activity entropy)\n');

% Reuse per-animal statistics from Section 7 (already computed)
if ~exist('AnimalStats_VarExplained', 'var')
    error('Section 8 compute requires AnimalStats_VarExplained from Section 7 Step 1. Please run Section 7 first (or set RUN.section7 = true).');
end

fprintf('Using per-animal statistics from Section 7 (for BIN_AN normalization)\n');

% First, extract raw activity entropy from the Entropy variable
fprintf('\nExtracting raw activity entropy from Entropy variable...\n');

% Entropy structure: fields are condition names
% Each field contains [trials × timepoints] entropy values per recording

% Initialize storage for raw entropy extracted from Entropy variable
raw_entropy_traces = struct();

% Loop through conditions to extract raw entropy
for c = 1:length(conditions)
    condition = conditions{c};

    if ~isfield(Entropy, condition)
        fprintf('  Warning: No Entropy data for %s. Skipping.\n', condition);
        continue;
    end

    % Each condition has multiple recordings
    nRecsRaw = length(Entropy.(condition));
    fprintf('  %s: %d recordings with entropy data\n', condition, nRecsRaw);

    raw_entropy_traces.(condition) = Entropy.(condition);
end

fprintf('Raw entropy extraction complete\n');

% Now compute binarized Grid40 entropy
fprintf('\nComputing binarized Grid40 entropy...\n');

% Loop through conditions to compute Grid40 binarized entropy
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('\nProcessing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No %s data in Grid40. Skipping.\n', conditionIndividual);
        continue;
    end

    % Check if animal stats exist
    if ~isfield(AnimalStats_VarExplained, condition)
        fprintf('  Warning: No animal stats for %s. Skipping.\n', condition);
        continue;
    end

    nRecs = length(Grid40.(conditionIndividual).AllNeurons);
    animalIDs = Rec.(condition).AnimalID;

    % Compute global (BIN) normalization statistics for this condition
    fprintf('  Computing recording-level (BIN) statistics...\n');
    bin_stats = struct();

    for r = 1:nRecs
        % Skip if in Skip list
        if ismember(r, Skip.(condition))
            continue;
        end

        % Extract Grid40 P1 data
        if ~isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1') || ...
           isempty(Grid40.(conditionIndividual).AllNeurons(r).P1)
            continue;
        end

        gridData_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

        % Handle cell vs non-cell
        if iscell(gridData_cell) && ~isempty(gridData_cell)
            gridData = gridData_cell{:};
        else
            gridData = gridData_cell;
        end

        % Reshape for easier processing: [gridCells × timepoints × trials]
        gridData_reshaped = reshape(gridData, [], size(gridData, 3), size(gridData, 4));

        % Get valid (non-NaN, non-dead) values
        activity_vec = gridData_reshaped(:);
        validIdx = ~isnan(activity_vec);
        if Remove_Low_Values
            validIdx = validIdx & (activity_vec >= DEAD_CELL_THRESHOLD);
        end
        activity_valid = activity_vec(validIdx);

        if ~isempty(activity_valid)
            bin_stats(r).mean = mean(activity_valid);
            bin_stats(r).std = std(activity_valid);
        end
    end

    % Initialize storage for this condition (both BIN and BIN_AN)
    % ENTROPY STORAGE
    EntropyPreservation_Results.(condition).entropy_raw = {};

    % BIN normalization (recording-level)
    EntropyPreservation_Results.(condition).entropy_bin_BIN_1p0 = {};
    EntropyPreservation_Results.(condition).entropy_bin_BIN_1p5 = {};
    EntropyPreservation_Results.(condition).entropy_bin_BIN_2p0 = {};

    % BIN_AN normalization (per-animal)
    EntropyPreservation_Results.(condition).entropy_bin_BINAN_1p0 = {};
    EntropyPreservation_Results.(condition).entropy_bin_BINAN_1p5 = {};
    EntropyPreservation_Results.(condition).entropy_bin_BINAN_2p0 = {};

    % ABS normalization (absolute dF/F cutoff — matches Ising pipeline)
    EntropyPreservation_Results.(condition).entropy_bin_ABS_1p0 = {};
    EntropyPreservation_Results.(condition).entropy_bin_ABS_1p5 = {};
    EntropyPreservation_Results.(condition).entropy_bin_ABS_2p0 = {};

    % MORAN'S I STORAGE
    EntropyPreservation_Results.(condition).moransI_raw = {};

    % BIN normalization (recording-level)
    EntropyPreservation_Results.(condition).moransI_bin_BIN_1p0 = {};
    EntropyPreservation_Results.(condition).moransI_bin_BIN_1p5 = {};
    EntropyPreservation_Results.(condition).moransI_bin_BIN_2p0 = {};

    % BIN_AN normalization (per-animal)
    EntropyPreservation_Results.(condition).moransI_bin_BINAN_1p0 = {};
    EntropyPreservation_Results.(condition).moransI_bin_BINAN_1p5 = {};
    EntropyPreservation_Results.(condition).moransI_bin_BINAN_2p0 = {};

    % ABS normalization (absolute dF/F cutoff)
    EntropyPreservation_Results.(condition).moransI_bin_ABS_1p0 = {};
    EntropyPreservation_Results.(condition).moransI_bin_ABS_1p5 = {};
    EntropyPreservation_Results.(condition).moransI_bin_ABS_2p0 = {};

    % Process each recording
    for r = 1:nRecs
        % Skip if in Skip list
        if ismember(r, Skip.(condition))
            continue;
        end

        % Get animal ID for this recording
        if r > length(animalIDs)
            continue;
        end
        currentAnimalID = animalIDs{r};
        validFieldName = ['Animal_' currentAnimalID];

        % Check if animal stats exist for BIN_AN
        if ~isfield(AnimalStats_VarExplained.(condition), validFieldName)
            fprintf('  Recording %d: No per-animal stats for %s. Skipping.\n', r, currentAnimalID);
            continue;
        end

        % Get per-animal mean and std (for BIN_AN)
        animal_mean_BINAN = AnimalStats_VarExplained.(condition).(validFieldName).mean;
        animal_std_BINAN = AnimalStats_VarExplained.(condition).(validFieldName).std;

        % Get recording-level mean and std (for BIN)
        if r > length(bin_stats) || ~isfield(bin_stats(r), 'mean') || isempty(bin_stats(r).mean)
            fprintf('  Recording %d: No BIN statistics available. Skipping.\n', r);
            continue;
        end
        rec_mean_BIN = bin_stats(r).mean;
        rec_std_BIN = bin_stats(r).std;

        % Extract Grid40 P1 data
        if ~isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1') || ...
           isempty(Grid40.(conditionIndividual).AllNeurons(r).P1)
            continue;
        end

        gridData_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

        % Handle cell vs non-cell
        if iscell(gridData_cell) && ~isempty(gridData_cell)
            gridData = gridData_cell{:};
        else
            gridData = gridData_cell;
        end

        % Get dimensions
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
        nGridCells = gridY * gridX;

        % Create grid coordinates for Moran's I computation
        [gridX_coords, gridY_coords] = meshgrid(1:gridX, 1:gridY);
        grid_coords = [gridY_coords(:), gridX_coords(:)];

        % Reshape: [gridY × gridX × timepoints × trials] → [gridCells × timepoints × trials]
        gridData_reshaped = reshape(gridData, [nGridCells, nTimepoints, nTrials]);

        % Initialize entropy arrays for this recording
        ent_raw = zeros(nTrials, nTimepoints);
        ent_bin_BIN_1p0 = zeros(nTrials, nTimepoints);
        ent_bin_BIN_1p5 = zeros(nTrials, nTimepoints);
        ent_bin_BIN_2p0 = zeros(nTrials, nTimepoints);
        ent_bin_BINAN_1p0 = zeros(nTrials, nTimepoints);
        ent_bin_BINAN_1p5 = zeros(nTrials, nTimepoints);
        ent_bin_BINAN_2p0 = zeros(nTrials, nTimepoints);
        ent_bin_ABS_1p0 = zeros(nTrials, nTimepoints);
        ent_bin_ABS_1p5 = zeros(nTrials, nTimepoints);
        ent_bin_ABS_2p0 = zeros(nTrials, nTimepoints);

        % Initialize Moran's I arrays for this recording
        moransI_raw = zeros(nTrials, nTimepoints);
        moransI_bin_BIN_1p0 = zeros(nTrials, nTimepoints);
        moransI_bin_BIN_1p5 = zeros(nTrials, nTimepoints);
        moransI_bin_BIN_2p0 = zeros(nTrials, nTimepoints);
        moransI_bin_BINAN_1p0 = zeros(nTrials, nTimepoints);
        moransI_bin_BINAN_1p5 = zeros(nTrials, nTimepoints);
        moransI_bin_BINAN_2p0 = zeros(nTrials, nTimepoints);
        moransI_bin_ABS_1p0 = zeros(nTrials, nTimepoints);
        moransI_bin_ABS_1p5 = zeros(nTrials, nTimepoints);
        moransI_bin_ABS_2p0 = zeros(nTrials, nTimepoints);

        % Compute entropy for each trial and timepoint
        for trial = 1:nTrials
            for t = 1:nTimepoints
                % Extract grid cell activities at this trial/timepoint
                activity = gridData_reshaped(:, t, trial);  % [gridCells × 1]

                % Remove NaN values
                validIdx = ~isnan(activity);

                % Also remove low values (dead/empty cells) if flag is enabled
                if Remove_Low_Values
                    validIdx = validIdx & (activity >= DEAD_CELL_THRESHOLD);
                end

                % Extract valid data
                activity_valid = activity(validIdx);

                % Skip if no valid data
                if isempty(activity_valid) || length(activity_valid) < 2
                    ent_raw(trial, t) = NaN;
                    ent_bin_BIN_1p0(trial, t) = NaN;
                    ent_bin_BIN_1p5(trial, t) = NaN;
                    ent_bin_BIN_2p0(trial, t) = NaN;
                    ent_bin_BINAN_1p0(trial, t) = NaN;
                    ent_bin_BINAN_1p5(trial, t) = NaN;
                    ent_bin_BINAN_2p0(trial, t) = NaN;
                    ent_bin_ABS_1p0(trial, t) = NaN;
                    ent_bin_ABS_1p5(trial, t) = NaN;
                    ent_bin_ABS_2p0(trial, t) = NaN;
                    continue;
                end

                % Compute raw entropy (from continuous Grid40 values)
                ent_raw(trial, t) = population_entropy(activity_valid);

                % Compute raw Moran's I (spatial clustering)
                activity_z_raw = (activity - nanmean(activity)) / (nanstd(activity) + eps);
                moransI_raw(trial, t) = calculate_moransI_grid(activity_z_raw, gridY, gridX);

                % ===== BIN NORMALIZATION (RECORDING-LEVEL) =====
                if rec_std_BIN > 0
                    activity_z_BIN = (activity_valid - rec_mean_BIN) / rec_std_BIN;

                    % Binarize at 1.0σ
                    activity_bin = double(activity_z_BIN > 1.0);
                    ent_bin_BIN_1p0(trial, t) = population_entropy(activity_bin);
                    activity_bin_full = double((activity - rec_mean_BIN) / rec_std_BIN > 1.0);
                    moransI_bin_BIN_1p0(trial, t) = calculate_moransI_grid(activity_bin_full, gridY, gridX);

                    % Binarize at 1.5σ
                    activity_bin = double(activity_z_BIN > 1.5);
                    ent_bin_BIN_1p5(trial, t) = population_entropy(activity_bin);
                    activity_bin_full = double((activity - rec_mean_BIN) / rec_std_BIN > 1.5);
                    moransI_bin_BIN_1p5(trial, t) = calculate_moransI_grid(activity_bin_full, gridY, gridX);

                    % Binarize at 2.0σ
                    activity_bin = double(activity_z_BIN > 2.0);
                    ent_bin_BIN_2p0(trial, t) = population_entropy(activity_bin);
                    activity_bin_full = double((activity - rec_mean_BIN) / rec_std_BIN > 2.0);
                    moransI_bin_BIN_2p0(trial, t) = calculate_moransI_grid(activity_bin_full, gridY, gridX);
                else
                    ent_bin_BIN_1p0(trial, t) = NaN;
                    ent_bin_BIN_1p5(trial, t) = NaN;
                    ent_bin_BIN_2p0(trial, t) = NaN;
                    moransI_bin_BIN_1p0(trial, t) = NaN;
                    moransI_bin_BIN_1p5(trial, t) = NaN;
                    moransI_bin_BIN_2p0(trial, t) = NaN;
                end

                % ===== BIN_AN NORMALIZATION (PER-ANIMAL) =====
                if animal_std_BINAN > 0
                    activity_z_BINAN = (activity_valid - animal_mean_BINAN) / animal_std_BINAN;

                    % Binarize at 1.0σ
                    activity_bin = double(activity_z_BINAN > 1.0);
                    ent_bin_BINAN_1p0(trial, t) = population_entropy(activity_bin);
                    activity_bin_full = double((activity - animal_mean_BINAN) / animal_std_BINAN > 1.0);
                    moransI_bin_BINAN_1p0(trial, t) = calculate_moransI_grid(activity_bin_full, gridY, gridX);

                    % Binarize at 1.5σ
                    activity_bin = double(activity_z_BINAN > 1.5);
                    ent_bin_BINAN_1p5(trial, t) = population_entropy(activity_bin);
                    activity_bin_full = double((activity - animal_mean_BINAN) / animal_std_BINAN > 1.5);
                    moransI_bin_BINAN_1p5(trial, t) = calculate_moransI_grid(activity_bin_full, gridY, gridX);

                    % Binarize at 2.0σ
                    activity_bin = double(activity_z_BINAN > 2.0);
                    ent_bin_BINAN_2p0(trial, t) = population_entropy(activity_bin);
                    activity_bin_full = double((activity - animal_mean_BINAN) / animal_std_BINAN > 2.0);
                    moransI_bin_BINAN_2p0(trial, t) = calculate_moransI_grid(activity_bin_full, gridY, gridX);
                else
                    ent_bin_BINAN_1p0(trial, t) = NaN;
                    ent_bin_BINAN_1p5(trial, t) = NaN;
                    ent_bin_BINAN_2p0(trial, t) = NaN;
                    moransI_bin_BINAN_1p0(trial, t) = NaN;
                    moransI_bin_BINAN_1p5(trial, t) = NaN;
                    moransI_bin_BINAN_2p0(trial, t) = NaN;
                end

                % ===== ABS NORMALIZATION (ABSOLUTE dF/F CUTOFF) =====
                % Matches the Ising pipeline (Figure5_dataAggregation.m:41,
                % raw_activity_threshold = 2.0). No per-recording or per-animal
                % normalization — the cutoff is in raw dF/F units.

                % Binarize at 1.0 dF/F
                activity_bin = double(activity_valid > 1.0);
                ent_bin_ABS_1p0(trial, t) = population_entropy(activity_bin);
                activity_bin_full = double(activity > 1.0);
                moransI_bin_ABS_1p0(trial, t) = calculate_moransI_grid(activity_bin_full, gridY, gridX);

                % Binarize at 1.5 dF/F
                activity_bin = double(activity_valid > 1.5);
                ent_bin_ABS_1p5(trial, t) = population_entropy(activity_bin);
                activity_bin_full = double(activity > 1.5);
                moransI_bin_ABS_1p5(trial, t) = calculate_moransI_grid(activity_bin_full, gridY, gridX);

                % Binarize at 2.0 dF/F (Ising production cutoff)
                activity_bin = double(activity_valid > 2.0);
                ent_bin_ABS_2p0(trial, t) = population_entropy(activity_bin);
                activity_bin_full = double(activity > 2.0);
                moransI_bin_ABS_2p0(trial, t) = calculate_moransI_grid(activity_bin_full, gridY, gridX);
            end
        end

        % Store entropy traces for this recording
        EntropyPreservation_Results.(condition).entropy_raw{end+1} = ent_raw;
        EntropyPreservation_Results.(condition).entropy_bin_BIN_1p0{end+1} = ent_bin_BIN_1p0;
        EntropyPreservation_Results.(condition).entropy_bin_BIN_1p5{end+1} = ent_bin_BIN_1p5;
        EntropyPreservation_Results.(condition).entropy_bin_BIN_2p0{end+1} = ent_bin_BIN_2p0;
        EntropyPreservation_Results.(condition).entropy_bin_BINAN_1p0{end+1} = ent_bin_BINAN_1p0;
        EntropyPreservation_Results.(condition).entropy_bin_BINAN_1p5{end+1} = ent_bin_BINAN_1p5;
        EntropyPreservation_Results.(condition).entropy_bin_BINAN_2p0{end+1} = ent_bin_BINAN_2p0;
        EntropyPreservation_Results.(condition).entropy_bin_ABS_1p0{end+1} = ent_bin_ABS_1p0;
        EntropyPreservation_Results.(condition).entropy_bin_ABS_1p5{end+1} = ent_bin_ABS_1p5;
        EntropyPreservation_Results.(condition).entropy_bin_ABS_2p0{end+1} = ent_bin_ABS_2p0;

        % Store Moran's I traces for this recording
        EntropyPreservation_Results.(condition).moransI_raw{end+1} = moransI_raw;
        EntropyPreservation_Results.(condition).moransI_bin_BIN_1p0{end+1} = moransI_bin_BIN_1p0;
        EntropyPreservation_Results.(condition).moransI_bin_BIN_1p5{end+1} = moransI_bin_BIN_1p5;
        EntropyPreservation_Results.(condition).moransI_bin_BIN_2p0{end+1} = moransI_bin_BIN_2p0;
        EntropyPreservation_Results.(condition).moransI_bin_BINAN_1p0{end+1} = moransI_bin_BINAN_1p0;
        EntropyPreservation_Results.(condition).moransI_bin_BINAN_1p5{end+1} = moransI_bin_BINAN_1p5;
        EntropyPreservation_Results.(condition).moransI_bin_BINAN_2p0{end+1} = moransI_bin_BINAN_2p0;
        EntropyPreservation_Results.(condition).moransI_bin_ABS_1p0{end+1} = moransI_bin_ABS_1p0;
        EntropyPreservation_Results.(condition).moransI_bin_ABS_1p5{end+1} = moransI_bin_ABS_1p5;
        EntropyPreservation_Results.(condition).moransI_bin_ABS_2p0{end+1} = moransI_bin_ABS_2p0;

        fprintf('  Recording %d (Animal %s): Entropy and Moran''s I computed for %d trials (BIN, BIN_AN & ABS)\n', ...
                r, currentAnimalID, nTrials);
    end

    fprintf('Condition %s complete: %d recordings processed\n', ...
            condition, length(EntropyPreservation_Results.(condition).entropy_raw));
end

fprintf('\n=== Step 1 complete: Raw and binarized entropy computation finished ===\n');

% Step 2: Calculate Frame-to-Frame and Distribution Correlation Metrics

fprintf('\n=== Step 2: Calculating correlation metrics (raw vs binarized) ===\n');

% Loop through conditions
for c = 1:length(conditions)
    condition = conditions{c};

    if ~isfield(EntropyPreservation_Results, condition)
        continue;
    end

    fprintf('\nProcessing condition: %s\n', condition);

    nRecsCondition = length(EntropyPreservation_Results.(condition).entropy_raw);

    % Initialize correlation storage for this condition
    % BIN normalization correlations (recording-level)
    EntropyPreservation_Results.(condition).corr_delta_BIN_1p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_delta_BIN_1p5 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_delta_BIN_2p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_abs_BIN_1p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_abs_BIN_1p5 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_abs_BIN_2p0 = zeros(nRecsCondition, 1);

    % BIN_AN normalization correlations (per-animal)
    EntropyPreservation_Results.(condition).corr_delta_BINAN_1p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_delta_BINAN_1p5 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_delta_BINAN_2p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_abs_BINAN_1p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_abs_BINAN_1p5 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_abs_BINAN_2p0 = zeros(nRecsCondition, 1);

    % ABS normalization correlations (absolute dF/F cutoff)
    EntropyPreservation_Results.(condition).corr_delta_ABS_1p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_delta_ABS_1p5 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_delta_ABS_2p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_abs_ABS_1p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_abs_ABS_1p5 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_abs_ABS_2p0 = zeros(nRecsCondition, 1);

    % Process each recording
    for rec_idx = 1:nRecsCondition
        % Get entropy traces for this recording
        ent_raw = EntropyPreservation_Results.(condition).entropy_raw{rec_idx};

        % BIN normalization
        ent_bin_BIN_1p0 = EntropyPreservation_Results.(condition).entropy_bin_BIN_1p0{rec_idx};
        ent_bin_BIN_1p5 = EntropyPreservation_Results.(condition).entropy_bin_BIN_1p5{rec_idx};
        ent_bin_BIN_2p0 = EntropyPreservation_Results.(condition).entropy_bin_BIN_2p0{rec_idx};

        % BIN_AN normalization
        ent_bin_BINAN_1p0 = EntropyPreservation_Results.(condition).entropy_bin_BINAN_1p0{rec_idx};
        ent_bin_BINAN_1p5 = EntropyPreservation_Results.(condition).entropy_bin_BINAN_1p5{rec_idx};
        ent_bin_BINAN_2p0 = EntropyPreservation_Results.(condition).entropy_bin_BINAN_2p0{rec_idx};

        % ABS normalization
        ent_bin_ABS_1p0 = EntropyPreservation_Results.(condition).entropy_bin_ABS_1p0{rec_idx};
        ent_bin_ABS_1p5 = EntropyPreservation_Results.(condition).entropy_bin_ABS_1p5{rec_idx};
        ent_bin_ABS_2p0 = EntropyPreservation_Results.(condition).entropy_bin_ABS_2p0{rec_idx};

        % ===== FRAME-TO-FRAME DELTA CORRELATIONS =====
        % Measures how well the CHANGES in entropy are preserved

        % Calculate frame-to-frame differences (derivatives)
        deltaEnt_raw = diff(ent_raw, 1, 2);  % [trials × timepoints-1]

        % BIN normalization deltas
        deltaEnt_BIN_1p0 = diff(ent_bin_BIN_1p0, 1, 2);
        deltaEnt_BIN_1p5 = diff(ent_bin_BIN_1p5, 1, 2);
        deltaEnt_BIN_2p0 = diff(ent_bin_BIN_2p0, 1, 2);

        % BIN_AN normalization deltas
        deltaEnt_BINAN_1p0 = diff(ent_bin_BINAN_1p0, 1, 2);
        deltaEnt_BINAN_1p5 = diff(ent_bin_BINAN_1p5, 1, 2);
        deltaEnt_BINAN_2p0 = diff(ent_bin_BINAN_2p0, 1, 2);

        % ABS normalization deltas
        deltaEnt_ABS_1p0 = diff(ent_bin_ABS_1p0, 1, 2);
        deltaEnt_ABS_1p5 = diff(ent_bin_ABS_1p5, 1, 2);
        deltaEnt_ABS_2p0 = diff(ent_bin_ABS_2p0, 1, 2);

        % Flatten across trials
        deltaRaw_vec = deltaEnt_raw(:);
        deltaBIN_1p0_vec = deltaEnt_BIN_1p0(:);
        deltaBIN_1p5_vec = deltaEnt_BIN_1p5(:);
        deltaBIN_2p0_vec = deltaEnt_BIN_2p0(:);
        deltaBINAN_1p0_vec = deltaEnt_BINAN_1p0(:);
        deltaBINAN_1p5_vec = deltaEnt_BINAN_1p5(:);
        deltaBINAN_2p0_vec = deltaEnt_BINAN_2p0(:);
        deltaABS_1p0_vec = deltaEnt_ABS_1p0(:);
        deltaABS_1p5_vec = deltaEnt_ABS_1p5(:);
        deltaABS_2p0_vec = deltaEnt_ABS_2p0(:);

        % ===== BIN NORMALIZATION DELTA CORRELATIONS =====
        validIdx_BIN = ~isnan(deltaRaw_vec) & ~isnan(deltaBIN_1p0_vec) & ...
                       ~isnan(deltaBIN_1p5_vec) & ~isnan(deltaBIN_2p0_vec);

        deltaRaw_valid_BIN = deltaRaw_vec(validIdx_BIN);
        deltaBIN_1p0_valid = deltaBIN_1p0_vec(validIdx_BIN);
        deltaBIN_1p5_valid = deltaBIN_1p5_vec(validIdx_BIN);
        deltaBIN_2p0_valid = deltaBIN_2p0_vec(validIdx_BIN);

        if length(deltaRaw_valid_BIN) > 10 && std(deltaRaw_valid_BIN) > 0
            EntropyPreservation_Results.(condition).corr_delta_BIN_1p0(rec_idx) = ...
                corr(deltaRaw_valid_BIN, deltaBIN_1p0_valid);
            EntropyPreservation_Results.(condition).corr_delta_BIN_1p5(rec_idx) = ...
                corr(deltaRaw_valid_BIN, deltaBIN_1p5_valid);
            EntropyPreservation_Results.(condition).corr_delta_BIN_2p0(rec_idx) = ...
                corr(deltaRaw_valid_BIN, deltaBIN_2p0_valid);
        else
            EntropyPreservation_Results.(condition).corr_delta_BIN_1p0(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_delta_BIN_1p5(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_delta_BIN_2p0(rec_idx) = NaN;
        end

        % ===== BIN_AN NORMALIZATION DELTA CORRELATIONS =====
        validIdx_BINAN = ~isnan(deltaRaw_vec) & ~isnan(deltaBINAN_1p0_vec) & ...
                         ~isnan(deltaBINAN_1p5_vec) & ~isnan(deltaBINAN_2p0_vec);

        deltaRaw_valid_BINAN = deltaRaw_vec(validIdx_BINAN);
        deltaBINAN_1p0_valid = deltaBINAN_1p0_vec(validIdx_BINAN);
        deltaBINAN_1p5_valid = deltaBINAN_1p5_vec(validIdx_BINAN);
        deltaBINAN_2p0_valid = deltaBINAN_2p0_vec(validIdx_BINAN);

        if length(deltaRaw_valid_BINAN) > 10 && std(deltaRaw_valid_BINAN) > 0
            EntropyPreservation_Results.(condition).corr_delta_BINAN_1p0(rec_idx) = ...
                corr(deltaRaw_valid_BINAN, deltaBINAN_1p0_valid);
            EntropyPreservation_Results.(condition).corr_delta_BINAN_1p5(rec_idx) = ...
                corr(deltaRaw_valid_BINAN, deltaBINAN_1p5_valid);
            EntropyPreservation_Results.(condition).corr_delta_BINAN_2p0(rec_idx) = ...
                corr(deltaRaw_valid_BINAN, deltaBINAN_2p0_valid);
        else
            EntropyPreservation_Results.(condition).corr_delta_BINAN_1p0(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_delta_BINAN_1p5(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_delta_BINAN_2p0(rec_idx) = NaN;
        end

        % ===== ABS NORMALIZATION DELTA CORRELATIONS =====
        validIdx_ABS = ~isnan(deltaRaw_vec) & ~isnan(deltaABS_1p0_vec) & ...
                       ~isnan(deltaABS_1p5_vec) & ~isnan(deltaABS_2p0_vec);

        deltaRaw_valid_ABS = deltaRaw_vec(validIdx_ABS);
        deltaABS_1p0_valid = deltaABS_1p0_vec(validIdx_ABS);
        deltaABS_1p5_valid = deltaABS_1p5_vec(validIdx_ABS);
        deltaABS_2p0_valid = deltaABS_2p0_vec(validIdx_ABS);

        if length(deltaRaw_valid_ABS) > 10 && std(deltaRaw_valid_ABS) > 0
            % Only compute correlation if the binarised trace is not constant
            if std(deltaABS_1p0_valid) > 0
                EntropyPreservation_Results.(condition).corr_delta_ABS_1p0(rec_idx) = ...
                    corr(deltaRaw_valid_ABS, deltaABS_1p0_valid);
            else
                EntropyPreservation_Results.(condition).corr_delta_ABS_1p0(rec_idx) = NaN;
            end
            if std(deltaABS_1p5_valid) > 0
                EntropyPreservation_Results.(condition).corr_delta_ABS_1p5(rec_idx) = ...
                    corr(deltaRaw_valid_ABS, deltaABS_1p5_valid);
            else
                EntropyPreservation_Results.(condition).corr_delta_ABS_1p5(rec_idx) = NaN;
            end
            if std(deltaABS_2p0_valid) > 0
                EntropyPreservation_Results.(condition).corr_delta_ABS_2p0(rec_idx) = ...
                    corr(deltaRaw_valid_ABS, deltaABS_2p0_valid);
            else
                EntropyPreservation_Results.(condition).corr_delta_ABS_2p0(rec_idx) = NaN;
            end
        else
            EntropyPreservation_Results.(condition).corr_delta_ABS_1p0(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_delta_ABS_1p5(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_delta_ABS_2p0(rec_idx) = NaN;
        end

        % ===== ABSOLUTE VALUE CORRELATIONS =====
        % Measures how well the ABSOLUTE entropy values correlate

        % Flatten entropy traces
        entRaw_vec = ent_raw(:);
        entBIN_1p0_vec = ent_bin_BIN_1p0(:);
        entBIN_1p5_vec = ent_bin_BIN_1p5(:);
        entBIN_2p0_vec = ent_bin_BIN_2p0(:);
        entBINAN_1p0_vec = ent_bin_BINAN_1p0(:);
        entBINAN_1p5_vec = ent_bin_BINAN_1p5(:);
        entBINAN_2p0_vec = ent_bin_BINAN_2p0(:);
        entABS_1p0_vec = ent_bin_ABS_1p0(:);
        entABS_1p5_vec = ent_bin_ABS_1p5(:);
        entABS_2p0_vec = ent_bin_ABS_2p0(:);

        % ===== BIN NORMALIZATION ABSOLUTE CORRELATIONS =====
        validIdx_abs_BIN = ~isnan(entRaw_vec) & ~isnan(entBIN_1p0_vec) & ...
                           ~isnan(entBIN_1p5_vec) & ~isnan(entBIN_2p0_vec);

        entRaw_valid_BIN = entRaw_vec(validIdx_abs_BIN);
        entBIN_1p0_valid = entBIN_1p0_vec(validIdx_abs_BIN);
        entBIN_1p5_valid = entBIN_1p5_vec(validIdx_abs_BIN);
        entBIN_2p0_valid = entBIN_2p0_vec(validIdx_abs_BIN);

        if length(entRaw_valid_BIN) > 10 && std(entRaw_valid_BIN) > 0
            EntropyPreservation_Results.(condition).corr_abs_BIN_1p0(rec_idx) = ...
                corr(entRaw_valid_BIN, entBIN_1p0_valid);
            EntropyPreservation_Results.(condition).corr_abs_BIN_1p5(rec_idx) = ...
                corr(entRaw_valid_BIN, entBIN_1p5_valid);
            EntropyPreservation_Results.(condition).corr_abs_BIN_2p0(rec_idx) = ...
                corr(entRaw_valid_BIN, entBIN_2p0_valid);
        else
            EntropyPreservation_Results.(condition).corr_abs_BIN_1p0(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_abs_BIN_1p5(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_abs_BIN_2p0(rec_idx) = NaN;
        end

        % ===== BIN_AN NORMALIZATION ABSOLUTE CORRELATIONS =====
        validIdx_abs_BINAN = ~isnan(entRaw_vec) & ~isnan(entBINAN_1p0_vec) & ...
                             ~isnan(entBINAN_1p5_vec) & ~isnan(entBINAN_2p0_vec);

        entRaw_valid_BINAN = entRaw_vec(validIdx_abs_BINAN);
        entBINAN_1p0_valid = entBINAN_1p0_vec(validIdx_abs_BINAN);
        entBINAN_1p5_valid = entBINAN_1p5_vec(validIdx_abs_BINAN);
        entBINAN_2p0_valid = entBINAN_2p0_vec(validIdx_abs_BINAN);

        if length(entRaw_valid_BINAN) > 10 && std(entRaw_valid_BINAN) > 0
            EntropyPreservation_Results.(condition).corr_abs_BINAN_1p0(rec_idx) = ...
                corr(entRaw_valid_BINAN, entBINAN_1p0_valid);
            EntropyPreservation_Results.(condition).corr_abs_BINAN_1p5(rec_idx) = ...
                corr(entRaw_valid_BINAN, entBINAN_1p5_valid);
            EntropyPreservation_Results.(condition).corr_abs_BINAN_2p0(rec_idx) = ...
                corr(entRaw_valid_BINAN, entBINAN_2p0_valid);
        else
            EntropyPreservation_Results.(condition).corr_abs_BINAN_1p0(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_abs_BINAN_1p5(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_abs_BINAN_2p0(rec_idx) = NaN;
        end

        % ===== ABS NORMALIZATION ABSOLUTE CORRELATIONS =====
        validIdx_abs_ABS = ~isnan(entRaw_vec) & ~isnan(entABS_1p0_vec) & ...
                           ~isnan(entABS_1p5_vec) & ~isnan(entABS_2p0_vec);

        entRaw_valid_ABS = entRaw_vec(validIdx_abs_ABS);
        entABS_1p0_valid = entABS_1p0_vec(validIdx_abs_ABS);
        entABS_1p5_valid = entABS_1p5_vec(validIdx_abs_ABS);
        entABS_2p0_valid = entABS_2p0_vec(validIdx_abs_ABS);

        if length(entRaw_valid_ABS) > 10 && std(entRaw_valid_ABS) > 0
            if std(entABS_1p0_valid) > 0
                EntropyPreservation_Results.(condition).corr_abs_ABS_1p0(rec_idx) = ...
                    corr(entRaw_valid_ABS, entABS_1p0_valid);
            else
                EntropyPreservation_Results.(condition).corr_abs_ABS_1p0(rec_idx) = NaN;
            end
            if std(entABS_1p5_valid) > 0
                EntropyPreservation_Results.(condition).corr_abs_ABS_1p5(rec_idx) = ...
                    corr(entRaw_valid_ABS, entABS_1p5_valid);
            else
                EntropyPreservation_Results.(condition).corr_abs_ABS_1p5(rec_idx) = NaN;
            end
            if std(entABS_2p0_valid) > 0
                EntropyPreservation_Results.(condition).corr_abs_ABS_2p0(rec_idx) = ...
                    corr(entRaw_valid_ABS, entABS_2p0_valid);
            else
                EntropyPreservation_Results.(condition).corr_abs_ABS_2p0(rec_idx) = NaN;
            end
        else
            EntropyPreservation_Results.(condition).corr_abs_ABS_1p0(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_abs_ABS_1p5(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_abs_ABS_2p0(rec_idx) = NaN;
        end

        fprintf('  Recording %d: BIN delta=[%.3f, %.3f, %.3f], BIN_AN delta=[%.3f, %.3f, %.3f]\n', ...
                rec_idx, ...
                EntropyPreservation_Results.(condition).corr_delta_BIN_1p0(rec_idx), ...
                EntropyPreservation_Results.(condition).corr_delta_BIN_1p5(rec_idx), ...
                EntropyPreservation_Results.(condition).corr_delta_BIN_2p0(rec_idx), ...
                EntropyPreservation_Results.(condition).corr_delta_BINAN_1p0(rec_idx), ...
                EntropyPreservation_Results.(condition).corr_delta_BINAN_1p5(rec_idx), ...
                EntropyPreservation_Results.(condition).corr_delta_BINAN_2p0(rec_idx));
    end

    % Calculate summary statistics for this condition
    % BIN normalization
    EntropyPreservation_Results.(condition).corr_delta_BIN_1p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_delta_BIN_1p0);
    EntropyPreservation_Results.(condition).corr_delta_BIN_1p5_mean = nanmean(EntropyPreservation_Results.(condition).corr_delta_BIN_1p5);
    EntropyPreservation_Results.(condition).corr_delta_BIN_2p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_delta_BIN_2p0);
    EntropyPreservation_Results.(condition).corr_abs_BIN_1p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_abs_BIN_1p0);
    EntropyPreservation_Results.(condition).corr_abs_BIN_1p5_mean = nanmean(EntropyPreservation_Results.(condition).corr_abs_BIN_1p5);
    EntropyPreservation_Results.(condition).corr_abs_BIN_2p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_abs_BIN_2p0);

    % BIN_AN normalization
    EntropyPreservation_Results.(condition).corr_delta_BINAN_1p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_delta_BINAN_1p0);
    EntropyPreservation_Results.(condition).corr_delta_BINAN_1p5_mean = nanmean(EntropyPreservation_Results.(condition).corr_delta_BINAN_1p5);
    EntropyPreservation_Results.(condition).corr_delta_BINAN_2p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_delta_BINAN_2p0);
    EntropyPreservation_Results.(condition).corr_abs_BINAN_1p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_abs_BINAN_1p0);
    EntropyPreservation_Results.(condition).corr_abs_BINAN_1p5_mean = nanmean(EntropyPreservation_Results.(condition).corr_abs_BINAN_1p5);
    EntropyPreservation_Results.(condition).corr_abs_BINAN_2p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_abs_BINAN_2p0);

    % ABS normalization
    EntropyPreservation_Results.(condition).corr_delta_ABS_1p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_delta_ABS_1p0);
    EntropyPreservation_Results.(condition).corr_delta_ABS_1p5_mean = nanmean(EntropyPreservation_Results.(condition).corr_delta_ABS_1p5);
    EntropyPreservation_Results.(condition).corr_delta_ABS_2p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_delta_ABS_2p0);
    EntropyPreservation_Results.(condition).corr_abs_ABS_1p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_abs_ABS_1p0);
    EntropyPreservation_Results.(condition).corr_abs_ABS_1p5_mean = nanmean(EntropyPreservation_Results.(condition).corr_abs_ABS_1p5);
    EntropyPreservation_Results.(condition).corr_abs_ABS_2p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_abs_ABS_2p0);

    fprintf('Condition %s summary:\n', condition);
    fprintf('  BIN:    Delta=[%.3f, %.3f, %.3f], Abs=[%.3f, %.3f, %.3f]\n', ...
            EntropyPreservation_Results.(condition).corr_delta_BIN_1p0_mean, ...
            EntropyPreservation_Results.(condition).corr_delta_BIN_1p5_mean, ...
            EntropyPreservation_Results.(condition).corr_delta_BIN_2p0_mean, ...
            EntropyPreservation_Results.(condition).corr_abs_BIN_1p0_mean, ...
            EntropyPreservation_Results.(condition).corr_abs_BIN_1p5_mean, ...
            EntropyPreservation_Results.(condition).corr_abs_BIN_2p0_mean);
    fprintf('  BIN_AN: Delta=[%.3f, %.3f, %.3f], Abs=[%.3f, %.3f, %.3f]\n', ...
            EntropyPreservation_Results.(condition).corr_delta_BINAN_1p0_mean, ...
            EntropyPreservation_Results.(condition).corr_delta_BINAN_1p5_mean, ...
            EntropyPreservation_Results.(condition).corr_delta_BINAN_2p0_mean, ...
            EntropyPreservation_Results.(condition).corr_abs_BINAN_1p0_mean, ...
            EntropyPreservation_Results.(condition).corr_abs_BINAN_1p5_mean, ...
            EntropyPreservation_Results.(condition).corr_abs_BINAN_2p0_mean);
    fprintf('  ABS:    Delta=[%.3f, %.3f, %.3f], Abs=[%.3f, %.3f, %.3f]\n', ...
            EntropyPreservation_Results.(condition).corr_delta_ABS_1p0_mean, ...
            EntropyPreservation_Results.(condition).corr_delta_ABS_1p5_mean, ...
            EntropyPreservation_Results.(condition).corr_delta_ABS_2p0_mean, ...
            EntropyPreservation_Results.(condition).corr_abs_ABS_1p0_mean, ...
            EntropyPreservation_Results.(condition).corr_abs_ABS_1p5_mean, ...
            EntropyPreservation_Results.(condition).corr_abs_ABS_2p0_mean);
end

fprintf('\n=== Step 2 complete: Correlation metrics calculated ===\n');

% Step 2.5: Trial-Mean Correlation Analysis (Entropy and Moran's I)

fprintf('\n=== Step 2.5: Calculating trial-mean correlations ===\n');

% Calculate trial means for each recording and compute correlations
for c = 1:length(conditions)
    condition = conditions{c};

    if ~isfield(EntropyPreservation_Results, condition)
        continue;
    end

    fprintf('\nProcessing trial means for condition: %s\n', condition);

    nRecsCondition = length(EntropyPreservation_Results.(condition).entropy_raw);

    % Initialize trial-mean correlation storage for entropy
    EntropyPreservation_Results.(condition).corr_trialmean_BIN_1p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_BIN_1p5 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_BIN_2p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_BINAN_1p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_BINAN_1p5 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_BINAN_2p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p5 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_ABS_2p0 = zeros(nRecsCondition, 1);

    % Process each recording
    for rec_idx = 1:nRecsCondition
        % Get entropy traces
        ent_raw = EntropyPreservation_Results.(condition).entropy_raw{rec_idx};
        ent_bin_BIN_1p0 = EntropyPreservation_Results.(condition).entropy_bin_BIN_1p0{rec_idx};
        ent_bin_BIN_1p5 = EntropyPreservation_Results.(condition).entropy_bin_BIN_1p5{rec_idx};
        ent_bin_BIN_2p0 = EntropyPreservation_Results.(condition).entropy_bin_BIN_2p0{rec_idx};
        ent_bin_BINAN_1p0 = EntropyPreservation_Results.(condition).entropy_bin_BINAN_1p0{rec_idx};
        ent_bin_BINAN_1p5 = EntropyPreservation_Results.(condition).entropy_bin_BINAN_1p5{rec_idx};
        ent_bin_BINAN_2p0 = EntropyPreservation_Results.(condition).entropy_bin_BINAN_2p0{rec_idx};
        ent_bin_ABS_1p0 = EntropyPreservation_Results.(condition).entropy_bin_ABS_1p0{rec_idx};
        ent_bin_ABS_1p5 = EntropyPreservation_Results.(condition).entropy_bin_ABS_1p5{rec_idx};
        ent_bin_ABS_2p0 = EntropyPreservation_Results.(condition).entropy_bin_ABS_2p0{rec_idx};

        % Calculate trial means [1 × timepoints]
        trialmean_raw = nanmean(ent_raw, 1);
        trialmean_BIN_1p0 = nanmean(ent_bin_BIN_1p0, 1);
        trialmean_BIN_1p5 = nanmean(ent_bin_BIN_1p5, 1);
        trialmean_BIN_2p0 = nanmean(ent_bin_BIN_2p0, 1);
        trialmean_BINAN_1p0 = nanmean(ent_bin_BINAN_1p0, 1);
        trialmean_BINAN_1p5 = nanmean(ent_bin_BINAN_1p5, 1);
        trialmean_BINAN_2p0 = nanmean(ent_bin_BINAN_2p0, 1);
        trialmean_ABS_1p0 = nanmean(ent_bin_ABS_1p0, 1);
        trialmean_ABS_1p5 = nanmean(ent_bin_ABS_1p5, 1);
        trialmean_ABS_2p0 = nanmean(ent_bin_ABS_2p0, 1);

        % Calculate correlations between trial means
        validIdx = ~isnan(trialmean_raw) & ~isnan(trialmean_BIN_1p0) & ...
                   ~isnan(trialmean_BIN_1p5) & ~isnan(trialmean_BIN_2p0);
        if sum(validIdx) > 2
            EntropyPreservation_Results.(condition).corr_trialmean_BIN_1p0(rec_idx) = ...
                corr(trialmean_raw(validIdx)', trialmean_BIN_1p0(validIdx)');
            EntropyPreservation_Results.(condition).corr_trialmean_BIN_1p5(rec_idx) = ...
                corr(trialmean_raw(validIdx)', trialmean_BIN_1p5(validIdx)');
            EntropyPreservation_Results.(condition).corr_trialmean_BIN_2p0(rec_idx) = ...
                corr(trialmean_raw(validIdx)', trialmean_BIN_2p0(validIdx)');
        else
            EntropyPreservation_Results.(condition).corr_trialmean_BIN_1p0(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_trialmean_BIN_1p5(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_trialmean_BIN_2p0(rec_idx) = NaN;
        end

        validIdx_BINAN = ~isnan(trialmean_raw) & ~isnan(trialmean_BINAN_1p0) & ...
                         ~isnan(trialmean_BINAN_1p5) & ~isnan(trialmean_BINAN_2p0);
        if sum(validIdx_BINAN) > 2
            EntropyPreservation_Results.(condition).corr_trialmean_BINAN_1p0(rec_idx) = ...
                corr(trialmean_raw(validIdx_BINAN)', trialmean_BINAN_1p0(validIdx_BINAN)');
            EntropyPreservation_Results.(condition).corr_trialmean_BINAN_1p5(rec_idx) = ...
                corr(trialmean_raw(validIdx_BINAN)', trialmean_BINAN_1p5(validIdx_BINAN)');
            EntropyPreservation_Results.(condition).corr_trialmean_BINAN_2p0(rec_idx) = ...
                corr(trialmean_raw(validIdx_BINAN)', trialmean_BINAN_2p0(validIdx_BINAN)');
        else
            EntropyPreservation_Results.(condition).corr_trialmean_BINAN_1p0(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_trialmean_BINAN_1p5(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_trialmean_BINAN_2p0(rec_idx) = NaN;
        end

        validIdx_ABS = ~isnan(trialmean_raw) & ~isnan(trialmean_ABS_1p0) & ...
                       ~isnan(trialmean_ABS_1p5) & ~isnan(trialmean_ABS_2p0);
        if sum(validIdx_ABS) > 2
            tm_raw_v = trialmean_raw(validIdx_ABS)';
            tm_abs1p0_v = trialmean_ABS_1p0(validIdx_ABS)';
            tm_abs1p5_v = trialmean_ABS_1p5(validIdx_ABS)';
            tm_abs2p0_v = trialmean_ABS_2p0(validIdx_ABS)';
            if std(tm_raw_v) > 0 && std(tm_abs1p0_v) > 0
                EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p0(rec_idx) = corr(tm_raw_v, tm_abs1p0_v);
            else
                EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p0(rec_idx) = NaN;
            end
            if std(tm_raw_v) > 0 && std(tm_abs1p5_v) > 0
                EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p5(rec_idx) = corr(tm_raw_v, tm_abs1p5_v);
            else
                EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p5(rec_idx) = NaN;
            end
            if std(tm_raw_v) > 0 && std(tm_abs2p0_v) > 0
                EntropyPreservation_Results.(condition).corr_trialmean_ABS_2p0(rec_idx) = corr(tm_raw_v, tm_abs2p0_v);
            else
                EntropyPreservation_Results.(condition).corr_trialmean_ABS_2p0(rec_idx) = NaN;
            end
        else
            EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p0(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p5(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_trialmean_ABS_2p0(rec_idx) = NaN;
        end
    end

    % Calculate means for each threshold
    EntropyPreservation_Results.(condition).corr_trialmean_BIN_1p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_BIN_1p0);
    EntropyPreservation_Results.(condition).corr_trialmean_BIN_1p5_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_BIN_1p5);
    EntropyPreservation_Results.(condition).corr_trialmean_BIN_2p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_BIN_2p0);
    EntropyPreservation_Results.(condition).corr_trialmean_BINAN_1p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_BINAN_1p0);
    EntropyPreservation_Results.(condition).corr_trialmean_BINAN_1p5_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_BINAN_1p5);
    EntropyPreservation_Results.(condition).corr_trialmean_BINAN_2p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_BINAN_2p0);
    EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p0);
    EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p5_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p5);
    EntropyPreservation_Results.(condition).corr_trialmean_ABS_2p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_ABS_2p0);

    % ===== Moran's I Trial-Mean Correlations =====
    fprintf('\nProcessing trial means for Moran''s I in condition: %s\n', condition);

    % Initialize trial-mean correlation storage for Moran's I
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_1p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_1p5 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_2p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_1p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_1p5 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_2p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p0 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p5 = zeros(nRecsCondition, 1);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_2p0 = zeros(nRecsCondition, 1);

    % Process each recording for Moran's I trial means
    for rec_idx = 1:nRecsCondition
        % Get Moran's I traces
        moransI_raw = EntropyPreservation_Results.(condition).moransI_raw{rec_idx};
        moransI_bin_BIN_1p0 = EntropyPreservation_Results.(condition).moransI_bin_BIN_1p0{rec_idx};
        moransI_bin_BIN_1p5 = EntropyPreservation_Results.(condition).moransI_bin_BIN_1p5{rec_idx};
        moransI_bin_BIN_2p0 = EntropyPreservation_Results.(condition).moransI_bin_BIN_2p0{rec_idx};
        moransI_bin_BINAN_1p0 = EntropyPreservation_Results.(condition).moransI_bin_BINAN_1p0{rec_idx};
        moransI_bin_BINAN_1p5 = EntropyPreservation_Results.(condition).moransI_bin_BINAN_1p5{rec_idx};
        moransI_bin_BINAN_2p0 = EntropyPreservation_Results.(condition).moransI_bin_BINAN_2p0{rec_idx};
        moransI_bin_ABS_1p0 = EntropyPreservation_Results.(condition).moransI_bin_ABS_1p0{rec_idx};
        moransI_bin_ABS_1p5 = EntropyPreservation_Results.(condition).moransI_bin_ABS_1p5{rec_idx};
        moransI_bin_ABS_2p0 = EntropyPreservation_Results.(condition).moransI_bin_ABS_2p0{rec_idx};

        % Calculate trial means [1 × timepoints]
        trialmean_moransI_raw = nanmean(moransI_raw, 1);
        trialmean_moransI_BIN_1p0 = nanmean(moransI_bin_BIN_1p0, 1);
        trialmean_moransI_BIN_1p5 = nanmean(moransI_bin_BIN_1p5, 1);
        trialmean_moransI_BIN_2p0 = nanmean(moransI_bin_BIN_2p0, 1);
        trialmean_moransI_BINAN_1p0 = nanmean(moransI_bin_BINAN_1p0, 1);
        trialmean_moransI_BINAN_1p5 = nanmean(moransI_bin_BINAN_1p5, 1);
        trialmean_moransI_BINAN_2p0 = nanmean(moransI_bin_BINAN_2p0, 1);
        trialmean_moransI_ABS_1p0 = nanmean(moransI_bin_ABS_1p0, 1);
        trialmean_moransI_ABS_1p5 = nanmean(moransI_bin_ABS_1p5, 1);
        trialmean_moransI_ABS_2p0 = nanmean(moransI_bin_ABS_2p0, 1);

        % Calculate correlations between trial means
        validIdx = ~isnan(trialmean_moransI_raw) & ~isnan(trialmean_moransI_BIN_1p0) & ...
                   ~isnan(trialmean_moransI_BIN_1p5) & ~isnan(trialmean_moransI_BIN_2p0);
        if sum(validIdx) > 2
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_1p0(rec_idx) = ...
                corr(trialmean_moransI_raw(validIdx)', trialmean_moransI_BIN_1p0(validIdx)');
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_1p5(rec_idx) = ...
                corr(trialmean_moransI_raw(validIdx)', trialmean_moransI_BIN_1p5(validIdx)');
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_2p0(rec_idx) = ...
                corr(trialmean_moransI_raw(validIdx)', trialmean_moransI_BIN_2p0(validIdx)');
        else
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_1p0(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_1p5(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_2p0(rec_idx) = NaN;
        end

        validIdx_BINAN = ~isnan(trialmean_moransI_raw) & ~isnan(trialmean_moransI_BINAN_1p0) & ...
                         ~isnan(trialmean_moransI_BINAN_1p5) & ~isnan(trialmean_moransI_BINAN_2p0);
        if sum(validIdx_BINAN) > 2
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_1p0(rec_idx) = ...
                corr(trialmean_moransI_raw(validIdx_BINAN)', trialmean_moransI_BINAN_1p0(validIdx_BINAN)');
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_1p5(rec_idx) = ...
                corr(trialmean_moransI_raw(validIdx_BINAN)', trialmean_moransI_BINAN_1p5(validIdx_BINAN)');
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_2p0(rec_idx) = ...
                corr(trialmean_moransI_raw(validIdx_BINAN)', trialmean_moransI_BINAN_2p0(validIdx_BINAN)');
        else
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_1p0(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_1p5(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_2p0(rec_idx) = NaN;
        end

        validIdx_ABS = ~isnan(trialmean_moransI_raw) & ~isnan(trialmean_moransI_ABS_1p0) & ...
                       ~isnan(trialmean_moransI_ABS_1p5) & ~isnan(trialmean_moransI_ABS_2p0);
        if sum(validIdx_ABS) > 2
            tm_mi_raw = trialmean_moransI_raw(validIdx_ABS)';
            tm_mi_abs1p0 = trialmean_moransI_ABS_1p0(validIdx_ABS)';
            tm_mi_abs1p5 = trialmean_moransI_ABS_1p5(validIdx_ABS)';
            tm_mi_abs2p0 = trialmean_moransI_ABS_2p0(validIdx_ABS)';
            if std(tm_mi_raw) > 0 && std(tm_mi_abs1p0) > 0
                EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p0(rec_idx) = corr(tm_mi_raw, tm_mi_abs1p0);
            else
                EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p0(rec_idx) = NaN;
            end
            if std(tm_mi_raw) > 0 && std(tm_mi_abs1p5) > 0
                EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p5(rec_idx) = corr(tm_mi_raw, tm_mi_abs1p5);
            else
                EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p5(rec_idx) = NaN;
            end
            if std(tm_mi_raw) > 0 && std(tm_mi_abs2p0) > 0
                EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_2p0(rec_idx) = corr(tm_mi_raw, tm_mi_abs2p0);
            else
                EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_2p0(rec_idx) = NaN;
            end
        else
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p0(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p5(rec_idx) = NaN;
            EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_2p0(rec_idx) = NaN;
        end
    end

    % Calculate means for each threshold
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_1p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_1p0);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_1p5_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_1p5);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_2p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_2p0);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_1p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_1p0);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_1p5_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_1p5);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_2p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_2p0);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p0);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p5_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p5);
    EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_2p0_mean = nanmean(EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_2p0);

    fprintf('Condition %s: Trial-mean entropy and Moran''s I correlations calculated\n', condition);
end

fprintf('\n=== Step 2.5 complete: Trial-mean correlations calculated ===\n');

% Step 3: Distribution Comparison Analysis

fprintf('\n=== Step 3: Analyzing entropy distributions (raw vs binarized) ===\n');

% Collect entropy distributions across all recordings for each condition
for c = 1:length(conditions)
    condition = conditions{c};

    if ~isfield(EntropyPreservation_Results, condition)
        continue;
    end

    fprintf('\nAnalyzing distributions for condition: %s\n', condition);

    nRecsCondition = length(EntropyPreservation_Results.(condition).entropy_raw);

    % Initialize storage for distribution statistics
    EntropyPreservation_Results.(condition).dist_raw = [];
    EntropyPreservation_Results.(condition).dist_BIN_1p0 = [];
    EntropyPreservation_Results.(condition).dist_BIN_1p5 = [];
    EntropyPreservation_Results.(condition).dist_BIN_2p0 = [];
    EntropyPreservation_Results.(condition).dist_BINAN_1p0 = [];
    EntropyPreservation_Results.(condition).dist_BINAN_1p5 = [];
    EntropyPreservation_Results.(condition).dist_BINAN_2p0 = [];
    EntropyPreservation_Results.(condition).dist_ABS_1p0 = [];
    EntropyPreservation_Results.(condition).dist_ABS_1p5 = [];
    EntropyPreservation_Results.(condition).dist_ABS_2p0 = [];

    % Collect all entropy values across recordings
    for rec_idx = 1:nRecsCondition
        % Get entropy traces
        ent_raw = EntropyPreservation_Results.(condition).entropy_raw{rec_idx};
        ent_bin_BIN_1p0 = EntropyPreservation_Results.(condition).entropy_bin_BIN_1p0{rec_idx};
        ent_bin_BIN_1p5 = EntropyPreservation_Results.(condition).entropy_bin_BIN_1p5{rec_idx};
        ent_bin_BIN_2p0 = EntropyPreservation_Results.(condition).entropy_bin_BIN_2p0{rec_idx};
        ent_bin_BINAN_1p0 = EntropyPreservation_Results.(condition).entropy_bin_BINAN_1p0{rec_idx};
        ent_bin_BINAN_1p5 = EntropyPreservation_Results.(condition).entropy_bin_BINAN_1p5{rec_idx};
        ent_bin_BINAN_2p0 = EntropyPreservation_Results.(condition).entropy_bin_BINAN_2p0{rec_idx};
        ent_bin_ABS_1p0 = EntropyPreservation_Results.(condition).entropy_bin_ABS_1p0{rec_idx};
        ent_bin_ABS_1p5 = EntropyPreservation_Results.(condition).entropy_bin_ABS_1p5{rec_idx};
        ent_bin_ABS_2p0 = EntropyPreservation_Results.(condition).entropy_bin_ABS_2p0{rec_idx};

        % Flatten and remove NaN values
        raw_vals = ent_raw(:);
        raw_vals(isnan(raw_vals)) = [];

        bin_1p0_vals = ent_bin_BIN_1p0(:);
        bin_1p0_vals(isnan(bin_1p0_vals)) = [];
        bin_1p5_vals = ent_bin_BIN_1p5(:);
        bin_1p5_vals(isnan(bin_1p5_vals)) = [];
        bin_2p0_vals = ent_bin_BIN_2p0(:);
        bin_2p0_vals(isnan(bin_2p0_vals)) = [];

        binan_1p0_vals = ent_bin_BINAN_1p0(:);
        binan_1p0_vals(isnan(binan_1p0_vals)) = [];
        binan_1p5_vals = ent_bin_BINAN_1p5(:);
        binan_1p5_vals(isnan(binan_1p5_vals)) = [];
        binan_2p0_vals = ent_bin_BINAN_2p0(:);
        binan_2p0_vals(isnan(binan_2p0_vals)) = [];

        abs_1p0_vals = ent_bin_ABS_1p0(:);
        abs_1p0_vals(isnan(abs_1p0_vals)) = [];
        abs_1p5_vals = ent_bin_ABS_1p5(:);
        abs_1p5_vals(isnan(abs_1p5_vals)) = [];
        abs_2p0_vals = ent_bin_ABS_2p0(:);
        abs_2p0_vals(isnan(abs_2p0_vals)) = [];

        % Append to condition-level distributions
        EntropyPreservation_Results.(condition).dist_raw = ...
            [EntropyPreservation_Results.(condition).dist_raw; raw_vals];
        EntropyPreservation_Results.(condition).dist_BIN_1p0 = ...
            [EntropyPreservation_Results.(condition).dist_BIN_1p0; bin_1p0_vals];
        EntropyPreservation_Results.(condition).dist_BIN_1p5 = ...
            [EntropyPreservation_Results.(condition).dist_BIN_1p5; bin_1p5_vals];
        EntropyPreservation_Results.(condition).dist_BIN_2p0 = ...
            [EntropyPreservation_Results.(condition).dist_BIN_2p0; bin_2p0_vals];
        EntropyPreservation_Results.(condition).dist_BINAN_1p0 = ...
            [EntropyPreservation_Results.(condition).dist_BINAN_1p0; binan_1p0_vals];
        EntropyPreservation_Results.(condition).dist_BINAN_1p5 = ...
            [EntropyPreservation_Results.(condition).dist_BINAN_1p5; binan_1p5_vals];
        EntropyPreservation_Results.(condition).dist_BINAN_2p0 = ...
            [EntropyPreservation_Results.(condition).dist_BINAN_2p0; binan_2p0_vals];
        EntropyPreservation_Results.(condition).dist_ABS_1p0 = ...
            [EntropyPreservation_Results.(condition).dist_ABS_1p0; abs_1p0_vals];
        EntropyPreservation_Results.(condition).dist_ABS_1p5 = ...
            [EntropyPreservation_Results.(condition).dist_ABS_1p5; abs_1p5_vals];
        EntropyPreservation_Results.(condition).dist_ABS_2p0 = ...
            [EntropyPreservation_Results.(condition).dist_ABS_2p0; abs_2p0_vals];
    end

    % Compute distribution statistics
    % Raw entropy
    EntropyPreservation_Results.(condition).dist_raw_mean = nanmean(EntropyPreservation_Results.(condition).dist_raw);
    EntropyPreservation_Results.(condition).dist_raw_std = nanstd(EntropyPreservation_Results.(condition).dist_raw);
    EntropyPreservation_Results.(condition).dist_raw_median = nanmedian(EntropyPreservation_Results.(condition).dist_raw);
    EntropyPreservation_Results.(condition).dist_raw_q25 = quantile(EntropyPreservation_Results.(condition).dist_raw, 0.25);
    EntropyPreservation_Results.(condition).dist_raw_q75 = quantile(EntropyPreservation_Results.(condition).dist_raw, 0.75);

    % BIN 1.0σ
    EntropyPreservation_Results.(condition).dist_BIN_1p0_mean = nanmean(EntropyPreservation_Results.(condition).dist_BIN_1p0);
    EntropyPreservation_Results.(condition).dist_BIN_1p0_std = nanstd(EntropyPreservation_Results.(condition).dist_BIN_1p0);
    EntropyPreservation_Results.(condition).dist_BIN_1p0_median = nanmedian(EntropyPreservation_Results.(condition).dist_BIN_1p0);

    % BIN 1.5σ
    EntropyPreservation_Results.(condition).dist_BIN_1p5_mean = nanmean(EntropyPreservation_Results.(condition).dist_BIN_1p5);
    EntropyPreservation_Results.(condition).dist_BIN_1p5_std = nanstd(EntropyPreservation_Results.(condition).dist_BIN_1p5);
    EntropyPreservation_Results.(condition).dist_BIN_1p5_median = nanmedian(EntropyPreservation_Results.(condition).dist_BIN_1p5);

    % BIN 2.0σ
    EntropyPreservation_Results.(condition).dist_BIN_2p0_mean = nanmean(EntropyPreservation_Results.(condition).dist_BIN_2p0);
    EntropyPreservation_Results.(condition).dist_BIN_2p0_std = nanstd(EntropyPreservation_Results.(condition).dist_BIN_2p0);
    EntropyPreservation_Results.(condition).dist_BIN_2p0_median = nanmedian(EntropyPreservation_Results.(condition).dist_BIN_2p0);

    % BIN_AN 1.0σ
    EntropyPreservation_Results.(condition).dist_BINAN_1p0_mean = nanmean(EntropyPreservation_Results.(condition).dist_BINAN_1p0);
    EntropyPreservation_Results.(condition).dist_BINAN_1p0_std = nanstd(EntropyPreservation_Results.(condition).dist_BINAN_1p0);
    EntropyPreservation_Results.(condition).dist_BINAN_1p0_median = nanmedian(EntropyPreservation_Results.(condition).dist_BINAN_1p0);

    % BIN_AN 1.5σ
    EntropyPreservation_Results.(condition).dist_BINAN_1p5_mean = nanmean(EntropyPreservation_Results.(condition).dist_BINAN_1p5);
    EntropyPreservation_Results.(condition).dist_BINAN_1p5_std = nanstd(EntropyPreservation_Results.(condition).dist_BINAN_1p5);
    EntropyPreservation_Results.(condition).dist_BINAN_1p5_median = nanmedian(EntropyPreservation_Results.(condition).dist_BINAN_1p5);

    % BIN_AN 2.0σ
    EntropyPreservation_Results.(condition).dist_BINAN_2p0_mean = nanmean(EntropyPreservation_Results.(condition).dist_BINAN_2p0);
    EntropyPreservation_Results.(condition).dist_BINAN_2p0_std = nanstd(EntropyPreservation_Results.(condition).dist_BINAN_2p0);
    EntropyPreservation_Results.(condition).dist_BINAN_2p0_median = nanmedian(EntropyPreservation_Results.(condition).dist_BINAN_2p0);

    % ABS 1.0 dF/F
    EntropyPreservation_Results.(condition).dist_ABS_1p0_mean = nanmean(EntropyPreservation_Results.(condition).dist_ABS_1p0);
    EntropyPreservation_Results.(condition).dist_ABS_1p0_std = nanstd(EntropyPreservation_Results.(condition).dist_ABS_1p0);
    EntropyPreservation_Results.(condition).dist_ABS_1p0_median = nanmedian(EntropyPreservation_Results.(condition).dist_ABS_1p0);

    % ABS 1.5 dF/F
    EntropyPreservation_Results.(condition).dist_ABS_1p5_mean = nanmean(EntropyPreservation_Results.(condition).dist_ABS_1p5);
    EntropyPreservation_Results.(condition).dist_ABS_1p5_std = nanstd(EntropyPreservation_Results.(condition).dist_ABS_1p5);
    EntropyPreservation_Results.(condition).dist_ABS_1p5_median = nanmedian(EntropyPreservation_Results.(condition).dist_ABS_1p5);

    % ABS 2.0 dF/F (Ising production cutoff)
    EntropyPreservation_Results.(condition).dist_ABS_2p0_mean = nanmean(EntropyPreservation_Results.(condition).dist_ABS_2p0);
    EntropyPreservation_Results.(condition).dist_ABS_2p0_std = nanstd(EntropyPreservation_Results.(condition).dist_ABS_2p0);
    EntropyPreservation_Results.(condition).dist_ABS_2p0_median = nanmedian(EntropyPreservation_Results.(condition).dist_ABS_2p0);

    fprintf('  Raw:        Mean=%.4f, Std=%.4f, Median=%.4f\n', ...
            EntropyPreservation_Results.(condition).dist_raw_mean, ...
            EntropyPreservation_Results.(condition).dist_raw_std, ...
            EntropyPreservation_Results.(condition).dist_raw_median);
    fprintf('  BIN 1.0σ:   Mean=%.4f, Std=%.4f, Median=%.4f\n', ...
            EntropyPreservation_Results.(condition).dist_BIN_1p0_mean, ...
            EntropyPreservation_Results.(condition).dist_BIN_1p0_std, ...
            EntropyPreservation_Results.(condition).dist_BIN_1p0_median);
    fprintf('  BIN_AN 1.5σ: Mean=%.4f, Std=%.4f, Median=%.4f\n', ...
            EntropyPreservation_Results.(condition).dist_BINAN_1p5_mean, ...
            EntropyPreservation_Results.(condition).dist_BINAN_1p5_std, ...
            EntropyPreservation_Results.(condition).dist_BINAN_1p5_median);
    fprintf('  ABS 2.0:    Mean=%.4f, Std=%.4f, Median=%.4f\n', ...
            EntropyPreservation_Results.(condition).dist_ABS_2p0_mean, ...
            EntropyPreservation_Results.(condition).dist_ABS_2p0_std, ...
            EntropyPreservation_Results.(condition).dist_ABS_2p0_median);
end

fprintf('\n=== Step 3 complete: Distribution analysis finished ===\n');

end  % RUN.section8_compute

% Step 4: Visualization
if RUN.section8_plots

% Prerequisite check: EntropyPreservation_Results must be in the workspace
% (either from Section 8 compute in this session, or loaded from a prior
% .mat file before running the script).
if ~exist('EntropyPreservation_Results', 'var') || ...
   ~isstruct(EntropyPreservation_Results) || ...
   isempty(fieldnames(EntropyPreservation_Results))
    error(['Section 8 plots require EntropyPreservation_Results in the workspace.\n' ...
           'Either set RUN.section8_compute = true to regenerate data, or load a\n' ...
           'cached version first:\n' ...
           '    load(''EntropyPreservation_Results.mat'')']);
end

fprintf('\n=== Step 4: Creating visualizations ===\n');
fprintf('Using EntropyPreservation_Results from workspace (%d conditions)\n', ...
        length(fieldnames(EntropyPreservation_Results)));

% Get custom blue-red colormap for correlation heatmaps
blue_red_cmap = get_blue_red_cmap();

% Plot 8.1: Example Single Trial Entropy Traces (Raw vs BIN vs BIN_AN for all 3 thresholds)
fprintf('Creating Plot 8.1: Example single trial entropy traces (all 3 thresholds)\n');

% Select exemplar recordings (Expert condition, trial 1)
exemplar_condition = 'Expert';
exemplar_recs = [1, 5];  % Recording 1 and Recording 5
exemplar_trial = 1;

for rec_loop_idx = 1:length(exemplar_recs)
    exemplar_rec_idx = exemplar_recs(rec_loop_idx);

    fprintf('  Creating entropy figure for Recording %d\n', exemplar_rec_idx);

if isfield(EntropyPreservation_Results, exemplar_condition) && ...
   length(EntropyPreservation_Results.(exemplar_condition).entropy_raw) >= exemplar_rec_idx

    % Get raw entropy trace
    ent_raw_ex = EntropyPreservation_Results.(exemplar_condition).entropy_raw{exemplar_rec_idx};

    % Get binarized entropy traces for all 3 thresholds (BIN normalization)
    ent_bin_1p0_ex = EntropyPreservation_Results.(exemplar_condition).entropy_bin_BIN_1p0{exemplar_rec_idx};
    ent_bin_1p5_ex = EntropyPreservation_Results.(exemplar_condition).entropy_bin_BIN_1p5{exemplar_rec_idx};
    ent_bin_2p0_ex = EntropyPreservation_Results.(exemplar_condition).entropy_bin_BIN_2p0{exemplar_rec_idx};

    % Get binarized entropy traces for all 3 thresholds (BIN_AN normalization)
    ent_binan_1p0_ex = EntropyPreservation_Results.(exemplar_condition).entropy_bin_BINAN_1p0{exemplar_rec_idx};
    ent_binan_1p5_ex = EntropyPreservation_Results.(exemplar_condition).entropy_bin_BINAN_1p5{exemplar_rec_idx};
    ent_binan_2p0_ex = EntropyPreservation_Results.(exemplar_condition).entropy_bin_BINAN_2p0{exemplar_rec_idx};

    % Get binarized entropy traces for absolute dF/F cutoffs (ABS — Ising production)
    ent_abs_1p0_ex = EntropyPreservation_Results.(exemplar_condition).entropy_bin_ABS_1p0{exemplar_rec_idx};
    ent_abs_1p5_ex = EntropyPreservation_Results.(exemplar_condition).entropy_bin_ABS_1p5{exemplar_rec_idx};
    ent_abs_2p0_ex = EntropyPreservation_Results.(exemplar_condition).entropy_bin_ABS_2p0{exemplar_rec_idx};

    % Extract trial 1 for all thresholds
    if size(ent_raw_ex, 1) >= exemplar_trial
        trace_raw = ent_raw_ex(exemplar_trial, :);

        % BIN normalization traces
        trace_bin_1p0 = ent_bin_1p0_ex(exemplar_trial, :);
        trace_bin_1p5 = ent_bin_1p5_ex(exemplar_trial, :);
        trace_bin_2p0 = ent_bin_2p0_ex(exemplar_trial, :);

        % BIN_AN normalization traces
        trace_binan_1p0 = ent_binan_1p0_ex(exemplar_trial, :);
        trace_binan_1p5 = ent_binan_1p5_ex(exemplar_trial, :);
        trace_binan_2p0 = ent_binan_2p0_ex(exemplar_trial, :);

        % ABS (absolute dF/F) traces
        trace_abs_1p0 = ent_abs_1p0_ex(exemplar_trial, :);
        trace_abs_1p5 = ent_abs_1p5_ex(exemplar_trial, :);
        trace_abs_2p0 = ent_abs_2p0_ex(exemplar_trial, :);

        % Timepoints for x-axis
        timepoints = 1:length(trace_raw);

        % Find timepoint of minimum raw entropy
        [~, min_ent_t] = min(trace_raw);

        % Create 2x3 figure (2 rows × 3 thresholds: traces on top, spatial below)
        figure('Name', sprintf('Entropy Trace Comparison - Rec %d', exemplar_rec_idx));
        tiledlayout(2, 3);

        % Define threshold details
        thresholds_all = [1.0, 1.5, 2.0];
        traces_bin = {trace_bin_1p0, trace_bin_1p5, trace_bin_2p0};
        traces_binan = {trace_binan_1p0, trace_binan_1p5, trace_binan_2p0};
        traces_abs = {trace_abs_1p0, trace_abs_1p5, trace_abs_2p0};
        corr_delta_BIN = [EntropyPreservation_Results.(exemplar_condition).corr_delta_BIN_1p0(exemplar_rec_idx), ...
                          EntropyPreservation_Results.(exemplar_condition).corr_delta_BIN_1p5(exemplar_rec_idx), ...
                          EntropyPreservation_Results.(exemplar_condition).corr_delta_BIN_2p0(exemplar_rec_idx)];
        corr_delta_BINAN = [EntropyPreservation_Results.(exemplar_condition).corr_delta_BINAN_1p0(exemplar_rec_idx), ...
                            EntropyPreservation_Results.(exemplar_condition).corr_delta_BINAN_1p5(exemplar_rec_idx), ...
                            EntropyPreservation_Results.(exemplar_condition).corr_delta_BINAN_2p0(exemplar_rec_idx)];
        corr_delta_ABS = [EntropyPreservation_Results.(exemplar_condition).corr_delta_ABS_1p0(exemplar_rec_idx), ...
                          EntropyPreservation_Results.(exemplar_condition).corr_delta_ABS_1p5(exemplar_rec_idx), ...
                          EntropyPreservation_Results.(exemplar_condition).corr_delta_ABS_2p0(exemplar_rec_idx)];

        % Get Grid40 data for spatial examples
        gridData = [];
        dataFlat = [];
        grid_coords = [];
        meanVal = NaN;
        stdVal = NaN;

        if isfield(Grid40, [exemplar_condition 'Individual']) && ...
           length(Grid40.([exemplar_condition 'Individual']).AllNeurons) >= exemplar_rec_idx && ...
           isfield(Grid40.([exemplar_condition 'Individual']).AllNeurons(exemplar_rec_idx), 'P1')

            gridData_cell = Grid40.([exemplar_condition 'Individual']).AllNeurons(exemplar_rec_idx).P1;

            if iscell(gridData_cell) && ~isempty(gridData_cell)
                gridData = gridData_cell{:};
            else
                gridData = gridData_cell;
            end

            % Get dimensions and grid coordinates
            [gridY, gridX, nTimepoints, nTrials] = size(gridData);
            nGridCells = gridY * gridX;
            [gridX_coords, gridY_coords] = meshgrid(1:gridX, 1:gridY);
            grid_coords = [gridY_coords(:), gridX_coords(:)];

            % Get data for exemplar trial
            if nTrials >= exemplar_trial
                dataTrial1 = gridData(:, :, :, exemplar_trial);
                dataFlat = reshape(dataTrial1, [nGridCells, nTimepoints]);

                % Calculate statistics for binarization
                validData = dataFlat(~isnan(dataFlat));
                if Remove_Low_Values
                    validData = validData(validData >= DEAD_CELL_THRESHOLD);
                end
                meanVal = mean(validData);
                stdVal = std(validData);
            end
        end

        % Top row: Time traces for all thresholds
        for t_idx = 1:3
            nexttile;
            hold on;

            % Plot raw activity
            plot(timepoints, trace_raw, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Raw Activity');

            % Plot BIN binarized trace
            plot(timepoints, traces_bin{t_idx}, 'Color', [0.2 0.4 0.8], 'LineWidth', 2, ...
                 'DisplayName', sprintf('BIN %.1fσ (r=%.3f)', thresholds_all(t_idx), corr_delta_BIN(t_idx)));

            % Plot BIN_AN binarized trace
            plot(timepoints, traces_binan{t_idx}, 'Color', [0.2 0.7 0.3], 'LineWidth', 2, ...
                 'DisplayName', sprintf('BIN_AN %.1fσ (r=%.3f)', thresholds_all(t_idx), corr_delta_BINAN(t_idx)));

            % Plot ABS binarized trace (absolute dF/F — Ising cutoff)
            if thresholds_all(t_idx) == 2.0
                lw_abs = 2.6;  % emphasise the Ising production threshold
            else
                lw_abs = 2.0;
            end
            plot(timepoints, traces_abs{t_idx}, 'Color', [0 0.5 0], 'LineWidth', lw_abs, ...
                 'LineStyle', '--', ...
                 'DisplayName', sprintf('ABS %.1f dF/F (r=%.3f)', thresholds_all(t_idx), corr_delta_ABS(t_idx)));

            xlabel('Timepoint (frames)');
            ylabel('Entropy (normalized)');
            if thresholds_all(t_idx) == 2.0
                title(sprintf('Threshold = %.1f (Ising: ABS 2.0)', thresholds_all(t_idx)), 'Color', [0 0.5 0]);
            else
                title(sprintf('Threshold = %.1f', thresholds_all(t_idx)));
            end
            ylim([0 1]);
            legend('Location', 'best', 'FontSize', 8);
            grid on;
            box on;
            hold off;
        end

        % Bottom row: Spatial patterns at minimum entropy timepoint for all thresholds
        for t_idx = 1:3
            nexttile;

            if ~isempty(dataFlat) && ~isnan(meanVal) && stdVal > 0
                % Show binarized pattern at BIN threshold
                frameData_z_BIN = (dataFlat(:, min_ent_t) - meanVal) / stdVal;
                bin_pattern = double(frameData_z_BIN > thresholds_all(t_idx));
                bin_grid = reshape(bin_pattern, [gridY, gridX]);

                imagesc(bin_grid);
                colormap(gca, 'gray');
                axis equal tight;
                set(gca, 'YDir', 'normal');
                xlabel('Grid X');
                ylabel('Grid Y');

                % Get entropy value at minimum entropy timepoint
                entropy_at_min = trace_raw(min_ent_t);
                title(sprintf('Spatial (t=%d, E=%.3f)', min_ent_t, entropy_at_min));
                colorbar;
            else
                % If no Grid40 data available, show placeholder
                text(0.5, 0.5, 'No Grid40 data available', ...
                     'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'middle');
                axis off;
            end
        end

        sgtitle(sprintf('Raw Activity Entropy vs Grid40 Binarized Entropy: %s Rec %d, Trial %d', ...
                exemplar_condition, exemplar_rec_idx, exemplar_trial), 'FontWeight', 'bold');
    end
end
end  % End loop over recordings

% Plot 8.1b: Moran's I Spatial Clustering (1.5σ, 2.0σ, 2.0 dF/F)
fprintf('Creating Plot 8.1b: Moran''s I spatial clustering (1.5σ, 2.0σ, 2.0 dF/F)\n');

% List of exemplars for the Moran's I spatial clustering plot.
% Each row = {condition, recording index, trial index}.
% This is decoupled from the Entropy Trace Comparison exemplars so you can
% add more Moran's I examples without affecting the other plots.
moransI_exemplars = {
    'Expert',   1,  1;   % Expert Rec 1, Trial 1
    'Expert',   5,  1;   % Expert Rec 5, Trial 1 (the historical example)
    'Expert',   2,  1;   % Expert Rec 2, Trial 1
    'Expert',   3,  1;   % Expert Rec 3, Trial 1
    'Expert',   6,  1;   % Expert Rec 6, Trial 1
    'Expert',   5,  5;   % Expert Rec 5, Trial 5 (same rec, different trial)
    'Beginner', 5,  1;   % Beginner Rec 5, Trial 1 (different condition for contrast)
};

for mi_loop_idx = 1:size(moransI_exemplars, 1)
    mi_condition = moransI_exemplars{mi_loop_idx, 1};
    mi_rec_idx   = moransI_exemplars{mi_loop_idx, 2};
    mi_trial     = moransI_exemplars{mi_loop_idx, 3};

    fprintf('  Creating Moran''s I figure for %s Rec %d Trial %d\n', ...
            mi_condition, mi_rec_idx, mi_trial);

if isfield(EntropyPreservation_Results, mi_condition) && ...
   length(EntropyPreservation_Results.(mi_condition).entropy_raw) >= mi_rec_idx

    % Get raw Grid40 data for the example recording
    if isfield(Grid40, [mi_condition 'Individual']) && ...
       length(Grid40.([mi_condition 'Individual']).AllNeurons) >= mi_rec_idx && ...
       isfield(Grid40.([mi_condition 'Individual']).AllNeurons(mi_rec_idx), 'P1')

        gridData_cell = Grid40.([mi_condition 'Individual']).AllNeurons(mi_rec_idx).P1;

        % Handle cell vs non-cell
        if iscell(gridData_cell) && ~isempty(gridData_cell)
            gridData = gridData_cell{:};
        else
            gridData = gridData_cell;
        end

        % Get dimensions and grid coordinates
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
        nGridCells = gridY * gridX;
        [gridX_coords, gridY_coords] = meshgrid(1:gridX, 1:gridY);
        grid_coords = [gridY_coords(:), gridX_coords(:)];

        % Get data for the chosen trial
        if nTrials >= mi_trial
            dataTrial1 = gridData(:, :, :, mi_trial);
            dataFlat = reshape(dataTrial1, [nGridCells, nTimepoints]);

            % Calculate statistics for binarization (same as before)
            validData = dataFlat(~isnan(dataFlat));
            if Remove_Low_Values
                validData = validData(validData >= DEAD_CELL_THRESHOLD);
            end
            meanVal = mean(validData);
            stdVal = std(validData);

            thresh_1p0 = meanVal + 1.0 * stdVal;
            thresh_1p5 = meanVal + 1.5 * stdVal;
            thresh_2p0 = meanVal + 2.0 * stdVal;

            % Get per-animal statistics for BIN_AN
            currentAnimalID = Rec.(mi_condition).AnimalID{mi_rec_idx};
            validFieldName = ['Animal_' currentAnimalID];
            if isfield(AnimalStats_VarExplained.(mi_condition), validFieldName)
                animal_mean_BINAN = AnimalStats_VarExplained.(mi_condition).(validFieldName).mean;
                animal_std_BINAN = AnimalStats_VarExplained.(mi_condition).(validFieldName).std;
            else
                animal_mean_BINAN = meanVal;
                animal_std_BINAN = stdVal;
            end

            % Initialize storage for Moran's I values across timepoints
            moransI_raw = zeros(nTimepoints, 1);
            moransI_BIN_1p0 = zeros(nTimepoints, 1);
            moransI_BIN_1p5 = zeros(nTimepoints, 1);
            moransI_BIN_2p0 = zeros(nTimepoints, 1);
            moransI_BINAN_1p0 = zeros(nTimepoints, 1);
            moransI_BINAN_1p5 = zeros(nTimepoints, 1);
            moransI_BINAN_2p0 = zeros(nTimepoints, 1);
            moransI_ABS_2p0 = zeros(nTimepoints, 1);

            % Compute Moran's I for each timepoint
            for t = 1:nTimepoints
                frameData = dataFlat(:, t);
                frameGrid = reshape(frameData, [gridY, gridX]);

                % Raw data Moran's I
                frameGrid_norm = (frameData - nanmean(frameData)) / nanstd(frameData);
                moransI_raw(t) = calculate_moransI_grid(frameGrid_norm, gridY, gridX);

                % Binarized data
                if stdVal > 0
                    % BIN thresholds
                    frameData_z_BIN = (frameData - meanVal) / stdVal;
                    bin_1p0 = double(frameData_z_BIN > 1.0);
                    bin_1p5 = double(frameData_z_BIN > 1.5);
                    bin_2p0 = double(frameData_z_BIN > 2.0);

                    moransI_BIN_1p0(t) = calculate_moransI_grid(bin_1p0, gridY, gridX);
                    moransI_BIN_1p5(t) = calculate_moransI_grid(bin_1p5, gridY, gridX);
                    moransI_BIN_2p0(t) = calculate_moransI_grid(bin_2p0, gridY, gridX);
                else
                    moransI_BIN_1p0(t) = NaN;
                    moransI_BIN_1p5(t) = NaN;
                    moransI_BIN_2p0(t) = NaN;
                end

                % BIN_AN thresholds
                if animal_std_BINAN > 0
                    frameData_z_BINAN = (frameData - animal_mean_BINAN) / animal_std_BINAN;
                    binan_1p0 = double(frameData_z_BINAN > 1.0);
                    binan_1p5 = double(frameData_z_BINAN > 1.5);
                    binan_2p0 = double(frameData_z_BINAN > 2.0);

                    moransI_BINAN_1p0(t) = calculate_moransI_grid(binan_1p0, gridY, gridX);
                    moransI_BINAN_1p5(t) = calculate_moransI_grid(binan_1p5, gridY, gridX);
                    moransI_BINAN_2p0(t) = calculate_moransI_grid(binan_2p0, gridY, gridX);
                else
                    moransI_BINAN_1p0(t) = NaN;
                    moransI_BINAN_1p5(t) = NaN;
                    moransI_BINAN_2p0(t) = NaN;
                end

                % ABS absolute dF/F threshold (Ising cutoff at 2.0 dF/F)
                abs_2p0 = double(frameData > 2.0);
                moransI_ABS_2p0(t) = calculate_moransI_grid(abs_2p0, gridY, gridX);
            end

            % Create figure with 2 rows × 3 columns
            % Columns: 1.5σ | 2.0σ | 2.0 dF/F (absolute, Ising cutoff)
            figure('Name', sprintf('Moran''s I Spatial Clustering - %s Rec %d Trial %d', ...
                                   mi_condition, mi_rec_idx, mi_trial));
            tiledlayout(2, 3);

            % Define columns: each column has (label, moransI trace, bin pattern fn)
            % Column 1: 1.5σ — BIN + BIN_AN traces, BIN spatial pattern
            % Column 2: 2.0σ — BIN + BIN_AN traces, BIN spatial pattern
            % Column 3: 2.0 dF/F — ABS trace only, ABS spatial pattern
            col_labels    = {'1.5σ', '2.0σ', '2.0 dF/F (Ising)'};
            col_is_abs    = [false, false, true];
            col_thresh    = [1.5, 2.0, 2.0];   % σ for cols 1–2, dF/F for col 3
            moransI_BIN_cols   = {moransI_BIN_1p5, moransI_BIN_2p0, []};
            moransI_BINAN_cols = {moransI_BINAN_1p5, moransI_BINAN_2p0, []};
            moransI_ABS_cols   = {[], [], moransI_ABS_2p0};

            % Find peak Moran's I timepoint (used for all spatial examples)
            [~, peak_t] = max(moransI_raw);
            if isempty(peak_t) || isnan(peak_t)
                peak_t = round(nTimepoints / 2);  % Fallback to midpoint if no valid peak
            end

            % Top row: Time traces for all 3 columns
            for t_idx = 1:3
                nexttile;
                hold on;

                timeaxis = 1:nTimepoints;
                plot(timeaxis, moransI_raw, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Raw');

                if col_is_abs(t_idx)
                    % ABS (absolute dF/F — Ising cutoff)
                    plot(timeaxis, moransI_ABS_cols{t_idx}, 'Color', [0 0.5 0], 'LineWidth', 2.6, ...
                         'LineStyle', '--', ...
                         'DisplayName', sprintf('ABS %.1f dF/F', col_thresh(t_idx)));
                else
                    % σ-normalized BIN and BIN_AN
                    plot(timeaxis, moransI_BIN_cols{t_idx}, 'Color', [0.2 0.4 0.8], 'LineWidth', 2, ...
                         'DisplayName', sprintf('BIN %.1fσ', col_thresh(t_idx)));
                    plot(timeaxis, moransI_BINAN_cols{t_idx}, 'Color', [0.2 0.7 0.3], 'LineWidth', 2, ...
                         'DisplayName', sprintf('BIN_AN %.1fσ', col_thresh(t_idx)));
                end

                xlabel('Timepoint (frames)');
                ylabel('Moran''s I');
                if col_is_abs(t_idx)
                    title(sprintf('Threshold = %s', col_labels{t_idx}), 'Color', [0 0.5 0], 'FontWeight', 'bold');
                else
                    title(sprintf('Threshold = %s', col_labels{t_idx}));
                end
                legend('Location', 'best', 'FontSize', 9);
                grid on;
                box on;
                ylim([-1 1]);
                hold off;
            end

            % Bottom row: Spatial patterns at peak Moran's I timepoint for all columns
            for t_idx = 1:3
                nexttile;

                if col_is_abs(t_idx)
                    % Absolute dF/F binarization on raw values
                    bin_pattern = double(dataFlat(:, peak_t) > col_thresh(t_idx));
                    moransI_peak_bin = moransI_ABS_cols{t_idx}(peak_t);
                    panel_title_color = [0 0.5 0];
                    panel_title_weight = 'bold';
                else
                    % Sigma binarization on z-scored data
                    if stdVal > 0
                        frameData_z_BIN = (dataFlat(:, peak_t) - meanVal) / stdVal;
                    else
                        frameData_z_BIN = zeros(size(dataFlat(:, peak_t)));
                    end
                    bin_pattern = double(frameData_z_BIN > col_thresh(t_idx));
                    moransI_peak_bin = moransI_BIN_cols{t_idx}(peak_t);
                    panel_title_color = 'k';
                    panel_title_weight = 'normal';
                end
                bin_grid = reshape(bin_pattern, [gridY, gridX]);

                imagesc(bin_grid);
                colormap(gca, 'gray');
                axis equal tight;
                set(gca, 'YDir', 'normal');
                xlabel('Grid X');
                ylabel('Grid Y');

                title(sprintf('Spatial (t=%d, I=%.3f)', peak_t, moransI_peak_bin), ...
                      'Color', panel_title_color, 'FontWeight', panel_title_weight);
                colorbar;
            end

            sgtitle(sprintf('Spatial Clustering (Moran''s I): %s Rec %d, Trial %d  |  1.5σ, 2.0σ, 2.0 dF/F (Ising)', ...
                    mi_condition, mi_rec_idx, mi_trial), 'FontWeight', 'bold');

            fprintf('  Moran''s I figure created: %s Rec %d Trial %d (1.5σ, 2.0σ, 2.0 dF/F)\n', ...
                    mi_condition, mi_rec_idx, mi_trial);
        end
    end
end
end  % End loop over moransI_exemplars

% Plot 8.1c: Dispersion Spatial Organization (All 3 Thresholds)
fprintf('Creating Plot 8.1c: Dispersion spatial organization (all 3 thresholds)\n');

for rec_loop_idx = 1:length(exemplar_recs)
    exemplar_rec_idx = exemplar_recs(rec_loop_idx);

    fprintf('  Creating Dispersion figure for Recording %d\n', exemplar_rec_idx);

if isfield(EntropyPreservation_Results, exemplar_condition) && ...
   length(EntropyPreservation_Results.(exemplar_condition).entropy_raw) >= exemplar_rec_idx

    % Get raw Grid40 data for the example recording
    if isfield(Grid40, [exemplar_condition 'Individual']) && ...
       length(Grid40.([exemplar_condition 'Individual']).AllNeurons) >= exemplar_rec_idx && ...
       isfield(Grid40.([exemplar_condition 'Individual']).AllNeurons(exemplar_rec_idx), 'P1')

        gridData_cell = Grid40.([exemplar_condition 'Individual']).AllNeurons(exemplar_rec_idx).P1;

        % Handle cell vs non-cell
        if iscell(gridData_cell) && ~isempty(gridData_cell)
            gridData = gridData_cell{:};
        else
            gridData = gridData_cell;
        end

        % Get dimensions and grid coordinates
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
        nGridCells = gridY * gridX;
        [gridX_coords, gridY_coords] = meshgrid(1:gridX, 1:gridY);
        grid_coords = [gridY_coords(:), gridX_coords(:)];

        % Get data for trial 1
        if nTrials >= exemplar_trial
            dataTrial1 = gridData(:, :, :, exemplar_trial);
            dataFlat = reshape(dataTrial1, [nGridCells, nTimepoints]);

            % Calculate statistics for binarization
            validData = dataFlat(~isnan(dataFlat));
            if Remove_Low_Values
                validData = validData(validData >= DEAD_CELL_THRESHOLD);
            end
            meanVal = mean(validData);
            stdVal = std(validData);

            % Get per-animal statistics for BIN_AN
            currentAnimalID = Rec.(exemplar_condition).AnimalID{exemplar_rec_idx};
            validFieldName = ['Animal_' currentAnimalID];
            if isfield(AnimalStats_VarExplained.(exemplar_condition), validFieldName)
                animal_mean_BINAN = AnimalStats_VarExplained.(exemplar_condition).(validFieldName).mean;
                animal_std_BINAN = AnimalStats_VarExplained.(exemplar_condition).(validFieldName).std;
            else
                animal_mean_BINAN = meanVal;
                animal_std_BINAN = stdVal;
            end

            % Initialize storage for dispersion values across timepoints
            dispersion_raw = zeros(nTimepoints, 1);
            dispersion_BIN_1p0 = zeros(nTimepoints, 1);
            dispersion_BIN_1p5 = zeros(nTimepoints, 1);
            dispersion_BIN_2p0 = zeros(nTimepoints, 1);
            dispersion_BINAN_1p0 = zeros(nTimepoints, 1);
            dispersion_BINAN_1p5 = zeros(nTimepoints, 1);
            dispersion_BINAN_2p0 = zeros(nTimepoints, 1);

            % Compute dispersion for each timepoint
            for t = 1:nTimepoints
                frameData = dataFlat(:, t);

                % Raw data dispersion (weighted)
                validIdx = ~isnan(frameData);
                if Remove_Low_Values
                    validIdx = validIdx & (frameData >= DEAD_CELL_THRESHOLD);
                end
                frameData_valid = frameData(validIdx);
                grid_coords_valid = grid_coords(validIdx, :);

                if length(frameData_valid) > 2
                    dispersion_raw(t) = calculate_dispersion_weighted(frameData_valid, grid_coords_valid);
                else
                    dispersion_raw(t) = NaN;
                end

                % Binarized data
                if stdVal > 0
                    % BIN thresholds
                    frameData_z_BIN = (frameData - meanVal) / stdVal;
                    bin_1p0 = double(frameData_z_BIN > 1.0);
                    bin_1p5 = double(frameData_z_BIN > 1.5);
                    bin_2p0 = double(frameData_z_BIN > 2.0);

                    dispersion_BIN_1p0(t) = calculate_dispersion_binary(bin_1p0, grid_coords);
                    dispersion_BIN_1p5(t) = calculate_dispersion_binary(bin_1p5, grid_coords);
                    dispersion_BIN_2p0(t) = calculate_dispersion_binary(bin_2p0, grid_coords);
                else
                    dispersion_BIN_1p0(t) = NaN;
                    dispersion_BIN_1p5(t) = NaN;
                    dispersion_BIN_2p0(t) = NaN;
                end

                % BIN_AN thresholds
                if animal_std_BINAN > 0
                    frameData_z_BINAN = (frameData - animal_mean_BINAN) / animal_std_BINAN;
                    binan_1p0 = double(frameData_z_BINAN > 1.0);
                    binan_1p5 = double(frameData_z_BINAN > 1.5);
                    binan_2p0 = double(frameData_z_BINAN > 2.0);

                    dispersion_BINAN_1p0(t) = calculate_dispersion_binary(binan_1p0, grid_coords);
                    dispersion_BINAN_1p5(t) = calculate_dispersion_binary(binan_1p5, grid_coords);
                    dispersion_BINAN_2p0(t) = calculate_dispersion_binary(binan_2p0, grid_coords);
                else
                    dispersion_BINAN_1p0(t) = NaN;
                    dispersion_BINAN_1p5(t) = NaN;
                    dispersion_BINAN_2p0(t) = NaN;
                end
            end

            % Create figure with 2 rows × 3 thresholds (traces on top, spatial below)
            figure('Name', sprintf('Dispersion Spatial Organization - Rec %d', exemplar_rec_idx));
            tiledlayout(2, 3);

            % Define threshold details
            thresholds_all = [1.0, 1.5, 2.0];
            dispersion_BIN_all = {dispersion_BIN_1p0, dispersion_BIN_1p5, dispersion_BIN_2p0};
            dispersion_BINAN_all = {dispersion_BINAN_1p0, dispersion_BINAN_1p5, dispersion_BINAN_2p0};

            % Find minimum dispersion timepoint (used for all spatial examples)
            [~, min_disp_t] = min(dispersion_raw);

            % Top row: Time traces for all thresholds
            for t_idx = 1:3
                nexttile;
                hold on;

                timeaxis = 1:nTimepoints;
                plot(timeaxis, dispersion_raw, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Raw');
                plot(timeaxis, dispersion_BIN_all{t_idx}, 'Color', [0.2 0.4 0.8], 'LineWidth', 2, ...
                     'DisplayName', sprintf('BIN %.1fσ', thresholds_all(t_idx)));
                plot(timeaxis, dispersion_BINAN_all{t_idx}, 'Color', [0.2 0.7 0.3], 'LineWidth', 2, ...
                     'DisplayName', sprintf('BIN_AN %.1fσ', thresholds_all(t_idx)));

                xlabel('Timepoint (frames)');
                ylabel('Dispersion (grid units)');
                title(sprintf('Threshold = %.1fσ', thresholds_all(t_idx)));
                legend('Location', 'best', 'FontSize', 9);
                grid on;
                box on;
                hold off;
            end

            % Bottom row: Spatial patterns at minimum dispersion timepoint for all thresholds
            for t_idx = 1:3
                nexttile;

                % Show binarized pattern at BIN threshold
                frameData_z_BIN = (dataFlat(:, min_disp_t) - meanVal) / stdVal;
                bin_pattern = double(frameData_z_BIN > thresholds_all(t_idx));
                bin_grid = reshape(bin_pattern, [gridY, gridX]);

                % Calculate centroid for visualization
                active_inds = find(bin_pattern > 0);
                if ~isempty(active_inds)
                    act_pos = grid_coords(active_inds, :);
                    centroid = mean(act_pos, 1);
                else
                    centroid = [gridY/2, gridX/2];
                end

                hold on;
                imagesc(bin_grid);
                colormap(gca, 'gray');
                % Overlay centroid
                plot(centroid(2), centroid(1), 'c*', 'MarkerSize', 20, 'LineWidth', 2);
                plot(centroid(2), centroid(1), 'co', 'MarkerSize', 25, 'LineWidth', 2);

                axis equal tight;
                set(gca, 'YDir', 'normal');
                xlabel('Grid X');
                ylabel('Grid Y');

                % Get dispersion value at minimum dispersion timepoint
                dispersion_at_min = dispersion_raw(min_disp_t);
                title(sprintf('Spatial (t=%d, D=%.3f)', min_disp_t, dispersion_at_min));
                colorbar;
                hold off;
            end

            sgtitle(sprintf('Spatial Organization (Dispersion): %s Rec %d, Trial %d', ...
                    exemplar_condition, exemplar_rec_idx, exemplar_trial), 'FontWeight', 'bold');

            fprintf('Plot 8.1c complete: Dispersion spatial organization created\n');
        end
    end
end
end  % End loop over recordings

% Plot 8.2: Frame-to-Frame Delta Correlations (BIN vs BIN_AN)
fprintf('Creating Plot 8.2: Frame-to-frame delta correlations (BIN vs BIN_AN)\n');

% Collect correlations by condition and threshold
figure('Name', 'Frame-to-Frame Delta Correlations - BIN vs BIN_AN vs ABS');
tiledlayout(2, 2);

thresholds = [1.0, 1.5, 2.0];
threshold_names = {'1.0', '1.5', '2.0'};
subplot_idx = 1;

for t = 1:length(thresholds)
    threshold_str = sprintf('%dp%d', fix(thresholds(t)), round(mod(thresholds(t), 1)*10));

    nexttile;
    hold on;

    % Collect all correlations across conditions
    all_corr_BIN = [];
    all_corr_BINAN = [];
    all_corr_ABS = [];
    cond_labels = {};

    for c = 1:length(conditions)
        condition = conditions{c};

        if ~isfield(EntropyPreservation_Results, condition)
            continue;
        end

        % Get field names dynamically
        field_BIN = sprintf('corr_delta_BIN_%s', threshold_str);
        field_BINAN = sprintf('corr_delta_BINAN_%s', threshold_str);
        field_ABS = sprintf('corr_delta_ABS_%s', threshold_str);

        if isfield(EntropyPreservation_Results.(condition), field_BIN)
            corr_BIN = EntropyPreservation_Results.(condition).(field_BIN);
            corr_BINAN = EntropyPreservation_Results.(condition).(field_BINAN);
            corr_ABS = EntropyPreservation_Results.(condition).(field_ABS);

            all_corr_BIN = [all_corr_BIN; corr_BIN];
            all_corr_BINAN = [all_corr_BINAN; corr_BINAN];
            all_corr_ABS = [all_corr_ABS; corr_ABS];

            % Add condition labels
            cond_labels = [cond_labels; repmat({condition}, length(corr_BIN), 1)];
        end
    end

    % Create scatter plot with three columns: BIN, BIN_AN, ABS
    x_pos_BIN = ones(size(all_corr_BIN)) * 0.8 + randn(size(all_corr_BIN)) * 0.05;
    x_pos_BINAN = ones(size(all_corr_BINAN)) * 2.0 + randn(size(all_corr_BINAN)) * 0.05;
    x_pos_ABS = ones(size(all_corr_ABS)) * 3.2 + randn(size(all_corr_ABS)) * 0.05;

    scatter(x_pos_BIN, all_corr_BIN, 60, [0.2 0.4 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(x_pos_BINAN, all_corr_BINAN, 60, [0.2 0.7 0.3], 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(x_pos_ABS, all_corr_ABS, 60, [0 0.5 0], 'filled', 'MarkerFaceAlpha', 0.7);

    % Plot means
    mean_BIN = nanmean(all_corr_BIN);
    mean_BINAN = nanmean(all_corr_BINAN);
    mean_ABS = nanmean(all_corr_ABS);
    plot(0.8, mean_BIN, 'o', 'Color', [0.2 0.4 0.8], 'MarkerSize', 12, 'LineWidth', 2.5);
    plot(2.0, mean_BINAN, 'o', 'Color', [0.2 0.7 0.3], 'MarkerSize', 12, 'LineWidth', 2.5);
    plot(3.2, mean_ABS, 's', 'Color', [0 0.5 0], 'MarkerSize', 14, 'LineWidth', 2.8);

    set(gca, 'XLim', [0.3 3.7], 'XTick', [0.8 2.0 3.2], ...
             'XTickLabel', {'BIN σ', 'BIN\_AN σ', 'ABS dF/F'});
    ylabel('Delta Correlation');
    titleColor = 'k';
    if t == 3  % 2.0 column: highlight the Ising cutoff
        titleColor = [0 0.5 0];
    end
    title(sprintf('Threshold %s', threshold_names{t}), 'Color', titleColor);
    grid on;
    box on;
    hold off;

    subplot_idx = subplot_idx + 1;
end

% Empty fourth subplot for layout
nexttile;
axis off;
text(0.05, 0.9, 'Frame-to-frame entropy delta correlations', ...
     'Units', 'normalized', 'FontWeight', 'bold');
text(0.05, 0.75, 'Blue  = BIN (recording-level z-score)', ...
     'Units', 'normalized', 'Color', [0.2 0.4 0.8]);
text(0.05, 0.65, 'Green = BIN\_AN (per-animal z-score)', ...
     'Units', 'normalized', 'Color', [0.2 0.7 0.3]);
text(0.05, 0.55, 'Dark green = ABS (absolute dF/F — Ising cutoff)', ...
     'Units', 'normalized', 'Color', [0 0.5 0], 'FontWeight', 'bold');

sgtitle(sprintf('Frame-to-Frame Entropy Correlations: BIN σ vs BIN\\_AN σ vs ABS dF/F (Ising cutoff)'), ...
        'FontWeight', 'bold');

% Plot 8.3: Entropy Distribution Histograms by Condition
fprintf('Creating Plot 8.3: Entropy distribution histograms\n');

figure('Name', 'Entropy Distribution Comparison');
tiledlayout(2, 2);

for c = 1:min(4, length(conditions))
    condition = conditions{c};

    if ~isfield(EntropyPreservation_Results, condition)
        continue;
    end

    nexttile;
    hold on;

    % Get distributions
    dist_raw = EntropyPreservation_Results.(condition).dist_raw;
    dist_bin_1p5 = EntropyPreservation_Results.(condition).dist_BIN_1p5;
    dist_binan_1p5 = EntropyPreservation_Results.(condition).dist_BINAN_1p5;
    dist_abs_2p0 = EntropyPreservation_Results.(condition).dist_ABS_2p0;

    % Create histograms
    histogram(dist_raw, 30, 'FaceColor', 'k', 'FaceAlpha', 0.4, 'EdgeColor', 'k', 'DisplayName', 'Raw');
    histogram(dist_bin_1p5, 30, 'FaceColor', [0.2 0.4 0.8], 'FaceAlpha', 0.3, 'EdgeColor', [0.2 0.4 0.8], ...
              'DisplayName', 'BIN 1.5σ');
    histogram(dist_binan_1p5, 30, 'FaceColor', [0.2 0.7 0.3], 'FaceAlpha', 0.3, 'EdgeColor', [0.2 0.7 0.3], ...
              'DisplayName', 'BIN_AN 1.5σ');
    histogram(dist_abs_2p0, 30, 'FaceColor', [0 0.5 0], 'FaceAlpha', 0.4, 'EdgeColor', [0 0.5 0], ...
              'LineWidth', 1.3, 'DisplayName', 'ABS 2.0 dF/F (Ising)');

    xlabel('Entropy');
    ylabel('Frequency');
    title(sprintf('%s Condition (n=%d)', condition, length(dist_raw)));
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    box on;
    hold off;
end

sgtitle('Entropy Value Distributions (Raw vs Binarized: BIN 1.5σ, BIN\_AN 1.5σ, ABS 2.0 dF/F)', 'FontWeight', 'bold');

% Plot 8.4: Correlation Summary by Condition and Threshold
fprintf('Creating Plot 8.4: Correlation summary heatmap\n');

figure('Name', 'Correlation Summary - Delta and Absolute');
tiledlayout(1, 3);

% Prepare data for heatmap
condNames = {};
bin_delta_data = [];
binan_delta_data = [];
abs_delta_data = [];

for c = 1:length(conditions)
    condition = conditions{c};

    if ~isfield(EntropyPreservation_Results, condition)
        continue;
    end

    condNames = [condNames; condition];

    % Get mean delta correlations for each threshold
    delta_BIN = [EntropyPreservation_Results.(condition).corr_delta_BIN_1p0_mean, ...
                 EntropyPreservation_Results.(condition).corr_delta_BIN_1p5_mean, ...
                 EntropyPreservation_Results.(condition).corr_delta_BIN_2p0_mean];
    delta_BINAN = [EntropyPreservation_Results.(condition).corr_delta_BINAN_1p0_mean, ...
                   EntropyPreservation_Results.(condition).corr_delta_BINAN_1p5_mean, ...
                   EntropyPreservation_Results.(condition).corr_delta_BINAN_2p0_mean];
    delta_ABS = [EntropyPreservation_Results.(condition).corr_delta_ABS_1p0_mean, ...
                 EntropyPreservation_Results.(condition).corr_delta_ABS_1p5_mean, ...
                 EntropyPreservation_Results.(condition).corr_delta_ABS_2p0_mean];

    bin_delta_data = [bin_delta_data; delta_BIN];
    binan_delta_data = [binan_delta_data; delta_BINAN];
    abs_delta_data = [abs_delta_data; delta_ABS];
end

% Plot BIN heatmap
nexttile;
imagesc(bin_delta_data);
colormap(gca, blue_red_cmap);
clim([0 1]);
set(gca, 'XTick', 1:3, 'XTickLabel', {'1.0σ', '1.5σ', '2.0σ'}, ...
         'YTick', 1:length(condNames), 'YTickLabel', condNames);
xlabel('Threshold');
ylabel('Condition');
title('BIN Normalization (Recording-level)');
colorbar;

% Add correlation values as text
for i = 1:size(bin_delta_data, 1)
    for j = 1:size(bin_delta_data, 2)
        text(j, i, sprintf('%.2f', bin_delta_data(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9);
    end
end

% Plot BIN_AN heatmap
nexttile;
imagesc(binan_delta_data);
colormap(gca, blue_red_cmap);
clim([0 1]);
set(gca, 'XTick', 1:3, 'XTickLabel', {'1.0σ', '1.5σ', '2.0σ'}, ...
         'YTick', 1:length(condNames), 'YTickLabel', condNames);
xlabel('Threshold');
ylabel('Condition');
title('BIN_AN Normalization (Per-animal)');
colorbar;

% Add correlation values as text
for i = 1:size(binan_delta_data, 1)
    for j = 1:size(binan_delta_data, 2)
        text(j, i, sprintf('%.2f', binan_delta_data(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9);
    end
end

% Plot ABS heatmap (absolute dF/F — Ising production)
nexttile;
imagesc(abs_delta_data);
colormap(gca, blue_red_cmap);
clim([0 1]);
set(gca, 'XTick', 1:3, 'XTickLabel', {'1.0 dF/F', '1.5 dF/F', '2.0 dF/F'}, ...
         'YTick', 1:length(condNames), 'YTickLabel', condNames);
xlabel('Threshold');
ylabel('Condition');
title('ABS Normalization (Ising cutoff)', 'Color', [0 0.5 0]);
colorbar;

% Add correlation values as text; green box around the 2.0 dF/F column
for i = 1:size(abs_delta_data, 1)
    for j = 1:size(abs_delta_data, 2)
        text(j, i, sprintf('%.2f', abs_delta_data(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9);
    end
end
% Highlight the production cutoff column (j=3, i.e. 2.0 dF/F)
hold on;
nRows_ABS = size(abs_delta_data, 1);
rectangle('Position', [2.5, 0.5, 1, nRows_ABS], ...
          'EdgeColor', [0 0.5 0], 'LineWidth', 2.5);
hold off;

sgtitle('Entropy Correlation Summary - Frame-to-Frame Delta (How Well Temporal Entropy Changes Are Preserved)', 'FontWeight', 'bold');

% Plot 8.4a: Entropy Trial-Mean Correlation Summary
fprintf('Creating Plot 8.4a: Entropy trial-mean correlation summary heatmap\n');

figure('Name', 'Entropy Correlation Summary - Trial Mean');
tiledlayout(1, 3);

% Prepare data for heatmap (trial-mean correlations)
condNames_tm = {};
bin_trialmean_data = [];
binan_trialmean_data = [];
abs_trialmean_data = [];

for c = 1:length(conditions)
    condition = conditions{c};

    if ~isfield(EntropyPreservation_Results, condition)
        continue;
    end

    condNames_tm = [condNames_tm; condition];

    % Get mean trial-mean correlations for each threshold
    trialmean_BIN = [EntropyPreservation_Results.(condition).corr_trialmean_BIN_1p0_mean, ...
                     EntropyPreservation_Results.(condition).corr_trialmean_BIN_1p5_mean, ...
                     EntropyPreservation_Results.(condition).corr_trialmean_BIN_2p0_mean];
    trialmean_BINAN = [EntropyPreservation_Results.(condition).corr_trialmean_BINAN_1p0_mean, ...
                       EntropyPreservation_Results.(condition).corr_trialmean_BINAN_1p5_mean, ...
                       EntropyPreservation_Results.(condition).corr_trialmean_BINAN_2p0_mean];
    trialmean_ABS = [EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p0_mean, ...
                     EntropyPreservation_Results.(condition).corr_trialmean_ABS_1p5_mean, ...
                     EntropyPreservation_Results.(condition).corr_trialmean_ABS_2p0_mean];

    bin_trialmean_data = [bin_trialmean_data; trialmean_BIN];
    binan_trialmean_data = [binan_trialmean_data; trialmean_BINAN];
    abs_trialmean_data = [abs_trialmean_data; trialmean_ABS];
end

% Plot BIN heatmap
nexttile;
imagesc(bin_trialmean_data);
colormap(gca, blue_red_cmap);
clim([0 1]);
set(gca, 'XTick', 1:3, 'XTickLabel', {'1.0σ', '1.5σ', '2.0σ'}, ...
         'YTick', 1:length(condNames_tm), 'YTickLabel', condNames_tm);
xlabel('Threshold');
ylabel('Condition');
title('BIN Normalization (Recording-level)');
colorbar;

% Add correlation values as text
for i = 1:size(bin_trialmean_data, 1)
    for j = 1:size(bin_trialmean_data, 2)
        text(j, i, sprintf('%.2f', bin_trialmean_data(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9);
    end
end

% Plot BIN_AN heatmap
nexttile;
imagesc(binan_trialmean_data);
colormap(gca, blue_red_cmap);
clim([0 1]);
set(gca, 'XTick', 1:3, 'XTickLabel', {'1.0σ', '1.5σ', '2.0σ'}, ...
         'YTick', 1:length(condNames_tm), 'YTickLabel', condNames_tm);
xlabel('Threshold');
ylabel('Condition');
title('BIN_AN Normalization (Per-animal)');
colorbar;

% Add correlation values as text
for i = 1:size(binan_trialmean_data, 1)
    for j = 1:size(binan_trialmean_data, 2)
        text(j, i, sprintf('%.2f', binan_trialmean_data(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9);
    end
end

% Plot ABS heatmap (absolute dF/F — Ising production)
nexttile;
imagesc(abs_trialmean_data);
colormap(gca, blue_red_cmap);
clim([0 1]);
set(gca, 'XTick', 1:3, 'XTickLabel', {'1.0 dF/F', '1.5 dF/F', '2.0 dF/F'}, ...
         'YTick', 1:length(condNames_tm), 'YTickLabel', condNames_tm);
xlabel('Threshold');
ylabel('Condition');
title('ABS Normalization (Ising cutoff)', 'Color', [0 0.5 0]);
colorbar;

for i = 1:size(abs_trialmean_data, 1)
    for j = 1:size(abs_trialmean_data, 2)
        text(j, i, sprintf('%.2f', abs_trialmean_data(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9);
    end
end
hold on;
rectangle('Position', [2.5, 0.5, 1, size(abs_trialmean_data, 1)], ...
          'EdgeColor', [0 0.5 0], 'LineWidth', 2.5);
hold off;

sgtitle('Entropy Correlation Summary - Trial Mean (How Well Average Entropy Across Trials Is Preserved)', 'FontWeight', 'bold');

% Plot 8.4b: Moran's I Frame-to-Frame Correlation Summary
fprintf('Creating Plot 8.4b: Moran''s I frame-to-frame correlation summary heatmap\n');

figure('Name', 'Moran''s I Correlation Summary - Frame-to-Frame');
tiledlayout(1, 3);

% Prepare data for heatmap (frame-to-frame correlations)
condNames_mi = {};
bin_moransI_delta_data = [];
binan_moransI_delta_data = [];
abs_moransI_delta_data = [];

for c = 1:length(conditions)
    condition = conditions{c};

    if ~isfield(EntropyPreservation_Results, condition)
        continue;
    end

    condNames_mi = [condNames_mi; condition];

    % Get mean delta correlations for each threshold (Moran's I uses delta by default like entropy)
    % Note: For Moran's I, frame-to-frame means consecutive timepoint correlations
    % We'll use absolute correlations since Moran's I can be negative
    delta_BIN_mi = [nanmean(EntropyPreservation_Results.(condition).corr_delta_BIN_1p0), ...
                    nanmean(EntropyPreservation_Results.(condition).corr_delta_BIN_1p5), ...
                    nanmean(EntropyPreservation_Results.(condition).corr_delta_BIN_2p0)];
    delta_BINAN_mi = [nanmean(EntropyPreservation_Results.(condition).corr_delta_BINAN_1p0), ...
                      nanmean(EntropyPreservation_Results.(condition).corr_delta_BINAN_1p5), ...
                      nanmean(EntropyPreservation_Results.(condition).corr_delta_BINAN_2p0)];
    delta_ABS_mi = [nanmean(EntropyPreservation_Results.(condition).corr_delta_ABS_1p0), ...
                    nanmean(EntropyPreservation_Results.(condition).corr_delta_ABS_1p5), ...
                    nanmean(EntropyPreservation_Results.(condition).corr_delta_ABS_2p0)];

    bin_moransI_delta_data = [bin_moransI_delta_data; delta_BIN_mi];
    binan_moransI_delta_data = [binan_moransI_delta_data; delta_BINAN_mi];
    abs_moransI_delta_data = [abs_moransI_delta_data; delta_ABS_mi];
end

% Plot BIN heatmap
nexttile;
imagesc(bin_moransI_delta_data);
colormap(gca, blue_red_cmap);
clim([0 1]);
set(gca, 'XTick', 1:3, 'XTickLabel', {'1.0σ', '1.5σ', '2.0σ'}, ...
         'YTick', 1:length(condNames_mi), 'YTickLabel', condNames_mi);
xlabel('Threshold');
ylabel('Condition');
title('BIN Normalization (Recording-level)');
colorbar;

% Add correlation values as text
for i = 1:size(bin_moransI_delta_data, 1)
    for j = 1:size(bin_moransI_delta_data, 2)
        text(j, i, sprintf('%.2f', bin_moransI_delta_data(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9);
    end
end

% Plot BIN_AN heatmap
nexttile;
imagesc(binan_moransI_delta_data);
colormap(gca, blue_red_cmap);
clim([0 1]);
set(gca, 'XTick', 1:3, 'XTickLabel', {'1.0σ', '1.5σ', '2.0σ'}, ...
         'YTick', 1:length(condNames_mi), 'YTickLabel', condNames_mi);
xlabel('Threshold');
ylabel('Condition');
title('BIN_AN Normalization (Per-animal)');
colorbar;

% Add correlation values as text
for i = 1:size(binan_moransI_delta_data, 1)
    for j = 1:size(binan_moransI_delta_data, 2)
        text(j, i, sprintf('%.2f', binan_moransI_delta_data(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9);
    end
end

% Plot ABS heatmap (absolute dF/F — Ising production)
nexttile;
imagesc(abs_moransI_delta_data);
colormap(gca, blue_red_cmap);
clim([0 1]);
set(gca, 'XTick', 1:3, 'XTickLabel', {'1.0 dF/F', '1.5 dF/F', '2.0 dF/F'}, ...
         'YTick', 1:length(condNames_mi), 'YTickLabel', condNames_mi);
xlabel('Threshold');
ylabel('Condition');
title('ABS Normalization (Ising cutoff)', 'Color', [0 0.5 0]);
colorbar;

for i = 1:size(abs_moransI_delta_data, 1)
    for j = 1:size(abs_moransI_delta_data, 2)
        text(j, i, sprintf('%.2f', abs_moransI_delta_data(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9);
    end
end
hold on;
rectangle('Position', [2.5, 0.5, 1, size(abs_moransI_delta_data, 1)], ...
          'EdgeColor', [0 0.5 0], 'LineWidth', 2.5);
hold off;

sgtitle('Moran''s I Correlation Summary - Frame-to-Frame Delta (How Well Temporal Spatial Clustering Changes Are Preserved)', 'FontWeight', 'bold');

% Plot 8.4c: Moran's I Trial-Mean Correlation Summary
fprintf('Creating Plot 8.4c: Moran''s I trial-mean correlation summary heatmap\n');

figure('Name', 'Moran''s I Correlation Summary - Trial Mean');
tiledlayout(1, 3);

% Prepare data for heatmap (trial-mean correlations)
bin_trialmean_moransI_data = [];
binan_trialmean_moransI_data = [];
abs_trialmean_moransI_data = [];

for c = 1:length(conditions)
    condition = conditions{c};

    if ~isfield(EntropyPreservation_Results, condition)
        continue;
    end

    % Get mean trial-mean correlations for each threshold (Moran's I)
    trialmean_moransI_BIN = [EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_1p0_mean, ...
                             EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_1p5_mean, ...
                             EntropyPreservation_Results.(condition).corr_trialmean_moransI_BIN_2p0_mean];
    trialmean_moransI_BINAN = [EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_1p0_mean, ...
                               EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_1p5_mean, ...
                               EntropyPreservation_Results.(condition).corr_trialmean_moransI_BINAN_2p0_mean];
    trialmean_moransI_ABS = [EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p0_mean, ...
                             EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_1p5_mean, ...
                             EntropyPreservation_Results.(condition).corr_trialmean_moransI_ABS_2p0_mean];

    bin_trialmean_moransI_data = [bin_trialmean_moransI_data; trialmean_moransI_BIN];
    binan_trialmean_moransI_data = [binan_trialmean_moransI_data; trialmean_moransI_BINAN];
    abs_trialmean_moransI_data = [abs_trialmean_moransI_data; trialmean_moransI_ABS];
end

% Plot BIN heatmap
nexttile;
imagesc(bin_trialmean_moransI_data);
colormap(gca, blue_red_cmap);
clim([0 1]);
set(gca, 'XTick', 1:3, 'XTickLabel', {'1.0σ', '1.5σ', '2.0σ'}, ...
         'YTick', 1:length(condNames_mi), 'YTickLabel', condNames_mi);
xlabel('Threshold');
ylabel('Condition');
title('BIN Normalization (Recording-level)');
colorbar;

% Add correlation values as text
for i = 1:size(bin_trialmean_moransI_data, 1)
    for j = 1:size(bin_trialmean_moransI_data, 2)
        text(j, i, sprintf('%.2f', bin_trialmean_moransI_data(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9);
    end
end

% Plot BIN_AN heatmap
nexttile;
imagesc(binan_trialmean_moransI_data);
colormap(gca, blue_red_cmap);
clim([0 1]);
set(gca, 'XTick', 1:3, 'XTickLabel', {'1.0σ', '1.5σ', '2.0σ'}, ...
         'YTick', 1:length(condNames_mi), 'YTickLabel', condNames_mi);
xlabel('Threshold');
ylabel('Condition');
title('BIN_AN Normalization (Per-animal)');
colorbar;

% Add correlation values as text
for i = 1:size(binan_trialmean_moransI_data, 1)
    for j = 1:size(binan_trialmean_moransI_data, 2)
        text(j, i, sprintf('%.2f', binan_trialmean_moransI_data(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9);
    end
end

% Plot ABS heatmap (absolute dF/F — Ising production)
nexttile;
imagesc(abs_trialmean_moransI_data);
colormap(gca, blue_red_cmap);
clim([0 1]);
set(gca, 'XTick', 1:3, 'XTickLabel', {'1.0 dF/F', '1.5 dF/F', '2.0 dF/F'}, ...
         'YTick', 1:length(condNames_mi), 'YTickLabel', condNames_mi);
xlabel('Threshold');
ylabel('Condition');
title('ABS Normalization (Ising cutoff)', 'Color', [0 0.5 0]);
colorbar;

for i = 1:size(abs_trialmean_moransI_data, 1)
    for j = 1:size(abs_trialmean_moransI_data, 2)
        text(j, i, sprintf('%.2f', abs_trialmean_moransI_data(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9);
    end
end
hold on;
rectangle('Position', [2.5, 0.5, 1, size(abs_trialmean_moransI_data, 1)], ...
          'EdgeColor', [0 0.5 0], 'LineWidth', 2.5);
hold off;

sgtitle('Moran''s I Correlation Summary - Trial Mean (How Well Average Spatial Clustering Across Trials Is Preserved)', 'FontWeight', 'bold');

% Plot 8.5: BIN vs BIN_AN vs ABS Direct Comparison
fprintf('Creating Plot 8.5: BIN vs BIN_AN vs ABS normalization comparison\n');

figure('Name', 'BIN vs BIN_AN vs ABS Comparison');
tiledlayout(2, 2);

% Prepare comparison data
bin_data = [];
binan_data = [];
abs_data_delta = [];
labels_data = {};

for c = 1:length(conditions)
    condition = conditions{c};

    if ~isfield(EntropyPreservation_Results, condition)
        continue;
    end

    % Collect all delta correlations
    for field_idx = 1:3
        threshold_str = sprintf('%dp%d', fix(thresholds(field_idx)), round(mod(thresholds(field_idx), 1)*10));
        field_BIN = sprintf('corr_delta_BIN_%s', threshold_str);
        field_BINAN = sprintf('corr_delta_BINAN_%s', threshold_str);
        field_ABS = sprintf('corr_delta_ABS_%s', threshold_str);

        if isfield(EntropyPreservation_Results.(condition), field_BIN)
            corr_BIN = nanmean(EntropyPreservation_Results.(condition).(field_BIN));
            corr_BINAN = nanmean(EntropyPreservation_Results.(condition).(field_BINAN));
            corr_ABS = nanmean(EntropyPreservation_Results.(condition).(field_ABS));

            bin_data = [bin_data; corr_BIN];
            binan_data = [binan_data; corr_BINAN];
            abs_data_delta = [abs_data_delta; corr_ABS];
            labels_data = [labels_data; {sprintf('%s, %s', condition, threshold_str)}];
        end
    end
end

% Plot 1: Scatter plot comparing BIN vs BIN_AN vs ABS (three-way)
nexttile;
hold on;
scatter(bin_data, binan_data, 80, [0.3 0.3 0.3], 'o', 'filled', 'MarkerFaceAlpha', 0.5, ...
        'DisplayName', 'BIN vs BIN\_AN');
scatter(bin_data, abs_data_delta, 90, [0 0.5 0], 's', 'filled', 'MarkerFaceAlpha', 0.7, ...
        'DisplayName', 'BIN vs ABS');
plot([0 1], [0 1], 'k--', 'LineWidth', 2, 'DisplayName', 'Equal performance');
xlabel('BIN Correlation (mean)');
ylabel('BIN\_AN / ABS Correlation (mean)');
title('Frame-to-Frame Delta: BIN vs others');
xlim([0 1]);
ylim([0 1]);
axis equal tight;
grid on;
box on;
legend('FontSize', 9, 'Location', 'southeast');
hold off;

% Plot 2: Advantage of each method over BIN (bar plot)
nexttile;
diff_binan = binan_data - bin_data;
diff_abs = abs_data_delta - bin_data;
xBars = 1:length(labels_data);
width = 0.38;
bar(xBars - width/2, diff_binan, width, 'FaceColor', [0.2 0.7 0.3], 'FaceAlpha', 0.7, ...
    'DisplayName', 'BIN\_AN − BIN');
hold on;
bar(xBars + width/2, diff_abs, width, 'FaceColor', [0 0.5 0], 'FaceAlpha', 0.7, ...
    'DisplayName', 'ABS − BIN');
set(gca, 'XTick', xBars, 'XTickLabel', labels_data, 'XTickLabelRotation', 45);
ylabel('Correlation Difference');
title('Advantage over BIN');
yline(0, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
legend('FontSize', 9, 'Location', 'best');
grid on;
box on;
hold off;

% Plot 3: Absolute (not frame-to-frame) correlations — three-way
nexttile;
abs_bin_data = [];
abs_binan_data = [];
abs_abs_data = [];

for c = 1:length(conditions)
    condition = conditions{c};

    if ~isfield(EntropyPreservation_Results, condition)
        continue;
    end

    for field_idx = 1:3
        threshold_str = sprintf('%dp%d', fix(thresholds(field_idx)), round(mod(thresholds(field_idx), 1)*10));
        field_BIN = sprintf('corr_abs_BIN_%s', threshold_str);
        field_BINAN = sprintf('corr_abs_BINAN_%s', threshold_str);
        field_ABS = sprintf('corr_abs_ABS_%s', threshold_str);

        if isfield(EntropyPreservation_Results.(condition), field_BIN)
            corr_BIN = nanmean(EntropyPreservation_Results.(condition).(field_BIN));
            corr_BINAN = nanmean(EntropyPreservation_Results.(condition).(field_BINAN));
            corr_ABS = nanmean(EntropyPreservation_Results.(condition).(field_ABS));

            abs_bin_data = [abs_bin_data; corr_BIN];
            abs_binan_data = [abs_binan_data; corr_BINAN];
            abs_abs_data = [abs_abs_data; corr_ABS];
        end
    end
end

hold on;
scatter(abs_bin_data, abs_binan_data, 80, [0.3 0.3 0.3], 'o', 'filled', 'MarkerFaceAlpha', 0.5, ...
        'DisplayName', 'BIN vs BIN\_AN');
scatter(abs_bin_data, abs_abs_data, 90, [0 0.5 0], 's', 'filled', 'MarkerFaceAlpha', 0.7, ...
        'DisplayName', 'BIN vs ABS');
plot([0 1], [0 1], 'k--', 'LineWidth', 2, 'HandleVisibility', 'off');
xlabel('BIN Correlation (mean)');
ylabel('BIN\_AN / ABS Correlation (mean)');
title('Absolute Entropy: BIN vs others');
xlim([0 1]);
ylim([0 1]);
axis equal tight;
grid on;
box on;
legend('FontSize', 9, 'Location', 'southeast');
hold off;

% Plot 4: Summary statistics table
nexttile;
axis off;

summary_text = sprintf(['Summary Statistics:\n\n' ...
    'BIN (Recording-level normalization):\n' ...
    '  Delta Corr: mean=%.3f, median=%.3f\n' ...
    '  Abs Corr: mean=%.3f, median=%.3f\n\n' ...
    'BIN_AN (Per-animal normalization):\n' ...
    '  Delta Corr: mean=%.3f, median=%.3f\n' ...
    '  Abs Corr: mean=%.3f, median=%.3f\n\n' ...
    'ABS (Absolute dF/F — Ising cutoff):\n' ...
    '  Delta Corr: mean=%.3f, median=%.3f\n' ...
    '  Abs Corr: mean=%.3f, median=%.3f\n\n' ...
    'ABS minus BIN:\n' ...
    '  Delta: mean=%.3f\n' ...
    '  Abs:   mean=%.3f'], ...
    nanmean(bin_data), nanmedian(bin_data), ...
    nanmean(abs_bin_data), nanmedian(abs_bin_data), ...
    nanmean(binan_data), nanmedian(binan_data), ...
    nanmean(abs_binan_data), nanmedian(abs_binan_data), ...
    nanmean(abs_data_delta), nanmedian(abs_data_delta), ...
    nanmean(abs_abs_data), nanmedian(abs_abs_data), ...
    nanmean(abs_data_delta - bin_data), ...
    nanmean(abs_abs_data - abs_bin_data));

text(0.05, 0.95, summary_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
     'FontSize', 9, 'FontName', 'monospaced', 'BackgroundColor', [0.95 0.95 0.95]);

sgtitle('Comparison of BIN, BIN\_AN, and ABS (Ising cutoff) Normalization Approaches', 'FontWeight', 'bold');

% Plot 8.5b: Same comparison restricted to the 2.0 cutoff (BIN 2σ, BIN_AN 2σ, ABS 2 dF/F)
% ---------------------------------------------------------------
% Focused version of plot 8.5: instead of pooling all three sub-thresholds
% (1.0, 1.5, 2.0), this uses only the 2.0 slot — the cutoff actually used
% by the Ising pipeline — so every point corresponds to one recording at
% the production threshold. Points are coloured per condition so you can
% see whether ABS 2.0 dF/F lies on the diagonal (= matches BIN/BIN_AN σ)
% for every recording individually, not just on aggregate.
fprintf('Creating Plot 8.5b: Focused comparison at threshold = 2.0 only\n');

figure('Name', 'BIN vs BIN_AN vs ABS Comparison - 2.0 cutoff only');
tiledlayout(2, 2);

% Collect per-recording correlations at the 2.0 slot, coloured by condition
bin_delta_2  = [];   binan_delta_2 = [];   abs_delta_2 = [];
bin_absv_2   = [];   binan_absv_2  = [];   abs_absv_2  = [];
cond_idx_rec = [];                           % condition index per recording
n_valid_cond = 0;
valid_conds  = {};

for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(EntropyPreservation_Results, condition)
        continue;
    end

    % Only the 2p0 slot
    f_delta_BIN   = 'corr_delta_BIN_2p0';
    f_delta_BINAN = 'corr_delta_BINAN_2p0';
    f_delta_ABS   = 'corr_delta_ABS_2p0';
    f_absv_BIN    = 'corr_abs_BIN_2p0';
    f_absv_BINAN  = 'corr_abs_BINAN_2p0';
    f_absv_ABS    = 'corr_abs_ABS_2p0';

    if ~isfield(EntropyPreservation_Results.(condition), f_delta_BIN)
        continue;
    end

    bd    = EntropyPreservation_Results.(condition).(f_delta_BIN)(:);
    bnd   = EntropyPreservation_Results.(condition).(f_delta_BINAN)(:);
    ad    = EntropyPreservation_Results.(condition).(f_delta_ABS)(:);
    bav   = EntropyPreservation_Results.(condition).(f_absv_BIN)(:);
    bnav  = EntropyPreservation_Results.(condition).(f_absv_BINAN)(:);
    aav   = EntropyPreservation_Results.(condition).(f_absv_ABS)(:);

    nRec_c = length(bd);
    bin_delta_2    = [bin_delta_2;  bd];
    binan_delta_2  = [binan_delta_2; bnd];
    abs_delta_2    = [abs_delta_2;  ad];
    bin_absv_2     = [bin_absv_2;   bav];
    binan_absv_2   = [binan_absv_2; bnav];
    abs_absv_2     = [abs_absv_2;   aav];
    cond_idx_rec   = [cond_idx_rec; c * ones(nRec_c, 1)];

    n_valid_cond = n_valid_cond + 1;
    valid_conds{end+1} = condition; %#ok<SAGROW>
end

% Panel 1: Frame-to-frame delta — per-recording scatter at 2.0
nexttile;
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    mask = cond_idx_rec == c;
    if ~any(mask)
        continue;
    end
    clr = conditionColors.(condition);
    % BIN vs BIN_AN: open circle
    scatter(bin_delta_2(mask), binan_delta_2(mask), 70, clr, 'o', ...
            'LineWidth', 1.4, 'MarkerFaceAlpha', 0.1, ...
            'DisplayName', sprintf('%s: BIN vs BIN\\_AN', condition));
    % BIN vs ABS: filled square
    scatter(bin_delta_2(mask), abs_delta_2(mask), 75, clr, 's', 'filled', ...
            'MarkerFaceAlpha', 0.85, 'MarkerEdgeColor', [0 0.3 0], ...
            'LineWidth', 1.2, ...
            'DisplayName', sprintf('%s: BIN vs ABS', condition));
end
plot([-0.5 1], [-0.5 1], 'k--', 'LineWidth', 1.5, 'DisplayName', 'Equal performance');
xlabel('BIN 2σ correlation (per recording)');
ylabel('BIN\_AN 2σ / ABS 2 dF/F correlation');
title('Frame-to-Frame Delta at threshold 2.0');
xlim([-0.5 1]); ylim([-0.5 1]);
axis square;
grid on; box on;
legend('FontSize', 7, 'Location', 'southeast');
hold off;

% Panel 2: Per-condition mean differences at threshold 2.0 (bar plot)
nexttile;
hold on;
mean_diff_binan_delta = nan(length(conditions), 1);
mean_diff_abs_delta   = nan(length(conditions), 1);
mean_diff_binan_absv  = nan(length(conditions), 1);
mean_diff_abs_absv    = nan(length(conditions), 1);
for c = 1:length(conditions)
    mask = cond_idx_rec == c;
    if ~any(mask)
        continue;
    end
    mean_diff_binan_delta(c) = nanmean(binan_delta_2(mask) - bin_delta_2(mask));
    mean_diff_abs_delta(c)   = nanmean(abs_delta_2(mask)   - bin_delta_2(mask));
    mean_diff_binan_absv(c)  = nanmean(binan_absv_2(mask)  - bin_absv_2(mask));
    mean_diff_abs_absv(c)    = nanmean(abs_absv_2(mask)    - bin_absv_2(mask));
end
xBars_c = 1:length(conditions);
width_c = 0.38;
bar(xBars_c - width_c/2, mean_diff_binan_delta, width_c, ...
    'FaceColor', [0.2 0.7 0.3], 'FaceAlpha', 0.8, 'DisplayName', 'BIN\_AN − BIN (delta)');
bar(xBars_c + width_c/2, mean_diff_abs_delta, width_c, ...
    'FaceColor', [0 0.5 0], 'FaceAlpha', 0.85, 'DisplayName', 'ABS − BIN (delta)');
yline(0, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
set(gca, 'XTick', xBars_c, 'XTickLabel', conditions);
ylabel('Mean correlation difference');
title('Advantage over BIN 2σ (frame-to-frame delta, at threshold 2.0)');
legend('FontSize', 8, 'Location', 'best');
grid on; box on;
hold off;

% Panel 3: Absolute entropy correlation — per-recording scatter at 2.0
nexttile;
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    mask = cond_idx_rec == c;
    if ~any(mask)
        continue;
    end
    clr = conditionColors.(condition);
    scatter(bin_absv_2(mask), binan_absv_2(mask), 70, clr, 'o', ...
            'LineWidth', 1.4, 'MarkerFaceAlpha', 0.1, ...
            'DisplayName', sprintf('%s: BIN vs BIN\\_AN', condition));
    scatter(bin_absv_2(mask), abs_absv_2(mask), 75, clr, 's', 'filled', ...
            'MarkerFaceAlpha', 0.85, 'MarkerEdgeColor', [0 0.3 0], ...
            'LineWidth', 1.2, ...
            'DisplayName', sprintf('%s: BIN vs ABS', condition));
end
plot([-0.2 1], [-0.2 1], 'k--', 'LineWidth', 1.5, 'DisplayName', 'Equal performance');
xlabel('BIN 2σ correlation (per recording)');
ylabel('BIN\_AN 2σ / ABS 2 dF/F correlation');
title('Absolute Entropy at threshold 2.0');
xlim([-0.2 1]); ylim([-0.2 1]);
axis square;
grid on; box on;
legend('FontSize', 7, 'Location', 'southeast');
hold off;

% Panel 4: Summary stats at threshold 2.0 only
nexttile;
axis off;
summary_text_2 = sprintf(['Threshold = 2.0 only\n' ...
    '(BIN 2σ, BIN_AN 2σ, ABS 2 dF/F)\n' ...
    'n = %d recordings (across %d conditions)\n\n' ...
    'Frame-to-frame delta correlation:\n' ...
    '  BIN 2σ      : mean=%.3f, median=%.3f\n' ...
    '  BIN_AN 2σ   : mean=%.3f, median=%.3f\n' ...
    '  ABS 2 dF/F  : mean=%.3f, median=%.3f\n\n' ...
    'Absolute entropy correlation:\n' ...
    '  BIN 2σ      : mean=%.3f, median=%.3f\n' ...
    '  BIN_AN 2σ   : mean=%.3f, median=%.3f\n' ...
    '  ABS 2 dF/F  : mean=%.3f, median=%.3f\n\n' ...
    'ABS − BIN (paired per recording):\n' ...
    '  Delta : mean=%.4f\n' ...
    '  Abs   : mean=%.4f'], ...
    length(bin_delta_2), n_valid_cond, ...
    nanmean(bin_delta_2),   nanmedian(bin_delta_2), ...
    nanmean(binan_delta_2), nanmedian(binan_delta_2), ...
    nanmean(abs_delta_2),   nanmedian(abs_delta_2), ...
    nanmean(bin_absv_2),    nanmedian(bin_absv_2), ...
    nanmean(binan_absv_2),  nanmedian(binan_absv_2), ...
    nanmean(abs_absv_2),    nanmedian(abs_absv_2), ...
    nanmean(abs_delta_2 - bin_delta_2), ...
    nanmean(abs_absv_2  - bin_absv_2));

text(0.05, 0.95, summary_text_2, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
     'FontSize', 9, 'FontName', 'monospaced', 'BackgroundColor', [0.95 0.95 0.95]);

sgtitle('BIN 2σ vs BIN\_AN 2σ vs ABS 2 dF/F (Ising cutoff) — per-recording, threshold 2 only', ...
        'FontWeight', 'bold');

fprintf('\n=== Step 4 complete: Visualizations created ===\n');

end  % RUN.section8_plots

end  % RUN.section8

%% =========================================================================
%% Step 5: Save Results
%% =========================================================================
if RUN.save_figures

fprintf('\n=== Step 5: Saving results ===\n');

% Define save path
savePathEntropy = 'Fig. 5 Model\ThresholdAnalysis\thres\';

% Create directory if it doesn't exist
if ~exist(savePathEntropy, 'dir')
    mkdir(savePathEntropy);
end

% Save all figures
fprintf('Saving figures to: %s\n', savePathEntropy);
saveMyFig("Fig5_thresholdForBinarisationDetermination", savePathEntropy, 'All');

% Save results structure to .mat file (only if we have data to save)
if exist('EntropyPreservation_Results', 'var')
    saveFileResults = fullfile(savePathEntropy, 'EntropyPreservation_Results.mat');
    save(saveFileResults, 'EntropyPreservation_Results', 'entropy_thresholds', ...
         'conditions', 'Skip');
    fprintf('Results saved to: %s\n', saveFileResults);
else
    fprintf('(EntropyPreservation_Results not present — skipping .mat save)\n');
end

end  % RUN.save_figures

fprintf('\n=== Section 8 complete: Raw vs Binarized Entropy Comparison Analysis Finished ===\n');

%% =========================================================================
%% SECTION 9: NMF Components vs Threshold-2 Binarization
%% =========================================================================
if RUN.section9

fprintf('\n--- Section 9: NMF vs Threshold-2 Comparison ---\n');

% Load entropy colourmap (used throughout the codebase for spatial images)
if ~exist('EntropyColourmap', 'var')
    load('EntropyColourMap', 'EntropyColourmap');
end
% Binary colourmap: active = black, inactive = white (flipped gray)
binary_cmap = [1 1 1; 0 0 0];

% Load NMF data if not already in workspace
if ~exist('NMF_littleRegularisation_Grid40_AllConditions', 'var')
    fprintf('Loading NMF data from <DATA_ROOT>/NMF_littleRegularisation_Grid40_AllConditions.mat...\n');
    load(mba_p('NMF_littleRegularisation_Grid40_AllConditions.mat'), ...
         'NMF_littleRegularisation_Grid40_AllConditions');
end
NMF_sec9 = NMF_littleRegularisation_Grid40_AllConditions;

% Use mean-subtracted variant (Treatment 2: "stim subtracted")
W_struct_sec9 = NMF_sec9.W_all2;
H_struct_sec9 = NMF_sec9.H_all2;

% Exemplar list: {condition, recording, trial}
nmf_exemplars = {
    'Expert',   5,  1;   % Expert Rec 5, Trial 1
    'Expert',   5,  5;   % Expert Rec 5, Trial 5
    'Expert',   2,  1;   % Expert Rec 2, Trial 1
    'Expert',   3,  1;   % Expert Rec 3, Trial 1
    'Expert',   6,  1;   % Expert Rec 6, Trial 1
    'Expert',   5, 10;   % Expert Rec 5, Trial 10 (different trial)
    'Beginner', 5,  1;   % Beginner Rec 5, Trial 1 (different condition)
};

gridDim_sec9 = [13, 26];

% Toggle: 'components' plots individual W·H per component (coloured lines)
%         'sum'        plots the full reconstruction sum(W*H) as a single trace
NMF_TRACE_MODE = 'components';  % Options: 'components' or 'sum'

for ei = 1:size(nmf_exemplars, 1)
    nmf_cond  = nmf_exemplars{ei, 1};
    nmf_rec   = nmf_exemplars{ei, 2};
    nmf_trial = nmf_exemplars{ei, 3};
    condIndiv = [nmf_cond 'Individual'];

    fprintf('  Processing %s Rec %d Trial %d\n', nmf_cond, nmf_rec, nmf_trial);

    % --- Access NMF W and H for this trial ---
    W_rec_cell = W_struct_sec9.(condIndiv).P1;
    H_rec_cell = H_struct_sec9.(condIndiv).P1;

    % Handle cell indexing ambiguity (N×1 vs 1×N)
    if size(W_rec_cell, 1) >= nmf_rec
        W_rec = W_rec_cell{nmf_rec};
        H_rec = H_rec_cell{nmf_rec};
    elseif size(W_rec_cell, 2) >= nmf_rec
        W_rec = W_rec_cell{1, nmf_rec};
        H_rec = H_rec_cell{1, nmf_rec};
    else
        fprintf('    Warning: recording %d out of range for %s.P1. Skipping.\n', nmf_rec, condIndiv);
        continue;
    end

    % Access the trial
    if size(W_rec, 2) < nmf_trial
        fprintf('    Warning: trial %d out of range (max %d). Skipping.\n', nmf_trial, size(W_rec, 2));
        continue;
    end
    W_trial = W_rec{1, nmf_trial};   % [338 × nComponents]
    H_trial = H_rec{1, nmf_trial};   % [nComponents × 185]

    nComp = size(W_trial, 2);
    nT_nmf = size(H_trial, 2);

    fprintf('    W: [%d × %d], H: [%d × %d]\n', ...
            size(W_trial, 1), size(W_trial, 2), size(H_trial, 1), size(H_trial, 2));

    % --- Access raw Grid40 for the same trial ---
    if ~isfield(Grid40, condIndiv) || ...
       length(Grid40.(condIndiv).AllNeurons) < nmf_rec || ...
       ~isfield(Grid40.(condIndiv).AllNeurons(nmf_rec), 'P1')
        fprintf('    Warning: Grid40.%s.AllNeurons(%d).P1 not found. Skipping.\n', condIndiv, nmf_rec);
        continue;
    end
    gridData_cell_s9 = Grid40.(condIndiv).AllNeurons(nmf_rec).P1;
    if iscell(gridData_cell_s9) && ~isempty(gridData_cell_s9)
        gridData_full_s9 = gridData_cell_s9{1};
    else
        gridData_full_s9 = gridData_cell_s9;
    end

    if size(gridData_full_s9, 4) < nmf_trial
        fprintf('    Warning: Grid40 has only %d trials, need %d. Skipping.\n', ...
                size(gridData_full_s9, 4), nmf_trial);
        continue;
    end
    gridData_trial_s9 = gridData_full_s9(:, :, :, nmf_trial);  % [13 × 26 × 185]

    % --- Compute threshold-2 binary for each frame ---
    binary_trial_s9 = gridData_trial_s9 > 2.0;  % [13 × 26 × 185] logical
    frac_active_s9 = squeeze(mean(mean(binary_trial_s9, 1), 2)) * 100;  % [185 × 1] %

    % --- Find peak activity frame (max fraction active) ---
    [~, peak_frame] = max(frac_active_s9);

    % --- Rank components by peak H amplitude ---
    [~, comp_order] = sort(max(H_trial, [], 2), 'descend');

    nShowComp = min(5, nComp);

    % --- Create the figure ---
    figure('Name', sprintf('NMF vs Threshold-2 - %s Rec %d Trial %d', ...
                           nmf_cond, nmf_rec, nmf_trial));
    tiledlayout(3, 5);

    % Row 1: Top-5 NMF spatial components (W), ranked by peak temporal amplitude
    for ci = 1:nShowComp
        comp_idx = comp_order(ci);
        nexttile;
        imagesc(reshape(W_trial(:, comp_idx), gridDim_sec9));
        colormap(gca, EntropyColourmap);
        axis equal tight;
        set(gca, 'YDir', 'normal');
        title(sprintf('W comp %d', comp_idx));
        colorbar;
    end
    % Fill remaining tiles if fewer than 5
    for ci = (nShowComp+1):5
        nexttile; axis off;
    end

    % Row 2: NMF temporal traces + fraction-active overlay (span 5 tiles)
    nexttile(6, [1, 5]);
    hold on;
    if strcmp(NMF_TRACE_MODE, 'components')
        % Individual per-component reconstruction traces
        cmap_comp = lines(nComp);
        for ci = 1:nComp
            recon_ci = mean(W_trial(:, ci) * H_trial(ci, :), 1);  % [1 × nT]
            plot(1:nT_nmf, recon_ci, 'Color', [cmap_comp(ci, :), 0.5], ...
                 'LineWidth', 1.2);
        end
        trace_label = 'Mean W·H per component';
        trace_title = 'NMF components (W·H)';
    else
        % Sum reconstruction: mean across cells of W*H
        full_recon_trace = mean(W_trial * H_trial, 1);  % [1 × nT]
        plot(1:nT_nmf, full_recon_trace, 'Color', [0.2 0.2 0.8], ...
             'LineWidth', 2.5);
        trace_label = 'Mean reconstruction (W·H sum)';
        trace_title = 'NMF reconstruction sum (W·H)';
    end
    % Overlay fraction active on right y-axis
    yyaxis right;
    plot(1:nT_nmf, frac_active_s9, '--', 'Color', [0 0.5 0], ...
         'LineWidth', 2.5, 'DisplayName', '% active at 2.0 dF/F');
    ylabel('% grid cells active');
    yyaxis left;
    ylabel(trace_label);
    xlabel('Timepoint (frames)');
    % Mark stimulus onset
    xline(81, ':', 'stim', 'HandleVisibility', 'off');
    % Mark peak frame
    xline(peak_frame, '-', 'peak', 'Color', [0.7 0 0], 'LineWidth', 1.2, ...
          'HandleVisibility', 'off');
    title(sprintf('%s + fraction active at 2.0 dF/F (green dashed)', trace_title));
    grid on; box on; hold off;

    % Row 3: Spatial comparison at peak frame
    % Tile 1: Raw Grid40 at peak frame
    nexttile;
    imagesc(gridData_trial_s9(:, :, peak_frame));
    colormap(gca, EntropyColourmap);
    clim([0, prctile(gridData_trial_s9(:), 99)]);
    axis equal tight; set(gca, 'YDir', 'normal');
    title(sprintf('Raw (frame %d)', peak_frame));
    colorbar;

    % Tile 2: NMF reconstruction at peak frame
    nexttile;
    recon_s9 = reshape(W_trial * H_trial(:, peak_frame), gridDim_sec9);
    imagesc(recon_s9);
    colormap(gca, EntropyColourmap);
    axis equal tight; set(gca, 'YDir', 'normal');
    title('NMF reconstruction');
    colorbar;

    % Tile 3: Binary mask at threshold=2 (active=black, inactive=white)
    nexttile;
    imagesc(binary_trial_s9(:, :, peak_frame));
    colormap(gca, binary_cmap);
    axis equal tight; set(gca, 'YDir', 'normal');
    title('Binary (>2.0 dF/F)', 'Color', [0 0.5 0], 'FontWeight', 'bold');
    colorbar;

    % Tile 4: Raw Grid40 with both binary (green) and NMF (red) contours
    nexttile;
    imagesc(gridData_trial_s9(:, :, peak_frame));
    colormap(gca, EntropyColourmap);
    clim([0, prctile(gridData_trial_s9(:), 99)]);
    hold on;
    % Green contours: threshold-2 binary boundary
    contour(double(binary_trial_s9(:, :, peak_frame)), [0.5 0.5], ...
            'LineColor', [0 0.8 0], 'LineWidth', 2.0);
    % Red contours: NMF reconstruction
    contour(recon_s9, 3, 'LineColor', 'r', 'LineWidth', 1.5);
    hold off;
    axis equal tight; set(gca, 'YDir', 'normal');
    title('Raw + contours (green=2dF/F, red=NMF)');

    % Tile 5: Summary text
    nexttile;
    axis off;
    text(0.05, 0.95, {
        sprintf('%s Rec %d Trial %d', nmf_cond, nmf_rec, nmf_trial);
        '';
        sprintf('Peak frame: %d', peak_frame);
        sprintf('Frac active at peak: %.1f%%', frac_active_s9(peak_frame));
        sprintf('NMF components: %d', nComp);
        'Preprocessing: Mean-subtracted';
        '';
        'Green dashed = threshold=2 dF/F';
        'Red contours = NMF reconstruction';
        }, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
         'FontSize', 9, 'FontName', 'monospaced');

    sgtitle(sprintf('NMF Decomposition vs Threshold-2 Binarization: %s Rec %d Trial %d', ...
                    nmf_cond, nmf_rec, nmf_trial), 'FontWeight', 'bold');

    fprintf('  Figure created for %s Rec %d Trial %d\n', nmf_cond, nmf_rec, nmf_trial);
end

%% Section 9b: Search ALL Expert trials for best NMF–binary overlap, plot top 3
% -----------------------------------------------------------------------
% For every valid Expert recording × trial, compute a spatial overlap score
% (Dice coefficient) between the NMF reconstruction and the threshold-2
% binary mask at the peak-activity frame. Then pick the 3 trials with the
% highest overlap and generate the same 3×5 figure for each.

fprintf('\n--- Section 9b: Searching Expert trials for best NMF–binary overlap ---\n');

search_condition = 'Expert';
condIndiv_search = [search_condition 'Individual'];
nTopTrials = 10;

% Collect overlap scores across all recordings × trials
overlap_table = [];  % [recording, trial, dice_score, peak_frame]
% Also collect Dice at multiple thresholds for distribution comparison
dice_thresholds = [0.5, 1.0, 2.0, 3.0, 5.0];
dice_multi = [];  % [recording, trial, dice_0p5, dice_1p0, dice_2p0, dice_3p0, dice_5p0]

W_search = W_struct_sec9.(condIndiv_search).P1;
H_search = H_struct_sec9.(condIndiv_search).P1;
nRecs_search = max(size(W_search));

for r = 1:nRecs_search
    % Skip if in Skip list
    if ismember(r, Skip.(search_condition))
        continue;
    end

    % Access NMF data for this recording
    try
        if size(W_search, 1) >= r
            W_rec_s = W_search{r};
            H_rec_s = H_search{r};
        else
            W_rec_s = W_search{1, r};
            H_rec_s = H_search{1, r};
        end
    catch
        continue;
    end

    if isempty(W_rec_s)
        continue;
    end

    nTrials_s = size(W_rec_s, 2);

    % Access raw Grid40 for this recording
    if ~isfield(Grid40.(condIndiv_search).AllNeurons(r), 'P1')
        continue;
    end
    gd_cell_s = Grid40.(condIndiv_search).AllNeurons(r).P1;
    if iscell(gd_cell_s) && ~isempty(gd_cell_s)
        gd_full_s = gd_cell_s{1};
    else
        gd_full_s = gd_cell_s;
    end
    if isempty(gd_full_s)
        continue;
    end

    nTrials_grid = size(gd_full_s, 4);
    nTrials_check = min(nTrials_s, nTrials_grid);

    for t = 1:nTrials_check
        try
            W_t = W_rec_s{1, t};
            H_t = H_rec_s{1, t};
        catch
            continue;
        end
        if isempty(W_t) || isempty(H_t)
            continue;
        end

        gd_trial = gd_full_s(:, :, :, t);  % [13 × 26 × 185]

        % Binary mask at threshold=2 per frame (used for peak detection)
        bin_trial = gd_trial > 2.0;
        frac_act = squeeze(mean(mean(bin_trial, 1), 2));
        [~, pk] = max(frac_act);

        % NMF reconstruction at peak frame
        recon_pk = reshape(W_t * H_t(:, pk), gridDim_sec9);

        % Binarise NMF reconstruction at its median to get an "active" mask
        nmf_mask = recon_pk > median(recon_pk(:)) + std(recon_pk(:));

        % Binary mask at peak frame (threshold=2)
        bin_mask = bin_trial(:, :, pk);

        % Dice coefficient at threshold=2: 2 * |A ∩ B| / (|A| + |B|)
        intersection = sum(nmf_mask(:) & bin_mask(:));
        dice = 2 * intersection / (sum(nmf_mask(:)) + sum(bin_mask(:)) + eps);

        overlap_table = [overlap_table; r, t, dice, pk]; %#ok<AGROW>

        % Dice at multiple thresholds (for distribution comparison)
        dice_row = [r, t];
        for di = 1:length(dice_thresholds)
            bin_mask_di = gd_trial(:, :, pk) > dice_thresholds(di);
            inter_di = sum(nmf_mask(:) & bin_mask_di(:));
            dice_di = 2 * inter_di / (sum(nmf_mask(:)) + sum(bin_mask_di(:)) + eps);
            dice_row = [dice_row, dice_di]; %#ok<AGROW>
        end
        dice_multi = [dice_multi; dice_row]; %#ok<AGROW>
    end

    if mod(r, 5) == 0
        fprintf('    Scanned Rec %d/%d (%d trials so far)\n', r, nRecs_search, size(overlap_table, 1));
    end
end

fprintf('  Scanned %d trials across %d recordings\n', size(overlap_table, 1), nRecs_search);

if ~isempty(overlap_table)
    all_dice = overlap_table(:, 3);
    fprintf('\n  Overall Dice coefficient statistics (NMF vs threshold-2 overlap):\n');
    fprintf('    Mean:   %.4f\n', mean(all_dice));
    fprintf('    Median: %.4f\n', median(all_dice));
    fprintf('    Std:    %.4f\n', std(all_dice));
    fprintf('    Min:    %.4f\n', min(all_dice));
    fprintf('    Max:    %.4f\n', max(all_dice));
    fprintf('    Q25:    %.4f\n', quantile(all_dice, 0.25));
    fprintf('    Q75:    %.4f\n', quantile(all_dice, 0.75));
    fprintf('    n:      %d trials\n', length(all_dice));

    % Sort by Dice score descending
    [~, sort_idx] = sort(overlap_table(:, 3), 'descend');
    overlap_sorted = overlap_table(sort_idx, :);

    % Print top results
    fprintf('\n  Top %d NMF–binary overlap trials (Dice coefficient):\n', min(nTopTrials, size(overlap_sorted, 1)));
    for k = 1:min(10, size(overlap_sorted, 1))
        fprintf('    #%d: Rec %d Trial %d — Dice = %.4f (peak frame %d)\n', ...
                k, overlap_sorted(k, 1), overlap_sorted(k, 2), ...
                overlap_sorted(k, 3), overlap_sorted(k, 4));
    end

    % Plot the top nTopTrials
    nPlot = min(nTopTrials, size(overlap_sorted, 1));
    for ki = 1:nPlot
        best_rec   = overlap_sorted(ki, 1);
        best_trial = overlap_sorted(ki, 2);
        best_dice  = overlap_sorted(ki, 3);
        best_pk    = overlap_sorted(ki, 4);

        fprintf('\n  Plotting top-%d: Rec %d Trial %d (Dice=%.4f)\n', ki, best_rec, best_trial, best_dice);

        % Access NMF data
        if size(W_search, 1) >= best_rec
            W_rec_best = W_search{best_rec};
            H_rec_best = H_search{best_rec};
        else
            W_rec_best = W_search{1, best_rec};
            H_rec_best = H_search{1, best_rec};
        end
        W_best = W_rec_best{1, best_trial};
        H_best = H_rec_best{1, best_trial};

        nComp_best = size(W_best, 2);
        nT_best = size(H_best, 2);

        % Access raw Grid40
        gd_cell_best = Grid40.(condIndiv_search).AllNeurons(best_rec).P1;
        if iscell(gd_cell_best) && ~isempty(gd_cell_best)
            gd_full_best = gd_cell_best{1};
        else
            gd_full_best = gd_cell_best;
        end
        gd_trial_best = gd_full_best(:, :, :, best_trial);

        bin_trial_best = gd_trial_best > 2.0;
        frac_act_best = squeeze(mean(mean(bin_trial_best, 1), 2)) * 100;

        [~, comp_order_best] = sort(max(H_best, [], 2), 'descend');
        nShowComp_best = min(5, nComp_best);

        % --- Create the figure (same layout as the exemplar plots) ---
        figure('Name', sprintf('NMF–Binary Overlap - Expert Rec %d Trial %d (Dice=%.3f)', ...
                               best_rec, best_trial, best_dice));
        tiledlayout(3, 5);

        % Row 1: Top-5 spatial components
        for ci = 1:nShowComp_best
            cidx = comp_order_best(ci);
            nexttile;
            imagesc(reshape(W_best(:, cidx), gridDim_sec9));
            colormap(gca, EntropyColourmap);
            axis equal tight; set(gca, 'YDir', 'normal');
            title(sprintf('W comp %d', cidx));
            colorbar;
        end
        for ci = (nShowComp_best+1):5
            nexttile; axis off;
        end

        % Row 2: NMF traces + fraction-active overlay
        nexttile(6, [1, 5]);
        hold on;
        if strcmp(NMF_TRACE_MODE, 'components')
            cmap_best = lines(nComp_best);
            for ci = 1:nComp_best
                recon_ci_best = mean(W_best(:, ci) * H_best(ci, :), 1);
                plot(1:nT_best, recon_ci_best, 'Color', [cmap_best(ci, :), 0.5], 'LineWidth', 1.2);
            end
            trace_label_b = 'Mean W·H per component';
            trace_title_b = 'NMF components (W·H)';
        else
            full_recon_best = mean(W_best * H_best, 1);
            plot(1:nT_best, full_recon_best, 'Color', [0.2 0.2 0.8], 'LineWidth', 2.5);
            trace_label_b = 'Mean reconstruction (W·H sum)';
            trace_title_b = 'NMF reconstruction sum (W·H)';
        end
        yyaxis right;
        plot(1:nT_best, frac_act_best, '--', 'Color', [0 0.5 0], 'LineWidth', 2.5);
        ylabel('% active');
        yyaxis left;
        ylabel(trace_label_b);
        xlabel('Timepoint (frames)');
        xline(81, ':', 'stim', 'HandleVisibility', 'off');
        xline(best_pk, '-', 'peak', 'Color', [0.7 0 0], 'LineWidth', 1.2, 'HandleVisibility', 'off');
        title(sprintf('%s + %% active at 2.0 dF/F (Dice=%.3f at frame %d)', trace_title_b, best_dice, best_pk));
        grid on; box on; hold off;

        % Row 3: Spatial comparison at peak frame
        nexttile;
        imagesc(gd_trial_best(:, :, best_pk));
        colormap(gca, EntropyColourmap);
        clim([0, prctile(gd_trial_best(:), 99)]);
        axis equal tight; set(gca, 'YDir', 'normal');
        title(sprintf('Raw (frame %d)', best_pk));
        colorbar;

        nexttile;
        recon_best = reshape(W_best * H_best(:, best_pk), gridDim_sec9);
        imagesc(recon_best);
        colormap(gca, EntropyColourmap);
        axis equal tight; set(gca, 'YDir', 'normal');
        title('NMF reconstruction');
        colorbar;

        nexttile;
        imagesc(bin_trial_best(:, :, best_pk));
        colormap(gca, binary_cmap);
        axis equal tight; set(gca, 'YDir', 'normal');
        title('Binary (>2.0 dF/F)', 'Color', [0 0.5 0], 'FontWeight', 'bold');
        colorbar;

        nexttile;
        imagesc(gd_trial_best(:, :, best_pk));
        colormap(gca, EntropyColourmap);
        clim([0, prctile(gd_trial_best(:), 99)]);
        hold on;
        contour(double(bin_trial_best(:, :, best_pk)), [0.5 0.5], ...
                'LineColor', [0 0.8 0], 'LineWidth', 2.0);
        contour(recon_best, 3, 'LineColor', 'r', 'LineWidth', 1.5);
        hold off;
        axis equal tight; set(gca, 'YDir', 'normal');
        title('Raw + contours (green=2dF/F, red=NMF)');

        nexttile;
        axis off;
        text(0.05, 0.95, {
            sprintf('Expert Rec %d Trial %d', best_rec, best_trial);
            '';
            sprintf('Dice coefficient: %.4f', best_dice);
            sprintf('Frac active: %.1f%%', frac_act_best(best_pk));
            '';
            'Green = threshold 2.0 dF/F';
            'Red = NMF reconstruction';
            }, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
             'FontSize', 9, 'FontName', 'monospaced');

        sgtitle(sprintf('NMF–Binary Overlap: Expert Rec %d Trial %d (Dice=%.3f)', ...
                        best_rec, best_trial, best_dice), 'FontWeight', 'bold');
    end
else
    fprintf('  No valid trials found for overlap analysis.\n');
end

%% Section 9c: W*H reconstruction vs fraction-active at 3 cutoffs (0.5, 2, 5)
% -----------------------------------------------------------------------
% One figure with 3 tiles showing how the NMF reconstruction trace compares
% to fraction-active at different absolute thresholds. Uses Expert Rec 5 Trial 1.

fprintf('\n--- Section 9c: W*H vs fraction-active at cutoffs 0.5, 1.0, 2.0, 5.0 ---\n');

cutoff_values = [0.5, 1.0, 2.0, 3.0, 5.0];
cutoff_colors = {[0.6 0.6 0.6], [0.2 0.4 0.8], [0 0.5 0], [0.8 0.5 0], [0.8 0 0]};
cutoff_rec = 5;
cutoff_trial = 1;
cutoff_condIndiv = 'ExpertIndividual';

% Access NMF data
try
    if size(W_struct_sec9.(cutoff_condIndiv).P1, 1) >= cutoff_rec
        W_cut = W_struct_sec9.(cutoff_condIndiv).P1{cutoff_rec};
        H_cut = H_struct_sec9.(cutoff_condIndiv).P1{cutoff_rec};
    else
        W_cut = W_struct_sec9.(cutoff_condIndiv).P1{1, cutoff_rec};
        H_cut = H_struct_sec9.(cutoff_condIndiv).P1{1, cutoff_rec};
    end
    W_cut_trial = W_cut{1, cutoff_trial};
    H_cut_trial = H_cut{1, cutoff_trial};

    % Raw Grid40
    gd_cut_cell = Grid40.(cutoff_condIndiv).AllNeurons(cutoff_rec).P1;
    if iscell(gd_cut_cell) && ~isempty(gd_cut_cell)
        gd_cut_full = gd_cut_cell{1};
    else
        gd_cut_full = gd_cut_cell;
    end
    gd_cut_trial = gd_cut_full(:, :, :, cutoff_trial);  % [13 × 26 × 185]

    % Full NMF reconstruction trace (mean across spatial cells)
    recon_trace = mean(W_cut_trial * H_cut_trial, 1);  % [1 × nT]
    nT_cut = length(recon_trace);

    figure('Name', 'NMF Reconstruction vs Fraction-Active at 5 Cutoffs');
    tiledlayout(1, 5);

    for ci_cut = 1:length(cutoff_values)
        cutoff_v = cutoff_values(ci_cut);
        bin_cut = gd_cut_trial > cutoff_v;
        frac_cut = squeeze(mean(mean(bin_cut, 1), 2)) * 100;

        nexttile;
        yyaxis left;
        plot(1:nT_cut, recon_trace, 'Color', [0.2 0.2 0.8], 'LineWidth', 2.2);
        ylabel('Mean W·H (reconstruction)');
        yyaxis right;
        plot(1:nT_cut, frac_cut, '--', 'Color', cutoff_colors{ci_cut}, 'LineWidth', 2.2);
        ylabel('% active');

        xlabel('Timepoint (frames)');
        xline(81, ':', 'stim', 'HandleVisibility', 'off');
        title(sprintf('Cutoff = %.1f dF/F (%.1f%% mean active)', cutoff_v, mean(frac_cut)));

        % Compute correlation between the two traces
        valid_idx = ~isnan(recon_trace') & ~isnan(frac_cut);
        if sum(valid_idx) > 2 && std(recon_trace(valid_idx)) > 0 && std(frac_cut(valid_idx)) > 0
            r_corr = corr(recon_trace(valid_idx)', frac_cut(valid_idx));
            text(0.05, 0.95, sprintf('r = %.3f', r_corr), 'Units', 'normalized', ...
                 'VerticalAlignment', 'top', 'FontWeight', 'bold', 'FontSize', 11);
        end

        grid on; box on;
    end

    sgtitle('NMF Reconstruction (W·H) vs Fraction-Active: Expert Rec 5 Trial 1', 'FontWeight', 'bold');
    fprintf('  3-cutoff comparison figure created\n');

catch ME_cut
    fprintf('  Warning: could not create cutoff comparison figure: %s\n', ME_cut.message);
end

%% Section 9d: Dice coefficient distributions at 4 cutoffs
% -----------------------------------------------------------------------
% Box plot comparing the Dice overlap distributions across all
% Expert trials at thresholds 0.5, 1.0, 2.0, and 5.0 dF/F.

fprintf('\n--- Section 9d: Dice distributions at cutoffs 0.5, 1.0, 2.0, 5.0 ---\n');

if ~isempty(dice_multi)
    nDiceThresh = length(dice_thresholds);  % should be 4
    dice_by_thresh = cell(nDiceThresh, 1);
    for di = 1:nDiceThresh
        dice_by_thresh{di} = dice_multi(:, 2 + di);  % columns 3, 4, 5, 6
    end

    dice_thresh_labels = arrayfun(@(x) sprintf('%.1f dF/F', x), dice_thresholds, 'UniformOutput', false);
    dice_thresh_colors = [0.6 0.6 0.6; 0.2 0.4 0.8; 0 0.5 0; 0.8 0 0];

    figure('Name', sprintf('Dice Coefficient Distributions at %d Cutoffs', nDiceThresh));
    tiledlayout(1, 2);

    % Tile 1: Box plot comparison
    nexttile;
    hold on;
    all_dice_vals = vertcat(dice_by_thresh{:});
    group_labels = [];
    for di = 1:nDiceThresh
        group_labels = [group_labels; di * ones(length(dice_by_thresh{di}), 1)]; %#ok<AGROW>
    end
    boxplot(all_dice_vals, group_labels, 'Labels', dice_thresh_labels, ...
            'Colors', dice_thresh_colors, 'Widths', 0.5);
    % Overlay individual data points with jitter
    for gi = 1:nDiceThresh
        mask_g = group_labels == gi;
        x_jitter = gi + (rand(sum(mask_g), 1) - 0.5) * 0.25;
        scatter(x_jitter, all_dice_vals(mask_g), 20, 'k', 'filled', ...
                'MarkerFaceAlpha', 0.3, 'HandleVisibility', 'off');
    end
    ylabel('Dice coefficient');
    title('NMF–Binary Overlap by Threshold');
    fprintf('\n  Dice distributions (n=%d trials):\n', length(dice_by_thresh{1}));
    for di = 1:nDiceThresh
        fprintf('    %s: mean=%.4f, median=%.4f, std=%.4f\n', ...
                dice_thresh_labels{di}, mean(dice_by_thresh{di}), ...
                median(dice_by_thresh{di}), std(dice_by_thresh{di}));
    end
    grid on; box on;
    hold off;

    % Tile 2: Paired comparison (each trial = one line connecting the thresholds)
    nexttile;
    hold on;
    nTrials_dice = size(dice_multi, 1);
    x_positions = 1:nDiceThresh;
    % Plot thin lines connecting each trial's Dice values
    for ti_d = 1:min(nTrials_dice, 200)  % cap at 200 for readability
        y_vals = zeros(1, nDiceThresh);
        for di = 1:nDiceThresh
            y_vals(di) = dice_by_thresh{di}(ti_d);
        end
        plot(x_positions, y_vals, '-', 'Color', [0.7 0.7 0.7 0.15], 'LineWidth', 0.5);
    end
    % Plot means
    mean_dice = cellfun(@mean, dice_by_thresh);
    plot(x_positions, mean_dice, 'ko-', 'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    % Highlight the 2.0 dF/F point (find its index)
    idx_2 = find(dice_thresholds == 2.0, 1);
    if ~isempty(idx_2)
        plot(idx_2, mean_dice(idx_2), 's', 'Color', [0 0.5 0], 'MarkerSize', 14, ...
             'LineWidth', 2.5, 'MarkerFaceColor', [0 0.5 0]);
    end

    set(gca, 'XTick', x_positions, 'XTickLabel', dice_thresh_labels);
    ylabel('Dice coefficient');
    title('Paired: each line = one trial');
    xlim([0.5, nDiceThresh + 0.5]);
    grid on; box on;
    hold off;

    sgtitle(sprintf('Dice Coefficient: NMF vs Binary Overlap at %d Thresholds (n=%d Expert trials)', ...
                    nDiceThresh, nTrials_dice), 'FontWeight', 'bold');
else
    fprintf('  No Dice data available for distribution plot.\n');
end

%% Section 9e: Paired inferential tests (Threshold 2.0 vs alternatives)
% -----------------------------------------------------------------------
% Each Expert trial appears at all five thresholds, so the comparisons are
% paired. We run a Friedman omnibus across the five thresholds, then per-
% pair Wilcoxon signed-rank tests of 2.0 vs each alternative, with Holm-
% Bonferroni adjustment over the four pairwise comparisons. Paired Hedges'
% g (mean of within-trial differences / sd of differences, bias-corrected)
% accompanies each pair. The per-trial Dice matrix and the test results are
% saved for the ED Figure 8 stats pipeline.

if ~isempty(dice_multi)
    fprintf('\n--- Section 9e: Paired tests (Threshold 2.0 vs alternatives) ---\n');

    idx_thresh_2 = find(dice_thresholds == 2.0, 1);
    if isempty(idx_thresh_2)
        warning('Threshold 2.0 not present in dice_thresholds — skipping paired tests.');
    else
        % Build trials × thresholds matrix; drop trials with any NaN so the
        % Friedman test and the per-pair signed-rank tests see the same n.
        dice_matrix = dice_multi(:, 3:(2 + nDiceThresh));   % [trials × 5]
        valid_rows  = all(~isnan(dice_matrix), 2);
        D           = dice_matrix(valid_rows, :);
        n_paired    = size(D, 1);

        % Omnibus Friedman test across all 5 thresholds
        [p_friedman, friedman_tbl, friedman_stats] = friedman(D, 1, 'off');
        chi2_friedman = friedman_tbl{2, 5};
        df_friedman   = friedman_tbl{2, 3};
        fprintf('  Friedman omnibus: chi2(%d) = %.3f, p = %.3g (n = %d trials × %d thresholds)\n', ...
                df_friedman, chi2_friedman, p_friedman, n_paired, nDiceThresh);

        % Per-pair Wilcoxon signed-rank: 2.0 vs each alternative
        ref_vec = D(:, idx_thresh_2);
        alt_idx = setdiff(1:nDiceThresh, idx_thresh_2);
        nPairs  = numel(alt_idx);
        pair_p_raw    = nan(nPairs, 1);
        pair_signrank = nan(nPairs, 1);
        pair_g        = nan(nPairs, 1);
        pair_median_diff = nan(nPairs, 1);
        pair_label = cell(nPairs, 1);
        for k = 1:nPairs
            alt = D(:, alt_idx(k));
            d   = ref_vec - alt;
            [p_sr, ~, st] = signrank(ref_vec, alt);
            pair_p_raw(k)    = p_sr;
            pair_signrank(k) = st.signedrank;
            pair_median_diff(k) = median(d);
            % Paired Hedges' g (bias-corrected dz on within-trial differences)
            d_mean = mean(d);
            d_sd   = std(d);
            if d_sd == 0
                gz = 0;
            else
                dz = d_mean / d_sd;
                Jc = 1 - 3 / (4 * (n_paired - 1) - 1);   % bias correction
                gz = Jc * dz;
            end
            pair_g(k) = gz;
            pair_label{k} = sprintf('2.0 vs %.1f dF/F', dice_thresholds(alt_idx(k)));
        end

        % Holm-Bonferroni adjustment across the 4 pairwise comparisons
        [p_sorted, sort_order] = sort(pair_p_raw);
        m = numel(p_sorted);
        p_holm_sorted = nan(m, 1);
        running_max = 0;
        for k = 1:m
            adj = (m - k + 1) * p_sorted(k);
            running_max = max(running_max, adj);
            p_holm_sorted(k) = min(running_max, 1);
        end
        pair_p_holm = nan(nPairs, 1);
        pair_p_holm(sort_order) = p_holm_sorted;

        % Console report
        fprintf('\n  %-22s %8s %14s %12s %12s %10s\n', ...
            'Comparison', 'n', 'median diff', 'p (uncorr)', 'p (Holm)', "Hedges' g");
        for k = 1:nPairs
            fprintf('  %-22s %8d %14.4f %12.3g %12.3g %10.3f\n', ...
                    pair_label{k}, n_paired, pair_median_diff(k), ...
                    pair_p_raw(k), pair_p_holm(k), pair_g(k));
        end

        % Build a results table for export
        T_dice = table(pair_label, repmat(n_paired, nPairs, 1), ...
                       pair_median_diff, pair_signrank, pair_p_raw, pair_p_holm, pair_g, ...
                       'VariableNames', {'comparison','n_paired','median_diff', ...
                                         'W_signedrank','p_uncorrected','p_holm','hedges_g'});

        % Save dice_multi + test results to .mat next to the figure outputs
        diceStatsPath = fullfile(savePathEntropy, ...
            sprintf('%s_thresholdBinarisation_DiceStats.mat', datestr(now,'yyyymmdd')));
        threshold_2_idx_in_columns = idx_thresh_2; %#ok<NASGU>
        save(diceStatsPath, ...
             'dice_multi', 'dice_thresholds', 'n_paired', ...
             'chi2_friedman', 'df_friedman', 'p_friedman', ...
             'pair_label', 'pair_signrank', 'pair_p_raw', 'pair_p_holm', ...
             'pair_g', 'pair_median_diff', ...
             'T_dice');
        fprintf('\n  Saved Dice stats to: %s\n', diceStatsPath);
    end
end

fprintf('Section 9 complete\n');

end  % RUN.section9

%% =========================================================================
%% Helper Functions for Section 7
%% =========================================================================

function ent = population_entropy(activity)
    % Calculate population-level entropy from activity vector
    % Input: activity [n × 1] - activity values (continuous or binary)
    % Output: ent - scalar entropy value (normalized 0-1)

    if isempty(activity) || length(activity) < 2
        ent = NaN;
        return;
    end

    % Remove NaN values
    activity = activity(~isnan(activity));

    if length(activity) < 2
        ent = NaN;
        return;
    end

    % For binary data (0 and 1)
    if all(ismember(activity, [0, 1]))
        p1 = mean(activity);  % Proportion active
        p0 = 1 - p1;          % Proportion inactive

        if p1 == 0 || p0 == 0
            ent = 0;  % No entropy if all same state
        else
            ent = -(p0 * log2(p0) + p1 * log2(p1));  % Binary entropy
            ent = ent / 1;  % Already normalized to [0, 1]
        end
    else
        % For continuous data, use histogram binning
        n_bins = min(10, max(2, round(sqrt(length(activity)))));
        [counts, ~] = histcounts(activity, n_bins);
        counts = counts(counts > 0);  % Remove zero bins

        if isempty(counts) || length(counts) < 2
            ent = 0;
        else
            p = counts / sum(counts);
            ent = -sum(p .* log2(p + eps));  % Shannon entropy
            ent = ent / log2(n_bins);  % Normalize by max entropy
        end
    end

    % Ensure ent is in [0, 1]
    ent = min(max(ent, 0), 1);
end

function morans = calculate_moransI_simple(grid_2D)
    % Calculate Moran's I autocorrelation for binary 2D grid
    % Input: grid_2D [gridY × gridX] - binary values (0 or 1)
    % Output: morans - scalar Moran's I value

    if all(grid_2D(:) == 0) || all(grid_2D(:) == 1)
        morans = NaN;  % No autocorrelation if all same value
        return;
    end

    [gridY, gridX] = size(grid_2D);
    n = gridY * gridX;

    % Flatten grid
    values = grid_2D(:);

    % Calculate mean
    mean_val = mean(values);

    % Create adjacency matrix (4-neighborhood)
    W = zeros(n);
    for i = 1:gridY
        for j = 1:gridX
            idx = (i-1)*gridX + j;

            % Right neighbor
            if j < gridX
                W(idx, idx+1) = 1;
                W(idx+1, idx) = 1;
            end

            % Bottom neighbor
            if i < gridY
                W(idx, idx+gridX) = 1;
                W(idx+gridX, idx) = 1;
            end
        end
    end

    % Normalize W (row-stochastic)
    W = W ./ (sum(W, 2) + eps);

    % Calculate Moran's I
    numerator = (values - mean_val)' * W * (values - mean_val);
    denominator = sum((values - mean_val).^2);

    morans = (n / sum(sum(W))) * (numerator / denominator);
end

%% =========================================================================
%% HELPER FUNCTIONS: Moran's I and Dispersion Calculations
%% =========================================================================

function morans = calculate_moransI_grid(values_vec, gridY, gridX)
% CALCULATE_MORANSI_GRID - Calculate Moran's I for gridded data
%
% Computes spatial autocorrelation using 4-connected neighbor weights
%
% INPUTS:
%   values_vec - Vector of values [nGridCells × 1]
%   gridY      - Grid height
%   gridX      - Grid width
%
% OUTPUTS:
%   morans     - Moran's I value (ranges from -1 to 1)

    n = length(values_vec);
    values = values_vec(:);
    mean_val = nanmean(values);

    if isnan(mean_val)
        morans = NaN;
        return;
    end

    % Build weight matrix for 4-connected neighbors
    W = zeros(n, n);

    for i = 1:gridY
        for j = 1:gridX
            idx = (i-1)*gridX + j;

            % Right neighbor
            if j < gridX
                W(idx, idx+1) = 1;
                W(idx+1, idx) = 1;
            end

            % Bottom neighbor
            if i < gridY
                W(idx, idx+gridX) = 1;
                W(idx+gridX, idx) = 1;
            end
        end
    end

    % Row-normalize weight matrix
    row_sums = sum(W, 2);
    row_sums(row_sums == 0) = 1;  % Avoid division by zero
    W = W ./ row_sums;

    % Calculate Moran's I
    centered = values - mean_val;
    numerator = centered' * W * centered;
    denominator = sum(centered.^2);

    if denominator == 0
        morans = NaN;
    else
        morans = (n / sum(sum(W))) * (numerator / denominator);
    end
end

function disp = calculate_dispersion_binary(activity_bin, grid_coords)
% CALCULATE_DISPERSION_BINARY - Calculate dispersion for binary data
%
% Finds active cells, computes centroid, averages Euclidean distances
%
% INPUTS:
%   activity_bin  - Binary activity vector [nGridCells × 1]
%   grid_coords   - Grid coordinates [nGridCells × 2]
%
% OUTPUTS:
%   disp          - Average Euclidean distance from centroid

    % Find active grid cells (binary: 1 = active, 0 = inactive)
    active_inds = find(activity_bin > 0);

    if isempty(active_inds)
        disp = NaN;  % No active cells
        return;
    end

    % Get coordinates of active cells
    act_pos = grid_coords(active_inds, :);

    % Calculate centroid (mean position)
    mean_pos = mean(act_pos, 1);

    % Calculate Euclidean distance from each active cell to centroid
    dist_v = sqrt(sum((act_pos - mean_pos).^2, 2));

    % Average distance = dispersion metric
    disp = mean(dist_v);
end

function disp = calculate_dispersion_weighted(activity, grid_coords)
% CALCULATE_DISPERSION_WEIGHTED - Calculate dispersion weighted by activity levels
%
% Uses activity-weighted centroid and activity-weighted average distance
%
% INPUTS:
%   activity      - Continuous activity vector [nGridCells × 1]
%   grid_coords   - Grid coordinates [nGridCells × 2]
%
% OUTPUTS:
%   disp          - Weighted average Euclidean distance from centroid

    % Threshold for considering activity (to avoid noise)
    activity_threshold = 0.01;

    % Find cells with meaningful activity
    active_mask = activity > activity_threshold;

    if sum(active_mask) == 0
        disp = NaN;  % No active cells
        return;
    end

    % Get active cell coordinates and activities
    active_coords = grid_coords(active_mask, :);
    active_weights = activity(active_mask);

    % Calculate weighted centroid
    total_weight = sum(active_weights);
    weighted_centroid = sum(active_coords .* active_weights, 1) / total_weight;

    % Calculate weighted average distance from centroid
    distances = sqrt(sum((active_coords - weighted_centroid).^2, 2));
    disp = sum(distances .* active_weights) / total_weight;
end
