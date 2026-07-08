%% =========================================================================
%% Figure 5: Grid40 Variance/Sigma Analysis for Binarization Threshold
%% =========================================================================
% This script analyzes Grid40 variance/sigma distributions to
% determine optimal binarization thresholds for Figure 5 Ising models.
%
% This script extracts the variance/sigma analysis 
% from the main threshold determination script, providing a streamlined workflow
% for variance-based threshold analysis.
%
% Analysis includes variance/sigma analyses:
% 1. Per-Grid-Cell Sigma (variance across time AND trials)
% 2. Per-Grid-Cell Per-Timepoint Sigma (variance across trials)
% 3. Per-Grid-Cell Per-Trial Sigma (variance across time)
% 4. Per-Trial Sigma by Animal (separated by individual animals)
% 5. Per-Trial Sigma Aggregated by Animal (across all conditions)
% 6. Per-Timepoint Sigma by Animal (separated by individual animals)
% 7. Per-Timepoint Sigma Aggregated by Animal (across all conditions)
% 8. Per-Timepoint Sigma - Expert vs NoSpout comparison
% 9. Global Sigma per Recording (pooling all cells × time × trials)
% 10. Per-Trial Global Sigma (pooling all cells × time per trial)
%
% Data requirements:
% - Grid40 structure with P1 data (gridY × gridX × timepoints × trials)
% - Rec structure with AnimalID information
% - params structure with condition colors (Naive, Beginner, Expert, NoSpout)

%% =========================================================================
%% SECTION 1: Setup and Parameters
%% =========================================================================

fprintf('\n=== Figure 5: Grid40 Variance/Sigma Analysis ===\n');

% Load data structures
% load(mba_p('RawData3.mat'),'Rec');
% load(mba_p('RawData3.mat'),'params');

% Load NMF data for threshold calculation
% if USE_NMF_THRESHOLD
%     load(mba_p('NMF_raw3.mat'), 'NMF_littleRegularisation_Grid40');
%     NMF = NMF_littleRegularisation_Grid40;
%     fprintf('NMF data loaded for threshold calculation\n');
% end

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

% NMF-based threshold selection
USE_NMF_THRESHOLD = true;  % Set to false to use 98th percentile
NMF_METHOD = 2;  % 1=2D blob detection, 2=3D blob detection, 3=Active cells across ALL components

if Remove_Low_Values
    fprintf('Filter enabled: Removing values < %.2f (dead/empty cells)\n', DEAD_CELL_THRESHOLD);
else
    fprintf('Filter disabled: Including all values\n');
end

% Print NMF threshold settings
if USE_NMF_THRESHOLD
    fprintf('Threshold mode: NMF-based (replacing arbitrary 98th percentile)\n');
    methodNames = {'2D blob detection', '3D blob detection', 'Active cells across ALL components'};
    fprintf('  NMF Method: %s\n', methodNames{NMF_METHOD});
    fprintf('  Conversion: Direct fraction → percentile\n');
else
    fprintf('Threshold mode: Arbitrary 98th percentile (original)\n');
end

% Define conditions to analyze
conditions = {'Naive', 'Beginner', 'Expert', 'NoSpout'};

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

% Define KDE evaluation points for sigma/variance plots (0-3 range)
kde_xi_3 = linspace(0, 3, 201);   % For sigma/variance range plots
fprintf('KDE evaluation points: 201 bins spanning 0-3 (sigma range)\n');

% Define timeframe for variance analysis
TimeFrameSelection = 1:80;  % Frames to include (e.g., 1:80 for pre-stimulus only)
% Options: 1:80 (pre-stim), 81:100 (stim), 1:85 (all frames)
fprintf('Timeframe selection for variance analysis: frames %d to %d\n', TimeFrameSelection(1), TimeFrameSelection(end));

%% =========================================================================
%% SECTION 1.5: NMF-Based Threshold Calculation
%% =========================================================================

if USE_NMF_THRESHOLD
    fprintf('\n--- Section 1.5: NMF-Based Threshold Calculation ---\n');

    % Initialize results structure
    NMF_Thresholds = struct();

    % Grid dimensions for Grid40
    gridY = 13;
    gridX = 26;
    totalGridCells = gridY * gridX;  % 338 cells

    % NMF parameters (from Figure3.m)
    pt = 2;  % Pretreatment: 1=Raw, 2=Mean-Subtracted, 3=Min-Subtracted
    position = 'P1';
    nmf_component_threshold = 1;  % params.ComponentFiltering.threshold

    % Blob detection parameters (from blobDetection.m and Figure3.m)
    blob_sigma = 2;  % Gaussian smoothing
    blob_threshold = 1;  % Binarization threshold
    blob_minSize = 5;  % Minimum blob size (pixels)

    fprintf('Grid40 dimensions: %d × %d = %d total cells\n', gridY, gridX, totalGridCells);
    fprintf('NMF pretreatment: %d (Mean-Subtracted)\n', pt);
    fprintf('Component threshold: %.2f\n', nmf_component_threshold);

    % Process each condition
    for c = 1:length(conditions)
        condition = conditions{c};
        conditionIndividual = [condition 'Individual'];

        fprintf('\nProcessing %n', condition);

        % Check if NMF data exists
        if ~isfield(NMF.W_all2, conditionIndividual) || ~isfield(NMF.W_all2.(conditionIndividual), position)
            fprintf('  Warning: No NMF data for %s.%s\n', conditionIndividual, position);
            NMF_Thresholds.(condition).activeCellCount_Method1 = NaN;
            NMF_Thresholds.(condition).activeCellCount_Method2 = NaN;
            NMF_Thresholds.(condition).activeCellCount_Method3 = NaN;
            continue;
        end

        nRecs = length(NMF.W_all2.(conditionIndividual).(position));

        % Initialize accumulators for all three methods
        method1_counts = [];  % 2D blob detection
        method2_counts = [];  % 3D blob detection
        method3_counts = [];  % Active cells across ALL components

        for r = 1:nRecs
            % Skip if in Skip list
            if ismember(r, Skip.(condition))
                continue;
            end

            % Get Grid40 data for this recording (for blob detection)
            if isfield(Grid40, conditionIndividual) && ...
               length(Grid40.(conditionIndividual).AllNeurons) >= r && ...
               isfield(Grid40.(conditionIndividual).AllNeurons(r), position)

                gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;
                if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
                    gridData = gridData_P1_cell{:};
                else
                    gridData = gridData_P1_cell;
                end
            else
                gridData = [];
            end

            % Get NMF W components for this recording
            if isempty(NMF.W_all2.(conditionIndividual).(position){r})
                continue;
            end

            nTrials = length(NMF.W_all2.(conditionIndividual).(position){r});

            for t = 1:nTrials
                W_matrix = NMF.W_all2.(conditionIndividual).(position){r}{t};

                if isempty(W_matrix)
                    continue;
                end

                nComponents = size(W_matrix, 2);

                % ----------------------------------------------------------------
                % METHOD 1: 2D Blob Detection
                % ----------------------------------------------------------------
                if NMF_METHOD == 1 && ~isempty(gridData)
                    % For each frame, count blobs
                    if ndims(gridData) == 4
                        [gY, gX, nFrames, nTrialsGrid] = size(gridData);
                        if t <= nTrialsGrid
                            trialData = gridData(:, :, :, t);  % [gridY × gridX × frames]
                            frameBlobCounts = zeros(nFrames, 1);

                            for f = 1:nFrames
                                frame = squeeze(trialData(:, :, f));
                                % Gaussian smoothing
                                frame_smoothed = imgaussfilt(frame, blob_sigma);
                                % Binarization
                                bw = imbinarize(frame_smoothed, blob_threshold);
                                % Remove small objects
                                bw_clean = bwareaopen(bw, blob_minSize);
                                % Count blobs
                                cc = bwconncomp(bw_clean);
                                frameBlobCounts(f) = cc.NumObjects;
                            end

                            % Average blobs per frame for this trial
                            method1_counts = [method1_counts; mean(frameBlobCounts)];
                        end
                    end
                end

                % ----------------------------------------------------------------
                % METHOD 2: 3D Blob Detection
                % ----------------------------------------------------------------
                if NMF_METHOD == 2 && ~isempty(gridData)
                    % Treat entire trial as 3D volume
                    if ndims(gridData) == 4
                        [gY, gX, nFrames, nTrialsGrid] = size(gridData);
                        if t <= nTrialsGrid
                            trialData = gridData(:, :, :, t);  % [gridY × gridX × frames]

                            % Apply Gaussian smoothing in 3D
                            trialData_smoothed = imgaussfilt3(trialData, blob_sigma);
                            % Binarization
                            bw = imbinarize(trialData_smoothed, blob_threshold);
                            % Remove small 3D objects
                            bw_clean = bwareaopen(bw, blob_minSize);
                            % Count 3D connected components
                            cc = bwconncomp(bw_clean);

                            % Blobs in this trial
                            method2_counts = [method2_counts; cc.NumObjects];
                        end
                    end
                end

                % ----------------------------------------------------------------
                % METHOD 3: Active Cells Across ALL Components
                % ----------------------------------------------------------------
                if NMF_METHOD == 3
                    % For this trial, get all components and find unique active cells
                    activeCellsMask = false(totalGridCells, 1);

                    for comp = 1:nComponents
                        W_component = W_matrix(:, comp);
                        % Mark cells active if they exceed threshold
                        activeCellsMask = activeCellsMask | (W_component > nmf_component_threshold);
                    end

                    % Count unique active cells
                    nActiveCells = sum(activeCellsMask);
                    method3_counts = [method3_counts; nActiveCells];
                end
            end
        end

        % Calculate mean active count for selected method
        if NMF_METHOD == 1
            meanActiveCount = mean(method1_counts, 'omitnan');
            NMF_Thresholds.(condition).activeCellCount_Method1 = meanActiveCount;
        elseif NMF_METHOD == 2
            meanActiveCount = mean(method2_counts, 'omitnan');
            NMF_Thresholds.(condition).activeCellCount_Method2 = meanActiveCount;
        else  % NMF_METHOD == 3
            meanActiveCount = mean(method3_counts, 'omitnan');
            NMF_Thresholds.(condition).activeCellCount_Method3 = meanActiveCount;
        end

        % Store all method counts for reference
        NMF_Thresholds.(condition).method1_all = method1_counts;
        NMF_Thresholds.(condition).method2_all = method2_counts;
        NMF_Thresholds.(condition).method3_all = method3_counts;

        % Calculate active fraction
        activeFraction = meanActiveCount / totalGridCells;
        NMF_Thresholds.(condition).activeFraction = activeFraction;

        % ----------------------------------------------------------------
        % APPROACH A: Direct Fraction → Percentile
        % ----------------------------------------------------------------
        % If X% of cells are active, we want (100-X)th percentile threshold
        percentile_A = 100 * (1 - activeFraction);
        NMF_Thresholds.(condition).percentile_ApproachA = percentile_A;

        fprintf('  Method %d: %.2f active cells/blobs per trial\n', NMF_METHOD, meanActiveCount);
        fprintf('  Active fraction: %.4f (%.2f%%)\n', activeFraction, activeFraction*100);
        fprintf('  Percentile threshold: %.2f%%\n', percentile_A);
    end

    fprintf('\nNMF-based threshold calculation complete.\n');

    % Summary table
    fprintf('\n=== Summary of NMF-Based Thresholds ===\n');
    fprintf('%-12s | %10s | %12s | %12s\n', 'Condition', 'Active', 'Fraction', 'Percentile');
    fprintf('%-12s-|-%10s-|-%12s-|-%12s\n', '------------', '----------', '------------', '------------');
    for c = 1:length(conditions)
        condition = conditions{c};
        if isfield(NMF_Thresholds, condition)
            fprintf('%-12s | %10.2f | %11.4f%% | %11.2f%%\n', ...
                condition, ...
                NMF_Thresholds.(condition).(['activeCellCount_Method' num2str(NMF_METHOD)]), ...
                NMF_Thresholds.(condition).activeFraction * 100, ...
                NMF_Thresholds.(condition).percentile_ApproachA);
        end
    end
    fprintf('\n');
end

%% =========================================================================
%% SECTION 2: Grid40 Variance/Sigma Analysis - Multiple Variance Types
%% =========================================================================

fprintf('\n--- Section 2: Grid40 Variance/Sigma Analysis ---\n');

% =========================================================================
% Analysis 1: Per-Grid-Cell Sigma - CDF by Condition
% =========================================================================
% This analysis computes sigma values per grid cell (pooling across time
% and trials) and visualizes the cumulative distribution across conditions.
%
% DATA COLLECTION:
% - For each grid cell, compute variance across all timepoints and trials
% - Variance calculation: var([nTimepoints × nTrials], all)
% - Result: One sigma value per grid cell per recording
% - Filter: Optionally remove dead cells (mean activity < 0.2)
%
% VISUALIZATION:
% - CDF plot comparing all conditions (Naive, Beginner, Expert, NoSpout)
% - X-axis: Sigma (standard deviation) values [0-3]
% - Y-axis: Cumulative probability [0-1]
% - Lines: One per condition, color-coded
% - Threshold markers (vertical dashed lines per condition):
%   * NMF: Percentile derived from NMF active cell fraction
%   * 98%: Original arbitrary threshold (if NMF disabled)
%
% PURPOSE: Determine binarization threshold for Ising model based on
%          overall activity variability per spatial location

fprintf('\n=== Analysis 1: Per-Grid-Cell Sigma ===\n');

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

% Create CDF plot
fprintf('Creating CDF plot by condition\n');

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

% Add threshold vertical lines for each condition
for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40SigmaPerCell.(condition);

    if ~isempty(data)
        [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

        % Determine threshold based on method
        if USE_NMF_THRESHOLD && isfield(NMF_Thresholds, condition)
            % Direct percentile from NMF active fraction
            targetPercentile = NMF_Thresholds.(condition).percentile_ApproachA;
            threshold_idx = find(f >= (targetPercentile/100), 1, 'first');
            if ~isempty(threshold_idx)
                threshold_sigma = xi(threshold_idx);
                labelText = sprintf('NMF: %.1f%%', targetPercentile);
            else
                threshold_sigma = NaN;
            end
        else
            % Original: Use arbitrary 98th percentile
            threshold_idx = find(f >= 0.98, 1, 'first');
            if ~isempty(threshold_idx)
                threshold_sigma = xi(threshold_idx);
                labelText = '98%';
            else
                threshold_sigma = NaN;
            end
        end

        % Draw threshold line
        if ~isnan(threshold_sigma) && threshold_sigma >= 0 && threshold_sigma <= 3
            xline(threshold_sigma, '--', labelText, 'Color', conditionColors.(condition), ...
                  'LineWidth', 1.5, 'Alpha', 0.7, 'HandleVisibility', 'off', ...
                  'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
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

% =========================================================================
% Analysis 2: Per-Timepoint Sigma - CDF by Condition
% =========================================================================
% This analysis computes sigma values per grid cell at each timepoint
% (variance across trials) and visualizes the cumulative distribution.
%
% DATA COLLECTION:
% - For each grid cell at each timepoint, compute variance across trials
% - Variance calculation: var([nTrials], dim=3)
% - Result: One sigma value per grid cell per timepoint per recording
%           (gridCells × timepoints values per recording)
% - Filter: Optionally remove dead cells (mean activity < 0.2)
%
% VISUALIZATION:
% - CDF plot comparing all conditions (Naive, Beginner, Expert, NoSpout)
% - X-axis: Sigma (standard deviation) values [0-3]
% - Y-axis: Cumulative probability [0-1]
% - Lines: One per condition, color-coded
% - Threshold markers (vertical dashed lines per condition):
%   * NMF: Percentile derived from NMF active cell fraction
%   * 98%: Original arbitrary threshold (if NMF disabled)
%
% PURPOSE: Assess temporal consistency - how variable is activity at each
%          timepoint across trial repetitions

fprintf('\n=== Analysis 2: Per-Timepoint Sigma ===\n');

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

% Create CDF plot
fprintf('Creating CDF plot by condition\n');

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

% Add threshold vertical lines for each condition
for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40SigmaPerTimepoint.(condition);

    if ~isempty(data)
        [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

        % Determine threshold based on method
        if USE_NMF_THRESHOLD && isfield(NMF_Thresholds, condition)
            % Direct percentile from NMF active fraction
            targetPercentile = NMF_Thresholds.(condition).percentile_ApproachA;
            threshold_idx = find(f >= (targetPercentile/100), 1, 'first');
            if ~isempty(threshold_idx)
                threshold_sigma = xi(threshold_idx);
                labelText = sprintf('NMF: %.1f%%', targetPercentile);
            else
                threshold_sigma = NaN;
            end
        else
            % Original: Use arbitrary 98th percentile
            threshold_idx = find(f >= 0.98, 1, 'first');
            if ~isempty(threshold_idx)
                threshold_sigma = xi(threshold_idx);
                labelText = '98%';
            else
                threshold_sigma = NaN;
            end
        end

        % Draw threshold line
        if ~isnan(threshold_sigma) && threshold_sigma >= 0 && threshold_sigma <= 3
            xline(threshold_sigma, '--', labelText, 'Color', conditionColors.(condition), ...
                  'LineWidth', 1.5, 'Alpha', 0.7, 'HandleVisibility', 'off', ...
                  'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
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

% =========================================================================
% Analysis 3: Per-Trial Sigma - CDF by Condition
% =========================================================================
% This analysis computes sigma values per grid cell in each trial
% (variance across time) and visualizes the cumulative distribution.
%
% DATA COLLECTION:
% - For each grid cell in each trial, compute variance across timepoints
% - Variance calculation: var([nTimepoints], dim=2)
% - Result: One sigma value per grid cell per trial per recording
%           (gridCells × trials values per recording)
% - Filter: Optionally remove dead cells (mean activity < 0.2)
%
% VISUALIZATION:
% - CDF plot comparing all conditions (Naive, Beginner, Expert, NoSpout)
% - X-axis: Sigma (standard deviation) values [0-3]
% - Y-axis: Cumulative probability [0-1]
% - Lines: One per condition, color-coded
% - Threshold markers (vertical dashed lines per condition):
%   * NMF: Percentile from NMF active cell fraction (MOST APPROPRIATE)
%   * 98%: Original arbitrary threshold (if NMF disabled)
%
% NOTE: This analysis matches the NMF calculation scale (per-trial)
%
% PURPOSE: Assess trial-to-trial spatial variability - how dynamic is each
%          spatial location within individual trials

fprintf('\n=== Analysis 3: Per-Trial Sigma ===\n');

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

% Create CDF plot
fprintf('Creating CDF plot by condition\n');

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

% Add threshold vertical lines for each condition
for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40SigmaPerTrial.(condition);

    if ~isempty(data)
        [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

        % Determine threshold based on method
        if USE_NMF_THRESHOLD && isfield(NMF_Thresholds, condition)
            % Direct percentile from NMF active fraction
            targetPercentile = NMF_Thresholds.(condition).percentile_ApproachA;
            threshold_idx = find(f >= (targetPercentile/100), 1, 'first');
            if ~isempty(threshold_idx)
                threshold_sigma = xi(threshold_idx);
                labelText = sprintf('NMF: %.1f%%', targetPercentile);
            else
                threshold_sigma = NaN;
            end
        else
            % Original: Use arbitrary 98th percentile
            threshold_idx = find(f >= 0.98, 1, 'first');
            if ~isempty(threshold_idx)
                threshold_sigma = xi(threshold_idx);
                labelText = '98%';
            else
                threshold_sigma = NaN;
            end
        end

        % Draw threshold line
        if ~isnan(threshold_sigma) && threshold_sigma >= 0 && threshold_sigma <= 3
            xline(threshold_sigma, '--', labelText, 'Color', conditionColors.(condition), ...
                  'LineWidth', 1.5, 'Alpha', 0.7, 'HandleVisibility', 'off', ...
                  'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
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

%% =========================================================================
%% Analysis 4: Per-Trial Sigma by Animal - Subplots per Condition
%% =========================================================================
% This analysis groups per-trial sigma data by individual animals within
% each condition and visualizes as a 2×2 subplot grid.
%
% DATA COLLECTION:
% - Same as Analysis 3 (per-trial sigma), but grouped by AnimalID
% - Variance calculation: var([nTimepoints], dim=2) per grid cell per trial
% - Result: Data aggregated by animal within each condition
%
% VISUALIZATION:
% - 2×2 subplot layout (Naive, Beginner, Expert, NoSpout)
% - Each subplot shows CDF lines for individual animals in that condition
% - X-axis: Sigma (standard deviation) values [0-3]
% - Y-axis: Cumulative probability [0-1]
% - Lines: One per animal (different colors per animal)
%
% PURPOSE: Assess inter-animal variability in spatial-temporal dynamics
%          within each training condition

fprintf('\n=== Analysis 4: Per-Trial Sigma by Animal ===\n');

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

% Create multi-panel plot
fprintf('Creating 2x2 subplot grid by animal\n');

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

%% =========================================================================
%% Analysis 5: Per-Trial Sigma Aggregated by Animal - All Conditions
%% =========================================================================
% This analysis pools per-trial sigma data by animal across all training
% conditions and visualizes as a single overlay plot.
%
% DATA COLLECTION:
% - Same as Analysis 3 (per-trial sigma), aggregated by AnimalID
% - Variance calculation: var([nTimepoints], dim=2) per grid cell per trial
% - Aggregation: All conditions pooled per animal ID
% - Result: One dataset per animal spanning all their recording conditions
%
% VISUALIZATION:
% - Single CDF plot with all animals overlaid
% - X-axis: Sigma (standard deviation) values [0-3]
% - Y-axis: Cumulative probability [0-1]
% - Lines: One per animal (different colors per animal)
%
% PURPOSE: Assess animal-specific neural dynamics signatures that persist
%          across different training stages

fprintf('\n=== Analysis 5: Per-Trial Sigma Aggregated by Animal ===\n');

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

% Create aggregated plot
fprintf('Creating aggregated CDF plot across all conditions\n');

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

%% =========================================================================
%% Analysis 6: Per-Timepoint Sigma by Animal - Subplots per Condition
%% =========================================================================
% This analysis groups per-timepoint sigma data by individual animals within
% each condition and visualizes as a 2×2 subplot grid.
%
% DATA COLLECTION:
% - Same as Analysis 2 (per-timepoint sigma), but grouped by AnimalID
% - Variance calculation: var([nTrials], dim=3) per grid cell per timepoint
% - Result: Data aggregated by animal within each condition
%
% VISUALIZATION:
% - 2×2 subplot layout (Naive, Beginner, Expert, NoSpout)
% - Each subplot shows CDF lines for individual animals in that condition
% - X-axis: Sigma (standard deviation) values [0-3]
% - Y-axis: Cumulative probability [0-1]
% - Lines: One per animal (different colors per animal)
%
% PURPOSE: Assess inter-animal variability in trial-to-trial reliability
%          at specific timepoints within each training condition

fprintf('\n=== Analysis 6: Per-Timepoint Sigma by Animal ===\n');

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

% Create multi-panel plot
fprintf('Creating 2x2 subplot grid by animal\n');

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

%% =========================================================================
%% Analysis 7: Per-Timepoint Sigma Aggregated by Animal - All Conditions
%% =========================================================================
% This analysis pools per-timepoint sigma data by animal across all training
% conditions and visualizes as a single overlay plot.
%
% DATA COLLECTION:
% - Same as Analysis 2 (per-timepoint sigma), aggregated by AnimalID
% - Variance calculation: var([nTrials], dim=3) per grid cell per timepoint
% - Aggregation: All conditions pooled per animal ID
% - Result: One dataset per animal spanning all their recording conditions
%
% VISUALIZATION:
% - Single CDF plot with all animals overlaid
% - X-axis: Sigma (standard deviation) values [0-3]
% - Y-axis: Cumulative probability [0-1]
% - Lines: One per animal (different colors per animal)
%
% PURPOSE: Assess animal-specific trial-to-trial reliability signatures that
%          persist across different training stages

fprintf('\n=== Analysis 7: Per-Timepoint Sigma Aggregated by Animal ===\n');

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

% Create aggregated plot
fprintf('Creating aggregated CDF plot across all conditions\n');

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

%% =========================================================================
%% Analysis 8: Expert vs NoSpout Per-Timepoint Sigma Comparison
%% =========================================================================
% This analysis directly compares Expert and NoSpout conditions by overlaying
% per-animal CDF curves to assess the impact of reward availability.
%
% DATA SOURCE:
% - Reuses data from Analysis 6 (per-timepoint sigma by animal)
% - Expert animals: All animals from Expert condition
% - NoSpout animals: All animals from NoSpout condition
%
% VISUALIZATION:
% - Single overlay plot comparing two conditions
% - Expert animals: Thin lines (0.5 width) in Expert condition color
% - NoSpout animals: Thin lines (0.5 width) in NoSpout condition color
% - Legend: Only condition labels shown (individual animals not labeled)
% - X-axis: Sigma (standard deviation) values [0-3]
% - Y-axis: Cumulative probability [0-1]
%
% PURPOSE: Direct comparison of Expert vs NoSpout to assess impact of reward
%          availability on trial-to-trial variability, while showing
%          inter-animal consistency within each condition

fprintf('\n=== Analysis 8: Expert vs NoSpout Comparison ===\n');
fprintf('Creating overlay plot with thin lines per animal\n');

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

%% =========================================================================
%% Analysis 9: Global Sigma per Recording - CDF by Condition
%% =========================================================================
% This analysis computes a single global sigma value per recording by
% pooling ALL grid cells × timepoints × trials together.
%
% DATA COLLECTION:
% - For each recording, pool all spatial locations and time into one vector
% - Variance calculation: var([gridCells × timepoints × trials], all)
% - Result: One sigma value per recording
% - Filter: Optionally remove dead cells before pooling
%
% VISUALIZATION:
% - CDF plot comparing all conditions (Naive, Beginner, Expert, NoSpout)
% - X-axis: Global sigma (standard deviation) values [0-3]
% - Y-axis: Cumulative probability [0-1]
% - Lines: One per condition, color-coded
% - Threshold markers: NMF-derived percentile values
%
% PURPOSE: Determine what global activity variability threshold corresponds
%          to the NMF-based active fraction across entire recordings

fprintf('\n=== Analysis 9: Global Sigma per Recording ===\n');

% Initialize data collection structure
Grid40GlobalSigmaPerRecording = struct();

% Collect global sigma values (one per recording)
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        Grid40GlobalSigmaPerRecording.(condition) = [];
        continue;
    end

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);
    allGlobalSigmaValues = [];

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

                % Filter based on Remove_Low_Values flag
                if Remove_Low_Values
                    % Calculate mean activity per grid cell
                    meanActivity = mean(gridData_reshaped, 2, 'omitnan');
                    % Only keep grid cells where mean activity >= threshold
                    validIdx = (meanActivity >= DEAD_CELL_THRESHOLD);
                    gridData_filtered = gridData_reshaped(validIdx, :);
                else
                    gridData_filtered = gridData_reshaped;
                end

                % Pool everything into one vector and calculate global sigma
                allValues = gridData_filtered(:);
                allValues = allValues(~isnan(allValues));

                if ~isempty(allValues)
                    globalSigma = std(allValues, 0);
                    allGlobalSigmaValues = [allGlobalSigmaValues; globalSigma];
                end
            end
        end
    end

    % Store in structure
    Grid40GlobalSigmaPerRecording.(condition) = allGlobalSigmaValues;

    fprintf('  %s: %d global sigma values collected (one per recording)\n', condition, length(allGlobalSigmaValues));
end

% Create CDF plot
fprintf('Creating CDF plot by condition\n');

figure('Name', 'Grid40 Global Sigma per Recording - CDF by Condition');
hold on;

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40GlobalSigmaPerRecording.(condition);

    if isempty(data)
        continue;
    end

    [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

    plot(xi, f, 'Color', conditionColors.(condition), 'LineWidth', 2, ...
         'DisplayName', sprintf('%s (n=%d recs)', condition, length(data)));
end

% Add threshold vertical lines for each condition
for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40GlobalSigmaPerRecording.(condition);

    if ~isempty(data)
        [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

        % Determine threshold based on method
        if USE_NMF_THRESHOLD && isfield(NMF_Thresholds, condition)
            % Direct percentile from NMF active fraction
            targetPercentile = NMF_Thresholds.(condition).percentile_ApproachA;
            threshold_idx = find(f >= (targetPercentile/100), 1, 'first');
            if ~isempty(threshold_idx)
                threshold_sigma = xi(threshold_idx);
                labelText = sprintf('NMF: %.1f%%', targetPercentile);
            else
                threshold_sigma = NaN;
            end
        else
            % Original: Use arbitrary 98th percentile
            threshold_idx = find(f >= 0.98, 1, 'first');
            if ~isempty(threshold_idx)
                threshold_sigma = xi(threshold_idx);
                labelText = '98%';
            else
                threshold_sigma = NaN;
            end
        end

        % Draw threshold line
        if ~isnan(threshold_sigma) && threshold_sigma >= 0 && threshold_sigma <= 3
            xline(threshold_sigma, '--', labelText, 'Color', conditionColors.(condition), ...
                  'LineWidth', 1.5, 'Alpha', 0.7, 'HandleVisibility', 'off', ...
                  'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
        end
    end
end

xlabel('Global Sigma (Standard Deviation)');
ylabel('Cumulative Probability');
title('Grid40 Global Sigma per Recording - CDF by Condition');
legend('Location', 'best');
xlim([0 3]);
grid on;
box on;
hold off;

%% =========================================================================
%% Analysis 10: Per-Trial Global Sigma - CDF and daboxplot by Condition
%% =========================================================================
% This analysis computes a single global sigma value per trial by
% pooling ALL grid cells × timepoints for each trial separately.
%
% DATA COLLECTION:
% - For each trial, pool all spatial locations and time into one vector
% - Variance calculation: var([gridCells × timepoints], all) per trial
% - Result: One sigma value per trial (multiple values per recording)
% - Filter: Optionally remove dead cells before pooling
%
% VISUALIZATION:
% 1. CDF plot comparing all conditions (Naive, Beginner, Expert, NoSpout)
% 2. daboxplot showing distribution across conditions with scatter points
%
% PURPOSE: Assess trial-by-trial global variability and how it differs
%          across training conditions using both CDF and boxplot views

fprintf('\n=== Analysis 10: Per-Trial Global Sigma ===\n');

% Initialize data collection structure
Grid40GlobalSigmaPerTrial = struct();

% Collect per-trial global sigma values
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        Grid40GlobalSigmaPerTrial.(condition) = [];
        continue;
    end

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);
    allTrialSigmaValues = [];

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

                % Calculate mean activity per grid cell (for filtering)
                gridData_forMean = reshape(gridData_P1, [gridY*gridX, nTimepoints_selected*nTrials]);
                meanActivity = mean(gridData_forMean, 2, 'omitnan');

                % Process each trial separately
                for t = 1:nTrials
                    % Extract single trial: [gridY × gridX × timepoints]
                    trialData = gridData_P1(:, :, :, t);

                    % Reshape to [gridCells × timepoints]
                    trialData_reshaped = reshape(trialData, [gridY*gridX, nTimepoints_selected]);

                    % Filter based on Remove_Low_Values flag
                    if Remove_Low_Values
                        % Only keep grid cells where mean activity >= threshold
                        validIdx = (meanActivity >= DEAD_CELL_THRESHOLD);
                        trialData_filtered = trialData_reshaped(validIdx, :);
                    else
                        trialData_filtered = trialData_reshaped;
                    end

                    % Pool everything into one vector and calculate sigma for this trial
                    allValues = trialData_filtered(:);
                    allValues = allValues(~isnan(allValues));

                    if ~isempty(allValues)
                        trialSigma = std(allValues, 0);
                        allTrialSigmaValues = [allTrialSigmaValues; trialSigma];
                    end
                end
            end
        end
    end

    % Store in structure
    Grid40GlobalSigmaPerTrial.(condition) = allTrialSigmaValues;

    fprintf('  %s: %d per-trial global sigma values collected\n', condition, length(allTrialSigmaValues));
end

% -------------------------------------------------------------------------
% Plot 1: CDF by Condition
% -------------------------------------------------------------------------
fprintf('Creating CDF plot by condition\n');

figure('Name', 'Grid40 Per-Trial Global Sigma - CDF by Condition');
hold on;

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40GlobalSigmaPerTrial.(condition);

    if isempty(data)
        continue;
    end

    [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

    plot(xi, f, 'Color', conditionColors.(condition), 'LineWidth', 2, ...
         'DisplayName', sprintf('%s (n=%d trials)', condition, length(data)));
end

% Add threshold vertical lines for each condition
for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40GlobalSigmaPerTrial.(condition);

    if ~isempty(data)
        [f, xi] = ksdensity(data, kde_xi_3, 'Function', 'cdf');

        % Determine threshold based on method
        if USE_NMF_THRESHOLD && isfield(NMF_Thresholds, condition)
            % Direct percentile from NMF active fraction
            targetPercentile = NMF_Thresholds.(condition).percentile_ApproachA;
            threshold_idx = find(f >= (targetPercentile/100), 1, 'first');
            if ~isempty(threshold_idx)
                threshold_sigma = xi(threshold_idx);
                labelText = sprintf('NMF: %.1f%%', targetPercentile);
            else
                threshold_sigma = NaN;
            end
        else
            % Original: Use arbitrary 98th percentile
            threshold_idx = find(f >= 0.98, 1, 'first');
            if ~isempty(threshold_idx)
                threshold_sigma = xi(threshold_idx);
                labelText = '98%';
            else
                threshold_sigma = NaN;
            end
        end

        % Draw threshold line
        if ~isnan(threshold_sigma) && threshold_sigma >= 0 && threshold_sigma <= 3
            xline(threshold_sigma, '--', labelText, 'Color', conditionColors.(condition), ...
                  'LineWidth', 1.5, 'Alpha', 0.7, 'HandleVisibility', 'off', ...
                  'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
        end
    end
end

xlabel('Per-Trial Global Sigma (Standard Deviation)');
ylabel('Cumulative Probability');
title('Grid40 Per-Trial Global Sigma - CDF by Condition');
legend('Location', 'best');
xlim([0 3]);
grid on;
box on;
hold off;

% -------------------------------------------------------------------------
% Plot 2: daboxplot by Condition
% -------------------------------------------------------------------------
fprintf('Creating daboxplot by condition\n');

% Prepare data as cell array for daboxplot
data_by_cond = cell(1, length(conditions));
condition_colors = [];

for c = 1:length(conditions)
    condition = conditions{c};
    data_by_cond{c} = Grid40GlobalSigmaPerTrial.(condition);
    condition_colors = [condition_colors; conditionColors.(condition)];
end

% Create daboxplot
figure('Name', 'Grid40 Per-Trial Global Sigma - Distribution by Condition');
h = daboxplot(data_by_cond, 'colors', condition_colors, 'xtlabels', conditions, ...
              'scatter', 2, 'outliers', 1, 'mean', 1, 'boxalpha', 0.5, 'whiskers', 0);

ylabel('Per-Trial Global Sigma', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Condition', 'FontSize', 12, 'FontWeight', 'bold');
title('Grid40 Per-Trial Global Sigma - Distribution by Condition', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 11, 'LineWidth', 1.5);
grid on;
box on;

fprintf('\nSection 2 complete: Grid40 variance/sigma analysis plots created\n');

fprintf('\n=== Analysis Complete ===\n');
fprintf('All variance/sigma distributions have been analyzed.\n');
if USE_NMF_THRESHOLD
    fprintf('Review CDF plots with NMF-based threshold markers.\n');
    methodNames = {'2D blob detection', '3D blob detection', 'Active cells across ALL components'};
    fprintf('Method used: %s\n', methodNames{NMF_METHOD});
    fprintf('Threshold approach: Direct fraction → percentile\n');
    fprintf('Threshold markers are labeled with "NMF" on the plots.\n');
else
    fprintf('Review CDF plots to determine optimal binarization thresholds.\n');
    fprintf('Using arbitrary 98th percentile markers (original method).\n');
end
