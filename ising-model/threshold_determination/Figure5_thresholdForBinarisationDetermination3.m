%% =========================================================================
%% Figure 5: Sparsity-Matched Binarization Threshold Determination
%% =========================================================================
% This script implements sparsity matching to determine optimal binarization
% thresholds for Figure 5 Ising models.
%
% SPARSITY MATCHING APPROACH:
% Instead of using variance/sigma, this script uses ABSOLUTE ACTIVITY VALUES
% from each trial to find thresholds that reproduce the same sparsity level
% (number of active cells) as detected by NMF/blob detection methods.
%
% WORKFLOW:
% 1. NMF/blob detection determines number of active cells per trial
% 2. For each trial, sort absolute Grid40 values in descending order
% 3. Find threshold = Nth highest value where N = target active cell count
% 4. Collect thresholds across trials and conditions
% 5. Visualize threshold distributions and validate sparsity matching
%
% Analysis outputs:
% 1. Per-Grid-Cell Sigma (variance across time AND trials) - for reference
% 2. Per-trial sparsity-matched thresholds by condition
% 3. CDF plots of thresholds by condition
% 4. Violin plots of thresholds by condition
% 5. Validation plots (actual vs target sparsity)
% 6. Per-trial sparsity matching quality assessment
%
% Data requirements:
% - Grid40 structure with P1 data (gridY × gridX × timepoints × trials)
% - NMF structure with W_all2 components for blob detection
% - Rec structure with AnimalID information
% - params structure with condition colors (Naive, Beginner, Expert, NoSpout)

%% =========================================================================
%% SECTION 1: Setup and Parameters
%% =========================================================================

fprintf('\n=== Figure 5: Sparsity-Matched Threshold Analysis ===\n');

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

        % SPARSITY MATCHING: Store per-trial active counts with metadata
        perTrial_activeCounts = [];  % Will store: [recID, trialID, activeCount]

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
                            avgBlobs = mean(frameBlobCounts);
                            method1_counts = [method1_counts; avgBlobs];

                            % SPARSITY MATCHING: Store per-trial count with metadata
                            perTrial_activeCounts = [perTrial_activeCounts; r, t, avgBlobs];
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
                            nBlobs = cc.NumObjects;
                            method2_counts = [method2_counts; nBlobs];

                            % SPARSITY MATCHING: Store per-trial count with metadata
                            perTrial_activeCounts = [perTrial_activeCounts; r, t, nBlobs];
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

                    % SPARSITY MATCHING: Store per-trial count with metadata
                    perTrial_activeCounts = [perTrial_activeCounts; r, t, nActiveCells];
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

        % SPARSITY MATCHING: Store per-trial active counts with metadata
        % Columns: [recID, trialID, activeCount]
        NMF_Thresholds.(condition).perTrial_activeCounts = perTrial_activeCounts;

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
% Analysis 2: Sparsity-Matched Thresholds - Per-Trial Approach
% =========================================================================
% This analysis implements TRUE sparsity matching by finding thresholds
% that produce exactly the same number of active cells as NMF/blob detection.
%
% ALGORITHM:
% For each trial:
%   1. Get absolute Grid40 values for that trial
%   2. Aggregate across time (max per grid cell) to get spatial activity map
%   3. Get target number of active cells from NMF for that trial
%   4. Sort grid cell values in descending order
%   5. Threshold = Nth highest value where N = target active cells
%   6. Collect threshold values across all trials
%
% TEMPORAL AGGREGATION:
%   Using MAX across timeframes within each trial (peak response per cell)
%   Alternative: Could use MEAN (average response) or other metrics
%
% DATA COLLECTION:
%   - One threshold value per trial
%   - Thresholds vary by condition and recording
%   - Filter: Optionally skip dead cells in threshold calculation
%
% VISUALIZATION:
%   - CDF of threshold values by condition
%   - Violin plots of thresholds by condition
%   - Validation: actual vs target active cell counts
%   - Per-trial matching quality assessment
%
% PURPOSE: Find data-driven thresholds that match NMF sparsity levels

fprintf('\n=== Analysis 2: Sparsity-Matched Thresholds ===\n');

% Initialize data collection structure
SparsityMatchedThresholds = struct();

% Temporal aggregation method for per-trial spatial activity
% Options: 'max' (peak response), 'mean' (average response), 'median'
TEMPORAL_AGGREGATION = 'max';
fprintf('Temporal aggregation method: %s across timeframes\n', upper(TEMPORAL_AGGREGATION));

% Collect per-trial thresholds for each condition
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        SparsityMatchedThresholds.(condition).thresholds = [];
        SparsityMatchedThresholds.(condition).targetCounts = [];
        SparsityMatchedThresholds.(condition).actualCounts = [];
        continue;
    end

    % Check if NMF thresholds exist for this condition
    if ~isfield(NMF_Thresholds, condition) || ...
       ~isfield(NMF_Thresholds.(condition), 'perTrial_activeCounts')
        fprintf('  Warning: No NMF per-trial data for %s\n', condition);
        SparsityMatchedThresholds.(condition).thresholds = [];
        SparsityMatchedThresholds.(condition).targetCounts = [];
        SparsityMatchedThresholds.(condition).actualCounts = [];
        continue;
    end

    % Get per-trial active counts: [recID, trialID, activeCount]
    perTrial_data = NMF_Thresholds.(condition).perTrial_activeCounts;

    % Initialize result arrays
    allThresholds = [];
    allTargetCounts = [];
    allActualCounts = [];
    allRecIDs = [];
    allTrialIDs = [];

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);

    for r = 1:nRecsGrid
        % Skip if in Skip list
        if ismember(r, Skip.(condition))
            continue;
        end

        % Check if P1 exists
        if ~isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
            continue;
        end

        gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

        % Dereference cell array
        if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
            gridData_P1 = gridData_P1_cell{:};
        else
            gridData_P1 = gridData_P1_cell;
        end

        if isempty(gridData_P1)
            continue;
        end

        % Grid40 data: [gridY × gridX × timepoints × trials]
        [gridY, gridX, nTimepoints, nTrials] = size(gridData_P1);

        % Subset timeframes for analysis
        gridData_P1 = gridData_P1(:, :, TimeFrameSelection, :);
        nTimepoints_selected = length(TimeFrameSelection);

        % Calculate mean activity per grid cell (for filtering dead cells)
        gridData_forMean = reshape(gridData_P1, [gridY*gridX, nTimepoints_selected*nTrials]);
        meanActivity = mean(gridData_forMean, 2, 'omitnan');

        % Process each trial
        for t = 1:nTrials
            % Get target active count for this trial from NMF
            trialIdx = find(perTrial_data(:,1) == r & perTrial_data(:,2) == t);
            if isempty(trialIdx)
                % No NMF data for this trial, skip
                continue;
            end
            targetActiveCount = perTrial_data(trialIdx, 3);

            % Skip if target is zero or NaN
            if isnan(targetActiveCount) || targetActiveCount <= 0
                continue;
            end

            % Get Grid40 data for this trial: [gridY × gridX × timepoints]
            trialData = gridData_P1(:, :, :, t);

            % Reshape to [gridCells × timepoints]
            trialData_reshaped = reshape(trialData, [gridY*gridX, nTimepoints_selected]);

            % Aggregate across time (max, mean, or median)
            switch TEMPORAL_AGGREGATION
                case 'max'
                    spatialActivity = max(trialData_reshaped, [], 2, 'omitnan');
                case 'mean'
                    spatialActivity = mean(trialData_reshaped, 2, 'omitnan');
                case 'median'
                    spatialActivity = median(trialData_reshaped, 2, 'omitnan');
                otherwise
                    error('Unknown temporal aggregation method: %s', TEMPORAL_AGGREGATION);
            end

            % Filter out dead cells if requested
            if Remove_Low_Values
                validIdx = (meanActivity >= DEAD_CELL_THRESHOLD) & ~isnan(spatialActivity);
                spatialActivity_filtered = spatialActivity(validIdx);
            else
                validIdx = ~isnan(spatialActivity);
                spatialActivity_filtered = spatialActivity(validIdx);
            end

            % Check if we have enough cells
            nValidCells = length(spatialActivity_filtered);
            if nValidCells == 0
                continue;
            end

            % Ensure target count doesn't exceed number of valid cells
            targetActiveCount_capped = min(targetActiveCount, nValidCells);

            % Sort spatial activity in descending order
            sortedActivity = sort(spatialActivity_filtered, 'descend');

            % Find threshold: Nth highest value
            if targetActiveCount_capped >= 1
                threshold = sortedActivity(round(targetActiveCount_capped));

                % Count actual number of cells above threshold
                actualActiveCount = sum(spatialActivity_filtered >= threshold);
            else
                threshold = NaN;
                actualActiveCount = 0;
            end

            % Store results
            allThresholds = [allThresholds; threshold];
            allTargetCounts = [allTargetCounts; targetActiveCount];
            allActualCounts = [allActualCounts; actualActiveCount];
            allRecIDs = [allRecIDs; r];
            allTrialIDs = [allTrialIDs; t];
        end
    end

    % Store results for this condition
    SparsityMatchedThresholds.(condition).thresholds = allThresholds;
    SparsityMatchedThresholds.(condition).targetCounts = allTargetCounts;
    SparsityMatchedThresholds.(condition).actualCounts = allActualCounts;
    SparsityMatchedThresholds.(condition).recIDs = allRecIDs;
    SparsityMatchedThresholds.(condition).trialIDs = allTrialIDs;

    fprintf('  %s: %d per-trial thresholds computed\n', condition, length(allThresholds));
    fprintf('    Mean threshold: %.3f (range: %.3f - %.3f)\n', ...
        mean(allThresholds, 'omitnan'), ...
        min(allThresholds), ...
        max(allThresholds));
    fprintf('    Mean target cells: %.1f\n', mean(allTargetCounts, 'omitnan'));
    fprintf('    Mean actual cells: %.1f\n', mean(allActualCounts, 'omitnan'));
end

% =========================================================================
% Visualization: daboxplot of Thresholds by Condition
% =========================================================================
fprintf('Creating daboxplot of thresholds by condition\n');

% Prepare data for daboxplot (matrix + groups format)
boxData = [];
boxGroups = [];
boxLabels = {};
boxColors = [];
groupIdx = 1;

for c = 1:length(conditions)
    condition = conditions{c};
    data = SparsityMatchedThresholds.(condition).thresholds;

    if ~isempty(data)
        data = data(~isnan(data));
        % Append data as column vector
        boxData = [boxData; data(:)];
        % Create group indices for this condition
        boxGroups = [boxGroups; groupIdx * ones(length(data), 1)];
        boxLabels{end+1} = condition;
        boxColors = [boxColors; conditionColors.(condition)];
        groupIdx = groupIdx + 1;
    end
end

figure('Name', 'Sparsity-Matched Thresholds - Distribution by Condition');
if ~isempty(boxData)
    daboxplot(boxData, 'groups', boxGroups, 'colors', boxColors, ...
              'xtlabels', boxLabels, 'scatter', 1, 'outliers', 1, ...
              'whiskers', 0, 'mean', 1, 'boxalpha', 0.7);
    ylabel('Threshold Value (Absolute Activity)');
    title('Distribution of Sparsity-Matched Thresholds by Condition');
    grid on;
else
    title('No threshold data available');
end

% =========================================================================
% Summary Statistics Table
% =========================================================================
fprintf('\n=== Summary of Sparsity-Matched Thresholds ===\n');
fprintf('%-12s | %10s | %10s | %10s | %10s | %10s\n', ...
        'Condition', 'nTrials', 'Mean Thresh', 'Median Thresh', 'Mean Error', 'Median Error');
fprintf('%-12s-|-%10s-|-%10s-|-%10s-|-%10s-|-%10s\n', ...
        '------------', '----------', '----------', '----------', '----------', '----------');

for c = 1:length(conditions)
    condition = conditions{c};
    
    thresholds = SparsityMatchedThresholds.(condition).thresholds;
    targetCounts = SparsityMatchedThresholds.(condition).targetCounts;
    actualCounts = SparsityMatchedThresholds.(condition).actualCounts;
    
    if isempty(thresholds)
        fprintf('%-12s | %10s | %10s | %10s | %10s | %10s\n', ...
                condition, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A');
        continue;
    end
    
    % Remove NaNs
    validIdx = ~isnan(thresholds);
    thresholds_valid = thresholds(validIdx);
    
    validIdx2 = ~isnan(targetCounts) & ~isnan(actualCounts);
    matchError = abs(actualCounts(validIdx2) - targetCounts(validIdx2));
    
    nTrials = length(thresholds_valid);
    meanThresh = mean(thresholds_valid);
    medianThresh = median(thresholds_valid);
    meanErr = mean(matchError);
    medianErr = median(matchError);
    
    fprintf('%-12s | %10d | %10.3f | %10.3f | %10.1f | %10.1f\n', ...
            condition, nTrials, meanThresh, medianThresh, meanErr, medianErr);
end

fprintf('\n=== Analysis Complete ===\n');
fprintf('Sparsity matching analysis finished.\n');
fprintf('Temporal aggregation: %s across timeframes\n', upper(TEMPORAL_AGGREGATION));
fprintf('NMF method: %s\n', methodNames{NMF_METHOD});
fprintf('\nPlot created:\n');
fprintf('  - daboxplot of threshold distributions by condition\n');
fprintf('\nThreshold values can be used for binarization in Ising models.\n');
fprintf('Review threshold distributions to select appropriate binarization values.\n');

