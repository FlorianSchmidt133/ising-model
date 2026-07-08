%% =========================================================================
%% Figure 5: Grid40 Threshold Inclusion Analysis for Binarization
%% =========================================================================
% This script analyzes what fraction of Grid40 data points are included
% at different threshold levels (mean ± k×sigma) to determine optimal
% binarization thresholds for Figure 5 Ising models.
%
% This script uses a threshold-based approach rather than sigma distribution:
% - Calculates global mean and sigma per condition
% - Evaluates fraction of data points below threshold = mean + k×sigma
% - Generates plots showing threshold values vs k-values (z-scores)
%
% Analysis includes 4 threshold inclusion analyses:
% 1. Global Pooled Data - All cells, timepoints, and trials pooled per condition
% 2. By Animal (Per Condition) - Data separated by animal within each condition
% 3. By Animal (Across Conditions) - Data aggregated by animal across all conditions
% 4. Expert vs NoSpout Comparison - Direct comparison of reward availability effect
%
% For Analyses 1-3:
% - Collects RAW activity values from all grid cells, timepoints, and trials
% - Calculates global mean and sigma per condition (or per animal)
% - Evaluates fraction included at thresholds: mean + k×sigma (k = -1 to 6)
%
% Analysis 1 generates 2 plot versions:
%   * Version A: X = threshold value (mean + k×sigma), Y = fraction included
%   * Version B: X = k (z-score multiplier), Y = fraction included
%
% Analyses 2-4 generate Version B only (z-scores) for easier comparison across animals
%
% Data requirements:
% - Grid40 structure with P1 data (gridY × gridX × timepoints × trials)
% - Rec structure with AnimalID information
% - params structure with condition colors (Naive, Beginner, Expert, NoSpout)

%% =========================================================================
%% SECTION 1: Setup and Parameters
%% =========================================================================

fprintf('\n=== Figure 5: Grid40 Threshold Inclusion Analysis ===\n');

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

% Define threshold evaluation parameters
k_range = linspace(-1, 6, 200);  % Standard deviation multipliers (k values)
fprintf('Threshold evaluation: k = -1 to +6 sigma (200 points)\n');

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

    % NMF parameters 
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
        % APPROACH: Direct Fraction → Percentile
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
%% SECTION 2: Grid40 Threshold Inclusion Analysis
%% =========================================================================

fprintf('\n--- Section 2: Grid40 Threshold Inclusion Analysis ---\n');

% =========================================================================
% Analysis 1: Global Pooled Data - Threshold Inclusion by Condition
% =========================================================================
% This analysis pools ALL grid cells, timepoints, and trials together per
% condition, then evaluates the fraction of data points included at different
% threshold levels (mean ± k×sigma).
%
% DATA COLLECTION:
% - For each condition, collect ALL activity values: cells × timepoints × trials
% - Result: Single pooled dataset per condition across all recordings
% - Filter: Optionally remove dead cells (mean activity < 0.2)
%
% VISUALIZATION:
% - Two versions of threshold inclusion plots:
%   * Version A: X = threshold value (mean + k×sigma), Y = fraction included
%   * Version B: X = k (z-score multiplier), Y = fraction included
% - Lines: One per condition, color-coded
% - Threshold markers based on NMF percentiles
%
% PURPOSE: Determine what global threshold level (mean + k×sigma) includes
%          the appropriate fraction of all data points based on NMF active cells

fprintf('\n=== Analysis 1: Global Pooled Data ===\n');

% Initialize data collection structure
Grid40DataGlobal = struct();

% Collect RAW activity values per grid cell
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        Grid40DataGlobal.(condition) = [];
        continue;
    end

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);
    allDataValues = [];

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

                % Collect ALL raw data values
                allDataValues = [allDataValues; gridData_filtered(:)];
            end
        end
    end

    % Remove NaN values and store
    allDataValues = allDataValues(~isnan(allDataValues));
    Grid40DataGlobal.(condition) = allDataValues;

    fprintf('  %s: %d raw data values collected\n', condition, length(allDataValues));
end

% -------------------------------------------------------------------------
% Plot Version A: Threshold Values (mean + k×sigma)
% -------------------------------------------------------------------------
fprintf('Creating threshold inclusion plot (Version A: threshold values)\n');

figure('Name', 'Grid40 Global Pooled Data - Threshold Inclusion (Version A)');
hold on;

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40DataGlobal.(condition);

    if isempty(data)
        continue;
    end

    % Calculate global mean and sigma
    global_mean = mean(data, 'omitnan');
    global_sigma = std(data, 'omitnan');

    % Calculate threshold values and fraction included
    threshold_values = global_mean + k_range * global_sigma;
    fraction_included = zeros(size(k_range));

    for i = 1:length(k_range)
        fraction_included(i) = sum(data <= threshold_values(i)) / length(data);
    end

    plot(threshold_values, fraction_included, 'Color', conditionColors.(condition), ...
         'LineWidth', 2, 'DisplayName', sprintf('%s (n=%d)', condition, length(data)));
end

% Add vertical threshold lines for each condition
for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40DataGlobal.(condition);

    if ~isempty(data) && USE_NMF_THRESHOLD && isfield(NMF_Thresholds, condition)
        % Calculate threshold value at NMF percentile
        global_mean = mean(data, 'omitnan');
        global_sigma = std(data, 'omitnan');

        targetFraction = NMF_Thresholds.(condition).percentile_ApproachA / 100;
        threshold_values_cond = global_mean + k_range * global_sigma;
        fraction_included_cond = zeros(size(k_range));

        for i = 1:length(k_range)
            fraction_included_cond(i) = sum(data <= threshold_values_cond(i)) / length(data);
        end

        [~, k_idx] = min(abs(fraction_included_cond - targetFraction));
        threshold_nmf = threshold_values_cond(k_idx);

        xline(threshold_nmf, '--', sprintf('NMF: %.1f%%', NMF_Thresholds.(condition).percentile_ApproachA), ...
              'Color', conditionColors.(condition), 'LineWidth', 1.5, 'Alpha', 0.7, ...
              'HandleVisibility', 'off', 'LabelVerticalAlignment', 'bottom');
    end
end

xlabel('dF/F Threshold (Raw Activity Value)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Fraction of Data Points Included', 'FontSize', 12, 'FontWeight', 'bold');
title('Grid40 Global Pooled Data - Threshold Inclusion (Threshold Values)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
box on;
set(gca, 'FontSize', 11, 'LineWidth', 1.5);
hold off;

% -------------------------------------------------------------------------
% Plot Version B: Z-scores (k multipliers)
% -------------------------------------------------------------------------
fprintf('Creating threshold inclusion plot (Version B: z-scores)\n');

figure('Name', 'Grid40 Global Pooled Data - Threshold Inclusion (Version B)');
hold on;

for c = 1:length(conditions)
    condition = conditions{c};
    data = Grid40DataGlobal.(condition);

    if isempty(data)
        continue;
    end

    % Calculate global mean and sigma
    global_mean = mean(data, 'omitnan');
    global_sigma = std(data, 'omitnan');

    % Calculate threshold values and fraction included
    threshold_values = global_mean + k_range * global_sigma;
    fraction_included = zeros(size(k_range));

    for i = 1:length(k_range)
        fraction_included(i) = sum(data <= threshold_values(i)) / length(data);
    end

    plot(k_range, fraction_included, 'Color', conditionColors.(condition), ...
         'LineWidth', 2, 'DisplayName', sprintf('%s (n=%d)', condition, length(data)));

    % Add vertical line at NMF threshold
    if USE_NMF_THRESHOLD && isfield(NMF_Thresholds, condition)
        targetFraction = NMF_Thresholds.(condition).percentile_ApproachA / 100;
        [~, k_idx] = min(abs(fraction_included - targetFraction));
        k_nmf = k_range(k_idx);
        xline(k_nmf, '--', sprintf('NMF: %.1f%%', NMF_Thresholds.(condition).percentile_ApproachA), ...
              'Color', conditionColors.(condition), 'LineWidth', 1.5, 'Alpha', 0.7, ...
              'HandleVisibility', 'off', 'LabelVerticalAlignment', 'bottom');
    end
end

xlabel('Standard Deviations from Mean (k)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Fraction of Data Points Included', 'FontSize', 12, 'FontWeight', 'bold');
title('Grid40 Global Pooled Data - Threshold Inclusion (Z-scores)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
box on;
set(gca, 'FontSize', 11, 'LineWidth', 1.5);
hold off;

%% =========================================================================
%% Analysis 2: By Animal (Per Condition) - Threshold Inclusion
%% =========================================================================
% This analysis pools all grid cells, timepoints, and trials by individual
% animal within each condition, then visualizes as a 2×2 subplot grid using
% threshold inclusion.
%
% DATA COLLECTION:
% - Same as Analysis 1 (global pooled data), but separated by AnimalID
% - Collect ALL raw activity values: cells × timepoints × trials per animal
% - Result: One dataset per animal within each condition
%
% VISUALIZATION:
% - 2×2 subplot layout (Naive, Beginner, Expert, NoSpout)
% - Each subplot shows threshold inclusion curves for individual animals
% - X-axis: Standard Deviations from Mean (k)
% - Y-axis: Fraction of Data Points Included
% - Lines: One per animal (different colors per animal)
%
% PURPOSE: Assess inter-animal variability in threshold inclusion patterns
%          within each training condition

fprintf('\n=== Analysis 2: By Animal (Per Condition) ===\n');

% Initialize data collection structure
Grid40DataPerTrial_ByAnimal = struct();

% Collect per-trial sigma values grouped by animal
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('Processing condition: %s\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Warning: No Grid40 data found for %s\n', conditionIndividual);
        Grid40DataPerTrial_ByAnimal.(condition) = struct();
        continue;
    end

    % Initialize condition structure
    Grid40DataPerTrial_ByAnimal.(condition) = struct();

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

                % Subset timeframes for analysis
                gridData_P1 = gridData_P1(:, :, TimeFrameSelection, :);
                nTimepoints_selected = length(TimeFrameSelection);

                % Reshape to [gridCells × timepoints_selected × trials]
                gridData_reshaped = reshape(gridData_P1, [gridY*gridX, nTimepoints_selected, nTrials]);

                % Filter based on Remove_Low_Values flag
                if Remove_Low_Values
                    % Calculate mean activity per grid cell (for filtering)
                    gridData_forMean = reshape(gridData_reshaped, [gridY*gridX, nTimepoints_selected*nTrials]);
                    meanActivity = mean(gridData_forMean, 2, 'omitnan');

                    % Filter dead cells
                    validIdx = (meanActivity >= DEAD_CELL_THRESHOLD);
                    gridData_filtered = gridData_reshaped(validIdx, :, :);
                else
                    gridData_filtered = gridData_reshaped;
                end

                % Collect ALL raw data values for this recording
                allDataValues = gridData_filtered(:);
                allDataValues = allDataValues(~isnan(allDataValues));

                % Store or append to animal's data
                if isfield(Grid40DataPerTrial_ByAnimal.(condition), animalID_field)
                    Grid40DataPerTrial_ByAnimal.(condition).(animalID_field) = ...
                        [Grid40DataPerTrial_ByAnimal.(condition).(animalID_field); allDataValues];
                else
                    Grid40DataPerTrial_ByAnimal.(condition).(animalID_field) = allDataValues;
                end
            end
        end
    end

    % Print summary
    if isstruct(Grid40DataPerTrial_ByAnimal.(condition))
        animalFields = fieldnames(Grid40DataPerTrial_ByAnimal.(condition));
        fprintf('  %s: %d animals found\n', condition, length(animalFields));
        for a = 1:length(animalFields)
            animalField = animalFields{a};
            nSamples = length(Grid40DataPerTrial_ByAnimal.(condition).(animalField));
            fprintf('    %s: %d data values\n', animalField, nSamples);
        end
    end
end

% -------------------------------------------------------------------------
% Plot Version B: Z-scores (k multipliers) - 2×2 Subplots by Condition
% -------------------------------------------------------------------------
fprintf('Creating 2x2 subplot grid by animal (Version B: Z-scores)\n');

figure('Name', 'Grid40 By Animal (Per Condition) - Threshold Inclusion (Version B)');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for c = 1:length(conditions)
    condition = conditions{c};

    nexttile;
    hold on;

    % Check if we have data for this condition
    if ~isfield(Grid40DataPerTrial_ByAnimal, condition) || ...
       ~isstruct(Grid40DataPerTrial_ByAnimal.(condition))
        title(sprintf('%s (no data)', condition));
        xlabel('Standard Deviations from Mean (k)', 'FontSize', 10);
        ylabel('Fraction of Data Points Included', 'FontSize', 10);
        xlim([-1 6]);
        ylim([0 1]);
        grid on;
        box on;
        hold off;
        continue;
    end

    % Get animal fields for this condition
    animalFields = fieldnames(Grid40DataPerTrial_ByAnimal.(condition));
    nAnimals = length(animalFields);

    if nAnimals == 0
        title(sprintf('%s (no animals)', condition));
        xlabel('Standard Deviations from Mean (k)', 'FontSize', 10);
        ylabel('Fraction of Data Points Included', 'FontSize', 10);
        xlim([-1 6]);
        ylim([0 1]);
        grid on;
        box on;
        hold off;
        continue;
    end

    % Generate distinct colors for each animal
    colorOrder = lines(nAnimals);

    % Plot threshold inclusion curve for each animal
    for a = 1:nAnimals
        animalField = animalFields{a};
        animalData = Grid40DataPerTrial_ByAnimal.(condition).(animalField);

        if isempty(animalData)
            continue;
        end

        % Calculate global mean and sigma for this animal
        global_mean = mean(animalData, 'omitnan');
        global_sigma = std(animalData, 'omitnan');

        % Calculate threshold values and fraction included
        threshold_values = global_mean + k_range * global_sigma;
        fraction_included = zeros(size(k_range));

        for i = 1:length(k_range)
            fraction_included(i) = sum(animalData <= threshold_values(i)) / length(animalData);
        end

        plot(k_range, fraction_included, 'Color', colorOrder(a, :), 'LineWidth', 1.5, ...
             'DisplayName', sprintf('%s (n=%d)', animalField, length(animalData)));

        % Add vertical line at NMF threshold
        if USE_NMF_THRESHOLD && isfield(NMF_Thresholds, condition) && ~isempty(NMF_Thresholds.(condition))
            targetFraction = NMF_Thresholds.(condition).percentile_ApproachA / 100;
            [~, k_idx] = min(abs(fraction_included - targetFraction));
            k_nmf = k_range(k_idx);

            if a == 1  % Only add label for first animal
                xline(k_nmf, '--', sprintf('NMF: %.1f%%', NMF_Thresholds.(condition).percentile_ApproachA), ...
                      'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, 'Alpha', 0.7, ...
                      'HandleVisibility', 'off', 'LabelVerticalAlignment', 'bottom');
            else
                xline(k_nmf, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, 'Alpha', 0.7, ...
                      'HandleVisibility', 'off');
            end
        end
    end

    xlabel('Standard Deviations from Mean (k)', 'FontSize', 10);
    ylabel('Fraction of Data Points Included', 'FontSize', 10);
    title(sprintf('%s (n=%d animals)', condition, nAnimals));
    legend('Location', 'best', 'FontSize', 8);
    xlim([-1 6]);
    ylim([0 1]);
    grid on;
    box on;
    hold off;
end

sgtitle('Grid40 By Animal (Per Condition) - Threshold Inclusion (Z-scores)', 'FontWeight', 'bold');

%% =========================================================================
%% Analysis 3: By Animal (Across Conditions) - Threshold Inclusion
%% =========================================================================
% This analysis pools all grid cells, timepoints, and trials by individual
% animal across ALL training conditions, then visualizes as a single overlay
% plot using threshold inclusion.
%
% DATA COLLECTION:
% - Same as Analysis 1 (global pooled data), aggregated by AnimalID
% - Collect ALL raw activity values: cells × timepoints × trials per animal
% - Aggregation: All conditions pooled per animal ID
% - Result: One dataset per animal spanning all their recording conditions
%
% VISUALIZATION:
% - Single threshold inclusion plot with all animals overlaid (Version B: Z-scores)
% - X-axis: Standard Deviations from Mean (k)
% - Y-axis: Fraction of Data Points Included
% - Lines: One per animal (different colors per animal)
%
% PURPOSE: Assess animal-specific threshold inclusion signatures that persist
%          across different training stages

fprintf('\n=== Analysis 3: By Animal (Across Conditions) ===\n');

% Initialize data collection structure
Grid40DataPerTrial_AllAnimals = struct();

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

                % Subset timeframes for analysis
                gridData_P1 = gridData_P1(:, :, TimeFrameSelection, :);
                nTimepoints_selected = length(TimeFrameSelection);

                % Reshape to [gridCells × timepoints_selected × trials]
                gridData_reshaped = reshape(gridData_P1, [gridY*gridX, nTimepoints_selected, nTrials]);

                % Filter based on Remove_Low_Values flag
                if Remove_Low_Values
                    % Calculate mean activity per grid cell (for filtering)
                    gridData_forMean = reshape(gridData_reshaped, [gridY*gridX, nTimepoints_selected*nTrials]);
                    meanActivity = mean(gridData_forMean, 2, 'omitnan');

                    % Filter dead cells
                    validIdx = (meanActivity >= DEAD_CELL_THRESHOLD);
                    gridData_filtered = gridData_reshaped(validIdx, :, :);
                else
                    gridData_filtered = gridData_reshaped;
                end

                % Collect ALL raw data values for this recording
                allDataValues = gridData_filtered(:);
                allDataValues = allDataValues(~isnan(allDataValues));

                % Append to animal's data (aggregating across all conditions)
                if isfield(Grid40DataPerTrial_AllAnimals, animalID_field)
                    Grid40DataPerTrial_AllAnimals.(animalID_field) = ...
                        [Grid40DataPerTrial_AllAnimals.(animalID_field); allDataValues];
                else
                    Grid40DataPerTrial_AllAnimals.(animalID_field) = allDataValues;
                end
            end
        end
    end
end

% Print summary
animalFields = fieldnames(Grid40DataPerTrial_AllAnimals);
nAnimalsTotal = length(animalFields);
fprintf('\nTotal animals found across all conditions: %d\n', nAnimalsTotal);
for a = 1:nAnimalsTotal
    animalField = animalFields{a};
    nSamples = length(Grid40DataPerTrial_AllAnimals.(animalField));
    fprintf('  %s: %d data values\n', animalField, nSamples);
end

% -------------------------------------------------------------------------
% Plot Version B: Z-scores (k multipliers) - All Animals Across All Conditions
% -------------------------------------------------------------------------
fprintf('Creating aggregated threshold inclusion plot across all conditions (Version B: Z-scores)\n');

figure('Name', 'Grid40 By Animal (Across Conditions) (Version B)');
hold on;

if nAnimalsTotal == 0
    title('No animals found');
    xlabel('Standard Deviations from Mean (k)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Fraction of Data Points Included', 'FontSize', 12, 'FontWeight', 'bold');
    xlim([-1 6]);
    ylim([0 1]);
    grid on;
    box on;
else
    % Generate distinct colors for each animal
    colorOrder = lines(nAnimalsTotal);

    % Plot threshold inclusion curve for each animal
    for a = 1:nAnimalsTotal
        animalField = animalFields{a};
        animalData = Grid40DataPerTrial_AllAnimals.(animalField);

        if isempty(animalData)
            continue;
        end

        % Calculate global mean and sigma for this animal
        global_mean = mean(animalData, 'omitnan');
        global_sigma = std(animalData, 'omitnan');

        % Calculate threshold values and fraction included
        threshold_values = global_mean + k_range * global_sigma;
        fraction_included = zeros(size(k_range));

        for i = 1:length(k_range)
            fraction_included(i) = sum(animalData <= threshold_values(i)) / length(animalData);
        end

        plot(k_range, fraction_included, 'Color', colorOrder(a, :), 'LineWidth', 2, ...
             'DisplayName', sprintf('%s (n=%d)', animalField, length(animalData)));
    end

    xlabel('Standard Deviations from Mean (k)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Fraction of Data Points Included', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Grid40 By Animal (Across Conditions) (n=%d animals, Z-scores)', nAnimalsTotal));
    legend('Location', 'best', 'FontSize', 9);
    xlim([-1 6]);
    ylim([0 1]);
    grid on;
    box on;
end

hold off;

%% =========================================================================
%% Analysis 4: Expert vs NoSpout Comparison - Threshold Inclusion
%% =========================================================================
% This analysis directly compares Expert and NoSpout conditions by overlaying
% per-animal threshold inclusion curves to assess the impact of reward availability.
%
% DATA SOURCE:
% - Reuses data from Analysis 2 (by animal per condition)
% - Expert animals: All animals from Expert condition
% - NoSpout animals: All animals from NoSpout condition
%
% VISUALIZATION:
% - Single overlay plot comparing two conditions (Version B: Z-scores only)
% - Expert animals: Thin lines (0.5 width) in Expert condition color
% - NoSpout animals: Thin lines (0.5 width) in NoSpout condition color
% - Legend: Only condition labels shown (individual animals not labeled)
% - X-axis: Standard Deviations from Mean (k)
% - Y-axis: Fraction of Data Points Included
%
% PURPOSE: Direct comparison of Expert vs NoSpout to assess impact of reward
%          availability on threshold inclusion patterns, while showing
%          inter-animal consistency within each condition

fprintf('\n=== Analysis 4: Expert vs NoSpout Comparison ===\n');
fprintf('Creating overlay plot with thin lines per animal (Version B: Z-scores)\n');

figure('Name', 'Grid40 Expert vs NoSpout Comparison (Version B)');
hold on;

% Plot Expert animals (thin lines in Expert color)
if isfield(Grid40DataPerTrial_ByAnimal, 'Expert') && ...
   isstruct(Grid40DataPerTrial_ByAnimal.Expert)

    expertAnimals = fieldnames(Grid40DataPerTrial_ByAnimal.Expert);
    nExpertAnimals = length(expertAnimals);

    fprintf('  Plotting %d Expert animals\n', nExpertAnimals);

    for a = 1:nExpertAnimals
        animalField = expertAnimals{a};
        animalData = Grid40DataPerTrial_ByAnimal.Expert.(animalField);

        if ~isempty(animalData)
            % Calculate global mean and sigma for this animal
            global_mean = mean(animalData, 'omitnan');
            global_sigma = std(animalData, 'omitnan');

            % Calculate threshold values and fraction included
            threshold_values = global_mean + k_range * global_sigma;
            fraction_included = zeros(size(k_range));

            for i = 1:length(k_range)
                fraction_included(i) = sum(animalData <= threshold_values(i)) / length(animalData);
            end

            plot(k_range, fraction_included, 'Color', conditionColors.Expert, 'LineWidth', 0.5, ...
                 'HandleVisibility', 'off');
        end
    end
end

% Plot NoSpout animals (thin lines in NoSpout color)
if isfield(Grid40DataPerTrial_ByAnimal, 'NoSpout') && ...
   isstruct(Grid40DataPerTrial_ByAnimal.NoSpout)

    noSpoutAnimals = fieldnames(Grid40DataPerTrial_ByAnimal.NoSpout);
    nNoSpoutAnimals = length(noSpoutAnimals);

    fprintf('  Plotting %d NoSpout animals\n', nNoSpoutAnimals);

    for a = 1:nNoSpoutAnimals
        animalField = noSpoutAnimals{a};
        animalData = Grid40DataPerTrial_ByAnimal.NoSpout.(animalField);

        if ~isempty(animalData)
            % Calculate global mean and sigma for this animal
            global_mean = mean(animalData, 'omitnan');
            global_sigma = std(animalData, 'omitnan');

            % Calculate threshold values and fraction included
            threshold_values = global_mean + k_range * global_sigma;
            fraction_included = zeros(size(k_range));

            for i = 1:length(k_range)
                fraction_included(i) = sum(animalData <= threshold_values(i)) / length(animalData);
            end

            plot(k_range, fraction_included, 'Color', conditionColors.NoSpout, 'LineWidth', 0.5, ...
                 'HandleVisibility', 'off');
        end
    end
end

% Add legend with dummy plots for conditions only (thicker lines)
plot(NaN, NaN, 'Color', conditionColors.Expert, 'LineWidth', 2, ...
     'DisplayName', 'Expert');
plot(NaN, NaN, 'Color', conditionColors.NoSpout, 'LineWidth', 2, ...
     'DisplayName', 'NoSpout');

xlabel('Standard Deviations from Mean (k)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Fraction of Data Points Included', 'FontSize', 12, 'FontWeight', 'bold');
title('Grid40 Expert vs NoSpout Comparison (by Animal, Z-scores)');
legend('Location', 'best', 'FontSize', 10);
xlim([-1 6]);
ylim([0 1]);
grid on;
box on;
hold off;

