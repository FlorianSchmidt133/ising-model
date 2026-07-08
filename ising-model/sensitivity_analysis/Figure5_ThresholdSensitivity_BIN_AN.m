%% =========================================================================
%% Threshold Sensitivity Analysis - BIN_AN Mode
%% =========================================================================
% This script analyzes how metrics (Entropy, Dispersion, Active Cells, Moran's I)
% change with different binarization threshold values using BIN_AN normalization.
%
% Purpose: Understand sensitivity of spatial and population metrics to the
% binarization threshold parameter, helping to validate threshold selection.
%
% Tested Thresholds: 1.0, 1.5, 2.0 (σ units for z-score binarization)
% Normalization: BIN_AN (per-animal statistics)
% Position: Position1 (P1) only
% Conditions: Naive, Beginner, Expert, NoSpout

% Load data structures
% load(mba_p('RawData3.mat'),'ActivityData');
% load(mba_p('RawData3.mat'),'params');

% Define Skip arrays (recordings to exclude from analysis)
Skip = [];
Skip.Naive = [1 9 10 16];
Skip.Beginner = [1 6 7 11];
Skip.Expert = [1 4 12 13 14];
Skip.ExpertRandom = [1];
Skip.ExpertAll = [1,4,5,13, 20,21,22,23,24,25,26];
Skip.NoSpout = [1 4 9 10 11 13 14];

% Define conditions to analyze
conditions = {'Naive', 'Beginner', 'Expert', 'NoSpout'};

% Define threshold values to test
thresholds = [1.0, 1.5, 2.0];
nThresholds = length(thresholds);

% Grid parameters 
gridSize = 40;
gridDimensions = [13 26];  % Grid structure: 13 rows × 26 columns

% Moran's I calculation flag
include_MoransI = true;

fprintf('\n=== Threshold Sensitivity Analysis (BIN_AN Mode) ===\n');
fprintf('Testing thresholds: ');
fprintf('%.1f ', thresholds);
fprintf('\n');

%% =========================================================================
%% Main Analysis: Loop Through Thresholds
%% =========================================================================

% Initialize results structure
% Results(threshold_idx).Condition.Metric.RecordingData = cell array
% Results(threshold_idx).Condition.Metric.MedianValue = scalar
Results = struct();

for t_idx = 1:nThresholds
    current_threshold = thresholds(t_idx);
    fprintf('\n=== Processing Threshold = %.1f σ ===\n', current_threshold);

    %% Calculate per-animal statistics for BIN_AN (same across all thresholds)
    % This section only needs to run once since animal stats don't depend on threshold
    if t_idx == 1
        fprintf('Calculating per-animal statistics for BIN_AN...\n');
        AnimalStats_P1 = struct();

        for c = 1:length(conditions)
            condition = conditions{c};
            conditionIndividual = [condition 'Individual'];

            % Check if Grid40 data exists
            if ~isfield(Grid40, conditionIndividual)
                continue;
            end

            % Get animal IDs for this condition
            if isfield(Rec, condition)
                nRecs = length(ActivityData.(condition));
                animalIDs = Rec.(condition).AnimalID;
                uniqueAnimalIDs = unique(animalIDs);

                % Store per-animal statistics
                AnimalStats_P1.(condition) = struct();

                for a = 1:length(uniqueAnimalIDs)
                    currentAnimalID = uniqueAnimalIDs{a};
                    validFieldName = ['Animal_' currentAnimalID];

                    % Find all recordings for this animal
                    animalRecordings = find(strcmp(animalIDs, currentAnimalID));

                    % Collect all grid data for this animal (P1)
                    allGridData_P1 = [];

                    for rec_idx = 1:length(animalRecordings)
                        r = animalRecordings(rec_idx);

                        % Skip if out of bounds or in Skip list
                        if r > nRecs || ismember(r, Skip.(condition))
                            continue;
                        end

                        % Collect P1 data
                        if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
                            gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;
                            if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
                                gridData_P1 = gridData_P1_cell{:};
                            else
                                gridData_P1 = gridData_P1_cell;
                            end

                            if ~isempty(gridData_P1)
                                allGridData_P1 = cat(4, allGridData_P1, gridData_P1);
                            end
                        end
                    end

                    % Calculate animal-level statistics
                    if ~isempty(allGridData_P1)
                        AnimalStats_P1.(condition).(validFieldName).mean = mean(allGridData_P1(:));
                        AnimalStats_P1.(condition).(validFieldName).std = std(allGridData_P1(:));
                    end
                end
            end
        end
    end

    %% Calculate metrics for each condition at current threshold
    for c = 1:length(conditions)
        condition = conditions{c};
        fprintf('  Condition: %s\n', condition);

        conditionIndividual = [condition 'Individual'];

        % Check if Grid40 data exists
        if ~isfield(Grid40, conditionIndividual)
            fprintf('    No Grid40 data found\n');
            continue;
        end

        nRecs = length(ActivityData.(condition));
        nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);

        % Initialize cell arrays for this condition's metrics
        Results(t_idx).(condition).Entropy_BIN_AN = {};
        Results(t_idx).(condition).Dispersion_BIN_AN = {};
        Results(t_idx).(condition).ActiveCells_BIN_AN = {};
        if include_MoransI
            Results(t_idx).(condition).MoransI_BIN_AN = {};
        end

        % Get animal IDs for this recording
        if isfield(Rec, condition)
            animalIDs = Rec.(condition).AnimalID;
        else
            animalIDs = {};
        end

        % Loop through recordings
        for r = 1:min(nRecs, nRecsGrid)
            % Check if this recording should be skipped
            if ismember(r, Skip.(condition))
                continue;
            end

            % Get animal statistics for this recording
            animalMean_P1 = [];
            animalStd_P1 = [];
            if ~isempty(animalIDs) && r <= length(animalIDs)
                currentAnimalID_r = animalIDs{r};
                validFieldName_r = ['Animal_' currentAnimalID_r];

                if isfield(AnimalStats_P1, condition) && ...
                   isfield(AnimalStats_P1.(condition), validFieldName_r)
                    animalMean_P1 = AnimalStats_P1.(condition).(validFieldName_r).mean;
                    animalStd_P1 = AnimalStats_P1.(condition).(validFieldName_r).std;
                end
            end

            % Process Position1 (P1)
            if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
                gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

                % Dereference cell array
                if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
                    gridData_P1 = gridData_P1_cell{:};
                else
                    gridData_P1 = gridData_P1_cell;
                end

                if ~isempty(gridData_P1)
                    % Calculate Entropy (BIN_AN only)
                    [~, ~, ~, ~, ~, ~, ~, ent_bin_an_P1, ~] = ...
                        calculate_entropy_from_grid(gridData_P1, gridDimensions, ...
                        animalMean_P1, animalStd_P1, [], [], current_threshold);
                    Results(t_idx).(condition).Entropy_BIN_AN{end+1} = ent_bin_an_P1;

                    % Calculate Dispersion (BIN_AN only)
                    [~, ~, ~, ~, ~, ~, ~, disp_bin_an_P1, ~] = ...
                        calculate_dispersion_from_grid(gridData_P1, gridDimensions, ...
                        animalMean_P1, animalStd_P1, [], [], current_threshold);
                    Results(t_idx).(condition).Dispersion_BIN_AN{end+1} = disp_bin_an_P1;

                    % Calculate Active Cells (BIN_AN only)
                    [~, ~, ~, ~, ~, ~, ~, actcells_bin_an_P1, ~] = ...
                        calculate_active_cells_from_grid(gridData_P1, gridDimensions, ...
                        animalMean_P1, animalStd_P1, [], [], current_threshold);
                    Results(t_idx).(condition).ActiveCells_BIN_AN{end+1} = actcells_bin_an_P1;

                    % Calculate Moran's I (BIN_AN only)
                    if include_MoransI
                        morans_bin_an_P1 = calculate_moransI_BIN_AN(gridData_P1, ...
                            gridDimensions, animalMean_P1, animalStd_P1, current_threshold);
                        Results(t_idx).(condition).MoransI_BIN_AN{end+1} = morans_bin_an_P1;
                    end
                end
            end
        end

        fprintf('    Processed %d recordings\n', length(Results(t_idx).(condition).Entropy_BIN_AN));
    end
end

fprintf('\n=== Threshold Processing Complete ===\n');

%% =========================================================================
%% Data Aggregation: Calculate Summary Statistics
%% =========================================================================
fprintf('\n=== Calculating Summary Statistics ===\n');

% Initialize summary structure
Summary = struct();
metrics = {'Entropy_BIN_AN', 'Dispersion_BIN_AN', 'ActiveCells_BIN_AN'};
if include_MoransI
    metrics{end+1} = 'MoransI_BIN_AN';
end

for t_idx = 1:nThresholds
    current_threshold = thresholds(t_idx);
    Summary(t_idx).threshold = current_threshold;

    % Initialize overall data collection (pooling all conditions)
    for m = 1:length(metrics)
        metric = metrics{m};
        Summary(t_idx).Overall.(metric).AllData = [];
    end

    % Process each condition
    for c = 1:length(conditions)
        condition = conditions{c};

        if ~isfield(Results(t_idx), condition)
            continue;
        end

        % Process each metric
        for m = 1:length(metrics)
            metric = metrics{m};

            if ~isfield(Results(t_idx).(condition), metric)
                continue;
            end

            % Get all recordings for this condition
            recordingData = Results(t_idx).(condition).(metric);

            if isempty(recordingData)
                continue;
            end

            % Calculate median across recordings (each recording contributes one median)
            recordingMedians = zeros(length(recordingData), 1);
            for rec = 1:length(recordingData)
                data = recordingData{rec};
                recordingMedians(rec) = median(data(:), 'omitnan');
            end

            % Store condition-level summary
            Summary(t_idx).(condition).(metric).median = median(recordingMedians, 'omitnan');
            Summary(t_idx).(condition).(metric).iqr = iqr(recordingMedians);
            Summary(t_idx).(condition).(metric).sem = std(recordingMedians, 'omitnan') / sqrt(length(recordingMedians));
            Summary(t_idx).(condition).(metric).n = length(recordingMedians);
            Summary(t_idx).(condition).(metric).recordingMedians = recordingMedians;

            % Add to overall pool
            Summary(t_idx).Overall.(metric).AllData = [Summary(t_idx).Overall.(metric).AllData; recordingMedians];
        end
    end

    % Calculate overall statistics (across all conditions)
    for m = 1:length(metrics)
        metric = metrics{m};
        allData = Summary(t_idx).Overall.(metric).AllData;

        if ~isempty(allData)
            Summary(t_idx).Overall.(metric).median = median(allData, 'omitnan');
            Summary(t_idx).Overall.(metric).iqr = iqr(allData);
            Summary(t_idx).Overall.(metric).sem = std(allData, 'omitnan') / sqrt(length(allData));
            Summary(t_idx).Overall.(metric).n = length(allData);
        end
    end
end

fprintf('Summary statistics calculated for %d thresholds\n', nThresholds);

%% =========================================================================
%% Define Temporal Windows for Visualization
%% =========================================================================
fprintf('\n=== Defining Temporal Windows ===\n');

temporalWindows = struct();
temporalWindows(1).name = 'WholeTrial';
temporalWindows(1).frames = [];  % Empty = use all frames
temporalWindows(1).label = 'Whole Trial';

temporalWindows(2).name = 'PreStim';
temporalWindows(2).frames = 1:80;
temporalWindows(2).label = 'Pre-Stimulus (1-80)';

temporalWindows(3).name = 'ImmediatePreStim';
temporalWindows(3).frames = 75:80;
temporalWindows(3).label = 'Immediate Pre-Stim (75-80)';

temporalWindows(4).name = 'PreStimOnset';
temporalWindows(4).frames = 75:85;
temporalWindows(4).label = 'Pre-Stim + Onset (75-85)';

temporalWindows(5).name = 'StimPeriod';
temporalWindows(5).frames = 81:100;
temporalWindows(5).label = 'Stimulus Period (81-100)';

fprintf('Defined %d temporal windows\n', length(temporalWindows));

%% =========================================================================
%% Visualization Loop: Iterate Through Temporal Windows
%% =========================================================================

for tw = 1:length(temporalWindows)
    windowName = temporalWindows(tw).name;
    windowFrames = temporalWindows(tw).frames;
    windowLabel = temporalWindows(tw).label;

    fprintf('\n=== Creating Visualizations for: %s ===\n', windowLabel);

    % Recalculate summary statistics for this temporal window
    Summary_windowed = recalculate_summary_for_window(Results, windowFrames, conditions, metrics, thresholds);

%% =========================================================================
%% Visualization 1: Line Plots (Metric vs Threshold)
%% =========================================================================
fprintf('  Creating Line Plots...\n');

for m = 1:length(metrics)
    metric = metrics{m};
    metricName = strrep(metric, '_BIN_AN', '');

    figure('Name', sprintf('Threshold Sensitivity - %s (%s)', metricName, windowLabel), 'Color', 'w');
    hold on;

    % Define colors from params (consistent with codebase plotting functions)
    colors = struct();
    for c_idx = 1:length(conditions)
        cond = conditions{c_idx};
        if isfield(params, cond)
            colors.(cond) = params.(cond);
        else
            % Default to gray if condition not found in params
            colors.(cond) = [0.5 0.5 0.5];
        end
    end
    % Define Overall trend color (typically black or dark gray)
    colors.Overall = [0.2 0.2 0.2];

    % Plot each condition
    for c = 1:length(conditions)
        condition = conditions{c};

        % Extract data for this condition across thresholds
        medians = zeros(nThresholds, 1);
        errors = zeros(nThresholds, 1);

        for t_idx = 1:nThresholds
            if isfield(Summary_windowed(t_idx), condition) && ...
               isfield(Summary_windowed(t_idx).(condition), metric)
                medians(t_idx) = Summary_windowed(t_idx).(condition).(metric).median;
                errors(t_idx) = Summary_windowed(t_idx).(condition).(metric).sem;
            else
                medians(t_idx) = NaN;
                errors(t_idx) = NaN;
            end
        end

        % Plot with error bars
        errorbar(thresholds, medians, errors, '-o', 'LineWidth', 2, ...
            'MarkerSize', 8, 'Color', colors.(condition), ...
            'DisplayName', condition);
    end

    % Plot overall trend
    overallMedians = zeros(nThresholds, 1);
    overallErrors = zeros(nThresholds, 1);

    for t_idx = 1:nThresholds
        overallMedians(t_idx) = Summary_windowed(t_idx).Overall.(metric).median;
        overallErrors(t_idx) = Summary_windowed(t_idx).Overall.(metric).sem;
    end

    errorbar(thresholds, overallMedians, overallErrors, '-o', 'LineWidth', 3, ...
        'MarkerSize', 10, 'Color', colors.Overall, ...
        'DisplayName', 'Overall', 'LineStyle', '--');

    % Format plot
    xlabel('Binarization Threshold (σ)', 'FontSize', 12);
    ylabel(sprintf('%s (BIN\\_AN)', metricName), 'FontSize', 12);
    title(sprintf('Threshold Sensitivity: %s (%s)', metricName, windowLabel), 'FontSize', 14);
    legend('Location', 'best');
    grid on;
    set(gca, 'FontSize', 11);

    hold off;
end

%% =========================================================================
%% Visualization 2: Box Plots (Conditions Comparison at Each Threshold)
%% =========================================================================
fprintf('  Creating Box Plots...\n');

    % Prepare color matrix from params (consistent across all subplots)
    color_matrix = zeros(length(conditions), 3);
    for c = 1:length(conditions)
        if isfield(params, conditions{c})
            color_matrix(c, :) = params.(conditions{c});
        else
            color_matrix(c, :) = [0.5 0.5 0.5];
        end
    end

    for t_idx = 1:nThresholds
        current_threshold = thresholds(t_idx);

        figure('Name', sprintf('Threshold %.1f - Condition Comparison (%s)', current_threshold, windowLabel), 'Color', 'w');

    nMetrics = length(metrics);

    for m = 1:nMetrics
        metric = metrics{m};
        metricName = strrep(metric, '_BIN_AN', '');

        subplot(2, 2, m);
        hold on;

        % Collect all recording-level medians for all conditions
        data_matrix = [];
        group_idx = [];

        for c = 1:length(conditions)
            condition = conditions{c};

            if isfield(Summary_windowed(t_idx), condition) && ...
               isfield(Summary_windowed(t_idx).(condition), metric) && ...
               isfield(Summary_windowed(t_idx).(condition).(metric), 'recordingMedians')

                recording_medians = Summary_windowed(t_idx).(condition).(metric).recordingMedians;

                if ~isempty(recording_medians)
                    % Add recording medians to data matrix
                    data_matrix = [data_matrix; recording_medians(:)];
                    % Assign group index for this condition
                    group_idx = [group_idx; c * ones(length(recording_medians), 1)];
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
        ylabel(sprintf('%s', metricName), 'FontSize', 10);
        title(sprintf('%s (Threshold = %.1fσ)', metricName, current_threshold), 'FontSize', 11);
        grid on;
        box on;

        hold off;
    end

    % Add overall title
        sgtitle(sprintf('Condition Comparison at Threshold = %.1fσ (%s)', current_threshold, windowLabel), ...
            'FontSize', 14, 'FontWeight', 'bold');
    end

    %% Save figures for this temporal window
    fprintf('  Saving figures for %s...\n', windowLabel);

    % Define save path for this temporal window
    savePathBase = 'Fig. 5 Model\ThresholdAnalysis\Sensitivity\BIN_AN_P1\';
    savePathWindow = fullfile(savePathBase, windowName);

    % Create directory if it doesn't exist
    if ~exist(savePathWindow, 'dir')
        mkdir(savePathWindow);
    end

    % Save all figures created in this temporal window
    figHandles = findall(0, 'Type', 'figure');
    for i = 1:length(figHandles)
        figName = get(figHandles(i), 'Name');
        if ~isempty(figName) && contains(figName, windowLabel)
            % Clean filename (remove special characters)
            figName_clean = strrep(figName, ':', '_');
            figName_clean = strrep(figName_clean, '(', '');
            figName_clean = strrep(figName_clean, ')', '');
            savePathFull = fullfile(savePathWindow, figName_clean);

            % Save as PNG
            saveas(figHandles(i), [savePathFull '.png']);

            % Save as FIG
            saveas(figHandles(i), [savePathFull '.fig']);

            fprintf('    Saved: %s\n', figName_clean);

            % Close figure to free memory
            close(figHandles(i));
        end
    end

end  % End temporal window loop

%% =========================================================================
%% Save Results Structure
%% =========================================================================
fprintf('\n=== Saving Results Structure ===\n');

% Save results structure (save once, contains all data)
savePathBase = 'Fig. 5 Model\ThresholdAnalysis\Sensitivity\BIN_AN_P1\';
if ~exist(savePathBase, 'dir')
    mkdir(savePathBase);
end

% Save results structure
save(fullfile(savePathBase, 'ThresholdSensitivity_Results.mat'), 'Results', 'Summary', 'thresholds', 'conditions', 'temporalWindows');

fprintf('\nAll results saved to: %s\n', savePathBase);
fprintf('Figures organized by temporal window in subdirectories\n');
fprintf('\n=== Threshold Sensitivity Analysis Complete ===\n');

%% =========================================================================
%% Helper Functions (from Figure5_ActivityEntropyDispersion_Comparison.m)
%% =========================================================================

%% Helper Function: Recalculate Summary for Temporal Window
function Summary_windowed = recalculate_summary_for_window(Results, windowFrames, conditions, metrics, thresholds)
% RECALCULATE_SUMMARY_FOR_WINDOW - Recalculate summary statistics for a specific temporal window
%
% INPUTS:
%   Results      - Full results structure with all data
%   windowFrames - Frame indices to include (empty = all frames)
%   conditions   - Cell array of condition names
%   metrics      - Cell array of metric names
%   thresholds   - Array of threshold values
%
% OUTPUTS:
%   Summary_windowed - Summary structure for this temporal window

    nThresholds = length(thresholds);
    Summary_windowed = struct();

    for t_idx = 1:nThresholds
        Summary_windowed(t_idx).threshold = thresholds(t_idx);

        % Initialize overall data collection
        for m = 1:length(metrics)
            metric = metrics{m};
            Summary_windowed(t_idx).Overall.(metric).AllData = [];
        end

        % Process each condition
        for c = 1:length(conditions)
            condition = conditions{c};

            if ~isfield(Results(t_idx), condition)
                continue;
            end

            % Process each metric
            for m = 1:length(metrics)
                metric = metrics{m};

                if ~isfield(Results(t_idx).(condition), metric)
                    continue;
                end

                % Get all recordings for this condition
                recordingData = Results(t_idx).(condition).(metric);

                if isempty(recordingData)
                    continue;
                end

                % Calculate median across recordings (windowed)
                recordingMedians = zeros(length(recordingData), 1);
                for rec = 1:length(recordingData)
                    data = recordingData{rec};  % [nTrials × nTimepoints]

                    % Apply temporal window filter
                    if ~isempty(windowFrames)
                        % Check if windowFrames are within bounds
                        validFrames = windowFrames(windowFrames <= size(data, 2));
                        if ~isempty(validFrames)
                            data_windowed = data(:, validFrames);
                        else
                            data_windowed = [];
                        end
                    else
                        data_windowed = data;  % Use all frames
                    end

                    % Calculate median for this recording
                    if ~isempty(data_windowed)
                        recordingMedians(rec) = median(data_windowed(:), 'omitnan');
                    else
                        recordingMedians(rec) = NaN;
                    end
                end

                % Remove NaN values
                recordingMedians = recordingMedians(~isnan(recordingMedians));

                if isempty(recordingMedians)
                    continue;
                end

                % Store condition-level summary
                Summary_windowed(t_idx).(condition).(metric).median = median(recordingMedians, 'omitnan');
                Summary_windowed(t_idx).(condition).(metric).iqr = iqr(recordingMedians);
                Summary_windowed(t_idx).(condition).(metric).sem = std(recordingMedians, 'omitnan') / sqrt(length(recordingMedians));
                Summary_windowed(t_idx).(condition).(metric).n = length(recordingMedians);
                Summary_windowed(t_idx).(condition).(metric).recordingMedians = recordingMedians;

                % Add to overall pool
                Summary_windowed(t_idx).Overall.(metric).AllData = ...
                    [Summary_windowed(t_idx).Overall.(metric).AllData; recordingMedians];
            end
        end

        % Calculate overall statistics (across all conditions)
        for m = 1:length(metrics)
            metric = metrics{m};
            allData = Summary_windowed(t_idx).Overall.(metric).AllData;

            if ~isempty(allData)
                Summary_windowed(t_idx).Overall.(metric).median = median(allData, 'omitnan');
                Summary_windowed(t_idx).Overall.(metric).iqr = iqr(allData);
                Summary_windowed(t_idx).Overall.(metric).sem = std(allData, 'omitnan') / sqrt(length(allData));
                Summary_windowed(t_idx).Overall.(metric).n = length(allData);
            end
        end
    end
end

%% Helper Function: Calculate Entropy from Grid Data
function [ent_raw, ent_rz, ent_tz, entZ_raw, entZ_rz, entZ_tz, ent_bin, ent_bin_an, ent_bin_all] = ...
    calculate_entropy_from_grid(gridData, gridDimensions, animalMean, animalStd, conditionMean, conditionStd, binarisation_threshold)
% CALCULATE_ENTROPY_FROM_GRID - Calculate population entropy from Grid40 data

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
        error('Expected Grid data to have 4 dimensions: [gridY, gridX, nTimepoints, nTrials]');
    end

    % Reshape grid to treat each grid cell as a "neuron"
    nGridCells = gridY * gridX;
    gridData_reshaped = reshape(gridData, [nGridCells, nTimepoints, nTrials]);

    % Initialize output arrays
    ent_raw = zeros(nTrials, nTimepoints);
    entZ_raw = zeros(nTrials, nTimepoints);

    % Calculate entropy for each trial and timepoint
    for trial = 1:nTrials
        for t = 1:nTimepoints
            % Extract grid cell activities at this trial/timepoint
            activity = gridData_reshaped(:, t, trial);

            % Calculate population entropy (using population_entropy function)
            ent_raw(trial, t) = population_entropy(activity);

            % Calculate entropy on z-scored data
            if std(activity) > 0
                activity_z = (activity - mean(activity)) / std(activity);
            else
                activity_z = activity;
            end
            entZ_raw(trial, t) = population_entropy(activity_z);
        end
    end

    % Calculate BINARISED entropy (per-recording)
    ent_bin = zeros(nTrials, nTimepoints);

    for trial = 1:nTrials
        for t = 1:nTimepoints
            activity = gridData_reshaped(:, t, trial);

            % Binarise: z-score → threshold → binary
            if std(activity) > 0
                activity_z = (activity - mean(activity)) / std(activity);
                activity_bin = double(activity_z > binarisation_threshold);
            else
                activity_bin = zeros(size(activity));
            end
            ent_bin(trial, t) = population_entropy(activity_bin);
        end
    end

    % Calculate BINARISED entropy with PER-ANIMAL thresholds (BIN_AN)
    ent_bin_an = zeros(nTrials, nTimepoints);

    if nargin >= 4 && ~isempty(animalMean) && ~isempty(animalStd) && animalStd > 0
        for trial = 1:nTrials
            for t = 1:nTimepoints
                activity = gridData_reshaped(:, t, trial);

                % Binarise using per-animal statistics
                activity_z_animal = (activity - animalMean) / animalStd;
                activity_bin_an = double(activity_z_animal > binarisation_threshold);
                ent_bin_an(trial, t) = population_entropy(activity_bin_an);
            end
        end
    else
        % Fall back to per-recording
        ent_bin_an = ent_bin;
    end

    % Calculate BINARISED entropy with GLOBAL thresholds (BIN_ALL)
    ent_bin_all = zeros(nTrials, nTimepoints);

    if nargin >= 6 && ~isempty(conditionMean) && ~isempty(conditionStd) && conditionStd > 0
        for trial = 1:nTrials
            for t = 1:nTimepoints
                activity = gridData_reshaped(:, t, trial);

                % Binarise using GLOBAL statistics
                activity_z_global = (activity - conditionMean) / conditionStd;
                activity_bin_all = double(activity_z_global > binarisation_threshold);
                ent_bin_all(trial, t) = population_entropy(activity_bin_all);
            end
        end
    else
        % Fall back to per-recording
        ent_bin_all = ent_bin;
    end

    % Calculate trial-zscore (TZ)
    baselineFrames = 1:80;
    ent_tz = zeros(size(ent_raw));
    entZ_tz = zeros(size(entZ_raw));

    for trial = 1:nTrials
        baseline_ent = ent_raw(trial, baselineFrames);
        if std(baseline_ent) > 0
            ent_tz(trial, :) = (ent_raw(trial, :) - mean(baseline_ent)) / std(baseline_ent);
        else
            ent_tz(trial, :) = ent_raw(trial, :);
        end

        baseline_entZ = entZ_raw(trial, baselineFrames);
        if std(baseline_entZ) > 0
            entZ_tz(trial, :) = (entZ_raw(trial, :) - mean(baseline_entZ)) / std(baseline_entZ);
        else
            entZ_tz(trial, :) = entZ_raw(trial, :);
        end
    end

    % Calculate recording-zscore (RZ)
    ent_rz = (ent_raw - mean(ent_raw(:))) / std(ent_raw(:));
    entZ_rz = (entZ_raw - mean(entZ_raw(:))) / std(entZ_raw(:));
end

%% Helper Function: Calculate Dispersion from Grid Data
function [disp_raw, disp_rz, disp_tz, dispZ_raw, dispZ_rz, dispZ_tz, disp_bin, disp_bin_an, disp_bin_all] = ...
    calculate_dispersion_from_grid(gridData, gridDimensions, animalMean, animalStd, globalMean, globalStd, binarisation_threshold)
% CALCULATE_DISPERSION_FROM_GRID - Calculate spatial dispersion from Grid40 data

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
        error('Expected Grid data to have 4 dimensions: [gridY, gridX, nTimepoints, nTrials]');
    end

    % Generate grid coordinates
    [gridX_coords, gridY_coords] = meshgrid(1:gridX, 1:gridY);
    grid_coords = [gridY_coords(:), gridX_coords(:)];

    % Reshape grid data
    nGridCells = gridY * gridX;
    gridData_reshaped = reshape(gridData, [nGridCells, nTimepoints, nTrials]);

    % Initialize output arrays
    disp_raw = zeros(nTrials, nTimepoints);
    dispZ_raw = zeros(nTrials, nTimepoints);

    % Calculate WEIGHTED dispersion for continuous data
    for trial = 1:nTrials
        for t = 1:nTimepoints
            activity = gridData_reshaped(:, t, trial);
            disp_raw(trial, t) = calculate_weighted_dispersion(activity, grid_coords);

            if std(activity) > 0
                activity_z = (activity - mean(activity)) / std(activity);
            else
                activity_z = activity;
            end
            dispZ_raw(trial, t) = calculate_weighted_dispersion(activity_z, grid_coords);
        end
    end

    % Calculate BINARISED dispersion (BIN)
    disp_bin = zeros(nTrials, nTimepoints);

    for trial = 1:nTrials
        for t = 1:nTimepoints
            activity = gridData_reshaped(:, t, trial);

            if std(activity) > 0
                activity_z = (activity - mean(activity)) / std(activity);
                activity_bin = double(activity_z > binarisation_threshold);
            else
                activity_bin = zeros(size(activity));
            end

            disp_bin(trial, t) = calculate_binary_dispersion(activity_bin, grid_coords);
        end
    end

    % Calculate BINARISED dispersion with PER-ANIMAL thresholds (BIN_AN)
    disp_bin_an = zeros(nTrials, nTimepoints);

    if nargin >= 4 && ~isempty(animalMean) && ~isempty(animalStd) && animalStd > 0
        for trial = 1:nTrials
            for t = 1:nTimepoints
                activity = gridData_reshaped(:, t, trial);
                activity_z_animal = (activity - animalMean) / animalStd;
                activity_bin_an = double(activity_z_animal > binarisation_threshold);
                disp_bin_an(trial, t) = calculate_binary_dispersion(activity_bin_an, grid_coords);
            end
        end
    else
        disp_bin_an = disp_bin;
    end

    % Calculate BINARISED dispersion with GLOBAL thresholds (BIN_ALL)
    disp_bin_all = zeros(nTrials, nTimepoints);

    if nargin >= 6 && ~isempty(globalMean) && ~isempty(globalStd) && globalStd > 0
        for trial = 1:nTrials
            for t = 1:nTimepoints
                activity = gridData_reshaped(:, t, trial);
                activity_z_global = (activity - globalMean) / globalStd;
                activity_bin_all = double(activity_z_global > binarisation_threshold);
                disp_bin_all(trial, t) = calculate_binary_dispersion(activity_bin_all, grid_coords);
            end
        end
    else
        disp_bin_all = disp_bin;
    end

    % Calculate trial-zscore (TZ)
    baselineFrames = 1:80;
    disp_tz = zeros(size(disp_raw));
    dispZ_tz = zeros(size(dispZ_raw));

    for trial = 1:nTrials
        baseline_disp = disp_raw(trial, baselineFrames);
        if std(baseline_disp) > 0
            disp_tz(trial, :) = (disp_raw(trial, :) - mean(baseline_disp)) / std(baseline_disp);
        else
            disp_tz(trial, :) = disp_raw(trial, :);
        end

        baseline_dispZ = dispZ_raw(trial, baselineFrames);
        if std(baseline_dispZ) > 0
            dispZ_tz(trial, :) = (dispZ_raw(trial, :) - mean(baseline_dispZ)) / std(baseline_dispZ);
        else
            dispZ_tz(trial, :) = dispZ_raw(trial, :);
        end
    end

    % Calculate recording-zscore (RZ)
    disp_rz = (disp_raw - mean(disp_raw(:))) / std(disp_raw(:));
    dispZ_rz = (dispZ_raw - mean(dispZ_raw(:))) / std(dispZ_raw(:));
end

%% Sub-helper: Calculate weighted dispersion
function disp = calculate_weighted_dispersion(activity, grid_coords)
    activity_threshold = 0.01;
    active_mask = activity > activity_threshold;

    if sum(active_mask) == 0
        disp = NaN;
        return;
    end

    active_coords = grid_coords(active_mask, :);
    active_weights = activity(active_mask);

    total_weight = sum(active_weights);
    weighted_centroid = sum(active_coords .* active_weights, 1) / total_weight;

    distances = sqrt(sum((active_coords - weighted_centroid).^2, 2));
    disp = sum(distances .* active_weights) / total_weight;
end

%% Sub-helper: Calculate binary dispersion
function disp = calculate_binary_dispersion(activity_bin, grid_coords)
    active_inds = find(activity_bin > 0);

    if isempty(active_inds)
        disp = NaN;
        return;
    end

    act_pos = grid_coords(active_inds, :);
    mean_pos = mean(act_pos, 1);
    dist_v = sqrt(sum((act_pos - mean_pos).^2, 2));
    disp = mean(dist_v);
end

%% Helper Function: Calculate Active Cells from Grid Data
function [actcells_raw, actcells_rz, actcells_tz, actcellsZ_raw, actcellsZ_rz, actcellsZ_tz, actcells_bin, actcells_bin_an, actcells_bin_all] = ...
    calculate_active_cells_from_grid(gridData, gridDimensions, animalMean, animalStd, globalMean, globalStd, binarisation_threshold)
% CALCULATE_ACTIVE_CELLS_FROM_GRID - Calculate number of active grid cells

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
        error('Expected Grid data to have 4 dimensions: [gridY, gridX, nTimepoints, nTrials]');
    end

    % Reshape grid data
    nGridCells = gridY * gridX;
    gridData_reshaped = reshape(gridData, [nGridCells, nTimepoints, nTrials]);

    activity_threshold = 0.01;

    % Initialize output arrays
    actcells_raw = zeros(nTrials, nTimepoints);
    actcellsZ_raw = zeros(nTrials, nTimepoints);

    % Calculate active cells for RAW continuous data
    for trial = 1:nTrials
        for t = 1:nTimepoints
            activity = gridData_reshaped(:, t, trial);
            actcells_raw(trial, t) = sum(activity > activity_threshold);

            if std(activity) > 0
                activity_z = (activity - mean(activity)) / std(activity);
            else
                activity_z = activity;
            end
            actcellsZ_raw(trial, t) = sum(activity_z > activity_threshold);
        end
    end

    % Calculate BINARISED active cells (BIN)
    actcells_bin = zeros(nTrials, nTimepoints);

    for trial = 1:nTrials
        for t = 1:nTimepoints
            activity = gridData_reshaped(:, t, trial);

            if std(activity) > 0
                activity_z = (activity - mean(activity)) / std(activity);
                activity_bin = double(activity_z > binarisation_threshold);
            else
                activity_bin = zeros(size(activity));
            end

            actcells_bin(trial, t) = sum(activity_bin);
        end
    end

    % Calculate BINARISED active cells with PER-ANIMAL thresholds (BIN_AN)
    actcells_bin_an = zeros(nTrials, nTimepoints);

    if nargin >= 4 && ~isempty(animalMean) && ~isempty(animalStd) && animalStd > 0
        for trial = 1:nTrials
            for t = 1:nTimepoints
                activity = gridData_reshaped(:, t, trial);
                activity_z_animal = (activity - animalMean) / animalStd;
                activity_bin_an = double(activity_z_animal > binarisation_threshold);
                actcells_bin_an(trial, t) = sum(activity_bin_an);
            end
        end
    else
        actcells_bin_an = actcells_bin;
    end

    % Calculate BINARISED active cells with GLOBAL thresholds (BIN_ALL)
    actcells_bin_all = zeros(nTrials, nTimepoints);

    if nargin >= 6 && ~isempty(globalMean) && ~isempty(globalStd) && globalStd > 0
        for trial = 1:nTrials
            for t = 1:nTimepoints
                activity = gridData_reshaped(:, t, trial);
                activity_z_global = (activity - globalMean) / globalStd;
                activity_bin_all = double(activity_z_global > binarisation_threshold);
                actcells_bin_all(trial, t) = sum(activity_bin_all);
            end
        end
    else
        actcells_bin_all = actcells_bin;
    end

    % Calculate trial-zscore (TZ)
    baselineFrames = 1:80;
    actcells_tz = zeros(size(actcells_raw));
    actcellsZ_tz = zeros(size(actcellsZ_raw));

    for trial = 1:nTrials
        baseline_actcells = actcells_raw(trial, baselineFrames);
        if std(baseline_actcells) > 0
            actcells_tz(trial, :) = (actcells_raw(trial, :) - mean(baseline_actcells)) / std(baseline_actcells);
        else
            actcells_tz(trial, :) = actcells_raw(trial, :);
        end

        baseline_actcellsZ = actcellsZ_raw(trial, baselineFrames);
        if std(baseline_actcellsZ) > 0
            actcellsZ_tz(trial, :) = (actcellsZ_raw(trial, :) - mean(baseline_actcellsZ)) / std(baseline_actcellsZ);
        else
            actcellsZ_tz(trial, :) = actcellsZ_raw(trial, :);
        end
    end

    % Calculate recording-zscore (RZ)
    if std(actcells_raw(:)) > 0
        actcells_rz = (actcells_raw - mean(actcells_raw(:))) / std(actcells_raw(:));
    else
        actcells_rz = actcells_raw;
    end

    if std(actcellsZ_raw(:)) > 0
        actcellsZ_rz = (actcellsZ_raw - mean(actcellsZ_raw(:))) / std(actcellsZ_raw(:));
    else
        actcellsZ_rz = actcellsZ_raw;
    end
end

%% Helper Function: Calculate Moran's I (BIN_AN only)
function morans_bin_an = calculate_moransI_BIN_AN(gridData, gridDimensions, animalMean, animalStd, binarisation_threshold)
% CALCULATE_MORANSI_BIN_AN - Calculate Moran's I spatial autocorrelation

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
        error('Expected Grid data to have 4 dimensions: [gridY, gridX, nTimepoints, nTrials]');
    end

    % Create spatial weight matrix
    valueMap = rand(gridY, gridX);
    currDist = 1;

    distanceMat = squareform(mL_distanceMat(valueMap(:,:,1)));
    uniqueDistances = unique(distanceMat);
    uniqueDistances(uniqueDistances == 0) = [];

    currDistInds = ismember(distanceMat, uniqueDistances(1:currDist));
    currWeightMat = zeros(size(distanceMat));
    currWeightMat(currDistInds) = distanceMat(currDistInds);
    currWeightMat(currWeightMat == inf) = 0;

    % Initialize output array
    morans_bin_an = zeros(nTrials, nTimepoints);

    if nargin >= 3 && ~isempty(animalMean) && ~isempty(animalStd) && animalStd > 0
        for trial = 1:nTrials
            for t = 1:nTimepoints
                grid_2D = gridData(:, :, t, trial);
                grid_2D_z = (grid_2D - animalMean) / animalStd;
                grid_2D_bin = double(grid_2D_z > binarisation_threshold);
                morans_bin_an(trial, t) = mL_moransI(grid_2D_bin, currWeightMat);
            end
        end
    else
        morans_bin_an(:) = NaN;
    end
end
