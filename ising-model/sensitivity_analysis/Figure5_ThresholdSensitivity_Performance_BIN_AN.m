%% =========================================================================
%% Threshold Sensitivity Analysis - Performance Comparison (Hit vs Miss)
%% =========================================================================
% This script analyzes how performance-related metrics (Entropy, Dispersion,
% Active Cells, Moran's I) change with different binarization threshold values
% for Hit vs Miss trials, using BIN_AN normalization.
%
% Purpose: Understand sensitivity of spatial and population metrics to the
% binarization threshold parameter for different behavioral outcomes.
%
% Tested Thresholds: 1.0, 1.5, 2.0 (σ units for z-score binarization)
% Normalization: BIN_AN (per-animal statistics)
% Position: Position1 (P1) only
% Condition: Expert only
% Performance Types: Hit, Miss, Nonresponsive

% Load data structures
% load(mba_p('RawData3.mat'),'ActivityData');
% load(mba_p('RawData3.mat'),'params');
% load(mba_p('RawData3.mat'),'Performance');
% load(mba_p('RawData3.mat'),'Stimuli');

% Define Skip arrays (recordings to exclude from analysis)
Skip = [];
Skip.Naive = [1 9 10 16];
Skip.Beginner = [1 6 7 11];
Skip.Expert = [1 4 12 13 14];
Skip.ExpertRandom = [1];
Skip.ExpertAll = [1,4,5,13, 20,21,22,23,24,25,26];
Skip.NoSpout = [1 4 9 10 11 13 14];

% Analysis parameters
condition = 'Expert';  % Single condition to analyze
MissNonresponsive = 2; % 0 = Miss, 1 = Nonresponsive, 2 = Miss & Nonresponsive (three-way)

% Define threshold values to test
thresholds = [1.0, 1.5, 2.0];
nThresholds = length(thresholds);

% Define performance state colors
params.Hit = [0, 0, 0];             % Black for hit trials
params.Miss = [0.8, 0, 0];          % Red for miss trials
params.NonResponsive = [1, 0.1, 0.8]; % Neon pink for nonresponsive trials

% Grid parameters
gridSize = 40;
gridDimensions = [13 26];  % Grid structure: 13 rows × 26 columns

% Moran's I calculation flag
include_MoransI = true;

% Performance types
if MissNonresponsive == 2
    perfTypes = {'Hit', 'Miss', 'Nonresponsive'};
else
    perfTypes = {'Hit', 'Miss'};
end

fprintf('\n=== Threshold Sensitivity Analysis (Performance Comparison - BIN_AN Mode) ===\n');
fprintf('Condition: %s\n', condition);
fprintf('Performance types: %s\n', strjoin(perfTypes, ', '));
fprintf('Testing thresholds: ');
fprintf('%.1f ', thresholds);
fprintf('\n');

%% =========================================================================
%% Main Analysis: Loop Through Thresholds
%% =========================================================================

% Initialize results structure
% Results(threshold_idx).(perfType).Metric.RecordingData = cell array
Results = struct();

% Get basic data info
if ~isfield(ActivityData, condition)
    error('Condition %s not found in ActivityData', condition);
end
nRecs = length(ActivityData.(condition));

conditionIndividual = [condition 'Individual'];
if ~isfield(Grid40, conditionIndividual)
    error('No Grid40 data found for %s', conditionIndividual);
end
nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);

%% Calculate per-animal statistics for BIN_AN (once, shared across all thresholds)
fprintf('\nCalculating per-animal statistics for BIN_AN...\n');
animalStats_P1 = struct();

if isfield(Rec, condition) && height(Rec.(condition)) >= nRecs
    animalIDs = Rec.(condition).AnimalID;
    uniqueAnimalIDs = unique(animalIDs);

    for a = 1:length(uniqueAnimalIDs)
        currentAnimalID = uniqueAnimalIDs{a};
        validFieldName = ['Animal_' currentAnimalID];

        % Find all recordings for this animal
        animalRecordings = find(strcmp(animalIDs, currentAnimalID));

        % Collect all grid data for this animal (P1 only)
        allGridData_P1 = [];

        for rec_idx = 1:length(animalRecordings)
            r = animalRecordings(rec_idx);

            if r > min(nRecs, nRecsGrid) || ismember(r, Skip.(condition))
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
            animalStats_P1.(validFieldName).mean = mean(allGridData_P1(:));
            animalStats_P1.(validFieldName).std = std(allGridData_P1(:));
        end
    end
    fprintf('Per-animal statistics calculated for %d animals\n', length(uniqueAnimalIDs));
else
    animalIDs = {};
end

%% Loop through thresholds
for t_idx = 1:nThresholds
    current_threshold = thresholds(t_idx);
    fprintf('\n=== Processing Threshold = %.1f σ ===\n', current_threshold);

    % Initialize storage for this threshold
    for perfType = perfTypes
        perfField = perfType{1};
        Results(t_idx).(perfField).Entropy_BIN_AN = {};
        Results(t_idx).(perfField).Dispersion_BIN_AN = {};
        Results(t_idx).(perfField).ActiveCells_BIN_AN = {};
        if include_MoransI
            Results(t_idx).(perfField).MoransI_BIN_AN = {};
        end
    end

    % Loop through recordings
    for r = 1:min(nRecs, nRecsGrid)
        % Check if this recording should be skipped
        if ismember(r, Skip.(condition))
            continue;
        end

        % Check data availability
        if r > length(Grid40.(conditionIndividual).AllNeurons)
            continue;
        end
        if r > length(Performance.(condition))
            continue;
        end
        if r > length(Stimuli.(condition))
            continue;
        end

        % Get animal statistics for this recording
        animalMean_P1 = [];
        animalStd_P1 = [];
        if ~isempty(animalIDs) && r <= length(animalIDs)
            currentAnimalID_r = animalIDs{r};
            validFieldName_r = ['Animal_' currentAnimalID_r];

            if isfield(animalStats_P1, validFieldName_r)
                animalMean_P1 = animalStats_P1.(validFieldName_r).mean;
                animalStd_P1 = animalStats_P1.(validFieldName_r).std;
            end
        end

        % Get trial classification (Hit vs Miss vs Nonresponsive)
        % Get P1 trial numbers
        if ~isfield(Stimuli.(condition)(r), 'TrialsPosition1') || isempty(Stimuli.(condition)(r).TrialsPosition1)
            continue;  % No P1 trials
        end
        p1TrialsAbsolute = Stimuli.(condition)(r).TrialsPosition1;

        % Get Hit trial indices
        hitTrialsAbsolute = [];
        if isfield(Performance.(condition)(r), 'hit')
            hitTrialsAbsolute = Performance.(condition)(r).hit;
        elseif isfield(Performance.(condition)(r), 'hitAll')
            hitTrialsAbsolute = Performance.(condition)(r).hitAll;
        end

        % Get Miss/Nonresponsive trial indices
        missTrialsAbsolute = [];
        nonrespTrialsAbsolute = [];

        if MissNonresponsive == 0
            % Miss trials only
            if isfield(Performance.(condition)(r), 'miss')
                missTrialsAbsolute = Performance.(condition)(r).miss;
            elseif isfield(Performance.(condition)(r), 'missAll')
                missTrialsAbsolute = Performance.(condition)(r).missAll;
            end
        elseif MissNonresponsive == 1
            % Nonresponsive trials only
            if isfield(Performance.(condition)(r), 'nonresponsiveTrials')
                missTrialsAbsolute = Performance.(condition)(r).nonresponsiveTrials;
            elseif isfield(Performance.(condition)(r), 'nonresponsiveTrialsAll')
                missTrialsAbsolute = Performance.(condition)(r).nonresponsiveTrialsAll;
            end
        else  % MissNonresponsive == 2
            % Three-way: Hit vs Miss vs Nonresponsive
            miss = [];
            nonresp = [];
            if isfield(Performance.(condition)(r), 'miss')
                miss = Performance.(condition)(r).miss;
            elseif isfield(Performance.(condition)(r), 'missAll')
                miss = Performance.(condition)(r).missAll;
            end
            if isfield(Performance.(condition)(r), 'nonresponsiveTrials')
                nonresp = Performance.(condition)(r).nonresponsiveTrials;
            elseif isfield(Performance.(condition)(r), 'nonresponsiveTrialsAll')
                nonresp = Performance.(condition)(r).nonresponsiveTrialsAll;
            end
            missTrialsAbsolute = setdiff(miss, nonresp);
            nonrespTrialsAbsolute = nonresp;
        end

        % Map to P1-relative indices
        p1HitTrialsAbsolute = intersect(p1TrialsAbsolute, hitTrialsAbsolute);
        p1MissTrialsAbsolute = intersect(p1TrialsAbsolute, missTrialsAbsolute);
        p1NonrespTrialsAbsolute = intersect(p1TrialsAbsolute, nonrespTrialsAbsolute);

        [~, hitTrialIndices] = ismember(p1HitTrialsAbsolute, p1TrialsAbsolute);
        hitTrialIndices = hitTrialIndices(hitTrialIndices > 0);

        [~, missTrialIndices] = ismember(p1MissTrialsAbsolute, p1TrialsAbsolute);
        missTrialIndices = missTrialIndices(missTrialIndices > 0);

        [~, nonrespTrialIndices] = ismember(p1NonrespTrialsAbsolute, p1TrialsAbsolute);
        nonrespTrialIndices = nonrespTrialIndices(nonrespTrialIndices > 0);

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
                % Process each performance type
                trialSets = struct();
                trialSets.Hit = hitTrialIndices;
                trialSets.Miss = missTrialIndices;
                if MissNonresponsive == 2
                    trialSets.Nonresponsive = nonrespTrialIndices;
                end

                for perfType = perfTypes
                    perfField = perfType{1};
                    trialInds = trialSets.(perfField);

                    if isempty(trialInds) || length(trialInds) == 0
                        continue;  % No trials of this type
                    end

                    % Extract grid data for this performance type
                    gridData_P1_perf = gridData_P1(:, :, :, trialInds);

                    % Ensure 4D
                    sz = size(gridData_P1_perf);
                    if length(sz) < 4
                        gridData_P1_perf = reshape(gridData_P1_perf, [sz, 1]);
                    end

                    % Calculate metrics (BIN_AN only)
                    [~, ~, ~, ~, ~, ~, ~, ent_bin_an, ~] = ...
                        calculate_entropy_from_grid(gridData_P1_perf, gridDimensions, ...
                        animalMean_P1, animalStd_P1, [], [], current_threshold);
                    Results(t_idx).(perfField).Entropy_BIN_AN{end+1} = ent_bin_an;

                    [~, ~, ~, ~, ~, ~, ~, disp_bin_an, ~] = ...
                        calculate_dispersion_from_grid(gridData_P1_perf, gridDimensions, ...
                        animalMean_P1, animalStd_P1, [], [], current_threshold);
                    Results(t_idx).(perfField).Dispersion_BIN_AN{end+1} = disp_bin_an;

                    [~, ~, ~, ~, ~, ~, ~, actcells_bin_an, ~] = ...
                        calculate_active_cells_from_grid(gridData_P1_perf, gridDimensions, ...
                        animalMean_P1, animalStd_P1, [], [], current_threshold);
                    Results(t_idx).(perfField).ActiveCells_BIN_AN{end+1} = actcells_bin_an;

                    if include_MoransI
                        morans_bin_an = calculate_moransI_BIN_AN(gridData_P1_perf, ...
                            gridDimensions, animalMean_P1, animalStd_P1, current_threshold);
                        Results(t_idx).(perfField).MoransI_BIN_AN{end+1} = morans_bin_an;
                    end
                end
            end
        end
    end

    fprintf('Threshold %.1f complete\n', current_threshold);
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

    % Process each performance type
    for perfType = perfTypes
        perfField = perfType{1};

        % Process each metric
        for m = 1:length(metrics)
            metric = metrics{m};

            if ~isfield(Results(t_idx), perfField) || ...
               ~isfield(Results(t_idx).(perfField), metric)
                continue;
            end

            % Get all recordings for this performance type
            recordingData = Results(t_idx).(perfField).(metric);

            if isempty(recordingData)
                continue;
            end

            % Calculate median across recordings
            recordingMedians = zeros(length(recordingData), 1);
            for rec = 1:length(recordingData)
                data = recordingData{rec};
                recordingMedians(rec) = median(data(:), 'omitnan');
            end

            % Store summary
            Summary(t_idx).(perfField).(metric).median = median(recordingMedians, 'omitnan');
            Summary(t_idx).(perfField).(metric).iqr = iqr(recordingMedians);
            Summary(t_idx).(perfField).(metric).sem = std(recordingMedians, 'omitnan') / sqrt(length(recordingMedians));
            Summary(t_idx).(perfField).(metric).n = length(recordingMedians);
            Summary(t_idx).(perfField).(metric).recordingMedians = recordingMedians;
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
    Summary_windowed = recalculate_summary_for_window_perf(Results, windowFrames, perfTypes, metrics, thresholds);

% Visualization 1: Line Plots (Metric vs Threshold)
fprintf('  Creating Line Plots...\n');

% Use explicitly defined performance colors
colors_perf = struct();
colors_perf.Hit = params.Hit;
colors_perf.Miss = params.Miss;
colors_perf.Nonresponsive = params.NonResponsive;

    for m = 1:length(metrics)
        metric = metrics{m};
        metricName = strrep(metric, '_BIN_AN', '');

        figure('Name', sprintf('Threshold Sensitivity - %s (Hit vs Miss) (%s)', metricName, windowLabel), 'Color', 'w');
        hold on;

    % Plot each performance type
    for perfType = perfTypes
        perfField = perfType{1};

        % Extract data across thresholds
        medians = zeros(nThresholds, 1);
        errors = zeros(nThresholds, 1);

        for t_idx = 1:nThresholds
            if isfield(Summary_windowed(t_idx), perfField) && ...
               isfield(Summary_windowed(t_idx).(perfField), metric)
                medians(t_idx) = Summary_windowed(t_idx).(perfField).(metric).median;
                errors(t_idx) = Summary_windowed(t_idx).(perfField).(metric).sem;
            else
                medians(t_idx) = NaN;
                errors(t_idx) = NaN;
            end
        end

        % Plot with error bars
        errorbar(thresholds, medians, errors, '-o', 'LineWidth', 2, ...
            'MarkerSize', 8, 'Color', colors_perf.(perfField), ...
            'DisplayName', perfField);
    end

        % Format plot
        xlabel('Binarization Threshold (σ)', 'FontSize', 12);
        ylabel(sprintf('%s (BIN\_AN)', metricName), 'FontSize', 12);
        title(sprintf('Threshold Sensitivity: %s (Hit vs Miss) (%s)', metricName, windowLabel), 'FontSize', 14);
        legend('Location', 'best');
        grid on;
        set(gca, 'FontSize', 11);

        hold off;
    end

% Visualization 2: Box Plots (Performance Type Comparison at Each Threshold)
fprintf('  Creating Box Plots...\n');

    % Prepare color matrix for performance types
    color_matrix = zeros(length(perfTypes), 3);
    for p = 1:length(perfTypes)
        perfField = perfTypes{p};
        color_matrix(p, :) = colors_perf.(perfField);
    end

    for t_idx = 1:nThresholds
        current_threshold = thresholds(t_idx);

        figure('Name', sprintf('Threshold %.1f - Performance Comparison (%s)', current_threshold, windowLabel), 'Color', 'w');

    nMetrics = length(metrics);

    for m = 1:nMetrics
        metric = metrics{m};
        metricName = strrep(metric, '_BIN_AN', '');

        subplot(2, 2, m);
        hold on;

        % Collect all recording-level medians for all performance types
        data_matrix = [];
        group_idx = [];

        for p = 1:length(perfTypes)
            perfField = perfTypes{p};

            if isfield(Summary_windowed(t_idx), perfField) && ...
               isfield(Summary_windowed(t_idx).(perfField), metric) && ...
               isfield(Summary_windowed(t_idx).(perfField).(metric), 'recordingMedians')

                recording_medians = Summary_windowed(t_idx).(perfField).(metric).recordingMedians;

                if ~isempty(recording_medians)
                    data_matrix = [data_matrix; recording_medians(:)];
                    group_idx = [group_idx; p * ones(length(recording_medians), 1)];
                end
            end
        end

        % Create daboxplot if data exists
        if ~isempty(data_matrix)
            daboxplot(data_matrix, 'groups', group_idx, 'colors', color_matrix, ...
                'xtlabels', perfTypes, 'scatter', 1, 'outliers', 1, ...
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
        sgtitle(sprintf('Performance Comparison at Threshold = %.1fσ (%s)', current_threshold, windowLabel), ...
            'FontSize', 14, 'FontWeight', 'bold');
    end

    % Save figures for this temporal window
    fprintf('  Saving figures for %s...\n', windowLabel);

    % Define save path for this temporal window
    savePathBase = 'Fig. 5 Model\ThresholdAnalysis\Sensitivity\Performance_BIN_AN_P1\';
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
savePathBase = 'Fig. 5 Model\ThresholdAnalysis\Sensitivity\Performance_BIN_AN_P1\';
if ~exist(savePathBase, 'dir')
    mkdir(savePathBase);
end

% Save results structure
save(fullfile(savePathBase, 'ThresholdSensitivity_Performance_Results.mat'), ...
    'Results', 'Summary', 'thresholds', 'perfTypes', 'condition', 'temporalWindows');

fprintf('\nAll results saved to: %s\n', savePathBase);
fprintf('Figures organized by temporal window in subdirectories\n');
fprintf('\n=== Threshold Sensitivity Analysis (Performance) Complete ===\n');

%% =========================================================================
%% Helper Functions (from Figure5_ActivityEntropyDispersion_Comparison.m)
%% =========================================================================

%% Helper Function: Recalculate Summary for Temporal Window (Performance)
function Summary_windowed = recalculate_summary_for_window_perf(Results, windowFrames, perfTypes, metrics, thresholds)
% RECALCULATE_SUMMARY_FOR_WINDOW_PERF - Recalculate summary statistics for a specific temporal window
%
% INPUTS:
%   Results      - Full results structure with all data
%   windowFrames - Frame indices to include (empty = all frames)
%   perfTypes    - Cell array of performance type names (Hit, Miss, Nonresponsive)
%   metrics      - Cell array of metric names
%   thresholds   - Array of threshold values
%
% OUTPUTS:
%   Summary_windowed - Summary structure for this temporal window

    nThresholds = length(thresholds);
    Summary_windowed = struct();

    for t_idx = 1:nThresholds
        Summary_windowed(t_idx).threshold = thresholds(t_idx);

        % Process each performance type
        for p = 1:length(perfTypes)
            perfField = perfTypes{p};

            if ~isfield(Results(t_idx), perfField)
                continue;
            end

            % Process each metric
            for m = 1:length(metrics)
                metric = metrics{m};

                if ~isfield(Results(t_idx).(perfField), metric)
                    continue;
                end

                % Get all recordings for this performance type
                recordingData = Results(t_idx).(perfField).(metric);

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

                % Store performance-type-level summary
                Summary_windowed(t_idx).(perfField).(metric).median = median(recordingMedians, 'omitnan');
                Summary_windowed(t_idx).(perfField).(metric).iqr = iqr(recordingMedians);
                Summary_windowed(t_idx).(perfField).(metric).sem = std(recordingMedians, 'omitnan') / sqrt(length(recordingMedians));
                Summary_windowed(t_idx).(perfField).(metric).n = length(recordingMedians);
                Summary_windowed(t_idx).(perfField).(metric).recordingMedians = recordingMedians;
            end
        end
    end
end

%% Helper Function: Calculate Entropy from Grid Data
function [ent_raw, ent_rz, ent_tz, entZ_raw, entZ_rz, entZ_tz, ent_bin, ent_bin_an, ent_bin_all] = ...
    calculate_entropy_from_grid(gridData, gridDimensions, animalMean, animalStd, conditionMean, conditionStd, binarisation_threshold)

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
        error('Expected Grid data to have 4 dimensions: [gridY, gridX, nTimepoints, nTrials]');
    end

    % Reshape grid
    nGridCells = gridY * gridX;
    gridData_reshaped = reshape(gridData, [nGridCells, nTimepoints, nTrials]);

    % Initialize output
    ent_raw = zeros(nTrials, nTimepoints);
    entZ_raw = zeros(nTrials, nTimepoints);

    % Calculate entropy
    for trial = 1:nTrials
        for t = 1:nTimepoints
            activity = gridData_reshaped(:, t, trial);
            ent_raw(trial, t) = population_entropy(activity);

            if std(activity) > 0
                activity_z = (activity - mean(activity)) / std(activity);
            else
                activity_z = activity;
            end
            entZ_raw(trial, t) = population_entropy(activity_z);
        end
    end

    % BIN
    ent_bin = zeros(nTrials, nTimepoints);
    for trial = 1:nTrials
        for t = 1:nTimepoints
            activity = gridData_reshaped(:, t, trial);
            if std(activity) > 0
                activity_z = (activity - mean(activity)) / std(activity);
                activity_bin = double(activity_z > binarisation_threshold);
            else
                activity_bin = zeros(size(activity));
            end
            ent_bin(trial, t) = population_entropy(activity_bin);
        end
    end

    % BIN_AN
    ent_bin_an = zeros(nTrials, nTimepoints);
    if nargin >= 4 && ~isempty(animalMean) && ~isempty(animalStd) && animalStd > 0
        for trial = 1:nTrials
            for t = 1:nTimepoints
                activity = gridData_reshaped(:, t, trial);
                activity_z_animal = (activity - animalMean) / animalStd;
                activity_bin_an = double(activity_z_animal > binarisation_threshold);
                ent_bin_an(trial, t) = population_entropy(activity_bin_an);
            end
        end
    else
        ent_bin_an = ent_bin;
    end

    % BIN_ALL
    ent_bin_all = zeros(nTrials, nTimepoints);
    if nargin >= 6 && ~isempty(conditionMean) && ~isempty(conditionStd) && conditionStd > 0
        for trial = 1:nTrials
            for t = 1:nTimepoints
                activity = gridData_reshaped(:, t, trial);
                activity_z_global = (activity - conditionMean) / conditionStd;
                activity_bin_all = double(activity_z_global > binarisation_threshold);
                ent_bin_all(trial, t) = population_entropy(activity_bin_all);
            end
        end
    else
        ent_bin_all = ent_bin;
    end

    % Trial-zscore
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

    % Recording-zscore
    ent_rz = (ent_raw - mean(ent_raw(:))) / std(ent_raw(:));
    entZ_rz = (entZ_raw - mean(entZ_raw(:))) / std(entZ_raw(:));
end

%% Helper Function: Calculate Dispersion from Grid Data
function [disp_raw, disp_rz, disp_tz, dispZ_raw, dispZ_rz, dispZ_tz, disp_bin, disp_bin_an, disp_bin_all] = ...
    calculate_dispersion_from_grid(gridData, gridDimensions, animalMean, animalStd, globalMean, globalStd, binarisation_threshold)

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
        error('Expected Grid data to have 4 dimensions: [gridY, gridX, nTimepoints, nTrials]');
    end

    % Generate grid coordinates
    [gridX_coords, gridY_coords] = meshgrid(1:gridX, 1:gridY);
    grid_coords = [gridY_coords(:), gridX_coords(:)];

    % Reshape
    nGridCells = gridY * gridX;
    gridData_reshaped = reshape(gridData, [nGridCells, nTimepoints, nTrials]);

    % Initialize
    disp_raw = zeros(nTrials, nTimepoints);
    dispZ_raw = zeros(nTrials, nTimepoints);

    % Calculate weighted dispersion
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

    % BIN
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

    % BIN_AN
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

    % BIN_ALL
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

    % Trial-zscore
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

    % Recording-zscore
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

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
        error('Expected Grid data to have 4 dimensions: [gridY, gridX, nTimepoints, nTrials]');
    end

    % Reshape
    nGridCells = gridY * gridX;
    gridData_reshaped = reshape(gridData, [nGridCells, nTimepoints, nTrials]);

    activity_threshold = 0.01;

    % Initialize
    actcells_raw = zeros(nTrials, nTimepoints);
    actcellsZ_raw = zeros(nTrials, nTimepoints);

    % Calculate RAW
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

    % BIN
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

    % BIN_AN
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

    % BIN_ALL
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

    % Trial-zscore
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

    % Recording-zscore
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

    % Initialize
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
