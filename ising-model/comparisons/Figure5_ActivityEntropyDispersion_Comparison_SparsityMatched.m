%% =========================================================================
%% Fig 5: Entropy Comparison - Grid vs Activity Data (Sparsity-Matched)
%% =========================================================================
% This script compares population entropy calculated from:
% 1. Grid40 data (spatially rasterized, 13×26 grid cells)
% 2. ActivityData (individual neurons, ~3000 cells)
%
% Purpose: Understand how spatial coarse-graining affects entropy measurements
% and whether grid-based approaches capture similar population dynamics
%
% NOTE: This version uses CONDITION-SPECIFIC z-score thresholds for binarization
%       to match population sparsity levels across training conditions

% Load data structures
% load(mba_p('RawData3.mat'),'ActivityData');
% load(mba_p('RawData3.mat'),'params');
% load(mba_p('RawData3.mat'),'Entropy'); % Entropy from ActivityData

% Define Skip arrays (recordings to exclude from analysis)
Skip = [];
Skip.Naive = [1 9 10 16];
Skip.Beginner = [1 6 7 11];
Skip.Expert = [1 4 12 13 14];
Skip.ExpertRandom = [1];
Skip.ExpertAll = [1,4,5,13, 20,21,22,23,24,25,26];
Skip.NoSpout = [1 4 9 10 11 13 14];

% Moran's I calculation flag (spatial autocorrelation metric)
% Note: Only BIN_AN type for Position1 is currently implemented
include_MoransI = true;

% Condition-specific binarisation thresholds (sparsity-matched)
% These thresholds equate sparsity levels across training conditions
% Note: Sparsity-matched thresholds ensure comparable population sparsity
%       levels across conditions, accounting for differences in baseline
%       activity and response magnitudes during training progression
binarisation_thresholds = struct();
binarisation_thresholds.Naive = 3.1;        % σ threshold for Naive
binarisation_thresholds.Beginner = 2.7;     % σ threshold for Beginner
binarisation_thresholds.Expert = 1.7;       % σ threshold for Expert
binarisation_thresholds.ExpertRandom = 1.7; % σ threshold (same as Expert)
binarisation_thresholds.NoSpout = 2.9;      % σ threshold for NoSpout
binarisation_thresholds.ExpertAll = 1.7;    % σ threshold (same as Expert)

%% =========================================================================
%% Section 1: Calculate Entropy and Dispersion from Grid40 Data
%% =========================================================================
fprintf('\n=== Calculating Entropy and Dispersion from Grid40 Data ===\n');

% Define conditions to analyze
conditions = {'Naive', 'Beginner', 'Expert', 'NoSpout'};

% Grid parameters (consistent with Figure 3)
gridSize = 40;
gridDimensions = [13 26];  % Grid structure: 13 rows × 26 columns

% Initialize Grid entropy structure
Entropy_Grid = struct();

% Initialize Grid dispersion structure
Dispersion_Grid = struct();

% Initialize Grid active cells structure
ActiveCells_Grid = struct();

% Initialize Grid Moran's I structure (spatial autocorrelation)
% Note: Only BIN_AN for Position1 is currently calculated (others are placeholders)
MoransI_Grid = struct();

%% =========================================================================
%% Section 1.1: Calculate GLOBAL statistics for BIN_ALL (across ALL conditions)
%% =========================================================================
fprintf('\n--- Calculating GLOBAL statistics for BIN_ALL (across ALL conditions) ---\n');

% Collect ALL grid data from ALL conditions
allGlobalGridData_P1 = [];
allGlobalGridData_P3 = [];

for c = 1:length(conditions)
    condition = conditions{c};
    fprintf('  Collecting data from condition: %s\n', condition);

    % Get recording count
    if ~isfield(ActivityData, condition)
        continue;
    end
    nRecs = length(ActivityData.(condition));

    % Define condition with Individual suffix for Grid40 access
    conditionIndividual = [condition 'Individual'];

    % Check if Grid40 data exists for this condition
    if ~isfield(Grid40, conditionIndividual)
        fprintf('    No Grid40 data (looking for %s)\n', conditionIndividual);
        continue;
    end

    % Get number of recordings from Grid40
    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);

    % Collect data from all recordings in this condition
    for r = 1:min(nRecs, nRecsGrid)
        % Skip recordings in Skip list
        if ismember(r, Skip.(condition))
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
                allGlobalGridData_P1 = cat(4, allGlobalGridData_P1, gridData_P1);
            end
        end

        % Collect P3 data
        if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P3')
            gridData_P3_cell = Grid40.(conditionIndividual).AllNeurons(r).P3;
            if ~isempty(gridData_P3_cell) && iscell(gridData_P3_cell)
                gridData_P3 = gridData_P3_cell{:};
            else
                gridData_P3 = gridData_P3_cell;
            end

            if ~isempty(gridData_P3)
                allGlobalGridData_P3 = cat(4, allGlobalGridData_P3, gridData_P3);
            end
        end
    end
end

% Calculate GLOBAL statistics (mean and std across ALL data points from ALL conditions)
globalMean_P1 = [];
globalStd_P1 = [];
globalMean_P3 = [];
globalStd_P3 = [];

if ~isempty(allGlobalGridData_P1)
    globalMean_P1 = mean(allGlobalGridData_P1(:));
    globalStd_P1 = std(allGlobalGridData_P1(:));
    fprintf('\n=== GLOBAL BIN_ALL Statistics ===\n');
    fprintf('P1: Global mean=%.4f, std=%.4f (across ALL conditions)\n', globalMean_P1, globalStd_P1);
else
    fprintf('\nWarning: No P1 data collected for global statistics\n');
end

if ~isempty(allGlobalGridData_P3)
    globalMean_P3 = mean(allGlobalGridData_P3(:));
    globalStd_P3 = std(allGlobalGridData_P3(:));
    fprintf('P3: Global mean=%.4f, std=%.4f (across ALL conditions)\n', globalMean_P3, globalStd_P3);
else
    fprintf('Warning: No P3 data collected for global statistics\n');
end

%% =========================================================================
%% Section 1.2: Process each condition (calculate entropies and dispersion)
%% =========================================================================

% Loop through conditions
for c = 1:length(conditions)
    condition = conditions{c};
    fprintf('\nProcessing condition: %s\n', condition);

    nRecs = length(ActivityData.(condition));

    % Initialize entropy storage for this condition
    % Store entropy for Position1 (P1), Position3 (P3), and All
    Entropy_Grid.(condition).Position1 = struct();
    Entropy_Grid.(condition).Position3 = struct();
    Entropy_Grid.(condition).All = struct();

    % Initialize CELL ARRAYS for different normalization types (per-recording structure)
    for pos = {'Position1', 'Position3', 'All'}
        posField = pos{1};
        Entropy_Grid.(condition).(posField).RecordingEntropy_Raw = {};
        Entropy_Grid.(condition).(posField).RecordingEntropy_RZ = {};
        Entropy_Grid.(condition).(posField).RecordingEntropy_TZ = {};
        Entropy_Grid.(condition).(posField).RecordingEntropy_BIN = {};
        Entropy_Grid.(condition).(posField).RecordingEntropy_BIN_AN = {};
        Entropy_Grid.(condition).(posField).RecordingEntropy_BIN_ALL = {};
        Entropy_Grid.(condition).(posField).RecordingEntropyZ_Raw = {};
        Entropy_Grid.(condition).(posField).RecordingEntropyZ_RZ = {};
        Entropy_Grid.(condition).(posField).RecordingEntropyZ_TZ = {};
        % Note: BIN, BIN_AN, and BIN_ALL already z-score as part of binarisation, so no separate Z-scored versions needed
    end

    % Initialize dispersion storage for this condition
    % Store dispersion for Position1 (P1), Position3 (P3), and All
    Dispersion_Grid.(condition).Position1 = struct();
    Dispersion_Grid.(condition).Position3 = struct();
    Dispersion_Grid.(condition).All = struct();

    % Initialize CELL ARRAYS for dispersion (per-recording structure)
    for pos = {'Position1', 'Position3', 'All'}
        posField = pos{1};
        Dispersion_Grid.(condition).(posField).RecordingDispersion_Raw = {};
        Dispersion_Grid.(condition).(posField).RecordingDispersion_RZ = {};
        Dispersion_Grid.(condition).(posField).RecordingDispersion_TZ = {};
        Dispersion_Grid.(condition).(posField).RecordingDispersion_BIN = {};
        Dispersion_Grid.(condition).(posField).RecordingDispersion_BIN_AN = {};
        Dispersion_Grid.(condition).(posField).RecordingDispersion_BIN_ALL = {};
        Dispersion_Grid.(condition).(posField).RecordingDispersionZ_Raw = {};
        Dispersion_Grid.(condition).(posField).RecordingDispersionZ_RZ = {};
        Dispersion_Grid.(condition).(posField).RecordingDispersionZ_TZ = {};
        % Note: BIN, BIN_AN, and BIN_ALL already z-score as part of binarisation, so no separate Z-scored versions needed
    end

    % Initialize active cells storage for this condition
    % Store active cells count for Position1 (P1), Position3 (P3), and All
    ActiveCells_Grid.(condition).Position1 = struct();
    ActiveCells_Grid.(condition).Position3 = struct();
    ActiveCells_Grid.(condition).All = struct();

    % Initialize CELL ARRAYS for active cells (per-recording structure)
    for pos = {'Position1', 'Position3', 'All'}
        posField = pos{1};
        ActiveCells_Grid.(condition).(posField).RecordingActiveCells_Raw = {};
        ActiveCells_Grid.(condition).(posField).RecordingActiveCells_RZ = {};
        ActiveCells_Grid.(condition).(posField).RecordingActiveCells_TZ = {};
        ActiveCells_Grid.(condition).(posField).RecordingActiveCells_BIN = {};
        ActiveCells_Grid.(condition).(posField).RecordingActiveCells_BIN_AN = {};
        ActiveCells_Grid.(condition).(posField).RecordingActiveCells_BIN_ALL = {};
        ActiveCells_Grid.(condition).(posField).RecordingActiveCellsZ_Raw = {};
        ActiveCells_Grid.(condition).(posField).RecordingActiveCellsZ_RZ = {};
        ActiveCells_Grid.(condition).(posField).RecordingActiveCellsZ_TZ = {};
        % Note: BIN, BIN_AN, and BIN_ALL already z-score as part of binarisation, so no separate Z-scored versions needed
    end

    % Initialize Moran's I storage for this condition (spatial autocorrelation)
    MoransI_Grid.(condition).Position1 = struct();
    MoransI_Grid.(condition).Position3 = struct();
    MoransI_Grid.(condition).All = struct();

    % Initialize CELL ARRAYS for Moran's I (per-recording structure)
    % NOTE: Currently only BIN_AN for Position1 is calculated; others are placeholders
    for pos = {'Position1', 'Position3', 'All'}
        posField = pos{1};
        MoransI_Grid.(condition).(posField).RecordingMoransI_Raw = {};          % Placeholder
        MoransI_Grid.(condition).(posField).RecordingMoransI_RZ = {};           % Placeholder
        MoransI_Grid.(condition).(posField).RecordingMoransI_TZ = {};           % Placeholder
        MoransI_Grid.(condition).(posField).RecordingMoransI_BIN = {};          % Placeholder
        MoransI_Grid.(condition).(posField).RecordingMoransI_BIN_AN = {};       % CALCULATED for Position1 only
        MoransI_Grid.(condition).(posField).RecordingMoransI_BIN_ALL = {};      % Placeholder
        MoransI_Grid.(condition).(posField).RecordingMoransIZ_Raw = {};         % Placeholder
        MoransI_Grid.(condition).(posField).RecordingMoransIZ_RZ = {};          % Placeholder
        MoransI_Grid.(condition).(posField).RecordingMoransIZ_TZ = {};          % Placeholder
    end

    % Define condition with Individual suffix for Grid40 access
    conditionIndividual = [condition 'Individual'];

    % Check if Grid40 data exists for this condition
    if ~isfield(Grid40, conditionIndividual)
        fprintf('  Condition %s: No Grid40 data (looking for %s)\n', condition, conditionIndividual);
        continue;
    end

    % Get number of recordings from Grid40
    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);

    %% Calculate per-animal statistics for BIN_AN binarisation
    % Get animal IDs for this condition
    if isfield(Rec, condition) && height(Rec.(condition)) >= nRecs
        animalIDs = Rec.(condition).AnimalID;
        uniqueAnimalIDs = unique(animalIDs);

        % Store per-animal statistics: mean and std for P1 and P3
        animalStats_P1 = struct();
        animalStats_P3 = struct();

        fprintf('  Calculating per-animal statistics for BIN_AN...\n');
        for a = 1:length(uniqueAnimalIDs)
            currentAnimalID = uniqueAnimalIDs{a};
            % Convert to valid MATLAB field name (prefix with 'Animal_')
            validFieldName = ['Animal_' currentAnimalID];

            % Find all recordings for this animal
            animalRecordings = find(strcmp(animalIDs, currentAnimalID));

            % Collect all grid data for this animal (P1 and P3 separately)
            allGridData_P1 = [];
            allGridData_P3 = [];

            for rec_idx = 1:length(animalRecordings)
                r = animalRecordings(rec_idx);

                % Skip if out of bounds or in Skip list
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

                % Collect P3 data
                if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P3')
                    gridData_P3_cell = Grid40.(conditionIndividual).AllNeurons(r).P3;
                    if ~isempty(gridData_P3_cell) && iscell(gridData_P3_cell)
                        gridData_P3 = gridData_P3_cell{:};
                    else
                        gridData_P3 = gridData_P3_cell;
                    end

                    if ~isempty(gridData_P3)
                        allGridData_P3 = cat(4, allGridData_P3, gridData_P3);
                    end
                end
            end

            % Calculate animal-level statistics (mean and std across all data points)
            if ~isempty(allGridData_P1)
                animalStats_P1.(validFieldName).mean = mean(allGridData_P1(:));
                animalStats_P1.(validFieldName).std = std(allGridData_P1(:));
            end

            if ~isempty(allGridData_P3)
                animalStats_P3.(validFieldName).mean = mean(allGridData_P3(:));
                animalStats_P3.(validFieldName).std = std(allGridData_P3(:));
            end
        end
        fprintf('  Per-animal statistics calculated for %d animals\n', length(uniqueAnimalIDs));
    else
        % If no Rec data available, use empty structures
        animalStats_P1 = struct();
        animalStats_P3 = struct();
        animalIDs = {};
    end

    % Loop through recordings
    for r = 1:min(nRecs, nRecsGrid)
        % Check if this recording should be skipped
        if ismember(r, Skip.(condition))
            fprintf('  Recording %d: SKIPPED\n', r);
            continue;
        end

        % Check if this recording has data
        if r > length(Grid40.(conditionIndividual).AllNeurons)
            fprintf('  Recording %d: Index exceeds Grid40 array\n', r);
            continue;
        end

        % Get animal ID for this recording (for BIN_AN calculation)
        currentAnimalID_r = '';
        animalMean_P1 = [];
        animalStd_P1 = [];
        animalMean_P3 = [];
        animalStd_P3 = [];
        if ~isempty(animalIDs) && r <= length(animalIDs)
            currentAnimalID_r = animalIDs{r};
            % Convert to valid MATLAB field name (prefix with 'Animal_')
            validFieldName_r = ['Animal_' currentAnimalID_r];

            % Get animal statistics if available
            if isfield(animalStats_P1, validFieldName_r)
                animalMean_P1 = animalStats_P1.(validFieldName_r).mean;
                animalStd_P1 = animalStats_P1.(validFieldName_r).std;
            end
            if isfield(animalStats_P3, validFieldName_r)
                animalMean_P3 = animalStats_P3.(validFieldName_r).mean;
                animalStd_P3 = animalStats_P3.(validFieldName_r).std;
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
                % Calculate entropy from Grid data (including BIN_AN and BIN_ALL)
                % BIN_ALL uses GLOBAL statistics (across all conditions)
                % Using condition-specific threshold for sparsity matching
                [ent_raw_P1, ent_rz_P1, ent_tz_P1, entZ_raw_P1, entZ_rz_P1, entZ_tz_P1, ent_bin_P1, ent_bin_an_P1, ent_bin_all_P1] = ...
                    calculate_entropy_from_grid(gridData_P1, gridDimensions, animalMean_P1, animalStd_P1, globalMean_P1, globalStd_P1, binarisation_thresholds.(condition));

                % Store entropy values in CELL ARRAYS (one cell per recording)
                Entropy_Grid.(condition).Position1.RecordingEntropy_Raw{end+1} = ent_raw_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropy_RZ{end+1} = ent_rz_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropy_TZ{end+1} = ent_tz_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropy_BIN{end+1} = ent_bin_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropy_BIN_AN{end+1} = ent_bin_an_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropy_BIN_ALL{end+1} = ent_bin_all_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropyZ_Raw{end+1} = entZ_raw_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropyZ_RZ{end+1} = entZ_rz_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropyZ_TZ{end+1} = entZ_tz_P1;

                % Calculate dispersion from Grid data (including BIN_AN and BIN_ALL)
                % Using condition-specific threshold for sparsity matching
                [disp_raw_P1, disp_rz_P1, disp_tz_P1, dispZ_raw_P1, dispZ_rz_P1, dispZ_tz_P1, disp_bin_P1, disp_bin_an_P1, disp_bin_all_P1] = ...
                    calculate_dispersion_from_grid(gridData_P1, gridDimensions, animalMean_P1, animalStd_P1, globalMean_P1, globalStd_P1, binarisation_thresholds.(condition));

                % Store dispersion values in CELL ARRAYS (one cell per recording)
                Dispersion_Grid.(condition).Position1.RecordingDispersion_Raw{end+1} = disp_raw_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersion_RZ{end+1} = disp_rz_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersion_TZ{end+1} = disp_tz_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersion_BIN{end+1} = disp_bin_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersion_BIN_AN{end+1} = disp_bin_an_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersion_BIN_ALL{end+1} = disp_bin_all_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersionZ_Raw{end+1} = dispZ_raw_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersionZ_RZ{end+1} = dispZ_rz_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersionZ_TZ{end+1} = dispZ_tz_P1;

                % Calculate active cells count from Grid data (including BIN_AN and BIN_ALL)
                % Using condition-specific threshold for sparsity matching
                [actcells_raw_P1, actcells_rz_P1, actcells_tz_P1, actcellsZ_raw_P1, actcellsZ_rz_P1, actcellsZ_tz_P1, actcells_bin_P1, actcells_bin_an_P1, actcells_bin_all_P1] = ...
                    calculate_active_cells_from_grid(gridData_P1, gridDimensions, animalMean_P1, animalStd_P1, globalMean_P1, globalStd_P1, binarisation_thresholds.(condition));

                % Store active cells values in CELL ARRAYS (one cell per recording)
                ActiveCells_Grid.(condition).Position1.RecordingActiveCells_Raw{end+1} = actcells_raw_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCells_RZ{end+1} = actcells_rz_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCells_TZ{end+1} = actcells_tz_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCells_BIN{end+1} = actcells_bin_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCells_BIN_AN{end+1} = actcells_bin_an_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCells_BIN_ALL{end+1} = actcells_bin_all_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCellsZ_Raw{end+1} = actcellsZ_raw_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCellsZ_RZ{end+1} = actcellsZ_rz_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCellsZ_TZ{end+1} = actcellsZ_tz_P1;

                % Calculate Moran's I from Grid data (spatial autocorrelation)
                % Only BIN_AN is currently implemented
                % Using condition-specific threshold for sparsity matching
                if include_MoransI
                    morans_bin_an_P1 = calculate_moransI_BIN_AN(gridData_P1, gridDimensions, animalMean_P1, animalStd_P1, binarisation_thresholds.(condition));
                    MoransI_Grid.(condition).Position1.RecordingMoransI_BIN_AN{end+1} = morans_bin_an_P1;
                end
            end
        end

        % Process Position3 (P3)
        if isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P3')
            gridData_P3_cell = Grid40.(conditionIndividual).AllNeurons(r).P3;

            % Dereference cell array
            if ~isempty(gridData_P3_cell) && iscell(gridData_P3_cell)
                gridData_P3 = gridData_P3_cell{:};
            else
                gridData_P3 = gridData_P3_cell;
            end

            if ~isempty(gridData_P3)
                % Calculate entropy from Grid data (including BIN_AN and BIN_ALL)
                % BIN_ALL uses GLOBAL statistics (across all conditions)
                % Using condition-specific threshold for sparsity matching
                [ent_raw_P3, ent_rz_P3, ent_tz_P3, entZ_raw_P3, entZ_rz_P3, entZ_tz_P3, ent_bin_P3, ent_bin_an_P3, ent_bin_all_P3] = ...
                    calculate_entropy_from_grid(gridData_P3, gridDimensions, animalMean_P3, animalStd_P3, globalMean_P3, globalStd_P3, binarisation_thresholds.(condition));

                % Store entropy values in CELL ARRAYS (one cell per recording)
                Entropy_Grid.(condition).Position3.RecordingEntropy_Raw{end+1} = ent_raw_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropy_RZ{end+1} = ent_rz_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropy_TZ{end+1} = ent_tz_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropy_BIN{end+1} = ent_bin_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropy_BIN_AN{end+1} = ent_bin_an_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropy_BIN_ALL{end+1} = ent_bin_all_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropyZ_Raw{end+1} = entZ_raw_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropyZ_RZ{end+1} = entZ_rz_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropyZ_TZ{end+1} = entZ_tz_P3;

                % Calculate dispersion from Grid data (including BIN_AN and BIN_ALL)
                % Using condition-specific threshold for sparsity matching
                [disp_raw_P3, disp_rz_P3, disp_tz_P3, dispZ_raw_P3, dispZ_rz_P3, dispZ_tz_P3, disp_bin_P3, disp_bin_an_P3, disp_bin_all_P3] = ...
                    calculate_dispersion_from_grid(gridData_P3, gridDimensions, animalMean_P3, animalStd_P3, globalMean_P3, globalStd_P3, binarisation_thresholds.(condition));

                % Store dispersion values in CELL ARRAYS (one cell per recording)
                Dispersion_Grid.(condition).Position3.RecordingDispersion_Raw{end+1} = disp_raw_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersion_RZ{end+1} = disp_rz_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersion_TZ{end+1} = disp_tz_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersion_BIN{end+1} = disp_bin_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersion_BIN_AN{end+1} = disp_bin_an_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersion_BIN_ALL{end+1} = disp_bin_all_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersionZ_Raw{end+1} = dispZ_raw_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersionZ_RZ{end+1} = dispZ_rz_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersionZ_TZ{end+1} = dispZ_tz_P3;

                % Calculate active cells count from Grid data (including BIN_AN and BIN_ALL)
                % Using condition-specific threshold for sparsity matching
                [actcells_raw_P3, actcells_rz_P3, actcells_tz_P3, actcellsZ_raw_P3, actcellsZ_rz_P3, actcellsZ_tz_P3, actcells_bin_P3, actcells_bin_an_P3, actcells_bin_all_P3] = ...
                    calculate_active_cells_from_grid(gridData_P3, gridDimensions, animalMean_P3, animalStd_P3, globalMean_P3, globalStd_P3, binarisation_thresholds.(condition));

                % Store active cells values in CELL ARRAYS (one cell per recording)
                ActiveCells_Grid.(condition).Position3.RecordingActiveCells_Raw{end+1} = actcells_raw_P3;
                ActiveCells_Grid.(condition).Position3.RecordingActiveCells_RZ{end+1} = actcells_rz_P3;
                ActiveCells_Grid.(condition).Position3.RecordingActiveCells_TZ{end+1} = actcells_tz_P3;
                ActiveCells_Grid.(condition).Position3.RecordingActiveCells_BIN{end+1} = actcells_bin_P3;
                ActiveCells_Grid.(condition).Position3.RecordingActiveCells_BIN_AN{end+1} = actcells_bin_an_P3;
                ActiveCells_Grid.(condition).Position3.RecordingActiveCells_BIN_ALL{end+1} = actcells_bin_all_P3;
                ActiveCells_Grid.(condition).Position3.RecordingActiveCellsZ_Raw{end+1} = actcellsZ_raw_P3;
                ActiveCells_Grid.(condition).Position3.RecordingActiveCellsZ_RZ{end+1} = actcellsZ_rz_P3;
                ActiveCells_Grid.(condition).Position3.RecordingActiveCellsZ_TZ{end+1} = actcellsZ_tz_P3;
            end
        end

        fprintf('  Recording %d: Entropy calculated from Grid\n', r);
    end

    % Combine P1 and P3 for "All" position (concatenate cell arrays)
    Entropy_Grid.(condition).All.RecordingEntropy_Raw = ...
        [Entropy_Grid.(condition).Position1.RecordingEntropy_Raw, ...
         Entropy_Grid.(condition).Position3.RecordingEntropy_Raw];
    Entropy_Grid.(condition).All.RecordingEntropy_RZ = ...
        [Entropy_Grid.(condition).Position1.RecordingEntropy_RZ, ...
         Entropy_Grid.(condition).Position3.RecordingEntropy_RZ];
    Entropy_Grid.(condition).All.RecordingEntropy_TZ = ...
        [Entropy_Grid.(condition).Position1.RecordingEntropy_TZ, ...
         Entropy_Grid.(condition).Position3.RecordingEntropy_TZ];
    Entropy_Grid.(condition).All.RecordingEntropy_BIN = ...
        [Entropy_Grid.(condition).Position1.RecordingEntropy_BIN, ...
         Entropy_Grid.(condition).Position3.RecordingEntropy_BIN];
    Entropy_Grid.(condition).All.RecordingEntropy_BIN_AN = ...
        [Entropy_Grid.(condition).Position1.RecordingEntropy_BIN_AN, ...
         Entropy_Grid.(condition).Position3.RecordingEntropy_BIN_AN];
    Entropy_Grid.(condition).All.RecordingEntropy_BIN_ALL = ...
        [Entropy_Grid.(condition).Position1.RecordingEntropy_BIN_ALL, ...
         Entropy_Grid.(condition).Position3.RecordingEntropy_BIN_ALL];
    Entropy_Grid.(condition).All.RecordingEntropyZ_Raw = ...
        [Entropy_Grid.(condition).Position1.RecordingEntropyZ_Raw, ...
         Entropy_Grid.(condition).Position3.RecordingEntropyZ_Raw];
    Entropy_Grid.(condition).All.RecordingEntropyZ_RZ = ...
        [Entropy_Grid.(condition).Position1.RecordingEntropyZ_RZ, ...
         Entropy_Grid.(condition).Position3.RecordingEntropyZ_RZ];
    Entropy_Grid.(condition).All.RecordingEntropyZ_TZ = ...
        [Entropy_Grid.(condition).Position1.RecordingEntropyZ_TZ, ...
         Entropy_Grid.(condition).Position3.RecordingEntropyZ_TZ];
    % Note: BIN, BIN_AN, and BIN_ALL already z-score as part of binarisation, so no separate Z-scored versions needed

    % Combine P1 and P3 dispersion for "All" position (concatenate cell arrays)
    Dispersion_Grid.(condition).All.RecordingDispersion_Raw = ...
        [Dispersion_Grid.(condition).Position1.RecordingDispersion_Raw, ...
         Dispersion_Grid.(condition).Position3.RecordingDispersion_Raw];
    Dispersion_Grid.(condition).All.RecordingDispersion_RZ = ...
        [Dispersion_Grid.(condition).Position1.RecordingDispersion_RZ, ...
         Dispersion_Grid.(condition).Position3.RecordingDispersion_RZ];
    Dispersion_Grid.(condition).All.RecordingDispersion_TZ = ...
        [Dispersion_Grid.(condition).Position1.RecordingDispersion_TZ, ...
         Dispersion_Grid.(condition).Position3.RecordingDispersion_TZ];
    Dispersion_Grid.(condition).All.RecordingDispersion_BIN = ...
        [Dispersion_Grid.(condition).Position1.RecordingDispersion_BIN, ...
         Dispersion_Grid.(condition).Position3.RecordingDispersion_BIN];
    Dispersion_Grid.(condition).All.RecordingDispersion_BIN_AN = ...
        [Dispersion_Grid.(condition).Position1.RecordingDispersion_BIN_AN, ...
         Dispersion_Grid.(condition).Position3.RecordingDispersion_BIN_AN];
    Dispersion_Grid.(condition).All.RecordingDispersion_BIN_ALL = ...
        [Dispersion_Grid.(condition).Position1.RecordingDispersion_BIN_ALL, ...
         Dispersion_Grid.(condition).Position3.RecordingDispersion_BIN_ALL];
    Dispersion_Grid.(condition).All.RecordingDispersionZ_Raw = ...
        [Dispersion_Grid.(condition).Position1.RecordingDispersionZ_Raw, ...
         Dispersion_Grid.(condition).Position3.RecordingDispersionZ_Raw];
    Dispersion_Grid.(condition).All.RecordingDispersionZ_RZ = ...
        [Dispersion_Grid.(condition).Position1.RecordingDispersionZ_RZ, ...
         Dispersion_Grid.(condition).Position3.RecordingDispersionZ_RZ];
    Dispersion_Grid.(condition).All.RecordingDispersionZ_TZ = ...
        [Dispersion_Grid.(condition).Position1.RecordingDispersionZ_TZ, ...
         Dispersion_Grid.(condition).Position3.RecordingDispersionZ_TZ];

    % Combine P1 and P3 active cells for "All" position (concatenate cell arrays)
    ActiveCells_Grid.(condition).All.RecordingActiveCells_Raw = ...
        [ActiveCells_Grid.(condition).Position1.RecordingActiveCells_Raw, ...
         ActiveCells_Grid.(condition).Position3.RecordingActiveCells_Raw];
    ActiveCells_Grid.(condition).All.RecordingActiveCells_RZ = ...
        [ActiveCells_Grid.(condition).Position1.RecordingActiveCells_RZ, ...
         ActiveCells_Grid.(condition).Position3.RecordingActiveCells_RZ];
    ActiveCells_Grid.(condition).All.RecordingActiveCells_TZ = ...
        [ActiveCells_Grid.(condition).Position1.RecordingActiveCells_TZ, ...
         ActiveCells_Grid.(condition).Position3.RecordingActiveCells_TZ];
    ActiveCells_Grid.(condition).All.RecordingActiveCells_BIN = ...
        [ActiveCells_Grid.(condition).Position1.RecordingActiveCells_BIN, ...
         ActiveCells_Grid.(condition).Position3.RecordingActiveCells_BIN];
    ActiveCells_Grid.(condition).All.RecordingActiveCells_BIN_AN = ...
        [ActiveCells_Grid.(condition).Position1.RecordingActiveCells_BIN_AN, ...
         ActiveCells_Grid.(condition).Position3.RecordingActiveCells_BIN_AN];
    ActiveCells_Grid.(condition).All.RecordingActiveCells_BIN_ALL = ...
        [ActiveCells_Grid.(condition).Position1.RecordingActiveCells_BIN_ALL, ...
         ActiveCells_Grid.(condition).Position3.RecordingActiveCells_BIN_ALL];
    ActiveCells_Grid.(condition).All.RecordingActiveCellsZ_Raw = ...
        [ActiveCells_Grid.(condition).Position1.RecordingActiveCellsZ_Raw, ...
         ActiveCells_Grid.(condition).Position3.RecordingActiveCellsZ_Raw];
    ActiveCells_Grid.(condition).All.RecordingActiveCellsZ_RZ = ...
        [ActiveCells_Grid.(condition).Position1.RecordingActiveCellsZ_RZ, ...
         ActiveCells_Grid.(condition).Position3.RecordingActiveCellsZ_RZ];
    ActiveCells_Grid.(condition).All.RecordingActiveCellsZ_TZ = ...
        [ActiveCells_Grid.(condition).Position1.RecordingActiveCellsZ_TZ, ...
         ActiveCells_Grid.(condition).Position3.RecordingActiveCellsZ_TZ];

    % Combine P1 and P3 Moran's I for "All" position (concatenate cell arrays)
    % NOTE: Currently only Position1 BIN_AN is calculated, so Position3 arrays are empty
    % Position3 and All will contain only Position1 data (no P3 calculation yet)
    MoransI_Grid.(condition).All.RecordingMoransI_Raw = ...
        [MoransI_Grid.(condition).Position1.RecordingMoransI_Raw, ...
         MoransI_Grid.(condition).Position3.RecordingMoransI_Raw];  % Both empty (placeholder)
    MoransI_Grid.(condition).All.RecordingMoransI_RZ = ...
        [MoransI_Grid.(condition).Position1.RecordingMoransI_RZ, ...
         MoransI_Grid.(condition).Position3.RecordingMoransI_RZ];  % Both empty (placeholder)
    MoransI_Grid.(condition).All.RecordingMoransI_TZ = ...
        [MoransI_Grid.(condition).Position1.RecordingMoransI_TZ, ...
         MoransI_Grid.(condition).Position3.RecordingMoransI_TZ];  % Both empty (placeholder)
    MoransI_Grid.(condition).All.RecordingMoransI_BIN = ...
        [MoransI_Grid.(condition).Position1.RecordingMoransI_BIN, ...
         MoransI_Grid.(condition).Position3.RecordingMoransI_BIN];  % Both empty (placeholder)
    MoransI_Grid.(condition).All.RecordingMoransI_BIN_AN = ...
        [MoransI_Grid.(condition).Position1.RecordingMoransI_BIN_AN, ...
         MoransI_Grid.(condition).Position3.RecordingMoransI_BIN_AN];  % P1 calculated, P3 empty
    MoransI_Grid.(condition).All.RecordingMoransI_BIN_ALL = ...
        [MoransI_Grid.(condition).Position1.RecordingMoransI_BIN_ALL, ...
         MoransI_Grid.(condition).Position3.RecordingMoransI_BIN_ALL];  % Both empty (placeholder)
    MoransI_Grid.(condition).All.RecordingMoransIZ_Raw = ...
        [MoransI_Grid.(condition).Position1.RecordingMoransIZ_Raw, ...
         MoransI_Grid.(condition).Position3.RecordingMoransIZ_Raw];  % Both empty (placeholder)
    MoransI_Grid.(condition).All.RecordingMoransIZ_RZ = ...
        [MoransI_Grid.(condition).Position1.RecordingMoransIZ_RZ, ...
         MoransI_Grid.(condition).Position3.RecordingMoransIZ_RZ];  % Both empty (placeholder)
    MoransI_Grid.(condition).All.RecordingMoransIZ_TZ = ...
        [MoransI_Grid.(condition).Position1.RecordingMoransIZ_TZ, ...
         MoransI_Grid.(condition).Position3.RecordingMoransIZ_TZ];  % Both empty (placeholder)

    fprintf('  Condition %s complete\n', condition);
end

if include_MoransI
    fprintf('\n=== Grid Entropy, Dispersion, Active Cells, and Moran''s I Calculation Complete ===\n');
else
    fprintf('\n=== Grid Entropy, Dispersion, and Active Cells Calculation Complete ===\n');
end

%% =========================================================================
%% Section 2: Entropy Comparison Analysis
%% =========================================================================
%% Section 2.1: Grid vs Activity Entropy Comparison
%% =========================================================================
% Control which entropy normalization to compare between Grid and Activity
% Type options:
%   'Raw'     - No normalization, absolute entropy values (both Activity and Grid)
%   'RZ'      - Recording z-score (normalized across all trials in recording, both Activity and Grid)
%   'TZ'      - Trial z-score (normalized within each trial relative to baseline, both Activity and Grid)
%   'BIN'     - Binarised entropy comparison: Raw Activity vs Binarised Grid (per-recording threshold at 1.5σ)
%               Activity uses continuous Raw entropy, Grid uses binarised (on/off) states
%               Compares continuous neuron-level entropy to discrete grid-state entropy
%   'BIN_AN'  - Binarised entropy comparison: Raw Activity vs Binarised Grid (per-animal threshold at 1.5σ)
%               Similar to BIN, but uses animal-level z-score statistics for binarisation
%               Reduces inter-animal variability by normalizing within each animal
%   'BIN_ALL' - Binarised entropy comparison: Raw Activity vs Binarised Grid (GLOBAL threshold at 1.5σ)
%               Similar to BIN, but uses GLOBAL z-score statistics across ALL conditions
%               Provides broadest normalization, removing all between-condition, between-recording, and between-animal variability
% UseZScored options:
%   false - Use entropy calculated from raw neural activity
%   true  - Use entropy calculated from z-scored neural activity (entZ_*)
Type = 'BIN_AN';  % Options: 'Raw', 'RZ', 'TZ', 'BIN', 'BIN_AN', 'BIN_ALL'
UseZScored = false;    % Set true to use entropy from z-scored activity

fprintf('\n=== Comparing Grid vs Activity Entropy ===\n');

% Define positions to analyze
Position = "All";
 Position = "P1";
% Position = "P3";
% Position = "P1_P3";
% Position = "P1_P3Trained";

% Call comparison plotting function
plotMedianEntropyBars_Grid(Entropy, Entropy_Grid, Rec, params, conditions, Skip, ...
    'Position', Position, 'Type', Type, 'UseZScored', UseZScored)

%% =========================================================================
%% Section 2.2: Correlation & Agreement Analysis
%% =========================================================================

% Comprehensive correlation analysis: Grid entropy vs Activity entropy
correlation_stats = plotEntropyCorrelation_GridVsActivity(Entropy, Entropy_Grid, params, conditions, Skip, ...
    'Position', Position, 'Type', Type, 'UseZScored', UseZScored)


%% =========================================================================
%% Section 3: Dispersion Analysis (Spatial Organization)
%% =========================================================================
% Dispersion measures how spatially concentrated vs. dispersed active grid cells are:
%   - Low dispersion = tight blob (organized activity)
%   - High dispersion = scattered activity (disorganized)

%% 3.1: Dispersion Distribution Across Conditions
plotMedianDispersionBars_Grid(Dispersion_Grid, Rec, params, conditions, Skip, ...
    'Position', Position, 'Type', Type);

%% 3.2: Dispersion vs Entropy Correlation
% Key Question: Are low-entropy states also low-dispersion (tight blobs)?
dispersion_entropy_corr = plotDispersionCorrelation_GridVsEntropy(Entropy_Grid, Dispersion_Grid, ...
    params, conditions, Skip, 'Position', Position, 'Type', Type, 'UseZScored', UseZScored);

%% 3.3: Temporal Dynamics of Dispersion
plotDispersionTimeCourse_Grid(Dispersion_Grid, params, conditions, Skip,'Position', Position, 'Type', Type);

%% 3.4: Dispersion Metric Explanation Graphic
% Generate explanatory figure showing how dispersion is calculated
plot_dispersion_metric_explanation();

%% =========================================================================
%% Section 4: Moran's I Analysis (Spatial Autocorrelation)
%% =========================================================================
if include_MoransI
    % Moran's I measures spatial autocorrelation - whether similar values cluster together:
    %   - Positive values = clustered activity (similar neighbors)
    %   - Negative values = dispersed activity (dissimilar neighbors)
    %   - Values near 0 = random spatial distribution
    % NOTE: Only BIN_AN for Position1 is currently calculated

    fprintf('\n=== Moran''s I Spatial Autocorrelation Analysis ===\n');

    %% 4.1: Moran's I Distribution Across Conditions
    % Show distribution of spatial autocorrelation values across conditions
    plotMedianMoransIBars_Grid(MoransI_Grid, Rec, params, conditions, Skip, ...
        'Position', Position, 'Type', Type);

    %% 4.2: Moran's I vs Entropy Correlation
    % Key Question: Are low-entropy states also highly autocorrelated (clustered)?
    % Expected: Negative correlation (low entropy = fewer states = more clustering)
    moransI_entropy_corr = plotMoransICorrelation_GridVsEntropy(Entropy_Grid, MoransI_Grid, ...
        params, conditions, Skip, 'Position', Position, 'Type', Type, 'UseZScored', UseZScored);

    %% 4.3: Temporal Dynamics of Moran's I
    % Show how spatial autocorrelation evolves during trials
    plotMoransITimeCourse_Grid(MoransI_Grid, params, conditions, Skip, ...
        'Position', Position, 'Type', Type);

    %% 4.4: Moran's I vs Dispersion Correlation
    % Key Question: Does high clustering (Moran's I) relate to spatial concentration (low dispersion)?
    % Expected: Positive correlation (high autocorrelation = compact blob = low dispersion)
    moransI_dispersion_corr = plotMoransICorrelation_GridVsDispersion(Dispersion_Grid, MoransI_Grid, ...
        params, conditions, Skip, 'Position', Position, 'Type', Type);

    %% 4.5: Moran's I Metric Explanation
    % Generate explanatory figure showing how Moran's I is calculated
    plot_moransI_metric_explanation();

    fprintf('\n=== Moran''s I Analysis Complete ===\n');
end

%% =========================================================================
%% Section 5: Active Cells Analysis (Population Sparsity)
%% =========================================================================
% Active cells count measures population sparsity - how many cells are recruited:
%   - Fewer active cells = sparse coding (high selectivity)
%   - More active cells = distributed coding (low selectivity)


%% 5.1: Active Cells Distribution Across Conditions
plotMedianActiveCellsBars_Grid(ActiveCells_Grid, Rec, params, conditions, Skip,'Position', Position, 'Type', Type);

%% 5.2: Active Cells vs Entropy Correlation
% Key Question: Are low-entropy states driven by fewer active cells?
% Expected: Positive correlation (fewer active cells = fewer possible states = lower entropy)
activecells_entropy_corr = plotActiveCellsCorrelation_GridVsEntropy(Entropy_Grid, ActiveCells_Grid, ...
    params, conditions, Skip, 'Position', Position, 'Type', Type, 'UseZScored', UseZScored);

%% 5.3: Temporal Dynamics of Active Cells
% Show how population recruitment evolves during trials
plotActiveCellsTimeCourse_Grid(ActiveCells_Grid, params, conditions, Skip,'Position', Position, 'Type', Type);

%% 5.4: Active Cells vs Dispersion Correlation
% Key Question: Does sparse coding (fewer active cells) relate to spatial concentration?
% Expected: Negative correlation (fewer active cells = more concentrated = lower dispersion)
activecells_dispersion_corr = plotActiveCellsCorrelation_GridVsDispersion(Dispersion_Grid, ActiveCells_Grid, params, conditions, Skip, 'Position', Position, 'Type', Type);

fprintf('\n=== Active Cells Analysis Complete ===\n');

%% =========================================================================
%% Section 6: Save Figures
%% =========================================================================

% Create folder path based on Type, UseZScored, and Position
zScoreLabel = '';
if UseZScored
    zScoreLabel = '_ZScored';
end

% Save Entropy Comparison figures
savePathEntropy = sprintf('Fig. 5 Model\\Comparisons\\Entropy\\%s%s_%s\\', ...
    Type, zScoreLabel, Position);
fprintf('\nSaving Entropy Comparison figures to: %s\n', savePathEntropy);
saveMyFig("Fig5_EntropyComparison", savePathEntropy, 'All')

% Save Dispersion Analysis figures
savePathDispersion = sprintf('Fig. 5 Model\\Comparisons\\Dispersion\\%s%s_%s\\', ...
    Type, zScoreLabel, Position);
fprintf('Saving Dispersion Analysis figures to: %s\n', savePathDispersion);
saveMyFig("Fig5_DispersionAnalysis", savePathDispersion, 'All')

% Save Dispersion Metric Explanation figures (Figure 1: Binary Only, Figure 2: Comparison)
savePathExplanation = sprintf('Fig. 5 Model\\Comparisons\\Dispersion\\Explanation\\');
fprintf('Saving Dispersion Metric Explanation figures to: %s\n', savePathExplanation);
saveMyFig("Figure 1: Binary Dispersion Metric", savePathExplanation, 'All')
saveMyFig("Figure 2: Binary vs Weighted Dispersion", savePathExplanation, 'All')

% Save Active Cells Analysis figures
savePathActiveCells = sprintf('Fig. 5 Model\\Comparisons\\ActiveCells\\%s%s_%s\\', ...
    Type, zScoreLabel, Position);
fprintf('Saving Active Cells Analysis figures to: %s\n', savePathActiveCells);
saveMyFig("Fig5_ActiveCellsAnalysis", savePathActiveCells, 'All')

% Save Moran's I Analysis figures (if calculated)
if include_MoransI
    savePathMoransI = sprintf('Fig. 5 Model\\Comparisons\\MoransI\\%s%s_%s\\', ...
        Type, zScoreLabel, Position);
    fprintf('Saving Moran''s I Analysis figures to: %s\n', savePathMoransI);
    saveMyFig("Fig5_MoransIAnalysis", savePathMoransI, 'All')

    % Save Moran's I Metric Explanation figures
    savePathMoransIExplanation = sprintf('Fig. 5 Model\\Comparisons\\MoransI\\Explanation\\');
    fprintf('Saving Moran''s I Metric Explanation figures to: %s\n', savePathMoransIExplanation);
    saveMyFig("Figure 1: Moran's I Calculation", savePathMoransIExplanation, 'All')
    saveMyFig("Figure 2: Entropy vs Dispersion vs Moran's I", savePathMoransIExplanation, 'All')
end

if include_MoransI
    fprintf('\n=== Figure 5 Entropy, Dispersion, Active Cells & Moran''s I Analysis Complete ===\n');
else
    fprintf('\n=== Figure 5 Entropy, Dispersion, and Active Cells Analysis Complete ===\n');
end

%% =========================================================================
%% Helper Function: Calculate Entropy from Grid Data
%% =========================================================================
function [ent_raw, ent_rz, ent_tz, entZ_raw, entZ_rz, entZ_tz, ent_bin, ent_bin_an, ent_bin_all] = ...
    calculate_entropy_from_grid(gridData, gridDimensions, animalMean, animalStd, conditionMean, conditionStd, binarisation_threshold)
% CALCULATE_ENTROPY_FROM_GRID - Calculate population entropy from Grid40 data
%
% INPUTS:
%   gridData       - Grid data [gridY × gridX × nTimepoints × nTrials]
%   gridDimensions - [gridY, gridX] dimensions
%   animalMean     - (Optional) Per-animal mean for BIN_AN calculation
%   animalStd      - (Optional) Per-animal std for BIN_AN calculation
%   globalMean     - (Optional) GLOBAL mean (across all conditions) for BIN_ALL calculation
%   globalStd      - (Optional) GLOBAL std (across all conditions) for BIN_ALL calculation
%
% OUTPUTS:
%   ent_raw     - Raw entropy [nTrials × nTimepoints]
%   ent_rz      - Recording-zscore entropy
%   ent_tz      - Trial-zscore entropy
%   entZ_*      - Same metrics on z-scored grid data (Raw, RZ, TZ only)
%   ent_bin     - Entropy from binarised data (per-recording z-score > 1.5σ threshold)
%                 Note: BIN already includes z-scoring in the binarisation process
%   ent_bin_an  - Entropy from binarised data (per-animal z-score > 1.5σ threshold)
%                 Note: BIN_AN already includes z-scoring in the binarisation process
%   ent_bin_all - Entropy from binarised data (GLOBAL z-score > 1.5σ threshold, across all conditions)
%                 Note: BIN_ALL already includes z-scoring in the binarisation process

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
        error('Expected Grid data to have 4 dimensions: [gridY, gridX, nTimepoints, nTrials]');
    end

    % Reshape grid to treat each grid cell as a "neuron"
    % From [gridY × gridX × nTimepoints × nTrials] to [nGridCells × nTimepoints × nTrials]
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
            % Z-score across grid cells
            if std(activity) > 0
                activity_z = (activity - mean(activity)) / std(activity);
            else
                activity_z = activity;
            end
            entZ_raw(trial, t) = population_entropy(activity_z);
        end
    end

    %% Calculate BINARISED entropy (inspired by BinariseDataOnTheGrid.py)
    % Initialize binarised entropy array
    ent_bin = zeros(nTrials, nTimepoints);

    for trial = 1:nTrials
        for t = 1:nTimepoints
            % Extract grid cell activities at this trial/timepoint
            activity = gridData_reshaped(:, t, trial);

            % Binarise: z-score → threshold → binary
            % Note: z-scoring is inherent to the binarisation process
            if std(activity) > 0
                activity_z = (activity - mean(activity)) / std(activity);
                activity_bin = double(activity_z > binarisation_threshold);
            else
                activity_bin = zeros(size(activity));
            end
            ent_bin(trial, t) = population_entropy(activity_bin);
        end
    end

    %% Calculate BINARISED entropy with PER-ANIMAL thresholds (BIN_AN)
    % Initialize BIN_AN entropy array
    ent_bin_an = zeros(nTrials, nTimepoints);

    % Check if per-animal statistics are provided
    if nargin >= 4 && ~isempty(animalMean) && ~isempty(animalStd) && animalStd > 0
        % Per-animal binarisation available
        for trial = 1:nTrials
            for t = 1:nTimepoints
                % Extract grid cell activities at this trial/timepoint
                activity = gridData_reshaped(:, t, trial);

                % Binarise using per-animal statistics: z-score → threshold → binary
                % Note: z-scoring is inherent to the binarisation process
                activity_z_animal = (activity - animalMean) / animalStd;
                activity_bin_an = double(activity_z_animal > binarisation_threshold);
                ent_bin_an(trial, t) = population_entropy(activity_bin_an);
            end
        end
    else
        % No per-animal statistics: fall back to per-recording (same as BIN)
        ent_bin_an = ent_bin;
    end

    %% Calculate BINARISED entropy with GLOBAL thresholds (BIN_ALL)
    % Initialize BIN_ALL entropy array
    ent_bin_all = zeros(nTrials, nTimepoints);

    % Check if GLOBAL statistics are provided (across ALL conditions)
    if nargin >= 6 && ~isempty(conditionMean) && ~isempty(conditionStd) && conditionStd > 0
        % GLOBAL binarisation available (across all conditions)
        for trial = 1:nTrials
            for t = 1:nTimepoints
                % Extract grid cell activities at this trial/timepoint
                activity = gridData_reshaped(:, t, trial);

                % Binarise using GLOBAL statistics: z-score → threshold → binary
                % Note: z-scoring is inherent to the binarisation process
                activity_z_global = (activity - conditionMean) / conditionStd;
                activity_bin_all = double(activity_z_global > binarisation_threshold);
                ent_bin_all(trial, t) = population_entropy(activity_bin_all);
            end
        end
    else
        % No GLOBAL statistics: fall back to per-recording (same as BIN)
        ent_bin_all = ent_bin;
    end

    % Calculate trial-zscore (TZ): normalize each trial relative to pre-stimulus baseline
    baselineFrames = 1:80;  % Pre-stimulus period
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

    % Calculate recording-zscore (RZ): normalize across all trials
    ent_rz = (ent_raw - mean(ent_raw(:))) / std(ent_raw(:));
    entZ_rz = (entZ_raw - mean(entZ_raw(:))) / std(entZ_raw(:));
end

%% =========================================================================
%% Helper Function: Calculate Dispersion from Grid Data
%% =========================================================================
function [disp_raw, disp_rz, disp_tz, dispZ_raw, dispZ_rz, dispZ_tz, disp_bin, disp_bin_an, disp_bin_all] = ...
    calculate_dispersion_from_grid(gridData, gridDimensions, animalMean, animalStd, globalMean, globalStd, binarisation_threshold)
% CALCULATE_DISPERSION_FROM_GRID - Calculate spatial dispersion ("blobness") from Grid40 data
%
% Dispersion measures how spatially concentrated vs. dispersed active grid cells are.
% Low dispersion = tight blob (organized activity)
% High dispersion = scattered activity (disorganized)
%
% Based on BinariseDataOnTheGrid.py:
%   For each timepoint:
%     1. Find active (or weighted by activity) grid cells
%     2. Calculate centroid (center of mass)
%     3. Compute average Euclidean distance from centroid
%
% INPUTS:
%   gridData       - Grid data [gridY × gridX × nTimepoints × nTrials]
%   gridDimensions - [gridY, gridX] dimensions
%   animalMean     - (Optional) Per-animal mean for BIN_AN calculation
%   animalStd      - (Optional) Per-animal std for BIN_AN calculation
%   globalMean     - (Optional) GLOBAL mean (across all conditions) for BIN_ALL calculation
%   globalStd      - (Optional) GLOBAL std (across all conditions) for BIN_ALL calculation
%
% OUTPUTS:
%   disp_raw     - Raw dispersion [nTrials × nTimepoints] (weighted by activity)
%   disp_rz      - Recording-zscore dispersion
%   disp_tz      - Trial-zscore dispersion
%   dispZ_*      - Same metrics on z-scored grid data (Raw, RZ, TZ only)
%   disp_bin     - Dispersion from binarised data (per-recording z-score > 1.5σ threshold)
%   disp_bin_an  - Dispersion from binarised data (per-animal z-score > 1.5σ threshold)
%   disp_bin_all - Dispersion from binarised data (GLOBAL z-score > 1.5σ threshold)

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
        error('Expected Grid data to have 4 dimensions: [gridY, gridX, nTimepoints, nTrials]');
    end

    % Generate grid coordinates
    % Create coordinate matrix: each grid cell has (row, col) position
    [gridX_coords, gridY_coords] = meshgrid(1:gridX, 1:gridY);
    grid_coords = [gridY_coords(:), gridX_coords(:)]; % [nGridCells × 2]

    % Reshape grid to treat each grid cell as a "neuron"
    % From [gridY × gridX × nTimepoints × nTrials] to [nGridCells × nTimepoints × nTrials]
    nGridCells = gridY * gridX;
    gridData_reshaped = reshape(gridData, [nGridCells, nTimepoints, nTrials]);

    % Initialize output arrays
    disp_raw = zeros(nTrials, nTimepoints);
    dispZ_raw = zeros(nTrials, nTimepoints);

    %% Calculate WEIGHTED dispersion for continuous data (Raw)
    fprintf('    Calculating dispersion (Raw)...\n');
    for trial = 1:nTrials
        for t = 1:nTimepoints
            % Extract grid cell activities at this trial/timepoint
            activity = gridData_reshaped(:, t, trial);

            % Calculate weighted dispersion
            disp_raw(trial, t) = calculate_weighted_dispersion(activity, grid_coords);

            % Calculate dispersion on z-scored data
            if std(activity) > 0
                activity_z = (activity - mean(activity)) / std(activity);
            else
                activity_z = activity;
            end
            dispZ_raw(trial, t) = calculate_weighted_dispersion(activity_z, grid_coords);
        end
    end

    %% Calculate BINARISED dispersion (BIN: per-recording threshold)
    disp_bin = zeros(nTrials, nTimepoints);

    fprintf('    Calculating dispersion (BIN)...\n');
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

            % Calculate binary dispersion (matching Python implementation)
            disp_bin(trial, t) = calculate_binary_dispersion(activity_bin, grid_coords);
        end
    end

    %% Calculate BINARISED dispersion with PER-ANIMAL thresholds (BIN_AN)
    disp_bin_an = zeros(nTrials, nTimepoints);

    if nargin >= 4 && ~isempty(animalMean) && ~isempty(animalStd) && animalStd > 0
        fprintf('    Calculating dispersion (BIN_AN)...\n');
        for trial = 1:nTrials
            for t = 1:nTimepoints
                activity = gridData_reshaped(:, t, trial);

                % Binarise using per-animal statistics
                activity_z_animal = (activity - animalMean) / animalStd;
                activity_bin_an = double(activity_z_animal > binarisation_threshold);
                disp_bin_an(trial, t) = calculate_binary_dispersion(activity_bin_an, grid_coords);
            end
        end
    else
        % Fall back to per-recording (same as BIN)
        disp_bin_an = disp_bin;
    end

    %% Calculate BINARISED dispersion with GLOBAL thresholds (BIN_ALL)
    disp_bin_all = zeros(nTrials, nTimepoints);

    if nargin >= 6 && ~isempty(globalMean) && ~isempty(globalStd) && globalStd > 0
        fprintf('    Calculating dispersion (BIN_ALL)...\n');
        for trial = 1:nTrials
            for t = 1:nTimepoints
                activity = gridData_reshaped(:, t, trial);

                % Binarise using GLOBAL statistics
                activity_z_global = (activity - globalMean) / globalStd;
                activity_bin_all = double(activity_z_global > binarisation_threshold);
                disp_bin_all(trial, t) = calculate_binary_dispersion(activity_bin_all, grid_coords);
            end
        end
    else
        % Fall back to per-recording (same as BIN)
        disp_bin_all = disp_bin;
    end

    %% Calculate trial-zscore (TZ): normalize each trial relative to pre-stimulus baseline
    baselineFrames = 1:80;  % Pre-stimulus period
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

    %% Calculate recording-zscore (RZ): normalize across all trials
    disp_rz = (disp_raw - mean(disp_raw(:))) / std(disp_raw(:));
    dispZ_rz = (dispZ_raw - mean(dispZ_raw(:))) / std(dispZ_raw(:));
end

%% =========================================================================
%% Sub-helper: Calculate weighted dispersion for continuous data
%% =========================================================================
function disp = calculate_weighted_dispersion(activity, grid_coords)
% Calculate dispersion weighted by activity levels
% Uses activity-weighted centroid and activity-weighted average distance

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

%% =========================================================================
%% Sub-helper: Calculate binary dispersion (exact Python implementation)
%% =========================================================================
function disp = calculate_binary_dispersion(activity_bin, grid_coords)
% Calculate dispersion for binary data (matches BinariseDataOnTheGrid.py)
% Finds active cells, computes centroid, averages Euclidean distances

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

%% =========================================================================
%% Helper Function: Calculate Active Cells Count from Grid Data
%% =========================================================================
function [actcells_raw, actcells_rz, actcells_tz, actcellsZ_raw, actcellsZ_rz, actcellsZ_tz, actcells_bin, actcells_bin_an, actcells_bin_all] = ...
    calculate_active_cells_from_grid(gridData, gridDimensions, animalMean, animalStd, globalMean, globalStd, binarisation_threshold)
% CALCULATE_ACTIVE_CELLS_FROM_GRID - Calculate number of active grid cells per frame
%
% Counts how many grid cells are active at each timepoint. This provides a measure
% of population sparsity and recruitment across different conditions.
%
% INPUTS:
%   gridData       - Grid data [gridY × gridX × nTimepoints × nTrials]
%   gridDimensions - [gridY, gridX] dimensions
%   animalMean     - (Optional) Per-animal mean for BIN_AN calculation
%   animalStd      - (Optional) Per-animal std for BIN_AN calculation
%   globalMean     - (Optional) GLOBAL mean (across all conditions) for BIN_ALL calculation
%   globalStd      - (Optional) GLOBAL std (across all conditions) for BIN_ALL calculation
%   binarisation_threshold - Z-score threshold for binarisation (default: 2σ)
%
% OUTPUTS:
%   actcells_raw     - Active cells count from raw data [nTrials × nTimepoints]
%   actcells_rz      - Recording-zscore active cells count
%   actcells_tz      - Trial-zscore active cells count
%   actcellsZ_*      - Same metrics on z-scored grid data (Raw, RZ, TZ only)
%   actcells_bin     - Active cells from binarised data (per-recording z-score > threshold)
%   actcells_bin_an  - Active cells from binarised data (per-animal z-score > threshold)
%   actcells_bin_all - Active cells from binarised data (GLOBAL z-score > threshold)

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
        error('Expected Grid data to have 4 dimensions: [gridY, gridX, nTimepoints, nTrials]');
    end

    % Reshape grid to treat each grid cell as a "neuron"
    % From [gridY × gridX × nTimepoints × nTrials] to [nGridCells × nTimepoints × nTrials]
    nGridCells = gridY * gridX;
    gridData_reshaped = reshape(gridData, [nGridCells, nTimepoints, nTrials]);

    % Activity threshold for continuous data (to avoid counting noise as activity)
    activity_threshold = 0.01;

    % Initialize output arrays
    actcells_raw = zeros(nTrials, nTimepoints);
    actcellsZ_raw = zeros(nTrials, nTimepoints);

    %% Calculate active cells count for RAW continuous data
    for trial = 1:nTrials
        for t = 1:nTimepoints
            % Extract grid cell activities at this trial/timepoint
            activity = gridData_reshaped(:, t, trial);

            % Count cells above activity threshold
            actcells_raw(trial, t) = sum(activity > activity_threshold);

            % Count active cells on z-scored data
            if std(activity) > 0
                activity_z = (activity - mean(activity)) / std(activity);
            else
                activity_z = activity;
            end
            actcellsZ_raw(trial, t) = sum(activity_z > activity_threshold);
        end
    end

    %% Calculate BINARISED active cells count (BIN: per-recording threshold)
    actcells_bin = zeros(nTrials, nTimepoints);

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

            % Count active cells (binary = 1)
            actcells_bin(trial, t) = sum(activity_bin);
        end
    end

    %% Calculate BINARISED active cells with PER-ANIMAL thresholds (BIN_AN)
    actcells_bin_an = zeros(nTrials, nTimepoints);

    if nargin >= 4 && ~isempty(animalMean) && ~isempty(animalStd) && animalStd > 0
        % Per-animal binarisation available
        for trial = 1:nTrials
            for t = 1:nTimepoints
                activity = gridData_reshaped(:, t, trial);

                % Binarise using per-animal statistics
                activity_z_animal = (activity - animalMean) / animalStd;
                activity_bin_an = double(activity_z_animal > binarisation_threshold);

                % Count active cells
                actcells_bin_an(trial, t) = sum(activity_bin_an);
            end
        end
    else
        % No per-animal statistics: fall back to per-recording (same as BIN)
        actcells_bin_an = actcells_bin;
    end

    %% Calculate BINARISED active cells with GLOBAL thresholds (BIN_ALL)
    actcells_bin_all = zeros(nTrials, nTimepoints);

    if nargin >= 6 && ~isempty(globalMean) && ~isempty(globalStd) && globalStd > 0
        % GLOBAL binarisation available
        for trial = 1:nTrials
            for t = 1:nTimepoints
                activity = gridData_reshaped(:, t, trial);

                % Binarise using GLOBAL statistics
                activity_z_global = (activity - globalMean) / globalStd;
                activity_bin_all = double(activity_z_global > binarisation_threshold);

                % Count active cells
                actcells_bin_all(trial, t) = sum(activity_bin_all);
            end
        end
    else
        % No GLOBAL statistics: fall back to per-recording (same as BIN)
        actcells_bin_all = actcells_bin;
    end

    %% Calculate trial-zscore (TZ): normalize each trial relative to pre-stimulus baseline
    baselineFrames = 1:80;  % Pre-stimulus period
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

    %% Calculate recording-zscore (RZ): normalize across all trials
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

%% =========================================================================
%% Helper Function: Calculate Moran's I from Grid Data (BIN_AN only)
%% =========================================================================
function morans_bin_an = calculate_moransI_BIN_AN(gridData, gridDimensions, animalMean, animalStd, binarisation_threshold)
% CALCULATE_MORANSI_BIN_AN - Calculate Moran's I spatial autocorrelation for BIN_AN data
%
% Moran's I measures spatial autocorrelation - whether similar values cluster together.
% Positive values indicate clustering, negative values indicate dispersion.
%
% INPUTS:
%   gridData       - Grid data [gridY × gridX × nTimepoints × nTrials]
%   gridDimensions - [gridY, gridX] dimensions
%   animalMean     - Per-animal mean for BIN_AN calculation
%   animalStd      - Per-animal std for BIN_AN calculation
%
% OUTPUTS:
%   morans_bin_an  - Moran's I for binarised data (per-animal z-score > 1.5σ threshold)
%                    [nTrials × nTimepoints]

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
        error('Expected Grid data to have 4 dimensions: [gridY, gridX, nTimepoints, nTrials]');
    end

    % Create spatial weight matrix (using nearest neighbor distance only)
    % This matches the approach in getMoransI4.m
    valueMap = rand(gridY, gridX);  % Template for dimensions
    currDist = 1;  % Only consider nearest neighbors

    % Compute distance matrix using mL_distanceMat
    distanceMat = squareform(mL_distanceMat(valueMap(:,:,1)));
    uniqueDistances = unique(distanceMat);
    uniqueDistances(uniqueDistances == 0) = [];  % Remove self-distance

    % Create weight matrix (only nearest neighbors have non-zero weights)
    currDistInds = ismember(distanceMat, uniqueDistances(1:currDist));
    currWeightMat = zeros(size(distanceMat));
    currWeightMat(currDistInds) = distanceMat(currDistInds);
    currWeightMat(currWeightMat == inf) = 0;

    % Initialize output array
    morans_bin_an = zeros(nTrials, nTimepoints);

    % Check if per-animal statistics are provided
    if nargin >= 3 && ~isempty(animalMean) && ~isempty(animalStd) && animalStd > 0
        % Calculate Moran's I for each trial and timepoint
        fprintf('      Calculating Moran''s I (BIN_AN)...\n');
        for trial = 1:nTrials
            for t = 1:nTimepoints
                % Extract grid at this trial/timepoint
                grid_2D = gridData(:, :, t, trial);

                % Binarise using per-animal statistics
                grid_2D_z = (grid_2D - animalMean) / animalStd;
                grid_2D_bin = double(grid_2D_z > binarisation_threshold);

                % Calculate Moran's I using mL_moransI
                morans_bin_an(trial, t) = mL_moransI(grid_2D_bin, currWeightMat);
            end
        end
    else
        % No per-animal statistics available - return NaN
        fprintf('      Warning: No per-animal statistics for Moran''s I calculation\n');
        morans_bin_an(:) = NaN;
    end
end

