%% =========================================================================
%% Fig 5: Entropy & Dispersion Performance Comparison - Hit vs Miss
%% =========================================================================
% This script compares population entropy, dispersion, and Moran's I between
% Hit and Miss trials within a single condition (Expert).
%
% Comparison: Hit trials vs Miss (or Nonresponsive) trials
% Data source: Grid40 data (spatially rasterized, 13×26 grid cells)
%
% Purpose: Understand how behavioral performance (Hit vs Miss) affects:
%   1. Population entropy (state diversity)
%   2. Spatial dispersion (activity concentration)
%   3. Spatial autocorrelation (Moran's I)

% Load data structures
% load(mba_p('RawData3.mat'),'ActivityData');
% load(mba_p('RawData3.mat'),'params');
% load(mba_p('RawData3.mat'),'Performance');
% load(mba_p('RawData3.mat'),'Stimuli'); % Stimuli position information for trial filtering
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

% Performance comparison parameters
condition = 'Expert';  % Single condition to analyze
MissNonresponsive = 2; % 0 = Miss, 1 = Nonresponsive, 2 = Miss & Nonresponsive
binarisation_threshold = 2;  % Standard deviations for z-score binarisation

% Define performance state colors
params.Hit = [0, 0, 0];             % Black for hit trials
params.Miss = [0.8, 0, 0];          % Red for miss trials
params.Nonresponsive = [1, 0.1, 0.8]; % Neon pink for nonresponsive trials

%% =========================================================================
%% Section 1: Calculate Entropy and Dispersion from Grid40 Data
%% =========================================================================
fprintf('\n=== Calculating Entropy and Dispersion from Grid40 Data (Hit vs Miss) ===\n');
fprintf('Condition: %s\n', condition);
fprintf('MissNonresponsive mode: %d (0=Miss, 1=Nonresponsive, 2=Both)\n', MissNonresponsive);

% Grid parameters (consistent with Figure 3)
gridSize = 40;
gridDimensions = [13 26];  % Grid structure: 13 rows × 26 columns

% Initialize Grid entropy structures for Hit and Miss
Entropy_Grid_Hit = struct();
Entropy_Grid_Miss = struct();

% For Mode 2 (three-way comparison): Initialize Nonresponsive structure
if MissNonresponsive == 2
    Entropy_Grid_Nonresponsive = struct();
end

% Initialize Grid dispersion structures for Hit and Miss
Dispersion_Grid_Hit = struct();
Dispersion_Grid_Miss = struct();

% For Mode 2 (three-way comparison): Initialize Nonresponsive structure
if MissNonresponsive == 2
    Dispersion_Grid_Nonresponsive = struct();
end

% Initialize Grid Moran's I structures (spatial autocorrelation)
% Note: Only BIN_AN for Position1 is currently calculated (others are placeholders)
MoransI_Grid_Hit = struct();
MoransI_Grid_Miss = struct();

% For Mode 2 (three-way comparison): Initialize Nonresponsive structure
if MissNonresponsive == 2
    MoransI_Grid_Nonresponsive = struct();
end

%% =========================================================================
%% Section 1.1: Calculate GLOBAL statistics for BIN_ALL (across ALL trials)
%% =========================================================================
fprintf('\n--- Calculating GLOBAL statistics for BIN_ALL (across ALL trials) ---\n');

% Collect ALL grid data from ALL trials (both Hit and Miss) for normalization
allGlobalGridData_P1 = [];
allGlobalGridData_P3 = [];

fprintf('  Collecting data from condition: %s\n', condition);

% Get recording count
if ~isfield(ActivityData, condition)
    error('Condition %s not found in ActivityData', condition);
end
nRecs = length(ActivityData.(condition));

% Define condition with Individual suffix for Grid40 access
conditionIndividual = [condition 'Individual'];

% Check if Grid40 data exists for this condition
if ~isfield(Grid40, conditionIndividual)
    error('No Grid40 data found for %s', conditionIndividual);
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

% Calculate GLOBAL statistics (mean and std across ALL data points)
globalMean_P1 = [];
globalStd_P1 = [];
globalMean_P3 = [];
globalStd_P3 = [];

if ~isempty(allGlobalGridData_P1)
    globalMean_P1 = mean(allGlobalGridData_P1(:));
    globalStd_P1 = std(allGlobalGridData_P1(:));
    fprintf('\n=== GLOBAL BIN_ALL Statistics ===\n');
    fprintf('P1: Global mean=%.4f, std=%.4f (across ALL trials)\n', globalMean_P1, globalStd_P1);
else
    fprintf('\nWarning: No P1 data collected for global statistics\n');
end

if ~isempty(allGlobalGridData_P3)
    globalMean_P3 = mean(allGlobalGridData_P3(:));
    globalStd_P3 = std(allGlobalGridData_P3(:));
    fprintf('P3: Global mean=%.4f, std=%.4f (across ALL trials)\n', globalMean_P3, globalStd_P3);
else
    fprintf('Warning: No P3 data collected for global statistics\n');
end

%% =========================================================================
%% Section 1.2: Process condition (calculate entropies and dispersion for Hit vs Miss)
%% =========================================================================

fprintf('\nProcessing condition: %s\n', condition);

% Initialize entropy storage for Hit and Miss (and Nonresponsive for Mode 2)
if MissNonresponsive == 2
    perfTypes = {'Hit', 'Miss', 'Nonresponsive'};
else
    perfTypes = {'Hit', 'Miss'};
end

for perfType = perfTypes
    perfField = perfType{1};

    if strcmp(perfField, 'Hit')
        Entropy_Grid = Entropy_Grid_Hit;
        Dispersion_Grid = Dispersion_Grid_Hit;
        MoransI_Grid = MoransI_Grid_Hit;
    elseif strcmp(perfField, 'Miss')
        Entropy_Grid = Entropy_Grid_Miss;
        Dispersion_Grid = Dispersion_Grid_Miss;
        MoransI_Grid = MoransI_Grid_Miss;
    else  % Nonresponsive (Mode 2 only)
        Entropy_Grid = Entropy_Grid_Nonresponsive;
        Dispersion_Grid = Dispersion_Grid_Nonresponsive;
        MoransI_Grid = MoransI_Grid_Nonresponsive;
    end

    % Initialize structures for this performance type
    Entropy_Grid.(condition).Position1 = struct();
    Entropy_Grid.(condition).Position3 = struct();
    Entropy_Grid.(condition).All = struct();

    % Initialize CELL ARRAYS for different normalization types
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
    end

    % Initialize dispersion storage
    Dispersion_Grid.(condition).Position1 = struct();
    Dispersion_Grid.(condition).Position3 = struct();
    Dispersion_Grid.(condition).All = struct();

    % Initialize CELL ARRAYS for dispersion
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
    end

    % Initialize Moran's I storage (spatial autocorrelation)
    MoransI_Grid.(condition).Position1 = struct();
    MoransI_Grid.(condition).Position3 = struct();
    MoransI_Grid.(condition).All = struct();

    % Initialize CELL ARRAYS for Moran's I
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

    % Store back to appropriate structure
    if strcmp(perfField, 'Hit')
        Entropy_Grid_Hit = Entropy_Grid;
        Dispersion_Grid_Hit = Dispersion_Grid;
        MoransI_Grid_Hit = MoransI_Grid;
    elseif strcmp(perfField, 'Miss')
        Entropy_Grid_Miss = Entropy_Grid;
        Dispersion_Grid_Miss = Dispersion_Grid;
        MoransI_Grid_Miss = MoransI_Grid;
    else  % Nonresponsive (Mode 2 only)
        Entropy_Grid_Nonresponsive = Entropy_Grid;
        Dispersion_Grid_Nonresponsive = Dispersion_Grid;
        MoransI_Grid_Nonresponsive = MoransI_Grid;
    end
end

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

    % Check if Performance data exists for this recording
    if r > length(Performance.(condition))
        fprintf('  Recording %d: No Performance data\n', r);
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

    % Get trial indices for Hit and Miss/Nonresponsive trials
    % Following the pattern from Figure4's plot function
    % Use Stimuli data to map absolute trial indices to position-relative indices

    % Check if Stimuli data exists for this recording
    if r > length(Stimuli.(condition))
        fprintf('  Recording %d: No Stimuli data\n', r);
        continue;
    end

    % Get P1 trial numbers (absolute indices from full recording)
    if ~isfield(Stimuli.(condition)(r), 'TrialsPosition1') || isempty(Stimuli.(condition)(r).TrialsPosition1)
        p1TrialsAbsolute = [];
    else
        p1TrialsAbsolute = Stimuli.(condition)(r).TrialsPosition1;
    end

    % Get Hit trial indices (absolute indices)
    hitTrialsAbsolute = [];
    if isfield(Performance.(condition)(r), 'hit')
        hitTrialsAbsolute = Performance.(condition)(r).hit;
    elseif isfield(Performance.(condition)(r), 'hitAll')
        hitTrialsAbsolute = Performance.(condition)(r).hitAll;
    end

    % Get Miss/Nonresponsive trial indices (absolute indices)
    missTrialsAbsolute = [];
    nonrespTrialsAbsolute = [];  % For Mode 2 three-way comparison

    if MissNonresponsive == 0
        % Miss trials only (including nonresponsive)
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
        % THREE-WAY comparison: Hit vs Miss (WITHOUT nonresponsive) vs Nonresponsive
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
        % Separate Miss and Nonresponsive into two distinct groups
        missTrialsAbsolute = setdiff(miss, nonresp);  % Miss WITHOUT nonresponsive
        nonrespTrialsAbsolute = nonresp;               % Nonresponsive only
    end

    % Intersect P1 trials with Hit trials (absolute indices)
    p1HitTrialsAbsolute = intersect(p1TrialsAbsolute, hitTrialsAbsolute);
    % Intersect P1 trials with Miss trials (absolute indices)
    p1MissTrialsAbsolute = intersect(p1TrialsAbsolute, missTrialsAbsolute);
    % For Mode 2: Intersect P1 trials with Nonresponsive trials (absolute indices)
    p1NonrespTrialsAbsolute = intersect(p1TrialsAbsolute, nonrespTrialsAbsolute);

    % Map absolute indices to P1-relative indices (for indexing Grid40.P1)
    [~, hitTrialIndices] = ismember(p1HitTrialsAbsolute, p1TrialsAbsolute);
    hitTrialIndices = hitTrialIndices(hitTrialIndices > 0); % Remove zeros

    [~, missTrialIndices] = ismember(p1MissTrialsAbsolute, p1TrialsAbsolute);
    missTrialIndices = missTrialIndices(missTrialIndices > 0); % Remove zeros

    % For Mode 2: Map nonresponsive trials
    [~, nonrespTrialIndices] = ismember(p1NonrespTrialsAbsolute, p1TrialsAbsolute);
    nonrespTrialIndices = nonrespTrialIndices(nonrespTrialIndices > 0); % Remove zeros

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
            % Separate Hit and Miss trials
            gridData_P1_Hit = gridData_P1(:, :, :, hitTrialIndices);
            gridData_P1_Miss = gridData_P1(:, :, :, missTrialIndices);

            % Ensure 4D dimensions (MATLAB squeezes singleton dimensions when indexing with single value)
            sz = size(gridData_P1_Miss);
            if length(sz) < 4
                gridData_P1_Miss = reshape(gridData_P1_Miss, [sz, 1]);
            end

            % Calculate entropy for Hit trials
            if ~isempty(hitTrialIndices) && length(hitTrialIndices) > 0
                [ent_raw_P1_Hit, ent_rz_P1_Hit, ent_tz_P1_Hit, entZ_raw_P1_Hit, entZ_rz_P1_Hit, entZ_tz_P1_Hit, ent_bin_P1_Hit, ent_bin_an_P1_Hit, ent_bin_all_P1_Hit] = ...
                    calculate_entropy_from_grid(gridData_P1_Hit, gridDimensions, animalMean_P1, animalStd_P1, globalMean_P1, globalStd_P1, binarisation_threshold);

                % Store Hit entropy values
                Entropy_Grid_Hit.(condition).Position1.RecordingEntropy_Raw{end+1} = ent_raw_P1_Hit;
                Entropy_Grid_Hit.(condition).Position1.RecordingEntropy_RZ{end+1} = ent_rz_P1_Hit;
                Entropy_Grid_Hit.(condition).Position1.RecordingEntropy_TZ{end+1} = ent_tz_P1_Hit;
                Entropy_Grid_Hit.(condition).Position1.RecordingEntropy_BIN{end+1} = ent_bin_P1_Hit;
                Entropy_Grid_Hit.(condition).Position1.RecordingEntropy_BIN_AN{end+1} = ent_bin_an_P1_Hit;
                Entropy_Grid_Hit.(condition).Position1.RecordingEntropy_BIN_ALL{end+1} = ent_bin_all_P1_Hit;
                Entropy_Grid_Hit.(condition).Position1.RecordingEntropyZ_Raw{end+1} = entZ_raw_P1_Hit;
                Entropy_Grid_Hit.(condition).Position1.RecordingEntropyZ_RZ{end+1} = entZ_rz_P1_Hit;
                Entropy_Grid_Hit.(condition).Position1.RecordingEntropyZ_TZ{end+1} = entZ_tz_P1_Hit;

                % Calculate dispersion for Hit trials
                [disp_raw_P1_Hit, disp_rz_P1_Hit, disp_tz_P1_Hit, dispZ_raw_P1_Hit, dispZ_rz_P1_Hit, dispZ_tz_P1_Hit, disp_bin_P1_Hit, disp_bin_an_P1_Hit, disp_bin_all_P1_Hit] = ...
                    calculate_dispersion_from_grid(gridData_P1_Hit, gridDimensions, animalMean_P1, animalStd_P1, globalMean_P1, globalStd_P1, binarisation_threshold);

                % Store Hit dispersion values
                Dispersion_Grid_Hit.(condition).Position1.RecordingDispersion_Raw{end+1} = disp_raw_P1_Hit;
                Dispersion_Grid_Hit.(condition).Position1.RecordingDispersion_RZ{end+1} = disp_rz_P1_Hit;
                Dispersion_Grid_Hit.(condition).Position1.RecordingDispersion_TZ{end+1} = disp_tz_P1_Hit;
                Dispersion_Grid_Hit.(condition).Position1.RecordingDispersion_BIN{end+1} = disp_bin_P1_Hit;
                Dispersion_Grid_Hit.(condition).Position1.RecordingDispersion_BIN_AN{end+1} = disp_bin_an_P1_Hit;
                Dispersion_Grid_Hit.(condition).Position1.RecordingDispersion_BIN_ALL{end+1} = disp_bin_all_P1_Hit;
                Dispersion_Grid_Hit.(condition).Position1.RecordingDispersionZ_Raw{end+1} = dispZ_raw_P1_Hit;
                Dispersion_Grid_Hit.(condition).Position1.RecordingDispersionZ_RZ{end+1} = dispZ_rz_P1_Hit;
                Dispersion_Grid_Hit.(condition).Position1.RecordingDispersionZ_TZ{end+1} = dispZ_tz_P1_Hit;

                % Calculate Moran's I for Hit trials (spatial autocorrelation)
                if include_MoransI
                    morans_bin_an_P1_Hit = calculate_moransI_BIN_AN(gridData_P1_Hit, gridDimensions, animalMean_P1, animalStd_P1, binarisation_threshold);
                    MoransI_Grid_Hit.(condition).Position1.RecordingMoransI_BIN_AN{end+1} = morans_bin_an_P1_Hit;
                end
            end

            % Calculate entropy for Miss trials
            if ~isempty(missTrialIndices) && length(missTrialIndices) > 0
                [ent_raw_P1_Miss, ent_rz_P1_Miss, ent_tz_P1_Miss, entZ_raw_P1_Miss, entZ_rz_P1_Miss, entZ_tz_P1_Miss, ent_bin_P1_Miss, ent_bin_an_P1_Miss, ent_bin_all_P1_Miss] = ...
                    calculate_entropy_from_grid(gridData_P1_Miss, gridDimensions, animalMean_P1, animalStd_P1, globalMean_P1, globalStd_P1, binarisation_threshold);

                % Store Miss entropy values
                Entropy_Grid_Miss.(condition).Position1.RecordingEntropy_Raw{end+1} = ent_raw_P1_Miss;
                Entropy_Grid_Miss.(condition).Position1.RecordingEntropy_RZ{end+1} = ent_rz_P1_Miss;
                Entropy_Grid_Miss.(condition).Position1.RecordingEntropy_TZ{end+1} = ent_tz_P1_Miss;
                Entropy_Grid_Miss.(condition).Position1.RecordingEntropy_BIN{end+1} = ent_bin_P1_Miss;
                Entropy_Grid_Miss.(condition).Position1.RecordingEntropy_BIN_AN{end+1} = ent_bin_an_P1_Miss;
                Entropy_Grid_Miss.(condition).Position1.RecordingEntropy_BIN_ALL{end+1} = ent_bin_all_P1_Miss;
                Entropy_Grid_Miss.(condition).Position1.RecordingEntropyZ_Raw{end+1} = entZ_raw_P1_Miss;
                Entropy_Grid_Miss.(condition).Position1.RecordingEntropyZ_RZ{end+1} = entZ_rz_P1_Miss;
                Entropy_Grid_Miss.(condition).Position1.RecordingEntropyZ_TZ{end+1} = entZ_tz_P1_Miss;

                % Calculate dispersion for Miss trials
                [disp_raw_P1_Miss, disp_rz_P1_Miss, disp_tz_P1_Miss, dispZ_raw_P1_Miss, dispZ_rz_P1_Miss, dispZ_tz_P1_Miss, disp_bin_P1_Miss, disp_bin_an_P1_Miss, disp_bin_all_P1_Miss] = ...
                    calculate_dispersion_from_grid(gridData_P1_Miss, gridDimensions, animalMean_P1, animalStd_P1, globalMean_P1, globalStd_P1, binarisation_threshold);

                % Store Miss dispersion values
                Dispersion_Grid_Miss.(condition).Position1.RecordingDispersion_Raw{end+1} = disp_raw_P1_Miss;
                Dispersion_Grid_Miss.(condition).Position1.RecordingDispersion_RZ{end+1} = disp_rz_P1_Miss;
                Dispersion_Grid_Miss.(condition).Position1.RecordingDispersion_TZ{end+1} = disp_tz_P1_Miss;
                Dispersion_Grid_Miss.(condition).Position1.RecordingDispersion_BIN{end+1} = disp_bin_P1_Miss;
                Dispersion_Grid_Miss.(condition).Position1.RecordingDispersion_BIN_AN{end+1} = disp_bin_an_P1_Miss;
                Dispersion_Grid_Miss.(condition).Position1.RecordingDispersion_BIN_ALL{end+1} = disp_bin_all_P1_Miss;
                Dispersion_Grid_Miss.(condition).Position1.RecordingDispersionZ_Raw{end+1} = dispZ_raw_P1_Miss;
                Dispersion_Grid_Miss.(condition).Position1.RecordingDispersionZ_RZ{end+1} = dispZ_rz_P1_Miss;
                Dispersion_Grid_Miss.(condition).Position1.RecordingDispersionZ_TZ{end+1} = dispZ_tz_P1_Miss;

                % Calculate Moran's I for Miss trials (spatial autocorrelation)
                if include_MoransI
                    morans_bin_an_P1_Miss = calculate_moransI_BIN_AN(gridData_P1_Miss, gridDimensions, animalMean_P1, animalStd_P1, binarisation_threshold);
                    MoransI_Grid_Miss.(condition).Position1.RecordingMoransI_BIN_AN{end+1} = morans_bin_an_P1_Miss;
                end
            end

            % FOR MODE 2: Calculate entropy and dispersion for Nonresponsive trials
            if MissNonresponsive == 2 && ~isempty(nonrespTrialIndices) && length(nonrespTrialIndices) > 0
                gridData_P1_Nonresp = gridData_P1(:, :, :, nonrespTrialIndices);

                % Ensure 4D dimensions (MATLAB squeezes singleton dimensions when indexing with single value)
                sz = size(gridData_P1_Nonresp);
                if length(sz) < 4
                    gridData_P1_Nonresp = reshape(gridData_P1_Nonresp, [sz, 1]);
                end

                [ent_raw_P1_Nonresp, ent_rz_P1_Nonresp, ent_tz_P1_Nonresp, entZ_raw_P1_Nonresp, entZ_rz_P1_Nonresp, entZ_tz_P1_Nonresp, ent_bin_P1_Nonresp, ent_bin_an_P1_Nonresp, ent_bin_all_P1_Nonresp] = ...
                    calculate_entropy_from_grid(gridData_P1_Nonresp, gridDimensions, animalMean_P1, animalStd_P1, globalMean_P1, globalStd_P1, binarisation_threshold);

                % Store Nonresponsive entropy values
                Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropy_Raw{end+1} = ent_raw_P1_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropy_RZ{end+1} = ent_rz_P1_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropy_TZ{end+1} = ent_tz_P1_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropy_BIN{end+1} = ent_bin_P1_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropy_BIN_AN{end+1} = ent_bin_an_P1_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropy_BIN_ALL{end+1} = ent_bin_all_P1_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropyZ_Raw{end+1} = entZ_raw_P1_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropyZ_RZ{end+1} = entZ_rz_P1_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropyZ_TZ{end+1} = entZ_tz_P1_Nonresp;

                % Calculate dispersion for Nonresponsive trials
                [disp_raw_P1_Nonresp, disp_rz_P1_Nonresp, disp_tz_P1_Nonresp, dispZ_raw_P1_Nonresp, dispZ_rz_P1_Nonresp, dispZ_tz_P1_Nonresp, disp_bin_P1_Nonresp, disp_bin_an_P1_Nonresp, disp_bin_all_P1_Nonresp] = ...
                    calculate_dispersion_from_grid(gridData_P1_Nonresp, gridDimensions, animalMean_P1, animalStd_P1, globalMean_P1, globalStd_P1, binarisation_threshold);

                % Store Nonresponsive dispersion values
                Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersion_Raw{end+1} = disp_raw_P1_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersion_RZ{end+1} = disp_rz_P1_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersion_TZ{end+1} = disp_tz_P1_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersion_BIN{end+1} = disp_bin_P1_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersion_BIN_AN{end+1} = disp_bin_an_P1_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersion_BIN_ALL{end+1} = disp_bin_all_P1_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersionZ_Raw{end+1} = dispZ_raw_P1_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersionZ_RZ{end+1} = dispZ_rz_P1_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersionZ_TZ{end+1} = dispZ_tz_P1_Nonresp;

                % Calculate Moran's I for Nonresponsive trials (spatial autocorrelation)
                if include_MoransI
                    morans_bin_an_P1_Nonresp = calculate_moransI_BIN_AN(gridData_P1_Nonresp, gridDimensions, animalMean_P1, animalStd_P1, binarisation_threshold);
                    MoransI_Grid_Nonresponsive.(condition).Position1.RecordingMoransI_BIN_AN{end+1} = morans_bin_an_P1_Nonresp;
                end
            end
        end
    end

    % Get P3-specific trial indices for Hit and Miss/Nonresponsive trials
    % Use Stimuli data to map absolute trial indices to P3-relative indices

    % Get P3 trial numbers (absolute indices from full recording)
    if ~isfield(Stimuli.(condition)(r), 'TrialsPosition3') || isempty(Stimuli.(condition)(r).TrialsPosition3)
        p3TrialsAbsolute = [];
    else
        p3TrialsAbsolute = Stimuli.(condition)(r).TrialsPosition3;
    end

    % Get Hit P3 trial indices (absolute indices)
    hitP3TrialsAbsolute = [];
    if isfield(Performance.(condition)(r), 'hitP3')
        hitP3TrialsAbsolute = Performance.(condition)(r).hitP3;
    elseif isfield(Performance.(condition)(r), 'hitP3All')
        hitP3TrialsAbsolute = Performance.(condition)(r).hitP3All;
    end

    % Get Miss/Nonresponsive P3 trial indices (absolute indices)
    missP3TrialsAbsolute = [];
    nonrespP3TrialsAbsolute = [];  % For Mode 2 three-way comparison

    if MissNonresponsive == 0
        % Miss P3 trials only (including nonresponsive)
        if isfield(Performance.(condition)(r), 'missP3')
            missP3TrialsAbsolute = Performance.(condition)(r).missP3;
        elseif isfield(Performance.(condition)(r), 'missP3All')
            missP3TrialsAbsolute = Performance.(condition)(r).missP3All;
        end
    elseif MissNonresponsive == 1
        % Nonresponsive P3 trials only
        if isfield(Performance.(condition)(r), 'nonresponsiveTrialsP3')
            missP3TrialsAbsolute = Performance.(condition)(r).nonresponsiveTrialsP3;
        elseif isfield(Performance.(condition)(r), 'nonresponsiveTrialsP3All')
            missP3TrialsAbsolute = Performance.(condition)(r).nonresponsiveTrialsP3All;
        end
    else  % MissNonresponsive == 2
        % THREE-WAY comparison: Hit vs Miss (WITHOUT nonresponsive) vs Nonresponsive
        missP3 = [];
        nonrespP3 = [];
        if isfield(Performance.(condition)(r), 'missP3')
            missP3 = Performance.(condition)(r).missP3;
        elseif isfield(Performance.(condition)(r), 'missP3All')
            missP3 = Performance.(condition)(r).missP3All;
        end
        if isfield(Performance.(condition)(r), 'nonresponsiveTrialsP3')
            nonrespP3 = Performance.(condition)(r).nonresponsiveTrialsP3;
        elseif isfield(Performance.(condition)(r), 'nonresponsiveTrialsP3All')
            nonrespP3 = Performance.(condition)(r).nonresponsiveTrialsP3All;
        end
        % Separate Miss and Nonresponsive into two distinct groups
        missP3TrialsAbsolute = setdiff(missP3, nonrespP3);  % Miss WITHOUT nonresponsive
        nonrespP3TrialsAbsolute = nonrespP3;                 % Nonresponsive only
    end

    % Intersect P3 trials with Hit P3 trials (absolute indices)
    p3HitTrialsAbsolute = intersect(p3TrialsAbsolute, hitP3TrialsAbsolute);
    % Intersect P3 trials with Miss P3 trials (absolute indices)
    p3MissTrialsAbsolute = intersect(p3TrialsAbsolute, missP3TrialsAbsolute);
    % For Mode 2: Intersect P3 trials with Nonresponsive P3 trials (absolute indices)
    p3NonrespTrialsAbsolute = intersect(p3TrialsAbsolute, nonrespP3TrialsAbsolute);

    % Map absolute indices to P3-relative indices (for indexing Grid40.P3)
    [~, hitTrialIndices_P3] = ismember(p3HitTrialsAbsolute, p3TrialsAbsolute);
    hitTrialIndices_P3 = hitTrialIndices_P3(hitTrialIndices_P3 > 0); % Remove zeros

    [~, missTrialIndices_P3] = ismember(p3MissTrialsAbsolute, p3TrialsAbsolute);
    missTrialIndices_P3 = missTrialIndices_P3(missTrialIndices_P3 > 0); % Remove zeros

    % For Mode 2: Map nonresponsive P3 trials
    [~, nonrespTrialIndices_P3] = ismember(p3NonrespTrialsAbsolute, p3TrialsAbsolute);
    nonrespTrialIndices_P3 = nonrespTrialIndices_P3(nonrespTrialIndices_P3 > 0); % Remove zeros

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
            % Separate Hit and Miss trials using P3-specific indices
            gridData_P3_Hit = gridData_P3(:, :, :, hitTrialIndices_P3);
            gridData_P3_Miss = gridData_P3(:, :, :, missTrialIndices_P3);

            % Ensure 4D dimensions (MATLAB squeezes singleton dimensions when indexing with single value)
            sz = size(gridData_P3_Miss);
            if length(sz) < 4
                gridData_P3_Miss = reshape(gridData_P3_Miss, [sz, 1]);
            end

            % Calculate entropy for Hit trials
            if ~isempty(hitTrialIndices_P3) && length(hitTrialIndices_P3) > 0
                [ent_raw_P3_Hit, ent_rz_P3_Hit, ent_tz_P3_Hit, entZ_raw_P3_Hit, entZ_rz_P3_Hit, entZ_tz_P3_Hit, ent_bin_P3_Hit, ent_bin_an_P3_Hit, ent_bin_all_P3_Hit] = ...
                    calculate_entropy_from_grid(gridData_P3_Hit, gridDimensions, animalMean_P3, animalStd_P3, globalMean_P3, globalStd_P3, binarisation_threshold);

                % Store Hit entropy values
                Entropy_Grid_Hit.(condition).Position3.RecordingEntropy_Raw{end+1} = ent_raw_P3_Hit;
                Entropy_Grid_Hit.(condition).Position3.RecordingEntropy_RZ{end+1} = ent_rz_P3_Hit;
                Entropy_Grid_Hit.(condition).Position3.RecordingEntropy_TZ{end+1} = ent_tz_P3_Hit;
                Entropy_Grid_Hit.(condition).Position3.RecordingEntropy_BIN{end+1} = ent_bin_P3_Hit;
                Entropy_Grid_Hit.(condition).Position3.RecordingEntropy_BIN_AN{end+1} = ent_bin_an_P3_Hit;
                Entropy_Grid_Hit.(condition).Position3.RecordingEntropy_BIN_ALL{end+1} = ent_bin_all_P3_Hit;
                Entropy_Grid_Hit.(condition).Position3.RecordingEntropyZ_Raw{end+1} = entZ_raw_P3_Hit;
                Entropy_Grid_Hit.(condition).Position3.RecordingEntropyZ_RZ{end+1} = entZ_rz_P3_Hit;
                Entropy_Grid_Hit.(condition).Position3.RecordingEntropyZ_TZ{end+1} = entZ_tz_P3_Hit;

                % Calculate dispersion for Hit trials
                [disp_raw_P3_Hit, disp_rz_P3_Hit, disp_tz_P3_Hit, dispZ_raw_P3_Hit, dispZ_rz_P3_Hit, dispZ_tz_P3_Hit, disp_bin_P3_Hit, disp_bin_an_P3_Hit, disp_bin_all_P3_Hit] = ...
                    calculate_dispersion_from_grid(gridData_P3_Hit, gridDimensions, animalMean_P3, animalStd_P3, globalMean_P3, globalStd_P3, binarisation_threshold);

                % Store Hit dispersion values
                Dispersion_Grid_Hit.(condition).Position3.RecordingDispersion_Raw{end+1} = disp_raw_P3_Hit;
                Dispersion_Grid_Hit.(condition).Position3.RecordingDispersion_RZ{end+1} = disp_rz_P3_Hit;
                Dispersion_Grid_Hit.(condition).Position3.RecordingDispersion_TZ{end+1} = disp_tz_P3_Hit;
                Dispersion_Grid_Hit.(condition).Position3.RecordingDispersion_BIN{end+1} = disp_bin_P3_Hit;
                Dispersion_Grid_Hit.(condition).Position3.RecordingDispersion_BIN_AN{end+1} = disp_bin_an_P3_Hit;
                Dispersion_Grid_Hit.(condition).Position3.RecordingDispersion_BIN_ALL{end+1} = disp_bin_all_P3_Hit;
                Dispersion_Grid_Hit.(condition).Position3.RecordingDispersionZ_Raw{end+1} = dispZ_raw_P3_Hit;
                Dispersion_Grid_Hit.(condition).Position3.RecordingDispersionZ_RZ{end+1} = dispZ_rz_P3_Hit;
                Dispersion_Grid_Hit.(condition).Position3.RecordingDispersionZ_TZ{end+1} = dispZ_tz_P3_Hit;
            end

            % Calculate entropy for Miss trials
            if ~isempty(missTrialIndices_P3) && length(missTrialIndices_P3) > 0
                [ent_raw_P3_Miss, ent_rz_P3_Miss, ent_tz_P3_Miss, entZ_raw_P3_Miss, entZ_rz_P3_Miss, entZ_tz_P3_Miss, ent_bin_P3_Miss, ent_bin_an_P3_Miss, ent_bin_all_P3_Miss] = ...
                    calculate_entropy_from_grid(gridData_P3_Miss, gridDimensions, animalMean_P3, animalStd_P3, globalMean_P3, globalStd_P3, binarisation_threshold);

                % Store Miss entropy values
                Entropy_Grid_Miss.(condition).Position3.RecordingEntropy_Raw{end+1} = ent_raw_P3_Miss;
                Entropy_Grid_Miss.(condition).Position3.RecordingEntropy_RZ{end+1} = ent_rz_P3_Miss;
                Entropy_Grid_Miss.(condition).Position3.RecordingEntropy_TZ{end+1} = ent_tz_P3_Miss;
                Entropy_Grid_Miss.(condition).Position3.RecordingEntropy_BIN{end+1} = ent_bin_P3_Miss;
                Entropy_Grid_Miss.(condition).Position3.RecordingEntropy_BIN_AN{end+1} = ent_bin_an_P3_Miss;
                Entropy_Grid_Miss.(condition).Position3.RecordingEntropy_BIN_ALL{end+1} = ent_bin_all_P3_Miss;
                Entropy_Grid_Miss.(condition).Position3.RecordingEntropyZ_Raw{end+1} = entZ_raw_P3_Miss;
                Entropy_Grid_Miss.(condition).Position3.RecordingEntropyZ_RZ{end+1} = entZ_rz_P3_Miss;
                Entropy_Grid_Miss.(condition).Position3.RecordingEntropyZ_TZ{end+1} = entZ_tz_P3_Miss;

                % Calculate dispersion for Miss trials
                [disp_raw_P3_Miss, disp_rz_P3_Miss, disp_tz_P3_Miss, dispZ_raw_P3_Miss, dispZ_rz_P3_Miss, dispZ_tz_P3_Miss, disp_bin_P3_Miss, disp_bin_an_P3_Miss, disp_bin_all_P3_Miss] = ...
                    calculate_dispersion_from_grid(gridData_P3_Miss, gridDimensions, animalMean_P3, animalStd_P3, globalMean_P3, globalStd_P3, binarisation_threshold);

                % Store Miss dispersion values
                Dispersion_Grid_Miss.(condition).Position3.RecordingDispersion_Raw{end+1} = disp_raw_P3_Miss;
                Dispersion_Grid_Miss.(condition).Position3.RecordingDispersion_RZ{end+1} = disp_rz_P3_Miss;
                Dispersion_Grid_Miss.(condition).Position3.RecordingDispersion_TZ{end+1} = disp_tz_P3_Miss;
                Dispersion_Grid_Miss.(condition).Position3.RecordingDispersion_BIN{end+1} = disp_bin_P3_Miss;
                Dispersion_Grid_Miss.(condition).Position3.RecordingDispersion_BIN_AN{end+1} = disp_bin_an_P3_Miss;
                Dispersion_Grid_Miss.(condition).Position3.RecordingDispersion_BIN_ALL{end+1} = disp_bin_all_P3_Miss;
                Dispersion_Grid_Miss.(condition).Position3.RecordingDispersionZ_Raw{end+1} = dispZ_raw_P3_Miss;
                Dispersion_Grid_Miss.(condition).Position3.RecordingDispersionZ_RZ{end+1} = dispZ_rz_P3_Miss;
                Dispersion_Grid_Miss.(condition).Position3.RecordingDispersionZ_TZ{end+1} = dispZ_tz_P3_Miss;
            end

            % FOR MODE 2: Calculate entropy and dispersion for Nonresponsive trials
            if MissNonresponsive == 2 && ~isempty(nonrespTrialIndices_P3) && length(nonrespTrialIndices_P3) > 0
                gridData_P3_Nonresp = gridData_P3(:, :, :, nonrespTrialIndices_P3);

                % Ensure 4D dimensions (MATLAB squeezes singleton dimensions when indexing with single value)
                sz = size(gridData_P3_Nonresp);
                if length(sz) < 4
                    gridData_P3_Nonresp = reshape(gridData_P3_Nonresp, [sz, 1]);
                end

                [ent_raw_P3_Nonresp, ent_rz_P3_Nonresp, ent_tz_P3_Nonresp, entZ_raw_P3_Nonresp, entZ_rz_P3_Nonresp, entZ_tz_P3_Nonresp, ent_bin_P3_Nonresp, ent_bin_an_P3_Nonresp, ent_bin_all_P3_Nonresp] = ...
                    calculate_entropy_from_grid(gridData_P3_Nonresp, gridDimensions, animalMean_P3, animalStd_P3, globalMean_P3, globalStd_P3, binarisation_threshold);

                % Store Nonresponsive entropy values
                Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropy_Raw{end+1} = ent_raw_P3_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropy_RZ{end+1} = ent_rz_P3_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropy_TZ{end+1} = ent_tz_P3_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropy_BIN{end+1} = ent_bin_P3_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropy_BIN_AN{end+1} = ent_bin_an_P3_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropy_BIN_ALL{end+1} = ent_bin_all_P3_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropyZ_Raw{end+1} = entZ_raw_P3_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropyZ_RZ{end+1} = entZ_rz_P3_Nonresp;
                Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropyZ_TZ{end+1} = entZ_tz_P3_Nonresp;

                % Calculate dispersion for Nonresponsive trials
                [disp_raw_P3_Nonresp, disp_rz_P3_Nonresp, disp_tz_P3_Nonresp, dispZ_raw_P3_Nonresp, dispZ_rz_P3_Nonresp, dispZ_tz_P3_Nonresp, disp_bin_P3_Nonresp, disp_bin_an_P3_Nonresp, disp_bin_all_P3_Nonresp] = ...
                    calculate_dispersion_from_grid(gridData_P3_Nonresp, gridDimensions, animalMean_P3, animalStd_P3, globalMean_P3, globalStd_P3, binarisation_threshold);

                % Store Nonresponsive dispersion values
                Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersion_Raw{end+1} = disp_raw_P3_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersion_RZ{end+1} = disp_rz_P3_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersion_TZ{end+1} = disp_tz_P3_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersion_BIN{end+1} = disp_bin_P3_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersion_BIN_AN{end+1} = disp_bin_an_P3_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersion_BIN_ALL{end+1} = disp_bin_all_P3_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersionZ_Raw{end+1} = dispZ_raw_P3_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersionZ_RZ{end+1} = dispZ_rz_P3_Nonresp;
                Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersionZ_TZ{end+1} = dispZ_tz_P3_Nonresp;
            end
        end
    end

    fprintf('  Recording %d: Entropy calculated from Grid (Hit vs Miss)\n', r);
end

% Combine P1 and P3 for "All" position (concatenate cell arrays)
% Hit trials
Entropy_Grid_Hit.(condition).All.RecordingEntropy_Raw = ...
    [Entropy_Grid_Hit.(condition).Position1.RecordingEntropy_Raw, ...
     Entropy_Grid_Hit.(condition).Position3.RecordingEntropy_Raw];
Entropy_Grid_Hit.(condition).All.RecordingEntropy_RZ = ...
    [Entropy_Grid_Hit.(condition).Position1.RecordingEntropy_RZ, ...
     Entropy_Grid_Hit.(condition).Position3.RecordingEntropy_RZ];
Entropy_Grid_Hit.(condition).All.RecordingEntropy_TZ = ...
    [Entropy_Grid_Hit.(condition).Position1.RecordingEntropy_TZ, ...
     Entropy_Grid_Hit.(condition).Position3.RecordingEntropy_TZ];
Entropy_Grid_Hit.(condition).All.RecordingEntropy_BIN = ...
    [Entropy_Grid_Hit.(condition).Position1.RecordingEntropy_BIN, ...
     Entropy_Grid_Hit.(condition).Position3.RecordingEntropy_BIN];
Entropy_Grid_Hit.(condition).All.RecordingEntropy_BIN_AN = ...
    [Entropy_Grid_Hit.(condition).Position1.RecordingEntropy_BIN_AN, ...
     Entropy_Grid_Hit.(condition).Position3.RecordingEntropy_BIN_AN];
Entropy_Grid_Hit.(condition).All.RecordingEntropy_BIN_ALL = ...
    [Entropy_Grid_Hit.(condition).Position1.RecordingEntropy_BIN_ALL, ...
     Entropy_Grid_Hit.(condition).Position3.RecordingEntropy_BIN_ALL];
Entropy_Grid_Hit.(condition).All.RecordingEntropyZ_Raw = ...
    [Entropy_Grid_Hit.(condition).Position1.RecordingEntropyZ_Raw, ...
     Entropy_Grid_Hit.(condition).Position3.RecordingEntropyZ_Raw];
Entropy_Grid_Hit.(condition).All.RecordingEntropyZ_RZ = ...
    [Entropy_Grid_Hit.(condition).Position1.RecordingEntropyZ_RZ, ...
     Entropy_Grid_Hit.(condition).Position3.RecordingEntropyZ_RZ];
Entropy_Grid_Hit.(condition).All.RecordingEntropyZ_TZ = ...
    [Entropy_Grid_Hit.(condition).Position1.RecordingEntropyZ_TZ, ...
     Entropy_Grid_Hit.(condition).Position3.RecordingEntropyZ_TZ];

% Miss trials
Entropy_Grid_Miss.(condition).All.RecordingEntropy_Raw = ...
    [Entropy_Grid_Miss.(condition).Position1.RecordingEntropy_Raw, ...
     Entropy_Grid_Miss.(condition).Position3.RecordingEntropy_Raw];
Entropy_Grid_Miss.(condition).All.RecordingEntropy_RZ = ...
    [Entropy_Grid_Miss.(condition).Position1.RecordingEntropy_RZ, ...
     Entropy_Grid_Miss.(condition).Position3.RecordingEntropy_RZ];
Entropy_Grid_Miss.(condition).All.RecordingEntropy_TZ = ...
    [Entropy_Grid_Miss.(condition).Position1.RecordingEntropy_TZ, ...
     Entropy_Grid_Miss.(condition).Position3.RecordingEntropy_TZ];
Entropy_Grid_Miss.(condition).All.RecordingEntropy_BIN = ...
    [Entropy_Grid_Miss.(condition).Position1.RecordingEntropy_BIN, ...
     Entropy_Grid_Miss.(condition).Position3.RecordingEntropy_BIN];
Entropy_Grid_Miss.(condition).All.RecordingEntropy_BIN_AN = ...
    [Entropy_Grid_Miss.(condition).Position1.RecordingEntropy_BIN_AN, ...
     Entropy_Grid_Miss.(condition).Position3.RecordingEntropy_BIN_AN];
Entropy_Grid_Miss.(condition).All.RecordingEntropy_BIN_ALL = ...
    [Entropy_Grid_Miss.(condition).Position1.RecordingEntropy_BIN_ALL, ...
     Entropy_Grid_Miss.(condition).Position3.RecordingEntropy_BIN_ALL];
Entropy_Grid_Miss.(condition).All.RecordingEntropyZ_Raw = ...
    [Entropy_Grid_Miss.(condition).Position1.RecordingEntropyZ_Raw, ...
     Entropy_Grid_Miss.(condition).Position3.RecordingEntropyZ_Raw];
Entropy_Grid_Miss.(condition).All.RecordingEntropyZ_RZ = ...
    [Entropy_Grid_Miss.(condition).Position1.RecordingEntropyZ_RZ, ...
     Entropy_Grid_Miss.(condition).Position3.RecordingEntropyZ_RZ];
Entropy_Grid_Miss.(condition).All.RecordingEntropyZ_TZ = ...
    [Entropy_Grid_Miss.(condition).Position1.RecordingEntropyZ_TZ, ...
     Entropy_Grid_Miss.(condition).Position3.RecordingEntropyZ_TZ];

% Combine P1 and P3 dispersion for "All" position (concatenate cell arrays)
% Hit trials
Dispersion_Grid_Hit.(condition).All.RecordingDispersion_Raw = ...
    [Dispersion_Grid_Hit.(condition).Position1.RecordingDispersion_Raw, ...
     Dispersion_Grid_Hit.(condition).Position3.RecordingDispersion_Raw];
Dispersion_Grid_Hit.(condition).All.RecordingDispersion_RZ = ...
    [Dispersion_Grid_Hit.(condition).Position1.RecordingDispersion_RZ, ...
     Dispersion_Grid_Hit.(condition).Position3.RecordingDispersion_RZ];
Dispersion_Grid_Hit.(condition).All.RecordingDispersion_TZ = ...
    [Dispersion_Grid_Hit.(condition).Position1.RecordingDispersion_TZ, ...
     Dispersion_Grid_Hit.(condition).Position3.RecordingDispersion_TZ];
Dispersion_Grid_Hit.(condition).All.RecordingDispersion_BIN = ...
    [Dispersion_Grid_Hit.(condition).Position1.RecordingDispersion_BIN, ...
     Dispersion_Grid_Hit.(condition).Position3.RecordingDispersion_BIN];
Dispersion_Grid_Hit.(condition).All.RecordingDispersion_BIN_AN = ...
    [Dispersion_Grid_Hit.(condition).Position1.RecordingDispersion_BIN_AN, ...
     Dispersion_Grid_Hit.(condition).Position3.RecordingDispersion_BIN_AN];
Dispersion_Grid_Hit.(condition).All.RecordingDispersion_BIN_ALL = ...
    [Dispersion_Grid_Hit.(condition).Position1.RecordingDispersion_BIN_ALL, ...
     Dispersion_Grid_Hit.(condition).Position3.RecordingDispersion_BIN_ALL];
Dispersion_Grid_Hit.(condition).All.RecordingDispersionZ_Raw = ...
    [Dispersion_Grid_Hit.(condition).Position1.RecordingDispersionZ_Raw, ...
     Dispersion_Grid_Hit.(condition).Position3.RecordingDispersionZ_Raw];
Dispersion_Grid_Hit.(condition).All.RecordingDispersionZ_RZ = ...
    [Dispersion_Grid_Hit.(condition).Position1.RecordingDispersionZ_RZ, ...
     Dispersion_Grid_Hit.(condition).Position3.RecordingDispersionZ_RZ];
Dispersion_Grid_Hit.(condition).All.RecordingDispersionZ_TZ = ...
    [Dispersion_Grid_Hit.(condition).Position1.RecordingDispersionZ_TZ, ...
     Dispersion_Grid_Hit.(condition).Position3.RecordingDispersionZ_TZ];

% Miss trials
Dispersion_Grid_Miss.(condition).All.RecordingDispersion_Raw = ...
    [Dispersion_Grid_Miss.(condition).Position1.RecordingDispersion_Raw, ...
     Dispersion_Grid_Miss.(condition).Position3.RecordingDispersion_Raw];
Dispersion_Grid_Miss.(condition).All.RecordingDispersion_RZ = ...
    [Dispersion_Grid_Miss.(condition).Position1.RecordingDispersion_RZ, ...
     Dispersion_Grid_Miss.(condition).Position3.RecordingDispersion_RZ];
Dispersion_Grid_Miss.(condition).All.RecordingDispersion_TZ = ...
    [Dispersion_Grid_Miss.(condition).Position1.RecordingDispersion_TZ, ...
     Dispersion_Grid_Miss.(condition).Position3.RecordingDispersion_TZ];
Dispersion_Grid_Miss.(condition).All.RecordingDispersion_BIN = ...
    [Dispersion_Grid_Miss.(condition).Position1.RecordingDispersion_BIN, ...
     Dispersion_Grid_Miss.(condition).Position3.RecordingDispersion_BIN];
Dispersion_Grid_Miss.(condition).All.RecordingDispersion_BIN_AN = ...
    [Dispersion_Grid_Miss.(condition).Position1.RecordingDispersion_BIN_AN, ...
     Dispersion_Grid_Miss.(condition).Position3.RecordingDispersion_BIN_AN];
Dispersion_Grid_Miss.(condition).All.RecordingDispersion_BIN_ALL = ...
    [Dispersion_Grid_Miss.(condition).Position1.RecordingDispersion_BIN_ALL, ...
     Dispersion_Grid_Miss.(condition).Position3.RecordingDispersion_BIN_ALL];
Dispersion_Grid_Miss.(condition).All.RecordingDispersionZ_Raw = ...
    [Dispersion_Grid_Miss.(condition).Position1.RecordingDispersionZ_Raw, ...
     Dispersion_Grid_Miss.(condition).Position3.RecordingDispersionZ_Raw];
Dispersion_Grid_Miss.(condition).All.RecordingDispersionZ_RZ = ...
    [Dispersion_Grid_Miss.(condition).Position1.RecordingDispersionZ_RZ, ...
     Dispersion_Grid_Miss.(condition).Position3.RecordingDispersionZ_RZ];
Dispersion_Grid_Miss.(condition).All.RecordingDispersionZ_TZ = ...
    [Dispersion_Grid_Miss.(condition).Position1.RecordingDispersionZ_TZ, ...
     Dispersion_Grid_Miss.(condition).Position3.RecordingDispersionZ_TZ];

% Combine P1 and P3 Moran's I for "All" position (concatenate cell arrays)
% NOTE: Currently only Position1 BIN_AN is calculated
% Hit trials
MoransI_Grid_Hit.(condition).All.RecordingMoransI_Raw = ...
    [MoransI_Grid_Hit.(condition).Position1.RecordingMoransI_Raw, ...
     MoransI_Grid_Hit.(condition).Position3.RecordingMoransI_Raw];
MoransI_Grid_Hit.(condition).All.RecordingMoransI_RZ = ...
    [MoransI_Grid_Hit.(condition).Position1.RecordingMoransI_RZ, ...
     MoransI_Grid_Hit.(condition).Position3.RecordingMoransI_RZ];
MoransI_Grid_Hit.(condition).All.RecordingMoransI_TZ = ...
    [MoransI_Grid_Hit.(condition).Position1.RecordingMoransI_TZ, ...
     MoransI_Grid_Hit.(condition).Position3.RecordingMoransI_TZ];
MoransI_Grid_Hit.(condition).All.RecordingMoransI_BIN = ...
    [MoransI_Grid_Hit.(condition).Position1.RecordingMoransI_BIN, ...
     MoransI_Grid_Hit.(condition).Position3.RecordingMoransI_BIN];
MoransI_Grid_Hit.(condition).All.RecordingMoransI_BIN_AN = ...
    [MoransI_Grid_Hit.(condition).Position1.RecordingMoransI_BIN_AN, ...
     MoransI_Grid_Hit.(condition).Position3.RecordingMoransI_BIN_AN];
MoransI_Grid_Hit.(condition).All.RecordingMoransI_BIN_ALL = ...
    [MoransI_Grid_Hit.(condition).Position1.RecordingMoransI_BIN_ALL, ...
     MoransI_Grid_Hit.(condition).Position3.RecordingMoransI_BIN_ALL];
MoransI_Grid_Hit.(condition).All.RecordingMoransIZ_Raw = ...
    [MoransI_Grid_Hit.(condition).Position1.RecordingMoransIZ_Raw, ...
     MoransI_Grid_Hit.(condition).Position3.RecordingMoransIZ_Raw];
MoransI_Grid_Hit.(condition).All.RecordingMoransIZ_RZ = ...
    [MoransI_Grid_Hit.(condition).Position1.RecordingMoransIZ_RZ, ...
     MoransI_Grid_Hit.(condition).Position3.RecordingMoransIZ_RZ];
MoransI_Grid_Hit.(condition).All.RecordingMoransIZ_TZ = ...
    [MoransI_Grid_Hit.(condition).Position1.RecordingMoransIZ_TZ, ...
     MoransI_Grid_Hit.(condition).Position3.RecordingMoransIZ_TZ];

% Miss trials
MoransI_Grid_Miss.(condition).All.RecordingMoransI_Raw = ...
    [MoransI_Grid_Miss.(condition).Position1.RecordingMoransI_Raw, ...
     MoransI_Grid_Miss.(condition).Position3.RecordingMoransI_Raw];
MoransI_Grid_Miss.(condition).All.RecordingMoransI_RZ = ...
    [MoransI_Grid_Miss.(condition).Position1.RecordingMoransI_RZ, ...
     MoransI_Grid_Miss.(condition).Position3.RecordingMoransI_RZ];
MoransI_Grid_Miss.(condition).All.RecordingMoransI_TZ = ...
    [MoransI_Grid_Miss.(condition).Position1.RecordingMoransI_TZ, ...
     MoransI_Grid_Miss.(condition).Position3.RecordingMoransI_TZ];
MoransI_Grid_Miss.(condition).All.RecordingMoransI_BIN = ...
    [MoransI_Grid_Miss.(condition).Position1.RecordingMoransI_BIN, ...
     MoransI_Grid_Miss.(condition).Position3.RecordingMoransI_BIN];
MoransI_Grid_Miss.(condition).All.RecordingMoransI_BIN_AN = ...
    [MoransI_Grid_Miss.(condition).Position1.RecordingMoransI_BIN_AN, ...
     MoransI_Grid_Miss.(condition).Position3.RecordingMoransI_BIN_AN];
MoransI_Grid_Miss.(condition).All.RecordingMoransI_BIN_ALL = ...
    [MoransI_Grid_Miss.(condition).Position1.RecordingMoransI_BIN_ALL, ...
     MoransI_Grid_Miss.(condition).Position3.RecordingMoransI_BIN_ALL];
MoransI_Grid_Miss.(condition).All.RecordingMoransIZ_Raw = ...
    [MoransI_Grid_Miss.(condition).Position1.RecordingMoransIZ_Raw, ...
     MoransI_Grid_Miss.(condition).Position3.RecordingMoransIZ_Raw];
MoransI_Grid_Miss.(condition).All.RecordingMoransIZ_RZ = ...
    [MoransI_Grid_Miss.(condition).Position1.RecordingMoransIZ_RZ, ...
     MoransI_Grid_Miss.(condition).Position3.RecordingMoransIZ_RZ];
MoransI_Grid_Miss.(condition).All.RecordingMoransIZ_TZ = ...
    [MoransI_Grid_Miss.(condition).Position1.RecordingMoransIZ_TZ, ...
     MoransI_Grid_Miss.(condition).Position3.RecordingMoransIZ_TZ];

% FOR MODE 2: Combine P1 and P3 for Nonresponsive trials
if MissNonresponsive == 2
    % Nonresponsive Entropy
    Entropy_Grid_Nonresponsive.(condition).All.RecordingEntropy_Raw = ...
        [Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropy_Raw, ...
         Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropy_Raw];
    Entropy_Grid_Nonresponsive.(condition).All.RecordingEntropy_RZ = ...
        [Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropy_RZ, ...
         Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropy_RZ];
    Entropy_Grid_Nonresponsive.(condition).All.RecordingEntropy_TZ = ...
        [Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropy_TZ, ...
         Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropy_TZ];
    Entropy_Grid_Nonresponsive.(condition).All.RecordingEntropy_BIN = ...
        [Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropy_BIN, ...
         Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropy_BIN];
    Entropy_Grid_Nonresponsive.(condition).All.RecordingEntropy_BIN_AN = ...
        [Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropy_BIN_AN, ...
         Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropy_BIN_AN];
    Entropy_Grid_Nonresponsive.(condition).All.RecordingEntropy_BIN_ALL = ...
        [Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropy_BIN_ALL, ...
         Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropy_BIN_ALL];
    Entropy_Grid_Nonresponsive.(condition).All.RecordingEntropyZ_Raw = ...
        [Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropyZ_Raw, ...
         Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropyZ_Raw];
    Entropy_Grid_Nonresponsive.(condition).All.RecordingEntropyZ_RZ = ...
        [Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropyZ_RZ, ...
         Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropyZ_RZ];
    Entropy_Grid_Nonresponsive.(condition).All.RecordingEntropyZ_TZ = ...
        [Entropy_Grid_Nonresponsive.(condition).Position1.RecordingEntropyZ_TZ, ...
         Entropy_Grid_Nonresponsive.(condition).Position3.RecordingEntropyZ_TZ];

    % Nonresponsive Dispersion
    Dispersion_Grid_Nonresponsive.(condition).All.RecordingDispersion_Raw = ...
        [Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersion_Raw, ...
         Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersion_Raw];
    Dispersion_Grid_Nonresponsive.(condition).All.RecordingDispersion_RZ = ...
        [Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersion_RZ, ...
         Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersion_RZ];
    Dispersion_Grid_Nonresponsive.(condition).All.RecordingDispersion_TZ = ...
        [Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersion_TZ, ...
         Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersion_TZ];
    Dispersion_Grid_Nonresponsive.(condition).All.RecordingDispersion_BIN = ...
        [Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersion_BIN, ...
         Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersion_BIN];
    Dispersion_Grid_Nonresponsive.(condition).All.RecordingDispersion_BIN_AN = ...
        [Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersion_BIN_AN, ...
         Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersion_BIN_AN];
    Dispersion_Grid_Nonresponsive.(condition).All.RecordingDispersion_BIN_ALL = ...
        [Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersion_BIN_ALL, ...
         Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersion_BIN_ALL];
    Dispersion_Grid_Nonresponsive.(condition).All.RecordingDispersionZ_Raw = ...
        [Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersionZ_Raw, ...
         Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersionZ_Raw];
    Dispersion_Grid_Nonresponsive.(condition).All.RecordingDispersionZ_RZ = ...
        [Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersionZ_RZ, ...
         Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersionZ_RZ];
    Dispersion_Grid_Nonresponsive.(condition).All.RecordingDispersionZ_TZ = ...
        [Dispersion_Grid_Nonresponsive.(condition).Position1.RecordingDispersionZ_TZ, ...
         Dispersion_Grid_Nonresponsive.(condition).Position3.RecordingDispersionZ_TZ];

    % Nonresponsive Moran's I
    MoransI_Grid_Nonresponsive.(condition).All.RecordingMoransI_Raw = ...
        [MoransI_Grid_Nonresponsive.(condition).Position1.RecordingMoransI_Raw, ...
         MoransI_Grid_Nonresponsive.(condition).Position3.RecordingMoransI_Raw];
    MoransI_Grid_Nonresponsive.(condition).All.RecordingMoransI_RZ = ...
        [MoransI_Grid_Nonresponsive.(condition).Position1.RecordingMoransI_RZ, ...
         MoransI_Grid_Nonresponsive.(condition).Position3.RecordingMoransI_RZ];
    MoransI_Grid_Nonresponsive.(condition).All.RecordingMoransI_TZ = ...
        [MoransI_Grid_Nonresponsive.(condition).Position1.RecordingMoransI_TZ, ...
         MoransI_Grid_Nonresponsive.(condition).Position3.RecordingMoransI_TZ];
    MoransI_Grid_Nonresponsive.(condition).All.RecordingMoransI_BIN = ...
        [MoransI_Grid_Nonresponsive.(condition).Position1.RecordingMoransI_BIN, ...
         MoransI_Grid_Nonresponsive.(condition).Position3.RecordingMoransI_BIN];
    MoransI_Grid_Nonresponsive.(condition).All.RecordingMoransI_BIN_AN = ...
        [MoransI_Grid_Nonresponsive.(condition).Position1.RecordingMoransI_BIN_AN, ...
         MoransI_Grid_Nonresponsive.(condition).Position3.RecordingMoransI_BIN_AN];
    MoransI_Grid_Nonresponsive.(condition).All.RecordingMoransI_BIN_ALL = ...
        [MoransI_Grid_Nonresponsive.(condition).Position1.RecordingMoransI_BIN_ALL, ...
         MoransI_Grid_Nonresponsive.(condition).Position3.RecordingMoransI_BIN_ALL];
    MoransI_Grid_Nonresponsive.(condition).All.RecordingMoransIZ_Raw = ...
        [MoransI_Grid_Nonresponsive.(condition).Position1.RecordingMoransIZ_Raw, ...
         MoransI_Grid_Nonresponsive.(condition).Position3.RecordingMoransIZ_Raw];
    MoransI_Grid_Nonresponsive.(condition).All.RecordingMoransIZ_RZ = ...
        [MoransI_Grid_Nonresponsive.(condition).Position1.RecordingMoransIZ_RZ, ...
         MoransI_Grid_Nonresponsive.(condition).Position3.RecordingMoransIZ_RZ];
    MoransI_Grid_Nonresponsive.(condition).All.RecordingMoransIZ_TZ = ...
        [MoransI_Grid_Nonresponsive.(condition).Position1.RecordingMoransIZ_TZ, ...
         MoransI_Grid_Nonresponsive.(condition).Position3.RecordingMoransIZ_TZ];
end

fprintf('  Condition %s complete (Hit vs Miss separation)\n', condition);


%% =========================================================================
%% Section 2: Entropy Comparison Analysis (Hit vs Miss)
%% =========================================================================
%% Section 2.1: Grid vs Activity Entropy Comparison (Hit vs Miss)
%% =========================================================================
% Control which entropy normalization to compare
% Type options: 'Raw', 'RZ', 'TZ', 'BIN', 'BIN_AN', 'BIN_ALL'
% UseZScored options: false (raw activity), true (z-scored activity)
Type = 'BIN_AN';  % Options: 'Raw', 'RZ', 'TZ', 'BIN', 'BIN_AN', 'BIN_ALL'
UseZScored = false;    % Set true to use entropy from z-scored activity

% Define positions to analyze
Position = "All";
 Position = "P1";
% Position = "P3";

% Define frame window to analyze
FrameWindow = 70:80;  % Options: 1:80 (baseline+pre-stim), 60:80 (pre-stim only), etc.
%%
% Call comparison plotting function (Hit vs Miss vs Nonresponsive)
if MissNonresponsive == 2
    % Three-way comparison
    plotMedianEntropyBars_Grid_Performance(Entropy_Grid_Hit, Entropy_Grid_Miss, ...
        Rec, params, condition, Skip, 'Position', Position, 'Type', Type, 'UseZScored', UseZScored, ...
        'FrameWindow', FrameWindow, 'Entropy_Grid_Nonresponsive', Entropy_Grid_Nonresponsive)
else
    % Two-way comparison
    plotMedianEntropyBars_Grid_Performance(Entropy_Grid_Hit, Entropy_Grid_Miss, ...
        Rec, params, condition, Skip, 'Position', Position, 'Type', Type, 'UseZScored', UseZScored, ...
        'FrameWindow', FrameWindow)
end

%% =========================================================================
%% Section 2.2: Correlation & Agreement Analysis (Hit vs Miss vs Nonresponsive)
%% =========================================================================
if MissNonresponsive == 2
    % Three-way pairwise correlations (Hit vs Miss, Hit vs Nonresponsive, Miss vs Nonresponsive)
    correlation_stats = plotEntropyCorrelation_HitVsMiss(Entropy_Grid_Hit, Entropy_Grid_Miss, ...
        params, condition, Skip, 'Position', Position, 'Type', Type, 'UseZScored', UseZScored, ...
        'FrameWindow', FrameWindow, 'Entropy_Grid_Nonresponsive', Entropy_Grid_Nonresponsive);
else
    % Two-way correlation (Hit vs Miss)
    correlation_stats = plotEntropyCorrelation_HitVsMiss(Entropy_Grid_Hit, Entropy_Grid_Miss, ...
        params, condition, Skip, 'Position', Position, 'Type', Type, 'UseZScored', UseZScored, ...
        'FrameWindow', FrameWindow);
end

%% =========================================================================
%% Section 3: Dispersion Analysis (Hit vs Miss vs Nonresponsive)
%% =========================================================================
%% 3.1: Dispersion Distribution Comparison (Hit vs Miss vs Nonresponsive)
if MissNonresponsive == 2
    % Three-way comparison
    plotMedianDispersionBars_Grid_Performance(Dispersion_Grid_Hit, Dispersion_Grid_Miss, Rec, params, condition, Skip, 'Position', Position, 'Type', Type,'FrameWindow', FrameWindow, 'Dispersion_Grid_Nonresponsive', Dispersion_Grid_Nonresponsive);
else
    % Two-way comparison
    plotMedianDispersionBars_Grid_Performance(Dispersion_Grid_Hit, Dispersion_Grid_Miss, Rec, params, condition, Skip, 'Position', Position, 'Type', Type, 'FrameWindow', FrameWindow);
end

%% 3.2: Dispersion vs Entropy Correlation (Hit vs Miss vs Nonresponsive)
% Combine data structures to enable multi-condition correlation plotting
% Reorganize: Hit/Miss/Nonresponsive become the "condition" level fields
Combined_Entropy = struct();
Combined_Dispersion = struct();
Combined_Entropy.Hit = Entropy_Grid_Hit.(condition);
Combined_Dispersion.Hit = Dispersion_Grid_Hit.(condition);
Combined_Entropy.Miss = Entropy_Grid_Miss.(condition);
Combined_Dispersion.Miss = Dispersion_Grid_Miss.(condition);

% Define performance types based on mode
if MissNonresponsive == 2
    % Three-way comparison
    Combined_Entropy.Nonresponsive = Entropy_Grid_Nonresponsive.(condition);
    Combined_Dispersion.Nonresponsive = Dispersion_Grid_Nonresponsive.(condition);
    perfTypes = {'Hit', 'Miss', 'Nonresponsive'};
else
    % Two-way comparison
    perfTypes = {'Hit', 'Miss'};
end

% Call unified correlation function (treats Hit/Miss/Nonresponsive as "conditions")
dispersion_entropy_corr = plotDispersionCorrelation_GridVsEntropy(Combined_Entropy, Combined_Dispersion, ...
    params, perfTypes, Skip, 'Position', Position, 'Type', Type, 'UseZScored', UseZScored);

%% 3.3: Temporal Dynamics of Dispersion (Hit vs Miss vs Nonresponsive)
if MissNonresponsive == 2
    plotDispersionTimeCourse_Grid_Performance(Dispersion_Grid_Hit, Dispersion_Grid_Miss,params, condition, Skip, 'Position', Position, 'Type', Type,'Dispersion_Grid_Nonresponsive', Dispersion_Grid_Nonresponsive);
else
    plotDispersionTimeCourse_Grid_Performance(Dispersion_Grid_Hit, Dispersion_Grid_Miss, ...
        params, condition, Skip, 'Position', Position, 'Type', Type);
end

%% =========================================================================
%% Section 4: Moran's I Analysis (Hit vs Miss)
%% =========================================================================
if include_MoransI

    %% 4.1: Moran's I Distribution (Hit vs Miss vs Nonresponsive)
    if MissNonresponsive == 2
        % Three-way comparison
        plotMedianMoransIBars_Grid_Performance(MoransI_Grid_Hit, MoransI_Grid_Miss, ...
            Rec, params, condition, Skip, 'Position', Position, 'Type', Type, ...
            'FrameWindow', FrameWindow, 'MoransI_Grid_Nonresponsive', MoransI_Grid_Nonresponsive);
    else
        % Two-way comparison
        plotMedianMoransIBars_Grid_Performance(MoransI_Grid_Hit, MoransI_Grid_Miss, ...
            Rec, params, condition, Skip, 'Position', Position, 'Type', Type, 'FrameWindow', FrameWindow);
    end

    %% 4.2: Moran's I vs Entropy Correlation (Hit vs Miss vs Nonresponsive)
    % Combine data structures to enable multi-condition correlation plotting
    Combined_Entropy_MoransI = struct();
    Combined_MoransI = struct();
    Combined_Entropy_MoransI.Hit = Entropy_Grid_Hit.(condition);
    Combined_MoransI.Hit = MoransI_Grid_Hit.(condition);
    Combined_Entropy_MoransI.Miss = Entropy_Grid_Miss.(condition);
    Combined_MoransI.Miss = MoransI_Grid_Miss.(condition);

    % Define performance types based on mode
    if MissNonresponsive == 2
        % Three-way comparison
        Combined_Entropy_MoransI.Nonresponsive = Entropy_Grid_Nonresponsive.(condition);
        Combined_MoransI.Nonresponsive = MoransI_Grid_Nonresponsive.(condition);
        perfTypes_MoransI = {'Hit', 'Miss', 'Nonresponsive'};
    else
        % Two-way comparison
        perfTypes_MoransI = {'Hit', 'Miss'};
    end

    % Call unified correlation function
    moransI_entropy_corr = plotMoransICorrelation_GridVsEntropy(Combined_Entropy_MoransI, Combined_MoransI, ...
        params, perfTypes_MoransI, Skip, 'Position', Position, 'Type', Type, 'UseZScored', UseZScored);

    %% 4.3: Temporal Dynamics of Moran's I (Hit vs Miss vs Nonresponsive)
    if MissNonresponsive == 2
        % Three-way comparison
        plotMoransITimeCourse_Grid_Performance(MoransI_Grid_Hit, MoransI_Grid_Miss, ...
            params, condition, Skip, 'Position', Position, 'Type', Type, ...
            'MoransI_Grid_Nonresponsive', MoransI_Grid_Nonresponsive);
    else
        % Two-way comparison
        plotMoransITimeCourse_Grid_Performance(MoransI_Grid_Hit, MoransI_Grid_Miss, ...
            params, condition, Skip, 'Position', Position, 'Type', Type);
    end

    %% 4.4: Moran's I vs Dispersion Correlation (Hit vs Miss vs Nonresponsive)
    % Combine data structures to enable multi-condition correlation plotting
    Combined_Dispersion_MoransI = struct();
    Combined_MoransI_Disp = struct();
    Combined_Dispersion_MoransI.Hit = Dispersion_Grid_Hit.(condition);
    Combined_MoransI_Disp.Hit = MoransI_Grid_Hit.(condition);
    Combined_Dispersion_MoransI.Miss = Dispersion_Grid_Miss.(condition);
    Combined_MoransI_Disp.Miss = MoransI_Grid_Miss.(condition);

    % Define performance types based on mode
    if MissNonresponsive == 2
        % Three-way comparison
        Combined_Dispersion_MoransI.Nonresponsive = Dispersion_Grid_Nonresponsive.(condition);
        Combined_MoransI_Disp.Nonresponsive = MoransI_Grid_Nonresponsive.(condition);
        perfTypes_Disp = {'Hit', 'Miss', 'Nonresponsive'};
    else
        % Two-way comparison
        perfTypes_Disp = {'Hit', 'Miss'};
    end

    % Call unified correlation function
    moransI_dispersion_corr = plotMoransICorrelation_GridVsDispersion(Combined_Dispersion_MoransI, Combined_MoransI_Disp, ...
        params, perfTypes_Disp, Skip, 'Position', Position, 'Type', Type);
end

%% =========================================================================
%% Section 5: Save Figures
%% =========================================================================

% Create folder path based on Type, UseZScored, Position, and Performance comparison
zScoreLabel = '';
if UseZScored
    zScoreLabel = '_ZScored';
end

% Define Miss type label
if MissNonresponsive == 0
    missLabel = 'Miss';
elseif MissNonresponsive == 1
    missLabel = 'Nonresponsive';
else
    missLabel = 'Miss_Nonresponsive';
end

% Save Entropy Comparison figures
savePathEntropy = sprintf('Fig. 5 Entropy Comparison\\\\Performance_HitVs%s\\\\%s%s_%s\\\\Entropy\\\\', ...
    missLabel, Type, zScoreLabel, Position);
fprintf('\nSaving Entropy Comparison figures to: %s\n', savePathEntropy);
% saveMyFig("Fig5_EntropyComparison_Performance", savePathEntropy, 'All')

% Save Dispersion Analysis figures
savePathDispersion = sprintf('Fig. 5 Entropy Comparison\\\\Performance_HitVs%s\\\\%s%s_%s\\\\Dispersion\\\\', ...
    missLabel, Type, zScoreLabel, Position);
fprintf('Saving Dispersion Analysis figures to: %s\n', savePathDispersion);
% saveMyFig("Fig5_DispersionAnalysis_Performance", savePathDispersion, 'All')

% Save Moran's I Analysis figures (if calculated)
if include_MoransI
    savePathMoransI = sprintf('Fig. 5 Entropy Comparison\\\\Performance_HitVs%s\\\\%s%s_%s\\\\MoransI\\\\', ...
        missLabel, Type, zScoreLabel, Position);
    fprintf('Saving Moran''s I Analysis figures to: %s\n', savePathMoransI);
    % saveMyFig("Fig5_MoransIAnalysis_Performance", savePathMoransI, 'All')
end

if include_MoransI
    fprintf('\n=== Figure 5 Performance Comparison Complete (Hit vs %s) ===\n', missLabel);
else
    fprintf('\n=== Figure 5 Entropy & Dispersion Performance Comparison Complete (Hit vs %s) ===\n', missLabel);
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
%   ent_bin_an  - Entropy from binarised data (per-animal z-score > 1.5σ threshold)
%   ent_bin_all - Entropy from binarised data (GLOBAL z-score > 1.5σ threshold)

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
                [gridY, gridX, nTimepoints] = size(gridData);
                nTrials = 1;
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

    %% Calculate BINARISED entropy (inspired by BinarizeDataOnTheGrid.py)
    % Initialize binarised entropy array
    ent_bin = zeros(nTrials, nTimepoints);

    for trial = 1:nTrials
        for t = 1:nTimepoints
            % Extract grid cell activities at this trial/timepoint
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
% CALCULATE_DISPERSION_FROM_GRID - Calculate spatial dispersion from Grid40 data
%
% Dispersion measures how spatially concentrated vs. dispersed active grid cells are.
% Low dispersion = tight blob (organized activity)
% High dispersion = scattered activity (disorganized)
%
% INPUTS:
%   gridData       - Grid data [gridY × gridX × nTimepoints × nTrials]
%   gridDimensions - [gridY, gridX] dimensions
%   animalMean     - (Optional) Per-animal mean for BIN_AN calculation
%   animalStd      - (Optional) Per-animal std for BIN_AN calculation
%   globalMean     - (Optional) GLOBAL mean for BIN_ALL calculation
%   globalStd      - (Optional) GLOBAL std for BIN_ALL calculation
%
% OUTPUTS:
%   disp_raw     - Raw dispersion [nTrials × nTimepoints] (weighted by activity)
%   disp_rz      - Recording-zscore dispersion
%   disp_tz      - Trial-zscore dispersion
%   dispZ_*      - Same metrics on z-scored grid data
%   disp_bin     - Dispersion from binarised data (per-recording threshold)
%   disp_bin_an  - Dispersion from binarised data (per-animal threshold)
%   disp_bin_all - Dispersion from binarised data (GLOBAL threshold)

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
        [gridY, gridX, nTimepoints] = size(gridData);
        nTrials = 1;
    end

    % Generate grid coordinates
    [gridX_coords, gridY_coords] = meshgrid(1:gridX, 1:gridY);
    grid_coords = [gridY_coords(:), gridX_coords(:)]; % [nGridCells × 2]

    % Reshape grid
    nGridCells = gridY * gridX;
    gridData_reshaped = reshape(gridData, [nGridCells, nTimepoints, nTrials]);

    % Initialize output arrays
    disp_raw = zeros(nTrials, nTimepoints);
    dispZ_raw = zeros(nTrials, nTimepoints);

    %% Calculate WEIGHTED dispersion for continuous data (Raw)
    for trial = 1:nTrials
        for t = 1:nTimepoints
            activity = gridData_reshaped(:, t, trial);
            disp_raw(trial, t) = calculate_weighted_dispersion(activity, grid_coords);

            % Z-scored data
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

    %% Calculate BINARISED dispersion with PER-ANIMAL thresholds (BIN_AN)
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

    %% Calculate BINARISED dispersion with GLOBAL thresholds (BIN_ALL)
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

    %% Calculate trial-zscore (TZ)
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

    %% Calculate recording-zscore (RZ)
    disp_rz = (disp_raw - mean(disp_raw(:))) / std(disp_raw(:));
    dispZ_rz = (dispZ_raw - mean(dispZ_raw(:))) / std(dispZ_raw(:));
end

%% =========================================================================
%% Sub-helper: Calculate weighted dispersion for continuous data
%% =========================================================================
function disp = calculate_weighted_dispersion(activity, grid_coords)
% Calculate dispersion weighted by activity levels

    activity_threshold = 0.01;
    active_mask = activity > activity_threshold;

    if sum(active_mask) == 0
        disp = NaN;
        return;
    end

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
%% Sub-helper: Calculate binary dispersion
%% =========================================================================
function disp = calculate_binary_dispersion(activity_bin, grid_coords)
% Calculate dispersion for binary data

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

%% =========================================================================
%% Helper Function: Calculate Moran's I from Grid Data (BIN_AN only)
%% =========================================================================
function morans_bin_an = calculate_moransI_BIN_AN(gridData, gridDimensions, animalMean, animalStd, binarisation_threshold)
% CALCULATE_MORANSI_BIN_AN - Calculate Moran's I spatial autocorrelation
%
% INPUTS:
%   gridData       - Grid data [gridY × gridX × nTimepoints × nTrials]
%   gridDimensions - [gridY, gridX] dimensions
%   animalMean     - Per-animal mean for BIN_AN calculation
%   animalStd      - Per-animal std for BIN_AN calculation
%
% OUTPUTS:
%   morans_bin_an  - Moran's I [nTrials × nTimepoints]

    % Get dimensions
    if ndims(gridData) == 4
        [gridY, gridX, nTimepoints, nTrials] = size(gridData);
    else
        error('Expected Grid data to have 4 dimensions');
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
