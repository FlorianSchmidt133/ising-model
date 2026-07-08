%% =========================================================================
%% Figure 5: Data Aggregation for Ising Model Training
%% =========================================================================
% This script prepares binarized Grid40 neural activity data with Moran's I
% spatial autocorrelation for Ising model training 
%
% Key features:
% - Binarises Grid40 data using raw threshold (default=2.0) or mean+k×sigma
% - Aggregates data by condition (pools all recordings per condition)
% - Calculates Moran's I spatial autocorrelation per trial × timepoint
% - Includes comprehensive timing metadata for subsetting
% - Outputs data in 3 formats: MATLAB (.mat), NumPy (.npz), and HDF5 (.h5)
%
% Variables/datasets in each file:
% - BinarisedData.Condition: [gridY × gridX × 185 × nTrialsTotal]
% - MoransI.Condition: [nTrialsTotal × 185]
% - TimingInfo: Stimulus timing and frame metadata
% - GridMetadata: Grid dimensions and structure info
% - Conditions: Cell array of condition names
% - binarisation_method, threshold_value, threshold_description
%
% Conditions: Naive, Beginner, Expert, Expert_Hit, Expert_Miss, NoSpout
% Timeframe: Full trial (frames 1:185)
% Pre-stim period: Frames 1:80
% Stimulus period: Frames 81:100
% Post-stim period: Frames 101:185

%% =========================================================================
%% SECTION 1: Setup and Parameters
%% =========================================================================

fprintf('\n=== Figure 5: Data Aggregation for Ising Models ===\n');
fprintf('Creating files (.mat, .npz, .h5) with binarised Grid40 data\n\n');

% -------------------------------------------------------------------------
% Binarization Method Selection
% -------------------------------------------------------------------------
BINARIZATION_METHOD = 'raw';  % Options: 'raw' or 'zscore'

% Method 1: Raw absolute threshold
raw_activity_threshold = 2.0;  % Default dF/F threshold

% Method 2: Mean + k×sigma threshold (per recording, then aggregate)
k_sigma_multiplier = 2.0;  % Standard deviations above mean

fprintf('Binarization method: %s\n', upper(BINARIZATION_METHOD));
if strcmp(BINARIZATION_METHOD, 'raw')
    fprintf('  Raw threshold: %.2f (dF/F units)\n', raw_activity_threshold);
else
    fprintf('  Z-score threshold: mean + %.1f×sigma\n', k_sigma_multiplier);
end

% -------------------------------------------------------------------------
% Timeframe Selection
% -------------------------------------------------------------------------
TimeFrameSelection = 1:185;  % Full trial
fprintf('Timeframe: frames %d to %d (full trial)\n', ...
    TimeFrameSelection(1), TimeFrameSelection(end));
fprintf('  Pre-stim period: frames 1:80\n');
fprintf('  Stimulus period: frames 81:100\n');
fprintf('  Post-stim period: frames 101:185\n');

% -------------------------------------------------------------------------
% Conditions and Skip Arrays
% -------------------------------------------------------------------------
conditions = {'Naive', 'Beginner', 'Expert', 'NoSpout'};

% Define Skip arrays (recordings to exclude from analysis)
Skip = struct();
Skip.Naive = [1 9 10 16];
Skip.Beginner = [1 6 7 11];
Skip.Expert = [1 4 12 13 14];
Skip.NoSpout = [1 4 9 10 11 13 14];

fprintf('Conditions: %s\n', strjoin(conditions, ', '));

% -------------------------------------------------------------------------
% Grid Metadata
% -------------------------------------------------------------------------
gridDimensions = [13, 26];  % [gridY, gridX]
gridSize = 40;  % microns per grid cell
totalGridCells = gridDimensions(1) * gridDimensions(2);  % 338 cells

fprintf('Grid: %d × %d = %d cells (%.0f μm per cell)\n', ...
    gridDimensions(1), gridDimensions(2), totalGridCells, gridSize);

% -------------------------------------------------------------------------
% Moran's I Parameters
% -------------------------------------------------------------------------
include_MoransI = true;
morans_distance_neighbors = 1;  % Only nearest neighbors

fprintf('Moran''s I: %s (nearest neighbors only)\n', ...
    string(include_MoransI));

% -------------------------------------------------------------------------
% Output Settings
% -------------------------------------------------------------------------
output_dir = 'Fig. 5 Ising Models\';
output_filename = 'ExperimentalData.mat';

fprintf('Output directory: %s\n', output_dir);

%% =========================================================================
%% SECTION 2: Load Data Structures
%% =========================================================================

fprintf('\n--- Section 2: Loading Data Structures ---\n');

% Load Grid40 rasterized data
% Structure: Grid40.ConditionIndividual.AllNeurons(recording).P1
% Data dimensions: [gridY × gridX × nTimepoints × nTrials]
if ~exist('Grid40', 'var')
    load(mba_p('Grid40.mat'), 'Grid40');
    fprintf('Grid40 structure loaded\n');
else
    fprintf('Grid40 already in workspace\n');
end

% Load recording metadata (for timing information)
load(mba_p('RawData3.mat'), 'Rec');
fprintf('Rec structure loaded\n');

% Load parameters structure (for condition colors and settings)
load(mba_p('RawData3.mat'), 'params');
fprintf('params structure loaded\n');

% Load Performance and Stimuli (for Expert Hit/Miss subsetting)
load(mba_p('RawData3.mat'), 'Performance');
fprintf('Performance structure loaded\n');
load(mba_p('RawData3.mat'), 'Stimuli');
fprintf('Stimuli structure loaded\n');

% Validate data structures
fprintf('\nConditions found in Grid40:\n');
for c = 1:length(conditions)
    conditionIndividual = [conditions{c} 'Individual'];
    if isfield(Grid40, conditionIndividual)
        nRecs = length(Grid40.(conditionIndividual).AllNeurons);
        fprintf('  %s: %d recordings\n', conditions{c}, nRecs);
    else
        fprintf('  %s: NOT FOUND\n', conditions{c});
    end
end

%% =========================================================================
%% SECTION 3: Binarize and Aggregate Data
%% =========================================================================

fprintf('\n--- Section 3: Binarizing and Aggregating Data ---\n');

% Initialize output structure
BinarisedData = struct();

% Initialize recording metadata
RecordingMetadata = struct();

% Process each condition
for c = 1:length(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    fprintf('\n=== Processing %s ===\n', condition);

    % Check if Grid40 data exists
    if ~isfield(Grid40, conditionIndividual)
        warning('No Grid40 data found for %s', condition);
        BinarisedData.(condition) = [];
        continue;
    end

    nRecsGrid = length(Grid40.(conditionIndividual).AllNeurons);

    % Collect all trials across recordings
    allTrialsData = [];
    nTrialsCollected = 0;
    nRecsProcessed = 0;
    recTrialCounts = [];
    recIndices = [];
    recAnimalNames = {};

    for r = 1:nRecsGrid
        % Skip excluded recordings
        if ismember(r, Skip.(condition))
            fprintf('  Rec %d: SKIPPED\n', r);
            continue;
        end

        % Check if P1 exists
        if ~isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
            fprintf('  Rec %d: No P1 data\n', r);
            continue;
        end

        gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

        % Dereference cell array if needed
        if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
            gridData_P1 = gridData_P1_cell{:};
        else
            gridData_P1 = gridData_P1_cell;
        end

        if isempty(gridData_P1)
            fprintf('  Rec %d: Empty data\n', r);
            continue;
        end

        % Get dimensions
        [gridY, gridX, nTimepoints, nTrials] = size(gridData_P1);

        % Validate timeframe selection
        if max(TimeFrameSelection) > nTimepoints
            warning('Rec %d: Requested timeframes exceed available (%d)', r, nTimepoints);
            continue;
        end

        % Select timeframe subset
        gridData_subset = gridData_P1(:, :, TimeFrameSelection, :);
        nTimepoints_selected = length(TimeFrameSelection);

        % Apply binarization based on selected method
        if strcmp(BINARIZATION_METHOD, 'raw')
            % Method 1: Raw absolute threshold
            binarized = double(gridData_subset > raw_activity_threshold);
            threshold_value = raw_activity_threshold;
        else
            % Method 2: Mean + k×sigma threshold (per recording)
            all_values = gridData_subset(:);
            recording_mean = mean(all_values, 'omitnan');
            recording_std = std(all_values, 'omitnan');
            threshold_value = recording_mean + k_sigma_multiplier * recording_std;
            binarized = double(gridData_subset > threshold_value);
        end

        % Concatenate trials from this recording
        allTrialsData = cat(4, allTrialsData, binarized);

        % Calculate sparsity for this recording
        sparsity = mean(binarized(:));

        nTrialsCollected = nTrialsCollected + nTrials;
        nRecsProcessed = nRecsProcessed + 1;

        % Collect recording metadata
        recTrialCounts = [recTrialCounts, nTrials];
        recIndices = [recIndices, r];
        if isfield(Rec, condition) && height(Rec.(condition)) >= r
            recAnimalNames{end+1} = char(Rec.(condition).AnimalName{r});
        else
            recAnimalNames{end+1} = sprintf('Unknown_Rec%d', r);
        end

        fprintf('  Rec %d: %d trials, threshold=%.3f, sparsity=%.3f\n', ...
            r, nTrials, threshold_value, sparsity);
    end

    % Store aggregated data for this condition
    if ~isempty(allTrialsData)
        BinarisedData.(condition) = allTrialsData;

        % Summary statistics
        overall_sparsity = mean(allTrialsData(:));
        fprintf('  %s TOTAL: %d recordings, %d trials, sparsity=%.3f\n', ...
            condition, nRecsProcessed, nTrialsCollected, overall_sparsity);
    else
        BinarisedData.(condition) = [];
        fprintf('  %s: NO DATA COLLECTED\n', condition);
    end

    % Store recording metadata for this condition
    RecordingMetadata.(condition).nTrials_per_recording = recTrialCounts;
    RecordingMetadata.(condition).recording_indices = recIndices;
    RecordingMetadata.(condition).animal_names = recAnimalNames;
    RecordingMetadata.(condition).n_recordings = nRecsProcessed;
end

fprintf('\nBinarization complete\n');

%% =========================================================================
%% SECTION 4: Calculate Moran's I Spatial Autocorrelation
%% =========================================================================

if ~include_MoransI
    fprintf('\n--- Skipping Moran''s I calculation (disabled) ---\n');
    MoransI = struct();
else
    fprintf('\n--- Section 4: Calculating Moran''s I Spatial Autocorrelation ---\n');
    fprintf('Using nearest neighbor weights (distance=%d)\n', morans_distance_neighbors);

    % Create spatial weight matrix (computed once for all conditions)
    % Use dummy grid to calculate distance matrix
    valueMap = rand(gridDimensions(1), gridDimensions(2));
    distanceMat = squareform(mL_distanceMat(valueMap));
    uniqueDistances = unique(distanceMat);
    uniqueDistances(uniqueDistances == 0) = [];  % Remove self-distance

    % Create weight matrix for nearest neighbors
    currDistInds = ismember(distanceMat, uniqueDistances(1:morans_distance_neighbors));
    weightMat = zeros(size(distanceMat));
    weightMat(currDistInds) = distanceMat(currDistInds);
    weightMat(weightMat == inf) = 0;

    fprintf('Weight matrix created: %d × %d\n', size(weightMat, 1), size(weightMat, 2));

    % Initialize Moran's I structure
    MoransI = struct();

    % Process each condition
    for c = 1:length(conditions)
        condition = conditions{c};

        fprintf('\n=== Processing %s ===\n', condition);

        if ~isfield(BinarisedData, condition) || isempty(BinarisedData.(condition))
            fprintf('  No binarized data for %s\n', condition);
            MoransI.(condition) = [];
            continue;
        end

        % Get binarized data
        binarized = BinarisedData.(condition);
        [gridY, gridX, nTimepoints, nTrials] = size(binarized);

        fprintf('  Computing Moran''s I for %d trials × %d timepoints\n', ...
            nTrials, nTimepoints);

        % Initialize Moran's I matrix [nTrials × nTimepoints]
        moransI_matrix = zeros(nTrials, nTimepoints);

        % Calculate for each trial and timepoint
        for trial = 1:nTrials
            for t = 1:nTimepoints
                % Extract 2D grid slice
                grid_2D = squeeze(binarized(:, :, t, trial));

                % Calculate Moran's I using helper function
                moransI_matrix(trial, t) = mL_moransI(grid_2D, weightMat);
            end

            % Progress indicator
            if mod(trial, 50) == 0 || trial == nTrials
                fprintf('    Progress: %d/%d trials completed\n', trial, nTrials);
            end
        end

        % Store results
        MoransI.(condition) = moransI_matrix;

        % Summary statistics
        mean_I = mean(moransI_matrix(:), 'omitnan');
        std_I = std(moransI_matrix(:), 'omitnan');
        min_I = min(moransI_matrix(:));
        max_I = max(moransI_matrix(:));

        fprintf('  %s Moran''s I: mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', ...
            condition, mean_I, std_I, min_I, max_I);
    end

    fprintf('\nMoran''s I calculation complete\n');
end

%% =========================================================================
%% SECTION 4b: Subset Expert into Expert_Hit and Expert_Miss
%% =========================================================================

fprintf('\n--- Section 4b: Subsetting Expert into Hit/Miss ---\n');

if isfield(BinarisedData, 'Expert') && ~isempty(BinarisedData.Expert) ...
        && isfield(RecordingMetadata, 'Expert')

    % Get per-recording trial counts and indices for Expert
    recTrialCounts_expert = RecordingMetadata.Expert.nTrials_per_recording;
    recIndices_expert = RecordingMetadata.Expert.recording_indices;
    recAnimalNames_expert = RecordingMetadata.Expert.animal_names;
    nRecsExpert = RecordingMetadata.Expert.n_recordings;

    % Total Expert trials
    nTrialsExpert = size(BinarisedData.Expert, 4);

    % Build global hit/miss masks across all Expert trials
    hitMask = false(1, nTrialsExpert);
    missMask = false(1, nTrialsExpert);

    % Per-recording metadata for Hit/Miss
    hitTrialCounts = [];
    missTrialCounts = [];

    trialOffset = 0;  % cumulative offset into pooled Expert array

    for ri = 1:nRecsExpert
        r = recIndices_expert(ri);  % original recording index
        nTrialsRec = recTrialCounts_expert(ri);

        % Absolute P1 trial indices for this recording
        p1Absolute = Stimuli.Expert(r).TrialsPosition1;

        % Absolute hit/miss trial indices
        hitAbsolute = Performance.Expert(r).hit;
        missAbsolute = Performance.Expert(r).miss;

        % Map absolute -> relative position within P1 trials
        [~, hitRelative] = ismember(hitAbsolute, p1Absolute);
        hitRelative = hitRelative(hitRelative > 0);  % keep only found

        [~, missRelative] = ismember(missAbsolute, p1Absolute);
        missRelative = missRelative(missRelative > 0);

        % Convert relative indices to global indices in pooled array
        hitGlobal = trialOffset + hitRelative;
        missGlobal = trialOffset + missRelative;

        hitMask(hitGlobal) = true;
        missMask(missGlobal) = true;

        hitTrialCounts = [hitTrialCounts, length(hitRelative)];
        missTrialCounts = [missTrialCounts, length(missRelative)];

        fprintf('  Rec %d (orig idx %d): %d P1 trials, %d hits, %d misses\n', ...
            ri, r, nTrialsRec, length(hitRelative), length(missRelative));

        trialOffset = trialOffset + nTrialsRec;
    end

    % Subset BinarisedData
    BinarisedData.Expert_Hit = BinarisedData.Expert(:, :, :, hitMask);
    BinarisedData.Expert_Miss = BinarisedData.Expert(:, :, :, missMask);

    % Subset MoransI (if computed)
    if isfield(MoransI, 'Expert') && ~isempty(MoransI.Expert)
        MoransI.Expert_Hit = MoransI.Expert(hitMask, :);
        MoransI.Expert_Miss = MoransI.Expert(missMask, :);
    end

    % Build RecordingMetadata for Hit/Miss
    RecordingMetadata.Expert_Hit.nTrials_per_recording = hitTrialCounts;
    RecordingMetadata.Expert_Hit.recording_indices = recIndices_expert;
    RecordingMetadata.Expert_Hit.animal_names = recAnimalNames_expert;
    RecordingMetadata.Expert_Hit.n_recordings = nRecsExpert;

    RecordingMetadata.Expert_Miss.nTrials_per_recording = missTrialCounts;
    RecordingMetadata.Expert_Miss.recording_indices = recIndices_expert;
    RecordingMetadata.Expert_Miss.animal_names = recAnimalNames_expert;
    RecordingMetadata.Expert_Miss.n_recordings = nRecsExpert;

    fprintf('\n  Expert_Hit: %d trials (sum per-rec: %d)\n', ...
        sum(hitMask), sum(hitTrialCounts));
    fprintf('  Expert_Miss: %d trials (sum per-rec: %d)\n', ...
        sum(missMask), sum(missTrialCounts));
    fprintf('  Expert total: %d, Hit+Miss: %d\n', ...
        nTrialsExpert, sum(hitMask) + sum(missMask));
else
    fprintf('  No Expert data to subset -- skipping\n');
end

% Update conditions list to include new sub-conditions for Sections 5-7
conditions = {'Naive', 'Beginner', 'Expert', 'Expert_Hit', 'Expert_Miss', 'NoSpout'};
Conditions = conditions;

fprintf('\nSection 4b complete\n');

%% =========================================================================
%% SECTION 5: Create Timing Structure
%% =========================================================================

fprintf('\n--- Section 5: Creating Timing Information Structure ---\n');

% Create comprehensive timing metadata
TimingInfo = struct();

% Global timing parameters
TimingInfo.imaging_rate = 10;  % Hz (standard for 2-photon imaging)
TimingInfo.frame_duration_ms = 100;  % milliseconds per frame

fprintf('Imaging parameters:\n');
fprintf('  Rate: %d Hz\n', TimingInfo.imaging_rate);
fprintf('  Frame duration: %d ms\n', TimingInfo.frame_duration_ms);

% Trial structure timing
TimingInfo.trial_structure = struct();
TimingInfo.trial_structure.total_frames = 185;  % Full trial length
TimingInfo.trial_structure.prestim_frames = 1:80;     % Pre-stimulus period
TimingInfo.trial_structure.stimulus_frames = 81:100;  % Stimulus presentation
TimingInfo.trial_structure.poststim_frames = 101:185; % Post-stimulus period

% Stimulus timing
TimingInfo.stimulus = struct();
TimingInfo.stimulus.onset_frame = 81;
TimingInfo.stimulus.offset_frame = 100;
TimingInfo.stimulus.duration_frames = 20;  % 100 - 81 + 1
TimingInfo.stimulus.duration_ms = TimingInfo.stimulus.duration_frames * ...
    TimingInfo.frame_duration_ms;

fprintf('Trial structure:\n');
fprintf('  Total frames: %d\n', TimingInfo.trial_structure.total_frames);
fprintf('  Pre-stim: frames %d-%d (%d frames)\n', ...
    TimingInfo.trial_structure.prestim_frames(1), ...
    TimingInfo.trial_structure.prestim_frames(end), ...
    length(TimingInfo.trial_structure.prestim_frames));
fprintf('  Stimulus: frames %d-%d (%d frames, %.1f ms)\n', ...
    TimingInfo.stimulus.onset_frame, ...
    TimingInfo.stimulus.offset_frame, ...
    TimingInfo.stimulus.duration_frames, ...
    TimingInfo.stimulus.duration_ms);
fprintf('  Post-stim: frames %d-%d (%d frames)\n', ...
    TimingInfo.trial_structure.poststim_frames(1), ...
    TimingInfo.trial_structure.poststim_frames(end), ...
    length(TimingInfo.trial_structure.poststim_frames));

fprintf('\nTiming structure created\n');

%% =========================================================================
%% SECTION 6: Organize and Save Output
%% =========================================================================

fprintf('\n--- Section 6: Organizing Output Variables ---\n');

% Binarisation method information
binarisation_method = BINARIZATION_METHOD;
if strcmp(BINARIZATION_METHOD, 'raw')
    threshold_value = raw_activity_threshold;
    threshold_description = sprintf('Raw absolute threshold: %.2f dF/F', ...
        raw_activity_threshold);
else
    threshold_value = k_sigma_multiplier;
    threshold_description = sprintf('Z-score threshold: mean + %.1f×sigma', ...
        k_sigma_multiplier);
end

fprintf('Method: %s\n', threshold_description);

% Grid metadata
GridMetadata = struct();
GridMetadata.gridDimensions = gridDimensions;
GridMetadata.gridSize_microns = gridSize;
GridMetadata.totalGridCells = totalGridCells;

% Condition information (cell array)
Conditions = conditions;

fprintf('Output variables organized\n');

% -------------------------------------------------------------------------
% Save Data
% -------------------------------------------------------------------------
fprintf('\n--- Saving Data ---\n');

% Create output directory if needed
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created directory: %s\n', output_dir);
end

% Full save path
save_path = fullfile(output_dir, output_filename);

% Save individual variables with compression for large files
fprintf('Saving to: %s\n', save_path);
save(save_path, 'BinarisedData', 'MoransI', 'TimingInfo', 'GridMetadata', ...
    'Conditions', 'RecordingMetadata', 'binarisation_method', ...
    'threshold_value', 'threshold_description', '-v7.3');

% Get file size
file_info = dir(save_path);
file_size_MB = file_info.bytes / (1024^2);
fprintf('File saved successfully (%.2f MB)\n', file_size_MB);

% -------------------------------------------------------------------------
% Export to Python-Compatible Formats
% -------------------------------------------------------------------------
fprintf('\n--- Exporting to Python Formats ---\n');

% Convert MATLAB structs to Python-compatible nested dictionaries
% This preserves the hierarchical structure when loading in Python

% Helper function to recursively convert struct to nested dict
function py_dict = struct_to_pydict(matlab_struct)
    if isstruct(matlab_struct)
        py_dict = struct();
        fields = fieldnames(matlab_struct);
        for i = 1:length(fields)
            field = fields{i};
            value = matlab_struct.(field);
            if isstruct(value)
                py_dict.(field) = struct_to_pydict(value);
            elseif iscell(value)
                % Convert cell arrays to regular arrays if possible
                if all(cellfun(@isnumeric, value))
                    py_dict.(field) = cell2mat(value);
                else
                    py_dict.(field) = value;
                end
            else
                py_dict.(field) = value;
            end
        end
    else
        py_dict = matlab_struct;
    end
end

% Convert main data structures
BinarisedData_py = struct_to_pydict(BinarisedData);
MoransI_py = struct_to_pydict(MoransI);
TimingInfo_py = struct_to_pydict(TimingInfo);
GridMetadata_py = struct_to_pydict(GridMetadata);

% -------------------------------------------------------------------------
% Save NPZ (NumPy Compressed Archive)
% -------------------------------------------------------------------------
npz_filename = 'ExperimentalData.npz';
npz_path = fullfile(output_dir, npz_filename);

fprintf('Saving NPZ format to: %s\n', npz_path);

% Create a structure with all variables for NPZ export
% Note: NPZ files store data as dictionaries, so we'll save the structs
% as nested dictionaries that can be loaded with allow_pickle=True
save_npz_temp = fullfile(output_dir, 'temp_for_npz.mat');
save(save_npz_temp, 'BinarisedData_py', 'MoransI_py', 'TimingInfo_py', ...
    'GridMetadata_py', 'Conditions', 'binarisation_method', ...
    'threshold_value', 'threshold_description', '-v7');

% Use Python to convert MAT to NPZ
python_npz_code = sprintf([...
    'import scipy.io\n' ...
    'import numpy as np\n' ...
    'import os\n\n' ...
    'mat_data = scipy.io.loadmat(r"%s")\n' ...
    'npz_dict = {}\n\n' ...
    '# Extract main data structures\n' ...
    'for key in [''BinarisedData_py'', ''MoransI_py'', ''TimingInfo_py'', ''GridMetadata_py'']:\n' ...
    '    if key in mat_data:\n' ...
    '        npz_dict[key.replace(''_py'', '''')] = mat_data[key]\n\n' ...
    '# Add simple variables\n' ...
    'for key in [''Conditions'', ''binarisation_method'', ''threshold_value'', ''threshold_description'']:\n' ...
    '    if key in mat_data:\n' ...
    '        npz_dict[key] = mat_data[key]\n\n' ...
    'np.savez_compressed(r"%s", **npz_dict)\n' ...
    'os.remove(r"%s")\n' ...
    'print("NPZ file saved successfully")\n'], ...
    save_npz_temp, npz_path, save_npz_temp);

python_npz_script = fullfile(output_dir, 'convert_to_npz.py');
fid = fopen(python_npz_script, 'w');
fprintf(fid, '%s', python_npz_code);
fclose(fid);

% Execute Python script
[status_npz, result_npz] = system(sprintf('python "%s"', python_npz_script));
if status_npz == 0
    fprintf('NPZ export successful\n');
    delete(python_npz_script);
    if exist(npz_path, 'file')
        npz_info = dir(npz_path);
        npz_size_MB = npz_info.bytes / (1024^2);
        fprintf('NPZ file size: %.2f MB\n', npz_size_MB);
    end
else
    warning('NPZ export failed: %s', result_npz);
end

% -------------------------------------------------------------------------
% Save H5 (HDF5 Format)
% -------------------------------------------------------------------------
h5_filename = 'ExperimentalData.h5';
h5_path = fullfile(output_dir, h5_filename);

fprintf('Saving H5 format to: %s\n', h5_path);

% Delete existing file if it exists
if exist(h5_path, 'file')
    delete(h5_path);
end

% Helper function to recursively write struct to HDF5
function write_struct_to_h5(h5_file, group_path, data_struct)
    fields = fieldnames(data_struct);
    for i = 1:length(fields)
        field = fields{i};
        value = data_struct.(field);
        current_path = [group_path '/' field];

        if isstruct(value)
            % Recursively write nested structs
            write_struct_to_h5(h5_file, current_path, value);
        elseif isnumeric(value) || islogical(value)
            % Write numeric/logical arrays as datasets
            h5create(h5_file, current_path, size(value), 'Datatype', class(value));
            h5write(h5_file, current_path, value);
        elseif ischar(value) || isstring(value)
            % Write strings
            h5create(h5_file, current_path, size(value), 'Datatype', 'string');
            h5write(h5_file, current_path, string(value));
        elseif iscell(value)
            % Handle cell arrays - convert to string array if all strings
            if all(cellfun(@ischar, value)) || all(cellfun(@isstring, value))
                str_array = string(value);
                h5create(h5_file, current_path, size(str_array), 'Datatype', 'string');
                h5write(h5_file, current_path, str_array);
            end
        end
    end
end

try
    % Write BinarisedData
    write_struct_to_h5(h5_path, '/BinarisedData', BinarisedData_py);

    % Write MoransI
    write_struct_to_h5(h5_path, '/MoransI', MoransI_py);

    % Write TimingInfo
    write_struct_to_h5(h5_path, '/TimingInfo', TimingInfo_py);

    % Write GridMetadata
    write_struct_to_h5(h5_path, '/GridMetadata', GridMetadata_py);

    % Write simple variables
    h5create(h5_path, '/Conditions', size(Conditions), 'Datatype', 'string');
    h5write(h5_path, '/Conditions', string(Conditions));

    h5create(h5_path, '/binarisation_method', [1 1], 'Datatype', 'string');
    h5write(h5_path, '/binarisation_method', string(binarisation_method));

    h5create(h5_path, '/threshold_value', [1 1]);
    h5write(h5_path, '/threshold_value', threshold_value);

    h5create(h5_path, '/threshold_description', [1 1], 'Datatype', 'string');
    h5write(h5_path, '/threshold_description', string(threshold_description));

    % Write RecordingMetadata
    for c_idx = 1:length(conditions)
        cond = conditions{c_idx};
        if isfield(RecordingMetadata, cond)
            rm = RecordingMetadata.(cond);
            base = ['/RecordingMetadata/' cond];
            h5create(h5_path, [base '/nTrials_per_recording'], size(rm.nTrials_per_recording));
            h5write(h5_path, [base '/nTrials_per_recording'], rm.nTrials_per_recording);
            h5create(h5_path, [base '/recording_indices'], size(rm.recording_indices));
            h5write(h5_path, [base '/recording_indices'], rm.recording_indices);
            h5create(h5_path, [base '/animal_names'], size(string(rm.animal_names)), 'Datatype', 'string');
            h5write(h5_path, [base '/animal_names'], string(rm.animal_names));
        end
    end

    fprintf('H5 export successful\n');
    if exist(h5_path, 'file')
        h5_info = dir(h5_path);
        h5_size_MB = h5_info.bytes / (1024^2);
        fprintf('H5 file size: %.2f MB\n', h5_size_MB);
    end
catch ME
    warning('H5 export failed: %s', ME.message);
    h5_size_MB = 0;
end

fprintf('\nAll export formats completed\n');

%% =========================================================================
%% SECTION 7: Generate Summary Report
%% =========================================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('  DATA AGGREGATION COMPLETE\n');
fprintf('========================================\n');
fprintf('\n');

fprintf('OUTPUT FILES:\n');
fprintf('  MATLAB format (.mat):\n');
fprintf('    Path: %s\n', save_path);
fprintf('    Size: %.2f MB\n', file_size_MB);
if exist('npz_path', 'var') && exist(npz_path, 'file')
    fprintf('\n');
    fprintf('  NumPy format (.npz):\n');
    fprintf('    Path: %s\n', npz_path);
    fprintf('    Size: %.2f MB\n', npz_size_MB);
    fprintf('    Load in Python: data = np.load(''ExperimentalData.npz'', allow_pickle=True)\n');
end
if exist('h5_path', 'var') && exist(h5_path, 'file')
    fprintf('\n');
    fprintf('  HDF5 format (.h5):\n');
    fprintf('    Path: %s\n', h5_path);
    fprintf('    Size: %.2f MB\n', h5_size_MB);
    fprintf('    Load in Python: import h5py; f = h5py.File(''ExperimentalData.h5'', ''r'')\n');
end
fprintf('\n');

fprintf('BINARIZATION:\n');
fprintf('  Method: %s\n', upper(BINARIZATION_METHOD));
fprintf('  %s\n', threshold_description);
fprintf('\n');

fprintf('DATA DIMENSIONS:\n');
fprintf('  Grid: %d × %d = %d cells\n', ...
    gridDimensions(1), gridDimensions(2), totalGridCells);
fprintf('  Timeframes: %d to %d (%d frames)\n', ...
    TimeFrameSelection(1), TimeFrameSelection(end), ...
    length(TimeFrameSelection));
fprintf('  Pre-stim period: frames 1:80\n');
fprintf('  Stimulus period: frames 81:100\n');
fprintf('  Post-stim period: frames 101:185\n');
fprintf('\n');

fprintf('CONDITIONS PROCESSED:\n');
for c = 1:length(conditions)
    condition = conditions{c};
    if isfield(BinarisedData, condition) && ~isempty(BinarisedData.(condition))

        dims = size(BinarisedData.(condition));
        nTrials = dims(4);
        sparsity = mean(BinarisedData.(condition)(:));

        fprintf('  %n', condition);
        fprintf('    Dimensions: %d × %d × %d × %d\n', dims(1), dims(2), dims(3), dims(4));
        fprintf('    Total trials: %d\n', nTrials);
        fprintf('    Sparsity: %.3f (%.1f%% active)\n', sparsity, sparsity*100);

        if isfield(MoransI, condition) && ~isempty(MoransI.(condition))
            mean_I = mean(MoransI.(condition)(:), 'omitnan');
            fprintf('    Moran''s I: %.4f (mean)\n', mean_I);
        end
        fprintf('\n');
    else
        fprintf('  %s: NO DATA\n\n', condition);
    end
end

fprintf('DATA STRUCTURE:\n');
fprintf('  Variables saved in ExperimentalData.mat:\n');
fprintf('\n');
fprintf('  BinarisedData.Condition\n');
fprintf('    Format: [gridY × gridX × nTimepoints × nTrials]\n');
fprintf('    Example: BinarisedData.Naive, BinarisedData.Expert\n');
fprintf('\n');
fprintf('  MoransI.Condition\n');
fprintf('    Format: [nTrials × nTimepoints]\n');
fprintf('    Example: MoransI.Beginner, MoransI.NoSpout\n');
fprintf('\n');
fprintf('  TimingInfo\n');
fprintf('    Fields: trial_structure, stimulus\n');
fprintf('    Use TimingInfo.trial_structure.prestim_frames (1:80) for subsetting\n');
fprintf('    Use TimingInfo.trial_structure.stimulus_frames (81:100)\n');
fprintf('    Use TimingInfo.trial_structure.poststim_frames (101:185)\n');
fprintf('\n');
fprintf('  GridMetadata\n');
fprintf('    Fields: gridDimensions, gridSize_microns, totalGridCells\n');
fprintf('\n');
fprintf('  Conditions\n');
fprintf('    Cell array: {''Naive'', ''Beginner'', ''Expert'', ''Expert_Hit'', ''Expert_Miss'', ''NoSpout''}\n');
fprintf('\n');
fprintf('  binarisation_method, threshold_value, threshold_description\n');
fprintf('    Metadata about binarisation approach\n');
fprintf('\n');

fprintf('PYTHON USAGE EXAMPLES:\n');
fprintf('\n');
fprintf('  NumPy (.npz):\n');
fprintf('    import numpy as np\n');
fprintf('    data = np.load(''ExperimentalData.npz'', allow_pickle=True)\n');
fprintf('    naive_data = data[''BinarisedData''].item()[''Naive'']\n');
fprintf('    prestim_frames = data[''TimingInfo''].item()[''trial_structure''][''prestim_frames'']\n');
fprintf('\n');
fprintf('  HDF5 (.h5):\n');
fprintf('    import h5py\n');
fprintf('    with h5py.File(''ExperimentalData.h5'', ''r'') as n');
fprintf('        naive_data = f[''BinarisedData/Naive''][:]\n');
fprintf('        prestim_frames = f[''TimingInfo/trial_structure/prestim_frames''][:]\n');
fprintf('\n');

fprintf('========================================\n');
fprintf('Ready for Ising model training!\n');
fprintf('========================================\n');
