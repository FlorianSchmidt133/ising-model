%% =========================================================================
%% Fig 5: Entropy Comparison - Grid vs Activity Data (Raw Threshold Version)
%% =========================================================================
% This script compares population entropy calculated from:
% 1. Grid40 data (spatially rasterized, 13×26 grid cells)
% 2. ActivityData (individual neurons, ~3000 cells)
%
% Purpose: Understand how spatial coarse-graining affects entropy measurements
% and whether grid-based approaches capture similar population dynamics
%
% NOTE: This version uses ABSOLUTE RAW ACTIVITY THRESHOLDS for binarization
%       instead of z-score based thresholds (activity > raw_threshold)

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
include_MoransI = true;

% Raw activity threshold for entropy and dispersion calculations
raw_activity_threshold = 2;  % Absolute raw activity value threshold for binarization

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
% Note: Only BIN for Position1 is currently calculated (others are placeholders)
MoransI_Grid = struct();


%% =========================================================================
%% Section 1.1: Process each condition (calculate entropies and dispersion)
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
        Entropy_Grid.(condition).(posField).RecordingEntropyZ_Raw = {};
        Entropy_Grid.(condition).(posField).RecordingEntropyZ_RZ = {};
        Entropy_Grid.(condition).(posField).RecordingEntropyZ_TZ = {};
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
        Dispersion_Grid.(condition).(posField).RecordingDispersionZ_Raw = {};
        Dispersion_Grid.(condition).(posField).RecordingDispersionZ_RZ = {};
        Dispersion_Grid.(condition).(posField).RecordingDispersionZ_TZ = {};
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
        ActiveCells_Grid.(condition).(posField).RecordingActiveCellsZ_Raw = {};
        ActiveCells_Grid.(condition).(posField).RecordingActiveCellsZ_RZ = {};
        ActiveCells_Grid.(condition).(posField).RecordingActiveCellsZ_TZ = {};
    end

    % Initialize Moran's I storage for this condition (spatial autocorrelation)
    MoransI_Grid.(condition).Position1 = struct();
    MoransI_Grid.(condition).Position3 = struct();
    MoransI_Grid.(condition).All = struct();

    % Initialize CELL ARRAYS for Moran's I (per-recording structure)
    % NOTE: Only BIN for Position1 is currently calculated
    for pos = {'Position1', 'Position3', 'All'}
        posField = pos{1};
        MoransI_Grid.(condition).(posField).RecordingMoransI_BIN = {};          % CALCULATED for Position1 only
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

    %% Per-animal statistics (NOT NEEDED for raw thresholds)
    % NOTE: Using absolute raw threshold (no per-animal or per-recording normalization)

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

        % NOTE: Animal-level statistics not needed for raw threshold approach

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
                % Calculate entropy from Grid data
                [ent_raw_P1, ent_rz_P1, ent_tz_P1, entZ_raw_P1, entZ_rz_P1, entZ_tz_P1, ent_bin_P1] = ...
                    calculate_entropy_from_grid(gridData_P1, gridDimensions, raw_activity_threshold);

                % Store entropy values in CELL ARRAYS (one cell per recording)
                Entropy_Grid.(condition).Position1.RecordingEntropy_Raw{end+1} = ent_raw_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropy_RZ{end+1} = ent_rz_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropy_TZ{end+1} = ent_tz_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropy_BIN{end+1} = ent_bin_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropyZ_Raw{end+1} = entZ_raw_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropyZ_RZ{end+1} = entZ_rz_P1;
                Entropy_Grid.(condition).Position1.RecordingEntropyZ_TZ{end+1} = entZ_tz_P1;

                % Calculate dispersion from Grid data
                [disp_raw_P1, disp_rz_P1, disp_tz_P1, dispZ_raw_P1, dispZ_rz_P1, dispZ_tz_P1, disp_bin_P1] = ...
                    calculate_dispersion_from_grid(gridData_P1, gridDimensions, raw_activity_threshold);

                % Store dispersion values in CELL ARRAYS (one cell per recording)
                Dispersion_Grid.(condition).Position1.RecordingDispersion_Raw{end+1} = disp_raw_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersion_RZ{end+1} = disp_rz_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersion_TZ{end+1} = disp_tz_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersion_BIN{end+1} = disp_bin_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersionZ_Raw{end+1} = dispZ_raw_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersionZ_RZ{end+1} = dispZ_rz_P1;
                Dispersion_Grid.(condition).Position1.RecordingDispersionZ_TZ{end+1} = dispZ_tz_P1;

                % Calculate active cells count from Grid data
                [actcells_raw_P1, actcells_rz_P1, actcells_tz_P1, actcellsZ_raw_P1, actcellsZ_rz_P1, actcellsZ_tz_P1, actcells_bin_P1] = ...
                    calculate_active_cells_from_grid(gridData_P1, gridDimensions, raw_activity_threshold);

                % Store active cells values in CELL ARRAYS (one cell per recording)
                ActiveCells_Grid.(condition).Position1.RecordingActiveCells_Raw{end+1} = actcells_raw_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCells_RZ{end+1} = actcells_rz_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCells_TZ{end+1} = actcells_tz_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCells_BIN{end+1} = actcells_bin_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCellsZ_Raw{end+1} = actcellsZ_raw_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCellsZ_RZ{end+1} = actcellsZ_rz_P1;
                ActiveCells_Grid.(condition).Position1.RecordingActiveCellsZ_TZ{end+1} = actcellsZ_tz_P1;

                % Calculate Moran's I from Grid data (spatial autocorrelation)
                % Uses raw activity threshold
                if include_MoransI
                    morans_bin_P1 = calculate_moransI_BIN(gridData_P1, gridDimensions, raw_activity_threshold);
                    MoransI_Grid.(condition).Position1.RecordingMoransI_BIN{end+1} = morans_bin_P1;
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
                % Calculate entropy from Grid data
                [ent_raw_P3, ent_rz_P3, ent_tz_P3, entZ_raw_P3, entZ_rz_P3, entZ_tz_P3, ent_bin_P3] = ...
                    calculate_entropy_from_grid(gridData_P3, gridDimensions, raw_activity_threshold);

                % Store entropy values in CELL ARRAYS (one cell per recording)
                Entropy_Grid.(condition).Position3.RecordingEntropy_Raw{end+1} = ent_raw_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropy_RZ{end+1} = ent_rz_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropy_TZ{end+1} = ent_tz_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropy_BIN{end+1} = ent_bin_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropyZ_Raw{end+1} = entZ_raw_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropyZ_RZ{end+1} = entZ_rz_P3;
                Entropy_Grid.(condition).Position3.RecordingEntropyZ_TZ{end+1} = entZ_tz_P3;

                % Calculate dispersion from Grid data
                [disp_raw_P3, disp_rz_P3, disp_tz_P3, dispZ_raw_P3, dispZ_rz_P3, dispZ_tz_P3, disp_bin_P3] = ...
                    calculate_dispersion_from_grid(gridData_P3, gridDimensions, raw_activity_threshold);

                % Store dispersion values in CELL ARRAYS (one cell per recording)
                Dispersion_Grid.(condition).Position3.RecordingDispersion_Raw{end+1} = disp_raw_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersion_RZ{end+1} = disp_rz_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersion_TZ{end+1} = disp_tz_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersion_BIN{end+1} = disp_bin_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersionZ_Raw{end+1} = dispZ_raw_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersionZ_RZ{end+1} = dispZ_rz_P3;
                Dispersion_Grid.(condition).Position3.RecordingDispersionZ_TZ{end+1} = dispZ_tz_P3;

                % Calculate active cells count from Grid data
                [actcells_raw_P3, actcells_rz_P3, actcells_tz_P3, actcellsZ_raw_P3, actcellsZ_rz_P3, actcellsZ_tz_P3, actcells_bin_P3] = ...
                    calculate_active_cells_from_grid(gridData_P3, gridDimensions, raw_activity_threshold);

                % Store active cells values in CELL ARRAYS (one cell per recording)
                ActiveCells_Grid.(condition).Position3.RecordingActiveCells_Raw{end+1} = actcells_raw_P3;
                ActiveCells_Grid.(condition).Position3.RecordingActiveCells_RZ{end+1} = actcells_rz_P3;
                ActiveCells_Grid.(condition).Position3.RecordingActiveCells_TZ{end+1} = actcells_tz_P3;
                ActiveCells_Grid.(condition).Position3.RecordingActiveCells_BIN{end+1} = actcells_bin_P3;
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
    Entropy_Grid.(condition).All.RecordingEntropyZ_Raw = ...
        [Entropy_Grid.(condition).Position1.RecordingEntropyZ_Raw, ...
         Entropy_Grid.(condition).Position3.RecordingEntropyZ_Raw];
    Entropy_Grid.(condition).All.RecordingEntropyZ_RZ = ...
        [Entropy_Grid.(condition).Position1.RecordingEntropyZ_RZ, ...
         Entropy_Grid.(condition).Position3.RecordingEntropyZ_RZ];
    Entropy_Grid.(condition).All.RecordingEntropyZ_TZ = ...
        [Entropy_Grid.(condition).Position1.RecordingEntropyZ_TZ, ...
         Entropy_Grid.(condition).Position3.RecordingEntropyZ_TZ];
    % Note: BIN uses raw activity threshold for binarisation (no z-scoring)

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
    % NOTE: Only Position1 BIN is calculated, Position3 is empty
    MoransI_Grid.(condition).All.RecordingMoransI_BIN = ...
        [MoransI_Grid.(condition).Position1.RecordingMoransI_BIN, ...
         MoransI_Grid.(condition).Position3.RecordingMoransI_BIN];  % P1 calculated, P3 empty

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
%   'BIN'     - Binarised entropy comparison: Raw Activity vs Binarised Grid
%               Uses absolute raw activity threshold (activity > raw_threshold)
%               Grid uses binarised (on/off) states based on raw values
%               Compares continuous neuron-level entropy to discrete grid-state entropy
% UseZScored options:
%   false - Use entropy calculated from raw neural activity
%   true  - Use entropy calculated from z-scored neural activity (entZ_*)
Type = 'BIN';  % Options: 'Raw', 'RZ', 'TZ', 'BIN'
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
    % NOTE: Only BIN for Position1 is currently calculated

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
% 
% % Create folder path based on Type, UseZScored, and Position
% zScoreLabel = '';
% if UseZScored
%     zScoreLabel = '_ZScored';
% end
% 
% % Save Entropy Comparison figures
% savePathEntropy = sprintf('Fig. 5 Activity, Entropy & Dispersion Comparison\\%s%s_%s\\', ...
%     Type, zScoreLabel, Position);
% fprintf('\nSaving Entropy Comparison figures to: %s\n', savePathEntropy);
% saveMyFig("Fig5_EntropyComparison", savePathEntropy, 'All')
% 
% % Save Dispersion Analysis figures
% savePathDispersion = sprintf('Fig. 5 Entropy Comparison\\Dispersion_%s%s_%s\\', ...
%     Type, zScoreLabel, Position);
% fprintf('Saving Dispersion Analysis figures to: %s\n', savePathDispersion);
% saveMyFig("Fig5_DispersionAnalysis", savePathDispersion, 'All')
% 
% % Save Dispersion Metric Explanation figures (Figure 1: Binary Only, Figure 2: Comparison)
% savePathExplanation = sprintf('Fig. 5 Entropy Comparison\\Dispersion_Explanation\\');
% fprintf('Saving Dispersion Metric Explanation figures to: %s\n', savePathExplanation);
% saveMyFig("Figure 1: Binary Dispersion Metric", savePathExplanation, 'All')
% saveMyFig("Figure 2: Binary vs Weighted Dispersion", savePathExplanation, 'All')
% 
% % Save Active Cells Analysis figures
% savePathActiveCells = sprintf('Fig. 5 Entropy Comparison\\ActiveCells_%s%s_%s\\', ...
%     Type, zScoreLabel, Position);
% fprintf('Saving Active Cells Analysis figures to: %s\n', savePathActiveCells);
% saveMyFig("Fig5_ActiveCellsAnalysis", savePathActiveCells, 'All')
% 
% % Save Moran's I Analysis figures (if calculated)
% if include_MoransI
%     savePathMoransI = sprintf('Fig. 5 Entropy Comparison\\MoransI_%s%s_%s\\', ...
%         Type, zScoreLabel, Position);
%     fprintf('Saving Moran''s I Analysis figures to: %s\n', savePathMoransI);
%     saveMyFig("Fig5_MoransIAnalysis", savePathMoransI, 'All')
% 
%     % Save Moran's I Metric Explanation figures
%     savePathMoransIExplanation = sprintf('Fig. 5 Entropy Comparison\\MoransI_Explanation\\');
%     fprintf('Saving Moran''s I Metric Explanation figures to: %s\n', savePathMoransIExplanation);
%     saveMyFig("Figure 1: Moran's I Calculation", savePathMoransIExplanation, 'All')
%     saveMyFig("Figure 2: Entropy vs Dispersion vs Moran's I", savePathMoransIExplanation, 'All')
% end
% 
% if include_MoransI
%     fprintf('\n=== Figure 5 Entropy, Dispersion, Active Cells & Moran''s I Analysis Complete ===\n');
% else
%     fprintf('\n=== Figure 5 Entropy, Dispersion, and Active Cells Analysis Complete ===\n');
% end

%% =========================================================================
%% Helper Function: Calculate Entropy from Grid Data
%% =========================================================================
function [ent_raw, ent_rz, ent_tz, entZ_raw, entZ_rz, entZ_tz, ent_bin] = ...
    calculate_entropy_from_grid(gridData, gridDimensions, raw_activity_threshold)
% CALCULATE_ENTROPY_FROM_GRID - Calculate population entropy from Grid40 data
%
% INPUTS:
%   gridData               - Grid data [gridY × gridX × nTimepoints × nTrials]
%   gridDimensions         - [gridY, gridX] dimensions
%   raw_activity_threshold - Absolute activity threshold for binarization
%
% OUTPUTS:
%   ent_raw     - Raw entropy [nTrials × nTimepoints]
%   ent_rz      - Recording-zscore entropy
%   ent_tz      - Trial-zscore entropy
%   entZ_*      - Same metrics on z-scored grid data (Raw, RZ, TZ only)
%   ent_bin     - Entropy from binarised data (activity > raw_activity_threshold)

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

    %% Calculate BINARISED entropy using ABSOLUTE RAW THRESHOLD
    % Initialize binarised entropy array
    ent_bin = zeros(nTrials, nTimepoints);

    for trial = 1:nTrials
        for t = 1:nTimepoints
            % Extract grid cell activities at this trial/timepoint
            activity = gridData_reshaped(:, t, trial);

            % Binarise using absolute raw activity threshold: activity > threshold
            activity_bin = double(activity > raw_activity_threshold);
            ent_bin(trial, t) = population_entropy(activity_bin);
        end
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
function [disp_raw, disp_rz, disp_tz, dispZ_raw, dispZ_rz, dispZ_tz, disp_bin] = ...
    calculate_dispersion_from_grid(gridData, gridDimensions, raw_activity_threshold)
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
%   gridData               - Grid data [gridY × gridX × nTimepoints × nTrials]
%   gridDimensions         - [gridY, gridX] dimensions
%   raw_activity_threshold - Absolute activity threshold for binarization
%
% OUTPUTS:
%   disp_raw     - Raw dispersion [nTrials × nTimepoints] (weighted by activity)
%   disp_rz      - Recording-zscore dispersion
%   disp_tz      - Trial-zscore dispersion
%   dispZ_*      - Same metrics on z-scored grid data (Raw, RZ, TZ only)
%   disp_bin     - Dispersion from binarised data (activity > raw_activity_threshold)

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

    %% Calculate BINARISED dispersion using ABSOLUTE RAW THRESHOLD
    disp_bin = zeros(nTrials, nTimepoints);

    fprintf('    Calculating dispersion (raw threshold)...\n');
    for trial = 1:nTrials
        for t = 1:nTimepoints
            activity = gridData_reshaped(:, t, trial);

            % Binarise using absolute raw activity threshold: activity > threshold
            activity_bin = double(activity > raw_activity_threshold);

            % Calculate binary dispersion (matching Python implementation)
            disp_bin(trial, t) = calculate_binary_dispersion(activity_bin, grid_coords);
        end
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
function [actcells_raw, actcells_rz, actcells_tz, actcellsZ_raw, actcellsZ_rz, actcellsZ_tz, actcells_bin] = ...
    calculate_active_cells_from_grid(gridData, gridDimensions, raw_activity_threshold)
% CALCULATE_ACTIVE_CELLS_FROM_GRID - Calculate number of active grid cells per frame
%
% Counts how many grid cells are active at each timepoint. This provides a measure
% of population sparsity and recruitment across different conditions.
%
% INPUTS:
%   gridData               - Grid data [gridY × gridX × nTimepoints × nTrials]
%   gridDimensions         - [gridY, gridX] dimensions
%   raw_activity_threshold - Absolute activity threshold for binarization
%
% OUTPUTS:
%   actcells_raw     - Active cells count from raw data [nTrials × nTimepoints]
%   actcells_rz      - Recording-zscore active cells count
%   actcells_tz      - Trial-zscore active cells count
%   actcellsZ_*      - Same metrics on z-scored grid data (Raw, RZ, TZ only)
%   actcells_bin     - Active cells from binarised data (activity > raw_activity_threshold)

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

    %% Calculate BINARISED active cells count using ABSOLUTE RAW THRESHOLD
    actcells_bin = zeros(nTrials, nTimepoints);

    for trial = 1:nTrials
        for t = 1:nTimepoints
            activity = gridData_reshaped(:, t, trial);

            % Binarise using absolute raw activity threshold: activity > threshold
            activity_bin = double(activity > raw_activity_threshold);

            % Count active cells (binary = 1)
            actcells_bin(trial, t) = sum(activity_bin);
        end
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
%% Helper Function: Calculate Moran's I from Grid Data
%% =========================================================================
function morans_bin = calculate_moransI_BIN(gridData, gridDimensions, raw_activity_threshold)
% CALCULATE_MORANSI_BIN - Calculate Moran's I spatial autocorrelation using raw threshold
%
% Moran's I measures spatial autocorrelation - whether similar values cluster together.
% Positive values indicate clustering, negative values indicate dispersion.
%
% INPUTS:
%   gridData               - Grid data [gridY × gridX × nTimepoints × nTrials]
%   gridDimensions         - [gridY, gridX] dimensions
%   raw_activity_threshold - Absolute activity threshold for binarization
%
% OUTPUTS:
%   morans_bin  - Moran's I for binarised data (activity > raw_activity_threshold)
%                 [nTrials × nTimepoints]

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
    morans_bin = zeros(nTrials, nTimepoints);

    % Calculate Moran's I for each trial and timepoint using raw threshold
    fprintf('      Calculating Moran''s I (raw threshold)...\n');
    for trial = 1:nTrials
        for t = 1:nTimepoints
            % Extract grid at this trial/timepoint
            grid_2D = gridData(:, :, t, trial);

            % Binarise using absolute raw activity threshold
            grid_2D_bin = double(grid_2D > raw_activity_threshold);

            % Calculate Moran's I using mL_moransI
            morans_bin(trial, t) = mL_moransI(grid_2D_bin, currWeightMat);
        end
    end
end

