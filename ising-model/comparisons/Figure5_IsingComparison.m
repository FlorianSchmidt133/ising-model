%% =========================================================================
%% Figure 5: Ising Model vs Experimental Data Comparison using Moran's I
%% =========================================================================
% This script compares Ising model simulations (5D parameter grid search)
% to experimental neural data from 4 conditions using Moran's I spatial
% autocorrelation.
%
% Ising Parameter Grid:
%   beta:             Inverse temperature
%   c:                Coupling strength
%   decay_const:      Decay constant
%   inhibition_range: Inhibition range
%   bias:             Bias term
%
%
% Experimental Conditions:
%   Naive, Beginner, Expert, NoSpout
%
% Comparison Metrics:
%   1. Distribution comparison (Wasserstein distance)
%   2. Time series dynamics (autocorrelation)
%
% Output:
%   - Best-matching simulations per condition
%   - Parameter regions characterizing each condition
%   - Visualization of comparisons

%% =========================================================================
%% SECTION 1: Configuration
%% =========================================================================

fprintf('\n=== Figure 5: Ising Model vs Experimental Data Comparison ===\n\n');

% -------------------------------------------------------------------------
% Grid Mode Selection
% -------------------------------------------------------------------------
% Options: 'subselect_centre' | 'subselect_tiled' | 'subselect_centre_vs_tiled'
%   'subselect_centre':        centre crop 13x26 from Ising 32x32 grid (single position)
%   'subselect_tiled':         Tile non-overlapping 13x26 grids, average Moran's I
%   'subselect_centre_vs_tiled': Compute BOTH centre and tiled, compare which matches better
config.gridMode = 'subselect_tiled';

% -------------------------------------------------------------------------
% Matching Metric Selection
% -------------------------------------------------------------------------
% Options: 'moransI' | 'activity' | 'autocorr' | 'blobCount' | 'blobPersistence' | 'combined' | 'moransI+activity' | 'spatial+persistence'
%   'moransI':            Match based on Moran's I distribution (Wasserstein distance)
%   'activity':           Match based on activity/sparsity distribution
%   'autocorr':           Match based on autocorrelation time constant (τ) - log-ratio distance
%   'blobCount':          Match based on blob count distribution (with temporal rescaling)
%   'blobPersistence':    Match based on blob lifetime distribution (with temporal rescaling)
%   'combined':           Weighted combination of all 4 metrics (Moran's I + Activity + Blob Count + Blob Persistence)
%   'moransI+activity':   Weighted combination of Moran's I and Activity only
%   'spatial+persistence': Moran's I + Activity + Blob Persistence (no blob count)
config.matchingMetric = 'spatial+persistence';

% Weights for combined matching (used if matchingMetric = 'combined' or 'moransI+activity')
% Weights are automatically re-normalized to sum to 1
config.matchingWeights.moransI = 0.25;
config.matchingWeights.activity = 0.25;
config.matchingWeights.blobCount = 0.25;
config.matchingWeights.blobPersistence = 0.25;

% -------------------------------------------------------------------------
% Sequential Two-Step Matching Process
% -------------------------------------------------------------------------
% Step 1: Temporal rescaling - Match Ising tau to GLOBAL experimental tau
%         This establishes real-time scale for each simulation
% Step 2: Blob matching - Detect blobs in both data, compare distributions
%
% Both steps use GLOBAL references (averaged across all conditions)

% Autocorrelation parameters (for Step 1 - temporal rescaling)
config.autocorr.maxLag = 100;           % Maximum lag for ACF computation
config.autocorr.fitRange = [1, 10];     % Lag range for exponential decay fit

% Blob detection parameters for Step 2 (matched filter)
config.blob_params.sigma = 1.0;         % Best combined: Gauss(1.0)+Th(0.5)
config.blob_params.threshold = 0.5;     % Best combined: Gauss(1.0)+Th(0.5)
config.blob_params.minBlobSize = 10;    % Minimum blob size in voxels

% Grid dimensions
config.experimentalGrid = [13, 26];  % [rows, cols] for experimental data

% Experimental data frame selection
% Options: 'all'     (frames 1-185, full trial)
%          'prestim' (frames 1-80, baseline only)
%          'nostim'  (frames 1-80 + 101-180, excludes stimulus; doubles trial count)
config.expFrameSelection = 'all';
config.expPrestimFrames = 80;   % Number of pre-stimulus frames
config.expStimOnsetFrame = 81;  % First stimulus frame
config.expStimOffsetFrame = 100; % Last stimulus frame

% Ising data path (needed for auto-detection of grid size)
config.isingDataPath = mba_p('IsingModelData_39x78_100K');

% Auto-detect Ising grid size from first simulation file
simFilesTemp = dir(fullfile(config.isingDataPath, 'sim_be_*.mat'));
if isempty(simFilesTemp)
    error('No simulation files found in: %s', config.isingDataPath);
end
tempData = load(fullfile(config.isingDataPath, simFilesTemp(1).name), 'stored_spins');
config.isingGrid = [size(tempData.stored_spins, 2), size(tempData.stored_spins, 3)];
fprintf('Auto-detected Ising grid size: %d x %d\n', config.isingGrid(1), config.isingGrid(2));
clear simFilesTemp tempData;

fprintf('Grid mode: %s\n', upper(config.gridMode));
fprintf('Matching metric: %s\n', upper(config.matchingMetric));
if ismember(config.matchingMetric, {'autocorr', 'blobCount', 'blobPersistence', 'combined', 'spatial+persistence'})
    fprintf('Sequential matching: Step 1 (temporal rescaling) + Step 2 (blob matching)\n');
    fprintf('  ACF: Max lag: %d, Fit range: [%d, %d]\n', ...
        config.autocorr.maxLag, config.autocorr.fitRange(1), config.autocorr.fitRange(2));
    fprintf('  Blob params: sigma=%.1f, threshold=%.2f, minBlobSize=%d\n', ...
        config.blob_params.sigma, config.blob_params.threshold, config.blob_params.minBlobSize);
end
switch config.gridMode
    case 'subselect_centre'
        config.targetGrid = config.experimentalGrid;
        fprintf('  Ising %dx%d -> %dx%d (centre crop, single position)\n', config.isingGrid(1), config.isingGrid(2), ...
            config.targetGrid(1), config.targetGrid(2));
    case 'subselect_tiled'
        config.targetGrid = config.experimentalGrid;
        fprintf('  Ising %dx%d -> %dx%d (tiled, samples pooled across positions)\n', config.isingGrid(1), config.isingGrid(2), ...
            config.targetGrid(1), config.targetGrid(2));
    case 'subselect_centre_vs_tiled'
        config.targetGrid = config.experimentalGrid;
        fprintf('  Ising %dx%d -> %dx%d (BOTH centre and tiled, comparison mode)\n', config.isingGrid(1), config.isingGrid(2), ...
            config.targetGrid(1), config.targetGrid(2));
end

% -------------------------------------------------------------------------
% Data Paths
% -------------------------------------------------------------------------
config.experimentalDataPath = 'Fig. 5 Model\IsingModels\Data\ExperimentalData.mat';
config.outputPath = fullfile(config.isingDataPath, 'IsingComparison');

% Cache settings
config.useCache = true;  % Set to false to force recomputation
config.cachePath = fullfile(config.outputPath, 'Data', sprintf('IsingData_cache_%s_%s.mat', config.gridMode, config.matchingMetric));

fprintf('\nData paths:\n');
fprintf('  Ising data: %s\n', config.isingDataPath);
fprintf('  Experimental data: %s\n', config.experimentalDataPath);
fprintf('  Output: %s\n', config.outputPath);

% -------------------------------------------------------------------------
% Experimental Conditions
% -------------------------------------------------------------------------
config.conditions = {'Naive', 'Beginner', 'Expert', 'NoSpout'};
fprintf('\nConditions: %s\n', strjoin(config.conditions, ', '));

% -------------------------------------------------------------------------
% Condition Colors (for plotting)
% -------------------------------------------------------------------------
config.colors.Naive = [0.3373, 0.7059, 0.9137];     % Light blue
config.colors.Beginner = [0.8431, 0.2549, 0.6078];  % Magenta/pink
config.colors.Expert = [0, 0.6196, 0.4510];         % Teal/green
config.colors.NoSpout = [0.8353, 0.3686, 0];        % Orange

% -------------------------------------------------------------------------
% Ising Parameter Values (from grid search)
% -------------------------------------------------------------------------
config.isingParams.beta_values = [0.4,0.45,0.475, 0.5, 0.525,0.5,0.575,0.6,0.625, 0.65,0.7, 0.8];
config.isingParams.c_values = [1,2, 4, 6, 8];
config.isingParams.decay_const_values = [2, 4, 6, 8,10];
config.isingParams.inhibition_range_values = [1, 4, 9, 13];
config.isingParams.bias_values = [-1.2,-1, -0.8, -0.6, -0.4];

fprintf('\nIsing parameter grid:\n');
fprintf('  beta: %s\n', mat2str(config.isingParams.beta_values));
fprintf('  c: %s\n', mat2str(config.isingParams.c_values));
fprintf('  decay_const: %s\n', mat2str(config.isingParams.decay_const_values));
fprintf('  inhibition_range: %s\n', mat2str(config.isingParams.inhibition_range_values));
fprintf('  bias: %s\n', mat2str(config.isingParams.bias_values));

% -------------------------------------------------------------------------
% Analysis Parameters
% -------------------------------------------------------------------------
config.nTopMatches = 10;  % Number of top matches to report per condition

% Local/Debug mode: limit timesteps for faster processing
config.limitTimesteps = true;       % Set to false for full analysis
config.maxTimesteps = 2000;         % Max timesteps when limitTimesteps is true

if config.limitTimesteps
    fprintf('\nTimestep limit: %d (debug mode - for faster local processing)\n', config.maxTimesteps);
else
    fprintf('\nTimestep limit: OFF (full simulation)\n');
end

% Create output directory
if ~exist(config.outputPath, 'dir')
    mkdir(config.outputPath);
    fprintf('\nCreated output directory: %s\n', config.outputPath);
end

%% =========================================================================
%% CHECK FOR PRE-COMPUTED RESULTS (from Python cluster script)
%% =========================================================================
% If IsingComparison_Results.mat exists, skip data processing and
% jump directly to visualization.

precomputedPath = fullfile(config.outputPath, sprintf('IsingComparison_Results_%s_%s.mat', config.gridMode, config.matchingMetric));

if exist(precomputedPath, 'file')
    fprintf('\n');
    fprintf('=========================================================\n');
    fprintf('  FOUND PRE-COMPUTED RESULTS\n');
    fprintf('=========================================================\n');
    fprintf('Loading: %s\n', precomputedPath);

    precomputed = loadHDF5Struct(precomputedPath);

    % Extract Results structure (handle Python vs MATLAB format differences)
    if isfield(precomputed, 'IsingData')
        IsingData = precomputed.IsingData;
    end
    if isfield(precomputed, 'Comparison')
        Comparison = precomputed.Comparison;
    end
    if isfield(precomputed, 'ExpStats')
        ExpStats = precomputed.ExpStats;
    end
    if isfield(precomputed, 'config')
        loadedConfig = precomputed.config;
        if isfield(loadedConfig, 'ising_grid')
            config.isingGrid = loadedConfig.ising_grid;
        elseif isfield(loadedConfig, 'isingGrid')
            config.isingGrid = loadedConfig.isingGrid;
        end
    end

    % Validate that required variables were loaded
    if ~exist('Comparison', 'var') || ~exist('IsingData', 'var')
        fprintf('Pre-computed results incomplete or incompatible. Reprocessing...\n');
        fprintf('=========================================================\n\n');
        goto_visualization = false;
    else
        % Load experimental data for visualization (still needed for some plots)
        if exist(config.experimentalDataPath, 'file')
            expData = load(config.experimentalDataPath);
            BinarisedData_Exp = expData.BinarisedData;
            MoransI_Exp = expData.MoransI;
        end

        % Compute ParameterTrends from loaded Comparison data
        fprintf('Computing ParameterTrends from loaded data...\n');
        ParameterTrends = struct();
        param_names = {'beta', 'c', 'decay_const', 'inhibition_range', 'bias'};
        param_values_all = {config.isingParams.beta_values, config.isingParams.c_values, ...
            config.isingParams.decay_const_values, config.isingParams.inhibition_range_values, ...
            config.isingParams.bias_values};

        for c = 1:length(config.conditions)
            condition = config.conditions{c};
            if ~isfield(Comparison, condition)
                continue;
            end
            best_params = Comparison.(condition).bestMatch_params;
            ParameterTrends.(condition) = struct();
            for p = 1:length(param_names)
                param_name = param_names{p};
                param_values = param_values_all{p};
                best_values = best_params.(param_name);
                counts = zeros(1, length(param_values));
                for v = 1:length(param_values)
                    counts(v) = sum(best_values == param_values(v));
                end
                ParameterTrends.(condition).(param_name).values = param_values;
                ParameterTrends.(condition).(param_name).counts = counts;
                ParameterTrends.(condition).(param_name).mode = param_values(counts == max(counts));
                ParameterTrends.(condition).(param_name).mean = mean(best_values);
            end
        end

        % Compute TemporalScaleFactor from loaded data (if Autocorr_tau available)
        if isfield(IsingData, 'Autocorr_tau') && isfield(ExpStats, 'Global') && ...
           isfield(ExpStats.Global, 'Autocorr') && isfield(ExpStats.Global.Autocorr, 'tau')
            tau_global = ExpStats.Global.Autocorr.tau;
            if ~isnan(tau_global) && tau_global > 0
                nSims = length(IsingData.Autocorr_tau);
                IsingData.TemporalScaleFactor = zeros(nSims, 1);
                for s = 1:nSims
                    if ~isnan(IsingData.Autocorr_tau(s)) && IsingData.Autocorr_tau(s) > 0
                        IsingData.TemporalScaleFactor(s) = tau_global / IsingData.Autocorr_tau(s);
                    else
                        IsingData.TemporalScaleFactor(s) = NaN;
                    end
                end
                fprintf('Computed TemporalScaleFactor for %d simulations (global tau=%.2f)\n', nSims, tau_global);
            end
        end

        fprintf('Loaded pre-computed results\n');
        fprintf('Skipping data processing, jumping to visualization...\n');
        fprintf('=========================================================\n\n');

        goto_visualization = true;
    end
else
    fprintf('\nNo pre-computed results found at: %s\n', precomputedPath);
    fprintf('Will process simulation data from scratch.\n\n');
    goto_visualization = false;
end

%% =========================================================================
%% SECTION 2: Load Experimental Data
%% =========================================================================

if ~goto_visualization

fprintf('\n--- Section 2: Loading Experimental Data ---\n');

% Load pre-computed experimental data
if exist(config.experimentalDataPath, 'file')
    expData = load(config.experimentalDataPath);
    fprintf('Loaded experimental data from: %s\n', config.experimentalDataPath);

    % Extract relevant fields
    BinarisedData_Exp = expData.BinarisedData;
    MoransI_Exp = expData.MoransI;

    % Display summary
    fprintf('\nExperimental data summary:\n');
    for c = 1:length(config.conditions)
        condition = config.conditions{c};
        if isfield(MoransI_Exp, condition) && ~isempty(MoransI_Exp.(condition))
            moransI = MoransI_Exp.(condition);
            fprintf('  %s: %d trials, %d timepoints, mean I=%.4f\n', ...
                condition, size(moransI, 1), size(moransI, 2), mean(moransI(:), 'omitnan'));
        else
            fprintf('  %s: NO DATA\n', condition);
        end
    end
else
    error('Experimental data file not found: %s', config.experimentalDataPath);
end

%% =========================================================================
%% SECTION 3: Create Weight Matrices for Moran's I
%% =========================================================================

fprintf('\n--- Section 3: Creating Weight Matrices ---\n');

% Weight matrix for experimental grid (13x26)
valueMap_exp = rand(config.experimentalGrid(1), config.experimentalGrid(2));
distanceMat_exp = squareform(mL_distanceMat(valueMap_exp));
uniqueDistances_exp = unique(distanceMat_exp);
uniqueDistances_exp(uniqueDistances_exp == 0) = [];
currDistInds_exp = ismember(distanceMat_exp, uniqueDistances_exp(1));
weightMat_exp = zeros(size(distanceMat_exp));
weightMat_exp(currDistInds_exp) = distanceMat_exp(currDistInds_exp);
weightMat_exp(weightMat_exp == inf) = 0;

fprintf('Experimental weight matrix: %d x %d\n', size(weightMat_exp, 1), size(weightMat_exp, 2));

% Weight matrix for Ising grid (32x32)
valueMap_ising = rand(config.isingGrid(1), config.isingGrid(2));
distanceMat_ising = squareform(mL_distanceMat(valueMap_ising));
uniqueDistances_ising = unique(distanceMat_ising);
uniqueDistances_ising(uniqueDistances_ising == 0) = [];
currDistInds_ising = ismember(distanceMat_ising, uniqueDistances_ising(1));
weightMat_ising = zeros(size(distanceMat_ising));
weightMat_ising(currDistInds_ising) = distanceMat_ising(currDistInds_ising);
weightMat_ising(weightMat_ising == inf) = 0;

fprintf('Ising weight matrix: %d x %d\n', size(weightMat_ising, 1), size(weightMat_ising, 2));

% Select weight matrix based on grid mode
switch config.gridMode
    case {'subselect_centre', 'subselect_tiled', 'subselect_centre_vs_tiled'}
        weightMat_comparison = weightMat_exp;
        comparisonGridSize = config.experimentalGrid;
end

%% =========================================================================
%% SECTION 4: Load and Process Ising Simulations
%% =========================================================================

fprintf('\n--- Section 4: Loading Ising Simulations ---\n');

% Get list of all simulation files (new naming format: sim_be_*.mat)
simFiles = dir(fullfile(config.isingDataPath, 'sim_be_*.mat'));
nSimFiles = length(simFiles);
fprintf('Found %d simulation files\n', nSimFiles);

% Check for cached data
cacheValid = false;
if config.useCache && exist(config.cachePath, 'file')
    fprintf('Found cache file: %s\n', config.cachePath);
    cacheData = load(config.cachePath);
    % Validate cache matches current configuration
    if isfield(cacheData, 'cacheInfo') && ...
       strcmp(cacheData.cacheInfo.gridMode, config.gridMode) && ...
       cacheData.cacheInfo.nSimulations >= nSimFiles - 1  % Allow for missing sim
        IsingData = cacheData.IsingData;
        fprintf('Loaded cached IsingData (%d simulations, gridMode: %s)\n', ...
            length(IsingData.simIDs), config.gridMode);
        cacheValid = true;
    else
        fprintf('Cache invalid (config mismatch). Reprocessing...\n');
    end
end

% Pre-compute tiled positions (needed for both processing and analysis)
if ismember(config.gridMode, {'subselect_tiled', 'subselect_centre_vs_tiled'})
    tiled_positions = generateNonOverlappingPositions(config.isingGrid, config.experimentalGrid);
    nTiledPositions = size(tiled_positions, 1);
    if cacheValid
        fprintf('Tiled positions: %d non-overlapping grids\n', nTiledPositions);
    end
end

% BACKWARD COMPATIBILITY: Regenerate MoransI_pooled from cache if missing
if cacheValid && strcmp(config.gridMode, 'subselect_tiled') && ...
   ~isfield(IsingData, 'MoransI_pooled') && ...
   isfield(IsingData, 'MoransI_perPosition')
    fprintf('  Regenerating MoransI_pooled from cached per-position data...\n');
    nSims = length(IsingData.simIDs);
    IsingData.MoransI_pooled = cell(1, nSims);
    for s = 1:nSims
        moransI_pooled = [];
        for p = 1:nTiledPositions
            moransI_pooled = [moransI_pooled, IsingData.MoransI_perPosition{p}{s}];
        end
        IsingData.MoransI_pooled{s} = moransI_pooled;
    end
    fprintf('  Done. Regenerated pooled data for %d simulations.\n', nSims);
end

% Pre-compute centre crop indices (needed for centre_vs_tiled mode even with cache)
if strcmp(config.gridMode, 'subselect_centre_vs_tiled') && cacheValid
    centre_rowStart = floor((config.isingGrid(1) - config.experimentalGrid(1)) / 2) + 1;
    centre_colStart = floor((config.isingGrid(2) - config.experimentalGrid(2)) / 2) + 1;
    centre_rowIdx = centre_rowStart:(centre_rowStart + config.experimentalGrid(1) - 1);
    centre_colIdx = centre_colStart:(centre_colStart + config.experimentalGrid(2) - 1);
    fprintf('centre crop indices: rows %d-%d, cols %d-%d\n', ...
        centre_rowIdx(1), centre_rowIdx(end), centre_colIdx(1), centre_colIdx(end));
end

if ~cacheValid
% =========================================================================
% PARALLEL PROCESSING: Initialize parallel pool and pre-allocate arrays
% =========================================================================
fprintf('Processing simulations with PARALLEL workers...\n');

% Start parallel pool
nWorkers = min(16, feature('numcores'));
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', nWorkers);
    fprintf('Started parallel pool with %d workers\n', pool.NumWorkers);
else
    fprintf('Using existing parallel pool with %d workers\n', pool.NumWorkers);
end

% PRE-ALLOCATE all temporary arrays for parfor (required for proper slicing)
% Cell arrays for variable-length outputs
temp_simIDs = cell(nSimFiles, 1);
temp_simFilenames = cell(nSimFiles, 1);
temp_MoransI_all = cell(nSimFiles, 1);
temp_Activity_all = cell(nSimFiles, 1);

% Numeric arrays for scalar outputs
temp_params_beta = zeros(nSimFiles, 1);
temp_params_c = zeros(nSimFiles, 1);
temp_params_decay_const = zeros(nSimFiles, 1);
temp_params_inhibition_range = zeros(nSimFiles, 1);
temp_params_bias = zeros(nSimFiles, 1);
temp_MoransI_mean = zeros(nSimFiles, 1);
temp_MoransI_std = zeros(nSimFiles, 1);
temp_Activity_mean = zeros(nSimFiles, 1);
temp_Activity_std = zeros(nSimFiles, 1);

% Always compute temporal features (autocorr/blob) for all metrics
% This ensures BlobPersistence UMAP and related figures can always be generated
useTemporalFeatures = true;
if useTemporalFeatures
    temp_Autocorr_tau = zeros(nSimFiles, 1);
    temp_Autocorr_acf = cell(nSimFiles, 1);
    temp_Autocorr_fitR2 = zeros(nSimFiles, 1);
    temp_BlobStats_counts = cell(nSimFiles, 1);
    temp_BlobStats_sizes = cell(nSimFiles, 1);
    temp_BlobStats_mean_count = zeros(nSimFiles, 1);
    temp_BlobStats_mean_size = zeros(nSimFiles, 1);
    % Blob persistence tracking arrays
    temp_BlobPersistence_lifetimes = cell(nSimFiles, 1);
    temp_BlobPersistence_mean = zeros(nSimFiles, 1);
end

% Grid-mode specific arrays
useTiled = strcmp(config.gridMode, 'subselect_tiled');
useCentreVsTiled = strcmp(config.gridMode, 'subselect_centre_vs_tiled');
if useTiled
    temp_MoransI_perPosition = cell(nSimFiles, 1);
    temp_MoransI_pooled = cell(nSimFiles, 1);
end
if useCentreVsTiled
    temp_MoransI_centre = cell(nSimFiles, 1);
    temp_MoransI_tiled = cell(nSimFiles, 1);
end

% Pre-compute indices for subselect modes
if strcmp(config.gridMode, 'subselect_centre')
    % centre crop: single position
    subselect_rowStart = floor((config.isingGrid(1) - config.experimentalGrid(1)) / 2) + 1;
    subselect_colStart = floor((config.isingGrid(2) - config.experimentalGrid(2)) / 2) + 1;
    subselect_rowIdx = subselect_rowStart:(subselect_rowStart + config.experimentalGrid(1) - 1);
    subselect_colIdx = subselect_colStart:(subselect_colStart + config.experimentalGrid(2) - 1);
    fprintf('centre crop indices: rows %d-%d, cols %d-%d\n', ...
        subselect_rowIdx(1), subselect_rowIdx(end), subselect_colIdx(1), subselect_colIdx(end));
elseif strcmp(config.gridMode, 'subselect_tiled')
    % Tiled: multiple non-overlapping positions
    tiled_positions = generateNonOverlappingPositions(config.isingGrid, config.experimentalGrid);
    nTiledPositions = size(tiled_positions, 1);
    fprintf('Tiled positions: %d non-overlapping grids\n', nTiledPositions);
    for p = 1:nTiledPositions
        rowStart = tiled_positions(p, 1);
        colStart = tiled_positions(p, 2);
        rowEnd = rowStart + config.experimentalGrid(1) - 1;
        colEnd = colStart + config.experimentalGrid(2) - 1;
        fprintf('  Position %d: rows %d-%d, cols %d-%d\n', p, rowStart, rowEnd, colStart, colEnd);
    end

    % Pre-compute index ranges for each position
    tiled_rowIdx = cell(nTiledPositions, 1);
    tiled_colIdx = cell(nTiledPositions, 1);
    for p = 1:nTiledPositions
        tiled_rowIdx{p} = tiled_positions(p,1):(tiled_positions(p,1) + config.experimentalGrid(1) - 1);
        tiled_colIdx{p} = tiled_positions(p,2):(tiled_positions(p,2) + config.experimentalGrid(2) - 1);
    end

    % Per-position storage pre-allocated in temp arrays above
elseif strcmp(config.gridMode, 'subselect_centre_vs_tiled')
    % BOTH centre crop and tiled positions
    fprintf('centre vs Tiled comparison mode\n');

    % centre crop indices
    centre_rowStart = floor((config.isingGrid(1) - config.experimentalGrid(1)) / 2) + 1;
    centre_colStart = floor((config.isingGrid(2) - config.experimentalGrid(2)) / 2) + 1;
    centre_rowIdx = centre_rowStart:(centre_rowStart + config.experimentalGrid(1) - 1);
    centre_colIdx = centre_colStart:(centre_colStart + config.experimentalGrid(2) - 1);
    fprintf('  centre crop: rows %d-%d, cols %d-%d\n', ...
        centre_rowIdx(1), centre_rowIdx(end), centre_colIdx(1), centre_colIdx(end));

    % Tiled positions
    tiled_positions = generateNonOverlappingPositions(config.isingGrid, config.experimentalGrid);
    nTiledPositions = size(tiled_positions, 1);
    fprintf('  Tiled positions: %d non-overlapping grids\n', nTiledPositions);
    for p = 1:nTiledPositions
        rowStart = tiled_positions(p, 1);
        colStart = tiled_positions(p, 2);
        rowEnd = rowStart + config.experimentalGrid(1) - 1;
        colEnd = colStart + config.experimentalGrid(2) - 1;
        fprintf('    Position %d: rows %d-%d, cols %d-%d\n', p, rowStart, rowEnd, colStart, colEnd);
    end

    % Pre-compute index ranges for tiled positions
    tiled_rowIdx = cell(nTiledPositions, 1);
    tiled_colIdx = cell(nTiledPositions, 1);
    for p = 1:nTiledPositions
        tiled_rowIdx{p} = tiled_positions(p,1):(tiled_positions(p,1) + config.experimentalGrid(1) - 1);
        tiled_colIdx{p} = tiled_positions(p,2):(tiled_positions(p,2) + config.experimentalGrid(2) - 1);
    end

    % Centre/tiled storage pre-allocated in temp arrays above
end

% =========================================================================
% MAIN PARFOR LOOP: Process all simulations in parallel
% =========================================================================
fprintf('Starting parallel processing of %d simulations...\n', nSimFiles);
tic;
parfor i = 1:nSimFiles
    % Store original filename for later file loading
    filename = simFiles(i).name;

    % Load simulation data
    simPath = fullfile(config.isingDataPath, filename);
    simData = load(simPath);

    % Extract parameters
    params = simData.params;
    stored_spins = simData.stored_spins;  % [nFrames x rows x cols]

    % Apply timestep limit for local/debug mode
    if config.limitTimesteps && size(stored_spins, 1) > config.maxTimesteps
        if i == 1
            fprintf('  NOTE: Limiting to %d timesteps (from %d) for faster processing\n', ...
                config.maxTimesteps, size(stored_spins, 1));
        end
        stored_spins = stored_spins(1:config.maxTimesteps, :, :);
    end

    % Generate parameter-based simulation ID string
    simID = paramsToIdString(params.beta, params.c, params.decay_const, ...
        params.inhibition_range, params.bias);

    % Store simulation ID and filename (indexed assignment for parfor)
    temp_simIDs{i} = simID;
    temp_simFilenames{i} = filename;

    % Store parameters (indexed assignment for parfor)
    temp_params_beta(i) = params.beta;
    temp_params_c(i) = params.c;
    temp_params_decay_const(i) = params.decay_const;
    temp_params_inhibition_range(i) = params.inhibition_range;
    temp_params_bias(i) = params.bias;

    % Calculate Moran's I for each frame
    nFrames = size(stored_spins, 1);
    moransI_timeseries = zeros(1, nFrames);

    % For subselect_tiled: also store per-position time series
    if strcmp(config.gridMode, 'subselect_tiled')
        moransI_perPos = zeros(nTiledPositions, nFrames);
    end

    % For subselect_centre_vs_tiled: store centre and per-position tiled
    if strcmp(config.gridMode, 'subselect_centre_vs_tiled')
        moransI_centre_ts = zeros(1, nFrames);
        moransI_tiled_perPos = zeros(nTiledPositions, nFrames);  % Per-position for pooling
    end

    for t = 1:nFrames
        % Get frame data
        frame = squeeze(stored_spins(t, :, :));  % [32 x 32]

        % Apply grid resizing if needed
        switch config.gridMode
            case 'subselect_centre'
                % centre crop to experimental grid size (single position)
                frame_selected = frame(subselect_rowIdx, subselect_colIdx);  % [13 x 26]
                % Check for degenerate cases
                if all(frame_selected(:) == 0) || all(frame_selected(:) == 1)
                    moransI_timeseries(t) = NaN;
                else
                    moransI_timeseries(t) = mL_moransI(double(frame_selected), weightMat_exp);
                end
            case 'subselect_tiled'
                % Compute Moran's I for each tiled position
                moransI_positions = zeros(nTiledPositions, 1);
                for p = 1:nTiledPositions
                    frame_selected = frame(tiled_rowIdx{p}, tiled_colIdx{p});
                    if all(frame_selected(:) == 0) || all(frame_selected(:) == 1)
                        moransI_positions(p) = NaN;
                    else
                        moransI_positions(p) = mL_moransI(double(frame_selected), weightMat_exp);
                    end
                    moransI_perPos(p, t) = moransI_positions(p);
                end
                % Store P1 for main time series (for display/plotting purposes)
                % NOTE: For WD comparison, we will POOL all positions (concatenate)
                moransI_timeseries(t) = moransI_positions(1);

            case 'subselect_centre_vs_tiled'
                % --- centre CROP ---
                frame_centre = frame(centre_rowIdx, centre_colIdx);
                if all(frame_centre(:) == 0) || all(frame_centre(:) == 1)
                    moransI_centre_ts(t) = NaN;
                else
                    moransI_centre_ts(t) = mL_moransI(double(frame_centre), weightMat_exp);
                end

                % --- TILED POSITIONS ---
                moransI_positions = zeros(nTiledPositions, 1);
                for p = 1:nTiledPositions
                    frame_tiled = frame(tiled_rowIdx{p}, tiled_colIdx{p});
                    if all(frame_tiled(:) == 0) || all(frame_tiled(:) == 1)
                        moransI_positions(p) = NaN;
                    else
                        moransI_positions(p) = mL_moransI(double(frame_tiled), weightMat_exp);
                    end
                    moransI_tiled_perPos(p, t) = moransI_positions(p);
                end

                % Use centre as main time series (for default comparison)
                moransI_timeseries(t) = moransI_centre_ts(t);
        end
    end

    % Store Moran's I statistics and full time series (indexed for parfor)
    temp_MoransI_mean(i) = mean(moransI_timeseries, 'omitnan');
    temp_MoransI_std(i) = std(moransI_timeseries, 'omitnan');
    temp_MoransI_all{i} = moransI_timeseries;

    % For subselect_tiled: store per-position data AND pooled version (indexed for parfor)
    if useTiled
        % Store per-position data as struct for this simulation
        perPosData = cell(nTiledPositions, 1);
        for p = 1:nTiledPositions
            perPosData{p} = moransI_perPos(p, :);
        end
        temp_MoransI_perPosition{i} = perPosData;

        % POOLED: Concatenate all position time series
        moransI_pooled = [];
        for p = 1:nTiledPositions
            moransI_pooled = [moransI_pooled, moransI_perPos(p, :)];
        end
        temp_MoransI_pooled{i} = moransI_pooled;
    end

    % For subselect_centre_vs_tiled: store centre and POOLED tiled (indexed for parfor)
    if useCentreVsTiled
        temp_MoransI_centre{i} = moransI_centre_ts;
        % POOLED: Concatenate all tiled position time series
        moransI_tiled_pooled = [];
        for p = 1:nTiledPositions
            moransI_tiled_pooled = [moransI_tiled_pooled, moransI_tiled_perPos(p, :)];
        end
        temp_MoransI_tiled{i} = moransI_tiled_pooled;
    end

    % Compute Activity (sparsity) for each frame
    % Convert -1/+1 spins to 0/1 binary to match experimental data scale
    binary_spins = (stored_spins + 1) / 2;  % Maps -1→0, +1→1
    activity_timeseries = squeeze(mean(binary_spins, [2 3]));  % Fraction active per frame
    temp_Activity_mean(i) = mean(activity_timeseries, 'omitnan');
    temp_Activity_std(i) = std(activity_timeseries, 'omitnan');
    temp_Activity_all{i} = activity_timeseries(:)';  % Store as row vector

    % Compute autocorrelation if enabled (indexed for parfor)
    if useTemporalFeatures
        [tau_ising, acf_ising, ~, fitResult_ising] = computeAutocorrDecay(...
            activity_timeseries, config.autocorr.maxLag, config.autocorr.fitRange);
        temp_Autocorr_tau(i) = tau_ising;
        temp_Autocorr_acf{i} = acf_ising;
        if ~isempty(fitResult_ising) && isfield(fitResult_ising, 'R2')
            temp_Autocorr_fitR2(i) = fitResult_ising.R2;
        else
            temp_Autocorr_fitR2(i) = NaN;
        end
    end

    % Compute blob statistics for this Ising simulation (Step 2 matching)
    if useTemporalFeatures
        nTimesteps = size(stored_spins, 1);
        blob_counts = zeros(1, nTimesteps);
        blob_sizes_all = [];

        % Initialize blob persistence tracking
        prev_blobs = {};
        active_blobs = struct('id', {}, 'start_frame', {}, 'pixels', {});
        blob_lifetimes = [];
        next_blob_id = 1;
        iouThreshold = 0.3;  % 30% IoU threshold for matching

        for t = 1:nTimesteps
            % Get spatial frame
            frameData = squeeze(stored_spins(t, :, :));

            % Convert -1/+1 to 0/1
            frameData = (frameData + 1) / 2;

            % Apply Gaussian smoothing
            smoothed = imgaussfilt(double(frameData), config.blob_params.sigma);

            % Threshold
            binary = smoothed > config.blob_params.threshold;

            % Connected component analysis
            CC = bwconncomp(binary);

            % Filter by minimum size and get current frame blobs
            current_blobs = {};
            if CC.NumObjects > 0
                blobSizes = cellfun(@numel, CC.PixelIdxList);
                validBlobs = blobSizes >= config.blob_params.minBlobSize;
                blob_counts(t) = sum(validBlobs);
                if any(validBlobs)
                    blob_sizes_all = [blob_sizes_all, blobSizes(validBlobs)];
                    validIdx = find(validBlobs);
                    for vi = 1:length(validIdx)
                        current_blobs{end+1} = CC.PixelIdxList{validIdx(vi)};
                    end
                end
            else
                blob_counts(t) = 0;
            end

            % --- Blob persistence tracking (overlap-based) ---
            if isempty(prev_blobs)
                % First frame - initialize all blobs as new
                for bi = 1:length(current_blobs)
                    active_blobs(end+1).id = next_blob_id;
                    active_blobs(end).start_frame = t;
                    active_blobs(end).pixels = current_blobs{bi};
                    next_blob_id = next_blob_id + 1;
                end
            else
                % Match current blobs to previous blobs using overlap
                matched_prev = false(length(prev_blobs), 1);
                matched_curr = false(length(current_blobs), 1);

                for ci = 1:length(current_blobs)
                    best_iou = 0;
                    best_match = 0;

                    for pi = 1:length(prev_blobs)
                        if matched_prev(pi), continue; end

                        % Compute IoU (Intersection over Union)
                        intersection = length(intersect(current_blobs{ci}, prev_blobs{pi}));
                        union_size = length(union(current_blobs{ci}, prev_blobs{pi}));
                        iou = intersection / union_size;

                        if iou > iouThreshold && iou > best_iou
                            best_iou = iou;
                            best_match = pi;
                        end
                    end

                    if best_match > 0
                        % Matched - update active blob
                        matched_prev(best_match) = true;
                        matched_curr(ci) = true;

                        % Find and update active blob pixels
                        for ai = 1:length(active_blobs)
                            if isequal(active_blobs(ai).pixels, prev_blobs{best_match})
                                active_blobs(ai).pixels = current_blobs{ci};
                                break;
                            end
                        end
                    end
                end

                % Handle unmatched previous blobs (blob ended)
                for pi = 1:length(prev_blobs)
                    if ~matched_prev(pi)
                        for ai = length(active_blobs):-1:1
                            if isequal(active_blobs(ai).pixels, prev_blobs{pi})
                                lifetime = t - active_blobs(ai).start_frame;
                                if lifetime > 0
                                    blob_lifetimes(end+1) = lifetime;
                                end
                                active_blobs(ai) = [];
                                break;
                            end
                        end
                    end
                end

                % Handle unmatched current blobs (new blob)
                for ci = 1:length(current_blobs)
                    if ~matched_curr(ci)
                        active_blobs(end+1).id = next_blob_id;
                        active_blobs(end).start_frame = t;
                        active_blobs(end).pixels = current_blobs{ci};
                        next_blob_id = next_blob_id + 1;
                    end
                end
            end

            % Update prev_blobs for next iteration
            prev_blobs = current_blobs;
        end

        % Finalize any still-active blobs at end of simulation
        for ai = 1:length(active_blobs)
            lifetime = nTimesteps - active_blobs(ai).start_frame + 1;
            if lifetime > 0
                blob_lifetimes(end+1) = lifetime;
            end
        end

        temp_BlobStats_counts{i} = blob_counts;
        temp_BlobStats_sizes{i} = blob_sizes_all;
        temp_BlobStats_mean_count(i) = mean(blob_counts);
        if ~isempty(blob_sizes_all)
            temp_BlobStats_mean_size(i) = mean(blob_sizes_all);
        else
            temp_BlobStats_mean_size(i) = 0;
        end

        % Store blob persistence data
        temp_BlobPersistence_lifetimes{i} = blob_lifetimes(:);
        if ~isempty(blob_lifetimes)
            temp_BlobPersistence_mean(i) = mean(blob_lifetimes);
        else
            temp_BlobPersistence_mean(i) = 0;
        end
    end

    % Note: Progress updates removed for parfor (not supported)
end

elapsed_total = toc;
fprintf('Completed parallel processing of %d simulations in %.1f seconds\n', nSimFiles, elapsed_total);
fprintf('Average time per simulation: %.3f seconds\n', elapsed_total / nSimFiles);

% =========================================================================
% POST-PARFOR: Transfer results from temp arrays to IsingData struct
% =========================================================================
fprintf('Assembling results into IsingData structure...\n');

% Initialize IsingData structure
IsingData = struct();

% Transfer basic fields (convert to column vectors)
IsingData.simIDs = temp_simIDs(:);
IsingData.simFilenames = temp_simFilenames(:);
IsingData.MoransI_mean = temp_MoransI_mean(:);
IsingData.MoransI_std = temp_MoransI_std(:);
IsingData.MoransI_all = temp_MoransI_all(:);
IsingData.Activity_mean = temp_Activity_mean(:);
IsingData.Activity_std = temp_Activity_std(:);
IsingData.Activity_all = temp_Activity_all(:);

% Transfer parameters
IsingData.params = struct();
IsingData.params.beta = temp_params_beta(:);
IsingData.params.c = temp_params_c(:);
IsingData.params.decay_const = temp_params_decay_const(:);
IsingData.params.inhibition_range = temp_params_inhibition_range(:);
IsingData.params.bias = temp_params_bias(:);

% Transfer grid-mode specific fields
if useTiled
    % Reorganize per-position data: from {sim}{pos} to {pos}{sim}
    IsingData.MoransI_perPosition = cell(nTiledPositions, 1);
    for p = 1:nTiledPositions
        IsingData.MoransI_perPosition{p} = cell(nSimFiles, 1);
        for s = 1:nSimFiles
            if ~isempty(temp_MoransI_perPosition{s})
                IsingData.MoransI_perPosition{p}{s} = temp_MoransI_perPosition{s}{p};
            end
        end
    end
    IsingData.MoransI_pooled = temp_MoransI_pooled(:);
end

if useCentreVsTiled
    IsingData.MoransI_centre = temp_MoransI_centre(:);
    IsingData.MoransI_tiled = temp_MoransI_tiled(:);
end

% Transfer autocorrelation/blob fields if they were computed
if useTemporalFeatures
    IsingData.Autocorr_tau = temp_Autocorr_tau(:);
    IsingData.Autocorr_acf = temp_Autocorr_acf(:);
    IsingData.Autocorr_fitR2 = temp_Autocorr_fitR2(:);
    IsingData.BlobStats_counts = temp_BlobStats_counts(:);
    IsingData.BlobStats_sizes = temp_BlobStats_sizes(:);
    IsingData.BlobStats_mean_count = temp_BlobStats_mean_count(:);
    IsingData.BlobStats_mean_size = temp_BlobStats_mean_size(:);
    % Transfer blob persistence data
    IsingData.BlobPersistence_lifetimes = temp_BlobPersistence_lifetimes(:);
    IsingData.BlobPersistence_mean = temp_BlobPersistence_mean(:);
end

% Create parameter table for easy access
IsingData.paramTable = table(IsingData.simIDs, IsingData.params.beta, ...
    IsingData.params.c, IsingData.params.decay_const, ...
    IsingData.params.inhibition_range, IsingData.params.bias, ...
    IsingData.MoransI_mean, IsingData.MoransI_std, ...
    'VariableNames', {'simID', 'beta', 'c', 'decay_const', 'inhibition_range', ...
    'bias', 'MoransI_mean', 'MoransI_std'});

    % Save to cache
    if config.useCache
        cacheInfo = struct('gridMode', config.gridMode, ...
            'nSimulations', length(IsingData.simIDs), ...
            'timestamp', datetime('now'));
        cacheDir = fileparts(config.cachePath);
        if ~exist(cacheDir, 'dir')
            mkdir(cacheDir);
        end
        save(config.cachePath, 'IsingData', 'cacheInfo', '-v7.3');
        fprintf('Saved IsingData to cache: %s\n', config.cachePath);
    end
end  % end if ~cacheValid

fprintf('\nIsing data summary:\n');
fprintf('  Total simulations: %d\n', length(IsingData.simIDs));
fprintf('  Moran''s I range: [%.4f, %.4f]\n', ...
    min(IsingData.MoransI_mean), max(IsingData.MoransI_mean));
fprintf('  Mean Moran''s I: %.4f (std: %.4f)\n', ...
    mean(IsingData.MoransI_mean), std(IsingData.MoransI_mean));
if isfield(IsingData, 'Autocorr_tau')
    valid_tau = IsingData.Autocorr_tau(~isnan(IsingData.Autocorr_tau));
    fprintf('  Autocorr tau range: [%.2f, %.2f] MC sweeps\n', min(valid_tau), max(valid_tau));
    fprintf('  Mean Autocorr tau: %.2f (std: %.2f), valid fits: %d/%d\n', ...
        mean(valid_tau), std(valid_tau), length(valid_tau), length(IsingData.Autocorr_tau));
end

%% =========================================================================
%% SECTION 5: Compute Experimental Statistics
%% =========================================================================

fprintf('\n--- Section 5: Computing Experimental Statistics ---\n');

% Frame selection info
switch config.expFrameSelection
    case 'prestim'
        fprintf('Using pre-stimulus frames only (1-%d)\n', config.expPrestimFrames);
    case 'nostim'
        fprintf('Using no-stimulus frames: pre-stim (1-%d) + post-stim (%d-%d), split into sub-trials\n', ...
            config.expPrestimFrames, config.expStimOffsetFrame + 1, ...
            config.expStimOffsetFrame + config.expPrestimFrames);
    otherwise
        fprintf('Using all frames\n');
end

% Auto-reduce maxLag for short sub-trials (prestim/nostim produce 80-frame trials)
if ismember(config.expFrameSelection, {'prestim', 'nostim'})
    maxAllowed = floor(config.expPrestimFrames / 2) - 1;  % = 39 for 80 frames
    if config.autocorr.maxLag > maxAllowed
        fprintf('Auto-reducing autocorr.maxLag from %d to %d (sub-trial length = %d frames)\n', ...
            config.autocorr.maxLag, maxAllowed, config.expPrestimFrames);
        config.autocorr.maxLag = maxAllowed;
    end
    if config.autocorr.fitRange(2) > maxAllowed
        config.autocorr.fitRange(2) = maxAllowed;
        fprintf('Clamped autocorr.fitRange upper bound to %d\n', maxAllowed);
    end
end

% Initialize experimental statistics structure
ExpStats = struct();

for c = 1:length(config.conditions)
    condition = config.conditions{c};

    if ~isfield(MoransI_Exp, condition) || isempty(MoransI_Exp.(condition))
        fprintf('%s: No data available\n', condition);
        continue;
    end

    moransI = MoransI_Exp.(condition);  % [nTrials x nTimepoints]

    % Apply frame selection to MoransI
    if strcmp(config.expFrameSelection, 'prestim') && size(moransI, 2) > config.expPrestimFrames
        moransI = moransI(:, 1:config.expPrestimFrames);
    elseif strcmp(config.expFrameSelection, 'nostim') && size(moransI, 2) > config.expPrestimFrames
        moransI = applyNostimSelection(moransI, config, 'trials_x_time');
    end

    % Compute statistics
    ExpStats.(condition).MoransI_all = moransI(:);  % Flatten for distribution
    ExpStats.(condition).MoransI_mean = mean(moransI(:), 'omitnan');
    ExpStats.(condition).MoransI_std = std(moransI(:), 'omitnan');
    ExpStats.(condition).MoransI_median = median(moransI(:), 'omitnan');
    ExpStats.(condition).MoransI_timecourse = mean(moransI, 1, 'omitnan');  % Mean across trials
    ExpStats.(condition).MoransI_trials = moransI;  % [nTrials x nTimepoints] for trial-averaged ACF

    % Compute histogram for distribution comparison
    [ExpStats.(condition).hist_counts, ExpStats.(condition).hist_edges] = ...
        histcounts(moransI(:), 50, 'Normalization', 'probability');
    ExpStats.(condition).hist_centres = ...
        (ExpStats.(condition).hist_edges(1:end-1) + ExpStats.(condition).hist_edges(2:end)) / 2;

    % Compute Activity (sparsity) statistics from BinarisedData
    if isfield(BinarisedData_Exp, condition) && ~isempty(BinarisedData_Exp.(condition))
        % BinarisedData is [rows x cols x trials*timepoints] or [rows x cols x trials x timepoints]
        binData = BinarisedData_Exp.(condition);
        % Apply frame selection to BinarisedData
        if ndims(binData) == 4 && strcmp(config.expFrameSelection, 'prestim') && size(binData, 3) > config.expPrestimFrames
            binData = binData(:, :, 1:config.expPrestimFrames, :);
        elseif ndims(binData) == 4 && strcmp(config.expFrameSelection, 'nostim') && size(binData, 3) > config.expPrestimFrames
            binData = applyNostimSelection(binData, config, 'spatial_4d');
        elseif ndims(binData) == 3 && strcmp(config.expFrameSelection, 'prestim') && size(binData, 3) > config.expPrestimFrames
            binData = binData(:, :, 1:config.expPrestimFrames);
        elseif ndims(binData) == 3 && strcmp(config.expFrameSelection, 'nostim') && size(binData, 3) > config.expPrestimFrames
            binData = applyNostimSelection(binData, config, 'spatial_3d');
        end
        % Compute mean activity per frame (mean across spatial dimensions)
        if ndims(binData) == 4
            activity = squeeze(mean(binData, [1 2]));  % [trials x timepoints]
            ExpStats.(condition).Activity_all = activity(:);  % Flatten
            ExpStats.(condition).Activity_trials = activity;  % [nTrials x nTimepoints] for trial-averaged ACF
        else
            activity = squeeze(mean(binData, [1 2]));  % [frames]
            ExpStats.(condition).Activity_all = activity(:);
            % No trial structure for 3D data, cannot compute trial-averaged ACF
        end
        ExpStats.(condition).Activity_mean = mean(ExpStats.(condition).Activity_all, 'omitnan');
        ExpStats.(condition).Activity_std = std(ExpStats.(condition).Activity_all, 'omitnan');
    end

    % Compute per-condition autocorrelation (for Step 1 temporal rescaling reference)
    if ismember(config.matchingMetric, {'autocorr', 'blobCount', 'blobPersistence', 'combined', 'spatial+persistence'})
        % Trial-averaged autocorrelation of activity time series
        if isfield(ExpStats.(condition), 'Activity_trials') && ~isempty(ExpStats.(condition).Activity_trials)
            activity_trials = ExpStats.(condition).Activity_trials;
            [tau_exp, acf_exp, lags_exp, fitResult_exp] = computeTrialAveragedAutocorr(...
                activity_trials, config.autocorr.maxLag, config.autocorr.fitRange);
            ExpStats.(condition).Autocorr.tau = tau_exp;
            ExpStats.(condition).Autocorr.acf = acf_exp;
            ExpStats.(condition).Autocorr.lags = lags_exp;
            ExpStats.(condition).Autocorr.fitResult = fitResult_exp;
            ExpStats.(condition).Autocorr.source = 'decay_constant_trial_avg';
            fprintf('  %s autocorr tau: %.2f frames (R²=%.3f, %d/%d trials)\n', ...
                condition, tau_exp, fitResult_exp.R2, fitResult_exp.validTrials, fitResult_exp.totalTrials);
        end
    end

    fprintf('%s: mean=%.4f, std=%.4f, median=%.4f, n=%d\n', ...
        condition, ExpStats.(condition).MoransI_mean, ExpStats.(condition).MoransI_std, ...
        ExpStats.(condition).MoransI_median, numel(moransI));
end

%% =========================================================================
%% SECTION 5b: Compute Global Experimental ACF for Temporal Rescaling (Step 1)
%% =========================================================================
% This section computes a GLOBAL experimental ACF by averaging across ALL conditions.
% This is used to establish the time scale for matching Ising simulations to real time.

if ismember(config.matchingMetric, {'autocorr', 'blobCount', 'blobPersistence', 'combined', 'spatial+persistence'})
    fprintf('\n--- Computing Global Experimental ACF for Temporal Rescaling ---\n');

    % First pass: find minimum timepoints across conditions
    minTimepoints = inf;
    for c = 1:length(config.conditions)
        condition = config.conditions{c};
        if isfield(ExpStats, condition) && isfield(ExpStats.(condition), 'Activity_trials')
            nTimepoints = size(ExpStats.(condition).Activity_trials, 2);
            minTimepoints = min(minTimepoints, nTimepoints);
        end
    end

    % Second pass: concatenate trials truncated to common length
    all_activity_trials = [];
    for c = 1:length(config.conditions)
        condition = config.conditions{c};
        if isfield(ExpStats, condition) && isfield(ExpStats.(condition), 'Activity_trials')
            trials = ExpStats.(condition).Activity_trials(:, 1:minTimepoints);
            all_activity_trials = [all_activity_trials; trials];
        end
    end

    if ~isempty(all_activity_trials)
        [tau_global, acf_global, lags_global, fitResult_global] = computeTrialAveragedAutocorr(...
            all_activity_trials, config.autocorr.maxLag, config.autocorr.fitRange);
        ExpStats.Global.Autocorr.tau = tau_global;
        ExpStats.Global.Autocorr.acf = acf_global;
        ExpStats.Global.Autocorr.lags = lags_global;
        ExpStats.Global.Autocorr.fitResult = fitResult_global;
        fprintf('Global experimental tau: %.2f frames (R²=%.3f, %d trials total)\n', ...
            tau_global, fitResult_global.R2, size(all_activity_trials, 1));
    else
        warning('No activity trial data available for global ACF computation');
        ExpStats.Global.Autocorr.tau = NaN;
    end
end

%% =========================================================================
%% SECTION 5c: Compute Global Experimental Blob Statistics (Step 2)
%% =========================================================================
% This section detects blobs in experimental data using the matched filter parameters.
% Blob count/size distributions will be compared with Ising simulations.

% Always compute blob statistics for all metrics (enables BlobPersistence UMAP)
fprintf('\n--- Computing Experimental Blob Statistics ---\n');

    % Detect blobs across all conditions and compute global statistics
    ExpStats.Global.BlobStats = struct();
    ExpStats.Global.BlobPersistence = struct();
    all_blob_counts = [];
    all_blob_sizes = [];
    all_blob_lifetimes = [];
    iouThreshold = 0.3;  % 30% IoU threshold for matching

    for c = 1:length(config.conditions)
        condition = config.conditions{c};
        if ~isfield(BinarisedData_Exp, condition)
            continue;
        end

        % Get binarised data for this condition
        % Data is [gridY × gridX × nTimepoints × nTrials]
        binData = BinarisedData_Exp.(condition);
        % Apply frame selection
        if strcmp(config.expFrameSelection, 'prestim') && size(binData, 3) > config.expPrestimFrames
            binData = binData(:, :, 1:config.expPrestimFrames, :);
        elseif strcmp(config.expFrameSelection, 'nostim') && size(binData, 3) > config.expPrestimFrames
            binData = applyNostimSelection(binData, config, 'spatial_4d');
        end
        nTrials = size(binData, 4);
        nFrames = size(binData, 3);

        for trial = 1:nTrials
            % Initialize blob persistence tracking for this trial
            prev_blobs = {};
            active_blobs = struct('id', {}, 'start_frame', {}, 'pixels', {});
            trial_lifetimes = [];
            next_blob_id = 1;

            for frame = 1:nFrames
                % Get spatial frame (reshape to grid)
                frameData = squeeze(binData(:, :, frame, trial));

                % Apply Gaussian smoothing
                smoothed = imgaussfilt(double(frameData), config.blob_params.sigma);

                % Threshold
                binary = smoothed > config.blob_params.threshold;

                % Connected component analysis
                CC = bwconncomp(binary);

                % Filter by minimum size and get current frame blobs
                current_blobs = {};
                if CC.NumObjects > 0
                    blobSizes = cellfun(@numel, CC.PixelIdxList);
                    validBlobs = blobSizes >= config.blob_params.minBlobSize;
                    all_blob_counts(end+1) = sum(validBlobs);
                    if any(validBlobs)
                        all_blob_sizes = [all_blob_sizes, blobSizes(validBlobs)];
                        validIdx = find(validBlobs);
                        for vi = 1:length(validIdx)
                            current_blobs{end+1} = CC.PixelIdxList{validIdx(vi)};
                        end
                    end
                else
                    all_blob_counts(end+1) = 0;
                end

                % --- Blob persistence tracking (overlap-based) ---
                if isempty(prev_blobs)
                    % First frame - initialize all blobs as new
                    for bi = 1:length(current_blobs)
                        active_blobs(end+1).id = next_blob_id;
                        active_blobs(end).start_frame = frame;
                        active_blobs(end).pixels = current_blobs{bi};
                        next_blob_id = next_blob_id + 1;
                    end
                else
                    % Match current blobs to previous blobs using overlap
                    matched_prev = false(length(prev_blobs), 1);
                    matched_curr = false(length(current_blobs), 1);

                    for ci = 1:length(current_blobs)
                        best_iou = 0;
                        best_match = 0;

                        for pi = 1:length(prev_blobs)
                            if matched_prev(pi), continue; end

                            % Compute IoU (Intersection over Union)
                            intersection = length(intersect(current_blobs{ci}, prev_blobs{pi}));
                            union_size = length(union(current_blobs{ci}, prev_blobs{pi}));
                            iou = intersection / union_size;

                            if iou > iouThreshold && iou > best_iou
                                best_iou = iou;
                                best_match = pi;
                            end
                        end

                        if best_match > 0
                            % Matched - update active blob
                            matched_prev(best_match) = true;
                            matched_curr(ci) = true;

                            % Find and update active blob pixels
                            for ai = 1:length(active_blobs)
                                if isequal(active_blobs(ai).pixels, prev_blobs{best_match})
                                    active_blobs(ai).pixels = current_blobs{ci};
                                    break;
                                end
                            end
                        end
                    end

                    % Handle unmatched previous blobs (blob ended)
                    for pi = 1:length(prev_blobs)
                        if ~matched_prev(pi)
                            for ai = length(active_blobs):-1:1
                                if isequal(active_blobs(ai).pixels, prev_blobs{pi})
                                    lifetime = frame - active_blobs(ai).start_frame;
                                    if lifetime > 0
                                        trial_lifetimes(end+1) = lifetime;
                                    end
                                    active_blobs(ai) = [];
                                    break;
                                end
                            end
                        end
                    end

                    % Handle unmatched current blobs (new blob)
                    for ci = 1:length(current_blobs)
                        if ~matched_curr(ci)
                            active_blobs(end+1).id = next_blob_id;
                            active_blobs(end).start_frame = frame;
                            active_blobs(end).pixels = current_blobs{ci};
                            next_blob_id = next_blob_id + 1;
                        end
                    end
                end

                % Update prev_blobs for next iteration
                prev_blobs = current_blobs;
            end

            % Finalize any still-active blobs at end of trial
            for ai = 1:length(active_blobs)
                lifetime = nFrames - active_blobs(ai).start_frame + 1;
                if lifetime > 0
                    trial_lifetimes(end+1) = lifetime;
                end
            end

            % Accumulate trial lifetimes
            all_blob_lifetimes = [all_blob_lifetimes, trial_lifetimes];
        end
    end

    ExpStats.Global.BlobStats.counts = all_blob_counts;
    ExpStats.Global.BlobStats.sizes = all_blob_sizes;
    ExpStats.Global.BlobStats.mean_count = mean(all_blob_counts);
    ExpStats.Global.BlobStats.std_count = std(all_blob_counts);
    if ~isempty(all_blob_sizes)
        ExpStats.Global.BlobStats.mean_size = mean(all_blob_sizes);
    else
        ExpStats.Global.BlobStats.mean_size = 0;
    end

    % Store blob persistence statistics
    ExpStats.Global.BlobPersistence.lifetimes = all_blob_lifetimes(:);
    if ~isempty(all_blob_lifetimes)
        ExpStats.Global.BlobPersistence.mean_lifetime = mean(all_blob_lifetimes);
        ExpStats.Global.BlobPersistence.std_lifetime = std(all_blob_lifetimes);
    else
        ExpStats.Global.BlobPersistence.mean_lifetime = 0;
        ExpStats.Global.BlobPersistence.std_lifetime = 0;
    end

fprintf('Global experimental blobs: mean count=%.2f (std=%.2f), mean size=%.2f pixels, n=%d frames\n', ...
    ExpStats.Global.BlobStats.mean_count, ExpStats.Global.BlobStats.std_count, ...
    ExpStats.Global.BlobStats.mean_size, length(all_blob_counts));
fprintf('Global experimental blob persistence: mean lifetime=%.2f (std=%.2f) frames, n=%d blobs\n', ...
    ExpStats.Global.BlobPersistence.mean_lifetime, ExpStats.Global.BlobPersistence.std_lifetime, ...
    length(all_blob_lifetimes));

%% =========================================================================
%% SECTION 6: Distribution Comparison (Wasserstein Distance)
%% =========================================================================

fprintf('\n--- Section 6: Distribution Comparison ---\n');

% Initialize comparison results
Comparison = struct();

for c = 1:length(config.conditions)
    condition = config.conditions{c};

    if ~isfield(ExpStats, condition)
        continue;
    end

    fprintf('\nComparing %s to Ising simulations (metric: %s)...\n', condition, config.matchingMetric);

    % Calculate distances based on selected matching metric
    nSims = length(IsingData.simIDs);
    wasserstein_dist = zeros(nSims, 1);
    mean_diff = zeros(nSims, 1);

    % Clear raw distance variables from previous iteration
    clear dist_moransI_raw dist_activity_raw dist_blob_raw dist_autocorr_raw

    switch config.matchingMetric
        case 'moransI'
            % Match based on Moran's I distribution (original behavior)
            exp_moransI = ExpStats.(condition).MoransI_all;
            exp_moransI_mean = ExpStats.(condition).MoransI_mean;
            parfor s = 1:nSims
                ising_values = IsingData.MoransI_all{s}(:);
                wasserstein_dist(s) = wasserstein_1d(exp_moransI, ising_values);
                mean_diff(s) = abs(IsingData.MoransI_mean(s) - exp_moransI_mean);
            end
            % Store raw Moran's I distance
            dist_moransI_raw = wasserstein_dist;

        case 'activity'
            % Match based on Activity (sparsity) distribution
            if ~isfield(ExpStats.(condition), 'Activity_all')
                warning('No Activity data for %s. Falling back to moransI.', condition);
                exp_moransI = ExpStats.(condition).MoransI_all;
                exp_moransI_mean = ExpStats.(condition).MoransI_mean;
                parfor s = 1:nSims
                    ising_values = IsingData.MoransI_all{s}(:);
                    wasserstein_dist(s) = wasserstein_1d(exp_moransI, ising_values);
                    mean_diff(s) = abs(IsingData.MoransI_mean(s) - exp_moransI_mean);
                end
                % Store raw Moran's I distance (fallback)
                dist_moransI_raw = wasserstein_dist;
            else
                exp_activity = ExpStats.(condition).Activity_all;
                exp_activity_mean = ExpStats.(condition).Activity_mean;
                parfor s = 1:nSims
                    ising_activity = IsingData.Activity_all{s}(:);
                    wasserstein_dist(s) = wasserstein_1d(exp_activity, ising_activity);
                    mean_diff(s) = abs(IsingData.Activity_mean(s) - exp_activity_mean);
                end
                % Store raw Activity distance
                dist_activity_raw = wasserstein_dist;
            end

        case 'autocorr'
            % Match based on autocorrelation time constant only
            if ~isfield(ExpStats.(condition), 'Autocorr') || ~isfield(IsingData, 'Autocorr_tau')
                warning('No autocorrelation data for %s. Falling back to moransI.', condition);
                exp_moransI = ExpStats.(condition).MoransI_all;
                exp_moransI_mean = ExpStats.(condition).MoransI_mean;
                parfor s = 1:nSims
                    ising_values = IsingData.MoransI_all{s}(:);
                    wasserstein_dist(s) = wasserstein_1d(exp_moransI, ising_values);
                    mean_diff(s) = abs(IsingData.MoransI_mean(s) - exp_moransI_mean);
                end
                % Store raw Moran's I distance (fallback)
                dist_moransI_raw = wasserstein_dist;
            else
                exp_tau = ExpStats.(condition).Autocorr.tau;
                if isnan(exp_tau) || exp_tau <= 0
                    warning('Invalid experimental tau for %s. Falling back to moransI.', condition);
                    exp_moransI = ExpStats.(condition).MoransI_all;
                    exp_moransI_mean = ExpStats.(condition).MoransI_mean;
                    parfor s = 1:nSims
                        ising_values = IsingData.MoransI_all{s}(:);
                        wasserstein_dist(s) = wasserstein_1d(exp_moransI, ising_values);
                        mean_diff(s) = abs(IsingData.MoransI_mean(s) - exp_moransI_mean);
                    end
                    % Store raw Moran's I distance (fallback)
                    dist_moransI_raw = wasserstein_dist;
                else
                    exp_moransI_mean = ExpStats.(condition).MoransI_mean;
                    Autocorr_tau = IsingData.Autocorr_tau;
                    MoransI_mean = IsingData.MoransI_mean;
                    parfor s = 1:nSims
                        ising_tau = Autocorr_tau(s);
                        if ~isnan(ising_tau) && ising_tau > 0
                            % Distance based on log-ratio of time constants
                            wasserstein_dist(s) = abs(log(ising_tau / exp_tau));
                        else
                            wasserstein_dist(s) = NaN;
                        end
                        mean_diff(s) = abs(MoransI_mean(s) - exp_moransI_mean);
                    end
                    % Store raw autocorrelation distance (log-ratio)
                    dist_autocorr_raw = wasserstein_dist;
                end
            end

        case 'blobCount'
            % Match based on blob count distribution with temporal rescaling
            % Step 1: Temporal rescaling using autocorrelation tau
            % Step 2: Wasserstein distance between blob count distributions
            if ~isfield(ExpStats, 'Global') || ~isfield(ExpStats.Global, 'BlobStats') || ...
               ~isfield(IsingData, 'BlobStats_counts') || ~isfield(IsingData, 'Autocorr_tau')
                warning('Missing blob/autocorr data for %s. Falling back to moransI.', condition);
                exp_moransI = ExpStats.(condition).MoransI_all;
                exp_moransI_mean = ExpStats.(condition).MoransI_mean;
                parfor s = 1:nSims
                    ising_values = IsingData.MoransI_all{s}(:);
                    wasserstein_dist(s) = wasserstein_1d(exp_moransI, ising_values);
                    mean_diff(s) = abs(IsingData.MoransI_mean(s) - exp_moransI_mean);
                end
                % Store raw Moran's I distance (fallback)
                dist_moransI_raw = wasserstein_dist;
            else
                % Get global experimental blob counts and tau
                exp_blob_counts = ExpStats.Global.BlobStats.counts(:);
                tau_global = ExpStats.Global.Autocorr.tau;

                if isnan(tau_global) || tau_global <= 0
                    warning('Invalid global tau for %s. Falling back to moransI.', condition);
                    exp_moransI = ExpStats.(condition).MoransI_all;
                    exp_moransI_mean = ExpStats.(condition).MoransI_mean;
                    parfor s = 1:nSims
                        ising_values = IsingData.MoransI_all{s}(:);
                        wasserstein_dist(s) = wasserstein_1d(exp_moransI, ising_values);
                        mean_diff(s) = abs(IsingData.MoransI_mean(s) - exp_moransI_mean);
                    end
                    % Store raw Moran's I distance (fallback)
                    dist_moransI_raw = wasserstein_dist;
                else
                    exp_moransI_mean = ExpStats.(condition).MoransI_mean;
                    Autocorr_tau = IsingData.Autocorr_tau;
                    BlobStats_counts = IsingData.BlobStats_counts;
                    MoransI_mean = IsingData.MoransI_mean;

                    dist_blob_all = zeros(nSims, 1);
                    parfor s = 1:nSims
                        if ~isnan(Autocorr_tau(s)) && Autocorr_tau(s) > 0
                            ising_blob_counts = BlobStats_counts{s}(:);
                            dist_blob_all(s) = wasserstein_1d(exp_blob_counts, ising_blob_counts);
                        else
                            dist_blob_all(s) = NaN;
                        end
                        mean_diff(s) = abs(MoransI_mean(s) - exp_moransI_mean);
                    end

                    wasserstein_dist = dist_blob_all;

                    % Store raw blob distance
                    dist_blob_raw = wasserstein_dist;

                    fprintf('  Using blobCount metric (temporal rescaling + blob matching)\n');
                end
            end

        case 'blobPersistence'
            % Match based on blob persistence (lifetime) distribution with temporal rescaling
            % Step 1: Temporal rescaling using autocorrelation tau
            % Step 2: Wasserstein distance between blob lifetime distributions
            if ~isfield(ExpStats, 'Global') || ~isfield(ExpStats.Global, 'BlobPersistence') || ...
               ~isfield(IsingData, 'BlobPersistence_lifetimes') || ~isfield(IsingData, 'Autocorr_tau')
                warning('Missing blob persistence/autocorr data for %s. Falling back to moransI.', condition);
                exp_moransI = ExpStats.(condition).MoransI_all;
                exp_moransI_mean = ExpStats.(condition).MoransI_mean;
                parfor s = 1:nSims
                    ising_values = IsingData.MoransI_all{s}(:);
                    wasserstein_dist(s) = wasserstein_1d(exp_moransI, ising_values);
                    mean_diff(s) = abs(IsingData.MoransI_mean(s) - exp_moransI_mean);
                end
                % Store raw Moran's I distance (fallback)
                dist_moransI_raw = wasserstein_dist;
            else
                % Get global experimental blob lifetimes and tau
                exp_lifetimes = ExpStats.Global.BlobPersistence.lifetimes(:);
                tau_global = ExpStats.Global.Autocorr.tau;

                if isnan(tau_global) || tau_global <= 0
                    warning('Invalid global tau for %s. Falling back to moransI.', condition);
                    exp_moransI = ExpStats.(condition).MoransI_all;
                    exp_moransI_mean = ExpStats.(condition).MoransI_mean;
                    parfor s = 1:nSims
                        ising_values = IsingData.MoransI_all{s}(:);
                        wasserstein_dist(s) = wasserstein_1d(exp_moransI, ising_values);
                        mean_diff(s) = abs(IsingData.MoransI_mean(s) - exp_moransI_mean);
                    end
                    % Store raw Moran's I distance (fallback)
                    dist_moransI_raw = wasserstein_dist;
                else
                    exp_moransI_mean = ExpStats.(condition).MoransI_mean;
                    Autocorr_tau = IsingData.Autocorr_tau;
                    BlobPersistence_lifetimes = IsingData.BlobPersistence_lifetimes;
                    MoransI_mean = IsingData.MoransI_mean;

                    dist_persistence_all = zeros(nSims, 1);
                    parfor s = 1:nSims
                        if ~isnan(Autocorr_tau(s)) && Autocorr_tau(s) > 0
                            % Rescale Ising lifetimes to real time (10 Hz)
                            scale_factor = tau_global / Autocorr_tau(s);
                            ising_lifetimes = BlobPersistence_lifetimes{s};
                            if ~isempty(ising_lifetimes)
                                ising_lifetimes_rescaled = ising_lifetimes * scale_factor;
                                dist_persistence_all(s) = wasserstein_1d(exp_lifetimes, ising_lifetimes_rescaled);
                            else
                                dist_persistence_all(s) = NaN;
                            end
                        else
                            dist_persistence_all(s) = NaN;
                        end
                        mean_diff(s) = abs(MoransI_mean(s) - exp_moransI_mean);
                    end

                    wasserstein_dist = dist_persistence_all;

                    % Store raw persistence distance
                    dist_persistence_raw = wasserstein_dist;

                    fprintf('  Using blobPersistence metric (temporal rescaling + lifetime matching)\n');
                end
            end

        case 'moransI+activity'
            % Weighted combination of Moran's I and Activity only (no autocorr/blob)
            if ~isfield(ExpStats.(condition), 'Activity_all')
                warning('No Activity data for %s. Using moransI only.', condition);
                exp_moransI = ExpStats.(condition).MoransI_all;
                exp_moransI_mean = ExpStats.(condition).MoransI_mean;
                parfor s = 1:nSims
                    ising_values = IsingData.MoransI_all{s}(:);
                    wasserstein_dist(s) = wasserstein_1d(exp_moransI, ising_values);
                    mean_diff(s) = abs(IsingData.MoransI_mean(s) - exp_moransI_mean);
                end
                dist_moransI_raw = wasserstein_dist;
            else
                % Compute individual distances
                exp_moransI = ExpStats.(condition).MoransI_all;
                exp_activity = ExpStats.(condition).Activity_all;

                dist_moransI_all = zeros(nSims, 1);
                dist_activity_all = zeros(nSims, 1);

                parfor s = 1:nSims
                    dist_moransI_all(s) = wasserstein_1d(exp_moransI, IsingData.MoransI_all{s}(:));
                    dist_activity_all(s) = wasserstein_1d(exp_activity, IsingData.Activity_all{s}(:));
                end

                % Normalize distances (z-score)
                dist_moransI_norm = (dist_moransI_all - nanmean(dist_moransI_all)) / nanstd(dist_moransI_all);
                dist_activity_norm = (dist_activity_all - nanmean(dist_activity_all)) / nanstd(dist_activity_all);
                dist_moransI_norm(isnan(dist_moransI_norm)) = 0;
                dist_activity_norm(isnan(dist_activity_norm)) = 0;

                % Weighted combination (renormalize weights to exclude autocorr)
                total_weight = config.matchingWeights.moransI + config.matchingWeights.activity;
                w_moransI = config.matchingWeights.moransI / total_weight;
                w_activity = config.matchingWeights.activity / total_weight;

                wasserstein_dist = w_moransI * dist_moransI_norm + w_activity * dist_activity_norm;

                fprintf('  Using moransI+activity metric (weights: M=%.2f, A=%.2f)\n', w_moransI, w_activity);

                % Mean difference
                for s = 1:nSims
                    mean_diff(s) = abs(IsingData.MoransI_mean(s) - ExpStats.(condition).MoransI_mean);
                end

                % Store raw distances
                dist_moransI_raw = dist_moransI_all;
                dist_activity_raw = dist_activity_all;
            end

        case 'spatial+persistence'
            % Weighted combination: Moran's I + Activity + Blob Persistence (NO blob count)
            if ~isfield(ExpStats.(condition), 'Activity_all')
                warning('No Activity data for %s. Using moransI only.', condition);
                exp_moransI = ExpStats.(condition).MoransI_all;
                exp_moransI_mean = ExpStats.(condition).MoransI_mean;
                parfor s = 1:nSims
                    ising_values = IsingData.MoransI_all{s}(:);
                    wasserstein_dist(s) = wasserstein_1d(exp_moransI, ising_values);
                    mean_diff(s) = abs(IsingData.MoransI_mean(s) - exp_moransI_mean);
                end
                dist_moransI_raw = wasserstein_dist;
            else
                % Compute Moran's I and Activity distances
                exp_moransI = ExpStats.(condition).MoransI_all;
                exp_activity = ExpStats.(condition).Activity_all;

                dist_moransI_all = zeros(nSims, 1);
                dist_activity_all = zeros(nSims, 1);

                parfor s = 1:nSims
                    dist_moransI_all(s) = wasserstein_1d(exp_moransI, IsingData.MoransI_all{s}(:));
                    dist_activity_all(s) = wasserstein_1d(exp_activity, IsingData.Activity_all{s}(:));
                end

                % Normalize distances (z-score)
                dist_moransI_norm = (dist_moransI_all - nanmean(dist_moransI_all)) / nanstd(dist_moransI_all);
                dist_activity_norm = (dist_activity_all - nanmean(dist_activity_all)) / nanstd(dist_activity_all);
                dist_moransI_norm(isnan(dist_moransI_norm)) = 0;
                dist_activity_norm(isnan(dist_activity_norm)) = 0;

                % Compute blob persistence distances (with temporal rescaling)
                dist_persistence_norm = zeros(nSims, 1);
                usePersistence = isfield(ExpStats, 'Global') && ...
                                 isfield(ExpStats.Global, 'BlobPersistence') && ...
                                 isfield(ExpStats.Global, 'Autocorr') && ...
                                 isfield(IsingData, 'BlobPersistence_lifetimes') && ...
                                 isfield(IsingData, 'Autocorr_tau');

                if usePersistence
                    tau_global = ExpStats.Global.Autocorr.tau;
                    if ~isnan(tau_global) && tau_global > 0
                        % Compute temporal scale factors if not already done
                        if ~isfield(IsingData, 'TemporalScaleFactor') || isempty(IsingData.TemporalScaleFactor)
                            IsingData.TemporalScaleFactor = zeros(nSims, 1);
                            for s = 1:nSims
                                if ~isnan(IsingData.Autocorr_tau(s)) && IsingData.Autocorr_tau(s) > 0
                                    IsingData.TemporalScaleFactor(s) = tau_global / IsingData.Autocorr_tau(s);
                                else
                                    IsingData.TemporalScaleFactor(s) = NaN;
                                end
                            end
                            fprintf('  Step 1: Computed temporal scale factors (global tau=%.2f frames)\n', tau_global);
                        end

                        exp_lifetimes = ExpStats.Global.BlobPersistence.lifetimes(:);
                        dist_persistence_all = zeros(nSims, 1);

                        for s = 1:nSims
                            if ~isnan(IsingData.TemporalScaleFactor(s))
                                ising_lifetimes = IsingData.BlobPersistence_lifetimes{s};
                                if ~isempty(ising_lifetimes)
                                    ising_lifetimes_rescaled = ising_lifetimes * IsingData.TemporalScaleFactor(s);
                                    dist_persistence_all(s) = wasserstein_1d(exp_lifetimes, ising_lifetimes_rescaled);
                                else
                                    dist_persistence_all(s) = NaN;
                                end
                            else
                                dist_persistence_all(s) = NaN;
                            end
                        end

                        dist_persistence_norm = (dist_persistence_all - nanmean(dist_persistence_all)) / nanstd(dist_persistence_all);
                        dist_persistence_norm(isnan(dist_persistence_norm)) = 0;
                        fprintf('  Computed blob persistence distances (exp: %.1f±%.1f frames)\n', ...
                            ExpStats.Global.BlobPersistence.mean_lifetime, ExpStats.Global.BlobPersistence.std_lifetime);
                    end
                end

                % Weighted combination: 3 metrics (skip blobCount)
                if usePersistence && any(dist_persistence_norm ~= 0)
                    total_weight = config.matchingWeights.moransI + ...
                                   config.matchingWeights.activity + ...
                                   config.matchingWeights.blobPersistence;
                    w_moransI = config.matchingWeights.moransI / total_weight;
                    w_activity = config.matchingWeights.activity / total_weight;
                    w_persistence = config.matchingWeights.blobPersistence / total_weight;

                    wasserstein_dist = w_moransI * dist_moransI_norm + ...
                                       w_activity * dist_activity_norm + ...
                                       w_persistence * dist_persistence_norm;

                    fprintf('  Using spatial+persistence metric (weights: M=%.2f, A=%.2f, P=%.2f)\n', ...
                        w_moransI, w_activity, w_persistence);
                else
                    % Fallback to Moran's I + Activity only
                    total_weight = config.matchingWeights.moransI + config.matchingWeights.activity;
                    w_moransI = config.matchingWeights.moransI / total_weight;
                    w_activity = config.matchingWeights.activity / total_weight;

                    wasserstein_dist = w_moransI * dist_moransI_norm + w_activity * dist_activity_norm;
                    fprintf('  Fallback to moransI+activity (persistence not available)\n');
                end

                % Mean difference
                for s = 1:nSims
                    mean_diff(s) = abs(IsingData.MoransI_mean(s) - ExpStats.(condition).MoransI_mean);
                end

                % Store raw distances
                dist_moransI_raw = dist_moransI_all;
                dist_activity_raw = dist_activity_all;
            end

        case 'combined'
            % Weighted combination of multiple metrics
            if ~isfield(ExpStats.(condition), 'Activity_all')
                warning('No Activity data for %s. Using moransI only.', condition);
                exp_moransI = ExpStats.(condition).MoransI_all;
                exp_moransI_mean = ExpStats.(condition).MoransI_mean;
                parfor s = 1:nSims
                    ising_values = IsingData.MoransI_all{s}(:);
                    wasserstein_dist(s) = wasserstein_1d(exp_moransI, ising_values);
                    mean_diff(s) = abs(IsingData.MoransI_mean(s) - exp_moransI_mean);
                end
                % Store raw Moran's I distance (fallback)
                dist_moransI_raw = wasserstein_dist;
            else
                % Compute individual distances first
                exp_moransI = ExpStats.(condition).MoransI_all;
                exp_activity = ExpStats.(condition).Activity_all;

                dist_moransI_all = zeros(nSims, 1);
                dist_activity_all = zeros(nSims, 1);

                parfor s = 1:nSims
                    dist_moransI_all(s) = wasserstein_1d(exp_moransI, IsingData.MoransI_all{s}(:));
                    dist_activity_all(s) = wasserstein_1d(exp_activity, IsingData.Activity_all{s}(:));
                end

                % Normalize distances (z-score) before combining
                % Handle NaN values
                dist_moransI_norm = (dist_moransI_all - nanmean(dist_moransI_all)) / nanstd(dist_moransI_all);
                dist_activity_norm = (dist_activity_all - nanmean(dist_activity_all)) / nanstd(dist_activity_all);

                % Replace any remaining NaN with 0 (neutral contribution)
                dist_moransI_norm(isnan(dist_moransI_norm)) = 0;
                dist_activity_norm(isnan(dist_activity_norm)) = 0;

                % =========================================================
                % SEQUENTIAL TWO-STEP MATCHING PROCESS
                % =========================================================
                % Step 1: Compute temporal scale factors using GLOBAL experimental ACF
                % Step 2: Compare blob count distributions using Wasserstein distance

                dist_blob_norm = zeros(nSims, 1);  % Default: no contribution
                useBlob = isfield(ExpStats, 'Global') && ...
                          isfield(ExpStats.Global, 'Autocorr') && ...
                          isfield(ExpStats.Global, 'BlobStats') && ...
                          isfield(IsingData, 'Autocorr_tau') && ...
                          isfield(IsingData, 'BlobStats_counts');

                if useBlob
                    % Step 1: Compute temporal scale factors
                    % (tau_global / tau_ising for each simulation)
                    tau_global = ExpStats.Global.Autocorr.tau;

                    if ~isnan(tau_global) && tau_global > 0
                        % Store temporal scale factors (for potential visualization)
                        if ~isfield(IsingData, 'TemporalScaleFactor') || isempty(IsingData.TemporalScaleFactor)
                            IsingData.TemporalScaleFactor = zeros(nSims, 1);
                            for s = 1:nSims
                                if ~isnan(IsingData.Autocorr_tau(s)) && IsingData.Autocorr_tau(s) > 0
                                    IsingData.TemporalScaleFactor(s) = tau_global / IsingData.Autocorr_tau(s);
                                else
                                    IsingData.TemporalScaleFactor(s) = NaN;
                                end
                            end
                            fprintf('  Step 1: Computed temporal scale factors (global tau=%.2f frames)\n', tau_global);
                        end

                        % Step 2: Blob count Wasserstein distance
                        exp_blob_counts = ExpStats.Global.BlobStats.counts(:);
                        dist_blob_all = zeros(nSims, 1);

                        for s = 1:nSims
                            if ~isnan(IsingData.TemporalScaleFactor(s))
                                % Get Ising blob counts
                                ising_blob_counts = IsingData.BlobStats_counts{s}(:);

                                % Wasserstein distance between blob count distributions
                                dist_blob_all(s) = wasserstein_1d(exp_blob_counts, ising_blob_counts);
                            else
                                dist_blob_all(s) = NaN;
                            end
                        end

                        % Normalize blob distance
                        dist_blob_norm = (dist_blob_all - nanmean(dist_blob_all)) / nanstd(dist_blob_all);
                        dist_blob_norm(isnan(dist_blob_norm)) = 0;
                        fprintf('  Step 2: Computed blob count distances (exp: %.1f±%.1f blobs/frame)\n', ...
                            ExpStats.Global.BlobStats.mean_count, ExpStats.Global.BlobStats.std_count);
                    end
                end

                % Step 3: Compute blob persistence distances (with temporal rescaling)
                dist_persistence_norm = zeros(nSims, 1);  % Default: no contribution
                usePersistence = isfield(ExpStats, 'Global') && ...
                                 isfield(ExpStats.Global, 'BlobPersistence') && ...
                                 isfield(IsingData, 'BlobPersistence_lifetimes') && ...
                                 isfield(IsingData, 'Autocorr_tau');

                if usePersistence && useBlob
                    tau_global = ExpStats.Global.Autocorr.tau;
                    if ~isnan(tau_global) && tau_global > 0
                        exp_lifetimes = ExpStats.Global.BlobPersistence.lifetimes(:);
                        dist_persistence_all = zeros(nSims, 1);

                        for s = 1:nSims
                            if ~isnan(IsingData.TemporalScaleFactor(s))
                                % Rescale Ising lifetimes to real time
                                ising_lifetimes = IsingData.BlobPersistence_lifetimes{s};
                                if ~isempty(ising_lifetimes)
                                    ising_lifetimes_rescaled = ising_lifetimes * IsingData.TemporalScaleFactor(s);
                                    dist_persistence_all(s) = wasserstein_1d(exp_lifetimes, ising_lifetimes_rescaled);
                                else
                                    dist_persistence_all(s) = NaN;
                                end
                            else
                                dist_persistence_all(s) = NaN;
                            end
                        end

                        % Normalize persistence distance
                        dist_persistence_norm = (dist_persistence_all - nanmean(dist_persistence_all)) / nanstd(dist_persistence_all);
                        dist_persistence_norm(isnan(dist_persistence_norm)) = 0;
                        fprintf('  Step 3: Computed blob persistence distances (exp: %.1f±%.1f frames)\n', ...
                            ExpStats.Global.BlobPersistence.mean_lifetime, ExpStats.Global.BlobPersistence.std_lifetime);
                    end
                end

                % Weighted combination: Moran's I + Activity + Blob Count + Blob Persistence
                if useBlob && usePersistence && any(dist_blob_norm ~= 0) && any(dist_persistence_norm ~= 0)
                    % Full 4-metric combination
                    total_weight = config.matchingWeights.moransI + config.matchingWeights.activity + ...
                                   config.matchingWeights.blobCount + config.matchingWeights.blobPersistence;
                    w_moransI = config.matchingWeights.moransI / total_weight;
                    w_activity = config.matchingWeights.activity / total_weight;
                    w_blob = config.matchingWeights.blobCount / total_weight;
                    w_persistence = config.matchingWeights.blobPersistence / total_weight;

                    wasserstein_dist = w_moransI * dist_moransI_norm + ...
                                       w_activity * dist_activity_norm + ...
                                       w_blob * dist_blob_norm + ...
                                       w_persistence * dist_persistence_norm;

                    fprintf('  Using combined metric (weights: M=%.2f, A=%.2f, B=%.2f, P=%.2f)\n', ...
                        w_moransI, w_activity, w_blob, w_persistence);
                elseif useBlob && any(dist_blob_norm ~= 0)
                    % 3-metric combination (no persistence)
                    total_weight = config.matchingWeights.moransI + config.matchingWeights.activity + config.matchingWeights.blobCount;
                    w_moransI = config.matchingWeights.moransI / total_weight;
                    w_activity = config.matchingWeights.activity / total_weight;
                    w_blob = config.matchingWeights.blobCount / total_weight;

                    wasserstein_dist = w_moransI * dist_moransI_norm + ...
                                       w_activity * dist_activity_norm + ...
                                       w_blob * dist_blob_norm;

                    fprintf('  Using 3-metric combination (weights: M=%.2f, A=%.2f, B=%.2f)\n', ...
                        w_moransI, w_activity, w_blob);
                else
                    % Fallback to two-metric combination
                    wasserstein_dist = config.matchingWeights.moransI * dist_moransI_norm + ...
                                       config.matchingWeights.activity * dist_activity_norm;
                    fprintf('  Using two-metric combination (blob matching not available)\n');
                end

                % Mean difference (use Moran's I for reference)
                for s = 1:nSims
                    mean_diff(s) = abs(IsingData.MoransI_mean(s) - ExpStats.(condition).MoransI_mean);
                end

                % Store raw individual Wasserstein distances
                dist_moransI_raw = dist_moransI_all;
                dist_activity_raw = dist_activity_all;
                if useBlob && exist('dist_blob_all', 'var')
                    dist_blob_raw = dist_blob_all;
                end
            end

        otherwise
            error('Unknown matching metric: %s', config.matchingMetric);
    end

    % Rank simulations by Wasserstein distance
    % Handle NaN values by replacing with Inf for sorting (places them last)
    wasserstein_for_sort = wasserstein_dist;
    wasserstein_for_sort(isnan(wasserstein_for_sort)) = Inf;
    [~, rank_order] = sort(wasserstein_for_sort, 'ascend');

    % Store results
    Comparison.(condition).wasserstein_dist = wasserstein_dist;
    Comparison.(condition).mean_diff = mean_diff;
    Comparison.(condition).rankings = rank_order;
    Comparison.(condition).bestMatch_idx = rank_order(1:config.nTopMatches);
    Comparison.(condition).bestMatch_simIDs = IsingData.simIDs(rank_order(1:config.nTopMatches));

    % Store raw individual Wasserstein distances (if computed)
    if exist('dist_moransI_raw', 'var')
        Comparison.(condition).dist_moransI_raw = dist_moransI_raw;
    end
    if exist('dist_activity_raw', 'var')
        Comparison.(condition).dist_activity_raw = dist_activity_raw;
    end
    if exist('dist_blob_raw', 'var')
        Comparison.(condition).dist_blob_raw = dist_blob_raw;
    end
    if exist('dist_autocorr_raw', 'var')
        Comparison.(condition).dist_autocorr_raw = dist_autocorr_raw;
    end

    % Extract parameters of best matches
    best_idx = rank_order(1:config.nTopMatches);
    Comparison.(condition).bestMatch_params = struct();
    Comparison.(condition).bestMatch_params.beta = IsingData.params.beta(best_idx);
    Comparison.(condition).bestMatch_params.c = IsingData.params.c(best_idx);
    Comparison.(condition).bestMatch_params.decay_const = IsingData.params.decay_const(best_idx);
    Comparison.(condition).bestMatch_params.inhibition_range = IsingData.params.inhibition_range(best_idx);
    Comparison.(condition).bestMatch_params.bias = IsingData.params.bias(best_idx);

    % Report top matches
    fprintf('  Top %d matches for %n', config.nTopMatches, condition);
    fprintf('  %20s  %6s  %6s  %8s  %10s  %6s  %12s\n', ...
        'SimID', 'beta', 'c', 'decay', 'inhib_range', 'bias', 'Wasserstein');
    for m = 1:min(5, config.nTopMatches)
        idx = best_idx(m);
        fprintf('  %20s  %6.2f  %6d  %8d  %10d  %6.2f  %12.4f\n', ...
            IsingData.simIDs{idx}, IsingData.params.beta(idx), ...
            IsingData.params.c(idx), IsingData.params.decay_const(idx), ...
            IsingData.params.inhibition_range(idx), IsingData.params.bias(idx), ...
            wasserstein_dist(idx));
    end
end

% -------------------------------------------------------------------------
% DUAL MATCHING for subselect_centre_vs_tiled mode
% -------------------------------------------------------------------------
if strcmp(config.gridMode, 'subselect_centre_vs_tiled')
    fprintf('\n--- Dual Matching: centre vs Tiled ---\n');

    % Initialize separate comparison structures
    Comparison_centre = struct();
    Comparison_tiled = struct();

    for c = 1:length(config.conditions)
        condition = config.conditions{c};

        if ~isfield(ExpStats, condition)
            continue;
        end

        exp_moransI = ExpStats.(condition).MoransI_all;
        nSims = length(IsingData.simIDs);

        % --- centre-BASED MATCHING ---
        wasserstein_centre = zeros(nSims, 1);
        parfor s = 1:nSims
            centre_values = IsingData.MoransI_centre{s}(:);
            wasserstein_centre(s) = wasserstein_1d(exp_moransI, centre_values);
        end
        wasserstein_for_sort = wasserstein_centre;
        wasserstein_for_sort(isnan(wasserstein_for_sort)) = Inf;
        [~, rank_centre] = sort(wasserstein_for_sort, 'ascend');

        Comparison_centre.(condition).wasserstein_dist = wasserstein_centre;
        Comparison_centre.(condition).rankings = rank_centre;
        Comparison_centre.(condition).bestMatch_idx = rank_centre(1:config.nTopMatches);
        Comparison_centre.(condition).bestMatch_simIDs = IsingData.simIDs(rank_centre(1:config.nTopMatches));

        % --- TILED-BASED MATCHING ---
        wasserstein_tiled = zeros(nSims, 1);
        parfor s = 1:nSims
            tiled_values = IsingData.MoransI_tiled{s}(:);
            wasserstein_tiled(s) = wasserstein_1d(exp_moransI, tiled_values);
        end
        wasserstein_for_sort = wasserstein_tiled;
        wasserstein_for_sort(isnan(wasserstein_for_sort)) = Inf;
        [~, rank_tiled] = sort(wasserstein_for_sort, 'ascend');

        Comparison_tiled.(condition).wasserstein_dist = wasserstein_tiled;
        Comparison_tiled.(condition).rankings = rank_tiled;
        Comparison_tiled.(condition).bestMatch_idx = rank_tiled(1:config.nTopMatches);
        Comparison_tiled.(condition).bestMatch_simIDs = IsingData.simIDs(rank_tiled(1:config.nTopMatches));

        % Report comparison
        best_centre = rank_centre(1);
        best_tiled = rank_tiled(1);
        fprintf('  %n', condition);
        fprintf('    centre best: %s (WD=%.4f)\n', IsingData.simIDs{best_centre}, wasserstein_centre(best_centre));
        fprintf('    Tiled best:  %s (WD=%.4f)\n', IsingData.simIDs{best_tiled}, wasserstein_tiled(best_tiled));
        if best_centre == best_tiled
            fprintf('    Same simulation selected!\n');
        else
            fprintf('    Different simulations selected\n');
        end
    end
end

%% =========================================================================
%% SECTION 7: Time Series Dynamics Analysis
%% =========================================================================

fprintf('\n--- Section 7: Time Series Dynamics Analysis ---\n');

DynamicsAnalysis = struct();

for c = 1:length(config.conditions)
    condition = config.conditions{c};

    if ~isfield(Comparison, condition) || ~isfield(ExpStats, condition)
        continue;
    end

    fprintf('\nAnalyzing dynamics for %s...\n', condition);

    % Get experimental time course
    exp_timecourse = ExpStats.(condition).MoransI_timecourse;

    % Calculate autocorrelation for experimental data (trial-averaged)
    if isfield(ExpStats.(condition), 'MoransI_trials') && ~isempty(ExpStats.(condition).MoransI_trials)
        moransI_trials = ExpStats.(condition).MoransI_trials;
        [~, exp_acf, exp_lags, ~] = computeTrialAveragedAutocorr(...
            moransI_trials, config.autocorr.maxLag, config.autocorr.fitRange);
        exp_lags = exp_lags';  % Make row vector to match original format
    else
        exp_acf = [];
        exp_lags = [];
    end

    % Get best matching Ising simulation
    best_idx = Comparison.(condition).bestMatch_idx(1);
    best_ising_timeseries = IsingData.MoransI_all{best_idx};

    % Ensure it's a numeric array, not a cell
    if iscell(best_ising_timeseries)
        best_ising_timeseries = best_ising_timeseries{1};
    end
    best_ising_timeseries = best_ising_timeseries(:);  % Make column vector

    % Remove NaN values for autocorrelation calculation
    best_ising_clean = best_ising_timeseries(~isnan(best_ising_timeseries));

    % Calculate autocorrelation for best Ising match
    if length(best_ising_clean) > config.autocorr.maxLag
        [ising_acf, ising_lags] = xcorr(best_ising_clean - mean(best_ising_clean), ...
            config.autocorr.maxLag, 'normalized');
        ising_acf = ising_acf(config.autocorr.maxLag+1:end);
        ising_lags = ising_lags(config.autocorr.maxLag+1:end);
    else
        ising_acf = [];
        ising_lags = [];
    end

    % Calculate temporal variance (with NaN handling)
    exp_temporal_var = var(exp_timecourse, 'omitnan');
    ising_temporal_var = var(best_ising_timeseries, 'omitnan');

    % Store results
    DynamicsAnalysis.(condition).exp_timecourse = exp_timecourse;
    DynamicsAnalysis.(condition).exp_acf = exp_acf;
    DynamicsAnalysis.(condition).exp_lags = exp_lags;
    DynamicsAnalysis.(condition).exp_temporal_var = exp_temporal_var;

    DynamicsAnalysis.(condition).ising_timeseries = best_ising_timeseries;
    DynamicsAnalysis.(condition).ising_acf = ising_acf;
    DynamicsAnalysis.(condition).ising_lags = ising_lags;
    DynamicsAnalysis.(condition).ising_temporal_var = ising_temporal_var;
    DynamicsAnalysis.(condition).best_simID = IsingData.simIDs{best_idx};

    % Add time constant comparison (if autocorrelation matching is enabled)
    if ismember(config.matchingMetric, {'autocorr', 'blobCount', 'blobPersistence', 'combined', 'spatial+persistence'}) && ...
       isfield(ExpStats.(condition), 'Autocorr') && isfield(ExpStats.(condition).Autocorr, 'tau') && ...
       isfield(IsingData, 'Autocorr_tau')

        exp_tau = ExpStats.(condition).Autocorr.tau;
        ising_tau = IsingData.Autocorr_tau(best_idx);

        DynamicsAnalysis.(condition).exp_tau = exp_tau;
        DynamicsAnalysis.(condition).ising_tau = ising_tau;
        DynamicsAnalysis.(condition).time_scaling = computeTimeScaling(exp_tau, ising_tau);

        % Also store autocorrelation fit quality
        if isfield(ExpStats.(condition).Autocorr, 'fitResult')
            DynamicsAnalysis.(condition).exp_tau_R2 = ExpStats.(condition).Autocorr.fitResult.R2;
        end
        if ~isempty(IsingData.Autocorr_fitR2)
            DynamicsAnalysis.(condition).ising_tau_R2 = IsingData.Autocorr_fitR2(best_idx);
        end

        fprintf('  Time constants:\n');
        fprintf('    Experimental τ: %.2f frames', exp_tau);
        if isfield(DynamicsAnalysis.(condition), 'exp_tau_R2')
            fprintf(' (R²=%.3f)', DynamicsAnalysis.(condition).exp_tau_R2);
        end
        fprintf('\n');
        fprintf('    Best Ising τ: %.2f MC sweeps', ising_tau);
        if isfield(DynamicsAnalysis.(condition), 'ising_tau_R2')
            fprintf(' (R²=%.3f)', DynamicsAnalysis.(condition).ising_tau_R2);
        end
        fprintf('\n');
        if ~isnan(DynamicsAnalysis.(condition).time_scaling)
            fprintf('    Time scaling: 1 MC sweep = %.3f frames\n', DynamicsAnalysis.(condition).time_scaling);
        end
    end

    fprintf('  Experimental: temporal var=%.6f\n', exp_temporal_var);
    fprintf('  Best Ising (%s): temporal var=%.6f\n', ...
        IsingData.simIDs{best_idx}, ising_temporal_var);
end

end  % End of: if ~goto_visualization (data processing sections 2-7)

% Compute ParameterTrends from comparison results (for visualizations)
ParameterTrends = struct();
param_names = {'beta', 'c', 'decay_const', 'inhibition_range', 'bias'};
param_values_all = {config.isingParams.beta_values, config.isingParams.c_values, ...
    config.isingParams.decay_const_values, config.isingParams.inhibition_range_values, ...
    config.isingParams.bias_values};

for c = 1:length(config.conditions)
    condition = config.conditions{c};
    if ~isfield(Comparison, condition)
        continue;
    end
    best_params = Comparison.(condition).bestMatch_params;
    ParameterTrends.(condition) = struct();
    for p = 1:length(param_names)
        param_name = param_names{p};
        param_values = param_values_all{p};
        best_values = best_params.(param_name);
        counts = zeros(1, length(param_values));
        for v = 1:length(param_values)
            counts(v) = sum(best_values == param_values(v));
        end
        ParameterTrends.(condition).(param_name).values = param_values;
        ParameterTrends.(condition).(param_name).counts = counts;
        ParameterTrends.(condition).(param_name).mode = param_values(counts == max(counts));
        ParameterTrends.(condition).(param_name).mean = mean(best_values);
    end
end

% -------------------------------------------------------------------------
% Decay Fit Range Sensitivity Analysis
% -------------------------------------------------------------------------
fprintf('\nRunning decay fit range sensitivity analysis...\n');
FitRangeSensitivity = analyze_decay_fitRange_sensitivity(IsingData, ExpStats, Comparison, config);

%% =========================================================================
%% SECTION 8: Visualizations
%% =========================================================================
% This section and below run regardless of whether data was computed fresh
% or loaded from pre-computed results.

fprintf('\n--- Section 8: Creating Visualizations ---\n');

% Apply publication-quality figure defaults
setPublicationDefaults();

% -------------------------------------------------------------------------
% Figure 1: Distribution Comparison
% -------------------------------------------------------------------------
figure('Name', 'Moran''s I Distribution Comparison');

nConditions = length(config.conditions);
for c = 1:nConditions
    condition = config.conditions{c};

    if ~isfield(ExpStats, condition) || ~isfield(Comparison, condition)
        continue;
    end

    subplot(2, 2, c);
    hold on;

    % Experimental distribution
    histogram(ExpStats.(condition).MoransI_all, 50, 'Normalization', 'probability', ...
        'FaceColor', [0.2 0.4 0.8], 'FaceAlpha', 0.6, 'DisplayName', 'Experimental');

    % Best Ising match distribution
    best_idx = Comparison.(condition).bestMatch_idx(1);
    histogram(IsingData.MoransI_all{best_idx}, 50, 'Normalization', 'probability', ...
        'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.6, 'DisplayName', ...
        sprintf('Ising %s', IsingData.simIDs{best_idx}));

    % Add vertical lines for means
    xline(ExpStats.(condition).MoransI_mean, 'b--', 'LineWidth', 2);
    xline(IsingData.MoransI_mean(best_idx), 'r--', 'LineWidth', 2);

    xlabel('Moran''s I');
    ylabel('Probability');
    title(sprintf('%s', condition));
    legend('Location', 'best');

    hold off;
end

sgtitle('Moran''s I Distribution: Experimental vs Best Ising Match');

% Save figure
saveMyFig('MoransI_Distribution_Comparison', fullfile(config.outputPath, 'Distributions'), gcf);

% -------------------------------------------------------------------------
% Figure 1-KDE: KDE Distribution Comparison (Moran's I)
% -------------------------------------------------------------------------
fprintf('Creating KDE distribution comparison (Moran''s I)...\n');

figure('Name', 'KDE: Moran''s I Distribution');
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [18 14]);

for c = 1:nConditions
    condition = config.conditions{c};

    if ~isfield(ExpStats, condition) || ~isfield(Comparison, condition)
        continue;
    end

    subplot(2, 2, c);
    hold on;

    % Experimental KDE
    exp_data = ExpStats.(condition).MoransI_all;
    exp_data = exp_data(~isnan(exp_data));
    [f_exp, xi_exp] = ksdensity(exp_data);
    fill([xi_exp, fliplr(xi_exp)], [f_exp, zeros(size(f_exp))], ...
        config.colors.(condition), 'FaceAlpha', 0.4, 'EdgeColor', config.colors.(condition), ...
        'LineWidth', 1.0, 'DisplayName', 'Experimental');

    % Best Ising match KDE
    best_idx = Comparison.(condition).bestMatch_idx(1);
    ising_data = IsingData.MoransI_all{best_idx};
    ising_data = ising_data(~isnan(ising_data));
    [f_ising, xi_ising] = ksdensity(ising_data);
    fill([xi_ising, fliplr(xi_ising)], [f_ising, zeros(size(f_ising))], ...
        [0.5 0.5 0.5], 'FaceAlpha', 0.3, 'EdgeColor', [0.3 0.3 0.3], ...
        'LineWidth', 1.0, 'DisplayName', sprintf('Ising %s', IsingData.simIDs{best_idx}));

    % Mean vertical lines
    xline(mean(exp_data), '--', 'Color', config.colors.(condition), 'LineWidth', 1.0);
    xline(mean(ising_data), '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.0);

    % Rug plot (small tick marks along x-axis)
    yl = ylim;
    rugHeight = yl(2) * 0.03;
    nRug = min(500, length(exp_data));  % Subsample for clarity
    rugIdx_exp = round(linspace(1, length(exp_data), nRug));
    plot(exp_data(rugIdx_exp), repmat(-rugHeight, nRug, 1), '|', ...
        'Color', config.colors.(condition), 'MarkerSize', 2, 'HandleVisibility', 'off');
    nRug_i = min(500, length(ising_data));
    rugIdx_ising = round(linspace(1, length(ising_data), nRug_i));
    plot(ising_data(rugIdx_ising), repmat(-rugHeight*2, nRug_i, 1), '|', ...
        'Color', [0.5 0.5 0.5], 'MarkerSize', 2, 'HandleVisibility', 'off');

    % WD annotation
    wd_val = Comparison.(condition).wasserstein_dist(best_idx);
    text(0.95, 0.95, sprintf('WD = %.4f', wd_val), 'Units', 'normalized', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 6);

    xlabel('Moran''s I');
    ylabel('Density');
    title(condition);
    legend('Location', 'best', 'FontSize', 6);
    hold off;
end

sgtitle('KDE: Moran''s I Distribution (Exp vs Best Ising)');
saveMyFig('KDE_MoransI_Distribution', fullfile(config.outputPath, 'Distributions'), gcf);

% -------------------------------------------------------------------------
% Figure 1-KDE-Activity: KDE Distribution Comparison (Activity)
% -------------------------------------------------------------------------
fprintf('Creating KDE distribution comparison (Activity)...\n');

figure('Name', 'KDE: Activity Distribution');
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [18 14]);

for c = 1:nConditions
    condition = config.conditions{c};

    if ~isfield(ExpStats, condition) || ~isfield(Comparison, condition)
        continue;
    end
    if ~isfield(ExpStats.(condition), 'Activity_all') || isempty(ExpStats.(condition).Activity_all)
        continue;
    end

    subplot(2, 2, c);
    hold on;

    % Experimental KDE
    exp_data = ExpStats.(condition).Activity_all;
    exp_data = exp_data(~isnan(exp_data));
    [f_exp, xi_exp] = ksdensity(exp_data);
    fill([xi_exp, fliplr(xi_exp)], [f_exp, zeros(size(f_exp))], ...
        config.colors.(condition), 'FaceAlpha', 0.4, 'EdgeColor', config.colors.(condition), ...
        'LineWidth', 1.0, 'DisplayName', 'Experimental');

    % Best Ising match KDE
    best_idx = Comparison.(condition).bestMatch_idx(1);
    ising_data = IsingData.Activity_all{best_idx};
    ising_data = ising_data(~isnan(ising_data));
    [f_ising, xi_ising] = ksdensity(ising_data);
    fill([xi_ising, fliplr(xi_ising)], [f_ising, zeros(size(f_ising))], ...
        [0.5 0.5 0.5], 'FaceAlpha', 0.3, 'EdgeColor', [0.3 0.3 0.3], ...
        'LineWidth', 1.0, 'DisplayName', sprintf('Ising %s', IsingData.simIDs{best_idx}));

    % Mean vertical lines
    xline(mean(exp_data), '--', 'Color', config.colors.(condition), 'LineWidth', 1.0);
    xline(mean(ising_data), '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.0);

    % Rug plot
    yl = ylim;
    rugHeight = yl(2) * 0.03;
    nRug = min(500, length(exp_data));
    rugIdx_exp = round(linspace(1, length(exp_data), nRug));
    plot(exp_data(rugIdx_exp), repmat(-rugHeight, nRug, 1), '|', ...
        'Color', config.colors.(condition), 'MarkerSize', 2, 'HandleVisibility', 'off');
    nRug_i = min(500, length(ising_data));
    rugIdx_ising = round(linspace(1, length(ising_data), nRug_i));
    plot(ising_data(rugIdx_ising), repmat(-rugHeight*2, nRug_i, 1), '|', ...
        'Color', [0.5 0.5 0.5], 'MarkerSize', 2, 'HandleVisibility', 'off');

    % WD annotation (activity-specific)
    if isfield(Comparison.(condition), 'dist_activity_raw')
        wd_val = Comparison.(condition).dist_activity_raw(best_idx);
        text(0.95, 0.95, sprintf('WD = %.4f', wd_val), 'Units', 'normalized', ...
            'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 6);
    end

    xlabel('Activity (fraction active)');
    ylabel('Density');
    title(condition);
    legend('Location', 'best', 'FontSize', 6);
    hold off;
end

sgtitle('KDE: Activity Distribution (Exp vs Best Ising)');
saveMyFig('KDE_Activity_Distribution', fullfile(config.outputPath, 'Distributions'), gcf);

% -------------------------------------------------------------------------
% Figure 2: Parameter Heatmaps
% -------------------------------------------------------------------------
figure('Name', 'Parameter Space Analysis');

% Create 2D slices of the parameter space (beta vs bias, averaged over others)
for c = 1:nConditions
    condition = config.conditions{c};

    if ~isfield(Comparison, condition)
        continue;
    end

    subplot(2, 2, c);

    % Create heatmap of Wasserstein distances (beta vs bias)
    beta_vals = config.isingParams.beta_values;
    bias_vals = config.isingParams.bias_values;

    heatmap_data = nan(length(beta_vals), length(bias_vals));

    for b1 = 1:length(beta_vals)
        for b2 = 1:length(bias_vals)
            % Find simulations with this beta-bias combination
            mask = (IsingData.params.beta == beta_vals(b1)) & ...
                   (IsingData.params.bias == bias_vals(b2));

            if any(mask)
                % Average Wasserstein distance for this parameter combination
                heatmap_data(b1, b2) = mean(Comparison.(condition).wasserstein_dist(mask));
            end
        end
    end

    imagesc(bias_vals, beta_vals, heatmap_data);
    colorbar;
    colormap(gca, flipud(hot));  % Lower distance = better match = brighter

    xlabel('Bias');
    ylabel('Beta');
    title(sprintf('%s: Match Quality (beta vs bias)', condition));

    set(gca, 'YDir', 'normal');
end

sgtitle('Parameter Space: Wasserstein Distance (lower = better match)');

% Save figure
saveMyFig('Parameter_Heatmaps', fullfile(config.outputPath, 'ParameterAnalysis'), gcf);

% -------------------------------------------------------------------------
% Figure 3: Best Match Parameters per Condition
% -------------------------------------------------------------------------
figure('Name', 'Best Match Parameter Distribution');

param_names = {'beta', 'c', 'decay_const', 'inhibition_range', 'bias'};
param_labels = {'Beta', 'Coupling (c)', 'Decay Const', 'Inhib Range', 'Bias'};

for p = 1:length(param_names)
    subplot(2, 3, p);

    % Collect best match values for each condition
    data_for_box = [];
    group_labels = {};

    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(Comparison, condition)
            vals = Comparison.(condition).bestMatch_params.(param_names{p});
            data_for_box = [data_for_box; vals(:)];
            group_labels = [group_labels; repmat({condition}, length(vals), 1)];
        end
    end

    if ~isempty(data_for_box)
        boxplot(data_for_box, group_labels);
        ylabel(param_labels{p});
        title(param_labels{p});

        % Rotate x-labels for readability
        xtickangle(45);
    end
end

sgtitle(sprintf('Best Match Parameters (Top %d matches)', config.nTopMatches));

% Save figure
saveMyFig('BestMatch_Parameters', fullfile(config.outputPath, 'ParameterAnalysis'), gcf);

% -------------------------------------------------------------------------
% Figure 3b: Condition-Pair Parameter Overlap (Jaccard Index)
% -------------------------------------------------------------------------
fprintf('Creating condition-pair parameter overlap matrix...\n');

figure('Name', 'Condition-Pair Parameter Overlap');
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [12 10]);

% Compute Jaccard index of top-10 bestMatch_idx between all condition pairs
jaccardMatrix = eye(nConditions);  % Diagonal = 1.0

for ci = 1:nConditions
    cond_i = config.conditions{ci};
    if ~isfield(Comparison, cond_i); continue; end
    nTop_i = min(10, length(Comparison.(cond_i).bestMatch_idx));
    set_i = Comparison.(cond_i).bestMatch_idx(1:nTop_i);

    for cj = (ci+1):nConditions
        cond_j = config.conditions{cj};
        if ~isfield(Comparison, cond_j); continue; end
        nTop_j = min(10, length(Comparison.(cond_j).bestMatch_idx));
        set_j = Comparison.(cond_j).bestMatch_idx(1:nTop_j);

        % Jaccard index = |intersection| / |union|
        intersect_count = length(intersect(set_i, set_j));
        union_count = length(union(set_i, set_j));
        if union_count > 0
            jaccard_val = intersect_count / union_count;
        else
            jaccard_val = 0;
        end
        jaccardMatrix(ci, cj) = jaccard_val;
        jaccardMatrix(cj, ci) = jaccard_val;
    end
end

imagesc(jaccardMatrix, [0 1]);
colorbar;
colormap(gca, flipud(bone));
axis square;
set(gca, 'XTick', 1:nConditions, 'XTickLabel', config.conditions);
set(gca, 'YTick', 1:nConditions, 'YTickLabel', config.conditions);
xtickangle(45);

% Overlay text values
for ci = 1:nConditions
    for cj = 1:nConditions
        val = jaccardMatrix(ci, cj);
        if val > 0.5
            txtColor = 'w';
        else
            txtColor = 'k';
        end
        text(cj, ci, sprintf('%.2f', val), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'FontSize', 7, 'Color', txtColor);
    end
end

title('Parameter Overlap: Jaccard Index of Top 10 Matches');
saveMyFig('ConditionPair_ParameterOverlap', fullfile(config.outputPath, 'ParameterAnalysis'), gcf);

% -------------------------------------------------------------------------
% Figure 4: Autocorrelation Comparison (Publication Quality)
% -------------------------------------------------------------------------
% Single-row layout with residual strips below each condition panel.
% Stronger fit-range shading, clean text, adaptive y-limits, pub line widths.

figure('Name', 'Temporal Dynamics: Autocorrelation');
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [24 10]);

% Count valid conditions for layout
validConds_acf = 0;
for c = 1:nConditions
    if isfield(DynamicsAnalysis, config.conditions{c}); validConds_acf = validConds_acf + 1; end
end
acf_col = 0;

for c = 1:nConditions
    condition = config.conditions{c};

    if ~isfield(DynamicsAnalysis, condition)
        continue;
    end
    acf_col = acf_col + 1;

    % Main ACF panel (top 75%)
    subplot(5, validConds_acf, [acf_col, acf_col + validConds_acf, acf_col + 2*validConds_acf]);
    hold on;

    % Shade fit range
    fitLo = config.autocorr.fitRange(1);
    fitHi = config.autocorr.fitRange(2);
    patch([fitLo fitHi fitHi fitLo], [-0.3 -0.3 1.1 1.1], ...
        [0.9 0.9 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    % Experimental autocorrelation
    if ~isempty(DynamicsAnalysis.(condition).exp_acf)
        plot(DynamicsAnalysis.(condition).exp_lags, DynamicsAnalysis.(condition).exp_acf, ...
            '-', 'Color', config.colors.(condition), 'LineWidth', 1.0, 'DisplayName', 'Experimental');
    end

    % Best Ising match autocorrelation
    plot(DynamicsAnalysis.(condition).ising_lags, DynamicsAnalysis.(condition).ising_acf, ...
        '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0, 'DisplayName', 'Best Ising');

    % Add fitted exponential decay curves
    exp_tau = NaN; ising_tau = NaN;
    if isfield(DynamicsAnalysis.(condition), 'exp_tau') && isfield(DynamicsAnalysis.(condition), 'ising_tau')
        exp_tau = DynamicsAnalysis.(condition).exp_tau;
        ising_tau = DynamicsAnalysis.(condition).ising_tau;

        if ~isnan(exp_tau) && ~isnan(ising_tau)
            lags_fit = 0:config.autocorr.maxLag;
            exp_fit = exp(-lags_fit / exp_tau);
            ising_fit = exp(-lags_fit / ising_tau);

            plot(lags_fit, exp_fit, '--', 'Color', config.colors.(condition), 'LineWidth', 0.75, ...
                'DisplayName', sprintf('Exp fit (\\tau=%.1f)', exp_tau));
            plot(lags_fit, ising_fit, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.75, ...
                'DisplayName', sprintf('Ising fit (\\tau=%.1f)', ising_tau));

            % Clean text annotation (no wheat background box)
            if isfield(DynamicsAnalysis.(condition), 'time_scaling') && ...
               ~isnan(DynamicsAnalysis.(condition).time_scaling)
                text(0.98, 0.92, sprintf('1 MC = %.2f fr', DynamicsAnalysis.(condition).time_scaling), ...
                    'Units', 'normalized', 'FontSize', 6, 'HorizontalAlignment', 'right');
            end
        end
    end

    xlabel('Lag');
    if acf_col == 1; ylabel('Autocorrelation'); end
    title(condition);
    legend('Location', 'northeast', 'FontSize', 5);

    % Adaptive y-limits
    acf_data = DynamicsAnalysis.(condition).exp_acf;
    if ~isempty(acf_data)
        yMin = min(-0.1, min(acf_data) - 0.05);
        yMax = max(1.05, max(acf_data) + 0.05);
        ylim([yMin yMax]);
    end
    xlim([0 config.autocorr.maxLag]);
    hold off;

    % Residual strip (bottom 25%)
    subplot(5, validConds_acf, [acf_col + 3*validConds_acf, acf_col + 4*validConds_acf]);
    hold on;

    % Compute residuals (exp - ising) at shared lags
    if ~isempty(DynamicsAnalysis.(condition).exp_acf) && ~isempty(DynamicsAnalysis.(condition).ising_acf)
        exp_acf_vals = DynamicsAnalysis.(condition).exp_acf;
        ising_acf_vals = DynamicsAnalysis.(condition).ising_acf;
        nLags = min(length(exp_acf_vals), length(ising_acf_vals));
        residuals = exp_acf_vals(1:nLags) - ising_acf_vals(1:nLags);
        lag_vals = 0:(nLags-1);

        bar(lag_vals, residuals, 1, 'FaceColor', config.colors.(condition), ...
            'EdgeColor', 'none', 'FaceAlpha', 0.6);
        yline(0, '-k', 'LineWidth', 0.5);

        % Shade fit range in residual panel too
        patch([fitLo fitHi fitHi fitLo], [-1 -1 1 1], ...
            [0.9 0.9 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'HandleVisibility', 'off');
    end

    xlim([0 config.autocorr.maxLag]);
    if acf_col == 1; ylabel('Residual'); end
    xlabel('Lag');
    hold off;
end

sgtitle('Moran''s I Temporal Autocorrelation');
saveMyFig('Autocorrelation_Comparison', fullfile(config.outputPath, 'TemporalDynamics'), gcf);

% -------------------------------------------------------------------------
% Figure 4a: Activity Temporal Autocorrelation (used for time rescaling)
% -------------------------------------------------------------------------
figure('Name', 'Temporal Dynamics: Activity Autocorrelation');

for c = 1:nConditions
    condition = config.conditions{c};

    if ~isfield(ExpStats, condition) || ~isfield(ExpStats.(condition), 'Activity_all')
        continue;
    end

    if ~isfield(Comparison, condition)
        continue;
    end

    subplot(2, 2, c);
    hold on;

    % Experimental activity autocorrelation
    exp_activity = ExpStats.(condition).Activity_all;
    exp_activity_clean = exp_activity(~isnan(exp_activity));
    if length(exp_activity_clean) > config.autocorr.maxLag
        [exp_acf_act, ~] = xcorr(exp_activity_clean - mean(exp_activity_clean), ...
            config.autocorr.maxLag, 'normalized');
        exp_acf_act = exp_acf_act(config.autocorr.maxLag+1:end);
        plot(0:config.autocorr.maxLag, exp_acf_act, 'b-', 'LineWidth', 2, 'DisplayName', 'Experimental');

        % Fit exponential decay to experimental activity
        [exp_tau_act, ~, ~, ~] = computeAutocorrDecay(...
            exp_activity_clean, config.autocorr.maxLag, config.autocorr.fitRange);
    else
        exp_tau_act = NaN;
    end

    % Best Ising match activity autocorrelation
    best_idx = Comparison.(condition).bestMatch_idx(1);
    ising_activity = IsingData.Activity_all{best_idx};
    ising_activity_clean = ising_activity(~isnan(ising_activity));
    if length(ising_activity_clean) > config.autocorr.maxLag
        [ising_acf_act, ~] = xcorr(ising_activity_clean - mean(ising_activity_clean), ...
            config.autocorr.maxLag, 'normalized');
        ising_acf_act = ising_acf_act(config.autocorr.maxLag+1:end);
        plot(0:config.autocorr.maxLag, ising_acf_act, 'r-', 'LineWidth', 2, ...
            'DisplayName', sprintf('Ising %s', IsingData.simIDs{best_idx}));
    end

    % Get Ising tau (already computed)
    ising_tau_act = IsingData.Autocorr_tau(best_idx);

    % Add fitted exponential decay curves
    if ~isnan(exp_tau_act) && ~isnan(ising_tau_act) && exp_tau_act > 0 && ising_tau_act > 0
        lags_fit = 0:config.autocorr.maxLag;
        exp_fit = exp(-lags_fit / exp_tau_act);
        ising_fit = exp(-lags_fit / ising_tau_act);

        plot(lags_fit, exp_fit, 'b--', 'LineWidth', 1.5, ...
            'DisplayName', sprintf('Exp fit (τ=%.1f)', exp_tau_act));
        plot(lags_fit, ising_fit, 'r--', 'LineWidth', 1.5, ...
            'DisplayName', sprintf('Ising fit (τ=%.1f)', ising_tau_act));

        % Add time scaling annotation
        time_scaling = exp_tau_act / ising_tau_act;
        text(0.5, 0.85, sprintf('1 MC = %.2f frames', time_scaling), ...
            'Units', 'normalized', 'FontSize', 8, 'BackgroundColor', [1 1 1 0.7]);
    end

    xlabel('Lag');
    ylabel('Autocorrelation');
    title(condition);
    legend('Location', 'best', 'FontSize', 7);
    xlim([0 config.autocorr.maxLag]);

    hold off;
end

sgtitle('Activity Temporal Autocorrelation (used for time rescaling)');

% Save figure
saveMyFig('Autocorrelation_Activity_Comparison', fullfile(config.outputPath, 'TemporalDynamics'), gcf);

% -------------------------------------------------------------------------
% Figure 4b: Time Constant Comparison (if autocorrelation matching enabled)
% -------------------------------------------------------------------------
if ismember(config.matchingMetric, {'autocorr', 'blobCount', 'blobPersistence', 'combined', 'spatial+persistence'})
    % Check if any condition has time constant data
    hasTimeConstantData = false;
    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(DynamicsAnalysis, condition) && isfield(DynamicsAnalysis.(condition), 'exp_tau')
            hasTimeConstantData = true;
            break;
        end
    end

    if hasTimeConstantData
        figure('Name', 'Time Constants Comparison');

        % Collect data
        tau_exp_all = nan(nConditions, 1);
        tau_ising_all = nan(nConditions, 1);
        scaling_all = nan(nConditions, 1);
        condition_labels = config.conditions;

        for c = 1:nConditions
            condition = config.conditions{c};
            if isfield(DynamicsAnalysis, condition) && isfield(DynamicsAnalysis.(condition), 'exp_tau')
                tau_exp_all(c) = DynamicsAnalysis.(condition).exp_tau;
                tau_ising_all(c) = DynamicsAnalysis.(condition).ising_tau;
                if isfield(DynamicsAnalysis.(condition), 'time_scaling')
                    scaling_all(c) = DynamicsAnalysis.(condition).time_scaling;
                end
            end
        end

        % Plot 1: Time constants comparison
        subplot(1, 2, 1);
        bar_data = [tau_exp_all, tau_ising_all];
        b = bar(bar_data);
        b(1).FaceColor = [0.2 0.4 0.8];  % Blue for experimental
        b(2).FaceColor = [0.8 0.2 0.2];  % Red for Ising

        set(gca, 'XTickLabel', condition_labels);
        xtickangle(45);
        ylabel('Time Constant τ');
        legend({'Experimental (frames)', 'Best Ising (MC sweeps)'}, 'Location', 'best');
        title('Autocorrelation Decay Time Constants');
        grid on;

        % Plot 2: Time scaling factors
        subplot(1, 2, 2);
        bar_colors = zeros(nConditions, 3);
        for c = 1:nConditions
            condition = config.conditions{c};
            if isfield(config.colors, condition)
                bar_colors(c, :) = config.colors.(condition);
            else
                bar_colors(c, :) = [0.5 0.5 0.5];
            end
        end

        b = bar(scaling_all);
        b.FaceColor = 'flat';
        for c = 1:nConditions
            b.CData(c,:) = bar_colors(c,:);
        end

        set(gca, 'XTickLabel', condition_labels);
        xtickangle(45);
        ylabel('Frames per MC Sweep');
        title('Time Scaling Factor');
        grid on;

        % Add horizontal reference line at 1
        hold on;
        yline(1, '--k', 'LineWidth', 1, 'Alpha', 0.5);
        hold off;

        sgtitle('Autocorrelation Time Constants (Sequential matching: temporal rescaling + blob detection)');

        % Save figure
        saveMyFig('TimeConstants_Comparison', fullfile(config.outputPath, 'TemporalDynamics'), gcf);
    end
end

% -------------------------------------------------------------------------
% Figure 4b-Dot: Paired Dot-Line Time Constants
% -------------------------------------------------------------------------
fprintf('Creating paired dot-line time constants figure...\n');

% Collect tau data for paired dot-line plot
tau_exp_dot = nan(nConditions, 1);
tau_ising_dot = nan(nConditions, 1);
hasTauData = false;

for c = 1:nConditions
    condition = config.conditions{c};
    if isfield(DynamicsAnalysis, condition) && isfield(DynamicsAnalysis.(condition), 'exp_tau')
        tau_exp_dot(c) = DynamicsAnalysis.(condition).exp_tau;
        tau_ising_dot(c) = DynamicsAnalysis.(condition).ising_tau;
        hasTauData = true;
    end
end

if hasTauData
    figure('Name', 'Paired Dot-Line: Time Constants');
    set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10]);

    % Panel 1: Paired tau values
    subplot(1, 2, 1);
    hold on;

    for c = 1:nConditions
        condition = config.conditions{c};
        if isnan(tau_exp_dot(c)) || isnan(tau_ising_dot(c)); continue; end

        condColor = config.colors.(condition);

        % Connecting line
        plot([c c], [tau_exp_dot(c) tau_ising_dot(c)], '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.75);

        % Experimental dot (condition-colored)
        plot(c, tau_exp_dot(c), 'o', 'Color', condColor, 'MarkerFaceColor', condColor, ...
            'MarkerSize', 8, 'LineWidth', 0.75);

        % Ising dot (gray)
        plot(c, tau_ising_dot(c), 'o', 'Color', [0.5 0.5 0.5], 'MarkerFaceColor', [0.5 0.5 0.5], ...
            'MarkerSize', 8, 'LineWidth', 0.75);
    end

    set(gca, 'XTick', 1:nConditions, 'XTickLabel', config.conditions);
    xtickangle(45);
    ylabel('Time Constant \tau');
    title('Exp (colored) vs Ising (gray)');
    hold off;

    % Panel 2: Tau ratio (paired dot-line)
    subplot(1, 2, 2);
    hold on;

    yline(1, '--k', 'LineWidth', 0.75, 'Alpha', 0.5);  % Reference line at ratio = 1

    for c = 1:nConditions
        condition = config.conditions{c};
        if isnan(tau_exp_dot(c)) || isnan(tau_ising_dot(c)) || tau_ising_dot(c) == 0; continue; end

        condColor = config.colors.(condition);
        tau_ratio_val = tau_exp_dot(c) / tau_ising_dot(c);

        % Connecting line from ratio=1 to actual
        plot([c c], [1 tau_ratio_val], '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.75);

        % Dot
        plot(c, tau_ratio_val, 'o', 'Color', condColor, 'MarkerFaceColor', condColor, ...
            'MarkerSize', 8, 'LineWidth', 0.75);
    end

    set(gca, 'XTick', 1:nConditions, 'XTickLabel', config.conditions);
    xtickangle(45);
    ylabel('\tau_{exp} / \tau_{ising}');
    title('Tau Ratio (1 = perfect match)');
    hold off;

    sgtitle('Time Constants: Paired Dot-Line');
    saveMyFig('TimeConstants_PairedDotLine', fullfile(config.outputPath, 'TemporalDynamics'), gcf);
end

% -------------------------------------------------------------------------
% Figure 4c: Conversion Factor Variability (Top 10 vs All Simulations)
% -------------------------------------------------------------------------
fprintf('\n--- Creating Conversion Factor Variability Figure ---\n');

% Check if TemporalScaleFactor exists
if ~isfield(IsingData, 'TemporalScaleFactor') || isempty(IsingData.TemporalScaleFactor)
    fprintf('  Skipping: No temporal scale factors available\n');
else
    figure('Name', 'Conversion Factor Variability');

    % Get all conversion factors (excluding NaN)
    all_cf = IsingData.TemporalScaleFactor;
    all_cf_valid = all_cf(~isnan(all_cf));

    % Panel 1: daboxplot - Top 10 per condition vs All
    subplot(1, 2, 1);

    % Prepare cell array for daboxplot (one cell per group)
    data_cells = cell(1, nConditions + 1);
    cf_group_labels = cell(1, nConditions + 1);
    cf_colors = zeros(nConditions + 1, 3);

    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(Comparison, condition)
            best_idx = Comparison.(condition).bestMatch_idx;
            cf_top10 = all_cf(best_idx);
            cf_top10_valid = cf_top10(~isnan(cf_top10));
            data_cells{c} = cf_top10_valid(:);
            cf_group_labels{c} = condition;
            cf_colors(c, :) = config.colors.(condition);
        end
    end

    % Add all simulations as final group
    data_cells{end} = all_cf_valid(:);
    cf_group_labels{end} = 'All Sims';
    cf_colors(end, :) = [0.5, 0.5, 0.5];  % Gray for all sims

    daboxplot(data_cells, 'colors', cf_colors, 'xtlabels', cf_group_labels, ...
        'scatter', 1, 'outliers', 1, 'mean', 1, 'boxalpha', 0.5, 'whiskers', 0);
    ylabel('Conversion Factor (frames/MC sweep)');
    title('Top 10 Matches vs All Simulations');
    xtickangle(45);

    % Panel 2: Spread comparison (std) with ratio annotation
    subplot(1, 2, 2);
    spread_values = zeros(1, nConditions + 1);

    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(Comparison, condition)
            best_idx = Comparison.(condition).bestMatch_idx;
            cf_top10 = all_cf(best_idx);
            spread_values(c) = std(cf_top10, 'omitnan');
        end
    end
    spread_values(end) = std(all_cf_valid);

    b = bar(spread_values);
    b.FaceColor = 'flat';
    for c = 1:nConditions
        b.CData(c,:) = cf_colors(c,:);
    end
    b.CData(end,:) = [0.5, 0.5, 0.5];

    set(gca, 'XTickLabel', cf_group_labels);
    ylabel('Std Dev of Conversion Factor');
    title('Spread: Top 10 vs All');
    xtickangle(45);

    % Add ratio annotations
    all_spread = spread_values(end);
    for c = 1:nConditions
        ratio = spread_values(c) / all_spread;
        text(c, spread_values(c) + 0.02*max(spread_values), ...
            sprintf('%.1f%%', ratio*100), 'HorizontalAlignment', 'center', 'FontSize', 8);
    end

    sgtitle('Conversion Factor Variability Analysis');
    saveMyFig('ConversionFactor_Variability', fullfile(config.outputPath, 'TemporalDynamics'), gcf);

    % Print summary statistics
    fprintf('Conversion Factor Summary:\n');
    fprintf('  All simulations: mean=%.3f, std=%.3f (n=%d)\n', ...
        mean(all_cf_valid), std(all_cf_valid), length(all_cf_valid));
    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(Comparison, condition)
            best_idx = Comparison.(condition).bestMatch_idx;
            cf_top10 = all_cf(best_idx);
            cf_valid = cf_top10(~isnan(cf_top10));
            fprintf('  %s top 10: mean=%.3f, std=%.3f (%.1f%% of all spread)\n', ...
                condition, mean(cf_valid), std(cf_valid), 100*std(cf_valid)/std(all_cf_valid));
        end
    end
end

% -------------------------------------------------------------------------
% Figure 5a: Summary - Parameter Profile
%% -------------------------------------------------------------------------
figure('Name', 'Summary: Parameter Profile');

% Bar chart of mean best-match parameters per condition
param_names_short = {'beta', 'c', 'decay', 'inhib', 'bias'};
nParams = length(param_names_short);
bar_data = zeros(nConditions, nParams);

for c = 1:nConditions
    condition = config.conditions{c};
    if isfield(ParameterTrends, condition)
        bar_data(c, 1) = ParameterTrends.(condition).beta.mean;
        bar_data(c, 2) = ParameterTrends.(condition).c.mean;
        bar_data(c, 3) = ParameterTrends.(condition).decay_const.mean;
        bar_data(c, 4) = ParameterTrends.(condition).inhibition_range.mean;
        bar_data(c, 5) = ParameterTrends.(condition).bias.mean;
    end
end

% Normalize for visualization (z-score across conditions)
bar_data_normalized = (bar_data - mean(bar_data, 1)) ./ std(bar_data, 0, 1);

imagesc(bar_data_normalized);
colorbar;
colormap(gca, redblue(64));
caxis([-2, 2]);

set(gca, 'XTick', 1:nParams, 'XTickLabel', param_names_short);
set(gca, 'YTick', 1:nConditions, 'YTickLabel', config.conditions);
xlabel('Parameter');
ylabel('Condition');
title('Parameter Profile (z-scored)');

% Save figure
saveMyFig('Summary_ParameterProfile', fullfile(config.outputPath, 'Summary'), gcf);

% -------------------------------------------------------------------------
% Figure 5b: Summary - Exp vs Best Match Comparison
%% -------------------------------------------------------------------------
figure('Name', 'Summary: Exp vs Best Match');

% Always show all 3 core metrics regardless of matching mode
metricsToShow = {'moransI', 'activity', 'blobPersistence'};

nMetrics = length(metricsToShow);
tiledlayout(1, nMetrics, 'TileSpacing', 'compact', 'Padding', 'compact');

for m = 1:nMetrics
    nexttile;
    metric = metricsToShow{m};

    exp_values = zeros(nConditions, 1);
    ising_values = zeros(nConditions, 1);

    for c = 1:nConditions
        condition = config.conditions{c};

        switch metric
            case 'moransI'
                if isfield(ExpStats, condition)
                    exp_values(c) = ExpStats.(condition).MoransI_mean;
                end
                if isfield(Comparison, condition)
                    best_idx = Comparison.(condition).bestMatch_idx(1);
                    ising_values(c) = IsingData.MoransI_mean(best_idx);
                end
                ylabel_str = 'Mean Moran''s I';
                title_str = 'Moran''s I';

            case 'activity'
                if isfield(ExpStats, condition) && isfield(ExpStats.(condition), 'Activity_mean')
                    exp_values(c) = ExpStats.(condition).Activity_mean;
                end
                if isfield(Comparison, condition) && ~isempty(IsingData.Activity_mean)
                    best_idx = Comparison.(condition).bestMatch_idx(1);
                    ising_values(c) = IsingData.Activity_mean(best_idx);
                end
                ylabel_str = 'Mean Activity';
                title_str = 'Activity Distribution';

            case 'autocorr'
                if isfield(ExpStats, condition) && isfield(ExpStats.(condition), 'Autocorr') && ...
                   isfield(ExpStats.(condition).Autocorr, 'tau')
                    exp_values(c) = ExpStats.(condition).Autocorr.tau;
                end
                if isfield(Comparison, condition) && isfield(IsingData, 'Autocorr_tau') && ...
                   ~isempty(IsingData.Autocorr_tau)
                    best_idx = Comparison.(condition).bestMatch_idx(1);
                    ising_values(c) = IsingData.Autocorr_tau(best_idx);
                end
                ylabel_str = 'Decay Constant (\tau)';
                title_str = 'Autocorrelation';

            case 'blobCount'
                if isfield(ExpStats, 'Global') && isfield(ExpStats.Global, 'BlobStats')
                    exp_values(c) = ExpStats.Global.BlobStats.mean_count;
                end
                if isfield(Comparison, condition) && isfield(IsingData, 'BlobStats_mean_count') && ...
                   ~isempty(IsingData.BlobStats_mean_count)
                    best_idx = Comparison.(condition).bestMatch_idx(1);
                    ising_values(c) = IsingData.BlobStats_mean_count(best_idx);
                end
                ylabel_str = 'Mean Blob Count';
                title_str = 'Blob Count';

            case 'blobPersistence'
                if isfield(ExpStats, 'Global') && isfield(ExpStats.Global, 'BlobPersistence')
                    exp_values(c) = ExpStats.Global.BlobPersistence.mean_lifetime;
                end
                if isfield(Comparison, condition) && isfield(IsingData, 'BlobPersistence_mean') && ...
                   ~isempty(IsingData.BlobPersistence_mean)
                    best_idx = Comparison.(condition).bestMatch_idx(1);
                    % Rescale to real time using tau
                    if isfield(IsingData, 'Autocorr_tau') && isfield(ExpStats, 'Global') && ...
                       isfield(ExpStats.Global, 'Autocorr') && ~isnan(ExpStats.Global.Autocorr.tau) && ...
                       ~isnan(IsingData.Autocorr_tau(best_idx)) && IsingData.Autocorr_tau(best_idx) > 0
                        scale_factor = ExpStats.Global.Autocorr.tau / IsingData.Autocorr_tau(best_idx);
                        ising_values(c) = IsingData.BlobPersistence_mean(best_idx) * scale_factor;
                    else
                        ising_values(c) = IsingData.BlobPersistence_mean(best_idx);
                    end
                end
                ylabel_str = 'Mean Blob Lifetime (frames)';
                title_str = 'Blob Persistence';
        end
    end

    bar_handle = bar([exp_values, ising_values]);
    bar_handle(1).FaceColor = [0.2 0.4 0.8];
    bar_handle(2).FaceColor = [0.8 0.2 0.2];

    set(gca, 'XTick', 1:nConditions, 'XTickLabel', config.conditions);
    xtickangle(45);
    ylabel(ylabel_str);
    title([title_str ': Exp vs Best Match']);

    % Add legend only on the last tile
    if m == nMetrics
        legend({'Experimental', 'Best Ising Match'}, 'Location', 'best');
    end
end

sgtitle('Summary: Exp vs Best Ising Match');

% Save figure
saveMyFig('Summary_ExpVsIsingMatch', fullfile(config.outputPath, 'Summary'), gcf);

% -------------------------------------------------------------------------
% Figure 5c: Radar/Spider Plot - Multi-Observable Match Quality
% -------------------------------------------------------------------------
fprintf('Creating radar/spider plot...\n');

figure('Name', 'Radar: Multi-Observable Match Quality');
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [14 14]);

% Define spokes: each metric, lower = better, normalize to [0,1]
radarLabels = {'Moran''s I WD', 'Activity WD', 'Blob Persist. WD', 'Autocorr \tau ratio'};
nSpokes = length(radarLabels);

% Collect raw values per condition for normalization
radarRaw = nan(nConditions, nSpokes);

for c = 1:nConditions
    condition = config.conditions{c};
    if ~isfield(Comparison, condition); continue; end

    best_idx = Comparison.(condition).bestMatch_idx(1);

    % Spoke 1: Moran's I Wasserstein distance
    if isfield(Comparison.(condition), 'dist_moransI_raw')
        radarRaw(c, 1) = Comparison.(condition).dist_moransI_raw(best_idx);
    elseif isfield(Comparison.(condition), 'wasserstein_dist')
        radarRaw(c, 1) = Comparison.(condition).wasserstein_dist(best_idx);
    end

    % Spoke 2: Activity Wasserstein distance
    if isfield(Comparison.(condition), 'dist_activity_raw')
        radarRaw(c, 2) = Comparison.(condition).dist_activity_raw(best_idx);
    end

    % Spoke 3: Blob Persistence Wasserstein distance
    if isfield(Comparison.(condition), 'dist_blobPersistence_raw')
        radarRaw(c, 3) = Comparison.(condition).dist_blobPersistence_raw(best_idx);
    end

    % Spoke 4: Autocorrelation tau ratio (|log(exp_tau/ising_tau)|, 0 = perfect)
    if isfield(DynamicsAnalysis, condition) && isfield(DynamicsAnalysis.(condition), 'exp_tau') && ...
       isfield(DynamicsAnalysis.(condition), 'ising_tau')
        exp_tau_r = DynamicsAnalysis.(condition).exp_tau;
        ising_tau_r = DynamicsAnalysis.(condition).ising_tau;
        if ~isnan(exp_tau_r) && ~isnan(ising_tau_r) && exp_tau_r > 0 && ising_tau_r > 0
            radarRaw(c, 4) = abs(log(exp_tau_r / ising_tau_r));
        end
    end
end

% Normalize each spoke to [0,1] range (0 = perfect match)
radarNorm = radarRaw;
for s = 1:nSpokes
    col = radarRaw(:, s);
    validCol = col(~isnan(col));
    if ~isempty(validCol) && max(validCol) > 0
        radarNorm(:, s) = col / max(validCol);
    end
end

% Plot using polarplot
theta = linspace(0, 2*pi, nSpokes + 1);  % Close the polygon
hold on;

for c = 1:nConditions
    condition = config.conditions{c};
    if all(isnan(radarNorm(c, :))); continue; end

    vals = radarNorm(c, :);
    vals(isnan(vals)) = 0;
    vals_closed = [vals, vals(1)];  % Close polygon

    polarplot(theta, vals_closed, '-o', 'Color', config.colors.(condition), ...
        'LineWidth', 1.0, 'MarkerSize', 5, 'MarkerFaceColor', config.colors.(condition), ...
        'DisplayName', condition);
end

% Configure polar axes
pax = gca;
pax.ThetaTick = rad2deg(theta(1:end-1));
pax.ThetaTickLabel = radarLabels;
pax.RLim = [0 1.1];
pax.RTick = [0 0.25 0.5 0.75 1.0];
title('Multi-Observable Match Quality (0 = perfect)');
legend('Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 6);
hold off;

saveMyFig('Radar_MatchQuality', fullfile(config.outputPath, 'Summary'), gcf);

% -------------------------------------------------------------------------
% Figure 5d: Phase Portrait (Activity vs Moran's I)
% -------------------------------------------------------------------------
fprintf('Creating phase portrait (Activity vs Moran''s I)...\n');

figure('Name', 'Phase Portrait: Activity vs Moran''s I');
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [24 8]);

for c = 1:nConditions
    condition = config.conditions{c};

    if ~isfield(ExpStats, condition) || ~isfield(Comparison, condition)
        continue;
    end
    if ~isfield(ExpStats.(condition), 'Activity_all') || ~isfield(ExpStats.(condition), 'MoransI_all')
        continue;
    end

    subplot(1, nConditions, c);
    hold on;

    % Experimental data
    exp_activity = ExpStats.(condition).Activity_all;
    exp_moransI = ExpStats.(condition).MoransI_all;

    % Ensure same length (both are flattened trial x time)
    nPts = min(length(exp_activity), length(exp_moransI));
    exp_a = exp_activity(1:nPts);
    exp_m = exp_moransI(1:nPts);

    % Remove NaN pairs
    valid = ~isnan(exp_a) & ~isnan(exp_m);
    exp_a = exp_a(valid);
    exp_m = exp_m(valid);

    % Experimental 2D density contours
    if length(exp_a) > 50
        try
            [f_exp_2d, xi_exp_2d] = ksdensity([exp_a(:), exp_m(:)], ...
                'Bandwidth', [], 'Function', 'pdf');
            % ksdensity 2D returns on a grid; use scatter with color for density
            % Alternative: contour approach with gridded data
            nGrid = 50;
            x_edges = linspace(min(exp_a), max(exp_a), nGrid);
            y_edges = linspace(min(exp_m), max(exp_m), nGrid);
            [X, Y] = meshgrid(x_edges, y_edges);
            pts = [X(:), Y(:)];
            f_grid = ksdensity([exp_a(:), exp_m(:)], pts);
            F = reshape(f_grid, nGrid, nGrid);
            contour(X, Y, F, 5, 'LineWidth', 0.75, 'LineColor', config.colors.(condition));
        catch
            % Fallback: simple scatter
            scatter(exp_a, exp_m, 4, config.colors.(condition), 'filled', ...
                'MarkerFaceAlpha', 0.1);
        end
    end

    % Best Ising match data
    best_idx = Comparison.(condition).bestMatch_idx(1);
    ising_activity = IsingData.Activity_all{best_idx};
    ising_moransI = IsingData.MoransI_all{best_idx};
    nPts_i = min(length(ising_activity), length(ising_moransI));
    ising_a = ising_activity(1:nPts_i);
    ising_m = ising_moransI(1:nPts_i);
    valid_i = ~isnan(ising_a) & ~isnan(ising_m);
    ising_a = ising_a(valid_i);
    ising_m = ising_m(valid_i);

    % Ising 2D density contours (gray)
    if length(ising_a) > 50
        try
            nGrid = 50;
            x_edges_i = linspace(min(ising_a), max(ising_a), nGrid);
            y_edges_i = linspace(min(ising_m), max(ising_m), nGrid);
            [Xi, Yi] = meshgrid(x_edges_i, y_edges_i);
            pts_i = [Xi(:), Yi(:)];
            f_grid_i = ksdensity([ising_a(:), ising_m(:)], pts_i);
            Fi = reshape(f_grid_i, nGrid, nGrid);
            contour(Xi, Yi, Fi, 5, 'LineWidth', 0.75, 'LineColor', [0.5 0.5 0.5], 'LineStyle', '--');
        catch
            scatter(ising_a, ising_m, 4, [0.5 0.5 0.5], 'filled', ...
                'MarkerFaceAlpha', 0.1);
        end
    end

    xlabel('Activity');
    ylabel('Moran''s I');
    title(condition);
    hold off;
end

sgtitle('Phase Portrait: Activity vs Moran''s I (colored = Exp, gray = Ising)');
saveMyFig('PhasePortrait_Activity_MoransI', fullfile(config.outputPath, 'TemporalDynamics'), gcf);

% -------------------------------------------------------------------------
% Figure 5e: Goodness-of-Fit Summary Table
% -------------------------------------------------------------------------
fprintf('Creating goodness-of-fit summary table...\n');

figure('Name', 'Goodness-of-Fit Summary Table');
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [18 10]);
axis off;

% Build table header
colHeaders = {'Condition', 'WD(MI)', 'WD(Act)', 'WD(comb)', 'tau ratio', 'KS p(MI)'};
nCols = length(colHeaders);

% Starting position
yStart = 0.92;
yStep = 0.12;
xPositions = linspace(0.02, 0.90, nCols);

% Draw header
for col = 1:nCols
    text(xPositions(col), yStart, colHeaders{col}, 'FontName', 'FixedWidth', ...
        'FontSize', 7, 'FontWeight', 'bold', 'Units', 'normalized');
end

% Separator line
annotation('line', [0.02 0.98], [yStart - 0.03, yStart - 0.03], 'LineWidth', 0.5);

% Fill rows
for c = 1:nConditions
    condition = config.conditions{c};
    yPos = yStart - yStep * c;

    if ~isfield(Comparison, condition) || ~isfield(ExpStats, condition)
        text(xPositions(1), yPos, condition, 'FontName', 'FixedWidth', 'FontSize', 7, ...
            'Units', 'normalized', 'Color', config.colors.(condition));
        continue;
    end

    best_idx = Comparison.(condition).bestMatch_idx(1);

    % Column 1: Condition name
    text(xPositions(1), yPos, condition, 'FontName', 'FixedWidth', 'FontSize', 7, ...
        'Units', 'normalized', 'Color', config.colors.(condition), 'FontWeight', 'bold');

    % Column 2: WD(MoransI)
    if isfield(Comparison.(condition), 'dist_moransI_raw')
        val = Comparison.(condition).dist_moransI_raw(best_idx);
        text(xPositions(2), yPos, sprintf('%.4f', val), 'FontName', 'FixedWidth', ...
            'FontSize', 7, 'Units', 'normalized');
    else
        text(xPositions(2), yPos, '-', 'FontName', 'FixedWidth', 'FontSize', 7, 'Units', 'normalized');
    end

    % Column 3: WD(Activity)
    if isfield(Comparison.(condition), 'dist_activity_raw')
        val = Comparison.(condition).dist_activity_raw(best_idx);
        text(xPositions(3), yPos, sprintf('%.4f', val), 'FontName', 'FixedWidth', ...
            'FontSize', 7, 'Units', 'normalized');
    else
        text(xPositions(3), yPos, '-', 'FontName', 'FixedWidth', 'FontSize', 7, 'Units', 'normalized');
    end

    % Column 4: WD(combined)
    val = Comparison.(condition).wasserstein_dist(best_idx);
    text(xPositions(4), yPos, sprintf('%.4f', val), 'FontName', 'FixedWidth', ...
        'FontSize', 7, 'Units', 'normalized');

    % Column 5: tau ratio
    if isfield(DynamicsAnalysis, condition) && isfield(DynamicsAnalysis.(condition), 'exp_tau')
        exp_tau_t = DynamicsAnalysis.(condition).exp_tau;
        ising_tau_t = DynamicsAnalysis.(condition).ising_tau;
        if ~isnan(exp_tau_t) && ~isnan(ising_tau_t) && ising_tau_t > 0
            text(xPositions(5), yPos, sprintf('%.2f', exp_tau_t / ising_tau_t), ...
                'FontName', 'FixedWidth', 'FontSize', 7, 'Units', 'normalized');
        else
            text(xPositions(5), yPos, 'NaN', 'FontName', 'FixedWidth', 'FontSize', 7, 'Units', 'normalized');
        end
    else
        text(xPositions(5), yPos, '-', 'FontName', 'FixedWidth', 'FontSize', 7, 'Units', 'normalized');
    end

    % Column 6: KS test p-value (Moran's I)
    exp_mi = ExpStats.(condition).MoransI_all;
    exp_mi = exp_mi(~isnan(exp_mi));
    ising_mi = IsingData.MoransI_all{best_idx};
    ising_mi = ising_mi(~isnan(ising_mi));
    if ~isempty(exp_mi) && ~isempty(ising_mi)
        [~, ks_p] = kstest2(exp_mi, ising_mi);
        if ks_p < 0.001
            text(xPositions(6), yPos, sprintf('%.1e', ks_p), 'FontName', 'FixedWidth', ...
                'FontSize', 7, 'Units', 'normalized');
        else
            text(xPositions(6), yPos, sprintf('%.4f', ks_p), 'FontName', 'FixedWidth', ...
                'FontSize', 7, 'Units', 'normalized');
        end
    else
        text(xPositions(6), yPos, '-', 'FontName', 'FixedWidth', 'FontSize', 7, 'Units', 'normalized');
    end
end

title('Goodness-of-Fit Summary', 'FontWeight', 'bold');
saveMyFig('GoodnessOfFit_SummaryTable', fullfile(config.outputPath, 'Summary'), gcf);

% -------------------------------------------------------------------------
% Figure 6: UMAP of Ising Parameter Space with Best Matches Highlighted
% -------------------------------------------------------------------------
figure('Name', 'UMAP: Ising Parameter Space');

% Create feature matrix for UMAP: [nSims x 5 parameters]
featureMatrix = [IsingData.params.beta, IsingData.params.c, ...
                 IsingData.params.decay_const, IsingData.params.inhibition_range, ...
                 IsingData.params.bias];

% Normalize features for UMAP (z-score each parameter)
featureMatrix_norm = (featureMatrix - mean(featureMatrix, 1)) ./ std(featureMatrix, 0, 1);

% Run UMAP
fprintf('Running UMAP on %d simulations with 5 parameters...\n', size(featureMatrix_norm, 1));
try
    [embedding, ~, ~] = run_umap(double(featureMatrix_norm), ...
        'n_neighbors', 30, ...
        'min_dist', 0.3, ...
        'n_components', 2, ...
        'metric', 'euclidean', ...
        'verbose', false);

    % Plot all simulations in gray
    scatter(embedding(:,1), embedding(:,2), 20, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.3);
    hold on;

    % Highlight best matches for each condition
    for c = 1:length(config.conditions)
        condition = config.conditions{c};
        if isfield(Comparison, condition)
            best_idx = Comparison.(condition).bestMatch_idx(1:3);
            conditionColor = config.colors.(condition);  % Get color from config
            scatter(embedding(best_idx, 1), embedding(best_idx, 2), 100, ...
                conditionColor, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
                'DisplayName', condition);
        end
    end

    hold off;
    xlabel('UMAP 1');
    ylabel('UMAP 2');
    title('Ising Parameter Space (UMAP) with Best Matches per Condition');
    legend('Location', 'best');

    % Save figure
    saveMyFig('UMAP_ParameterSpace', fullfile(config.outputPath, 'UMAP'), gcf);

    % Store embedding in results
    Results.UMAP_embedding = embedding;
    Results.UMAP_featureMatrix = featureMatrix;

    % --- Top 10 version of same UMAP (using helper) ---
    plotUMAPComparison(embedding, [], Comparison, ...
        config.conditions, config.colors, 10, ...
        'Ising Parameter Space (Top 10)', 'UMAP', config.outputPath);

catch ME
    warning('UMAP failed: %s. Skipping UMAP figure.', ME.message);
end

% -------------------------------------------------------------------------
% Figure 7: UMAP of Moran's I Statistics (Ising + Experimental)
% Always generated regardless of matchingMetric
% -------------------------------------------------------------------------
figure('Name', 'UMAP: Moran''s I Feature Space');

fprintf('Computing Moran''s I statistics for UMAP feature space...\n');

try
    % Compute Moran's I statistics for each Ising simulation
    nSims = length(IsingData.simIDs);
    isingFeatures = zeros(nSims, 6);
    MoransI_all = IsingData.MoransI_all;
    parfor s = 1:nSims
        moransI_ts = MoransI_all{s};
        moransI_ts = moransI_ts(~isnan(moransI_ts));  % Remove NaNs
        if ~isempty(moransI_ts)
            isingFeatures(s, :) = [mean(moransI_ts), std(moransI_ts), median(moransI_ts), ...
                                   skewness(moransI_ts), kurtosis(moransI_ts), iqr(moransI_ts)];
        end
    end

    % Compute same statistics for each experimental condition
    nConds = length(config.conditions);
    expFeatures = zeros(nConds, 6);
    for c = 1:nConds
        condition = config.conditions{c};
        if isfield(ExpStats, condition)
            moransI_exp = ExpStats.(condition).MoransI_all;
            moransI_exp = moransI_exp(~isnan(moransI_exp));
            if ~isempty(moransI_exp)
                expFeatures(c, 1) = mean(moransI_exp);
                expFeatures(c, 2) = std(moransI_exp);
                expFeatures(c, 3) = median(moransI_exp);
                expFeatures(c, 4) = skewness(moransI_exp);
                expFeatures(c, 5) = kurtosis(moransI_exp);
                expFeatures(c, 6) = iqr(moransI_exp);
            end
        end
    end

    % Combine all features
    allFeatures = [isingFeatures; expFeatures];

    % Normalize features (z-score)
    allFeatures_norm = (allFeatures - mean(allFeatures, 1)) ./ std(allFeatures, 0, 1);

    % Run UMAP on combined data
    fprintf('Running UMAP on %d simulations + %d conditions...\n', nSims, nConds);
    [embedding_combined, ~, ~] = run_umap(double(allFeatures_norm), ...
        'n_neighbors', 199, 'min_dist', 0.3, 'n_components', 2, ...
        'metric', 'euclidean', 'verbose', false);

    % Split embedding back
    embedding_ising = embedding_combined(1:nSims, :);
    embedding_exp = embedding_combined(nSims+1:end, :);

    % Plot Ising simulations in gray
    scatter(embedding_ising(:,1), embedding_ising(:,2), 100, [0.7 0.7 0.7], ...
        'filled', 'MarkerFaceAlpha', 0.4, 'HandleVisibility', 'off');
    hold on;

    % Highlight top 3 best-matching Ising simulations for each condition
    for c = 1:nConds
        condition = config.conditions{c};
        if isfield(Comparison, condition)
            best_idx = Comparison.(condition).bestMatch_idx(1:3);
            conditionColor = config.colors.(condition);
            scatter(embedding_ising(best_idx, 1), embedding_ising(best_idx, 2), 100, ...
                conditionColor, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
                'DisplayName', [condition ' (Top 3 Ising)']);
        end
    end

    % Plot experimental conditions with their colors (larger markers, star shape)
    for c = 1:nConds
        condition = config.conditions{c};
        conditionColor = config.colors.(condition);
        scatter(embedding_exp(c, 1), embedding_exp(c, 2), 300, conditionColor, ...
            'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2, ...
            'DisplayName', [condition ' (Exp)']);
    end

    hold off;
    xlabel('UMAP 1');
    ylabel('UMAP 2');
    title('Moran''s I Feature Space: Ising Simulations + Experimental Conditions');
    legend('Location', 'best');

    % Save figure
    saveMyFig('UMAP_MoransI_FeatureSpace', fullfile(config.outputPath, 'UMAP'), gcf);

    % Store in results
    Results.UMAP_MoransI.embedding_ising = embedding_ising;
    Results.UMAP_MoransI.embedding_exp = embedding_exp;
    Results.UMAP_MoransI.isingFeatures = isingFeatures;
    Results.UMAP_MoransI.expFeatures = expFeatures;

    % --- Top 10 version of same UMAP (using helper) ---
    plotUMAPComparison(embedding_ising, embedding_exp, Comparison, ...
        config.conditions, config.colors, 10, ...
        'Moran''s I Feature Space: Top 10 Matches', 'UMAP', config.outputPath);

catch ME
    warning('UMAP (Moran''s I feature space) failed: %s', ME.message);
end

% -------------------------------------------------------------------------
% Figure 7b: UMAP of Activity Statistics (Ising + Experimental)
% Always generated regardless of matchingMetric
% -------------------------------------------------------------------------
figure('Name', 'UMAP: Activity Feature Space');

fprintf('Computing Activity statistics for UMAP feature space...\n');

try
    % Compute Activity statistics for each Ising simulation
    nSims = length(IsingData.simIDs);
    isingActivityFeatures = zeros(nSims, 6);
    Activity_all = IsingData.Activity_all;
    parfor s = 1:nSims
        activity_ts = Activity_all{s};
        activity_ts = activity_ts(~isnan(activity_ts));  % Remove NaNs
        if ~isempty(activity_ts)
            isingActivityFeatures(s, :) = [mean(activity_ts), std(activity_ts), median(activity_ts), ...
                                           skewness(activity_ts), kurtosis(activity_ts), iqr(activity_ts)];
        end
    end

    % Compute same statistics for each experimental condition
    nConds = length(config.conditions);
    expActivityFeatures = zeros(nConds, 6);
    for c = 1:nConds
        condition = config.conditions{c};
        if isfield(ExpStats, condition) && isfield(ExpStats.(condition), 'Activity_all')
            activity_exp = ExpStats.(condition).Activity_all;
            activity_exp = activity_exp(~isnan(activity_exp));
            if ~isempty(activity_exp)
                expActivityFeatures(c, 1) = mean(activity_exp);
                expActivityFeatures(c, 2) = std(activity_exp);
                expActivityFeatures(c, 3) = median(activity_exp);
                expActivityFeatures(c, 4) = skewness(activity_exp);
                expActivityFeatures(c, 5) = kurtosis(activity_exp);
                expActivityFeatures(c, 6) = iqr(activity_exp);
            end
        end
    end

    % Combine all features
    allActivityFeatures = [isingActivityFeatures; expActivityFeatures];

    % Normalize features (z-score)
    allActivityFeatures_norm = (allActivityFeatures - mean(allActivityFeatures, 1)) ./ std(allActivityFeatures, 0, 1);

    % Run UMAP on combined data
    fprintf('Running UMAP on %d simulations + %d conditions (Activity)...\n', nSims, nConds);
    [embedding_activity_combined, ~, ~] = run_umap(double(allActivityFeatures_norm), ...
        'n_neighbors', 199, 'min_dist', 0.3, 'n_components', 2, ...
        'metric', 'euclidean', 'verbose', false);

    % Split embedding back
    embedding_activity_ising = embedding_activity_combined(1:nSims, :);
    embedding_activity_exp = embedding_activity_combined(nSims+1:end, :);

    % Plot Ising simulations in gray
    scatter(embedding_activity_ising(:,1), embedding_activity_ising(:,2), 100, [0.7 0.7 0.7], ...
        'filled', 'MarkerFaceAlpha', 0.4, 'HandleVisibility', 'off');
    hold on;

    % Highlight top 3 best-matching Ising simulations for each condition
    for c = 1:nConds
        condition = config.conditions{c};
        if isfield(Comparison, condition)
            best_idx = Comparison.(condition).bestMatch_idx(1:3);
            conditionColor = config.colors.(condition);
            scatter(embedding_activity_ising(best_idx, 1), embedding_activity_ising(best_idx, 2), 100, ...
                conditionColor, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
                'DisplayName', [condition ' (Top 3 Ising)']);
        end
    end

    % Plot experimental conditions with their colors (larger markers, star shape)
    for c = 1:nConds
        condition = config.conditions{c};
        conditionColor = config.colors.(condition);
        scatter(embedding_activity_exp(c, 1), embedding_activity_exp(c, 2), 300, conditionColor, ...
            'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2, ...
            'DisplayName', [condition ' (Exp)']);
    end

    hold off;
    xlabel('UMAP 1');
    ylabel('UMAP 2');
    title('Activity Feature Space: Ising Simulations + Experimental Conditions');
    legend('Location', 'best');

    % Save figure
    saveMyFig('UMAP_Activity_FeatureSpace', fullfile(config.outputPath, 'UMAP'), gcf);

    % Store in results
    Results.UMAP_Activity.embedding_ising = embedding_activity_ising;
    Results.UMAP_Activity.embedding_exp = embedding_activity_exp;
    Results.UMAP_Activity.isingFeatures = isingActivityFeatures;
    Results.UMAP_Activity.expFeatures = expActivityFeatures;

    % --- Top 10 version of same UMAP (using helper) ---
    plotUMAPComparison(embedding_activity_ising, embedding_activity_exp, Comparison, ...
        config.conditions, config.colors, 10, ...
        'Activity Feature Space: Top 10 Matches', 'UMAP', config.outputPath);

catch ME
    warning('UMAP (Activity feature space) failed: %s', ME.message);
end

% -------------------------------------------------------------------------
% Figure 7b2: UMAP of Moran's I + Activity Features Combined
% Always generated regardless of matchingMetric
% -------------------------------------------------------------------------
figure('Name', 'UMAP: Moran''s I + Activity Feature Space');

fprintf('Computing Moran''s I + Activity combined features for UMAP...\n');

try
    % Feature counts: 6 Moran's I + 6 Activity = 12 total
    nSims = length(IsingData.simIDs);
    nConds = length(config.conditions);

    isingMoransActivityFeatures = zeros(nSims, 12);
    expMoransActivityFeatures = zeros(nConds, 12);

    % Extract data for parfor compatibility
    MoransI_all = IsingData.MoransI_all;
    Activity_all = IsingData.Activity_all;

    % Compute Ising features
    parfor s = 1:nSims
        moransI_ts = MoransI_all{s};
        moransI_ts = moransI_ts(~isnan(moransI_ts));
        activity_ts = Activity_all{s};
        activity_ts = activity_ts(~isnan(activity_ts));

        feat = zeros(1, 12);
        if ~isempty(moransI_ts)
            feat(1:6) = [mean(moransI_ts), std(moransI_ts), median(moransI_ts), ...
                         skewness(moransI_ts), kurtosis(moransI_ts), iqr(moransI_ts)];
        end
        if ~isempty(activity_ts)
            feat(7:12) = [mean(activity_ts), std(activity_ts), median(activity_ts), ...
                          skewness(activity_ts), kurtosis(activity_ts), iqr(activity_ts)];
        end
        isingMoransActivityFeatures(s, :) = feat;
    end

    % Compute experimental features
    for c = 1:nConds
        condition = config.conditions{c};
        if isfield(ExpStats, condition)
            % Moran's I features
            if isfield(ExpStats.(condition), 'MoransI_all')
                moransI_exp = ExpStats.(condition).MoransI_all;
                moransI_exp = moransI_exp(~isnan(moransI_exp));
                if ~isempty(moransI_exp)
                    expMoransActivityFeatures(c, 1:6) = [mean(moransI_exp), std(moransI_exp), median(moransI_exp), ...
                                                         skewness(moransI_exp), kurtosis(moransI_exp), iqr(moransI_exp)];
                end
            end
            % Activity features
            if isfield(ExpStats.(condition), 'Activity_all')
                activity_exp = ExpStats.(condition).Activity_all;
                activity_exp = activity_exp(~isnan(activity_exp));
                if ~isempty(activity_exp)
                    expMoransActivityFeatures(c, 7:12) = [mean(activity_exp), std(activity_exp), median(activity_exp), ...
                                                          skewness(activity_exp), kurtosis(activity_exp), iqr(activity_exp)];
                end
            end
        end
    end

    % Combine all features
    allMoransActivityFeatures = [isingMoransActivityFeatures; expMoransActivityFeatures];

    % Normalize features (z-score)
    allMoransActivityFeatures_norm = (allMoransActivityFeatures - mean(allMoransActivityFeatures, 1)) ./ std(allMoransActivityFeatures, 0, 1);
    allMoransActivityFeatures_norm(isnan(allMoransActivityFeatures_norm)) = 0;

    % Run UMAP on combined data
    fprintf('Running UMAP on %d simulations + %d conditions (Moran''s I + Activity: 12 features)...\n', nSims, nConds);
    [embedding_moransActivity_all, ~, ~] = run_umap(double(allMoransActivityFeatures_norm), ...
        'n_neighbors', 199, 'min_dist', 0.3, 'n_components', 2, ...
        'metric', 'euclidean', 'verbose', false);

    % Split embedding back
    embedding_moransActivity_ising = embedding_moransActivity_all(1:nSims, :);
    embedding_moransActivity_exp = embedding_moransActivity_all(nSims+1:end, :);

    % Plot Ising simulations in gray
    scatter(embedding_moransActivity_ising(:,1), embedding_moransActivity_ising(:,2), 100, [0.7 0.7 0.7], ...
        'filled', 'MarkerFaceAlpha', 0.4, 'HandleVisibility', 'off');
    hold on;

    % Highlight top 3 best-matching Ising simulations for each condition
    for c = 1:nConds
        condition = config.conditions{c};
        if isfield(Comparison, condition)
            best_idx = Comparison.(condition).bestMatch_idx(1:min(3, length(Comparison.(condition).bestMatch_idx)));
            conditionColor = config.colors.(condition);
            scatter(embedding_moransActivity_ising(best_idx, 1), embedding_moransActivity_ising(best_idx, 2), 150, ...
                conditionColor, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
                'DisplayName', [condition ' (Ising)']);
        end
    end

    % Plot experimental conditions as large squares
    for c = 1:nConds
        condition = config.conditions{c};
        conditionColor = config.colors.(condition);
        scatter(embedding_moransActivity_exp(c, 1), embedding_moransActivity_exp(c, 2), 300, ...
            conditionColor, 's', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2, ...
            'DisplayName', [condition ' (Exp)']);
    end

    hold off;
    xlabel('UMAP 1');
    ylabel('UMAP 2');
    title('Moran''s I + Activity Feature Space (Ising vs Experimental)');
    legend('Location', 'best');

    % Save figure
    saveMyFig('UMAP_MoransI_Activity_FeatureSpace', fullfile(config.outputPath, 'UMAP'), gcf);

    % Store results
    Results.UMAP_MoransActivity.embedding_ising = embedding_moransActivity_ising;
    Results.UMAP_MoransActivity.embedding_exp = embedding_moransActivity_exp;
    Results.UMAP_MoransActivity.isingFeatures = isingMoransActivityFeatures;
    Results.UMAP_MoransActivity.expFeatures = expMoransActivityFeatures;
    Results.UMAP_MoransActivity.featureNames = {'MoransI_mean', 'MoransI_std', 'MoransI_median', ...
        'MoransI_skewness', 'MoransI_kurtosis', 'MoransI_iqr', ...
        'Activity_mean', 'Activity_std', 'Activity_median', ...
        'Activity_skewness', 'Activity_kurtosis', 'Activity_iqr'};

    % --- Top 10 version of same UMAP (using helper) ---
    plotUMAPComparison(embedding_moransActivity_ising, embedding_moransActivity_exp, Comparison, ...
        config.conditions, config.colors, 10, ...
        'Moran''s I + Activity Feature Space (Top 10)', 'UMAP', config.outputPath);

catch ME
    warning('UMAP (Moran''s I + Activity feature space) failed: %s', ME.message);
end

% -------------------------------------------------------------------------
% Figure 7c: UMAP of Combined Features (Moran's I + Activity + Autocorr)
% Only generated when matching more than one metric
% -------------------------------------------------------------------------
if ismember(config.matchingMetric, {'combined', 'moransI+activity', 'spatial+persistence'})
figure('Name', 'UMAP: Combined Feature Space');

fprintf('Computing Combined features (Moran''s I + Activity + Autocorr) for UMAP...\n');

try
    % Feature counts: 6 Moran's I + 6 Activity + 2 Autocorr = 14 total
    nSims = length(IsingData.simIDs);
    nConds = length(config.conditions);

    isingCombinedFeatures = zeros(nSims, 14);
    expCombinedFeatures = zeros(nConds, 14);

    % Extract data for parfor compatibility
    MoransI_all = IsingData.MoransI_all;
    Activity_all = IsingData.Activity_all;
    hasAutocorrTau = isfield(IsingData, 'Autocorr_tau') && ~isempty(IsingData.Autocorr_tau);
    hasAutocorrR2 = isfield(IsingData, 'Autocorr_fitR2') && ~isempty(IsingData.Autocorr_fitR2);
    if hasAutocorrTau
        Autocorr_tau = IsingData.Autocorr_tau;
    else
        Autocorr_tau = [];
    end
    if hasAutocorrR2
        Autocorr_fitR2 = IsingData.Autocorr_fitR2;
    else
        Autocorr_fitR2 = [];
    end

    % Compute features for each Ising simulation
    parfor s = 1:nSims
        features = zeros(1, 14);

        % Moran's I features (1-6)
        moransI_ts = MoransI_all{s};
        moransI_ts = moransI_ts(~isnan(moransI_ts));
        if ~isempty(moransI_ts)
            features(1:6) = [mean(moransI_ts), std(moransI_ts), median(moransI_ts), ...
                             skewness(moransI_ts), kurtosis(moransI_ts), iqr(moransI_ts)];
        end

        % Activity features (7-12)
        activity_ts = Activity_all{s};
        activity_ts = activity_ts(~isnan(activity_ts));
        if ~isempty(activity_ts)
            features(7:12) = [mean(activity_ts), std(activity_ts), median(activity_ts), ...
                              skewness(activity_ts), kurtosis(activity_ts), iqr(activity_ts)];
        end

        % Autocorrelation features (13-14)
        if hasAutocorrTau && length(Autocorr_tau) >= s
            features(13) = Autocorr_tau(s);
        end
        if hasAutocorrR2 && length(Autocorr_fitR2) >= s
            features(14) = Autocorr_fitR2(s);
        end

        isingCombinedFeatures(s, :) = features;
    end

    % Compute features for each experimental condition
    for c = 1:nConds
        condition = config.conditions{c};
        if ~isfield(ExpStats, condition)
            continue;
        end

        % Moran's I features (1-6)
        if isfield(ExpStats.(condition), 'MoransI_all')
            moransI_exp = ExpStats.(condition).MoransI_all;
            moransI_exp = moransI_exp(~isnan(moransI_exp));
            if ~isempty(moransI_exp)
                expCombinedFeatures(c, 1) = mean(moransI_exp);
                expCombinedFeatures(c, 2) = std(moransI_exp);
                expCombinedFeatures(c, 3) = median(moransI_exp);
                expCombinedFeatures(c, 4) = skewness(moransI_exp);
                expCombinedFeatures(c, 5) = kurtosis(moransI_exp);
                expCombinedFeatures(c, 6) = iqr(moransI_exp);
            end
        end

        % Activity features (7-12)
        if isfield(ExpStats.(condition), 'Activity_all')
            activity_exp = ExpStats.(condition).Activity_all;
            activity_exp = activity_exp(~isnan(activity_exp));
            if ~isempty(activity_exp)
                expCombinedFeatures(c, 7) = mean(activity_exp);
                expCombinedFeatures(c, 8) = std(activity_exp);
                expCombinedFeatures(c, 9) = median(activity_exp);
                expCombinedFeatures(c, 10) = skewness(activity_exp);
                expCombinedFeatures(c, 11) = kurtosis(activity_exp);
                expCombinedFeatures(c, 12) = iqr(activity_exp);
            end
        end

        % Autocorrelation features (13-14)
        if isfield(ExpStats.(condition), 'Autocorr') && isfield(ExpStats.(condition).Autocorr, 'tau')
            expCombinedFeatures(c, 13) = ExpStats.(condition).Autocorr.tau;
        end
        if isfield(ExpStats.(condition), 'Autocorr') && isfield(ExpStats.(condition).Autocorr, 'fitResult')
            expCombinedFeatures(c, 14) = ExpStats.(condition).Autocorr.fitResult.R2;
        end
    end

    % Combine all features
    allCombinedFeatures = [isingCombinedFeatures; expCombinedFeatures];

    % Handle NaN values: replace with column mean before normalization
    for col = 1:size(allCombinedFeatures, 2)
        col_data = allCombinedFeatures(:, col);
        nan_mask = isnan(col_data);
        if any(nan_mask) && ~all(nan_mask)
            col_data(nan_mask) = mean(col_data(~nan_mask));
            allCombinedFeatures(:, col) = col_data;
        end
    end

    % Normalize features (z-score)
    allCombinedFeatures_norm = (allCombinedFeatures - mean(allCombinedFeatures, 1)) ./ std(allCombinedFeatures, 0, 1);

    % Handle any remaining NaN/Inf from zero std columns
    allCombinedFeatures_norm(isnan(allCombinedFeatures_norm) | isinf(allCombinedFeatures_norm)) = 0;

    % Run UMAP on combined data
    fprintf('Running UMAP on %d simulations + %d conditions (Combined: 14 features)...\n', nSims, nConds);
    [embedding_combined_all, ~, ~] = run_umap(double(allCombinedFeatures_norm), ...
        'n_neighbors', 199, 'min_dist', 0.3, 'n_components', 2, ...
        'metric', 'euclidean', 'verbose', false);

    % Split embedding back
    embedding_combined_ising = embedding_combined_all(1:nSims, :);
    embedding_combined_exp = embedding_combined_all(nSims+1:end, :);

    % Plot Ising simulations in gray
    scatter(embedding_combined_ising(:,1), embedding_combined_ising(:,2), 100, [0.7 0.7 0.7], ...
        'filled', 'MarkerFaceAlpha', 0.4, 'HandleVisibility', 'off');
    hold on;

    % Highlight top 3 best-matching Ising simulations for each condition
    for c = 1:nConds
        condition = config.conditions{c};
        if isfield(Comparison, condition)
            best_idx = Comparison.(condition).bestMatch_idx(1:3);
            conditionColor = config.colors.(condition);
            scatter(embedding_combined_ising(best_idx, 1), embedding_combined_ising(best_idx, 2), 100, ...
                conditionColor, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
                'DisplayName', [condition ' (Top 3 Ising)']);
        end
    end

    % Plot experimental conditions with their colors (larger markers, star shape)
    for c = 1:nConds
        condition = config.conditions{c};
        conditionColor = config.colors.(condition);
        scatter(embedding_combined_exp(c, 1), embedding_combined_exp(c, 2), 300, conditionColor, ...
            'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2, ...
            'DisplayName', [condition ' (Exp)']);
    end

    hold off;
    xlabel('UMAP 1');
    ylabel('UMAP 2');
    title('Combined Feature Space: Moran''s I + Activity + Autocorr');
    legend('Location', 'best');

    % Save figure
    saveMyFig('UMAP_Combined_FeatureSpace', fullfile(config.outputPath, 'UMAP'), gcf);

    % Store in results
    Results.UMAP_Combined.embedding_ising = embedding_combined_ising;
    Results.UMAP_Combined.embedding_exp = embedding_combined_exp;
    Results.UMAP_Combined.isingFeatures = isingCombinedFeatures;
    Results.UMAP_Combined.expFeatures = expCombinedFeatures;
    Results.UMAP_Combined.featureNames = {'MoransI_mean', 'MoransI_std', 'MoransI_median', ...
        'MoransI_skewness', 'MoransI_kurtosis', 'MoransI_iqr', ...
        'Activity_mean', 'Activity_std', 'Activity_median', ...
        'Activity_skewness', 'Activity_kurtosis', 'Activity_iqr', ...
        'Autocorr_tau', 'Autocorr_fitR2'};

    % --- Top 10 version of same UMAP (using helper) ---
    plotUMAPComparison(embedding_combined_ising, embedding_combined_exp, Comparison, ...
        config.conditions, config.colors, 10, ...
        'Combined Feature Space: Top 10 Matches', 'UMAP', config.outputPath);

catch ME
    warning('UMAP (Combined feature space) failed: %s', ME.message);
end
end  % end if combined

% -------------------------------------------------------------------------
% Figure 7d: UMAP of Spatial+Persistence Features (Moran's I + Activity + BlobPersistence)
% Only generated when matching more than one metric
% -------------------------------------------------------------------------
if ismember(config.matchingMetric, {'combined', 'spatial+persistence'})
figure('Name', 'UMAP: Spatial+Persistence Feature Space');

fprintf('Computing Spatial+Persistence features (Moran''s I + Activity + BlobPersistence) for UMAP...\n');

try
    % Feature counts: 6 Moran's I + 6 Activity + 6 BlobPersistence = 18 total
    nSims = length(IsingData.simIDs);
    nConds = length(config.conditions);

    isingSpatialPersistFeatures = zeros(nSims, 18);
    expSpatialPersistFeatures = zeros(nConds, 18);

    % Extract data for parfor compatibility
    MoransI_all = IsingData.MoransI_all;
    Activity_all = IsingData.Activity_all;
    hasBlobPersistence = isfield(IsingData, 'BlobPersistence_lifetimes') && ~isempty(IsingData.BlobPersistence_lifetimes);
    if hasBlobPersistence
        BlobPersistence_lifetimes = IsingData.BlobPersistence_lifetimes;
    else
        BlobPersistence_lifetimes = cell(nSims, 1);
    end

    % Compute features for each Ising simulation
    parfor s = 1:nSims
        features = zeros(1, 18);

        % Moran's I features (1-6)
        moransI_ts = MoransI_all{s};
        moransI_ts = moransI_ts(~isnan(moransI_ts));
        if ~isempty(moransI_ts)
            features(1:6) = [mean(moransI_ts), std(moransI_ts), median(moransI_ts), ...
                             skewness(moransI_ts), kurtosis(moransI_ts), iqr(moransI_ts)];
        end

        % Activity features (7-12)
        activity_ts = Activity_all{s};
        activity_ts = activity_ts(~isnan(activity_ts));
        if ~isempty(activity_ts)
            features(7:12) = [mean(activity_ts), std(activity_ts), median(activity_ts), ...
                              skewness(activity_ts), kurtosis(activity_ts), iqr(activity_ts)];
        end

        % Blob Persistence features (13-18)
        if hasBlobPersistence && ~isempty(BlobPersistence_lifetimes{s})
            lifetimes = BlobPersistence_lifetimes{s};
            lifetimes = lifetimes(~isnan(lifetimes));
            if length(lifetimes) >= 2
                features(13:18) = [mean(lifetimes), std(lifetimes), median(lifetimes), ...
                                   skewness(lifetimes), kurtosis(lifetimes), iqr(lifetimes)];
            elseif ~isempty(lifetimes)
                features(13) = mean(lifetimes);
            end
        end

        isingSpatialPersistFeatures(s, :) = features;
    end

    % Compute features for each experimental condition
    for c = 1:nConds
        condition = config.conditions{c};
        if ~isfield(ExpStats, condition)
            continue;
        end

        % Moran's I features (1-6)
        if isfield(ExpStats.(condition), 'MoransI_all')
            moransI_exp = ExpStats.(condition).MoransI_all;
            moransI_exp = moransI_exp(~isnan(moransI_exp));
            if ~isempty(moransI_exp)
                expSpatialPersistFeatures(c, 1) = mean(moransI_exp);
                expSpatialPersistFeatures(c, 2) = std(moransI_exp);
                expSpatialPersistFeatures(c, 3) = median(moransI_exp);
                expSpatialPersistFeatures(c, 4) = skewness(moransI_exp);
                expSpatialPersistFeatures(c, 5) = kurtosis(moransI_exp);
                expSpatialPersistFeatures(c, 6) = iqr(moransI_exp);
            end
        end

        % Activity features (7-12)
        if isfield(ExpStats.(condition), 'Activity_all')
            activity_exp = ExpStats.(condition).Activity_all;
            activity_exp = activity_exp(~isnan(activity_exp));
            if ~isempty(activity_exp)
                expSpatialPersistFeatures(c, 7) = mean(activity_exp);
                expSpatialPersistFeatures(c, 8) = std(activity_exp);
                expSpatialPersistFeatures(c, 9) = median(activity_exp);
                expSpatialPersistFeatures(c, 10) = skewness(activity_exp);
                expSpatialPersistFeatures(c, 11) = kurtosis(activity_exp);
                expSpatialPersistFeatures(c, 12) = iqr(activity_exp);
            end
        end

        % Blob Persistence features (13-18) - from Global stats
        if isfield(ExpStats, 'Global') && isfield(ExpStats.Global, 'BlobPersistence') && ...
           isfield(ExpStats.Global.BlobPersistence, 'lifetimes') && ~isempty(ExpStats.Global.BlobPersistence.lifetimes)
            lifetimes = ExpStats.Global.BlobPersistence.lifetimes;
            lifetimes = lifetimes(~isnan(lifetimes));
            if length(lifetimes) >= 2
                expSpatialPersistFeatures(c, 13) = mean(lifetimes);
                expSpatialPersistFeatures(c, 14) = std(lifetimes);
                expSpatialPersistFeatures(c, 15) = median(lifetimes);
                expSpatialPersistFeatures(c, 16) = skewness(lifetimes);
                expSpatialPersistFeatures(c, 17) = kurtosis(lifetimes);
                expSpatialPersistFeatures(c, 18) = iqr(lifetimes);
            elseif ~isempty(lifetimes)
                expSpatialPersistFeatures(c, 13) = mean(lifetimes);
            end
        end
    end

    % Combine all features
    allSpatialPersistFeatures = [isingSpatialPersistFeatures; expSpatialPersistFeatures];

    % Handle NaN values: replace with column mean before normalization
    for col = 1:size(allSpatialPersistFeatures, 2)
        col_data = allSpatialPersistFeatures(:, col);
        nan_mask = isnan(col_data);
        if any(nan_mask) && ~all(nan_mask)
            col_data(nan_mask) = mean(col_data(~nan_mask));
            allSpatialPersistFeatures(:, col) = col_data;
        end
    end

    % Normalize features (z-score)
    allSpatialPersistFeatures_norm = (allSpatialPersistFeatures - mean(allSpatialPersistFeatures, 1)) ./ std(allSpatialPersistFeatures, 0, 1);

    % Handle any remaining NaN/Inf from zero std columns
    allSpatialPersistFeatures_norm(isnan(allSpatialPersistFeatures_norm) | isinf(allSpatialPersistFeatures_norm)) = 0;

    % Run UMAP on combined data
    fprintf('Running UMAP on %d simulations + %d conditions (Spatial+Persistence: 18 features)...\n', nSims, nConds);
    [embedding_spatialPersist_all, ~, ~] = run_umap(double(allSpatialPersistFeatures_norm), ...
        'n_neighbors', 199, 'min_dist', 0.3, 'n_components', 2, ...
        'metric', 'euclidean', 'verbose', false);

    % Split embedding back
    embedding_spatialPersist_ising = embedding_spatialPersist_all(1:nSims, :);
    embedding_spatialPersist_exp = embedding_spatialPersist_all(nSims+1:end, :);

    % Plot Ising simulations in gray
    scatter(embedding_spatialPersist_ising(:,1), embedding_spatialPersist_ising(:,2), 100, [0.7 0.7 0.7], ...
        'filled', 'MarkerFaceAlpha', 0.4, 'HandleVisibility', 'off');
    hold on;

    % Highlight top 3 best-matching Ising simulations for each condition
    for c = 1:nConds
        condition = config.conditions{c};
        if isfield(Comparison, condition)
            best_idx = Comparison.(condition).bestMatch_idx(1:min(3, length(Comparison.(condition).bestMatch_idx)));
            conditionColor = config.colors.(condition);
            scatter(embedding_spatialPersist_ising(best_idx, 1), embedding_spatialPersist_ising(best_idx, 2), 100, ...
                conditionColor, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
                'DisplayName', [condition ' (Top 3 Ising)']);
        end
    end

    % Plot experimental conditions with their colors (larger markers, star shape)
    for c = 1:nConds
        condition = config.conditions{c};
        conditionColor = config.colors.(condition);
        scatter(embedding_spatialPersist_exp(c, 1), embedding_spatialPersist_exp(c, 2), 300, conditionColor, ...
            'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2, ...
            'DisplayName', [condition ' (Exp)']);
    end

    hold off;
    xlabel('UMAP 1');
    ylabel('UMAP 2');
    title('Spatial+Persistence Feature Space: Moran''s I + Activity + BlobPersistence');
    legend('Location', 'best');

    % Save figure
    saveMyFig('UMAP_SpatialPersistence_FeatureSpace', fullfile(config.outputPath, 'UMAP'), gcf);

    % Store in results
    Results.UMAP_SpatialPersistence.embedding_ising = embedding_spatialPersist_ising;
    Results.UMAP_SpatialPersistence.embedding_exp = embedding_spatialPersist_exp;
    Results.UMAP_SpatialPersistence.isingFeatures = isingSpatialPersistFeatures;
    Results.UMAP_SpatialPersistence.expFeatures = expSpatialPersistFeatures;
    Results.UMAP_SpatialPersistence.featureNames = {'MoransI_mean', 'MoransI_std', 'MoransI_median', ...
        'MoransI_skewness', 'MoransI_kurtosis', 'MoransI_iqr', ...
        'Activity_mean', 'Activity_std', 'Activity_median', ...
        'Activity_skewness', 'Activity_kurtosis', 'Activity_iqr', ...
        'BlobPersist_mean', 'BlobPersist_std', 'BlobPersist_median', ...
        'BlobPersist_skewness', 'BlobPersist_kurtosis', 'BlobPersist_iqr'};

    % --- Top 10 version of same UMAP (using helper) ---
    plotUMAPComparison(embedding_spatialPersist_ising, embedding_spatialPersist_exp, Comparison, ...
        config.conditions, config.colors, 10, ...
        'Spatial+Persistence Feature Space: Top 10', 'UMAP', config.outputPath);

catch ME
    warning('UMAP (Spatial+Persistence feature space) failed: %s', ME.message);
end
end  % end if spatial+persistence

% -------------------------------------------------------------------------
% Figure 7e: UMAP of BlobPersistence Features Only
% Always generated regardless of matchingMetric
% -------------------------------------------------------------------------
figure('Name', 'UMAP: BlobPersistence Feature Space');

fprintf('Computing BlobPersistence features for UMAP feature space...\n');

try
    % Feature counts: 6 BlobPersistence features (mean, std, median, skewness, kurtosis, iqr)
    nSims = length(IsingData.simIDs);
    nConds = length(config.conditions);

    isingBlobPersistFeatures = zeros(nSims, 6);
    expBlobPersistFeatures = zeros(nConds, 6);

    % Extract data for parfor compatibility
    hasBlobPersistence = isfield(IsingData, 'BlobPersistence_lifetimes') && ~isempty(IsingData.BlobPersistence_lifetimes);
    if hasBlobPersistence
        BlobPersistence_lifetimes = IsingData.BlobPersistence_lifetimes;
    else
        BlobPersistence_lifetimes = cell(nSims, 1);
    end

    % Compute features for each Ising simulation
    parfor s = 1:nSims
        features = zeros(1, 6);

        % Blob Persistence features (1-6)
        if hasBlobPersistence && ~isempty(BlobPersistence_lifetimes{s})
            lifetimes = BlobPersistence_lifetimes{s};
            lifetimes = lifetimes(~isnan(lifetimes));
            if length(lifetimes) >= 2
                features(1:6) = [mean(lifetimes), std(lifetimes), median(lifetimes), ...
                                 skewness(lifetimes), kurtosis(lifetimes), iqr(lifetimes)];
            elseif ~isempty(lifetimes)
                features(1) = mean(lifetimes);
            end
        end

        isingBlobPersistFeatures(s, :) = features;
    end

    % Compute features for experimental data (from Global stats, same for all conditions)
    if isfield(ExpStats, 'Global') && isfield(ExpStats.Global, 'BlobPersistence') && ...
       isfield(ExpStats.Global.BlobPersistence, 'lifetimes') && ~isempty(ExpStats.Global.BlobPersistence.lifetimes)
        lifetimes = ExpStats.Global.BlobPersistence.lifetimes;
        lifetimes = lifetimes(~isnan(lifetimes));
        if length(lifetimes) >= 2
            expFeatures = [mean(lifetimes), std(lifetimes), median(lifetimes), ...
                           skewness(lifetimes), kurtosis(lifetimes), iqr(lifetimes)];
        elseif ~isempty(lifetimes)
            expFeatures = [mean(lifetimes), 0, 0, 0, 0, 0];
        else
            expFeatures = zeros(1, 6);
        end
    else
        expFeatures = zeros(1, 6);
    end

    % Use same experimental features for all conditions (Global reference)
    for c = 1:nConds
        expBlobPersistFeatures(c, :) = expFeatures;
    end

    % Combine all features
    allBlobPersistFeatures = [isingBlobPersistFeatures; expBlobPersistFeatures];

    % Handle NaN values: replace with column mean before normalization
    for col = 1:size(allBlobPersistFeatures, 2)
        col_data = allBlobPersistFeatures(:, col);
        nan_mask = isnan(col_data);
        if any(nan_mask) && ~all(nan_mask)
            col_data(nan_mask) = mean(col_data(~nan_mask));
            allBlobPersistFeatures(:, col) = col_data;
        end
    end

    % Normalize features (z-score)
    allBlobPersistFeatures_norm = (allBlobPersistFeatures - mean(allBlobPersistFeatures, 1)) ./ std(allBlobPersistFeatures, 0, 1);

    % Handle any remaining NaN/Inf from zero std columns
    allBlobPersistFeatures_norm(isnan(allBlobPersistFeatures_norm) | isinf(allBlobPersistFeatures_norm)) = 0;

    % Run UMAP on combined data
    fprintf('Running UMAP on %d simulations + %d conditions (BlobPersistence: 6 features)...\n', nSims, nConds);
    [embedding_blobPersist_all, ~, ~] = run_umap(double(allBlobPersistFeatures_norm), ...
        'n_neighbors', 199, 'min_dist', 0.3, 'n_components', 2, ...
        'metric', 'euclidean', 'verbose', false);

    % Split embedding back
    embedding_blobPersist_ising = embedding_blobPersist_all(1:nSims, :);
    embedding_blobPersist_exp = embedding_blobPersist_all(nSims+1:end, :);

    % Plot Ising simulations in gray
    scatter(embedding_blobPersist_ising(:,1), embedding_blobPersist_ising(:,2), 100, [0.7 0.7 0.7], ...
        'filled', 'MarkerFaceAlpha', 0.4, 'HandleVisibility', 'off');
    hold on;

    % Highlight top 3 best-matching Ising simulations for each condition
    for c = 1:nConds
        condition = config.conditions{c};
        if isfield(Comparison, condition)
            best_idx = Comparison.(condition).bestMatch_idx(1:min(3, length(Comparison.(condition).bestMatch_idx)));
            conditionColor = config.colors.(condition);
            scatter(embedding_blobPersist_ising(best_idx, 1), embedding_blobPersist_ising(best_idx, 2), 100, ...
                conditionColor, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
                'DisplayName', [condition ' (Top 3 Ising)']);
        end
    end

    % Plot experimental conditions with their colors (larger markers, star shape)
    for c = 1:nConds
        condition = config.conditions{c};
        conditionColor = config.colors.(condition);
        scatter(embedding_blobPersist_exp(c, 1), embedding_blobPersist_exp(c, 2), 300, conditionColor, ...
            'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2, ...
            'DisplayName', [condition ' (Exp)']);
    end

    hold off;
    xlabel('UMAP 1');
    ylabel('UMAP 2');
    title('BlobPersistence Feature Space: Blob Lifetime Statistics');
    legend('Location', 'best');

    % Save figure
    saveMyFig('UMAP_BlobPersistence_FeatureSpace', fullfile(config.outputPath, 'UMAP'), gcf);

    % Store in results
    Results.UMAP_BlobPersistence.embedding_ising = embedding_blobPersist_ising;
    Results.UMAP_BlobPersistence.embedding_exp = embedding_blobPersist_exp;
    Results.UMAP_BlobPersistence.isingFeatures = isingBlobPersistFeatures;
    Results.UMAP_BlobPersistence.expFeatures = expBlobPersistFeatures;
    Results.UMAP_BlobPersistence.featureNames = {'BlobPersist_mean', 'BlobPersist_std', 'BlobPersist_median', ...
        'BlobPersist_skewness', 'BlobPersist_kurtosis', 'BlobPersist_iqr'};

    % --- Top 10 version of same UMAP (using helper) ---
    plotUMAPComparison(embedding_blobPersist_ising, embedding_blobPersist_exp, Comparison, ...
        config.conditions, config.colors, 10, ...
        'BlobPersistence Feature Space: Top 10', 'UMAP', config.outputPath);

catch ME
    warning('UMAP (BlobPersistence feature space) failed: %s', ME.message);
end

% -------------------------------------------------------------------------
% Figure 8: Multi-Observable Comparison (Best Match vs Experimental)
%% -------------------------------------------------------------------------
fprintf('\n--- Creating Multi-Observable Comparison Figures ---\n');

for c = 1:length(config.conditions)
    condition = config.conditions{c};

    if ~isfield(Comparison, condition) || ~isfield(ExpStats, condition)
        continue;
    end

    fprintf('Creating multi-observable figure for %s...\n', condition);

    figure('Name', sprintf('Multi-Observable Comparison: %s', condition));

    % Get best matching simulation
    best_idx = Comparison.(condition).bestMatch_idx(1);
    best_simID = IsingData.simIDs{best_idx};

    % Load best simulation data for activity calculations
    simPath = fullfile(config.isingDataPath, IsingData.simFilenames{best_idx});
    simData = load(simPath);
    stored_spins = simData.stored_spins;  % [4000 x 32 x 32]

    % --- Panel 1: Moran's I Distribution ---
    subplot(2, 2, 1);
    hold on;
    histogram(ExpStats.(condition).MoransI_all, 50, 'Normalization', 'pdf', ...
        'FaceColor', config.colors.(condition), 'FaceAlpha', 0.5, 'DisplayName', 'Experimental');
    histogram(IsingData.MoransI_all{best_idx}, 50, 'Normalization', 'pdf', ...
        'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.5, 'DisplayName', sprintf('Ising %s', best_simID));
    xlabel('Moran''s I');
    ylabel('PDF');
    title('Moran''s I Distribution');
    legend('Location', 'best');
    hold off;

    % --- Panel 2: Activity/Sparsity Distribution ---
    subplot(2, 2, 2);
    hold on;
    % Experimental: compute sparsity per trial x timepoint from BinarisedData
    exp_sparsity = mean(BinarisedData_Exp.(condition), [1 2]);  % Mean across grid
    exp_sparsity = squeeze(exp_sparsity(:));  % Flatten
    % Ising: compute sparsity per frame (convert -1/+1 to 0/1 to match experimental scale)
    ising_sparsity = squeeze(mean((stored_spins + 1) / 2, [2 3]));  % Fraction active per frame
    histogram(exp_sparsity, 50, 'Normalization', 'pdf', ...
        'FaceColor', config.colors.(condition), 'FaceAlpha', 0.5, 'DisplayName', 'Experimental');
    histogram(ising_sparsity, 50, 'Normalization', 'pdf', ...
        'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.5, 'DisplayName', 'Ising');
    xlabel('Activity (fraction active)');
    ylabel('PDF');
    title('Activity Distribution');
    legend('Location', 'best');
    hold off;

    % --- Panel 3: Temporal Autocorrelation ---
    subplot(2, 2, 3);
    hold on;
    maxLag = 50;
    % Experimental autocorrelation (trial-averaged on Moran's I)
    if isfield(ExpStats.(condition), 'MoransI_trials') && ~isempty(ExpStats.(condition).MoransI_trials)
        moransI_trials = ExpStats.(condition).MoransI_trials;
        [~, exp_acf, ~, ~] = computeTrialAveragedAutocorr(moransI_trials, maxLag, [1, maxLag]);
    else
        exp_acf = zeros(1, maxLag + 1);
    end
    % Ising autocorrelation (unchanged - already continuous time series)
    ising_moransI = IsingData.MoransI_all{best_idx};
    ising_moransI_clean = ising_moransI(~isnan(ising_moransI));
    [ising_acf, ~] = xcorr(ising_moransI_clean - mean(ising_moransI_clean), maxLag, 'normalized');
    ising_acf = ising_acf(maxLag+1:end);
    plot(0:maxLag, exp_acf, '-', 'Color', config.colors.(condition), 'LineWidth', 2, 'DisplayName', 'Experimental');
    plot(0:maxLag, ising_acf, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'DisplayName', 'Ising');
    xlabel('Lag');
    ylabel('Autocorrelation');
    title('Temporal Autocorrelation (Moran''s I)');
    legend('Location', 'best');
    xlim([0 maxLag]);
    hold off;

    % --- Panel 4: Summary Statistics Table ---
    subplot(2, 2, 4);
    axis off;
    % Build stats text dynamically based on available metrics
    stats_text = {
        sprintf('Condition: %s', condition);
        sprintf('Best Match: %s', best_simID);
        '';
        'Moran''s I:';
        sprintf('  Exp mean: %.4f', ExpStats.(condition).MoransI_mean);
        sprintf('  Ising mean: %.4f', IsingData.MoransI_mean(best_idx));
        '';
        'Activity:';
        sprintf('  Exp mean: %.4f', mean(exp_sparsity));
        sprintf('  Ising mean: %.4f', mean(ising_sparsity));
        '';
    };

    % Add raw Wasserstein distances section
    raw_dist_lines = {'Raw Wasserstein Distances:'};
    if isfield(Comparison.(condition), 'dist_moransI_raw')
        raw_dist_lines{end+1} = sprintf('  Moran''s I: %.4f', Comparison.(condition).dist_moransI_raw(best_idx));
    end
    if isfield(Comparison.(condition), 'dist_activity_raw')
        raw_dist_lines{end+1} = sprintf('  Activity: %.4f', Comparison.(condition).dist_activity_raw(best_idx));
    end
    if isfield(Comparison.(condition), 'dist_blob_raw')
        raw_dist_lines{end+1} = sprintf('  Blob count: %.4f', Comparison.(condition).dist_blob_raw(best_idx));
    end
    if isfield(Comparison.(condition), 'dist_autocorr_raw')
        raw_dist_lines{end+1} = sprintf('  Autocorr (log-ratio): %.4f', Comparison.(condition).dist_autocorr_raw(best_idx));
    end

    stats_text = [stats_text; raw_dist_lines(:)];
    stats_text{end+1} = '';

    % Add combined distance with correct label
    if strcmp(config.matchingMetric, 'combined')
        stats_text{end+1} = sprintf('Z-scored combined Wasserstein dist: %.4f', Comparison.(condition).wasserstein_dist(best_idx));
    else
        stats_text{end+1} = sprintf('Wasserstein dist: %.4f', Comparison.(condition).wasserstein_dist(best_idx));
    end

    text(0.1, 0.9, stats_text, 'VerticalAlignment', 'top', 'FontSize', 10, 'FontName', 'FixedWidth');
    title('Summary Statistics');

    sgtitle(sprintf('%s: Best Match Comparison (%s)', condition, best_simID));

    % Save figure
    saveMyFig(sprintf('MultiObservable_%s', condition), fullfile(config.outputPath, 'MultiObservable'), gcf);
end

% -------------------------------------------------------------------------
% Figure 9: Simplified Distribution Comparison (Two-Panel)
% -------------------------------------------------------------------------
fprintf('\n--- Creating Simplified Distribution Comparison Figures ---\n');

for c = 1:length(config.conditions)
    condition = config.conditions{c};

    if ~isfield(Comparison, condition) || ~isfield(ExpStats, condition)
        continue;
    end

    fprintf('Creating simplified distribution figure for %s...\n', condition);

    figure('Name', sprintf('Distribution Comparison: %s', condition));

    % Get best matching simulation
    best_idx = Comparison.(condition).bestMatch_idx(1);
    best_simID = IsingData.simIDs{best_idx};

    % Load best simulation data for activity calculations
    simPath = fullfile(config.isingDataPath, IsingData.simFilenames{best_idx});
    simData = load(simPath);
    stored_spins = simData.stored_spins;  % [4000 x 32 x 32]

    % --- Panel 1: Moran's I Distribution ---
    subplot(1, 2, 1);
    hold on;
    histogram(ExpStats.(condition).MoransI_all, 50, 'Normalization', 'pdf', ...
        'FaceColor', config.colors.(condition), 'FaceAlpha', 0.5, 'DisplayName', 'Experimental');
    histogram(IsingData.MoransI_all{best_idx}, 50, 'Normalization', 'pdf', ...
        'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.5, 'DisplayName', sprintf('Ising %s', best_simID));
    xlabel('Moran''s I');
    ylabel('PDF');
    title('Moran''s I Distribution');
    legend('Location', 'best');
    hold off;

    % --- Panel 2: Activity/Sparsity Distribution ---
    subplot(1, 2, 2);
    hold on;
    % Experimental: compute sparsity per trial x timepoint from BinarisedData
    exp_sparsity = mean(BinarisedData_Exp.(condition), [1 2]);  % Mean across grid
    exp_sparsity = squeeze(exp_sparsity(:));  % Flatten
    % Ising: compute sparsity per frame (convert -1/+1 to 0/1 to match experimental scale)
    ising_sparsity = squeeze(mean((stored_spins + 1) / 2, [2 3]));  % Fraction active per frame
    histogram(exp_sparsity, 50, 'Normalization', 'pdf', ...
        'FaceColor', config.colors.(condition), 'FaceAlpha', 0.5, 'DisplayName', 'Experimental');
    histogram(ising_sparsity, 50, 'Normalization', 'pdf', ...
        'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.5, 'DisplayName', 'Ising');
    xlabel('Activity (fraction active)');
    ylabel('PDF');
    title('Activity Distribution');
    legend('Location', 'best');
    hold off;

    sgtitle(sprintf('%s: Distribution Comparison (%s)', condition, best_simID));

    % Save figure
    saveMyFig(sprintf('DistributionComparison_%s', condition), fullfile(config.outputPath, 'Distributions'), gcf);
end

% -------------------------------------------------------------------------
% Figure 10: Q-Q Plots - Best Match Only
% -------------------------------------------------------------------------
fprintf('\n--- Creating Q-Q Plot Figures ---\n');

figure('Name', 'Q-Q Plots: Moran''s I (Best Match)');

nQuantiles = 100;
p = linspace(0.01, 0.99, nQuantiles);  % 1st to 99th percentile

for c = 1:nConditions
    condition = config.conditions{c};

    if ~isfield(ExpStats, condition) || ~isfield(Comparison, condition)
        continue;
    end

    subplot(2, 2, c);

    % Get experimental data
    exp_data = ExpStats.(condition).MoransI_all;
    exp_data = exp_data(~isnan(exp_data));

    best_idx = Comparison.(condition).bestMatch_idx(1);
    ising_data = IsingData.MoransI_all{best_idx};
    ising_data = ising_data(~isnan(ising_data));

    % Compute quantiles
    exp_q = quantile(exp_data, p);
    ising_q = quantile(ising_data, p);

    % Plot Q-Q
    plot(exp_q, ising_q, 'o', 'Color', config.colors.(condition), ...
        'MarkerSize', 4, 'MarkerFaceColor', config.colors.(condition));
    hold on;

    % Reference line
    all_vals = [exp_q(:); ising_q(:)];
    min_val = min(all_vals);
    max_val = max(all_vals);
    plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.5);

    xlabel('Experimental Quantiles');
    ylabel('Ising Quantiles');
    title(sprintf('%s (%s)', condition, IsingData.simIDs{best_idx}));
    axis equal;
    hold off;
end

sgtitle('Q-Q Plot: Moran''s I Distribution (Best Match)');
saveMyFig('QQ_MoransI_BestMatch', fullfile(config.outputPath, 'QQPlots'), gcf);

% -------------------------------------------------------------------------
% Figure 11: Q-Q Plots - Top 3 Matches
% -------------------------------------------------------------------------
figure('Name', 'Q-Q Plots: Moran''s I (Top 3 Matches)');

for c = 1:nConditions
    condition = config.conditions{c};

    if ~isfield(ExpStats, condition) || ~isfield(Comparison, condition)
        continue;
    end

    subplot(2, 2, c);

    % Get experimental data
    exp_data = ExpStats.(condition).MoransI_all;
    exp_data = exp_data(~isnan(exp_data));
    exp_q = quantile(exp_data, p);

    % Plot top 3 matches
    alphas = [1.0, 0.6, 0.3];  % Decreasing alpha for ranks 1, 2, 3
    legend_entries = {};

    for rank = 1:3
        best_idx = Comparison.(condition).bestMatch_idx(rank);
        ising_data = IsingData.MoransI_all{best_idx};
        ising_data = ising_data(~isnan(ising_data));
        ising_q = quantile(ising_data, p);

        scatter(exp_q, ising_q, 16, config.colors.(condition), 'filled', ...
            'MarkerFaceAlpha', alphas(rank), 'MarkerEdgeAlpha', alphas(rank));
        hold on;
        legend_entries{end+1} = sprintf('%s (#%d)', IsingData.simIDs{best_idx}, rank);
    end

    % Reference line
    all_vals = [exp_q(:); ising_q(:)];
    plot([min(all_vals), max(all_vals)], [min(all_vals), max(all_vals)], ...
        'k--', 'LineWidth', 1.5);
    legend_entries{end+1} = 'y=x';

    xlabel('Experimental Quantiles');
    ylabel('Ising Quantiles');
    title(condition);
    legend(legend_entries, 'Location', 'best', 'FontSize', 7);
    axis equal;
    hold off;
end

sgtitle('Q-Q Plot: Moran''s I Distribution (Top 3 Matches)');
saveMyFig('QQ_MoransI_Top3Matches', fullfile(config.outputPath, 'QQPlots'), gcf);

% -------------------------------------------------------------------------
% Figure 12: Q-Q Plots - Activity Best Match Only
% -------------------------------------------------------------------------
figure('Name', 'Q-Q Plots: Activity (Best Match)');

for c = 1:nConditions
    condition = config.conditions{c};

    if ~isfield(ExpStats, condition) || ~isfield(Comparison, condition)
        continue;
    end

    % Check if Activity data exists
    if ~isfield(ExpStats.(condition), 'Activity_all') || isempty(ExpStats.(condition).Activity_all)
        continue;
    end

    subplot(2, 2, c);

    % Get experimental activity data
    exp_data = ExpStats.(condition).Activity_all;
    exp_data = exp_data(~isnan(exp_data));

    best_idx = Comparison.(condition).bestMatch_idx(1);
    ising_data = IsingData.Activity_all{best_idx};
    ising_data = ising_data(~isnan(ising_data));

    % Compute quantiles
    exp_q = quantile(exp_data, p);
    ising_q = quantile(ising_data, p);

    % Plot Q-Q
    plot(exp_q, ising_q, 'o', 'Color', config.colors.(condition), ...
        'MarkerSize', 4, 'MarkerFaceColor', config.colors.(condition));
    hold on;

    % Reference line
    all_vals = [exp_q(:); ising_q(:)];
    min_val = min(all_vals);
    max_val = max(all_vals);
    plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.5);

    xlabel('Experimental Quantiles');
    ylabel('Ising Quantiles');
    title(sprintf('%s (%s)', condition, IsingData.simIDs{best_idx}));
    axis equal;
    hold off;
end

sgtitle('Q-Q Plot: Activity Distribution (Best Match)');
saveMyFig('QQ_Activity_BestMatch', fullfile(config.outputPath, 'QQPlots'), gcf);

% -------------------------------------------------------------------------
% Figure 13: Q-Q Plots - Activity Top 3 Matches
% -------------------------------------------------------------------------
figure('Name', 'Q-Q Plots: Activity (Top 3 Matches)');

for c = 1:nConditions
    condition = config.conditions{c};

    if ~isfield(ExpStats, condition) || ~isfield(Comparison, condition)
        continue;
    end

    % Check if Activity data exists
    if ~isfield(ExpStats.(condition), 'Activity_all') || isempty(ExpStats.(condition).Activity_all)
        continue;
    end

    subplot(2, 2, c);

    % Get experimental activity data
    exp_data = ExpStats.(condition).Activity_all;
    exp_data = exp_data(~isnan(exp_data));
    exp_q = quantile(exp_data, p);

    % Plot top 3 matches
    alphas = [1.0, 0.6, 0.3];  % Decreasing alpha for ranks 1, 2, 3
    legend_entries = {};

    for rank = 1:3
        best_idx = Comparison.(condition).bestMatch_idx(rank);
        ising_data = IsingData.Activity_all{best_idx};
        ising_data = ising_data(~isnan(ising_data));
        ising_q = quantile(ising_data, p);

        scatter(exp_q, ising_q, 16, config.colors.(condition), 'filled', ...
            'MarkerFaceAlpha', alphas(rank), 'MarkerEdgeAlpha', alphas(rank));
        hold on;
        legend_entries{end+1} = sprintf('%s (#%d)', IsingData.simIDs{best_idx}, rank);
    end

    % Reference line
    all_vals = [exp_q(:); ising_q(:)];
    plot([min(all_vals), max(all_vals)], [min(all_vals), max(all_vals)], ...
        'k--', 'LineWidth', 1.5);
    legend_entries{end+1} = 'y=x';

    xlabel('Experimental Quantiles');
    ylabel('Ising Quantiles');
    title(condition);
    legend(legend_entries, 'Location', 'best', 'FontSize', 7);
    axis equal;
    hold off;
end

sgtitle('Q-Q Plot: Activity Distribution (Top 3 Matches)');
saveMyFig('QQ_Activity_Top3Matches', fullfile(config.outputPath, 'QQPlots'), gcf);

% -------------------------------------------------------------------------
% Figure 14: Tiled Mode Analysis - Single Position vs Pooled WD
% -------------------------------------------------------------------------
% NOTE: "Pooling" = concatenating samples from all positions (2x samples)
%       NOT frame-by-frame averaging (which changes variance)
if strcmp(config.gridMode, 'subselect_tiled') && nTiledPositions >= 2
    fprintf('\n--- Creating Tiled Mode Analysis Figure ---\n');

    figure('Name', 'Tiled Mode: Position Pooling Effect');

    % Compute WD for single position (P1) vs pooled across all positions
    TiledAnalysis = struct();

    for c = 1:nConditions
        condition = config.conditions{c};

        if ~isfield(ExpStats, condition) || ~isfield(Comparison, condition)
            continue;
        end

        % Get experimental data
        exp_data = ExpStats.(condition).MoransI_all;
        exp_data = exp_data(~isnan(exp_data));

        % Get best-match simulation index
        best_idx = Comparison.(condition).bestMatch_idx(1);

        % WD using POOLED Moran's I (concatenated from all positions)
        sim_pooled = IsingData.MoransI_pooled{best_idx};
        WD_pooled = wasserstein_1d(exp_data, sim_pooled(:));

        % WD using single position (P1 only)
        sim_P1 = IsingData.MoransI_perPosition{1}{best_idx};
        WD_single = wasserstein_1d(exp_data, sim_P1(:));

        % Store results
        TiledAnalysis.(condition).WD_single = WD_single;
        TiledAnalysis.(condition).WD_pooled = WD_pooled;
        TiledAnalysis.(condition).WD_change_pct = 100 * (WD_pooled - WD_single) / WD_single;
        TiledAnalysis.(condition).improved = WD_pooled < WD_single;

        fprintf('  %s: WD(P1)=%.4f, WD(pooled)=%.4f (%+.1f%%)\n', condition, ...
            WD_single, WD_pooled, TiledAnalysis.(condition).WD_change_pct);
    end

    % Panel 1: WD bars (Single vs Pooled)
    subplot(1, 3, 1);
    WD_data = zeros(nConditions, 2);
    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(TiledAnalysis, condition)
            WD_data(c, 1) = TiledAnalysis.(condition).WD_single;
            WD_data(c, 2) = TiledAnalysis.(condition).WD_pooled;
        end
    end
    b = bar(WD_data);
    b(1).FaceColor = [0.7 0.7 0.7];  % Single position (gray)
    b(2).FaceColor = [0.2 0.6 0.8];  % Pooled (blue)
    xlabel('Condition');
    ylabel('Wasserstein Distance');
    title('Data vs Model: WD Comparison');
    xticks(1:nConditions);
    xticklabels(config.conditions);
    xtickangle(45);
    legend({'Single (P1)', sprintf('Pooled (%d positions)', nTiledPositions)}, 'Location', 'best');
    grid on;

    % Panel 2: % Change bars
    subplot(1, 3, 2);
    hold on;
    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(TiledAnalysis, condition)
            change_pct = TiledAnalysis.(condition).WD_change_pct;
            if change_pct < 0
                barColor = [0.2 0.7 0.3];  % Green for improvement
            else
                barColor = [0.8 0.3 0.3];  % Red for worsening
            end
            bar(c, change_pct, 'FaceColor', barColor);
        end
    end
    hold off;
    xlabel('Condition');
    ylabel('% Change in WD');
    title('Pooling Effect (neg = better match)');
    xticks(1:nConditions);
    xticklabels(config.conditions);
    xtickangle(45);
    yline(0, 'k-', 'LineWidth', 1);
    grid on;

    % Panel 3: Summary
    subplot(1, 3, 3);
    axis off;
    summaryText = {
        '=== TILED MODE ANALYSIS ===', ...
        '', ...
        sprintf('Positions: %d', nTiledPositions), ...
        'WD(Experimental, Simulation)', ...
        '', ...
    };
    n_improved = 0;
    all_changes = [];
    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(TiledAnalysis, condition)
            summaryText{end+1} = sprintf('%s:', condition);
            summaryText{end+1} = sprintf('  P1: %.4f', TiledAnalysis.(condition).WD_single);
            summaryText{end+1} = sprintf('  Pooled: %.4f (%+.1f%%)', ...
                TiledAnalysis.(condition).WD_pooled, TiledAnalysis.(condition).WD_change_pct);
            all_changes(end+1) = TiledAnalysis.(condition).WD_change_pct;
            if TiledAnalysis.(condition).improved
                n_improved = n_improved + 1;
            end
        end
    end
    summaryText{end+1} = '';
    summaryText{end+1} = sprintf('Improved: %d/%d', n_improved, length(all_changes));
    if ~isempty(all_changes)
        summaryText{end+1} = sprintf('Mean change: %+.1f%%', mean(all_changes));
    end
    text(0.05, 0.95, summaryText, 'VerticalAlignment', 'top', ...
        'FontSize', 9, 'FontName', 'FixedWidth', 'Units', 'normalized');
    title('Summary');

    sgtitle(sprintf('Tiled Mode: Single Position vs %d-Position Pooled', nTiledPositions), 'FontWeight', 'bold');

    saveMyFig('TiledMode_PositionPooling', fullfile(config.outputPath, 'TiledAnalysis'), gcf);
    fprintf('Tiled analysis figure saved\n');
end

% -------------------------------------------------------------------------
% Figure 16: Pooled-Based Matching vs P1-Based Matching
% -------------------------------------------------------------------------
% Re-match using POOLED distribution and compare to original P1-based matching
if strcmp(config.gridMode, 'subselect_tiled') && nTiledPositions >= 2
    fprintf('\n--- Creating Pooled vs P1 Matching Comparison Figure ---\n');

    PooledMatching = struct();

    for c = 1:nConditions
        condition = config.conditions{c};

        if ~isfield(ExpStats, condition)
            continue;
        end

        exp_data = ExpStats.(condition).MoransI_all;
        exp_data = exp_data(~isnan(exp_data));
        nSims = length(IsingData.simIDs);

        % Compute WD using POOLED data for ALL simulations
        wasserstein_pooled = zeros(nSims, 1);
        parfor s = 1:nSims
            pooled_values = IsingData.MoransI_pooled{s}(:);
            wasserstein_pooled(s) = wasserstein_1d(exp_data, pooled_values);
        end

        % Rank simulations by pooled-based WD
        wasserstein_for_sort = wasserstein_pooled;
        wasserstein_for_sort(isnan(wasserstein_for_sort)) = Inf;
        [~, rank_pooled] = sort(wasserstein_for_sort, 'ascend');

        % Get original P1-based results
        rank_P1 = Comparison.(condition).rankings;
        best_P1_idx = Comparison.(condition).bestMatch_idx(1);
        best_P1_simID = IsingData.simIDs{best_P1_idx};
        WD_P1_best = Comparison.(condition).wasserstein_dist(best_P1_idx);

        % Get pooled-based results
        best_pooled_idx = rank_pooled(1);
        best_pooled_simID = IsingData.simIDs{best_pooled_idx};
        WD_pooled_best = wasserstein_pooled(best_pooled_idx);

        % Check if best match changed
        best_match_changed = (best_P1_idx ~= best_pooled_idx);

        % Store results
        PooledMatching.(condition).wasserstein_pooled = wasserstein_pooled;
        PooledMatching.(condition).rankings_pooled = rank_pooled;
        PooledMatching.(condition).best_P1_idx = best_P1_idx;
        PooledMatching.(condition).best_P1_simID = best_P1_simID;
        PooledMatching.(condition).WD_P1_best = WD_P1_best;
        PooledMatching.(condition).best_pooled_idx = best_pooled_idx;
        PooledMatching.(condition).best_pooled_simID = best_pooled_simID;
        PooledMatching.(condition).WD_pooled_best = WD_pooled_best;
        PooledMatching.(condition).best_match_changed = best_match_changed;

        % Where did the P1-best rank in pooled matching?
        PooledMatching.(condition).P1best_rank_in_pooled = find(rank_pooled == best_P1_idx);
        % Where did the pooled-best rank in P1 matching?
        PooledMatching.(condition).pooledbest_rank_in_P1 = find(rank_P1 == best_pooled_idx);

        if best_match_changed
            status_str = '** CHANGED **';
        else
            status_str = '(same)';
        end
        fprintf('  %s: P1-best=sim_%d (WD=%.4f), Pooled-best=sim_%d (WD=%.4f) %s\n', ...
            condition, best_P1_simID, WD_P1_best, best_pooled_simID, WD_pooled_best, status_str);
    end

    % Count how many conditions changed
    n_changed = 0;
    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(PooledMatching, condition) && PooledMatching.(condition).best_match_changed
            n_changed = n_changed + 1;
        end
    end

    % Create figure
    figure('Name', 'Pooled vs P1 Matching Comparison');

    % Panel 1: Bar chart comparing best-match WD
    subplot(2, 2, 1);
    WD_comparison = zeros(nConditions, 2);
    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(PooledMatching, condition)
            WD_comparison(c, 1) = PooledMatching.(condition).WD_P1_best;
            WD_comparison(c, 2) = PooledMatching.(condition).WD_pooled_best;
        end
    end
    b = bar(WD_comparison);
    b(1).FaceColor = [0.7 0.7 0.7];
    b(2).FaceColor = [0.2 0.6 0.8];
    xlabel('Condition');
    ylabel('Wasserstein Distance');
    title('Best Match WD: P1 vs Pooled Selection');
    xticks(1:nConditions);
    xticklabels(config.conditions);
    xtickangle(45);
    legend({'P1-based match', 'Pooled-based match'}, 'Location', 'best');
    grid on;

    % Panel 2: Scatter plot of P1-rank vs Pooled-rank for top matches
    subplot(2, 2, 2);
    hold on;
    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(PooledMatching, condition)
            % Plot top 20 matches
            top_n = min(20, length(PooledMatching.(condition).rankings_pooled));
            P1_ranks = 1:top_n;
            pooled_ranks = zeros(1, top_n);
            for r = 1:top_n
                idx = Comparison.(condition).rankings(r);
                pooled_ranks(r) = find(PooledMatching.(condition).rankings_pooled == idx);
            end
            scatter(P1_ranks, pooled_ranks, 50, config.colors.(condition), 'filled', 'MarkerFaceAlpha', 0.7);
        end
    end
    plot([1 top_n], [1 top_n], 'k--', 'LineWidth', 1.5);
    hold off;
    xlabel('Rank (P1-based matching)');
    ylabel('Rank (Pooled-based matching)');
    title('Rank Comparison (Top 20)');
    legend([config.conditions, {'Identity'}], 'Location', 'best');
    grid on;

    % Panel 3: Best-match simulation IDs
    subplot(2, 2, 3);
    axis off;
    matchText = {'=== BEST MATCH COMPARISON ===', ''};
    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(PooledMatching, condition)
            if PooledMatching.(condition).best_match_changed
                status = sprintf('CHANGED (P1 rank %d -> pooled rank 1)', ...
                    PooledMatching.(condition).pooledbest_rank_in_P1);
            else
                status = 'SAME';
            end
            matchText{end+1} = sprintf('%s:', upper(condition));
            matchText{end+1} = sprintf('  P1-best: %s', PooledMatching.(condition).best_P1_simID);
            matchText{end+1} = sprintf('  Pooled-best: %s', PooledMatching.(condition).best_pooled_simID);
            matchText{end+1} = sprintf('  Status: %s', status);
            matchText{end+1} = '';
        end
    end
    text(0.05, 0.95, matchText, 'VerticalAlignment', 'top', ...
        'FontSize', 9, 'FontName', 'FixedWidth', 'Units', 'normalized');
    title('Best Match Details');

    % Panel 4: Summary
    subplot(2, 2, 4);
    axis off;
    summaryText = {
        '=== SUMMARY ===', ...
        '', ...
        sprintf('Conditions analyzed: %d', nConditions), ...
        sprintf('Best match changed: %d/%d', n_changed, nConditions), ...
        '', ...
        'Interpretation:', ...
        '  If best match changes frequently,', ...
        '  pooling provides additional information', ...
        '  that refines the model selection.', ...
    };
    text(0.05, 0.95, summaryText, 'VerticalAlignment', 'top', ...
        'FontSize', 9, 'FontName', 'FixedWidth', 'Units', 'normalized');
    title('Summary');

    sgtitle('Pooled vs P1-Based Matching: Does Best Match Change?', 'FontWeight', 'bold');

    saveMyFig('PooledMatching_Comparison', fullfile(config.outputPath, 'TiledAnalysis'), gcf);
    fprintf('Pooled matching comparison figure saved\n');
    fprintf('  Best match changed in %d/%d conditions\n', n_changed, nConditions);
end

% -------------------------------------------------------------------------
% Figure 15: centre vs Tiled Comparison (subselect_centre_vs_tiled mode)
% -------------------------------------------------------------------------
if strcmp(config.gridMode, 'subselect_centre_vs_tiled')
    fprintf('\n--- Creating centre vs Tiled Comparison Figure ---\n');

    figure('Name', 'centre vs Tiled Comparison');

    % Compute comparison statistics
    centreVsTiled = struct();

    for c = 1:nConditions
        condition = config.conditions{c};

        if ~isfield(Comparison_centre, condition) || ~isfield(Comparison_tiled, condition)
            continue;
        end

        % Get best-match indices from each approach
        best_idx_centre = Comparison_centre.(condition).bestMatch_idx(1);
        best_idx_tiled = Comparison_tiled.(condition).bestMatch_idx(1);

        % Get the best WD achieved by each approach
        WD_centre = Comparison_centre.(condition).wasserstein_dist(best_idx_centre);
        WD_tiled = Comparison_tiled.(condition).wasserstein_dist(best_idx_tiled);

        % Store results
        centreVsTiled.(condition).WD_centre = WD_centre;
        centreVsTiled.(condition).WD_tiled = WD_tiled;
        centreVsTiled.(condition).WD_change_pct = 100 * (WD_tiled - WD_centre) / WD_centre;
        centreVsTiled.(condition).tiled_better = WD_tiled < WD_centre;
        centreVsTiled.(condition).same_sim = (best_idx_centre == best_idx_tiled);
        centreVsTiled.(condition).simID_centre = IsingData.simIDs{best_idx_centre};
        centreVsTiled.(condition).simID_tiled = IsingData.simIDs{best_idx_tiled};

        fprintf('  %s: centre WD=%.4f, Tiled WD=%.4f (%+.1f%%)\n', condition, ...
            WD_centre, WD_tiled, centreVsTiled.(condition).WD_change_pct);
    end

    % Panel 1: WD bars (centre vs Tiled)
    subplot(1, 3, 1);
    WD_data = zeros(nConditions, 2);
    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(centreVsTiled, condition)
            WD_data(c, 1) = centreVsTiled.(condition).WD_centre;
            WD_data(c, 2) = centreVsTiled.(condition).WD_tiled;
        end
    end
    b = bar(WD_data);
    b(1).FaceColor = [0.8 0.4 0.2];  % centre (orange)
    b(2).FaceColor = [0.2 0.6 0.8];  % Tiled (blue)
    xlabel('Condition');
    ylabel('Wasserstein Distance');
    title('Best Match WD: centre vs Tiled');
    xticks(1:nConditions);
    xticklabels(config.conditions);
    xtickangle(45);
    legend({'centre crop', sprintf('Tiled pooled (%d pos)', nTiledPositions)}, 'Location', 'best');
    grid on;

    % Panel 2: % Change bars
    subplot(1, 3, 2);
    hold on;
    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(centreVsTiled, condition)
            change_pct = centreVsTiled.(condition).WD_change_pct;
            if change_pct < 0
                barColor = [0.2 0.7 0.3];  % Green: tiled better
            else
                barColor = [0.8 0.3 0.3];  % Red: centre better
            end
            bar(c, change_pct, 'FaceColor', barColor);
        end
    end
    hold off;
    xlabel('Condition');
    ylabel('% Change (Tiled vs centre)');
    title('WD Change (neg = tiled better)');
    xticks(1:nConditions);
    xticklabels(config.conditions);
    xtickangle(45);
    yline(0, 'k-', 'LineWidth', 1);
    grid on;

    % Panel 3: Summary
    subplot(1, 3, 3);
    axis off;
    summaryText = {
        '=== centre vs TILED ===', ...
        '', ...
        sprintf('Tiled positions: %d', nTiledPositions), ...
        '', ...
    };
    n_tiled_better = 0;
    n_same_sim = 0;
    all_changes = [];
    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(centreVsTiled, condition)
            summaryText{end+1} = sprintf('%s:', condition);
            summaryText{end+1} = sprintf('  centre: sim_%d (%.4f)', ...
                centreVsTiled.(condition).simID_centre, centreVsTiled.(condition).WD_centre);
            summaryText{end+1} = sprintf('  Tiled:  sim_%d (%.4f)', ...
                centreVsTiled.(condition).simID_tiled, centreVsTiled.(condition).WD_tiled);
            all_changes(end+1) = centreVsTiled.(condition).WD_change_pct;
            if centreVsTiled.(condition).tiled_better
                n_tiled_better = n_tiled_better + 1;
            end
            if centreVsTiled.(condition).same_sim
                n_same_sim = n_same_sim + 1;
            end
        end
    end
    summaryText{end+1} = '';
    summaryText{end+1} = sprintf('Tiled better: %d/%d', n_tiled_better, length(all_changes));
    summaryText{end+1} = sprintf('Same sim: %d/%d', n_same_sim, length(all_changes));
    if ~isempty(all_changes)
        summaryText{end+1} = sprintf('Mean change: %+.1f%%', mean(all_changes));
    end
    text(0.05, 0.95, summaryText, 'VerticalAlignment', 'top', ...
        'FontSize', 8, 'FontName', 'FixedWidth', 'Units', 'normalized');
    title('Summary');

    sgtitle('centre Crop vs Tiled Average: Which Matches Data Better?', 'FontWeight', 'bold');

    saveMyFig('centreVsTiled_Comparison', fullfile(config.outputPath, 'centreVsTiled'), gcf);
    fprintf('centre vs Tiled comparison figure saved\n');

    % ---------------------------------------------------------------------
    % Figure 17: Best-Match Parameters for Centre vs Tiled
    % ---------------------------------------------------------------------
    fprintf('\n--- Creating Best-Match Parameters Figure ---\n');

    figure('Name', 'Best-Match Parameters: Centre vs Tiled');

    param_names = {'beta', 'c', 'decay_const', 'inhibition_range', 'bias'};
    param_labels = {'Beta', 'c', 'Decay Const', 'Inhib Range', 'Bias'};
    nParams = length(param_names);

    % Collect parameters for all conditions
    params_centre = zeros(nConditions, nParams);
    params_tiled = zeros(nConditions, nParams);

    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(Comparison_centre, condition) && isfield(Comparison_tiled, condition)
            best_idx_centre = Comparison_centre.(condition).bestMatch_idx(1);
            best_idx_tiled = Comparison_tiled.(condition).bestMatch_idx(1);

            for p = 1:nParams
                params_centre(c, p) = IsingData.params.(param_names{p})(best_idx_centre);
                params_tiled(c, p) = IsingData.params.(param_names{p})(best_idx_tiled);
            end
        end
    end

    % Create paired comparison plot for each parameter
    for p = 1:nParams
        subplot(2, 3, p);
        hold on;

        % Compute jitter amount based on parameter range
        all_vals = [params_centre(:, p); params_tiled(:, p)];
        val_range = range(all_vals);
        if val_range == 0
            val_range = abs(mean(all_vals)) * 0.1;  % 10% of mean if all values identical
            if val_range == 0
                val_range = 1;
            end
        end
        jitter_amount = 0.2 * val_range;  % x% of range

        % Plot each condition as a line connecting Centre to Tiled
        rng(42);  % Fixed seed for reproducibility
        h_lines = gobjects(nConditions, 1);  % Store line handles for legend
        for c = 1:nConditions
            condition = config.conditions{c};
            x_vals = [1, 2];  % 1 = Centre, 2 = Tiled
            % Add small y jitter to separate overlapping points
            jitter_y = jitter_amount * (rand() - 0.5);
            y_vals = [params_centre(c, p) + jitter_y, params_tiled(c, p) + jitter_y];

            % Plot line (store handle for legend)
            h_lines(c) = plot(x_vals, y_vals, '-', 'Color', config.colors.(condition), 'LineWidth', 2);
            % Plot points
            scatter(x_vals, y_vals, 100, config.colors.(condition), 'filled', 'MarkerEdgeColor', 'k');
        end
        hold off;

        ylabel(param_labels{p});
        title(param_labels{p});
        xlim([0.5, 2.5]);
        xticks([1, 2]);
        xticklabels({'Centre', 'Tiled'});
        grid on;

        if p == 1
            legend(h_lines, config.conditions, 'Location', 'best');
        end
    end

    % Panel 6: Summary table as text
    subplot(2, 3, 6);
    axis off;
    tableText = {'=== BEST-MATCH PARAMETERS ===', ''};
    for c = 1:nConditions
        condition = config.conditions{c};
        if isfield(Comparison_centre, condition) && isfield(Comparison_tiled, condition)
            best_idx_centre = Comparison_centre.(condition).bestMatch_idx(1);
            best_idx_tiled = Comparison_tiled.(condition).bestMatch_idx(1);
            tableText{end+1} = sprintf('%s:', upper(condition));
            tableText{end+1} = sprintf('  Centre (%s):', IsingData.simIDs{best_idx_centre});
            tableText{end+1} = sprintf('    beta=%.1f, c=%d, decay=%d', ...
                IsingData.params.beta(best_idx_centre), ...
                IsingData.params.c(best_idx_centre), ...
                IsingData.params.decay_const(best_idx_centre));
            tableText{end+1} = sprintf('    inhib=%d, bias=%.1f', ...
                IsingData.params.inhibition_range(best_idx_centre), ...
                IsingData.params.bias(best_idx_centre));
            tableText{end+1} = sprintf('  Tiled (%s):', IsingData.simIDs{best_idx_tiled});
            tableText{end+1} = sprintf('    beta=%.1f, c=%d, decay=%d', ...
                IsingData.params.beta(best_idx_tiled), ...
                IsingData.params.c(best_idx_tiled), ...
                IsingData.params.decay_const(best_idx_tiled));
            tableText{end+1} = sprintf('    inhib=%d, bias=%.1f', ...
                IsingData.params.inhibition_range(best_idx_tiled), ...
                IsingData.params.bias(best_idx_tiled));
            tableText{end+1} = '';
        end
    end
    text(0.05, 0.95, tableText, 'VerticalAlignment', 'top', ...
        'FontSize', 8, 'FontName', 'FixedWidth', 'Units', 'normalized');
    title('Parameter Summary');

    sgtitle('Best-Match Ising Parameters: Centre Crop vs Tiled Pooled', 'FontWeight', 'bold');

    saveMyFig('BestMatchParameters_CentreVsTiled', fullfile(config.outputPath, 'centreVsTiled'), gcf);
    fprintf('Best-match parameters figure saved\n');
end

% -------------------------------------------------------------------------
% Figure 17b: Parameter Covariance Matrix (Top 10 Best Matches)
% -------------------------------------------------------------------------
fprintf('\n--- Creating Parameter Covariance Matrix Figure ---\n');

figure('Name', 'Parameter Covariance Matrix');

param_names_cov = {'beta', 'c', 'decay_const', 'inhibition_range', 'bias'};
param_labels_cov = {'\beta', 'c', 'decay', 'inhib', 'bias'};
nParams_cov = length(param_names_cov);

% Collect all parameters for pooled analysis
allParamMatrix = [];

% Left side: Per-condition covariance matrices
for c = 1:nConditions
    condition = config.conditions{c};

    if ~isfield(Comparison, condition) || ~isfield(Comparison.(condition), 'bestMatch_params')
        continue;
    end

    % Build parameter matrix [nTopMatches x 5]
    paramMatrix = zeros(config.nTopMatches, nParams_cov);
    for p = 1:nParams_cov
        paramMatrix(:, p) = Comparison.(condition).bestMatch_params.(param_names_cov{p});
    end

    % Accumulate for pooled analysis
    allParamMatrix = [allParamMatrix; paramMatrix];

    % Compute covariance matrix
    covMat = cov(paramMatrix);

    subplot(1, nConditions + 1, c);
    imagesc(covMat);
    colorbar;
    colormap(gca, 'parula');
    axis square;
    set(gca, 'XTick', 1:nParams_cov, 'XTickLabel', param_labels_cov);
    set(gca, 'YTick', 1:nParams_cov, 'YTickLabel', param_labels_cov);
    title(sprintf('%s (n=%d)', condition, config.nTopMatches));
end

% Right side: Pooled covariance matrix
subplot(1, nConditions + 1, nConditions + 1);
covMat_pooled = cov(allParamMatrix);
imagesc(covMat_pooled);
colorbar;
colormap(gca, 'parula');
axis square;
set(gca, 'XTick', 1:nParams_cov, 'XTickLabel', param_labels_cov);
set(gca, 'YTick', 1:nParams_cov, 'YTickLabel', param_labels_cov);
title(sprintf('Pooled (n=%d)', size(allParamMatrix, 1)));

sgtitle('Parameter Covariance Matrix (Top 10 Best Matches)', 'FontWeight', 'bold');
saveMyFig('ParameterCovarianceMatrix', config.outputPath, gcf);
fprintf('Parameter covariance matrix figure saved\n');

% -------------------------------------------------------------------------
% Figure 17c: Parameter Correlation Matrix (Top 10 Best Matches)
% -------------------------------------------------------------------------
fprintf('\n--- Creating Parameter Correlation Matrix Figure ---\n');

figure('Name', 'Parameter Correlation Matrix');

% Collect all parameters for pooled analysis
allParamMatrix = [];

% Left side: Per-condition correlation matrices
for c = 1:nConditions
    condition = config.conditions{c};

    if ~isfield(Comparison, condition) || ~isfield(Comparison.(condition), 'bestMatch_params')
        continue;
    end

    % Build parameter matrix [nTopMatches x 5]
    paramMatrix = zeros(config.nTopMatches, nParams_cov);
    for p = 1:nParams_cov
        paramMatrix(:, p) = Comparison.(condition).bestMatch_params.(param_names_cov{p});
    end

    % Accumulate for pooled analysis
    allParamMatrix = [allParamMatrix; paramMatrix];

    % Compute correlation matrix
    corrMat = corr(paramMatrix);

    subplot(1, nConditions + 1, c);
    imagesc(corrMat, [-1, 1]);
    hold on;
    % Overlay grey on diagonal using RGB image with alpha mask
    n = size(corrMat, 1);
    greyOverlay = repmat(reshape([0.5 0.5 0.5], 1, 1, 3), n, n);
    hOverlay = image(greyOverlay);
    set(hOverlay, 'AlphaData', eye(n));
    hold off;
    colorbar;
    % Red-white-blue diverging colormap
    nColors = 256;
    cmap_div = [linspace(0,1,nColors/2)', linspace(0,1,nColors/2)', ones(nColors/2,1); ...
                ones(nColors/2,1), linspace(1,0,nColors/2)', linspace(1,0,nColors/2)'];
    colormap(gca, cmap_div);
    axis square;
    set(gca, 'XTick', 1:nParams_cov, 'XTickLabel', param_labels_cov);
    set(gca, 'YTick', 1:nParams_cov, 'YTickLabel', param_labels_cov);
    title(sprintf('%s (n=%d)', condition, config.nTopMatches));
end

% Right side: Pooled correlation matrix
subplot(1, nConditions + 1, nConditions + 1);
corrMat_pooled = corr(allParamMatrix);
imagesc(corrMat_pooled, [-1, 1]);
hold on;
% Overlay grey on diagonal using RGB image with alpha mask
n = size(corrMat_pooled, 1);
greyOverlay = repmat(reshape([0.5 0.5 0.5], 1, 1, 3), n, n);
hOverlay = image(greyOverlay);
set(hOverlay, 'AlphaData', eye(n));
hold off;
colorbar;
% Red-white-blue diverging colormap
nColors = 256;
cmap_div = [linspace(0,1,nColors/2)', linspace(0,1,nColors/2)', ones(nColors/2,1); ...
            ones(nColors/2,1), linspace(1,0,nColors/2)', linspace(1,0,nColors/2)'];
colormap(gca, cmap_div);
axis square;
set(gca, 'XTick', 1:nParams_cov, 'XTickLabel', param_labels_cov);
set(gca, 'YTick', 1:nParams_cov, 'YTickLabel', param_labels_cov);
title(sprintf('Pooled (n=%d)', size(allParamMatrix, 1)));

sgtitle('Parameter Correlation Matrix (Top 10 Best Matches)', 'FontWeight', 'bold');
saveMyFig('ParameterCorrelationMatrix', config.outputPath, gcf);
fprintf('Parameter correlation matrix figure saved\n');

% -------------------------------------------------------------------------
% Figure 18: Parameter Sensitivity Analysis (1D) - ALL simulations
% -------------------------------------------------------------------------
fprintf('\n--- Creating Parameter Sensitivity Figure ---\n');

figure('Name', 'Parameter Sensitivity Analysis');

param_names_sens = {'beta', 'c', 'decay_const', 'inhibition_range', 'bias'};
param_labels_sens = {'Beta', 'Coupling (c)', 'Decay Const', 'Inhib Range', 'Bias'};

% Compute TrendAnalysis structure
TrendAnalysis = struct();
TrendAnalysis.slopeAtMin = zeros(5, nConditions);
TrendAnalysis.slopeAtMax = zeros(5, nConditions);
TrendAnalysis.recommendExtendLow = false(5, nConditions);
TrendAnalysis.recommendExtendHigh = false(5, nConditions);
TrendAnalysis.meanW_perParam = cell(5, nConditions);  % Store mean WD curves

for p = 1:5
    subplot(2, 3, p);
    hold on;

    paramName = param_names_sens{p};
    paramValues = config.isingParams.([paramName '_values']);
    nParamVals = length(paramValues);

    for c = 1:nConditions
        condition = config.conditions{c};

        if ~isfield(Comparison, condition)
            continue;
        end

        % Compute mean WD ± SEM at each parameter value
        meanW = zeros(1, nParamVals);
        semW = zeros(1, nParamVals);
        for v = 1:nParamVals
            mask = IsingData.params.(paramName) == paramValues(v);
            W_vals = Comparison.(condition).wasserstein_dist(mask);
            meanW(v) = mean(W_vals, 'omitnan');
            semW(v) = std(W_vals, 'omitnan') / sqrt(sum(mask));
        end

        % Store for later use
        TrendAnalysis.meanW_perParam{p, c} = meanW;

        % Plot with error bars
        errorbar(paramValues, meanW, semW, 'o-', 'Color', config.colors.(condition), ...
            'MarkerFaceColor', config.colors.(condition), 'LineWidth', 1.5, ...
            'MarkerSize', 8, 'DisplayName', condition);

        % Compute slopes at boundaries
        if nParamVals >= 2
            slopeMin = (meanW(2) - meanW(1)) / (paramValues(2) - paramValues(1));
            slopeMax = (meanW(end) - meanW(end-1)) / (paramValues(end) - paramValues(end-1));
            TrendAnalysis.slopeAtMin(p, c) = slopeMin;
            TrendAnalysis.slopeAtMax(p, c) = slopeMax;
            [~, minIdx] = min(meanW);
            TrendAnalysis.recommendExtendLow(p, c) = (slopeMin > 0) || (minIdx == 1);
            TrendAnalysis.recommendExtendHigh(p, c) = (slopeMax < 0) || (minIdx == nParamVals);
        end
    end

    hold off;

    % Add boundary markers
    xline(min(paramValues), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
    xline(max(paramValues), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');

    xlabel(param_labels_sens{p});
    ylabel('Mean Wasserstein');
    title(param_labels_sens{p});

    if p == 1
        legend('Location', 'best');
    end

    % Add extension indicators at boundaries
    yRange = ylim;
    yTop = yRange(2);
    yStep = (yRange(2) - yRange(1)) * 0.08;

    for c = 1:nConditions
        condition = config.conditions{c};
        if ~isfield(config.colors, condition)
            continue;
        end
        condColor = config.colors.(condition);

        if TrendAnalysis.recommendExtendLow(p, c)
            text(paramValues(1), yTop - c*yStep, sprintf('%s<', condition(1)), ...
                'Color', condColor, 'FontWeight', 'bold', 'FontSize', 8, ...
                'HorizontalAlignment', 'center');
        end
        if TrendAnalysis.recommendExtendHigh(p, c)
            text(paramValues(end), yTop - c*yStep, sprintf('>%s', condition(1)), ...
                'Color', condColor, 'FontWeight', 'bold', 'FontSize', 8, ...
                'HorizontalAlignment', 'center');
        end
    end
end

% Subplot 6: Legend and info
subplot(2, 3, 6);
axis off;
hold on;
for c = 1:nConditions
    plot(NaN, NaN, 'o-', 'Color', config.colors.(config.conditions{c}), ...
        'MarkerFaceColor', config.colors.(config.conditions{c}), 'LineWidth', 1.5);
end
hold off;
legend(config.conditions, 'Location', 'north');

infoText = {
    'Red dashed lines = boundaries';
    '';
    'Letters at edges indicate';
    'extension recommendations:';
    '  E< = Expert needs LOWER';
    '  >E = Expert needs HIGHER';
    '';
    'Based on slope at boundary';
};
text(0.5, 0.35, infoText, 'HorizontalAlignment', 'center', 'FontSize', 8);

sgtitle('Parameter Sensitivity: Mean Wasserstein vs Parameter Value', 'FontWeight', 'bold');
saveMyFig('Fig18_ParameterSensitivity', fullfile(config.outputPath, 'ParameterAnalysis'), gcf);
fprintf('Figure 18 (Parameter Sensitivity) saved\n');

% -------------------------------------------------------------------------
% Figure 19a-e: Fixed-Parameter Analysis
% -------------------------------------------------------------------------
fprintf('\n--- Creating Fixed-Parameter Analysis Figures ---\n');

% Store fixed-parameter analysis results
FixedParameterAnalysis = struct();

for fixedParamIdx = 1:5
    fixedParamName = param_names_sens{fixedParamIdx};
    fixedParamLabel = param_labels_sens{fixedParamIdx};
    fixedValues = config.isingParams.([fixedParamName '_values']);
    nFixedValues = length(fixedValues);

    % Get indices of other parameters
    otherParamIdx = setdiff(1:5, fixedParamIdx);

    figure('Name', sprintf('Fixed %s Analysis', fixedParamLabel));

    % Define line styles for different fixed values
    lineStyles = {'-', '--', ':', '-.', '-'};
    if nFixedValues > 5
        lineStyles = repmat(lineStyles, 1, ceil(nFixedValues/5));
    end

    % Initialize storage for this fixed parameter
    FixedParameterAnalysis.(fixedParamName) = struct();
    FixedParameterAnalysis.(fixedParamName).fixedValues = fixedValues;
    FixedParameterAnalysis.(fixedParamName).slopeAtMin = zeros(4, nConditions, nFixedValues);
    FixedParameterAnalysis.(fixedParamName).slopeAtMax = zeros(4, nConditions, nFixedValues);

    % Subplots 1-4: Other parameters
    for sp = 1:4
        subplot(2, 3, sp);
        hold on;

        otherParam = param_names_sens{otherParamIdx(sp)};
        otherParamLabel = param_labels_sens{otherParamIdx(sp)};
        otherValues = config.isingParams.([otherParam '_values']);
        nOtherValues = length(otherValues);

        % For each fixed value, plot Mean WD vs other param
        for fv = 1:nFixedValues
            fixedMask = IsingData.params.(fixedParamName) == fixedValues(fv);

            for c = 1:nConditions
                condition = config.conditions{c};

                if ~isfield(Comparison, condition)
                    continue;
                end

                % Compute mean WD at each other param value (with fixed param constraint)
                meanW = zeros(1, nOtherValues);
                semW = zeros(1, nOtherValues);
                for ov = 1:nOtherValues
                    mask = fixedMask & (IsingData.params.(otherParam) == otherValues(ov));
                    W_vals = Comparison.(condition).wasserstein_dist(mask);
                    if sum(mask) > 0
                        meanW(ov) = mean(W_vals, 'omitnan');
                        semW(ov) = std(W_vals, 'omitnan') / sqrt(sum(mask));
                    else
                        meanW(ov) = NaN;
                        semW(ov) = NaN;
                    end
                end

                % Plot (use line style to distinguish fixed values)
                plot(otherValues, meanW, lineStyles{fv}, 'Color', config.colors.(condition), ...
                    'LineWidth', 1.2, 'HandleVisibility', 'off');

                % Compute slopes at boundaries (for this fixed value)
                if nOtherValues >= 2 && ~any(isnan(meanW))
                    slopeMin = (meanW(2) - meanW(1)) / (otherValues(2) - otherValues(1));
                    slopeMax = (meanW(end) - meanW(end-1)) / (otherValues(end) - otherValues(end-1));
                    FixedParameterAnalysis.(fixedParamName).slopeAtMin(sp, c, fv) = slopeMin;
                    FixedParameterAnalysis.(fixedParamName).slopeAtMax(sp, c, fv) = slopeMax;
                end
            end
        end

        hold off;

        % Add boundary markers
        xline(min(otherValues), 'r--', 'LineWidth', 0.5, 'HandleVisibility', 'off');
        xline(max(otherValues), 'r--', 'LineWidth', 0.5, 'HandleVisibility', 'off');

        xlabel(otherParamLabel);
        ylabel('Mean WD');
        title(otherParamLabel);
    end

    % Subplot 5: Best WD achievable at each fixed value
    subplot(2, 3, 5);
    hold on;

    for c = 1:nConditions
        condition = config.conditions{c};

        if ~isfield(Comparison, condition)
            continue;
        end

        bestWD_perFixed = zeros(1, nFixedValues);
        for fv = 1:nFixedValues
            fixedMask = IsingData.params.(fixedParamName) == fixedValues(fv);
            W_vals = Comparison.(condition).wasserstein_dist(fixedMask);
            if isempty(W_vals)
                bestWD_perFixed(fv) = NaN;
            else
                bestWD_perFixed(fv) = min(W_vals);
            end
        end

        plot(fixedValues, bestWD_perFixed, 'o-', 'Color', config.colors.(condition), ...
            'MarkerFaceColor', config.colors.(condition), 'LineWidth', 1.5, ...
            'MarkerSize', 8, 'DisplayName', condition);
    end

    hold off;
    xlabel(fixedParamLabel);
    ylabel('Best WD (min)');
    title(sprintf('Best Match at Fixed %s', fixedParamLabel));
    legend('Location', 'best');

    % Subplot 6: Legend
    subplot(2, 3, 6);
    axis off;

    % Create legend for line styles (fixed values)
    legendText = {sprintf('Fixed %s values:', fixedParamLabel), ''};
    for fv = 1:nFixedValues
        legendText{end+1} = sprintf('  %s : %.2g', lineStyles{fv}, fixedValues(fv));
    end
    legendText{end+1} = '';
    legendText{end+1} = 'Colors = Conditions';
    legendText{end+1} = '(see subplot 5 legend)';

    text(0.1, 0.9, legendText, 'VerticalAlignment', 'top', ...
        'FontSize', 9, 'FontName', 'FixedWidth', 'Units', 'normalized');

    sgtitle(sprintf('Fixed %s Analysis: How Other Parameters Affect Match Quality', fixedParamLabel), 'FontWeight', 'bold');
    saveMyFig(sprintf('Fig19%s_Fixed_%s', char('a'+fixedParamIdx-1), fixedParamName), ...
              fullfile(config.outputPath, 'ParameterAnalysis'), gcf);
    fprintf('Figure 19%s (Fixed %s) saved\n', char('a'+fixedParamIdx-1), fixedParamName);
end

fprintf('\nAll figures saved to: %s\n', config.outputPath);

% -------------------------------------------------------------------------
% Figure XX: Top 10 Matches Parameter Table (4 Conditions)
% -------------------------------------------------------------------------
fprintf('\n--- Creating Top 10 Matches Parameter Table ---\n');

figure('Name', 'Top 10 Matches Parameters');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for c = 1:length(config.conditions)
    condition = config.conditions{c};

    nexttile;
    axis off;

    if isfield(Comparison, condition)
        % Get top 10 indices
        nMatches = min(10, length(Comparison.(condition).bestMatch_idx));
        top_idx = Comparison.(condition).bestMatch_idx(1:nMatches);

        % Build header
        header = sprintf('%4s %5s %3s %5s %6s %6s', 'Rank', 'Beta', 'c', 'Decay', 'InhibR', 'Bias');
        tableStr = {header, repmat('-', 1, length(header))};

        % Build table rows
        for i = 1:nMatches
            idx = top_idx(i);
            row = sprintf('%4d %5.2f %3d %5d %6d %6.2f', ...
                i, ...
                IsingData.params.beta(idx), ...
                IsingData.params.c(idx), ...
                IsingData.params.decay_const(idx), ...
                IsingData.params.inhibition_range(idx), ...
                IsingData.params.bias(idx));
            tableStr{end+1} = row;
        end

        % Display as text
        text(0.5, 0.5, tableStr, 'FontName', 'FixedWidth', 'FontSize', 8, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    else
        text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
    end

    title(condition, 'FontWeight', 'bold');
end

sgtitle('Top 10 Best-Matching Ising Parameters per Condition', 'FontWeight', 'bold');
saveMyFig('Top10_Parameters_Table', fullfile(config.outputPath, 'Summary'), gcf);
fprintf('Top 10 Parameters Table saved\n');

%% =========================================================================
%% SECTION 9: Save Results
%% =========================================================================

fprintf('\n--- Section 9: Saving Results ---\n');

% Create results structure
Results = struct();
Results.config = config;
Results.IsingData = IsingData;
Results.ExpStats = ExpStats;
Results.Comparison = Comparison;
Results.ParameterTrends = ParameterTrends;
Results.DynamicsAnalysis = DynamicsAnalysis;
Results.timestamp = datetime('now');

% Add TiledAnalysis if available (subselect_tiled mode)
if exist('TiledAnalysis', 'var') && ~isempty(TiledAnalysis)
    Results.TiledAnalysis = TiledAnalysis;
end

% Add PooledMatching analysis if available (subselect_tiled mode)
if exist('PooledMatching', 'var') && ~isempty(PooledMatching)
    Results.PooledMatching = PooledMatching;
end

% Add centreVsTiled analysis if available (subselect_centre_vs_tiled mode)
if exist('centreVsTiled', 'var') && ~isempty(centreVsTiled)
    Results.centreVsTiled = centreVsTiled;
    Results.Comparison_centre = Comparison_centre;
    Results.Comparison_tiled = Comparison_tiled;
end

% Add TrendAnalysis (Figure 18 - Parameter Sensitivity)
if exist('TrendAnalysis', 'var') && ~isempty(TrendAnalysis)
    Results.TrendAnalysis = TrendAnalysis;
end

% Add FixedParameterAnalysis (Figure 19a-e)
if exist('FixedParameterAnalysis', 'var') && ~isempty(FixedParameterAnalysis)
    Results.FixedParameterAnalysis = FixedParameterAnalysis;
end

% Save results
results_path = fullfile(config.outputPath, sprintf('IsingComparison_Results_%s_%s.mat', config.gridMode, config.matchingMetric));
save(results_path, 'Results', '-v7.3');
fprintf('Results saved to: %s\n', results_path);

%% =========================================================================
%% SECTION 10: Summary Report
%% =========================================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('  ISING COMPARISON COMPLETE\n');
fprintf('========================================\n');
fprintf('\n');

fprintf('CONFIGURATION:\n');
fprintf('  Grid mode: %s\n', config.gridMode);
fprintf('  Ising simulations: %d\n', length(IsingData.simIDs));
fprintf('  Conditions: %s\n', strjoin(config.conditions, ', '));
fprintf('\n');

fprintf('BEST MATCHES PER CONDITION:\n');
for c = 1:length(config.conditions)
    condition = config.conditions{c};
    if isfield(Comparison, condition)
        best_idx = Comparison.(condition).bestMatch_idx(1);
        simID = IsingData.simIDs{best_idx};
        wasserstein = Comparison.(condition).wasserstein_dist(best_idx);

        fprintf('\n  %n', condition);
        fprintf('    Best match: %s (Wasserstein=%.4f)\n', simID, wasserstein);
        fprintf('    Parameters: beta=%.2f, c=%d, decay=%d, inhib_range=%d, bias=%.2f\n', ...
            IsingData.params.beta(best_idx), IsingData.params.c(best_idx), ...
            IsingData.params.decay_const(best_idx), IsingData.params.inhibition_range(best_idx), ...
            IsingData.params.bias(best_idx));
        fprintf('    Moran''s I: Exp=%.4f, Ising=%.4f\n', ...
            ExpStats.(condition).MoransI_mean, IsingData.MoransI_mean(best_idx));
    end
end

fprintf('\n');
fprintf('OUTPUT FILES:\n');
fprintf('  Results: %s\n', results_path);
fprintf('  Figures: %s\n', config.outputPath);
fprintf('\n');
fprintf('========================================\n');

%% =========================================================================
%% Helper Functions
%% =========================================================================

function d = wasserstein_1d(x, y)
    % Compute 1D Wasserstein (Earth Mover's) distance between two samples
    % Uses efficient quantile-based approach for large arrays

    % Remove NaN values
    x = x(:);
    y = y(:);
    x = x(~isnan(x));
    y = y(~isnan(y));

    % Handle empty inputs
    if isempty(x) || isempty(y)
        d = NaN;
        return;
    end

    % Sort both samples
    x = sort(x);
    y = sort(y);

    % Use quantile matching for efficiency with large arrays
    % Sample at most 1000 quantiles to avoid memory issues
    n = min(1000, min(length(x), length(y)));
    q = linspace(0, 1, n);

    x_quantiles = quantile(x, q);
    y_quantiles = quantile(y, q);

    % Wasserstein distance = mean absolute difference between quantile functions
    d = mean(abs(x_quantiles - y_quantiles));
end

function cmap = redblue(n)
    % Create red-blue diverging colormap
    if nargin < 1
        n = 64;
    end

    % Red to white to blue
    half = floor(n/2);

    r = [linspace(0, 1, half), linspace(1, 0.7, n-half)];
    g = [linspace(0, 1, half), linspace(1, 0.2, n-half)];
    b = [linspace(0.7, 1, half), linspace(1, 0, n-half)];

    cmap = [r', g', b'];
end

function positions = generateNonOverlappingPositions(simGrid, expGrid)
    % Generate non-overlapping grid positions starting from upper-left
    %
    % Inputs:
    %   simGrid - [rows, cols] of simulation grid
    %   expGrid - [rows, cols] of experimental grid to fit
    %
    % Output:
    %   positions - [nPositions x 2] matrix with [rowStart, colStart] per row

    % Calculate how many non-overlapping grids fit
    nRowPositions = floor(simGrid(1) / expGrid(1));
    nColPositions = floor(simGrid(2) / expGrid(2));

    % Generate starting positions
    rowStarts = 1:expGrid(1):(nRowPositions * expGrid(1));
    colStarts = 1:expGrid(2):(nColPositions * expGrid(2));

    % If no positions fit, try at least one starting from (1,1)
    if isempty(rowStarts)
        if simGrid(1) >= expGrid(1)
            rowStarts = 1;
        else
            rowStarts = [];
        end
    end
    if isempty(colStarts)
        if simGrid(2) >= expGrid(2)
            colStarts = 1;
        else
            colStarts = [];
        end
    end

    % Generate all combinations
    if ~isempty(rowStarts) && ~isempty(colStarts)
        [C, R] = meshgrid(colStarts, rowStarts);
        positions = [R(:), C(:)];
    else
        positions = [];
    end
end

function data = loadHDF5Struct(filepath, groupPath)
    % Load HDF5 group as struct (for Python h5py-saved files)
    if nargin < 2
        groupPath = '/';
    end
    info = h5info(filepath, groupPath);
    data = struct();

    % Fields under /IsingData that should be cell arrays (columns -> cells)
    isingDataCellFields = {'MoransI_all', 'Activity_all', 'MoransI_centre', 'MoransI_tiled', ...
                           'MoransI_pooled', 'MoransI_perPosition'};
    isIsingData = contains(groupPath, '/IsingData');

    % Load datasets
    for i = 1:length(info.Datasets)
        name = info.Datasets(i).Name;
        fullPath = [groupPath '/' name];
        rawData = h5read(filepath, fullPath);

        % Convert column cell arrays to row cell arrays for MATLAB compatibility
        if iscell(rawData) && iscolumn(rawData)
            rawData = rawData';
        end

        % Convert 2D matrices to cell arrays ONLY for IsingData fields
        if isIsingData && ismember(name, isingDataCellFields) && isnumeric(rawData) && ismatrix(rawData)
            % Each column becomes a cell
            nCols = size(rawData, 2);
            cellData = cell(1, nCols);
            for c = 1:nCols
                cellData{c} = rawData(:, c);
            end
            rawData = cellData;
        end

        % Sanitize field name to be valid MATLAB identifier
        validName = matlab.lang.makeValidName(name);
        data.(validName) = rawData;
    end
    % Recursively load groups
    for i = 1:length(info.Groups)
        groupName = info.Groups(i).Name;
        [~, name] = fileparts(groupName);
        % Sanitize field name to be valid MATLAB identifier
        validName = matlab.lang.makeValidName(name);
        data.(validName) = loadHDF5Struct(filepath, groupName);
    end
end


function filename = paramsToFilename(beta, c, decay_const, rad, bias)
    % Generate filename from parameters
    % Format: sim_be_{beta}_c_{c}_d_{decay}_r_{rad}_bi_{bias}.mat
    % Example: sim_be_0.5_c_4_d_6_r_9_bi_-0.8.mat
    filename = sprintf('sim_be_%.1f_c_%d_d_%d_r_%d_bi_%.1f.mat', ...
        beta, c, decay_const, rad, bias);
end


function idStr = paramsToIdString(beta, c, decay_const, rad, bias)
    % Generate a short identifier string from parameters
    % Example: be0.5_c4_d6_r9_bi-0.8
    idStr = sprintf('be%.1f_c%d_d%d_r%d_bi%.1f', beta, c, decay_const, rad, bias);
end


%% =========================================================================
%% Autocorrelation Helper Functions
%% =========================================================================

function [tau, acf, lags, fitResult] = computeAutocorrDecay(timeseries, maxLag, fitRange)
% COMPUTEAUTOCORRDECAY Compute autocorrelation and fit exponential decay
%
% Computes the autocorrelation of a time series and fits an exponential
% decay model to extract the decay time constant (tau).
%
% Inputs:
%   timeseries - 1D time series data (can have NaN values)
%   maxLag     - Maximum lag for autocorrelation computation
%   fitRange   - [minLag, maxLag] range for exponential fit
%
% Outputs:
%   tau       - Decay time constant (in same units as lag)
%   acf       - Autocorrelation function (positive lags only)
%   lags      - Lag values (0 to maxLag)
%   fitResult - Fit statistics struct with fields:
%               .R2       - Coefficient of determination
%               .slope    - Log-linear slope
%               .intercept - Log-linear intercept
%               .method   - Fit method used
%
% Example:
%   [tau, acf, lags, fit] = computeAutocorrDecay(activity, 100, [1, 50]);
%   fprintf('Decay time constant: %.2f frames (R²=%.3f)\n', tau, fit.R2);

    % Handle input
    ts = timeseries(:);
    ts = ts(~isnan(ts));

    % Check for sufficient data
    if length(ts) < maxLag * 2
        tau = NaN;
        acf = [];
        lags = [];
        fitResult = struct('R2', NaN, 'method', 'insufficient_data');
        return;
    end

    % Compute autocorrelation (normalized)
    [acf_full, lags_full] = xcorr(ts - mean(ts), maxLag, 'normalized');

    % Extract positive lags only (including lag 0)
    acf = acf_full(maxLag+1:end);
    lags = lags_full(maxLag+1:end);

    % Fit exponential decay: acf = exp(-lag/tau)
    [tau, fitResult] = fitExponentialDecay(acf, lags, fitRange);
end


function [tau, fitResult] = fitExponentialDecay(acf, lags, fitRange)
% FITEXPONENTIALDECAY Fit simple exponential decay to autocorrelation
%
% Fits the model: acf = exp(-lag / tau) using log-linear regression.
% This is equivalent to fitting: log(acf) = -lag / tau
%
% Inputs:
%   acf      - Autocorrelation values
%   lags     - Lag values (must be same length as acf)
%   fitRange - [minLag, maxLag] range to use for fitting
%
% Outputs:
%   tau       - Decay time constant
%   fitResult - Struct with fit statistics

    % Ensure column vectors
    acf = acf(:);
    lags = lags(:);

    % Select fit range
    idx = lags >= fitRange(1) & lags <= fitRange(2);
    acf_fit = acf(idx);
    lags_fit = lags(idx);

    % Handle negative/zero values (need positive for log)
    valid = acf_fit > 0;
    if sum(valid) < 3
        tau = NaN;
        fitResult = struct('R2', NaN, 'method', 'insufficient_positive_values');
        return;
    end

    log_acf = log(acf_fit(valid));
    lags_valid = lags_fit(valid);

    % Linear regression: log(acf) = -lag/tau
    % Slope = -1/tau, so tau = -1/slope
    p = polyfit(lags_valid, log_acf, 1);
    tau = -1 / p(1);

    % Handle negative tau (if autocorrelation increases with lag)
    if tau < 0
        tau = NaN;
        fitResult = struct('R2', NaN, 'method', 'negative_slope', 'slope', p(1));
        return;
    end

    % Compute R² (coefficient of determination)
    predicted = polyval(p, lags_valid);
    SS_res = sum((log_acf - predicted).^2);
    SS_tot = sum((log_acf - mean(log_acf)).^2);

    if SS_tot == 0
        R2 = NaN;
    else
        R2 = 1 - SS_res / SS_tot;
    end

    fitResult = struct('R2', R2, 'slope', p(1), 'intercept', p(2), 'method', 'log-linear');
end


function tau_ratio = computeTimeScaling(tau_exp, tau_ising)
% COMPUTETIMESCALING Compute ratio for MC sweep to frame conversion
%
% Given the decay time constants from experimental data and Ising simulation,
% computes the scaling factor to convert between MC sweeps and real time frames.
%
% Inputs:
%   tau_exp   - Experimental decay time constant (in frames)
%   tau_ising - Ising simulation decay time constant (in MC sweeps)
%
% Output:
%   tau_ratio - Scaling factor: 1 MC sweep = tau_ratio frames
%
% Usage:
%   To convert Ising time to real time: real_time = ising_time * tau_ratio

    if isnan(tau_exp) || isnan(tau_ising) || tau_exp <= 0
        tau_ratio = NaN;
    else
        tau_ratio = tau_ising / tau_exp;
    end
end


function [tau, acf_mean, lags, fitResult, acf_std] = computeTrialAveragedAutocorr(data, maxLag, fitRange)
% COMPUTETRIALAVERAGEDAUTOCORR Compute trial-averaged autocorrelation
%
% Computes autocorrelation for each trial independently, averages across
% trials, then fits exponential decay. Avoids artifacts from trial concatenation.
%
% Inputs:
%   data     - [nTrials x nTimepoints] matrix
%   maxLag   - Maximum lag for autocorrelation
%   fitRange - [minLag, maxLag] for exponential fit
%
% Outputs:
%   tau      - Decay time constant from averaged ACF
%   acf_mean - Mean autocorrelation across trials
%   lags     - Lag values (0:maxLag)
%   fitResult - Fit statistics (R2, etc.)
%   acf_std  - Standard deviation across trials

    [nTrials, ~] = size(data);
    acf_all = zeros(nTrials, maxLag + 1);
    validTrials = 0;

    for t = 1:nTrials
        trial_data = data(t, :);
        trial_data = trial_data(~isnan(trial_data));
        if length(trial_data) > maxLag * 2
            [acf, ~] = xcorr(trial_data - mean(trial_data), maxLag, 'normalized');
            acf_all(t, :) = acf(maxLag+1:end);
            validTrials = validTrials + 1;
        else
            acf_all(t, :) = NaN;
        end
    end

    acf_mean = mean(acf_all, 1, 'omitnan');
    acf_std = std(acf_all, 0, 1, 'omitnan');
    lags = (0:maxLag)';

    % Fit exponential decay using existing function
    [tau, fitResult] = fitExponentialDecay(acf_mean, lags, fitRange);
    fitResult.validTrials = validTrials;
    fitResult.totalTrials = nTrials;
end


function setPublicationDefaults()
% SETPUBLICATIONDEFAULTS Apply publication-quality figure defaults
%
% Sets global defaults for Arial font, publication-standard sizes,
% and clean axis styling. Call once before creating figures.

    set(groot, 'DefaultAxesFontName', 'Arial');
    set(groot, 'DefaultTextFontName', 'Arial');
    set(groot, 'DefaultAxesFontSize', 6);          % 6pt tick labels (base)
    set(groot, 'DefaultAxesTitleFontSizeMultiplier', 8/6);  % 8pt titles
    set(groot, 'DefaultAxesLabelFontSizeMultiplier', 7/6);  % 7pt axis labels
    set(groot, 'DefaultTextFontSize', 7);
    set(groot, 'DefaultAxesTickDir', 'out');
    set(groot, 'DefaultAxesBox', 'off');
    set(groot, 'DefaultAxesLineWidth', 0.5);
    set(groot, 'DefaultLineLineWidth', 1.0);
    set(groot, 'DefaultAxesTickLength', [0.02 0.025]);
end


function plotUMAPComparison(embedding_ising, embedding_exp, ...
    Comparison, conditions, colors, top_n, figureName, saveDir, outputPath)
% PLOTUMAPCOMPARISON Plot UMAP scatter with best-match highlights
%
% Plots all Ising simulations in gray, overlays top-N best matches
% per condition with condition colors, and adds experimental markers.
%
% Inputs:
%   embedding_ising - [nSims x 2] UMAP embedding for Ising sims
%   embedding_exp   - [nConds x 2] UMAP embedding for experimental
%   Comparison      - Comparison struct with bestMatch_idx per condition
%   conditions      - Cell array of condition names
%   colors          - Struct with RGB color per condition
%   top_n           - Number of top matches to highlight
%   figureName      - Name for the figure
%   saveDir         - Subdirectory for saving (e.g., 'UMAP')
%   outputPath      - Base output path

    figure('Name', figureName);

    % Plot all Ising simulations in gray
    scatter(embedding_ising(:,1), embedding_ising(:,2), 100, [0.7 0.7 0.7], ...
        'filled', 'MarkerFaceAlpha', 0.4, 'HandleVisibility', 'off');
    hold on;

    nConds = length(conditions);

    % Highlight top-N best-matching Ising simulations for each condition
    for c = 1:nConds
        condition = conditions{c};
        if isfield(Comparison, condition)
            nTop = min(top_n, length(Comparison.(condition).bestMatch_idx));
            best_idx = Comparison.(condition).bestMatch_idx(1:nTop);
            conditionColor = colors.(condition);
            scatter(embedding_ising(best_idx, 1), embedding_ising(best_idx, 2), 100, ...
                conditionColor, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
                'DisplayName', sprintf('%s (Top %d)', condition, top_n));
        end
    end

    % Plot experimental conditions (larger markers, star shape)
    if ~isempty(embedding_exp)
        for c = 1:nConds
            condition = conditions{c};
            conditionColor = colors.(condition);
            scatter(embedding_exp(c, 1), embedding_exp(c, 2), 300, conditionColor, ...
                'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2, ...
                'DisplayName', [condition ' (Exp)']);
        end
    end

    hold off;
    xlabel('UMAP 1');
    ylabel('UMAP 2');
    title(figureName);
    legend('Location', 'best');

    % Save figure
    saveMyFig(matlab.lang.makeValidName(figureName), fullfile(outputPath, saveDir), gcf);
end


function dataOut = applyNostimSelection(dataIn, config, format)
% APPLYNOSTIMSELECTION Split data into pre-stim and post-stim sub-trials
%
% Splits each trial into two independent sub-trials by removing stimulus
% frames, then concatenates along the trial dimension.
%   Pre-stim:  frames 1 to expPrestimFrames
%   Post-stim: frames (expStimOffsetFrame+1) to (expStimOffsetFrame+expPrestimFrames)
%
% Formats:
%   'trials_x_time' - [N x T] -> [2N x prestimFrames]
%   'spatial_4d'    - [rows x cols x T x N] -> [rows x cols x prestimFrames x 2N]
%   'spatial_3d'    - [rows x cols x T] -> [rows x cols x 2*prestimFrames]

    preFrames = 1:config.expPrestimFrames;
    postStart = config.expStimOffsetFrame + 1;
    postEnd   = config.expStimOffsetFrame + config.expPrestimFrames;

    switch format
        case 'trials_x_time'
            % dataIn is [nTrials x nTimepoints]
            postEnd = min(postEnd, size(dataIn, 2));
            prePart  = dataIn(:, preFrames);
            postPart = dataIn(:, postStart:postEnd);
            dataOut  = [prePart; postPart];

        case 'spatial_4d'
            % dataIn is [rows x cols x nTimepoints x nTrials]
            postEnd = min(postEnd, size(dataIn, 3));
            prePart  = dataIn(:, :, preFrames, :);
            postPart = dataIn(:, :, postStart:postEnd, :);
            dataOut  = cat(4, prePart, postPart);

        case 'spatial_3d'
            % dataIn is [rows x cols x nTimepoints] (no trial dimension)
            postEnd = min(postEnd, size(dataIn, 3));
            prePart  = dataIn(:, :, preFrames);
            postPart = dataIn(:, :, postStart:postEnd);
            dataOut  = cat(3, prePart, postPart);

        otherwise
            error('applyNostimSelection: Unknown format ''%s''', format);
    end
end
