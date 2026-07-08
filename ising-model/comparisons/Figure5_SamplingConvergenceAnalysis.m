%% =========================================================================
%% Figure 5: Sampling Convergence Analysis for Ising Simulations
%% =========================================================================
%
% SETUP: Add repository to path (required for cluster execution)
% -------------------------------------------------------------------------
scriptPath = fileparts(mfilename('fullpath'));
repoRoot = fullfile(scriptPath, '..', '..', '..', '..');  % Navigate up to MouseBrainActivity
addpath(genpath(repoRoot));
fprintf('Added repository to path: %s\n', repoRoot);
% -------------------------------------------------------------------------
%
% PURPOSE:
% Answer two fundamental questions about Ising simulations WITHOUT
% comparison to experimental data:
%
% (1) SPATIAL HOMOGENEITY:
%     Is there spatial inhomogeneity in sampling area?
%     If P1 differs from Centre, it could indicate a simulation bug.
%
% (2) SAMPLE POOLING BENEFIT:
%     Does pooling samples from P1+P2 improve statistical estimates
%     of the Moran's I distribution?
%
% METHODOLOGY:
% - Define Ground Truth (G): Centre crop, full simulation duration T
% - Temporal subsampling: T/16, T/11, T/8, ... T/2 with sqrt(2) spacing
% - Compare WD(G, subset) for Centre, P1, and P1+P2 pooled regions
%
% OUTPUT:
% - Figure 1: Main convergence plot (WD vs inverse data fraction)
% - Figure 2: Spatial regions diagram
% - Figure 3: Square grid analysis (spatial homogeneity vs grid size)
% - Figure 4: Distance-from-centre analysis (2x2 grid)

%% =========================================================================


%% =========================================================================
%% SECTION 1: Configuration
%% =========================================================================

fprintf('=== Figure 5: Sampling Convergence Analysis ===\n');
fprintf('--- Section 1: Configuration ---\n');

config = struct();

% -------------------------------------------------------------------------
% Environment Detection (Local vs Cluster)
% -------------------------------------------------------------------------
% Detect if running on cluster by checking for NFS path or hostname
if ispc
    % Windows = local machine
    config.isCluster = false;
    fprintf('Environment: LOCAL (Windows)\n');
elseif isfolder('/path/to/data')
    % Linux with NFS path = ISTA cluster
    config.isCluster = true;
    fprintf('Environment: CLUSTER (ISTA HPC)\n');
else
    % Default to local if unsure
    config.isCluster = false;
    fprintf('Environment: LOCAL (unknown system)\n');
end

% -------------------------------------------------------------------------
% Analysis Mode Selection
% -------------------------------------------------------------------------
% 'best_match': Use best-match simulations per condition (from IsingComparison_Results.mat)
% 'all_sims': Process all simulations
config.analysisMode = 'all_sims';  % Change to 'best_match' for subset

% -------------------------------------------------------------------------
% Paths (Environment-dependent)
% -------------------------------------------------------------------------
if config.isCluster
    % CLUSTER PATHS
    config.isingDataPath = '/path/to/data/IsingSims';
    config.resultsPath = '/path/to/data/IsingSims/IsingComparison_Results.mat';
    config.outputPathBase = '/path/to/data/IsingSims/SamplingConvergence';
else
    % LOCAL PATHS (Windows)
    % config.isingDataPath = mba_p('IsingModelData_for_Florian');  % 32x32 grid, ~1000 sims
    config.isingDataPath = mba_p('IsingModelData_39x78_100K');    % 39x78 grid, 100K timesteps
    config.resultsPath = 'Fig. 5 Model\IsingModels\IsingComparison\Data\IsingComparison_Results.mat';
    config.outputPathBase = 'Fig. 5 Model\IsingModels\IsingComparison\SamplingConvergence';
end
% Create subfolder based on analysis mode
config.outputPath = fullfile(config.outputPathBase, config.analysisMode);

% -------------------------------------------------------------------------
% Grid Configuration
% -------------------------------------------------------------------------
% config.isingGrid will be auto-detected from simulation data
config.expGrid = [4, 4];         % Experimental crop size (fixed to match FOV)

% -------------------------------------------------------------------------
% Temporal Subsampling Configuration
% -------------------------------------------------------------------------
% Use sqrt(2) spacing for denser x-axis coverage
% Values represent inverse fractions: 16 means T/16, 8 means T/8, etc.
config.inverseFractions = [16, 16/sqrt(2), 8, 8/sqrt(2), 4, 4/sqrt(2), 2];
% Approximate values: [16, 11.3, 8, 5.7, 4, 2.8, 2]

fprintf('Analysis mode: %s\n', config.analysisMode);
fprintf('Experimental grid: [%d x %d]\n', config.expGrid(1), config.expGrid(2));
fprintf('Inverse fractions: ');
fprintf('%.1f ', config.inverseFractions);
fprintf('\n');
fprintf('(Ising grid size will be auto-detected from simulation data)\n');

%% =========================================================================
%% CHECK FOR PRE-COMPUTED RESULTS (from Python script)
%% =========================================================================
% If SamplingConvergence_Results.mat exists, skip data processing and
% jump directly to figure generation.

% Check in the isingDataPath/SamplingConvergence folder
precomputedPath = fullfile(config.isingDataPath, 'SamplingConvergence', 'SamplingConvergence_Results.mat');

if exist(precomputedPath, 'file')
    fprintf('\n');
    fprintf('=========================================================\n');
    fprintf('  FOUND PRE-COMPUTED RESULTS\n');
    fprintf('=========================================================\n');
    fprintf('Loading: %s\n', precomputedPath);

    precomputed = load(precomputedPath);

    % Extract Results structure (handle Python vs MATLAB format differences)
    if isfield(precomputed, 'Results')
        Results = precomputed.Results;
    else
        Results = precomputed;
    end

    % Update config with loaded values
    if isfield(Results, 'config')
        loadedConfig = Results.config;
        if isfield(loadedConfig, 'isingGrid')
            config.isingGrid = loadedConfig.isingGrid;
        end
        if isfield(loadedConfig, 'nSims')
            nSims = loadedConfig.nSims;
        else
            nSims = 1;  % Unknown
        end
    else
        nSims = 1;
    end

    % Extract aggregate results
    if isfield(Results, 'Aggregate')
        Aggregate = Results.Aggregate;
    end

    % Extract square grid results
    if isfield(Results, 'SquareGridResults')
        SquareGridResults = Results.SquareGridResults;
    end

    % Extract distance analysis
    if isfield(Results, 'DistanceAnalysis')
        DistanceAnalysis = Results.DistanceAnalysis;
    end

    fprintf('Loaded results for %d simulations\n', nSims);
    fprintf('Skipping data processing, jumping to figure generation...\n');
    fprintf('=========================================================\n\n');

    % Jump to figure generation (Section 7)
    % We need to define some variables that the figure sections expect
    config.outputPath = fullfile(config.isingDataPath, 'SamplingConvergence');
    if ~exist(config.outputPath, 'dir')
        mkdir(config.outputPath);
    end

    % Go directly to Section 7 (Figure 1)
    goto_figure_generation = true;
else
    fprintf('\nNo pre-computed results found at: %s\n', precomputedPath);
    fprintf('Will process simulation data from scratch.\n\n');
    goto_figure_generation = false;
end

%% =========================================================================
%% SECTION 2: Load Simulation Data
%% =========================================================================

if ~goto_figure_generation

fprintf('\n--- Section 2: Load Simulation Data ---\n');

% Create output directory if it doesn't exist
if ~exist(config.outputPath, 'dir')
    mkdir(config.outputPath);
    fprintf('Created output directory: %s\n', config.outputPath);
end

% Get list of simulations based on analysis mode
if strcmp(config.analysisMode, 'best_match')
    % Load results to get best-match simulation IDs
    if exist(config.resultsPath, 'file')
        resultsData = load(config.resultsPath);
        fprintf('Loaded IsingComparison_Results.mat\n');
    else
        error('Results file not found (required for best_match mode): %s', config.resultsPath);
    end
    % Use best-match simulations from IsingComparison results
    if isfield(resultsData, 'Results') && isfield(resultsData.Results, 'bestMatchSimIDs')
        simIDs = resultsData.Results.bestMatchSimIDs;
    else
        % Fallback: use unique simulation IDs from results
        simIDs = unique(resultsData.Results.simID);
    end
    fprintf('Using %d best-match simulations\n', length(simIDs));
else
    % Use all simulations
    simFiles = dir(fullfile(config.isingDataPath, 'sim_*.mat'));
    simIDs = nan(length(simFiles), 1);  % Use NaN instead of 0 to allow sim_0.mat
    for i = 1:length(simFiles)
        tokens = regexp(simFiles(i).name, 'sim_(\d+)\.mat', 'tokens');
        if ~isempty(tokens)
            simIDs(i) = str2double(tokens{1}{1});
        end
    end
    simIDs = simIDs(~isnan(simIDs));  % Keep all valid IDs including 0
    fprintf('Using all %d simulations\n', length(simIDs));
end

nSims = length(simIDs);

% Auto-detect grid size from first simulation file
if ~isempty(simIDs)
    firstSimPath = fullfile(config.isingDataPath, sprintf('sim_%d.mat', simIDs(1)));
    firstSimInfo = whos('-file', firstSimPath, 'stored_spins');
    % stored_spins is [T x rows x cols]
    config.isingGrid = [firstSimInfo.size(2), firstSimInfo.size(3)];
    fprintf('Auto-detected Ising grid: [%d x %d]\n', config.isingGrid(1), config.isingGrid(2));
else
    error('No simulation files found in: %s', config.isingDataPath);
end

%% =========================================================================
%% SECTION 3: Define Spatial Regions
%% =========================================================================

fprintf('\n--- Section 3: Define Spatial Regions ---\n');

% Centre crop (matches experimental FOV position)
% Centered on the Ising grid
centre_rowStart = floor((config.isingGrid(1) - config.expGrid(1)) / 2) + 1;  % 10
centre_colStart = floor((config.isingGrid(2) - config.expGrid(2)) / 2) + 1;  % 4
centre_rows = centre_rowStart:(centre_rowStart + config.expGrid(1) - 1);     % 10:22
centre_cols = centre_colStart:(centre_colStart + config.expGrid(2) - 1);     % 4:29

% P1: Top-left position (non-central, tests spatial homogeneity)
P1_rows = 1:config.expGrid(1);      % 1:13
P1_cols = 1:config.expGrid(2);      % 1:26

% P2: Position below P1 (for pooling test)
P2_rows = (config.expGrid(1) + 1):(2 * config.expGrid(1));  % 14:26
P2_cols = 1:config.expGrid(2);      % 1:26

fprintf('Centre crop: rows %d:%d, cols %d:%d\n', centre_rows(1), centre_rows(end), centre_cols(1), centre_cols(end));
fprintf('P1: rows %d:%d, cols %d:%d\n', P1_rows(1), P1_rows(end), P1_cols(1), P1_cols(end));
fprintf('P2: rows %d:%d, cols %d:%d\n', P2_rows(1), P2_rows(end), P2_cols(1), P2_cols(end));

%% =========================================================================
%% SECTION 4: Create Weight Matrix for Moran's I
%% =========================================================================

fprintf('\n--- Section 4: Create Weight Matrix ---\n');

% Create weight matrix for the experimental grid size
valueMap = rand(config.expGrid(1), config.expGrid(2));
distanceMat = squareform(mL_distanceMat(valueMap));
uniqueDistances = unique(distanceMat);
uniqueDistances(uniqueDistances == 0) = [];

% Use nearest neighbor distances only
currDistInds = ismember(distanceMat, uniqueDistances(1));
weightMat = zeros(size(distanceMat));
weightMat(currDistInds) = distanceMat(currDistInds);
weightMat(weightMat == inf) = 0;

fprintf('Weight matrix size: [%d x %d]\n', size(weightMat, 1), size(weightMat, 2));

%% =========================================================================
%% SECTION 5: Main Analysis Loop
%% =========================================================================

fprintf('\n--- Section 5: Main Analysis Loop ---\n');

nFractions = length(config.inverseFractions);

% Initialize storage for all simulations
Results = struct();
Results.config = config;
Results.simIDs = simIDs;
Results.inverseFractions = config.inverseFractions;

% Storage for aggregated results
% Each cell contains WD values: [nSims x nSegments]
WD_Centre_all = cell(nFractions, 1);
WD_P1_all = cell(nFractions, 1);
WD_P1P2_all = cell(nFractions, 1);

for f = 1:nFractions
    WD_Centre_all{f} = [];
    WD_P1_all{f} = [];
    WD_P1P2_all{f} = [];
end

% Process each simulation
for s = 1:nSims
    simID = simIDs(s);
    simPath = fullfile(config.isingDataPath, sprintf('sim_%d.mat', simID));

    if ~exist(simPath, 'file')
        fprintf('  Skipping sim_%d (file not found)\n', simID);
        continue;
    end

    % Load simulation
    simData = load(simPath);
    stored_spins = simData.stored_spins;  % [T x h x w]
    T = size(stored_spins, 1);  % Should be 4000

    % Compute Moran's I for full duration (Ground Truth)
    moransI_Centre = zeros(1, T);
    moransI_P1 = zeros(1, T);
    moransI_P2 = zeros(1, T);

    for t = 1:T
        frame = squeeze(stored_spins(t, :, :));

        % Centre crop
        frame_centre = frame(centre_rows, centre_cols);
        if all(frame_centre(:) == 0) || all(frame_centre(:) == 1)
            moransI_Centre(t) = NaN;
        else
            moransI_Centre(t) = mL_moransI(double(frame_centre), weightMat);
        end

        % P1
        frame_P1 = frame(P1_rows, P1_cols);
        if all(frame_P1(:) == 0) || all(frame_P1(:) == 1)
            moransI_P1(t) = NaN;
        else
            moransI_P1(t) = mL_moransI(double(frame_P1), weightMat);
        end

        % P2
        frame_P2 = frame(P2_rows, P2_cols);
        if all(frame_P2(:) == 0) || all(frame_P2(:) == 1)
            moransI_P2(t) = NaN;
        else
            moransI_P2(t) = mL_moransI(double(frame_P2), weightMat);
        end
    end

    % Ground Truth: Centre, full T (remove NaNs)
    GT_clean = moransI_Centre(~isnan(moransI_Centre));

    % Temporal subsampling analysis
    for f = 1:nFractions
        invFrac = config.inverseFractions(f);
        segmentLength = floor(T / invFrac);
        nSegments = floor(T / segmentLength);

        for seg = 1:nSegments
            startIdx = (seg - 1) * segmentLength + 1;
            endIdx = seg * segmentLength;

            % Extract segments
            seg_Centre = moransI_Centre(startIdx:endIdx);
            seg_P1 = moransI_P1(startIdx:endIdx);
            seg_P2 = moransI_P2(startIdx:endIdx);

            % Clean NaNs
            seg_Centre_clean = seg_Centre(~isnan(seg_Centre));
            seg_P1_clean = seg_P1(~isnan(seg_P1));
            seg_P2_clean = seg_P2(~isnan(seg_P2));

            % P1+P2 pooled
            seg_P1P2_clean = [seg_P1_clean, seg_P2_clean];

            % Compute WD(G, segment)
            WD_Centre = wasserstein_1d(GT_clean, seg_Centre_clean);
            WD_P1 = wasserstein_1d(GT_clean, seg_P1_clean);
            WD_P1P2 = wasserstein_1d(GT_clean, seg_P1P2_clean);

            % Store
            WD_Centre_all{f}(end+1) = WD_Centre;
            WD_P1_all{f}(end+1) = WD_P1;
            WD_P1P2_all{f}(end+1) = WD_P1P2;
        end
    end

    % Clear memory
    clear stored_spins simData;

    % Progress update
    if mod(s, max(1, floor(nSims/10))) == 0
        fprintf('  Processed %d/%d simulations\n', s, nSims);
    end
end

fprintf('Analysis complete.\n');

%% =========================================================================
%% SECTION 6: Aggregate Results
%% =========================================================================

fprintf('\n--- Section 6: Aggregate Results ---\n');

% Compute mean and std for each fraction (use 'omitnan' to handle NaN values)
Aggregate = struct();
Aggregate.inverseFractions = config.inverseFractions;
Aggregate.WD_Centre_mean = zeros(nFractions, 1);
Aggregate.WD_Centre_std = zeros(nFractions, 1);
Aggregate.WD_P1_mean = zeros(nFractions, 1);
Aggregate.WD_P1_std = zeros(nFractions, 1);
Aggregate.WD_P1P2_mean = zeros(nFractions, 1);
Aggregate.WD_P1P2_std = zeros(nFractions, 1);

for f = 1:nFractions
    Aggregate.WD_Centre_mean(f) = mean(WD_Centre_all{f}, 'omitnan');
    Aggregate.WD_Centre_std(f) = std(WD_Centre_all{f}, 'omitnan');
    Aggregate.WD_P1_mean(f) = mean(WD_P1_all{f}, 'omitnan');
    Aggregate.WD_P1_std(f) = std(WD_P1_all{f}, 'omitnan');
    Aggregate.WD_P1P2_mean(f) = mean(WD_P1P2_all{f}, 'omitnan');
    Aggregate.WD_P1P2_std(f) = std(WD_P1P2_all{f}, 'omitnan');

    fprintf('Fraction 1/%.1f: Centre=%.4f, P1=%.4f, P1+P2=%.4f\n', ...
        config.inverseFractions(f), ...
        Aggregate.WD_Centre_mean(f), ...
        Aggregate.WD_P1_mean(f), ...
        Aggregate.WD_P1P2_mean(f));
end

Results.Aggregate = Aggregate;
Results.WD_Centre_all = WD_Centre_all;
Results.WD_P1_all = WD_P1_all;
Results.WD_P1P2_all = WD_P1P2_all;

end  % End of: if ~goto_figure_generation (data processing sections 2-6)

%% =========================================================================
%% SECTION 7: Visualization (Figure 1)
%% =========================================================================
% This section and below run regardless of whether data was computed fresh
% or loaded from pre-computed results.

% Ensure required variables are available for visualization
nFractions = length(config.inverseFractions);

% If loaded from pre-computed results, extract WD_all arrays and compute region definitions
if goto_figure_generation
    % Extract WD_all arrays from Results (top-level, not inside Aggregate)
    % Python saves as [nFractions × nValues] 2D array, convert to cell array
    if isfield(Results, 'WD_Centre_all')
        WD_Centre_2d = Results.WD_Centre_all;
        WD_P1_2d = Results.WD_P1_all;
        WD_P1P2_2d = Results.WD_P1P2_all;

        % Create cell arrays (one cell per fraction)
        nFrac = size(WD_Centre_2d, 1);
        WD_Centre_all = cell(1, nFrac);
        WD_P1_all = cell(1, nFrac);
        WD_P1P2_all = cell(1, nFrac);
        for f = 1:nFrac
            WD_Centre_all{f} = WD_Centre_2d(f, :)';
            WD_P1_all{f} = WD_P1_2d(f, :)';
            WD_P1P2_all{f} = WD_P1P2_2d(f, :)';
        end
    end

    % Compute region definitions from config (needed for Figure 2)
    centre_rowStart = floor((config.isingGrid(1) - config.expGrid(1)) / 2) + 1;
    centre_colStart = floor((config.isingGrid(2) - config.expGrid(2)) / 2) + 1;
    centre_rows = centre_rowStart:(centre_rowStart + config.expGrid(1) - 1);
    centre_cols = centre_colStart:(centre_colStart + config.expGrid(2) - 1);

    P1_rows = 1:config.expGrid(1);
    P1_cols = 1:config.expGrid(2);

    P2_rows = (config.expGrid(1) + 1):(2 * config.expGrid(1));
    P2_cols = 1:config.expGrid(2);

    % GT_clean is not available from pre-computed results - set to empty for summary
    GT_clean = [];
end

fprintf('\n--- Section 7: Visualization ---\n');

% =========================================================================
% FIGURE 1: Main Convergence Plot
% =========================================================================
figure('Name', 'Sampling Convergence Analysis');

% Panel 1: WD vs Inverse Data Fraction with individual points
subplot(2, 2, 1);
hold on;

% Colors
colorCentre = [0.2, 0.2, 0.2];   % Dark gray
colorP1 = [0.2, 0.4, 0.8];       % Blue
colorP1P2 = [0.8, 0.2, 0.2];     % Red

% Plot individual points with jitter
jitterAmount = 0.15;
for f = 1:nFractions
    x_base = config.inverseFractions(f);

    % Centre
    x_jitter = x_base + randn(size(WD_Centre_all{f})) * jitterAmount;
    scatter(x_jitter, WD_Centre_all{f}, 8, colorCentre, 'filled', 'MarkerFaceAlpha', 0.2);

    % P1
    x_jitter = x_base + randn(size(WD_P1_all{f})) * jitterAmount;
    scatter(x_jitter, WD_P1_all{f}, 8, colorP1, 'filled', 'MarkerFaceAlpha', 0.2);

    % P1+P2
    x_jitter = x_base + randn(size(WD_P1P2_all{f})) * jitterAmount;
    scatter(x_jitter, WD_P1P2_all{f}, 8, colorP1P2, 'filled', 'MarkerFaceAlpha', 0.2);
end

% Plot mean lines
plot(config.inverseFractions, Aggregate.WD_Centre_mean, 'o-', 'Color', colorCentre, ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colorCentre);
plot(config.inverseFractions, Aggregate.WD_P1_mean, 's-', 'Color', colorP1, ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colorP1);
plot(config.inverseFractions, Aggregate.WD_P1P2_mean, 'd-', 'Color', colorP1P2, ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colorP1P2);

hold off;
xlabel('Inverse Data Fraction (1/f)');
ylabel('WD(G, subset)');
title('Sampling Convergence: WD vs Sample Size');
legend({'', '', '', 'Centre', 'P1', 'P1+P2 pooled'}, 'Location', 'best');
set(gca, 'XScale', 'log');
grid on;

% Panel 2: Same plot but log-log scale
subplot(2, 2, 2);
hold on;
plot(config.inverseFractions, Aggregate.WD_Centre_mean, 'o-', 'Color', colorCentre, ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colorCentre);
plot(config.inverseFractions, Aggregate.WD_P1_mean, 's-', 'Color', colorP1, ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colorP1);
plot(config.inverseFractions, Aggregate.WD_P1P2_mean, 'd-', 'Color', colorP1P2, ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colorP1P2);
hold off;
xlabel('Inverse Data Fraction (1/f)');
ylabel('WD(G, subset)');
title('Log-Log Scale');
legend({'Centre', 'P1', 'P1+P2 pooled'}, 'Location', 'best');
set(gca, 'XScale', 'log', 'YScale', 'log');
grid on;

% Panel 3: Spatial Homogeneity (P1 - Centre difference)
subplot(2, 2, 3);
diff_P1_Centre = Aggregate.WD_P1_mean - Aggregate.WD_Centre_mean;
bar(1:nFractions, diff_P1_Centre, 'FaceColor', colorP1);
hold on;
yline(0, 'k--', 'LineWidth', 1.5);
hold off;
xticks(1:nFractions);
xticklabels(arrayfun(@(x) sprintf('1/%.0f', x), config.inverseFractions, 'UniformOutput', false));
xlabel('Data Fraction');
ylabel('WD(P1) - WD(Centre)');
title('Spatial Homogeneity Test');
if mean(diff_P1_Centre) > 0.005
    text(0.5, 0.9, 'P1 > Centre: Possible spatial bias', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'Color', 'r', 'FontWeight', 'bold');
else
    text(0.5, 0.9, 'Spatially homogeneous', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'Color', [0.2, 0.7, 0.3], 'FontWeight', 'bold');
end
grid on;

% Panel 4: Pooling Benefit (P1+P2 - P1 difference)
subplot(2, 2, 4);
diff_P1P2_P1 = Aggregate.WD_P1P2_mean - Aggregate.WD_P1_mean;
barColors = zeros(nFractions, 3);
for f = 1:nFractions
    if diff_P1P2_P1(f) < 0
        barColors(f, :) = [0.2, 0.7, 0.3];  % Green = improvement
    else
        barColors(f, :) = [0.8, 0.3, 0.3];  % Red = no improvement
    end
end
for f = 1:nFractions
    bar(f, diff_P1P2_P1(f), 'FaceColor', barColors(f, :));
    hold on;
end
yline(0, 'k--', 'LineWidth', 1.5);
hold off;
xticks(1:nFractions);
xticklabels(arrayfun(@(x) sprintf('1/%.0f', x), config.inverseFractions, 'UniformOutput', false));
xlabel('Data Fraction');
ylabel('WD(P1+P2) - WD(P1)');
title('Pooling Benefit Test');
grid on;

sgtitle(sprintf('Sampling Convergence Analysis (%s, n=%d sims)', config.analysisMode, nSims), 'FontWeight', 'bold');

saveMyFig('SamplingConvergence_MainFigure', config.outputPath, gcf);
fprintf('Figure 1 saved\n');

% =========================================================================
% FIGURE 2: Spatial Regions Diagram
% =========================================================================
figure('Name', 'Spatial Regions');

% Create visualization of the grid with regions marked
regionMap = zeros(config.isingGrid);
regionMap(centre_rows, centre_cols) = 1;  % Centre
regionMap(P1_rows, P1_cols) = regionMap(P1_rows, P1_cols) + 2;  % P1
regionMap(P2_rows, P2_cols) = regionMap(P2_rows, P2_cols) + 4;  % P2

imagesc(regionMap);
colormap([1 1 1; 0.5 0.5 0.5; 0.2 0.4 0.8; 0.4 0.5 0.8; 0.8 0.2 0.2; 0.6 0.3 0.6; 0.5 0.4 0.7]);
axis equal tight;
set(gca, 'YDir', 'reverse');
xlabel('Column');
ylabel('Row');
title('Spatial Regions on Ising Grid');

% Add labels
text(mean(centre_cols), mean(centre_rows), 'Centre', 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'FontSize', 12, 'Color', 'w');
text(mean(P1_cols), mean(P1_rows), 'P1', 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'FontSize', 12, 'Color', 'w');
text(mean(P2_cols), mean(P2_rows), 'P2', 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'FontSize', 12, 'Color', 'w');

saveMyFig('SamplingConvergence_SpatialRegions', config.outputPath, gcf);
fprintf('Figure 2 saved\n');

%% =========================================================================
%% SECTION 8: Summary Report
%% =========================================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('  SAMPLING CONVERGENCE ANALYSIS\n');
fprintf('========================================\n');
fprintf('\n');
fprintf('CONFIGURATION:\n');
fprintf('  Analysis mode: %s\n', config.analysisMode);
fprintf('  Simulations analyzed: %d\n', nSims);
fprintf('  Ising grid: [%d x %d]\n', config.isingGrid(1), config.isingGrid(2));
fprintf('  Experimental grid: [%d x %d]\n', config.expGrid(1), config.expGrid(2));
fprintf('\n');
fprintf('----------------------------------------\n');
fprintf('GROUND TRUTH (Centre, full T):\n');
fprintf('----------------------------------------\n');
if ~isempty(GT_clean)
    fprintf('  Mean Moran''s I: %.4f +/- %.4f\n', mean(GT_clean), std(GT_clean));
else
    fprintf('  (Ground truth statistics not available from pre-computed results)\n');
end
fprintf('\n');
fprintf('----------------------------------------\n');
fprintf('SPATIAL HOMOGENEITY (P1 vs Centre):\n');
fprintf('----------------------------------------\n');
fprintf('  If P1 = Centre, there is NO spatial inhomogeneity\n');
fprintf('  Difference WD(P1) - WD(Centre):\n');
for f = 1:nFractions
    diff = Aggregate.WD_P1_mean(f) - Aggregate.WD_Centre_mean(f);
    pct = 100 * diff / Aggregate.WD_Centre_mean(f);
    fprintf('    1/%.0f: %+.4f (%+.1f%%)\n', config.inverseFractions(f), diff, pct);
end
meanDiff = mean(Aggregate.WD_P1_mean - Aggregate.WD_Centre_mean);
fprintf('  Mean difference: %+.4f\n', meanDiff);
if abs(meanDiff) < 0.005
    fprintf('  >> Result: NO significant spatial inhomogeneity detected\n');
else
    fprintf('  >> Result: POSSIBLE spatial inhomogeneity detected\n');
end
fprintf('\n');
fprintf('----------------------------------------\n');
fprintf('POOLING BENEFIT (P1+P2 vs P1):\n');
fprintf('----------------------------------------\n');
fprintf('  Negative = pooling IMPROVES estimate\n');
fprintf('  Difference WD(P1+P2) - WD(P1):\n');
for f = 1:nFractions
    diff = Aggregate.WD_P1P2_mean(f) - Aggregate.WD_P1_mean(f);
    pct = 100 * diff / Aggregate.WD_P1_mean(f);
    fprintf('    1/%.0f: %+.4f (%.1f%%)\n', config.inverseFractions(f), diff, pct);
end
meanPoolingDiff = mean(Aggregate.WD_P1P2_mean - Aggregate.WD_P1_mean);
fprintf('  Mean difference: %+.4f\n', meanPoolingDiff);
if meanPoolingDiff < 0
    improvement = -100 * meanPoolingDiff / mean(Aggregate.WD_P1_mean);
    fprintf('  >> Result: Pooling IMPROVES estimate by %.1f%% on average\n', improvement);
else
    fprintf('  >> Result: Pooling does NOT improve estimate\n');
end
fprintf('========================================\n');

%% =========================================================================
%% SECTION 9: Save Results
%% =========================================================================

fprintf('\n--- Section 9: Saving Results ---\n');

resultsFile = fullfile(config.outputPath, 'SamplingConvergence_Results.mat');
save(resultsFile, 'Results', '-v7.3');
fprintf('Results saved to: %s\n', resultsFile);

%% =========================================================================
%%                    PART 2: SQUARE GRID ANALYSIS
%% =========================================================================
%% Analyze how spatial homogeneity varies with sampling grid size
%% Using square grids: 10x10, 8x8, 6x6, 4x4, 2x2

fprintf('\n');
fprintf('###########################################################\n');
fprintf('##           PART 2: SQUARE GRID ANALYSIS                ##\n');
fprintf('###########################################################\n');

%% =========================================================================
%% SECTION 10-11: Square Grid Configuration and Analysis
%% =========================================================================
% Skip data processing if pre-computed results were loaded

if ~goto_figure_generation

fprintf('\n--- Section 10: Square Grid Configuration ---\n');

% Define square grid sizes to test (dynamically based on grid size)
minGridDim = min(config.isingGrid);
maxSquareSize = floor(minGridDim / 2);
squareGridSizes = [10, 8, 6, 4, 2];
squareGridSizes = squareGridSizes(squareGridSizes <= maxSquareSize);
if isempty(squareGridSizes)
    squareGridSizes = [floor(maxSquareSize/2), 2];
end
nSquareGrids = length(squareGridSizes);

fprintf('Square grid sizes to analyze: ');
fprintf('%dx%d ', [squareGridSizes; squareGridSizes]);
fprintf('\n');

% Initialize results for square grids
SquareGridResults = struct();
SquareGridResults.gridSizes = squareGridSizes;
SquareGridResults.nGrids = nSquareGrids;

% Store spatial homogeneity metric for each grid size
SquareGridResults.WD_diff_mean = zeros(nSquareGrids, 1);  % Mean |WD(TopLeft) - WD(Centre)|
SquareGridResults.WD_diff_std = zeros(nSquareGrids, 1);
SquareGridResults.WD_diff_all = cell(nSquareGrids, 1);    % All individual differences

%% =========================================================================
%% SECTION 11: Square Grid Analysis Loop
%% =========================================================================

fprintf('\n--- Section 11: Square Grid Analysis ---\n');

% We'll use the same simulations as Part 1
% For each grid size, compute Centre vs TopLeft across all temporal fractions

for g = 1:nSquareGrids
    gridSize = squareGridSizes(g);
    fprintf('\nProcessing %dx%d grid...\n', gridSize, gridSize);

    % Define regions for this grid size
    % Centre crop
    sqCentre_rowStart = floor((config.isingGrid(1) - gridSize) / 2) + 1;
    sqCentre_colStart = floor((config.isingGrid(2) - gridSize) / 2) + 1;
    sqCentre_rows = sqCentre_rowStart:(sqCentre_rowStart + gridSize - 1);
    sqCentre_cols = sqCentre_colStart:(sqCentre_colStart + gridSize - 1);

    % TopLeft: Top-left corner
    sqTopLeft_rows = 1:gridSize;
    sqTopLeft_cols = 1:gridSize;

    fprintf('  Centre: rows %d:%d, cols %d:%d\n', sqCentre_rows(1), sqCentre_rows(end), ...
        sqCentre_cols(1), sqCentre_cols(end));
    fprintf('  TopLeft: rows %d:%d, cols %d:%d\n', sqTopLeft_rows(1), sqTopLeft_rows(end), ...
        sqTopLeft_cols(1), sqTopLeft_cols(end));

    % Create weight matrix for this grid size
    valueMap_sq = rand(gridSize, gridSize);
    distanceMat_sq = squareform(mL_distanceMat(valueMap_sq));
    uniqueDistances_sq = unique(distanceMat_sq);
    uniqueDistances_sq(uniqueDistances_sq == 0) = [];
    currDistInds_sq = ismember(distanceMat_sq, uniqueDistances_sq(1));
    weightMat_sq = zeros(size(distanceMat_sq));
    weightMat_sq(currDistInds_sq) = distanceMat_sq(currDistInds_sq);
    weightMat_sq(weightMat_sq == inf) = 0;

    % Collect all WD differences across simulations and fractions
    all_WD_diffs = [];

    for s = 1:nSims
        simID = simIDs(s);
        simPath = fullfile(config.isingDataPath, sprintf('sim_%d.mat', simID));

        if ~exist(simPath, 'file')
            continue;
        end

        % Load simulation
        simData = load(simPath);
        stored_spins = simData.stored_spins;
        T = size(stored_spins, 1);

        % Compute Moran's I for full duration (Ground Truth for this grid)
        GT_sq = zeros(1, T);
        moransI_Centre_sq = zeros(1, T);
        moransI_TopLeft_sq = zeros(1, T);

        for t = 1:T
            frame = squeeze(stored_spins(t, :, :));

            % Centre
            frame_centre = frame(sqCentre_rows, sqCentre_cols);
            if all(frame_centre(:) == 0) || all(frame_centre(:) == 1)
                moransI_Centre_sq(t) = NaN;
            else
                moransI_Centre_sq(t) = mL_moransI(double(frame_centre), weightMat_sq);
            end

            % TopLeft
            frame_TopLeft = frame(sqTopLeft_rows, sqTopLeft_cols);
            if all(frame_TopLeft(:) == 0) || all(frame_TopLeft(:) == 1)
                moransI_TopLeft_sq(t) = NaN;
            else
                moransI_TopLeft_sq(t) = mL_moransI(double(frame_TopLeft), weightMat_sq);
            end
        end

        GT_sq_clean = moransI_Centre_sq(~isnan(moransI_Centre_sq));

        % Temporal subsampling analysis for this simulation
        for f = 1:nFractions
            invFrac = config.inverseFractions(f);
            segmentLength = floor(T / invFrac);
            nSegments = floor(T / segmentLength);

            for seg = 1:nSegments
                startIdx = (seg - 1) * segmentLength + 1;
                endIdx = seg * segmentLength;

                seg_Centre = moransI_Centre_sq(startIdx:endIdx);
                seg_TopLeft = moransI_TopLeft_sq(startIdx:endIdx);

                seg_Centre_clean = seg_Centre(~isnan(seg_Centre));
                seg_TopLeft_clean = seg_TopLeft(~isnan(seg_TopLeft));

                WD_Centre = wasserstein_1d(GT_sq_clean, seg_Centre_clean);
                WD_TopLeft = wasserstein_1d(GT_sq_clean, seg_TopLeft_clean);

                % Store the difference
                all_WD_diffs(end+1) = WD_TopLeft - WD_Centre;
            end
        end

        clear stored_spins simData;
    end

    % Store results for this grid size (use 'omitnan' to handle sparse NaN values)
    SquareGridResults.WD_diff_all{g} = all_WD_diffs;
    SquareGridResults.WD_diff_mean(g) = mean(all_WD_diffs, 'omitnan');
    SquareGridResults.WD_diff_std(g) = std(all_WD_diffs, 'omitnan');

    fprintf('  Mean WD(TopLeft)-WD(Centre): %+.4f +/- %.4f\n', ...
        SquareGridResults.WD_diff_mean(g), SquareGridResults.WD_diff_std(g));
end

end  % End of: if ~goto_figure_generation (square grid data processing)

%% =========================================================================
%% SECTION 12: Square Grid Visualization
%% =========================================================================

% Extract variables from SquareGridResults for visualization
squareGridSizes = SquareGridResults.gridSizes;
nSquareGrids = length(squareGridSizes);

% Convert WD_diff_all from 2D array to cell array (Python saves as 2D array)
if goto_figure_generation && ~iscell(SquareGridResults.WD_diff_all)
    WD_diff_all_2d = SquareGridResults.WD_diff_all;
    nGrids = size(WD_diff_all_2d, 1);
    WD_diff_all_temp = cell(1, nGrids);
    for g = 1:nGrids
        WD_diff_all_temp{g} = WD_diff_all_2d(g, :)';
    end
    SquareGridResults.WD_diff_all = WD_diff_all_temp;
end

fprintf('\n--- Section 12: Square Grid Visualization ---\n');

% =========================================================================
% FIGURE 3: Spatial Homogeneity vs Grid Size
% =========================================================================
figure('Name', 'Spatial Homogeneity vs Grid Size');

% X-axis: grid area (side^2) - sort in increasing order for xticks
% Cast to double to avoid integer/double arithmetic issues from Python data
[gridAreas, sortIdx] = sort(double(squareGridSizes).^2);
gridLabels = arrayfun(@(x) sprintf('%dx%d', x, x), squareGridSizes(sortIdx), 'UniformOutput', false);
% Reorder results to match sorted order
WD_diff_mean_sorted = SquareGridResults.WD_diff_mean(sortIdx);
WD_diff_std_sorted = SquareGridResults.WD_diff_std(sortIdx);
WD_diff_all_sorted = SquareGridResults.WD_diff_all(sortIdx);

% Main panel: Mean difference with error bars
subplot(1, 2, 1);
hold on;

% Plot individual points with jitter
for g = 1:nSquareGrids
    y_vals = WD_diff_all_sorted{g};
    x_vals = gridAreas(g) + randn(size(y_vals)) * gridAreas(g) * 0.05;
    scatter(x_vals, y_vals, 10, [0.5, 0.5, 0.5], 'filled', 'MarkerFaceAlpha', 0.2);
end

% Plot mean with error bars
errorbar(gridAreas, WD_diff_mean_sorted, WD_diff_std_sorted, ...
    'o-', 'Color', [0.2, 0.4, 0.8], 'LineWidth', 2, 'MarkerSize', 10, ...
    'MarkerFaceColor', [0.2, 0.4, 0.8]);

yline(0, 'k--', 'LineWidth', 1.5);
hold off;

xlabel('Grid Area (cells)');
ylabel('WD(TopLeft) - WD(Centre)');
title('Spatial Homogeneity vs Grid Size');
set(gca, 'XScale', 'log');
xticks(gridAreas);
xticklabels(gridLabels);
grid on;

% Add text annotation
if mean(abs(SquareGridResults.WD_diff_mean)) < 0.005
    text(0.5, 0.95, 'No significant spatial inhomogeneity', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center', ...
        'FontWeight', 'bold', 'Color', [0.2, 0.7, 0.3]);
end

% Second panel: Summary statistics
subplot(1, 2, 2);
axis off;

summaryText = {
    '=== SQUARE GRID ANALYSIS ===', ...
    '', ...
    'Grid Size | Mean Diff | Std', ...
    '----------|-----------|------', ...
};
for g = 1:nSquareGrids
    summaryText{end+1} = sprintf('%4dx%-4d | %+.4f  | %.4f', ...
        squareGridSizes(g), squareGridSizes(g), ...
        SquareGridResults.WD_diff_mean(g), SquareGridResults.WD_diff_std(g));
end
summaryText{end+1} = '';
summaryText{end+1} = sprintf('Overall mean: %+.4f', mean(SquareGridResults.WD_diff_mean));

text(0.05, 0.95, summaryText, 'VerticalAlignment', 'top', ...
    'FontSize', 9, 'FontName', 'FixedWidth', 'Units', 'normalized');
title('Summary');

sgtitle('Part 2: Spatial Homogeneity Across Grid Sizes', 'FontWeight', 'bold');

saveMyFig('SamplingConvergence_SquareGrids', config.outputPath, gcf);
fprintf('Figure 3 saved\n');

%% =========================================================================
%% SECTION 13: Square Grid Summary
%% =========================================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('  SQUARE GRID ANALYSIS SUMMARY\n');
fprintf('========================================\n');
fprintf('\n');
fprintf('Grid Size | Mean WD(TopLeft)-WD(Centre) | Std\n');
fprintf('----------|------------------------|--------\n');
for g = 1:nSquareGrids
    fprintf('%4dx%-4d  |       %+.4f          | %.4f\n', ...
        squareGridSizes(g), squareGridSizes(g), ...
        SquareGridResults.WD_diff_mean(g), SquareGridResults.WD_diff_std(g));
end
fprintf('\n');

overall_mean = mean(SquareGridResults.WD_diff_mean);
fprintf('Overall mean difference: %+.4f\n', overall_mean);
if abs(overall_mean) < 0.005
    fprintf('>> Conclusion: NO systematic spatial inhomogeneity detected\n');
else
    fprintf('>> Conclusion: POSSIBLE spatial inhomogeneity - investigate further\n');
end
fprintf('========================================\n');

%% =========================================================================
%% SECTION 14: Distance-from-Centre Analysis (2x2 grid)
%% =========================================================================
%% For the smallest grid (2x2), analyze ALL positions and plot WD vs distance
%% from the centre of the simulation grid.

% Skip data processing if pre-computed results were loaded
if ~goto_figure_generation

fprintf('\n--- Section 14: Distance-from-Centre Analysis (2x2 grid) ---\n');

% Use 2x2 grid - gives different positions for non-square grids
distGridSize = 2;

% Generate all non-overlapping positions for 2x2 grid (handle non-square grids)
nPosPerDimRow = floor(config.isingGrid(1) / distGridSize);
nPosPerDimCol = floor(config.isingGrid(2) / distGridSize);
nTotalPos = nPosPerDimRow * nPosPerDimCol;

fprintf('Analyzing %dx%d grid: %d row positions x %d col positions = %d total positions\n', ...
    distGridSize, distGridSize, nPosPerDimRow, nPosPerDimCol, nTotalPos);

% Create weight matrix for 2x2 grid
valueMap_dist = rand(distGridSize, distGridSize);
distanceMat_dist = squareform(mL_distanceMat(valueMap_dist));
uniqueDistances_dist = unique(distanceMat_dist);
uniqueDistances_dist(uniqueDistances_dist == 0) = [];
currDistInds_dist = ismember(distanceMat_dist, uniqueDistances_dist(1));
weightMat_dist = zeros(size(distanceMat_dist));
weightMat_dist(currDistInds_dist) = distanceMat_dist(currDistInds_dist);
weightMat_dist(weightMat_dist == inf) = 0;

% Centre of the grid
gridCentre = (config.isingGrid + 1) / 2;  % [16.5, 16.5]

% Compute Ground Truth: central 2x2 (positions 15:16, 15:16)
distCentre_rowStart = floor((config.isingGrid(1) - distGridSize) / 2) + 1;
distCentre_colStart = floor((config.isingGrid(2) - distGridSize) / 2) + 1;
distCentre_rows = distCentre_rowStart:(distCentre_rowStart + distGridSize - 1);
distCentre_cols = distCentre_colStart:(distCentre_colStart + distGridSize - 1);

fprintf('Centre position: rows %d:%d, cols %d:%d\n', ...
    distCentre_rows(1), distCentre_rows(end), distCentre_cols(1), distCentre_cols(end));

% Initialize storage
DistanceAnalysis = struct();
DistanceAnalysis.gridSize = distGridSize;
DistanceAnalysis.nPositions = nTotalPos;
DistanceAnalysis.positionCentres = zeros(nTotalPos, 2);  % [row, col] of each position centre
DistanceAnalysis.distanceFromCentre = zeros(nTotalPos, 1);
DistanceAnalysis.WD_to_GT = zeros(nTotalPos, 1);
DistanceAnalysis.meanMoransI = zeros(nTotalPos, 1);

% Generate position coordinates and compute distances
posIdx = 0;
for pr = 1:nPosPerDimRow
    for pc = 1:nPosPerDimCol
        posIdx = posIdx + 1;

        % Row and column indices for this position
        rowStart = (pr - 1) * distGridSize + 1;
        colStart = (pc - 1) * distGridSize + 1;

        % Centre of this position
        posCentre_row = rowStart + (distGridSize - 1) / 2;
        posCentre_col = colStart + (distGridSize - 1) / 2;

        DistanceAnalysis.positionCentres(posIdx, :) = [posCentre_row, posCentre_col];

        % Distance from grid centre
        DistanceAnalysis.distanceFromCentre(posIdx) = sqrt(...
            (posCentre_row - gridCentre(1))^2 + (posCentre_col - gridCentre(2))^2);
    end
end

% Process simulations to compute WD for each position
fprintf('Computing Moran''s I for all %d positions across %d simulations...\n', nTotalPos, nSims);

% Aggregate WD values across simulations for each position
WD_all_positions = zeros(nTotalPos, nSims);

for s = 1:nSims
    simID = simIDs(s);
    simPath = fullfile(config.isingDataPath, sprintf('sim_%d.mat', simID));

    if ~exist(simPath, 'file')
        WD_all_positions(:, s) = NaN;
        continue;
    end

    % Load simulation
    simData = load(simPath);
    stored_spins = simData.stored_spins;
    T = size(stored_spins, 1);

    % Compute Ground Truth (central 2x2, full duration)
    GT_dist = zeros(1, T);
    for t = 1:T
        frame = squeeze(stored_spins(t, :, :));
        frame_centre = frame(distCentre_rows, distCentre_cols);
        if all(frame_centre(:) == 0) || all(frame_centre(:) == 1)
            GT_dist(t) = NaN;
        else
            GT_dist(t) = mL_moransI(double(frame_centre), weightMat_dist);
        end
    end
    GT_dist_clean = GT_dist(~isnan(GT_dist));

    % Compute Moran's I for each position
    posIdx = 0;
    for pr = 1:nPosPerDimRow
        for pc = 1:nPosPerDimCol
            posIdx = posIdx + 1;

            rowStart = (pr - 1) * distGridSize + 1;
            colStart = (pc - 1) * distGridSize + 1;
            rows = rowStart:(rowStart + distGridSize - 1);
            cols = colStart:(colStart + distGridSize - 1);

            % Compute Moran's I for this position
            moransI_pos = zeros(1, T);
            for t = 1:T
                frame = squeeze(stored_spins(t, :, :));
                frame_pos = frame(rows, cols);
                if all(frame_pos(:) == 0) || all(frame_pos(:) == 1)
                    moransI_pos(t) = NaN;
                else
                    moransI_pos(t) = mL_moransI(double(frame_pos), weightMat_dist);
                end
            end
            moransI_pos_clean = moransI_pos(~isnan(moransI_pos));

            % Compute WD to Ground Truth
            WD_all_positions(posIdx, s) = wasserstein_1d(GT_dist_clean, moransI_pos_clean);
        end
    end

    clear stored_spins simData;

    if mod(s, max(1, floor(nSims/5))) == 0
        fprintf('  Processed %d/%d simulations\n', s, nSims);
    end
end

% Average WD across simulations
DistanceAnalysis.WD_to_GT = mean(WD_all_positions, 2, 'omitnan');
DistanceAnalysis.WD_to_GT_std = std(WD_all_positions, 0, 2, 'omitnan');

end  % End of: if ~goto_figure_generation (distance analysis data processing)

%% =========================================================================
%% SECTION 15: Distance-from-Centre Visualization
%% =========================================================================

% Extract variables from DistanceAnalysis for visualization
distGridSize = double(DistanceAnalysis.gridSize);
nTotalPos = double(DistanceAnalysis.nPositions);
% Use Python's saved grid dimensions (cast to double for arithmetic)
nPosPerDimRow = double(DistanceAnalysis.nPosRow);
nPosPerDimCol = double(DistanceAnalysis.nPosCol);

fprintf('\n--- Section 15: Distance-from-Centre Visualization ---\n');

% =========================================================================
% FIGURE 4: WD vs Distance from Centre
% =========================================================================
figure('Name', 'Distance from Centre Analysis');

% Panel 1: Scatter plot WD vs Distance
subplot(1, 2, 1);
scatter(DistanceAnalysis.distanceFromCentre, DistanceAnalysis.WD_to_GT, ...
    30, [0.2, 0.4, 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
hold on;

% Fit a line to see trend
validIdx = ~isnan(DistanceAnalysis.WD_to_GT);
if sum(validIdx) > 2
    p = polyfit(DistanceAnalysis.distanceFromCentre(validIdx), ...
        DistanceAnalysis.WD_to_GT(validIdx), 1);
    xFit = linspace(min(DistanceAnalysis.distanceFromCentre), ...
        max(DistanceAnalysis.distanceFromCentre), 100);
    yFit = polyval(p, xFit);
    plot(xFit, yFit, 'r-', 'LineWidth', 2);

    % Compute correlation
    [r, pval] = corr(DistanceAnalysis.distanceFromCentre(validIdx), ...
        DistanceAnalysis.WD_to_GT(validIdx));

    text(0.05, 0.95, sprintf('r = %.3f, p = %.3f\nslope = %.4f', r, pval, p(1)), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontSize', 10, 'BackgroundColor', 'w');
end
hold off;

xlabel('Distance from Grid Centre');
ylabel('WD(position, GT_{centre})');
title(sprintf('%dx%d Grid: WD vs Distance from Centre', distGridSize, distGridSize));
grid on;

% Panel 2: Heatmap of WD values
subplot(1, 2, 2);

% Reshape WD values into grid (handle non-square grids)
WD_heatmap = reshape(DistanceAnalysis.WD_to_GT, [nPosPerDimRow, nPosPerDimCol]);
imagesc(WD_heatmap);
colorbar;
colormap(hot);
axis equal tight;

% Mark centre
hold on;
centrePosRow = ceil(nPosPerDimRow / 2);
centrePosCol = ceil(nPosPerDimCol / 2);
plot(centrePosCol, centrePosRow, 'go', 'MarkerSize', 15, 'LineWidth', 3);
hold off;

xlabel('Column Position');
ylabel('Row Position');
title('Spatial Map of WD to Centre');
set(gca, 'YDir', 'reverse');

sgtitle(sprintf('Distance-from-Centre Analysis (%dx%d grid, %d positions)', ...
    distGridSize, distGridSize, nTotalPos), 'FontWeight', 'bold');

saveMyFig('SamplingConvergence_DistanceFromCentre', config.outputPath, gcf);
fprintf('Figure 4 saved\n');

% Store results
Results.DistanceAnalysis = DistanceAnalysis;

%% =========================================================================
%% SECTION 16: Distance Analysis Summary
%% =========================================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('  DISTANCE-FROM-CENTRE ANALYSIS\n');
fprintf('========================================\n');
fprintf('Grid size: %dx%d\n', distGridSize, distGridSize);
fprintf('Total positions: %d\n', nTotalPos);
fprintf('\n');

if exist('r', 'var') && exist('pval', 'var')
    fprintf('Correlation (WD vs Distance):\n');
    fprintf('  r = %.4f\n', r);
    fprintf('  p = %.4f\n', pval);
    fprintf('  slope = %.4f\n', p(1));
    fprintf('\n');

    if pval < 0.05 && p(1) > 0
        fprintf('>> Result: SIGNIFICANT positive correlation - WD increases with distance\n');
        fprintf('           This suggests spatial inhomogeneity (edge effects)\n');
    elseif pval < 0.05 && p(1) < 0
        fprintf('>> Result: SIGNIFICANT negative correlation - WD decreases with distance\n');
        fprintf('           This is unexpected - investigate further\n');
    else
        fprintf('>> Result: NO significant correlation - spatially homogeneous\n');
    end
end
fprintf('========================================\n');

%% =========================================================================
%% SECTION 17: Save Final Results
%% =========================================================================

fprintf('\n--- Section 17: Saving Final Results ---\n');

Results.SquareGridResults = SquareGridResults;
Results.timestamp = datetime('now');

resultsFile = fullfile(config.outputPath, 'SamplingConvergence_Results.mat');
save(resultsFile, 'Results', '-v7.3');
fprintf('Final results saved to: %s\n', resultsFile);

%% =========================================================================
%% HELPER FUNCTIONS
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

function saveMyFig(name, outputPath, figHandle)
    % Save figure in multiple formats
    if nargin < 3
        figHandle = gcf;
    end

    dateStr = datestr(now, 'yyyymmdd');
    baseName = sprintf('%s_%s_%s', dateStr, name, get(figHandle, 'Name'));
    baseName = strrep(baseName, ' ', '_');

    % PNG
    pngPath = fullfile(outputPath, [baseName '.png']);
    saveas(figHandle, pngPath);
    fprintf('Figure (Handle: %d) saved as: %s\n', figHandle.Number, pngPath);

    % SVG
    svgPath = fullfile(outputPath, [baseName '.svg']);
    saveas(figHandle, svgPath);
    fprintf('Figure (Handle: %d) saved as: %s\n', figHandle.Number, svgPath);
end
