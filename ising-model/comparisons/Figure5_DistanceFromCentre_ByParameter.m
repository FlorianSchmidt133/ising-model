%% =========================================================================
%% Figure 5: Distance-from-Centre Analysis by Ising Parameter
%% =========================================================================
%
% PURPOSE:
% Analyze spatial homogeneity of Ising simulations by examining how WD
% (Wasserstein distance) varies with distance from grid centre.
% Split results by each of the 5 Ising model parameters to see if
% spatial homogeneity depends on parameter values.
%
% OUTPUT:
% - Figure 1: Heatmaps split by beta (inverse temperature)
% - Figure 2: Heatmaps split by c
% - Figure 3: Heatmaps split by decay_const
% - Figure 4: Heatmaps split by inhibition_range
% - Figure 5: Heatmaps split by bias
%
% Each figure has 4 panels (one per parameter value), showing the
% spatial map of WD(position, GT_centre) averaged across simulations
% with that parameter value.
%
%% =========================================================================


%% =========================================================================
%% SECTION 1: Configuration
%% =========================================================================

fprintf('=== Distance-from-Centre Analysis by Parameter ===\n');
fprintf('--- Section 1: Configuration ---\n');

config = struct();

% -------------------------------------------------------------------------
% Paths
% -------------------------------------------------------------------------
config.isingDataPath = mba_p('IsingModelData_for_Florian');
config.outputPath = 'Fig. 5 Model\IsingModels\IsingComparison\SpatialHomogeneity';

% -------------------------------------------------------------------------
% Grid Configuration
% -------------------------------------------------------------------------
config.isingGrid = [32, 32];       % Ising simulation grid
config.gridSize = 2;               % Size of sampling grid (2x2 default)
                                   % Change to 4 for 4x4, etc.

% -------------------------------------------------------------------------
% Ising Parameter Values (from grid search)
% -------------------------------------------------------------------------
config.params.beta = [0.5, 0.6, 0.7, 0.8];
config.params.c = [2, 4, 6, 8];
config.params.decay_const = [2, 4, 6, 8];
config.params.inhibition_range = [1, 4, 9, 13];
config.params.bias = [-1, -0.8, -0.6, -0.4];

% Parameter names for iteration
config.paramNames = {'beta', 'c', 'decay_const', 'inhibition_range', 'bias'};
config.paramLabels = {'Beta (Inverse Temp)', 'c', 'Decay Constant', 'Inhibition Range', 'Bias'};

fprintf('Ising grid: [%d x %d]\n', config.isingGrid(1), config.isingGrid(2));
fprintf('Sampling grid: %dx%d\n', config.gridSize, config.gridSize);

% Create output directory
if ~exist(config.outputPath, 'dir')
    mkdir(config.outputPath);
    fprintf('Created output directory: %s\n', config.outputPath);
end

%% =========================================================================
%% SECTION 2: Load Simulation Parameters
%% =========================================================================

fprintf('\n--- Section 2: Load Simulation Parameters ---\n');

% Find all simulation files
simFiles = dir(fullfile(config.isingDataPath, 'sim_*.mat'));
nSims = length(simFiles);
fprintf('Found %d simulation files\n', nSims);

% Initialize parameter storage
simIDs = zeros(nSims, 1);
simParams = struct();
simParams.beta = zeros(nSims, 1);
simParams.c = zeros(nSims, 1);
simParams.decay_const = zeros(nSims, 1);
simParams.inhibition_range = zeros(nSims, 1);
simParams.bias = zeros(nSims, 1);

% Load parameters from each simulation
fprintf('Loading simulation parameters...\n');
validSims = true(nSims, 1);

for i = 1:nSims
    % Extract simID from filename
    tokens = regexp(simFiles(i).name, 'sim_(\d+)\.mat', 'tokens');
    if ~isempty(tokens)
        simIDs(i) = str2double(tokens{1}{1});
    else
        validSims(i) = false;
        continue;
    end

    % Load simulation to get parameters
    simPath = fullfile(config.isingDataPath, simFiles(i).name);
    try
        simData = load(simPath, 'params');
        simParams.beta(i) = simData.params.beta;
        simParams.c(i) = simData.params.c;
        simParams.decay_const(i) = simData.params.decay_const;
        simParams.inhibition_range(i) = simData.params.inhibition_range;
        simParams.bias(i) = simData.params.bias;
    catch
        validSims(i) = false;
    end

    if mod(i, 200) == 0
        fprintf('  Loaded parameters for %d/%d simulations\n', i, nSims);
    end
end

% Filter to valid simulations
simIDs = simIDs(validSims);
simParams.beta = simParams.beta(validSims);
simParams.c = simParams.c(validSims);
simParams.decay_const = simParams.decay_const(validSims);
simParams.inhibition_range = simParams.inhibition_range(validSims);
simParams.bias = simParams.bias(validSims);
nSims = length(simIDs);

fprintf('Loaded parameters for %d valid simulations\n', nSims);

%% =========================================================================
%% SECTION 3: Define Spatial Grid
%% =========================================================================

fprintf('\n--- Section 3: Define Spatial Grid ---\n');

gridSize = config.gridSize;
nPosPerDim = floor(config.isingGrid(1) / gridSize);
nTotalPos = nPosPerDim^2;

fprintf('Grid size: %dx%d\n', gridSize, gridSize);
fprintf('Positions per dimension: %d\n', nPosPerDim);
fprintf('Total positions: %d\n', nTotalPos);

% Centre of the grid
gridCentre = (config.isingGrid + 1) / 2;  % [16.5, 16.5]

% Centre position indices for Ground Truth
distCentre_rowStart = floor((config.isingGrid(1) - gridSize) / 2) + 1;
distCentre_colStart = floor((config.isingGrid(2) - gridSize) / 2) + 1;
distCentre_rows = distCentre_rowStart:(distCentre_rowStart + gridSize - 1);
distCentre_cols = distCentre_colStart:(distCentre_colStart + gridSize - 1);

fprintf('Centre position: rows %d:%d, cols %d:%d\n', ...
    distCentre_rows(1), distCentre_rows(end), distCentre_cols(1), distCentre_cols(end));

% Compute position centres and distances
positionCentres = zeros(nTotalPos, 2);
distanceFromCentre = zeros(nTotalPos, 1);

posIdx = 0;
for pr = 1:nPosPerDim
    for pc = 1:nPosPerDim
        posIdx = posIdx + 1;

        rowStart = (pr - 1) * gridSize + 1;
        colStart = (pc - 1) * gridSize + 1;

        posCentre_row = rowStart + (gridSize - 1) / 2;
        posCentre_col = colStart + (gridSize - 1) / 2;

        positionCentres(posIdx, :) = [posCentre_row, posCentre_col];
        distanceFromCentre(posIdx) = sqrt(...
            (posCentre_row - gridCentre(1))^2 + (posCentre_col - gridCentre(2))^2);
    end
end

%% =========================================================================
%% SECTION 4: Create Weight Matrix
%% =========================================================================

fprintf('\n--- Section 4: Create Weight Matrix ---\n');

valueMap = rand(gridSize, gridSize);
distanceMat = squareform(mL_distanceMat(valueMap));
uniqueDistances = unique(distanceMat);
uniqueDistances(uniqueDistances == 0) = [];
currDistInds = ismember(distanceMat, uniqueDistances(1));
weightMat = zeros(size(distanceMat));
weightMat(currDistInds) = distanceMat(currDistInds);
weightMat(weightMat == inf) = 0;

fprintf('Weight matrix size: [%d x %d]\n', size(weightMat, 1), size(weightMat, 2));

%% =========================================================================
%% SECTION 5: Main Analysis Loop
%% =========================================================================

fprintf('\n--- Section 5: Main Analysis Loop ---\n');
fprintf('Computing WD for all %d positions across %d simulations...\n', nTotalPos, nSims);

% Store WD for each position and simulation
WD_all = zeros(nTotalPos, nSims);

for s = 1:nSims
    simID = simIDs(s);
    simPath = fullfile(config.isingDataPath, sprintf('sim_%d.mat', simID));

    if ~exist(simPath, 'file')
        WD_all(:, s) = NaN;
        continue;
    end

    % Load simulation
    simData = load(simPath, 'stored_spins');
    stored_spins = simData.stored_spins;
    T = size(stored_spins, 1);

    % Compute Ground Truth (centre position, full duration)
    GT = zeros(1, T);
    for t = 1:T
        frame = squeeze(stored_spins(t, :, :));
        frame_centre = frame(distCentre_rows, distCentre_cols);
        if all(frame_centre(:) == 0) || all(frame_centre(:) == 1)
            GT(t) = NaN;
        else
            GT(t) = mL_moransI(double(frame_centre), weightMat);
        end
    end
    GT_clean = GT(~isnan(GT));

    % Compute Moran's I for each position
    posIdx = 0;
    for pr = 1:nPosPerDim
        for pc = 1:nPosPerDim
            posIdx = posIdx + 1;

            rowStart = (pr - 1) * gridSize + 1;
            colStart = (pc - 1) * gridSize + 1;
            rows = rowStart:(rowStart + gridSize - 1);
            cols = colStart:(colStart + gridSize - 1);

            moransI_pos = zeros(1, T);
            for t = 1:T
                frame = squeeze(stored_spins(t, :, :));
                frame_pos = frame(rows, cols);
                if all(frame_pos(:) == 0) || all(frame_pos(:) == 1)
                    moransI_pos(t) = NaN;
                else
                    moransI_pos(t) = mL_moransI(double(frame_pos), weightMat);
                end
            end
            moransI_pos_clean = moransI_pos(~isnan(moransI_pos));

            WD_all(posIdx, s) = wasserstein_1d(GT_clean, moransI_pos_clean);
        end
    end

    clear stored_spins simData;

    if mod(s, max(1, floor(nSims/10))) == 0
        fprintf('  Processed %d/%d simulations\n', s, nSims);
    end
end

fprintf('Analysis complete.\n');

%% =========================================================================
%% SECTION 6-10: Parameter-Specific Figures
%% =========================================================================

fprintf('\n--- Section 6-10: Creating Parameter-Specific Figures ---\n');

Results = struct();
Results.config = config;
Results.simIDs = simIDs;
Results.simParams = simParams;
Results.positionCentres = positionCentres;
Results.distanceFromCentre = distanceFromCentre;
Results.WD_all = WD_all;

for p = 1:length(config.paramNames)
    paramName = config.paramNames{p};
    paramLabel = config.paramLabels{p};
    paramValues = config.params.(paramName);
    nValues = length(paramValues);

    fprintf('\nCreating figure for %s...\n', paramName);

    % Create figure
    figure('Name', sprintf('Distance from Centre - %s', paramLabel));

    % Compute heatmaps for each parameter value
    heatmaps = cell(nValues, 1);
    correlations = zeros(nValues, 2);  % [r, p]

    for v = 1:nValues
        val = paramValues(v);

        % Find simulations with this parameter value
        simMask = (simParams.(paramName) == val);
        nSimsWithVal = sum(simMask);

        % Average WD across these simulations
        WD_subset = WD_all(:, simMask);
        WD_mean = mean(WD_subset, 2, 'omitnan');

        % Reshape to heatmap
        heatmaps{v} = reshape(WD_mean, [nPosPerDim, nPosPerDim]);

        % Compute correlation with distance
        validIdx = ~isnan(WD_mean);
        if sum(validIdx) > 2
            [r, pval] = corr(distanceFromCentre(validIdx), WD_mean(validIdx));
            correlations(v, :) = [r, pval];
        else
            correlations(v, :) = [NaN, NaN];
        end

        fprintf('  %s = %.2f: %d simulations, r = %.3f\n', ...
            paramName, val, nSimsWithVal, correlations(v, 1));
    end

    % Find common colorbar limits
    allVals = [];
    for v = 1:nValues
        allVals = [allVals; heatmaps{v}(:)];
    end
    cLim = [min(allVals, [], 'omitnan'), max(allVals, [], 'omitnan')];

    % Plot heatmaps
    for v = 1:nValues
        subplot(1, nValues, v);

        imagesc(heatmaps{v});
        caxis(cLim);
        colorbar;
        colormap(hot);
        axis equal tight;
        set(gca, 'YDir', 'reverse');

        % Mark centre
        hold on;
        centrePosRow = ceil(nPosPerDim / 2);
        centrePosCol = ceil(nPosPerDim / 2);
        plot(centrePosCol, centrePosRow, 'go', 'MarkerSize', 12, 'LineWidth', 2);
        hold off;

        % Title with parameter value and correlation
        if ~isnan(correlations(v, 1))
            if correlations(v, 2) < 0.05
                sigStr = '*';
            else
                sigStr = '';
            end
            titleStr = sprintf('%s = %.2f\nr = %.3f%s', ...
                paramName, paramValues(v), correlations(v, 1), sigStr);
        else
            titleStr = sprintf('%s = %.2f', paramName, paramValues(v));
        end
        title(titleStr, 'FontSize', 10);

        xlabel('Col');
        ylabel('Row');
    end

    sgtitle(sprintf('Spatial Homogeneity by %s (%dx%d grid)', ...
        paramLabel, gridSize, gridSize), 'FontWeight', 'bold');

    % Save figure
    saveMyFig(sprintf('DistanceFromCentre_%s', paramName), config.outputPath, gcf);
    fprintf('  Figure saved\n');

    % Store results
    Results.(paramName).heatmaps = heatmaps;
    Results.(paramName).correlations = correlations;
    Results.(paramName).paramValues = paramValues;
end

%% =========================================================================
%% SECTION 11: Summary Statistics
%% =========================================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('  SUMMARY: SPATIAL HOMOGENEITY BY PARAMETER\n');
fprintf('========================================\n');
fprintf('\n');

for p = 1:length(config.paramNames)
    paramName = config.paramNames{p};
    paramLabel = config.paramLabels{p};
    paramValues = config.params.(paramName);
    corrs = Results.(paramName).correlations;

    fprintf('%n', paramLabel);
    fprintf('  Value   |    r    |    p    | Significant\n');
    fprintf('  --------|---------|---------|------------\n');
    for v = 1:length(paramValues)
        if corrs(v, 2) < 0.05
            sigStr = 'YES';
        else
            sigStr = 'no';
        end
        fprintf('  %7.2f | %+.4f | %.4f  | %s\n', ...
            paramValues(v), corrs(v, 1), corrs(v, 2), sigStr);
    end
    fprintf('\n');
end

fprintf('========================================\n');

%% =========================================================================
%% SECTION 12: Save Results
%% =========================================================================

fprintf('\n--- Section 12: Save Results ---\n');

Results.timestamp = datetime('now');
resultsFile = fullfile(config.outputPath, 'DistanceFromCentre_ByParameter_Results.mat');
save(resultsFile, 'Results', '-v7.3');
fprintf('Results saved to: %s\n', resultsFile);

fprintf('\nDone!\n');

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
    baseName = sprintf('%s_%s', dateStr, name);

    % PNG
    pngPath = fullfile(outputPath, [baseName '.png']);
    saveas(figHandle, pngPath);
    fprintf('  Saved: %s\n', pngPath);

    % SVG
    svgPath = fullfile(outputPath, [baseName '.svg']);
    saveas(figHandle, svgPath);
end
