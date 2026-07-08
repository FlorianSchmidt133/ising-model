%% =========================================================================
%% Figure 5: Spatial Homogeneity Visualization for Ising Simulations
%% =========================================================================
%
% PURPOSE:
% Visualize spatial homogeneity in Ising model simulations by analyzing
% how Wasserstein Distance (WD) varies with distance from the grid centre.
%
% INPUT:
% - updated_ising.mat: Contains stored_spins (2000 x 39 x 78)
%
% OUTPUT:
% - Figure with scatter plot (WD vs distance) and heatmap (spatial WD map)
%
% METHODOLOGY:
% - Use 2x2 analysis windows across the 39x78 grid
% - Compute Moran's I for each position across all timeframes
% - Define Ground Truth as the central position's Moran's I distribution
% - Compute WD between each position and Ground Truth
% - Visualize spatial distribution of WD values

%% =========================================================================
%% SECTION 1: Configuration
%% =========================================================================

fprintf('=== Figure 5: Spatial Homogeneity Visualization ===\n');
fprintf('--- Section 1: Configuration ---\n');

config = struct();

% -------------------------------------------------------------------------
% Paths
% -------------------------------------------------------------------------
config.dataPath = 'Fig. 5 Model\IsingModels\Data\updated_ising.mat';
config.outputPath = 'Fig. 5 Model\IsingModels\SpatialHomogeneity';

% -------------------------------------------------------------------------
% Grid Configuration
% -------------------------------------------------------------------------
config.analysisGridSize = 2;  % 2x2 analysis windows

fprintf('Data path: %s\n', config.dataPath);
fprintf('Output path: %s\n', config.outputPath);
fprintf('Analysis grid size: %dx%d\n', config.analysisGridSize, config.analysisGridSize);

%% =========================================================================
%% SECTION 2: Load Data
%% =========================================================================

fprintf('\n--- Section 2: Load Data ---\n');

% Create output directory if it doesn't exist
if ~exist(config.outputPath, 'dir')
    mkdir(config.outputPath);
    fprintf('Created output directory: %s\n', config.outputPath);
end

% Load simulation data
if ~exist(config.dataPath, 'file')
    error('Data file not found: %s', config.dataPath);
end

data = load(config.dataPath);
stored_spins = data.stored_spins;

% Get dimensions
[T, nRows, nCols] = size(stored_spins);
config.isingGrid = [nRows, nCols];

fprintf('Loaded stored_spins: [%d x %d x %d] (T x rows x cols)\n', T, nRows, nCols);

%% =========================================================================
%% SECTION 3: Setup Spatial Analysis
%% =========================================================================

fprintf('\n--- Section 3: Setup Spatial Analysis ---\n');

distGridSize = config.analysisGridSize;

% Calculate number of non-overlapping positions
nPosRows = floor(nRows / distGridSize);  % 19 positions in row direction
nPosCols = floor(nCols / distGridSize);  % 39 positions in col direction
nTotalPos = nPosRows * nPosCols;         % 741 total positions

fprintf('Analysis grid: %dx%d windows\n', distGridSize, distGridSize);
fprintf('Positions: %d rows x %d cols = %d total\n', nPosRows, nPosCols, nTotalPos);

% Create weight matrix for Moran's I calculation (2x2 grid)
valueMap_dist = rand(distGridSize, distGridSize);
distanceMat_dist = squareform(mL_distanceMat(valueMap_dist));
uniqueDistances_dist = unique(distanceMat_dist);
uniqueDistances_dist(uniqueDistances_dist == 0) = [];
currDistInds_dist = ismember(distanceMat_dist, uniqueDistances_dist(1));
weightMat_dist = zeros(size(distanceMat_dist));
weightMat_dist(currDistInds_dist) = distanceMat_dist(currDistInds_dist);
weightMat_dist(weightMat_dist == inf) = 0;

fprintf('Weight matrix size: [%d x %d]\n', size(weightMat_dist, 1), size(weightMat_dist, 2));

% Centre of the simulation grid
gridCentre = ([nRows, nCols] + 1) / 2;  % [20, 39.5]
fprintf('Grid centre: [%.1f, %.1f]\n', gridCentre(1), gridCentre(2));

% Define centre position for Ground Truth
distCentre_rowStart = floor((nRows - distGridSize) / 2) + 1;
distCentre_colStart = floor((nCols - distGridSize) / 2) + 1;
distCentre_rows = distCentre_rowStart:(distCentre_rowStart + distGridSize - 1);
distCentre_cols = distCentre_colStart:(distCentre_colStart + distGridSize - 1);

fprintf('Centre position: rows %d:%d, cols %d:%d\n', ...
    distCentre_rows(1), distCentre_rows(end), distCentre_cols(1), distCentre_cols(end));

% Initialize storage
DistanceAnalysis = struct();
DistanceAnalysis.gridSize = distGridSize;
DistanceAnalysis.nPositions = nTotalPos;
DistanceAnalysis.nPosRows = nPosRows;
DistanceAnalysis.nPosCols = nPosCols;
DistanceAnalysis.positionCentres = zeros(nTotalPos, 2);  % [row, col] of each position centre
DistanceAnalysis.distanceFromCentre = zeros(nTotalPos, 1);
DistanceAnalysis.WD_to_GT = zeros(nTotalPos, 1);

% Generate position coordinates and compute distances
posIdx = 0;
for pr = 1:nPosRows
    for pc = 1:nPosCols
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

fprintf('Distance range: [%.2f, %.2f]\n', ...
    min(DistanceAnalysis.distanceFromCentre), max(DistanceAnalysis.distanceFromCentre));

%% =========================================================================
%% SECTION 4: Compute Moran's I and Wasserstein Distance
%% =========================================================================

fprintf('\n--- Section 4: Compute Moran''s I and WD ---\n');
fprintf('Computing Moran''s I for all %d positions across %d timeframes...\n', nTotalPos, T);

% Compute Ground Truth: Moran's I for central position across all timeframes
fprintf('Computing Ground Truth (central position)...\n');
GT_moransI = zeros(1, T);
for t = 1:T
    frame = squeeze(stored_spins(t, :, :));
    frame_centre = frame(distCentre_rows, distCentre_cols);
    if all(frame_centre(:) == 0) || all(frame_centre(:) == 1)
        GT_moransI(t) = NaN;
    else
        GT_moransI(t) = mL_moransI(double(frame_centre), weightMat_dist);
    end
end
GT_clean = GT_moransI(~isnan(GT_moransI));
fprintf('Ground Truth: %d valid samples (%.1f%% NaN)\n', ...
    length(GT_clean), 100 * sum(isnan(GT_moransI)) / T);

% Compute Moran's I and WD for each position
fprintf('Processing all positions...\n');
posIdx = 0;
for pr = 1:nPosRows
    for pc = 1:nPosCols
        posIdx = posIdx + 1;

        % Define position boundaries
        rowStart = (pr - 1) * distGridSize + 1;
        colStart = (pc - 1) * distGridSize + 1;
        rows = rowStart:(rowStart + distGridSize - 1);
        cols = colStart:(colStart + distGridSize - 1);

        % Compute Moran's I for this position across all timeframes
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
        DistanceAnalysis.WD_to_GT(posIdx) = wasserstein_1d(GT_clean, moransI_pos_clean);
    end

    % Progress update
    if mod(pr, max(1, floor(nPosRows/5))) == 0
        fprintf('  Processed row %d/%d (%.0f%%)\n', pr, nPosRows, 100*pr/nPosRows);
    end
end

fprintf('Analysis complete.\n');

%% =========================================================================
%% SECTION 5: Visualization
%% =========================================================================

fprintf('\n--- Section 5: Visualization ---\n');

figure('Name', 'Spatial Homogeneity Analysis');

% -------------------------------------------------------------------------
% Panel 1: Scatter plot - WD vs Distance from Centre
% -------------------------------------------------------------------------
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

% -------------------------------------------------------------------------
% Panel 2: Heatmap of WD values
% -------------------------------------------------------------------------
subplot(1, 2, 2);

% Reshape WD values into grid
WD_heatmap = reshape(DistanceAnalysis.WD_to_GT, [nPosRows, nPosCols]);
imagesc(WD_heatmap);
colorbar;
colormap(hot);
axis equal tight;

% Mark centre
hold on;
centrePosRow = ceil(nPosRows / 2);
centrePosCol = ceil(nPosCols / 2);
plot(centrePosCol, centrePosRow, 'go', 'MarkerSize', 15, 'LineWidth', 3);
hold off;

xlabel('Column Position');
ylabel('Row Position');
title('Spatial Map of WD to Centre');
set(gca, 'YDir', 'reverse');

sgtitle(sprintf('Spatial Homogeneity Analysis (%dx%d grid, %d positions)', ...
    distGridSize, distGridSize, nTotalPos), 'FontWeight', 'bold');

saveMyFig('SpatialHomogeneity_DistanceFromCentre', config.outputPath, gcf);
fprintf('Figure saved\n');

%% =========================================================================
%% SECTION 6: Summary Report
%% =========================================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('  SPATIAL HOMOGENEITY ANALYSIS\n');
fprintf('========================================\n');
fprintf('Simulation grid: %d x %d\n', nRows, nCols);
fprintf('Analysis grid: %dx%d\n', distGridSize, distGridSize);
fprintf('Total positions: %d (%d x %d)\n', nTotalPos, nPosRows, nPosCols);
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

% Save results
resultsFile = fullfile(config.outputPath, 'SpatialHomogeneity_Results.mat');
save(resultsFile, 'DistanceAnalysis', 'config', '-v7.3');
fprintf('Results saved to: %s\n', resultsFile);

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
