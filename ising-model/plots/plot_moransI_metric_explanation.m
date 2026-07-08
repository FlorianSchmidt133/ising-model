function plot_moransI_metric_explanation()
% PLOT_MORANSI_METRIC_EXPLANATION - Visual explanation of Moran's I calculation
%
% Creates explanatory figures showing:
%   Figure 1: How Moran's I is calculated from binary grid data
%   Figure 2: Comparison of Entropy vs Dispersion vs Moran's I metrics

%% Figure 1: Moran's I Calculation Explanation
fprintf('\nCreating Figure 1: Moran''s I Calculation\n');

figure('Color', 'w', 'Name', 'Figure 1: Moran''s I Calculation', 'Position', [100 100 1400 800]);

% Example grid size (simplified for visualization)
gridSize = [8, 8];

% Create three example patterns
% Pattern A: Clustered (high positive Moran's I)
pattern_A = zeros(gridSize);
pattern_A(3:5, 3:5) = 1;  % Tight cluster

% Pattern B: Random (Moran's I near 0)
rng(42);  % For reproducibility
pattern_B = double(rand(gridSize) > 0.7);

% Pattern C: Dispersed/Checkerboard (negative Moran's I)
pattern_C = zeros(gridSize);
[X, Y] = meshgrid(1:gridSize(2), 1:gridSize(1));
pattern_C = mod(X + Y, 2);  % Checkerboard pattern

%% Subplot 1: Clustered Pattern
subplot(2, 3, 1);
imagesc(pattern_A);
colormap(gca, [1 1 1; 0.2 0.6 0.9]);
axis equal tight;
title('Pattern A: Clustered', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'XTick', [], 'YTick', []);
text(gridSize(2)/2, -1, 'Active cells cluster together', ...
    'HorizontalAlignment', 'center', 'FontSize', 11);

% Calculate and display Moran's I
moransI_A = calculate_simple_moransI(pattern_A);
text(gridSize(2)/2, gridSize(1)+1.5, sprintf('Moran''s I = %.3f (positive)', moransI_A), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', [0 0.5 0]);

%% Subplot 2: Random Pattern
subplot(2, 3, 2);
imagesc(pattern_B);
colormap(gca, [1 1 1; 0.2 0.6 0.9]);
axis equal tight;
title('Pattern B: Random', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'XTick', [], 'YTick', []);
text(gridSize(2)/2, -1, 'No spatial structure', ...
    'HorizontalAlignment', 'center', 'FontSize', 11);

moransI_B = calculate_simple_moransI(pattern_B);
text(gridSize(2)/2, gridSize(1)+1.5, sprintf('Moran''s I = %.3f (near zero)', moransI_B), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.5 0.5 0.5]);

%% Subplot 3: Dispersed Pattern
subplot(2, 3, 3);
imagesc(pattern_C);
colormap(gca, [1 1 1; 0.2 0.6 0.9]);
axis equal tight;
title('Pattern C: Dispersed', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'XTick', [], 'YTick', []);
text(gridSize(2)/2, -1, 'Dissimilar neighbors', ...
    'HorizontalAlignment', 'center', 'FontSize', 11);

moransI_C = calculate_simple_moransI(pattern_C);
text(gridSize(2)/2, gridSize(1)+1.5, sprintf('Moran''s I = %.3f (negative)', moransI_C), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.8 0 0]);

%% Subplot 4-6: Explanation of Calculation Steps
subplot(2, 3, 4:6);
axis off;

% Title
text(0.5, 0.95, 'Moran''s I Calculation Formula', ...
    'FontSize', 16, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Formula - use LaTeX interpreter for proper math rendering
text(0.5, 0.80, '$I = \frac{N}{\sum_i \sum_j w_{ij}} \cdot \frac{\sum_i \sum_j w_{ij} (X_i - \bar{X})(X_j - \bar{X})}{\sum_i (X_i - \bar{X})^2}$', ...
    'FontSize', 13, 'HorizontalAlignment', 'center', 'Interpreter', 'latex');

% Explanation text
yPos = 0.65;
explanations = {
    'where:', ...
    '  • N = number of grid cells', ...
    '  • X_i = activity value at grid cell i (0 or 1 for binary)', ...
    '  • X̄ = mean activity across all grid cells', ...
    '  • w_{ij} = spatial weight between cells i and j', ...
    '', ...
    'Spatial Weights (w_{ij}):', ...
    '  • Nearest neighbors only (4-connected or 8-connected)', ...
    '  • w_{ij} = 1 if cells i and j are neighbors', ...
    '  • w_{ij} = 0 otherwise', ...
    '', ...
    'Interpretation:', ...
    '  • Positive I → similar values cluster (spatial autocorrelation)', ...
    '  • I ≈ 0 → random spatial pattern (no autocorrelation)', ...
    '  • Negative I → dissimilar values are neighbors (spatial dispersion)'
};

for i = 1:length(explanations)
    text(0.05, yPos, explanations{i}, 'FontSize', 11, 'VerticalAlignment', 'top');
    yPos = yPos - 0.045;
end

%% Figure 2: Comparison of Metrics
fprintf('\nCreating Figure 2: Entropy vs Dispersion vs Moran''s I\n');

figure('Color', 'w', 'Name', 'Figure 2: Entropy vs Dispersion vs Moran''s I', 'Position', [150 150 1400 500]);

% Use same three patterns from Figure 1
patterns = {pattern_A, pattern_B, pattern_C};
pattern_names = {'Clustered', 'Random', 'Dispersed'};

for p = 1:3
    pattern = patterns{p};

    subplot(1, 3, p);

    % Calculate all three metrics
    entropy_val = calculate_simple_entropy(pattern);
    dispersion_val = calculate_simple_dispersion(pattern);
    moransI_val = calculate_simple_moransI(pattern);

    % Create bar plot
    metrics = [entropy_val, dispersion_val, moransI_val];
    metric_names = {'Entropy', 'Dispersion', 'Moran''s I'};
    colors = [0.8 0.2 0.2; 0.2 0.6 0.2; 0.2 0.2 0.8];

    hold on;
    for i = 1:3
        bar(i, metrics(i), 'FaceColor', colors(i, :), 'EdgeColor', 'k', 'LineWidth', 1.5);
    end

    % Add pattern visualization inset
    axes('Position', [subplot(1,3,p).Position(1)+0.02, ...
                       subplot(1,3,p).Position(2)+subplot(1,3,p).Position(4)*0.6, ...
                       subplot(1,3,p).Position(3)*0.3, ...
                       subplot(1,3,p).Position(4)*0.3]);
    imagesc(pattern);
    colormap(gca, [1 1 1; 0.2 0.6 0.9]);
    axis equal tight off;

    % Format main axes
    subplot(1, 3, p);
    xlim([0.5, 3.5]);
    xticks(1:3);
    xticklabels(metric_names);
    xtickangle(45);
    ylabel('Normalized Value');
    title(sprintf('Pattern: %s', pattern_names{p}), 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    box on;
end

% Add overall title
sgtitle('Comparison: Three Spatial Metrics', 'FontSize', 16, 'FontWeight', 'bold');

fprintf('  Figure 1 & 2 complete\n');
end

%% Helper Functions

function I = calculate_simple_moransI(grid)
    % Simple Moran's I calculation for explanation purposes
    % Uses 4-connected neighbors (von Neumann neighborhood)

    [nRows, nCols] = size(grid);
    N = nRows * nCols;

    % Vectorize grid
    X = grid(:);
    X_mean = mean(X);
    X_centered = X - X_mean;

    % Create simple weight matrix (4-connected neighbors)
    W = zeros(N, N);
    for i = 1:nRows
        for j = 1:nCols
            idx = sub2ind([nRows, nCols], i, j);

            % Add neighbors (4-connected)
            if i > 1
                neighbor_idx = sub2ind([nRows, nCols], i-1, j);
                W(idx, neighbor_idx) = 1;
            end
            if i < nRows
                neighbor_idx = sub2ind([nRows, nCols], i+1, j);
                W(idx, neighbor_idx) = 1;
            end
            if j > 1
                neighbor_idx = sub2ind([nRows, nCols], i, j-1);
                W(idx, neighbor_idx) = 1;
            end
            if j < nCols
                neighbor_idx = sub2ind([nRows, nCols], i, j+1);
                W(idx, neighbor_idx) = 1;
            end
        end
    end

    % Calculate Moran's I
    sum_weights = sum(W(:));
    numerator = sum(sum(W .* (X_centered * X_centered')));
    denominator = sum(X_centered.^2);

    if denominator == 0
        I = 0;
    else
        I = (N / sum_weights) * (numerator / denominator);
    end
end

function ent = calculate_simple_entropy(grid)
    % Simplified entropy calculation (proportion of active cells)
    p_active = sum(grid(:)) / numel(grid);
    if p_active == 0 || p_active == 1
        ent = 0;
    else
        ent = -p_active * log2(p_active) - (1-p_active) * log2(1-p_active);
    end
end

function disp = calculate_simple_dispersion(grid)
    % Simplified dispersion (average distance from centroid)
    [nRows, nCols] = size(grid);
    [X, Y] = meshgrid(1:nCols, 1:nRows);

    % Find active cells
    active_mask = grid > 0;

    if sum(active_mask(:)) == 0
        disp = 0;
        return;
    end

    % Centroid of active cells
    centroid_x = mean(X(active_mask));
    centroid_y = mean(Y(active_mask));

    % Average distance from centroid
    distances = sqrt((X(active_mask) - centroid_x).^2 + (Y(active_mask) - centroid_y).^2);
    disp = mean(distances);

    % Normalize to [0, 1]
    max_possible_dist = sqrt(nRows^2 + nCols^2) / 2;
    disp = disp / max_possible_dist;
end
