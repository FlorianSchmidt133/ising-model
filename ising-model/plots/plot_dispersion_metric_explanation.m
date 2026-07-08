function plot_dispersion_metric_explanation()
% PLOT_DISPERSION_METRIC_EXPLANATION - Create explanatory visualizations of dispersion metric
%
% Creates TWO comprehensive figures:
%
% FIGURE 1: Binary Dispersion Only
%   - Shows 4 binary patterns with increasing dispersion
%   - Demonstrates how spatial organization affects dispersion values
%   - Used for thresholded/binarized data (BIN, BIN_AN, BIN_ALL)
%
% FIGURE 2: Binary vs Weighted Comparison
%   - Compares binary and weighted calculations on the same patterns
%   - Explains when to use each method
%   - Binary: for binarized data | Weighted: for continuous data
%
% Usage:
%   plot_dispersion_metric_explanation()
%
% This function is used in Figure5_EntropyComparison.m to explain the
% dispersion metric used for analyzing spatial organization of neural activity.

    fprintf('\n=== Creating Dispersion Metric Explanation Figures ===\n');

    % Grid dimensions (matching actual data)
    gridY = 13;
    gridX = 26;

    % Generate grid coordinates
    [gridX_coords, gridY_coords] = meshgrid(1:gridX, 1:gridY);
    grid_coords = [gridY_coords(:), gridX_coords(:)]; % [nGridCells × 2]

    %% Create Figure 1: Binary Dispersion Only
    create_figure1_binary_only(gridY, gridX, grid_coords);

    %% Create Figure 2: Binary vs Weighted Comparison
    create_figure2_comparison(gridY, gridX, grid_coords);

    fprintf('  Both figures created successfully\n');
end

%% =========================================================================
%% Figure 1: Binary Dispersion Only
%% =========================================================================
function create_figure1_binary_only(gridY, gridX, grid_coords)
    fprintf('  Creating Figure 1: Binary Dispersion Only...\n');

    center_y = round(gridY/2);
    center_x = round(gridX/2);

    %% Create 4 binary patterns with increasing dispersion
    % Pattern 1: Tight 3×3 blob (very low dispersion)
    pattern1 = zeros(gridY, gridX);
    pattern1(center_y-1:center_y+1, center_x-1:center_x+1) = 1;

    % Pattern 2: Medium 5×5 blob (low-medium dispersion)
    pattern2 = zeros(gridY, gridX);
    pattern2(center_y-2:center_y+2, center_x-2:center_x+2) = 1;

    % Pattern 3: Ring pattern (medium dispersion)
    pattern3 = zeros(gridY, gridX);
    radius = 3;
    for i = 1:gridY
        for j = 1:gridX
            dist = sqrt((i-center_y)^2 + (j-center_x)^2);
            if dist >= radius-0.5 && dist <= radius+0.5
                pattern3(i,j) = 1;
            end
        end
    end

    % Pattern 4: Scattered cells (high dispersion)
    pattern4 = zeros(gridY, gridX);
    rng(42); % For reproducibility
    scatter_indices = randperm(gridY*gridX, 15);
    pattern4(scatter_indices) = 1;

    %% Calculate dispersion for each pattern
    disp1 = calculate_binary_dispersion(pattern1(:), grid_coords);
    disp2 = calculate_binary_dispersion(pattern2(:), grid_coords);
    disp3 = calculate_binary_dispersion(pattern3(:), grid_coords);
    disp4 = calculate_binary_dispersion(pattern4(:), grid_coords);

    fprintf('    Pattern 1 (Tight 3×3): Dispersion = %.2f\n', disp1);
    fprintf('    Pattern 2 (Medium 5×5): Dispersion = %.2f\n', disp2);
    fprintf('    Pattern 3 (Ring): Dispersion = %.2f\n', disp3);
    fprintf('    Pattern 4 (Scattered): Dispersion = %.2f\n', disp4);

    %% Create figure
    fig1 = figure('Name', 'Figure 1: Binary Dispersion Metric');
    set(fig1, 'Color', 'w');

    patterns = {pattern1, pattern2, pattern3, pattern4};
    dispersions = [disp1, disp2, disp3, disp4];
    titles = {'Tight 3×3 Blob', 'Medium 5×5 Blob', 'Ring Pattern', 'Scattered'};
    subtitles = {'(Very Low)', '(Low-Medium)', '(Medium)', '(High Dispersion)'};

    %% Row 1: Simple pattern view
    for p = 1:4
        subplot(3, 4, p);
        imagesc(patterns{p});
        colormap(gca, 'hot');
        axis equal tight;
        title(sprintf('%s\n%s', titles{p}, subtitles{p}), 'FontWeight', 'bold');
        xlabel('Grid X');
        ylabel('Grid Y');
        set(gca, 'YDir', 'normal');
    end

    %% Row 2: Detailed calculation view with centroid and distances
    for p = 1:4
        subplot(3, 4, 4 + p);
        hold on;

        % Plot grid cells
        imagesc(patterns{p});
        colormap(gca, 'hot');
        axis equal tight;
        set(gca, 'YDir', 'normal');

        % Calculate and plot centroid
        activity = patterns{p}(:);
        active_inds = find(activity > 0);
        act_pos = grid_coords(active_inds, :);
        centroid = mean(act_pos, 1);

        % Plot centroid
        plot(centroid(2), centroid(1), 'c*', 'MarkerSize', 15, 'LineWidth', 2);
        plot(centroid(2), centroid(1), 'co', 'MarkerSize', 20, 'LineWidth', 2);

        % Plot distance lines (max 10 for clarity)
        n_lines = min(10, length(active_inds));
        line_inds = active_inds(round(linspace(1, length(active_inds), n_lines)));

        for i = 1:length(line_inds)
            idx = line_inds(i);
            pos = grid_coords(idx, :);
            dist = sqrt(sum((pos - centroid).^2));

            % Color lines by distance
            plot([centroid(2), pos(2)], [centroid(1), pos(1)], ...
                 '-', 'Color', [0.2 0.8 0.9 0.5], 'LineWidth', 1.5);
        end

        xlabel('Grid X');
        ylabel('Grid Y');
        title(sprintf('Dispersion = %.2f grid units', dispersions(p)), 'FontSize', 10);

        if p == 1
            legend({'Centroid', '','Distances'}, 'Location', 'northeast', 'FontSize', 8);
        end

        hold off;
    end

    %% Row 3: Mathematical explanation
    subplot(3, 4, [9, 10, 11, 12]);
    axis off;

    text_x = 0.05;
    text_y = 0.95;
    line_spacing = 0.10;

    text(text_x, text_y, 'Binary Dispersion Metric (for Thresholded Data)', ...
         'FontSize', 14, 'FontWeight', 'bold', 'Units', 'normalized');

    text_y = text_y - line_spacing*1.2;
    text(text_x, text_y, 'Algorithm:', ...
         'FontSize', 12, 'FontWeight', 'bold', 'Units', 'normalized');

    text_y = text_y - line_spacing*0.8;
    text(text_x+0.02, text_y, '1. Identify active cells (binary: activity > 0)', ...
         'FontSize', 10, 'Units', 'normalized');

    text_y = text_y - line_spacing*0.8;
    text(text_x+0.02, text_y, '2. Calculate centroid: centroid = mean(active_positions)', ...
         'FontSize', 10, 'FontName', 'Courier', 'Units', 'normalized');

    text_y = text_y - line_spacing*0.8;
    text(text_x+0.02, text_y, '3. Compute Euclidean distances: distance_i = √[(x_i - centroid_x)² + (y_i - centroid_y)²]', ...
         'FontSize', 10, 'FontName', 'Courier', 'Units', 'normalized');

    text_y = text_y - line_spacing*0.8;
    text(text_x+0.02, text_y, '4. Average distances: dispersion = mean(distances)', ...
         'FontSize', 10, 'FontName', 'Courier', 'Units', 'normalized');

    text_y = text_y - line_spacing*1.5;
    text(text_x, text_y, 'When to Use Binary Dispersion:', ...
         'FontSize', 12, 'FontWeight', 'bold', 'Color', [0 0 0.7], 'Units', 'normalized');

    text_y = text_y - line_spacing*0.8;
    text(text_x+0.02, text_y, '• For binarized/thresholded data', ...
         'FontSize', 10, 'Units', 'normalized');

    text_y = text_y - line_spacing*0.8;
    text(text_x+0.02, text_y, '• After z-score thresholding (e.g., activity > 1.5σ)', ...
         'FontSize', 10, 'Units', 'normalized');

    text_y = text_y - line_spacing*0.8;
    text(text_x+0.02, text_y, '• Treats cells as simply ON or OFF (ignores activity intensity)', ...
         'FontSize', 10, 'Units', 'normalized');

    text_y = text_y - line_spacing*1.3;
    text(text_x, text_y, 'Biological Interpretation:', ...
         'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.8 0 0], 'Units', 'normalized');

    text_y = text_y - line_spacing*0.8;
    text(text_x+0.02, text_y, '• Low dispersion (~1-2 units) = Tight blob = Organized, localized activity', ...
         'FontSize', 10, 'Color', [0 0.5 0], 'Units', 'normalized');

    text_y = text_y - line_spacing*0.8;
    text(text_x+0.02, text_y, '• High dispersion (~6-10 units) = Scattered = Disorganized, distributed activity', ...
         'FontSize', 10, 'Color', [0.8 0 0], 'Units', 'normalized');

    sgtitle('Binary Dispersion: Quantifying Spatial Organization of Thresholded Activity', ...
            'FontSize', 16, 'FontWeight', 'bold');
end

%% =========================================================================
%% Figure 2: Binary vs Weighted Comparison
%% =========================================================================
function create_figure2_comparison(gridY, gridX, grid_coords)
    fprintf('  Creating Figure 2: Binary vs Weighted Comparison...\n');

    center_y = round(gridY/2);
    center_x = round(gridX/2);

    %% Create 2 example patterns with continuous activity
    % Pattern A: Hotspot in Ring (moderate ring + one bright cell)
    patternA = zeros(gridY, gridX);
    ring_radius = 3;
    for i = 1:gridY
        for j = 1:gridX
            dist = sqrt((i-center_y)^2 + (j-center_x)^2);
            % Create ring of moderate activity
            if dist >= ring_radius-0.7 && dist <= ring_radius+0.7
                patternA(i,j) = 0.3;  % Moderate uniform activity
            end
        end
    end
    % Add one very bright hotspot in the ring (offset from center)
    hotspot_y = center_y - 1;
    hotspot_x = center_x + 2;
    patternA(hotspot_y, hotspot_x) = 1.0;  % Very bright cell

    % Pattern B: Gradient Cloud (strong center, weak periphery)
    patternB = zeros(gridY, gridX);
    for i = 1:gridY
        for j = 1:gridX
            dist = sqrt((i-center_y)^2 + (j-center_x)^2);
            if dist <= 4.5
                % Strong central peak with gradient decay
                patternB(i,j) = max(0.02, 1.0 * exp(-dist^2 / 8));
            end
            % Add very weak peripheral cells (just above threshold)
            if dist > 4.5 && dist <= 6
                patternB(i,j) = 0.03 + 0.02*rand();  % Very weak, noisy periphery
            end
        end
    end

    %% Calculate both binary and weighted dispersion
    % Pattern A
    dispA_weighted = calculate_weighted_dispersion(patternA(:), grid_coords);
    dispA_binary = calculate_binary_dispersion(double(patternA(:) > 0.01), grid_coords);

    % Pattern B
    dispB_weighted = calculate_weighted_dispersion(patternB(:), grid_coords);
    dispB_binary = calculate_binary_dispersion(double(patternB(:) > 0.01), grid_coords);

    fprintf('    Pattern A (Hotspot in Ring): Binary = %.2f, Weighted = %.2f\n', dispA_binary, dispA_weighted);
    fprintf('    Pattern B (Gradient Cloud): Binary = %.2f, Weighted = %.2f\n', dispB_binary, dispB_weighted);

    %% Create figure
    fig2 = figure('Name', 'Figure 2: Binary vs Weighted Dispersion');
    set(fig2, 'Color', 'w');

    %% Pattern A: Binary vs Weighted
    % Binary version
    subplot(2, 4, 1);
    imagesc(double(patternA > 0.01));
    colormap(gca, 'hot');
    axis equal tight;
    set(gca, 'YDir', 'normal');
    title('A: Hotspot in Ring (Binary)', 'FontWeight', 'bold');
    xlabel('Grid X');
    ylabel('Grid Y');

    subplot(2, 4, 2);
    hold on;
    imagesc(double(patternA > 0.01));
    colormap(gca, 'hot');
    axis equal tight;
    set(gca, 'YDir', 'normal');

    % Plot centroid for binary
    activity_bin = double(patternA(:) > 0.01);
    active_inds = find(activity_bin > 0);
    act_pos = grid_coords(active_inds, :);
    centroid_bin = mean(act_pos, 1);
    plot(centroid_bin(2), centroid_bin(1), 'c*', 'MarkerSize', 15, 'LineWidth', 2);
    plot(centroid_bin(2), centroid_bin(1), 'co', 'MarkerSize', 20, 'LineWidth', 2);

    % Plot some distance lines
    n_lines = min(8, length(active_inds));
    line_inds = active_inds(round(linspace(1, length(active_inds), n_lines)));
    for i = 1:length(line_inds)
        idx = line_inds(i);
        pos = grid_coords(idx, :);
        plot([centroid_bin(2), pos(2)], [centroid_bin(1), pos(1)], ...
             '-', 'Color', [0.2 0.8 0.9 0.5], 'LineWidth', 1.5);
    end

    title(sprintf('Binary Dispersion = %.2f', dispA_binary), 'FontSize', 10);
    xlabel('Grid X');
    ylabel('Grid Y');
    hold off;

    % Weighted version
    subplot(2, 4, 3);
    imagesc(patternA);
    colormap(gca, 'hot');
    axis equal tight;
    set(gca, 'YDir', 'normal');
    title('A: Hotspot in Ring (Weighted)', 'FontWeight', 'bold');
    xlabel('Grid X');
    ylabel('Grid Y');
    colorbar('Location', 'eastoutside');

    subplot(2, 4, 4);
    hold on;
    imagesc(patternA);
    colormap(gca, 'hot');
    alpha(0.7);
    axis equal tight;
    set(gca, 'YDir', 'normal');

    % Plot weighted centroid
    activity = patternA(:);
    active_mask = activity > 0.01;
    active_coords = grid_coords(active_mask, :);
    active_weights = activity(active_mask);
    total_weight = sum(active_weights);
    centroid_weighted = sum(active_coords .* active_weights, 1) / total_weight;
    plot(centroid_weighted(2), centroid_weighted(1), 'c*', 'MarkerSize', 15, 'LineWidth', 2);
    plot(centroid_weighted(2), centroid_weighted(1), 'co', 'MarkerSize', 20, 'LineWidth', 2);

    % Plot weighted distance lines
    active_inds_w = find(active_mask);
    n_lines = min(8, length(active_inds_w));
    line_inds = active_inds_w(round(linspace(1, length(active_inds_w), n_lines)));
    for i = 1:length(line_inds)
        idx = line_inds(i);
        pos = grid_coords(idx, :);
        plot([centroid_weighted(2), pos(2)], [centroid_weighted(1), pos(1)], ...
             '-', 'Color', [0.2 0.8 0.9 0.5], 'LineWidth', 1.5);
    end

    title(sprintf('Weighted Dispersion = %.2f', dispA_weighted), 'FontSize', 10);
    xlabel('Grid X');
    ylabel('Grid Y');
    hold off;

    %% Pattern B: Binary vs Weighted
    % Binary version
    subplot(2, 4, 5);
    imagesc(double(patternB > 0.01));
    colormap(gca, 'hot');
    axis equal tight;
    set(gca, 'YDir', 'normal');
    title('B: Gradient Cloud (Binary)', 'FontWeight', 'bold');
    xlabel('Grid X');
    ylabel('Grid Y');

    subplot(2, 4, 6);
    hold on;
    imagesc(double(patternB > 0.01));
    colormap(gca, 'hot');
    axis equal tight;
    set(gca, 'YDir', 'normal');

    % Plot centroid for binary
    activity_bin = double(patternB(:) > 0.01);
    active_inds = find(activity_bin > 0);
    act_pos = grid_coords(active_inds, :);
    centroid_bin = mean(act_pos, 1);
    plot(centroid_bin(2), centroid_bin(1), 'c*', 'MarkerSize', 15, 'LineWidth', 2);
    plot(centroid_bin(2), centroid_bin(1), 'co', 'MarkerSize', 20, 'LineWidth', 2);

    % Plot distance lines
    n_lines = min(8, length(active_inds));
    line_inds = active_inds(round(linspace(1, length(active_inds), n_lines)));
    for i = 1:length(line_inds)
        idx = line_inds(i);
        pos = grid_coords(idx, :);
        plot([centroid_bin(2), pos(2)], [centroid_bin(1), pos(1)], ...
             '-', 'Color', [0.2 0.8 0.9 0.5], 'LineWidth', 1.5);
    end

    title(sprintf('Binary Dispersion = %.2f', dispB_binary), 'FontSize', 10);
    xlabel('Grid X');
    ylabel('Grid Y');
    hold off;

    % Weighted version
    subplot(2, 4, 7);
    imagesc(patternB);
    colormap(gca, 'hot');
    axis equal tight;
    set(gca, 'YDir', 'normal');
    title('B: Gradient Cloud (Weighted)', 'FontWeight', 'bold');
    xlabel('Grid X');
    ylabel('Grid Y');
    colorbar('Location', 'eastoutside');

    subplot(2, 4, 8);
    hold on;
    imagesc(patternB);
    colormap(gca, 'hot');
    alpha(0.7);
    axis equal tight;
    set(gca, 'YDir', 'normal');

    % Plot weighted centroid
    activity = patternB(:);
    active_mask = activity > 0.01;
    active_coords = grid_coords(active_mask, :);
    active_weights = activity(active_mask);
    total_weight = sum(active_weights);
    centroid_weighted = sum(active_coords .* active_weights, 1) / total_weight;
    plot(centroid_weighted(2), centroid_weighted(1), 'c*', 'MarkerSize', 15, 'LineWidth', 2);
    plot(centroid_weighted(2), centroid_weighted(1), 'co', 'MarkerSize', 20, 'LineWidth', 2);

    % Plot weighted distance lines
    active_inds_w = find(active_mask);
    n_lines = min(8, length(active_inds_w));
    line_inds = active_inds_w(round(linspace(1, length(active_inds_w), n_lines)));
    for i = 1:length(line_inds)
        idx = line_inds(i);
        pos = grid_coords(idx, :);
        plot([centroid_weighted(2), pos(2)], [centroid_weighted(1), pos(1)], ...
             '-', 'Color', [0.2 0.8 0.9 0.5], 'LineWidth', 1.5);
    end

    title(sprintf('Weighted Dispersion = %.2f', dispB_weighted), 'FontSize', 10);
    xlabel('Grid X');
    ylabel('Grid Y');
    hold off;

    %% Add overall explanation
    annotation('textbox', [0.05, 0.02, 0.9, 0.08], 'String', ...
        {'BINARY DISPERSION: Treats cells as ON/OFF after thresholding. Centroid = mean(active positions). Use for binarised data', ...
         'WEIGHTED DISPERSION: Weights by activity intensity. Centroid = Σ(position × activity) / Σ(activity). Use for raw and Z-scored data', ...
         'KEY DIFFERENCE: Weighted accounts for activity intensity → more accurate centroid and dispersion for continuous data.'}, ...
        'FontSize', 10, 'FontWeight', 'bold', 'EdgeColor', [0.3 0.3 0.7], 'LineWidth', 2, ...
        'BackgroundColor', [0.95 0.95 1], 'FitBoxToText', 'off', 'HorizontalAlignment', 'left');

    sgtitle('Binary vs Weighted Dispersion', ...
            'FontSize', 16, 'FontWeight', 'bold');
end

%% =========================================================================
%% Helper Function: Calculate binary dispersion
%% =========================================================================
function disp = calculate_binary_dispersion(activity_bin, grid_coords)
% CALCULATE_BINARY_DISPERSION - Calculate dispersion for binary data
%
% Matches BinarizeDataOnTheGrid.py implementation
% Finds active cells, computes centroid, averages Euclidean distances
%
% INPUTS:
%   activity_bin  - Binary activity vector [nGridCells × 1]
%   grid_coords   - Grid coordinates [nGridCells × 2]
%
% OUTPUTS:
%   disp          - Average Euclidean distance from centroid

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
%% Helper Function: Calculate weighted dispersion
%% =========================================================================
function disp = calculate_weighted_dispersion(activity, grid_coords)
% CALCULATE_WEIGHTED_DISPERSION - Calculate dispersion weighted by activity levels
%
% Uses activity-weighted centroid and activity-weighted average distance
%
% INPUTS:
%   activity      - Continuous activity vector [nGridCells × 1]
%   grid_coords   - Grid coordinates [nGridCells × 2]
%
% OUTPUTS:
%   disp          - Weighted average Euclidean distance from centroid

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
