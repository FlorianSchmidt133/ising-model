function Results = analyze_decay_fitRange_sensitivity(IsingData, ExpStats, Comparison, config)
% ANALYZE_DECAY_FITRANGE_SENSITIVITY Examine how fit range affects tau extraction
%
% Analyzes how varying the decay curve fit range (fitRange) affects the
% extracted tau (decay time constant) and conversion factors for the top 10
% matched Ising simulations per condition.
%
% Inputs:
%   IsingData   - Structure containing Ising simulation results
%                 Required fields: Autocorr_acf (cell array of ACF values)
%   ExpStats    - Structure containing experimental statistics
%                 Required fields: Global.Activity_trials (trial-averaged activity)
%   Comparison  - Structure with condition-specific matching results
%                 Required fields per condition: bestMatch_idx (top match indices)
%   config      - Configuration structure
%                 Required fields: conditions, autocorr.maxLag, outputPath, colors
%
% Outputs:
%   Results     - Structure with sensitivity analysis results:
%                 .fitEndPoints      - [1x10] vector: 10, 20, ..., 100
%                 .exp_tau           - [1x10] experimental tau for each fitRange
%                 .exp_R2            - [1x10] experimental fit R² for each fitRange
%                 .ising_tau         - struct per condition, each [10x10] (top10 x fitRanges)
%                 .ising_R2          - struct per condition, each [10x10]
%                 .conversionFactor  - struct per condition, each [10x10]
%
% Example:
%   Results = analyze_decay_fitRange_sensitivity(IsingData, ExpStats, Comparison, config);
%   % Verify: Results.exp_tau(1) should match ExpStats.Global.Autocorr.tau
%
% See also: Figure5_IsingComparison, computeAutocorrDecay, fitExponentialDecay

fprintf('\n=== Decay Fit Range Sensitivity Analysis ===\n');

% =========================================================================
% Configuration
% =========================================================================

% Define fit range endpoints to test
fitEndPoints = 10:10:100;
nFitRanges = length(fitEndPoints);

% Get configuration parameters
maxLag = config.autocorr.maxLag;
conditions = config.conditions;
nConditions = length(conditions);

% Initialize results structure
Results = struct();
Results.fitEndPoints = fitEndPoints;
Results.maxLag = maxLag;

% =========================================================================
% Experimental Data: Recompute tau for each fit range
% =========================================================================

fprintf('Computing experimental tau for %d fit ranges...\n', nFitRanges);

% Compute trial-averaged ACF by iterating through all conditions
% (handles variable trial lengths across conditions)
acf_accumulated = [];
validTrials = 0;

for c = 1:nConditions
    condition = conditions{c};
    if ~isfield(ExpStats, condition) || ~isfield(ExpStats.(condition), 'Activity_trials')
        continue;
    end

    activity_trials = ExpStats.(condition).Activity_trials;
    [nTrials, ~] = size(activity_trials);

    for t = 1:nTrials
        trial_data = activity_trials(t, :);
        trial_data = trial_data(~isnan(trial_data));
        if length(trial_data) > maxLag * 2
            [acf, ~] = xcorr(trial_data - mean(trial_data), maxLag, 'normalized');
            acf_trial = acf(maxLag+1:end);  % Positive lags only
            acf_accumulated = [acf_accumulated; acf_trial(:)']; %#ok<AGROW>
            validTrials = validTrials + 1;
        end
    end
end

if validTrials == 0
    warning('No valid trials found for experimental ACF computation.');
    exp_acf_mean = NaN(1, maxLag + 1);
else
    exp_acf_mean = mean(acf_accumulated, 1, 'omitnan');
end
exp_lags = (0:maxLag)';

fprintf('  Computed ACF from %d valid trials across %d conditions\n', validTrials, nConditions);

% Now fit with each range
Results.exp_tau = zeros(1, nFitRanges);
Results.exp_R2 = zeros(1, nFitRanges);

for f = 1:nFitRanges
    fitRange = [1, fitEndPoints(f)];
    [tau, fitResult] = fitExponentialDecay_local(exp_acf_mean, exp_lags, fitRange);
    Results.exp_tau(f) = tau;
    if isfield(fitResult, 'R2')
        Results.exp_R2(f) = fitResult.R2;
    else
        Results.exp_R2(f) = NaN;
    end
end

fprintf('  Experimental tau: range [%.2f, %.2f] across fit ranges\n', ...
    min(Results.exp_tau), max(Results.exp_tau));

% =========================================================================
% Ising Data: Refit tau for top 10 matches per condition
% =========================================================================

fprintf('Computing Ising tau for top 10 matches per condition...\n');

% Check if ACF data is available
if ~isfield(IsingData, 'Autocorr_acf') || isempty(IsingData.Autocorr_acf)
    warning('IsingData.Autocorr_acf not available. Cannot compute Ising tau sensitivity.');
    Results.ising_tau = struct();
    Results.ising_R2 = struct();
    Results.conversionFactor = struct();
    return;
end

% Lags array for Ising ACF
ising_lags = (0:maxLag)';

% Initialize per-condition storage
Results.ising_tau = struct();
Results.ising_R2 = struct();
Results.conversionFactor = struct();

for c = 1:nConditions
    condition = conditions{c};

    if ~isfield(Comparison, condition)
        fprintf('  Skipping %s (not in Comparison)\n', condition);
        continue;
    end

    % Get top 10 match indices
    nTop = min(10, length(Comparison.(condition).bestMatch_idx));
    top_idx = Comparison.(condition).bestMatch_idx(1:nTop);

    % Initialize matrices for this condition
    tau_matrix = zeros(nTop, nFitRanges);
    R2_matrix = zeros(nTop, nFitRanges);
    cf_matrix = zeros(nTop, nFitRanges);

    for m = 1:nTop
        sim_idx = top_idx(m);

        % Get stored ACF for this simulation
        ising_acf = IsingData.Autocorr_acf{sim_idx};

        if isempty(ising_acf) || all(isnan(ising_acf))
            tau_matrix(m, :) = NaN;
            R2_matrix(m, :) = NaN;
            cf_matrix(m, :) = NaN;
            continue;
        end

        % Ensure correct length
        if length(ising_acf) ~= maxLag + 1
            warning('ACF length mismatch for sim %d. Expected %d, got %d.', ...
                sim_idx, maxLag + 1, length(ising_acf));
            tau_matrix(m, :) = NaN;
            R2_matrix(m, :) = NaN;
            cf_matrix(m, :) = NaN;
            continue;
        end

        % Refit with each range
        for f = 1:nFitRanges
            fitRange = [1, fitEndPoints(f)];
            [tau, fitResult] = fitExponentialDecay_local(ising_acf(:), ising_lags, fitRange);
            tau_matrix(m, f) = tau;

            if isfield(fitResult, 'R2')
                R2_matrix(m, f) = fitResult.R2;
            else
                R2_matrix(m, f) = NaN;
            end

            % Conversion factor: tau_exp / tau_ising
            % (frames per MC sweep, or how many real frames correspond to 1 Ising time step)
            if ~isnan(tau) && tau > 0 && ~isnan(Results.exp_tau(f)) && Results.exp_tau(f) > 0
                cf_matrix(m, f) = Results.exp_tau(f) / tau;
            else
                cf_matrix(m, f) = NaN;
            end
        end
    end

    Results.ising_tau.(condition) = tau_matrix;
    Results.ising_R2.(condition) = R2_matrix;
    Results.conversionFactor.(condition) = cf_matrix;

    fprintf('  %s: processed %d top matches\n', condition, nTop);
end

% =========================================================================
% Visualization: 2x2 subplot figure
% =========================================================================

fprintf('Creating sensitivity analysis figure...\n');

fig = figure('Name', 'Decay Fit Range Sensitivity Analysis');

% Define colors
if isfield(config, 'colors')
    colors = config.colors;
else
    % Default colors if not provided
    colors.Naive = [0.3373, 0.7059, 0.9137];
    colors.Beginner = [0.8431, 0.2549, 0.6078];
    colors.Expert = [0, 0.6196, 0.4510];
    colors.NoSpout = [0.8353, 0.3686, 0];
end
expColor = [0.2, 0.2, 0.2];  % Dark gray for experimental

% -------------------------------------------------------------------------
% Subplot 1: Experimental tau vs fitRange
% -------------------------------------------------------------------------
subplot(2, 2, 1);
hold on;

plot(fitEndPoints, Results.exp_tau, 'o-', 'Color', expColor, ...
    'MarkerFaceColor', expColor, 'LineWidth', 2, 'MarkerSize', 8);

% Add R² annotation at each point
for f = 1:nFitRanges
    if ~isnan(Results.exp_R2(f))
        text(fitEndPoints(f), Results.exp_tau(f) + 0.5, ...
            sprintf('%.2f', Results.exp_R2(f)), ...
            'FontSize', 7, 'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5]);
    end
end

xlabel('Fit Range End Point (frames)');
ylabel('Experimental \tau (frames)');
title('Experimental Tau vs Fit Range');
grid on;

% Add reference line at original fitRange endpoint
if config.autocorr.fitRange(2) <= 100
    xline(config.autocorr.fitRange(2), '--k', 'LineWidth', 1);
    text(config.autocorr.fitRange(2), max(Results.exp_tau)*0.95, ...
        sprintf('Used in simulation selection: %d', config.autocorr.fitRange(2)), ...
        'FontSize', 8, 'HorizontalAlignment', 'left');
end

hold off;

% -------------------------------------------------------------------------
% Subplot 2: Ising tau vs fitRange (mean±std per condition)
% -------------------------------------------------------------------------
subplot(2, 2, 2);
hold on;

for c = 1:nConditions
    condition = conditions{c};

    if ~isfield(Results.ising_tau, condition)
        continue;
    end

    tau_matrix = Results.ising_tau.(condition);
    tau_mean = mean(tau_matrix, 1, 'omitnan');
    tau_std = std(tau_matrix, 0, 1, 'omitnan');

    if isfield(colors, condition)
        condColor = colors.(condition);
    else
        condColor = [0.5, 0.5, 0.5];
    end

    % Plot shaded error region (exclude from legend)
    fill([fitEndPoints, fliplr(fitEndPoints)], ...
        [tau_mean + tau_std, fliplr(tau_mean - tau_std)], ...
        condColor, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % Plot mean line
    plot(fitEndPoints, tau_mean, 'o-', 'Color', condColor, ...
        'MarkerFaceColor', condColor, 'LineWidth', 2, 'MarkerSize', 6, ...
        'DisplayName', condition);
end

xlabel('Fit Range End Point (frames)');
ylabel('Ising \tau (MC sweeps)');
title('Ising Tau vs Fit Range (Top 10 Matches)');
legend('Location', 'best');
grid on;
hold off;

% -------------------------------------------------------------------------
% Subplot 3: Conversion factor vs fitRange (mean±std per condition)
% -------------------------------------------------------------------------
subplot(2, 2, 3);
hold on;

for c = 1:nConditions
    condition = conditions{c};

    if ~isfield(Results.conversionFactor, condition)
        continue;
    end

    cf_matrix = Results.conversionFactor.(condition);
    cf_mean = mean(cf_matrix, 1, 'omitnan');
    cf_std = std(cf_matrix, 0, 1, 'omitnan');

    if isfield(colors, condition)
        condColor = colors.(condition);
    else
        condColor = [0.5, 0.5, 0.5];
    end

    % Plot shaded error region (exclude from legend)
    fill([fitEndPoints, fliplr(fitEndPoints)], ...
        [cf_mean + cf_std, fliplr(cf_mean - cf_std)], ...
        condColor, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % Plot mean line
    plot(fitEndPoints, cf_mean, 'o-', 'Color', condColor, ...
        'MarkerFaceColor', condColor, 'LineWidth', 2, 'MarkerSize', 6, ...
        'DisplayName', condition);
end

xlabel('Fit Range End Point (frames)');
ylabel('Conversion Factor (\tau_{exp} / \tau_{ising})');
title('Temporal Conversion Factor vs Fit Range');
legend('Location', 'best');
grid on;
hold off;

% -------------------------------------------------------------------------
% Subplot 4: R² (fit quality) vs fitRange
% -------------------------------------------------------------------------
subplot(2, 2, 4);
hold on;

% Experimental R²
plot(fitEndPoints, Results.exp_R2, 's-', 'Color', expColor, ...
    'MarkerFaceColor', expColor, 'LineWidth', 2, 'MarkerSize', 8, ...
    'DisplayName', 'Experimental');

% Ising R² per condition
for c = 1:nConditions
    condition = conditions{c};

    if ~isfield(Results.ising_R2, condition)
        continue;
    end

    R2_matrix = Results.ising_R2.(condition);
    R2_mean = mean(R2_matrix, 1, 'omitnan');

    if isfield(colors, condition)
        condColor = colors.(condition);
    else
        condColor = [0.5, 0.5, 0.5];
    end

    plot(fitEndPoints, R2_mean, 'o-', 'Color', condColor, ...
        'MarkerFaceColor', condColor, 'LineWidth', 1.5, 'MarkerSize', 5, ...
        'DisplayName', sprintf('Ising (%s)', condition));
end

xlabel('Fit Range End Point (frames)');
ylabel('R^2 (Fit Quality)');
title('Fit Quality vs Fit Range');
legend('Location', 'best', 'FontSize', 8);
grid on;
ylim([0, 1]);
hold off;

% -------------------------------------------------------------------------
% Save figure
% -------------------------------------------------------------------------
if isfield(config, 'outputPath') && ~isempty(config.outputPath)
    outputDir = fullfile(config.outputPath, 'TemporalDynamics');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    figPath = fullfile(outputDir, 'decay_fitRange_sensitivity.png');
    saveas(fig, figPath);
    fprintf('Figure saved to: %s\n', figPath);

    % Also save as .fig for editing
    figPathFig = fullfile(outputDir, 'decay_fitRange_sensitivity.fig');
    saveas(fig, figPathFig);
end

% =========================================================================
% Summary statistics
% =========================================================================

fprintf('\n=== Summary Statistics ===\n');
fprintf('Fit Range Endpoints: %s\n', mat2str(fitEndPoints));
fprintf('\nExperimental Tau:\n');
fprintf('  Range: [%.2f, %.2f] frames\n', min(Results.exp_tau), max(Results.exp_tau));
fprintf('  Original (fitRange [1,%d]): %.2f frames\n', ...
    config.autocorr.fitRange(2), Results.exp_tau(fitEndPoints == config.autocorr.fitRange(2)));
fprintf('  R² range: [%.3f, %.3f]\n', min(Results.exp_R2), max(Results.exp_R2));

for c = 1:nConditions
    condition = conditions{c};
    if isfield(Results.ising_tau, condition)
        tau_all = Results.ising_tau.(condition)(:);
        cf_all = Results.conversionFactor.(condition)(:);
        fprintf('\n%s (top 10 matches):\n', condition);
        fprintf('  Tau range: [%.2f, %.2f] MC sweeps\n', ...
            min(tau_all, [], 'omitnan'), max(tau_all, [], 'omitnan'));
        fprintf('  Conversion factor range: [%.3f, %.3f]\n', ...
            min(cf_all, [], 'omitnan'), max(cf_all, [], 'omitnan'));
    end
end

fprintf('\n=== Sensitivity Analysis Complete ===\n');

end

% =========================================================================
% Local helper function: fitExponentialDecay_local
% =========================================================================
function [tau, fitResult] = fitExponentialDecay_local(acf, lags, fitRange)
% FITEXPONENTIALDECAY_LOCAL Fit simple exponential decay to autocorrelation
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
