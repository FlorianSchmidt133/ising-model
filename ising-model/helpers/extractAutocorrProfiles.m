%% extractAutocorrProfiles.m
% Extracts temporal autocorrelation profiles from NMF H components
% for use in Ising model comparison (Figure 5)
%
% This script loads NMF data, collapses H components (max across components
% per timepoint), and computes autocorrelation with exponential decay fitting.
%
% Loads from: <DATA_ROOT>/NMF_littleRegularisation_Grid40.mat
% Saves to: Fig. 5 Model\IsingModels\Data\AutocorrProfiles.mat
%
% See also: Figure5_IsingComparison.m

%% Parameters
conditions = {'Naive', 'Beginner', 'Expert', 'NoSpout'};
position = 'P1';  % Position options: 'P1', 'P3', or 'All'
maxLag = 100;     % Maximum lag for autocorrelation
fitRange = [1, 50];  % Lag range for exponential fit

% Input/Output paths
nmfPath = mba_p('NMF_littleRegularisation_Grid40.mat');
outputPath = 'Fig. 5 Model\IsingModels\Data\AutocorrProfiles.mat';

%% Load NMF data
fprintf('Loading NMF data from: %s\n', nmfPath);
if ~exist(nmfPath, 'file')
    error('NMF data file not found: %s', nmfPath);
end

load(nmfPath, 'H_all2');  % Mean-subtracted H components
fprintf('Loaded H_all2 structure\n');

%% Initialize output structure
AutocorrProfiles = struct();
AutocorrProfiles.metadata.nmfPath = nmfPath;
AutocorrProfiles.metadata.position = position;
AutocorrProfiles.metadata.maxLag = maxLag;
AutocorrProfiles.metadata.fitRange = fitRange;
AutocorrProfiles.metadata.createdDate = datestr(now);

%% Process each condition
fprintf('\n=== Processing Conditions ===\n');

for c = 1:length(conditions)
    condition = conditions{c};
    H_condName = [condition 'Individual'];

    fprintf('\nProcessing %s (field: %s)...\n', condition, H_condName);

    % Check if condition exists in H_all2
    if ~isfield(H_all2, H_condName)
        fprintf('  Skipping: %s not found in H_all2\n', H_condName);
        continue;
    end

    % Check if position exists
    if ~isfield(H_all2.(H_condName), position)
        fprintf('  Skipping: position %s not found for %s\n', position, H_condName);
        continue;
    end

    H_cond = H_all2.(H_condName).(position);

    % Collect collapsed H data across recordings and trials
    nRecs = length(H_cond);
    all_collapsed = [];
    per_rec_tau = [];

    fprintf('  Found %d recordings\n', nRecs);

    for rec = 1:nRecs
        rec_data = H_cond{rec};
        if isempty(rec_data)
            continue;
        end

        % rec_data is typically a cell array [ranks x trials]
        rec_collapsed = [];

        for trial = 1:size(rec_data, 2)
            H_trial = rec_data{1, trial};  % Get first rank (typically only one)
            if isempty(H_trial) || ~isnumeric(H_trial)
                continue;
            end

            % H_trial is [components x timepoints]
            % Max across components per timepoint (collapsed H)
            H_collapsed = max(H_trial, [], 1);
            rec_collapsed = [rec_collapsed; H_collapsed(:)'];
        end

        if ~isempty(rec_collapsed)
            all_collapsed = [all_collapsed; rec_collapsed];

            % Compute per-recording autocorrelation
            rec_vec = rec_collapsed(:);
            if length(rec_vec) > maxLag * 2
                [acf_rec, ~] = xcorr(rec_vec - mean(rec_vec), maxLag, 'normalized');
                acf_rec = acf_rec(maxLag+1:end);  % Positive lags only
                lags = 0:maxLag;
                [tau_rec, ~] = fitExponentialDecay(acf_rec, lags, fitRange);
                per_rec_tau = [per_rec_tau; tau_rec];
            end
        end
    end

    % Compute aggregate autocorrelation for this condition
    if ~isempty(all_collapsed)
        collapsed_vec = all_collapsed(:);
        fprintf('  Total datapoints: %d\n', length(collapsed_vec));

        [acf, lags] = xcorr(collapsed_vec - mean(collapsed_vec), maxLag, 'normalized');
        acf = acf(maxLag+1:end);  % Positive lags only
        lags = 0:maxLag;

        % Fit exponential: acf = exp(-lag/tau)
        [tau, fitResult] = fitExponentialDecay(acf, lags, fitRange);

        % Store results
        AutocorrProfiles.(condition).acf = acf;
        AutocorrProfiles.(condition).lags = lags;
        AutocorrProfiles.(condition).tau = tau;
        AutocorrProfiles.(condition).fitResult = fitResult;
        AutocorrProfiles.(condition).nDatapoints = length(collapsed_vec);
        AutocorrProfiles.(condition).per_rec_tau = per_rec_tau;
        AutocorrProfiles.(condition).mean_per_rec_tau = mean(per_rec_tau, 'omitnan');
        AutocorrProfiles.(condition).std_per_rec_tau = std(per_rec_tau, 'omitnan');

        fprintf('  Aggregate tau: %.2f (R²=%.3f)\n', tau, fitResult.R2);
        fprintf('  Per-recording tau: %.2f ± %.2f (n=%d)\n', ...
            AutocorrProfiles.(condition).mean_per_rec_tau, ...
            AutocorrProfiles.(condition).std_per_rec_tau, ...
            length(per_rec_tau));
    else
        fprintf('  No valid data found\n');
    end
end

%% Save results
% Create output directory if needed
outputDir = fileparts(outputPath);
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
    fprintf('\nCreated output directory: %s\n', outputDir);
end

save(outputPath, 'AutocorrProfiles', '-v7.3');
fprintf('\n=== Saved autocorrelation profiles to: %s ===\n', outputPath);

%% Visualization
figure('Name', 'H Component Autocorrelation Profiles');

nConds = length(conditions);
colors = struct();
colors.Naive = [0.3373, 0.7059, 0.9137];     % Light blue
colors.Beginner = [0.8431, 0.2549, 0.6078];  % Magenta/pink
colors.Expert = [0, 0.6196, 0.4510];         % Teal/green
colors.NoSpout = [0.8353, 0.3686, 0];        % Orange

subplot(1, 2, 1);
hold on;
legendEntries = {};
for c = 1:nConds
    condition = conditions{c};
    if isfield(AutocorrProfiles, condition) && isfield(AutocorrProfiles.(condition), 'acf')
        plot(AutocorrProfiles.(condition).lags, AutocorrProfiles.(condition).acf, ...
            'LineWidth', 2, 'Color', colors.(condition));
        legendEntries{end+1} = sprintf('%s (τ=%.1f)', condition, AutocorrProfiles.(condition).tau);
    end
end
hold off;
xlabel('Lag (frames)');
ylabel('Autocorrelation');
title('Collapsed H Autocorrelation');
legend(legendEntries, 'Location', 'best');
xlim([0 maxLag]);

subplot(1, 2, 2);
tau_vals = zeros(nConds, 1);
condition_names = {};
color_array = [];
for c = 1:nConds
    condition = conditions{c};
    if isfield(AutocorrProfiles, condition) && isfield(AutocorrProfiles.(condition), 'tau')
        tau_vals(c) = AutocorrProfiles.(condition).tau;
        condition_names{c} = condition;
        color_array = [color_array; colors.(condition)];
    else
        tau_vals(c) = NaN;
        condition_names{c} = condition;
        color_array = [color_array; [0.5 0.5 0.5]];
    end
end

b = bar(tau_vals);
b.FaceColor = 'flat';
for i = 1:length(tau_vals)
    b.CData(i,:) = color_array(i,:);
end
set(gca, 'XTickLabel', condition_names);
xtickangle(45);
ylabel('Time Constant τ (frames)');
title('Decay Time Constants');

sgtitle('H Component Autocorrelation Analysis');

fprintf('\n=== Analysis Complete ===\n');

%% Helper Function
function [tau, fitResult] = fitExponentialDecay(acf, lags, fitRange)
% FITEXPONENTIALDECAY Fit simple exponential decay to autocorrelation
%
% Model: acf = exp(-lag / tau)
% Uses log-linear regression for robustness

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
        fitResult = struct('R2', NaN, 'method', 'failed');
        return;
    end

    log_acf = log(acf_fit(valid));
    lags_valid = lags_fit(valid);

    % Linear regression: log(acf) = -lag/tau --> slope = -1/tau
    p = polyfit(lags_valid, log_acf, 1);
    tau = -1 / p(1);

    % Handle negative tau (if autocorrelation increases)
    if tau < 0
        tau = NaN;
        fitResult = struct('R2', NaN, 'method', 'negative_slope');
        return;
    end

    % Compute R²
    predicted = polyval(p, lags_valid);
    SS_res = sum((log_acf - predicted).^2);
    SS_tot = sum((log_acf - mean(log_acf)).^2);
    R2 = 1 - SS_res / SS_tot;

    fitResult = struct('R2', R2, 'slope', p(1), 'intercept', p(2), 'method', 'log-linear');
end
