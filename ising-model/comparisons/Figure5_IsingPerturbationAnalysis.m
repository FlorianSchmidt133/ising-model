%% Figure5_IsingPerturbationAnalysis.m
% Analyze Ising model perturbation experiments
%
% This script analyzes results from run_ising_perturbations.py to address:
%   1. Region Growth: How much does activity spread from stimulus?
%   2. Persistence: How long does activity persist after stimulus offset?
%   3. Size-Dependent Gating: Is there a threshold size for propagation?
%
% Stimulus Modes:
%   - Clamped: Region held at +1 throughout stimulus duration
%   - Double Pulse: Set to +1 at onset AND offset (mimics SC behavior)
%
% Usage:
%   Run this script after running run_ising_perturbations.py

%% SECTION 1: Configuration

% Allow external override of config (for SLURM/command-line usage)
if ~exist('config', 'var')
    config = struct();
end
if ~isfield(config, 'figuresOnly')
    config.figuresOnly = false;
end
if ~isfield(config, 'stimMode')
    config.stimMode = 'clamped';
end
% Map stimMode to struct field name (underscores stripped in field names).
% For bias-encoded modes ('bias_2p00') strip the bias suffix so storage
% lookups (Data.<cond>.<stimModeKey>, PropDynamics.<cond>.<stimModeKey>,
% etc.) hit the raw modeKey populated by per-raw-mode loops below.
stimModeKey = strrep(rawModeFromDisplayMode(config.stimMode), '_', '');

% --- Paths (use existing values if set, otherwise use defaults) ---
if ~isfield(config, 'perturbationResultsPath')
    config.perturbationResultsPath = mba_p('IsingModelData_39x78_100K\IsingPerturbations');
end
if ~isfield(config, 'outputPath')
    config.outputPath = 'Fig. 5 Model\PerturbationAnalysis';
end
% Per-bias-value runs get their own subfolder so per-duration outputs from
% different bias values don't overwrite each other. Skip if the stim mode
% already appears anywhere along outputPath (avoids nested duplicates when
% run_ising_pipeline.sh has already inserted '/<stimMode>/' upstream).
if isBiasEncodedMode(config.stimMode)
    pathSegs = strsplit(config.outputPath, filesep);
    if ~any(strcmp(pathSegs, config.stimMode))
        config.outputPath = fullfile(config.outputPath, config.stimMode);
    end
end

% --- Find most recent results file (skip in figuresOnly mode) ---
% Prefer per-mode file: PerturbationResults_<rawMode>_*.mat
% Fall back to legacy monolithic: PerturbationResults_<YYYYMMDD>_<HHMMSS>.mat
if ~config.figuresOnly
    if ~isfield(config, 'resultsFile') || isempty(config.resultsFile)
        rawMode = rawModeFromDisplayMode(config.stimMode);
        perModePattern = sprintf('PerturbationResults_%s_*.mat', rawMode);
        resultFiles = dir(fullfile(config.perturbationResultsPath, perModePattern));

        if isempty(resultFiles)
            % Fallback: legacy monolithic file (PerturbationResults_<timestamp>.mat,
            % no mode infix). Use a strict regex to avoid matching per-mode files
            % for OTHER modes that would also satisfy the bare 'PerturbationResults_*.mat'
            % glob.
            allFiles = dir(fullfile(config.perturbationResultsPath, 'PerturbationResults_*.mat'));
            isLegacy = ~cellfun(@isempty, regexp({allFiles.name}, ...
                '^PerturbationResults_\d{8}_\d{6}\.mat$', 'once'));
            resultFiles = allFiles(isLegacy);
            if ~isempty(resultFiles)
                fprintf('  No per-mode file for ''%s'' — falling back to legacy monolithic.\n', rawMode);
            end
        end

        if isempty(resultFiles)
            error('No perturbation results found in %s for mode ''%s''', ...
                config.perturbationResultsPath, rawMode);
        end
        [~, idx] = max([resultFiles.datenum]);
        config.resultsFile = fullfile(resultFiles(idx).folder, resultFiles(idx).name);
    end
    fprintf('Loading results from: %s\n', config.resultsFile);
end

% --- Colors (match Figure5_IsingComparison.m) ---
config.colors.Naive = [0.3373, 0.7059, 0.9137];     % Light blue
config.colors.Beginner = [0.8431, 0.2549, 0.6078]; % Magenta
config.colors.Expert = [0, 0.6196, 0.4510];        % Teal
config.colors.NoSpout = [0.8353, 0.3686, 0];       % Orange

% --- Conditions ---
config.conditions = {'Naive', 'Beginner', 'Expert', 'NoSpout'};

% --- Time Display Settings ---
config.showRealTime = true;  % true = show time in seconds, false = show in frames (default)
config.samplingRate = 10;     % Hz (experimental imaging rate)

% --- EC50 area duration views (used by figures placed in Figure 6) ---
if ~isfield(config, 'targetDurationSec')
    config.targetDurationSec = 2.0;
end
if ~isfield(config, 'collapseTargetDurationsSec')
    config.collapseTargetDurationsSec = [0.5 1.0 2.0 5.0 10.0];
end

% Create output directory
if ~exist(config.outputPath, 'dir')
    mkdir(config.outputPath);
end

if ~config.figuresOnly
%% SECTION 2: Load Perturbation Results
fprintf('\n--- Loading Perturbation Results ---\n');

% Detect file format: try load() first (MATLAB v7.3 .mat via hdf5storage),
% fall back to direct HDF5 reading (h5py-written streaming format).
config.useHDF5Direct = false;
try
    Results = load(config.resultsFile);
catch ME
    fprintf('  load() failed (%s), using direct HDF5 reading.\n', ME.identifier);
    config.useHDF5Direct = true;
    Results = struct();
    Results.pre_stim_frames     = h5read(config.resultsFile, '/pre_stim_frames');
    Results.post_stim_frames    = h5read(config.resultsFile, '/post_stim_frames');
    Results.stimulus_durations  = h5read(config.resultsFile, '/stimulus_durations');
    Results.stimulus_sizes      = h5read(config.resultsFile, '/stimulus_sizes');
    Results.n_top_matches       = h5read(config.resultsFile, '/n_top_matches');
    Results.n_replicates        = h5read(config.resultsFile, '/n_replicates');
    Results.grid_size           = h5read(config.resultsFile, '/grid_size');
    Results.global_mean_sf      = h5read(config.resultsFile, '/global_mean_sf');
    Results.stimulus_modes      = h5read(config.resultsFile, '/stimulus_modes');
    Results.conditions          = h5read(config.resultsFile, '/conditions');
    try
        Results.stimulus_bias_values = h5read(config.resultsFile, '/stimulus_bias_values');
    catch
        Results.stimulus_bias_values = [];
    end
end

% Extract configuration
preStimFrames = double(Results.pre_stim_frames);
stimulusDurations = double(Results.stimulus_durations(:)');  % Multiple durations now
postStimFrames = double(Results.post_stim_frames);

stimulusSizes = double(Results.stimulus_sizes(:)');
% Handle HDF5 string arrays (stored as column vectors in HDF5)
if iscell(Results.stimulus_modes)
    % HDF5 stores each string as a column vector, transpose to row
    stimulusModes = cellfun(@(x) char(x(:)'), Results.stimulus_modes, 'UniformOutput', false);
elseif ischar(Results.stimulus_modes)
    stimulusModes = cellstr(Results.stimulus_modes);
elseif isstring(Results.stimulus_modes)
    stimulusModes = cellstr(Results.stimulus_modes);
else
    stimulusModes = cellstr(string(Results.stimulus_modes));
end
% Filter to only the requested stim mode (avoids dimension mismatches
% when different modes have different replicate/frame counts). For
% bias-encoded modes ('bias_2p00') filter on the raw mode — the bias
% value is selected via slicing in SECTION 3.
stimulusModes = stimulusModes(ismember(stimulusModes, {rawModeFromDisplayMode(config.stimMode)}));

if iscell(Results.conditions)
    conditions = cellfun(@(x) char(x(:)'), Results.conditions, 'UniformOutput', false);
elseif ischar(Results.conditions)
    conditions = cellstr(Results.conditions);
elseif isstring(Results.conditions)
    conditions = cellstr(Results.conditions);
else
    conditions = cellstr(string(Results.conditions));
end
nTopMatches = double(Results.n_top_matches);
nReplicates = double(Results.n_replicates);
gridSize = double(Results.grid_size);

% New: Extract stimulus bias values for bias mode
if isfield(Results, 'stimulus_bias_values') && ~isempty(Results.stimulus_bias_values)
    stimulusBiasValues = double(Results.stimulus_bias_values(:)');
    nBiasValues = length(stimulusBiasValues);
else
    stimulusBiasValues = [];
    nBiasValues = 0;
end

fprintf('Grid size: %d x %d\n', gridSize(1), gridSize(2));
fprintf('Stimulus durations: %s frames\n', mat2str(stimulusDurations));
fprintf('Stimulus sizes: %s\n', mat2str(stimulusSizes));
fprintf('Stimulus modes: %s\n', strjoin(stimulusModes, ', '));
if nBiasValues > 0
    fprintf('Stimulus bias values (bias mode): %s\n', mat2str(stimulusBiasValues));

    % Find index for low bias value (0.25)
    lowBiasValue = 0.25;
    [~, lowBiasIdx] = min(abs(stimulusBiasValues - lowBiasValue));
    fprintf('Low bias index: %d (value=%.2f)\n', lowBiasIdx, stimulusBiasValues(lowBiasIdx));
else
    lowBiasIdx = 1;  % Default if no bias values
end
fprintf('Replicates (header): %d\n', nReplicates);

% --- Load temporal scale factors from comparison results ---
if ~isfield(config, 'comparisonResultsPath')
    config.comparisonResultsPath = mba_p('IsingModelData_39x78_100K\IsingComparison');
end

compFiles = dir(fullfile(config.comparisonResultsPath, 'IsingComparison_Results_*.mat'));
if ~isempty(compFiles)
    [~, cidx] = max([compFiles.datenum]);
    compFile = fullfile(compFiles(cidx).folder, compFiles(cidx).name);
    fprintf('Loading temporal scale factors from: %s\n', compFile);

    config.temporalScaleFactors = struct();
    for c = 1:length(config.conditions)
        cond = config.conditions{c};
        try
            bestIdx = h5read(compFile, sprintf('/Comparison/%s/bestMatch_idx', cond));
            bestIdx = double(bestIdx(:)') + 1;  % Python 0-indexed → MATLAB 1-indexed
            allSF = h5read(compFile, sprintf('/Comparison/%s/temporal_scale_factors', cond));
            allSF = double(allSF(:)');
            config.temporalScaleFactors.(cond) = allSF(bestIdx(1:min(nTopMatches, length(bestIdx))));
            fprintf('  %s: mean temporal_scale_factor = %.4f (range %.2f-%.2f)\n', ...
                cond, mean(config.temporalScaleFactors.(cond)), ...
                min(config.temporalScaleFactors.(cond)), max(config.temporalScaleFactors.(cond)));
        catch ME
            fprintf('  Warning: Could not load scale factors for %s: %s\n', cond, ME.message);
            config.temporalScaleFactors.(cond) = ones(1, nTopMatches);  % Default: no scaling
        end
    end

    % Compute global mean scale factor for time axis
    allSFs = [];
    for c = 1:length(config.conditions)
        cond = config.conditions{c};
        allSFs = [allSFs, config.temporalScaleFactors.(cond)];
    end
    config.globalMeanSF = mean(allSFs);
    fprintf('  Global mean temporal scale factor: %.4f\n', config.globalMeanSF);

    % Load experimental decay taus for comparison
    config.expTau = struct();
    for c = 1:length(config.conditions)
        cond = config.conditions{c};
        try
            expTau = h5read(compFile, sprintf('/DynamicsAnalysis/%s/exp_tau', cond));
            config.expTau.(cond) = double(expTau);
            fprintf('  %s: exp_tau = %.4f\n', cond, config.expTau.(cond));
        catch
            config.expTau.(cond) = NaN;
        end
    end
else
    fprintf('Warning: No comparison results found, using default temporal scaling\n');
    config.globalMeanSF = 1;
    config.temporalScaleFactors = struct();
    config.expTau = struct();
    for c = 1:length(config.conditions)
        cond = config.conditions{c};
        config.temporalScaleFactors.(cond) = ones(1, 10);
        config.expTau.(cond) = NaN;
    end
end

% Create display modes - bias-using modes split into 'high_<mode>' / 'low_<mode>' pairs.
% Bias-encoded stimMode (e.g. 'bias_2p00') pins a single bias value: after
% per-invocation slicing further down, the raw mode's data is collapsed to 4D
% and we iterate the displayMode loop on the raw mode key.
metricsDisplayModes = {};
for mm = 1:length(stimulusModes)
    metricsDisplayModes = [metricsDisplayModes, displayModesForMode(stimulusModes{mm})];
end

% Filter to selected stimulus mode
if isBiasEncodedMode(config.stimMode)
    metricsDisplayModes = {rawModeFromDisplayMode(config.stimMode)};
elseif isBiasMode(config.stimMode)
    metricsDisplayModes = metricsDisplayModes(ismember(metricsDisplayModes, displayModesForMode(config.stimMode)));
else
    metricsDisplayModes = metricsDisplayModes(ismember(metricsDisplayModes, {config.stimMode}));
end
nDisplayModes = length(metricsDisplayModes);

% Initialize storage for all durations
% NOTE: AllDurationData is NOT accumulated here — each duration's Data is
% already saved to per-duration PerturbationAnalysis.mat files and is
% lazy-loaded via loadDurationData() when needed in cross-duration sections.
AllDurationMetrics = struct();
AllDurationGating = struct();
AllDurationStats = struct();
AllDurationMetrics_BM = struct();
AllDurationGating_BM = struct();
AllDurationStats_BM = struct();

%% MAIN LOOP: Analyze Each Duration
for durIdx = 1:length(stimulusDurations)
    currentDuration = stimulusDurations(durIdx);
    durKey = sprintf('dur_%d', currentDuration);

    fprintf('\n========================================\n');
    fprintf('=== Analyzing Duration: %d frames ===\n', currentDuration);
    fprintf('========================================\n');

    % Create output subfolder for this duration
    durOutputPath = fullfile(config.outputPath, durKey);
    if ~exist(durOutputPath, 'dir')
        mkdir(durOutputPath);
    end

%% SECTION 3: Extract and Organize Data
fprintf('\n--- Extracting Data ---\n');

% For HDF5 direct mode, build Results.experiments struct for current duration
% by reading only the needed groups from disk (avoids loading all durations)
if config.useHDF5Direct
    Results.experiments = loadExperimentsForDuration(config.resultsFile, ...
        conditions, nTopMatches, stimulusModes, stimulusSizes, ...
        currentDuration, stimulusBiasValues);
end

% Initialize storage structures
Data = struct();

% Compute total frames for current duration
stimOnFrames = currentDuration;
totalFrames = preStimFrames + stimOnFrames + postStimFrames;

% Infer actual replicate count for this duration from data
durNReplicates = nReplicates;
try
    probeCond = conditions{1};
    probeSimKey = 'sim_0';
    probeSizeKey = sprintf('size_%d', stimulusSizes(1));
    probeDurKey = sprintf('dur_%d', currentDuration);
    probeDurData = Results.experiments.(probeCond).(probeSimKey).(config.stimMode).(probeSizeKey).(probeDurKey);
    if isstruct(probeDurData) && isfield(probeDurData, 'activity')
        durNReplicates = size(probeDurData.activity, 1);
    end
catch ME
    fprintf('  Probe failed for dur_%d: %s\n', currentDuration, ME.message);
end
if durNReplicates ~= nReplicates
    fprintf('  Duration %d: nReplicates=%d (header=%d)\n', currentDuration, durNReplicates, nReplicates);
end

for c = 1:length(conditions)
    condition = conditions{c};

    if ~isfield(Results.experiments, condition)
        fprintf('  %s: Not found in results\n', condition);
        continue;
    end

    condData = Results.experiments.(condition);

    Data.(condition) = struct();
    Data.(condition).nSims = nTopMatches;
    Data.(condition).stimulusSizes = stimulusSizes;
    Data.(condition).stimulusDurations = stimulusDurations;

    % Pre-allocate arrays for each mode (using default duration)
    for m = 1:length(stimulusModes)
        mode = stimulusModes{m};
        modeKey = strrep(mode, '_', '');  % 'double_pulse' -> 'doublepulse'

        nSizes = length(stimulusSizes);

        % For bias mode, add extra dimension for bias values
        if isBiasMode(mode) && nBiasValues > 0
            % Time series metrics [nSims x nSizes x nBiasValues x durNReplicates x nFrames]
            Data.(condition).(modeKey).activity = zeros(nTopMatches, nSizes, nBiasValues, durNReplicates, totalFrames);
            Data.(condition).(modeKey).activity_crop = zeros(nTopMatches, nSizes, nBiasValues, durNReplicates, totalFrames);
            Data.(condition).(modeKey).stim_activity = zeros(nTopMatches, nSizes, nBiasValues, durNReplicates, totalFrames);
            Data.(condition).(modeKey).stimulus_blob_area = zeros(nTopMatches, nSizes, nBiasValues, durNReplicates, totalFrames);
            Data.(condition).(modeKey).stimulus_blob_extent = zeros(nTopMatches, nSizes, nBiasValues, durNReplicates, totalFrames);
            Data.(condition).(modeKey).valid_blob_count = zeros(nTopMatches, nSizes, nBiasValues, durNReplicates, totalFrames);
            Data.(condition).(modeKey).morans_I = zeros(nTopMatches, nSizes, nBiasValues, durNReplicates, totalFrames);
            Data.(condition).(modeKey).propagation_velocity = zeros(nTopMatches, nSizes, nBiasValues, durNReplicates, totalFrames);
            Data.(condition).(modeKey).wavefront_anisotropy = zeros(nTopMatches, nSizes, nBiasValues, durNReplicates, totalFrames);

            % Derived metrics [nSims x nSizes x nBiasValues x durNReplicates]
            Data.(condition).(modeKey).time_to_max_extent = zeros(nTopMatches, nSizes, nBiasValues, durNReplicates);
            Data.(condition).(modeKey).max_propagation_velocity = zeros(nTopMatches, nSizes, nBiasValues, durNReplicates);
            Data.(condition).(modeKey).mean_anisotropy_during_stim = zeros(nTopMatches, nSizes, nBiasValues, durNReplicates);

            % Pre-stimulus summary [nSims x nSizes x nBiasValues] - struct array
            Data.(condition).(modeKey).prestim_summary = cell(nTopMatches, nSizes, nBiasValues);

            continue;  % Skip the standard initialization below
        end

        % Standard initialization for clamped and double_pulse modes
        % Time series metrics [nSims x nSizes x durNReplicates x nFrames]
        Data.(condition).(modeKey).activity = zeros(nTopMatches, nSizes, durNReplicates, totalFrames);
        Data.(condition).(modeKey).activity_crop = zeros(nTopMatches, nSizes, durNReplicates, totalFrames);
        Data.(condition).(modeKey).stim_activity = zeros(nTopMatches, nSizes, durNReplicates, totalFrames);
        Data.(condition).(modeKey).stimulus_blob_area = zeros(nTopMatches, nSizes, durNReplicates, totalFrames);
        Data.(condition).(modeKey).stimulus_blob_extent = zeros(nTopMatches, nSizes, durNReplicates, totalFrames);
        % NEW time series metrics
        Data.(condition).(modeKey).valid_blob_count = zeros(nTopMatches, nSizes, durNReplicates, totalFrames);
        Data.(condition).(modeKey).morans_I = zeros(nTopMatches, nSizes, durNReplicates, totalFrames);
        Data.(condition).(modeKey).propagation_velocity = zeros(nTopMatches, nSizes, durNReplicates, totalFrames);
        Data.(condition).(modeKey).wavefront_anisotropy = zeros(nTopMatches, nSizes, durNReplicates, totalFrames);

        % Derived metrics [nSims x nSizes x durNReplicates]
        Data.(condition).(modeKey).time_to_max_extent = zeros(nTopMatches, nSizes, durNReplicates);
        Data.(condition).(modeKey).max_propagation_velocity = zeros(nTopMatches, nSizes, durNReplicates);
        Data.(condition).(modeKey).mean_anisotropy_during_stim = zeros(nTopMatches, nSizes, durNReplicates);

        % Pre-stimulus summary [nSims x nSizes] - struct array
        Data.(condition).(modeKey).prestim_summary = cell(nTopMatches, nSizes);
    end

    % Extract data for each simulation
    for sim = 0:(nTopMatches-1)
        simKey = sprintf('sim_%d', sim);

        if ~isfield(condData, simKey)
            continue;
        end

        simData = condData.(simKey);

        for m = 1:length(stimulusModes)
            mode = stimulusModes{m};
            modeKey = strrep(mode, '_', '');

            if ~isfield(simData, mode)
                continue;
            end

            modeData = simData.(mode);

            for s = 1:length(stimulusSizes)
                sizeKey = sprintf('size_%d', stimulusSizes(s));

                if ~isfield(modeData, sizeKey)
                    continue;
                end

                sizeData = modeData.(sizeKey);

                % Get data for current duration
                durKeyData = sprintf('dur_%d', currentDuration);
                if ~isfield(sizeData, durKeyData)
                    continue;
                end

                durData = sizeData.(durKeyData);

                % Check if data is empty (missing job)
                if isempty(durData) || (isnumeric(durData) && numel(durData) == 0)
                    continue;
                end

                % Handle bias mode specially - it has extra level for bias values
                if isBiasMode(mode) && nBiasValues > 0 && isstruct(durData)
                    % Loop over bias values
                    for b = 1:nBiasValues
                        biasKey = sprintf('bias_%sp%s', ...
                            strrep(sprintf('%.0f', floor(stimulusBiasValues(b))), '-', 'm'), ...
                            sprintf('%02d', round(mod(stimulusBiasValues(b), 1) * 100)));
                        % Handle key format: 'bias_0p25', 'bias_1p00', etc.
                        biasKey = sprintf('bias_%s', strrep(sprintf('%.2f', stimulusBiasValues(b)), '.', 'p'));

                        if ~isfield(durData, biasKey)
                            continue;
                        end

                        biasData = durData.(biasKey);
                        if isempty(biasData) || (isnumeric(biasData) && numel(biasData) == 0)
                            continue;
                        end

                        % Defensive frame-count check (belt+suspenders for the
                        % combine-side filter) — legacy files with different
                        % POST_STIM_FRAMES would otherwise crash MATLAB.
                        expected_frames = preStimFrames + currentDuration + postStimFrames;
                        if isfield(biasData, 'activity')
                            src_frames = size(biasData.activity, 2);
                            if src_frames ~= expected_frames
                                warning('Skip %s sim%d size%d dur%d bias_%g: frames %d vs expected %d (legacy data)', ...
                                        modeKey, sim, s, currentDuration, stimulusBiasValues(b), src_frames, expected_frames);
                                continue;
                            end
                        end

                        % Determine usable rep count (truncate to fit slot;
                        % short sources leave trailing rows as pre-alloc zeros).
                        % Mirrors the non-bias path below.
                        nrb = 0;
                        if isfield(biasData, 'activity')
                            nrb = min(size(biasData.activity, 1), durNReplicates);
                        end

                        % Extract time series metrics [nReplicates x nFrames]
                        if isfield(biasData, 'activity') && nrb > 0
                            Data.(condition).(modeKey).activity(sim+1, s, b, 1:nrb, :) = biasData.activity(1:nrb, :);
                        end
                        if isfield(biasData, 'activity_crop') && nrb > 0
                            Data.(condition).(modeKey).activity_crop(sim+1, s, b, 1:nrb, :) = biasData.activity_crop(1:nrb, :);
                        end
                        if isfield(biasData, 'stim_activity') && nrb > 0
                            Data.(condition).(modeKey).stim_activity(sim+1, s, b, 1:nrb, :) = biasData.stim_activity(1:nrb, :);
                        end
                        if isfield(biasData, 'stimulus_blob_area') && nrb > 0
                            Data.(condition).(modeKey).stimulus_blob_area(sim+1, s, b, 1:nrb, :) = biasData.stimulus_blob_area(1:nrb, :);
                        end
                        if isfield(biasData, 'stimulus_blob_extent') && nrb > 0
                            Data.(condition).(modeKey).stimulus_blob_extent(sim+1, s, b, 1:nrb, :) = biasData.stimulus_blob_extent(1:nrb, :);
                        end
                        if isfield(biasData, 'valid_blob_count') && nrb > 0
                            Data.(condition).(modeKey).valid_blob_count(sim+1, s, b, 1:nrb, :) = biasData.valid_blob_count(1:nrb, :);
                        end
                        if isfield(biasData, 'morans_I') && nrb > 0
                            Data.(condition).(modeKey).morans_I(sim+1, s, b, 1:nrb, :) = biasData.morans_I(1:nrb, :);
                        end
                        if isfield(biasData, 'propagation_velocity') && nrb > 0
                            Data.(condition).(modeKey).propagation_velocity(sim+1, s, b, 1:nrb, :) = biasData.propagation_velocity(1:nrb, :);
                        end
                        if isfield(biasData, 'wavefront_anisotropy') && nrb > 0
                            Data.(condition).(modeKey).wavefront_anisotropy(sim+1, s, b, 1:nrb, :) = biasData.wavefront_anisotropy(1:nrb, :);
                        end

                        % Extract derived metrics [nReplicates]
                        if isfield(biasData, 'time_to_max_extent') && nrb > 0
                            Data.(condition).(modeKey).time_to_max_extent(sim+1, s, b, 1:nrb) = biasData.time_to_max_extent(1:nrb);
                        end
                        if isfield(biasData, 'max_propagation_velocity') && nrb > 0
                            Data.(condition).(modeKey).max_propagation_velocity(sim+1, s, b, 1:nrb) = biasData.max_propagation_velocity(1:nrb);
                        end
                        if isfield(biasData, 'mean_anisotropy_during_stim') && nrb > 0
                            Data.(condition).(modeKey).mean_anisotropy_during_stim(sim+1, s, b, 1:nrb) = biasData.mean_anisotropy_during_stim(1:nrb);
                        end

                        % Extract pre-stimulus summary
                        if isfield(biasData, 'prestim_summary')
                            Data.(condition).(modeKey).prestim_summary{sim+1, s, b} = biasData.prestim_summary;
                        end
                    end
                    continue;  % Skip standard extraction
                end

                % Standard extraction for clamped and double_pulse modes
                % Assign available replicates (some combos may have fewer
                % than durNReplicates; rest stays as pre-allocated zeros)
                nr = 0;
                if isfield(durData, 'activity')
                    nr = min(size(durData.activity, 1), durNReplicates);
                end

                % Defensive frame-count check (belt+suspenders for the
                % combine-side filter) — legacy files with different
                % POST_STIM_FRAMES would otherwise crash MATLAB.
                if nr > 0
                    expected_frames = preStimFrames + currentDuration + postStimFrames;
                    src_frames = size(durData.activity, 2);
                    if src_frames ~= expected_frames
                        warning('Skip %s sim%d size%d dur%d: frames %d vs expected %d (legacy data)', ...
                                modeKey, sim, s, currentDuration, src_frames, expected_frames);
                        continue;
                    end
                end

                if isfield(durData, 'activity') && nr > 0
                    Data.(condition).(modeKey).activity(sim+1, s, 1:nr, :) = durData.activity(1:nr, :);
                end
                if isfield(durData, 'activity_crop') && nr > 0
                    Data.(condition).(modeKey).activity_crop(sim+1, s, 1:nr, :) = durData.activity_crop(1:nr, :);
                end
                if isfield(durData, 'stim_activity') && nr > 0
                    Data.(condition).(modeKey).stim_activity(sim+1, s, 1:nr, :) = durData.stim_activity(1:nr, :);
                end
                if isfield(durData, 'stimulus_blob_area') && nr > 0
                    Data.(condition).(modeKey).stimulus_blob_area(sim+1, s, 1:nr, :) = durData.stimulus_blob_area(1:nr, :);
                end
                if isfield(durData, 'stimulus_blob_extent') && nr > 0
                    Data.(condition).(modeKey).stimulus_blob_extent(sim+1, s, 1:nr, :) = durData.stimulus_blob_extent(1:nr, :);
                end
                % NEW time series metrics
                if isfield(durData, 'valid_blob_count') && nr > 0
                    Data.(condition).(modeKey).valid_blob_count(sim+1, s, 1:nr, :) = durData.valid_blob_count(1:nr, :);
                end
                if isfield(durData, 'morans_I') && nr > 0
                    Data.(condition).(modeKey).morans_I(sim+1, s, 1:nr, :) = durData.morans_I(1:nr, :);
                end
                if isfield(durData, 'propagation_velocity') && nr > 0
                    Data.(condition).(modeKey).propagation_velocity(sim+1, s, 1:nr, :) = durData.propagation_velocity(1:nr, :);
                end
                if isfield(durData, 'wavefront_anisotropy') && nr > 0
                    Data.(condition).(modeKey).wavefront_anisotropy(sim+1, s, 1:nr, :) = durData.wavefront_anisotropy(1:nr, :);
                end

                % Extract derived metrics [nReplicates]
                if isfield(durData, 'time_to_max_extent') && nr > 0
                    Data.(condition).(modeKey).time_to_max_extent(sim+1, s, 1:nr) = durData.time_to_max_extent(1:nr);
                end
                if isfield(durData, 'max_propagation_velocity') && nr > 0
                    Data.(condition).(modeKey).max_propagation_velocity(sim+1, s, 1:nr) = durData.max_propagation_velocity(1:nr);
                end
                if isfield(durData, 'mean_anisotropy_during_stim') && nr > 0
                    Data.(condition).(modeKey).mean_anisotropy_during_stim(sim+1, s, 1:nr) = durData.mean_anisotropy_during_stim(1:nr);
                end

                % Extract pre-stimulus summary
                if isfield(durData, 'prestim_summary')
                    Data.(condition).(modeKey).prestim_summary{sim+1, s} = durData.prestim_summary;
                end
            end
        end
    end

    fprintf('  %s: Extracted data for %d simulations\n', condition, nTopMatches);
end

% Create high<mode>/low<mode> Data fields by slicing the bias-mode 5D arrays.
% Applies to 'bias' and every 'double_pulse_bias[N]' variant — gives each
% bias-using mode a (high, low) pair of 4D-shaped fields with the same
% layout as clamped/doublepulse, so downstream code can index them uniformly.
%
% When config.stimMode is bias-encoded (e.g. 'bias_2p00'), the active
% bias index is also pre-sliced into its own 4D field keyed by the raw
% modeKey (overwriting the 5D field). This collapses the bias dimension
% to one value for this run, so all stimModeKey-based figure sites
% downstream see 4D data without ndims==5 branches.
stimRawMode = rawModeFromDisplayMode(config.stimMode);
if isBiasEncodedMode(config.stimMode)
    stimBiasIdx = biasIdxFromMode(config.stimMode, stimulusBiasValues);
    if stimBiasIdx <= 0
        error(['Bias-encoded stimMode ''%s'' references a value not in the ' ...
            'results file''s stimulus_bias_values [%s]. Re-run perturbations ' ...
            'with the desired bias value or pick a different stimMode.'], ...
            config.stimMode, mat2str(stimulusBiasValues));
    end
else
    stimBiasIdx = -1;
end

for cSlice = 1:length(conditions)
    condSlice = conditions{cSlice};
    if ~isfield(Data, condSlice)
        continue;
    end
    for mSlice = 1:length(stimulusModes)
        mode = stimulusModes{mSlice};
        if ~isBiasMode(mode)
            continue;
        end
        modeKey = strrep(mode, '_', '');  % 'double_pulse_bias10' -> 'doublepulsebias10'
        if ~isfield(Data.(condSlice), modeKey)
            continue;
        end
        highKey = ['high' modeKey];
        lowKey  = ['low'  modeKey];
        biasFields = fieldnames(Data.(condSlice).(modeKey));
        for fSlice = 1:length(biasFields)
            val = Data.(condSlice).(modeKey).(biasFields{fSlice});
            if isnumeric(val) && ndims(val) == 5
                Data.(condSlice).(highKey).(biasFields{fSlice}) = squeeze(val(:, :, end, :, :));
                Data.(condSlice).(lowKey).(biasFields{fSlice})  = squeeze(val(:, :, lowBiasIdx, :, :));
            elseif isnumeric(val) && ndims(val) == 4
                Data.(condSlice).(highKey).(biasFields{fSlice}) = squeeze(val(:, :, end, :));
                Data.(condSlice).(lowKey).(biasFields{fSlice})  = squeeze(val(:, :, lowBiasIdx, :));
            elseif iscell(val) && ndims(val) == 3
                Data.(condSlice).(highKey).(biasFields{fSlice}) = val(:, :, end);
                Data.(condSlice).(lowKey).(biasFields{fSlice})  = val(:, :, lowBiasIdx);
            else
                Data.(condSlice).(highKey).(biasFields{fSlice}) = val;
                Data.(condSlice).(lowKey).(biasFields{fSlice})  = val;
            end
        end

        % Per-invocation collapse: when the run pins a single bias value,
        % overwrite the raw modeKey field with the sliced 4D version so that
        % every downstream consumer (per-raw-mode loops, stimModeKey-keyed
        % figure sites) sees 4D data without needing ndims==5 branches.
        if stimBiasIdx > 0 && strcmp(mode, stimRawMode)
            for fSlice = 1:length(biasFields)
                val = Data.(condSlice).(modeKey).(biasFields{fSlice});
                if isnumeric(val) && ndims(val) == 5
                    Data.(condSlice).(modeKey).(biasFields{fSlice}) = squeeze(val(:, :, stimBiasIdx, :, :));
                elseif isnumeric(val) && ndims(val) == 4
                    Data.(condSlice).(modeKey).(biasFields{fSlice}) = squeeze(val(:, :, stimBiasIdx, :));
                elseif iscell(val) && ndims(val) == 3
                    Data.(condSlice).(modeKey).(biasFields{fSlice}) = val(:, :, stimBiasIdx);
                end
            end
        end
    end
end

DataFull = Data;  % Preserve all-sims data before simPass loop

for simPass = 1:2

if simPass == 2
    %% --- Best-Match Pass: replicate sim 1 into all slots ---
    fprintf('\n=== Best Match Pass (sim 1 only) ===\n');
    Data = DataFull;  % start from full data
    for cBM = 1:length(conditions)
        condBM = conditions{cBM};
        if ~isfield(Data, condBM), continue; end
        modesBM = fieldnames(Data.(condBM));
        for mBM = 1:length(modesBM)
            modeBM = modesBM{mBM};
            if isstruct(Data.(condBM).(modeBM))
                fieldsBM = fieldnames(Data.(condBM).(modeBM));
                for fBM = 1:length(fieldsBM)
                    val = Data.(condBM).(modeBM).(fieldsBM{fBM});
                    if isnumeric(val) && size(val, 1) == nTopMatches
                        val_bm = val(1, :, :, :, :);  % select sim 1 (extra :'s are safe)
                        repDims = [nTopMatches, ones(1, max(ndims(val_bm), 2) - 1)];
                        Data.(condBM).(modeBM).(fieldsBM{fBM}) = repmat(val_bm, repDims);
                    elseif iscell(val) && size(val, 1) == nTopMatches
                        val_bm = val(1, :, :);
                        repDims = [nTopMatches, ones(1, max(ndims(val_bm), 2) - 1)];
                        Data.(condBM).(modeBM).(fieldsBM{fBM}) = repmat(val_bm, repDims);
                    end
                end
            end
        end
    end
end

%% SECTION 4: Compute Summary Metrics
fprintf('\n--- Computing Summary Metrics ---\n');

% Time vector
timeVec = (1:totalFrames) - preStimFrames;  % 0 = stimulus onset
stimOnIdx = (preStimFrames + 1):(preStimFrames + stimOnFrames);
postStimIdx = (preStimFrames + stimOnFrames + 1):totalFrames;
preStimIdx = 1:preStimFrames;

% Time conversion for plotting (frames vs real time)
% Apply temporal scale factor: each MC sweep = globalMeanSF / samplingRate seconds
if config.showRealTime
    timeScale = config.globalMeanSF / config.samplingRate;  % seconds per MC sweep
    timeVec_plot = timeVec * timeScale;
    stimOnTime_plot = stimOnFrames * timeScale;
    timeUnit = 's';
    timeLabel = 'Time (s from stim onset)';
    durationStr = sprintf('%.1f s', currentDuration * config.globalMeanSF / config.samplingRate);
    velocityScale = config.samplingRate / config.globalMeanSF;  % px/sweep to px/s
    velocityUnit = 'px/s';
else
    timeScale = 1;  % 1 frame = 1 frame (no conversion)
    timeVec_plot = timeVec;
    stimOnTime_plot = stimOnFrames;
    timeUnit = 'frames';
    timeLabel = 'Time (frames from stim onset)';
    durationStr = sprintf('%d frames', currentDuration);
    velocityScale = 1;
    velocityUnit = 'px/frame';
end

if simPass == 2
    durationStr = [durationStr ' [Best Match]'];
end

Metrics = struct();

for c = 1:length(conditions)
    condition = conditions{c};

    if ~isfield(Data, condition)
        continue;
    end

    Metrics.(condition) = struct();

    for m = 1:nDisplayModes
        displayMode = metricsDisplayModes{m};
        metricsKey = strrep(displayMode, '_', '');  % 'high_bias' -> 'highbias'

        % Map display mode to data key
        dataKey = dataKeyFromDisplayMode(displayMode);

        % Check if data exists for this mode
        if ~isfield(Data.(condition), dataKey)
            continue;
        end

        nSims = Data.(condition).nSims;
        nSizes = length(stimulusSizes);

        % Initialize metric arrays [nSims x nSizes]
        Metrics.(condition).(metricsKey).baseline_activity = zeros(nSims, nSizes);
        Metrics.(condition).(metricsKey).peak_activity = zeros(nSims, nSizes);
        Metrics.(condition).(metricsKey).amplification = zeros(nSims, nSizes);
        Metrics.(condition).(metricsKey).max_blob_extent = zeros(nSims, nSizes);
        Metrics.(condition).(metricsKey).max_blob_area = zeros(nSims, nSizes);
        Metrics.(condition).(metricsKey).half_decay_time = zeros(nSims, nSizes);
        Metrics.(condition).(metricsKey).decay_tau = zeros(nSims, nSizes);
        Metrics.(condition).(metricsKey).decay_tau_nls = zeros(nSims, nSizes);
        Metrics.(condition).(metricsKey).return_to_baseline = zeros(nSims, nSizes);
        Metrics.(condition).(metricsKey).post_stim_auc = zeros(nSims, nSizes);
        Metrics.(condition).(metricsKey).propagation_success = zeros(nSims, nSizes);

        for sim = 1:nSims
            for s = 1:nSizes
                stimSize = stimulusSizes(s);
                stimRadius = stimSize / 2;

                % Get activity time series (average across replicates)
                % Handle bias mode (5D) vs other modes (4D)
                actData = Data.(condition).(dataKey).activity;
                if isHighBiasDisplayMode(displayMode) && ndims(actData) == 5
                    activity = squeeze(actData(sim, s, end, :, :));  % Use highest bias
                elseif isLowBiasDisplayMode(displayMode) && ndims(actData) == 5
                    activity = squeeze(actData(sim, s, lowBiasIdx, :, :));  % Use low bias
                else
                    activity = squeeze(actData(sim, s, :, :));
                end
                meanActivity = mean(activity, 1);  % [1 x nFrames]

                % Get propagation time series (stimulus blob extent)
                extData = Data.(condition).(dataKey).stimulus_blob_extent;
                if isHighBiasDisplayMode(displayMode) && ndims(extData) == 5
                    blobExtent = squeeze(extData(sim, s, end, :, :));
                elseif isLowBiasDisplayMode(displayMode) && ndims(extData) == 5
                    blobExtent = squeeze(extData(sim, s, lowBiasIdx, :, :));
                else
                    blobExtent = squeeze(extData(sim, s, :, :));
                end
                meanBlobExtent = mean(blobExtent, 1);

                % Get blob area time series
                areaData = Data.(condition).(dataKey).stimulus_blob_area;
                if isHighBiasDisplayMode(displayMode) && ndims(areaData) == 5
                    blobArea = squeeze(areaData(sim, s, end, :, :));
                elseif isLowBiasDisplayMode(displayMode) && ndims(areaData) == 5
                    blobArea = squeeze(areaData(sim, s, lowBiasIdx, :, :));
                else
                    blobArea = squeeze(areaData(sim, s, :, :));
                end
                meanBlobArea = mean(blobArea, 1);

                % --- Baseline metrics ---
                baseline = mean(meanActivity(preStimIdx));
                baselineStd = std(meanActivity(preStimIdx));
                Metrics.(condition).(metricsKey).baseline_activity(sim, s) = baseline;

                % --- Peak metrics ---
                peakActivity = max(meanActivity);
                Metrics.(condition).(metricsKey).peak_activity(sim, s) = peakActivity;
                Metrics.(condition).(metricsKey).amplification(sim, s) = peakActivity / max(baseline, 0.01);

                % --- Propagation metrics (stim-on frames only) ---
                maxExtent = max(meanBlobExtent(stimOnIdx));
                maxArea = max(meanBlobArea(stimOnIdx));
                Metrics.(condition).(metricsKey).max_blob_extent(sim, s) = maxExtent;
                Metrics.(condition).(metricsKey).max_blob_area(sim, s) = maxArea;

                % Propagation success: stimulus blob spread beyond stimulus region
                propagationThreshold = stimRadius + 5;  % At least 5 pixels beyond stimulus
                Metrics.(condition).(metricsKey).propagation_success(sim, s) = maxExtent > propagationThreshold;

                % --- Persistence metrics ---
                postStimActivity = meanActivity(postStimIdx);
                stimOffsetActivity = meanActivity(preStimFrames + stimOnFrames);

                % Per-simulation temporal scale factor (seconds per MC sweep)
                if isfield(config.temporalScaleFactors, condition)
                    simSF = config.temporalScaleFactors.(condition)(min(sim, length(config.temporalScaleFactors.(condition))));
                else
                    simSF = config.globalMeanSF;
                end
                secPerSweep = simSF / config.samplingRate;

                % Half-decay time
                halfTarget = (stimOffsetActivity + baseline) / 2;
                halfIdx = find(postStimActivity < halfTarget, 1, 'first');
                if isempty(halfIdx)
                    halfDecay = postStimFrames;
                else
                    halfDecay = halfIdx;
                end
                % Store in MC sweeps (time conversion applied at plot time)
                Metrics.(condition).(metricsKey).half_decay_time(sim, s) = halfDecay;

                % Return to baseline
                baselineThreshold = baseline + baselineStd;
                returnIdx = find(postStimActivity < baselineThreshold, 1, 'first');
                if isempty(returnIdx)
                    returnTime = postStimFrames;
                else
                    returnTime = returnIdx;
                end
                Metrics.(condition).(metricsKey).return_to_baseline(sim, s) = returnTime;

                % AUC above baseline (scale by time step width)
                auc = sum(max(0, postStimActivity - baseline));
                Metrics.(condition).(metricsKey).post_stim_auc(sim, s) = auc;

                % Exponential decay fit (log-linear)
                [tau, ~] = fitExponentialDecay(postStimActivity, baseline);
                Metrics.(condition).(metricsKey).decay_tau(sim, s) = tau;

                % Exponential decay fit (nonlinear least squares)
                [tauNLS, ~] = fitExponentialDecayNLS(postStimActivity, baseline);
                Metrics.(condition).(metricsKey).decay_tau_nls(sim, s) = tauNLS;
            end
        end
    end

    fprintf('  %s: Computed metrics\n', condition);
end

% --- Temporal scaling diagnostics ---
fprintf('\n--- Temporal Scaling Diagnostics ---\n');
fprintf('Global mean scale factor: %.4f (1 MC sweep = %.4f s at %d Hz)\n', ...
    config.globalMeanSF, config.globalMeanSF / config.samplingRate, config.samplingRate);
fprintf('Implied stimulus duration: %.2f s (raw: %d sweeps)\n', ...
    currentDuration * config.globalMeanSF / config.samplingRate, currentDuration);

for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Metrics, condition), continue; end

    metricsKey = 'clamped';
    if ~isfield(Metrics.(condition), metricsKey), continue; end

    % Mean decay tau across sizes for largest size (most reliable)
    largestSizeIdx = length(stimulusSizes);
    decayTaus = Metrics.(condition).(metricsKey).decay_tau(:, largestSizeIdx);
    validTaus = decayTaus(~isnan(decayTaus) & ~isinf(decayTaus));

    if ~isempty(validTaus)
        meanTauSweeps = mean(validTaus);
        meanTauSeconds = meanTauSweeps * config.globalMeanSF / config.samplingRate;

        fprintf('  %s (size=%d, clamped):\n', condition, stimulusSizes(largestSizeIdx));
        fprintf('    Decay tau: %.1f MC sweeps = %.2f s\n', meanTauSweeps, meanTauSeconds);

        if isfield(config.expTau, condition) && ~isnan(config.expTau.(condition))
            expTauSeconds = config.expTau.(condition) / config.samplingRate;
            fprintf('    Experimental autocorr tau: %.4f s\n', expTauSeconds);
            fprintf('    Perturbation/autocorr ratio: %.1fx slower\n', meanTauSeconds / max(expTauSeconds, eps));
        end
    end
end
fprintf('\n');

%% SECTION 5: Gating Analysis
fprintf('\n--- Gating Analysis ---\n');

Gating = struct();

for c = 1:length(conditions)
    condition = conditions{c};

    if ~isfield(Metrics, condition)
        continue;
    end

    Gating.(condition) = struct();

    for m = 1:nDisplayModes
        displayMode = metricsDisplayModes{m};
        metricsKey = strrep(displayMode, '_', '');

        % Check if metrics exist for this mode
        if ~isfield(Metrics.(condition), metricsKey)
            continue;
        end

        % Get response amplitude vs size [nSims x nSizes]
        amplification = Metrics.(condition).(metricsKey).amplification;
        propagationSuccess = Metrics.(condition).(metricsKey).propagation_success;

        % Mean across simulations
        meanAmplification = mean(amplification, 1);
        meanPropSuccess = mean(propagationSuccess, 1);

        Gating.(condition).(metricsKey).mean_amplification = meanAmplification;
        Gating.(condition).(metricsKey).mean_propagation_success = meanPropSuccess;

        % Find threshold size (first size with >50% propagation success)
        thresholdSizes = zeros(size(amplification, 1), 1);
        for sim = 1:size(amplification, 1)
            successRate = propagationSuccess(sim, :);
            threshIdx = find(successRate > 0.5, 1, 'first');
            if isempty(threshIdx)
                thresholdSizes(sim) = NaN;
            else
                thresholdSizes(sim) = stimulusSizes(threshIdx);
            end
        end
        Gating.(condition).(metricsKey).threshold_sizes = thresholdSizes;
        Gating.(condition).(metricsKey).mean_threshold = nanmean(thresholdSizes);

        % Fit Hill function to dose-response curve
        [EC50, hillN, fitR2] = fitHillFunction(stimulusSizes, meanAmplification);
        Gating.(condition).(metricsKey).EC50 = EC50;
        Gating.(condition).(metricsKey).hill_coefficient = hillN;
        Gating.(condition).(metricsKey).hill_fit_R2 = fitR2;

        % Fit Hill function per seed for confidence intervals
        EC50_perSeed = zeros(size(amplification, 1), 1);
        for sim = 1:size(amplification, 1)
            [ec50_s, ~, ~] = fitHillFunction(stimulusSizes, amplification(sim, :));
            EC50_perSeed(sim) = ec50_s;
        end
        Gating.(condition).(metricsKey).EC50_perSeed = EC50_perSeed;

        fprintf('  %s (%s): EC50=%.1f, Hill n=%.2f, Mean threshold=%.1f\n', ...
            condition, strrep(displayMode, '_', ' '), EC50, hillN, Gating.(condition).(metricsKey).mean_threshold);
    end
end

%% SECTION 5B: Pre-Stimulus State Effects
fprintf('\n--- Pre-Stimulus State Effects ---\n');

PreStimEffects = struct();

for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Data, condition)
        continue;
    end

    PreStimEffects.(condition) = struct();

    for m = 1:length(stimulusModes)
        mode = stimulusModes{m};
        modeKey = strrep(mode, '_', '');

        nSims = Data.(condition).nSims;
        nSizes = length(stimulusSizes);

        % Storage for correlations
        PreStimEffects.(condition).(modeKey).baseline_vs_peak = zeros(nSims, nSizes);
        PreStimEffects.(condition).(modeKey).baseline_vs_extent = zeros(nSims, nSizes);
        PreStimEffects.(condition).(modeKey).baseline_moransI = zeros(nSims, nSizes);

        for sim = 1:nSims
            for s = 1:nSizes
                % Get pre-stim summary if available
                if ~isempty(Data.(condition).(modeKey).prestim_summary{sim, s})
                    prestim = Data.(condition).(modeKey).prestim_summary{sim, s};

                    % Store baseline Moran's I
                    if isfield(prestim, 'mean_morans_I')
                        PreStimEffects.(condition).(modeKey).baseline_moransI(sim, s) = prestim.mean_morans_I;
                    end
                end

                % Get response metrics across replicates
                peakActivity = squeeze(max(Data.(condition).(modeKey).activity(sim, s, :, :), [], 4));
                maxExtent = squeeze(max(Data.(condition).(modeKey).stimulus_blob_extent(sim, s, :, :), [], 4));

                % Get baseline activity per replicate (mean of pre-stim period)
                preStimActivity = squeeze(mean(Data.(condition).(modeKey).activity(sim, s, :, preStimIdx), 4));

                % Correlate baseline with response (across replicates)
                if numel(preStimActivity) > 2 && numel(peakActivity) > 2
                    r_peak = corr(preStimActivity(:), peakActivity(:), 'Type', 'Spearman');
                    r_extent = corr(preStimActivity(:), maxExtent(:), 'Type', 'Spearman');
                    PreStimEffects.(condition).(modeKey).baseline_vs_peak(sim, s) = r_peak;
                    PreStimEffects.(condition).(modeKey).baseline_vs_extent(sim, s) = r_extent;
                end
            end
        end
    end

    fprintf('  %s: Pre-stim effects computed\n', condition);
end

%% SECTION 5C: Propagation Dynamics
fprintf('\n--- Propagation Dynamics ---\n');

PropDynamics = struct();

for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Data, condition)
        continue;
    end

    PropDynamics.(condition) = struct();

    for m = 1:length(stimulusModes)
        mode = stimulusModes{m};
        modeKey = strrep(mode, '_', '');

        nSims = Data.(condition).nSims;
        nSizes = length(stimulusSizes);

        % Storage for propagation metrics [nSims x nSizes]
        PropDynamics.(condition).(modeKey).max_velocity = zeros(nSims, nSizes);
        PropDynamics.(condition).(modeKey).mean_velocity = zeros(nSims, nSizes);
        PropDynamics.(condition).(modeKey).time_to_max = zeros(nSims, nSizes);
        PropDynamics.(condition).(modeKey).mean_anisotropy = zeros(nSims, nSizes);

        for sim = 1:nSims
            for s = 1:nSizes
                % Get velocity time series [nReplicates x nFrames]
                % Handle bias mode (5D) vs other modes (4D)
                velData = Data.(condition).(modeKey).propagation_velocity;
                if isBiasMode(mode) && ndims(velData) == 5
                    velocity = squeeze(velData(sim, s, end, :, :));
                else
                    velocity = squeeze(velData(sim, s, :, :));
                end
                meanVelocity = mean(velocity, 1);

                % During stimulus period
                PropDynamics.(condition).(modeKey).max_velocity(sim, s) = max(meanVelocity(stimOnIdx));
                PropDynamics.(condition).(modeKey).mean_velocity(sim, s) = mean(meanVelocity(stimOnIdx));

                % Time to max (average across replicates)
                ttmData = Data.(condition).(modeKey).time_to_max_extent;
                if isBiasMode(mode) && ndims(ttmData) == 4
                    ttm = squeeze(ttmData(sim, s, end, :));
                else
                    ttm = squeeze(ttmData(sim, s, :));
                end
                PropDynamics.(condition).(modeKey).time_to_max(sim, s) = mean(ttm);

                % Anisotropy
                anisoData = Data.(condition).(modeKey).wavefront_anisotropy;
                if isBiasMode(mode) && ndims(anisoData) == 5
                    aniso = squeeze(anisoData(sim, s, end, :, :));
                else
                    aniso = squeeze(anisoData(sim, s, :, :));
                end
                meanAniso = mean(aniso, 1);
                PropDynamics.(condition).(modeKey).mean_anisotropy(sim, s) = mean(meanAniso(stimOnIdx));
            end
        end
    end

    fprintf('  %s: Propagation dynamics computed\n', condition);
end

%% SECTION 5D: Blob Interactions
fprintf('\n--- Blob Interaction Analysis ---\n');

BlobInteractions = struct();

for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Data, condition)
        continue;
    end

    BlobInteractions.(condition) = struct();

    for m = 1:length(stimulusModes)
        mode = stimulusModes{m};
        modeKey = strrep(mode, '_', '');

        nSims = Data.(condition).nSims;
        nSizes = length(stimulusSizes);

        % Storage
        BlobInteractions.(condition).(modeKey).prestim_blob_count = zeros(nSims, nSizes);
        BlobInteractions.(condition).(modeKey).stim_blob_count = zeros(nSims, nSizes);
        BlobInteractions.(condition).(modeKey).blob_count_change = zeros(nSims, nSizes);

        for sim = 1:nSims
            for s = 1:nSizes
                % Get blob count time series
                % Handle bias mode (5D) vs other modes (4D)
                bcData = Data.(condition).(modeKey).valid_blob_count;
                if isBiasMode(mode) && ndims(bcData) == 5
                    blobCount = squeeze(bcData(sim, s, end, :, :));
                else
                    blobCount = squeeze(bcData(sim, s, :, :));
                end
                meanBlobCount = mean(blobCount, 1);

                % Pre-stim average
                prestimCount = mean(meanBlobCount(preStimIdx));
                BlobInteractions.(condition).(modeKey).prestim_blob_count(sim, s) = prestimCount;

                % During stim average
                stimCount = mean(meanBlobCount(stimOnIdx));
                BlobInteractions.(condition).(modeKey).stim_blob_count(sim, s) = stimCount;

                % Change
                BlobInteractions.(condition).(modeKey).blob_count_change(sim, s) = stimCount - prestimCount;
            end
        end
    end

    fprintf('  %s: Blob interactions computed\n', condition);
end

%% SECTION 5E: Moran's I Dynamics
fprintf('\n--- Morans I Dynamics Analysis ---\n');

MoransIDynamics = struct();

for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Data, condition)
        continue;
    end

    MoransIDynamics.(condition) = struct();

    for m = 1:length(stimulusModes)
        mode = stimulusModes{m};
        modeKey = strrep(mode, '_', '');

        nSims = Data.(condition).nSims;
        nSizes = length(stimulusSizes);

        % Storage
        MoransIDynamics.(condition).(modeKey).prestim_moransI = zeros(nSims, nSizes);
        MoransIDynamics.(condition).(modeKey).peak_moransI = zeros(nSims, nSizes);
        MoransIDynamics.(condition).(modeKey).moransI_increase = zeros(nSims, nSizes);
        MoransIDynamics.(condition).(modeKey).return_to_baseline_time = zeros(nSims, nSizes);

        for sim = 1:nSims
            for s = 1:nSizes
                % Get Moran's I time series
                % Handle bias mode (5D) vs other modes (4D)
                miData = Data.(condition).(modeKey).morans_I;
                if isBiasMode(mode) && ndims(miData) == 5
                    moransI = squeeze(miData(sim, s, end, :, :));
                else
                    moransI = squeeze(miData(sim, s, :, :));
                end
                meanMoransI = mean(moransI, 1);

                % Pre-stim baseline
                baselineMoransI = mean(meanMoransI(preStimIdx));
                MoransIDynamics.(condition).(modeKey).prestim_moransI(sim, s) = baselineMoransI;

                % Peak during stim+post
                peakMoransI = max(meanMoransI(stimOnIdx(1):end));
                MoransIDynamics.(condition).(modeKey).peak_moransI(sim, s) = peakMoransI;

                % Increase
                MoransIDynamics.(condition).(modeKey).moransI_increase(sim, s) = peakMoransI - baselineMoransI;

                % Return to baseline time (in post-stim period)
                baselineThreshold = baselineMoransI + 0.1 * (peakMoransI - baselineMoransI);
                postMoransI = meanMoransI(postStimIdx);
                returnIdx = find(postMoransI < baselineThreshold, 1, 'first');
                if isempty(returnIdx)
                    returnTime = length(postStimIdx);
                else
                    returnTime = returnIdx;
                end
                MoransIDynamics.(condition).(modeKey).return_to_baseline_time(sim, s) = returnTime;
            end
        end
    end

    fprintf('  %s: Morans I dynamics computed\n', condition);
end

%% SECTION 6: Statistical Comparisons
fprintf('\n--- Statistical Comparisons ---\n');

Stats = struct();

% Compare conditions for each metric and mode
metricsToCompare = {'amplification', 'decay_tau', 'decay_tau_nls', 'half_decay_time', 'max_blob_extent'};

for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};
    metricsKey = strrep(displayMode, '_', '');

    Stats.(metricsKey) = struct();

    for mi = 1:length(metricsToCompare)
        metricName = metricsToCompare{mi};
        Stats.(metricsKey).(metricName) = struct();

        % Collect data for each stimulus size
        for s = 1:length(stimulusSizes)
            stimSize = stimulusSizes(s);
            sizeKey = sprintf('size_%d', stimSize);

            % Collect values across conditions
            allValues = [];
            groupLabels = {};

            for c = 1:length(conditions)
                condition = conditions{c};
                if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKey)
                    continue;
                end

                values = Metrics.(condition).(metricsKey).(metricName)(:, s);
                values = values(~isnan(values));

                allValues = [allValues; values(:)];
                groupLabels = [groupLabels; repmat({condition}, length(values), 1)];
            end

            % Kruskal-Wallis test
            if length(unique(groupLabels)) > 1 && length(allValues) > 3
                [p, tbl, stats] = kruskalwallis(allValues, groupLabels, 'off');
                uniqueGroups = unique(groupLabels);
                Stats.(metricsKey).(metricName).(sizeKey).pValue = p;
                Stats.(metricsKey).(metricName).(sizeKey).stats = stats;
                % Pull H (chi-square approx) + df from ANOVA table
                try
                    Stats.(metricsKey).(metricName).(sizeKey).H  = tbl{2,5};
                    Stats.(metricsKey).(metricName).(sizeKey).df = tbl{2,3};
                catch
                    Stats.(metricsKey).(metricName).(sizeKey).H  = NaN;
                    Stats.(metricsKey).(metricName).(sizeKey).df = NaN;
                end

                % Pairwise Wilcoxon rank-sum + Hedges' g (always computed,
                % regardless of kruskal p — effect sizes are informative
                % even when the omnibus test is non-significant)
                pairwise = struct();
                for gi = 1:length(uniqueGroups)
                    for gj = (gi+1):length(uniqueGroups)
                        gA = uniqueGroups{gi};
                        gB = uniqueGroups{gj};
                        xA = allValues(strcmp(groupLabels, gA));
                        xB = allValues(strcmp(groupLabels, gB));
                        if numel(xA) < 2 || numel(xB) < 2, continue; end
                        [pRS, ~, statsRS] = ranksum(xA, xB);
                        pairKey = sprintf('%s_vs_%s', gA, gB);
                        pairwise.(pairKey).pValue = pRS;
                        pairwise.(pairKey).W      = statsRS.ranksum;
                        pairwise.(pairKey).g      = hedgesG(xA, xB);
                        pairwise.(pairKey).n_A    = numel(xA);
                        pairwise.(pairKey).n_B    = numel(xB);
                    end
                end
                Stats.(metricsKey).(metricName).(sizeKey).pairwise = pairwise;

                % Post-hoc if significant (MATLAB multcompare on mean ranks)
                if p < 0.05
                    posthoc = multcompare(stats, 'Display', 'off');
                    Stats.(metricsKey).(metricName).(sizeKey).posthoc = posthoc;
                end
            end
        end
    end
end

% Compare threshold sizes across conditions
for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};
    metricsKey = strrep(displayMode, '_', '');

    allThresholds = [];
    groupLabels = {};

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Gating, condition) || ~isfield(Gating.(condition), metricsKey)
            continue;
        end

        thresholds = Gating.(condition).(metricsKey).threshold_sizes;
        thresholds = thresholds(~isnan(thresholds));

        allThresholds = [allThresholds; thresholds(:)];
        groupLabels = [groupLabels; repmat({condition}, length(thresholds), 1)];
    end

    if length(unique(groupLabels)) > 1 && length(allThresholds) > 3
        [p, tbl, stats] = kruskalwallis(allThresholds, groupLabels, 'off');
        uniqueGroups = unique(groupLabels);
        Stats.(metricsKey).threshold_size.pValue = p;
        Stats.(metricsKey).threshold_size.stats  = stats;
        try
            Stats.(metricsKey).threshold_size.H  = tbl{2,5};
            Stats.(metricsKey).threshold_size.df = tbl{2,3};
        catch
            Stats.(metricsKey).threshold_size.H  = NaN;
            Stats.(metricsKey).threshold_size.df = NaN;
        end

        pairwiseT = struct();
        for gi = 1:length(uniqueGroups)
            for gj = (gi+1):length(uniqueGroups)
                gA = uniqueGroups{gi};
                gB = uniqueGroups{gj};
                xA = allThresholds(strcmp(groupLabels, gA));
                xB = allThresholds(strcmp(groupLabels, gB));
                if numel(xA) < 2 || numel(xB) < 2, continue; end
                [pRS, ~, statsRS] = ranksum(xA, xB);
                pairKey = sprintf('%s_vs_%s', gA, gB);
                pairwiseT.(pairKey).pValue = pRS;
                pairwiseT.(pairKey).W      = statsRS.ranksum;
                pairwiseT.(pairKey).g      = hedgesG(xA, xB);
                pairwiseT.(pairKey).n_A    = numel(xA);
                pairwiseT.(pairKey).n_B    = numel(xB);
            end
        end
        Stats.(metricsKey).threshold_size.pairwise = pairwiseT;

        fprintf('  Threshold size comparison (%s): H = %.3f, p = %.4f\n', ...
            displayMode, Stats.(metricsKey).threshold_size.H, p);
    end
end

%% SECTION 7: Visualization
fprintf('\n--- Creating Figures ---\n');

%% Figure 1: Activity Time Courses
fig1 = figure('Name', sprintf('Activity Time Courses (dur=%d)', currentDuration));

% Create display modes - bias-using modes split into 'high_<mode>' / 'low_<mode>' pairs.
% Bias-encoded stimMode collapses to the raw mode after per-invocation slicing.
displayModes = {};
for mm = 1:length(stimulusModes)
    displayModes = [displayModes, displayModesForMode(stimulusModes{mm})];
end

% Filter to selected stimulus mode
if isBiasEncodedMode(config.stimMode)
    displayModes = {rawModeFromDisplayMode(config.stimMode)};
elseif isBiasMode(config.stimMode)
    displayModes = displayModes(ismember(displayModes, displayModesForMode(config.stimMode)));
else
    displayModes = displayModes(ismember(displayModes, {config.stimMode}));
end
nDisplayModes = length(displayModes);

% Find index for low bias value (0.25)
lowBiasValue = 0.25;
[~, lowBiasIdx] = min(abs(stimulusBiasValues - lowBiasValue));

for m = 1:nDisplayModes
    displayMode = displayModes{m};

    % Map display mode to data key
    modeKey = dataKeyFromDisplayMode(displayMode);

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Data, condition)
            continue;
        end

        subplot(nDisplayModes, length(conditions), (m-1)*length(conditions) + c);
        hold on;

        % Color gradient for stimulus sizes
        nSizes = length(stimulusSizes);
        cmap = parula(nSizes);

        for s = 1:nSizes
            % Average across simulations and replicates
            % Handle bias mode (5D) vs other modes (4D)
            activityData = Data.(condition).(modeKey).activity;
            if isHighBiasDisplayMode(displayMode) && ndims(activityData) == 5
                % High bias: use highest bias value (most similar to clamped)
                activity = squeeze(activityData(:, s, end, :, :));  % [nSims x nReps x nFrames]
                meanActivity = squeeze(mean(mean(activity, 1), 2));
            elseif isLowBiasDisplayMode(displayMode) && ndims(activityData) == 5
                % Low bias: use bias value 0.25
                activity = squeeze(activityData(:, s, lowBiasIdx, :, :));  % [nSims x nReps x nFrames]
                meanActivity = squeeze(mean(mean(activity, 1), 2));
            else
                activity = squeeze(activityData(:, s, :, :));  % [nSims x nReps x nFrames]
                meanActivity = squeeze(mean(mean(activity, 1), 2));
            end

            % Compute SEM across simulations (collapse reps first)
            if ndims(activity) == 3
                simMeans = squeeze(mean(activity, 2));  % [nSims x nFrames]
            else
                simMeans = activity(:);  % fallback
            end
            if size(simMeans, 1) > 1
                semActivity = std(simMeans, 0, 1) / sqrt(size(simMeans, 1));
                fill([timeVec_plot, fliplr(timeVec_plot)], ...
                    [meanActivity(:)' + semActivity(:)', fliplr(meanActivity(:)' - semActivity(:)')], ...
                    cmap(s,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            end
            plot(timeVec_plot, meanActivity, 'Color', cmap(s,:), 'LineWidth', 2);
        end

        % Mark stimulus period
        xline(0, 'k--', 'LineWidth', 1);
        xline(stimOnTime_plot, 'k--', 'LineWidth', 1);

        % Shade stimulus period
        yl = ylim;
        patch([0 stimOnTime_plot stimOnTime_plot 0], [yl(1) yl(1) yl(2) yl(2)], ...
            [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

        xlabel(timeLabel);
        ylabel('Activity');
        titleMode = strrep(displayMode, '_', ' ');
        title(sprintf('%s - %s', condition, titleMode), 'Interpreter', 'none');

        if c == length(conditions) && m == 1
            % Add colorbar for stimulus sizes
            cb = colorbar;
            cb.Ticks = linspace(0, 1, 5);
            cb.TickLabels = arrayfun(@(x) sprintf('%d', x), ...
                stimulusSizes(round(linspace(1, nSizes, 5))), 'UniformOutput', false);
            ylabel(cb, 'Stimulus Size (px)');
        end
    end
end

% Synchronize y-axis limits across all subplots
allAx = findobj(fig1, 'Type', 'axes');
allAx = allAx(~arrayfun(@(a) strcmp(get(a, 'Tag'), 'Colorbar'), allAx));
if ~isempty(allAx)
    allYLim = arrayfun(@(a) get(a, 'YLim'), allAx, 'UniformOutput', false);
    allYLim = vertcat(allYLim{:});
    globalYLim = [min(allYLim(:,1)), max(allYLim(:,2))];
    set(allAx, 'YLim', globalYLim);
    for iAx = 1:length(allAx)
        patches = findobj(allAx(iAx), 'Type', 'patch');
        for p = patches'
            if length(get(p, 'YData')) == 4
                set(p, 'YData', [globalYLim(1) globalYLim(1) globalYLim(2) globalYLim(2)]);
            end
        end
    end
end

sgtitle(sprintf('Activity Time Courses (Stimulus Duration: %s)', durationStr));

%% Figure 1b: Activity Time Courses - Centre Crop (13x26 experimental FOV)
fig1b = figure('Name', sprintf('Activity Time Courses Crop (dur=%d)', currentDuration));

for m = 1:nDisplayModes
    displayMode = displayModes{m};

    % Map display mode to data key
    modeKey = dataKeyFromDisplayMode(displayMode);

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Data, condition)
            continue;
        end

        subplot(nDisplayModes, length(conditions), (m-1)*length(conditions) + c);
        hold on;

        % Color gradient for stimulus sizes
        nSizes = length(stimulusSizes);
        cmap = parula(nSizes);

        for s = 1:nSizes
            % Use activity_crop instead of activity
            if isfield(Data.(condition).(modeKey), 'activity_crop')
                activityData = Data.(condition).(modeKey).activity_crop;
            else
                % Fallback to activity if crop not available
                activityData = Data.(condition).(modeKey).activity;
            end
            if isHighBiasDisplayMode(displayMode) && ndims(activityData) == 5
                activity = squeeze(activityData(:, s, end, :, :));
                meanActivity = squeeze(mean(mean(activity, 1), 2));
            elseif isLowBiasDisplayMode(displayMode) && ndims(activityData) == 5
                activity = squeeze(activityData(:, s, lowBiasIdx, :, :));
                meanActivity = squeeze(mean(mean(activity, 1), 2));
            else
                activity = squeeze(activityData(:, s, :, :));
                meanActivity = squeeze(mean(mean(activity, 1), 2));
            end

            % Compute SEM across simulations (collapse reps first)
            if ndims(activity) == 3
                simMeans = squeeze(mean(activity, 2));  % [nSims x nFrames]
            else
                simMeans = activity(:);  % fallback
            end
            if size(simMeans, 1) > 1
                semActivity = std(simMeans, 0, 1) / sqrt(size(simMeans, 1));
                fill([timeVec_plot, fliplr(timeVec_plot)], ...
                    [meanActivity(:)' + semActivity(:)', fliplr(meanActivity(:)' - semActivity(:)')], ...
                    cmap(s,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            end
            plot(timeVec_plot, meanActivity, 'Color', cmap(s,:), 'LineWidth', 2);
        end

        % Mark stimulus period
        xline(0, 'k--', 'LineWidth', 1);
        xline(stimOnTime_plot, 'k--', 'LineWidth', 1);

        % Shade stimulus period
        yl = ylim;
        patch([0 stimOnTime_plot stimOnTime_plot 0], [yl(1) yl(1) yl(2) yl(2)], ...
            [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

        xlabel(timeLabel);
        ylabel('Activity (13\times26 crop)');
        titleMode = strrep(displayMode, '_', ' ');
        title(sprintf('%s - %s', condition, titleMode), 'Interpreter', 'none');

        if c == length(conditions) && m == 1
            cb = colorbar;
            cb.Ticks = linspace(0, 1, 5);
            cb.TickLabels = arrayfun(@(x) sprintf('%d', x), ...
                stimulusSizes(round(linspace(1, nSizes, 5))), 'UniformOutput', false);
            ylabel(cb, 'Stimulus Size (px)');
        end
    end
end

% Synchronize y-axis limits across all subplots
allAx = findobj(fig1b, 'Type', 'axes');
allAx = allAx(~arrayfun(@(a) strcmp(get(a, 'Tag'), 'Colorbar'), allAx));
if ~isempty(allAx)
    allYLim = arrayfun(@(a) get(a, 'YLim'), allAx, 'UniformOutput', false);
    allYLim = vertcat(allYLim{:});
    globalYLim = [min(allYLim(:,1)), max(allYLim(:,2))];
    set(allAx, 'YLim', globalYLim);
    for iAx = 1:length(allAx)
        patches = findobj(allAx(iAx), 'Type', 'patch');
        for p = patches'
            if length(get(p, 'YData')) == 4
                set(p, 'YData', [globalYLim(1) globalYLim(1) globalYLim(2) globalYLim(2)]);
            end
        end
    end
end

sgtitle(sprintf('Activity Time Courses - Centre Crop (Stimulus Duration: %s)', durationStr));

%% Figure 1c: Focused Clamped Activity (best-match only, 2x2 conditions)
if simPass == 2
    fig1c = figure('Name', sprintf('Clamped Activity - Best Match (dur=%d)', currentDuration));

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Data, condition) || ~isfield(Data.(condition), stimModeKey)
            continue;
        end

        subplot(2, 2, c);
        hold on;

        nSizes = length(stimulusSizes);
        cmap = parula(nSizes);

        for s = 1:nSizes
            actArr = collapseBiasPlot(Data.(condition).(stimModeKey).activity);
            activity = squeeze(actArr(:, s, :, :));
            meanActivity = squeeze(mean(mean(activity, 1), 2));

            if ndims(activity) == 3
                simMeans = squeeze(mean(activity, 2));
            else
                simMeans = activity(:);
            end
            if size(simMeans, 1) > 1
                semActivity = std(simMeans, 0, 1) / sqrt(size(simMeans, 1));
                fill([timeVec_plot, fliplr(timeVec_plot)], ...
                    [meanActivity(:)' + semActivity(:)', fliplr(meanActivity(:)' - semActivity(:)')], ...
                    cmap(s,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            end
            plot(timeVec_plot, meanActivity, 'Color', cmap(s,:), 'LineWidth', 2);
        end

        xline(0, 'k--', 'LineWidth', 1);
        xline(stimOnTime_plot, 'k--', 'LineWidth', 1);
        yl = ylim;
        patch([0 stimOnTime_plot stimOnTime_plot 0], [yl(1) yl(1) yl(2) yl(2)], ...
            [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

        xlabel(timeLabel);
        ylabel('Activity');
        title(condition);

        if c == length(conditions)
            cb = colorbar;
            cb.Ticks = linspace(0, 1, 5);
            cb.TickLabels = arrayfun(@(x) sprintf('%d', x), ...
                stimulusSizes(round(linspace(1, nSizes, 5))), 'UniformOutput', false);
            ylabel(cb, 'Stimulus Size (px)');
        end
    end

    % Synchronize y-axis limits across all subplots
    allAx = findobj(fig1c, 'Type', 'axes');
    allAx = allAx(~arrayfun(@(a) strcmp(get(a, 'Tag'), 'Colorbar'), allAx));
    if ~isempty(allAx)
        allYLim = arrayfun(@(a) get(a, 'YLim'), allAx, 'UniformOutput', false);
        allYLim = vertcat(allYLim{:});
        globalYLim = [min(allYLim(:,1)), max(allYLim(:,2))];
        set(allAx, 'YLim', globalYLim);
        for iAx = 1:length(allAx)
            patches = findobj(allAx(iAx), 'Type', 'patch');
            for p = patches'
                if length(get(p, 'YData')) == 4
                    set(p, 'YData', [globalYLim(1) globalYLim(1) globalYLim(2) globalYLim(2)]);
                end
            end
        end
    end

    sgtitle(sprintf('Clamped Activity [Best Match] (Duration: %s)', durationStr));
end

%% Figure 1d: Focused Clamped Activity - Centre Crop (best-match only, 2x2 conditions)
if simPass == 2
    fig1d = figure('Name', sprintf('Clamped Activity Crop - Best Match (dur=%d)', currentDuration));

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Data, condition) || ~isfield(Data.(condition), stimModeKey)
            continue;
        end

        subplot(2, 2, c);
        hold on;

        nSizes = length(stimulusSizes);
        cmap = parula(nSizes);

        for s = 1:nSizes
            if isfield(Data.(condition).(stimModeKey), 'activity_crop')
                activityData = collapseBiasPlot(Data.(condition).(stimModeKey).activity_crop);
            else
                activityData = collapseBiasPlot(Data.(condition).(stimModeKey).activity);
            end
            activity = squeeze(activityData(:, s, :, :));
            meanActivity = squeeze(mean(mean(activity, 1), 2));

            if ndims(activity) == 3
                simMeans = squeeze(mean(activity, 2));
            else
                simMeans = activity(:);
            end
            if size(simMeans, 1) > 1
                semActivity = std(simMeans, 0, 1) / sqrt(size(simMeans, 1));
                fill([timeVec_plot, fliplr(timeVec_plot)], ...
                    [meanActivity(:)' + semActivity(:)', fliplr(meanActivity(:)' - semActivity(:)')], ...
                    cmap(s,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            end
            plot(timeVec_plot, meanActivity, 'Color', cmap(s,:), 'LineWidth', 2);
        end

        xline(0, 'k--', 'LineWidth', 1);
        xline(stimOnTime_plot, 'k--', 'LineWidth', 1);
        yl = ylim;
        patch([0 stimOnTime_plot stimOnTime_plot 0], [yl(1) yl(1) yl(2) yl(2)], ...
            [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

        xlabel(timeLabel);
        ylabel('Activity (13\times26 crop)');
        title(condition);

        if c == length(conditions)
            cb = colorbar;
            cb.Ticks = linspace(0, 1, 5);
            cb.TickLabels = arrayfun(@(x) sprintf('%d', x), ...
                stimulusSizes(round(linspace(1, nSizes, 5))), 'UniformOutput', false);
            ylabel(cb, 'Stimulus Size (px)');
        end
    end

    % Synchronize y-axis limits across all subplots
    allAx = findobj(fig1d, 'Type', 'axes');
    allAx = allAx(~arrayfun(@(a) strcmp(get(a, 'Tag'), 'Colorbar'), allAx));
    if ~isempty(allAx)
        allYLim = arrayfun(@(a) get(a, 'YLim'), allAx, 'UniformOutput', false);
        allYLim = vertcat(allYLim{:});
        globalYLim = [min(allYLim(:,1)), max(allYLim(:,2))];
        set(allAx, 'YLim', globalYLim);
        for iAx = 1:length(allAx)
            patches = findobj(allAx(iAx), 'Type', 'patch');
            for p = patches'
                if length(get(p, 'YData')) == 4
                    set(p, 'YData', [globalYLim(1) globalYLim(1) globalYLim(2) globalYLim(2)]);
                end
            end
        end
    end

    sgtitle(sprintf('Clamped Activity - Centre Crop [Best Match] (Duration: %s)', durationStr));
end

%% Figure 2: Dose-Response Curves (Amplification vs Size)
fig2 = figure('Name', sprintf('Dose-Response Curves (dur=%d)', currentDuration));

for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};
    metricsKey = strrep(displayMode, '_', '');

    subplot(1, nDisplayModes, m);
    hold on;

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKey)
            continue;
        end

        % Get amplification [nSims x nSizes]
        amplification = Metrics.(condition).(metricsKey).amplification;

        % Mean and SEM
        meanAmp = mean(amplification, 1);
        semAmp = std(amplification, 0, 1) / sqrt(size(amplification, 1));

        % Plot with error bars - distinct markers per condition, with jitter
        color = config.colors.(condition);
        markerShapes = {'o', 's', 'd', '^'};
        jitter = (c - (length(conditions)+1)/2) * 0.3;
        errorbar(stimulusSizes + jitter, meanAmp, semAmp, [markerShapes{c} '-'], ...
            'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5, 'MarkerSize', 6);

        % Fit and plot Hill curve
        EC50 = Gating.(condition).(metricsKey).EC50;
        hillN = Gating.(condition).(metricsKey).hill_coefficient;
        if ~isnan(EC50) && ~isnan(hillN)
            xFit = linspace(min(stimulusSizes), max(stimulusSizes), 100);
            Rmax = max(meanAmp);
            yFit = Rmax * (xFit.^hillN) ./ (EC50^hillN + xFit.^hillN);
            plot(xFit, yFit, '--', 'Color', color, 'LineWidth', 1);

            % Mark EC50
            plot(EC50, Rmax/2, 'v', 'Color', color, 'MarkerSize', 8, 'MarkerFaceColor', color);
        end
    end

    xlabel('Stimulus Size (pixels)');
    ylabel('Amplification (Peak / Baseline)');
    title(sprintf('Dose-Response: %s', strrep(displayMode, '_', ' ')), 'Interpreter', 'none');
    legend(conditions, 'Location', 'southeast');
    grid on;
end

sgtitle(sprintf('Size-Dependent Response Amplification (Duration: %s)', durationStr));

%% Figure 2b: Net Propagation vs Size (Dose-Response Style)
fig2b = figure('Name', sprintf('Net Propagation vs Size (dur=%d)', currentDuration));

% Compute stimulus radii for net propagation
stimRadii = stimulusSizes / 2;

for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};
    metricsKey = strrep(displayMode, '_', '');

    subplot(1, nDisplayModes, m);
    hold on;

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKey)
            continue;
        end

        % Get max_blob_extent and compute net propagation [nSims x nSizes]
        extent = Metrics.(condition).(metricsKey).max_blob_extent;
        netProp = extent - repmat(stimRadii, size(extent, 1), 1);

        % Mean and SEM
        meanNetProp = mean(netProp, 1);
        semNetProp = std(netProp, 0, 1) / sqrt(size(netProp, 1));

        % Plot with error bars
        color = config.colors.(condition);
        errorbar(stimulusSizes, meanNetProp, semNetProp, 'o-', ...
            'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
    end

    xlabel('Stimulus Size (pixels)');
    ylabel('Net Propagation (px beyond stimulus)');
    title(sprintf('%s', strrep(displayMode, '_', ' ')), 'Interpreter', 'none');
    legend(conditions, 'Location', 'northwest');
    yline(0, 'k--', 'LineWidth', 1);  % Zero line = no spread beyond stimulus
    grid on;
end

sgtitle(sprintf('Size-Dependent Net Propagation (Duration: %s)', durationStr));

%% Figure 3: Persistence (Decay Time)
fig3 = figure('Name', sprintf('Persistence Analysis (dur=%d)', currentDuration));

for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};
    metricsKey = strrep(displayMode, '_', '');

    % Subplot 1: Half-decay time vs size
    subplot(2, nDisplayModes, m);
    hold on;

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKey)
            continue;
        end

        halfDecay = Metrics.(condition).(metricsKey).half_decay_time;
        meanHalfDecay = mean(halfDecay, 1) * timeScale;  % Scale to seconds if real time
        semHalfDecay = std(halfDecay, 0, 1) / sqrt(size(halfDecay, 1)) * timeScale;

        color = config.colors.(condition);
        errorbar(stimulusSizes, meanHalfDecay, semHalfDecay, 'o-', ...
            'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
    end

    xlabel('Stimulus Size (pixels)');
    ylabel(sprintf('Half-Decay Time (%s)', timeUnit));
    title(sprintf('Half-Decay Time: %s', strrep(displayMode, '_', ' ')), 'Interpreter', 'none');
    legend(conditions, 'Location', 'best');
    grid on;

    % Subplot 2: Decay tau vs size
    subplot(2, nDisplayModes, nDisplayModes + m);
    hold on;

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKey)
            continue;
        end

        decayTau = Metrics.(condition).(metricsKey).decay_tau;
        decayTau(isinf(decayTau)) = NaN;
        meanTau = nanmean(decayTau, 1) * timeScale;  % Scale to seconds if real time
        semTau = nanstd(decayTau, 0, 1) / sqrt(sum(~isnan(decayTau(:,1)))) * timeScale;

        color = config.colors.(condition);
        errorbar(stimulusSizes, meanTau, semTau, 'o-', ...
            'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
    end

    xlabel('Stimulus Size (pixels)');
    ylabel(sprintf('Decay \\tau (%s)', timeUnit));
    title(sprintf('Exponential Decay Time Constant: %s', strrep(displayMode, '_', ' ')), 'Interpreter', 'none');
    legend(conditions, 'Location', 'best');
    grid on;
end

sgtitle(sprintf('Persistence After Stimulus Offset (Duration: %s)', durationStr));

%% Figure 4: Gating Threshold Comparison
fig4 = figure('Name', sprintf('Gating Thresholds (dur=%d)', currentDuration));

for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};
    metricsKey = strrep(displayMode, '_', '');

    subplot(1, nDisplayModes, m);
    hold on;

    % Collect threshold data
    thresholdData = [];
    groupIdx = [];
    groupNames = {};

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Gating, condition) || ~isfield(Gating.(condition), metricsKey)
            continue;
        end

        thresholds = Gating.(condition).(metricsKey).threshold_sizes;
        thresholds = thresholds(~isnan(thresholds));

        if ~isempty(thresholds)
            thresholdData = [thresholdData; thresholds(:)];
            groupIdx = [groupIdx; repmat(c, length(thresholds), 1)];
            groupNames{c} = condition;
        end
    end

    % Box plot
    if ~isempty(thresholdData)
        boxplot(thresholdData, groupIdx, 'Labels', conditions(unique(groupIdx)));

        % Overlay individual points
        for c = unique(groupIdx)'
            idx = groupIdx == c;
            x = c + 0.1*(rand(sum(idx), 1) - 0.5);
            scatter(x, thresholdData(idx), 50, config.colors.(conditions{c}), 'filled', 'MarkerFaceAlpha', 0.6);
        end

        % Add p-value if available
        if isfield(Stats, metricsKey) && isfield(Stats.(metricsKey), 'threshold_size')
            p = Stats.(metricsKey).threshold_size.pValue;
            text(0.5, 0.95, sprintf('p = %.4f', p), 'Units', 'normalized', 'FontSize', 10);
        end
    end

    ylabel('Threshold Size (pixels)');
    title(sprintf('Propagation Threshold: %s', strrep(displayMode, '_', ' ')), 'Interpreter', 'none');

    % Annotate with EC50 values (more informative than all-1px thresholds)
    ec50Text = {};
    for ci = 1:length(conditions)
        cond = conditions{ci};
        if isfield(Gating, cond) && isfield(Gating.(cond), metricsKey)
            ec50 = Gating.(cond).(metricsKey).EC50;
            if ~isnan(ec50)
                ec50Text{end+1} = sprintf('%s EC50=%.1f', cond, ec50);
            end
        end
    end
    if ~isempty(ec50Text)
        text(0.02, 0.05, strjoin(ec50Text, ' | '), 'Units', 'normalized', ...
            'FontSize', 7, 'FontAngle', 'italic', 'VerticalAlignment', 'bottom');
    end
end

sgtitle(sprintf('Size-Dependent Gating (Duration: %s)', durationStr));

%% Figure 5: Clamped vs Double Pulse Comparison
% Only generate when both modes were computed
fig5 = [];
hasBothModes = isfield(Metrics.(conditions{1}), stimModeKey) && isfield(Metrics.(conditions{1}), 'doublepulse');
if hasBothModes
fig5 = figure('Name', sprintf('Mode Comparison (dur=%d)', currentDuration));

% Compare key metrics between modes
metricsToPlot = {'amplification', 'decay_tau', 'decay_tau_nls', 'max_blob_extent'};
metricLabels = {'Amplification', 'Decay \tau', 'Decay \tau (NLS)', 'Max Propagation'};

for mi = 1:length(metricsToPlot)
    metricName = metricsToPlot{mi};

    subplot(1, length(metricsToPlot), mi);
    hold on;

    % Use a representative stimulus size (e.g., 8 pixels)
    repSize = 8;
    sizeIdx = find(stimulusSizes == repSize, 1);
    if isempty(sizeIdx)
        sizeIdx = round(length(stimulusSizes) / 2);
    end

    xPos = 1;
    xticks_pos = [];
    xticks_labels = {};

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Metrics, condition)
            continue;
        end

        clampedVals = Metrics.(condition).(metricsKey).(metricName)(:, sizeIdx);
        pulseVals = Metrics.(condition).doublepulse.(metricName)(:, sizeIdx);

        clampedVals = clampedVals(~isnan(clampedVals) & ~isinf(clampedVals));
        pulseVals = pulseVals(~isnan(pulseVals) & ~isinf(pulseVals));

        color = config.colors.(condition);

        % Clamped
        bar(xPos, mean(clampedVals), 0.35, 'FaceColor', color, 'EdgeColor', 'k');
        errorbar(xPos, mean(clampedVals), std(clampedVals)/sqrt(length(clampedVals)), 'k', 'LineWidth', 1);

        % Double pulse
        bar(xPos + 0.4, mean(pulseVals), 0.35, 'FaceColor', color, 'FaceAlpha', 0.5, 'EdgeColor', 'k');
        errorbar(xPos + 0.4, mean(pulseVals), std(pulseVals)/sqrt(length(pulseVals)), 'k', 'LineWidth', 1);

        xticks_pos = [xticks_pos, xPos + 0.2];
        xticks_labels{end+1} = condition;

        xPos = xPos + 1.2;
    end

    set(gca, 'XTick', xticks_pos, 'XTickLabel', xticks_labels);
    ylabel(metricLabels{mi});

    % Cap y-axis for decay_tau / decay_tau_nls to avoid 10^16 display overflow
    if strcmp(metricName, 'decay_tau') || strcmp(metricName, 'decay_tau_nls')
        yl = ylim;
        allVals = findall(gca, 'Type', 'bar');
        barYData = [];
        for bh = 1:length(allVals)
            barYData = [barYData; allVals(bh).YData(:)];
        end
        barYData = barYData(isfinite(barYData) & barYData > 0);
        if ~isempty(barYData)
            medVal = median(barYData);
            maxReasonable = min(yl(2), medVal * 10);
            if maxReasonable > 0
                ylim([0, maxReasonable]);
            end
        end
    end

    title(sprintf('%s (size=%d)', metricLabels{mi}, stimulusSizes(sizeIdx)));

    if mi == 1
        legend({'Clamped', '', 'Double Pulse', ''}, 'Location', 'best');
    end
end

sgtitle(sprintf('Clamped vs Double Pulse (Duration: %s)', durationStr));
end % hasBothModes

%% Figure 6: Propagation Distance Over Time
fig6 = figure('Name', sprintf('Propagation Distance (dur=%d)', currentDuration));

% Use representative stimulus size
repSize = 10;
sizeIdx = find(stimulusSizes == repSize, 1);
if isempty(sizeIdx)
    sizeIdx = round(length(stimulusSizes) / 2);
end

for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};

    % Map display mode to data key
    dataKey = dataKeyFromDisplayMode(displayMode);

    subplot(1, nDisplayModes, m);
    hold on;

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Data, condition) || ~isfield(Data.(condition), dataKey)
            continue;
        end

        % Get propagation [nSims x nReplicates x nFrames]
        % Handle bias mode (5D) vs other modes (4D)
        propData = Data.(condition).(dataKey).stimulus_blob_extent;
        if isHighBiasDisplayMode(displayMode) && ndims(propData) == 5
            propagation = squeeze(propData(:, sizeIdx, end, :, :));  % Use highest bias
        elseif isLowBiasDisplayMode(displayMode) && ndims(propData) == 5
            propagation = squeeze(propData(:, sizeIdx, lowBiasIdx, :, :));  % Use low bias
        else
            propagation = squeeze(propData(:, sizeIdx, :, :));
        end
        meanProp = squeeze(mean(mean(propagation, 1), 2));

        % Compute SEM
        if ndims(propagation) == 3
            simMeans = squeeze(mean(propagation, 2));  % [nSims x nFrames]
        else
            simMeans = propagation;
        end
        if size(simMeans, 1) > 1
            semProp = std(simMeans, 0, 1) / sqrt(size(simMeans, 1));
            fill([timeVec_plot, fliplr(timeVec_plot)], ...
                [meanProp(:)' + semProp(:)', fliplr(meanProp(:)' - semProp(:)')], ...
                color, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        end

        color = config.colors.(condition);
        plot(timeVec_plot, meanProp, 'Color', color, 'LineWidth', 2);
    end

    % Mark stimulus period
    xline(0, 'k--', 'LineWidth', 1);
    xline(stimOnTime_plot, 'k--', 'LineWidth', 1);

    % Shade stimulus period
    yl = ylim;
    patch([0 stimOnTime_plot stimOnTime_plot 0], [yl(1) yl(1) yl(2) yl(2)], ...
        [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

    % Mark stimulus radius
    yline(stimulusSizes(sizeIdx)/2, 'r--', 'Stimulus Radius', 'LineWidth', 2, 'FontSize', 8, 'LabelHorizontalAlignment', 'left');

    xlabel(timeLabel);
    ylabel('Max Propagation Distance (pixels)');
    titleMode = strrep(displayMode, '_', ' ');
    title(sprintf('Propagation: %s (size=%d)', titleMode, stimulusSizes(sizeIdx)), 'Interpreter', 'none');
    if m == 1
        legend(conditions, 'Location', 'best');
    end
    grid on;
end

sgtitle(sprintf('Propagation Distance (Duration: %s)', durationStr));

%% Figure 7: Net Propagation Heatmap (Condition x Size) - Both Modes
fig7 = figure('Name', sprintf('Net Propagation Heatmap (dur=%d)', currentDuration));

% Compute stimulus radii for net propagation calculation
stimulusRadii = stimulusSizes / 2;

% Get global color limits for consistent scaling (using net propagation)
allNetProp = [];
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Metrics, condition)
        continue;
    end
    for m = 1:nDisplayModes
        metricsKey = strrep(metricsDisplayModes{m}, '_', '');
        if isfield(Metrics.(condition), metricsKey)
            % Subtract stimulus radius for each size
            for s = 1:length(stimulusSizes)
                netProp = Metrics.(condition).(metricsKey).max_blob_extent(:, s) - stimulusRadii(s);
                allNetProp = [allNetProp; netProp(:)];
            end
        end
    end
end
% Symmetric color limits centered on 0 for diverging colormap
maxAbsVal = max(abs(allNetProp));
clims = [-maxAbsVal, maxAbsVal];

for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};
    metricsKey = strrep(displayMode, '_', '');

    subplot(1, nDisplayModes, m);

    % Build NET propagation matrix [conditions x sizes]
    propMatrix = zeros(length(conditions), length(stimulusSizes));
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKey)
            continue;
        end
        for s = 1:length(stimulusSizes)
            % Mean max blob extent minus stimulus radius = net propagation
            meanExtent = mean(Metrics.(condition).(metricsKey).max_blob_extent(:, s), 'all');
            propMatrix(c, s) = meanExtent - stimulusRadii(s);
        end
    end

    % Create heatmap with diverging colormap (blue=negative, white=0, red=positive)
    imagesc(propMatrix, clims);
    colormap(redblue(256));  % Diverging colormap
    cb = colorbar;
    cb.Label.String = 'Net Propagation (px beyond stimulus)';

    % Labels
    set(gca, 'XTick', 1:length(stimulusSizes), 'XTickLabel', stimulusSizes);
    set(gca, 'YTick', 1:length(conditions), 'YTickLabel', conditions);
    xlabel('Stimulus Size (pixels)');
    ylabel('Condition');

    % Add values as text overlay
    for c = 1:length(conditions)
        for s = 1:length(stimulusSizes)
            val = propMatrix(c, s);
            % Use appropriate text color based on value
            if abs(val) < maxAbsVal * 0.3
                txtColor = 'k';  % Dark text near zero (white background)
            else
                txtColor = 'w';  % White text on colored background
            end
            text(s, c, sprintf('%.1f', val), 'HorizontalAlignment', 'center', ...
                'Color', txtColor, 'FontSize', 7);
        end
    end

    title(sprintf('%s', strrep(displayMode, '_', ' ')), 'Interpreter', 'none');
end

sgtitle(sprintf('Net Propagation Beyond Stimulus (Duration: %s)', durationStr));

%% Figure 8: Pre-Stimulus State Effects
fig8 = figure('Name', sprintf('Pre-Stimulus Effects (dur=%d)', currentDuration));

% Panel A: Baseline Moran's I by condition (was Panel C)
subplot(1, 3, 1);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(PreStimEffects, condition)
        continue;
    end

    color = config.colors.(condition);
    moransVals = PreStimEffects.(condition).(stimModeKey).baseline_moransI(:);
    moransVals = moransVals(~isnan(moransVals) & moransVals ~= 0);

    if ~isempty(moransVals)
        scatter(c * ones(size(moransVals)) + 0.1*(rand(size(moransVals))-0.5), moransVals, ...
            30, color, 'filled', 'MarkerFaceAlpha', 0.6);
        plot([c-0.3, c+0.3], [mean(moransVals), mean(moransVals)], 'k-', 'LineWidth', 2);
    end
end
set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
ylabel('Baseline Morans I');
title('A) Spatial Clustering');

% Panel B: Baseline vs Peak response correlation (was Panel A)
subplot(1, 3, 2);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(PreStimEffects, condition)
        continue;
    end

    color = config.colors.(condition);
    corrVals = PreStimEffects.(condition).(stimModeKey).baseline_vs_peak(:);
    corrVals = corrVals(~isnan(corrVals));

    if ~isempty(corrVals)
        scatter(c * ones(size(corrVals)) + 0.1*(rand(size(corrVals))-0.5), corrVals, ...
            30, color, 'filled', 'MarkerFaceAlpha', 0.6);
        plot([c-0.3, c+0.3], [mean(corrVals), mean(corrVals)], 'k-', 'LineWidth', 2);
    end
end
set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
ylabel('Correlation (r)');
yline(0, 'k--');
title('B) Baseline vs Peak');

% Panel C: Baseline vs Extent correlation (was Panel B)
subplot(1, 3, 3);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(PreStimEffects, condition)
        continue;
    end

    color = config.colors.(condition);
    corrVals = PreStimEffects.(condition).(stimModeKey).baseline_vs_extent(:);
    corrVals = corrVals(~isnan(corrVals));

    if ~isempty(corrVals)
        scatter(c * ones(size(corrVals)) + 0.1*(rand(size(corrVals))-0.5), corrVals, ...
            30, color, 'filled', 'MarkerFaceAlpha', 0.6);
        plot([c-0.3, c+0.3], [mean(corrVals), mean(corrVals)], 'k-', 'LineWidth', 2);
    end
end
set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
yline(0, 'k--');
title('C) Baseline vs Propagation');

sgtitle(sprintf('Pre-Stimulus State Effects (Duration: %s)', durationStr));

%% Figure 9: Propagation Dynamics
fig9 = figure('Name', sprintf('Propagation Dynamics (dur=%d)', currentDuration));

% Panel A: Max velocity vs stimulus size
subplot(1, 2, 1);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(PropDynamics, condition)
        continue;
    end

    velocity = PropDynamics.(condition).(stimModeKey).max_velocity;
    meanVel = mean(velocity, 1) * velocityScale;  % Scale to px/s if real time
    semVel = std(velocity, 0, 1) / sqrt(size(velocity, 1)) * velocityScale;

    color = config.colors.(condition);
    errorbar(stimulusSizes, meanVel, semVel, 'o-', ...
        'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
end
xlabel('Stimulus Size (pixels)');
ylabel(sprintf('Max Velocity (%s)', velocityUnit));
title('A) Peak Propagation Velocity');
legend(conditions, 'Location', 'best');
grid on;

% Panel B: Time to max extent vs size
subplot(1, 2, 2);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(PropDynamics, condition)
        continue;
    end

    ttm = PropDynamics.(condition).(stimModeKey).time_to_max;
    meanTTM = mean(ttm, 1) * timeScale;  % Scale to seconds if real time
    semTTM = std(ttm, 0, 1) / sqrt(size(ttm, 1)) * timeScale;

    color = config.colors.(condition);
    errorbar(stimulusSizes, meanTTM, semTTM, 'o-', ...
        'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
end
xlabel('Stimulus Size (pixels)');
ylabel(sprintf('Time to Max (%s)', timeUnit));
title('B) Time to Peak Spread');
legend(conditions, 'Location', 'best');
grid on;
ylim([0, inf]);  % Clip negative time-to-peak values

sgtitle(sprintf('Propagation Dynamics (Duration: %s)', durationStr));

%% Figure 10: Blob Interactions
fig10 = figure('Name', sprintf('Blob Interactions (dur=%d)', currentDuration));

% Panel A: Blob count change vs size
subplot(2, 2, 1);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(BlobInteractions, condition)
        continue;
    end

    change = BlobInteractions.(condition).(stimModeKey).blob_count_change;
    meanChange = mean(change, 1);
    semChange = std(change, 0, 1) / sqrt(size(change, 1));

    color = config.colors.(condition);
    errorbar(stimulusSizes, meanChange, semChange, 'o-', ...
        'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
end
xlabel('Stimulus Size (pixels)');
ylabel('\Delta Blob Count');
yline(0, 'k--');
title('A) Stimulus Effect on Blob Count');
legend(conditions, 'Location', 'best');
grid on;

% Panel B: Pre-stim vs stim blob count
subplot(2, 2, 2);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(BlobInteractions, condition)
        continue;
    end

    prestim = BlobInteractions.(condition).(stimModeKey).prestim_blob_count(:);
    stim = BlobInteractions.(condition).(stimModeKey).stim_blob_count(:);

    color = config.colors.(condition);
    scatter(prestim, stim, 30, color, 'filled', 'MarkerFaceAlpha', 0.5);
end
plot([0 5], [0 5], 'k:', 'LineWidth', 1);
xlabel('Pre-stim Blob Count');
ylabel('During-stim Blob Count');
title('B) Blob Count: Pre vs During Stim');
legend(conditions, 'Location', 'best');

% Panel C: Mean blob counts by condition
subplot(2, 2, 3);
hold on;
xPos = 1;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(BlobInteractions, condition)
        continue;
    end

    color = config.colors.(condition);
    prestim = mean(BlobInteractions.(condition).(stimModeKey).prestim_blob_count, 'all');
    stim = mean(BlobInteractions.(condition).(stimModeKey).stim_blob_count, 'all');

    bar(xPos, prestim, 0.35, 'FaceColor', color, 'EdgeColor', 'k');
    bar(xPos + 0.4, stim, 0.35, 'FaceColor', color, 'FaceAlpha', 0.5, 'EdgeColor', 'k');

    xPos = xPos + 1.2;
end
set(gca, 'XTick', 1.2*(0:length(conditions)-1) + 1.2, 'XTickLabel', conditions);
ylabel('Blob Count');
title('C) Pre-stim (solid) vs Stim (light)');

% Panel D: Summary
subplot(2, 2, 4);
axis off;
summaryText = {'\bf Blob Interaction Summary \rm', ''};
for c = 1:length(conditions)
    condition = conditions{c};
    if isfield(BlobInteractions, condition)
        prestim = mean(BlobInteractions.(condition).(stimModeKey).prestim_blob_count, 'all');
        stim = mean(BlobInteractions.(condition).(stimModeKey).stim_blob_count, 'all');
        change = mean(BlobInteractions.(condition).(stimModeKey).blob_count_change, 'all');
        summaryText{end+1} = sprintf('%s: Pre=%.1f, Stim=%.1f, \x0394=%.1f', ...
            condition, prestim, stim, change);
    end
end
text(0.1, 0.8, summaryText, 'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 10);

sgtitle(sprintf('Blob Interaction Analysis (Duration: %s)', durationStr));

%% Figure 11: Moran's I Dynamics
fig11 = figure('Name', sprintf('Morans I Dynamics (dur=%d)', currentDuration));

% Panel A: Moran's I time course
repSize = 8;
sizeIdx = find(stimulusSizes == repSize, 1);
if isempty(sizeIdx), sizeIdx = 4; end

subplot(1, 3, 1);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Data, condition)
        continue;
    end

    moransArr = collapseBiasPlot(Data.(condition).(stimModeKey).morans_I);
    moransI = squeeze(moransArr(:, sizeIdx, :, :));
    meanMoransI = squeeze(mean(mean(moransI, 1), 2));

    color = config.colors.(condition);
    plot(timeVec_plot, meanMoransI, 'Color', color, 'LineWidth', 2);
end
xline(0, 'k--'); xline(stimOnTime_plot, 'k--');
xlabel(timeLabel);
ylabel('Morans I');
title(sprintf('A) Morans I Time Course (size=%d)', stimulusSizes(sizeIdx)));
legend(conditions, 'Location', 'best');

% Panel B: Moran's I increase vs size
subplot(1, 3, 2);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(MoransIDynamics, condition)
        continue;
    end

    increase = MoransIDynamics.(condition).(stimModeKey).moransI_increase;
    meanInc = mean(increase, 1);
    semInc = std(increase, 0, 1) / sqrt(size(increase, 1));

    color = config.colors.(condition);
    errorbar(stimulusSizes, meanInc, semInc, 'o-', ...
        'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
end
xlabel('Stimulus Size (pixels)');
ylabel('\Delta Morans I');
title('B) Clustering Increase vs Size');
grid on;

% Panel C: Return to baseline time (was D)
subplot(1, 3, 3);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(MoransIDynamics, condition)
        continue;
    end

    returnTime = MoransIDynamics.(condition).(stimModeKey).return_to_baseline_time;
    meanRT = mean(returnTime, 1) * timeScale;  % Scale to seconds if real time
    semRT = std(returnTime, 0, 1) / sqrt(size(returnTime, 1)) * timeScale;

    color = config.colors.(condition);
    errorbar(stimulusSizes, meanRT, semRT, 'o-', ...
        'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
end
xlabel('Stimulus Size (pixels)');
ylabel(sprintf('Return Time (%s)', timeUnit));
title('C) Time to Return to Baseline');
grid on;

sgtitle(sprintf('Morans I Dynamics (Duration: %s)', durationStr));

%% Figure 12: Phase Portrait (Activity vs Blob Extent)
fig12 = figure('Name', sprintf('Phase Portrait (dur=%d)', currentDuration));

% Use clamped mode, size=8
repSize = 8;
sizeIdx12 = find(stimulusSizes == repSize, 1);
if isempty(sizeIdx12), sizeIdx12 = round(length(stimulusSizes) / 2); end

for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Data, condition) || ~isfield(Data.(condition), stimModeKey)
        continue;
    end

    subplot(1, length(conditions), c);
    hold on;

    % Get activity and blob extent time series (collapsed to 4D for bias modes)
    actData = collapseBiasPlot(Data.(condition).(stimModeKey).activity);
    extData = collapseBiasPlot(Data.(condition).(stimModeKey).stimulus_blob_extent);
    activity = squeeze(actData(:, sizeIdx12, :, :));  % [nSims x nReps x nFrames]
    extent = squeeze(extData(:, sizeIdx12, :, :));

    meanAct = squeeze(mean(mean(activity, 1), 2));  % [nFrames]
    meanExt = squeeze(mean(mean(extent, 1), 2));

    % Color-code by time (cool -> warm)
    nPts = length(meanAct);
    timeColors = cool2warm(nPts);

    % Plot trajectory with color gradient
    for t = 1:(nPts-1)
        plot(meanExt(t:t+1), meanAct(t:t+1), '-', 'Color', timeColors(t,:), 'LineWidth', 2);
    end

    % Mark stimulus onset and offset
    stimOnPt = preStimFrames + 1;
    stimOffPt = preStimFrames + stimOnFrames;
    plot(meanExt(stimOnPt), meanAct(stimOnPt), 'k^', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'LineWidth', 1.5);
    plot(meanExt(stimOffPt), meanAct(stimOffPt), 'kv', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'LineWidth', 1.5);
    % Mark start and end
    plot(meanExt(1), meanAct(1), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', [0.5 0.5 0.5]);
    plot(meanExt(end), meanAct(end), 'ks', 'MarkerSize', 8, 'MarkerFaceColor', [0.5 0.5 0.5]);

    xlabel('Blob Extent (px)');
    ylabel('Activity');
    title(sprintf('%s', condition));
    if c == 1
        legend({'', 'Stim On', 'Stim Off', 'Start', 'End'}, 'Location', 'best', 'FontSize', 7);
    end

    % Add colorbar for time
    if c == length(conditions)
        colormap(gca, cool2warm(256));
        cb = colorbar;
        cb.Ticks = [0, 0.5, 1];
        cb.TickLabels = {'Pre', 'Stim', 'Post'};
        ylabel(cb, 'Time');
    end
end

sgtitle(sprintf('Phase Portrait: Activity vs Propagation (size=%d, Duration: %s)', stimulusSizes(sizeIdx12), durationStr));

%% Figure 13: Recovery Dynamics Summary
fig13 = figure('Name', sprintf('Recovery Dynamics (dur=%d)', currentDuration));

metricsKey13 = stimModeKey;

% Panel A: Post-stim AUC vs stimulus size
subplot(2, 2, 1);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKey13)
        continue;
    end
    auc = Metrics.(condition).(metricsKey13).post_stim_auc;
    meanAUC = mean(auc, 1);
    semAUC = std(auc, 0, 1) / sqrt(size(auc, 1));
    color = config.colors.(condition);
    errorbar(stimulusSizes, meanAUC, semAUC, 'o-', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
end
xlabel('Stimulus Size (px)');
ylabel('Post-stim AUC');
title('A) Post-Stimulus Excess Activity');
legend(conditions, 'Location', 'northwest', 'FontSize', 7);
grid on;

% Panel B: Return-to-baseline time vs stimulus size
subplot(2, 2, 2);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKey13)
        continue;
    end
    rtb = Metrics.(condition).(metricsKey13).return_to_baseline;
    meanRTB = mean(rtb, 1) * timeScale;
    semRTB = std(rtb, 0, 1) / sqrt(size(rtb, 1)) * timeScale;
    color = config.colors.(condition);
    errorbar(stimulusSizes, meanRTB, semRTB, 'o-', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
end
xlabel('Stimulus Size (px)');
ylabel(sprintf('Return to Baseline (%s)', timeUnit));
title('B) Recovery Time vs Size');
grid on;

% Panel C: Amplification vs half-decay scatter
subplot(2, 2, 3);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKey13)
        continue;
    end
    amp = Metrics.(condition).(metricsKey13).amplification;
    halfDecay = Metrics.(condition).(metricsKey13).half_decay_time * timeScale;
    color = config.colors.(condition);
    scatter(amp(:), halfDecay(:), 20, color, 'filled', 'MarkerFaceAlpha', 0.4);
end
xlabel('Amplification (Peak/Baseline)');
ylabel(sprintf('Half-Decay Time (%s)', timeUnit));
title('C) Amplification vs Persistence');
legend(conditions, 'Location', 'best', 'FontSize', 7);
grid on;

% Panel D: Mean decay tau per condition (bar plot with dots)
subplot(2, 2, 4);
hold on;
xPos = 1;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKey13)
        continue;
    end
    % Use largest stimulus size for most reliable tau
    largestSizeIdx = length(stimulusSizes);
    tauVals = Metrics.(condition).(metricsKey13).decay_tau(:, largestSizeIdx) * timeScale;
    tauVals = tauVals(~isnan(tauVals) & ~isinf(tauVals));
    if ~isempty(tauVals)
        color = config.colors.(condition);
        bar(xPos, mean(tauVals), 0.6, 'FaceColor', color, 'EdgeColor', 'k', 'FaceAlpha', 0.7);
        scatter(xPos + 0.15*(rand(size(tauVals))-0.5), tauVals, 25, color, 'filled', 'MarkerFaceAlpha', 0.6);
        errorbar(xPos, mean(tauVals), std(tauVals)/sqrt(length(tauVals)), 'k', 'LineWidth', 1.5);
    end
    xPos = xPos + 1;
end
set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
ylabel(sprintf('Decay \\tau (%s)', timeUnit));
title(sprintf('D) Decay Tau (size=%d)', stimulusSizes(end)));
grid on;

sgtitle(sprintf('Recovery Dynamics Summary (Duration: %s)', durationStr));

%% Figure 14: Condition Comparison Radar Chart
fig14 = figure('Name', sprintf('Condition Radar (dur=%d)', currentDuration));

metricsKey14 = stimModeKey;
largestSizeIdx14 = length(stimulusSizes);

% Collect 5 key metrics per condition
metricNames14 = {'Peak Amplification', 'Max Propagation', 'Persistence', 'Sensitivity', 'Spatial Coherence'};
nMetrics14 = length(metricNames14);
nConds14 = length(conditions);
radarData = zeros(nConds14, nMetrics14);

for c = 1:nConds14
    condition = conditions{c};
    if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKey14)
        continue;
    end

    % 1. Peak amplification (largest size)
    radarData(c, 1) = mean(Metrics.(condition).(metricsKey14).amplification(:, largestSizeIdx14));

    % 2. Max propagation extent (largest size)
    radarData(c, 2) = mean(Metrics.(condition).(metricsKey14).max_blob_extent(:, largestSizeIdx14));

    % 3. Persistence = half-decay time (largest size)
    radarData(c, 3) = mean(Metrics.(condition).(metricsKey14).half_decay_time(:, largestSizeIdx14)) * timeScale;

    % 4. Sensitivity = 1/EC50 (higher = more sensitive)
    if isfield(Gating, condition) && isfield(Gating.(condition), metricsKey14)
        ec50 = Gating.(condition).(metricsKey14).EC50;
        if ~isnan(ec50) && ec50 > 0
            radarData(c, 4) = 1 / ec50;
        end
    end

    % 5. Spatial coherence = peak Moran's I increase (largest size)
    if isfield(MoransIDynamics, condition) && isfield(MoransIDynamics.(condition), metricsKey14)
        radarData(c, 5) = mean(MoransIDynamics.(condition).(metricsKey14).moransI_increase(:, largestSizeIdx14));
    end
end

% Normalize each metric to [0, 1] across conditions
radarNorm = zeros(size(radarData));
for mi = 1:nMetrics14
    minVal = min(radarData(:, mi));
    maxVal = max(radarData(:, mi));
    if maxVal > minVal
        radarNorm(:, mi) = (radarData(:, mi) - minVal) / (maxVal - minVal);
    else
        radarNorm(:, mi) = 0.5;
    end
end

% Create radar chart using polarplot
angles = linspace(0, 2*pi, nMetrics14 + 1);  % +1 to close the polygon

pax = polaraxes('Position', [0.12 0.15 0.33 0.7]);
hold(pax, 'on');

for c = 1:nConds14
    condition = conditions{c};
    color = config.colors.(condition);

    % Close the polygon by repeating the first point
    vals = [radarNorm(c, :), radarNorm(c, 1)];

    polarplot(pax, angles, vals, '-o', 'Color', color, 'LineWidth', 2, ...
        'MarkerFaceColor', color, 'MarkerSize', 6);
end

% Configure polar axes
pax.ThetaTick = rad2deg(angles(1:nMetrics14));
pax.ThetaTickLabel = metricNames14;
pax.RLim = [0, 1.1];
pax.RTickLabel = {};
legend(pax, conditions, 'Location', 'bestoutside', 'FontSize', 8);
title(pax, 'Normalized Metric Comparison');

% Panel 2: Grouped bar chart (alternative view)
subplot(1, 2, 2);
hold on;
b = bar(radarNorm);
for c = 1:nConds14
    b(c).FaceColor = 'flat';
    % Bar plots color by group, not by series - set each bar's color
end
% Color each group
barColors = zeros(nConds14, 3);
for c = 1:nConds14
    barColors(c,:) = config.colors.(conditions{c});
end

% Re-plot as grouped bars with proper colors
cla;
x = 1:nMetrics14;
barWidth = 0.8 / nConds14;
for c = 1:nConds14
    color = config.colors.(conditions{c});
    xOffset = (c - (nConds14+1)/2) * barWidth;
    bar(x + xOffset, radarNorm(c, :), barWidth, 'FaceColor', color, 'EdgeColor', 'k');
end
set(gca, 'XTick', 1:nMetrics14, 'XTickLabel', metricNames14, 'XTickLabelRotation', 30);
ylabel('Normalized Value [0-1]');
title('Grouped Bar Comparison');
legend(conditions, 'Location', 'best', 'FontSize', 7);
ylim([0, 1.15]);
grid on;

sgtitle(sprintf('Condition Comparison (size=%d, Duration: %s)', stimulusSizes(largestSizeIdx14), durationStr));

%% Decay Tau Figures: Per-Size + Averaged (saved to decay_tau/ subfolder)
fprintf('\n--- Generating Decay Tau Figures ---\n');

metricsKeyTau = stimModeKey;

% Compute output path for this simPass (needed for saving)
if simPass == 1
    simPassOutputPath = durOutputPath;
else
    simPassOutputPath = fullfile(durOutputPath, 'bestMatch');
    if ~exist(simPassOutputPath, 'dir'), mkdir(simPassOutputPath); end
end

tauOutputPath = fullfile(simPassOutputPath, 'decay_tau');
if ~exist(tauOutputPath, 'dir'), mkdir(tauOutputPath); end

% --- Per-size figures ---
for s = 1:length(stimulusSizes)
    stimSize = stimulusSizes(s);

    % == Temporal Scaling Diagnostics for this size ==
    figTSD = figure('Name', sprintf('TemporalScalingDiag size=%d dur=%d', stimSize, currentDuration));

    condLabelsTSD = {};
    pertTauMeans = [];
    pertTauSEMs = [];
    expTauVals = [];
    ratioVals = [];
    pertTauAll = {};
    condColorsTSD = [];

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyTau)
            continue;
        end
        tauRaw = Metrics.(condition).(metricsKeyTau).decay_tau(:, s) * timeScale;
        tauValid = tauRaw(~isnan(tauRaw) & ~isinf(tauRaw));
        if isempty(tauValid), continue; end

        condLabelsTSD{end+1} = condition;
        pertTauMeans(end+1) = mean(tauValid);
        pertTauSEMs(end+1) = std(tauValid) / sqrt(length(tauValid));
        pertTauAll{end+1} = tauValid;
        condColorsTSD(end+1, :) = config.colors.(condition);

        if isfield(config.expTau, condition) && ~isnan(config.expTau.(condition))
            expTauVals(end+1) = config.expTau.(condition) / config.samplingRate;
        else
            expTauVals(end+1) = NaN;
        end
        ratioVals(end+1) = pertTauMeans(end) / max(expTauVals(end), eps);
    end

    nCondsTSD = length(condLabelsTSD);
    if nCondsTSD > 0
        hold on;
        xPosTSD = 1:nCondsTSD;
        for ci = 1:nCondsTSD
            bar(xPosTSD(ci), pertTauMeans(ci), 0.6, 'FaceColor', condColorsTSD(ci,:), ...
                'EdgeColor', 'k', 'FaceAlpha', 0.7);
        end
        errorbar(xPosTSD, pertTauMeans, pertTauSEMs, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
        for ci = 1:nCondsTSD
            jitterTSD = 0.15 * (rand(size(pertTauAll{ci})) - 0.5);
            scatter(xPosTSD(ci) + jitterTSD, pertTauAll{ci}, 25, condColorsTSD(ci,:), ...
                'filled', 'MarkerFaceAlpha', 0.6);
        end
        for ci = 1:nCondsTSD
            if ~isnan(expTauVals(ci))
                plot(xPosTSD(ci) + [-0.3, 0.3], [expTauVals(ci), expTauVals(ci)], ...
                    '--', 'Color', condColorsTSD(ci,:)*0.6, 'LineWidth', 2);
            end
        end
        for ci = 1:nCondsTSD
            if ~isnan(expTauVals(ci))
                yTop = pertTauMeans(ci) + pertTauSEMs(ci);
                text(xPosTSD(ci), yTop * 1.08, sprintf('%.1fx', ratioVals(ci)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
            end
        end
        set(gca, 'XTick', xPosTSD, 'XTickLabel', condLabelsTSD);
        ylabel(sprintf('Decay \\tau (%s)', timeUnit));
        title(sprintf('Perturbation Decay \\tau vs Exp. Autocorr \\tau (size=%d)', stimSize));
        legend({'Perturbation \tau (bar)', 'Exp. autocorr \tau (dashed)'}, ...
            'Location', 'best', 'FontSize', 7);
        grid on;
        sgtitle(sprintf('Temporal Scaling Diagnostics (Duration: %s)', durationStr));
    end

    tsdName = sprintf('TemporalScalingDiagnostics_size_%d', stimSize);
    savemyfig(figTSD, fullfile(tauOutputPath, tsdName));
    fprintf('  Saved: decay_tau/%s\n', tsdName);
    close(figTSD);

    % == Decay Tau Scatter for this size ==
    figDTS = figure('Name', sprintf('DecayTauScatter size=%d dur=%d', stimSize, currentDuration));

    condLabelsDTS = {};
    tauAllPerCond = {};
    condColorsDTS = [];

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyTau)
            continue;
        end
        tauRaw = Metrics.(condition).(metricsKeyTau).decay_tau(:, s) * timeScale;
        tauValid = tauRaw(~isnan(tauRaw) & ~isinf(tauRaw));
        if isempty(tauValid), continue; end

        condLabelsDTS{end+1} = condition;
        tauAllPerCond{end+1} = tauValid;
        condColorsDTS(end+1, :) = config.colors.(condition);
    end

    nCondsDTS = length(condLabelsDTS);
    if nCondsDTS > 0
        hold on;
        for ci = 1:nCondsDTS
            tauV = tauAllPerCond{ci};
            jitterDTS = 0.2 * (rand(size(tauV)) - 0.5);
            scatter(ci + jitterDTS, tauV, 40, condColorsDTS(ci,:), 'filled', ...
                'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
            plot(ci + [-0.25, 0.25], [median(tauV), median(tauV)], '-', ...
                'Color', condColorsDTS(ci,:)*0.5, 'LineWidth', 2);
        end
        set(gca, 'YScale', 'log');
        set(gca, 'XTick', 1:nCondsDTS, 'XTickLabel', condLabelsDTS);
        ylabel('\tau [s]');
        xlabel('Condition');
        title(sprintf('Stim. Offset Decay \\tau (size=%d, Duration: %s)', stimSize, durationStr));
        grid on;

        % Significance brackets (pairwise Wilcoxon rank-sum)
        bracketY = max(cellfun(@max, tauAllPerCond)) * 1.15;
        bracketStep = bracketY * 0.2;
        bracketCount = 0;
        for ci = 1:nCondsDTS
            for cj = (ci+1):nCondsDTS
                [pVal, ~] = ranksum(tauAllPerCond{ci}, tauAllPerCond{cj});
                if pVal < 0.05
                    if pVal < 0.001
                        sigStr = '***';
                    elseif pVal < 0.01
                        sigStr = '**';
                    else
                        sigStr = '*';
                    end
                    yBracket = bracketY + bracketCount * bracketStep;
                    bracketCount = bracketCount + 1;
                    line([ci, ci, cj, cj], [yBracket*0.95, yBracket, yBracket, yBracket*0.95], ...
                        'Color', 'k', 'LineWidth', 1);
                    text((ci + cj)/2, yBracket * 1.03, sigStr, ...
                        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
                end
            end
        end
        xlim([0.5, nCondsDTS + 0.5]);
    end

    dtsName = sprintf('DecayTauScatter_size_%d', stimSize);
    savemyfig(figDTS, fullfile(tauOutputPath, dtsName));
    fprintf('  Saved: decay_tau/%s\n', dtsName);
    close(figDTS);

    % == Decay Tau Scatter for this size (LINEAR Y-axis) ==
    figDTS_lin = figure('Name', sprintf('DecayTauScatter linear size=%d dur=%d', stimSize, currentDuration));

    if nCondsDTS > 0
        hold on;
        for ci = 1:nCondsDTS
            tauV = tauAllPerCond{ci};
            jitterDTS = 0.2 * (rand(size(tauV)) - 0.5);
            scatter(ci + jitterDTS, tauV, 40, condColorsDTS(ci,:), 'filled', ...
                'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
            plot(ci + [-0.25, 0.25], [median(tauV), median(tauV)], '-', ...
                'Color', condColorsDTS(ci,:)*0.5, 'LineWidth', 2);
        end
        set(gca, 'XTick', 1:nCondsDTS, 'XTickLabel', condLabelsDTS);
        ylabel('\tau [s]');
        xlabel('Condition');
        title(sprintf('Stim. Offset Decay \\tau (size=%d, Duration: %s, linear)', stimSize, durationStr));
        grid on;

        % Significance brackets (pairwise Wilcoxon rank-sum)
        bracketY = max(cellfun(@max, tauAllPerCond)) * 1.15;
        bracketStep = bracketY * 0.2;
        bracketCount = 0;
        for ci = 1:nCondsDTS
            for cj = (ci+1):nCondsDTS
                [pVal, ~] = ranksum(tauAllPerCond{ci}, tauAllPerCond{cj});
                if pVal < 0.05
                    if pVal < 0.001
                        sigStr = '***';
                    elseif pVal < 0.01
                        sigStr = '**';
                    else
                        sigStr = '*';
                    end
                    yBracket = bracketY + bracketCount * bracketStep;
                    bracketCount = bracketCount + 1;
                    line([ci, ci, cj, cj], [yBracket*0.95, yBracket, yBracket, yBracket*0.95], ...
                        'Color', 'k', 'LineWidth', 1);
                    text((ci + cj)/2, yBracket * 1.03, sigStr, ...
                        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
                end
            end
        end
        xlim([0.5, nCondsDTS + 0.5]);
    end

    dtsNameLin = sprintf('DecayTauScatter_size_%d_linear', stimSize);
    savemyfig(figDTS_lin, fullfile(tauOutputPath, dtsNameLin));
    fprintf('  Saved: decay_tau/%s\n', dtsNameLin);
    close(figDTS_lin);

    % == NLS Decay Tau Scatter for this size (LOG Y-axis) ==
    figDTS_nls = figure('Name', sprintf('DecayTauScatter NLS size=%d dur=%d', stimSize, currentDuration));

    condLabelsNLS = {};
    tauAllPerCondNLS = {};
    condColorsNLS = [];

    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyTau)
            continue;
        end
        tauRawNLS = Metrics.(condition).(metricsKeyTau).decay_tau_nls(:, s) * timeScale;
        tauValidNLS = tauRawNLS(~isnan(tauRawNLS) & ~isinf(tauRawNLS));
        if isempty(tauValidNLS), continue; end

        condLabelsNLS{end+1} = condition;
        tauAllPerCondNLS{end+1} = tauValidNLS;
        condColorsNLS(end+1, :) = config.colors.(condition);
    end

    nCondsNLS = length(condLabelsNLS);
    if nCondsNLS > 0
        hold on;
        for ci = 1:nCondsNLS
            tauV = tauAllPerCondNLS{ci};
            jitterNLS = 0.2 * (rand(size(tauV)) - 0.5);
            scatter(ci + jitterNLS, tauV, 40, condColorsNLS(ci,:), 'filled', ...
                'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
            plot(ci + [-0.25, 0.25], [median(tauV), median(tauV)], '-', ...
                'Color', condColorsNLS(ci,:)*0.5, 'LineWidth', 2);
        end
        set(gca, 'YScale', 'log');
        set(gca, 'XTick', 1:nCondsNLS, 'XTickLabel', condLabelsNLS);
        ylabel('\tau [s]');
        xlabel('Condition');
        title(sprintf('Stim. Offset Decay \\tau NLS (size=%d, Duration: %s)', stimSize, durationStr));
        grid on;

        % Significance brackets (pairwise Wilcoxon rank-sum)
        bracketY = max(cellfun(@max, tauAllPerCondNLS)) * 1.15;
        bracketStep = bracketY * 0.2;
        bracketCount = 0;
        for ci = 1:nCondsNLS
            for cj = (ci+1):nCondsNLS
                [pVal, ~] = ranksum(tauAllPerCondNLS{ci}, tauAllPerCondNLS{cj});
                if pVal < 0.05
                    if pVal < 0.001
                        sigStr = '***';
                    elseif pVal < 0.01
                        sigStr = '**';
                    else
                        sigStr = '*';
                    end
                    yBracket = bracketY + bracketCount * bracketStep;
                    bracketCount = bracketCount + 1;
                    line([ci, ci, cj, cj], [yBracket*0.95, yBracket, yBracket, yBracket*0.95], ...
                        'Color', 'k', 'LineWidth', 1);
                    text((ci + cj)/2, yBracket * 1.03, sigStr, ...
                        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
                end
            end
        end
        xlim([0.5, nCondsNLS + 0.5]);
    end

    dtsNameNLS = sprintf('DecayTauScatter_size_%d_nls', stimSize);
    savemyfig(figDTS_nls, fullfile(tauOutputPath, dtsNameNLS));
    fprintf('  Saved: decay_tau/%s\n', dtsNameNLS);
    close(figDTS_nls);

    % == NLS Decay Tau Scatter for this size (LINEAR Y-axis) ==
    figDTS_nls_lin = figure('Name', sprintf('DecayTauScatter NLS linear size=%d dur=%d', stimSize, currentDuration));

    if nCondsNLS > 0
        hold on;
        for ci = 1:nCondsNLS
            tauV = tauAllPerCondNLS{ci};
            jitterNLS = 0.2 * (rand(size(tauV)) - 0.5);
            scatter(ci + jitterNLS, tauV, 40, condColorsNLS(ci,:), 'filled', ...
                'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
            plot(ci + [-0.25, 0.25], [median(tauV), median(tauV)], '-', ...
                'Color', condColorsNLS(ci,:)*0.5, 'LineWidth', 2);
        end
        set(gca, 'XTick', 1:nCondsNLS, 'XTickLabel', condLabelsNLS);
        ylabel('\tau [s]');
        xlabel('Condition');
        title(sprintf('Stim. Offset Decay \\tau NLS (size=%d, Duration: %s, linear)', stimSize, durationStr));
        grid on;

        % Significance brackets (pairwise Wilcoxon rank-sum)
        bracketY = max(cellfun(@max, tauAllPerCondNLS)) * 1.15;
        bracketStep = bracketY * 0.2;
        bracketCount = 0;
        for ci = 1:nCondsNLS
            for cj = (ci+1):nCondsNLS
                [pVal, ~] = ranksum(tauAllPerCondNLS{ci}, tauAllPerCondNLS{cj});
                if pVal < 0.05
                    if pVal < 0.001
                        sigStr = '***';
                    elseif pVal < 0.01
                        sigStr = '**';
                    else
                        sigStr = '*';
                    end
                    yBracket = bracketY + bracketCount * bracketStep;
                    bracketCount = bracketCount + 1;
                    line([ci, ci, cj, cj], [yBracket*0.95, yBracket, yBracket, yBracket*0.95], ...
                        'Color', 'k', 'LineWidth', 1);
                    text((ci + cj)/2, yBracket * 1.03, sigStr, ...
                        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
                end
            end
        end
        xlim([0.5, nCondsNLS + 0.5]);
    end

    dtsNameNLSLin = sprintf('DecayTauScatter_size_%d_nls_linear', stimSize);
    savemyfig(figDTS_nls_lin, fullfile(tauOutputPath, dtsNameNLSLin));
    fprintf('  Saved: decay_tau/%s\n', dtsNameNLSLin);
    close(figDTS_nls_lin);
end

% --- Size-averaged figures ---

% == Temporal Scaling Diagnostics (averaged across sizes) ==
figTSD_avg = figure('Name', sprintf('TemporalScalingDiag avg dur=%d', currentDuration));

condLabelsAvg = {};
pertTauMeansAvg = [];
pertTauSEMsAvg = [];
expTauValsAvg = [];
ratioValsAvg = [];
pertTauAllAvg = {};
condColorsAvg = [];

for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyTau)
        continue;
    end
    % Average decay_tau across all sizes for each simulation
    tauRaw = mean(Metrics.(condition).(metricsKeyTau).decay_tau, 2) * timeScale;
    tauValid = tauRaw(~isnan(tauRaw) & ~isinf(tauRaw));
    if isempty(tauValid), continue; end

    condLabelsAvg{end+1} = condition;
    pertTauMeansAvg(end+1) = mean(tauValid);
    pertTauSEMsAvg(end+1) = std(tauValid) / sqrt(length(tauValid));
    pertTauAllAvg{end+1} = tauValid;
    condColorsAvg(end+1, :) = config.colors.(condition);

    if isfield(config.expTau, condition) && ~isnan(config.expTau.(condition))
        expTauValsAvg(end+1) = config.expTau.(condition) / config.samplingRate;
    else
        expTauValsAvg(end+1) = NaN;
    end
    ratioValsAvg(end+1) = pertTauMeansAvg(end) / max(expTauValsAvg(end), eps);
end

nCondsAvg = length(condLabelsAvg);
if nCondsAvg > 0
    hold on;
    xPosAvg = 1:nCondsAvg;
    for ci = 1:nCondsAvg
        bar(xPosAvg(ci), pertTauMeansAvg(ci), 0.6, 'FaceColor', condColorsAvg(ci,:), ...
            'EdgeColor', 'k', 'FaceAlpha', 0.7);
    end
    errorbar(xPosAvg, pertTauMeansAvg, pertTauSEMsAvg, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
    for ci = 1:nCondsAvg
        jitterAvg = 0.15 * (rand(size(pertTauAllAvg{ci})) - 0.5);
        scatter(xPosAvg(ci) + jitterAvg, pertTauAllAvg{ci}, 25, condColorsAvg(ci,:), ...
            'filled', 'MarkerFaceAlpha', 0.6);
    end
    for ci = 1:nCondsAvg
        if ~isnan(expTauValsAvg(ci))
            plot(xPosAvg(ci) + [-0.3, 0.3], [expTauValsAvg(ci), expTauValsAvg(ci)], ...
                '--', 'Color', condColorsAvg(ci,:)*0.6, 'LineWidth', 2);
        end
    end
    for ci = 1:nCondsAvg
        if ~isnan(expTauValsAvg(ci))
            yTop = pertTauMeansAvg(ci) + pertTauSEMsAvg(ci);
            text(xPosAvg(ci), yTop * 1.08, sprintf('%.1fx', ratioValsAvg(ci)), ...
                'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
        end
    end
    set(gca, 'XTick', xPosAvg, 'XTickLabel', condLabelsAvg);
    ylabel(sprintf('Decay \\tau (%s)', timeUnit));
    title('Perturbation Decay \tau vs Exp. Autocorr \tau (avg across sizes)');
    legend({'Perturbation \tau (bar)', 'Exp. autocorr \tau (dashed)'}, ...
        'Location', 'best', 'FontSize', 7);
    grid on;
    sgtitle(sprintf('Temporal Scaling Diagnostics (Duration: %s)', durationStr));
end

savemyfig(figTSD_avg, fullfile(tauOutputPath, 'TemporalScalingDiagnostics_avg'));
fprintf('  Saved: decay_tau/TemporalScalingDiagnostics_avg\n');
close(figTSD_avg);

% == Decay Tau Scatter (averaged across sizes) ==
figDTS_avg = figure('Name', sprintf('DecayTauScatter avg dur=%d', currentDuration));

condLabelsAvgS = {};
tauAllPerCondAvg = {};
condColorsAvgS = [];

for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyTau)
        continue;
    end
    tauRaw = mean(Metrics.(condition).(metricsKeyTau).decay_tau, 2) * timeScale;
    tauValid = tauRaw(~isnan(tauRaw) & ~isinf(tauRaw));
    if isempty(tauValid), continue; end

    condLabelsAvgS{end+1} = condition;
    tauAllPerCondAvg{end+1} = tauValid;
    condColorsAvgS(end+1, :) = config.colors.(condition);
end

nCondsAvgS = length(condLabelsAvgS);
if nCondsAvgS > 0
    hold on;
    for ci = 1:nCondsAvgS
        tauV = tauAllPerCondAvg{ci};
        jitterAvgS = 0.2 * (rand(size(tauV)) - 0.5);
        scatter(ci + jitterAvgS, tauV, 40, condColorsAvgS(ci,:), 'filled', ...
            'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        plot(ci + [-0.25, 0.25], [median(tauV), median(tauV)], '-', ...
            'Color', condColorsAvgS(ci,:)*0.5, 'LineWidth', 2);
    end
    set(gca, 'YScale', 'log');
    set(gca, 'XTick', 1:nCondsAvgS, 'XTickLabel', condLabelsAvgS);
    ylabel('\tau [s]');
    xlabel('Condition');
    title(sprintf('Stim. Offset Decay \\tau (avg across sizes, Duration: %s)', durationStr));
    grid on;

    % Significance brackets (pairwise Wilcoxon rank-sum)
    bracketY = max(cellfun(@max, tauAllPerCondAvg)) * 1.15;
    bracketStep = bracketY * 0.2;
    bracketCount = 0;
    for ci = 1:nCondsAvgS
        for cj = (ci+1):nCondsAvgS
            [pVal, ~] = ranksum(tauAllPerCondAvg{ci}, tauAllPerCondAvg{cj});
            if pVal < 0.05
                if pVal < 0.001
                    sigStr = '***';
                elseif pVal < 0.01
                    sigStr = '**';
                else
                    sigStr = '*';
                end
                yBracket = bracketY + bracketCount * bracketStep;
                bracketCount = bracketCount + 1;
                line([ci, ci, cj, cj], [yBracket*0.95, yBracket, yBracket, yBracket*0.95], ...
                    'Color', 'k', 'LineWidth', 1);
                text((ci + cj)/2, yBracket * 1.03, sigStr, ...
                    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
            end
        end
    end
    xlim([0.5, nCondsAvgS + 0.5]);
end

savemyfig(figDTS_avg, fullfile(tauOutputPath, 'DecayTauScatter_avg'));
fprintf('  Saved: decay_tau/DecayTauScatter_avg\n');
close(figDTS_avg);

% == Decay Tau Scatter averaged (LINEAR Y-axis) ==
figDTS_avg_lin = figure('Name', sprintf('DecayTauScatter avg linear dur=%d', currentDuration));

if nCondsAvgS > 0
    hold on;
    for ci = 1:nCondsAvgS
        tauV = tauAllPerCondAvg{ci};
        jitterAvgS = 0.2 * (rand(size(tauV)) - 0.5);
        scatter(ci + jitterAvgS, tauV, 40, condColorsAvgS(ci,:), 'filled', ...
            'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        plot(ci + [-0.25, 0.25], [median(tauV), median(tauV)], '-', ...
            'Color', condColorsAvgS(ci,:)*0.5, 'LineWidth', 2);
    end
    set(gca, 'XTick', 1:nCondsAvgS, 'XTickLabel', condLabelsAvgS);
    ylabel('\tau [s]');
    xlabel('Condition');
    title(sprintf('Stim. Offset Decay \\tau (avg across sizes, Duration: %s, linear)', durationStr));
    grid on;

    % Significance brackets (pairwise Wilcoxon rank-sum)
    bracketY = max(cellfun(@max, tauAllPerCondAvg)) * 1.15;
    bracketStep = bracketY * 0.2;
    bracketCount = 0;
    for ci = 1:nCondsAvgS
        for cj = (ci+1):nCondsAvgS
            [pVal, ~] = ranksum(tauAllPerCondAvg{ci}, tauAllPerCondAvg{cj});
            if pVal < 0.05
                if pVal < 0.001
                    sigStr = '***';
                elseif pVal < 0.01
                    sigStr = '**';
                else
                    sigStr = '*';
                end
                yBracket = bracketY + bracketCount * bracketStep;
                bracketCount = bracketCount + 1;
                line([ci, ci, cj, cj], [yBracket*0.95, yBracket, yBracket, yBracket*0.95], ...
                    'Color', 'k', 'LineWidth', 1);
                text((ci + cj)/2, yBracket * 1.03, sigStr, ...
                    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
            end
        end
    end
    xlim([0.5, nCondsAvgS + 0.5]);
end

savemyfig(figDTS_avg_lin, fullfile(tauOutputPath, 'DecayTauScatter_avg_linear'));
fprintf('  Saved: decay_tau/DecayTauScatter_avg_linear\n');
close(figDTS_avg_lin);

% == NLS Decay Tau Scatter (averaged across sizes, LOG Y-axis) ==
figDTS_avg_nls = figure('Name', sprintf('DecayTauScatter NLS avg dur=%d', currentDuration));

condLabelsAvgNLS = {};
tauAllPerCondAvgNLS = {};
condColorsAvgNLS = [];

for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyTau)
        continue;
    end
    tauRawNLS = mean(Metrics.(condition).(metricsKeyTau).decay_tau_nls, 2) * timeScale;
    tauValidNLS = tauRawNLS(~isnan(tauRawNLS) & ~isinf(tauRawNLS));
    if isempty(tauValidNLS), continue; end

    condLabelsAvgNLS{end+1} = condition;
    tauAllPerCondAvgNLS{end+1} = tauValidNLS;
    condColorsAvgNLS(end+1, :) = config.colors.(condition);
end

nCondsAvgNLS = length(condLabelsAvgNLS);
if nCondsAvgNLS > 0
    hold on;
    for ci = 1:nCondsAvgNLS
        tauV = tauAllPerCondAvgNLS{ci};
        jitterAvgNLS = 0.2 * (rand(size(tauV)) - 0.5);
        scatter(ci + jitterAvgNLS, tauV, 40, condColorsAvgNLS(ci,:), 'filled', ...
            'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        plot(ci + [-0.25, 0.25], [median(tauV), median(tauV)], '-', ...
            'Color', condColorsAvgNLS(ci,:)*0.5, 'LineWidth', 2);
    end
    set(gca, 'YScale', 'log');
    set(gca, 'XTick', 1:nCondsAvgNLS, 'XTickLabel', condLabelsAvgNLS);
    ylabel('\tau [s]');
    xlabel('Condition');
    title(sprintf('Stim. Offset Decay \\tau NLS (avg across sizes, Duration: %s)', durationStr));
    grid on;

    % Significance brackets (pairwise Wilcoxon rank-sum)
    bracketY = max(cellfun(@max, tauAllPerCondAvgNLS)) * 1.15;
    bracketStep = bracketY * 0.2;
    bracketCount = 0;
    for ci = 1:nCondsAvgNLS
        for cj = (ci+1):nCondsAvgNLS
            [pVal, ~] = ranksum(tauAllPerCondAvgNLS{ci}, tauAllPerCondAvgNLS{cj});
            if pVal < 0.05
                if pVal < 0.001
                    sigStr = '***';
                elseif pVal < 0.01
                    sigStr = '**';
                else
                    sigStr = '*';
                end
                yBracket = bracketY + bracketCount * bracketStep;
                bracketCount = bracketCount + 1;
                line([ci, ci, cj, cj], [yBracket*0.95, yBracket, yBracket, yBracket*0.95], ...
                    'Color', 'k', 'LineWidth', 1);
                text((ci + cj)/2, yBracket * 1.03, sigStr, ...
                    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
            end
        end
    end
    xlim([0.5, nCondsAvgNLS + 0.5]);
end

savemyfig(figDTS_avg_nls, fullfile(tauOutputPath, 'DecayTauScatter_avg_nls'));
fprintf('  Saved: decay_tau/DecayTauScatter_avg_nls\n');
close(figDTS_avg_nls);

% == NLS Decay Tau Scatter averaged (LINEAR Y-axis) ==
figDTS_avg_nls_lin = figure('Name', sprintf('DecayTauScatter NLS avg linear dur=%d', currentDuration));

if nCondsAvgNLS > 0
    hold on;
    for ci = 1:nCondsAvgNLS
        tauV = tauAllPerCondAvgNLS{ci};
        jitterAvgNLS = 0.2 * (rand(size(tauV)) - 0.5);
        scatter(ci + jitterAvgNLS, tauV, 40, condColorsAvgNLS(ci,:), 'filled', ...
            'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        plot(ci + [-0.25, 0.25], [median(tauV), median(tauV)], '-', ...
            'Color', condColorsAvgNLS(ci,:)*0.5, 'LineWidth', 2);
    end
    set(gca, 'XTick', 1:nCondsAvgNLS, 'XTickLabel', condLabelsAvgNLS);
    ylabel('\tau [s]');
    xlabel('Condition');
    title(sprintf('Stim. Offset Decay \\tau NLS (avg across sizes, Duration: %s, linear)', durationStr));
    grid on;

    % Significance brackets (pairwise Wilcoxon rank-sum)
    bracketY = max(cellfun(@max, tauAllPerCondAvgNLS)) * 1.15;
    bracketStep = bracketY * 0.2;
    bracketCount = 0;
    for ci = 1:nCondsAvgNLS
        for cj = (ci+1):nCondsAvgNLS
            [pVal, ~] = ranksum(tauAllPerCondAvgNLS{ci}, tauAllPerCondAvgNLS{cj});
            if pVal < 0.05
                if pVal < 0.001
                    sigStr = '***';
                elseif pVal < 0.01
                    sigStr = '**';
                else
                    sigStr = '*';
                end
                yBracket = bracketY + bracketCount * bracketStep;
                bracketCount = bracketCount + 1;
                line([ci, ci, cj, cj], [yBracket*0.95, yBracket, yBracket, yBracket*0.95], ...
                    'Color', 'k', 'LineWidth', 1);
                text((ci + cj)/2, yBracket * 1.03, sigStr, ...
                    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
            end
        end
    end
    xlim([0.5, nCondsAvgNLS + 0.5]);
end

savemyfig(figDTS_avg_nls_lin, fullfile(tauOutputPath, 'DecayTauScatter_avg_nls_linear'));
fprintf('  Saved: decay_tau/DecayTauScatter_avg_nls_linear\n');
close(figDTS_avg_nls_lin);

%% Pre-Stim Effects Figures: Per-Size + Averaged (saved to pre_stim_effects/ subfolder)
fprintf('\n--- Generating Pre-Stim Effects Figures ---\n');

preStimOutputPath = fullfile(simPassOutputPath, 'pre_stim_effects');
if ~exist(preStimOutputPath, 'dir'), mkdir(preStimOutputPath); end

% --- Per-size figures ---
for s = 1:length(stimulusSizes)
    stimSize = stimulusSizes(s);

    figPS = figure('Name', sprintf('PreStimEffects size=%d dur=%d', stimSize, currentDuration));

    % Panel A: Baseline Moran's I by condition
    subplot(1, 3, 1);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(PreStimEffects, condition), continue; end
        color = config.colors.(condition);
        moransVals = PreStimEffects.(condition).(stimModeKey).baseline_moransI(:, s);
        moransVals = moransVals(~isnan(moransVals) & moransVals ~= 0);
        if ~isempty(moransVals)
            scatter(c * ones(size(moransVals)) + 0.1*(rand(size(moransVals))-0.5), moransVals, ...
                30, color, 'filled', 'MarkerFaceAlpha', 0.6);
            plot([c-0.3, c+0.3], [mean(moransVals), mean(moransVals)], 'k-', 'LineWidth', 2);
        end
    end
    set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
    ylabel('Baseline Morans I');
    title('A) Spatial Clustering');

    % Panel B: Baseline vs Peak response correlation
    subplot(1, 3, 2);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(PreStimEffects, condition), continue; end
        color = config.colors.(condition);
        corrVals = PreStimEffects.(condition).(stimModeKey).baseline_vs_peak(:, s);
        corrVals = corrVals(~isnan(corrVals));
        if ~isempty(corrVals)
            scatter(c * ones(size(corrVals)) + 0.1*(rand(size(corrVals))-0.5), corrVals, ...
                30, color, 'filled', 'MarkerFaceAlpha', 0.6);
            plot([c-0.3, c+0.3], [mean(corrVals), mean(corrVals)], 'k-', 'LineWidth', 2);
        end
    end
    set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
    ylabel('Correlation (r)');
    yline(0, 'k--');
    title('B) Baseline vs Peak');

    % Panel C: Baseline vs Extent correlation
    subplot(1, 3, 3);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(PreStimEffects, condition), continue; end
        color = config.colors.(condition);
        corrVals = PreStimEffects.(condition).(stimModeKey).baseline_vs_extent(:, s);
        corrVals = corrVals(~isnan(corrVals));
        if ~isempty(corrVals)
            scatter(c * ones(size(corrVals)) + 0.1*(rand(size(corrVals))-0.5), corrVals, ...
                30, color, 'filled', 'MarkerFaceAlpha', 0.6);
            plot([c-0.3, c+0.3], [mean(corrVals), mean(corrVals)], 'k-', 'LineWidth', 2);
        end
    end
    set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
    yline(0, 'k--');
    title('C) Baseline vs Propagation');

    sgtitle(sprintf('Pre-Stimulus State Effects (size=%d, Duration: %s)', stimSize, durationStr));

    psName = sprintf('PreStimEffects_size_%d', stimSize);
    savemyfig(figPS, fullfile(preStimOutputPath, psName));
    fprintf('  Saved: pre_stim_effects/%s\n', psName);
    close(figPS);
end

% --- Size-averaged figure ---
figPS_avg = figure('Name', sprintf('PreStimEffects avg dur=%d', currentDuration));

% Panel A: Baseline Moran's I (averaged across sizes)
subplot(1, 3, 1);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(PreStimEffects, condition), continue; end
    color = config.colors.(condition);
    moransVals = mean(PreStimEffects.(condition).(stimModeKey).baseline_moransI, 2);
    moransVals = moransVals(~isnan(moransVals) & moransVals ~= 0);
    if ~isempty(moransVals)
        scatter(c * ones(size(moransVals)) + 0.1*(rand(size(moransVals))-0.5), moransVals, ...
            30, color, 'filled', 'MarkerFaceAlpha', 0.6);
        plot([c-0.3, c+0.3], [mean(moransVals), mean(moransVals)], 'k-', 'LineWidth', 2);
    end
end
set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
ylabel('Baseline Morans I');
title('A) Spatial Clustering');

% Panel B: Baseline vs Peak (averaged across sizes)
subplot(1, 3, 2);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(PreStimEffects, condition), continue; end
    color = config.colors.(condition);
    corrVals = mean(PreStimEffects.(condition).(stimModeKey).baseline_vs_peak, 2);
    corrVals = corrVals(~isnan(corrVals));
    if ~isempty(corrVals)
        scatter(c * ones(size(corrVals)) + 0.1*(rand(size(corrVals))-0.5), corrVals, ...
            30, color, 'filled', 'MarkerFaceAlpha', 0.6);
        plot([c-0.3, c+0.3], [mean(corrVals), mean(corrVals)], 'k-', 'LineWidth', 2);
    end
end
set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
ylabel('Correlation (r)');
yline(0, 'k--');
title('B) Baseline vs Peak');

% Panel C: Baseline vs Extent (averaged across sizes)
subplot(1, 3, 3);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(PreStimEffects, condition), continue; end
    color = config.colors.(condition);
    corrVals = mean(PreStimEffects.(condition).(stimModeKey).baseline_vs_extent, 2);
    corrVals = corrVals(~isnan(corrVals));
    if ~isempty(corrVals)
        scatter(c * ones(size(corrVals)) + 0.1*(rand(size(corrVals))-0.5), corrVals, ...
            30, color, 'filled', 'MarkerFaceAlpha', 0.6);
        plot([c-0.3, c+0.3], [mean(corrVals), mean(corrVals)], 'k-', 'LineWidth', 2);
    end
end
set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
yline(0, 'k--');
title('C) Baseline vs Propagation');

sgtitle(sprintf('Pre-Stimulus State Effects (avg across sizes, Duration: %s)', durationStr));

savemyfig(figPS_avg, fullfile(preStimOutputPath, 'PreStimEffects_avg'));
fprintf('  Saved: pre_stim_effects/PreStimEffects_avg\n');
close(figPS_avg);

%% Blob Interactions Figures: Per-Size + Averaged (saved to blob_interactions/ subfolder)
fprintf('\n--- Generating Blob Interactions Figures ---\n');

blobOutputPath = fullfile(simPassOutputPath, 'blob_interactions');
if ~exist(blobOutputPath, 'dir'), mkdir(blobOutputPath); end

% --- Per-size figures ---
for s = 1:length(stimulusSizes)
    stimSize = stimulusSizes(s);

    figBI = figure('Name', sprintf('BlobInteractions size=%d dur=%d', stimSize, currentDuration));

    % Panel A: Blob count change vs size (same for every size variant)
    subplot(2, 2, 1);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(BlobInteractions, condition), continue; end
        change = BlobInteractions.(condition).(stimModeKey).blob_count_change;
        meanChange = mean(change, 1);
        semChange = std(change, 0, 1) / sqrt(size(change, 1));
        color = config.colors.(condition);
        errorbar(stimulusSizes, meanChange, semChange, 'o-', ...
            'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
    end
    xlabel('Stimulus Size (pixels)');
    ylabel('\Delta Blob Count');
    yline(0, 'k--');
    title('A) Stimulus Effect on Blob Count');
    legend(conditions, 'Location', 'best');
    grid on;

    % Panel B: Pre-stim vs stim blob count (per-size)
    subplot(2, 2, 2);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(BlobInteractions, condition), continue; end
        prestim = BlobInteractions.(condition).(stimModeKey).prestim_blob_count(:, s);
        stim = BlobInteractions.(condition).(stimModeKey).stim_blob_count(:, s);
        color = config.colors.(condition);
        scatter(prestim, stim, 30, color, 'filled', 'MarkerFaceAlpha', 0.5);
    end
    maxVal = max([xlim ylim]);
    maxVal = ceil(maxVal * 1.1 * 10) / 10;  % 10% padding, round up to 0.1
    plot([0 maxVal], [0 maxVal], 'k:', 'LineWidth', 1);
    xlim([0 maxVal]);
    ylim([0 maxVal]);
    xlabel('Pre-stim Blob Count');
    ylabel('During-stim Blob Count');
    title(sprintf('B) Blob Count: Pre vs During (size=%d)', stimSize));
    legend(conditions, 'Location', 'best');

    % Panel C: Mean blob counts by condition (per-size)
    subplot(2, 2, 3);
    hold on;
    xPos = 1;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(BlobInteractions, condition), continue; end
        color = config.colors.(condition);
        prestim = mean(BlobInteractions.(condition).(stimModeKey).prestim_blob_count(:, s));
        stim = mean(BlobInteractions.(condition).(stimModeKey).stim_blob_count(:, s));
        bar(xPos, prestim, 0.35, 'FaceColor', color, 'EdgeColor', 'k');
        bar(xPos + 0.4, stim, 0.35, 'FaceColor', color, 'FaceAlpha', 0.5, 'EdgeColor', 'k');
        xPos = xPos + 1.2;
    end
    set(gca, 'XTick', 1.2*(0:length(conditions)-1) + 1.2, 'XTickLabel', conditions);
    ylabel('Blob Count');
    title('C) Pre-stim (solid) vs Stim (light)');

    % Panel D: Summary (per-size)
    subplot(2, 2, 4);
    axis off;
    summaryText = {sprintf('\\bf Blob Interaction Summary (size=%d) \\rm', stimSize), ''};
    for c = 1:length(conditions)
        condition = conditions{c};
        if isfield(BlobInteractions, condition)
            prestim = mean(BlobInteractions.(condition).(stimModeKey).prestim_blob_count(:, s));
            stim = mean(BlobInteractions.(condition).(stimModeKey).stim_blob_count(:, s));
            change = mean(BlobInteractions.(condition).(stimModeKey).blob_count_change(:, s));
            summaryText{end+1} = sprintf('%s: Pre=%.1f, Stim=%.1f, \\Delta=%.1f', ...
                condition, prestim, stim, change);
        end
    end
    text(0.1, 0.8, summaryText, 'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 10);

    sgtitle(sprintf('Blob Interaction Analysis (size=%d, Duration: %s)', stimSize, durationStr));

    biName = sprintf('BlobInteractions_size_%d', stimSize);
    savemyfig(figBI, fullfile(blobOutputPath, biName));
    fprintf('  Saved: blob_interactions/%s\n', biName);
    close(figBI);
end

% --- Size-averaged figure ---
figBI_avg = figure('Name', sprintf('BlobInteractions avg dur=%d', currentDuration));

% Panel A: Blob count change vs size (same as main)
subplot(2, 2, 1);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(BlobInteractions, condition), continue; end
    change = BlobInteractions.(condition).(stimModeKey).blob_count_change;
    meanChange = mean(change, 1);
    semChange = std(change, 0, 1) / sqrt(size(change, 1));
    color = config.colors.(condition);
    errorbar(stimulusSizes, meanChange, semChange, 'o-', ...
        'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
end
xlabel('Stimulus Size (pixels)');
ylabel('\Delta Blob Count');
yline(0, 'k--');
title('A) Stimulus Effect on Blob Count');
legend(conditions, 'Location', 'best');
grid on;

% Panel B: Pre-stim vs stim blob count (averaged across sizes)
subplot(2, 2, 2);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(BlobInteractions, condition), continue; end
    prestim = mean(BlobInteractions.(condition).(stimModeKey).prestim_blob_count, 2);
    stim = mean(BlobInteractions.(condition).(stimModeKey).stim_blob_count, 2);
    color = config.colors.(condition);
    scatter(prestim, stim, 30, color, 'filled', 'MarkerFaceAlpha', 0.5);
end
maxVal = max([xlim ylim]);
maxVal = ceil(maxVal * 1.1 * 10) / 10;  % 10% padding, round up to 0.1
plot([0 maxVal], [0 maxVal], 'k:', 'LineWidth', 1);
xlim([0 maxVal]);
ylim([0 maxVal]);
xlabel('Pre-stim Blob Count');
ylabel('During-stim Blob Count');
title('B) Blob Count: Pre vs During (avg)');
legend(conditions, 'Location', 'best');

% Panel C: Mean blob counts by condition (averaged across sizes)
subplot(2, 2, 3);
hold on;
xPos = 1;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(BlobInteractions, condition), continue; end
    color = config.colors.(condition);
    prestim = mean(BlobInteractions.(condition).(stimModeKey).prestim_blob_count, 'all');
    stim = mean(BlobInteractions.(condition).(stimModeKey).stim_blob_count, 'all');
    bar(xPos, prestim, 0.35, 'FaceColor', color, 'EdgeColor', 'k');
    bar(xPos + 0.4, stim, 0.35, 'FaceColor', color, 'FaceAlpha', 0.5, 'EdgeColor', 'k');
    xPos = xPos + 1.2;
end
set(gca, 'XTick', 1.2*(0:length(conditions)-1) + 1.2, 'XTickLabel', conditions);
ylabel('Blob Count');
title('C) Pre-stim (solid) vs Stim (light)');

% Panel D: Summary (averaged across sizes)
subplot(2, 2, 4);
axis off;
summaryText = {'\bf Blob Interaction Summary (avg across sizes) \rm', ''};
for c = 1:length(conditions)
    condition = conditions{c};
    if isfield(BlobInteractions, condition)
        prestim = mean(BlobInteractions.(condition).(stimModeKey).prestim_blob_count, 'all');
        stim = mean(BlobInteractions.(condition).(stimModeKey).stim_blob_count, 'all');
        change = mean(BlobInteractions.(condition).(stimModeKey).blob_count_change, 'all');
        summaryText{end+1} = sprintf('%s: Pre=%.1f, Stim=%.1f, \\Delta=%.1f', ...
            condition, prestim, stim, change);
    end
end
text(0.1, 0.8, summaryText, 'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 10);

sgtitle(sprintf('Blob Interaction Analysis (avg across sizes, Duration: %s)', durationStr));

savemyfig(figBI_avg, fullfile(blobOutputPath, 'BlobInteractions_avg'));
fprintf('  Saved: blob_interactions/BlobInteractions_avg\n');
close(figBI_avg);

%% Recovery Dynamics Figures: Per-Size + Averaged (saved to recovery_dynamics/ subfolder)
fprintf('\n--- Generating Recovery Dynamics Figures ---\n');

metricsKeyRD = stimModeKey;
recoveryOutputPath = fullfile(simPassOutputPath, 'recovery_dynamics');
if ~exist(recoveryOutputPath, 'dir'), mkdir(recoveryOutputPath); end

% --- Per-size figures ---
for s = 1:length(stimulusSizes)
    stimSize = stimulusSizes(s);

    figRD = figure('Name', sprintf('RecoveryDynamics size=%d dur=%d', stimSize, currentDuration));

    % Panel A: Post-stim AUC vs stimulus size (same for every size variant)
    subplot(2, 2, 1);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyRD), continue; end
        auc = Metrics.(condition).(metricsKeyRD).post_stim_auc;
        meanAUC = mean(auc, 1);
        semAUC = std(auc, 0, 1) / sqrt(size(auc, 1));
        color = config.colors.(condition);
        errorbar(stimulusSizes, meanAUC, semAUC, 'o-', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
    end
    xlabel('Stimulus Size (px)');
    ylabel('Post-stim AUC');
    title('A) Post-Stimulus Excess Activity');
    legend(conditions, 'Location', 'northwest', 'FontSize', 7);
    grid on;

    % Panel B: Return-to-baseline time vs stimulus size (same for every size variant)
    subplot(2, 2, 2);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyRD), continue; end
        rtb = Metrics.(condition).(metricsKeyRD).return_to_baseline;
        meanRTB = mean(rtb, 1) * timeScale;
        semRTB = std(rtb, 0, 1) / sqrt(size(rtb, 1)) * timeScale;
        color = config.colors.(condition);
        errorbar(stimulusSizes, meanRTB, semRTB, 'o-', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
    end
    xlabel('Stimulus Size (px)');
    ylabel(sprintf('Return to Baseline (%s)', timeUnit));
    title('B) Recovery Time vs Size');
    grid on;

    % Panel C: Amplification vs half-decay scatter (per-size)
    subplot(2, 2, 3);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyRD), continue; end
        amp = Metrics.(condition).(metricsKeyRD).amplification(:, s);
        halfDecay = Metrics.(condition).(metricsKeyRD).half_decay_time(:, s) * timeScale;
        color = config.colors.(condition);
        scatter(amp, halfDecay, 20, color, 'filled', 'MarkerFaceAlpha', 0.4);
    end
    xlabel('Amplification (Peak/Baseline)');
    ylabel(sprintf('Half-Decay Time (%s)', timeUnit));
    title(sprintf('C) Amplification vs Persistence (size=%d)', stimSize));
    legend(conditions, 'Location', 'best', 'FontSize', 7);
    grid on;

    % Panel D: Decay tau bar+dots (per-size)
    subplot(2, 2, 4);
    hold on;
    xPos = 1;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyRD), continue; end
        tauVals = Metrics.(condition).(metricsKeyRD).decay_tau(:, s) * timeScale;
        tauVals = tauVals(~isnan(tauVals) & ~isinf(tauVals));
        if ~isempty(tauVals)
            color = config.colors.(condition);
            bar(xPos, mean(tauVals), 0.6, 'FaceColor', color, 'EdgeColor', 'k', 'FaceAlpha', 0.7);
            scatter(xPos + 0.15*(rand(size(tauVals))-0.5), tauVals, 25, color, 'filled', 'MarkerFaceAlpha', 0.6);
            errorbar(xPos, mean(tauVals), std(tauVals)/sqrt(length(tauVals)), 'k', 'LineWidth', 1.5);
        end
        xPos = xPos + 1;
    end
    set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
    ylabel(sprintf('Decay \\tau (%s)', timeUnit));
    title(sprintf('D) Decay Tau (size=%d)', stimSize));
    grid on;

    sgtitle(sprintf('Recovery Dynamics Summary (size=%d, Duration: %s)', stimSize, durationStr));

    rdName = sprintf('RecoveryDynamics_size_%d', stimSize);
    savemyfig(figRD, fullfile(recoveryOutputPath, rdName));
    fprintf('  Saved: recovery_dynamics/%s\n', rdName);
    close(figRD);
end

% --- Size-averaged figure ---
figRD_avg = figure('Name', sprintf('RecoveryDynamics avg dur=%d', currentDuration));

% Panel A: Post-stim AUC vs stimulus size (same as main)
subplot(2, 2, 1);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyRD), continue; end
    auc = Metrics.(condition).(metricsKeyRD).post_stim_auc;
    meanAUC = mean(auc, 1);
    semAUC = std(auc, 0, 1) / sqrt(size(auc, 1));
    color = config.colors.(condition);
    errorbar(stimulusSizes, meanAUC, semAUC, 'o-', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
end
xlabel('Stimulus Size (px)');
ylabel('Post-stim AUC');
title('A) Post-Stimulus Excess Activity');
legend(conditions, 'Location', 'northwest', 'FontSize', 7);
grid on;

% Panel B: Return-to-baseline time vs stimulus size (same as main)
subplot(2, 2, 2);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyRD), continue; end
    rtb = Metrics.(condition).(metricsKeyRD).return_to_baseline;
    meanRTB = mean(rtb, 1) * timeScale;
    semRTB = std(rtb, 0, 1) / sqrt(size(rtb, 1)) * timeScale;
    color = config.colors.(condition);
    errorbar(stimulusSizes, meanRTB, semRTB, 'o-', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
end
xlabel('Stimulus Size (px)');
ylabel(sprintf('Return to Baseline (%s)', timeUnit));
title('B) Recovery Time vs Size');
grid on;

% Panel C: Amplification vs half-decay scatter (averaged across sizes)
subplot(2, 2, 3);
hold on;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyRD), continue; end
    amp = mean(Metrics.(condition).(metricsKeyRD).amplification, 2);
    halfDecay = mean(Metrics.(condition).(metricsKeyRD).half_decay_time, 2) * timeScale;
    color = config.colors.(condition);
    scatter(amp, halfDecay, 20, color, 'filled', 'MarkerFaceAlpha', 0.4);
end
xlabel('Amplification (Peak/Baseline)');
ylabel(sprintf('Half-Decay Time (%s)', timeUnit));
title('C) Amplification vs Persistence (avg)');
legend(conditions, 'Location', 'best', 'FontSize', 7);
grid on;

% Panel D: Decay tau bar+dots (averaged across sizes)
subplot(2, 2, 4);
hold on;
xPos = 1;
for c = 1:length(conditions)
    condition = conditions{c};
    if ~isfield(Metrics, condition) || ~isfield(Metrics.(condition), metricsKeyRD), continue; end
    tauVals = mean(Metrics.(condition).(metricsKeyRD).decay_tau, 2) * timeScale;
    tauVals = tauVals(~isnan(tauVals) & ~isinf(tauVals));
    if ~isempty(tauVals)
        color = config.colors.(condition);
        bar(xPos, mean(tauVals), 0.6, 'FaceColor', color, 'EdgeColor', 'k', 'FaceAlpha', 0.7);
        scatter(xPos + 0.15*(rand(size(tauVals))-0.5), tauVals, 25, color, 'filled', 'MarkerFaceAlpha', 0.6);
        errorbar(xPos, mean(tauVals), std(tauVals)/sqrt(length(tauVals)), 'k', 'LineWidth', 1.5);
    end
    xPos = xPos + 1;
end
set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
ylabel(sprintf('Decay \\tau (%s)', timeUnit));
title('D) Decay Tau (avg across sizes)');
grid on;

sgtitle(sprintf('Recovery Dynamics Summary (avg across sizes, Duration: %s)', durationStr));

savemyfig(figRD_avg, fullfile(recoveryOutputPath, 'RecoveryDynamics_avg'));
fprintf('  Saved: recovery_dynamics/RecoveryDynamics_avg\n');
close(figRD_avg);

%% SECTION 8: Save Results for This Duration
fprintf('\n--- Saving Results (Duration: %d) ---\n', currentDuration);

% Save figures to duration subfolder in multiple formats
figNames = {'ActivityTimeCourses', 'ActivityTimeCourses_Crop', 'DoseResponse', 'NetPropagationDoseResponse', 'Persistence', 'GatingThreshold', 'ModeComparison', 'PropagationDistance', 'NetPropagationHeatmap', 'PreStimEffects', 'PropagationDynamics', 'BlobInteractions', 'MoransIDynamics', 'PhasePortrait', 'RecoveryDynamics', 'ConditionRadar'};
allFigs = {fig1, fig1b, fig2, fig2b, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13, fig14};
skipMask = cellfun(@isempty, allFigs);
figNames(skipMask) = [];
figs = [allFigs{~skipMask}];

if simPass == 2
    figNames{end+1} = 'ActivityTimeCourses_Clamped_Focused';
    figs(end+1) = fig1c;
    figNames{end+1} = 'ActivityTimeCourses_Clamped_Focused_Crop';
    figs(end+1) = fig1d;
end

if simPass == 1
    simPassOutputPath = durOutputPath;
else
    simPassOutputPath = fullfile(durOutputPath, 'bestMatch');
    if ~exist(simPassOutputPath, 'dir'), mkdir(simPassOutputPath); end
end

for i = 1:length(figs)
    basePath = fullfile(simPassOutputPath, figNames{i});
    savemyfig(figs(i), basePath);
    fprintf('  Saved: %s (.pdf, .png, .fig)\n', figNames{i});
    % Close figure immediately after saving to free memory
    if ishandle(figs(i))
        close(figs(i));
    end
end

if simPass == 1
    % Store metrics for cross-duration comparison (all-sims only)
    % NOTE: Data is NOT accumulated — it's already saved to per-duration .mat files
    AllDurationMetrics.(durKey) = Metrics;
    AllDurationGating.(durKey) = Gating;
    AllDurationStats.(durKey) = Stats;

    % Store new analysis structures for cross-duration comparison
    if ~exist('AllDurationPreStimEffects', 'var')
        AllDurationPreStimEffects = struct();
    end
    if ~exist('AllDurationPropDynamics', 'var')
        AllDurationPropDynamics = struct();
    end
    if ~exist('AllDurationBlobInteractions', 'var')
        AllDurationBlobInteractions = struct();
    end
    if ~exist('AllDurationMoransIDynamics', 'var')
        AllDurationMoransIDynamics = struct();
    end
    AllDurationPreStimEffects.(durKey) = PreStimEffects;
    AllDurationPropDynamics.(durKey) = PropDynamics;
    AllDurationBlobInteractions.(durKey) = BlobInteractions;
    AllDurationMoransIDynamics.(durKey) = MoransIDynamics;
elseif simPass == 2
    % Store best-match data for cross-duration comparison
    % NOTE: Data is NOT accumulated — it's already saved to per-duration bestMatch/ .mat files
    AllDurationMetrics_BM.(durKey) = Metrics;
    AllDurationGating_BM.(durKey) = Gating;
    AllDurationStats_BM.(durKey) = Stats;
    if ~exist('AllDurationPreStimEffects_BM', 'var')
        AllDurationPreStimEffects_BM = struct();
    end
    if ~exist('AllDurationPropDynamics_BM', 'var')
        AllDurationPropDynamics_BM = struct();
    end
    if ~exist('AllDurationBlobInteractions_BM', 'var')
        AllDurationBlobInteractions_BM = struct();
    end
    if ~exist('AllDurationMoransIDynamics_BM', 'var')
        AllDurationMoransIDynamics_BM = struct();
    end
    AllDurationPreStimEffects_BM.(durKey) = PreStimEffects;
    AllDurationPropDynamics_BM.(durKey) = PropDynamics;
    AllDurationBlobInteractions_BM.(durKey) = BlobInteractions;
    AllDurationMoransIDynamics_BM.(durKey) = MoransIDynamics;
end

% Save duration-specific analysis (including new structures)
analysisFile = fullfile(simPassOutputPath, 'PerturbationAnalysis.mat');
save(analysisFile, 'Data', 'Metrics', 'Gating', 'Stats', 'config', ...
    'stimulusSizes', 'stimulusModes', 'conditions', ...
    'preStimFrames', 'stimOnFrames', 'postStimFrames', 'timeVec', 'currentDuration', ...
    'PreStimEffects', 'PropDynamics', 'BlobInteractions', 'MoransIDynamics', '-v7.3');
fprintf('  Saved analysis: %s\n', analysisFile);

end  % End of simPass loop

% Free per-duration temporaries to reduce memory accumulation
clear DataFull Data Metrics Gating Stats PreStimEffects PropDynamics BlobInteractions MoransIDynamics;

end  % End of main duration loop
hasRawData = exist(fullfile(config.outputPath, sprintf('dur_%d', stimulusDurations(1)), 'PerturbationAnalysis.mat'), 'file') == 2;

else  % config.figuresOnly == true
%% figuresOnly: load cached all-duration summary
summaryFile = fullfile(config.outputPath, 'AllDurationAnalysis.mat');
if ~exist(summaryFile, 'file')
    error('figuresOnly=true but %s not found. Run full analysis first.', summaryFile);
end
fprintf('=== figuresOnly mode: loading %s ===\n', summaryFile);
savedOutputPath = config.outputPath;
savedShowRealTime = config.showRealTime;
savedSamplingRate = config.samplingRate;
savedColors = config.colors;
savedConditions = config.conditions;
savedStimMode = config.stimMode;
savedTargetDurationSec = config.targetDurationSec;
savedCollapseTargetDurationsSec = config.collapseTargetDurationsSec;
load(summaryFile);
% Restore fields that the loaded config doesn't have or that we need to override
config.figuresOnly = true;
config.outputPath = savedOutputPath;
config.showRealTime = savedShowRealTime;
config.samplingRate = savedSamplingRate;
config.colors = savedColors;
config.conditions = savedConditions;
config.stimMode = savedStimMode;
config.targetDurationSec          = savedTargetDurationSec;
config.collapseTargetDurationsSec = savedCollapseTargetDurationsSec;
% Strip bias suffix (e.g. 'double_pulse_bias10_2p00' → 'doublepulsebias10')
% so storage lookups Data.<cond>.<stimModeKey> hit the raw modeKey set at
% line 1083 by the per-raw-mode loop. Without this, Section 9's
% propagation-dynamics figures crash with 'Unrecognized field name
% "doublepulsebias102p00"' for bias-encoded calls (Fig6h/EC50 path).
stimModeKey = strrep(rawModeFromDisplayMode(config.stimMode), '_', '');

% Check if per-duration .mat files exist for lazy-loading raw data
hasRawData = exist(fullfile(config.outputPath, sprintf('dur_%d', stimulusDurations(1)), 'PerturbationAnalysis.mat'), 'file') == 2;
if ~hasRawData
    fprintf('  Note: Per-duration .mat files not found — figures needing raw time-series will be skipped.\n');
end

% Derive variables that SECTION 9 needs
nConditions = length(conditions);
nSizes = length(stimulusSizes);
nDurations = length(stimulusDurations);

% Display modes (normally set after SECTION 2)
metricsDisplayModes = {};
for mm = 1:length(stimulusModes)
    metricsDisplayModes = [metricsDisplayModes, displayModesForMode(stimulusModes{mm})];
end

% Filter to selected stimulus mode (bias-encoded modes collapse to the raw mode
% after per-invocation slicing in SECTION 3).
if isBiasEncodedMode(config.stimMode)
    metricsDisplayModes = {rawModeFromDisplayMode(config.stimMode)};
elseif isBiasMode(config.stimMode)
    metricsDisplayModes = metricsDisplayModes(ismember(metricsDisplayModes, displayModesForMode(config.stimMode)));
else
    metricsDisplayModes = metricsDisplayModes(ismember(metricsDisplayModes, {config.stimMode}));
end
nDisplayModes = length(metricsDisplayModes);

% Bias values (normally set in SECTION 2)
if isfield(config, 'stimulusBiasValues')
    stimulusBiasValues = config.stimulusBiasValues;
    lowBiasValue = 0.25;
    [~, lowBiasIdx] = min(abs(stimulusBiasValues - lowBiasValue));
elseif exist('stimulusBiasValues', 'var') && ~isempty(stimulusBiasValues)
    lowBiasValue = 0.25;
    [~, lowBiasIdx] = min(abs(stimulusBiasValues - lowBiasValue));
else
    stimulusBiasValues = [];
    lowBiasIdx = 1;
end

% Time display variables (normally set inside the per-duration loop)
if config.showRealTime
    timeScale = config.globalMeanSF / config.samplingRate;
    timeUnit = 's';
    timeLabel = 'Time (s from stim onset)';
    velocityScale = config.samplingRate / config.globalMeanSF;
    velocityUnit = 'px/s';
else
    timeScale = 1;
    timeUnit = 'frames';
    timeLabel = 'Time (frames from stim onset)';
    velocityScale = 1;
    velocityUnit = 'px/frame';
end

end  % if ~config.figuresOnly

% Save originals for cross-duration pass loop
AllDurationMetrics_AllSims = AllDurationMetrics;
AllDurationGating_AllSims = AllDurationGating;
AllDurationStats_AllSims = AllDurationStats;
AllDurationPreStimEffects_AllSims = AllDurationPreStimEffects;
AllDurationPropDynamics_AllSims = AllDurationPropDynamics;
AllDurationBlobInteractions_AllSims = AllDurationBlobInteractions;
AllDurationMoransIDynamics_AllSims = AllDurationMoransIDynamics;
hasRawData_AllSims = hasRawData;

for crossDurPass = 1:2
if crossDurPass == 2
    if ~exist('AllDurationMetrics_BM', 'var') || isempty(fieldnames(AllDurationMetrics_BM))
        fprintf('Skipping best-match cross-duration pass (no BM data)\n');
        continue;
    end
    fprintf('\n=== Best-Match Cross-Duration Pass ===\n');
    AllDurationMetrics = AllDurationMetrics_BM;
    AllDurationGating = AllDurationGating_BM;
    AllDurationStats = AllDurationStats_BM;
    AllDurationPreStimEffects = AllDurationPreStimEffects_BM;
    AllDurationPropDynamics = AllDurationPropDynamics_BM;
    AllDurationBlobInteractions = AllDurationBlobInteractions_BM;
    AllDurationMoransIDynamics = AllDurationMoransIDynamics_BM;
    hasRawData = exist(fullfile(config.outputPath, sprintf('dur_%d', stimulusDurations(1)), 'bestMatch', 'PerturbationAnalysis.mat'), 'file') == 2;
end

%% SECTION 9: Cross-Duration Comparison Figures
fprintf('\n========================================\n');
fprintf('=== Creating Cross-Duration Comparisons ===\n');
fprintf('========================================\n');

% Create DurationComparison subfolder
if crossDurPass == 2
    comparisonPath = fullfile(config.outputPath, 'DurationComparison', 'bestMatch');
else
    comparisonPath = fullfile(config.outputPath, 'DurationComparison');
end
if ~exist(comparisonPath, 'dir')
    mkdir(comparisonPath);
end

%% Figure: EC50 Bar Plot
figEC50 = figure('Name', 'EC50 Comparison');

nDurations = length(stimulusDurations);
nConditions = length(conditions);
barWidth = 0.35;

for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};
    metricsKey = strrep(displayMode, '_', '');

    subplot(1, nDisplayModes, m);
    hold on;

    % Collect EC50 values [conditions x durations]
    EC50_matrix = zeros(nConditions, nDurations);
    for d = 1:nDurations
        dKey = sprintf('dur_%d', stimulusDurations(d));
        for c = 1:nConditions
            condition = conditions{c};
            if isfield(AllDurationGating.(dKey), condition) && isfield(AllDurationGating.(dKey).(condition), metricsKey)
                EC50_matrix(c, d) = AllDurationGating.(dKey).(condition).(metricsKey).EC50;
            else
                EC50_matrix(c, d) = NaN;
            end
        end
    end

    % Grouped bar plot
    b = bar(EC50_matrix);
    for d = 1:nDurations
        b(d).FaceColor = 'flat';
        cmap = parula(nDurations);
        b(d).CData = repmat(cmap(d,:), nConditions, 1);
    end

    set(gca, 'XTick', 1:nConditions, 'XTickLabel', conditions);
    ylabel('EC50 (stimulus size)');
    title(sprintf('EC50 by Condition: %s', strrep(displayMode, '_', ' ')));
    if config.showRealTime
        legend(arrayfun(@(x) sprintf('%.1f s', x * config.globalMeanSF / config.samplingRate), stimulusDurations, 'UniformOutput', false), 'Location', 'best');
    else
        legend(arrayfun(@(x) sprintf('%d frames', x), stimulusDurations, 'UniformOutput', false), 'Location', 'best');
    end
    grid on;
end

sgtitle('Half-Maximal Response Threshold (EC50)');
savemyfig(figEC50, fullfile(comparisonPath, 'EC50_BarPlot'));
fprintf('  Saved: EC50_BarPlot (.pdf, .png, .fig)\n');
close(figEC50);

%% Figure: Hill Coefficient Plot
figHill = figure('Name', 'Hill Coefficient Comparison');

for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};
    metricsKey = strrep(displayMode, '_', '');

    subplot(1, nDisplayModes + 1, m);
    hold on;

    % Collect Hill coefficients [conditions x durations]
    Hill_matrix = zeros(nConditions, nDurations);
    for d = 1:nDurations
        dKey = sprintf('dur_%d', stimulusDurations(d));
        for c = 1:nConditions
            condition = conditions{c};
            if isfield(AllDurationGating.(dKey), condition) && isfield(AllDurationGating.(dKey).(condition), metricsKey)
                Hill_matrix(c, d) = AllDurationGating.(dKey).(condition).(metricsKey).hill_coefficient;
            else
                Hill_matrix(c, d) = NaN;
            end
        end
    end

    % Grouped bar plot
    b = bar(Hill_matrix);
    for d = 1:nDurations
        b(d).FaceColor = 'flat';
        cmap = parula(nDurations);
        b(d).CData = repmat(cmap(d,:), nConditions, 1);
    end

    set(gca, 'XTick', 1:nConditions, 'XTickLabel', conditions);
    ylabel('Hill Coefficient (n)');
    title(sprintf('Hill Coefficient: %s', strrep(displayMode, '_', ' ')));
    if config.showRealTime
        legend(arrayfun(@(x) sprintf('%.1f s', x * config.globalMeanSF / config.samplingRate), stimulusDurations, 'UniformOutput', false), 'Location', 'best');
    else
        legend(arrayfun(@(x) sprintf('%d frames', x), stimulusDurations, 'UniformOutput', false), 'Location', 'best');
    end
    grid on;
end

% Interpretation table panel
subplot(1, nDisplayModes + 1, nDisplayModes + 1);
axis off;
hold on;

% Create interpretation text
interpText = {
    '\bf Hill n \rm', '\bf Interpretation \rm';
    'n = 1', {'Gradual, hyperbolic response', '(no cooperativity)'};
    'n > 1', {'Steep, switch-like - small size', 'changes cause big response changes'};
    'n < 1', {'Shallow - response changes', 'gradually across sizes'}
};

% Draw table
tableTop = 0.85;
rowHeight = 0.2;
col1X = 0.05;
col2X = 0.25;

for row = 1:size(interpText, 1)
    yPos = tableTop - (row-1) * rowHeight;
    text(col1X, yPos, interpText{row, 1}, 'FontSize', 10, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');
    text(col2X, yPos, interpText{row, 2}, 'FontSize', 9, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');
end

% Add horizontal lines
yLine1 = tableTop - 0.5*rowHeight + 0.02;
plot([col1X-0.02, 0.98], [yLine1, yLine1], 'k-', 'LineWidth', 1);

title('Interpretation');
xlim([0 1]);
ylim([0 1]);

sgtitle('Dose-Response Steepness (Hill Coefficient)');
savemyfig(figHill, fullfile(comparisonPath, 'HillCoefficient'));
fprintf('  Saved: HillCoefficient (.pdf, .png, .fig)\n');
close(figHill);

%% Figure: Net Propagation Heatmap - All Durations
figHeatmapAll = figure('Name', 'Net Propagation All Durations');

% Compute stimulus radii
stimulusRadii = stimulusSizes / 2;

% Get global color limits using NET propagation
allNetPropValues = [];
for d = 1:nDurations
    dKey = sprintf('dur_%d', stimulusDurations(d));
    for c = 1:nConditions
        condition = conditions{c};
        if ~isfield(AllDurationMetrics.(dKey), condition)
            continue;
        end
        for m = 1:nDisplayModes
            metricsKey = strrep(metricsDisplayModes{m}, '_', '');
            if isfield(AllDurationMetrics.(dKey).(condition), metricsKey)
                for s = 1:length(stimulusSizes)
                    netProp = AllDurationMetrics.(dKey).(condition).(metricsKey).max_blob_extent(:, s) - stimulusRadii(s);
                    allNetPropValues = [allNetPropValues; netProp(:)];
                end
            end
        end
    end
end
% Symmetric color limits for diverging colormap
maxAbsVal = max(abs(allNetPropValues));
if isempty(maxAbsVal) || ~isfinite(maxAbsVal) || maxAbsVal == 0
    maxAbsVal = 1;  % fallback so imagesc(X, [low high]) doesn't crash on
                    % composite caches that lack max_blob_extent
end
globalClims = [-maxAbsVal, maxAbsVal];

% nDisplayModes rows x nDurations columns
for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};
    metricsKey = strrep(displayMode, '_', '');

    for d = 1:nDurations
        dKey = sprintf('dur_%d', stimulusDurations(d));

        subplot(nDisplayModes, nDurations, (m-1)*nDurations + d);

        % Build NET propagation matrix
        propMatrix = zeros(nConditions, length(stimulusSizes));
        for c = 1:nConditions
            condition = conditions{c};
            if ~isfield(AllDurationMetrics.(dKey), condition) || ~isfield(AllDurationMetrics.(dKey).(condition), metricsKey)
                continue;
            end
            for s = 1:length(stimulusSizes)
                meanExtent = mean(AllDurationMetrics.(dKey).(condition).(metricsKey).max_blob_extent(:, s), 'all');
                propMatrix(c, s) = meanExtent - stimulusRadii(s);
            end
        end

        imagesc(propMatrix, globalClims);
        colormap(redblue(256));
        if d == nDurations
            cb = colorbar;
            cb.Label.String = 'Net Prop. (px)';
        end

        set(gca, 'XTick', 1:length(stimulusSizes), 'XTickLabel', stimulusSizes);
        set(gca, 'YTick', 1:nConditions, 'YTickLabel', conditions);

        if m == nDisplayModes
            xlabel('Size (px)');
        end
        if d == 1
            ylabel(sprintf('%s', strrep(displayMode, '_', ' ')));
        end
        if config.showRealTime
            title(sprintf('%.1f s', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));
        else
            title(sprintf('%d frames', stimulusDurations(d)));
        end
    end
end

sgtitle('Net Propagation Beyond Stimulus (All Durations)');
savemyfig(figHeatmapAll, fullfile(comparisonPath, 'NetPropagationHeatmap_AllDurations'));
fprintf('  Saved: NetPropagationHeatmap_AllDurations (.pdf, .png, .fig)\n');
close(figHeatmapAll);

%% Figure: EC50 vs Duration
figEC50vsDur = figure('Name', 'EC50 vs Duration');

for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};
    metricsKey = strrep(displayMode, '_', '');

    subplot(1, nDisplayModes, m);
    hold on;

    legendHandles = gobjects(nConditions, 1);
    for c = 1:nConditions
        condition = conditions{c};
        color = config.colors.(condition);

        EC50_vals = zeros(1, nDurations);
        EC50_sem = zeros(1, nDurations);
        for d = 1:nDurations
            dKey = sprintf('dur_%d', stimulusDurations(d));
            if isfield(AllDurationGating.(dKey), condition) && isfield(AllDurationGating.(dKey).(condition), metricsKey)
                EC50_vals(d) = AllDurationGating.(dKey).(condition).(metricsKey).EC50;
                if isfield(AllDurationGating.(dKey).(condition).(metricsKey), 'EC50_perSeed')
                    perSeed = AllDurationGating.(dKey).(condition).(metricsKey).EC50_perSeed;
                    perSeed = perSeed(~isnan(perSeed));
                    EC50_vals(d) = mean(perSeed);
                    EC50_sem(d) = std(perSeed) / sqrt(length(perSeed));
                end
            else
                EC50_vals(d) = NaN;
                EC50_sem(d) = NaN;
            end
        end

        if config.showRealTime
            xVals = stimulusDurations * config.globalMeanSF / config.samplingRate;
        else
            xVals = stimulusDurations;
        end

        % Shaded CI (mean ± SEM)
        valid = ~isnan(EC50_vals) & ~isnan(EC50_sem);
        if any(valid)
            xFill = [xVals(valid), fliplr(xVals(valid))];
            yFill = [EC50_vals(valid) + EC50_sem(valid), fliplr(EC50_vals(valid) - EC50_sem(valid))];
            fill(xFill, yFill, color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        end
        legendHandles(c) = plot(xVals, EC50_vals, 'o-', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 2);
    end

    xlabel(sprintf('Stimulus Duration (%s)', timeUnit));
    ylabel('EC50 (stimulus size)');
    title(sprintf('EC50 vs Duration: %s', strrep(displayMode, '_', ' ')));
    legend(legendHandles, conditions, 'Location', 'best');
    grid on;
end

sgtitle('Sensitivity Changes with Stimulus Duration');
savemyfig(figEC50vsDur, fullfile(comparisonPath, 'EC50_vs_Duration'));
fprintf('  Saved: EC50_vs_Duration (.pdf, .png, .fig)\n');
close(figEC50vsDur);

%% Figure: EC50 vs Duration (area, log scale)
figEC50area = figure('Name', 'EC50 vs Duration (area)');

for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};
    metricsKey = strrep(displayMode, '_', '');

    subplot(1, nDisplayModes, m);
    hold on;

    legendHandles = gobjects(nConditions, 1);
    for c = 1:nConditions
        condition = conditions{c};
        color = config.colors.(condition);

        EC50_area_mean = zeros(1, nDurations);
        EC50_area_sem = zeros(1, nDurations);
        for d = 1:nDurations
            dKey = sprintf('dur_%d', stimulusDurations(d));
            if isfield(AllDurationGating.(dKey), condition) && isfield(AllDurationGating.(dKey).(condition), metricsKey)
                if isfield(AllDurationGating.(dKey).(condition).(metricsKey), 'EC50_perSeed')
                    perSeed = AllDurationGating.(dKey).(condition).(metricsKey).EC50_perSeed;
                    perSeed = perSeed(~isnan(perSeed));
                    perSeedArea = perSeed .^ 2;
                    EC50_area_mean(d) = mean(perSeedArea);
                    EC50_area_sem(d) = std(perSeedArea) / sqrt(length(perSeedArea));
                else
                    EC50_area_mean(d) = AllDurationGating.(dKey).(condition).(metricsKey).EC50 ^ 2;
                    EC50_area_sem(d) = NaN;
                end
            else
                EC50_area_mean(d) = NaN;
                EC50_area_sem(d) = NaN;
            end
        end

        if config.showRealTime
            xVals = stimulusDurations * config.globalMeanSF / config.samplingRate;
        else
            xVals = stimulusDurations;
        end

        % Shaded CI (mean ± SEM)
        valid = ~isnan(EC50_area_mean) & ~isnan(EC50_area_sem);
        if any(valid)
            xFill = [xVals(valid), fliplr(xVals(valid))];
            yFill = [EC50_area_mean(valid) + EC50_area_sem(valid), fliplr(EC50_area_mean(valid) - EC50_area_sem(valid))];
            fill(xFill, yFill, color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        end
        legendHandles(c) = plot(xVals, EC50_area_mean, 'o-', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 2);
    end

    set(gca, 'YScale', 'log');
    xlabel(sprintf('Stimulus Duration (%s)', timeUnit));
    ylabel('EC50 (stimulus area, pixels)');
    title(sprintf('EC50 vs Duration: %s', strrep(displayMode, '_', ' ')));
    legend(legendHandles, conditions, 'Location', 'best');
    grid on;
end

sgtitle('Sensitivity Changes with Stimulus Duration (area)');
savemyfig(figEC50area, fullfile(comparisonPath, 'EC50_vs_Duration_area'));
fprintf('  Saved: EC50_vs_Duration_area (.pdf, .png, .fig)\n');
close(figEC50area);

%% =====================================================================
%% EC50 area: single-duration view + collapse-across-durations view
%% =====================================================================
durationsSec = stimulusDurations * config.globalMeanSF / config.samplingRate;

% --- Single-duration view: pick the duration closest to targetDurationSec
[~, dIdxA] = min(abs(durationsSec - config.targetDurationSec));
realT = durationsSec(dIdxA);
dKeyA = sprintf('dur_%d', stimulusDurations(dIdxA));
fprintf('\n--- EC50 single-duration view: closest duration to %.2f s = %.2f s (%d frames, %s)\n', ...
    config.targetDurationSec, realT, stimulusDurations(dIdxA), dKeyA);

% --- Collapse view: indices for collapse list (closest match per target)
collapseIdxList = zeros(1, numel(config.collapseTargetDurationsSec));
for ti = 1:numel(config.collapseTargetDurationsSec)
    [~, collapseIdxList(ti)] = min(abs(durationsSec - config.collapseTargetDurationsSec(ti)));
end
collapseIdxList = unique(collapseIdxList, 'stable');
fprintf('--- EC50 collapse view: collapsing across durations (s): %s\n', ...
    mat2str(round(durationsSec(collapseIdxList), 2)));

% --- Metric-key resolution: prefer exact match, fall back to digit-stripped
% (e.g. cached file uses 'doublepulse' even when stimMode is 'double_pulse10').
ec50Modes = {};
if exist('metricsDisplayModes', 'var') && ~isempty(metricsDisplayModes)
    ec50Modes = metricsDisplayModes;
else
    ec50Modes = {config.stimMode};
end
ec50ModeKeys = cell(1, numel(ec50Modes));
sampleDur    = sprintf('dur_%d', stimulusDurations(dIdxA));
sampleNode   = struct();
if isfield(AllDurationGating, sampleDur)
    sampleConds = fieldnames(AllDurationGating.(sampleDur));
    for cc = 1:numel(sampleConds)
        if isstruct(AllDurationGating.(sampleDur).(sampleConds{cc}))
            sampleNode = AllDurationGating.(sampleDur).(sampleConds{cc});
            break;
        end
    end
end
availKeys = {};
if ~isempty(fieldnames(sampleNode))
    availKeys = fieldnames(sampleNode);
end
for mm = 1:numel(ec50Modes)
    cand = strrep(ec50Modes{mm}, '_', '');
    if ismember(cand, availKeys)
        ec50ModeKeys{mm} = cand;
    else
        stripped = regexprep(cand, '\d+$', '');
        if ismember(stripped, availKeys)
            ec50ModeKeys{mm} = stripped;
            fprintf('  EC50 panels: metric key "%s" not in cache; falling back to "%s".\n', cand, stripped);
        else
            ec50ModeKeys{mm} = cand;  % keep so downstream isfield checks return false cleanly
        end
    end
end
ec50nModes = numel(ec50Modes);

for m = 1:ec50nModes
    displayMode = ec50Modes{m};
    metricsKey = ec50ModeKeys{m};
    modeSuffix = strrep(displayMode, '_', '-');

    % Gather per-condition pooled values for both variants
    perCondAtTarget = cell(1, nConditions);   % Variant A: EC50_perSeed.^2 at 2 s
    perCondPooled   = cell(1, nConditions);   % Variant B: pooled across collapse durations

    sawPerSeed = false;
    usedEC50Fallback = false;
    for c = 1:nConditions
        condition = conditions{c};

        % Variant A: per-seed if available, else fall back to single EC50
        perCondAtTarget{c} = [];
        if isfield(AllDurationGating, dKeyA) && ...
                isfield(AllDurationGating.(dKeyA), condition) && ...
                isfield(AllDurationGating.(dKeyA).(condition), metricsKey)
            node = AllDurationGating.(dKeyA).(condition).(metricsKey);
            if isfield(node, 'EC50_perSeed') && ~isempty(node.EC50_perSeed)
                ps = node.EC50_perSeed;
                ps = ps(~isnan(ps));
                perCondAtTarget{c} = ps(:).^2;
                sawPerSeed = true;
            elseif isfield(node, 'EC50') && isfinite(node.EC50)
                perCondAtTarget{c} = node.EC50.^2;
                usedEC50Fallback = true;
            end
        end

        % Variant B: pool across collapse durations, per-seed if available
        % else fall back to one squared EC50 per duration.
        pooled = [];
        for ii = 1:numel(collapseIdxList)
            dKeyP = sprintf('dur_%d', stimulusDurations(collapseIdxList(ii)));
            if ~isfield(AllDurationGating, dKeyP) || ...
                    ~isfield(AllDurationGating.(dKeyP), condition) || ...
                    ~isfield(AllDurationGating.(dKeyP).(condition), metricsKey)
                continue;
            end
            node = AllDurationGating.(dKeyP).(condition).(metricsKey);
            if isfield(node, 'EC50_perSeed') && ~isempty(node.EC50_perSeed)
                ps = node.EC50_perSeed;
                ps = ps(~isnan(ps));
                pooled = [pooled; ps(:).^2]; %#ok<AGROW>
                sawPerSeed = true;
            elseif isfield(node, 'EC50') && isfinite(node.EC50)
                pooled = [pooled; node.EC50.^2]; %#ok<AGROW>
                usedEC50Fallback = true;
            end
        end
        perCondPooled{c} = pooled;
    end
    if usedEC50Fallback && ~sawPerSeed
        fprintf('  [%s] No EC50_perSeed in cache — using squared median EC50. Variant A will show 1 dot/condition; Variant B STD is across %d duration values.\n', ...
            displayMode, numel(collapseIdxList));
    elseif usedEC50Fallback
        fprintf('  [%s] Mixed cache (some conditions/durations missing EC50_perSeed) — falling back to squared EC50 where needed.\n', displayMode);
    end

    %% --- Single-duration figure 1: swarm of EC50^2 at target duration ---
    %% Rendered twice: once with linear y-axis, once with log y-axis.
    for scaleType = {'linear', 'log'}
        yscale = scaleType{1};
        figA1 = figure('Name', sprintf('EC50 area at %.1fs swarm %s (%s)', realT, yscale, displayMode));
        hold on;
        rng(0, 'twister');
        for c = 1:nConditions
            y = perCondAtTarget{c};
            if isempty(y), continue; end
            x = c + (rand(numel(y), 1) - 0.5) * 0.35;
            scatter(x, y, 28, config.colors.(conditions{c}), 'filled', ...
                'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'none');
            mu = mean(y);
            sem = std(y) / sqrt(numel(y));
            plot([c-0.3 c+0.3], [mu mu], '-', 'Color', config.colors.(conditions{c}), 'LineWidth', 2);
            plot([c c], [mu-sem mu+sem], '-', 'Color', config.colors.(conditions{c}), 'LineWidth', 1.5);
        end
        set(gca, 'XTick', 1:nConditions, 'XTickLabel', conditions, 'YScale', yscale);
        ylabel('EC50 area (pixels^2)');
        title(sprintf('EC50 area at %.0f s', realT));
        grid on;
        savemyfig(figA1, fullfile(comparisonPath, sprintf('EC50_area_at_2s_swarm_%s_%s', yscale, modeSuffix)));
        fprintf('  Saved: EC50_area_at_2s_swarm_%s_%s (.pdf, .png, .fig)\n', yscale, modeSuffix);
        close(figA1);
    end

    %% --- Single-duration figure 2: 4-point mean+/-SEM dots at target ---
    %% Rendered twice: once with linear y-axis, once with log y-axis.
    for scaleType = {'linear', 'log'}
        yscale = scaleType{1};
        figA2 = figure('Name', sprintf('EC50 area at %.1fs dots %s (%s)', realT, yscale, displayMode));
        hold on;
        for c = 1:nConditions
            y = perCondAtTarget{c};
            if isempty(y), continue; end
            mu = mean(y);
            sem = std(y) / sqrt(numel(y));
            errorbar(c, mu, sem, 'o', 'Color', config.colors.(conditions{c}), ...
                'MarkerFaceColor', config.colors.(conditions{c}), 'LineWidth', 2, 'MarkerSize', 8);
        end
        set(gca, 'XTick', 1:nConditions, 'XTickLabel', conditions, 'YScale', yscale);
        xlim([0.5 nConditions + 0.5]);
        ylabel('EC50 area (pixels^2)');
        title(sprintf('EC50 area at %.0f s', realT));
        grid on;
        savemyfig(figA2, fullfile(comparisonPath, sprintf('EC50_area_at_2s_dots_%s_%s', yscale, modeSuffix)));
        fprintf('  Saved: EC50_area_at_2s_dots_%s_%s (.pdf, .png, .fig)\n', yscale, modeSuffix);
        close(figA2);
    end

    %% --- Single-duration figure 3: boxplot at target duration ---
    %% Rendered twice: once with linear y-axis, once with log y-axis.
    %% A third variant (_box_stats_*) overlays pairwise ranksum brackets.
    for scaleType = {'linear', 'log'}
        yscale = scaleType{1};
        for withStats = [false, true]
            stemTag = 'box'; if withStats, stemTag = 'box_stats'; end
            figA3 = figure('Name', sprintf('EC50 area at %.1fs %s %s (%s)', realT, stemTag, yscale, displayMode));
            hold on;
            allY = [];
            for c = 1:nConditions
                y = perCondAtTarget{c};
                if isempty(y), continue; end
                col = config.colors.(conditions{c});
                bc = boxchart(c * ones(numel(y), 1), y(:), 'BoxWidth', 0.55, 'MarkerStyle', 'none');
                bc.BoxFaceColor     = col;
                bc.BoxFaceAlpha     = 0.45;
                bc.BoxEdgeColor     = col * 0.6;
                bc.WhiskerLineColor = col * 0.6;
                bc.LineWidth        = 1.4;
                rng(0, 'twister');
                xj = c + (rand(numel(y), 1) - 0.5) * 0.25;
                scatter(xj, y, 22, col, 'filled', 'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor', 'none');
                allY = [allY; y(:)]; %#ok<AGROW>
            end
            set(gca, 'XTick', 1:nConditions, 'XTickLabel', conditions, 'YScale', yscale);
            xlim([0.5 nConditions + 0.5]);
            ylabel('EC50 area (pixels^2)');
            title(sprintf('EC50 area at %.0f s', realT));
            grid on;
            if withStats && ~isempty(allY)
                % Stack pairwise ranksum brackets above the boxes. All 6 pairs
                % for 4 conditions, ordered adjacent-first then long-range, so
                % the bottom bracket touches the closest neighbour pair.
                pairs = [1 2; 2 3; 3 4; 1 3; 2 4; 1 4];
                yBase = max(allY) * 1.05;
                if strcmp(yscale, 'log')
                    stepFactor = 1.10;   % multiplicative step in log space
                else
                    stepFactor = max(allY) * 0.06;  % additive step in linear
                end
                for k = 1:size(pairs, 1)
                    a = perCondAtTarget{pairs(k,1)};
                    b = perCondAtTarget{pairs(k,2)};
                    if isempty(a) || isempty(b), continue; end
                    try, p_ = ranksum(a, b); catch, p_ = NaN; end
                    if strcmp(yscale, 'log')
                        yk = yBase * (stepFactor^(k-1));
                    else
                        yk = yBase + (k-1) * stepFactor;
                    end
                    plotSigBracket_local(gca, pairs(k,1), pairs(k,2), yk, p_, yscale);
                end
                % Expand ylim to fit the stacked brackets
                if strcmp(yscale, 'log')
                    ylim_max = yBase * (stepFactor^size(pairs,1));
                else
                    ylim_max = yBase + size(pairs,1) * stepFactor;
                end
                cur_ylim = ylim(gca);
                ylim(gca, [cur_ylim(1), ylim_max]);
            end
            savemyfig(figA3, fullfile(comparisonPath, sprintf('EC50_area_at_2s_%s_%s_%s', stemTag, yscale, modeSuffix)));
            fprintf('  Saved: EC50_area_at_2s_%s_%s_%s (.pdf, .png, .fig)\n', stemTag, yscale, modeSuffix);
            close(figA3);
        end
    end

    %% --- Single-duration figure 3: line plot with X clipped to <= target ---
    figA3 = figure('Name', sprintf('EC50 area vs Duration (clip %.1fs, %s)', config.targetDurationSec, displayMode));
    hold on;
    legendHandles = gobjects(nConditions, 1);
    if config.showRealTime
        xVals = durationsSec;
    else
        xVals = stimulusDurations;
        clipLimit = config.targetDurationSec * config.samplingRate / config.globalMeanSF;
    end
    for c = 1:nConditions
        condition = conditions{c};
        color = config.colors.(condition);
        EC50_area_mean = nan(1, length(stimulusDurations));
        EC50_area_sem  = nan(1, length(stimulusDurations));
        for d = 1:length(stimulusDurations)
            dKey = sprintf('dur_%d', stimulusDurations(d));
            if ~isfield(AllDurationGating, dKey) || ...
                    ~isfield(AllDurationGating.(dKey), condition) || ...
                    ~isfield(AllDurationGating.(dKey).(condition), metricsKey)
                continue;
            end
            node = AllDurationGating.(dKey).(condition).(metricsKey);
            if isfield(node, 'EC50_perSeed') && ~isempty(node.EC50_perSeed)
                ps = node.EC50_perSeed;
                ps = ps(~isnan(ps)) .^ 2;
                if ~isempty(ps)
                    EC50_area_mean(d) = mean(ps);
                    EC50_area_sem(d) = std(ps) / sqrt(numel(ps));
                end
            elseif isfield(node, 'EC50') && isfinite(node.EC50)
                EC50_area_mean(d) = node.EC50.^2;
                EC50_area_sem(d) = NaN;
            end
        end
        if config.showRealTime
            keep = (xVals <= config.targetDurationSec + eps) & ~isnan(EC50_area_mean);
        else
            keep = (xVals <= clipLimit + eps) & ~isnan(EC50_area_mean);
        end
        if ~any(keep)
            continue;
        end
        xK = xVals(keep);
        yK = EC50_area_mean(keep);
        eK = EC50_area_sem(keep);
        validBand = ~isnan(eK);
        if any(validBand)
            xFill = [xK(validBand), fliplr(xK(validBand))];
            yFill = [yK(validBand) + eK(validBand), fliplr(yK(validBand) - eK(validBand))];
            fill(xFill, yFill, color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        end
        legendHandles(c) = plot(xK, yK, 'o-', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 2);
    end
    set(gca, 'YScale', 'log');
    if config.showRealTime
        xlabel('Stimulus Duration (s)');
        xlim([0 config.targetDurationSec]);
    else
        xlabel('Stimulus Duration (frames)');
    end
    ylabel('EC50 area (pixels^2)');
    title(sprintf('EC50 area vs Duration (<= %.1f s) — %s', config.targetDurationSec, strrep(displayMode, '_', ' ')));
    valid = arrayfun(@(h) isgraphics(h), legendHandles);
    if any(valid)
        legend(legendHandles(valid), conditions(valid), 'Location', 'best');
    end
    grid on;
    savemyfig(figA3, fullfile(comparisonPath, sprintf('EC50_area_vs_Duration_clip2s_%s', modeSuffix)));
    fprintf('  Saved: EC50_area_vs_Duration_clip2s_%s (.pdf, .png, .fig)\n', modeSuffix);
    close(figA3);

    %% --- Collapse figure 1: per-condition swarm of pooled EC50^2 ---
    figB1 = figure('Name', sprintf('EC50 area pooled swarm (%s)', displayMode));
    hold on;
    rng(0, 'twister');
    for c = 1:nConditions
        y = perCondPooled{c};
        if isempty(y), continue; end
        x = c + (rand(numel(y), 1) - 0.5) * 0.35;
        scatter(x, y, 22, config.colors.(conditions{c}), 'filled', ...
            'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'none');
        mu = mean(y);
        sd = std(y);
        plot([c-0.3 c+0.3], [mu mu], '-', 'Color', config.colors.(conditions{c}), 'LineWidth', 2);
        plot([c c], [mu-sd mu+sd], '-', 'Color', config.colors.(conditions{c}), 'LineWidth', 1.5);
    end
    set(gca, 'XTick', 1:nConditions, 'XTickLabel', conditions, 'YScale', 'log');
    ylabel('EC50 area (pixels^2)');
    title(sprintf('EC50 area pooled across durations %s s — %s', ...
        mat2str(round(durationsSec(collapseIdxList), 1)), strrep(displayMode, '_', ' ')));
    grid on;
    savemyfig(figB1, fullfile(comparisonPath, sprintf('EC50_area_collapseTargetDurations_swarm_%s', modeSuffix)));
    fprintf('  Saved: EC50_area_collapseTargetDurations_swarm_%s (.pdf, .png, .fig)\n', modeSuffix);
    close(figB1);

    %% --- Collapse figure 2: 4-point mean+/-STD summary ---
    figB2 = figure('Name', sprintf('EC50 area pooled mean+/-STD (%s)', displayMode));
    hold on;
    fprintf('  [%s] Collapse summary (mean +/- STD over pooled EC50^2):\n', displayMode);
    for c = 1:nConditions
        y = perCondPooled{c};
        if isempty(y)
            fprintf('    %s: no data\n', conditions{c});
            continue;
        end
        mu = mean(y);
        sd = std(y);
        errorbar(c, mu, sd, 'o', 'Color', config.colors.(conditions{c}), ...
            'MarkerFaceColor', config.colors.(conditions{c}), 'LineWidth', 2, 'MarkerSize', 9);
        fprintf('    %s: mean=%.2f, std=%.2f, n=%d\n', conditions{c}, mu, sd, numel(y));
    end
    set(gca, 'XTick', 1:nConditions, 'XTickLabel', conditions, 'YScale', 'log');
    xlim([0.5 nConditions + 0.5]);
    ylabel('EC50 area (pixels^2)');
    title(sprintf('EC50 area pooled across durations %s s — %s', ...
        mat2str(round(durationsSec(collapseIdxList), 1)), strrep(displayMode, '_', ' ')));
    grid on;
    savemyfig(figB2, fullfile(comparisonPath, sprintf('EC50_area_collapseTargetDurations_meanStd_%s', modeSuffix)));
    fprintf('  Saved: EC50_area_collapseTargetDurations_meanStd_%s (.pdf, .png, .fig)\n', modeSuffix);
    close(figB2);
end

%% Figure: Threshold vs Duration
figThreshvsDur = figure('Name', 'Threshold vs Duration');

for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};
    metricsKey = strrep(displayMode, '_', '');

    subplot(1, nDisplayModes, m);
    hold on;

    for c = 1:nConditions
        condition = conditions{c};
        color = config.colors.(condition);

        thresh_vals = zeros(1, nDurations);
        for d = 1:nDurations
            dKey = sprintf('dur_%d', stimulusDurations(d));
            if isfield(AllDurationGating.(dKey), condition) && isfield(AllDurationGating.(dKey).(condition), metricsKey)
                thresh_vals(d) = AllDurationGating.(dKey).(condition).(metricsKey).mean_threshold;
            else
                thresh_vals(d) = NaN;
            end
        end

        if config.showRealTime
            plot(stimulusDurations * config.globalMeanSF / config.samplingRate, thresh_vals, 'o-', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 2);
        else
            plot(stimulusDurations, thresh_vals, 'o-', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 2);
        end
    end

    xlabel(sprintf('Stimulus Duration (%s)', timeUnit));
    ylabel('Threshold Size (pixels)');
    title(sprintf('Threshold vs Duration: %s', strrep(displayMode, '_', ' ')));
    legend(conditions, 'Location', 'best');
    grid on;
end

sgtitle('Propagation Threshold Changes with Duration');
savemyfig(figThreshvsDur, fullfile(comparisonPath, 'Threshold_vs_Duration'));
fprintf('  Saved: Threshold_vs_Duration (.pdf, .png, .fig)\n');
close(figThreshvsDur);

%% Propagation Dynamics Figures (All Durations) - Split into 2 separate figures
fprintf('\n--- Creating Propagation Dynamics Figures (2 separate) ---\n');

nDurs = length(stimulusDurations);

% Figure 1: Peak Propagation Velocity
figPropVelocity = figure('Name', 'Max Velocity - All Durations');
for dIdx = 1:nDurs
    durKey = sprintf('dur_%d', stimulusDurations(dIdx));
    durSeconds = stimulusDurations(dIdx) * config.globalMeanSF / config.samplingRate;
    PropDynamicsData = AllDurationPropDynamics.(durKey);

    subplot(1, nDurs, dIdx);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(PropDynamicsData, condition)
            continue;
        end
        velocity = PropDynamicsData.(condition).(stimModeKey).max_velocity;
        meanVel = mean(velocity, 1) * velocityScale;
        semVel = std(velocity, 0, 1) / sqrt(size(velocity, 1)) * velocityScale;
        color = config.colors.(condition);
        errorbar(stimulusSizes, meanVel, semVel, 'o-', ...
            'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
    end
    xlabel('Stimulus Size (pixels)');
    if dIdx == 1
        ylabel(sprintf('Max Velocity (%s)', velocityUnit));
        legend(conditions, 'Location', 'best');
    end
    title(sprintf('%.1fs', durSeconds));
    grid on;
end
sgtitle('Peak Propagation Velocity - All Durations');
savemyfig(figPropVelocity, fullfile(comparisonPath, 'PropDynamics_MaxVelocity_AllDurations'));
fprintf('  Saved: PropDynamics_MaxVelocity_AllDurations (.pdf, .png, .fig)\n');
close(figPropVelocity);

% Figure 2: Time to Peak Spread
figPropTimeToMax = figure('Name', 'Time to Max - All Durations');
for dIdx = 1:nDurs
    durKey = sprintf('dur_%d', stimulusDurations(dIdx));
    durSeconds = stimulusDurations(dIdx) * config.globalMeanSF / config.samplingRate;
    PropDynamicsData = AllDurationPropDynamics.(durKey);

    subplot(1, nDurs, dIdx);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(PropDynamicsData, condition)
            continue;
        end
        ttm = PropDynamicsData.(condition).(stimModeKey).time_to_max;
        meanTTM = mean(ttm, 1) * timeScale;
        semTTM = std(ttm, 0, 1) / sqrt(size(ttm, 1)) * timeScale;
        color = config.colors.(condition);
        errorbar(stimulusSizes, meanTTM, semTTM, 'o-', ...
            'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
    end
    xlabel('Stimulus Size (pixels)');
    if dIdx == 1
        ylabel(sprintf('Time to Max (%s)', timeUnit));
    end
    title(sprintf('%.1fs', durSeconds));
    grid on;
end
sgtitle('Time to Peak Spread - All Durations');
savemyfig(figPropTimeToMax, fullfile(comparisonPath, 'PropDynamics_TimeToMax_AllDurations'));
fprintf('  Saved: PropDynamics_TimeToMax_AllDurations (.pdf, .png, .fig)\n');
close(figPropTimeToMax);

%% Pre-Stimulus State Effects Figures (All Durations) - Split into 3 separate figures
fprintf('\n--- Creating Pre-Stimulus State Effects Figures (3 separate) ---\n');

% Figure 1: Baseline Moran's I by condition
figPreStimBaseline = figure('Name', 'Baseline Morans I - All Durations');
for dIdx = 1:nDurs
    durKey = sprintf('dur_%d', stimulusDurations(dIdx));
    durSeconds = stimulusDurations(dIdx) * config.globalMeanSF / config.samplingRate;
    PreStimData = AllDurationPreStimEffects.(durKey);

    subplot(1, nDurs, dIdx);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(PreStimData, condition)
            continue;
        end
        color = config.colors.(condition);
        moransVals = PreStimData.(condition).(stimModeKey).baseline_moransI(:);
        moransVals = moransVals(~isnan(moransVals) & moransVals ~= 0);
        if ~isempty(moransVals)
            scatter(c * ones(size(moransVals)) + 0.1*(rand(size(moransVals))-0.5), moransVals, ...
                30, color, 'filled', 'MarkerFaceAlpha', 0.6);
            plot([c-0.3, c+0.3], [mean(moransVals), mean(moransVals)], 'k-', 'LineWidth', 2);
        end
    end
    set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
    if dIdx == 1
        ylabel('Baseline Morans I');
    end
    title(sprintf('%.1fs', durSeconds));
end
sgtitle('Baseline Moran''s I - All Durations');
savemyfig(figPreStimBaseline, fullfile(comparisonPath, 'PreStim_BaselineMoransI_AllDurations'));
fprintf('  Saved: PreStim_BaselineMoransI_AllDurations (.pdf, .png, .fig)\n');
close(figPreStimBaseline);

% Figure 2: Baseline vs Peak response correlation
figPreStimPeak = figure('Name', 'Baseline vs Peak Correlation - All Durations');
for dIdx = 1:nDurs
    durKey = sprintf('dur_%d', stimulusDurations(dIdx));
    durSeconds = stimulusDurations(dIdx) * config.globalMeanSF / config.samplingRate;
    PreStimData = AllDurationPreStimEffects.(durKey);

    subplot(1, nDurs, dIdx);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(PreStimData, condition)
            continue;
        end
        color = config.colors.(condition);
        corrVals = PreStimData.(condition).(stimModeKey).baseline_vs_peak(:);
        corrVals = corrVals(~isnan(corrVals));
        if ~isempty(corrVals)
            scatter(c * ones(size(corrVals)) + 0.1*(rand(size(corrVals))-0.5), corrVals, ...
                30, color, 'filled', 'MarkerFaceAlpha', 0.6);
            plot([c-0.3, c+0.3], [mean(corrVals), mean(corrVals)], 'k-', 'LineWidth', 2);
        end
    end
    set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
    if dIdx == 1
        ylabel('r (Baseline vs Peak)');
    end
    title(sprintf('%.1fs', durSeconds));
    yline(0, 'k--');
end
sgtitle('Baseline vs Peak Correlation - All Durations');
savemyfig(figPreStimPeak, fullfile(comparisonPath, 'PreStim_BaselineVsPeak_AllDurations'));
fprintf('  Saved: PreStim_BaselineVsPeak_AllDurations (.pdf, .png, .fig)\n');
close(figPreStimPeak);

% Figure 3: Baseline vs Extent correlation
figPreStimExtent = figure('Name', 'Baseline vs Extent Correlation - All Durations');
for dIdx = 1:nDurs
    durKey = sprintf('dur_%d', stimulusDurations(dIdx));
    durSeconds = stimulusDurations(dIdx) * config.globalMeanSF / config.samplingRate;
    PreStimData = AllDurationPreStimEffects.(durKey);

    subplot(1, nDurs, dIdx);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(PreStimData, condition)
            continue;
        end
        color = config.colors.(condition);
        corrVals = PreStimData.(condition).(stimModeKey).baseline_vs_extent(:);
        corrVals = corrVals(~isnan(corrVals));
        if ~isempty(corrVals)
            scatter(c * ones(size(corrVals)) + 0.1*(rand(size(corrVals))-0.5), corrVals, ...
                30, color, 'filled', 'MarkerFaceAlpha', 0.6);
            plot([c-0.3, c+0.3], [mean(corrVals), mean(corrVals)], 'k-', 'LineWidth', 2);
        end
    end
    set(gca, 'XTick', 1:length(conditions), 'XTickLabel', conditions);
    if dIdx == 1
        ylabel('r (Baseline vs Extent)');
    end
    title(sprintf('%.1fs', durSeconds));
    yline(0, 'k--');
end
sgtitle('Baseline vs Extent Correlation - All Durations');
savemyfig(figPreStimExtent, fullfile(comparisonPath, 'PreStim_BaselineVsExtent_AllDurations'));
fprintf('  Saved: PreStim_BaselineVsExtent_AllDurations (.pdf, .png, .fig)\n');
close(figPreStimExtent);

%% Propagation Distance Figures (All Durations) - Split into one figure per mode
if hasRawData
fprintf('\n--- Creating Propagation Distance Figures (per mode) ---\n');

% Use representative stimulus size
repSize = 10;
sizeIdx = find(stimulusSizes == repSize, 1);
if isempty(sizeIdx)
    sizeIdx = round(length(stimulusSizes) / 2);
end

for m = 1:nDisplayModes
    displayMode = metricsDisplayModes{m};

    % Map display mode to data key
    dataKey = dataKeyFromDisplayMode(displayMode);

    nRows = ceil(nDurs / 5);
    nCols = min(nDurs, 5);
    figPropDist = figure('Name', sprintf('Propagation Distance %s - All Durations', displayMode));

    for dIdx = 1:nDurs
        durKey = sprintf('dur_%d', stimulusDurations(dIdx));
        durSeconds = stimulusDurations(dIdx) * config.globalMeanSF / config.samplingRate;
        DataDur = loadDurationData(config.outputPath, durKey, crossDurPass);

        currentDur = stimulusDurations(dIdx);
        stimOnFramesDur = currentDur;
        totalFramesDur = preStimFrames + stimOnFramesDur + postStimFrames;
        timeVecDur = (1:totalFramesDur) - preStimFrames;
        if config.showRealTime
            timeVec_plotDur = timeVecDur * config.globalMeanSF / config.samplingRate;
            stimOnTime_plotDur = stimOnFramesDur * config.globalMeanSF / config.samplingRate;
        else
            timeVec_plotDur = timeVecDur;
            stimOnTime_plotDur = stimOnFramesDur;
        end

        subplot(nRows, nCols, dIdx);
        hold on;

        for c = 1:length(conditions)
            condition = conditions{c};
            if ~isfield(DataDur, condition) || ~isfield(DataDur.(condition), dataKey)
                continue;
            end

            propData = DataDur.(condition).(dataKey).stimulus_blob_extent;
            if isHighBiasDisplayMode(displayMode) && ndims(propData) == 5
                propagation = squeeze(propData(:, sizeIdx, end, :, :));
            elseif isLowBiasDisplayMode(displayMode) && ndims(propData) == 5
                propagation = squeeze(propData(:, sizeIdx, lowBiasIdx, :, :));
            else
                propagation = squeeze(propData(:, sizeIdx, :, :));
            end
            meanProp = squeeze(mean(mean(propagation, 1), 2));

            color = config.colors.(condition);
            plot(timeVec_plotDur, meanProp, 'Color', color, 'LineWidth', 2);
        end

        xline(0, 'k--', 'LineWidth', 1);
        xline(stimOnTime_plotDur, 'k--', 'LineWidth', 1);

        yl = ylim;
        patch([0 stimOnTime_plotDur stimOnTime_plotDur 0], [yl(1) yl(1) yl(2) yl(2)], ...
            [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

        yline(stimulusSizes(sizeIdx)/2, 'r:', 'LineWidth', 1);

        xlabel(timeLabel);
        if dIdx == 1
            ylabel('Blob Extent (px)');
            legend(conditions, 'Location', 'best');
        end
        title(sprintf('%.1fs', durSeconds));
        grid on;
    end

    modeTitleStr = strrep(displayMode, '_', ' ');
    sgtitle(sprintf('Propagation Distance (%s) - All Durations', modeTitleStr));
    saveName = sprintf('PropagationDistance_%s_AllDurations', displayMode);
    savemyfig(figPropDist, fullfile(comparisonPath, saveName));
    fprintf('  Saved: %s (.pdf, .png, .fig)\n', saveName);
    close(figPropDist);
end
else
    fprintf('\n--- Skipping Propagation Distance Figures (no raw data) ---\n');
end  % hasRawData - Propagation Distance

%% Moran's I Dynamics Figures (All Durations) - Split into 3 separate figures
fprintf('\n--- Creating Morans I Dynamics Figures (3 separate) ---\n');

% Use representative stimulus size
repSize = 8;
sizeIdxMorans = find(stimulusSizes == repSize, 1);
if isempty(sizeIdxMorans), sizeIdxMorans = 4; end

% Figure 1: Moran's I Time Course
if hasRawData
figMoransITimeCourse = figure('Name', 'Morans I Time Course - All Durations');
for dIdx = 1:nDurs
    durKey = sprintf('dur_%d', stimulusDurations(dIdx));
    durSeconds = stimulusDurations(dIdx) * config.globalMeanSF / config.samplingRate;
    DataDur = loadDurationData(config.outputPath, durKey, crossDurPass);

    currentDur = stimulusDurations(dIdx);
    stimOnFramesDur = currentDur;
    totalFramesDur = preStimFrames + stimOnFramesDur + postStimFrames;
    timeVecDur = (1:totalFramesDur) - preStimFrames;
    if config.showRealTime
        timeVec_plotDur = timeVecDur * config.globalMeanSF / config.samplingRate;
        stimOnTime_plotDur = stimOnFramesDur * config.globalMeanSF / config.samplingRate;
    else
        timeVec_plotDur = timeVecDur;
        stimOnTime_plotDur = stimOnFramesDur;
    end

    subplot(1, nDurs, dIdx);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(DataDur, condition)
            continue;
        end
        moransArr = collapseBiasPlot(DataDur.(condition).(stimModeKey).morans_I);
        moransI = squeeze(moransArr(:, sizeIdxMorans, :, :));
        meanMoransI = squeeze(mean(mean(moransI, 1), 2));
        color = config.colors.(condition);
        plot(timeVec_plotDur, meanMoransI, 'Color', color, 'LineWidth', 2);
    end
    xline(0, 'k--'); xline(stimOnTime_plotDur, 'k--');
    xlabel(timeLabel);
    if dIdx == 1
        ylabel('Morans I');
        legend(conditions, 'Location', 'best');
    end
    title(sprintf('%.1fs', durSeconds));
end
sgtitle('Moran''s I Time Course - All Durations');
savemyfig(figMoransITimeCourse, fullfile(comparisonPath, 'MoransI_TimeCourse_AllDurations'));
fprintf('  Saved: MoransI_TimeCourse_AllDurations (.pdf, .png, .fig)\n');
close(figMoransITimeCourse);
else
    fprintf('  Skipping: MoransI_TimeCourse_AllDurations (no raw data)\n');
end  % hasRawData - Moran's I Time Course

% Figure 2: Moran's I Increase vs Size
figMoransIIncrease = figure('Name', 'Morans I Increase - All Durations');
for dIdx = 1:nDurs
    durKey = sprintf('dur_%d', stimulusDurations(dIdx));
    durSeconds = stimulusDurations(dIdx) * config.globalMeanSF / config.samplingRate;
    MoransIDynamicsData = AllDurationMoransIDynamics.(durKey);

    subplot(1, nDurs, dIdx);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(MoransIDynamicsData, condition)
            continue;
        end
        increase = MoransIDynamicsData.(condition).(stimModeKey).moransI_increase;
        meanInc = mean(increase, 1);
        semInc = std(increase, 0, 1) / sqrt(size(increase, 1));
        color = config.colors.(condition);
        errorbar(stimulusSizes, meanInc, semInc, 'o-', ...
            'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
    end
    xlabel('Stimulus Size (pixels)');
    if dIdx == 1
        ylabel('\Delta Morans I');
    end
    title(sprintf('%.1fs', durSeconds));
    grid on;
end
sgtitle('Moran''s I Increase vs Size - All Durations');
savemyfig(figMoransIIncrease, fullfile(comparisonPath, 'MoransI_Increase_AllDurations'));
fprintf('  Saved: MoransI_Increase_AllDurations (.pdf, .png, .fig)\n');
close(figMoransIIncrease);

% Figure 3: Return to Baseline Time
figMoransIReturn = figure('Name', 'Morans I Return Time - All Durations');
for dIdx = 1:nDurs
    durKey = sprintf('dur_%d', stimulusDurations(dIdx));
    durSeconds = stimulusDurations(dIdx) * config.globalMeanSF / config.samplingRate;
    MoransIDynamicsData = AllDurationMoransIDynamics.(durKey);

    subplot(1, nDurs, dIdx);
    hold on;
    for c = 1:length(conditions)
        condition = conditions{c};
        if ~isfield(MoransIDynamicsData, condition)
            continue;
        end
        returnTime = MoransIDynamicsData.(condition).(stimModeKey).return_to_baseline_time;
        meanRT = mean(returnTime, 1) * timeScale;
        semRT = std(returnTime, 0, 1) / sqrt(size(returnTime, 1)) * timeScale;
        color = config.colors.(condition);
        errorbar(stimulusSizes, meanRT, semRT, 'o-', ...
            'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5);
    end
    xlabel('Stimulus Size (pixels)');
    if dIdx == 1
        ylabel(sprintf('Return Time (%s)', timeUnit));
    end
    title(sprintf('%.1fs', durSeconds));
    grid on;
end
sgtitle('Moran''s I Return Time - All Durations');
savemyfig(figMoransIReturn, fullfile(comparisonPath, 'MoransI_ReturnTime_AllDurations'));
fprintf('  Saved: MoransI_ReturnTime_AllDurations (.pdf, .png, .fig)\n');
close(figMoransIReturn);

%% Figure: Summary of Expert Differences (All Modes)
fprintf('\n--- Creating Summary Figure ---\n');
figSummary = figure('Name', 'Expert Condition Summary');

compConditions = {'Expert', 'NoSpout'};  % Compare these two in dose-response
stimRadii = stimulusSizes / 2;
longestDurKey = sprintf('dur_%d', max(stimulusDurations));

for modeIdx = 1:nDisplayModes
    displayMode = metricsDisplayModes{modeIdx};
    metricsKey = strrep(displayMode, '_', '');
    modeTitleStr = strrep(displayMode, '_', ' ');

    % --- Column 1: EC50 Comparison ---
    subplot(nDisplayModes, 3, (modeIdx-1)*3 + 1);
    hold on;

    EC50_means = zeros(1, nConditions);
    EC50_sems = zeros(1, nConditions);
    for c = 1:nConditions
        condition = conditions{c};
        ec50_vals = zeros(1, nDurations);
        for d = 1:nDurations
            dKey = sprintf('dur_%d', stimulusDurations(d));
            if isfield(AllDurationGating.(dKey), condition) && isfield(AllDurationGating.(dKey).(condition), metricsKey)
                ec50_vals(d) = AllDurationGating.(dKey).(condition).(metricsKey).EC50;
            end
        end
        EC50_means(c) = mean(ec50_vals);
        EC50_sems(c) = std(ec50_vals) / sqrt(nDurations);
    end

    b = bar(1:nConditions, EC50_means, 0.6);
    b.FaceColor = 'flat';
    for c = 1:nConditions
        b.CData(c,:) = config.colors.(conditions{c});
    end
    errorbar(1:nConditions, EC50_means, EC50_sems, 'k.', 'LineWidth', 1.5);
    set(gca, 'XTick', 1:nConditions, 'XTickLabel', conditions);

    % Y-label shows mode name (first column only)
    ylabel(modeTitleStr);

    % Title only on first row
    if modeIdx == 1
        title('A) Sensitivity (EC50)');
        text(0.95, 0.95, '\downarrow = more sensitive', 'Units', 'normalized', ...
            'HorizontalAlignment', 'right', 'FontSize', 7, 'FontAngle', 'italic');
    end

    % Significance annotation
    sigDurationsEC50 = {};
    for d = 1:nDurations
        dKey = sprintf('dur_%d', stimulusDurations(d));
        if isfield(AllDurationStats.(dKey), metricsKey) && isfield(AllDurationStats.(dKey).(metricsKey), 'EC50')
            if AllDurationStats.(dKey).(metricsKey).EC50.pValue < 0.05
                if config.showRealTime
                    sigDurationsEC50{end+1} = sprintf('%.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate);
                else
                    sigDurationsEC50{end+1} = sprintf('%d', stimulusDurations(d));
                end
            end
        end
    end
    if ~isempty(sigDurationsEC50)
        sigTextEC50 = sprintf('Sig: %s', strjoin(sigDurationsEC50, ', '));
    else
        sigTextEC50 = 'No sig.';
    end
    text(0.05, 0.05, sigTextEC50, 'Units', 'normalized', 'FontSize', 7, 'FontAngle', 'italic');

    % --- Column 2: Hill Coefficient ---
    subplot(nDisplayModes, 3, (modeIdx-1)*3 + 2);
    hold on;

    Hill_means = zeros(1, nConditions);
    Hill_sems = zeros(1, nConditions);
    for c = 1:nConditions
        condition = conditions{c};
        hill_vals = zeros(1, nDurations);
        for d = 1:nDurations
            dKey = sprintf('dur_%d', stimulusDurations(d));
            if isfield(AllDurationGating.(dKey), condition) && isfield(AllDurationGating.(dKey).(condition), metricsKey)
                hill_vals(d) = AllDurationGating.(dKey).(condition).(metricsKey).hill_coefficient;
            end
        end
        Hill_means(c) = mean(hill_vals);
        Hill_sems(c) = std(hill_vals) / sqrt(nDurations);
    end

    b = bar(1:nConditions, Hill_means, 0.6);
    b.FaceColor = 'flat';
    for c = 1:nConditions
        b.CData(c,:) = config.colors.(conditions{c});
    end
    errorbar(1:nConditions, Hill_means, Hill_sems, 'k.', 'LineWidth', 1.5);
    set(gca, 'XTick', 1:nConditions, 'XTickLabel', conditions);
    yline(1, 'k--', 'LineWidth', 1);

    % Title only on first row
    if modeIdx == 1
        title('B) Response Steepness (Hill n)');
        text(0.95, 0.95, 'n>1: switch-like', 'Units', 'normalized', ...
            'HorizontalAlignment', 'right', 'FontSize', 7, 'FontAngle', 'italic');
        text(0.95, 0.85, 'n<1: gradual', 'Units', 'normalized', ...
            'HorizontalAlignment', 'right', 'FontSize', 7, 'FontAngle', 'italic');
    end

    % Significance annotation
    sigDurationsHill = {};
    for d = 1:nDurations
        dKey = sprintf('dur_%d', stimulusDurations(d));
        if isfield(AllDurationStats.(dKey), metricsKey) && isfield(AllDurationStats.(dKey).(metricsKey), 'hill_coefficient')
            if AllDurationStats.(dKey).(metricsKey).hill_coefficient.pValue < 0.05
                if config.showRealTime
                    sigDurationsHill{end+1} = sprintf('%.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate);
                else
                    sigDurationsHill{end+1} = sprintf('%d', stimulusDurations(d));
                end
            end
        end
    end
    if ~isempty(sigDurationsHill)
        sigTextHill = sprintf('Sig: %s', strjoin(sigDurationsHill, ', '));
    else
        sigTextHill = 'No sig.';
    end
    text(0.05, 0.05, sigTextHill, 'Units', 'normalized', 'FontSize', 7, 'FontAngle', 'italic');

    % --- Column 3: Dose-response curves (Expert vs NoSpout) ---
    subplot(nDisplayModes, 3, (modeIdx-1)*3 + 3);
    hold on;

    for ci = 1:length(compConditions)
        condition = compConditions{ci};
        if ~isfield(AllDurationMetrics.(longestDurKey), condition) || ~isfield(AllDurationMetrics.(longestDurKey).(condition), metricsKey)
            continue;
        end

        % Get net propagation [nSims x nSizes]
        extent = AllDurationMetrics.(longestDurKey).(condition).(metricsKey).max_blob_extent;
        netProp = extent - repmat(stimRadii, size(extent, 1), 1);

        meanNetProp = mean(netProp, 1);
        semNetProp = std(netProp, 0, 1) / sqrt(size(netProp, 1));

        color = config.colors.(condition);
        errorbar(stimulusSizes, meanNetProp, semNetProp, 'o-', ...
            'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5, 'MarkerSize', 4);
    end

    yline(0, 'k--', 'LineWidth', 1);
    xlabel('Stimulus Size (px)');
    grid on;

    % Title only on first row
    if modeIdx == 1
        if config.showRealTime
            title(sprintf('C) Expert vs NoSpout (%.1f s)', max(stimulusDurations) * config.globalMeanSF / config.samplingRate));
        else
            title(sprintf('C) Expert vs NoSpout (%d fr)', max(stimulusDurations)));
        end
        legend(compConditions, 'Location', 'northwest', 'FontSize', 7);
    end
end

sgtitle(sprintf('Summary: How Expert Differs from Other Conditions (%s)', strrep(config.stimMode, '_', ' ')));
savemyfig(figSummary, fullfile(comparisonPath, 'ExpertSummary'));
fprintf('  Saved: ExpertSummary (.pdf, .png, .fig)\n');
close(figSummary);
end  % crossDurPass loop

% Restore all-sims data
AllDurationMetrics = AllDurationMetrics_AllSims;
AllDurationGating = AllDurationGating_AllSims;
AllDurationStats = AllDurationStats_AllSims;
AllDurationPreStimEffects = AllDurationPreStimEffects_AllSims;
AllDurationPropDynamics = AllDurationPropDynamics_AllSims;
AllDurationBlobInteractions = AllDurationBlobInteractions_AllSims;
AllDurationMoransIDynamics = AllDurationMoransIDynamics_AllSims;
hasRawData = hasRawData_AllSims;

%% ========================================
%% SECTION 10: Bias Value Comparison (Bias Mode Only)
%% ========================================
if any(cellfun(@isBiasMode, stimulusModes)) && ~isempty(stimulusBiasValues) && hasRawData
    fprintf('\n========================================\n');
    fprintf('=== Creating Bias Value Comparisons ===\n');
    fprintf('========================================\n');

    % Create BiasComparison subfolder
    biasCompPath = fullfile(config.outputPath, 'BiasComparison');
    if ~exist(biasCompPath, 'dir')
        mkdir(biasCompPath);
    end

    % Create duration-specific subfolders (using real time)
    for d = 1:nDurs
        durSeconds = stimulusDurations(d) * config.globalMeanSF / config.samplingRate;
        durPath = fullfile(biasCompPath, sprintf('%.1fs', durSeconds));
        if ~exist(durPath, 'dir')
            mkdir(durPath);
        end
    end

    % Use the active mode's key — was hard-coded to 'bias' which made the
    % BiasComparison plots empty for the new 'double_pulse_bias[N]' modes.
    % stimModeKey is set globally as strrep(config.stimMode, '_', '').
    modeKey = stimModeKey;
    nBiasVals = length(stimulusBiasValues);
    nDurs = length(stimulusDurations);
    nConds = length(conditions);
    nSizes = length(stimulusSizes);
    stimRadii = stimulusSizes / 2;

    % Pre-compute metrics across all bias values for all durations
    BiasMetrics = struct();
    for d = 1:nDurs
        dKey = sprintf('dur_%d', stimulusDurations(d));
        BiasMetrics.(dKey) = struct();

        % Lazy-load this duration's Data from disk (bias section uses all-sims pass)
        DataDurBias = loadDurationData(config.outputPath, dKey, 1);

        for c = 1:nConds
            condition = conditions{c};
            BiasMetrics.(dKey).(condition) = struct();

            % Initialize metric arrays [nBiasValues]
            BiasMetrics.(dKey).(condition).EC50 = zeros(1, nBiasVals);
            BiasMetrics.(dKey).(condition).hill_coefficient = zeros(1, nBiasVals);
            BiasMetrics.(dKey).(condition).mean_threshold = zeros(1, nBiasVals);
            BiasMetrics.(dKey).(condition).mean_amplification = zeros(nSizes, nBiasVals);
            BiasMetrics.(dKey).(condition).mean_net_propagation = zeros(nSizes, nBiasVals);
            BiasMetrics.(dKey).(condition).propagation_success = zeros(nSizes, nBiasVals);
            BiasMetrics.(dKey).(condition).std_net_propagation = zeros(nSizes, nBiasVals);
            BiasMetrics.(dKey).(condition).cov_net_propagation = zeros(nSizes, nBiasVals);

            % Check if data exists
            if ~isfield(DataDurBias, condition) || ~isfield(DataDurBias.(condition), modeKey)
                continue;
            end

            activityData = DataDurBias.(condition).(modeKey).activity;
            extentData = DataDurBias.(condition).(modeKey).stimulus_blob_extent;

            % Verify 5D structure
            if ndims(activityData) ~= 5
                fprintf('  Warning: %s %s bias data is not 5D, skipping\n', dKey, condition);
                continue;
            end

            [nSims, ~, nBiasCheck, nReps, nFrames] = size(activityData);
            if nBiasCheck ~= nBiasVals
                fprintf('  Warning: Bias dimension mismatch for %s %s\n', dKey, condition);
                continue;
            end

            % Loop through each bias value
            for b = 1:nBiasVals
                % Extract data for this bias value [nSims x nSizes x nReps x nFrames]
                actForBias = squeeze(activityData(:, :, b, :, :));
                extForBias = squeeze(extentData(:, :, b, :, :));

                % Compute metrics for each size
                for s = 1:nSizes
                    % Get all replicate data [nSims x nReps x nFrames] -> average over reps
                    actSizeData = squeeze(actForBias(:, s, :, :));  % [nSims x nReps x nFrames]
                    extSizeData = squeeze(extForBias(:, s, :, :));

                    % Average across replicates first
                    meanActAcrossReps = squeeze(mean(actSizeData, 2));  % [nSims x nFrames]
                    meanExtAcrossReps = squeeze(mean(extSizeData, 2));

                    % Baseline and peak
                    baseline = mean(meanActAcrossReps(:, 1:preStimFrames), 2);  % [nSims x 1]
                    peakAct = max(meanActAcrossReps(:, stimOnFrames:end), [], 2);

                    % Amplification
                    amplification = peakAct ./ max(baseline, 0.01);
                    BiasMetrics.(dKey).(condition).mean_amplification(s, b) = mean(amplification);

                    % Net propagation
                    maxExtent = max(meanExtAcrossReps(:, stimOnFrames:end), [], 2);
                    netProp = maxExtent - stimRadii(s);
                    BiasMetrics.(dKey).(condition).mean_net_propagation(s, b) = mean(netProp);

                    % Propagation success (net > 0)
                    BiasMetrics.(dKey).(condition).propagation_success(s, b) = mean(netProp > 0);

                    % Propagation variability (CoV)
                    stdNetProp = std(netProp);
                    BiasMetrics.(dKey).(condition).std_net_propagation(s, b) = stdNetProp;
                    meanNetProp = BiasMetrics.(dKey).(condition).mean_net_propagation(s, b);
                    if meanNetProp > 0
                        BiasMetrics.(dKey).(condition).cov_net_propagation(s, b) = stdNetProp / meanNetProp;
                    else
                        BiasMetrics.(dKey).(condition).cov_net_propagation(s, b) = NaN;
                    end
                end

                % Fit Hill function for EC50 using mean amplification across sizes
                meanAmp = BiasMetrics.(dKey).(condition).mean_amplification(:, b);
                [ec50, hillN, ~] = fitHillFunction(stimulusSizes, meanAmp);
                BiasMetrics.(dKey).(condition).EC50(b) = ec50;
                BiasMetrics.(dKey).(condition).hill_coefficient(b) = hillN;

                % Threshold (size where net propagation first exceeds 0)
                netPropForBias = BiasMetrics.(dKey).(condition).mean_net_propagation(:, b);
                threshIdx = find(netPropForBias > 0, 1, 'first');
                if ~isempty(threshIdx)
                    BiasMetrics.(dKey).(condition).mean_threshold(b) = stimulusSizes(threshIdx);
                else
                    BiasMetrics.(dKey).(condition).mean_threshold(b) = max(stimulusSizes);
                end
            end

            fprintf('  %s %s: Computed metrics for %d bias values\n', dKey, condition, nBiasVals);
        end
    end

    %% Figure: EC50 vs Bias Value (one per duration)
    fprintf('\n--- Creating EC50 vs Bias Figures ---\n');
    for d = 1:nDurs
        dKey = sprintf('dur_%d', stimulusDurations(d));
        durPath = fullfile(biasCompPath, sprintf('%.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));

        figEC50Bias = figure('Name', sprintf('EC50 vs Bias Value - %.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));
        hold on;

        for c = 1:nConds
            condition = conditions{c};
            if ~isfield(BiasMetrics.(dKey), condition)
                continue;
            end

            ec50Vals = BiasMetrics.(dKey).(condition).EC50;
            color = config.colors.(condition);

            plot(stimulusBiasValues, ec50Vals, 'o-', 'Color', color, ...
                'MarkerFaceColor', color, 'LineWidth', 2, 'MarkerSize', 6);
        end

        xlabel('Bias Value');
        ylabel('EC50 (stimulus size)');
        legend(conditions, 'Location', 'best');
        grid on;
        xlim([min(stimulusBiasValues)*0.8, max(stimulusBiasValues)*1.2]);
        title(sprintf('EC50 Sensitivity vs Bias Strength (%.1f s)', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));

        savemyfig(figEC50Bias, fullfile(durPath, 'EC50_vs_Bias'));
        fprintf('  Saved: %.1fs/EC50_vs_Bias\n', stimulusDurations(d) * config.globalMeanSF / config.samplingRate);
        close(figEC50Bias);
    end

    %% Figure: Hill Coefficient vs Bias Value (one per duration)
    fprintf('\n--- Creating Hill Coefficient vs Bias Figures ---\n');
    for d = 1:nDurs
        dKey = sprintf('dur_%d', stimulusDurations(d));
        durPath = fullfile(biasCompPath, sprintf('%.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));

        figHillBias = figure('Name', sprintf('Hill Coefficient vs Bias Value - %.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));
        hold on;

        for c = 1:nConds
            condition = conditions{c};
            if ~isfield(BiasMetrics.(dKey), condition)
                continue;
            end

            hillVals = BiasMetrics.(dKey).(condition).hill_coefficient;
            color = config.colors.(condition);

            plot(stimulusBiasValues, hillVals, 'o-', 'Color', color, ...
                'MarkerFaceColor', color, 'LineWidth', 2, 'MarkerSize', 6);
        end

        yline(1, 'k--', 'n=1');
        xlabel('Bias Value');
        ylabel('Hill Coefficient (n)');
        legend(conditions, 'Location', 'best');
        grid on;
        xlim([min(stimulusBiasValues)*0.8, max(stimulusBiasValues)*1.2]);
        title(sprintf('Response Steepness vs Bias Strength (%.1f s)', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));

        savemyfig(figHillBias, fullfile(durPath, 'Hill_vs_Bias'));
        fprintf('  Saved: %.1fs/Hill_vs_Bias\n', stimulusDurations(d) * config.globalMeanSF / config.samplingRate);
        close(figHillBias);
    end

    %% Figure: Dose-Response Curves at Different Biases (one per duration)
    fprintf('\n--- Creating Dose-Response Gradient Figures ---\n');
    % Color gradient for bias values (darker = smaller bias, lighter = larger bias)
    biasColors = parula(nBiasVals);

    for d = 1:nDurs
        dKey = sprintf('dur_%d', stimulusDurations(d));
        durPath = fullfile(biasCompPath, sprintf('%.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));

        figDoseResp = figure('Name', sprintf('Dose-Response at Different Biases - %.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));

        for c = 1:nConds
            condition = conditions{c};
            subplot(2, 2, c);
            hold on;

            if ~isfield(BiasMetrics.(dKey), condition)
                title(sprintf('%s: No data', condition));
                continue;
            end

            % Plot dose-response for each bias value
            for b = 1:nBiasVals
                netProp = BiasMetrics.(dKey).(condition).mean_net_propagation(:, b);

                plot(stimulusSizes, netProp, 'o-', 'Color', biasColors(b,:), ...
                    'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerFaceColor', biasColors(b,:));
            end

            yline(0, 'k--', 'LineWidth', 1);
            xlabel('Stimulus Size (px)');
            ylabel('Net Propagation (px)');
            title(condition);
            grid on;

            if c == nConds
                % Add colorbar-like legend
                cb = colorbar;
                cb.Ticks = linspace(0, 1, nBiasVals);
                cb.TickLabels = arrayfun(@(x) sprintf('%.2f', x), stimulusBiasValues, 'UniformOutput', false);
                cb.Label.String = 'Bias Value';
            end
        end

        sgtitle(sprintf('Dose-Response Curves at Different Bias Values (%.1f s)', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));
        savemyfig(figDoseResp, fullfile(durPath, 'DoseResponse_BiasGradient'));
        fprintf('  Saved: %.1fs/DoseResponse_BiasGradient\n', stimulusDurations(d) * config.globalMeanSF / config.samplingRate);
        close(figDoseResp);
    end

    %% Figure: Bias x Size Heatmap (one per duration)
    fprintf('\n--- Creating Bias x Size Heatmaps ---\n');
    for d = 1:nDurs
        dKey = sprintf('dur_%d', stimulusDurations(d));
        durPath = fullfile(biasCompPath, sprintf('%.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));

        figHeatmap = figure('Name', sprintf('Bias x Size Heatmap - %.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));

        for c = 1:nConds
            condition = conditions{c};
            subplot(2, 2, c);

            if ~isfield(BiasMetrics.(dKey), condition)
                title(sprintf('%s: No data', condition));
                continue;
            end

            % Heatmap: rows = bias values, cols = stimulus sizes
            netPropMatrix = BiasMetrics.(dKey).(condition).mean_net_propagation';  % [nBias x nSizes]

            imagesc(netPropMatrix);
            colormap(gca, redblue(256));
            maxAbs = max(abs(netPropMatrix(:)));
            if maxAbs > 0
                caxis([-maxAbs, maxAbs]);
            end
            colorbar;

            set(gca, 'XTick', 1:nSizes, 'XTickLabel', stimulusSizes);
            set(gca, 'YTick', 1:nBiasVals, 'YTickLabel', arrayfun(@(x) sprintf('%.2f', x), stimulusBiasValues, 'UniformOutput', false));
            xlabel('Stimulus Size (px)');
            ylabel('Bias Value');
            title(condition);
        end

        sgtitle(sprintf('Net Propagation: Bias x Size Interaction (%.1f s)', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));
        savemyfig(figHeatmap, fullfile(durPath, 'Bias_Size_Heatmap'));
        fprintf('  Saved: %.1fs/Bias_Size_Heatmap\n', stimulusDurations(d) * config.globalMeanSF / config.samplingRate);
        close(figHeatmap);
    end

    %% Figure: Small Bias Deep Dive (one per duration)
    fprintf('\n--- Creating Small Bias Deep Dive Figures ---\n');
    smallBiasIdx = find(stimulusBiasValues <= 1);

    if ~isempty(smallBiasIdx)
    for d = 1:nDurs
        dKey = sprintf('dur_%d', stimulusDurations(d));
        durPath = fullfile(biasCompPath, sprintf('%.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));

        % Lazy-load this duration's Data from disk for bias deep dive
        DataDurBias = loadDurationData(config.outputPath, dKey, 1);

        figSmallBias = figure('Name', sprintf('Small Bias Deep Dive - %.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));

        % Panel A: EC50 comparison at small biases
        subplot(2, 3, 1);
        hold on;

        for c = 1:nConds
            condition = conditions{c};
            if ~isfield(BiasMetrics.(dKey), condition)
                continue;
            end

            ec50SmallBias = BiasMetrics.(dKey).(condition).EC50(smallBiasIdx);
            color = config.colors.(condition);

            bar_x = (1:length(smallBiasIdx)) + (c-1)*0.2 - 0.3;
            bar(bar_x, ec50SmallBias, 0.18, 'FaceColor', color);
        end

        set(gca, 'XTick', 1:length(smallBiasIdx), 'XTickLabel', arrayfun(@(x) sprintf('%.2f', x), stimulusBiasValues(smallBiasIdx), 'UniformOutput', false));
        xlabel('Bias Value');
        ylabel('EC50 (stimulus size)');
        title('A) EC50 at Small Biases');
        legend(conditions, 'Location', 'best', 'FontSize', 7);

        % Panel B: Threshold comparison at small biases
        subplot(2, 3, 2);
        hold on;

        for c = 1:nConds
            condition = conditions{c};
            if ~isfield(BiasMetrics.(dKey), condition)
                continue;
            end

            threshSmallBias = BiasMetrics.(dKey).(condition).mean_threshold(smallBiasIdx);
            color = config.colors.(condition);

            bar_x = (1:length(smallBiasIdx)) + (c-1)*0.2 - 0.3;
            bar(bar_x, threshSmallBias, 0.18, 'FaceColor', color);
        end

        set(gca, 'XTick', 1:length(smallBiasIdx), 'XTickLabel', arrayfun(@(x) sprintf('%.2f', x), stimulusBiasValues(smallBiasIdx), 'UniformOutput', false));
        xlabel('Bias Value');
        ylabel('Threshold Size (px)');
        title('B) Threshold at Small Biases');

        % Panel C: Hill coefficient at small biases
        subplot(2, 3, 3);
        hold on;

        for c = 1:nConds
            condition = conditions{c};
            if ~isfield(BiasMetrics.(dKey), condition)
                continue;
            end

            hillSmallBias = BiasMetrics.(dKey).(condition).hill_coefficient(smallBiasIdx);
            color = config.colors.(condition);

            bar_x = (1:length(smallBiasIdx)) + (c-1)*0.2 - 0.3;
            bar(bar_x, hillSmallBias, 0.18, 'FaceColor', color);
        end

        yline(1, 'k--');
        set(gca, 'XTick', 1:length(smallBiasIdx), 'XTickLabel', arrayfun(@(x) sprintf('%.2f', x), stimulusBiasValues(smallBiasIdx), 'UniformOutput', false));
        xlabel('Bias Value');
        ylabel('Hill Coefficient (n)');
        title('C) Hill Coeff at Small Biases');

        % Panel D: Expert vs Naive dose-response at smallest bias
        subplot(2, 3, 4);
        hold on;

        smallestBiasIdx = smallBiasIdx(1);
        compConds = {'Naive', 'Expert'};
        for ci = 1:length(compConds)
            cond = compConds{ci};
            if ~isfield(BiasMetrics.(dKey), cond)
                continue;
            end

            netProp = BiasMetrics.(dKey).(cond).mean_net_propagation(:, smallestBiasIdx);
            color = config.colors.(cond);

            plot(stimulusSizes, netProp, 'o-', 'Color', color, ...
                'MarkerFaceColor', color, 'LineWidth', 2);
        end

        yline(0, 'k--');
        xlabel('Stimulus Size (px)');
        ylabel('Net Propagation (px)');
        title(sprintf('D) Smallest Bias (%.2f)', stimulusBiasValues(smallestBiasIdx)));
        legend(compConds, 'Location', 'northwest');
        grid on;

        % Panel E: Activity time course at threshold size (Expert, smallest bias)
        subplot(2, 3, 5);
        hold on;

        % Find threshold size for Expert at smallest bias
        if isfield(BiasMetrics.(dKey), 'Expert')
            threshSize = BiasMetrics.(dKey).Expert.mean_threshold(smallestBiasIdx);
            threshSizeIdx = find(stimulusSizes >= threshSize, 1, 'first');
            if isempty(threshSizeIdx)
                threshSizeIdx = round(nSizes/2);
            end

            % Plot time courses for different small biases
            for bi = 1:length(smallBiasIdx)
                b = smallBiasIdx(bi);

                if isfield(DataDurBias, 'Expert') && isfield(DataDurBias.Expert, modeKey)
                    actData = DataDurBias.Expert.(modeKey).activity;
                    if ndims(actData) == 5
                        % [nSims x nSizes x nBias x nReps x nFrames]
                        nFrames = size(actData, 5);
                        timeVecLocal = (1:nFrames) - preStimFrames;
                        if config.showRealTime
                            timeVecLocal = timeVecLocal * config.globalMeanSF / config.samplingRate;
                        end

                        actForBias = squeeze(actData(:, threshSizeIdx, b, :, :));
                        meanAct = squeeze(mean(mean(actForBias, 1), 2));  % Average over sims and reps

                        plot(timeVecLocal, meanAct, '-', 'Color', biasColors(b,:), 'LineWidth', 1.5);
                    end
                end
            end

            xline(0, 'k--');
            if config.showRealTime
                xline(stimulusDurations(d) * config.globalMeanSF / config.samplingRate, 'k--');
            else
                xline(stimulusDurations(d), 'k--');
            end
            xlabel(timeLabel);
            ylabel('Activity');
            title(sprintf('E) Expert Time Course (size=%d)', stimulusSizes(threshSizeIdx)));

            % Add legend for small biases
            legendStrs = arrayfun(@(x) sprintf('bias=%.2f', x), stimulusBiasValues(smallBiasIdx), 'UniformOutput', false);
            legend(legendStrs, 'Location', 'best', 'FontSize', 7);
        end

        % Panel F: Summary text
        subplot(2, 3, 6);
        axis off;

        summaryText = {
            '\bf Small Bias Analysis Summary \rm', ...
            '', ...
            sprintf('Bias values analyzed: %s', mat2str(stimulusBiasValues(smallBiasIdx))), ...
            '', ...
            '\bf Key Questions: \rm', ...
            '1. Does sensitivity (EC50) change at small biases?', ...
            '2. Is threshold size affected by bias strength?', ...
            '3. Does Expert maintain advantage at weak biases?', ...
            '', ...
            '\bf Interpretation: \rm', ...
            'Small biases test whether the system can', ...
            'amplify weak inputs preferentially.'
        };

        text(0.05, 0.95, summaryText, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'FontSize', 9);

        sgtitle(sprintf('Small Bias Deep Dive (bias \\leq 1, %.1f s)', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));
        savemyfig(figSmallBias, fullfile(durPath, 'SmallBias_DeepDive'));
        fprintf('  Saved: %.1fs/SmallBias_DeepDive\n', stimulusDurations(d) * config.globalMeanSF / config.samplingRate);
        close(figSmallBias);
    end  % end duration loop
    end  % end if ~isempty(smallBiasIdx)

    %% Figure: Condition Differences at Each Bias Level (one per duration)
    fprintf('\n--- Creating Condition Comparison Figures ---\n');
    for d = 1:nDurs
        dKey = sprintf('dur_%d', stimulusDurations(d));
        durPath = fullfile(biasCompPath, sprintf('%.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));
        figCondComp = figure('Name', sprintf('Condition Comparison Across Biases - %.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));

    % Panel A: EC50 bar plot for each bias
    subplot(3, 2, 1);
    hold on;

    ec50Matrix = zeros(nConds, nBiasVals);
    for c = 1:nConds
        condition = conditions{c};
        if isfield(BiasMetrics.(dKey), condition)
            ec50Matrix(c, :) = BiasMetrics.(dKey).(condition).EC50;
        end
    end

    b = bar(ec50Matrix);
    for bi = 1:nBiasVals
        b(bi).FaceColor = 'flat';
        b(bi).CData = repmat(biasColors(bi,:), nConds, 1);
    end

    set(gca, 'XTick', 1:nConds, 'XTickLabel', conditions);
    ylabel('EC50 (stimulus size)');
    title('A) EC50 by Condition at Each Bias');
    legend(arrayfun(@(x) sprintf('bias=%.2f', x), stimulusBiasValues, 'UniformOutput', false), 'Location', 'best', 'FontSize', 7);

    % Panel B: Attention effect (Expert vs NoSpout)
    subplot(3, 2, 2);
    hold on;

    if isfield(BiasMetrics.(dKey), 'Expert') && isfield(BiasMetrics.(dKey), 'NoSpout')
        ec50Expert = BiasMetrics.(dKey).Expert.EC50;
        ec50NoSpout = BiasMetrics.(dKey).NoSpout.EC50;
        ec50Diff = ec50NoSpout - ec50Expert;  % Positive = Attending more sensitive

        bar(stimulusBiasValues, ec50Diff, 'FaceColor', [0.4 0.6 0.8]);
        xlabel('Bias Value');
        ylabel('\Delta EC50 (NoSpout - Expert)');
        title('B) Attention - Advantage in Sensitivity');
        yline(0, 'k--');
        grid on;
        xlim([min(stimulusBiasValues)*0.8, max(stimulusBiasValues)*1.2]);

        % Annotate
        text(0.95, 0.95, '+ve = Attending more sensitive', 'Units', 'normalized', ...
            'HorizontalAlignment', 'right', 'FontSize', 8, 'FontAngle', 'italic');
    end

    % Panels C-F: Propagation success heatmaps for all conditions
    panelLabels = {'C', 'D', 'E', 'F'};
    for c = 1:nConds
        condition = conditions{c};
        subplot(3, 2, 2 + c);

        if isfield(BiasMetrics.(dKey), condition)
            propSuccess = BiasMetrics.(dKey).(condition).propagation_success';
            imagesc(propSuccess);
            colormap(gca, parula);
            colorbar;
            caxis([0 1]);
            set(gca, 'XTick', 1:nSizes, 'XTickLabel', stimulusSizes);
            set(gca, 'YTick', 1:nBiasVals, 'YTickLabel', arrayfun(@(x) sprintf('%.2f', x), stimulusBiasValues, 'UniformOutput', false));
            xlabel('Stimulus Size (px)');
            ylabel('Bias Value');
            title(sprintf('%s) Propagation Success (netProp > 0) - %s', panelLabels{c}, condition));
        else
            title(sprintf('%s) %s: No data', panelLabels{c}, condition));
        end
    end

    if config.showRealTime
        sgtitle(sprintf('Condition Differences Across Bias Values (%.1f s)', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));
    else
        sgtitle(sprintf('Condition Differences Across Bias Values (%d frames)', stimulusDurations(d)));
    end
    savemyfig(figCondComp, fullfile(durPath, 'Condition_Comparison'));
    fprintf('  Saved: %.1fs/Condition_Comparison\n', stimulusDurations(d) * config.globalMeanSF / config.samplingRate);
    close(figCondComp);
    end  % end duration loop

    %% Figure: Propagation Reliability (one per duration)
    fprintf('\n--- Creating Propagation Reliability Figures ---\n');
    for d = 1:nDurs
        dKey = sprintf('dur_%d', stimulusDurations(d));
        durPath = fullfile(biasCompPath, sprintf('%.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));

        figReliability = figure('Name', sprintf('Propagation Reliability - %.1fs', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));

        % Panel A: CoV heatmap (Expert)
        subplot(2, 2, 1);
        if isfield(BiasMetrics.(dKey), 'Expert')
            covData = BiasMetrics.(dKey).Expert.cov_net_propagation';
            imagesc(covData);
            colormap(gca, flipud(hot));
            colorbar;
            set(gca, 'XTick', 1:nSizes, 'XTickLabel', stimulusSizes);
            set(gca, 'YTick', 1:nBiasVals, 'YTickLabel', arrayfun(@(x) sprintf('%.2f', x), stimulusBiasValues, 'UniformOutput', false));
            xlabel('Stimulus Size (px)');
            ylabel('Bias Value');
            title('A) CoV of Propagation (Expert)');
        end

        % Panel B: CoV heatmap (Naive)
        subplot(2, 2, 2);
        if isfield(BiasMetrics.(dKey), 'Naive')
            covData = BiasMetrics.(dKey).Naive.cov_net_propagation';
            imagesc(covData);
            colormap(gca, flipud(hot));
            colorbar;
            set(gca, 'XTick', 1:nSizes, 'XTickLabel', stimulusSizes);
            set(gca, 'YTick', 1:nBiasVals, 'YTickLabel', arrayfun(@(x) sprintf('%.2f', x), stimulusBiasValues, 'UniformOutput', false));
            xlabel('Stimulus Size (px)');
            ylabel('Bias Value');
            title('B) CoV of Propagation (Naive)');
        end

        % Panel C: Mean CoV vs Bias (line plot)
        subplot(2, 2, 3);
        hold on;
        for c = 1:nConds
            condition = conditions{c};
            if ~isfield(BiasMetrics.(dKey), condition)
                continue;
            end
            covData = BiasMetrics.(dKey).(condition).cov_net_propagation;
            meanCoV = nanmean(covData, 1);  % Average across sizes
            color = config.colors.(condition);
            plot(stimulusBiasValues, meanCoV, 'o-', 'Color', color, ...
                'MarkerFaceColor', color, 'LineWidth', 2, 'MarkerSize', 6);
        end
        xlabel('Bias Value');
        ylabel('Mean CoV');
        title('C) Propagation Variability vs Bias');
        legend(conditions, 'Location', 'best');
        grid on;

        % Panel D: Propagation success rate vs Bias (averaged across sizes)
        subplot(2, 2, 4);
        hold on;
        for c = 1:nConds
            condition = conditions{c};
            if ~isfield(BiasMetrics.(dKey), condition)
                continue;
            end
            propSuccess = BiasMetrics.(dKey).(condition).propagation_success;
            meanSuccess = mean(propSuccess, 1);  % Average across sizes
            color = config.colors.(condition);
            plot(stimulusBiasValues, meanSuccess, 'o-', 'Color', color, ...
                'MarkerFaceColor', color, 'LineWidth', 2, 'MarkerSize', 6);
        end
        xlabel('Bias Value');
        ylabel('Propagation Success Rate');
        title('D) Mean Propagation Probability vs Bias');
        legend(conditions, 'Location', 'best');
        ylim([0 1]);
        grid on;

        sgtitle(sprintf('Propagation Reliability Across Bias Values (%.1f s)', stimulusDurations(d) * config.globalMeanSF / config.samplingRate));
        savemyfig(figReliability, fullfile(durPath, 'Propagation_Reliability'));
        fprintf('  Saved: %.1fs/Propagation_Reliability\n', stimulusDurations(d) * config.globalMeanSF / config.samplingRate);
        close(figReliability);
    end

    %% Statistical Tests for Bias Effects
    fprintf('\n--- Computing Bias Effect Statistics ---\n');
    BiasStats = struct();

    dKey = sprintf('dur_%d', max(stimulusDurations));

    % Test: Does EC50 change significantly with bias value?
    for c = 1:nConds
        condition = conditions{c};
        if ~isfield(BiasMetrics.(dKey), condition)
            continue;
        end

        ec50Vals = BiasMetrics.(dKey).(condition).EC50;

        % Correlation between bias and EC50
        [r, p] = corr(log(stimulusBiasValues(:)), ec50Vals(:), 'Type', 'Spearman');
        BiasStats.(condition).ec50_bias_correlation = r;
        BiasStats.(condition).ec50_bias_pvalue = p;

        fprintf('  %s: EC50 vs Bias correlation r=%.3f, p=%.4f\n', condition, r, p);
    end

    % Test: Expert vs Naive difference at small vs large biases
    if isfield(BiasMetrics.(dKey), 'Expert') && isfield(BiasMetrics.(dKey), 'Naive')
        smallBiasVals = stimulusBiasValues <= 1;
        largeBiasVals = stimulusBiasValues > 1;

        ec50DiffSmall = mean(BiasMetrics.(dKey).Naive.EC50(smallBiasVals) - BiasMetrics.(dKey).Expert.EC50(smallBiasVals));
        ec50DiffLarge = mean(BiasMetrics.(dKey).Naive.EC50(largeBiasVals) - BiasMetrics.(dKey).Expert.EC50(largeBiasVals));

        BiasStats.expert_advantage_small_bias = ec50DiffSmall;
        BiasStats.expert_advantage_large_bias = ec50DiffLarge;

        fprintf('  Expert-Naive EC50 difference: Small bias=%.2f, Large bias=%.2f\n', ec50DiffSmall, ec50DiffLarge);
    end

    % Save bias comparison results
    biasAnalysisFile = fullfile(biasCompPath, 'BiasAnalysis.mat');
    save(biasAnalysisFile, 'BiasMetrics', 'BiasStats', 'stimulusBiasValues', 'stimulusDurations', 'conditions');
    fprintf('  Saved: BiasAnalysis.mat\n');

    fprintf('\n=== Bias Comparison Complete ===\n');
end

%% Save All-Duration Summary
if ~config.figuresOnly
    summaryFile = fullfile(config.outputPath, 'AllDurationAnalysis.mat');
    % AllDurationData is NOT saved here — it's already in per-duration PerturbationAnalysis.mat files
    % (omitting it avoids the ~50GB memory spike during save)
    save(summaryFile, 'AllDurationMetrics', 'AllDurationGating', 'AllDurationStats', ...
        'AllDurationPreStimEffects', 'AllDurationPropDynamics', 'AllDurationBlobInteractions', 'AllDurationMoransIDynamics', ...
        'config', 'stimulusSizes', 'stimulusDurations', 'stimulusModes', 'conditions', ...
        'preStimFrames', 'postStimFrames', 'stimulusBiasValues');
    fprintf('  Saved summary: %s\n', summaryFile);
end

fprintf('\n=== Analysis Complete ===\n');

%% Helper Functions

function tf = isBiasMode(mode)
% Return true for any mode that uses a stim_bias value (5D data layout):
% the original 'bias' mode and the new 'double_pulse_bias[N]' family.
% Also accepts bias-encoded display modes ('bias_2p00').
    rawMode = rawModeFromDisplayMode(mode);
    tf = strcmp(rawMode, 'bias') || startsWith(rawMode, 'double_pulse_bias');
end

function lbl = biasValueLabel(value)
% Format a bias value as '0p25', '0p50', '1p00', '2p00', '4p00', '8p00'.
% Matches the on-disk perturb_*.mat naming and the BiasComparison subdir
% convention.
    lbl = strrep(sprintf('%.2f', value), '.', 'p');
end

function tf = isBiasEncodedMode(mode)
% A bias-encoded mode carries a trailing '_<digits>p<digits>' suffix
% (e.g. 'bias_2p00', 'double_pulse_bias10_0p25'). Used to detect when
% config.stimMode pins a single bias value.
    tf = ~isempty(regexp(mode, '_\d+p\d+$', 'once'));
end

function rawMode = rawModeFromDisplayMode(displayMode)
% Strip 'high_'/'low_' prefix or trailing '_<biasLabel>' to recover the
% raw stimulus mode used in file/struct lookups.
%   'high_bias'                   -> 'bias'
%   'low_double_pulse_bias10'     -> 'double_pulse_bias10'
%   'bias_2p00'                   -> 'bias'
%   'double_pulse_bias10_0p25'    -> 'double_pulse_bias10'
%   'clamped'                     -> 'clamped'
    if isHighBiasDisplayMode(displayMode)
        rawMode = extractAfter(displayMode, 'high_');
    elseif isLowBiasDisplayMode(displayMode)
        rawMode = extractAfter(displayMode, 'low_');
    elseif ~isempty(regexp(displayMode, '_\d+p\d+$', 'once'))
        rawMode = regexprep(displayMode, '_\d+p\d+$', '');
    else
        rawMode = displayMode;
    end
end

function lbl = biasLabelFromMode(mode)
% Extract the bias-label suffix ('2p00', '0p25', ...) from a bias-encoded
% mode. Returns '' for non-bias-encoded modes.
    tok = regexp(mode, '_(\d+p\d+)$', 'tokens', 'once');
    if isempty(tok)
        lbl = '';
    else
        lbl = tok{1};
    end
end

function biasIdx = biasIdxFromMode(mode, stimulusBiasValues)
% Resolve the index into stimulusBiasValues for a bias-encoded mode.
% Returns -1 for non-bias-encoded modes or if the value is not in the list.
    biasIdx = -1;
    lbl = biasLabelFromMode(mode);
    if isempty(lbl), return; end
    val = str2double(strrep(lbl, 'p', '.'));
    if isnan(val) || isempty(stimulusBiasValues), return; end
    [~, biasIdx] = min(abs(stimulusBiasValues(:) - val));
    if abs(stimulusBiasValues(biasIdx) - val) > 1e-6
        biasIdx = -1;
    end
end

function tf = isHighBiasDisplayMode(displayMode)
% Display modes for bias-using modes are prefixed 'high_' / 'low_'.
    tf = startsWith(displayMode, 'high_');
end

function tf = isLowBiasDisplayMode(displayMode)
    tf = startsWith(displayMode, 'low_');
end

function tf = isBiasDisplayMode(displayMode)
    tf = isHighBiasDisplayMode(displayMode) || isLowBiasDisplayMode(displayMode);
end

function key = dataKeyFromDisplayMode(displayMode)
% Map a display mode (e.g. 'high_bias', 'low_double_pulse_bias10') to the
% underlying Data struct field (underscores stripped, prefix removed).
    if isHighBiasDisplayMode(displayMode)
        baseMode = extractAfter(displayMode, 'high_');
    elseif isLowBiasDisplayMode(displayMode)
        baseMode = extractAfter(displayMode, 'low_');
    else
        baseMode = displayMode;
    end
    key = strrep(baseMode, '_', '');
end

function arr = collapseBiasPlot(arr)
% For 5D bias-mode arrays [sims, sizes, nBiasValues, reps, frames], collapse
% the bias-value dim (3rd) by averaging so the result is 4D and indexable
% with the same (:, sizeIdx, :, :) pattern used for non-bias modes. Pass-
% through for non-5D arrays. Use this at every plot site where 5D bias
% data would otherwise crash with "Vectors must be the same length" or
% similar dim-mismatch errors.
    if ndims(arr) == 5
        arr = squeeze(mean(arr, 3));
    end
end

function names = displayModesForMode(mode)
% Expand a stimulus mode into its display modes. Bias-using modes split
% into a (high, low) pair; bias-encoded modes ('bias_2p00') map to
% themselves; all others map to themselves.
    if isBiasEncodedMode(mode)
        names = {mode};
    elseif isBiasMode(mode)
        names = {sprintf('high_%s', mode), sprintf('low_%s', mode)};
    else
        names = {mode};
    end
end

function savemyfig(fig, basePath)
% Save figure in multiple formats: PDF (vector), SVG (vector), PNG (raster), FIG (editable)

    try
        exportgraphics(fig, [basePath '.pdf'], 'ContentType', 'vector');
    catch
        warning('Could not save PDF: %s', basePath);
    end

    try
        print(fig, [basePath '.svg'], '-dsvg');
    catch
        warning('Could not save SVG: %s', basePath);
    end

    try
        exportgraphics(fig, [basePath '.png'], 'Resolution', 300);
    catch
        warning('Could not save PNG: %s', basePath);
    end

    try
        savefig(fig, [basePath '.fig']);
    catch
        warning('Could not save FIG: %s', basePath);
    end
end

function DataDur = loadDurationData(outputPath, durKey, crossDurPass)
% Load per-duration Data struct from saved PerturbationAnalysis.mat
    if crossDurPass == 2
        matFile = fullfile(outputPath, durKey, 'bestMatch', 'PerturbationAnalysis.mat');
    else
        matFile = fullfile(outputPath, durKey, 'PerturbationAnalysis.mat');
    end
    if exist(matFile, 'file')
        loaded = load(matFile, 'Data');
        DataDur = loaded.Data;
    else
        DataDur = struct();
    end
end

function [tau, fitR2] = fitExponentialDecay(signal, baseline)
% Fit exponential decay: y = A * exp(-t/tau) + baseline
% Only fits where signal is above 5% of initial amplitude to avoid log(~0) overflow

    signal = signal(:) - baseline;
    t = (1:length(signal))';

    % Initial amplitude
    A0 = signal(1);
    if A0 <= 0
        tau = NaN;
        fitR2 = NaN;
        return;
    end

    % Only fit contiguous region where signal > 5% of initial amplitude
    threshold = 0.05 * A0;
    lastValid = find(signal > threshold, 1, 'last');
    % Require contiguous from start: stop at first below-threshold point
    firstBelow = find(signal(1:end) <= threshold, 1, 'first');
    if isempty(firstBelow)
        fitEnd = length(signal);
    else
        fitEnd = firstBelow - 1;
    end

    if fitEnd < 3
        tau = NaN;
        fitR2 = NaN;
        return;
    end

    % Log-linear regression on valid region
    fitSignal = signal(1:fitEnd);
    fitT = t(1:fitEnd);
    logSig = log(fitSignal);
    validIdx = isfinite(logSig);

    if sum(validIdx) < 3
        tau = NaN;
        fitR2 = NaN;
        return;
    end

    % Linear fit: log(y) = log(A) - t/tau
    p = polyfit(fitT(validIdx), logSig(validIdx), 1);
    tau = -1 / p(1);

    % R-squared
    yPred = polyval(p, fitT(validIdx));
    SS_res = sum((logSig(validIdx) - yPred).^2);
    SS_tot = sum((logSig(validIdx) - mean(logSig(validIdx))).^2);
    if SS_tot == 0
        fitR2 = NaN;
    else
        fitR2 = 1 - SS_res / SS_tot;
    end

    % Ensure positive tau and cap at 2x signal length
    if tau < 0 || tau > 2 * length(signal)
        tau = NaN;
        fitR2 = NaN;
    end
end

function [tau, fitR2] = fitExponentialDecayNLS(signal, baseline)
% Fit exponential decay using nonlinear least squares: y = A * exp(-t/tau) + c
% More robust than log-linear regression — handles noisy / borderline signals.

    signal = signal(:);
    n = length(signal);
    t = (1:n)';

    if n < 3
        tau = NaN;
        fitR2 = NaN;
        return;
    end

    % Initial guesses
    A0 = signal(1) - baseline;
    if A0 <= 0
        A0 = max(signal) - baseline;
    end
    if A0 <= 0
        tau = NaN;
        fitR2 = NaN;
        return;
    end
    tau0 = n / 3;
    c0 = baseline;

    % Model: y = p(1) * exp(-t / p(2)) + p(3)
    expModel = @(p, t) p(1) * exp(-t / p(2)) + p(3);

    % Bounds: A > 0, tau in (0, 2*n], c unbounded
    lb = [0,   1e-6, -Inf];
    ub = [Inf, 2*n,   Inf];
    p0 = [A0, tau0, c0];

    opts = optimoptions('lsqcurvefit', 'Display', 'off', 'MaxIterations', 200);

    try
        [pFit, resnorm] = lsqcurvefit(expModel, p0, t, signal, lb, ub, opts);
        tau = pFit(2);

        % R-squared
        yPred = expModel(pFit, t);
        SS_res = sum((signal - yPred).^2);
        SS_tot = sum((signal - mean(signal)).^2);
        if SS_tot == 0
            fitR2 = NaN;
        else
            fitR2 = 1 - SS_res / SS_tot;
        end
    catch
        tau = NaN;
        fitR2 = NaN;
    end
end

function [EC50, n, R2] = fitHillFunction(sizes, response)
% Fit Hill function: R = Rmax * S^n / (EC50^n + S^n)

    % Normalize response
    Rmax = max(response);
    if Rmax == 0
        EC50 = NaN;
        n = NaN;
        R2 = NaN;
        return;
    end

    response = response(:) / Rmax;
    sizes = sizes(:);

    % Initial guess
    EC50_init = median(sizes);
    n_init = 2;

    % Optimization
    try
        opts = optimset('Display', 'off', 'MaxFunEvals', 1000);
        params = fminsearch(@(p) hillError(p, sizes, response), [EC50_init, n_init], opts);
        EC50 = params(1);
        n = params(2);

        % Compute R-squared
        predicted = (sizes.^n) ./ (EC50^n + sizes.^n);
        SS_res = sum((response - predicted).^2);
        SS_tot = sum((response - mean(response)).^2);
        R2 = 1 - SS_res / SS_tot;
    catch
        EC50 = NaN;
        n = NaN;
        R2 = NaN;
    end
end

function err = hillError(params, sizes, response)
    EC50 = params(1);
    n = params(2);

    % Constraints: EC50 > 0, n > 0
    if EC50 <= 0 || n <= 0
        err = Inf;
        return;
    end

    predicted = (sizes.^n) ./ (EC50^n + sizes.^n);
    err = sum((response - predicted).^2);
end

function cmap = redblue(n)
% REDBLUE - Red-White-Blue diverging colormap
%   cmap = redblue(n) returns an n-by-3 colormap matrix
%   Blue at low values, white at center, red at high values
%   Ideal for displaying data with positive and negative values

    if nargin < 1
        n = 256;
    end

    % Build colormap: blue -> white -> red
    half = floor(n / 2);

    % Blue to white (lower half)
    r1 = linspace(0.2, 1, half)';
    g1 = linspace(0.2, 1, half)';
    b1 = linspace(0.8, 1, half)';

    % White to red (upper half)
    r2 = linspace(1, 0.8, n - half)';
    g2 = linspace(1, 0.2, n - half)';
    b2 = linspace(1, 0.2, n - half)';

    cmap = [r1, g1, b1; r2, g2, b2];
end

function cmap = cool2warm(n)
% COOL2WARM - Cool-to-warm colormap (blue -> white -> red)
    if nargin < 1, n = 256; end
    half = floor(n / 2);
    % Blue to white
    r1 = linspace(0.2, 1, half)';
    g1 = linspace(0.2, 1, half)';
    b1 = linspace(0.8, 1, half)';
    % White to red
    r2 = linspace(1, 0.8, n - half)';
    g2 = linspace(1, 0.2, n - half)';
    b2 = linspace(1, 0.2, n - half)';
    cmap = [r1, g1, b1; r2, g2, b2];
end

function experiments = loadExperimentsForDuration(filepath, conditions, nTopMatches, modes, sizes, duration, biasValues)
% LOADEXPERIMENTSFORDURATION  Read experiments for one duration from HDF5.
%   Builds the same nested struct that load() would produce from a
%   hdf5storage-written .mat, but only for the requested duration.
%   This avoids loading the entire 50+ GB dataset into memory.
%
%   experiments = loadExperimentsForDuration(filepath, conditions, ...
%       nTopMatches, modes, sizes, duration, biasValues)

    experiments = struct();
    durKey = sprintf('dur_%d', duration);

    metricNames = {'activity', 'activity_crop', 'stim_activity', ...
        'stimulus_blob_area', 'stimulus_blob_extent', 'valid_blob_count', ...
        'morans_I', 'propagation_velocity', 'wavefront_anisotropy', ...
        'time_to_max_extent', 'max_propagation_velocity', ...
        'mean_anisotropy_during_stim'};

    for c = 1:length(conditions)
        cond = conditions{c};
        condStruct = struct();

        for sim = 0:(nTopMatches-1)
            simKey = sprintf('sim_%d', sim);
            simStruct = struct();

            for m = 1:length(modes)
                mode = modes{m};
                modeStruct = struct();

                for s = 1:length(sizes)
                    sizeKey = sprintf('size_%d', sizes(s));

                    if isBiasMode(mode) && ~isempty(biasValues)
                        % Bias mode: read each bias value sub-group
                        durStruct = struct();
                        for b = 1:length(biasValues)
                            biasKey = sprintf('bias_%s', ...
                                strrep(sprintf('%.2f', biasValues(b)), '.', 'p'));
                            basePath = sprintf('/experiments/%s/%s/%s/%s/%s/%s', ...
                                cond, simKey, mode, sizeKey, durKey, biasKey);

                            leafData = readLeafGroup(filepath, basePath, metricNames);
                            if ~isempty(fieldnames(leafData))
                                durStruct.(biasKey) = leafData;
                            end
                        end
                        sizeStruct.(durKey) = durStruct;
                    else
                        % Clamped / double_pulse
                        basePath = sprintf('/experiments/%s/%s/%s/%s/%s', ...
                            cond, simKey, mode, sizeKey, durKey);
                        leafData = readLeafGroup(filepath, basePath, metricNames);
                        if isempty(fieldnames(leafData))
                            sizeStruct.(durKey) = [];
                        else
                            sizeStruct.(durKey) = leafData;
                        end
                    end

                    modeStruct.(sizeKey) = sizeStruct;
                    clear sizeStruct;
                end

                simStruct.(mode) = modeStruct;
            end

            condStruct.(simKey) = simStruct;
        end

        experiments.(cond) = condStruct;
    end
end

function leafData = readLeafGroup(filepath, basePath, metricNames)
% READLEAFGROUP  Try to read all metric datasets from an HDF5 group.
%   Returns struct with fields for each successfully read metric.
%   NOTE: h5py writes in C (row-major) order; MATLAB h5read transposes axes.
%   Python shape (nReplicates, nFrames) becomes MATLAB (nFrames, nReplicates).
%   We transpose 2-D arrays back to (nReplicates, nFrames) for consistency
%   with the load()-based code path.
    leafData = struct();

    % Quick existence check — if the group doesn't exist, skip all metrics
    try
        h5info(filepath, basePath);
    catch
        return;
    end

    for i = 1:length(metricNames)
        try
            val = double(h5read(filepath, [basePath '/' metricNames{i}]));
            % Transpose 2-D arrays to undo C→Fortran axis reversal
            if ndims(val) == 2 %#ok<ISMAT>
                val = val';
            end
            leafData.(metricNames{i}) = val;
        catch
        end
    end

    % prestim_summary (stored as a sub-group with scalar datasets)
    try
        psPath = [basePath '/prestim_summary'];
        psInfo = h5info(filepath, psPath);
        ps = struct();
        for i = 1:length(psInfo.Datasets)
            ps.(psInfo.Datasets(i).Name) = double(h5read(filepath, ...
                [psPath '/' psInfo.Datasets(i).Name]));
        end
        leafData.prestim_summary = ps;
    catch
    end
end

function g = hedgesG(x, y)
    x = x(isfinite(x));
    y = y(isfinite(y));
    nx = numel(x);
    ny = numel(y);
    if nx < 2 || ny < 2
        g = NaN;
        return;
    end
    sp2 = ((nx-1)*var(x) + (ny-1)*var(y)) / (nx+ny-2);
    if sp2 <= 0
        g = NaN;
        return;
    end
    d = (mean(x) - mean(y)) / sqrt(sp2);
    df = nx + ny - 2;
    j = 1 - 3 / (4*df - 1);
    g = d * j;
end

function plotSigBracket_local(ax, x1, x2, y, p, yscale)
% Pairwise significance bracket: short tick down at x1, x2 and bar at y;
% labelled with conventional asterisks (or "n.s."). Works on linear OR log
% y-axis (handles tick offset proportionally).
    if strcmp(yscale, 'log')
        yTick = y * 0.97;
    else
        yTick = y - (y - 0) * 0.015;
    end
    line(ax, [x1, x1, x2, x2], [yTick, y, y, yTick], 'Color', 'k', 'LineWidth', 0.7, 'HandleVisibility', 'off');
    if isnan(p),       lbl = 'n.s.';
    elseif p < 0.001,  lbl = '***';
    elseif p < 0.01,   lbl = '**';
    elseif p < 0.05,   lbl = '*';
    else,              lbl = 'n.s.';
    end
    if strcmp(yscale, 'log')
        yText = y * 1.04;
    else
        yText = y + (y - 0) * 0.015;
    end
    text(ax, (x1 + x2) / 2, yText, lbl, ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'k');
end
