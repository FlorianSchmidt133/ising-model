%% =========================================================================
%% Figure 5: Combined Reliability Analysis for Ising Simulations
%% =========================================================================
%
% 1. TEMPORAL SPLIT-HALF RELIABILITY:
%    - Split each simulation's frames into two temporal halves
%    - Compare Moran's I statistics between halves using ICC, correlation, WD
%
% 2. SPATIAL SAMPLING CONSISTENCY:
%    - Compare Moran's I distributions between Position 1 and Position 2
%    - Assess whether spatial sampling location affects results
%
% 3. SPATIAL AVERAGING EFFECT (within-simulation):
%    - Test whether averaging Moran's I across spatial positions improves
%      temporal split-half reliability
%
% 4. DATA-TO-MODEL COMPARISON:
%    - Test whether averaging two spatial positions (P1+P2) improves the
%      match between experimental data and Ising simulations
%    - Compare: WD(Exp, Sim_P1) vs WD(Exp, Sim_(P1+P2)/2)
%
% Analysis Mode:
%   - 'best_match': Analyze only best-matching simulation per condition (default)
%   - 'all_sims':   Analyze all simulations, aggregate results
%
% Output:
%   - Combined dashboard figure with all reliability analyses
%   - Comprehensive results structure saved to .mat file
%   - Console summary report
%
% Interpretation:
%   - Low WD: Consistent distributions (good reliability)
%   - High WD: Position-dependent results (poor reliability)
%
% Dependencies:
%   - Requires IsingComparison_Results.mat for 'best_match' mode
%   - OR can compute from raw simulation files for 'all_sims' mode

%% =========================================================================
%% SECTION 1: Configuration
%% =========================================================================

fprintf('\n=== Combined Reliability Analysis for Ising Simulations ===\n\n');

% -------------------------------------------------------------------------
% Analysis Mode Flag
% -------------------------------------------------------------------------
% 'best_match': Process only best-matching simulation per condition (faster)
% 'all_sims':   Process all simulations and aggregate (comprehensive)
config.analysisMode = 'best_match';

% -------------------------------------------------------------------------
% Data Paths
% -------------------------------------------------------------------------
config.isingDataPath = mba_p('IsingModelData_for_Florian');
config.resultsPath = 'Fig. 5 Model\IsingModels\IsingComparison\Data\IsingComparison_Results.mat';
config.outputPath = 'Fig. 5 Model\IsingModels\IsingComparison\Reliability';

% -------------------------------------------------------------------------
% Experimental Grid Configuration
% -------------------------------------------------------------------------
config.experimentalGrid = [13, 26];  % Fixed experimental grid size

% -------------------------------------------------------------------------
% Temporal Analysis Parameters
% -------------------------------------------------------------------------
config.autocorrMaxLag = 200;       % Max lag for ACF decay time
config.acfThreshold = 1/exp(1);    % Threshold for decay time (1/e)
config.isingParams.missing_sim = 493;  % Known missing simulation

% -------------------------------------------------------------------------
% Conditions
% -------------------------------------------------------------------------
config.conditions = {'Naive', 'Beginner', 'Expert', 'NoSpout'};

% -------------------------------------------------------------------------
% Condition Colors (matching params from FS_load_default_settings)
% -------------------------------------------------------------------------
config.conditionColors.Naive = [0.3373, 0.7059, 0.9137];      % Light Blue
config.conditionColors.Beginner = [0.8431, 0.2549, 0.6078];   % Magenta/Pink
config.conditionColors.Expert = [0, 0.6196, 0.4510];          % Teal/Green
config.conditionColors.NoSpout = [0.8353, 0.3686, 0];         % Orange

fprintf('Configuration:\n');
fprintf('  Analysis mode: %s\n', config.analysisMode);
fprintf('  Experimental grid: [%d x %d]\n', config.experimentalGrid(1), config.experimentalGrid(2));
fprintf('  ACF decay threshold: %.4f (1/e)\n', config.acfThreshold);

% Create output directory if needed
if ~exist(config.outputPath, 'dir')
    mkdir(config.outputPath);
end

%% =========================================================================
%% SECTION 2: Load Data and Detect Grid Size
%% =========================================================================

fprintf('\n--- Section 2: Loading Data ---\n');

% Get list of all simulation files
simFiles = dir(fullfile(config.isingDataPath, 'sim_*.mat'));
nSimFiles = length(simFiles);
fprintf('Found %d simulation files\n', nSimFiles);

% Auto-detect simulation grid size from first valid file
fprintf('Auto-detecting simulation grid size...\n');
firstSimData = load(fullfile(config.isingDataPath, simFiles(1).name));
[nFramesPerSim, simRows, simCols] = size(firstSimData.stored_spins);
config.isingGrid = [simRows, simCols];
config.nFramesPerSim = nFramesPerSim;
config.halfFrames = floor(nFramesPerSim / 2);

fprintf('  Simulation grid: [%d x %d]\n', simRows, simCols);
fprintf('  Frames per simulation: %d\n', nFramesPerSim);
fprintf('  Frames per half (temporal): %d\n', config.halfFrames);

clear firstSimData;

%% =========================================================================
%% SECTION 3: Setup Grid Positions and Weight Matrix
%% =========================================================================

fprintf('\n--- Section 3: Setting Up Grid and Weight Matrix ---\n');

% Generate non-overlapping positions
positions = generateNonOverlappingPositions(config.isingGrid, config.experimentalGrid);
nPositions = size(positions, 1);

fprintf('Non-overlapping grid positions:\n');
fprintf('  Number of positions: %d\n', nPositions);
for p = 1:nPositions
    rowStart = positions(p, 1);
    colStart = positions(p, 2);
    rowEnd = rowStart + config.experimentalGrid(1) - 1;
    colEnd = colStart + config.experimentalGrid(2) - 1;
    fprintf('  Position %d: rows %d:%d, cols %d:%d\n', p, rowStart, rowEnd, colStart, colEnd);
end

if nPositions < 2
    warning('Only %d non-overlapping position(s). Need at least 2 for spatial analyses.', nPositions);
end

% Create weight matrix for experimental grid (13x26)
valueMap_exp = rand(config.experimentalGrid(1), config.experimentalGrid(2));
distanceMat_exp = squareform(mL_distanceMat(valueMap_exp));
uniqueDistances_exp = unique(distanceMat_exp);
uniqueDistances_exp(uniqueDistances_exp == 0) = [];
currDistInds_exp = ismember(distanceMat_exp, uniqueDistances_exp(1));
weightMat = zeros(size(distanceMat_exp));
weightMat(currDistInds_exp) = distanceMat_exp(currDistInds_exp);
weightMat(weightMat == inf) = 0;

fprintf('Weight matrix created: [%d x %d]\n', size(weightMat, 1), size(weightMat, 2));

% Pre-compute position indices
P1_rows = positions(1,1):(positions(1,1) + config.experimentalGrid(1) - 1);
P1_cols = positions(1,2):(positions(1,2) + config.experimentalGrid(2) - 1);
if nPositions >= 2
    P2_rows = positions(2,1):(positions(2,1) + config.experimentalGrid(1) - 1);
    P2_cols = positions(2,2):(positions(2,2) + config.experimentalGrid(2) - 1);
end

%% =========================================================================
%% SECTION 4: Load Best-Match Information (or process all sims)
%% =========================================================================

if strcmp(config.analysisMode, 'best_match')
    fprintf('\n--- Section 4: Loading Best-Match Information ---\n');

    if ~exist(config.resultsPath, 'file')
        error('Best match mode requires IsingComparison_Results.mat. File not found: %s', config.resultsPath);
    end

    compResults = load(config.resultsPath);
    if isfield(compResults, 'Results')
        compResults = compResults.Results;
    end

    if ~isfield(compResults, 'Comparison')
        error('IsingComparison_Results.mat does not contain Comparison structure.');
    end

    Comparison = compResults.Comparison;
    IsingData = compResults.IsingData;

    % Get best-match simulation IDs per condition
    bestMatch = struct();
    conditionsFound = {};

    for c = 1:length(config.conditions)
        cond = config.conditions{c};
        if isfield(Comparison, cond) && isfield(Comparison.(cond), 'bestMatch_simIDs')
            simID = Comparison.(cond).bestMatch_simIDs(1);
            bestMatch.(cond).simID = simID;
            conditionsFound{end+1} = cond;

            % Get parameters
            simIdx = find(IsingData.simIDs == simID, 1);
            if ~isempty(simIdx)
                bestMatch.(cond).params.beta = IsingData.params.beta(simIdx);
                bestMatch.(cond).params.c = IsingData.params.c(simIdx);
                bestMatch.(cond).params.decay_const = IsingData.params.decay_const(simIdx);
                bestMatch.(cond).params.inhibition_range = IsingData.params.inhibition_range(simIdx);
                bestMatch.(cond).params.bias = IsingData.params.bias(simIdx);
            end

            fprintf('  %s: sim_%d (beta=%.1f, c=%d)\n', cond, simID, ...
                bestMatch.(cond).params.beta, bestMatch.(cond).params.c);
        end
    end

    nConditions = length(conditionsFound);
    fprintf('Found %d conditions with best-match simulations\n', nConditions);

    clear compResults Comparison IsingData;
else
    fprintf('\n--- Section 4: All-Sims Mode ---\n');
    % For all_sims mode, we'll process all simulations later
    conditionsFound = {};
    nConditions = 0;
end

%% =========================================================================
%% SECTION 4b: Load Experimental Data for Data-Model Comparison
%% =========================================================================

fprintf('\n--- Section 4b: Loading Experimental Data ---\n');

config.experimentalDataPath = 'Fig. 5 Model\IsingModels\Data\DataForAbir.mat';

if exist(config.experimentalDataPath, 'file')
    expData = load(config.experimentalDataPath);
    MoransI_Exp = expData.MoransI;
    fprintf('Loaded experimental Moran''s I data from: %s\n', config.experimentalDataPath);

    % Display summary
    for c = 1:length(config.conditions)
        condition = config.conditions{c};
        if isfield(MoransI_Exp, condition) && ~isempty(MoransI_Exp.(condition))
            moransI = MoransI_Exp.(condition);
            fprintf('  %s: %d trials, %d timepoints, mean I=%.4f\n', ...
                condition, size(moransI, 1), size(moransI, 2), mean(moransI(:), 'omitnan'));
        end
    end
    experimentalDataAvailable = true;
else
    warning('Experimental data file not found: %s', config.experimentalDataPath);
    experimentalDataAvailable = false;
end

%% =========================================================================
%% SECTION 5: Compute Moran's I Time Series (ONCE)
%% =========================================================================

fprintf('\n--- Section 5: Computing Moran''s I Time Series ---\n');

if strcmp(config.analysisMode, 'best_match')
    % =====================================================================
    % BEST_MATCH MODE: Compute for each condition's best-match simulation
    % =====================================================================

    MoransI = struct();

    for c = 1:nConditions
        cond = conditionsFound{c};
        simID = bestMatch.(cond).simID;
        simPath = fullfile(config.isingDataPath, sprintf('sim_%d.mat', simID));

        fprintf('  Processing %s (sim_%d)...\n', cond, simID);

        if ~exist(simPath, 'file')
            warning('Simulation file not found: %s', simPath);
            continue;
        end

        % Load simulation
        simData = load(simPath);
        stored_spins = simData.stored_spins;
        nFrames = size(stored_spins, 1);

        % Compute Moran's I for Position 1
        moransI_P1 = zeros(1, nFrames);
        for t = 1:nFrames
            frame = squeeze(stored_spins(t, P1_rows, P1_cols));
            if all(frame(:) == 0) || all(frame(:) == 1)
                moransI_P1(t) = NaN;
            else
                moransI_P1(t) = mL_moransI(double(frame), weightMat);
            end
        end

        % Compute Moran's I for Position 2 (if available)
        if nPositions >= 2
            moransI_P2 = zeros(1, nFrames);
            for t = 1:nFrames
                frame = squeeze(stored_spins(t, P2_rows, P2_cols));
                if all(frame(:) == 0) || all(frame(:) == 1)
                    moransI_P2(t) = NaN;
                else
                    moransI_P2(t) = mL_moransI(double(frame), weightMat);
                end
            end
        else
            moransI_P2 = [];
        end

        % Store results
        MoransI.(cond).P1 = moransI_P1;
        MoransI.(cond).P2 = moransI_P2;
        MoransI.(cond).simID = simID;
        MoransI.(cond).nFrames = nFrames;

        fprintf('    P1: mean=%.4f, std=%.4f\n', mean(moransI_P1, 'omitnan'), std(moransI_P1, 'omitnan'));
        if nPositions >= 2
            fprintf('    P2: mean=%.4f, std=%.4f\n', mean(moransI_P2, 'omitnan'), std(moransI_P2, 'omitnan'));
        end
    end

    fprintf('Moran''s I computation complete for %d conditions\n', nConditions);

else
    % =====================================================================
    % ALL_SIMS MODE: Process all simulations
    % =====================================================================

    fprintf('Processing all %d simulations...\n', nSimFiles);

    % Initialize storage
    AllSimsData = struct();
    AllSimsData.simIDs = zeros(nSimFiles, 1);
    AllSimsData.P1_mean = zeros(nSimFiles, 1);
    AllSimsData.P1_std = zeros(nSimFiles, 1);
    AllSimsData.P2_mean = zeros(nSimFiles, 1);
    AllSimsData.P2_std = zeros(nSimFiles, 1);
    AllSimsData.half1_mean = zeros(nSimFiles, 1);
    AllSimsData.half2_mean = zeros(nSimFiles, 1);
    AllSimsData.within_wasserstein = zeros(nSimFiles, 1);

    progressInterval = max(1, ceil(nSimFiles / 20));
    tic;

    validIdx = 0;
    for i = 1:nSimFiles
        [~, fname] = fileparts(simFiles(i).name);
        simID = str2double(regexp(fname, '\d+', 'match', 'once'));

        if simID == config.isingParams.missing_sim
            continue;
        end

        simPath = fullfile(config.isingDataPath, simFiles(i).name);
        simData = load(simPath);
        stored_spins = simData.stored_spins;
        nFrames = size(stored_spins, 1);

        % Compute Moran's I for Position 1
        moransI_P1 = zeros(1, nFrames);
        for t = 1:nFrames
            frame = squeeze(stored_spins(t, P1_rows, P1_cols));
            if all(frame(:) == 0) || all(frame(:) == 1)
                moransI_P1(t) = NaN;
            else
                moransI_P1(t) = mL_moransI(double(frame), weightMat);
            end
        end

        % Compute Moran's I for Position 2
        if nPositions >= 2
            moransI_P2 = zeros(1, nFrames);
            for t = 1:nFrames
                frame = squeeze(stored_spins(t, P2_rows, P2_cols));
                if all(frame(:) == 0) || all(frame(:) == 1)
                    moransI_P2(t) = NaN;
                else
                    moransI_P2(t) = mL_moransI(double(frame), weightMat);
                end
            end
        end

        validIdx = validIdx + 1;
        AllSimsData.simIDs(validIdx) = simID;
        AllSimsData.P1_mean(validIdx) = mean(moransI_P1, 'omitnan');
        AllSimsData.P1_std(validIdx) = std(moransI_P1, 'omitnan');

        if nPositions >= 2
            AllSimsData.P2_mean(validIdx) = mean(moransI_P2, 'omitnan');
            AllSimsData.P2_std(validIdx) = std(moransI_P2, 'omitnan');
        end

        % Split-half statistics
        halfpoint = floor(nFrames / 2);
        half1 = moransI_P1(1:halfpoint);
        half2 = moransI_P1(halfpoint+1:end);
        half1_clean = half1(~isnan(half1));
        half2_clean = half2(~isnan(half2));

        AllSimsData.half1_mean(validIdx) = mean(half1_clean);
        AllSimsData.half2_mean(validIdx) = mean(half2_clean);
        AllSimsData.within_wasserstein(validIdx) = wasserstein_1d(half1_clean, half2_clean);

        if mod(i, progressInterval) == 0
            elapsed = toc;
            remaining = elapsed / i * (nSimFiles - i);
            fprintf('  Progress: %d/%d (%.1f%%), ~%.1fs remaining\n', ...
                i, nSimFiles, 100*i/nSimFiles, remaining);
        end
    end

    % Trim arrays
    AllSimsData.simIDs = AllSimsData.simIDs(1:validIdx);
    AllSimsData.P1_mean = AllSimsData.P1_mean(1:validIdx);
    AllSimsData.P1_std = AllSimsData.P1_std(1:validIdx);
    AllSimsData.P2_mean = AllSimsData.P2_mean(1:validIdx);
    AllSimsData.P2_std = AllSimsData.P2_std(1:validIdx);
    AllSimsData.half1_mean = AllSimsData.half1_mean(1:validIdx);
    AllSimsData.half2_mean = AllSimsData.half2_mean(1:validIdx);
    AllSimsData.within_wasserstein = AllSimsData.within_wasserstein(1:validIdx);

    nSims = validIdx;
    fprintf('Completed processing %d simulations in %.1f seconds\n', nSims, toc);
end

%% =========================================================================
%% SECTION 6: Analysis 1 - Temporal Split-Half Reliability
%% =========================================================================

fprintf('\n--- Section 6: Temporal Split-Half Reliability Analysis ---\n');

TemporalReliability = struct();

if strcmp(config.analysisMode, 'best_match')
    % Per-condition analysis
    for c = 1:nConditions
        cond = conditionsFound{c};
        moransI_ts = MoransI.(cond).P1;
        nFrames = MoransI.(cond).nFrames;
        halfpoint = floor(nFrames / 2);

        % Split into halves
        half1 = moransI_ts(1:halfpoint);
        half2 = moransI_ts(halfpoint+1:end);

        half1_clean = half1(~isnan(half1));
        half2_clean = half2(~isnan(half2));

        % Basic statistics
        TemporalReliability.(cond).half1_mean = mean(half1_clean);
        TemporalReliability.(cond).half2_mean = mean(half2_clean);
        TemporalReliability.(cond).half1_std = std(half1_clean);
        TemporalReliability.(cond).half2_std = std(half2_clean);

        % Wasserstein distance
        TemporalReliability.(cond).wasserstein = wasserstein_1d(half1_clean, half2_clean);

        % Bland-Altman (frame-by-frame)
        minLen = min(length(half1_clean), length(half2_clean));
        h1 = half1_clean(1:minLen);
        h2 = half2_clean(1:minLen);
        TemporalReliability.(cond).ba_averages = (h1 + h2) / 2;
        TemporalReliability.(cond).ba_differences = h2 - h1;
        TemporalReliability.(cond).ba_bias = mean(h2 - h1);
        TemporalReliability.(cond).ba_loa_upper = TemporalReliability.(cond).ba_bias + 1.96 * std(h2 - h1);
        TemporalReliability.(cond).ba_loa_lower = TemporalReliability.(cond).ba_bias - 1.96 * std(h2 - h1);

        fprintf('  %s: WD=%.4f\n', cond, TemporalReliability.(cond).wasserstein);
    end
else
    % All-sims aggregate analysis
    mean1 = AllSimsData.half1_mean;
    mean2 = AllSimsData.half2_mean;

    validIdx = ~isnan(mean1) & ~isnan(mean2);
    mean1_valid = mean1(validIdx);
    mean2_valid = mean2(validIdx);

    TemporalReliability.aggregate.mad = mean(abs(mean1_valid - mean2_valid));

    TemporalReliability.aggregate.differences = mean2_valid - mean1_valid;
    TemporalReliability.aggregate.averages = (mean1_valid + mean2_valid) / 2;
    TemporalReliability.aggregate.bias = mean(TemporalReliability.aggregate.differences);
    TemporalReliability.aggregate.loa_upper = TemporalReliability.aggregate.bias + 1.96 * std(TemporalReliability.aggregate.differences);
    TemporalReliability.aggregate.loa_lower = TemporalReliability.aggregate.bias - 1.96 * std(TemporalReliability.aggregate.differences);

    wassDist = AllSimsData.within_wasserstein;
    wassDist_valid = wassDist(~isnan(wassDist));
    TemporalReliability.aggregate.wasserstein_mean = mean(wassDist_valid);
    TemporalReliability.aggregate.wasserstein_p95 = prctile(wassDist_valid, 95);

    fprintf('  Aggregate: WD_mean=%.4f\n', TemporalReliability.aggregate.wasserstein_mean);
end

%% =========================================================================
%% SECTION 7: Analysis 2 - Spatial Sampling Consistency
%% =========================================================================

fprintf('\n--- Section 7: Spatial Sampling Consistency Analysis ---\n');

SpatialConsistency = struct();

if nPositions < 2
    fprintf('  Skipping - requires at least 2 non-overlapping positions\n');
    SpatialConsistency.available = false;
else
    SpatialConsistency.available = true;

    if strcmp(config.analysisMode, 'best_match')
        for c = 1:nConditions
            cond = conditionsFound{c};
            P1 = MoransI.(cond).P1;
            P2 = MoransI.(cond).P2;

            % Wasserstein distance between positions
            SpatialConsistency.(cond).WD = wasserstein_1d(P1, P2);

            % Basic statistics
            SpatialConsistency.(cond).P1_mean = mean(P1, 'omitnan');
            SpatialConsistency.(cond).P2_mean = mean(P2, 'omitnan');
            SpatialConsistency.(cond).P1_std = std(P1, 'omitnan');
            SpatialConsistency.(cond).P2_std = std(P2, 'omitnan');
            SpatialConsistency.(cond).mean_diff = SpatialConsistency.(cond).P2_mean - SpatialConsistency.(cond).P1_mean;

            fprintf('  %s: WD(P1,P2)=%.4f, mean_diff=%.4f\n', cond, ...
                SpatialConsistency.(cond).WD, SpatialConsistency.(cond).mean_diff);
        end
    else
        % All-sims aggregate
        P1_means = AllSimsData.P1_mean;
        P2_means = AllSimsData.P2_mean;

        validIdx = ~isnan(P1_means) & ~isnan(P2_means);
        SpatialConsistency.aggregate.mean_P1 = mean(P1_means(validIdx));
        SpatialConsistency.aggregate.mean_P2 = mean(P2_means(validIdx));
        SpatialConsistency.aggregate.WD = wasserstein_1d(P1_means(validIdx), P2_means(validIdx));

        fprintf('  Aggregate: WD=%.4f\n', SpatialConsistency.aggregate.WD);
    end
end

%% =========================================================================
%% SECTION 8: Analysis 3 - Spatial Pooling Effect
%% =========================================================================
% NOTE: "Pooling" means concatenating samples from P1 and P2 (2x samples),
% NOT frame-by-frame averaging which would change the distribution variance.

fprintf('\n--- Section 8: Spatial Pooling Effect Analysis ---\n');

SpatialPooling = struct();

if nPositions < 2
    fprintf('  Skipping - requires at least 2 non-overlapping positions\n');
    SpatialPooling.available = false;
else
    SpatialPooling.available = true;

    if strcmp(config.analysisMode, 'best_match')
        for c = 1:nConditions
            cond = conditionsFound{c};
            P1 = MoransI.(cond).P1;
            P2 = MoransI.(cond).P2;
            nFrames = MoransI.(cond).nFrames;
            halfpoint = floor(nFrames / 2);

            % Split-half for single position (P1)
            half1_single = P1(1:halfpoint);
            half2_single = P1(halfpoint+1:end);

            % Split-half for POOLED: concatenate corresponding temporal halves
            % This preserves temporal structure while pooling spatial samples
            half1_pooled = [P1(1:halfpoint), P2(1:halfpoint)];           % All first-half frames from both positions
            half2_pooled = [P1(halfpoint+1:end), P2(halfpoint+1:end)];   % All second-half frames from both positions

            % Clean NaNs
            h1_single = half1_single(~isnan(half1_single));
            h2_single = half2_single(~isnan(half2_single));
            h1_pooled = half1_pooled(~isnan(half1_pooled));
            h2_pooled = half2_pooled(~isnan(half2_pooled));

            % Compute WD for both
            SpatialPooling.(cond).WD_single = wasserstein_1d(h1_single, h2_single);
            SpatialPooling.(cond).WD_pooled = wasserstein_1d(h1_pooled, h2_pooled);
            SpatialPooling.(cond).WD_change_pct = 100 * (SpatialPooling.(cond).WD_pooled - SpatialPooling.(cond).WD_single) / SpatialPooling.(cond).WD_single;
            SpatialPooling.(cond).improved = SpatialPooling.(cond).WD_pooled < SpatialPooling.(cond).WD_single;

            fprintf('  %s: WD single=%.4f, pooled=%.4f (change: %+.1f%%)\n', cond, ...
                SpatialPooling.(cond).WD_single, SpatialPooling.(cond).WD_pooled, ...
                SpatialPooling.(cond).WD_change_pct);
        end
    end
end

%% =========================================================================
%% SECTION 8b: Data-to-Model Spatial Pooling Effect
%% =========================================================================
% Test whether POOLING Moran's I samples from two spatial positions (P1+P2)
% improves the match between experimental data and Ising simulations.
%
% NOTE: "Pooling" = concatenating samples (2x samples from same distribution)
%       NOT frame-by-frame averaging (which changes variance)
%
% Compare:
%   WD_single: WD(Experimental, Simulation_P1)
%   WD_pooled: WD(Experimental, [Simulation_P1; Simulation_P2])

fprintf('\n--- Section 8b: Data-to-Model Spatial Pooling Effect ---\n');

DataModelComparison = struct();

if experimentalDataAvailable && strcmp(config.analysisMode, 'best_match') && nPositions >= 2
    DataModelComparison.available = true;

    for c = 1:nConditions
        cond = conditionsFound{c};

        % Check if experimental data exists for this condition
        if ~isfield(MoransI_Exp, cond) || isempty(MoransI_Exp.(cond))
            fprintf('  %s: No experimental data available\n', cond);
            continue;
        end

        % Get experimental Moran's I (flatten to 1D)
        exp_moransI = MoransI_Exp.(cond)(:);
        exp_moransI = exp_moransI(~isnan(exp_moransI));

        % Get simulation Moran's I from P1 (single position)
        sim_P1 = MoransI.(cond).P1(:);

        % POOLED: Concatenate all samples from P1 and P2 (2x samples)
        sim_pooled = [MoransI.(cond).P1(:); MoransI.(cond).P2(:)];

        % Compute Wasserstein distances: Data vs Model
        DataModelComparison.(cond).WD_single = wasserstein_1d(exp_moransI, sim_P1);
        DataModelComparison.(cond).WD_pooled = wasserstein_1d(exp_moransI, sim_pooled);

        % Compute % change (negative = improvement)
        DataModelComparison.(cond).WD_change_pct = 100 * ...
            (DataModelComparison.(cond).WD_pooled - DataModelComparison.(cond).WD_single) / ...
            DataModelComparison.(cond).WD_single;
        DataModelComparison.(cond).improved = DataModelComparison.(cond).WD_pooled < DataModelComparison.(cond).WD_single;

        % Store additional info
        DataModelComparison.(cond).exp_mean = mean(exp_moransI);
        DataModelComparison.(cond).sim_P1_mean = mean(sim_P1, 'omitnan');
        DataModelComparison.(cond).sim_pooled_mean = mean(sim_pooled, 'omitnan');

        fprintf('  %s: WD(Data,P1)=%.4f, WD(Data,Pooled)=%.4f (change: %+.1f%%)\n', cond, ...
            DataModelComparison.(cond).WD_single, DataModelComparison.(cond).WD_pooled, ...
            DataModelComparison.(cond).WD_change_pct);
    end
else
    DataModelComparison.available = false;
    if ~experimentalDataAvailable
        fprintf('  Skipping - experimental data not available\n');
    elseif nPositions < 2
        fprintf('  Skipping - requires at least 2 non-overlapping positions\n');
    else
        fprintf('  Skipping - only available in best_match mode\n');
    end
end

%% =========================================================================
%% SECTION 9: Visualization
%% =========================================================================

fprintf('\n--- Section 9: Creating Visualizations ---\n');

% Define color schemes
colors_spatial = struct('P1', [0.2 0.4 0.8], 'P2', [0.8 0.4 0.2]);  % Blue/Orange
colors_temporal = struct('T1', [0.2 0.6 0.4], 'T2', [0.6 0.2 0.6]); % Green/Purple

if strcmp(config.analysisMode, 'best_match')

    % =====================================================================
    % FIGURE 1: Distribution Comparisons (2x6)
    % =====================================================================
    figure('Name', 'Distribution Comparisons');

    % ---------------------------------------------------------------------
    % ROW 1: Spatial Sampling (P1 vs P2)
    % ---------------------------------------------------------------------

    % Panel (1,1): Spatial explainer - Grid positions
    subplot(2, 6, 1);
    hold on;
    % Grid outline
    rectangle('Position', [0.5, 0.5, config.isingGrid(2), config.isingGrid(1)], 'EdgeColor', 'k', 'LineWidth', 2);

    % P1 position - use patch for legend support
    rowStart = positions(1, 1);
    colStart = positions(1, 2);
    expW = config.experimentalGrid(2);
    expH = config.experimentalGrid(1);
    h1 = patch([colStart-0.5, colStart-0.5+expW, colStart-0.5+expW, colStart-0.5], ...
               [rowStart-0.5, rowStart-0.5, rowStart-0.5+expH, rowStart-0.5+expH], ...
               colors_spatial.P1, 'FaceAlpha', 0.3, 'EdgeColor', colors_spatial.P1, 'LineWidth', 2);
    text(colStart + expW/2, rowStart + expH/2, ...
        'P1', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12, 'Color', colors_spatial.P1);

    % P2 position - use patch for legend support
    if nPositions >= 2
        rowStart = positions(2, 1);
        colStart = positions(2, 2);
        h2 = patch([colStart-0.5, colStart-0.5+expW, colStart-0.5+expW, colStart-0.5], ...
                   [rowStart-0.5, rowStart-0.5, rowStart-0.5+expH, rowStart-0.5+expH], ...
                   colors_spatial.P2, 'FaceAlpha', 0.3, 'EdgeColor', colors_spatial.P2, 'LineWidth', 2);
        text(colStart + expW/2, rowStart + expH/2, ...
            'P2', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12, 'Color', colors_spatial.P2);
    end
    hold off;
    set(gca, 'YDir', 'reverse');
    axis equal;
    xlim([0, config.isingGrid(2) + 1]);
    ylim([0, config.isingGrid(1) + 1]);
    xlabel('Column'); ylabel('Row');
    title('Spatial Sampling');
    if nPositions >= 2
        legend([h1, h2], {'P1', 'P2'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
    else
        legend(h1, {'P1'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
    end

    % Panels (1,2)-(1,5): P1 vs P2 distributions for each condition
    for c = 1:nConditions
        subplot(2, 6, 1 + c);
        cond = conditionsFound{c};

        P1 = MoransI.(cond).P1;
        P2 = MoransI.(cond).P2;
        P1 = P1(~isnan(P1));
        P2 = P2(~isnan(P2));

        hold on;
        histogram(P1, 40, 'FaceColor', colors_spatial.P1, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        histogram(P2, 40, 'FaceColor', colors_spatial.P2, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        hold off;
        xlabel('Moran''s I');
        ylabel('Count');
        title(sprintf('%s (within sim WD=%.4f)', cond, SpatialConsistency.(cond).WD));
        legend({'P1', 'P2'}, 'Location', 'best');
        grid on;
    end

    % Panel (1,6): Spatial Consistency WD bar chart
    if SpatialConsistency.available
        ax_spatial = subplot(2, 6, 6);
        wd_spatial = zeros(nConditions, 1);
        for c = 1:nConditions
            wd_spatial(c) = SpatialConsistency.(conditionsFound{c}).WD;
        end
        hold on;
        for c = 1:nConditions
            cond = conditionsFound{c};
            bar(c, wd_spatial(c), 'FaceColor', config.conditionColors.(cond));
        end
        hold off;
        xlabel('Condition');
        ylabel('Wasserstein Distance');
        title('Spatial Consistency WD(P1,P2)');
        xticks(1:nConditions);
        xticklabels(conditionsFound);
        xtickangle(45);
        grid on;
    end

    % ---------------------------------------------------------------------
    % ROW 2: Temporal Splitting (T1 vs T2)
    % ---------------------------------------------------------------------

    % Panel (2,1): Temporal explainer - Time split visualization
    subplot(2, 6, 7);
    nFrames = config.nFramesPerSim;
    halfpoint = floor(nFrames / 2);

    % Draw colored rectangles for T1 and T2 using patch
    hold on;
    hT1 = patch([0, halfpoint, halfpoint, 0], [0.6, 0.6, 1.4, 1.4], colors_temporal.T1, 'EdgeColor', 'none');
    hT2 = patch([halfpoint, nFrames, nFrames, halfpoint], [0.6, 0.6, 1.4, 1.4], colors_temporal.T2, 'EdgeColor', 'none');
    hold off;

    xlim([0 nFrames]);
    ylim([0.4 1.6]);
    xlabel('Frame');
    set(gca, 'YTick', []);
    title('Temporal Splitting');

    % Add text labels
    text(halfpoint/2, 1, 'T1', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'w');
    text(halfpoint + (nFrames-halfpoint)/2, 1, 'T2', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'w');

    legend([hT1, hT2], {'T1 (first half)', 'T2 (second half)'}, 'Location', 'southoutside', 'Orientation', 'horizontal');

    % Panels (2,2)-(2,5): T1 vs T2 distributions for each condition
    for c = 1:nConditions
        subplot(2, 6, 7 + c);
        cond = conditionsFound{c};

        % Get time series and split
        ts = MoransI.(cond).P1;  % Use P1 for temporal analysis
        nFrames_cond = length(ts);
        halfpoint_cond = floor(nFrames_cond / 2);

        T1 = ts(1:halfpoint_cond);
        T2 = ts(halfpoint_cond+1:end);
        T1 = T1(~isnan(T1));
        T2 = T2(~isnan(T2));

        hold on;
        histogram(T1, 40, 'FaceColor', colors_temporal.T1, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        histogram(T2, 40, 'FaceColor', colors_temporal.T2, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        hold off;
        xlabel('Moran''s I');
        ylabel('Count');
        title(sprintf('%s (within sim WD=%.4f)', cond, TemporalReliability.(cond).wasserstein));
        legend({'T1', 'T2'}, 'Location', 'best');
        grid on;
    end

    % Panel (2,6): Temporal Split-Half WD bar chart
    ax_temporal = subplot(2, 6, 12);
    wd_temporal = zeros(nConditions, 1);
    for c = 1:nConditions
        wd_temporal(c) = TemporalReliability.(conditionsFound{c}).wasserstein;
    end
    hold on;
    for c = 1:nConditions
        cond = conditionsFound{c};
        bar(c, wd_temporal(c), 'FaceColor', config.conditionColors.(cond));
    end
    hold off;
    xlabel('Condition');
    ylabel('Wasserstein Distance');
    title('Temporal Split-Half WD');
    xticks(1:nConditions);
    xticklabels(conditionsFound);
    xtickangle(45);
    grid on;

    % Synchronize y-axes for bar charts
    if SpatialConsistency.available
        ymax = max([max(wd_spatial), max(wd_temporal)]);
        ylim(ax_spatial, [0, ymax * 1.1]);
        ylim(ax_temporal, [0, ymax * 1.1]);
    end

    sgtitle('Distribution Comparisons: Spatial (P1 Vs P2) and Temporal (T1 vs T2)', 'FontWeight', 'bold');

    % Save Figure 1
    saveMyFig('DistributionComparisons', config.outputPath, gcf);
    fprintf('Figure 1 (Distribution Comparisons) saved\n');

    % =====================================================================
    % FIGURE 2: Summary (1x3)
    % =====================================================================
    figure('Name', 'Reliability Summary');

    % Compute WD values for summaries
    wd_spatial_sum = zeros(nConditions, 1);
    for c = 1:nConditions
        if SpatialConsistency.available
            wd_spatial_sum(c) = SpatialConsistency.(conditionsFound{c}).WD;
        end
    end

    % -------------------------------------------------------------------------
    % Panel 1: Spatial Pooling Effect
    % -------------------------------------------------------------------------
    if SpatialPooling.available
        subplot(1, 3, 1);
        WD_data = zeros(nConditions, 2);
        for c = 1:nConditions
            WD_data(c, 1) = SpatialPooling.(conditionsFound{c}).WD_single;
            WD_data(c, 2) = SpatialPooling.(conditionsFound{c}).WD_pooled;
        end
        b = bar(WD_data);
        b(1).FaceColor = [0.7 0.7 0.7];
        b(2).FaceColor = [0.2 0.6 0.8];
        xlabel('Condition');
        ylabel('Wasserstein Distance');
        title('Spatial Pooling Effect');
        xticks(1:nConditions);
        xticklabels(conditionsFound);
        xtickangle(45);
        legend({'Single (P1)', 'Pooled (P1+P2)'}, 'Location', 'best');
        grid on;
    end

    % -------------------------------------------------------------------------
    % Panel 2: Pooling % change bars
    % -------------------------------------------------------------------------
    if SpatialPooling.available
        subplot(1, 3, 2);
        change_pct = zeros(nConditions, 1);
        for c = 1:nConditions
            change_pct(c) = SpatialPooling.(conditionsFound{c}).WD_change_pct;
        end
        for c = 1:nConditions
            if change_pct(c) < 0
                barColor = [0.2 0.7 0.3];  % Green for improvement
            else
                barColor = [0.8 0.3 0.3];  % Red for worsening
            end
            bar(c, change_pct(c), 'FaceColor', barColor);
            hold on;
        end
        hold off;
        xlabel('Condition');
        ylabel('% Change in WD');
        title('Pooling Effect (neg=better)');
        xticks(1:nConditions);
        xticklabels(conditionsFound);
        xtickangle(45);
        yline(0, 'k-', 'LineWidth', 1);
        grid on;
    end

    % -------------------------------------------------------------------------
    % Panel 3: Summary
    % -------------------------------------------------------------------------
    subplot(1, 3, 3);
    axis off;
    summaryText = {
        '=== POOLING EFFECT ===', ...
        '', ...
    };
    if SpatialPooling.available
        % Per-condition % changes
        for c = 1:nConditions
            cond = conditionsFound{c};
            change_pct = SpatialPooling.(cond).WD_change_pct;
            summaryText{end+1} = sprintf('%s: %+.1f%%', upper(cond), change_pct);
        end
        summaryText{end+1} = '';

        % Aggregate statistics
        n_improved = 0;
        all_changes = zeros(nConditions, 1);
        for c = 1:nConditions
            all_changes(c) = SpatialPooling.(conditionsFound{c}).WD_change_pct;
            if SpatialPooling.(conditionsFound{c}).improved
                n_improved = n_improved + 1;
            end
        end
        summaryText{end+1} = sprintf('Improved: %d/%d', n_improved, nConditions);
        summaryText{end+1} = sprintf('Mean change: %+.1f%%', mean(all_changes));
    end
    text(0.05, 0.95, summaryText, 'VerticalAlignment', 'top', ...
        'FontSize', 9, 'FontName', 'FixedWidth', 'Units', 'normalized');
    title('Summary');

    sgtitle('Split-Half Reliability: Single vs Pooled Sampling', 'FontWeight', 'bold');

    % Save Figure 2
    saveMyFig('ReliabilitySummary', config.outputPath, gcf);
    fprintf('Figure 2 (Reliability Summary) saved\n');

    % =====================================================================
    % FIGURE 3: Data-to-Model Comparison (1x3)
    % =====================================================================
    % Test whether POOLING two spatial positions improves match to data
    if DataModelComparison.available
        figure('Name', 'Data-to-Model Comparison');

        % -----------------------------------------------------------------
        % Panel 1: WD bars (Single vs Pooled)
        % -----------------------------------------------------------------
        subplot(1, 3, 1);
        WD_data_model = zeros(nConditions, 2);
        validConditions = {};
        for c = 1:nConditions
            cond = conditionsFound{c};
            if isfield(DataModelComparison, cond)
                WD_data_model(c, 1) = DataModelComparison.(cond).WD_single;
                WD_data_model(c, 2) = DataModelComparison.(cond).WD_pooled;
                validConditions{end+1} = cond;
            end
        end
        b = bar(WD_data_model);
        b(1).FaceColor = [0.7 0.7 0.7];  % Single position (gray)
        b(2).FaceColor = [0.2 0.6 0.8];  % Pooled (blue)
        xlabel('Condition');
        ylabel('Wasserstein Distance');
        title('Data vs Model: WD Comparison');
        xticks(1:nConditions);
        xticklabels(conditionsFound);
        xtickangle(45);
        legend({'Single (P1)', 'Pooled (P1+P2)'}, 'Location', 'best');
        grid on;

        % -----------------------------------------------------------------
        % Panel 2: % Change bars
        % -----------------------------------------------------------------
        subplot(1, 3, 2);
        hold on;
        for c = 1:nConditions
            cond = conditionsFound{c};
            if isfield(DataModelComparison, cond)
                change_pct = DataModelComparison.(cond).WD_change_pct;
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
        xticklabels(conditionsFound);
        xtickangle(45);
        yline(0, 'k-', 'LineWidth', 1);
        grid on;

        % -----------------------------------------------------------------
        % Panel 3: Summary
        % -----------------------------------------------------------------
        subplot(1, 3, 3);
        axis off;
        summaryText = {
            '=== DATA-MODEL COMPARISON ===', ...
            '', ...
            'WD(Experimental, Simulation)', ...
            '', ...
        };
        n_improved = 0;
        all_changes = [];
        for c = 1:nConditions
            cond = conditionsFound{c};
            if isfield(DataModelComparison, cond)
                summaryText{end+1} = sprintf('%s:', upper(cond));
                summaryText{end+1} = sprintf('  P1: %.4f', DataModelComparison.(cond).WD_single);
                summaryText{end+1} = sprintf('  Pooled: %.4f (%+.1f%%)', ...
                    DataModelComparison.(cond).WD_pooled, DataModelComparison.(cond).WD_change_pct);
                all_changes(end+1) = DataModelComparison.(cond).WD_change_pct;
                if DataModelComparison.(cond).improved
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

        sgtitle('Data-to-Model: Single vs Pooled Sampling', 'FontWeight', 'bold');

        % Save Figure 3
        saveMyFig('DataModelComparison', config.outputPath, gcf);
        fprintf('Figure 3 (Data-to-Model Comparison) saved\n');
    end

else
    % ALL_SIMS MODE visualization
    figure('Name', 'All-Sims Reliability Analysis');

    % Panel 1: Half1 vs Half2 scatter
    subplot(1, 3, 1);
    mean1 = AllSimsData.half1_mean;
    mean2 = AllSimsData.half2_mean;
    validIdx = ~isnan(mean1) & ~isnan(mean2);
    scatter(mean1(validIdx), mean2(validIdx), 20, [0.2 0.4 0.8], 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    lims = [min([mean1(validIdx); mean2(validIdx)]), max([mean1(validIdx); mean2(validIdx)])];
    plot(lims, lims, 'k--', 'LineWidth', 1.5);
    hold off;
    xlabel('Half 1 Mean Moran''s I');
    ylabel('Half 2 Mean Moran''s I');
    title('Half1 vs Half2 Mean');
    axis equal;
    grid on;

    % Panel 2: Wasserstein distribution
    subplot(1, 3, 2);
    wassDist = AllSimsData.within_wasserstein;
    histogram(wassDist(~isnan(wassDist)), 50, 'FaceColor', [0.6 0.2 0.6]);
    xlabel('Within-Sim Wasserstein Distance');
    ylabel('Count');
    title(sprintf('Mean WD=%.4f', TemporalReliability.aggregate.wasserstein_mean));
    grid on;

    % Panel 3: Summary
    subplot(1, 3, 3);
    axis off;
    summaryText = {
        '=== ALL-SIMS RELIABILITY ===', ...
        '', ...
        sprintf('Simulations: %d', nSims), ...
        sprintf('Mean WD: %.4f', TemporalReliability.aggregate.wasserstein_mean), ...
        sprintf('95th pctl WD: %.4f', TemporalReliability.aggregate.wasserstein_p95), ...
    };
    text(0.05, 0.95, summaryText, 'VerticalAlignment', 'top', ...
        'FontSize', 10, 'FontName', 'FixedWidth', 'Units', 'normalized');
    title('Summary');

    sgtitle('All-Sims Reliability Analysis', 'FontWeight', 'bold');

    saveMyFig('AllSims_ReliabilityAnalysis', config.outputPath, gcf);
end

%% =========================================================================
%% SECTION 10: Save Results
%% =========================================================================

fprintf('\n--- Section 10: Saving Results ---\n');

CombinedResults = struct();
CombinedResults.config = config;
CombinedResults.positions = positions;
CombinedResults.nPositions = nPositions;
CombinedResults.timestamp = datetime('now');

if strcmp(config.analysisMode, 'best_match')
    CombinedResults.MoransI = MoransI;
    CombinedResults.bestMatch = bestMatch;
    CombinedResults.conditionsFound = conditionsFound;
end

CombinedResults.TemporalReliability = TemporalReliability;
CombinedResults.SpatialConsistency = SpatialConsistency;
CombinedResults.SpatialPooling = SpatialPooling;
CombinedResults.DataModelComparison = DataModelComparison;

if strcmp(config.analysisMode, 'all_sims')
    CombinedResults.AllSimsData = AllSimsData;
    CombinedResults.nSims = nSims;
end

resultsFile = fullfile(config.outputPath, 'CombinedReliabilityAnalysis_Results.mat');
save(resultsFile, 'CombinedResults', '-v7.3');
fprintf('Results saved to: %s\n', resultsFile);

%% =========================================================================
%% SECTION 11: Summary Report
%% =========================================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('  COMBINED RELIABILITY ANALYSIS\n');
fprintf('========================================\n');
fprintf('\n');
fprintf('CONFIGURATION:\n');
fprintf('  Analysis mode: %s\n', config.analysisMode);
fprintf('  Simulation grid: [%d x %d]\n', config.isingGrid(1), config.isingGrid(2));
fprintf('  Experimental grid: [%d x %d]\n', config.experimentalGrid(1), config.experimentalGrid(2));
fprintf('  Non-overlapping positions: %d\n', nPositions);
fprintf('  Frames per simulation: %d\n', config.nFramesPerSim);
fprintf('\n');

fprintf('----------------------------------------\n');
fprintf('1. TEMPORAL SPLIT-HALF RELIABILITY:\n');
fprintf('----------------------------------------\n');
if strcmp(config.analysisMode, 'best_match')
    for c = 1:nConditions
        cond = conditionsFound{c};
        fprintf('  %s: WD=%.4f\n', upper(cond), TemporalReliability.(cond).wasserstein);
    end
else
    fprintf('  Mean WD: %.4f\n', TemporalReliability.aggregate.wasserstein_mean);
    fprintf('  95th pctl WD: %.4f\n', TemporalReliability.aggregate.wasserstein_p95);
end
fprintf('\n');

fprintf('----------------------------------------\n');
fprintf('2. SPATIAL SAMPLING CONSISTENCY:\n');
fprintf('----------------------------------------\n');
if SpatialConsistency.available
    if strcmp(config.analysisMode, 'best_match')
        for c = 1:nConditions
            cond = conditionsFound{c};
            fprintf('  %s: WD(P1,P2)=%.4f\n', cond, SpatialConsistency.(cond).WD);
        end
    else
        fprintf('  Aggregate WD: %.4f\n', SpatialConsistency.aggregate.WD);
    end
else
    fprintf('  N/A (need >= 2 positions)\n');
end
fprintf('\n');

fprintf('----------------------------------------\n');
fprintf('3. SPATIAL POOLING EFFECT:\n');
fprintf('----------------------------------------\n');
if SpatialPooling.available && strcmp(config.analysisMode, 'best_match')
    for c = 1:nConditions
        cond = conditionsFound{c};
        if SpatialPooling.(cond).improved
            status = 'IMPROVED';
        else
            status = 'worsened';
        end
        fprintf('  %s: %.4f -> %.4f (%+.1f%%, %s)\n', cond, ...
            SpatialPooling.(cond).WD_single, SpatialPooling.(cond).WD_pooled, ...
            SpatialPooling.(cond).WD_change_pct, status);
    end
else
    fprintf('  N/A\n');
end
fprintf('\n');

fprintf('----------------------------------------\n');
fprintf('4. DATA-TO-MODEL COMPARISON:\n');
fprintf('----------------------------------------\n');
fprintf('   WD(Experimental, Simulation)\n');
if DataModelComparison.available && strcmp(config.analysisMode, 'best_match')
    for c = 1:nConditions
        cond = conditionsFound{c};
        if isfield(DataModelComparison, cond)
            if DataModelComparison.(cond).improved
                status = 'IMPROVED';
            else
                status = 'worsened';
            end
            fprintf('  %s: P1=%.4f, Pooled=%.4f (%+.1f%%, %s)\n', cond, ...
                DataModelComparison.(cond).WD_single, DataModelComparison.(cond).WD_pooled, ...
                DataModelComparison.(cond).WD_change_pct, status);
        end
    end
else
    fprintf('  N/A\n');
end
fprintf('\n');
fprintf('========================================\n');

%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================

function tau = computeACFDecayTime(timeseries, maxLag, threshold)
    % Compute the lag at which autocorrelation drops below threshold
    %
    % Inputs:
    %   timeseries - 1D time series
    %   maxLag     - maximum lag to compute
    %   threshold  - threshold value (default: 1/e)
    %
    % Output:
    %   tau - decay time (lag at which ACF < threshold), Inf if never drops

    if nargin < 3
        threshold = 1/exp(1);
    end

    if length(timeseries) < maxLag + 1
        tau = NaN;
        return;
    end

    timeseries = timeseries(:);
    timeseries = timeseries - mean(timeseries);

    if std(timeseries) == 0
        tau = Inf;
        return;
    end

    [acf, lags] = xcorr(timeseries, maxLag, 'normalized');
    acf = acf(maxLag+1:end);
    lags = lags(maxLag+1:end);

    belowThreshold = find(acf < threshold, 1, 'first');

    if isempty(belowThreshold)
        tau = Inf;
    else
        tau = lags(belowThreshold);
    end
end

function d = wasserstein_1d(x, y)
    % Compute 1D Wasserstein (Earth Mover's) distance between two samples

    x = x(:);
    y = y(:);
    x = x(~isnan(x));
    y = y(~isnan(y));

    if isempty(x) || isempty(y)
        d = NaN;
        return;
    end

    x = sort(x);
    y = sort(y);

    n = min(1000, min(length(x), length(y)));
    q = linspace(0, 1, n);

    x_quantiles = quantile(x, q);
    y_quantiles = quantile(y, q);

    d = mean(abs(x_quantiles - y_quantiles));
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
