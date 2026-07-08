%% =========================================================================
%% Figure 5: Combined Ising Analysis (Reliability + Sampling Convergence + Spatial Homogeneity)
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
% Merge of three previously-separate Figure 5 scripts into a single
% entry-point driven by an IsingComparison_Results_*.mat file:
%
%   (1) Reliability analysis
%       (temporal split-half, spatial consistency, spatial pooling,
%        data-to-model comparison)
%   (2) Sampling convergence analysis
%       (Centre / P1 / P1+P2 WD curves, square-grid scan,
%        distance-from-centre scan)
%   (3) Spatial homogeneity analysis
%       (2x2 tiled WD map across a single best-match simulation)
%
% All three sections now source their simulation(s) from the comparison
% file's Results.Comparison.<Condition>.bestMatch_simIDs — there is no
% longer an 'all_sims' mode. Each section is individually toggleable.
%
% INPUT:
%   - IsingComparison_Results_*.mat (from cluster Python pipeline)
%   - sim_<ID>.mat files (raw Ising simulations) referenced by best-match IDs
%   - (optional) DataForAbir.mat for the data-to-model comparison figure
%
% OUTPUT:
%   - <outputPathBase>/Reliability/           — 3 figures + CombinedReliabilityAnalysis_Results.mat
%   - <outputPathBase>/SamplingConvergence/   — 4 figures + SamplingConvergence_Results.mat
%   - <outputPathBase>/SpatialHomogeneity/    — 1 figure  + SpatialHomogeneity_Results.mat

%% =========================================================================
%% SECTION 1: Configuration
%% =========================================================================

fprintf('\n=== Figure 5: Combined Ising Analysis ===\n\n');

config = struct();

% -------------------------------------------------------------------------
% Section toggles (all default true)
% -------------------------------------------------------------------------
config.runReliability         = true;
config.runSamplingConvergence = true;
config.runSpatialHomogeneity  = true;

% -------------------------------------------------------------------------
% Shared paths and data identifiers
% -------------------------------------------------------------------------
config.isingDataPath       = mba_p('IsingModelData_39x78_100K');
config.isingComparisonPath = mba_p('IsingComparison_Results_full_trial_subselect_centre_vs_tiled_moransI+activity_optimized.mat');
config.outputPathBase      = '.';

% -------------------------------------------------------------------------
% Shared conditions and colours
% -------------------------------------------------------------------------
config.conditions = {'Naive', 'Beginner', 'Expert', 'NoSpout'};

config.conditionColors.Naive    = [0.3373, 0.7059, 0.9137];   % Light Blue
config.conditionColors.Beginner = [0.8431, 0.2549, 0.6078];   % Magenta/Pink
config.conditionColors.Expert   = [0,      0.6196, 0.4510];   % Teal/Green
config.conditionColors.NoSpout  = [0.8353, 0.3686, 0     ];   % Orange

% -------------------------------------------------------------------------
% Reliability sub-config
% -------------------------------------------------------------------------
config.reliability.experimentalGrid     = [13, 26];
config.reliability.autocorrMaxLag       = 200;
config.reliability.acfThreshold         = 1/exp(1);
config.reliability.missing_sim          = 493;
config.reliability.experimentalDataPath = 'Fig. 5 Model\IsingModels\Data\DataForAbir.mat';
config.reliability.outputPath           = fullfile(config.outputPathBase, 'Reliability');

% -------------------------------------------------------------------------
% SamplingConvergence sub-config
% -------------------------------------------------------------------------
config.samplingConvergence.expGrid          = [4, 4];
config.samplingConvergence.inverseFractions = [16, 16/sqrt(2), 8, 8/sqrt(2), 4, 4/sqrt(2), 2];
config.samplingConvergence.squareGridSizes  = [10, 8, 6, 4, 2];
config.samplingConvergence.distGridSize     = 2;
config.samplingConvergence.topN             = 1;      % per-condition best matches to pool
config.samplingConvergence.forceRecompute   = false;
config.samplingConvergence.outputPath       = fullfile(config.outputPathBase, 'SamplingConvergence');

% -------------------------------------------------------------------------
% SpatialHomogeneity sub-config
% -------------------------------------------------------------------------
config.spatialHomogeneity.analysisGridSize = 2;
config.spatialHomogeneity.sourceCondition  = 'Expert';  % which condition's best match
config.spatialHomogeneity.sourceRank       = 1;         % 1 = top-1 best match
config.spatialHomogeneity.outputPath       = fullfile(config.outputPathBase, 'SpatialHomogeneity');

fprintf('Section toggles:  Reliability=%d  SamplingConvergence=%d  SpatialHomogeneity=%d\n', ...
    config.runReliability, config.runSamplingConvergence, config.runSpatialHomogeneity);
fprintf('Comparison file:  %s\n', config.isingComparisonPath);
fprintf('Ising data path:  %s\n', config.isingDataPath);
fprintf('Output base:      %s\n\n', config.outputPathBase);

%% =========================================================================
%% SECTION 2: Shared Setup — Load comparison file and build bestMatchMap
%% =========================================================================

fprintf('--- Section 2: Loading IsingComparison_Results (HDF5) and building best-match map ---\n');

if ~exist(config.isingComparisonPath, 'file')
    error('IsingComparison_Results file not found: %s', config.isingComparisonPath);
end

% The comparison file is Python HDF5, NOT a MATLAB .mat — use h5read.
bestMatchMap = loadComparisonBestMatches(config.isingComparisonPath, config.conditions);

conditionsFound = fieldnames(bestMatchMap)';
for c = 1:length(conditionsFound)
    cond = conditionsFound{c};
    ids  = bestMatchMap.(cond).simIDs;
    p    = bestMatchMap.(cond).params;
    fprintf('  %s: top-1 %s (beta=%.2f, c=%.2f), %d total best matches\n', ...
        cond, ids{1}, p.beta(1), p.c(1), length(ids));
end
for c = 1:length(config.conditions)
    if ~ismember(config.conditions{c}, conditionsFound)
        warning('Condition %s not found in comparison file — skipping.', config.conditions{c});
    end
end

if isempty(conditionsFound)
    error('No configured conditions were found in the comparison file — nothing to analyse.');
end

% Auto-detect isingGrid and nFramesPerSim from the first condition's top-1
% best-match simulation. If the file is missing, print copy instructions.
firstSimID   = bestMatchMap.(conditionsFound{1}).simIDs{1};
firstSimPath = requireSimPath(firstSimID, config.isingDataPath);

firstSimInfo = whos('-file', firstSimPath, 'stored_spins');
% stored_spins is [T x rows x cols]
config.nFramesPerSim = firstSimInfo.size(1);
config.isingGrid     = [firstSimInfo.size(2), firstSimInfo.size(3)];

fprintf('Grid auto-detection: using %s (top-1 of %s)\n', firstSimID, conditionsFound{1});
fprintf('Auto-detected Ising grid: [%d x %d], frames per sim: %d\n', ...
    config.isingGrid(1), config.isingGrid(2), config.nFramesPerSim);

clear firstSimInfo firstSimPath firstSimID;
fprintf('Best-match map ready for %d condition(s): %s\n\n', ...
    length(conditionsFound), strjoin(conditionsFound, ', '));

%% =========================================================================
%% SECTION 3: RELIABILITY ANALYSIS
%% =========================================================================

if config.runReliability
    fprintf('\n###########################################################\n');
    fprintf('##              SECTION 3: RELIABILITY                   ##\n');
    fprintf('###########################################################\n\n');

    rel = config.reliability;
    if ~exist(rel.outputPath, 'dir'); mkdir(rel.outputPath); end

    % -------------------------------------------------------------------------
    % 3.1 Grid positions and weight matrix
    % -------------------------------------------------------------------------
    fprintf('--- 3.1 Setting up grid positions and weight matrix ---\n');

    rel_positions = generateNonOverlappingPositions(config.isingGrid, rel.experimentalGrid);
    rel_nPositions = size(rel_positions, 1);

    fprintf('Non-overlapping positions: %d\n', rel_nPositions);
    for p = 1:rel_nPositions
        rowStart = rel_positions(p, 1);
        colStart = rel_positions(p, 2);
        rowEnd = rowStart + rel.experimentalGrid(1) - 1;
        colEnd = colStart + rel.experimentalGrid(2) - 1;
        fprintf('  Position %d: rows %d:%d, cols %d:%d\n', p, rowStart, rowEnd, colStart, colEnd);
    end

    if rel_nPositions < 2
        warning('Only %d non-overlapping position(s); spatial analyses will be skipped.', rel_nPositions);
    end

    % Weight matrix for the experimental grid (nearest-neighbour)
    valueMap_exp    = rand(rel.experimentalGrid(1), rel.experimentalGrid(2));
    distanceMat_exp = squareform(mL_distanceMat(valueMap_exp));
    uniqueDist_exp  = unique(distanceMat_exp);
    uniqueDist_exp(uniqueDist_exp == 0) = [];
    currDistInds_exp = ismember(distanceMat_exp, uniqueDist_exp(1));
    rel_weightMat = zeros(size(distanceMat_exp));
    rel_weightMat(currDistInds_exp) = distanceMat_exp(currDistInds_exp);
    rel_weightMat(rel_weightMat == inf) = 0;

    rel_P1_rows = rel_positions(1,1):(rel_positions(1,1) + rel.experimentalGrid(1) - 1);
    rel_P1_cols = rel_positions(1,2):(rel_positions(1,2) + rel.experimentalGrid(2) - 1);
    if rel_nPositions >= 2
        rel_P2_rows = rel_positions(2,1):(rel_positions(2,1) + rel.experimentalGrid(1) - 1);
        rel_P2_cols = rel_positions(2,2):(rel_positions(2,2) + rel.experimentalGrid(2) - 1);
    end

    % -------------------------------------------------------------------------
    % 3.2 Load experimental Moran's I (for data-to-model comparison)
    % -------------------------------------------------------------------------
    fprintf('\n--- 3.2 Loading experimental data ---\n');
    if exist(rel.experimentalDataPath, 'file')
        expData    = load(rel.experimentalDataPath);
        MoransI_Exp = expData.MoransI;
        experimentalDataAvailable = true;
        fprintf('Loaded experimental Moran''s I from: %s\n', rel.experimentalDataPath);
        for c = 1:length(conditionsFound)
            condition = conditionsFound{c};
            if isfield(MoransI_Exp, condition) && ~isempty(MoransI_Exp.(condition))
                m = MoransI_Exp.(condition);
                fprintf('  %s: %d trials x %d frames, mean I=%.4f\n', ...
                    condition, size(m,1), size(m,2), mean(m(:), 'omitnan'));
            end
        end
    else
        warning('Experimental data file not found: %s', rel.experimentalDataPath);
        experimentalDataAvailable = false;
        MoransI_Exp = struct();
    end

    % -------------------------------------------------------------------------
    % 3.3 Compute Moran's I time series per condition (top-1 best-match sim)
    % -------------------------------------------------------------------------
    fprintf('\n--- 3.3 Computing Moran''s I per condition (best-match) ---\n');
    MoransI = struct();
    bestMatch = struct();
    nConditions = length(conditionsFound);

    % Validate that every condition's top-1 sim file exists locally
    rel_requiredSimIDs = cell(nConditions, 1);
    for c = 1:nConditions
        rel_requiredSimIDs{c} = bestMatchMap.(conditionsFound{c}).simIDs{1};
    end
    assertSimsExist(rel_requiredSimIDs, config.isingDataPath, 'Reliability');

    for c = 1:nConditions
        cond = conditionsFound{c};
        simID = bestMatchMap.(cond).simIDs{1};
        simPath = fullfile(config.isingDataPath, simID2Filename(simID));
        bestMatch.(cond).simID = simID;
        bestMatch.(cond).params = struct( ...
            'beta',             bestMatchMap.(cond).params.beta(1), ...
            'c',                bestMatchMap.(cond).params.c(1), ...
            'decay_const',      bestMatchMap.(cond).params.decay_const(1), ...
            'inhibition_range', bestMatchMap.(cond).params.inhibition_range(1), ...
            'bias',             bestMatchMap.(cond).params.bias(1));

        fprintf('  Processing %s (%s)...\n', cond, simID);

        simData = load(simPath);
        stored_spins = simData.stored_spins;
        nFrames = size(stored_spins, 1);

        moransI_P1 = zeros(1, nFrames);
        for t = 1:nFrames
            frame = squeeze(stored_spins(t, rel_P1_rows, rel_P1_cols));
            if all(frame(:) == 0) || all(frame(:) == 1)
                moransI_P1(t) = NaN;
            else
                moransI_P1(t) = mL_moransI(double(frame), rel_weightMat);
            end
        end

        if rel_nPositions >= 2
            moransI_P2 = zeros(1, nFrames);
            for t = 1:nFrames
                frame = squeeze(stored_spins(t, rel_P2_rows, rel_P2_cols));
                if all(frame(:) == 0) || all(frame(:) == 1)
                    moransI_P2(t) = NaN;
                else
                    moransI_P2(t) = mL_moransI(double(frame), rel_weightMat);
                end
            end
        else
            moransI_P2 = [];
        end

        MoransI.(cond).P1      = moransI_P1;
        MoransI.(cond).P2      = moransI_P2;
        MoransI.(cond).simID   = simID;
        MoransI.(cond).nFrames = nFrames;

        fprintf('    P1: mean=%.4f, std=%.4f\n', mean(moransI_P1, 'omitnan'), std(moransI_P1, 'omitnan'));
        if rel_nPositions >= 2
            fprintf('    P2: mean=%.4f, std=%.4f\n', mean(moransI_P2, 'omitnan'), std(moransI_P2, 'omitnan'));
        end
        clear simData stored_spins;
    end

    % -------------------------------------------------------------------------
    % 3.4 Temporal split-half reliability
    % -------------------------------------------------------------------------
    fprintf('\n--- 3.4 Temporal split-half reliability ---\n');
    TemporalReliability = struct();

    for c = 1:nConditions
        cond = conditionsFound{c};
        ts = MoransI.(cond).P1;
        nFrames = MoransI.(cond).nFrames;
        halfpoint = floor(nFrames / 2);

        half1 = ts(1:halfpoint);
        half2 = ts(halfpoint+1:end);
        h1 = half1(~isnan(half1));
        h2 = half2(~isnan(half2));

        TemporalReliability.(cond).half1_mean = mean(h1);
        TemporalReliability.(cond).half2_mean = mean(h2);
        TemporalReliability.(cond).half1_std  = std(h1);
        TemporalReliability.(cond).half2_std  = std(h2);
        TemporalReliability.(cond).wasserstein = wasserstein_1d(h1, h2);

        minLen = min(length(h1), length(h2));
        h1m = h1(1:minLen); h2m = h2(1:minLen);
        TemporalReliability.(cond).ba_averages    = (h1m + h2m) / 2;
        TemporalReliability.(cond).ba_differences = h2m - h1m;
        TemporalReliability.(cond).ba_bias        = mean(h2m - h1m);
        TemporalReliability.(cond).ba_loa_upper   = TemporalReliability.(cond).ba_bias + 1.96 * std(h2m - h1m);
        TemporalReliability.(cond).ba_loa_lower   = TemporalReliability.(cond).ba_bias - 1.96 * std(h2m - h1m);

        fprintf('  %s: WD=%.4f\n', cond, TemporalReliability.(cond).wasserstein);
    end

    % -------------------------------------------------------------------------
    % 3.5 Spatial sampling consistency (P1 vs P2)
    % -------------------------------------------------------------------------
    fprintf('\n--- 3.5 Spatial sampling consistency ---\n');
    SpatialConsistency = struct();

    if rel_nPositions < 2
        SpatialConsistency.available = false;
        fprintf('  Skipping — requires at least 2 non-overlapping positions\n');
    else
        SpatialConsistency.available = true;
        for c = 1:nConditions
            cond = conditionsFound{c};
            P1 = MoransI.(cond).P1;
            P2 = MoransI.(cond).P2;

            SpatialConsistency.(cond).WD        = wasserstein_1d(P1, P2);
            SpatialConsistency.(cond).P1_mean   = mean(P1, 'omitnan');
            SpatialConsistency.(cond).P2_mean   = mean(P2, 'omitnan');
            SpatialConsistency.(cond).P1_std    = std(P1, 'omitnan');
            SpatialConsistency.(cond).P2_std    = std(P2, 'omitnan');
            SpatialConsistency.(cond).mean_diff = SpatialConsistency.(cond).P2_mean - SpatialConsistency.(cond).P1_mean;

            fprintf('  %s: WD(P1,P2)=%.4f, mean_diff=%.4f\n', cond, ...
                SpatialConsistency.(cond).WD, SpatialConsistency.(cond).mean_diff);
        end
    end

    % -------------------------------------------------------------------------
    % 3.6 Spatial pooling effect (within simulation)
    % -------------------------------------------------------------------------
    fprintf('\n--- 3.6 Spatial pooling effect ---\n');
    SpatialPooling = struct();

    if rel_nPositions < 2
        SpatialPooling.available = false;
        fprintf('  Skipping — requires at least 2 non-overlapping positions\n');
    else
        SpatialPooling.available = true;
        for c = 1:nConditions
            cond = conditionsFound{c};
            P1 = MoransI.(cond).P1;
            P2 = MoransI.(cond).P2;
            nFrames = MoransI.(cond).nFrames;
            halfpoint = floor(nFrames / 2);

            half1_single = P1(1:halfpoint);
            half2_single = P1(halfpoint+1:end);
            half1_pooled = [P1(1:halfpoint),      P2(1:halfpoint)];
            half2_pooled = [P1(halfpoint+1:end),  P2(halfpoint+1:end)];

            h1s = half1_single(~isnan(half1_single));
            h2s = half2_single(~isnan(half2_single));
            h1p = half1_pooled(~isnan(half1_pooled));
            h2p = half2_pooled(~isnan(half2_pooled));

            SpatialPooling.(cond).WD_single     = wasserstein_1d(h1s, h2s);
            SpatialPooling.(cond).WD_pooled     = wasserstein_1d(h1p, h2p);
            SpatialPooling.(cond).WD_change_pct = 100 * ...
                (SpatialPooling.(cond).WD_pooled - SpatialPooling.(cond).WD_single) / ...
                SpatialPooling.(cond).WD_single;
            SpatialPooling.(cond).improved = SpatialPooling.(cond).WD_pooled < SpatialPooling.(cond).WD_single;

            fprintf('  %s: WD single=%.4f, pooled=%.4f (change: %+.1f%%)\n', cond, ...
                SpatialPooling.(cond).WD_single, SpatialPooling.(cond).WD_pooled, ...
                SpatialPooling.(cond).WD_change_pct);
        end
    end

    % -------------------------------------------------------------------------
    % 3.7 Data-to-model spatial pooling effect
    % -------------------------------------------------------------------------
    fprintf('\n--- 3.7 Data-to-model spatial pooling effect ---\n');
    DataModelComparison = struct();

    if experimentalDataAvailable && rel_nPositions >= 2
        DataModelComparison.available = true;

        for c = 1:nConditions
            cond = conditionsFound{c};
            if ~isfield(MoransI_Exp, cond) || isempty(MoransI_Exp.(cond))
                fprintf('  %s: no experimental data\n', cond);
                continue;
            end

            exp_moransI = MoransI_Exp.(cond)(:);
            exp_moransI = exp_moransI(~isnan(exp_moransI));

            sim_P1     = MoransI.(cond).P1(:);
            sim_pooled = [MoransI.(cond).P1(:); MoransI.(cond).P2(:)];

            DataModelComparison.(cond).WD_single = wasserstein_1d(exp_moransI, sim_P1);
            DataModelComparison.(cond).WD_pooled = wasserstein_1d(exp_moransI, sim_pooled);
            DataModelComparison.(cond).WD_change_pct = 100 * ...
                (DataModelComparison.(cond).WD_pooled - DataModelComparison.(cond).WD_single) / ...
                DataModelComparison.(cond).WD_single;
            DataModelComparison.(cond).improved = ...
                DataModelComparison.(cond).WD_pooled < DataModelComparison.(cond).WD_single;
            DataModelComparison.(cond).exp_mean        = mean(exp_moransI);
            DataModelComparison.(cond).sim_P1_mean     = mean(sim_P1,    'omitnan');
            DataModelComparison.(cond).sim_pooled_mean = mean(sim_pooled,'omitnan');

            fprintf('  %s: WD(Data,P1)=%.4f, WD(Data,Pooled)=%.4f (%+.1f%%)\n', cond, ...
                DataModelComparison.(cond).WD_single, DataModelComparison.(cond).WD_pooled, ...
                DataModelComparison.(cond).WD_change_pct);
        end
    else
        DataModelComparison.available = false;
        if ~experimentalDataAvailable
            fprintf('  Skipping — experimental data not available\n');
        else
            fprintf('  Skipping — requires at least 2 non-overlapping positions\n');
        end
    end

    % -------------------------------------------------------------------------
    % 3.8 Visualisation
    % -------------------------------------------------------------------------
    fprintf('\n--- 3.8 Creating reliability figures ---\n');

    colors_spatial  = struct('P1', [0.2 0.4 0.8], 'P2', [0.8 0.4 0.2]);
    colors_temporal = struct('T1', [0.2 0.6 0.4], 'T2', [0.6 0.2 0.6]);

    % ---- FIGURE 1: Distribution Comparisons (2x6) ------------------------
    figure('Name', 'Distribution Comparisons');

    % Panel (1,1): spatial explainer
    subplot(2, 6, 1); hold on;
    rectangle('Position', [0.5, 0.5, config.isingGrid(2), config.isingGrid(1)], ...
        'EdgeColor', 'k', 'LineWidth', 2);
    expW = rel.experimentalGrid(2);
    expH = rel.experimentalGrid(1);

    rowStart = rel_positions(1,1); colStart = rel_positions(1,2);
    h1 = patch([colStart-0.5, colStart-0.5+expW, colStart-0.5+expW, colStart-0.5], ...
               [rowStart-0.5, rowStart-0.5, rowStart-0.5+expH, rowStart-0.5+expH], ...
               colors_spatial.P1, 'FaceAlpha', 0.3, 'EdgeColor', colors_spatial.P1, 'LineWidth', 2);
    text(colStart + expW/2, rowStart + expH/2, 'P1', ...
        'HorizontalAlignment','center','FontWeight','bold','FontSize',12,'Color',colors_spatial.P1);

    if rel_nPositions >= 2
        rowStart = rel_positions(2,1); colStart = rel_positions(2,2);
        h2 = patch([colStart-0.5, colStart-0.5+expW, colStart-0.5+expW, colStart-0.5], ...
                   [rowStart-0.5, rowStart-0.5, rowStart-0.5+expH, rowStart-0.5+expH], ...
                   colors_spatial.P2, 'FaceAlpha', 0.3, 'EdgeColor', colors_spatial.P2, 'LineWidth', 2);
        text(colStart + expW/2, rowStart + expH/2, 'P2', ...
            'HorizontalAlignment','center','FontWeight','bold','FontSize',12,'Color',colors_spatial.P2);
    end
    hold off;
    set(gca, 'YDir', 'reverse'); axis equal;
    xlim([0, config.isingGrid(2) + 1]); ylim([0, config.isingGrid(1) + 1]);
    xlabel('Column'); ylabel('Row'); title('Spatial Sampling');
    if rel_nPositions >= 2
        legend([h1, h2], {'P1', 'P2'}, 'Location','southoutside','Orientation','horizontal');
    else
        legend(h1, {'P1'}, 'Location','southoutside','Orientation','horizontal');
    end

    % Panels (1,2)-(1,5): P1 vs P2 distributions per condition
    for c = 1:nConditions
        subplot(2, 6, 1 + c);
        cond = conditionsFound{c};
        P1 = MoransI.(cond).P1; P2 = MoransI.(cond).P2;
        P1 = P1(~isnan(P1)); P2 = P2(~isnan(P2));
        hold on;
        histogram(P1, 40, 'FaceColor', colors_spatial.P1, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        histogram(P2, 40, 'FaceColor', colors_spatial.P2, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        hold off;
        xlabel('Moran''s I'); ylabel('Count');
        if SpatialConsistency.available
            title(sprintf('%s (within sim WD=%.4f)', cond, SpatialConsistency.(cond).WD));
        else
            title(cond);
        end
        legend({'P1','P2'}, 'Location', 'best'); grid on;
    end

    % Panel (1,6): spatial WD bar
    if SpatialConsistency.available
        ax_spatial = subplot(2, 6, 6);
        wd_spatial = zeros(nConditions, 1);
        for c = 1:nConditions
            wd_spatial(c) = SpatialConsistency.(conditionsFound{c}).WD;
        end
        hold on;
        for c = 1:nConditions
            bar(c, wd_spatial(c), 'FaceColor', config.conditionColors.(conditionsFound{c}));
        end
        hold off;
        xlabel('Condition'); ylabel('Wasserstein Distance');
        title('Spatial Consistency WD(P1,P2)');
        xticks(1:nConditions); xticklabels(conditionsFound); xtickangle(45); grid on;
    end

    % Panel (2,1): temporal explainer
    subplot(2, 6, 7);
    nFrames = config.nFramesPerSim;
    halfpoint = floor(nFrames / 2);
    hold on;
    hT1 = patch([0, halfpoint, halfpoint, 0],        [0.6, 0.6, 1.4, 1.4], colors_temporal.T1, 'EdgeColor', 'none');
    hT2 = patch([halfpoint, nFrames, nFrames, halfpoint], [0.6, 0.6, 1.4, 1.4], colors_temporal.T2, 'EdgeColor', 'none');
    hold off;
    xlim([0 nFrames]); ylim([0.4 1.6]);
    xlabel('Frame'); set(gca, 'YTick', []); title('Temporal Splitting');
    text(halfpoint/2, 1, 'T1', 'HorizontalAlignment','center','FontWeight','bold','FontSize',12,'Color','w');
    text(halfpoint + (nFrames-halfpoint)/2, 1, 'T2', 'HorizontalAlignment','center','FontWeight','bold','FontSize',12,'Color','w');
    legend([hT1, hT2], {'T1 (first half)','T2 (second half)'}, 'Location','southoutside','Orientation','horizontal');

    % Panels (2,2)-(2,5): T1 vs T2 distributions per condition
    for c = 1:nConditions
        subplot(2, 6, 7 + c);
        cond = conditionsFound{c};
        ts = MoransI.(cond).P1;
        nFrames_c = length(ts);
        halfpoint_c = floor(nFrames_c / 2);

        T1 = ts(1:halfpoint_c);
        T2 = ts(halfpoint_c+1:end);
        T1 = T1(~isnan(T1)); T2 = T2(~isnan(T2));

        hold on;
        histogram(T1, 40, 'FaceColor', colors_temporal.T1, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        histogram(T2, 40, 'FaceColor', colors_temporal.T2, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        hold off;
        xlabel('Moran''s I'); ylabel('Count');
        title(sprintf('%s (within sim WD=%.4f)', cond, TemporalReliability.(cond).wasserstein));
        legend({'T1','T2'}, 'Location', 'best'); grid on;
    end

    % Panel (2,6): temporal WD bar
    ax_temporal = subplot(2, 6, 12);
    wd_temporal = zeros(nConditions, 1);
    for c = 1:nConditions
        wd_temporal(c) = TemporalReliability.(conditionsFound{c}).wasserstein;
    end
    hold on;
    for c = 1:nConditions
        bar(c, wd_temporal(c), 'FaceColor', config.conditionColors.(conditionsFound{c}));
    end
    hold off;
    xlabel('Condition'); ylabel('Wasserstein Distance');
    title('Temporal Split-Half WD');
    xticks(1:nConditions); xticklabels(conditionsFound); xtickangle(45); grid on;

    if SpatialConsistency.available
        ymax = max([max(wd_spatial), max(wd_temporal)]);
        ylim(ax_spatial,  [0, ymax * 1.1]);
        ylim(ax_temporal, [0, ymax * 1.1]);
    end

    sgtitle('Distribution Comparisons: Spatial (P1 vs P2) and Temporal (T1 vs T2)', 'FontWeight','bold');
    saveMyFig('DistributionComparisons', rel.outputPath, gcf);
    fprintf('Saved figure: DistributionComparisons\n');

    % ---- FIGURE 2: Reliability Summary (1x3) -----------------------------
    figure('Name', 'Reliability Summary');

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
        xlabel('Condition'); ylabel('Wasserstein Distance');
        title('Spatial Pooling Effect');
        xticks(1:nConditions); xticklabels(conditionsFound); xtickangle(45);
        legend({'Single (P1)','Pooled (P1+P2)'}, 'Location','best'); grid on;

        subplot(1, 3, 2); hold on;
        change_pct = zeros(nConditions, 1);
        for c = 1:nConditions
            change_pct(c) = SpatialPooling.(conditionsFound{c}).WD_change_pct;
            if change_pct(c) < 0
                barColor = [0.2 0.7 0.3];
            else
                barColor = [0.8 0.3 0.3];
            end
            bar(c, change_pct(c), 'FaceColor', barColor);
        end
        hold off;
        xlabel('Condition'); ylabel('% Change in WD');
        title('Pooling Effect (neg=better)');
        xticks(1:nConditions); xticklabels(conditionsFound); xtickangle(45);
        yline(0, 'k-', 'LineWidth', 1); grid on;
    end

    subplot(1, 3, 3); axis off;
    summaryText = {'=== POOLING EFFECT ===',''};
    if SpatialPooling.available
        for c = 1:nConditions
            cond = conditionsFound{c};
            summaryText{end+1} = sprintf('%s: %+.1f%%', upper(cond), SpatialPooling.(cond).WD_change_pct); %#ok<SAGROW>
        end
        summaryText{end+1} = '';
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
    text(0.05, 0.95, summaryText, 'VerticalAlignment','top', ...
        'FontSize', 9, 'FontName','FixedWidth','Units','normalized');
    title('Summary');

    sgtitle('Split-Half Reliability: Single vs Pooled Sampling', 'FontWeight','bold');
    saveMyFig('ReliabilitySummary', rel.outputPath, gcf);
    fprintf('Saved figure: ReliabilitySummary\n');

    % ---- FIGURE 3: Data-to-Model Comparison ------------------------------
    if DataModelComparison.available
        figure('Name', 'Data-to-Model Comparison');

        subplot(1, 3, 1);
        WD_data_model = zeros(nConditions, 2);
        for c = 1:nConditions
            cond = conditionsFound{c};
            if isfield(DataModelComparison, cond)
                WD_data_model(c, 1) = DataModelComparison.(cond).WD_single;
                WD_data_model(c, 2) = DataModelComparison.(cond).WD_pooled;
            end
        end
        b = bar(WD_data_model);
        b(1).FaceColor = [0.7 0.7 0.7];
        b(2).FaceColor = [0.2 0.6 0.8];
        xlabel('Condition'); ylabel('Wasserstein Distance');
        title('Data vs Model: WD Comparison');
        xticks(1:nConditions); xticklabels(conditionsFound); xtickangle(45);
        legend({'Single (P1)','Pooled (P1+P2)'}, 'Location','best'); grid on;

        subplot(1, 3, 2); hold on;
        for c = 1:nConditions
            cond = conditionsFound{c};
            if isfield(DataModelComparison, cond)
                change_pct = DataModelComparison.(cond).WD_change_pct;
                if change_pct < 0
                    barColor = [0.2 0.7 0.3];
                else
                    barColor = [0.8 0.3 0.3];
                end
                bar(c, change_pct, 'FaceColor', barColor);
            end
        end
        hold off;
        xlabel('Condition'); ylabel('% Change in WD');
        title('Pooling Effect (neg = better match)');
        xticks(1:nConditions); xticklabels(conditionsFound); xtickangle(45);
        yline(0, 'k-', 'LineWidth', 1); grid on;

        subplot(1, 3, 3); axis off;
        summaryText = {'=== DATA-MODEL COMPARISON ===','','WD(Experimental, Simulation)',''};
        n_improved = 0; all_changes = [];
        for c = 1:nConditions
            cond = conditionsFound{c};
            if isfield(DataModelComparison, cond)
                summaryText{end+1} = sprintf('%s:', upper(cond)); %#ok<SAGROW>
                summaryText{end+1} = sprintf('  P1: %.4f', DataModelComparison.(cond).WD_single); %#ok<SAGROW>
                summaryText{end+1} = sprintf('  Pooled: %.4f (%+.1f%%)', ...
                    DataModelComparison.(cond).WD_pooled, DataModelComparison.(cond).WD_change_pct); %#ok<SAGROW>
                all_changes(end+1) = DataModelComparison.(cond).WD_change_pct; %#ok<SAGROW>
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
        text(0.05, 0.95, summaryText, 'VerticalAlignment','top', ...
            'FontSize', 9, 'FontName','FixedWidth','Units','normalized');
        title('Summary');

        sgtitle('Data-to-Model: Single vs Pooled Sampling', 'FontWeight','bold');
        saveMyFig('DataModelComparison', rel.outputPath, gcf);
        fprintf('Saved figure: DataModelComparison\n');
    end

    % -------------------------------------------------------------------------
    % 3.9 Save reliability results
    % -------------------------------------------------------------------------
    fprintf('\n--- 3.9 Saving reliability results ---\n');

    % Flatten the saved config to stay backwards-compatible with existing readers
    cfg_rel = struct();
    cfg_rel.analysisMode        = 'best_match';
    cfg_rel.isingDataPath       = config.isingDataPath;
    cfg_rel.isingComparisonPath = config.isingComparisonPath;
    cfg_rel.outputPath          = rel.outputPath;
    cfg_rel.isingGrid           = config.isingGrid;
    cfg_rel.nFramesPerSim       = config.nFramesPerSim;
    cfg_rel.halfFrames          = floor(config.nFramesPerSim / 2);
    cfg_rel.conditions          = config.conditions;
    cfg_rel.conditionColors     = config.conditionColors;
    cfg_rel.experimentalGrid    = rel.experimentalGrid;
    cfg_rel.autocorrMaxLag      = rel.autocorrMaxLag;
    cfg_rel.acfThreshold        = rel.acfThreshold;
    cfg_rel.isingParams.missing_sim = rel.missing_sim;
    cfg_rel.experimentalDataPath = rel.experimentalDataPath;

    CombinedResults = struct();
    CombinedResults.config              = cfg_rel;
    CombinedResults.positions           = rel_positions;
    CombinedResults.nPositions          = rel_nPositions;
    CombinedResults.timestamp           = datetime('now');
    CombinedResults.MoransI             = MoransI;
    CombinedResults.bestMatch           = bestMatch;
    CombinedResults.conditionsFound     = conditionsFound;
    CombinedResults.TemporalReliability = TemporalReliability;
    CombinedResults.SpatialConsistency  = SpatialConsistency;
    CombinedResults.SpatialPooling      = SpatialPooling;
    CombinedResults.DataModelComparison = DataModelComparison;

    resultsFile = fullfile(rel.outputPath, 'CombinedReliabilityAnalysis_Results.mat');
    save(resultsFile, 'CombinedResults', '-v7.3');
    fprintf('Results saved to: %s\n', resultsFile);

    % -------------------------------------------------------------------------
    % 3.10 Console summary report
    % -------------------------------------------------------------------------
    fprintf('\n========================================\n');
    fprintf('  RELIABILITY ANALYSIS SUMMARY\n');
    fprintf('========================================\n');
    fprintf('CONFIGURATION:\n');
    fprintf('  Ising grid: [%d x %d]\n', config.isingGrid(1), config.isingGrid(2));
    fprintf('  Experimental grid: [%d x %d]\n', rel.experimentalGrid(1), rel.experimentalGrid(2));
    fprintf('  Non-overlapping positions: %d\n', rel_nPositions);
    fprintf('  Frames per simulation: %d\n', config.nFramesPerSim);
    fprintf('\n1. TEMPORAL SPLIT-HALF RELIABILITY:\n');
    for c = 1:nConditions
        cond = conditionsFound{c};
        fprintf('   %s: WD=%.4f\n', upper(cond), TemporalReliability.(cond).wasserstein);
    end

    fprintf('\n2. SPATIAL SAMPLING CONSISTENCY:\n');
    if SpatialConsistency.available
        for c = 1:nConditions
            cond = conditionsFound{c};
            fprintf('   %s: WD(P1,P2)=%.4f\n', cond, SpatialConsistency.(cond).WD);
        end
    else
        fprintf('   N/A (need >= 2 positions)\n');
    end

    fprintf('\n3. SPATIAL POOLING EFFECT:\n');
    if SpatialPooling.available
        for c = 1:nConditions
            cond = conditionsFound{c};
            status = 'worsened';
            if SpatialPooling.(cond).improved
                status = 'IMPROVED';
            end
            fprintf('   %s: %.4f -> %.4f (%+.1f%%, %s)\n', cond, ...
                SpatialPooling.(cond).WD_single, SpatialPooling.(cond).WD_pooled, ...
                SpatialPooling.(cond).WD_change_pct, status);
        end
    else
        fprintf('   N/A\n');
    end

    fprintf('\n4. DATA-TO-MODEL COMPARISON:\n');
    if DataModelComparison.available
        for c = 1:nConditions
            cond = conditionsFound{c};
            if isfield(DataModelComparison, cond)
                status = 'worsened';
                if DataModelComparison.(cond).improved
                    status = 'IMPROVED';
                end
                fprintf('   %s: P1=%.4f, Pooled=%.4f (%+.1f%%, %s)\n', cond, ...
                    DataModelComparison.(cond).WD_single, DataModelComparison.(cond).WD_pooled, ...
                    DataModelComparison.(cond).WD_change_pct, status);
            end
        end
    else
        fprintf('   N/A\n');
    end
    fprintf('========================================\n\n');

    % Clear section-local temporaries
    clear rel rel_requiredSimIDs rel_positions rel_nPositions ...
          rel_weightMat rel_P1_rows rel_P1_cols ...
          rel_P2_rows rel_P2_cols valueMap_exp distanceMat_exp uniqueDist_exp ...
          currDistInds_exp expData MoransI_Exp experimentalDataAvailable ...
          MoransI bestMatch nConditions TemporalReliability SpatialConsistency ...
          SpatialPooling DataModelComparison CombinedResults cfg_rel resultsFile ...
          colors_spatial colors_temporal ax_spatial ax_temporal wd_spatial wd_temporal ...
          WD_data WD_data_model b change_pct barColor n_improved all_changes ...
          summaryText h1 h2 hT1 hT2 h1m h2m h1s h2s h1p h2p half1 half2 ...
          half1_single half2_single half1_pooled half2_pooled exp_moransI sim_P1 sim_pooled ...
          ts nFrames nFrames_c halfpoint halfpoint_c T1 T2 P1 P2 moransI_P1 moransI_P2 ...
          simID simPath simData stored_spins frame rowStart colStart rowEnd colEnd ...
          expW expH ymax minLen;

else
    fprintf('\n[Reliability section skipped per config.runReliability]\n\n');
end

%% =========================================================================
%% SECTION 4: SAMPLING CONVERGENCE ANALYSIS
%% =========================================================================

if config.runSamplingConvergence
    fprintf('\n###########################################################\n');
    fprintf('##           SECTION 4: SAMPLING CONVERGENCE             ##\n');
    fprintf('###########################################################\n\n');

    sc = config.samplingConvergence;
    if ~exist(sc.outputPath, 'dir'); mkdir(sc.outputPath); end

    % -------------------------------------------------------------------------
    % 4.0 Cache check
    % -------------------------------------------------------------------------
    precomputedPath = fullfile(sc.outputPath, 'SamplingConvergence_Results.mat');
    goto_figure_generation = exist(precomputedPath, 'file') && ~sc.forceRecompute;

    if goto_figure_generation
        fprintf('=========================================================\n');
        fprintf('  FOUND PRE-COMPUTED SAMPLINGCONVERGENCE RESULTS\n');
        fprintf('=========================================================\n');
        fprintf('Loading: %s\n', precomputedPath);

        precomputed = load(precomputedPath);
        if isfield(precomputed, 'Results')
            Results = precomputed.Results;
        else
            Results = precomputed;
        end
        if isfield(Results, 'config') && isfield(Results.config, 'isingGrid')
            sc_isingGrid = Results.config.isingGrid;
        else
            sc_isingGrid = config.isingGrid;
        end
        if isfield(Results, 'Aggregate');          Aggregate          = Results.Aggregate;          end
        if isfield(Results, 'SquareGridResults');  SquareGridResults  = Results.SquareGridResults;  end
        if isfield(Results, 'DistanceAnalysis');   DistanceAnalysis   = Results.DistanceAnalysis;   end
        if isfield(Results, 'simIDs')
            sc_simIDs = Results.simIDs;
            if ~iscell(sc_simIDs); sc_simIDs = cellstr(sc_simIDs); end
            sc_nSims  = length(sc_simIDs);
        else
            sc_simIDs = {};
            sc_nSims  = 1;
        end
        fprintf('Loaded results for %d simulations. Skipping compute.\n\n', sc_nSims);
        clear precomputed;
    else
        if exist(precomputedPath, 'file')
            fprintf('forceRecompute=true — ignoring existing %s\n', precomputedPath);
        else
            fprintf('No pre-computed results at: %s — will compute from scratch.\n', precomputedPath);
        end
    end

    % -------------------------------------------------------------------------
    % 4.1 Compute from scratch (gated by cache check)
    % -------------------------------------------------------------------------
    if ~goto_figure_generation

    % Build the union of per-condition top-N best-match simIDs (cell of strings)
    sc_simIDs = {};
    for c = 1:length(conditionsFound)
        cond = conditionsFound{c};
        ids  = bestMatchMap.(cond).simIDs;
        take = min(sc.topN, length(ids));
        sc_simIDs = [sc_simIDs; ids(1:take)]; %#ok<AGROW>
    end
    sc_simIDs = unique(sc_simIDs, 'stable');
    sc_nSims  = length(sc_simIDs);
    sc_isingGrid = config.isingGrid;

    fprintf('Using %d unique best-match simulations (topN=%d per condition)\n', sc_nSims, sc.topN);

    % Validate that every needed sim file exists locally
    assertSimsExist(sc_simIDs, config.isingDataPath, 'SamplingConvergence');

    % -------- 4.1a Define spatial regions (Centre / P1 / P2) --------------
    centre_rowStart = floor((sc_isingGrid(1) - sc.expGrid(1)) / 2) + 1;
    centre_colStart = floor((sc_isingGrid(2) - sc.expGrid(2)) / 2) + 1;
    centre_rows = centre_rowStart:(centre_rowStart + sc.expGrid(1) - 1);
    centre_cols = centre_colStart:(centre_colStart + sc.expGrid(2) - 1);

    sc_P1_rows = 1:sc.expGrid(1);
    sc_P1_cols = 1:sc.expGrid(2);
    sc_P2_rows = (sc.expGrid(1) + 1):(2 * sc.expGrid(1));
    sc_P2_cols = 1:sc.expGrid(2);

    fprintf('Centre crop: rows %d:%d, cols %d:%d\n', ...
        centre_rows(1), centre_rows(end), centre_cols(1), centre_cols(end));
    fprintf('P1: rows %d:%d, cols %d:%d\n', sc_P1_rows(1), sc_P1_rows(end), sc_P1_cols(1), sc_P1_cols(end));
    fprintf('P2: rows %d:%d, cols %d:%d\n', sc_P2_rows(1), sc_P2_rows(end), sc_P2_cols(1), sc_P2_cols(end));

    % -------- 4.1b Weight matrix for expGrid ------------------------------
    valueMap_sc    = rand(sc.expGrid(1), sc.expGrid(2));
    distanceMat_sc = squareform(mL_distanceMat(valueMap_sc));
    uniqueDist_sc  = unique(distanceMat_sc);
    uniqueDist_sc(uniqueDist_sc == 0) = [];
    currDistInds_sc = ismember(distanceMat_sc, uniqueDist_sc(1));
    sc_weightMat = zeros(size(distanceMat_sc));
    sc_weightMat(currDistInds_sc) = distanceMat_sc(currDistInds_sc);
    sc_weightMat(sc_weightMat == inf) = 0;

    % -------- 4.1c Main convergence loop ----------------------------------
    fprintf('\n--- 4.1 Main convergence loop ---\n');
    nFractions = length(sc.inverseFractions);

    Results = struct();
    Results.simIDs = sc_simIDs;
    Results.inverseFractions = sc.inverseFractions;

    WD_Centre_all = cell(nFractions, 1);
    WD_P1_all     = cell(nFractions, 1);
    WD_P1P2_all   = cell(nFractions, 1);
    for f = 1:nFractions
        WD_Centre_all{f} = [];
        WD_P1_all{f}     = [];
        WD_P1P2_all{f}   = [];
    end

    GT_clean = [];   % last simulation's ground truth (used in summary)

    for s = 1:sc_nSims
        simID = sc_simIDs{s};
        simPath = fullfile(config.isingDataPath, simID2Filename(simID));

        simData = load(simPath);
        stored_spins = simData.stored_spins;
        T = size(stored_spins, 1);

        moransI_Centre = zeros(1, T);
        moransI_P1     = zeros(1, T);
        moransI_P2     = zeros(1, T);
        for t = 1:T
            frame = squeeze(stored_spins(t, :, :));

            frame_centre = frame(centre_rows, centre_cols);
            if all(frame_centre(:) == 0) || all(frame_centre(:) == 1)
                moransI_Centre(t) = NaN;
            else
                moransI_Centre(t) = mL_moransI(double(frame_centre), sc_weightMat);
            end

            frame_P1 = frame(sc_P1_rows, sc_P1_cols);
            if all(frame_P1(:) == 0) || all(frame_P1(:) == 1)
                moransI_P1(t) = NaN;
            else
                moransI_P1(t) = mL_moransI(double(frame_P1), sc_weightMat);
            end

            frame_P2 = frame(sc_P2_rows, sc_P2_cols);
            if all(frame_P2(:) == 0) || all(frame_P2(:) == 1)
                moransI_P2(t) = NaN;
            else
                moransI_P2(t) = mL_moransI(double(frame_P2), sc_weightMat);
            end
        end

        GT_clean = moransI_Centre(~isnan(moransI_Centre));

        for f = 1:nFractions
            invFrac = sc.inverseFractions(f);
            segmentLength = floor(T / invFrac);
            nSegments = floor(T / segmentLength);

            for seg = 1:nSegments
                startIdx = (seg - 1) * segmentLength + 1;
                endIdx = seg * segmentLength;

                seg_Centre = moransI_Centre(startIdx:endIdx);
                seg_P1     = moransI_P1(startIdx:endIdx);
                seg_P2     = moransI_P2(startIdx:endIdx);

                seg_Centre_clean = seg_Centre(~isnan(seg_Centre));
                seg_P1_clean     = seg_P1(~isnan(seg_P1));
                seg_P2_clean     = seg_P2(~isnan(seg_P2));
                seg_P1P2_clean   = [seg_P1_clean, seg_P2_clean];

                WD_Centre = wasserstein_1d(GT_clean, seg_Centre_clean);
                WD_P1     = wasserstein_1d(GT_clean, seg_P1_clean);
                WD_P1P2   = wasserstein_1d(GT_clean, seg_P1P2_clean);

                WD_Centre_all{f}(end+1) = WD_Centre;
                WD_P1_all{f}(end+1)     = WD_P1;
                WD_P1P2_all{f}(end+1)   = WD_P1P2;
            end
        end

        clear stored_spins simData;

        if mod(s, max(1, floor(sc_nSims/10))) == 0
            fprintf('  Processed %d/%d simulations\n', s, sc_nSims);
        end
    end

    % -------- 4.1d Aggregate ----------------------------------------------
    fprintf('\n--- 4.1d Aggregate ---\n');
    Aggregate = struct();
    Aggregate.inverseFractions = sc.inverseFractions;
    Aggregate.WD_Centre_mean = zeros(nFractions, 1);
    Aggregate.WD_Centre_std  = zeros(nFractions, 1);
    Aggregate.WD_P1_mean     = zeros(nFractions, 1);
    Aggregate.WD_P1_std      = zeros(nFractions, 1);
    Aggregate.WD_P1P2_mean   = zeros(nFractions, 1);
    Aggregate.WD_P1P2_std    = zeros(nFractions, 1);

    for f = 1:nFractions
        Aggregate.WD_Centre_mean(f) = mean(WD_Centre_all{f}, 'omitnan');
        Aggregate.WD_Centre_std(f)  = std(WD_Centre_all{f},  'omitnan');
        Aggregate.WD_P1_mean(f)     = mean(WD_P1_all{f},     'omitnan');
        Aggregate.WD_P1_std(f)      = std(WD_P1_all{f},      'omitnan');
        Aggregate.WD_P1P2_mean(f)   = mean(WD_P1P2_all{f},   'omitnan');
        Aggregate.WD_P1P2_std(f)    = std(WD_P1P2_all{f},    'omitnan');
        fprintf('Fraction 1/%.1f: Centre=%.4f, P1=%.4f, P1+P2=%.4f\n', ...
            sc.inverseFractions(f), Aggregate.WD_Centre_mean(f), ...
            Aggregate.WD_P1_mean(f), Aggregate.WD_P1P2_mean(f));
    end

    Results.Aggregate     = Aggregate;
    Results.WD_Centre_all = WD_Centre_all;
    Results.WD_P1_all     = WD_P1_all;
    Results.WD_P1P2_all   = WD_P1P2_all;

    end  % end of: if ~goto_figure_generation (main convergence compute)

    % -------------------------------------------------------------------------
    % 4.2 Figure 1 (main convergence) + Figure 2 (spatial regions)
    % -------------------------------------------------------------------------
    nFractions = length(sc.inverseFractions);

    % When loading from cache, convert Python 2D arrays into cell arrays
    if goto_figure_generation
        if isfield(Results, 'WD_Centre_all') && ~iscell(Results.WD_Centre_all)
            WD_Centre_2d = Results.WD_Centre_all;
            WD_P1_2d     = Results.WD_P1_all;
            WD_P1P2_2d   = Results.WD_P1P2_all;
            nFrac = size(WD_Centre_2d, 1);
            WD_Centre_all = cell(1, nFrac);
            WD_P1_all     = cell(1, nFrac);
            WD_P1P2_all   = cell(1, nFrac);
            for f = 1:nFrac
                WD_Centre_all{f} = WD_Centre_2d(f, :)';
                WD_P1_all{f}     = WD_P1_2d(f, :)';
                WD_P1P2_all{f}   = WD_P1P2_2d(f, :)';
            end
        elseif isfield(Results, 'WD_Centre_all')
            WD_Centre_all = Results.WD_Centre_all;
            WD_P1_all     = Results.WD_P1_all;
            WD_P1P2_all   = Results.WD_P1P2_all;
        end

        % Recompute region definitions so Figure 2 can draw them
        centre_rowStart = floor((sc_isingGrid(1) - sc.expGrid(1)) / 2) + 1;
        centre_colStart = floor((sc_isingGrid(2) - sc.expGrid(2)) / 2) + 1;
        centre_rows = centre_rowStart:(centre_rowStart + sc.expGrid(1) - 1);
        centre_cols = centre_colStart:(centre_colStart + sc.expGrid(2) - 1);
        sc_P1_rows = 1:sc.expGrid(1);
        sc_P1_cols = 1:sc.expGrid(2);
        sc_P2_rows = (sc.expGrid(1) + 1):(2 * sc.expGrid(1));
        sc_P2_cols = 1:sc.expGrid(2);

        GT_clean = [];   % not available from cached results
    end

    fprintf('\n--- 4.2 Figure 1: Main convergence plot ---\n');

    figure('Name', 'Sampling Convergence Analysis');

    colorCentre = [0.2, 0.2, 0.2];
    colorP1     = [0.2, 0.4, 0.8];
    colorP1P2   = [0.8, 0.2, 0.2];

    subplot(2, 2, 1); hold on;
    jitterAmount = 0.15;
    for f = 1:nFractions
        x_base = sc.inverseFractions(f);

        x_jitter = x_base + randn(size(WD_Centre_all{f})) * jitterAmount;
        scatter(x_jitter, WD_Centre_all{f}, 8, colorCentre, 'filled', 'MarkerFaceAlpha', 0.2);

        x_jitter = x_base + randn(size(WD_P1_all{f})) * jitterAmount;
        scatter(x_jitter, WD_P1_all{f}, 8, colorP1, 'filled', 'MarkerFaceAlpha', 0.2);

        x_jitter = x_base + randn(size(WD_P1P2_all{f})) * jitterAmount;
        scatter(x_jitter, WD_P1P2_all{f}, 8, colorP1P2, 'filled', 'MarkerFaceAlpha', 0.2);
    end

    plot(sc.inverseFractions, Aggregate.WD_Centre_mean, 'o-', 'Color', colorCentre, ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colorCentre);
    plot(sc.inverseFractions, Aggregate.WD_P1_mean,     's-', 'Color', colorP1, ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colorP1);
    plot(sc.inverseFractions, Aggregate.WD_P1P2_mean,   'd-', 'Color', colorP1P2, ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colorP1P2);
    hold off;
    xlabel('Inverse Data Fraction (1/f)'); ylabel('WD(G, subset)');
    title('Sampling Convergence: WD vs Sample Size');
    legend({'','','','Centre','P1','P1+P2 pooled'}, 'Location', 'best');
    set(gca, 'XScale', 'log'); grid on;

    subplot(2, 2, 2); hold on;
    plot(sc.inverseFractions, Aggregate.WD_Centre_mean, 'o-', 'Color', colorCentre, ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colorCentre);
    plot(sc.inverseFractions, Aggregate.WD_P1_mean,     's-', 'Color', colorP1, ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colorP1);
    plot(sc.inverseFractions, Aggregate.WD_P1P2_mean,   'd-', 'Color', colorP1P2, ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colorP1P2);
    hold off;
    xlabel('Inverse Data Fraction (1/f)'); ylabel('WD(G, subset)');
    title('Log-Log Scale');
    legend({'Centre','P1','P1+P2 pooled'}, 'Location', 'best');
    set(gca, 'XScale', 'log', 'YScale', 'log'); grid on;

    subplot(2, 2, 3);
    diff_P1_Centre = Aggregate.WD_P1_mean - Aggregate.WD_Centre_mean;
    bar(1:nFractions, diff_P1_Centre, 'FaceColor', colorP1); hold on;
    yline(0, 'k--', 'LineWidth', 1.5); hold off;
    xticks(1:nFractions);
    xticklabels(arrayfun(@(x) sprintf('1/%.0f', x), sc.inverseFractions, 'UniformOutput', false));
    xlabel('Data Fraction'); ylabel('WD(P1) - WD(Centre)');
    title('Spatial Homogeneity Test');
    if mean(diff_P1_Centre) > 0.005
        text(0.5, 0.9, 'P1 > Centre: Possible spatial bias', 'Units','normalized', ...
            'HorizontalAlignment','center','Color','r','FontWeight','bold');
    else
        text(0.5, 0.9, 'Spatially homogeneous', 'Units','normalized', ...
            'HorizontalAlignment','center','Color',[0.2,0.7,0.3],'FontWeight','bold');
    end
    grid on;

    subplot(2, 2, 4);
    diff_P1P2_P1 = Aggregate.WD_P1P2_mean - Aggregate.WD_P1_mean;
    barColors = zeros(nFractions, 3);
    for f = 1:nFractions
        if diff_P1P2_P1(f) < 0
            barColors(f, :) = [0.2, 0.7, 0.3];
        else
            barColors(f, :) = [0.8, 0.3, 0.3];
        end
    end
    for f = 1:nFractions
        bar(f, diff_P1P2_P1(f), 'FaceColor', barColors(f, :)); hold on;
    end
    yline(0, 'k--', 'LineWidth', 1.5); hold off;
    xticks(1:nFractions);
    xticklabels(arrayfun(@(x) sprintf('1/%.0f', x), sc.inverseFractions, 'UniformOutput', false));
    xlabel('Data Fraction'); ylabel('WD(P1+P2) - WD(P1)');
    title('Pooling Benefit Test'); grid on;

    sgtitle(sprintf('Sampling Convergence Analysis (best-match, n=%d sims)', sc_nSims), 'FontWeight', 'bold');
    saveMyFig('SamplingConvergence_MainFigure', sc.outputPath, gcf);
    fprintf('Saved figure: SamplingConvergence_MainFigure\n');

    fprintf('\n--- 4.2b Figure 2: Spatial regions diagram ---\n');
    figure('Name', 'Spatial Regions');

    regionMap = zeros(sc_isingGrid);
    regionMap(centre_rows, centre_cols) = 1;
    regionMap(sc_P1_rows, sc_P1_cols) = regionMap(sc_P1_rows, sc_P1_cols) + 2;
    regionMap(sc_P2_rows, sc_P2_cols) = regionMap(sc_P2_rows, sc_P2_cols) + 4;

    imagesc(regionMap);
    colormap([1 1 1; 0.5 0.5 0.5; 0.2 0.4 0.8; 0.4 0.5 0.8; 0.8 0.2 0.2; 0.6 0.3 0.6; 0.5 0.4 0.7]);
    axis equal tight; set(gca, 'YDir', 'reverse');
    xlabel('Column'); ylabel('Row'); title('Spatial Regions on Ising Grid');

    text(mean(centre_cols), mean(centre_rows), 'Centre', ...
        'HorizontalAlignment','center','FontWeight','bold','FontSize',12,'Color','w');
    text(mean(sc_P1_cols), mean(sc_P1_rows), 'P1', ...
        'HorizontalAlignment','center','FontWeight','bold','FontSize',12,'Color','w');
    text(mean(sc_P2_cols), mean(sc_P2_rows), 'P2', ...
        'HorizontalAlignment','center','FontWeight','bold','FontSize',12,'Color','w');

    saveMyFig('SamplingConvergence_SpatialRegions', sc.outputPath, gcf);
    fprintf('Saved figure: SamplingConvergence_SpatialRegions\n');

    % -------- 4.2c Main convergence console summary -----------------------
    fprintf('\n========================================\n');
    fprintf('  SAMPLING CONVERGENCE ANALYSIS\n');
    fprintf('========================================\n');
    fprintf('Analysis mode: best_match (topN=%d per condition)\n', sc.topN);
    fprintf('Simulations analysed: %d\n', sc_nSims);
    fprintf('Ising grid: [%d x %d]\n', sc_isingGrid(1), sc_isingGrid(2));
    fprintf('Experimental grid: [%d x %d]\n\n', sc.expGrid(1), sc.expGrid(2));

    fprintf('GROUND TRUTH (Centre, full T):\n');
    if ~isempty(GT_clean)
        fprintf('  Mean Moran''s I: %.4f +/- %.4f\n\n', mean(GT_clean), std(GT_clean));
    else
        fprintf('  (not available from cached results)\n\n');
    end

    fprintf('SPATIAL HOMOGENEITY (P1 vs Centre):\n');
    for f = 1:nFractions
        diff = Aggregate.WD_P1_mean(f) - Aggregate.WD_Centre_mean(f);
        pct  = 100 * diff / Aggregate.WD_Centre_mean(f);
        fprintf('  1/%.0f: %+.4f (%+.1f%%)\n', sc.inverseFractions(f), diff, pct);
    end
    meanDiff = mean(Aggregate.WD_P1_mean - Aggregate.WD_Centre_mean);
    fprintf('  Mean difference: %+.4f\n', meanDiff);
    if abs(meanDiff) < 0.005
        fprintf('  >> NO significant spatial inhomogeneity detected\n\n');
    else
        fprintf('  >> POSSIBLE spatial inhomogeneity detected\n\n');
    end

    fprintf('POOLING BENEFIT (P1+P2 vs P1):\n');
    for f = 1:nFractions
        diff = Aggregate.WD_P1P2_mean(f) - Aggregate.WD_P1_mean(f);
        pct  = 100 * diff / Aggregate.WD_P1_mean(f);
        fprintf('  1/%.0f: %+.4f (%+.1f%%)\n', sc.inverseFractions(f), diff, pct);
    end
    meanPoolingDiff = mean(Aggregate.WD_P1P2_mean - Aggregate.WD_P1_mean);
    fprintf('  Mean difference: %+.4f\n', meanPoolingDiff);
    if meanPoolingDiff < 0
        improvement = -100 * meanPoolingDiff / mean(Aggregate.WD_P1_mean);
        fprintf('  >> Pooling IMPROVES estimate by %.1f%% on average\n', improvement);
    else
        fprintf('  >> Pooling does NOT improve estimate\n');
    end
    fprintf('========================================\n\n');

    % -------------------------------------------------------------------------
    % 4.3 Part 2 — Square Grid Analysis
    % -------------------------------------------------------------------------
    if ~goto_figure_generation
        fprintf('\n###########################################################\n');
        fprintf('##           PART 2: SQUARE GRID ANALYSIS                ##\n');
        fprintf('###########################################################\n');

        minGridDim = min(sc_isingGrid);
        maxSquareSize = floor(minGridDim / 2);
        squareGridSizes = sc.squareGridSizes(sc.squareGridSizes <= maxSquareSize);
        if isempty(squareGridSizes)
            squareGridSizes = [max(2, floor(maxSquareSize/2)), 2];
            squareGridSizes = unique(squareGridSizes, 'stable');
        end
        nSquareGrids = length(squareGridSizes);

        fprintf('Square grid sizes: ');
        fprintf('%dx%d ', [squareGridSizes; squareGridSizes]);
        fprintf('\n');

        SquareGridResults = struct();
        SquareGridResults.gridSizes = squareGridSizes;
        SquareGridResults.nGrids    = nSquareGrids;
        SquareGridResults.WD_diff_mean = zeros(nSquareGrids, 1);
        SquareGridResults.WD_diff_std  = zeros(nSquareGrids, 1);
        SquareGridResults.WD_diff_all  = cell(nSquareGrids, 1);

        for g = 1:nSquareGrids
            gridSize = squareGridSizes(g);
            fprintf('\nProcessing %dx%d grid...\n', gridSize, gridSize);

            sqCentre_rowStart = floor((sc_isingGrid(1) - gridSize) / 2) + 1;
            sqCentre_colStart = floor((sc_isingGrid(2) - gridSize) / 2) + 1;
            sqCentre_rows = sqCentre_rowStart:(sqCentre_rowStart + gridSize - 1);
            sqCentre_cols = sqCentre_colStart:(sqCentre_colStart + gridSize - 1);

            sqTopLeft_rows = 1:gridSize;
            sqTopLeft_cols = 1:gridSize;

            valueMap_sq    = rand(gridSize, gridSize);
            distanceMat_sq = squareform(mL_distanceMat(valueMap_sq));
            uniqueDist_sq  = unique(distanceMat_sq);
            uniqueDist_sq(uniqueDist_sq == 0) = [];
            currDistInds_sq = ismember(distanceMat_sq, uniqueDist_sq(1));
            weightMat_sq = zeros(size(distanceMat_sq));
            weightMat_sq(currDistInds_sq) = distanceMat_sq(currDistInds_sq);
            weightMat_sq(weightMat_sq == inf) = 0;

            all_WD_diffs = [];

            for s = 1:sc_nSims
                simID = sc_simIDs{s};
                simPath = fullfile(config.isingDataPath, simID2Filename(simID));

                simData = load(simPath);
                stored_spins = simData.stored_spins;
                T = size(stored_spins, 1);

                moransI_Centre_sq  = zeros(1, T);
                moransI_TopLeft_sq = zeros(1, T);

                for t = 1:T
                    frame = squeeze(stored_spins(t, :, :));

                    frame_centre = frame(sqCentre_rows, sqCentre_cols);
                    if all(frame_centre(:) == 0) || all(frame_centre(:) == 1)
                        moransI_Centre_sq(t) = NaN;
                    else
                        moransI_Centre_sq(t) = mL_moransI(double(frame_centre), weightMat_sq);
                    end

                    frame_TopLeft = frame(sqTopLeft_rows, sqTopLeft_cols);
                    if all(frame_TopLeft(:) == 0) || all(frame_TopLeft(:) == 1)
                        moransI_TopLeft_sq(t) = NaN;
                    else
                        moransI_TopLeft_sq(t) = mL_moransI(double(frame_TopLeft), weightMat_sq);
                    end
                end

                GT_sq_clean = moransI_Centre_sq(~isnan(moransI_Centre_sq));

                for f = 1:nFractions
                    invFrac = sc.inverseFractions(f);
                    segmentLength = floor(T / invFrac);
                    nSegments = floor(T / segmentLength);
                    for seg = 1:nSegments
                        startIdx = (seg - 1) * segmentLength + 1;
                        endIdx = seg * segmentLength;

                        seg_Centre  = moransI_Centre_sq(startIdx:endIdx);
                        seg_TopLeft = moransI_TopLeft_sq(startIdx:endIdx);
                        seg_Centre_clean  = seg_Centre(~isnan(seg_Centre));
                        seg_TopLeft_clean = seg_TopLeft(~isnan(seg_TopLeft));

                        WD_Centre_sq  = wasserstein_1d(GT_sq_clean, seg_Centre_clean);
                        WD_TopLeft_sq = wasserstein_1d(GT_sq_clean, seg_TopLeft_clean);

                        all_WD_diffs(end+1) = WD_TopLeft_sq - WD_Centre_sq; %#ok<SAGROW>
                    end
                end

                clear stored_spins simData;
            end

            SquareGridResults.WD_diff_all{g} = all_WD_diffs;
            SquareGridResults.WD_diff_mean(g) = mean(all_WD_diffs, 'omitnan');
            SquareGridResults.WD_diff_std(g)  = std(all_WD_diffs,  'omitnan');

            fprintf('  Mean WD(TopLeft)-WD(Centre): %+.4f +/- %.4f\n', ...
                SquareGridResults.WD_diff_mean(g), SquareGridResults.WD_diff_std(g));
        end
    end  % end of: if ~goto_figure_generation (square grid compute)

    % -------- 4.3b Figure 3: Square Grid visualisation --------------------
    squareGridSizes = SquareGridResults.gridSizes;
    nSquareGrids = length(squareGridSizes);

    if goto_figure_generation && ~iscell(SquareGridResults.WD_diff_all)
        WD_diff_all_2d = SquareGridResults.WD_diff_all;
        nGrids_cached = size(WD_diff_all_2d, 1);
        WD_diff_all_cells = cell(1, nGrids_cached);
        for g = 1:nGrids_cached
            WD_diff_all_cells{g} = WD_diff_all_2d(g, :)';
        end
        SquareGridResults.WD_diff_all = WD_diff_all_cells;
    end

    fprintf('\n--- 4.3b Figure 3: Square grid visualisation ---\n');
    figure('Name', 'Spatial Homogeneity vs Grid Size');

    [gridAreas, sortIdx] = sort(double(squareGridSizes).^2);
    gridLabels = arrayfun(@(x) sprintf('%dx%d', x, x), squareGridSizes(sortIdx), 'UniformOutput', false);
    WD_diff_mean_sorted = SquareGridResults.WD_diff_mean(sortIdx);
    WD_diff_std_sorted  = SquareGridResults.WD_diff_std(sortIdx);
    WD_diff_all_sorted  = SquareGridResults.WD_diff_all(sortIdx);

    subplot(1, 2, 1); hold on;
    for g = 1:nSquareGrids
        y_vals = WD_diff_all_sorted{g};
        x_vals = gridAreas(g) + randn(size(y_vals)) * gridAreas(g) * 0.05;
        scatter(x_vals, y_vals, 10, [0.5, 0.5, 0.5], 'filled', 'MarkerFaceAlpha', 0.2);
    end
    errorbar(gridAreas, WD_diff_mean_sorted, WD_diff_std_sorted, ...
        'o-', 'Color', [0.2, 0.4, 0.8], 'LineWidth', 2, 'MarkerSize', 10, ...
        'MarkerFaceColor', [0.2, 0.4, 0.8]);
    yline(0, 'k--', 'LineWidth', 1.5); hold off;

    xlabel('Grid Area (cells)'); ylabel('WD(TopLeft) - WD(Centre)');
    title('Spatial Homogeneity vs Grid Size');
    set(gca, 'XScale', 'log');
    xticks(gridAreas); xticklabels(gridLabels); grid on;

    if mean(abs(SquareGridResults.WD_diff_mean)) < 0.005
        text(0.5, 0.95, 'No significant spatial inhomogeneity', ...
            'Units','normalized','HorizontalAlignment','center', ...
            'FontWeight','bold','Color',[0.2,0.7,0.3]);
    end

    subplot(1, 2, 2); axis off;
    summaryText = {'=== SQUARE GRID ANALYSIS ===','', ...
        'Grid Size | Mean Diff | Std','----------|-----------|------'};
    for g = 1:nSquareGrids
        summaryText{end+1} = sprintf('%4dx%-4d | %+.4f  | %.4f', ...
            squareGridSizes(g), squareGridSizes(g), ...
            SquareGridResults.WD_diff_mean(g), SquareGridResults.WD_diff_std(g)); %#ok<SAGROW>
    end
    summaryText{end+1} = '';
    summaryText{end+1} = sprintf('Overall mean: %+.4f', mean(SquareGridResults.WD_diff_mean));
    text(0.05, 0.95, summaryText, 'VerticalAlignment','top', ...
        'FontSize', 9, 'FontName','FixedWidth','Units','normalized');
    title('Summary');

    sgtitle('Part 2: Spatial Homogeneity Across Grid Sizes', 'FontWeight','bold');
    saveMyFig('SamplingConvergence_SquareGrids', sc.outputPath, gcf);
    fprintf('Saved figure: SamplingConvergence_SquareGrids\n');

    fprintf('\n========================================\n');
    fprintf('  SQUARE GRID ANALYSIS SUMMARY\n');
    fprintf('========================================\n');
    fprintf('Grid Size | Mean WD(TopLeft)-WD(Centre) | Std\n');
    fprintf('----------|------------------------|--------\n');
    for g = 1:nSquareGrids
        fprintf('%4dx%-4d  |       %+.4f          | %.4f\n', ...
            squareGridSizes(g), squareGridSizes(g), ...
            SquareGridResults.WD_diff_mean(g), SquareGridResults.WD_diff_std(g));
    end
    overall_mean = mean(SquareGridResults.WD_diff_mean);
    fprintf('\nOverall mean difference: %+.4f\n', overall_mean);
    if abs(overall_mean) < 0.005
        fprintf('>> NO systematic spatial inhomogeneity detected\n');
    else
        fprintf('>> POSSIBLE spatial inhomogeneity — investigate further\n');
    end
    fprintf('========================================\n\n');

    % -------------------------------------------------------------------------
    % 4.4 Part 3 — Distance-from-Centre Analysis (2x2 grid)
    % -------------------------------------------------------------------------
    if ~goto_figure_generation
        fprintf('\n###########################################################\n');
        fprintf('##         PART 3: DISTANCE-FROM-CENTRE ANALYSIS         ##\n');
        fprintf('###########################################################\n');

        distGridSize = sc.distGridSize;
        nPosPerDimRow = floor(sc_isingGrid(1) / distGridSize);
        nPosPerDimCol = floor(sc_isingGrid(2) / distGridSize);
        nTotalPos = nPosPerDimRow * nPosPerDimCol;

        fprintf('Analysing %dx%d grid: %d row x %d col positions = %d total\n', ...
            distGridSize, distGridSize, nPosPerDimRow, nPosPerDimCol, nTotalPos);

        valueMap_dist    = rand(distGridSize, distGridSize);
        distanceMat_dist = squareform(mL_distanceMat(valueMap_dist));
        uniqueDist_dist  = unique(distanceMat_dist);
        uniqueDist_dist(uniqueDist_dist == 0) = [];
        currDistInds_dist = ismember(distanceMat_dist, uniqueDist_dist(1));
        weightMat_dist = zeros(size(distanceMat_dist));
        weightMat_dist(currDistInds_dist) = distanceMat_dist(currDistInds_dist);
        weightMat_dist(weightMat_dist == inf) = 0;

        gridCentre = (sc_isingGrid + 1) / 2;

        distCentre_rowStart = floor((sc_isingGrid(1) - distGridSize) / 2) + 1;
        distCentre_colStart = floor((sc_isingGrid(2) - distGridSize) / 2) + 1;
        distCentre_rows = distCentre_rowStart:(distCentre_rowStart + distGridSize - 1);
        distCentre_cols = distCentre_colStart:(distCentre_colStart + distGridSize - 1);

        DistanceAnalysis = struct();
        DistanceAnalysis.gridSize   = distGridSize;
        DistanceAnalysis.nPositions = nTotalPos;
        DistanceAnalysis.nPosRow    = nPosPerDimRow;   % used by visualisation
        DistanceAnalysis.nPosCol    = nPosPerDimCol;
        DistanceAnalysis.nPosRows   = nPosPerDimRow;   % plural alias (matches SpatialHomogeneity naming)
        DistanceAnalysis.nPosCols   = nPosPerDimCol;
        DistanceAnalysis.positionCentres    = zeros(nTotalPos, 2);
        DistanceAnalysis.distanceFromCentre = zeros(nTotalPos, 1);
        DistanceAnalysis.WD_to_GT           = zeros(nTotalPos, 1);

        posIdx = 0;
        for pr = 1:nPosPerDimRow
            for pc = 1:nPosPerDimCol
                posIdx = posIdx + 1;
                rowStart = (pr - 1) * distGridSize + 1;
                colStart = (pc - 1) * distGridSize + 1;
                posCentre_row = rowStart + (distGridSize - 1) / 2;
                posCentre_col = colStart + (distGridSize - 1) / 2;
                DistanceAnalysis.positionCentres(posIdx, :) = [posCentre_row, posCentre_col];
                DistanceAnalysis.distanceFromCentre(posIdx) = sqrt( ...
                    (posCentre_row - gridCentre(1))^2 + (posCentre_col - gridCentre(2))^2);
            end
        end

        fprintf('Computing Moran''s I for all %d positions across %d simulations...\n', nTotalPos, sc_nSims);
        WD_all_positions = zeros(nTotalPos, sc_nSims);

        for s = 1:sc_nSims
            simID = sc_simIDs{s};
            simPath = fullfile(config.isingDataPath, simID2Filename(simID));

            simData = load(simPath);
            stored_spins = simData.stored_spins;
            T = size(stored_spins, 1);

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

            posIdx = 0;
            for pr = 1:nPosPerDimRow
                for pc = 1:nPosPerDimCol
                    posIdx = posIdx + 1;
                    rowStart = (pr - 1) * distGridSize + 1;
                    colStart = (pc - 1) * distGridSize + 1;
                    rows = rowStart:(rowStart + distGridSize - 1);
                    cols = colStart:(colStart + distGridSize - 1);

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
                    WD_all_positions(posIdx, s) = wasserstein_1d(GT_dist_clean, moransI_pos_clean);
                end
            end

            clear stored_spins simData;

            if mod(s, max(1, floor(sc_nSims/5))) == 0
                fprintf('  Processed %d/%d simulations\n', s, sc_nSims);
            end
        end

        DistanceAnalysis.WD_to_GT     = mean(WD_all_positions, 2, 'omitnan');
        DistanceAnalysis.WD_to_GT_std = std(WD_all_positions, 0, 2, 'omitnan');
    end  % end of: if ~goto_figure_generation (distance compute)

    % -------- 4.4b Figure 4: Distance-from-Centre visualisation ----------
    distGridSize = double(DistanceAnalysis.gridSize);
    nTotalPos    = double(DistanceAnalysis.nPositions);
    if isfield(DistanceAnalysis, 'nPosRow')
        nPosPerDimRow = double(DistanceAnalysis.nPosRow);
        nPosPerDimCol = double(DistanceAnalysis.nPosCol);
    else
        nPosPerDimRow = double(DistanceAnalysis.nPosRows);
        nPosPerDimCol = double(DistanceAnalysis.nPosCols);
    end

    fprintf('\n--- 4.4b Figure 4: Distance-from-Centre visualisation ---\n');
    figure('Name', 'Distance from Centre Analysis');

    subplot(1, 2, 1);
    % Force to double column vectors — the cached data may arrive as
    % integers / rows / higher-dim arrays and downstream stats need scalars
    dfc_vec = double(DistanceAnalysis.distanceFromCentre(:));
    wd_vec  = double(DistanceAnalysis.WD_to_GT(:));
    scatter(dfc_vec, wd_vec, 30, [0.2, 0.4, 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    validIdx = ~isnan(wd_vec);
    if sum(validIdx) > 2
        xv = dfc_vec(validIdx);
        yv = wd_vec(validIdx);
        p = polyfit(xv, yv, 1);
        xFit = linspace(min(dfc_vec), max(dfc_vec), 100);
        yFit = polyval(p, xFit);
        plot(xFit, yFit, 'r-', 'LineWidth', 2);
        [r_mat, pval_mat] = corr(xv, yv);
        r    = r_mat(1);
        pval = pval_mat(1);
        slope = p(1);
        text(0.05, 0.95, sprintf('r = %.3f, p = %.3f\nslope = %.4f', r, pval, slope), ...
            'Units','normalized','VerticalAlignment','top', ...
            'FontSize', 10, 'BackgroundColor', 'w');
    end
    hold off;
    xlabel('Distance from Grid Centre');
    ylabel('WD(position, GT_{centre})');
    title(sprintf('%dx%d Grid: WD vs Distance from Centre', distGridSize, distGridSize));
    grid on;

    subplot(1, 2, 2);
    WD_heatmap = reshape(wd_vec, [nPosPerDimRow, nPosPerDimCol]);
    imagesc(WD_heatmap); colorbar; colormap(hot); axis equal tight;
    hold on;
    centrePosRow = ceil(nPosPerDimRow / 2);
    centrePosCol = ceil(nPosPerDimCol / 2);
    plot(centrePosCol, centrePosRow, 'go', 'MarkerSize', 15, 'LineWidth', 3);
    hold off;
    xlabel('Column Position'); ylabel('Row Position');
    title('Spatial Map of WD to Centre'); set(gca, 'YDir', 'reverse');

    sgtitle(sprintf('Distance-from-Centre Analysis (%dx%d grid, %d positions)', ...
        distGridSize, distGridSize, nTotalPos), 'FontWeight','bold');
    saveMyFig('SamplingConvergence_DistanceFromCentre', sc.outputPath, gcf);
    fprintf('Saved figure: SamplingConvergence_DistanceFromCentre\n');

    fprintf('\n========================================\n');
    fprintf('  DISTANCE-FROM-CENTRE ANALYSIS\n');
    fprintf('========================================\n');
    fprintf('Grid size: %dx%d\n', distGridSize, distGridSize);
    fprintf('Total positions: %d\n', nTotalPos);
    if exist('r', 'var') && exist('pval', 'var') && exist('slope', 'var')
        fprintf('Correlation (WD vs Distance): r=%.4f, p=%.4f, slope=%.4f\n', r, pval, slope);
        if pval < 0.05 && slope > 0
            fprintf('>> SIGNIFICANT positive correlation — WD increases with distance\n');
        elseif pval < 0.05 && slope < 0
            fprintf('>> SIGNIFICANT negative correlation — unexpected\n');
        else
            fprintf('>> NO significant correlation — spatially homogeneous\n');
        end
    end
    fprintf('========================================\n\n');

    % -------------------------------------------------------------------------
    % 4.5 Save SamplingConvergence results
    % -------------------------------------------------------------------------
    fprintf('--- 4.5 Saving SamplingConvergence results ---\n');

    % Flatten saved config so downstream readers (including the cache
    % path itself) see a backwards-compatible shape.
    cfg_sc = struct();
    cfg_sc.analysisMode      = 'best_match';
    cfg_sc.isingDataPath     = config.isingDataPath;
    cfg_sc.isingComparisonPath = config.isingComparisonPath;
    cfg_sc.outputPath        = sc.outputPath;
    cfg_sc.isingGrid         = sc_isingGrid;
    cfg_sc.expGrid           = sc.expGrid;
    cfg_sc.inverseFractions  = sc.inverseFractions;
    cfg_sc.squareGridSizes   = sc.squareGridSizes;
    cfg_sc.distGridSize      = sc.distGridSize;
    cfg_sc.topN              = sc.topN;
    cfg_sc.nSims             = sc_nSims;

    Results.config            = cfg_sc;
    Results.simIDs            = sc_simIDs;
    Results.SquareGridResults = SquareGridResults;
    Results.DistanceAnalysis  = DistanceAnalysis;
    Results.timestamp         = datetime('now');

    save(precomputedPath, 'Results', '-v7.3');
    fprintf('Results saved to: %s\n\n', precomputedPath);

    % Clear section-local temporaries
    clear sc precomputedPath goto_figure_generation sc_isingGrid sc_simIDs sc_nSims ...
          centre_rows centre_cols centre_rowStart centre_colStart ...
          sc_P1_rows sc_P1_cols sc_P2_rows sc_P2_cols valueMap_sc distanceMat_sc ...
          uniqueDist_sc currDistInds_sc sc_weightMat nFractions WD_Centre_all ...
          WD_P1_all WD_P1P2_all WD_Centre_2d WD_P1_2d WD_P1P2_2d nFrac f s t ...
          simID simPath simData stored_spins T moransI_Centre moransI_P1 moransI_P2 ...
          frame frame_centre frame_P1 frame_P2 GT_clean invFrac segmentLength nSegments ...
          seg startIdx endIdx seg_Centre seg_P1 seg_P2 seg_Centre_clean seg_P1_clean ...
          seg_P2_clean seg_P1P2_clean WD_Centre WD_P1 WD_P1P2 Aggregate Results ...
          colorCentre colorP1 colorP1P2 jitterAmount x_base x_jitter ...
          diff_P1_Centre diff_P1P2_P1 barColors regionMap ...
          minGridDim maxSquareSize squareGridSizes nSquareGrids SquareGridResults ...
          g gridSize sqCentre_rowStart sqCentre_colStart sqCentre_rows sqCentre_cols ...
          sqTopLeft_rows sqTopLeft_cols valueMap_sq distanceMat_sq uniqueDist_sq ...
          currDistInds_sq weightMat_sq all_WD_diffs moransI_Centre_sq moransI_TopLeft_sq ...
          GT_sq_clean frame_TopLeft WD_Centre_sq WD_TopLeft_sq ...
          gridAreas sortIdx gridLabels WD_diff_mean_sorted WD_diff_std_sorted ...
          WD_diff_all_sorted WD_diff_all_2d nGrids_cached WD_diff_all_cells ...
          distGridSize nPosPerDimRow nPosPerDimCol nTotalPos valueMap_dist ...
          distanceMat_dist uniqueDist_dist currDistInds_dist weightMat_dist ...
          gridCentre distCentre_rowStart distCentre_colStart distCentre_rows distCentre_cols ...
          DistanceAnalysis posIdx pr pc rowStart colStart posCentre_row posCentre_col ...
          rows cols moransI_pos moransI_pos_clean GT_dist GT_dist_clean ...
          WD_all_positions WD_heatmap nFrames_cached ...
          centrePosRow centrePosCol validIdx r pval p slope r_mat pval_mat xv yv ...
          dfc_vec wd_vec xFit yFit ...
          cfg_sc meanDiff meanPoolingDiff improvement diff pct overall_mean summaryText ...
          y_vals x_vals;

else
    fprintf('\n[SamplingConvergence section skipped per config.runSamplingConvergence]\n\n');
end

%% =========================================================================
%% SECTION 5: SPATIAL HOMOGENEITY ANALYSIS
%% =========================================================================

if config.runSpatialHomogeneity
    fprintf('\n###########################################################\n');
    fprintf('##           SECTION 5: SPATIAL HOMOGENEITY              ##\n');
    fprintf('###########################################################\n\n');

    sh = config.spatialHomogeneity;
    if ~exist(sh.outputPath, 'dir'); mkdir(sh.outputPath); end

    % -------------------------------------------------------------------------
    % 5.1 Resolve which simulation to analyse (best match from comparison file)
    % -------------------------------------------------------------------------
    sourceCondition = sh.sourceCondition;
    if ~isfield(bestMatchMap, sourceCondition)
        fallback = conditionsFound{1};
        warning('SpatialHomogeneity: requested condition "%s" not in comparison file — falling back to "%s"', ...
            sourceCondition, fallback);
        sourceCondition = fallback;
    end

    simIDs_cond = bestMatchMap.(sourceCondition).simIDs;
    rank = max(1, min(sh.sourceRank, length(simIDs_cond)));
    if rank ~= sh.sourceRank
        warning('SpatialHomogeneity: requested rank %d exceeds available matches (%d) — using rank %d', ...
            sh.sourceRank, length(simIDs_cond), rank);
    end
    resolvedSimID = simIDs_cond{rank};

    simPath = requireSimPath(resolvedSimID, config.isingDataPath);
    fprintf('Source: condition=%s, rank=%d, simID=%s\n', sourceCondition, rank, resolvedSimID);
    fprintf('Simulation file: %s\n', simPath);

    % -------------------------------------------------------------------------
    % 5.2 Load simulation and set up spatial grid
    % -------------------------------------------------------------------------
    simData = load(simPath);
    stored_spins = simData.stored_spins;
    [T, nRows, nCols] = size(stored_spins);
    sh_isingGrid = [nRows, nCols];
    fprintf('Loaded stored_spins: [%d x %d x %d]\n', T, nRows, nCols);

    distGridSize = sh.analysisGridSize;
    nPosRows  = floor(nRows / distGridSize);
    nPosCols  = floor(nCols / distGridSize);
    nTotalPos = nPosRows * nPosCols;
    fprintf('Analysis grid: %dx%d windows, %d x %d = %d positions\n', ...
        distGridSize, distGridSize, nPosRows, nPosCols, nTotalPos);

    % -------------------------------------------------------------------------
    % 5.3 Weight matrix + position layout
    % -------------------------------------------------------------------------
    valueMap_sh    = rand(distGridSize, distGridSize);
    distanceMat_sh = squareform(mL_distanceMat(valueMap_sh));
    uniqueDist_sh  = unique(distanceMat_sh);
    uniqueDist_sh(uniqueDist_sh == 0) = [];
    currDistInds_sh = ismember(distanceMat_sh, uniqueDist_sh(1));
    weightMat_sh = zeros(size(distanceMat_sh));
    weightMat_sh(currDistInds_sh) = distanceMat_sh(currDistInds_sh);
    weightMat_sh(weightMat_sh == inf) = 0;

    gridCentre = ([nRows, nCols] + 1) / 2;
    fprintf('Grid centre: [%.1f, %.1f]\n', gridCentre(1), gridCentre(2));

    distCentre_rowStart = floor((nRows - distGridSize) / 2) + 1;
    distCentre_colStart = floor((nCols - distGridSize) / 2) + 1;
    distCentre_rows = distCentre_rowStart:(distCentre_rowStart + distGridSize - 1);
    distCentre_cols = distCentre_colStart:(distCentre_colStart + distGridSize - 1);

    fprintf('Centre position: rows %d:%d, cols %d:%d\n', ...
        distCentre_rows(1), distCentre_rows(end), distCentre_cols(1), distCentre_cols(end));

    DistanceAnalysis = struct();
    DistanceAnalysis.gridSize   = distGridSize;
    DistanceAnalysis.nPositions = nTotalPos;
    DistanceAnalysis.nPosRows   = nPosRows;
    DistanceAnalysis.nPosCols   = nPosCols;
    DistanceAnalysis.positionCentres    = zeros(nTotalPos, 2);
    DistanceAnalysis.distanceFromCentre = zeros(nTotalPos, 1);
    DistanceAnalysis.WD_to_GT           = zeros(nTotalPos, 1);

    posIdx = 0;
    for pr = 1:nPosRows
        for pc = 1:nPosCols
            posIdx = posIdx + 1;
            rowStart = (pr - 1) * distGridSize + 1;
            colStart = (pc - 1) * distGridSize + 1;
            posCentre_row = rowStart + (distGridSize - 1) / 2;
            posCentre_col = colStart + (distGridSize - 1) / 2;
            DistanceAnalysis.positionCentres(posIdx, :) = [posCentre_row, posCentre_col];
            DistanceAnalysis.distanceFromCentre(posIdx) = sqrt( ...
                (posCentre_row - gridCentre(1))^2 + (posCentre_col - gridCentre(2))^2);
        end
    end

    fprintf('Distance range: [%.2f, %.2f]\n', ...
        min(DistanceAnalysis.distanceFromCentre), max(DistanceAnalysis.distanceFromCentre));

    % -------------------------------------------------------------------------
    % 5.4 Compute ground-truth and per-position Moran's I / WD
    % -------------------------------------------------------------------------
    fprintf('\n--- 5.4 Computing Moran''s I and WD ---\n');

    GT_moransI = zeros(1, T);
    for t = 1:T
        frame = squeeze(stored_spins(t, :, :));
        frame_centre = frame(distCentre_rows, distCentre_cols);
        if all(frame_centre(:) == 0) || all(frame_centre(:) == 1)
            GT_moransI(t) = NaN;
        else
            GT_moransI(t) = mL_moransI(double(frame_centre), weightMat_sh);
        end
    end
    GT_clean = GT_moransI(~isnan(GT_moransI));
    fprintf('Ground Truth: %d valid samples (%.1f%% NaN)\n', ...
        length(GT_clean), 100 * sum(isnan(GT_moransI)) / T);

    fprintf('Processing all positions...\n');
    posIdx = 0;
    for pr = 1:nPosRows
        for pc = 1:nPosCols
            posIdx = posIdx + 1;
            rowStart = (pr - 1) * distGridSize + 1;
            colStart = (pc - 1) * distGridSize + 1;
            rows = rowStart:(rowStart + distGridSize - 1);
            cols = colStart:(colStart + distGridSize - 1);

            moransI_pos = zeros(1, T);
            for t = 1:T
                frame = squeeze(stored_spins(t, :, :));
                frame_pos = frame(rows, cols);
                if all(frame_pos(:) == 0) || all(frame_pos(:) == 1)
                    moransI_pos(t) = NaN;
                else
                    moransI_pos(t) = mL_moransI(double(frame_pos), weightMat_sh);
                end
            end
            moransI_pos_clean = moransI_pos(~isnan(moransI_pos));

            DistanceAnalysis.WD_to_GT(posIdx) = wasserstein_1d(GT_clean, moransI_pos_clean);
        end

        if mod(pr, max(1, floor(nPosRows/5))) == 0
            fprintf('  Processed row %d/%d (%.0f%%)\n', pr, nPosRows, 100*pr/nPosRows);
        end
    end

    fprintf('Analysis complete.\n');

    % -------------------------------------------------------------------------
    % 5.5 Visualisation
    % -------------------------------------------------------------------------
    fprintf('\n--- 5.5 Visualisation ---\n');
    figure('Name', 'Spatial Homogeneity Analysis');

    subplot(1, 2, 1);
    % Force to double column vectors — downstream stats need scalars
    dfc_vec = double(DistanceAnalysis.distanceFromCentre(:));
    wd_vec  = double(DistanceAnalysis.WD_to_GT(:));
    scatter(dfc_vec, wd_vec, 30, [0.2, 0.4, 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    validIdx = ~isnan(wd_vec);
    if sum(validIdx) > 2
        xv = dfc_vec(validIdx);
        yv = wd_vec(validIdx);
        p = polyfit(xv, yv, 1);
        xFit = linspace(min(dfc_vec), max(dfc_vec), 100);
        yFit = polyval(p, xFit);
        plot(xFit, yFit, 'r-', 'LineWidth', 2);
        [r_mat, pval_mat] = corr(xv, yv);
        r    = r_mat(1);
        pval = pval_mat(1);
        slope = p(1);
        text(0.05, 0.95, sprintf('r = %.3f, p = %.3f\nslope = %.4f', r, pval, slope), ...
            'Units','normalized','VerticalAlignment','top', ...
            'FontSize', 10, 'BackgroundColor', 'w');
    end
    hold off;
    xlabel('Distance from Grid Centre');
    ylabel('WD(position, GT_{centre})');
    title(sprintf('%dx%d Grid: WD vs Distance from Centre', distGridSize, distGridSize));
    grid on;

    subplot(1, 2, 2);
    WD_heatmap = reshape(wd_vec, [nPosRows, nPosCols]);
    imagesc(WD_heatmap); colorbar; colormap(hot); axis equal tight;
    hold on;
    centrePosRow = ceil(nPosRows / 2);
    centrePosCol = ceil(nPosCols / 2);
    plot(centrePosCol, centrePosRow, 'go', 'MarkerSize', 15, 'LineWidth', 3);
    hold off;
    xlabel('Column Position'); ylabel('Row Position');
    title('Spatial Map of WD to Centre');
    set(gca, 'YDir', 'reverse');

    sgtitle(sprintf('Spatial Homogeneity Analysis (%s, %s, rank=%d)', ...
        resolvedSimID, sourceCondition, rank), 'FontWeight','bold');
    saveMyFig('SpatialHomogeneity_DistanceFromCentre', sh.outputPath, gcf);
    fprintf('Saved figure: SpatialHomogeneity_DistanceFromCentre\n');

    % -------------------------------------------------------------------------
    % 5.6 Console summary + save
    % -------------------------------------------------------------------------
    fprintf('\n========================================\n');
    fprintf('  SPATIAL HOMOGENEITY ANALYSIS\n');
    fprintf('========================================\n');
    fprintf('Source sim: %s (%s, rank %d)\n', resolvedSimID, sourceCondition, rank);
    fprintf('Simulation grid: %d x %d\n', nRows, nCols);
    fprintf('Analysis grid: %dx%d\n', distGridSize, distGridSize);
    fprintf('Total positions: %d (%d x %d)\n', nTotalPos, nPosRows, nPosCols);
    if exist('r', 'var') && exist('pval', 'var') && exist('slope', 'var')
        fprintf('Correlation (WD vs Distance): r=%.4f, p=%.4f, slope=%.4f\n', r, pval, slope);
        if pval < 0.05 && slope > 0
            fprintf('>> SIGNIFICANT positive correlation — WD increases with distance\n');
        elseif pval < 0.05 && slope < 0
            fprintf('>> SIGNIFICANT negative correlation — unexpected\n');
        else
            fprintf('>> NO significant correlation — spatially homogeneous\n');
        end
    end
    fprintf('========================================\n\n');

    % Flatten config for backwards-compatible save
    cfg_sh = struct();
    cfg_sh.dataPath          = simPath;
    cfg_sh.outputPath        = sh.outputPath;
    cfg_sh.analysisGridSize  = distGridSize;
    cfg_sh.isingGrid         = sh_isingGrid;
    cfg_sh.sourceCondition   = sourceCondition;
    cfg_sh.sourceRank        = rank;
    cfg_sh.sourceSimID       = resolvedSimID;

    resultsFile = fullfile(sh.outputPath, 'SpatialHomogeneity_Results.mat');
    % Use a struct-wrap so we can save a flat `config` top-level variable
    % without clobbering the outer unified config struct.
    saveBundle = struct('DistanceAnalysis', DistanceAnalysis, 'config', cfg_sh); %#ok<NASGU>
    save(resultsFile, '-struct', 'saveBundle', '-v7.3');
    fprintf('Results saved to: %s\n', resultsFile);

    % Clear section-local temporaries
    clear sh sourceCondition simIDs_cond rank resolvedSimID simPath simData ...
          stored_spins T nRows nCols sh_isingGrid distGridSize nPosRows nPosCols nTotalPos ...
          valueMap_sh distanceMat_sh uniqueDist_sh currDistInds_sh weightMat_sh ...
          gridCentre distCentre_rowStart distCentre_colStart distCentre_rows distCentre_cols ...
          DistanceAnalysis posIdx pr pc rowStart colStart posCentre_row posCentre_col ...
          GT_moransI GT_clean t frame frame_centre frame_pos rows cols moransI_pos ...
          moransI_pos_clean validIdx p xFit yFit r pval slope r_mat pval_mat xv yv ...
          dfc_vec wd_vec WD_heatmap centrePosRow centrePosCol ...
          cfg_sh resultsFile saveBundle fallback;

else
    fprintf('\n[SpatialHomogeneity section skipped per config.runSpatialHomogeneity]\n\n');
end

%% =========================================================================
%% SECTION 6: Final combined summary
%% =========================================================================

fprintf('\n########################################\n');
fprintf('  Figure5_CombinedIsingAnalysis — DONE\n');
fprintf('########################################\n');
fprintf('Output base: %s\n', config.outputPathBase);
if config.runReliability
    fprintf('  [x] Reliability           -> %s\n', config.reliability.outputPath);
end
if config.runSamplingConvergence
    fprintf('  [x] SamplingConvergence   -> %s\n', config.samplingConvergence.outputPath);
end
if config.runSpatialHomogeneity
    fprintf('  [x] SpatialHomogeneity    -> %s\n', config.spatialHomogeneity.outputPath);
end
fprintf('\n');

%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================

function bestMatchMap = loadComparisonBestMatches(hdf5Path, conditions)
    % Read the IsingComparison_Results HDF5 file produced by
    % Figure5_IsingComparison_optimized.py and build a bestMatchMap
    % keyed by condition, containing per-sim ID strings plus params.
    %
    % The file is NOT a MATLAB .mat — it is plain Python HDF5 with:
    %   /Comparison/<cond>/bestMatch_idx    — int64 [topN], 0-based
    %   /Comparison/<cond>/bestMatch_simIDs — scalar UTF-8 string (Python repr)
    %   /IsingData/simIDs                   — variable-length UTF-8 strings [nSims]
    %   /IsingData/params/{beta,c,decay_const,inhibition_range,bias} — [nSims]
    %
    % We use bestMatch_idx (clean integer array) rather than the string.

    % Load global simIDs and parameter vectors ONCE
    all_simIDs_raw = h5read(hdf5Path, '/IsingData/simIDs');
    if isstring(all_simIDs_raw)
        all_simIDs = cellstr(all_simIDs_raw);
    elseif iscell(all_simIDs_raw)
        all_simIDs = all_simIDs_raw;
    else
        all_simIDs = cellstr(all_simIDs_raw);
    end

    params_all.beta             = double(h5read(hdf5Path, '/IsingData/params/beta'));
    params_all.c                = double(h5read(hdf5Path, '/IsingData/params/c'));
    params_all.decay_const      = double(h5read(hdf5Path, '/IsingData/params/decay_const'));
    params_all.inhibition_range = double(h5read(hdf5Path, '/IsingData/params/inhibition_range'));
    params_all.bias             = double(h5read(hdf5Path, '/IsingData/params/bias'));

    bestMatchMap = struct();
    for c = 1:length(conditions)
        cond = conditions{c};
        idxPath = sprintf('/Comparison/%s/bestMatch_idx', cond);
        try
            idx0 = h5read(hdf5Path, idxPath);   % 0-based Python indices
        catch
            continue;  % Condition not present in the file
        end
        idx1 = double(idx0(:)) + 1;  % to 1-based MATLAB

        bestMatchMap.(cond).simIDs                   = all_simIDs(idx1);
        bestMatchMap.(cond).params.beta              = params_all.beta(idx1);
        bestMatchMap.(cond).params.c                 = params_all.c(idx1);
        bestMatchMap.(cond).params.decay_const       = params_all.decay_const(idx1);
        bestMatchMap.(cond).params.inhibition_range  = params_all.inhibition_range(idx1);
        bestMatchMap.(cond).params.bias              = params_all.bias(idx1);
    end
end

function fname = simID2Filename(simID)
    % Convert an Ising simID string to its on-disk filename.
    %
    % simID example:  'be0.45_c1_d10_r13_bi-1.0'
    % filename:       'sim_be_0.45_c_1_d_10_r_13_bi_-1.mat'
    %
    % Transformation:
    %   (1) insert '_' between each letter prefix and its numeric value
    %   (2) strip trailing '.0' from values (they're stored as integers on disk)
    s = char(simID);
    s = regexprep(s, '([a-z]+)(?=-?\d)', '$1_');
    s = regexprep(s, '(\d)\.0(?=_|$)', '$1');
    fname = sprintf('sim_%s.mat', s);
end

function simPath = requireSimPath(simID, isingDataPath)
    % Resolve a simID to its on-disk path. Error with a clear
    % "copy this file" instruction if the file does not exist.
    fname = simID2Filename(simID);
    simPath = fullfile(isingDataPath, fname);
    if ~exist(simPath, 'file')
        printMissingSimMessage({simID}, isingDataPath);
        error('Missing simulation file for simID ''%s''. See console output above.', simID);
    end
end

function assertSimsExist(simIDsCell, isingDataPath, label)
    % Check that every simID in the cell array has a corresponding
    % on-disk file. If any are missing, print the complete list and error.
    missing = {};
    for k = 1:length(simIDsCell)
        fname = simID2Filename(simIDsCell{k});
        p = fullfile(isingDataPath, fname);
        if ~exist(p, 'file')
            missing{end+1} = simIDsCell{k}; %#ok<AGROW>
        end
    end
    if ~isempty(missing)
        printMissingSimMessage(missing, isingDataPath);
        error('%s: %d required simulation file(s) not found locally. See console output above.', ...
            label, length(missing));
    end
end

function printMissingSimMessage(missingSimIDs, isingDataPath)
    % Print a formatted message listing simulation files that need to be
    % copied from the cluster to the local isingDataPath folder.
    fprintf(2, '\n');
    fprintf(2, '#####################################################################\n');
    fprintf(2, '#  MISSING SIMULATION FILE(S)                                       #\n');
    fprintf(2, '#####################################################################\n');
    fprintf(2, 'The merged script needs the following simulation file(s) which are\n');
    fprintf(2, 'NOT present in your local data folder:\n\n');
    fprintf(2, '  Local folder: %s\n\n', isingDataPath);
    fprintf(2, 'Please copy these files from the cluster:\n');
    for k = 1:length(missingSimIDs)
        fname = simID2Filename(missingSimIDs{k});
        fprintf(2, '  [%2d] %s  (simID: %s)\n', k, fname, missingSimIDs{k});
    end
    fprintf(2, '\n');
    fprintf(2, 'The cluster source folder is typically:\n');
    fprintf(2, '  /path/to/data/IsingSims\n');
    fprintf(2, '\n');
    fprintf(2, 'Example scp command for a single file:\n');
    fprintf(2, '  scp user@cluster:/path/to/data/IsingSims/%s \\\n', ...
        simID2Filename(missingSimIDs{1}));
    fprintf(2, '      "%s"\n', isingDataPath);
    fprintf(2, '#####################################################################\n');
    fprintf(2, '\n');
end

function tau = computeACFDecayTime(timeseries, maxLag, threshold)
    % Compute the lag at which autocorrelation drops below threshold.
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
    acf  = acf(maxLag+1:end);
    lags = lags(maxLag+1:end);
    belowThreshold = find(acf < threshold, 1, 'first');
    if isempty(belowThreshold)
        tau = Inf;
    else
        tau = lags(belowThreshold);
    end
end

function positions = generateNonOverlappingPositions(simGrid, expGrid)
    % Generate non-overlapping grid positions starting from upper-left.
    nRowPositions = floor(simGrid(1) / expGrid(1));
    nColPositions = floor(simGrid(2) / expGrid(2));

    rowStarts = 1:expGrid(1):(nRowPositions * expGrid(1));
    colStarts = 1:expGrid(2):(nColPositions * expGrid(2));

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

    if ~isempty(rowStarts) && ~isempty(colStarts)
        [C, R] = meshgrid(colStarts, rowStarts);
        positions = [R(:), C(:)];
    else
        positions = [];
    end
end

function d = wasserstein_1d(x, y)
    % 1D Wasserstein (Earth Mover's) distance via quantile matching.
    x = x(:); y = y(:);
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

function saveMyFig(name, outputPath, figHandle)
    % Save a figure as PNG + SVG with a date-stamped, name-based filename.
    if nargin < 3
        figHandle = gcf;
    end
    dateStr = datestr(now, 'yyyymmdd');
    baseName = sprintf('%s_%s_%s', dateStr, name, get(figHandle, 'Name'));
    baseName = strrep(baseName, ' ', '_');

    if ~exist(outputPath, 'dir')
        mkdir(outputPath);
    end

    pngPath = fullfile(outputPath, [baseName '.png']);
    saveas(figHandle, pngPath);
    fprintf('Figure (Handle: %d) saved as: %s\n', figHandle.Number, pngPath);

    svgPath = fullfile(outputPath, [baseName '.svg']);
    saveas(figHandle, svgPath);
    fprintf('Figure (Handle: %d) saved as: %s\n', figHandle.Number, svgPath);
end
