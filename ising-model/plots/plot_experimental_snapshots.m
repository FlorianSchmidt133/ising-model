function plot_experimental_snapshots(conditions, opts)
% PLOT_EXPERIMENTAL_SNAPSHOTS  Show neural activity snapshots at Moran's I percentiles.
%   Generates one figure per condition with two rows (continuous dF/F heatmap
%   and binarised activity) and columns at specified Moran's I percentiles.
%
%   plot_experimental_snapshots()                        — all 4 conditions
%   plot_experimental_snapshots({'Expert'})              — single condition
%   plot_experimental_snapshots({'Expert'}, Threshold=3) — custom threshold

arguments
    conditions (1,:) cell   = {'Naive', 'Beginner', 'Expert', 'NoSpout'}
    opts.Threshold   double = 2.0
    opts.Percentiles (1,:) double = [10 30 50 70 90]
    opts.SaveDir     string = "."
    opts.ExSel           double  = 5
    opts.StimFrames      (1,:) double = 82:84
    opts.nTrialsIsing    double  = 3
    opts.IsingDataPath   (1,:) char = mba_p('IsingModelData_39x78_100K')
    opts.IsingGridMode   (1,:) char = 'subselect_tiled'
    opts.IsingMetric     (1,:) char = 'spatial+persistence'
    opts.ShowIsingComparison logical = true
    opts.ExperimentalDataPath (1,:) char = mba_p('ExperimentalData.mat')
end

%% --- Setup ----------------------------------------------------------------
gridDimensions = [13, 26];  % [gridY, gridX]
morans_distance_neighbors = 1;

% Skip lists per condition
Skip.Naive   = [1 9 10 16];
Skip.Beginner = [1 6 7 11];
Skip.Expert  = [1 4 12 13 14];
Skip.NoSpout = [1 4 9 10 11 13 14];

%% --- Load Grid40 ----------------------------------------------------------
if evalin('base', "exist('Grid40','var')")
    Grid40 = evalin('base', 'Grid40');
    fprintf('Using Grid40 from base workspace.\n');
else
    fprintf('Loading Grid40 from <DATA_ROOT>/Grid40.mat ...\n');
    tmp = load(mba_p('Grid40.mat'), 'Grid40');
    Grid40 = tmp.Grid40;
    clear tmp;
end

%% --- Load EntropyColourmap ------------------------------------------------
S = load('EntropyColourMap.mat');
entropyColormap = S.EntropyColourmap;

%% --- Compute spatial weight matrix (once for all conditions) --------------
valueMap = rand(gridDimensions(1), gridDimensions(2));
distanceMat = squareform(mL_distanceMat(valueMap));
uniqueDistances = unique(distanceMat);
uniqueDistances(uniqueDistances == 0) = [];

currDistInds = ismember(distanceMat, uniqueDistances(1:morans_distance_neighbors));
weightMat = zeros(size(distanceMat));
weightMat(currDistInds) = distanceMat(currDistInds);
weightMat(weightMat == inf) = 0;

%% --- Try loading pre-computed Moran's I from ExperimentalData -------------
precomputedMI = struct();
if ~isempty(opts.ExperimentalDataPath) && exist(opts.ExperimentalDataPath, 'file')
    fprintf('Loading pre-computed Moran''s I from %s ...\n', opts.ExperimentalDataPath);
    tmp = load(opts.ExperimentalDataPath, 'MoransI');
    if isfield(tmp, 'MoransI')
        precomputedMI = tmp.MoransI;
    end
    clear tmp;
end

%% --- Process each condition -----------------------------------------------
nPctiles = numel(opts.Percentiles);

for c = 1:numel(conditions)
    condition = conditions{c};
    conditionIndividual = [condition 'Individual'];

    if ~isfield(Grid40, conditionIndividual)
        warning('Field %s not found in Grid40 — skipping.', conditionIndividual);
        continue;
    end

    fprintf('\n=== %s ===\n', condition);

    %% Pool recordings
    nRecsGrid = numel(Grid40.(conditionIndividual).AllNeurons);
    pooledRaw = [];  % will be [gridY, gridX, time, trials_total]

    for r = 1:nRecsGrid
        % Skip excluded recordings
        if isfield(Skip, condition) && ismember(r, Skip.(condition))
            fprintf('  Rec %d: SKIPPED\n', r);
            continue;
        end

        if ~isfield(Grid40.(conditionIndividual).AllNeurons(r), 'P1')
            fprintf('  Rec %d: No P1 data\n', r);
            continue;
        end

        gridData_P1_cell = Grid40.(conditionIndividual).AllNeurons(r).P1;

        % Dereference cell array if needed
        if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
            gridData_P1 = gridData_P1_cell{:};
        else
            gridData_P1 = gridData_P1_cell;
        end

        if isempty(gridData_P1)
            fprintf('  Rec %d: Empty data\n', r);
            continue;
        end

        [~, ~, ~, nTrials] = size(gridData_P1);
        pooledRaw = cat(4, pooledRaw, gridData_P1);
        fprintf('  Rec %d: %d trials pooled\n', r, nTrials);
    end

    if isempty(pooledRaw)
        warning('No data for condition %s.', condition);
        continue;
    end

    [gridY, gridX, nTime, nTrials] = size(pooledRaw);
    fprintf('Pooled: %d trials, %d timepoints, grid [%d x %d]\n', ...
        nTrials, nTime, gridY, gridX);

    %% Binarize
    binarized = double(pooledRaw > opts.Threshold);

    %% Moran's I per frame (pre-computed or on-the-fly)
    nFrames = nTime * nTrials;
    moransI_vec = [];

    if isfield(precomputedMI, condition) && ~isempty(precomputedMI.(condition))
        MI_matrix = precomputedMI.(condition);  % [nTrials × nTime]
        if size(MI_matrix, 1) == nTrials && size(MI_matrix, 2) == nTime
            moransI_vec = reshape(MI_matrix', [], 1);
            fprintf('Using pre-computed Moran''s I (%d trials x %d timepoints)\n', ...
                nTrials, nTime);
        else
            fprintf('Pre-computed MI size mismatch (%dx%d vs %dx%d), recomputing...\n', ...
                size(MI_matrix, 1), size(MI_matrix, 2), nTrials, nTime);
        end
    end

    if isempty(moransI_vec)
        moransI_vec = nan(nFrames, 1);
        fprintf('Computing Moran''s I for %d frames...\n', nFrames);
        idx = 0;
        for trial = 1:nTrials
            for t = 1:nTime
                idx = idx + 1;
                grid_2D = binarized(:, :, t, trial);
                moransI_vec(idx) = mL_moransI(grid_2D, weightMat);
            end
        end
    end

    validMask = ~isnan(moransI_vec);
    validMI = moransI_vec(validMask);
    validIdx = find(validMask);

    if isempty(validMI)
        warning('All Moran''s I values are NaN for %s.', condition);
        continue;
    end

    %% Select frames at percentiles
    pctValues = prctile(validMI, opts.Percentiles);
    selFrameIdx = nan(1, nPctiles);
    selMI       = nan(1, nPctiles);

    for p = 1:nPctiles
        [~, closestPos] = min(abs(validMI - pctValues(p)));
        selFrameIdx(p) = validIdx(closestPos);
        selMI(p) = validMI(closestPos);
    end

    %% Convert linear frame index back to (time, trial)
    selTime  = nan(1, nPctiles);
    selTrial = nan(1, nPctiles);
    for p = 1:nPctiles
        fi = selFrameIdx(p);
        selTrial(p) = ceil(fi / nTime);
        selTime(p)  = fi - (selTrial(p) - 1) * nTime;
    end

    %% Determine colour-scale max from continuous data at selected frames
    vmax = 0;
    for p = 1:nPctiles
        frame = pooledRaw(:, :, selTime(p), selTrial(p));
        vmax = max(vmax, max(frame(:), [], 'omitnan'));
    end
    if vmax == 0
        vmax = 1;
    end

    %% Plot figure
    fig = figure('Name', sprintf('Experimental Snapshots — %s', condition));
    sgtitle(condition, 'FontSize', 12, 'FontWeight', 'bold');

    for p = 1:nPctiles
        contFrame = pooledRaw(:, :, selTime(p), selTrial(p));
        binFrame  = binarized(:, :, selTime(p), selTrial(p));

        % Top row: continuous dF/F
        ax1 = subplot(2, nPctiles, p);
        imagesc(flipud(contFrame), [0, vmax]);
        colormap(ax1, entropyColormap);
        axis image;
        set(ax1, 'XTick', [], 'YTick', [], 'Box', 'on', 'LineWidth', 0.5);
        title(sprintf('MI=%.3f', selMI(p)), 'FontSize', 7);
        if p == 1
            ylabel('Data', 'FontSize', 8, 'Visible', 'on');
        end

        % Bottom row: binarised
        ax2 = subplot(2, nPctiles, nPctiles + p);
        imagesc(flipud(binFrame), [0, 1]);
        colormap(ax2, [1 1 1; 1 0 0]);
        axis image;
        set(ax2, 'XTick', [], 'YTick', [], 'Box', 'on', 'LineWidth', 0.5);
        if p == 1
            ylabel('Binarised', 'FontSize', 8, 'Visible', 'on');
        end
    end

    % Tighten vertical spacing: shift bottom row up
    for p = 1:nPctiles
        ax = subplot(2, nPctiles, nPctiles + p);
        pos = get(ax, 'Position');
        set(ax, 'Position', pos + [0, 0.08, 0, 0]);
    end

    %% Save
    if opts.SaveDir ~= ""
        saveDir = fullfile(char(opts.SaveDir), 'Fig. 5 Model');
        if ~exist(saveDir, 'dir')
            mkdir(saveDir);
        end
        savePath = fullfile(saveDir, sprintf('experimental_snapshots_%s.png', condition));
        exportgraphics(fig, savePath, 'Resolution', 300);
        savePathSvg = fullfile(saveDir, sprintf('experimental_snapshots_%s.svg', condition));
        exportgraphics(fig, savePathSvg, 'ContentType', 'vector');
        fprintf('Saved: %s\n', savePath);
        fprintf('Saved: %s\n', savePathSvg);
    end
end

%% --- Expert stim-period Ising comparison (if enabled) --------------------
if opts.ShowIsingComparison && ismember('Expert', conditions)
    fprintf('\n=== Expert Stim-Period Ising Comparison ===\n');

    %% Extract single Expert recording
    gridData_P1_cell = Grid40.ExpertIndividual.AllNeurons(opts.ExSel).P1;
    if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
        gridData_P1 = gridData_P1_cell{:};
    else
        gridData_P1 = gridData_P1_cell;
    end

    [gridY, gridX, ~, nTrialsAll] = size(gridData_P1);

    %% Compute stim-period snapshot and Moran's I for trial 3
    selTrial = 3;
    snapshot = mean(gridData_P1(:, :, opts.StimFrames, selTrial), 3);
    snapshot_bin = double(snapshot > opts.Threshold);
    expSnapshot = snapshot_bin;            % [gridY × gridX]
    expMI = mL_moransI(snapshot_bin, weightMat);  % scalar

    %% Load pre-computed Ising comparison results
    resultsPath = fullfile(opts.IsingDataPath, 'IsingComparison', ...
        sprintf('IsingComparison_Results_%s_%s.mat', opts.IsingGridMode, opts.IsingMetric));
    fprintf('Loading Ising results from: %s\n', resultsPath);
    loaded = load(resultsPath, 'Results');
    Results = loaded.Results;

    best_idx = Results.Comparison.Expert.bestMatch_idx(1);
    simFilename = Results.IsingData.simFilenames{best_idx};
    nPositions = numel(Results.IsingData.MoransI_perPosition);
    nIsingExamples = 3;  % number of Ising matches to show
    fprintf('Best-matching simulation: %s (index %d, %d tile positions)\n', ...
        simFilename, best_idx, nPositions);

    %% Load best-matching Ising simulation
    simPath = fullfile(opts.IsingDataPath, simFilename);
    fprintf('Loading simulation: %s\n', simPath);
    simData = load(simPath, 'stored_spins');
    stored_spins = simData.stored_spins;  % [nFrames x 39 x 78]

    % Compute tiled positions (same logic as generateNonOverlappingPositions)
    isingRows = size(stored_spins, 2);  % 39
    isingCols = size(stored_spins, 3);  % 78
    nRowPos = floor(isingRows / gridY);
    nColPos = floor(isingCols / gridX);
    [C, R] = meshgrid(1:gridX:(nColPos*gridX), 1:gridY:(nRowPos*gridY));
    tiled_positions = [R(:), C(:)];  % [nPositions x 2]

    % Collect per-position MI timeseries into a matrix
    nIsingFrames = size(stored_spins, 1);
    tiled_MI = nan(nPositions, nIsingFrames);
    for p = 1:nPositions
        ts = Results.IsingData.MoransI_perPosition{p}{best_idx};
        tiled_MI(p, 1:numel(ts)) = ts;
    end

    % Convert stored_spins to binary once
    ising_binary = (stored_spins + 1) / 2;  % -1/+1 -> 0/1

    %% Find top-N (position, frame) matches for the selected trial
    isingSnapshots = zeros(gridY, gridX, nIsingExamples);
    isingMI = zeros(nIsingExamples, 1);

    mi_diff = abs(tiled_MI - expMI);
    [~, sortIdx] = sort(mi_diff(:));

    for ex = 1:nIsingExamples
        [bestPos, bestFrame] = ind2sub(size(tiled_MI), sortIdx(ex));
        rIdx = tiled_positions(bestPos,1):(tiled_positions(bestPos,1) + gridY - 1);
        cIdx = tiled_positions(bestPos,2):(tiled_positions(bestPos,2) + gridX - 1);
        isingSnapshots(:,:,ex) = squeeze(ising_binary(bestFrame, rIdx, cIdx));
        isingMI(ex) = tiled_MI(bestPos, bestFrame);
    end

    fprintf('  Exp MI=%.4f -> Ising MI: %s\n', expMI, ...
        strjoin(arrayfun(@(x) sprintf('%.4f', x), isingMI, 'UniformOutput', false), ' / '));

    %% Plot comparison figure
    nCols = 1 + nIsingExamples;
    figIsing = figure('Name', 'Expert Stim-Period — Experiment vs Ising');
    sgtitle('Expert Stim-Period: Experiment vs Ising', 'FontSize', 12, 'FontWeight', 'bold');

    % Column 1: experimental binarised snapshot
    ax1 = subplot(1, nCols, 1);
    imagesc(flipud(expSnapshot), [0, 1]);
    colormap(ax1, [1 1 1; 1 0 0]);
    axis image;
    set(ax1, 'XTick', [], 'YTick', [], 'Box', 'on', 'LineWidth', 0.5);
    title(sprintf('Experimental Data\nMI=%.3f', expMI), 'FontSize', 7);

    % Columns 2..nCols: Ising examples
    for ex = 1:nIsingExamples
        ax2 = subplot(1, nCols, 1 + ex);
        imagesc(isingSnapshots(:,:,ex), [0, 1]);
        colormap(ax2, [1 1 1; 1 0 0]);
        axis image;
        set(ax2, 'XTick', [], 'YTick', [], 'Box', 'on', 'LineWidth', 0.5);
        title(sprintf('Ising %d\nMI=%.3f', ex, isingMI(ex)), 'FontSize', 7);
    end

    %% Save Ising comparison figure
    if opts.SaveDir ~= ""
        saveDir = fullfile(char(opts.SaveDir), 'Fig. 5 Model');
        if ~exist(saveDir, 'dir')
            mkdir(saveDir);
        end
        savePath = fullfile(saveDir, 'experimental_ising_comparison_Expert.png');
        exportgraphics(figIsing, savePath, 'Resolution', 300);
        savePathSvg = fullfile(saveDir, 'experimental_ising_comparison_Expert.svg');
        exportgraphics(figIsing, savePathSvg, 'ContentType', 'vector');
        fprintf('Saved: %s\n', savePath);
        fprintf('Saved: %s\n', savePathSvg);
    end
end

fprintf('\nDone.\n');
end
