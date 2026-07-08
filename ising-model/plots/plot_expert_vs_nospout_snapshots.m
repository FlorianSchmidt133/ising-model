function plot_expert_vs_nospout_snapshots(opts)
% PLOT_EXPERT_VS_NOSPOUT_SNAPSHOTS  Compare binarised stim-period snapshots
%   between Expert and NoSpout conditions.
%
%   plot_expert_vs_nospout_snapshots()           — default parameters
%   plot_expert_vs_nospout_snapshots(Threshold=3) — custom threshold

arguments
    opts.Threshold       double  = 2.0
    opts.StimFrames      (1,:) double = 82:84
    opts.ExSel           double  = 5
    opts.ExpertTrials    (1,:) double = [1, 2, 3]
    opts.NoSel           double  = 5
    opts.NoSel2          double  = 6
    opts.NoSpoutTrialRange (1,:) double = 11:30
    opts.NoSpoutTrials   (1,:) double = [12, 24, 18]
    opts.SaveDir         string  = "."
end

%% --- Setup ----------------------------------------------------------------
gridDimensions = [13, 26];
morans_distance_neighbors = 1;

% Load Grid40
if evalin('base', "exist('Grid40','var')")
    Grid40 = evalin('base', 'Grid40');
    fprintf('Using Grid40 from base workspace.\n');
else
    fprintf('Loading Grid40 from <DATA_ROOT>/Grid40.mat ...\n');
    tmp = load(mba_p('Grid40.mat'), 'Grid40');
    Grid40 = tmp.Grid40;
    clear tmp;
end

% Compute spatial weight matrix
valueMap = rand(gridDimensions(1), gridDimensions(2));
distanceMat = squareform(mL_distanceMat(valueMap));
uniqueDistances = unique(distanceMat);
uniqueDistances(uniqueDistances == 0) = [];

currDistInds = ismember(distanceMat, uniqueDistances(1:morans_distance_neighbors));
weightMat = zeros(size(distanceMat));
weightMat(currDistInds) = distanceMat(currDistInds);
weightMat(weightMat == inf) = 0;

%% --- Expert snapshots -----------------------------------------------------
gridData_P1_cell = Grid40.ExpertIndividual.AllNeurons(opts.ExSel).P1;
if ~isempty(gridData_P1_cell) && iscell(gridData_P1_cell)
    gridData_P1 = gridData_P1_cell{:};
else
    gridData_P1 = gridData_P1_cell;
end

nExpert = numel(opts.ExpertTrials);
expSnapshots = zeros(gridDimensions(1), gridDimensions(2), nExpert);
expMI = zeros(nExpert, 1);

for k = 1:nExpert
    t = opts.ExpertTrials(k);
    snap = mean(gridData_P1(:, :, opts.StimFrames, t), 3);
    expSnapshots(:,:,k) = double(snap > opts.Threshold);
    expMI(k) = mL_moransI(expSnapshots(:,:,k), weightMat);
    fprintf('Expert trial %d: MI = %.4f\n', t, expMI(k));
end

%% --- NoSpout snapshots ----------------------------------------------------
noSpoutData = cat(4, ...
    Grid40.NoSpoutIndividual.AllNeurons(opts.NoSel).P1{:}(:,:,:, opts.NoSpoutTrialRange), ...
    Grid40.NoSpoutIndividual.AllNeurons(opts.NoSel2).P1{:}(:,:,:, opts.NoSpoutTrialRange));

nNoSpout = numel(opts.NoSpoutTrials);
noSpoutSnapshots = zeros(gridDimensions(1), gridDimensions(2), nNoSpout);
noSpoutMI = zeros(nNoSpout, 1);

for k = 1:nNoSpout
    t = opts.NoSpoutTrials(k);
    snap = mean(noSpoutData(:, :, opts.StimFrames, t), 3);
    noSpoutSnapshots(:,:,k) = double(snap > opts.Threshold);
    noSpoutMI(k) = mL_moransI(noSpoutSnapshots(:,:,k), weightMat);
    fprintf('NoSpout trial %d: MI = %.4f\n', t, noSpoutMI(k));
end

%% --- Plot figure (3 rows x 2 columns) ------------------------------------
nRows = max(nExpert, nNoSpout);
fig = figure('Name', 'Stim-Period: Expert vs NoSpout');
sgtitle('Stim-Period: Expert vs NoSpout', 'FontSize', 12, 'FontWeight', 'bold');

% Left column: Expert trials
for k = 1:nExpert
    ax = subplot(nRows, 2, (k-1)*2 + 1);
    imagesc(flipud(expSnapshots(:,:,k)), [0, 1]);
    colormap(ax, [1 1 1; 1 0 0]);
    axis image;
    set(ax, 'XTick', [], 'YTick', [], 'Box', 'on', 'LineWidth', 0.5);
    title(sprintf('Expert trial %d', opts.ExpertTrials(k)), 'FontSize', 7);
    if k == 1
        ylabel('Expert', 'FontSize', 8, 'FontWeight', 'bold', 'Visible', 'on');
    end
end

% Right column: NoSpout trials
for k = 1:nNoSpout
    ax = subplot(nRows, 2, (k-1)*2 + 2);
    imagesc(flipud(noSpoutSnapshots(:,:,k)), [0, 1]);
    colormap(ax, [1 1 1; 1 0 0]);
    axis image;
    set(ax, 'XTick', [], 'YTick', [], 'Box', 'on', 'LineWidth', 0.5);
    title(sprintf('NoSpout trial %d', opts.NoSpoutTrials(k)), 'FontSize', 7);
    if k == 1
        ylabel('NoSpout', 'FontSize', 8, 'FontWeight', 'bold', 'Visible', 'on');
    end
end

%% --- Save -----------------------------------------------------------------
if opts.SaveDir ~= ""
    saveDir = fullfile(char(opts.SaveDir), 'Fig. 5 Model');
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end
    savePath = fullfile(saveDir, 'expert_vs_nospout_stim_comparison.png');
    exportgraphics(fig, savePath, 'Resolution', 300);
    savePathSvg = fullfile(saveDir, 'expert_vs_nospout_stim_comparison.svg');
    exportgraphics(fig, savePathSvg, 'ContentType', 'vector');
    fprintf('Saved: %s\n', savePath);
    fprintf('Saved: %s\n', savePathSvg);
end

fprintf('\nDone.\n');
end
