%% =========================================================================
%% === Figure 6: State-Dependent Amplification (Ising perturbations)
%% =========================================================================
%
% Produces the data-side figures for paper Figure 6 (panels a-l).
%
% Panel mapping:
%   a — Schematic (5x5 grid Active/Inactive/Stimulus). Hand-drawn; no script.
%   b — Per-condition raster + fraction-active timecourse, DATA.
%       Script: plot_FractionActive_BeforeDuring.m
%   c — Per-condition raster + fraction-active timecourse, MODEL (Ising).
%       Rendered by Python (Figure5/comparisons/ising_visualizations.py); see stub.
%   d — Pre-stim fraction active swarm: Data + Model.
%       Scripts: plot_FractionActive_PerRecording_Swarm.m (data),
%                plot_FractionActive_Ising_Swarm.m       (model)
%   e — Stim fraction active swarm: Data + Model. (Same scripts as d.)
%   f — EC50 across stim durations and conditions.
%       Script: Figure5_IsingPerturbationAnalysis.m (figuresOnly mode)
%   g — Single-frame snapshots Expert vs NoSpout, DATA.
%       Script: plot_expert_vs_nospout_snapshots.m
%   h — Single-frame snapshots Expert vs NoSpout, MODEL.
%       Rendered by Python (Figure5/comparisons/generate_perturbation_snapshots.py); stub.
%   i — Single-frame snapshots Hit vs Miss (small stim), DATA.
%       Rendered by Python (Figure5/plots/plot_hit_vs_miss_snapshots.py); stub.
%   j — Largest connected group (4x4 stim), Data + Model.
%       Scripts: plotSwarm_BinarisedConnected.m (data),
%                local plot_IsingConnectedSwarm  (model, defined at bottom of this file)
%   k — Largest connected group (small/2x2 stim), Data + Model.
%   l — Schematic (gain/beta attention modulation). Hand-drawn; no script.
%
% Run mode (matches Figure4 convention):
%   runMode = 'figures'  (default) — produce panels b/d/e/f/g/j/k figures.
%   runMode = 'stats'              — delegate to compute_figure6_stats.m
%                                    (writes Figure6.md, Figure6_stats_results.mat).
%   runMode = 'both'               — figures, then stats, in one MATLAB session.
%
%
%% Load data — same files compute_figure6_stats.m loads, so plots and stats agree
if ~exist('ActivityData','var'), load(mba_p('RawData3.mat'),          'ActivityData'); end
if ~exist('params','var'),       load(mba_p('RawData3.mat'),          'params');       end
if ~exist('Stimuli','var'),      load(mba_p('RawData3.mat'),          'Stimuli');      end
if ~exist('Performance','var'),  load(mba_p('RawData_September18.mat'),'Performance'); end
if ~exist('Rec','var'),          load(mba_p('correctRec.mat'),         'Rec');          end
if ~exist('Grid40','var'),       load(mba_p('Grid40_September18.mat'), 'Grid40');       end

params.NaiveIndividual           = params.Naive;
params.BeginnerIndividual        = params.Beginner;
params.ExpertIndividual          = params.Expert;
params.NoSpoutIndividual         = params.NoSpout;
params.SmallStimExpertIndividual = params.Expert;
params.SmallStimExpert           = params.Expert;

%% Skip lists 
Skip = struct();
Skip.Naive           = [1 9 10 14 15 16];
Skip.Beginner        = [1 6 7 11 12];
Skip.Expert          = [1 3 4 11:17];
Skip.Expert          = [1 2 3 4 12 13 14 15];

Skip.NoSpout         = [1 4 9 10 11 13 14];
Skip.SmallStimExpert = [];

%% Path setup — pick up plotters from Figure4 and Figure5 trees
thisDir   = fileparts(mfilename('fullpath'));
fig4Plots = fullfile(thisDir, '..', 'Figure4', 'plots');
fig5Plots = fullfile(thisDir, '..', 'Figure5', 'plots');
fig5Comp  = fullfile(thisDir, '..', 'Figure5', 'comparisons');
sharedHlp = fullfile(thisDir, '..', 'shared', 'helpers');
for d = {fig4Plots, fig5Plots, fig5Comp, sharedHlp}
    if ~contains(path, d{1}), addpath(d{1}); end
end

%% Constants
ISING_NPZ  = mba_p('IsingPertubation\moransI+activity\double_pulse10\heatmap_data.npz');
ISING_MODE = 'double_pulse10';
ISING_DUR  = 59;
SAVE_ROOT  = 'Fig. 6 Ising Model';

%% Run mode: set runMode = 'figures' / 'stats' / 'both' before running
runMode   = 'figures';
doFigures = ismember(runMode, {'figures', 'both'});
doStats   = ismember(runMode, {'stats',   'both'});

%% =========================================================================
%% Panel a — Schematic (Active/Inactive/Stimulus). No script.
%% =========================================================================
if doFigures

%% =========================================================================
%% Panel b — Data raster + fraction-active timecourse, all 4 conditions
%% =========================================================================
plot_FractionActive_BeforeDuring(Grid40, Performance, Skip, Rec, params, ...
    'SmoothingWindow',       1, ...
    'BinarisationThreshold', 2, ...
    'Conditions',            {'NaiveIndividual','BeginnerIndividual','ExpertIndividual','NoSpoutIndividual'}, ...
    'TrialTypes',            {'all','all','all','all'}, ...
    'DisplayWindow',         [-1, 3]);

saveMyFig("Fig6_b_FractionActive_BeforeDuring", ...
    fullfile(SAVE_ROOT, 'b_FractionActive_BeforeDuring'), 'All');
close all;

%% =========================================================================
%% Panel c — Model raster + fraction-active timecourse (Ising)
%% =========================================================================
% Rendered by Python. Uncomment to invoke from MATLAB; requires a Python env
% with h5py / numpy / matplotlib on PATH.
%
% pyScript = fullfile(fig5Comp, 'ising_visualizations.py');
% pyOut    = fullfile(SAVE_ROOT, 'c_Ising_Raster_Traces');
% if ~exist(pyOut, 'dir'), mkdir(pyOut); end
% system(sprintf('python "%s" --npz "%s" --out "%s"', pyScript, ISING_NPZ, pyOut));

%% =========================================================================
%% Panels d, e — Fraction active swarms (pre-stim and stim)
%% =========================================================================
% --- Data side: per-animal aggregation, mirrors compute_figure6_stats.m ---
plot_FractionActive_PerRecording_Swarm(Grid40, Performance, Stimuli, Skip, Rec, params, ...
    'SmoothingWindow',       1, ...
    'BinarisationThreshold', 2, ...
    'Conditions',            {'Naive','BeginnerIndividual','ExpertIndividual','NoSpoutIndividual'}, ...
    'TrialTypes',            {'all','miss','all','all'}, ...
    'StimFrames',            81:105, ...
    'AggregationLevel',      'animal', ...
    'MaskType',              'activity');

plot_FractionActive_PerRecording_Swarm(Grid40, Performance, Stimuli, Skip, Rec, params, ...
    'SmoothingWindow',       1, ...
    'BinarisationThreshold', 2, ...
    'Conditions',            {'Naive','BeginnerIndividual','ExpertIndividual','NoSpoutIndividual'}, ...
    'TrialTypes',            {'all','miss','all','all'}, ...
    'StimFrames',            81:105, ...
    'modes', {'prestim','stim','nostim','full'},...
    'AggregationLevel',      'animal');



saveMyFig("Fig6_de_FractionActive_PerAnimal_Swarm", ...
    fullfile(SAVE_ROOT, 'de_FractionActive_Swarm_Data'), 'All');
close all;

plot_FractionActive_PerRecording_Swarm(Grid40, Performance, Stimuli, Skip, Rec, params, ...
    'SmoothingWindow',       1, ...
    'BinarisationThreshold', 2, ...
    'Conditions',            {'Naive','BeginnerIndividual','ExpertIndividual','NoSpoutIndividual'}, ...
    'TrialTypes',            {'all','miss','all','all'}, ...
    'StimFrames',            81:105);

% --- Data side, normalized to Naive mean (Naive omitted from plot) ---
plot_FractionActive_PerRecording_Swarm(Grid40, Performance, Stimuli, Skip, Rec, params, ...
    'SmoothingWindow',       1, ...
    'BinarisationThreshold', 2, ...
    'Conditions',            {'Naive','BeginnerIndividual','ExpertIndividual','NoSpoutIndividual'}, ...
    'TrialTypes',            {'all','miss','all','all'}, ...
    'StimFrames',            81:105, ...
    'AggregationLevel',      'animal', ...
    'NormalizeToNaive',      true);

saveMyFig("Fig6_de_FractionActive_PerAnimal_Swarm_NormalizedNaive", ...
    fullfile(SAVE_ROOT, 'de_FractionActive_Swarm_Data_NormalizedNaive'), 'All');
close all;

% --- Model side ---
plot_FractionActive_Ising_Swarm(ISING_NPZ, params, ...
    'StimSize',     2, ...
    'StimDuration', ISING_DUR, ...
    'StimMode',     ISING_MODE, ...
    'Conditions',   {'Naive','Beginner','Expert','NoSpout'}, ...
    'BatchSize',    100);

saveMyFig("Fig6_de_FractionActive_Ising_Swarm", ...
    fullfile(SAVE_ROOT, 'de_FractionActive_Swarm_Model'), 'All');
close all;

% --- Model side, normalized to Naive mean ---
plot_FractionActive_Ising_Swarm(ISING_NPZ, params, ...
    'StimSize',         2, ...
    'StimDuration',     ISING_DUR, ...
    'StimMode',         ISING_MODE, ...
    'Conditions',       {'Naive','Beginner','Expert','NoSpout'}, ...
    'BatchSize',        100, ...
    'NormalizeToNaive', true);

 results = verify_FractionActive_Ising_pair(ISING_NPZ, params, 'Beginner', 'NoSpout', ...
      'StimSize',         2, ...
      'StimDuration',     ISING_DUR, ...
      'StimMode',         ISING_MODE, ...
      'BatchSize',        100, ...
      'NormalizeToNaive', true, ...
      'Period',           'prestim');

saveMyFig("Fig6_de_FractionActive_Ising_Swarm_NormalizedNaive", ...
    fullfile(SAVE_ROOT, 'de_FractionActive_Swarm_Model_NormalizedNaive'), 'All');
close all;

%% =========================================================================
%% EC50 across stim durations (Hill fits)
%% =========================================================================
% Run Figure5_IsingPerturbationAnalysis.m in figuresOnly mode so it loads
% the pre-saved Gating .mat (<DATA_ROOT>/Analysis\AllDurationAnalysis.mat by default).
config = struct();
config.figuresOnly = true;
config.stimMode    = ISING_MODE;
config.outputPath  = mba_p('Analysis');
config.targetDurationSec          = 2.0;
config.collapseTargetDurationsSec = [0.5 1.0 2.0 5.0 10.0];
try
    Figure5_IsingPerturbationAnalysis;
catch ME
    warning('Figure6:EC50Panel', 'EC50 panel failed: %s', ME.message);
end
clear config;
close all;

%% =========================================================================
%% Panel g — Snapshots: Expert vs NoSpout, DATA
%% =========================================================================
plot_expert_vs_nospout_snapshots('SaveDir', fullfile(SAVE_ROOT, 'g_Expert_vs_NoSpout_Snapshots'));
close all;

%% =========================================================================
%% Panel h — Snapshots: Expert vs NoSpout, MODEL (Ising)
%% =========================================================================
% Rendered by Python. Uncomment to invoke.
%
% pyScript = fullfile(fig5Comp, 'generate_perturbation_snapshots.py');
% pyOut    = fullfile(SAVE_ROOT, 'h_Ising_Snapshots');
% if ~exist(pyOut, 'dir'), mkdir(pyOut); end
% system(sprintf('python "%s" --npz "%s" --out "%s"', pyScript, ISING_NPZ, pyOut));

%% =========================================================================
%% Panel i — Snapshots: Hit vs Miss (small stim), DATA
%% =========================================================================
% Rendered by Python; no MATLAB equivalent. Uncomment to invoke.
%
% pyScript = fullfile(fig5Plots, 'plot_hit_vs_miss_snapshots.py');
% pyOut    = fullfile(SAVE_ROOT, 'i_HitMiss_Snapshots');
% if ~exist(pyOut, 'dir'), mkdir(pyOut); end
% system(sprintf('python "%s" --output "%s"', pyScript, pyOut));

%% =========================================================================
%% Panel j — Largest connected group (4x4 stim), Data + Model
%% =========================================================================
cfgJ = struct();
cfgJ.StimFrames   = 81:105;
cfgJ.BinThreshold = 2;
cfgJ.Groups = struct(...
    'Condition', {'Naive',                'Beginner',              'Expert',           'NoSpout'}, ...
    'GridCond',  {'NaiveIndividual',      'BeginnerIndividual',    'ExpertIndividual', 'NoSpoutIndividual'}, ...
    'TrialType', {'P1',                   'miss',                  'P1',               'P1'}, ...
    'Label',     {'Naive',                'Beginner',              'Expert',           'NoSpout'}, ...
    'Color',     {[0.3373 0.7059 0.9137], [0.8431 0.2549 0.6078], [0 0.6196 0.4510], [0.8353 0.3686 0]});

% --- Data side ---
plotSwarm_BinarisedConnected(Grid40, Performance, Stimuli, Skip, cfgJ);
saveMyFig("Fig6_j_LargestConnected_Data", ...
    fullfile(SAVE_ROOT, 'j_LargestConnected_Data'), 'All');
close all;

% --- Model side: per-rep mean blob_area in stim window ---
plot_IsingConnectedSwarm(ISING_NPZ, ...
    {'Naive','Beginner','Expert','NoSpout'}, 4, ISING_DUR, ISING_MODE, cfgJ.Groups, ...
    'Ising — Largest connected (4x4 stim)');
saveMyFig("Fig6_j_LargestConnected_Model", ...
    fullfile(SAVE_ROOT, 'j_LargestConnected_Model'), 'All');
close all;

%% =========================================================================
%% Panel k — Largest connected group (small / 2x2 stim)
%% =========================================================================
% --- Data side: SmallStimExpert Hit vs Miss ---
cfgK = struct();
cfgK.StimFrames   = 81:105;
cfgK.BinThreshold = 2;
cfgK.Groups = struct(...
    'Condition', {'SmallStimExpert',           'SmallStimExpert'}, ...
    'GridCond',  {'SmallStimExpertIndividual', 'SmallStimExpertIndividual'}, ...
    'TrialType', {'hit',                       'miss'}, ...
    'Label',     {'Hit',                       'Miss'}, ...
    'Color',     {[0 0 0],                     [1 0 0]});

plotSwarm_BinarisedConnected(Grid40, Performance, Stimuli, Skip, cfgK);
saveMyFig("Fig6_k_LargestConnected_SmallStim_Data", ...
    fullfile(SAVE_ROOT, 'k_LargestConnected_SmallStim_Data'), 'All');
close all;

% --- Data side (variant): SmallStimExpert Hit + Miss + NoSpout (P1 trials) ---
% NoSpout has no hit/miss distinction (no rewards) and isn't small-stim
% filtered upstream (ComputeAndPrepStructs.m only builds SmallStimBeginner/
% SmallStimExpert), so we use the full NoSpout condition with TrialType='P1'.
% The slight stim-size mix-in is acceptable for this qualitative comparison
% and pairs with the model side, which already shows Expert vs NoSpout.
cfgK_withNoSpout = cfgK;
cfgK_withNoSpout.Groups(end+1) = struct( ...
    'Condition', 'NoSpout', ...
    'GridCond',  'NoSpoutIndividual', ...
    'TrialType', 'P1', ...
    'Label',     'NoSpout', ...
    'Color',     [0.8353 0.3686 0]);
plotSwarm_BinarisedConnected(Grid40, Performance, Stimuli, Skip, cfgK_withNoSpout);
saveMyFig("Fig6_k_LargestConnected_SmallStim_Data_withNoSpout", ...
    fullfile(SAVE_ROOT, 'k_LargestConnected_SmallStim_Data_withNoSpout'), 'All');
close all;

% --- Model side: 2x2 Expert vs NoSpout ---
modelGroupsK = struct(...
    'Condition', {'Expert',           'NoSpout'}, ...
    'GridCond',  {'ExpertIndividual', 'NoSpoutIndividual'}, ...
    'TrialType', {'P1',               'P1'}, ...
    'Label',     {'Expert',           'NoSpout'}, ...
    'Color',     {[0 0.6196 0.4510],  [0.8353 0.3686 0]});

plot_IsingConnectedSwarm(ISING_NPZ, ...
    {'Expert','NoSpout'}, 2, ISING_DUR, ISING_MODE, modelGroupsK, ...
    'Ising — Largest connected (2x2 stim)');
saveMyFig("Fig6_k_LargestConnected_SmallStim_Model", ...
    fullfile(SAVE_ROOT, 'k_LargestConnected_SmallStim_Model'), 'All');
close all;

%% =========================================================================
%% Panel l — Schematic (gain/beta attention). No script.
%% =========================================================================

end % if doFigures

%% =========================================================================
%% === STATISTICS ===
%% =========================================================================
if doStats
% Delegate to the stats script. Skip is inherited from this workspace via
% `if ~exist('Skip','var')` inside the stats script — single source of truth.
statsScript = fullfile(fileparts(mfilename('fullpath')), '..', 'statistics', 'compute_figure6_stats.m');
run(statsScript);
end % if doStats

%% =========================================================================
%% Local helpers
%% =========================================================================
function plot_IsingConnectedSwarm(npzPath, conditions, stimSize, stimDur, stimMode, groupsStruct, titleStr)
%PLOT_ISINGCONNECTEDSWARM  Swarm of per-rep mean largest-connected-component
% area inside the stim window. Mirrors the data-side `plotSwarm_BinarisedConnected`
% layout so panels j/k Model line up visually with their Data counterparts.
    isingRaw = loadIsingHeatmapData(char(npzPath));
    spagKey  = matlab.lang.makeValidName(sprintf('__spaghetti_indices_%d__', stimDur));
    if ~isfield(isingRaw, spagKey)
        error('plot_IsingConnectedSwarm:MissingKey', 'Spaghetti indices key %s not found', spagKey);
    end
    frameIdx = isingRaw.(spagKey)(:)';
    PRE_STIM_FRAMES = 400;
    stimMask = frameIdx >= PRE_STIM_FRAMES & frameIdx < (PRE_STIM_FRAMES + stimDur);

    nC = numel(conditions);
    colors = zeros(nC, 3);
    labels = {groupsStruct.Label};
    for ci = 1:nC
        match = find(strcmp(labels, conditions{ci}), 1);
        if ~isempty(match) && isfield(groupsStruct(match), 'Color')
            colors(ci, :) = groupsStruct(match).Color;
        else
            defaults = [0.3373 0.7059 0.9137; 0.8431 0.2549 0.6078; ...
                        0.0000 0.6196 0.4510; 0.8353 0.3686 0.0000];
            colors(ci, :) = defaults(mod(ci-1, size(defaults,1))+1, :);
        end
    end

    tsData = isingToTsData(isingRaw, stimSize, stimDur, conditions, colors, stimMode);

    allValues = [];
    allGroups = {};
    for gi = 1:nC
        mat = tsData.blob_area{gi};
        if isempty(mat), continue; end
        if size(mat, 2) ~= numel(frameIdx) && size(mat, 1) == numel(frameIdx)
            mat = mat';
        end
        perRep = mean(mat(:, stimMask), 2, 'omitnan');
        allValues = [allValues; perRep(:)];
        allGroups = [allGroups; repmat(conditions(gi), numel(perRep), 1)];
    end

    figure('Name', titleStr, 'Color', 'w');
    g = gramm('x', allGroups, 'y', allValues, 'color', allGroups);
    g.set_color_options('map', colors);
    g.set_order_options('x', conditions, 'color', conditions);
    g.geom_swarm('alpha', 0.5, 'point_size', 5, 'type', 'up', 'corral', 'none');
    g.set_names('x', 'Condition', 'y', 'Largest connected component (per rep)');
    g.set_title(titleStr);
    g.no_legend();
    g.draw();
end
