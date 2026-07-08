function demo_figure6c_expert()
%DEMO_FIGURE6C_EXPERT  Reproduce the Figure 6c *Expert model* trace exactly.
%   Reads the exact per-replicate model activity that Figure 6c plots for the
%   Expert condition from the double-pulse bias-matched perturbation aggregate,
%   using the same HDF5 leaf and selection as
%   ising-model/comparisons/Figure6c_BiasMatched_RasterTraces.m:
%
%     condition = Expert,  sim = 3,  size = 3,  duration = 51 frames,
%     mode = double_pulse_bias10,  matched local bias = 1.75
%
%   Plots the 50-replicate raster and the mean +/- SEM fraction-active
%   timecourse on the same seconds axis (SamplingRate = 10 Hz).
%
%   The perturbation aggregate is large and lives with the data, not in the
%   repo. Point the demo at it via the MBA_PERTURB_AGG environment variable:
%       setenv('MBA_PERTURB_AGG', 'D:\path\to\PerturbationResults_double_pulse_bias10_*.mat');
%       demo_figure6c_expert
%   (or drop the file in example_data/ as PerturbationResults_double_pulse_bias10.mat).

here = fileparts(mfilename('fullpath'));
repo = fileparts(here);
addpath(genpath(fullfile(repo, 'utils')));   % mba_p, mba_config

% ---- Figure 6c Expert selection (matches Figure6c_BiasMatched_RasterTraces) ----
cond       = 'Expert';
simIdx     = 3;                    % 0-indexed sim number
mode       = 'double_pulse_bias10';
stimSize   = 3;
chosenDur  = 51;                   % frames
biasVal    = 1.75;                 % matched local bias
SamplingRate = 10;                 % Hz (Figure 6c default)

% ---- Locate the perturbation aggregate ----
aggFile = getenv('MBA_PERTURB_AGG');
if isempty(aggFile)
    aggFile = mba_p('PerturbationResults_double_pulse_bias10.mat');
end
if ~isfile(aggFile)
    error('demo:noAggregate', [ ...
        'Perturbation aggregate not found:\n  %s\n\n' ...
        'Set MBA_PERTURB_AGG to the double_pulse_bias10 aggregate that Figure 6c\n' ...
        'reads (the matched variant whose bias grid includes 1.75), e.g.:\n' ...
        '  setenv(''MBA_PERTURB_AGG'', ''<...>\\PerturbationResults_double_pulse_bias10_*.mat'')'], ...
        aggFile);
end

% ---- Read metadata + the exact activity crop (same leaf as loadActivityCropReps) ----
preStimFrames = double(h5read(aggFile, '/pre_stim_frames'));
globalMeanSF  = double(h5read(aggFile, '/global_mean_sf'));
secPerFrame   = globalMeanSF / SamplingRate;

biasKey  = sprintf('bias_%s', strrep(sprintf('%.2f', biasVal), '.', 'p'));   % 'bias_1p75'
leafPath = sprintf('/experiments/%s/sim_%d/%s/size_%d/dur_%d/%s/activity_crop', ...
                   cond, simIdx, mode, stimSize, chosenDur, biasKey);
reps = h5read(aggFile, leafPath);
if size(reps, 1) > size(reps, 2), reps = reps'; end   % -> replicates x frames

nReps   = size(reps, 1);
nFrames = size(reps, 2);
stimOnsetFrame = preStimFrames + 1;
t = ((1:nFrames) - stimOnsetFrame) * secPerFrame;      % seconds, 0 = stim onset
stimEndSec = chosenDur * secPerFrame;

% Fraction active -> percent for display if stored as a fraction
disp_reps = reps;
if max(reps(:)) <= 1.0, disp_reps = reps * 100; end
mu  = mean(disp_reps, 1);
sem = std(disp_reps, 0, 1) ./ sqrt(nReps);

fprintf('Loaded Figure 6c Expert model: %d replicates x %d frames\n', nReps, nFrames);
fprintf('  secPerFrame=%.4f s, stim window [0, %.2f] s\n', secPerFrame, stimEndSec);

% ---- Plot: raster + mean +/- SEM (Figure 6c model panels) ----
figure('Name', 'Figure6c_Expert_model');
tl = tiledlayout(2, 1);
title(tl, 'Figure 6c: Expert model');

nexttile;
imagesc(t, 1:nReps, disp_reps);
hold on; xline(0, 'w-'); xline(stimEndSec, 'w-');
S = load('EntropyColourMap.mat'); colorbar; colormap(gca, S.EntropyColourmap);
xlabel('time from stim onset (s)'); ylabel('replicate'); title('Per-replicate fraction active (%)');

nexttile;
fill([t fliplr(t)], [mu+sem fliplr(mu-sem)], [0.2 0.2 0.2], ...
     'FaceAlpha', 0.25, 'EdgeColor', 'none'); hold on;
plot(t, mu, 'k', 'LineWidth', 1.5);
yl = ylim; patch([0 stimEndSec stimEndSec 0], [yl(1) yl(1) yl(2) yl(2)], ...
      [1 0.9 0.6], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
xlabel('time from stim onset (s)'); ylabel('% active'); grid on;
title('Mean \pm SEM across replicates');

cfg = mba_config();
if ~exist(cfg.figure_root, 'dir'); mkdir(cfg.figure_root); end
outPath = fullfile(cfg.figure_root, 'demo_figure6c_expert.png');
saveas(gcf, outPath);
fprintf('Saved figure: %s\n', outPath);
end
