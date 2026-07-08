function out = Figure6h_BiasMatched_EC50(varargin)
%FIGURE6H_BIASMATCHED_EC50  EC50-vs-duration ensemble using the bias-matched
%   pipeline. Replaces the legacy clamped (`double_pulse10`) EC50 input.
%
%   For each condition the matcher picks a "best bias" value (per size,
%   window). This wrapper uses the matcher's selection to stitch a per-
%   condition composite of the per-mode `AllDurationAnalysis.mat` caches
%   (one per `double_pulse_bias10_<bv>` folder), then runs
%   Figure5_IsingPerturbationAnalysis.m in figuresOnly mode against that
%   composite to render the 7 EC50 panels:
%
%     1. EC50_vs_Duration              (linear y, all durations)
%     2. EC50_vs_Duration_area         (log y, area=EC50^2 — the canonical
%                                       Fig 6h panel)
%     3. EC50_area_at_2s_swarm_{linear,log}   (single-duration swarm, both y-scales)
%     4. EC50_area_at_2s_dots_{linear,log}    (mean ± SEM at target duration, both y-scales)
%     4b. EC50_area_at_2s_box_{linear,log}    (boxplot at target duration, both y-scales)
%     5. EC50_area_vs_Duration_clip2s  (line plot clipped to <= 2s)
%     6. EC50_area_collapseTargetDurations_swarm   (pooled across [0.5..10s])
%     7. EC50_area_collapseTargetDurations_meanStd (pooled mean ± STD)
%
%   Inputs (Name-Value):
%     'BiasMatcherDir'      Folder containing matcher_output_size<N>.mat.
%       Default: <PerturbAnalysisRoot>/BiasMatchExperiment.
%     'BiasMatchSize'       Stim size used to read matcher's best bias.
%       Default 2.
%     'BiasMatchWindow'     Matcher window for best-bias selection.
%       Default 'full'. Other options: 'onset' / 'offset'.
%     'PerturbAnalysisRoot' Directory containing the per-mode subfolders
%       (`double_pulse_bias10_*`, `double_pulse10`). On cluster:
%       '~/IsingPerturbations/moransI+activity'.
%       Default mba_p('IsingModelData_39x78_100K\IsingPerturbations').
%     'OutputDir'           Where the composite cache + the 7 PNGs go.
%       Default 'Fig. 6 Ising Model\h_BiasMatched_EC50'.
%     'TargetDurationSec'   Default 2.0.
%     'CollapseTargetDurationsSec' Default [0.5 1.0 2.0 5.0 10.0].
%     'Conditions'          Default {'Naive','Beginner','Expert','NoSpout'}.
%     'ClampedMode'         Used as fallback when best bias is Inf/clamped.
%                           Default 'double_pulse10'.

    p = inputParser;
    addParameter(p, 'BiasMatcherDir',         '', @ischar);
    addParameter(p, 'BiasMatchSize',          2, @isnumeric);
    addParameter(p, 'BiasMatchWindow',        'full', @ischar);
    addParameter(p, 'PerturbAnalysisRoot', ...
        mba_p('IsingModelData_39x78_100K\IsingPerturbations'), @ischar);
    addParameter(p, 'OutputDir', ...
        'Fig. 6 Ising Model\h_BiasMatched_EC50', @ischar);
    addParameter(p, 'TargetDurationSec',      2.0, @isnumeric);
    addParameter(p, 'CollapseTargetDurationsSec', [0.5 1.0 2.0 5.0 10.0], @isnumeric);
    addParameter(p, 'Conditions', ...
        {'Naive','Beginner','Expert','NoSpout'}, @iscell);
    addParameter(p, 'ClampedMode',            'double_pulse10', @ischar);
    addParameter(p, 'MatchingMode', 'perCondition', @(x) ischar(x) && ...
        any(strcmpi(x, {'perCondition', 'global', 'expert'})));
    parse(p, varargin{:});
    opts = p.Results;

    if isempty(opts.BiasMatcherDir)
        opts.BiasMatcherDir = fullfile(opts.PerturbAnalysisRoot, ...
            sprintf('BiasMatchExperiment_%s', opts.MatchingMode));
    end
    legacyOutDir = 'Fig. 6 Ising Model\h_BiasMatched_EC50';
    if strcmp(opts.OutputDir, legacyOutDir)
        opts.OutputDir = sprintf('%s_%s', legacyOutDir, opts.MatchingMode);
    end
    if ~exist(opts.OutputDir, 'dir'), mkdir(opts.OutputDir); end

    fprintf('=== Figure6h_BiasMatched_EC50 ===\n');
    fprintf('Matcher dir : %s\n', opts.BiasMatcherDir);
    fprintf('Bias size   : %d\n', opts.BiasMatchSize);
    fprintf('Bias window : %s\n', opts.BiasMatchWindow);
    fprintf('Perturb root: %s\n', opts.PerturbAnalysisRoot);
    fprintf('Output dir  : %s\n', opts.OutputDir);

    %% --- 1. Load matcher and resolve per-condition best biases ----------
    matcherFile = fullfile(opts.BiasMatcherDir, ...
        sprintf('matcher_output_size%d.mat', opts.BiasMatchSize));
    if ~exist(matcherFile, 'file')
        % Fall back to "all sizes" file
        matcherFile = fullfile(opts.BiasMatcherDir, 'matcher_output.mat');
        if ~exist(matcherFile, 'file')
            error('Matcher output not found in %s', opts.BiasMatcherDir);
        end
    end
    fprintf('Loading matcher: %s\n', matcherFile);
    M = load(matcherFile, 'out');
    if ~isfield(M, 'out')
        error('Loaded file does not contain `out` struct: %s', matcherFile);
    end
    sizeKey = sprintf('size_%d', opts.BiasMatchSize);
    if isfield(M.out, 'bySize') && isfield(M.out.bySize, sizeKey)
        sizeOut = M.out.bySize.(sizeKey);
    elseif isfield(M.out, 'bestBiasPerCondition')
        sizeOut = M.out;
    else
        error('Matcher output has no recognised structure for size=%d', opts.BiasMatchSize);
    end

    bestBiasByCond = struct();
    modeNameByCond = struct();
    for c = 1:numel(opts.Conditions)
        cond = opts.Conditions{c};
        if ~isfield(sizeOut.bestBiasPerCondition, cond) || ...
                ~isfield(sizeOut.bestBiasPerCondition.(cond), opts.BiasMatchWindow)
            warning('No bestBias for %s/%s; skipping condition', cond, opts.BiasMatchWindow);
            continue;
        end
        bv = sizeOut.bestBiasPerCondition.(cond).(opts.BiasMatchWindow).bias_value;
        bestBiasByCond.(cond) = bv;
        if isinf(bv)
            modeNameByCond.(cond) = opts.ClampedMode;
        else
            modeNameByCond.(cond) = sprintf('double_pulse_bias10_%s', ...
                strrep(sprintf('%.2f', bv), '.', 'p'));
        end
        fprintf('  %-10s bias=%-8s  mode=%s\n', cond, ...
            biasLabel(bv), modeNameByCond.(cond));
    end

    %% --- 2. Load per-mode caches and stitch composite -------------------
    % Use the FIRST condition's mode cache as the template (provides
    % auxiliary variables like stimulusDurations, metricsDisplayModes etc.).
    condList = fieldnames(modeNameByCond);
    if isempty(condList)
        error('No conditions resolved to a bias mode — nothing to stitch.');
    end

    templateMode = modeNameByCond.(condList{1});
    templateCache = perModeCachePath(opts.PerturbAnalysisRoot, templateMode);
    if ~exist(templateCache, 'file')
        error('Template cache missing: %s', templateCache);
    end
    fprintf('Template cache: %s\n', templateCache);
    base = load(templateCache);  % loads many vars

    % Composite Gating / friends (initialised from base, then overwritten
    % per condition with that condition's mode cache entries).
    composite = struct();
    composite.AllDurationMetrics            = base.AllDurationMetrics;
    composite.AllDurationGating             = base.AllDurationGating;
    composite.AllDurationStats              = base.AllDurationStats;
    composite.AllDurationPreStimEffects     = safe_get(base, 'AllDurationPreStimEffects');
    composite.AllDurationPropDynamics       = safe_get(base, 'AllDurationPropDynamics');
    composite.AllDurationBlobInteractions   = safe_get(base, 'AllDurationBlobInteractions');
    composite.AllDurationMoransIDynamics    = safe_get(base, 'AllDurationMoransIDynamics');
    composite.AllDurationMetrics_BM         = safe_get(base, 'AllDurationMetrics_BM');
    composite.AllDurationGating_BM          = safe_get(base, 'AllDurationGating_BM');
    composite.AllDurationStats_BM           = safe_get(base, 'AllDurationStats_BM');

    % Auxiliary variables — keep template values
    composite.stimulusDurations             = base.stimulusDurations;
    composite.stimulusSizes                 = base.stimulusSizes;
    composite.stimulusModes                 = base.stimulusModes;
    composite.conditions                    = base.conditions;
    composite.config                        = base.config;
    if isfield(base, 'stimulusBiasValues')
        composite.stimulusBiasValues = base.stimulusBiasValues;
    end

    % Now override per-condition entries with each condition's correct
    % mode cache.
    durKeys = fieldnames(composite.AllDurationGating);
    for ci = 1:numel(condList)
        cond = condList{ci};
        modeName = modeNameByCond.(cond);
        cachePath = perModeCachePath(opts.PerturbAnalysisRoot, modeName);
        if ~exist(cachePath, 'file')
            warning('Per-mode cache missing for %s (%s): %s', cond, modeName, cachePath);
            continue;
        end
        fprintf('  loading %-10s cache: %s\n', cond, cachePath);
        cur = load(cachePath, 'AllDurationGating', 'AllDurationMetrics', ...
            'AllDurationStats', 'AllDurationGating_BM', ...
            'AllDurationMetrics_BM', 'AllDurationStats_BM');
        for dk = 1:numel(durKeys)
            durKey = durKeys{dk};
            for fld = ["AllDurationGating", "AllDurationMetrics", "AllDurationStats", ...
                       "AllDurationGating_BM", "AllDurationMetrics_BM", "AllDurationStats_BM"]
                fldName = char(fld);
                if isfield(cur, fldName) && isfield(cur.(fldName), durKey) && ...
                        isfield(cur.(fldName).(durKey), cond)
                    composite.(fldName).(durKey).(cond) = cur.(fldName).(durKey).(cond);
                end
            end
        end
    end

    %% --- 3. Save composite cache as AllDurationAnalysis.mat -------------
    % unpack composite into distinct variables for save() (figuresOnly path
    % uses bare load() which restores them into workspace).
    AllDurationMetrics            = composite.AllDurationMetrics;
    AllDurationGating             = composite.AllDurationGating;
    AllDurationStats              = composite.AllDurationStats;
    AllDurationPreStimEffects     = composite.AllDurationPreStimEffects;
    AllDurationPropDynamics       = composite.AllDurationPropDynamics;
    AllDurationBlobInteractions   = composite.AllDurationBlobInteractions;
    AllDurationMoransIDynamics    = composite.AllDurationMoransIDynamics;
    AllDurationMetrics_BM         = composite.AllDurationMetrics_BM;
    AllDurationGating_BM          = composite.AllDurationGating_BM;
    AllDurationStats_BM           = composite.AllDurationStats_BM;
    stimulusDurations             = composite.stimulusDurations;
    stimulusSizes                 = composite.stimulusSizes;
    stimulusModes                 = composite.stimulusModes;
    conditions                    = composite.conditions;
    if isfield(composite, 'stimulusBiasValues')
        stimulusBiasValues = composite.stimulusBiasValues;
    end
    config = composite.config; %#ok<NASGU>

    % Figure5_IsingPerturbationAnalysis with isBiasEncodedMode appends the
    % stimMode segment to outputPath. Save composite into that subfolder so
    % figuresOnly finds it directly.
    summaryDir = fullfile(opts.OutputDir, 'double_pulse_bias10_2p00');
    if ~exist(summaryDir, 'dir'), mkdir(summaryDir); end
    summaryFile = fullfile(summaryDir, 'AllDurationAnalysis.mat');
    fprintf('Saving composite cache: %s\n', summaryFile);
    save(summaryFile, '-v7.3', ...
        'AllDurationMetrics', 'AllDurationGating', 'AllDurationStats', ...
        'AllDurationPreStimEffects', 'AllDurationPropDynamics', ...
        'AllDurationBlobInteractions', 'AllDurationMoransIDynamics', ...
        'AllDurationMetrics_BM', 'AllDurationGating_BM', 'AllDurationStats_BM', ...
        'stimulusDurations', 'stimulusSizes', 'stimulusModes', 'conditions', ...
        'config');
    if exist('stimulusBiasValues', 'var')
        save(summaryFile, 'stimulusBiasValues', '-append');
    end

    %% --- 4. Run Figure5_IsingPerturbationAnalysis in figuresOnly mode ---
    % stimMode must be bias-ENCODED (with `_<bv>` suffix) so the figuresOnly
    % branch in Figure5_IsingPerturbationAnalysis.m takes the
    % isBiasEncodedMode path: rawMode='double_pulse_bias10' →
    % metricsKey='doublepulsebias10'. The plain bias mode 'double_pulse_bias10'
    % (no encoded suffix) would instead trigger high/low collapse and look
    % for highdoublepulsebias10 / lowdoublepulsebias10 keys that don't exist
    % in per-mode caches. The actual bias value we encode here is irrelevant
    % to plot rendering — it just selects the right key-resolution branch.
    config = struct(); %#ok<NASGU>
    config = struct( ...
        'figuresOnly', true, ...
        'stimMode',    'double_pulse_bias10_2p00', ...
        'outputPath',  opts.OutputDir, ...
        'targetDurationSec',          opts.TargetDurationSec, ...
        'collapseTargetDurationsSec', opts.CollapseTargetDurationsSec); %#ok<NASGU>

    thisDir = fileparts(mfilename('fullpath'));
    if ~contains(path, thisDir), addpath(thisDir); end
    try
        Figure5_IsingPerturbationAnalysis;
    catch ME
        % figuresOnly mode generates many figures; if a downstream one
        % crashes we still want to keep the EC50 panels that landed before.
        warning('Figure5_IsingPerturbationAnalysis raised: %s\n%s', ...
            ME.message, getReport(ME, 'extended', 'hyperlinks', 'off'));
    end

    out = struct();
    out.opts          = opts;
    out.bestBiasByCond = bestBiasByCond;
    out.modeNameByCond = modeNameByCond;
    out.summaryFile   = summaryFile;
    out.outputDir     = opts.OutputDir;
end


%% ====================================================================
function lab = biasLabel(bv)
    if isinf(bv)
        lab = 'clamped';
    else
        lab = sprintf('%.2f', bv);
    end
end


%% ====================================================================
function p = perModeCachePath(perturbRoot, modeName)
    p = fullfile(perturbRoot, modeName, 'Analysis', 'AllDurationAnalysis.mat');
end


%% ====================================================================
function v = safe_get(s, fld)
    if isfield(s, fld), v = s.(fld); else, v = struct(); end
end
