function out = analyze_ExperimentalData_FractionActive(varargin)
%ANALYZE_EXPERIMENTALDATA_FRACTIONACTIVE  Reproduce the Python pipeline's
%   fraction-active metric directly from ExperimentalData.mat.
%
%   The Python figure exp_vs_ising_top_matches_per_animal_masked reads
%   BinarisedData.<cond> + RecordingMetadata.<cond> from
%   Fig. 5 Ising Models\ExperimentalData.mat (produced by
%   Figure5_dataAggregation.m), splits trials by recording, masks each
%   recording with rec_mask = np.any(rec_slice > 0, axis=(0,1)), and
%   averages.
%
%   This function does the exact same thing in MATLAB. Use it to
%   distinguish data-source discrepancies (different inputs into the two
%   pipelines) from metric discrepancies (different averaging / masking).
%   If this script's output equals the Python Activity_mean_masked per
%   condition but the swarm bar from
%   plot_FractionActive_PerRecording_Swarm does not, the gap is in the
%   swarm's input pipeline (workspace Skip, trial-type selection,
%   AnimalID-vs-AnimalName grouping) — not in the metric.
%
%   Name-Value Parameters:
%     'DataFile'   path to ExperimentalData.mat
%                  (default: 'Fig. 5 Ising Models\ExperimentalData.mat')
%     'RecFile'    path to RawData*.mat that holds the Rec table used for
%                  AnimalID lookup
%                 
%     'Rec'        optional Rec struct/table (skips loading from RecFile if
%                  supplied). When omitted the function loads Rec from
%                  RecFile.
%     'FrameRange' string OR cell array of strings. Each entry is one of:
%                    'prestim'  frames 1:80
%                    'stim'     frames 81:100
%                    'nostim'   1:80 ∪ 101:180  (mirrors apply_nostim_selection
%                               in Figure5_IsingComparison_optimized.py)
%                    'full'     1:185 (full trial; alias 'all' kept for
%                               backward compatibility)
%                  default: 'prestim'.
%                  Pass a cell to compute multiple modes in one call —
%                  results land at out.byMode.<mode>.<field>. With a single
%                  mode the fields are also promoted to top level for
%                  backward compatibility.
%     'MaskType'   'activity' (mirror Python) | 'none'
%                  default: 'activity'
%     'AnimalKey'  'AnimalID' (default — matches plot_FractionActive_*) |
%                  'AnimalName' (matches the strings stored inside
%                  RecordingMetadata.animal_names by Figure5_dataAggregation.m).
%                  AnimalID is looked up via Rec.<cond>.AnimalID indexed by
%                  RecordingMetadata.<cond>.recording_indices.
%     'Conditions' default: {'Naive','Beginner','Expert','NoSpout'}
%     'Plot'       logical, default true
%
%   Returns a struct with per-recording values, per-animal values, group
%   labels, and a summary table.

    %% Parse inputs
    p = inputParser;
    addParameter(p, 'DataFile', 'Fig. 5 Ising Models\ExperimentalData.mat', @ischar);
    addParameter(p, 'RecFile',  mba_p('correctRec.mat'), @ischar);
    addParameter(p, 'Rec',      [], @(x) isempty(x) || isstruct(x));
    addParameter(p, 'FrameRange', 'prestim', @validFrameRange);
    addParameter(p, 'MaskType', 'activity', ...
        @(x) ischar(x) && any(strcmpi(x, {'activity','none'})));
    addParameter(p, 'AnimalKey', 'AnimalID', ...
        @(x) ischar(x) && any(strcmpi(x, {'AnimalID','AnimalName'})));
    addParameter(p, 'Conditions', {'Naive','Beginner','Expert','NoSpout'}, @iscell);
    addParameter(p, 'Plot', true, @islogical);
    parse(p, varargin{:});
    opts = p.Results;

    modes      = resolveModes(opts.FrameRange);
    maskType   = lower(opts.MaskType);
    animalKey  = opts.AnimalKey;   % preserve case for field-name lookup

    %% Load
    fprintf('Loading %s\n', opts.DataFile);
    if ~exist(opts.DataFile, 'file')
        error('Data file not found: %s', opts.DataFile);
    end
    S = load(opts.DataFile, 'BinarisedData', 'RecordingMetadata', 'TimingInfo');
    BinarisedData     = S.BinarisedData;
    RecordingMetadata = S.RecordingMetadata;
    TimingInfo        = S.TimingInfo;

    if strcmpi(animalKey, 'AnimalID')
        if ~isempty(opts.Rec)
            Rec = opts.Rec;
            fprintf('Using Rec passed in via parameter (AnimalID lookup)\n');
        else
            if ~exist(opts.RecFile, 'file')
                error(['AnimalKey=AnimalID requires Rec. Pass it via the ''Rec'' ' ...
                       'parameter or set ''RecFile'' to a valid path. Got: %s'], ...
                       opts.RecFile);
            end
            fprintf('Loading Rec from %s\n', opts.RecFile);
            R = load(opts.RecFile, 'Rec');
            Rec = R.Rec;
        end
    else
        Rec = [];
    end

    fprintf('Modes      : %s\n', strjoin(modes, ', '));
    fprintf('Mask type  : %s\n', maskType);
    fprintf('Animal key : %s\n\n', animalKey);

    out = struct();
    out.opts   = opts;
    out.modes  = modes;
    out.byMode = struct();

    for mi = 1:numel(modes)
        frameRange = modes{mi};
        frames = framesForMode(frameRange, TimingInfo);
        fprintf('\n=== Mode: %s — frames [%s] (n=%d) ===\n', ...
            frameRange, frameRangeStr(frames), numel(frames));

    %% Per-recording loop
    allRec   = [];
    allCond  = {};
    allAnim  = {};
    allLabel = {};

    fprintf('%-10s %-4s %-6s %-14s %-10s %s\n', ...
        'Cond','rec','nTrl','animal','fracAct','validPx');
    fprintf('%s\n', repmat('-', 1, 64));

    for c = 1:numel(opts.Conditions)
        cond = opts.Conditions{c};
        if ~isfield(BinarisedData, cond) || isempty(BinarisedData.(cond))
            warning('Condition %s missing from BinarisedData; skipping', cond);
            continue;
        end
        if ~isfield(RecordingMetadata, cond)
            warning('Condition %s missing from RecordingMetadata; skipping', cond);
            continue;
        end

        bin  = BinarisedData.(cond);                 % [gridY, gridX, 185, nTrialsTot]
        meta = RecordingMetadata.(cond);
        nTpr = double(meta.nTrials_per_recording(:)');

        % Original recording indices in Rec/Grid40 order, used for
        % AnimalID lookup. Falls back to 1:nRec if missing (older files).
        if isfield(meta, 'recording_indices') && ~isempty(meta.recording_indices)
            recIdxList = double(meta.recording_indices(:)');
        else
            recIdxList = 1:numel(nTpr);
        end

        % Stored animal_names (always present — Figure5_dataAggregation.m
        % defaults to AnimalName, so use as fallback for AnimalKey='AnimalName').
        storedNames = meta.animal_names;
        if isstring(storedNames),  storedNames = cellstr(storedNames);  end
        if iscell(storedNames) && size(storedNames,1) > 1
            storedNames = storedNames(:)';
        end

        gridY = size(bin,1); gridX = size(bin,2);
        nRec  = numel(nTpr);
        cursor = 0;

        for r = 1:nRec
            n = nTpr(r);
            if n <= 0
                cursor = cursor + max(n,0);
                continue;
            end
            recBin = bin(:, :, frames, cursor + (1:n));   % [gY, gX, |frames|, n]
            cursor = cursor + n;

            switch maskType
                case 'activity'
                    recMask = any(any(recBin > 0, 3), 4);
                    recMask = reshape(recMask, gridY, gridX);
                case 'none'
                    recMask = true(gridY, gridX);
            end
            nValidPx = nnz(recMask);
            if nValidPx == 0
                continue;
            end

            recFlat = reshape(recBin, [], size(recBin,3), size(recBin,4));
            recFlat = recFlat(recMask(:), :, :);
            recVal  = mean(recFlat, 'all');

            origIdx = recIdxList(min(r, numel(recIdxList)));
            animalID = lookupAnimal(animalKey, Rec, cond, origIdx, storedNames, r);

            allRec(end+1, 1)   = recVal;             %#ok<AGROW>
            allCond{end+1, 1}  = cond;               %#ok<AGROW>
            allAnim{end+1, 1}  = animalID;           %#ok<AGROW>
            allLabel{end+1, 1} = sprintf('%s#r%d', cond, origIdx); %#ok<AGROW>

            fprintf('%-10s %-4d %-6d %-14s %-10.4f %d/%d\n', ...
                cond, origIdx, n, animalID, recVal, nValidPx, gridY*gridX);
        end
    end

    %% Per-animal aggregation
    [animalVals, animalCond, animalNames] = aggregateByAnimal(allRec, allCond, allAnim);

    %% Summary table
    uniqueConds = unique(allCond, 'stable');
    fprintf('\n=== SUMMARY  (frameRange=%s, mask=%s) ===\n', frameRange, maskType);
    fprintf('%-10s %-7s %-7s %-7s %-9s %-9s %-9s %-9s\n', ...
        'Cond','nRec','nAnim','nTrls','recMean','recStd','animMean','animStd');
    fprintf('%s\n', repmat('-', 1, 76));
    summary = struct();
    for ci = 1:numel(uniqueConds)
        cond = uniqueConds{ci};
        rm = strcmp(allCond, cond);
        am = strcmp(animalCond, cond);
        meta = RecordingMetadata.(cond);
        summary.(cond).n_recordings = sum(rm);
        summary.(cond).n_animals    = sum(am);
        summary.(cond).rec_mean     = mean(allRec(rm),    'omitnan');
        summary.(cond).rec_std      = std( allRec(rm),    'omitnan');
        summary.(cond).animal_mean  = mean(animalVals(am),'omitnan');
        summary.(cond).animal_std   = std( animalVals(am),'omitnan');
        summary.(cond).n_trials_total = sum(double(meta.nTrials_per_recording(:)));
        fprintf('%-10s %-7d %-7d %-7d %-9.4f %-9.4f %-9.4f %-9.4f\n', cond, ...
            summary.(cond).n_recordings, summary.(cond).n_animals, ...
            summary.(cond).n_trials_total, ...
            summary.(cond).rec_mean,    summary.(cond).rec_std, ...
            summary.(cond).animal_mean, summary.(cond).animal_std);
    end

    %% Per-mode output
    modeOut = struct();
    modeOut.frames        = frames;
    modeOut.frameRange    = frameRange;
    modeOut.rec_values    = allRec;
    modeOut.rec_cond      = allCond;
    modeOut.rec_animal    = allAnim;
    modeOut.rec_label     = allLabel;
    modeOut.animal_values = animalVals;
    modeOut.animal_cond   = animalCond;
    modeOut.animal_names  = animalNames;
    modeOut.summary       = summary;
    out.byMode.(frameRange) = modeOut;

    %% Plot
    if opts.Plot
        figure('Name', sprintf('ExperimentalData fracActive (mask=%s, frames=%s)', ...
            maskType, frameRange), 'Color', 'w');
        subplot(1,2,1);
        plotSwarm(allRec, allCond, uniqueConds, ...
            sprintf('per recording  (%s, mask=%s)', frameRange, maskType));
        subplot(1,2,2);
        plotSwarm(animalVals, animalCond, uniqueConds, ...
            sprintf('per animal  (%s, mask=%s)', frameRange, maskType));
    end
    end  % per-mode loop

    %% Backward-compat: with one mode, promote per-mode fields to top level
    if numel(modes) == 1
        modeFields = fieldnames(out.byMode.(modes{1}));
        for fi = 1:numel(modeFields)
            out.(modeFields{fi}) = out.byMode.(modes{1}).(modeFields{fi});
        end
    end
end


%% ========================================================================
function tf = validFrameRange(x)
    valid = {'prestim','stim','nostim','full','all'};
    if ischar(x) || (isstring(x) && isscalar(x))
        tf = any(strcmpi(char(x), valid));
    elseif iscell(x)
        tf = ~isempty(x) && all(cellfun(@(m) (ischar(m) || (isstring(m) && isscalar(m))) ...
            && any(strcmpi(char(m), valid)), x));
    else
        tf = false;
    end
end


%% ========================================================================
function modes = resolveModes(frameRange)
    if iscell(frameRange)
        modes = cellfun(@(m) lower(char(m)), frameRange, 'UniformOutput', false);
    else
        modes = {lower(char(frameRange))};
    end
    % Normalize 'all' -> 'full'
    modes = cellfun(@(m) ternary(strcmp(m,'all'), 'full', m), modes, 'UniformOutput', false);
    % De-duplicate while preserving order
    [~, ia] = unique(modes, 'stable');
    modes = modes(ia);
end


%% ========================================================================
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end


%% ========================================================================
function frames = framesForMode(mode, TimingInfo)
%FRAMESFORMODE  Resolve frame indices for a named mode.
%   prestim : TimingInfo.trial_structure.prestim_frames    (default 1:80)
%   stim    : TimingInfo.trial_structure.stimulus_frames   (default 81:100)
%   nostim  : prestim ∪ post-stim (post = next prestim_len frames after stim)
%             Mirrors apply_nostim_selection in
%             Figure5_IsingComparison_optimized.py:567-600.
%   full    : 1:total_frames                               (default 1:185)
    if isempty(fieldnames(TimingInfo))
        prestim = 1:80;
        stim    = 81:100;
        total   = 185;
    else
        prestim = TimingInfo.trial_structure.prestim_frames;
        stim    = TimingInfo.trial_structure.stimulus_frames;
        total   = TimingInfo.trial_structure.total_frames;
    end
    prestim = prestim(:)';
    stim    = stim(:)';
    stimEnd = stim(end);
    prestimLen = numel(prestim);
    post = (stimEnd+1):min(stimEnd+prestimLen, total);
    switch lower(mode)
        case 'prestim', frames = prestim;
        case 'stim',    frames = stim;
        case 'nostim',  frames = unique([prestim, post]);
        case {'full','all'}, frames = 1:total;
        otherwise
            error('Unknown frame mode: %s', mode);
    end
end


%% ========================================================================
function s = frameRangeStr(frames)
%FRAMERANGESTR  Compact summary of (possibly disjoint) frame indices.
    if isempty(frames), s = '<empty>'; return; end
    d = diff(frames);
    breaks = [0, find(d ~= 1), numel(frames)];
    parts = strings(1, numel(breaks)-1);
    for k = 1:numel(breaks)-1
        seg = frames(breaks(k)+1 : breaks(k+1));
        if numel(seg) == 1
            parts(k) = sprintf('%d', seg);
        else
            parts(k) = sprintf('%d:%d', seg(1), seg(end));
        end
    end
    s = char(strjoin(parts, ', '));
end


%% ========================================================================
function [animalVals, animalCond, animalNames] = aggregateByAnimal(recVals, recCond, recAnim)
    animalVals  = [];
    animalCond  = {};
    animalNames = {};
    uniqueConds = unique(recCond, 'stable');
    for ci = 1:numel(uniqueConds)
        cond = uniqueConds{ci};
        cmask = strcmp(recCond, cond);
        condRec  = recVals(cmask);
        condAnim = recAnim(cmask);
        validMask = ~cellfun(@isempty, condAnim);
        if ~any(validMask), continue; end
        uA = unique(condAnim(validMask), 'stable');
        for ai = 1:numel(uA)
            am = strcmp(condAnim, uA{ai});
            animalVals(end+1, 1)  = mean(condRec(am), 'omitnan'); %#ok<AGROW>
            animalCond{end+1, 1}  = cond;                          %#ok<AGROW>
            animalNames{end+1, 1} = uA{ai};                        %#ok<AGROW>
        end
    end
end


%% ========================================================================
function animalID = lookupAnimal(animalKey, Rec, cond, origIdx, storedNames, r)
%LOOKUPANIMAL  Resolve an animal identifier for one recording.
%   animalKey='AnimalID'   : Rec.<cond>.AnimalID{origIdx} (table or struct).
%   animalKey='AnimalName' : storedNames{r} as written by
%                            Figure5_dataAggregation.m.
%   Falls back to '' on missing fields.
    animalID = '';
    if strcmpi(animalKey, 'AnimalID')
        if isempty(Rec) || ~isfield(Rec, cond)
            return;
        end
        recCond = Rec.(cond);
        if istable(recCond)
            if ~ismember('AnimalID', recCond.Properties.VariableNames), return; end
            if origIdx > height(recCond), return; end
            raw = recCond.AnimalID(origIdx);
        elseif isstruct(recCond)
            if ~isfield(recCond, 'AnimalID'), return; end
            ids = recCond.AnimalID;
            if iscell(ids)
                if origIdx > numel(ids), return; end
                raw = ids{origIdx};
            else
                if origIdx > size(ids,1), return; end
                raw = ids(origIdx);
            end
        else
            return;
        end
        if iscell(raw) && ~isempty(raw), raw = raw{1}; end
        if isstring(raw) || ischar(raw)
            animalID = strtrim(char(raw));
        elseif isnumeric(raw)
            animalID = num2str(raw);
        end
    else  % AnimalName -- use stored names
        if iscell(storedNames) && r <= numel(storedNames) && ~isempty(storedNames{r})
            animalID = strtrim(char(storedNames{r}));
        end
    end
end


%% ========================================================================
function plotSwarm(values, groups, uniqueG, titleStr)
    nG = numel(uniqueG);
    hold on;
    for gi = 1:nG
        m = strcmp(groups, uniqueG{gi});
        v = values(m);
        if isempty(v), continue; end
        bar(gi, mean(v, 'omitnan'), 0.6, ...
            'FaceColor', [0.7 0.78 0.95], 'EdgeColor', 'none');
        swarmchart(gi*ones(numel(v),1), v, 60, 'filled', ...
            'MarkerFaceColor', 'k', 'MarkerFaceAlpha', 0.7, ...
            'XJitterWidth', 0.4);
    end
    xticks(1:nG); xticklabels(uniqueG); xtickangle(30);
    ylabel('Fraction active');
    title(titleStr);
    set(gca, 'FontSize', 10);
    grid on; box on;
    hold off;
end
