function T = compare_FractionActive_inputs(varargin)
%COMPARE_FRACTIONACTIVE_INPUTS  Localize per-recording fracActive divergence.
%   Side-by-side per-recording diagnostic of three pipelines that all aim
%   to compute "fraction active" for the same recordings:
%     * frac_swarm           - reproduces plot_FractionActive_PerRecording_Swarm
%                              (uses Grid40.<cond>Individual.AllNeurons(r).All,
%                               Performance/Stimuli filtering, RT filter, mask)
%     * frac_agg_recompute   - reproduces Figure5_dataAggregation.m on the
%                              CURRENT workspace Grid40 (uses .P1 directly)
%     * frac_agg_cached      - reads BinarisedData/RecordingMetadata from
%                              ExperimentalData.mat (the file the Python
%                              pipeline ingests)
%
%   Use: same metric + same mask + different output ⇒ inputs differ. This
%   script prints every relevant intermediate count (Skip membership in
%   each pipeline, nAll/nP1, hit/miss counts, AnimalID vs AnimalName,
%   plus the three frac_* values) so you can pinpoint which input flips
%   between rows.
%
%   Name-Value Parameters:
%     'Grid40','Performance','Stimuli','Rec','Skip' - workspace structs.
%         If any are [], they are loaded from 'RawDataFile' / 'RecFile' /
%         'Grid40File'. Pass the workspace ones explicitly for speed.
%     'Grid40File'  default mba_p('Grid40.mat')
%     'RawDataFile' default mba_p('RawData3.mat')    (Performance/Stimuli/Skip)
%     'RecFile'     default mba_p('correctRec.mat')  (matches
%                                                analyze_ExperimentalData_FractionActive)
%     'DataFile'    default 'Fig. 5 Ising Models\ExperimentalData.mat'
%     'FrameRange'  'prestim' (default) | 'stim' | 'all'
%     'MaskType'    'activity' (default) | 'anatomical' | 'none'
%     'TrialType'   per-condition cell, default {'all','all','all','all'}
%     'BinarisationThreshold' default 2
%     'DisplayWindow' [-1 3] default — RT window for active conditions
%     'Conditions'  default {'Naive','Beginner','Expert','NoSpout'}
%     'HardcodedSkip' struct, defaults to the values in
%                     Figure5_dataAggregation.m:70-73
%     'CSV'         optional path; if non-empty, writetable(T,CSV)
%
%   Returns a MATLAB table T (one row per condition × recording).

    %% Parse inputs
    p = inputParser;
    addParameter(p, 'Grid40',      [], @(x) isempty(x) || isstruct(x));
    addParameter(p, 'Performance', [], @(x) isempty(x) || isstruct(x));
    addParameter(p, 'Stimuli',     [], @(x) isempty(x) || isstruct(x));
    addParameter(p, 'Rec',         [], @(x) isempty(x) || isstruct(x));
    addParameter(p, 'Skip',        [], @(x) isempty(x) || isstruct(x));
    addParameter(p, 'Grid40File',  mba_p('Grid40.mat'), @ischar);
    addParameter(p, 'RawDataFile', mba_p('RawData3.mat'), @ischar);
    addParameter(p, 'RecFile',     mba_p('correctRec.mat'), @ischar);
    addParameter(p, 'DataFile',    'Fig. 5 Ising Models\ExperimentalData.mat', @ischar);
    addParameter(p, 'FrameRange',  'prestim', ...
        @(x) ischar(x) && any(strcmpi(x, {'prestim','stim','all'})));
    addParameter(p, 'MaskType',    'activity', ...
        @(x) ischar(x) && any(strcmpi(x, {'activity','anatomical','none'})));
    addParameter(p, 'TrialType',   {'all','all','all','all'}, @iscell);
    addParameter(p, 'BinarisationThreshold', 2, @isnumeric);
    addParameter(p, 'DisplayWindow', [-1, 3], @isnumeric);
    addParameter(p, 'Conditions',  {'Naive','Beginner','Expert','NoSpout'}, @iscell);
    addParameter(p, 'HardcodedSkip', defaultHardcodedSkip(), @isstruct);
    addParameter(p, 'CSV', '', @ischar);
    parse(p, varargin{:});
    opts = p.Results;

    frameRange = lower(opts.FrameRange);
    maskType   = lower(opts.MaskType);
    binThresh  = opts.BinarisationThreshold;
    displayWin = opts.DisplayWindow;
    conds      = opts.Conditions;
    trialTypes = opts.TrialType;
    if numel(trialTypes) < numel(conds)
        trialTypes = [trialTypes, repmat({'all'}, 1, numel(conds)-numel(trialTypes))];
    end

    %% Resolve workspace structs (load only what's missing)
    Grid40      = ensureStruct(opts.Grid40,      opts.Grid40File,  'Grid40');
    Performance = ensureStruct(opts.Performance, opts.RawDataFile, 'Performance');
    Stimuli     = ensureStruct(opts.Stimuli,     opts.RawDataFile, 'Stimuli');
    Skip        = ensureStruct(opts.Skip,        opts.RawDataFile, 'Skip');
    Rec         = ensureStruct(opts.Rec,         opts.RecFile,     'Rec');

    %% Load cached BinarisedData + RecordingMetadata + TimingInfo
    fprintf('Loading cached data from %s\n', opts.DataFile);
    if exist(opts.DataFile, 'file')
        S = load(opts.DataFile, 'BinarisedData', 'RecordingMetadata', 'TimingInfo');
        BinarisedData     = S.BinarisedData;
        RecordingMetadata = S.RecordingMetadata;
        TimingInfo        = S.TimingInfo;
    else
        warning('DataFile not found: %s — frac_agg_cached will be NaN', opts.DataFile);
        BinarisedData     = struct();
        RecordingMetadata = struct();
        TimingInfo        = struct();
    end

    % Frame window
    switch frameRange
        case 'prestim'
            if ~isempty(fieldnames(TimingInfo))
                frames = TimingInfo.trial_structure.prestim_frames;
            else
                frames = 1:80;
            end
        case 'stim'
            if ~isempty(fieldnames(TimingInfo))
                frames = TimingInfo.trial_structure.stimulus_frames;
            else
                frames = 81:100;
            end
        case 'all'
            if ~isempty(fieldnames(TimingInfo))
                frames = 1:TimingInfo.trial_structure.total_frames;
            else
                frames = 1:185;
            end
    end
    fprintf('Frame range: %s = frames %d:%d (n=%d)\n', ...
        frameRange, frames(1), frames(end), numel(frames));
    fprintf('Mask type  : %s\n', maskType);
    fprintf('TrialType  : %s\n', strjoin(trialTypes(1:numel(conds)), ', '));
    fprintf('\n');

    %% Per-condition / per-recording loop
    rows = struct('cond', {}, 'r', {}, 'inSkipWS', {}, 'inSkipAgg', {}, ...
        'nAll', {}, 'nP1', {}, 'nP1capAll', {}, ...
        'nHit', {}, 'nMiss', {}, 'nMissRT', {}, 'nSelSwarm', {}, ...
        'animalID', {}, 'animalName', {}, ...
        'frac_swarm', {}, 'frac_agg_recompute', {}, 'frac_agg_cached', {});

    for c = 1:numel(conds)
        cond  = conds{c};
        ttype = trialTypes{c};
        condInd  = [cond 'Individual'];
        baseCond = cond;
        isPassive = contains(baseCond, 'NoSpout') || contains(baseCond, 'Naive');

        if ~isfield(Grid40, condInd)
            warning('Grid40 missing %s — skipping condition', condInd);
            continue;
        end
        recList = Grid40.(condInd).AllNeurons;
        nRecs = numel(recList);

        skipWS  = ws_skip_set(Skip, cond);
        skipAgg = ws_skip_set(opts.HardcodedSkip, cond);

        % Pre-compute the cache lookup table for this condition once
        if isfield(BinarisedData, cond) && ~isempty(BinarisedData.(cond)) ...
                && isfield(RecordingMetadata, cond)
            cache = struct();
            cache.bin       = BinarisedData.(cond);
            cache.nTpr      = double(RecordingMetadata.(cond).nTrials_per_recording(:)');
            if isfield(RecordingMetadata.(cond), 'recording_indices') ...
                    && ~isempty(RecordingMetadata.(cond).recording_indices)
                cache.recIdx = double(RecordingMetadata.(cond).recording_indices(:)');
            else
                cache.recIdx = 1:numel(cache.nTpr);
            end
            cache.cumsum = [0, cumsum(cache.nTpr)];
            cache.gridY  = size(cache.bin, 1);
            cache.gridX  = size(cache.bin, 2);
        else
            cache = [];
        end

        for r = 1:nRecs
            row = blankRow(cond, r);
            row.inSkipWS  = ismember(r, skipWS);
            row.inSkipAgg = ismember(r, skipAgg);

            % Animal IDs
            row.animalID   = lookupAnimalField(Rec, cond, r, 'AnimalID');
            row.animalName = lookupAnimalField(Rec, cond, r, 'AnimalName');

            % Grid40 .All / .P1
            recEntry = recList(r);
            allArr = derefMaybeCell(getfieldOrEmpty(recEntry, 'All'));
            p1Arr  = derefMaybeCell(getfieldOrEmpty(recEntry, 'P1'));
            if ~isempty(allArr) && ndims(allArr) == 4
                row.nAll = size(allArr, 4);
            end
            if ~isempty(p1Arr) && ndims(p1Arr) == 4
                row.nP1 = size(p1Arr, 4);
            end

            % Stimuli P1 indices (absolute into .All)
            p1AbsIdx = [];
            if isfield(Stimuli, baseCond) && r <= numel(Stimuli.(baseCond)) ...
                    && isfield(Stimuli.(baseCond)(r), 'TrialsPosition1')
                p1AbsIdx = Stimuli.(baseCond)(r).TrialsPosition1(:)';
            end
            if ~isnan(row.nAll)
                row.nP1capAll = numel(intersect(1:row.nAll, p1AbsIdx));
            end

            % Performance counts
            if ~isPassive && isfield(Performance, baseCond) ...
                    && r <= numel(Performance.(baseCond))
                pf = Performance.(baseCond)(r);
                if isfield(pf, 'hit'),  row.nHit  = numel(pf.hit);  end
                if isfield(pf, 'miss'), row.nMiss = numel(pf.miss); end
                if isfield(pf, 'miss') && isfield(pf, 'FirstLicks10') ...
                        && ~isempty(pf.FirstLicks10)
                    rts = pf.FirstLicks10;
                    cnt = 0;
                    for ti = 1:numel(pf.miss)
                        t = pf.miss(ti);
                        if t > numel(rts), continue; end
                        rt = rts(t);
                        if ~isnan(rt) && rt > 2, cnt = cnt + 1; end
                    end
                    row.nMissRT = cnt;
                end
            end

            % --- frac_swarm: reproduce plot_FractionActive_PerRecording_Swarm
            row.frac_swarm = computeSwarmFrac(allArr, recEntry, baseCond, ...
                isPassive, ttype, p1AbsIdx, ...
                Performance, r, frames, binThresh, displayWin, maskType);
            % nSelSwarm is filled inside computeSwarmFrac via shared global —
            % we instead compute it cheaply here for the print-out
            row.nSelSwarm = countSwarmSelected(row.nAll, isPassive, ttype, ...
                Performance, baseCond, r, p1AbsIdx, displayWin);

            % --- frac_agg_recompute: reproduce Figure5_dataAggregation on workspace .P1
            row.frac_agg_recompute = computeAggFrac(p1Arr, frames, binThresh, ...
                maskType, recEntry);

            % --- frac_agg_cached: read from ExperimentalData.mat
            if ~isempty(cache)
                idx = find(cache.recIdx == r, 1);
                if ~isempty(idx)
                    s = cache.cumsum(idx);
                    n = cache.nTpr(idx);
                    if n > 0
                        recBin = cache.bin(:, :, frames, s + (1:n));
                        recMask = buildCacheMask(recBin, recEntry, ...
                            cache.gridY, cache.gridX, maskType);
                        if any(recMask, 'all')
                            recFlat = reshape(recBin, [], size(recBin,3), size(recBin,4));
                            recFlat = recFlat(recMask(:), :, :);
                            row.frac_agg_cached = mean(recFlat, 'all');
                        end
                    end
                end
            end

            rows(end+1) = row; %#ok<AGROW>
        end
    end

    %% Build table
    T = struct2table(rows);

    %% Print grouped table to stdout
    printGrouped(T);

    %% Summary
    printSummary(T);

    %% Optional CSV
    if ~isempty(opts.CSV)
        writetable(T, opts.CSV);
        fprintf('\nWrote %s\n', opts.CSV);
    end
end


%% ========================================================================
function S = ensureStruct(provided, file, varName)
    if ~isempty(provided)
        S = provided;
        return;
    end
    if ~exist(file, 'file')
        error('%s not provided and file not found: %s', varName, file);
    end
    fprintf('Loading %s from %s\n', varName, file);
    L = load(file, varName);
    S = L.(varName);
end


%% ========================================================================
function s = ws_skip_set(SkipStruct, cond)
    s = [];
    if isstruct(SkipStruct) && isfield(SkipStruct, cond)
        s = SkipStruct.(cond);
    end
end


%% ========================================================================
function row = blankRow(cond, r)
    row = struct( ...
        'cond',              cond, ...
        'r',                 r, ...
        'inSkipWS',          false, ...
        'inSkipAgg',         false, ...
        'nAll',              NaN, ...
        'nP1',               NaN, ...
        'nP1capAll',         NaN, ...
        'nHit',              NaN, ...
        'nMiss',             NaN, ...
        'nMissRT',           NaN, ...
        'nSelSwarm',         NaN, ...
        'animalID',          "", ...
        'animalName',        "", ...
        'frac_swarm',        NaN, ...
        'frac_agg_recompute',NaN, ...
        'frac_agg_cached',   NaN);
end


%% ========================================================================
function v = getfieldOrEmpty(s, name)
    if isfield(s, name), v = s.(name); else, v = []; end
end


%% ========================================================================
function out = derefMaybeCell(v)
    if isempty(v), out = []; return; end
    if iscell(v)
        if isempty(v{1}), out = []; else, out = v{1}; end
    else
        out = v;
    end
end


%% ========================================================================
function id = lookupAnimalField(Rec, cond, r, fieldName)
    id = "";
    if ~isfield(Rec, cond), return; end
    rc = Rec.(cond);
    if istable(rc)
        if ~ismember(fieldName, rc.Properties.VariableNames), return; end
        if r > height(rc), return; end
        raw = rc.(fieldName)(r);
    elseif isstruct(rc)
        if ~isfield(rc, fieldName), return; end
        f = rc.(fieldName);
        if iscell(f)
            if r > numel(f), return; end
            raw = f{r};
        else
            if r > size(f,1), return; end
            raw = f(r);
        end
    else
        return;
    end
    if iscell(raw) && ~isempty(raw), raw = raw{1}; end
    if isstring(raw) || ischar(raw)
        id = string(strtrim(char(raw)));
    elseif isnumeric(raw)
        id = string(num2str(raw));
    end
end


%% ========================================================================
function frac = computeSwarmFrac(allArr, recEntry, baseCond, isPassive, ...
        ttype, p1AbsIdx, Performance, r, frames, binThresh, displayWin, maskType)
%COMPUTESWARMFRAC  Reproduce the swarm pipeline on .All for a single recording.
    frac = NaN;
    if isempty(allArr) || ndims(allArr) ~= 4, return; end
    [gY, gX, nF, nT] = size(allArr);
    frames = frames(frames <= nF);
    if isempty(frames), return; end

    selectedTrials = swarmSelected(nT, isPassive, ttype, ...
        Performance, baseCond, r, p1AbsIdx, displayWin);
    if isempty(selectedTrials), return; end

    bin = allArr(:, :, frames, selectedTrials) > binThresh;
    recMask = buildSwarmMask(bin, recEntry, gY, gX, maskType);
    if ~any(recMask, 'all'), return; end
    flat = reshape(bin, [], size(bin,3), size(bin,4));
    flat = flat(recMask(:), :, :);
    frac = mean(flat, 'all');
end


%% ========================================================================
function n = countSwarmSelected(nAll, isPassive, ttype, Performance, baseCond, r, p1AbsIdx, displayWin)
%COUNTSWARMSELECTED  Count of trials the swarm would average over.
    n = 0;
    if isnan(nAll) || nAll <= 0, return; end
    sel = swarmSelected(nAll, isPassive, ttype, Performance, baseCond, r, p1AbsIdx, displayWin);
    n = numel(sel);
end


%% ========================================================================
function selectedTrials = swarmSelected(nT, isPassive, ttype, Performance, baseCond, r, p1AbsIdx, displayWin)
    selectedTrials = [];
    if isPassive
        selectedTrials = 1:nT;
    else
        if ~isfield(Performance, baseCond) || r > numel(Performance.(baseCond)), return; end
        pf = Performance.(baseCond)(r);
        switch ttype
            case 'hit'
                if ~isfield(pf, 'hit') || isempty(pf.hit), return; end
                selectedTrials = pf.hit(:)';
            case 'miss'
                if ~isfield(pf, 'miss') || isempty(pf.miss), return; end
                selectedTrials = pf.miss(:)';
            case 'all'
                selectedTrials = 1:nT;
        end
    end
    selectedTrials = selectedTrials(selectedTrials <= nT);
    if ~isempty(p1AbsIdx)
        selectedTrials = intersect(selectedTrials, p1AbsIdx);
    end
    if isempty(selectedTrials), return; end
    if ~isPassive && isfield(Performance, baseCond) && r <= numel(Performance.(baseCond))
        pf = Performance.(baseCond)(r);
        if isfield(pf, 'FirstLicks10') && ~isempty(pf.FirstLicks10)
            rts = pf.FirstLicks10;
            keepMask = false(size(selectedTrials));
            for ti = 1:numel(selectedTrials)
                t = selectedTrials(ti);
                if t > numel(rts), continue; end
                rt = rts(t);
                if isnan(rt), continue; end
                if strcmp(ttype, 'miss')
                    if rt > 2, keepMask(ti) = true; end
                else
                    if rt > 0 && rt >= displayWin(1) && rt <= displayWin(2)
                        keepMask(ti) = true;
                    end
                end
            end
            selectedTrials = selectedTrials(keepMask);
        end
    end
end


%% ========================================================================
function frac = computeAggFrac(p1Arr, frames, binThresh, maskType, recEntry)
%COMPUTEAGGFRAC  Reproduce Figure5_dataAggregation per-recording on workspace .P1.
    frac = NaN;
    if isempty(p1Arr) || ndims(p1Arr) ~= 4, return; end
    [gY, gX, nF, nT] = size(p1Arr);  %#ok<ASGLU>
    frames = frames(frames <= nF);
    if isempty(frames) || nT == 0, return; end
    bin = p1Arr(:, :, frames, :) > binThresh;
    recMask = buildSwarmMask(bin, recEntry, gY, gX, maskType);
    if ~any(recMask, 'all'), return; end
    flat = reshape(bin, [], size(bin,3), size(bin,4));
    flat = flat(recMask(:), :, :);
    frac = mean(flat, 'all');
end


%% ========================================================================
function recMask = buildSwarmMask(bin, recEntry, gY, gX, maskType)
%BUILDSWARMMASK  Mask construction matching plot_FractionActive_PerRecording_Swarm.
    switch maskType
        case 'activity'
            recMask = reshape(any(any(bin, 3), 4), gY, gX);
        case 'anatomical'
            if isfield(recEntry, 'CellIDs') && ~isempty(recEntry.CellIDs)
                cellIDs = recEntry.CellIDs;
                recMask = ~cellfun(@isempty, cellIDs);
                if ~isequal(size(recMask), [gY, gX])
                    recMask = true(gY, gX);
                end
            else
                recMask = true(gY, gX);
            end
        case 'none'
            recMask = true(gY, gX);
    end
end


%% ========================================================================
function recMask = buildCacheMask(recBin, recEntry, gY, gX, maskType)
%BUILDCACHEMASK  Mask construction for ExperimentalData.mat-derived bins.
    switch maskType
        case 'activity'
            recMask = reshape(any(any(recBin > 0, 3), 4), gY, gX);
        case 'anatomical'
            if isfield(recEntry, 'CellIDs') && ~isempty(recEntry.CellIDs)
                cellIDs = recEntry.CellIDs;
                recMask = ~cellfun(@isempty, cellIDs);
                if ~isequal(size(recMask), [gY, gX])
                    recMask = true(gY, gX);
                end
            else
                recMask = true(gY, gX);
            end
        case 'none'
            recMask = true(gY, gX);
    end
end


%% ========================================================================
function S = defaultHardcodedSkip()
%DEFAULTHARDCODEDSKIP  Mirrors Figure5_dataAggregation.m:70-73.
    S = struct();
    S.Naive    = [1 9 10 16];
    S.Beginner = [1 6 7 11];
    S.Expert   = [1 4 12 13 14];
    S.NoSpout  = [1 4 9 10 11 13 14];
end


%% ========================================================================
function printGrouped(T)
    conds = unique(string(T.cond), 'stable');
    fprintf('%-9s %-3s %-4s %-4s %-4s %-4s %-5s %-4s %-4s %-6s %-5s %-12s %-12s %-9s %-11s %-9s\n', ...
        'cond','r','wsS','aggS','nAll','nP1','capP1','nHt','nMs','nMsRT','nSel', ...
        'animalID','animalName','frac_swrm','frac_agg_R','frac_agg_C');
    fprintf('%s\n', repmat('-', 1, 132));
    for ci = 1:numel(conds)
        cond = conds(ci);
        m = string(T.cond) == cond;
        idx = find(m);
        for k = 1:numel(idx)
            row = T(idx(k), :);
            fprintf('%-9s %-3d %-4d %-4d %-4d %-4d %-5d %-4s %-4s %-6s %-5s %-12s %-12s %-9.4f %-11.4f %-9.4f\n', ...
                row.cond{1}, row.r, ...
                row.inSkipWS, row.inSkipAgg, ...
                fmtNum(row.nAll), fmtNum(row.nP1), fmtNum(row.nP1capAll), ...
                fmtStr(fmtNum(row.nHit)), fmtStr(fmtNum(row.nMiss)), ...
                fmtStr(fmtNum(row.nMissRT)), fmtStr(fmtNum(row.nSelSwarm)), ...
                truncate(char(row.animalID), 12), truncate(char(row.animalName), 12), ...
                row.frac_swarm, row.frac_agg_recompute, row.frac_agg_cached);
        end
        fprintf('\n');
    end
end


%% ========================================================================
function v = fmtNum(x)
    if isnan(x), v = -1; else, v = x; end
end


%% ========================================================================
function s = fmtStr(v)
    if v < 0, s = '-'; else, s = sprintf('%d', v); end
end


%% ========================================================================
function s = truncate(s, n)
    if numel(s) > n, s = s(1:n); end
end


%% ========================================================================
function printSummary(T)
    fprintf('\n=== SUMMARY ===\n');
    conds = unique(string(T.cond), 'stable');
    fprintf('%-10s %-6s %-9s %-9s %-9s %-9s %-9s %-9s\n', ...
        'cond','nRec', ...
        'fr_swrm','fr_agg_R','fr_agg_C', ...
        'd(S-Ar)','d(Ar-Ac)','d(S-Ac)');
    fprintf('%s\n', repmat('-', 1, 80));
    for ci = 1:numel(conds)
        cond = conds(ci);
        m = string(T.cond) == cond & ~T.inSkipWS;   % use WS skip for swarm count
        s  = T.frac_swarm(m);
        ar = T.frac_agg_recompute(m);
        ac = T.frac_agg_cached(m);
        fprintf('%-10s %-6d %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f\n', ...
            cond, sum(m), nanmean(s), nanmean(ar), nanmean(ac), ...
            nanmean(s - ar), nanmean(ar - ac), nanmean(s - ac));
    end
end


%% ========================================================================
function m = nanmean(v)
    v = v(~isnan(v));
    if isempty(v), m = NaN; else, m = mean(v); end
end
