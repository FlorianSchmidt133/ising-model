%% compare_blob_methods_comprehensive.m
% Comprehensive comparison of blob detection methods
%
% Applies each method to BOTH BinarisedData AND Raw Grid40 data, then:
% 1. Finds which method produces the smallest difference between datasets
% 2. Identifies which method produces the longest blobs in both datasets

%% Configuration
config = struct();
config.experimentalDataPath = 'Fig. 5 Model\IsingModels\Data\ExperimentalData.mat';
config.grid40Path = mba_p('Grid40.mat');
config.condition = 'Expert';
config.maxRecordings = 5;
config.maxTrialsPerRec = 20;
config.minBlobSize = 4;
config.maxBlobSize = 1/3;  % Max blob size as fraction of grid (~33%)
config.iouThreshold = 0.3;

fprintf('=== Comprehensive Blob Detection Methods Comparison ===\n');
fprintf('Goal: Find method with smallest difference between BinarisedData and Grid40\n\n');

%% Load Data (skip if already in workspace)
fprintf('Loading data...\n');

if evalin('base', 'exist(''BinarisedData'', ''var'')')
    BinarisedData = evalin('base', 'BinarisedData');
    fprintf('  Using BinarisedData from workspace\n');
else
    expData = load(config.experimentalDataPath);
    BinarisedData = expData.BinarisedData;
    fprintf('  Loaded BinarisedData from file\n');
end

if evalin('base', 'exist(''Grid40'', ''var'')')
    Grid40 = evalin('base', 'Grid40');
    fprintf('  Using Grid40 from workspace\n');
else
    rawData = load(config.grid40Path, 'Grid40');
    Grid40 = rawData.Grid40;
    fprintf('  Loaded Grid40 from file\n');
end

%% Prepare data references
condition = config.condition;
binData = BinarisedData.(condition);
[gridY, gridX, nFrames, nTrialsTotal] = size(binData);
nTrials = min(nTrialsTotal, config.maxRecordings * config.maxTrialsPerRec);

% Get Grid40 data
conditionIndividual = [condition 'Individual'];
recList = Grid40.(conditionIndividual).AllNeurons;

fprintf('\nGrid: %d x %d, %d frames\n', gridY, gridX, nFrames);
fprintf('Processing up to %d trials per dataset\n', nTrials);

%% Define all methods to test
methods = {};

% --- Category 1: Gaussian Smoothing Variants ---
sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5];
for s = sigmas
    for t = thresholds
        methods{end+1} = struct('name', sprintf('Gauss(%.1f)+Th(%.1f)', s, t), ...
            'type', 'gaussian', 'sigma', s, 'threshold', t);
    end
end

% --- Category 2: Morphological Operations ---
for r = [1, 2, 3]
    methods{end+1} = struct('name', sprintf('Closing(r=%d)', r), ...
        'type', 'closing', 'radius', r);
end

for r = [1, 2]
    methods{end+1} = struct('name', sprintf('Opening(r=%d)', r), ...
        'type', 'opening', 'radius', r);
end

for r = [1, 2]
    methods{end+1} = struct('name', sprintf('Open+Close(r=%d)', r), ...
        'type', 'open_close', 'radius', r);
end

for r = [1, 2]
    methods{end+1} = struct('name', sprintf('Close+Open(r=%d)', r), ...
        'type', 'close_open', 'radius', r);
end

for r = [1, 2]
    methods{end+1} = struct('name', sprintf('Dilate(r=%d)', r), ...
        'type', 'dilate', 'radius', r);
end

% --- Category 3: Alternative Filters ---
methods{end+1} = struct('name', 'Direct Binary', 'type', 'direct');
methods{end+1} = struct('name', 'Median 3x3', 'type', 'median', 'window', 3);
methods{end+1} = struct('name', 'Median 5x5', 'type', 'median', 'window', 5);

% --- Category 4: Temporal Smoothing ---
for tw = [3, 5]
    methods{end+1} = struct('name', sprintf('Temporal Avg(%d)', tw), ...
        'type', 'temporal_avg', 'window', tw);
end

for ts = [0.5, 1.0]
    methods{end+1} = struct('name', sprintf('Temporal Gauss(%.1f)', ts), ...
        'type', 'temporal_gauss', 'sigma', ts);
end

% --- Category 5: Combined Methods ---
methods{end+1} = struct('name', 'TempAvg(3)+Gauss(1.5)+Th(0.3)', ...
    'type', 'temporal_spatial', 'tempWindow', 3, 'sigma', 1.5, 'threshold', 0.3);

methods{end+1} = struct('name', 'TempAvg(3)+Closing(2)', ...
    'type', 'temporal_morph', 'tempWindow', 3, 'radius', 2);

methods{end+1} = struct('name', 'Gauss(1.5)+Close(1)', ...
    'type', 'gauss_close', 'sigma', 1.5, 'radius', 1);

methods{end+1} = struct('name', 'Gauss(2.0)+Close(1)', ...
    'type', 'gauss_close', 'sigma', 2.0, 'radius', 1);

nMethods = length(methods);
fprintf('\nTesting %d methods on BOTH datasets...\n', nMethods);

%% Process all methods on BOTH datasets
results = struct();

for m = 1:nMethods
    method = methods{m};
    fprintf('\n[%d/%d] %s\n', m, nMethods, method.name);

    % === Process BinarisedData ===
    fprintf('  BinarisedData: ');
    lifetimes_bin = [];
    for trial = 1:nTrials
        trialData = binData(:,:,:,trial);
        lifetimes = trackBlobsForTrial(trialData, method, config, false);
        lifetimes_bin = [lifetimes_bin, lifetimes];
    end
    if isempty(lifetimes_bin)
        mean_bin = NaN;
    else
        mean_bin = mean(lifetimes_bin);
    end
    fprintf('%.2f frames (n=%d)\n', mean_bin, length(lifetimes_bin));

    % === Process Raw Grid40 ===
    fprintf('  Raw Grid40:    ');
    lifetimes_raw = [];
    nRecs = min(length(recList), config.maxRecordings);
    for r = 1:nRecs
        if ~isfield(recList(r), 'P1') || isempty(recList(r).P1)
            continue;
        end
        gridData = recList(r).P1;
        if iscell(gridData), gridData = gridData{1}; end
        if isempty(gridData), continue; end

        [~, ~, ~, nTrialsRec] = size(gridData);
        nTrialsToProcess = min(nTrialsRec, config.maxTrialsPerRec);

        for trial = 1:nTrialsToProcess
            trialData = gridData(:,:,:,trial);
            lifetimes = trackBlobsForTrial(trialData, method, config, true);
            lifetimes_raw = [lifetimes_raw, lifetimes];
        end
    end
    if isempty(lifetimes_raw)
        mean_raw = NaN;
    else
        mean_raw = mean(lifetimes_raw);
    end
    fprintf('%.2f frames (n=%d)\n', mean_raw, length(lifetimes_raw));

    % Store results
    results(m).name = method.name;
    results(m).type = method.type;
    results(m).mean_bin = mean_bin;
    results(m).std_bin = std(lifetimes_bin);
    results(m).n_bin = length(lifetimes_bin);
    results(m).mean_raw = mean_raw;
    results(m).std_raw = std(lifetimes_raw);
    results(m).n_raw = length(lifetimes_raw);

    % Handle NaN values - push invalid methods to bottom of rankings
    if isnan(mean_bin) || isnan(mean_raw)
        results(m).difference = Inf;  % Push to bottom of consistency ranking
        results(m).avg_persistence = 0;  % Push to bottom of longest ranking
    else
        results(m).difference = abs(mean_bin - mean_raw);
        results(m).avg_persistence = (mean_bin + mean_raw) / 2;
    end

    results(m).lifetimes_bin = lifetimes_bin;
    results(m).lifetimes_raw = lifetimes_raw;
end

%% === RESULTS: Smallest Difference (Best Consistency) ===
fprintf('\n\n========================================\n');
fprintf('RANKING 1: SMALLEST DIFFERENCE (Best Consistency)\n');
fprintf('========================================\n');

differences = [results.difference];
[~, sortByDiff] = sort(differences);

fprintf('\n%-40s %12s %12s %12s\n', 'Method', 'Binarised', 'Raw', 'Difference');
fprintf('%s\n', repmat('-', 1, 78));

for i = 1:min(20, nMethods)
    idx = sortByDiff(i);
    fprintf('%-40s %12.2f %12.2f %12.2f\n', ...
        results(idx).name, results(idx).mean_bin, results(idx).mean_raw, results(idx).difference);
end

bestConsistencyIdx = sortByDiff(1);
fprintf('\n>>> BEST CONSISTENCY: %s (diff = %.2f frames)\n', ...
    results(bestConsistencyIdx).name, results(bestConsistencyIdx).difference);

%% === RESULTS: Longest Blobs ===
fprintf('\n\n========================================\n');
fprintf('RANKING 2: LONGEST BLOBS\n');
fprintf('========================================\n');

avgPersistence = [results.avg_persistence];
[~, sortByLength] = sort(avgPersistence, 'descend');

fprintf('\n%-40s %12s %12s %12s\n', 'Method', 'Binarised', 'Raw', 'Average');
fprintf('%s\n', repmat('-', 1, 78));

for i = 1:min(20, nMethods)
    idx = sortByLength(i);
    fprintf('%-40s %12.2f %12.2f %12.2f\n', ...
        results(idx).name, results(idx).mean_bin, results(idx).mean_raw, results(idx).avg_persistence);
end

longestBlobIdx = sortByLength(1);
fprintf('\n>>> LONGEST BLOBS: %s (avg = %.2f frames)\n', ...
    results(longestBlobIdx).name, results(longestBlobIdx).avg_persistence);

%% === RESULTS: Combined Score (50% Consistency + 50% Persistence) ===
fprintf('\n\n========================================\n');
fprintf('RANKING 3: BEST COMBINED SCORE\n');
fprintf('(50%% Consistency + 50%% Persistence)\n');
fprintf('========================================\n');

% Identify valid methods (not Inf difference, not 0 persistence)
validMask = ~isinf([results.difference]) & [results.avg_persistence] > 0;
validIndices = find(validMask);

if ~isempty(validIndices)
    % Get valid values for normalization
    validDiffs = [results(validMask).difference];
    validPersist = [results(validMask).avg_persistence];

    % Compute normalized scores for all methods
    combinedScores = zeros(nMethods, 1);
    for m = 1:nMethods
        if validMask(m)
            % Normalize difference: lower is better → invert
            if max(validDiffs) > min(validDiffs)
                norm_consistency = 1 - (results(m).difference - min(validDiffs)) / (max(validDiffs) - min(validDiffs));
            else
                norm_consistency = 1;  % All same difference
            end

            % Normalize persistence: higher is better
            if max(validPersist) > min(validPersist)
                norm_persistence = (results(m).avg_persistence - min(validPersist)) / (max(validPersist) - min(validPersist));
            else
                norm_persistence = 1;  % All same persistence
            end

            % Combined score (equal weight)
            combinedScores(m) = 0.5 * norm_consistency + 0.5 * norm_persistence;
        else
            combinedScores(m) = -Inf;  % Invalid methods get lowest score
        end
        results(m).combined_score = combinedScores(m);
    end

    [~, sortByCombined] = sort(combinedScores, 'descend');

    fprintf('\n%-40s %12s %12s %12s\n', 'Method', 'Difference', 'Avg Persist', 'Score');
    fprintf('%s\n', repmat('-', 1, 78));

    for i = 1:min(20, nMethods)
        idx = sortByCombined(i);
        if combinedScores(idx) > -Inf
            fprintf('%-40s %12.2f %12.2f %12.3f\n', ...
                results(idx).name, results(idx).difference, results(idx).avg_persistence, combinedScores(idx));
        end
    end

    bestCombinedIdx = sortByCombined(1);
    fprintf('\n>>> BEST COMBINED: %s (score = %.3f)\n', ...
        results(bestCombinedIdx).name, combinedScores(bestCombinedIdx));
else
    bestCombinedIdx = 1;
    fprintf('No valid methods for combined scoring.\n');
end

%% === Summary ===
fprintf('\n\n========================================\n');
fprintf('SUMMARY\n');
fprintf('========================================\n');
fprintf('Best Consistency: %s\n', results(bestConsistencyIdx).name);
fprintf('  - BinarisedData: %.2f frames\n', results(bestConsistencyIdx).mean_bin);
fprintf('  - Raw Grid40:    %.2f frames\n', results(bestConsistencyIdx).mean_raw);
fprintf('  - Difference:    %.2f frames\n', results(bestConsistencyIdx).difference);
fprintf('\nLongest Blobs: %s\n', results(longestBlobIdx).name);
fprintf('  - BinarisedData: %.2f frames\n', results(longestBlobIdx).mean_bin);
fprintf('  - Raw Grid40:    %.2f frames\n', results(longestBlobIdx).mean_raw);
fprintf('  - Average:       %.2f frames\n', results(longestBlobIdx).avg_persistence);
fprintf('\nBest Combined: %s\n', results(bestCombinedIdx).name);
fprintf('  - BinarisedData: %.2f frames\n', results(bestCombinedIdx).mean_bin);
fprintf('  - Raw Grid40:    %.2f frames\n', results(bestCombinedIdx).mean_raw);
fprintf('  - Difference:    %.2f frames\n', results(bestCombinedIdx).difference);
fprintf('  - Avg Persist:   %.2f frames\n', results(bestCombinedIdx).avg_persistence);
fprintf('  - Score:         %.3f\n', results(bestCombinedIdx).combined_score);

%% Generate Figure
figure('Name', 'Blob Method Comparison', 'Color', 'w');

% Get top 10 by consistency for plotting
top10 = sortByDiff(1:min(10, nMethods));

subplot(2,2,1);
bar([results(top10).mean_bin; results(top10).mean_raw]', 'grouped');
set(gca, 'XTick', 1:length(top10), 'XTickLabel', {results(top10).name});
xtickangle(45);
ylabel('Mean Persistence (frames)');
title('Top 10 by Consistency: Binarised vs Raw');
legend('Binarised', 'Raw', 'Location', 'best');
grid on;

subplot(2,2,2);
bar([results(top10).difference]);
set(gca, 'XTick', 1:length(top10), 'XTickLabel', {results(top10).name});
xtickangle(45);
ylabel('|Binarised - Raw| (frames)');
title('Difference Between Datasets');
grid on;

% Histogram of best consistency method
subplot(2,2,3);
allLifetimes_best = [results(bestConsistencyIdx).lifetimes_bin, results(bestConsistencyIdx).lifetimes_raw];
if ~isempty(allLifetimes_best)
    edges = 0:2:max(allLifetimes_best) + 5;
    if ~isempty(results(bestConsistencyIdx).lifetimes_bin)
        histogram(results(bestConsistencyIdx).lifetimes_bin, edges, 'FaceColor', [0.2 0.6 0.9], ...
            'FaceAlpha', 0.5, 'DisplayName', 'Binarised');
    end
    hold on;
    if ~isempty(results(bestConsistencyIdx).lifetimes_raw)
        histogram(results(bestConsistencyIdx).lifetimes_raw, edges, 'FaceColor', [0.9 0.3 0.2], ...
            'FaceAlpha', 0.5, 'DisplayName', 'Raw');
    end
    xlabel('Blob Lifetime (frames)');
    ylabel('Frequency');
    legend('Location', 'best');
else
    text(0.5, 0.5, 'No blobs detected', 'HorizontalAlignment', 'center', 'Units', 'normalized');
end
title(sprintf('Best Consistency: %s', results(bestConsistencyIdx).name));
grid on;

% Histogram of longest blob method
subplot(2,2,4);
allLifetimes_longest = [results(longestBlobIdx).lifetimes_bin, results(longestBlobIdx).lifetimes_raw];
if ~isempty(allLifetimes_longest)
    edges = 0:2:max(allLifetimes_longest) + 5;
    if ~isempty(results(longestBlobIdx).lifetimes_bin)
        histogram(results(longestBlobIdx).lifetimes_bin, edges, 'FaceColor', [0.2 0.6 0.9], ...
            'FaceAlpha', 0.5, 'DisplayName', 'Binarised');
    end
    hold on;
    if ~isempty(results(longestBlobIdx).lifetimes_raw)
        histogram(results(longestBlobIdx).lifetimes_raw, edges, 'FaceColor', [0.9 0.3 0.2], ...
            'FaceAlpha', 0.5, 'DisplayName', 'Raw');
    end
    xlabel('Blob Lifetime (frames)');
    ylabel('Frequency');
    legend('Location', 'best');
else
    text(0.5, 0.5, 'No blobs detected', 'HorizontalAlignment', 'center', 'Units', 'normalized');
end
title(sprintf('Longest Blobs: %s', results(longestBlobIdx).name));
grid on;

%% Save results
save('Fig. 5 Model\IsingModels\Data\blob_method_comparison_results.mat', ...
    'results', 'methods', 'config', 'bestConsistencyIdx', 'longestBlobIdx', 'bestCombinedIdx');
fprintf('\nResults saved to blob_method_comparison_results.mat\n');

%% ======================== HELPER FUNCTIONS ========================

function lifetimes = trackBlobsForTrial(trialData, method, config, isRawData)
    [gridY, gridX, nFrames] = size(trialData);

    % Preprocess based on method
    processedData = preprocessTrial(trialData, method, isRawData);

    % Calculate max blob size in pixels
    gridSize = gridY * gridX;
    maxBlobPixels = gridSize * config.maxBlobSize;

    % Track blobs using IoU
    prev_blobs = {};
    active_blobs = struct('id', {}, 'start_frame', {}, 'pixels', {});
    lifetimes = [];
    next_blob_id = 1;

    for t = 1:nFrames
        frameData = processedData(:, :, t);

        % Ensure binary
        if max(frameData(:)) > 1
            binary = imbinarize(frameData, 'adaptive');
        else
            binary = frameData > 0.5;
        end

        % Connected components
        CC = bwconncomp(binary);

        current_blobs = {};
        if CC.NumObjects > 0
            blobSizes = cellfun(@numel, CC.PixelIdxList);
            % Filter by min AND max blob size
            for bi = 1:CC.NumObjects
                blobSize = blobSizes(bi);
                if blobSize >= config.minBlobSize && blobSize <= maxBlobPixels
                    current_blobs{end+1} = CC.PixelIdxList{bi};
                end
            end
        end

        % IoU tracking
        if isempty(prev_blobs)
            for bi = 1:length(current_blobs)
                active_blobs(end+1).id = next_blob_id;
                active_blobs(end).start_frame = t;
                active_blobs(end).pixels = current_blobs{bi};
                next_blob_id = next_blob_id + 1;
            end
        else
            matched_prev = false(length(prev_blobs), 1);
            matched_curr = false(length(current_blobs), 1);

            for ci = 1:length(current_blobs)
                best_iou = 0;
                best_match = 0;

                for pi = 1:length(prev_blobs)
                    if matched_prev(pi), continue; end

                    intersection = length(intersect(current_blobs{ci}, prev_blobs{pi}));
                    union_size = length(union(current_blobs{ci}, prev_blobs{pi}));
                    iou = intersection / union_size;

                    if iou > config.iouThreshold && iou > best_iou
                        best_iou = iou;
                        best_match = pi;
                    end
                end

                if best_match > 0
                    matched_prev(best_match) = true;
                    matched_curr(ci) = true;

                    for ai = 1:length(active_blobs)
                        if isequal(active_blobs(ai).pixels, prev_blobs{best_match})
                            active_blobs(ai).pixels = current_blobs{ci};
                            break;
                        end
                    end
                end
            end

            % End unmatched blobs
            for pi = 1:length(prev_blobs)
                if ~matched_prev(pi)
                    for ai = length(active_blobs):-1:1
                        if isequal(active_blobs(ai).pixels, prev_blobs{pi})
                            lifetime = t - active_blobs(ai).start_frame;
                            if lifetime > 0
                                lifetimes(end+1) = lifetime;
                            end
                            active_blobs(ai) = [];
                            break;
                        end
                    end
                end
            end

            % Add new blobs
            for ci = 1:length(current_blobs)
                if ~matched_curr(ci)
                    active_blobs(end+1).id = next_blob_id;
                    active_blobs(end).start_frame = t;
                    active_blobs(end).pixels = current_blobs{ci};
                    next_blob_id = next_blob_id + 1;
                end
            end
        end

        prev_blobs = current_blobs;
    end

    % Finalize active blobs
    for ai = 1:length(active_blobs)
        lifetime = nFrames - active_blobs(ai).start_frame + 1;
        if lifetime > 0
            lifetimes(end+1) = lifetime;
        end
    end
end

function processedData = preprocessTrial(trialData, method, isRawData)
    [gridY, gridX, nFrames] = size(trialData);
    processedData = zeros(gridY, gridX, nFrames);

    % For raw data, need to handle continuous values differently
    if isRawData
        % Raw data needs initial binarization step for most methods
        switch method.type
            case 'gaussian'
                for t = 1:nFrames
                    smoothed = imgaussfilt(double(trialData(:,:,t)), method.sigma);
                    % For raw data, use adaptive or Otsu thresholding
                    processedData(:,:,t) = imbinarize(smoothed, method.threshold);
                end

            case 'direct'
                for t = 1:nFrames
                    processedData(:,:,t) = imbinarize(double(trialData(:,:,t)));
                end

            otherwise
                % For morphological ops on raw data, binarize first then apply
                for t = 1:nFrames
                    binary = imbinarize(double(trialData(:,:,t)));
                    processedData(:,:,t) = applyMorphMethod(binary, method);
                end
        end
    else
        % Binarised data - apply method directly
        switch method.type
            case 'gaussian'
                for t = 1:nFrames
                    smoothed = imgaussfilt(double(trialData(:,:,t)), method.sigma);
                    processedData(:,:,t) = smoothed > method.threshold;
                end

            case 'direct'
                processedData = trialData > 0.5;

            case 'closing'
                se = strel('disk', method.radius);
                for t = 1:nFrames
                    binary = trialData(:,:,t) > 0.5;
                    processedData(:,:,t) = imclose(binary, se);
                end

            case 'opening'
                se = strel('disk', method.radius);
                for t = 1:nFrames
                    binary = trialData(:,:,t) > 0.5;
                    processedData(:,:,t) = imopen(binary, se);
                end

            case 'open_close'
                se = strel('disk', method.radius);
                for t = 1:nFrames
                    binary = trialData(:,:,t) > 0.5;
                    opened = imopen(binary, se);
                    processedData(:,:,t) = imclose(opened, se);
                end

            case 'close_open'
                se = strel('disk', method.radius);
                for t = 1:nFrames
                    binary = trialData(:,:,t) > 0.5;
                    closed = imclose(binary, se);
                    processedData(:,:,t) = imopen(closed, se);
                end

            case 'dilate'
                se = strel('disk', method.radius);
                for t = 1:nFrames
                    binary = trialData(:,:,t) > 0.5;
                    processedData(:,:,t) = imdilate(binary, se);
                end

            case 'median'
                for t = 1:nFrames
                    filtered = medfilt2(double(trialData(:,:,t)), [method.window, method.window]);
                    processedData(:,:,t) = filtered > 0.5;
                end

            case 'temporal_avg'
                halfWin = floor(method.window / 2);
                for t = 1:nFrames
                    tStart = max(1, t - halfWin);
                    tEnd = min(nFrames, t + halfWin);
                    avgFrame = mean(trialData(:,:,tStart:tEnd), 3);
                    processedData(:,:,t) = avgFrame > 0.5;
                end

            case 'temporal_gauss'
                for y = 1:gridY
                    for x = 1:gridX
                        timeSeries = squeeze(double(trialData(y,x,:)));
                        smoothed = imgaussfilt(timeSeries, method.sigma);
                        processedData(y,x,:) = smoothed > 0.5;
                    end
                end

            case 'temporal_spatial'
                halfWin = floor(method.tempWindow / 2);
                for t = 1:nFrames
                    tStart = max(1, t - halfWin);
                    tEnd = min(nFrames, t + halfWin);
                    avgFrame = mean(trialData(:,:,tStart:tEnd), 3);
                    smoothed = imgaussfilt(double(avgFrame), method.sigma);
                    processedData(:,:,t) = smoothed > method.threshold;
                end

            case 'temporal_morph'
                halfWin = floor(method.tempWindow / 2);
                se = strel('disk', method.radius);
                for t = 1:nFrames
                    tStart = max(1, t - halfWin);
                    tEnd = min(nFrames, t + halfWin);
                    avgFrame = mean(trialData(:,:,tStart:tEnd), 3);
                    binary = avgFrame > 0.5;
                    processedData(:,:,t) = imclose(binary, se);
                end

            case 'gauss_close'
                se = strel('disk', method.radius);
                for t = 1:nFrames
                    smoothed = imgaussfilt(double(trialData(:,:,t)), method.sigma);
                    binary = smoothed > 0.3;
                    processedData(:,:,t) = imclose(binary, se);
                end

            otherwise
                processedData = trialData > 0.5;
        end
    end
end

function processed = applyMorphMethod(binary, method)
    switch method.type
        case 'closing'
            se = strel('disk', method.radius);
            processed = imclose(binary, se);
        case 'opening'
            se = strel('disk', method.radius);
            processed = imopen(binary, se);
        case 'open_close'
            se = strel('disk', method.radius);
            processed = imclose(imopen(binary, se), se);
        case 'close_open'
            se = strel('disk', method.radius);
            processed = imopen(imclose(binary, se), se);
        case 'dilate'
            se = strel('disk', method.radius);
            processed = imdilate(binary, se);
        case 'median'
            processed = medfilt2(double(binary), [method.window, method.window]) > 0.5;
        otherwise
            processed = binary;
    end
end
