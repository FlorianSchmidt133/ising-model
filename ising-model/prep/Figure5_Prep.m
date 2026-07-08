%% =========================================================================
%% === Fig 5 Prep: Break Period Data Preparation for Ising Models ===
%% =========================================================================
% This script prepares break period data for Ising model training.
% Break periods provide longer continuous sequences of neural activity
% compared to individual trials, enabling analysis of spontaneous dynamics.
%
% Data is kept at the individual break level (not concatenated) to allow
% analysis of temporal evolution across break sequences (e.g., early vs late breaks).

%% =========================================================================
%% === SECTION 01: Setup and Parameters ===
%% =========================================================================

% Load primary data structures
% load(mba_p('RawData3.mat'),'ActivityData');
% load(mba_p('RawData3.mat'),'params');
% load(mba_p('RawData3.mat'),'CaData');

% Define Skip arrays (recordings to exclude from analysis)
Skip = [];
Skip.Naive = [1 9 10 16];
Skip.Beginner = [1 6 7 11];
Skip.Expert = [1 4 12 13 14];
Skip.NoSpout = [1 4 9 10 11 13 14];
Skip.ExpertRandom = [1];
Skip.ExpertAll = [1,4,5,13, 20,21,22,23,24,25,26];

% Define conditions to analyze
conditions = {'Naive', 'Beginner', 'Expert', 'NoSpout'};
conditions = {'Naive', 'Beginner', 'ExpertRandom', 'NoSpout'};

% Rasterization parameters (consistent with Figure 3)
gridSize = 40;
gridDimensions = [13 26];
pixelSize = 1.5674; % microns per pixel

% Blob detection parameters (consistent with Figure 3)
blob_params = struct();
blob_params.sigma = 2;           % Standard deviation for Gaussian smoothing
blob_params.threshold = 1;       % Threshold for binarization
blob_params.minBlobSize = 5;     % Minimum blob size in pixels

% Analysis parameters
calculation_method = 'CoV';      % Coefficient of Variation
selection_method = 'EyePosition'; % Consistent with Figure 1
dataTreatment = 'Raw';           % Raw data (not mean-subtracted)

fprintf('\n=== Figure 5 Prep: Break Period Data Preparation ===\n');
fprintf('Conditions: %s\n', strjoin(conditions, ', '));
fprintf('Grid size: %d x %d (gridSize=%d)\n', gridDimensions(1), gridDimensions(2), gridSize);

%% =========================================================================
%% === SECTION 02: Extract Break Data  ===
%% =========================================================================

fprintf('\n--- Section 2: Extracting Break Data ---\n');

% Initialize structures
ActivityData_Breaks = struct();
BreakMetadata = struct();

for c = 1:length(conditions)
    condition = conditions{c};
    fprintf('\nProcessing condition: %s\n', condition);

    % Get number of recordings for this condition
    nRecs = length(ActivityData.(condition));

    % Initialize condition structure
    ActivityData_Breaks.(condition) = struct();
    BreakMetadata.(condition) = struct();

    for r = 1:nRecs
        % Check if this recording should be skipped
        if ismember(r, Skip.(condition))
            fprintf('  Recording %d: SKIPPED\n', r);
            continue;
        end

        % Check if Break field exists
        if ~isfield(ActivityData.(condition)(r), 'Break')
            fprintf('  Recording %d: No Break field found\n', r);
            continue;
        end

        % Extract break data
        breakData = ActivityData.(condition)(r).Break;

        % Handle different possible Break field structures
            if ndims(breakData) == 3
                % Break is 3D: (neurons x timepoints x breaks)
                nBreaks = size(breakData, 3);
                ActivityData_Breaks.(condition)(r).Break = cell(1, nBreaks);
                % Split third dimension into separate cell elements
                for b = 1:nBreaks
                    ActivityData_Breaks.(condition)(r).Break{b} = breakData(:, :, b);
                end
            elseif ndims(breakData) == 2
                % Break is 2D: (neurons x timepoints) - single break
                nBreaks = 1;
                ActivityData_Breaks.(condition)(r).Break = {breakData};
            else
                fprintf('  Recording %d: Unexpected numeric array dimensions (%d)\n', r, ndims(breakData));
                continue;
            end
      
        

        % Store metadata for each break
        BreakMetadata.(condition)(r).nBreaks = nBreaks;
        BreakMetadata.(condition)(r).breakDurations = zeros(1, nBreaks);
        BreakMetadata.(condition)(r).breakIndices = 1:nBreaks;

        % Calculate duration and validate each break
        for b = 1:nBreaks
            breakMat = ActivityData_Breaks.(condition)(r).Break{b};

            if isstruct(breakMat)
                % If still struct, try to extract data field
                if isfield(breakMat, 'data')
                    breakMat = breakMat.data;
                    ActivityData_Breaks.(condition)(r).Break{b} = breakMat;
                end
            end

            if isnumeric(breakMat)
                [nNeurons, nTimepoints] = size(breakMat);
                BreakMetadata.(condition)(r).breakDurations(b) = nTimepoints;
                BreakMetadata.(condition)(r).breakNeurons(b) = nNeurons;
            end
        end

        fprintf('  Recording %d: %d breaks extracted (durations: %s frames)\n', ...
            r, nBreaks, mat2str(BreakMetadata.(condition)(r).breakDurations));
    end
end

fprintf('\n--- Section 2 Complete: Break data extracted ---\n');

%% =========================================================================
%% === SECTION 03: Rasterize Break Data ===
%% =========================================================================

fprintf('\n--- Section 3: Rasterizing Break Data ---\n');

% Initialize rasterized break structure
Grid40_Breaks = struct();

for c = 1:length(conditions)
    condition = conditions{c};
    fprintf('\nRasterizing condition: %s\n', condition);

    nRecs = length(ActivityData.(condition));

    % Initialize condition structure
    Grid40_Breaks.(condition) = struct();

    for r = 1:nRecs
        % Check if this recording should be skipped or has no breaks
        if ismember(r, Skip.(condition))
            continue;
        end

        % Check if this recording exists in ActivityData_Breaks
        if r > length(ActivityData_Breaks.(condition)) || ...
           ~isfield(ActivityData_Breaks.(condition)(r), 'Break')
            continue;
        end

        % Get cell locations from CaData
        cellPositions = [CaData.(condition)(r).CaX,CaData.(condition)(r).CaY];

        nBreaks = BreakMetadata.(condition)(r).nBreaks;
        Grid40_Breaks.(condition)(r).Break = cell(1, nBreaks);

        % Rasterize each break separately
        for b = 1:nBreaks
            breakData = ActivityData_Breaks.(condition)(r).Break{b};

            if ~isnumeric(breakData) || isempty(breakData)
                continue;
            end

            [nNeurons, nTimepoints] = size(breakData);

            % Initialize rasterized grid for this break
            rasterBreak = zeros(gridDimensions(1), gridDimensions(2), nTimepoints);

            % Rasterize: assign each neuron's activity to grid location
            for n = 1:nNeurons
                % Get grid coordinates for this neuron
                x_pos = cellPositions(n, 1);
                y_pos = cellPositions(n, 2);

                % Convert to grid indices
                x_idx = round(x_pos / gridSize) + 1;
                y_idx = round(y_pos / gridSize) + 1;

                % Ensure within grid bounds
                x_idx = max(1, min(x_idx, gridDimensions(2)));
                y_idx = max(1, min(y_idx, gridDimensions(1)));

                % Add neuron's activity to grid (average if multiple neurons per grid)
                for t = 1:nTimepoints
                    if rasterBreak(y_idx, x_idx, t) == 0
                        rasterBreak(y_idx, x_idx, t) = breakData(n, t);
                    else
                        % Average with existing value if grid cell already occupied
                        rasterBreak(y_idx, x_idx, t) = (rasterBreak(y_idx, x_idx, t) + breakData(n, t)) / 2;
                    end
                end
            end

            % Store rasterized break
            Grid40_Breaks.(condition)(r).Break{b} = rasterBreak;
        end

        fprintf('  Recording %d: %d breaks rasterized\n', r, nBreaks);
    end
end

fprintf('\n--- Section 3 Complete: Break data rasterized ---\n');

%% =========================================================================
%% === SECTION 04: Calculate Entropy for Breaks ===
%% =========================================================================

fprintf('\n--- Section 4: Calculating Entropy for Breaks ---\n');
fprintf('Break period analysis: Processing all neurons (no cluster-based filtering)\n\n');

% Initialize entropy structure
Entropy_Breaks = struct();

% Loop through conditions
for c = 1:length(conditions)
    condition = conditions{c};
    fprintf('\n=== Processing Condition: %s ===\n', condition);
    fprintf('  Processing: All neurons\n');

    nRecs = length(ActivityData.(condition));

    % Initialize entropy structure for this condition
    E = struct();
        E.RecordingEntropy_Raw = {};
        E.RecordingEntropy_RZ = {};
        E.RecordingEntropy_TZ = {};
        E.RecordingEntropy_subsampled = {};
        E.RecordingEntropyZ_Raw = {};
        E.RecordingEntropyZ_RZ = {};
        E.RecordingEntropyZ_TZ = {};
        E.RecordingEntropyZ_subsampled = {};

        % Loop through recordings for this condition
        for r = 1:nRecs
            % Check if this recording should be skipped or has no breaks
            if ismember(r, Skip.(condition))
                continue;
            end

            if r > length(ActivityData_Breaks.(condition)) || ...
               ~isfield(ActivityData_Breaks.(condition)(r), 'Break')
                continue;
            end

            nBreaks = BreakMetadata.(condition)(r).nBreaks;

            % Initialize temporary storage for this recording
            tempR = [];           % Raw entropy
            tempZ = [];           % Trial-z-scored entropy (TZ)
            temp = [];            % For calculating recording-z-score (RZ)
            tempR_ZData = [];     % Raw entropy on z-scored activity
            tempZ_ZData = [];     % TZ entropy on z-scored activity
            temp_ZData = [];      % For calculating RZ on z-scored activity
            temp_subsampled = [];
            temp_subsampled_ZData = [];

            % Process each break
            for b = 1:nBreaks
                breakData = ActivityData_Breaks.(condition)(r).Break{b};

                if ~isnumeric(breakData) || isempty(breakData)
                    continue;
                end

                [nNeurons, nTimepoints] = size(breakData);

                % Use all neurons (no cluster-based filtering during break periods)
                data = breakData;

                % Handle subsampling for large recordings
                if nNeurons > 1000
                    data_subsampled = breakData(1:1000, :);
                end

                % Check if we have z-scored activity data
                if isfield(ActivityData.(condition)(r), 'BreakZ') && ~isempty(ActivityData.(condition)(r).BreakZ)
                    % Use pre-computed z-scored data
                    breakDataZ = ActivityData.(condition)(r).BreakZ;
                    dataZ = breakDataZ;
                    if nNeurons > 1000
                        dataZ_subsampled = breakDataZ(1:1000, :);
                    end
                else
                    % Create z-scored data on the fly (z-score each neuron)
                    dataZ = data;
                    for neuronIdx_i = 1:size(data, 1)
                        neuron_trace = data(neuronIdx_i, :);
                        if std(neuron_trace) > 0
                            dataZ(neuronIdx_i, :) = (neuron_trace - mean(neuron_trace)) / std(neuron_trace);
                        end
                    end

                    if nNeurons > 1000
                        dataZ_subsampled = dataZ(1:1000, :);
                    end
                end

                % Calculate population entropy using population_entropy function
                population_ent = population_entropy(data);

                % Trial-z-score (TZ): normalize relative to first portion (baseline)
                baseline_length = min(80, round(nTimepoints * 0.2)); % Use first 80 frames or 20% of break
                if baseline_length > 1 && std(population_ent(1:baseline_length)) > 0
                    population_entZ = (population_ent - mean(population_ent(1:baseline_length))) / sqrt(var(population_ent(1:baseline_length)));
                else
                    population_entZ = population_ent; % Can't z-score
                end

                % Store for this break
                tempR = [tempR; population_ent];
                tempZ = [tempZ; population_entZ];
                temp = [temp; population_ent];

                % Calculate entropy on z-scored activity data
                population_ent_ZData = population_entropy(dataZ);

                % Trial-z-score on z-scored activity
                if baseline_length > 1 && std(population_ent_ZData(1:baseline_length)) > 0
                    population_entZ_ZData = (population_ent_ZData - mean(population_ent_ZData(1:baseline_length))) / sqrt(var(population_ent_ZData(1:baseline_length)));
                else
                    population_entZ_ZData = population_ent_ZData;
                end

                tempR_ZData = [tempR_ZData; population_ent_ZData];
                tempZ_ZData = [tempZ_ZData; population_entZ_ZData];
                temp_ZData = [temp_ZData; population_ent_ZData];

                % Handle subsampled entropy for large recordings
                if nNeurons > 1000
                    temp_subsampled = [temp_subsampled; population_entropy2(data_subsampled)];
                    temp_subsampled_ZData = [temp_subsampled_ZData; population_entropy2(dataZ_subsampled)];
                end
            end

            % Recording-z-score (RZ): normalize across all breaks in recording
            tempF = temp - mean(temp(:));
            if std(temp(:)) > 0
                tempF = tempF ./ std(temp(:));
            end

            tempF_ZData = temp_ZData - mean(temp_ZData(:));
            if std(temp_ZData(:)) > 0
                tempF_ZData = tempF_ZData ./ std(temp_ZData(:));
            end

            % Store for this recording
            E.RecordingEntropy_Raw{end+1} = tempR;
            E.RecordingEntropy_RZ{end+1} = tempF;
            E.RecordingEntropy_TZ{end+1} = tempZ;

            E.RecordingEntropyZ_Raw{end+1} = tempR_ZData;
            E.RecordingEntropyZ_RZ{end+1} = tempF_ZData;
            E.RecordingEntropyZ_TZ{end+1} = tempZ_ZData;

            if nNeurons > 1000
                E.RecordingEntropy_subsampled{end+1} = temp_subsampled;
                E.RecordingEntropyZ_subsampled{end+1} = temp_subsampled_ZData;
            end

            fprintf('    Recording %d: %d breaks processed\n', r, nBreaks);
        end

    % Note: Unlike trials which have fixed WindowLength, breaks have variable durations
    % Therefore, we keep concatenated versions as cell arrays (one cell per recording)
    % rather than attempting to vertcat matrices with inconsistent dimensions

    if ~isempty(E.RecordingEntropy_Raw)
        % Store as cell arrays preserving per-recording structure
        E.Entropy_Raw = E.RecordingEntropy_Raw;
        E.Entropy_RZ = E.RecordingEntropy_RZ;
        E.Entropy_TZ = E.RecordingEntropy_TZ;

        E.EntropyZ_Raw = E.RecordingEntropyZ_Raw;
        E.EntropyZ_RZ = E.RecordingEntropyZ_RZ;
        E.EntropyZ_TZ = E.RecordingEntropyZ_TZ;

        try
            E.Entropy_subsampled = E.RecordingEntropy_subsampled;
            E.EntropyZ_subsampled = E.RecordingEntropyZ_subsampled;
        catch
            E.Entropy_subsampled = {};
            E.EntropyZ_subsampled = {};
        end

        % Create summary statistics across all breaks
        allRaw = [];
        allRZ = [];
        allTZ = [];
        for recIdx = 1:length(E.RecordingEntropy_Raw)
            allRaw = [allRaw; E.RecordingEntropy_Raw{recIdx}(:)];
            allRZ = [allRZ; E.RecordingEntropy_RZ{recIdx}(:)];
            allTZ = [allTZ; E.RecordingEntropy_TZ{recIdx}(:)];
        end
        E.Entropy_Raw_Mean = nanmean(allRaw);
        E.Entropy_Raw_Std = nanstd(allRaw);
        E.Entropy_RZ_Mean = nanmean(allRZ);
        E.Entropy_RZ_Std = nanstd(allRZ);
        E.Entropy_TZ_Mean = nanmean(allTZ);
        E.Entropy_TZ_Std = nanstd(allTZ);
    else
        E.Entropy_Raw = {};
        E.Entropy_RZ = {};
        E.Entropy_TZ = {};
        E.EntropyZ_Raw = {};
        E.EntropyZ_RZ = {};
        E.EntropyZ_TZ = {};
        E.Entropy_subsampled = {};
        E.EntropyZ_subsampled = {};
        E.Entropy_Raw_Mean = NaN;
        E.Entropy_Raw_Std = NaN;
        E.Entropy_RZ_Mean = NaN;
        E.Entropy_RZ_Std = NaN;
        E.Entropy_TZ_Mean = NaN;
        E.Entropy_TZ_Std = NaN;
    end

    % Store in final structure: Entropy_Breaks.Condition
    Entropy_Breaks.(condition) = E;

    fprintf('  Condition %s complete: entropy calculated\n', condition);
end

fprintf('\n--- Section 4 Complete: Entropy calculated ---\n');
fprintf('Structure: Entropy_Breaks.Condition\n');
fprintf('All neurons analyzed (no cluster-based filtering during break periods)\n');
fprintf('Each contains: Raw, RZ, TZ, subsampled (when >1000 neurons)\n');
fprintf('Calculated on both raw activity and z-scored activity data\n');

%% =========================================================================
%% === SECTION 05: CoV Analysis on Breaks ===
%% =========================================================================

fprintf('\n--- Section 5: Calculating CoV for Breaks ---\n');

% Initialize CoV structure
CoV_Breaks = struct();

for c = 1:length(conditions)
    condition = conditions{c};
    fprintf('\nCalculating CoV for condition: %s\n', condition);

    nRecs = length(ActivityData.(condition));
    CoV_Breaks.(condition) = struct();

    for r = 1:nRecs
        % Check if this recording should be skipped or has no breaks
        if ismember(r, Skip.(condition))
            continue;
        end

        % Check if this recording exists in ActivityData_Breaks
        if r > length(ActivityData_Breaks.(condition)) || ...
           ~isfield(ActivityData_Breaks.(condition)(r), 'Break')
            continue;
        end

        nBreaks = BreakMetadata.(condition)(r).nBreaks;
        CoV_Breaks.(condition)(r).Break = cell(1, nBreaks);
        CoV_Breaks.(condition)(r).BreakMean = zeros(1, nBreaks);
        CoV_Breaks.(condition)(r).BreakMedian = zeros(1, nBreaks);
        CoV_Breaks.(condition)(r).NeuronCoV = cell(1, nBreaks);

        % Calculate CoV for each break separately
        for b = 1:nBreaks
            breakData = ActivityData_Breaks.(condition)(r).Break{b};

            if ~isnumeric(breakData) || isempty(breakData)
                continue;
            end

            [nNeurons, nTimepoints] = size(breakData);
            neuronCoV = zeros(nNeurons, 1);

            % Calculate CoV for each neuron across the break period
            for n = 1:nNeurons
                activity_n = breakData(n, :);

                % Remove NaN values
                validIdx = ~isnan(activity_n);
                activity_valid = activity_n(validIdx);

                if length(activity_valid) > 1 && mean(activity_valid) > 0
                    % CoV = std / mean
                    neuronCoV(n) = std(activity_valid) / mean(activity_valid);
                else
                    neuronCoV(n) = NaN;
                end
            end

            % Store CoV results for this break
            CoV_Breaks.(condition)(r).NeuronCoV{b} = neuronCoV;
            CoV_Breaks.(condition)(r).BreakMean(b) = nanmean(neuronCoV);
            CoV_Breaks.(condition)(r).BreakMedian(b) = nanmedian(neuronCoV);
        end

        fprintf('  Recording %d: CoV calculated for %d breaks\n', r, nBreaks);
    end
end

fprintf('\n--- Section 5 Complete: CoV calculated ---\n');

%% =========================================================================
%% === SECTION 06: Blob Detection on Breaks ===
%% =========================================================================
% Uses the same blob detection approach as blobDetection.m:
% - 2D frame-by-frame analysis: Detects instantaneous spatial blobs at each timepoint
% - 3D volume analysis: Detects spatiotemporal blobs across the entire break duration
% Both methods use: imgaussfilt/imgaussfilt3 → imbinarize → bwareaopen → bwconncomp

fprintf('\n--- Section 6: Blob Detection on Breaks (2D + 3D methods) ---\n');

% Initialize blob detection structure
BlobDetection_Breaks = struct();

% Define pretreatment (3 = min-subtracted)
pt = 0; % Use mean-subtracted data

for c = 1:length(conditions)
    condition = conditions{c};
    fprintf('\nBlob detection for condition: %s\n', condition);

    nRecs = length(ActivityData.(condition));
    BlobDetection_Breaks.(condition) = struct();

    for r = 1:nRecs
        % Check if this recording should be skipped or has no breaks
        if ismember(r, Skip.(condition))
            continue;
        end

        % Check if this recording exists in Grid40_Breaks
        if r > length(Grid40_Breaks.(condition)) || ...
           ~isfield(Grid40_Breaks.(condition)(r), 'Break')
            continue;
        end

        nBreaks = length(Grid40_Breaks.(condition)(r).Break);
        BlobDetection_Breaks.(condition)(r).Break = cell(1, nBreaks);
        BlobDetection_Breaks.(condition)(r).BlobCount2D = zeros(1, nBreaks);
        BlobDetection_Breaks.(condition)(r).BlobCount3D = zeros(1, nBreaks);
        BlobDetection_Breaks.(condition)(r).BlobRate2D = zeros(1, nBreaks);
        BlobDetection_Breaks.(condition)(r).BlobRate3D = zeros(1, nBreaks);

        % Apply blob detection to each break separately (both 2D and 3D methods)
        for b = 1:nBreaks
            rasterBreak = Grid40_Breaks.(condition)(r).Break{b};

            if isempty(rasterBreak)
                continue;
            end

            [gridY, gridX, nTimepoints] = size(rasterBreak);

            % Apply preprocessing (following blobDetection.m lines 30-34)
            if pt == 2
                % Mean subtraction across time (adapted for 3D breaks, not 4D trials)
                meanActivity = mean(rasterBreak, 3);
                rasterBreak_processed = rasterBreak - repmat(meanActivity, [1, 1, nTimepoints]);
            elseif pt == 3
                % Min subtraction: subtract min of temporal mean
                minActivity = min(mean(rasterBreak, 3), [], 3);
                rasterBreak_processed = rasterBreak - minActivity;
            else
                rasterBreak_processed = rasterBreak;
            end

            %% --- 2D Processing (Frame-by-Frame) ---
            % Following blobDetection.m lines 41-62
            blobCounts2D = zeros(1, nTimepoints);
            blobSizes2D = cell(1, nTimepoints);
            blobCentroids2D = cell(1, nTimepoints);

            % Detect blobs at each timepoint
            for t = 1:nTimepoints
                frame = rasterBreak_processed(:, :, t);

                % Preprocessing: smooth, binarize, and remove small objects
                frame_smoothed = imgaussfilt(frame, blob_params.sigma);
                bw = imbinarize(frame_smoothed, blob_params.threshold);
                bw_clean = bwareaopen(bw, blob_params.minBlobSize);

                % Blob detection on the 2D frame
                cc = bwconncomp(bw_clean);
                blobCounts2D(t) = cc.NumObjects;

                % Store blob properties
                if cc.NumObjects > 0
                    sizes = cellfun(@length, cc.PixelIdxList);
                    blobSizes2D{t} = sizes;

                    % Calculate centroids
                    centroids = zeros(cc.NumObjects, 2);
                    for blobIdx = 1:cc.NumObjects
                        [rows, cols] = ind2sub([gridY, gridX], cc.PixelIdxList{blobIdx});
                        centroids(blobIdx, :) = [mean(cols), mean(rows)];
                    end
                    blobCentroids2D{t} = centroids;
                end
            end

            %% --- 3D Processing (Break as Volume) ---
            % Following blobDetection.m lines 64-79
            % Treat the entire break as a 3D spatiotemporal volume
            volume = rasterBreak_processed;

            % Preprocessing: smooth 3D volume, binarize, and remove small 3D objects
            volume_smoothed = imgaussfilt3(volume, blob_params.sigma);
            bw_vol = imbinarize(volume_smoothed, blob_params.threshold);
            bw_vol_clean = bwareaopen(bw_vol, blob_params.minBlobSize, 26);

            % Blob detection using 26-connected neighborhood in 3D
            cc3D = bwconncomp(bw_vol_clean, 26);
            blobCount3D = cc3D.NumObjects;

            % Store 3D blob properties
            blobSizes3D = [];
            blobCentroids3D = [];
            if blobCount3D > 0
                blobSizes3D = cellfun(@length, cc3D.PixelIdxList);

                % Calculate 3D centroids (x, y, time)
                blobCentroids3D = zeros(blobCount3D, 3);
                for blobIdx = 1:blobCount3D
                    [rows, cols, times] = ind2sub([gridY, gridX, nTimepoints], cc3D.PixelIdxList{blobIdx});
                    blobCentroids3D(blobIdx, :) = [mean(cols), mean(rows), mean(times)];
                end
            end

            %% Store results for this break
            breakResults = struct();

            % 2D results (frame-by-frame)
            breakResults.method2D.blobCountsPerFrame = blobCounts2D;
            breakResults.method2D.meanBlobCount = mean(blobCounts2D);
            breakResults.method2D.totalBlobs = sum(blobCounts2D);
            breakResults.method2D.blobSizes = blobSizes2D;
            breakResults.method2D.blobCentroids = blobCentroids2D;
            breakResults.method2D.blobRate = sum(blobCounts2D) / nTimepoints; % Blobs per frame

            % 3D results (volume analysis)
            breakResults.method3D.blobCount = blobCount3D;
            breakResults.method3D.blobSizes = blobSizes3D;
            breakResults.method3D.blobCentroids = blobCentroids3D;
            breakResults.method3D.blobRate = blobCount3D / (nTimepoints / 10); % Blobs per second (10 Hz imaging)

            % Store in main structure
            BlobDetection_Breaks.(condition)(r).Break{b} = breakResults;
            BlobDetection_Breaks.(condition)(r).BlobCount2D(b) = sum(blobCounts2D);
            BlobDetection_Breaks.(condition)(r).BlobCount3D(b) = blobCount3D;
            BlobDetection_Breaks.(condition)(r).BlobRate2D(b) = breakResults.method2D.blobRate;
            BlobDetection_Breaks.(condition)(r).BlobRate3D(b) = breakResults.method3D.blobRate;
        end

        fprintf('  Recording %d: Blob detection (2D + 3D) completed for %d breaks\n', r, nBreaks);
    end
end

fprintf('\n--- Section 6 Complete: Blob detection completed (2D frame-by-frame + 3D volume) ---\n');

%% =========================================================================
%% === SECTION 07: Save Prepared Data ===
%% =========================================================================

% fprintf('\n--- Section 7: Saving Prepared Data ---\n');
% 
% % Define save path
% savePath = 'Fig. 5 Ising Models\';
% if ~exist(savePath, 'dir')
%     mkdir(savePath);
% end
% 
% saveFile = fullfile(savePath, 'Figure5_PreparedData.mat');
% 
% % Save all prepared structures
% save(saveFile, 'ActivityData_Breaks', 'Grid40_Breaks', 'Entropy_Breaks', ...
%     'CoV_Breaks', 'BlobDetection_Breaks', 'BreakMetadata', ...
%     'blob_params', 'gridSize', 'gridDimensions', 'conditions', ...
%     'conditions', 'Skip', '-v7.3');
% 
% fprintf('Data saved to: %s\n', saveFile);
% 
% % Print summary statistics
% fprintf('\n=== Summary Statistics ===\n');
% for c = 1:length(conditions)
%     condition = conditions{c};
%     fprintf('\n%n', condition);
% 
%     totalBreaks = 0;
%     totalFrames = 0;
% 
%     for r = 1:length(ActivityData.(condition))
%         if ismember(r, Skip.(condition))
%             continue;
%         end
% 
%         % Check if this recording exists in BreakMetadata
%         if r <= length(BreakMetadata.(condition)) && ...
%            isfield(BreakMetadata.(condition)(r), 'nBreaks')
%             nBreaks = BreakMetadata.(condition)(r).nBreaks;
%             totalBreaks = totalBreaks + nBreaks;
%             totalFrames = totalFrames + sum(BreakMetadata.(condition)(r).breakDurations);
% 
%             fprintf('  Rec %d: %d breaks, %d total frames\n', ...
%                 r, nBreaks, sum(BreakMetadata.(condition)(r).breakDurations));
%         end
%     end
% 
%     fprintf('  Total: %d breaks, %d frames\n', totalBreaks, totalFrames);
% end
% 
% fprintf('\n=== Figure 5 Prep Complete ===\n');
% fprintf('Break data prepared and ready for Ising model training\n');

%% =========================================================================
%% === SECTION 08: Visualization and Quality Control Plots ===
%% =========================================================================

fprintf('\n--- Section 8: Creating Visualization Plots ---\n');

% Define condition colors (consistent with main figures)
conditionColors = struct();
conditionColors.Naive = params.Naive;
conditionColors.Beginner = params.Beginner;
conditionColors.Expert = params.Expert;
conditionColors.NoSpout = params.NoSpout;
conditionColors.ExpertRandom = params.Expert * 0.7; % Slightly darker Expert color
conditionColors.ExpertAll = params.Expert * 0.5; % Even darker Expert color

%% =========================================================================
%% === SECTION 8.1: CoV Visualization ===
%% =========================================================================

fprintf('\n--- Section 8.1: CoV Visualization ---\n');

% Panel A: Violin plot comparing mean CoV across conditions
figure('Name', 'CoV Comparison Across Conditions');
tiledlayout();
% Pre-allocate colors for all conditions to ensure consistent indexing
covColors = zeros(length(conditions), 3);
for c = 1:length(conditions)
    condition = conditions{c};
    if isfield(conditionColors, condition)
        covColors(c, :) = conditionColors.(condition);
    else
        covColors(c, :) = [0.5 0.5 0.5];
    end
end

% Collect CoV data for each condition
covData = cell(1, length(conditions));
covLabels = cell(1, length(conditions));

for c = 1:length(conditions)
    condition = conditions{c};

    % Collect all break-level mean CoV values
    conditionCoV = [];
    for r = 1:length(CoV_Breaks.(condition))
        if isfield(CoV_Breaks.(condition)(r), 'BreakMean') && ...
           ~isempty(CoV_Breaks.(condition)(r).BreakMean)
            conditionCoV = [conditionCoV, CoV_Breaks.(condition)(r).BreakMean];
        end
    end

    % Store for plotting (colors already assigned above)
    if ~isempty(conditionCoV)
        covData{c} = conditionCoV;
        covLabels{c} = condition;
    end
end

% Create violin plot
nexttile;
hold on;
for c = 1:length(covData)
    if ~isempty(covData{c})
        % Violin plot using histogram-based approach
        [counts, edges] = histcounts(covData{c}, 20);
        centers = (edges(1:end-1) + edges(2:end)) / 2;
        counts_norm = counts / max(counts) * 0.3; % Normalize width

        % Plot violin shape
        fill([c - counts_norm, fliplr(c + counts_norm)], ...
             [centers, fliplr(centers)], covColors(c, :), ...
             'FaceAlpha', 0.3, 'EdgeColor', 'none');

        % Add median line
        plot([c-0.3, c+0.3], [median(covData{c}), median(covData{c})], ...
             'k-', 'LineWidth', 2);

        % Add scatter points
        scatter(c * ones(size(covData{c})), covData{c}, 20, covColors(c, :), ...
                'filled', 'MarkerFaceAlpha', 0.3);
    end
end
xticks(1:length(covLabels));
xticklabels(covLabels);
ylabel('Coefficient of Variation (CoV)');
title('CoV Across Conditions');
box on;
grid on;

% Panel B: Distribution histogram
nexttile;
hold on;

for c = 1:length(conditions)
    condition = conditions{c};

    % Collect all neuron-level CoV values
    allNeuronCoV = [];
    for r = 1:length(CoV_Breaks.(condition))
        if isfield(CoV_Breaks.(condition)(r), 'NeuronCoV')
            for b = 1:length(CoV_Breaks.(condition)(r).NeuronCoV)
                if ~isempty(CoV_Breaks.(condition)(r).NeuronCoV{b})
                    allNeuronCoV = [allNeuronCoV; CoV_Breaks.(condition)(r).NeuronCoV{b}];
                end
            end
        end
    end

    % Plot CDF
    if ~isempty(allNeuronCoV)
        % Remove NaN and extreme outliers
        allNeuronCoV = allNeuronCoV(~isnan(allNeuronCoV) & allNeuronCoV < 10);

        % Compute KDE-based CDF
        [f, xi] = ksdensity(allNeuronCoV, 'Function', 'cdf', ...
                           'Support', [0, Inf], 'NumPoints', 200);

        % Plot the CDF
        plot(xi, f, 'Color', covColors(c, :), ...
             'LineWidth', 2, 'DisplayName', condition);
    end
end

xlabel('Coefficient of Variation');
ylabel('Cumulative Probability');
title('CoV Cumulative Distribution');
legend('Location', 'best');
grid on;
box on;
xlim([1 3])

fprintf('  CoV visualization complete\n');

%% =========================================================================
%% === SECTION 8.2: Entropy Visualization ===
%% =========================================================================

fprintf('\n--- Section 8.2: Entropy Visualization ---\n');

% Panel A: Example entropy traces
figure('Name', 'Entropy Analysis');
tiledlayout(2,1);
nexttile;
hold on;

% Show one example trace from each condition
for c = 1:length(conditions)
    condition = conditions{c};

    if isfield(Entropy_Breaks, condition)

        entropyData = Entropy_Breaks.(condition).RecordingEntropy_Raw;
        if ~isempty(entropyData)
            % Search for first valid break across all recordings
            tracePlotted = false;
            for recIdx = 1:length(entropyData)
                if size(entropyData{recIdx}, 1) >= 1 && size(entropyData{recIdx}, 2) > 0
                    trace = entropyData{recIdx}(1, :);
                    % Check if trace has valid data
                    if ~all(isnan(trace)) && length(trace) > 0
                        plot(trace, 'Color', covColors(c, :), ...
                             'LineWidth', 1.5, 'DisplayName', condition);
                        tracePlotted = true;
                        break;
                    end
                end
            end
        end
    end
end

xlabel('Time (frames)');
ylabel('Population Entropy (bits)');
title('Example Entropy Traces');
legend('Location', 'best');
grid on;
box on;

% Panel B: Entropy comparison across conditions
nexttile;
% Collect recording-level mean entropy values
all_entropy_data = [];
group_idx = [];
condition_labels = {};
colors = [];

for c = 1:length(conditions)
    condition = conditions{c};

    if isfield(Entropy_Breaks, condition)
        entropyData = Entropy_Breaks.(condition).RecordingEntropy_Raw;

        if ~isempty(entropyData)
            % This condition has data, assign it the next group index
            current_group_idx = length(condition_labels) + 1;

            % Calculate mean for each recording
            for recIdx = 1:length(entropyData)
                recording_mean = mean(entropyData{recIdx}(:));
                if ~isnan(recording_mean)
                    all_entropy_data = [all_entropy_data; recording_mean];
                    group_idx = [group_idx; current_group_idx];
                end
            end

            % Add condition label and color
            condition_labels{end+1} = condition;
            colors = [colors; covColors(c, :)];
        end
    end
end

% Create daboxplot
if ~isempty(all_entropy_data)
    daboxplot(all_entropy_data, 'groups', group_idx, 'colors', colors, ...
        'xtlabels', condition_labels, 'scatter', 1, 'outliers', 1, ...
        'mean', 1, 'boxalpha', 0.5, 'whiskers', 0);
    ylabel('Mean Entropy (Raw)');
    title('Entropy Across Conditions');
    grid on;
end

fprintf('  Entropy visualization complete\n');

%% =========================================================================
%% === SECTION 8.3: Blob Detection Visualization (2D and 3D Methods) ===
%% =========================================================================

fprintf('\n--- Section 8.3: Blob Detection Visualization ---\n');

figure('Name', 'Blob Detection - 2D Method');
tiledlayout(1,3)
% Panel: Blob rate comparison across conditions
nexttile;
blobRate2D_mean = [];
blobRate2D_sem = [];
blobLabels = {};

for c = 1:length(conditions)
    condition = conditions{c};

    if isfield(BlobDetection_Breaks, condition)
        % Collect all 2D blob rates
        allRates = [];
        for r = 1:length(BlobDetection_Breaks.(condition))
            if isfield(BlobDetection_Breaks.(condition)(r), 'BlobRate2D')
                allRates = [allRates, BlobDetection_Breaks.(condition)(r).BlobRate2D];
            end
        end

        if ~isempty(allRates)
            blobRate2D_mean(end+1) = mean(allRates);
            blobRate2D_sem(end+1) = std(allRates) / sqrt(length(allRates));
            blobLabels{end+1} = condition;
        end
    end
end

if ~isempty(blobRate2D_mean)
    bar(blobRate2D_mean);
    hold on;
    errorbar(1:length(blobRate2D_mean), blobRate2D_mean, blobRate2D_sem, ...
            'k.', 'LineWidth', 1.5);
    xticklabels(blobLabels);
    ylabel('Blob Rate (blobs/frame)');
    title('2D Blob Rate Across Conditions');
    grid on;
end



% Panel: 3D blob count comparison
nexttile;
blobRate3D_mean = [];
blobRate3D_sem = [];

for c = 1:length(conditions)
    condition = conditions{c};

    if isfield(BlobDetection_Breaks, condition)
        % Collect all 3D blob rates
        allRates = [];
        for r = 1:length(BlobDetection_Breaks.(condition))
            if isfield(BlobDetection_Breaks.(condition)(r), 'BlobRate3D')
                allRates = [allRates, BlobDetection_Breaks.(condition)(r).BlobRate3D];
            end
        end

        if ~isempty(allRates)
            blobRate3D_mean(end+1) = mean(allRates);
            blobRate3D_sem(end+1) = std(allRates) / sqrt(length(allRates));
        end
    end
end

if ~isempty(blobCount3D_mean)
    bar(blobCount3D_mean);
    hold on;
    errorbar(1:length(blobCount3D_mean), blobCount3D_mean, blobCount3D_sem, ...
            'k.', 'LineWidth', 1.5);
    xticklabels(blobLabels);
    ylabel('Mean 3D Blob Rate per Break');
    title('3D Spatiotemporal Blob Rate');
    grid on;
end
% 
% % Panel: 3D blob size distribution
% nexttile;
% hold on;
% 
% for c = 1:min(4, length(conditions))
%     condition = conditions{c};
% 
%     if isfield(BlobDetection_Breaks, condition)
%         % Collect all 3D blob sizes
%         allSizes = [];
%         for r = 1:length(BlobDetection_Breaks.(condition))
%             if isfield(BlobDetection_Breaks.(condition)(r), 'Break')
%                 nBreaks = length(BlobDetection_Breaks.(condition)(r).Break);
%                 for b = 1:nBreaks
%                     if ~isempty(BlobDetection_Breaks.(condition)(r).Break{b}) && ...
%                        isfield(BlobDetection_Breaks.(condition)(r).Break{b}, 'method3D')
%                         sizes = BlobDetection_Breaks.(condition)(r).Break{b}.method3D.blobSizes;
%                         allSizes = [allSizes; sizes(:)];
%                     end
%                 end
%             end
%         end
% 
%         % Plot histogram
%         if ~isempty(allSizes) && length(allSizes) > 5
%             [counts, edges] = histcounts(log10(allSizes + 1), 30);
%             centers = (edges(1:end-1) + edges(2:end)) / 2;
%             plot(centers, counts / sum(counts), 'LineWidth', 2, ...
%                 'DisplayName', condition, 'Color', covColors(c, :));
%         end
%     end
% end
% 
% xlabel('log10(Blob Size + 1) [voxels]');
% ylabel('Probability Density');
% title('3D Blob Size Distribution');
% legend('Location', 'best');
% grid on;
% box on;
% 
% 
% % Panel: 2D vs 3D blob count correlation
% nexttile;
% hold on;
% 
% for c = 1:length(conditions)
%     condition = conditions{c};
% 
%     if isfield(BlobDetection_Breaks, condition)
%         blob2D = [];
%         blob3D = [];
% 
%         for r = 1:length(BlobDetection_Breaks.(condition))
%             if isfield(BlobDetection_Breaks.(condition)(r), 'BlobCount2D') && ...
%                isfield(BlobDetection_Breaks.(condition)(r), 'BlobCount3D')
% 
%                 blob2D = [blob2D, BlobDetection_Breaks.(condition)(r).BlobCount2D];
%                 blob3D = [blob3D, BlobDetection_Breaks.(condition)(r).BlobCount3D];
%             end
%         end
% 
%         % Plot scatter
%         if ~isempty(blob2D) && ~isempty(blob3D)
%             scatter(blob2D, blob3D, 50, covColors(c, :), 'filled', ...
%                    'MarkerFaceAlpha', 0.6, 'DisplayName', condition);
%         end
%     end
% end
% 
% % Add unity line
% xlims = xlim;
% ylims = ylim;
% maxVal = max([xlims(2), ylims(2)]);
% plot([0, maxVal], [0, maxVal], 'k--', 'LineWidth', 1, 'DisplayName', 'Unity');
% 
% xlabel('2D Blob Count (total)');
% ylabel('3D Blob Count');
% title('2D vs 3D Method Correlation');
% legend('Location', 'best');
% grid on;
% box on;
% axis equal tight;



% Panel: Blob rate comparison (2D vs 3D methods)
nexttile;
% Collect data for side-by-side comparison (only include conditions with data)
rate2D = [];
rate3D = [];
rateLabels = {};

for c = 1:length(conditions)
    condition = conditions{c};

    if isfield(BlobDetection_Breaks, condition)
        % Collect rates
        rates2D_cond = [];
        rates3D_cond = [];

        for r = 1:length(BlobDetection_Breaks.(condition))
            if isfield(BlobDetection_Breaks.(condition)(r), 'BlobRate2D')
                rates2D_cond = [rates2D_cond, BlobDetection_Breaks.(condition)(r).BlobRate2D];
            end
            if isfield(BlobDetection_Breaks.(condition)(r), 'BlobRate3D')
                rates3D_cond = [rates3D_cond, BlobDetection_Breaks.(condition)(r).BlobRate3D];
            end
        end

        % Only add to arrays if both 2D and 3D data exist
        if ~isempty(rates2D_cond) && ~isempty(rates3D_cond)
            rate2D(end+1, 1) = mean(rates2D_cond);
            rate2D(end, 2) = std(rates2D_cond) / sqrt(length(rates2D_cond));

            rate3D(end+1, 1) = mean(rates3D_cond);
            rate3D(end, 2) = std(rates3D_cond) / sqrt(length(rates3D_cond));

            rateLabels{end+1} = condition;
        end
    end
end

% Create grouped bar plot (only if we have data)
if ~isempty(rate2D) && ~isempty(rate3D)
    x = 1:length(rateLabels);
    width = 0.35;

    hold on;
    bar(x - width/2, rate2D(:, 1), width, 'DisplayName', '2D (frame-by-frame)');
    bar(x + width/2, rate3D(:, 1), width, 'DisplayName', '3D (volume)');
    errorbar(x - width/2, rate2D(:, 1), rate2D(:, 2), 'k.', 'LineWidth', 1);
    errorbar(x + width/2, rate3D(:, 1), rate3D(:, 2), 'k.', 'LineWidth', 1);
    xticks(x);
    xticklabels(rateLabels);
    ylabel('Blob Rate');
    title('Method Comparison: Blob Rates');
    legend('Location', 'best');
    grid on;
end
