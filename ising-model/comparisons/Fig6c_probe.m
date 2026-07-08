function Fig6c_probe()
%FIG6C_PROBE  Diagnostic: dump dims + sample values from matcher_output_size3.mat
%   to figure out why averaged traces look flat.
    f = 'Paper/K0_25-50-25_perCondition_double_pulse_bias10/_matcher_mats/matcher_output_size3.mat';
    S = load(f, 'out');
    if isfield(S.out, 'bySize') && isfield(S.out.bySize, 'size_3')
        mo = S.out.bySize.size_3;
    else
        mo = S.out;
    end
    fprintf('Fields in mo:\n');
    disp(fieldnames(mo));

    fprintf('\nsize(simTracesCondSim): %s\n', mat2str(size(mo.simTracesCondSim)));
    fprintf('size(simTraces): %s\n', mat2str(size(mo.simTraces)));
    fprintf('size(simTimeAxis): %s\n', mat2str(size(mo.simTimeAxis)));
    fprintf('simTimeAxis range: %.2f to %.2f s, n=%d\n', ...
        mo.simTimeAxis(1), mo.simTimeAxis(end), numel(mo.simTimeAxis));

    fprintf('\ninfo.stimulusBiasValues:\n');
    disp(mo.info.stimulusBiasValues(:)');
    fprintf('info.stimulusSizes: %s\n', mat2str(mo.info.stimulusSizes(:)'));
    fprintf('info.stimulusDurations: %s\n', mat2str(mo.info.stimulusDurations(:)'));

    if isfield(mo, 'info') && isfield(mo.info, 'preStimFrames')
        fprintf('preStimFrames: %d, postStimFrames: %d, totalFrames: %d\n', ...
            mo.info.preStimFrames, mo.info.postStimFrames, ...
            numel(mo.simTimeAxis));
    end

    if isfield(mo, 'bestBiasPerCondition')
        fprintf('\nbestBiasPerCondition:\n');
        bbpc = mo.bestBiasPerCondition;
        condNames = fieldnames(bbpc);
        for k = 1:numel(condNames)
            cond = condNames{k};
            wNames = fieldnames(bbpc.(cond));
            fprintf('  %s: ', cond);
            for w = 1:numel(wNames)
                wn = wNames{w};
                bv = bbpc.(cond).(wn).bias_value;
                if isstring(bv) || ischar(bv), bv = NaN; end
                fprintf('%s=%.2f  ', wn, bv);
                if w >= 4, fprintf('...'); break; end
            end
            fprintf('\n');
        end
    end

    % Probe simTracesCondSim values per condition at each bias for STIM PERIOD
    biases = double(mo.info.stimulusBiasValues(:));
    tAxis = mo.simTimeAxis(:)';
    stimMask    = tAxis >= 0 & tAxis <= 2.5;
    baseMask    = tAxis >= -5 & tAxis < 0;
    nC = size(mo.simTracesCondSim, 1);
    nS = size(mo.simTracesCondSim, 2);
    nB = size(mo.simTracesCondSim, 3);
    condNames = {'Naive', 'Beginner', 'Expert', 'NoSpout'};
    fprintf('\n=== Stim-period mean (top-10 sim pool) per (cond, bias) ===\n');
    fprintf('  %-10s ', 'bias');
    for c = 1:nC, fprintf('  %-10s', condNames{min(c, numel(condNames))}); end
    fprintf('\n');
    nTop = min(10, nS);
    for b = 1:nB
        fprintf('  %-10.3f ', biases(b));
        for c = 1:nC
            perSim = squeeze(mo.simTracesCondSim(c, 1:nTop, b, :));  % [10 x nFrames]
            if all(isnan(perSim(:)))
                fprintf('  %-10s', 'NaN');
            else
                muTrace = mean(perSim, 1, 'omitnan');
                stimVal = mean(muTrace(stimMask), 'omitnan');
                baseVal = mean(muTrace(baseMask), 'omitnan');
                fprintf('  s%.3f/b%.3f', stimVal, baseVal);
            end
        end
        fprintf('\n');
    end

    % Peak (max in stim window) per (cond, bias)
    fprintf('\n=== Stim-period PEAK (top-10 sim pool) per (cond, bias) ===\n');
    fprintf('  %-10s ', 'bias');
    for c = 1:nC, fprintf('  %-10s', condNames{min(c, numel(condNames))}); end
    fprintf('\n');
    for b = 1:nB
        fprintf('  %-10.3f ', biases(b));
        for c = 1:nC
            perSim = squeeze(mo.simTracesCondSim(c, 1:nTop, b, :));
            if all(isnan(perSim(:)))
                fprintf('  %-10s', 'NaN');
            else
                muTrace = mean(perSim, 1, 'omitnan');
                pk = max(muTrace(stimMask));
                fprintf('  %-10.4f', pk);
            end
        end
        fprintf('\n');
    end
end
