#!/bin/bash
#SBATCH --job-name=match_dp10_bias
#SBATCH --partition=defaultp
#SBATCH --constraint=matlab
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=match_dp10_bias_%j.out
#SBATCH --error=match_dp10_bias_%j.err
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# =============================================================================
# Run match_experimental_to_isingBias on the cluster.
# Inputs:
#   $HOME/IsingPerturbations/moransI+activity/ExperimentalData.mat       (uploaded once)
#   $HOME/IsingPerturbations/moransI+activity/PerturbationResults_double_pulse_bias10_*.mat
# Outputs:
#   $HOME/IsingPerturbations/moransI+activity/BiasMatchExperiment/*.png + .csv
# =============================================================================

PERTURB_DIR="$HOME/IsingPerturbations/moransI+activity"
EXP_DATA="$PERTURB_DIR/ExperimentalData.mat"
SCRIPT_DIR="$HOME/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"

# Optional 2nd positional arg = MatchingMode (perCondition | global | expert).
# Output dir gets a per-mode suffix so the three modes don't collide.
MATCHING_MODE="${2:-perCondition}"
OUTPUT_DIR="$PERTURB_DIR/BiasMatched/${MATCHING_MODE}/Matcher"

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$EXP_DATA" ]; then
    echo "ERROR: ExperimentalData.mat not found at $EXP_DATA" >&2
    echo "Upload it first:  scp ExperimentalData.mat user@cluster:$EXP_DATA" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/match_experimental_to_isingBias.m" ]; then
    echo "ERROR: match_experimental_to_isingBias.m not found at $SCRIPT_DIR" >&2
    echo "git pull on cluster, then resubmit" >&2
    exit 1
fi

# Optional first arg = single stimulus size. When provided, the matcher
# only processes that one size (writes to OutputDir/size_<N>/). Useful for
# parallel-by-size submission:
#   for sz in 2 3 4; do sbatch submit_match_dp10_bias.sh $sz; done
STIMULUS_SIZE="${1:-}"
SIZE_ARG=""
if [ -n "$STIMULUS_SIZE" ]; then
    SIZE_ARG=", 'StimulusSize', $STIMULUS_SIZE"
    echo "Running for SINGLE size = $STIMULUS_SIZE"
else
    echo "Running default StimulusSizes loop ([2, 3, 4])"
fi

# BaselineAlign toggle: 3rd positional arg (default true). Shifts the
# experimental trace by (model baseline - exp baseline) so both start at
# the same baseline. Pass "false" as 3rd arg to recover the un-aligned
# (legacy) comparison. Positional rather than env-var because
# #SBATCH --export=NONE wipes env vars on the compute node.
BASELINE_ALIGN="${3:-true}"
echo "BaselineAlign = $BASELINE_ALIGN"

module load matlab

srun matlab -batch "\
    cd('$SCRIPT_DIR'); \
    out = match_experimental_to_isingBias( \
        'ExperimentalDataFile', '$EXP_DATA', \
        'PerturbationResultsPath', '$PERTURB_DIR', \
        'OutputDir', '$OUTPUT_DIR', \
        'Mode', 'double_pulse_bias10', \
        'MatchingMode', '$MATCHING_MODE', \
        'BaselineAlign', $BASELINE_ALIGN$SIZE_ARG); \
    save(fullfile('$OUTPUT_DIR', 'matcher_output_size${STIMULUS_SIZE:-all}.mat'), 'out', '-v7.3'); \
    fprintf('Best bias per (condition, window):\\n'); \
    if isfield(out, 'bySize'); \
        fnsSize = fieldnames(out.bySize); \
        for ks = 1:numel(fnsSize); \
            so = out.bySize.(fnsSize{ks}); \
            if ~isfield(so, 'bestBiasPerCondition'); continue; end; \
            fprintf('--- %s ---\\n', fnsSize{ks}); \
            fns = fieldnames(so.bestBiasPerCondition); \
            for i = 1:numel(fns); \
                bb = so.bestBiasPerCondition.(fns{i}); \
                if isfield(bb, 'full'); \
                    fprintf('  %-10s full=%.2f onset=%.2f offset=%.2f\\n', fns{i}, \
                        bb.full.bias_value, bb.onset.bias_value, bb.offset.bias_value); \
                end; \
            end; \
        end; \
    end"

RC=$?
echo "matlab exit code: $RC"
exit $RC
