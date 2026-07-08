#!/bin/bash
#SBATCH --job-name=fig6g_panels
#SBATCH --partition=defaultp
#SBATCH --constraint=matlab
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=fig6g_panels_%j.out
#SBATCH --error=fig6g_panels_%j.err
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# =============================================================================
# Generate Figure 6g (Model side) bias-matched connected-component swarm.
# Reads:
#   $HOME/IsingPerturbations/moransI+activity/BiasMatchExperiment/size_<N>/matcher_output_size<N>.mat
#   $HOME/IsingPerturbations/moransI+activity/PerturbationResults_double_pulse_bias10_*.mat
#   $HOME/IsingPerturbations/moransI+activity/PerturbationResults_double_pulse10_*.mat
# Writes:
#   $HOME/IsingPerturbations/moransI+activity/Fig6g_BiasMatched_ConnectedSwarm/size_<N>/window_<wn>/Fig6g_*.png
# =============================================================================

PERTURB_DIR="$HOME/IsingPerturbations/moransI+activity"
SCRIPT_DIR="$HOME/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"

# Optional 1st positional arg = MatchingMode (perCondition | global | expert)
MATCHING_MODE="${1:-perCondition}"
MATCHER_DIR="$PERTURB_DIR/BiasMatched/${MATCHING_MODE}/Matcher"
OUTPUT_DIR="$PERTURB_DIR/BiasMatched/${MATCHING_MODE}/Fig6g"

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$SCRIPT_DIR/Figure6g_BiasMatched_ConnectedSwarm.m" ]; then
    echo "ERROR: Figure6g_BiasMatched_ConnectedSwarm.m not on cluster yet" >&2
    exit 1
fi

echo "MatchingMode = $MATCHING_MODE"
echo "Matcher dir  = $MATCHER_DIR"
echo "Output dir   = $OUTPUT_DIR"

module load matlab

srun matlab -batch "\
    cd('$SCRIPT_DIR'); \
    out = Figure6g_BiasMatched_ConnectedSwarm( \
        'MatcherOutputDir', '$MATCHER_DIR', \
        'PerturbationResultsPath', '$PERTURB_DIR', \
        'OutputDir', '$OUTPUT_DIR', \
        'MatchingMode', '$MATCHING_MODE'); \
    save(fullfile('$OUTPUT_DIR', 'fig6g_output.mat'), 'out', '-v7.3'); \
    fprintf('Done.\\n');"

RC=$?
echo "matlab exit code: $RC"
exit $RC
