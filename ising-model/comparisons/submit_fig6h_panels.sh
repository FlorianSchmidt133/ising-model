#!/bin/bash
#SBATCH --job-name=fig6h_panels
#SBATCH --partition=defaultp
#SBATCH --constraint=matlab
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=fig6h_panels_%j.out
#SBATCH --error=fig6h_panels_%j.err
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# =============================================================================
# Generate Figure 6h (and the new EC50 panel variants) using the bias-matched
# pipeline. Reads:
#   $HOME/IsingPerturbations/moransI+activity/BiasMatchExperiment/size_<N>/matcher_output_size<N>.mat
#   $HOME/IsingPerturbations/moransI+activity/double_pulse_bias10_<bv>/Analysis/AllDurationAnalysis.mat
# Writes:
#   $HOME/IsingPerturbations/moransI+activity/Fig6h_BiasMatched_EC50/{AllDurationAnalysis.mat,...png}
# =============================================================================

PERTURB_DIR="$HOME/IsingPerturbations/moransI+activity"
SCRIPT_DIR="$HOME/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"

# Positional args:
#   1: BiasMatchSize     (default 2)
#   2: BiasMatchWindow   (default 'full')
#   3: MatchingMode      (default 'perCondition')
BMS="${1:-2}"
BMW="${2:-full}"
MATCHING_MODE="${3:-perCondition}"

MATCHER_DIR="$PERTURB_DIR/BiasMatched/${MATCHING_MODE}/Matcher"
OUTPUT_DIR="$PERTURB_DIR/BiasMatched/${MATCHING_MODE}/Fig6h"
OUT_SUB="$OUTPUT_DIR/size_${BMS}/window_${BMW}"
mkdir -p "$OUT_SUB"

if [ ! -f "$SCRIPT_DIR/Figure6h_BiasMatched_EC50.m" ]; then
    echo "ERROR: Figure6h_BiasMatched_EC50.m not on cluster yet" >&2
    exit 1
fi

echo "MatchingMode = $MATCHING_MODE"
echo "Matcher dir  = $MATCHER_DIR"
echo "Output sub   = $OUT_SUB"

module load matlab

srun matlab -batch "\
    cd('$SCRIPT_DIR'); \
    out = Figure6h_BiasMatched_EC50( \
        'BiasMatcherDir',      '$MATCHER_DIR', \
        'BiasMatchSize',        $BMS, \
        'BiasMatchWindow',      '$BMW', \
        'PerturbAnalysisRoot', '$PERTURB_DIR', \
        'OutputDir',           '$OUT_SUB', \
        'MatchingMode',        '$MATCHING_MODE'); \
    save(fullfile('$OUT_SUB', 'fig6h_output.mat'), 'out', '-v7.3'); \
    fprintf('Done.\\n');"

RC=$?
echo "matlab exit code: $RC"
exit $RC
