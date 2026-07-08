#!/bin/bash
#SBATCH --job-name=fig6f_panels
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=fig6f_panels_%j.out
#SBATCH --error=fig6f_panels_%j.err
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# =============================================================================
# Pull Fig6f snapshot files into BiasMatched/<mode>/Fig6f/.
# Reads:
#   $HOME/IsingPerturbations/moransI+activity/BiasMatched/<mode>/Matcher/size_<N>/BiasMatch_Summary.csv
#   $HOME/IsingPerturbations/moransI+activity/double_pulse_bias10_<bv>/PerturbationSnapshots/dur_59/Snapshots/*.png
# Writes:
#   $HOME/IsingPerturbations/moransI+activity/BiasMatched/<mode>/Fig6f/size_<N>/window_<wn>/{snapshot_*.png, README.txt}
#   $HOME/IsingPerturbations/moransI+activity/BiasMatched/<mode>/Fig6f/Fig6f_PullSummary.csv
# =============================================================================

SCRIPT_DIR="$HOME/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"

MATCHING_MODE="${1:-perCondition}"

if [ ! -f "$SCRIPT_DIR/fig6f_pull_snapshots.sh" ]; then
    echo "ERROR: fig6f_pull_snapshots.sh not on cluster yet" >&2
    exit 1
fi

echo "MatchingMode = $MATCHING_MODE"

srun bash "$SCRIPT_DIR/fig6f_pull_snapshots.sh" "$MATCHING_MODE"

RC=$?
echo "exit code: $RC"
exit $RC
