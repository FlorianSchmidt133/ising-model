#!/bin/bash
# Submit perturbation jobs with 150 replicates for double_pulse10
# Usage: bash submit_perturb_150.sh

SCRIPT_DIR="$HOME/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"
ISING_DATA="/path/to/data/IsingSims"
COMPARISON="$ISING_DATA/IsingComparison/moransI+activity/IsingComparison_Results_full_trial_subselect_centre_vs_tiled_moransI+activity_optimized.mat"
OUTPUT="/path/to/data/IsingPerturbations/moransI+activity"

TOTAL=$(python3 "$SCRIPT_DIR/run_ising_perturbations.py" --comparison "$COMPARISON" --ising-data "$ISING_DATA" --count-jobs | tail -1)
ARRAY_MAX=$(( (TOTAL + 10 - 1) / 10 - 1 ))
echo "Total jobs: $TOTAL, Array range: 0-$ARRAY_MAX"

sbatch --partition=defaultp --export=NONE --array=0-$ARRAY_MAX --cpus-per-task=1 --mem=2G --time=08:00:00 --job-name="perturb_dp10_150" \
  "$SCRIPT_DIR/run_ising_pipeline.sh" \
  perturbations --metric moransI+activity --frame-label full_trial \
  --output "$OUTPUT" --ising-data "$ISING_DATA" --n-replicates 150 --force
