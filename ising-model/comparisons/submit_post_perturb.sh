#!/bin/bash
# Run combine → analyze (all new modes) → grid plots after perturbation jobs complete
# Usage: bash submit_post_perturb.sh

SCRIPT_DIR="$HOME/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"
ISING_DATA="/path/to/data/IsingSims"
OUTPUT="/path/to/data/IsingPerturbations/moransI+activity"

# 1. Combine
COMBINE_JOB=$(sbatch --parsable --partition=defaultp --export=NONE --cpus-per-task=4 --mem=64G --time=04:00:00 --job-name="combine_moransI" \
  --output="$OUTPUT/logs/combine_%j.out" --error="$OUTPUT/logs/combine_%j.err" \
  "$SCRIPT_DIR/run_ising_pipeline.sh" combine --metric moransI+activity --frame-label full_trial \
  --output "$OUTPUT" --ising-data "$ISING_DATA" --n-replicates 150)
echo "Combine job: $COMBINE_JOB"

# 2. Analyze for each new mode (depends on combine)
for mode in double_pulse3 double_pulse5 double_pulse10; do
  ANALYZE_JOB=$(sbatch --parsable --dependency=afterok:$COMBINE_JOB \
    --partition=defaultp --constraint=matlab --export=NONE --cpus-per-task=4 --mem=96G --time=04:00:00 \
    --job-name="analyze_${mode}" \
    --output="$OUTPUT/logs/analyze_${mode}_%j.out" --error="$OUTPUT/logs/analyze_${mode}_%j.err" \
    "$SCRIPT_DIR/run_ising_pipeline.sh" analyze --metric moransI+activity --stim-mode "$mode")
  echo "Analyze $mode: $ANALYZE_JOB (depends on $COMBINE_JOB)"
done

# 3. Grid plots (depends on all analyze jobs finishing — use afterany on last one)
GRID_JOB=$(sbatch --parsable --dependency=afterany:$ANALYZE_JOB \
  --partition=defaultp --export=NONE --cpus-per-task=1 --mem=8G --time=00:10:00 \
  --job-name="gridplot" \
  "$HOME/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure1/plots/slurm_isingPerturbationPlotGrid.sh" moransI+activity)
echo "Grid plot discovery: $GRID_JOB (depends on last analyze)"
