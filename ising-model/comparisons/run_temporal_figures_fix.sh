#!/bin/bash
#SBATCH --job-name=TemporalFix
#SBATCH --output=logs/temporal_figures_fix_%j.out
#SBATCH --error=logs/temporal_figures_fix_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=defaultp
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# =============================================================================
# Regenerate temporal figures only (timeseries NaN fix)
# =============================================================================
#
# Usage:
#   sbatch run_temporal_figures_fix.sh
#
# =============================================================================

module purge
module load python/3.11
export OMP_NUM_THREADS=1

SCRIPT_DIR="/path/to/data/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"
cd "$SCRIPT_DIR"
mkdir -p logs

ISING_DATA="/path/to/data/IsingSims"
EXP_DATA="/path/to/data/ExperimentalData/ExperimentalData.mat"

METRICS=("spatial+persistence" "moransI+activity")

for METRIC in "${METRICS[@]}"; do
for FRAME_LABEL in prestim nostim full_trial; do
    RESULTS_FILE="$ISING_DATA/IsingComparison/$METRIC/IsingComparison_Results_${FRAME_LABEL}_subselect_centre_vs_tiled_${METRIC}_optimized.mat"

    if [ ! -f "$RESULTS_FILE" ]; then
        echo "WARNING: No results file for $FRAME_LABEL, skipping"
        continue
    fi

    echo "=============================================="
    echo "Regenerating temporal figures: $FRAME_LABEL"
    echo "Results file: $RESULTS_FILE"
    echo "=============================================="

    srun python -c "
import sys, os, gc
sys.path.insert(0, '.')
import h5py
import numpy as np
from ising_visualizations import IsingVisualizer

results_file = '$RESULTS_FILE'
metric = '$METRIC'
frame_label = '$FRAME_LABEL'
exp_data_path = '$EXP_DATA'
output_dir = os.path.join('$ISING_DATA', 'IsingComparison', metric, 'figures')

def load_group(group):
    d = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            d[key] = load_group(item)
        else:
            d[key] = item[()]
    return d

# Load only the groups needed for temporal plots
with h5py.File(results_file, 'r') as f:
    root = f['Results'] if 'Results' in f else f
    results = {}
    for key in ['Comparison', 'IsingData']:
        if key in root:
            print(f'  Loading {key}...')
            results[key] = load_group(root[key])

config = {'conditions': ['Naive', 'Beginner', 'Expert', 'NoSpout']}
viz = IsingVisualizer(results, config, output_dir,
                      exp_data_path=exp_data_path,
                      frame_label=frame_label)

base_conds = ['Naive', 'Beginner', 'Expert', 'NoSpout']

# Full trial variant
subfolder = os.path.join('temporal', '${FRAME_LABEL}_base')
print(f'  Generating: {subfolder} (full trial)')
viz._current_subfolder = subfolder
viz.conditions = base_conds
viz.plot_timeseries_comparison()

# Pre-stimulus variant (first 80 frames)
subfolder_pre = os.path.join('temporal', 'prestim_base')
print(f'  Generating: {subfolder_pre} (prestim, 80 frames)')
viz._current_subfolder = subfolder_pre
viz.plot_timeseries_comparison(max_frames=80)

print('Done.')
gc.collect()
"

    RC=$?
    if [ $RC -ne 0 ]; then
        echo "ERROR: Failed for $METRIC / $FRAME_LABEL (exit code $RC)"
    fi
done
done

echo ""
echo "All done at $(date)"
