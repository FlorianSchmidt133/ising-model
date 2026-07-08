#!/bin/bash
#SBATCH --job-name=ParamCorrFix
#SBATCH --output=logs/parameter_correlation_fix_%j.out
#SBATCH --error=logs/parameter_correlation_fix_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=defaultp
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# =============================================================================
# Regenerate parameter_correlation figure only (Illustrator color fix)
# =============================================================================
#
# Usage:
#   sbatch run_parameter_correlation_fix.sh
#
# =============================================================================

module purge
module load python/3.11
export OMP_NUM_THREADS=1

SCRIPT_DIR="/path/to/data/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"
cd "$SCRIPT_DIR"
mkdir -p logs

ISING_DATA="/path/to/data/IsingSims"

METRICS=("spatial+persistence" "moransI+activity")

for METRIC in "${METRICS[@]}"; do
    # Try full_trial results file, then fallback without frame prefix
    RESULTS_FILE="$ISING_DATA/IsingComparison/IsingComparison_Results_full_trial_subselect_centre_vs_tiled_${METRIC}_optimized.mat"
    if [ ! -f "$RESULTS_FILE" ]; then
        RESULTS_FILE="$ISING_DATA/IsingComparison/$METRIC/IsingComparison_Results_full_trial_subselect_centre_vs_tiled_${METRIC}_optimized.mat"
    fi
    if [ ! -f "$RESULTS_FILE" ]; then
        RESULTS_FILE="$ISING_DATA/IsingComparison/IsingComparison_Results_subselect_centre_vs_tiled_${METRIC}_optimized.mat"
    fi
    if [ ! -f "$RESULTS_FILE" ]; then
        echo "WARNING: No results file found for metric=$METRIC, skipping"
        continue
    fi

    echo "=============================================="
    echo "Regenerating parameter_correlation for: $METRIC"
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

# Load only the Comparison group (plot_parameter_correlation needs nothing else)
with h5py.File(results_file, 'r') as f:
    if 'Results' in f and 'Comparison' in f['Results']:
        comp = load_group(f['Results/Comparison'])
    elif 'Comparison' in f:
        comp = load_group(f['Comparison'])
    else:
        raise KeyError('No Comparison group found in ' + results_file)

results = {'Comparison': comp}
config = {'conditions': ['Naive', 'Beginner', 'Expert', 'NoSpout']}

viz = IsingVisualizer(results, config, output_dir, frame_label='full_trial')

base_conds = ['Naive', 'Beginner', 'Expert', 'NoSpout']
subfolder = os.path.join('parameters', 'full_trial_base')
print(f'  Generating: {subfolder}/parameter_correlation')
viz._run_category('parameters', ['plot_parameter_correlation'], base_conds, subfolder)

print(f'Done. Figures in: {output_dir}')
gc.collect()
"

    RC=$?
    if [ $RC -ne 0 ]; then
        echo "ERROR: Failed for metric=$METRIC (exit code $RC)"
    fi
done

echo ""
echo "All done at $(date)"
