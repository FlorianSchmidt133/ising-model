#!/bin/bash
#SBATCH --job-name=ising_perturb
#SBATCH --array=0-23039
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/perturb_%A_%a.out
#SBATCH --error=logs/perturb_%A_%a.err
#SBATCH --no-requeue
#SBATCH --mail-user=your.email@example.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# =============================================================================
# Ising Model Perturbation Experiments - SLURM Array Job
# =============================================================================
# This script runs perturbation experiments on the IST Austria cluster.
# Each array task runs one job (condition × simulation × size × duration × mode).
#
# Total jobs: 4 conditions × 10 sims × 8 sizes × 9 durations × (2 + 6 bias values) = 23040 jobs
#
# Usage:
#   cd /path/to/data/git/MouseBrainActivity/Neuron\ Activity\ Analysis/main_scripts/Figure5/comparisons
#   mkdir -p logs
#   sbatch run_ising_perturbations_cluster.sh
#
# Monitor:
#   squeue -u $USER
#   sacct -j <jobid>
#
# Cancel:
#   scancel <jobid>
#   scancel -u $USER -n ising_perturb
# =============================================================================

# Configuration
SCRIPT_DIR="$HOME/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"
OUTPUT_DIR="$HOME/IsingPerturbations"
ISING_DATA="$HOME/IsingSims"

# For single-CPU jobs use single thread
export OMP_NUM_THREADS=1

# Load Python with scientific packages (numpy, scipy, numba)
module load python/3.11

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Print job info
echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Python: $(which python)"
echo "========================================"

# Run the perturbation experiment
cd "$SCRIPT_DIR"
srun --cpu_bind=verbose python run_ising_perturbations.py \
    --output "$OUTPUT_DIR" \
    --ising-data "$ISING_DATA" \
    --index $SLURM_ARRAY_TASK_ID \
    "$@"

# Report completion
echo "========================================"
echo "End time: $(date)"
echo "Exit code: $?"
echo "========================================"
