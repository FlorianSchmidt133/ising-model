#!/bin/bash
#SBATCH --job-name=ising_sim
#SBATCH --array=0-65771
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ising_%A_%a.out
#SBATCH --error=logs/ising_%A_%a.err
#SBATCH --no-requeue
#SBATCH --mail-user=your.email@example.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# =============================================================================
# SLURM Array Job for Ising Model Simulations (Fine Beta Grid)
# =============================================================================
# Runs 24624 simulations in parallel on ISTA cluster
#
# Parameter grid (19 x 9 x 9 x 4 x 4 = 24624 combinations):
#   - beta: [0.4, 0.45, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.65, 0.7, 0.8]
#   - c: [1, 2, 3, 4, 5, 6, 7, 8, 9]
#   - decay_const: [2, 4, 5, 6, 7, 8, 9, 10, 11]
#   - rad: [2, 4, 9, 13]
#   - bias: [-1, -0.8, -0.6, -0.4]
#
# Each simulation: 100K timesteps + 2K burn-in on 39x78 grid
# Estimated time per simulation: ~90 seconds (with Numba JIT)
# Estimated memory: ~3GB peak
#
# Usage:
#   cd /path/to/data/git/MouseBrainActivity/Neuron\ Activity\ Analysis/main_scripts/Figure5
#   mkdir -p logs
#   sbatch run_ising_slurm.sh [rook|queen]
#
# Monitor:
#   squeue -u $USER
#   sacct -j <jobid>
#
# Cancel:
#   scancel <jobid>
#   scancel -u $USER  # all jobs
#
# =============================================================================

# Configuration
# Usage: sbatch run_ising_slurm.sh [rook|queen] [REFRACTORY_K]
#   Examples:
#     sbatch run_ising_slurm.sh                  # rook, K=0  -> IsingSims/
#     sbatch run_ising_slurm.sh queen            # queen, K=0 -> IsingSims_queen/
#     sbatch run_ising_slurm.sh rook 10          # rook, K=10 -> IsingSims_refractoryK10/
#     sbatch run_ising_slurm.sh queen 10         # queen,K=10 -> IsingSims_queen_refractoryK10/
CONNECTIVITY="${1:-rook}"
REFRACTORY="${2:-0}"
SCRIPT_DIR="/path/to/data/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5"

# Compose output dir from connectivity + refractory marker
OUTPUT_DIR="/path/to/data/IsingSims"
if [ "$CONNECTIVITY" = "queen" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_queen"
fi
if [ "$REFRACTORY" -gt 0 ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_refractoryK${REFRACTORY}"
fi

# For single-CPU jobs use single thread
export OMP_NUM_THREADS=1

# Load Python with scientific packages (numpy, scipy, numba)
module load python/3.11

# Print job info
echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Connectivity: $CONNECTIVITY"
echo "Refractory K: $REFRACTORY"
echo "Output dir: $OUTPUT_DIR"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Python: $(which python)"
echo "========================================"

# Run single simulation with srun
cd "$SCRIPT_DIR"
srun --cpu_bind=verbose python generate_all_ising_simulations.py \
    --output "$OUTPUT_DIR" \
    --index $SLURM_ARRAY_TASK_ID \
    --connectivity "$CONNECTIVITY" \
    --refractory "$REFRACTORY"

# Report completion
echo "========================================"
echo "End time: $(date)"
echo "Exit code: $?"
echo "========================================"
