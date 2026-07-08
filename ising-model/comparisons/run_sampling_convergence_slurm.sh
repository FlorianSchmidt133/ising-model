#!/bin/bash
#SBATCH --job-name=matlab_samp
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/matlab_sampling_%j.out
#SBATCH --error=logs/matlab_sampling_%j.err
#SBATCH --mail-user=your.email@example.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --no-requeue
#SBATCH --constraint=matlab
#SBATCH --partition=defaultp
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# =============================================================================
# SLURM Job for MATLAB Sampling Convergence Analysis
# =============================================================================
#
# RESOURCE JUSTIFICATION:
#
# --time=2-00:00:00 (2 days)
#   - Processing 2500 simulations × 100K frames each
#   - Each simulation requires computing Moran's I for every frame
#   - Moran's I involves matrix operations on weight matrices
#   - Estimate: ~30-60 sec per simulation × 2500 = 20-40 hours
#   - 2 days provides safety margin for I/O overhead
#
# --mem=64G
#   - Each simulation file: ~300MB (100K × 39 × 78 × int8)
#   - Need to hold: stored_spins + Moran's I arrays + weight matrices
#   - Peak memory per sim: ~1-2GB during processing
#   - MATLAB overhead + workspace: ~2-4GB
#   - 64G provides headroom for MATLAB's memory management
#
# --cpus-per-task=4
#   - MATLAB can parallelize some matrix operations internally
#   - Not using parfor (script is sequential), but BLAS/LAPACK benefit
#   - More CPUs = faster matrix operations in mL_moransI
#   - 4 is a reasonable balance (not hogging resources)
#
# --no-requeue
#   - Script has no checkpointing; restart would lose all progress
#   - Better to fail and investigate than restart from scratch
#
# USAGE:
#   cd /path/to/data/git/MouseBrainActivity/Neuron\ Activity\ Analysis/main_scripts/Figure5/comparisons
#   mkdir -p logs
#   sbatch run_sampling_convergence_slurm.sh
#
# MONITOR:
#   squeue -u $USER
#   tail -f logs/matlab_sampling_*.out
#
# =============================================================================

# Configuration
SCRIPT_DIR="/path/to/data/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"

# Set threading for MATLAB's internal parallelization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load MATLAB module
module load matlab

# Print job info
echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Start time: $(date)"
echo "Script dir: $SCRIPT_DIR"
echo "========================================"

# Change to script directory (so relative paths work)
cd "$SCRIPT_DIR"

# Run MATLAB script
# -nodisplay: No GUI (required for cluster)
# -nosplash: Skip splash screen
# -batch: Preferred over -r for non-interactive scripts (cleaner exit handling)
srun matlab -nodisplay -nosplash -batch "Figure5_SamplingConvergenceAnalysis"

# Capture exit code
EXIT_CODE=$?

echo "========================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
