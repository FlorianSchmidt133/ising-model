#!/bin/bash
#SBATCH --job-name=Fig5_Ising
#SBATCH --output=Fig5_Ising_%j.out
#SBATCH --error=Fig5_Ising_%j.err
#SBATCH --partition=defaultp
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # 32 CPUs for parfor parallel workers
#SBATCH --mem=512G                  # 512GB RAM for parallel processing (increased from 256GB)
#SBATCH --time=1-00:00:00           # 1 day (running from NFS without scratch)
#SBATCH --constraint=matlab         # Ensure node has MATLAB available
#SBATCH --mail-user=your.email@example.com
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# =============================================================================
# Figure 5: Ising Model vs Experimental Data Comparison
# SLURM submission script for ISTA HPC cluster (PARALLEL VERSION)
# =============================================================================
#
# Resources:
#   32 CPUs for MATLAB parfor parallel workers
#   256GB RAM for parallel data loading
#   Runs directly from NFS (scratch not available on cluster)
#
# Usage:
#   sbatch slurm_figure5_ising.sh
#
# Monitor:
#   squeue -u $USER
#
# Output:
#   /path/to/data/IsingSims/IsingComparison/
#
# =============================================================================

echo "========================================"
echo "Figure 5: Ising Model Comparison (PARALLEL)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK (for parfor workers)"
echo "Memory: 512G"
echo "Start time: $(date)"
echo "========================================"

# Load MATLAB module
echo "Loading MATLAB module..."
module load matlab

# Define paths
REPO_PATH="$HOME/git/MouseBrainActivity"
HELPER_PATH="$REPO_PATH/Helper Functions and Packages"
SCRIPT_PATH="$REPO_PATH/Neuron Activity Analysis/main_scripts/Figure5/comparisons"

echo "Repository: $REPO_PATH"
echo "Script path: $SCRIPT_PATH"
echo ""

# Check that script exists
if [ ! -f "$SCRIPT_PATH/Figure5_IsingComparison_cluster.m" ]; then
    echo "ERROR: Script not found: $SCRIPT_PATH/Figure5_IsingComparison_cluster.m"
    exit 1
fi

# Run MATLAB in batch mode
echo "Starting MATLAB with $SLURM_CPUS_PER_TASK parallel workers..."
matlab -nodisplay -nosplash -batch "\
    addpath(genpath('$HELPER_PATH')); \
    cd('$SCRIPT_PATH'); \
    maxNumCompThreads($SLURM_CPUS_PER_TASK); \
    Figure5_IsingComparison_cluster"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
