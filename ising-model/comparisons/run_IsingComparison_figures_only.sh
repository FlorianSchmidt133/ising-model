#!/bin/bash
#SBATCH --job-name=IsingFigs
#SBATCH --output=logs/IsingComparison_figures_%j.out
#SBATCH --error=logs/IsingComparison_figures_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --partition=defaultp
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# =============================================================================
# Ising Model Comparison - Figures Only Mode
# =============================================================================
#
# Regenerate figures from existing results file without re-running simulations.
# Uses reduced resources since only UMAP and plotting is needed.
#
# Usage:
#   sbatch run_IsingComparison_figures_only.sh RESULTS_FILE [METRIC]
#
# Arguments:
#   RESULTS_FILE - Path to existing results file (.mat or .pkl) (required)
#   METRIC       - Matching metric used: moransI, activity, spatial+persistence, combined
#                  Default: spatial+persistence (needed for correct data paths)
#
# Example:
#   sbatch run_IsingComparison_figures_only.sh /path/to/data/IsingSims/IsingComparison/IsingComparison_Results_subselect_centre_vs_tiled_spatial+persistence_optimized.mat
#
# =============================================================================

# Check for required argument
if [ -z "$1" ]; then
    echo "ERROR: Results file path is required"
    echo "Usage: sbatch run_IsingComparison_figures_only.sh RESULTS_FILE [METRIC]"
    exit 1
fi

# Parse arguments
RESULTS_FILE=$1
METRIC=${2:-spatial+persistence}

echo "=============================================="
echo "Ising Model Comparison - Figures Only Mode"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Results file: $RESULTS_FILE"
echo "Metric: $METRIC"
echo "Start time: $(date)"
echo "=============================================="

# Set up environment
echo ""
echo "Setting up environment..."

# Load required modules
module purge
module load python/3.11

# Navigate to script directory (hardcoded for SLURM compatibility)
cd /path/to/data/git/MouseBrainActivity/Neuron\ Activity\ Analysis/main_scripts/Figure5/comparisons

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Working directory: $(pwd)"

# Check if results file exists
if [ ! -f "$RESULTS_FILE" ]; then
    echo "ERROR: Results file not found: $RESULTS_FILE"
    exit 1
fi

# Environment settings - UMAP benefits from some parallelism
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMBA_NUM_THREADS=8
export NUMBA_THREADING_LAYER=omp

# =============================================================================
# FIGURE GENERATION
# =============================================================================

echo ""
echo "=============================================="
echo "Generating figures from results"
echo "=============================================="
echo "Start time: $(date)"

srun python Figure5_IsingComparison_optimized.py \
    --figures-only "$RESULTS_FILE" \
    --metric "$METRIC"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================="

# Print job statistics
if command -v sacct &> /dev/null; then
    echo ""
    echo "Job statistics:"
    sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize,State
fi

exit $EXIT_CODE
