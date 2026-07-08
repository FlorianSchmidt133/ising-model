#!/bin/bash
#SBATCH --job-name=samp_conv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/samp_conv_%j.out
#SBATCH --error=logs/samp_conv_%j.err
#SBATCH --mail-user=your.email@example.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --no-requeue
#SBATCH --partition=defaultp
#SBATCH --export=NONE

# Unset SLURM export environment (ISTA cluster requirement)
unset SLURM_EXPORT_ENV

# =============================================================================
# SLURM Job for Python Sampling Convergence Analysis
# =============================================================================
#
# This script processes ALL simulations in parallel using multiprocessing,
# then computes aggregated statistics and outputs final results.
#
# RESOURCE JUSTIFICATION:
#
# --cpus-per-task=32
#   - Uses Python multiprocessing.Pool with 32 workers
#   - Each worker processes one simulation at a time
#   - 32 workers = 32 simulations processed in parallel
#
# --time=12:00:00 (12 hours)
#   - 2500 simulations / 32 workers = ~78 batches
#   - ~20 seconds per simulation = ~26 minutes total
#   - 12 hours provides large safety margin for I/O and overhead
#
# --mem=64G
#   - 32 workers × ~2GB peak memory each
#   - Shared data structures for aggregation
#   - 64G is conservative
#
# USAGE:
#   cd /path/to/data/git/MouseBrainActivity/Neuron\ Activity\ Analysis/main_scripts/Figure5/comparisons
#   mkdir -p logs
#   sbatch run_sampling_convergence_python_slurm.sh
#
# MONITOR:
#   squeue -u $USER
#   tail -f logs/samp_conv_*.out
#
# =============================================================================

# === Configuration ===
INPUT_DIR="/path/to/data/IsingSims"
OUTPUT_DIR="/path/to/data/IsingSims/SamplingConvergence"
SCRIPT_DIR="/path/to/data/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"
WORKERS=32

# === Setup ===
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}"
echo "Start time: $(date)"
echo "========================================"

# Create output directory if needed
mkdir -p "${OUTPUT_DIR}"

# Load Python module (ISTA cluster)
module load python/3.11

# Verify Python
echo "Python: $(which python)"
python --version

# Change to script directory
cd "$SCRIPT_DIR"

# === Run Analysis ===
echo ""
echo "Running sampling convergence analysis..."
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Workers: ${WORKERS}"
echo ""

srun python sampling_convergence_analysis.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --workers ${WORKERS}

# Capture exit code
EXIT_CODE=$?

# === Completion ===
echo ""
echo "========================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit $EXIT_CODE
