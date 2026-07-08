#!/bin/bash
#SBATCH --job-name=IsingComp
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=defaultp
#SBATCH --qos=normal
#SBATCH --time=02-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH --mail-user=your.email@example.com
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=1

# ============================================================================
# Figure 5: Ising Model Comparison - SLURM Submission Script
# ============================================================================
# Submit with: sbatch run_IsingComparison.sh
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# ============================================================================

echo "=========================================="
echo "  Ising Model Comparison - Cluster Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load required modules
module load python/3.11

# Navigate to script directory
cd /path/to/data/git/MouseBrainActivity/Neuron\ Activity\ Analysis/main_scripts/Figure5/comparisons

# Verify paths exist
echo "Checking paths..."
if [ ! -d "/path/to/data/IsingSims" ]; then
    echo "ERROR: Ising sims directory not found!"
    exit 1
fi

if [ ! -f "/path/to/data/ExperimentalData/ExperimentalData.mat" ]; then
    echo "ERROR: Experimental data file not found!"
    exit 1
fi

echo "Paths OK"
echo ""

# Count simulation files
N_SIMS=$(ls /path/to/data/IsingSims/sim_*.mat 2>/dev/null | wc -l)
echo "Found $N_SIMS simulation files"
echo ""

# Run the Python script
echo "Starting Python script..."
srun python Figure5_IsingComparison.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  Job completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "  Job FAILED with exit code $?"
    echo "=========================================="
fi

echo "End time: $(date)"
