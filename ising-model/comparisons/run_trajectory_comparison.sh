#!/bin/bash
# =============================================================================
# Ising Model Trajectory Comparison - SLURM Submission Script
# =============================================================================
# Submits a two-phase pipeline:
#   Phase 1: Array job (16 tasks x 5 batch = 80 trajectory comparisons)
#   Phase 2: Combine job (aggregates results into TrajectoryResults.h5)
#            Automatically starts after all array tasks succeed.
#
# Usage (run from login node, NOT via sbatch):
#   cd ~/repo/ising-model/comparisons
#   bash run_trajectory_comparison.sh [submit [OUTPUT_DIR] [COMPARISON_PATH] [EXP_DATA_PATH]]
#
# Monitor:
#   squeue -u $USER
#   sacct -j <jobid>
#
# Cancel:
#   scancel <jobid>
#   scancel -u $USER -n ising_traj
# =============================================================================

# Dispatch based on first positional argument (internal use by sbatch)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE="${1:-submit}"

if [ "$PHASE" = "compute" ]; then
    # Configuration passed as positional arguments from submitter
    SCRIPT_DIR="$2"
    OUTPUT_DIR="$3"
    COMPARISON="$4"
    EXP_DATA="$5"
    # =========================================================================
    # Phase 1: Array worker (runs under SLURM)
    # =========================================================================
    unset SLURM_EXPORT_ENV

    export OMP_NUM_THREADS=1
    module load python/3.11

    mkdir -p "$OUTPUT_DIR"

    echo "========================================"
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
    echo "Node: $SLURMD_NODENAME"
    echo "Start time: $(date)"
    echo "Python: $(which python)"
    echo "Output dir: $OUTPUT_DIR"
    echo "========================================"

    cd "$SCRIPT_DIR"
    srun --cpu_bind=verbose python trajectory_comparison.py \
        --output "$OUTPUT_DIR" \
        --comparison "$COMPARISON" \
        --exp-data "$EXP_DATA" \
        --index $SLURM_ARRAY_TASK_ID \
        --batch-size 5
    RC=$?

    echo "========================================"
    echo "End time: $(date)"
    echo "Exit code: $RC"
    echo "========================================"
    exit $RC

elif [ "$PHASE" = "combine" ]; then
    # Configuration passed as positional arguments from submitter
    SCRIPT_DIR="$2"
    OUTPUT_DIR="$3"

    # =========================================================================
    # Phase 2: Combine results (runs under SLURM after array completes)
    # =========================================================================
    unset SLURM_EXPORT_ENV

    export OMP_NUM_THREADS=1
    module load python/3.11

    echo "========================================"
    echo "Combine Job: $SLURM_JOB_ID"
    echo "Node: $SLURMD_NODENAME"
    echo "Start time: $(date)"
    echo "Output dir: $OUTPUT_DIR"
    echo "========================================"

    cd "$SCRIPT_DIR"
    srun --cpu_bind=verbose python trajectory_comparison.py \
        --output "$OUTPUT_DIR" \
        --combine
    RC=$?

    echo "========================================"
    echo "End time: $(date)"
    echo "Exit code: $RC"
    echo "========================================"
    exit $RC

else
    # =========================================================================
    # Submitter (runs on login node)
    # =========================================================================
    OUTPUT_DIR="${2:-/path/to/data/IsingTrajectories}"
    COMPARISON="${3:-auto}"
    EXP_DATA="${4:-/path/to/data/ExperimentalData/ExperimentalData.mat}"

    mkdir -p logs
    mkdir -p "$OUTPUT_DIR"

    echo "========================================"
    echo "  Trajectory Comparison Pipeline"
    echo "========================================"
    echo "Output dir: $OUTPUT_DIR"
    echo "Comparison: $COMPARISON"
    echo "Exp data:   $EXP_DATA"
    echo ""

    # Submit Phase 1: array job
    ARRAY_JOBID=$(sbatch --parsable \
        --job-name=ising_traj \
        --array=0-15 \
        --time=02:00:00 \
        --mem=4G \
        --cpus-per-task=1 \
        --output=logs/traj_%A_%a.out \
        --error=logs/traj_%A_%a.err \
        --no-requeue \
        --mail-user=your.email@example.com \
        --mail-type=FAIL \
        --export=NONE \
        "$SCRIPT_DIR/run_trajectory_comparison.sh" compute "$SCRIPT_DIR" "$OUTPUT_DIR" "$COMPARISON" "$EXP_DATA")

    echo "Phase 1 (compute): Submitted array job $ARRAY_JOBID (16 tasks x 5 = 80 jobs)"

    # Submit Phase 2: combine job (depends on all array tasks succeeding)
    COMBINE_JOBID=$(sbatch --parsable \
        --job-name=ising_traj_combine \
        --time=00:30:00 \
        --mem=8G \
        --cpus-per-task=1 \
        --output=logs/traj_combine_%j.out \
        --error=logs/traj_combine_%j.err \
        --no-requeue \
        --mail-user=your.email@example.com \
        --mail-type=END,FAIL \
        --dependency=afterok:$ARRAY_JOBID \
        --export=NONE \
        "$SCRIPT_DIR/run_trajectory_comparison.sh" combine "$SCRIPT_DIR" "$OUTPUT_DIR")

    echo "Phase 2 (combine): Submitted job $COMBINE_JOBID (depends on $ARRAY_JOBID)"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Results: $OUTPUT_DIR/TrajectoryResults.h5"
    echo "========================================"
fi
