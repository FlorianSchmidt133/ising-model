#!/bin/bash
#SBATCH --job-name=downstream_reader
#SBATCH --output=logs/downstream_%A_%a.out
#SBATCH --error=logs/downstream_%A_%a.err
#SBATCH --partition=defaultp
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --array=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
unset SLURM_EXPORT_ENV

# =============================================================================
# Ising Downstream Reader - SLURM Pipeline Script
# =============================================================================
#
# Full pipeline: simulate → combine → classify + figures
#
# Usage:
#   # Run full pipeline (auto-launches array → combine → classify)
#   sbatch run_downstream_reader.sh simulate
#
#   # Simulate specific stim modes
#   sbatch run_downstream_reader.sh simulate --stim-modes bias
#
#   # Just simulate (no chaining, no auto-launch)
#   sbatch --array=0-99 run_downstream_reader.sh simulate --no-chain
#
#   # Combine results (array guard: only task 0 runs)
#   sbatch run_downstream_reader.sh combine
#
#   # Classify and generate figures (array guard: only task 0 runs)
#   sbatch run_downstream_reader.sh classify
#
#   # Dry run: show what would be executed (array guard: only task 0 runs)
#   sbatch run_downstream_reader.sh scan
#
#   # Run everything locally on one big node (no array)
#   sbatch --cpus-per-task=32 --mem=64G --time=12:00:00 \
#       run_downstream_reader.sh local --workers 32
#
#   # Re-run classify + detect in parallel (no simulate/combine needed)
#   sbatch run_downstream_reader.sh classify+detect
#
#   # Full pipeline with ALL modes + both classifiers (dynamic array sizing)
#   sbatch run_downstream_reader.sh simulateAll
#
# Job count: depends on modes. Default: 2 cond × 10 sims × 5 sizes = 100 jobs
# With 3 modes + 3 bias values: 2 × 10 × 5 × (1 + 1 + 3) = 500 jobs
# Each job runs 50 replicates with shared burn-in.
#
# =============================================================================

# --- Parse arguments ---
MODE=${1:-simulate}
shift || true

NO_CHAIN=false
WORKERS=1
METRIC="combined"
STIM_MODES=""

while [ $# -gt 0 ]; do
    case "$1" in
        --no-chain)    NO_CHAIN=true ;;
        --workers)     WORKERS="$2"; shift ;;
        --metric)      METRIC="$2"; shift ;;
        --stim-modes)  STIM_MODES="$2"; shift ;;
        *)             ;; # pass through
    esac
    shift
done

# --- Paths (hardcoded for cluster) ---
SCRIPT_DIR="/path/to/data/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"
ISING_DATA="/path/to/data/IsingSims"
OUTPUT_DIR="/path/to/data/IsingDownstream"
CLASSIFY_DIR="$OUTPUT_DIR/classification"

echo "=============================================="
echo "Ising Downstream Reader Pipeline"
echo "=============================================="
echo "Mode: $MODE"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Start: $(date)"
echo "=============================================="

# --- Environment setup ---
module purge
module load python/3.11

cd "$SCRIPT_DIR"
mkdir -p logs
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$CLASSIFY_DIR"

# Single-thread per process (Numba, BLAS)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=omp

echo "Working directory: $(pwd)"
echo ""

# =============================================================================
case "$MODE" in

# =============================================================================
# SCAN: dry run
# =============================================================================
scan)
    # Array guard: only task 0 runs (tasks 1-99 exit when header default applies)
    if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
        exit 0
    fi
    echo "--- Scan mode ---"
    srun python ising_downstream_reader.py \
        --scan \
        --ising-data "$ISING_DATA" \
        ${STIM_MODES:+--stim-modes $STIM_MODES}
    exit $?
    ;;

# =============================================================================
# SIMULATE: run one job per array task (index = SLURM_ARRAY_TASK_ID)
# =============================================================================
simulate)
    if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
        echo "ERROR: SLURM_ARRAY_TASK_ID not set. Submit via: sbatch run_downstream_reader.sh simulate"
        exit 1
    fi

    # --- Auto-launch: single-task default → submit proper array + chain ---
    if [ "${SLURM_ARRAY_TASK_MAX:-0}" -eq 0 ] && [ "$NO_CHAIN" = false ]; then
        echo "--- Simulate launcher mode ---"
        echo "Querying job count..."

        N_JOBS=$(python ising_downstream_reader.py \
            --job-count \
            --ising-data "$ISING_DATA" \
            ${STIM_MODES:+--stim-modes $STIM_MODES})

        if [ -z "$N_JOBS" ] || [ "$N_JOBS" -lt 1 ] 2>/dev/null; then
            echo "ERROR: Failed to determine job count (got: '$N_JOBS')"
            exit 1
        fi

        MAX_IDX=$((N_JOBS - 1))
        echo "Total jobs: $N_JOBS (array 0-$MAX_IDX)"
        echo ""

        # 1. Submit simulate array
        SIM_JOB=$(sbatch --parsable \
            --partition=defaultp \
            --export=NONE \
            --array=0-${MAX_IDX} \
            --cpus-per-task=1 \
            --mem=4G \
            --time=02:00:00 \
            --job-name="downstream_simulate" \
            --output="$OUTPUT_DIR/logs/sim_%A_%a.out" \
            --error="$OUTPUT_DIR/logs/sim_%A_%a.err" \
            "$SCRIPT_DIR/run_downstream_reader.sh" simulate --no-chain \
            ${STIM_MODES:+--stim-modes "$STIM_MODES"})

        # 2. Combine (waits for ALL simulate tasks)
        COMBINE_JOB=$(sbatch --parsable \
            --dependency=afterok:$SIM_JOB \
            --partition=defaultp \
            --export=NONE \
            --array=0 \
            --cpus-per-task=2 \
            --mem=8G \
            --time=00:30:00 \
            --job-name="downstream_combine" \
            --output="$OUTPUT_DIR/logs/combine_%j.out" \
            --error="$OUTPUT_DIR/logs/combine_%j.err" \
            "$SCRIPT_DIR/run_downstream_reader.sh" combine)

        # 3. Classify (waits for combine)
        CLASSIFY_JOB=$(sbatch --parsable \
            --dependency=afterok:$COMBINE_JOB \
            --partition=defaultp \
            --export=NONE \
            --array=0 \
            --cpus-per-task=8 \
            --mem=32G \
            --time=06:00:00 \
            --job-name="downstream_classify" \
            --output="$OUTPUT_DIR/logs/classify_%j.out" \
            --error="$OUTPUT_DIR/logs/classify_%j.err" \
            "$SCRIPT_DIR/run_downstream_reader.sh" classify --no-chain)

        echo "=============================================="
        echo "Simulate pipeline submitted:"
        echo "=============================================="
        echo "  Simulate array: $SIM_JOB ($N_JOBS tasks)"
        echo "  Combine job:    $COMBINE_JOB (depends on $SIM_JOB)"
        echo "  Classify job:   $CLASSIFY_JOB (depends on $COMBINE_JOB)"
        echo ""
        echo "Output: $OUTPUT_DIR"
        echo ""
        echo "Monitor: squeue -u $USER"
        exit 0
    fi

    INDEX=$SLURM_ARRAY_TASK_ID
    echo "--- Simulate mode: job index $INDEX ---"

    # Per-task Numba cache to avoid NFS contention across array tasks
    export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID}"

    srun python ising_downstream_reader.py \
        --index "$INDEX" \
        --output "$CLASSIFY_DIR" \
        --ising-data "$ISING_DATA" \
        --format npz \
        ${STIM_MODES:+--stim-modes $STIM_MODES}

    RC=$?
    echo ""
    echo "Job $INDEX completed with exit code $RC"
    echo "End: $(date)"

    # --- Chain downstream jobs (only task 0 submits) ---
    if [ "$NO_CHAIN" = false ] && [ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && [ $RC -eq 0 ]; then
        DEP_JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

        echo ""
        echo "=============================================="
        echo "Chaining downstream jobs..."
        echo "=============================================="

        # Combine job (waits for ALL array tasks to succeed)
        COMBINE_JOB=$(sbatch --parsable \
            --dependency=afterok:$DEP_JOB_ID \
            --partition=defaultp \
            --export=NONE \
            --array=0 \
            --cpus-per-task=2 \
            --mem=8G \
            --time=00:30:00 \
            --job-name="downstream_combine" \
            --output="$OUTPUT_DIR/logs/combine_%j.out" \
            --error="$OUTPUT_DIR/logs/combine_%j.err" \
            "$SCRIPT_DIR/run_downstream_reader.sh" combine)

        # Classify job (waits for combine)
        CLASSIFY_JOB=$(sbatch --parsable \
            --dependency=afterok:$COMBINE_JOB \
            --partition=defaultp \
            --export=NONE \
            --array=0 \
            --cpus-per-task=8 \
            --mem=32G \
            --time=06:00:00 \
            --job-name="downstream_classify" \
            --output="$OUTPUT_DIR/logs/classify_%j.out" \
            --error="$OUTPUT_DIR/logs/classify_%j.err" \
            "$SCRIPT_DIR/run_downstream_reader.sh" classify --no-chain)

        echo "  Simulate array: $DEP_JOB_ID (100 tasks)"
        echo "  Combine job:    $COMBINE_JOB (depends on $DEP_JOB_ID)"
        echo "  Classify job:   $CLASSIFY_JOB (depends on $COMBINE_JOB)"
        echo ""
        echo "Output: $OUTPUT_DIR"
    fi

    exit $RC
    ;;

# =============================================================================
# COMBINE: aggregate per-job .npz files into dataset
# =============================================================================
combine)
    # Array guard: only task 0 runs (tasks 1-99 exit when header default applies)
    if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
        exit 0
    fi
    echo "--- Combine mode ---"

    srun python ising_downstream_reader.py \
        --combine \
        --output "$CLASSIFY_DIR"

    RC=$?
    echo ""
    echo "Combine completed with exit code $RC"
    echo "End: $(date)"
    exit $RC
    ;;

# =============================================================================
# CLASSIFY: run classification sweep + generate figures
# =============================================================================
classify)
    # Array guard: only task 0 runs (tasks 1-99 exit when header default applies)
    if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
        exit 0
    fi

    # Auto-resubmit with proper resources if running with header defaults
    if [ "$NO_CHAIN" = false ]; then
        echo "--- Classify launcher: resubmitting with proper resources ---"
        CLASSIFY_JOB=$(sbatch --parsable \
            --partition=defaultp \
            --export=NONE \
            --array=0 \
            --cpus-per-task=8 \
            --mem=32G \
            --time=06:00:00 \
            --job-name="downstream_classify" \
            --output="$OUTPUT_DIR/logs/classify_%j.out" \
            --error="$OUTPUT_DIR/logs/classify_%j.err" \
            "$SCRIPT_DIR/run_downstream_reader.sh" classify --no-chain)
        echo "Classify job submitted: $CLASSIFY_JOB"
        echo "Monitor: squeue -u $USER"
        exit 0
    fi

    echo "--- Classify mode ---"

    # Classification benefits from multi-threaded BLAS for sklearn
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    export JOBLIB_TEMP_FOLDER="${TMPDIR:-/tmp}"

    srun python ising_downstream_reader.py \
        --classify \
        --output "$CLASSIFY_DIR"

    RC=$?
    echo ""
    echo "Classification completed with exit code $RC"
    echo "End: $(date)"
    exit $RC
    ;;

# =============================================================================
# LOCAL: run everything on one node (no array jobs needed)
# =============================================================================
local)
    # Array guard: only task 0 runs (tasks 1-99 exit when header default applies)
    if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
        exit 0
    fi
    echo "--- Local mode (workers=$WORKERS) ---"

    # Allow multi-threading for local mode
    export OMP_NUM_THREADS=1

    srun python ising_downstream_reader.py \
        --local \
        --workers "$WORKERS" \
        --output "$CLASSIFY_DIR" \
        --ising-data "$ISING_DATA" \
        --format npz \
        ${STIM_MODES:+--stim-modes $STIM_MODES}

    RC=$?
    echo ""
    echo "Local run completed with exit code $RC"
    echo "End: $(date)"
    exit $RC
    ;;

# =============================================================================
# DETECT: submit detect-sim array + detect-combine chain
# =============================================================================
detect)
    # Array guard: only task 0 runs
    if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
        exit 0
    fi

    if [ "$NO_CHAIN" = false ]; then
        DETECT_MODES="${STIM_MODES:-clamped double_pulse bias}"

        # Get number of detect grouped jobs
        N_DETECT=$(python ising_downstream_reader.py \
            --detect-job-count \
            --ising-data "$ISING_DATA" \
            ${DETECT_MODES:+--stim-modes $DETECT_MODES})

        if [ -z "$N_DETECT" ] || [ "$N_DETECT" -lt 1 ] 2>/dev/null; then
            echo "ERROR: Failed to determine detect job count (got: '$N_DETECT')"
            exit 1
        fi

        MAX_IDX=$((N_DETECT - 1))

        # 1. Submit detect simulation array (1 CPU per task)
        DETECT_SIM=$(sbatch --parsable \
            --partition=defaultp --export=NONE \
            --array=0-${MAX_IDX} \
            --cpus-per-task=1 --mem=8G --time=03:00:00 \
            --job-name="downstream_detect_sim" \
            --output="$OUTPUT_DIR/logs/detect_sim_%A_%a.out" \
            --error="$OUTPUT_DIR/logs/detect_sim_%A_%a.err" \
            "$SCRIPT_DIR/run_downstream_reader.sh" detect-sim --no-chain \
            --stim-modes "$DETECT_MODES")

        # 2. Bridge: detect-clf launcher queries dynamic clf count after sim completes
        DETECT_CLF_BRIDGE=$(sbatch --parsable \
            --dependency=afterok:$DETECT_SIM \
            --partition=defaultp --export=NONE --array=0 \
            --cpus-per-task=1 --mem=2G --time=00:10:00 \
            --job-name="downstream_detect_clf_bridge" \
            --output="$OUTPUT_DIR/logs/detect_clf_bridge_%j.out" \
            --error="$OUTPUT_DIR/logs/detect_clf_bridge_%j.err" \
            "$SCRIPT_DIR/run_downstream_reader.sh" detect-clf --no-chain)

        echo "Detect pipeline submitted:"
        echo "  Simulate:    $DETECT_SIM ($N_DETECT tasks)"
        echo "  Clf bridge:  $DETECT_CLF_BRIDGE (depends on $DETECT_SIM)"
        echo "  (classify array + aggregate will be submitted by bridge)"
        exit 0
    fi

    # Fallback: --no-chain runs all locally (existing behavior)
    echo "--- Detect mode (local) ---"
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8

    srun python ising_downstream_reader.py \
        --detect \
        --output "$OUTPUT_DIR" \
        --ising-data "$ISING_DATA" \
        --workers "${WORKERS:-8}" \
        ${STIM_MODES:+--stim-modes $STIM_MODES}

    RC=$?
    echo ""
    echo "Detection completed with exit code $RC"
    echo "End: $(date)"
    exit $RC
    ;;

# =============================================================================
# DETECT-SIM: run one detect grouped job (SLURM array worker)
# =============================================================================
detect-sim)
    # Per-task Numba cache to avoid NFS contention across array tasks
    export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID}"

    srun python ising_downstream_reader.py \
        --detect-index "$SLURM_ARRAY_TASK_ID" \
        --output "$OUTPUT_DIR" \
        --ising-data "$ISING_DATA" \
        ${STIM_MODES:+--stim-modes $STIM_MODES}

    RC=$?
    echo ""
    echo "Detect job $SLURM_ARRAY_TASK_ID completed with exit code $RC"
    echo "End: $(date)"
    exit $RC
    ;;

# =============================================================================
# DETECT-COMBINE: aggregate detect results, classify, plot
# =============================================================================
detect-combine)
    if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
        exit 0
    fi

    echo "--- Detect combine mode ---"
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8

    srun python ising_downstream_reader.py \
        --detect-combine \
        --output "$OUTPUT_DIR" \
        --ising-data "$ISING_DATA" \
        --workers "${WORKERS:-8}" \
        ${STIM_MODES:+--stim-modes $STIM_MODES}

    RC=$?
    echo ""
    echo "Detect combine completed with exit code $RC"
    echo "End: $(date)"
    exit $RC
    ;;

# =============================================================================
# DETECT-CLF: submit classify array + aggregate (skip detect-sim)
# =============================================================================
detect-clf)
    if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
        exit 0
    fi
    echo "--- Detect classify launcher ---"

    # Dynamically determine classify job count from detect_job files
    N_CLF=$(python ising_downstream_reader.py \
        --detect-clf-job-count \
        --output "$OUTPUT_DIR")

    if [ -z "$N_CLF" ] || [ "$N_CLF" -lt 1 ] 2>/dev/null; then
        echo "ERROR: Failed to determine clf job count (got: '$N_CLF')"
        exit 1
    fi

    CLF_MAX=$((N_CLF - 1))
    echo "  Classify jobs: $N_CLF (array 0-$CLF_MAX)"

    DETECT_CLF=$(sbatch --parsable \
        --partition=defaultp --export=NONE \
        --array=0-${CLF_MAX} \
        --cpus-per-task=8 --mem=8G --time=01:00:00 \
        --job-name="downstream_detect_clf" \
        --output="$OUTPUT_DIR/logs/detect_clf_%A_%a.out" \
        --error="$OUTPUT_DIR/logs/detect_clf_%A_%a.err" \
        "$SCRIPT_DIR/run_downstream_reader.sh" detect-classify-job --no-chain)

    DETECT_AGG=$(sbatch --parsable \
        --dependency=afterok:$DETECT_CLF \
        --partition=defaultp --export=NONE --array=0 \
        --cpus-per-task=4 --mem=16G --time=00:30:00 \
        --job-name="downstream_detect_agg" \
        --output="$OUTPUT_DIR/logs/detect_agg_%j.out" \
        --error="$OUTPUT_DIR/logs/detect_agg_%j.err" \
        "$SCRIPT_DIR/run_downstream_reader.sh" detect-aggregate --no-chain)

    echo "Detect classify+aggregate submitted:"
    echo "  Classify:  $DETECT_CLF ($N_CLF tasks)"
    echo "  Aggregate: $DETECT_AGG (depends on $DETECT_CLF)"
    exit 0
    ;;

# =============================================================================
# DETECT-CLASSIFY-JOB: classify one (condition, stim_size, stim_mode, stim_bias) tuple (SLURM array)
# =============================================================================
detect-classify-job)
    echo "--- Detect classify job: index $SLURM_ARRAY_TASK_ID ---"
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8

    srun python ising_downstream_reader.py \
        --detect-classify-job \
        --index "$SLURM_ARRAY_TASK_ID" \
        --output "$OUTPUT_DIR"

    RC=$?
    echo ""
    echo "Detect classify job $SLURM_ARRAY_TASK_ID completed with exit code $RC"
    echo "End: $(date)"
    exit $RC
    ;;

# =============================================================================
# DETECT-AGGREGATE: combine clf results, plot, save
# =============================================================================
detect-aggregate)
    if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
        exit 0
    fi

    echo "--- Detect aggregate mode ---"
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4

    srun python ising_downstream_reader.py \
        --detect-aggregate \
        --output "$OUTPUT_DIR"

    RC=$?
    echo ""
    echo "Detect aggregate completed with exit code $RC"
    echo "End: $(date)"
    exit $RC
    ;;

# =============================================================================
# CLASSIFY+DETECT: re-run classify and detect in parallel (combine already done)
# =============================================================================
classify+detect)
    # Array guard: only task 0 runs
    if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
        exit 0
    fi
    echo "--- Classify+Detect launcher mode ---"

    # Default to all stim modes if none specified
    DETECT_MODES="${STIM_MODES:-clamped double_pulse bias}"

    # 1. Classify (single job, multi-core)
    CLASSIFY_JOB=$(sbatch --parsable \
        --partition=defaultp \
        --export=NONE \
        --array=0 \
        --cpus-per-task=8 \
        --mem=32G \
        --time=06:00:00 \
        --job-name="downstream_classify" \
        --output="$OUTPUT_DIR/logs/classify_%j.out" \
        --error="$OUTPUT_DIR/logs/classify_%j.err" \
        "$SCRIPT_DIR/run_downstream_reader.sh" classify --no-chain)

    # 2. Detect pipeline (array sim → combine, runs in parallel with classify)
    N_DETECT=$(python ising_downstream_reader.py \
        --detect-job-count \
        --ising-data "$ISING_DATA" \
        ${DETECT_MODES:+--stim-modes $DETECT_MODES})

    if [ -z "$N_DETECT" ] || [ "$N_DETECT" -lt 1 ] 2>/dev/null; then
        echo "ERROR: Failed to determine detect job count (got: '$N_DETECT')"
        exit 1
    fi

    MAX_IDX=$((N_DETECT - 1))

    DETECT_SIM=$(sbatch --parsable \
        --partition=defaultp --export=NONE \
        --array=0-${MAX_IDX} \
        --cpus-per-task=1 --mem=8G --time=03:00:00 \
        --job-name="downstream_detect_sim" \
        --output="$OUTPUT_DIR/logs/detect_sim_%A_%a.out" \
        --error="$OUTPUT_DIR/logs/detect_sim_%A_%a.err" \
        "$SCRIPT_DIR/run_downstream_reader.sh" detect-sim --no-chain \
        --stim-modes "$DETECT_MODES")

    # Bridge: detect-clf launcher queries dynamic clf count after sim completes
    DETECT_CLF_BRIDGE=$(sbatch --parsable \
        --dependency=afterok:$DETECT_SIM \
        --partition=defaultp --export=NONE --array=0 \
        --cpus-per-task=1 --mem=2G --time=00:10:00 \
        --job-name="downstream_detect_clf_bridge" \
        --output="$OUTPUT_DIR/logs/detect_clf_bridge_%j.out" \
        --error="$OUTPUT_DIR/logs/detect_clf_bridge_%j.err" \
        "$SCRIPT_DIR/run_downstream_reader.sh" detect-clf --no-chain)

    echo "=============================================="
    echo "Classify+Detect submitted (parallel):"
    echo "=============================================="
    echo "  Classify job:     $CLASSIFY_JOB"
    echo "  Detect simulate:  $DETECT_SIM ($N_DETECT tasks)"
    echo "  Clf bridge:       $DETECT_CLF_BRIDGE (depends on $DETECT_SIM)"
    echo "  (classify array + aggregate will be submitted by bridge)"
    echo ""
    echo "Output: $OUTPUT_DIR"
    echo ""
    echo "Monitor: squeue -u $USER"
    exit 0
    ;;

# =============================================================================
# SIMULATE-ALL: full pipeline with all modes + both classifiers
# =============================================================================
simulateAll)
    # Array guard: only task 0 runs
    if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
        exit 0
    fi
    echo "--- SimulateAll mode ---"
    echo "Querying job count for all modes..."

    ALL_STIM_MODES="clamped double_pulse bias"

    N_JOBS=$(python ising_downstream_reader.py \
        --job-count \
        --ising-data "$ISING_DATA" \
        --stim-modes $ALL_STIM_MODES)

    if [ -z "$N_JOBS" ] || [ "$N_JOBS" -lt 1 ] 2>/dev/null; then
        echo "ERROR: Failed to determine job count (got: '$N_JOBS')"
        exit 1
    fi

    MAX_IDX=$((N_JOBS - 1))
    echo "Total jobs: $N_JOBS (array 0-$MAX_IDX)"
    echo ""

    # 1. Submit simulate array
    SIM_JOB=$(sbatch --parsable \
        --partition=defaultp \
        --export=NONE \
        --array=0-${MAX_IDX} \
        --cpus-per-task=1 \
        --mem=4G \
        --time=02:00:00 \
        --job-name="downstream_sim_all" \
        --output="$OUTPUT_DIR/logs/sim_%A_%a.out" \
        --error="$OUTPUT_DIR/logs/sim_%A_%a.err" \
        "$SCRIPT_DIR/run_downstream_reader.sh" simulate --no-chain --stim-modes "$ALL_STIM_MODES")

    # 2. Combine (waits for ALL simulate tasks)
    COMBINE_JOB=$(sbatch --parsable \
        --dependency=afterok:$SIM_JOB \
        --partition=defaultp \
        --export=NONE \
        --array=0 \
        --cpus-per-task=2 \
        --mem=8G \
        --time=00:30:00 \
        --job-name="downstream_combine" \
        --output="$OUTPUT_DIR/logs/combine_%j.out" \
        --error="$OUTPUT_DIR/logs/combine_%j.err" \
        "$SCRIPT_DIR/run_downstream_reader.sh" combine)

    # 3a. Classify: Naive vs Expert (waits for combine)
    CLASSIFY_JOB=$(sbatch --parsable \
        --dependency=afterok:$COMBINE_JOB \
        --partition=defaultp \
        --export=NONE \
        --array=0 \
        --cpus-per-task=8 \
        --mem=48G \
        --time=06:00:00 \
        --job-name="downstream_classify" \
        --output="$OUTPUT_DIR/logs/classify_%j.out" \
        --error="$OUTPUT_DIR/logs/classify_%j.err" \
        "$SCRIPT_DIR/run_downstream_reader.sh" classify --no-chain)

    # === Detect pipeline (independent, starts immediately) ===
    N_DETECT=$(python ising_downstream_reader.py \
        --detect-job-count \
        --ising-data "$ISING_DATA" \
        --stim-modes $ALL_STIM_MODES)

    if [ -z "$N_DETECT" ] || [ "$N_DETECT" -lt 1 ] 2>/dev/null; then
        echo "ERROR: Failed to determine detect job count (got: '$N_DETECT')"
        exit 1
    fi

    DETECT_MAX_IDX=$((N_DETECT - 1))

    DETECT_SIM=$(sbatch --parsable \
        --partition=defaultp --export=NONE \
        --array=0-${DETECT_MAX_IDX} \
        --cpus-per-task=1 --mem=8G --time=03:00:00 \
        --job-name="downstream_detect_sim" \
        --output="$OUTPUT_DIR/logs/detect_sim_%A_%a.out" \
        --error="$OUTPUT_DIR/logs/detect_sim_%A_%a.err" \
        "$SCRIPT_DIR/run_downstream_reader.sh" detect-sim --no-chain \
        --stim-modes "$ALL_STIM_MODES")

    # Bridge: detect-clf launcher queries dynamic clf count after sim completes
    DETECT_CLF_BRIDGE=$(sbatch --parsable \
        --dependency=afterok:$DETECT_SIM \
        --partition=defaultp --export=NONE --array=0 \
        --cpus-per-task=1 --mem=2G --time=00:10:00 \
        --job-name="downstream_detect_clf_bridge" \
        --output="$OUTPUT_DIR/logs/detect_clf_bridge_%j.out" \
        --error="$OUTPUT_DIR/logs/detect_clf_bridge_%j.err" \
        "$SCRIPT_DIR/run_downstream_reader.sh" detect-clf --no-chain)

    echo "=============================================="
    echo "SimulateAll pipeline submitted:"
    echo "=============================================="
    echo "  Classify pipeline:"
    echo "    Simulate array: $SIM_JOB ($N_JOBS tasks, modes: $ALL_STIM_MODES)"
    echo "    Combine job:    $COMBINE_JOB (depends on $SIM_JOB)"
    echo "    Classify job:   $CLASSIFY_JOB (depends on $COMBINE_JOB)"
    echo "  Detect pipeline (independent):"
    echo "    Detect simulate:  $DETECT_SIM ($N_DETECT tasks)"
    echo "    Clf bridge:       $DETECT_CLF_BRIDGE (depends on $DETECT_SIM)"
    echo "    (classify array + aggregate will be submitted by bridge)"
    echo ""
    echo "Output: $OUTPUT_DIR"
    echo ""
    echo "Monitor: squeue -u $USER"
    exit 0
    ;;

# =============================================================================
*)
    echo "ERROR: Unknown mode '$MODE'"
    echo "Valid modes: scan, simulate, combine, classify, detect, detect-sim, detect-combine, detect-clf, detect-classify-job, detect-aggregate, classify+detect, local, simulateAll"
    echo "Options: --stim-modes 'clamped double_pulse bias' --workers N --no-chain"
    exit 1
    ;;

esac
