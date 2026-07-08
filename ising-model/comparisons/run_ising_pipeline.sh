#!/bin/bash
#SBATCH --job-name=IsingPipeline
#SBATCH --output=logs/ising_pipeline_%j_%a.out
#SBATCH --error=logs/ising_pipeline_%j_%a.err
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --partition=defaultp
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# =============================================================================
# Unified Ising Model Analysis Pipeline - SLURM Script
# =============================================================================
#
# This script runs different stages of the Ising model analysis pipeline.
#
# MODES:
#   fullPipeline  - Run comparison + submit perturbation chain (the common case)
#   comparison    - Run Ising comparison analysis only
#   perturbations - Run perturbation experiments (use with --array)
#   combine       - Combine individual perturbation results
#   analyze       - Analyze perturbation results and generate figures
#   retry         - Check for failed tasks and resubmit them (up to 3 retries)
#   figures       - Regenerate figures from existing comparison results
#   snapshots     - Snapshot figures + multi-replicate heatmaps/asymmetry analysis
#                   Auto-submits 6 jobs: snap workers/combine/figures + heatmap workers/combine/figures
#
# USAGE:
#   # === Full pipeline (comparison + perturbation chain) ===
#   sbatch --array=0-2 run_ising_pipeline.sh fullPipeline --metric moransI+activity
#
#   # Rerun just the perturbation chain (comparison results already exist)
#   sbatch --cpus-per-task=1 --mem=1G --time=00:05:00 \
#       run_ising_pipeline.sh fullPipeline --metric moransI+activity --skip-comparison
#
#   # Submit only the analysis job (assumes perturbation results exist)
#   sbatch --cpus-per-task=1 --mem=1G --time=00:05:00 \
#       run_ising_pipeline.sh fullPipeline --metric moransI+activity --analysis-only
#
#   # === Comparison only ===
#   sbatch run_ising_pipeline.sh comparison --metric moransI
#
#   # === Perturbation experiments (as array job with batching) ===
#   # Array range is dynamic — use --count-jobs to compute it:
#   TOTAL=$(python run_ising_perturbations.py --comparison <path> --count-jobs)
#   ARRAY_MAX=$(( (TOTAL + 10 - 1) / 10 - 1 ))
#   sbatch --array=0-$ARRAY_MAX --cpus-per-task=1 --mem=2G --time=04:00:00 \
#          run_ising_pipeline.sh perturbations --metric moransI+activity
#
#   # Legacy mode (no batching, for debugging):
#   TOTAL=$(python run_ising_perturbations.py --comparison <path> --count-jobs)
#   sbatch --array=0-$((TOTAL-1)) --cpus-per-task=1 --mem=2G --time=00:10:00 \
#          run_ising_pipeline.sh perturbations --batch-size 1
#
#   # === Combine perturbation results ===
#   sbatch --cpus-per-task=4 --mem=64G --time=01:00:00 \
#          run_ising_pipeline.sh combine --metric moransI+activity
#
#   # === Analyze perturbation results (requires MATLAB node) ===
#   sbatch --constraint=matlab --cpus-per-task=4 --mem=96G --time=04:00:00 \
#          run_ising_pipeline.sh analyze --metric moransI+activity
#
#   # Analyze a specific stim mode:
#   sbatch --constraint=matlab --cpus-per-task=4 --mem=96G --time=04:00:00 \
#          run_ising_pipeline.sh analyze --metric moransI+activity --stim-mode double_pulse
#
#   # Analyze all stim modes in parallel:
#   sbatch --constraint=matlab --cpus-per-task=4 --mem=96G --time=04:00:00 \
#          run_ising_pipeline.sh analyze --metric moransI+activity --stim-mode all
#
#   # === Regenerate comparison figures ===
#   sbatch --cpus-per-task=4 --mem=32G --time=01:00:00 \
#          run_ising_pipeline.sh figures --metric moransI+activity
#
#   # === Snapshots + heatmaps (both pipelines in parallel) ===
#   sbatch run_ising_pipeline.sh snapshots --metric spatial+persistence_weighted
#
#   # With custom replicate count:
#   sbatch run_ising_pipeline.sh snapshots --metric spatial+persistence_weighted --n-reps 200
#
#   # Dry run (shows both pipelines):
#   sbatch run_ising_pipeline.sh snapshots --metric spatial+persistence_weighted --scan
#
#   # Regenerate snapshot figures only:
#   sbatch run_ising_pipeline.sh snapshots --metric spatial+persistence_weighted --figures-only
#
#   # Regenerate heatmap figures from saved data:
#   sbatch run_ising_pipeline.sh snapshots --metric spatial+persistence_weighted --heatmap-only
#
#   # Combine heatmap worker results:
#   sbatch run_ising_pipeline.sh snapshots --metric spatial+persistence_weighted --combine
#
#   # Count heatmap SLURM jobs:
#   sbatch run_ising_pipeline.sh snapshots --metric spatial+persistence_weighted --count-jobs
#
#   # === Force regeneration with new random seeds ===
#   sbatch --array=0-2 run_ising_pipeline.sh fullPipeline --metric moransI+activity --force --seed-offset 1
#
#   # === Increase replicates ===
#   sbatch --array=0-2 run_ising_pipeline.sh fullPipeline --metric moransI+activity --force --n-replicates 200
#
#   # === Retry failed tasks ===
#   sbatch --cpus-per-task=1 --mem=4G --time=00:30:00 \
#          run_ising_pipeline.sh retry --parent-job 54170123 --metric moransI+activity
#
# OPTIONS (for fullPipeline and comparison modes):
#   --metric <name>         Metric name (moransI, activity, moransI+activity,
#                           spatial+persistence, combined). Default: combined.
#   --workers <n>           Number of parallel workers (default: $SLURM_CPUS_PER_TASK)
#   --max-timesteps <n>     Max Ising timesteps (default: 10000)
#   --frame-label <lbl>     Frame label (prestim, full_trial, nostim). In array mode,
#                           auto-mapped from task ID if not specified.
#
# OPTIONS (for fullPipeline only):
#   --skip-comparison       Skip comparison, only submit downstream chain
#   --analysis-only         Skip comparison + perturbations, submit only analyze job
#
# OPTIONS (for perturbations, combine, analyze modes):
#   --metric <name>         Metric name (see above)
#   --frame-label <lbl>     Frame label in comparison filename (prestim, full_trial, nostim; default: full_trial)
#   --output <path>         Output directory for results
#   --ising-data <path>     Path to Ising simulation data
#   --batch-size <n>        Jobs per array task (default: 10). Use 1 for legacy mode.
#   --stim-mode <mode>      (analyze) Stimulus mode: clamped (default), double_pulse, high_bias, low_bias, all
#
# OPTIONS (for retry mode):
#   --parent-job <id>       SLURM job ID of the perturbation array to check
#   --metric <name>         Metric name for resubmitting tasks
#   --frame-label <lbl>     Frame label in comparison filename (prestim, full_trial, nostim; default: full_trial)
#   --output <path>         Output directory (same as original job)
#   --ising-data <path>     Path to Ising simulation data
#   --max-retries <n>       Maximum retry attempts (default: 3)
#   --batch-size <n>        Jobs per array task (default: 10, must match original job)
#
#   # Chain full pipeline manually with dependencies:
#   JOB1=$(sbatch --parsable --array=0-2 run_ising_pipeline.sh comparison --metric moransI+activity)
#   # Compute array range dynamically:
#   TOTAL=$(python run_ising_perturbations.py --comparison <path> --count-jobs)
#   ARRAY_MAX=$(( (TOTAL + 10 - 1) / 10 - 1 ))
#   JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 --array=0-$ARRAY_MAX \
#          --cpus-per-task=1 --mem=2G --time=04:00:00 \
#          run_ising_pipeline.sh perturbations --metric moransI+activity --frame-label full_trial)
#   JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 \
#          --cpus-per-task=4 --mem=64G --time=01:00:00 \
#          run_ising_pipeline.sh combine --metric moransI+activity --frame-label full_trial)
#   sbatch --constraint=matlab --dependency=afterok:$JOB3 --cpus-per-task=4 --mem=32G --time=01:00:00 \
#          run_ising_pipeline.sh analyze --metric moransI+activity --stim-mode clamped
#
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="/path/to/data/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"

# Base paths — overridden below when connectivity=queen
ISING_DATA_BASE_ROOK="/path/to/data/IsingSims"
ISING_DATA_BASE_QUEEN="/path/to/data/IsingSims_queen"
PERTURBATION_BASE_ROOK="/path/to/data/IsingPerturbations"
PERTURBATION_BASE_QUEEN="/path/to/data/IsingPerturbations_queen"

# Defaults (may be overridden after --connectivity is parsed in each mode)
ISING_DATA="$ISING_DATA_BASE_ROOK"
PERTURBATION_OUTPUT="$PERTURBATION_BASE_ROOK"
COMPARISON_OUTPUT="$ISING_DATA_BASE_ROOK/IsingComparison"

# Helper: call after parsing --connectivity (and optionally --refractory) in
# each mode to set paths. When REFRACTORY > 0, all base paths get the suffix
# `_refractoryK${REFRACTORY}` appended.
set_connectivity_paths() {
    if [ "${CONNECTIVITY:-rook}" = "queen" ]; then
        ISING_DATA="$ISING_DATA_BASE_QUEEN"
        PERTURBATION_OUTPUT="$PERTURBATION_BASE_QUEEN"
        COMPARISON_OUTPUT="$ISING_DATA_BASE_QUEEN/IsingComparison"
        CONN_SUFFIX="_queen"
    else
        ISING_DATA="$ISING_DATA_BASE_ROOK"
        PERTURBATION_OUTPUT="$PERTURBATION_BASE_ROOK"
        COMPARISON_OUTPUT="$ISING_DATA_BASE_ROOK/IsingComparison"
        CONN_SUFFIX=""
    fi
    if [ -n "${REFRACTORY:-}" ] && [ "${REFRACTORY}" -gt 0 ] 2>/dev/null; then
        ISING_DATA="${ISING_DATA}_refractoryK${REFRACTORY}"
        PERTURBATION_OUTPUT="${PERTURBATION_OUTPUT}_refractoryK${REFRACTORY}"
        COMPARISON_OUTPUT="${ISING_DATA}/IsingComparison"
    fi
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
if [[ "${1:-}" == --* ]] || [[ -z "${1:-}" ]]; then
    MODE="fullPipeline"
    # Don't shift — all args are flags for the mode handler
else
    MODE="$1"
    shift
fi

# Validate mode
case "$MODE" in
    fullPipeline|comparison|perturbations|combine|analyze|retry|figures|snapshots)
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'"
        echo "Valid modes: fullPipeline, comparison, perturbations, combine, analyze, retry, figures, snapshots"
        exit 1
        ;;
esac

# -----------------------------------------------------------------------------
# Print job info
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Ising Model Analysis Pipeline"
echo "=============================================="
echo "Mode: $MODE"
echo "Job ID: $SLURM_JOB_ID"
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
fi
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Start time: $(date)"
echo "Additional args: $@"
echo "=============================================="

# -----------------------------------------------------------------------------
# Set up environment
# -----------------------------------------------------------------------------
echo ""
echo "Setting up environment..."

# Ensure module function is available (some nodes lack pre-loaded lmod)
if ! type module &>/dev/null; then
    source /usr/share/lmod/lmod/init/bash 2>/dev/null || \
    source /etc/profile.d/lmod.sh 2>/dev/null || \
    source /etc/profile.d/modules.sh 2>/dev/null || true
fi
module purge
# Try a sequence of python module versions. Some versions on the cluster have a
# broken modulefile (codename concat error) that fails to load in
# non-interactive bash even though `module load` returns 0. Sanity-check via
# import after each load and stop at the first one that works.
PYTHON_LOADED=""
for _PY in python/3.11 python/3.11.13 python/3.11.12 python/3.11.11 python/3.11.10 python/3.11.9; do
    module purge 2>/dev/null
    module load "$_PY" 2>/dev/null || true
    if python -c "import numba, numpy, scipy" 2>/dev/null; then
        PYTHON_LOADED="$_PY"
        break
    fi
done
if [ -z "$PYTHON_LOADED" ]; then
    # Soft-fail: this node's python lmod is broken. Requeue this task onto a
    # different node and exit 0 so the parent array isn't recorded as FAILED
    # (which would permanently break afterok dependencies — see commit history
    # for the prior class of stuck snap_agg/heatmap_agg jobs).
    echo "WARN: no python module on $(hostname) yields usable numba/numpy/scipy stack." >&2
    echo "  Tried: python/3.11 python/3.11.13 ... python/3.11.9" >&2
    if [ -n "$SLURM_ARRAY_JOB_ID" ] && [ -n "$SLURM_ARRAY_TASK_ID" ]; then
        REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    elif [ -n "$SLURM_JOB_ID" ]; then
        REQUEUE_TARGET="$SLURM_JOB_ID"
    else
        REQUEUE_TARGET=""
    fi
    if [ -n "$REQUEUE_TARGET" ]; then
        echo "  Requeueing task $REQUEUE_TARGET onto another node." >&2
        scontrol requeue "$REQUEUE_TARGET" 2>&1 || true
        # Sleep so slurm processes the requeue before the script returns,
        # then exit 0 so the originating record looks like success — the
        # requeue itself takes precedence.
        sleep 10
        exit 0
    else
        # Not running under SLURM (interactive run on a bad node). Hard-fail.
        echo "  Not under SLURM; cannot requeue. Failing." >&2
        python -c "import sys; print('python:', sys.executable)" 2>&1 || true
        exit 1
    fi
fi

# Disable HDF5 file locking — required for writing .mat (HDF5) files on NFS
export HDF5_USE_FILE_LOCKING=FALSE

cd "$SCRIPT_DIR"
mkdir -p logs

echo "Working directory: $(pwd)"

# -----------------------------------------------------------------------------
# Run the appropriate mode
# -----------------------------------------------------------------------------

case "$MODE" in
    # =========================================================================
    # FULL PIPELINE MODE (comparison + downstream chain)
    # =========================================================================
    fullPipeline)
        echo ""
        echo "=============================================="
        echo "Running Full Ising Pipeline"
        echo "=============================================="

        # Environment optimized for process-level parallelism
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMBA_NUM_THREADS=1
        export NUMBA_THREADING_LAYER=omp

        # Default arguments
        METRIC="combined"
        WORKERS="${SLURM_CPUS_PER_TASK:-64}"
        MAX_TIMESTEPS="10000"
        SKIP_COMPARISON=false
        ANALYSIS_ONLY=false
        FRAME_LABEL=""
        EXTRA_ARGS=""
        FORCE_FLAG=""
        SEED_OFFSET=""
        N_REPLICATES_OVERRIDE=""
        CONNECTIVITY="rook"
        N_COMP_CHUNKS=1
        OUTPUT_SUFFIX=""
        PERTURB_FRAME_LABEL="full_trial"

        # Parse arguments
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --metric)
                    METRIC="$2"
                    shift 2
                    ;;
                --workers)
                    WORKERS="$2"
                    shift 2
                    ;;
                --max-timesteps)
                    MAX_TIMESTEPS="$2"
                    shift 2
                    ;;
                --skip-comparison)
                    SKIP_COMPARISON=true
                    shift
                    ;;
                --analysis-only)
                    ANALYSIS_ONLY=true
                    SKIP_COMPARISON=true
                    shift
                    ;;
                --frame-label)
                    FRAME_LABEL="$2"
                    shift 2
                    ;;
                --force)
                    FORCE_FLAG="--force"
                    shift
                    ;;
                --seed-offset)
                    SEED_OFFSET="$2"
                    shift 2
                    ;;
                --n-replicates)
                    N_REPLICATES_OVERRIDE="$2"
                    shift 2
                    ;;
                --connectivity)
                    CONNECTIVITY="$2"
                    shift 2
                    ;;
                --n-comparison-chunks)
                    N_COMP_CHUNKS="$2"
                    shift 2
                    ;;
                --output-suffix)
                    OUTPUT_SUFFIX="$2"
                    EXTRA_ARGS="$EXTRA_ARGS --output-suffix $2"
                    shift 2
                    ;;
                --perturbation-frame-label)
                    PERTURB_FRAME_LABEL="$2"
                    shift 2
                    ;;
                --refractory)
                    REFRACTORY="$2"
                    EXTRA_ARGS="$EXTRA_ARGS --refractory $2"
                    shift 2
                    ;;
                *)
                    EXTRA_ARGS="$EXTRA_ARGS $1"
                    shift
                    ;;
            esac
        done

        set_connectivity_paths
        CONN_FLAG="--connectivity $CONNECTIVITY"
        METRIC_DIR="${METRIC}${OUTPUT_SUFFIX}"

        # Array task -> frame-label mapping
        if [ -z "$FRAME_LABEL" ] && [ -n "$SLURM_ARRAY_TASK_ID" ]; then
            case "$SLURM_ARRAY_TASK_ID" in
                0) FRAME_LABEL="prestim" ;;
                1) FRAME_LABEL="full_trial" ;;
                2) FRAME_LABEL="nostim" ;;
                *)
                    echo "ERROR: Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (expected 0, 1, or 2)"
                    exit 1
                    ;;
            esac
            echo "Array task $SLURM_ARRAY_TASK_ID -> frame label: $FRAME_LABEL"
        fi

        echo "Metric: $METRIC"
        echo "Workers: $WORKERS"
        echo "Max timesteps: $MAX_TIMESTEPS"
        echo "Skip comparison: $SKIP_COMPARISON"
        echo "Analysis only: $ANALYSIS_ONLY"
        if [ -n "$FRAME_LABEL" ]; then
            echo "Frame label: $FRAME_LABEL"
        fi

        # --- Run comparison ---
        if [ "$SKIP_COMPARISON" = false ]; then
            echo ""
            echo "=============================================="
            echo "Running Ising Comparison Analysis"
            echo "=============================================="

            FRAME_ARG=""
            if [ -n "$FRAME_LABEL" ]; then
                FRAME_ARG="--frame-label $FRAME_LABEL"
            fi

            if [ "$N_COMP_CHUNKS" -gt 1 ]; then
                # --- Chunked comparison: submit chunk workers + merge ---
                echo "Chunked mode: $N_COMP_CHUNKS chunks"
                echo "Detecting total sim count..."
                N_SIMS=$(python -c "from glob import glob; print(len(glob('$ISING_DATA/sim_be_*.mat')))")
                CHUNK_SIZE=$(( (N_SIMS + N_COMP_CHUNKS - 1) / N_COMP_CHUNKS ))
                echo "Total sims: $N_SIMS, chunk size: $CHUNK_SIZE"

                COMP_OUTPUT="$COMPARISON_OUTPUT/$METRIC"
                mkdir -p "$COMP_OUTPUT/logs"

                CHUNK_IDS=()
                for ci in $(seq 0 $((N_COMP_CHUNKS - 1))); do
                    SIM_START=$((ci * CHUNK_SIZE))
                    SIM_END=$(( (ci + 1) * CHUNK_SIZE ))
                    if [ $SIM_END -gt $N_SIMS ]; then SIM_END=$N_SIMS; fi

                    CID=$(sbatch --parsable \
                        --partition=defaultp --export=NONE \
                        --cpus-per-task=96 --mem=128G --time=06:00:00 \
                        --job-name="comp_${FRAME_LABEL}_c${ci}" \
                        --output="$COMP_OUTPUT/logs/comp_${FRAME_LABEL}_c${ci}_%j.out" \
                        --error="$COMP_OUTPUT/logs/comp_${FRAME_LABEL}_c${ci}_%j.err" \
                        "$SCRIPT_DIR/run_ising_pipeline.sh" comparison \
                            --metric "$METRIC" --workers 96 \
                            --max-timesteps "$MAX_TIMESTEPS" \
                            --connectivity "$CONNECTIVITY" \
                            --no-cache \
                            $FRAME_ARG \
                            --sim-start "$SIM_START" --sim-end "$SIM_END" \
                            $EXTRA_ARGS)
                    echo "  Chunk $ci: sims [$SIM_START:$SIM_END] -> job $CID"
                    CHUNK_IDS+=("$CID")
                done

                # Merge job: depends on all chunks
                CHUNK_DEP=$(IFS=:; echo "${CHUNK_IDS[*]}")
                MERGE_JOB=$(sbatch --parsable \
                    --dependency=afterok:"$CHUNK_DEP" \
                    --partition=defaultp --export=NONE \
                    --cpus-per-task=16 --mem=64G --time=02:00:00 \
                    --job-name="merge_${FRAME_LABEL}" \
                    --output="$COMP_OUTPUT/logs/merge_${FRAME_LABEL}_%j.out" \
                    --error="$COMP_OUTPUT/logs/merge_${FRAME_LABEL}_%j.err" \
                    "$SCRIPT_DIR/run_ising_pipeline.sh" comparison \
                        --metric "$METRIC" --connectivity "$CONNECTIVITY" \
                        $FRAME_ARG --merge-chunks \
                        $EXTRA_ARGS)
                echo "  Merge job: $MERGE_JOB (depends on ${#CHUNK_IDS[@]} chunks)"

                # Override EXIT_CODE and DEP_JOB_ID so downstream depends on merge
                EXIT_CODE=0
                DEP_JOB_ID="$MERGE_JOB"
            else
                # --- Single-task comparison (original behavior) ---
                echo "Workers: $WORKERS"
                echo "Numba threads: 1"

                srun python Figure5_IsingComparison_optimized.py \
                    --metric "$METRIC" \
                    --n-workers "$WORKERS" \
                    --numba-threads 1 \
                    --checkpoint-interval 500 \
                    --max-timesteps "$MAX_TIMESTEPS" \
                    $CONN_FLAG \
                    $FRAME_ARG \
                    $EXTRA_ARGS

                EXIT_CODE=$?
                echo "Comparison exit code: $EXIT_CODE"

                if command -v sacct &> /dev/null; then
                    echo ""
                    echo "Job statistics:"
                    sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize,State
                fi
            fi
        else
            echo ""
            echo "=============================================="
            echo "Skipping comparison (--skip-comparison)"
            echo "=============================================="
            EXIT_CODE=0
        fi

        # --- Submit downstream jobs ---
        # In array mode, only task 0 submits downstream jobs.
        # afterok:<array_base_id> waits for ALL array tasks to complete successfully.
        SHOULD_SUBMIT_DOWNSTREAM=true
        if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
            SHOULD_SUBMIT_DOWNSTREAM=false
            echo "Array task $SLURM_ARRAY_TASK_ID: skipping downstream submission (handled by task 0)"
        fi

        if [ "$SHOULD_SUBMIT_DOWNSTREAM" = true ] && [ $EXIT_CODE -eq 0 ]; then
            # Use array base job ID for dependencies (waits for ALL array tasks)
            # In chunked mode, DEP_JOB_ID was already set to the merge job
            if [ -z "${DEP_JOB_ID:-}" ]; then
                DEP_JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
            fi
            PERTURB_DIR="$PERTURBATION_OUTPUT/${METRIC}${OUTPUT_SUFFIX}"
            PIPELINE_SCRIPT="$SCRIPT_DIR/run_ising_pipeline.sh"

            mkdir -p "$PERTURB_DIR"
            mkdir -p "$PERTURB_DIR/logs"

            if [ "$ANALYSIS_ONLY" = true ]; then
                # --- Analysis-only: submit just the analyze job ---
                echo ""
                echo "=============================================="
                echo "Submitting analysis job only (--analysis-only)..."
                echo "=============================================="

                ANALYZE_JOB=$(sbatch --parsable \
                    --dependency=afterok:$DEP_JOB_ID \
                    --partition=defaultp \
                    --constraint=matlab \
                    --export=NONE \
                    --cpus-per-task=4 \
                    --mem=96G \
                    --time=04:00:00 \
                    --job-name="analyze_$METRIC" \
                    --output="$PERTURB_DIR/logs/analyze_%j.out" \
                    --error="$PERTURB_DIR/logs/analyze_%j.err" \
                    "$PIPELINE_SCRIPT" analyze --metric "$METRIC" --stim-mode clamped \
                        --connectivity "$CONNECTIVITY" \
                        ${REFRACTORY:+--refractory "$REFRACTORY"} \
                        ${OUTPUT_SUFFIX:+--output-suffix "$OUTPUT_SUFFIX"} \
                        --results "$PERTURB_DIR/PerturbationResults_*.mat")

                echo "Submitted analysis job: $ANALYZE_JOB (depends on $DEP_JOB_ID)"
                echo "Output directory: $PERTURB_DIR/clamped/Analysis"
            else
                # --- Full chain: perturbation -> retry -> combine -> analyze ---
                echo ""
                echo "=============================================="
                echo "Submitting perturbation pipeline..."
                echo "=============================================="

                # Dynamically compute array range from comparison results
                COMPARISON_FILE="$ISING_DATA/IsingComparison/${METRIC}${OUTPUT_SUFFIX}/IsingComparison_Results_${PERTURB_FRAME_LABEL}_subselect_centre_vs_tiled_${METRIC}${CONN_SUFFIX}_optimized.mat"
                BATCH_SIZE=10
                TOTAL_JOBS=$(python "$SCRIPT_DIR/run_ising_perturbations.py" \
                    --comparison "$COMPARISON_FILE" --ising-data "$ISING_DATA" --count-jobs | tail -1)
                ARRAY_MAX=$(( (TOTAL_JOBS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
                echo "Computed: $TOTAL_JOBS total jobs -> $((ARRAY_MAX+1)) array tasks (batch_size=$BATCH_SIZE)"

                PERTURB_JOB=$(sbatch --parsable \
                    --dependency=afterok:$DEP_JOB_ID \
                    --partition=defaultp \
                    --export=NONE \
                    --array=0-$ARRAY_MAX \
                    --cpus-per-task=1 \
                    --mem=2G \
                    --time=04:00:00 \
                    --job-name="perturb_$METRIC" \
                    --output="$PERTURB_DIR/logs/perturb_%A_%a.out" \
                    --error="$PERTURB_DIR/logs/perturb_%A_%a.err" \
                    "$PIPELINE_SCRIPT" perturbations --metric "$METRIC" --frame-label "$PERTURB_FRAME_LABEL" \
                        --output "$PERTURB_DIR" --ising-data "$ISING_DATA" \
                        --connectivity "$CONNECTIVITY" \
                        ${OUTPUT_SUFFIX:+--output-suffix "$OUTPUT_SUFFIX"} \
                        $FORCE_FLAG ${SEED_OFFSET:+--seed-offset $SEED_OFFSET} \
                        ${N_REPLICATES_OVERRIDE:+--n-replicates $N_REPLICATES_OVERRIDE})

                RETRY_JOB=$(sbatch --parsable \
                    --dependency=afterany:$PERTURB_JOB \
                    --partition=defaultp \
                    --export=NONE \
                    --cpus-per-task=1 \
                    --mem=4G \
                    --time=03:00:00 \
                    --job-name="retry_$METRIC" \
                    --output="$PERTURB_DIR/logs/retry_%j.out" \
                    --error="$PERTURB_DIR/logs/retry_%j.err" \
                    "$PIPELINE_SCRIPT" retry --parent-job "$PERTURB_JOB" --metric "$METRIC" \
                        --frame-label "$PERTURB_FRAME_LABEL" --output "$PERTURB_DIR" --ising-data "$ISING_DATA" \
                        --connectivity "$CONNECTIVITY" \
                        ${OUTPUT_SUFFIX:+--output-suffix "$OUTPUT_SUFFIX"} \
                        $FORCE_FLAG ${SEED_OFFSET:+--seed-offset $SEED_OFFSET} \
                        ${N_REPLICATES_OVERRIDE:+--n-replicates $N_REPLICATES_OVERRIDE})

                COMBINE_JOB=$(sbatch --parsable \
                    --dependency=afterok:$RETRY_JOB \
                    --partition=defaultp \
                    --export=NONE \
                    --cpus-per-task=4 \
                    --mem=64G \
                    --time=04:00:00 \
                    --job-name="combine_$METRIC" \
                    --output="$PERTURB_DIR/logs/combine_%j.out" \
                    --error="$PERTURB_DIR/logs/combine_%j.err" \
                    "$PIPELINE_SCRIPT" combine --metric "$METRIC" --frame-label "$PERTURB_FRAME_LABEL" \
                        --output "$PERTURB_DIR" --ising-data "$ISING_DATA" \
                        --connectivity "$CONNECTIVITY" \
                        ${OUTPUT_SUFFIX:+--output-suffix "$OUTPUT_SUFFIX"} \
                        ${N_REPLICATES_OVERRIDE:+--n-replicates $N_REPLICATES_OVERRIDE})

                ANALYZE_JOB=$(sbatch --parsable \
                    --dependency=afterok:$COMBINE_JOB \
                    --partition=defaultp \
                    --constraint=matlab \
                    --export=NONE \
                    --cpus-per-task=4 \
                    --mem=96G \
                    --time=04:00:00 \
                    --job-name="analyze_$METRIC" \
                    --output="$PERTURB_DIR/logs/analyze_%j.out" \
                    --error="$PERTURB_DIR/logs/analyze_%j.err" \
                    "$PIPELINE_SCRIPT" analyze --metric "$METRIC" --stim-mode clamped \
                        --connectivity "$CONNECTIVITY" \
                        ${REFRACTORY:+--refractory "$REFRACTORY"} \
                        ${OUTPUT_SUFFIX:+--output-suffix "$OUTPUT_SUFFIX"} \
                        --results "$PERTURB_DIR/PerturbationResults_*.mat")

                echo "Submitted perturbation pipeline:"
                echo "  Perturbation array job: $PERTURB_JOB ($((ARRAY_MAX+1)) tasks x $BATCH_SIZE jobs/task = $TOTAL_JOBS total jobs)"
                echo "  Retry job: $RETRY_JOB (depends on $PERTURB_JOB, checks for failures)"
                echo "  Combine job: $COMBINE_JOB (depends on $RETRY_JOB)"
                echo "  Analysis job: $ANALYZE_JOB (depends on $COMBINE_JOB)"
                echo ""
                echo "Output directory: $PERTURB_DIR"
                echo ""
                echo "Manual retry one-liner (if needed later):"
                echo "  sbatch --array=\$(sacct -j $PERTURB_JOB --format=JobID%30,State -n | grep -E 'FAILED|NODE_FAIL|TIMEOUT' | grep '_' | sed 's/.*_\\([0-9]*\\).*/\\1/' | sort -n | uniq | tr '\\n' ',' | sed 's/,\$//') --cpus-per-task=1 --mem=2G --time=00:20:00 --partition=defaultp --export=NONE --job-name='manual_retry' --output='$PERTURB_DIR/logs/manual_retry_%A_%a.out' --error='$PERTURB_DIR/logs/manual_retry_%A_%a.err' '$PIPELINE_SCRIPT' perturbations --metric '$METRIC' --output '$PERTURB_DIR' --ising-data '$ISING_DATA'"
            fi
        fi

        echo ""
        echo "=============================================="
        echo "fullPipeline completed"
        echo "Exit code: $EXIT_CODE"
        echo "End time: $(date)"
        echo "=============================================="

        exit $EXIT_CODE
        ;;

    # =========================================================================
    # COMPARISON MODE
    # =========================================================================
    comparison)
        echo ""
        echo "=============================================="
        echo "Running Ising Comparison Analysis"
        echo "=============================================="

        # Environment optimized for process-level parallelism
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMBA_NUM_THREADS=1
        export NUMBA_THREADING_LAYER=omp
        export PYTHONUNBUFFERED=1

        # Default arguments for comparison
        METRIC="combined"
        WORKERS="${SLURM_CPUS_PER_TASK:-64}"
        MAX_TIMESTEPS="10000"
        EXTRA_ARGS=""
        CONNECTIVITY="rook"

        # Parse comparison-specific arguments
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --metric)
                    METRIC="$2"
                    shift 2
                    ;;
                --workers)
                    WORKERS="$2"
                    shift 2
                    ;;
                --max-timesteps)
                    MAX_TIMESTEPS="$2"
                    shift 2
                    ;;
                --connectivity)
                    CONNECTIVITY="$2"
                    shift 2
                    ;;
                *)
                    EXTRA_ARGS="$EXTRA_ARGS $1"
                    shift
                    ;;
            esac
        done

        set_connectivity_paths

        echo "Metric: $METRIC"
        echo "Workers: $WORKERS"
        echo "Max timesteps: $MAX_TIMESTEPS"

        python Figure5_IsingComparison_optimized.py \
            --metric "$METRIC" \
            --n-workers "$WORKERS" \
            --numba-threads 1 \
            --checkpoint-interval 500 \
            --max-timesteps "$MAX_TIMESTEPS" \
            --connectivity "$CONNECTIVITY" \
            $EXTRA_ARGS
        ;;

    # =========================================================================
    # PERTURBATIONS MODE (Array Job)
    # =========================================================================
    perturbations)
        echo ""
        echo "=============================================="
        echo "Running Perturbation Experiment"
        echo "=============================================="

        # Single-CPU jobs use single thread
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1

        # Isolate Numba cache per array task to avoid contention
        export NUMBA_CACHE_DIR="/tmp/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

        # Check if running as array job
        if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
            echo "ERROR: perturbations mode requires --array flag"
            echo "Usage: Use --count-jobs to compute the array range dynamically:"
            echo "  TOTAL=\$(python run_ising_perturbations.py --comparison <path> --count-jobs)"
            echo "  ARRAY_MAX=\$(( (TOTAL + 10 - 1) / 10 - 1 ))"
            echo "  sbatch --array=0-\$ARRAY_MAX run_ising_pipeline.sh perturbations --metric <metric>"
            exit 1
        fi

        # Parse perturbations-specific arguments
        OUTPUT="$PERTURBATION_OUTPUT"
        ISING_PATH="$ISING_DATA"
        METRIC=""
        FRAME_LABEL="full_trial"
        BATCH_SIZE=10  # Default: 10 jobs per array task (use --count-jobs for exact total)
        FORCE_FLAG=""
        SEED_OFFSET=""
        N_REPLICATES_OVERRIDE=""
        MODES_FILTER=""
        FILTER_CONDITIONS=""
        FILTER_SIMS=""
        FILTER_SIZES=""
        FILTER_BIASES=""
        CONNECTIVITY="rook"
        OUTPUT_SUFFIX=""

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --output)
                    OUTPUT="$2"
                    shift 2
                    ;;
                --ising-data)
                    ISING_PATH="$2"
                    shift 2
                    ;;
                --metric)
                    METRIC="$2"
                    shift 2
                    ;;
                --frame-label)
                    FRAME_LABEL="$2"
                    shift 2
                    ;;
                --batch-size)
                    BATCH_SIZE="$2"
                    shift 2
                    ;;
                --force)
                    FORCE_FLAG="--force"
                    shift
                    ;;
                --seed-offset)
                    SEED_OFFSET="$2"
                    shift 2
                    ;;
                --n-replicates)
                    N_REPLICATES_OVERRIDE="$2"
                    shift 2
                    ;;
                --modes)
                    MODES_FILTER="$2"
                    shift 2
                    ;;
                --filter-conditions)
                    FILTER_CONDITIONS="$2"
                    shift 2
                    ;;
                --filter-sims)
                    FILTER_SIMS="$2"
                    shift 2
                    ;;
                --filter-sizes)
                    FILTER_SIZES="$2"
                    shift 2
                    ;;
                --filter-biases)
                    FILTER_BIASES="$2"
                    shift 2
                    ;;
                --connectivity)
                    CONNECTIVITY="$2"
                    shift 2
                    ;;
                --output-suffix)
                    OUTPUT_SUFFIX="$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done

        set_connectivity_paths

        # Append metric subdirectory when metric is provided and output wasn't overridden
        if [ -n "$METRIC" ] && [ "$OUTPUT" = "$PERTURBATION_OUTPUT" ]; then
            OUTPUT="$OUTPUT/${METRIC}${OUTPUT_SUFFIX}"
        fi

        mkdir -p "$OUTPUT"

        echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
        echo "Batch size: $BATCH_SIZE"
        echo "Output: $OUTPUT"
        echo "Ising Data: $ISING_PATH"

        # Convert --metric to --comparison path
        COMPARISON_PATH=""
        if [ -n "$METRIC" ]; then
            COMPARISON_PATH="$ISING_PATH/IsingComparison/${METRIC}${OUTPUT_SUFFIX}/IsingComparison_Results_${FRAME_LABEL}_subselect_centre_vs_tiled_${METRIC}${CONN_SUFFIX}_optimized.mat"
            echo "Metric: $METRIC"
            echo "Frame label: $FRAME_LABEL"
            echo "Comparison file: $COMPARISON_PATH"
        fi

        # Build command with batch-size support
        CMD="python run_ising_perturbations.py --output \"$OUTPUT\" --ising-data \"$ISING_PATH\" --index \"$SLURM_ARRAY_TASK_ID\" --batch-size \"$BATCH_SIZE\" --connectivity \"$CONNECTIVITY\""
        if [ -n "$COMPARISON_PATH" ]; then
            CMD="$CMD --comparison \"$COMPARISON_PATH\""
        fi
        if [ -n "$FORCE_FLAG" ]; then CMD="$CMD --force"; fi
        if [ -n "$SEED_OFFSET" ]; then CMD="$CMD --seed-offset \"$SEED_OFFSET\""; fi
        if [ -n "$N_REPLICATES_OVERRIDE" ]; then CMD="$CMD --n-replicates \"$N_REPLICATES_OVERRIDE\""; fi
        if [ -n "$MODES_FILTER" ]; then CMD="$CMD --modes \"$MODES_FILTER\""; fi
        if [ -n "$FILTER_CONDITIONS" ]; then CMD="$CMD --filter-conditions \"$FILTER_CONDITIONS\""; fi
        if [ -n "$FILTER_SIMS" ];       then CMD="$CMD --filter-sims \"$FILTER_SIMS\""; fi
        if [ -n "$FILTER_SIZES" ];      then CMD="$CMD --filter-sizes \"$FILTER_SIZES\""; fi
        if [ -n "$FILTER_BIASES" ];     then CMD="$CMD --filter-biases \"$FILTER_BIASES\""; fi

        srun /bin/bash -c "$CMD"
        ;;

    # =========================================================================
    # COMBINE MODE
    # =========================================================================
    combine)
        echo ""
        echo "=============================================="
        echo "Combining Perturbation Results"
        echo "=============================================="

        export OMP_NUM_THREADS=1

        # Parse combine-specific arguments
        OUTPUT="$PERTURBATION_OUTPUT"
        ISING_PATH="$ISING_DATA"
        METRIC=""
        FRAME_LABEL="full_trial"
        N_REPLICATES_OVERRIDE=""
        COMBINE_MODE=""   # If set: combine just this one mode; else: fan out
        CONNECTIVITY="rook"
        OUTPUT_SUFFIX=""

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --output)
                    OUTPUT="$2"
                    shift 2
                    ;;
                --ising-data)
                    ISING_PATH="$2"
                    shift 2
                    ;;
                --metric)
                    METRIC="$2"
                    shift 2
                    ;;
                --frame-label)
                    FRAME_LABEL="$2"
                    shift 2
                    ;;
                --n-replicates)
                    N_REPLICATES_OVERRIDE="$2"
                    shift 2
                    ;;
                --mode)
                    COMBINE_MODE="$2"
                    shift 2
                    ;;
                --connectivity)
                    CONNECTIVITY="$2"
                    shift 2
                    ;;
                --output-suffix)
                    OUTPUT_SUFFIX="$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done

        set_connectivity_paths

        # Append metric subdirectory when metric is provided and output wasn't overridden
        if [ -n "$METRIC" ] && [ "$OUTPUT" = "$PERTURBATION_OUTPUT" ]; then
            OUTPUT="$OUTPUT/${METRIC}${OUTPUT_SUFFIX}"
        fi

        echo "Output directory: $OUTPUT"

        # Convert --metric to --comparison path
        COMPARISON_PATH=""
        if [ -n "$METRIC" ]; then
            COMPARISON_PATH="$ISING_PATH/IsingComparison/${METRIC}${OUTPUT_SUFFIX}/IsingComparison_Results_${FRAME_LABEL}_subselect_centre_vs_tiled_${METRIC}${CONN_SUFFIX}_optimized.mat"
            echo "Metric: $METRIC"
            echo "Frame label: $FRAME_LABEL"
            echo "Comparison file: $COMPARISON_PATH"
        fi

        # =====================================================================
        # FAN-OUT MODE: no --mode given → submit one child job per stim mode
        # =====================================================================
        if [ -z "$COMBINE_MODE" ]; then
            # Mirror STIMULUS_MODES from run_ising_perturbations.py
            ALL_MODES=(clamped \
                       double_pulse double_pulse3 double_pulse5 double_pulse10 \
                       bias \
                       double_pulse_bias double_pulse_bias3 double_pulse_bias5 double_pulse_bias10)

            mkdir -p "$OUTPUT/logs"
            CHILD_IDS=()
            echo ""
            echo "Fanning out combine across ${#ALL_MODES[@]} stimulus modes:"
            for m in "${ALL_MODES[@]}"; do
                CID=$(sbatch --parsable \
                    --partition=defaultp --export=NONE \
                    --cpus-per-task=1 --mem=8G --time=01:30:00 \
                    --job-name="combine_${m}_${METRIC}" \
                    --output="$OUTPUT/logs/combine_${m}_%j.out" \
                    --error="$OUTPUT/logs/combine_${m}_%j.err" \
                    "$SCRIPT_DIR/run_ising_pipeline.sh" combine \
                        --mode "$m" \
                        ${METRIC:+--metric "$METRIC"} \
                        --frame-label "$FRAME_LABEL" \
                        --output "$OUTPUT" --ising-data "$ISING_PATH" \
                        --connectivity "$CONNECTIVITY" \
                        ${OUTPUT_SUFFIX:+--output-suffix "$OUTPUT_SUFFIX"} \
                        ${N_REPLICATES_OVERRIDE:+--n-replicates "$N_REPLICATES_OVERRIDE"})
                echo "  Submitted combine_${m}: $CID"
                CHILD_IDS+=("$CID")
            done

            # Sentinel job: depends on ALL children. We --wait on it here so
            # this outer combine job's exit code reflects the full per-mode
            # fan-out's status — letting `fullPipeline`'s
            # `--dependency=afterok:$COMBINE_JOB` chain work correctly.
            DEP_LIST=$(IFS=:; echo "${CHILD_IDS[*]}")
            echo ""
            echo "Submitting sentinel (waits for all ${#CHILD_IDS[@]} children)..."
            sbatch --parsable --wait \
                --dependency=afterok:"$DEP_LIST" \
                --partition=defaultp --export=NONE \
                --cpus-per-task=1 --mem=1G --time=00:05:00 \
                --job-name="combine_done_${METRIC}" \
                --output="$OUTPUT/logs/combine_done_%j.out" \
                --error="$OUTPUT/logs/combine_done_%j.err" \
                --wrap="echo 'All ${#ALL_MODES[@]} per-mode combine jobs completed.'; ls -lh '$OUTPUT'/PerturbationResults_*.mat 2>/dev/null | tail -20"
            SENTINEL_RC=$?
            echo "Sentinel exit code: $SENTINEL_RC"
            exit $SENTINEL_RC
        fi

        # =====================================================================
        # WORKER MODE: --mode given → actually run the combine for one mode
        # =====================================================================
        echo "Per-mode combine: $COMBINE_MODE"

        CMD="python run_ising_perturbations.py --output \"$OUTPUT\" --ising-data \"$ISING_PATH\" --combine --mode \"$COMBINE_MODE\" --format mat --connectivity \"$CONNECTIVITY\""
        if [ -n "$COMPARISON_PATH" ]; then
            CMD="$CMD --comparison \"$COMPARISON_PATH\""
        fi
        if [ -n "$N_REPLICATES_OVERRIDE" ]; then
            CMD="$CMD --n-replicates $N_REPLICATES_OVERRIDE"
        fi

        srun /bin/bash -c "$CMD"
        ;;

    # =========================================================================
    # ANALYZE MODE (MATLAB version)
    # =========================================================================
    analyze)
        echo ""
        echo "=============================================="
        echo "Analyzing Perturbation Results (MATLAB)"
        echo "=============================================="

        # Parse analyze-specific arguments
        # Note: RESULTS_PATH default is *.npz but MATLAB needs *.mat
        # The MATLAB script will find .mat files automatically
        RESULTS_PATH="$PERTURBATION_OUTPUT/PerturbationResults_*.npz"
        OUTPUT=""
        METRIC=""
        FIGURES_ONLY=false
        STIM_MODE="clamped"
        CONNECTIVITY="rook"
        REFRACTORY=""
        OUTPUT_SUFFIX=""

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --results)
                    RESULTS_PATH="$2"
                    shift 2
                    ;;
                --output)
                    OUTPUT="$2"
                    shift 2
                    ;;
                --metric)
                    METRIC="$2"
                    shift 2
                    ;;
                --figures-only)
                    FIGURES_ONLY=true
                    shift
                    ;;
                --stim-mode)
                    STIM_MODE="$2"
                    shift 2
                    ;;
                --connectivity)
                    CONNECTIVITY="$2"
                    shift 2
                    ;;
                --refractory)
                    REFRACTORY="$2"
                    shift 2
                    ;;
                --output-suffix)
                    OUTPUT_SUFFIX="$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done

        set_connectivity_paths

        # --- stim-mode=all / dp10_full: fan out to multiple modes in parallel ---
        # 'all'      : every raw STIMULUS_MODE (legacy behavior; bias modes use
        #              the high_/low_ collapse internally).
        # 'dp10_full': double_pulse10 + double_pulse_bias10 expanded across all
        #              6 STIMULUS_BIAS_VALUES — 7 jobs total. Each bias-encoded
        #              stim-mode is handled by Figure5_IsingPerturbationAnalysis.m
        #              via its per-invocation slicing path (see SECTION 3).
        if [ "$STIM_MODE" = "all" ] || [ "$STIM_MODE" = "dp10_full" ]; then
            echo "Launching fan-out for STIM_MODE=$STIM_MODE..."
            FAN_ARGS=""
            if [ -n "$METRIC" ]; then FAN_ARGS="$FAN_ARGS --metric $METRIC"; fi
            if [ "$FIGURES_ONLY" = true ]; then FAN_ARGS="$FAN_ARGS --figures-only"; fi
            if [ -n "$OUTPUT" ]; then FAN_ARGS="$FAN_ARGS --output $OUTPUT"; fi
            if [ -n "$REFRACTORY" ]; then FAN_ARGS="$FAN_ARGS --refractory $REFRACTORY"; fi
            if [ -n "$OUTPUT_SUFFIX" ]; then FAN_ARGS="$FAN_ARGS --output-suffix $OUTPUT_SUFFIX"; fi
            if [ "$CONNECTIVITY" != "rook" ]; then FAN_ARGS="$FAN_ARGS --connectivity $CONNECTIVITY"; fi
            if [ "$RESULTS_PATH" != "$PERTURBATION_OUTPUT/PerturbationResults_*.npz" ]; then
                FAN_ARGS="$FAN_ARGS --results $RESULTS_PATH"
            fi

            if [ "$STIM_MODE" = "all" ]; then
                # NOTE: pass RAW mode names only (not high_*/low_* display modes).
                # The MATLAB script splits each bias-using mode into high+low
                # display traces internally.
                FAN_MODES=(clamped \
                           double_pulse double_pulse3 double_pulse5 double_pulse10 \
                           bias \
                           double_pulse_bias double_pulse_bias3 double_pulse_bias5 double_pulse_bias10)
            else
                # dp10_full: clamped dp10 + 16 bias-encoded variants of double_pulse_bias10
                FAN_MODES=(double_pulse10 \
                           double_pulse_bias10_0p15 \
                           double_pulse_bias10_0p25 \
                           double_pulse_bias10_0p50 \
                           double_pulse_bias10_0p75 \
                           double_pulse_bias10_1p00 \
                           double_pulse_bias10_1p25 \
                           double_pulse_bias10_1p50 \
                           double_pulse_bias10_1p75 \
                           double_pulse_bias10_2p00 \
                           double_pulse_bias10_4p00 \
                           double_pulse_bias10_6p00 \
                           double_pulse_bias10_8p00 \
                           double_pulse_bias10_10p00 \
                           double_pulse_bias10_12p00 \
                           double_pulse_bias10_14p00 \
                           double_pulse_bias10_16p00)
            fi

            for mode in "${FAN_MODES[@]}"; do
                echo "  Submitting --stim-mode $mode"
                bash "$SCRIPT_DIR/run_ising_pipeline.sh" analyze $FAN_ARGS --stim-mode "$mode" &
            done
            wait
            exit 0
        fi

        # Default paths with metric subdirectory (honor OUTPUT_SUFFIX so
        # variants like _40-60_box24k land in their own dirs instead of the
        # baseline METRIC dir).
        METRIC_DIR="${METRIC}${OUTPUT_SUFFIX}"
        if [ -n "$METRIC" ]; then
            RESULTS_PATH="$PERTURBATION_OUTPUT/$METRIC_DIR/PerturbationResults_*.npz"
        fi
        if [ -z "$OUTPUT" ]; then
            OUTPUT="$PERTURBATION_OUTPUT"
            if [ -n "$METRIC" ]; then
                OUTPUT="$PERTURBATION_OUTPUT/$METRIC_DIR/$STIM_MODE/Analysis"
            fi
        fi

        mkdir -p "$OUTPUT"

        # Get directory from RESULTS_PATH (strip the filename pattern)
        RESULTS_DIR=$(dirname "$RESULTS_PATH")

        echo "Results directory: $RESULTS_DIR"
        echo "Output directory: $OUTPUT"
        if [ -n "$METRIC" ]; then
            echo "Metric: $METRIC"
        fi
        echo "Stim mode: $STIM_MODE"

        # Load MATLAB module (requires --constraint=matlab when submitting)
        if ! module load matlab 2>/dev/null; then
            echo "ERROR: MATLAB failed to load on this node ($HOSTNAME)."
            echo "Resubmit with: sbatch --constraint=matlab ..."
            exit 2
        fi

        # Build MATLAB command - set config paths, let MATLAB find .mat files
        MATLAB_CMD="config = struct(); "
        MATLAB_CMD+="config.perturbationResultsPath = '$RESULTS_DIR'; "
        MATLAB_CMD+="config.outputPath = '$OUTPUT'; "
        if [ -n "$METRIC" ]; then
            MATLAB_CMD+="config.comparisonResultsPath = '$COMPARISON_OUTPUT/$METRIC_DIR'; "
        else
            MATLAB_CMD+="config.comparisonResultsPath = '$COMPARISON_OUTPUT'; "
        fi
        MATLAB_CMD+="config.stimMode = '$STIM_MODE'; "
        if [ "$FIGURES_ONLY" = true ]; then
            MATLAB_CMD+="config.figuresOnly = true; "
        fi
        MATLAB_CMD+="cd('$SCRIPT_DIR'); "
        MATLAB_CMD+="Figure5_IsingPerturbationAnalysis"

        echo "Running MATLAB analysis..."
        echo "MATLAB command: $MATLAB_CMD"
        srun matlab -batch "$MATLAB_CMD"
        ;;

    # =========================================================================
    # RETRY MODE - Check for failed tasks and resubmit
    # =========================================================================
    retry)
        echo ""
        echo "=============================================="
        echo "Checking for Failed Tasks"
        echo "=============================================="

        # Parse retry-specific arguments
        PARENT_JOB=""
        OUTPUT="$PERTURBATION_OUTPUT"
        ISING_PATH="$ISING_DATA"
        METRIC=""
        FRAME_LABEL="full_trial"
        MAX_RETRIES=3
        BATCH_SIZE=10  # Must match original job
        FORCE_FLAG=""
        SEED_OFFSET=""
        N_REPLICATES_OVERRIDE=""
        CONNECTIVITY="rook"
        OUTPUT_SUFFIX=""

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --parent-job)
                    PARENT_JOB="$2"
                    shift 2
                    ;;
                --output)
                    OUTPUT="$2"
                    shift 2
                    ;;
                --ising-data)
                    ISING_PATH="$2"
                    shift 2
                    ;;
                --metric)
                    METRIC="$2"
                    shift 2
                    ;;
                --frame-label)
                    FRAME_LABEL="$2"
                    shift 2
                    ;;
                --max-retries)
                    MAX_RETRIES="$2"
                    shift 2
                    ;;
                --batch-size)
                    BATCH_SIZE="$2"
                    shift 2
                    ;;
                --force)
                    FORCE_FLAG="--force"
                    shift
                    ;;
                --seed-offset)
                    SEED_OFFSET="$2"
                    shift 2
                    ;;
                --n-replicates)
                    N_REPLICATES_OVERRIDE="$2"
                    shift 2
                    ;;
                --connectivity)
                    CONNECTIVITY="$2"
                    shift 2
                    ;;
                --output-suffix)
                    OUTPUT_SUFFIX="$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done

        set_connectivity_paths

        if [ -z "$PARENT_JOB" ]; then
            echo "ERROR: --parent-job is required for retry mode"
            exit 1
        fi

        echo "Parent job: $PARENT_JOB"
        echo "Output: $OUTPUT"
        echo "Ising Data: $ISING_PATH"
        echo "Metric: $METRIC"
        echo "Max retries: $MAX_RETRIES"
        echo "Batch size: $BATCH_SIZE"

        # Function to get failed task indices from a job
        get_failed_tasks() {
            local JOB_ID="$1"
            # Get failed tasks: extract array index from JobID like "12345_42"
            # Filter for FAILED state, extract the array index part
            sacct -j "$JOB_ID" --format=JobID%30,State%15 -n 2>/dev/null | \
                grep -E "FAILED|NODE_FAIL|TIMEOUT" | \
                grep "_" | \
                sed 's/.*_\([0-9]*\).*/\1/' | \
                sort -n | uniq | tr '\n' ',' | sed 's/,$//'
        }

        CURRENT_JOB="$PARENT_JOB"
        RETRY_COUNT=0

        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            echo ""
            echo "--- Retry attempt $((RETRY_COUNT + 1)) of $MAX_RETRIES ---"
            echo "Checking job $CURRENT_JOB for failed tasks..."

            # Wait a moment for sacct to update
            sleep 5

            FAILED_TASKS=$(get_failed_tasks "$CURRENT_JOB")

            if [ -z "$FAILED_TASKS" ]; then
                echo "No failed tasks found. All tasks completed successfully!"
                exit 0
            fi

            # Count failed tasks
            FAILED_COUNT=$(echo "$FAILED_TASKS" | tr ',' '\n' | wc -l)
            echo "Found $FAILED_COUNT failed tasks: $FAILED_TASKS"

            # Resubmit failed tasks
            echo "Resubmitting failed tasks..."

            RETRY_JOB=$(sbatch --parsable \
                --partition=defaultp \
                --export=NONE \
                --array="$FAILED_TASKS" \
                --cpus-per-task=1 \
                --mem=2G \
                --time=02:00:00 \
                --job-name="retry${RETRY_COUNT}_${METRIC:-perturb}" \
                --output="$OUTPUT/logs/retry${RETRY_COUNT}_%A_%a.out" \
                --error="$OUTPUT/logs/retry${RETRY_COUNT}_%A_%a.err" \
                "$SCRIPT_DIR/run_ising_pipeline.sh" perturbations \
                    --metric "$METRIC" --frame-label "$FRAME_LABEL" \
                    --output "$OUTPUT" --ising-data "$ISING_PATH" \
                    --batch-size "$BATCH_SIZE" \
                    ${OUTPUT_SUFFIX:+--output-suffix "$OUTPUT_SUFFIX"} \
                    --connectivity "$CONNECTIVITY" \
                    $FORCE_FLAG ${SEED_OFFSET:+--seed-offset $SEED_OFFSET} \
                    ${N_REPLICATES_OVERRIDE:+--n-replicates $N_REPLICATES_OVERRIDE})

            echo "Submitted retry job: $RETRY_JOB"
            echo "Waiting for retry job to complete..."

            # Wait for the retry job to complete
            while squeue -j "$RETRY_JOB" 2>/dev/null | grep -q "$RETRY_JOB"; do
                sleep 60
            done

            echo "Retry job $RETRY_JOB completed."

            # Update for next iteration
            CURRENT_JOB="$RETRY_JOB"
            RETRY_COUNT=$((RETRY_COUNT + 1))
        done

        # Final check after all retries
        echo ""
        echo "--- Final check after $MAX_RETRIES retry attempts ---"
        sleep 5
        FAILED_TASKS=$(get_failed_tasks "$CURRENT_JOB")

        if [ -z "$FAILED_TASKS" ]; then
            echo "All tasks completed successfully after retries!"
            exit 0
        else
            FAILED_COUNT=$(echo "$FAILED_TASKS" | tr ',' '\n' | wc -l)
            echo "WARNING: $FAILED_COUNT tasks still failed after $MAX_RETRIES retries: $FAILED_TASKS"
            echo "Manual intervention may be required."
            # Exit with success to allow combine job to proceed (it can handle missing data)
            exit 0
        fi
        ;;

    # =========================================================================
    # FIGURES MODE - Regenerate figures from existing comparison results
    # =========================================================================
    figures)
        # Array task guard: figures loops over both frames, so only task 0 runs
        if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
            echo "Array task $SLURM_ARRAY_TASK_ID: figures handled by task 0, exiting"
            exit 0
        fi

        echo ""
        echo "=============================================="
        echo "Regenerating Figures"
        echo "=============================================="

        # UMAP benefits from thread parallelism
        export OMP_NUM_THREADS=8
        export MKL_NUM_THREADS=8
        export OPENBLAS_NUM_THREADS=8

        # Parse figures-specific arguments
        METRIC="combined"
        CONNECTIVITY="rook"
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --metric)
                    METRIC="$2"
                    shift 2
                    ;;
                --connectivity)
                    CONNECTIVITY="$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done

        set_connectivity_paths

        echo "Metric: $METRIC"

        EXIT_CODE=0
        for FRAME_LABEL in prestim nostim full_trial; do
            RESULTS_FILE="$ISING_DATA/IsingComparison/$METRIC/IsingComparison_Results_${FRAME_LABEL}_subselect_centre_vs_tiled_${METRIC}${CONN_SUFFIX}_optimized.mat"
            echo ""
            echo "--- $FRAME_LABEL ---"
            echo "Results file: $RESULTS_FILE"

            if [ ! -f "$RESULTS_FILE" ]; then
                echo "WARNING: Results file not found: $RESULTS_FILE (skipping)"
                continue
            fi

            srun python Figure5_IsingComparison_optimized.py --figures-only "$RESULTS_FILE" --metric "$METRIC" --connectivity "$CONNECTIVITY"
            RC=$?
            if [ $RC -ne 0 ]; then
                echo "ERROR: Figures generation failed for $FRAME_LABEL (exit code $RC)"
                EXIT_CODE=$RC
            fi
        done

        echo ""
        echo "=============================================="
        echo "Figures generation completed"
        echo "Exit code: $EXIT_CODE"
        echo "End time: $(date)"
        echo "=============================================="

        exit $EXIT_CODE
        ;;

    # =========================================================================
    # SNAPSHOTS MODE - Snapshot figures + heatmap/asymmetry analysis
    # =========================================================================
    #
    # Default behaviour (no quick flags):
    #   Submitter job → launches 6 SLURM jobs (2 parallel pipelines):
    #     1. Snapshot workers (--mode snapshots --index N, array, 1 CPU, 1G, 5min)
    #     2. Snapshot combine (--combine, afterok snap workers, 1 CPU, 1G, 5min)
    #     3. Snapshot figures (--figures-only, afterok snap combine, 1 CPU, 8G, 30min)
    #     4. Heatmap workers  (--mode heatmap --index N, array, 1 CPU, 1G, 30min)
    #     5. Heatmap combine  (--combine, afterok heatmap workers, 1 CPU, 1G, 5min)
    #     6. Heatmap figures  (--heatmap-only, afterok heatmap combine, 1 CPU, 4G, 30min)
    #
    # Quick flags (--scan, --figures-only, --heatmap-only, --combine,
    #   --count-jobs) → pass through directly to Python, no submission.
    #
    # Internal flags (set by the submitter, not by the user):
    #   --internal-mode snapshots  → run Python with --mode snapshots
    #   --internal-mode heatmap    → run Python with --mode heatmap --index $SLURM_ARRAY_TASK_ID
    #
    snapshots)
        echo ""
        echo "=============================================="
        echo "Perturbation Snapshots / Heatmaps"
        echo "=============================================="

        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1

        # Share Numba cache per job (not per task) — Numba uses atomic writes,
        # so concurrent tasks on the same node safely share the cache.
        # First task JIT-compiles (~275s); subsequent tasks read from cache (~0s).
        if [ -n "$SLURM_JOB_ID" ]; then
            export NUMBA_CACHE_DIR="/tmp/numba_snap_${SLURM_JOB_ID}"
        fi

        # Parse arguments
        OUTPUT="$PERTURBATION_OUTPUT"
        ISING_PATH="$ISING_DATA"
        METRIC=""
        FRAME_LABEL="full_trial"
        N_REPS=""              # --n-reps value
        STIM_MODE="clamped"    # stimulus mode: clamped | double_pulse | bias_0p5 | bias_2p0 | all
        INTERNAL_MODE=""       # set by submitter: snapshots | heatmap
        IS_QUICK=false         # true if a quick/pass-through flag was given
        EXTRA_ARGS=""
        SKIP_WORKERS=false     # --skip-workers: omit worker arrays + their afterok deps;
                               # only submit combine + figures (used to retroactively
                               # trigger downstream stages after a backfill).
        # Snapshot worker --time. Was 00:20:00; tasks were timing out due to
        # cold-cache Numba JIT (~275 s) plus matplotlib 300-DPI rendering.
        # 01:00:00 leaves comfortable headroom; override via --snapshot-worker-time.
        SNAP_WORKER_TIME="01:00:00"
        CONNECTIVITY="rook"

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --output)
                    OUTPUT="$2"
                    shift 2
                    ;;
                --ising-data)
                    ISING_PATH="$2"
                    shift 2
                    ;;
                --metric)
                    METRIC="$2"
                    shift 2
                    ;;
                --frame-label)
                    FRAME_LABEL="$2"
                    shift 2
                    ;;
                --internal-mode)
                    INTERNAL_MODE="$2"
                    shift 2
                    ;;
                --n-reps)
                    N_REPS="$2"
                    shift 2
                    ;;
                --stim-mode)
                    STIM_MODE="$2"
                    shift 2
                    ;;
                --snapshot-worker-time)
                    SNAP_WORKER_TIME="$2"
                    shift 2
                    ;;
                --scan|--count-jobs|--figures-only|--heatmap-only|--combine)
                    EXTRA_ARGS="$EXTRA_ARGS $1"
                    IS_QUICK=true
                    shift
                    ;;
                --skip-workers)
                    SKIP_WORKERS=true
                    shift
                    ;;
                --sizes|--durations|--time-scale|--sampling-rate)
                    EXTRA_ARGS="$EXTRA_ARGS $1 $2"
                    shift 2
                    ;;
                --connectivity)
                    CONNECTIVITY="$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done

        set_connectivity_paths

        # Append metric subdirectory when metric is provided and output wasn't overridden
        if [ -n "$METRIC" ] && [ "$OUTPUT" = "$PERTURBATION_OUTPUT" ]; then
            OUTPUT="$OUTPUT/$METRIC"
        fi

        # --- stim-mode=all / dp10_full: fan out to multiple modes in parallel ---
        # 'all'      : every fully-clamped mode + every bias family x every bias
        #              value (5 + 5*6 = 35 total). Mirrors SNAPSHOT_STIM_MODES
        #              in generate_perturbation_snapshots.py (post 6-bias-value
        #              extension; old '0p5'/'2p0' single-decimal names are gone).
        # 'dp10_full': double_pulse10 + 6 double_pulse_bias10_* variants (7 total).
        #              Matches submit_dp10_pipeline.sh's analyze fan-out.
        if [ "$STIM_MODE" = "all" ] || [ "$STIM_MODE" = "dp10_full" ]; then
            echo "Launching fan-out for STIM_MODE=$STIM_MODE..."
            REPS="${N_REPS:-100}"
            # Build args that are common across modes (exclude --stim-mode)
            FAN_ARGS=""
            if [ -n "$METRIC" ]; then FAN_ARGS="$FAN_ARGS --metric $METRIC"; fi
            if [ "$FRAME_LABEL" != "full_trial" ]; then FAN_ARGS="$FAN_ARGS --frame-label $FRAME_LABEL"; fi
            if [ -n "$N_REPS" ]; then FAN_ARGS="$FAN_ARGS --n-reps $N_REPS"; fi
            FAN_ARGS="$FAN_ARGS --snapshot-worker-time $SNAP_WORKER_TIME --connectivity $CONNECTIVITY"

            if [ "$STIM_MODE" = "dp10_full" ]; then
                FAN_MODES=(double_pulse10 \
                           double_pulse_bias10_0p25 double_pulse_bias10_0p50 \
                           double_pulse_bias10_1p00 double_pulse_bias10_2p00 \
                           double_pulse_bias10_0p15 double_pulse_bias10_0p75 \
                           double_pulse_bias10_1p25 double_pulse_bias10_1p50 \
                           double_pulse_bias10_1p75 double_pulse_bias10_4p00 \
                           double_pulse_bias10_6p00 double_pulse_bias10_8p00 \
                           double_pulse_bias10_10p00 double_pulse_bias10_12p00 \
                           double_pulse_bias10_14p00 double_pulse_bias10_16p00)
            else
                # Full enumeration: 5 fully-clamped + (5 families x 9 bias values)
                FAN_MODES=(clamped \
                           double_pulse double_pulse3 double_pulse5 double_pulse10)
                for fam in bias double_pulse_bias double_pulse_bias3 \
                           double_pulse_bias5 double_pulse_bias10; do
                    for bv in 0p15 0p25 0p50 0p75 1p00 1p25 1p50 1p75 2p00 4p00 6p00 8p00 10p00 12p00 14p00 16p00; do
                        FAN_MODES+=("${fam}_${bv}")
                    done
                done
            fi

            for mode in "${FAN_MODES[@]}"; do
                echo "  Submitting --stim-mode $mode"
                bash "$SCRIPT_DIR/run_ising_pipeline.sh" snapshots $FAN_ARGS --stim-mode "$mode" $EXTRA_ARGS &
            done
            wait
            exit 0
        fi

        # Append stim-mode subdirectory
        OUTPUT="$OUTPUT/$STIM_MODE"

        mkdir -p "$OUTPUT"

        # Convert --metric to --comparison path
        COMPARISON_PATH=""
        if [ -n "$METRIC" ]; then
            COMPARISON_PATH="$ISING_PATH/IsingComparison/$METRIC/IsingComparison_Results_${FRAME_LABEL}_subselect_centre_vs_tiled_${METRIC}${CONN_SUFFIX}_optimized.mat"
            echo "Metric: $METRIC"
            echo "Frame label: $FRAME_LABEL"
            echo "Comparison file: $COMPARISON_PATH"
        fi

        echo "Stim mode: $STIM_MODE"
        echo "Output: $OUTPUT"

        # =================================================================
        # Route: quick flags → pass through directly to Python
        # =================================================================
        if [ "$IS_QUICK" = true ]; then
            # --scan: show dry-run info for BOTH pipelines
            SCAN_MODE=""
            for arg in $EXTRA_ARGS; do
                if [ "$arg" = "--scan" ]; then SCAN_MODE="--mode all"; break; fi
            done

            CMD="python \"$SCRIPT_DIR/generate_perturbation_snapshots.py\" --output \"$OUTPUT\" --connectivity \"$CONNECTIVITY\""
            if [ -n "$COMPARISON_PATH" ]; then
                CMD="$CMD --comparison \"$COMPARISON_PATH\""
            elif [ -n "$ISING_PATH" ]; then
                CMD="$CMD --ising-data \"$ISING_PATH\""
            fi
            if [ -n "$SCAN_MODE" ]; then
                CMD="$CMD $SCAN_MODE"
            fi
            if [ -n "$N_REPS" ]; then
                CMD="$CMD --n-reps $N_REPS"
            fi
            CMD="$CMD --stim-mode $STIM_MODE $EXTRA_ARGS"

            echo "Running: $CMD"
            srun /bin/bash -c "$CMD"
            RC=$?
            exit $RC
        fi

        # =================================================================
        # Route: internal-mode → child job launched by the submitter
        # =================================================================
        if [ -n "$INTERNAL_MODE" ]; then
            CMD="python \"$SCRIPT_DIR/generate_perturbation_snapshots.py\" --output \"$OUTPUT\" --connectivity \"$CONNECTIVITY\""
            if [ -n "$COMPARISON_PATH" ]; then
                CMD="$CMD --comparison \"$COMPARISON_PATH\""
            elif [ -n "$ISING_PATH" ]; then
                CMD="$CMD --ising-data \"$ISING_PATH\""
            fi
            CMD="$CMD --mode $INTERNAL_MODE --stim-mode $STIM_MODE"
            if [ -n "$N_REPS" ]; then
                CMD="$CMD --n-reps $N_REPS"
            fi
            # Array worker: pass --index from SLURM array task ID
            if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
                CMD="$CMD --index $SLURM_ARRAY_TASK_ID"
            fi
            CMD="$CMD $EXTRA_ARGS"

            echo "Running ($INTERNAL_MODE): $CMD"
            srun /bin/bash -c "$CMD"
            RC=$?
            exit $RC
        fi

        # =================================================================
        # Route: SUBMITTER — launch both pipelines in parallel
        # =================================================================
        echo ""
        echo "Submitting snapshot + heatmap pipelines in parallel..."

        mkdir -p "$OUTPUT/logs"
        REPS="${N_REPS:-100}"
        PIPELINE_SCRIPT="$SCRIPT_DIR/run_ising_pipeline.sh"

        # Build common args forwarded to child jobs
        CHILD_ARGS="--metric $METRIC --stim-mode $STIM_MODE --connectivity $CONNECTIVITY"
        if [ "$OUTPUT" != "$PERTURBATION_OUTPUT/$METRIC/$STIM_MODE" ]; then
            CHILD_ARGS="$CHILD_ARGS --output $OUTPUT"
        fi
        if [ "$FRAME_LABEL" != "full_trial" ]; then
            CHILD_ARGS="$CHILD_ARGS --frame-label $FRAME_LABEL"
        fi

        # --- Count combos for array range (shared by both pipelines) ---
        N_JOBS=$(python "$SCRIPT_DIR/generate_perturbation_snapshots.py" \
            --output "$OUTPUT" --comparison "$COMPARISON_PATH" \
            --count-jobs $EXTRA_ARGS | tail -1)
        echo "Total combos: $N_JOBS"

        # ====== Snapshot pipeline (workers -> combine -> figures) ======

        # 1. Snapshot worker array (skipped under --skip-workers)
        SNAP_COMBINE_DEP=""
        if [ "$SKIP_WORKERS" = false ]; then
            SNAP_WORKER_JOB=$(sbatch --parsable \
                --partition=defaultp \
                --export=NONE \
                --array=0-$((N_JOBS-1)) \
                --cpus-per-task=1 \
                --mem=1G \
                --time=$SNAP_WORKER_TIME \
                --job-name="snap_${STIM_MODE}_${METRIC}" \
                --output="$OUTPUT/logs/snapshot_%A_%a.out" \
                --error="$OUTPUT/logs/snapshot_%A_%a.err" \
                "$PIPELINE_SCRIPT" snapshots \
                    $CHILD_ARGS --internal-mode snapshots $EXTRA_ARGS)
            SNAP_COMBINE_DEP="--dependency=afterok:$SNAP_WORKER_JOB"
        else
            echo "  --skip-workers: not submitting snapshot worker array."
        fi

        # 2. Snapshot combine (afterok: snapshot workers, unless skipped)
        SNAP_COMBINE_JOB=$(sbatch --parsable \
            $SNAP_COMBINE_DEP \
            --partition=defaultp \
            --export=NONE \
            --cpus-per-task=1 \
            --mem=8G \
            --time=00:15:00 \
            --job-name="snap_agg_${STIM_MODE}_${METRIC}" \
            --output="$OUTPUT/logs/snapshot_combine_%j.out" \
            --error="$OUTPUT/logs/snapshot_combine_%j.err" \
            "$PIPELINE_SCRIPT" snapshots $CHILD_ARGS --combine)

        # 3. Snapshot figures (afterok: snapshot combine)
        SNAP_FIGURES_JOB=$(sbatch --parsable \
            --dependency=afterok:$SNAP_COMBINE_JOB \
            --partition=defaultp \
            --export=NONE \
            --cpus-per-task=1 \
            --mem=8G \
            --time=03:00:00 \
            --job-name="snap_fig_${STIM_MODE}_${METRIC}" \
            --output="$OUTPUT/logs/snapshot_figures_%j.out" \
            --error="$OUTPUT/logs/snapshot_figures_%j.err" \
            "$PIPELINE_SCRIPT" snapshots $CHILD_ARGS --figures-only $EXTRA_ARGS)

        # ====== Heatmap pipeline (workers -> combine -> figures) ======

        # 4. Heatmap worker array
        # Scale wall time with replicate count (base ~30s per rep per combo,
        # increased from 20s to account for FOV-crop blob metrics)
        HEATMAP_SECS=$(( REPS * 30 + 600 ))          # 30s/rep + 10min overhead
        HEATMAP_HH=$(( HEATMAP_SECS / 3600 ))
        HEATMAP_MM=$(( (HEATMAP_SECS % 3600) / 60 ))
        HEATMAP_TIME=$(printf "%02d:%02d:00" $HEATMAP_HH $HEATMAP_MM)
        echo "Heatmap wall time: $HEATMAP_TIME ($REPS reps)"

        HEATMAP_COMBINE_DEP=""
        if [ "$SKIP_WORKERS" = false ]; then
            HEATMAP_WORKER_JOB=$(sbatch --parsable \
                --partition=defaultp \
                --export=NONE \
                --array=0-$((N_JOBS-1)) \
                --cpus-per-task=1 \
                --mem=1G \
                --time=$HEATMAP_TIME \
                --job-name="heatmap_${STIM_MODE}_${METRIC}" \
                --output="$OUTPUT/logs/heatmap_%A_%a.out" \
                --error="$OUTPUT/logs/heatmap_%A_%a.err" \
                "$PIPELINE_SCRIPT" snapshots \
                    $CHILD_ARGS --internal-mode heatmap --n-reps "$REPS" $EXTRA_ARGS)
            HEATMAP_COMBINE_DEP="--dependency=afterok:$HEATMAP_WORKER_JOB"
        else
            echo "  --skip-workers: not submitting heatmap worker array."
        fi

        # 5. Heatmap combine (afterok: heatmap workers, unless skipped)
        HEATMAP_COMBINE_JOB=$(sbatch --parsable \
            $HEATMAP_COMBINE_DEP \
            --partition=defaultp \
            --export=NONE \
            --cpus-per-task=1 \
            --mem=8G \
            --time=00:15:00 \
            --job-name="heatmap_agg_${STIM_MODE}_${METRIC}" \
            --output="$OUTPUT/logs/heatmap_combine_%j.out" \
            --error="$OUTPUT/logs/heatmap_combine_%j.err" \
            "$PIPELINE_SCRIPT" snapshots $CHILD_ARGS --combine)

        # 6. Heatmap figures (afterok: heatmap combine)
        HEATMAP_FIGURES_JOB=$(sbatch --parsable \
            --dependency=afterok:$HEATMAP_COMBINE_JOB \
            --partition=defaultp \
            --export=NONE \
            --cpus-per-task=1 \
            --mem=32G \
            --time=10:00:00 \
            --job-name="heatmap_fig_${STIM_MODE}_${METRIC}" \
            --output="$OUTPUT/logs/heatmap_figures_%j.out" \
            --error="$OUTPUT/logs/heatmap_figures_%j.err" \
            "$PIPELINE_SCRIPT" snapshots $CHILD_ARGS --heatmap-only $EXTRA_ARGS)

        echo ""
        echo "Submitted pipelines ($N_JOBS combos):"
        echo "  Snapshot workers:  $SNAP_WORKER_JOB ($N_JOBS tasks, 1CPU, 1G, $SNAP_WORKER_TIME)"
        echo "  Snapshot combine:  $SNAP_COMBINE_JOB (depends on $SNAP_WORKER_JOB)"
        echo "  Snapshot figures:  $SNAP_FIGURES_JOB (depends on $SNAP_COMBINE_JOB, 60min)"
        echo "  Heatmap workers:   $HEATMAP_WORKER_JOB ($N_JOBS tasks x $REPS reps, 1CPU, 1G, 30min)"
        echo "  Heatmap combine:   $HEATMAP_COMBINE_JOB (depends on $HEATMAP_WORKER_JOB)"
        echo "  Heatmap figures:   $HEATMAP_FIGURES_JOB (depends on $HEATMAP_COMBINE_JOB)"
        echo "Output: $OUTPUT"

        exit 0
        ;;

esac

# -----------------------------------------------------------------------------
# Report completion
# -----------------------------------------------------------------------------
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job completed"
echo "Mode: $MODE"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================="

# Print job statistics if available
if command -v sacct &> /dev/null && [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo ""
    echo "Job statistics:"
    sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,State
fi

exit $EXIT_CODE
