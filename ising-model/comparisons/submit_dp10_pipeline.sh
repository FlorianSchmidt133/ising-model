#!/bin/bash
# =============================================================================
# Submit double_pulse10 + double_pulse_bias10 perturbation pipeline
# =============================================================================
# Generates perturbations and figures for ONLY:
#   - double_pulse10                  (clamped variant; 1 analyze job)
#   - double_pulse_bias10 x 6 biases  (0.25, 0.50, 1.00, 2.00, 4.00, 8.00;
#                                      6 analyze jobs, one per bias value)
#
# Pipeline: perturbations -> retry -> combine -> analyze (7 parallel jobs)
# Each step depends on the previous via --dependency=afterok.
#
# Usage (from cluster login node):
#   cd $HOME/git/MouseBrainActivity/Neuron\ Activity\ Analysis/main_scripts/Figure5/comparisons
#   bash submit_dp10_pipeline.sh
#
# Optional flags:
#   --metric <name>      metric subfolder (default: moransI+activity)
#   --frame-label <lbl>  comparison file frame-label (default: full_trial)
#   --n-replicates <n>   replicates per per-combo file (default: 150)
#   --force              regenerate even if per-combo files exist
#   --skip-perturb       skip generation; assume per-combo files exist
#   --analyze-only       skip generation + combine; assume per-mode .mat files exist
#   --snapshots-only         skip everything else; submit 7 snapshot+heatmap
#                             pipelines, one per dp10 stim mode
#   --snapshot-n-reps N      heatmap replicate count for snapshots (default: 100)
#   --snapshot-worker-time T snapshot WORKER --time (default: 01:00:00). Bump if
#                             tasks are timing out on cold-cache nodes (Numba JIT
#                             + matplotlib 300-DPI rendering takes >20 min on
#                             first run per node)
#   --dry-run                print what would be submitted, don't sbatch
#
# Output figures land at:
#   $PERTURBATION_OUTPUT/<metric>/double_pulse10/Analysis/...
#   $PERTURBATION_OUTPUT/<metric>/double_pulse_bias10_<biasLabel>/Analysis/...
#
# To pull figures back to local Paper, see RSYNC_BACK at the bottom.
# =============================================================================

set -e

SCRIPT_DIR="$HOME/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"
ISING_DATA="$HOME/IsingSims"
PERTURBATION_OUTPUT="$HOME/IsingPerturbations"
PIPELINE_SCRIPT="$SCRIPT_DIR/run_ising_pipeline.sh"

# Defaults
METRIC="moransI+activity"
FRAME_LABEL="full_trial"
N_REPLICATES=150
FORCE=""
SKIP_PERTURB=false
ANALYZE_ONLY=false
SNAPSHOTS_ONLY=false
SNAPSHOT_N_REPS=100
SNAPSHOT_WORKER_TIME="01:00:00"
DRY_RUN=false
MODES_FILTER="double_pulse10,double_pulse_bias10"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --metric)               METRIC="$2"; shift 2 ;;
        --frame-label)          FRAME_LABEL="$2"; shift 2 ;;
        --n-replicates)         N_REPLICATES="$2"; shift 2 ;;
        --force)                FORCE="--force"; shift ;;
        --skip-perturb)         SKIP_PERTURB=true; shift ;;
        --analyze-only)         SKIP_PERTURB=true; ANALYZE_ONLY=true; shift ;;
        --snapshots-only)       SNAPSHOTS_ONLY=true; shift ;;
        --snapshot-n-reps)      SNAPSHOT_N_REPS="$2"; shift 2 ;;
        --snapshot-worker-time) SNAPSHOT_WORKER_TIME="$2"; shift 2 ;;
        --dry-run)              DRY_RUN=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

PERTURB_DIR="$PERTURBATION_OUTPUT/$METRIC"
COMPARISON_FILE="$ISING_DATA/IsingComparison/$METRIC/IsingComparison_Results_${FRAME_LABEL}_subselect_centre_vs_tiled_${METRIC}_optimized.mat"

mkdir -p "$PERTURB_DIR/logs"

echo "=============================================="
echo "dp10 + dp_bias10 pipeline"
echo "=============================================="
echo "Modes filter:  $MODES_FILTER"
echo "Metric:        $METRIC"
echo "Frame label:   $FRAME_LABEL"
echo "Replicates:    $N_REPLICATES"
echo "Force:         ${FORCE:-no}"
echo "Skip perturb:  $SKIP_PERTURB"
echo "Analyze only:  $ANALYZE_ONLY"
echo "Snapshots only:$SNAPSHOTS_ONLY"
echo "Output dir:    $PERTURB_DIR"
echo "Comparison:    $COMPARISON_FILE"
echo "=============================================="

# -----------------------------------------------------------------------------
# Snapshots-only short-circuit
# -----------------------------------------------------------------------------
# Independent of the perturb/combine/analyze chain. Submits 7 separate
# snapshot+heatmap pipeline chains (each chain is itself 6 sbatch jobs:
# snapshot workers + combine + figures, plus heatmap workers + combine +
# figures), one per dp10 stim mode. Reads the per-combo perturb_*.mat files
# already on disk in $PERTURB_DIR — no aggregate needed.
if [ "$SNAPSHOTS_ONLY" = true ]; then
    SNAPSHOT_MODES=(double_pulse10 \
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
    echo ""
    echo "Submitting snapshot+heatmap pipelines for ${#SNAPSHOT_MODES[@]} modes..."
    for mode in "${SNAPSHOT_MODES[@]}"; do
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY-RUN] bash $PIPELINE_SCRIPT snapshots --metric $METRIC --frame-label $FRAME_LABEL --stim-mode $mode --n-reps $SNAPSHOT_N_REPS --snapshot-worker-time $SNAPSHOT_WORKER_TIME"
        else
            echo "  $mode:"
            bash "$PIPELINE_SCRIPT" snapshots \
                --metric "$METRIC" --frame-label "$FRAME_LABEL" \
                --stim-mode "$mode" --n-reps "$SNAPSHOT_N_REPS" \
                --snapshot-worker-time "$SNAPSHOT_WORKER_TIME"
        fi
    done
    echo ""
    echo "=============================================="
    echo "All snapshot+heatmap chains submitted."
    echo "Each mode's chain: snapshot workers -> combine -> figures + heatmap workers -> combine -> figures."
    echo "Outputs land at $PERTURB_DIR/<stim_mode>/{snapshots,heatmap}/"
    echo "=============================================="
    exit 0
fi

# Login-node Python: scientific stack relies on python/3.11 module — system
# python carries an incompatible numpy/scipy combo. Load before any host-side
# python call (--count-jobs below).
if [ "$DRY_RUN" = false ] && [ "$SKIP_PERTURB" = false ]; then
    if ! type module &>/dev/null; then
        source /usr/share/lmod/lmod/init/bash 2>/dev/null || \
        source /etc/profile.d/lmod.sh 2>/dev/null || \
        source /etc/profile.d/modules.sh 2>/dev/null || true
    fi
    # Try a sequence of python module versions; some on the cluster have a broken
    # modulefile that fails to load in non-interactive bash. Stop at the first
    # version whose import check passes.
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
        echo "ERROR: no python module on login node yields a usable numba/numpy/scipy stack." >&2
        exit 1
    fi
    echo "Loaded $PYTHON_LOADED"
fi

submit() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] sbatch $*"
        echo "DRYRUN_$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

DEP_FLAG=""
PERTURB_JOB=""
COMBINE_JOB=""

# -----------------------------------------------------------------------------
# Stage 1: perturbations array
# -----------------------------------------------------------------------------
if [ "$SKIP_PERTURB" = false ]; then
    BATCH_SIZE=10
    if [ "$DRY_RUN" = true ]; then
        TOTAL_JOBS=10000  # placeholder
    else
        # --count-jobs prints informational lines to stderr and only the
        # integer count to stdout; tail -1 gets that integer. We capture
        # stderr too so a python crash surfaces in this script's log.
        COUNT_OUT=$(python "$SCRIPT_DIR/run_ising_perturbations.py" \
            --comparison "$COMPARISON_FILE" --ising-data "$ISING_DATA" \
            --modes "$MODES_FILTER" --count-jobs 2>&1)
        TOTAL_JOBS=$(echo "$COUNT_OUT" | tail -1)
    fi
    if ! [[ "$TOTAL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --count-jobs did not return a positive integer." >&2
        echo "Got: '$TOTAL_JOBS'" >&2
        echo "Full output:" >&2
        echo "$COUNT_OUT" >&2
        exit 1
    fi
    ARRAY_MAX=$(( (TOTAL_JOBS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    echo "Stage 1: perturbations array ($TOTAL_JOBS jobs / $BATCH_SIZE batch = $((ARRAY_MAX+1)) tasks)"

    # run_ising_pipeline.sh perturbations doesn't accept --modes directly,
    # so we pipe through extra args via the script's pass-through mechanism.
    # We invoke run_ising_perturbations.py directly inside our own wrapper.
    # Delegate to run_ising_pipeline.sh perturbations (proper #!/bin/bash, lmod init).
    PERTURB_JOB=$(submit \
        --partition=defaultp --export=NONE \
        --array=0-$ARRAY_MAX \
        --cpus-per-task=1 --mem=2G --time=04:00:00 \
        --job-name="dp10_perturb_$METRIC" \
        --output="$PERTURB_DIR/logs/dp10_perturb_%A_%a.out" \
        --error="$PERTURB_DIR/logs/dp10_perturb_%A_%a.err" \
        "$PIPELINE_SCRIPT" perturbations \
            --metric "$METRIC" --frame-label "$FRAME_LABEL" \
            --output "$PERTURB_DIR" --ising-data "$ISING_DATA" \
            --batch-size $BATCH_SIZE \
            --n-replicates $N_REPLICATES \
            --modes "$MODES_FILTER" \
            $FORCE)
    echo "  Perturbation job: $PERTURB_JOB"
    DEP_FLAG="--dependency=afterok:$PERTURB_JOB"
fi

# -----------------------------------------------------------------------------
# Stage 2: combine (one job per mode in the filter)
# -----------------------------------------------------------------------------
if [ "$ANALYZE_ONLY" = false ]; then
    COMBINE_IDS=()
    IFS=',' read -ra MODE_LIST <<< "$MODES_FILTER"
    for m in "${MODE_LIST[@]}"; do
        m_trimmed=$(echo "$m" | xargs)  # strip whitespace
        CID=$(submit \
            $DEP_FLAG \
            --partition=defaultp --export=NONE \
            --cpus-per-task=1 --mem=8G --time=01:30:00 \
            --job-name="dp10_combine_${m_trimmed}_$METRIC" \
            --output="$PERTURB_DIR/logs/dp10_combine_${m_trimmed}_%j.out" \
            --error="$PERTURB_DIR/logs/dp10_combine_${m_trimmed}_%j.err" \
            "$PIPELINE_SCRIPT" combine \
                --mode "$m_trimmed" \
                --metric "$METRIC" --frame-label "$FRAME_LABEL" \
                --output "$PERTURB_DIR" --ising-data "$ISING_DATA" \
                --n-replicates "$N_REPLICATES")
        echo "  Combine $m_trimmed: $CID"
        COMBINE_IDS+=("$CID")
    done
    DEP_LIST=$(IFS=:; echo "${COMBINE_IDS[*]}")
    DEP_FLAG="--dependency=afterok:$DEP_LIST"
fi

# -----------------------------------------------------------------------------
# Stage 3: analyze — 7 INDEPENDENT sbatch jobs (one per mode/bias variant).
# Submitting as separate sbatch jobs (not a single-job srun fan-out) avoids
# srun step-creation contention inside a single allocation. Each job gets
# its own 4h slot and they run in genuine parallel as the scheduler permits.
# -----------------------------------------------------------------------------
ANALYZE_MODES=(double_pulse10 \
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
ANALYZE_IDS=()
for mode in "${ANALYZE_MODES[@]}"; do
    AID=$(submit \
        $DEP_FLAG \
        --partition=defaultp --constraint=matlab --export=NONE \
        --cpus-per-task=4 --mem=96G --time=04:00:00 \
        --job-name="dp10_analyze_${mode}" \
        --output="$PERTURB_DIR/logs/dp10_analyze_${mode}_%j.out" \
        --error="$PERTURB_DIR/logs/dp10_analyze_${mode}_%j.err" \
        "$PIPELINE_SCRIPT" analyze \
            --metric "$METRIC" --stim-mode "$mode" \
            --results "$PERTURB_DIR/PerturbationResults_*.mat")
    echo "  Analyze $mode: $AID"
    ANALYZE_IDS+=("$AID")
done

echo ""
echo "=============================================="
echo "Submitted dp10 pipeline."
echo "=============================================="
echo "Outputs land under:"
echo "  $PERTURB_DIR/double_pulse10/Analysis/"
echo "  $PERTURB_DIR/double_pulse_bias10_<biasLabel>/Analysis/"
echo ""
echo "Tail logs:"
echo "  tail -f $PERTURB_DIR/logs/dp10_*.err"
echo ""
echo "Pull figures back to local Paper after analyze finishes:"
cat <<RSYNC_BACK
  rsync -avz --include='*/' --include='*.png' --include='*.pdf' --exclude='*' \\
      user@cluster:'$PERTURB_DIR/double_pulse10/Analysis/' \\
      'Paper/Fig. 5 Model/PerturbationAnalysis/double_pulse10/'
  for b in 0p25 0p50 1p00 2p00 4p00 8p00; do
      rsync -avz --include='*/' --include='*.png' --include='*.pdf' --exclude='*' \\
          user@cluster:"$PERTURB_DIR/double_pulse_bias10_\$b/Analysis/" \\
          "Paper/Fig. 5 Model/PerturbationAnalysis/double_pulse_bias10_\$b/"
  done
RSYNC_BACK
