#!/bin/bash
# =============================================================================
# Submit the full bias-matched pipeline for all 3 MatchingModes in parallel.
#
# Modes:
#   perCondition  — each condition's own best bias (current default)
#   global        — single bias minimising sum-RMSE across non-Naive conds
#   expert        — Expert's best bias propagated to every condition
#
# Per mode this submits:
#   3 matcher jobs (one per stim size 2/3/4)
#   3 figure jobs (Fig6c, Fig6g, Fig6h) with afterok deps on the matcher trio
#
# Total: 3 modes × (3 matchers + 3 figures) = 18 jobs.
# Wall time ~25-30 min (matcher dominates; figures fan out after).
#
# Usage (from cluster login node):
#   bash submit_biasmatched_all_modes.sh
# =============================================================================

set -e

SCRIPT_DIR="$HOME/git/MouseBrainActivity/Neuron Activity Analysis/main_scripts/Figure5/comparisons"
PERTURB_DIR="$HOME/IsingPerturbations/moransI+activity"

# Propagate BaselineAlign toggle to all matcher children so all chains
# use a consistent setting. Default true (baseline-aligned scoring is
# the new default). Override via: BASELINE_ALIGN=false bash submit_...
export BASELINE_ALIGN="${BASELINE_ALIGN:-true}"
echo "BaselineAlign = $BASELINE_ALIGN"

MODES=(perCondition global expert)
SIZES=(2 3 4)

echo "=============================================="
echo "Submitting bias-matched pipeline for ${#MODES[@]} modes:"
printf "  - %s\n" "${MODES[@]}"
echo "=============================================="

for mode in "${MODES[@]}"; do
    echo ""
    echo "=== Mode: $mode ==="

    # 1. Matcher: 3 sizes in parallel for this mode. Third positional arg
    #    is BASELINE_ALIGN so all matchers in the chain share one toggle.
    MATCHER_IDS=()
    for sz in "${SIZES[@]}"; do
        MID=$(sbatch --parsable \
            "$SCRIPT_DIR/submit_match_dp10_bias.sh" "$sz" "$mode" "$BASELINE_ALIGN")
        echo "  matcher size=$sz mode=$mode -> $MID"
        MATCHER_IDS+=("$MID")
    done
    DEP_LIST=$(IFS=:; echo "${MATCHER_IDS[*]}")
    DEP_FLAG="--dependency=afterok:$DEP_LIST"

    # 2. Fig6c (uses all 3 sizes' matcher outputs, so depends on all 3)
    FID6C=$(sbatch --parsable $DEP_FLAG \
        "$SCRIPT_DIR/submit_fig6c_panels.sh" "$mode")
    echo "  fig6c       mode=$mode -> $FID6C  (afterok:$DEP_LIST)"

    # 3. Fig6g (same dep — uses all 3 sizes)
    FID6G=$(sbatch --parsable $DEP_FLAG \
        "$SCRIPT_DIR/submit_fig6g_panels.sh" "$mode")
    echo "  fig6g       mode=$mode -> $FID6G  (afterok:$DEP_LIST)"

    # 4. Fig6h (size=2/window=full default, depends on size_2 matcher only —
    #    but for simplicity we keep the same all-3-sizes dep so it fires once
    #    the whole matcher batch is done)
    FID6H=$(sbatch --parsable $DEP_FLAG \
        "$SCRIPT_DIR/submit_fig6h_panels.sh" 2 full "$mode")
    echo "  fig6h full  mode=$mode -> $FID6H  (afterok:$DEP_LIST)"

    # 4b. Fig6h for the stim window (RMSE on [0, 2.5s] — entire stim duration
    #     plus immediate post-offset peak, no baseline or tail contamination).
    FID6H_STIM=$(sbatch --parsable $DEP_FLAG \
        "$SCRIPT_DIR/submit_fig6h_panels.sh" 2 stim "$mode")
    echo "  fig6h stim  mode=$mode -> $FID6H_STIM  (afterok:$DEP_LIST)"

    # 4c. Fig6h for BothPeaks (union of onset + offset windows — RMSE only
    #     in the two peak regions, ignoring the during-stim plateau).
    FID6H_BP=$(sbatch --parsable $DEP_FLAG \
        "$SCRIPT_DIR/submit_fig6h_panels.sh" 2 BothPeaks "$mode")
    echo "  fig6h BothPeaks   mode=$mode -> $FID6H_BP   (afterok:$DEP_LIST)"

    # 4d. Fig6h for StimPeriod ([0, 2s] — exact stim duration only, no
    #     post-offset extension).
    FID6H_SP=$(sbatch --parsable $DEP_FLAG \
        "$SCRIPT_DIR/submit_fig6h_panels.sh" 2 StimPeriod "$mode")
    echo "  fig6h StimPeriod  mode=$mode -> $FID6H_SP   (afterok:$DEP_LIST)"

    # 4e-4j. Naive-anchored RMSE window variants. For each rated window,
    #        bias is anchored on Naive's per-window argmin and propagated
    #        to all conds.
    for nv_win in full_naive onset_naive offset_naive stim_naive BothPeaks_naive StimPeriod_naive; do
        FID6H_NV=$(sbatch --parsable $DEP_FLAG \
            "$SCRIPT_DIR/submit_fig6h_panels.sh" 2 "$nv_win" "$mode")
        echo "  fig6h $nv_win  mode=$mode -> $FID6H_NV   (afterok:$DEP_LIST)"
    done

    # 5. Fig6h for the bias6 window (fixed bias=6.0 per the matcher's bias6
    #    post-process). Reuses the same composite cache as the full window.
    FID6H_BIAS6=$(sbatch --parsable $DEP_FLAG \
        "$SCRIPT_DIR/submit_fig6h_panels.sh" 2 bias6 "$mode")
    echo "  fig6h bias6 mode=$mode -> $FID6H_BIAS6  (afterok:$DEP_LIST)"

    # 5b. Fig6h fold_global: shared bias minimising sum |dataFold-modelFold|
    #     over non-Naive conds; per-cond sim picked at that bias.
    FID6H_FG=$(sbatch --parsable $DEP_FLAG \
        "$SCRIPT_DIR/submit_fig6h_panels.sh" 2 fold_global "$mode")
    echo "  fig6h fold_global  mode=$mode -> $FID6H_FG  (afterok:$DEP_LIST)"

    # 5c. Fig6h fold_nospout: shared bias = NoSpout's joint argmin |fold diff|;
    #     per-cond sim picked at that bias. NoSpout is the un-rewarded control.
    FID6H_FN=$(sbatch --parsable $DEP_FLAG \
        "$SCRIPT_DIR/submit_fig6h_panels.sh" 2 fold_nospout "$mode")
    echo "  fig6h fold_nospout mode=$mode -> $FID6H_FN  (afterok:$DEP_LIST)"

    # 5d. Fig6h fold_naive: shared bias = Naive's joint argmin |fold diff|;
    #     per-cond sim picked at that bias. Naive is the pre-training control.
    FID6H_FNV=$(sbatch --parsable $DEP_FLAG \
        "$SCRIPT_DIR/submit_fig6h_panels.sh" 2 fold_naive "$mode")
    echo "  fig6h fold_naive   mode=$mode -> $FID6H_FNV  (afterok:$DEP_LIST)"

    # 6. Fig6f: pull pre-generated snapshots for each (size, window) using
    #    Expert's matched bias. Pure copy operation — no MATLAB.
    FID6F=$(sbatch --parsable $DEP_FLAG \
        "$SCRIPT_DIR/submit_fig6f_panels.sh" "$mode")
    echo "  fig6f       mode=$mode -> $FID6F  (afterok:$DEP_LIST)"
done

echo ""
echo "=============================================="
echo "All 3 mode chains submitted."
echo ""
echo "Outputs land at:"
echo "  $PERTURB_DIR/BiasMatched/<mode>/Matcher/size_<N>/"
echo "  $PERTURB_DIR/BiasMatched/<mode>/Fig6c/size_<N>/window_<wn>/"
echo "  $PERTURB_DIR/BiasMatched/<mode>/Fig6g/size_<N>/window_<wn>/"
echo "  $PERTURB_DIR/BiasMatched/<mode>/Fig6h/size_<N>/window_<wn>/double_pulse_bias10_2p00/"
echo "  $PERTURB_DIR/BiasMatched/<mode>/Fig6f/size_<N>/window_<wn>/snapshot_*.png"
echo "=============================================="
