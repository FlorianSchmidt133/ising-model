#!/bin/bash
# =============================================================================
# fig6f_pull_snapshots.sh
#
# Read each mode's matcher BiasMatch_Summary.csv, look up Expert's matched
# bias per window (full/onset/offset; bias6 hardcoded to 6.00), then copy
# all FOV-crop snapshot variants from the bias's PerturbationSnapshots dir
# into BiasMatched/<mode>/Fig6f/size_<N>/window_<wn>/.
#
# Halfcrop variants (incl. snapshot_expert_nospout_examples_*) are SKIPPED —
# Fig6f panel uses FOV crop.
#
# Per (mode, size, window) destination folder, the script copies:
#   from <PERTURB>/double_pulse_bias10_<bv>/PerturbationSnapshots/dur_59/Snapshots/:
#     snapshot_size_<S>_*.png            -> all_conditions.png
#     snapshot_detail_size_<S>_*.png     -> detail.png
#     snapshot_expert_examples_size_<S>_*.png -> expert_examples.png
#   from <PERTURB>/double_pulse_bias10_<bv>/PerturbationSnapshots/dur_59/
#        Single_Replicates/FOVcrop/size_<S>/:
#     snapshot_expert_rep<R>_fovcrop_*.png  -> expert_rep<R>.png   (all R)
#     snapshot_nospout_rep<R>_fovcrop_*.png -> nospout_rep<R>.png  (all R)
#
# Existing PNGs in the dest dir are wiped before each pull so the layout
# reflects the current matcher pick exactly.
#
# Usage:
#   bash fig6f_pull_snapshots.sh <MATCHING_MODE>   # one of: perCondition | global | expert
# =============================================================================

set -u

PERTURB_DIR="$HOME/IsingPerturbations/moransI+activity"
MATCHING_MODE="${1:?usage: $0 <MATCHING_MODE>}"
MATCHER_DIR="$PERTURB_DIR/BiasMatched/${MATCHING_MODE}/Matcher"
OUTPUT_DIR="$PERTURB_DIR/BiasMatched/${MATCHING_MODE}/Fig6f"

mkdir -p "$OUTPUT_DIR"
CSV_OUT="$OUTPUT_DIR/Fig6f_PullSummary.csv"
echo "mode,size,window,expert_bias,n_main,n_singles,status" > "$CSV_OUT"

bias_label() {
    printf "%.2f" "$1" | tr '.' 'p'
}

SIZES=(2 3 4)
WINDOWS=(full onset offset bias6)

n_dirs_ok=0
n_dirs_partial=0
n_dirs_skipped=0

for sz in "${SIZES[@]}"; do
    csv="$MATCHER_DIR/size_${sz}/BiasMatch_Summary.csv"
    if [ ! -f "$csv" ]; then
        echo "WARN: matcher CSV missing: $csv" >&2
        for wn in "${WINDOWS[@]}"; do
            echo "$MATCHING_MODE,$sz,$wn,,0,0,matcher_csv_missing" >> "$CSV_OUT"
            n_dirs_skipped=$((n_dirs_skipped + 1))
        done
        continue
    fi

    for wn in "${WINDOWS[@]}"; do
        if [ "$wn" = "bias6" ]; then
            bv="6.00"
        else
            bv=$(awk -F, -v w="$wn" '$1=="Expert" && $5==w && $7==1 {print $4; exit}' "$csv")
        fi

        dst_dir="$OUTPUT_DIR/size_${sz}/window_${wn}"
        mkdir -p "$dst_dir"

        # Wipe stale PNGs (but keep README/CSV which we will overwrite below)
        find "$dst_dir" -maxdepth 1 -name '*.png' -delete 2>/dev/null

        if [ -z "$bv" ]; then
            echo "WARN: no Expert IsBest=1 row for size=$sz window=$wn" >&2
            echo "$MATCHING_MODE,$sz,$wn,,0,0,no_pick" >> "$CSV_OUT"
            n_dirs_skipped=$((n_dirs_skipped + 1))
            continue
        fi

        if [ "$bv" = "Inf" ] || [ "$bv" = "inf" ]; then
            echo "WARN: clamped (Inf) pick for size=$sz window=$wn — skipping" >&2
            echo "$MATCHING_MODE,$sz,$wn,Inf,0,0,clamped_unsupported" >> "$CSV_OUT"
            n_dirs_skipped=$((n_dirs_skipped + 1))
            continue
        fi

        bvlabel=$(bias_label "$bv")
        src_root="$PERTURB_DIR/double_pulse_bias10_${bvlabel}/PerturbationSnapshots/dur_59"
        src_main="$src_root/Snapshots"
        src_fov="$src_root/Single_Replicates/FOVcrop/size_${sz}"

        # --- Main FOV-crop figures ----------------------------------------
        n_main=0
        if [ -f "$src_main/snapshot_size_${sz}_double_pulse_bias10_${bvlabel}.png" ]; then
            cp "$src_main/snapshot_size_${sz}_double_pulse_bias10_${bvlabel}.png" \
               "$dst_dir/all_conditions.png"
            n_main=$((n_main + 1))
        fi
        if [ -f "$src_main/snapshot_detail_size_${sz}_double_pulse_bias10_${bvlabel}.png" ]; then
            cp "$src_main/snapshot_detail_size_${sz}_double_pulse_bias10_${bvlabel}.png" \
               "$dst_dir/detail.png"
            n_main=$((n_main + 1))
        fi
        if [ -f "$src_main/snapshot_expert_examples_size_${sz}_double_pulse_bias10_${bvlabel}.png" ]; then
            cp "$src_main/snapshot_expert_examples_size_${sz}_double_pulse_bias10_${bvlabel}.png" \
               "$dst_dir/expert_examples.png"
            n_main=$((n_main + 1))
        fi

        # --- FOV-crop single replicates -----------------------------------
        n_singles=0
        if [ -d "$src_fov" ]; then
            for cond in expert nospout; do
                # Glob over rep numbers — naming guarantees rep<R> where R is integer.
                for f in "$src_fov"/snapshot_${cond}_rep*_fovcrop_size_${sz}_double_pulse_bias10_${bvlabel}.png; do
                    [ -f "$f" ] || continue
                    base=$(basename "$f")
                    # extract rep<R> token
                    rep_tok=$(echo "$base" | sed -E 's/^snapshot_(expert|nospout)_(rep[0-9]+)_fovcrop_.*/\2/')
                    cp "$f" "$dst_dir/${cond}_${rep_tok}.png"
                    n_singles=$((n_singles + 1))
                done
            done
        fi

        # --- Status -------------------------------------------------------
        if [ "$n_main" -eq 3 ] && [ "$n_singles" -gt 0 ]; then
            status="ok"; n_dirs_ok=$((n_dirs_ok + 1))
        elif [ "$n_main" -gt 0 ] || [ "$n_singles" -gt 0 ]; then
            status="partial"; n_dirs_partial=$((n_dirs_partial + 1))
        else
            status="missing_source"; n_dirs_skipped=$((n_dirs_skipped + 1))
        fi

        cat > "$dst_dir/README.txt" <<EOF
Mode:         $MATCHING_MODE
Stim size:    $sz
Window:       $wn
Expert bias:  $bv  (label: $bvlabel)
Source dir:   $src_root
Files copied (FOV crop only):
  $n_main main figures (all_conditions.png, detail.png, expert_examples.png)
  $n_singles single-rep timelines (expert_rep<R>.png, nospout_rep<R>.png)
Status:       $status

Halfcrop variants (snapshot_expert_nospout_examples_*) are intentionally
skipped — the Fig6f panel uses the FOV crop. Find Halfcrop sources at
$src_root/Single_Replicates/Halfcrop/size_$sz/ if needed.
EOF

        echo "$MATCHING_MODE,$sz,$wn,$bv,$n_main,$n_singles,$status" >> "$CSV_OUT"
        echo "  $MATCHING_MODE size=$sz $wn -> bias=$bv ($n_main main + $n_singles singles, $status)"
    done
done

echo ""
echo "Fig6f pull complete for mode=$MATCHING_MODE: $n_dirs_ok ok, $n_dirs_partial partial, $n_dirs_skipped skipped/missing"
echo "Summary CSV: $CSV_OUT"
