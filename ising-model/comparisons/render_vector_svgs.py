"""Reconstruct vector SVG panels from rendered Ising snapshot PNGs.

The composite PNGs in
``Paper\\Fig. 6 Ising Model\\BiasMatched\\perCondition\\Fig6f\\size_3\\window_full``
are raster renderings of a 13x26 FOV-cropped Ising grid. Each panel shows a
binary state (black = active, white = inactive) plus a dashed red 3x3
stimulus rectangle at the grid centre.

This script:
1. Detects each panel's axis-spine bounding box.
2. Samples cell centres on the 13x26 grid to recover the binary state.
3. Writes a true vector SVG with `<rect>` shapes per active cell, plus the
   outer axis box and the dashed red stimulus rect.

Output overwrites the raster-embedded ``frame_<NN>_<label>.svg`` files
created by ``split_snapshot_frames.py``.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

# --- Grid geometry (must match run_ising_perturbations.py) -----------------
L, M = 39, 78
CROP_ROW_START, CROP_ROW_END = 13, 26
CROP_COL_START, CROP_COL_END = 26, 52
GRID_ROWS = CROP_ROW_END - CROP_ROW_START  # 13
GRID_COLS = CROP_COL_END - CROP_COL_START  # 26

# Stimulus region for size=3 (odd) at grid centre (row=L//2=19, col=M//2=39)
STIM_SIZE = 3
STIM_CENTER_ROW = L // 2  # 19
STIM_CENTER_COL = M // 2  # 39
STIM_HALF = STIM_SIZE // 2  # 1
STIM_R0 = STIM_CENTER_ROW - STIM_HALF - CROP_ROW_START  # 5
STIM_R1 = STIM_CENTER_ROW + STIM_HALF - CROP_ROW_START  # 7 (inclusive)
STIM_C0 = STIM_CENTER_COL - STIM_HALF - CROP_COL_START  # 12
STIM_C1 = STIM_CENTER_COL + STIM_HALF - CROP_COL_START  # 14 (inclusive)

# --- I/O defaults ---------------------------------------------------------
DEFAULT_INPUT_DIR = (
    r"Paper\Fig. 6 Ising Model\BiasMatched\perCondition\Fig6f"
    r"\size_3\window_full"
)
SKIP_NAMES = {"detail.png", "expert_examples.png", "all_conditions.png"}

# --- Pixel thresholds -----------------------------------------------------
DARK_THRESHOLD = 60     # grayscale value <= this counts as a dark cell
RED_MIN_R = 150         # pixel is "red" if R high and G/B low
RED_MAX_GB = 100

# --- SVG output geometry --------------------------------------------------
CELL_SIZE = 20          # SVG units per cell
SPINE_WIDTH = 1.0       # axis spine stroke width in SVG units
STIM_STROKE = 1.5
STIM_DASH = "3,2"


def is_red_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of red-channel-dominant pixels."""
    return (
        (rgb[..., 0] >= RED_MIN_R)
        & (rgb[..., 1] <= RED_MAX_GB)
        & (rgb[..., 2] <= RED_MAX_GB)
    )


def detect_axis_box(gray: np.ndarray):
    """Find (top, bottom, left, right) pixel indices of the panel axis spine.

    Two-pass: find the top/bottom horizontal spine rows (dark coverage spans
    most of the image width), then within that vertical range locate the
    left/right spine columns (dark coverage spans most of the panel height).
    Locating left/right against the panel height — not the image height —
    is essential because the title strip extends well above the panel.
    """
    h, w = gray.shape
    is_dark = gray < DARK_THRESHOLD

    # Spine rows have ~94 % dark coverage; title text rows top out around
    # 55 %. Use 0.85 to discriminate solid spine from any title content.
    row_span = is_dark.sum(axis=1)
    row_thresh = int(w * 0.85)
    row_hits = np.where(row_span >= row_thresh)[0]
    if len(row_hits) < 2:
        return None
    top = int(row_hits[0])
    bottom = int(row_hits[-1])
    if bottom - top < 10:
        return None

    panel_height = bottom - top + 1
    sub = is_dark[top:bottom + 1, :]
    col_span = sub.sum(axis=0)
    col_thresh = max(5, int(panel_height * 0.7))
    col_hits = np.where(col_span >= col_thresh)[0]
    if len(col_hits) < 2:
        return None
    left = int(col_hits[0])
    right = int(col_hits[-1])
    return top, bottom, left, right


def sample_cell(rgb: np.ndarray, cy: float, cx: float, half: int = 2) -> bool:
    """Return True if the cell at (cy, cx) is active (black).

    Samples a (2*half+1) square around the cell centre, masks out any red
    pixels (dashed stim rect), and returns True if the mean greyscale of
    the remaining pixels is below DARK_THRESHOLD.
    """
    h, w, _ = rgb.shape
    y0 = max(0, int(cy) - half)
    y1 = min(h, int(cy) + half + 1)
    x0 = max(0, int(cx) - half)
    x1 = min(w, int(cx) + half + 1)
    patch = rgb[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    red_mask = is_red_mask(patch)
    if red_mask.all():
        # Cell centre completely inside dashed rect line; sample wider corners
        return False
    gray_patch = patch.mean(axis=2)
    valid = gray_patch[~red_mask]
    if valid.size == 0:
        return False
    return float(valid.mean()) < DARK_THRESHOLD


def reconstruct_grid(rgb: np.ndarray, box):
    """Sample the GRID_ROWS x GRID_COLS binary grid from inside `box`."""
    top, bottom, left, right = box
    interior_top = top + 1
    interior_bottom = bottom - 1
    interior_left = left + 1
    interior_right = right - 1
    height = interior_bottom - interior_top + 1
    width = interior_right - interior_left + 1
    cell_h = height / GRID_ROWS
    cell_w = width / GRID_COLS

    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    for r in range(GRID_ROWS):
        cy = interior_top + (r + 0.5) * cell_h
        for c in range(GRID_COLS):
            cx = interior_left + (c + 0.5) * cell_w
            grid[r, c] = sample_cell(rgb, cy, cx)
    return grid


def grid_to_svg(grid: np.ndarray) -> str:
    """Build a true-vector SVG from a GRID_ROWS x GRID_COLS binary grid."""
    w = GRID_COLS * CELL_SIZE
    h = GRID_ROWS * CELL_SIZE
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'shape-rendering="crispEdges">',
        '  <g id="cells" fill="black">',
    ]
    for r in range(GRID_ROWS):
        y = r * CELL_SIZE
        for c in range(GRID_COLS):
            if grid[r, c]:
                x = c * CELL_SIZE
                parts.append(
                    f'    <rect x="{x}" y="{y}" '
                    f'width="{CELL_SIZE}" height="{CELL_SIZE}"/>'
                )
    parts.append('  </g>')
    # Outer axis spine
    parts.append(
        f'  <rect x="0" y="0" width="{w}" height="{h}" '
        f'fill="none" stroke="black" stroke-width="{SPINE_WIDTH}"/>'
    )
    # Dashed red stimulus rect (3x3 at grid centre)
    stim_x = STIM_C0 * CELL_SIZE
    stim_y = STIM_R0 * CELL_SIZE
    stim_w = (STIM_C1 - STIM_C0 + 1) * CELL_SIZE
    stim_h = (STIM_R1 - STIM_R0 + 1) * CELL_SIZE
    parts.append(
        f'  <rect x="{stim_x}" y="{stim_y}" '
        f'width="{stim_w}" height="{stim_h}" '
        f'fill="none" stroke="red" stroke-width="{STIM_STROKE}" '
        f'stroke-dasharray="{STIM_DASH}"/>'
    )
    parts.append('</svg>\n')
    return "\n".join(parts)


def process_frame_png(path: str):
    """Render one per-frame PNG (already title-stripped) as a vector SVG.

    Returns 'ok', 'no-axis', or 'no-png' status.
    """
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img)
    gray = np.asarray(img.convert("L"))
    box = detect_axis_box(gray)
    if box is None:
        return "no-axis", None
    grid = reconstruct_grid(rgb, box)
    svg = grid_to_svg(grid)
    out_path = os.path.splitext(path)[0] + ".svg"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)
    return "ok", int(grid.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: input dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Input: {args.input_dir}")
    print(f"Grid:  {GRID_ROWS}x{GRID_COLS}, stim rows {STIM_R0}-{STIM_R1}, "
          f"cols {STIM_C0}-{STIM_C1}")

    # Walk per-replicate subfolders (created by split_snapshot_frames.py)
    rep_dirs = sorted(
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
        and ("_rep" in d)
    )
    print(f"Found {len(rep_dirs)} replicate subfolders")

    n_ok = 0
    n_fail = 0
    fails = []
    for rd in rep_dirs:
        sub = os.path.join(args.input_dir, rd)
        pngs = sorted(
            f for f in os.listdir(sub)
            if f.lower().endswith(".png") and f.startswith("frame_")
        )
        for fname in pngs:
            path = os.path.join(sub, fname)
            status, n_cells = process_frame_png(path)
            if status == "ok":
                n_ok += 1
            else:
                n_fail += 1
                fails.append(f"{rd}/{fname}: {status}")
        print(f"  {rd}: wrote {len(pngs)} vector SVGs")

    print(f"\nDone. {n_ok} vector SVGs written, {n_fail} failed.")
    if fails:
        print("Failures:")
        for line in fails[:20]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
