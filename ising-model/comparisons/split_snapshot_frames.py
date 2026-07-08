"""Split per-replicate snapshot timelines into per-frame PNGs.

Each composite (e.g. ``expert_rep1.png``) is a 1x14 matplotlib row of binary
grid panels (1 Pre + 10 stim-percentage frames + 3 Post). This script crops
each composite into 14 individual PNGs placed in a per-replicate subfolder.

Default input directory is the size_3/window_full bias-matched set used by
Fig 6 panel b; pass ``--input-dir`` to point at any other composite folder
with the same layout.
"""
import argparse
import base64
import io
import os
import sys

import numpy as np
from PIL import Image

DEFAULT_INPUT_DIR = (
    r"Paper\Fig. 6 Ising Model\BiasMatched\perCondition\Fig6f"
    r"\size_3\window_full"
)

# Hard-coded label list for dur=59 size=3 (the BiasMatched output that
# currently lives in DEFAULT_INPUT_DIR). 14 frames: 1 Pre + 10 stim + 3 Post.
FRAME_LABELS_DUR59 = [
    "Pre",
    "10pct", "20pct", "31pct", "41pct", "51pct",
    "59pct", "69pct", "80pct", "90pct", "98pct",
    "Post_p89", "Post_p159", "Post_p259",
]

# Files to skip (composite of composites)
SKIP_NAMES = {"detail.png", "expert_examples.png", "all_conditions.png"}

# Pixel value at or below this counts as "dark" when locating panel edges.
DARK_THRESHOLD = 50

# Vertical band (fraction of image height) at the very top of the figure that
# holds the suptitle. Scanning that band would link every panel together
# because the title spans the whole width.
SUPTITLE_FRAC = 0.07

# Minimum width of a panel run (px). Filters out stray dark dots / artifacts.
MIN_PANEL_WIDTH = 80

# Horizontal padding (px) added to each detected panel before cropping.
PANEL_PAD = 8


def detect_panel_runs(gray: np.ndarray):
    """Return list of (x_start, x_end) panel intervals, including the
    leftmost row-label block first.

    Strategy: skip the suptitle band, then mark columns that contain any
    near-black pixel. Group contiguous dark columns into runs and discard
    runs narrower than MIN_PANEL_WIDTH (kills stray text artifacts).
    """
    h, w = gray.shape
    y0 = int(h * SUPTITLE_FRAC)
    has_dark = (gray[y0:, :] < DARK_THRESHOLD).any(axis=0)

    runs = []
    in_run = False
    start = 0
    for c in range(w):
        if has_dark[c]:
            if not in_run:
                in_run = True
                start = c
        else:
            if in_run:
                runs.append((start, c - 1))
                in_run = False
    if in_run:
        runs.append((start, w - 1))

    runs = [r for r in runs if (r[1] - r[0] + 1) >= MIN_PANEL_WIDTH]
    return runs


def equal_width_panels(width: int, left_margin_frac: float = 0.06, n_panels: int = 14):
    """Fallback: divide the strip after the left margin into n equal panels."""
    left = int(width * left_margin_frac)
    panel_w = (width - left) / n_panels
    runs = []
    for i in range(n_panels):
        x0 = int(round(left + i * panel_w))
        x1 = int(round(left + (i + 1) * panel_w)) - 1
        runs.append((x0, x1))
    return runs


def detect_axis_box(panel_gray: np.ndarray):
    """Return (y_top, y_bottom) of the panel axis spine within a panel crop.

    The axis spine top and bottom are horizontal black lines spanning most of
    the panel width. Rows in the title area have only a few dark pixels (text
    characters) and never span the full width. We pick the top/bottom rows
    where dark-pixel coverage exceeds a high threshold.
    """
    h, w = panel_gray.shape
    dark_per_row = (panel_gray < DARK_THRESHOLD).sum(axis=1)
    span_threshold = int(w * 0.7)  # axis spine spans almost the full width
    spine_rows = np.where(dark_per_row >= span_threshold)[0]
    if len(spine_rows) == 0:
        return None
    return int(spine_rows[0]), int(spine_rows[-1])


def png_to_svg_string(panel_img: Image.Image) -> str:
    """Embed a PIL image as a minimal SVG (image element + viewBox)."""
    buf = io.BytesIO()
    panel_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    w, h = panel_img.size
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        f'  <image x="0" y="0" width="{w}" height="{h}" '
        f'xlink:href="data:image/png;base64,{b64}"/>\n'
        f'</svg>\n'
    )


def split_composite(path: str, labels, write_png: bool, write_svg: bool):
    """Crop one composite into a sibling subfolder, one file per panel.

    PNG output keeps the column title above the grid for reference. SVG
    output is cropped tight to the axis spine so there is no title text.
    """
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)
    out_dir = os.path.join(os.path.dirname(path), stem)
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(path).convert("RGB")
    gray = np.asarray(img.convert("L"))
    w, h = img.size

    runs = detect_panel_runs(gray)
    n_expected = len(labels)
    # First run is the row label ("Expert 1"); after that we want n_expected panels.
    panel_runs = runs[1:] if len(runs) == n_expected + 1 else runs

    used_fallback = False
    if len(panel_runs) != n_expected:
        print(
            f"  WARN {name}: detected {len(panel_runs)} panel runs "
            f"(expected {n_expected}); falling back to equal-width slicing"
        )
        panel_runs = equal_width_panels(w, n_panels=n_expected)
        used_fallback = True

    svg_fallbacks = 0
    for i, (c0, c1) in enumerate(panel_runs):
        x0 = max(0, c0 - PANEL_PAD)
        x1 = min(w, c1 + 1 + PANEL_PAD)
        crop = img.crop((x0, 0, x1, h))

        if write_png:
            fname = f"frame_{i + 1:02d}_{labels[i]}.png"
            crop.save(os.path.join(out_dir, fname))

        if write_svg:
            # Crop further to remove the column title above the axis spine.
            crop_gray = np.asarray(crop.convert("L"))
            box = detect_axis_box(crop_gray)
            if box is None:
                svg_fallbacks += 1
                # Fall back to keeping the bottom 70 % of the panel crop.
                top = int(h * 0.30)
                bottom = h
            else:
                top, bottom = box
                # Include the spine itself (1-2 px) at both ends.
                top = max(0, top)
                bottom = min(crop.size[1], bottom + 1)
            tight = crop.crop((0, top, crop.size[0], bottom))
            svg_name = f"frame_{i + 1:02d}_{labels[i]}.svg"
            with open(os.path.join(out_dir, svg_name), "w", encoding="utf-8") as f:
                f.write(png_to_svg_string(tight))

    return len(panel_runs), used_fallback, svg_fallbacks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    p.add_argument(
        "--pattern",
        default="rep",
        help="Substring required in the filename stem (default: 'rep')",
    )
    p.add_argument(
        "--no-png",
        action="store_true",
        help="Skip the PNG output (which keeps the column title)",
    )
    p.add_argument(
        "--no-svg",
        action="store_true",
        help="Skip the SVG output (which strips the column title)",
    )
    args = p.parse_args()
    write_png = not args.no_png
    write_svg = not args.no_svg

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: input dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    labels = FRAME_LABELS_DUR59
    print(f"Input: {args.input_dir}")
    print(f"Labels ({len(labels)}): {labels}")

    files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(".png")
        and f not in SKIP_NAMES
        and args.pattern in f
    )
    print(f"Found {len(files)} composite PNGs")

    n_ok = 0
    n_fallback = 0
    svg_fb_total = 0
    for fname in files:
        path = os.path.join(args.input_dir, fname)
        n_panels, used_fb, svg_fbs = split_composite(
            path, labels, write_png=write_png, write_svg=write_svg
        )
        if used_fb:
            n_fallback += 1
        else:
            n_ok += 1
        svg_fb_total += svg_fbs
        extra = []
        if used_fb:
            extra.append("equal-width fallback")
        if svg_fbs:
            extra.append(f"{svg_fbs} svg axis-fallback")
        suffix = f" ({'; '.join(extra)})" if extra else ""
        print(f"  {fname}: wrote {n_panels} frames{suffix}")

    print(
        f"\nDone. {n_ok} composites split via panel detection, "
        f"{n_fallback} via equal-width fallback. "
        f"SVG axis-detection fallbacks: {svg_fb_total}."
    )


if __name__ == "__main__":
    main()
