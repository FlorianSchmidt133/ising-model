#!/usr/bin/env python3
"""
demo_ising.py: minimal, self-contained demo of the Ising model.

Runs a single small Ising simulation using the *actual* model code
(`ising-model/generate_all_ising_simulations.py::monte_carlo`), then plots a few
spin snapshots and the fraction-active timecourse. No experimental data needed.

Run:
    python demos/demo_ising.py
Output: a PNG written under the configured figure root (default: results/).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- make the repo importable (repo root + ising-model/) ---
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "ising-model"))

import config  # noqa: E402  central data/output locations
from generate_all_ising_simulations import monte_carlo  # noqa: E402  the real model


def main():
    # Expert-condition best-match parameters (top-1 of the per-condition
    # best-match sets in statistics/Figure5_cluster_stats/EDFigure10_top10_params.json),
    # on the paper's 39x78 model grid (3x the 13x26 experimental grid).
    # Note: Figure 6c additionally applies a double-pulse bias=1.75 perturbation
    # during the stimulus; here we run the unperturbed Expert base model.
    L, M = 39, 78          # model grid (paper)
    beta = 0.56            # inverse temperature
    c = 4                  # coupling (inhibition) strength
    decay_const = 4        # H-field decay constant
    rad = 13               # inhibition kernel radius
    bias_val = -0.8        # bias term
    n_steps = 2000         # recording steps (demo-sized; the full sweeps use 1e5)

    print(f"Running Ising simulation ({L}x{M}, beta={beta}, c={c}, "
          f"decay={decay_const}, rad={rad}, bias={bias_val}, {n_steps} steps)...")
    print("(first run compiles the Numba kernels - give it a few seconds)")
    store_spins, wait_time = monte_carlo(
        L, M, beta, c, decay_const, rad, bias_val, n_steps)
    print(f"Done. burn-in used: {wait_time} steps; "
          f"recorded array shape: {store_spins.shape}")

    # Fraction of "active" (+1) sites per recorded frame
    frac_active = np.mean(store_spins == 1, axis=(1, 2)) * 100.0

    # --- figure: snapshots + fraction-active timecourse ---
    snap_idx = np.linspace(0, n_steps - 1, 4).astype(int)
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.2)
    for k, t in enumerate(snap_idx):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(store_spins[t], cmap="gray", vmin=-1, vmax=1,
                  interpolation="nearest", aspect="equal")
        ax.set_title(f"frame {t}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :])
    ax.plot(frac_active, color=(0.7, 0, 0), lw=1.5)
    ax.set_xlabel("recorded frame"); ax.set_ylabel("% sites active")
    ax.set_title(f"Fraction active (mean {frac_active.mean():.1f}%)")
    ax.grid(True, alpha=0.3)
    fig.suptitle("Ising model demo", fontsize=13)

    out_dir = str(config.FIGURE_ROOT)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_ising.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
