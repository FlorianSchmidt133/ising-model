"""
Plot Expert vs NoSpout binarised stim-period snapshots.

Python equivalent of plot_expert_vs_nospout_snapshots.m
Loads Grid40.mat (HDF5 v7.3), binarises stim-period averages, computes
Moran's I, and plots a 3x2 comparison figure.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

# Import Moran's I from the existing optimized module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'comparisons'))
from morans_i_optimized import create_weight_matrix_jit, morans_i_single_jit


def load_p1_recording(f, condition, rec_idx_0based):
    """Load a single P1 recording from the HDF5 file.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle for Grid40.mat.
    condition : str
        e.g. 'ExpertIndividual' or 'NoSpoutIndividual'.
    rec_idx_0based : int
        0-based recording index.

    Returns
    -------
    data : ndarray, shape (rows, cols, frames, trials)
        Transposed from HDF5 storage order.
    """
    p1 = f['Grid40'][condition]['AllNeurons']['P1']
    ref = p1[0, rec_idx_0based]
    inner = f[ref]
    data_ref = inner[0, 0]
    # HDF5 stores MATLAB (rows, cols, frames, trials) as (trials, frames, cols, rows)
    raw = np.array(f[data_ref])
    return raw.transpose(3, 2, 1, 0)  # -> (rows, cols, frames, trials)


def parse_args():
    p = argparse.ArgumentParser(
        description='Expert vs NoSpout binarised stim-period snapshots')
    p.add_argument('--data', default=r'Grid40.mat',
                   help='Path to Grid40.mat')
    p.add_argument('--threshold', type=float, default=2.0,
                   help='Binarisation threshold (default: 2.0)')
    p.add_argument('--stim-frames', type=int, nargs=2, default=[82, 84],
                   help='Stim frame range, MATLAB 1-indexed start end (default: 82 84)')
    p.add_argument('--ex-sel', type=int, default=5,
                   help='Expert recording index, 1-indexed (default: 5)')
    p.add_argument('--expert-trials', type=int, nargs='+', default=[1, 2, 3],
                   help='Expert trial numbers, 1-indexed (default: 1 2 3)')
    p.add_argument('--no-sel', type=int, nargs='+', default=[5, 6],
                   help='NoSpout recording indices, 1-indexed (default: 5 6)')
    p.add_argument('--nospout-trial-range', type=int, nargs=2, default=[11, 30],
                   help='NoSpout trial range, 1-indexed start end (default: 11 30)')
    p.add_argument('--nospout-trials', type=int, nargs='+', default=[12, 24, 18],
                   help='NoSpout trial numbers, 1-indexed (default: 12 24 18)')
    p.add_argument('--output', default=r'Paper',
                   help='Save directory (default: Paper)')
    p.add_argument('--formats', nargs='+', default=['png', 'svg'],
                   help='Output formats (default: png svg)')
    return p.parse_args()


def main():
    args = parse_args()

    grid_dims = (13, 26)
    # MATLAB 1-indexed -> Python 0-indexed slice
    stim_slice = slice(args.stim_frames[0] - 1, args.stim_frames[1])

    # Pre-compute weight matrix for Moran's I
    weight_mat = create_weight_matrix_jit(grid_dims[0], grid_dims[1], False)

    with h5py.File(args.data, 'r') as f:
        # --- Expert snapshots -------------------------------------------------
        expert_data = load_p1_recording(
            f, 'ExpertIndividual', args.ex_sel - 1)
        # expert_data: (rows, cols, frames, trials)

        expert_snapshots = []
        for t in args.expert_trials:
            snap = expert_data[:, :, stim_slice, t - 1].mean(axis=2)
            binary = (snap > args.threshold).astype(float)
            mi = morans_i_single_jit(binary, weight_mat)
            print(f'Expert trial {t}: MI = {mi:.4f}')
            expert_snapshots.append(binary)

        # --- NoSpout snapshots ------------------------------------------------
        tr_start = args.nospout_trial_range[0] - 1  # 0-indexed
        tr_end = args.nospout_trial_range[1]         # exclusive upper bound

        nospout_parts = []
        for idx in args.no_sel:
            rec = load_p1_recording(f, 'NoSpoutIndividual', idx - 1)
            nospout_parts.append(rec[:, :, :, tr_start:tr_end])
        nospout_data = np.concatenate(nospout_parts, axis=3)

        nospout_snapshots = []
        for t in args.nospout_trials:
            snap = nospout_data[:, :, stim_slice, t - 1].mean(axis=2)
            binary = (snap > args.threshold).astype(float)
            mi = morans_i_single_jit(binary, weight_mat)
            print(f'NoSpout trial {t}: MI = {mi:.4f}')
            nospout_snapshots.append(binary)

    # --- Plot (3 rows x 2 columns) -------------------------------------------
    n_rows = max(len(expert_snapshots), len(nospout_snapshots))
    cmap = ListedColormap([[1, 1, 1], [1, 0, 0]])  # white, red

    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(10, 3.5 * n_rows))
    fig.suptitle('Stim-Period: Expert vs NoSpout',
                 fontsize=12, fontweight='bold')

    # Ensure axes is 2-D even for a single row
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for k, snap in enumerate(expert_snapshots):
        ax = axes[k, 0]
        flipped = np.flipud(snap)
        flipped = np.roll(flipped, 2, axis=1)
        flipped[:, :2] = 0  # clear wrapped columns
        ax.imshow(flipped, cmap=cmap, vmin=0, vmax=1,
                  aspect='equal', interpolation='nearest')
        ax.add_patch(Rectangle((18.5, 6.5), 4, 3, linewidth=1.5,
                               edgecolor='black', facecolor='none'))
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            ax.set_title('Expert', fontsize=7)
            ax.set_ylabel('Expert', fontsize=8, fontweight='bold')

    for k, snap in enumerate(nospout_snapshots):
        ax = axes[k, 1]
        flipped = np.flipud(snap)
        if k == 1:
            flipped = np.roll(flipped, -1, axis=1)
            flipped[:, -1] = 0  # clear wrapped column
        ax.imshow(flipped, cmap=cmap, vmin=0, vmax=1,
                  aspect='equal', interpolation='nearest')
        ax.add_patch(Rectangle((18.5, 6.5), 4, 3, linewidth=1.5,
                               edgecolor='black', facecolor='none'))
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            ax.set_title('NoSpout', fontsize=7)
            ax.set_ylabel('NoSpout', fontsize=8, fontweight='bold')

    # Hide any unused subplot slots
    for k in range(len(expert_snapshots), n_rows):
        axes[k, 0].set_visible(False)
    for k in range(len(nospout_snapshots), n_rows):
        axes[k, 1].set_visible(False)

    fig.tight_layout()

    # --- Save -----------------------------------------------------------------
    if args.output:
        save_dir = os.path.join(args.output, 'Fig. 5 Model')
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, 'expert_vs_nospout_stim_comparison')
        for fmt in args.formats:
            path = f'{base}.{fmt}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f'Saved: {path}')

    plt.show()
    print('\nDone.')


if __name__ == '__main__':
    main()
