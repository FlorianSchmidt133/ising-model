"""
Plot Hit vs Miss binarised stim-period snapshots for SmallStimExpert.

Loads Grid40.mat (HDF5 v7.3) for grid data, binarises stim-period averages,
computes Moran's I, and plots comparison figures (Hit | Miss) using
user-specified trial numbers.
"""

import argparse
import math
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

    Handles both double-referenced (ExpertIndividual) and single-referenced
    (SmallStimExpertIndividual) storage layouts.

    Returns
    -------
    data : ndarray, shape (rows, cols, frames, trials)
    """
    p1 = f['Grid40'][condition]['AllNeurons']['P1']
    ref = p1[0, rec_idx_0based]
    inner = f[ref]
    if isinstance(inner, h5py.Dataset):
        raw = np.array(inner)
    else:
        data_ref = inner[0, 0]
        raw = np.array(f[data_ref])
    # HDF5 stores MATLAB (rows, cols, frames, trials) as (trials, frames, cols, rows)
    return raw.transpose(3, 2, 1, 0)  # -> (rows, cols, frames, trials)


def parse_args():
    p = argparse.ArgumentParser(
        description='Hit vs Miss binarised stim-period snapshots (SmallStimExpert)')
    p.add_argument('--data', default=r'Grid40.mat',
                   help='Path to Grid40.mat')
    p.add_argument('--threshold', type=float, default=2.0,
                   help='Binarisation threshold (default: 2.0)')
    p.add_argument('--stim-frames', type=int, nargs=2, default=[82, 84],
                   help='Stim frame range, MATLAB 1-indexed start end (default: 82 84)')
    p.add_argument('--rec', type=int, default=1,
                   help='Recording index, 1-indexed (default: 1)')
    p.add_argument('--hit-trials', type=int, nargs='+',
                   default=[40, 47, 54, 42, 29, 32],
                   help='Hit trial numbers, 1-indexed')
    p.add_argument('--miss-trials', type=int, nargs='+',
                   default=[19, 11, 46, 52, 43, 56],
                   help='Miss trial numbers, 1-indexed')
    p.add_argument('--rows-per-fig', type=int, default=3,
                   help='Max rows per figure (default: 3)')
    p.add_argument('--output', default=r'Paper',
                   help='Save directory (default: Paper)')
    p.add_argument('--formats', nargs='+', default=['png', 'svg'],
                   help='Output formats (default: png svg)')
    return p.parse_args()


def main():
    args = parse_args()

    grid_dims = (13, 26)
    stim_slice = slice(args.stim_frames[0] - 1, args.stim_frames[1])

    # Pre-compute weight matrix for Moran's I
    weight_mat = create_weight_matrix_jit(grid_dims[0], grid_dims[1], False)

    # --- Load grid data ---
    with h5py.File(args.data, 'r') as f:
        data = load_p1_recording(
            f, 'SmallStimExpertIndividual', args.rec - 1)
    print(f'Grid data shape: {data.shape}')

    # --- Binarise specified trials ---
    def binarise_trial(t):
        snap = data[:, :, stim_slice, t - 1].mean(axis=2)
        binary = (snap > args.threshold).astype(float)
        mi = morans_i_single_jit(binary, weight_mat)
        return (t, mi, binary)

    hits = [binarise_trial(t) for t in args.hit_trials]
    misses = [binarise_trial(t) for t in args.miss_trials]

    print('Hit trials:')
    for t, mi, _ in hits:
        print(f'  Trial {t}: MI = {mi:.4f}')
    print('Miss trials:')
    for t, mi, _ in misses:
        print(f'  Trial {t}: MI = {mi:.4f}')

    # --- Per-trial spatial shifts (col_shift, row_shift) ---
    # Positive col_shift = move data right; positive row_shift = move data up
    # (row_shift is applied to flipped array, so positive = roll upward = negative axis-0 roll)
    trial_shifts = {
        29: (1, 0),    # one column right
        56: (-1, 1),   # one column left, one row up
        47: (-1, 1),   # one column left, one row up
        11: (0, 1),    # one row up
    }

    def apply_shift(arr, trial_num):
        if trial_num not in trial_shifts:
            return arr
        col_shift, row_shift = trial_shifts[trial_num]
        shifted = arr.copy()
        if col_shift != 0:
            shifted = np.roll(shifted, col_shift, axis=1)
            if col_shift > 0:
                shifted[:, :col_shift] = 0
            else:
                shifted[:, col_shift:] = 0
        if row_shift != 0:
            shifted = np.roll(shifted, -row_shift, axis=0)
            if row_shift > 0:
                shifted[-row_shift:, :] = 0
            else:
                shifted[:-row_shift, :] = 0
        return shifted

    # --- Split into figures ---
    n_figures = max(math.ceil(len(hits) / args.rows_per_fig),
                    math.ceil(len(misses) / args.rows_per_fig))
    cmap = ListedColormap([[1, 1, 1], [1, 0, 0]])  # white, red

    save_dir = None
    if args.output:
        save_dir = os.path.join(args.output, 'Fig. 5 Model')
        os.makedirs(save_dir, exist_ok=True)

    for fig_idx in range(n_figures):
        h_start = fig_idx * args.rows_per_fig
        h_end = min(h_start + args.rows_per_fig, len(hits))
        m_start = fig_idx * args.rows_per_fig
        m_end = min(m_start + args.rows_per_fig, len(misses))
        n_rows = max(h_end - h_start, m_end - m_start)

        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.5 * n_rows))
        fig.suptitle(f'Stim-Period: Hit vs Miss (SmallStimExpert) [{fig_idx + 1}/{n_figures}]',
                     fontsize=12, fontweight='bold')

        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for k in range(n_rows):
            # --- Hit panel ---
            if h_start + k < h_end:
                t, mi, binary = hits[h_start + k]
                ax = axes[k, 0]
                flipped = apply_shift(np.flipud(binary), t)
                ax.imshow(flipped, cmap=cmap, vmin=0, vmax=1,
                          aspect='equal', interpolation='nearest')
                ax.add_patch(Rectangle((5.5, 3.5), 3, 2, linewidth=1.5,
                                       edgecolor='black', facecolor='none'))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel('')
                if k == 0:
                    ax.set_title('Hit', fontsize=7)
            else:
                axes[k, 0].set_visible(False)

            # --- Miss panel ---
            if m_start + k < m_end:
                t, mi, binary = misses[m_start + k]
                ax = axes[k, 1]
                flipped = apply_shift(np.flipud(binary), t)
                ax.imshow(flipped, cmap=cmap, vmin=0, vmax=1,
                          aspect='equal', interpolation='nearest')
                ax.add_patch(Rectangle((5.5, 3.5), 3, 2, linewidth=1.5,
                                       edgecolor='black', facecolor='none'))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel('')
                if k == 0:
                    ax.set_title('Miss', fontsize=7)
            else:
                axes[k, 1].set_visible(False)

        fig.tight_layout()

        # --- Save ---
        if save_dir:
            base = os.path.join(save_dir, f'hit_vs_miss_stim_comparison_{fig_idx + 1}')
            for fmt in args.formats:
                path = f'{base}.{fmt}'
                fig.savefig(path, dpi=300, bbox_inches='tight')
                print(f'Saved: {path}')

    plt.show()
    print('\nDone.')


if __name__ == '__main__':
    main()
