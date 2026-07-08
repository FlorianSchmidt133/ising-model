"""
Plot binarised stim-period snapshots for SmallStimNoSpout recordings.

Loads Grid40 rasterised data for the two SmallStimNoSpout recordings
(Animal26_240906_1340, Animal29_240927_1156), binarises stim-period
averages, and plots example trial snapshots with stimulus rectangle overlay.

Can also be used for any condition in Grid40.mat (e.g. SmallStimExpertIndividual).
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'comparisons'))
from morans_i_optimized import create_weight_matrix_jit, morans_i_single_jit


def load_p1_recording(f, condition, rec_idx_0based):
    """Load a single P1 recording from an HDF5 Grid40.mat file.

    Handles double-referenced (e.g. ExpertIndividual) and single-referenced
    (e.g. SmallStimExpertIndividual) storage layouts.

    Returns
    -------
    data : ndarray, shape (rows, cols, frames, trials)
    """
    p1 = f['Grid40'][condition]['AllNeurons']['P1']
    ref = p1[0, rec_idx_0based]
    inner = f[ref]
    if isinstance(inner, h5py.Dataset):
        raw = np.array(inner)
        if raw.dtype == object or raw.ndim < 3:
            data_ref = inner[0, 0]
            raw = np.array(f[data_ref])
    else:
        data_ref = inner[0, 0]
        raw = np.array(f[data_ref])
    # HDF5 stores MATLAB (rows, cols, frames, trials) as (trials, frames, cols, rows)
    return raw.transpose(3, 2, 1, 0)


def load_rasterised_grid(mat_path):
    """Load grid data from a rasterised .mat file.

    Tries h5py first (v7.3 HDF5), falls back to scipy (v5).
    The struct layout is: <VarName>.AllNeurons(1).All  [rows x cols x frames x trials]
    Falls back to .P1 if .All is empty.

    Returns
    -------
    data : ndarray, shape (rows, cols, frames, trials)
    """
    # Try HDF5 (v7.3) first
    try:
        with h5py.File(mat_path, 'r') as f:
            keys = [k for k in f.keys() if not k.startswith('#')]
            if not keys:
                raise ValueError(f'No variables in {mat_path}')
            var = f[keys[0]]
            all_neurons = var['AllNeurons']
            # Try .All first, then .P1
            for field in ['All', 'P1']:
                if field not in all_neurons:
                    continue
                ref = all_neurons[field][0, 0]
                inner = f[ref]
                if isinstance(inner, h5py.Dataset):
                    raw = np.array(inner, dtype=float)
                else:
                    data_ref = inner[0, 0]
                    raw = np.array(f[data_ref], dtype=float)
                if raw.size == 0:
                    continue
                return raw.transpose(3, 2, 1, 0)
            raise ValueError(f'No grid data in {mat_path}')
    except Exception:
        pass

    # Fall back to scipy (v5 format)
    import scipy.io as sio
    mat = sio.loadmat(mat_path, squeeze_me=False)
    keys = [k for k in mat.keys() if not k.startswith('_')]
    if not keys:
        raise ValueError(f'No data found in {mat_path}')

    struct = mat[keys[0]]
    all_neurons = struct['AllNeurons'][0, 0]
    for field in ['All', 'P1']:
        try:
            arr = all_neurons[field][0, 0]
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                arr = arr.flat[0]
            arr = np.array(arr, dtype=float)
            if arr.size > 0:
                return arr
        except (KeyError, IndexError, ValueError):
            continue
    raise ValueError(f'No grid data in {mat_path}')


def find_rasterised_files(data_dir, target_recs):
    """Find Grid40 rasterised .mat files, grouped by recording.

    Returns dict: {rec_name: [file1, file2, ...]} where multiple files
    are sessions of the same recording.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.mat') or 'Grid40' not in fname:
            continue
        for rec in target_recs:
            if rec in fname:
                grouped[rec].append(os.path.join(data_dir, fname))
                break
    return dict(grouped)


def detect_stim_rect(data, stim_slice, threshold, persistence_thresh=0.3):
    """Auto-detect stimulus rectangle from persistent activity across trials.

    Binarises all trials during stim frames, computes the fraction of
    (frame x trial) observations where each cell is active, then finds
    the bounding box of cells exceeding the persistence threshold.

    Parameters
    ----------
    data : ndarray, shape (rows, cols, frames, trials)
    stim_slice : slice
    threshold : float — binarisation threshold
    persistence_thresh : float — minimum fraction of stim observations active

    Returns
    -------
    rect : tuple (x, y, w, h) in display coordinates (after flipud), or None
    persistence : ndarray (rows, cols) — persistence map (for diagnostics)
    """
    stim_data = data[:, :, stim_slice, :]
    binary = (stim_data > threshold).astype(float)
    persistence = binary.mean(axis=(2, 3))

    mask = persistence >= persistence_thresh
    if not mask.any():
        return None, persistence

    # flipud to match display coordinates (imshow uses flipud)
    mask_flipped = np.flipud(mask)
    rows, cols = np.where(mask_flipped)
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()

    x = c_min - 0.5
    y = r_min - 0.5
    w = c_max - c_min + 1
    h = r_max - r_min + 1

    return (x, y, w, h), persistence


def parse_args():
    p = argparse.ArgumentParser(
        description='Binarised stim-period snapshots for SmallStimNoSpout')
    p.add_argument('--data', default=r'Grid40_September18.mat',
                   help='Path to Grid40.mat (HDF5) for named conditions')
    p.add_argument('--rasterised-dir', default=r'RasterisedData',
                   help='Directory with rasterised Grid40 .mat files')
    p.add_argument('--condition', default='SmallStimNoSpoutIndividual',
                   help='Grid40 condition name (default: SmallStimNoSpoutIndividual)')
    p.add_argument('--target-recs', nargs='+',
                   default=['Animal26_240906_1340', 'Animal29_240927_1156'],
                   help='Recording names for rasterised file lookup')
    p.add_argument('--threshold', type=float, default=2.0,
                   help='Binarisation threshold (default: 2.0)')
    p.add_argument('--stim-frames', type=int, nargs=2, default=[82, 84],
                   help='Stim frame range, MATLAB 1-indexed start end (default: 82 84)')
    p.add_argument('--n-trials', type=int, default=6,
                   help='Number of random trials per recording (default: 6)')
    p.add_argument('--rows-per-fig', type=int, default=3,
                   help='Max rows per figure (default: 3)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for trial selection (default: 42)')
    p.add_argument('--stim-rect', type=float, nargs=4, default=None,
                   metavar=('X', 'Y', 'W', 'H'),
                   help='Manual stimulus rectangle (x y w h in display coords)')
    p.add_argument('--no-rect', action='store_true',
                   help='Disable stimulus rectangle entirely')
    p.add_argument('--persistence-thresh', type=float, default=0.3,
                   help='Persistence threshold for auto-detecting stim rect (default: 0.3)')
    p.add_argument('--output', default=r'Paper',
                   help='Save directory (default: Paper)')
    p.add_argument('--formats', nargs='+', default=['png', 'svg'],
                   help='Output formats (default: png svg)')
    p.add_argument('--no-show', action='store_true',
                   help='Do not call plt.show()')
    return p.parse_args()


def plot_recording_snapshots(data, rec_label, args, weight_mat,
                             stim_rect=None):
    """Generate snapshot figures for one recording.

    Parameters
    ----------
    data : ndarray, shape (rows, cols, frames, trials)
    rec_label : str
    args : argparse.Namespace
    weight_mat : ndarray for Moran's I
    stim_rect : tuple (x, y, w, h) or None — stimulus rectangle in display coords
    """
    stim_slice = slice(args.stim_frames[0] - 1, args.stim_frames[1])
    n_trials_total = data.shape[3]

    rng = np.random.default_rng(args.seed)
    n_sel = min(args.n_trials, n_trials_total)
    sel_trials = sorted(rng.choice(n_trials_total, size=n_sel, replace=False))

    snapshots = []
    for t_idx in sel_trials:
        snap = data[:, :, stim_slice, t_idx].mean(axis=2)
        binary = (snap > args.threshold).astype(float)
        mi = morans_i_single_jit(binary, weight_mat)
        snapshots.append((t_idx + 1, mi, binary))  # +1 for MATLAB indexing
        print(f'  Trial {t_idx + 1}: MI = {mi:.4f}')

    cmap = ListedColormap([[1, 1, 1], [1, 0, 0]])
    n_figures = math.ceil(len(snapshots) / args.rows_per_fig)

    save_dir = None
    if args.output:
        save_dir = os.path.join(args.output, 'Fig. 5 Model')
        os.makedirs(save_dir, exist_ok=True)

    figs = []
    for fig_idx in range(n_figures):
        start = fig_idx * args.rows_per_fig
        end = min(start + args.rows_per_fig, len(snapshots))
        n_rows = end - start

        fig, axes = plt.subplots(n_rows, 1, figsize=(5, 3.5 * n_rows))
        fig.suptitle(
            f'Stim-Period: {rec_label} [{fig_idx + 1}/{n_figures}]',
            fontsize=12, fontweight='bold')

        if n_rows == 1:
            axes = [axes]

        for k in range(n_rows):
            t, mi, binary = snapshots[start + k]
            ax = axes[k]
            ax.imshow(np.flipud(binary), cmap=cmap, vmin=0, vmax=1,
                      aspect='equal', interpolation='nearest')
            if stim_rect is not None:
                ax.add_patch(Rectangle(
                    (stim_rect[0], stim_rect[1]),
                    stim_rect[2], stim_rect[3],
                    linewidth=1.5, edgecolor='black', facecolor='none'))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Trial {t} (MI={mi:.3f})', fontsize=8)

        fig.tight_layout()
        figs.append(fig)

        if save_dir:
            safe_label = rec_label.replace(' ', '_').replace('/', '_')
            base = os.path.join(
                save_dir, f'nospout_snapshots_{safe_label}_{fig_idx + 1}')
            for fmt in args.formats:
                path = f'{base}.{fmt}'
                fig.savefig(path, dpi=300, bbox_inches='tight')
                print(f'Saved: {path}')

    return figs


def resolve_stim_rect(data, args):
    """Determine stimulus rectangle: manual, auto-detected, or disabled."""
    if args.no_rect:
        print('  Stimulus rectangle: disabled')
        return None
    if args.stim_rect is not None:
        print(f'  Stimulus rectangle: manual {tuple(args.stim_rect)}')
        return tuple(args.stim_rect)

    stim_slice = slice(args.stim_frames[0] - 1, args.stim_frames[1])
    rect, persistence = detect_stim_rect(
        data, stim_slice, args.threshold, args.persistence_thresh)
    if rect is not None:
        print(f'  Stimulus rectangle: auto-detected at '
              f'({rect[0]:.1f}, {rect[1]:.1f}, {rect[2]:.0f}, {rect[3]:.0f})')
        peak_row, peak_col = np.unravel_index(
            persistence.argmax(), persistence.shape)
        print(f'  Peak persistence: {persistence.max():.3f} '
              f'at grid cell ({peak_row}, {peak_col})')
    else:
        print(f'  Stimulus rectangle: no persistent region found '
              f'(thresh={args.persistence_thresh})')
    return rect


def main():
    args = parse_args()

    grid_dims = (13, 26)
    weight_mat = create_weight_matrix_jit(grid_dims[0], grid_dims[1], False)

    # Try loading from Grid40.mat (HDF5) first
    loaded_from_hdf5 = False
    if os.path.exists(args.data):
        try:
            with h5py.File(args.data, 'r') as f:
                if 'Grid40' in f and args.condition in f['Grid40']:
                    cond_group = f['Grid40'][args.condition]
                    if 'AllNeurons' in cond_group:
                        p1 = cond_group['AllNeurons']['P1']
                        n_recs = p1.shape[1]
                        print(f'Found {args.condition} in {args.data}: '
                              f'{n_recs} recordings')
                        for r in range(n_recs):
                            print(f'\nRecording {r + 1}/{n_recs}:')
                            data = load_p1_recording(f, args.condition, r)
                            print(f'  Shape: {data.shape}')
                            stim_rect = resolve_stim_rect(data, args)
                            plot_recording_snapshots(
                                data,
                                f'{args.condition} rec {r + 1}',
                                args, weight_mat,
                                stim_rect=stim_rect)
                        loaded_from_hdf5 = True
        except Exception as e:
            print(f'Could not load {args.condition} from {args.data}: {e}')

    # Fall back to rasterised files
    if not loaded_from_hdf5:
        print(f'{args.condition} not found in HDF5, '
              f'loading from rasterised files...')
        grouped = find_rasterised_files(
            args.rasterised_dir, args.target_recs)
        if not grouped:
            print(f'No rasterised Grid40 files found for '
                  f'{args.target_recs} in {args.rasterised_dir}')
            sys.exit(1)

        for i, (rec_name, file_list) in enumerate(grouped.items()):
            print(f'\nRecording {i + 1}/{len(grouped)}: {rec_name} '
                  f'({len(file_list)} session files)')
            try:
                session_data = []
                for fpath in file_list:
                    d = load_rasterised_grid(fpath)
                    print(f'  {os.path.basename(fpath)}: shape={d.shape}')
                    session_data.append(d)
                data = np.concatenate(session_data, axis=3)
                print(f'  Combined: {data.shape}')
                stim_rect = resolve_stim_rect(data, args)
                plot_recording_snapshots(
                    data, f'SmallStimNoSpout {rec_name}',
                    args, weight_mat,
                    stim_rect=stim_rect)
            except Exception as e:
                print(f'  Error: {e}')
                continue

    if not args.no_show:
        plt.show()
    print('\nDone.')


if __name__ == '__main__':
    main()
