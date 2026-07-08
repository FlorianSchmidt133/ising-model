"""
Port of plot_experimental_snapshots.m to Python.

Generates neural activity snapshot figures at Moran's I percentiles.
For each condition, produces a 2-row figure:
  - Top row: continuous dF/F heatmaps at selected percentile frames
  - Bottom row: binarised activity at the same frames
Saves PNG (300 dpi) and SVG outputs.

Usage:
    python "Neuron Activity Analysis/main_scripts/Figure5/plots/plot_experimental_snapshots.py"
"""

import sys
import os
import numpy as np
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

# Add parent of 'comparisons' to path so we can import morans_i_optimized
_script_dir = Path(__file__).resolve().parent
_figure5_dir = _script_dir.parent
sys.path.insert(0, str(_figure5_dir / 'comparisons'))

from morans_i_optimized import (
    create_weight_matrix_jit,
    morans_i_batch_nonan,
    morans_i_single_jit,
    morans_i_tiled_batch_separate,
    generate_tiled_positions,
)


# ── Configuration ────────────────────────────────────────────────────────────

GRID_DIMENSIONS = (13, 26)  # (gridY, gridX)

# Skip lists per condition (0-indexed, converted from MATLAB 1-indexed)
SKIP = {
    'Naive':   {0, 8, 9, 15},
    'Beginner': {0, 5, 6, 10},
    'Expert':  {0, 3, 11, 12, 13},
    'NoSpout': {0, 3, 8, 9, 10, 12, 13},
}

GRID40_PATH = r'Grid40.mat'
COLORMAP_PATH = os.path.join(
    _figure5_dir.parent.parent,  # "Neuron Activity Analysis/"
    'EntropyColourMap.mat'
)

# ── Ising comparison constants ──────────────────────────────────────────────
ISING_GRID = (39, 78)
DEFAULT_ISING_DATA_PATH = r'IsingModelData_39x78_100K'
DEFAULT_ISING_GRID_MODE = 'subselect_tiled'
DEFAULT_ISING_METRIC = 'spatial+persistence'
DEFAULT_EX_SEL = 4          # 0-indexed (MATLAB ExSel=5)
DEFAULT_STIM_FRAMES = (81, 82, 83)  # 0-indexed (MATLAB 82:84)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_entropy_colormap(path=COLORMAP_PATH):
    """Load EntropyColourMap.mat (v5) and return a matplotlib ListedColormap."""
    mat = sio.loadmat(path)
    rgb = mat['EntropyColourmap']  # shape (64, 3), float values in [0, 1]
    return ListedColormap(rgb, name='entropy')


def load_grid40_p1(f, condition, skip_set):
    """Load and pool P1 data from HDF5 Grid40 for one condition.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle for Grid40.mat.
    condition : str
        e.g. 'Naive', 'Expert'.
    skip_set : set of int
        0-indexed recording indices to skip.

    Returns
    -------
    pooled : ndarray, shape (gridY, gridX, nTime, nTrials_total)
        Concatenated P1 data across included recordings, or None if empty.
    """
    cond_key = f'{condition}Individual'
    grid40 = f['Grid40']

    if cond_key not in grid40:
        print(f'  Field {cond_key} not found in Grid40 — skipping.')
        return None

    # Navigate to P1 dataset: Grid40.<cond>Individual.AllNeurons.P1
    all_neurons = grid40[cond_key]['AllNeurons']
    p1_dataset = all_neurons['P1']
    n_recs = p1_dataset.shape[1]

    chunks = []
    for r in range(n_recs):
        if r in skip_set:
            print(f'  Rec {r + 1}: SKIPPED')
            continue

        # Double-dereference: p1_dataset[0, r] is an HDF5 object reference
        ref = p1_dataset[0, r]
        deref1 = f[ref]

        # Second dereference: the cell contents
        inner_ref = deref1[0, 0]
        data = f[inner_ref][()]  # read the actual numeric array

        if data.ndim != 4 or data.size == 0:
            print(f'  Rec {r + 1}: Empty or invalid data (shape={data.shape})')
            continue

        # HDF5 stores MATLAB (gridY, gridX, nTime, nTrials) transposed as
        # (nTrials, nTime, gridX, gridY).  Transpose back.
        data = data.transpose(3, 2, 1, 0)  # → (gridY, gridX, nTime, nTrials)

        n_trials = data.shape[3]
        chunks.append(data)
        print(f'  Rec {r + 1}: {n_trials} trials pooled')

    if not chunks:
        return None

    return np.concatenate(chunks, axis=3)


def load_grid40_p1_single(f, condition, rec_idx):
    """Load a single recording's P1 data from HDF5 Grid40.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle for Grid40.mat.
    condition : str
        e.g. 'Expert'.
    rec_idx : int
        0-indexed recording index.

    Returns
    -------
    data : ndarray, shape (gridY, gridX, nTime, nTrials) or None
    """
    cond_key = f'{condition}Individual'
    grid40 = f['Grid40']

    if cond_key not in grid40:
        print(f'  Field {cond_key} not found in Grid40.')
        return None

    all_neurons = grid40[cond_key]['AllNeurons']
    p1_dataset = all_neurons['P1']
    n_recs = p1_dataset.shape[1]

    if rec_idx < 0 or rec_idx >= n_recs:
        print(f'  rec_idx={rec_idx} out of range (0..{n_recs - 1})')
        return None

    # Double-dereference: same pattern as load_grid40_p1
    ref = p1_dataset[0, rec_idx]
    deref1 = f[ref]
    inner_ref = deref1[0, 0]
    data = f[inner_ref][()]

    if data.ndim != 4 or data.size == 0:
        print(f'  Rec {rec_idx + 1}: Empty or invalid data (shape={data.shape})')
        return None

    # HDF5 stores MATLAB (gridY, gridX, nTime, nTrials) transposed as
    # (nTrials, nTime, gridX, gridY).  Transpose back.
    data = data.transpose(3, 2, 1, 0)  # → (gridY, gridX, nTime, nTrials)
    return data


# ── Main plotting function ──────────────────────────────────────────────────

def plot_experimental_snapshots(
    conditions=('Naive', 'Beginner', 'Expert', 'NoSpout'),
    threshold=2.0,
    percentiles=(10, 30, 50, 70, 90),
    save_dir=r'Paper',
    show_ising_comparison=True,
    ex_sel=DEFAULT_EX_SEL,
    stim_frames=DEFAULT_STIM_FRAMES,
    ising_data_path=DEFAULT_ISING_DATA_PATH,
    ising_grid_mode=DEFAULT_ISING_GRID_MODE,
    ising_metric=DEFAULT_ISING_METRIC,
):
    """Generate Moran's I percentile snapshot figures for each condition.

    Parameters
    ----------
    conditions : sequence of str
        Training conditions to process.
    threshold : float
        Binarization threshold for dF/F.
    percentiles : sequence of int/float
        Moran's I percentiles at which to select frames.
    save_dir : str or None
        Root save directory.  Figures go to ``<save_dir>/Fig. 5 Model/``.
        Pass None to skip saving.
    show_ising_comparison : bool
        If True and 'Expert' in conditions, show Ising model comparison.
    ex_sel : int
        0-indexed Expert recording index for Ising comparison.
    stim_frames : tuple of int
        0-indexed stimulus frame indices to average.
    ising_data_path : str
        Path to Ising simulation data directory.
    ising_grid_mode : str
        Grid mode for Ising comparison results filename.
    ising_metric : str
        Metric for Ising comparison results filename.
    """
    # 1. Load colormap
    entropy_cmap = load_entropy_colormap()
    binary_cmap = ListedColormap([[1, 1, 1], [1, 0, 0]])

    # 2. Create weight matrix (once for all conditions)
    weight_mat = create_weight_matrix_jit(GRID_DIMENSIONS[0], GRID_DIMENSIONS[1], False)

    # 3. Open Grid40
    print(f'Loading Grid40 from {GRID40_PATH} ...')
    f = h5py.File(GRID40_PATH, 'r')

    n_pctiles = len(percentiles)
    n_examples = 2  # number of example figures per condition
    example_labels = ['A', 'B']

    try:
        for condition in conditions:
            print(f'\n=== {condition} ===')

            skip_set = SKIP.get(condition, set())
            pooled = load_grid40_p1(f, condition, skip_set)

            if pooled is None:
                print(f'No data for condition {condition}.')
                continue

            gridY, gridX, n_time, n_trials = pooled.shape
            print(f'Pooled: {n_trials} trials, {n_time} timepoints, '
                  f'grid [{gridY} x {gridX}]')

            # Binarize
            binarized = (pooled > threshold).astype(np.float64)

            # Reshape to (nFrames, gridY, gridX) for batch Moran's I
            n_frames = n_time * n_trials
            # pooled is (gridY, gridX, nTime, nTrials)
            # Reshape: iterate trials then time → frame order matches MATLAB
            # MATLAB loops trial=1..nTrials outer, t=1..nTime inner
            # so frame index = (trial-1)*nTime + t
            bin_frames = np.empty((n_frames, gridY, gridX), dtype=np.float64)
            idx = 0
            for trial in range(n_trials):
                for t in range(n_time):
                    bin_frames[idx] = binarized[:, :, t, trial]
                    idx += 1

            print(f'Computing Moran\'s I for {n_frames} frames...')
            morans_vec = morans_i_batch_nonan(bin_frames, weight_mat)

            # Filter valid (non-NaN)
            valid_mask = ~np.isnan(morans_vec)
            valid_mi = morans_vec[valid_mask]
            valid_idx = np.where(valid_mask)[0]

            if valid_mi.size == 0:
                print(f'All Moran\'s I values are NaN for {condition}.')
                continue

            # Select 2 closest frames per percentile for two example figures
            pct_values = np.percentile(valid_mi, percentiles)
            # sel_frame_idx[example][percentile], sel_mi[example][percentile]
            sel_frame_idx = np.empty((n_examples, n_pctiles), dtype=int)
            sel_mi = np.empty((n_examples, n_pctiles))

            for p in range(n_pctiles):
                dists = np.abs(valid_mi - pct_values[p])
                # argsort to get the closest frames in order
                sorted_pos = np.argsort(dists)
                for ex in range(n_examples):
                    pos = sorted_pos[ex]
                    sel_frame_idx[ex, p] = valid_idx[pos]
                    sel_mi[ex, p] = valid_mi[pos]

            # Convert linear frame indices to (time, trial)
            sel_trial = sel_frame_idx // n_time
            sel_time = sel_frame_idx % n_time

            # Determine shared colour-scale max across both examples
            vmax = 0.0
            for ex in range(n_examples):
                for p in range(n_pctiles):
                    frame = pooled[:, :, sel_time[ex, p], sel_trial[ex, p]]
                    frame_max = np.nanmax(frame)
                    if frame_max > vmax:
                        vmax = frame_max
            if vmax == 0:
                vmax = 1.0

            # Plot one figure per example
            for ex in range(n_examples):
                label = example_labels[ex]
                fig, axes = plt.subplots(2, n_pctiles,
                                         figsize=(2.2 * n_pctiles, 3.5))
                fig.suptitle(f'{condition} (example {label})',
                             fontsize=12, fontweight='bold')

                for p in range(n_pctiles):
                    cont_frame = pooled[:, :, sel_time[ex, p],
                                        sel_trial[ex, p]]
                    bin_frame = binarized[:, :, sel_time[ex, p],
                                          sel_trial[ex, p]]

                    # Top row: continuous dF/F
                    ax1 = axes[0, p]
                    ax1.imshow(cont_frame, vmin=0, vmax=vmax,
                               cmap=entropy_cmap, aspect='equal',
                               interpolation='nearest')
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    ax1.set_title(f'MI={sel_mi[ex, p]:.3f}', fontsize=7)
                    if p == 0:
                        ax1.set_ylabel('Data', fontsize=8)

                    # Bottom row: binarised
                    ax2 = axes[1, p]
                    ax2.imshow(bin_frame, vmin=0, vmax=1, cmap=binary_cmap,
                               aspect='equal', interpolation='nearest')
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    if p == 0:
                        ax2.set_ylabel('Binarised', fontsize=8)

                fig.subplots_adjust(hspace=0.15, wspace=0.08)

                # Save
                if save_dir is not None:
                    out_dir = os.path.join(save_dir, 'Fig. 5 Model')
                    os.makedirs(out_dir, exist_ok=True)

                    png_path = os.path.join(
                        out_dir,
                        f'experimental_snapshots_{condition}_{label}.png')
                    fig.savefig(png_path, dpi=300, bbox_inches='tight')
                    print(f'Saved: {png_path}')

                    svg_path = os.path.join(
                        out_dir,
                        f'experimental_snapshots_{condition}_{label}.svg')
                    fig.savefig(svg_path, format='svg', bbox_inches='tight')
                    print(f'Saved: {svg_path}')

                plt.close(fig)

        # ── Expert stim-period Ising comparison ─────────────────────────────
        if show_ising_comparison and 'Expert' in conditions:
            print('\n=== Expert Stim-Period Ising Comparison ===')

            # 1. Load single Expert recording
            ex_data = load_grid40_p1_single(f, 'Expert', ex_sel)
            if ex_data is None:
                print('Could not load Expert recording for Ising comparison.')
            else:
                gridY_ex, gridX_ex, n_time_ex, n_trials_ex = ex_data.shape

                # 2. Compute stim-period snapshot for trial 3 (0-indexed: 2)
                sel_trial_ising = 2
                stim_frames_arr = np.array(stim_frames)
                snapshot = np.mean(
                    ex_data[:, :, stim_frames_arr, sel_trial_ising], axis=2)
                snapshot_bin = (snapshot > threshold).astype(np.float64)

                # 3. Compute experimental Moran's I
                exp_mi = morans_i_single_jit(snapshot_bin, weight_mat)
                print(f'Experimental MI = {exp_mi:.4f}')

                # 4. Load IsingComparison results
                results_path = os.path.join(
                    ising_data_path, 'IsingComparison',
                    f'IsingComparison_Results_{ising_grid_mode}_{ising_metric}.mat')
                print(f'Loading Ising results from: {results_path}')

                sim_filename = None
                best_idx = None
                f_results = None
                try:
                    f_results = h5py.File(results_path, 'r')

                    # Navigate structure (handle both root layouts)
                    if 'Results' in f_results:
                        root = f_results['Results']
                    else:
                        root = f_results

                    ising_data_h5 = root['IsingData']
                    comparison_h5 = root['Comparison']

                    # best_idx is 0-indexed (Python-generated results)
                    best_idx = int(
                        comparison_h5['Expert']['bestMatch_idx'][()].flatten()[0])

                    # Reconstruct filename from params
                    params_h5 = ising_data_h5['params']
                    beta = float(params_h5['beta'][()].flatten()[best_idx])
                    c = float(params_h5['c'][()].flatten()[best_idx])
                    decay = float(
                        params_h5['decay_const'][()].flatten()[best_idx])
                    rad = float(
                        params_h5['inhibition_range'][()].flatten()[best_idx])
                    bias = float(params_h5['bias'][()].flatten()[best_idx])

                    # Format params matching generation script:
                    # whole-number floats become ints in the filename
                    def _fmt(v):
                        return int(v) if v == int(v) else v

                    sim_filename = (
                        f"sim_be_{_fmt(beta)}_c_{int(c)}_d_{int(decay)}"
                        f"_r_{int(rad)}_bi_{_fmt(bias)}.mat")
                except Exception as e_hdf5:
                    print(f'HDF5 load failed ({e_hdf5}), trying scipy...')
                    try:
                        mat_results = sio.loadmat(
                            results_path, squeeze_me=True,
                            struct_as_record=False)
                        if hasattr(mat_results.get('Results', None),
                                   'IsingData'):
                            res_obj = mat_results['Results']
                        else:
                            res_obj = type('R', (), mat_results)

                        ising_d = res_obj.IsingData
                        comp_d = res_obj.Comparison

                        best_idx = int(
                            np.asarray(comp_d.Expert.bestMatch_idx).flatten()[0])

                        p = ising_d.params
                        beta = float(np.asarray(p.beta).flatten()[best_idx])
                        c = float(np.asarray(p.c).flatten()[best_idx])
                        decay = float(
                            np.asarray(p.decay_const).flatten()[best_idx])
                        rad = float(
                            np.asarray(
                                p.inhibition_range).flatten()[best_idx])
                        bias = float(np.asarray(p.bias).flatten()[best_idx])

                        def _fmt(v):
                            return int(v) if v == int(v) else v

                        sim_filename = (
                            f"sim_be_{_fmt(beta)}_c_{int(c)}_d_{int(decay)}"
                            f"_r_{int(rad)}_bi_{_fmt(bias)}.mat")
                    except Exception as e_scipy:
                        print(f'scipy.io.loadmat also failed: {e_scipy}')
                finally:
                    if f_results is not None:
                        try:
                            f_results.close()
                        except Exception:
                            pass

                if sim_filename is None:
                    print('Could not load Ising comparison results.')
                else:
                    print(f'Best-matching simulation: {sim_filename} '
                          f'(index {best_idx})')

                    # 5. Load best-matching Ising simulation
                    sim_path = os.path.join(ising_data_path, sim_filename)
                    print(f'Loading simulation: {sim_path}')
                    try:
                        sim_data = sio.loadmat(sim_path)
                        stored_spins = sim_data['stored_spins']
                    except NotImplementedError:
                        # v7.3 HDF5 format
                        with h5py.File(sim_path, 'r') as f_sim:
                            stored_spins = f_sim['stored_spins'][()]
                            # HDF5 from MATLAB: dimensions reversed
                            stored_spins = stored_spins.transpose(2, 1, 0)

                    # Convert to binary: -1/+1 → 0/1
                    ising_binary = ((stored_spins + 1) / 2).astype(np.float64)

                    # 6. Compute tiled MI
                    row_starts, col_starts = generate_tiled_positions(
                        ISING_GRID, (gridY_ex, gridX_ex))
                    n_positions = len(row_starts)
                    print(f'Computing tiled MI ({n_positions} positions, '
                          f'{ising_binary.shape[0]} frames)...')

                    tiled_mi = morans_i_tiled_batch_separate(
                        ising_binary, weight_mat,
                        row_starts, col_starts,
                        gridY_ex, gridX_ex)

                    # 7. Find top-3 closest matches
                    n_ising_examples = 3
                    mi_diff = np.abs(tiled_mi - exp_mi)
                    flat_sort = np.argsort(mi_diff.ravel())

                    ising_snapshots = np.zeros(
                        (gridY_ex, gridX_ex, n_ising_examples))
                    ising_mi_vals = np.zeros(n_ising_examples)

                    for ex in range(n_ising_examples):
                        best_pos, best_frame = np.unravel_index(
                            flat_sort[ex], tiled_mi.shape)
                        r_start = row_starts[best_pos]
                        c_start = col_starts[best_pos]
                        ising_snapshots[:, :, ex] = ising_binary[
                            best_frame,
                            r_start:r_start + gridY_ex,
                            c_start:c_start + gridX_ex]
                        ising_mi_vals[ex] = tiled_mi[best_pos, best_frame]

                    mi_str = ' / '.join(
                        f'{v:.4f}' for v in ising_mi_vals)
                    print(f'  Exp MI={exp_mi:.4f} -> Ising MI: {mi_str}')

                    # 8. Plot 1×4 figure
                    n_cols = 1 + n_ising_examples
                    fig_ising, axes_ising = plt.subplots(
                        1, n_cols, figsize=(2.5 * n_cols, 2.5))
                    fig_ising.suptitle(
                        'Expert Stim-Period: Experiment vs Ising',
                        fontsize=12, fontweight='bold')

                    # Column 1: experimental binarised snapshot
                    ax = axes_ising[0]
                    ax.imshow(snapshot_bin, vmin=0, vmax=1,
                              cmap=binary_cmap, aspect='equal',
                              interpolation='nearest')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Experimental Data\nMI={exp_mi:.3f}',
                                 fontsize=7)

                    # Columns 2-4: Ising examples
                    for ex in range(n_ising_examples):
                        ax = axes_ising[1 + ex]
                        ax.imshow(ising_snapshots[:, :, ex], vmin=0, vmax=1,
                                  cmap=binary_cmap, aspect='equal',
                                  interpolation='nearest')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(
                            f'Ising {ex + 1}\nMI={ising_mi_vals[ex]:.3f}',
                            fontsize=7)

                    fig_ising.subplots_adjust(wspace=0.08)

                    # 9. Save
                    if save_dir is not None:
                        out_dir = os.path.join(save_dir, 'Fig. 5 Model')
                        os.makedirs(out_dir, exist_ok=True)

                        png_path = os.path.join(
                            out_dir,
                            'experimental_ising_comparison_Expert.png')
                        fig_ising.savefig(
                            png_path, dpi=300, bbox_inches='tight')
                        print(f'Saved: {png_path}')

                        svg_path = os.path.join(
                            out_dir,
                            'experimental_ising_comparison_Expert.svg')
                        fig_ising.savefig(
                            svg_path, format='svg', bbox_inches='tight')
                        print(f'Saved: {svg_path}')

                    plt.close(fig_ising)

                    # 10. Duplicate with faint grid overlay
                    fig_grid, axes_grid = plt.subplots(
                        1, n_cols, figsize=(2.5 * n_cols, 2.5))
                    fig_grid.suptitle(
                        'Expert Stim-Period: Experiment vs Ising',
                        fontsize=12, fontweight='bold')

                    panels = [snapshot_bin] + [
                        ising_snapshots[:, :, ex]
                        for ex in range(n_ising_examples)]
                    titles = [f'Experimental Data\nMI={exp_mi:.3f}'] + [
                        f'Ising {ex + 1}\nMI={ising_mi_vals[ex]:.3f}'
                        for ex in range(n_ising_examples)]

                    for ax, panel, title in zip(axes_grid, panels, titles):
                        ax.imshow(panel, vmin=0, vmax=1,
                                  cmap=binary_cmap, aspect='equal',
                                  interpolation='nearest')
                        h, w = panel.shape
                        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
                        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
                        ax.grid(which='minor', color='gray',
                                linewidth=0.3, alpha=0.3)
                        ax.tick_params(which='minor', length=0)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(title, fontsize=7)

                    fig_grid.subplots_adjust(wspace=0.08)

                    if save_dir is not None:
                        png_path = os.path.join(
                            out_dir,
                            'experimental_ising_comparison_Expert_grid.png')
                        fig_grid.savefig(
                            png_path, dpi=300, bbox_inches='tight')
                        print(f'Saved: {png_path}')

                        svg_path = os.path.join(
                            out_dir,
                            'experimental_ising_comparison_Expert_grid.svg')
                        fig_grid.savefig(
                            svg_path, format='svg', bbox_inches='tight')
                        print(f'Saved: {svg_path}')

                    plt.close(fig_grid)

    finally:
        f.close()

    print('\nDone.')


if __name__ == '__main__':
    plot_experimental_snapshots()
