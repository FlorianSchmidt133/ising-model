#!/usr/bin/env python3
"""
plot_nmf_binary_overlap.py

Generates publication-quality vector figures (SVG + PDF) showing the spatial
overlap between NMF-decomposed components and threshold-based binarization
of Grid40 neural activity data.

Reproduces the MATLAB Section 9 "NMF-Binary Overlap" panels but with proper
vector output that opens cleanly in Illustrator.

Usage:
    python plot_nmf_binary_overlap.py
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat
import os

# =============================================================================
# Configuration
# =============================================================================
NMF_PATH = r'NMF_littleRegularisation_Grid40_AllConditions.mat'
GRID40_PATH = r'Grid40.mat'
CMAP_PATH = 'EntropyColourMap.mat'  # bundled in utils/ (see README)
OUTPUT_DIR = r'Paper\Fig. 5 Model\ThresholdAnalysis\thres\python'

CONDITION = 'ExpertIndividual'
SKIP_RECS = [0, 3, 11, 12, 13]   # 0-indexed: MATLAB [1, 4, 12, 13, 14]
GRID_DIMS = (13, 26)
N_TOP = 10
DICE_THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 5.0]
STIM_ONSET = 81  # frame index (1-indexed, used for plotting)


# =============================================================================
# Helper functions
# =============================================================================

def load_grid40_data(f_grid, condition, rec_idx):
    """Load full 4D Grid40 data for one recording.

    Returns: (nTrials, nTimepoints, gridY, gridX) ndarray
    """
    p1_refs = f_grid['Grid40'][condition]['AllNeurons']['P1']
    rec_ref = f_grid[p1_refs[0, rec_idx]]
    data_4d = np.array(f_grid[rec_ref[0, 0]])  # (nTrials, 185, 26, 13)
    # Transpose last two dims to get (nTrials, 185, 13, 26) = MATLAB convention
    data_4d = np.transpose(data_4d, (0, 1, 3, 2))
    return data_4d


def load_nmf_trial(f_nmf, condition, rec_idx, trial_idx):
    """Load W (338 x nComp) and H (nComp x 185) for one trial.

    Returns: (W, H) tuple of ndarrays
    """
    nmf = f_nmf['NMF_littleRegularisation_Grid40_AllConditions']
    w_refs = nmf['W_all2'][condition]['P1']  # (nRecs, 1) object refs
    h_refs = nmf['H_all2'][condition]['P1']

    w_rec = f_nmf[w_refs[rec_idx, 0]]   # (nTrials, 1) object refs
    h_rec = f_nmf[h_refs[rec_idx, 0]]

    W = np.array(f_nmf[w_rec[trial_idx, 0]]).T   # (10, 338) -> (338, 10)
    H = np.array(f_nmf[h_rec[trial_idx, 0]]).T   # (185, 10) -> (10, 185)
    return W, H


def get_n_trials_nmf(f_nmf, condition, rec_idx):
    """Get number of trials available in NMF data for a recording."""
    nmf = f_nmf['NMF_littleRegularisation_Grid40_AllConditions']
    w_refs = nmf['W_all2'][condition]['P1']
    w_rec = f_nmf[w_refs[rec_idx, 0]]
    return w_rec.shape[0]


def compute_dice(mask_a, mask_b):
    """Dice coefficient between two boolean masks."""
    intersection = np.sum(mask_a & mask_b)
    total = np.sum(mask_a) + np.sum(mask_b)
    if total == 0:
        return 0.0
    return 2.0 * intersection / total


def nmf_to_grid(vec, grid_dims):
    """Reshape a 338-element NMF vector to (13, 26) grid using MATLAB
    column-major (Fortran) order to match MATLAB's reshape convention."""
    return vec.reshape(grid_dims, order='F')


def make_nmf_mask(W, H, frame_idx, grid_dims):
    """Binarise NMF reconstruction at median + 1*std."""
    recon = nmf_to_grid(W @ H[:, frame_idx], grid_dims)
    threshold = np.median(recon) + np.std(recon)
    return recon > threshold, recon


def plot_overlap_panel(fig_path_stem, W, H, grid_trial, rec, trial,
                       peak_frame, dice_score, entropy_cmap, binary_cmap):
    """Create a 3-row x 5-col panel figure for one trial.

    Row 1: Top-5 NMF spatial components (W)
    Row 2: W*H temporal traces + fraction-active overlay
    Row 3: Raw | NMF recon | Binary | Contour overlay | Summary text
    """
    n_comp = W.shape[1]
    n_tp = H.shape[1]

    # Rank components by peak temporal amplitude
    comp_amplitudes = np.array([np.mean(W[:, ci] * np.max(np.abs(H[ci, :])))
                                for ci in range(n_comp)])
    comp_order = np.argsort(comp_amplitudes)[::-1]

    # Binary at threshold=2 for all frames
    binary_trial = grid_trial > 2.0
    frac_active = np.mean(binary_trial.reshape(n_tp, -1), axis=1) * 100

    # Peak frame data
    raw_frame = grid_trial[peak_frame, :, :]  # (13, 26)
    bin_frame = binary_trial[peak_frame, :, :]
    nmf_mask, recon_frame = make_nmf_mask(W, H, peak_frame, GRID_DIMS)

    # --- Create figure ---
    fig = plt.figure(figsize=(22, 13))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[1, 1.2, 1],
                 hspace=0.35, wspace=0.3)

    # Row 1: Top-5 spatial components
    n_show = min(5, n_comp)
    for ci in range(n_show):
        ax = fig.add_subplot(gs[0, ci])
        cidx = comp_order[ci]
        w_spatial = nmf_to_grid(W[:, cidx], GRID_DIMS)
        im = ax.imshow(w_spatial, cmap=entropy_cmap, origin='lower',
                       aspect='equal')
        ax.set_title(f'W comp {cidx + 1}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ci in range(n_show, 5):
        ax = fig.add_subplot(gs[0, ci])
        ax.axis('off')

    # Row 2: W*H temporal traces + fraction-active (span all 5 cols)
    ax_trace = fig.add_subplot(gs[1, :])
    colors = plt.cm.tab10(np.linspace(0, 1, n_comp))
    for ci in range(n_comp):
        recon_ci = np.mean(W[:, ci:ci+1] @ H[ci:ci+1, :], axis=0)
        ax_trace.plot(np.arange(1, n_tp + 1), recon_ci,
                      color=colors[ci], alpha=0.5, linewidth=1.2)
    ax_trace.set_ylabel('Mean W·H per component', fontsize=10)
    ax_trace.set_xlabel('Timepoint (frames)', fontsize=10)

    # Fraction active on twin axis
    ax_twin = ax_trace.twinx()
    ax_twin.plot(np.arange(1, n_tp + 1), frac_active, '--',
                 color=(0, 0.5, 0), linewidth=2.5, label='% active at 2.0 dF/F')
    ax_twin.set_ylabel('% grid cells active', fontsize=10, color=(0, 0.5, 0))
    ax_twin.tick_params(axis='y', labelcolor=(0, 0.5, 0))

    # Mark stim onset and peak frame
    ax_trace.axvline(STIM_ONSET, color='gray', linestyle=':', linewidth=1,
                     label='stim')
    ax_trace.axvline(peak_frame + 1, color=(0.7, 0, 0), linewidth=1.2,
                     label='peak')
    ax_trace.set_title(f'NMF components (W·H) + % active at 2.0 dF/F '
                       f'(Dice={dice_score:.3f} at frame {peak_frame + 1})',
                       fontsize=11)
    ax_trace.grid(True, alpha=0.3)

    # Row 3: Spatial comparison at peak frame
    vmax = np.nanpercentile(grid_trial, 99)

    # Tile 1: Raw
    ax1 = fig.add_subplot(gs[2, 0])
    im1 = ax1.imshow(raw_frame, cmap=entropy_cmap, origin='lower',
                     aspect='equal', vmin=0, vmax=vmax)
    ax1.set_title(f'Raw (frame {peak_frame + 1})', fontsize=10)
    ax1.set_xticks([]); ax1.set_yticks([])
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Tile 2: NMF reconstruction
    ax2 = fig.add_subplot(gs[2, 1])
    im2 = ax2.imshow(recon_frame, cmap=entropy_cmap, origin='lower',
                     aspect='equal')
    ax2.set_title('NMF reconstruction', fontsize=10)
    ax2.set_xticks([]); ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Tile 3: Binary mask (active=black, inactive=white)
    ax3 = fig.add_subplot(gs[2, 2])
    im3 = ax3.imshow(bin_frame.astype(float), cmap=binary_cmap,
                     origin='lower', aspect='equal', vmin=0, vmax=1)
    ax3.set_title('Binary (>2.0 dF/F)', fontsize=10, color=(0, 0.5, 0),
                  fontweight='bold')
    ax3.set_xticks([]); ax3.set_yticks([])
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Tile 4: Raw + contours (green=binary boundary, red=NMF)
    ax4 = fig.add_subplot(gs[2, 3])
    ax4.imshow(raw_frame, cmap=entropy_cmap, origin='lower',
               aspect='equal', vmin=0, vmax=vmax)
    # Green contours: binary boundary
    ax4.contour(bin_frame.astype(float), levels=[0.5],
                colors=[(0, 0.8, 0)], linewidths=2.0)
    # Red contours: NMF reconstruction
    ax4.contour(recon_frame, levels=3, colors=['r'], linewidths=1.5)
    ax4.set_title('Raw + contours\n(green=2dF/F, red=NMF)', fontsize=9)
    ax4.set_xticks([]); ax4.set_yticks([])

    # Tile 5: Summary text
    ax5 = fig.add_subplot(gs[2, 4])
    ax5.axis('off')
    summary = (f"Expert Rec {rec + 1} Trial {trial + 1}\n\n"
               f"Dice coefficient: {dice_score:.4f}\n"
               f"Peak frame: {peak_frame + 1}\n"
               f"Frac active: {frac_active[peak_frame]:.1f}%\n"
               f"NMF components: {n_comp}\n"
               f"Preprocessing: Mean-subtracted\n\n"
               f"Green = threshold 2.0 dF/F\n"
               f"Red = NMF reconstruction")
    ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
             verticalalignment='top', fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle(f'NMF–Binary Overlap: Expert Rec {rec + 1} Trial {trial + 1} '
                 f'(Dice={dice_score:.3f})', fontsize=14, fontweight='bold')

    # Save
    for fmt in ['svg', 'pdf']:
        fig.savefig(f'{fig_path_stem}.{fmt}', format=fmt,
                    bbox_inches='tight', dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Output directory: {OUTPUT_DIR}')

    # Load colourmap
    cmap_data = loadmat(CMAP_PATH)['EntropyColourmap']
    entropy_cmap = ListedColormap(cmap_data)
    binary_cmap = ListedColormap([[1, 1, 1], [0, 0, 0]])

    # Open data files
    f_grid = h5py.File(GRID40_PATH, 'r')
    f_nmf = h5py.File(NMF_PATH, 'r')

    p1_refs = f_grid['Grid40'][CONDITION]['AllNeurons']['P1']
    n_recs = p1_refs.shape[1]
    print(f'Found {n_recs} recordings for {CONDITION}')

    # =========================================================================
    # Phase 1: Search all Expert trials for best NMF-binary overlap
    # =========================================================================
    print('\n--- Searching Expert trials for NMF-binary overlap ---')

    # overlap_results: list of dicts with rec, trial, dice_2, peak_frame,
    # and dice at each threshold
    overlap_results = []

    for r in range(n_recs):
        if r in SKIP_RECS:
            continue

        try:
            grid_data = load_grid40_data(f_grid, CONDITION, r)
        except Exception:
            continue

        n_trials_grid = grid_data.shape[0]

        try:
            n_trials_nmf = get_n_trials_nmf(f_nmf, CONDITION, r)
        except Exception:
            continue

        n_trials = min(n_trials_grid, n_trials_nmf)

        for t in range(n_trials):
            try:
                W, H = load_nmf_trial(f_nmf, CONDITION, r, t)
            except Exception:
                continue

            if W.size == 0 or H.size == 0:
                continue

            trial_data = grid_data[t]  # (185, 13, 26)

            # Find peak frame (max fraction active at threshold=2)
            bin_trial = trial_data > 2.0
            frac_act = np.mean(bin_trial.reshape(trial_data.shape[0], -1),
                               axis=1)
            peak_frame = int(np.argmax(frac_act))

            # NMF mask at peak frame
            nmf_mask, _ = make_nmf_mask(W, H, peak_frame, GRID_DIMS)

            # Dice at threshold=2
            bin_mask_2 = bin_trial[peak_frame]
            dice_2 = compute_dice(nmf_mask, bin_mask_2)

            # Dice at all thresholds
            dice_all = {}
            for thr in DICE_THRESHOLDS:
                bin_mask_thr = trial_data[peak_frame] > thr
                dice_all[thr] = compute_dice(nmf_mask, bin_mask_thr)

            overlap_results.append({
                'rec': r,
                'trial': t,
                'dice_2': dice_2,
                'peak_frame': peak_frame,
                'dice_all': dice_all,
            })

        if (r + 1) % 5 == 0:
            print(f'  Scanned Rec {r + 1}/{n_recs} '
                  f'({len(overlap_results)} trials so far)')

    print(f'  Total: {len(overlap_results)} trials scanned')

    if not overlap_results:
        print('  No valid trials found. Exiting.')
        f_grid.close()
        f_nmf.close()
        return

    # Overall Dice statistics
    all_dice_2 = [r['dice_2'] for r in overlap_results]
    print(f'\n  Overall Dice (threshold=2.0) statistics:')
    print(f'    Mean:   {np.mean(all_dice_2):.4f}')
    print(f'    Median: {np.median(all_dice_2):.4f}')
    print(f'    Std:    {np.std(all_dice_2):.4f}')
    print(f'    Min:    {np.min(all_dice_2):.4f}')
    print(f'    Max:    {np.max(all_dice_2):.4f}')
    print(f'    n:      {len(all_dice_2)} trials')

    # Sort by Dice at threshold=2 descending
    overlap_sorted = sorted(overlap_results, key=lambda x: x['dice_2'],
                            reverse=True)

    print(f'\n  Top {min(N_TOP, len(overlap_sorted))} trials:')
    for k, entry in enumerate(overlap_sorted[:N_TOP]):
        print(f'    #{k+1}: Rec {entry["rec"]+1} Trial {entry["trial"]+1} '
              f'— Dice = {entry["dice_2"]:.4f} '
              f'(peak frame {entry["peak_frame"]+1})')

    # =========================================================================
    # Phase 2: Plot top-N 3x5 panels
    # =========================================================================
    print(f'\n--- Generating top-{N_TOP} overlap panel figures ---')

    n_plot = min(N_TOP, len(overlap_sorted))
    for rank in range(n_plot):
        entry = overlap_sorted[rank]
        r, t = entry['rec'], entry['trial']
        dice_score = entry['dice_2']
        peak_frame = entry['peak_frame']

        print(f'  Plotting #{rank+1}: Rec {r+1} Trial {t+1} '
              f'(Dice={dice_score:.3f})')

        W, H = load_nmf_trial(f_nmf, CONDITION, r, t)
        grid_data = load_grid40_data(f_grid, CONDITION, r)
        grid_trial = grid_data[t]  # (185, 13, 26)

        fig_stem = os.path.join(
            OUTPUT_DIR,
            f'NMF_overlap_top{rank+1:02d}_Rec{r+1}_Trial{t+1}')

        plot_overlap_panel(fig_stem, W, H, grid_trial, r, t,
                           peak_frame, dice_score, entropy_cmap, binary_cmap)

    # =========================================================================
    # Phase 3: Dice distribution figure
    # =========================================================================
    print('\n--- Generating Dice distribution figure ---')

    fig_dice, (ax_box, ax_paired) = plt.subplots(1, 2, figsize=(14, 6))

    # Collect per-threshold distributions
    dice_by_thresh = {}
    for thr in DICE_THRESHOLDS:
        dice_by_thresh[thr] = [r['dice_all'][thr] for r in overlap_results]

    # Tile 1: Box plot
    box_data = [dice_by_thresh[thr] for thr in DICE_THRESHOLDS]
    box_labels = [f'{thr} dF/F' for thr in DICE_THRESHOLDS]
    box_colors = [(0.6, 0.6, 0.6), (0.2, 0.4, 0.8), (0, 0.5, 0),
                  (0.8, 0.5, 0), (0.8, 0, 0)]

    bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True,
                        widths=0.5)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for median_line in bp['medians']:
        median_line.set_color('black')
        median_line.set_linewidth(2)

    # Overlay individual points with jitter
    for gi, thr in enumerate(DICE_THRESHOLDS):
        x_jitter = (gi + 1) + (np.random.rand(len(box_data[gi])) - 0.5) * 0.25
        ax_box.scatter(x_jitter, box_data[gi], s=15, c='black', alpha=0.3,
                       zorder=3)

    ax_box.set_ylabel('Dice coefficient', fontsize=11)
    ax_box.set_title('NMF–Binary Overlap by Threshold', fontsize=12)
    ax_box.grid(True, alpha=0.3)

    # Print stats
    print(f'\n  Dice distributions (n={len(overlap_results)} trials):')
    for thr in DICE_THRESHOLDS:
        vals = dice_by_thresh[thr]
        print(f'    {thr} dF/F: mean={np.mean(vals):.4f}, '
              f'median={np.median(vals):.4f}, std={np.std(vals):.4f}')

    # Tile 2: Paired line plot
    n_trials_plot = min(len(overlap_results), 200)
    x_pos = np.arange(1, len(DICE_THRESHOLDS) + 1)

    for ti in range(n_trials_plot):
        y_vals = [overlap_results[ti]['dice_all'][thr]
                  for thr in DICE_THRESHOLDS]
        ax_paired.plot(x_pos, y_vals, '-', color=(0.7, 0.7, 0.7),
                       alpha=0.15, linewidth=0.5)

    # Mean line
    mean_vals = [np.mean(dice_by_thresh[thr]) for thr in DICE_THRESHOLDS]
    ax_paired.plot(x_pos, mean_vals, 'ko-', linewidth=3, markersize=10,
                   markerfacecolor='k', zorder=5)

    # Highlight 2.0 dF/F
    idx_2 = DICE_THRESHOLDS.index(2.0)
    ax_paired.plot(idx_2 + 1, mean_vals[idx_2], 's', color=(0, 0.5, 0),
                   markersize=14, linewidth=2.5, markerfacecolor=(0, 0.5, 0),
                   zorder=6)

    ax_paired.set_xticks(x_pos)
    ax_paired.set_xticklabels(box_labels)
    ax_paired.set_ylabel('Dice coefficient', fontsize=11)
    ax_paired.set_title('Paired: each line = one trial', fontsize=12)
    ax_paired.set_xlim(0.5, len(DICE_THRESHOLDS) + 0.5)
    ax_paired.grid(True, alpha=0.3)

    fig_dice.suptitle(
        f'Dice Coefficient: NMF vs Binary Overlap at {len(DICE_THRESHOLDS)} '
        f'Thresholds (n={len(overlap_results)} Expert trials)',
        fontsize=13, fontweight='bold')
    fig_dice.tight_layout()

    dice_stem = os.path.join(OUTPUT_DIR, 'Dice_distribution_comparison')
    for fmt in ['svg', 'pdf']:
        fig_dice.savefig(f'{dice_stem}.{fmt}', format=fmt,
                         bbox_inches='tight', dpi=150)
    plt.close(fig_dice)

    # Cleanup
    f_grid.close()
    f_nmf.close()

    print(f'\nAll figures saved to: {OUTPUT_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
