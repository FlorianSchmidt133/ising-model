# -*- coding: utf-8 -*-
"""
Generate Perturbation Snapshot & Heatmap Figures

Modes:
  snapshots (default) - 1 replicate per (cond, size, dur), captures grid snapshots
  heatmap             - n_reps replicates, generates probability heatmaps + asymmetry metrics
  all                 - Both snapshots and heatmap analyses

Usage:
    # Snapshot mode (default)
    python generate_perturbation_snapshots.py --output test --comparison <path>

    # Heatmap mode (100 replicates)
    python generate_perturbation_snapshots.py --output test --comparison <path> \
        --mode heatmap --n-reps 100

    # Heatmap mode - single SLURM array task
    python generate_perturbation_snapshots.py --output test --comparison <path> \
        --mode heatmap --index 0

    # Combine per-combo SLURM results
    python generate_perturbation_snapshots.py --output test --combine

    # Regenerate heatmap figures from saved data
    python generate_perturbation_snapshots.py --output test --heatmap-only

    # Regenerate snapshot figures from saved data
    python generate_perturbation_snapshots.py --output test --figures-only

    # Dry run
    python generate_perturbation_snapshots.py --output test --comparison <path> --mode heatmap --scan

    # Count SLURM jobs
    python generate_perturbation_snapshots.py --output test --comparison <path> --count-jobs
"""

import numpy as np
import os
import argparse
import time
from numba import njit
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Import non-Numba utilities from run_ising_perturbations (safe to share)
# ---------------------------------------------------------------------------
from run_ising_perturbations import (
    load_comparison_results,
    find_comparison_results,
    compute_stimulus_durations,
    get_stimulus_region,
    apply_stimulus,
    compute_wavefront_anisotropy,
    L, M,
    PRE_STIM_FRAMES,
    POST_STIM_FRAMES,
    CROP_ROW_START, CROP_ROW_END,
    CROP_COL_START, CROP_COL_END,
    CONDITIONS,
    STIMULUS_SIZES,
    BURN_IN_MIN,
    BURN_IN_TAU_MULT,
    DECORR_MIN,
    DECORR_TAU_MULT,
    BLOB_SIGMA,
    BLOB_THRESHOLD,
    MIN_BLOB_SIZE,
    STIM_MODE_SEED_OFFSET,
)
from scipy.ndimage import gaussian_filter, label as scipy_label

# Stimulus modes supported by the snapshot/heatmap pipeline.
# Maps CLI name -> internal mode + optional bias value.
# Bias values mirror STIMULUS_BIAS_VALUES in run_ising_perturbations.py.
_BIAS_FAMILIES = ('bias', 'double_pulse_bias', 'double_pulse_bias3',
                  'double_pulse_bias5', 'double_pulse_bias10')
_BIAS_VALUES = (0.15, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0)


def _bias_label(value):
    return ('%.2f' % value).replace('.', 'p')


SNAPSHOT_STIM_MODES = {
    'clamped':        {'mode': 'clamped',        'stim_bias': None},
    'double_pulse':   {'mode': 'double_pulse',   'stim_bias': None},
    'double_pulse3':  {'mode': 'double_pulse3',  'stim_bias': None},
    'double_pulse5':  {'mode': 'double_pulse5',  'stim_bias': None},
    'double_pulse10': {'mode': 'double_pulse10', 'stim_bias': None},
}
for _family in _BIAS_FAMILIES:
    for _bv in _BIAS_VALUES:
        SNAPSHOT_STIM_MODES['%s_%s' % (_family, _bias_label(_bv))] = {
            'mode': _family, 'stim_bias': _bv,
        }
del _family, _bv


def mode_uses_bias(stim_mode):
    """Return True if the mode requires a stim_bias value (uses bias_field path)."""
    return stim_mode == 'bias' or stim_mode.startswith('double_pulse_bias')

# Half-crop region (midpoint between full grid and FOV crop, centred at grid centre)
HALF_CROP_ROW_START, HALF_CROP_ROW_END = 6, 32
HALF_CROP_COL_START, HALF_CROP_COL_END = 13, 65

# Per-size replicate overrides (default: 15)
DEFAULT_EXAMPLE_REPS = 15
REPS_PER_SIZE = {2: 20, 3: 20, 4: 20}


# =============================================================================
# Numba functions — duplicated per-module for caching (see MEMORY.md)
# =============================================================================

def build_diamond_kernel(rad):
    """
    Construct the diamond-shaped inhibition kernel.

    EXCLUDES the center (0,0) and 4 nearest neighbors (±1,0), (0,±1)
    so that NN contribute only to excitation (J=+1) and more distant
    neurons contribute only to inhibition.
    """
    kernel = np.zeros((2*rad+1, 2*rad+1), dtype=np.float64)
    center = rad
    for i in range(rad+1):
        kernel[i, rad-i:(rad+i+1)] = 1
        kernel[-i-1, rad-i:(rad+i+1)] = 1

    kernel[center, center] = 0
    kernel[center-1, center] = 0
    kernel[center+1, center] = 0
    kernel[center, center-1] = 0
    kernel[center, center+1] = 0

    return kernel


@njit(cache=True)
def heat_bath_numba(config, beta, c, decay_const, H, K, bias_val, K_sum):
    """
    Numba-optimized Monte Carlo sweep using heat-bath algorithm.
    """
    L = config.shape[0]
    M = config.shape[1]
    kL = K.shape[0]
    kM = K.shape[1]
    radL = kL // 2
    radM = kM // 2
    dt = 1.0 / (L * M)

    for _ in range(L * M):
        i = np.random.randint(0, L)
        j = np.random.randint(0, M)

        h = (config[(i + 1) % L, j] +
             config[(i - 1 + L) % L, j] +
             config[i, (j + 1) % M] +
             config[i, (j - 1 + M) % M])

        local_field = float(h) + H[i, j] + bias_val

        pi_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * local_field))

        if np.random.random() < pi_plus:
            config[i, j] = 1
        else:
            config[i, j] = -1

        inhib_sum = 0.0
        for di in range(kL):
            for dj in range(kM):
                if K[di, dj] > 0:
                    row_idx = (i - radL + di + L) % L
                    col_idx = (j - radM + dj + M) % M
                    inhib_sum += K[di, dj] * config[row_idx, col_idx]

        inhib = inhib_sum / K_sum

        H[i, j] = H[i, j] - (c * inhib * dt) - (decay_const * H[i, j] * dt)

    return config, H


@njit(cache=True)
def heat_bath_numba_with_bias_field(config, beta, c, decay_const, H, K, bias_field, K_sum):
    """
    Numba-optimized Monte Carlo sweep with 2D bias field support.

    Duplicated per-module for Numba caching (see comment at top of Numba section).
    """
    L = config.shape[0]
    M = config.shape[1]
    kL = K.shape[0]
    kM = K.shape[1]
    radL = kL // 2
    radM = kM // 2
    dt = 1.0 / (L * M)

    for _ in range(L * M):
        i = np.random.randint(0, L)
        j = np.random.randint(0, M)

        h = (config[(i + 1) % L, j] +
             config[(i - 1 + L) % L, j] +
             config[i, (j + 1) % M] +
             config[i, (j - 1 + M) % M])

        local_field = float(h) + H[i, j] + bias_field[i, j]

        pi_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * local_field))

        if np.random.random() < pi_plus:
            config[i, j] = 1
        else:
            config[i, j] = -1

        inhib_sum = 0.0
        for di in range(kL):
            for dj in range(kM):
                if K[di, dj] > 0:
                    row_idx = (i - radL + di + L) % L
                    col_idx = (j - radM + dj + M) % M
                    inhib_sum += K[di, dj] * config[row_idx, col_idx]

        inhib = inhib_sum / K_sum

        H[i, j] = H[i, j] - (c * inhib * dt) - (decay_const * H[i, j] * dt)

    return config, H


# =============================================================================
# Decorrelated Equilibrium Sampling
# =============================================================================

def prepare_decorrelated_states(params, n_reps, seed):
    """
    Generate n_reps decorrelated equilibrium states for multi-replicate runs.

    Uses the same pattern as run_ising_perturbations.run_perturbation_job():
    single burn-in from random init, then sample states with decorrelation
    gap between them.

    Parameters
    ----------
    params : dict
        Simulation parameters (beta, c, decay_const, inhibition_range, bias).
    n_reps : int
        Number of independent equilibrium states to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of (spin_config, H) tuples
        Each tuple contains copies of the spin configuration and H field
        at a decorrelated equilibrium point.
    """
    beta = params['beta']
    c = params['c']
    decay_const = params['decay_const']
    rad = params['inhibition_range']
    bias = params['bias']

    kernel = build_diamond_kernel(rad)
    K_sum = float(np.sum(kernel))

    np.random.seed(seed)
    spin_config = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, M))
    H = np.zeros((L, M), dtype=np.float64)

    # Burn in to reach equilibrium
    tau = (L * M) / decay_const
    burn_in_steps = max(BURN_IN_MIN, int(BURN_IN_TAU_MULT * tau))
    for _ in range(burn_in_steps):
        spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias, K_sum)

    # Sample decorrelated states along the trajectory
    decorrelation_gap = max(DECORR_MIN, int(DECORR_TAU_MULT * tau))
    eq_states = [(spin_config.copy(), H.copy())]
    for _ in range(n_reps - 1):
        for __ in range(decorrelation_gap):
            spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias, K_sum)
        eq_states.append((spin_config.copy(), H.copy()))

    return eq_states


# =============================================================================
# Blob Asymmetry Metrics
# =============================================================================

def compute_blob_asymmetry_metrics(frame, stim_rows, stim_cols, center_row, center_col):
    """
    Compute spatial asymmetry metrics on a single snapshot frame.

    Uses the same blob detection pipeline as run_ising_perturbations
    (gaussian smooth -> threshold -> label -> find stimulus-connected blob).

    Parameters
    ----------
    frame : ndarray (L, M), int8
        Spin configuration (+1 or -1).
    stim_rows, stim_cols : ndarray
        Row and column indices of the stimulus region.
    center_row, center_col : int
        Centre of the stimulus region.

    Returns
    -------
    dict with keys:
        eccentricity : float
            Inertia-tensor eccentricity of stimulus-connected blob.
            0 = circle, 1 = line. NaN if blob too small.
        centroid_mag : float
            Distance of blob centroid from stimulus centre (px).
        centroid_dir : float
            Direction of centroid offset (radians, 0=right, pi/2=down).
        anisotropy : float
            Wavefront anisotropy (0=isotropic, 1=directional).
        anisotropy_dir : float
            Primary direction of wavefront spread (radians).
        sectors : ndarray (8,)
            Fraction of excess blob area in each of 8 compass sectors
            (N, NE, E, SE, S, SW, W, NW). Sums to 1.
    """
    nan_result = {
        'eccentricity': np.nan,
        'centroid_mag': np.nan,
        'centroid_dir': np.nan,
        'anisotropy': np.nan,
        'anisotropy_dir': np.nan,
        'sectors': np.full(8, np.nan),
        'gauss_blob_area': np.nan,
    }

    # Convert -1/+1 to 0/1
    binary_frame = (frame.astype(np.float64) + 1.0) / 2.0

    # Blob detection pipeline
    smoothed = gaussian_filter(binary_frame, sigma=BLOB_SIGMA)
    binary_blob = smoothed > BLOB_THRESHOLD
    labeled, num_features = scipy_label(binary_blob)

    if num_features == 0:
        return nan_result

    # Find blob overlapping stimulus region
    stim_labels = labeled[np.ix_(stim_rows, stim_cols)]
    label_counts = np.bincount(stim_labels.flatten())
    if len(label_counts) <= 1:
        return nan_result

    label_counts[0] = 0  # Ignore background
    stim_blob_label = np.argmax(label_counts)
    if stim_blob_label == 0 or label_counts[stim_blob_label] == 0:
        return nan_result

    blob_mask = (labeled == stim_blob_label)
    blob_area = np.sum(blob_mask)
    if blob_area < MIN_BLOB_SIZE:
        return nan_result

    rows, cols = np.where(blob_mask)

    # --- Eccentricity from inertia tensor ---
    r_centered = rows - np.mean(rows)
    c_centered = cols - np.mean(cols)
    Ixx = np.sum(c_centered ** 2)
    Iyy = np.sum(r_centered ** 2)
    Ixy = -np.sum(r_centered * c_centered)

    # Eigenvalues of 2x2 inertia tensor
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy ** 2
    discriminant = max(0.0, trace ** 2 - 4.0 * det)
    lam1 = (trace + np.sqrt(discriminant)) / 2.0
    lam2 = (trace - np.sqrt(discriminant)) / 2.0

    if lam1 > 0:
        eccentricity = np.sqrt(1.0 - lam2 / lam1)
    else:
        eccentricity = 0.0

    # --- Centroid offset ---
    centroid_r = np.mean(rows)
    centroid_c = np.mean(cols)
    centroid_mag = np.sqrt((centroid_r - center_row) ** 2 + (centroid_c - center_col) ** 2)
    centroid_dir = np.arctan2(centroid_r - center_row, centroid_c - center_col)

    # --- Wavefront anisotropy (re-use existing function) ---
    distances = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
    max_dist = np.max(distances)
    wavefront_mask = distances >= (max_dist - 1.0)
    wavefront_pixels = list(zip(rows[wavefront_mask].tolist(),
                                cols[wavefront_mask].tolist()))
    anisotropy, anisotropy_dir = compute_wavefront_anisotropy(
        wavefront_pixels, center_row, center_col)

    # --- Radial sectors (8 compass directions) ---
    # Angles from stimulus centre to each blob pixel
    angles = np.arctan2(rows - center_row, cols - center_col)  # [-pi, pi]
    # Map to [0, 2*pi)
    angles = angles % (2.0 * np.pi)

    # 8 sectors, each 45 degrees, starting from East (0) going counter-clockwise
    # Sector 0: E  [337.5, 22.5)
    # Sector 1: NE [22.5, 67.5)
    # etc.
    sector_counts = np.zeros(8)
    sector_edges = np.linspace(0, 2 * np.pi, 9)  # 0, 45, 90, ..., 360 degrees
    # Shift by half-sector so that "East" is centered on 0
    shifted_angles = (angles + np.pi / 8.0) % (2.0 * np.pi)
    sector_idx = np.clip((shifted_angles / (np.pi / 4.0)).astype(int), 0, 7)
    for s in range(8):
        sector_counts[s] = np.sum(sector_idx == s)

    # Subtract stimulus-region pixels to get "excess" blob area per sector
    stim_rr, stim_cc = np.meshgrid(stim_rows, stim_cols, indexing='ij')
    stim_angles = np.arctan2(stim_rr.flatten() - center_row,
                             stim_cc.flatten() - center_col) % (2.0 * np.pi)
    shifted_stim = (stim_angles + np.pi / 8.0) % (2.0 * np.pi)
    stim_sector_idx = np.clip((shifted_stim / (np.pi / 4.0)).astype(int), 0, 7)
    for s in range(8):
        # Only subtract stim pixels that are actually in the blob
        stim_in_sector = np.sum(stim_sector_idx == s)
        sector_counts[s] = max(0.0, sector_counts[s] - stim_in_sector)

    total_excess = np.sum(sector_counts)
    if total_excess > 0:
        sectors = sector_counts / total_excess
    else:
        sectors = np.ones(8) / 8.0

    return {
        'eccentricity': float(eccentricity),
        'centroid_mag': float(centroid_mag),
        'centroid_dir': float(centroid_dir),
        'anisotropy': float(anisotropy),
        'anisotropy_dir': float(anisotropy_dir),
        'sectors': sectors,
        'gauss_blob_area': float(blob_area),
    }


def compute_connected_blob_area(frame, stim_rows, stim_cols):
    """Area of the raw connected component of active cells including the stimulus.

    No Gaussian smoothing — uses raw binary 4-connectivity (default for
    scipy_label), matching nearest-neighbour excitatory coupling.

    Returns 0 if no component overlaps the stimulus region.
    """
    binary_frame = ((frame + 1) // 2).astype(np.uint8)  # -1->0, +1->1
    labeled, num_features = scipy_label(binary_frame)
    if num_features == 0:
        return 0
    stim_labels = labeled[np.ix_(stim_rows, stim_cols)]
    label_counts = np.bincount(stim_labels.flatten())
    if len(label_counts) <= 1:
        return 0
    label_counts[0] = 0
    stim_blob_label = np.argmax(label_counts)
    if stim_blob_label == 0:
        return 0
    return int(np.sum(labeled == stim_blob_label))


def compute_connected_blob_area_net(frame, stim_rows, stim_cols):
    """Area of the connected blob EXCLUDING cells in the stimulus region.

    Same blob detection as compute_connected_blob_area, but subtracts
    cells that fall within the stimulus grid from the count.
    Measures net propagation beyond the stimulus footprint.
    """
    binary_frame = ((frame + 1) // 2).astype(np.uint8)
    labeled, num_features = scipy_label(binary_frame)
    if num_features == 0:
        return 0
    stim_labels = labeled[np.ix_(stim_rows, stim_cols)]
    label_counts = np.bincount(stim_labels.flatten())
    if len(label_counts) <= 1:
        return 0
    label_counts[0] = 0
    stim_blob_label = np.argmax(label_counts)
    if stim_blob_label == 0:
        return 0
    blob_mask = (labeled == stim_blob_label)
    stim_mask = np.zeros_like(blob_mask)
    stim_mask[np.ix_(stim_rows, stim_cols)] = True
    return int(np.sum(blob_mask & ~stim_mask))


def compute_connected_blob_area_crop(frame, stim_rows, stim_cols,
                                     crop_row_start, crop_row_end,
                                     crop_col_start, crop_col_end):
    """Area of the connected blob restricted to the FOV crop region.

    Finds the same stimulus-connected blob as compute_connected_blob_area,
    but counts only pixels within [crop_row_start:crop_row_end,
    crop_col_start:crop_col_end].  This makes the metric comparable to
    experimental Grid40 blob areas which are inherently FOV-sized.
    """
    binary_frame = ((frame + 1) // 2).astype(np.uint8)
    labeled, num_features = scipy_label(binary_frame)
    if num_features == 0:
        return 0
    stim_labels = labeled[np.ix_(stim_rows, stim_cols)]
    label_counts = np.bincount(stim_labels.flatten())
    if len(label_counts) <= 1:
        return 0
    label_counts[0] = 0
    stim_blob_label = np.argmax(label_counts)
    if stim_blob_label == 0:
        return 0
    blob_mask = (labeled == stim_blob_label)
    return int(blob_mask[crop_row_start:crop_row_end,
                         crop_col_start:crop_col_end].sum())


def compute_connected_blob_area_net_crop(frame, stim_rows, stim_cols,
                                         crop_row_start, crop_row_end,
                                         crop_col_start, crop_col_end):
    """Net blob area (excluding stimulus) restricted to the FOV crop region.

    Combines the stimulus-exclusion logic of compute_connected_blob_area_net
    with the FOV crop of compute_connected_blob_area_crop.
    """
    binary_frame = ((frame + 1) // 2).astype(np.uint8)
    labeled, num_features = scipy_label(binary_frame)
    if num_features == 0:
        return 0
    stim_labels = labeled[np.ix_(stim_rows, stim_cols)]
    label_counts = np.bincount(stim_labels.flatten())
    if len(label_counts) <= 1:
        return 0
    label_counts[0] = 0
    stim_blob_label = np.argmax(label_counts)
    if stim_blob_label == 0:
        return 0
    blob_mask = (labeled == stim_blob_label)
    stim_mask = np.zeros_like(blob_mask)
    stim_mask[np.ix_(stim_rows, stim_cols)] = True
    cropped = (blob_mask & ~stim_mask)
    return int(cropped[crop_row_start:crop_row_end,
                       crop_col_start:crop_col_end].sum())


# =============================================================================
# Snapshot Simulation
# =============================================================================

# Timepoint labels for the 5 snapshots
TIMEPOINT_LABELS = ['Pre-stimulus', 'Stim onset', 'Stim middle', 'Stim offset', 'Post-stimulus']


def compute_snapshot_indices(stim_duration):
    """Return the 5 frame indices at which to capture snapshots."""
    return [
        PRE_STIM_FRAMES - 1,                                       # Pre-stimulus
        PRE_STIM_FRAMES + 1,                                       # Stim onset (+1 frame)
        PRE_STIM_FRAMES + stim_duration // 2,                      # Stim middle
        PRE_STIM_FRAMES + stim_duration - 1,                       # Stim offset
        PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES // 3,   # Post-stimulus
    ]


def compute_detailed_snapshot_indices(stim_duration):
    """
    Return ~13 frame indices for detailed timeline figures.

    1x pre-stimulus, 10x stimulus at 10% increments, 3x post-stimulus.
    Deduplicated and sorted (short durations may collapse some percentages).
    """
    indices = set()

    # 1 pre-stimulus frame
    indices.add(PRE_STIM_FRAMES - 1)

    # 10 stimulus frames at 10%, 20%, ..., 100%
    for k in range(1, 11):
        frac = k / 10.0
        raw = PRE_STIM_FRAMES + round(frac * stim_duration)
        # Clamp to last stimulus frame at most
        clamped = min(raw, PRE_STIM_FRAMES + stim_duration - 1)
        indices.add(clamped)

    # 3 post-stimulus frames
    indices.add(PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES // 10)     # close to offset
    indices.add(PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES // 3)
    indices.add(PRE_STIM_FRAMES + stim_duration + 2 * POST_STIM_FRAMES // 3)

    return sorted(indices)


def run_perturbation_with_snapshots(params, stim_size, stim_duration, seed_offset=0,
                                    initial_state=None, snapshot_indices_override=None,
                                    extra_post_frames=0,
                                    stim_mode='clamped', stim_bias=None):
    """
    Run a perturbation and capture grid snapshots.

    Captures the union of basic (5) and detailed (13) snapshot indices
    in a single simulation pass (or a custom set if overridden).

    Parameters
    ----------
    params : dict
        Simulation parameters (beta, c, decay_const, inhibition_range, bias).
    stim_size : int
        Square stimulus side length in pixels.
    stim_duration : int
        Stimulus duration in MC sweeps.
    seed_offset : int
        Offset added to seed for independent replicates (default 0).
    initial_state : tuple of (spin_config, H), optional
        If provided, skip random init + burn-in and start from this
        equilibrium state.  Used for decorrelated multi-replicate runs.
    snapshot_indices_override : set of int, optional
        If provided, capture snapshots at exactly these frame indices
        instead of the default union of basic + detailed indices.
    extra_post_frames : int
        Additional post-stimulus frames beyond POST_STIM_FRAMES (default 0).
    stim_mode : str
        Stimulus mode: 'clamped', 'double_pulse', or 'bias'.
    stim_bias : float or None
        Additive bias for 'bias' mode (required when stim_mode='bias').

    Returns
    -------
    snapshots : dict of {int: ndarray}
        Frame index -> spin configuration copy for all captured frames.
    """
    beta = params['beta']
    c = params['c']
    decay_const = params['decay_const']
    rad = params['inhibition_range']
    bias = params['bias']

    kernel = build_diamond_kernel(rad)
    K_sum = float(np.sum(kernel))

    if initial_state is not None:
        # Use provided equilibrium state — skip random init + burn-in
        spin_config = initial_state[0].copy()
        H = initial_state[1].copy()
    else:
        # Deterministic seed (unique per condition+size+duration combo is ensured
        # by the caller iterating different params; we just fix the replicate seed)
        np.random.seed(42 + stim_size + stim_duration * 100 + seed_offset * 10000)
        spin_config = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, M))
        H = np.zeros((L, M), dtype=np.float64)

        # Burn-in
        tau = (L * M) / decay_const
        burn_in_steps = max(BURN_IN_MIN, int(BURN_IN_TAU_MULT * tau))
        for _ in range(burn_in_steps):
            spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias, K_sum)

    # Stimulus region
    center_row = L // 2
    center_col = M // 2
    stim_rows, stim_cols = get_stimulus_region(stim_size, center_row, center_col, L, M)

    # Pre-compute bias field for any bias-using mode
    if mode_uses_bias(stim_mode):
        bias_field = np.ones((L, M), dtype=np.float64) * bias
        for i in stim_rows:
            for j in stim_cols:
                bias_field[i, j] += stim_bias

    total_frames = PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES
    if snapshot_indices_override is not None:
        snapshot_indices = set(snapshot_indices_override)
    else:
        snapshot_indices = set(compute_snapshot_indices(stim_duration)) | set(compute_detailed_snapshot_indices(stim_duration))
    snapshots = {}

    frame_idx = 0

    # --- PRE-STIMULUS ---
    for _ in range(PRE_STIM_FRAMES):
        spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias, K_sum)
        if frame_idx in snapshot_indices:
            snapshots[frame_idx] = spin_config.copy()
        frame_idx += 1

    # --- STIMULUS ON ---
    for t in range(stim_duration):
        if stim_mode == 'clamped':
            spin_config = apply_stimulus(spin_config, stim_rows, stim_cols)
            spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias, K_sum)
            spin_config = apply_stimulus(spin_config, stim_rows, stim_cols)
        elif stim_mode.startswith('double_pulse_bias'):
            # Soft (additive-bias) analogue of double_pulse: bias_field at onset/offset, plain dynamics in between.
            dpb_suffix = stim_mode[len('double_pulse_bias'):]
            pulse_width = int(dpb_suffix) if dpb_suffix else 1
            if t < pulse_width or t >= stim_duration - pulse_width:
                spin_config, H = heat_bath_numba_with_bias_field(
                    spin_config, beta, c, decay_const, H, kernel, bias_field, K_sum)
            else:
                spin_config, H = heat_bath_numba(
                    spin_config, beta, c, decay_const, H, kernel, bias, K_sum)
        elif stim_mode.startswith('double_pulse'):
            # Parse clamp width: double_pulse=1, double_pulse3=3, etc.
            dp_suffix = stim_mode[len('double_pulse'):]
            clamp_width = int(dp_suffix) if dp_suffix else 1
            if t < clamp_width or t >= stim_duration - clamp_width:
                spin_config = apply_stimulus(spin_config, stim_rows, stim_cols)
            spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias, K_sum)
        elif stim_mode == 'bias':
            spin_config, H = heat_bath_numba_with_bias_field(
                spin_config, beta, c, decay_const, H, kernel, bias_field, K_sum)
        if frame_idx in snapshot_indices:
            snapshots[frame_idx] = spin_config.copy()
        frame_idx += 1

    # --- POST-STIMULUS ---
    for _ in range(POST_STIM_FRAMES + extra_post_frames):
        spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias, K_sum)
        if frame_idx in snapshot_indices:
            snapshots[frame_idx] = spin_config.copy()
        frame_idx += 1

    return snapshots


# =============================================================================
# Batch Runner
# =============================================================================

def generate_all_snapshots(best_matches, stimulus_durations, output_dir,
                           size_filter=None, duration_filter=None,
                           index=None, stim_mode='clamped', stim_bias=None):
    """
    Run 1 replicate per (condition x size x duration) and collect snapshots.

    Parameters
    ----------
    best_matches : dict
        Loaded comparison results.
    stimulus_durations : list of int
        MC sweep durations.
    output_dir : str
        Where to save snapshot_data.npz.
    size_filter, duration_filter : list or None
        Optional subsets.
    index : int or None
        If provided, run only the combo at this SLURM index (same
        enumeration as heatmap mode).  When the combo's condition is
        Expert or NoSpout, 6 example replicates for that (size, duration)
        are also run.
    stim_mode : str
        Stimulus mode: 'clamped', 'double_pulse', or 'bias'.
    stim_bias : float or None
        Additive bias for 'bias' mode.

    Returns
    -------
    snapshot_data : dict
        Keyed by (condition, size, duration) -> dict of {frame_idx: array}.
    """
    sizes = size_filter if size_filter else STIMULUS_SIZES
    durations = duration_filter if duration_filter else stimulus_durations

    combos = enumerate_heatmap_combos(sizes, durations)

    if index is not None:
        if index < 0 or index >= len(combos):
            raise ValueError(f"Index {index} out of range [0, {len(combos)-1}]")
        combos = [combos[index]]
        print(f"SLURM index {index}: running 1 snapshot combo ({combos[0]})")

    total = len(combos)
    print(f"Running {total} perturbation snapshots")

    snapshot_data = {}
    done = 0
    t0 = time.time()

    for cond, size, dur in combos:
        if cond not in best_matches:
            print(f"  Skipping {cond} (not in comparison results)")
            continue
        params = best_matches[cond]['simulations'][0]  # sim_idx=0 (best match)

        done += 1
        print(f"  [{done}/{total}] {cond}  size={size}  dur={dur} ...", end='', flush=True)
        t1 = time.time()
        snaps = run_perturbation_with_snapshots(params, size, dur,
                                                    stim_mode=stim_mode,
                                                    stim_bias=stim_bias)
        elapsed = time.time() - t1
        print(f"  {elapsed:.1f}s")
        snapshot_data[(cond, size, dur)] = snaps

        # Example replicates for Expert and NoSpout
        if cond in ('Expert', 'NoSpout'):
            n_example_reps = REPS_PER_SIZE.get(size, DEFAULT_EXAMPLE_REPS)
            for rep_idx in range(1, n_example_reps + 1):
                print(f"    {cond} rep={rep_idx}  size={size}  dur={dur} ...", end='', flush=True)
                t1 = time.time()
                snaps = run_perturbation_with_snapshots(params, size, dur,
                                                        seed_offset=rep_idx,
                                                        stim_mode=stim_mode,
                                                        stim_bias=stim_bias)
                elapsed = time.time() - t1
                print(f"  {elapsed:.1f}s")
                snapshot_data[(cond, size, dur, rep_idx)] = snaps

    total_time = time.time() - t0
    print(f"Done. Total time: {total_time:.1f}s")

    return snapshot_data


def save_snapshot_data(snapshot_data, sizes, durations, output_dir, suffix=''):
    """
    Save snapshot data to npz file.

    Parameters
    ----------
    snapshot_data : dict
        Data from generate_all_snapshots().
    sizes : list of int
        Stimulus sizes used.
    durations : list of int
        Stimulus durations used.
    output_dir : str
        Output directory.
    suffix : str
        Optional suffix for filename (e.g., '_idx42' for SLURM per-combo files).
    """
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f'snapshot_data{suffix}.npz')

    # Pack into flat dict for npz
    # 3-tuple keys: "cond_size_dur_f{frame_idx}"
    # 4-tuple keys: "Expert_size_dur_r{rep}_f{frame_idx}"
    save_dict = {}
    for tup, snaps in snapshot_data.items():
        for frame_idx, snap in snaps.items():
            if len(tup) == 4:
                cond, size, dur, rep = tup
                key = f"{cond}_{size}_{dur}_r{rep}_f{frame_idx}"
            else:
                cond, size, dur = tup
                key = f"{cond}_{size}_{dur}_f{frame_idx}"
            save_dict[key] = snap

    # Also store the lists of sizes/durations/conditions used
    save_dict['__sizes__'] = np.array(sizes)
    save_dict['__durations__'] = np.array(durations)
    # Store conditions as bytes for npz compatibility
    save_dict['__conditions__'] = np.array(CONDITIONS, dtype='U')

    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved snapshot data to {npz_path} ({os.path.getsize(npz_path) / 1e6:.1f} MB)")
    return npz_path


def load_snapshot_data(npz_path):
    """Load snapshot data from a previously saved .npz file."""
    import re
    print(f"Loading snapshot data from {npz_path}")
    raw = np.load(npz_path, allow_pickle=False)

    sizes = raw['__sizes__'].tolist()
    durations = raw['__durations__'].tolist()
    conditions = raw['__conditions__'].tolist()

    snapshot_data = {}

    # Load main 3-tuple entries (cond_size_dur_f{frame})
    for cond in conditions:
        for size in sizes:
            for dur in durations:
                prefix = f"{cond}_{size}_{dur}_f"
                snaps = {}
                for key in raw.files:
                    if key.startswith(prefix):
                        frame_idx = int(key[len(prefix):])
                        snaps[frame_idx] = raw[key]
                if snaps:
                    snapshot_data[(cond, size, dur)] = snaps

    # Load 4-tuple replicate entries (Expert_size_dur_r{rep}_f{frame})
    rep_pattern = re.compile(r'^(\w+)_(\d+)_(\d+)_r(\d+)_f(\d+)$')
    n_reps = 0
    for key in raw.files:
        m = rep_pattern.match(key)
        if m:
            cond = m.group(1)
            size = int(m.group(2))
            dur = int(m.group(3))
            rep = int(m.group(4))
            frame_idx = int(m.group(5))
            tup = (cond, size, dur, rep)
            if tup not in snapshot_data:
                snapshot_data[tup] = {}
                n_reps += 1
            snapshot_data[tup][frame_idx] = raw[key]

    n_main = sum(1 for k in snapshot_data if len(k) == 3)
    print(f"Loaded {n_main} main snapshot sets + {n_reps} replicate sets "
          f"({len(conditions)} conditions x {len(sizes)} sizes x {len(durations)} durations)")
    return snapshot_data, sizes, durations


# =============================================================================
# Heatmap + Asymmetry Batch Runner
# =============================================================================

def enumerate_heatmap_combos(sizes, durations):
    """Return flat list of (condition, size, duration) combos for SLURM indexing."""
    combos = []
    for cond in CONDITIONS:
        for size in sizes:
            for dur in durations:
                combos.append((cond, size, dur))
    return combos


def generate_heatmap_data(best_matches, stimulus_durations, output_dir,
                          n_reps=100, size_filter=None, duration_filter=None,
                          index=None, stim_mode='clamped', stim_bias=None):
    """
    Run multi-replicate heatmaps + asymmetry analysis.

    For each (condition, size, duration):
      1. Prepare n_reps decorrelated equilibrium states.
      2. For each replicate, run perturbation with snapshots.
      3. Accumulate probability sum (running average, not stored per-rep).
      4. Compute asymmetry metrics at each snapshot timepoint.

    Parameters
    ----------
    best_matches : dict
        Loaded comparison results.
    stimulus_durations : list of int
        MC sweep durations.
    output_dir : str
        Where to save results.
    n_reps : int
        Number of replicates per combo (default 100).
    size_filter, duration_filter : list or None
        Optional subsets.
    index : int or None
        If provided, run only the combo at this SLURM index.

    Returns
    -------
    heatmap_data : dict
        Contains probability heatmaps and asymmetry metrics.
    """
    sizes = size_filter if size_filter else STIMULUS_SIZES
    durations = duration_filter if duration_filter else stimulus_durations
    combos = enumerate_heatmap_combos(sizes, durations)

    if index is not None:
        if index < 0 or index >= len(combos):
            raise ValueError(f"Index {index} out of range [0, {len(combos)-1}]")
        combos = [combos[index]]
        print(f"SLURM index {index}: running 1 combo ({combos[0]})")

    total = len(combos)
    print(f"Running heatmap analysis: {total} combos x {n_reps} replicates")

    center_row = L // 2
    center_col = M // 2

    heatmap_data = {
        '__sizes__': sizes,
        '__durations__': durations,
        '__conditions__': CONDITIONS,
        '__n_reps__': n_reps,
    }

    t0 = time.time()

    for combo_idx, (cond, size, dur) in enumerate(combos):
        if cond not in best_matches:
            print(f"  Skipping {cond} (not in comparison results)")
            continue

        params = best_matches[cond]['simulations'][0]  # sim_idx=0 (best match)
        stim_rows, stim_cols = get_stimulus_region(size, center_row, center_col, L, M)

        # Compute snapshot frame indices for this duration
        detailed_indices = compute_detailed_snapshot_indices(dur)
        n_timepoints = len(detailed_indices)

        # Store frame indices per duration
        dur_frame_key = f'__frame_indices_{dur}__'
        if dur_frame_key not in heatmap_data:
            heatmap_data[dur_frame_key] = np.array(detailed_indices, dtype=np.int32)

        print(f"  [{combo_idx+1}/{total}] {cond} size={size} dur={dur} "
              f"({n_timepoints} timepoints) ...", flush=True)

        # Prepare decorrelated equilibrium states
        mode_offset = STIM_MODE_SEED_OFFSET.get(stim_mode, 0)
        seed = (hash((cond, size, dur)) + mode_offset * 100000) % (2**31)
        t1 = time.time()
        eq_states = prepare_decorrelated_states(params, n_reps, seed)
        print(f"    Prepared {n_reps} equilibrium states in {time.time()-t1:.1f}s")

        # Running probability sum per timepoint (memory-efficient)
        prob_sums = {fidx: np.zeros((L, M), dtype=np.float64) for fidx in detailed_indices}

        # Asymmetry metric arrays [n_reps, n_timepoints]
        eccentricity = np.full((n_reps, n_timepoints), np.nan)
        centroid_mag = np.full((n_reps, n_timepoints), np.nan)
        centroid_dir = np.full((n_reps, n_timepoints), np.nan)
        aniso = np.full((n_reps, n_timepoints), np.nan)
        aniso_dir = np.full((n_reps, n_timepoints), np.nan)
        gauss_blob_area = np.full((n_reps, n_timepoints), np.nan)
        sectors = np.full((n_reps, n_timepoints, 8), np.nan)

        # Spaghetti: every frame from 100 pre-stim through 500 post-stim
        spaghetti_pre = PRE_STIM_FRAMES
        spaghetti_extra_post = 200
        spaghetti_start = PRE_STIM_FRAMES - spaghetti_pre
        spaghetti_end = PRE_STIM_FRAMES + dur + POST_STIM_FRAMES + spaghetti_extra_post
        spaghetti_indices = list(range(spaghetti_start, spaghetti_end))
        n_spaghetti = len(spaghetti_indices)

        # Store spaghetti frame indices per duration
        spag_key = f'__spaghetti_indices_{dur}__'
        if spag_key not in heatmap_data:
            heatmap_data[spag_key] = np.array(spaghetti_indices, dtype=np.int32)

        # Per-replicate fraction-active traces [n_reps, n_spaghetti]
        frac_full = np.full((n_reps, n_spaghetti), np.nan)
        frac_half = np.full((n_reps, n_spaghetti), np.nan)
        frac_crop = np.full((n_reps, n_spaghetti), np.nan)

        # Per-replicate connected blob area [n_reps, n_spaghetti]
        blob_area = np.full((n_reps, n_spaghetti), np.nan)
        blob_area_net = np.full((n_reps, n_spaghetti), np.nan)
        blob_area_crop = np.full((n_reps, n_spaghetti), np.nan)
        blob_area_net_crop = np.full((n_reps, n_spaghetti), np.nan)
        peak_blob_size = np.full(n_reps, np.nan)
        peak_blob_size_crop = np.full(n_reps, np.nan)
        peak_prob_sum = np.zeros((L, M), dtype=np.float64)
        peak_prob_count = 0

        # Combined snapshot indices: detailed (for heatmap/asymmetry) + spaghetti (for fraction)
        all_snapshot_indices = set(detailed_indices) | set(spaghetti_indices)

        t2 = time.time()
        for rep in range(n_reps):
            # Run simulation capturing both detailed and spaghetti frames
            snaps = run_perturbation_with_snapshots(
                params, size, dur,
                initial_state=eq_states[rep],
                snapshot_indices_override=all_snapshot_indices,
                extra_post_frames=spaghetti_extra_post,
                stim_mode=stim_mode, stim_bias=stim_bias)

            # --- Detailed indices: heatmap probability + asymmetry ---
            for tp_idx, fidx in enumerate(detailed_indices):
                if fidx not in snaps:
                    continue
                frame = snaps[fidx]

                # Accumulate probability: convert -1/+1 to 0/1
                prob_sums[fidx] += (frame.astype(np.float64) + 1.0) / 2.0

                # Compute asymmetry metrics
                metrics = compute_blob_asymmetry_metrics(
                    frame, stim_rows, stim_cols, center_row, center_col)
                eccentricity[rep, tp_idx] = metrics['eccentricity']
                centroid_mag[rep, tp_idx] = metrics['centroid_mag']
                centroid_dir[rep, tp_idx] = metrics['centroid_dir']
                aniso[rep, tp_idx] = metrics['anisotropy']
                aniso_dir[rep, tp_idx] = metrics['anisotropy_dir']
                gauss_blob_area[rep, tp_idx] = metrics['gauss_blob_area']
                sectors[rep, tp_idx, :] = metrics['sectors']

            # --- Spaghetti indices: fraction-active (3 crop levels) ---
            for sp_idx, fidx in enumerate(spaghetti_indices):
                if fidx not in snaps:
                    continue
                frame = snaps[fidx]
                binary = (frame.astype(np.float64) + 1.0) / 2.0
                frac_full[rep, sp_idx] = binary.mean()
                frac_half[rep, sp_idx] = binary[HALF_CROP_ROW_START:HALF_CROP_ROW_END,
                                                HALF_CROP_COL_START:HALF_CROP_COL_END].mean()
                frac_crop[rep, sp_idx] = binary[CROP_ROW_START:CROP_ROW_END,
                                                CROP_COL_START:CROP_COL_END].mean()
                blob_area[rep, sp_idx] = compute_connected_blob_area(
                    frame, stim_rows, stim_cols)
                blob_area_net[rep, sp_idx] = compute_connected_blob_area_net(
                    frame, stim_rows, stim_cols)
                blob_area_crop[rep, sp_idx] = compute_connected_blob_area_crop(
                    frame, stim_rows, stim_cols,
                    CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END)
                blob_area_net_crop[rep, sp_idx] = compute_connected_blob_area_net_crop(
                    frame, stim_rows, stim_cols,
                    CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END)

            # Find peak blob frame during stimulus period for this replicate
            stim_sp_mask = (np.array(spaghetti_indices) >= PRE_STIM_FRAMES) & \
                           (np.array(spaghetti_indices) < PRE_STIM_FRAMES + dur)
            rep_blob = blob_area[rep, :].copy()
            rep_blob[~stim_sp_mask] = -1
            peak_sp_idx = int(np.argmax(rep_blob))
            peak_fidx = spaghetti_indices[peak_sp_idx]
            peak_blob_size[rep] = blob_area[rep, peak_sp_idx]
            peak_blob_size_crop[rep] = blob_area_crop[rep, peak_sp_idx]

            if peak_fidx in snaps:
                peak_prob_sum += (snaps[peak_fidx].astype(np.float64) + 1.0) / 2.0
                peak_prob_count += 1

            if (rep + 1) % 25 == 0 or rep == n_reps - 1:
                elapsed_reps = time.time() - t2
                print(f"    Replicate {rep+1}/{n_reps} ({elapsed_reps:.1f}s)")

        # Divide sums to get probability heatmaps
        for fidx in detailed_indices:
            prob_map = (prob_sums[fidx] / n_reps).astype(np.float32)
            heatmap_data[f'{cond}_{size}_{dur}_f{fidx}'] = prob_map

        # Store asymmetry metrics
        prefix = f'{cond}_{size}_{dur}'
        heatmap_data[f'{prefix}_eccentricity'] = eccentricity.astype(np.float32)
        heatmap_data[f'{prefix}_centroid_mag'] = centroid_mag.astype(np.float32)
        heatmap_data[f'{prefix}_centroid_dir'] = centroid_dir.astype(np.float32)
        heatmap_data[f'{prefix}_anisotropy'] = aniso.astype(np.float32)
        heatmap_data[f'{prefix}_anisotropy_dir'] = aniso_dir.astype(np.float32)
        heatmap_data[f'{prefix}_sectors'] = sectors.astype(np.float32)
        heatmap_data[f'{prefix}_gauss_blob_area'] = gauss_blob_area.astype(np.float32)
        heatmap_data[f'{prefix}_frac_full'] = frac_full.astype(np.float32)
        heatmap_data[f'{prefix}_frac_half'] = frac_half.astype(np.float32)
        heatmap_data[f'{prefix}_frac_crop'] = frac_crop.astype(np.float32)
        heatmap_data[f'{prefix}_blob_area'] = blob_area.astype(np.float32)
        heatmap_data[f'{prefix}_blob_area_net'] = blob_area_net.astype(np.float32)
        heatmap_data[f'{prefix}_blob_area_crop'] = blob_area_crop.astype(np.float32)
        heatmap_data[f'{prefix}_blob_area_net_crop'] = blob_area_net_crop.astype(np.float32)
        heatmap_data[f'{prefix}_peak_blob_size'] = peak_blob_size.astype(np.float32)
        heatmap_data[f'{prefix}_peak_blob_size_crop'] = peak_blob_size_crop.astype(np.float32)
        if peak_prob_count > 0:
            heatmap_data[f'{prefix}_peak_blob_prob'] = (peak_prob_sum / peak_prob_count).astype(np.float32)

        elapsed_combo = time.time() - t1
        print(f"    Done in {elapsed_combo:.1f}s")

    total_time = time.time() - t0
    print(f"Heatmap analysis complete. Total time: {total_time:.1f}s")

    return heatmap_data


def save_heatmap_data(heatmap_data, output_dir, suffix=''):
    """
    Save heatmap + asymmetry data to npz file.

    Parameters
    ----------
    heatmap_data : dict
        Data from generate_heatmap_data().
    output_dir : str
        Output directory.
    suffix : str
        Optional suffix for filename (e.g., '_idx42' for SLURM per-combo files).
    """
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f'heatmap_data{suffix}.npz')

    # Convert non-array metadata to arrays for npz
    save_dict = {}
    for key, val in heatmap_data.items():
        if isinstance(val, (list, tuple)):
            if all(isinstance(v, str) for v in val):
                save_dict[key] = np.array(val, dtype='U')
            else:
                save_dict[key] = np.array(val)
        elif isinstance(val, (int, float)):
            save_dict[key] = np.array(val)
        else:
            save_dict[key] = val

    np.savez_compressed(npz_path, **save_dict)
    size_mb = os.path.getsize(npz_path) / 1e6
    print(f"Saved heatmap data to {npz_path} ({size_mb:.1f} MB)")
    return npz_path


def load_heatmap_data(npz_path):
    """
    Load heatmap + asymmetry data from npz file.

    Parameters
    ----------
    npz_path : str
        Path to heatmap_data.npz.

    Returns
    -------
    heatmap_data : dict
        Reconstructed data dict.
    """
    print(f"Loading heatmap data from {npz_path}")
    raw = np.load(npz_path, allow_pickle=False)

    heatmap_data = {}
    for key in raw.files:
        val = raw[key]
        # Restore scalar metadata
        if key == '__n_reps__':
            heatmap_data[key] = int(val)
        elif key in ('__sizes__', '__durations__'):
            heatmap_data[key] = val.tolist()
        elif key == '__conditions__':
            heatmap_data[key] = val.tolist()
        else:
            heatmap_data[key] = val

    sizes = heatmap_data.get('__sizes__', [])
    durations = heatmap_data.get('__durations__', [])
    n_reps = heatmap_data.get('__n_reps__', 0)
    print(f"  Loaded: {len(sizes)} sizes, {len(durations)} durations, {n_reps} reps")
    return heatmap_data


def combine_heatmap_files(output_dir):
    """
    Combine per-combo heatmap npz files into a single heatmap_data.npz.

    Looks for files matching heatmap_data_idx*.npz in output_dir.
    """
    from glob import glob as globfn
    pattern = os.path.join(output_dir, 'heatmap_data_idx*.npz')
    files = sorted(globfn(pattern))
    if not files:
        print(f"No per-combo heatmap files found matching {pattern}")
        return

    print(f"Combining {len(files)} per-combo heatmap files...")

    combined = {}
    for fpath in files:
        raw = np.load(fpath, allow_pickle=False)
        for key in raw.files:
            if key.startswith('__') and key in combined:
                continue  # Metadata already captured
            combined[key] = raw[key]

    # Save combined
    npz_path = os.path.join(output_dir, 'heatmap_data.npz')
    np.savez_compressed(npz_path, **combined)
    size_mb = os.path.getsize(npz_path) / 1e6
    print(f"Saved combined heatmap data to {npz_path} ({size_mb:.1f} MB)")


def combine_snapshot_files(output_dir):
    """
    Combine per-combo snapshot npz files into a single snapshot_data.npz.

    Looks for files matching snapshot_data_idx*.npz in output_dir.
    No-op if no per-combo files are found.
    """
    from glob import glob as globfn
    pattern = os.path.join(output_dir, 'snapshot_data_idx*.npz')
    files = sorted(globfn(pattern))
    if not files:
        print(f"No per-combo snapshot files found matching {pattern}")
        return

    print(f"Combining {len(files)} per-combo snapshot files...")

    combined = {}
    for fpath in files:
        raw = np.load(fpath, allow_pickle=False)
        for key in raw.files:
            if key.startswith('__') and key in combined:
                continue  # Metadata already captured
            combined[key] = raw[key]

    # Save combined
    npz_path = os.path.join(output_dir, 'snapshot_data.npz')
    np.savez_compressed(npz_path, **combined)
    size_mb = os.path.getsize(npz_path) / 1e6
    print(f"Saved combined snapshot data to {npz_path} ({size_mb:.1f} MB)")


# =============================================================================
# Plotting
# =============================================================================

# Condition colours (matching Figure5_IsingPerturbationAnalysis.m:41-44)
EXPERT_NOSPOUT = ['Expert', 'NoSpout']

CONDITION_COLORS = {
    'Naive':    (0.3373, 0.7059, 0.9137),
    'Beginner': (0.8431, 0.2549, 0.6078),
    'Expert':   (0.0000, 0.6196, 0.4510),
    'NoSpout':  (0.8353, 0.3686, 0.0000),
}


def plot_snapshot_figure(snapshot_data, stim_size, stim_duration, output_dir,
                         stim_mode_label='clamped'):
    """
    Generate a single figure for one (size, duration) combination.

    Layout: 8 rows x 5 columns
        Rows 0,1 = Naive    (full grid, centre crop)
        Rows 2,3 = Beginner (full grid, centre crop)
        Rows 4,5 = Expert   (full grid, centre crop)
        Rows 6,7 = NoSpout  (full grid, centre crop)
        Columns  = 5 timepoints
    """
    cmap_sim = ListedColormap(['white', 'black'])

    # Stimulus region coordinates (for overlay rectangles)
    center_row = L // 2
    center_col = M // 2
    stim_rows, stim_cols = get_stimulus_region(stim_size, center_row, center_col, L, M)
    stim_r0, stim_r1 = int(stim_rows[0]), int(stim_rows[-1])
    stim_c0, stim_c1 = int(stim_cols[0]), int(stim_cols[-1])

    # Frame index labels
    snap_indices = compute_snapshot_indices(stim_duration)

    fig, axes = plt.subplots(8, 5, figsize=(14, 18))

    for cond_idx, cond in enumerate(CONDITIONS):
        key = (cond, stim_size, stim_duration)
        if key not in snapshot_data:
            continue
        snaps_dict = snapshot_data[key]
        color = CONDITION_COLORS[cond]

        row_full = cond_idx * 2
        row_crop = cond_idx * 2 + 1

        for tp_idx in range(5):
            frame = snaps_dict[snap_indices[tp_idx]]
            binary = (frame.astype(np.float64) + 1.0) / 2.0  # -1/+1 -> 0/1

            # --- Full grid panel ---
            ax = axes[row_full, tp_idx]
            ax.imshow(binary, cmap=cmap_sim, interpolation='nearest',
                      origin='upper', aspect='equal', vmin=0, vmax=1)

            # Centre-crop rectangle (gray dashed)
            rect_crop = Rectangle(
                (CROP_COL_START - 0.5, CROP_ROW_START - 0.5),
                CROP_COL_END - CROP_COL_START,
                CROP_ROW_END - CROP_ROW_START,
                linewidth=1.0, edgecolor='gray', facecolor='none', linestyle='--')
            ax.add_patch(rect_crop)

            # Stimulus rectangle (red dashed)
            rect_stim = Rectangle(
                (stim_c0 - 0.5, stim_r0 - 0.5),
                stim_c1 - stim_c0 + 1,
                stim_r1 - stim_r0 + 1,
                linewidth=1.2, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect_stim)

            ax.set_xticks([])
            ax.set_yticks([])

            # Row label (only leftmost column)
            if tp_idx == 0:
                ax.set_ylabel(f'{cond}\n(full)', fontsize=9, fontweight='bold',
                              color=color, rotation=0, labelpad=50, va='center')

            # Column header (only top row)
            if cond_idx == 0:
                label = TIMEPOINT_LABELS[tp_idx]
                onset_rel = snap_indices[tp_idx] - PRE_STIM_FRAMES
                ax.set_title(f'{label}\n{onset_rel:+d}', fontsize=8)

            # --- Centre-crop panel ---
            ax_c = axes[row_crop, tp_idx]
            crop = binary[CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
            ax_c.imshow(crop, cmap=cmap_sim, interpolation='nearest',
                        origin='upper', aspect='equal', vmin=0, vmax=1)

            # Stimulus rectangle in crop coordinates
            crop_stim_r0 = stim_r0 - CROP_ROW_START
            crop_stim_c0 = stim_c0 - CROP_COL_START
            crop_stim_h = stim_r1 - stim_r0 + 1
            crop_stim_w = stim_c1 - stim_c0 + 1

            # Only draw if stimulus overlaps the crop region
            if (stim_r1 >= CROP_ROW_START and stim_r0 < CROP_ROW_END and
                    stim_c1 >= CROP_COL_START and stim_c0 < CROP_COL_END):
                rect_stim_crop = Rectangle(
                    (crop_stim_c0 - 0.5, crop_stim_r0 - 0.5),
                    crop_stim_w, crop_stim_h,
                    linewidth=1.2, edgecolor='red', facecolor='none', linestyle='--')
                ax_c.add_patch(rect_stim_crop)

            ax_c.set_xticks([])
            ax_c.set_yticks([])

            if tp_idx == 0:
                ax_c.set_ylabel(f'{cond}\n(crop)', fontsize=9, fontweight='bold',
                                color=color, rotation=0, labelpad=50, va='center')

    fig.suptitle(f'Perturbation Snapshots — size={stim_size}, duration={stim_duration} MC sweeps ({stim_mode_label})',
                 fontsize=12, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0.08, 0.0, 1.0, 0.98])

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Snapshots'),
                 f'snapshot_size_{stim_size}_{stim_mode_label}')


def plot_detailed_snapshot_figure(snapshot_data, stim_size, stim_duration, output_dir,
                                  stim_mode_label='clamped'):
    """
    Generate a detailed timeline figure for one (size, duration) combination.

    Layout: 8 rows x N columns (N = number of unique detailed indices, typically 13)
        Rows: 4 conditions x 2 views (full grid, centre crop)
        Columns: pre-stim, 10%..100% stim, post x2
    """
    cmap_sim = ListedColormap(['white', 'black'])

    # Stimulus region coordinates
    center_row = L // 2
    center_col = M // 2
    stim_rows, stim_cols = get_stimulus_region(stim_size, center_row, center_col, L, M)
    stim_r0, stim_r1 = int(stim_rows[0]), int(stim_rows[-1])
    stim_c0, stim_c1 = int(stim_cols[0]), int(stim_cols[-1])

    # Get the detailed frame indices (sorted, deduplicated)
    detailed_indices = compute_detailed_snapshot_indices(stim_duration)
    n_cols = len(detailed_indices)

    # Build column labels
    stim_start = PRE_STIM_FRAMES
    stim_end = PRE_STIM_FRAMES + stim_duration - 1
    col_labels = []
    for fidx in detailed_indices:
        if fidx < stim_start:
            col_labels.append('Pre')
        elif fidx > stim_end:
            col_labels.append('Post')
        else:
            # Stimulus: compute percentage
            pct = round(100.0 * (fidx - stim_start) / stim_duration)
            pct = min(pct, 100)
            col_labels.append(f'{pct}%')

    fig, axes = plt.subplots(8, n_cols, figsize=(2.6 * n_cols, 18))

    for cond_idx, cond in enumerate(CONDITIONS):
        key = (cond, stim_size, stim_duration)
        if key not in snapshot_data:
            continue
        snaps_dict = snapshot_data[key]
        color = CONDITION_COLORS[cond]

        row_full = cond_idx * 2
        row_crop = cond_idx * 2 + 1

        for col_idx, fidx in enumerate(detailed_indices):
            if fidx not in snaps_dict:
                continue
            frame = snaps_dict[fidx]
            binary = (frame.astype(np.float64) + 1.0) / 2.0

            # --- Full grid panel ---
            ax = axes[row_full, col_idx]
            ax.imshow(binary, cmap=cmap_sim, interpolation='nearest',
                      origin='upper', aspect='equal', vmin=0, vmax=1)

            # Centre-crop rectangle (gray dashed)
            rect_crop = Rectangle(
                (CROP_COL_START - 0.5, CROP_ROW_START - 0.5),
                CROP_COL_END - CROP_COL_START,
                CROP_ROW_END - CROP_ROW_START,
                linewidth=1.0, edgecolor='gray', facecolor='none', linestyle='--')
            ax.add_patch(rect_crop)

            # Stimulus rectangle (red dashed)
            rect_stim = Rectangle(
                (stim_c0 - 0.5, stim_r0 - 0.5),
                stim_c1 - stim_c0 + 1,
                stim_r1 - stim_r0 + 1,
                linewidth=1.2, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect_stim)

            ax.set_xticks([])
            ax.set_yticks([])

            # Row label (only leftmost column)
            if col_idx == 0:
                ax.set_ylabel(f'{cond}\n(full)', fontsize=9, fontweight='bold',
                              color=color, rotation=0, labelpad=50, va='center')

            # Column header (only top row)
            if cond_idx == 0:
                onset_rel = fidx - PRE_STIM_FRAMES
                ax.set_title(f'{col_labels[col_idx]}\n{onset_rel:+d}', fontsize=7)

            # --- Centre-crop panel ---
            ax_c = axes[row_crop, col_idx]
            crop = binary[CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
            ax_c.imshow(crop, cmap=cmap_sim, interpolation='nearest',
                        origin='upper', aspect='equal', vmin=0, vmax=1)

            # Stimulus rectangle in crop coordinates
            crop_stim_r0 = stim_r0 - CROP_ROW_START
            crop_stim_c0 = stim_c0 - CROP_COL_START
            crop_stim_h = stim_r1 - stim_r0 + 1
            crop_stim_w = stim_c1 - stim_c0 + 1

            if (stim_r1 >= CROP_ROW_START and stim_r0 < CROP_ROW_END and
                    stim_c1 >= CROP_COL_START and stim_c0 < CROP_COL_END):
                rect_stim_crop = Rectangle(
                    (crop_stim_c0 - 0.5, crop_stim_r0 - 0.5),
                    crop_stim_w, crop_stim_h,
                    linewidth=1.2, edgecolor='red', facecolor='none', linestyle='--')
                ax_c.add_patch(rect_stim_crop)

            ax_c.set_xticks([])
            ax_c.set_yticks([])

            if col_idx == 0:
                ax_c.set_ylabel(f'{cond}\n(crop)', fontsize=9, fontweight='bold',
                                color=color, rotation=0, labelpad=50, va='center')

    fig.suptitle(f'Detailed Timeline — size={stim_size}, duration={stim_duration} MC sweeps ({stim_mode_label})',
                 fontsize=12, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0.06, 0.0, 1.0, 0.98])

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Snapshots'),
                 f'snapshot_detail_size_{stim_size}_{stim_mode_label}')


def plot_expert_examples_figure(snapshot_data, stim_size, stim_duration, output_dir,
                                stim_mode_label='clamped'):
    """
    Generate a figure showing 6 independent Expert replicates.

    Layout: 12 rows x N columns (same detailed timeline as plot_detailed_snapshot_figure)
        Rows: 6 replicates x 2 views (full grid, centre crop)
        Columns: pre-stim, 10%..100% stim, post x2
    """
    cmap_sim = ListedColormap(['white', 'black'])

    # Stimulus region coordinates
    center_row = L // 2
    center_col = M // 2
    stim_rows, stim_cols = get_stimulus_region(stim_size, center_row, center_col, L, M)
    stim_r0, stim_r1 = int(stim_rows[0]), int(stim_rows[-1])
    stim_c0, stim_c1 = int(stim_cols[0]), int(stim_cols[-1])

    # Get the detailed frame indices (sorted, deduplicated)
    detailed_indices = compute_detailed_snapshot_indices(stim_duration)
    n_cols = len(detailed_indices)

    # Build column labels
    stim_start = PRE_STIM_FRAMES
    stim_end = PRE_STIM_FRAMES + stim_duration - 1
    col_labels = []
    for fidx in detailed_indices:
        if fidx < stim_start:
            col_labels.append('Pre')
        elif fidx > stim_end:
            col_labels.append('Post')
        else:
            pct = round(100.0 * (fidx - stim_start) / stim_duration)
            pct = min(pct, 100)
            col_labels.append(f'{pct}%')

    N_REPS = REPS_PER_SIZE.get(stim_size, DEFAULT_EXAMPLE_REPS)
    color = CONDITION_COLORS['Expert']

    fig, axes = plt.subplots(N_REPS * 2, n_cols, figsize=(2.6 * n_cols, N_REPS * 2 * 2.27))

    for rep_idx in range(1, N_REPS + 1):
        key = ('Expert', stim_size, stim_duration, rep_idx)
        if key not in snapshot_data:
            continue
        snaps_dict = snapshot_data[key]

        row_full = (rep_idx - 1) * 2
        row_crop = (rep_idx - 1) * 2 + 1

        for col_idx, fidx in enumerate(detailed_indices):
            if fidx not in snaps_dict:
                continue
            frame = snaps_dict[fidx]
            binary = (frame.astype(np.float64) + 1.0) / 2.0

            # --- Full grid panel ---
            ax = axes[row_full, col_idx]
            ax.imshow(binary, cmap=cmap_sim, interpolation='nearest',
                      origin='upper', aspect='equal', vmin=0, vmax=1)

            # Centre-crop rectangle (gray dashed)
            rect_crop = Rectangle(
                (CROP_COL_START - 0.5, CROP_ROW_START - 0.5),
                CROP_COL_END - CROP_COL_START,
                CROP_ROW_END - CROP_ROW_START,
                linewidth=1.0, edgecolor='gray', facecolor='none', linestyle='--')
            ax.add_patch(rect_crop)

            # Stimulus rectangle (red dashed)
            rect_stim = Rectangle(
                (stim_c0 - 0.5, stim_r0 - 0.5),
                stim_c1 - stim_c0 + 1,
                stim_r1 - stim_r0 + 1,
                linewidth=1.2, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect_stim)

            ax.set_xticks([])
            ax.set_yticks([])

            # Row label (only leftmost column)
            if col_idx == 0:
                ax.set_ylabel(f'Rep {rep_idx}\n(full)', fontsize=9, fontweight='bold',
                              color=color, rotation=0, labelpad=50, va='center')

            # Column header (only top row)
            if rep_idx == 1:
                onset_rel = fidx - PRE_STIM_FRAMES
                ax.set_title(f'{col_labels[col_idx]}\n{onset_rel:+d}', fontsize=7)

            # --- Centre-crop panel ---
            ax_c = axes[row_crop, col_idx]
            crop = binary[CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
            ax_c.imshow(crop, cmap=cmap_sim, interpolation='nearest',
                        origin='upper', aspect='equal', vmin=0, vmax=1)

            # Stimulus rectangle in crop coordinates
            crop_stim_r0 = stim_r0 - CROP_ROW_START
            crop_stim_c0 = stim_c0 - CROP_COL_START
            crop_stim_h = stim_r1 - stim_r0 + 1
            crop_stim_w = stim_c1 - stim_c0 + 1

            if (stim_r1 >= CROP_ROW_START and stim_r0 < CROP_ROW_END and
                    stim_c1 >= CROP_COL_START and stim_c0 < CROP_COL_END):
                rect_stim_crop = Rectangle(
                    (crop_stim_c0 - 0.5, crop_stim_r0 - 0.5),
                    crop_stim_w, crop_stim_h,
                    linewidth=1.2, edgecolor='red', facecolor='none', linestyle='--')
                ax_c.add_patch(rect_stim_crop)

            ax_c.set_xticks([])
            ax_c.set_yticks([])

            if col_idx == 0:
                ax_c.set_ylabel(f'Rep {rep_idx}\n(crop)', fontsize=9, fontweight='bold',
                                color=color, rotation=0, labelpad=50, va='center')

    fig.suptitle(f'Expert Replicates — size={stim_size}, duration={stim_duration} MC sweeps ({stim_mode_label})',
                 fontsize=12, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0.06, 0.0, 1.0, 0.98])

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Snapshots'),
                 f'snapshot_expert_examples_size_{stim_size}_{stim_mode_label}')


def plot_expert_nospout_examples_figure(snapshot_data, stim_size, stim_duration, output_dir,
                                        stim_mode_label='clamped'):
    """
    Generate 3 figures showing Expert + NoSpout replicates side by side.

    Each figure has 4 rows x N columns:
        Rows 0-1: Expert rep pair
        Rows 2-3: NoSpout rep pair
    Single "halfway zoom" view per row (between full grid and centre crop).

    Figure 1 uses reps 1,2; Figure 2 uses reps 3,4; Figure 3 uses reps 5,6.
    """
    cmap_sim = ListedColormap(['white', 'black'])

    # Stimulus region coordinates (full-grid space)
    center_row = L // 2
    center_col = M // 2
    stim_rows, stim_cols = get_stimulus_region(stim_size, center_row, center_col, L, M)
    stim_r0, stim_r1 = int(stim_rows[0]), int(stim_rows[-1])
    stim_c0, stim_c1 = int(stim_cols[0]), int(stim_cols[-1])

    # Halfway crop bounds: half the margins of the full-to-centre-crop
    # Full grid: (39, 78), centre crop: [13:26, 26:52]
    # Half margins: top~6, bottom~7, left=13, right=13
    HALF_ROW_START = 6
    HALF_ROW_END = 33    # 6 + 27 = 33
    HALF_COL_START = 13
    HALF_COL_END = 65    # 13 + 52 = 65

    # Stimulus rectangle in halfway-crop coordinates
    half_stim_r0 = stim_r0 - HALF_ROW_START
    half_stim_c0 = stim_c0 - HALF_COL_START
    half_stim_h = stim_r1 - stim_r0 + 1
    half_stim_w = stim_c1 - stim_c0 + 1

    # Detailed frame indices and column labels
    detailed_indices = compute_detailed_snapshot_indices(stim_duration)
    n_cols = len(detailed_indices)

    stim_start = PRE_STIM_FRAMES
    stim_end = PRE_STIM_FRAMES + stim_duration - 1
    col_labels = []
    for fidx in detailed_indices:
        if fidx < stim_start:
            col_labels.append('Pre')
        elif fidx > stim_end:
            col_labels.append('Post')
        else:
            pct = round(100.0 * (fidx - stim_start) / stim_duration)
            pct = min(pct, 100)
            col_labels.append(f'{pct}%')

    # 3 figures, each with a unique pair of rep indices
    for fig_num in range(1, 4):
        rep_a = (fig_num - 1) * 2 + 1   # 1, 3, 5
        rep_b = (fig_num - 1) * 2 + 2   # 2, 4, 6

        fig, axes = plt.subplots(4, n_cols, figsize=(2.6 * n_cols, 10))

        row_configs = [
            ('Expert', rep_a, 0),
            ('Expert', rep_b, 1),
            ('NoSpout', rep_a, 2),
            ('NoSpout', rep_b, 3),
        ]

        for cond, rep_idx, row in row_configs:
            color = CONDITION_COLORS[cond]
            key = (cond, stim_size, stim_duration, rep_idx)
            if key not in snapshot_data:
                continue
            snaps_dict = snapshot_data[key]

            # Label: "Expert 1" / "NoSpout 2" etc. (local numbering within figure)
            local_num = 1 if rep_idx == rep_a else 2
            label = f'{cond} {local_num}'

            for col_idx, fidx in enumerate(detailed_indices):
                if fidx not in snaps_dict:
                    continue
                frame = snaps_dict[fidx]
                binary = (frame.astype(np.float64) + 1.0) / 2.0

                # Halfway crop
                crop = binary[HALF_ROW_START:HALF_ROW_END, HALF_COL_START:HALF_COL_END]

                ax = axes[row, col_idx]
                ax.imshow(crop, cmap=cmap_sim, interpolation='nearest',
                          origin='upper', aspect='equal', vmin=0, vmax=1)

                # Stimulus rectangle in halfway-crop coordinates
                if (stim_r1 >= HALF_ROW_START and stim_r0 < HALF_ROW_END and
                        stim_c1 >= HALF_COL_START and stim_c0 < HALF_COL_END):
                    rect_stim = Rectangle(
                        (half_stim_c0 - 0.5, half_stim_r0 - 0.5),
                        half_stim_w, half_stim_h,
                        linewidth=1.2, edgecolor='red', facecolor='none', linestyle='--')
                    ax.add_patch(rect_stim)

                ax.set_xticks([])
                ax.set_yticks([])

                # Row label (leftmost column only)
                if col_idx == 0:
                    ax.set_ylabel(label, fontsize=9, fontweight='bold',
                                  color=color, rotation=0, labelpad=50, va='center')

                # Column header (top row only)
                if row == 0:
                    onset_rel = fidx - PRE_STIM_FRAMES
                    ax.set_title(f'{col_labels[col_idx]}\n{onset_rel:+d}', fontsize=7)

        fig.suptitle(f'Expert + NoSpout Replicates (set {fig_num}) — '
                     f'size={stim_size}, dur={stim_duration} MC sweeps ({stim_mode_label})',
                     fontsize=11, fontweight='bold', y=0.995)
        fig.tight_layout(rect=[0.06, 0.0, 1.0, 0.98])

        _save_figure(fig, output_dir,
                     ('PerturbationSnapshots', f'dur_{stim_duration}', 'Snapshots'),
                     f'snapshot_expert_nospout_examples_{fig_num}_size_{stim_size}_{stim_mode_label}')


def plot_single_replicate_figures(snapshot_data, stim_size, stim_duration, output_dir,
                                   stim_mode_label='clamped'):
    """Generate individual single-row timeline figures for Expert and NoSpout replicates.

    Produces up to 10 figures per condition × 2 crop levels (halfway + FOV).
    """
    cmap_sim = ListedColormap(['white', 'black'])

    center_row = L // 2
    center_col = M // 2
    stim_rows, stim_cols = get_stimulus_region(stim_size, center_row, center_col, L, M)
    stim_r0, stim_r1 = int(stim_rows[0]), int(stim_rows[-1])
    stim_c0, stim_c1 = int(stim_cols[0]), int(stim_cols[-1])

    # Halfway crop bounds (same as plot_expert_nospout_examples_figure)
    HALF_ROW_START, HALF_ROW_END = 6, 33
    HALF_COL_START, HALF_COL_END = 13, 65

    # Crop configurations: (label, row_start, row_end, col_start, col_end, filename_tag)
    crops = [
        ('halfcrop', HALF_ROW_START, HALF_ROW_END, HALF_COL_START, HALF_COL_END),
        ('fovcrop', CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END),
    ]

    detailed_indices = compute_detailed_snapshot_indices(stim_duration)
    n_cols = len(detailed_indices)

    stim_start = PRE_STIM_FRAMES
    stim_end = PRE_STIM_FRAMES + stim_duration - 1
    col_labels = []
    for fidx in detailed_indices:
        if fidx < stim_start:
            col_labels.append('Pre')
        elif fidx > stim_end:
            col_labels.append('Post')
        else:
            pct = round(100.0 * (fidx - stim_start) / stim_duration)
            pct = min(pct, 100)
            col_labels.append(f'{pct}%')

    for cond in ('Expert', 'NoSpout'):
        color = CONDITION_COLORS[cond]
        n_reps = REPS_PER_SIZE.get(stim_size, DEFAULT_EXAMPLE_REPS)
        for rep_idx in range(1, n_reps + 1):
            key = (cond, stim_size, stim_duration, rep_idx)
            if key not in snapshot_data:
                continue
            snaps_dict = snapshot_data[key]

            for crop_tag, r_start, r_end, c_start, c_end in crops:
                fig, axes = plt.subplots(1, n_cols, figsize=(2.6 * n_cols, 3))
                if n_cols == 1:
                    axes = [axes]

                # Stimulus rectangle in crop coordinates
                crop_stim_r0 = stim_r0 - r_start
                crop_stim_c0 = stim_c0 - c_start
                crop_stim_h = stim_r1 - stim_r0 + 1
                crop_stim_w = stim_c1 - stim_c0 + 1

                for col_idx, fidx in enumerate(detailed_indices):
                    ax = axes[col_idx]
                    if fidx not in snaps_dict:
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue
                    frame = snaps_dict[fidx]
                    binary = (frame.astype(np.float64) + 1.0) / 2.0
                    crop = binary[r_start:r_end, c_start:c_end]

                    ax.imshow(crop, cmap=cmap_sim, interpolation='nearest',
                              origin='upper', aspect='equal', vmin=0, vmax=1)

                    # Stimulus rectangle
                    if (stim_r1 >= r_start and stim_r0 < r_end and
                            stim_c1 >= c_start and stim_c0 < c_end):
                        rect_stim = Rectangle(
                            (crop_stim_c0 - 0.5, crop_stim_r0 - 0.5),
                            crop_stim_w, crop_stim_h,
                            linewidth=1.2, edgecolor='red', facecolor='none', linestyle='--')
                        ax.add_patch(rect_stim)

                    ax.set_xticks([])
                    ax.set_yticks([])

                    if col_idx == 0:
                        ax.set_ylabel(f'{cond} {rep_idx}', fontsize=9, fontweight='bold',
                                      color=color, rotation=0, labelpad=50, va='center')

                    onset_rel = fidx - PRE_STIM_FRAMES
                    ax.set_title(f'{col_labels[col_idx]}\n{onset_rel:+d}', fontsize=7)

                fig.suptitle(f'{cond} Replicate {rep_idx} [{crop_tag}] — '
                             f'size={stim_size}, dur={stim_duration} MC sweeps ({stim_mode_label})',
                             fontsize=11, fontweight='bold', y=0.995)
                fig.tight_layout(rect=[0.06, 0.0, 1.0, 0.98])

                crop_subfolder = 'FOVcrop' if crop_tag == 'fovcrop' else 'Halfcrop'
                _save_figure(fig, output_dir,
                             ('PerturbationSnapshots', f'dur_{stim_duration}', 'Single_Replicates',
                              crop_subfolder, f'size_{stim_size}'),
                             f'snapshot_{cond.lower()}_rep{rep_idx}_{crop_tag}_size_{stim_size}_{stim_mode_label}')


# =============================================================================
# Heatmap & Asymmetry Plots
# =============================================================================

def _heatmap_column_labels(detailed_indices, stim_duration):
    """Build column labels for heatmap timeline figures."""
    stim_start = PRE_STIM_FRAMES
    stim_end = PRE_STIM_FRAMES + stim_duration - 1
    labels = []
    for fidx in detailed_indices:
        if fidx < stim_start:
            labels.append('Pre')
        elif fidx > stim_end:
            labels.append('Post')
        else:
            pct = round(100.0 * (fidx - stim_start) / stim_duration)
            pct = min(pct, 100)
            labels.append(f'{pct}%')
    return labels


def _metric_config(metric):
    """Return (ylabel_peak, ylabel_mean, title_prefix, fname_prefix, subfolder) for a metric."""
    if metric == 'frac_crop':
        return ('Peak active fraction (FOV crop)', 'Mean active fraction (FOV crop)',
                'Frac Active (FOV)', 'frac_crop', 'frac_active')
    if metric == 'blob_area_net':
        return ('Peak net blob area (pixels)', 'Mean net blob area (pixels)',
                'Net Blob Area', 'blob_net', 'net_propagation')
    return ('Peak blob area (pixels)', 'Mean blob area (pixels)',
            'Blob Area', 'blob', 'raw_blob')


def _cond_subfolder(conditions):
    """Return subfolder name and filename tag for a condition subset."""
    if set(conditions) == set(EXPERT_NOSPOUT):
        return 'Expert_NoSpout', '_expNS'
    return 'All4Conditions', ''


def _save_figure(fig, output_dir, subdir_parts, name):
    """Save figure in png + pdf and close it."""
    subdir = os.path.join(output_dir, *subdir_parts)
    os.makedirs(subdir, exist_ok=True)
    for fmt in ('png', 'pdf'):
        filepath = os.path.join(subdir, f'{name}.{fmt}')
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    rel = '/'.join(subdir_parts) + '/' + name
    print(f"  Saved: {rel}")


def _find_peak_propagation_idx(heatmap_data, stim_size, stim_duration, detailed_indices):
    """Return (tp_idx, frame_idx) of the frame with largest total excess P(active)."""
    best_tp = 0
    best_excess = -np.inf
    for tp_idx, fidx in enumerate(detailed_indices):
        total_excess = 0.0
        for cond in CONDITIONS:
            key = f'{cond}_{stim_size}_{stim_duration}_f{fidx}'
            if key in heatmap_data:
                total_excess += np.sum(heatmap_data[key] - 0.5)
        if total_excess > best_excess:
            best_excess = total_excess
            best_tp = tp_idx
    return best_tp, detailed_indices[best_tp]


def plot_heatmap_timeline(heatmap_data, stim_size, stim_duration, output_dir,
                          stim_mode_label='clamped'):
    """
    Figure A: Probability heatmap timeline — all conditions.

    Layout: 4 rows (conditions) x N columns (timepoints).
    Colormap: diverging centred at 0.5.
    """
    dur_key = f'__frame_indices_{stim_duration}__'
    if dur_key not in heatmap_data:
        return
    detailed_indices = heatmap_data[dur_key].tolist()
    n_cols = len(detailed_indices)
    col_labels = _heatmap_column_labels(detailed_indices, stim_duration)

    center_row = L // 2
    center_col = M // 2
    stim_rows, stim_cols = get_stimulus_region(stim_size, center_row, center_col, L, M)
    stim_r0, stim_r1 = int(stim_rows[0]), int(stim_rows[-1])
    stim_c0, stim_c1 = int(stim_cols[0]), int(stim_cols[-1])

    fig, axes = plt.subplots(len(CONDITIONS), n_cols,
                             figsize=(2.2 * n_cols, 2.5 * len(CONDITIONS)))
    if len(CONDITIONS) == 1:
        axes = axes[np.newaxis, :]

    for cond_idx, cond in enumerate(CONDITIONS):
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        for col_idx, fidx in enumerate(detailed_indices):
            key = f'{cond}_{stim_size}_{stim_duration}_f{fidx}'
            ax = axes[cond_idx, col_idx]

            if key in heatmap_data:
                prob = heatmap_data[key]
                im = ax.imshow(prob, cmap='Greys', interpolation='nearest',
                               origin='upper', aspect='equal', vmin=0, vmax=1)
            else:
                ax.set_facecolor('#eeeeee')

            # Overlays
            rect_crop = Rectangle(
                (CROP_COL_START - 0.5, CROP_ROW_START - 0.5),
                CROP_COL_END - CROP_COL_START, CROP_ROW_END - CROP_ROW_START,
                linewidth=0.8, edgecolor='gray', facecolor='none', linestyle='--')
            ax.add_patch(rect_crop)

            rect_stim = Rectangle(
                (stim_c0 - 0.5, stim_r0 - 0.5),
                stim_c1 - stim_c0 + 1, stim_r1 - stim_r0 + 1,
                linewidth=1.0, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect_stim)

            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx == 0:
                ax.set_ylabel(cond, fontsize=9, fontweight='bold',
                              color=color, rotation=0, labelpad=40, va='center')
            if cond_idx == 0:
                onset_rel = fidx - PRE_STIM_FRAMES
                ax.set_title(f'{col_labels[col_idx]}\n{onset_rel:+d}', fontsize=7)

    fig.suptitle(f'P(active) Heatmap [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (n={heatmap_data.get("__n_reps__", "?")} reps)',
                 fontsize=11, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0.06, 0.0, 1.0, 0.97])

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Heatmaps'),
                 f'heatmap_prob_size_{stim_size}_dur_{stim_duration}')


def plot_heatmap_crop_timeline(heatmap_data, stim_size, stim_duration, output_dir,
                               stim_mode_label='clamped'):
    """
    Figure B: Centre-crop probability heatmap timeline — all conditions.
    """
    dur_key = f'__frame_indices_{stim_duration}__'
    if dur_key not in heatmap_data:
        return
    detailed_indices = heatmap_data[dur_key].tolist()
    n_cols = len(detailed_indices)
    col_labels = _heatmap_column_labels(detailed_indices, stim_duration)

    center_row = L // 2
    center_col = M // 2
    stim_rows, stim_cols = get_stimulus_region(stim_size, center_row, center_col, L, M)
    stim_r0, stim_r1 = int(stim_rows[0]), int(stim_rows[-1])
    stim_c0, stim_c1 = int(stim_cols[0]), int(stim_cols[-1])

    fig, axes = plt.subplots(len(CONDITIONS), n_cols,
                             figsize=(2.2 * n_cols, 2.5 * len(CONDITIONS)))
    if len(CONDITIONS) == 1:
        axes = axes[np.newaxis, :]

    for cond_idx, cond in enumerate(CONDITIONS):
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        for col_idx, fidx in enumerate(detailed_indices):
            key = f'{cond}_{stim_size}_{stim_duration}_f{fidx}'
            ax = axes[cond_idx, col_idx]

            if key in heatmap_data:
                prob = heatmap_data[key]
                crop = prob[CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
                ax.imshow(crop, cmap='Greys', interpolation='nearest',
                          origin='upper', aspect='equal', vmin=0, vmax=1)

                # Stimulus rectangle in crop coordinates
                if (stim_r1 >= CROP_ROW_START and stim_r0 < CROP_ROW_END and
                        stim_c1 >= CROP_COL_START and stim_c0 < CROP_COL_END):
                    rect = Rectangle(
                        (stim_c0 - CROP_COL_START - 0.5, stim_r0 - CROP_ROW_START - 0.5),
                        stim_c1 - stim_c0 + 1, stim_r1 - stim_r0 + 1,
                        linewidth=1.0, edgecolor='red', facecolor='none', linestyle='--')
                    ax.add_patch(rect)
            else:
                ax.set_facecolor('#eeeeee')

            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx == 0:
                ax.set_ylabel(cond, fontsize=9, fontweight='bold',
                              color=color, rotation=0, labelpad=40, va='center')
            if cond_idx == 0:
                onset_rel = fidx - PRE_STIM_FRAMES
                ax.set_title(f'{col_labels[col_idx]}\n{onset_rel:+d}', fontsize=7)

    fig.suptitle(f'P(active) Crop [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (n={heatmap_data.get("__n_reps__", "?")} reps)',
                 fontsize=11, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0.06, 0.0, 1.0, 0.97])

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Heatmaps'),
                 f'heatmap_prob_crop_size_{stim_size}_dur_{stim_duration}')


def plot_heatmap_expert_focus(heatmap_data, stim_size, stim_duration, output_dir,
                              stim_mode_label='clamped'):
    """
    Figure C: Expert-only heatmap — 2 rows (full grid + crop) x N columns.
    """
    dur_key = f'__frame_indices_{stim_duration}__'
    if dur_key not in heatmap_data:
        return
    detailed_indices = heatmap_data[dur_key].tolist()
    n_cols = len(detailed_indices)
    col_labels = _heatmap_column_labels(detailed_indices, stim_duration)

    center_row = L // 2
    center_col = M // 2
    stim_rows, stim_cols = get_stimulus_region(stim_size, center_row, center_col, L, M)
    stim_r0, stim_r1 = int(stim_rows[0]), int(stim_rows[-1])
    stim_c0, stim_c1 = int(stim_cols[0]), int(stim_cols[-1])

    cond = 'Expert'
    color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))

    fig, axes = plt.subplots(2, n_cols, figsize=(2.2 * n_cols, 5))

    for col_idx, fidx in enumerate(detailed_indices):
        key = f'{cond}_{stim_size}_{stim_duration}_f{fidx}'

        # Full grid
        ax_full = axes[0, col_idx]
        if key in heatmap_data:
            prob = heatmap_data[key]
            ax_full.imshow(prob, cmap='Greys', interpolation='nearest',
                           origin='upper', aspect='equal', vmin=0, vmax=1)
        rect_crop = Rectangle(
            (CROP_COL_START - 0.5, CROP_ROW_START - 0.5),
            CROP_COL_END - CROP_COL_START, CROP_ROW_END - CROP_ROW_START,
            linewidth=0.8, edgecolor='gray', facecolor='none', linestyle='--')
        ax_full.add_patch(rect_crop)
        rect_stim = Rectangle(
            (stim_c0 - 0.5, stim_r0 - 0.5),
            stim_c1 - stim_c0 + 1, stim_r1 - stim_r0 + 1,
            linewidth=1.0, edgecolor='red', facecolor='none', linestyle='--')
        ax_full.add_patch(rect_stim)
        ax_full.set_xticks([])
        ax_full.set_yticks([])
        if col_idx == 0:
            ax_full.set_ylabel('Full', fontsize=9, fontweight='bold',
                               color=color, rotation=0, labelpad=30, va='center')
        onset_rel = fidx - PRE_STIM_FRAMES
        ax_full.set_title(f'{col_labels[col_idx]}\n{onset_rel:+d}', fontsize=7)

        # Crop
        ax_crop = axes[1, col_idx]
        if key in heatmap_data:
            crop = heatmap_data[key][CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
            ax_crop.imshow(crop, cmap='Greys', interpolation='nearest',
                           origin='upper', aspect='equal', vmin=0, vmax=1)
            if (stim_r1 >= CROP_ROW_START and stim_r0 < CROP_ROW_END and
                    stim_c1 >= CROP_COL_START and stim_c0 < CROP_COL_END):
                rect = Rectangle(
                    (stim_c0 - CROP_COL_START - 0.5, stim_r0 - CROP_ROW_START - 0.5),
                    stim_c1 - stim_c0 + 1, stim_r1 - stim_r0 + 1,
                    linewidth=1.0, edgecolor='red', facecolor='none', linestyle='--')
                ax_crop.add_patch(rect)
        ax_crop.set_xticks([])
        ax_crop.set_yticks([])
        if col_idx == 0:
            ax_crop.set_ylabel('Crop', fontsize=9, fontweight='bold',
                               color=color, rotation=0, labelpad=30, va='center')

    fig.suptitle(f'Expert P(active) [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (n={heatmap_data.get("__n_reps__", "?")} reps)',
                 fontsize=11, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0.05, 0.0, 1.0, 0.95])

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Heatmaps'),
                 f'heatmap_prob_expert_size_{stim_size}_dur_{stim_duration}')


def plot_heatmap_expert_nospout_focus(heatmap_data, stim_size, stim_duration, output_dir,
                                      stim_mode_label='clamped'):
    """
    Expert vs NoSpout heatmap — 4 rows (Expert full, Expert crop,
    NoSpout full, NoSpout crop) x N columns.
    """
    dur_key = f'__frame_indices_{stim_duration}__'
    if dur_key not in heatmap_data:
        return
    detailed_indices = heatmap_data[dur_key].tolist()
    n_cols = len(detailed_indices)
    col_labels = _heatmap_column_labels(detailed_indices, stim_duration)

    center_row = L // 2
    center_col = M // 2
    stim_rows, stim_cols = get_stimulus_region(stim_size, center_row, center_col, L, M)
    stim_r0, stim_r1 = int(stim_rows[0]), int(stim_rows[-1])
    stim_c0, stim_c1 = int(stim_cols[0]), int(stim_cols[-1])

    fig, axes = plt.subplots(4, n_cols, figsize=(2.2 * n_cols, 10))

    for row_pair, cond in enumerate(('Expert', 'NoSpout')):
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        full_row = row_pair * 2
        crop_row = row_pair * 2 + 1

        for col_idx, fidx in enumerate(detailed_indices):
            key = f'{cond}_{stim_size}_{stim_duration}_f{fidx}'

            # Full grid
            ax_full = axes[full_row, col_idx]
            if key in heatmap_data:
                prob = heatmap_data[key]
                ax_full.imshow(prob, cmap='Greys', interpolation='nearest',
                               origin='upper', aspect='equal', vmin=0, vmax=1)
            rect_crop = Rectangle(
                (CROP_COL_START - 0.5, CROP_ROW_START - 0.5),
                CROP_COL_END - CROP_COL_START, CROP_ROW_END - CROP_ROW_START,
                linewidth=0.8, edgecolor='gray', facecolor='none', linestyle='--')
            ax_full.add_patch(rect_crop)
            rect_stim = Rectangle(
                (stim_c0 - 0.5, stim_r0 - 0.5),
                stim_c1 - stim_c0 + 1, stim_r1 - stim_r0 + 1,
                linewidth=1.0, edgecolor='red', facecolor='none', linestyle='--')
            ax_full.add_patch(rect_stim)
            ax_full.set_xticks([])
            ax_full.set_yticks([])
            if col_idx == 0:
                ax_full.set_ylabel(f'{cond}\nFull', fontsize=9, fontweight='bold',
                                   color=color, rotation=0, labelpad=40, va='center')
            if row_pair == 0:
                onset_rel = fidx - PRE_STIM_FRAMES
                ax_full.set_title(f'{col_labels[col_idx]}\n{onset_rel:+d}', fontsize=7)

            # Crop
            ax_crop = axes[crop_row, col_idx]
            if key in heatmap_data:
                crop = heatmap_data[key][CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
                ax_crop.imshow(crop, cmap='Greys', interpolation='nearest',
                               origin='upper', aspect='equal', vmin=0, vmax=1)
                if (stim_r1 >= CROP_ROW_START and stim_r0 < CROP_ROW_END and
                        stim_c1 >= CROP_COL_START and stim_c0 < CROP_COL_END):
                    rect = Rectangle(
                        (stim_c0 - CROP_COL_START - 0.5, stim_r0 - CROP_ROW_START - 0.5),
                        stim_c1 - stim_c0 + 1, stim_r1 - stim_r0 + 1,
                        linewidth=1.0, edgecolor='red', facecolor='none', linestyle='--')
                    ax_crop.add_patch(rect)
            ax_crop.set_xticks([])
            ax_crop.set_yticks([])
            if col_idx == 0:
                ax_crop.set_ylabel(f'{cond}\nCrop', fontsize=9, fontweight='bold',
                                   color=color, rotation=0, labelpad=40, va='center')

    fig.suptitle(f'Expert vs NoSpout P(active) [{stim_mode_label}] — size={stim_size}, dur={stim_duration} '
                 f'(n={heatmap_data.get("__n_reps__", "?")} reps)',
                 fontsize=11, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0.07, 0.0, 1.0, 0.95])

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Heatmaps'),
                 f'heatmap_prob_expert_nospout_size_{stim_size}_dur_{stim_duration}')


def plot_asymmetry_polar(heatmap_data, stim_size, stim_duration, output_dir,
                         stim_mode_label='clamped'):
    """
    Figure D: Polar scatter of centroid offsets + eccentricity boxplots.

    Layout: 2 rows x 4 columns (one per condition).
    Row 1: Polar scatter (direction vs magnitude).
    Row 2: Box plots of eccentricity.
    """
    dur_key = f'__frame_indices_{stim_duration}__'
    if dur_key not in heatmap_data:
        return
    detailed_indices = heatmap_data[dur_key].tolist()

    # Pick a representative timepoint: middle of stimulus
    stim_start = PRE_STIM_FRAMES
    stim_mid_idx = None
    for tp_idx, fidx in enumerate(detailed_indices):
        if fidx >= stim_start + stim_duration // 2:
            stim_mid_idx = tp_idx
            break
    if stim_mid_idx is None:
        stim_mid_idx = len(detailed_indices) // 2

    # Compute global radial max across all conditions for consistent axes
    global_rmax = 0.0
    for cond in CONDITIONS:
        mag_key = f'{cond}_{stim_size}_{stim_duration}_centroid_mag'
        if mag_key in heatmap_data:
            mags = heatmap_data[mag_key][:, stim_mid_idx]
            valid = ~np.isnan(mags)
            if np.any(valid):
                global_rmax = max(global_rmax, np.percentile(mags[valid], 95))
    global_rmax = max(global_rmax * 1.1, 1.0)

    fig = plt.figure(figsize=(16, 8))

    for cond_idx, cond in enumerate(CONDITIONS):
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        prefix = f'{cond}_{stim_size}_{stim_duration}'

        dir_key = f'{prefix}_centroid_dir'
        mag_key = f'{prefix}_centroid_mag'
        ecc_key = f'{prefix}_eccentricity'

        if dir_key not in heatmap_data:
            continue

        dirs = heatmap_data[dir_key][:, stim_mid_idx]
        mags = heatmap_data[mag_key][:, stim_mid_idx]
        eccs = heatmap_data[ecc_key][:, stim_mid_idx]

        # Filter NaN
        valid = ~np.isnan(dirs) & ~np.isnan(mags)

        # Row 1: Polar scatter
        ax_polar = fig.add_subplot(2, len(CONDITIONS), cond_idx + 1, projection='polar')
        if np.any(valid):
            ax_polar.scatter(dirs[valid], mags[valid], c=[color], alpha=0.5, s=15)
        ax_polar.set_title(cond, fontsize=10, fontweight='bold', color=color, pad=12)
        ax_polar.set_rticks([])
        ax_polar.set_rlim(0, global_rmax)

        # Row 2: Eccentricity boxplot
        ax_box = fig.add_subplot(2, len(CONDITIONS), len(CONDITIONS) + cond_idx + 1)
        valid_ecc = eccs[~np.isnan(eccs)]
        if len(valid_ecc) > 0:
            bp = ax_box.boxplot([valid_ecc], vert=True, patch_artist=True, widths=0.5)
            bp['boxes'][0].set_facecolor(color + (0.3,))
            bp['boxes'][0].set_edgecolor(color)
            bp['medians'][0].set_color('black')
        ax_box.set_ylabel('Eccentricity' if cond_idx == 0 else '')
        ax_box.set_xticks([])
        ax_box.set_ylim(0, 1)
        ax_box.set_yticks([0, 0.5, 1.0])
        if cond_idx == len(CONDITIONS) - 1:
            ax_right = ax_box.secondary_yaxis('right')
            ax_right.set_yticks([0.05, 0.95])
            ax_right.set_yticklabels(['circular', 'elongated'], fontsize=7, fontstyle='italic')
            ax_right.tick_params(length=0)
        ax_box.set_title(cond, fontsize=9, color=color)

    fig.suptitle(f'Asymmetry [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (stim midpoint)',
                 fontsize=11, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Asymmetry'),
                 f'asymmetry_polar_size_{stim_size}_dur_{stim_duration}')


def plot_asymmetry_sectors(heatmap_data, stim_size, stim_duration, output_dir,
                           stim_mode_label='clamped'):
    """
    Figure E: Radar/rose plots of sector fractions.

    Layout: 1 row x 4 columns. Mean sectors + individual replicate traces.
    """
    dur_key = f'__frame_indices_{stim_duration}__'
    if dur_key not in heatmap_data:
        return
    detailed_indices = heatmap_data[dur_key].tolist()

    # Pick stim midpoint
    stim_start = PRE_STIM_FRAMES
    stim_mid_idx = None
    for tp_idx, fidx in enumerate(detailed_indices):
        if fidx >= stim_start + stim_duration // 2:
            stim_mid_idx = tp_idx
            break
    if stim_mid_idx is None:
        stim_mid_idx = len(detailed_indices) // 2

    sector_labels = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    # Compute global radial max across all conditions for consistent axes
    global_rmax = 0.0
    for cond in CONDITIONS:
        sec_key = f'{cond}_{stim_size}_{stim_duration}_sectors'
        if sec_key in heatmap_data:
            all_sec = heatmap_data[sec_key][:, stim_mid_idx, :]
            valid = ~np.any(np.isnan(all_sec), axis=1)
            if np.any(valid):
                cond_max = np.max(np.mean(all_sec[valid], axis=0))
                global_rmax = max(global_rmax, cond_max)
    global_rmax = max(global_rmax * 1.1, 0.2)

    fig, axes_list = plt.subplots(1, len(CONDITIONS), figsize=(4 * len(CONDITIONS), 4),
                                  subplot_kw=dict(projection='polar'))
    if len(CONDITIONS) == 1:
        axes_list = [axes_list]

    for cond_idx, cond in enumerate(CONDITIONS):
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        prefix = f'{cond}_{stim_size}_{stim_duration}'
        sec_key = f'{prefix}_sectors'

        ax = axes_list[cond_idx]

        if sec_key not in heatmap_data:
            ax.set_title(cond, fontsize=10, color=color)
            continue

        all_sectors = heatmap_data[sec_key][:, stim_mid_idx, :]  # (n_reps, 8)

        # Individual replicates (thin, transparent)
        for rep in range(all_sectors.shape[0]):
            vals = all_sectors[rep]
            if np.any(np.isnan(vals)):
                continue
            closed = np.append(vals, vals[0])
            closed_angles = np.append(angles, angles[0])
            ax.plot(closed_angles, closed, color=color, alpha=0.08, linewidth=0.5)

        # Mean across replicates (thick)
        valid_mask = ~np.any(np.isnan(all_sectors), axis=1)
        if np.any(valid_mask):
            mean_sectors = np.mean(all_sectors[valid_mask], axis=0)
            closed = np.append(mean_sectors, mean_sectors[0])
            closed_angles = np.append(angles, angles[0])
            ax.plot(closed_angles, closed, color=color, linewidth=2.5, label='Mean')
            ax.fill(closed_angles, closed, color=color, alpha=0.15)

        # Uniform reference
        uniform = np.ones(9) / 8.0
        ax.plot(np.append(angles, angles[0]), uniform, 'k--', linewidth=0.8, alpha=0.5)

        ax.set_thetagrids(np.degrees(angles), sector_labels)
        ax.set_title(cond, fontsize=10, fontweight='bold', color=color, pad=12)
        ax.set_ylim(0, global_rmax)

    fig.suptitle(f'Sector Fractions [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (stim midpoint)',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Asymmetry'),
                 f'asymmetry_sectors_size_{stim_size}_dur_{stim_duration}')


def plot_asymmetry_polar_peak(heatmap_data, stim_size, stim_duration, output_dir,
                              stim_mode_label='clamped'):
    """
    Figure D-peak: Polar scatter at peak propagation timepoint.

    Same layout as plot_asymmetry_polar but uses the frame with largest
    total excess P(active) instead of the stimulus midpoint.
    """
    dur_key = f'__frame_indices_{stim_duration}__'
    if dur_key not in heatmap_data:
        return
    detailed_indices = heatmap_data[dur_key].tolist()

    peak_tp_idx, peak_fidx = _find_peak_propagation_idx(
        heatmap_data, stim_size, stim_duration, detailed_indices)

    # Compute global radial max across all conditions for consistent axes
    global_rmax = 0.0
    for cond in CONDITIONS:
        mag_key = f'{cond}_{stim_size}_{stim_duration}_centroid_mag'
        if mag_key in heatmap_data:
            mags = heatmap_data[mag_key][:, peak_tp_idx]
            valid = ~np.isnan(mags)
            if np.any(valid):
                global_rmax = max(global_rmax, np.percentile(mags[valid], 95))
    global_rmax = max(global_rmax * 1.1, 1.0)

    fig = plt.figure(figsize=(16, 8))

    for cond_idx, cond in enumerate(CONDITIONS):
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        prefix = f'{cond}_{stim_size}_{stim_duration}'

        dir_key = f'{prefix}_centroid_dir'
        mag_key = f'{prefix}_centroid_mag'
        ecc_key = f'{prefix}_eccentricity'

        if dir_key not in heatmap_data:
            continue

        dirs = heatmap_data[dir_key][:, peak_tp_idx]
        mags = heatmap_data[mag_key][:, peak_tp_idx]
        eccs = heatmap_data[ecc_key][:, peak_tp_idx]

        # Filter NaN
        valid = ~np.isnan(dirs) & ~np.isnan(mags)

        # Row 1: Polar scatter
        ax_polar = fig.add_subplot(2, len(CONDITIONS), cond_idx + 1, projection='polar')
        if np.any(valid):
            ax_polar.scatter(dirs[valid], mags[valid], c=[color], alpha=0.5, s=15)
        ax_polar.set_title(cond, fontsize=10, fontweight='bold', color=color, pad=12)
        ax_polar.set_rticks([])
        ax_polar.set_rlim(0, global_rmax)

        # Row 2: Eccentricity boxplot
        ax_box = fig.add_subplot(2, len(CONDITIONS), len(CONDITIONS) + cond_idx + 1)
        valid_ecc = eccs[~np.isnan(eccs)]
        if len(valid_ecc) > 0:
            bp = ax_box.boxplot([valid_ecc], vert=True, patch_artist=True, widths=0.5)
            bp['boxes'][0].set_facecolor(color + (0.3,))
            bp['boxes'][0].set_edgecolor(color)
            bp['medians'][0].set_color('black')
        ax_box.set_ylabel('Eccentricity' if cond_idx == 0 else '')
        ax_box.set_xticks([])
        ax_box.set_ylim(0, 1)
        ax_box.set_yticks([0, 0.5, 1.0])
        if cond_idx == len(CONDITIONS) - 1:
            ax_right = ax_box.secondary_yaxis('right')
            ax_right.set_yticks([0.05, 0.95])
            ax_right.set_yticklabels(['circular', 'elongated'], fontsize=7, fontstyle='italic')
            ax_right.tick_params(length=0)
        ax_box.set_title(cond, fontsize=9, color=color)

    fig.suptitle(f'Asymmetry [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (peak propagation, frame {peak_fidx})',
                 fontsize=11, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Asymmetry'),
                 f'asymmetry_polar_peak_size_{stim_size}_dur_{stim_duration}')


def plot_asymmetry_sectors_peak(heatmap_data, stim_size, stim_duration, output_dir,
                                stim_mode_label='clamped'):
    """
    Figure E-peak: Radar/rose plots at peak propagation timepoint.

    Same layout as plot_asymmetry_sectors but uses the frame with largest
    total excess P(active) instead of the stimulus midpoint.
    """
    dur_key = f'__frame_indices_{stim_duration}__'
    if dur_key not in heatmap_data:
        return
    detailed_indices = heatmap_data[dur_key].tolist()

    peak_tp_idx, peak_fidx = _find_peak_propagation_idx(
        heatmap_data, stim_size, stim_duration, detailed_indices)

    sector_labels = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    # Compute global radial max across all conditions for consistent axes
    global_rmax = 0.0
    for cond in CONDITIONS:
        sec_key = f'{cond}_{stim_size}_{stim_duration}_sectors'
        if sec_key in heatmap_data:
            all_sec = heatmap_data[sec_key][:, peak_tp_idx, :]
            valid = ~np.any(np.isnan(all_sec), axis=1)
            if np.any(valid):
                cond_max = np.max(np.mean(all_sec[valid], axis=0))
                global_rmax = max(global_rmax, cond_max)
    global_rmax = max(global_rmax * 1.1, 0.2)

    fig, axes_list = plt.subplots(1, len(CONDITIONS), figsize=(4 * len(CONDITIONS), 4),
                                  subplot_kw=dict(projection='polar'))
    if len(CONDITIONS) == 1:
        axes_list = [axes_list]

    for cond_idx, cond in enumerate(CONDITIONS):
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        prefix = f'{cond}_{stim_size}_{stim_duration}'
        sec_key = f'{prefix}_sectors'

        ax = axes_list[cond_idx]

        if sec_key not in heatmap_data:
            ax.set_title(cond, fontsize=10, color=color)
            continue

        all_sectors = heatmap_data[sec_key][:, peak_tp_idx, :]  # (n_reps, 8)

        # Individual replicates (thin, transparent)
        for rep in range(all_sectors.shape[0]):
            vals = all_sectors[rep]
            if np.any(np.isnan(vals)):
                continue
            closed = np.append(vals, vals[0])
            closed_angles = np.append(angles, angles[0])
            ax.plot(closed_angles, closed, color=color, alpha=0.08, linewidth=0.5)

        # Mean across replicates (thick)
        valid_mask = ~np.any(np.isnan(all_sectors), axis=1)
        if np.any(valid_mask):
            mean_sectors = np.mean(all_sectors[valid_mask], axis=0)
            closed = np.append(mean_sectors, mean_sectors[0])
            closed_angles = np.append(angles, angles[0])
            ax.plot(closed_angles, closed, color=color, linewidth=2.5, label='Mean')
            ax.fill(closed_angles, closed, color=color, alpha=0.15)

        # Uniform reference
        uniform = np.ones(9) / 8.0
        ax.plot(np.append(angles, angles[0]), uniform, 'k--', linewidth=0.8, alpha=0.5)

        ax.set_thetagrids(np.degrees(angles), sector_labels)
        ax.set_title(cond, fontsize=10, fontweight='bold', color=color, pad=12)
        ax.set_ylim(0, global_rmax)

    fig.suptitle(f'Sector Fractions [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (peak propagation, frame {peak_fidx})',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Asymmetry'),
                 f'asymmetry_sectors_peak_size_{stim_size}_dur_{stim_duration}')


def plot_asymmetry_summary(heatmap_data, stim_duration, sizes, output_dir,
                           stim_mode_label='clamped'):
    """
    Figure F: Asymmetry metrics vs stimulus size (one duration).

    Layout: 2x2 grid:
      (0,0) Eccentricity vs size
      (0,1) Centroid offset vs size
      (1,0) Anisotropy vs size
      (1,1) Sector uniformity (Rayleigh test p-value)
    """
    dur_key = f'__frame_indices_{stim_duration}__'
    if dur_key not in heatmap_data:
        return
    detailed_indices = heatmap_data[dur_key].tolist()

    # Pick stim midpoint
    stim_start = PRE_STIM_FRAMES
    stim_mid_idx = None
    for tp_idx, fidx in enumerate(detailed_indices):
        if fidx >= stim_start + stim_duration // 2:
            stim_mid_idx = tp_idx
            break
    if stim_mid_idx is None:
        stim_mid_idx = len(detailed_indices) // 2

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for cond in CONDITIONS:
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))

        ecc_means, ecc_sems = [], []
        mag_means, mag_sems = [], []
        aniso_means, aniso_sems = [], []
        rayleigh_pvals = []
        valid_sizes = []

        for size in sizes:
            prefix = f'{cond}_{size}_{stim_duration}'
            ecc_key = f'{prefix}_eccentricity'
            if ecc_key not in heatmap_data:
                continue

            eccs = heatmap_data[ecc_key][:, stim_mid_idx]
            mags = heatmap_data[f'{prefix}_centroid_mag'][:, stim_mid_idx]
            anisos = heatmap_data[f'{prefix}_anisotropy'][:, stim_mid_idx]
            dirs_raw = heatmap_data[f'{prefix}_centroid_dir'][:, stim_mid_idx]

            valid_eccs = eccs[~np.isnan(eccs)]
            valid_mags = mags[~np.isnan(mags)]
            valid_anisos = anisos[~np.isnan(anisos)]
            valid_dirs = dirs_raw[~np.isnan(dirs_raw)]

            if len(valid_eccs) == 0:
                continue

            valid_sizes.append(size)
            n = len(valid_eccs)

            ecc_means.append(np.mean(valid_eccs))
            ecc_sems.append(np.std(valid_eccs) / np.sqrt(n))

            mag_means.append(np.mean(valid_mags))
            mag_sems.append(np.std(valid_mags) / np.sqrt(n))

            aniso_means.append(np.mean(valid_anisos))
            aniso_sems.append(np.std(valid_anisos) / np.sqrt(n))

            # Rayleigh test for uniformity of centroid directions
            if len(valid_dirs) >= 3:
                R = np.sqrt(np.mean(np.cos(valid_dirs))**2 + np.mean(np.sin(valid_dirs))**2)
                n_d = len(valid_dirs)
                # Rayleigh test statistic: Z = n * R^2
                Z = n_d * R ** 2
                # Approximate p-value
                p_val = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n_d) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n_d**2))
                rayleigh_pvals.append(max(p_val, 1e-10))
            else:
                rayleigh_pvals.append(np.nan)

        if not valid_sizes:
            continue

        x = np.array(valid_sizes)

        # (0,0) Eccentricity
        axes[0, 0].errorbar(x, ecc_means, yerr=ecc_sems, color=color,
                            marker='o', markersize=4, capsize=3, label=cond)

        # (0,1) Centroid offset
        axes[0, 1].errorbar(x, mag_means, yerr=mag_sems, color=color,
                            marker='o', markersize=4, capsize=3, label=cond)

        # (1,0) Anisotropy
        axes[1, 0].errorbar(x, aniso_means, yerr=aniso_sems, color=color,
                            marker='o', markersize=4, capsize=3, label=cond)

        # (1,1) Rayleigh p-value
        valid_p = [(s, p) for s, p in zip(valid_sizes, rayleigh_pvals) if not np.isnan(p)]
        if valid_p:
            sx, sp = zip(*valid_p)
            axes[1, 1].semilogy(sx, sp, color=color, marker='o', markersize=4, label=cond)

    axes[0, 0].set_ylabel('Eccentricity')
    axes[0, 0].set_xlabel('Stimulus size (px)')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].set_ylabel('Centroid offset (px)')
    axes[0, 1].set_xlabel('Stimulus size (px)')
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].set_ylabel('Wavefront anisotropy')
    axes[1, 0].set_xlabel('Stimulus size (px)')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].set_ylabel('Rayleigh p-value')
    axes[1, 1].set_xlabel('Stimulus size (px)')
    axes[1, 1].axhline(0.05, color='gray', linestyle='--', linewidth=0.8, label='p=0.05')
    axes[1, 1].legend(fontsize=7)

    fig.suptitle(f'Asymmetry Summary [{stim_mode_label}] — dur={stim_duration}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', 'Asymmetry'),
                 f'asymmetry_summary_dur_{stim_duration}')


def plot_spaghetti_fraction(heatmap_data, stim_size, stim_duration, output_dir, crop='full',
                            stim_mode_label='clamped'):
    """
    Spaghetti plot of fraction-active over time, one line per replicate.

    Parameters
    ----------
    heatmap_data : dict
    stim_size, stim_duration : int
    output_dir : str
    crop : str
        'full' (entire grid), 'half' (half-way crop), or 'fov' (experimental FOV crop).
    """
    # Use spaghetti indices (every-frame resolution); fall back to detailed indices
    spag_key = f'__spaghetti_indices_{stim_duration}__'
    dur_key = f'__frame_indices_{stim_duration}__'
    if spag_key in heatmap_data:
        frame_indices = heatmap_data[spag_key].tolist()
    elif dur_key in heatmap_data:
        frame_indices = heatmap_data[dur_key].tolist()
    else:
        return

    # Map crop name to data key suffix
    suffix_map = {'full': 'frac_full', 'half': 'frac_half', 'fov': 'frac_crop'}
    frac_suffix = suffix_map.get(crop, 'frac_full')

    crop_labels = {'full': 'Full grid', 'half': 'Half crop', 'fov': 'FOV crop'}
    crop_label = crop_labels.get(crop, crop)

    n_reps = heatmap_data.get('__n_reps__', '?')

    # X-axis: frame indices relative to stimulus onset
    x = np.array(frame_indices) - PRE_STIM_FRAMES

    # Data-driven y-axis max across all conditions
    y_max = 0.0
    for cond in CONDITIONS:
        fk = f'{cond}_{stim_size}_{stim_duration}_{frac_suffix}'
        if fk in heatmap_data:
            y_max = max(y_max, float(np.nanmax(heatmap_data[fk])))
    y_max = max(y_max * 1.05, 0.1)

    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4.5 * len(CONDITIONS), 4),
                             sharey=True)
    if len(CONDITIONS) == 1:
        axes = [axes]

    for cond_idx, cond in enumerate(CONDITIONS):
        ax = axes[cond_idx]
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        prefix = f'{cond}_{stim_size}_{stim_duration}'
        frac_key = f'{prefix}_{frac_suffix}'

        if frac_key not in heatmap_data:
            ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
            continue

        traces = heatmap_data[frac_key]  # (n_reps, n_timepoints)

        # Guard against spaghetti index / trace length mismatch
        x_cond = x[:traces.shape[1]] if len(x) > traces.shape[1] else x

        # Individual replicates (thin, semi-transparent)
        for rep in range(traces.shape[0]):
            ax.plot(x_cond, traces[rep, :len(x_cond)], color=color, alpha=0.15, linewidth=0.5)

        # Mean across replicates (thick)
        mean_trace = np.nanmean(traces, axis=0)
        ax.plot(x_cond, mean_trace[:len(x_cond)], color=color, linewidth=2.0, label=f'Mean (n={n_reps})')

        # Stimulus onset/offset markers
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(stim_duration, color='gray', linestyle='--', linewidth=0.8)

        ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
        ax.set_xlabel('Frame (rel. to stim onset)')
        if cond_idx == 0:
            ax.set_ylabel('Fraction active')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylim(0, y_max)

    fig.suptitle(f'Fraction Active Spaghetti ({crop_label}) [{stim_mode_label}] '
                 f'— size={stim_size}, dur={stim_duration} (n={n_reps} reps)',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Fraction_Active'),
                 f'spaghetti_frac_{crop}_size_{stim_size}_dur_{stim_duration}')


def plot_spaghetti_blob_area(heatmap_data, stim_size, stim_duration, output_dir,
                             stim_mode_label='clamped'):
    """Spaghetti plot of connected blob area over time, one line per replicate."""
    spag_key = f'__spaghetti_indices_{stim_duration}__'
    if spag_key not in heatmap_data:
        return

    frame_indices = heatmap_data[spag_key].tolist()
    n_reps = heatmap_data.get('__n_reps__', '?')
    x = np.array(frame_indices) - PRE_STIM_FRAMES

    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4.5 * len(CONDITIONS), 4),
                             sharey=True)
    if len(CONDITIONS) == 1:
        axes = [axes]

    for cond_idx, cond in enumerate(CONDITIONS):
        ax = axes[cond_idx]
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        prefix = f'{cond}_{stim_size}_{stim_duration}'
        blob_key = f'{prefix}_blob_area'

        if blob_key not in heatmap_data:
            ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
            continue

        traces = heatmap_data[blob_key]  # (n_reps, n_spaghetti)

        x_cond = x[:traces.shape[1]] if len(x) > traces.shape[1] else x
        for rep in range(traces.shape[0]):
            ax.plot(x_cond, traces[rep, :len(x_cond)], color=color, alpha=0.15, linewidth=0.5)

        mean_trace = np.nanmean(traces, axis=0)
        ax.plot(x_cond, mean_trace[:len(x_cond)], color=color, linewidth=2.0, label=f'Mean (n={n_reps})')

        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(stim_duration, color='gray', linestyle='--', linewidth=0.8)

        ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
        ax.set_xlabel('Frame (rel. to stim onset)')
        if cond_idx == 0:
            ax.set_ylabel('Blob area (pixels)')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Connected Blob Area [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (n={n_reps} reps)',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Blob_Area'),
                 f'spaghetti_blob_area_size_{stim_size}_dur_{stim_duration}')


def plot_mean_fraction(heatmap_data, stim_size, stim_duration, output_dir, crop='full',
                       stim_mode_label='clamped'):
    """Mean + SEM fraction-active over time (no individual traces)."""
    spag_key = f'__spaghetti_indices_{stim_duration}__'
    dur_key = f'__frame_indices_{stim_duration}__'
    if spag_key in heatmap_data:
        frame_indices = heatmap_data[spag_key].tolist()
    elif dur_key in heatmap_data:
        frame_indices = heatmap_data[dur_key].tolist()
    else:
        return

    suffix_map = {'full': 'frac_full', 'half': 'frac_half', 'fov': 'frac_crop'}
    frac_suffix = suffix_map.get(crop, 'frac_full')
    crop_labels = {'full': 'Full grid', 'half': 'Half crop', 'fov': 'FOV crop'}
    crop_label = crop_labels.get(crop, crop)
    n_reps = heatmap_data.get('__n_reps__', '?')

    x = np.array(frame_indices) - PRE_STIM_FRAMES

    # Data-driven y-axis max from mean+SEM (not raw individual traces)
    y_max = 0.0
    for cond in CONDITIONS:
        fk = f'{cond}_{stim_size}_{stim_duration}_{frac_suffix}'
        if fk in heatmap_data:
            tr = heatmap_data[fk]
            m = np.nanmean(tr, axis=0)
            n_v = np.sum(~np.isnan(tr), axis=0)
            s = np.nanstd(tr, axis=0) / np.sqrt(np.maximum(n_v, 1))
            y_max = max(y_max, float(np.nanmax(m + s)))
    y_max = max(y_max * 1.1, 0.01)

    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4.5 * len(CONDITIONS), 4),
                             sharey=True)
    if len(CONDITIONS) == 1:
        axes = [axes]

    for cond_idx, cond in enumerate(CONDITIONS):
        ax = axes[cond_idx]
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        frac_key = f'{cond}_{stim_size}_{stim_duration}_{frac_suffix}'

        if frac_key not in heatmap_data:
            ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
            continue

        traces = heatmap_data[frac_key]
        x_cond = x[:traces.shape[1]] if len(x) > traces.shape[1] else x
        n_pts = len(x_cond)
        mean_trace = np.nanmean(traces[:, :n_pts], axis=0)
        n_valid = np.sum(~np.isnan(traces[:, :n_pts]), axis=0)
        sem_trace = np.nanstd(traces[:, :n_pts], axis=0) / np.sqrt(np.maximum(n_valid, 1))

        ax.fill_between(x_cond, mean_trace - sem_trace, mean_trace + sem_trace,
                         color=color, alpha=0.2)
        ax.plot(x_cond, mean_trace, color=color, linewidth=2.0, label=f'Mean (n={n_reps})')

        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(stim_duration, color='gray', linestyle='--', linewidth=0.8)

        ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
        ax.set_xlabel('Frame (rel. to stim onset)')
        if cond_idx == 0:
            ax.set_ylabel('Fraction active')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylim(0, y_max)

    fig.suptitle(f'Mean Fraction Active ({crop_label}) [{stim_mode_label}] '
                 f'— size={stim_size}, dur={stim_duration} (n={n_reps} reps)',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Fraction_Active'),
                 f'mean_frac_{crop}_size_{stim_size}_dur_{stim_duration}')


def plot_batch_fraction(heatmap_data, stim_size, stim_duration, output_dir, crop='full',
                        batch_size=100, stim_mode_label='clamped',
                        time_scale_factor=1.0, sampling_rate=10.0):
    """Batch-averaged fraction-active over time (each batch = mean of batch_size reps)."""
    spag_key = f'__spaghetti_indices_{stim_duration}__'
    dur_key = f'__frame_indices_{stim_duration}__'
    if spag_key in heatmap_data:
        frame_indices = heatmap_data[spag_key].tolist()
    elif dur_key in heatmap_data:
        frame_indices = heatmap_data[dur_key].tolist()
    else:
        return

    suffix_map = {'full': 'frac_full', 'half': 'frac_half', 'fov': 'frac_crop'}
    frac_suffix = suffix_map.get(crop, 'frac_full')
    crop_labels = {'full': 'Full grid', 'half': 'Half crop', 'fov': 'FOV crop'}
    crop_label = crop_labels.get(crop, crop)
    n_reps = heatmap_data.get('__n_reps__', '?')

    x = np.array(frame_indices) - PRE_STIM_FRAMES

    # Convert to seconds when a time scale factor is provided
    use_seconds = (time_scale_factor != 1.0)
    if use_seconds:
        x = x * time_scale_factor / sampling_rate
        stim_off_x = stim_duration * time_scale_factor / sampling_rate
        dur_label = f'{stim_off_x:.1f} s'
        x_label = 'Time [s]'
    else:
        stim_off_x = stim_duration
        dur_label = str(stim_duration)
        x_label = 'Frame (rel. to stim onset)'

    # Data-driven y-axis max from batch-averaged traces
    y_max = 0.0
    for cond in CONDITIONS:
        fk = f'{cond}_{stim_size}_{stim_duration}_{frac_suffix}'
        if fk in heatmap_data:
            tr = heatmap_data[fk]
            n_b = tr.shape[0] // batch_size
            for b in range(n_b):
                bt = np.nanmean(tr[b * batch_size:(b + 1) * batch_size, :], axis=0)
                y_max = max(y_max, float(np.nanmax(bt)))
    y_max = max(y_max * 1.1, 0.01)

    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4.5 * len(CONDITIONS), 4),
                             sharey=True)
    if len(CONDITIONS) == 1:
        axes = [axes]

    for cond_idx, cond in enumerate(CONDITIONS):
        ax = axes[cond_idx]
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        frac_key = f'{cond}_{stim_size}_{stim_duration}_{frac_suffix}'

        if frac_key not in heatmap_data:
            ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
            continue

        traces = heatmap_data[frac_key]
        x_cond = x[:traces.shape[1]] if len(x) > traces.shape[1] else x
        n_pts = len(x_cond)
        n_total = traces.shape[0]
        n_batches = n_total // batch_size

        for b in range(n_batches):
            batch_trace = np.nanmean(traces[b * batch_size:(b + 1) * batch_size, :n_pts], axis=0)
            label = f'Batch {b+1}' if b == 0 else None
            ax.plot(x_cond, batch_trace, color=color, alpha=0.5, linewidth=1.0, label=label)

        mean_trace = np.nanmean(traces[:, :n_pts], axis=0)
        ax.plot(x_cond, mean_trace, color='black', linewidth=2.0, label=f'Mean (n={n_reps})')

        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(stim_off_x, color='gray', linestyle='--', linewidth=0.8)

        ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
        ax.set_xlabel(x_label)
        if cond_idx == 0:
            ax.set_ylabel('Fraction active')
        if use_seconds:
            ax.set_xlim(-8, 10.5)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylim(0, y_max)

    fig.suptitle(f'Batch-Averaged Fraction Active ({crop_label}) [{stim_mode_label}] '
                 f'— size={stim_size}, dur={dur_label} (batches of {batch_size}, n={n_reps} reps)',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()

    time_suffix = '_seconds' if use_seconds else ''
    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Fraction_Active'),
                 f'batch{batch_size}_frac_{crop}_size_{stim_size}_dur_{stim_duration}{time_suffix}')


def plot_mean_blob_area(heatmap_data, stim_size, stim_duration, output_dir,
                        stim_mode_label='clamped'):
    """Mean + SEM connected blob area over time (no individual traces)."""
    spag_key = f'__spaghetti_indices_{stim_duration}__'
    if spag_key not in heatmap_data:
        return

    frame_indices = heatmap_data[spag_key].tolist()
    n_reps = heatmap_data.get('__n_reps__', '?')
    x = np.array(frame_indices) - PRE_STIM_FRAMES

    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4.5 * len(CONDITIONS), 4),
                             sharey=True)
    if len(CONDITIONS) == 1:
        axes = [axes]

    for cond_idx, cond in enumerate(CONDITIONS):
        ax = axes[cond_idx]
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        blob_key = f'{cond}_{stim_size}_{stim_duration}_blob_area'

        if blob_key not in heatmap_data:
            ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
            continue

        traces = heatmap_data[blob_key]
        x_cond = x[:traces.shape[1]] if len(x) > traces.shape[1] else x
        n_pts = len(x_cond)
        mean_trace = np.nanmean(traces[:, :n_pts], axis=0)
        n_valid = np.sum(~np.isnan(traces[:, :n_pts]), axis=0)
        sem_trace = np.nanstd(traces[:, :n_pts], axis=0) / np.sqrt(np.maximum(n_valid, 1))

        ax.fill_between(x_cond, mean_trace - sem_trace, mean_trace + sem_trace,
                         color=color, alpha=0.2)
        ax.plot(x_cond, mean_trace, color=color, linewidth=2.0, label=f'Mean (n={n_reps})')

        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(stim_duration, color='gray', linestyle='--', linewidth=0.8)

        ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
        ax.set_xlabel('Frame (rel. to stim onset)')
        if cond_idx == 0:
            ax.set_ylabel('Blob area (pixels)')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Mean Blob Area [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (n={n_reps} reps)',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Blob_Area'),
                 f'mean_blob_area_size_{stim_size}_dur_{stim_duration}')


def plot_batch_blob_area(heatmap_data, stim_size, stim_duration, output_dir,
                         batch_size=100, stim_mode_label='clamped'):
    """Batch-averaged connected blob area over time (each batch = mean of batch_size reps)."""
    spag_key = f'__spaghetti_indices_{stim_duration}__'
    if spag_key not in heatmap_data:
        return

    frame_indices = heatmap_data[spag_key].tolist()
    n_reps = heatmap_data.get('__n_reps__', '?')
    x = np.array(frame_indices) - PRE_STIM_FRAMES

    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4.5 * len(CONDITIONS), 4),
                             sharey=True)
    if len(CONDITIONS) == 1:
        axes = [axes]

    for cond_idx, cond in enumerate(CONDITIONS):
        ax = axes[cond_idx]
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        blob_key = f'{cond}_{stim_size}_{stim_duration}_blob_area'

        if blob_key not in heatmap_data:
            ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
            continue

        traces = heatmap_data[blob_key]
        x_cond = x[:traces.shape[1]] if len(x) > traces.shape[1] else x
        n_pts = len(x_cond)
        n_total = traces.shape[0]
        n_batches = n_total // batch_size

        for b in range(n_batches):
            batch_trace = np.nanmean(traces[b * batch_size:(b + 1) * batch_size, :n_pts], axis=0)
            label = f'Batch {b+1}' if b == 0 else None
            ax.plot(x_cond, batch_trace, color=color, alpha=0.5, linewidth=1.0, label=label)

        mean_trace = np.nanmean(traces[:, :n_pts], axis=0)
        ax.plot(x_cond, mean_trace, color='black', linewidth=2.0, label=f'Mean (n={n_reps})')

        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(stim_duration, color='gray', linestyle='--', linewidth=0.8)

        ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
        ax.set_xlabel('Frame (rel. to stim onset)')
        if cond_idx == 0:
            ax.set_ylabel('Blob area (pixels)')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Batch-Averaged Blob Area [{stim_mode_label}] '
                 f'— size={stim_size}, dur={stim_duration} (batches of {batch_size}, n={n_reps} reps)',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Blob_Area'),
                 f'batch{batch_size}_blob_area_size_{stim_size}_dur_{stim_duration}')


def plot_peak_blob_spatial(heatmap_data, stim_size, stim_duration, output_dir,
                           stim_mode_label='clamped', clim=None):
    """P(active) heatmap averaged at each replicate's peak-blob frame.

    clim : None, 'auto', or (vmin, vmax)
        None  → [0, 1] (default).
        'auto' → global min/max across all conditions.
        tuple  → explicit (vmin, vmax).
    """
    center_row = L // 2
    center_col = M // 2
    stim_rows, stim_cols = get_stimulus_region(stim_size, center_row, center_col, L, M)
    stim_r0, stim_r1 = int(stim_rows[0]), int(stim_rows[-1])
    stim_c0, stim_c1 = int(stim_cols[0]), int(stim_cols[-1])

    n_reps = heatmap_data.get('__n_reps__', '?')

    # Collect probability maps for all conditions
    prob_maps = {}
    for cond in CONDITIONS:
        prob_key = f'{cond}_{stim_size}_{stim_duration}_peak_blob_prob'
        if prob_key in heatmap_data:
            prob_maps[cond] = heatmap_data[prob_key]

    # Determine colour limits
    if clim == 'auto' and prob_maps:
        all_vals = np.concatenate([p.ravel() for p in prob_maps.values()])
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    elif isinstance(clim, (tuple, list)) and len(clim) == 2:
        vmin, vmax = float(clim[0]), float(clim[1])
    else:
        vmin, vmax = 0, 1

    # Filename suffix
    if clim == 'auto':
        clim_suffix = '_clim_auto'
        clim_label = f'clim=auto [{vmin:.2f}, {vmax:.2f}]'
    elif isinstance(clim, (tuple, list)):
        clim_suffix = f'_clim_{clim[0]}_{clim[1]}'
        clim_label = f'clim=[{vmin}, {vmax}]'
    else:
        clim_suffix = ''
        clim_label = ''

    fig, axes = plt.subplots(2, len(CONDITIONS),
                             figsize=(3.5 * len(CONDITIONS), 6))
    if len(CONDITIONS) == 1:
        axes = axes[:, np.newaxis]

    for cond_idx, cond in enumerate(CONDITIONS):
        for row in range(2):
            ax = axes[row, cond_idx]
            color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))

            if cond not in prob_maps:
                ax.set_facecolor('#eeeeee')
                if row == 0:
                    ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            prob = prob_maps[cond]

            if row == 0:
                data = prob
            else:
                data = prob[CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]

            ax.imshow(data, cmap='Greys', interpolation='nearest',
                      origin='upper', aspect='equal', vmin=vmin, vmax=vmax)

            # Stimulus region rectangle
            if row == 0:
                sr0, sr1, sc0, sc1 = stim_r0, stim_r1, stim_c0, stim_c1
            else:
                sr0 = stim_r0 - CROP_ROW_START
                sr1 = stim_r1 - CROP_ROW_START
                sc0 = stim_c0 - CROP_COL_START
                sc1 = stim_c1 - CROP_COL_START
            rect_stim = Rectangle(
                (sc0 - 0.5, sr0 - 0.5),
                sc1 - sc0 + 1, sr1 - sr0 + 1,
                linewidth=1.0, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect_stim)

            if row == 0:
                rect_crop = Rectangle(
                    (CROP_COL_START - 0.5, CROP_ROW_START - 0.5),
                    CROP_COL_END - CROP_COL_START, CROP_ROW_END - CROP_ROW_START,
                    linewidth=0.8, edgecolor='gray', facecolor='none', linestyle='--')
                ax.add_patch(rect_crop)

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
            if cond_idx == 0:
                ax.set_ylabel('Full grid' if row == 0 else 'FOV crop',
                              fontsize=9, fontweight='bold')

    title = f'P(active) at Peak Blob [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (n={n_reps} reps)'
    if clim_label:
        title += f'  {clim_label}'
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Peak_Probability_Maps'),
                 f'peak_blob_spatial_size_{stim_size}_dur_{stim_duration}{clim_suffix}')


def _annotate_effect_size(ax, datasets, conditions):
    """Add Cohen's d and p-value annotation for Expert vs NoSpout comparison."""
    from scipy.stats import mannwhitneyu
    cond_data = dict(zip(conditions, datasets))
    if 'Expert' not in cond_data or 'NoSpout' not in cond_data:
        return
    e, n = cond_data['Expert'], cond_data['NoSpout']
    if len(e) < 2 or len(n) < 2:
        return
    diff = np.mean(e) - np.mean(n)
    pooled_std = np.sqrt((np.var(e) + np.var(n)) / 2)
    d = diff / pooled_std if pooled_std > 0 else 0.0
    _, p = mannwhitneyu(e, n, alternative='two-sided')
    p_str = f'p={p:.1e}' if p < 0.001 else f'p={p:.3f}'
    ax.text(0.98, 0.97, f'd={d:+.2f}, {p_str}',
            transform=ax.transAxes, fontsize=7, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def _adaptive_swarmplot(ax, x_pos, values, color, alpha=0.35, sizes=(4, 3, 2)):
    """Swarmplot with adaptive dot size to avoid gutter stacking."""
    import warnings
    for sz in sizes:
        # Only remove collections added by this attempt, not previous conditions
        n_before = len(ax.collections)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sns.swarmplot(x=[x_pos] * len(values), y=values, color=color,
                          alpha=alpha, size=sz, ax=ax, warn_thresh=0.85)
            guttered = any("points cannot be placed" in str(wi.message) for wi in w)
        if not guttered:
            break
        # Remove only what this attempt added before retrying
        while len(ax.collections) > n_before:
            ax.collections[-1].remove()


def plot_peak_blob_scatter(heatmap_data, stim_size, stim_duration, output_dir,
                           max_reps=None, stim_mode_label='clamped',
                           conditions=None):
    """Scatter plot of peak blob size per replicate, grouped by condition."""
    if conditions is None:
        conditions = CONDITIONS
    cond_sub, cond_tag = _cond_subfolder(conditions)
    total_reps = heatmap_data.get('__n_reps__', '?')
    display_reps = max_reps if max_reps is not None else total_reps

    fig, ax = plt.subplots(figsize=(max(3.5, 1.5 * len(conditions)), 4.5))

    plotted_conds = []
    plotted_datasets = []
    for cond_idx, cond in enumerate(conditions):
        peak_key = f'{cond}_{stim_size}_{stim_duration}_peak_blob_size'
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        if peak_key not in heatmap_data:
            continue
        values = heatmap_data[peak_key]
        if max_reps is not None:
            values = values[:max_reps]
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            continue

        _adaptive_swarmplot(ax, cond_idx, valid, color)
        mean_val = np.mean(valid)
        sem_val = np.std(valid) / np.sqrt(len(valid))
        ax.errorbar(cond_idx, mean_val, yerr=sem_val, color='black',
                    fmt='_', markersize=12, markeredgewidth=2, capsize=4,
                    linewidth=1.5)
        plotted_conds.append(cond)
        plotted_datasets.append(valid)

    if plotted_datasets:
        _annotate_effect_size(ax, plotted_datasets, plotted_conds)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9, fontweight='bold')
    for tick_idx, cond in enumerate(conditions):
        ax.get_xticklabels()[tick_idx].set_color(
            CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5)))
    ax.set_ylabel('Peak blob area (pixels)')
    ax.grid(True, alpha=0.3, axis='y')

    ax.set_title(f'Peak Blob Size [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (n={display_reps} reps)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    suffix = f'_n{max_reps}' if max_reps is not None else ''
    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'peak_blob', 'Propagation_Scatter', f'size_{stim_size}', cond_sub),
                 f'peak_blob_scatter_size_{stim_size}_dur_{stim_duration}{suffix}{cond_tag}')


def plot_peak_blob_violin(heatmap_data, stim_size, stim_duration, output_dir,
                          stim_mode_label='clamped', conditions=None):
    """Violin plot of peak blob size per replicate, grouped by condition."""
    if conditions is None:
        conditions = CONDITIONS
    cond_sub, cond_tag = _cond_subfolder(conditions)
    total_reps = heatmap_data.get('__n_reps__', '?')

    fig, ax = plt.subplots(figsize=(max(3.5, 1.5 * len(conditions)), 4.5))

    positions = []
    datasets = []
    colors = []
    for cond_idx, cond in enumerate(conditions):
        peak_key = f'{cond}_{stim_size}_{stim_duration}_peak_blob_size'
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        if peak_key not in heatmap_data:
            continue
        values = heatmap_data[peak_key]
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            continue
        positions.append(cond_idx)
        datasets.append(valid)
        colors.append(color)

    if datasets:
        parts = ax.violinplot(datasets, positions=positions, showmeans=False,
                              showmedians=False, showextrema=False)
        for body, color in zip(parts['bodies'], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)
        for pos, valid, color in zip(positions, datasets, colors):
            mean_val = np.mean(valid)
            sem_val = np.std(valid) / np.sqrt(len(valid))
            ax.errorbar(pos, mean_val, yerr=sem_val, color='black',
                        fmt='_', markersize=12, markeredgewidth=2, capsize=4,
                        linewidth=1.5)
        all_vals = np.concatenate(datasets)
        y_cap = max(np.percentile(all_vals, 99) * 1.15, 1.0)
        ax.set_ylim(0, y_cap)
        _annotate_effect_size(ax, datasets, [conditions[p] for p in positions])

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9, fontweight='bold')
    for tick_idx, cond in enumerate(conditions):
        ax.get_xticklabels()[tick_idx].set_color(
            CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5)))
    ax.set_ylabel('Peak blob area (pixels)')
    ax.grid(True, alpha=0.3, axis='y')

    ax.set_title(f'Peak Blob Size [violin] [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (n={total_reps} reps)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'peak_blob', 'Propagation_Violin', cond_sub),
                 f'peak_blob_violin_size_{stim_size}_dur_{stim_duration}{cond_tag}')


def plot_peak_blob_box(heatmap_data, stim_size, stim_duration, output_dir,
                       stim_mode_label='clamped', conditions=None):
    """Box plot of peak blob size per replicate, grouped by condition."""
    if conditions is None:
        conditions = CONDITIONS
    cond_sub, cond_tag = _cond_subfolder(conditions)
    total_reps = heatmap_data.get('__n_reps__', '?')

    fig, ax = plt.subplots(figsize=(max(3.5, 1.5 * len(conditions)), 4.5))

    positions = []
    datasets = []
    colors = []
    for cond_idx, cond in enumerate(conditions):
        peak_key = f'{cond}_{stim_size}_{stim_duration}_peak_blob_size'
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        if peak_key not in heatmap_data:
            continue
        values = heatmap_data[peak_key]
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            continue
        positions.append(cond_idx)
        datasets.append(valid)
        colors.append(color)

    if datasets:
        bp = ax.boxplot(datasets, positions=positions, widths=0.5,
                        whis=1.5, showfliers=False, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        _annotate_effect_size(ax, datasets, [conditions[p] for p in positions])

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9, fontweight='bold')
    for tick_idx, cond in enumerate(conditions):
        ax.get_xticklabels()[tick_idx].set_color(
            CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5)))
    ax.set_ylabel('Peak blob area (pixels)')
    ax.grid(True, alpha=0.3, axis='y')

    ax.set_title(f'Peak Blob Size [box] [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (n={total_reps} reps)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'peak_blob', 'Propagation_Boxplot', cond_sub),
                 f'peak_blob_box_size_{stim_size}_dur_{stim_duration}{cond_tag}')


def plot_blob_area_subwindow_box(heatmap_data, stim_size, stim_duration, output_dir,
                                 window='first_10pct', stim_mode_label='clamped',
                                 conditions=None, metric='blob_area'):
    """Box plot of a metric in a sub-window of the stimulus period."""
    if conditions is None:
        conditions = CONDITIONS
    cond_sub, cond_tag = _cond_subfolder(conditions)
    ylabel_peak, ylabel_mean, title_pfx, fname_pfx, metric_sub = _metric_config(metric)
    total_reps = heatmap_data.get('__n_reps__', '?')

    spag_key = f'__spaghetti_indices_{stim_duration}__'
    if spag_key not in heatmap_data:
        return
    spaghetti_indices = np.asarray(heatmap_data[spag_key])

    stim_mask = (spaghetti_indices >= PRE_STIM_FRAMES) & \
                (spaghetti_indices < PRE_STIM_FRAMES + stim_duration)
    stim_col_indices = np.where(stim_mask)[0]
    n_stim = len(stim_col_indices)
    if n_stim == 0:
        return

    window_cfg = {
        'first_10pct': ('First 10% peak',  'first10pct', 0.10, 'peak'),
        'first_20pct': ('First 20% peak',  'first20pct', 0.20, 'peak'),
        'last_10pct':  ('Last 10% peak',   'last10pct',  0.10, 'peak'),
        'last_20pct':  ('Last 20% peak',   'last20pct',  0.20, 'peak'),
        'peak_all':    ('Peak (full stim)', 'peak_all',   1.00, 'peak'),
        'mean_all':    ('Mean (full stim)', 'mean_all',   1.00, 'mean'),
    }
    if window not in window_cfg:
        return
    title_label, fname_tag, frac, agg = window_cfg[window]

    n_sel = max(1, round(frac * n_stim))
    if window.startswith('first'):
        sel_cols = stim_col_indices[:n_sel]
    elif window.startswith('last'):
        sel_cols = stim_col_indices[-n_sel:]
    else:
        sel_cols = stim_col_indices

    fig, ax = plt.subplots(figsize=(max(3.5, 1.5 * len(conditions)), 4.5))

    positions = []
    datasets = []
    colors = []
    for cond_idx, cond in enumerate(conditions):
        data_key = f'{cond}_{stim_size}_{stim_duration}_{metric}'
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        if data_key not in heatmap_data:
            continue
        data = heatmap_data[data_key]
        valid_cols = sel_cols[sel_cols < data.shape[1]]
        if len(valid_cols) == 0:
            continue
        sub = data[:, valid_cols]
        values = np.nanmax(sub, axis=1) if agg == 'peak' else np.nanmean(sub, axis=1)
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            continue
        positions.append(cond_idx)
        datasets.append(valid)
        colors.append(color)

    if datasets:
        bp = ax.boxplot(datasets, positions=positions, widths=0.5,
                        whis=1.5, showfliers=False, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        _annotate_effect_size(ax, datasets, [conditions[p] for p in positions])

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9, fontweight='bold')
    for tick_idx, cond in enumerate(conditions):
        ax.get_xticklabels()[tick_idx].set_color(
            CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5)))
    ax.set_ylabel(ylabel_peak if agg == 'peak' else ylabel_mean)
    ax.grid(True, alpha=0.3, axis='y')

    ax.set_title(f'{title_pfx} [box] [{stim_mode_label}] — size={stim_size}, '
                 f'dur={stim_duration} (n={total_reps} reps)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'subwindow', window, 'Propagation_Boxplot', metric_sub, cond_sub),
                 f'{fname_pfx}_box_{fname_tag}_size_{stim_size}_dur_{stim_duration}{cond_tag}')


def plot_blob_area_subwindow_violin(heatmap_data, stim_size, stim_duration, output_dir,
                                     window='first_10pct', stim_mode_label='clamped',
                                     conditions=None, metric='blob_area'):
    """Violin plot of a metric in a sub-window of the stimulus period."""
    if conditions is None:
        conditions = CONDITIONS
    cond_sub, cond_tag = _cond_subfolder(conditions)
    ylabel_peak, ylabel_mean, title_pfx, fname_pfx, metric_sub = _metric_config(metric)
    total_reps = heatmap_data.get('__n_reps__', '?')

    spag_key = f'__spaghetti_indices_{stim_duration}__'
    if spag_key not in heatmap_data:
        return
    spaghetti_indices = np.asarray(heatmap_data[spag_key])

    stim_mask = (spaghetti_indices >= PRE_STIM_FRAMES) & \
                (spaghetti_indices < PRE_STIM_FRAMES + stim_duration)
    stim_col_indices = np.where(stim_mask)[0]
    n_stim = len(stim_col_indices)
    if n_stim == 0:
        return

    window_cfg = {
        'first_10pct': ('First 10% peak',  'first10pct', 0.10, 'peak'),
        'first_20pct': ('First 20% peak',  'first20pct', 0.20, 'peak'),
        'last_10pct':  ('Last 10% peak',   'last10pct',  0.10, 'peak'),
        'last_20pct':  ('Last 20% peak',   'last20pct',  0.20, 'peak'),
        'peak_all':    ('Peak (full stim)', 'peak_all',   1.00, 'peak'),
        'mean_all':    ('Mean (full stim)', 'mean_all',   1.00, 'mean'),
    }
    if window not in window_cfg:
        return
    title_label, fname_tag, frac, agg = window_cfg[window]

    n_sel = max(1, round(frac * n_stim))
    if window.startswith('first'):
        sel_cols = stim_col_indices[:n_sel]
    elif window.startswith('last'):
        sel_cols = stim_col_indices[-n_sel:]
    else:
        sel_cols = stim_col_indices

    fig, ax = plt.subplots(figsize=(max(3.5, 1.5 * len(conditions)), 4.5))

    positions = []
    datasets = []
    colors = []
    for cond_idx, cond in enumerate(conditions):
        data_key = f'{cond}_{stim_size}_{stim_duration}_{metric}'
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        if data_key not in heatmap_data:
            continue
        data = heatmap_data[data_key]
        valid_cols = sel_cols[sel_cols < data.shape[1]]
        if len(valid_cols) == 0:
            continue
        sub = data[:, valid_cols]
        values = np.nanmax(sub, axis=1) if agg == 'peak' else np.nanmean(sub, axis=1)
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            continue
        positions.append(cond_idx)
        datasets.append(valid)
        colors.append(color)

    if datasets:
        parts = ax.violinplot(datasets, positions=positions, showmeans=False,
                              showmedians=False, showextrema=False)
        for body, color in zip(parts['bodies'], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)
        for pos, valid, color in zip(positions, datasets, colors):
            mean_val = np.mean(valid)
            sem_val = np.std(valid) / np.sqrt(len(valid))
            ax.errorbar(pos, mean_val, yerr=sem_val, color='black',
                        fmt='_', markersize=12, markeredgewidth=2, capsize=4,
                        linewidth=1.5)
        all_vals = np.concatenate(datasets)
        y_cap = max(np.percentile(all_vals, 99) * 1.15, 1.0)
        ax.set_ylim(0, y_cap)
        _annotate_effect_size(ax, datasets, [conditions[p] for p in positions])

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9, fontweight='bold')
    for tick_idx, cond in enumerate(conditions):
        ax.get_xticklabels()[tick_idx].set_color(
            CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5)))
    ax.set_ylabel(ylabel_peak if agg == 'peak' else ylabel_mean)
    ax.grid(True, alpha=0.3, axis='y')

    ax.set_title(f'{title_pfx} [violin] [{stim_mode_label}] — size={stim_size}, '
                 f'dur={stim_duration} (n={total_reps} reps)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'subwindow', window, 'Propagation_Violin', metric_sub, cond_sub),
                 f'{fname_pfx}_violin_{fname_tag}_size_{stim_size}_dur_{stim_duration}{cond_tag}')


def plot_blob_area_subwindow_scatter(heatmap_data, stim_size, stim_duration, output_dir,
                                      window='first_10pct', max_reps=None,
                                      stim_mode_label='clamped', conditions=None,
                                      metric='blob_area'):
    """Scatter plot of a metric in a sub-window of the stimulus period."""
    if conditions is None:
        conditions = CONDITIONS
    cond_sub, cond_tag = _cond_subfolder(conditions)
    ylabel_peak, ylabel_mean, title_pfx, fname_pfx, metric_sub = _metric_config(metric)
    total_reps = heatmap_data.get('__n_reps__', '?')
    display_reps = max_reps if max_reps is not None else total_reps

    spag_key = f'__spaghetti_indices_{stim_duration}__'
    if spag_key not in heatmap_data:
        return
    spaghetti_indices = np.asarray(heatmap_data[spag_key])

    stim_mask = (spaghetti_indices >= PRE_STIM_FRAMES) & \
                (spaghetti_indices < PRE_STIM_FRAMES + stim_duration)
    stim_col_indices = np.where(stim_mask)[0]
    n_stim = len(stim_col_indices)
    if n_stim == 0:
        return

    window_cfg = {
        'first_10pct': ('First 10% peak',  'first10pct', 0.10, 'peak'),
        'first_20pct': ('First 20% peak',  'first20pct', 0.20, 'peak'),
        'last_10pct':  ('Last 10% peak',   'last10pct',  0.10, 'peak'),
        'last_20pct':  ('Last 20% peak',   'last20pct',  0.20, 'peak'),
        'peak_all':    ('Peak (full stim)', 'peak_all',   1.00, 'peak'),
        'mean_all':    ('Mean (full stim)', 'mean_all',   1.00, 'mean'),
    }
    if window not in window_cfg:
        return
    title_label, fname_tag, frac, agg = window_cfg[window]

    n_sel = max(1, round(frac * n_stim))
    if window.startswith('first'):
        sel_cols = stim_col_indices[:n_sel]
    elif window.startswith('last'):
        sel_cols = stim_col_indices[-n_sel:]
    else:
        sel_cols = stim_col_indices

    fig, ax = plt.subplots(figsize=(max(3.5, 1.5 * len(conditions)), 4.5))

    plotted_conds = []
    plotted_datasets = []
    for cond_idx, cond in enumerate(conditions):
        data_key = f'{cond}_{stim_size}_{stim_duration}_{metric}'
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        if data_key not in heatmap_data:
            continue
        data = heatmap_data[data_key]
        valid_cols = sel_cols[sel_cols < data.shape[1]]
        if len(valid_cols) == 0:
            continue
        sub = data[:, valid_cols]
        values = np.nanmax(sub, axis=1) if agg == 'peak' else np.nanmean(sub, axis=1)
        if max_reps is not None:
            values = values[:max_reps]
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            continue

        _adaptive_swarmplot(ax, cond_idx, valid, color)
        mean_val = np.mean(valid)
        sem_val = np.std(valid) / np.sqrt(len(valid))
        ax.errorbar(cond_idx, mean_val, yerr=sem_val, color='black',
                    fmt='_', markersize=12, markeredgewidth=2, capsize=4,
                    linewidth=1.5)
        plotted_conds.append(cond)
        plotted_datasets.append(valid)

    if plotted_datasets:
        _annotate_effect_size(ax, plotted_datasets, plotted_conds)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9, fontweight='bold')
    for tick_idx, cond in enumerate(conditions):
        ax.get_xticklabels()[tick_idx].set_color(
            CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5)))
    ax.set_ylabel(ylabel_peak if agg == 'peak' else ylabel_mean)
    ax.grid(True, alpha=0.3, axis='y')

    ax.set_title(f'{title_pfx} [scatter] [{stim_mode_label}] — size={stim_size}, '
                 f'dur={stim_duration} (n={display_reps} reps)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    suffix = f'_n{max_reps}' if max_reps is not None else ''
    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'subwindow', window, 'Propagation_Scatter', f'size_{stim_size}', metric_sub, cond_sub),
                 f'{fname_pfx}_scatter_{fname_tag}_size_{stim_size}_dur_{stim_duration}{suffix}{cond_tag}')


def _parse_clamp_width(stim_mode_label):
    """Extract clamp/pulse width from stim mode label.

    'double_pulse5'              -> 5  (clamp)
    'double_pulse_bias5'         -> 5  (bias-pulse)
    'double_pulse_bias10_2p00'   -> 10 (bias-encoded; ignore '_<bias>' suffix)
    'clamped'                    -> 1
    """
    # Strip bias suffix (e.g. '_2p00') if present — bias-encoded modes
    # append '_<biasLabel>' to the family name.
    base = stim_mode_label.split('_')
    if len(base) >= 2 and base[-1].count('p') == 1 and base[-1].replace('p', '').isdigit():
        base = '_'.join(base[:-1])
    else:
        base = stim_mode_label
    if base.startswith('double_pulse_bias'):
        suffix = base[len('double_pulse_bias'):]
        return int(suffix) if suffix else 1
    if base.startswith('double_pulse'):
        suffix = base[len('double_pulse'):]
        return int(suffix) if suffix else 1
    return 1


FREE_DYNAMICS_MODES = {
    # mode_key:         (subfolder_name,             title_tag)
    'both':             ('both_with_buffer',         'both+buf'),
    'no_buffer':        ('no_buffer',                'no buf'),
    'inter_pulse':      ('inter_pulse',              'inter-pulse'),
    'inter_pulse_full': ('inter_pulse_with_buffer',  'inter-pulse+buf'),
    'post_stim':        ('post_stim',                'post-stim'),
    'post_stim_full':   ('post_stim_with_buffer',    'post-stim+buf'),
}


def _get_free_dynamics_mask(spaghetti_indices, stim_duration, buffer=15,
                            post_stim_frames=25, clamp_width=1,
                            window_mode='both'):
    """Return boolean mask over spaghetti columns for free-dynamics frames.

    Window modes:
        both             - inter-pulse + post-stim, both with buffer
        no_buffer        - inter-pulse + post-stim, no buffer
        inter_pulse      - inter-pulse only, with buffer
        inter_pulse_full - inter-pulse only, no buffer
        post_stim        - post-stim only, with buffer
        post_stim_full   - post-stim only, no buffer
    """
    rel = np.asarray(spaghetti_indices, dtype=np.float64) - PRE_STIM_FRAMES
    onset_end = float(clamp_width)
    offset_start = float(stim_duration - clamp_width)

    use_buffer = window_mode in ('both', 'inter_pulse', 'post_stim')
    buf = buffer if use_buffer else 0

    inter_pulse = (rel >= onset_end + buf) & (rel < offset_start - buf)

    post_start = float(stim_duration) + buf
    post_end = post_start + post_stim_frames
    post_window = (rel >= post_start) & (rel < post_end)

    if window_mode in ('inter_pulse', 'inter_pulse_full'):
        mask = inter_pulse
    elif window_mode in ('post_stim', 'post_stim_full'):
        mask = post_window
    else:  # 'both', 'no_buffer'
        mask = inter_pulse | post_window

    return mask, rel


def plot_free_dynamics_blob_box(heatmap_data, stim_size, stim_duration, output_dir,
                                 buffer=15, post_stim_frames=25,
                                 stim_mode_label='clamped', conditions=None,
                                 metric='blob_area', window_mode='both'):
    """Box plot of mean metric during free-dynamics frames."""
    if conditions is None:
        conditions = CONDITIONS
    cond_sub, cond_tag = _cond_subfolder(conditions)
    _, ylabel_mean, title_pfx, fname_pfx, metric_sub = _metric_config(metric)
    mode_subfolder, mode_tag = FREE_DYNAMICS_MODES[window_mode]
    total_reps = heatmap_data.get('__n_reps__', '?')
    spag_key = f'__spaghetti_indices_{stim_duration}__'
    if spag_key not in heatmap_data:
        return
    mask, rel = _get_free_dynamics_mask(
        heatmap_data[spag_key], stim_duration, buffer, post_stim_frames,
        clamp_width=_parse_clamp_width(stim_mode_label),
        window_mode=window_mode)
    if mask.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(max(3.5, 1.5 * len(conditions)), 4.5))

    positions = []
    datasets = []
    colors = []
    for cond_idx, cond in enumerate(conditions):
        data_key = f'{cond}_{stim_size}_{stim_duration}_{metric}'
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        if data_key not in heatmap_data:
            continue
        data = heatmap_data[data_key]
        m = mask[:data.shape[1]] if len(mask) > data.shape[1] else mask
        values = np.nanmean(data[:, m], axis=1)
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            continue
        positions.append(cond_idx)
        datasets.append(valid)
        colors.append(color)

    if datasets:
        bp = ax.boxplot(datasets, positions=positions, widths=0.5,
                        whis=1.5, showfliers=False, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        _annotate_effect_size(ax, datasets, [conditions[p] for p in positions])

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9, fontweight='bold')
    for tick_idx, cond in enumerate(conditions):
        ax.get_xticklabels()[tick_idx].set_color(
            CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5)))
    ax.set_ylabel(ylabel_mean)
    ax.grid(True, alpha=0.3, axis='y')

    ax.set_title(f'{title_pfx} [box] [{stim_mode_label}] — size={stim_size}, '
                 f'dur={stim_duration} (n={total_reps} reps)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'free_dynamics', mode_subfolder,
                  'Propagation_Boxplot', metric_sub, cond_sub),
                 f'free_dynamics_{fname_pfx}_box_size_{stim_size}_dur_{stim_duration}_{mode_subfolder}{cond_tag}')


def plot_free_dynamics_blob_violin(heatmap_data, stim_size, stim_duration, output_dir,
                                    buffer=15, post_stim_frames=25,
                                    stim_mode_label='clamped', conditions=None,
                                    metric='blob_area', window_mode='both'):
    """Violin plot of mean metric during free-dynamics frames."""
    if conditions is None:
        conditions = CONDITIONS
    cond_sub, cond_tag = _cond_subfolder(conditions)
    _, ylabel_mean, title_pfx, fname_pfx, metric_sub = _metric_config(metric)
    mode_subfolder, mode_tag = FREE_DYNAMICS_MODES[window_mode]
    total_reps = heatmap_data.get('__n_reps__', '?')
    spag_key = f'__spaghetti_indices_{stim_duration}__'
    if spag_key not in heatmap_data:
        return
    mask, rel = _get_free_dynamics_mask(
        heatmap_data[spag_key], stim_duration, buffer, post_stim_frames,
        clamp_width=_parse_clamp_width(stim_mode_label),
        window_mode=window_mode)
    if mask.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(max(3.5, 1.5 * len(conditions)), 4.5))

    positions = []
    datasets = []       # outlier-filtered (for violin shape)
    datasets_full = []  # unfiltered (for mean/SEM)
    colors = []
    for cond_idx, cond in enumerate(conditions):
        data_key = f'{cond}_{stim_size}_{stim_duration}_{metric}'
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        if data_key not in heatmap_data:
            continue
        data = heatmap_data[data_key]
        m = mask[:data.shape[1]] if len(mask) > data.shape[1] else mask
        values = np.nanmean(data[:, m], axis=1)
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            continue
        # IQR-based outlier removal for violin shape only
        q1, q3 = np.percentile(valid, [25, 75])
        iqr = q3 - q1
        valid_clean = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
        if len(valid_clean) == 0:
            valid_clean = valid
        positions.append(cond_idx)
        datasets.append(valid_clean)
        datasets_full.append(valid)
        colors.append(color)

    if datasets:
        parts = ax.violinplot(datasets, positions=positions, showmeans=False,
                              showmedians=False, showextrema=False)
        for body, color in zip(parts['bodies'], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)
        for pos, valid, color in zip(positions, datasets_full, colors):
            mean_val = np.mean(valid)
            sem_val = np.std(valid) / np.sqrt(len(valid))
            ax.errorbar(pos, mean_val, yerr=sem_val, color='black',
                        fmt='_', markersize=12, markeredgewidth=2, capsize=4,
                        linewidth=1.5)
        all_vals = np.concatenate(datasets)
        y_cap = np.percentile(all_vals, 99) * 1.15
        ax.set_ylim(0, y_cap)
        _annotate_effect_size(ax, datasets_full, [conditions[p] for p in positions])

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9, fontweight='bold')
    for tick_idx, cond in enumerate(conditions):
        ax.get_xticklabels()[tick_idx].set_color(
            CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5)))
    ax.set_ylabel(ylabel_mean)
    ax.grid(True, alpha=0.3, axis='y')

    ax.set_title(f'{title_pfx} [violin] [{stim_mode_label}] — size={stim_size}, '
                 f'dur={stim_duration} (n={total_reps} reps)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'free_dynamics', mode_subfolder,
                  'Propagation_Violin', metric_sub, cond_sub),
                 f'free_dynamics_{fname_pfx}_violin_size_{stim_size}_dur_{stim_duration}_{mode_subfolder}{cond_tag}')


def plot_free_dynamics_blob_scatter(heatmap_data, stim_size, stim_duration, output_dir,
                                     buffer=15, post_stim_frames=25, max_reps=None,
                                     stim_mode_label='clamped', conditions=None,
                                     metric='blob_area', window_mode='both'):
    """Scatter plot of mean metric during free-dynamics frames."""
    if conditions is None:
        conditions = CONDITIONS
    cond_sub, cond_tag = _cond_subfolder(conditions)
    _, ylabel_mean, title_pfx, fname_pfx, metric_sub = _metric_config(metric)
    mode_subfolder, mode_tag = FREE_DYNAMICS_MODES[window_mode]
    total_reps = heatmap_data.get('__n_reps__', '?')
    display_reps = max_reps if max_reps is not None else total_reps
    spag_key = f'__spaghetti_indices_{stim_duration}__'
    if spag_key not in heatmap_data:
        return
    mask, rel = _get_free_dynamics_mask(
        heatmap_data[spag_key], stim_duration, buffer, post_stim_frames,
        clamp_width=_parse_clamp_width(stim_mode_label),
        window_mode=window_mode)
    if mask.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(max(3.5, 1.5 * len(conditions)), 4.5))

    all_scatter_vals = []
    plotted_conds = []
    plotted_full = []
    for cond_idx, cond in enumerate(conditions):
        data_key = f'{cond}_{stim_size}_{stim_duration}_{metric}'
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        if data_key not in heatmap_data:
            continue
        data = heatmap_data[data_key]
        m = mask[:data.shape[1]] if len(mask) > data.shape[1] else mask
        values = np.nanmean(data[:, m], axis=1)
        if max_reps is not None:
            values = values[:max_reps]
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            continue

        # IQR-based outlier removal for scatter points
        q1, q3 = np.percentile(valid, [25, 75])
        iqr = q3 - q1
        valid_clean = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
        if len(valid_clean) == 0:
            valid_clean = valid

        _adaptive_swarmplot(ax, cond_idx, valid_clean, color)
        # Mean/SEM from full (unfiltered) data
        mean_val = np.mean(valid)
        sem_val = np.std(valid) / np.sqrt(len(valid))
        ax.errorbar(cond_idx, mean_val, yerr=sem_val, color='black',
                    fmt='_', markersize=12, markeredgewidth=2, capsize=4,
                    linewidth=1.5)
        all_scatter_vals.append(valid_clean)
        plotted_conds.append(cond)
        plotted_full.append(valid)

    if all_scatter_vals:
        all_vals = np.concatenate(all_scatter_vals)
        y_cap = np.percentile(all_vals, 99) * 1.15
        ax.set_ylim(0, y_cap)
        _annotate_effect_size(ax, plotted_full, plotted_conds)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9, fontweight='bold')
    for tick_idx, cond in enumerate(conditions):
        ax.get_xticklabels()[tick_idx].set_color(
            CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5)))
    ax.set_ylabel(ylabel_mean)
    ax.grid(True, alpha=0.3, axis='y')

    ax.set_title(f'{title_pfx} [scatter] [{stim_mode_label}] — size={stim_size}, '
                 f'dur={stim_duration} (n={display_reps} reps)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    suffix = f'_n{max_reps}' if max_reps is not None else ''
    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'free_dynamics', mode_subfolder,
                  'Propagation_Scatter', f'size_{stim_size}', metric_sub, cond_sub),
                 f'free_dynamics_{fname_pfx}_scatter_size_{stim_size}_dur_{stim_duration}_{mode_subfolder}{suffix}{cond_tag}')


def plot_radial_profile_peak(heatmap_data, stim_size, stim_duration, output_dir,
                              stim_mode_label='clamped'):
    """Radial distance profile of P(active) at peak-blob frame, all conditions overlaid."""
    center_row = L // 2
    center_col = M // 2

    # Distance from centre for every pixel
    rows, cols = np.meshgrid(np.arange(L), np.arange(M), indexing='ij')
    dist_map = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
    max_dist = int(np.ceil(dist_map.max()))
    bin_edges = np.arange(0, max_dist + 1, dtype=np.float64)
    bin_centres = bin_edges[:-1] + 0.5

    n_reps = heatmap_data.get('__n_reps__', '?')
    fig, ax = plt.subplots(figsize=(6, 4))
    any_plotted = False

    for cond in CONDITIONS:
        prob_key = f'{cond}_{stim_size}_{stim_duration}_peak_blob_prob'
        if prob_key not in heatmap_data:
            continue
        prob = heatmap_data[prob_key]
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))

        profile = np.full(len(bin_centres), np.nan)
        for b in range(len(bin_centres)):
            mask = (dist_map >= bin_edges[b]) & (dist_map < bin_edges[b + 1])
            if mask.any():
                profile[b] = np.nanmean(prob[mask])

        ax.plot(bin_centres, profile, color=color, linewidth=2.0, label=cond)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    # Stimulus edge marker
    stim_half = stim_size / 2.0
    ax.axvline(stim_half, color='red', linestyle='--', linewidth=1.0, label='Stim edge')

    ax.set_xlabel('Distance from centre (px)')
    ax.set_ylabel('P(active)')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(L, M) // 2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Radial Profile at Peak [{stim_mode_label}] — size={stim_size}, dur={stim_duration} '
                 f'(n={n_reps} reps)', fontsize=11, fontweight='bold')
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Radial_Profile'),
                 f'radial_profile_peak_size_{stim_size}_dur_{stim_duration}')


def plot_stim_avg_heatmap(heatmap_data, stim_size, stim_duration, output_dir,
                           stim_mode_label='clamped', clim=None):
    """P(active) heatmap averaged over the entire stimulus period.

    clim : None, 'auto', or (vmin, vmax)
        None  → [0, 1] (default).
        'auto' → global min/max across all conditions.
        tuple  → explicit (vmin, vmax).
    """
    center_row = L // 2
    center_col = M // 2
    stim_rows, stim_cols = get_stimulus_region(stim_size, center_row, center_col, L, M)
    stim_r0, stim_r1 = int(stim_rows[0]), int(stim_rows[-1])
    stim_c0, stim_c1 = int(stim_cols[0]), int(stim_cols[-1])

    frame_key = f'__frame_indices_{stim_duration}__'
    if frame_key not in heatmap_data:
        return
    frame_indices = heatmap_data[frame_key]
    stim_indices = [int(f) for f in frame_indices
                    if PRE_STIM_FRAMES <= f < PRE_STIM_FRAMES + stim_duration]
    if not stim_indices:
        return

    n_reps = heatmap_data.get('__n_reps__', '?')

    # Pre-compute stimulus-averaged maps for all conditions
    avg_maps = {}
    for cond in CONDITIONS:
        prefix = f'{cond}_{stim_size}_{stim_duration}'
        frames = []
        for fidx in stim_indices:
            key = f'{prefix}_f{fidx}'
            if key in heatmap_data:
                frames.append(heatmap_data[key])
        if frames:
            avg_maps[cond] = np.mean(frames, axis=0)

    # Determine colour limits
    if clim == 'auto' and avg_maps:
        all_vals = np.concatenate([p.ravel() for p in avg_maps.values()])
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    elif isinstance(clim, (tuple, list)) and len(clim) == 2:
        vmin, vmax = float(clim[0]), float(clim[1])
    else:
        vmin, vmax = 0, 1

    # Filename suffix
    if clim == 'auto':
        clim_suffix = '_clim_auto'
        clim_label = f'clim=auto [{vmin:.2f}, {vmax:.2f}]'
    elif isinstance(clim, (tuple, list)):
        clim_suffix = f'_clim_{clim[0]}_{clim[1]}'
        clim_label = f'clim=[{vmin}, {vmax}]'
    else:
        clim_suffix = ''
        clim_label = ''

    fig, axes = plt.subplots(2, len(CONDITIONS),
                             figsize=(3.5 * len(CONDITIONS), 6))
    if len(CONDITIONS) == 1:
        axes = axes[:, np.newaxis]

    for cond_idx, cond in enumerate(CONDITIONS):
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))

        for row in range(2):
            ax = axes[row, cond_idx]

            if cond not in avg_maps:
                ax.set_facecolor('#eeeeee')
                if row == 0:
                    ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            stim_avg = avg_maps[cond]

            if row == 0:
                data = stim_avg
            else:
                data = stim_avg[CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]

            ax.imshow(data, cmap='Greys', interpolation='nearest',
                      origin='upper', aspect='equal', vmin=vmin, vmax=vmax)

            # Stimulus rectangle
            if row == 0:
                sr0, sr1, sc0, sc1 = stim_r0, stim_r1, stim_c0, stim_c1
            else:
                sr0 = stim_r0 - CROP_ROW_START
                sr1 = stim_r1 - CROP_ROW_START
                sc0 = stim_c0 - CROP_COL_START
                sc1 = stim_c1 - CROP_COL_START
            rect_stim = Rectangle(
                (sc0 - 0.5, sr0 - 0.5),
                sc1 - sc0 + 1, sr1 - sr0 + 1,
                linewidth=1.0, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect_stim)

            if row == 0:
                rect_crop = Rectangle(
                    (CROP_COL_START - 0.5, CROP_ROW_START - 0.5),
                    CROP_COL_END - CROP_COL_START, CROP_ROW_END - CROP_ROW_START,
                    linewidth=0.8, edgecolor='gray', facecolor='none', linestyle='--')
                ax.add_patch(rect_crop)

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(cond, fontsize=10, color=color, fontweight='bold')
            if cond_idx == 0:
                ax.set_ylabel('Full grid' if row == 0 else 'FOV crop',
                              fontsize=9, fontweight='bold')

    title = f'Stimulus-Averaged P(active) [{stim_mode_label}] — size={stim_size}, dur={stim_duration} (n={n_reps} reps)'
    if clim_label:
        title += f'  {clim_label}'
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Heatmaps'),
                 f'stim_avg_heatmap_size_{stim_size}_dur_{stim_duration}{clim_suffix}')


def plot_connected_blob_area_timeseries(heatmap_data, stim_size, stim_duration, output_dir,
                                         stim_mode_label='clamped'):
    """Mean ± SEM timeseries of connected blob area (raw 4-connectivity)."""
    spag_key = f'__spaghetti_indices_{stim_duration}__'
    if spag_key not in heatmap_data:
        return
    spaghetti_indices = heatmap_data[spag_key]
    x = np.array(spaghetti_indices, dtype=np.float64) - PRE_STIM_FRAMES

    n_reps = heatmap_data.get('__n_reps__', '?')
    fig, ax = plt.subplots(figsize=(8, 4))
    any_plotted = False

    for cond in CONDITIONS:
        area_key = f'{cond}_{stim_size}_{stim_duration}_blob_area'
        if area_key not in heatmap_data:
            continue
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        data = heatmap_data[area_key]
        mean_val = np.nanmean(data, axis=0)
        n_valid = np.sum(~np.isnan(data), axis=0).clip(1)
        sem_val = np.nanstd(data, axis=0) / np.sqrt(n_valid)

        ax.plot(x, mean_val, color=color, linewidth=2.0, label=cond)
        ax.fill_between(x, mean_val - sem_val, mean_val + sem_val,
                        color=color, alpha=0.15)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.axvspan(0, stim_duration, color='gold', alpha=0.12)
    ax.set_xlabel('Frame (relative to stim onset)')
    ax.set_ylabel('Connected blob area (pixels)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Connected Blob Area [{stim_mode_label}] — size={stim_size}, dur={stim_duration} '
                 f'(n={n_reps} reps)', fontsize=11, fontweight='bold')
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Centroid_Timeseries'),
                 f'connected_blob_area_size_{stim_size}_dur_{stim_duration}')


def plot_gauss_blob_area_timeseries(heatmap_data, stim_size, stim_duration, output_dir,
                                     stim_mode_label='clamped'):
    """Mean ± SEM timeseries of Gaussian blob area (smoothed + thresholded)."""
    frame_key = f'__frame_indices_{stim_duration}__'
    if frame_key not in heatmap_data:
        return
    frame_indices = heatmap_data[frame_key]
    x = np.array(frame_indices, dtype=np.float64) - PRE_STIM_FRAMES

    n_reps = heatmap_data.get('__n_reps__', '?')
    fig, ax = plt.subplots(figsize=(8, 4))
    any_plotted = False

    for cond in CONDITIONS:
        area_key = f'{cond}_{stim_size}_{stim_duration}_gauss_blob_area'
        if area_key not in heatmap_data:
            continue
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        data = heatmap_data[area_key]
        mean_val = np.nanmean(data, axis=0)
        n_valid = np.sum(~np.isnan(data), axis=0).clip(1)
        sem_val = np.nanstd(data, axis=0) / np.sqrt(n_valid)

        ax.plot(x, mean_val, color=color, linewidth=2.0, label=cond)
        ax.fill_between(x, mean_val - sem_val, mean_val + sem_val,
                        color=color, alpha=0.15)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.axvspan(0, stim_duration, color='gold', alpha=0.12)
    ax.set_xlabel('Frame (relative to stim onset)')
    ax.set_ylabel('Gaussian blob area (pixels)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Gaussian Blob Area [{stim_mode_label}] — size={stim_size}, dur={stim_duration} '
                 f'(n={n_reps} reps)', fontsize=11, fontweight='bold')
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Centroid_Timeseries'),
                 f'gauss_blob_area_size_{stim_size}_dur_{stim_duration}')


def plot_frac_crop_timeseries(heatmap_data, stim_size, stim_duration, output_dir,
                               stim_mode_label='clamped'):
    """Mean ± SEM timeseries of active fraction in the FOV crop region."""
    spag_key = f'__spaghetti_indices_{stim_duration}__'
    if spag_key not in heatmap_data:
        return
    spaghetti_indices = heatmap_data[spag_key]
    x = np.array(spaghetti_indices, dtype=np.float64) - PRE_STIM_FRAMES

    n_reps = heatmap_data.get('__n_reps__', '?')
    fig, ax = plt.subplots(figsize=(8, 4))
    any_plotted = False

    for cond in CONDITIONS:
        frac_key = f'{cond}_{stim_size}_{stim_duration}_frac_crop'
        if frac_key not in heatmap_data:
            continue
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))
        data = heatmap_data[frac_key]
        mean_val = np.nanmean(data, axis=0)
        n_valid = np.sum(~np.isnan(data), axis=0).clip(1)
        sem_val = np.nanstd(data, axis=0) / np.sqrt(n_valid)

        ax.plot(x, mean_val, color=color, linewidth=2.0, label=cond)
        ax.fill_between(x, mean_val - sem_val, mean_val + sem_val,
                        color=color, alpha=0.15)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.axvspan(0, stim_duration, color='gold', alpha=0.12)
    ax.set_xlabel('Frame (relative to stim onset)')
    ax.set_ylabel('Active fraction (FOV crop)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Active Fraction [FOV crop] [{stim_mode_label}] — size={stim_size}, dur={stim_duration} '
                 f'(n={n_reps} reps)', fontsize=11, fontweight='bold')
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Centroid_Timeseries'),
                 f'frac_crop_timeseries_size_{stim_size}_dur_{stim_duration}')


def plot_centroid_timeseries(heatmap_data, stim_size, stim_duration, output_dir,
                              stim_mode_label='clamped'):
    """Temporal evolution of centroid offset magnitude and direction."""
    frame_key = f'__frame_indices_{stim_duration}__'
    if frame_key not in heatmap_data:
        return
    frame_indices = heatmap_data[frame_key]
    x = np.array(frame_indices, dtype=np.float64) - PRE_STIM_FRAMES

    n_reps = heatmap_data.get('__n_reps__', '?')
    fig, ax_mag = plt.subplots(figsize=(8, 4))
    ax_dir = ax_mag.twinx()
    any_plotted = False

    for cond in CONDITIONS:
        prefix = f'{cond}_{stim_size}_{stim_duration}'
        mag_key = f'{prefix}_centroid_mag'
        dir_key = f'{prefix}_centroid_dir'
        if mag_key not in heatmap_data:
            continue
        color = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))

        # Magnitude: mean + SEM
        mag = heatmap_data[mag_key]  # (n_reps, n_timepoints)
        mean_mag = np.nanmean(mag, axis=0)
        n_valid = np.sum(~np.isnan(mag), axis=0).clip(1)
        sem_mag = np.nanstd(mag, axis=0) / np.sqrt(n_valid)

        ax_mag.plot(x, mean_mag, color=color, linewidth=2.0, label=cond)
        ax_mag.fill_between(x, mean_mag - sem_mag, mean_mag + sem_mag,
                            color=color, alpha=0.15)

        # Direction: circular mean → degrees
        if dir_key in heatmap_data:
            dirs = heatmap_data[dir_key]  # (n_reps, n_timepoints)
            mean_dir = np.degrees(np.arctan2(
                np.nanmean(np.sin(dirs), axis=0),
                np.nanmean(np.cos(dirs), axis=0)))
            ax_dir.plot(x, mean_dir, color=color, linewidth=1.2, linestyle='--', alpha=0.7)

        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    # Stimulus period shading
    ax_mag.axvspan(0, stim_duration, color='gold', alpha=0.12)

    ax_mag.set_xlabel('Frame (relative to stim onset)')
    ax_mag.set_ylabel('Centroid offset (px)')
    ax_dir.set_ylabel('Direction (°)', rotation=270, labelpad=15)
    ax_dir.set_ylim(-180, 180)
    ax_mag.legend(fontsize=8, loc='upper left')
    ax_mag.grid(True, alpha=0.3)
    ax_mag.set_title(f'Centroid Offset [{stim_mode_label}] — size={stim_size}, dur={stim_duration} '
                     f'(n={n_reps} reps)', fontsize=11, fontweight='bold')
    fig.tight_layout()

    _save_figure(fig, output_dir,
                 ('PerturbationSnapshots', f'dur_{stim_duration}', 'Centroid_Timeseries'),
                 f'centroid_timeseries_size_{stim_size}_dur_{stim_duration}')


def generate_all_heatmap_figures(heatmap_data, sizes, durations, output_dir,
                                 stim_mode_label='clamped',
                                 time_scale_factor=1.0, sampling_rate=10.0):
    """Generate all heatmap + asymmetry + spaghetti + blob figures."""
    n_combos = len(sizes) * len(durations)
    n_reps = heatmap_data.get('__n_reps__', 0)
    extra_scatter = 1 if (isinstance(n_reps, int) and n_reps > 50) else 0
    n_figs = n_combos * (60 + extra_scatter) + len(durations)
    print(f"Generating {n_figs} heatmap/asymmetry figures ...")

    sml = stim_mode_label
    for dur in durations:
        for size in sizes:
            print(f"  size={size}  dur={dur}")
            plot_heatmap_timeline(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_heatmap_crop_timeline(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_heatmap_expert_focus(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_heatmap_expert_nospout_focus(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_asymmetry_polar(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_asymmetry_sectors(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_asymmetry_polar_peak(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_asymmetry_sectors_peak(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_spaghetti_fraction(heatmap_data, size, dur, output_dir, crop='full', stim_mode_label=sml)
            plot_spaghetti_fraction(heatmap_data, size, dur, output_dir, crop='half', stim_mode_label=sml)
            plot_spaghetti_fraction(heatmap_data, size, dur, output_dir, crop='fov', stim_mode_label=sml)
            plot_spaghetti_blob_area(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_mean_fraction(heatmap_data, size, dur, output_dir, crop='full', stim_mode_label=sml)
            plot_mean_fraction(heatmap_data, size, dur, output_dir, crop='half', stim_mode_label=sml)
            plot_mean_fraction(heatmap_data, size, dur, output_dir, crop='fov', stim_mode_label=sml)
            plot_mean_blob_area(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            # Always produce frames version
            plot_batch_fraction(heatmap_data, size, dur, output_dir, crop='full', stim_mode_label=sml)
            plot_batch_fraction(heatmap_data, size, dur, output_dir, crop='half', stim_mode_label=sml)
            plot_batch_fraction(heatmap_data, size, dur, output_dir, crop='fov', stim_mode_label=sml)
            # Also produce seconds version when time scale is available
            if time_scale_factor != 1.0:
                plot_batch_fraction(heatmap_data, size, dur, output_dir, crop='full', stim_mode_label=sml,
                                    time_scale_factor=time_scale_factor, sampling_rate=sampling_rate)
                plot_batch_fraction(heatmap_data, size, dur, output_dir, crop='half', stim_mode_label=sml,
                                    time_scale_factor=time_scale_factor, sampling_rate=sampling_rate)
                plot_batch_fraction(heatmap_data, size, dur, output_dir, crop='fov', stim_mode_label=sml,
                                    time_scale_factor=time_scale_factor, sampling_rate=sampling_rate)
            plot_batch_blob_area(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_peak_blob_spatial(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_peak_blob_spatial(heatmap_data, size, dur, output_dir, stim_mode_label=sml, clim='auto')
            plot_peak_blob_spatial(heatmap_data, size, dur, output_dir, stim_mode_label=sml, clim=(0.3, 1))
            plot_peak_blob_spatial(heatmap_data, size, dur, output_dir, stim_mode_label=sml, clim=(0.5, 1))
            for conds in (None, EXPERT_NOSPOUT):
                plot_peak_blob_violin(heatmap_data, size, dur, output_dir,
                                      stim_mode_label=sml, conditions=conds)
                plot_peak_blob_box(heatmap_data, size, dur, output_dir,
                                    stim_mode_label=sml, conditions=conds)
                plot_peak_blob_scatter(heatmap_data, size, dur, output_dir,
                                       max_reps=50, stim_mode_label=sml, conditions=conds)
                for met in ('blob_area', 'blob_area_net', 'frac_crop'):
                    windows = ('first_10pct', 'first_20pct', 'last_10pct', 'last_20pct',
                               'peak_all', 'mean_all')
                    for window in windows:
                        plot_blob_area_subwindow_box(heatmap_data, size, dur, output_dir,
                                                     window=window, stim_mode_label=sml,
                                                     conditions=conds, metric=met)
                        for n_reps in (50, 100, 200):
                            plot_blob_area_subwindow_scatter(heatmap_data, size, dur, output_dir,
                                                              window=window, max_reps=n_reps,
                                                              stim_mode_label=sml,
                                                              conditions=conds, metric=met)
                        plot_blob_area_subwindow_violin(heatmap_data, size, dur, output_dir,
                                                         window=window, stim_mode_label=sml,
                                                         conditions=conds, metric=met)
                    for wm in FREE_DYNAMICS_MODES:
                        plot_free_dynamics_blob_box(heatmap_data, size, dur, output_dir,
                                                    stim_mode_label=sml, conditions=conds, metric=met, window_mode=wm)
                        for n_reps in (50, 100, 200):
                            plot_free_dynamics_blob_scatter(heatmap_data, size, dur, output_dir,
                                                            max_reps=n_reps, stim_mode_label=sml,
                                                            conditions=conds, metric=met, window_mode=wm)
                        plot_free_dynamics_blob_violin(heatmap_data, size, dur, output_dir,
                                                       stim_mode_label=sml, conditions=conds, metric=met, window_mode=wm)
            plot_radial_profile_peak(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_stim_avg_heatmap(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_stim_avg_heatmap(heatmap_data, size, dur, output_dir, stim_mode_label=sml, clim='auto')
            plot_stim_avg_heatmap(heatmap_data, size, dur, output_dir, stim_mode_label=sml, clim=(0.3, 1))
            plot_stim_avg_heatmap(heatmap_data, size, dur, output_dir, stim_mode_label=sml, clim=(0.5, 1))
            plot_centroid_timeseries(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_connected_blob_area_timeseries(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_gauss_blob_area_timeseries(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
            plot_frac_crop_timeseries(heatmap_data, size, dur, output_dir, stim_mode_label=sml)
        plot_asymmetry_summary(heatmap_data, dur, sizes, output_dir, stim_mode_label=sml)

    print("All heatmap/asymmetry figures generated.")


def generate_all_figures(snapshot_data, sizes, durations, output_dir,
                         stim_mode_label='clamped'):
    """Generate summary, detailed, expert replicate, and combined Expert+NoSpout figures."""
    total = len(sizes) * len(durations)
    has_reps = any(len(k) == 4 for k in snapshot_data)
    # 2 base + 1 expert reps + 3 expert+nospout combined + up to 40 single-rep = 46 per combo
    n_figs = total * 46 if has_reps else total * 2
    parts = f"{total} summary + {total} detailed"
    if has_reps:
        parts += (f" + {total} expert replicates + {total * 3} expert+nospout combined"
                  f" + up to {total * 40} single-replicate")
    print(f"Generating {n_figs} snapshot figures ({parts}) ...")
    sml = stim_mode_label
    done = 0
    for dur in durations:
        for size in sizes:
            done += 1
            print(f"  [{done}/{total}] size={size}  dur={dur}")
            try:
                plot_snapshot_figure(snapshot_data, size, dur, output_dir, stim_mode_label=sml)
            except KeyError as e:
                print(f"    WARNING: plot_snapshot_figure skipped (missing frame {e})")
            try:
                plot_detailed_snapshot_figure(snapshot_data, size, dur, output_dir, stim_mode_label=sml)
            except KeyError as e:
                print(f"    WARNING: plot_detailed_snapshot_figure skipped (missing frame {e})")
            if has_reps:
                try:
                    plot_expert_examples_figure(snapshot_data, size, dur, output_dir, stim_mode_label=sml)
                except KeyError as e:
                    print(f"    WARNING: plot_expert_examples_figure skipped (missing frame {e})")
                try:
                    plot_expert_nospout_examples_figure(snapshot_data, size, dur, output_dir, stim_mode_label=sml)
                except KeyError as e:
                    print(f"    WARNING: plot_expert_nospout_examples_figure skipped (missing frame {e})")
                try:
                    plot_single_replicate_figures(snapshot_data, size, dur, output_dir, stim_mode_label=sml)
                except KeyError as e:
                    print(f"    WARNING: plot_single_replicate_figures skipped (missing frame {e})")
    print("All figures generated.")


# =============================================================================
# CLI
# =============================================================================

def parse_int_list(s):
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description='Generate Ising perturbation snapshot & heatmap figures')
    parser.add_argument('--output', required=True,
                        help='Output directory')
    parser.add_argument('--comparison', default=None,
                        help='Path to comparison results .mat file')
    parser.add_argument('--ising-data', default=None,
                        help='Ising data root (fallback for finding comparison)')
    parser.add_argument('--durations', default=None, type=parse_int_list,
                        help='Subset of durations (comma-separated, e.g. 10,50,100)')
    parser.add_argument('--sizes', default=None, type=parse_int_list,
                        help='Subset of sizes (comma-separated, e.g. 4,8,12)')
    parser.add_argument('--figures-only', action='store_true',
                        help='Regenerate figures from saved snapshot_data.npz')
    parser.add_argument('--scan', action='store_true',
                        help='Dry run: show what would be generated')

    # Heatmap / asymmetry mode arguments
    parser.add_argument('--mode', choices=['snapshots', 'heatmap', 'all'],
                        default='snapshots',
                        help='Analysis mode: snapshots (default), heatmap, or all')
    parser.add_argument('--n-reps', type=int, default=100,
                        help='Number of replicates for heatmap mode (default: 100)')
    parser.add_argument('--heatmap-only', action='store_true',
                        help='Regenerate heatmap/asymmetry figures from saved heatmap_data.npz')
    parser.add_argument('--index', type=int, default=None,
                        help='SLURM array task index (for snapshot or heatmap mode)')
    parser.add_argument('--combine', action='store_true',
                        help='Combine per-combo npz files (snapshot + heatmap) into single files')
    parser.add_argument('--count-jobs', action='store_true',
                        help='Print total combo count and exit (for SLURM array sizing)')
    parser.add_argument('--stim-mode', default='clamped',
                        choices=list(SNAPSHOT_STIM_MODES.keys()),
                        help='Stimulus mode (default: clamped)')
    parser.add_argument('--time-scale', type=float, default=1.0,
                        help='Temporal scale factor (globalMeanSF) for frame-to-seconds conversion (default: 1.0 = frames)')
    parser.add_argument('--sampling-rate', type=float, default=10.0,
                        help='Experimental sampling rate in Hz (default: 10.0)')

    args = parser.parse_args()

    # Resolve stimulus mode parameters
    stim_cfg = SNAPSHOT_STIM_MODES[args.stim_mode]
    stim_mode = stim_cfg['mode']
    stim_bias = stim_cfg['stim_bias']

    # --- Heatmap-only: regenerate figures from saved data ---
    if args.heatmap_only:
        npz_path = os.path.join(args.output, 'heatmap_data.npz')
        if not os.path.exists(npz_path):
            print(f"ERROR: heatmap_data.npz not found at {npz_path}")
            return 1

        heatmap_data = load_heatmap_data(npz_path)
        sizes = heatmap_data['__sizes__']
        durations = heatmap_data['__durations__']

        if args.sizes:
            sizes = [s for s in sizes if s in args.sizes]
        if args.durations:
            durations = [d for d in durations if d in args.durations]

        # Auto-detect global_mean_sf from comparison results
        time_sf = args.time_scale
        if time_sf == 1.0:
            try:
                if args.comparison:
                    _, time_sf = load_comparison_results(args.comparison)
                elif args.ising_data:
                    comp_path = find_comparison_results(args.ising_data)
                    _, time_sf = load_comparison_results(comp_path)
            except (FileNotFoundError, Exception) as e:
                print(f"Warning: could not auto-detect time scale factor: {e}")
                time_sf = 1.0

        generate_all_heatmap_figures(heatmap_data, sizes, durations, args.output,
                                     stim_mode_label=args.stim_mode,
                                     time_scale_factor=time_sf,
                                     sampling_rate=args.sampling_rate)
        return 0

    # --- Combine per-combo files (heatmap + snapshot) ---
    if args.combine:
        combine_heatmap_files(args.output)
        combine_snapshot_files(args.output)
        return 0

    # --- Figures-only (snapshots) ---
    if args.figures_only:
        npz_path = os.path.join(args.output, 'snapshot_data.npz')
        if not os.path.exists(npz_path):
            alt = os.path.join(args.output, 'PerturbationSnapshots', 'snapshot_data.npz')
            if os.path.exists(alt):
                npz_path = alt
            else:
                print(f"ERROR: snapshot_data.npz not found at {npz_path}")
                return 1

        snapshot_data, sizes, durations = load_snapshot_data(npz_path)

        if args.sizes:
            sizes = [s for s in sizes if s in args.sizes]
        if args.durations:
            durations = [d for d in durations if d in args.durations]

        generate_all_figures(snapshot_data, sizes, durations, args.output,
                             stim_mode_label=args.stim_mode)
        return 0

    # --- Need comparison results for everything below ---
    if args.comparison:
        comparison_path = args.comparison
    elif args.ising_data:
        comparison_path = find_comparison_results(args.ising_data)
        print(f"Found comparison results: {comparison_path}")
    else:
        print("ERROR: --comparison or --ising-data required (unless --figures-only / --heatmap-only)")
        return 1

    best_matches, global_mean_sf = load_comparison_results(comparison_path)
    stimulus_durations = compute_stimulus_durations(global_mean_sf)

    # Apply filters
    sizes = args.sizes if args.sizes else STIMULUS_SIZES
    durations = args.durations if args.durations else stimulus_durations

    # --- Count-jobs: print heatmap combo count and exit ---
    if args.count_jobs:
        combos = enumerate_heatmap_combos(sizes, durations)
        print(len(combos))
        return 0

    # --- Scan (dry run) ---
    if args.scan:
        n_main_jobs = len(CONDITIONS) * len(sizes) * len(durations)
        n_expert_rep_jobs = 4 * len(sizes) * len(durations)
        n_figs = len(sizes) * len(durations)

        if args.mode in ('snapshots', 'all'):
            print(f"\n=== DRY RUN (snapshots) ===")
            print(f"Conditions: {CONDITIONS}")
            print(f"Sizes: {sizes}")
            print(f"Durations: {durations}")
            print(f"Total snapshot runs: {n_main_jobs} main + {n_expert_rep_jobs} Expert replicates = {n_main_jobs + n_expert_rep_jobs}")
            print(f"Figures to generate: {n_figs * 3} ({n_figs} summary + {n_figs} detailed + {n_figs} expert replicates)")
            print(f"Output directory: {args.output}")
            print(f"\nSnapshot output structure:")
            print(f"  {args.output}/snapshot_data.npz")
            for dur in durations:
                for size in sizes:
                    print(f"  {args.output}/PerturbationSnapshots/dur_{dur}/snapshot_size_{size}_{args.stim_mode}.{{png,pdf}}")
                    print(f"  {args.output}/PerturbationSnapshots/dur_{dur}/snapshot_detail_size_{size}_{args.stim_mode}.{{png,pdf}}")
                    print(f"  {args.output}/PerturbationSnapshots/dur_{dur}/snapshot_expert_examples_size_{size}_{args.stim_mode}.{{png,pdf}}")

        if args.mode in ('heatmap', 'all'):
            combos = enumerate_heatmap_combos(sizes, durations)
            n_heatmap_sims = len(combos) * args.n_reps
            n_heatmap_figs = n_figs * 8 + len(durations)

            print(f"\n=== DRY RUN (heatmap) ===")
            print(f"Conditions: {CONDITIONS}")
            print(f"Sizes: {sizes}")
            print(f"Durations: {durations}")
            print(f"Replicates: {args.n_reps}")
            print(f"Total combos: {len(combos)} (SLURM array tasks)")
            print(f"Total simulations: {n_heatmap_sims}")
            print(f"Figures to generate: {n_heatmap_figs}")
            print(f"  Per (size, duration): heatmap_prob, heatmap_prob_crop, heatmap_prob_expert,")
            print(f"                        heatmap_prob_expert_nospout, asymmetry_polar, asymmetry_sectors,")
            print(f"                        asymmetry_polar_peak, asymmetry_sectors_peak")
            print(f"  Per duration: asymmetry_summary")
            print(f"\nHeatmap output structure:")
            print(f"  {args.output}/heatmap_data.npz")
            for dur in durations:
                for size in sizes:
                    print(f"  {args.output}/PerturbationSnapshots/dur_{dur}/heatmap_prob_size_{size}_dur_{dur}.{{png,pdf}}")
                    print(f"  {args.output}/PerturbationSnapshots/dur_{dur}/heatmap_prob_crop_size_{size}_dur_{dur}.{{png,pdf}}")
                    print(f"  {args.output}/PerturbationSnapshots/dur_{dur}/heatmap_prob_expert_size_{size}_dur_{dur}.{{png,pdf}}")
                    print(f"  {args.output}/PerturbationSnapshots/dur_{dur}/heatmap_prob_expert_nospout_size_{size}_dur_{dur}.{{png,pdf}}")
                    print(f"  {args.output}/PerturbationSnapshots/dur_{dur}/asymmetry_polar_size_{size}_dur_{dur}.{{png,pdf}}")
                    print(f"  {args.output}/PerturbationSnapshots/dur_{dur}/asymmetry_sectors_size_{size}_dur_{dur}.{{png,pdf}}")
                    print(f"  {args.output}/PerturbationSnapshots/dur_{dur}/asymmetry_polar_peak_size_{size}_dur_{dur}.{{png,pdf}}")
                    print(f"  {args.output}/PerturbationSnapshots/dur_{dur}/asymmetry_sectors_peak_size_{size}_dur_{dur}.{{png,pdf}}")
                print(f"  {args.output}/PerturbationSnapshots/asymmetry_summary_dur_{dur}.{{png,pdf}}")

        return 0

    # --- Run simulations ---

    print(f"Stimulus mode: {args.stim_mode} (internal: {stim_mode}, bias: {stim_bias})")

    if args.mode in ('snapshots', 'all'):
        snapshot_data = generate_all_snapshots(
            best_matches, stimulus_durations, args.output,
            size_filter=args.sizes, duration_filter=args.durations,
            index=args.index,
            stim_mode=stim_mode, stim_bias=stim_bias)
        suffix = f'_idx{args.index}' if args.index is not None else ''
        save_snapshot_data(snapshot_data, sizes, durations, args.output, suffix=suffix)
        # Generate figures only when running all combos (not per-combo SLURM tasks)
        if args.index is None:
            generate_all_figures(snapshot_data, sizes, durations, args.output,
                                 stim_mode_label=args.stim_mode)

    if args.mode in ('heatmap', 'all'):
        heatmap_data = generate_heatmap_data(
            best_matches, stimulus_durations, args.output,
            n_reps=args.n_reps,
            size_filter=args.sizes, duration_filter=args.durations,
            index=args.index,
            stim_mode=stim_mode, stim_bias=stim_bias)

        # Save results
        suffix = f'_idx{args.index}' if args.index is not None else ''
        save_heatmap_data(heatmap_data, args.output, suffix=suffix)

        # Generate figures (skip for per-combo SLURM tasks — figures from combined data)
        if args.index is None:
            time_sf = args.time_scale if args.time_scale != 1.0 else global_mean_sf
            generate_all_heatmap_figures(heatmap_data, sizes, durations, args.output,
                                         stim_mode_label=args.stim_mode,
                                         time_scale_factor=time_sf,
                                         sampling_rate=args.sampling_rate)

    return 0


if __name__ == '__main__':
    exit(main())
