# -*- coding: utf-8 -*-
"""
Ising Model Perturbation Experiments

This script runs perturbation experiments on the top 10 best-matching Ising
simulations per condition (Naive, Beginner, Expert, NoSpout) to analyze:
1. Region growth in response to localized stimuli
2. Persistence of activity after stimulus offset
3. Size-dependent gating effects

Stimulus Modes:
- Clamped: Region held at +1 throughout stimulus duration
- Double Pulse: Set to +1 at stimulus onset AND offset (mimics SC behavior)
- Bias: Additive bias added to stim region's local field for entire stim duration
- Double Pulse Bias: Soft analogue of Double Pulse — bias boost applied at
  onset/offset windows only (no clamp), plain dynamics in between

Usage:
    # Run all perturbation experiments (local)
    python run_ising_perturbations.py --output IsingPerturbations --workers 8

    # Run single job by index (for SLURM array jobs, legacy mode)
    python run_ising_perturbations.py --output IsingPerturbations --index 0

    # Run batch of jobs per array task (recommended for SLURM, reduces overhead)
    # With --batch-size 10 and --index 0, runs jobs 0-9
    # With --batch-size 10 and --index 1, runs jobs 10-19
    python run_ising_perturbations.py --output IsingPerturbations --index 0 --batch-size 10

    # Scan and show what would be run
    python run_ising_perturbations.py --output IsingPerturbations --scan
"""

import numpy as np
from scipy.io import savemat, loadmat
try:
    import hdf5storage
    HAS_HDF5STORAGE = True
except ImportError:
    HAS_HDF5STORAGE = False
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as scipy_label
import os
import re
import argparse
import time
from itertools import product
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from numba import njit
import h5py
import gc


# =============================================================================
# Configuration
# =============================================================================

# Grid dimensions (matching original simulations)
L, M = 39, 78

# Centre crop matching experimental FOV (13 rows × 26 cols)
CROP_ROW_START, CROP_ROW_END = 13, 26
CROP_COL_START, CROP_COL_END = 26, 52

# Perturbation experiment parameters
PRE_STIM_FRAMES = 400     # Equilibration before stimulus
POST_STIM_FRAMES = 300    # Recovery period after stimulus offset

# Fixed MC sweep counts (simulation-scale reference, always included)
BASE_MC_SWEEP_DURATIONS = [10, 25, 50, 100]

# Target real-time durations (converted to MC sweeps via temporal_scale_factor)
TARGET_DURATION_SECONDS = [0.5, 1.0, 2.0, 5.0, 10.0]

# Experimental sampling rate (Hz) — must match Figure5_IsingPerturbationAnalysis.m:51
SAMPLING_RATE = 10.0
N_REPLICATES = 50         # Independent runs per simulation

# Stimulus sizes to test (in pixels, square regions)
STIMULUS_SIZES = [1, 2, 3, 4, 6, 8, 10, 12]

# Stimulus modes
STIMULUS_MODES = [
    'clamped',
    'double_pulse', 'double_pulse3', 'double_pulse5', 'double_pulse10',
    'bias',
    'double_pulse_bias', 'double_pulse_bias3', 'double_pulse_bias5', 'double_pulse_bias10',
    'double_pulse_bias10_offsetlag5',
]

# Deterministic seed offset per stimulus mode (avoids hash() non-determinism)
STIM_MODE_SEED_OFFSET = {
    'clamped': 0, 'double_pulse': 1, 'bias': 2,
    'double_pulse3': 3, 'double_pulse5': 4, 'double_pulse10': 5,
    'double_pulse_bias': 6, 'double_pulse_bias3': 7,
    'double_pulse_bias5': 8, 'double_pulse_bias10': 9,
    'double_pulse_bias10_offsetlag5': 10,
}


# Regex parser for double_pulse_bias[N][_offsetlag[L]] mode strings.
# Group 1: pulse width (may be empty -> default 1); group 2: offset lag (optional, default 0).
_DPB_PATTERN = re.compile(r'^double_pulse_bias(\d*)(?:_offsetlag(\d+))?$')


def parse_double_pulse_bias_mode(stim_mode):
    """Parse a double_pulse_bias mode string.

    Returns (pulse_width, offset_lag) where pulse_width >= 1 and offset_lag >= 0.
    Returns None if the mode does not match the double_pulse_bias[N][_offsetlagL] schema.
    """
    m = _DPB_PATTERN.match(stim_mode)
    if m is None:
        return None
    pw_str = m.group(1)
    ol_str = m.group(2)
    pulse_width = int(pw_str) if pw_str else 1
    offset_lag = int(ol_str) if ol_str is not None else 0
    return pulse_width, offset_lag


def mode_uses_bias(stim_mode):
    """Return True if the mode uses a stim_bias value (bias_field path)."""
    return stim_mode == 'bias' or stim_mode.startswith('double_pulse_bias')

# Additive offset for independent re-runs; set via --seed-offset CLI
SEED_OFFSET = 0

# Optional allowlist of stimulus modes to enumerate; set via --modes CLI.
# When None, all STIMULUS_MODES are used. When set, only listed modes
# (and their bias-value sweeps) are generated/aggregated.
MODES_FILTER = None

# Optional allowlists for partial / targeted re-runs. Each is None by default
# (no filtering); set via the matching --filter-* CLI flag. They scope the
# job list emitted by generate_all_jobs() — useful for backfilling a subset
# of leaves (e.g. a few biases under specific sims) without re-running the
# whole sweep.
COND_FILTER = None    # set of condition names, e.g. {'Beginner'}
SIM_FILTER = None     # set of int sim indices, e.g. {0, 1, 2, 3}
SIZE_FILTER = None    # set of int stim sizes, e.g. {2, 3, 4}
BIAS_FILTER = None    # set of floats; matched approximately (within 1e-6)
# Override the Wasserstein top-10 selection in load_comparison_results: when
# set, maps cond -> list of GLOBAL grid indices that become this cond's
# best_matches in the order given. Conds omitted from this dict are SKIPPED
# (no fallback to Wasserstein top-10). Populated from --manual-indices JSON.
MANUAL_INDICES = None

# Stimulus bias intensities to test for 'bias' mode (additive to global bias in stimulus region)
# Higher values approach clamped behavior, lower values represent weaker contrast.
# History:
#   2026-05-04: added 6.0, 10.0, 12.0 to refine trained-condition offset window.
#   2026-05-04 (later): added 14.0, 16.0 to bracket Beginner/Expert offset.
#   2026-05-04 (later): switched matcher to read activity_crop (was the wrong
#                       dataset). With FOV-cropped traces, trained-group bests
#                       dropped from clamped/8 to ~1.0; added 0.15, 0.75, 1.25,
#                       1.50, 1.75 to refine the 0.5-2.0 mid-low range and
#                       bracket Naive/NoSpout below 0.25.
STIMULUS_BIAS_VALUES = [0.15, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]

# Conditions to analyze
CONDITIONS = ['Naive', 'Beginner', 'Expert', 'NoSpout']
N_TOP_MATCHES = 10

# Blob detection parameters (matching Figure5_IsingComparison.m)
BLOB_SIGMA = 1.0
BLOB_THRESHOLD = 0.5
MIN_BLOB_SIZE = 10

# Burn-in for equilibration: adaptive based on H-field timescale tau = (L*M)/decay_const
# 7*tau gives 99.9% equilibration; floor of 2000 matches generate_all_ising_simulations.py
BURN_IN_MIN = 2000       # Minimum burn-in (safety floor)
BURN_IN_TAU_MULT = 7     # Multiplier on tau (7 => 99.9% equilibrated)
DECORR_TAU_MULT = 2      # Gap between equilibrium samples: 2*tau for ~87% decorrelation
DECORR_MIN = 500         # Minimum decorrelation gap (safety floor)


def compute_stimulus_durations(global_mean_sf):
    """
    Compute MC sweep durations: fixed base values + dynamically converted real-time targets.

    Parameters
    ----------
    global_mean_sf : float
        Mean temporal_scale_factor (frames/MC_sweep) across best-matched sims.

    Returns
    -------
    list of int
        Sorted, deduplicated MC sweep counts.
    """
    mc_sweeps = set(BASE_MC_SWEEP_DURATIONS)
    for t in TARGET_DURATION_SECONDS:
        sweeps = max(1, round(t * SAMPLING_RATE / global_mean_sf))
        mc_sweeps.add(sweeps)
    return sorted(mc_sweeps)


# =============================================================================
# Core Simulation Functions (from generate_all_ising_simulations.py)
# =============================================================================

def build_diamond_kernel(rad):
    """
    Construct the diamond-shaped inhibition kernel.

    EXCLUDES the center (0,0) and 4 nearest neighbors (±1,0), (0,±1)
    so that NN contribute only to excitation (J=+1) and more distant
    neurons contribute only to inhibition.
    """
    kernel = np.zeros((2*rad+1, 2*rad+1), dtype=np.float64)
    center = rad  # center index
    for i in range(rad+1):
        kernel[i, rad-i:(rad+i+1)] = 1
        kernel[-i-1, rad-i:(rad+i+1)] = 1

    # Exclude center neuron
    kernel[center, center] = 0

    # Exclude 4 nearest neighbors
    kernel[center-1, center] = 0  # top
    kernel[center+1, center] = 0  # bottom
    kernel[center, center-1] = 0  # left
    kernel[center, center+1] = 0  # right

    return kernel


@njit(cache=True)
def heat_bath_numba(config, beta, c, decay_const, H, K, bias_val, K_sum):
    """
    Numba-optimized Monte Carlo sweep using heat-bath algorithm.

    Parameters:
    -----------
    config : ndarray (L x M), int8
        Current spin configuration (+1 or -1)
    beta : float
        Inverse temperature
    c : float
        Coupling strength
    decay_const : float
        Decay constant for field H
    H : ndarray (L x M), float64
        Auxiliary field
    K : ndarray, float64
        Inhibition kernel
    bias_val : float
        Bias value (scalar, applied uniformly)
    K_sum : float
        Pre-computed sum of kernel (for normalization)

    Returns:
    --------
    config, H : Updated spin configuration and field (modified in-place)
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

        # Nearest-neighbor field with periodic boundary
        h = (config[(i + 1) % L, j] +
             config[(i - 1 + L) % L, j] +
             config[i, (j + 1) % M] +
             config[i, (j - 1 + M) % M])

        local_field = float(h) + H[i, j] + bias_val

        # Probability for heat-bath algorithm
        pi_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * local_field))

        # Spin update
        if np.random.random() < pi_plus:
            config[i, j] = 1
        else:
            config[i, j] = -1

        # Compute inhibition sum manually (avoid fancy indexing)
        inhib_sum = 0.0
        for di in range(kL):
            for dj in range(kM):
                if K[di, dj] > 0:
                    row_idx = (i - radL + di + L) % L
                    col_idx = (j - radM + dj + M) % M
                    inhib_sum += K[di, dj] * config[row_idx, col_idx]

        inhib = inhib_sum / K_sum

        # Update field H
        H[i, j] = H[i, j] - (c * inhib * dt) - (decay_const * H[i, j] * dt)

    return config, H


@njit(cache=True)
def heat_bath_numba_with_bias_field(config, beta, c, decay_const, H, K, bias_field, K_sum):
    """
    Numba-optimized Monte Carlo sweep with 2D bias field support.

    This version allows spatially-varying bias (e.g., for local stimulus application).

    Parameters:
    -----------
    config : ndarray (L x M), int8
        Current spin configuration (+1 or -1)
    beta : float
        Inverse temperature
    c : float
        Coupling strength
    decay_const : float
        Decay constant for field H
    H : ndarray (L x M), float64
        Auxiliary field
    K : ndarray, float64
        Inhibition kernel
    bias_field : ndarray (L x M), float64
        Spatially-varying bias field (allows local stimulus bias)
    K_sum : float
        Pre-computed sum of kernel (for normalization)

    Returns:
    --------
    config, H : Updated spin configuration and field (modified in-place)
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

        # Nearest-neighbor field with periodic boundary
        h = (config[(i + 1) % L, j] +
             config[(i - 1 + L) % L, j] +
             config[i, (j + 1) % M] +
             config[i, (j - 1 + M) % M])

        # Use spatially-varying bias field
        local_field = float(h) + H[i, j] + bias_field[i, j]

        # Probability for heat-bath algorithm
        pi_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * local_field))

        # Spin update
        if np.random.random() < pi_plus:
            config[i, j] = 1
        else:
            config[i, j] = -1

        # Compute inhibition sum manually (avoid fancy indexing)
        inhib_sum = 0.0
        for di in range(kL):
            for dj in range(kM):
                if K[di, dj] > 0:
                    row_idx = (i - radL + di + L) % L
                    col_idx = (j - radM + dj + M) % M
                    inhib_sum += K[di, dj] * config[row_idx, col_idx]

        inhib = inhib_sum / K_sum

        # Update field H
        H[i, j] = H[i, j] - (c * inhib * dt) - (decay_const * H[i, j] * dt)

    return config, H


# =============================================================================
# Perturbation Experiment Functions
# =============================================================================

def get_stimulus_region(stim_size, center_row, center_col, L, M):
    """
    Get row and column indices for a square stimulus region centered at (center_row, center_col).

    Returns:
    --------
    stim_rows, stim_cols : numpy arrays
        Row and column indices for the stimulus region
    """
    half_size = stim_size // 2

    # For odd sizes, center exactly; for even, shift slightly
    if stim_size % 2 == 0:
        row_start = center_row - half_size
        row_end = center_row + half_size
        col_start = center_col - half_size
        col_end = center_col + half_size
    else:
        row_start = center_row - half_size
        row_end = center_row + half_size + 1
        col_start = center_col - half_size
        col_end = center_col + half_size + 1

    # Clamp to grid bounds
    row_start = max(0, row_start)
    row_end = min(L, row_end)
    col_start = max(0, col_start)
    col_end = min(M, col_end)

    stim_rows = np.arange(row_start, row_end)
    stim_cols = np.arange(col_start, col_end)

    return stim_rows, stim_cols


def apply_stimulus(config, stim_rows, stim_cols):
    """Apply stimulus by setting specified region to +1."""
    for i in stim_rows:
        for j in stim_cols:
            config[i, j] = 1
    return config


def compute_morans_I(binary_frame):
    """
    Compute Moran's I for spatial autocorrelation.

    Moran's I measures spatial clustering:
        -1 = dispersed (checkerboard pattern)
         0 = random
        +1 = clustered

    Parameters:
    -----------
    binary_frame : ndarray (L x M), float64
        Binary activity frame (0/1 values)

    Returns:
    --------
    morans_I : float
        Spatial autocorrelation coefficient in [-1, 1]
    """
    L, M = binary_frame.shape
    n = L * M
    x = binary_frame.flatten()
    x_mean = np.mean(x)
    x_centered = x - x_mean

    # Build spatial weights using 4-neighbor adjacency
    # For efficiency, compute using vectorized neighbor sums
    padded = np.pad(binary_frame, 1, mode='wrap')  # Periodic boundary
    neighbor_sum = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                    padded[1:-1, :-2] + padded[1:-1, 2:])

    # Moran's I = (n/W) * (sum(w_ij * x_i * x_j) / sum((x_i - mean)^2))
    W = 4 * n  # Total weight (each cell has 4 neighbors)
    numerator = np.sum(x_centered.reshape(L, M) * (neighbor_sum - 4 * x_mean))
    denominator = np.sum(x_centered ** 2)

    if denominator == 0:
        return 0.0

    morans_I = (n / W) * (numerator / denominator)
    return np.clip(morans_I, -1, 1)


def compute_frame_metrics(frame, stim_rows, stim_cols, center_row, center_col):
    """
    Compute metrics for a single frame.

    Parameters:
    -----------
    frame : ndarray (L x M), int8
        Spin configuration (+1 or -1)
    stim_rows, stim_cols : arrays
        Stimulus region indices
    center_row, center_col : int
        Center of stimulus

    Returns:
    --------
    dict with metrics:
        - activity: fraction of +1 spins
        - stim_activity: activity in stimulus region
        - stimulus_blob_area: area of blob connected to stimulus
        - stimulus_blob_extent: max distance from center to edge of stimulus blob
        - valid_blob_count: number of blobs >= MIN_BLOB_SIZE
        - morans_I: spatial autocorrelation (-1 to +1)
        - wavefront_pixels: list of (row, col) at max extent
    """
    L, M = frame.shape

    # Convert -1/+1 to 0/1
    binary_frame = (frame.astype(np.float64) + 1) / 2

    # Global activity
    activity = np.mean(binary_frame)

    # Activity in centre crop (matching experimental FOV)
    activity_crop = np.mean(binary_frame[CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END])

    # Stimulus region activity
    stim_region = binary_frame[np.ix_(stim_rows, stim_cols)]
    stim_activity = np.mean(stim_region) if stim_region.size > 0 else 0.0

    # Moran's I for spatial autocorrelation
    morans_I = compute_morans_I(binary_frame)

    # Blob detection - find blob connected to stimulus
    smoothed = gaussian_filter(binary_frame, sigma=BLOB_SIGMA)
    binary_blob = smoothed > BLOB_THRESHOLD
    labeled, num_features = scipy_label(binary_blob)

    stimulus_blob_area = 0
    stimulus_blob_extent = 0.0
    valid_blob_count = 0
    wavefront_pixels = []

    if num_features > 0:
        # Count valid blobs (>= MIN_BLOB_SIZE)
        for blob_id in range(1, num_features + 1):
            blob_mask = (labeled == blob_id)
            if np.sum(blob_mask) >= MIN_BLOB_SIZE:
                valid_blob_count += 1

        # Find which blob overlaps with stimulus region
        stim_labels = labeled[np.ix_(stim_rows, stim_cols)]
        label_counts = np.bincount(stim_labels.flatten())

        # Find most common non-zero label in stimulus region
        if len(label_counts) > 1:
            label_counts[0] = 0  # Ignore background (label 0)
            stim_blob_label = np.argmax(label_counts)

            if stim_blob_label > 0 and label_counts[stim_blob_label] > 0:
                # Get the stimulus-connected blob
                stim_blob_mask = (labeled == stim_blob_label)
                stimulus_blob_area = np.sum(stim_blob_mask)

                # Compute max extent from stimulus center to any pixel in blob
                rows, cols = np.where(stim_blob_mask)
                if len(rows) > 0:
                    distances = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
                    stimulus_blob_extent = np.max(distances)

                    # Get wavefront pixels (at max extent, within 1 pixel tolerance)
                    max_dist = np.max(distances)
                    wavefront_mask = distances >= (max_dist - 1.0)
                    wavefront_pixels = list(zip(rows[wavefront_mask].tolist(),
                                                 cols[wavefront_mask].tolist()))

    return {
        'activity': activity,
        'activity_crop': activity_crop,
        'stim_activity': stim_activity,
        'stimulus_blob_area': stimulus_blob_area,
        'stimulus_blob_extent': stimulus_blob_extent,
        'valid_blob_count': valid_blob_count,
        'morans_I': morans_I,
        'wavefront_pixels': wavefront_pixels
    }


def compute_propagation_velocity(extent_history, window=5):
    """
    Compute instantaneous propagation velocity from extent time series.

    Uses linear regression over a sliding window to estimate velocity.

    Parameters:
    -----------
    extent_history : list of float
        History of stimulus blob extent values
    window : int
        Number of frames for velocity estimation

    Returns:
    --------
    velocity : float
        Propagation velocity in pixels/frame (non-negative)
    """
    if len(extent_history) < window:
        return 0.0

    recent = extent_history[-window:]
    frames = np.arange(len(recent))

    if len(frames) > 1:
        # Linear regression: extent = velocity * t + intercept
        coeffs = np.polyfit(frames, recent, 1)
        velocity = coeffs[0]
        return max(0.0, velocity)  # Only positive (outward) velocity
    return 0.0


def compute_wavefront_anisotropy(wavefront_pixels, center_row, center_col):
    """
    Compute anisotropy of wavefront spread (how directional is the spread).

    Uses circular statistics on the angles from center to wavefront pixels.

    Parameters:
    -----------
    wavefront_pixels : list of (row, col) tuples
        Pixels at the current wavefront
    center_row, center_col : int
        Center of stimulus

    Returns:
    --------
    anisotropy : float
        0 = isotropic (uniform spread), 1 = highly directional
    direction : float
        Primary direction of spread (radians, 0 = right, pi/2 = down)
    """
    if len(wavefront_pixels) < 3:
        return 0.0, 0.0

    # Get angles from center to each wavefront pixel
    rows = np.array([p[0] for p in wavefront_pixels])
    cols = np.array([p[1] for p in wavefront_pixels])

    angles = np.arctan2(rows - center_row, cols - center_col)

    # Circular statistics for anisotropy (resultant length)
    mean_cos = np.mean(np.cos(angles))
    mean_sin = np.mean(np.sin(angles))

    # Resultant length: 0 = uniform distribution, 1 = all same direction
    anisotropy = np.sqrt(mean_cos**2 + mean_sin**2)

    # Mean direction
    direction = np.arctan2(mean_sin, mean_cos)

    return anisotropy, direction


def compute_prestim_summary(prestim_metrics, stim_rows, stim_cols):
    """
    Compute summary statistics for the pre-stimulus period.

    Parameters:
    -----------
    prestim_metrics : list of dict
        Frame-by-frame metrics from pre-stimulus period
    stim_rows, stim_cols : arrays
        Stimulus region indices (for checking blob presence)

    Returns:
    --------
    dict with pre-stimulus state characterization
    """
    n_frames = len(prestim_metrics)
    if n_frames == 0:
        return {
            'mean_activity': 0.0,
            'std_activity': 0.0,
            'max_activity': 0.0,
            'mean_blob_count': 0.0,
            'max_blob_count': 0,
            'mean_morans_I': 0.0,
            'stim_site_activity': 0.0,
        }

    # Activity statistics
    activity_trace = np.array([m['activity'] for m in prestim_metrics])
    mean_activity = np.mean(activity_trace)
    std_activity = np.std(activity_trace)
    max_activity = np.max(activity_trace)

    # Blob count statistics
    blob_counts = np.array([m['valid_blob_count'] for m in prestim_metrics])
    mean_blob_count = np.mean(blob_counts)
    max_blob_count = int(np.max(blob_counts))

    # Moran's I statistics
    morans_trace = np.array([m['morans_I'] for m in prestim_metrics])
    mean_morans_I = np.mean(morans_trace)

    # Activity at stimulus site (last 50 frames before stim)
    stim_activity_trace = np.array([m['stim_activity'] for m in prestim_metrics])
    last_frames = min(50, n_frames)
    stim_site_activity = np.mean(stim_activity_trace[-last_frames:])

    return {
        'mean_activity': float(mean_activity),
        'std_activity': float(std_activity),
        'max_activity': float(max_activity),
        'mean_blob_count': float(mean_blob_count),
        'max_blob_count': max_blob_count,
        'mean_morans_I': float(mean_morans_I),
        'stim_site_activity': float(stim_site_activity),
    }


def run_single_perturbation(params, stim_size, stim_duration, stim_mode, replicate_idx, config_dict, stim_bias=None, initial_state=None):
    """
    Run a single perturbation experiment.

    Parameters:
    -----------
    params : dict
        Simulation parameters (beta, c, decay_const, inhibition_range, bias)
    stim_size : int
        Size of stimulus square (in pixels)
    stim_duration : int
        Duration of stimulus (in frames)
    stim_mode : str
        'clamped', 'double_pulse[N]', 'bias', or 'double_pulse_bias[N]'.
    replicate_idx : int
        Replicate index (for random seed)
    config_dict : dict
        Configuration dictionary
    stim_bias : float, optional
        Additive bias applied to stimulus region (in 'bias' mode for the full
        stim duration, in 'double_pulse_bias[N]' modes only at onset/offset
        windows). Required for any mode where mode_uses_bias() is True.
    initial_state : tuple of (spin_config, H), optional
        If provided, use this state and skip random init + burn-in.
        Used for shared-equilibrium replicates where each trial starts
        from the same clean equilibrium state.

    Returns:
    --------
    (metrics, final_state) : tuple
        metrics : dict with time series of metrics
        final_state : tuple of (spin_config, H) at end of post-stim period
    """
    if mode_uses_bias(stim_mode) and stim_bias is None:
        raise ValueError(f"stim_bias must be provided when stim_mode='{stim_mode}'")

    # Extract parameters
    beta = params['beta']
    c = params['c']
    decay_const = params['decay_const']
    rad = params['inhibition_range']
    bias = params['bias']

    # Build inhibition kernel
    kernel = build_diamond_kernel(rad)
    K_sum = float(np.sum(kernel))

    if initial_state is not None:
        # Use provided state (shared equilibrium) — skip init and burn-in
        spin_config, H = initial_state
        # Make copies so we don't mutate the caller's arrays
        spin_config = spin_config.copy()
        H = H.copy()
    else:
        # Original behavior: random init + burn-in
        np.random.seed(replicate_idx * 1000 + stim_size + stim_duration * 100 + SEED_OFFSET * 1000000)
        spin_config = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, M))
        H = np.zeros((L, M), dtype=np.float64)

    # Get stimulus region (center of grid)
    center_row = L // 2
    center_col = M // 2
    stim_rows, stim_cols = get_stimulus_region(stim_size, center_row, center_col, L, M)

    # Total frames
    total_frames = PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES

    # Storage for metrics (including new ones)
    metrics = {
        'activity': np.zeros(total_frames),
        'activity_crop': np.zeros(total_frames),
        'stim_activity': np.zeros(total_frames),
        'stimulus_blob_area': np.zeros(total_frames),
        'stimulus_blob_extent': np.zeros(total_frames),
        'valid_blob_count': np.zeros(total_frames),
        'morans_I': np.zeros(total_frames),
        'propagation_velocity': np.zeros(total_frames),
        'wavefront_anisotropy': np.zeros(total_frames),
    }

    # Storage for pre-stimulus metrics (for summary computation)
    prestim_metrics_list = []

    # Track extent history for velocity calculation
    extent_history = []

    frame_idx = 0

    # Initial burn-in (only when no initial_state provided)
    if initial_state is None:
        # H field timescale: tau = (L*M) / decay_const sweeps
        tau = (L * M) / decay_const
        burn_in_steps = max(BURN_IN_MIN, int(BURN_IN_TAU_MULT * tau))
        for _ in range(burn_in_steps):
            spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias, K_sum)

    # --- PRE-STIMULUS PERIOD ---
    for t in range(PRE_STIM_FRAMES):
        spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias, K_sum)

        # Compute metrics
        m = compute_frame_metrics(spin_config, stim_rows, stim_cols, center_row, center_col)

        # Store basic metrics
        metrics['activity'][frame_idx] = m['activity']
        metrics['activity_crop'][frame_idx] = m['activity_crop']
        metrics['stim_activity'][frame_idx] = m['stim_activity']
        metrics['stimulus_blob_area'][frame_idx] = m['stimulus_blob_area']
        metrics['stimulus_blob_extent'][frame_idx] = m['stimulus_blob_extent']
        metrics['valid_blob_count'][frame_idx] = m['valid_blob_count']
        metrics['morans_I'][frame_idx] = m['morans_I']

        # No propagation during pre-stim
        metrics['propagation_velocity'][frame_idx] = 0.0
        metrics['wavefront_anisotropy'][frame_idx] = 0.0

        # Store for pre-stim summary
        prestim_metrics_list.append(m)

        frame_idx += 1

    # Compute pre-stimulus summary
    prestim_summary = compute_prestim_summary(prestim_metrics_list, stim_rows, stim_cols)

    # --- STIMULUS ON PERIOD ---
    for t in range(stim_duration):
        if stim_mode == 'clamped':
            # Apply stimulus before sweep
            spin_config = apply_stimulus(spin_config, stim_rows, stim_cols)

            # Run dynamics
            spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias, K_sum)

            # Re-apply stimulus after sweep (ensures clamping)
            spin_config = apply_stimulus(spin_config, stim_rows, stim_cols)

        elif stim_mode.startswith('double_pulse_bias'):
            # Soft (additive-bias) analogue of double_pulse: bias boost in stim
            # region only at onset/offset windows; plain dynamics in between.
            # When mode has _offsetlagL suffix, the offset pulse is delayed by
            # L frames past stim_end and is fired in the POST_STIM loop below.
            parsed = parse_double_pulse_bias_mode(stim_mode)
            if parsed is None:
                raise ValueError(
                    f"stim_mode {stim_mode!r} starts with 'double_pulse_bias' "
                    "but does not match expected schema "
                    "'double_pulse_bias[N][_offsetlagL]'."
                )
            pulse_width, offset_lag = parsed
            onset_active = (t < pulse_width)
            # When offset_lag == 0, offset pulse sits at the tail of the stim
            # window (legacy behavior). When offset_lag > 0, the offset pulse
            # is deferred to the POST_STIM loop, so no offset firing here.
            offset_active_in_stim = (
                offset_lag == 0 and t >= stim_duration - pulse_width
            )
            if onset_active or offset_active_in_stim:
                bias_field = np.full((L, M), bias, dtype=np.float64)
                for i in stim_rows:
                    for j in stim_cols:
                        bias_field[i, j] += stim_bias
                spin_config, H = heat_bath_numba_with_bias_field(
                    spin_config, beta, c, decay_const, H, kernel, bias_field, K_sum
                )
            else:
                spin_config, H = heat_bath_numba(
                    spin_config, beta, c, decay_const, H, kernel, bias, K_sum
                )

        elif stim_mode.startswith('double_pulse'):
            # Parse clamp width: double_pulse=1, double_pulse3=3, etc.
            dp_suffix = stim_mode[len('double_pulse'):]
            clamp_width = int(dp_suffix) if dp_suffix else 1
            if t < clamp_width or t >= stim_duration - clamp_width:
                spin_config = apply_stimulus(spin_config, stim_rows, stim_cols)

            # Run dynamics
            spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias, K_sum)

        elif stim_mode == 'bias':
            # Apply stimulus by adding local bias to stimulus region
            # Create 2D bias field: global bias + local stimulus bias in stimulus region
            bias_field = np.ones((L, M), dtype=np.float64) * bias
            for i in stim_rows:
                for j in stim_cols:
                    bias_field[i, j] += stim_bias

            # Run dynamics with spatially-varying bias field
            spin_config, H = heat_bath_numba_with_bias_field(
                spin_config, beta, c, decay_const, H, kernel, bias_field, K_sum
            )

        # Compute metrics
        m = compute_frame_metrics(spin_config, stim_rows, stim_cols, center_row, center_col)

        # Store basic metrics
        metrics['activity'][frame_idx] = m['activity']
        metrics['activity_crop'][frame_idx] = m['activity_crop']
        metrics['stim_activity'][frame_idx] = m['stim_activity']
        metrics['stimulus_blob_area'][frame_idx] = m['stimulus_blob_area']
        metrics['stimulus_blob_extent'][frame_idx] = m['stimulus_blob_extent']
        metrics['valid_blob_count'][frame_idx] = m['valid_blob_count']
        metrics['morans_I'][frame_idx] = m['morans_I']

        # Track extent for velocity calculation
        extent_history.append(m['stimulus_blob_extent'])

        # Compute propagation velocity
        velocity = compute_propagation_velocity(extent_history)
        metrics['propagation_velocity'][frame_idx] = velocity

        # Compute wavefront anisotropy
        # Note: anisotropy is only meaningful for stim_size >= 3;
        # smaller stimuli have too few wavefront pixels for reliable circular statistics.
        anisotropy, _ = compute_wavefront_anisotropy(
            m['wavefront_pixels'], center_row, center_col
        )
        metrics['wavefront_anisotropy'][frame_idx] = anisotropy

        frame_idx += 1

    # --- POST-STIMULUS PERIOD ---
    # If using a double_pulse_bias[N]_offsetlagL mode, the offset bias pulse is
    # fired in this loop, starting at post-stim frame `offset_lag` and lasting
    # `pulse_width` frames. This shifts the offset peak past stim_end.
    post_pulse_width = 0
    post_offset_lag = 0
    if stim_mode.startswith('double_pulse_bias'):
        parsed = parse_double_pulse_bias_mode(stim_mode)
        if parsed is not None:
            post_pulse_width, post_offset_lag = parsed
            if post_offset_lag > 0 and (
                post_offset_lag + post_pulse_width > POST_STIM_FRAMES
            ):
                raise ValueError(
                    f"Mode {stim_mode!r}: offset_lag ({post_offset_lag}) + "
                    f"pulse_width ({post_pulse_width}) exceeds POST_STIM_FRAMES "
                    f"({POST_STIM_FRAMES}). Increase POST_STIM_FRAMES."
                )

    for t in range(POST_STIM_FRAMES):
        if (
            post_offset_lag > 0
            and post_offset_lag <= t < post_offset_lag + post_pulse_width
        ):
            bias_field = np.full((L, M), bias, dtype=np.float64)
            for i in stim_rows:
                for j in stim_cols:
                    bias_field[i, j] += stim_bias
            spin_config, H = heat_bath_numba_with_bias_field(
                spin_config, beta, c, decay_const, H, kernel, bias_field, K_sum
            )
        else:
            spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias, K_sum)

        # Compute metrics
        m = compute_frame_metrics(spin_config, stim_rows, stim_cols, center_row, center_col)

        # Store basic metrics
        metrics['activity'][frame_idx] = m['activity']
        metrics['activity_crop'][frame_idx] = m['activity_crop']
        metrics['stim_activity'][frame_idx] = m['stim_activity']
        metrics['stimulus_blob_area'][frame_idx] = m['stimulus_blob_area']
        metrics['stimulus_blob_extent'][frame_idx] = m['stimulus_blob_extent']
        metrics['valid_blob_count'][frame_idx] = m['valid_blob_count']
        metrics['morans_I'][frame_idx] = m['morans_I']

        # Continue tracking velocity during decay
        extent_history.append(m['stimulus_blob_extent'])
        velocity = compute_propagation_velocity(extent_history)
        metrics['propagation_velocity'][frame_idx] = velocity

        # Compute wavefront anisotropy
        anisotropy, _ = compute_wavefront_anisotropy(
            m['wavefront_pixels'], center_row, center_col
        )
        metrics['wavefront_anisotropy'][frame_idx] = anisotropy

        frame_idx += 1

    # Compute derived metrics
    stim_start = PRE_STIM_FRAMES
    stim_end = PRE_STIM_FRAMES + stim_duration

    # Time to max extent (frames from stim onset)
    max_extent_idx = np.argmax(metrics['stimulus_blob_extent'])
    time_to_max_extent = max_extent_idx - stim_start

    # Max propagation velocity during stimulus
    max_propagation_velocity = np.max(metrics['propagation_velocity'][stim_start:stim_end])

    # Mean anisotropy during stimulus
    mean_anisotropy_during_stim = np.mean(metrics['wavefront_anisotropy'][stim_start:stim_end])

    # Add derived metrics and pre-stim summary
    metrics['prestim_summary'] = prestim_summary
    metrics['time_to_max_extent'] = time_to_max_extent
    metrics['max_propagation_velocity'] = max_propagation_velocity
    metrics['mean_anisotropy_during_stim'] = mean_anisotropy_during_stim

    # Return metrics and final state
    final_state = (spin_config, H)
    return metrics, final_state


def run_perturbation_job(args):
    """
    Worker function for parallel execution.

    Parameters:
    -----------
    args : tuple
        (job_info, config_dict) where job_info contains simulation parameters and job details

    Returns:
    --------
    dict with all results for this job
    """
    job_info, config_dict = args

    condition = job_info['condition']
    sim_idx = job_info['sim_idx']
    params = job_info['params']
    stim_size = job_info['stim_size']
    stim_duration = job_info['stim_duration']
    stim_mode = job_info['stim_mode']
    stim_bias = job_info.get('stim_bias', None)  # For bias-using modes

    if mode_uses_bias(stim_mode) and stim_bias is None:
        raise ValueError(f"stim_bias must be provided when stim_mode='{stim_mode}'")

    # Total frames for this duration
    total_frames = PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES

    # Run all replicates for this job - time series metrics
    timeseries_keys = [
        'activity', 'activity_crop', 'stim_activity', 'stimulus_blob_area', 'stimulus_blob_extent',
        'valid_blob_count', 'morans_I', 'propagation_velocity', 'wavefront_anisotropy'
    ]

    all_metrics = {key: np.zeros((N_REPLICATES, total_frames)) for key in timeseries_keys}

    # Derived metrics (scalars per replicate)
    all_metrics['time_to_max_extent'] = np.zeros(N_REPLICATES)
    all_metrics['max_propagation_velocity'] = np.zeros(N_REPLICATES)
    all_metrics['mean_anisotropy_during_stim'] = np.zeros(N_REPLICATES)

    # Pre-stimulus summary (collect across replicates)
    prestim_summaries = []

    # --- Single burn-in, decorrelated equilibrium samples ---
    mode_offset = STIM_MODE_SEED_OFFSET[stim_mode]
    bias_component = int(round(stim_bias * 100)) if stim_bias is not None else 0
    np.random.seed(mode_offset * 100000 + sim_idx * 10000 + stim_size * 100 + stim_duration + bias_component + SEED_OFFSET * 1000000)

    # Initialize once
    beta = params['beta']
    c = params['c']
    decay_const = params['decay_const']
    rad = params['inhibition_range']
    bias_val = params['bias']
    kernel = build_diamond_kernel(rad)
    K_sum = float(np.sum(kernel))

    spin_config = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, M))
    H = np.zeros((L, M), dtype=np.float64)

    # Burn in once to reach equilibrium
    tau = (L * M) / decay_const
    burn_in_steps = max(BURN_IN_MIN, int(BURN_IN_TAU_MULT * tau))
    for _ in range(burn_in_steps):
        spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias_val, K_sum)

    # Sample N_REPLICATES decorrelated equilibrium states along the trajectory.
    # Gap of 2*tau sweeps between samples ensures near-independence (~87% decorrelation).
    decorrelation_gap = max(DECORR_MIN, int(DECORR_TAU_MULT * tau))
    eq_states = [(spin_config.copy(), H.copy())]
    for _ in range(N_REPLICATES - 1):
        for __ in range(decorrelation_gap):
            spin_config, H = heat_bath_numba(spin_config, beta, c, decay_const, H, kernel, bias_val, K_sum)
        eq_states.append((spin_config.copy(), H.copy()))

    # Run replicates from independent equilibrium states
    for rep in range(N_REPLICATES):
        metrics, _ = run_single_perturbation(
            params, stim_size, stim_duration, stim_mode, rep, config_dict,
            stim_bias=stim_bias, initial_state=eq_states[rep])

        # Store time series
        for key in timeseries_keys:
            all_metrics[key][rep, :] = metrics[key]

        # Store derived metrics
        all_metrics['time_to_max_extent'][rep] = metrics['time_to_max_extent']
        all_metrics['max_propagation_velocity'][rep] = metrics['max_propagation_velocity']
        all_metrics['mean_anisotropy_during_stim'][rep] = metrics['mean_anisotropy_during_stim']

        # Store pre-stim summary
        prestim_summaries.append(metrics['prestim_summary'])

    # Average pre-stim summary across replicates
    avg_prestim_summary = {}
    for key in prestim_summaries[0].keys():
        values = [ps[key] for ps in prestim_summaries]
        avg_prestim_summary[key] = float(np.mean(values))

    all_metrics['prestim_summary'] = avg_prestim_summary

    return {
        'condition': condition,
        'sim_idx': sim_idx,
        'stim_size': stim_size,
        'stim_duration': stim_duration,
        'stim_mode': stim_mode,
        'stim_bias': stim_bias if stim_bias is not None else np.nan,  # np.nan for non-bias modes
        'params': params,
        'metrics': all_metrics
    }


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_mat_file(filepath):
    """
    Load a .mat file, handling both v7.3 (HDF5) and older formats.

    Returns:
    --------
    dict-like object with file contents
    """
    try:
        # Try scipy first (works for v7 and earlier)
        return loadmat(filepath, simplify_cells=True)
    except (NotImplementedError, ValueError):
        # v7.3 format (HDF5) - use h5py
        # ValueError is raised when scipy encounters HDF5 format
        return h5py.File(filepath, 'r')


def extract_hdf5_value(obj):
    """Extract value from HDF5 dataset or group."""
    if isinstance(obj, h5py.Dataset):
        val = obj[()]
        # Handle MATLAB strings stored as uint16 arrays
        if val.dtype == np.uint16:
            return ''.join(chr(c) for c in val.flatten())
        return val
    elif isinstance(obj, h5py.Group):
        return {key: extract_hdf5_value(obj[key]) for key in obj.keys()}
    return obj


def load_comparison_results(comparison_path):
    """
    Load comparison results from MATLAB .mat file.

    Returns:
    --------
    (best_matches, global_mean_sf) : tuple
        best_matches : dict with best matches per condition
        global_mean_sf : float, mean temporal_scale_factor across all best-matched sims
    """
    print(f"Loading comparison results from: {comparison_path}")

    # Load .mat file (handles both v7 and v7.3)
    data = load_mat_file(comparison_path)
    is_hdf5 = isinstance(data, h5py.File)

    best_matches = {}

    if is_hdf5:
        # HDF5 format (MATLAB v7.3)
        # Note: structure is flat (no 'Results' wrapper)
        # - data['IsingData']['params'] contains simulation parameters
        # - data['Comparison']['<condition>'] contains best match info

        # Get IsingData params
        ising_params = data['IsingData']['params']
        beta_vals = ising_params['beta'][()].flatten()
        c_vals = ising_params['c'][()].flatten()
        decay_vals = ising_params['decay_const'][()].flatten()
        inhib_vals = ising_params['inhibition_range'][()].flatten()
        bias_vals = ising_params['bias'][()].flatten()

        comparison = data['Comparison']

        for condition in CONDITIONS:
            # Manual override: skip conds not listed; use listed indices verbatim.
            if MANUAL_INDICES is not None:
                if condition not in MANUAL_INDICES:
                    continue
                best_idx = [int(i) for i in MANUAL_INDICES[condition]]
                if not best_idx:
                    continue
                # temporal_scale_factors still pulled from the .mat (per-sim), if available
                if condition in comparison and 'temporal_scale_factors' in comparison[condition]:
                    tsf_all = comparison[condition]['temporal_scale_factors'][()].flatten()
                else:
                    tsf_all = None
                best_matches[condition] = {'indices': best_idx, 'simulations': []}
                for idx in best_idx:
                    params = {
                        'beta': float(beta_vals[idx]),
                        'c': float(c_vals[idx]),
                        'decay_const': float(decay_vals[idx]),
                        'inhibition_range': int(inhib_vals[idx]),
                        'bias': float(bias_vals[idx]),
                    }
                    best_matches[condition]['simulations'].append(params)
                if tsf_all is not None and tsf_all.size > max(best_idx):
                    best_matches[condition]['temporal_scale_factors'] = tsf_all[best_idx]
                else:
                    # No per-sim TSF available — fall back to ones so global_mean_sf computes.
                    best_matches[condition]['temporal_scale_factors'] = np.ones(len(best_idx))
                print(f"  {condition}: Loaded {len(best_idx)} manual matches (override)")
                continue

            if condition not in comparison:
                print(f"  Warning: {condition} not found in comparison results")
                continue

            cond_data = comparison[condition]

            # Get best match indices (already 0-indexed from Python np.argsort)
            best_idx = cond_data['bestMatch_idx'][()].flatten()[:N_TOP_MATCHES]
            best_idx = [int(idx) for idx in best_idx]

            best_matches[condition] = {
                'indices': best_idx,
                'simulations': []
            }

            for idx in best_idx:
                params = {
                    'beta': float(beta_vals[idx]),
                    'c': float(c_vals[idx]),
                    'decay_const': float(decay_vals[idx]),
                    'inhibition_range': int(inhib_vals[idx]),
                    'bias': float(bias_vals[idx])
                }
                best_matches[condition]['simulations'].append(params)

            # Extract temporal_scale_factors for best-matched sims
            tsf = cond_data['temporal_scale_factors'][()].flatten()
            best_matches[condition]['temporal_scale_factors'] = tsf[[int(i) for i in best_idx]]

            print(f"  {condition}: Loaded {len(best_matches[condition]['simulations'])} best matches")

        # Compute global mean temporal scale factor
        all_sf = np.concatenate([best_matches[c]['temporal_scale_factors']
                                 for c in best_matches])
        global_mean_sf = float(np.mean(all_sf))
        print(f"  Global mean temporal scale factor: {global_mean_sf:.4f}")

        data.close()

    else:
        # Standard scipy format
        if 'Results' in data:
            results = data['Results']
        else:
            results = data

        ising_data = results['IsingData']
        comparison = results['Comparison']

        for condition in CONDITIONS:
            # Manual override (scipy branch — same semantics as HDF5 above).
            if MANUAL_INDICES is not None:
                if condition not in MANUAL_INDICES:
                    continue
                best_idx = [int(i) for i in MANUAL_INDICES[condition]]
                if not best_idx:
                    continue
                best_matches[condition] = {'indices': best_idx, 'simulations': []}
                for idx in best_idx:
                    params = {
                        'beta': float(ising_data['params']['beta'][idx]),
                        'c': float(ising_data['params']['c'][idx]),
                        'decay_const': float(ising_data['params']['decay_const'][idx]),
                        'inhibition_range': int(ising_data['params']['inhibition_range'][idx]),
                        'bias': float(ising_data['params']['bias'][idx]),
                    }
                    best_matches[condition]['simulations'].append(params)
                if condition in comparison and 'temporal_scale_factors' in comparison[condition]:
                    tsf = np.asarray(comparison[condition]['temporal_scale_factors']).flatten()
                    if tsf.size > max(best_idx):
                        best_matches[condition]['temporal_scale_factors'] = tsf[best_idx]
                    else:
                        best_matches[condition]['temporal_scale_factors'] = np.ones(len(best_idx))
                else:
                    best_matches[condition]['temporal_scale_factors'] = np.ones(len(best_idx))
                print(f"  {condition}: Loaded {len(best_idx)} manual matches (override)")
                continue

            if condition not in comparison:
                print(f"  Warning: {condition} not found in comparison results")
                continue

            cond_data = comparison[condition]

            # Get best match indices (already 0-indexed from Python np.argsort)
            best_idx = cond_data['bestMatch_idx']
            if isinstance(best_idx, np.ndarray):
                best_idx = best_idx.flatten()[:N_TOP_MATCHES]
            else:
                best_idx = [best_idx]

            best_idx = [int(idx) for idx in best_idx]

            best_matches[condition] = {
                'indices': best_idx,
                'simulations': []
            }

            for idx in best_idx:
                params = {
                    'beta': float(ising_data['params']['beta'][idx]),
                    'c': float(ising_data['params']['c'][idx]),
                    'decay_const': float(ising_data['params']['decay_const'][idx]),
                    'inhibition_range': int(ising_data['params']['inhibition_range'][idx]),
                    'bias': float(ising_data['params']['bias'][idx])
                }
                best_matches[condition]['simulations'].append(params)

            # Extract temporal_scale_factors for best-matched sims
            tsf = np.asarray(cond_data['temporal_scale_factors']).flatten()
            best_matches[condition]['temporal_scale_factors'] = tsf[[int(i) for i in best_idx]]

            print(f"  {condition}: Loaded {len(best_matches[condition]['simulations'])} best matches")

        # Compute global mean temporal scale factor
        all_sf = np.concatenate([best_matches[c]['temporal_scale_factors']
                                 for c in best_matches])
        global_mean_sf = float(np.mean(all_sf))
        print(f"  Global mean temporal scale factor: {global_mean_sf:.4f}")

    return best_matches, global_mean_sf


def find_comparison_results(ising_data_path):
    """
    Find the most recent comparison results file.

    Returns:
    --------
    str : Path to comparison results file
    """
    # Look for comparison results in standard locations
    search_patterns = [
        os.path.join(ising_data_path, 'IsingComparison', 'IsingComparison_Results_*.mat'),
        os.path.join(ising_data_path, 'IsingComparison', '*', 'IsingComparison_Results_*.mat'),
        os.path.join(ising_data_path, '..', 'IsingComparison', 'IsingComparison_Results_*.mat'),
        os.path.join(ising_data_path, '..', 'IsingComparison', '*', 'IsingComparison_Results_*.mat'),
        os.path.join(ising_data_path, 'IsingComparison_Results_*.mat'),
    ]

    for pattern in search_patterns:
        files = glob(pattern)
        if files:
            # Return most recent
            files.sort(key=os.path.getmtime, reverse=True)
            return files[0]

    raise FileNotFoundError(
        f"Could not find comparison results. Searched patterns:\n" +
        "\n".join(search_patterns)
    )


# =============================================================================
# Job Generation and Execution
# =============================================================================

def generate_all_jobs(best_matches, stimulus_durations):
    """
    Generate list of all perturbation jobs to run.

    For bias-using modes (any mode where mode_uses_bias() is True — i.e.
    'bias' and 'double_pulse_bias[N]'), generates separate jobs for each
    stimulus bias value. For 'clamped' and 'double_pulse[N]' modes,
    stim_bias is None.

    Parameters:
    -----------
    best_matches : dict
        Best-matched simulations per condition
    stimulus_durations : list of int
        MC sweep durations to test

    Returns:
    --------
    list of job_info dicts
    """
    jobs = []

    active_modes = STIMULUS_MODES
    if MODES_FILTER is not None:
        active_modes = [m for m in STIMULUS_MODES if m in MODES_FILTER]
        if not active_modes:
            print(f"WARNING: --modes filter {MODES_FILTER} matched no entries in "
                  f"STIMULUS_MODES {STIMULUS_MODES}; no jobs generated.")

    for condition in CONDITIONS:
        if condition not in best_matches:
            continue
        if COND_FILTER is not None and condition not in COND_FILTER:
            continue

        for sim_idx, params in enumerate(best_matches[condition]['simulations']):
            if SIM_FILTER is not None and sim_idx not in SIM_FILTER:
                continue
            for stim_size in STIMULUS_SIZES:
                if SIZE_FILTER is not None and stim_size not in SIZE_FILTER:
                    continue
                for stim_duration in stimulus_durations:
                    for stim_mode in active_modes:
                        if mode_uses_bias(stim_mode):
                            # For bias-using modes, create jobs for each bias value
                            for stim_bias in STIMULUS_BIAS_VALUES:
                                if BIAS_FILTER is not None and not any(
                                        abs(stim_bias - bf) < 1e-6 for bf in BIAS_FILTER):
                                    continue
                                jobs.append({
                                    'condition': condition,
                                    'sim_idx': sim_idx,
                                    'params': params,
                                    'stim_size': stim_size,
                                    'stim_duration': stim_duration,
                                    'stim_mode': stim_mode,
                                    'stim_bias': stim_bias
                                })
                        else:
                            # For clamped and double_pulse, no stimulus bias.
                            # Skip non-bias modes when BIAS_FILTER is set —
                            # the user explicitly targets specific biases.
                            if BIAS_FILTER is not None:
                                continue
                            jobs.append({
                                'condition': condition,
                                'sim_idx': sim_idx,
                                'params': params,
                                'stim_size': stim_size,
                                'stim_duration': stim_duration,
                                'stim_mode': stim_mode,
                                'stim_bias': None
                            })

    return jobs


def save_single_result(result, output_dir, format_type):
    """
    Save a single perturbation result to disk and return lightweight metadata.

    Parameters:
    -----------
    result : dict
        Full result dict from run_perturbation_job()
    output_dir : str
        Output directory
    format_type : str
        'mat', 'npz', or 'both'

    Returns:
    --------
    dict with lightweight metadata (no metrics arrays)
    """
    condition = result['condition']
    sim_idx = result['sim_idx']
    stim_size = result['stim_size']
    stim_duration = result['stim_duration']
    stim_mode = result['stim_mode']
    stim_bias = result.get('stim_bias', None)

    # Generate base filename via shared helper
    base_filename = make_result_filename(result)

    if format_type in ('mat', 'both'):
        output_file_mat = os.path.join(output_dir, base_filename + '.mat')
        savemat(output_file_mat, result, do_compression=True)

    if format_type in ('npz', 'both'):
        output_file_npz = os.path.join(output_dir, base_filename + '.npz')
        np.savez_compressed(output_file_npz, **result)

    # Return lightweight metadata only (no metrics arrays)
    return {
        'condition': condition,
        'sim_idx': sim_idx,
        'stim_size': stim_size,
        'stim_duration': stim_duration,
        'stim_mode': stim_mode,
        'stim_bias': stim_bias,
    }


def run_all_parallel(output_dir, workers, comparison_path=None, ising_data_path=None, format_type='both'):
    """
    Run all perturbation experiments in parallel.

    Parameters:
    -----------
    output_dir : str
        Output directory for results
    workers : int
        Number of parallel workers
    comparison_path : str, optional
        Path to comparison results .mat file
    ising_data_path : str, optional
        Path to Ising simulation data directory
    format_type : str
        'mat', 'npz', or 'both' - output format(s) to save
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find and load comparison results
    if comparison_path is None:
        if ising_data_path is None:
            ising_data_path = r'IsingModelData_39x78_100K'
        comparison_path = find_comparison_results(ising_data_path)

    best_matches, global_mean_sf = load_comparison_results(comparison_path)
    stimulus_durations = compute_stimulus_durations(global_mean_sf)

    # Generate all jobs
    all_jobs = generate_all_jobs(best_matches, stimulus_durations)

    print(f"\nTotal jobs: {len(all_jobs)}")
    print(f"  Conditions: {len([c for c in CONDITIONS if c in best_matches])}")
    print(f"  Simulations per condition: {N_TOP_MATCHES}")
    print(f"  Stimulus sizes: {len(STIMULUS_SIZES)}")
    print(f"  Stimulus durations (MC sweeps): {stimulus_durations}")
    print(f"  Real-time equivalents: {[round(d * global_mean_sf / SAMPLING_RATE, 2) for d in stimulus_durations]}s")
    print(f"  Stimulus modes: {len(STIMULUS_MODES)}")
    print(f"  Replicates per job: {N_REPLICATES}")
    print(f"  Workers: {workers}")
    print(f"  Format: {format_type}")
    print()

    config_dict = {
        'L': L, 'M': M,
        'pre_stim_frames': PRE_STIM_FRAMES,
        'stimulus_durations': stimulus_durations,
        'post_stim_frames': POST_STIM_FRAMES,
        'n_replicates': N_REPLICATES,
        'stimulus_sizes': STIMULUS_SIZES,
        'stimulus_modes': STIMULUS_MODES,
        'global_mean_sf': global_mean_sf,
        'target_duration_seconds': TARGET_DURATION_SECONDS,
        'sampling_rate': SAMPLING_RATE,
    }

    # Prepare arguments for parallel execution
    args_list = [(job, config_dict) for job in all_jobs]

    start_time = time.time()

    # Run in parallel, saving each result immediately to reduce memory
    results_metadata = []
    with Pool(workers) as pool:
        for result in tqdm(
            pool.imap(run_perturbation_job, args_list),
            total=len(all_jobs),
            desc="Perturbation experiments"
        ):
            metadata = save_single_result(result, output_dir, format_type)
            results_metadata.append(metadata)
            del result
            gc.collect()

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Individual results saved to: {output_dir}")
    print(f"Use --combine to aggregate into a single file.")

    return results_metadata


def organize_results(results_list, best_matches, config_dict):
    """
    Organize results into structured format for MATLAB.

    Structure:
    - experiments[condition][sim_key]['clamped'][size_key][dur_key] = metrics
    - experiments[condition][sim_key]['double_pulse'][size_key][dur_key] = metrics
    - experiments[condition][sim_key]['bias'][size_key][dur_key][bias_key] = metrics
    """
    # Initialize structure
    experiments = {}

    for condition in CONDITIONS:
        if condition not in best_matches:
            continue

        experiments[condition] = {}
        for sim_idx in range(N_TOP_MATCHES):
            sim_key = f'sim_{sim_idx}'
            experiments[condition][sim_key] = {
                'params': best_matches[condition]['simulations'][sim_idx],
            }
            for mode in STIMULUS_MODES:
                experiments[condition][sim_key][mode] = {}

            for stim_size in STIMULUS_SIZES:
                size_key = f'size_{stim_size}'
                for mode in STIMULUS_MODES:
                    experiments[condition][sim_key][mode][size_key] = {}

                for stim_duration in config_dict['stimulus_durations']:
                    dur_key = f'dur_{stim_duration}'
                    for mode in STIMULUS_MODES:
                        if mode_uses_bias(mode):
                            # For bias-using modes, add extra level for bias values
                            experiments[condition][sim_key][mode][size_key][dur_key] = {}
                            for stim_bias in STIMULUS_BIAS_VALUES:
                                bias_key = f'bias_{stim_bias:.2f}'.replace('.', 'p')
                                experiments[condition][sim_key][mode][size_key][dur_key][bias_key] = None
                        else:
                            experiments[condition][sim_key][mode][size_key][dur_key] = None

    # Fill in results
    for result in results_list:
        condition = result['condition']
        sim_idx = result['sim_idx']
        stim_size = result['stim_size']
        stim_duration = result['stim_duration']
        stim_mode = result['stim_mode']
        stim_bias = result.get('stim_bias', None)

        sim_key = f'sim_{sim_idx}'
        size_key = f'size_{stim_size}'
        dur_key = f'dur_{stim_duration}'

        if mode_uses_bias(stim_mode) and stim_bias is not None:
            bias_key = f'bias_{stim_bias:.2f}'.replace('.', 'p')
            experiments[condition][sim_key][stim_mode][size_key][dur_key][bias_key] = result['metrics']
        elif stim_mode in experiments[condition][sim_key]:
            experiments[condition][sim_key][stim_mode][size_key][dur_key] = result['metrics']

    # Build final structure
    output = {
        'config': config_dict,
        'best_matches': best_matches,
        'experiments': experiments,
        'stimulus_sizes': np.array(STIMULUS_SIZES),
        'stimulus_durations': np.array(config_dict['stimulus_durations']),
        'stimulus_modes': STIMULUS_MODES,
        'stimulus_bias_values': np.array(STIMULUS_BIAS_VALUES),
        'conditions': CONDITIONS,
        'n_top_matches': N_TOP_MATCHES,
        'n_replicates': N_REPLICATES,
        'pre_stim_frames': PRE_STIM_FRAMES,
        'post_stim_frames': POST_STIM_FRAMES,
        'grid_size': np.array([L, M]),
        'global_mean_sf': config_dict.get('global_mean_sf', np.nan),
        # Metric descriptions for MATLAB documentation
        'metric_descriptions': {
            # Time series metrics [nReplicates x nFrames]
            'activity': 'Global fraction of +1 spins',
            'activity_crop': 'Activity in 13x26 centre crop (experimental FOV)',
            'stim_activity': 'Activity within stimulus region',
            'stimulus_blob_area': 'Connected pixels in blob touching stimulus',
            'stimulus_blob_extent': 'Max distance from stimulus center to blob edge (px)',
            'valid_blob_count': 'Number of blobs >= MIN_BLOB_SIZE',
            'morans_I': 'Spatial autocorrelation (-1=dispersed, 0=random, +1=clustered)',
            'propagation_velocity': 'Instantaneous wavefront velocity (px/frame)',
            'wavefront_anisotropy': 'Directional bias of spread (0=isotropic, 1=directional)',
            # Derived metrics [nReplicates]
            'time_to_max_extent': 'Frames from stim onset to max extent',
            'max_propagation_velocity': 'Peak velocity during stimulus period',
            'mean_anisotropy_during_stim': 'Mean anisotropy during stimulus',
            # Pre-stimulus summary (averaged across replicates)
            'prestim_summary': {
                'mean_activity': 'Mean activity in 200 pre-stim frames',
                'std_activity': 'Std of activity in pre-stim period',
                'max_activity': 'Peak spontaneous activity',
                'mean_blob_count': 'Average blob count in pre-stim',
                'max_blob_count': 'Max blob count in pre-stim',
                'mean_morans_I': 'Mean Morans I in pre-stim (baseline clustering)',
                'stim_site_activity': 'Activity at stimulus site before stim',
            }
        }
    }

    return output


def make_result_filename(job):
    """Build the base filename for a perturbation result (no extension)."""
    stim_bias = job.get('stim_bias', None)
    has_bias = (stim_bias is not None
                and not (isinstance(stim_bias, float) and np.isnan(stim_bias)))
    if mode_uses_bias(job['stim_mode']) and has_bias:
        bias_str = f"_bias{stim_bias:.2f}".replace('.', 'p')
        return f'perturb_{job["condition"]}_sim{job["sim_idx"]}_size{job["stim_size"]}_dur{job["stim_duration"]}_{job["stim_mode"]}{bias_str}'
    return f'perturb_{job["condition"]}_sim{job["sim_idx"]}_size{job["stim_size"]}_dur{job["stim_duration"]}_{job["stim_mode"]}'


def run_single_job_cli(index, output_dir, comparison_path=None, ising_data_path=None, format_type='both', force=False):
    """
    Run a single job by index (for SLURM array jobs).

    Parameters:
    -----------
    index : int
        Job index
    output_dir : str
        Output directory for results
    comparison_path : str, optional
        Path to comparison results .mat file
    ising_data_path : str, optional
        Path to Ising simulation data directory
    format_type : str
        'mat', 'npz', or 'both' - output format(s) to save
    force : bool
        If False (default), skip jobs whose output file already exists
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find and load comparison results
    if comparison_path is None:
        if ising_data_path is None:
            ising_data_path = r'IsingModelData_39x78_100K'
        comparison_path = find_comparison_results(ising_data_path)

    best_matches, global_mean_sf = load_comparison_results(comparison_path)
    stimulus_durations = compute_stimulus_durations(global_mean_sf)

    # Generate all jobs
    all_jobs = generate_all_jobs(best_matches, stimulus_durations)

    if index < 0 or index >= len(all_jobs):
        raise ValueError(f"Index {index} out of range [0, {len(all_jobs)-1}]")

    job = all_jobs[index]
    base_filename = make_result_filename(job)

    # Skip if output already exists
    if not force:
        mat_exists = os.path.isfile(os.path.join(output_dir, base_filename + '.mat'))
        npz_exists = os.path.isfile(os.path.join(output_dir, base_filename + '.npz'))
        if mat_exists or npz_exists:
            print(f"SKIP job {index}/{len(all_jobs)-1}: {base_filename} (output exists, use --force to override)")
            return

    print(f"Running job {index}/{len(all_jobs)-1}")
    print(f"  Condition: {job['condition']}")
    print(f"  Simulation: {job['sim_idx']}")
    print(f"  Params: beta={job['params']['beta']}, c={job['params']['c']}")
    print(f"  Stimulus size: {job['stim_size']}")
    print(f"  Stimulus duration: {job['stim_duration']}")
    print(f"  Stimulus mode: {job['stim_mode']}")
    if job.get('stim_bias') is not None:
        print(f"  Stimulus bias: {job['stim_bias']}")
    print(f"  Format: {format_type}")

    config_dict = {
        'L': L, 'M': M,
        'pre_stim_frames': PRE_STIM_FRAMES,
        'stimulus_durations': stimulus_durations,
        'post_stim_frames': POST_STIM_FRAMES,
        'n_replicates': N_REPLICATES,
        'global_mean_sf': global_mean_sf,
        'target_duration_seconds': TARGET_DURATION_SECONDS,
        'sampling_rate': SAMPLING_RATE,
    }

    start_time = time.time()
    result = run_perturbation_job((job, config_dict))
    elapsed = time.time() - start_time

    print(f"  Completed in {elapsed:.1f} seconds")

    # Save .mat file
    if format_type in ('mat', 'both'):
        output_file_mat = os.path.join(output_dir, base_filename + '.mat')
        savemat(output_file_mat, result, do_compression=True)
        print(f"  Saved to: {output_file_mat}")

    # Save .npz file
    if format_type in ('npz', 'both'):
        output_file_npz = os.path.join(output_dir, base_filename + '.npz')
        np.savez_compressed(output_file_npz, **result)
        print(f"  Saved to: {output_file_npz}")


def run_batch_jobs_cli(array_index, batch_size, output_dir, comparison_path=None, ising_data_path=None, format_type='both', force=False):
    """
    Run a batch of jobs for a single SLURM array task.

    This function runs multiple sequential jobs per array task to reduce
    scheduling overhead. The comparison results are loaded once and reused
    across all jobs in the batch.

    Parameters:
    -----------
    array_index : int
        SLURM array task index (0 to num_array_tasks-1)
    batch_size : int
        Number of jobs to run per array task
    output_dir : str
        Output directory for results
    comparison_path : str, optional
        Path to comparison results .mat file
    ising_data_path : str, optional
        Path to Ising simulation data directory
    format_type : str
        'mat', 'npz', or 'both' - output format(s) to save
    force : bool
        If False (default), skip jobs whose output file already exists
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find and load comparison results ONCE for the entire batch
    if comparison_path is None:
        if ising_data_path is None:
            ising_data_path = r'IsingModelData_39x78_100K'
        comparison_path = find_comparison_results(ising_data_path)

    print(f"Loading comparison results (cached for batch)...")
    batch_start_time = time.time()
    best_matches, global_mean_sf = load_comparison_results(comparison_path)
    stimulus_durations = compute_stimulus_durations(global_mean_sf)
    load_elapsed = time.time() - batch_start_time
    print(f"  Loaded in {load_elapsed:.1f} seconds")

    # Generate all jobs
    all_jobs = generate_all_jobs(best_matches, stimulus_durations)
    total_jobs = len(all_jobs)

    # Calculate job range for this array task
    job_start = array_index * batch_size
    job_end = min(job_start + batch_size, total_jobs)

    if job_start >= total_jobs:
        print(f"Array index {array_index} exceeds total jobs ({total_jobs}). Nothing to do.")
        return

    print(f"\n{'='*60}")
    print(f"Batch execution: array_index={array_index}, batch_size={batch_size}")
    print(f"Running jobs {job_start} to {job_end-1} (of {total_jobs} total)")
    print(f"{'='*60}\n")

    config_dict = {
        'L': L, 'M': M,
        'pre_stim_frames': PRE_STIM_FRAMES,
        'stimulus_durations': stimulus_durations,
        'post_stim_frames': POST_STIM_FRAMES,
        'n_replicates': N_REPLICATES,
        'global_mean_sf': global_mean_sf,
        'target_duration_seconds': TARGET_DURATION_SECONDS,
        'sampling_rate': SAMPLING_RATE,
    }

    # Run each job in the batch sequentially
    jobs_completed = 0
    jobs_failed = 0
    jobs_skipped = 0

    for job_idx in range(job_start, job_end):
        job = all_jobs[job_idx]
        base_filename = make_result_filename(job)

        # Skip if output already exists
        if not force:
            mat_exists = os.path.isfile(os.path.join(output_dir, base_filename + '.mat'))
            npz_exists = os.path.isfile(os.path.join(output_dir, base_filename + '.npz'))
            if mat_exists or npz_exists:
                print(f"  SKIP job {job_idx}: {base_filename} (output exists)")
                jobs_skipped += 1
                continue

        print(f"\n--- Job {job_idx}/{total_jobs-1} ({jobs_completed+1}/{job_end-job_start} in batch) ---")
        print(f"  Condition: {job['condition']}")
        print(f"  Simulation: {job['sim_idx']}")
        print(f"  Params: beta={job['params']['beta']:.4f}, c={job['params']['c']:.4f}")
        print(f"  Stimulus: size={job['stim_size']}, dur={job['stim_duration']}, mode={job['stim_mode']}")
        if job.get('stim_bias') is not None:
            print(f"  Stimulus bias: {job['stim_bias']}")

        try:
            start_time = time.time()
            result = run_perturbation_job((job, config_dict))
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f} seconds")

            # Save .mat file
            if format_type in ('mat', 'both'):
                output_file_mat = os.path.join(output_dir, base_filename + '.mat')
                savemat(output_file_mat, result, do_compression=True)
                print(f"  Saved: {base_filename}.mat")

            # Save .npz file
            if format_type in ('npz', 'both'):
                output_file_npz = os.path.join(output_dir, base_filename + '.npz')
                np.savez_compressed(output_file_npz, **result)
                print(f"  Saved: {base_filename}.npz")

            jobs_completed += 1

            # Explicit cleanup to prevent memory accumulation across batch jobs
            del result
            gc.collect()

        except Exception as e:
            print(f"  ERROR: {e}")
            jobs_failed += 1

    # Summary
    total_elapsed = time.time() - batch_start_time
    jobs_ran = jobs_completed + jobs_failed
    print(f"\n{'='*60}")
    print(f"Batch complete: {jobs_completed} succeeded, {jobs_failed} failed, {jobs_skipped} skipped")
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    if jobs_ran > 0:
        print(f"Average time per job: {total_elapsed/jobs_ran:.1f} seconds")
    print(f"{'='*60}")


def print_scan_report(comparison_path=None, ising_data_path=None):
    """
    Print a report showing what jobs would be run.
    """
    # Find and load comparison results
    if comparison_path is None:
        if ising_data_path is None:
            ising_data_path = r'IsingModelData_39x78_100K'
        try:
            comparison_path = find_comparison_results(ising_data_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

    best_matches, global_mean_sf = load_comparison_results(comparison_path)
    stimulus_durations = compute_stimulus_durations(global_mean_sf)
    all_jobs = generate_all_jobs(best_matches, stimulus_durations)

    print("\n" + "=" * 60)
    print("  Ising Perturbation Experiment Plan")
    print("=" * 60)
    print(f"\nComparison results: {comparison_path}")
    print(f"\nGrid size: {L} x {M}")
    print(f"Frame structure: {PRE_STIM_FRAMES} pre + [duration] stim + {POST_STIM_FRAMES} post")
    print(f"Replicates per job: {N_REPLICATES}")
    print(f"\nStimulus sizes: {STIMULUS_SIZES}")
    print(f"Stimulus durations (MC sweeps): {stimulus_durations}")
    print(f"  Real-time equivalents: {[round(d * global_mean_sf / SAMPLING_RATE, 2) for d in stimulus_durations]}s")
    print(f"  Global mean temporal scale factor: {global_mean_sf:.4f}")
    print(f"Stimulus modes: {STIMULUS_MODES}")
    print(f"Stimulus bias values (for 'bias' mode): {STIMULUS_BIAS_VALUES}")
    print(f"\nConditions with best matches:")
    for condition in CONDITIONS:
        if condition in best_matches:
            print(f"  {condition}: {len(best_matches[condition]['simulations'])} simulations")

    print(f"\nTotal jobs: {len(all_jobs)}")
    print(f"Total individual runs: {len(all_jobs) * N_REPLICATES}")

    # Estimate time
    est_time_per_job = 30  # seconds (rough estimate)
    est_total_time = len(all_jobs) * est_time_per_job
    print(f"\nEstimated time (single core): {est_total_time/3600:.1f} hours")
    print(f"Estimated time (8 cores): {est_total_time/8/3600:.1f} hours")
    print(f"Estimated time (32 cores): {est_total_time/32/3600:.1f} hours")

    print("\n" + "=" * 60)


def combine_results(output_dir, comparison_path=None, ising_data_path=None, format_type='both', mode=None):
    """
    Combine individual job results into a single HDF5 file using streaming
    writes.  Each result file is loaded one at a time and its metrics are
    written directly to the output HDF5 file, keeping peak memory at O(1 file)
    instead of O(all files).

    Parameters:
    -----------
    output_dir : str
        Directory containing individual result files
    comparison_path : str, optional
        Path to comparison results .mat file
    ising_data_path : str, optional
        Path to Ising simulation data directory
    format_type : str
        'mat' (default) — HDF5 file readable by h5py / MATLAB h5read.
        'npz' — requires all data in memory; not recommended for large runs.
        'both' — writes HDF5 first, then npz (memory-intensive).
    mode : str, optional
        If set, combine only files for that stimulus mode and write to
        ``PerturbationResults_<mode>_<timestamp>.mat``. If None (default),
        combine ALL modes into a single ``PerturbationResults_<timestamp>.mat``
        (legacy monolithic behaviour).
    """
    from glob import glob

    if mode is not None and mode not in STIMULUS_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {STIMULUS_MODES}")

    print("Combining individual result files...")

    # Find all individual result files
    pattern = os.path.join(output_dir, 'perturb_*.mat')
    result_files = glob(pattern)

    if not result_files:
        print(f"ERROR: No result files found matching: {pattern}")
        raise SystemExit(1)

    if mode is not None:
        # Cheap pre-filter on filename. Each result file is named
        # perturb_<cond>_sim<N>_size<S>_dur<D>_<mode>[_bias<X>p<Y>].{mat,npz}
        #
        # Naive substring matching (e.g. `_<mode>_bias`) was wrong: for
        # mode='double_pulse', the substring `_double_pulse_bias` ALSO appears
        # in 'perturb_..._double_pulse_bias_bias0p25.mat' (a double_pulse_bias
        # file). For mode='bias', `_bias_bias` appears in every
        # double_pulse_bias[N] file too. The streaming-loop authoritative
        # re-check skipped the writes, but the wasted NFS file-loads timed out.
        # Anchor the regex on the `_dur<digits>_` prefix so the mode name
        # cannot match a longer mode containing it as a prefix.
        import re
        mode_pat = re.compile(rf'_dur\d+_{re.escape(mode)}(?:_bias\d+p\d+)?\.(?:mat|npz)$')
        keep = [fp for fp in result_files if mode_pat.search(os.path.basename(fp))]
        before = len(result_files)
        result_files = keep
        print(f"  Mode filter '{mode}': {len(result_files)} of {before} files match")
        if not result_files:
            print(f"ERROR: No files matched mode '{mode}' under {output_dir}")
            raise SystemExit(1)

    print(f"Found {len(result_files)} result files")

    # Load comparison results for structure
    if comparison_path is None:
        if ising_data_path is None:
            ising_data_path = r'IsingModelData_39x78_100K'
        comparison_path = find_comparison_results(ising_data_path)

    best_matches, global_mean_sf = load_comparison_results(comparison_path)
    stimulus_durations = compute_stimulus_durations(global_mean_sf)

    # Output file (per-mode if mode given, else legacy monolithic)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    mode_suffix = f'_{mode}' if mode else ''
    output_file = os.path.join(output_dir, f'PerturbationResults{mode_suffix}_{timestamp}.mat')
    print(f"Streaming results to: {output_file}")

    # Track which slots were filled for missing-job reporting
    filled_slots = set()

    # Infer max replicate count across a sample of result files
    actual_n_replicates = N_REPLICATES
    if result_files:
        max_reps = 0
        sample = result_files[::max(1, len(result_files) // 20)]  # ~20 evenly spaced files
        for _fpath in sample:
            try:
                _probe = loadmat(_fpath, simplify_cells=True)
                _act = _probe.get('metrics', {}).get('activity', None)
                if _act is not None:
                    max_reps = max(max_reps, int(np.asarray(_act).shape[0]))
                del _probe
            except Exception:
                continue
        if max_reps > 0:
            actual_n_replicates = max(max_reps, N_REPLICATES)
            if actual_n_replicates != N_REPLICATES:
                print(f"  Inferred n_replicates={actual_n_replicates} from data "
                      f"(CLI/default was {N_REPLICATES})")

    with h5py.File(output_file, 'w') as f:
        # --- Top-level metadata (same keys the old dict had) ---
        f.create_dataset('pre_stim_frames', data=np.int64(PRE_STIM_FRAMES))
        f.create_dataset('post_stim_frames', data=np.int64(POST_STIM_FRAMES))
        f.create_dataset('stimulus_durations', data=np.array(stimulus_durations, dtype=np.int64))
        f.create_dataset('stimulus_sizes', data=np.array(STIMULUS_SIZES, dtype=np.int64))
        f.create_dataset('n_top_matches', data=np.int64(N_TOP_MATCHES))
        f.create_dataset('n_replicates', data=np.int64(actual_n_replicates))
        f.create_dataset('grid_size', data=np.array([L, M], dtype=np.int64))
        f.create_dataset('global_mean_sf', data=np.float64(global_mean_sf))
        f.create_dataset('stimulus_bias_values', data=np.array(STIMULUS_BIAS_VALUES, dtype=np.float64))

        # String datasets — store as variable-length UTF-8
        str_dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('stimulus_modes', data=STIMULUS_MODES, dtype=str_dt)
        f.create_dataset('conditions', data=CONDITIONS, dtype=str_dt)

        # --- Config group (detailed metadata) ---
        cfg = f.create_group('config')
        cfg.create_dataset('L', data=np.int64(L))
        cfg.create_dataset('M', data=np.int64(M))
        cfg.create_dataset('pre_stim_frames', data=np.int64(PRE_STIM_FRAMES))
        cfg.create_dataset('post_stim_frames', data=np.int64(POST_STIM_FRAMES))
        cfg.create_dataset('n_replicates', data=np.int64(actual_n_replicates))
        cfg.create_dataset('stimulus_durations', data=np.array(stimulus_durations, dtype=np.int64))
        cfg.create_dataset('stimulus_sizes', data=np.array(STIMULUS_SIZES, dtype=np.int64))
        cfg.create_dataset('stimulus_bias_values', data=np.array(STIMULUS_BIAS_VALUES, dtype=np.float64))
        cfg.create_dataset('global_mean_sf', data=np.float64(global_mean_sf))
        cfg.create_dataset('target_duration_seconds', data=np.array(TARGET_DURATION_SECONDS, dtype=np.float64))
        cfg.create_dataset('sampling_rate', data=np.float64(SAMPLING_RATE))
        cfg.create_dataset('seed_offset', data=np.int64(SEED_OFFSET))
        cfg.create_dataset('stimulus_modes', data=STIMULUS_MODES, dtype=str_dt)

        # --- Best matches ---
        bm_grp = f.create_group('best_matches')
        for cond in CONDITIONS:
            if cond not in best_matches:
                continue
            cond_grp = bm_grp.create_group(cond)
            sims = best_matches[cond]['simulations']
            for i, sim_params in enumerate(sims):
                sim_grp = cond_grp.create_group(f'sim_{i}')
                for key, val in sim_params.items():
                    sim_grp.create_dataset(key, data=np.float64(val))

        # --- Experiments: stream individual result files ---
        exp_grp = f.create_group('experiments')

        # Set of expected durations for the current sweep — files with
        # `stim_duration` outside this set are legacy artifacts that the
        # MATLAB analyzer would never index anyway, so drop them.
        expected_durations = set(int(d) for d in stimulus_durations)

        loaded = 0
        errors = 0
        skipped_shape = 0
        skipped_dur   = 0
        for i, fpath in enumerate(tqdm(result_files, desc="Loading results")):
            try:
                data = loadmat(fpath, simplify_cells=True)
                condition = str(data['condition'])
                sim_idx = int(data['sim_idx'])
                stim_size = int(data['stim_size'])
                stim_duration = int(data['stim_duration'])
                stim_mode = str(data['stim_mode'])
                stim_bias = data.get('stim_bias', None)

                # Authoritative re-check: filename heuristic above is fast but
                # not bulletproof, so verify against the file's own metadata.
                if mode is not None and stim_mode != mode:
                    del data
                    continue

                # Drop legacy files whose duration isn't in the current sweep
                # (e.g. older runs with different temporal_scale_factor produced
                # dur_141 / dur_354 / dur_35 etc.). They'd otherwise bloat the
                # HDF5 file and the analyzer wouldn't index them anyway.
                if stim_duration not in expected_durations:
                    skipped_dur += 1
                    if skipped_dur <= 5:
                        print(f"  SKIP legacy duration: {os.path.basename(fpath)} "
                              f"dur={stim_duration} not in {sorted(expected_durations)}")
                    del data
                    continue

                # Drop legacy files whose activity-array frame count doesn't
                # match PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES.
                # Earlier sweeps used different PRE/POST values; the resulting
                # shape mismatches would crash MATLAB at line 502 / 448.
                expected_frames = PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES
                act = np.asarray(data.get('metrics', {}).get('activity'))
                if act.ndim != 2 or act.shape[1] != expected_frames:
                    skipped_shape += 1
                    if skipped_shape <= 5:
                        print(f"  SKIP shape-mismatch: {os.path.basename(fpath)} "
                              f"activity_shape={act.shape if act.size else 'missing'} "
                              f"expected_frames={expected_frames}")
                    del data
                    continue

                sim_key = f'sim_{sim_idx}'
                size_key = f'size_{stim_size}'
                dur_key = f'dur_{stim_duration}'

                # Build HDF5 group path
                if mode_uses_bias(stim_mode) and stim_bias is not None:
                    if isinstance(stim_bias, np.ndarray):
                        stim_bias = float(stim_bias.item())
                    # Skip NaN bias values (clamped/double_pulse stored as NaN)
                    if np.isnan(stim_bias):
                        continue
                    bias_key = f'bias_{stim_bias:.2f}'.replace('.', 'p')
                    group_path = f'{condition}/{sim_key}/{stim_mode}/{size_key}/{dur_key}/{bias_key}'
                    slot_key = (condition, sim_idx, stim_mode, stim_size, stim_duration, bias_key)
                else:
                    group_path = f'{condition}/{sim_key}/{stim_mode}/{size_key}/{dur_key}'
                    slot_key = (condition, sim_idx, stim_mode, stim_size, stim_duration)

                # Write metrics directly to HDF5
                grp = exp_grp.require_group(group_path)
                metrics = data['metrics']
                for key, val in metrics.items():
                    if isinstance(val, dict):
                        # prestim_summary — write as sub-group
                        sub_grp = grp.require_group(key)
                        for sk, sv in val.items():
                            if sk not in sub_grp:
                                sub_grp.create_dataset(sk, data=np.float64(sv))
                    else:
                        arr = np.asarray(val)
                        if key not in grp:
                            grp.create_dataset(key, data=arr,
                                               compression='gzip', compression_opts=4)

                filled_slots.add(slot_key)
                loaded += 1

                # Free memory from loadmat immediately
                del data, metrics
                if (i + 1) % 1000 == 0:
                    gc.collect()

            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                errors += 1

    print(f"Loaded {loaded}/{len(result_files)} files ({errors} errors)")
    if skipped_dur > 0 or skipped_shape > 0:
        print(f"  Skipped {skipped_dur} legacy-duration files "
              f"and {skipped_shape} shape-mismatch files (legacy data with "
              f"different PRE/POST_STIM_FRAMES — would have crashed analyze).")

    # Count missing jobs (restricted to the active mode if mode-filtered)
    jobs_per_sim = len(STIMULUS_SIZES) * len(stimulus_durations)
    modes_for_count = [mode] if mode is not None else STIMULUS_MODES
    total_expected_per_mode = {
        m: jobs_per_sim * (len(STIMULUS_BIAS_VALUES) if mode_uses_bias(m) else 1)
        for m in modes_for_count
    }
    total_expected = len(CONDITIONS) * N_TOP_MATCHES * sum(total_expected_per_mode.values())
    missing = total_expected - len(filled_slots)
    if missing > 0:
        print(f"Warning: {missing}/{total_expected} jobs did not complete")

    # Report file size
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  HDF5 file size: {size_mb:.1f} MB")

    # Optionally also save .npz (requires all data in memory — not recommended
    # for large runs; kept for backward compatibility when explicitly requested)
    if format_type in ('npz', 'both'):
        print("WARNING: .npz format requires loading all data into memory.")
        print("  For large runs (>10k files), this may cause out-of-memory errors.")
        print("  Re-loading from the HDF5 file just written...")
        try:
            with h5py.File(output_file, 'r') as hf:
                output = {}
                # Scalar / 1-D metadata
                for key in ['pre_stim_frames', 'post_stim_frames', 'stimulus_durations',
                            'stimulus_sizes', 'n_top_matches', 'n_replicates', 'grid_size',
                            'global_mean_sf', 'stimulus_bias_values']:
                    output[key] = hf[key][()]
                # String metadata
                output['stimulus_modes'] = [s.decode('utf-8') if isinstance(s, bytes) else s
                                            for s in hf['stimulus_modes'][()]]
                output['conditions'] = [s.decode('utf-8') if isinstance(s, bytes) else s
                                        for s in hf['conditions'][()]]
                # Recursively load experiments (memory-intensive!)
                def load_group(grp):
                    d = {}
                    for k in grp:
                        item = grp[k]
                        if isinstance(item, h5py.Group):
                            d[k] = load_group(item)
                        else:
                            d[k] = item[()]
                    return d
                output['experiments'] = load_group(hf['experiments'])
                output['config'] = load_group(hf['config'])
                output['best_matches'] = load_group(hf['best_matches'])

            output_file_npz = output_file.replace('.mat', '.npz')
            print(f"Saving .npz to: {output_file_npz}")
            np.savez_compressed(output_file_npz, **output)
            size_mb = os.path.getsize(output_file_npz) / (1024 * 1024)
            print(f"  .npz file size: {size_mb:.1f} MB")
            del output
            gc.collect()
        except MemoryError:
            print("ERROR: Not enough memory to save .npz format. HDF5 file is available.")

    print("Done!")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run Ising model perturbation experiments'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=r'IsingPerturbations',
        help='Output directory for results'
    )
    parser.add_argument(
        '--comparison', '-c',
        type=str,
        default=None,
        help='Path to comparison results .mat file'
    )
    parser.add_argument(
        '--ising-data', '-d',
        type=str,
        default=r'IsingModelData_39x78_100K',
        help='Path to Ising simulation data directory'
    )
    parser.add_argument(
        '--index', '-i',
        type=int,
        default=None,
        help='Run single job by index (for SLURM array jobs)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1,
        help='Number of jobs to run per array task (default: 1). Use with --index.'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=max(1, cpu_count() - 1),
        help='Number of parallel workers (default: CPU count - 1)'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['mat', 'npz', 'both'],
        default='both',
        help='Output format: mat, npz, or both (default: both)'
    )
    parser.add_argument(
        '--scan',
        action='store_true',
        help='Show what jobs would be run (no execution)'
    )
    parser.add_argument(
        '--combine',
        action='store_true',
        help='Combine individual job results into single .mat file'
    )
    parser.add_argument(
        '--mode', type=str, default=None,
        choices=STIMULUS_MODES,
        help='If set with --combine, only combine files for this stimulus mode and write '
             'PerturbationResults_<mode>_<timestamp>.mat. Omit for legacy monolithic combine.'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run even if output file already exists (default: skip existing)'
    )
    parser.add_argument(
        '--seed-offset', type=int, default=0,
        help='Additive offset to random seed for independent re-runs (default: 0)'
    )
    parser.add_argument(
        '--n-replicates', type=int, default=None,
        help='Override N_REPLICATES per job (default: 50)'
    )
    parser.add_argument(
        '--count-jobs',
        action='store_true',
        help='Print total job count and exit (for SLURM array range)'
    )
    parser.add_argument(
        '--modes', type=str, default=None,
        help=('Comma-separated list of stimulus modes to restrict generation/'
              'aggregation to (e.g. "double_pulse10,double_pulse_bias10"). '
              'Modes must be members of STIMULUS_MODES. Affects index '
              'enumeration in --index/--count-jobs runs and the fan-out in '
              'legacy --combine without --mode.')
    )
    parser.add_argument(
        '--filter-conditions', type=str, default=None,
        help='Comma-separated condition names to restrict job enumeration '
             '(e.g. "Beginner"). Used for partial/targeted re-runs.'
    )
    parser.add_argument(
        '--filter-sims', type=str, default=None,
        help='Comma-separated sim indices to restrict job enumeration '
             '(e.g. "0,1,2,3"). Used for partial/targeted re-runs.'
    )
    parser.add_argument(
        '--filter-sizes', type=str, default=None,
        help='Comma-separated stim sizes to restrict job enumeration '
             '(e.g. "2,3,4"). Used for partial/targeted re-runs.'
    )
    parser.add_argument(
        '--filter-biases', type=str, default=None,
        help='Comma-separated stim_bias values to restrict job enumeration '
             '(e.g. "0.25,0.5,1,2,4,8"). Used for partial/targeted re-runs. '
             'When set, non-bias modes (clamped, double_pulse) are skipped.'
    )
    parser.add_argument(
        '--manual-indices', type=str, default=None,
        help='Path to a JSON file mapping condition name -> list of global '
             'Ising-grid indices. When provided, bypasses the Wasserstein '
             'top-10 selection: each cond\'s best_matches list is built '
             'directly from the listed indices in order. Conds omitted from '
             'the JSON are skipped entirely (no fallback).'
    )

    args = parser.parse_args()

    # Apply global overrides from CLI
    global SEED_OFFSET, N_REPLICATES, MODES_FILTER, COND_FILTER, SIM_FILTER, SIZE_FILTER, BIAS_FILTER, MANUAL_INDICES
    SEED_OFFSET = args.seed_offset
    if args.n_replicates is not None:
        N_REPLICATES = args.n_replicates
    if args.modes is not None:
        requested = [m.strip() for m in args.modes.split(',') if m.strip()]
        unknown = [m for m in requested if m not in STIMULUS_MODES]
        if unknown:
            parser.error(
                f'--modes contains unknown mode(s): {unknown}. '
                f'Valid: {STIMULUS_MODES}')
        MODES_FILTER = requested
        print(f'Modes filter: {MODES_FILTER}')
    if args.filter_conditions is not None:
        COND_FILTER = set(c.strip() for c in args.filter_conditions.split(',') if c.strip())
        print(f'Condition filter: {sorted(COND_FILTER)}')
    if args.filter_sims is not None:
        SIM_FILTER = set(int(s.strip()) for s in args.filter_sims.split(',') if s.strip())
        print(f'Sim filter: {sorted(SIM_FILTER)}')
    if args.filter_sizes is not None:
        SIZE_FILTER = set(int(s.strip()) for s in args.filter_sizes.split(',') if s.strip())
        print(f'Size filter: {sorted(SIZE_FILTER)}')
    if args.filter_biases is not None:
        BIAS_FILTER = set(float(b.strip()) for b in args.filter_biases.split(',') if b.strip())
        print(f'Bias filter: {sorted(BIAS_FILTER)}')
    if args.manual_indices is not None:
        import json as _json
        with open(args.manual_indices, 'r') as _f:
            mi_raw = _json.load(_f)
        if not isinstance(mi_raw, dict):
            parser.error(f'--manual-indices JSON must be an object (cond -> [int, ...]).')
        # Coerce to {str: [int, ...]} and validate cond names against CONDITIONS.
        MANUAL_INDICES = {}
        for cond, idx_list in mi_raw.items():
            if cond not in CONDITIONS:
                parser.error(f'--manual-indices: unknown condition "{cond}" (valid: {CONDITIONS}).')
            MANUAL_INDICES[cond] = [int(i) for i in idx_list]
        print(f'Manual indices override: { {c: len(v) for c, v in MANUAL_INDICES.items()} }')

    # Count-jobs mode (for dynamic SLURM array range)
    if args.count_jobs:
        import sys
        comparison_path = args.comparison or find_comparison_results(args.ising_data)
        # Redirect informational output to stderr — only the count goes to stdout
        real_stdout = sys.stdout
        sys.stdout = sys.stderr
        best_matches, global_mean_sf = load_comparison_results(comparison_path)
        sys.stdout = real_stdout
        stimulus_durations = compute_stimulus_durations(global_mean_sf)
        all_jobs = generate_all_jobs(best_matches, stimulus_durations)
        print(len(all_jobs))
        return

    # Scan mode
    if args.scan:
        print_scan_report(args.comparison, args.ising_data)
        return

    # Combine mode
    if args.combine:
        combine_results(args.output, args.comparison, args.ising_data, args.format, mode=args.mode)
        return

    # Single job or batch mode (for SLURM)
    if args.index is not None:
        if args.batch_size > 1:
            # Batch mode: run multiple jobs per array task
            run_batch_jobs_cli(args.index, args.batch_size, args.output, args.comparison, args.ising_data, args.format, force=args.force)
        else:
            # Single job mode (backward compatible)
            run_single_job_cli(args.index, args.output, args.comparison, args.ising_data, args.format, force=args.force)
        return

    # Parallel mode
    run_all_parallel(args.output, args.workers, args.comparison, args.ising_data, args.format)


if __name__ == '__main__':
    main()
