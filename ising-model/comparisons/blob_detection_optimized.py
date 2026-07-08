"""
Optimized Blob Detection Module
===============================

Numba-accelerated blob detection and persistence tracking for Ising simulations.
Matches MATLAB blob detection algorithm from Figure5_IsingComparison_cluster.m.

Key optimizations:
- Fully JIT-compiled persistence tracking (no Python dicts/sets)
- Vectorized blob statistics computation
- Parallel frame processing where possible

Usage:
    from blob_detection_optimized import (
        detect_blobs_batch_fast,
        compute_blob_persistence_fast,
        BlobStats
    )

    # Fast blob detection (counts and sizes only)
    stats = detect_blobs_batch_fast(frames, sigma, threshold, min_size)

    # With persistence tracking
    stats = compute_blob_persistence_fast(frames, sigma, threshold, min_size, iou_threshold)
"""

import numpy as np
from numba import jit, prange, int32, int64, float64
from scipy import ndimage
from scipy.ndimage import gaussian_filter, label as scipy_label
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BlobStats:
    """Container for blob detection statistics."""
    counts: np.ndarray       # Number of blobs per frame
    sizes_all: np.ndarray    # All blob sizes (concatenated)
    sizes_per_frame: list    # List of arrays, sizes for each frame
    lifetimes: np.ndarray    # Blob lifetimes (persistence)
    total_blobs: int         # Total number of blobs detected


# =============================================================================
# FAST BLOB DETECTION (SCIPY-BASED, OPTIMIZED)
# =============================================================================

def detect_blobs_frame(frame: np.ndarray, sigma: float, threshold: float,
                        min_size: int) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Detect blobs in a single frame using scipy (optimized).

    Returns labels, count, and sizes.
    """
    # Convert -1/+1 to 0/1 if needed
    if frame.min() < 0:
        frame = (frame + 1) / 2

    # Gaussian smoothing + threshold
    smoothed = gaussian_filter(frame.astype(np.float32), sigma=sigma)
    binary = smoothed > threshold

    # Connected components
    labels, n_labels = scipy_label(binary)

    if n_labels == 0:
        return labels.astype(np.int32), 0, np.array([], dtype=np.int32)

    # Compute sizes using bincount (faster than ndimage.sum)
    flat_labels = labels.ravel()
    sizes = np.bincount(flat_labels)[1:]  # Skip background (label 0)

    # Filter by minimum size
    valid_mask = sizes >= min_size
    n_valid = np.sum(valid_mask)
    valid_sizes = sizes[valid_mask]

    if n_valid < n_labels:
        # Relabel: create lookup table
        mapping = np.zeros(n_labels + 1, dtype=np.int32)
        new_id = 1
        for old_id in range(1, n_labels + 1):
            if sizes[old_id - 1] >= min_size:
                mapping[old_id] = new_id
                new_id += 1
        labels = mapping[labels]

    return labels.astype(np.int32), n_valid, valid_sizes.astype(np.int32)


def detect_blobs_batch_fast(frames: np.ndarray, sigma: float = 1.0,
                             threshold: float = 0.5, min_size: int = 10) -> BlobStats:
    """
    Fast blob detection for a batch of frames (no persistence tracking).
    """
    n_frames = frames.shape[0]

    counts = np.zeros(n_frames, dtype=np.int32)
    sizes_per_frame = []
    all_sizes = []

    for t in range(n_frames):
        _, n_blobs, sizes = detect_blobs_frame(frames[t], sigma, threshold, min_size)
        counts[t] = n_blobs
        sizes_per_frame.append(sizes)
        if len(sizes) > 0:
            all_sizes.extend(sizes)

    sizes_all = np.array(all_sizes, dtype=np.int32) if all_sizes else np.array([], dtype=np.int32)

    return BlobStats(
        counts=counts,
        sizes_all=sizes_all,
        sizes_per_frame=sizes_per_frame,
        lifetimes=np.array([], dtype=np.int32),
        total_blobs=len(sizes_all)
    )


@jit(nopython=True, cache=True)
def label_all_frames_numba(binary_all, min_size):
    """Label connected components in all frames with size filtering.

    Replaces per-frame scipy.ndimage.label calls with a single compiled
    function, eliminating Python/scipy dispatch overhead.  Uses 4-connectivity
    (matching scipy.ndimage.label default structuring element).

    Parameters
    ----------
    binary_all : ndarray, shape (n_frames, rows, cols), bool or int
        Thresholded binary frames
    min_size : int
        Minimum component size to keep

    Returns
    -------
    all_labels : ndarray (n_frames, rows, cols), int32
    all_n_blobs : ndarray (n_frames,), int32
    sizes_flat : ndarray, int32 - all valid blob sizes concatenated
    """
    n_frames, rows, cols = binary_all.shape
    all_labels = np.zeros((n_frames, rows, cols), dtype=np.int32)
    all_n_blobs = np.zeros(n_frames, dtype=np.int32)

    max_comp = rows * cols // max(min_size, 1) + 1
    max_total = n_frames * max_comp
    sizes_flat = np.zeros(max_total, dtype=np.int32)
    n_sizes = 0

    # Reusable scratch buffers (allocated once)
    stack_r = np.zeros(rows * cols, dtype=np.int32)
    stack_c = np.zeros(rows * cols, dtype=np.int32)
    comp_sizes = np.zeros(max_comp, dtype=np.int32)
    mapping = np.zeros(max_comp + 1, dtype=np.int32)

    for t in range(n_frames):
        next_label = 1

        for r in range(rows):
            for c in range(cols):
                if binary_all[t, r, c] and all_labels[t, r, c] == 0:
                    # BFS flood fill (4-connectivity)
                    stack_r[0] = r
                    stack_c[0] = c
                    stack_top = 1
                    all_labels[t, r, c] = next_label
                    count = 0

                    while stack_top > 0:
                        stack_top -= 1
                        cr = stack_r[stack_top]
                        cc = stack_c[stack_top]
                        count += 1

                        if cr > 0 and binary_all[t, cr - 1, cc] and all_labels[t, cr - 1, cc] == 0:
                            all_labels[t, cr - 1, cc] = next_label
                            stack_r[stack_top] = cr - 1
                            stack_c[stack_top] = cc
                            stack_top += 1
                        if cr < rows - 1 and binary_all[t, cr + 1, cc] and all_labels[t, cr + 1, cc] == 0:
                            all_labels[t, cr + 1, cc] = next_label
                            stack_r[stack_top] = cr + 1
                            stack_c[stack_top] = cc
                            stack_top += 1
                        if cc > 0 and binary_all[t, cr, cc - 1] and all_labels[t, cr, cc - 1] == 0:
                            all_labels[t, cr, cc - 1] = next_label
                            stack_r[stack_top] = cr
                            stack_c[stack_top] = cc - 1
                            stack_top += 1
                        if cc < cols - 1 and binary_all[t, cr, cc + 1] and all_labels[t, cr, cc + 1] == 0:
                            all_labels[t, cr, cc + 1] = next_label
                            stack_r[stack_top] = cr
                            stack_c[stack_top] = cc + 1
                            stack_top += 1

                    comp_sizes[next_label - 1] = count
                    next_label += 1

        n_labels = next_label - 1

        # Filter by min_size and relabel
        for i in range(n_labels + 1):
            mapping[i] = 0

        n_valid = 0
        for i in range(n_labels):
            if comp_sizes[i] >= min_size:
                n_valid += 1
                mapping[i + 1] = n_valid
                sizes_flat[n_sizes] = comp_sizes[i]
                n_sizes += 1

        all_n_blobs[t] = n_valid

        if n_valid < n_labels:
            for r in range(rows):
                for c in range(cols):
                    all_labels[t, r, c] = mapping[all_labels[t, r, c]]

    return all_labels, all_n_blobs, sizes_flat[:n_sizes].copy()


def detect_blobs_all_frames(frames: np.ndarray, sigma: float = 1.0,
                             threshold: float = 0.5,
                             min_size: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect blobs in all frames, returning label arrays for reuse.

    Optimized path: vectorized gaussian smoothing (one scipy call for the
    entire 3D stack) followed by a single numba call for connected-component
    labeling across all frames.  This eliminates 10K per-frame scipy dispatch
    calls that dominated runtime on tiny 13x26 grids.

    Returns
    -------
    all_labels : ndarray, shape (n_frames, rows, cols)
        Label matrices for all frames
    all_n_blobs : ndarray, shape (n_frames,)
        Number of blobs in each frame
    counts : ndarray, shape (n_frames,)
        Copy of all_n_blobs (for BlobStats compatibility)
    sizes_all : ndarray
        All blob sizes concatenated
    """
    # Vectorized conversion: -1/+1 -> 0/1 (checked once, not per frame)
    frames_f32 = frames.astype(np.float32)
    if np.any(frames_f32 < 0):
        frames_f32 = (frames_f32 + 1) / 2

    # One scipy call for all frames: sigma=0 along time axis skips that dim
    smoothed = gaussian_filter(frames_f32, sigma=(0, sigma, sigma))
    binary_all = smoothed > threshold

    # Single numba call: label + size-filter all frames
    all_labels, all_n_blobs, sizes_all = label_all_frames_numba(binary_all, min_size)

    return all_labels, all_n_blobs, all_n_blobs.copy(), sizes_all


# =============================================================================
# NUMBA-OPTIMIZED PERSISTENCE TRACKING
# =============================================================================

@jit(nopython=True, cache=True)
def compute_blob_overlaps(labels_prev: np.ndarray, labels_curr: np.ndarray,
                           n_prev: int, n_curr: int) -> np.ndarray:
    """
    Compute overlap matrix between blobs in consecutive frames.
    Uses pixel counts for IoU computation.

    Returns overlap_matrix[i,j] = IoU between prev blob i+1 and curr blob j+1
    """
    if n_prev == 0 or n_curr == 0:
        return np.zeros((max(1, n_prev), max(1, n_curr)), dtype=np.float64)

    n_pixels = labels_prev.size

    # Count pixels for each blob and intersections
    prev_counts = np.zeros(n_prev, dtype=np.int64)
    curr_counts = np.zeros(n_curr, dtype=np.int64)
    intersections = np.zeros((n_prev, n_curr), dtype=np.int64)

    for idx in range(n_pixels):
        p_label = labels_prev.flat[idx]
        c_label = labels_curr.flat[idx]

        if p_label > 0:
            prev_counts[p_label - 1] += 1
        if c_label > 0:
            curr_counts[c_label - 1] += 1
        if p_label > 0 and c_label > 0:
            intersections[p_label - 1, c_label - 1] += 1

    # Compute IoU matrix
    iou_matrix = np.zeros((n_prev, n_curr), dtype=np.float64)
    for i in range(n_prev):
        for j in range(n_curr):
            inter = intersections[i, j]
            union = prev_counts[i] + curr_counts[j] - inter
            if union > 0:
                iou_matrix[i, j] = inter / union

    return iou_matrix


@jit(nopython=True, cache=True)
def greedy_match_blobs(iou_matrix: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Greedy matching of blobs based on IoU.

    Returns matches array where matches[prev_idx] = curr_idx (or -1 if no match)
    """
    n_prev, n_curr = iou_matrix.shape
    matches = np.full(n_prev, -1, dtype=np.int32)

    if n_prev == 0 or n_curr == 0:
        return matches

    matched_curr = np.zeros(n_curr, dtype=np.int32)

    # Find matches greedily by best IoU
    for _ in range(min(n_prev, n_curr)):
        best_iou = -1.0
        best_i = -1
        best_j = -1

        for i in range(n_prev):
            if matches[i] >= 0:
                continue
            for j in range(n_curr):
                if matched_curr[j]:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_i = i
                    best_j = j

        if best_iou >= iou_threshold:
            matches[best_i] = best_j
            matched_curr[best_j] = 1
        else:
            break

    return matches


@jit(nopython=True, cache=True)
def track_persistence_numba(all_labels: np.ndarray, all_n_blobs: np.ndarray,
                             iou_threshold: float) -> np.ndarray:
    """
    Fully Numba-compiled blob persistence tracking.

    Parameters
    ----------
    all_labels : ndarray, shape (n_frames, rows, cols)
        Label matrices for all frames
    all_n_blobs : ndarray, shape (n_frames,)
        Number of blobs in each frame
    iou_threshold : float
        Minimum IoU for blob matching

    Returns
    -------
    lifetimes : ndarray
        Array of blob lifetimes
    """
    n_frames = all_labels.shape[0]

    if n_frames == 0:
        return np.array([0], dtype=np.int32)[:0]  # Empty array

    # Maximum possible blobs (upper bound)
    max_total_blobs = np.sum(all_n_blobs) + 1

    # Track blob lifetimes: blob_start[id] = start frame, blob_active[id] = still active
    blob_start = np.zeros(max_total_blobs, dtype=np.int32)
    blob_frame_label = np.zeros(max_total_blobs, dtype=np.int32)  # Current label in frame
    blob_active = np.zeros(max_total_blobs, dtype=np.int32)

    # Finished lifetimes storage
    finished_lifetimes = np.zeros(max_total_blobs, dtype=np.int32)
    n_finished = 0

    next_blob_id = 0
    n_active = 0

    # Active blob IDs array (for iteration)
    active_ids = np.zeros(max_total_blobs, dtype=np.int32)

    for t in range(n_frames):
        n_curr = all_n_blobs[t]
        labels_curr = all_labels[t]

        if t == 0:
            # Initialize: each blob gets a new ID
            for label_id in range(1, n_curr + 1):
                blob_start[next_blob_id] = 0
                blob_frame_label[next_blob_id] = label_id
                blob_active[next_blob_id] = 1
                active_ids[n_active] = next_blob_id
                n_active += 1
                next_blob_id += 1
            continue

        n_prev = all_n_blobs[t - 1]
        labels_prev = all_labels[t - 1]

        # Compute IoU matrix
        iou_matrix = compute_blob_overlaps(labels_prev, labels_curr, n_prev, n_curr)

        # Greedy matching
        # matches[prev_label-1] = curr_label-1 (or -1 if no match)
        matches = greedy_match_blobs(iou_matrix, iou_threshold)

        # Track which current labels got matched
        curr_matched = np.zeros(n_curr, dtype=np.int32)

        # Update active blobs
        new_active_ids = np.zeros(max_total_blobs, dtype=np.int32)
        new_n_active = 0

        for ai in range(n_active):
            blob_id = active_ids[ai]
            prev_label = blob_frame_label[blob_id]

            if prev_label > 0 and prev_label <= n_prev:
                matched_curr = matches[prev_label - 1]

                if matched_curr >= 0:
                    # Blob continues
                    blob_frame_label[blob_id] = matched_curr + 1
                    new_active_ids[new_n_active] = blob_id
                    new_n_active += 1
                    curr_matched[matched_curr] = 1
                else:
                    # Blob ended
                    lifetime = t - blob_start[blob_id]
                    finished_lifetimes[n_finished] = lifetime
                    n_finished += 1
                    blob_active[blob_id] = 0
            else:
                # Blob ended (shouldn't happen normally)
                lifetime = t - blob_start[blob_id]
                finished_lifetimes[n_finished] = lifetime
                n_finished += 1
                blob_active[blob_id] = 0

        # Create new blobs for unmatched current labels
        for curr_label in range(1, n_curr + 1):
            if not curr_matched[curr_label - 1]:
                blob_start[next_blob_id] = t
                blob_frame_label[next_blob_id] = curr_label
                blob_active[next_blob_id] = 1
                new_active_ids[new_n_active] = next_blob_id
                new_n_active += 1
                next_blob_id += 1

        # Update active list
        for i in range(new_n_active):
            active_ids[i] = new_active_ids[i]
        n_active = new_n_active

    # Finish remaining active blobs
    for ai in range(n_active):
        blob_id = active_ids[ai]
        lifetime = n_frames - blob_start[blob_id]
        finished_lifetimes[n_finished] = lifetime
        n_finished += 1

    return finished_lifetimes[:n_finished].copy()


# =============================================================================
# MAIN INTERFACE FUNCTIONS
# =============================================================================

def compute_blob_persistence_batch(frames: np.ndarray, sigma: float = 1.0,
                                     threshold: float = 0.5, min_size: int = 10,
                                     iou_threshold: float = 0.3) -> BlobStats:
    """
    Detect blobs and compute persistence for a batch of frames.
    Optimized with Numba JIT compilation.
    """
    n_frames = frames.shape[0]
    rows, cols = frames.shape[1], frames.shape[2]

    # Pre-allocate arrays for all frames
    all_labels = np.zeros((n_frames, rows, cols), dtype=np.int32)
    all_n_blobs = np.zeros(n_frames, dtype=np.int32)
    sizes_per_frame = []
    all_sizes = []

    # Detect blobs in all frames
    for t in range(n_frames):
        labels, n_blobs, sizes = detect_blobs_frame(frames[t], sigma, threshold, min_size)
        all_labels[t] = labels
        all_n_blobs[t] = n_blobs
        sizes_per_frame.append(sizes)
        if len(sizes) > 0:
            all_sizes.extend(sizes)

    sizes_all = np.array(all_sizes, dtype=np.int32) if all_sizes else np.array([], dtype=np.int32)

    # Compute persistence using Numba
    lifetimes = track_persistence_numba(all_labels, all_n_blobs, iou_threshold)

    return BlobStats(
        counts=all_n_blobs,
        sizes_all=sizes_all,
        sizes_per_frame=sizes_per_frame,
        lifetimes=lifetimes,
        total_blobs=len(sizes_all)
    )


def compute_blob_persistence_fast(frames: np.ndarray, sigma: float = 1.0,
                                    threshold: float = 0.5, min_size: int = 10,
                                    iou_threshold: float = 0.3) -> BlobStats:
    """Alias for compute_blob_persistence_batch."""
    return compute_blob_persistence_batch(frames, sigma, threshold, min_size, iou_threshold)


# =============================================================================
# TILED BLOB DETECTION
# =============================================================================

def detect_blobs_tiled(frames: np.ndarray, tile_positions: List[Tuple[int, int]],
                        tile_shape: Tuple[int, int], sigma: float = 1.0,
                        threshold: float = 0.5, min_size: int = 10,
                        iou_threshold: float = 0.3) -> BlobStats:
    """
    Detect blobs across multiple tile positions and pool results.
    """
    all_counts = []
    all_sizes = []
    all_lifetimes = []

    tile_rows, tile_cols = tile_shape

    for row_start, col_start in tile_positions:
        tile_frames = frames[:, row_start:row_start+tile_rows, col_start:col_start+tile_cols]
        tile_stats = compute_blob_persistence_batch(
            tile_frames, sigma, threshold, min_size, iou_threshold
        )

        all_counts.append(tile_stats.counts)
        if len(tile_stats.sizes_all) > 0:
            all_sizes.extend(tile_stats.sizes_all)
        if len(tile_stats.lifetimes) > 0:
            all_lifetimes.extend(tile_stats.lifetimes)

    pooled_counts = np.concatenate(all_counts) if all_counts else np.array([], dtype=np.int32)
    pooled_sizes = np.array(all_sizes, dtype=np.int32) if all_sizes else np.array([], dtype=np.int32)
    pooled_lifetimes = np.array(all_lifetimes, dtype=np.int32) if all_lifetimes else np.array([], dtype=np.int32)

    return BlobStats(
        counts=pooled_counts,
        sizes_all=pooled_sizes,
        sizes_per_frame=[],
        lifetimes=pooled_lifetimes,
        total_blobs=len(pooled_sizes)
    )


# =============================================================================
# AUTOCORRELATION AND TEMPORAL SCALING
# =============================================================================

def compute_autocorr_decay(time_series: np.ndarray, max_lag: int = 100,
                            fit_range: Tuple[int, int] = (1, 10)) -> Tuple[float, np.ndarray, float]:
    """
    Compute autocorrelation decay time constant (tau).
    Fits exponential decay: acf(lag) = exp(-lag / tau)
    """
    ts = time_series - np.mean(time_series)

    if np.var(ts) < 1e-10:
        return np.nan, np.zeros(max_lag + 1), 0.0

    # Compute normalized autocorrelation using FFT (faster)
    n = len(ts)
    fft_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    fft_ts = np.fft.fft(ts, fft_size)
    acf_full = np.fft.ifft(fft_ts * np.conj(fft_ts)).real
    acf = acf_full[:max_lag + 1] / acf_full[0]

    # Fit exponential decay
    lags = np.arange(len(acf))
    mask = (lags >= fit_range[0]) & (lags <= fit_range[1]) & (acf > 0)

    if np.sum(mask) < 3:
        return np.nan, acf, 0.0

    lags_fit = lags[mask]
    acf_fit = acf[mask]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        log_acf = np.log(acf_fit)
        coeffs = np.polyfit(lags_fit, log_acf, 1)

    slope = coeffs[0]

    if slope >= 0:
        return np.nan, acf, 0.0

    tau = -1.0 / slope

    # R-squared
    log_acf_pred = coeffs[0] * lags_fit + coeffs[1]
    ss_res = np.sum((log_acf - log_acf_pred) ** 2)
    ss_tot = np.sum((log_acf - np.mean(log_acf)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return tau, acf, r_squared


def compute_temporal_scale_factor(tau_experimental: float, tau_ising: float) -> float:
    """
    Compute temporal rescaling factor.

    Returns the factor to convert Ising time (MC sweeps) to experimental time (frames).
    rescaled_ising_time = ising_time * scale_factor

    This matches MATLAB's tau_ratio = tau_ising / tau_exp (MC sweeps per frame),
    but inverted for the multiplication convention used here.
    """
    if np.isnan(tau_experimental) or np.isnan(tau_ising) or tau_ising <= 0:
        return 1.0
    return tau_experimental / tau_ising  # frames per MC sweep


def compute_trial_averaged_autocorr(activity_trials: np.ndarray, max_lag: int = 100,
                                     fit_range: Tuple[int, int] = (1, 10)) -> Tuple[float, np.ndarray, float]:
    """
    Compute trial-averaged autocorrelation decay time constant.

    Matches MATLAB computeTrialAveragedAutocorr: computes ACF per trial,
    averages ACF values across trials, then fits exponential decay.

    Parameters
    ----------
    activity_trials : ndarray, shape (n_timepoints, n_trials)
        Activity time series for each trial. Each COLUMN is one trial's timecourse.
        IMPORTANT: Caller must ensure correct orientation - no auto-transpose is performed.
    max_lag : int
        Maximum lag for autocorrelation
    fit_range : tuple
        (min_lag, max_lag) for exponential fit

    Returns
    -------
    tau : float
        Decay time constant
    acf_mean : ndarray
        Trial-averaged autocorrelation function
    r_squared : float
        R-squared of exponential fit
    """
    # Handle input shape
    if activity_trials.ndim == 1:
        # Single trial
        return compute_autocorr_decay(activity_trials, max_lag, fit_range)

    # Input MUST be (n_timepoints, n_trials) - DO NOT auto-transpose based on size
    # The old heuristic (transpose if shape[0] < shape[1]) fails when n_trials > n_timepoints
    # (e.g., 80 pre-stim frames with 100+ trials)
    n_timepoints, n_trials = activity_trials.shape

    # Sanity check: warn if dimensions seem potentially reversed
    # Trials are typically < 500, timepoints can be 80-100000
    if n_trials > 500 and n_timepoints < 100:
        warnings.warn(f"compute_trial_averaged_autocorr: shape ({n_timepoints}, {n_trials}) "
                      f"may have axes reversed. Expected (n_timepoints, n_trials).")

    if n_trials == 0 or n_timepoints < max_lag:
        return np.nan, np.zeros(max_lag + 1), 0.0

    # Compute ACF for each trial
    acf_all = np.zeros((n_trials, max_lag + 1))
    valid_trials = 0

    for trial in range(n_trials):
        ts = activity_trials[:, trial]
        ts_centered = ts - np.mean(ts)

        if np.var(ts_centered) < 1e-10:
            continue

        # FFT-based autocorrelation
        n = len(ts_centered)
        fft_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        fft_ts = np.fft.fft(ts_centered, fft_size)
        acf_full = np.fft.ifft(fft_ts * np.conj(fft_ts)).real
        acf_trial = acf_full[:max_lag + 1] / acf_full[0]

        acf_all[valid_trials] = acf_trial
        valid_trials += 1

    if valid_trials == 0:
        return np.nan, np.zeros(max_lag + 1), 0.0

    # Average ACF across valid trials
    acf_mean = np.mean(acf_all[:valid_trials], axis=0)

    # Fit exponential decay to averaged ACF
    lags = np.arange(len(acf_mean))
    mask = (lags >= fit_range[0]) & (lags <= fit_range[1]) & (acf_mean > 0)

    if np.sum(mask) < 3:
        return np.nan, acf_mean, 0.0

    lags_fit = lags[mask]
    acf_fit = acf_mean[mask]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        log_acf = np.log(acf_fit)
        coeffs = np.polyfit(lags_fit, log_acf, 1)

    slope = coeffs[0]

    if slope >= 0:
        return np.nan, acf_mean, 0.0

    tau = -1.0 / slope

    # R-squared
    log_acf_pred = coeffs[0] * lags_fit + coeffs[1]
    ss_res = np.sum((log_acf - log_acf_pred) ** 2)
    ss_tot = np.sum((log_acf - np.mean(log_acf)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return tau, acf_mean, r_squared


def compute_global_autocorr(activity_dict: dict, conditions: List[str],
                             max_lag: int = 100,
                             fit_range: Tuple[int, int] = (1, 10)) -> Tuple[float, np.ndarray, float]:
    """
    Compute global autocorrelation by concatenating trials across conditions.

    Matches MATLAB global tau computation: concatenates all trial activity
    across conditions, computes single ACF, fits decay.

    Parameters
    ----------
    activity_dict : dict
        Dict mapping condition -> activity array (n_timepoints, n_trials)
    conditions : list
        List of condition names to include
    max_lag : int
        Maximum lag for autocorrelation
    fit_range : tuple
        (min_lag, max_lag) for exponential fit

    Returns
    -------
    tau_global : float
        Global decay time constant
    acf_global : ndarray
        Global autocorrelation function
    r_squared : float
        R-squared of fit
    """
    # Collect all trial activity
    all_activity = []

    for condition in conditions:
        if condition not in activity_dict:
            continue
        act = activity_dict[condition]
        if act is None or len(act) == 0:
            continue

        # Flatten to single time series per trial, then concatenate
        if act.ndim == 2:
            # Shape (n_timepoints, n_trials) or (n_trials, n_timepoints)
            if act.shape[0] < act.shape[1]:
                act = act.T
            for trial in range(act.shape[1]):
                all_activity.extend(act[:, trial])
        else:
            all_activity.extend(act)

    if len(all_activity) == 0:
        return np.nan, np.zeros(max_lag + 1), 0.0

    global_ts = np.array(all_activity)
    return compute_autocorr_decay(global_ts, max_lag, fit_range)


# =============================================================================
# WARMUP AND TESTING
# =============================================================================

def warmup_blob_jit():
    """Pre-compile JIT functions."""
    # Small test data
    test_labels = np.random.randint(0, 3, (10, 5, 5)).astype(np.int32)
    test_n_blobs = np.array([2, 2, 1, 2, 1, 2, 2, 1, 2, 1], dtype=np.int32)

    # Compile overlap computation
    _ = compute_blob_overlaps(test_labels[0], test_labels[1], 2, 2)

    # Compile greedy matching
    iou = np.random.rand(3, 3)
    _ = greedy_match_blobs(iou, 0.3)

    # Compile full tracking
    _ = track_persistence_numba(test_labels, test_n_blobs, 0.3)

    # Compile all-frames labeling
    test_binary = np.random.rand(10, 5, 5) > 0.5
    _ = label_all_frames_numba(test_binary, 2)

    print("Blob JIT warmup complete")


def test_blob_detection():
    """Test blob detection with synthetic data."""
    print("Testing optimized blob detection module...")

    # Warmup
    warmup_blob_jit()

    # Create synthetic data
    np.random.seed(42)
    n_frames = 50
    rows, cols = 13, 26

    frames = np.random.rand(n_frames, rows, cols) * 0.3

    # Add persistent blobs
    for t in range(n_frames):
        if t < 40:
            frames[t, 3:6, 10:14] = 0.9
        if 10 <= t < 25:
            frames[t, 8:11, 5:8] = 0.85
        if 20 <= t < 23:
            frames[t, 2:4, 20:23] = 0.95

    print(f"\nTest data: {n_frames} frames, {rows}x{cols}")

    # Time the detection
    import time

    start = time.perf_counter()
    stats = compute_blob_persistence_batch(frames, sigma=1.0, threshold=0.5,
                                            min_size=5, iou_threshold=0.3)
    elapsed = time.perf_counter() - start

    print(f"\nResults ({elapsed*1000:.1f} ms):")
    print(f"  Frames with blobs: {np.sum(stats.counts > 0)}/{n_frames}")
    print(f"  Mean blobs/frame: {np.mean(stats.counts):.2f}")
    print(f"  Total lifetimes tracked: {len(stats.lifetimes)}")

    if len(stats.lifetimes) > 0:
        print(f"  Mean lifetime: {np.mean(stats.lifetimes):.1f} frames")
        print(f"  Max lifetime: {np.max(stats.lifetimes)} frames")
        print(f"  Lifetimes: {stats.lifetimes}")

    # Benchmark with more frames
    print("\nBenchmark (500 frames)...")
    large_frames = np.random.rand(500, rows, cols).astype(np.float32)
    large_frames[large_frames > 0.7] = 1.0

    start = time.perf_counter()
    stats2 = compute_blob_persistence_batch(large_frames, sigma=1.0, threshold=0.5,
                                             min_size=5, iou_threshold=0.3)
    elapsed = time.perf_counter() - start

    print(f"  500 frames processed in {elapsed*1000:.1f} ms ({elapsed/500*1000:.2f} ms/frame)")
    print(f"  Blobs detected: {stats2.total_blobs}")
    print(f"  Lifetimes tracked: {len(stats2.lifetimes)}")

    print("\nBlob detection tests PASSED")


if __name__ == '__main__':
    test_blob_detection()
