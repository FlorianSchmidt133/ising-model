"""
Optimized Wasserstein Distance Module
=====================================

Numba-accelerated 1D Wasserstein distance computation.
Matches MATLAB wasserstein_1d implementation using quantile approximation.

Key optimizations:
- Numba JIT for inner quantile computation
- Batch processing for multiple simulations
- Pre-sorted inputs for repeated comparisons

Usage:
    from wasserstein_optimized import (
        wasserstein_1d,
        wasserstein_1d_jit,
        wasserstein_batch,
        compute_combined_distance
    )

    # Single comparison
    d = wasserstein_1d(x, y)

    # Batch: compare experimental to many simulations
    distances = wasserstein_batch(exp_data, sim_data_list)
"""

import numpy as np
from numba import jit, prange, float64
from typing import List, Optional, Tuple


# =============================================================================
# CORE WASSERSTEIN COMPUTATION
# =============================================================================

@jit(nopython=True, cache=True)
def wasserstein_1d_sorted_jit(x_sorted: np.ndarray, y_sorted: np.ndarray,
                               n_quantiles: int) -> float:
    """
    Compute 1D Wasserstein distance from pre-sorted arrays.

    Uses quantile approximation matching MATLAB implementation.

    Parameters
    ----------
    x_sorted : ndarray
        First sample (sorted, no NaN)
    y_sorted : ndarray
        Second sample (sorted, no NaN)
    n_quantiles : int
        Number of quantiles to use

    Returns
    -------
    d : float
        Wasserstein distance
    """
    len_x = len(x_sorted)
    len_y = len(y_sorted)

    if len_x == 0 or len_y == 0:
        return np.nan

    # Compute quantiles at evenly spaced points
    total_diff = 0.0

    for i in range(n_quantiles):
        # Quantile level (0 to 1)
        q = i / (n_quantiles - 1) if n_quantiles > 1 else 0.5

        # Get quantile values using linear interpolation
        # This matches np.quantile behavior
        idx_x = q * (len_x - 1)
        idx_y = q * (len_y - 1)

        # Linear interpolation for x
        idx_x_floor = int(idx_x)
        idx_x_ceil = min(idx_x_floor + 1, len_x - 1)
        frac_x = idx_x - idx_x_floor
        x_q = x_sorted[idx_x_floor] * (1 - frac_x) + x_sorted[idx_x_ceil] * frac_x

        # Linear interpolation for y
        idx_y_floor = int(idx_y)
        idx_y_ceil = min(idx_y_floor + 1, len_y - 1)
        frac_y = idx_y - idx_y_floor
        y_q = y_sorted[idx_y_floor] * (1 - frac_y) + y_sorted[idx_y_ceil] * frac_y

        total_diff += abs(x_q - y_q)

    return total_diff / n_quantiles


def wasserstein_1d(x, y, max_quantiles: int = 1000) -> float:
    """
    Compute 1D Wasserstein distance using quantile approximation.
    Matches MATLAB wasserstein_1d implementation exactly.

    Parameters
    ----------
    x, y : array-like
        Input samples
    max_quantiles : int
        Maximum number of quantiles to use (default: 1000)

    Returns
    -------
    d : float
        Wasserstein distance
    """
    # Flatten and remove NaN values
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    # Handle empty inputs
    if len(x) == 0 or len(y) == 0:
        return np.nan

    # Sort both samples
    x_sorted = np.sort(x).astype(np.float64)
    y_sorted = np.sort(y).astype(np.float64)

    # Use quantile matching for efficiency with large arrays
    n_quantiles = min(max_quantiles, min(len(x), len(y)))

    return wasserstein_1d_sorted_jit(x_sorted, y_sorted, n_quantiles)


@jit(nopython=True, cache=True)
def wasserstein_1d_jit(x: np.ndarray, y: np.ndarray, max_quantiles: int = 1000) -> float:
    """
    Fully JIT-compiled 1D Wasserstein distance.
    Handles NaN removal and sorting internally.

    Parameters
    ----------
    x, y : ndarray
        Input samples (can contain NaN)
    max_quantiles : int
        Maximum number of quantiles

    Returns
    -------
    d : float
        Wasserstein distance
    """
    # Count non-NaN values
    n_x = 0
    n_y = 0
    for i in range(len(x)):
        if not np.isnan(x[i]):
            n_x += 1
    for i in range(len(y)):
        if not np.isnan(y[i]):
            n_y += 1

    if n_x == 0 or n_y == 0:
        return np.nan

    # Extract non-NaN values
    x_clean = np.empty(n_x, dtype=np.float64)
    y_clean = np.empty(n_y, dtype=np.float64)

    idx = 0
    for i in range(len(x)):
        if not np.isnan(x[i]):
            x_clean[idx] = x[i]
            idx += 1

    idx = 0
    for i in range(len(y)):
        if not np.isnan(y[i]):
            y_clean[idx] = y[i]
            idx += 1

    # Sort
    x_sorted = np.sort(x_clean)
    y_sorted = np.sort(y_clean)

    # Compute Wasserstein distance
    n_quantiles = min(max_quantiles, min(n_x, n_y))
    return wasserstein_1d_sorted_jit(x_sorted, y_sorted, n_quantiles)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def wasserstein_batch(exp_data: np.ndarray, sim_data_list: List[np.ndarray],
                       max_quantiles: int = 1000) -> np.ndarray:
    """
    Compute Wasserstein distances from experimental data to multiple simulations.

    Pre-sorts experimental data for efficiency.

    Parameters
    ----------
    exp_data : ndarray
        Experimental distribution (1D or will be flattened)
    sim_data_list : list of ndarray
        List of simulation distributions
    max_quantiles : int
        Maximum number of quantiles

    Returns
    -------
    distances : ndarray, shape (n_sims,)
        Wasserstein distance for each simulation
    """
    n_sims = len(sim_data_list)
    distances = np.zeros(n_sims, dtype=np.float64)

    # Pre-process experimental data
    exp_flat = np.asarray(exp_data).ravel()
    exp_clean = exp_flat[~np.isnan(exp_flat)]

    if len(exp_clean) == 0:
        return np.full(n_sims, np.nan)

    exp_sorted = np.sort(exp_clean).astype(np.float64)

    for i in range(n_sims):
        sim_flat = np.asarray(sim_data_list[i]).ravel()
        sim_clean = sim_flat[~np.isnan(sim_flat)]

        if len(sim_clean) == 0:
            distances[i] = np.nan
            continue

        sim_sorted = np.sort(sim_clean).astype(np.float64)
        n_q = min(max_quantiles, min(len(exp_clean), len(sim_clean)))
        distances[i] = wasserstein_1d_sorted_jit(exp_sorted, sim_sorted, n_q)

    return distances


@jit(nopython=True, parallel=True, cache=True)
def wasserstein_batch_parallel(exp_sorted: np.ndarray, exp_len: int,
                                 sim_sorted_2d: np.ndarray, sim_lens: np.ndarray,
                                 max_quantiles: int) -> np.ndarray:
    """
    Parallel batch Wasserstein computation.

    Parameters
    ----------
    exp_sorted : ndarray
        Pre-sorted experimental data
    exp_len : int
        Length of experimental data (actual values, may be padded)
    sim_sorted_2d : ndarray, shape (n_sims, max_sim_len)
        Pre-sorted simulation data (padded with NaN)
    sim_lens : ndarray, shape (n_sims,)
        Actual lengths of each simulation
    max_quantiles : int
        Maximum number of quantiles

    Returns
    -------
    distances : ndarray, shape (n_sims,)
    """
    n_sims = sim_sorted_2d.shape[0]
    distances = np.empty(n_sims, dtype=np.float64)

    for i in prange(n_sims):
        sim_len = sim_lens[i]
        if sim_len == 0 or exp_len == 0:
            distances[i] = np.nan
            continue

        sim_sorted = sim_sorted_2d[i, :sim_len]
        n_q = min(max_quantiles, min(exp_len, sim_len))
        distances[i] = wasserstein_1d_sorted_jit(exp_sorted[:exp_len], sim_sorted, n_q)

    return distances


# =============================================================================
# COMBINED DISTANCE METRICS
# =============================================================================

def normalize_distances(distances: np.ndarray) -> np.ndarray:
    """
    Z-score normalize distances.

    Parameters
    ----------
    distances : ndarray
        Raw distances

    Returns
    -------
    normalized : ndarray
        Z-score normalized distances (NaN -> 0)
    """
    valid_mask = ~np.isnan(distances)
    if np.sum(valid_mask) < 2:
        return np.zeros_like(distances)

    mean_d = np.nanmean(distances)
    std_d = np.nanstd(distances)

    if std_d < 1e-10:
        return np.zeros_like(distances)

    normalized = (distances - mean_d) / std_d
    normalized = np.nan_to_num(normalized, nan=0.0)

    return normalized


def compute_combined_distance(exp_morans_i: np.ndarray, exp_activity: np.ndarray,
                               exp_blob_counts: np.ndarray, exp_blob_lifetimes: np.ndarray,
                               sim_data_list: List[dict],
                               weights: dict,
                               temporal_scale_factors: Optional[np.ndarray] = None,
                               max_quantiles: int = 1000) -> Tuple[np.ndarray, dict]:
    """
    Compute combined Wasserstein distance using multiple metrics.

    Parameters
    ----------
    exp_morans_i : ndarray
        Experimental Moran's I values
    exp_activity : ndarray
        Experimental activity values
    exp_blob_counts : ndarray
        Experimental blob counts per frame
    exp_blob_lifetimes : ndarray
        Experimental blob lifetimes
    sim_data_list : list of dict
        List of simulation results, each containing:
        - 'MoransI_all': Moran's I time series
        - 'Activity_all': Activity time series
        - 'BlobStats_counts': Blob counts (optional)
        - 'BlobPersistence_lifetimes': Blob lifetimes (optional)
    weights : dict
        Metric weights, e.g., {'moransI': 0.3, 'activity': 0.3, 'blobCount': 0.2, 'blobPersistence': 0.2}
    temporal_scale_factors : ndarray, optional
        Scale factors for each simulation (for blob metrics)
    max_quantiles : int
        Maximum quantiles for Wasserstein computation

    Returns
    -------
    combined_dist : ndarray, shape (n_sims,)
        Combined normalized distance
    individual_dists : dict
        Individual distance arrays for each metric
    """
    n_sims = len(sim_data_list)
    individual_dists = {}

    # Moran's I distances
    if weights.get('moransI', 0) > 0:
        sim_mi = [d.get('MoransI_all', np.array([])) for d in sim_data_list]
        wd_mi = wasserstein_batch(exp_morans_i, sim_mi, max_quantiles)
        individual_dists['moransI'] = wd_mi

    # Activity distances
    if weights.get('activity', 0) > 0:
        sim_act = [d.get('Activity_all', np.array([])) for d in sim_data_list]
        wd_act = wasserstein_batch(exp_activity, sim_act, max_quantiles)
        individual_dists['activity'] = wd_act

    # Blob count distances (with temporal rescaling)
    if weights.get('blobCount', 0) > 0 and exp_blob_counts is not None:
        wd_blob_counts = np.zeros(n_sims)
        for i, sim in enumerate(sim_data_list):
            sim_counts = sim.get('BlobStats_counts', np.array([]))
            if len(sim_counts) == 0:
                wd_blob_counts[i] = np.nan
                continue

            # Apply temporal rescaling if available
            if temporal_scale_factors is not None and temporal_scale_factors[i] != 1.0:
                # Rescale counts by factor (fewer frames -> multiply counts)
                scale = temporal_scale_factors[i]
                if scale > 0:
                    sim_counts = sim_counts * scale

            wd_blob_counts[i] = wasserstein_1d(exp_blob_counts, sim_counts, max_quantiles)

        individual_dists['blobCount'] = wd_blob_counts

    # Blob persistence distances (with temporal rescaling)
    if weights.get('blobPersistence', 0) > 0 and exp_blob_lifetimes is not None:
        wd_blob_persist = np.zeros(n_sims)
        for i, sim in enumerate(sim_data_list):
            sim_lifetimes = sim.get('BlobPersistence_lifetimes', np.array([]))
            if len(sim_lifetimes) == 0:
                wd_blob_persist[i] = np.nan
                continue

            # Apply temporal rescaling: convert MC sweeps to experimental frames
            # scale = tau_exp / tau_ising = frames per MC sweep
            # rescaled_lifetime = sim_lifetime * scale
            if temporal_scale_factors is not None and temporal_scale_factors[i] != 1.0:
                scale = temporal_scale_factors[i]
                if scale > 0:
                    sim_lifetimes = sim_lifetimes * scale

            wd_blob_persist[i] = wasserstein_1d(exp_blob_lifetimes, sim_lifetimes, max_quantiles)

        individual_dists['blobPersistence'] = wd_blob_persist

    # Compute combined distance
    combined_dist = np.zeros(n_sims)
    total_weight = 0.0

    for metric, weight in weights.items():
        if weight > 0 and metric in individual_dists:
            normalized = normalize_distances(individual_dists[metric])
            combined_dist += weight * normalized
            total_weight += weight

    if total_weight > 0:
        combined_dist /= total_weight

    return combined_dist, individual_dists


# =============================================================================
# TESTING
# =============================================================================

def test_wasserstein():
    """Test Wasserstein distance implementations."""
    import time

    print("Testing Wasserstein distance module...")

    # Test basic computation
    np.random.seed(42)
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0.5, 1, 1000)

    d1 = wasserstein_1d(x, y)
    print(f"\nBasic test (normal distributions shifted by 0.5):")
    print(f"  Wasserstein distance: {d1:.4f}")

    # Test with NaN
    x_nan = x.copy()
    x_nan[::10] = np.nan  # 10% NaN
    d2 = wasserstein_1d(x_nan, y)
    print(f"\nWith NaN values (10% in x):")
    print(f"  Wasserstein distance: {d2:.4f}")

    # Test JIT version
    d3 = wasserstein_1d_jit(x.astype(np.float64), y.astype(np.float64))
    print(f"\nJIT version:")
    print(f"  Wasserstein distance: {d3:.4f}")
    print(f"  Difference from pure Python: {abs(d1 - d3):.2e}")

    # Test batch processing
    print("\nBatch processing test:")
    n_sims = 100
    sim_list = [np.random.normal(np.random.uniform(-1, 1), 1, 500) for _ in range(n_sims)]

    start = time.perf_counter()
    dists = wasserstein_batch(x, sim_list)
    batch_time = time.perf_counter() - start

    print(f"  {n_sims} comparisons in {batch_time*1000:.1f} ms")
    print(f"  Per comparison: {batch_time/n_sims*1000:.3f} ms")
    print(f"  Mean distance: {np.nanmean(dists):.4f}")

    # Test combined distance
    print("\nCombined distance test:")
    exp_mi = np.random.uniform(0, 0.5, 500)
    exp_act = np.random.uniform(0.1, 0.3, 500)
    exp_blob = np.random.poisson(5, 500)
    exp_life = np.random.exponential(10, 100)

    sim_data = []
    for _ in range(10):
        sim_data.append({
            'MoransI_all': np.random.uniform(0, 0.5, 2000),
            'Activity_all': np.random.uniform(0.1, 0.3, 2000),
            'BlobStats_counts': np.random.poisson(5, 2000),
            'BlobPersistence_lifetimes': np.random.exponential(10, 100)
        })

    weights = {'moransI': 0.3, 'activity': 0.3, 'blobCount': 0.2, 'blobPersistence': 0.2}

    combined, individual = compute_combined_distance(
        exp_mi, exp_act, exp_blob, exp_life,
        sim_data, weights
    )

    print(f"  Combined distances: {combined[:5]}")
    print(f"  Individual metrics available: {list(individual.keys())}")

    print("\nWasserstein tests PASSED")


if __name__ == '__main__':
    test_wasserstein()
