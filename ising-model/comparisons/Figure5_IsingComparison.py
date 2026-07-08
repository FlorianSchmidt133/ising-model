#!/usr/bin/env python3
"""
Figure 5: Ising Model vs Experimental Data Comparison
======================================================
Python implementation for cluster execution.

This script processes Ising simulations and compares them to experimental
neural data using Moran's I spatial autocorrelation and Wasserstein distance.

Usage:
    python Figure5_IsingComparison.py [--local]

Cluster paths:
    Ising sims: /path/to/data/IsingSims
    Experimental: /path/to/data/ExperimentalData
"""

import os
import sys
import re
import argparse
import time
from glob import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import h5py
from scipy import io as sio
from scipy.spatial.distance import pdist, squareform
# wasserstein_distance removed - using custom wasserstein_1d to match MATLAB
from tqdm import tqdm


# =============================================================================
# HDF5 SAVE HELPER
# =============================================================================

def save_to_hdf5(filename, data_dict):
    """Save nested dict to HDF5 file (MATLAB v7.3 compatible)."""
    def write_item(group, key, value):
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            for k, v in value.items():
                write_item(subgroup, k, v)
        elif isinstance(value, (list, tuple)):
            # Convert list to object array for MATLAB compatibility
            try:
                arr = np.array(value)
                if arr.dtype == object:
                    # Handle list of arrays with different shapes
                    vlen_dt = h5py.special_dtype(vlen=np.float64)
                    ds = group.create_dataset(key, (len(value),), dtype=vlen_dt)
                    for i, item in enumerate(value):
                        if item is not None:
                            ds[i] = np.array(item).flatten()
                else:
                    group.create_dataset(key, data=arr, compression='gzip')
            except Exception:
                # Fallback: save as string
                group.create_dataset(key, data=str(value))
        elif isinstance(value, np.ndarray):
            group.create_dataset(key, data=value, compression='gzip')
        elif isinstance(value, str):
            group.create_dataset(key, data=value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            group.create_dataset(key, data=value)
        elif value is None:
            group.create_dataset(key, data=0)
        else:
            group.create_dataset(key, data=str(value))

    with h5py.File(filename, 'w') as f:
        for key, value in data_dict.items():
            write_item(f, key, value)


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_config(is_local=False):
    """Get configuration based on environment."""
    config = {}

    if is_local:
        # Local paths (Windows)
        config['ising_data_path'] = r'IsingModelData_39x78_100K'
        config['experimental_data_path'] = r'ExperimentalData\ExperimentalData.mat'
        config['output_path'] = r'IsingModelData_39x78_100K\IsingComparison'
    else:
        # Cluster paths
        config['ising_data_path'] = '/path/to/data/IsingSims'
        config['experimental_data_path'] = '/path/to/data/ExperimentalData/ExperimentalData.mat'
        config['output_path'] = '/path/to/data/IsingSims/IsingComparison'

    # Grid mode: 'subselect_centre' | 'subselect_tiled' | 'subselect_centre_vs_tiled'
    config['grid_mode'] = 'subselect_centre_vs_tiled'

    # Matching metric: 'moransI' | 'activity' | 'combined'
    config['matching_metric'] = 'combined'
    config['matching_weights'] = {'moransI': 0.5, 'activity': 0.5}

    # Experimental grid dimensions
    config['experimental_grid'] = (13, 26)  # (rows, cols)

    # Analysis parameters
    config['n_top_matches'] = 10
    config['conditions'] = ['Naive', 'Beginner', 'Expert', 'NoSpout']

    return config


# =============================================================================
# MORAN'S I IMPLEMENTATION
# =============================================================================

def create_weight_matrix(grid_shape, queen=False):
    """
    Create spatial weight matrix for Moran's I calculation.

    Parameters
    ----------
    grid_shape : tuple
        (rows, cols) of the grid
    queen : bool, optional
        If False (default), use Rook adjacency (4-connectivity: up/down/left/right).
        If True, use Queen adjacency (8-connectivity: rook + diagonals).

    Returns
    -------
    weight_mat : ndarray
        Flattened weight matrix of shape (n_cells, n_cells)
    """
    rows, cols = grid_shape
    n_cells = rows * cols

    # Create coordinate grid
    row_coords, col_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    coords = np.column_stack([row_coords.ravel(), col_coords.ravel()])

    # Compute pairwise distances
    dist_mat = squareform(pdist(coords, metric='euclidean'))

    # Select neighbors based on connectivity
    if queen:
        # Queen: all 8 neighbors (distance <= sqrt(2), i.e. rook + diagonals)
        weight_mat = ((dist_mat > 0) & (dist_mat <= np.sqrt(2) + 1e-9)).astype(float)
    else:
        # Rook: 4 nearest neighbors only (distance = 1)
        weight_mat = (dist_mat == 1).astype(float)

    return weight_mat


def morans_i(values, weight_mat):
    """
    Compute Moran's I spatial autocorrelation.
    Matches MATLAB mL_moransI implementation with proper NaN handling.

    Parameters
    ----------
    values : ndarray
        2D grid of values
    weight_mat : ndarray
        Weight matrix (flattened, n_cells x n_cells)

    Returns
    -------
    I : float
        Moran's I statistic
    """
    x = values.ravel().astype(float)

    # Find NaN indices
    nan_mask = np.isnan(x)

    # Number of non-NaN elements
    n = np.sum(~nan_mask)

    # Handle all-NaN case
    if n == 0:
        return np.nan

    # Center using nanmean (matches MATLAB)
    x_mean = np.nanmean(x)
    x_dev = x - x_mean

    # Set NaN positions to 0 in deviations
    x_dev[nan_mask] = 0

    # Zero out weights for NaN positions (copy to avoid modifying original)
    weight_mat = weight_mat.copy()
    weight_mat[nan_mask, :] = 0
    weight_mat[:, nan_mask] = 0

    # Sum of weights
    W = np.sum(weight_mat)

    if W == 0:
        return np.nan

    # Numerator: sum of weighted cross-products
    # Using matrix multiplication for efficiency (matches MATLAB approach)
    weighted_dev_i = weight_mat @ x_dev
    numerator = np.sum(weighted_dev_i * x_dev)

    # Denominator: sum of squared deviations
    denominator = np.sum(x_dev ** 2)

    if denominator == 0:
        return np.nan

    I = (n / W) * (numerator / denominator)

    return I


def wasserstein_1d(x, y, max_quantiles=1000):
    """
    Compute 1D Wasserstein distance using quantile approximation.
    Matches MATLAB wasserstein_1d implementation.

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
    x = np.sort(x)
    y = np.sort(y)

    # Use quantile matching for efficiency with large arrays
    n = min(max_quantiles, min(len(x), len(y)))
    q = np.linspace(0, 1, n)

    x_quantiles = np.quantile(x, q)
    y_quantiles = np.quantile(y, q)

    # Wasserstein distance = mean absolute difference between quantile functions
    d = np.mean(np.abs(x_quantiles - y_quantiles))

    return d


def generate_tiled_positions(ising_grid, exp_grid):
    """
    Generate non-overlapping tile positions.

    Parameters
    ----------
    ising_grid : tuple
        (rows, cols) of Ising grid
    exp_grid : tuple
        (rows, cols) of experimental grid (tile size)

    Returns
    -------
    positions : list of tuples
        List of (row_start, col_start) for each tile
    """
    positions = []
    row_start = 0
    while row_start + exp_grid[0] <= ising_grid[0]:
        col_start = 0
        while col_start + exp_grid[1] <= ising_grid[1]:
            positions.append((row_start, col_start))
            col_start += exp_grid[1]
        row_start += exp_grid[0]
    return positions


def params_to_filename(beta, c, decay_const, rad, bias):
    """
    Generate filename from parameters.

    Format: sim_be_{beta}_c_{c}_d_{decay}_r_{rad}_bi_{bias}.mat
    Example: sim_be_0.5_c_4_d_6_r_9_bi_-0.8.mat
    """
    return f"sim_be_{beta}_c_{int(c)}_d_{int(decay_const)}_r_{int(rad)}_bi_{bias}.mat"


def filename_to_params(filename):
    """
    Parse parameters from new-format filename.

    Returns dict with keys: beta, c, decay_const, inhibition_range, bias
    Returns None if filename doesn't match expected pattern.
    """
    basename = os.path.basename(filename)
    pattern = r'sim_be_([\d.]+)_c_(\d+)_d_(\d+)_r_(\d+)_bi_([-\d.]+)\.mat'
    match = re.match(pattern, basename)
    if match:
        return {
            'beta': float(match.group(1)),
            'c': int(match.group(2)),
            'decay_const': int(match.group(3)),
            'inhibition_range': int(match.group(4)),
            'bias': float(match.group(5))
        }
    return None


def params_to_id_string(beta, c, decay_const, rad, bias):
    """Generate a short identifier string from parameters."""
    return f"be{beta}_c{int(c)}_d{int(decay_const)}_r{int(rad)}_bi{bias}"


# =============================================================================
# DATA PROCESSING
# =============================================================================

def load_experimental_data(config):
    """Load and process experimental data (supports v7.3 HDF5 format)."""
    print(f"Loading experimental data from: {config['experimental_data_path']}")

    morans_i_exp = {}
    binarised_data_exp = {}

    # Use h5py for v7.3 mat files (HDF5 format)
    with h5py.File(config['experimental_data_path'], 'r') as f:
        # Extract Moran's I data
        if 'MoransI' in f:
            mi_group = f['MoransI']
            for condition in config['conditions']:
                if condition in mi_group:
                    # h5py stores data transposed compared to MATLAB
                    morans_i_exp[condition] = np.array(mi_group[condition]).T

        # Extract BinarisedData
        if 'BinarisedData' in f:
            bd_group = f['BinarisedData']
            for condition in config['conditions']:
                if condition in bd_group:
                    binarised_data_exp[condition] = np.array(bd_group[condition]).T

    return morans_i_exp, binarised_data_exp


def process_simulation(sim_path, config, weight_mat_exp, centre_idx, tiled_positions):
    """
    Process a single Ising simulation.

    Returns
    -------
    result : dict or None
        Contains Moran's I time series, activity, and parameters.
        Returns None if processing fails.
    """
    try:
        sim_data = sio.loadmat(sim_path, squeeze_me=True)

        stored_spins = sim_data['stored_spins']  # [T x rows x cols]
        params = sim_data['params']

        n_frames = stored_spins.shape[0]
        exp_grid = config['experimental_grid']

        result = {
            'params': {
                'beta': float(params['beta'].item() if hasattr(params['beta'], 'item') else params['beta']),
                'c': float(params['c'].item() if hasattr(params['c'], 'item') else params['c']),
                'decay_const': float(params['decay_const'].item() if hasattr(params['decay_const'], 'item') else params['decay_const']),
                'inhibition_range': float(params['inhibition_range'].item() if hasattr(params['inhibition_range'], 'item') else params['inhibition_range']),
                'bias': float(params['bias'].item() if hasattr(params['bias'], 'item') else params['bias']),
            }
        }

        grid_mode = config['grid_mode']

        if grid_mode == 'subselect_centre':
            morans_i_ts = np.zeros(n_frames)
            for t in range(n_frames):
                frame = stored_spins[t, centre_idx[0]:centre_idx[1], centre_idx[2]:centre_idx[3]]
                morans_i_ts[t] = morans_i(frame, weight_mat_exp)
            result['MoransI_all'] = morans_i_ts

        elif grid_mode == 'subselect_tiled':
            # Compute for all tiled positions and pool
            all_morans_i = []
            for pos in tiled_positions:
                row_start, col_start = pos
                morans_i_pos = np.zeros(n_frames)
                for t in range(n_frames):
                    frame = stored_spins[t, row_start:row_start+exp_grid[0], col_start:col_start+exp_grid[1]]
                    morans_i_pos[t] = morans_i(frame, weight_mat_exp)
                all_morans_i.append(morans_i_pos)
            result['MoransI_all'] = np.concatenate(all_morans_i)

        elif grid_mode == 'subselect_centre_vs_tiled':
            # Centre crop
            morans_i_centre = np.zeros(n_frames)
            for t in range(n_frames):
                frame = stored_spins[t, centre_idx[0]:centre_idx[1], centre_idx[2]:centre_idx[3]]
                morans_i_centre[t] = morans_i(frame, weight_mat_exp)
            result['MoransI_centre'] = morans_i_centre

            # Tiled (pooled)
            all_morans_i_tiled = []
            for pos in tiled_positions:
                row_start, col_start = pos
                morans_i_pos = np.zeros(n_frames)
                for t in range(n_frames):
                    frame = stored_spins[t, row_start:row_start+exp_grid[0], col_start:col_start+exp_grid[1]]
                    morans_i_pos[t] = morans_i(frame, weight_mat_exp)
                all_morans_i_tiled.append(morans_i_pos)
            result['MoransI_tiled'] = np.concatenate(all_morans_i_tiled)

            # Use centre as main (for compatibility)
            result['MoransI_all'] = morans_i_centre

        # Convert -1/+1 spins to 0/1 binary, then compute activity (fraction active per frame)
        binary_spins = (stored_spins + 1) / 2  # Maps -1→0, +1→1
        activity_ts = np.mean(binary_spins, axis=(1, 2))
        result['Activity_all'] = activity_ts
        result['Activity_mean'] = np.nanmean(activity_ts)
        result['Activity_std'] = np.nanstd(activity_ts)
        result['MoransI_mean'] = np.nanmean(result['MoransI_all'])
        result['MoransI_std'] = np.nanstd(result['MoransI_all'])

        return result

    except Exception as e:
        print(f"Error processing {sim_path}: {e}")
        return None


def compute_wasserstein_distances(exp_morans_i, ising_data, config):
    """
    Compute Wasserstein distances between experimental and Ising distributions.
    """
    n_sims = len(ising_data)
    distances = np.zeros(n_sims)

    exp_values = exp_morans_i.ravel()
    exp_values = exp_values[~np.isnan(exp_values)]

    for i, sim_result in enumerate(ising_data):
        ising_values = sim_result['MoransI_all']
        ising_values = ising_values[~np.isnan(ising_values)]

        if len(exp_values) > 0 and len(ising_values) > 0:
            distances[i] = wasserstein_1d(exp_values, ising_values)
        else:
            distances[i] = np.nan

    return distances


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ising Model Comparison')
    parser.add_argument('--local', action='store_true', help='Run with local paths')
    args = parser.parse_args()

    print("=" * 70)
    print("  Figure 5: Ising Model vs Experimental Data Comparison")
    print("  Python Cluster Implementation")
    print("=" * 70)

    # Get configuration
    config = get_config(is_local=args.local)
    print(f"\nEnvironment: {'LOCAL' if args.local else 'CLUSTER'}")
    print(f"Grid mode: {config['grid_mode']}")
    print(f"Matching metric: {config['matching_metric']}")

    # Create output directory
    os.makedirs(config['output_path'], exist_ok=True)
    print(f"Output path: {config['output_path']}")

    # Find simulation files (new naming format: sim_be_*.mat)
    sim_pattern = os.path.join(config['ising_data_path'], 'sim_be_*.mat')
    sim_files = sorted(glob(sim_pattern))
    n_sims = len(sim_files)
    print(f"\nFound {n_sims} simulation files")

    if n_sims == 0:
        print("ERROR: No simulation files found!")
        sys.exit(1)

    # Auto-detect grid size from first simulation
    first_sim = sio.loadmat(sim_files[0], squeeze_me=True)
    stored_spins = first_sim['stored_spins']
    ising_grid = (stored_spins.shape[1], stored_spins.shape[2])
    config['ising_grid'] = ising_grid
    print(f"Auto-detected Ising grid: {ising_grid[0]} x {ising_grid[1]}")

    # Create weight matrix for experimental grid
    weight_mat_exp = create_weight_matrix(config['experimental_grid'])
    print(f"Created weight matrix for {config['experimental_grid'][0]}x{config['experimental_grid'][1]} grid")

    # Compute centre crop indices
    exp_grid = config['experimental_grid']
    centre_row_start = (ising_grid[0] - exp_grid[0]) // 2
    centre_col_start = (ising_grid[1] - exp_grid[1]) // 2
    centre_idx = (centre_row_start, centre_row_start + exp_grid[0],
                  centre_col_start, centre_col_start + exp_grid[1])
    print(f"Centre crop: rows {centre_idx[0]}:{centre_idx[1]}, cols {centre_idx[2]}:{centre_idx[3]}")

    # Generate tiled positions
    tiled_positions = generate_tiled_positions(ising_grid, exp_grid)
    print(f"Tiled positions: {len(tiled_positions)} non-overlapping grids")

    # Load experimental data
    morans_i_exp, binarised_data_exp = load_experimental_data(config)
    print(f"Loaded experimental data for conditions: {list(morans_i_exp.keys())}")

    # Process all simulations in parallel
    print(f"\nProcessing {n_sims} simulations...")
    start_time = time.time()

    # Determine number of workers
    n_workers = min(cpu_count(), 16)
    print(f"Using {n_workers} parallel workers")

    # Create partial function with fixed arguments
    process_func = partial(process_simulation,
                           config=config,
                           weight_mat_exp=weight_mat_exp,
                           centre_idx=centre_idx,
                           tiled_positions=tiled_positions)

    # Extract simulation IDs from filenames (parameter-based naming)
    sim_ids = []
    for p in sim_files:
        params = filename_to_params(p)
        if params:
            sim_ids.append(params_to_id_string(
                params['beta'], params['c'], params['decay_const'],
                params['inhibition_range'], params['bias']))
        else:
            sim_ids.append(os.path.basename(p).replace('.mat', ''))

    # Process simulations in parallel
    with Pool(n_workers) as pool:
        ising_data = list(tqdm(
            pool.imap(process_func, sim_files),
            total=len(sim_files),
            desc="Processing simulations"
        ))

    elapsed = time.time() - start_time
    print(f"Processing complete: {elapsed:.1f}s ({elapsed/n_sims:.2f}s per simulation)")

    # Remove failed simulations
    valid_mask = [d is not None for d in ising_data]
    ising_data = [d for d in ising_data if d is not None]
    sim_ids = [s for s, v in zip(sim_ids, valid_mask) if v]
    print(f"Valid simulations: {len(ising_data)}/{n_sims}")

    # Compute statistics for experimental data
    print("\nComputing experimental statistics...")
    exp_stats = {}
    for condition in config['conditions']:
        if condition in morans_i_exp:
            mi = morans_i_exp[condition]
            exp_stats[condition] = {
                'MoransI_all': mi.ravel(),
                'MoransI_mean': np.nanmean(mi),
                'MoransI_std': np.nanstd(mi),
            }

            # Compute Activity statistics from BinarisedData (matches MATLAB)
            if condition in binarised_data_exp and binarised_data_exp[condition] is not None:
                bin_data = binarised_data_exp[condition]
                # bin_data shape: [rows, cols, time, trials] after h5py transpose
                # Compute mean activity per frame (mean across spatial dimensions)
                activity = np.mean(bin_data, axis=(0, 1))  # [time, trials]
                exp_stats[condition]['Activity_all'] = activity.ravel()  # Flatten
                exp_stats[condition]['Activity_mean'] = np.nanmean(exp_stats[condition]['Activity_all'])
                exp_stats[condition]['Activity_std'] = np.nanstd(exp_stats[condition]['Activity_all'])

            print(f"  {condition}: mean={exp_stats[condition]['MoransI_mean']:.4f}, "
                  f"std={exp_stats[condition]['MoransI_std']:.4f}")

    # Compute Wasserstein distances for each condition
    print("\nComputing Wasserstein distances...")
    comparison = {}

    for condition in config['conditions']:
        if condition not in exp_stats:
            continue

        print(f"  {condition}...")
        exp_mi = exp_stats[condition]['MoransI_all']

        # Compute WD based on matching metric
        if config['matching_metric'] == 'moransI':
            wd = compute_wasserstein_distances(exp_mi, ising_data, config)
        elif config['matching_metric'] == 'combined':
            # Check if Activity data exists (matching MATLAB fallback behavior)
            if 'Activity_all' not in exp_stats[condition]:
                print(f"    Warning: No Activity data for {condition}. Using moransI only.")
                wd = compute_wasserstein_distances(exp_mi, ising_data, config)
            else:
                # Moran's I distances
                wd_mi = compute_wasserstein_distances(exp_mi, ising_data, config)
                wd_mi_norm = (wd_mi - np.nanmean(wd_mi)) / np.nanstd(wd_mi)
                wd_mi_norm = np.nan_to_num(wd_mi_norm, nan=0.0)

                # Activity distances
                exp_activity = exp_stats[condition]['Activity_all']
                exp_activity_clean = exp_activity.ravel()
                exp_activity_clean = exp_activity_clean[~np.isnan(exp_activity_clean)]
                wd_activity = np.zeros(len(ising_data))
                for s, sim in enumerate(ising_data):
                    sim_activity = sim['Activity_all'].ravel()
                    wd_activity[s] = wasserstein_1d(exp_activity_clean, sim_activity)
                wd_activity_norm = (wd_activity - np.nanmean(wd_activity)) / np.nanstd(wd_activity)
                wd_activity_norm = np.nan_to_num(wd_activity_norm, nan=0.0)

                # Weighted combination (matching MATLAB implementation)
                wd = (config['matching_weights']['moransI'] * wd_mi_norm +
                      config['matching_weights']['activity'] * wd_activity_norm)
        else:
            wd = compute_wasserstein_distances(exp_mi, ising_data, config)

        # Rank simulations
        rankings = np.argsort(wd)
        best_idx = rankings[:config['n_top_matches']]

        comparison[condition] = {
            'wasserstein_dist': wd,
            'rankings': rankings,
            'bestMatch_idx': best_idx,
            'bestMatch_simIDs': [sim_ids[i] for i in best_idx],
            'bestMatch_params': {
                'beta': [ising_data[i]['params']['beta'] for i in best_idx],
                'c': [ising_data[i]['params']['c'] for i in best_idx],
                'decay_const': [ising_data[i]['params']['decay_const'] for i in best_idx],
                'inhibition_range': [ising_data[i]['params']['inhibition_range'] for i in best_idx],
                'bias': [ising_data[i]['params']['bias'] for i in best_idx],
            }
        }

        # Print top matches
        print(f"    Top 5 matches:")
        for j in range(min(5, len(best_idx))):
            idx = best_idx[j]
            print(f"      {sim_ids[idx]}: WD={wd[idx]:.4f}")

    # Prepare results for saving
    print("\nPreparing results for saving...")

    # Convert ising_data to format suitable for MATLAB
    IsingData = {
        'simIDs': np.array(sim_ids),
        'MoransI_mean': np.array([d['MoransI_mean'] for d in ising_data]),
        'MoransI_std': np.array([d['MoransI_std'] for d in ising_data]),
        'Activity_mean': np.array([d['Activity_mean'] for d in ising_data]),
        'Activity_std': np.array([d['Activity_std'] for d in ising_data]),
        'MoransI_all': [d['MoransI_all'] for d in ising_data],
        'Activity_all': [d['Activity_all'] for d in ising_data],
        'params': {
            'beta': np.array([d['params']['beta'] for d in ising_data]),
            'c': np.array([d['params']['c'] for d in ising_data]),
            'decay_const': np.array([d['params']['decay_const'] for d in ising_data]),
            'inhibition_range': np.array([d['params']['inhibition_range'] for d in ising_data]),
            'bias': np.array([d['params']['bias'] for d in ising_data]),
        }
    }

    # Add centre/tiled specific data if applicable
    if config['grid_mode'] == 'subselect_centre_vs_tiled':
        IsingData['MoransI_centre'] = [d.get('MoransI_centre', d['MoransI_all']) for d in ising_data]
        IsingData['MoransI_tiled'] = [d.get('MoransI_tiled', d['MoransI_all']) for d in ising_data]

    Results = {
        'config': config,
        'IsingData': IsingData,
        'ExpStats': exp_stats,
        'Comparison': comparison,
        'timestamp': datetime.now().isoformat(),
    }

    # Save results
    output_file = os.path.join(config['output_path'], f"IsingComparison_Results_{config['grid_mode']}_{config['matching_metric']}.mat")
    print(f"\nSaving results to: {output_file}")
    save_to_hdf5(output_file, Results)

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
