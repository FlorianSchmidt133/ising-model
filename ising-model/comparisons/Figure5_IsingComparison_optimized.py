#!/usr/bin/env python3
"""
Figure 5: Ising Model vs Experimental Data Comparison (Optimized)
==================================================================

Numba-optimized Python implementation for cluster execution.
Achieves 50-100x speedup over pure NumPy/MATLAB implementation.

This script processes Ising simulations and compares them to experimental
neural data using multiple metrics:
- Moran's I spatial autocorrelation
- Activity (sparsity) distributions
- Blob count distributions (with temporal rescaling)
- Blob persistence/lifetime distributions

Usage:
    python Figure5_IsingComparison_optimized.py [--local] [--metric METRIC] [--n-workers N]

Options:
    --local           Use local Windows paths instead of cluster paths
    --metric          Matching metric: moransI, activity, spatial+persistence, combined
    --n-workers       Number of parallel workers (default: auto-detect)
    --checkpoint      Resume from checkpoint file
    --max-sims        Maximum simulations to process (for testing)
    --visualize       Generate visualization figures (default: True)
    --no-visualize    Skip visualization generation

Cluster paths:
    Ising sims: /path/to/data/IsingSims
    Experimental: /path/to/data/ExperimentalData
"""

import os
import sys

# Must set NUMBA_NUM_THREADS before importing numba modules
def _set_numba_threads():
    """Parse --numba-threads from command line and set environment variable.

    Handles both ``--numba-threads N`` and ``--numba-threads=N`` formats.
    """
    for i, arg in enumerate(sys.argv):
        if arg == '--numba-threads' and i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            if val != '0':  # 0 means auto (don't set)
                os.environ['NUMBA_NUM_THREADS'] = val
            return
        if arg.startswith('--numba-threads='):
            val = arg.split('=', 1)[1]
            if val != '0':
                os.environ['NUMBA_NUM_THREADS'] = val
            return
_set_numba_threads()

import copy
import re
import argparse
import time
import traceback
import pickle
from glob import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import h5py
from scipy import io as sio
from tqdm import tqdm

# Import optimized modules
from morans_i_optimized import (
    create_weight_matrix_jit,
    morans_i_batch,
    morans_i_centre_crop,
    morans_i_tiled_batch,
    generate_tiled_positions,
    compute_centre_indices,
    warmup_jit
)
from blob_detection_optimized import (
    compute_blob_persistence_batch,
    detect_blobs_all_frames,
    track_persistence_numba,
    detect_blobs_tiled,
    compute_autocorr_decay,
    compute_temporal_scale_factor,
    compute_trial_averaged_autocorr,
    compute_global_autocorr,
    BlobStats
)
from wasserstein_optimized import (
    wasserstein_1d,
    wasserstein_batch,
    compute_combined_distance,
    normalize_distances
)

import hashlib
import json

# Visualization module (optional - imported at runtime if needed)
HAS_VISUALIZATION = False
try:
    from ising_visualizations import IsingVisualizer, plot_comparison_results
    HAS_VISUALIZATION = True
except ImportError:
    pass


# =============================================================================
# HDF5 SAVE HELPER
# =============================================================================

def save_to_hdf5(filename: str, data_dict: dict):
    """Save nested dict to HDF5 file (MATLAB v7.3 compatible)."""
    def write_item(group, key, value):
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            for k, v in value.items():
                write_item(subgroup, k, v)
        elif isinstance(value, (list, tuple)):
            try:
                try:
                    arr = np.array(value)
                except (ValueError, TypeError):
                    # Ragged sequences (arrays of different lengths) raise
                    # ValueError in NumPy >= 2.0; create object array explicitly
                    arr = np.empty(len(value), dtype=object)
                    arr[:] = value
                if arr.dtype == object:
                    vlen_dt = h5py.special_dtype(vlen=np.float64)
                    ds = group.create_dataset(key, (len(value),), dtype=vlen_dt)
                    for i, item in enumerate(value):
                        if item is not None:
                            ds[i] = np.array(item).flatten()
                elif arr.dtype.kind in ('U', 'S'):
                    # Variable-length UTF-8 strings (h5py 3.x rejects raw
                    # NumPy <U... arrays without an explicit string dtype,
                    # which previously caused the whole list to fall through
                    # to the str(value) repr fallback)
                    group.create_dataset(
                        key, data=arr.astype(object),
                        dtype=h5py.string_dtype(encoding='utf-8'))
                else:
                    group.create_dataset(key, data=arr, compression='gzip')
            except Exception:
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

    with h5py.File(filename, 'w', libver='latest') as f:
        for key, value in data_dict.items():
            write_item(f, key, value)


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_config(is_local: bool = False, metric: str = 'combined',
               connectivity: str = 'rook', refractory_K: int = 0) -> dict:
    """Get configuration based on environment.

    When refractory_K > 0, all paths route to `IsingSims[_queen]_refractoryK{N}`
    instead of the K=0 root.
    """
    config = {}

    if is_local:
        base = r'IsingModelData_39x78_100K'
        if connectivity == 'queen':
            base = r'IsingModelData_39x78_100K_queen'
        if refractory_K > 0:
            base = f"{base}_refractoryK{int(refractory_K)}"
        config['ising_data_path'] = base
        config['experimental_data_path'] = r'Paper\Fig. 5 Model\IsingModels\Data\ExperimentalData.mat'
        config['output_path'] = os.path.join(base, 'IsingComparison', metric)
        config['cache_path'] = os.path.join(base, 'cache')
    else:
        base = '/path/to/data/IsingSims'
        if connectivity == 'queen':
            base = '/path/to/data/IsingSims_queen'
        if refractory_K > 0:
            base = f"{base}_refractoryK{int(refractory_K)}"
        config['ising_data_path'] = base
        config['experimental_data_path'] = '/path/to/data/ExperimentalData/ExperimentalData.mat'
        config['output_path'] = os.path.join(base, 'IsingComparison', metric)
        config['cache_path'] = os.path.join(base, 'cache')

    # Grid mode: 'subselect_centre' | 'subselect_tiled' | 'subselect_centre_vs_tiled'
    config['grid_mode'] = 'subselect_centre_vs_tiled'

    # Matching metric options:
    # 'moransI'           - Moran's I only
    # 'activity'          - Activity only
    # 'autocorr'          - Autocorrelation tau only (log-ratio distance)
    # 'blobCount'         - Blob count distribution only
    # 'blobPersistence'   - Blob persistence/lifetime distribution only
    # 'moransI+activity'  - Moran's I and activity combined
    # 'spatial+persistence' - Moran's I and blob persistence
    # 'combined'          - Full combined metric (all four)
    config['matching_metric'] = metric

    # Weights for combined metric (used when metric='combined' or custom)
    config['matching_weights'] = {
        'moransI': 0.25,
        'activity': 0.25,
        'blobCount': 0.25,
        'blobPersistence': 0.25
    }

    # Experimental grid dimensions
    config['experimental_grid'] = (13, 26)

    # Blob detection parameters (match MATLAB)
    config['blob_params'] = {
        'sigma': 1.0,
        'threshold': 0.5,
        'min_size': 10,
        'iou_threshold': 0.3
    }

    # Autocorrelation parameters
    # NOTE: max_lag must be < exp_prestim_frames (80) when using prestim selection
    # NOTE: fit_range (1, 10) matches MATLAB - short-time fit for single timescale
    config['autocorr'] = {
        'max_lag': 50,
        'fit_range': (1, 10)
    }

    # Timestep limiting (matching MATLAB config.limitTimesteps)
    config['limit_timesteps'] = True      # Set to False for full 100K frame analysis
    config['max_timesteps'] = 10000       # Max timesteps when limit_timesteps is True

    # Experimental data frame selection
    # Options: 'all' (frames 1-185), 'prestim' (frames 1-80), or 'nostim' (1-80 + 101-180, doubles trials)
    config['exp_frame_selection'] = 'prestim'
    config['exp_prestim_frames'] = 80     # Number of pre-stimulus frames
    config['exp_stim_onset_frame'] = 81   # First stimulus frame (1-indexed, MATLAB convention)
    config['exp_stim_offset_frame'] = 100 # Last stimulus frame (1-indexed, MATLAB convention)
    config['exp_full_trial_frames'] = 185
    config['blob_trial_chunks'] = [80, 185]
    config['blob_min_lifetime'] = 2  # Exclude transient blobs (lifetime=1) from comparison

    # Analysis parameters
    config['n_top_matches'] = 10
    config['conditions'] = ['Naive', 'Beginner', 'Expert', 'Expert_Hit', 'Expert_Miss', 'NoSpout']
    config['max_quantiles'] = 1000

    # Cache configuration
    config['use_cache'] = True

    return config


def compute_config_hash(config: dict) -> str:
    """Compute hash of configuration for cache validation.

    Note: autocorr parameters are excluded because they only affect
    experimental data analysis (trial-averaged ACF), not Ising simulation
    processing. Including them would invalidate the cache unnecessarily
    when adjusting max_lag or fit_range.
    """
    # Only include parameters that affect Ising simulation processing
    cache_relevant = {
        'grid_mode': config['grid_mode'],
        'experimental_grid': config['experimental_grid'],
        'blob_params': config['blob_params'],
        'blob_trial_chunks': config.get('blob_trial_chunks', []),
        # autocorr excluded - only affects experimental stats, not Ising processing
    }
    config_str = json.dumps(cache_relevant, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


# =============================================================================
# FILE PARSING UTILITIES
# =============================================================================

def filename_to_params(filename: str) -> Optional[dict]:
    """Parse parameters from simulation filename.

    Accepts the optional `_queen` and `_rK{N}` markers before the extension
    (e.g. `sim_be_0.5_c_4_d_6_r_9_bi_-0.8_rK10.mat`).
    """
    basename = os.path.basename(filename)
    pattern = (r'sim_be_([\d.]+)_c_(\d+)_d_(\d+)_r_(\d+)_bi_([-\d.]+?)'
               r'(?:_queen)?(?:_rK\d+)?\.(?:mat|npz)')
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


def params_to_id_string(beta, c, decay_const, rad, bias) -> str:
    """Generate short identifier string from parameters."""
    return f"be{beta}_c{int(c)}_d{int(decay_const)}_r{int(rad)}_bi{bias}"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_experimental_data(config: dict) -> Tuple[dict, dict, dict]:
    """Load and process experimental data (supports v7.3 HDF5 format).

    Returns
    -------
    morans_i_exp : dict
        Moran's I data per condition
    binarised_data_exp : dict
        Binarised grid data per condition
    recording_metadata : dict
        Per-recording metadata (nTrials_per_recording, recording_indices,
        animal_names) per condition. Empty dict if not present in file.
    """
    print(f"Loading experimental data from: {config['experimental_data_path']}")

    morans_i_exp = {}
    binarised_data_exp = {}
    recording_metadata = {}

    with h5py.File(config['experimental_data_path'], 'r') as f:
        if 'MoransI' in f:
            mi_group = f['MoransI']
            for condition in config['conditions']:
                if condition in mi_group:
                    morans_i_exp[condition] = np.array(mi_group[condition]).T

        if 'BinarisedData' in f:
            bd_group = f['BinarisedData']
            for condition in config['conditions']:
                if condition in bd_group:
                    # Shape: [rows, cols, time, trials] - DO NOT transpose
                    # np.mean(axis=(0,1)) will average spatial dims → [time, trials]
                    binarised_data_exp[condition] = np.array(bd_group[condition])

        # Load RecordingMetadata (backward compatible - may not exist)
        if 'RecordingMetadata' in f:
            rm_group = f['RecordingMetadata']
            for condition in config['conditions']:
                if condition in rm_group:
                    cond_group = rm_group[condition]
                    meta = {}
                    if 'nTrials_per_recording' in cond_group:
                        meta['nTrials_per_recording'] = np.array(
                            cond_group['nTrials_per_recording']).ravel().astype(int)
                    if 'recording_indices' in cond_group:
                        meta['recording_indices'] = np.array(
                            cond_group['recording_indices']).ravel().astype(int)
                    if 'animal_names' in cond_group:
                        raw = cond_group['animal_names'][()]
                        names = []
                        for ref in raw.flatten():
                            if isinstance(ref, h5py.h5r.Reference):
                                # Dereference and decode uint16 char array
                                deref = f[ref]
                                char_array = deref[()]
                                if char_array.dtype == np.uint16:
                                    name = ''.join(chr(c) for c in char_array.flatten())
                                    names.append(name)
                                else:
                                    names.append(f'Unknown_{len(names)}')
                            elif isinstance(ref, bytes):
                                names.append(ref.decode('utf-8'))
                            else:
                                names.append(str(ref))
                        meta['animal_names'] = names

                    # Validate trial counts match
                    if 'nTrials_per_recording' in meta and condition in morans_i_exp:
                        expected = morans_i_exp[condition].shape[0]
                        actual = int(np.sum(meta['nTrials_per_recording']))
                        if expected != actual:
                            print(f"  WARNING: {condition} trial count mismatch: "
                                  f"MoransI has {expected} trials, metadata sums to {actual}")

                    recording_metadata[condition] = meta
            print(f"  Loaded RecordingMetadata for: {list(recording_metadata.keys())}")
        else:
            print("  RecordingMetadata not found in file (backward compatible - skipping)")

    return morans_i_exp, binarised_data_exp, recording_metadata


def load_simulation_file(sim_path: str) -> Optional[Tuple[np.ndarray, dict]]:
    """Load a single simulation file."""
    try:
        sim_data = sio.loadmat(sim_path, squeeze_me=True)
        stored_spins = sim_data['stored_spins']  # [T x rows x cols]
        params = sim_data['params']

        params_dict = {
            'beta': float(params['beta'].item() if hasattr(params['beta'], 'item') else params['beta']),
            'c': float(params['c'].item() if hasattr(params['c'], 'item') else params['c']),
            'decay_const': float(params['decay_const'].item() if hasattr(params['decay_const'], 'item') else params['decay_const']),
            'inhibition_range': float(params['inhibition_range'].item() if hasattr(params['inhibition_range'], 'item') else params['inhibition_range']),
            'bias': float(params['bias'].item() if hasattr(params['bias'], 'item') else params['bias']),
        }

        return stored_spins, params_dict
    except Exception as e:
        print(f"Error loading {sim_path}: {e}")
        return None


# =============================================================================
# OPTIMIZED SIMULATION PROCESSING
# =============================================================================

def process_simulation_optimized(sim_path: str, config: dict,
                                   weight_mat: np.ndarray,
                                   centre_idx: Tuple[int, int, int, int],
                                   tile_row_starts: np.ndarray,
                                   tile_col_starts: np.ndarray) -> Optional[dict]:
    """
    Process a single Ising simulation using optimized Numba functions.

    Returns dictionary with all computed statistics.
    """
    # Load simulation
    result = load_simulation_file(sim_path)
    if result is None:
        return None

    stored_spins, params_dict = result
    del result  # Drop tuple reference so full array can be freed
    n_frames_original = stored_spins.shape[0]
    exp_grid = config['experimental_grid']

    # Apply timestep limiting (matching MATLAB config.limitTimesteps)
    if config.get('limit_timesteps', False) and n_frames_original > config.get('max_timesteps', 5000):
        max_ts = config['max_timesteps']
        stored_spins = stored_spins[:max_ts, :, :].copy()  # Copy! View keeps full 100K array alive
        n_frames = max_ts
    else:
        n_frames = n_frames_original

    output = {'params': params_dict, 'n_frames_used': n_frames, 'n_frames_original': n_frames_original}

    # Convert to float64 for Numba (contiguous array)
    spins_f64 = np.ascontiguousarray(stored_spins.astype(np.float64))

    grid_mode = config['grid_mode']

    # ==========================================================================
    # MORAN'S I COMPUTATION (OPTIMIZED)
    # ==========================================================================

    if grid_mode == 'subselect_centre':
        # Centre crop only
        morans_i_ts = morans_i_centre_crop(
            spins_f64, weight_mat,
            centre_idx[0], centre_idx[1],
            centre_idx[2], centre_idx[3]
        )
        output['MoransI_all'] = morans_i_ts

    elif grid_mode == 'subselect_tiled':
        # Tiled positions - pooled
        morans_i_ts = morans_i_tiled_batch(
            spins_f64, weight_mat,
            tile_row_starts, tile_col_starts,
            exp_grid[0], exp_grid[1]
        )
        output['MoransI_all'] = morans_i_ts

    elif grid_mode == 'subselect_centre_vs_tiled':
        # Both centre and tiled
        morans_i_centre = morans_i_centre_crop(
            spins_f64, weight_mat,
            centre_idx[0], centre_idx[1],
            centre_idx[2], centre_idx[3]
        )
        output['MoransI_centre'] = morans_i_centre

        morans_i_tiled = morans_i_tiled_batch(
            spins_f64, weight_mat,
            tile_row_starts, tile_col_starts,
            exp_grid[0], exp_grid[1]
        )
        output['MoransI_tiled'] = morans_i_tiled

        # Use centre as main for compatibility
        output['MoransI_all'] = morans_i_centre

    # ==========================================================================
    # ACTIVITY COMPUTATION (full Ising grid)
    # ==========================================================================

    binary_spins = (stored_spins + 1) / 2
    activity_ts = np.mean(binary_spins, axis=(1, 2))
    output['Activity_all'] = activity_ts
    output['Activity_mean'] = np.nanmean(activity_ts)
    output['Activity_std'] = np.nanstd(activity_ts)

    # ==========================================================================
    # BLOB DETECTION AND PERSISTENCE
    # ==========================================================================

    blob_params = config['blob_params']

    # Centre crop for blob detection (blob detection uses experimental-sized grid)
    centre_crop = stored_spins[:, centre_idx[0]:centre_idx[1],
                               centre_idx[2]:centre_idx[3]]

    # Run blob DETECTION once on all frames (the expensive scipy part)
    all_labels, all_n_blobs, blob_counts, blob_sizes = detect_blobs_all_frames(
        centre_crop, sigma=blob_params['sigma'],
        threshold=blob_params['threshold'],
        min_size=blob_params['min_size'])

    output['BlobStats_counts'] = blob_counts
    output['BlobStats_sizes'] = blob_sizes

    # Continuous persistence TRACKING (cheap numba part, reuses labels)
    continuous_lifetimes = track_persistence_numba(
        all_labels, all_n_blobs, blob_params['iou_threshold'])
    output['BlobPersistence_lifetimes'] = continuous_lifetimes

    # Trial-split persistence tracking (reuses pre-computed labels — no re-detection)
    for chunk_size in config.get('blob_trial_chunks', []):
        n_chunks = n_frames // chunk_size
        if n_chunks == 0:
            output[f'BlobPersistence_lifetimes_chunk{chunk_size}'] = continuous_lifetimes
            continue
        all_chunk_lifetimes = []
        for c_i in range(n_chunks):
            s = c_i * chunk_size
            e = s + chunk_size
            chunk_lifetimes = track_persistence_numba(
                all_labels[s:e], all_n_blobs[s:e], blob_params['iou_threshold'])
            if len(chunk_lifetimes) > 0:
                all_chunk_lifetimes.extend(chunk_lifetimes)
        output[f'BlobPersistence_lifetimes_chunk{chunk_size}'] = (
            np.array(all_chunk_lifetimes, dtype=np.int32) if all_chunk_lifetimes
            else np.array([], dtype=np.int32)
        )

    # ==========================================================================
    # AUTOCORRELATION DECAY
    # ==========================================================================

    tau, acf, r2 = compute_autocorr_decay(
        activity_ts,
        max_lag=config['autocorr']['max_lag'],
        fit_range=config['autocorr']['fit_range']
    )
    output['Autocorr_tau'] = tau
    output['Autocorr_acf'] = acf
    output['Autocorr_r2'] = r2

    # ==========================================================================
    # SUMMARY STATISTICS
    # ==========================================================================

    output['MoransI_mean'] = np.nanmean(output['MoransI_all'])
    output['MoransI_std'] = np.nanstd(output['MoransI_all'])

    return output


def process_simulation_wrapper(args):
    """Wrapper for multiprocessing (unpacks arguments)."""
    sim_path, config, weight_mat, centre_idx, tile_row_starts, tile_col_starts = args
    return process_simulation_optimized(
        sim_path, config, weight_mat, centre_idx, tile_row_starts, tile_col_starts
    )


# =============================================================================
# EXPERIMENTAL STATISTICS
# =============================================================================

def apply_nostim_selection(data: np.ndarray, config: dict, axis_time: int = 1) -> np.ndarray:
    """Split data into pre-stim and post-stim sub-trials, concatenate along trial axis.

    Pre-stim:  frames 0 : prestim_frames           (0-indexed: 0-79)
    Post-stim: frames stim_offset : stim_offset+prestim_frames  (0-indexed: 100-179)

    This doubles the trial count and produces 80-frame sub-trials.

    Parameters
    ----------
    data : np.ndarray
        Array with trials on axis 0 and time on *axis_time*.
    config : dict
        Must contain 'exp_prestim_frames' and 'exp_stim_offset_frame'.
    axis_time : int
        Which axis is the time dimension (default 1).
    """
    prestim = config['exp_prestim_frames']              # 80
    stim_off = config['exp_stim_offset_frame']          # 100 (1-indexed)
    post_start = stim_off                               # 0-indexed start of post-stim
    post_end = post_start + prestim                     # 0-indexed exclusive end

    # Build slicing tuples for pre and post segments
    pre_slices = [slice(None)] * data.ndim
    pre_slices[axis_time] = slice(0, prestim)

    post_slices = [slice(None)] * data.ndim
    post_slices[axis_time] = slice(post_start, post_end)

    pre = data[tuple(pre_slices)]
    post = data[tuple(post_slices)]

    # Concatenate along the trial axis (axis 0)
    return np.concatenate([pre, post], axis=0)


def compute_experimental_statistics(morans_i_exp: dict, binarised_data_exp: dict,
                                      config: dict, recording_metadata: dict = None) -> dict:
    """
    Compute comprehensive statistics from experimental data.

    Includes per-condition:
    - Moran's I distribution and statistics
    - Activity distribution and statistics
    - Activity_trials matrix for trial-level analysis
    - Blob count distribution
    - Blob persistence/lifetime distribution
    - Trial-averaged autocorrelation (MATLAB-compatible)
    - Histograms for visualization
    """
    exp_stats = {}
    blob_params = config['blob_params']
    autocorr_params = config['autocorr']

    # Frame selection: 'all', 'prestim', or 'nostim'
    frame_selection = config.get('exp_frame_selection', 'all')
    prestim_frames = config.get('exp_prestim_frames', 80)
    stim_offset = config.get('exp_stim_offset_frame', 100)
    if frame_selection == 'prestim':
        print(f"  Using pre-stimulus frames only (1-{prestim_frames})")
    elif frame_selection == 'nostim':
        print(f"  Using nostim mode: pre-stim (1-{prestim_frames}) + post-stim ({stim_offset+1}-{stim_offset+prestim_frames}), doubles trials")
    else:
        print(f"  Using all frames")

    # Auto-reduce autocorrelation max_lag for short sub-trials
    if frame_selection in ('prestim', 'nostim'):
        max_allowed = prestim_frames // 2 - 1  # = 39
        if autocorr_params['max_lag'] > max_allowed:
            print(f"  Auto-reducing autocorr max_lag from {autocorr_params['max_lag']} to {max_allowed}")
            autocorr_params = {**autocorr_params, 'max_lag': max_allowed}
            if autocorr_params['fit_range'][1] > max_allowed:
                autocorr_params['fit_range'] = (autocorr_params['fit_range'][0], max_allowed)

    for condition in config['conditions']:
        if condition not in morans_i_exp:
            continue

        mi = morans_i_exp[condition]
        # Apply frame selection to MoransI: shape is [trials, time]
        if frame_selection == 'prestim':
            if mi.shape[1] > prestim_frames:
                mi = mi[:, :prestim_frames]
        elif frame_selection == 'nostim':
            mi = apply_nostim_selection(mi, config, axis_time=1)

        stats = {
            'MoransI_all': mi.ravel(),
            'MoransI_trials': mi,  # Keep [trials, time] matrix for timecourse plotting
            'MoransI_mean': np.nanmean(mi),
            'MoransI_std': np.nanstd(mi),
        }

        # Compute Moran's I histogram for visualization
        mi_valid = mi.ravel()[~np.isnan(mi.ravel())]
        if len(mi_valid) > 0:
            stats['MoransI_hist_counts'], stats['MoransI_hist_edges'] = np.histogram(
                mi_valid, bins=50, range=(-0.5, 1.0)
            )

        # Activity from BinarisedData
        if condition in binarised_data_exp and binarised_data_exp[condition] is not None:
            bin_data = binarised_data_exp[condition]
            # Shape from HDF5: [trials, time, cols, rows] = (N, 185, 26, 13)
            print(f"    {condition}: bin_data.shape = {bin_data.shape}")
            # Apply frame selection (axis 1 = time)
            if frame_selection == 'prestim' and bin_data.shape[1] > prestim_frames:
                bin_data = bin_data[:, :prestim_frames, :, :]
            elif frame_selection == 'nostim':
                bin_data = apply_nostim_selection(bin_data, config, axis_time=1)

            # Standard activity: simple spatial mean (no mask)
            activity = np.mean(bin_data, axis=(2, 3)).T
            print(f"    {condition}: activity.shape = {activity.shape} (expected: [time, trials])")
            stats['Activity_all'] = activity.ravel()
            stats['Activity_mean'] = np.nanmean(stats['Activity_all'])
            stats['Activity_std'] = np.nanstd(stats['Activity_all'])

            # Masked activity variant (for dual summary figure comparison)
            # Masks out empty grid cells (blood vessels) per recording
            rec_meta = (recording_metadata or {}).get(condition, {})
            n_trials_per_rec = rec_meta.get('nTrials_per_recording', None)

            if n_trials_per_rec is not None and len(n_trials_per_rec) > 0:
                bin_data_masked = bin_data.astype(np.float64)
                trial_offset = 0
                total_valid = 0
                total_empty = 0
                for rec_i, n_rec_trials in enumerate(n_trials_per_rec):
                    rec_slice = bin_data[trial_offset:trial_offset + n_rec_trials]
                    rec_mask = np.any(rec_slice > 0, axis=(0, 1))
                    n_valid = int(np.sum(rec_mask))
                    n_empty = rec_mask.size - n_valid
                    total_valid += n_valid
                    total_empty += n_empty
                    bin_data_masked[trial_offset:trial_offset + n_rec_trials, :, ~rec_mask] = np.nan
                    trial_offset += n_rec_trials
                n_recs = len(n_trials_per_rec)
                print(f"    {condition}: Per-recording neuron mask ({n_recs} recs): "
                      f"avg {total_valid/n_recs:.0f}/{rec_mask.size} cells populated, "
                      f"avg {total_empty/n_recs:.0f} empty")
            else:
                neuron_mask = np.any(bin_data > 0, axis=(0, 1))
                n_valid = int(np.sum(neuron_mask))
                n_total = neuron_mask.size
                print(f"    {condition}: Global neuron mask: {n_valid}/{n_total} cells populated "
                      f"({n_total - n_valid} empty)")
                bin_data_masked = bin_data.astype(np.float64)
                bin_data_masked[:, :, ~neuron_mask] = np.nan

            activity_masked = np.nanmean(bin_data_masked, axis=(2, 3)).T
            stats['Activity_all_masked'] = activity_masked.ravel()
            stats['Activity_mean_masked'] = np.nanmean(activity_masked)
            stats['Activity_std_masked'] = np.nanstd(activity_masked.ravel())

            # Store Activity_trials matrices (time x trials) for trial-level analysis
            stats['Activity_trials'] = activity.copy()
            stats['Activity_trials_masked'] = activity_masked.copy()

            # Activity histogram
            act_valid = stats['Activity_all'][~np.isnan(stats['Activity_all'])]
            if len(act_valid) > 0:
                stats['Activity_hist_counts'], stats['Activity_hist_edges'] = np.histogram(
                    act_valid, bins=50, range=(0, 1)
                )

            # Blob detection on experimental data
            # bin_data shape: [trials, time, cols, rows]
            n_trials = bin_data.shape[0]
            n_time = bin_data.shape[1]

            all_blob_counts = []
            all_blob_lifetimes = []
            blob_counts_per_trial = []
            n_lifetimes_per_trial = []

            for trial in range(n_trials):
                # bin_data[trial] shape: [time, cols, rows] -> need [time, rows, cols]
                frames = np.transpose(bin_data[trial, :, :, :], (0, 2, 1))

                blob_stats = compute_blob_persistence_batch(
                    frames,
                    sigma=blob_params['sigma'],
                    threshold=blob_params['threshold'],
                    min_size=blob_params['min_size'],
                    iou_threshold=blob_params['iou_threshold']
                )

                all_blob_counts.extend(blob_stats.counts)
                blob_counts_per_trial.append(blob_stats.counts)
                n_lifetimes_per_trial.append(len(blob_stats.lifetimes))
                if len(blob_stats.lifetimes) > 0:
                    all_blob_lifetimes.extend(blob_stats.lifetimes)

            stats['BlobStats_counts'] = np.array(all_blob_counts)
            stats['BlobPersistence_lifetimes'] = np.array(all_blob_lifetimes)
            stats['BlobStats_counts_per_trial'] = blob_counts_per_trial

            # Pre-compute filtered BlobPersistence mean (same filter as Ising)
            min_lt = config.get('blob_min_lifetime', 1)
            lt_arr = np.array(all_blob_lifetimes)
            lt_filtered = lt_arr[lt_arr >= min_lt] if len(lt_arr) > 0 else lt_arr
            stats['BlobPersistence_mean'] = float(np.mean(lt_filtered)) if len(lt_filtered) > 0 else 0.0

            # Per-recording blob persistence means
            if n_trials_per_rec is not None and len(n_trials_per_rec) > 0:
                boundaries = np.cumsum([0] + n_lifetimes_per_trial)
                rec_bp_means = []
                trial_idx = 0
                for n_t in n_trials_per_rec:
                    if trial_idx + n_t <= len(boundaries) - 1:
                        rec_lts = np.concatenate([
                            lt_arr[boundaries[t]:boundaries[t+1]]
                            for t in range(trial_idx, trial_idx + n_t)
                        ]) if any(boundaries[t+1] > boundaries[t] for t in range(trial_idx, trial_idx + n_t)) else np.array([])
                    else:
                        rec_lts = np.array([])
                    trial_idx += n_t
                    if len(rec_lts) > 0:
                        filtered = rec_lts[rec_lts >= min_lt]
                        rec_bp_means.append(float(np.mean(filtered)) if len(filtered) > 0 else float(np.mean(rec_lts)))
                    else:
                        rec_bp_means.append(np.nan)
                stats['BlobPersistence_per_recording'] = np.array(rec_bp_means)

            # Blob count histogram
            if len(all_blob_counts) > 0:
                max_count = max(all_blob_counts) if all_blob_counts else 20
                stats['BlobCount_hist_counts'], stats['BlobCount_hist_edges'] = np.histogram(
                    all_blob_counts, bins=min(max_count + 1, 50), range=(0, max_count + 1)
                )

            # Blob lifetime histogram
            if len(all_blob_lifetimes) > 0:
                max_life = max(all_blob_lifetimes) if all_blob_lifetimes else 100
                stats['BlobLife_hist_counts'], stats['BlobLife_hist_edges'] = np.histogram(
                    all_blob_lifetimes, bins=min(max_life + 1, 50), range=(0, max_life + 1)
                )

            # Trial-averaged autocorrelation (matches MATLAB computeTrialAveragedAutocorr)
            tau_trial_avg, acf_trial_avg, r2_trial_avg = compute_trial_averaged_autocorr(
                activity,
                max_lag=autocorr_params['max_lag'],
                fit_range=autocorr_params['fit_range']
            )
            stats['Autocorr_tau_trial_averaged'] = tau_trial_avg
            stats['Autocorr_acf_trial_averaged'] = acf_trial_avg
            stats['Autocorr_r2_trial_averaged'] = r2_trial_avg

            # Also compute global activity ACF (average across trials first, then compute)
            global_activity = np.mean(activity, axis=1)  # Average across trials
            tau, acf, r2 = compute_autocorr_decay(
                global_activity,
                max_lag=autocorr_params['max_lag'],
                fit_range=autocorr_params['fit_range']
            )
            stats['Autocorr_tau'] = tau
            stats['Autocorr_acf'] = acf
            stats['Autocorr_r2'] = r2

        exp_stats[condition] = stats

    return exp_stats


def compute_global_experimental_statistics(exp_stats: dict, config: dict) -> dict:
    """
    Compute global statistics across all conditions.

    Matches MATLAB's global tau computation by concatenating all trials
    across conditions.
    """
    global_stats = {}
    autocorr_params = config['autocorr']

    # Collect activity trials from all conditions
    activity_dict = {}
    for condition in config['conditions']:
        if condition not in exp_stats:
            continue
        if 'Activity_trials' in exp_stats[condition]:
            activity_dict[condition] = exp_stats[condition]['Activity_trials']

    # Compute global autocorrelation
    tau_global, acf_global, r2_global = compute_global_autocorr(
        activity_dict,
        config['conditions'],
        max_lag=autocorr_params['max_lag'],
        fit_range=autocorr_params['fit_range']
    )

    global_stats['Autocorr_tau_global'] = tau_global
    global_stats['Autocorr_acf_global'] = acf_global
    global_stats['Autocorr_r2_global'] = r2_global

    # Aggregate Moran's I across conditions
    all_mi = []
    for condition in config['conditions']:
        if condition in exp_stats and 'MoransI_all' in exp_stats[condition]:
            all_mi.extend(exp_stats[condition]['MoransI_all'])
    if len(all_mi) > 0:
        global_stats['MoransI_all'] = np.array(all_mi)
        global_stats['MoransI_mean'] = np.nanmean(all_mi)
        global_stats['MoransI_std'] = np.nanstd(all_mi)

    return global_stats


# =============================================================================
# COMPARISON AND RANKING
# =============================================================================

# Metric registry: metric_name -> (keys, weights, fallback)
#   keys:     list of individual_dists keys to combine
#   weights:  dict key->weight, 'equal' for equal weights, or None for single-key
#   fallback: 'primary' = fall back to first key if others missing
#             'available' = use whatever keys are present
#             None = skip condition if keys missing
#             (only used for multi-key metrics)
METRIC_DEFS: Dict[str, Tuple[List[str], Any, Optional[str]]] = {
    'moransI':             (['moransI'], None, None),
    'activity':            (['activity'], None, None),
    'autocorr':            (['autocorr'], None, None),
    'blobCount':           (['blobCount'], None, None),
    'blobPersistence':     (['blobPersistence'], None, None),
    'moransI+activity':    (['moransI', 'activity'],
                            {'moransI': 0.5, 'activity': 0.5}, 'primary'),
    'spatial+persistence': (['moransI', 'activity', 'blobPersistence'],
                            'equal', 'available'),
    'moransI+activity_weighted': (['moransI', 'activity'],
                            {'moransI': 0.75, 'activity': 0.25}, 'primary'),
    'spatial+persistence_weighted': (['moransI', 'activity', 'blobPersistence'],
                            {'moransI': 0.5, 'activity': 0.25, 'blobPersistence': 0.25}, 'available'),
    'combined':            (['moransI', 'activity', 'blobCount', 'blobPersistence'],
                            'config', None),
}


def apply_custom_weights(args):
    """Override METRIC_DEFS weights based on CLI flags."""
    mw = getattr(args, 'moransI_weight', None)
    aw = getattr(args, 'activity_weight', None)
    pw = getattr(args, 'persistence_weight', None)
    if mw is None and aw is None and pw is None:
        return
    # If persistence weight is set, override spatial+persistence metric
    if pw is not None:
        mwv = mw if mw is not None else 0.5
        awv = aw if aw is not None else 0.25
        pwv = pw
        METRIC_DEFS['spatial+persistence'] = (
            ['moransI', 'activity', 'blobPersistence'],
            {'moransI': mwv, 'activity': awv, 'blobPersistence': pwv},
            'available'
        )
        print(f"Custom matching weights: moransI={mwv}, activity={awv}, blobPersistence={pwv}")
    else:
        # Override moransI+activity metric
        mwv = mw if mw is not None else 0.5
        awv = aw if aw is not None else 0.5
        METRIC_DEFS['moransI+activity'] = (
            ['moransI', 'activity'],
            {'moransI': mwv, 'activity': awv},
            'primary'
        )
        print(f"Custom matching weights: moransI={mwv}, activity={awv}")


def compute_metric_distance(
    metric: str, individual_dists: dict, n_sims: int,
    config_weights: dict, condition: str
) -> Optional[np.ndarray]:
    """Compute the combined distance array for a given metric.

    Returns the distance array, or ``None`` to skip this condition.
    """
    if metric not in METRIC_DEFS:
        print(f"    Warning: Unknown metric '{metric}', falling back to moransI")
        return individual_dists.get('moransI')

    keys, weights, fallback = METRIC_DEFS[metric]

    # Single-key metric
    if weights is None:
        key = keys[0]
        if key not in individual_dists:
            print(f"    Warning: No {key} data for {condition}")
            return None
        return individual_dists[key].copy()

    # Multi-key metric — determine effective keys & weights
    if weights == 'config':
        effective_weights = {k: config_weights.get(k, 0.0) for k in keys
                            if k in individual_dists and config_weights.get(k, 0.0) > 0}
    elif weights == 'equal':
        available_keys = [k for k in keys if k in individual_dists]
        if not available_keys:
            print(f"    Warning: No data for {condition}")
            return None
        # Check if all required keys are present
        missing = [k for k in keys if k not in individual_dists]
        if missing and fallback == 'available':
            print(f"    Warning: Missing data for {condition}, falling back to available metrics")
        elif missing and fallback == 'primary':
            print(f"    Warning: Missing data for {condition}, using {keys[0]} only")
            return individual_dists[keys[0]].copy()
        elif missing:
            print(f"    Warning: Missing data for {condition}")
            return None
        w = 1.0 / len(available_keys)
        effective_weights = {k: w for k in available_keys}
    else:
        # Explicit weight dict
        missing = [k for k in keys if k not in individual_dists]
        if missing:
            if fallback == 'primary':
                print(f"    Warning: Missing data for {condition}, using {keys[0]} only")
                return individual_dists[keys[0]].copy()
            elif fallback == 'available':
                available_keys = [k for k in keys if k in individual_dists]
                if not available_keys:
                    print(f"    Warning: No data for {condition}")
                    return None
                print(f"    Warning: Missing data for {condition}, falling back to available metrics")
                w = 1.0 / len(available_keys)
                effective_weights = {k: w for k in available_keys}
            else:
                print(f"    Warning: Missing data for {condition}")
                return None
        else:
            effective_weights = weights

    # Combine weighted normalized distances
    total_weight = sum(effective_weights.values())
    if total_weight == 0:
        return individual_dists.get('moransI', np.zeros(n_sims)).copy()

    wd = np.zeros(n_sims)
    for k, w in effective_weights.items():
        wd += w * normalize_distances(individual_dists[k])
    wd /= total_weight

    return wd


def compute_autocorr_distance(exp_tau: float, sim_taus: np.ndarray) -> np.ndarray:
    """
    Compute autocorrelation-based distance using log-ratio.

    Matches MATLAB implementation: distance = abs(log(tau_sim / tau_exp))
    """
    if np.isnan(exp_tau) or exp_tau <= 0:
        return np.full(len(sim_taus), np.nan)

    distances = np.zeros(len(sim_taus))
    for i, sim_tau in enumerate(sim_taus):
        if np.isnan(sim_tau) or sim_tau <= 0:
            distances[i] = np.nan
        else:
            distances[i] = abs(np.log(sim_tau / exp_tau))

    return distances


def compute_comparison(exp_stats: dict, ising_data: List[dict], sim_ids: List[str],
                        config: dict) -> dict:
    """
    Compute Wasserstein distances and rankings for each condition.

    Supports all matching metrics:
    - moransI: Moran's I distribution only
    - activity: Activity distribution only
    - autocorr: Autocorrelation tau (log-ratio distance)
    - blobCount: Blob count distribution only
    - blobPersistence: Blob persistence distribution only
    - moransI+activity: Combination of Moran's I and activity
    - spatial+persistence: Moran's I and blob persistence
    - combined: Full combined metric (all four)
    """
    comparison = {}
    metric = config['matching_metric']
    weights = config['matching_weights']

    for condition in config['conditions']:
        if condition not in exp_stats:
            continue

        print(f"  Computing distances for {condition}...")
        stats = exp_stats[condition]

        # Get experimental data
        exp_mi = stats['MoransI_all']
        exp_activity = stats.get('Activity_all', np.array([]))
        exp_blob_counts = stats.get('BlobStats_counts', None)
        exp_blob_lifetimes = stats.get('BlobPersistence_lifetimes', None)
        exp_tau = stats.get('Autocorr_tau_trial_averaged', stats.get('Autocorr_tau', np.nan))

        # Compute temporal scale factors
        temporal_scale_factors = np.ones(len(ising_data))
        sim_taus = np.array([d.get('Autocorr_tau', np.nan) for d in ising_data])
        if not np.isnan(exp_tau):
            for i, sim in enumerate(ising_data):
                sim_tau = sim.get('Autocorr_tau', np.nan)
                temporal_scale_factors[i] = compute_temporal_scale_factor(exp_tau, sim_tau)

        # Store individual distances for all metrics (always compute for DynamicsAnalysis)
        individual_dists = {}

        # Always compute Moran's I distances
        sim_mi = [d['MoransI_all'] for d in ising_data]
        individual_dists['moransI'] = wasserstein_batch(exp_mi, sim_mi, config['max_quantiles'])

        # Always compute Activity distances if available
        if len(exp_activity) > 0:
            sim_act = [d['Activity_all'] for d in ising_data]
            if config.get('log_activity', False):
                # log-transform activity to compress the high tail and
                # emphasise relative differences in the low-activity regime
                exp_activity_use = np.log10(np.asarray(exp_activity) + 1e-6)
                sim_act = [np.log10(np.asarray(a) + 1e-6) for a in sim_act]
            else:
                exp_activity_use = exp_activity
            individual_dists['activity'] = wasserstein_batch(exp_activity_use, sim_act, config['max_quantiles'])

        # Always compute autocorrelation distances
        individual_dists['autocorr'] = compute_autocorr_distance(exp_tau, sim_taus)

        # Compute blob count distances
        if exp_blob_counts is not None and len(exp_blob_counts) > 0:
            wd_blob_counts = np.zeros(len(ising_data))
            for i, sim in enumerate(ising_data):
                sim_counts = sim.get('BlobStats_counts', np.array([]))
                if len(sim_counts) == 0:
                    wd_blob_counts[i] = np.nan
                else:
                    wd_blob_counts[i] = wasserstein_1d(exp_blob_counts, sim_counts, config['max_quantiles'])
            individual_dists['blobCount'] = wd_blob_counts

        # Compute blob persistence distances (with temporal scaling)
        # Filter out transient blobs (lifetime < min_lifetime)
        min_lt = config.get('blob_min_lifetime', 1)

        # Use trial-chunked lifetimes matching experimental frame selection
        frame_sel = config.get('exp_frame_selection', 'all')
        prestim_frames = config.get('exp_prestim_frames', 80)
        full_trial_frames = config.get('exp_full_trial_frames', 185)
        blob_chunk = prestim_frames if frame_sel in ('prestim', 'nostim') else full_trial_frames
        blob_lifetime_key = f'BlobPersistence_lifetimes_chunk{blob_chunk}'

        if exp_blob_lifetimes is not None and len(exp_blob_lifetimes) > 0:
            exp_lt_filtered = exp_blob_lifetimes[exp_blob_lifetimes >= min_lt]
            if len(exp_lt_filtered) == 0:
                exp_lt_filtered = exp_blob_lifetimes  # Fallback if all filtered out

            wd_blob_persist = np.zeros(len(ising_data))
            for i, sim in enumerate(ising_data):
                sim_lifetimes = sim.get(blob_lifetime_key,
                                        sim.get('BlobPersistence_lifetimes', np.array([])))
                if len(sim_lifetimes) == 0:
                    wd_blob_persist[i] = np.nan
                else:
                    sim_lt_filtered = sim_lifetimes[sim_lifetimes >= min_lt]
                    if len(sim_lt_filtered) == 0:
                        wd_blob_persist[i] = np.nan  # No persistent blobs = can't compare
                        continue
                    # Apply temporal rescaling
                    scale = temporal_scale_factors[i]
                    if scale > 0 and scale != 1.0:
                        sim_lt_scaled = sim_lt_filtered * scale
                    else:
                        sim_lt_scaled = sim_lt_filtered
                    wd_blob_persist[i] = wasserstein_1d(exp_lt_filtered, sim_lt_scaled, config['max_quantiles'])
            individual_dists['blobPersistence'] = wd_blob_persist

        # Compute combined distance via metric registry
        wd = compute_metric_distance(
            metric, individual_dists, len(ising_data), weights, condition
        )
        if wd is None:
            continue

        # Sanitize NaN/inf before ranking — push invalid entries to the end
        finite_mask = np.isfinite(wd)
        if not np.all(finite_mask):
            max_finite = np.max(wd[finite_mask]) if np.any(finite_mask) else 1.0
            wd = np.where(finite_mask, wd, max_finite + 1.0)

        # Rank simulations
        rankings = np.argsort(wd)
        best_idx = rankings[:config['n_top_matches']]

        comparison[condition] = {
            'wasserstein_dist': wd,
            'individual_dists': individual_dists,
            'rankings': rankings,
            'bestMatch_idx': best_idx,
            'bestMatch_simIDs': [sim_ids[i] for i in best_idx],
            'bestMatch_params': {
                'beta': np.array([ising_data[i]['params']['beta'] for i in best_idx]),
                'c': np.array([ising_data[i]['params']['c'] for i in best_idx]),
                'decay_const': np.array([ising_data[i]['params']['decay_const'] for i in best_idx]),
                'inhibition_range': np.array([ising_data[i]['params']['inhibition_range'] for i in best_idx]),
                'bias': np.array([ising_data[i]['params']['bias'] for i in best_idx]),
            },
            'temporal_scale_factors': temporal_scale_factors,
        }

        # Print top matches
        print(f"    Top 5 matches:")
        for j in range(min(5, len(best_idx))):
            idx = best_idx[j]
            print(f"      {sim_ids[idx]}: WD={wd[idx]:.4f}")

    return comparison


def compute_comparison_dual(exp_stats: dict, ising_data: List[dict], sim_ids: List[str],
                             config: dict) -> Tuple[dict, dict]:
    """
    Compute comparison using BOTH centre crop and tiled Moran's I.

    Returns separate comparison dicts for each method for analysis.
    """
    comparison_centre = {}
    comparison_tiled = {}

    for condition in config['conditions']:
        if condition not in exp_stats:
            continue

        print(f"  Computing DUAL distances for {condition}...")
        stats = exp_stats[condition]

        exp_mi = stats['MoransI_all']

        # Compute using CENTRE crop Moran's I
        sim_mi_centre = [d.get('MoransI_centre', d['MoransI_all']) for d in ising_data]
        wd_centre = wasserstein_batch(exp_mi, sim_mi_centre, config['max_quantiles'])

        # Compute using TILED Moran's I
        sim_mi_tiled = [d.get('MoransI_tiled', d['MoransI_all']) for d in ising_data]
        wd_tiled = wasserstein_batch(exp_mi, sim_mi_tiled, config['max_quantiles'])

        # Rank and store results
        rankings_centre = np.argsort(wd_centre)
        best_idx_centre = rankings_centre[:config['n_top_matches']]

        rankings_tiled = np.argsort(wd_tiled)
        best_idx_tiled = rankings_tiled[:config['n_top_matches']]

        comparison_centre[condition] = {
            'wasserstein_dist': wd_centre,
            'rankings': rankings_centre,
            'bestMatch_idx': best_idx_centre,
            'bestMatch_simIDs': [sim_ids[i] for i in best_idx_centre],
        }

        comparison_tiled[condition] = {
            'wasserstein_dist': wd_tiled,
            'rankings': rankings_tiled,
            'bestMatch_idx': best_idx_tiled,
            'bestMatch_simIDs': [sim_ids[i] for i in best_idx_tiled],
        }

        # Report comparison
        print(f"    Centre crop - Best match: {sim_ids[best_idx_centre[0]]} (WD={wd_centre[best_idx_centre[0]]:.4f})")
        print(f"    Tiled       - Best match: {sim_ids[best_idx_tiled[0]]} (WD={wd_tiled[best_idx_tiled[0]]:.4f})")

        # Check overlap in top matches
        overlap = len(set(best_idx_centre) & set(best_idx_tiled))
        print(f"    Overlap in top {config['n_top_matches']}: {overlap}")

    return comparison_centre, comparison_tiled


def compute_dynamics_analysis(exp_stats: dict, ising_data: List[dict],
                               comparison: dict, config: dict) -> dict:
    """
    Compute DynamicsAnalysis struct matching MATLAB Section 7.

    For each condition:
    - Experimental timecourse and ACF
    - Best Ising match ACF
    - Time constant comparison
    - Temporal scaling factors
    """
    dynamics = {}

    for condition in config['conditions']:
        if condition not in exp_stats or condition not in comparison:
            continue

        stats = exp_stats[condition]
        comp = comparison[condition]

        cond_dynamics = {}

        # Experimental ACF
        cond_dynamics['exp_tau'] = stats.get('Autocorr_tau_trial_averaged', stats.get('Autocorr_tau', np.nan))
        cond_dynamics['exp_acf'] = stats.get('Autocorr_acf_trial_averaged', stats.get('Autocorr_acf', np.array([])))
        cond_dynamics['exp_tau_r2'] = stats.get('Autocorr_r2_trial_averaged', stats.get('Autocorr_r2', np.nan))

        # Best match Ising ACF
        best_idx = comp['bestMatch_idx']
        if len(best_idx) > 0:
            best_sim = ising_data[best_idx[0]]
            cond_dynamics['best_ising_tau'] = best_sim.get('Autocorr_tau', np.nan)
            cond_dynamics['best_ising_acf'] = best_sim.get('Autocorr_acf', np.array([]))
            cond_dynamics['best_ising_r2'] = best_sim.get('Autocorr_r2', np.nan)

            # Time constant comparison (guard against zero denominators)
            exp_t = cond_dynamics['exp_tau']
            ising_t = cond_dynamics['best_ising_tau']
            if (not np.isnan(exp_t) and not np.isnan(ising_t)
                    and exp_t > 0 and ising_t > 0):
                cond_dynamics['tau_ratio'] = ising_t / exp_t
                cond_dynamics['tau_scale_factor'] = exp_t / ising_t
            else:
                cond_dynamics['tau_ratio'] = np.nan
                cond_dynamics['tau_scale_factor'] = np.nan

        # Store scale factors for all best matches
        cond_dynamics['scale_factors_top'] = comp['temporal_scale_factors'][best_idx]

        dynamics[condition] = cond_dynamics

    return dynamics


def compute_parameter_trends(ising_data: List[dict], comparison: dict, config: dict) -> dict:
    """
    Compute ParameterTrends analysis matching MATLAB.

    Analyzes which parameter values appear in best matches for each condition.
    """
    trends = {}
    param_names = ['beta', 'c', 'decay_const', 'inhibition_range', 'bias']

    for condition in config['conditions']:
        if condition not in comparison:
            continue

        comp = comparison[condition]

        cond_trends = {}
        for param in param_names:
            values = comp['bestMatch_params'][param]
            cond_trends[param] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }

        trends[condition] = cond_trends

    # Cross-condition summary
    trends['summary'] = {}
    for param in param_names:
        all_values = []
        for condition in config['conditions']:
            if condition in comparison:
                all_values.extend(comparison[condition]['bestMatch_params'][param])
        if len(all_values) > 0:
            trends['summary'][param] = {
                'mean': np.mean(all_values),
                'std': np.std(all_values),
            }

    return trends


# =============================================================================
# CHECKPOINTING AND CACHE
# =============================================================================

def save_checkpoint(checkpoint_path: str, ising_data: List[dict], sim_ids: List[str],
                     processed_files: List[str]):
    """Save processing checkpoint."""
    checkpoint = {
        'ising_data': ising_data,
        'sim_ids': sim_ids,
        'processed_files': processed_files,
        'timestamp': datetime.now().isoformat()
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved: {len(processed_files)} simulations")


def load_checkpoint(checkpoint_path: str) -> Optional[dict]:
    """Load processing checkpoint."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            print(f"WARNING: Corrupt checkpoint file ({e}), ignoring")
    return None


def get_cache_path(config: dict) -> str:
    """Get cache file path based on config hash."""
    config_hash = compute_config_hash(config)
    cache_dir = config.get('cache_path', config.get('output_path', '.'))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'ising_cache_{config_hash}.pkl')


def save_cache(cache_path: str, ising_data: List[dict], sim_ids: List[str],
               config: dict, sim_files: List[str]):
    """Save processed Ising data to cache."""
    cache = {
        'ising_data': ising_data,
        'sim_ids': sim_ids,
        'config_hash': compute_config_hash(config),
        'sim_files': [os.path.basename(f) for f in sim_files],
        'n_simulations': len(ising_data),
        'timestamp': datetime.now().isoformat()
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cache saved: {len(ising_data)} simulations -> {cache_path}")


def load_cache(cache_path: str, config: dict, sim_files: List[str]) -> Optional[Tuple[List[dict], List[str], set]]:
    """
    Load processed Ising data from cache if valid.

    Supports incremental cache growth: if the cache contains a subset of
    the current simulation files, the cached data is returned alongside
    the set of cached basenames so the caller can process only new files.

    Returns (ising_data, sim_ids, cached_basenames) on full or partial hit,
    None on config mismatch or corruption.
    """
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)

        # Validate config hash
        current_hash = compute_config_hash(config)
        if cache.get('config_hash') != current_hash:
            print(f"Cache invalid: config hash mismatch")
            return None

        # Validate simulation files
        cached_files = set(cache.get('sim_files', []))
        current_files = set(os.path.basename(f) for f in sim_files)

        if cached_files == current_files:
            print(f"Cache valid (full): {cache.get('n_simulations', 0)} simulations "
                  f"from {cache.get('timestamp', 'unknown')}")
            return cache['ising_data'], cache['sim_ids'], cached_files

        if cached_files.issubset(current_files):
            print(f"Cache valid (partial): {len(cached_files)}/{len(current_files)} simulations "
                  f"from {cache.get('timestamp', 'unknown')}")
            return cache['ising_data'], cache['sim_ids'], cached_files

        # Cached files contain entries not in current set — stale cache
        print(f"Cache invalid: simulation file mismatch "
              f"(cached={len(cached_files)}, current={len(current_files)})")
        return None

    except Exception as e:
        print(f"Cache load error: {e}")
        return None


# =============================================================================
# MAIN — EXTRACTED STAGES
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse and post-process command-line arguments."""
    parser = argparse.ArgumentParser(description='Optimized Ising Model Comparison')
    parser.add_argument('--local', action='store_true', help='Run with local paths')
    parser.add_argument('--metric', type=str, default='combined',
                        choices=['moransI', 'activity', 'autocorr', 'blobCount', 'blobPersistence',
                                 'moransI+activity', 'moransI+activity_weighted',
                                 'spatial+persistence', 'spatial+persistence_weighted',
                                 'combined'],
                        help='Matching metric')
    parser.add_argument('--n-workers', type=int, default=64,
                        help='Number of parallel workers (default: 64)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint file')
    parser.add_argument('--max-sims', type=int, default=0,
                        help='Maximum simulations to process (0=all)')
    parser.add_argument('--checkpoint-interval', type=int, default=500,
                        help='Save checkpoint every N simulations')
    parser.add_argument('--max-timesteps', type=int, default=5000,
                        help='Max timesteps per simulation (0=all, default=5000)')
    parser.add_argument('--no-limit-timesteps', action='store_true',
                        help='Process all 100K timesteps (slow)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable cache (reprocess all simulations)')
    parser.add_argument('--numba-threads', type=int, default=1,
                        help='Numba threads per worker (default: 1 for process-level parallelism)')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualization figures (default: True)')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--figures-only', type=str, default=None,
                        help='Path to existing results file - skip processing and only generate figures')
    parser.add_argument('--no-sensitivity', action='store_true',
                        help='Skip ACF fit range sensitivity analysis (runs by default with --visualize)')
    parser.add_argument('--frame-label', type=str, default=None,
                        choices=['prestim', 'full_trial', 'nostim'],
                        help='Process only this frame selection (default: all three)')
    parser.add_argument('--connectivity', type=str, choices=['rook', 'queen'], default='rook',
                        help='Neighbor connectivity: rook (4-connected) or queen (8-connected)')
    parser.add_argument('--sim-start', type=int, default=0,
                        help='Start index into sorted sim file list (inclusive)')
    parser.add_argument('--sim-end', type=int, default=0,
                        help='End index into sorted sim file list (exclusive, 0=all)')
    parser.add_argument('--merge-chunks', action='store_true',
                        help='Merge chunked partial results and run matching/figures')
    parser.add_argument('--output-suffix', type=str, default='',
                        help='Suffix appended to output directory (e.g., _1k for reduced timesteps)')
    parser.add_argument('--min-beta', type=float, default=None,
                        help='Filter: minimum beta value (inclusive)')
    parser.add_argument('--max-beta', type=float, default=None,
                        help='Filter: maximum beta value (inclusive)')
    parser.add_argument('--min-bias', type=float, default=None,
                        help='Filter: minimum bias value (inclusive)')
    parser.add_argument('--max-bias', type=float, default=None,
                        help='Filter: maximum bias value (inclusive)')
    parser.add_argument('--moransI-weight', type=float, default=None,
                        help='Override moransI weight in moransI+activity or spatial+persistence metric')
    parser.add_argument('--activity-weight', type=float, default=None,
                        help='Override activity weight in moransI+activity or spatial+persistence metric')
    parser.add_argument('--persistence-weight', type=float, default=None,
                        help='Override blobPersistence weight in spatial+persistence metric')
    parser.add_argument('--log-activity', action='store_true',
                        help='Use log10(activity+1e-6) for Wasserstein activity distance')
    parser.add_argument('--refractory', type=int, default=0,
                        help='Use IsingSims_refractoryK{N} as data root (default 0=off)')
    args = parser.parse_args()

    if args.no_visualize:
        args.visualize = False

    return args


def load_or_process_simulations(
    config: dict, args: argparse.Namespace, sim_files: List[str],
    weight_mat: np.ndarray, centre_idx: Tuple[int, int, int, int],
    tile_row_starts: np.ndarray, tile_col_starts: np.ndarray
) -> Tuple[List[dict], List[str]]:
    """Load simulations from cache/checkpoint, process remaining, and return results.

    Returns (ising_data, sim_ids) with guaranteed alignment.
    """
    cache_path = get_cache_path(config)
    use_cache = config.get('use_cache', True) and not args.no_cache

    ising_data = []
    sim_ids = []
    cached_basenames = set()

    if use_cache:
        print(f"\nChecking cache: {cache_path}")
        cached = load_cache(cache_path, config, sim_files)
        if cached is not None:
            ising_data, sim_ids, cached_basenames = cached
            print(f"Using cached data: {len(ising_data)} simulations")

    # Check for checkpoint (resumes partial processing after timeout/crash)
    # Skip checkpoint entirely in chunk mode — chunks share output dir so the
    # checkpoint file would be raced across workers.
    checkpoint_path = os.path.join(config['output_path'], 'processing_checkpoint.pkl')
    processed_basenames = set()

    is_chunk_mode = (getattr(args, 'sim_end', 0) > 0)
    checkpoint = None if is_chunk_mode else load_checkpoint(checkpoint_path)
    if checkpoint:
        checkpoint_basenames = set(os.path.basename(f) for f in checkpoint['processed_files'])
        new_in_checkpoint = checkpoint_basenames - cached_basenames

        if len(new_in_checkpoint) > 0:
            if len(checkpoint['ising_data']) > len(ising_data):
                ising_data = checkpoint['ising_data']
                sim_ids = checkpoint['sim_ids']
                processed_basenames = checkpoint_basenames
                print(f"\nResumed from checkpoint: {len(processed_basenames)} simulations "
                      f"(supersedes cache with {len(cached_basenames)} sims)")
            else:
                n_before = len(ising_data)
                cache_id_set = set(sim_ids)
                for idx, sid in enumerate(checkpoint['sim_ids']):
                    if sid not in cache_id_set and idx < len(checkpoint['ising_data']):
                        ising_data.append(checkpoint['ising_data'][idx])
                        sim_ids.append(sid)
                processed_basenames = checkpoint_basenames | cached_basenames
                print(f"\nMerged checkpoint: {len(ising_data) - n_before} new sims, "
                      f"{len(ising_data)} total")
        else:
            processed_basenames = cached_basenames
            print(f"\nCheckpoint found but fully covered by cache ({len(cached_basenames)} sims)")
    else:
        processed_basenames = cached_basenames

    # Filter to unprocessed files (exclude both cached and checkpointed)
    remaining_files = [f for f in sim_files
                       if os.path.basename(f) not in processed_basenames]
    print(f"\nRemaining simulations to process: {len(remaining_files)}")

    if len(remaining_files) > 0:
        n_workers = args.n_workers if args.n_workers > 0 else min(cpu_count(), 32)
        numba_threads = os.environ.get('NUMBA_NUM_THREADS', 'auto')
        print(f"Using {n_workers} parallel workers, Numba threads per worker: {numba_threads}")

        print(f"\nProcessing {len(remaining_files)} simulations...")
        process_start = time.time()

        process_args = [
            (f, config, weight_mat, centre_idx, tile_row_starts, tile_col_starts)
            for f in remaining_files
        ]

        # NOTE: sim_ids are built in lock-step with ising_data to avoid
        # misalignment when a simulation file fails to load (returns None).
        with Pool(n_workers) as pool:
            results_iter = pool.imap(process_simulation_wrapper, process_args)

            for i, result in enumerate(tqdm(results_iter, total=len(remaining_files),
                                              desc="Processing simulations")):
                if result is not None:
                    ising_data.append(result)
                    processed_basenames.add(os.path.basename(remaining_files[i]))
                    params = filename_to_params(remaining_files[i])
                    if params:
                        sim_ids.append(params_to_id_string(
                            params['beta'], params['c'], params['decay_const'],
                            params['inhibition_range'], params['bias']))
                    else:
                        sim_ids.append(os.path.basename(remaining_files[i]).replace('.mat', ''))

                if (i + 1) % args.checkpoint_interval == 0 and not is_chunk_mode:
                    save_checkpoint(checkpoint_path, ising_data, sim_ids,
                                    list(processed_basenames))

        process_time = time.time() - process_start
        print(f"Processing complete: {process_time:.1f}s ({process_time/len(remaining_files):.2f}s per simulation)")

        # Save merged cache (cached + newly processed)
        if use_cache and len(ising_data) > 0:
            save_cache(cache_path, ising_data, sim_ids, config, sim_files)
            print(f"Cache updated: {len(ising_data)} total simulations")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Removed checkpoint file")

    assert len(sim_ids) == len(ising_data), (
        f"sim_ids/ising_data length mismatch: {len(sim_ids)} vs {len(ising_data)}"
    )

    return ising_data, sim_ids


def build_and_save_results(
    config_run: dict, config: dict, ising_data: List[dict], sim_ids: List[str],
    exp_stats: dict, global_exp_stats: dict,
    comparison: dict, comparison_centre: Optional[dict], comparison_tiled: Optional[dict],
    dynamics_analysis: dict, parameter_trends: dict,
    recording_metadata: dict, start_time: float, frame_label: str
) -> Tuple[dict, str]:
    """Assemble HDF5 results dict and save to disk.

    Returns (Results dict, output_file path).
    """
    print("\nPreparing results for saving...")

    min_lt = config_run.get('blob_min_lifetime', 1)

    # Use trial-chunked lifetimes matching experimental frame selection
    frame_sel = config_run.get('exp_frame_selection', 'all')
    prestim_frames = config_run.get('exp_prestim_frames', 80)
    full_trial_frames = config_run.get('exp_full_trial_frames', 185)
    blob_chunk = prestim_frames if frame_sel in ('prestim', 'nostim') else full_trial_frames
    blob_key = f'BlobPersistence_lifetimes_chunk{blob_chunk}'

    IsingData = {
        'simIDs': np.array(sim_ids, dtype=object),
        'MoransI_mean': np.array([d['MoransI_mean'] for d in ising_data]),
        'MoransI_std': np.array([d['MoransI_std'] for d in ising_data]),
        'Activity_mean': np.array([d['Activity_mean'] for d in ising_data]),
        'Activity_std': np.array([d['Activity_std'] for d in ising_data]),
        'Autocorr_tau': np.array([d.get('Autocorr_tau', np.nan) for d in ising_data]),
        'Autocorr_acf': [d.get('Autocorr_acf', np.array([])) for d in ising_data],
        'MoransI_all': [d['MoransI_all'] for d in ising_data],
        'Activity_all': [d['Activity_all'] for d in ising_data],
        'BlobStats_counts': [d.get('BlobStats_counts', np.array([])) for d in ising_data],
        'BlobPersistence_lifetimes': [d.get(blob_key, d.get('BlobPersistence_lifetimes', np.array([]))) for d in ising_data],
        'BlobPersistence_mean': np.array([
            (lambda lts: float(np.mean(lts[lts >= min_lt])) if len(lts[lts >= min_lt]) > 0
             else float(np.mean(lts)) if len(lts) > 0 else 0.0)(
                d.get(blob_key, d.get('BlobPersistence_lifetimes', np.array([]))))
            for d in ising_data
        ]),
        'params': {
            'beta': np.array([d['params']['beta'] for d in ising_data]),
            'c': np.array([d['params']['c'] for d in ising_data]),
            'decay_const': np.array([d['params']['decay_const'] for d in ising_data]),
            'inhibition_range': np.array([d['params']['inhibition_range'] for d in ising_data]),
            'bias': np.array([d['params']['bias'] for d in ising_data]),
        }
    }

    if config_run['grid_mode'] == 'subselect_centre_vs_tiled':
        IsingData['MoransI_centre'] = [d.get('MoransI_centre', d['MoransI_all']) for d in ising_data]
        IsingData['MoransI_tiled'] = [d.get('MoransI_tiled', d['MoransI_all']) for d in ising_data]

    Results = {
        'config': {
            'grid_mode': config_run['grid_mode'],
            'matching_metric': config_run['matching_metric'],
            'connectivity': config_run.get('connectivity', 'rook'),
            'experimental_grid': config_run['experimental_grid'],
            'ising_grid': config_run['ising_grid'],
            'blob_params': config_run['blob_params'],
            'matching_weights': config_run['matching_weights'],
            'limit_timesteps': config_run.get('limit_timesteps', False),
            'max_timesteps': config_run.get('max_timesteps', 0),
            'blob_min_lifetime': config_run.get('blob_min_lifetime', 1),
            'exp_frame_selection': config_run['exp_frame_selection'],
            'exp_prestim_frames': config_run['exp_prestim_frames'],
            'exp_stim_onset_frame': config_run.get('exp_stim_onset_frame', 81),
            'exp_stim_offset_frame': config_run.get('exp_stim_offset_frame', 100),
        },
        'IsingData': IsingData,
        'ExpStats': exp_stats,
        'GlobalExpStats': global_exp_stats,
        'Comparison': comparison,
        'DynamicsAnalysis': dynamics_analysis,
        'ParameterTrends': parameter_trends,
        'RecordingMetadata': recording_metadata,
        'timestamp': datetime.now().isoformat(),
        'processing_time_s': time.time() - start_time,
    }

    if comparison_centre is not None:
        Results['Comparison_centre'] = comparison_centre
        Results['Comparison_tiled'] = comparison_tiled

    connectivity_suffix = '_queen' if config_run.get('connectivity') == 'queen' else ''
    chunk_suffix = config_run.get('chunk_suffix', '')
    output_file = os.path.join(
        config['output_path'],
        f"IsingComparison_Results_{frame_label}_{config_run['grid_mode']}_{config_run['matching_metric']}{connectivity_suffix}{chunk_suffix}_optimized.mat"
    )
    print(f"\nSaving results to: {output_file}")
    save_to_hdf5(output_file, Results)

    return Results, output_file


def run_frame_analysis(
    frame_sel: str, frame_label: str,
    config: dict, ising_data: List[dict], sim_ids: List[str],
    morans_i_exp: dict, binarised_data_exp: dict,
    recording_metadata: dict, args: argparse.Namespace, start_time: float
) -> str:
    """Run experimental statistics, comparison, and save for one frame selection.

    Returns the output file path.
    """
    print(f"\n{'=' * 70}")
    print(f"  Processing frame selection: {frame_label} ({frame_sel})")
    print(f"{'=' * 70}")

    config_run = copy.deepcopy(config)
    config_run['exp_frame_selection'] = frame_sel

    # Compute experimental statistics (frame-dependent)
    print("\nComputing experimental statistics...")
    exp_stats = compute_experimental_statistics(
        morans_i_exp, binarised_data_exp, config_run, recording_metadata)
    for condition in exp_stats:
        stats = exp_stats[condition]
        print(f"  {condition}: MI mean={stats['MoransI_mean']:.4f}, std={stats['MoransI_std']:.4f}")
        if 'Autocorr_tau_trial_averaged' in stats:
            print(f"           Autocorr tau (trial-avg)={stats['Autocorr_tau_trial_averaged']:.2f}")
        elif 'Autocorr_tau' in stats:
            print(f"           Autocorr tau={stats['Autocorr_tau']:.2f}")

    # Compute global experimental statistics
    print("\nComputing global experimental statistics...")
    global_exp_stats = compute_global_experimental_statistics(exp_stats, config_run)
    if 'Autocorr_tau_global' in global_exp_stats:
        print(f"  Global Autocorr tau: {global_exp_stats['Autocorr_tau_global']:.2f}")

    # Compute comparison
    print("\nComputing Wasserstein distances...")
    comparison = compute_comparison(exp_stats, ising_data, sim_ids, config_run)

    # Compute centre vs tiled dual comparison (if using subselect_centre_vs_tiled mode)
    comparison_centre = None
    comparison_tiled = None
    if config_run['grid_mode'] == 'subselect_centre_vs_tiled':
        print("\nComputing centre vs tiled comparison...")
        comparison_centre, comparison_tiled = compute_comparison_dual(exp_stats, ising_data, sim_ids, config_run)

    # Compute DynamicsAnalysis
    print("\nComputing dynamics analysis...")
    dynamics_analysis = compute_dynamics_analysis(exp_stats, ising_data, comparison, config_run)
    for condition in dynamics_analysis:
        da = dynamics_analysis[condition]
        if 'tau_ratio' in da and not np.isnan(da.get('tau_ratio', np.nan)):
            print(f"  {condition}: tau_ratio={da['tau_ratio']:.2f}, scale_factor={da['tau_scale_factor']:.2f}")

    # Compute ParameterTrends
    print("\nComputing parameter trends...")
    parameter_trends = compute_parameter_trends(ising_data, comparison, config_run)
    if 'summary' in parameter_trends:
        print("  Cross-condition summary:")
        for param, stats in parameter_trends['summary'].items():
            print(f"    {param}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    # Build and save results
    Results, output_file = build_and_save_results(
        config_run, config, ising_data, sim_ids,
        exp_stats, global_exp_stats,
        comparison, comparison_centre, comparison_tiled,
        dynamics_analysis, parameter_trends,
        recording_metadata, start_time, frame_label
    )

    # Generate visualizations
    if args.visualize and HAS_VISUALIZATION:
        print("\n" + "=" * 70)
        print(f"  Generating Visualizations ({frame_label})")
        print("=" * 70)
        try:
            figures_dir = os.path.join(config['output_path'], 'figures')
            os.makedirs(figures_dir, exist_ok=True)

            viz = IsingVisualizer(Results, config_run, figures_dir,
                                  ising_data_path=config['ising_data_path'],
                                  exp_data_path=config['experimental_data_path'],
                                  frame_label=frame_label)
            viz.plot_all()
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            traceback.print_exc()
    elif args.visualize and not HAS_VISUALIZATION:
        print("\nNote: Visualization skipped (ising_visualizations module not available)")
        print("  To enable visualizations, ensure ising_visualizations.py is in the same directory")

    # Run ACF fit range sensitivity analysis (by default when visualizing)
    if args.visualize and not args.no_sensitivity:
        print("\n" + "=" * 70)
        print(f"  Running ACF Fit Range Sensitivity Analysis ({frame_label})")
        print("=" * 70)
        try:
            from autocorr_sensitivity import analyze_fit_range_sensitivity, plot_sensitivity_results
            sensitivity_results = analyze_fit_range_sensitivity(
                ising_data, exp_stats, comparison, config_run
            )
            # Plot base variant (4 conditions)
            base_config = copy.deepcopy(config_run)
            base_config['conditions'] = [c for c in config_run['conditions']
                                         if c in ('Naive', 'Beginner', 'Expert', 'NoSpout')]
            sensitivity_dir_base = os.path.join(
                config['output_path'], 'figures', 'sensitivity', f'{frame_label}_base')
            plot_sensitivity_results(sensitivity_results, base_config, sensitivity_dir_base)

            # Plot with_hit_miss variant (all 6 conditions) if hit/miss data exists
            has_hm = any(c in sensitivity_results.get('ising_tau', {})
                         for c in ('Expert_Hit', 'Expert_Miss'))
            if has_hm:
                sensitivity_dir_hm = os.path.join(
                    config['output_path'], 'figures', 'sensitivity', f'{frame_label}_with_hit_miss')
                plot_sensitivity_results(sensitivity_results, config_run, sensitivity_dir_hm)
        except ImportError:
            print("Warning: autocorr_sensitivity module not available")
        except Exception as e:
            print(f"Warning: Sensitivity analysis failed: {e}")
            traceback.print_exc()

    return output_file


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # Handle --figures-only: skip all processing and just regenerate figures
    if args.figures_only:
        print("=" * 70)
        print("  Regenerating figures from existing results")
        print("=" * 70)
        print(f"\nLoading results from: {args.figures_only}")

        if not HAS_VISUALIZATION:
            print("ERROR: Visualization module not available")
            sys.exit(1)

        config = get_config(is_local=args.local, metric=args.metric, connectivity=args.connectivity, refractory_K=args.refractory)

        from ising_visualizations import plot_comparison_results
        output_dir = os.path.join(os.path.dirname(args.figures_only), 'figures')
        plot_comparison_results(args.figures_only, output_dir=output_dir,
                                ising_data_path=config['ising_data_path'],
                                exp_data_path=config['experimental_data_path'])
        print(f"\nFigures saved to: {output_dir}")
        sys.exit(0)

    # Handle --merge-chunks: load partial chunk files, merge, run full analysis
    if args.merge_chunks:
        import pickle
        print("=" * 70)
        print("  Merging chunked comparison results")
        print("=" * 70)

        config = get_config(is_local=args.local, metric=args.metric, connectivity=args.connectivity, refractory_K=args.refractory)
        if args.output_suffix:
            config['output_path'] = config['output_path'] + args.output_suffix
        chunk_pattern = os.path.join(config['output_path'], 'chunk_*.pkl')
        chunk_files = sorted(glob(chunk_pattern))
        print(f"\nFound {len(chunk_files)} chunk files in {config['output_path']}")

        if len(chunk_files) == 0:
            print("ERROR: No chunk files found!")
            sys.exit(1)

        ising_data = []
        sim_ids = []
        for cf in chunk_files:
            with open(cf, 'rb') as f:
                chunk = pickle.load(f)
            ising_data.extend(chunk['ising_data'])
            sim_ids.extend(chunk['sim_ids'])
            print(f"  Loaded {len(chunk['ising_data'])} sims from {os.path.basename(cf)}")

        print(f"\nTotal merged: {len(ising_data)} simulations")

        start_time = time.time()
        queen = (args.connectivity == 'queen')
        exp_grid = config['experimental_grid']
        weight_mat = create_weight_matrix_jit(exp_grid[0], exp_grid[1], queen)

        # Set ising_grid (needed by build_and_save_results); known from sim convention
        config['ising_grid'] = (39, 78)
        config['connectivity'] = args.connectivity
        config['log_activity'] = args.log_activity
        if args.no_limit_timesteps:
            config['limit_timesteps'] = False
        else:
            config['limit_timesteps'] = True
            config['max_timesteps'] = args.max_timesteps

        # Custom matching weights override
        apply_custom_weights(args)

        morans_i_exp, binarised_data_exp, recording_metadata = load_experimental_data(config)

        ALL_FRAME_SELECTIONS = [
            ('prestim', 'prestim'),
            ('nostim',  'nostim'),
            ('all',     'full_trial'),
        ]
        if args.frame_label:
            frame_selections = [fs for fs in ALL_FRAME_SELECTIONS if fs[1] == args.frame_label]
        else:
            frame_selections = ALL_FRAME_SELECTIONS

        output_files = []
        for frame_sel, frame_label in frame_selections:
            output_file = run_frame_analysis(
                frame_sel, frame_label, config, ising_data, sim_ids,
                morans_i_exp, binarised_data_exp, recording_metadata,
                args, start_time
            )
            output_files.append(output_file)

        for cf in chunk_files:
            os.remove(cf)
        print(f"\nCleaned up {len(chunk_files)} chunk files")

        total_time = time.time() - start_time
        print(f"\nMerge complete in {total_time:.1f}s")
        for of in output_files:
            print(f"Results saved to: {of}")
        sys.exit(0)

    print("=" * 70)
    print("  Figure 5: Ising Model vs Experimental Data Comparison")
    print("  OPTIMIZED Python Implementation (Numba JIT)")
    print("=" * 70)

    start_time = time.time()

    # Get configuration
    config = get_config(is_local=args.local, metric=args.metric, connectivity=args.connectivity, refractory_K=args.refractory)

    if args.output_suffix:
        config['output_path'] = config['output_path'] + args.output_suffix

    if args.no_limit_timesteps:
        config['limit_timesteps'] = False
    else:
        config['limit_timesteps'] = True
        config['max_timesteps'] = args.max_timesteps

    config['connectivity'] = args.connectivity
    config['log_activity'] = args.log_activity

    # Custom matching weights override
    apply_custom_weights(args)

    print(f"\nEnvironment: {'LOCAL' if args.local else 'CLUSTER'}")
    print(f"Grid mode: {config['grid_mode']}")
    print(f"Matching metric: {config['matching_metric']}")
    print(f"Connectivity: {config['connectivity']}")
    if config['limit_timesteps']:
        print(f"Timestep limit: {config['max_timesteps']} (use --no-limit-timesteps for full 100K)")
    else:
        print("Timestep limit: DISABLED (processing all frames)")

    os.makedirs(config['output_path'], exist_ok=True)
    print(f"Output path: {config['output_path']}")

    # JIT warmup
    print("\nWarming up JIT compilation...")
    warmup_jit()

    # Find simulation files
    sim_pattern = os.path.join(config['ising_data_path'], 'sim_be_*.mat')
    sim_files = sorted(glob(sim_pattern))
    n_sims_total = len(sim_files)
    print(f"\nFound {n_sims_total} simulation files")

    if n_sims_total == 0:
        print("ERROR: No simulation files found!")
        sys.exit(1)

    if args.max_sims > 0:
        sim_files = sim_files[:args.max_sims]

    # Parameter filter (restrict to a sub-box of the parameter grid)
    has_filter = any(getattr(args, f) is not None
                     for f in ['min_beta', 'max_beta', 'min_bias', 'max_bias'])
    if has_filter:
        filtered = []
        for f in sim_files:
            p = filename_to_params(f)
            if p is None:
                continue
            if args.min_beta is not None and p['beta'] < args.min_beta:
                continue
            if args.max_beta is not None and p['beta'] > args.max_beta:
                continue
            if args.min_bias is not None and p['bias'] < args.min_bias:
                continue
            if args.max_bias is not None and p['bias'] > args.max_bias:
                continue
            filtered.append(f)
        print(f"Parameter filter: {len(sim_files)} -> {len(filtered)} sims "
              f"(beta in [{args.min_beta},{args.max_beta}], bias in [{args.min_bias},{args.max_bias}])")
        sim_files = filtered

    # Chunk mode: slice sim_files by index range
    is_chunk = (args.sim_end > 0)
    if is_chunk:
        sim_files = sim_files[args.sim_start:args.sim_end]
        print(f"Chunk mode: sims [{args.sim_start}:{args.sim_end}] → {len(sim_files)} simulations")
    elif args.sim_start > 0:
        sim_files = sim_files[args.sim_start:]
        print(f"Sim range: [{args.sim_start}:end] → {len(sim_files)} simulations")

    n_sims = len(sim_files)

    # Chunk mode + empty after filter: save empty pickle and exit cleanly,
    # so merge dependency (afterok) still fires.
    if is_chunk and n_sims == 0:
        import pickle
        chunk_file = os.path.join(
            config['output_path'],
            f"chunk_{args.sim_start}_{args.sim_end}.pkl"
        )
        os.makedirs(config['output_path'], exist_ok=True)
        with open(chunk_file, 'wb') as f:
            pickle.dump({'ising_data': [], 'sim_ids': []}, f, protocol=4)
        print(f"\nEmpty chunk: {chunk_file} (0 sims after filter), exit 0")
        sys.exit(0)

    # Auto-detect grid size
    first_sim = sio.loadmat(sim_files[0], squeeze_me=True)
    stored_spins = first_sim['stored_spins']
    ising_grid = (stored_spins.shape[1], stored_spins.shape[2])
    config['ising_grid'] = ising_grid
    print(f"Auto-detected Ising grid: {ising_grid[0]} x {ising_grid[1]}")

    # Create weight matrix (optimized)
    exp_grid = config['experimental_grid']
    queen = (args.connectivity == 'queen')
    weight_mat = create_weight_matrix_jit(exp_grid[0], exp_grid[1], queen)
    connectivity_label = 'queen (8-connected)' if queen else 'rook (4-connected)'
    print(f"Created weight matrix for {exp_grid[0]}x{exp_grid[1]} grid [{connectivity_label}]")

    # Compute centre crop indices
    row_start, row_end, col_start, col_end = compute_centre_indices(ising_grid, exp_grid)
    centre_idx = (row_start, row_end, col_start, col_end)
    print(f"Centre crop: rows {row_start}:{row_end}, cols {col_start}:{col_end}")

    # Generate tiled positions
    tile_row_starts, tile_col_starts = generate_tiled_positions(ising_grid, exp_grid)
    n_tiles = len(tile_row_starts)
    print(f"Tiled positions: {n_tiles} non-overlapping grids")

    # Load experimental data
    morans_i_exp, binarised_data_exp, recording_metadata = load_experimental_data(config)
    print(f"Loaded experimental data for conditions: {list(morans_i_exp.keys())}")

    # Load / process simulations
    ising_data, sim_ids = load_or_process_simulations(
        config, args, sim_files, weight_mat, centre_idx,
        tile_row_starts, tile_col_starts
    )

    n_valid = len(ising_data)
    print(f"\nValid simulations: {n_valid}/{n_sims}")

    # Chunk mode: save partial results and exit (skip matching/figures)
    # Save even empty chunks so merge dependency can fire (param filter may
    # leave some chunks with 0 sims when chunk size > filtered count).
    if is_chunk:
        import pickle
        chunk_file = os.path.join(
            config['output_path'],
            f"chunk_{args.sim_start}_{args.sim_end}.pkl"
        )
        os.makedirs(config['output_path'], exist_ok=True)
        with open(chunk_file, 'wb') as f:
            pickle.dump({'ising_data': ising_data, 'sim_ids': sim_ids}, f, protocol=4)
        elapsed = time.time() - start_time
        print(f"\nChunk saved: {chunk_file} ({n_valid} sims, {elapsed:.1f}s)")
        sys.exit(0)

    if n_valid == 0:
        print("ERROR: No valid simulations processed!")
        sys.exit(1)

    # Frame-dependent analysis loop
    ALL_FRAME_SELECTIONS = [
        ('prestim', 'prestim'),
        ('nostim',  'nostim'),
        ('all',     'full_trial'),
    ]

    if args.frame_label:
        frame_selections = [fs for fs in ALL_FRAME_SELECTIONS if fs[1] == args.frame_label]
        print(f"  Frame selection filter: {args.frame_label}")
    else:
        frame_selections = ALL_FRAME_SELECTIONS
        print(f"  Frame selections: all ({len(ALL_FRAME_SELECTIONS)})")

    output_files = []
    for frame_sel, frame_label in frame_selections:
        output_file = run_frame_analysis(
            frame_sel, frame_label, config, ising_data, sim_ids,
            morans_i_exp, binarised_data_exp, recording_metadata,
            args, start_time
        )
        output_files.append(output_file)

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    for of in output_files:
        print(f"Results saved to: {of}")
    if args.visualize and HAS_VISUALIZATION:
        print(f"Figures saved to: {os.path.join(config['output_path'], 'figures')}")

    if n_valid > 0:
        per_sim_time = total_time / n_valid
        n_frames_est = 2000
        moran_calls_est = n_frames_est * (1 + n_tiles)
        total_moran_calls = n_valid * moran_calls_est
        morans_per_sec = total_moran_calls / total_time
        print(f"\nPerformance summary:")
        print(f"  Simulations processed: {n_valid}")
        print(f"  Time per simulation: {per_sim_time:.2f}s")
        print(f"  Estimated Moran's I/sec: {morans_per_sec:.0f}")


if __name__ == '__main__':
    main()
