# -*- coding: utf-8 -*-
"""
Trajectory Comparison: Ising Model vs Experimental Dynamics
============================================================

Tests whether the Ising model reproduces experimental *dynamics* by
initializing from observed neural states and comparing how activity
evolves over time.

Two analysis modes:
1. Spontaneous Dynamics: Initialize from prestim frames binned by
   Moran's I terciles (low/mid/high), evolve freely, compare trajectories.
2. Stimulus-Off Decay: Initialize from peak-activity frames right after
   stimulus offset, compare decay dynamics.

Usage:
    # Scan jobs (dry run)
    python trajectory_comparison.py --output IsingTrajectories --scan

    # Run single SLURM array task (batch of 5 jobs)
    python trajectory_comparison.py -o /path/out -i $SLURM_ARRAY_TASK_ID -b 5

    # Run all jobs locally
    python trajectory_comparison.py --output IsingTrajectories --local --workers 4

    # Combine individual results into TrajectoryResults.h5
    python trajectory_comparison.py --output IsingTrajectories --combine
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
import argparse
import time
import gc
from glob import glob
from multiprocessing import Pool, cpu_count
from numba import njit
import h5py


# =============================================================================
# Import from existing modules
# =============================================================================

from morans_i_optimized import (
    create_weight_matrix_jit,
    morans_i_single_jit,
    compute_centre_indices,
)


# =============================================================================
# Constants
# =============================================================================

L, M = 39, 78                     # Ising grid
EXP_ROWS, EXP_COLS = 13, 26       # Experimental grid
N_REPLICATES = 30                  # Stochastic replicates per initial frame
N_FRAMES_PER_BIN = 5              # Frames per MI tercile (spontaneous mode)
N_STIM_OFF_TRIALS = 5             # Trials to select (stimulus-off mode)
N_EVOLUTION = 30                   # Trajectory length in experimental frames
BURN_IN_MIN = 2000                 # Minimum thermalization sweeps
BURN_IN_TAU_MULT = 7              # Burn-in = max(BURN_IN_MIN, tau * MULT)
CLAMPED_MIN = 50                   # Minimum clamped burn-in
CLAMPED_TAU_MULT = 3
CONDITIONS = ['Naive', 'Beginner', 'Expert', 'NoSpout']

# Centre crop indices (computed once)
CENTRE_RS, CENTRE_RE, CENTRE_CS, CENTRE_CE = compute_centre_indices(
    (L, M), (EXP_ROWS, EXP_COLS)
)


# =============================================================================
# Core Simulation Functions (duplicated from generate_all_ising_simulations.py)
# =============================================================================

def build_diamond_kernel(rad, queen=False):
    """
    Construct the diamond-shaped inhibition kernel.

    EXCLUDES the center (0,0) and nearest neighbors so that NN
    contribute only to excitation (J=+1) and more distant neurons
    contribute only to inhibition.  Rook excludes 4 NN; queen excludes 8 NN.
    """
    kernel = np.zeros((2 * rad + 1, 2 * rad + 1), dtype=np.float64)
    center = rad
    for i in range(rad + 1):
        kernel[i, rad - i:(rad + i + 1)] = 1
        kernel[-i - 1, rad - i:(rad + i + 1)] = 1

    # Exclude center neuron
    kernel[center, center] = 0

    # Exclude 4 nearest neighbors (rook)
    kernel[center - 1, center] = 0
    kernel[center + 1, center] = 0
    kernel[center, center - 1] = 0
    kernel[center, center + 1] = 0

    if queen:
        kernel[center - 1, center - 1] = 0
        kernel[center - 1, center + 1] = 0
        kernel[center + 1, center - 1] = 0
        kernel[center + 1, center + 1] = 0

    return kernel


@njit(cache=True)
def heat_bath_numba(config, beta, c, decay_const, H, K, bias_val, K_sum, queen):
    """
    Numba-optimized Monte Carlo sweep using heat-bath algorithm.

    Parameters
    ----------
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
    queen : bool
        If True, use 8-connectivity (queen); if False, 4-connectivity (rook)

    Returns
    -------
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
        if queen:
            h += (config[(i + 1) % L, (j + 1) % M] +
                  config[(i + 1) % L, (j - 1 + M) % M] +
                  config[(i - 1 + L) % L, (j + 1) % M] +
                  config[(i - 1 + L) % L, (j - 1 + M) % M])

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


# =============================================================================
# Data Loading
# =============================================================================

def load_best_match_info(comparison_path, conditions):
    """
    Load top-1 best match params + tau + scale_factor per condition
    directly from comparison HDF5 file.

    Parameters
    ----------
    comparison_path : str
        Path to IsingComparison_Results_*.mat (HDF5 v7.3)
    conditions : list of str
        Condition names to load

    Returns
    -------
    dict[condition] -> {
        'params': {beta, c, decay_const, inhibition_range, bias},
        'tau_ising': float,
        'temporal_scale_factor': float
    }
    """
    best_matches = {}

    with h5py.File(comparison_path, 'r') as f:
        ising_params = f['IsingData']['params']
        beta_vals = ising_params['beta'][()].flatten()
        c_vals = ising_params['c'][()].flatten()
        decay_vals = ising_params['decay_const'][()].flatten()
        inhib_vals = ising_params['inhibition_range'][()].flatten()
        bias_vals = ising_params['bias'][()].flatten()

        tau_all = f['IsingData']['Autocorr_tau'][()].flatten()

        comparison = f['Comparison']

        for condition in conditions:
            if condition not in comparison:
                print(f"  Warning: {condition} not found in comparison results")
                continue

            cond_data = comparison[condition]

            # Get top-1 best match index (1-indexed in MATLAB -> 0-indexed)
            best_idx_raw = cond_data['bestMatch_idx'][()].flatten()
            best_idx = int(best_idx_raw[0]) - 1

            params = {
                'beta': float(beta_vals[best_idx]),
                'c': float(c_vals[best_idx]),
                'decay_const': float(decay_vals[best_idx]),
                'inhibition_range': int(inhib_vals[best_idx]),
                'bias': float(bias_vals[best_idx]),
            }

            tau_ising = float(tau_all[best_idx])

            # Get temporal scale factor
            scale_factors = cond_data['temporal_scale_factors'][()].flatten()
            scale_factor = float(scale_factors[best_idx])

            best_matches[condition] = {
                'params': params,
                'tau_ising': tau_ising,
                'temporal_scale_factor': scale_factor,
            }

            print(f"  {condition}: beta={params['beta']:.3f}, c={params['c']:.1f}, "
                  f"tau={tau_ising:.2f}, scale={scale_factor:.3f}")

    return best_matches


def load_experimental_data(exp_data_path, conditions):
    """
    Load MoransI + BinarisedData from ExperimentalData.mat (HDF5 v7.3).

    Parameters
    ----------
    exp_data_path : str
        Path to ExperimentalData.mat
    conditions : list of str
        Condition names to load

    Returns
    -------
    morans_i : dict[cond] -> (nTrials, 185)    [.T applied]
    binarised_data : dict[cond] -> (nTrials, 185, 26, 13)  [no transpose]
    """
    morans_i = {}
    binarised_data = {}

    with h5py.File(exp_data_path, 'r') as f:
        if 'MoransI' in f:
            mi_group = f['MoransI']
            for condition in conditions:
                if condition in mi_group:
                    morans_i[condition] = np.array(mi_group[condition]).T

        if 'BinarisedData' in f:
            bd_group = f['BinarisedData']
            for condition in conditions:
                if condition in bd_group:
                    binarised_data[condition] = np.array(bd_group[condition])

    return morans_i, binarised_data


def find_comparison_results(search_dir):
    """Find the most recent comparison results file."""
    search_patterns = [
        os.path.join(search_dir, 'IsingComparison', 'IsingComparison_Results_*.mat'),
        os.path.join(search_dir, '..', 'IsingComparison', 'IsingComparison_Results_*.mat'),
        os.path.join(search_dir, 'IsingComparison_Results_*.mat'),
    ]

    for pattern in search_patterns:
        files = glob(pattern)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            return files[0]

    raise FileNotFoundError(
        f"Could not find comparison results. Searched patterns:\n" +
        "\n".join(search_patterns)
    )


# =============================================================================
# Frame Selection
# =============================================================================

def select_spontaneous_frames(morans_i, binarised_data, n_per_bin=N_FRAMES_PER_BIN,
                               n_evolution=N_EVOLUTION):
    """
    Select frames from prestim window, binned by Moran's I terciles.

    Parameters
    ----------
    morans_i : (nTrials, 185) -- MI per trial x time
    binarised_data : (nTrials, 185, 26, 13) -- raw binary data
    n_per_bin : int
        Frames per tercile
    n_evolution : int
        Trajectory length (limits max start frame)

    Returns
    -------
    dict with 'low'/'mid'/'high' keys, each containing:
        'frames': list of (13, 26) arrays (rows, cols)
        'mi_values': list of floats
        'trial_frame_idx': list of (trial, frame) tuples
        'exp_trajectories': list of (n_evolution, 13, 26) arrays
    """
    n_trials = morans_i.shape[0]
    prestim_frames = 80  # frames 0-79
    max_start = prestim_frames - n_evolution  # frame 50

    # Pool all MI values from valid prestim frames
    mi_pool = []
    idx_pool = []
    for trial in range(n_trials):
        for frame in range(max_start):
            mi_val = morans_i[trial, frame]
            if not np.isnan(mi_val):
                mi_pool.append(mi_val)
                idx_pool.append((trial, frame))

    mi_pool = np.array(mi_pool)

    # Compute tercile boundaries
    p33 = np.percentile(mi_pool, 33.33)
    p67 = np.percentile(mi_pool, 66.67)

    # Assign to bins
    bins = {'low': [], 'mid': [], 'high': []}
    for i, mi_val in enumerate(mi_pool):
        if mi_val <= p33:
            bins['low'].append(i)
        elif mi_val <= p67:
            bins['mid'].append(i)
        else:
            bins['high'].append(i)

    # Sample from each bin
    rng = np.random.RandomState(42)
    result = {}
    for bin_name, indices in bins.items():
        if len(indices) < n_per_bin:
            selected = indices
        else:
            selected = rng.choice(indices, size=n_per_bin, replace=False)

        frames = []
        mi_values = []
        trial_frame_idx = []
        exp_trajectories = []

        for idx in selected:
            trial, frame = idx_pool[idx]
            # Extract frame: bin_data[trial, frame, :, :].T -> (13, 26)
            exp_frame = binarised_data[trial, frame, :, :].T
            frames.append(exp_frame.astype(np.float64))
            mi_values.append(float(mi_pool[idx]))
            trial_frame_idx.append((int(trial), int(frame)))

            # Extract trajectory from this start frame
            traj_end = min(frame + n_evolution, binarised_data.shape[1])
            traj_len = traj_end - frame
            traj = np.zeros((n_evolution, EXP_ROWS, EXP_COLS), dtype=np.float64)
            for t in range(traj_len):
                traj[t] = binarised_data[trial, frame + t, :, :].T
            # Pad with last frame if trajectory is shorter than n_evolution
            for t in range(traj_len, n_evolution):
                traj[t] = traj[traj_len - 1]
            exp_trajectories.append(traj)

        result[bin_name] = {
            'frames': frames,
            'mi_values': mi_values,
            'trial_frame_idx': trial_frame_idx,
            'exp_trajectories': exp_trajectories,
        }

    return result


def select_stimulus_off_frames(morans_i, binarised_data, n_trials=N_STIM_OFF_TRIALS,
                                n_evolution=N_EVOLUTION):
    """
    Select peak-activity frames from stimulus offset window.

    Parameters
    ----------
    morans_i : (nTrials, 185) -- MI per trial x time
    binarised_data : (nTrials, 185, 26, 13) -- raw binary data
    n_trials : int
        Number of top trials to select
    n_evolution : int
        Trajectory length

    Returns
    -------
    dict with 'frames', 'mi_values', 'trial_frame_idx',
    'exp_trajectories', 'peak_activities'
    """
    n_total_trials = binarised_data.shape[0]
    n_time = binarised_data.shape[1]

    # Stimulus offset window: frames 100-105 (Python 0-indexed)
    stim_off_start = 100
    stim_off_end = min(106, n_time)

    # Per trial: find frame with peak mean activity in offset window
    peak_activities = []
    peak_frames = []
    for trial in range(n_total_trials):
        window = binarised_data[trial, stim_off_start:stim_off_end, :, :]
        mean_act = np.mean(window, axis=(1, 2))  # mean over spatial dims per frame
        peak_idx = np.argmax(mean_act)
        peak_frame = stim_off_start + peak_idx
        peak_activities.append(float(mean_act[peak_idx]))
        peak_frames.append(peak_frame)

    # Rank trials by peak activity, take top n_trials
    ranking = np.argsort(peak_activities)[::-1]
    selected_trials = ranking[:min(n_trials, len(ranking))]

    frames = []
    mi_values = []
    trial_frame_idx = []
    exp_trajectories = []
    selected_peak_activities = []

    for trial in selected_trials:
        frame = peak_frames[trial]
        exp_frame = binarised_data[trial, frame, :, :].T  # (13, 26)
        frames.append(exp_frame.astype(np.float64))
        mi_values.append(float(morans_i[trial, frame]))
        trial_frame_idx.append((int(trial), int(frame)))
        selected_peak_activities.append(peak_activities[trial])

        # Extract trajectory
        traj_end = min(frame + n_evolution, n_time)
        traj_len = traj_end - frame
        traj = np.zeros((n_evolution, EXP_ROWS, EXP_COLS), dtype=np.float64)
        for t in range(traj_len):
            traj[t] = binarised_data[trial, frame + t, :, :].T
        for t in range(traj_len, n_evolution):
            traj[t] = traj[traj_len - 1]
        exp_trajectories.append(traj)

    return {
        'frames': frames,
        'mi_values': mi_values,
        'trial_frame_idx': trial_frame_idx,
        'exp_trajectories': exp_trajectories,
        'peak_activities': selected_peak_activities,
    }


# =============================================================================
# Trajectory Simulation
# =============================================================================

def run_trajectory_replicate(exp_frame, params, tau_ising, scale_factor,
                              weight_mat, seed, n_evolution=N_EVOLUTION, queen=False):
    """
    One replicate: thermalize -> inject -> clamp -> release -> record.

    Parameters
    ----------
    exp_frame : (13, 26) binary {0,1}
    params : dict with beta, c, decay_const, inhibition_range, bias
    tau_ising : float (autocorrelation time in MC sweeps)
    scale_factor : float (tau_exp / tau_ising)
    weight_mat : (338, 338) for Moran's I on centre crop
    seed : int for np.random.seed
    n_evolution : int
        Number of experimental frames in trajectory

    Returns
    -------
    dict with:
        'mi': (n_sweeps,) float64
        'activity': (n_sweeps,) float64
        'spatial_corr': (n_sweeps,) float64
        'snapshots': (n_sweeps, 13, 26) int8
    """
    n_sweeps = min(200, max(n_evolution, int(np.ceil(n_evolution / max(scale_factor, 1e-6)))))

    np.random.seed(seed)

    # 1. Random init
    config = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, M))
    H = np.zeros((L, M), dtype=np.float64)
    kernel = build_diamond_kernel(params['inhibition_range'], queen)
    K_sum = float(np.sum(kernel))

    beta = params['beta']
    c = params['c']
    decay_const = params['decay_const']
    bias = params['bias']

    # 2. Thermalize
    burn_in = max(BURN_IN_MIN, int(BURN_IN_TAU_MULT * tau_ising))
    for _ in range(burn_in):
        config, H = heat_bath_numba(config, beta, c, decay_const, H, kernel, bias, K_sum, queen)

    # 3. Inject experimental frame into centre crop
    exp_spin = (2 * exp_frame - 1).astype(np.int8)  # {0,1} -> {-1,+1}
    rs, re = CENTRE_RS, CENTRE_RE
    cs, ce = CENTRE_CS, CENTRE_CE
    config[rs:re, cs:ce] = exp_spin

    # 4. Clamped burn-in (let surrounding spins + H field relax to boundary)
    n_clamp = max(CLAMPED_MIN, int(CLAMPED_TAU_MULT * tau_ising))
    for _ in range(n_clamp):
        config, H = heat_bath_numba(config, beta, c, decay_const, H, kernel, bias, K_sum, queen)
        config[rs:re, cs:ce] = exp_spin  # Reset centre after each sweep

    # Precompute reference for spatial correlation
    ref_flat = exp_frame.ravel().astype(np.float64)
    ref_mean = np.mean(ref_flat)
    ref_std = np.std(ref_flat)

    # 5. Free evolution - record per sweep
    mi_arr = np.zeros(n_sweeps, dtype=np.float64)
    act_arr = np.zeros(n_sweeps, dtype=np.float64)
    corr_arr = np.zeros(n_sweeps, dtype=np.float64)
    snap_arr = np.zeros((n_sweeps, EXP_ROWS, EXP_COLS), dtype=np.int8)

    for t in range(n_sweeps):
        config, H = heat_bath_numba(config, beta, c, decay_const, H, kernel, bias, K_sum, queen)

        crop = config[rs:re, cs:ce].copy()
        snap_arr[t] = crop

        # Activity: convert {-1,+1} to {0,1} and mean
        crop_binary = (crop.astype(np.float64) + 1.0) / 2.0
        act_arr[t] = np.mean(crop_binary)

        # Moran's I
        mi_arr[t] = morans_i_single_jit(crop_binary, weight_mat)

        # Spatial correlation with initial frame
        crop_flat = crop_binary.ravel()
        crop_mean = np.mean(crop_flat)
        crop_std = np.std(crop_flat)
        if ref_std > 0 and crop_std > 0:
            corr_arr[t] = np.mean((crop_flat - crop_mean) * (ref_flat - ref_mean)) / (crop_std * ref_std)
        else:
            corr_arr[t] = 0.0

    return {
        'mi': mi_arr,
        'activity': act_arr,
        'spatial_corr': corr_arr,
        'snapshots': snap_arr,
    }


# =============================================================================
# Comparison Metrics
# =============================================================================

def _exp_decay(t, a, tau):
    """Exponential decay function for fitting."""
    return a * np.exp(-t / tau)


def _fit_decay_tau(signal):
    """Fit exponential decay to a signal, return time constant."""
    t = np.arange(len(signal), dtype=np.float64)
    try:
        popt, _ = curve_fit(_exp_decay, t, signal, p0=[signal[0], len(signal) / 3.0],
                            maxfev=2000, bounds=([0, 0.1], [np.inf, np.inf]))
        return popt[1]
    except (RuntimeError, ValueError):
        return np.nan


def compute_trajectory_metrics(exp_trajectory, ising_results, scale_factor,
                                n_evolution=N_EVOLUTION):
    """
    Compare experimental vs Ising trajectories (raw + rescaled).

    Parameters
    ----------
    exp_trajectory : (n_evolution, 13, 26) binary data
    ising_results : list of N_REPLICATES dicts from run_trajectory_replicate
    scale_factor : tau_exp / tau_ising (frames per MC sweep)
    n_evolution : int

    Returns
    -------
    dict with raw + rescaled comparison metrics
    """
    # Compute experimental MI and activity per frame
    weight_mat = create_weight_matrix_jit(EXP_ROWS, EXP_COLS, False)
    exp_mi = np.zeros(n_evolution)
    exp_activity = np.zeros(n_evolution)
    for t in range(n_evolution):
        frame = exp_trajectory[t].astype(np.float64)
        exp_mi[t] = morans_i_single_jit(frame, weight_mat)
        exp_activity[t] = np.mean(frame)

    # Stack Ising replicates
    n_reps = len(ising_results)
    n_sweeps = ising_results[0]['mi'].shape[0]
    ising_mi = np.stack([r['mi'] for r in ising_results])        # (n_reps, n_sweeps)
    ising_act = np.stack([r['activity'] for r in ising_results])  # (n_reps, n_sweeps)

    metrics = {}

    # --- Raw comparison: sweep t vs frame t (1:1) ---
    n_compare_raw = min(n_sweeps, n_evolution)
    ising_mi_mean = np.mean(ising_mi[:, :n_compare_raw], axis=0)
    ising_mi_std = np.std(ising_mi[:, :n_compare_raw], axis=0)
    ising_act_mean = np.mean(ising_act[:, :n_compare_raw], axis=0)

    raw_mi_rmse = np.sqrt(np.nanmean((ising_mi_mean - exp_mi[:n_compare_raw]) ** 2))
    raw_act_rmse = np.sqrt(np.nanmean((ising_act_mean - exp_activity[:n_compare_raw]) ** 2))

    # Envelope fraction: fraction of exp MI within Ising mean +/- 1 SD
    in_envelope = 0
    for t in range(n_compare_raw):
        if not np.isnan(exp_mi[t]):
            if (ising_mi_mean[t] - ising_mi_std[t]) <= exp_mi[t] <= (ising_mi_mean[t] + ising_mi_std[t]):
                in_envelope += 1
    raw_envelope_frac = in_envelope / n_compare_raw if n_compare_raw > 0 else 0.0

    metrics['raw_mi_rmse'] = raw_mi_rmse
    metrics['raw_activity_rmse'] = raw_act_rmse
    metrics['raw_mi_envelope_frac'] = raw_envelope_frac

    # --- Rescaled comparison: Ising sweep t -> exp time t * scale_factor ---
    ising_time = np.arange(n_sweeps) * scale_factor  # in experimental frames
    exp_time = np.arange(n_evolution, dtype=np.float64)

    # Interpolate Ising MI/activity onto experimental frame indices
    if ising_time[-1] >= exp_time[-1]:
        # Ising covers enough time
        ising_mi_mean_all = np.mean(ising_mi, axis=0)
        ising_act_mean_all = np.mean(ising_act, axis=0)
        ising_mi_std_all = np.std(ising_mi, axis=0)

        interp_mi = interp1d(ising_time, ising_mi_mean_all, kind='linear',
                              fill_value='extrapolate')
        interp_act = interp1d(ising_time, ising_act_mean_all, kind='linear',
                               fill_value='extrapolate')
        interp_mi_std = interp1d(ising_time, ising_mi_std_all, kind='linear',
                                  fill_value='extrapolate')

        rescaled_ising_mi = interp_mi(exp_time)
        rescaled_ising_act = interp_act(exp_time)
        rescaled_ising_mi_std = interp_mi_std(exp_time)

        rescaled_mi_rmse = np.sqrt(np.nanmean((rescaled_ising_mi - exp_mi) ** 2))
        rescaled_act_rmse = np.sqrt(np.nanmean((rescaled_ising_act - exp_activity) ** 2))

        # Rescaled envelope fraction
        in_env = 0
        for t in range(n_evolution):
            if not np.isnan(exp_mi[t]):
                lo = rescaled_ising_mi[t] - rescaled_ising_mi_std[t]
                hi = rescaled_ising_mi[t] + rescaled_ising_mi_std[t]
                if lo <= exp_mi[t] <= hi:
                    in_env += 1
        rescaled_envelope_frac = in_env / n_evolution
    else:
        # Ising time too short to cover full experimental range
        n_valid = int(np.floor(ising_time[-1])) + 1
        n_valid = min(n_valid, n_evolution)
        if n_valid > 1:
            ising_mi_mean_all = np.mean(ising_mi, axis=0)
            ising_act_mean_all = np.mean(ising_act, axis=0)
            interp_mi = interp1d(ising_time, ising_mi_mean_all, kind='linear',
                                  fill_value='extrapolate')
            interp_act = interp1d(ising_time, ising_act_mean_all, kind='linear',
                                   fill_value='extrapolate')
            rescaled_ising_mi = interp_mi(exp_time[:n_valid])
            rescaled_ising_act = interp_act(exp_time[:n_valid])
            rescaled_mi_rmse = np.sqrt(np.nanmean((rescaled_ising_mi - exp_mi[:n_valid]) ** 2))
            rescaled_act_rmse = np.sqrt(np.nanmean((rescaled_ising_act - exp_activity[:n_valid]) ** 2))
        else:
            rescaled_mi_rmse = np.nan
            rescaled_act_rmse = np.nan
        rescaled_envelope_frac = np.nan

    metrics['rescaled_mi_rmse'] = rescaled_mi_rmse
    metrics['rescaled_activity_rmse'] = rescaled_act_rmse
    metrics['rescaled_mi_envelope_frac'] = rescaled_envelope_frac

    # --- Decay time constants ---
    # Ising MI decay (mean across replicates)
    ising_mi_mean_full = np.mean(ising_mi, axis=0)
    if len(ising_mi_mean_full) > 3 and ising_mi_mean_full[0] > 0:
        metrics['ising_mi_decay_tau'] = _fit_decay_tau(ising_mi_mean_full)
    else:
        metrics['ising_mi_decay_tau'] = np.nan

    # Experimental MI decay
    if len(exp_mi) > 3 and exp_mi[0] > 0:
        metrics['exp_mi_decay_tau'] = _fit_decay_tau(exp_mi)
    else:
        metrics['exp_mi_decay_tau'] = np.nan

    # Ising activity decay
    ising_act_mean_full = np.mean(ising_act, axis=0)
    if len(ising_act_mean_full) > 3:
        metrics['ising_activity_decay_tau'] = _fit_decay_tau(ising_act_mean_full)
    else:
        metrics['ising_activity_decay_tau'] = np.nan

    # Experimental activity decay
    if len(exp_activity) > 3:
        metrics['exp_activity_decay_tau'] = _fit_decay_tau(exp_activity)
    else:
        metrics['exp_activity_decay_tau'] = np.nan

    return metrics


# =============================================================================
# HDF5 Save Helper (from Figure5_IsingComparison_optimized.py)
# =============================================================================

def save_to_hdf5(filename, data_dict):
    """Save nested dict to HDF5 file (MATLAB v7.3 compatible)."""
    def write_item(group, key, value):
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            for k, v in value.items():
                write_item(subgroup, k, v)
        elif isinstance(value, (list, tuple)):
            try:
                arr = np.array(value)
                if arr.dtype == object:
                    vlen_dt = h5py.special_dtype(vlen=np.float64)
                    ds = group.create_dataset(key, (len(value),), dtype=vlen_dt)
                    for i, item in enumerate(value):
                        if item is not None:
                            ds[i] = np.array(item).flatten()
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

    with h5py.File(filename, 'w') as f:
        for key, value in data_dict.items():
            write_item(f, key, value)


# =============================================================================
# Worker + Job Generation
# =============================================================================

def generate_all_jobs(best_matches, frame_selections):
    """
    Generate job list: 4 conditions x 20 frames = 80 jobs.

    Each job = (condition, mode, bin_name, frame_idx, frame_info, best_match_info).

    Parameters
    ----------
    best_matches : dict from load_best_match_info
    frame_selections : dict[condition] -> {
        'spontaneous': dict from select_spontaneous_frames,
        'stimulus_off': dict from select_stimulus_off_frames
    }

    Returns
    -------
    list of job dicts
    """
    jobs = []

    for condition in CONDITIONS:
        if condition not in best_matches or condition not in frame_selections:
            continue

        match_info = best_matches[condition]
        sel = frame_selections[condition]

        # Spontaneous: 3 bins x 5 frames = 15 jobs
        spontaneous = sel['spontaneous']
        for bin_name in ['low', 'mid', 'high']:
            bin_data = spontaneous[bin_name]
            for i in range(len(bin_data['frames'])):
                jobs.append({
                    'condition': condition,
                    'mode': f'spontaneous_{bin_name}',
                    'bin_name': bin_name,
                    'frame_idx': i,
                    'frame': bin_data['frames'][i],
                    'mi_value': bin_data['mi_values'][i],
                    'trial_frame': bin_data['trial_frame_idx'][i],
                    'exp_trajectory': bin_data['exp_trajectories'][i],
                    'params': match_info['params'],
                    'tau_ising': match_info['tau_ising'],
                    'scale_factor': match_info['temporal_scale_factor'],
                })

        # Stimulus-off: 5 jobs
        stim_off = sel['stimulus_off']
        for i in range(len(stim_off['frames'])):
            jobs.append({
                'condition': condition,
                'mode': 'stimulus_off',
                'bin_name': 'stimulus_off',
                'frame_idx': i,
                'frame': stim_off['frames'][i],
                'mi_value': stim_off['mi_values'][i],
                'trial_frame': stim_off['trial_frame_idx'][i],
                'exp_trajectory': stim_off['exp_trajectories'][i],
                'params': match_info['params'],
                'tau_ising': match_info['tau_ising'],
                'scale_factor': match_info['temporal_scale_factor'],
            })

    return jobs


def run_trajectory_job(job, weight_mat, queen=False):
    """
    Worker: one condition x one initial frame -> N_REPLICATES replicates.

    Parameters
    ----------
    job : dict from generate_all_jobs
    weight_mat : (338, 338) weight matrix
    queen : bool
        If True, use 8-connectivity (queen); if False, 4-connectivity (rook)

    Returns
    -------
    dict with all results for saving
    """
    condition = job['condition']
    mode = job['mode']
    frame_idx = job['frame_idx']
    exp_frame = job['frame']
    params = job['params']
    tau_ising = job['tau_ising']
    scale_factor = job['scale_factor']
    exp_trajectory = job['exp_trajectory']

    n_sweeps = min(200, max(N_EVOLUTION,
                            int(np.ceil(N_EVOLUTION / max(scale_factor, 1e-6)))))

    # Run replicates
    ising_results = []
    for rep in range(N_REPLICATES):
        seed = hash((condition, mode, frame_idx, rep)) % (2**31)
        result = run_trajectory_replicate(
            exp_frame, params, tau_ising, scale_factor,
            weight_mat, seed, N_EVOLUTION, queen
        )
        ising_results.append(result)

    # Compute comparison metrics
    metrics = compute_trajectory_metrics(exp_trajectory, ising_results, scale_factor)

    # Stack replicate arrays
    ising_mi = np.stack([r['mi'] for r in ising_results])
    ising_activity = np.stack([r['activity'] for r in ising_results])
    ising_spatial_corr = np.stack([r['spatial_corr'] for r in ising_results])

    # Compute experimental MI and activity
    exp_mi = np.zeros(N_EVOLUTION)
    exp_activity = np.zeros(N_EVOLUTION)
    for t in range(N_EVOLUTION):
        frame = exp_trajectory[t].astype(np.float64)
        exp_mi[t] = morans_i_single_jit(frame, weight_mat)
        exp_activity[t] = np.mean(frame)

    return {
        'condition': condition,
        'mode': mode,
        'frame_idx': frame_idx,
        'trial_idx': job['trial_frame'][0],
        'source_frame_idx': job['trial_frame'][1],
        'initial_mi': job['mi_value'],
        'initial_frame': exp_frame.astype(np.int8),
        'params': params,
        'tau_ising': tau_ising,
        'scale_factor': scale_factor,
        'n_sweeps': n_sweeps,
        'ising_mi': ising_mi,
        'ising_activity': ising_activity,
        'ising_spatial_corr': ising_spatial_corr,
        'exp_mi': exp_mi,
        'exp_activity': exp_activity,
        'metrics': metrics,
    }


def _run_job_wrapper(args):
    """Wrapper for multiprocessing (unpacks args)."""
    if len(args) == 3:
        job, weight_mat, queen = args
    else:
        job, weight_mat = args
        queen = False
    return run_trajectory_job(job, weight_mat, queen)


# =============================================================================
# Combine Results
# =============================================================================

def combine_results(output_dir):
    """Aggregate individual .npz result files into a single TrajectoryResults.h5."""
    pattern = os.path.join(output_dir, 'traj_*.npz')
    result_files = sorted(glob(pattern))

    if not result_files:
        print(f"No result files found matching: {pattern}")
        return

    print(f"Found {len(result_files)} result files")

    # Build HDF5 structure
    output_data = {
        'config': {
            'L': L, 'M': M,
            'EXP_ROWS': EXP_ROWS, 'EXP_COLS': EXP_COLS,
            'N_REPLICATES': N_REPLICATES,
            'N_EVOLUTION': N_EVOLUTION,
            'BURN_IN_MIN': BURN_IN_MIN,
            'BURN_IN_TAU_MULT': BURN_IN_TAU_MULT,
            'CLAMPED_MIN': CLAMPED_MIN,
            'CLAMPED_TAU_MULT': CLAMPED_TAU_MULT,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    loaded = 0
    for fpath in result_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            condition = str(data['condition'])
            mode = str(data['mode'])
            frame_idx = int(data['frame_idx'])

            # Determine path in HDF5
            if mode.startswith('spontaneous_'):
                bin_name = mode.replace('spontaneous_', '')
                group_path = f'{condition}/spontaneous/{bin_name}/frame_{frame_idx}'
            else:
                group_path = f'{condition}/stimulus_off/frame_{frame_idx}'

            # Build nested dict from group path
            parts = group_path.split('/')
            current = output_data
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            frame_data = {
                'initial_frame': data['initial_frame'],
                'initial_mi': float(data['initial_mi']),
                'trial_idx': int(data['trial_idx']),
                'source_frame_idx': int(data['source_frame_idx']),
                'exp_mi': data['exp_mi'],
                'exp_activity': data['exp_activity'],
                'ising_mi': data['ising_mi'],
                'ising_activity': data['ising_activity'],
                'ising_spatial_corr': data['ising_spatial_corr'],
                'scale_factor': float(data['scale_factor']),
                'tau_ising': float(data['tau_ising']),
                'n_sweeps': int(data['n_sweeps']),
            }

            # Add metrics
            metrics_raw = data['metrics'].item() if data['metrics'].ndim == 0 else data['metrics']
            if isinstance(metrics_raw, dict):
                frame_data['metrics'] = {}
                for k, v in metrics_raw.items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        frame_data['metrics'][k] = float(v)
                    elif isinstance(v, np.ndarray):
                        frame_data['metrics'][k] = v

            current[parts[-1]] = frame_data
            loaded += 1

        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    print(f"Loaded {loaded}/{len(result_files)} files")

    # Add per-condition summaries
    for condition in CONDITIONS:
        if condition not in output_data:
            continue

        cond_data = output_data[condition]

        # Collect metrics across all frames in condition
        all_raw_mi_rmse = []
        all_rescaled_mi_rmse = []
        all_envelope_frac = []

        for mode_key in ['spontaneous', 'stimulus_off']:
            if mode_key not in cond_data:
                continue
            mode_data = cond_data[mode_key]
            if mode_key == 'spontaneous':
                for bin_name in ['low', 'mid', 'high']:
                    if bin_name not in mode_data:
                        continue
                    bin_data = mode_data[bin_name]
                    for fkey, fdata in bin_data.items():
                        if isinstance(fdata, dict) and 'metrics' in fdata:
                            m = fdata['metrics']
                            if isinstance(m, dict):
                                if 'raw_mi_rmse' in m:
                                    all_raw_mi_rmse.append(m['raw_mi_rmse'])
                                if 'rescaled_mi_rmse' in m:
                                    all_rescaled_mi_rmse.append(m['rescaled_mi_rmse'])
                                if 'raw_mi_envelope_frac' in m:
                                    all_envelope_frac.append(m['raw_mi_envelope_frac'])
            else:
                for fkey, fdata in mode_data.items():
                    if isinstance(fdata, dict) and 'metrics' in fdata:
                        m = fdata['metrics']
                        if isinstance(m, dict):
                            if 'raw_mi_rmse' in m:
                                all_raw_mi_rmse.append(m['raw_mi_rmse'])
                            if 'rescaled_mi_rmse' in m:
                                all_rescaled_mi_rmse.append(m['rescaled_mi_rmse'])
                            if 'raw_mi_envelope_frac' in m:
                                all_envelope_frac.append(m['raw_mi_envelope_frac'])

        if all_raw_mi_rmse:
            if 'condition_summary' not in cond_data:
                cond_data['condition_summary'] = {}
            cond_data['condition_summary']['mean_raw_mi_rmse'] = float(np.nanmean(all_raw_mi_rmse))
            cond_data['condition_summary']['mean_rescaled_mi_rmse'] = float(np.nanmean(all_rescaled_mi_rmse))
            cond_data['condition_summary']['mean_envelope_frac'] = float(np.nanmean(all_envelope_frac))

    # Save
    output_file = os.path.join(output_dir, 'TrajectoryResults.h5')
    print(f"Saving to: {output_file}")
    save_to_hdf5(output_file, output_data)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")
    print("Done!")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Trajectory comparison: Ising model vs experimental dynamics'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        default=r'IsingTrajectories',
        help='Output directory for results'
    )
    parser.add_argument(
        '--comparison', '-c', type=str, default=None,
        help='Path to comparison results .mat file (auto-detect if not specified)'
    )
    parser.add_argument(
        '--exp-data', '-e', type=str, default=None,
        help='Path to ExperimentalData.mat'
    )
    parser.add_argument(
        '--index', '-i', type=int, default=None,
        help='SLURM array task index'
    )
    parser.add_argument(
        '--batch-size', '-b', type=int, default=1,
        help='Jobs per array task (default: 1)'
    )
    parser.add_argument(
        '--workers', '-w', type=int, default=max(1, cpu_count() - 1),
        help='Parallel workers (local mode)'
    )
    parser.add_argument(
        '--local', action='store_true',
        help='Run all jobs locally'
    )
    parser.add_argument(
        '--scan', action='store_true',
        help='Dry run -- show job list'
    )
    parser.add_argument(
        '--combine', action='store_true',
        help='Aggregate individual results into TrajectoryResults.h5'
    )
    parser.add_argument(
        '--connectivity', type=str, choices=['rook', 'queen'], default='rook',
        help='Neighbor connectivity: rook (4-connected) or queen (8-connected)'
    )

    args = parser.parse_args()
    queen = (args.connectivity == 'queen')

    # Combine mode
    if args.combine:
        combine_results(args.output)
        return

    # Resolve paths
    if args.exp_data is None:
        # Default paths
        if os.path.exists(r'Paper\Fig. 5 Model\IsingModels\Data\ExperimentalData.mat'):
            args.exp_data = r'Paper\Fig. 5 Model\IsingModels\Data\ExperimentalData.mat'
        else:
            args.exp_data = '/path/to/data/ExperimentalData/ExperimentalData.mat'

    if args.comparison is None or args.comparison == 'auto':
        # Auto-detect comparison results
        search_dirs = [
            r'IsingModelData_39x78_100K',
            '/path/to/data/IsingSims',
            os.path.dirname(args.exp_data),
        ]
        for search_dir in search_dirs:
            if os.path.isdir(search_dir):
                try:
                    args.comparison = find_comparison_results(search_dir)
                    break
                except FileNotFoundError:
                    continue
        if args.comparison is None:
            raise FileNotFoundError("Could not auto-detect comparison results. Use --comparison.")

    # Load data
    print("=" * 60)
    print("  Trajectory Comparison: Ising Model vs Experiment")
    print("=" * 60)

    print(f"\nComparison results: {args.comparison}")
    print(f"Experimental data: {args.exp_data}")
    print(f"Output directory: {args.output}")

    print("\nLoading best match info...")
    best_matches = load_best_match_info(args.comparison, CONDITIONS)

    print("\nLoading experimental data...")
    morans_i, binarised_data = load_experimental_data(args.exp_data, CONDITIONS)

    # Frame selection
    print("\nSelecting frames...")
    frame_selections = {}
    for condition in CONDITIONS:
        if condition not in best_matches:
            continue
        if condition not in morans_i or condition not in binarised_data:
            print(f"  {condition}: Missing experimental data, skipping")
            continue

        mi = morans_i[condition]
        bd = binarised_data[condition]
        print(f"  {condition}: MI shape={mi.shape}, BinData shape={bd.shape}")

        spontaneous = select_spontaneous_frames(mi, bd)
        stim_off = select_stimulus_off_frames(mi, bd)

        n_spont = sum(len(v['frames']) for v in spontaneous.values())
        n_stim = len(stim_off['frames'])
        print(f"    Spontaneous: {n_spont} frames (low={len(spontaneous['low']['frames'])}, "
              f"mid={len(spontaneous['mid']['frames'])}, high={len(spontaneous['high']['frames'])})")
        print(f"    Stimulus-off: {n_stim} frames")

        frame_selections[condition] = {
            'spontaneous': spontaneous,
            'stimulus_off': stim_off,
        }

    # Generate jobs
    all_jobs = generate_all_jobs(best_matches, frame_selections)
    print(f"\nTotal jobs: {len(all_jobs)}")

    # Scan mode
    if args.scan:
        print("\n" + "=" * 60)
        print("  Job List (Dry Run)")
        print("=" * 60)
        for i, job in enumerate(all_jobs):
            trial, frame = job['trial_frame']
            print(f"  [{i:3d}] {job['condition']:10s} {job['mode']:20s} "
                  f"frame={job['frame_idx']} trial={trial} t={frame} MI={job['mi_value']:.4f}")
        print(f"\nTotal: {len(all_jobs)} jobs x {N_REPLICATES} replicates = "
              f"{len(all_jobs) * N_REPLICATES} simulation runs")
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create weight matrix for Moran's I
    weight_mat = create_weight_matrix_jit(EXP_ROWS, EXP_COLS, False)

    # Execution
    if args.index is not None:
        # SLURM array mode
        job_start = args.index * args.batch_size
        job_end = min(job_start + args.batch_size, len(all_jobs))

        if job_start >= len(all_jobs):
            print(f"Array index {args.index} exceeds total jobs ({len(all_jobs)}). Nothing to do.")
            return

        print(f"\nRunning jobs {job_start} to {job_end - 1}")
        batch_start = time.time()

        for job_idx in range(job_start, job_end):
            job = all_jobs[job_idx]
            print(f"\n--- Job {job_idx} ({job['condition']} {job['mode']} frame={job['frame_idx']}) ---")

            start = time.time()
            result = run_trajectory_job(job, weight_mat, queen)
            elapsed = time.time() - start
            print(f"  Completed in {elapsed:.1f}s")

            # Save individual result
            filename = f"traj_{job['condition']}_{job['mode']}_frame{job['frame_idx']}.npz"
            output_path = os.path.join(args.output, filename)
            np.savez_compressed(output_path, **result)
            print(f"  Saved: {filename}")

            del result
            gc.collect()

        total_elapsed = time.time() - batch_start
        n_done = job_end - job_start
        print(f"\nBatch complete: {n_done} jobs in {total_elapsed:.1f}s "
              f"({total_elapsed / n_done:.1f}s/job)")

    elif args.local:
        # Local parallel mode
        print(f"\nRunning all {len(all_jobs)} jobs locally with {args.workers} workers")
        start = time.time()

        job_args = [(job, weight_mat, queen) for job in all_jobs]

        if args.workers > 1:
            with Pool(args.workers) as pool:
                for i, result in enumerate(pool.imap(_run_job_wrapper, job_args)):
                    job = all_jobs[i]
                    filename = f"traj_{job['condition']}_{job['mode']}_frame{job['frame_idx']}.npz"
                    output_path = os.path.join(args.output, filename)
                    np.savez_compressed(output_path, **result)
                    print(f"  [{i + 1}/{len(all_jobs)}] {filename}")
                    del result
                    gc.collect()
        else:
            for i, (job, wm, q) in enumerate(job_args):
                result = run_trajectory_job(job, wm, q)
                filename = f"traj_{job['condition']}_{job['mode']}_frame{job['frame_idx']}.npz"
                output_path = os.path.join(args.output, filename)
                np.savez_compressed(output_path, **result)
                print(f"  [{i + 1}/{len(all_jobs)}] {filename}")
                del result
                gc.collect()

        elapsed = time.time() - start
        print(f"\nAll jobs complete: {elapsed:.1f}s ({elapsed / len(all_jobs):.1f}s/job)")

    else:
        print("\nSpecify --index (SLURM) or --local to run jobs.")
        print("Use --scan to preview the job list.")


if __name__ == '__main__':
    main()
