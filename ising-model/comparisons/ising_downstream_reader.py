# -*- coding: utf-8 -*-
"""
Ising Model Downstream Reader Classifier
=========================================

Places virtual neurons with circular receptive fields on the Ising grid,
extracts temporal features from their responses to stimulus perturbations,
and classifies whether the dynamics came from a Naive or Expert condition.

Key result: classification accuracy vs receptive field size.

Architecture:
    1. Load best-match Ising parameters from IsingComparison_Results_*.mat
    2. For each (condition, sim, stim_size) run perturbation with full grid
       snapshots, extract virtual-neuron responses at all RF radii
    3. Compute temporal, spatial, and population features per replicate
    4. Train classifiers (LogReg, RF, SVM) with GroupKFold CV
    5. Plot accuracy vs RF radius and additional analysis figures

Usage:
    # Show job plan (dry run)
    python ising_downstream_reader.py --output IsingDownstream --scan

    # Run single job by index (SLURM array)
    python ising_downstream_reader.py --output IsingDownstream --index 0

    # Run all jobs locally in parallel
    python ising_downstream_reader.py --output IsingDownstream --local --workers 8

    # Combine per-job results into dataset
    python ising_downstream_reader.py --output IsingDownstream --combine

    # Run classification + figures on combined dataset
    python ising_downstream_reader.py --output IsingDownstream --classify
"""

import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr
import os
import sys
import re
import csv
import argparse
import time
from glob import glob
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from joblib import Parallel, delayed
from numba import njit
import h5py
import gc
import tempfile
os.environ.setdefault('JOBLIB_TEMP_FOLDER', tempfile.gettempdir())

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.lines as mlines

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix
)


# =============================================================================
# Configuration
# =============================================================================

# Binary classification: Naive vs Expert only
CONDITIONS = ['Naive', 'Expert']
CONDITION_LABELS = {'Naive': 0, 'Expert': 1}
N_TOP_MATCHES = 10

# Grid dimensions (matching original simulations)
L, M = 39, 78

# Centre crop matching experimental FOV (13 rows x 26 cols)
CROP_ROW_START, CROP_ROW_END = 13, 26
CROP_COL_START, CROP_COL_END = 26, 52

# Stimulus center
CENTER_ROW = L // 2   # 19
CENTER_COL = M // 2   # 39

# Timing
PRE_STIM_FRAMES = 400
POST_STIM_FRAMES = 300
N_REPLICATES = 50

# Burn-in / decorrelation
BURN_IN_MIN = 2000
BURN_IN_TAU_MULT = 7
DECORR_TAU_MULT = 2
DECORR_MIN = 500

# Initial sweep defaults
DEFAULT_STIM_MODE = 'clamped'
DEFAULT_STIM_DURATION = 50
DEFAULT_STIM_SIZES = [1, 2, 4, 8, 12]

# Virtual neuron configuration
RF_RADII = [1, 2, 3, 4, 6, 8, 10, 15, 20]
RING_DISTANCES = [5, 10, 15, 20]
RING_ANGLES_DEG = [0, 90, 180, 270]  # right, down, left, up
N_NEURONS = 1 + len(RING_DISTANCES) * len(RING_ANGLES_DEG)  # 17

# Feature counts
N_TEMPORAL_FEATURES = 10
N_SPATIAL_FEATURES = 2
N_NEURON_FEATURES = N_TEMPORAL_FEATURES + N_SPATIAL_FEATURES  # 12
N_POPULATION_FEATURES = 4
N_TOTAL_FEATURES = N_NEURONS * N_NEURON_FEATURES + N_POPULATION_FEATURES  # 208

# Condition colours
COLOR_NAIVE = np.array([0.337, 0.706, 0.914])
COLOR_EXPERT = np.array([0.0, 0.620, 0.451])

# Model line colours
MODEL_COLORS = {
    'LogReg_L2':    '#E64B35',
    'RandomForest': '#4DBBD5',
    'LinearSVM':    '#7E6148',
    'RBF_SVM':      '#00A087',
}
MODEL_MARKERS = {
    'LogReg_L2':    'o',
    'RandomForest': 's',
    'LinearSVM':    '^',
    'RBF_SVM':      'D',
}

# Feature category colours for importance plots
FEATURE_CAT_COLORS = {
    'temporal':    '#3B7DD8',
    'spatial':     '#4CA64C',
    'population':  '#E8922E',
    'propagation': '#9B59B6',
}

FEATURE_CATEGORY_MAP = {
    'baseline_activity': 'temporal',
    'peak_response': 'temporal',
    'response_amplitude': 'temporal',
    'response_latency': 'temporal',
    'time_to_peak': 'temporal',
    'decay_half_life': 'temporal',
    'auc_stim': 'temporal',
    'auc_post': 'temporal',
    'onset_slope': 'temporal',
    'sustained_index': 'temporal',
    'distance_from_center': 'spatial',
    'angle_from_center': 'spatial',
    'spatial_gradient': 'population',
    'spread_latency_gradient': 'population',
    'population_synchrony': 'population',
    'max_responding_distance': 'population',
    'ring_mean_amp_d5': 'propagation',
    'ring_mean_amp_d10': 'propagation',
    'ring_mean_amp_d15': 'propagation',
    'ring_mean_amp_d20': 'propagation',
    'amplitude_decay_slope': 'propagation',
    'response_reach': 'propagation',
    'center_surround_ratio': 'propagation',
    'propagation_speed': 'propagation',
}


# =============================================================================
# Core Simulation Functions (duplicated from run_ising_perturbations.py)
# =============================================================================

def build_diamond_kernel(rad, queen=False):
    """
    Construct the diamond-shaped inhibition kernel.

    EXCLUDES the center (0,0) and 4 nearest neighbors (+/-1,0), (0,+/-1)
    so that NN contribute only to excitation (J=+1) and more distant
    neurons contribute only to inhibition.

    When queen=True, also excludes the 4 diagonal nearest neighbors
    (+/-1, +/-1) so that all 8-connected neighbors contribute to excitation.
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

    if queen:
        kernel[center-1, center-1] = 0  # top-left
        kernel[center-1, center+1] = 0  # top-right
        kernel[center+1, center-1] = 0  # bottom-left
        kernel[center+1, center+1] = 0  # bottom-right

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
        If True, include 4 diagonal neighbors in excitation sum.

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
def heat_bath_numba_with_bias_field(config, beta, c, decay_const, H, K, bias_field, K_sum, queen):
    """Numba-optimized Monte Carlo sweep with 2D bias field support."""
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

        if queen:
            h += (config[(i + 1) % L, (j + 1) % M] +
                  config[(i + 1) % L, (j - 1 + M) % M] +
                  config[(i - 1 + L) % L, (j + 1) % M] +
                  config[(i - 1 + L) % L, (j - 1 + M) % M])

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


def get_stimulus_region(stim_size, center_row, center_col, L, M):
    """Get row and column indices for a square stimulus region."""
    half_size = stim_size // 2

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

    row_start = max(0, row_start)
    row_end = min(L, row_end)
    col_start = max(0, col_start)
    col_end = min(M, col_end)

    return np.arange(row_start, row_end), np.arange(col_start, col_end)


def apply_stimulus(config, stim_rows, stim_cols):
    """Apply stimulus by setting specified region to +1."""
    for i in stim_rows:
        for j in stim_cols:
            config[i, j] = 1
    return config


# =============================================================================
# Snapshot Perturbation Runner
# =============================================================================

def run_snapshot_perturbation(params, stim_size, stim_duration, stim_mode,
                              initial_state, stim_bias=None,
                              kernel=None, K_sum=None, queen=False):
    """
    Run a single perturbation and return full grid snapshots.

    Returns
    -------
    snapshots : np.ndarray, shape (total_frames, L, M), dtype uint8
        Binary grid (0/1) at every frame.
    """
    beta = params['beta']
    c = params['c']
    decay_const = params['decay_const']
    rad = params['inhibition_range']
    bias = params['bias']

    if kernel is None:
        kernel = build_diamond_kernel(rad, queen=queen)
        K_sum = float(np.sum(kernel))

    spin_config = initial_state[0].copy()
    H = initial_state[1].copy()

    stim_rows, stim_cols = get_stimulus_region(stim_size, CENTER_ROW, CENTER_COL, L, M)

    total_frames = PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES
    snapshots = np.zeros((total_frames, L, M), dtype=np.uint8)

    frame_idx = 0

    # --- PRE-STIMULUS ---
    for _ in range(PRE_STIM_FRAMES):
        spin_config, H = heat_bath_numba(
            spin_config, beta, c, decay_const, H, kernel, bias, K_sum, queen)
        snapshots[frame_idx] = ((spin_config + 1) // 2).astype(np.uint8)
        frame_idx += 1

    # --- STIMULUS ON ---
    for t in range(stim_duration):
        if stim_mode == 'clamped':
            spin_config = apply_stimulus(spin_config, stim_rows, stim_cols)
            spin_config, H = heat_bath_numba(
                spin_config, beta, c, decay_const, H, kernel, bias, K_sum, queen)
            spin_config = apply_stimulus(spin_config, stim_rows, stim_cols)
        elif stim_mode.startswith('double_pulse'):
            dp_suffix = stim_mode[len('double_pulse'):]
            clamp_width = int(dp_suffix) if dp_suffix else 1
            if t < clamp_width or t >= stim_duration - clamp_width:
                spin_config = apply_stimulus(spin_config, stim_rows, stim_cols)
            spin_config, H = heat_bath_numba(
                spin_config, beta, c, decay_const, H, kernel, bias, K_sum, queen)
        elif stim_mode == 'bias':
            bias_field = np.ones((L, M), dtype=np.float64) * bias
            for i in stim_rows:
                for j in stim_cols:
                    bias_field[i, j] += stim_bias
            spin_config, H = heat_bath_numba_with_bias_field(
                spin_config, beta, c, decay_const, H, kernel, bias_field, K_sum, queen)

        snapshots[frame_idx] = ((spin_config + 1) // 2).astype(np.uint8)
        frame_idx += 1

    # --- POST-STIMULUS ---
    for _ in range(POST_STIM_FRAMES):
        spin_config, H = heat_bath_numba(
            spin_config, beta, c, decay_const, H, kernel, bias, K_sum, queen)
        snapshots[frame_idx] = ((spin_config + 1) // 2).astype(np.uint8)
        frame_idx += 1

    return snapshots


def run_free_evolution_snapshots(params, total_frames, initial_state,
                                 kernel=None, K_sum=None, queen=False):
    """
    Run free evolution (no stimulus) and return full grid snapshots.

    Parameters
    ----------
    params : dict
        Simulation parameters (beta, c, decay_const, inhibition_range, bias)
    total_frames : int
        Number of frames to simulate
    initial_state : tuple of (spin_config, H)
        Initial equilibrium state
    kernel : ndarray, optional
        Pre-built diamond kernel (avoids rebuilding)
    K_sum : float, optional
        Sum of kernel elements
    queen : bool
        If True, use 8-connected (queen) neighbors instead of 4-connected (rook).

    Returns
    -------
    snapshots : np.ndarray, shape (total_frames, L, M), dtype uint8
        Binary grid (0/1) at every frame.
    """
    beta = params['beta']
    c = params['c']
    decay_const = params['decay_const']
    rad = params['inhibition_range']
    bias = params['bias']

    if kernel is None:
        kernel = build_diamond_kernel(rad, queen=queen)
        K_sum = float(np.sum(kernel))

    spin_config = initial_state[0].copy()
    H = initial_state[1].copy()

    snapshots = np.zeros((total_frames, L, M), dtype=np.uint8)

    for frame_idx in range(total_frames):
        spin_config, H = heat_bath_numba(
            spin_config, beta, c, decay_const, H, kernel, bias, K_sum, queen)
        snapshots[frame_idx] = ((spin_config + 1) // 2).astype(np.uint8)

    return snapshots


# =============================================================================
# Virtual Neuron Model
# =============================================================================

class VirtualNeuron:
    """
    A virtual neuron with a circular receptive field on a periodic grid.

    The neuron computes its response as the mean activity (fraction of 1s)
    within its RF mask at each time frame.
    """

    def __init__(self, row, col, rf_radius, grid_L, grid_M):
        self.row = row
        self.col = col
        self.rf_radius = rf_radius
        self.grid_L = grid_L
        self.grid_M = grid_M
        self._pixel_rows, self._pixel_cols = self._build_circular_mask()
        self.n_pixels = len(self._pixel_rows)

    def _build_circular_mask(self):
        """Build arrays of absolute (row, col) coordinates within circular RF."""
        r = self.rf_radius
        rows = []
        cols = []
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if dr * dr + dc * dc <= r * r:
                    abs_row = (self.row + dr) % self.grid_L
                    abs_col = (self.col + dc) % self.grid_M
                    rows.append(abs_row)
                    cols.append(abs_col)
        return np.array(rows, dtype=np.intp), np.array(cols, dtype=np.intp)

    def response_timeseries(self, grid_snapshots):
        """
        Compute mean activity within RF mask for each frame.

        Parameters
        ----------
        grid_snapshots : np.ndarray, shape (n_frames, L, M), dtype uint8

        Returns
        -------
        np.ndarray, shape (n_frames,), dtype float32
        """
        rf_values = grid_snapshots[:, self._pixel_rows, self._pixel_cols]
        return rf_values.mean(axis=1).astype(np.float32)


def build_neuron_population(rf_radius):
    """
    Create 17 VirtualNeurons: 1 at stimulus center + 4 rings x 4 directions.

    Layout:
    - Center: (19, 39) = (L//2, M//2)
    - Ring distances: [5, 10, 15, 20] pixels from center
    - Directions: 0 (right), 90 (down), 180 (left), 270 (up) degrees

    Returns
    -------
    neurons : list of VirtualNeuron
    metadata : list of dict with keys: label, distance, angle
    """
    neurons = []
    metadata = []

    # Center neuron
    neurons.append(VirtualNeuron(CENTER_ROW, CENTER_COL, rf_radius, L, M))
    metadata.append({'label': 'center', 'distance': 0, 'angle': 0})

    # Ring neurons
    for dist in RING_DISTANCES:
        for angle_deg in RING_ANGLES_DEG:
            angle_rad = np.radians(angle_deg)
            dr = int(round(dist * np.sin(angle_rad)))
            dc = int(round(dist * np.cos(angle_rad)))
            row = (CENTER_ROW + dr) % L
            col = (CENTER_COL + dc) % M
            neurons.append(VirtualNeuron(row, col, rf_radius, L, M))
            metadata.append({
                'label': f'd{dist}_a{angle_deg}',
                'distance': dist,
                'angle': angle_deg
            })

    return neurons, metadata


# =============================================================================
# Simulation Orchestrator
# =============================================================================

def run_downstream_reader_job(params, stim_size, stim_duration, stim_mode,
                               stim_bias=None, seed=None, queen=False):
    """
    Run full downstream reader experiment for one parameter combination.

    Uses shared-equilibrium strategy: single burn-in, then decorrelated
    equilibrium samples for N_REPLICATES independent perturbation trials.
    For each trial, extracts virtual neuron responses at all RF radii.

    Returns
    -------
    dict with:
        'responses' : ndarray (n_radii, N_NEURONS, N_REPLICATES, total_frames), float32
        'neuron_metadata' : list of dict
        'rf_radii' : list of int
        'params', 'stim_size', 'stim_duration', 'stim_mode', 'stim_bias'
        'total_frames', 'pre_stim_frames', 'post_stim_frames'
    """
    beta = params['beta']
    c = params['c']
    decay_const = params['decay_const']
    rad = params['inhibition_range']
    bias_val = params['bias']

    kernel = build_diamond_kernel(rad, queen=queen)
    K_sum = float(np.sum(kernel))

    total_frames = PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES
    n_radii = len(RF_RADII)

    # Pre-build neuron populations for all radii (reused across replicates)
    populations = []
    neuron_meta = None
    for rf_radius in RF_RADII:
        neurons, meta = build_neuron_population(rf_radius)
        populations.append(neurons)
        if neuron_meta is None:
            neuron_meta = meta

    # Pre-allocate output
    responses = np.zeros((n_radii, N_NEURONS, N_REPLICATES, total_frames),
                         dtype=np.float32)

    # --- Shared equilibrium: single burn-in ---
    if seed is not None:
        np.random.seed(seed)

    spin_config = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, M))
    H = np.zeros((L, M), dtype=np.float64)

    tau = (L * M) / decay_const
    burn_in_steps = max(BURN_IN_MIN, int(BURN_IN_TAU_MULT * tau))
    for _ in range(burn_in_steps):
        spin_config, H = heat_bath_numba(
            spin_config, beta, c, decay_const, H, kernel, bias_val, K_sum, queen)

    # --- Decorrelated equilibrium samples ---
    decorr_gap = max(DECORR_MIN, int(DECORR_TAU_MULT * tau))
    eq_states = [(spin_config.copy(), H.copy())]
    for _ in range(N_REPLICATES - 1):
        for __ in range(decorr_gap):
            spin_config, H = heat_bath_numba(
                spin_config, beta, c, decay_const, H, kernel, bias_val, K_sum, queen)
        eq_states.append((spin_config.copy(), H.copy()))

    # --- Run replicates ---
    for rep in range(N_REPLICATES):
        snapshots = run_snapshot_perturbation(
            params, stim_size, stim_duration, stim_mode,
            initial_state=eq_states[rep], stim_bias=stim_bias,
            queen=queen)

        for ri, neurons in enumerate(populations):
            for ni, neuron in enumerate(neurons):
                responses[ri, ni, rep, :] = neuron.response_timeseries(snapshots)

        del snapshots

    del eq_states
    gc.collect()

    return {
        'responses': responses,
        'neuron_metadata': neuron_meta,
        'rf_radii': RF_RADII,
        'params': params,
        'stim_size': stim_size,
        'stim_duration': stim_duration,
        'stim_mode': stim_mode,
        'stim_bias': stim_bias,
        'total_frames': total_frames,
        'pre_stim_frames': PRE_STIM_FRAMES,
        'post_stim_frames': POST_STIM_FRAMES,
    }


# =============================================================================
# Data Loading (duplicated from run_ising_perturbations.py)
# =============================================================================

def load_mat_file(filepath):
    """Load a .mat file, handling both v7.3 (HDF5) and older formats."""
    try:
        return loadmat(filepath, simplify_cells=True)
    except (NotImplementedError, ValueError):
        return h5py.File(filepath, 'r')


def extract_hdf5_value(obj):
    """Extract value from HDF5 dataset or group."""
    if isinstance(obj, h5py.Dataset):
        val = obj[()]
        if val.dtype == np.uint16:
            return ''.join(chr(c) for c in val.flatten())
        return val
    elif isinstance(obj, h5py.Group):
        return {key: extract_hdf5_value(obj[key]) for key in obj.keys()}
    return obj


def load_comparison_results(comparison_path):
    """
    Load comparison results from MATLAB .mat file.
    Only loads Naive and Expert conditions (binary classification).
    """
    print(f"Loading comparison results from: {comparison_path}")

    data = load_mat_file(comparison_path)
    is_hdf5 = isinstance(data, h5py.File)

    best_matches = {}

    if is_hdf5:
        ising_params = data['IsingData']['params']
        beta_vals = ising_params['beta'][()].flatten()
        c_vals = ising_params['c'][()].flatten()
        decay_vals = ising_params['decay_const'][()].flatten()
        inhib_vals = ising_params['inhibition_range'][()].flatten()
        bias_vals = ising_params['bias'][()].flatten()

        comparison = data['Comparison']

        for condition in CONDITIONS:
            if condition not in comparison:
                print(f"  Warning: {condition} not found in comparison results")
                continue

            cond_data = comparison[condition]
            best_idx = cond_data['bestMatch_idx'][()].flatten()[:N_TOP_MATCHES]
            best_idx = [int(idx) for idx in best_idx]  # already 0-indexed (Python np.argsort)

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

            print(f"  {condition}: Loaded {len(best_matches[condition]['simulations'])} best matches")

        data.close()

    else:
        if 'Results' in data:
            results = data['Results']
        else:
            results = data

        ising_data = results['IsingData']
        comparison = results['Comparison']

        for condition in CONDITIONS:
            if condition not in comparison:
                print(f"  Warning: {condition} not found in comparison results")
                continue

            cond_data = comparison[condition]
            best_idx = cond_data['bestMatch_idx']
            if isinstance(best_idx, np.ndarray):
                best_idx = best_idx.flatten()[:N_TOP_MATCHES]
            else:
                best_idx = [best_idx]

            best_idx = [int(idx) for idx in best_idx]  # already 0-indexed (Python np.argsort)

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

            print(f"  {condition}: Loaded {len(best_matches[condition]['simulations'])} best matches")

    return best_matches


def find_comparison_results(ising_data_path):
    """Find the most recent comparison results file."""
    search_patterns = [
        os.path.join(ising_data_path, 'IsingComparison', 'IsingComparison_Results_*.mat'),
        os.path.join(ising_data_path, '..', 'IsingComparison', 'IsingComparison_Results_*.mat'),
        os.path.join(ising_data_path, 'IsingComparison_Results_*.mat'),
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
# Feature Extraction (operates on virtual neuron responses)
# =============================================================================

def extract_neuron_features(neuron_responses, stim_duration, neuron_metadata):
    """
    Extract per-neuron temporal and spatial features from virtual neuron responses.

    Parameters
    ----------
    neuron_responses : ndarray (N_NEURONS, total_frames), float32
        Mean RF activity per frame for each neuron.
    stim_duration : int
        Duration of stimulus period in frames.
    neuron_metadata : list of dict
        Per-neuron info with 'distance' and 'angle' keys.

    Returns
    -------
    features : ndarray (N_NEURONS, N_NEURON_FEATURES)
    feature_names : list of str
    """
    n_neurons = neuron_responses.shape[0]
    total_frames = neuron_responses.shape[1]
    stim_start = PRE_STIM_FRAMES
    stim_end = PRE_STIM_FRAMES + stim_duration

    pre_stim = neuron_responses[:, :stim_start]
    stim_period = neuron_responses[:, stim_start:stim_end]
    post_period = neuron_responses[:, stim_end:]
    response_period = neuron_responses[:, stim_start:]

    features = np.zeros((n_neurons, N_NEURON_FEATURES), dtype=np.float64)

    # --- Temporal features ---

    # 0: baseline_activity
    baseline = np.mean(pre_stim, axis=1)
    features[:, 0] = baseline

    # 1: peak_response
    peak_response = np.max(response_period, axis=1)
    features[:, 1] = peak_response

    # 2: response_amplitude
    features[:, 2] = peak_response - baseline

    # 3: response_latency (first frame exceeding baseline + 2*std)
    pre_std = np.std(pre_stim, axis=1)
    threshold = baseline + 2.0 * pre_std
    sentinel = float(response_period.shape[1])
    latencies = np.full(n_neurons, sentinel)
    for n in range(n_neurons):
        crossings = np.where(response_period[n, :] > threshold[n])[0]
        if len(crossings) > 0:
            latencies[n] = float(crossings[0])
    features[:, 3] = latencies

    # 4: time_to_peak (frame of peak relative to stim onset)
    features[:, 4] = np.argmax(response_period, axis=1).astype(np.float64)

    # 5: decay_half_life (frames from peak to half-peak-amplitude in post-stim)
    half_peak = peak_response / 2.0
    decay_sentinel = float(post_period.shape[1])
    half_lives = np.full(n_neurons, decay_sentinel)
    for n in range(n_neurons):
        if peak_response[n] > 0:
            below = np.where(post_period[n, :] <= half_peak[n])[0]
            if len(below) > 0:
                half_lives[n] = float(below[0])
    features[:, 5] = half_lives

    # 6: auc_stim
    features[:, 6] = np.sum(stim_period, axis=1)

    # 7: auc_post
    features[:, 7] = np.sum(post_period, axis=1)

    # 8: onset_slope (linear fit slope over first 10 stim frames)
    n_onset = min(10, stim_period.shape[1])
    onset_data = stim_period[:, :n_onset]
    t_onset = np.arange(n_onset, dtype=np.float64)
    t_mean = np.mean(t_onset)
    t_centered = t_onset - t_mean
    t_var = np.sum(t_centered ** 2)
    if t_var > 0:
        data_mean = np.mean(onset_data, axis=1, keepdims=True)
        data_centered = onset_data - data_mean
        slopes = np.dot(data_centered, t_centered) / t_var
    else:
        slopes = np.zeros(n_neurons)
    features[:, 8] = slopes

    # 9: sustained_index
    n_sustain = min(10, stim_period.shape[1])
    sustained_mean = np.mean(stim_period[:, -n_sustain:], axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sustained_idx = np.where(peak_response > 0,
                                 sustained_mean / peak_response, 0.0)
    features[:, 9] = sustained_idx

    # --- Spatial features ---
    # 10: distance_from_center
    distances = np.array([m['distance'] for m in neuron_metadata], dtype=np.float64)
    features[:, 10] = distances

    # 11: angle_from_center (in radians)
    angles = np.array([np.radians(m['angle']) for m in neuron_metadata], dtype=np.float64)
    features[:, 11] = angles

    feature_names = [
        'baseline_activity', 'peak_response', 'response_amplitude',
        'response_latency', 'time_to_peak', 'decay_half_life',
        'auc_stim', 'auc_post', 'onset_slope', 'sustained_index',
        'distance_from_center', 'angle_from_center'
    ]

    return features, feature_names


def extract_population_features(neuron_responses, neuron_features, stim_duration):
    """
    Extract population-level features across all neurons.

    Returns
    -------
    pop_features : ndarray (N_POPULATION_FEATURES,)
    pop_feature_names : list of str
    """
    stim_start = PRE_STIM_FRAMES
    response_ts = neuron_responses[:, stim_start:]

    distances = neuron_features[:, 10]
    amplitudes = neuron_features[:, 2]
    latencies = neuron_features[:, 3]

    pop = np.zeros(N_POPULATION_FEATURES, dtype=np.float64)

    # 0: spatial_gradient
    if np.std(distances) > 0 and np.std(amplitudes) > 0:
        pop[0], _ = pearsonr(distances, amplitudes)
    else:
        pop[0] = 0.0

    # 1: spread_latency_gradient
    sentinel = float(response_ts.shape[1])
    responding = latencies < sentinel
    if np.sum(responding) > 3:
        d_resp = distances[responding]
        l_resp = latencies[responding]
        if np.std(d_resp) > 0 and np.std(l_resp) > 0:
            pop[1], _ = pearsonr(d_resp, l_resp)
        else:
            pop[1] = 0.0
    else:
        pop[1] = 0.0

    # 2: population_synchrony (mean pairwise correlation)
    n_neurons = response_ts.shape[0]
    if n_neurons > 1:
        means = np.mean(response_ts, axis=1, keepdims=True)
        stds = np.std(response_ts, axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        normed = (response_ts - means) / stds
        corr_mat = np.dot(normed, normed.T) / normed.shape[1]
        triu_idx = np.triu_indices(n_neurons, k=1)
        pop[2] = np.mean(corr_mat[triu_idx])
    else:
        pop[2] = 0.0

    # 3: max_responding_distance
    pop[3] = distances[np.argmax(amplitudes)]

    pop_feature_names = [
        'spatial_gradient', 'spread_latency_gradient',
        'population_synchrony', 'max_responding_distance'
    ]

    return pop, pop_feature_names


def build_feature_vector(neuron_responses, stim_duration, neuron_metadata):
    """
    Build a single flat feature vector from one replicate's neuron responses.

    Parameters
    ----------
    neuron_responses : ndarray (N_NEURONS, total_frames)
    stim_duration : int
    neuron_metadata : list of dict

    Returns
    -------
    feature_vec : ndarray (N_TOTAL_FEATURES,)
    feature_names : list of str
    """
    neuron_feats, neuron_names = extract_neuron_features(
        neuron_responses, stim_duration, neuron_metadata)
    pop_feats, pop_names = extract_population_features(
        neuron_responses, neuron_feats, stim_duration)

    flat_neuron = neuron_feats.flatten()
    feature_vec = np.concatenate([flat_neuron, pop_feats])

    # Build feature names with neuron labels
    all_names = []
    for n_idx, meta in enumerate(neuron_metadata):
        for fname in neuron_names:
            all_names.append(f'{meta["label"]}_{fname}')
    all_names.extend(pop_names)

    return feature_vec, all_names


def remove_spatial_features(X, feature_names):
    """
    Remove static spatial features (distance_from_center, angle_from_center)
    that are identical across Naive and Expert and cannot discriminate.

    Returns cleaned X and feature_names with those columns dropped.
    """
    keep_mask = np.ones(len(feature_names), dtype=bool)
    for i, name in enumerate(feature_names):
        if name.endswith('_distance_from_center') or name.endswith('_angle_from_center'):
            keep_mask[i] = False
    X_clean = X[:, keep_mask]
    fn_clean = [n for n, k in zip(feature_names, keep_mask) if k]
    n_dropped = int(np.sum(~keep_mask))
    if n_dropped > 0:
        print(f"  Removed {n_dropped} static spatial features "
              f"({X.shape[1]} -> {X_clean.shape[1]})")
    return X_clean, fn_clean


def augment_propagation_features(X, feature_names):
    """
    Derive 8 propagation features from existing columns:
      - ring_mean_amp_d5/d10/d15/d20 (4): mean response_amplitude per distance ring
      - amplitude_decay_slope (1): slope of amplitude vs distance
      - response_reach (1): max distance where mean amp > 10% of center
      - center_surround_ratio (1): center_amp / mean(ring amps)
      - propagation_speed (1): slope of latency vs distance
    """
    n_samples = X.shape[0]
    fn_lookup = {name: i for i, name in enumerate(feature_names)}

    # Get center amplitude
    center_amp_idx = fn_lookup.get('center_response_amplitude')
    if center_amp_idx is None:
        print("  Warning: center_response_amplitude not found, skipping propagation features")
        return X, feature_names
    center_amp = X[:, center_amp_idx]

    # Compute ring mean amplitudes for each distance
    ring_distances = [5, 10, 15, 20]
    ring_angles = [0, 90, 180, 270]
    ring_means = np.zeros((n_samples, len(ring_distances)))

    for di, dist in enumerate(ring_distances):
        amp_indices = []
        for angle in ring_angles:
            key = f'd{dist}_a{angle}_response_amplitude'
            if key in fn_lookup:
                amp_indices.append(fn_lookup[key])
        if amp_indices:
            ring_means[:, di] = np.mean(X[:, amp_indices], axis=1)

    # 1-4: ring_mean_amp_d5/d10/d15/d20
    new_features = [ring_means[:, i] for i in range(len(ring_distances))]
    new_names = [f'ring_mean_amp_d{d}' for d in ring_distances]

    # 5: amplitude_decay_slope — slope of [center, ring5, ring10, ring15, ring20] vs distance
    distances = np.array([0.0] + [float(d) for d in ring_distances])
    d_mean = np.mean(distances)
    d_centered = distances - d_mean
    d_var = np.sum(d_centered ** 2)
    all_amps = np.column_stack([center_amp.reshape(-1, 1), ring_means])
    amp_mean = np.mean(all_amps, axis=1, keepdims=True)
    amp_centered = all_amps - amp_mean
    decay_slope = np.dot(amp_centered, d_centered) / max(d_var, 1e-12)
    new_features.append(decay_slope)
    new_names.append('amplitude_decay_slope')

    # 6: response_reach — max distance where ring mean amp > 10% of center amp
    threshold_amp = np.abs(center_amp) * 0.1
    reach = np.zeros(n_samples)
    for di, dist in enumerate(ring_distances):
        above = ring_means[:, di] > threshold_amp
        reach[above] = float(dist)
    new_features.append(reach)
    new_names.append('response_reach')

    # 7: center_surround_ratio — center_amp / mean(all ring amps + epsilon)
    mean_ring = np.mean(ring_means, axis=1)
    cs_ratio = center_amp / (mean_ring + 1e-8)
    new_features.append(cs_ratio)
    new_names.append('center_surround_ratio')

    # 8: propagation_speed — slope of response_latency vs distance
    # Gather latencies for all 17 neurons
    lat_vals = []
    lat_dists = []
    center_lat_idx = fn_lookup.get('center_response_latency')
    if center_lat_idx is not None:
        lat_vals.append(X[:, center_lat_idx])
        lat_dists.append(0.0)
    for dist in ring_distances:
        for angle in ring_angles:
            key = f'd{dist}_a{angle}_response_latency'
            if key in fn_lookup:
                lat_vals.append(X[:, fn_lookup[key]])
                lat_dists.append(float(dist))

    if len(lat_vals) >= 2:
        lat_array = np.column_stack(lat_vals)  # (n_samples, n_neurons)
        lat_d = np.array(lat_dists)
        ld_mean = np.mean(lat_d)
        ld_centered = lat_d - ld_mean
        ld_var = np.sum(ld_centered ** 2)
        lat_mean = np.mean(lat_array, axis=1, keepdims=True)
        lat_centered = lat_array - lat_mean
        prop_speed = np.dot(lat_centered, ld_centered) / max(ld_var, 1e-12)
    else:
        prop_speed = np.zeros(n_samples)
    new_features.append(prop_speed)
    new_names.append('propagation_speed')

    # Stack new features
    new_feat_array = np.column_stack(new_features)
    X_aug = np.column_stack([X, new_feat_array])
    fn_aug = list(feature_names) + new_names

    print(f"  Added {len(new_names)} propagation features "
          f"({X.shape[1]} -> {X_aug.shape[1]})")
    return X_aug, fn_aug


def _decode_feature_label(raw_name):
    """
    Convert raw feature name to a readable label for plots.

    Examples:
        'd20_a90_auc_post'          -> 'Far (d=20, 90°) AUC post'
        'center_baseline_activity'  -> 'Center baseline'
        'spatial_gradient'          -> 'Spatial gradient (pop.)'
        'ring_mean_amp_d15'         -> 'Ring mean amp d=15 (prop.)'
    """
    # Propagation features (population-level)
    prop_names = {
        'ring_mean_amp_d5':  'Ring mean amp d=5 (prop.)',
        'ring_mean_amp_d10': 'Ring mean amp d=10 (prop.)',
        'ring_mean_amp_d15': 'Ring mean amp d=15 (prop.)',
        'ring_mean_amp_d20': 'Ring mean amp d=20 (prop.)',
        'amplitude_decay_slope': 'Amp decay slope (prop.)',
        'response_reach': 'Response reach (prop.)',
        'center_surround_ratio': 'Center/surround ratio (prop.)',
        'propagation_speed': 'Propagation speed (prop.)',
    }
    if raw_name in prop_names:
        return prop_names[raw_name]

    # Population-level features
    pop_names = {
        'spatial_gradient': 'Spatial gradient (pop.)',
        'spread_latency_gradient': 'Spread latency grad. (pop.)',
        'population_synchrony': 'Population synchrony (pop.)',
        'max_responding_distance': 'Max responding dist. (pop.)',
    }
    if raw_name in pop_names:
        return pop_names[raw_name]

    # Per-neuron features: center_<feat> or d<dist>_a<angle>_<feat>
    feat_short = {
        'baseline_activity': 'Baseline',
        'peak_response': 'Peak resp.',
        'response_amplitude': 'Amplitude',
        'response_latency': 'Latency',
        'time_to_peak': 'Time to peak',
        'decay_half_life': 'Decay t½',
        'auc_stim': 'AUC stim',
        'auc_post': 'AUC post',
        'onset_slope': 'Onset slope',
        'sustained_index': 'Sustained idx',
        'distance_from_center': 'Distance',
        'angle_from_center': 'Angle',
    }

    # Center neuron
    m_center = re.match(r'^center_(.+)$', raw_name)
    if m_center:
        feat = m_center.group(1)
        return f"Center {feat_short.get(feat, feat)}"

    # Ring neuron
    m_ring = re.match(r'^d(\d+)_a(\d+)_(.+)$', raw_name)
    if m_ring:
        dist = int(m_ring.group(1))
        angle = int(m_ring.group(2))
        feat = m_ring.group(3)
        if dist <= 10:
            dist_label = 'Near'
        else:
            dist_label = 'Far'
        return f"{dist_label} (d={dist}, {angle}°) {feat_short.get(feat, feat)}"

    return raw_name


# =============================================================================
# Job Generation and Execution
# =============================================================================

def generate_jobs(best_matches, stim_modes, stim_duration, stim_sizes,
                  stim_bias_values=None):
    """
    Generate list of all jobs to run.
    Each job = one (condition, sim_idx, stim_size, stim_mode) combination.
    For 'bias' mode, generates separate jobs for each bias value.
    """
    jobs = []
    for condition in CONDITIONS:
        if condition not in best_matches:
            continue
        for sim_idx, params in enumerate(best_matches[condition]['simulations']):
            for stim_size in stim_sizes:
                for stim_mode in stim_modes:
                    if stim_mode == 'bias' and stim_bias_values:
                        for stim_bias in stim_bias_values:
                            jobs.append({
                                'condition': condition,
                                'sim_idx': sim_idx,
                                'params': params,
                                'stim_size': stim_size,
                                'stim_duration': stim_duration,
                                'stim_mode': stim_mode,
                                'stim_bias': stim_bias,
                            })
                    else:
                        jobs.append({
                            'condition': condition,
                            'sim_idx': sim_idx,
                            'params': params,
                            'stim_size': stim_size,
                            'stim_duration': stim_duration,
                            'stim_mode': stim_mode,
                        })
    return jobs


def run_feature_job(args):
    """
    Worker function: run all replicates for one job, extract features
    at all RF radii.

    Returns
    -------
    dict with features per RF radius, labels, groups, metadata
    """
    job_info, config_dict = args

    condition = job_info['condition']
    sim_idx = job_info['sim_idx']
    params = job_info['params']
    stim_size = job_info['stim_size']
    stim_duration = job_info['stim_duration']
    stim_mode = job_info['stim_mode']
    stim_bias = job_info.get('stim_bias')

    label = CONDITION_LABELS[condition]
    group_id = CONDITION_LABELS[condition] * N_TOP_MATCHES + sim_idx

    # Seed for reproducibility (deterministic across modes and bias values)
    STIM_MODE_SEED_OFFSET = {
        'clamped': 0, 'double_pulse': 1, 'bias': 2,
        'double_pulse3': 3, 'double_pulse5': 4, 'double_pulse10': 5,
    }
    mode_offset = STIM_MODE_SEED_OFFSET.get(stim_mode, 0)
    bias_component = int(round(stim_bias * 100)) if stim_bias is not None else 0
    seed = mode_offset * 100000 + sim_idx * 10000 + stim_size * 100 + stim_duration + bias_component + hash(condition) % 1000

    # Run simulation and extract neuron responses at all RF radii
    queen = config_dict.get('queen', False)
    result = run_downstream_reader_job(
        params, stim_size, stim_duration, stim_mode, stim_bias=stim_bias,
        seed=seed, queen=queen)

    responses = result['responses']  # (n_radii, N_NEURONS, N_REPLICATES, total_frames)
    neuron_meta = result['neuron_metadata']
    n_radii = len(RF_RADII)

    # Extract features for each RF radius
    features_by_radius = {}
    feature_names_by_radius = {}

    for ri, rf_radius in enumerate(RF_RADII):
        rep_features = np.zeros((N_REPLICATES, N_TOTAL_FEATURES), dtype=np.float64)
        fn = None

        for rep in range(N_REPLICATES):
            neuron_resp = responses[ri, :, rep, :]  # (N_NEURONS, total_frames)
            fv, names = build_feature_vector(neuron_resp, stim_duration, neuron_meta)
            rep_features[rep, :] = fv
            if fn is None:
                fn = names

        features_by_radius[rf_radius] = rep_features
        feature_names_by_radius[rf_radius] = fn

    labels = np.full(N_REPLICATES, label, dtype=np.int32)
    groups = np.full(N_REPLICATES, group_id, dtype=np.int32)

    del responses
    gc.collect()

    return {
        'features_by_radius': features_by_radius,
        'feature_names_by_radius': feature_names_by_radius,
        'labels': labels,
        'groups': groups,
        'condition': condition,
        'sim_idx': sim_idx,
        'stim_size': stim_size,
        'stim_duration': stim_duration,
        'stim_mode': stim_mode,
        'stim_bias': stim_bias,
        'params': params,
    }


def run_detection_job(args):
    """
    Worker function for stimulus detection analysis.

    Per (condition, sim_idx, stim_size):
    1. Shared burn-in -> 2*N_REPLICATES equilibrium states
    2. First half -> stim trials via run_snapshot_perturbation (y=1)
    3. Second half -> control trials via run_free_evolution_snapshots (y=0)
    4. Extract per-frame features using VirtualNeurons

    Parameters
    ----------
    args : tuple of (job_info, config_dict)

    Returns
    -------
    dict with X_single, X_window, y, frame_idx, sim_idx arrays
    """
    job_info, config_dict = args

    condition = job_info['condition']
    sim_idx = job_info['sim_idx']
    params = job_info['params']
    stim_size = job_info['stim_size']
    stim_duration = job_info['stim_duration']
    stim_mode = job_info['stim_mode']
    stim_bias = job_info.get('stim_bias')
    rf_radius = config_dict.get('detect_rf_radius', 6)
    window_half = config_dict.get('detect_window', 2)

    total_frames = PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES
    n_total_reps = 2 * N_REPLICATES  # half stim, half control

    # Build neuron population at specified RF radius
    neurons, neuron_meta = build_neuron_population(rf_radius)
    n_neurons = len(neurons)  # 17

    # Seed for reproducibility
    STIM_MODE_SEED_OFFSET = {
        'clamped': 0, 'double_pulse': 1, 'bias': 2,
        'double_pulse3': 3, 'double_pulse5': 4, 'double_pulse10': 5,
    }
    mode_offset = STIM_MODE_SEED_OFFSET.get(stim_mode, 0)
    bias_component = int(round(stim_bias * 100)) if stim_bias is not None else 0
    seed = (mode_offset * 100000 + sim_idx * 10000 + stim_size * 100
            + stim_duration + bias_component + 77777)  # offset from main jobs
    np.random.seed(seed)

    # --- Shared equilibrium ---
    beta = params['beta']
    c = params['c']
    decay_const = params['decay_const']
    rad = params['inhibition_range']
    bias_val = params['bias']
    queen = config_dict.get('queen', False)
    kernel = build_diamond_kernel(rad, queen=queen)
    K_sum = float(np.sum(kernel))

    spin_config = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, M))
    H = np.zeros((L, M), dtype=np.float64)

    tau = (L * M) / decay_const
    burn_in_steps = max(BURN_IN_MIN, int(BURN_IN_TAU_MULT * tau))
    for _ in range(burn_in_steps):
        spin_config, H = heat_bath_numba(
            spin_config, beta, c, decay_const, H, kernel, bias_val, K_sum, queen)

    decorr_gap = max(DECORR_MIN, int(DECORR_TAU_MULT * tau))
    eq_states = [(spin_config.copy(), H.copy())]
    for _ in range(n_total_reps - 1):
        for __ in range(decorr_gap):
            spin_config, H = heat_bath_numba(
                spin_config, beta, c, decay_const, H, kernel, bias_val, K_sum, queen)
        eq_states.append((spin_config.copy(), H.copy()))

    # --- Run stim and control trials ---
    all_responses = np.zeros((n_total_reps, n_neurons, total_frames), dtype=np.float32)
    labels = np.zeros(n_total_reps, dtype=np.int32)

    for rep in range(n_total_reps):
        if rep < N_REPLICATES:
            # Stim trial
            snapshots = run_snapshot_perturbation(
                params, stim_size, stim_duration, stim_mode,
                initial_state=eq_states[rep], stim_bias=stim_bias,
                queen=queen)
            labels[rep] = 1
        else:
            # Control trial (free evolution, same total frames)
            snapshots = run_free_evolution_snapshots(
                params, total_frames, initial_state=eq_states[rep],
                queen=queen)
            labels[rep] = 0

        for ni, neuron in enumerate(neurons):
            all_responses[rep, ni, :] = neuron.response_timeseries(snapshots)

        del snapshots

    del eq_states
    gc.collect()

    # --- Extract per-frame features (vectorized) ---
    window_size = 2 * window_half + 1  # default: 5

    # Single-frame features: reshape (reps, neurons, frames) -> (reps*frames, neurons)
    X_single = all_responses.transpose(0, 2, 1).reshape(-1, n_neurons)

    # Windowed features: edge-padded sliding window
    padded = np.pad(all_responses,
                    ((0, 0), (0, 0), (window_half, window_half)),
                    mode='edge')  # (reps, neurons, frames+2*window_half)
    win_idx = np.arange(window_size) + np.arange(total_frames)[:, None]
    windowed = padded[:, :, win_idx]           # (reps, neurons, frames, window_size)
    windowed = windowed.transpose(0, 2, 3, 1)  # (reps, frames, window_size, neurons)
    X_window = windowed.reshape(-1, n_neurons * window_size).astype(np.float32)

    # Trial-level identity expanded to frames (1=stim trial, 0=control)
    trial_type = np.repeat(labels, total_frames)

    # Frame-level labels: onset_onward (1 for stim trials at frames >= PRE_STIM_FRAMES)
    frame_arr = np.tile(np.arange(total_frames, dtype=np.int32), n_total_reps)
    y_onset = ((trial_type == 1) & (frame_arr >= PRE_STIM_FRAMES)).astype(np.int32)

    # Frame-level labels: stim_only (1 for stim trials during stimulus period only)
    y_stim = ((trial_type == 1)
              & (frame_arr >= PRE_STIM_FRAMES)
              & (frame_arr < PRE_STIM_FRAMES + stim_duration)).astype(np.int32)

    frame_indices = frame_arr
    sim_indices = np.full(n_total_reps * total_frames, sim_idx, dtype=np.int32)

    return {
        'X_single': X_single,
        'X_window': X_window,
        'y_onset': y_onset,
        'y_stim': y_stim,
        'trial_type': trial_type,
        'frame_indices': frame_indices,
        'sim_indices': sim_indices,
        'condition': condition,
        'sim_idx': sim_idx,
        'stim_size': stim_size,
        'n_neurons': n_neurons,
        'window_size': window_size,
        'total_frames': total_frames,
    }


def generate_grouped_jobs(best_matches, stim_modes, stim_duration, stim_sizes,
                          stim_bias_values=None):
    """
    Group jobs by (condition, sim_idx) so equilibrium is computed once per group.

    Each group collects all stim configs (size x mode x bias) that share the
    same Ising parameters, allowing the grouped worker to reuse equilibrium
    states and control trials across stim configs.
    """
    grouped = {}
    for condition in CONDITIONS:
        if condition not in best_matches:
            continue
        for sim_idx, params in enumerate(best_matches[condition]['simulations']):
            stim_configs = []
            for stim_size in stim_sizes:
                for stim_mode in stim_modes:
                    if stim_mode == 'bias' and stim_bias_values:
                        for sb in stim_bias_values:
                            stim_configs.append((stim_size, stim_mode, sb))
                    else:
                        stim_configs.append((stim_size, stim_mode, None))
            grouped[(condition, sim_idx)] = {
                'condition': condition,
                'sim_idx': sim_idx,
                'params': params,
                'stim_configs': stim_configs,
                'stim_duration': stim_duration,
            }
    return list(grouped.values())


def run_detection_job_grouped(args):
    """
    Worker: one (condition, sim_idx) with all stim configs.

    Equilibrium + control trials computed once, stim trials per config.
    Returns list of result dicts (one per stim config).
    """
    group_info, config_dict = args
    condition = group_info['condition']
    sim_idx = group_info['sim_idx']
    params = group_info['params']
    stim_configs = group_info['stim_configs']
    stim_duration = group_info['stim_duration']
    rf_radius = config_dict.get('detect_rf_radius', 6)
    window_half = config_dict.get('detect_window', 2)

    total_frames = PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES
    n_total_reps = 2 * N_REPLICATES

    neurons, neuron_meta = build_neuron_population(rf_radius)
    n_neurons = len(neurons)

    # Deterministic seed for equilibrium (independent of stim config)
    seed = sim_idx * 10000 + 77777
    np.random.seed(seed)

    # --- Shared equilibrium (done ONCE) ---
    beta = params['beta']
    c = params['c']
    decay_const = params['decay_const']
    rad = params['inhibition_range']
    bias_val = params['bias']
    queen = config_dict.get('queen', False)
    kernel = build_diamond_kernel(rad, queen=queen)
    K_sum = float(np.sum(kernel))

    spin_config = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, M))
    H = np.zeros((L, M), dtype=np.float64)

    tau = (L * M) / decay_const
    burn_in_steps = max(BURN_IN_MIN, int(BURN_IN_TAU_MULT * tau))
    for _ in range(burn_in_steps):
        spin_config, H = heat_bath_numba(
            spin_config, beta, c, decay_const, H, kernel, bias_val, K_sum, queen)

    decorr_gap = max(DECORR_MIN, int(DECORR_TAU_MULT * tau))
    eq_states = [(spin_config.copy(), H.copy())]
    for _ in range(n_total_reps - 1):
        for __ in range(decorr_gap):
            spin_config, H = heat_bath_numba(
                spin_config, beta, c, decay_const, H, kernel, bias_val, K_sum, queen)
        eq_states.append((spin_config.copy(), H.copy()))

    # --- Control trials (done ONCE, shared across all stim configs) ---
    control_responses = np.zeros((N_REPLICATES, n_neurons, total_frames),
                                 dtype=np.float32)
    for rep in range(N_REPLICATES):
        snapshots = run_free_evolution_snapshots(
            params, total_frames, initial_state=eq_states[N_REPLICATES + rep],
            kernel=kernel, K_sum=K_sum, queen=queen)
        for ni, neuron in enumerate(neurons):
            control_responses[rep, ni, :] = neuron.response_timeseries(snapshots)
        del snapshots

    # --- Per-stim-config: stim trials only ---
    results_list = []
    window_size = 2 * window_half + 1

    for stim_size, stim_mode, stim_bias in stim_configs:
        stim_responses = np.zeros((N_REPLICATES, n_neurons, total_frames),
                                  dtype=np.float32)
        for rep in range(N_REPLICATES):
            snapshots = run_snapshot_perturbation(
                params, stim_size, stim_duration, stim_mode,
                initial_state=eq_states[rep], stim_bias=stim_bias,
                kernel=kernel, K_sum=K_sum, queen=queen)
            for ni, neuron in enumerate(neurons):
                stim_responses[rep, ni, :] = neuron.response_timeseries(snapshots)
            del snapshots

        # Combine stim + control responses
        all_responses = np.concatenate([stim_responses, control_responses], axis=0)
        labels = np.array([1]*N_REPLICATES + [0]*N_REPLICATES, dtype=np.int32)
        del stim_responses

        # Extract per-frame features (vectorized)
        X_single = all_responses.transpose(0, 2, 1).reshape(-1, n_neurons)

        padded = np.pad(all_responses,
                        ((0, 0), (0, 0), (window_half, window_half)),
                        mode='edge')
        win_idx = np.arange(window_size) + np.arange(total_frames)[:, None]
        windowed = padded[:, :, win_idx]
        windowed = windowed.transpose(0, 2, 3, 1)
        X_window = windowed.reshape(-1, n_neurons * window_size).astype(np.float32)

        # Trial-level identity expanded to frames
        trial_type = np.repeat(labels, total_frames)

        # Frame-level labels
        frame_indices = np.tile(np.arange(total_frames, dtype=np.int32), n_total_reps)
        y_onset = ((trial_type == 1) & (frame_indices >= PRE_STIM_FRAMES)).astype(np.int32)
        y_stim_label = ((trial_type == 1)
                        & (frame_indices >= PRE_STIM_FRAMES)
                        & (frame_indices < PRE_STIM_FRAMES + stim_duration)).astype(np.int32)

        sim_indices_arr = np.full(n_total_reps * total_frames, sim_idx, dtype=np.int32)

        results_list.append({
            'X_single': X_single,
            'X_window': X_window,
            'y_onset': y_onset,
            'y_stim': y_stim_label,
            'trial_type': trial_type,
            'frame_indices': frame_indices,
            'sim_indices': sim_indices_arr,
            'condition': condition,
            'sim_idx': sim_idx,
            'stim_size': stim_size,
            'stim_mode': stim_mode,
            'stim_bias': stim_bias,
            'n_neurons': n_neurons,
            'window_size': window_size,
            'total_frames': total_frames,
        })

    del eq_states, control_responses
    gc.collect()
    return results_list


def build_dataset(rf_radius, results_list):
    """
    Assemble feature dataset for a specific RF radius from pre-computed results.

    Parameters
    ----------
    rf_radius : int
        RF radius to extract features for
    results_list : list of dict
        Output from run_feature_job for all jobs

    Returns
    -------
    X : ndarray (n_samples, N_TOTAL_FEATURES)
    y : ndarray (n_samples,)
    groups : ndarray (n_samples,)
    feature_names : list of str
    """
    X_parts = []
    y_parts = []
    g_parts = []
    feature_names = None

    for res in results_list:
        X_parts.append(res['features_by_radius'][rf_radius])
        y_parts.append(res['labels'])
        g_parts.append(res['groups'])
        if feature_names is None:
            feature_names = res['feature_names_by_radius'][rf_radius]

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)

    # Replace NaN with 0
    nan_mask = np.isnan(X)
    if np.any(nan_mask):
        X[nan_mask] = 0.0

    return X, y, groups, feature_names


# =============================================================================
# Classification
# =============================================================================

def _set_pub_style():
    """Apply publication-quality matplotlib rcParams."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 1.0,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.fontsize': 9,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def _get_feature_category(feat_name):
    """Return category string for a feature name."""
    for key, cat in FEATURE_CATEGORY_MAP.items():
        if key in feat_name:
            return cat
    return 'temporal'


def _build_models():
    """Instantiate the four classifiers."""
    import inspect
    # LinearSVC: 'dual' param accepts 'auto' only in sklearn >=1.3
    svc_params = inspect.signature(LinearSVC.__init__).parameters
    dual_val = 'auto' if 'auto' in str(svc_params.get('dual', '')) else False
    try:
        svm = LinearSVC(dual=dual_val, max_iter=2000, random_state=42)
    except (TypeError, ValueError):
        svm = LinearSVC(max_iter=2000, random_state=42)
    # CalibratedClassifierCV: 'estimator' replaces 'base_estimator' in >=1.2
    cal_params = inspect.signature(CalibratedClassifierCV.__init__).parameters
    if 'estimator' in cal_params:
        cal_svm = CalibratedClassifierCV(estimator=svm, cv=3)
        rbf_svm = CalibratedClassifierCV(
            estimator=SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
            cv=3)
    else:
        cal_svm = CalibratedClassifierCV(base_estimator=svm, cv=3)
        rbf_svm = CalibratedClassifierCV(
            base_estimator=SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
            cv=3)
    return {
        'LogReg_L2': LogisticRegression(
            penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            random_state=42, n_jobs=1),
        'LinearSVM': cal_svm,
        'RBF_SVM': rbf_svm,
    }

ALL_MODEL_NAMES = ['LogReg_L2', 'RandomForest', 'LinearSVM', 'RBF_SVM']


def run_classification(X, y, groups, feature_names):
    """
    Run 4 classifiers with GroupKFold cross-validation.

    Returns
    -------
    results : dict with per-model metrics, feature importances, confusion matrices.
    """
    n_splits = min(10, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    results = {}
    for mname in ALL_MODEL_NAMES:
        results[mname] = {
            'accuracy': [], 'balanced_accuracy': [], 'auc': [], 'f1': [],
            'y_true_all': [], 'y_prob_all': [], 'y_pred_all': [],
        }

    rf_importances_acc = np.zeros(X.shape[1])
    lr_coef_acc = np.zeros(X.shape[1])
    n_folds_done = 0

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = _build_models()

        for mname, clf in models.items():
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            y_prob = clf.predict_proba(X_test_s)[:, 1]

            results[mname]['accuracy'].append(accuracy_score(y_test, y_pred))
            results[mname]['balanced_accuracy'].append(
                balanced_accuracy_score(y_test, y_pred))
            results[mname]['f1'].append(f1_score(y_test, y_pred))

            try:
                auc_val = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc_val = np.nan
            results[mname]['auc'].append(auc_val)

            results[mname]['y_true_all'].extend(y_test.tolist())
            results[mname]['y_prob_all'].extend(y_prob.tolist())
            results[mname]['y_pred_all'].extend(y_pred.tolist())

            if mname == 'RandomForest':
                rf_importances_acc += clf.feature_importances_
            elif mname == 'LogReg_L2':
                lr_coef_acc += np.abs(clf.coef_.ravel())

        n_folds_done += 1

    results['rf_importances'] = rf_importances_acc / max(n_folds_done, 1)
    results['lr_coef_abs'] = lr_coef_acc / max(n_folds_done, 1)
    results['feature_names'] = list(feature_names)

    for mname in ALL_MODEL_NAMES:
        for metric in ['accuracy', 'balanced_accuracy', 'auc', 'f1']:
            arr = np.array(results[mname][metric])
            results[mname][f'{metric}_mean'] = float(np.nanmean(arr))
            results[mname][f'{metric}_std'] = float(np.nanstd(arr))
        results[mname]['y_true_all'] = np.array(results[mname]['y_true_all'])
        results[mname]['y_prob_all'] = np.array(results[mname]['y_prob_all'])
        results[mname]['y_pred_all'] = np.array(results[mname]['y_pred_all'])
        results[mname]['confusion_matrix'] = confusion_matrix(
            results[mname]['y_true_all'], results[mname]['y_pred_all'])

    return results


def _get_feature_indices_by_category(feature_names):
    """
    Parse feature names and return index arrays for each category.

    Returns
    -------
    dict mapping category str -> np.ndarray of column indices
    """
    cats = {'temporal': [], 'spatial': [], 'population': [], 'propagation': []}
    for i, name in enumerate(feature_names):
        cat = _get_feature_category(name)
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(i)
    return {k: np.array(v, dtype=int) for k, v in cats.items() if v}


def _get_feature_indices_by_distance(feature_names):
    """
    Parse feature names and return index arrays grouped by neuron distance.

    Groups:
        center  — neuron at distance 0 (label 'center_*')
        near    — neurons at d5, d10
        far     — neurons at d15, d20
        population — population-level features and propagation features

    Returns
    -------
    dict mapping group str -> np.ndarray of column indices
    """
    groups = {'center': [], 'near': [], 'far': [], 'population': []}
    pop_names = {'spatial_gradient', 'spread_latency_gradient',
                 'population_synchrony', 'max_responding_distance'}
    prop_names = {'ring_mean_amp_d5', 'ring_mean_amp_d10',
                  'ring_mean_amp_d15', 'ring_mean_amp_d20',
                  'amplitude_decay_slope', 'response_reach',
                  'center_surround_ratio', 'propagation_speed'}
    for i, name in enumerate(feature_names):
        if name in pop_names or name in prop_names:
            groups['population'].append(i)
        elif name.startswith('center_'):
            groups['center'].append(i)
        elif name.startswith('d5_') or name.startswith('d10_'):
            groups['near'].append(i)
        elif name.startswith('d15_') or name.startswith('d20_'):
            groups['far'].append(i)
        else:
            # Fallback: check for distance pattern dXX_
            m = re.match(r'd(\d+)_', name)
            if m:
                d = int(m.group(1))
                if d <= 10:
                    groups['near'].append(i)
                else:
                    groups['far'].append(i)
            else:
                groups['population'].append(i)
    return {k: np.array(v, dtype=int) for k, v in groups.items() if v}


def _ablation_one_category(X_sub, y, groups):
    """Run RF GroupKFold CV on a feature subset. Returns (mean_acc, std_acc, fold_accs)."""
    n_splits = min(10, len(np.unique(groups)))
    if n_splits < 2:
        return None
    gkf = GroupKFold(n_splits=n_splits)
    fold_accs = []
    for train_idx, test_idx in gkf.split(X_sub, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_sub[train_idx])
        X_te = scaler.transform(X_sub[test_idx])
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            random_state=42, n_jobs=1)
        clf.fit(X_tr, y[train_idx])
        fold_accs.append(accuracy_score(y[test_idx], clf.predict(X_te)))
    accs = np.array(fold_accs)
    return float(np.mean(accs)), float(np.std(accs)), fold_accs


def run_feature_ablation(X, y, groups, feature_names):
    """
    Run classification using only features from each category (temporal,
    spatial, propagation, population) and report accuracy. Parallelised with joblib.

    Returns
    -------
    dict mapping category str -> dict with accuracy_mean, accuracy_std, n_folds
    """
    cat_indices = _get_feature_indices_by_category(feature_names)

    # Build jobs: per-category subsets + all-features
    job_names = list(cat_indices.keys()) + ['all']
    job_X = [X[:, idx] for idx in cat_indices.values()] + [X]
    job_n_features = [len(idx) for idx in cat_indices.values()] + [X.shape[1]]

    raw_results = Parallel(n_jobs=-1)(
        delayed(_ablation_one_category)(Xj, y, groups) for Xj in job_X)

    results = {}
    for name, res, n_feat in zip(job_names, raw_results, job_n_features):
        if res is None:
            continue
        acc_mean, acc_std, fold_accs = res
        results[name] = {
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'n_features': n_feat,
            'n_folds': len(fold_accs),
        }
        print(f"  {name:12s}: acc={acc_mean:.3f} +/- {acc_std:.3f}  "
              f"({n_feat} features)")

    return results


def run_distance_ablation(X, y, groups, feature_names):
    """
    Run classification using neurons grouped by distance from stimulus center.

    Groups: center, near (d5+d10), far (d15+d20), all. Parallelised with joblib.

    Returns
    -------
    dict mapping group str -> dict with accuracy_mean, accuracy_std, n_folds
    """
    dist_indices = _get_feature_indices_by_distance(feature_names)

    # Build jobs: per-group subsets + all-features
    job_names = list(dist_indices.keys()) + ['all']
    job_X = [X[:, idx] for idx in dist_indices.values()] + [X]
    job_n_features = [len(idx) for idx in dist_indices.values()] + [X.shape[1]]

    raw_results = Parallel(n_jobs=-1)(
        delayed(_ablation_one_category)(Xj, y, groups) for Xj in job_X)

    results = {}
    for name, res, n_feat in zip(job_names, raw_results, job_n_features):
        if res is None:
            continue
        acc_mean, acc_std, fold_accs = res
        results[name] = {
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'n_features': n_feat,
            'n_folds': len(fold_accs),
        }
        print(f"  {name:12s}: acc={acc_mean:.3f} +/- {acc_std:.3f}  "
              f"({n_feat} features)")

    return results


def find_optimal_radius(sweep_results, model='RandomForest', metric='accuracy_mean'):
    """Return the RF radius that maximises the given metric."""
    best_r = None
    best_val = -np.inf
    for r, res in sweep_results.items():
        val = res[model][metric]
        if val > best_val:
            best_val = val
            best_r = r
    return best_r


def run_detection_classification(X, y, frame_indices, sim_indices, total_frames,
                                  trial_type=None):
    """
    Run stimulus detection classification with GroupKFold by simulation.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,) -- frame-level training labels (1=stim active, 0=otherwise)
    frame_indices : ndarray (n_samples,)
    sim_indices : ndarray (n_samples,)
    total_frames : int
    trial_type : ndarray (n_samples,) or None
        Trial-level identity (1=stim trial, 0=control trial).  Used to select
        stim-trial frames for P(detected) computation.  If None, falls back to y.

    Returns
    -------
    dict with p_detected, p_detected_std, accuracy_by_frame arrays
    """
    if trial_type is None:
        trial_type = y
    unique_sims = np.unique(sim_indices)
    n_groups = len(unique_sims)
    n_splits = min(5, n_groups)

    if n_splits < 2:
        print("    Warning: too few groups for cross-validation")
        return None

    gkf = GroupKFold(n_splits=n_splits)

    # Accumulate per-frame predictions across folds
    prob_accum = np.zeros(len(y))
    pred_accum = np.zeros(len(y), dtype=np.int32)
    fold_count = np.zeros(len(y), dtype=np.int32)

    fold_accs = []

    for train_idx, test_idx in gkf.split(X, y, sim_indices):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
            class_weight='balanced')
        clf.fit(X_train, y[train_idx])

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        prob_accum[test_idx] = y_prob
        pred_accum[test_idx] = y_pred
        fold_count[test_idx] = 1

        fold_accs.append(accuracy_score(y[test_idx], y_pred))

    # Per-frame aggregation
    p_detected = np.zeros(total_frames)
    p_detected_std = np.zeros(total_frames)
    accuracy_by_frame = np.zeros(total_frames)
    n_samples_per_frame = np.zeros(total_frames, dtype=np.int32)

    for t in range(total_frames):
        mask = (frame_indices == t) & (fold_count > 0)
        if np.sum(mask) == 0:
            continue

        probs_t = prob_accum[mask]
        preds_t = pred_accum[mask]
        true_t = y[mask]

        # P(detected) = mean predicted probability of class 1 for STIM trials only
        stim_mask_t = trial_type[mask] == 1
        if np.sum(stim_mask_t) > 0:
            p_detected[t] = np.mean(probs_t[stim_mask_t])
            p_detected_std[t] = np.std(probs_t[stim_mask_t])

        # Accuracy at this frame
        accuracy_by_frame[t] = accuracy_score(true_t, preds_t)
        n_samples_per_frame[t] = np.sum(mask)

    return {
        'p_detected': p_detected,
        'p_detected_std': p_detected_std,
        'accuracy_by_frame': accuracy_by_frame,
        'n_samples_per_frame': n_samples_per_frame,
        'overall_accuracy': float(np.mean(fold_accs)),
        'overall_accuracy_std': float(np.std(fold_accs)),
        'n_folds': n_splits,
    }


def _one_permutation(X, y_perm, groups, n_splits):
    """Run one permutation: GroupKFold CV with LogReg, return mean accuracy."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_accs = []
    for train_idx, test_idx in gkf.split(X, y_perm, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(
            penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(X_tr, y_perm[train_idx])
        fold_accs.append(accuracy_score(y_perm[test_idx], clf.predict(X_te)))
    return np.mean(fold_accs)


def run_permutation_test(X, y, groups, n_permutations=100):
    """
    Estimate chance-level accuracy by shuffling group labels.
    Uses LogReg only for speed. Parallelised with joblib.
    """
    rng = np.random.RandomState(0)
    unique_groups = np.unique(groups)
    n_splits = min(10, len(unique_groups))

    # Pre-generate all permuted labels deterministically
    y_perms = []
    for _ in range(n_permutations):
        group_labels = {g: y[groups == g][0] for g in unique_groups}
        shuffled_labels = list(group_labels.values())
        rng.shuffle(shuffled_labels)
        y_perm = np.empty_like(y)
        for i, g in enumerate(unique_groups):
            y_perm[groups == g] = shuffled_labels[i]
        y_perms.append(y_perm)

    chance_accs = Parallel(n_jobs=-1, verbose=1)(
        delayed(_one_permutation)(X, yp, groups, n_splits) for yp in y_perms)

    return np.array(chance_accs)


def _one_lc_point(X, y, groups, unique_groups, n_groups, n_train, seed):
    """Run one learning-curve point: split by groups, fit RF, return accs."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_groups)
    train_groups = set(unique_groups[perm[:n_train]])
    test_groups = set(unique_groups[perm[n_train:]])
    if len(test_groups) == 0:
        last = unique_groups[perm[-1]]
        train_groups.discard(last)
        test_groups = {last}
    train_mask = np.isin(groups, list(train_groups))
    test_mask = np.isin(groups, list(test_groups))
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_mask])
    X_te = scaler.transform(X[test_mask])
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        random_state=42, n_jobs=1)
    clf.fit(X_tr, y[train_mask])
    return (accuracy_score(y[train_mask], clf.predict(X_tr)),
            accuracy_score(y[test_mask], clf.predict(X_te)))


def compute_learning_curve(X, y, groups, train_fractions=None):
    """Accuracy vs number of training groups (simulations). Parallelised."""
    if train_fractions is None:
        train_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_repeats = 10

    # Build job list with deterministic seeds matching original sequential order
    base_rng = np.random.RandomState(42)
    jobs = []  # (fi, rep, n_train, seed)
    n_train_groups_list = []
    for fi, frac in enumerate(train_fractions):
        n_train = max(2, int(round(frac * n_groups)))
        n_train_groups_list.append(n_train)
        for rep in range(n_repeats):
            seed = base_rng.randint(0, 2**31)
            jobs.append((fi, rep, n_train, seed))

    raw_results = Parallel(n_jobs=-1)(
        delayed(_one_lc_point)(X, y, groups, unique_groups, n_groups, nt, s)
        for _, _, nt, s in jobs)

    # Unpack into arrays
    train_acc = np.zeros((len(train_fractions), n_repeats))
    test_acc = np.zeros((len(train_fractions), n_repeats))
    for idx, (fi, rep, _, _) in enumerate(jobs):
        train_acc[fi, rep], test_acc[fi, rep] = raw_results[idx]

    return {
        'fractions': np.array(train_fractions),
        'n_train_groups': np.array(n_train_groups_list),
        'train_acc': train_acc,
        'test_acc': test_acc,
    }


def run_stim_mode_classification(X_by_r, fn_by_r, y, groups, stim_modes_arr,
                                  valid_radii):
    """
    Run classification sweep per stimulus mode.

    Parameters
    ----------
    X_by_r : dict[int -> ndarray]
    fn_by_r : dict[int -> list]
    y, groups, stim_modes_arr : ndarray
    valid_radii : list of int

    Returns
    -------
    sweep_by_mode : dict[mode_str -> dict[radius -> results]]
    """
    unique_modes = sorted(set(stim_modes_arr))
    print(f"\nRunning per-stim-mode classification (modes: {unique_modes})...")

    pairs = [(mode, r) for mode in unique_modes for r in valid_radii]

    def _classify_mode_r(mode, r, X_r, fn_r):
        mode_mask = stim_modes_arr == mode
        if np.sum(mode_mask) == 0:
            return None
        X_m = X_r[mode_mask]
        y_m = y[mode_mask]
        g_m = groups[mode_mask]
        if len(np.unique(y_m)) < 2 or len(np.unique(g_m)) < 2:
            return None
        return mode, r, run_classification(X_m, y_m, g_m, fn_r)

    par_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(_classify_mode_r)(mode, r, X_by_r[r], fn_by_r[r])
        for mode, r in pairs)

    sweep_by_mode = defaultdict(dict)
    for item in par_results:
        if item is not None:
            mode, r, res = item
            sweep_by_mode[mode][r] = res

    sweep_by_mode = dict(sweep_by_mode)
    for mode in sorted(sweep_by_mode.keys()):
        best_r_m = find_optimal_radius(sweep_by_mode[mode])
        if best_r_m is not None:
            best_acc_m = sweep_by_mode[mode][best_r_m]['RandomForest']['accuracy_mean']
            print(f"  mode={mode}: best RF acc={best_acc_m:.3f} at r={best_r_m}")

    return sweep_by_mode if sweep_by_mode else None


# =============================================================================
# Figures
# =============================================================================

def _save_fig(fig, output_dir, name, close=True):
    """Save figure as PNG and PDF."""
    os.makedirs(output_dir, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'{name}.{ext}'))
    print(f"    Saved: {name}.png / .pdf")
    if close:
        plt.close(fig)


def plot_accuracy_vs_rf_radius(sweep_results, output_dir, chance_accs=None):
    """Panel A: classification accuracy vs receptive-field radius (MAIN RESULT)."""
    _set_pub_style()
    fig, ax = plt.subplots(figsize=(5.5, 4))

    radii = np.array(sorted(sweep_results.keys()))

    for mname in ALL_MODEL_NAMES:
        if mname not in sweep_results[radii[0]]:
            continue
        means = np.array([sweep_results[r][mname]['accuracy_mean'] for r in radii])
        stds = np.array([sweep_results[r][mname]['accuracy_std'] for r in radii])
        n_folds = len(sweep_results[radii[0]][mname]['accuracy'])
        sems = stds / np.sqrt(max(n_folds, 1))

        ax.plot(radii, means,
                color=MODEL_COLORS[mname], marker=MODEL_MARKERS[mname],
                markersize=5, linewidth=1.5,
                label=mname.replace('_', ' '), zorder=3)
        ax.fill_between(radii, means - sems, means + sems,
                        color=MODEL_COLORS[mname], alpha=0.15)

        # Per-fold dots
        for r_idx, r in enumerate(radii):
            fold_accs = np.array(sweep_results[r][mname]['accuracy'])
            ax.scatter([r] * len(fold_accs), fold_accs,
                       color=MODEL_COLORS[mname], alpha=0.15, s=12,
                       zorder=2, edgecolors='none')

    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label='Chance')

    if chance_accs is not None:
        thresh = np.percentile(chance_accs, 95)
        ax.axhline(thresh, color='grey', linestyle=':', linewidth=0.8,
                    label=f'Permutation 95% ({thresh:.2f})')

    best_r = find_optimal_radius(sweep_results)
    if best_r is not None:
        ax.axvline(best_r, color='black', linestyle=':', linewidth=0.7, alpha=0.5)

    ax.set_xlabel('Receptive field radius (pixels)')
    ax.set_ylabel('Classification accuracy')
    ax.set_title('Naive vs Expert classification')
    ax.legend(loc='lower right', fontsize=7)
    ax.set_ylim([0.4, 1.0])
    ax.set_xticks(radii)

    _save_fig(fig, output_dir, 'panelA_accuracy_vs_rf')


def plot_roc_curves(results_at_optimal, output_dir):
    """Panel B: ROC curves at optimal RF radius."""
    _set_pub_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    for mname in ALL_MODEL_NAMES:
        if mname not in results_at_optimal:
            continue
        y_true = results_at_optimal[mname]['y_true_all']
        y_prob = results_at_optimal[mname]['y_prob_all']
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = results_at_optimal[mname]['auc_mean']
        ax.plot(fpr, tpr, color=MODEL_COLORS[mname], linewidth=1.5,
                label=f"{mname.replace('_', ' ')} (AUC={auc_val:.2f})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curves')
    ax.legend(loc='lower right', fontsize=7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')

    _save_fig(fig, output_dir, 'panelB_roc')


def plot_feature_importance(results_at_optimal, output_dir, top_n=15):
    """Panel C: Feature importance bar plots with readable labels."""
    _set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))

    feat_names = results_at_optimal['feature_names']
    rf_imp = results_at_optimal['rf_importances']
    lr_coef = results_at_optimal['lr_coef_abs']

    for ax, values, title in [
        (axes[0], rf_imp, 'Random Forest importances'),
        (axes[1], lr_coef, 'Logistic Regression |coef|'),
    ]:
        order = np.argsort(values)[::-1][:top_n]
        raw_names = [feat_names[i] for i in order]
        display_names = [_decode_feature_label(n) for n in raw_names]
        vals = values[order]
        colors = [FEATURE_CAT_COLORS.get(_get_feature_category(n), '#888888')
                  for n in raw_names]

        y_pos = np.arange(len(display_names))
        ax.barh(y_pos, vals, color=colors, edgecolor='none', height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title, fontsize=10)

    # Include all categories that appear in the top features
    legend_cats = [c for c in ['temporal', 'propagation', 'spatial', 'population']
                   if c in FEATURE_CAT_COLORS]
    legend_handles = [
        mlines.Line2D([], [], color=FEATURE_CAT_COLORS[cat],
                       marker='s', linestyle='None', markersize=8, label=cat)
        for cat in legend_cats
    ]
    axes[1].legend(handles=legend_handles, loc='lower right', fontsize=7)

    fig.tight_layout()
    _save_fig(fig, output_dir, 'panelC_feature_importance')


def plot_confusion_matrix(results_at_optimal, output_dir):
    """Panel D: Confusion matrix heatmap for best model."""
    _set_pub_style()

    available_models = [m for m in ALL_MODEL_NAMES if m in results_at_optimal]
    best_model = max(
        available_models,
        key=lambda m: results_at_optimal[m]['accuracy_mean'])

    cm_mat = results_at_optimal[best_model]['confusion_matrix']
    cm_pct = cm_mat.astype(float) / cm_mat.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100, aspect='equal')

    labels = ['Naive', 'Expert']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion matrix ({best_model.replace("_", " ")})')

    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm_pct[i, j] > 60 else 'black'
            ax.text(j, i, f'{cm_mat[i, j]}\n({cm_pct[i, j]:.1f}%)',
                    ha='center', va='center', fontsize=10, color=text_color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='%')
    fig.tight_layout()
    _save_fig(fig, output_dir, 'panelD_confusion')


def plot_accuracy_by_stim_size(sweep_by_stim, stim_sizes, output_dir):
    """Panel E: Accuracy vs RF radius, one subplot per stimulus size."""
    _set_pub_style()
    n_sizes = len(stim_sizes)
    ncols = min(3, n_sizes)
    nrows = int(np.ceil(n_sizes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, sz in enumerate(stim_sizes):
        ax = axes[idx // ncols, idx % ncols]
        sweep = sweep_by_stim[sz]
        radii = np.array(sorted(sweep.keys()))

        for mname in ALL_MODEL_NAMES:
            if mname not in sweep[radii[0]]:
                continue
            means = np.array([sweep[r][mname]['accuracy_mean'] for r in radii])
            stds = np.array([sweep[r][mname]['accuracy_std'] for r in radii])
            n_folds = len(sweep[radii[0]][mname]['accuracy'])
            sems = stds / np.sqrt(max(n_folds, 1))
            label = mname.replace('_', ' ') if idx == 0 else None
            ax.plot(radii, means, color=MODEL_COLORS[mname],
                    marker=MODEL_MARKERS[mname], markersize=4, linewidth=1.2,
                    label=label)
            ax.fill_between(radii, means - sems, means + sems,
                            color=MODEL_COLORS[mname], alpha=0.12)

        ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.7)
        ax.set_title(f'Stim size = {sz}', fontsize=10)
        ax.set_xlabel('RF radius')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0.4, 1.0])
        ax.set_xticks(radii)

    for idx in range(n_sizes, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    # Shared legend from first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=7,
               bbox_to_anchor=(0.98, 0.02))

    fig.tight_layout(rect=[0, 0, 1, 1])
    _save_fig(fig, output_dir, 'panelE_accuracy_by_stim')


def plot_neuron_schematic(rf_radii_to_show=None, output_dir='.'):
    """Panel F: Grid schematic with neuron positions and RF circles."""
    if rf_radii_to_show is None:
        rf_radii_to_show = [2, 8, 15]

    _set_pub_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.add_patch(Rectangle((0, 0), M, L,
                            fill=False, edgecolor='black', linewidth=1))

    crop_w = CROP_COL_END - CROP_COL_START
    crop_h = CROP_ROW_END - CROP_ROW_START
    ax.add_patch(Rectangle(
        (CROP_COL_START, CROP_ROW_START), crop_w, crop_h,
        fill=False, edgecolor='grey', linewidth=1, linestyle='--',
        label='Experimental FOV'))

    stim_size = 4
    half = stim_size // 2
    cr, cc = CENTER_ROW, CENTER_COL
    ax.add_patch(Rectangle(
        (cc - half, cr - half), stim_size, stim_size,
        facecolor='red', edgecolor='darkred', alpha=0.4, linewidth=1,
        label=f'Stimulus ({stim_size}x{stim_size})'))

    neuron_positions = [(cr, cc)]
    for dist in RING_DISTANCES:
        for angle_deg in RING_ANGLES_DEG:
            angle_rad = np.deg2rad(angle_deg)
            r_pos = (cr + dist * np.sin(angle_rad)) % L
            c_pos = (cc + dist * np.cos(angle_rad)) % M
            neuron_positions.append((r_pos, c_pos))

    for pos in neuron_positions:
        ax.plot(pos[1], pos[0], 'ko', markersize=4, zorder=5)

    rf_line_styles = ['-', '--', ':']
    rf_circle_colors = ['#2166AC', '#B2182B', '#1B7837']
    for i, rf_r in enumerate(rf_radii_to_show):
        ls = rf_line_styles[i % len(rf_line_styles)]
        col = rf_circle_colors[i % len(rf_circle_colors)]
        circle = Circle((cc, cr), rf_r, fill=False,
                         edgecolor=col, linewidth=1.2, linestyle=ls,
                         label=f'RF r={rf_r}')
        ax.add_patch(circle)

    ax.set_xlim([-2, M + 2])
    ax.set_ylim([L + 2, -2])
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title('Virtual neuron placement & RF radii')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=7, frameon=True, fancybox=False)

    fig.tight_layout()
    _save_fig(fig, output_dir, 'panelF_neuron_schematic')


def plot_learning_curve(lc_results, output_dir):
    """Accuracy vs number of training groups."""
    _set_pub_style()
    fig, ax = plt.subplots(figsize=(5, 4))

    n_groups = lc_results['n_train_groups']
    n_repeats = lc_results['train_acc'].shape[1]
    train_mean = np.mean(lc_results['train_acc'], axis=1)
    train_std = np.std(lc_results['train_acc'], axis=1)
    train_sem = train_std / np.sqrt(n_repeats)
    test_mean = np.mean(lc_results['test_acc'], axis=1)
    test_std = np.std(lc_results['test_acc'], axis=1)
    test_sem = test_std / np.sqrt(n_repeats)

    ax.plot(n_groups, train_mean, 'o-', color='#2166AC', linewidth=1.5,
            markersize=5, label='Train')
    ax.fill_between(n_groups, train_mean - train_sem, train_mean + train_sem,
                    color='#2166AC', alpha=0.15)
    ax.plot(n_groups, test_mean, 's-', color='#B2182B', linewidth=1.5,
            markersize=5, label='Test')
    ax.fill_between(n_groups, test_mean - test_sem, test_mean + test_sem,
                    color='#B2182B', alpha=0.15)

    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.7)
    ax.set_xlabel('Number of training groups (simulations)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning curve (Random Forest)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([0.3, 1.05])

    fig.tight_layout()
    _save_fig(fig, output_dir, 'learning_curve')


def plot_ablation_bar(ablation_results, output_dir):
    """Bar chart of accuracy by feature category."""
    _set_pub_style()
    fig, ax = plt.subplots(figsize=(5.5, 4))

    ordered_cats = [k for k in ['temporal', 'propagation', 'spatial', 'population', 'all']
                    if k in ablation_results]
    means = [ablation_results[c]['accuracy_mean'] for c in ordered_cats]
    stds = [ablation_results[c]['accuracy_std'] for c in ordered_cats]
    n_feats = [ablation_results[c]['n_features'] for c in ordered_cats]
    n_folds_list = [ablation_results[c].get('n_folds', 1) for c in ordered_cats]
    sems = [s / np.sqrt(max(nf, 1)) for s, nf in zip(stds, n_folds_list)]

    bar_colors = [FEATURE_CAT_COLORS.get(c, '#888888') for c in ordered_cats]
    bar_colors = [c if c != '#888888' else '#555555' for c in bar_colors]
    for i, c in enumerate(ordered_cats):
        if c == 'all':
            bar_colors[i] = '#555555'

    x = np.arange(len(ordered_cats))
    ax.bar(x, means, yerr=sems, capsize=4, color=bar_colors,
           edgecolor='none', width=0.6, alpha=0.85)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.7)
    ax.set_xticks(x)
    labels = [f'{c}\n({n} feat)' for c, n in zip(ordered_cats, n_feats)]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Accuracy (RF)')
    ax.set_title('Feature category ablation')
    ax.set_ylim([0.4, 1.08])

    for i, (m, s) in enumerate(zip(means, sems)):
        y_pos = min(m + s + 0.02, 1.04)
        ax.text(i, y_pos, f'{m:.2f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    _save_fig(fig, output_dir, 'ablation_feature_category')


def plot_distance_ablation_bar(distance_results, output_dir):
    """Bar chart of accuracy by neuron distance group."""
    _set_pub_style()
    fig, ax = plt.subplots(figsize=(5, 4))

    ordered_groups = [k for k in ['center', 'near', 'far', 'population', 'all']
                      if k in distance_results]
    means = [distance_results[g]['accuracy_mean'] for g in ordered_groups]
    stds = [distance_results[g]['accuracy_std'] for g in ordered_groups]
    n_feats = [distance_results[g]['n_features'] for g in ordered_groups]
    n_folds_list = [distance_results[g].get('n_folds', 1) for g in ordered_groups]
    sems = [s / np.sqrt(max(nf, 1)) for s, nf in zip(stds, n_folds_list)]

    dist_colors = {
        'center': '#D62728',
        'near': '#FF7F0E',
        'far': '#2CA02C',
        'population': '#E8922E',
        'all': '#555555',
    }
    bar_colors = [dist_colors.get(g, '#888888') for g in ordered_groups]

    x = np.arange(len(ordered_groups))
    ax.bar(x, means, yerr=sems, capsize=4, color=bar_colors,
           edgecolor='none', width=0.6, alpha=0.85)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.7)
    ax.set_xticks(x)
    group_labels = {
        'center': 'Center\n(d=0)',
        'near': 'Near\n(d5+d10)',
        'far': 'Far\n(d15+d20)',
        'population': 'Pop.\nfeatures',
        'all': 'All',
    }
    labels = [f'{group_labels.get(g, g)}\n({n} feat)' for g, n in zip(ordered_groups, n_feats)]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Accuracy (RF)')
    ax.set_title('Per-distance neuron ablation')
    ax.set_ylim([0.4, 1.08])

    for i, (m, s) in enumerate(zip(means, sems)):
        y_pos = min(m + s + 0.02, 1.04)
        ax.text(i, y_pos, f'{m:.2f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    _save_fig(fig, output_dir, 'ablation_distance_group')


def plot_stimulus_detection(detection_results, stim_duration, output_dir, feature_label):
    """
    Plot stimulus detection results.

    Parameters
    ----------
    detection_results : dict mapping condition -> detection classification results
    stim_duration : int
    output_dir : str
    feature_label : str -- 'single_frame' or 'windowed'
    """
    _set_pub_style()
    os.makedirs(output_dir, exist_ok=True)

    stim_start = PRE_STIM_FRAMES
    stim_end = PRE_STIM_FRAMES + stim_duration

    cond_colors = {'Naive': COLOR_NAIVE, 'Expert': COLOR_EXPERT}

    # --- Figure 1: Side-by-side per condition ---
    conditions = [c for c in CONDITIONS if c in detection_results]
    n_cond = len(conditions)
    fig, axes = plt.subplots(1, n_cond, figsize=(5.5 * n_cond, 4), squeeze=False)

    for ci, cond in enumerate(conditions):
        ax = axes[0, ci]
        res = detection_results[cond]
        total_frames = len(res['p_detected'])
        frames = np.arange(total_frames) - stim_start  # relative to stim onset

        color = cond_colors.get(cond, 'black')

        # Gold fill for stimulus period
        ax.axvspan(0, stim_duration, alpha=0.15, color='gold', label='Stimulus')

        # P(detected) line with SEM band
        sem = res['p_detected_std'] / np.sqrt(res.get('n_folds', 5))
        ax.plot(frames, res['p_detected'], color=color, linewidth=1.5, label='P(detected)')
        ax.fill_between(frames,
                        res['p_detected'] - sem,
                        res['p_detected'] + sem,
                        color=color, alpha=0.2)

        # Baseline line
        ax.axhline(0.0, color='grey', linestyle='--', linewidth=0.8, label='Baseline')

        # Vertical lines at onset/offset
        ax.axvline(0, color='black', linestyle=':', linewidth=0.7, alpha=0.5)
        ax.axvline(stim_duration, color='black', linestyle=':', linewidth=0.7, alpha=0.5)

        # Peak P(detected) during stimulus period
        p_det = res['p_detected']
        peak_val = np.max(p_det[stim_start:stim_end]) if stim_end <= len(p_det) else np.max(p_det)

        ax.set_xlabel('Frame relative to stimulus onset')
        ax.set_ylabel('P(stimulus detected)')
        ax.set_title(f'{cond} ({feature_label})\npeak={peak_val:.3f}')
        ax.set_ylim([0, 1.05])
        ax.legend(loc='upper right', fontsize=7)

    fig.tight_layout()
    _save_fig(fig, output_dir, f'stimulus_detection_sidebyside_{feature_label}')

    # --- Figure 2: Overlay ---
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.axvspan(0, stim_duration, alpha=0.15, color='gold', label='Stimulus')

    for cond in conditions:
        res = detection_results[cond]
        total_frames = len(res['p_detected'])
        frames = np.arange(total_frames) - stim_start
        color = cond_colors.get(cond, 'black')

        # Peak P(detected) during stimulus period
        p_det = res['p_detected']
        peak_val = np.max(p_det[stim_start:stim_end]) if stim_end <= len(p_det) else np.max(p_det)

        sem = res['p_detected_std'] / np.sqrt(res.get('n_folds', 5))
        ax.plot(frames, res['p_detected'], color=color, linewidth=1.5,
                label=f'{cond} (peak={peak_val:.3f})')
        ax.fill_between(frames,
                        res['p_detected'] - sem,
                        res['p_detected'] + sem,
                        color=color, alpha=0.15)

    ax.axhline(0.0, color='grey', linestyle='--', linewidth=0.8, label='Baseline')
    ax.axvline(0, color='black', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.axvline(stim_duration, color='black', linestyle=':', linewidth=0.7, alpha=0.5)

    ax.set_xlabel('Frame relative to stimulus onset')
    ax.set_ylabel('P(stimulus detected)')
    ax.set_title(f'Stimulus detection: Naive vs Expert ({feature_label})')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    _save_fig(fig, output_dir, f'stimulus_detection_overlay_{feature_label}')


def plot_detection_vs_stim_size(detection_results_single, detection_results_window,
                                 stim_sizes, stim_duration, output_dir,
                                 title_suffix='', fname_suffix=''):
    """
    Summary plot: peak P(detected) vs stimulus size for each condition.

    Parameters
    ----------
    detection_results_single : dict mapping (cond, stim_size) -> result dict
    detection_results_window : dict mapping (cond, stim_size) -> result dict
    stim_sizes : list of int
    stim_duration : int
    output_dir : str
    title_suffix : str, optional
    fname_suffix : str, optional
    """
    _set_pub_style()
    os.makedirs(output_dir, exist_ok=True)

    cond_colors = {'Naive': COLOR_NAIVE, 'Expert': COLOR_EXPERT}
    stim_start = PRE_STIM_FRAMES
    stim_end = PRE_STIM_FRAMES + stim_duration

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, (ftype_label, results_dict) in zip(
            axes, [('Single-frame', detection_results_single),
                   ('Windowed', detection_results_window)]):

        for cond in CONDITIONS:
            peaks = []
            peak_sems = []
            sizes_found = []
            for sz in sorted(stim_sizes):
                key = (cond, sz)
                if key not in results_dict:
                    continue
                res = results_dict[key]
                # Peak P(detected) during stimulus period
                p_det = res['p_detected']
                p_det_std = res['p_detected_std']
                if stim_end <= len(p_det):
                    peak_frame_idx = np.argmax(p_det[stim_start:stim_end]) + stim_start
                else:
                    peak_frame_idx = np.argmax(p_det)
                peaks.append(p_det[peak_frame_idx])
                peak_std = p_det_std[peak_frame_idx]
                peak_sems.append(peak_std / np.sqrt(res.get('n_folds', 5)))
                sizes_found.append(sz)

            if sizes_found:
                color = cond_colors.get(cond, 'black')
                ax.errorbar(sizes_found, peaks, yerr=peak_sems, fmt='o-',
                            color=color, linewidth=1.5, markersize=6,
                            capsize=3, label=cond)

        ax.axhline(0.0, color='grey', linestyle='--', linewidth=0.8, label='Baseline')
        ax.set_xlabel('Stimulus size')
        ax.set_ylabel('Peak P(detected)')
        ax.set_title(f'{ftype_label} features')
        ax.set_ylim([0, 1.05])
        ax.legend(loc='lower right', fontsize=8)

    fig.suptitle(f'Stimulus detection vs stimulus size{title_suffix}',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, f'detection_vs_stim_size{fname_suffix}')


def plot_detection_vs_bias(detection_results_single, detection_results_window,
                            stim_sizes, bias_values, stim_duration, output_dir):
    """
    Summary plot: peak P(detected) vs bias value for each (condition, stim_size).

    Parameters
    ----------
    detection_results_single : dict mapping (cond, sz, mode, bias) -> result dict
    detection_results_window : dict mapping (cond, sz, mode, bias) -> result dict
    stim_sizes : list of int
    bias_values : list of float
    stim_duration : int
    output_dir : str
    """
    _set_pub_style()
    os.makedirs(output_dir, exist_ok=True)

    cond_colors = {'Naive': COLOR_NAIVE, 'Expert': COLOR_EXPERT}
    stim_start = PRE_STIM_FRAMES
    stim_end = PRE_STIM_FRAMES + stim_duration

    n_sizes = len(stim_sizes)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, (ftype_label, results_dict) in zip(
            axes, [('Single-frame', detection_results_single),
                   ('Windowed', detection_results_window)]):

        for cond in CONDITIONS:
            color = cond_colors.get(cond, 'black')
            for si, sz in enumerate(sorted(stim_sizes)):
                peaks = []
                peak_sems = []
                biases_found = []
                for bv in sorted(bias_values):
                    key = (cond, sz, 'bias', bv)
                    if key not in results_dict:
                        continue
                    res = results_dict[key]
                    p_det = res['p_detected']
                    p_det_std = res['p_detected_std']
                    if stim_end <= len(p_det):
                        peak_idx = np.argmax(p_det[stim_start:stim_end]) + stim_start
                    else:
                        peak_idx = np.argmax(p_det)
                    peaks.append(p_det[peak_idx])
                    peak_sems.append(p_det_std[peak_idx] / np.sqrt(
                        res.get('n_folds', 5)))
                    biases_found.append(bv)

                if biases_found:
                    alpha = 0.4 + 0.6 * (si / max(n_sizes - 1, 1))
                    ax.errorbar(biases_found, peaks, yerr=peak_sems,
                                fmt='o-', color=color, alpha=alpha,
                                linewidth=1.5, markersize=5, capsize=3,
                                label=f'{cond} sz={sz}')

        ax.axhline(0.0, color='grey', linestyle='--', linewidth=0.8,
                   label='Baseline')
        ax.set_xlabel('Bias value')
        ax.set_ylabel('Peak P(detected)')
        ax.set_title(f'{ftype_label} features')
        ax.set_ylim([0, 1.05])
        ax.legend(loc='lower right', fontsize=6, ncol=2)

    fig.suptitle('Stimulus detection vs bias strength', fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, 'detection_vs_bias')


def plot_accuracy_by_stim_mode(sweep_by_mode, output_dir):
    """Panel G: Accuracy vs RF radius, one subplot per stimulus mode."""
    _set_pub_style()
    modes = sorted(sweep_by_mode.keys())
    n_modes = len(modes)
    ncols = min(3, n_modes)
    nrows = int(np.ceil(n_modes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, mode in enumerate(modes):
        ax = axes[idx // ncols, idx % ncols]
        sweep = sweep_by_mode[mode]
        radii = np.array(sorted(sweep.keys()))

        for mname in ALL_MODEL_NAMES:
            if mname not in sweep[radii[0]]:
                continue
            means = np.array([sweep[r][mname]['accuracy_mean'] for r in radii])
            stds = np.array([sweep[r][mname]['accuracy_std'] for r in radii])
            n_folds = len(sweep[radii[0]][mname]['accuracy'])
            sems = stds / np.sqrt(max(n_folds, 1))
            label = mname.replace('_', ' ') if idx == 0 else None
            ax.plot(radii, means, color=MODEL_COLORS[mname],
                    marker=MODEL_MARKERS[mname], markersize=4, linewidth=1.2,
                    label=label)
            ax.fill_between(radii, means - sems, means + sems,
                            color=MODEL_COLORS[mname], alpha=0.12)

        ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.7)
        ax.set_title(f'Mode: {mode}', fontsize=10)
        ax.set_xlabel('RF radius')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0.4, 1.0])
        ax.set_xticks(radii)

    for idx in range(n_modes, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=7,
               bbox_to_anchor=(0.98, 0.02))

    fig.tight_layout(rect=[0, 0, 1, 1])
    _save_fig(fig, output_dir, 'panelG_accuracy_by_stim_mode')


def plot_all(sweep_results, output_dir, chance_accs=None,
             sweep_by_stim=None, stim_sizes=None, lc_results=None,
             rf_radii_to_show=None, sweep_by_mode=None):
    """Generate all panels (A-G) plus learning curve."""
    print("Generating figures ...")

    plot_accuracy_vs_rf_radius(sweep_results, output_dir, chance_accs=chance_accs)

    opt_r = find_optimal_radius(sweep_results)
    if opt_r is not None:
        res_opt = sweep_results[opt_r]
        plot_roc_curves(res_opt, output_dir)
        plot_feature_importance(res_opt, output_dir)
        plot_confusion_matrix(res_opt, output_dir)

    if sweep_by_stim is not None and stim_sizes is not None:
        plot_accuracy_by_stim_size(sweep_by_stim, stim_sizes, output_dir)

    plot_neuron_schematic(rf_radii_to_show=rf_radii_to_show, output_dir=output_dir)

    if sweep_by_mode is not None:
        plot_accuracy_by_stim_mode(sweep_by_mode, output_dir)

    if lc_results is not None:
        plot_learning_curve(lc_results, output_dir)

    print("All figures saved.")


# =============================================================================
# Save Results
# =============================================================================

def save_results(sweep_results, output_dir, sweep_by_mode=None):
    """Save sweep results as .npz and summary .csv."""
    os.makedirs(output_dir, exist_ok=True)

    # CSV summary
    csv_path = os.path.join(output_dir, 'downstream_reader_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rf_radius', 'model',
            'accuracy_mean', 'accuracy_std',
            'balanced_acc_mean', 'balanced_acc_std',
            'auc_mean', 'auc_std',
            'f1_mean', 'f1_std',
            'stim_mode',
        ])
        for r in sorted(sweep_results.keys()):
            res = sweep_results[r]
            for mname in ALL_MODEL_NAMES:
                if mname not in res:
                    continue
                writer.writerow([
                    r, mname,
                    f"{res[mname]['accuracy_mean']:.4f}",
                    f"{res[mname]['accuracy_std']:.4f}",
                    f"{res[mname]['balanced_accuracy_mean']:.4f}",
                    f"{res[mname]['balanced_accuracy_std']:.4f}",
                    f"{res[mname]['auc_mean']:.4f}",
                    f"{res[mname]['auc_std']:.4f}",
                    f"{res[mname]['f1_mean']:.4f}",
                    f"{res[mname]['f1_std']:.4f}",
                    '',
                ])

        # Per-stim-mode rows
        if sweep_by_mode:
            writer.writerow([])  # blank separator
            for mode in sorted(sweep_by_mode.keys()):
                mode_sweep = sweep_by_mode[mode]
                for r in sorted(mode_sweep.keys()):
                    res = mode_sweep[r]
                    for mname in ALL_MODEL_NAMES:
                        if mname not in res:
                            continue
                        writer.writerow([
                            r, mname,
                            f"{res[mname]['accuracy_mean']:.4f}",
                            f"{res[mname]['accuracy_std']:.4f}",
                            f"{res[mname]['balanced_accuracy_mean']:.4f}",
                            f"{res[mname]['balanced_accuracy_std']:.4f}",
                            f"{res[mname]['auc_mean']:.4f}",
                            f"{res[mname]['auc_std']:.4f}",
                            f"{res[mname]['f1_mean']:.4f}",
                            f"{res[mname]['f1_std']:.4f}",
                            mode,
                        ])

        # Best model summary
        opt_r = find_optimal_radius(sweep_results)
        if opt_r is not None:
            available_models = [m for m in ALL_MODEL_NAMES
                                if m in sweep_results[opt_r]]
            best_model = max(available_models,
                             key=lambda m: sweep_results[opt_r][m]['accuracy_mean'])
            best_acc = sweep_results[opt_r][best_model]['accuracy_mean']
            best_auc = sweep_results[opt_r][best_model]['auc_mean']
            writer.writerow([])
            writer.writerow([f'# Best model summary'])
            writer.writerow([f'# optimal_radius={opt_r}, '
                             f'best_model={best_model}, '
                             f'accuracy={best_acc:.3f}, '
                             f'auc={best_auc:.3f}'])

    print(f"  Saved CSV: {csv_path}")

    # NPZ
    npz_data = {}
    npz_data['rf_radii'] = np.array(sorted(sweep_results.keys()))
    for mname in ALL_MODEL_NAMES:
        if mname not in sweep_results[npz_data['rf_radii'][0]]:
            continue
        for metric in ['accuracy', 'balanced_accuracy', 'auc', 'f1']:
            means = [sweep_results[r][mname][f'{metric}_mean']
                     for r in npz_data['rf_radii']]
            stds = [sweep_results[r][mname][f'{metric}_std']
                    for r in npz_data['rf_radii']]
            npz_data[f'{mname}_{metric}_mean'] = np.array(means)
            npz_data[f'{mname}_{metric}_std'] = np.array(stds)

    for r in npz_data['rf_radii']:
        npz_data[f'rf_importances_r{r}'] = sweep_results[r]['rf_importances']
        npz_data[f'lr_coef_abs_r{r}'] = sweep_results[r]['lr_coef_abs']

    sample_r = npz_data['rf_radii'][0]
    npz_data['feature_names'] = np.array(sweep_results[sample_r]['feature_names'])

    npz_path = os.path.join(output_dir, 'downstream_reader_results.npz')
    np.savez_compressed(npz_path, **npz_data)
    print(f"  Saved NPZ: {npz_path}")


def save_ablation_csv(ablation_results, distance_results, output_dir):
    """Save ablation and distance analysis results as CSV."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'downstream_ablation_summary.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['analysis', 'group', 'accuracy_mean', 'accuracy_std',
                         'n_features'])

        if ablation_results:
            for cat in ['temporal', 'propagation', 'spatial', 'population', 'all']:
                if cat in ablation_results:
                    r = ablation_results[cat]
                    writer.writerow([
                        'feature_category', cat,
                        f"{r['accuracy_mean']:.4f}",
                        f"{r['accuracy_std']:.4f}",
                        r['n_features'],
                    ])

        if distance_results:
            for grp in ['center', 'near', 'far', 'population', 'all']:
                if grp in distance_results:
                    r = distance_results[grp]
                    writer.writerow([
                        'distance_group', grp,
                        f"{r['accuracy_mean']:.4f}",
                        f"{r['accuracy_std']:.4f}",
                        r['n_features'],
                    ])

    print(f"  Saved ablation CSV: {csv_path}")


def save_detection_csv(detection_results_single, detection_results_window,
                       stim_duration, output_dir, label_variant=''):
    """Save detection results as human-readable CSV.

    Accepts keys as either (cond, sz) or (cond, sz, mode, bias) tuples.

    Parameters
    ----------
    label_variant : str
        Label variant name (e.g. 'onset_onward', 'stim_only').
        Used for filename and as a column value.
    """
    os.makedirs(output_dir, exist_ok=True)
    suffix = f'_{label_variant}' if label_variant else ''
    csv_path = os.path.join(output_dir, f'detection_summary{suffix}.csv')

    stim_start = PRE_STIM_FRAMES
    stim_end = PRE_STIM_FRAMES + stim_duration

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'condition', 'stim_size', 'stim_mode', 'stim_bias',
            'label_variant', 'feature_mode',
            'peak_p_detected', 'peak_p_detected_sem',
            'mean_p_detected_stim', 'mean_p_detected_stim_sem',
            'onset_latency', 'decay_halflife',
            'overall_accuracy', 'overall_accuracy_std',
        ])

        for fmode, results_dict in [('single_frame', detection_results_single),
                                     ('windowed', detection_results_window)]:
            for key, res in sorted(results_dict.items(),
                                   key=lambda x: x[0]):
                # Handle both 2-tuple and 4-tuple keys
                if len(key) == 4:
                    cond, stim_size, mode, bias = key
                else:
                    cond, stim_size = key
                    mode, bias = 'clamped', None

                p_detected = np.asarray(res['p_detected'], dtype=float)
                p_detected_std = np.asarray(res['p_detected_std'], dtype=float)
                n_folds = res.get('n_folds', 5)

                # Compute baseline from pre-stimulus period
                baseline_mean = float(np.mean(p_detected[:stim_start]))

                # Peak detection probability during stimulus
                peak_frame = stim_start + np.argmax(p_detected[stim_start:stim_end])
                peak_p = float(np.max(p_detected[stim_start:stim_end]))
                peak_sem = float(p_detected_std[peak_frame] / np.sqrt(n_folds))

                # Mean detection probability during stimulus
                mean_p = float(np.mean(p_detected[stim_start:stim_end]))
                mean_sem = float(np.mean(p_detected_std[stim_start:stim_end])
                                 / np.sqrt(n_folds))

                # Onset latency: first frame above baseline + 2*std
                baseline_std = float(np.std(p_detected[:stim_start]))
                threshold = baseline_mean + 2 * baseline_std
                above = np.where(p_detected[stim_start:stim_end] > threshold)[0]
                onset_latency = int(above[0]) if len(above) > 0 else -1

                # Decay half-life: first frame after stim_end where p_detected
                # drops below half of peak elevation above baseline
                half_level = baseline_mean + (peak_p - baseline_mean) / 2
                post_stim = p_detected[stim_end:]
                below = np.where(post_stim < half_level)[0]
                decay_halflife = int(below[0]) if len(below) > 0 else -1

                bias_str = f"{bias}" if bias is not None else ""
                writer.writerow([
                    cond, stim_size, mode, bias_str,
                    label_variant, fmode,
                    f"{peak_p:.4f}", f"{peak_sem:.4f}",
                    f"{mean_p:.4f}", f"{mean_sem:.4f}",
                    onset_latency, decay_halflife,
                    f"{float(res['overall_accuracy']):.4f}",
                    f"{float(res['overall_accuracy_std']):.4f}",
                ])

    print(f"  Saved CSV: {csv_path}")


# =============================================================================
# CLI
# =============================================================================

def print_scan_report(comparison_path=None, ising_data_path=None,
                      stim_modes=None, stim_duration=DEFAULT_STIM_DURATION,
                      stim_sizes=None, stim_bias_values=None):
    """Print a report of what jobs would be run."""
    if stim_sizes is None:
        stim_sizes = DEFAULT_STIM_SIZES
    if stim_modes is None:
        stim_modes = [DEFAULT_STIM_MODE]

    if comparison_path is None:
        if ising_data_path is None:
            ising_data_path = r'IsingModelData_39x78_100K'
        try:
            comparison_path = find_comparison_results(ising_data_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

    best_matches = load_comparison_results(comparison_path)
    all_jobs = generate_jobs(best_matches, stim_modes, stim_duration, stim_sizes,
                             stim_bias_values=stim_bias_values)

    n_conditions = len([c for c in CONDITIONS if c in best_matches])
    n_samples = len(all_jobs) * N_REPLICATES

    print("\n" + "=" * 60)
    print("  Ising Downstream Reader - Job Plan")
    print("=" * 60)
    print(f"\nComparison results: {comparison_path}")
    print(f"Classification: Naive (0) vs Expert (1)")
    print(f"\nGrid: {L} x {M}, periodic boundaries")
    print(f"Stimulus center: ({CENTER_ROW}, {CENTER_COL})")
    print(f"Virtual neurons: {N_NEURONS} (1 center + {len(RING_DISTANCES)} rings x {len(RING_ANGLES_DEG)} dirs)")
    print(f"RF radii: {RF_RADII}")
    print(f"\nStimulus modes: {stim_modes}")
    if stim_bias_values:
        print(f"Bias values: {stim_bias_values}")
    print(f"Stimulus duration: {stim_duration} frames")
    print(f"Stimulus sizes: {stim_sizes}")
    print(f"Replicates per job: {N_REPLICATES}")
    print(f"\nConditions:")
    for condition in CONDITIONS:
        if condition in best_matches:
            print(f"  {condition}: {len(best_matches[condition]['simulations'])} simulations")

    # Per-mode job breakdown
    print(f"\nJobs per mode:")
    for mode in stim_modes:
        mode_jobs = [j for j in all_jobs if j['stim_mode'] == mode]
        print(f"  {mode}: {len(mode_jobs)} jobs")
        if mode == 'bias' and stim_bias_values:
            for bv in stim_bias_values:
                bv_jobs = [j for j in mode_jobs if j.get('stim_bias') == bv]
                print(f"    bias={bv}: {len(bv_jobs)} jobs")

    print(f"\nTotal jobs: {len(all_jobs)}")
    print(f"Total samples per RF radius: {n_samples}")
    n_mode_slots = sum(
        len(stim_bias_values) if (m == 'bias' and stim_bias_values) else 1
        for m in stim_modes
    )
    print(f"  = {n_conditions} cond x {N_TOP_MATCHES} sims x {len(stim_sizes)} sizes x {n_mode_slots} mode(s) x {N_REPLICATES} reps")
    print(f"\nFeature dimensions per RF radius:")
    print(f"  Per neuron: {N_TEMPORAL_FEATURES} temporal + {N_SPATIAL_FEATURES} spatial = {N_NEURON_FEATURES}")
    print(f"  Neuron features: {N_NEURONS} x {N_NEURON_FEATURES} = {N_NEURONS * N_NEURON_FEATURES}")
    print(f"  Population features: {N_POPULATION_FEATURES}")
    print(f"  Total: {N_TOTAL_FEATURES}")
    print(f"\nDataset shape per RF radius: X=({n_samples}, {N_TOTAL_FEATURES})")
    print(f"GroupKFold groups: {n_conditions * N_TOP_MATCHES}")
    print("\n" + "=" * 60)


def run_classify(output_dir, rf_radii=None):
    """
    Run classification sweep on pre-computed combined dataset.
    Loads from output_dir, runs classifiers, generates figures.
    """
    if rf_radii is None:
        rf_radii = RF_RADII

    # Load combined dataset
    dataset_pattern = os.path.join(output_dir, 'downstream_dataset_*.npz')
    dataset_files = sorted(glob(dataset_pattern))
    if not dataset_files:
        print(f"No dataset files found matching: {dataset_pattern}")
        print("Run --combine first to create the dataset.")
        return

    dataset_path = dataset_files[-1]  # Most recent
    print(f"Loading dataset: {dataset_path}")
    npz = np.load(dataset_path, allow_pickle=True)

    # Extract arrays into a plain dict (NpzFile holds file handles that
    # can't be pickled by joblib's loky backend)
    data = {k: npz[k] for k in npz.files}
    del npz

    # Run classification sweep
    print("\nRunning classification sweep over RF radii...")
    y = data['y']
    groups = data['groups']

    # Filter to radii that exist in dataset
    valid_radii = [r for r in rf_radii if f'X_r{r}' in data]
    missing = [r for r in rf_radii if f'X_r{r}' not in data]
    for r in missing:
        print(f"  RF radius {r} not found in dataset, skipping")

    # Pre-extract per-radius arrays (plain numpy, pickle-safe)
    X_by_r = {r: data[f'X_r{r}'] for r in valid_radii}
    fn_by_r = {}
    for r in valid_radii:
        fn = data[f'feature_names_r{r}']
        fn_by_r[r] = fn.tolist() if isinstance(fn, np.ndarray) else fn

    # Feature engineering: remove static spatial, add propagation features
    print("\nApplying feature engineering...")
    for r in valid_radii:
        X_by_r[r], fn_by_r[r] = remove_spatial_features(X_by_r[r], fn_by_r[r])
        X_by_r[r], fn_by_r[r] = augment_propagation_features(X_by_r[r], fn_by_r[r])

    def _classify_one_radius(r, X, fn):
        return r, run_classification(X, y, groups, fn)

    n_jobs = min(len(valid_radii), os.cpu_count() or 4)
    results_list = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_classify_one_radius)(r, X_by_r[r], fn_by_r[r])
        for r in valid_radii)

    sweep_results = {}
    for r, res in results_list:
        sweep_results[r] = res
        available = [m for m in ALL_MODEL_NAMES if m in res]
        best_acc = max(res[m]['accuracy_mean'] for m in available)
        print(f"  RF radius = {r} ... best accuracy = {best_acc:.3f}")

    if not sweep_results:
        print("No results to plot.")
        return

    # Flush loky executor to release /dev/shm semaphores before next Parallel
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)

    # Permutation test at optimal radius
    opt_r = find_optimal_radius(sweep_results)
    print(f"\nOptimal RF radius: {opt_r}")
    print("Running permutation test (100 shuffles)...")
    X_opt = X_by_r[opt_r]
    feature_names_opt = fn_by_r[opt_r]
    chance_accs = run_permutation_test(X_opt, y, groups)

    # Learning curve at optimal radius
    print("Computing learning curve...")
    lc_results = compute_learning_curve(X_opt, y, groups)

    # Per-stimulus-size classification (Panel E)
    sweep_by_stim = None
    stim_sizes_arr = None
    if 'stim_sizes' in data:
        stim_sizes_all = data['stim_sizes']
        unique_stim = sorted(np.unique(stim_sizes_all).astype(int).tolist())
        stim_sizes_arr = unique_stim
        print(f"\nRunning per-stimulus-size classification (sizes: {unique_stim})...")

        # Build (sz, r) pairs and run in parallel
        pairs = [(sz, r) for sz in unique_stim
                 for r in valid_radii]

        def _classify_sz_r(sz, r, X_r, fn_r):
            sz_mask = stim_sizes_all == sz
            if np.sum(sz_mask) == 0:
                return None
            X_sz = X_r[sz_mask]
            y_sz = y[sz_mask]
            g_sz = groups[sz_mask]
            if len(np.unique(y_sz)) < 2 or len(np.unique(g_sz)) < 2:
                return None
            return sz, r, run_classification(X_sz, y_sz, g_sz, fn_r)

        par_results = Parallel(n_jobs=-1, verbose=1)(
            delayed(_classify_sz_r)(sz, r, X_by_r[r], fn_by_r[r])
            for sz, r in pairs)

        sweep_by_stim = defaultdict(dict)
        for item in par_results:
            if item is not None:
                sz, r, res = item
                sweep_by_stim[sz][r] = res

        sweep_by_stim = dict(sweep_by_stim)  # convert back from defaultdict
        for sz in sorted(sweep_by_stim.keys()):
            best_r_sz = find_optimal_radius(sweep_by_stim[sz])
            best_acc_sz = sweep_by_stim[sz][best_r_sz]['RandomForest']['accuracy_mean']
            print(f"  stim_size={sz}: best RF acc={best_acc_sz:.3f} at r={best_r_sz}")

        if not sweep_by_stim:
            sweep_by_stim = None
            stim_sizes_arr = None
    else:
        print("\nNo stim_sizes in dataset — skipping Panel E. Re-run --combine to include.")

    # Per-stim-mode classification (Panel G)
    sweep_by_mode = None
    if 'stim_modes' in data:
        stim_modes_arr = data['stim_modes']
        # Convert bytes to str if needed
        if hasattr(stim_modes_arr[0], 'decode'):
            stim_modes_arr = np.array([s.decode() if isinstance(s, bytes) else str(s)
                                       for s in stim_modes_arr])
        else:
            stim_modes_arr = np.array([str(s) for s in stim_modes_arr])
        unique_modes = sorted(set(stim_modes_arr))
        if len(unique_modes) > 1:
            sweep_by_mode = run_stim_mode_classification(
                X_by_r, fn_by_r, y, groups, stim_modes_arr, valid_radii)
        else:
            print(f"\nOnly one stim mode ({unique_modes[0]}) — skipping Panel G.")
    else:
        print("\nNo stim_modes in dataset — skipping Panel G.")

    # Feature category ablation
    print("\nRunning feature category ablation at optimal radius...")
    ablation_results = run_feature_ablation(X_opt, y, groups, feature_names_opt)

    # Per-distance neuron analysis
    print("Running per-distance neuron analysis at optimal radius...")
    distance_results = run_distance_ablation(X_opt, y, groups, feature_names_opt)

    # Save and plot
    fig_dir = os.path.join(output_dir, 'figures')
    save_results(sweep_results, output_dir, sweep_by_mode=sweep_by_mode)
    plot_all(sweep_results, fig_dir, chance_accs=chance_accs,
             sweep_by_stim=sweep_by_stim,
             stim_sizes=stim_sizes_arr,
             lc_results=lc_results,
             sweep_by_mode=sweep_by_mode)

    # Ablation and distance plots
    if ablation_results:
        plot_ablation_bar(ablation_results, fig_dir)
    if distance_results:
        plot_distance_ablation_bar(distance_results, fig_dir)

    save_ablation_csv(ablation_results, distance_results, output_dir)

    print("\nDone!")


def run_detect_one(args, job_index):
    """Run a single detect grouped job by index (for SLURM array)."""
    os.makedirs(args.output, exist_ok=True)
    detect_dir = os.path.join(args.output, 'detection')
    os.makedirs(detect_dir, exist_ok=True)

    # Load best matches
    if args.comparison is None:
        comparison_path = find_comparison_results(args.ising_data)
    else:
        comparison_path = args.comparison
    best_matches = load_comparison_results(comparison_path)

    # Resolve stimulus modes
    if args.stim_modes:
        stim_modes = args.stim_modes
    else:
        stim_modes = [args.stim_mode]

    if 'bias' in stim_modes:
        stim_bias_values = args.stim_bias or [1.0, 2.0, 4.0]
    else:
        stim_bias_values = None

    # Generate grouped jobs
    grouped_jobs = generate_grouped_jobs(
        best_matches, stim_modes, args.stim_duration, args.stim_sizes,
        stim_bias_values=stim_bias_values)

    if job_index < 0 or job_index >= len(grouped_jobs):
        print(f"ERROR: detect-index {job_index} out of range "
              f"(0-{len(grouped_jobs)-1})")
        sys.exit(1)

    group = grouped_jobs[job_index]
    condition = group['condition']
    sim_idx = group['sim_idx']
    stim_duration = group['stim_duration']

    config_dict = {
        'detect_rf_radius': args.detect_rf_radius,
        'detect_window': args.detect_window,
        'queen': queen,
    }

    print(f"\nDetect job {job_index}: {condition} sim{sim_idx}")
    print(f"  Stim configs: {len(group['stim_configs'])}")

    start = time.time()
    results_list = run_detection_job_grouped((group, config_dict))
    elapsed = time.time() - start

    total_frames = PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES

    # Save per-group .npz
    save_dict = {
        'condition': condition,
        'sim_idx': sim_idx,
        'n_configs': len(results_list),
        'total_frames': total_frames,
    }
    for i, r in enumerate(results_list):
        save_dict[f'cfg{i}_X_single'] = r['X_single']
        save_dict[f'cfg{i}_X_window'] = r['X_window']
        save_dict[f'cfg{i}_y_onset'] = r['y_onset']
        save_dict[f'cfg{i}_y_stim'] = r['y_stim']
        save_dict[f'cfg{i}_trial_type'] = r['trial_type']
        save_dict[f'cfg{i}_frame_indices'] = r['frame_indices']
        save_dict[f'cfg{i}_sim_indices'] = r['sim_indices']
        save_dict[f'cfg{i}_stim_size'] = r['stim_size']
        save_dict[f'cfg{i}_stim_mode'] = r['stim_mode']
        if r.get('stim_bias') is not None:
            save_dict[f'cfg{i}_stim_bias'] = r['stim_bias']

    out_path = os.path.join(detect_dir,
                            f'detect_job_{condition}_sim{sim_idx}.npz')
    np.savez_compressed(out_path, **save_dict)
    print(f"  Saved: {out_path} ({elapsed/60:.1f} min)")


def run_detect_combine(args):
    """Load detect-sim results, aggregate, classify per stim_size, plot."""
    detect_dir = os.path.join(args.output, 'detection')
    fig_dir = os.path.join(detect_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Load all per-group .npz files
    pattern = os.path.join(detect_dir, 'detect_job_*.npz')
    result_files = sorted(glob(pattern))
    if not result_files:
        print(f"ERROR: No detect job files found in {detect_dir}")
        sys.exit(1)

    print(f"\nLoading {len(result_files)} detect job files...")

    results_by_cond = defaultdict(list)
    total_frames = None

    for fpath in result_files:
        data = np.load(fpath, allow_pickle=True)
        condition = str(data['condition'])
        sim_idx = int(data['sim_idx'])
        n_configs = int(data['n_configs'])
        if total_frames is None:
            total_frames = int(data['total_frames'])

        for i in range(n_configs):
            bias_key = f'cfg{i}_stim_bias'
            # Backward compat: old files have cfg{i}_y instead of new keys
            if f'cfg{i}_y_onset' in data:
                y_onset = data[f'cfg{i}_y_onset']
                y_stim = data[f'cfg{i}_y_stim']
                trial_type = data[f'cfg{i}_trial_type']
            else:
                old_y = data[f'cfg{i}_y']
                y_onset = old_y
                y_stim = old_y
                trial_type = old_y
            result = {
                'X_single': data[f'cfg{i}_X_single'],
                'X_window': data[f'cfg{i}_X_window'],
                'y_onset': y_onset,
                'y_stim': y_stim,
                'trial_type': trial_type,
                'frame_indices': data[f'cfg{i}_frame_indices'],
                'sim_indices': data[f'cfg{i}_sim_indices'],
                'stim_size': int(data[f'cfg{i}_stim_size']),
                'stim_mode': str(data[f'cfg{i}_stim_mode']),
                'stim_bias': float(data[bias_key]) if bias_key in data else None,
                'condition': condition,
                'sim_idx': sim_idx,
            }
            results_by_cond[condition].append(result)
        data.close()
        print(f"  Loaded: {os.path.basename(fpath)} "
              f"({condition} sim{sim_idx}, {n_configs} configs)")

    stim_sizes = args.stim_sizes
    stim_duration = args.stim_duration
    rf_radius = args.detect_rf_radius
    window_half = args.detect_window

    # Aggregate and classify per (condition, stim_size, stim_mode, stim_bias)
    cond_config_data = defaultdict(list)
    for cond in CONDITIONS:
        if cond not in results_by_cond:
            continue
        for result in results_by_cond[cond]:
            key = (cond, result['stim_size'], result['stim_mode'],
                   result.get('stim_bias'))
            cond_config_data[key].append(result)

    classify_jobs = []
    label_variants = [('onset_onward', 'y_onset'), ('stim_only', 'y_stim')]
    for (cond, stim_size, mode, bias), results in cond_config_data.items():
        X_s = np.vstack([r['X_single'] for r in results])
        X_w = np.vstack([r['X_window'] for r in results])
        fr = np.concatenate([r['frame_indices'] for r in results])
        si = np.concatenate([r['sim_indices'] for r in results])
        tt = np.concatenate([r['trial_type'] for r in results])
        bias_str = f" bias={bias}" if bias is not None else ""
        for variant_name, y_key in label_variants:
            y_c = np.concatenate([r[y_key] for r in results])
            print(f"  {cond} size={stim_size} mode={mode}{bias_str} "
                  f"({variant_name}): {X_s.shape[0]} samples "
                  f"(y=1: {np.sum(y_c==1)}, y=0: {np.sum(y_c==0)})")
            classify_jobs.append((cond, stim_size, mode, bias, 'single',
                                  variant_name, X_s, y_c, tt, fr, si))
            classify_jobs.append((cond, stim_size, mode, bias, 'window',
                                  variant_name, X_w, y_c, tt, fr, si))

    # Free intermediate data before launching workers
    del results_by_cond, cond_config_data
    gc.collect()

    def _detect_classify_one(cond, sz, mode, bias, ftype, variant, X, y_c,
                             tt, frames, sims, tf):
        return (cond, sz, mode, bias, ftype, variant,
                run_detection_classification(X, y_c, frames, sims, tf,
                                             trial_type=tt))

    n_par = min(len(classify_jobs), 4)
    print(f"\n  Running {len(classify_jobs)} detection classifiers "
          f"in parallel (n_jobs={n_par})...")
    par_results = Parallel(n_jobs=n_par, verbose=1)(
        delayed(_detect_classify_one)(c, sz, m, b, ft, vn, X, y_c, tt, fr,
                                      si, total_frames)
        for c, sz, m, b, ft, vn, X, y_c, tt, fr, si in classify_jobs)

    # Collect results: {(cond, sz, mode, bias, variant): result_dict}
    # Separate into single/window dicts per variant
    detection_results_by_variant = {}
    for variant_name, _ in label_variants:
        detection_results_by_variant[variant_name] = {
            'single': {}, 'window': {}}

    for cond, sz, mode, bias, ftype, variant, res in par_results:
        if res is None:
            continue
        key = (cond, sz, mode, bias)
        detection_results_by_variant[variant][ftype][key] = res
        bias_str = f" bias={bias}" if bias is not None else ""
        print(f"    {cond} size={sz} mode={mode}{bias_str} "
              f"({ftype}, {variant}): acc={res['overall_accuracy']:.3f}")

    # --- Per-variant plotting, CSV, save ---
    for variant_name, _ in label_variants:
        v_single = detection_results_by_variant[variant_name]['single']
        v_window = detection_results_by_variant[variant_name]['window']
        v_fig_dir = os.path.join(fig_dir, variant_name)

        all_modes = sorted(set(k[2] for k in
                               list(v_single.keys()) + list(v_window.keys())))

        for mode in all_modes:
            mode_fig_dir = os.path.join(v_fig_dir, mode)
            bias_values = sorted(set(
                k[3] for k in list(v_single.keys()) + list(v_window.keys())
                if k[2] == mode and k[3] is not None))

            if mode == 'bias' and bias_values:
                for bias_val in bias_values:
                    for stim_size in stim_sizes:
                        for ftype, results_dict in [
                                ('single_frame', v_single),
                                ('windowed', v_window)]:
                            size_results = {
                                c: results_dict[(c, stim_size, mode, bias_val)]
                                for c in CONDITIONS
                                if (c, stim_size, mode, bias_val) in results_dict}
                            if size_results:
                                plot_stimulus_detection(
                                    size_results, stim_duration, mode_fig_dir,
                                    f'{ftype}_size{stim_size}_bias{bias_val}')
                plot_detection_vs_bias(
                    v_single, v_window,
                    stim_sizes, bias_values, stim_duration, mode_fig_dir)
            else:
                single_view = {(k[0], k[1]): v
                               for k, v in v_single.items()
                               if k[2] == mode and k[3] is None}
                window_view = {(k[0], k[1]): v
                               for k, v in v_window.items()
                               if k[2] == mode and k[3] is None}

                for stim_size in stim_sizes:
                    for ftype, results_dict in [('single_frame', single_view),
                                                 ('windowed', window_view)]:
                        size_results = {c: results_dict[(c, stim_size)]
                                       for c in CONDITIONS
                                       if (c, stim_size) in results_dict}
                        if size_results:
                            plot_stimulus_detection(
                                size_results, stim_duration, mode_fig_dir,
                                f'{ftype}_size{stim_size}')

                if single_view or window_view:
                    plot_detection_vs_stim_size(
                        single_view, window_view,
                        stim_sizes, stim_duration, mode_fig_dir,
                        title_suffix=f' ({mode})' if mode != 'clamped' else '',
                        fname_suffix='')

        save_detection_csv(v_single, v_window,
                           stim_duration, detect_dir,
                           label_variant=variant_name)

    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dict = {
        'stim_duration': stim_duration,
        'stim_sizes': np.array(stim_sizes),
        'rf_radius': rf_radius,
        'window_half': window_half,
        'total_frames': total_frames,
    }
    for variant_name, _ in label_variants:
        v_single = detection_results_by_variant[variant_name]['single']
        v_window = detection_results_by_variant[variant_name]['window']
        for (cond, sz, mode, bias), res in v_single.items():
            bias_tag = f'_bias{bias}' if bias is not None else ''
            for key, val in res.items():
                save_dict[f'{variant_name}_{cond}_size{sz}_{mode}{bias_tag}_single_{key}'] = val
        for (cond, sz, mode, bias), res in v_window.items():
            bias_tag = f'_bias{bias}' if bias is not None else ''
            for key, val in res.items():
                save_dict[f'{variant_name}_{cond}_size{sz}_{mode}{bias_tag}_window_{key}'] = val

    out_path = os.path.join(detect_dir, f'detection_results_{timestamp}.npz')
    np.savez_compressed(out_path, **save_dict)
    print(f"\n  Saved: {out_path}")
    print("  Detection analysis complete!")


def build_detect_clf_index_map(detect_dir):
    """Build (condition, stim_size, stim_mode, stim_bias) index map from detect_job files.

    Scans all detect_job_*.npz files and collects unique classify keys.
    Returns a sorted list of (condition, stim_size, stim_mode, stim_bias) tuples.
    For clamped mode, stim_bias will be None.
    """
    pattern = os.path.join(detect_dir, 'detect_job_*.npz')
    result_files = sorted(glob(pattern))

    seen_keys = set()
    index_map = []

    for fpath in result_files:
        data = np.load(fpath, allow_pickle=True)
        condition = str(data['condition'])
        n_configs = int(data['n_configs'])
        for i in range(n_configs):
            sz = int(data[f'cfg{i}_stim_size'])
            mode = str(data[f'cfg{i}_stim_mode'])
            bias_key = f'cfg{i}_stim_bias'
            bias = float(data[bias_key]) if bias_key in data else None
            clf_key = (condition, sz, mode, bias)
            if clf_key not in seen_keys:
                seen_keys.add(clf_key)
                index_map.append(clf_key)
        data.close()

    # Sort for deterministic ordering: by condition, size, mode, bias
    def _sort_key(item):
        return (item[0], item[1], item[2], item[3] if item[3] is not None else -1)
    index_map.sort(key=_sort_key)
    return index_map


def run_detect_classify_job(args):
    """Run classification for a single (condition, stim_size, stim_mode, stim_bias) tuple.

    Designed for SLURM array: --detect-classify-job --index <N>.
    Dynamically builds the index map from detect_job files, then loads
    matching configs, vstacks, runs classification (single + window),
    and saves a per-tuple .npz file.
    """
    idx = args.index
    detect_dir = os.path.join(args.output, 'detection')
    os.makedirs(detect_dir, exist_ok=True)

    index_map = build_detect_clf_index_map(detect_dir)
    if idx is None or idx < 0 or idx >= len(index_map):
        print(f"ERROR: --index must be 0-{len(index_map)-1} "
              f"for --detect-classify-job (got: {idx})")
        sys.exit(1)

    condition, stim_size, stim_mode, stim_bias = index_map[idx]
    bias_str = f" bias={stim_bias}" if stim_bias is not None else ""
    print(f"\nDetect classify job {idx}: {condition} size={stim_size} "
          f"mode={stim_mode}{bias_str}")

    # Load detect_job files for this condition
    pattern = os.path.join(detect_dir, f'detect_job_{condition}_sim*.npz')
    result_files = sorted(glob(pattern))
    if not result_files:
        print(f"ERROR: No detect job files found for {condition} in {detect_dir}")
        sys.exit(1)
    print(f"  Loading {len(result_files)} detect job files for {condition}...")

    total_frames = None
    matching_results = []

    for fpath in result_files:
        data = np.load(fpath, allow_pickle=True)
        if total_frames is None:
            total_frames = int(data['total_frames'])
        n_configs = int(data['n_configs'])

        for i in range(n_configs):
            sz = int(data[f'cfg{i}_stim_size'])
            if sz != stim_size:
                continue
            mode = str(data[f'cfg{i}_stim_mode'])
            if mode != stim_mode:
                continue
            bias_key = f'cfg{i}_stim_bias'
            bias = float(data[bias_key]) if bias_key in data else None
            if bias != stim_bias:
                continue
            # Backward compat: old files have cfg{i}_y
            if f'cfg{i}_y_onset' in data:
                y_onset = data[f'cfg{i}_y_onset']
                y_stim = data[f'cfg{i}_y_stim']
                trial_type = data[f'cfg{i}_trial_type']
            else:
                old_y = data[f'cfg{i}_y']
                y_onset = old_y
                y_stim = old_y
                trial_type = old_y
            matching_results.append({
                'X_single': data[f'cfg{i}_X_single'],
                'X_window': data[f'cfg{i}_X_window'],
                'y_onset': y_onset,
                'y_stim': y_stim,
                'trial_type': trial_type,
                'frame_indices': data[f'cfg{i}_frame_indices'],
                'sim_indices': data[f'cfg{i}_sim_indices'],
            })
        data.close()

    if not matching_results:
        print(f"  WARNING: No configs found for {condition} size={stim_size} "
              f"mode={stim_mode}{bias_str}")
        sys.exit(0)

    # Vstack
    X_single = np.vstack([r['X_single'] for r in matching_results])
    X_window = np.vstack([r['X_window'] for r in matching_results])
    trial_type = np.concatenate([r['trial_type'] for r in matching_results])
    frame_indices = np.concatenate([r['frame_indices'] for r in matching_results])
    sim_indices = np.concatenate([r['sim_indices'] for r in matching_results])

    label_arrays = {
        'onset_onward': np.concatenate([r['y_onset'] for r in matching_results]),
        'stim_only': np.concatenate([r['y_stim'] for r in matching_results]),
    }
    del matching_results
    gc.collect()

    start = time.time()

    # Save
    save_dict = {
        'condition': condition,
        'stim_size': stim_size,
        'stim_mode': stim_mode,
        'total_frames': total_frames,
    }
    if stim_bias is not None:
        save_dict['stim_bias'] = stim_bias

    # Run classification for both label variants x both feature types
    for variant_name, y_arr in label_arrays.items():
        print(f"  {condition} size={stim_size} mode={stim_mode}{bias_str} "
              f"({variant_name}): {X_single.shape[0]} samples "
              f"(y=1: {np.sum(y_arr==1)}, y=0: {np.sum(y_arr==0)})")

        print(f"  Classifying single-frame features ({variant_name})...")
        res_single = run_detection_classification(
            X_single, y_arr, frame_indices, sim_indices, total_frames,
            trial_type=trial_type)
        if res_single:
            print(f"    acc={res_single['overall_accuracy']:.3f}")
            for key, val in res_single.items():
                save_dict[f'single_{variant_name}_{key}'] = val

        print(f"  Classifying windowed features ({variant_name})...")
        res_window = run_detection_classification(
            X_window, y_arr, frame_indices, sim_indices, total_frames,
            trial_type=trial_type)
        if res_window:
            print(f"    acc={res_window['overall_accuracy']:.3f}")
            for key, val in res_window.items():
                save_dict[f'window_{variant_name}_{key}'] = val

    del X_single, X_window
    gc.collect()

    elapsed = time.time() - start

    # Filename: detect_clf_{cond}_size{sz}_{mode}[_bias{b}].npz
    fname = f'detect_clf_{condition}_size{stim_size}_{stim_mode}'
    if stim_bias is not None:
        fname += f'_bias{stim_bias}'
    fname += '.npz'
    out_path = os.path.join(detect_dir, fname)
    np.savez_compressed(out_path, **save_dict)
    print(f"  Saved: {out_path} ({elapsed/60:.1f} min)")


def run_detect_aggregate(args):
    """Aggregate detect-clf results, generate plots, save combined file.

    Loads all detect_clf_*.npz files produced by --detect-classify-job,
    groups by (condition, stim_size, stim_mode, stim_bias), then runs
    plotting and saving.  Backward-compatible: files without stim_mode
    are treated as mode='clamped', bias=None.
    """
    detect_dir = os.path.join(args.output, 'detection')
    fig_dir = os.path.join(detect_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    pattern = os.path.join(detect_dir, 'detect_clf_*.npz')
    clf_files = sorted(glob(pattern))
    if not clf_files:
        print(f"ERROR: No detect_clf_*.npz files found in {detect_dir}")
        sys.exit(1)

    print(f"\nLoading {len(clf_files)} classification result files...")

    stim_sizes = args.stim_sizes
    stim_duration = args.stim_duration
    rf_radius = args.detect_rf_radius
    window_half = args.detect_window
    total_frames = None

    # Collect results per variant: {variant: {'single': {key: res}, 'window': {key: res}}}
    label_variants = ['onset_onward', 'stim_only']
    detection_results_by_variant = {v: {'single': {}, 'window': {}}
                                    for v in label_variants}

    for fpath in clf_files:
        data = np.load(fpath, allow_pickle=True)
        cond = str(data['condition'])
        sz = int(data['stim_size'])
        mode = str(data['stim_mode']) if 'stim_mode' in data else 'clamped'
        bias = float(data['stim_bias']) if 'stim_bias' in data else None
        if total_frames is None:
            total_frames = int(data['total_frames'])

        bias_str = f" bias={bias}" if bias is not None else ""
        clf_key = (cond, sz, mode, bias)

        # New format: keys like single_onset_onward_overall_accuracy
        # Old format: keys like single_overall_accuracy
        has_new_format = any(k.startswith('single_onset_onward_')
                             for k in data.files)

        if has_new_format:
            for variant in label_variants:
                for ftype in ['single', 'window']:
                    prefix = f'{ftype}_{variant}_'
                    res = {}
                    for key in data.files:
                        if key.startswith(prefix):
                            res[key[len(prefix):]] = data[key]
                    if res and 'overall_accuracy' in res:
                        detection_results_by_variant[variant][ftype][clf_key] = res
                        print(f"  {cond} size={sz} mode={mode}{bias_str} "
                              f"({ftype}, {variant}): "
                              f"acc={float(res['overall_accuracy']):.3f}")
        else:
            # Old format: treat as onset_onward (backward compat)
            for ftype in ['single', 'window']:
                prefix = f'{ftype}_'
                res = {}
                for key in data.files:
                    if key.startswith(prefix):
                        res[key[len(prefix):]] = data[key]
                if res and 'overall_accuracy' in res:
                    detection_results_by_variant['onset_onward'][ftype][clf_key] = res
                    print(f"  {cond} size={sz} mode={mode}{bias_str} "
                          f"({ftype}, onset_onward [compat]): "
                          f"acc={float(res['overall_accuracy']):.3f}")

        data.close()

    # --- Per-variant plotting, CSV, save ---
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dict = {
        'stim_duration': stim_duration,
        'stim_sizes': np.array(stim_sizes),
        'rf_radius': rf_radius,
        'window_half': window_half,
        'total_frames': total_frames,
    }

    for variant in label_variants:
        v_single = detection_results_by_variant[variant]['single']
        v_window = detection_results_by_variant[variant]['window']
        if not v_single and not v_window:
            continue

        v_fig_dir = os.path.join(fig_dir, variant)

        all_modes = sorted(set(k[2] for k in
                               list(v_single.keys()) + list(v_window.keys())))

        for mode in all_modes:
            mode_fig_dir = os.path.join(v_fig_dir, mode)
            bias_values = sorted(set(
                k[3] for k in list(v_single.keys()) + list(v_window.keys())
                if k[2] == mode and k[3] is not None))

            if mode == 'bias' and bias_values:
                for bias_val in bias_values:
                    for stim_size in stim_sizes:
                        for ftype, results_dict in [
                                ('single_frame', v_single),
                                ('windowed', v_window)]:
                            size_results = {
                                c: results_dict[(c, stim_size, mode, bias_val)]
                                for c in CONDITIONS
                                if (c, stim_size, mode, bias_val) in results_dict}
                            if size_results:
                                plot_stimulus_detection(
                                    size_results, stim_duration, mode_fig_dir,
                                    f'{ftype}_size{stim_size}_bias{bias_val}')

                plot_detection_vs_bias(
                    v_single, v_window,
                    stim_sizes, bias_values, stim_duration, mode_fig_dir)
            else:
                single_view = {(k[0], k[1]): v
                               for k, v in v_single.items()
                               if k[2] == mode and k[3] is None}
                window_view = {(k[0], k[1]): v
                               for k, v in v_window.items()
                               if k[2] == mode and k[3] is None}

                for stim_size in stim_sizes:
                    for ftype, results_dict in [('single_frame', single_view),
                                                 ('windowed', window_view)]:
                        size_results = {c: results_dict[(c, stim_size)]
                                       for c in CONDITIONS
                                       if (c, stim_size) in results_dict}
                        if size_results:
                            plot_stimulus_detection(
                                size_results, stim_duration, mode_fig_dir,
                                f'{ftype}_size{stim_size}')

                if single_view or window_view:
                    plot_detection_vs_stim_size(
                        single_view, window_view,
                        stim_sizes, stim_duration, mode_fig_dir,
                        title_suffix=f' ({mode})' if mode != 'clamped' else '',
                        fname_suffix='')

        save_detection_csv(v_single, v_window,
                           stim_duration, detect_dir,
                           label_variant=variant)

        # Add to combined save dict
        for (cond, sz, mode, bias), res in v_single.items():
            bias_tag = f'_bias{bias}' if bias is not None else ''
            for key, val in res.items():
                save_dict[f'{variant}_{cond}_size{sz}_{mode}{bias_tag}_single_{key}'] = val
        for (cond, sz, mode, bias), res in v_window.items():
            bias_tag = f'_bias{bias}' if bias is not None else ''
            for key, val in res.items():
                save_dict[f'{variant}_{cond}_size{sz}_{mode}{bias_tag}_window_{key}'] = val

    out_path = os.path.join(detect_dir, f'detection_results_{timestamp}.npz')
    np.savez_compressed(out_path, **save_dict)
    print(f"\n  Saved: {out_path}")
    print("  Detection aggregation complete!")


def run_detect(args):
    """
    Run stimulus detection analysis (local all-in-one mode).

    Loads best-match params, generates detection jobs, runs them,
    classifies, and plots results.
    """
    os.makedirs(args.output, exist_ok=True)
    detect_dir = os.path.join(args.output, 'detection')
    fig_dir = os.path.join(detect_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Load best matches
    if args.comparison is None:
        comparison_path = find_comparison_results(args.ising_data)
    else:
        comparison_path = args.comparison

    best_matches = load_comparison_results(comparison_path)

    # Resolve stimulus modes
    if args.stim_modes:
        stim_modes = args.stim_modes
    else:
        stim_modes = [args.stim_mode]

    if 'bias' in stim_modes:
        stim_bias_values = args.stim_bias or [1.0, 2.0, 4.0]
    else:
        stim_bias_values = None

    stim_duration = args.stim_duration
    stim_sizes = args.stim_sizes
    rf_radius = args.detect_rf_radius
    window_half = args.detect_window

    config_dict = {
        'detect_rf_radius': rf_radius,
        'detect_window': window_half,
        'queen': queen,
    }

    print(f"\nStimulus Detection Analysis")
    print(f"  RF radius: {rf_radius}")
    print(f"  Window half-width: {window_half} ({2*window_half+1} frames)")
    print(f"  Stim modes: {stim_modes}")
    print(f"  Stim duration: {stim_duration}")
    print(f"  Stim sizes: {stim_sizes}")

    # Generate GROUPED jobs (one per condition-sim pair)
    grouped_jobs = generate_grouped_jobs(
        best_matches, stim_modes, stim_duration, stim_sizes,
        stim_bias_values=stim_bias_values)

    n_stim_configs = len(grouped_jobs[0]['stim_configs']) if grouped_jobs else 0
    print(f"  Grouped jobs: {len(grouped_jobs)} "
          f"({n_stim_configs} stim configs each)")

    args_list = [(gj, config_dict) for gj in grouped_jobs]

    start_time = time.time()
    results_by_cond = defaultdict(list)

    n_workers = args.workers
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for results_list in pool.imap(run_detection_job_grouped, args_list):
                for result in results_list:
                    results_by_cond[result['condition']].append(result)
                cond = results_list[0]['condition']
                sim = results_list[0]['sim_idx']
                print(f"    Done: {cond} sim{sim} "
                      f"({len(results_list)} configs)")
    else:
        for job_args in args_list:
            results_list = run_detection_job_grouped(job_args)
            for result in results_list:
                results_by_cond[result['condition']].append(result)
            cond = results_list[0]['condition']
            sim = results_list[0]['sim_idx']
            print(f"    Done: {cond} sim{sim} "
                  f"({len(results_list)} configs)")

    elapsed = time.time() - start_time
    print(f"\n  Simulation completed in {elapsed/60:.1f} minutes")

    # Aggregate and classify per (condition, stim_size, stim_mode, stim_bias)
    total_frames = PRE_STIM_FRAMES + stim_duration + POST_STIM_FRAMES

    # Pre-aggregate per (condition, stim_size, stim_mode, stim_bias)
    cond_config_data = defaultdict(list)
    for cond in CONDITIONS:
        if cond not in results_by_cond:
            continue
        for result in results_by_cond[cond]:
            key = (cond, result['stim_size'], result['stim_mode'],
                   result.get('stim_bias'))
            cond_config_data[key].append(result)

    # Build classification jobs for both label variants
    label_variants = [('onset_onward', 'y_onset'), ('stim_only', 'y_stim')]
    classify_jobs = []
    for (cond, stim_size, mode, bias), results in cond_config_data.items():
        X_s = np.vstack([r['X_single'] for r in results])
        X_w = np.vstack([r['X_window'] for r in results])
        fr = np.concatenate([r['frame_indices'] for r in results])
        si = np.concatenate([r['sim_indices'] for r in results])
        tt = np.concatenate([r['trial_type'] for r in results])
        bias_str = f" bias={bias}" if bias is not None else ""
        for variant_name, y_key in label_variants:
            y_c = np.concatenate([r[y_key] for r in results])
            print(f"  {cond} size={stim_size} mode={mode}{bias_str} "
                  f"({variant_name}): {X_s.shape[0]} samples "
                  f"(y=1: {np.sum(y_c==1)}, y=0: {np.sum(y_c==0)})")
            classify_jobs.append((cond, stim_size, mode, bias, 'single',
                                  variant_name, X_s, y_c, tt, fr, si))
            classify_jobs.append((cond, stim_size, mode, bias, 'window',
                                  variant_name, X_w, y_c, tt, fr, si))

    # Free intermediate data before launching workers
    del results_by_cond, cond_config_data
    gc.collect()

    def _detect_classify_one(cond, sz, mode, bias, ftype, variant, X, y_c,
                             tt, frames, sims, tf):
        return (cond, sz, mode, bias, ftype, variant,
                run_detection_classification(X, y_c, frames, sims, tf,
                                             trial_type=tt))

    n_par = min(len(classify_jobs), 4)
    print(f"\n  Running {len(classify_jobs)} detection classifiers "
          f"in parallel (n_jobs={n_par})...")
    par_results = Parallel(n_jobs=n_par, verbose=1)(
        delayed(_detect_classify_one)(c, sz, m, b, ft, vn, X, y_c, tt, fr,
                                      si, total_frames)
        for c, sz, m, b, ft, vn, X, y_c, tt, fr, si in classify_jobs)

    # Collect results per variant
    detection_results_by_variant = {}
    for variant_name, _ in label_variants:
        detection_results_by_variant[variant_name] = {
            'single': {}, 'window': {}}

    for cond, sz, mode, bias, ftype, variant, res in par_results:
        if res is None:
            continue
        key = (cond, sz, mode, bias)
        detection_results_by_variant[variant][ftype][key] = res
        bias_str = f" bias={bias}" if bias is not None else ""
        print(f"    {cond} size={sz} mode={mode}{bias_str} "
              f"({ftype}, {variant}): acc={res['overall_accuracy']:.3f}")

    # --- Per-variant plotting, CSV, save ---
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dict = {
        'stim_duration': stim_duration,
        'stim_sizes': np.array(stim_sizes),
        'rf_radius': rf_radius,
        'window_half': window_half,
        'total_frames': total_frames,
    }

    for variant_name, _ in label_variants:
        v_single = detection_results_by_variant[variant_name]['single']
        v_window = detection_results_by_variant[variant_name]['window']
        v_fig_dir = os.path.join(fig_dir, variant_name)

        all_modes = sorted(set(k[2] for k in
                               list(v_single.keys()) + list(v_window.keys())))

        for mode in all_modes:
            mode_fig_dir = os.path.join(v_fig_dir, mode)
            bias_values = sorted(set(
                k[3] for k in list(v_single.keys()) + list(v_window.keys())
                if k[2] == mode and k[3] is not None))

            if mode == 'bias' and bias_values:
                for bias_val in bias_values:
                    for stim_size in stim_sizes:
                        for ftype, results_dict in [
                                ('single_frame', v_single),
                                ('windowed', v_window)]:
                            size_results = {
                                c: results_dict[(c, stim_size, mode, bias_val)]
                                for c in CONDITIONS
                                if (c, stim_size, mode, bias_val) in results_dict}
                            if size_results:
                                plot_stimulus_detection(
                                    size_results, stim_duration, mode_fig_dir,
                                    f'{ftype}_size{stim_size}_bias{bias_val}')
                plot_detection_vs_bias(
                    v_single, v_window,
                    stim_sizes, bias_values, stim_duration, mode_fig_dir)
            else:
                single_view = {(k[0], k[1]): v
                               for k, v in v_single.items()
                               if k[2] == mode and k[3] is None}
                window_view = {(k[0], k[1]): v
                               for k, v in v_window.items()
                               if k[2] == mode and k[3] is None}

                for stim_size in stim_sizes:
                    for ftype, results_dict in [('single_frame', single_view),
                                                 ('windowed', window_view)]:
                        size_results = {c: results_dict[(c, stim_size)]
                                       for c in CONDITIONS
                                       if (c, stim_size) in results_dict}
                        if size_results:
                            plot_stimulus_detection(
                                size_results, stim_duration, mode_fig_dir,
                                f'{ftype}_size{stim_size}')

                if single_view or window_view:
                    plot_detection_vs_stim_size(
                        single_view, window_view,
                        stim_sizes, stim_duration, mode_fig_dir,
                        title_suffix=f' ({mode})' if mode != 'clamped' else '',
                        fname_suffix='')

        save_detection_csv(v_single, v_window,
                           stim_duration, detect_dir,
                           label_variant=variant_name)

        for (cond, sz, mode, bias), res in v_single.items():
            bias_tag = f'_bias{bias}' if bias is not None else ''
            for key, val in res.items():
                save_dict[f'{variant_name}_{cond}_size{sz}_{mode}{bias_tag}_single_{key}'] = val
        for (cond, sz, mode, bias), res in v_window.items():
            bias_tag = f'_bias{bias}' if bias is not None else ''
            for key, val in res.items():
                save_dict[f'{variant_name}_{cond}_size{sz}_{mode}{bias_tag}_window_{key}'] = val

    out_path = os.path.join(detect_dir, f'detection_results_{timestamp}.npz')
    np.savez_compressed(out_path, **save_dict)
    print(f"\n  Saved: {out_path}")
    print("  Detection analysis complete!")


def combine_job_results(output_dir):
    """Combine per-job .npz files into a single dataset file."""
    pattern = os.path.join(output_dir, 'simulate', 'downstream_job_*.npz')
    result_files = sorted(glob(pattern))

    if not result_files:
        # Fallback: check root for files from older runs
        pattern_legacy = os.path.join(output_dir, 'downstream_job_*.npz')
        result_files = sorted(glob(pattern_legacy))
        if result_files:
            print(f"  [compat] Found {len(result_files)} job files in root (legacy layout)")

    if not result_files:
        print(f"No result files found matching: {pattern}")
        return

    print(f"Found {len(result_files)} result files")

    # Accumulate per RF radius
    X_by_radius = defaultdict(list)
    y_parts = []
    g_parts = []
    stim_sizes_parts = []
    conditions_parts = []
    stim_modes_parts = []
    feature_names_by_radius = {}

    for fpath in result_files:
        data = np.load(fpath, allow_pickle=True)
        n_samples = len(data['labels'])
        y_parts.append(data['labels'])
        g_parts.append(data['groups'])

        # Preserve stim_size per sample (Fix 1)
        if 'stim_size' in data:
            stim_sz = data['stim_size']
            if np.ndim(stim_sz) == 0:
                stim_sizes_parts.append(np.full(n_samples, int(stim_sz)))
            else:
                stim_sizes_parts.append(stim_sz)
        # Preserve condition string per sample (Fix 4)
        if 'condition' in data:
            cond = data['condition']
            if np.ndim(cond) == 0:
                conditions_parts.append(np.full(n_samples, str(cond), dtype=object))
            else:
                conditions_parts.append(cond)
        # Preserve stim_mode per sample
        if 'stim_mode' in data:
            sm = data['stim_mode']
            if np.ndim(sm) == 0:
                stim_modes_parts.append(np.full(n_samples, str(sm), dtype=object))
            else:
                stim_modes_parts.append(sm)

        for r in RF_RADII:
            key = f'features_r{r}'
            if key in data:
                X_by_radius[r].append(data[key])
                if r not in feature_names_by_radius:
                    fn_key = f'feature_names_r{r}'
                    if fn_key in data:
                        fn = data[fn_key]
                        if isinstance(fn, np.ndarray):
                            feature_names_by_radius[r] = fn.tolist()
                        else:
                            feature_names_by_radius[r] = list(fn)

    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)

    save_dict = {'y': y, 'groups': groups}

    if stim_sizes_parts:
        stim_sizes = np.concatenate(stim_sizes_parts)
        save_dict['stim_sizes'] = stim_sizes
        print(f"  stim_sizes: {sorted(np.unique(stim_sizes).tolist())}")
    if conditions_parts:
        conditions = np.concatenate(conditions_parts)
        save_dict['conditions'] = conditions
    if stim_modes_parts:
        stim_modes_arr = np.concatenate(stim_modes_parts)
        save_dict['stim_modes'] = stim_modes_arr
        print(f"  stim_modes: {sorted(set(stim_modes_arr.tolist()))}")

    for r in RF_RADII:
        if r in X_by_radius:
            X = np.vstack(X_by_radius[r])
            nan_mask = np.isnan(X)
            if np.any(nan_mask):
                X[nan_mask] = 0.0
            save_dict[f'X_r{r}'] = X
            if r in feature_names_by_radius:
                save_dict[f'feature_names_r{r}'] = np.array(
                    feature_names_by_radius[r], dtype=object)

    print(f"Combined dataset: y={y.shape}, groups={groups.shape}")
    print(f"  Naive samples: {np.sum(y == 0)}, Expert samples: {np.sum(y == 1)}")
    print(f"  Unique groups: {len(np.unique(groups))}")
    print(f"  RF radii with data: {sorted(X_by_radius.keys())}")

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(output_dir, f'downstream_dataset_{timestamp}.npz')
    np.savez_compressed(out_path, **save_dict)
    print(f"Saved: {out_path}")
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Ising downstream reader classifier: virtual neurons with '
                    'circular receptive fields classify Naive vs Expert dynamics')
    parser.add_argument('--output', '-o', type=str, default=r'IsingDownstream',
                        help='Output directory for results')
    parser.add_argument('--comparison', '-c', type=str, default=None,
                        help='Path to comparison results .mat file')
    parser.add_argument('--ising-data', '-d', type=str,
                        default=r'IsingModelData_39x78_100K',
                        help='Path to Ising simulation data directory')
    parser.add_argument('--index', '-i', type=int, default=None,
                        help='Run single job by index (for SLURM array jobs)')
    parser.add_argument('--local', action='store_true',
                        help='Run all jobs locally in parallel')
    parser.add_argument('--scan', action='store_true',
                        help='Show what jobs would be run (no execution)')
    parser.add_argument('--combine', action='store_true',
                        help='Combine individual job .npz files into dataset')
    parser.add_argument('--classify', action='store_true',
                        help='Run classification sweep on combined dataset')
    parser.add_argument('--detect', action='store_true',
                        help='Run stimulus detection analysis (stim vs no-stim per condition)')
    parser.add_argument('--detect-rf-radius', type=int, default=6,
                        help='RF radius for detection neurons (default: 6)')
    parser.add_argument('--detect-window', type=int, default=2,
                        help='Half-width of temporal window for windowed features (default: 2 -> 5 frames)')
    parser.add_argument('--workers', '-w', type=int,
                        default=max(1, cpu_count() - 1),
                        help='Number of parallel workers')
    parser.add_argument('--format', '-f', type=str,
                        choices=['npz', 'mat', 'both'], default='npz',
                        help='Output format (default: npz)')
    parser.add_argument('--stim-mode', type=str, default=DEFAULT_STIM_MODE,
                        help=f'Stimulus mode (default: {DEFAULT_STIM_MODE})')
    parser.add_argument('--stim-duration', type=int, default=DEFAULT_STIM_DURATION,
                        help=f'Stimulus duration in frames (default: {DEFAULT_STIM_DURATION})')
    parser.add_argument('--stim-sizes', type=int, nargs='+',
                        default=DEFAULT_STIM_SIZES,
                        help=f'Stimulus sizes (default: {DEFAULT_STIM_SIZES})')
    parser.add_argument('--stim-modes', nargs='+', default=None,
                        help='Multiple stimulus modes to test (overrides --stim-mode). '
                             'Options: clamped, double_pulse, bias')
    parser.add_argument('--stim-bias', type=float, nargs='+', default=None,
                        help='Bias values for bias mode (default: [1.0, 2.0, 4.0])')
    parser.add_argument('--rf-radii', type=int, nargs='+', default=None,
                        help=f'RF radii to sweep (default: {RF_RADII})')
    parser.add_argument('--job-count', action='store_true',
                        help='Print total job count and exit (for SLURM array sizing)')
    parser.add_argument('--detect-index', type=int, default=None,
                        help='Run single detect group by index (SLURM array)')
    parser.add_argument('--detect-combine', action='store_true',
                        help='Combine detect results, classify per stim_size, plot')
    parser.add_argument('--detect-job-count', action='store_true',
                        help='Print number of detect grouped jobs and exit')
    parser.add_argument('--detect-classify-job', action='store_true',
                        help='Run single (condition, stim_size) classification job by --index')
    parser.add_argument('--detect-aggregate', action='store_true',
                        help='Aggregate detect-clf results, plot, and save combined file')
    parser.add_argument('--detect-clf-job-count', action='store_true',
                        help='Print number of detect classify jobs and exit')
    parser.add_argument('--connectivity', type=str, choices=['rook', 'queen'], default='rook',
                        help='Neighbor connectivity: rook (4-connected) or queen (8-connected)')

    args = parser.parse_args()

    rf_radii = args.rf_radii if args.rf_radii else RF_RADII
    queen = (args.connectivity == 'queen')

    # Resolve stimulus modes
    if args.stim_modes:
        stim_modes = args.stim_modes
    else:
        stim_modes = [args.stim_mode]

    if 'bias' in stim_modes:
        stim_bias_values = args.stim_bias or [1.0, 2.0, 4.0]
    else:
        stim_bias_values = None

    # Scan mode
    if args.scan:
        print_scan_report(args.comparison, args.ising_data,
                          stim_modes, args.stim_duration, args.stim_sizes,
                          stim_bias_values)
        return

    # Job count query (for SLURM dynamic array sizing)
    # Redirect stdout→stderr so only the bare integer reaches the shell capture
    if args.job_count:
        _real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            if args.comparison is None:
                comparison_path = find_comparison_results(args.ising_data)
            else:
                comparison_path = args.comparison
            best_matches = load_comparison_results(comparison_path)
            all_jobs = generate_jobs(best_matches, stim_modes,
                                     args.stim_duration, args.stim_sizes,
                                     stim_bias_values=stim_bias_values)
        finally:
            sys.stdout = _real_stdout
        print(len(all_jobs))
        return

    # Detect job count query (for SLURM dynamic array sizing)
    if args.detect_job_count:
        _real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            if args.comparison is None:
                comparison_path = find_comparison_results(args.ising_data)
            else:
                comparison_path = args.comparison
            best_matches = load_comparison_results(comparison_path)
            grouped_jobs = generate_grouped_jobs(
                best_matches, stim_modes, args.stim_duration, args.stim_sizes,
                stim_bias_values=stim_bias_values)
        finally:
            sys.stdout = _real_stdout
        print(len(grouped_jobs))
        return

    # Detect classify job count query (for dynamic SLURM array sizing)
    if args.detect_clf_job_count:
        detect_dir = os.path.join(args.output, 'detection')
        if not os.path.isdir(detect_dir):
            print(0)
            return
        _real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            index_map = build_detect_clf_index_map(detect_dir)
        finally:
            sys.stdout = _real_stdout
        print(len(index_map))
        return

    # Combine mode
    if args.combine:
        combine_job_results(args.output)
        return

    # Classify mode
    if args.classify:
        run_classify(args.output, rf_radii=rf_radii)
        return

    # Detect: single group by index (SLURM array)
    if args.detect_index is not None:
        run_detect_one(args, args.detect_index)
        return

    # Detect: single (condition, stim_size) classification (SLURM array)
    if args.detect_classify_job:
        run_detect_classify_job(args)
        return

    # Detect: aggregate clf results, plot, save combined
    if args.detect_aggregate:
        run_detect_aggregate(args)
        return

    # Detect: combine results, classify, plot (legacy single-job mode)
    if args.detect_combine:
        run_detect_combine(args)
        return

    # Detect mode (local, all-in-one)
    if args.detect:
        run_detect(args)
        return

    # Single job by index (SLURM)
    if args.index is not None:
        os.makedirs(args.output, exist_ok=True)

        if args.comparison is None:
            comparison_path = find_comparison_results(args.ising_data)
        else:
            comparison_path = args.comparison

        best_matches = load_comparison_results(comparison_path)
        all_jobs = generate_jobs(best_matches, stim_modes,
                                 args.stim_duration, args.stim_sizes,
                                 stim_bias_values=stim_bias_values)

        if args.index < 0 or args.index >= len(all_jobs):
            raise ValueError(f"Index {args.index} out of range [0, {len(all_jobs)-1}]")

        job = all_jobs[args.index]
        config_dict = {
            'stim_modes': stim_modes,
            'stim_duration': args.stim_duration,
            'stim_sizes': args.stim_sizes,
            'queen': queen,
        }

        print(f"Running job {args.index}/{len(all_jobs)-1}")
        print(f"  Condition: {job['condition']}")
        print(f"  Simulation: {job['sim_idx']}")
        print(f"  Params: beta={job['params']['beta']:.4f}, c={job['params']['c']:.4f}")
        print(f"  Stimulus: size={job['stim_size']}, dur={job['stim_duration']}, mode={job['stim_mode']}")

        start_time = time.time()
        result = run_feature_job((job, config_dict))
        elapsed = time.time() - start_time

        print(f"  Completed in {elapsed:.1f} seconds")

        mode_suffix = f"_{job['stim_mode']}"
        if job['stim_mode'] == 'bias' and job.get('stim_bias') is not None:
            bias_str = f"{job['stim_bias']:.2f}".replace('.', 'p')
            mode_suffix = f"_bias_bias{bias_str}"
        base_filename = (f"downstream_job_{job['condition']}_sim{job['sim_idx']}"
                         f"_size{job['stim_size']}_dur{job['stim_duration']}"
                         f"{mode_suffix}")
        sim_dir = os.path.join(args.output, 'simulate')
        os.makedirs(sim_dir, exist_ok=True)
        out_path = os.path.join(sim_dir, base_filename + '.npz')

        save_dict = {
            'labels': result['labels'],
            'groups': result['groups'],
            'condition': result['condition'],
            'sim_idx': result['sim_idx'],
            'stim_size': result['stim_size'],
            'stim_duration': result['stim_duration'],
            'stim_mode': result['stim_mode'],
        }
        if result.get('stim_bias') is not None:
            save_dict['stim_bias'] = result['stim_bias']
        for r in RF_RADII:
            save_dict[f'features_r{r}'] = result['features_by_radius'][r]
            save_dict[f'feature_names_r{r}'] = np.array(
                result['feature_names_by_radius'][r], dtype=object)

        np.savez_compressed(out_path, **save_dict)
        print(f"  Saved: {out_path}")
        return

    # Local parallel mode
    if args.local:
        os.makedirs(args.output, exist_ok=True)

        if args.comparison is None:
            comparison_path = find_comparison_results(args.ising_data)
        else:
            comparison_path = args.comparison

        best_matches = load_comparison_results(comparison_path)
        all_jobs = generate_jobs(best_matches, stim_modes,
                                 args.stim_duration, args.stim_sizes,
                                 stim_bias_values=stim_bias_values)

        config_dict = {
            'stim_modes': stim_modes,
            'stim_duration': args.stim_duration,
            'stim_sizes': args.stim_sizes,
            'queen': queen,
        }
        args_list = [(job, config_dict) for job in all_jobs]

        print(f"Running {len(all_jobs)} jobs with {args.workers} workers...")
        start_time = time.time()

        results_list = []
        if args.workers > 1:
            with Pool(args.workers) as pool:
                for result in pool.imap(run_feature_job, args_list):
                    results_list.append(result)
                    # Save individual job result
                    job = result
                    mode_suffix = f"_{job['stim_mode']}"
                    if job['stim_mode'] == 'bias' and job.get('stim_bias') is not None:
                        bias_str = f"{job['stim_bias']:.2f}".replace('.', 'p')
                        mode_suffix = f"_bias_bias{bias_str}"
                    base_fn = (f"downstream_job_{job['condition']}_sim{job['sim_idx']}"
                               f"_size{job['stim_size']}_dur{job['stim_duration']}"
                               f"{mode_suffix}")
                    sim_dir = os.path.join(args.output, 'simulate')
                    os.makedirs(sim_dir, exist_ok=True)
                    out_path = os.path.join(sim_dir, base_fn + '.npz')
                    save_dict = {
                        'labels': job['labels'],
                        'groups': job['groups'],
                        'condition': job['condition'],
                        'sim_idx': job['sim_idx'],
                        'stim_size': job['stim_size'],
                        'stim_duration': job['stim_duration'],
                        'stim_mode': job['stim_mode'],
                    }
                    if job.get('stim_bias') is not None:
                        save_dict['stim_bias'] = job['stim_bias']
                    for r in RF_RADII:
                        save_dict[f'features_r{r}'] = job['features_by_radius'][r]
                        save_dict[f'feature_names_r{r}'] = np.array(
                            job['feature_names_by_radius'][r], dtype=object)
                    np.savez_compressed(out_path, **save_dict)
        else:
            for job_args in args_list:
                result = run_feature_job(job_args)
                results_list.append(result)

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed/60:.1f} minutes")

        # Also run classification sweep directly
        print("\nRunning classification sweep...")
        sweep_results = {}
        for r in rf_radii:
            X, y, groups, feature_names = build_dataset(r, results_list)
            print(f"  RF radius = {r}, X={X.shape} ...", end=' ', flush=True)
            res = run_classification(X, y, groups, feature_names)
            sweep_results[r] = res
            available = [m for m in ALL_MODEL_NAMES if m in res]
            best_acc = max(res[m]['accuracy_mean'] for m in available)
            print(f"best accuracy = {best_acc:.3f}")

        # Flush loky executor to release /dev/shm semaphores before next Parallel
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)

        # Permutation test
        opt_r = find_optimal_radius(sweep_results)
        print(f"\nOptimal RF radius: {opt_r}")
        print("Running permutation test...")
        X_opt, y, groups, fn_opt = build_dataset(opt_r, results_list)
        chance_accs = run_permutation_test(X_opt, y, groups)

        # Learning curve
        print("Computing learning curve...")
        lc_results = compute_learning_curve(X_opt, y, groups)

        # Per-stim-size analysis
        print("Running per-stimulus-size classification...")
        stim_sizes = args.stim_sizes
        sweep_by_stim = {}
        for sz in stim_sizes:
            sz_results = [r for r in results_list if r['stim_size'] == sz]
            if not sz_results:
                continue
            sz_sweep = {}
            for r in rf_radii:
                X_sz, y_sz, g_sz, fn_sz = build_dataset(r, sz_results)
                if len(np.unique(y_sz)) < 2:
                    continue
                sz_sweep[r] = run_classification(X_sz, y_sz, g_sz, fn_sz)
            if sz_sweep:
                sweep_by_stim[sz] = sz_sweep

        # Feature category ablation
        print("\nRunning feature category ablation at optimal radius...")
        ablation_results = run_feature_ablation(X_opt, y, groups, fn_opt)

        # Per-distance neuron analysis
        print("Running per-distance neuron analysis at optimal radius...")
        distance_results = run_distance_ablation(X_opt, y, groups, fn_opt)

        # Save and plot
        fig_dir = os.path.join(args.output, 'figures')
        save_results(sweep_results, args.output)
        plot_all(sweep_results, fig_dir, chance_accs=chance_accs,
                 sweep_by_stim=sweep_by_stim if sweep_by_stim else None,
                 stim_sizes=stim_sizes if sweep_by_stim else None,
                 lc_results=lc_results)

        if ablation_results:
            plot_ablation_bar(ablation_results, fig_dir)
        if distance_results:
            plot_distance_ablation_bar(distance_results, fig_dir)

        save_ablation_csv(ablation_results, distance_results, args.output)

        print("\nDone!")
        return

    # Default: print help
    parser.print_help()


if __name__ == '__main__':
    main()
