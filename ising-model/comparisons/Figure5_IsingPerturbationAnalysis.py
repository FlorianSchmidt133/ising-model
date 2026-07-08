# -*- coding: utf-8 -*-
"""
Figure5_IsingPerturbationAnalysis.py
Analyze Ising model perturbation experiments

This script analyzes results from run_ising_perturbations.py to address:
  1. Region Growth: How much does activity spread from stimulus?
  2. Persistence: How long does activity persist after stimulus offset?
  3. Size-Dependent Gating: Is there a threshold size for propagation?

Stimulus Modes:
  - Clamped: Region held at +1 throughout stimulus duration
  - Double Pulse: Set to +1 at onset AND offset (mimics SC behavior)
  - Bias: Local bias applied to stimulus region (graded contrast)

Usage:
    python Figure5_IsingPerturbationAnalysis.py --results IsingPerturbations\\PerturbationResults_*.npz
    python Figure5_IsingPerturbationAnalysis.py --results IsingPerturbations\\PerturbationResults_*.mat --output ./analysis
    python Figure5_IsingPerturbationAnalysis.py --results results.npz --show-real-time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.stats import kruskal
import h5py
import argparse
import warnings
from pathlib import Path
from glob import glob
import warnings


# =============================================================================
# Mode helpers — handle 'bias' and 'double_pulse_bias[N]' family uniformly
# =============================================================================

def mode_uses_bias(mode):
    """True for any mode stored as 5D (bias-value dim): 'bias', 'double_pulse_bias[N]'."""
    return mode == 'bias' or mode.startswith('double_pulse_bias')


def is_high_bias_display_mode(display_mode):
    return display_mode.startswith('high_')


def is_low_bias_display_mode(display_mode):
    return display_mode.startswith('low_')


def is_bias_display_mode(display_mode):
    return is_high_bias_display_mode(display_mode) or is_low_bias_display_mode(display_mode)


def data_key_from_display_mode(display_mode):
    """Map display mode (e.g. 'high_double_pulse_bias10') to its raw-data key."""
    if is_high_bias_display_mode(display_mode):
        base = display_mode[len('high_'):]
    elif is_low_bias_display_mode(display_mode):
        base = display_mode[len('low_'):]
    else:
        base = display_mode
    return base.replace('_', '')


def display_modes_for_mode(mode):
    """Bias-using modes split into ['high_<mode>', 'low_<mode>']; others pass through."""
    if mode_uses_bias(mode):
        return [f'high_{mode}', f'low_{mode}']
    return [mode]


def raw_mode_from_display_mode(display_mode):
    """Strip 'high_'/'low_' prefix but keep underscores intact (for filenames).
    'high_double_pulse_bias10' -> 'double_pulse_bias10', 'clamped' -> 'clamped'."""
    if is_high_bias_display_mode(display_mode):
        return display_mode[len('high_'):]
    if is_low_bias_display_mode(display_mode):
        return display_mode[len('low_'):]
    return display_mode


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration for perturbation analysis."""

    def __init__(self):
        # Colors (match Figure5_IsingComparison.m)
        self.colors = {
            'Naive': np.array([0.3373, 0.7059, 0.9137]),      # Light blue
            'Beginner': np.array([0.8431, 0.2549, 0.6078]),   # Magenta
            'Expert': np.array([0, 0.6196, 0.4510]),          # Teal
            'NoSpout': np.array([0.8353, 0.3686, 0]),         # Orange
        }

        # Conditions
        self.conditions = ['Naive', 'Beginner', 'Expert', 'NoSpout']

        # Time display settings
        self.show_real_time = False  # True = show time in seconds
        self.sampling_rate = 10      # Hz (experimental imaging rate)


# =============================================================================
# Helper Functions
# =============================================================================

def fit_exponential_decay(signal, baseline):
    """
    Fit exponential decay: y = A * exp(-t/tau) + baseline

    Parameters:
    -----------
    signal : ndarray
        Post-stimulus signal
    baseline : float
        Baseline value to subtract

    Returns:
    --------
    tau : float
        Decay time constant
    fit_r2 : float
        R-squared of fit
    """
    signal = np.asarray(signal).flatten() - baseline
    t = np.arange(1, len(signal) + 1)

    # Avoid log of non-positive values
    signal = np.maximum(signal, np.finfo(float).eps)

    # Log-linear regression
    log_sig = np.log(signal)
    valid_idx = np.isfinite(log_sig)

    if np.sum(valid_idx) < 3:
        return np.nan, np.nan

    # Linear fit: log(y) = log(A) - t/tau
    coeffs = np.polyfit(t[valid_idx], log_sig[valid_idx], 1)
    tau = -1 / coeffs[0] if coeffs[0] != 0 else np.nan

    # R-squared
    y_pred = np.polyval(coeffs, t[valid_idx])
    ss_res = np.sum((log_sig[valid_idx] - y_pred) ** 2)
    ss_tot = np.sum((log_sig[valid_idx] - np.mean(log_sig[valid_idx])) ** 2)
    fit_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Ensure positive tau
    if tau is not None and tau < 0:
        tau = np.nan

    return tau, fit_r2


def fit_hill_function(sizes, response):
    """
    Fit Hill function: R = Rmax * S^n / (EC50^n + S^n)

    Parameters:
    -----------
    sizes : ndarray
        Stimulus sizes
    response : ndarray
        Response values (e.g., amplification)

    Returns:
    --------
    EC50 : float
        Half-maximal effective concentration
    n : float
        Hill coefficient
    R2 : float
        R-squared of fit
    """
    # Normalize response
    Rmax = np.max(response)
    if Rmax == 0:
        return np.nan, np.nan, np.nan

    response_norm = np.asarray(response).flatten() / Rmax
    sizes = np.asarray(sizes).flatten()

    def hill_error(params):
        EC50, n = params
        if EC50 <= 0 or n <= 0:
            return np.inf
        predicted = (sizes ** n) / (EC50 ** n + sizes ** n)
        return np.sum((response_norm - predicted) ** 2)

    # Initial guess
    EC50_init = np.median(sizes)
    n_init = 2.0

    try:
        result = minimize(hill_error, [EC50_init, n_init], method='Nelder-Mead',
                         options={'maxiter': 1000, 'disp': False})
        EC50, n = result.x

        # Compute R-squared
        predicted = (sizes ** n) / (EC50 ** n + sizes ** n)
        ss_res = np.sum((response_norm - predicted) ** 2)
        ss_tot = np.sum((response_norm - np.mean(response_norm)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return EC50, n, R2
    except Exception:
        return np.nan, np.nan, np.nan


def redblue(n=256):
    """
    Create red-white-blue diverging colormap.

    Parameters:
    -----------
    n : int
        Number of colors

    Returns:
    --------
    cmap : LinearSegmentedColormap
    """
    # Blue to white to red
    colors = [
        (0.2, 0.2, 0.8),   # Blue
        (1.0, 1.0, 1.0),   # White
        (0.8, 0.2, 0.2),   # Red
    ]
    return LinearSegmentedColormap.from_list('redblue', colors, N=n)


# =============================================================================
# Data Loading
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
    except NotImplementedError:
        # v7.3 format - use h5py
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


# =============================================================================
# Main Analysis Class
# =============================================================================

class PerturbationAnalysis:
    """Analyze Ising model perturbation experiments."""

    def __init__(self, config=None):
        """
        Initialize analysis.

        Parameters:
        -----------
        config : Config, optional
            Configuration object
        """
        self.config = config or Config()
        self.results = None
        self.data = {}
        self.metrics = {}
        self.gating = {}
        self.stats = {}

        # Parameters extracted from results
        self.pre_stim_frames = None
        self.stimulus_durations = None
        self.post_stim_frames = None
        self.stimulus_sizes = None
        self.stimulus_modes = None
        self.stimulus_bias_values = None
        self.n_top_matches = None
        self.n_replicates = None
        self.grid_size = None
        self.conditions = None

    def load_results(self, filepath):
        """
        Load perturbation results from .mat or .npz file.

        Parameters:
        -----------
        filepath : str
            Path to results file (.mat or .npz)
        """
        filepath = Path(filepath)
        print(f"Loading results from: {filepath}")

        if filepath.suffix == '.npz':
            # Load numpy archive
            data = np.load(filepath, allow_pickle=True)
            self.results = {key: data[key] for key in data.files}
            # Handle object arrays (dicts stored as 0-d arrays)
            for key in self.results:
                if self.results[key].ndim == 0:
                    self.results[key] = self.results[key].item()
            self._is_hdf5 = False
        else:
            # Load .mat file
            self.results = load_mat_file(filepath)
            self._is_hdf5 = isinstance(self.results, h5py.File)

        # Extract configuration
        self._extract_config()

        print(f"Grid size: {self.grid_size}")
        print(f"Stimulus durations: {self.stimulus_durations} frames")
        print(f"Stimulus sizes: {self.stimulus_sizes}")
        print(f"Stimulus modes: {self.stimulus_modes}")
        if self.stimulus_bias_values is not None and len(self.stimulus_bias_values) > 0:
            print(f"Stimulus bias values: {self.stimulus_bias_values}")
        print(f"Replicates: {self.n_replicates}")

    def _get_value(self, obj):
        """Extract value from HDF5 dataset or return as-is."""
        if isinstance(obj, h5py.Dataset):
            val = obj[()]
            # Handle MATLAB strings stored as uint16 arrays
            if val.dtype == np.uint16:
                return ''.join(chr(c) for c in val.flatten())
            return val
        elif isinstance(obj, h5py.Group):
            return obj
        return obj

    def _get_string_array(self, obj):
        """Extract string array from HDF5 format."""
        if isinstance(obj, h5py.Dataset):
            # Check if it's references to strings
            if obj.dtype == h5py.special_dtype(ref=h5py.Reference):
                strings = []
                for ref in obj[()].flatten():
                    deref = self.results[ref]
                    if isinstance(deref, h5py.Dataset):
                        val = deref[()]
                        if val.dtype == np.uint16:
                            strings.append(''.join(chr(c) for c in val.flatten()))
                        else:
                            strings.append(str(val))
                return strings
            val = obj[()]
            if val.dtype == np.uint16:
                return [''.join(chr(c) for c in val.flatten())]
            return [v.decode('utf-8') if isinstance(v, bytes) else str(v)
                    for v in val.flatten()]
        elif isinstance(obj, np.ndarray):
            return [v.decode('utf-8') if isinstance(v, bytes) else str(v).strip()
                    for v in obj.flatten()]
        elif isinstance(obj, list):
            return [v.decode('utf-8') if isinstance(v, bytes) else str(v).strip()
                    for v in obj]
        else:
            return [obj.decode('utf-8') if isinstance(obj, bytes) else str(obj)]

    def _extract_config(self):
        """Extract configuration from loaded results."""
        R = self.results

        # Handle HDF5 datasets
        self.pre_stim_frames = int(np.atleast_1d(self._get_value(R['pre_stim_frames'])).flatten()[0])
        self.stimulus_durations = np.atleast_1d(self._get_value(R['stimulus_durations'])).flatten().astype(int)
        self.post_stim_frames = int(np.atleast_1d(self._get_value(R['post_stim_frames'])).flatten()[0])
        self.stimulus_sizes = np.atleast_1d(self._get_value(R['stimulus_sizes'])).flatten().astype(int)

        # Handle stimulus modes
        self.stimulus_modes = self._get_string_array(R['stimulus_modes'])

        # Handle conditions
        self.conditions = self._get_string_array(R['conditions'])

        self.n_top_matches = int(np.atleast_1d(self._get_value(R['n_top_matches'])).flatten()[0])
        self.n_replicates = int(np.atleast_1d(self._get_value(R['n_replicates'])).flatten()[0])
        self.grid_size = np.atleast_1d(self._get_value(R['grid_size'])).flatten().astype(int)

        # Stimulus bias values (optional)
        if 'stimulus_bias_values' in R:
            self.stimulus_bias_values = np.atleast_1d(self._get_value(R['stimulus_bias_values'])).flatten()
        else:
            self.stimulus_bias_values = np.array([])

    def extract_data(self, duration):
        """
        Extract and organize data for a specific stimulus duration.

        Parameters:
        -----------
        duration : int
            Stimulus duration in frames

        Returns:
        --------
        dict : Organized data by condition and mode
        """
        dur_key = f'dur_{duration}'
        total_frames = self.pre_stim_frames + duration + self.post_stim_frames

        experiments = self.results['experiments']
        if isinstance(experiments, h5py.Group):
            pass  # Keep as HDF5 group, will handle access differently
        elif isinstance(experiments, np.ndarray) and experiments.ndim == 0:
            experiments = experiments.item()

        data = {}
        n_sizes = len(self.stimulus_sizes)
        n_bias = len(self.stimulus_bias_values) if self.stimulus_bias_values is not None else 0

        for condition in self.conditions:
            if condition not in experiments:
                print(f"  {condition}: Not found in results")
                continue

            cond_data = experiments[condition]
            if isinstance(cond_data, h5py.Group):
                pass  # Keep as HDF5 group
            elif isinstance(cond_data, np.ndarray) and cond_data.ndim == 0:
                cond_data = cond_data.item()

            data[condition] = {
                'n_sims': self.n_top_matches,
                'stimulus_sizes': self.stimulus_sizes,
            }

            # Initialize storage for each mode
            for mode in self.stimulus_modes:
                mode_key = mode.replace('_', '')

                if mode_uses_bias(mode) and n_bias > 0:
                    # 5D arrays for bias mode
                    shape_ts = (self.n_top_matches, n_sizes, n_bias, self.n_replicates, total_frames)
                    shape_scalar = (self.n_top_matches, n_sizes, n_bias, self.n_replicates)
                else:
                    # 4D arrays for clamped/double_pulse
                    shape_ts = (self.n_top_matches, n_sizes, self.n_replicates, total_frames)
                    shape_scalar = (self.n_top_matches, n_sizes, self.n_replicates)

                data[condition][mode_key] = {
                    'activity': np.zeros(shape_ts),
                    'stim_activity': np.zeros(shape_ts),
                    'stimulus_blob_area': np.zeros(shape_ts),
                    'stimulus_blob_extent': np.zeros(shape_ts),
                    'valid_blob_count': np.zeros(shape_ts),
                    'morans_I': np.zeros(shape_ts),
                    'propagation_velocity': np.zeros(shape_ts),
                    'wavefront_anisotropy': np.zeros(shape_ts),
                    'time_to_max_extent': np.zeros(shape_scalar),
                    'max_propagation_velocity': np.zeros(shape_scalar),
                    'mean_anisotropy_during_stim': np.zeros(shape_scalar),
                }

            # Extract data for each simulation
            for sim in range(self.n_top_matches):
                sim_key = f'sim_{sim}'

                if sim_key not in cond_data:
                    continue

                sim_data = cond_data[sim_key]
                if isinstance(sim_data, h5py.Group):
                    pass
                elif isinstance(sim_data, np.ndarray) and sim_data.ndim == 0:
                    sim_data = sim_data.item()

                for mode in self.stimulus_modes:
                    mode_key = mode.replace('_', '')

                    if mode not in sim_data:
                        continue

                    mode_data = sim_data[mode]
                    if isinstance(mode_data, h5py.Group):
                        pass
                    elif isinstance(mode_data, np.ndarray) and mode_data.ndim == 0:
                        mode_data = mode_data.item()

                    for s, size in enumerate(self.stimulus_sizes):
                        size_key = f'size_{size}'

                        if size_key not in mode_data:
                            continue

                        size_data = mode_data[size_key]
                        if isinstance(size_data, h5py.Group):
                            pass
                        elif isinstance(size_data, np.ndarray) and size_data.ndim == 0:
                            size_data = size_data.item()

                        if dur_key not in size_data:
                            continue

                        dur_data = size_data[dur_key]
                        if isinstance(dur_data, h5py.Group):
                            pass
                        elif isinstance(dur_data, np.ndarray) and dur_data.ndim == 0:
                            dur_data = dur_data.item()

                        # Handle bias mode with extra dimension
                        if mode_uses_bias(mode) and n_bias > 0 and (isinstance(dur_data, dict) or isinstance(dur_data, h5py.Group)):
                            for b, bias_val in enumerate(self.stimulus_bias_values):
                                bias_key = f"bias_{bias_val:.2f}".replace('.', 'p')

                                if bias_key not in dur_data:
                                    continue

                                bias_data = dur_data[bias_key]
                                if isinstance(bias_data, h5py.Group):
                                    pass
                                elif isinstance(bias_data, np.ndarray) and bias_data.ndim == 0:
                                    bias_data = bias_data.item()

                                if bias_data is None or (isinstance(bias_data, np.ndarray) and bias_data.size == 0):
                                    continue

                                self._extract_metrics(data[condition][mode_key], bias_data,
                                                     sim, s, b,
                                                     n_replicates=self.n_replicates,
                                                     total_frames=total_frames)
                        else:
                            # Standard extraction for clamped/double_pulse
                            if dur_data is None or (isinstance(dur_data, np.ndarray) and dur_data.size == 0):
                                continue

                            self._extract_metrics(data[condition][mode_key], dur_data, sim, s,
                                                 n_replicates=self.n_replicates,
                                                 total_frames=total_frames)

            print(f"  {condition}: Extracted data for {self.n_top_matches} simulations")

        return data

    def _extract_metrics(self, storage, source, sim, s, b=None, n_replicates=None, total_frames=None):
        """Extract metrics from source to storage arrays."""
        if isinstance(source, np.ndarray) and source.ndim == 0:
            source = source.item()

        # Handle HDF5 Group as dict-like
        if not isinstance(source, (dict, h5py.Group)):
            return

        metrics = ['activity', 'stim_activity', 'stimulus_blob_area', 'stimulus_blob_extent',
                   'valid_blob_count', 'morans_I', 'propagation_velocity', 'wavefront_anisotropy']
        scalars = ['time_to_max_extent', 'max_propagation_velocity', 'mean_anisotropy_during_stim']

        for metric in metrics:
            if metric in source:
                val = source[metric]
                # Handle HDF5 Dataset
                if isinstance(val, h5py.Dataset):
                    val = val[()]
                if val is not None:
                    val = np.asarray(val)

                    # Validate and fix dimensions using KNOWN expected shapes
                    # Expected: [nReplicates x nFrames]
                    # DO NOT use size-based heuristic - it causes pre-stimulus artifacts
                    if val.ndim == 2 and n_replicates is not None and total_frames is not None:
                        expected_shape = (n_replicates, total_frames)

                        if val.shape == expected_shape:
                            pass  # Correct shape
                        elif val.shape == (total_frames, n_replicates):
                            val = val.T  # Fix HDF5/MATLAB column-major transposition
                        else:
                            # Unexpected shape - warn and use fallback
                            warnings.warn(
                                f"_extract_metrics: {metric} has shape {val.shape}, "
                                f"expected {expected_shape}. Using size heuristic as fallback."
                            )
                            if val.shape[0] > val.shape[1]:
                                val = val.T

                    if val.size > 0:
                        if b is not None:
                            storage[metric][sim, s, b, :, :] = val
                        else:
                            storage[metric][sim, s, :, :] = val

        for metric in scalars:
            if metric in source:
                val = source[metric]
                # Handle HDF5 Dataset
                if isinstance(val, h5py.Dataset):
                    val = val[()]
                if val is not None:
                    val = np.asarray(val)
                    if val.size > 0:
                        if b is not None:
                            storage[metric][sim, s, b, :] = val.flatten()
                        else:
                            storage[metric][sim, s, :] = val.flatten()

    def compute_metrics(self, data, duration):
        """
        Compute summary metrics from extracted data.

        Parameters:
        -----------
        data : dict
            Extracted data from extract_data()
        duration : int
            Stimulus duration in frames

        Returns:
        --------
        dict : Computed metrics by condition and mode
        """
        print("\n--- Computing Summary Metrics ---")

        total_frames = self.pre_stim_frames + duration + self.post_stim_frames
        pre_stim_idx = slice(0, self.pre_stim_frames)
        stim_on_idx = slice(self.pre_stim_frames, self.pre_stim_frames + duration)
        post_stim_idx = slice(self.pre_stim_frames + duration, total_frames)

        # Create display modes (split bias into high_bias and low_bias)
        display_modes = []
        for mode in self.stimulus_modes:
            display_modes.extend(display_modes_for_mode(mode))

        # Find low bias index
        low_bias_value = 0.25
        if len(self.stimulus_bias_values) > 0:
            low_bias_idx = np.argmin(np.abs(self.stimulus_bias_values - low_bias_value))
        else:
            low_bias_idx = 0

        metrics = {}

        for condition in self.conditions:
            if condition not in data:
                continue

            metrics[condition] = {}
            n_sims = data[condition]['n_sims']
            n_sizes = len(self.stimulus_sizes)

            for display_mode in display_modes:
                metrics_key = display_mode.replace('_', '')

                # Map display mode to data key
                data_key = data_key_from_display_mode(display_mode)

                if data_key not in data[condition]:
                    continue

                # Initialize metric arrays
                metrics[condition][metrics_key] = {
                    'baseline_activity': np.zeros((n_sims, n_sizes)),
                    'peak_activity': np.zeros((n_sims, n_sizes)),
                    'amplification': np.zeros((n_sims, n_sizes)),
                    'max_blob_extent': np.zeros((n_sims, n_sizes)),
                    'max_blob_area': np.zeros((n_sims, n_sizes)),
                    'half_decay_time': np.zeros((n_sims, n_sizes)),
                    'decay_tau': np.zeros((n_sims, n_sizes)),
                    'return_to_baseline': np.zeros((n_sims, n_sizes)),
                    'post_stim_auc': np.zeros((n_sims, n_sizes)),
                    'propagation_success': np.zeros((n_sims, n_sizes)),
                }

                for sim in range(n_sims):
                    for s in range(n_sizes):
                        stim_size = self.stimulus_sizes[s]
                        stim_radius = stim_size / 2

                        # Get activity time series
                        act_data = data[condition][data_key]['activity']
                        if is_high_bias_display_mode(display_mode) and act_data.ndim == 5:
                            activity = act_data[sim, s, -1, :, :]  # Highest bias
                        elif is_low_bias_display_mode(display_mode) and act_data.ndim == 5:
                            activity = act_data[sim, s, low_bias_idx, :, :]
                        else:
                            activity = act_data[sim, s, :, :]
                        mean_activity = np.mean(activity, axis=0)

                        # Get blob extent
                        ext_data = data[condition][data_key]['stimulus_blob_extent']
                        if is_high_bias_display_mode(display_mode) and ext_data.ndim == 5:
                            blob_extent = ext_data[sim, s, -1, :, :]
                        elif is_low_bias_display_mode(display_mode) and ext_data.ndim == 5:
                            blob_extent = ext_data[sim, s, low_bias_idx, :, :]
                        else:
                            blob_extent = ext_data[sim, s, :, :]
                        mean_blob_extent = np.mean(blob_extent, axis=0)

                        # Get blob area
                        area_data = data[condition][data_key]['stimulus_blob_area']
                        if is_high_bias_display_mode(display_mode) and area_data.ndim == 5:
                            blob_area = area_data[sim, s, -1, :, :]
                        elif is_low_bias_display_mode(display_mode) and area_data.ndim == 5:
                            blob_area = area_data[sim, s, low_bias_idx, :, :]
                        else:
                            blob_area = area_data[sim, s, :, :]
                        mean_blob_area = np.mean(blob_area, axis=0)

                        # Baseline metrics
                        baseline = np.mean(mean_activity[pre_stim_idx])
                        baseline_std = np.std(mean_activity[pre_stim_idx])
                        metrics[condition][metrics_key]['baseline_activity'][sim, s] = baseline

                        # Peak metrics
                        peak_activity = np.max(mean_activity)
                        metrics[condition][metrics_key]['peak_activity'][sim, s] = peak_activity
                        metrics[condition][metrics_key]['amplification'][sim, s] = peak_activity / max(baseline, 0.01)

                        # Propagation metrics (stim-on frames only)
                        max_extent = np.max(mean_blob_extent[stim_on_idx])
                        max_area = np.max(mean_blob_area[stim_on_idx])
                        metrics[condition][metrics_key]['max_blob_extent'][sim, s] = max_extent
                        metrics[condition][metrics_key]['max_blob_area'][sim, s] = max_area

                        # Propagation success
                        propagation_threshold = stim_radius + 5
                        metrics[condition][metrics_key]['propagation_success'][sim, s] = float(max_extent > propagation_threshold)

                        # Persistence metrics
                        post_stim_activity = mean_activity[post_stim_idx]
                        stim_offset_activity = mean_activity[self.pre_stim_frames + duration - 1]

                        # Half-decay time
                        half_target = (stim_offset_activity + baseline) / 2
                        half_idx = np.where(post_stim_activity < half_target)[0]
                        half_decay = half_idx[0] if len(half_idx) > 0 else self.post_stim_frames
                        metrics[condition][metrics_key]['half_decay_time'][sim, s] = half_decay

                        # Return to baseline
                        baseline_threshold = baseline + baseline_std
                        return_idx = np.where(post_stim_activity < baseline_threshold)[0]
                        return_time = return_idx[0] if len(return_idx) > 0 else self.post_stim_frames
                        metrics[condition][metrics_key]['return_to_baseline'][sim, s] = return_time

                        # AUC above baseline
                        auc = np.sum(np.maximum(0, post_stim_activity - baseline))
                        metrics[condition][metrics_key]['post_stim_auc'][sim, s] = auc

                        # Exponential decay fit
                        tau, _ = fit_exponential_decay(post_stim_activity, baseline)
                        metrics[condition][metrics_key]['decay_tau'][sim, s] = tau

            print(f"  {condition}: Computed metrics")

        return metrics

    def gating_analysis(self, metrics):
        """
        Perform size-dependent gating analysis.

        Parameters:
        -----------
        metrics : dict
            Computed metrics from compute_metrics()

        Returns:
        --------
        dict : Gating analysis results
        """
        print("\n--- Gating Analysis ---")

        # Create display modes
        display_modes = []
        for mode in self.stimulus_modes:
            display_modes.extend(display_modes_for_mode(mode))

        gating = {}

        for condition in self.conditions:
            if condition not in metrics:
                continue

            gating[condition] = {}

            for display_mode in display_modes:
                metrics_key = display_mode.replace('_', '')

                if metrics_key not in metrics[condition]:
                    continue

                # Get response amplitude vs size
                amplification = metrics[condition][metrics_key]['amplification']
                propagation_success = metrics[condition][metrics_key]['propagation_success']

                # Mean across simulations
                mean_amplification = np.mean(amplification, axis=0)
                mean_prop_success = np.mean(propagation_success, axis=0)

                gating[condition][metrics_key] = {
                    'mean_amplification': mean_amplification,
                    'mean_propagation_success': mean_prop_success,
                }

                # Find threshold size
                threshold_sizes = []
                for sim in range(amplification.shape[0]):
                    success_rate = propagation_success[sim, :]
                    thresh_idx = np.where(success_rate > 0.5)[0]
                    if len(thresh_idx) > 0:
                        threshold_sizes.append(self.stimulus_sizes[thresh_idx[0]])
                    else:
                        threshold_sizes.append(np.nan)

                gating[condition][metrics_key]['threshold_sizes'] = np.array(threshold_sizes)
                gating[condition][metrics_key]['mean_threshold'] = np.nanmean(threshold_sizes)

                # Fit Hill function
                EC50, hill_n, fit_R2 = fit_hill_function(self.stimulus_sizes, mean_amplification)
                gating[condition][metrics_key]['EC50'] = EC50
                gating[condition][metrics_key]['hill_coefficient'] = hill_n
                gating[condition][metrics_key]['hill_fit_R2'] = fit_R2

                print(f"  {condition} ({display_mode}): EC50={EC50:.1f}, Hill n={hill_n:.2f}, "
                      f"Mean threshold={gating[condition][metrics_key]['mean_threshold']:.1f}")

        return gating

    def statistical_comparisons(self, metrics, gating):
        """
        Perform statistical comparisons across conditions.

        Parameters:
        -----------
        metrics : dict
            Computed metrics
        gating : dict
            Gating analysis results

        Returns:
        --------
        dict : Statistical test results
        """
        print("\n--- Statistical Comparisons ---")

        # Create display modes
        display_modes = []
        for mode in self.stimulus_modes:
            display_modes.extend(display_modes_for_mode(mode))

        stats = {}
        metrics_to_compare = ['amplification', 'decay_tau', 'half_decay_time', 'max_blob_extent']

        for display_mode in display_modes:
            metrics_key = display_mode.replace('_', '')
            stats[metrics_key] = {}

            for metric_name in metrics_to_compare:
                stats[metrics_key][metric_name] = {}

                # Compare across conditions for each stimulus size
                for s, stim_size in enumerate(self.stimulus_sizes):
                    size_key = f'size_{stim_size}'

                    all_values = []
                    group_labels = []

                    for condition in self.conditions:
                        if condition not in metrics or metrics_key not in metrics[condition]:
                            continue

                        values = metrics[condition][metrics_key][metric_name][:, s]
                        values = values[~np.isnan(values)]

                        all_values.extend(values)
                        group_labels.extend([condition] * len(values))

                    # Kruskal-Wallis test
                    if len(set(group_labels)) > 1 and len(all_values) > 3:
                        groups = [np.array(all_values)[np.array(group_labels) == c]
                                 for c in self.conditions if c in set(group_labels)]
                        groups = [g for g in groups if len(g) > 0]

                        if len(groups) > 1:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                try:
                                    stat, p_value = kruskal(*groups)
                                except ValueError:
                                    # All values are identical - no statistical difference
                                    stat, p_value = 0.0, 1.0
                            stats[metrics_key][metric_name][size_key] = {
                                'statistic': stat,
                                'p_value': p_value,
                            }

            # Compare threshold sizes
            all_thresholds = []
            group_labels = []

            for condition in self.conditions:
                if condition not in gating or metrics_key not in gating[condition]:
                    continue

                thresholds = gating[condition][metrics_key]['threshold_sizes']
                thresholds = thresholds[~np.isnan(thresholds)]

                all_thresholds.extend(thresholds)
                group_labels.extend([condition] * len(thresholds))

            if len(set(group_labels)) > 1 and len(all_thresholds) > 3:
                groups = [np.array(all_thresholds)[np.array(group_labels) == c]
                         for c in self.conditions if c in set(group_labels)]
                groups = [g for g in groups if len(g) > 0]

                if len(groups) > 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            stat, p_value = kruskal(*groups)
                        except ValueError:
                            # All values are identical - no statistical difference
                            stat, p_value = 0.0, 1.0
                    stats[metrics_key]['threshold_size'] = {
                        'statistic': stat,
                        'p_value': p_value,
                    }
                    print(f"  Threshold size comparison ({display_mode}): p = {p_value:.4f}")

        return stats

    def plot_all(self, data, metrics, gating, stats, duration, output_dir):
        """
        Generate all visualization figures.

        Parameters:
        -----------
        data : dict
            Extracted data
        metrics : dict
            Computed metrics
        gating : dict
            Gating analysis results
        stats : dict
            Statistical test results
        duration : int
            Stimulus duration in frames
        output_dir : str or Path
            Output directory for figures
        """
        print("\n--- Creating Figures ---")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Time conversion
        if self.config.show_real_time:
            time_scale = 1 / self.config.sampling_rate
            time_unit = 's'
            time_label = 'Time (s from stim onset)'
            duration_str = f'{duration / self.config.sampling_rate:.1f} s'
        else:
            time_scale = 1
            time_unit = 'frames'
            time_label = 'Time (frames from stim onset)'
            duration_str = f'{duration} frames'

        total_frames = self.pre_stim_frames + duration + self.post_stim_frames
        # Match MATLAB convention: time=0 at last pre-stimulus frame (1-based indexing)
        time_vec = (np.arange(total_frames) + 1 - self.pre_stim_frames) * time_scale
        stim_on_time = duration * time_scale

        # Create display modes
        display_modes = []
        for mode in self.stimulus_modes:
            display_modes.extend(display_modes_for_mode(mode))
        n_display_modes = len(display_modes)

        # Find low bias index
        if len(self.stimulus_bias_values) > 0:
            low_bias_idx = np.argmin(np.abs(self.stimulus_bias_values - 0.25))
        else:
            low_bias_idx = 0

        # =====================================================================
        # Figure 1: Activity Time Courses
        # =====================================================================
        fig1, axes1 = plt.subplots(n_display_modes, len(self.conditions),
                                   figsize=(4*len(self.conditions), 3*n_display_modes))
        if n_display_modes == 1:
            axes1 = axes1.reshape(1, -1)
        if len(self.conditions) == 1:
            axes1 = axes1.reshape(-1, 1)

        for m, display_mode in enumerate(display_modes):
            data_key = data_key_from_display_mode(display_mode)

            for c, condition in enumerate(self.conditions):
                if condition not in data:
                    continue

                ax = axes1[m, c]
                cmap = plt.cm.viridis(np.linspace(0, 1, len(self.stimulus_sizes)))

                for s, stim_size in enumerate(self.stimulus_sizes):
                    act_data = data[condition][data_key]['activity']
                    if is_high_bias_display_mode(display_mode) and act_data.ndim == 5:
                        activity = act_data[:, s, -1, :, :]
                    elif is_low_bias_display_mode(display_mode) and act_data.ndim == 5:
                        activity = act_data[:, s, low_bias_idx, :, :]
                    else:
                        activity = act_data[:, s, :, :]
                    mean_activity = np.mean(np.mean(activity, axis=0), axis=0)

                    ax.plot(time_vec, mean_activity, color=cmap[s], linewidth=1.5,
                           label=f'{stim_size}' if m == 0 and c == len(self.conditions)-1 else '')

                # Mark stimulus period
                ax.axvline(0, color='k', linestyle='--', linewidth=1)
                ax.axvline(stim_on_time, color='k', linestyle='--', linewidth=1)
                ax.axvspan(0, stim_on_time, alpha=0.2, color='gray')

                ax.set_xlabel(time_label)
                ax.set_ylabel('Activity')
                title_mode = display_mode.replace('_', ' ')
                ax.set_title(f'{condition} - {title_mode}')

        fig1.suptitle(f'Activity Time Courses (Duration: {duration_str})')
        fig1.tight_layout()
        fig1.savefig(output_dir / f'ActivityTimeCourses_dur{duration}.png', dpi=150)
        fig1.savefig(output_dir / f'ActivityTimeCourses_dur{duration}.pdf')
        plt.close(fig1)
        print(f"  Saved: ActivityTimeCourses_dur{duration}")

        # =====================================================================
        # Figure 2: Dose-Response Curves
        # =====================================================================
        fig2, axes2 = plt.subplots(1, n_display_modes, figsize=(4*n_display_modes, 4))
        if n_display_modes == 1:
            axes2 = [axes2]

        for m, display_mode in enumerate(display_modes):
            metrics_key = display_mode.replace('_', '')
            ax = axes2[m]

            for condition in self.conditions:
                if condition not in metrics or metrics_key not in metrics[condition]:
                    continue

                amplification = metrics[condition][metrics_key]['amplification']
                mean_amp = np.mean(amplification, axis=0)
                sem_amp = np.std(amplification, axis=0) / np.sqrt(amplification.shape[0])

                color = self.config.colors[condition]
                ax.errorbar(self.stimulus_sizes, mean_amp, yerr=sem_amp, fmt='o-',
                           color=color, markerfacecolor=color, linewidth=1.5, label=condition)

                # Plot Hill fit
                if condition in gating and metrics_key in gating[condition]:
                    EC50 = gating[condition][metrics_key]['EC50']
                    hill_n = gating[condition][metrics_key]['hill_coefficient']
                    if not np.isnan(EC50) and not np.isnan(hill_n):
                        x_fit = np.linspace(min(self.stimulus_sizes), max(self.stimulus_sizes), 100)
                        Rmax = np.max(mean_amp)
                        y_fit = Rmax * (x_fit ** hill_n) / (EC50 ** hill_n + x_fit ** hill_n)
                        ax.plot(x_fit, y_fit, '--', color=color, linewidth=1)
                        ax.plot(EC50, Rmax/2, 'v', color=color, markersize=8, markerfacecolor=color)

            ax.set_xlabel('Stimulus Size (pixels)')
            ax.set_ylabel('Amplification (Peak / Baseline)')
            ax.set_title(f'Dose-Response: {display_mode.replace("_", " ")}')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)

        fig2.suptitle(f'Size-Dependent Response Amplification (Duration: {duration_str})')
        fig2.tight_layout()
        fig2.savefig(output_dir / f'DoseResponse_dur{duration}.png', dpi=150)
        fig2.savefig(output_dir / f'DoseResponse_dur{duration}.pdf')
        plt.close(fig2)
        print(f"  Saved: DoseResponse_dur{duration}")

        # =====================================================================
        # Figure 3: Persistence Analysis
        # =====================================================================
        fig3, axes3 = plt.subplots(2, n_display_modes, figsize=(4*n_display_modes, 6))
        if n_display_modes == 1:
            axes3 = axes3.reshape(-1, 1)

        for m, display_mode in enumerate(display_modes):
            metrics_key = display_mode.replace('_', '')

            # Half-decay time
            ax1 = axes3[0, m]
            for condition in self.conditions:
                if condition not in metrics or metrics_key not in metrics[condition]:
                    continue

                half_decay = metrics[condition][metrics_key]['half_decay_time'] * time_scale
                mean_hd = np.mean(half_decay, axis=0)
                sem_hd = np.std(half_decay, axis=0) / np.sqrt(half_decay.shape[0])

                color = self.config.colors[condition]
                ax1.errorbar(self.stimulus_sizes, mean_hd, yerr=sem_hd, fmt='o-',
                            color=color, markerfacecolor=color, linewidth=1.5, label=condition)

            ax1.set_xlabel('Stimulus Size (pixels)')
            ax1.set_ylabel(f'Half-Decay Time ({time_unit})')
            ax1.set_title(f'Half-Decay: {display_mode.replace("_", " ")}')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            # Decay tau
            ax2 = axes3[1, m]
            for condition in self.conditions:
                if condition not in metrics or metrics_key not in metrics[condition]:
                    continue

                decay_tau = metrics[condition][metrics_key]['decay_tau'] * time_scale
                decay_tau = np.where(np.isinf(decay_tau), np.nan, decay_tau)
                mean_tau = np.nanmean(decay_tau, axis=0)
                sem_tau = np.nanstd(decay_tau, axis=0) / np.sqrt(np.sum(~np.isnan(decay_tau[:, 0])))

                color = self.config.colors[condition]
                ax2.errorbar(self.stimulus_sizes, mean_tau, yerr=sem_tau, fmt='o-',
                            color=color, markerfacecolor=color, linewidth=1.5, label=condition)

            ax2.set_xlabel('Stimulus Size (pixels)')
            ax2.set_ylabel(f'Decay tau ({time_unit})')
            ax2.set_title(f'Decay Tau: {display_mode.replace("_", " ")}')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

        fig3.suptitle(f'Persistence After Stimulus Offset (Duration: {duration_str})')
        fig3.tight_layout()
        fig3.savefig(output_dir / f'Persistence_dur{duration}.png', dpi=150)
        fig3.savefig(output_dir / f'Persistence_dur{duration}.pdf')
        plt.close(fig3)
        print(f"  Saved: Persistence_dur{duration}")

        # =====================================================================
        # Figure 4: Net Propagation Heatmap
        # =====================================================================
        fig4, axes4 = plt.subplots(1, n_display_modes, figsize=(5*n_display_modes, 4))
        if n_display_modes == 1:
            axes4 = [axes4]

        stim_radii = self.stimulus_sizes / 2

        # Get global color limits
        all_net_prop = []
        for condition in self.conditions:
            if condition not in metrics:
                continue
            for display_mode in display_modes:
                metrics_key = display_mode.replace('_', '')
                if metrics_key in metrics[condition]:
                    for s in range(len(self.stimulus_sizes)):
                        net_prop = metrics[condition][metrics_key]['max_blob_extent'][:, s] - stim_radii[s]
                        all_net_prop.extend(net_prop)

        max_abs_val = np.max(np.abs(all_net_prop)) if all_net_prop else 1
        clims = (-max_abs_val, max_abs_val)

        for m, display_mode in enumerate(display_modes):
            metrics_key = display_mode.replace('_', '')
            ax = axes4[m]

            # Build net propagation matrix
            prop_matrix = np.zeros((len(self.conditions), len(self.stimulus_sizes)))
            for c, condition in enumerate(self.conditions):
                if condition not in metrics or metrics_key not in metrics[condition]:
                    continue
                for s in range(len(self.stimulus_sizes)):
                    mean_extent = np.mean(metrics[condition][metrics_key]['max_blob_extent'][:, s])
                    prop_matrix[c, s] = mean_extent - stim_radii[s]

            im = ax.imshow(prop_matrix, cmap=redblue(), aspect='auto',
                          vmin=clims[0], vmax=clims[1])
            plt.colorbar(im, ax=ax, label='Net Propagation (px)')

            ax.set_xticks(np.arange(len(self.stimulus_sizes)))
            ax.set_xticklabels(self.stimulus_sizes)
            ax.set_yticks(np.arange(len(self.conditions)))
            ax.set_yticklabels(self.conditions)
            ax.set_xlabel('Stimulus Size (pixels)')
            ax.set_ylabel('Condition')
            ax.set_title(display_mode.replace('_', ' '))

            # Add text annotations
            for c in range(len(self.conditions)):
                for s in range(len(self.stimulus_sizes)):
                    val = prop_matrix[c, s]
                    txt_color = 'k' if abs(val) < max_abs_val * 0.3 else 'w'
                    ax.text(s, c, f'{val:.1f}', ha='center', va='center',
                           color=txt_color, fontsize=7)

        fig4.suptitle(f'Net Propagation Beyond Stimulus (Duration: {duration_str})')
        fig4.tight_layout()
        fig4.savefig(output_dir / f'NetPropagationHeatmap_dur{duration}.png', dpi=150)
        fig4.savefig(output_dir / f'NetPropagationHeatmap_dur{duration}.pdf')
        plt.close(fig4)
        print(f"  Saved: NetPropagationHeatmap_dur{duration}")

        # =====================================================================
        # Figure 5: Gating Threshold Comparison
        # =====================================================================
        fig5, axes5 = plt.subplots(1, n_display_modes, figsize=(4*n_display_modes, 4))
        if n_display_modes == 1:
            axes5 = [axes5]

        for m, display_mode in enumerate(display_modes):
            metrics_key = display_mode.replace('_', '')
            ax = axes5[m]

            # Collect threshold data
            all_thresholds = []
            group_idx = []
            valid_conditions = []

            for c, condition in enumerate(self.conditions):
                if condition not in gating or metrics_key not in gating[condition]:
                    continue

                thresholds = gating[condition][metrics_key]['threshold_sizes']
                thresholds = thresholds[~np.isnan(thresholds)]

                if len(thresholds) > 0:
                    all_thresholds.extend(thresholds)
                    group_idx.extend([c] * len(thresholds))
                    if condition not in valid_conditions:
                        valid_conditions.append(condition)

            if len(all_thresholds) > 0:
                # Box plot
                data_by_group = [np.array(all_thresholds)[np.array(group_idx) == c]
                                for c in range(len(self.conditions))
                                if c in group_idx]
                positions = sorted(set(group_idx))

                bp = ax.boxplot(data_by_group, positions=positions, widths=0.6)

                # Overlay individual points with jitter
                for c in set(group_idx):
                    idx = np.array(group_idx) == c
                    x = c + 0.1 * (np.random.rand(np.sum(idx)) - 0.5)
                    y = np.array(all_thresholds)[idx]
                    condition = self.conditions[c]
                    ax.scatter(x, y, c=[self.config.colors[condition]], s=50, alpha=0.6)

                # Add p-value if available
                if metrics_key in stats and 'threshold_size' in stats[metrics_key]:
                    p = stats[metrics_key]['threshold_size']['p_value']
                    ax.text(0.05, 0.95, f'p = {p:.4f}', transform=ax.transAxes,
                           fontsize=10, verticalalignment='top')

            ax.set_xticks(range(len(self.conditions)))
            ax.set_xticklabels(self.conditions, rotation=45, ha='right')
            ax.set_ylabel('Threshold Size (pixels)')
            ax.set_title(f'Propagation Threshold: {display_mode.replace("_", " ")}')

        fig5.suptitle(f'Size-Dependent Gating (Duration: {duration_str})')
        fig5.tight_layout()
        fig5.savefig(output_dir / f'GatingThresholds_dur{duration}.png', dpi=150)
        fig5.savefig(output_dir / f'GatingThresholds_dur{duration}.pdf')
        plt.close(fig5)
        print(f"  Saved: GatingThresholds_dur{duration}")

        # =====================================================================
        # Figure 6: Net Propagation vs Size (Dose-Response Style)
        # =====================================================================
        fig6, axes6 = plt.subplots(1, n_display_modes, figsize=(4*n_display_modes, 4))
        if n_display_modes == 1:
            axes6 = [axes6]

        for m, display_mode in enumerate(display_modes):
            metrics_key = display_mode.replace('_', '')
            ax = axes6[m]

            for condition in self.conditions:
                if condition not in metrics or metrics_key not in metrics[condition]:
                    continue

                extent = metrics[condition][metrics_key]['max_blob_extent']
                net_prop = extent - stim_radii
                mean_net = np.mean(net_prop, axis=0)
                sem_net = np.std(net_prop, axis=0) / np.sqrt(net_prop.shape[0])

                color = self.config.colors[condition]
                ax.errorbar(self.stimulus_sizes, mean_net, yerr=sem_net, fmt='o-',
                           color=color, markerfacecolor=color, linewidth=1.5, label=condition)

            ax.axhline(0, color='k', linestyle='--', linewidth=1)
            ax.set_xlabel('Stimulus Size (pixels)')
            ax.set_ylabel('Net Propagation (px beyond stimulus)')
            ax.set_title(f'{display_mode.replace("_", " ")}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        fig6.suptitle(f'Size-Dependent Net Propagation (Duration: {duration_str})')
        fig6.tight_layout()
        fig6.savefig(output_dir / f'NetPropagationDoseResponse_dur{duration}.png', dpi=150)
        fig6.savefig(output_dir / f'NetPropagationDoseResponse_dur{duration}.pdf')
        plt.close(fig6)
        print(f"  Saved: NetPropagationDoseResponse_dur{duration}")

        # =====================================================================
        # Figure 7: Mode Comparison (Clamped vs Double Pulse)
        # =====================================================================
        fig7, axes7 = plt.subplots(1, 3, figsize=(12, 4))
        metrics_to_plot = ['amplification', 'decay_tau', 'max_blob_extent']
        metric_labels = ['Amplification', r'Decay $\tau$', 'Max Propagation']

        # Use a representative stimulus size
        rep_size = 8
        size_idx = np.where(self.stimulus_sizes == rep_size)[0]
        if len(size_idx) == 0:
            size_idx = len(self.stimulus_sizes) // 2
        else:
            size_idx = size_idx[0]

        for mi, (metric_name, metric_label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes7[mi]
            x_pos = 0
            xticks_pos = []
            xticks_labels = []

            for condition in self.conditions:
                if condition not in metrics:
                    continue
                if 'clamped' not in metrics[condition] or 'doublepulse' not in metrics[condition]:
                    continue

                clamped_vals = metrics[condition]['clamped'][metric_name][:, size_idx]
                pulse_vals = metrics[condition]['doublepulse'][metric_name][:, size_idx]

                clamped_vals = clamped_vals[~np.isnan(clamped_vals) & ~np.isinf(clamped_vals)]
                pulse_vals = pulse_vals[~np.isnan(pulse_vals) & ~np.isinf(pulse_vals)]

                if len(clamped_vals) == 0 or len(pulse_vals) == 0:
                    continue

                color = self.config.colors[condition]

                # Clamped
                ax.bar(x_pos, np.mean(clamped_vals), 0.35, color=color, edgecolor='k')
                ax.errorbar(x_pos, np.mean(clamped_vals), yerr=np.std(clamped_vals)/np.sqrt(len(clamped_vals)),
                           color='k', linewidth=1)

                # Double pulse
                ax.bar(x_pos + 0.4, np.mean(pulse_vals), 0.35, color=color, alpha=0.5, edgecolor='k')
                ax.errorbar(x_pos + 0.4, np.mean(pulse_vals), yerr=np.std(pulse_vals)/np.sqrt(len(pulse_vals)),
                           color='k', linewidth=1)

                xticks_pos.append(x_pos + 0.2)
                xticks_labels.append(condition)
                x_pos += 1.2

            ax.set_xticks(xticks_pos)
            ax.set_xticklabels(xticks_labels)
            ax.set_ylabel(metric_label)
            ax.set_title(f'{metric_label} (size={self.stimulus_sizes[size_idx]})')

            if mi == 0:
                ax.legend(['Clamped', 'Double Pulse'], loc='best')

        fig7.suptitle(f'Clamped vs Double Pulse (Duration: {duration_str})')
        fig7.tight_layout()
        fig7.savefig(output_dir / f'ModeComparison_dur{duration}.png', dpi=150)
        fig7.savefig(output_dir / f'ModeComparison_dur{duration}.pdf')
        plt.close(fig7)
        print(f"  Saved: ModeComparison_dur{duration}")

        # =====================================================================
        # Figure 8: Propagation Distance Over Time
        # =====================================================================
        rep_size = 10
        size_idx = np.where(self.stimulus_sizes == rep_size)[0]
        if len(size_idx) == 0:
            size_idx = len(self.stimulus_sizes) // 2
        else:
            size_idx = size_idx[0]

        n_rows = (n_display_modes + 1) // 2
        n_cols = min(2, n_display_modes)
        fig8, axes8 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_display_modes == 1:
            axes8 = np.array([[axes8]])
        elif n_rows == 1:
            axes8 = axes8.reshape(1, -1)
        elif n_cols == 1:
            axes8 = axes8.reshape(-1, 1)

        for m, display_mode in enumerate(display_modes):
            row, col = m // n_cols, m % n_cols
            data_key = data_key_from_display_mode(display_mode)

            ax = axes8[row, col] if n_display_modes > 1 else axes8[0, 0]

            for condition in self.conditions:
                if condition not in data or data_key not in data[condition]:
                    continue

                prop_data = data[condition][data_key]['stimulus_blob_extent']
                if is_high_bias_display_mode(display_mode) and prop_data.ndim == 5:
                    propagation = prop_data[:, size_idx, -1, :, :]
                elif is_low_bias_display_mode(display_mode) and prop_data.ndim == 5:
                    propagation = prop_data[:, size_idx, low_bias_idx, :, :]
                else:
                    propagation = prop_data[:, size_idx, :, :]

                mean_prop = np.mean(np.mean(propagation, axis=0), axis=0)
                color = self.config.colors[condition]
                ax.plot(time_vec, mean_prop, color=color, linewidth=2, label=condition)

            # Mark stimulus period
            ax.axvline(0, color='k', linestyle='--', linewidth=1)
            ax.axvline(stim_on_time, color='k', linestyle='--', linewidth=1)
            ax.axvspan(0, stim_on_time, alpha=0.2, color='gray')

            # Mark stimulus radius
            ax.axhline(self.stimulus_sizes[size_idx]/2, color='r', linestyle=':', linewidth=1)

            ax.set_xlabel(time_label)
            ax.set_ylabel('Max Propagation Distance (pixels)')
            title_mode = display_mode.replace('_', ' ')
            ax.set_title(f'Propagation: {title_mode} (size={self.stimulus_sizes[size_idx]})')
            if m == 0:
                ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        fig8.suptitle(f'Propagation Distance (Duration: {duration_str})')
        fig8.tight_layout()
        fig8.savefig(output_dir / f'PropagationDistance_dur{duration}.png', dpi=150)
        fig8.savefig(output_dir / f'PropagationDistance_dur{duration}.pdf')
        plt.close(fig8)
        print(f"  Saved: PropagationDistance_dur{duration}")

        # =====================================================================
        # Figure 9: Pre-Stimulus State Effects
        # =====================================================================
        pre_stim_effects = self._compute_prestim_effects(data, duration)

        fig9, axes9 = plt.subplots(1, 3, figsize=(12, 4))

        # Panel A: Baseline Moran's I by condition
        ax = axes9[0]
        for c, condition in enumerate(self.conditions):
            if condition not in pre_stim_effects:
                continue

            color = self.config.colors[condition]
            morans_vals = pre_stim_effects[condition].get('baseline_moransI', [])
            morans_vals = np.array(morans_vals).flatten()
            morans_vals = morans_vals[~np.isnan(morans_vals) & (morans_vals != 0)]

            if len(morans_vals) > 0:
                jitter = 0.1 * (np.random.rand(len(morans_vals)) - 0.5)
                ax.scatter(c + jitter, morans_vals, c=[color], s=30, alpha=0.6)
                ax.plot([c-0.3, c+0.3], [np.mean(morans_vals)]*2, 'k-', linewidth=2)

        ax.set_xticks(range(len(self.conditions)))
        ax.set_xticklabels(self.conditions)
        ax.set_ylabel("Baseline Moran's I")
        ax.set_title('A) Pre-Stimulus Spatial Clustering')

        # Panel B: Baseline vs Peak response correlation
        ax = axes9[1]
        for c, condition in enumerate(self.conditions):
            if condition not in pre_stim_effects:
                continue

            color = self.config.colors[condition]
            corr_vals = pre_stim_effects[condition].get('baseline_vs_peak', [])
            corr_vals = np.array(corr_vals).flatten()
            corr_vals = corr_vals[~np.isnan(corr_vals)]

            if len(corr_vals) > 0:
                jitter = 0.1 * (np.random.rand(len(corr_vals)) - 0.5)
                ax.scatter(c + jitter, corr_vals, c=[color], s=30, alpha=0.6)
                ax.plot([c-0.3, c+0.3], [np.mean(corr_vals)]*2, 'k-', linewidth=2)

        ax.set_xticks(range(len(self.conditions)))
        ax.set_xticklabels(self.conditions)
        ax.axhline(0, color='k', linestyle='--')
        ax.set_ylabel('Correlation (r)')
        ax.set_title('B) Baseline Activity vs Peak Response')

        # Panel C: Baseline vs Extent correlation
        ax = axes9[2]
        for c, condition in enumerate(self.conditions):
            if condition not in pre_stim_effects:
                continue

            color = self.config.colors[condition]
            corr_vals = pre_stim_effects[condition].get('baseline_vs_extent', [])
            corr_vals = np.array(corr_vals).flatten()
            corr_vals = corr_vals[~np.isnan(corr_vals)]

            if len(corr_vals) > 0:
                jitter = 0.1 * (np.random.rand(len(corr_vals)) - 0.5)
                ax.scatter(c + jitter, corr_vals, c=[color], s=30, alpha=0.6)
                ax.plot([c-0.3, c+0.3], [np.mean(corr_vals)]*2, 'k-', linewidth=2)

        ax.set_xticks(range(len(self.conditions)))
        ax.set_xticklabels(self.conditions)
        ax.axhline(0, color='k', linestyle='--')
        ax.set_ylabel('Correlation (r)')
        ax.set_title('C) Baseline Activity vs Propagation Extent')

        fig9.suptitle(f'Pre-Stimulus State Effects (Duration: {duration_str})')
        fig9.tight_layout()
        fig9.savefig(output_dir / f'PreStimEffects_dur{duration}.png', dpi=150)
        fig9.savefig(output_dir / f'PreStimEffects_dur{duration}.pdf')
        plt.close(fig9)
        print(f"  Saved: PreStimEffects_dur{duration}")

        # =====================================================================
        # Figure 10: Propagation Dynamics
        # =====================================================================
        prop_dynamics = self._compute_prop_dynamics(data, metrics, duration)

        # Velocity conversion
        if self.config.show_real_time:
            velocity_scale = self.config.sampling_rate
            velocity_unit = 'px/s'
        else:
            velocity_scale = 1
            velocity_unit = 'px/frame'

        fig10, axes10 = plt.subplots(1, 2, figsize=(10, 4))

        # Panel A: Max velocity vs stimulus size
        ax = axes10[0]
        for condition in self.conditions:
            if condition not in prop_dynamics:
                continue

            velocity = prop_dynamics[condition].get('max_velocity', np.array([]))
            if velocity.size == 0:
                continue

            mean_vel = np.mean(velocity, axis=0) * velocity_scale
            sem_vel = np.std(velocity, axis=0) / np.sqrt(velocity.shape[0]) * velocity_scale

            color = self.config.colors[condition]
            ax.errorbar(self.stimulus_sizes, mean_vel, yerr=sem_vel, fmt='o-',
                       color=color, markerfacecolor=color, linewidth=1.5, label=condition)

        ax.set_xlabel('Stimulus Size (pixels)')
        ax.set_ylabel(f'Max Velocity ({velocity_unit})')
        ax.set_title('A) Peak Propagation Velocity')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Panel B: Time to max extent vs size
        ax = axes10[1]
        for condition in self.conditions:
            if condition not in prop_dynamics:
                continue

            ttm = prop_dynamics[condition].get('time_to_max', np.array([]))
            if ttm.size == 0:
                continue

            mean_ttm = np.mean(ttm, axis=0) * time_scale
            sem_ttm = np.std(ttm, axis=0) / np.sqrt(ttm.shape[0]) * time_scale

            color = self.config.colors[condition]
            ax.errorbar(self.stimulus_sizes, mean_ttm, yerr=sem_ttm, fmt='o-',
                       color=color, markerfacecolor=color, linewidth=1.5, label=condition)

        ax.set_xlabel('Stimulus Size (pixels)')
        ax.set_ylabel(f'Time to Max ({time_unit})')
        ax.set_title('B) Time to Peak Spread')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        fig10.suptitle(f'Propagation Dynamics (Duration: {duration_str})')
        fig10.tight_layout()
        fig10.savefig(output_dir / f'PropagationDynamics_dur{duration}.png', dpi=150)
        fig10.savefig(output_dir / f'PropagationDynamics_dur{duration}.pdf')
        plt.close(fig10)
        print(f"  Saved: PropagationDynamics_dur{duration}")

        # =====================================================================
        # Figure 11: Blob Interactions
        # =====================================================================
        blob_interactions = self._compute_blob_interactions(data, duration)

        fig11, axes11 = plt.subplots(2, 2, figsize=(10, 8))

        # Panel A: Blob count change vs size
        ax = axes11[0, 0]
        for condition in self.conditions:
            if condition not in blob_interactions:
                continue

            change = blob_interactions[condition].get('blob_count_change', np.array([]))
            if change.size == 0:
                continue

            mean_change = np.mean(change, axis=0)
            sem_change = np.std(change, axis=0) / np.sqrt(change.shape[0])

            color = self.config.colors[condition]
            ax.errorbar(self.stimulus_sizes, mean_change, yerr=sem_change, fmt='o-',
                       color=color, markerfacecolor=color, linewidth=1.5, label=condition)

        ax.axhline(0, color='k', linestyle='--')
        ax.set_xlabel('Stimulus Size (pixels)')
        ax.set_ylabel(r'$\Delta$ Blob Count')
        ax.set_title('A) Stimulus Effect on Blob Count')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Panel B: Pre-stim vs stim blob count
        ax = axes11[0, 1]
        for condition in self.conditions:
            if condition not in blob_interactions:
                continue

            prestim = blob_interactions[condition].get('prestim_blob_count', np.array([]))
            stim = blob_interactions[condition].get('stim_blob_count', np.array([]))

            if prestim.size == 0 or stim.size == 0:
                continue

            color = self.config.colors[condition]
            ax.scatter(prestim.flatten(), stim.flatten(), c=[color], s=30, alpha=0.5, label=condition)

        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        max_val = np.ceil(max_val * 1.1)  # 10% padding, rounded up
        ax.plot([0, max_val], [0, max_val], 'k:', linewidth=1)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xlabel('Pre-stim Blob Count')
        ax.set_ylabel('During-stim Blob Count')
        ax.set_title('B) Blob Count: Pre vs During Stim')
        ax.legend(loc='best')

        # Panel C: Mean blob counts by condition
        ax = axes11[1, 0]
        x_pos = 0
        for condition in self.conditions:
            if condition not in blob_interactions:
                continue

            prestim = blob_interactions[condition].get('prestim_blob_count', np.array([]))
            stim = blob_interactions[condition].get('stim_blob_count', np.array([]))

            if prestim.size == 0:
                continue

            color = self.config.colors[condition]
            prestim_mean = np.mean(prestim)
            stim_mean = np.mean(stim)

            ax.bar(x_pos, prestim_mean, 0.35, color=color, edgecolor='k')
            ax.bar(x_pos + 0.4, stim_mean, 0.35, color=color, alpha=0.5, edgecolor='k')
            x_pos += 1.2

        ax.set_xticks(np.arange(0, len(self.conditions)*1.2, 1.2) + 0.2)
        ax.set_xticklabels(self.conditions)
        ax.set_ylabel('Blob Count')
        ax.set_title('C) Pre-stim (solid) vs Stim (light)')

        # Panel D: Summary
        ax = axes11[1, 1]
        ax.axis('off')
        summary_lines = ['Blob Interaction Summary', '']
        for condition in self.conditions:
            if condition in blob_interactions:
                prestim = blob_interactions[condition].get('prestim_blob_count', np.array([]))
                stim = blob_interactions[condition].get('stim_blob_count', np.array([]))
                change = blob_interactions[condition].get('blob_count_change', np.array([]))
                if prestim.size > 0:
                    summary_lines.append(f'{condition}: Pre={np.mean(prestim):.1f}, '
                                        f'Stim={np.mean(stim):.1f}, Δ={np.mean(change):.1f}')
        ax.text(0.1, 0.8, '\n'.join(summary_lines), transform=ax.transAxes,
               verticalalignment='top', fontsize=10)

        fig11.suptitle(f'Blob Interaction Analysis (Duration: {duration_str})')
        fig11.tight_layout()
        fig11.savefig(output_dir / f'BlobInteractions_dur{duration}.png', dpi=150)
        fig11.savefig(output_dir / f'BlobInteractions_dur{duration}.pdf')
        plt.close(fig11)
        print(f"  Saved: BlobInteractions_dur{duration}")

        # =====================================================================
        # Figure 12: Moran's I Dynamics
        # =====================================================================
        morans_dynamics = self._compute_morans_dynamics(data, duration)

        rep_size = 8
        size_idx = np.where(self.stimulus_sizes == rep_size)[0]
        if len(size_idx) == 0:
            size_idx = 3  # Default to index 3
        else:
            size_idx = size_idx[0]

        fig12, axes12 = plt.subplots(1, 3, figsize=(14, 4))

        # Panel A: Moran's I time course
        ax = axes12[0]
        for condition in self.conditions:
            if condition not in data or 'clamped' not in data[condition]:
                continue

            morans_data = data[condition]['clamped'].get('morans_I', np.array([]))
            if morans_data.size == 0:
                continue

            if morans_data.ndim >= 3:
                morans_I = morans_data[:, size_idx, :, :]
                mean_morans = np.mean(np.mean(morans_I, axis=0), axis=0)
            else:
                continue

            color = self.config.colors[condition]
            ax.plot(time_vec, mean_morans, color=color, linewidth=2, label=condition)

        ax.axvline(0, color='k', linestyle='--')
        ax.axvline(stim_on_time, color='k', linestyle='--')
        ax.set_xlabel(time_label)
        ax.set_ylabel("Moran's I")
        ax.set_title(f"A) Moran's I Time Course (size={self.stimulus_sizes[size_idx]})")
        ax.legend(loc='best')

        # Panel B: Moran's I increase vs size
        ax = axes12[1]
        for condition in self.conditions:
            if condition not in morans_dynamics:
                continue

            increase = morans_dynamics[condition].get('moransI_increase', np.array([]))
            if increase.size == 0:
                continue

            mean_inc = np.mean(increase, axis=0)
            sem_inc = np.std(increase, axis=0) / np.sqrt(increase.shape[0])

            color = self.config.colors[condition]
            ax.errorbar(self.stimulus_sizes, mean_inc, yerr=sem_inc, fmt='o-',
                       color=color, markerfacecolor=color, linewidth=1.5, label=condition)

        ax.set_xlabel('Stimulus Size (pixels)')
        ax.set_ylabel(r"$\Delta$ Moran's I")
        ax.set_title('B) Clustering Increase vs Size')
        ax.grid(True, alpha=0.3)

        # Panel C: Return to baseline time
        ax = axes12[2]
        for condition in self.conditions:
            if condition not in morans_dynamics:
                continue

            return_time = morans_dynamics[condition].get('return_to_baseline_time', np.array([]))
            if return_time.size == 0:
                continue

            mean_rt = np.mean(return_time, axis=0) * time_scale
            sem_rt = np.std(return_time, axis=0) / np.sqrt(return_time.shape[0]) * time_scale

            color = self.config.colors[condition]
            ax.errorbar(self.stimulus_sizes, mean_rt, yerr=sem_rt, fmt='o-',
                       color=color, markerfacecolor=color, linewidth=1.5, label=condition)

        ax.set_xlabel('Stimulus Size (pixels)')
        ax.set_ylabel(f'Return Time ({time_unit})')
        ax.set_title('C) Time to Return to Baseline')
        ax.grid(True, alpha=0.3)

        fig12.suptitle(f"Moran's I Dynamics (Duration: {duration_str})")
        fig12.tight_layout()
        fig12.savefig(output_dir / f'MoransIDynamics_dur{duration}.png', dpi=150)
        fig12.savefig(output_dir / f'MoransIDynamics_dur{duration}.pdf')
        plt.close(fig12)
        print(f"  Saved: MoransIDynamics_dur{duration}")

    def _compute_prestim_effects(self, data, duration):
        """Compute pre-stimulus effects for analysis."""
        pre_stim_idx = slice(0, self.pre_stim_frames)
        stim_on_idx = slice(self.pre_stim_frames, self.pre_stim_frames + duration)

        effects = {}
        for condition in self.conditions:
            if condition not in data:
                continue

            effects[condition] = {
                'baseline_moransI': [],
                'baseline_vs_peak': [],
                'baseline_vs_extent': [],
            }

            if 'clamped' not in data[condition]:
                continue

            for sim in range(data[condition]['n_sims']):
                for s in range(len(self.stimulus_sizes)):
                    # Activity data
                    act_data = data[condition]['clamped']['activity']
                    if act_data.ndim < 4:
                        continue

                    activity = act_data[sim, s, :, :]
                    mean_activity = np.mean(activity, axis=0)

                    # Baseline and peak
                    baseline = np.mean(mean_activity[pre_stim_idx])
                    peak = np.max(mean_activity)

                    # Moran's I data
                    morans_data = data[condition]['clamped'].get('morans_I', np.array([]))
                    if morans_data.ndim >= 4:
                        morans_I = morans_data[sim, s, :, :]
                        baseline_morans = np.mean(morans_I[:, pre_stim_idx])
                        effects[condition]['baseline_moransI'].append(baseline_morans)

                    # Extent data
                    ext_data = data[condition]['clamped']['stimulus_blob_extent']
                    if ext_data.ndim >= 4:
                        extent = ext_data[sim, s, :, :]
                        max_extent = np.max(np.mean(extent, axis=0))

                        # Correlations across replicates
                        baseline_per_rep = np.mean(activity[:, pre_stim_idx], axis=1)
                        peak_per_rep = np.max(activity, axis=1)
                        extent_per_rep = np.max(extent, axis=1)

                        if len(baseline_per_rep) > 2:
                            corr_peak = np.corrcoef(baseline_per_rep, peak_per_rep)[0, 1]
                            corr_extent = np.corrcoef(baseline_per_rep, extent_per_rep)[0, 1]
                            effects[condition]['baseline_vs_peak'].append(corr_peak)
                            effects[condition]['baseline_vs_extent'].append(corr_extent)

        return effects

    def _compute_prop_dynamics(self, data, metrics, duration):
        """Compute propagation dynamics metrics."""
        dynamics = {}

        for condition in self.conditions:
            if condition not in data:
                continue

            dynamics[condition] = {
                'max_velocity': [],
                'time_to_max': [],
            }

            if 'clamped' not in data[condition]:
                continue

            n_sims = data[condition]['n_sims']
            n_sizes = len(self.stimulus_sizes)

            max_vel = np.zeros((n_sims, n_sizes))
            time_max = np.zeros((n_sims, n_sizes))

            for sim in range(n_sims):
                for s in range(n_sizes):
                    # Velocity data
                    vel_data = data[condition]['clamped'].get('propagation_velocity', np.array([]))
                    if vel_data.ndim >= 4:
                        velocity = vel_data[sim, s, :, :]
                        max_vel[sim, s] = np.max(np.mean(velocity, axis=0))

                    # Time to max extent
                    ext_data = data[condition]['clamped']['stimulus_blob_extent']
                    if ext_data.ndim >= 4:
                        extent = ext_data[sim, s, :, :]
                        mean_extent = np.mean(extent, axis=0)
                        time_max[sim, s] = np.argmax(mean_extent)

            dynamics[condition]['max_velocity'] = max_vel
            dynamics[condition]['time_to_max'] = time_max

        return dynamics

    def _compute_blob_interactions(self, data, duration):
        """Compute blob interaction metrics."""
        pre_stim_idx = slice(0, self.pre_stim_frames)
        stim_on_idx = slice(self.pre_stim_frames, self.pre_stim_frames + duration)

        interactions = {}

        for condition in self.conditions:
            if condition not in data:
                continue

            interactions[condition] = {
                'prestim_blob_count': [],
                'stim_blob_count': [],
                'blob_count_change': [],
            }

            if 'clamped' not in data[condition]:
                continue

            n_sims = data[condition]['n_sims']
            n_sizes = len(self.stimulus_sizes)

            prestim_count = np.zeros((n_sims, n_sizes))
            stim_count = np.zeros((n_sims, n_sizes))
            change = np.zeros((n_sims, n_sizes))

            for sim in range(n_sims):
                for s in range(n_sizes):
                    blob_data = data[condition]['clamped'].get('valid_blob_count', np.array([]))
                    if blob_data.ndim >= 4:
                        blob_count = blob_data[sim, s, :, :]
                        mean_blob = np.mean(blob_count, axis=0)

                        prestim_count[sim, s] = np.mean(mean_blob[pre_stim_idx])
                        stim_count[sim, s] = np.mean(mean_blob[stim_on_idx])
                        change[sim, s] = stim_count[sim, s] - prestim_count[sim, s]

            interactions[condition]['prestim_blob_count'] = prestim_count
            interactions[condition]['stim_blob_count'] = stim_count
            interactions[condition]['blob_count_change'] = change

        return interactions

    def _compute_morans_dynamics(self, data, duration):
        """Compute Moran's I dynamics metrics."""
        pre_stim_idx = slice(0, self.pre_stim_frames)
        stim_on_idx = slice(self.pre_stim_frames, self.pre_stim_frames + duration)
        post_stim_idx = slice(self.pre_stim_frames + duration, None)

        dynamics = {}

        for condition in self.conditions:
            if condition not in data:
                continue

            dynamics[condition] = {
                'moransI_increase': [],
                'return_to_baseline_time': [],
            }

            if 'clamped' not in data[condition]:
                continue

            n_sims = data[condition]['n_sims']
            n_sizes = len(self.stimulus_sizes)

            morans_increase = np.zeros((n_sims, n_sizes))
            return_time = np.zeros((n_sims, n_sizes))

            for sim in range(n_sims):
                for s in range(n_sizes):
                    morans_data = data[condition]['clamped'].get('morans_I', np.array([]))
                    if morans_data.ndim >= 4:
                        morans_I = morans_data[sim, s, :, :]
                        mean_morans = np.mean(morans_I, axis=0)

                        baseline_morans = np.mean(mean_morans[pre_stim_idx])
                        peak_morans = np.max(mean_morans[stim_on_idx])
                        morans_increase[sim, s] = peak_morans - baseline_morans

                        # Return to baseline
                        post_morans = mean_morans[post_stim_idx]
                        threshold = baseline_morans + 0.1 * (peak_morans - baseline_morans)
                        return_idx = np.where(post_morans < threshold)[0]
                        if len(return_idx) > 0:
                            return_time[sim, s] = return_idx[0]
                        else:
                            return_time[sim, s] = self.post_stim_frames

            dynamics[condition]['moransI_increase'] = morans_increase
            dynamics[condition]['return_to_baseline_time'] = return_time

        return dynamics

    def plot_summary(self, output_dir):
        """
        Generate cross-duration summary figures.

        Parameters:
        -----------
        output_dir : str or Path
            Output directory for figures
        """
        print("\n--- Creating Cross-Duration Summary Figures ---")

        output_dir = Path(output_dir)
        comparison_dir = output_dir / 'DurationComparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # Time conversion settings
        if self.config.show_real_time:
            time_scale = 1 / self.config.sampling_rate
            time_unit = 's'
        else:
            time_scale = 1
            time_unit = 'frames'

        n_durations = len(self.stimulus_durations)
        n_conditions = len(self.conditions)

        # Create display modes
        display_modes = []
        for mode in self.stimulus_modes:
            display_modes.extend(display_modes_for_mode(mode))
        n_display_modes = len(display_modes)

        stim_radii = self.stimulus_sizes / 2

        # Find low bias index
        if len(self.stimulus_bias_values) > 0:
            low_bias_idx = np.argmin(np.abs(self.stimulus_bias_values - 0.25))
        else:
            low_bias_idx = 0

        # =====================================================================
        # Figure: EC50 Bar Plot
        # =====================================================================
        fig_ec50, axes = plt.subplots(1, n_display_modes, figsize=(4*n_display_modes, 4))
        if n_display_modes == 1:
            axes = [axes]

        cmap = plt.cm.viridis(np.linspace(0, 1, n_durations))

        for m, display_mode in enumerate(display_modes):
            metrics_key = display_mode.replace('_', '')
            ax = axes[m]

            ec50_matrix = np.zeros((n_conditions, n_durations))
            for d, duration in enumerate(self.stimulus_durations):
                dur_key = f'dur_{duration}'
                for c, condition in enumerate(self.conditions):
                    if dur_key in self.gating and condition in self.gating[dur_key]:
                        if metrics_key in self.gating[dur_key][condition]:
                            ec50_matrix[c, d] = self.gating[dur_key][condition][metrics_key].get('EC50', np.nan)

            x = np.arange(n_conditions)
            width = 0.8 / n_durations

            for d in range(n_durations):
                offset = (d - n_durations/2 + 0.5) * width
                ax.bar(x + offset, ec50_matrix[:, d], width, color=cmap[d])

            ax.set_xticks(x)
            ax.set_xticklabels(self.conditions)
            ax.set_ylabel('EC50 (stimulus size)')
            ax.set_title(f'EC50 by Condition: {display_mode.replace("_", " ")}')
            if self.config.show_real_time:
                labels = [f'{d/self.config.sampling_rate:.1f} s' for d in self.stimulus_durations]
            else:
                labels = [f'{d} frames' for d in self.stimulus_durations]
            ax.legend(labels, loc='best')
            ax.grid(True, alpha=0.3)

        fig_ec50.suptitle('Half-Maximal Response Threshold (EC50)')
        fig_ec50.tight_layout()
        fig_ec50.savefig(comparison_dir / 'EC50_BarPlot.png', dpi=150)
        fig_ec50.savefig(comparison_dir / 'EC50_BarPlot.pdf')
        plt.close(fig_ec50)
        print("  Saved: EC50_BarPlot")

        # =====================================================================
        # Figure: Hill Coefficient Plot
        # =====================================================================
        fig_hill, axes = plt.subplots(1, n_display_modes, figsize=(4*n_display_modes, 4))
        if n_display_modes == 1:
            axes = [axes]

        for m, display_mode in enumerate(display_modes):
            metrics_key = display_mode.replace('_', '')
            ax = axes[m]

            hill_matrix = np.zeros((n_conditions, n_durations))
            for d, duration in enumerate(self.stimulus_durations):
                dur_key = f'dur_{duration}'
                for c, condition in enumerate(self.conditions):
                    if dur_key in self.gating and condition in self.gating[dur_key]:
                        if metrics_key in self.gating[dur_key][condition]:
                            hill_matrix[c, d] = self.gating[dur_key][condition][metrics_key].get('hill_coefficient', np.nan)

            x = np.arange(n_conditions)
            width = 0.8 / n_durations

            for d in range(n_durations):
                offset = (d - n_durations/2 + 0.5) * width
                ax.bar(x + offset, hill_matrix[:, d], width, color=cmap[d])

            ax.axhline(1, color='k', linestyle='--', linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(self.conditions)
            ax.set_ylabel('Hill Coefficient (n)')
            ax.set_title(f'Hill Coefficient: {display_mode.replace("_", " ")}')
            if self.config.show_real_time:
                labels = [f'{d/self.config.sampling_rate:.1f} s' for d in self.stimulus_durations]
            else:
                labels = [f'{d} frames' for d in self.stimulus_durations]
            ax.legend(labels, loc='best')
            ax.grid(True, alpha=0.3)

        fig_hill.suptitle('Dose-Response Steepness (Hill Coefficient)')
        fig_hill.tight_layout()
        fig_hill.savefig(comparison_dir / 'HillCoefficient.png', dpi=150)
        fig_hill.savefig(comparison_dir / 'HillCoefficient.pdf')
        plt.close(fig_hill)
        print("  Saved: HillCoefficient")

        # =====================================================================
        # Figure: Net Propagation Heatmap - All Durations
        # =====================================================================
        # Get global color limits
        all_net_prop = []
        for dur_key in self.metrics:
            for condition in self.conditions:
                if condition in self.metrics[dur_key]:
                    for dm in display_modes:
                        mk = dm.replace('_', '')
                        if mk in self.metrics[dur_key][condition]:
                            for s in range(len(self.stimulus_sizes)):
                                net_prop = self.metrics[dur_key][condition][mk]['max_blob_extent'][:, s] - stim_radii[s]
                                all_net_prop.extend(net_prop)

        max_abs_val = np.max(np.abs(all_net_prop)) if all_net_prop else 1
        global_clims = (-max_abs_val, max_abs_val)

        fig_heatmap, axes = plt.subplots(n_display_modes, n_durations,
                                         figsize=(3*n_durations, 3*n_display_modes))
        if n_display_modes == 1:
            axes = axes.reshape(1, -1)
        if n_durations == 1:
            axes = axes.reshape(-1, 1)

        for m, display_mode in enumerate(display_modes):
            metrics_key = display_mode.replace('_', '')

            for d, duration in enumerate(self.stimulus_durations):
                dur_key = f'dur_{duration}'
                ax = axes[m, d]

                prop_matrix = np.zeros((n_conditions, len(self.stimulus_sizes)))
                for c, condition in enumerate(self.conditions):
                    if dur_key in self.metrics and condition in self.metrics[dur_key]:
                        if metrics_key in self.metrics[dur_key][condition]:
                            for s in range(len(self.stimulus_sizes)):
                                mean_extent = np.mean(self.metrics[dur_key][condition][metrics_key]['max_blob_extent'][:, s])
                                prop_matrix[c, s] = mean_extent - stim_radii[s]

                im = ax.imshow(prop_matrix, cmap=redblue(), aspect='auto',
                              vmin=global_clims[0], vmax=global_clims[1])

                if d == n_durations - 1:
                    plt.colorbar(im, ax=ax, label='Net Prop. (px)')

                ax.set_xticks(np.arange(len(self.stimulus_sizes)))
                ax.set_xticklabels(self.stimulus_sizes)
                ax.set_yticks(np.arange(n_conditions))
                ax.set_yticklabels(self.conditions)

                if m == n_display_modes - 1:
                    ax.set_xlabel('Size (px)')
                if d == 0:
                    ax.set_ylabel(display_mode.replace('_', ' '))

                if self.config.show_real_time:
                    ax.set_title(f'{duration/self.config.sampling_rate:.1f} s')
                else:
                    ax.set_title(f'{duration} frames')

        fig_heatmap.suptitle('Net Propagation Beyond Stimulus (All Durations)')
        fig_heatmap.tight_layout()
        fig_heatmap.savefig(comparison_dir / 'NetPropagationHeatmap_AllDurations.png', dpi=150)
        fig_heatmap.savefig(comparison_dir / 'NetPropagationHeatmap_AllDurations.pdf')
        plt.close(fig_heatmap)
        print("  Saved: NetPropagationHeatmap_AllDurations")

        # =====================================================================
        # Figure: EC50 vs Duration
        # =====================================================================
        fig_ec50_dur, axes = plt.subplots(1, n_display_modes, figsize=(4*n_display_modes, 4))
        if n_display_modes == 1:
            axes = [axes]

        for m, display_mode in enumerate(display_modes):
            metrics_key = display_mode.replace('_', '')
            ax = axes[m]

            for condition in self.conditions:
                ec50_vals = []
                for duration in self.stimulus_durations:
                    dur_key = f'dur_{duration}'
                    if dur_key in self.gating and condition in self.gating[dur_key]:
                        if metrics_key in self.gating[dur_key][condition]:
                            ec50_vals.append(self.gating[dur_key][condition][metrics_key].get('EC50', np.nan))
                        else:
                            ec50_vals.append(np.nan)
                    else:
                        ec50_vals.append(np.nan)

                color = self.config.colors[condition]
                if self.config.show_real_time:
                    x_vals = np.array(self.stimulus_durations) / self.config.sampling_rate
                else:
                    x_vals = self.stimulus_durations

                ax.plot(x_vals, ec50_vals, 'o-', color=color, markerfacecolor=color,
                       linewidth=2, label=condition)

            ax.set_xlabel(f'Stimulus Duration ({time_unit})')
            ax.set_ylabel('EC50 (stimulus size)')
            ax.set_title(f'EC50 vs Duration: {display_mode.replace("_", " ")}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        fig_ec50_dur.suptitle('Sensitivity Changes with Stimulus Duration')
        fig_ec50_dur.tight_layout()
        fig_ec50_dur.savefig(comparison_dir / 'EC50_vs_Duration.png', dpi=150)
        fig_ec50_dur.savefig(comparison_dir / 'EC50_vs_Duration.pdf')
        plt.close(fig_ec50_dur)
        print("  Saved: EC50_vs_Duration")

        # =====================================================================
        # Figure: Threshold vs Duration
        # =====================================================================
        fig_thresh, axes = plt.subplots(1, n_display_modes, figsize=(4*n_display_modes, 4))
        if n_display_modes == 1:
            axes = [axes]

        for m, display_mode in enumerate(display_modes):
            metrics_key = display_mode.replace('_', '')
            ax = axes[m]

            for condition in self.conditions:
                thresh_vals = []
                for duration in self.stimulus_durations:
                    dur_key = f'dur_{duration}'
                    if dur_key in self.gating and condition in self.gating[dur_key]:
                        if metrics_key in self.gating[dur_key][condition]:
                            thresh_vals.append(self.gating[dur_key][condition][metrics_key].get('mean_threshold', np.nan))
                        else:
                            thresh_vals.append(np.nan)
                    else:
                        thresh_vals.append(np.nan)

                color = self.config.colors[condition]
                if self.config.show_real_time:
                    x_vals = np.array(self.stimulus_durations) / self.config.sampling_rate
                else:
                    x_vals = self.stimulus_durations

                ax.plot(x_vals, thresh_vals, 'o-', color=color, markerfacecolor=color,
                       linewidth=2, label=condition)

            ax.set_xlabel(f'Stimulus Duration ({time_unit})')
            ax.set_ylabel('Threshold Size (pixels)')
            ax.set_title(f'Threshold vs Duration: {display_mode.replace("_", " ")}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        fig_thresh.suptitle('Propagation Threshold Changes with Duration')
        fig_thresh.tight_layout()
        fig_thresh.savefig(comparison_dir / 'Threshold_vs_Duration.png', dpi=150)
        fig_thresh.savefig(comparison_dir / 'Threshold_vs_Duration.pdf')
        plt.close(fig_thresh)
        print("  Saved: Threshold_vs_Duration")

        # =====================================================================
        # Figure: Combined Propagation Dynamics
        # =====================================================================
        if self.config.show_real_time:
            velocity_scale = self.config.sampling_rate
            velocity_unit = 'px/s'
        else:
            velocity_scale = 1
            velocity_unit = 'px/frame'

        fig_prop_dyn, axes = plt.subplots(2, n_durations, figsize=(4*n_durations, 6))
        if n_durations == 1:
            axes = axes.reshape(-1, 1)

        for d, duration in enumerate(self.stimulus_durations):
            dur_key = f'dur_{duration}'

            # Compute prop dynamics for this duration
            if dur_key in self.data:
                prop_dynamics = self._compute_prop_dynamics(self.data[dur_key],
                                                           self.metrics.get(dur_key, {}), duration)
            else:
                continue

            # Row 1: Peak Propagation Velocity
            ax = axes[0, d]
            for condition in self.conditions:
                if condition not in prop_dynamics:
                    continue

                velocity = prop_dynamics[condition].get('max_velocity', np.array([]))
                if velocity.size == 0:
                    continue

                mean_vel = np.mean(velocity, axis=0) * velocity_scale
                sem_vel = np.std(velocity, axis=0) / np.sqrt(velocity.shape[0]) * velocity_scale

                color = self.config.colors[condition]
                ax.errorbar(self.stimulus_sizes, mean_vel, yerr=sem_vel, fmt='o-',
                           color=color, markerfacecolor=color, linewidth=1.5, label=condition)

            ax.set_xlabel('Stimulus Size (pixels)')
            if d == 0:
                ax.set_ylabel(f'Max Velocity ({velocity_unit})')
                ax.legend(loc='best')
            if self.config.show_real_time:
                ax.set_title(f'{duration/self.config.sampling_rate:.1f}s')
            else:
                ax.set_title(f'{duration} frames')
            ax.grid(True, alpha=0.3)

            # Row 2: Time to Peak Spread
            ax = axes[1, d]
            for condition in self.conditions:
                if condition not in prop_dynamics:
                    continue

                ttm = prop_dynamics[condition].get('time_to_max', np.array([]))
                if ttm.size == 0:
                    continue

                mean_ttm = np.mean(ttm, axis=0) * time_scale
                sem_ttm = np.std(ttm, axis=0) / np.sqrt(ttm.shape[0]) * time_scale

                color = self.config.colors[condition]
                ax.errorbar(self.stimulus_sizes, mean_ttm, yerr=sem_ttm, fmt='o-',
                           color=color, markerfacecolor=color, linewidth=1.5)

            ax.set_xlabel('Stimulus Size (pixels)')
            if d == 0:
                ax.set_ylabel(f'Time to Max ({time_unit})')
            ax.grid(True, alpha=0.3)

        fig_prop_dyn.suptitle('Propagation Dynamics - All Durations')
        fig_prop_dyn.tight_layout()
        fig_prop_dyn.savefig(comparison_dir / 'PropagationDynamics_AllDurations.png', dpi=150)
        fig_prop_dyn.savefig(comparison_dir / 'PropagationDynamics_AllDurations.pdf')
        plt.close(fig_prop_dyn)
        print("  Saved: PropagationDynamics_AllDurations")

        # =====================================================================
        # Figure: Combined Pre-Stimulus State Effects
        # =====================================================================
        fig_prestim, axes = plt.subplots(3, n_durations, figsize=(4*n_durations, 9))
        if n_durations == 1:
            axes = axes.reshape(-1, 1)

        for d, duration in enumerate(self.stimulus_durations):
            dur_key = f'dur_{duration}'

            if dur_key in self.data:
                pre_stim_effects = self._compute_prestim_effects(self.data[dur_key], duration)
            else:
                continue

            # Row 1: Baseline Moran's I
            ax = axes[0, d]
            for c, condition in enumerate(self.conditions):
                if condition not in pre_stim_effects:
                    continue

                color = self.config.colors[condition]
                morans_vals = pre_stim_effects[condition].get('baseline_moransI', [])
                morans_vals = np.array(morans_vals).flatten()
                morans_vals = morans_vals[~np.isnan(morans_vals) & (morans_vals != 0)]

                if len(morans_vals) > 0:
                    jitter = 0.1 * (np.random.rand(len(morans_vals)) - 0.5)
                    ax.scatter(c + jitter, morans_vals, c=[color], s=30, alpha=0.6)
                    ax.plot([c-0.3, c+0.3], [np.mean(morans_vals)]*2, 'k-', linewidth=2)

            ax.set_xticks(range(len(self.conditions)))
            ax.set_xticklabels(self.conditions)
            if d == 0:
                ax.set_ylabel("Baseline Moran's I")
            if self.config.show_real_time:
                ax.set_title(f'{duration/self.config.sampling_rate:.1f}s')
            else:
                ax.set_title(f'{duration} frames')

            # Row 2: Baseline vs Peak correlation
            ax = axes[1, d]
            for c, condition in enumerate(self.conditions):
                if condition not in pre_stim_effects:
                    continue

                color = self.config.colors[condition]
                corr_vals = pre_stim_effects[condition].get('baseline_vs_peak', [])
                corr_vals = np.array(corr_vals).flatten()
                corr_vals = corr_vals[~np.isnan(corr_vals)]

                if len(corr_vals) > 0:
                    jitter = 0.1 * (np.random.rand(len(corr_vals)) - 0.5)
                    ax.scatter(c + jitter, corr_vals, c=[color], s=30, alpha=0.6)
                    ax.plot([c-0.3, c+0.3], [np.mean(corr_vals)]*2, 'k-', linewidth=2)

            ax.set_xticks(range(len(self.conditions)))
            ax.set_xticklabels(self.conditions)
            if d == 0:
                ax.set_ylabel('r (Baseline vs Peak)')
            ax.axhline(0, color='k', linestyle='--')

            # Row 3: Baseline vs Extent correlation
            ax = axes[2, d]
            for c, condition in enumerate(self.conditions):
                if condition not in pre_stim_effects:
                    continue

                color = self.config.colors[condition]
                corr_vals = pre_stim_effects[condition].get('baseline_vs_extent', [])
                corr_vals = np.array(corr_vals).flatten()
                corr_vals = corr_vals[~np.isnan(corr_vals)]

                if len(corr_vals) > 0:
                    jitter = 0.1 * (np.random.rand(len(corr_vals)) - 0.5)
                    ax.scatter(c + jitter, corr_vals, c=[color], s=30, alpha=0.6)
                    ax.plot([c-0.3, c+0.3], [np.mean(corr_vals)]*2, 'k-', linewidth=2)

            ax.set_xticks(range(len(self.conditions)))
            ax.set_xticklabels(self.conditions)
            if d == 0:
                ax.set_ylabel('r (Baseline vs Extent)')
            ax.axhline(0, color='k', linestyle='--')

        fig_prestim.suptitle('Pre-Stimulus State Effects - All Durations')
        fig_prestim.tight_layout()
        fig_prestim.savefig(comparison_dir / 'PreStimEffects_AllDurations.png', dpi=150)
        fig_prestim.savefig(comparison_dir / 'PreStimEffects_AllDurations.pdf')
        plt.close(fig_prestim)
        print("  Saved: PreStimEffects_AllDurations")

        # =====================================================================
        # Figure: Propagation Distance - All Durations
        # =====================================================================
        # Define time label for multi-duration figures
        if self.config.show_real_time:
            time_label = 'Time (s from stim onset)'
        else:
            time_label = 'Time (frames from stim onset)'

        # Use representative stimulus size (10px or closest)
        rep_size = 10
        size_idx = np.argmin(np.abs(self.stimulus_sizes - rep_size))

        fig_prop_dist, axes = plt.subplots(n_display_modes, n_durations,
                                           figsize=(4*n_durations, 3*n_display_modes))
        if n_display_modes == 1:
            axes = axes.reshape(1, -1)
        if n_durations == 1:
            axes = axes.reshape(-1, 1)

        for d, duration in enumerate(self.stimulus_durations):
            dur_key = f'dur_{duration}'

            # Time vectors for this duration (MATLAB convention: time=0 at last pre-stim frame)
            total_frames = self.pre_stim_frames + duration + self.post_stim_frames
            time_vec_dur = np.arange(total_frames) + 1 - self.pre_stim_frames
            if self.config.show_real_time:
                time_vec_plot = time_vec_dur / self.config.sampling_rate
                stim_on_time = duration / self.config.sampling_rate
                dur_title = f'{duration/self.config.sampling_rate:.1f}s'
            else:
                time_vec_plot = time_vec_dur
                stim_on_time = duration
                dur_title = f'{duration} frames'

            for m, display_mode in enumerate(display_modes):
                ax = axes[m, d]

                data_key = data_key_from_display_mode(display_mode)

                for condition in self.conditions:
                    if dur_key not in self.data:
                        continue
                    if condition not in self.data[dur_key]:
                        continue
                    if data_key not in self.data[dur_key][condition]:
                        continue

                    prop_data = self.data[dur_key][condition][data_key].get('stimulus_blob_extent')
                    if prop_data is None:
                        continue

                    if is_high_bias_display_mode(display_mode) and prop_data.ndim == 5:
                        propagation = prop_data[:, size_idx, -1, :, :]
                    elif is_low_bias_display_mode(display_mode) and prop_data.ndim == 5:
                        propagation = prop_data[:, size_idx, low_bias_idx, :, :]
                    else:
                        propagation = prop_data[:, size_idx, :, :]

                    mean_prop = np.mean(np.mean(propagation, axis=0), axis=0)

                    color = self.config.colors[condition]
                    ax.plot(time_vec_plot, mean_prop, color=color, linewidth=2, label=condition)

                # Mark stimulus period
                ax.axvline(0, color='k', linestyle='--', linewidth=1)
                ax.axvline(stim_on_time, color='k', linestyle='--', linewidth=1)
                ax.axvspan(0, stim_on_time, alpha=0.2, color='gray')

                # Mark stimulus radius
                ax.axhline(self.stimulus_sizes[size_idx] / 2, color='r', linestyle=':', linewidth=1)

                ax.set_xlabel(time_label)
                if d == 0:
                    ax.set_ylabel(display_mode.replace('_', ' '))
                if m == 0:
                    ax.set_title(dur_title)
                if d == 0 and m == 0:
                    ax.legend(loc='best')
                ax.grid(True, alpha=0.3)

        fig_prop_dist.suptitle('Propagation Distance - All Durations')
        fig_prop_dist.tight_layout()
        fig_prop_dist.savefig(comparison_dir / 'PropagationDistance_AllDurations.png', dpi=150)
        fig_prop_dist.savefig(comparison_dir / 'PropagationDistance_AllDurations.pdf')
        plt.close(fig_prop_dist)
        print("  Saved: PropagationDistance_AllDurations")

        # =====================================================================
        # Figure: Moran's I Dynamics - All Durations
        # =====================================================================
        # Use representative stimulus size (8px or closest)
        rep_size_morans = 8
        size_idx_morans = np.argmin(np.abs(self.stimulus_sizes - rep_size_morans))

        fig_morans, axes = plt.subplots(3, n_durations, figsize=(4*n_durations, 9))
        if n_durations == 1:
            axes = axes.reshape(-1, 1)

        for d, duration in enumerate(self.stimulus_durations):
            dur_key = f'dur_{duration}'

            # Time vectors for this duration (MATLAB convention: time=0 at last pre-stim frame)
            total_frames = self.pre_stim_frames + duration + self.post_stim_frames
            time_vec_dur = np.arange(total_frames) + 1 - self.pre_stim_frames
            if self.config.show_real_time:
                time_vec_plot = time_vec_dur / self.config.sampling_rate
                stim_on_time = duration / self.config.sampling_rate
                dur_title = f'{duration/self.config.sampling_rate:.1f}s'
            else:
                time_vec_plot = time_vec_dur
                stim_on_time = duration
                dur_title = f'{duration} frames'

            # Row 1: Moran's I time course
            ax = axes[0, d]
            for condition in self.conditions:
                if dur_key not in self.data:
                    continue
                if condition not in self.data[dur_key]:
                    continue
                if 'clamped' not in self.data[dur_key][condition]:
                    continue

                morans_data = self.data[dur_key][condition]['clamped'].get('morans_I')
                if morans_data is None:
                    continue

                morans_I = morans_data[:, size_idx_morans, :, :]
                mean_morans = np.mean(np.mean(morans_I, axis=0), axis=0)

                color = self.config.colors[condition]
                ax.plot(time_vec_plot, mean_morans, color=color, linewidth=2, label=condition)

            ax.axvline(0, color='k', linestyle='--')
            ax.axvline(stim_on_time, color='k', linestyle='--')
            ax.set_xlabel(time_label)
            if d == 0:
                ax.set_ylabel("Moran's I")
                ax.legend(loc='best')
            ax.set_title(dur_title)
            ax.grid(True, alpha=0.3)

            # Row 2: Moran's I increase vs size
            ax = axes[1, d]
            for c, condition in enumerate(self.conditions):
                if dur_key not in self.data:
                    continue
                if condition not in self.data[dur_key]:
                    continue
                if 'clamped' not in self.data[dur_key][condition]:
                    continue

                morans_data = self.data[dur_key][condition]['clamped'].get('morans_I')
                if morans_data is None:
                    continue

                # Compute Moran's I increase for each size
                morans_increase = []
                for s in range(len(self.stimulus_sizes)):
                    morans_s = morans_data[:, s, :, :]  # [nSims x nReps x nFrames]
                    mean_morans_s = np.mean(np.mean(morans_s, axis=0), axis=0)  # [nFrames]
                    baseline_morans = np.mean(mean_morans_s[:self.pre_stim_frames])
                    stim_end = self.pre_stim_frames + duration
                    max_morans = np.max(mean_morans_s[self.pre_stim_frames:stim_end])
                    morans_increase.append(max_morans - baseline_morans)

                color = self.config.colors[condition]
                x_pos = np.arange(len(self.stimulus_sizes)) + c * 0.15 - 0.225
                ax.bar(x_pos, morans_increase, width=0.15, color=color, label=condition)

            ax.set_xticks(np.arange(len(self.stimulus_sizes)))
            ax.set_xticklabels(self.stimulus_sizes)
            ax.set_xlabel('Stimulus Size (px)')
            if d == 0:
                ax.set_ylabel("Moran's I Increase")
            ax.grid(True, alpha=0.3, axis='y')

            # Row 3: Moran's I change (post - pre)
            ax = axes[2, d]
            for c, condition in enumerate(self.conditions):
                if dur_key not in self.data:
                    continue
                if condition not in self.data[dur_key]:
                    continue
                if 'clamped' not in self.data[dur_key][condition]:
                    continue

                morans_data = self.data[dur_key][condition]['clamped'].get('morans_I')
                if morans_data is None:
                    continue

                # Compute Moran's I change (post - pre) for each size
                morans_change = []
                for s in range(len(self.stimulus_sizes)):
                    morans_s = morans_data[:, s, :, :]
                    mean_morans_s = np.mean(np.mean(morans_s, axis=0), axis=0)
                    pre_morans = np.mean(mean_morans_s[:self.pre_stim_frames])
                    stim_end = self.pre_stim_frames + duration
                    post_start = stim_end
                    post_end = min(post_start + self.post_stim_frames // 2, len(mean_morans_s))
                    post_morans = np.mean(mean_morans_s[post_start:post_end])
                    morans_change.append(post_morans - pre_morans)

                color = self.config.colors[condition]
                x_pos = np.arange(len(self.stimulus_sizes)) + c * 0.15 - 0.225
                ax.bar(x_pos, morans_change, width=0.15, color=color)

            ax.axhline(0, color='k', linestyle='--')
            ax.set_xticks(np.arange(len(self.stimulus_sizes)))
            ax.set_xticklabels(self.stimulus_sizes)
            ax.set_xlabel('Stimulus Size (px)')
            if d == 0:
                ax.set_ylabel("Moran's I Change (post-pre)")
            ax.grid(True, alpha=0.3, axis='y')

        fig_morans.suptitle("Moran's I Dynamics - All Durations")
        fig_morans.tight_layout()
        fig_morans.savefig(comparison_dir / 'MoransIDynamics_AllDurations.png', dpi=150)
        fig_morans.savefig(comparison_dir / 'MoransIDynamics_AllDurations.pdf')
        plt.close(fig_morans)
        print("  Saved: MoransIDynamics_AllDurations")

        # =====================================================================
        # Figure: Expert Summary
        # =====================================================================
        comp_conditions = ['Expert', 'NoSpout']
        longest_dur = max(self.stimulus_durations)
        longest_dur_key = f'dur_{longest_dur}'

        fig_summary, axes = plt.subplots(n_display_modes, 3, figsize=(12, 3*n_display_modes))
        if n_display_modes == 1:
            axes = axes.reshape(1, -1)

        for m, display_mode in enumerate(display_modes):
            metrics_key = display_mode.replace('_', '')

            # Column 1: EC50 Comparison
            ax = axes[m, 0]
            ec50_means = []
            ec50_sems = []
            for condition in self.conditions:
                ec50_vals = []
                for duration in self.stimulus_durations:
                    dur_key = f'dur_{duration}'
                    if dur_key in self.gating and condition in self.gating[dur_key]:
                        if metrics_key in self.gating[dur_key][condition]:
                            ec50_vals.append(self.gating[dur_key][condition][metrics_key].get('EC50', np.nan))
                ec50_vals = np.array(ec50_vals)
                ec50_means.append(np.nanmean(ec50_vals))
                ec50_sems.append(np.nanstd(ec50_vals) / np.sqrt(np.sum(~np.isnan(ec50_vals))))

            x = np.arange(n_conditions)
            colors = [self.config.colors[c] for c in self.conditions]
            ax.bar(x, ec50_means, color=colors, edgecolor='k')
            ax.errorbar(x, ec50_means, yerr=ec50_sems, fmt='none', color='k', linewidth=1.5)
            ax.set_xticks(x)
            ax.set_xticklabels(self.conditions)
            ax.set_ylabel(display_mode.replace('_', ' '))
            if m == 0:
                ax.set_title('A) Sensitivity (EC50)')

            # Column 2: Hill Coefficient
            ax = axes[m, 1]
            hill_means = []
            hill_sems = []
            for condition in self.conditions:
                hill_vals = []
                for duration in self.stimulus_durations:
                    dur_key = f'dur_{duration}'
                    if dur_key in self.gating and condition in self.gating[dur_key]:
                        if metrics_key in self.gating[dur_key][condition]:
                            hill_vals.append(self.gating[dur_key][condition][metrics_key].get('hill_coefficient', np.nan))
                hill_vals = np.array(hill_vals)
                hill_means.append(np.nanmean(hill_vals))
                hill_sems.append(np.nanstd(hill_vals) / np.sqrt(np.sum(~np.isnan(hill_vals))))

            ax.bar(x, hill_means, color=colors, edgecolor='k')
            ax.errorbar(x, hill_means, yerr=hill_sems, fmt='none', color='k', linewidth=1.5)
            ax.axhline(1, color='k', linestyle='--')
            ax.set_xticks(x)
            ax.set_xticklabels(self.conditions)
            if m == 0:
                ax.set_title('B) Response Steepness (Hill n)')

            # Column 3: Expert vs NoSpout dose-response
            ax = axes[m, 2]
            if longest_dur_key in self.metrics:
                for condition in comp_conditions:
                    if condition not in self.metrics[longest_dur_key]:
                        continue
                    if metrics_key not in self.metrics[longest_dur_key][condition]:
                        continue

                    extent = self.metrics[longest_dur_key][condition][metrics_key]['max_blob_extent']
                    net_prop = extent - stim_radii
                    mean_net = np.mean(net_prop, axis=0)
                    sem_net = np.std(net_prop, axis=0) / np.sqrt(net_prop.shape[0])

                    color = self.config.colors[condition]
                    ax.errorbar(self.stimulus_sizes, mean_net, yerr=sem_net, fmt='o-',
                               color=color, markerfacecolor=color, linewidth=1.5, label=condition)

            ax.axhline(0, color='k', linestyle='--')
            ax.set_xlabel('Stimulus Size (px)')
            ax.grid(True, alpha=0.3)
            if m == 0:
                if self.config.show_real_time:
                    ax.set_title(f'C) Expert vs NoSpout ({longest_dur/self.config.sampling_rate:.1f} s)')
                else:
                    ax.set_title(f'C) Expert vs NoSpout ({longest_dur} fr)')
                ax.legend(loc='best')

        fig_summary.suptitle('Summary: How Expert Differs from Other Conditions (All Modes)')
        fig_summary.tight_layout()
        fig_summary.savefig(comparison_dir / 'ExpertSummary.png', dpi=150)
        fig_summary.savefig(comparison_dir / 'ExpertSummary.pdf')
        plt.close(fig_summary)
        print("  Saved: ExpertSummary")

    def plot_bias_analysis(self, output_dir):
        """
        Generate bias-specific analysis figures.

        Parameters:
        -----------
        output_dir : str or Path
            Output directory for figures
        """
        print("\n--- Creating Bias Analysis Figures ---")

        output_dir = Path(output_dir)
        bias_dir = output_dir / 'BiasComparison'
        bias_dir.mkdir(parents=True, exist_ok=True)

        n_bias_vals = len(self.stimulus_bias_values)
        n_conditions = len(self.conditions)
        n_sizes = len(self.stimulus_sizes)
        stim_radii = self.stimulus_sizes / 2

        # Time conversion settings
        if self.config.show_real_time:
            time_scale = 1 / self.config.sampling_rate
            time_unit = 's'
            time_label = 'Time (s from stim onset)'
        else:
            time_scale = 1
            time_unit = 'frames'
            time_label = 'Time (frames from stim onset)'

        # Find low bias index
        low_bias_idx = np.argmin(np.abs(self.stimulus_bias_values - 0.25))

        # Color gradient for bias values
        bias_colors = plt.cm.viridis(np.linspace(0, 1, n_bias_vals))

        # Compute bias metrics for all durations
        bias_metrics = self._compute_bias_metrics()

        # Process each duration
        for duration in self.stimulus_durations:
            dur_key = f'dur_{duration}'

            if self.config.show_real_time:
                dur_str = f'{duration/self.config.sampling_rate:.1f}s'
            else:
                dur_str = f'{duration}frames'

            dur_bias_dir = bias_dir / dur_str
            dur_bias_dir.mkdir(parents=True, exist_ok=True)

            if dur_key not in bias_metrics:
                continue

            # =================================================================
            # Figure: EC50 vs Bias Value
            # =================================================================
            fig1, ax = plt.subplots(figsize=(8, 5))

            for condition in self.conditions:
                if condition not in bias_metrics[dur_key]:
                    continue

                ec50_vals = bias_metrics[dur_key][condition].get('EC50', [])
                if len(ec50_vals) == 0:
                    continue

                color = self.config.colors[condition]
                ax.plot(self.stimulus_bias_values, ec50_vals, 'o-', color=color,
                       markerfacecolor=color, linewidth=2, markersize=6, label=condition)

            ax.set_xlabel('Bias Value')
            ax.set_ylabel('EC50 (stimulus size)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([min(self.stimulus_bias_values)*0.8, max(self.stimulus_bias_values)*1.2])
            ax.set_title(f'EC50 Sensitivity vs Bias Strength ({dur_str})')

            fig1.tight_layout()
            fig1.savefig(dur_bias_dir / 'EC50_vs_Bias.png', dpi=150)
            fig1.savefig(dur_bias_dir / 'EC50_vs_Bias.pdf')
            plt.close(fig1)
            print(f"  Saved: {dur_str}/EC50_vs_Bias")

            # =================================================================
            # Figure: Hill Coefficient vs Bias Value
            # =================================================================
            fig2, ax = plt.subplots(figsize=(8, 5))

            for condition in self.conditions:
                if condition not in bias_metrics[dur_key]:
                    continue

                hill_vals = bias_metrics[dur_key][condition].get('hill_coefficient', [])
                if len(hill_vals) == 0:
                    continue

                color = self.config.colors[condition]
                ax.plot(self.stimulus_bias_values, hill_vals, 'o-', color=color,
                       markerfacecolor=color, linewidth=2, markersize=6, label=condition)

            ax.axhline(1, color='k', linestyle='--', label='n=1')
            ax.set_xlabel('Bias Value')
            ax.set_ylabel('Hill Coefficient (n)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([min(self.stimulus_bias_values)*0.8, max(self.stimulus_bias_values)*1.2])
            ax.set_title(f'Response Steepness vs Bias Strength ({dur_str})')

            fig2.tight_layout()
            fig2.savefig(dur_bias_dir / 'Hill_vs_Bias.png', dpi=150)
            fig2.savefig(dur_bias_dir / 'Hill_vs_Bias.pdf')
            plt.close(fig2)
            print(f"  Saved: {dur_str}/Hill_vs_Bias")

            # =================================================================
            # Figure: Dose-Response Curves at Different Biases
            # =================================================================
            fig3, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()

            for c, condition in enumerate(self.conditions):
                ax = axes[c]

                if condition not in bias_metrics[dur_key]:
                    ax.set_title(f'{condition}: No data')
                    continue

                net_prop = bias_metrics[dur_key][condition].get('mean_net_propagation', np.array([]))
                if net_prop.size == 0:
                    ax.set_title(f'{condition}: No data')
                    continue

                for b in range(n_bias_vals):
                    if net_prop.ndim == 2 and net_prop.shape[1] == n_bias_vals:
                        ax.plot(self.stimulus_sizes, net_prop[:, b], 'o-',
                               color=bias_colors[b], linewidth=1.5, markersize=5,
                               markerfacecolor=bias_colors[b])

                ax.axhline(0, color='k', linestyle='--', linewidth=1)
                ax.set_xlabel('Stimulus Size (px)')
                ax.set_ylabel('Net Propagation (px)')
                ax.set_title(condition)
                ax.grid(True, alpha=0.3)

                if c == n_conditions - 1:
                    # Add colorbar legend
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                               norm=plt.Normalize(vmin=min(self.stimulus_bias_values),
                                                                 vmax=max(self.stimulus_bias_values)))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax)
                    cbar.set_label('Bias Value')

            fig3.suptitle(f'Dose-Response Curves at Different Bias Values ({dur_str})')
            fig3.tight_layout()
            fig3.savefig(dur_bias_dir / 'DoseResponse_BiasGradient.png', dpi=150)
            fig3.savefig(dur_bias_dir / 'DoseResponse_BiasGradient.pdf')
            plt.close(fig3)
            print(f"  Saved: {dur_str}/DoseResponse_BiasGradient")

            # =================================================================
            # Figure: Bias x Size Heatmap
            # =================================================================
            fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for c, condition in enumerate(self.conditions):
                ax = axes[c]

                if condition not in bias_metrics[dur_key]:
                    ax.set_title(f'{condition}: No data')
                    continue

                net_prop = bias_metrics[dur_key][condition].get('mean_net_propagation', np.array([]))
                if net_prop.size == 0:
                    ax.set_title(f'{condition}: No data')
                    continue

                # Transpose for heatmap (rows=bias, cols=sizes)
                if net_prop.ndim == 2:
                    net_prop_matrix = net_prop.T
                else:
                    continue

                max_abs = np.nanmax(np.abs(net_prop_matrix)) if np.any(~np.isnan(net_prop_matrix)) else 1
                im = ax.imshow(net_prop_matrix, cmap=redblue(), aspect='auto',
                              vmin=-max_abs, vmax=max_abs)
                plt.colorbar(im, ax=ax)

                ax.set_xticks(np.arange(n_sizes))
                ax.set_xticklabels(self.stimulus_sizes)
                ax.set_yticks(np.arange(n_bias_vals))
                ax.set_yticklabels([f'{b:.2f}' for b in self.stimulus_bias_values])
                ax.set_xlabel('Stimulus Size (px)')
                ax.set_ylabel('Bias Value')
                ax.set_title(condition)

            fig4.suptitle(f'Net Propagation: Bias x Size Interaction ({dur_str})')
            fig4.tight_layout()
            fig4.savefig(dur_bias_dir / 'Bias_Size_Heatmap.png', dpi=150)
            fig4.savefig(dur_bias_dir / 'Bias_Size_Heatmap.pdf')
            plt.close(fig4)
            print(f"  Saved: {dur_str}/Bias_Size_Heatmap")

            # =================================================================
            # Figure: Condition Comparison Across Biases
            # =================================================================
            fig5, axes = plt.subplots(3, 2, figsize=(12, 12))

            # Panel A: EC50 bar plot for each bias
            ax = axes[0, 0]
            ec50_matrix = np.zeros((n_conditions, n_bias_vals))
            for c, condition in enumerate(self.conditions):
                if condition in bias_metrics[dur_key]:
                    ec50_vals = bias_metrics[dur_key][condition].get('EC50', np.zeros(n_bias_vals))
                    if len(ec50_vals) == n_bias_vals:
                        ec50_matrix[c, :] = ec50_vals

            x = np.arange(n_conditions)
            width = 0.8 / n_bias_vals

            for b in range(n_bias_vals):
                offset = (b - n_bias_vals/2 + 0.5) * width
                ax.bar(x + offset, ec50_matrix[:, b], width, color=bias_colors[b])

            ax.set_xticks(x)
            ax.set_xticklabels(self.conditions)
            ax.set_ylabel('EC50 (stimulus size)')
            ax.set_title('A) EC50 by Condition at Each Bias')

            # Panel B: Attention effect (Expert vs NoSpout)
            ax = axes[0, 1]
            if 'Expert' in bias_metrics[dur_key] and 'NoSpout' in bias_metrics[dur_key]:
                ec50_expert = bias_metrics[dur_key]['Expert'].get('EC50', [])
                ec50_nospout = bias_metrics[dur_key]['NoSpout'].get('EC50', [])
                if len(ec50_expert) == n_bias_vals and len(ec50_nospout) == n_bias_vals:
                    ec50_diff = np.array(ec50_nospout) - np.array(ec50_expert)
                    ax.bar(self.stimulus_bias_values, ec50_diff, width=0.1, color=[0.4, 0.6, 0.8])
                    ax.axhline(0, color='k', linestyle='--')
                    ax.set_xlabel('Bias Value')
                    ax.set_ylabel(r'$\Delta$ EC50 (NoSpout - Expert)')
                    ax.set_title('B) Attention Advantage in Sensitivity')
                    ax.grid(True, alpha=0.3)

            # Panels C-F: Propagation success heatmaps
            panel_labels = ['C', 'D', 'E', 'F']
            for c, condition in enumerate(self.conditions):
                row = (c + 2) // 2
                col = c % 2
                ax = axes[row, col]

                if condition in bias_metrics[dur_key]:
                    prop_success = bias_metrics[dur_key][condition].get('propagation_success', np.array([]))
                    if prop_success.size > 0 and prop_success.ndim == 2:
                        im = ax.imshow(prop_success.T, cmap='viridis', aspect='auto', vmin=0, vmax=1)
                        plt.colorbar(im, ax=ax)
                        ax.set_xticks(np.arange(n_sizes))
                        ax.set_xticklabels(self.stimulus_sizes)
                        ax.set_yticks(np.arange(n_bias_vals))
                        ax.set_yticklabels([f'{b:.2f}' for b in self.stimulus_bias_values])
                        ax.set_xlabel('Stimulus Size (px)')
                        ax.set_ylabel('Bias Value')
                ax.set_title(f'{panel_labels[c]}) Propagation Success - {condition}')

            fig5.suptitle(f'Condition Differences Across Bias Values ({dur_str})')
            fig5.tight_layout()
            fig5.savefig(dur_bias_dir / 'Condition_Comparison.png', dpi=150)
            fig5.savefig(dur_bias_dir / 'Condition_Comparison.pdf')
            plt.close(fig5)
            print(f"  Saved: {dur_str}/Condition_Comparison")

            # =================================================================
            # Figure: Propagation Reliability
            # =================================================================
            fig6, axes = plt.subplots(2, 2, figsize=(10, 8))

            # Panel A: CoV heatmap (Expert)
            ax = axes[0, 0]
            if 'Expert' in bias_metrics[dur_key]:
                cov_data = bias_metrics[dur_key]['Expert'].get('cov_net_propagation', np.array([]))
                if cov_data.size > 0 and cov_data.ndim == 2:
                    im = ax.imshow(cov_data.T, cmap='hot_r', aspect='auto')
                    plt.colorbar(im, ax=ax)
                    ax.set_xticks(np.arange(n_sizes))
                    ax.set_xticklabels(self.stimulus_sizes)
                    ax.set_yticks(np.arange(n_bias_vals))
                    ax.set_yticklabels([f'{b:.2f}' for b in self.stimulus_bias_values])
                    ax.set_xlabel('Stimulus Size (px)')
                    ax.set_ylabel('Bias Value')
            ax.set_title('A) CoV of Propagation (Expert)')

            # Panel B: CoV heatmap (Naive)
            ax = axes[0, 1]
            if 'Naive' in bias_metrics[dur_key]:
                cov_data = bias_metrics[dur_key]['Naive'].get('cov_net_propagation', np.array([]))
                if cov_data.size > 0 and cov_data.ndim == 2:
                    im = ax.imshow(cov_data.T, cmap='hot_r', aspect='auto')
                    plt.colorbar(im, ax=ax)
                    ax.set_xticks(np.arange(n_sizes))
                    ax.set_xticklabels(self.stimulus_sizes)
                    ax.set_yticks(np.arange(n_bias_vals))
                    ax.set_yticklabels([f'{b:.2f}' for b in self.stimulus_bias_values])
                    ax.set_xlabel('Stimulus Size (px)')
                    ax.set_ylabel('Bias Value')
            ax.set_title('B) CoV of Propagation (Naive)')

            # Panel C: Mean CoV vs Bias
            ax = axes[1, 0]
            for condition in self.conditions:
                if condition not in bias_metrics[dur_key]:
                    continue

                cov_data = bias_metrics[dur_key][condition].get('cov_net_propagation', np.array([]))
                if cov_data.size == 0:
                    continue

                if cov_data.ndim == 2:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mean_cov = np.nanmean(cov_data, axis=0)
                else:
                    continue

                color = self.config.colors[condition]
                ax.plot(self.stimulus_bias_values, mean_cov, 'o-', color=color,
                       markerfacecolor=color, linewidth=2, markersize=6, label=condition)

            ax.set_xlabel('Bias Value')
            ax.set_ylabel('Mean CoV')
            ax.set_title('C) Propagation Variability vs Bias')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            # Panel D: Propagation success rate vs Bias
            ax = axes[1, 1]
            for condition in self.conditions:
                if condition not in bias_metrics[dur_key]:
                    continue

                prop_success = bias_metrics[dur_key][condition].get('propagation_success', np.array([]))
                if prop_success.size == 0:
                    continue

                if prop_success.ndim == 2:
                    mean_success = np.mean(prop_success, axis=0)
                else:
                    continue

                color = self.config.colors[condition]
                ax.plot(self.stimulus_bias_values, mean_success, 'o-', color=color,
                       markerfacecolor=color, linewidth=2, markersize=6, label=condition)

            ax.set_xlabel('Bias Value')
            ax.set_ylabel('Propagation Success Rate')
            ax.set_title('D) Mean Propagation Probability vs Bias')
            ax.legend(loc='best')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)

            fig6.suptitle(f'Propagation Reliability Across Bias Values ({dur_str})')
            fig6.tight_layout()
            fig6.savefig(dur_bias_dir / 'Propagation_Reliability.png', dpi=150)
            fig6.savefig(dur_bias_dir / 'Propagation_Reliability.pdf')
            plt.close(fig6)
            print(f"  Saved: {dur_str}/Propagation_Reliability")

            # =================================================================
            # Figure: Small Bias Deep Dive
            # =================================================================
            # Filter small bias values (bias <= 1.0)
            small_bias_mask = self.stimulus_bias_values <= 1.0
            small_bias_indices = np.where(small_bias_mask)[0]

            if len(small_bias_indices) > 0:
                small_bias_values = self.stimulus_bias_values[small_bias_indices]

                fig7, axes = plt.subplots(2, 3, figsize=(14, 8))

                # Panel A: EC50 at small biases (grouped bar)
                ax = axes[0, 0]
                bar_width = 0.8 / n_conditions
                for c, condition in enumerate(self.conditions):
                    if condition not in bias_metrics[dur_key]:
                        continue
                    ec50_small = bias_metrics[dur_key][condition]['EC50'][small_bias_indices]
                    x_pos = np.arange(len(small_bias_indices)) + c * bar_width - bar_width * (n_conditions - 1) / 2
                    color = self.config.colors[condition]
                    ax.bar(x_pos, ec50_small, bar_width * 0.9, color=color, label=condition)

                ax.set_xticks(np.arange(len(small_bias_indices)))
                ax.set_xticklabels([f'{v:.2f}' for v in small_bias_values])
                ax.set_xlabel('Bias Value')
                ax.set_ylabel('EC50 (stimulus size)')
                ax.set_title('A) EC50 at Small Biases')
                ax.legend(loc='best', fontsize=7)
                ax.grid(True, alpha=0.3, axis='y')

                # Panel B: Threshold at small biases
                ax = axes[0, 1]
                for c, condition in enumerate(self.conditions):
                    if condition not in bias_metrics[dur_key]:
                        continue
                    thresh_small = bias_metrics[dur_key][condition]['mean_threshold'][small_bias_indices]
                    x_pos = np.arange(len(small_bias_indices)) + c * bar_width - bar_width * (n_conditions - 1) / 2
                    color = self.config.colors[condition]
                    ax.bar(x_pos, thresh_small, bar_width * 0.9, color=color)

                ax.set_xticks(np.arange(len(small_bias_indices)))
                ax.set_xticklabels([f'{v:.2f}' for v in small_bias_values])
                ax.set_xlabel('Bias Value')
                ax.set_ylabel('Threshold Size (px)')
                ax.set_title('B) Threshold at Small Biases')
                ax.grid(True, alpha=0.3, axis='y')

                # Panel C: Hill coefficient at small biases
                ax = axes[0, 2]
                for c, condition in enumerate(self.conditions):
                    if condition not in bias_metrics[dur_key]:
                        continue
                    hill_small = bias_metrics[dur_key][condition]['hill_coefficient'][small_bias_indices]
                    x_pos = np.arange(len(small_bias_indices)) + c * bar_width - bar_width * (n_conditions - 1) / 2
                    color = self.config.colors[condition]
                    ax.bar(x_pos, hill_small, bar_width * 0.9, color=color)

                ax.axhline(1, color='k', linestyle='--')
                ax.set_xticks(np.arange(len(small_bias_indices)))
                ax.set_xticklabels([f'{v:.2f}' for v in small_bias_values])
                ax.set_xlabel('Bias Value')
                ax.set_ylabel('Hill Coefficient (n)')
                ax.set_title('C) Hill Coeff at Small Biases')
                ax.grid(True, alpha=0.3, axis='y')

                # Panel D: Expert vs Naive dose-response at smallest bias
                ax = axes[1, 0]
                smallest_bias_idx = small_bias_indices[0]
                comp_conds = ['Naive', 'Expert']
                for condition in comp_conds:
                    if condition not in bias_metrics[dur_key]:
                        continue
                    net_prop = bias_metrics[dur_key][condition]['mean_net_propagation'][:, smallest_bias_idx]
                    color = self.config.colors[condition]
                    ax.plot(self.stimulus_sizes, net_prop, 'o-', color=color,
                           markerfacecolor=color, linewidth=2, label=condition)

                ax.axhline(0, color='k', linestyle='--')
                ax.set_xlabel('Stimulus Size (px)')
                ax.set_ylabel('Net Propagation (px)')
                ax.set_title(f'D) Smallest Bias ({small_bias_values[0]:.2f})')
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)

                # Panel E: Activity time course at threshold size (Expert, different small biases)
                ax = axes[1, 1]
                if 'Expert' in bias_metrics[dur_key]:
                    # Find threshold size for Expert at smallest bias
                    thresh_size = bias_metrics[dur_key]['Expert']['mean_threshold'][smallest_bias_idx]
                    thresh_size_idx = np.argmin(np.abs(self.stimulus_sizes - thresh_size))
                    if thresh_size_idx >= n_sizes:
                        thresh_size_idx = n_sizes // 2

                    # Time vectors (MATLAB convention: time=0 at last pre-stim frame)
                    total_frames = self.pre_stim_frames + duration + self.post_stim_frames
                    time_vec_local = np.arange(total_frames) + 1 - self.pre_stim_frames
                    if self.config.show_real_time:
                        time_vec_local = time_vec_local / self.config.sampling_rate
                        stim_end_time = duration / self.config.sampling_rate
                    else:
                        stim_end_time = duration

                    # Plot time courses for different small biases
                    if dur_key in self.data and 'Expert' in self.data[dur_key] and 'bias' in self.data[dur_key]['Expert']:
                        act_data = self.data[dur_key]['Expert']['bias']['activity']
                        if act_data.ndim == 5:
                            small_bias_colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(small_bias_indices)))
                            for bi, b in enumerate(small_bias_indices):
                                act_for_bias = act_data[:, thresh_size_idx, b, :, :]
                                mean_act = np.mean(np.mean(act_for_bias, axis=0), axis=0)
                                ax.plot(time_vec_local, mean_act, '-', color=small_bias_colors[bi],
                                       linewidth=1.5, label=f'bias={small_bias_values[bi]:.2f}')

                            ax.axvline(0, color='k', linestyle='--')
                            ax.axvline(stim_end_time, color='k', linestyle='--')
                            ax.set_xlabel(time_label)
                            ax.set_ylabel('Activity')
                            ax.set_title(f'E) Expert Time Course (size={self.stimulus_sizes[thresh_size_idx]})')
                            ax.legend(loc='best', fontsize=7)
                            ax.grid(True, alpha=0.3)

                # Panel F: Summary text
                ax = axes[1, 2]
                ax.axis('off')
                summary_text = (
                    "Small Bias Analysis Summary\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"Bias values analyzed: {[f'{v:.2f}' for v in small_bias_values]}\n\n"
                    "Key Questions:\n"
                    "1. Does sensitivity (EC50) change at small biases?\n"
                    "2. Is threshold size affected by bias strength?\n"
                    "3. Does Expert maintain advantage at weak biases?\n\n"
                    "Interpretation:\n"
                    "Small biases test whether the system can\n"
                    "amplify weak inputs preferentially."
                )
                ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=9, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                fig7.suptitle(f'Small Bias Deep Dive (bias ≤ 1, {dur_str})')
                fig7.tight_layout()
                fig7.savefig(dur_bias_dir / 'SmallBias_DeepDive.png', dpi=150)
                fig7.savefig(dur_bias_dir / 'SmallBias_DeepDive.pdf')
                plt.close(fig7)
                print(f"  Saved: {dur_str}/SmallBias_DeepDive")

    def _compute_bias_metrics(self):
        """Compute metrics for bias mode analysis."""
        bias_metrics = {}

        n_bias_vals = len(self.stimulus_bias_values)
        n_sizes = len(self.stimulus_sizes)
        stim_radii = self.stimulus_sizes / 2

        for duration in self.stimulus_durations:
            dur_key = f'dur_{duration}'
            bias_metrics[dur_key] = {}

            if dur_key not in self.data:
                continue

            total_frames = self.pre_stim_frames + duration + self.post_stim_frames
            stim_on_idx = slice(self.pre_stim_frames, self.pre_stim_frames + duration)

            for condition in self.conditions:
                if condition not in self.data[dur_key]:
                    continue

                if 'bias' not in self.data[dur_key][condition]:
                    continue

                bias_metrics[dur_key][condition] = {
                    'EC50': np.zeros(n_bias_vals),
                    'hill_coefficient': np.zeros(n_bias_vals),
                    'mean_threshold': np.zeros(n_bias_vals),
                    'mean_amplification': np.zeros((n_sizes, n_bias_vals)),
                    'mean_net_propagation': np.zeros((n_sizes, n_bias_vals)),
                    'propagation_success': np.zeros((n_sizes, n_bias_vals)),
                    'cov_net_propagation': np.zeros((n_sizes, n_bias_vals)),
                }

                activity_data = self.data[dur_key][condition]['bias']['activity']
                extent_data = self.data[dur_key][condition]['bias']['stimulus_blob_extent']

                # Check if 5D
                if activity_data.ndim != 5:
                    continue

                n_sims = activity_data.shape[0]

                for b in range(n_bias_vals):
                    for s in range(n_sizes):
                        # Get data for this bias and size [nSims x nReps x nFrames]
                        act_for_bias = activity_data[:, s, b, :, :]
                        ext_for_bias = extent_data[:, s, b, :, :]

                        # Average across replicates
                        mean_act = np.mean(act_for_bias, axis=1)  # [nSims x nFrames]
                        mean_ext = np.mean(ext_for_bias, axis=1)

                        # Baseline and peak
                        baseline = np.mean(mean_act[:, :self.pre_stim_frames], axis=1)
                        peak_act = np.max(mean_act[:, stim_on_idx], axis=1)

                        # Amplification
                        amplification = peak_act / np.maximum(baseline, 0.01)
                        bias_metrics[dur_key][condition]['mean_amplification'][s, b] = np.mean(amplification)

                        # Net propagation
                        max_extent = np.max(mean_ext[:, stim_on_idx], axis=1)
                        net_prop = max_extent - stim_radii[s]
                        bias_metrics[dur_key][condition]['mean_net_propagation'][s, b] = np.mean(net_prop)

                        # Propagation success
                        bias_metrics[dur_key][condition]['propagation_success'][s, b] = np.mean(net_prop > 0)

                        # CoV
                        std_net = np.std(net_prop)
                        mean_net = bias_metrics[dur_key][condition]['mean_net_propagation'][s, b]
                        if mean_net > 0:
                            bias_metrics[dur_key][condition]['cov_net_propagation'][s, b] = std_net / mean_net
                        else:
                            bias_metrics[dur_key][condition]['cov_net_propagation'][s, b] = np.nan

                    # Fit Hill function for this bias value
                    mean_amp = bias_metrics[dur_key][condition]['mean_amplification'][:, b]
                    ec50, hill_n, _ = fit_hill_function(self.stimulus_sizes, mean_amp)
                    bias_metrics[dur_key][condition]['EC50'][b] = ec50
                    bias_metrics[dur_key][condition]['hill_coefficient'][b] = hill_n

                    # Threshold (size where net propagation first > 0)
                    net_prop_for_bias = bias_metrics[dur_key][condition]['mean_net_propagation'][:, b]
                    thresh_idx = np.where(net_prop_for_bias > 0)[0]
                    if len(thresh_idx) > 0:
                        bias_metrics[dur_key][condition]['mean_threshold'][b] = self.stimulus_sizes[thresh_idx[0]]
                    else:
                        bias_metrics[dur_key][condition]['mean_threshold'][b] = max(self.stimulus_sizes)

        return bias_metrics

    def run(self, results_path, output_dir='./PerturbationAnalysis'):
        """
        Run the complete analysis pipeline.

        Parameters:
        -----------
        results_path : str
            Path to perturbation results file
        output_dir : str
            Output directory for figures
        """
        # Load results
        self.load_results(results_path)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Analyze each duration
        for duration in self.stimulus_durations:
            print(f"\n{'='*60}")
            print(f"=== Analyzing Duration: {duration} frames ===")
            print(f"{'='*60}")

            dur_output = output_dir / f'dur_{duration}'
            dur_output.mkdir(parents=True, exist_ok=True)

            # Extract data
            data = self.extract_data(duration)

            # Compute metrics
            metrics = self.compute_metrics(data, duration)

            # Gating analysis
            gating = self.gating_analysis(metrics)

            # Statistical comparisons
            stats = self.statistical_comparisons(metrics, gating)

            # Generate figures
            self.plot_all(data, metrics, gating, stats, duration, dur_output)

            # Store for later
            self.data[f'dur_{duration}'] = data
            self.metrics[f'dur_{duration}'] = metrics
            self.gating[f'dur_{duration}'] = gating
            self.stats[f'dur_{duration}'] = stats

        # Generate cross-duration summary figures
        if len(self.stimulus_durations) > 1:
            self.plot_summary(output_dir)

        # Generate bias analysis figures if bias mode was used
        if any(mode_uses_bias(m) for m in self.stimulus_modes) and len(self.stimulus_bias_values) > 0:
            self.plot_bias_analysis(output_dir)

        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze Ising model perturbation experiments'
    )
    parser.add_argument(
        '--results', '-r',
        type=str,
        required=True,
        help='Path to perturbation results file (.mat or .npz)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./PerturbationAnalysis',
        help='Output directory for figures (default: ./PerturbationAnalysis)'
    )
    parser.add_argument(
        '--show-real-time',
        action='store_true',
        help='Display time in seconds instead of frames'
    )
    parser.add_argument(
        '--sampling-rate',
        type=float,
        default=10.0,
        help='Sampling rate in Hz for real-time display (default: 10)'
    )
    parser.add_argument(
        '--metric', '-m',
        type=str,
        default=None,
        help='Metric name for organizing output (creates metric-named subdirectory)'
    )
    parser.add_argument(
        '--stim-mode',
        type=str,
        default='clamped',
        help='Stimulus mode (or display mode like high_bias). Used to find the per-mode '
             'PerturbationResults file. Default: clamped.'
    )

    args = parser.parse_args()

    # Find results file (handle wildcards)
    # If --results is a directory or a bare wildcard, prefer per-mode lookup first.
    raw_mode = raw_mode_from_display_mode(args.stim_mode)
    results_files = glob(args.results)

    # If the user passed a wildcard that matches both legacy and per-mode files,
    # prefer the per-mode file matching --stim-mode. If no per-mode file exists,
    # restrict to legacy monolithic files (PerturbationResults_<8digits>_<6digits>.mat)
    # to avoid accidentally loading data for the WRONG mode.
    if results_files:
        import re
        per_mode_matches = [f for f in results_files
                            if re.search(rf'PerturbationResults_{re.escape(raw_mode)}_\d{{8}}_\d{{6}}\.mat$', f)]
        if per_mode_matches:
            results_files = per_mode_matches
            print(f"Selected per-mode files for '{raw_mode}'")
        else:
            legacy_matches = [f for f in results_files
                              if re.search(r'PerturbationResults_\d{8}_\d{6}\.mat$', f)]
            if legacy_matches:
                results_files = legacy_matches
                print(f"No per-mode file for '{raw_mode}' — falling back to legacy monolithic.")
            # else: leave results_files as-is (user may have given an exact path)

    if not results_files:
        print(f"Error: No results files found matching: {args.results} (mode='{raw_mode}')")
        return

    # Use most recent if multiple files match
    results_path = max(results_files, key=lambda f: Path(f).stat().st_mtime)
    print(f"Using results file: {results_path}")

    # Create configuration
    config = Config()
    config.show_real_time = args.show_real_time
    config.sampling_rate = args.sampling_rate

    # Build output directory (add metric subdirectory if specified)
    output_dir = Path(args.output)
    if args.metric:
        output_dir = output_dir / args.metric

    # Run analysis
    analysis = PerturbationAnalysis(config)
    analysis.run(results_path, output_dir)


if __name__ == '__main__':
    main()
