#!/usr/bin/env python3
"""
Ising Model Comparison Visualizations
======================================

Visualization functions for comparing Ising model simulations with experimental data.
Matches MATLAB Figure5_IsingComparison.m Section 8 visualizations.

Usage:
    from ising_visualizations import IsingVisualizer

    viz = IsingVisualizer(results, config)
    viz.plot_all()  # Generate all figures
    viz.plot_morans_i_distributions()  # Or individual plots
"""

import gc
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from matplotlib import cm
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from itertools import combinations
import warnings

# Required for computing distribution statistics
from scipy import stats as scipy_stats

# Wasserstein distance (for cross-condition analyses)
try:
    from wasserstein_optimized import wasserstein_1d as _wasserstein_1d
    HAS_WASSERSTEIN_OPT = True
except ImportError:
    HAS_WASSERSTEIN_OPT = False
    def _wasserstein_1d(x, y, max_quantiles=1000):
        from scipy.stats import wasserstein_distance
        x_clean = np.asarray(x).ravel(); x_clean = x_clean[~np.isnan(x_clean)]
        y_clean = np.asarray(y).ravel(); y_clean = y_clean[~np.isnan(y_clean)]
        if len(x_clean) == 0 or len(y_clean) == 0: return np.nan
        return wasserstein_distance(x_clean, y_clean)

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# MDE disabled — UMAP preferred; keeping code for potential future use
# try:
#     import pymde
#     HAS_MDE = True
# except ImportError:
#     HAS_MDE = False
HAS_MDE = False


# =============================================================================
# STATISTICAL HELPER FUNCTIONS
# =============================================================================

def _significance_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def _holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction to an array of p-values."""
    try:
        from statsmodels.stats.multitest import multipletests
        _, corrected, _, _ = multipletests(p_values, method='holm')
        return corrected
    except ImportError:
        pass
    # Manual fallback
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    corrected = np.empty(n)
    for rank, idx in enumerate(order):
        corrected[idx] = min(p_values[idx] * (n - rank), 1.0)
    # Enforce monotonicity
    cummax = 0.0
    for idx in order:
        cummax = max(cummax, corrected[idx])
        corrected[idx] = cummax
    return corrected


# =============================================================================
# HELPER FUNCTIONS FOR FEATURE COMPUTATION
# =============================================================================

def compute_6_statistics(data: np.ndarray) -> np.ndarray:
    """
    Compute 6 distribution statistics matching MATLAB implementation.

    Parameters
    ----------
    data : np.ndarray
        1D array of values

    Returns
    -------
    np.ndarray
        6-element array: [mean, std, median, skewness, kurtosis, iqr]
    """
    data = np.asarray(data).ravel()
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    if len(data) == 1:
        return np.array([data[0], 0.0, data[0], 0.0, 0.0, 0.0])

    if len(data) < 4:
        # Not enough data for skewness/kurtosis
        return np.array([
            np.mean(data),
            np.std(data),
            np.median(data),
            0.0,
            0.0,
            np.percentile(data, 75) - np.percentile(data, 25)
        ])

    return np.array([
        np.mean(data),
        np.std(data),
        np.median(data),
        scipy_stats.skew(data),
        scipy_stats.kurtosis(data),
        np.percentile(data, 75) - np.percentile(data, 25)  # IQR
    ])


def _normalize_animal_names(raw):
    """Return a flat list of animal-name strings regardless of how the
    field was serialized.

    Older results files saved this field as a single bytes blob containing
    the Python repr of a list (e.g. b"['Animal11', 'Animal12', ...]")
    because h5py 3.x rejected raw NumPy unicode arrays and the save fell
    through to a str(value) fallback. Newer files write a 1-D string
    dataset. Decode either form into a plain Python list of str so the
    per-animal grouping in the dot overlay works for both.
    """
    if raw is None:
        return None
    if isinstance(raw, (bytes, np.bytes_)):
        s = raw.decode('utf-8', errors='replace')
        if s.startswith('[') and s.endswith(']'):
            import ast
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed]
            except (SyntaxError, ValueError):
                pass
        return [s]
    if hasattr(raw, 'flatten'):
        return [x.decode('utf-8') if isinstance(x, (bytes, np.bytes_))
                else str(x) for x in raw.flatten()]
    if isinstance(raw, (list, tuple)):
        return [x.decode('utf-8') if isinstance(x, (bytes, np.bytes_))
                else str(x) for x in raw]
    return [str(raw)]


def split_trials_by_recording(trials_matrix: np.ndarray,
                               nTrials_per_recording: np.ndarray) -> list:
    """
    Split (nTrials_total, nTime) matrix into per-recording chunks.

    Parameters
    ----------
    trials_matrix : np.ndarray
        Shape (nTrials_total, nTime) — pooled trials across recordings
    nTrials_per_recording : np.ndarray
        1D array with number of trials per recording

    Returns
    -------
    list of np.ndarray
        Each element has shape (nTrials_rec, nTime) for one recording
    """
    chunks = []
    offset = 0
    for n in nTrials_per_recording:
        n = int(n)
        chunks.append(trials_matrix[offset:offset + n, :])
        offset += n
    return chunks


# =============================================================================
# COLOR SCHEMES (matching MATLAB exactly)
# =============================================================================

# MATLAB RGB values converted to hex
CONDITION_COLORS = {
    'Naive': (0.3373, 0.7059, 0.9137),       # Light blue [0.3373, 0.7059, 0.9137]
    'Beginner': (0.8431, 0.2549, 0.6078),    # Magenta/pink [0.8431, 0.2549, 0.6078]
    'Expert': (0.0, 0.6196, 0.4510),         # Teal/green [0, 0.6196, 0.4510]
    'Expert_Hit': (0.0, 0.45, 0.70),         # Dark blue
    'Expert_Miss': (0.90, 0.20, 0.20),       # Red
    'NoSpout': (0.8353, 0.3686, 0.0),        # Orange [0.8353, 0.3686, 0]
}

# Hex versions for string-based contexts
CONDITION_COLORS_HEX = {
    'Naive': '#56B4E9',       # Light blue
    'Beginner': '#D741A1',    # Magenta/pink
    'Expert': '#009E73',      # Teal/green
    'Expert_Hit': '#0073B3',  # Dark blue
    'Expert_Miss': '#E63333', # Red
    'NoSpout': '#D55E00',     # Orange
}

def _make_half_circle(side):
    """Build a matplotlib Path for a left or right semicircle marker.

    Parameters
    ----------
    side : str
        'left' or 'right'.
    """
    # Unit circle vertices: 20 points along the semicircle arc + centre + close
    n_pts = 20
    if side == 'left':
        # Arc from 90° to 270° (top -> left -> bottom)
        angles = np.linspace(np.pi / 2, 3 * np.pi / 2, n_pts)
    else:
        # Arc from 270° to 450° (= 90°) (bottom -> right -> top)
        angles = np.linspace(-np.pi / 2, np.pi / 2, n_pts)

    verts = [(np.cos(a), np.sin(a)) for a in angles]
    verts.append((0.0, verts[-1][1]))  # close to vertical midline
    verts.append(verts[0])             # close polygon

    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


MARKER_LEFT_HALF = _make_half_circle('left')
MARKER_RIGHT_HALF = _make_half_circle('right')


PARAM_COLORS = {
    'beta': '#9467bd',        # Purple
    'c': '#8c564b',           # Brown
    'decay_const': '#e377c2', # Pink
    'inhibition_range': '#7f7f7f',  # Gray
    'bias': '#bcbd22',        # Yellow-green
}

SUBMETRIC_LABELS = {
    'moransI': "Moran's I", 'activity': 'Activity', 'autocorr': 'Autocorr',
    'blobCount': 'Blob count', 'blobPersistence': 'Blob persist.',
}
SUBMETRIC_COLORS = {
    'moransI': '#1b9e77', 'activity': '#d95f02', 'autocorr': '#7570b3',
    'blobCount': '#e7298a', 'blobPersistence': '#66a61e',
}

# Figure size constants (inches) -- journal column widths
FIG_WIDTH_SINGLE = 3.5
FIG_WIDTH_1_5 = 5.5
FIG_WIDTH_FULL = 7.5
FIG_HEIGHT_UNIT = 2.5

# Publication-quality rcParams
PUBLICATION_RCPARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.transparent': False,
}

# Condition sets for dual-variant figure generation
BASE_CONDITIONS = ['Naive', 'Beginner', 'Expert', 'NoSpout']
HIT_MISS_CONDITIONS = ['Expert_Hit', 'Expert_Miss']

# Plot methods grouped by category for subfolder organization
# Order is preserved: list of (category_name, [method_names])
PLOT_CATEGORIES = [
    ('distributions', [
        'plot_morans_i_distributions',
        'plot_morans_i_distributions_kde',
        'plot_morans_i_distributions_kde_raw',
        'plot_activity_distributions',
        'plot_activity_distributions_kde',
        'plot_multi_observable_comparison',
        'plot_multi_observable_comparison_raw',
        'plot_simplified_distributions',
        'plot_qq_plots',
        'plot_blob_survival_curves',
    ]),
    ('parameters', [
        'plot_best_match_parameters',
        'plot_best_match_parameters_top100',
        'plot_best_match_parameters_violin',
        'plot_best_match_parameters_top100_violin',
        'plot_parameter_statistics',
        'plot_parameter_statistics_top100',
        'plot_parameter_heatmap',
        'plot_parameter_trends_summary',
        'plot_parameter_covariance',
        'plot_parameter_correlation',
        'plot_parameter_sensitivity',
        'plot_fixed_parameter_analysis',
        'plot_top10_parameters_table',
        'plot_condition_parameter_overlap',
    ]),
    ('temporal', [
        'plot_timeseries_comparison',
        'plot_spatial_snapshots',
        'plot_autocorrelation_comparison',
        'plot_activity_autocorrelation',
        'plot_time_constants',
        'plot_time_constants_dotline',
        'plot_conversion_factor_variability',
        'plot_phase_portrait',
    ]),
    ('summary', [
        'plot_exp_vs_ising_summary',
        'plot_exp_vs_ising_top_matches',
        'plot_exp_vs_ising_violin',
        'plot_exp_vs_ising_dotline',
        'plot_wasserstein_distances',
        'plot_radar_match_quality',
        'plot_summary_table',
    ]),
    ('mode_comparison', [
        'plot_tiled_mode_analysis',
        'plot_centre_vs_tiled_comparison',
        'plot_best_match_params_centre_vs_tiled',
        'plot_pooled_matching_comparison',
    ]),
    ('umap/feature_based', [
        'plot_umap_parameter_space',
        'plot_umap_morans_i_only',
        'plot_umap_morans_i_only_shaded',
        'plot_umap_morans_i_per_recording',
        'plot_umap_morans_i_per_animal',
        'plot_umap_morans_i_activity_features',
        'plot_umap_morans_i_activity_features_shaded',
        'plot_umap_activity_features',
        'plot_umap_spatial_persistence_features',
        'plot_umap_blob_persistence_features',
    ]),
    ('umap/wd_based', [
        'plot_umap_morans_i_only_wd',
        'plot_umap_morans_i_only_wd_shaded',
        'plot_umap_morans_i_activity_features_wd',
        'plot_umap_morans_i_activity_features_wd_shaded',
        'plot_umap_activity_features_wd',
        'plot_umap_spatial_persistence_features_wd',
    ]),
    # densMAP and PCA siblings for the MI+Activity WD figures only — added
    # to give the boss alternative projections that preserve more of the
    # global Wasserstein-distance structure than UMAP does.
    ('densmap/wd_based', [
        'plot_densmap_morans_i_activity_features_wd',
        'plot_densmap_morans_i_activity_features_wd_shaded',
    ]),
    ('pca/wd_based', [
        'plot_pca_morans_i_activity_features_wd',
        'plot_pca_morans_i_activity_features_wd_shaded',
    ]),
    # MDE disabled — UMAP preferred; keeping code for potential future use
    # ('mde/feature_based', [
    #     'plot_mde_morans_i_only',
    #     'plot_mde_morans_i_only_cloud',
    #     'plot_mde_morans_i_per_recording',
    #     'plot_mde_morans_i_per_animal',
    #     'plot_mde_morans_i_activity_features',
    #     'plot_mde_morans_i_activity_features_cloud',
    #     'plot_mde_activity_features',
    #     'plot_mde_spatial_persistence_features',
    #     'plot_mde_blob_persistence_features',
    # ]),
    # ('mde/wd_based', [
    #     'plot_mde_morans_i_only_wd',
    #     'plot_mde_morans_i_only_wd_cloud',
    #     'plot_mde_morans_i_activity_features_wd',
    #     'plot_mde_morans_i_activity_features_wd_cloud',
    #     'plot_mde_activity_features_wd',
    #     'plot_mde_spatial_persistence_features_wd',
    # ]),
    ('match_specificity', [
        'plot_match_specificity_bar',
        'plot_match_specificity_heatmap',
    ]),
    ('distributions_extended', [
        'plot_wd_rank_curve',
        'plot_wd_rank_curve_raw',
        'plot_wd_rank_curve_top100',
        'plot_wd_rank_curve_top100_raw',
        'plot_wd_rank_curve_overlay',
        'plot_wd_rank_curve_overlay_raw',
        'plot_wd_rank_curve_overlay_top100',
        'plot_wd_rank_curve_overlay_top100_raw',
        'plot_wd_rank_curve_overlay_top500',
        'plot_wd_rank_curve_overlay_top500_raw',
        'plot_wd_rank_curve_overlay_top1000',
        'plot_wd_rank_curve_overlay_top1000_raw',
        'plot_wd_rank_curve_overlay_top5000',
        'plot_wd_rank_curve_overlay_top5000_raw',
        'plot_intercondition_wd_matrix',
        'plot_intercondition_wd_matrix_raw',
        'plot_topn_wd_to_all_conditions',
        'plot_topn_wd_to_all_conditions_raw',
        'plot_metric_decomposition',
        'plot_metric_decomposition_raw',
        'plot_rank_concordance',
        'plot_rank_concordance_raw',
        'plot_leave_one_metric_out',
        'plot_leave_one_metric_out_raw',
        'plot_parameter_corner_plot',
        'plot_parameter_corner_plot_raw',
        'plot_rank_gap_evolution',
        'plot_rank_gap_evolution_raw',
        'plot_rank_gap_evolution_top100',
        'plot_rank_gap_evolution_top100_raw',
    ]),
]


class IsingVisualizer:
    """Visualization class for Ising comparison results."""

    def __init__(self, results: dict, config: dict, output_dir: str = None,
                 ising_data_path: str = None, exp_data_path: str = None,
                 frame_label: str = ''):
        """
        Initialize visualizer.

        Parameters
        ----------
        results : dict
            Results from Figure5_IsingComparison_optimized.py containing:
            - IsingData, ExpStats, Comparison, DynamicsAnalysis, ParameterTrends
        config : dict
            Configuration dict
        output_dir : str, optional
            Directory to save figures. If None, uses config['output_path']/figures
        ising_data_path : str, optional
            Path to Ising simulation data directory (for loading raw frames)
        exp_data_path : str, optional
            Path to experimental data file (for loading raw frames)
        frame_label : str, optional
            Prefix for variant subfolders (e.g. 'prestim', 'full_trial').
            When set, subfolders become e.g. 'prestim_base' instead of 'base'.
        """
        self.results = results
        self.config = config
        self._frame_label = frame_label
        self.conditions = config.get('conditions', ['Naive', 'Beginner', 'Expert', 'NoSpout'])
        self.ising_data_path = ising_data_path
        self.exp_data_path = exp_data_path

        if output_dir is None:
            output_dir = os.path.join(config.get('output_path', '.'), 'figures')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Subfolder routing state (used by _save_figure / plot_all)
        self._current_subfolder = ''
        self._all_conditions = list(self.conditions)

        # Embedding cache for reuse across methods (cleared in plot_all)
        self._embedding_cache = {}

        # Set publication style
        self._set_publication_style()

    def _set_publication_style(self):
        """Apply publication-quality matplotlib settings."""
        plt.rcParams.update(PUBLICATION_RCPARAMS)

    def _get_color(self, condition: str):
        """Get color for condition (returns RGB tuple or hex string)."""
        return CONDITION_COLORS.get(condition, (0.2, 0.2, 0.2))

    def _get_best_idx(self, condition: str, raw=False) -> np.ndarray:
        """Get full ranking of Ising indices sorted by WD (best first).

        Uses the complete 'rankings' array when available (all simulations
        sorted by ascending WD).  Falls back to the truncated 'bestMatch_idx'
        for older result files that lack 'rankings'.

        raw=True: recompute rankings from raw (un-normalized) combined WD.
        """
        comparison = self.results.get('Comparison', {})
        if condition not in comparison:
            return np.array([], dtype=int)
        cond_data = comparison[condition]
        return self._get_rankings_for(cond_data, raw=raw)

    def _get_temporal_scale(self, condition: str) -> np.ndarray:
        """Get per-simulation temporal scale factors for a condition.

        Returns array of scale factors (tau_exp / tau_ising) for each sim.
        Falls back to 1.0 (no scaling) if not available.
        """
        comparison = self.results.get('Comparison', {})
        cond_data = comparison.get(condition, {})
        factors = cond_data.get('temporal_scale_factors', None)
        if factors is not None:
            return np.asarray(factors)
        return np.ones(len(self.results.get('IsingData', {}).get('simIDs', [])))

    def _get_active_metric_keys(self):
        """Return the sub-metric keys that were used for matching."""
        _METRIC_KEYS = {
            'moransI':             ['moransI'],
            'activity':            ['activity'],
            'autocorr':            ['autocorr'],
            'blobCount':           ['blobCount'],
            'blobPersistence':     ['blobPersistence'],
            'moransI+activity':    ['moransI', 'activity'],
            'moransI+activity_weighted': ['moransI', 'activity'],
            'spatial+persistence': ['moransI', 'activity', 'blobPersistence'],
            'spatial+persistence_weighted': ['moransI', 'activity', 'blobPersistence'],
            'combined':            ['moransI', 'activity', 'blobCount', 'blobPersistence'],
        }
        metric = self.config.get('matching_metric', 'combined')
        return _METRIC_KEYS.get(metric, list(SUBMETRIC_LABELS.keys()))

    def _compute_raw_combined_wd(self, cond_data):
        """Compute raw (un-normalized) combined WD from individual_dists."""
        indiv = cond_data.get('individual_dists', {})
        active_keys = self._get_active_metric_keys()
        arrays = []
        for mk in active_keys:
            arr = indiv.get(mk)
            if arr is not None:
                arrays.append(np.atleast_1d(np.asarray(arr, dtype=float)))
        if len(arrays) == 0:
            return None
        return np.mean(arrays, axis=0)

    def _get_rankings_for(self, cond_data, raw=False):
        """Return ranking indices sorted by ascending WD.
        raw=False: stored rankings (normalized WD order).
        raw=True:  recomputed from raw combined WD.
        """
        if raw:
            raw_wd = self._compute_raw_combined_wd(cond_data)
            if raw_wd is not None and len(raw_wd) > 0:
                return np.argsort(raw_wd)
        # Fall back to stored rankings
        r = cond_data.get('rankings', None)
        if r is None or len(np.atleast_1d(r)) == 0:
            r = cond_data.get('bestMatch_idx', np.array([]))
        return np.atleast_1d(np.asarray(r, dtype=int))

    def _save_figure(self, fig, name: str, formats: List[str] = ['png', 'pdf']):
        """Save figure to output directory (routed through current subfolder)."""
        if self._current_subfolder:
            save_dir = os.path.join(self.output_dir, self._current_subfolder)
        else:
            save_dir = self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        for fmt in formats:
            filepath = os.path.join(save_dir, f"{name}.{fmt}")
            fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
        display_name = os.path.join(self._current_subfolder, name) if self._current_subfolder else name
        print(f"  Saved: {display_name}")

    def _has_hit_miss_data(self) -> bool:
        """Check if Expert_Hit/Expert_Miss have data in results."""
        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        return any(c in exp_stats or c in comparison for c in HIT_MISS_CONDITIONS)

    def _run_category(self, category_name: str, methods: list,
                      conditions: list, subfolder: str):
        """Run a group of plot methods with specific conditions and subfolder."""
        self._current_subfolder = subfolder
        self.conditions = conditions

        for method_name in methods:
            # Skip methods when optional dependencies are missing
            if method_name.startswith('plot_umap_') and not HAS_UMAP:
                continue
            # if method_name.startswith('plot_mde_') and not HAS_MDE:
            #     continue

            method = getattr(self, method_name, None)
            if method is None:
                continue

            try:
                method()  # all UMAP/MDE methods have default top_n_list
            except Exception as e:
                print(f"  Warning: {method_name} failed: {e}")
            gc.collect()

    # =========================================================================
    # DISTRIBUTION COMPARISONS
    # =========================================================================

    def plot_morans_i_distributions(self):
        """Plot Moran's I distribution comparison (Exp vs Best Ising)."""
        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        n_conditions = len(self.conditions)
        fig, axes = plt.subplots(1, n_conditions, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))
        if n_conditions == 1:
            axes = [axes]

        for i, condition in enumerate(self.conditions):
            ax = axes[i]

            if condition not in exp_stats or condition not in comparison:
                ax.set_visible(False)
                continue

            # Experimental data
            exp_mi = exp_stats[condition].get('MoransI_all', np.array([]))
            if len(exp_mi) == 0:
                ax.set_visible(False)
                continue

            # Best match Ising data
            best_idx = self._get_best_idx(condition)
            if len(best_idx) > 0:
                ising_mi_all = ising_data.get('MoransI_all', [])
                if len(ising_mi_all) > best_idx[0]:
                    ising_mi = np.array(ising_mi_all[best_idx[0]])
                else:
                    ising_mi = np.array([])
            else:
                ising_mi = np.array([])

            # Plot histograms
            bins = np.linspace(-0.5, 1.0, 50)
            exp_mi_clean = exp_mi[~np.isnan(exp_mi)]
            ax.hist(exp_mi_clean, bins=bins, alpha=0.6, label='Experimental',
                    color=self._get_color(condition), density=True)

            # Add mean vertical line for experimental data
            exp_mean = np.mean(exp_mi_clean)
            ax.axvline(exp_mean, color=self._get_color(condition), linestyle='--',
                       linewidth=2, label=f'Exp mean: {exp_mean:.3f}')

            if len(ising_mi) > 0:
                ising_mi_clean = ising_mi[~np.isnan(ising_mi)]
                ax.hist(ising_mi_clean, bins=bins, alpha=0.6, label='Best Ising',
                        color='gray', density=True)
                # Add mean vertical line for Ising data
                ising_mean = np.mean(ising_mi_clean)
                ax.axvline(ising_mean, color='gray', linestyle='--',
                           linewidth=2, label=f'Ising mean: {ising_mean:.3f}')

            ax.set_xlabel("Moran's I")
            ax.set_ylabel('Density')
            ax.set_title(condition)
            ax.legend(fontsize=8, loc='upper right')
            ax.set_xlim(-0.5, 1.0)

        fig.suptitle("Moran's I Distribution: Experimental vs Best Ising Match", fontsize=12)
        fig.tight_layout()
        self._save_figure(fig, 'morans_i_distributions')

    def plot_activity_distributions(self):
        """Plot activity distribution comparison."""
        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        n_conditions = len(self.conditions)
        fig, axes = plt.subplots(1, n_conditions, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))
        if n_conditions == 1:
            axes = [axes]

        for i, condition in enumerate(self.conditions):
            ax = axes[i]

            if condition not in exp_stats or condition not in comparison:
                ax.set_visible(False)
                continue

            exp_act = exp_stats[condition].get('Activity_all', np.array([]))
            if len(exp_act) == 0:
                ax.set_visible(False)
                continue

            best_idx = self._get_best_idx(condition)
            ising_act_all = ising_data.get('Activity_all', [])
            ising_act = np.array(ising_act_all[best_idx[0]]) if len(best_idx) > 0 and len(ising_act_all) > best_idx[0] else np.array([])

            bins = np.linspace(0, 1, 50)
            ax.hist(exp_act[~np.isnan(exp_act)], bins=bins, alpha=0.6, label='Experimental',
                    color=self._get_color(condition), density=True)
            if len(ising_act) > 0:
                ax.hist(ising_act[~np.isnan(ising_act)], bins=bins, alpha=0.6, label='Best Ising',
                        color='gray', density=True)

            ax.set_xlabel('Activity')
            ax.set_ylabel('Density')
            ax.set_title(condition)
            ax.set_xlim(0, 0.4)
            ax.legend(fontsize=8)

        fig.suptitle('Activity Distribution: Experimental vs Best Ising Match', fontsize=12)
        fig.tight_layout()
        self._save_figure(fig, 'activity_distributions')

    # =========================================================================
    # PARAMETER ANALYSIS
    # =========================================================================

    def _get_parameter_values_by_condition(self, param, top_n=None):
        """Extract parameter values per condition.

        Parameters
        ----------
        param : str
            Parameter name (e.g. 'beta', 'c', 'decay_const').
        top_n : int or None
            If set, use parameters from the top-N ranked simulations.

        Returns
        -------
        dict : {condition_name: np.ndarray of values}
            Only conditions with non-empty data are included.
        """
        comparison = self.results.get('Comparison', {})
        ising_params = self.results.get('IsingData', {}).get('params', {})
        result = {}
        for condition in self.conditions:
            if condition not in comparison:
                continue
            if top_n is not None and param in ising_params:
                rankings = self._get_rankings_for(comparison[condition], raw=False)
                all_vals = np.asarray(ising_params[param])
                indices = rankings[:top_n]
                valid = indices[indices < len(all_vals)]
                values = all_vals[valid] if len(valid) > 0 else np.array([])
            else:
                values = comparison[condition].get('bestMatch_params', {}).get(param, np.array([]))
            values = np.asarray(values).ravel()
            if len(values) > 0:
                result[condition] = values
        return result

    def plot_best_match_parameters(self, top_n=None, violin=False):
        """Plot best match parameters as boxplots (or violin plots) per condition.

        Parameters
        ----------
        top_n : int or None
            If set, use parameters from the top-N ranked simulations
            (looked up via IsingData params + rankings) instead of the
            pre-stored bestMatch_params.  None uses bestMatch_params as-is.
        violin : bool
            If True, use violin plots instead of boxplots.
        """
        ising_params = self.results.get('IsingData', {}).get('params', {})

        param_names = ['beta', 'c', 'decay_const', 'inhibition_range', 'bias']
        n_params = len(param_names)

        fig, axes = plt.subplots(1, n_params, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))

        for i, param in enumerate(param_names):
            ax = axes[i]
            cond_data = self._get_parameter_values_by_condition(param, top_n=top_n)
            data = list(cond_data.values())
            labels = list(cond_data.keys())

            if len(data) > 0:
                if violin:
                    vp = ax.violinplot(data, showmedians=True, showextrema=True)
                    for j, body in enumerate(vp['bodies']):
                        body.set_facecolor(self._get_color(labels[j]))
                        body.set_alpha(0.6)
                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                        if partname in vp:
                            vp[partname].set_color('black')
                            vp[partname].set_linewidth(0.8)
                    ax.set_xticks(range(1, len(labels) + 1))
                    ax.set_xticklabels(labels)
                else:
                    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
                    for j, patch in enumerate(bp['boxes']):
                        patch.set_facecolor(self._get_color(labels[j]))
                        patch.set_alpha(0.6)

            # Set y-ticks to parameter scanning grid values
            if param in ising_params:
                grid_values = np.sort(np.unique(np.asarray(ising_params[param])))
                if len(grid_values) > 0:
                    ax.set_yticks(grid_values)

            ax.set_ylabel(param)
            ax.set_title(param)
            ax.tick_params(axis='x', rotation=45)

        top_label = f' (Top {top_n})' if top_n is not None else ''
        plot_type = 'Violin' if violin else 'Boxplot'
        fig.suptitle(f'Best Match Parameters by Condition{top_label}', fontsize=12)
        fig.tight_layout()
        suffix = ''
        if top_n is not None:
            suffix += f'_top{top_n}'
        if violin:
            suffix += '_violin'
        self._save_figure(fig, f'best_match_parameters{suffix}')

    def plot_best_match_parameters_top100(self):
        """Best match parameters boxplot — top 100."""
        return self.plot_best_match_parameters(top_n=100)

    def plot_best_match_parameters_violin(self):
        """Best match parameters as violin plots."""
        return self.plot_best_match_parameters(violin=True)

    def plot_best_match_parameters_top100_violin(self):
        """Best match parameters as violin plots — top 100."""
        return self.plot_best_match_parameters(top_n=100, violin=True)

    def plot_parameter_statistics(self, top_n=None):
        """Pairwise condition comparison statistics for best-match parameters.

        For each of the 5 Ising parameters, produces a table showing:
        - Kruskal-Wallis H-test (omnibus: are any conditions different?)
        - Pairwise Mann-Whitney U tests for all condition pairs
        - Holm-Bonferroni corrected p-values
        - Significance stars

        Parameters
        ----------
        top_n : int or None
            If set, use parameters from the top-N ranked simulations.
        """
        param_names = ['beta', 'c', 'decay_const', 'inhibition_range', 'bias']
        n_params = len(param_names)

        fig, axes = plt.subplots(n_params, 1,
                                 figsize=(FIG_WIDTH_FULL, 2.0 * n_params))

        for i, param in enumerate(param_names):
            ax = axes[i]
            ax.axis('off')

            cond_data = self._get_parameter_values_by_condition(param, top_n=top_n)
            conditions_with_data = list(cond_data.keys())

            if len(conditions_with_data) < 2:
                ax.text(0.5, 0.5, f'{param}: insufficient data',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            # Kruskal-Wallis omnibus test
            groups = [cond_data[c] for c in conditions_with_data]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    kw_stat, kw_p = scipy_stats.kruskal(*groups)
                except ValueError:
                    kw_stat, kw_p = 0.0, 1.0

            # Pairwise Mann-Whitney U tests
            pairs = list(combinations(conditions_with_data, 2))
            pair_labels = []
            u_stats = []
            p_values = []

            for c1, c2 in pairs:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        u, p = scipy_stats.mannwhitneyu(
                            cond_data[c1], cond_data[c2],
                            alternative='two-sided')
                    except ValueError:
                        u, p = np.nan, 1.0
                pair_labels.append(f'{c1} vs {c2}')
                u_stats.append(u)
                p_values.append(p)

            # Multiple comparison correction
            p_arr = np.array(p_values)
            corrected = _holm_bonferroni(p_arr)

            # Build table data
            col_labels = ['Pair', 'U stat', 'p-value', 'p (corrected)', 'Sig.']
            table_data = []
            for j, (label, u, p, pc) in enumerate(
                    zip(pair_labels, u_stats, p_values, corrected)):
                stars = _significance_stars(pc)
                table_data.append([
                    label,
                    f'{u:.0f}' if not np.isnan(u) else '--',
                    f'{p:.2e}',
                    f'{pc:.2e}',
                    stars
                ])

            table = ax.table(cellText=table_data, colLabels=col_labels,
                             loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.4)

            # Highlight significant cells
            for row_idx in range(len(table_data)):
                sig_cell = table[row_idx + 1, 4]  # +1 for header row
                if table_data[row_idx][4] != 'ns':
                    sig_cell.set_text_props(fontweight='bold', color='red')

            # Title: parameter name, Kruskal-Wallis result, sample sizes
            n_str = ', '.join(
                f'{c}: n={len(cond_data[c])}' for c in conditions_with_data)
            kw_stars = _significance_stars(kw_p)
            ax.set_title(
                f'{param}  \u2014  Kruskal-Wallis: H={kw_stat:.2f}, '
                f'p={kw_p:.2e} {kw_stars}\n({n_str})',
                fontsize=8, fontweight='bold')

        top_label = f' (Top {top_n})' if top_n is not None else ''
        fig.suptitle(f'Pairwise Parameter Statistics{top_label}', fontsize=12)
        fig.tight_layout()
        suffix = f'_top{top_n}' if top_n is not None else ''
        self._save_figure(fig, f'parameter_statistics{suffix}')

    def plot_parameter_statistics_top100(self):
        """Pairwise parameter statistics — top 100."""
        return self.plot_parameter_statistics(top_n=100)

    def plot_parameter_heatmap(self):
        """
        Plot parameter space heatmap (beta vs bias) with mean Wasserstein distance.

        Matches MATLAB: 2D imagesc heatmap with flipud(hot) colormap showing
        mean Wasserstein distance at each (beta, bias) grid cell.
        """
        ising_data = self.results.get('IsingData', {})
        comparison = self.results.get('Comparison', {})

        params = ising_data.get('params', {})
        if not params:
            print("  No parameter data for heatmap")
            return

        betas = np.array(params.get('beta', []))
        biases = np.array(params.get('bias', []))

        if len(betas) == 0:
            return

        # Get unique beta and bias values for grid
        unique_betas = np.sort(np.unique(betas))
        unique_biases = np.sort(np.unique(biases))

        n_beta = len(unique_betas)
        n_bias = len(unique_biases)

        # Create figure with subplots for each condition
        n_conditions = len(self.conditions)
        fig, axes = plt.subplots(1, n_conditions, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))
        if n_conditions == 1:
            axes = [axes]

        # Create reversed hot colormap (like MATLAB flipud(hot))
        from matplotlib.colors import LinearSegmentedColormap
        hot_cmap = plt.cm.hot
        hot_r_cmap = LinearSegmentedColormap.from_list('hot_r', hot_cmap(np.linspace(1, 0, 256)))

        for i, condition in enumerate(self.conditions):
            ax = axes[i]

            if condition not in comparison:
                ax.set_visible(False)
                continue

            # Get Wasserstein distances for this condition
            wd = comparison[condition].get('wasserstein_dist', np.array([]))
            if len(wd) == 0:
                ax.set_visible(False)
                continue

            # Create grid of mean Wasserstein distances
            wd_grid = np.full((n_bias, n_beta), np.nan)
            for j in range(len(betas)):
                beta_idx = np.where(unique_betas == betas[j])[0]
                bias_idx = np.where(unique_biases == biases[j])[0]
                if len(beta_idx) > 0 and len(bias_idx) > 0:
                    if np.isnan(wd_grid[bias_idx[0], beta_idx[0]]):
                        wd_grid[bias_idx[0], beta_idx[0]] = wd[j]
                    else:
                        # Average multiple simulations at same (beta, bias)
                        wd_grid[bias_idx[0], beta_idx[0]] = (wd_grid[bias_idx[0], beta_idx[0]] + wd[j]) / 2

            # Plot as 2D heatmap (imagesc style)
            im = ax.imshow(wd_grid, cmap=hot_r_cmap, aspect='auto',
                          extent=[unique_betas.min(), unique_betas.max(),
                                  unique_biases.min(), unique_biases.max()],
                          origin='lower')
            plt.colorbar(im, ax=ax, label='Mean Wasserstein Distance')

            # Mark best matches
            best_idx = self._get_best_idx(condition)
            if len(best_idx) > 0:
                ax.scatter(betas[best_idx], biases[best_idx],
                          c='cyan', s=100, edgecolors='black', linewidth=1.5,
                          marker='*', zorder=10, label=f'Top {len(best_idx)}')
                ax.legend(fontsize=8, loc='upper right')

            ax.set_xlabel('Beta')
            ax.set_ylabel('Bias')
            ax.set_title(f'{condition}\nParameter Space (WD)')

        fig.suptitle('Parameter Space Heatmap: Mean Wasserstein Distance', fontsize=12)
        fig.tight_layout()
        self._save_figure(fig, 'parameter_heatmap')

    def plot_parameter_trends_summary(self):
        """Plot parameter trends heatmap summary."""
        trends = self.results.get('ParameterTrends', {})
        if not trends:
            print("  No ParameterTrends data")
            return

        param_names = ['beta', 'c', 'decay_const', 'inhibition_range', 'bias']
        conditions_with_data = [c for c in self.conditions if c in trends]

        if len(conditions_with_data) == 0:
            return

        # Build matrix of mean values
        matrix = np.zeros((len(conditions_with_data), len(param_names)))
        for i, cond in enumerate(conditions_with_data):
            for j, param in enumerate(param_names):
                if param in trends[cond]:
                    matrix[i, j] = trends[cond][param]['mean']

        # Normalize each column for visualization
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            matrix_norm = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0) + 1e-10)

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))
        im = ax.imshow(matrix_norm, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(len(param_names)))
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.set_yticks(range(len(conditions_with_data)))
        ax.set_yticklabels(conditions_with_data)

        # Add text annotations
        for i in range(len(conditions_with_data)):
            for j in range(len(param_names)):
                text = f'{matrix[i, j]:.2f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=8)

        plt.colorbar(im, ax=ax, label='Normalized value')
        ax.set_title('Best Match Parameter Means by Condition')
        fig.tight_layout()
        self._save_figure(fig, 'parameter_trends_summary')

    # =========================================================================
    # TEMPORAL DYNAMICS
    # =========================================================================

    def plot_timeseries_comparison(self, max_frames=None):
        """
        Plot Activity and Moran's I time series comparing Data vs Simulation.

        Creates 5 figures per condition, each showing a single trial vs its
        best-matched simulation segment:
        - Top: Activity over time (red=Data, black=Simulation)
        - Bottom: Moran's I over time (red=Data, black=Simulation)

        Each single trial is slid across the simulation to find
        the segment with highest correlation.

        Parameters
        ----------
        max_frames : int, optional
            If set, truncate trial time series to this many frames before
            matching and plotting (e.g. 80 for pre-stimulus only).

        Trial data is loaded from the experimental data file (BinarisedData and MoransI).
        """
        import h5py

        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        NUM_TRIALS_TO_SHOW = 5  # Number of individual trials to display

        # Load trial data from experimental file
        if self.exp_data_path is None:
            print("    Skipping timeseries comparison: no experimental data path provided")
            return

        try:
            exp_file = h5py.File(self.exp_data_path, 'r')
        except Exception as e:
            print(f"    Skipping timeseries comparison: cannot open experimental data file: {e}")
            return

        try:
            for condition in self.conditions:
                if condition not in comparison:
                    continue

                # Get best match index
                best_idx_list = self._get_best_idx(condition)
                if len(best_idx_list) == 0:
                    continue
                best_idx = int(best_idx_list[0])

                # Load trial data from experimental file
                # BinarisedData: shape (trials, frames, width, height) = (N, 185, 26, 13)
                # MoransI: shape (frames, trials) = (185, N)
                try:
                    binarised_data = exp_file['BinarisedData'][condition][:]
                    morans_i_data = exp_file['MoransI'][condition][:]
                except KeyError:
                    print(f"    Skipping {condition}: data not found in experimental file")
                    continue

                # binarised_data shape: (trials, frames, width, height)
                n_trials = binarised_data.shape[0]
                n_frames = binarised_data.shape[1]

                # Compute activity per trial: mean across spatial dimensions
                # activity_trials: shape (trials, frames)
                activity_trials = np.mean(binarised_data, axis=(2, 3))

                # morans_i_data shape: (frames, trials) -> transpose to (trials, frames)
                if morans_i_data.shape[0] == n_frames:
                    mi_trials = morans_i_data.T  # (frames, trials) -> (trials, frames)
                else:
                    mi_trials = morans_i_data    # already (trials, frames)

                print(f"  {condition}: {n_trials} trials, {n_frames} frames each")

                # Get simulation data (time series)
                ising_activity = None
                ising_mi = None
                activity_all = ising_data.get('Activity_all', [])
                mi_all = ising_data.get('MoransI_all', [])

                if len(activity_all) > best_idx:
                    ising_activity = np.array(activity_all[best_idx]).flatten()
                if len(mi_all) > best_idx:
                    ising_mi = np.array(mi_all[best_idx]).flatten()

                if ising_activity is None:
                    print(f"    Skipping {condition}: no simulation activity data for best_idx={best_idx}")
                    continue

                print(f"    Simulation length: {len(ising_activity)} frames")

                # Select NUM_TRIALS_TO_SHOW evenly spaced trials
                trial_indices = np.linspace(0, n_trials - 1, min(NUM_TRIALS_TO_SHOW, n_trials), dtype=int)
                print(f"    Using {len(trial_indices)} trials: {trial_indices}")

                # Process each selected trial
                for trial_num, trial_idx in enumerate(trial_indices):
                    # Extract single trial time series
                    exp_activity_ts = activity_trials[trial_idx, :]
                    exp_mi_ts = mi_trials[trial_idx, :] if trial_idx < mi_trials.shape[0] else None

                    # Truncate to max_frames if set (e.g. prestim = 80 frames)
                    if max_frames is not None:
                        exp_activity_ts = exp_activity_ts[:max_frames]
                        if exp_mi_ts is not None:
                            exp_mi_ts = exp_mi_ts[:max_frames]

                    # Find best matching segment in simulation using sliding window correlation
                    trial_len = len(exp_activity_ts)
                    sim_len = len(ising_activity)
                    sim_start = 0
                    match_info = ""

                    if trial_len > 0 and sim_len >= trial_len:
                        best_corr = -np.inf
                        best_sim_start = 0

                        # Normalize experimental trial
                        exp_std = np.nanstd(exp_activity_ts)
                        if exp_std > 1e-8:
                            exp_norm = (exp_activity_ts - np.nanmean(exp_activity_ts)) / exp_std

                            # Slide trial across simulation (step for efficiency with long simulations)
                            sim_step = max(1, (sim_len - trial_len) // 500)

                            for s_start in range(0, sim_len - trial_len + 1, sim_step):
                                sim_window = ising_activity[s_start:s_start + trial_len]

                                # Normalize simulation window
                                sim_std = np.nanstd(sim_window)
                                if sim_std < 1e-8:
                                    continue
                                sim_norm = (sim_window - np.nanmean(sim_window)) / sim_std

                                # Compute correlation
                                corr = np.nanmean(exp_norm * sim_norm)

                                if corr > best_corr:
                                    best_corr = corr
                                    best_sim_start = s_start

                            sim_start = best_sim_start
                            match_info = f" (r={best_corr:.2f})"

                    # Compute segment length (use trial length)
                    segment_len = trial_len
                    if ising_activity is not None:
                        segment_len = min(segment_len, len(ising_activity) - sim_start)

                    if segment_len <= 0:
                        print(f"      Skipping {condition} trial {trial_idx}: no valid segment")
                        continue

                    # Use common x-axis for both traces
                    x_common = np.arange(segment_len)

                    # Create figure with 2 rows
                    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

                    # Panel 1: Activity
                    ax = axes[0]
                    ax.plot(x_common, exp_activity_ts[:segment_len], color='red', linewidth=1, label='Data', alpha=0.8)
                    if ising_activity is not None:
                        ax.plot(x_common, ising_activity[sim_start:sim_start + segment_len], color='black', linewidth=1, label='Simulation', alpha=0.8)
                    ax.set_ylabel('Activity')
                    ax.legend(loc='upper right', fontsize=8)
                    ax.set_title(f'{condition}: Time Series Segment (Trial {trial_idx + 1}){match_info}')

                    # Panel 2: Moran's I
                    ax = axes[1]
                    if exp_mi_ts is not None:
                        mi_segment_len = min(segment_len, len(exp_mi_ts))
                        ax.plot(x_common[:mi_segment_len], np.nan_to_num(exp_mi_ts[:mi_segment_len], nan=0.0), color='red', linewidth=1, label='Data', alpha=0.8)
                    if ising_mi is not None:
                        mi_segment_len = min(segment_len, len(ising_mi) - sim_start)
                        ax.plot(x_common[:mi_segment_len], np.nan_to_num(ising_mi[sim_start:sim_start + mi_segment_len], nan=0.0), color='black', linewidth=1, label='Simulation', alpha=0.8)
                    ax.set_ylabel("Moran's I")
                    ax.set_xlabel('Time [samples]')
                    ax.legend(loc='upper right', fontsize=8)

                    fig.tight_layout()
                    self._save_figure(fig, f'timeseries_comparison_{condition.lower()}_{trial_num + 1}')

        finally:
            exp_file.close()

    def plot_spatial_snapshots(self):
        """
        Plot spatial activity snapshots comparing experimental vs simulation data.

        Generates 3 separate figures per condition, each showing:
        - Left: Experimental frame (red dots on white)
        - Right: Best-matching simulation frame by Moran's I (black squares on white)

        Frames are selected at low/medium/high Moran's I quartiles from experimental data,
        and matched to simulation frames with closest Moran's I values.
        Both frames are displayed at 13x26 grid size with Moran's I values annotated.
        """
        import h5py
        from scipy import io as sio
        from glob import glob

        # Import Moran's I functions for frame matching
        try:
            from morans_i_optimized import morans_i_single_jit, create_weight_matrix_jit
        except ImportError:
            print("  Skipping spatial snapshots: morans_i_optimized module not available")
            return

        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        # Check if we have data paths
        if self.ising_data_path is None or self.exp_data_path is None:
            print("  Skipping spatial snapshots: data paths not provided")
            return

        # Target grid size
        TARGET_ROWS = 13
        TARGET_COLS = 26

        # Pre-compute weight matrix for Moran's I
        weight_mat = create_weight_matrix_jit(TARGET_ROWS, TARGET_COLS, False)

        # Process each condition
        for condition in self.conditions:
            print(f"    Processing spatial snapshots for {condition}...")

            # Load experimental data for this condition
            try:
                with h5py.File(self.exp_data_path, 'r') as f:
                    if 'BinarisedData' not in f or condition not in f['BinarisedData']:
                        print(f"      Skipping {condition}: no BinarisedData found")
                        continue

                    # Shape after transpose: [rows, cols, time, trials]
                    bin_data = np.array(f['BinarisedData'][condition]).T

                    # Get dimensions
                    exp_rows, exp_cols, n_time, n_trials = bin_data.shape

                    # Flatten across trials to get all frames
                    # Shape: [rows, cols, time * trials]
                    all_exp_frames_raw = bin_data.reshape(exp_rows, exp_cols, -1)
                    n_exp_frames = all_exp_frames_raw.shape[2]

                    # Centre crop if needed to get 13x26
                    if exp_rows >= TARGET_ROWS and exp_cols >= TARGET_COLS:
                        row_start = (exp_rows - TARGET_ROWS) // 2
                        col_start = (exp_cols - TARGET_COLS) // 2
                        all_exp_frames = all_exp_frames_raw[
                            row_start:row_start + TARGET_ROWS,
                            col_start:col_start + TARGET_COLS,
                            :
                        ]
                    else:
                        # Pad if smaller (unlikely but handle gracefully)
                        all_exp_frames = np.zeros((TARGET_ROWS, TARGET_COLS, n_exp_frames))
                        r_off = (TARGET_ROWS - exp_rows) // 2
                        c_off = (TARGET_COLS - exp_cols) // 2
                        all_exp_frames[r_off:r_off + exp_rows, c_off:c_off + exp_cols, :] = all_exp_frames_raw

            except Exception as e:
                print(f"      Skipping {condition}: error loading experimental data - {e}")
                continue

            # Compute Moran's I for all experimental frames
            exp_morans = np.zeros(n_exp_frames)
            for t in range(n_exp_frames):
                exp_morans[t] = morans_i_single_jit(all_exp_frames[:, :, t], weight_mat)

            # Filter out NaN values
            valid_mask = ~np.isnan(exp_morans)
            valid_indices = np.where(valid_mask)[0]
            valid_morans = exp_morans[valid_mask]

            if len(valid_morans) < 5:
                print(f"      Skipping {condition}: insufficient valid Moran's I values ({len(valid_morans)})")
                continue

            # Select 5 frames at evenly spaced percentiles of Moran's I
            percentiles = [10, 30, 50, 70, 90]
            target_values = [np.percentile(valid_morans, p) for p in percentiles]

            # Find frames closest to each percentile
            selected_exp_indices = []
            selected_exp_morans = []
            for target_val in target_values:
                dist = np.abs(valid_morans - target_val)
                best_local_idx = np.argmin(dist)
                global_idx = valid_indices[best_local_idx]
                selected_exp_indices.append(global_idx)
                selected_exp_morans.append(valid_morans[best_local_idx])

            # Load simulation data - find best match simulation file
            try:
                if condition not in comparison:
                    print(f"      Skipping {condition}: no comparison data")
                    continue

                best_idx = self._get_best_idx(condition)
                if len(best_idx) == 0:
                    print(f"      Skipping {condition}: no best match found")
                    continue

                # Find simulation file by parameters
                sim_files = glob(os.path.join(self.ising_data_path, 'sim_be_*.mat'))
                sim_file = None
                params = ising_data.get('params', {})

                if params and len(params.get('beta', [])) > best_idx[0]:
                    beta = params['beta'][best_idx[0]]
                    c = params['c'][best_idx[0]]
                    decay = params['decay_const'][best_idx[0]]
                    rad = params['inhibition_range'][best_idx[0]]

                    for f in sim_files:
                        basename = os.path.basename(f)
                        if (f'be_{beta}' in basename and f'_c_{int(c)}_' in basename and
                            f'_d_{int(decay)}_' in basename and f'_r_{int(rad)}_' in basename):
                            sim_file = f
                            break

                if sim_file is None:
                    print(f"      Skipping {condition}: simulation file not found")
                    continue

                # Load simulation frames
                sim_data = sio.loadmat(sim_file, squeeze_me=True)
                stored_spins = sim_data['stored_spins']  # [T, rows, cols]
                # CRITICAL: Convert to float64 for Moran's I computation (loaded as int8)
                stored_spins = np.ascontiguousarray(stored_spins.astype(np.float64))
                n_sim_frames, sim_rows, sim_cols = stored_spins.shape

                # Centre crop simulation to 13x26
                if sim_rows >= TARGET_ROWS and sim_cols >= TARGET_COLS:
                    row_start = (sim_rows - TARGET_ROWS) // 2
                    col_start = (sim_cols - TARGET_COLS) // 2
                    all_sim_frames = stored_spins[
                        :,
                        row_start:row_start + TARGET_ROWS,
                        col_start:col_start + TARGET_COLS
                    ]
                else:
                    # Pad if smaller
                    all_sim_frames = np.zeros((n_sim_frames, TARGET_ROWS, TARGET_COLS))
                    r_off = (TARGET_ROWS - sim_rows) // 2
                    c_off = (TARGET_COLS - sim_cols) // 2
                    all_sim_frames[:, r_off:r_off + sim_rows, c_off:c_off + sim_cols] = stored_spins

                # Convert -1/+1 to 0/1 and ensure contiguous float64 for JIT
                all_sim_frames = (all_sim_frames + 1) / 2
                all_sim_frames = np.ascontiguousarray(all_sim_frames, dtype=np.float64)

            except Exception as e:
                print(f"      Skipping {condition}: error loading simulation data - {e}")
                continue

            # Compute Moran's I for all simulation frames
            sim_morans = np.zeros(n_sim_frames)
            for t in range(n_sim_frames):
                sim_morans[t] = morans_i_single_jit(all_sim_frames[t], weight_mat)

            # Filter out NaN frames (zero-variance frames where all cells are identical)
            valid_sim_mask = ~np.isnan(sim_morans)
            n_valid_sim = np.sum(valid_sim_mask)
            print(f"      Valid simulation frames: {n_valid_sim}/{n_sim_frames} ({100*n_valid_sim/n_sim_frames:.1f}%)")
            if n_valid_sim < 3:
                print(f"      Skipping {condition}: insufficient valid simulation frames")
                continue

            # Create 2x5 montage: top=experimental, bottom=simulation
            from matplotlib.colors import ListedColormap
            n_snaps = len(selected_exp_indices)
            fig, axes = plt.subplots(2, n_snaps, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT * 1.6))
            if n_snaps == 1:
                axes = axes.reshape(2, 1)

            cmap_exp = ListedColormap(['white', 'red'])
            cmap_sim = ListedColormap(['white', 'black'])

            for col, (exp_frame_idx, exp_mi) in enumerate(zip(selected_exp_indices, selected_exp_morans)):
                sim_mi_dist = np.abs(sim_morans - exp_mi)
                sim_mi_dist[~valid_sim_mask] = np.inf
                best_sim_idx = np.argmin(sim_mi_dist)
                sim_mi = sim_morans[best_sim_idx]

                exp_frame = all_exp_frames[:, :, exp_frame_idx]
                sim_frame = all_sim_frames[best_sim_idx]

                # Top: experimental
                ax = axes[0, col]
                ax.imshow(exp_frame, cmap=cmap_exp, interpolation='nearest',
                          origin='upper', aspect='equal')
                # Thin grid lines for cell boundaries
                for r in range(exp_frame.shape[0]):
                    ax.axhline(r - 0.5, color='lightgray', linewidth=0.2)
                for c_line in range(exp_frame.shape[1]):
                    ax.axvline(c_line - 0.5, color='lightgray', linewidth=0.2)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'MI={exp_mi:.3f}', fontsize=6)
                if col == 0:
                    ax.set_ylabel('Data', fontsize=7)

                # Bottom: simulation
                ax = axes[1, col]
                ax.imshow(sim_frame, cmap=cmap_sim, interpolation='nearest',
                          origin='upper', aspect='equal')
                for r in range(sim_frame.shape[0]):
                    ax.axhline(r - 0.5, color='lightgray', linewidth=0.2)
                for c_line in range(sim_frame.shape[1]):
                    ax.axvline(c_line - 0.5, color='lightgray', linewidth=0.2)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'MI={sim_mi:.3f}', fontsize=6)
                if col == 0:
                    ax.set_ylabel('Sim', fontsize=7)

            fig.suptitle(f'{condition}: Spatial Snapshots (P10-P90)')
            fig.tight_layout()
            self._save_figure(fig, f'spatial_snapshots_{condition.lower()}')

    def plot_autocorrelation_comparison(self):
        """
        Plot autocorrelation comparison between experimental and best Ising.

        Single-row layout with residual strip below each panel.
        Shaded fit range, clean text annotations, publication line widths.
        """
        dynamics = self.results.get('DynamicsAnalysis', {})

        fit_range = self.config.get('autocorr', {}).get('fit_range', (1, 10))

        n_conditions = len(self.conditions)
        fig, axes = plt.subplots(2, n_conditions,
                                 figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT * 1.4),
                                 gridspec_kw={'height_ratios': [3, 1]},
                                 sharex='col')
        if n_conditions == 1:
            axes = axes.reshape(2, 1)

        for i, condition in enumerate(self.conditions):
            ax_main = axes[0, i]
            ax_res = axes[1, i]

            if condition not in dynamics:
                ax_main.set_visible(False)
                ax_res.set_visible(False)
                continue

            da = dynamics[condition]
            exp_acf = da.get('exp_acf', np.array([]))
            ising_acf = da.get('best_ising_acf', np.array([]))
            exp_tau = da.get('exp_tau', np.nan)
            ising_tau = da.get('best_ising_tau', np.nan)
            exp_r2 = da.get('exp_tau_r2', np.nan)
            ising_r2 = da.get('best_ising_r2', np.nan)

            if len(exp_acf) == 0:
                ax_main.set_visible(False)
                ax_res.set_visible(False)
                continue

            max_lag = min(100, len(exp_acf))
            lags = np.arange(max_lag)

            # Shaded fit range with stronger emphasis
            ax_main.axvspan(fit_range[0], fit_range[1], alpha=0.25, color='lightblue',
                            zorder=0)
            # Labeled bracket at top
            ax_main.annotate('', xy=(fit_range[0], 1.02), xytext=(fit_range[1], 1.02),
                             xycoords=('data', 'axes fraction'),
                             textcoords=('data', 'axes fraction'),
                             arrowprops=dict(arrowstyle='|-|', color='steelblue',
                                             lw=0.75, mutation_scale=3))
            ax_main.text((fit_range[0] + fit_range[1]) / 2, 1.05, 'fit',
                         transform=ax_main.get_xaxis_transform(),
                         ha='center', va='bottom', fontsize=5, color='steelblue')

            # Experimental ACF
            ax_main.plot(lags, exp_acf[:max_lag], color=self._get_color(condition),
                         linewidth=1.0, label='Exp')

            # Experimental fit
            if not np.isnan(exp_tau) and exp_tau > 0:
                exp_fit = np.exp(-lags / exp_tau)
                ax_main.plot(lags, exp_fit, color=self._get_color(condition),
                             linewidth=0.75, linestyle='--', alpha=0.8,
                             label=f'Exp fit (t={exp_tau:.1f})')

            # Ising ACF
            max_lag_ising = max_lag
            if len(ising_acf) > 0:
                max_lag_ising = min(max_lag, len(ising_acf))
                lags_ising = np.arange(max_lag_ising)
                ax_main.plot(lags_ising, ising_acf[:max_lag_ising], color='gray',
                             linewidth=1.0, linestyle='--', label='Ising')

                if not np.isnan(ising_tau) and ising_tau > 0:
                    ising_fit = np.exp(-lags_ising / ising_tau)
                    ax_main.plot(lags_ising, ising_fit, color='dimgray',
                                 linewidth=0.75, linestyle=':', alpha=0.8,
                                 label=f'Ising fit (t={ising_tau:.1f})')

            ax_main.axhline(0, color='black', linewidth=0.3, linestyle=':')
            ax_main.set_title(condition)
            if i == 0:
                ax_main.set_ylabel('ACF')
            ax_main.legend(fontsize=5, loc='upper right')

            # Adaptive y-limits
            acf_vals = exp_acf[:max_lag]
            y_min = min(0, np.nanmin(acf_vals)) - 0.05
            ax_main.set_ylim(y_min, 1.05)
            ax_main.set_xlim(0, max_lag)

            # Clean text annotations (no wheat boxes)
            ann_parts = []
            if not np.isnan(exp_r2):
                ann_parts.append(f'Exp R2={exp_r2:.3f}')
            if not np.isnan(ising_r2):
                ann_parts.append(f'Ising R2={ising_r2:.3f}')
            scale_factor = da.get('tau_scale_factor', np.nan)
            if not np.isnan(scale_factor):
                ann_parts.append(f'1 MC ~ {scale_factor:.2f} frames')
            if ann_parts:
                ax_main.text(0.98, 0.55, '\n'.join(ann_parts),
                             transform=ax_main.transAxes, fontsize=5,
                             ha='right', va='top', color='dimgray')

            # Residual strip (exp_acf - ising_acf)
            if len(ising_acf) > 0:
                common_len = min(max_lag, len(ising_acf))
                residual = exp_acf[:common_len] - ising_acf[:common_len]
                ax_res.bar(np.arange(common_len), residual, width=1.0,
                           color=self._get_color(condition), alpha=0.5, linewidth=0)
                ax_res.axhline(0, color='black', linewidth=0.3)
            ax_res.set_xlabel('Lag')
            if i == 0:
                ax_res.set_ylabel('Residual')
            ax_res.set_xlim(0, max_lag)

        fig.suptitle('Autocorrelation: Exp vs Best Ising')
        fig.tight_layout()
        self._save_figure(fig, 'autocorrelation_comparison')

    def plot_time_constants(self):
        """Plot time constant comparison across conditions."""
        dynamics = self.results.get('DynamicsAnalysis', {})

        conditions_with_data = []
        exp_taus = []
        ising_taus = []
        scale_factors = []

        for condition in self.conditions:
            if condition in dynamics:
                da = dynamics[condition]
                if not np.isnan(da.get('exp_tau', np.nan)):
                    conditions_with_data.append(condition)
                    exp_taus.append(da.get('exp_tau', np.nan))
                    ising_taus.append(da.get('best_ising_tau', np.nan))
                    scale_factors.append(da.get('tau_scale_factor', np.nan))

        if len(conditions_with_data) == 0:
            print("  No time constant data")
            return

        fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))

        # Tau comparison
        ax = axes[0]
        x = np.arange(len(conditions_with_data))
        width = 0.35
        ax.bar(x - width/2, exp_taus, width, label='Experimental',
               color=[self._get_color(c) for c in conditions_with_data])
        ax.bar(x + width/2, ising_taus, width, label='Best Ising', color='gray', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions_with_data, rotation=45)
        ax.set_ylabel('Tau (frames/MC)')
        ax.set_title('Time Constants')
        ax.legend()

        # Tau ratio
        ax = axes[1]
        tau_ratios = [ising_taus[i]/exp_taus[i] if exp_taus[i] > 0 else np.nan
                      for i in range(len(exp_taus))]
        ax.bar(x, tau_ratios, color=[self._get_color(c) for c in conditions_with_data])
        ax.axhline(1, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions_with_data, rotation=45)
        ax.set_ylabel('Tau Ratio (Ising/Exp)')
        ax.set_title('Tau Ratio')

        # Scale factors
        ax = axes[2]
        ax.bar(x, scale_factors, color=[self._get_color(c) for c in conditions_with_data])
        ax.axhline(1, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions_with_data, rotation=45)
        ax.set_ylabel('Scale Factor (Exp/Ising)')
        ax.set_title('Temporal Scale Factor')

        fig.suptitle('Time Constant Analysis', fontsize=12)
        fig.tight_layout()
        self._save_figure(fig, 'time_constants')

    # =========================================================================
    # SUMMARY PLOTS
    # =========================================================================

    def plot_exp_vs_ising_summary(self):
        """
        Plot summary comparison of experimental vs best Ising match.

        Produces standard (unmasked) figure plus a masked variant if
        Activity_mean_masked is available in ExpStats.
        """
        self._plot_exp_vs_ising_summary_impl(use_masked=False)
        # Produce masked variant if masked activity data exists
        exp_stats = self.results.get('ExpStats', {})
        if any('Activity_mean_masked' in exp_stats.get(c, {}) for c in self.conditions):
            self._plot_exp_vs_ising_summary_impl(use_masked=True)

    def _plot_exp_vs_ising_summary_impl(self, use_masked=False):
        """Core implementation for plot_exp_vs_ising_summary."""
        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        metrics = ["Moran's I mean", "Activity mean", "BlobPersistence"]
        n_metrics = len(metrics)

        exp_color = (0.2, 0.4, 0.8)
        ising_color = (0.8, 0.2, 0.2)

        fig, axes = plt.subplots(1, n_metrics, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))

        for i, metric in enumerate(metrics):
            ax = axes[i]
            exp_values = []
            ising_values = []
            exp_stds = []
            ising_stds = []
            conditions_plotted = []

            for condition in self.conditions:
                if condition not in exp_stats or condition not in comparison:
                    continue

                best_idx = self._get_best_idx(condition)
                if len(best_idx) == 0:
                    continue

                if metric == "Moran's I mean":
                    exp_val = exp_stats[condition].get('MoransI_mean', np.nan)
                    exp_std = exp_stats[condition].get('MoransI_std', 0)
                    ising_means = ising_data.get('MoransI_mean', [])
                    ising_val = ising_means[best_idx[0]] if len(ising_means) > best_idx[0] else np.nan
                    ising_stds_arr = ising_data.get('MoransI_std', [])
                    ising_std = ising_stds_arr[best_idx[0]] if len(ising_stds_arr) > best_idx[0] else 0
                elif metric == "Activity mean":
                    if use_masked:
                        exp_val = exp_stats[condition].get('Activity_mean_masked', np.nan)
                        exp_std = exp_stats[condition].get('Activity_std_masked', 0)
                    else:
                        exp_val = exp_stats[condition].get('Activity_mean', np.nan)
                        exp_std = exp_stats[condition].get('Activity_std', 0)
                    ising_means = ising_data.get('Activity_mean', [])
                    ising_val = ising_means[best_idx[0]] if len(ising_means) > best_idx[0] else np.nan
                    ising_stds_arr = ising_data.get('Activity_std', [])
                    ising_std = ising_stds_arr[best_idx[0]] if len(ising_stds_arr) > best_idx[0] else 0
                else:  # BlobPersistence
                    if 'BlobPersistence_mean' in exp_stats[condition]:
                        exp_val = exp_stats[condition]['BlobPersistence_mean']
                        exp_std = 0
                    else:
                        exp_lifetimes = exp_stats[condition].get('BlobPersistence_lifetimes', None)
                        min_lt = self.config.get('blob_min_lifetime',
                                                  self.results.get('config', {}).get('blob_min_lifetime', 1))
                        if exp_lifetimes is not None and len(exp_lifetimes) > 0:
                            lt_arr = np.asarray(exp_lifetimes).ravel()
                            lt_filtered = lt_arr[lt_arr >= min_lt]
                            exp_val = np.nanmean(lt_filtered) if len(lt_filtered) > 0 else np.nanmean(lt_arr)
                            exp_std = np.nanstd(lt_filtered) if len(lt_filtered) > 0 else np.nanstd(lt_arr)
                        else:
                            exp_val = np.nan
                            exp_std = 0

                    ising_bp_means = ising_data.get('BlobPersistence_mean', np.array([]))
                    if hasattr(ising_bp_means, '__len__') and len(ising_bp_means) > best_idx[0]:
                        scale_factors = self._get_temporal_scale(condition)
                        ising_val = float(ising_bp_means[best_idx[0]]) * scale_factors[best_idx[0]]
                    else:
                        ising_val = np.nan
                    ising_std = 0

                if not np.isnan(exp_val):
                    exp_values.append(exp_val)
                    ising_values.append(ising_val if not np.isnan(ising_val) else 0)
                    exp_stds.append(exp_std if not np.isnan(exp_std) else 0)
                    ising_stds.append(ising_std if not np.isnan(ising_std) else 0)
                    conditions_plotted.append(condition)

            if len(exp_values) == 0:
                ax.set_visible(False)
                continue

            x = np.arange(len(conditions_plotted))
            width = 0.35
            ax.bar(x - width/2, exp_values, width, yerr=exp_stds, capsize=3,
                   label='Experimental', color=exp_color)
            ax.bar(x + width/2, ising_values, width, yerr=ising_stds, capsize=3,
                   label='Best Ising', color=ising_color)
            ax.set_xticks(x)
            ax.set_xticklabels(conditions_plotted, rotation=45)
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.legend(fontsize=8)

        suffix = ' (masked)' if use_masked else ''
        fig.suptitle(f'Experimental vs Best Ising Match Summary{suffix}', fontsize=12)
        fig.tight_layout()
        save_name = 'exp_vs_ising_summary_masked' if use_masked else 'exp_vs_ising_summary'
        self._save_figure(fig, save_name)

    def plot_exp_vs_ising_top_matches(self):
        """
        Plot summary bars with scatter dots for top-10 Ising matches.

        Produces standard and masked variants, plus per-recording and
        per-animal experimental dot overlays. Each variant is rendered
        twice: once with the rank-1 Ising sim called out as a gold star,
        and once (suffix `_no_star`) with rank-1 drawn as a plain black
        dot like the other 9. The combined 3-panel figure stays in the
        parent subfolder; per-metric singletons (one panel each) are
        also rendered into per-metric subfolders.
        """
        exp_stats = self.results.get('ExpStats', {})
        has_masked = any('Activity_mean_masked' in exp_stats.get(c, {}) for c in self.conditions)

        # None = combined 3-panel figure (current layout); each single
        # entry produces a one-panel figure into a per-metric subfolder.
        metric_groups = [None,
                         ["Moran's I mean"],
                         ["Activity mean"],
                         ["BlobPersistence"]]

        for metrics_to_plot in metric_groups:
            for highlight in (True, False):
                # Original (no experimental dots)
                self._plot_exp_vs_ising_top_matches_impl(
                    use_masked=False, highlight_rank1=highlight,
                    metrics_to_plot=metrics_to_plot)
                if has_masked:
                    self._plot_exp_vs_ising_top_matches_impl(
                        use_masked=True, highlight_rank1=highlight,
                        metrics_to_plot=metrics_to_plot)

                # Per-recording dots
                self._plot_exp_vs_ising_top_matches_impl(
                    use_masked=False, exp_dots='recording',
                    highlight_rank1=highlight,
                    metrics_to_plot=metrics_to_plot)
                if has_masked:
                    self._plot_exp_vs_ising_top_matches_impl(
                        use_masked=True, exp_dots='recording',
                        highlight_rank1=highlight,
                        metrics_to_plot=metrics_to_plot)

                # Per-animal dots
                self._plot_exp_vs_ising_top_matches_impl(
                    use_masked=False, exp_dots='animal',
                    highlight_rank1=highlight,
                    metrics_to_plot=metrics_to_plot)
                if has_masked:
                    self._plot_exp_vs_ising_top_matches_impl(
                        use_masked=True, exp_dots='animal',
                        highlight_rank1=highlight,
                        metrics_to_plot=metrics_to_plot)

    def _plot_exp_vs_ising_top_matches_impl(self, use_masked=False, exp_dots=None,
                                            highlight_rank1=True,
                                            metrics_to_plot=None):
        """Core implementation for plot_exp_vs_ising_top_matches.

        Parameters
        ----------
        use_masked : bool
            Use masked activity values.
        exp_dots : None, 'recording', or 'animal'
            Overlay per-recording or per-animal experimental means on the
            Experimental bars.
        highlight_rank1 : bool
            If True, draw the rank-1 Ising sim as a gold star with a
            dedicated legend entry. If False, render it as an additional
            black dot indistinguishable from ranks 2..10. Filename gets
            an `_no_star` suffix when False.
        metrics_to_plot : list[str] or None
            Which metrics to render. None (default) yields the combined
            3-panel figure. A single-element list yields a one-panel
            singleton, saved into a per-metric subfolder of the current
            output directory (morans_i_mean / activity_mean /
            blob_persistence).
        """
        from collections import defaultdict

        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})
        rec_meta = self.results.get('RecordingMetadata', {})

        ALL_METRICS = ["Moran's I mean", "Activity mean", "BlobPersistence"]
        METRIC_SUBFOLDER = {
            "Moran's I mean":  'morans_i_mean',
            "Activity mean":   'activity_mean',
            "BlobPersistence": 'blob_persistence',
        }
        metrics = list(metrics_to_plot) if metrics_to_plot else ALL_METRICS
        n_metrics = len(metrics)

        exp_color = (0.2, 0.4, 0.8)
        ising_color = (0.8, 0.2, 0.2)

        # Singleton: square-ish panel (one third of the combined width).
        if n_metrics == 1:
            figsize = (FIG_WIDTH_FULL / 3, FIG_HEIGHT_UNIT)
        else:
            figsize = (FIG_WIDTH_FULL, FIG_HEIGHT_UNIT)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        axes = np.atleast_1d(axes)

        for i, metric in enumerate(metrics):
            ax = axes[i]
            exp_values = []
            ising_values = []
            exp_stds = []
            ising_stds = []
            top10_dots = []
            exp_scatter = []
            conditions_plotted = []

            for condition in self.conditions:
                if condition not in exp_stats or condition not in comparison:
                    continue

                best_idx = self._get_best_idx(condition)
                if len(best_idx) == 0:
                    continue

                n_top = min(10, len(best_idx))

                # --- compute experimental dot means for this condition/metric ---
                edots = []
                if exp_dots is not None:
                    meta = rec_meta.get(condition, {})
                    nTrials_per_rec = meta.get('nTrials_per_recording', None)
                    animal_names = _normalize_animal_names(
                        meta.get('animal_names', None))

                    if nTrials_per_rec is not None and len(nTrials_per_rec) > 0:
                        if metric == "Moran's I mean":
                            trials_mat = exp_stats[condition].get('MoransI_trials', None)
                        elif metric == "BlobPersistence":
                            trials_mat = None  # handled below
                        else:  # Activity mean
                            key = 'Activity_trials_masked' if use_masked else 'Activity_trials'
                            trials_mat = exp_stats[condition].get(key, None)

                        rec_means = None
                        if metric == "BlobPersistence":
                            bp_per_rec = exp_stats[condition].get('BlobPersistence_per_recording', None)
                            if bp_per_rec is not None:
                                bp_per_rec = np.asarray(bp_per_rec).ravel()
                                rec_means = list(bp_per_rec[~np.isnan(bp_per_rec)])
                        elif trials_mat is not None:
                            trials_mat = np.asarray(trials_mat)
                            if trials_mat.ndim == 2:
                                # Activity_trials is [time, trials]; transpose to [trials, time]
                                if metric != "Moran's I mean":
                                    trials_mat = trials_mat.T
                                chunks = split_trials_by_recording(trials_mat, nTrials_per_rec)
                                rec_means = [np.nanmean(ch) for ch in chunks]

                        if rec_means is not None:
                            if (exp_dots == 'animal' and animal_names is not None
                                    and hasattr(animal_names, '__len__')
                                    and len(animal_names) == len(rec_means)):
                                animal_recs = defaultdict(list)
                                for rm, name in zip(rec_means, animal_names):
                                    animal_recs[name].append(rm)
                                edots = [np.nanmean(v) for v in animal_recs.values()]
                            else:
                                edots = rec_means

                if metric == "Moran's I mean":
                    exp_val = exp_stats[condition].get('MoransI_mean', np.nan)
                    exp_std = exp_stats[condition].get('MoransI_std', 0)
                    ising_means = ising_data.get('MoransI_mean', [])
                    ising_val = ising_means[best_idx[0]] if len(ising_means) > best_idx[0] else np.nan
                    ising_stds_arr = ising_data.get('MoransI_std', [])
                    ising_std = ising_stds_arr[best_idx[0]] if len(ising_stds_arr) > best_idx[0] else 0
                    dots = [ising_means[best_idx[j]] for j in range(n_top) if len(ising_means) > best_idx[j]]
                elif metric == "Activity mean":
                    if use_masked:
                        exp_val = exp_stats[condition].get('Activity_mean_masked', np.nan)
                        exp_std = exp_stats[condition].get('Activity_std_masked', 0)
                    else:
                        exp_val = exp_stats[condition].get('Activity_mean', np.nan)
                        exp_std = exp_stats[condition].get('Activity_std', 0)
                    ising_means = ising_data.get('Activity_mean', [])
                    ising_val = ising_means[best_idx[0]] if len(ising_means) > best_idx[0] else np.nan
                    ising_stds_arr = ising_data.get('Activity_std', [])
                    ising_std = ising_stds_arr[best_idx[0]] if len(ising_stds_arr) > best_idx[0] else 0
                    dots = [ising_means[best_idx[j]] for j in range(n_top) if len(ising_means) > best_idx[j]]
                else:  # BlobPersistence
                    if 'BlobPersistence_mean' in exp_stats[condition]:
                        exp_val = exp_stats[condition]['BlobPersistence_mean']
                        exp_std = 0
                    else:
                        exp_lifetimes = exp_stats[condition].get('BlobPersistence_lifetimes', None)
                        min_lt = self.config.get('blob_min_lifetime',
                                                  self.results.get('config', {}).get('blob_min_lifetime', 1))
                        if exp_lifetimes is not None and len(exp_lifetimes) > 0:
                            lt_arr = np.asarray(exp_lifetimes).ravel()
                            lt_filtered = lt_arr[lt_arr >= min_lt]
                            exp_val = np.nanmean(lt_filtered) if len(lt_filtered) > 0 else np.nanmean(lt_arr)
                            exp_std = np.nanstd(lt_filtered) if len(lt_filtered) > 0 else np.nanstd(lt_arr)
                        else:
                            exp_val = np.nan
                            exp_std = 0

                    ising_bp_means = ising_data.get('BlobPersistence_mean', np.array([]))
                    scale_factors = self._get_temporal_scale(condition)
                    if hasattr(ising_bp_means, '__len__') and len(ising_bp_means) > best_idx[0]:
                        ising_val = float(ising_bp_means[best_idx[0]]) * scale_factors[best_idx[0]]
                    else:
                        ising_val = np.nan
                    ising_std = 0

                    dots = []
                    if hasattr(ising_bp_means, '__len__'):
                        for j in range(n_top):
                            if len(ising_bp_means) > best_idx[j]:
                                v = float(ising_bp_means[best_idx[j]]) * scale_factors[best_idx[j]]
                                if not np.isnan(v):
                                    dots.append(v)

                # Make bars consistent with the dots they overlay:
                #   red bar = mean of top-10 Ising dots (was rank-1 only)
                #   blue bar = mean of per-animal/per-recording exp dots when those
                #              dots are being drawn (was pooled grand mean)
                if len(dots) > 0:
                    dots_mean = float(np.nanmean(dots))
                    if not np.isnan(dots_mean):
                        ising_val = dots_mean
                        ising_std = float(np.nanstd(dots))
                if exp_dots is not None and len(edots) > 0:
                    edots_mean = float(np.nanmean(edots))
                    if not np.isnan(edots_mean):
                        exp_val = edots_mean
                        exp_std = float(np.nanstd(edots))

                if not np.isnan(exp_val):
                    exp_values.append(exp_val)
                    ising_values.append(ising_val if not np.isnan(ising_val) else 0)
                    exp_stds.append(exp_std if not np.isnan(exp_std) else 0)
                    ising_stds.append(ising_std if not np.isnan(ising_std) else 0)
                    top10_dots.append(dots)
                    exp_scatter.append(edots)
                    conditions_plotted.append(condition)

            if len(exp_values) == 0:
                ax.set_visible(False)
                continue

            x = np.arange(len(conditions_plotted))
            width = 0.35

            ax.bar(x - width/2, exp_values, width,
                   label='Experimental', color=exp_color, alpha=0.8)
            ax.bar(x + width/2, ising_values, width,
                   label='Best Ising', color=ising_color, alpha=0.8)

            for ci, dots in enumerate(top10_dots):
                if len(dots) == 0:
                    continue
                jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(dots))
                if highlight_rank1:
                    # Ranks 2..N in plain black
                    if len(dots) > 1:
                        ax.scatter(x[ci] + width/2 + jitter[1:], dots[1:], color='black',
                                   s=20, zorder=5, alpha=0.7, edgecolors='white', linewidth=0.5)
                    # Rank-1 (the single best Ising match) called out with a star
                    ax.scatter(x[ci] + width/2 + jitter[0], dots[0],
                               color='#FFD700', marker='*', s=80, zorder=6,
                               edgecolors='black', linewidth=0.8)
                else:
                    # All 10 dots rendered identically (no rank-1 callout)
                    ax.scatter(x[ci] + width/2 + jitter, dots, color='black',
                               s=20, zorder=5, alpha=0.7, edgecolors='white', linewidth=0.5)

            if exp_dots is not None:
                for ci, dots in enumerate(exp_scatter):
                    if len(dots) > 0:
                        jitter = np.random.default_rng(43).uniform(-0.08, 0.08, size=len(dots))
                        ax.scatter(x[ci] - width/2 + jitter, dots, color='black',
                                   s=20, zorder=5, alpha=0.7, edgecolors='white', linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(conditions_plotted, rotation=45)
            ax.set_ylabel(metric)
            ax.set_title(metric)
            if i == 0:
                if highlight_rank1:
                    handles, labels = ax.get_legend_handles_labels()
                    rank1_handle = Line2D([0], [0], marker='*', color='w',
                                          markerfacecolor='#FFD700',
                                          markeredgecolor='black', markersize=10,
                                          label='Rank-1 Ising')
                    ax.legend(handles + [rank1_handle],
                              labels + ['Rank-1 Ising'], fontsize=8)
                else:
                    ax.legend(fontsize=8)

        dots_suffix = {None: '', 'recording': ' (per-recording)', 'animal': ' (per-animal)'}
        suffix = ' (masked)' if use_masked else ''
        star_suffix = '' if highlight_rank1 else ' (no star)'
        fig.suptitle(
            f'Exp vs Best Ising Match (top-10 dots){dots_suffix[exp_dots]}{suffix}{star_suffix}',
            fontsize=12)
        fig.tight_layout()
        dots_file = {'recording': '_per_recording', 'animal': '_per_animal'}.get(exp_dots, '')
        star_file = '' if highlight_rank1 else '_no_star'
        save_name = (f'exp_vs_ising_top_matches{dots_file}'
                     f'{"_masked" if use_masked else ""}{star_file}')

        if n_metrics == 1:
            # Route singleton into a per-metric subfolder. Push, save, pop.
            saved_subfolder = self._current_subfolder
            self._current_subfolder = os.path.join(
                saved_subfolder, METRIC_SUBFOLDER[metrics[0]])
            try:
                self._save_figure(fig, save_name)
            finally:
                self._current_subfolder = saved_subfolder
        else:
            self._save_figure(fig, save_name)

    def plot_exp_vs_ising_violin(self):
        """
        Side-by-side violin plots of experimental vs best Ising match distributions.

        Produces standard and masked variants.
        """
        self._plot_exp_vs_ising_violin_impl(use_masked=False)
        exp_stats = self.results.get('ExpStats', {})
        if any('Activity_all_masked' in exp_stats.get(c, {}) for c in self.conditions):
            self._plot_exp_vs_ising_violin_impl(use_masked=True)

    def _plot_exp_vs_ising_violin_impl(self, use_masked=False):
        """Core implementation for plot_exp_vs_ising_violin."""
        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        metrics = ["Moran's I mean", "Activity mean", "BlobPersistence"]
        n_metrics = len(metrics)

        exp_color = (0.2, 0.4, 0.8)
        ising_color = (0.8, 0.2, 0.2)

        fig, axes = plt.subplots(1, n_metrics, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))

        for i, metric in enumerate(metrics):
            ax = axes[i]
            positions = []
            conditions_plotted = []
            pos_idx = 0

            for condition in self.conditions:
                if condition not in exp_stats or condition not in comparison:
                    continue

                best_idx = self._get_best_idx(condition)
                if len(best_idx) == 0:
                    continue

                if metric == "Moran's I mean":
                    exp_dist = exp_stats[condition].get('MoransI_all', np.array([]))
                    ising_all = ising_data.get('MoransI_all', [])
                    ising_dist = ising_all[best_idx[0]] if len(ising_all) > best_idx[0] else np.array([])
                elif metric == "Activity mean":
                    if use_masked:
                        exp_dist = exp_stats[condition].get('Activity_all_masked', np.array([]))
                    else:
                        exp_dist = exp_stats[condition].get('Activity_all', np.array([]))
                    ising_all = ising_data.get('Activity_all', [])
                    ising_dist = ising_all[best_idx[0]] if len(ising_all) > best_idx[0] else np.array([])
                else:  # BlobPersistence
                    exp_dist = exp_stats[condition].get('BlobPersistence_lifetimes', np.array([]))
                    if exp_dist is None:
                        exp_dist = np.array([])
                    ising_bp = ising_data.get('BlobPersistence_lifetimes', [])
                    if len(ising_bp) > best_idx[0]:
                        ising_dist = ising_bp[best_idx[0]]
                        if ising_dist is None:
                            ising_dist = np.array([])
                    else:
                        ising_dist = np.array([])

                exp_dist = np.asarray(exp_dist).ravel()
                ising_dist = np.asarray(ising_dist).ravel()
                exp_dist = exp_dist[~np.isnan(exp_dist)]
                ising_dist = ising_dist[~np.isnan(ising_dist)]

                if len(exp_dist) < 2 and len(ising_dist) < 2:
                    continue

                center = pos_idx * 2
                offset = 0.4

                if len(exp_dist) >= 2:
                    vp_exp = ax.violinplot(exp_dist, positions=[center - offset],
                                           showmeans=True, showmedians=False, widths=0.6)
                    for body in vp_exp['bodies']:
                        body.set_facecolor(exp_color)
                        body.set_alpha(0.7)
                    for key in ('cmeans', 'cmins', 'cmaxes', 'cbars'):
                        if key in vp_exp:
                            vp_exp[key].set_color(exp_color)

                if len(ising_dist) >= 2:
                    vp_ising = ax.violinplot(ising_dist, positions=[center + offset],
                                              showmeans=True, showmedians=False, widths=0.6)
                    for body in vp_ising['bodies']:
                        body.set_facecolor(ising_color)
                        body.set_alpha(0.7)
                    for key in ('cmeans', 'cmins', 'cmaxes', 'cbars'):
                        if key in vp_ising:
                            vp_ising[key].set_color(ising_color)

                positions.append(center)
                conditions_plotted.append(condition)
                pos_idx += 1

            if len(conditions_plotted) == 0:
                ax.set_visible(False)
                continue

            ax.set_xticks(positions)
            ax.set_xticklabels(conditions_plotted, rotation=45)
            ax.set_ylabel(metric)
            ax.set_title(metric)

            if i == 0:
                from matplotlib.patches import Patch
                ax.legend(handles=[Patch(facecolor=exp_color, alpha=0.7, label='Experimental'),
                                   Patch(facecolor=ising_color, alpha=0.7, label='Best Ising')],
                          fontsize=8)

        suffix = ' (masked)' if use_masked else ''
        fig.suptitle(f'Exp vs Best Ising Match (Violin){suffix}', fontsize=12)
        fig.tight_layout()
        save_name = 'exp_vs_ising_violin_masked' if use_masked else 'exp_vs_ising_violin'
        self._save_figure(fig, save_name)

    def plot_wasserstein_distances(self):
        """Plot Wasserstein distance distributions per condition."""
        comparison = self.results.get('Comparison', {})

        fig, axes = plt.subplots(1, len(self.conditions), figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))
        if len(self.conditions) == 1:
            axes = [axes]

        for i, condition in enumerate(self.conditions):
            ax = axes[i]

            if condition not in comparison:
                ax.set_visible(False)
                continue

            wd = comparison[condition].get('wasserstein_dist', np.array([]))
            if len(wd) == 0:
                ax.set_visible(False)
                continue

            wd_valid = wd[~np.isnan(wd)]
            ax.hist(wd_valid, bins=50, color=self._get_color(condition), alpha=0.7)

            ax.set_xlabel('Wasserstein Distance')
            ax.set_ylabel('Count')
            ax.set_title(f'{condition}\nMin WD={np.nanmin(wd):.4f}')

        fig.suptitle('Wasserstein Distance Distribution', fontsize=12)
        fig.tight_layout()
        self._save_figure(fig, 'wasserstein_distances')

    # =========================================================================
    # UMAP VISUALIZATIONS (Optional - requires umap-learn)
    # =========================================================================

    def _compute_6stat_ising_features(self, metric_key: str) -> np.ndarray:
        """Build (n_sims, 6) feature matrix from Ising data for a given metric.

        Parameters
        ----------
        metric_key : str
            Key into IsingData, e.g. 'MoransI_all', 'Activity_all',
            'BlobPersistence_lifetimes'.

        Returns
        -------
        np.ndarray
            Shape (n_sims, 6).
        """
        ising_data = self.results.get('IsingData', {})
        data_all = ising_data.get(metric_key, [])
        n_sims = len(data_all)
        features = np.zeros((n_sims, 6))
        for s in range(n_sims):
            arr = np.array(data_all[s]) if isinstance(data_all[s], (list, np.ndarray)) else np.array([])
            features[s, :] = compute_6_statistics(arr)
        return features

    def _compute_6stat_exp_features(self, metric_key: str) -> np.ndarray:
        """Build (n_conds, 6) feature matrix from ExpStats for a given metric.

        Parameters
        ----------
        metric_key : str
            Key into ExpStats[condition], e.g. 'MoransI_all', 'Activity_all',
            'BlobPersistence_lifetimes'.

        Returns
        -------
        np.ndarray
            Shape (n_conds, 6).
        """
        exp_stats = self.results.get('ExpStats', {})
        n_conds = len(self.conditions)
        features = np.zeros((n_conds, 6))
        for c, condition in enumerate(self.conditions):
            if condition in exp_stats:
                data = exp_stats[condition].get(metric_key, np.array([]))
                features[c, :] = compute_6_statistics(data)
        return features

    def _prepare_umap_features(self, all_features: np.ndarray) -> np.ndarray:
        """NaN-fill with column mean, then z-score normalize.

        Parameters
        ----------
        all_features : np.ndarray
            Shape (n_samples, n_features), may contain NaN.

        Returns
        -------
        np.ndarray
            Normalized feature matrix with no NaN/inf values.
        """
        out = all_features.copy()
        for col in range(out.shape[1]):
            col_data = out[:, col]
            nan_mask = np.isnan(col_data)
            if np.any(nan_mask) and not np.all(nan_mask):
                col_data[nan_mask] = np.nanmean(col_data)
                out[:, col] = col_data
        out = (out - np.nanmean(out, axis=0)) / (np.nanstd(out, axis=0) + 1e-10)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    # =========================================================================
    # SHARED HELPERS FOR MDE / MATCH-SPECIFICITY
    # =========================================================================

    def _compute_embedding(self, features_norm: np.ndarray, method: str,
                           cache_key: str = None,
                           precomputed: bool = False) -> np.ndarray:
        """Compute 2D embedding using the specified method.

        Parameters
        ----------
        features_norm : np.ndarray
            Normalized feature matrix (n_samples, n_features), or a symmetric
            distance matrix (n_samples, n_samples) when precomputed=True.
        method : str
            One of 'umap', 'densmap', 'pca'.
        cache_key : str, optional
            If provided, cache the result under this key.
        precomputed : bool
            If True, ``features_norm`` is a pairwise distance matrix. PCA
            does not support this and will raise.

        Returns
        -------
        np.ndarray
            Shape (n_samples, 2).
        """
        if cache_key and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # MDE disabled — UMAP preferred; keeping code for potential future use
        # if method == 'mde':
        #     if precomputed:
        #         import torch
        #         from scipy.sparse import csr_matrix
        #         n = features_norm.shape[0]
        #         k = min(15, n - 1)
        #         # Build sparse k-NN graph from distance matrix
        #         rows, cols, vals = [], [], []
        #         for i in range(n):
        #             dists = features_norm[i].copy()
        #             dists[i] = np.inf  # exclude self
        #             nn_idx = np.argpartition(dists, k)[:k]
        #             for j in nn_idx:
        #                 rows.append(i); cols.append(j); vals.append(dists[j])
        #                 rows.append(j); cols.append(i); vals.append(dists[j])
        #         sparse_graph = csr_matrix(
        #             (vals, (rows, cols)), shape=(n, n))
        #         embedding = pymde.preserve_neighbors(
        #             sparse_graph, embedding_dim=2,
        #             constraint=pymde.Standardized()
        #         ).embed().numpy()
        #     else:
        #         import torch
        #         tensor = torch.tensor(features_norm, dtype=torch.float32)
        #         embedding = pymde.preserve_neighbors(
        #             tensor, embedding_dim=2,
        #             constraint=pymde.Standardized()
        #         ).embed().numpy()
        # elif method == 'umap':
        if method == 'umap':
            if precomputed:
                reducer = umap.UMAP(
                    n_neighbors=min(15, features_norm.shape[0] - 1),
                    min_dist=0.3, random_state=42,
                    metric='precomputed')
            else:
                reducer = umap.UMAP(
                    n_neighbors=min(199, features_norm.shape[0] - 1),
                    min_dist=0.3, random_state=42)
            embedding = reducer.fit_transform(features_norm)
        elif method == 'densmap':
            # densMAP is a UMAP variant that preserves local density. It uses
            # the same precomputed/raw branch as UMAP and the same
            # random_state monkey-patch applied by the seed fanout.
            if precomputed:
                reducer = umap.UMAP(
                    n_neighbors=min(15, features_norm.shape[0] - 1),
                    min_dist=0.3, random_state=42,
                    metric='precomputed', densmap=True)
            else:
                reducer = umap.UMAP(
                    n_neighbors=min(199, features_norm.shape[0] - 1),
                    min_dist=0.3, random_state=42,
                    densmap=True)
            embedding = reducer.fit_transform(features_norm)
        elif method == 'pca':
            if precomputed:
                raise ValueError(
                    "PCA needs raw features, not a precomputed distance "
                    "matrix. Use _build_std_features(...) + "
                    "_prepare_umap_features(...) for the feature matrix.")
            if not HAS_SKLEARN:
                raise RuntimeError(
                    "scikit-learn not available — install scikit-learn to "
                    "use PCA embeddings.")
            embedding = PCA(n_components=2).fit_transform(features_norm)
        else:
            raise ValueError(f"Unknown embedding method: {method}")

        if cache_key:
            self._embedding_cache[cache_key] = embedding
        return embedding

    def _min_wd_across_conditions(self) -> np.ndarray:
        """Compute minimum Wasserstein distance for each Ising sim across conditions.

        Returns
        -------
        np.ndarray
            Shape (n_sims,) — minimum WD across all conditions for each sim.
        """
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})
        n_sims = len(ising_data.get('MoransI_all', []))
        if n_sims == 0:
            return np.array([])

        min_wd = np.full(n_sims, np.inf)
        for condition in self.conditions:
            if condition in comparison:
                wd = comparison[condition].get('wasserstein_dist', np.array([]))
                wd = np.atleast_1d(np.asarray(wd, dtype=float))
                if len(wd) == n_sims:
                    min_wd = np.minimum(min_wd, wd)
        # Replace inf with NaN for sims with no WD data
        min_wd[np.isinf(min_wd)] = np.nan
        return min_wd

    def _get_wd_embedding(self, metric_keys, method: str, cache_key,
                          extra_scalar_keys=None):
        """Get WD-based embedding, using cache when available.

        Avoids rebuilding the ~7 GB distance matrix when the embedding
        is already cached.

        Returns
        -------
        embedding : np.ndarray or None
            Shape (n_total, 2), or None if no data.
        n_sims : int
            Number of Ising simulations (first n_sims rows).
        """
        if cache_key in self._embedding_cache:
            embedding = self._embedding_cache[cache_key]
            exp_stats = self.results.get('ExpStats', {})
            n_exp = sum(1 for c in self.conditions if c in exp_stats)
            n_sims = embedding.shape[0] - n_exp
            return embedding, n_sims

        dist_matrix, n_sims = self._build_wd_distance_matrix(
            metric_keys, extra_scalar_keys=extra_scalar_keys)
        if n_sims == 0:
            return None, 0
        embedding = self._compute_embedding(
            dist_matrix, method, cache_key=cache_key, precomputed=True)
        del dist_matrix
        return embedding, n_sims

    def _build_wd_distance_matrix(self, metric_keys,
                                  n_quantiles: int = 100,
                                  extra_scalar_keys=None):
        """Build pairwise WD-approximation distance matrix via quantile L1.

        The 1-Wasserstein distance equals the L1 distance between quantile
        functions: W1 = integral |F_inv(p) - G_inv(p)| dp, approximated as
        (1/N) * sum |q_i - r_i|.  We evaluate quantiles at N equally-spaced
        percentiles, then use scipy pdist(metric='cityblock') for speed.

        Parameters
        ----------
        metric_keys : str or list of str
            Key(s) into IsingData / ExpStats, e.g. 'MoransI_all' or
            ['MoransI_all', 'Activity_all'].
        n_quantiles : int
            Number of equally-spaced percentiles (default 100).
        extra_scalar_keys : list of str, optional
            Extra scalar keys to append as z-scored columns
            (e.g. ['Autocorr_tau', 'Autocorr_fitR2']).

        Returns
        -------
        dist_matrix : np.ndarray
            Shape (n_total, n_total) symmetric distance matrix.
        n_sims : int
            Number of Ising simulations (first n_sims rows/cols).
        """
        from scipy.spatial.distance import pdist, squareform

        # Backward compat: single string → list
        if isinstance(metric_keys, str):
            metric_keys = [metric_keys]

        ising_data = self.results.get('IsingData', {})
        exp_stats = self.results.get('ExpStats', {})

        # Determine n_sims from first metric key
        data_first = ising_data.get(metric_keys[0], [])
        n_sims = len(data_first)

        percentiles = np.linspace(0, 100, n_quantiles)

        # Build quantile feature blocks for each metric key
        quantile_blocks = []
        for metric_key in metric_keys:
            data_all = ising_data.get(metric_key, [])
            distributions = []
            for s in range(n_sims):
                if s < len(data_all):
                    arr = np.asarray(data_all[s]).ravel()
                    arr = arr[~np.isnan(arr)]
                else:
                    arr = np.array([])
                distributions.append(arr)
            for condition in self.conditions:
                if condition in exp_stats:
                    arr = np.asarray(exp_stats[condition].get(metric_key,
                                                             np.array([]))).ravel()
                    arr = arr[~np.isnan(arr)]
                    distributions.append(arr)

            n_total = len(distributions)
            block = np.zeros((n_total, n_quantiles))
            for i, dist in enumerate(distributions):
                if len(dist) > 0:
                    block[i, :] = np.percentile(dist, percentiles)
            quantile_blocks.append(block)

        # Horizontally concatenate all quantile blocks
        features = np.hstack(quantile_blocks)

        # Append z-scored scalar columns if requested
        if extra_scalar_keys:
            n_total = features.shape[0]
            for skey in extra_scalar_keys:
                col = np.full((n_total, 1), np.nan)
                ising_vals = ising_data.get(skey, [])
                for s in range(min(n_sims, len(ising_vals))):
                    v = ising_vals[s]
                    if not isinstance(v, (list, np.ndarray)):
                        col[s, 0] = v
                idx = n_sims
                for condition in self.conditions:
                    if condition in exp_stats:
                        v = exp_stats[condition].get(f'{skey}_trial_averaged',
                                                    exp_stats[condition].get(skey, np.nan))
                        if not isinstance(v, (list, np.ndarray)):
                            col[idx, 0] = v
                        idx += 1
                # Z-score the scalar column
                valid = col[~np.isnan(col)]
                if len(valid) > 1:
                    mu, sigma = np.mean(valid), np.std(valid)
                    if sigma > 0:
                        col = (col - mu) / sigma
                col = np.nan_to_num(col, nan=0.0)
                features = np.hstack([features, col])

        # Pairwise L1 distance, normalized by total column count
        dist_condensed = pdist(features, metric='cityblock') / features.shape[1]
        dist_matrix = squareform(dist_condensed)
        del dist_condensed, features

        return dist_matrix, n_sims

    def _compute_sim_to_sim_wd(self, indices_a, indices_b, metric='MoransI_all'):
        """Compute pairwise Wasserstein distances between two sets of simulations.

        Parameters
        ----------
        indices_a, indices_b : array-like
            0-indexed simulation indices.
        metric : str
            Key into IsingData (e.g. 'MoransI_all', 'Activity_all').

        Returns
        -------
        np.ndarray
            Shape (len(indices_a), len(indices_b)).
        """
        ising_data = self.results.get('IsingData', {})
        data_all = ising_data.get(metric, [])
        n_sims = len(data_all)
        indices_a = np.atleast_1d(np.asarray(indices_a, dtype=int))
        indices_b = np.atleast_1d(np.asarray(indices_b, dtype=int))
        mat = np.full((len(indices_a), len(indices_b)), np.nan)
        for i, ia in enumerate(indices_a):
            if ia < 0 or ia >= n_sims:
                continue
            arr_a = np.asarray(data_all[ia]).ravel()
            for j, ib in enumerate(indices_b):
                if ib < 0 or ib >= n_sims:
                    continue
                arr_b = np.asarray(data_all[ib]).ravel()
                mat[i, j] = _wasserstein_1d(arr_a, arr_b)
        return mat

    def _scatter_top_n_matches(self, ax, embedding_ising, top_n, style, zoomed=False):
        """Plot top-N Ising matches with split markers for overlapping indices.

        Parameters
        ----------
        ax : matplotlib Axes
        embedding_ising : (n_sims, 2) array of Ising coordinates.
        top_n : int — how many best matches per condition.
        style : 'standard' or 'continuous' — controls edge color and size.

        Returns
        -------
        extra_handles : list of Line2D
            Legend handles for any overlap pairs (caller should merge into legend).
        """
        comparison = self.results.get('Comparison', {})

        # 1. Collect top-N indices per condition → idx_to_conditions
        idx_to_conditions = defaultdict(list)  # {ising_idx: [(condition, rank), ...]}
        per_cond_n_top = {}                    # {condition: n_top actually used}
        for condition in self.conditions:
            if condition not in comparison:
                continue
            best_idx = self._get_best_idx(condition)
            if len(best_idx) == 0:
                continue
            n_top = min(top_n, len(best_idx))
            per_cond_n_top[condition] = n_top
            for rank, idx in enumerate(best_idx[:n_top]):
                idx_to_conditions[idx].append((condition, rank))

        # 2. Partition into single-condition and multi-condition
        single_cond = defaultdict(list)   # {condition: [indices]}
        multi_cond = defaultdict(list)    # {(condA, condB): [indices]}
        for idx, cond_ranks in idx_to_conditions.items():
            if len(cond_ranks) == 1:
                single_cond[cond_ranks[0][0]].append(idx)
            else:
                # Pick the two conditions with lowest rank (best match)
                sorted_cr = sorted(cond_ranks, key=lambda cr: cr[1])
                pair = tuple(sorted([sorted_cr[0][0], sorted_cr[1][0]]))
                multi_cond[pair].append(idx)

        # Styling
        if style == 'continuous':
            edge_single = 'white'
            s_single = 140 if zoomed else 30
            s_half = 140 if zoomed else 30
            edge_half = 'white'
        else:
            edge_single = 'black'
            s_single = 100 if zoomed else 30
            s_half = 100 if zoomed else 30
            edge_half = 'black'

        # 3. Plot single-condition points (preserve condition order for legend)
        for condition in self.conditions:
            if condition not in per_cond_n_top:
                continue
            indices = single_cond.get(condition, [])
            n_top = per_cond_n_top[condition]
            if len(indices) > 0:
                idx_arr = np.array(indices)
                ax.scatter(
                    embedding_ising[idx_arr, 0], embedding_ising[idx_arr, 1],
                    c=[self._get_color(condition)], s=s_single,
                    edgecolors=edge_single, linewidth=1.5,
                    label=f'{condition} (Top {n_top} Ising)', zorder=3)
            else:
                # Still need the legend entry — plot invisible point
                ax.scatter([], [], c=[self._get_color(condition)], s=s_single,
                           edgecolors=edge_single, linewidth=1.5,
                           label=f'{condition} (Top {n_top} Ising)', zorder=3)

        # 4. Plot multi-condition (split markers)
        extra_handles = []
        for (condA, condB), indices in multi_cond.items():
            idx_arr = np.array(indices)
            colorA = self._get_color(condA)
            colorB = self._get_color(condB)

            ax.scatter(
                embedding_ising[idx_arr, 0], embedding_ising[idx_arr, 1],
                c=[colorA], marker=MARKER_LEFT_HALF, s=s_half,
                edgecolors=edge_half, linewidth=1.0, zorder=3)
            ax.scatter(
                embedding_ising[idx_arr, 0], embedding_ising[idx_arr, 1],
                c=[colorB], marker=MARKER_RIGHT_HALF, s=s_half,
                edgecolors=edge_half, linewidth=1.0, zorder=3)

            # Legend handle for this overlap pair
            handle = Line2D(
                [0], [0], marker=MARKER_LEFT_HALF, color='w',
                markerfacecolor=colorA, markeredgecolor=edge_half,
                markersize=8, linewidth=0,
                label=f'{condA}+{condB} ({len(indices)} shared)')
            extra_handles.append(handle)

        return extra_handles

    def _draw_match_clouds(self, ax, embedding_ising, top_n):
        """Draw filled convex hulls around each condition's top-N matches."""
        from scipy.spatial import ConvexHull
        comparison = self.results.get('Comparison', {})
        hulls = {}
        hull_areas = {}
        for condition in self.conditions:
            if condition not in comparison:
                continue
            best_idx = self._get_best_idx(condition)
            if len(best_idx) < 3:
                continue
            n_top = min(top_n, len(best_idx))
            pts = embedding_ising[np.array(best_idx[:n_top])]
            try:
                hull = ConvexHull(pts)
            except Exception:
                continue
            hull_pts = pts[hull.vertices]
            hulls[condition] = hull_pts
            hull_areas[condition] = hull.volume  # 2D: volume = area
            color = self._get_color(condition)
            polygon = plt.Polygon(hull_pts, alpha=0.15, fc=color, ec=color,
                                  lw=2, zorder=2)
            ax.add_patch(polygon)

        # Overlap annotations
        annot_lines = [f'{c}: area={a:.2f}' for c, a in hull_areas.items()]
        cond_with_hull = [c for c in self.conditions if c in hulls]
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
            for i, c1 in enumerate(cond_with_hull):
                for c2 in cond_with_hull[i + 1:]:
                    p1 = ShapelyPolygon(hulls[c1])
                    p2 = ShapelyPolygon(hulls[c2])
                    if p1.intersects(p2):
                        overlap = p1.intersection(p2).area
                        pct1 = 100 * overlap / p1.area if p1.area > 0 else 0
                        pct2 = 100 * overlap / p2.area if p2.area > 0 else 0
                        annot_lines.append(
                            f'{c1}/{c2} overlap: {pct1:.0f}%/{pct2:.0f}%')
        except ImportError:
            pass
        if annot_lines:
            ax.text(0.02, 0.02, '\n'.join(annot_lines),
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment='bottom', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.7))

    def _plot_embedding_generic(self, embedding_ising: np.ndarray,
                                embedding_exp: np.ndarray,
                                exp_labels: list, title: str,
                                filename_base: str, top_n_list: List[int],
                                method_label: str, style: str,
                                wd_for_color: np.ndarray = None,
                                exp_marker: str = None,
                                exp_marker_size: float = None,
                                cloud: bool = False):
        """Render full + zoomed embedding figures for each top_n value.

        Parameters
        ----------
        embedding_ising : (n_sims, 2) Ising simulation coordinates.
        embedding_exp : (n_exp, 2) experimental point coordinates.
        exp_labels : condition label for each experimental point.
        title : figure title (may contain newlines).
        filename_base : base filename without suffix.
        top_n_list : list of top-N values.
        method_label : axis label prefix (e.g. 'MDE', 'Laplacian Eigenmap').
        style : 'standard' (UMAP-like) or 'continuous' (WD colormap).
        wd_for_color : (n_sims,) min WD values for continuous coloring.
        exp_marker : marker for exp points (default: 'p' for standard, 's' for continuous).
        exp_marker_size : size for exp points (default: 300 for standard, 250 for continuous).
        """
        comparison = self.results.get('Comparison', {})

        if exp_marker is None:
            exp_marker = '*'
        if exp_marker_size is None:
            exp_marker_size = 350 if style == 'continuous' else 400

        for top_n in top_n_list:
            for zoomed in [False, True]:
                fig, ax = plt.subplots(figsize=(10, 8))

                # --- Ising background ---
                bg_s = {'continuous': 120 if zoomed else 25,
                        'standard': 100 if zoomed else 20}[style]
                if style == 'continuous' and wd_for_color is not None:
                    vmin = np.nanpercentile(wd_for_color, 2)
                    vmax = np.nanpercentile(wd_for_color, 85)
                    sc = ax.scatter(
                        embedding_ising[:, 0], embedding_ising[:, 1],
                        c=wd_for_color, cmap='Greys_r', alpha=0.6, s=bg_s,
                        vmin=vmin, vmax=vmax)
                    plt.colorbar(sc, ax=ax, label='Min Wasserstein Distance',
                                 shrink=0.8)
                else:
                    ax.scatter(embedding_ising[:, 0], embedding_ising[:, 1],
                               c='lightgray', alpha=0.4, s=bg_s)

                # --- Top-N Ising best matches (with split markers for overlaps) ---
                extra_handles = self._scatter_top_n_matches(
                    ax, embedding_ising, top_n, style, zoomed=zoomed)

                if cloud:
                    self._draw_match_clouds(ax, embedding_ising, top_n)

                # --- Experimental points ---
                for c, label in enumerate(exp_labels):
                    ax.scatter(
                        embedding_exp[c, 0], embedding_exp[c, 1],
                        c=[self._get_color(label)], s=exp_marker_size,
                        marker=exp_marker, edgecolors='black', linewidth=2,
                        label=f'{label} (Exp)', zorder=5)

                ax.set_xlabel(f'{method_label} 1')
                ax.set_ylabel(f'{method_label} 2')

                # Zoom
                if zoomed:
                    poi_coords = list(embedding_exp)
                    for condition in self.conditions:
                        if condition in comparison:
                            best_idx = self._get_best_idx(condition)
                            if len(best_idx) > 0:
                                n_top = min(top_n, len(best_idx))
                                for idx in best_idx[:n_top]:
                                    poi_coords.append(embedding_ising[idx])
                    poi_coords = np.array(poi_coords)
                    x_min, x_max = poi_coords[:, 0].min(), poi_coords[:, 0].max()
                    y_min, y_max = poi_coords[:, 1].min(), poi_coords[:, 1].max()
                    x_pad = 0.2 * (x_max - x_min) if x_max > x_min else 1.0
                    y_pad = 0.2 * (y_max - y_min) if y_max > y_min else 1.0
                    ax.set_xlim(x_min - x_pad, x_max + x_pad)
                    ax.set_ylim(y_min - y_pad, y_max + y_pad)
                    ax.set_title(f'{title} -- Zoomed (Top {top_n} Matches)')
                else:
                    ax.set_title(f'{title} (Top {top_n} Matches)')

                handles, labels = ax.get_legend_handles_labels()
                handles.extend(extra_handles)
                ax.legend(handles=handles, loc='best', fontsize=8)
                fig.tight_layout()
                suffix = f'_top{top_n}' if top_n != 3 else ''
                zoom_tag = '_zoomed' if zoomed else ''
                self._save_figure(fig, f'{filename_base}{suffix}{zoom_tag}')

    def _build_std_features(self, metric_keys: list,
                            extra_scalar_keys: list = None):
        """Build feature matrices for standard embedding plots.

        Parameters
        ----------
        metric_keys : list of str
            Keys into IsingData / ExpStats for 6-stat features
            (e.g. ['MoransI_all', 'Activity_all']).
        extra_scalar_keys : list of str, optional
            Extra scalar columns to append (e.g. ['Autocorr_tau', 'Autocorr_fitR2']).

        Returns
        -------
        ising_feat : np.ndarray  (n_sims, n_features)
        exp_feat : np.ndarray    (n_conds, n_features)
        n_sims : int
        """
        ising_data = self.results.get('IsingData', {})
        exp_stats = self.results.get('ExpStats', {})

        # 6-stat features for each metric key
        ising_blocks = []
        exp_blocks = []
        n_sims = None
        for key in metric_keys:
            ising_f = self._compute_6stat_ising_features(key)
            if n_sims is None:
                n_sims = ising_f.shape[0]
            # Pad if shorter
            if ising_f.shape[0] < n_sims:
                ising_f = np.vstack([ising_f,
                                     np.zeros((n_sims - ising_f.shape[0], 6))])
            ising_blocks.append(ising_f[:n_sims])
            exp_blocks.append(self._compute_6stat_exp_features(key))

        if n_sims is None or n_sims == 0:
            return np.zeros((0, 6)), np.zeros((0, 6)), 0

        # Extra scalar columns
        if extra_scalar_keys:
            n_conds = len(self.conditions)
            for skey in extra_scalar_keys:
                ising_col = np.full((n_sims, 1), np.nan)
                exp_col = np.full((n_conds, 1), np.nan)
                ising_vals = ising_data.get(skey, [])
                for s in range(min(n_sims, len(ising_vals))):
                    v = ising_vals[s]
                    if not isinstance(v, (list, np.ndarray)):
                        ising_col[s, 0] = v
                for c, cond in enumerate(self.conditions):
                    if cond in exp_stats:
                        # Try trial-averaged variant first
                        v = exp_stats[cond].get(f'{skey}_trial_averaged',
                                                exp_stats[cond].get(skey, np.nan))
                        if not isinstance(v, (list, np.ndarray)):
                            exp_col[c, 0] = v
                ising_blocks.append(ising_col)
                exp_blocks.append(exp_col)

        ising_feat = np.hstack(ising_blocks)
        exp_feat = np.hstack(exp_blocks)

        # Fill missing blob data from Global stats if applicable
        if 'BlobPersistence_lifetimes' in metric_keys:
            if 'Global' in exp_stats and 'BlobPersistence' in exp_stats['Global']:
                global_blob = exp_stats['Global']['BlobPersistence']
                if 'lifetimes' in global_blob:
                    global_feat = compute_6_statistics(global_blob['lifetimes'])
                    # Find the column block for blob features
                    blob_idx = metric_keys.index('BlobPersistence_lifetimes')
                    col_start = blob_idx * 6
                    for c in range(exp_feat.shape[0]):
                        if np.all(np.isnan(exp_feat[c, col_start:col_start + 6])):
                            exp_feat[c, col_start:col_start + 6] = global_feat

        return ising_feat, exp_feat, n_sims

    def _build_per_recording_features(self, metric_key: str):
        """Build per-recording feature matrix for embedding.

        Returns
        -------
        ising_feat : (n_sims, 6)
        rec_feat : (n_recs, 6)
        rec_labels : list of str  (condition per recording)
        n_sims : int
        """
        rec_meta = self.results.get('RecordingMetadata', {})
        exp_stats = self.results.get('ExpStats', {})

        ising_feat = self._compute_6stat_ising_features(metric_key)
        n_sims = ising_feat.shape[0]
        if n_sims == 0 or not rec_meta:
            return ising_feat, np.zeros((0, 6)), [], n_sims

        rec_features_list = []
        rec_labels = []
        for condition in self.conditions:
            if condition not in exp_stats or condition not in rec_meta:
                continue
            meta = rec_meta[condition]
            nTrials_per_rec = meta.get('nTrials_per_recording', None)
            if nTrials_per_rec is None or len(nTrials_per_rec) == 0:
                continue
            # Derive the trials key from metric_key (e.g. MoransI_all -> MoransI_trials)
            trials_key = metric_key.replace('_all', '_trials')
            mi_trials = exp_stats[condition].get(trials_key, None)
            if mi_trials is None:
                continue
            chunks = split_trials_by_recording(mi_trials, nTrials_per_rec)
            for chunk in chunks:
                rec_features_list.append(compute_6_statistics(chunk.ravel()))
                rec_labels.append(condition)

        if len(rec_features_list) == 0:
            return ising_feat, np.zeros((0, 6)), [], n_sims

        return ising_feat, np.array(rec_features_list), rec_labels, n_sims

    def _build_per_animal_features(self, metric_key: str):
        """Build per-animal feature matrix for embedding.

        Returns
        -------
        ising_feat : (n_sims, 6)
        animal_feat : (n_animals, 6)
        animal_labels : list of str  (condition per animal)
        animal_markers : dict  {animal_name: marker_shape}
        n_sims : int
        """
        from collections import defaultdict

        rec_meta = self.results.get('RecordingMetadata', {})
        exp_stats = self.results.get('ExpStats', {})

        ising_feat = self._compute_6stat_ising_features(metric_key)
        n_sims = ising_feat.shape[0]
        empty = (ising_feat, np.zeros((0, 6)), [], {}, n_sims)
        if n_sims == 0 or not rec_meta:
            return empty

        animal_features_list = []
        animal_labels = []
        animal_names_list = []

        for condition in self.conditions:
            if condition not in exp_stats or condition not in rec_meta:
                continue
            meta = rec_meta[condition]
            nTrials_per_rec = meta.get('nTrials_per_recording', None)
            animal_names = meta.get('animal_names', None)
            if nTrials_per_rec is None or animal_names is None:
                continue
            trials_key = metric_key.replace('_all', '_trials')
            mi_trials = exp_stats[condition].get(trials_key, None)
            if mi_trials is None:
                continue
            chunks = split_trials_by_recording(mi_trials, nTrials_per_rec)
            animal_chunks = defaultdict(list)
            for chunk, name in zip(chunks, animal_names):
                animal_chunks[name].append(chunk)
            for name, chunk_list in animal_chunks.items():
                pooled = np.concatenate(chunk_list, axis=0)
                animal_features_list.append(compute_6_statistics(pooled.ravel()))
                animal_labels.append(condition)
                animal_names_list.append(name)

        if len(animal_features_list) == 0:
            return empty

        unique_animals = sorted(set(animal_names_list))
        marker_shapes = ['o', 's', '^', 'v', '<', '>', 'p', 'h', '8', 'D', '*', 'X', 'P']
        animal_marker_map = {name: marker_shapes[i % len(marker_shapes)]
                             for i, name in enumerate(unique_animals)}

        return (ising_feat, np.array(animal_features_list),
                list(zip(animal_labels, animal_names_list)),
                animal_marker_map, n_sims)

    def _plot_embedding_per_recording(self, embedding_ising, embedding_recs,
                                      rec_labels, title, filename_base,
                                      top_n_list, method_label, style,
                                      wd_for_color=None, cloud=False):
        """Render per-recording embedding (diamonds for each recording)."""
        comparison = self.results.get('Comparison', {})

        for top_n in top_n_list:
            for zoomed in [False, True]:
                fig, ax = plt.subplots(figsize=(10, 8))

                # Background
                if style == 'continuous' and wd_for_color is not None:
                    vmin = np.nanpercentile(wd_for_color, 2)
                    vmax = np.nanpercentile(wd_for_color, 85)
                    sc = ax.scatter(embedding_ising[:, 0], embedding_ising[:, 1],
                                   c=wd_for_color, cmap='Greys_r', alpha=0.6, s=120,
                                   vmin=vmin, vmax=vmax)
                    plt.colorbar(sc, ax=ax, label='Min Wasserstein Distance',
                                 shrink=0.8)
                else:
                    ax.scatter(embedding_ising[:, 0], embedding_ising[:, 1],
                               c='lightgray', alpha=0.4, s=100)

                # Top-N Ising matches (with split markers for overlaps)
                extra_handles = self._scatter_top_n_matches(
                    ax, embedding_ising, top_n, style, zoomed=zoomed)

                if cloud:
                    self._draw_match_clouds(ax, embedding_ising, top_n)

                # Per-recording diamonds
                for condition in self.conditions:
                    mask = [i for i, c in enumerate(rec_labels) if c == condition]
                    if len(mask) > 0:
                        idx = np.array(mask)
                        marker = 's' if style == 'continuous' else 'D'
                        ax.scatter(embedding_recs[idx, 0], embedding_recs[idx, 1],
                                   c=[self._get_color(condition)], s=200,
                                   marker=marker, edgecolors='black', linewidth=2,
                                   label=f'{condition} (n={len(mask)} recs)',
                                   zorder=4)

                ax.set_xlabel(f'{method_label} 1')
                ax.set_ylabel(f'{method_label} 2')

                if zoomed:
                    poi_coords = list(embedding_recs)
                    for condition in self.conditions:
                        if condition in comparison:
                            best_idx = self._get_best_idx(condition)
                            if len(best_idx) > 0:
                                for idx in best_idx[:min(top_n, len(best_idx))]:
                                    poi_coords.append(embedding_ising[idx])
                    poi_coords = np.array(poi_coords)
                    xmin, xmax = poi_coords[:, 0].min(), poi_coords[:, 0].max()
                    ymin, ymax = poi_coords[:, 1].min(), poi_coords[:, 1].max()
                    xp = 0.2 * (xmax - xmin) if xmax > xmin else 1.0
                    yp = 0.2 * (ymax - ymin) if ymax > ymin else 1.0
                    ax.set_xlim(xmin - xp, xmax + xp)
                    ax.set_ylim(ymin - yp, ymax + yp)
                    ax.set_title(f'{title} -- Zoomed (Top {top_n} Matches)')
                else:
                    ax.set_title(f'{title} (Top {top_n} Matches)')

                handles, labels = ax.get_legend_handles_labels()
                handles.extend(extra_handles)
                ax.legend(handles=handles, loc='best', fontsize=8)
                fig.tight_layout()
                suffix = f'_top{top_n}' if top_n != 3 else ''
                zoom_tag = '_zoomed' if zoomed else ''
                self._save_figure(fig, f'{filename_base}{suffix}{zoom_tag}')

    def _plot_embedding_per_animal(self, embedding_ising, embedding_animals,
                                   animal_info, animal_marker_map, title,
                                   filename_base, top_n_list, method_label,
                                   style, wd_for_color=None, cloud=False):
        """Render per-animal embedding (unique marker per animal)."""
        comparison = self.results.get('Comparison', {})
        n_animals = len(animal_info)

        for top_n in top_n_list:
            for zoomed in [False, True]:
                fig, ax = plt.subplots(figsize=(10, 8))

                if style == 'continuous' and wd_for_color is not None:
                    vmin = np.nanpercentile(wd_for_color, 2)
                    vmax = np.nanpercentile(wd_for_color, 85)
                    sc = ax.scatter(embedding_ising[:, 0], embedding_ising[:, 1],
                                   c=wd_for_color, cmap='Greys_r', alpha=0.6, s=120,
                                   vmin=vmin, vmax=vmax)
                    plt.colorbar(sc, ax=ax, label='Min Wasserstein Distance',
                                 shrink=0.8)
                else:
                    ax.scatter(embedding_ising[:, 0], embedding_ising[:, 1],
                               c='lightgray', alpha=0.4, s=100)

                # Top-N Ising matches (with split markers for overlaps)
                extra_handles = self._scatter_top_n_matches(
                    ax, embedding_ising, top_n, style, zoomed=zoomed)

                if cloud:
                    self._draw_match_clouds(ax, embedding_ising, top_n)

                plotted_animals = set()
                for i, (cond, name) in enumerate(animal_info):
                    marker = animal_marker_map[name]
                    lbl = f'{name}' if name not in plotted_animals else None
                    ax.scatter(embedding_animals[i, 0], embedding_animals[i, 1],
                               c=[self._get_color(cond)], s=250, marker=marker,
                               edgecolors='black', linewidth=2, label=lbl,
                               zorder=4)
                    plotted_animals.add(name)

                ax.set_xlabel(f'{method_label} 1')
                ax.set_ylabel(f'{method_label} 2')

                if zoomed:
                    poi_coords = [embedding_animals[i] for i in range(n_animals)]
                    for condition in self.conditions:
                        if condition in comparison:
                            best_idx = self._get_best_idx(condition)
                            if len(best_idx) > 0:
                                for idx in best_idx[:min(top_n, len(best_idx))]:
                                    poi_coords.append(embedding_ising[idx])
                    poi_coords = np.array(poi_coords)
                    xmin, xmax = poi_coords[:, 0].min(), poi_coords[:, 0].max()
                    ymin, ymax = poi_coords[:, 1].min(), poi_coords[:, 1].max()
                    xp = 0.2 * (xmax - xmin) if xmax > xmin else 1.0
                    yp = 0.2 * (ymax - ymin) if ymax > ymin else 1.0
                    ax.set_xlim(xmin - xp, xmax + xp)
                    ax.set_ylim(ymin - yp, ymax + yp)
                    ax.set_title(f'{title} -- Zoomed (Top {top_n} Matches)\n'
                                 f'{n_animals} animal-condition points | '
                                 f'marker shape = animal identity')
                else:
                    ax.set_title(f'{title} (Top {top_n} Matches)\n'
                                 f'{n_animals} animal-condition points | '
                                 f'marker shape = animal identity')

                handles, labels = ax.get_legend_handles_labels()
                handles.extend(extra_handles)
                ax.legend(handles=handles, loc='best', fontsize=7, ncol=2)
                fig.tight_layout()
                suffix = f'_top{top_n}' if top_n != 3 else ''
                zoom_tag = '_zoomed' if zoomed else ''
                self._save_figure(fig, f'{filename_base}{suffix}{zoom_tag}')

    def _plot_umap(self, embedding_ising: np.ndarray, embedding_exp: np.ndarray,
                   exp_labels: list, title: str, filename_base: str,
                   top_n_list: List[int],
                   exp_marker: str = '*', exp_marker_size: float = 400,
                   method_label: str = 'UMAP'):
        """Render full + zoomed figures for each top_n value.

        Parameters
        ----------
        embedding_ising : np.ndarray
            Shape (n_sims, 2) embedding coordinates for Ising simulations.
        embedding_exp : np.ndarray
            Shape (n_exp, 2) embedding coordinates for experimental points.
        exp_labels : list
            Condition label for each experimental point.
        title : str
            Figure title (may contain \\n for subtitle).
        filename_base : str
            Base filename without suffix (e.g. 'umap_morans_i_only').
        top_n_list : List[int]
            List of top-N values to generate figures for.
        exp_marker : str
            Marker style for experimental points.
        exp_marker_size : float
            Marker size for experimental points.
        method_label : str
            Axis-label prefix; defaults to 'UMAP' so existing callers
            stay identical. Pass 'PCA' or 'densMAP' for those variants.
        """
        comparison = self.results.get('Comparison', {})

        for top_n in top_n_list:
            fig, ax = plt.subplots(figsize=(10, 8))

            ax.scatter(embedding_ising[:, 0], embedding_ising[:, 1],
                       c='lightgray', alpha=0.4, s=20)

            # Top-N Ising matches (with split markers for overlaps)
            extra_handles = self._scatter_top_n_matches(
                ax, embedding_ising, top_n, 'standard')

            for c, label in enumerate(exp_labels):
                ax.scatter(embedding_exp[c, 0], embedding_exp[c, 1],
                           c=[self._get_color(label)], s=exp_marker_size,
                           marker=exp_marker, edgecolors='black', linewidth=2,
                           label=f'{label} (Exp)', zorder=5)

            ax.set_xlabel(f'{method_label} 1')
            ax.set_ylabel(f'{method_label} 2')
            ax.set_title(f'{title} (Top {top_n} Matches)')
            handles, labels = ax.get_legend_handles_labels()
            handles.extend(extra_handles)
            ax.legend(handles=handles, loc='best', fontsize=8)
            fig.tight_layout()

            suffix = f'_top{top_n}' if top_n != 3 else ''
            self._save_figure(fig, f'{filename_base}{suffix}')

            # --- Zoomed figure ---
            poi_coords = list(embedding_exp)
            for condition in self.conditions:
                if condition in comparison:
                    best_idx = self._get_best_idx(condition)
                    if len(best_idx) > 0:
                        n_top = min(top_n, len(best_idx))
                        for idx in best_idx[:n_top]:
                            poi_coords.append(embedding_ising[idx])
            poi_coords = np.array(poi_coords)

            x_min, x_max = poi_coords[:, 0].min(), poi_coords[:, 0].max()
            y_min, y_max = poi_coords[:, 1].min(), poi_coords[:, 1].max()
            x_pad = 0.2 * (x_max - x_min) if x_max > x_min else 1.0
            y_pad = 0.2 * (y_max - y_min) if y_max > y_min else 1.0

            fig_z, ax_z = plt.subplots(figsize=(10, 8))

            ax_z.scatter(embedding_ising[:, 0], embedding_ising[:, 1],
                         c='lightgray', alpha=0.4, s=100)

            # Top-N Ising matches (with split markers for overlaps)
            extra_handles_z = self._scatter_top_n_matches(
                ax_z, embedding_ising, top_n, 'standard', zoomed=True)

            for c, label in enumerate(exp_labels):
                ax_z.scatter(embedding_exp[c, 0], embedding_exp[c, 1],
                             c=[self._get_color(label)], s=exp_marker_size,
                             marker=exp_marker, edgecolors='black', linewidth=2,
                             label=f'{label} (Exp)', zorder=5)

            ax_z.set_xlim(x_min - x_pad, x_max + x_pad)
            ax_z.set_ylim(y_min - y_pad, y_max + y_pad)
            ax_z.set_xlabel(f'{method_label} 1')
            ax_z.set_ylabel(f'{method_label} 2')
            ax_z.set_title(f'{title} -- Zoomed (Top {top_n} Matches)')
            handles_z, labels_z = ax_z.get_legend_handles_labels()
            handles_z.extend(extra_handles_z)
            ax_z.legend(handles=handles_z, loc='best', fontsize=8)
            fig_z.tight_layout()

            self._save_figure(fig_z, f'{filename_base}{suffix}_zoomed')

    def plot_umap_parameter_space(self):
        """Plot UMAP of parameter space colored by condition match."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return

        ising_data = self.results.get('IsingData', {})
        comparison = self.results.get('Comparison', {})

        params = ising_data.get('params', {})
        if not params or len(params.get('beta', [])) == 0:
            return

        # Build feature matrix
        n_sims = len(params['beta'])
        features = np.column_stack([
            params.get('beta', np.zeros(n_sims)),
            params.get('c', np.zeros(n_sims)),
            params.get('decay_const', np.zeros(n_sims)),
            params.get('inhibition_range', np.zeros(n_sims)),
            params.get('bias', np.zeros(n_sims)),
        ])

        # Normalize
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

        # Run UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(features)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(embedding[:, 0], embedding[:, 1], c='lightgray', alpha=0.3, s=10)

        # Highlight best matches per condition
        for condition in self.conditions:
            if condition in comparison:
                best_idx = self._get_best_idx(condition)
                if len(best_idx) > 0:
                    ax.scatter(embedding[best_idx, 0], embedding[best_idx, 1],
                              c=self._get_color(condition), s=50, label=condition,
                              edgecolors='black', linewidth=0.5)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP of Parameter Space')
        ax.legend()
        fig.tight_layout()
        self._save_figure(fig, 'umap_parameter_space')

    def plot_umap_morans_i_only(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot UMAP of pure Moran's I feature space (6 features only)."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return

        ising_features = self._compute_6stat_ising_features('MoransI_all')
        if ising_features.shape[0] == 0:
            print("  No Moran's I data for UMAP")
            return

        exp_features = self._compute_6stat_exp_features('MoransI_all')
        n_sims = ising_features.shape[0]

        all_features = np.vstack([ising_features, exp_features])
        features_norm = self._prepare_umap_features(all_features)

        reducer = umap.UMAP(n_neighbors=min(199, len(features_norm) - 1),
                            min_dist=0.3, random_state=42)
        embedding_all = reducer.fit_transform(features_norm)

        self._plot_umap(
            embedding_all[:n_sims], embedding_all[n_sims:],
            list(self.conditions),
            "Moran's I Feature Space\n6 features: mean, std, median, skewness, kurtosis, IQR",
            'umap_morans_i_only', top_n_list)

    def plot_umap_morans_i_only_shaded(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot UMAP of Moran's I features with WD-shaded dots."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return
        ising_feat, exp_feat, n_sims = self._build_std_features(['MoransI_all'])
        if n_sims == 0:
            return
        all_features = np.vstack([ising_feat, exp_feat])
        features_norm = self._prepare_umap_features(all_features)
        embedding = self._compute_embedding(features_norm, 'umap',
                                            cache_key=('morans_i_only', 'umap'))
        wd_color = self._min_wd_across_conditions()
        self._plot_embedding_generic(
            embedding[:n_sims], embedding[n_sims:],
            list(self.conditions),
            "UMAP: Moran's I Feature Space\n6 features",
            'umap_morans_i_only_shaded', top_n_list,
            method_label='UMAP', style='continuous', wd_for_color=wd_color)

    def plot_umap_morans_i_only_wd(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot UMAP using WD-based distances (quantile L1)."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return

        embedding_all, n_sims = self._get_wd_embedding(
            'MoransI_all', 'umap', ('morans_i_only_wd', 'umap'))
        if embedding_all is None:
            print("  No Moran's I data for UMAP")
            return

        self._plot_umap(
            embedding_all[:n_sims], embedding_all[n_sims:],
            list(self.conditions),
            "Moran's I (WD Embedding)\nQuantile L1 distances",
            'umap_morans_i_only_wd', top_n_list)

    def plot_umap_morans_i_only_wd_shaded(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot UMAP using WD distances with WD-shaded dots."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return
        embedding, n_sims = self._get_wd_embedding(
            'MoransI_all', 'umap', ('morans_i_only_wd', 'umap'))
        if embedding is None:
            return
        wd_color = self._min_wd_across_conditions()
        self._plot_embedding_generic(
            embedding[:n_sims], embedding[n_sims:],
            list(self.conditions),
            "UMAP: Moran's I (WD Embedding)\nQuantile L1 distances",
            'umap_morans_i_only_wd_shaded', top_n_list,
            method_label='UMAP', style='continuous', wd_for_color=wd_color)

    def plot_umap_activity_features_wd(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot UMAP using WD-based distances for Activity only."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return
        embedding_all, n_sims = self._get_wd_embedding(
            'Activity_all', 'umap', ('activity_wd', 'umap'))
        if embedding_all is None:
            print("  No Activity data for UMAP")
            return
        self._plot_umap(
            embedding_all[:n_sims], embedding_all[n_sims:],
            list(self.conditions),
            "Activity (WD Embedding)\nQuantile L1 distances",
            'umap_activity_features_wd', top_n_list)

    def plot_umap_morans_i_activity_features_wd(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot UMAP using WD-based distances for Moran's I + Activity."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return
        embedding_all, n_sims = self._get_wd_embedding(
            ['MoransI_all', 'Activity_all'], 'umap', ('mi_activity_wd', 'umap'))
        if embedding_all is None:
            print("  No data for UMAP")
            return
        self._plot_umap(
            embedding_all[:n_sims], embedding_all[n_sims:],
            list(self.conditions),
            "Moran's I + Activity (WD Embedding)\nQuantile L1 distances",
            'umap_morans_i_activity_features_wd', top_n_list)

    def plot_umap_morans_i_activity_features_wd_shaded(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot UMAP using WD distances for MI + Activity with WD-shaded dots."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return
        embedding, n_sims = self._get_wd_embedding(
            ['MoransI_all', 'Activity_all'], 'umap', ('mi_activity_wd', 'umap'))
        if embedding is None:
            return
        wd_color = self._min_wd_across_conditions()
        self._plot_embedding_generic(
            embedding[:n_sims], embedding[n_sims:],
            list(self.conditions),
            "UMAP: Moran's I + Activity (WD Embedding)\nQuantile L1 distances",
            'umap_morans_i_activity_features_wd_shaded', top_n_list,
            method_label='UMAP', style='continuous', wd_for_color=wd_color)

    def plot_densmap_morans_i_activity_features_wd(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot densMAP using WD-based distances for Moran's I + Activity."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return
        embedding_all, n_sims = self._get_wd_embedding(
            ['MoransI_all', 'Activity_all'], 'densmap',
            ('mi_activity_wd', 'densmap'))
        if embedding_all is None:
            print("  No data for densMAP")
            return
        self._plot_umap(
            embedding_all[:n_sims], embedding_all[n_sims:],
            list(self.conditions),
            "Moran's I + Activity (WD Embedding, densMAP)\n"
            "Quantile L1 distances",
            'densmap_morans_i_activity_features_wd', top_n_list,
            method_label='densMAP')

    def plot_densmap_morans_i_activity_features_wd_shaded(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot densMAP WD distances for MI + Activity with WD-shaded dots."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return
        embedding, n_sims = self._get_wd_embedding(
            ['MoransI_all', 'Activity_all'], 'densmap',
            ('mi_activity_wd', 'densmap'))
        if embedding is None:
            return
        wd_color = self._min_wd_across_conditions()
        self._plot_embedding_generic(
            embedding[:n_sims], embedding[n_sims:],
            list(self.conditions),
            "densMAP: Moran's I + Activity (WD Embedding)\n"
            "Quantile L1 distances",
            'densmap_morans_i_activity_features_wd_shaded', top_n_list,
            method_label='densMAP', style='continuous', wd_for_color=wd_color)

    def plot_pca_morans_i_activity_features_wd(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot PCA of the same MI+Activity feature matrix used by the WD
        embeddings. PCA does not use the WD distance matrix — it operates
        directly on the normalised feature matrix — but the feature set is
        identical, so naming keeps the ``_wd`` suffix for parity.
        """
        if not HAS_SKLEARN:
            print("  scikit-learn not available (install scikit-learn)")
            return
        ising_feat, exp_feat, n_sims = self._build_std_features(
            ['MoransI_all', 'Activity_all'])
        if n_sims == 0:
            print("  No data for PCA")
            return
        all_features = np.vstack([ising_feat, exp_feat])
        features_norm = self._prepare_umap_features(all_features)
        embedding = self._compute_embedding(
            features_norm, 'pca', cache_key=('mi_activity_wd', 'pca'))
        self._plot_umap(
            embedding[:n_sims], embedding[n_sims:],
            list(self.conditions),
            "Moran's I + Activity (PCA)\n"
            "Linear projection of normalised features",
            'pca_morans_i_activity_features_wd', top_n_list,
            method_label='PCA')

    def plot_pca_morans_i_activity_features_wd_shaded(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot PCA of MI + Activity features with WD-shaded grey dots."""
        if not HAS_SKLEARN:
            print("  scikit-learn not available (install scikit-learn)")
            return
        ising_feat, exp_feat, n_sims = self._build_std_features(
            ['MoransI_all', 'Activity_all'])
        if n_sims == 0:
            return
        all_features = np.vstack([ising_feat, exp_feat])
        features_norm = self._prepare_umap_features(all_features)
        embedding = self._compute_embedding(
            features_norm, 'pca', cache_key=('mi_activity_wd', 'pca'))
        wd_color = self._min_wd_across_conditions()
        self._plot_embedding_generic(
            embedding[:n_sims], embedding[n_sims:],
            list(self.conditions),
            "PCA: Moran's I + Activity\nLinear projection of normalised features",
            'pca_morans_i_activity_features_wd_shaded', top_n_list,
            method_label='PCA', style='continuous', wd_for_color=wd_color)

    def plot_umap_morans_i_per_recording(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """
        Plot UMAP of Moran's I feature space with one point per recording.

        Each recording's trials are split using RecordingMetadata, 6 statistics
        are computed per recording, and each recording is plotted as a diamond
        marker colored by condition.
        """
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return

        rec_meta = self.results.get('RecordingMetadata', {})
        if not rec_meta:
            print("  RecordingMetadata not available — skipping per-recording UMAP")
            return

        ising_data = self.results.get('IsingData', {})
        comparison = self.results.get('Comparison', {})
        exp_stats = self.results.get('ExpStats', {})

        mi_all = ising_data.get('MoransI_all', [])
        if len(mi_all) == 0:
            print("  No Moran's I data for UMAP")
            return

        n_sims = len(mi_all)

        # Compute 6 statistics for each Ising simulation
        ising_features = np.zeros((n_sims, 6))
        for s in range(n_sims):
            mi_data = np.array(mi_all[s]) if isinstance(mi_all[s], (list, np.ndarray)) else np.array([])
            ising_features[s, :] = compute_6_statistics(mi_data)

        # Compute per-recording features
        rec_features_list = []
        rec_conditions = []
        for condition in self.conditions:
            if condition not in exp_stats or condition not in rec_meta:
                continue
            meta = rec_meta[condition]
            nTrials_per_rec = meta.get('nTrials_per_recording', None)
            if nTrials_per_rec is None or len(nTrials_per_rec) == 0:
                continue

            mi_trials = exp_stats[condition].get('MoransI_trials', None)
            if mi_trials is None:
                continue

            chunks = split_trials_by_recording(mi_trials, nTrials_per_rec)
            for chunk in chunks:
                rec_features_list.append(compute_6_statistics(chunk.ravel()))
                rec_conditions.append(condition)

        if len(rec_features_list) == 0:
            print("  No per-recording features computed — skipping")
            return

        rec_features = np.array(rec_features_list)
        n_recs = len(rec_features_list)

        # Combine: [ising; recordings]
        all_features = np.vstack([ising_features, rec_features])

        # Handle NaN values
        for col in range(all_features.shape[1]):
            col_data = all_features[:, col]
            nan_mask = np.isnan(col_data)
            if np.any(nan_mask) and not np.all(nan_mask):
                col_data[nan_mask] = np.nanmean(col_data)
                all_features[:, col] = col_data

        # Z-score normalize
        features_norm = (all_features - np.nanmean(all_features, axis=0)) / (np.nanstd(all_features, axis=0) + 1e-10)
        features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Run UMAP
        reducer = umap.UMAP(n_neighbors=min(199, len(features_norm) - 1), min_dist=0.3, random_state=42)
        embedding_all = reducer.fit_transform(features_norm)

        embedding_ising = embedding_all[:n_sims, :]
        embedding_recs = embedding_all[n_sims:, :]

        for top_n in top_n_list:
            # --- Full figure ---
            fig, ax = plt.subplots(figsize=(10, 8))

            ax.scatter(embedding_ising[:, 0], embedding_ising[:, 1],
                      c='lightgray', alpha=0.4, s=100)

            # Top-N Ising matches per condition
            for condition in self.conditions:
                if condition in comparison:
                    best_idx = self._get_best_idx(condition)
                    if len(best_idx) > 0:
                        n_top = min(top_n, len(best_idx))
                        top_idx = best_idx[:n_top]
                        ax.scatter(embedding_ising[top_idx, 0], embedding_ising[top_idx, 1],
                                  c=[self._get_color(condition)], s=100, edgecolors='black',
                                  linewidth=1.5, label=f'{condition} (Top {n_top} Ising)')

            # Per-recording diamonds
            for c_idx, condition in enumerate(self.conditions):
                mask = [i for i, c in enumerate(rec_conditions) if c == condition]
                if len(mask) > 0:
                    idx = np.array(mask)
                    ax.scatter(embedding_recs[idx, 0], embedding_recs[idx, 1],
                              c=[self._get_color(condition)], s=200, marker='D',
                              edgecolors='black', linewidth=2,
                              label=f'{condition} (n={len(mask)} recs)')

            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(f"Moran's I Feature Space — Per Recording (Top {top_n} Matches)\n"
                         f"6 features: mean, std, median, skewness, kurtosis, IQR")
            ax.legend(loc='best', fontsize=8)
            fig.tight_layout()

            suffix = f'_top{top_n}' if top_n != 3 else ''
            self._save_figure(fig, f'umap_morans_i_per_recording{suffix}')

            # --- Zoomed figure ---
            poi_coords = list(embedding_recs)
            for condition in self.conditions:
                if condition in comparison:
                    best_idx = self._get_best_idx(condition)
                    if len(best_idx) > 0:
                        n_top = min(top_n, len(best_idx))
                        for idx in best_idx[:n_top]:
                            poi_coords.append(embedding_ising[idx])
            poi_coords = np.array(poi_coords)

            x_min, x_max = poi_coords[:, 0].min(), poi_coords[:, 0].max()
            y_min, y_max = poi_coords[:, 1].min(), poi_coords[:, 1].max()
            x_pad = 0.2 * (x_max - x_min) if x_max > x_min else 1.0
            y_pad = 0.2 * (y_max - y_min) if y_max > y_min else 1.0

            fig_z, ax_z = plt.subplots(figsize=(10, 8))

            ax_z.scatter(embedding_ising[:, 0], embedding_ising[:, 1],
                        c='lightgray', alpha=0.4, s=100)

            for condition in self.conditions:
                if condition in comparison:
                    best_idx = self._get_best_idx(condition)
                    if len(best_idx) > 0:
                        n_top = min(top_n, len(best_idx))
                        top_idx = best_idx[:n_top]
                        ax_z.scatter(embedding_ising[top_idx, 0], embedding_ising[top_idx, 1],
                                    c=[self._get_color(condition)], s=100, edgecolors='black',
                                    linewidth=1.5, label=f'{condition} (Top {n_top} Ising)')

            for c_idx, condition in enumerate(self.conditions):
                mask = [i for i, c in enumerate(rec_conditions) if c == condition]
                if len(mask) > 0:
                    idx = np.array(mask)
                    ax_z.scatter(embedding_recs[idx, 0], embedding_recs[idx, 1],
                                c=[self._get_color(condition)], s=200, marker='D',
                                edgecolors='black', linewidth=2,
                                label=f'{condition} (n={len(mask)} recs)')

            ax_z.set_xlim(x_min - x_pad, x_max + x_pad)
            ax_z.set_ylim(y_min - y_pad, y_max + y_pad)
            ax_z.set_xlabel('UMAP 1')
            ax_z.set_ylabel('UMAP 2')
            ax_z.set_title(f"Moran's I Feature Space — Per Recording Zoomed (Top {top_n} Matches)\n"
                           f"6 features: mean, std, median, skewness, kurtosis, IQR")
            ax_z.legend(loc='best', fontsize=8)
            fig_z.tight_layout()

            self._save_figure(fig_z, f'umap_morans_i_per_recording{suffix}_zoomed')

    def plot_umap_morans_i_per_animal(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """
        Plot UMAP of Moran's I feature space with one point per animal-condition pair.

        Recordings from the same animal within a condition are pooled, then 6
        statistics are computed. Each animal gets a unique marker shape.
        """
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return

        rec_meta = self.results.get('RecordingMetadata', {})
        if not rec_meta:
            print("  RecordingMetadata not available — skipping per-animal UMAP")
            return

        ising_data = self.results.get('IsingData', {})
        comparison = self.results.get('Comparison', {})
        exp_stats = self.results.get('ExpStats', {})

        mi_all = ising_data.get('MoransI_all', [])
        if len(mi_all) == 0:
            print("  No Moran's I data for UMAP")
            return

        n_sims = len(mi_all)

        # Compute 6 statistics for each Ising simulation
        ising_features = np.zeros((n_sims, 6))
        for s in range(n_sims):
            mi_data = np.array(mi_all[s]) if isinstance(mi_all[s], (list, np.ndarray)) else np.array([])
            ising_features[s, :] = compute_6_statistics(mi_data)

        # Compute per-animal features: group recordings by animal within each condition
        from collections import defaultdict

        animal_features_list = []
        animal_conditions = []
        animal_names_list = []

        for condition in self.conditions:
            if condition not in exp_stats or condition not in rec_meta:
                continue
            meta = rec_meta[condition]
            nTrials_per_rec = meta.get('nTrials_per_recording', None)
            animal_names = meta.get('animal_names', None)
            if nTrials_per_rec is None or animal_names is None:
                continue

            mi_trials = exp_stats[condition].get('MoransI_trials', None)
            if mi_trials is None:
                continue

            chunks = split_trials_by_recording(mi_trials, nTrials_per_rec)

            # Group chunks by animal name
            animal_chunks = defaultdict(list)
            for chunk, name in zip(chunks, animal_names):
                animal_chunks[name].append(chunk)

            # Pool all recordings per animal and compute stats
            for name, chunk_list in animal_chunks.items():
                pooled = np.concatenate(chunk_list, axis=0)
                animal_features_list.append(compute_6_statistics(pooled.ravel()))
                animal_conditions.append(condition)
                animal_names_list.append(name)

        if len(animal_features_list) == 0:
            print("  No per-animal features computed — skipping")
            return

        animal_features = np.array(animal_features_list)
        n_animals = len(animal_features_list)

        # Build marker map: unique marker per animal
        unique_animals = sorted(set(animal_names_list))
        marker_shapes = ['o', 's', '^', 'v', '<', '>', 'p', 'h', '8', 'D', '*', 'X', 'P']
        animal_marker_map = {name: marker_shapes[i % len(marker_shapes)]
                             for i, name in enumerate(unique_animals)}

        # Combine: [ising; animals]
        all_features = np.vstack([ising_features, animal_features])

        # Handle NaN
        for col in range(all_features.shape[1]):
            col_data = all_features[:, col]
            nan_mask = np.isnan(col_data)
            if np.any(nan_mask) and not np.all(nan_mask):
                col_data[nan_mask] = np.nanmean(col_data)
                all_features[:, col] = col_data

        # Z-score normalize
        features_norm = (all_features - np.nanmean(all_features, axis=0)) / (np.nanstd(all_features, axis=0) + 1e-10)
        features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Run UMAP
        reducer = umap.UMAP(n_neighbors=min(199, len(features_norm) - 1), min_dist=0.3, random_state=42)
        embedding_all = reducer.fit_transform(features_norm)

        embedding_ising = embedding_all[:n_sims, :]
        embedding_animals = embedding_all[n_sims:, :]

        for top_n in top_n_list:
            # --- Full figure ---
            fig, ax = plt.subplots(figsize=(10, 8))

            ax.scatter(embedding_ising[:, 0], embedding_ising[:, 1],
                      c='lightgray', alpha=0.4, s=100)

            # Top-N Ising matches
            for condition in self.conditions:
                if condition in comparison:
                    best_idx = self._get_best_idx(condition)
                    if len(best_idx) > 0:
                        n_top = min(top_n, len(best_idx))
                        top_idx = best_idx[:n_top]
                        ax.scatter(embedding_ising[top_idx, 0], embedding_ising[top_idx, 1],
                                  c=[self._get_color(condition)], s=100, edgecolors='black',
                                  linewidth=1.5, label=f'{condition} (Top {n_top} Ising)')

            # Per-animal markers: unique shape per animal, color by condition
            plotted_animals = set()
            for i, (cond, name) in enumerate(zip(animal_conditions, animal_names_list)):
                marker = animal_marker_map[name]
                # Only add animal name to legend once
                lbl = f'{name}' if name not in plotted_animals else None
                ax.scatter(embedding_animals[i, 0], embedding_animals[i, 1],
                          c=[self._get_color(cond)], s=250, marker=marker,
                          edgecolors='black', linewidth=2, label=lbl)
                plotted_animals.add(name)

            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(f"Moran's I Feature Space — Per Animal (Top {top_n} Matches)\n"
                         f"6 features | {n_animals} animal-condition points | "
                         f"marker shape = animal identity")
            ax.legend(loc='best', fontsize=7, ncol=2)
            fig.tight_layout()

            suffix = f'_top{top_n}' if top_n != 3 else ''
            self._save_figure(fig, f'umap_morans_i_per_animal{suffix}')

            # --- Zoomed figure ---
            poi_coords = list(embedding_animals)
            for condition in self.conditions:
                if condition in comparison:
                    best_idx = self._get_best_idx(condition)
                    if len(best_idx) > 0:
                        n_top = min(top_n, len(best_idx))
                        for idx in best_idx[:n_top]:
                            poi_coords.append(embedding_ising[idx])
            poi_coords = np.array(poi_coords)

            x_min, x_max = poi_coords[:, 0].min(), poi_coords[:, 0].max()
            y_min, y_max = poi_coords[:, 1].min(), poi_coords[:, 1].max()
            x_pad = 0.2 * (x_max - x_min) if x_max > x_min else 1.0
            y_pad = 0.2 * (y_max - y_min) if y_max > y_min else 1.0

            fig_z, ax_z = plt.subplots(figsize=(10, 8))

            ax_z.scatter(embedding_ising[:, 0], embedding_ising[:, 1],
                        c='lightgray', alpha=0.4, s=100)

            for condition in self.conditions:
                if condition in comparison:
                    best_idx = self._get_best_idx(condition)
                    if len(best_idx) > 0:
                        n_top = min(top_n, len(best_idx))
                        top_idx = best_idx[:n_top]
                        ax_z.scatter(embedding_ising[top_idx, 0], embedding_ising[top_idx, 1],
                                    c=[self._get_color(condition)], s=100, edgecolors='black',
                                    linewidth=1.5, label=f'{condition} (Top {n_top} Ising)')

            plotted_animals_z = set()
            for i, (cond, name) in enumerate(zip(animal_conditions, animal_names_list)):
                marker = animal_marker_map[name]
                lbl = f'{name}' if name not in plotted_animals_z else None
                ax_z.scatter(embedding_animals[i, 0], embedding_animals[i, 1],
                            c=[self._get_color(cond)], s=250, marker=marker,
                            edgecolors='black', linewidth=2, label=lbl)
                plotted_animals_z.add(name)

            ax_z.set_xlim(x_min - x_pad, x_max + x_pad)
            ax_z.set_ylim(y_min - y_pad, y_max + y_pad)
            ax_z.set_xlabel('UMAP 1')
            ax_z.set_ylabel('UMAP 2')
            ax_z.set_title(f"Moran's I Feature Space — Per Animal Zoomed (Top {top_n} Matches)\n"
                           f"6 features | {n_animals} animal-condition points | "
                           f"marker shape = animal identity")
            ax_z.legend(loc='best', fontsize=7, ncol=2)
            fig_z.tight_layout()

            self._save_figure(fig_z, f'umap_morans_i_per_animal{suffix}_zoomed')

    def plot_umap_morans_i_activity_features(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot UMAP of Moran's I + Activity feature space (12 features)."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return

        mi_feat = self._compute_6stat_ising_features('MoransI_all')
        act_feat = self._compute_6stat_ising_features('Activity_all')
        if mi_feat.shape[0] == 0:
            print("  No Moran's I data for UMAP")
            return
        n_sims = mi_feat.shape[0]
        # Pad Activity features if shorter
        if act_feat.shape[0] < n_sims:
            act_feat = np.vstack([act_feat, np.zeros((n_sims - act_feat.shape[0], 6))])
        ising_features = np.hstack([mi_feat, act_feat[:n_sims]])

        mi_exp = self._compute_6stat_exp_features('MoransI_all')
        act_exp = self._compute_6stat_exp_features('Activity_all')
        exp_features = np.hstack([mi_exp, act_exp])

        all_features = np.vstack([ising_features, exp_features])
        features_norm = self._prepare_umap_features(all_features)

        reducer = umap.UMAP(n_neighbors=min(199, len(features_norm) - 1),
                            min_dist=0.3, random_state=42)
        embedding_all = reducer.fit_transform(features_norm)

        self._plot_umap(
            embedding_all[:n_sims], embedding_all[n_sims:],
            list(self.conditions),
            "Moran's I + Activity Feature Space\n12 features",
            'umap_morans_i_activity_features', top_n_list)

    def plot_umap_morans_i_activity_features_shaded(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot UMAP of Moran's I + Activity features with WD-shaded dots."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return
        ising_feat, exp_feat, n_sims = self._build_std_features(
            ['MoransI_all', 'Activity_all'])
        if n_sims == 0:
            return
        all_features = np.vstack([ising_feat, exp_feat])
        features_norm = self._prepare_umap_features(all_features)
        embedding = self._compute_embedding(features_norm, 'umap',
                                            cache_key=('mi_activity', 'umap'))
        wd_color = self._min_wd_across_conditions()
        self._plot_embedding_generic(
            embedding[:n_sims], embedding[n_sims:],
            list(self.conditions),
            "UMAP: Moran's I + Activity Feature Space\n12 features",
            'umap_morans_i_activity_features_shaded', top_n_list,
            method_label='UMAP', style='continuous', wd_for_color=wd_color)

    def plot_umap_activity_features(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot UMAP of activity feature space (6 features)."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return

        ising_features = self._compute_6stat_ising_features('Activity_all')
        if ising_features.shape[0] == 0:
            print("  No activity data for UMAP")
            return
        n_sims = ising_features.shape[0]

        exp_features = self._compute_6stat_exp_features('Activity_all')

        all_features = np.vstack([ising_features, exp_features])
        features_norm = self._prepare_umap_features(all_features)

        reducer = umap.UMAP(n_neighbors=min(199, len(features_norm) - 1),
                            min_dist=0.3, random_state=42)
        embedding_all = reducer.fit_transform(features_norm)

        self._plot_umap(
            embedding_all[:n_sims], embedding_all[n_sims:],
            list(self.conditions),
            'Activity Feature Space\n6 features: mean, std, median, skewness, kurtosis, IQR',
            'umap_activity_features', top_n_list)

    def plot_umap_spatial_persistence_features(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot UMAP of spatial+persistence features (18 features)."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return

        mi_feat = self._compute_6stat_ising_features('MoransI_all')
        if mi_feat.shape[0] == 0:
            print("  No data for spatial+persistence UMAP")
            return
        n_sims = mi_feat.shape[0]
        act_feat = self._compute_6stat_ising_features('Activity_all')
        blob_feat = self._compute_6stat_ising_features('BlobPersistence_lifetimes')
        # Pad shorter feature arrays
        for arr in [act_feat, blob_feat]:
            if arr.shape[0] < n_sims:
                pad = np.zeros((n_sims - arr.shape[0], 6))
                arr = np.vstack([arr, pad])
        ising_features = np.hstack([mi_feat, act_feat[:n_sims], blob_feat[:n_sims]])

        mi_exp = self._compute_6stat_exp_features('MoransI_all')
        act_exp = self._compute_6stat_exp_features('Activity_all')
        blob_exp = self._compute_6stat_exp_features('BlobPersistence_lifetimes')

        # Fill missing blob data from Global stats
        exp_stats = self.results.get('ExpStats', {})
        if 'Global' in exp_stats and 'BlobPersistence' in exp_stats['Global']:
            global_blob = exp_stats['Global']['BlobPersistence']
            if 'lifetimes' in global_blob:
                global_feat = compute_6_statistics(global_blob['lifetimes'])
                for c in range(blob_exp.shape[0]):
                    if np.all(np.isnan(blob_exp[c])):
                        blob_exp[c] = global_feat

        exp_features = np.hstack([mi_exp, act_exp, blob_exp])

        all_features = np.vstack([ising_features, exp_features])
        features_norm = self._prepare_umap_features(all_features)

        reducer = umap.UMAP(n_neighbors=min(199, len(features_norm) - 1),
                            min_dist=0.3, random_state=42)
        embedding_all = reducer.fit_transform(features_norm)

        self._plot_umap(
            embedding_all[:n_sims], embedding_all[n_sims:],
            list(self.conditions),
            'Spatial+Persistence Feature Space\n18 features: 6 MI + 6 Activity + 6 BlobPersistence',
            'umap_spatial_persistence_features', top_n_list)

    def plot_umap_spatial_persistence_features_wd(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """Plot UMAP using WD-based distances for spatial+persistence features."""
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return
        embedding_all, n_sims = self._get_wd_embedding(
            ['MoransI_all', 'Activity_all', 'BlobPersistence_lifetimes'],
            'umap', ('spatial_persistence_wd', 'umap'))
        if embedding_all is None:
            print("  No data for UMAP")
            return
        self._plot_umap(
            embedding_all[:n_sims], embedding_all[n_sims:],
            list(self.conditions),
            "Spatial+Persistence (WD Embedding)\nQuantile L1 distances",
            'umap_spatial_persistence_features_wd', top_n_list)

    def plot_umap_blob_persistence_features(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
        """
        Plot UMAP of blob persistence feature space with experimental embedding.

        Matches MATLAB Figure 7e: 6 features (blob lifetime statistics only).

        Parameters
        ----------
        top_n_list : List[int]
            List of top-N values to generate figures for
        """
        if not HAS_UMAP:
            print("  UMAP not available (install umap-learn)")
            return

        ising_data = self.results.get('IsingData', {})
        comparison = self.results.get('Comparison', {})
        exp_stats = self.results.get('ExpStats', {})

        blob_lifetimes_all = ising_data.get('BlobPersistence_lifetimes', [])

        if len(blob_lifetimes_all) == 0:
            print("  No blob persistence data for UMAP")
            return

        n_sims = len(blob_lifetimes_all)
        n_conds = len(self.conditions)

        ising_features = self._compute_6stat_ising_features('BlobPersistence_lifetimes')
        n_sims = ising_features.shape[0]

        exp_features = self._compute_6stat_exp_features('BlobPersistence_lifetimes')

        # Fill from Global stats (matching MATLAB behavior)
        global_blob_features = np.full(6, np.nan)
        if 'Global' in exp_stats:
            global_stats = exp_stats['Global']
            if isinstance(global_stats, dict) and 'BlobPersistence' in global_stats:
                blob_persist = global_stats['BlobPersistence']
                if isinstance(blob_persist, dict) and 'lifetimes' in blob_persist:
                    global_blob_features = compute_6_statistics(blob_persist['lifetimes'])
        for c in range(n_conds):
            if np.all(np.isnan(exp_features[c])):
                exp_features[c] = global_blob_features

        all_features = np.vstack([ising_features, exp_features])
        valid_rows = ~np.all(np.isnan(all_features), axis=1)
        if np.sum(valid_rows) < 10:
            print("  Insufficient valid data for blob persistence UMAP")
            return

        features_norm = self._prepare_umap_features(all_features)

        reducer = umap.UMAP(n_neighbors=min(199, len(features_norm) - 1),
                            min_dist=0.3, random_state=42)
        embedding_all = reducer.fit_transform(features_norm)

        self._plot_umap(
            embedding_all[:n_sims], embedding_all[n_sims:],
            list(self.conditions),
            'BlobPersistence Feature Space\n6 features: blob lifetime statistics',
            'umap_blob_persistence_features', top_n_list)

    # =========================================================================
    # MDE EMBEDDING METHODS (disabled — UMAP preferred)
    # =========================================================================

    # def plot_mde_morans_i_only(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding of Moran's I feature space (6 features)."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     ising_feat, exp_feat, n_sims = self._build_std_features(['MoransI_all'])
    #     if n_sims == 0:
    #         return
    #     all_features = np.vstack([ising_feat, exp_feat])
    #     features_norm = self._prepare_umap_features(all_features)
    #     embedding = self._compute_embedding(features_norm, 'mde',
    #                                         cache_key=('morans_i_only', 'mde'))
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_generic(
    #         embedding[:n_sims], embedding[n_sims:],
    #         list(self.conditions),
    #         "MDE: Moran's I Feature Space\n6 features",
    #         'mde_morans_i_only', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color)

    # def plot_mde_morans_i_only_cloud(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding of Moran's I feature space with convex hull clouds."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     ising_feat, exp_feat, n_sims = self._build_std_features(['MoransI_all'])
    #     if n_sims == 0:
    #         return
    #     all_features = np.vstack([ising_feat, exp_feat])
    #     features_norm = self._prepare_umap_features(all_features)
    #     embedding = self._compute_embedding(features_norm, 'mde',
    #                                         cache_key=('morans_i_only', 'mde'))
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_generic(
    #         embedding[:n_sims], embedding[n_sims:],
    #         list(self.conditions),
    #         "MDE: Moran's I Feature Space (Cloud)\n6 features",
    #         'mde_morans_i_only_cloud', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color,
    #         cloud=True)

    # def plot_mde_morans_i_only_wd(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding using WD-based distances (quantile L1)."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     embedding, n_sims = self._get_wd_embedding(
    #         'MoransI_all', 'mde', ('morans_i_only_wd', 'mde'))
    #     if embedding is None:
    #         return
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_generic(
    #         embedding[:n_sims], embedding[n_sims:],
    #         list(self.conditions),
    #         "MDE: Moran's I (WD Embedding)\nQuantile L1 distances",
    #         'mde_morans_i_only_wd', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color)

    # def plot_mde_morans_i_only_wd_cloud(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding using WD-based distances with convex hull clouds."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     embedding, n_sims = self._get_wd_embedding(
    #         'MoransI_all', 'mde', ('morans_i_only_wd', 'mde'))
    #     if embedding is None:
    #         return
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_generic(
    #         embedding[:n_sims], embedding[n_sims:],
    #         list(self.conditions),
    #         "MDE: Moran's I (WD Embedding, Cloud)\nQuantile L1 distances",
    #         'mde_morans_i_only_wd_cloud', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color,
    #         cloud=True)

    # def plot_mde_activity_features_wd(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding using WD-based distances for Activity only."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     embedding, n_sims = self._get_wd_embedding(
    #         'Activity_all', 'mde', ('activity_wd', 'mde'))
    #     if embedding is None:
    #         return
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_generic(
    #         embedding[:n_sims], embedding[n_sims:],
    #         list(self.conditions),
    #         "MDE: Activity (WD Embedding)\nQuantile L1 distances",
    #         'mde_activity_features_wd', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color)

    # def plot_mde_morans_i_activity_features_wd(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding using WD-based distances for Moran's I + Activity."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     embedding, n_sims = self._get_wd_embedding(
    #         ['MoransI_all', 'Activity_all'], 'mde', ('mi_activity_wd', 'mde'))
    #     if embedding is None:
    #         return
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_generic(
    #         embedding[:n_sims], embedding[n_sims:],
    #         list(self.conditions),
    #         "MDE: Moran's I + Activity (WD Embedding)\nQuantile L1 distances",
    #         'mde_morans_i_activity_features_wd', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color)

    # def plot_mde_morans_i_activity_features_wd_cloud(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding using WD-based distances for MI + Activity with clouds."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     embedding, n_sims = self._get_wd_embedding(
    #         ['MoransI_all', 'Activity_all'], 'mde', ('mi_activity_wd', 'mde'))
    #     if embedding is None:
    #         return
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_generic(
    #         embedding[:n_sims], embedding[n_sims:],
    #         list(self.conditions),
    #         "MDE: Moran's I + Activity (WD Embedding, Cloud)\nQuantile L1 distances",
    #         'mde_morans_i_activity_features_wd_cloud', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color,
    #         cloud=True)

    # def plot_mde_morans_i_per_recording(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding of Moran's I with per-recording points."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     ising_feat, rec_feat, rec_labels, n_sims = \
    #         self._build_per_recording_features('MoransI_all')
    #     if n_sims == 0 or len(rec_labels) == 0:
    #         print("  No per-recording features for MDE")
    #         return
    #     all_features = np.vstack([ising_feat, rec_feat])
    #     features_norm = self._prepare_umap_features(all_features)
    #     embedding = self._compute_embedding(features_norm, 'mde')
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_per_recording(
    #         embedding[:n_sims], embedding[n_sims:],
    #         rec_labels,
    #         "MDE: Moran's I — Per Recording",
    #         'mde_morans_i_per_recording', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color)

    # def plot_mde_morans_i_per_animal(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding of Moran's I with per-animal points."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     ising_feat, animal_feat, animal_info, animal_markers, n_sims = \
    #         self._build_per_animal_features('MoransI_all')
    #     if n_sims == 0 or len(animal_info) == 0:
    #         print("  No per-animal features for MDE")
    #         return
    #     all_features = np.vstack([ising_feat, animal_feat])
    #     features_norm = self._prepare_umap_features(all_features)
    #     embedding = self._compute_embedding(features_norm, 'mde')
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_per_animal(
    #         embedding[:n_sims], embedding[n_sims:],
    #         animal_info, animal_markers,
    #         "MDE: Moran's I — Per Animal",
    #         'mde_morans_i_per_animal', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color)

    # def plot_mde_morans_i_activity_features(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding of Moran's I + Activity features (12 features)."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     ising_feat, exp_feat, n_sims = self._build_std_features(
    #         ['MoransI_all', 'Activity_all'])
    #     if n_sims == 0:
    #         return
    #     all_features = np.vstack([ising_feat, exp_feat])
    #     features_norm = self._prepare_umap_features(all_features)
    #     embedding = self._compute_embedding(features_norm, 'mde',
    #                                         cache_key=('mi_activity', 'mde'))
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_generic(
    #         embedding[:n_sims], embedding[n_sims:],
    #         list(self.conditions),
    #         "MDE: Moran's I + Activity Feature Space\n12 features",
    #         'mde_morans_i_activity_features', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color)

    # def plot_mde_morans_i_activity_features_cloud(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding of Moran's I + Activity features with convex hull clouds."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     ising_feat, exp_feat, n_sims = self._build_std_features(
    #         ['MoransI_all', 'Activity_all'])
    #     if n_sims == 0:
    #         return
    #     all_features = np.vstack([ising_feat, exp_feat])
    #     features_norm = self._prepare_umap_features(all_features)
    #     embedding = self._compute_embedding(features_norm, 'mde',
    #                                         cache_key=('mi_activity', 'mde'))
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_generic(
    #         embedding[:n_sims], embedding[n_sims:],
    #         list(self.conditions),
    #         "MDE: Moran's I + Activity Feature Space (Cloud)\n12 features",
    #         'mde_morans_i_activity_features_cloud', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color,
    #         cloud=True)

    # def plot_mde_activity_features(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding of Activity feature space (6 features)."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     ising_feat, exp_feat, n_sims = self._build_std_features(['Activity_all'])
    #     if n_sims == 0:
    #         return
    #     all_features = np.vstack([ising_feat, exp_feat])
    #     features_norm = self._prepare_umap_features(all_features)
    #     embedding = self._compute_embedding(features_norm, 'mde',
    #                                         cache_key=('activity', 'mde'))
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_generic(
    #         embedding[:n_sims], embedding[n_sims:],
    #         list(self.conditions),
    #         'MDE: Activity Feature Space\n6 features',
    #         'mde_activity_features', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color)

    # def plot_mde_spatial_persistence_features(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding of spatial+persistence features (18 features)."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     ising_feat, exp_feat, n_sims = self._build_std_features(
    #         ['MoransI_all', 'Activity_all', 'BlobPersistence_lifetimes'])
    #     if n_sims == 0:
    #         return
    #     all_features = np.vstack([ising_feat, exp_feat])
    #     features_norm = self._prepare_umap_features(all_features)
    #     embedding = self._compute_embedding(features_norm, 'mde',
    #                                         cache_key=('spatial_persistence', 'mde'))
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_generic(
    #         embedding[:n_sims], embedding[n_sims:],
    #         list(self.conditions),
    #         'MDE: Spatial+Persistence Feature Space\n18 features',
    #         'mde_spatial_persistence_features', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color)

    # def plot_mde_spatial_persistence_features_wd(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding using WD-based distances for spatial+persistence features."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     embedding, n_sims = self._get_wd_embedding(
    #         ['MoransI_all', 'Activity_all', 'BlobPersistence_lifetimes'],
    #         'mde', ('spatial_persistence_wd', 'mde'))
    #     if embedding is None:
    #         return
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_generic(
    #         embedding[:n_sims], embedding[n_sims:],
    #         list(self.conditions),
    #         "MDE: Spatial+Persistence (WD Embedding)\nQuantile L1 distances",
    #         'mde_spatial_persistence_features_wd', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color)

    # def plot_mde_blob_persistence_features(self, top_n_list: List[int] = [3, 10, 20, 50, 100]):
    #     """MDE embedding of blob persistence features (6 features)."""
    #     if not HAS_MDE:
    #         print("  MDE not available (install pymde)")
    #         return
    #     ising_feat, exp_feat, n_sims = self._build_std_features(
    #         ['BlobPersistence_lifetimes'])
    #     if n_sims == 0:
    #         return
    #     all_features = np.vstack([ising_feat, exp_feat])
    #     valid_rows = ~np.all(np.isnan(all_features), axis=1)
    #     if np.sum(valid_rows) < 10:
    #         print("  Insufficient valid data for blob persistence MDE")
    #         return
    #     features_norm = self._prepare_umap_features(all_features)
    #     embedding = self._compute_embedding(features_norm, 'mde',
    #                                         cache_key=('blob_persistence', 'mde'))
    #     wd_color = self._min_wd_across_conditions()
    #     self._plot_embedding_generic(
    #         embedding[:n_sims], embedding[n_sims:],
    #         list(self.conditions),
    #         'MDE: BlobPersistence Feature Space\n6 features',
    #         'mde_blob_persistence_features', top_n_list,
    #         method_label='MDE', style='continuous', wd_for_color=wd_color)


    # =========================================================================
    # MATCH SPECIFICITY METHODS
    # =========================================================================

    def plot_match_specificity_bar(self, top_n: int = 10):
        """Bar chart: self-distance vs cross-distance using WD.

        For each condition, compare mean Wasserstein distance between
        experimental distribution and its own top-N Ising matches vs other
        conditions' top-N matches.

        Produces two figures:
        - match_specificity_bar.png      (linear y-axis)
        - match_specificity_bar_log.png  (log y-axis — compresses outliers)
        """
        comparison = self.results.get('Comparison', {})
        feature_spaces = [
            ('MoransI', ['MoransI_all']),
            ('Activity', ['Activity_all']),
            ('Spatial+Persistence', ['MoransI_all', 'Activity_all', 'BlobPersistence_lifetimes']),
        ]

        n_spaces = len(feature_spaces)
        n_cols = 3

        # --- Step 1: compute WD distances, cache per feature space ---
        all_results = {}
        for fs_idx, (fs_name, metric_keys) in enumerate(feature_spaces):
            dist_matrix, n_sims = self._build_wd_distance_matrix(metric_keys)
            if n_sims == 0:
                continue

            cond_list = [c for c in self.conditions if c in comparison]
            self_means, self_sems = [], []
            cross_means, cross_sems = [], []

            for cond in cond_list:
                exp_idx = n_sims + list(self.conditions).index(cond)
                best_idx = self._get_best_idx(cond)
                own_top = best_idx[:min(top_n, len(best_idx))]

                # Self distance: WD from exp to its own top-N Ising matches
                if len(own_top) > 0:
                    dists_self = dist_matrix[exp_idx, own_top]
                    self_means.append(np.mean(dists_self))
                    self_sems.append(np.std(dists_self) / max(np.sqrt(len(dists_self)), 1))
                else:
                    self_means.append(np.nan)
                    self_sems.append(0)

                # Cross distance: WD from exp to other conditions' top-N matches
                other_dists = []
                for other_cond in cond_list:
                    if other_cond == cond:
                        continue
                    other_idx = self._get_best_idx(other_cond)
                    other_top = other_idx[:min(top_n, len(other_idx))]
                    if len(other_top) > 0:
                        other_dists.extend(dist_matrix[exp_idx, other_top])
                if len(other_dists) > 0:
                    cross_means.append(np.mean(other_dists))
                    cross_sems.append(np.std(other_dists) / max(np.sqrt(len(other_dists)), 1))
                else:
                    cross_means.append(np.nan)
                    cross_sems.append(0)

            all_results[fs_idx] = {
                'cond_list': cond_list,
                'colors': [self._get_color(c) for c in cond_list],
                'self_means': self_means, 'self_sems': self_sems,
                'cross_means': cross_means, 'cross_sems': cross_sems,
            }

        # --- Step 2: bar charts (linear + log scale) ---
        for scale_mode in ['linear', 'log']:
            fig, axes = plt.subplots(1, n_cols, figsize=(14, 4))

            for fs_idx, (fs_name, _) in enumerate(feature_spaces):
                ax = axes[fs_idx]
                if fs_idx not in all_results:
                    ax.set_visible(False)
                    continue
                r = all_results[fs_idx]
                x_pos = np.arange(len(r['cond_list']))
                bar_w = 0.35
                ax.bar(x_pos - bar_w / 2, r['self_means'], bar_w,
                       yerr=r['self_sems'], color=r['colors'],
                       edgecolor='black', linewidth=0.5,
                       label='To own matches')
                ax.bar(x_pos + bar_w / 2, r['cross_means'], bar_w,
                       yerr=r['cross_sems'], color='lightgray',
                       edgecolor='black', linewidth=0.5, hatch='//',
                       label='To other matches')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(r['cond_list'], rotation=30, ha='right')
                ax.set_ylabel('Mean WD')
                ax.set_title(fs_name, fontsize=8)
                if fs_idx == 0:
                    ax.legend(fontsize=6)
                if scale_mode == 'log':
                    ax.set_yscale('log')

            suffix = '_log' if scale_mode == 'log' else ''
            title_suffix = ' (Log Scale)' if scale_mode == 'log' else ''
            fig.suptitle(
                f'Match Specificity: Self vs Cross WD (Top {top_n}){title_suffix}',
                fontsize=11)
            fig.tight_layout()
            self._save_figure(fig, f'match_specificity_bar{suffix}')

    def plot_match_specificity_heatmap(self, top_n: int = 10):
        """Heatmap of pairwise distances between exp points and Ising matches.

        Produces two figures per feature space (dendrogram+heatmap and
        focused exp×ising companion) for three feature spaces:
        MoransI, MI+Activity, and Spatial+Persistence.
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        from matplotlib.colors import PowerNorm
        from matplotlib.patches import Rectangle

        comparison = self.results.get('Comparison', {})

        feature_spaces = [
            ('MoransI',             ['MoransI_all'],
             'moransi'),
            ('MI+Activity',         ['MoransI_all', 'Activity_all'],
             'mi_activity'),
            ('Spatial+Persistence', ['MoransI_all', 'Activity_all', 'BlobPersistence_lifetimes'],
             'spatial_persistence'),
        ]

        for fs_name, metric_keys, fs_suffix in feature_spaces:
            wd_matrix, n_sims = self._build_wd_distance_matrix(metric_keys)
            if n_sims == 0:
                continue

            cond_list = [c for c in self.conditions if c in comparison]
            if len(cond_list) == 0:
                continue

            # Build labels + index references into wd_matrix
            labels = []
            point_indices = []  # int → single point, array → group (mean WD)
            for cond in cond_list:
                c_idx = list(self.conditions).index(cond)
                exp_idx = n_sims + c_idx
                labels.append(f'{cond}_exp')
                point_indices.append(exp_idx)

                best_idx = self._get_best_idx(cond)
                top_idx = best_idx[:min(top_n, len(best_idx))]
                if len(top_idx) > 0:
                    labels.append(f'{cond}_ising')
                    point_indices.append(np.array(top_idx))

            # Build pairwise mean-WD matrix between representative points
            n_pts = len(labels)
            dist_matrix = np.zeros((n_pts, n_pts))
            for i in range(n_pts):
                for j in range(n_pts):
                    idx_i = point_indices[i]
                    idx_j = point_indices[j]
                    if np.isscalar(idx_i) and np.isscalar(idx_j):
                        dist_matrix[i, j] = wd_matrix[idx_i, idx_j]
                    elif np.isscalar(idx_i):
                        dist_matrix[i, j] = np.mean(wd_matrix[idx_i, idx_j])
                    elif np.isscalar(idx_j):
                        dist_matrix[i, j] = np.mean(wd_matrix[idx_i, idx_j])
                    else:
                        dist_matrix[i, j] = np.mean(wd_matrix[np.ix_(idx_i, idx_j)])

            # -- Dendrogram + heatmap with GridSpec layout --
            fig = plt.figure(figsize=(14, 7))
            gs = GridSpec(2, 3, figure=fig,
                          width_ratios=[1, 0.12, 3],
                          height_ratios=[0.12, 1],
                          hspace=0.02, wspace=0.02)
            ax_dendro = fig.add_subplot(gs[1, 0])
            ax_strip_y = fig.add_subplot(gs[1, 1])
            ax_heat = fig.add_subplot(gs[1, 2])
            ax_strip_x = fig.add_subplot(gs[0, 2])

            # Linkage
            condensed = squareform(dist_matrix)
            Z = linkage(condensed, method='average')
            dendro = dendrogram(Z, labels=labels, ax=ax_dendro,
                                orientation='left', leaf_font_size=7)
            ax_dendro.set_xlabel('Distance')

            # Reorder by dendrogram
            order = dendro['leaves']
            ordered_dist = dist_matrix[np.ix_(order, order)]
            ordered_labels = [labels[i] for i in order]

            # PowerNorm color scale for visible contrast in low-distance range
            norm = PowerNorm(gamma=0.4, vmin=0, vmax=ordered_dist.max())
            im = ax_heat.imshow(ordered_dist, cmap='viridis_r', aspect='auto',
                                norm=norm)
            ax_heat.set_xticks(range(n_pts))
            ax_heat.set_yticks(range(n_pts))
            ax_heat.set_xticklabels(ordered_labels, rotation=45, ha='right',
                                    fontsize=6)
            ax_heat.set_yticklabels(ordered_labels, fontsize=6)

            # Condition color strips
            for idx, lab in enumerate(ordered_labels):
                cond_name = lab.rsplit('_', 1)[0]
                c = self._get_color(cond_name)
                ax_strip_y.add_patch(Rectangle((0, idx - 0.5), 1, 1,
                                               facecolor=c, edgecolor='black',
                                               linewidth=0.5))
                ax_strip_x.add_patch(Rectangle((idx - 0.5, 0), 1, 1,
                                               facecolor=c, edgecolor='black',
                                               linewidth=0.5))
            ax_strip_y.set_xlim(0, 1)
            ax_strip_y.set_ylim(-0.5, n_pts - 0.5)
            ax_strip_y.invert_yaxis()
            ax_strip_y.set_xticks([])
            ax_strip_y.set_yticks([])
            for sp in ax_strip_y.spines.values():
                sp.set_visible(False)

            ax_strip_x.set_xlim(-0.5, n_pts - 0.5)
            ax_strip_x.set_ylim(0, 1)
            ax_strip_x.set_xticks([])
            ax_strip_x.set_yticks([])
            for sp in ax_strip_x.spines.values():
                sp.set_visible(False)

            # Self-match cell highlighting
            for i, lab_i in enumerate(ordered_labels):
                cond_i, type_i = lab_i.rsplit('_', 1)
                for j, lab_j in enumerate(ordered_labels):
                    cond_j, type_j = lab_j.rsplit('_', 1)
                    if cond_i == cond_j and type_i != type_j:
                        rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         linewidth=2, edgecolor=self._get_color(cond_i),
                                         facecolor='none', zorder=3)
                        ax_heat.add_patch(rect)

            # Annotate with PowerNorm-aware text color
            normed_vals = norm(ordered_dist)
            for i in range(n_pts):
                for j in range(n_pts):
                    txt_color = 'white' if normed_vals[i, j] < 0.5 else 'black'
                    ax_heat.text(j, i, f'{ordered_dist[i, j]:.2f}',
                                 ha='center', va='center', fontsize=5,
                                 color=txt_color)

            plt.colorbar(im, ax=ax_heat, label='Mean WD', shrink=0.8)
            fig.suptitle(f'Match Specificity Heatmap (Top {top_n}, {fs_name})',
                         fontsize=11)
            fig.tight_layout()
            self._save_figure(fig, f'match_specificity_heatmap_{fs_suffix}')

            # -- Focused exp × ising submatrix companion figure --
            exp_indices = [i for i, l in enumerate(labels) if l.endswith('_exp')]
            ising_indices = [i for i, l in enumerate(labels) if l.endswith('_ising')]
            if len(exp_indices) == 0 or len(ising_indices) == 0:
                continue
            sub_dist = dist_matrix[np.ix_(exp_indices, ising_indices)]
            exp_conds = [labels[i].replace('_exp', '') for i in exp_indices]
            ising_conds = [labels[i].replace('_ising', '') for i in ising_indices]

            fig2, ax2 = plt.subplots(figsize=(6, 5))
            norm2 = PowerNorm(gamma=0.4, vmin=0, vmax=sub_dist.max())
            im2 = ax2.imshow(sub_dist, cmap='viridis_r', norm=norm2)
            ax2.set_xticks(range(len(ising_conds)))
            ax2.set_xticklabels(ising_conds, rotation=30, ha='right')
            ax2.set_yticks(range(len(exp_conds)))
            ax2.set_yticklabels(exp_conds)
            ax2.set_xlabel('Ising match centroid')
            ax2.set_ylabel('Experimental point')

            # Annotate + highlight diagonal
            normed2 = norm2(sub_dist)
            for i in range(len(exp_conds)):
                for j in range(len(ising_conds)):
                    txt_color = 'white' if normed2[i, j] < 0.5 else 'black'
                    ax2.text(j, i, f'{sub_dist[i, j]:.2f}', ha='center',
                             va='center', fontsize=8,
                             fontweight='bold' if i == j else 'normal',
                             color=txt_color)
                    if exp_conds[i] == ising_conds[j]:
                        rect = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2.5,
                                         edgecolor=self._get_color(exp_conds[i]),
                                         facecolor='none', zorder=3)
                        ax2.add_patch(rect)

            plt.colorbar(im2, ax=ax2, label='Mean WD', shrink=0.8)
            fig2.suptitle(f'Exp \u2192 Ising Match WD (Top {top_n}, {fs_name})')
            fig2.tight_layout()
            self._save_figure(fig2, f'match_specificity_heatmap_{fs_suffix}_expXising')

    def plot_activity_autocorrelation(self):
        """
        Plot activity temporal autocorrelation comparison.

        Matches MATLAB Figure 4a: Uses 2×4 layout (linear scale top, log scale bottom)
        with exponential fit overlays, fit range shading, R² annotations, and tau annotations.
        """
        exp_stats = self.results.get('ExpStats', {})
        ising_data = self.results.get('IsingData', {})
        comparison = self.results.get('Comparison', {})

        # Get fit_range from config (default to [1, 10] matching MATLAB)
        fit_range = self.config.get('autocorr', {}).get('fit_range', (1, 10))

        # Use 2×N layout: 2 rows (linear/log) x N conditions
        n_conditions = len(self.conditions)
        fig, axes = plt.subplots(2, n_conditions, figsize=(4*n_conditions, 8))
        if n_conditions == 1:
            axes = axes.reshape(2, 1)

        for i, condition in enumerate(self.conditions):
            ax_linear = axes[0, i]
            ax_log = axes[1, i]

            if condition not in exp_stats or condition not in comparison:
                ax_linear.set_visible(False)
                ax_log.set_visible(False)
                continue

            # Experimental activity ACF
            exp_acf = exp_stats[condition].get('Autocorr_acf_trial_averaged',
                       exp_stats[condition].get('Autocorr_acf', np.array([])))
            exp_tau = exp_stats[condition].get('Autocorr_tau_trial_averaged',
                       exp_stats[condition].get('Autocorr_tau', np.nan))
            exp_r2 = exp_stats[condition].get('Autocorr_r2_trial_averaged',
                       exp_stats[condition].get('Autocorr_r2', np.nan))

            if len(exp_acf) == 0:
                ax_linear.set_visible(False)
                ax_log.set_visible(False)
                continue

            max_lag = min(100, len(exp_acf))
            lags = np.arange(max_lag)

            # Best match Ising activity ACF
            ising_tau = np.nan
            ising_r2 = np.nan
            ising_acf = np.array([])
            best_idx = self._get_best_idx(condition)
            if len(best_idx) > 0:
                ising_acf_all = ising_data.get('Autocorr_acf', [])
                ising_tau_all = ising_data.get('Autocorr_tau', [])
                ising_r2_all = ising_data.get('Autocorr_r2', [])
                if len(ising_acf_all) > best_idx[0]:
                    ising_acf = np.array(ising_acf_all[best_idx[0]]) if isinstance(ising_acf_all, list) else np.array([])
                    ising_tau = ising_tau_all[best_idx[0]] if len(ising_tau_all) > best_idx[0] else np.nan
                    if hasattr(ising_r2_all, '__len__') and len(ising_r2_all) > best_idx[0]:
                        ising_r2 = ising_r2_all[best_idx[0]]

            # =============== LINEAR SCALE PLOT (top row) ===============
            # Shade fit range region
            ax_linear.axvspan(fit_range[0], fit_range[1], alpha=0.15, color='gray',
                              label=f'Fit range [{fit_range[0]}, {fit_range[1]}]')

            # Plot experimental ACF data
            ax_linear.plot(lags, exp_acf[:max_lag], color=self._get_color(condition), linewidth=2,
                    label=f'Exp data')

            # Plot exponential fit for experimental data
            if not np.isnan(exp_tau) and exp_tau > 0:
                exp_fit = np.exp(-lags / exp_tau)
                ax_linear.plot(lags, exp_fit, color=self._get_color(condition), linewidth=1.5,
                        linestyle=':', alpha=0.8, label=f'Exp fit (τ={exp_tau:.1f})')

            if len(ising_acf) > 0:
                max_lag_ising = min(max_lag, len(ising_acf))
                lags_ising = np.arange(max_lag_ising)
                ax_linear.plot(lags_ising, ising_acf[:max_lag_ising], color='gray', linewidth=2, linestyle='--',
                        label=f'Ising data')

                # Plot exponential fit for Ising data
                if not np.isnan(ising_tau) and ising_tau > 0:
                    ising_fit = np.exp(-lags_ising / ising_tau)
                    ax_linear.plot(lags_ising, ising_fit, color='dimgray', linewidth=1.5,
                            linestyle=':', alpha=0.8, label=f'Ising fit (τ={ising_tau:.1f})')

            ax_linear.set_xlabel('Lag (frames/MC)')
            ax_linear.set_ylabel('Autocorrelation')
            ax_linear.set_title(condition)
            ax_linear.legend(fontsize=6, loc='upper right')
            ax_linear.set_xlim(0, max_lag)
            ax_linear.axhline(0, color='black', linewidth=0.5, linestyle=':')

            # Add R² annotations
            r2_text = []
            if not np.isnan(exp_r2):
                r2_text.append(f'Exp R²={exp_r2:.3f}')
            if not np.isnan(ising_r2):
                r2_text.append(f'Ising R²={ising_r2:.3f}')
            if r2_text:
                ax_linear.text(0.98, 0.85, '\n'.join(r2_text),
                       transform=ax_linear.transAxes, fontsize=7, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

            # Add time scaling annotation
            if not np.isnan(exp_tau) and not np.isnan(ising_tau) and ising_tau > 0:
                scale_factor = exp_tau / ising_tau
                ax_linear.text(0.98, 0.02, f'1 MC ≈ {scale_factor:.2f} frames',
                       transform=ax_linear.transAxes, fontsize=8, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # =============== LOG SCALE PLOT (bottom row) ===============
            # Shade fit range region
            ax_log.axvspan(fit_range[0], fit_range[1], alpha=0.15, color='gray',
                           label=f'Fit range [{fit_range[0]}, {fit_range[1]}]')

            # Plot experimental ACF data (log scale)
            exp_acf_plot = np.copy(exp_acf[:max_lag])
            exp_acf_plot[exp_acf_plot <= 0] = np.nan
            ax_log.semilogy(lags, exp_acf_plot, color=self._get_color(condition), linewidth=2,
                    label=f'Exp data')

            # Plot exponential fit for experimental data
            if not np.isnan(exp_tau) and exp_tau > 0:
                exp_fit = np.exp(-lags / exp_tau)
                ax_log.semilogy(lags, exp_fit, color=self._get_color(condition), linewidth=1.5,
                        linestyle=':', alpha=0.8, label=f'Exp fit (τ={exp_tau:.1f})')

            if len(ising_acf) > 0:
                max_lag_ising = min(max_lag, len(ising_acf))
                lags_ising = np.arange(max_lag_ising)
                ising_acf_plot = np.copy(ising_acf[:max_lag_ising])
                ising_acf_plot[ising_acf_plot <= 0] = np.nan
                ax_log.semilogy(lags_ising, ising_acf_plot, color='gray', linewidth=2, linestyle='--',
                        label=f'Ising data')

                # Plot exponential fit for Ising data
                if not np.isnan(ising_tau) and ising_tau > 0:
                    ising_fit = np.exp(-lags_ising / ising_tau)
                    ax_log.semilogy(lags_ising, ising_fit, color='dimgray', linewidth=1.5,
                            linestyle=':', alpha=0.8, label=f'Ising fit (τ={ising_tau:.1f})')

            ax_log.set_xlabel('Lag (frames/MC)')
            ax_log.set_ylabel('Autocorrelation (log scale)')
            ax_log.legend(fontsize=6, loc='upper right')
            ax_log.set_xlim(0, max_lag)
            ax_log.set_ylim(1e-2, 1.5)  # Reasonable log scale range

        # Hide unused subplots
        for i in range(n_conditions, 4):
            axes[0, i].set_visible(False)
            axes[1, i].set_visible(False)

        fig.suptitle('Activity Temporal Autocorrelation', fontsize=12)
        fig.tight_layout()
        self._save_figure(fig, 'activity_autocorrelation')

    def plot_conversion_factor_variability(self):
        """
        Plot temporal conversion factor variability across simulations.

        Matches MATLAB Figure 4c:
        - Panel 1: Top 10 vs All Sims boxplot comparison
        - Panel 2: Std bars with ratio % annotations
        """
        comparison = self.results.get('Comparison', {})

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: Box plots comparing Top 10 vs All Simulations
        ax = axes[0]
        data_top10 = []
        data_all = []
        labels = []
        colors = []

        for condition in self.conditions:
            if condition in comparison:
                scale_factors = comparison[condition].get('temporal_scale_factors', np.array([]))
                if len(scale_factors) > 0:
                    valid_scales = scale_factors[~np.isnan(scale_factors)]
                    if len(valid_scales) > 0:
                        # Get scale factors for top 10 best matches
                        best_idx = self._get_best_idx(condition)
                        if len(best_idx) > 0:
                            top_n = min(10, len(best_idx))
                            top_idx = best_idx[:top_n]
                            top_scales = scale_factors[top_idx]
                            top_scales = top_scales[~np.isnan(top_scales)]
                            if len(top_scales) > 0:
                                data_top10.append(top_scales)
                                data_all.append(valid_scales)
                                labels.append(condition)
                                colors.append(self._get_color(condition))

        if len(data_top10) > 0:
            n_conds = len(labels)
            x = np.arange(n_conds)
            width = 0.35

            # Create grouped box plots
            positions_top10 = x - width/2
            positions_all = x + width/2

            bp1 = ax.boxplot(data_top10, positions=positions_top10, widths=width*0.8,
                            patch_artist=True)
            bp2 = ax.boxplot(data_all, positions=positions_all, widths=width*0.8,
                            patch_artist=True)

            for j, patch in enumerate(bp1['boxes']):
                patch.set_facecolor(colors[j])
                patch.set_alpha(0.8)
            for j, patch in enumerate(bp2['boxes']):
                patch.set_facecolor('lightgray')
                patch.set_alpha(0.6)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Scale Factor (frames/MC)')
            ax.set_title('Temporal Scale Factor\n(Top 10 vs All Sims)')
            ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Top 10', 'All Sims'],
                     loc='upper right', fontsize=8)
        else:
            ax.set_visible(False)

        # Panel 2: Std bars with ratio % annotations
        ax = axes[1]
        conditions_plotted = []
        stds_top10 = []
        stds_all = []

        for condition in self.conditions:
            if condition in comparison:
                scale_factors = comparison[condition].get('temporal_scale_factors', np.array([]))
                if len(scale_factors) > 0:
                    valid_scales = scale_factors[~np.isnan(scale_factors)]
                    best_idx = self._get_best_idx(condition)
                    if len(best_idx) > 0 and len(valid_scales) > 0:
                        top_n = min(10, len(best_idx))
                        top_idx = best_idx[:top_n]
                        top_scales = scale_factors[top_idx]
                        top_scales = top_scales[~np.isnan(top_scales)]
                        if len(top_scales) > 1:
                            conditions_plotted.append(condition)
                            stds_top10.append(np.std(top_scales))
                            stds_all.append(np.std(valid_scales))

        if len(conditions_plotted) > 0:
            x = np.arange(len(conditions_plotted))
            width = 0.35

            bars1 = ax.bar(x - width/2, stds_top10, width, label='Top 10',
                          color=[self._get_color(c) for c in conditions_plotted], alpha=0.8)
            bars2 = ax.bar(x + width/2, stds_all, width, label='All Sims',
                          color='lightgray', alpha=0.6)

            # Add ratio % annotations
            for i in range(len(conditions_plotted)):
                if stds_all[i] > 0:
                    ratio = (stds_top10[i] / stds_all[i]) * 100
                    ax.text(x[i], max(stds_top10[i], stds_all[i]) * 1.05,
                           f'{ratio:.0f}%', ha='center', va='bottom', fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(conditions_plotted, rotation=45, ha='right')
            ax.set_ylabel('Standard Deviation')
            ax.set_title('Scale Factor Variability\n(Std with Ratio %)')
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.set_visible(False)

        fig.suptitle('Conversion Factor Variability', fontsize=12)
        fig.tight_layout()
        self._save_figure(fig, 'conversion_factor_variability')

    # =========================================================================
    # Q-Q PLOTS (MATLAB Figures 10-13)
    # =========================================================================

    def plot_qq_plots(self):
        """
        Generate Q-Q plots comparing experimental vs Ising distributions.
        MATLAB Figures 10-13: Q-Q plots for Moran's I and Activity (best match and top 3).
        """
        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        n_quantiles = 100
        p = np.linspace(0.01, 0.99, n_quantiles)

        for metric in ['MoransI', 'Activity']:
            for mode in ['best', 'top3']:
                n_conds = len([c for c in self.conditions if c in comparison])
                n_conds = max(n_conds, len(self.conditions))  # use all conditions
                n_cols = min(n_conds, 3)
                n_rows = -(-n_conds // n_cols)  # ceiling division
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
                axes = np.atleast_2d(axes)

                for c, condition in enumerate(self.conditions):
                    ax = axes.flat[c]

                    if condition not in exp_stats or condition not in comparison:
                        ax.set_visible(False)
                        continue

                    # Get experimental data
                    exp_data = exp_stats[condition].get(f'{metric}_all', np.array([]))
                    if len(exp_data) == 0:
                        ax.set_visible(False)
                        continue

                    exp_data = np.array(exp_data)
                    exp_data = exp_data[~np.isnan(exp_data)]
                    if len(exp_data) == 0:
                        ax.set_visible(False)
                        continue

                    exp_q = np.quantile(exp_data, p)

                    best_idx = self._get_best_idx(condition)
                    ising_metric_all = ising_data.get(f'{metric}_all', [])

                    if mode == 'best':
                        # Single best match
                        if len(best_idx) > 0 and len(ising_metric_all) > best_idx[0]:
                            ising_data_arr = np.array(ising_metric_all[best_idx[0]])
                            ising_data_arr = ising_data_arr[~np.isnan(ising_data_arr)]
                            if len(ising_data_arr) > 0:
                                ising_q = np.quantile(ising_data_arr, p)

                                ax.plot(exp_q, ising_q, 'o', color=self._get_color(condition),
                                        markersize=4, markerfacecolor=self._get_color(condition))

                                # Reference line y=x
                                all_vals = np.concatenate([exp_q, ising_q])
                                ax.plot([all_vals.min(), all_vals.max()],
                                       [all_vals.min(), all_vals.max()], 'k--', lw=1.5)
                    else:
                        # Top 3 matches with decreasing alpha
                        alphas = [1.0, 0.6, 0.3]
                        all_ising_q = []
                        for rank in range(min(3, len(best_idx))):
                            if len(ising_metric_all) > best_idx[rank]:
                                ising_data_arr = np.array(ising_metric_all[best_idx[rank]])
                                ising_data_arr = ising_data_arr[~np.isnan(ising_data_arr)]
                                if len(ising_data_arr) > 0:
                                    ising_q = np.quantile(ising_data_arr, p)
                                    all_ising_q.extend(ising_q)
                                    ax.scatter(exp_q, ising_q, s=16, c=[self._get_color(condition)],
                                              alpha=alphas[rank], label=f'Rank {rank+1}' if c == 0 else None)

                        # Reference line y=x
                        if len(all_ising_q) > 0:
                            all_vals = np.concatenate([exp_q, np.array(all_ising_q)])
                            ax.plot([all_vals.min(), all_vals.max()],
                                   [all_vals.min(), all_vals.max()], 'k--', lw=1.5)

                    ax.set_xlabel('Experimental Quantiles')
                    ax.set_ylabel('Ising Quantiles')
                    ax.set_title(condition)
                    ax.set_aspect('equal', adjustable='box')

                # Hide unused axes
                for idx in range(len(self.conditions), n_rows * n_cols):
                    axes.flat[idx].set_visible(False)

                metric_name = "Moran's I" if metric == 'MoransI' else 'Activity'
                mode_name = 'Best Match' if mode == 'best' else 'Top 3 Matches'
                fig.suptitle(f'Q-Q Plot: {metric_name} Distribution ({mode_name})', fontweight='bold')
                if mode == 'top3':
                    # Add legend for top3 mode
                    handles, labels = axes.flat[0].get_legend_handles_labels()
                    if handles:
                        fig.legend(handles, labels, loc='upper right', fontsize=8)
                fig.tight_layout()
                self._save_figure(fig, f'qq_{metric.lower()}_{mode}')

    # =========================================================================
    # PARAMETER COVARIANCE/CORRELATION (MATLAB Figures 17b-c)
    # =========================================================================

    def plot_parameter_covariance(self):
        """
        Plot parameter covariance matrix for top 10 best matches.
        MATLAB Figure 17b.
        """
        comparison = self.results.get('Comparison', {})

        param_names = ['beta', 'c', 'decay_const', 'inhibition_range', 'bias']
        param_labels = ['β', 'c', 'decay', 'inhib', 'bias']
        n_params = len(param_names)
        n_conds = len(self.conditions)

        fig, axes = plt.subplots(1, n_conds + 1, figsize=(4*(n_conds+1), 4))
        if n_conds + 1 == 1:
            axes = [axes]

        all_params = []
        conditions_with_data = []

        for c, condition in enumerate(self.conditions):
            if condition not in comparison:
                axes[c].set_visible(False)
                continue

            # Build parameter matrix [n_top x 5]
            params_cond = comparison[condition].get('bestMatch_params', {})
            if not params_cond:
                axes[c].set_visible(False)
                continue

            # Stack parameters into matrix
            param_arrays = []
            min_len = float('inf')
            for p in param_names:
                arr = np.array(params_cond.get(p, []))
                if len(arr) > 0:
                    min_len = min(min_len, len(arr))
                    param_arrays.append(arr)

            if len(param_arrays) != n_params or min_len == 0 or min_len == float('inf'):
                axes[c].set_visible(False)
                continue

            param_matrix = np.column_stack([arr[:int(min_len)] for arr in param_arrays])
            all_params.append(param_matrix)
            conditions_with_data.append(condition)

            # Compute covariance
            if param_matrix.shape[0] > 1:
                cov_mat = np.cov(param_matrix, rowvar=False)
            else:
                cov_mat = np.zeros((n_params, n_params))

            ax = axes[c]
            im = ax.imshow(cov_mat, cmap='viridis')
            ax.set_xticks(range(n_params))
            ax.set_xticklabels(param_labels, fontsize=8)
            ax.set_yticks(range(n_params))
            ax.set_yticklabels(param_labels, fontsize=8)
            ax.set_title(f'{condition} (n={param_matrix.shape[0]})')
            plt.colorbar(im, ax=ax, shrink=0.8)

        # Pooled covariance in last subplot
        ax = axes[-1]
        if len(all_params) > 0:
            all_params_pooled = np.vstack(all_params)
            if all_params_pooled.shape[0] > 1:
                cov_pooled = np.cov(all_params_pooled, rowvar=False)
            else:
                cov_pooled = np.zeros((n_params, n_params))

            im = ax.imshow(cov_pooled, cmap='viridis')
            ax.set_xticks(range(n_params))
            ax.set_xticklabels(param_labels, fontsize=8)
            ax.set_yticks(range(n_params))
            ax.set_yticklabels(param_labels, fontsize=8)
            ax.set_title(f'Pooled (n={len(all_params_pooled)})')
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.set_visible(False)

        fig.suptitle('Parameter Covariance Matrix (Top Best Matches)', fontweight='bold')
        fig.tight_layout()
        self._save_figure(fig, 'parameter_covariance')

    def plot_parameter_correlation(self):
        """
        Plot parameter correlation matrix for top 10 best matches.
        MATLAB Figure 17c. Uses red-white-blue diverging colormap.
        """
        comparison = self.results.get('Comparison', {})

        param_names = ['beta', 'c', 'decay_const', 'inhibition_range', 'bias']
        param_labels = ['β', 'c', 'decay', 'inhib', 'bias']
        n_params = len(param_names)
        n_conds = len(self.conditions)

        fig, axes = plt.subplots(1, n_conds + 1, figsize=(4*(n_conds+1), 4))
        if n_conds + 1 == 1:
            axes = [axes]

        all_params = []
        conditions_with_data = []

        for c, condition in enumerate(self.conditions):
            if condition not in comparison:
                axes[c].set_visible(False)
                continue

            params_cond = comparison[condition].get('bestMatch_params', {})
            if not params_cond:
                axes[c].set_visible(False)
                continue

            param_arrays = []
            min_len = float('inf')
            for p in param_names:
                arr = np.array(params_cond.get(p, []))
                if len(arr) > 0:
                    min_len = min(min_len, len(arr))
                    param_arrays.append(arr)

            if len(param_arrays) != n_params or min_len == 0 or min_len == float('inf'):
                axes[c].set_visible(False)
                continue

            param_matrix = np.column_stack([arr[:int(min_len)] for arr in param_arrays])
            all_params.append(param_matrix)
            conditions_with_data.append(condition)

            # Compute correlation
            if param_matrix.shape[0] > 1:
                corr_mat = np.corrcoef(param_matrix, rowvar=False)
            else:
                corr_mat = np.eye(n_params)

            ax = axes[c]
            # Use pcolormesh (vector) instead of imshow (raster) so that
            # Adobe Illustrator preserves colours when opening the PDF.
            edges = np.arange(n_params + 1) - 0.5
            im = ax.pcolormesh(edges, edges, corr_mat, cmap='RdBu_r',
                               vmin=-1, vmax=1, edgecolors='none', linewidth=0)
            ax.set_xlim(-0.5, n_params - 0.5)
            ax.set_ylim(n_params - 0.5, -0.5)  # invert y to match imshow
            ax.set_aspect('equal')
            ax.set_xticks(range(n_params))
            ax.set_xticklabels(param_labels, fontsize=8)
            ax.set_yticks(range(n_params))
            ax.set_yticklabels(param_labels, fontsize=8)
            ax.set_title(f'{condition} (n={param_matrix.shape[0]})')
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Dark grey diagonal overlay (self-correlation = 1)
            for i in range(n_params):
                ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=True,
                                           facecolor='dimgray', edgecolor='none'))

        # Pooled correlation in last subplot
        ax = axes[-1]
        if len(all_params) > 0:
            all_params_pooled = np.vstack(all_params)
            if all_params_pooled.shape[0] > 1:
                corr_pooled = np.corrcoef(all_params_pooled, rowvar=False)
            else:
                corr_pooled = np.eye(n_params)

            edges = np.arange(n_params + 1) - 0.5
            im = ax.pcolormesh(edges, edges, corr_pooled, cmap='RdBu_r',
                               vmin=-1, vmax=1, edgecolors='none', linewidth=0)
            ax.set_xlim(-0.5, n_params - 0.5)
            ax.set_ylim(n_params - 0.5, -0.5)
            ax.set_aspect('equal')
            ax.set_xticks(range(n_params))
            ax.set_xticklabels(param_labels, fontsize=8)
            ax.set_yticks(range(n_params))
            ax.set_yticklabels(param_labels, fontsize=8)
            ax.set_title(f'Pooled (n={len(all_params_pooled)})')
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Dark grey diagonal overlay (self-correlation = 1)
            for i in range(n_params):
                ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=True,
                                           facecolor='dimgray', edgecolor='none'))
        else:
            ax.set_visible(False)

        fig.suptitle('Parameter Correlation Matrix (Top Best Matches)', fontweight='bold')
        fig.tight_layout()
        self._save_figure(fig, 'parameter_correlation', formats=['png', 'pdf', 'svg'])

    # =========================================================================
    # PARAMETER SENSITIVITY (MATLAB Figure 18)
    # =========================================================================

    def plot_parameter_sensitivity(self):
        """
        1D parameter sensitivity analysis showing mean Wasserstein vs parameter value.
        MATLAB Figure 18.
        """
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        params = ising_data.get('params', {})
        if not params:
            print("  No parameter data for sensitivity analysis")
            return

        param_names = ['beta', 'c', 'decay_const', 'inhibition_range', 'bias']
        param_labels = ['Beta', 'Coupling (c)', 'Decay Const', 'Inhib Range', 'Bias']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for p_idx, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
            ax = axes[p_idx]

            # Get unique parameter values
            param_data = np.array(params.get(param_name, []))
            if len(param_data) == 0:
                ax.set_visible(False)
                continue

            param_values = np.unique(param_data)

            for condition in self.conditions:
                if condition not in comparison:
                    continue

                wd = np.array(comparison[condition].get('wasserstein_dist', []))
                if len(wd) == 0 or len(wd) != len(param_data):
                    continue

                # Compute mean ± SEM at each parameter value
                means, sems = [], []
                for val in param_values:
                    mask = param_data == val
                    w_vals = wd[mask]
                    w_vals = w_vals[~np.isnan(w_vals)]
                    if len(w_vals) > 0:
                        means.append(np.mean(w_vals))
                        sems.append(np.std(w_vals) / np.sqrt(len(w_vals)) if len(w_vals) > 1 else 0)
                    else:
                        means.append(np.nan)
                        sems.append(np.nan)

                means = np.array(means)
                sems = np.array(sems)
                valid = ~np.isnan(means)

                if np.any(valid):
                    ax.errorbar(param_values[valid], means[valid], yerr=sems[valid],
                               fmt='o-', color=self._get_color(condition), lw=1.5,
                               markersize=8, label=condition,
                               markerfacecolor=self._get_color(condition),
                               capsize=3)

            # Boundary markers
            if len(param_values) > 0:
                ax.axvline(param_values.min(), color='r', ls='--', lw=1, alpha=0.5)
                ax.axvline(param_values.max(), color='r', ls='--', lw=1, alpha=0.5)

            ax.set_xlabel(param_label)
            ax.set_ylabel('Mean Wasserstein')
            ax.set_title(param_label)
            if p_idx == 0:
                ax.legend(loc='best', fontsize=8)

        # Legend subplot (6th panel)
        ax = axes[5]
        ax.axis('off')
        info_text = ['Red dashed lines = parameter boundaries', '',
                     'Shows how match quality varies', 'with each parameter value', '',
                     'Lower Wasserstein = better match']
        ax.text(0.5, 0.5, '\n'.join(info_text), ha='center', va='center',
               fontsize=10, transform=ax.transAxes)

        fig.suptitle('Parameter Sensitivity: Mean Wasserstein vs Parameter Value', fontweight='bold')
        fig.tight_layout()
        self._save_figure(fig, 'parameter_sensitivity')

    # =========================================================================
    # FIXED PARAMETER ANALYSIS (MATLAB Figures 19a-e)
    # =========================================================================

    def plot_fixed_parameter_analysis(self):
        """
        2D analysis with one parameter fixed at a time.
        MATLAB Figures 19a-e (one per fixed parameter).
        """
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        params = ising_data.get('params', {})
        if not params:
            print("  No parameter data for fixed-parameter analysis")
            return

        param_names = ['beta', 'c', 'decay_const', 'inhibition_range', 'bias']
        param_labels = ['Beta', 'Coupling (c)', 'Decay Const', 'Inhib Range', 'Bias']
        line_styles = ['-', '--', ':', '-.', '-']
        line_markers = ['o', 's', '^', 'D', 'v']

        for fixed_idx, fixed_param in enumerate(param_names):
            fixed_label = param_labels[fixed_idx]

            fixed_data = np.array(params.get(fixed_param, []))
            if len(fixed_data) == 0:
                continue

            fixed_values = np.unique(fixed_data)
            other_params = [p for i, p in enumerate(param_names) if i != fixed_idx]
            other_labels = [l for i, l in enumerate(param_labels) if i != fixed_idx]

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            # Subplots 0-3: Other parameters vs mean WD at each fixed value
            for sp, (other_param, other_label) in enumerate(zip(other_params, other_labels)):
                ax = axes[sp]

                other_data = np.array(params.get(other_param, []))
                if len(other_data) == 0 or len(other_data) != len(fixed_data):
                    ax.set_visible(False)
                    continue

                other_values = np.unique(other_data)

                for fv_idx, fixed_val in enumerate(fixed_values):
                    fixed_mask = fixed_data == fixed_val

                    for condition in self.conditions:
                        if condition not in comparison:
                            continue

                        wd = np.array(comparison[condition].get('wasserstein_dist', []))
                        if len(wd) == 0 or len(wd) != len(fixed_data):
                            continue

                        means = []
                        for ov in other_values:
                            mask = fixed_mask & (other_data == ov)
                            w_vals = wd[mask]
                            w_vals = w_vals[~np.isnan(w_vals)]
                            if len(w_vals) > 0:
                                means.append(np.mean(w_vals))
                            else:
                                means.append(np.nan)

                        means = np.array(means)
                        valid = ~np.isnan(means)

                        if np.any(valid):
                            # Use different line style for each fixed value
                            ls = line_styles[fv_idx % len(line_styles)]
                            ax.plot(other_values[valid], means[valid], ls,
                                   color=self._get_color(condition), lw=1.2, alpha=0.7)

                # Boundary markers
                if len(other_values) > 0:
                    ax.axvline(other_values.min(), color='r', ls='--', lw=0.5, alpha=0.3)
                    ax.axvline(other_values.max(), color='r', ls='--', lw=0.5, alpha=0.3)

                ax.set_xlabel(other_label)
                ax.set_ylabel('Mean WD')
                ax.set_title(other_label)

            # Subplot 4: Best WD at each fixed value
            ax = axes[4]
            for condition in self.conditions:
                if condition not in comparison:
                    continue

                wd = np.array(comparison[condition].get('wasserstein_dist', []))
                if len(wd) == 0 or len(wd) != len(fixed_data):
                    continue

                best_wd = []
                for fv in fixed_values:
                    mask = fixed_data == fv
                    w_vals = wd[mask]
                    w_vals = w_vals[~np.isnan(w_vals)]
                    if len(w_vals) > 0:
                        best_wd.append(np.min(w_vals))
                    else:
                        best_wd.append(np.nan)

                best_wd = np.array(best_wd)
                valid = ~np.isnan(best_wd)
                if np.any(valid):
                    ax.plot(fixed_values[valid], best_wd[valid], 'o-',
                           color=self._get_color(condition), lw=1.5, markersize=8,
                           label=condition, markerfacecolor=self._get_color(condition))

            ax.set_xlabel(fixed_label)
            ax.set_ylabel('Best WD (min)')
            ax.set_title(f'Best Match at Fixed {fixed_label}')
            ax.legend(loc='best', fontsize=8)

            # Subplot 5: Legend for line styles
            ax = axes[5]
            ax.axis('off')
            legend_text = [f'Fixed {fixed_label} values:', '']
            for fv_idx, fv in enumerate(fixed_values[:min(5, len(fixed_values))]):
                ls_name = ['solid', 'dashed', 'dotted', 'dashdot', 'solid'][fv_idx % 5]
                legend_text.append(f'  {ls_name}: {fv:.3g}')
            if len(fixed_values) > 5:
                legend_text.append(f'  ... and {len(fixed_values)-5} more')
            legend_text.append('')
            legend_text.append('Colors = conditions')
            ax.text(0.1, 0.9, '\n'.join(legend_text), transform=ax.transAxes,
                   va='top', family='monospace', fontsize=9)

            fig.suptitle(f'Fixed {fixed_label} Analysis: How Other Parameters Affect Match Quality',
                        fontweight='bold')
            fig.tight_layout()
            self._save_figure(fig, f'fixed_parameter_{fixed_param}')

    # =========================================================================
    # MULTI-OBSERVABLE COMPARISON (MATLAB Figure 8)
    # =========================================================================

    def plot_multi_observable_comparison(self, raw=False):
        """
        Generate multi-observable comparison figure for each condition.
        MATLAB Figure 8: 4-panel comparison showing Moran's I, Activity,
        Autocorrelation, and Summary statistics.
        """
        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        for condition in self.conditions:
            if condition not in comparison or condition not in exp_stats:
                continue

            # Get best matching simulation index
            best_idx_list = self._get_best_idx(condition)
            if len(best_idx_list) == 0:
                continue
            best_idx = best_idx_list[0]

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Panel 1: Moran's I Distribution
            ax = axes[0, 0]
            exp_mi = exp_stats[condition].get('MoransI_all', np.array([]))
            ising_mi_all = ising_data.get('MoransI_all', [])
            ising_mi = np.array(ising_mi_all[best_idx]) if len(ising_mi_all) > best_idx else np.array([])

            exp_mi_valid = exp_mi[~np.isnan(exp_mi)] if len(exp_mi) > 0 else np.array([])
            ising_mi_valid = ising_mi[~np.isnan(ising_mi)] if len(ising_mi) > 0 else np.array([])
            all_mi = np.concatenate([v for v in (exp_mi_valid, ising_mi_valid) if len(v) > 0])
            mi_bins = np.linspace(all_mi.min(), all_mi.max(), 51) if len(all_mi) > 0 else 50
            if len(exp_mi_valid) > 0:
                ax.hist(exp_mi_valid, bins=mi_bins, density=True, alpha=0.5,
                        color=self._get_color(condition), label='Experimental')
            if len(ising_mi_valid) > 0:
                ax.hist(ising_mi_valid, bins=mi_bins, density=True, alpha=0.5,
                        color='gray', label='Ising')
            ax.set_xlabel("Moran's I")
            ax.set_ylabel('PDF')
            ax.set_title("Moran's I Distribution")
            ax.legend()

            # Panel 2: Activity Distribution
            ax = axes[0, 1]
            exp_act = exp_stats[condition].get('Activity_all', np.array([]))
            ising_act_all = ising_data.get('Activity_all', [])
            ising_act = np.array(ising_act_all[best_idx]) if len(ising_act_all) > best_idx else np.array([])

            exp_act_valid = exp_act[~np.isnan(exp_act)] if len(exp_act) > 0 else np.array([])
            ising_act_valid = ising_act[~np.isnan(ising_act)] if len(ising_act) > 0 else np.array([])
            all_act = np.concatenate([v for v in (exp_act_valid, ising_act_valid) if len(v) > 0])
            act_bins = np.linspace(all_act.min(), all_act.max(), 51) if len(all_act) > 0 else 50
            if len(exp_act_valid) > 0:
                ax.hist(exp_act_valid, bins=act_bins, density=True, alpha=0.5,
                        color=self._get_color(condition), label='Experimental')
            if len(ising_act_valid) > 0:
                ax.hist(ising_act_valid, bins=act_bins, density=True, alpha=0.5,
                        color='gray', label='Ising')
            ax.set_xlabel('Activity (fraction active)')
            ax.set_ylabel('PDF')
            ax.set_title('Activity Distribution')
            ax.legend()

            # Panel 3: Temporal Autocorrelation (Moran's I)
            ax = axes[1, 0]
            max_lag = 50

            # Experimental ACF
            exp_acf = exp_stats[condition].get('Autocorr_acf_trial_averaged',
                       exp_stats[condition].get('Autocorr_acf', np.array([])))
            if isinstance(exp_acf, np.ndarray) and len(exp_acf) > 0:
                exp_acf = exp_acf[:min(max_lag + 1, len(exp_acf))]
                ax.plot(range(len(exp_acf)), exp_acf, '-', color=self._get_color(condition),
                        lw=2, label='Experimental')

            # Ising ACF
            ising_acf_all = ising_data.get('Autocorr_acf', [])
            if len(ising_acf_all) > best_idx:
                ising_acf = np.array(ising_acf_all[best_idx]) if isinstance(ising_acf_all[best_idx], (list, np.ndarray)) else np.array([])
                if len(ising_acf) > 0:
                    ising_acf = ising_acf[:min(max_lag + 1, len(ising_acf))]
                    ax.plot(range(len(ising_acf)), ising_acf, '-', color='gray', lw=2, label='Ising')

            ax.set_xlabel('Lag')
            ax.set_ylabel('Autocorrelation')
            ax.set_title("Temporal Autocorrelation (Moran's I)")
            ax.legend()
            ax.axhline(0, color='black', linewidth=0.5, linestyle=':')

            # Panel 4: Summary Statistics
            ax = axes[1, 1]
            ax.axis('off')

            # Build summary text
            exp_mi_mean = np.nanmean(exp_mi) if len(exp_mi) > 0 else np.nan
            ising_mi_mean = np.nanmean(ising_mi) if len(ising_mi) > 0 else np.nan
            exp_act_mean = np.nanmean(exp_act) if len(exp_act) > 0 else np.nan
            ising_act_mean = np.nanmean(ising_act) if len(ising_act) > 0 else np.nan
            if raw:
                raw_wd = self._compute_raw_combined_wd(comparison[condition])
                wd_best = raw_wd[best_idx] if raw_wd is not None and len(raw_wd) > best_idx else np.nan
                wd_label = 'Raw WD'
            else:
                wd = comparison[condition].get('wasserstein_dist', np.array([]))
                wd_best = wd[best_idx] if len(wd) > best_idx else np.nan
                wd_label = 'Wasserstein dist'

            stats_text = [
                f"Condition: {condition}",
                f"Best Match Index: {best_idx}",
                "",
                "Moran's I:",
                f"  Exp mean: {exp_mi_mean:.4f}",
                f"  Ising mean: {ising_mi_mean:.4f}",
                "",
                "Activity:",
                f"  Exp mean: {exp_act_mean:.4f}",
                f"  Ising mean: {ising_act_mean:.4f}",
                "",
                f"{wd_label}: {wd_best:.4f}"
            ]
            ax.text(0.1, 0.9, '\n'.join(stats_text), transform=ax.transAxes,
                    va='top', fontsize=10, family='monospace')
            ax.set_title('Summary Statistics')

            title_suffix = ' (raw WD)' if raw else ''
            fig.suptitle(f'{condition}: Best Match Comparison{title_suffix}', fontweight='bold')
            fig.tight_layout()
            suffix = '_raw' if raw else ''
            self._save_figure(fig, f'multi_observable_{condition.lower()}{suffix}')

    def plot_multi_observable_comparison_raw(self):
        """Multi-observable comparison with raw WD."""
        return self.plot_multi_observable_comparison(raw=True)

    # =========================================================================
    # SIMPLIFIED DISTRIBUTION COMPARISON (MATLAB Figure 9)
    # =========================================================================

    def plot_simplified_distributions(self):
        """
        Generate simplified two-panel distribution comparison.
        MATLAB Figure 9: Moran's I and Activity distributions side by side.
        """
        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        for condition in self.conditions:
            if condition not in comparison or condition not in exp_stats:
                continue

            best_idx_list = self._get_best_idx(condition)
            if len(best_idx_list) == 0:
                continue
            best_idx = best_idx_list[0]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Panel 1: Moran's I
            ax = axes[0]
            exp_mi = exp_stats[condition].get('MoransI_all', np.array([]))
            ising_mi_all = ising_data.get('MoransI_all', [])
            ising_mi = np.array(ising_mi_all[best_idx]) if len(ising_mi_all) > best_idx else np.array([])

            exp_mi_valid = exp_mi[~np.isnan(exp_mi)] if len(exp_mi) > 0 else np.array([])
            ising_mi_valid = ising_mi[~np.isnan(ising_mi)] if len(ising_mi) > 0 else np.array([])
            all_mi = np.concatenate([v for v in (exp_mi_valid, ising_mi_valid) if len(v) > 0])
            mi_bins = np.linspace(all_mi.min(), all_mi.max(), 51) if len(all_mi) > 0 else 50
            if len(exp_mi_valid) > 0:
                ax.hist(exp_mi_valid, bins=mi_bins, density=True, alpha=0.5,
                        color=self._get_color(condition), label='Experimental')
            if len(ising_mi_valid) > 0:
                ax.hist(ising_mi_valid, bins=mi_bins, density=True, alpha=0.5,
                        color='gray', label='Ising')
            ax.set_xlabel("Moran's I")
            ax.set_ylabel('PDF')
            ax.set_title("Moran's I Distribution")
            ax.legend()

            # Panel 2: Activity
            ax = axes[1]
            exp_act = exp_stats[condition].get('Activity_all', np.array([]))
            ising_act_all = ising_data.get('Activity_all', [])
            ising_act = np.array(ising_act_all[best_idx]) if len(ising_act_all) > best_idx else np.array([])

            exp_act_valid = exp_act[~np.isnan(exp_act)] if len(exp_act) > 0 else np.array([])
            ising_act_valid = ising_act[~np.isnan(ising_act)] if len(ising_act) > 0 else np.array([])
            all_act = np.concatenate([v for v in (exp_act_valid, ising_act_valid) if len(v) > 0])
            act_bins = np.linspace(all_act.min(), all_act.max(), 51) if len(all_act) > 0 else 50
            if len(exp_act_valid) > 0:
                ax.hist(exp_act_valid, bins=act_bins, density=True, alpha=0.5,
                        color=self._get_color(condition), label='Experimental')
            if len(ising_act_valid) > 0:
                ax.hist(ising_act_valid, bins=act_bins, density=True, alpha=0.5,
                        color='gray', label='Ising')
            ax.set_xlabel('Activity (fraction active)')
            ax.set_ylabel('PDF')
            ax.set_title('Activity Distribution')
            ax.legend()

            fig.suptitle(f'{condition}: Distribution Comparison', fontweight='bold')
            fig.tight_layout()
            self._save_figure(fig, f'distribution_comparison_{condition.lower()}')

    # =========================================================================
    # TILED MODE ANALYSIS (MATLAB Figure 14)
    # =========================================================================

    def plot_tiled_mode_analysis(self):
        """
        Tiled mode analysis: Single position vs pooled positions WD.
        MATLAB Figure 14. Only applicable when gridMode='subselect_tiled'.
        """
        # Check if tiled mode data is available
        if 'TiledAnalysis' not in self.results:
            print("  Skipping tiled mode analysis (not in tiled mode)")
            return

        tiled = self.results['TiledAnalysis']
        conditions_with_data = [c for c in self.conditions if c in tiled]

        if len(conditions_with_data) == 0:
            print("  Skipping tiled mode analysis (no data)")
            return

        n_conds = len(conditions_with_data)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: WD bars (Single vs Pooled)
        ax = axes[0]
        wd_single = []
        wd_pooled = []
        for condition in conditions_with_data:
            wd_single.append(tiled[condition].get('WD_single', np.nan))
            wd_pooled.append(tiled[condition].get('WD_pooled', np.nan))

        x = np.arange(n_conds)
        ax.bar(x - 0.2, wd_single, 0.4, label='Single (P1)', color='gray')
        ax.bar(x + 0.2, wd_pooled, 0.4, label='Pooled', color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions_with_data, rotation=45)
        ax.set_ylabel('Wasserstein Distance')
        ax.set_title('Data vs Model: WD Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: % Change
        ax = axes[1]
        for c, condition in enumerate(conditions_with_data):
            change = tiled[condition].get('WD_change_pct', 0)
            color = 'green' if change < 0 else 'red'
            ax.bar(c, change, color=color)
        ax.axhline(0, color='k', lw=1)
        ax.set_xticks(range(n_conds))
        ax.set_xticklabels(conditions_with_data, rotation=45)
        ax.set_ylabel('% Change in WD')
        ax.set_title('Pooling Effect (neg = better match)')
        ax.grid(True, alpha=0.3)

        # Panel 3: Summary
        ax = axes[2]
        ax.axis('off')
        n_improved = sum(1 for cond in conditions_with_data
                        if tiled[cond].get('improved', False))
        summary = [
            "=== TILED MODE ANALYSIS ===",
            "",
            f"Improved: {n_improved}/{n_conds}",
        ]
        for condition in conditions_with_data:
            summary.append(f"\n{condition}:")
            summary.append(f"  P1: {tiled[condition].get('WD_single', np.nan):.4f}")
            summary.append(f"  Pooled: {tiled[condition].get('WD_pooled', np.nan):.4f} "
                          f"({tiled[condition].get('WD_change_pct', 0):+.1f}%)")
        ax.text(0.05, 0.95, '\n'.join(summary), transform=ax.transAxes,
                va='top', fontsize=9, family='monospace')

        fig.suptitle('Tiled Mode: Single Position vs Pooled', fontweight='bold')
        fig.tight_layout()
        self._save_figure(fig, 'tiled_mode_analysis')

    # =========================================================================
    # CENTRE VS TILED COMPARISON (MATLAB Figure 15)
    # =========================================================================

    def plot_centre_vs_tiled_comparison(self):
        """
        Centre crop vs tiled average comparison.
        MATLAB Figure 15. Only applicable when gridMode='subselect_centre_vs_tiled'.
        """
        if 'centreVsTiled' not in self.results:
            print("  Skipping centre vs tiled (not in centre_vs_tiled mode)")
            return

        cvt = self.results['centreVsTiled']
        conditions_with_data = [c for c in self.conditions if c in cvt]

        if len(conditions_with_data) == 0:
            print("  Skipping centre vs tiled (no data)")
            return

        n_conds = len(conditions_with_data)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: WD bars (Centre vs Tiled)
        ax = axes[0]
        wd_centre = []
        wd_tiled = []
        for condition in conditions_with_data:
            wd_centre.append(cvt[condition].get('WD_centre', np.nan))
            wd_tiled.append(cvt[condition].get('WD_tiled', np.nan))

        x = np.arange(n_conds)
        ax.bar(x - 0.2, wd_centre, 0.4, label='Centre', color='orange')
        ax.bar(x + 0.2, wd_tiled, 0.4, label='Tiled', color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions_with_data, rotation=45)
        ax.set_ylabel('Wasserstein Distance')
        ax.set_title('Best Match WD: Centre vs Tiled')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: % Change
        ax = axes[1]
        for c, condition in enumerate(conditions_with_data):
            change = cvt[condition].get('WD_change_pct', 0)
            color = 'green' if change < 0 else 'red'
            ax.bar(c, change, color=color)
        ax.axhline(0, color='k', lw=1)
        ax.set_xticks(range(n_conds))
        ax.set_xticklabels(conditions_with_data, rotation=45)
        ax.set_ylabel('% Change (Tiled vs Centre)')
        ax.set_title('Which approach matches better?')
        ax.grid(True, alpha=0.3)

        # Panel 3: Summary
        ax = axes[2]
        ax.axis('off')
        summary = ["=== CENTRE VS TILED ===", ""]
        for condition in conditions_with_data:
            wd_c = cvt[condition].get('WD_centre', np.nan)
            wd_t = cvt[condition].get('WD_tiled', np.nan)
            winner = 'Centre' if wd_c < wd_t else 'Tiled'
            change = cvt[condition].get('WD_change_pct', 0)
            summary.append(f"{condition}: {winner} wins ({change:+.1f}%)")
        ax.text(0.05, 0.95, '\n'.join(summary), transform=ax.transAxes,
                va='top', fontsize=10, family='monospace')

        fig.suptitle('Centre Crop vs Tiled: Which Matches Better?', fontweight='bold')
        fig.tight_layout()
        self._save_figure(fig, 'centre_vs_tiled_comparison')

    # =========================================================================
    # BEST-MATCH PARAMETERS: CENTRE VS TILED (MATLAB Figure 17)
    # =========================================================================

    def plot_best_match_params_centre_vs_tiled(self):
        """
        Best-match parameter comparison between Centre and Tiled approaches.
        MATLAB Figure 17 (in centre_vs_tiled mode).
        Shows paired parameter values for each condition.
        """
        if 'centreVsTiled' not in self.results:
            print("  Skipping best-match params centre vs tiled (not in centre_vs_tiled mode)")
            return

        cvt = self.results['centreVsTiled']
        conditions_with_data = [c for c in self.conditions if c in cvt]

        if len(conditions_with_data) == 0:
            print("  Skipping best-match params centre vs tiled (no data)")
            return

        # Check if parameter data is available
        if 'IsingData' not in self.results or 'params' not in self.results['IsingData']:
            print("  Skipping best-match params centre vs tiled (no Ising params)")
            return

        n_conds = len(conditions_with_data)
        params = self.results['IsingData']['params']
        param_names = ['beta', 'c', 'decay_const', 'inhibition_range', 'bias']
        param_labels = ['Beta', 'c', 'Decay Const', 'Inhib Range', 'Bias']

        fig, axes = plt.subplots(2, 3, figsize=(14, 10))
        axes = axes.flatten()

        # Collect parameters for all conditions
        params_centre = np.zeros((n_conds, len(param_names)))
        params_tiled = np.zeros((n_conds, len(param_names)))

        for c, condition in enumerate(conditions_with_data):
            best_idx_centre = cvt[condition].get('best_idx_centre', 0)
            best_idx_tiled = cvt[condition].get('best_idx_tiled', 0)

            for p, pname in enumerate(param_names):
                if pname in params:
                    params_centre[c, p] = params[pname][best_idx_centre]
                    params_tiled[c, p] = params[pname][best_idx_tiled]

        # Create paired comparison plot for each parameter
        np.random.seed(42)  # Fixed seed for reproducibility
        for p in range(len(param_names)):
            ax = axes[p]

            # Compute jitter amount based on parameter range
            all_vals = np.concatenate([params_centre[:, p], params_tiled[:, p]])
            val_range = np.ptp(all_vals)
            if val_range == 0:
                val_range = abs(np.mean(all_vals)) * 0.1 if np.mean(all_vals) != 0 else 1
            jitter_amount = 0.2 * val_range

            # Plot each condition as a line connecting Centre to Tiled
            for c, condition in enumerate(conditions_with_data):
                x_vals = [1, 2]  # 1 = Centre, 2 = Tiled
                jitter_y = jitter_amount * (np.random.rand() - 0.5)
                y_vals = [params_centre[c, p] + jitter_y, params_tiled[c, p] + jitter_y]

                color = self._get_color(condition)
                ax.plot(x_vals, y_vals, '-', color=color, linewidth=2, label=condition if p == 0 else None)
                ax.scatter(x_vals, y_vals, s=100, c=[color], edgecolors='k', zorder=5)

            ax.set_ylabel(param_labels[p])
            ax.set_title(param_labels[p])
            ax.set_xlim(0.5, 2.5)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Centre', 'Tiled'])
            ax.grid(True, alpha=0.3)

            if p == 0:
                ax.legend(loc='best')

        # Panel 6: Summary table
        ax = axes[5]
        ax.axis('off')
        table_text = ["=== BEST-MATCH PARAMETERS ===", ""]

        for c, condition in enumerate(conditions_with_data):
            best_idx_centre = cvt[condition].get('best_idx_centre', 0)
            best_idx_tiled = cvt[condition].get('best_idx_tiled', 0)
            sim_id_centre = cvt[condition].get('sim_id_centre', str(best_idx_centre))
            sim_id_tiled = cvt[condition].get('sim_id_tiled', str(best_idx_tiled))

            table_text.append(f"{condition.upper()}:")
            table_text.append(f"  Centre ({sim_id_centre}):")
            table_text.append(f"    beta={params_centre[c, 0]:.1f}, c={int(params_centre[c, 1])}, decay={int(params_centre[c, 2])}")
            table_text.append(f"    inhib={int(params_centre[c, 3])}, bias={params_centre[c, 4]:.1f}")
            table_text.append(f"  Tiled ({sim_id_tiled}):")
            table_text.append(f"    beta={params_tiled[c, 0]:.1f}, c={int(params_tiled[c, 1])}, decay={int(params_tiled[c, 2])}")
            table_text.append(f"    inhib={int(params_tiled[c, 3])}, bias={params_tiled[c, 4]:.1f}")
            table_text.append("")

        ax.text(0.05, 0.95, '\n'.join(table_text), transform=ax.transAxes,
                va='top', fontsize=8, family='monospace')
        ax.set_title('Parameter Summary')

        fig.suptitle('Best-Match Ising Parameters: Centre Crop vs Tiled Pooled', fontweight='bold')
        fig.tight_layout()
        self._save_figure(fig, 'best_match_params_centre_vs_tiled')

    # =========================================================================
    # POOLED MATCHING COMPARISON (MATLAB Figure 16)
    # =========================================================================

    def plot_pooled_matching_comparison(self):
        """
        Compare P1-based matching vs Pooled-based matching.
        MATLAB Figure 16. Only applicable when gridMode='subselect_tiled'.
        """
        if 'PooledMatching' not in self.results:
            print("  Skipping pooled matching comparison (not in tiled mode)")
            return

        pm = self.results['PooledMatching']
        conditions_with_data = [c for c in self.conditions if c in pm]

        if len(conditions_with_data) == 0:
            print("  Skipping pooled matching comparison (no data)")
            return

        n_conds = len(conditions_with_data)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel 1: Best-match WD comparison
        ax = axes[0, 0]
        wd_p1 = []
        wd_pooled = []
        for condition in conditions_with_data:
            wd_p1.append(pm[condition].get('WD_P1_best', np.nan))
            wd_pooled.append(pm[condition].get('WD_pooled_best', np.nan))

        x = np.arange(n_conds)
        ax.bar(x - 0.2, wd_p1, 0.4, label='P1-based', color='gray')
        ax.bar(x + 0.2, wd_pooled, 0.4, label='Pooled-based', color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions_with_data, rotation=45)
        ax.set_ylabel('Wasserstein Distance')
        ax.set_title('Best Match WD: P1 vs Pooled Selection')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: Rank scatter (Top 20)
        ax = axes[0, 1]
        top_n = 20
        for c, condition in enumerate(conditions_with_data):
            p1_ranks = np.arange(1, top_n + 1)
            pooled_ranks = pm[condition].get('rank_comparison', p1_ranks)
            if isinstance(pooled_ranks, np.ndarray):
                pooled_ranks = pooled_ranks[:top_n]
            else:
                pooled_ranks = p1_ranks
            ax.scatter(p1_ranks, pooled_ranks[:len(p1_ranks)],
                      c=[self._get_color(condition)],
                      alpha=0.7, s=50, label=condition)
        ax.plot([1, top_n], [1, top_n], 'k--', lw=1.5, label='Identity')
        ax.set_xlabel('Rank (P1-based)')
        ax.set_ylabel('Rank (Pooled-based)')
        ax.set_title('Rank Comparison (Top 20)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 3: Match details
        ax = axes[1, 0]
        ax.axis('off')
        text_lines = ["=== BEST MATCH COMPARISON ===", ""]
        for condition in conditions_with_data:
            changed = pm[condition].get('best_match_changed', False)
            status = "CHANGED" if changed else "SAME"
            text_lines.append(f"{condition.upper()}:")
            text_lines.append(f"  P1-best idx: {pm[condition].get('best_P1_idx', 'N/A')}")
            text_lines.append(f"  Pooled-best idx: {pm[condition].get('best_pooled_idx', 'N/A')}")
            text_lines.append(f"  Status: {status}")
            text_lines.append("")
        ax.text(0.05, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
                va='top', fontsize=9, family='monospace')
        ax.set_title('Best Match Details')

        # Panel 4: Summary
        ax = axes[1, 1]
        ax.axis('off')
        n_changed = sum(1 for cond in conditions_with_data
                       if pm[cond].get('best_match_changed', False))
        summary = [
            "=== SUMMARY ===",
            "",
            f"Conditions analyzed: {n_conds}",
            f"Best match changed: {n_changed}/{n_conds}",
            "",
            "Interpretation:",
            "  If best match changes frequently,",
            "  pooling provides additional info",
            "  that refines model selection."
        ]
        ax.text(0.05, 0.95, '\n'.join(summary), transform=ax.transAxes,
                va='top', fontsize=9, family='monospace')
        ax.set_title('Summary')

        fig.suptitle('Pooled vs P1-Based Matching: Does Best Match Change?', fontweight='bold')
        fig.tight_layout()
        self._save_figure(fig, 'pooled_matching_comparison')

    # =========================================================================
    # TOP 10 PARAMETERS TABLE (MATLAB Figure XX)
    # =========================================================================

    def plot_top10_parameters_table(self):
        """
        Display top 10 best-matching Ising parameters for each condition.
        MATLAB Figure XX.
        """
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        params = ising_data.get('params', {})
        if not params:
            print("  No parameter data for top 10 table")
            return

        n_conds = len(self.conditions)
        n_cols = min(n_conds, 3)
        n_rows = -(-n_conds // n_cols)  # ceiling division
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        axes = np.atleast_2d(axes).flatten()

        for c, condition in enumerate(self.conditions):
            ax = axes[c]
            ax.axis('off')

            if condition not in comparison:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(condition, fontweight='bold')
                continue

            # Get top 10 matches
            best_idx_list = self._get_best_idx(condition)
            n_matches = min(10, len(best_idx_list))

            if n_matches == 0:
                ax.text(0.5, 0.5, 'No matches', ha='center', va='center')
                ax.set_title(condition, fontweight='bold')
                continue

            top_idx = best_idx_list[:n_matches]

            # Build table
            header = f"{'Rank':>4} {'Beta':>8} {'c':>4} {'Decay':>8} {'Inhib':>6} {'Bias':>8}"
            separator = "-" * len(header)
            rows = [header, separator]

            beta_arr = np.array(params.get('beta', []))
            c_arr = np.array(params.get('c', []))
            decay_arr = np.array(params.get('decay_const', []))
            inhib_arr = np.array(params.get('inhibition_range', []))
            bias_arr = np.array(params.get('bias', []))

            for i, idx in enumerate(top_idx):
                beta_val = beta_arr[idx] if idx < len(beta_arr) else np.nan
                c_val = c_arr[idx] if idx < len(c_arr) else np.nan
                decay_val = decay_arr[idx] if idx < len(decay_arr) else np.nan
                inhib_val = inhib_arr[idx] if idx < len(inhib_arr) else np.nan
                bias_val = bias_arr[idx] if idx < len(bias_arr) else np.nan

                row = f"{i+1:>4} {beta_val:>8.3f} {int(c_val) if not np.isnan(c_val) else 'N/A':>4} "
                row += f"{int(decay_val) if not np.isnan(decay_val) else 'N/A':>8} "
                row += f"{int(inhib_val) if not np.isnan(inhib_val) else 'N/A':>6} "
                row += f"{bias_val:>8.3f}"
                rows.append(row)

            ax.text(0.5, 0.5, '\n'.join(rows), transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, family='monospace')
            ax.set_title(condition, fontweight='bold')

        # Hide unused subplots
        for c in range(n_conds, n_rows * n_cols):
            axes[c].set_visible(False)

        fig.suptitle('Top 10 Best-Matching Ising Parameters per Condition', fontweight='bold')
        fig.tight_layout()
        self._save_figure(fig, 'top10_parameters_table')

    # =========================================================================
    # NEW VISUALIZATIONS
    # =========================================================================

    def plot_radar_match_quality(self):
        """Radar/spider plot of match quality per condition across metrics."""
        comparison = self.results.get('Comparison', {})
        dynamics = self.results.get('DynamicsAnalysis', {})

        metric_names = ["MI WD", "Activity WD", "Blob WD", "Tau ratio"]
        conditions_with_data = [c for c in self.conditions if c in comparison]
        if len(conditions_with_data) == 0:
            print("  No comparison data for radar plot")
            return

        # Collect per-condition metric values (lower = better match)
        data = {}
        for condition in conditions_with_data:
            comp = comparison[condition]
            wd = comp.get('wasserstein_dist', np.array([]))
            best_idx = self._get_best_idx(condition)
            if len(best_idx) == 0 or len(wd) == 0:
                continue

            mi_wd = wd[best_idx[0]] if len(wd) > best_idx[0] else np.nan

            # Activity WD (if stored separately)
            act_wd = comp.get('activity_wasserstein_dist', np.array([]))
            if hasattr(act_wd, '__len__') and len(act_wd) > best_idx[0]:
                act_wd_val = act_wd[best_idx[0]]
            else:
                act_wd_val = mi_wd  # fallback to combined WD

            # Blob persistence WD
            blob_wd = comp.get('blob_wasserstein_dist', np.array([]))
            if hasattr(blob_wd, '__len__') and len(blob_wd) > best_idx[0]:
                blob_wd_val = blob_wd[best_idx[0]]
            else:
                blob_wd_val = np.nan

            # Tau ratio
            tau_ratio = np.nan
            if condition in dynamics:
                da = dynamics[condition]
                exp_tau = da.get('exp_tau', np.nan)
                ising_tau = da.get('best_ising_tau', np.nan)
                if not np.isnan(exp_tau) and not np.isnan(ising_tau) and exp_tau > 0:
                    tau_ratio = abs(1.0 - ising_tau / exp_tau)  # 0 = perfect match

            data[condition] = [mi_wd, act_wd_val, blob_wd_val, tau_ratio]

        if len(data) == 0:
            print("  No data for radar plot")
            return

        # Normalize each metric to [0, 1] across conditions (0 = best)
        all_vals = np.array(list(data.values()))
        col_max = np.nanmax(all_vals, axis=0)
        col_max[col_max == 0] = 1.0

        n_metrics = len(metric_names)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # close polygon

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE, FIG_WIDTH_SINGLE),
                               subplot_kw=dict(polar=True))

        for condition, vals in data.items():
            normalized = [v / m if not np.isnan(v) else 0 for v, m in zip(vals, col_max)]
            normalized += normalized[:1]
            ax.plot(angles, normalized, linewidth=1.0, label=condition,
                    color=self._get_color(condition))
            ax.fill(angles, normalized, alpha=0.15, color=self._get_color(condition))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_title('Match Quality (lower = better)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()
        self._save_figure(fig, 'radar_match_quality')

    def plot_condition_parameter_overlap(self):
        """NxN heatmap of Jaccard index of top-10 best-match simulation indices."""
        comparison = self.results.get('Comparison', {})

        conditions_with_data = [c for c in self.conditions if c in comparison
                                and len(self._get_best_idx(c)) > 0]
        if len(conditions_with_data) < 2:
            print("  Need at least 2 conditions for overlap matrix")
            return

        n = len(conditions_with_data)
        jaccard = np.zeros((n, n))

        for i, ci in enumerate(conditions_with_data):
            idx_i = set(self._get_best_idx(ci)[:10].tolist())
            for j, cj in enumerate(conditions_with_data):
                idx_j = set(self._get_best_idx(cj)[:10].tolist())
                intersection = len(idx_i & idx_j)
                union = len(idx_i | idx_j)
                jaccard[i, j] = intersection / union if union > 0 else 0

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE, FIG_WIDTH_SINGLE))
        im = ax.imshow(jaccard, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(conditions_with_data, rotation=45, ha='right')
        ax.set_yticks(range(n))
        ax.set_yticklabels(conditions_with_data)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{jaccard[i, j]:.2f}', ha='center', va='center',
                        fontsize=6, color='white' if jaccard[i, j] > 0.5 else 'black')

        plt.colorbar(im, ax=ax, label='Jaccard Index', shrink=0.8)
        ax.set_title('Top-10 Parameter Overlap')
        fig.tight_layout()
        self._save_figure(fig, 'condition_parameter_overlap')

    def plot_phase_portrait(self):
        """2D density: Activity (x) vs Moran's I (y) for Exp and Ising."""
        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        conditions_with_data = [c for c in self.conditions
                                if c in exp_stats and c in comparison]
        if len(conditions_with_data) == 0:
            print("  No data for phase portrait")
            return

        n = len(conditions_with_data)
        fig, axes = plt.subplots(1, n, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))
        if n == 1:
            axes = [axes]

        for i, condition in enumerate(conditions_with_data):
            ax = axes[i]

            exp_mi = np.asarray(exp_stats[condition].get('MoransI_all', [])).ravel()
            exp_act = np.asarray(exp_stats[condition].get('Activity_all', [])).ravel()

            best_idx = self._get_best_idx(condition)

            if len(exp_mi) == 0 or len(exp_act) == 0:
                ax.set_visible(False)
                continue

            # Pair up: use minimum length
            min_len = min(len(exp_mi), len(exp_act))
            exp_mi = exp_mi[:min_len]
            exp_act = exp_act[:min_len]

            # Remove NaN pairs
            valid = ~(np.isnan(exp_mi) | np.isnan(exp_act))
            exp_mi = exp_mi[valid]
            exp_act = exp_act[valid]

            if len(exp_mi) < 10:
                ax.set_visible(False)
                continue

            # Experimental density contours
            try:
                from scipy.stats import gaussian_kde
                xy_exp = np.vstack([exp_act, exp_mi])
                kde_exp = gaussian_kde(xy_exp)
                xgrid = np.linspace(exp_act.min(), exp_act.max(), 50)
                ygrid = np.linspace(exp_mi.min(), exp_mi.max(), 50)
                X, Y = np.meshgrid(xgrid, ygrid)
                Z_exp = kde_exp(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                ax.contour(X, Y, Z_exp, levels=5, colors=[self._get_color(condition)],
                           linewidths=0.75, alpha=0.8)
            except Exception:
                ax.scatter(exp_act, exp_mi, s=2, alpha=0.3, color=self._get_color(condition))

            # Ising density contours (gray)
            if len(best_idx) > 0:
                ising_mi_all = ising_data.get('MoransI_all', [])
                ising_act_all = ising_data.get('Activity_all', [])
                if len(ising_mi_all) > best_idx[0] and len(ising_act_all) > best_idx[0]:
                    ising_mi = np.asarray(ising_mi_all[best_idx[0]]).ravel()
                    ising_act = np.asarray(ising_act_all[best_idx[0]]).ravel()
                    min_len_i = min(len(ising_mi), len(ising_act))
                    ising_mi = ising_mi[:min_len_i]
                    ising_act = ising_act[:min_len_i]
                    valid_i = ~(np.isnan(ising_mi) | np.isnan(ising_act))
                    ising_mi = ising_mi[valid_i]
                    ising_act = ising_act[valid_i]

                    if len(ising_mi) >= 10:
                        try:
                            xy_ising = np.vstack([ising_act, ising_mi])
                            kde_ising = gaussian_kde(xy_ising)
                            Z_ising = kde_ising(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                            ax.contour(X, Y, Z_ising, levels=5, colors='gray',
                                       linewidths=0.75, alpha=0.6, linestyles='--')
                        except Exception:
                            ax.scatter(ising_act, ising_mi, s=2, alpha=0.2, color='gray')

            ax.set_xlabel('Activity')
            ax.set_ylabel("Moran's I")
            ax.set_title(condition)

        fig.suptitle('Phase Portrait: Activity vs Moran\'s I')
        fig.tight_layout()
        self._save_figure(fig, 'phase_portrait')

    def plot_summary_table(self):
        """Formatted table summarizing match quality per condition."""
        comparison = self.results.get('Comparison', {})
        dynamics = self.results.get('DynamicsAnalysis', {})
        exp_stats = self.results.get('ExpStats', {})
        ising_data = self.results.get('IsingData', {})

        conditions_with_data = [c for c in self.conditions if c in comparison]
        if len(conditions_with_data) == 0:
            print("  No comparison data for summary table")
            return

        col_labels = ['WD(MI)', 'WD(Act)', 'WD(comb)', 'Tau ratio', 'KS p-val']
        table_data = []

        for condition in conditions_with_data:
            comp = comparison[condition]
            best_idx = self._get_best_idx(condition)
            wd = comp.get('wasserstein_dist', np.array([]))

            if len(best_idx) == 0:
                table_data.append(['--'] * len(col_labels))
                continue

            bi = best_idx[0]

            # MI WD
            mi_wd = f'{wd[bi]:.4f}' if len(wd) > bi else '--'

            # Activity WD
            act_wd_arr = comp.get('activity_wasserstein_dist', np.array([]))
            act_wd = f'{act_wd_arr[bi]:.4f}' if hasattr(act_wd_arr, '__len__') and len(act_wd_arr) > bi else '--'

            # Combined WD
            comb_wd = mi_wd  # use MI WD as combined if no separate field

            # Tau ratio
            tau_str = '--'
            if condition in dynamics:
                da = dynamics[condition]
                exp_tau = da.get('exp_tau', np.nan)
                ising_tau = da.get('best_ising_tau', np.nan)
                if not np.isnan(exp_tau) and not np.isnan(ising_tau) and exp_tau > 0:
                    tau_str = f'{ising_tau / exp_tau:.2f}'

            # KS test p-value
            ks_str = '--'
            if condition in exp_stats:
                exp_mi = exp_stats[condition].get('MoransI_all', np.array([]))
                ising_mi_all = ising_data.get('MoransI_all', [])
                if len(exp_mi) > 0 and len(ising_mi_all) > bi:
                    ising_mi = np.asarray(ising_mi_all[bi]).ravel()
                    exp_mi_clean = exp_mi[~np.isnan(exp_mi)]
                    ising_mi_clean = ising_mi[~np.isnan(ising_mi)]
                    if len(exp_mi_clean) > 0 and len(ising_mi_clean) > 0:
                        ks_stat, ks_p = scipy_stats.ks_2samp(exp_mi_clean, ising_mi_clean)
                        ks_str = f'{ks_p:.2e}'

            table_data.append([mi_wd, act_wd, comb_wd, tau_str, ks_str])

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))
        ax.axis('off')

        table = ax.table(cellText=table_data, rowLabels=conditions_with_data,
                         colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.4)

        # Color row labels
        for i, condition in enumerate(conditions_with_data):
            cell = table[i + 1, -1]  # row label cell
            cell.set_text_props(color=self._get_color(condition), fontweight='bold')

        ax.set_title('Goodness-of-Fit Summary', pad=20)
        fig.tight_layout()
        self._save_figure(fig, 'summary_table')

    def plot_blob_survival_curves(self):
        """Kaplan-Meier-style survival curves for blob lifetimes."""
        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        conditions_with_data = [c for c in self.conditions
                                if c in exp_stats and c in comparison]
        if len(conditions_with_data) == 0:
            print("  No data for blob survival curves")
            return

        n = len(conditions_with_data)
        fig, axes = plt.subplots(1, n, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))
        if n == 1:
            axes = [axes]

        any_plotted = False
        for i, condition in enumerate(conditions_with_data):
            ax = axes[i]

            exp_lt = exp_stats[condition].get('BlobPersistence_lifetimes', None)
            if exp_lt is None or len(exp_lt) == 0:
                ax.set_visible(False)
                continue

            exp_lt = np.asarray(exp_lt).ravel()
            exp_lt = exp_lt[~np.isnan(exp_lt)]
            if len(exp_lt) == 0:
                ax.set_visible(False)
                continue

            # Compute survival function: fraction with lifetime >= t
            def survival_fn(lifetimes):
                lifetimes = np.sort(lifetimes)
                n_total = len(lifetimes)
                unique_t = np.unique(lifetimes)
                surv_t = np.array([np.sum(lifetimes >= t) / n_total for t in unique_t])
                return unique_t, surv_t

            t_exp, s_exp = survival_fn(exp_lt)
            ax.step(t_exp, s_exp, where='post', color=self._get_color(condition),
                    linewidth=1.0, label='Exp')

            # Ising survival
            best_idx = self._get_best_idx(condition)
            ising_bp = ising_data.get('BlobPersistence_lifetimes', [])
            if len(best_idx) > 0 and len(ising_bp) > best_idx[0]:
                ising_lt = np.asarray(ising_bp[best_idx[0]]).ravel()
                ising_lt = ising_lt[~np.isnan(ising_lt)]
                if len(ising_lt) > 0:
                    t_ising, s_ising = survival_fn(ising_lt)
                    ax.step(t_ising, s_ising, where='post', color='gray',
                            linewidth=1.0, linestyle='--', label='Ising')

            ax.set_xlabel('Lifetime')
            ax.set_ylabel('Fraction surviving')
            ax.set_title(condition)
            ax.legend()
            ax.set_ylim(0, 1.05)
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            print("  No blob data available for survival curves")
            return

        fig.suptitle('Blob Persistence Survival Curves')
        fig.tight_layout()
        self._save_figure(fig, 'blob_survival_curves')

    # =========================================================================
    # NEW VARIANT METHODS (Step 4)
    # =========================================================================

    def plot_morans_i_distributions_kde(self, raw=False):
        """KDE + rug plot for Moran's I distributions."""
        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        n_conditions = len(self.conditions)
        fig, axes = plt.subplots(1, n_conditions, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))
        if n_conditions == 1:
            axes = [axes]

        for i, condition in enumerate(self.conditions):
            ax = axes[i]

            if condition not in exp_stats or condition not in comparison:
                ax.set_visible(False)
                continue

            exp_mi = exp_stats[condition].get('MoransI_all', np.array([]))
            if len(exp_mi) == 0:
                ax.set_visible(False)
                continue
            exp_mi = exp_mi[~np.isnan(exp_mi)]

            best_idx = self._get_best_idx(condition)

            # KDE for experimental
            from scipy.stats import gaussian_kde
            clip_min, clip_max = -0.5, 1.0
            x_grid = np.linspace(clip_min, clip_max, 200)

            if len(exp_mi) >= 2:
                kde_exp = gaussian_kde(exp_mi)
                y_exp = kde_exp(x_grid)
                ax.fill_between(x_grid, y_exp, alpha=0.3, color=self._get_color(condition))
                ax.plot(x_grid, y_exp, color=self._get_color(condition), linewidth=1.0,
                        label='Exp')
                ax.axvline(np.mean(exp_mi), color=self._get_color(condition),
                           linestyle='--', linewidth=0.75)
                # Rug
                ax.plot(exp_mi, np.zeros_like(exp_mi) - 0.02 * y_exp.max(), '|',
                        color=self._get_color(condition), markersize=3, alpha=0.3)

            # KDE for Ising
            if len(best_idx) > 0:
                ising_mi_all = ising_data.get('MoransI_all', [])
                if len(ising_mi_all) > best_idx[0]:
                    ising_mi = np.asarray(ising_mi_all[best_idx[0]]).ravel()
                    ising_mi = ising_mi[~np.isnan(ising_mi)]
                    if len(ising_mi) >= 2:
                        kde_ising = gaussian_kde(ising_mi)
                        y_ising = kde_ising(x_grid)
                        ax.fill_between(x_grid, y_ising, alpha=0.2, color='gray')
                        ax.plot(x_grid, y_ising, color='gray', linewidth=1.0,
                                label='Ising')
                        ax.axvline(np.mean(ising_mi), color='gray',
                                   linestyle='--', linewidth=0.75)

                        # WD annotation
                        if raw:
                            raw_wd = self._compute_raw_combined_wd(comparison[condition])
                            if raw_wd is not None and len(raw_wd) > best_idx[0]:
                                ax.text(0.98, 0.95, f'Raw WD={raw_wd[best_idx[0]]:.3f}',
                                        transform=ax.transAxes, ha='right', va='top',
                                        fontsize=6)
                        else:
                            wd = comparison[condition].get('wasserstein_dist', np.array([]))
                            if len(wd) > best_idx[0]:
                                ax.text(0.98, 0.95, f'WD={wd[best_idx[0]]:.3f}',
                                        transform=ax.transAxes, ha='right', va='top',
                                        fontsize=6)

            ax.set_xlabel("Moran's I")
            ax.set_ylabel('Density')
            ax.set_title(condition)
            ax.set_xlim(clip_min, clip_max)
            ax.legend()

        wd_label = 'raw WD' if raw else 'KDE'
        fig.suptitle(f"Moran's I Distribution ({wd_label})")
        fig.tight_layout()
        suffix = '_raw' if raw else ''
        self._save_figure(fig, f'morans_i_distributions_kde{suffix}')

    def plot_morans_i_distributions_kde_raw(self):
        """KDE + rug plot with raw WD annotation."""
        return self.plot_morans_i_distributions_kde(raw=True)

    def plot_activity_distributions_kde(self):
        """KDE + rug plot for Activity distributions."""
        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        n_conditions = len(self.conditions)
        fig, axes = plt.subplots(1, n_conditions, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))
        if n_conditions == 1:
            axes = [axes]

        for i, condition in enumerate(self.conditions):
            ax = axes[i]

            if condition not in exp_stats or condition not in comparison:
                ax.set_visible(False)
                continue

            exp_act = exp_stats[condition].get('Activity_all', np.array([]))
            if len(exp_act) == 0:
                ax.set_visible(False)
                continue
            exp_act = exp_act[~np.isnan(exp_act)]

            best_idx = self._get_best_idx(condition)

            from scipy.stats import gaussian_kde
            clip_min, clip_max = 0.0, 1.0
            x_grid = np.linspace(clip_min, clip_max, 200)

            if len(exp_act) >= 2:
                kde_exp = gaussian_kde(exp_act)
                y_exp = kde_exp(x_grid)
                ax.fill_between(x_grid, y_exp, alpha=0.3, color=self._get_color(condition))
                ax.plot(x_grid, y_exp, color=self._get_color(condition), linewidth=1.0,
                        label='Exp')
                ax.axvline(np.mean(exp_act), color=self._get_color(condition),
                           linestyle='--', linewidth=0.75)
                ax.plot(exp_act, np.zeros_like(exp_act) - 0.02 * y_exp.max(), '|',
                        color=self._get_color(condition), markersize=3, alpha=0.3)

            if len(best_idx) > 0:
                ising_act_all = ising_data.get('Activity_all', [])
                if len(ising_act_all) > best_idx[0]:
                    ising_act = np.asarray(ising_act_all[best_idx[0]]).ravel()
                    ising_act = ising_act[~np.isnan(ising_act)]
                    if len(ising_act) >= 2:
                        kde_ising = gaussian_kde(ising_act)
                        y_ising = kde_ising(x_grid)
                        ax.fill_between(x_grid, y_ising, alpha=0.2, color='gray')
                        ax.plot(x_grid, y_ising, color='gray', linewidth=1.0,
                                label='Ising')
                        ax.axvline(np.mean(ising_act), color='gray',
                                   linestyle='--', linewidth=0.75)

            ax.set_xlabel('Activity')
            ax.set_ylabel('Density')
            ax.set_title(condition)
            ax.set_xlim(0, 0.4)
            ax.legend()

        fig.suptitle('Activity Distribution (KDE)')
        fig.tight_layout()
        self._save_figure(fig, 'activity_distributions_kde')

    def plot_exp_vs_ising_dotline(self):
        """Paired dot-line plot of Exp vs Ising means per metric.

        Produces standard and masked variants.
        """
        self._plot_exp_vs_ising_dotline_impl(use_masked=False)
        exp_stats = self.results.get('ExpStats', {})
        if any('Activity_mean_masked' in exp_stats.get(c, {}) for c in self.conditions):
            self._plot_exp_vs_ising_dotline_impl(use_masked=True)

    def _plot_exp_vs_ising_dotline_impl(self, use_masked=False):
        """Core implementation for plot_exp_vs_ising_dotline."""
        exp_stats = self.results.get('ExpStats', {})
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})

        metrics = ["Moran's I", "Activity", "BlobPersistence"]
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))

        for m_idx, metric in enumerate(metrics):
            ax = axes[m_idx]
            conditions_plotted = []

            for c_idx, condition in enumerate(self.conditions):
                if condition not in exp_stats or condition not in comparison:
                    continue
                best_idx = self._get_best_idx(condition)
                if len(best_idx) == 0:
                    continue

                bi = best_idx[0]

                if metric == "Moran's I":
                    exp_val = exp_stats[condition].get('MoransI_mean', np.nan)
                    ising_means = ising_data.get('MoransI_mean', [])
                    ising_val = ising_means[bi] if len(ising_means) > bi else np.nan
                elif metric == "Activity":
                    if use_masked:
                        exp_val = exp_stats[condition].get('Activity_mean_masked', np.nan)
                    else:
                        exp_val = exp_stats[condition].get('Activity_mean', np.nan)
                    ising_means = ising_data.get('Activity_mean', [])
                    ising_val = ising_means[bi] if len(ising_means) > bi else np.nan
                else:
                    if 'BlobPersistence_mean' in exp_stats[condition]:
                        exp_val = exp_stats[condition]['BlobPersistence_mean']
                    else:
                        exp_lt = exp_stats[condition].get('BlobPersistence_lifetimes', None)
                        min_lt = self.config.get('blob_min_lifetime',
                                                  self.results.get('config', {}).get('blob_min_lifetime', 1))
                        if exp_lt is not None and np.ndim(exp_lt) > 0 and len(exp_lt) > 0:
                            lt_arr = np.asarray(exp_lt).ravel()
                            lt_filtered = lt_arr[lt_arr >= min_lt]
                            exp_val = np.nanmean(lt_filtered) if len(lt_filtered) > 0 else np.nanmean(lt_arr)
                        else:
                            exp_val = np.nan
                    ising_bp = ising_data.get('BlobPersistence_lifetimes', [])
                    if np.ndim(ising_bp) > 0 and len(ising_bp) > bi:
                        bp_el = ising_bp[bi]
                        if bp_el is not None and np.ndim(bp_el) > 0 and len(bp_el) > 0:
                            ising_val = np.nanmean(bp_el)
                        elif bp_el is not None and np.ndim(bp_el) == 0:
                            ising_val = float(bp_el)
                        else:
                            ising_val = np.nan
                    else:
                        ising_val = np.nan
                    # Apply temporal scaling to match what the matching algorithm used
                    if not np.isnan(ising_val):
                        scale_factors = self._get_temporal_scale(condition)
                        ising_val *= scale_factors[bi]

                if np.isnan(exp_val):
                    continue

                x_pos = len(conditions_plotted)
                ax.plot([x_pos, x_pos], [exp_val, ising_val], color='gray',
                        linewidth=0.5, zorder=1)
                ax.scatter(x_pos, exp_val, color=self._get_color(condition),
                           s=40, zorder=2, edgecolors='black', linewidth=0.5)
                if not np.isnan(ising_val):
                    ax.scatter(x_pos, ising_val, color='gray', s=40, zorder=2,
                               edgecolors='black', linewidth=0.5)

                conditions_plotted.append(condition)

            if len(conditions_plotted) > 0:
                ax.set_xticks(range(len(conditions_plotted)))
                ax.set_xticklabels(conditions_plotted, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(metric)

        if len(axes) > 0:
            from matplotlib.lines import Line2D
            axes[0].legend(handles=[
                Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                       markeredgecolor='black', markersize=5, label='Exp'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markeredgecolor='black', markersize=5, label='Ising'),
            ], fontsize=6)

        suffix = ' (masked)' if use_masked else ''
        fig.suptitle(f'Exp vs Ising: Paired Dot-Line{suffix}')
        fig.tight_layout()
        save_name = 'exp_vs_ising_dotline_masked' if use_masked else 'exp_vs_ising_dotline'
        self._save_figure(fig, save_name)

    def plot_time_constants_dotline(self):
        """Paired dot-line plot for time constants with reference line at ratio=1."""
        dynamics = self.results.get('DynamicsAnalysis', {})

        conditions_with_data = []
        exp_taus = []
        ising_taus = []

        for condition in self.conditions:
            if condition in dynamics:
                da = dynamics[condition]
                et = da.get('exp_tau', np.nan)
                it = da.get('best_ising_tau', np.nan)
                if not np.isnan(et):
                    conditions_with_data.append(condition)
                    exp_taus.append(et)
                    ising_taus.append(it)

        if len(conditions_with_data) == 0:
            print("  No time constant data for dot-line plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_1_5, FIG_HEIGHT_UNIT))

        # Panel 1: Tau values
        ax = axes[0]
        for i, (cond, et, it) in enumerate(zip(conditions_with_data, exp_taus, ising_taus)):
            ax.plot([i, i], [et, it], color='gray', linewidth=0.5, zorder=1)
            ax.scatter(i, et, color=self._get_color(cond), s=40, zorder=2,
                       edgecolors='black', linewidth=0.5)
            if not np.isnan(it):
                ax.scatter(i, it, color='gray', s=40, zorder=2,
                           edgecolors='black', linewidth=0.5)
        ax.set_xticks(range(len(conditions_with_data)))
        ax.set_xticklabels(conditions_with_data, rotation=45, ha='right')
        ax.set_ylabel('Tau')
        ax.set_title('Time Constants')

        # Panel 2: Tau ratio
        ax = axes[1]
        tau_ratios = [it / et if et > 0 and not np.isnan(it) else np.nan
                      for et, it in zip(exp_taus, ising_taus)]
        for i, (cond, ratio) in enumerate(zip(conditions_with_data, tau_ratios)):
            if not np.isnan(ratio):
                ax.scatter(i, ratio, color=self._get_color(cond), s=40,
                           edgecolors='black', linewidth=0.5, zorder=2)
        ax.axhline(1, color='black', linestyle='--', linewidth=0.5)
        ax.set_xticks(range(len(conditions_with_data)))
        ax.set_xticklabels(conditions_with_data, rotation=45, ha='right')
        ax.set_ylabel('Tau Ratio (Ising/Exp)')
        ax.set_title('Tau Ratio')

        fig.suptitle('Time Constants: Paired Dot-Line')
        fig.tight_layout()
        self._save_figure(fig, 'time_constants_dotline')

    # =========================================================================
    # DISTRIBUTIONS EXTENDED
    # =========================================================================

    def plot_wd_rank_curve(self, raw=False, top_n=None):
        """WD vs match rank — one tile per condition with sub-metric overlays."""
        comparison = self.results.get('Comparison', {})
        n_conditions = len(self.conditions)
        fig, axes = plt.subplots(1, n_conditions, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))
        if n_conditions == 1:
            axes = [axes]

        legend_added = False
        for i, condition in enumerate(self.conditions):
            ax = axes[i]
            if condition not in comparison:
                ax.set_visible(False)
                continue

            cond_data = comparison[condition]
            if raw:
                wd = self._compute_raw_combined_wd(cond_data)
                if wd is None:
                    ax.set_visible(False)
                    continue
            else:
                wd = np.atleast_1d(np.asarray(cond_data.get('wasserstein_dist', []), dtype=float))
            rankings = self._get_rankings_for(cond_data, raw)
            if len(wd) == 0 or len(rankings) == 0:
                ax.set_visible(False)
                continue

            if top_n is not None:
                rankings = rankings[:top_n]

            n_sims = len(wd)
            n_ranked = len(rankings)
            ranks = np.arange(1, n_ranked + 1)
            sorted_wd = wd[rankings] if len(rankings) <= n_sims else np.sort(wd)[:n_ranked]

            wd_type = 'Raw Combined WD' if raw else 'Combined WD'
            ax.plot(ranks, sorted_wd, color=self._get_color(condition), linewidth=1.2,
                    label=wd_type, zorder=3)

            # Sub-metric overlays
            indiv = cond_data.get('individual_dists', {})
            handles_sub = []
            active_keys = self._get_active_metric_keys()
            for metric_key in active_keys:
                label = SUBMETRIC_LABELS.get(metric_key, metric_key)
                metric_wd = indiv.get(metric_key, None)
                if metric_wd is None:
                    continue
                metric_wd = np.atleast_1d(np.asarray(metric_wd, dtype=float))
                if len(metric_wd) != n_sims:
                    continue
                sorted_metric = metric_wd[rankings] if len(rankings) <= n_sims else np.sort(metric_wd)[:n_ranked]
                h, = ax.plot(ranks, sorted_metric, color=SUBMETRIC_COLORS.get(metric_key, 'gray'),
                             alpha=0.35, linewidth=0.7, label=label)
                handles_sub.append(h)

            # Mark best match boundary
            best_idx = self._get_best_idx(condition, raw=raw)
            if len(best_idx) > 0:
                n_top = len(best_idx)
                ax.axvline(n_top, color='gray', linestyle='--', linewidth=0.6, alpha=0.7)
                ax.axvspan(0.5, n_top + 0.5, alpha=0.06, color=self._get_color(condition))

            ax.set_xlabel('Rank')
            ylabel = 'Raw WD' if raw else 'WD'
            ax.set_ylabel(ylabel)
            min_wd = np.nanmin(wd) if len(wd) > 0 else 0
            ax.set_title(f'{condition}\nMin {ylabel}={min_wd:.4f}')

            if not legend_added and handles_sub:
                ax.legend(fontsize=5, loc='upper left')
                legend_added = True

        title = 'Raw Wasserstein Distance vs Rank' if raw else 'Wasserstein Distance vs Rank'
        if top_n is not None:
            title += f' (Top {top_n})'
        fig.suptitle(title)
        fig.tight_layout()
        suffix = '_raw' if raw else ''
        top_suffix = f'_top{top_n}' if top_n is not None else ''
        self._save_figure(fig, f'wd_rank_curve{suffix}{top_suffix}')

    def plot_wd_rank_curve_raw(self):
        """WD rank curve with raw (un-normalized) WD."""
        return self.plot_wd_rank_curve(raw=True)

    def plot_wd_rank_curve_top100(self):
        """WD rank curve zoomed to top 100."""
        return self.plot_wd_rank_curve(top_n=100)

    def plot_wd_rank_curve_top100_raw(self):
        """WD rank curve zoomed to top 100 with raw WD."""
        return self.plot_wd_rank_curve(raw=True, top_n=100)

    def plot_wd_rank_curve_overlay(self, raw=False, top_n=None):
        """All conditions overlaid on a single axis — combined WD only."""
        comparison = self.results.get('Comparison', {})
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE, FIG_HEIGHT_UNIT))

        any_plotted = False
        for condition in self.conditions:
            if condition not in comparison:
                continue
            cond_data = comparison[condition]
            if raw:
                wd = self._compute_raw_combined_wd(cond_data)
                if wd is None:
                    continue
            else:
                wd = np.atleast_1d(np.asarray(cond_data.get('wasserstein_dist', []), dtype=float))
            rankings = self._get_rankings_for(cond_data, raw)
            if len(wd) == 0:
                continue

            if top_n is not None:
                rankings = rankings[:top_n]

            n_sims = len(wd)
            n_ranked = len(rankings)
            ranks = np.arange(1, n_ranked + 1)
            sorted_wd = wd[rankings] if len(rankings) <= n_sims else np.sort(wd)[:n_ranked]

            ax.plot(ranks, sorted_wd, color=self._get_color(condition),
                    linewidth=1.0, label=condition)
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            return

        ax.set_xlabel('Rank')
        ylabel = 'Raw Wasserstein Distance' if raw else 'Wasserstein Distance'
        ax.set_ylabel(ylabel)
        title = 'Raw WD Rank Curves — All Conditions' if raw else 'WD Rank Curves — All Conditions'
        if top_n is not None:
            title += f' (Top {top_n})'
        ax.set_title(title)
        ax.legend(fontsize=6)
        fig.tight_layout()
        suffix = '_raw' if raw else ''
        top_suffix = f'_top{top_n}' if top_n is not None else ''
        self._save_figure(fig, f'wd_rank_curve_overlay{suffix}{top_suffix}')

    def plot_wd_rank_curve_overlay_raw(self):
        """WD rank curve overlay with raw (un-normalized) WD."""
        return self.plot_wd_rank_curve_overlay(raw=True)

    def plot_wd_rank_curve_overlay_top100(self):
        """WD rank curve overlay zoomed to top 100."""
        return self.plot_wd_rank_curve_overlay(top_n=100)

    def plot_wd_rank_curve_overlay_top100_raw(self):
        """WD rank curve overlay zoomed to top 100 with raw WD."""
        return self.plot_wd_rank_curve_overlay(raw=True, top_n=100)

    def plot_wd_rank_curve_overlay_top500(self):
        """WD rank curve overlay zoomed to top 500."""
        return self.plot_wd_rank_curve_overlay(top_n=500)

    def plot_wd_rank_curve_overlay_top500_raw(self):
        """WD rank curve overlay zoomed to top 500 with raw WD."""
        return self.plot_wd_rank_curve_overlay(raw=True, top_n=500)

    def plot_wd_rank_curve_overlay_top1000(self):
        """WD rank curve overlay zoomed to top 1000."""
        return self.plot_wd_rank_curve_overlay(top_n=1000)

    def plot_wd_rank_curve_overlay_top1000_raw(self):
        """WD rank curve overlay zoomed to top 1000 with raw WD."""
        return self.plot_wd_rank_curve_overlay(raw=True, top_n=1000)

    def plot_wd_rank_curve_overlay_top5000(self):
        """WD rank curve overlay zoomed to top 5000."""
        return self.plot_wd_rank_curve_overlay(top_n=5000)

    def plot_wd_rank_curve_overlay_top5000_raw(self):
        """WD rank curve overlay zoomed to top 5000 with raw WD."""
        return self.plot_wd_rank_curve_overlay(raw=True, top_n=5000)

    def plot_intercondition_wd_matrix(self, top_n=20, metric='MoransI_all', raw=False):
        """Sim-to-sim WD heatmap for each condition pair's top-N."""
        from matplotlib.patches import Rectangle
        comparison = self.results.get('Comparison', {})

        # Get conditions with data
        conds_with_data = [c for c in self.conditions if c in comparison
                           and len(self._get_best_idx(c, raw=raw)) > 0]
        if len(conds_with_data) < 2:
            return

        # All pairs
        pairs = [(conds_with_data[a], conds_with_data[b])
                 for a in range(len(conds_with_data))
                 for b in range(a + 1, len(conds_with_data))]
        n_pairs = len(pairs)
        n_cols = min(3, n_pairs)
        n_rows = int(np.ceil(n_pairs / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT * 1.5 * n_rows))
        if n_pairs == 1:
            axes = np.array([[axes]])
        axes = np.atleast_2d(axes)

        for p, (condA, condB) in enumerate(pairs):
            r, c = divmod(p, n_cols)
            ax = axes[r, c]

            idx_a = self._get_best_idx(condA, raw=raw)[:top_n]
            idx_b = self._get_best_idx(condB, raw=raw)[:top_n]
            if len(idx_a) == 0 or len(idx_b) == 0:
                ax.set_visible(False)
                continue

            wd_matrix = self._compute_sim_to_sim_wd(idx_a, idx_b, metric=metric)
            im = ax.imshow(wd_matrix, cmap='viridis_r', aspect='auto')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Highlight overlapping indices
            set_a = set(idx_a.tolist())
            set_b = set(idx_b.tolist())
            overlap = set_a & set_b
            n_overlap = len(overlap)
            for oi in overlap:
                if oi in idx_a.tolist() and oi in idx_b.tolist():
                    ri = idx_a.tolist().index(oi)
                    ci_idx = idx_b.tolist().index(oi)
                    ax.add_patch(Rectangle((ci_idx - 0.5, ri - 0.5), 1, 1,
                                           fill=False, edgecolor='red', linewidth=1.2))

            n_ticks = min(5, len(idx_a))
            tick_pos = np.linspace(0, len(idx_a) - 1, n_ticks, dtype=int)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_pos + 1)
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_pos + 1)
            ax.set_xlabel(f'{condB} rank')
            ax.set_ylabel(f'{condA} rank')
            ax.set_title(f'{condA} vs {condB}\noverlap: {n_overlap}/{top_n}')

        # Hide unused subplots
        for p in range(n_pairs, n_rows * n_cols):
            r, c = divmod(p, n_cols)
            axes[r, c].set_visible(False)

        raw_prefix = 'Raw ' if raw else ''
        fig.suptitle(f'{raw_prefix}Inter-condition Sim-to-Sim WD (top {top_n}, {metric})')
        fig.tight_layout()
        suffix = '_raw' if raw else ''
        self._save_figure(fig, f'intercondition_wd_matrix{suffix}')

    def plot_intercondition_wd_matrix_raw(self, top_n=20, metric='MoransI_all'):
        """Inter-condition WD matrix with raw (un-normalized) rankings."""
        return self.plot_intercondition_wd_matrix(top_n=top_n, metric=metric, raw=True)

    def plot_topn_wd_to_all_conditions(self, top_n=20, raw=False):
        """Top-N sims of each condition evaluated against all conditions' WD."""
        comparison = self.results.get('Comparison', {})

        conds_with_data = [c for c in self.conditions if c in comparison
                           and len(self._get_best_idx(c, raw=raw)) > 0]
        if len(conds_with_data) == 0:
            return

        wd_prefix = 'Raw ' if raw else ''
        save_suffix = '_raw' if raw else ''

        # --- Heatmap per focal condition ---
        for focal in conds_with_data:
            best_idx = self._get_best_idx(focal, raw=raw)[:top_n]
            if len(best_idx) == 0:
                continue

            matrix = np.full((len(best_idx), len(conds_with_data)), np.nan)
            for j, target in enumerate(conds_with_data):
                if raw:
                    wd = self._compute_raw_combined_wd(comparison[target])
                    if wd is None:
                        continue
                else:
                    wd = np.atleast_1d(np.asarray(
                        comparison[target].get('wasserstein_dist', []), dtype=float))
                for i, idx in enumerate(best_idx):
                    if idx < len(wd):
                        matrix[i, j] = wd[idx]

            fig, ax = plt.subplots(figsize=(FIG_WIDTH_1_5, FIG_HEIGHT_UNIT * 2))
            im = ax.imshow(matrix, cmap='viridis_r', aspect='auto')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(range(len(conds_with_data)))
            ax.set_xticklabels(conds_with_data, rotation=45, ha='right')
            ax.set_ylabel(f'Top-{len(best_idx)} rank ({focal})')
            ax.set_yticks(range(len(best_idx)))
            ax.set_yticklabels(range(1, len(best_idx) + 1))

            # Annotate cells
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if not np.isnan(matrix[i, j]):
                        ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                                fontsize=4, color='white' if matrix[i, j] > np.nanmedian(matrix) else 'black')

            ax.set_title(f'Top-{len(best_idx)} of {focal} — {wd_prefix}WD to all conditions')
            fig.tight_layout()
            self._save_figure(fig, f'topn_wd_{focal}{save_suffix}')

        # --- Bar chart summary ---
        n_conds = len(conds_with_data)
        fig, axes = plt.subplots(1, n_conds, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))
        if n_conds == 1:
            axes = [axes]

        for i, focal in enumerate(conds_with_data):
            ax = axes[i]
            best_idx = self._get_best_idx(focal, raw=raw)[:top_n]
            if len(best_idx) == 0:
                ax.set_visible(False)
                continue

            means = []
            sems = []
            for target in conds_with_data:
                if raw:
                    wd = self._compute_raw_combined_wd(comparison[target])
                    if wd is None:
                        wd = np.array([])
                else:
                    wd = np.atleast_1d(np.asarray(
                        comparison[target].get('wasserstein_dist', []), dtype=float))
                vals = wd[best_idx[best_idx < len(wd)]]
                vals = vals[~np.isnan(vals)]
                means.append(np.mean(vals) if len(vals) > 0 else 0)
                sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

            x = np.arange(n_conds)
            colors = [self._get_color(c) for c in conds_with_data]
            ax.bar(x, means, yerr=sems, color=colors, alpha=0.8, capsize=2)
            ax.set_xticks(x)
            ax.set_xticklabels(conds_with_data, rotation=45, ha='right')
            ax.set_ylabel(f'Mean {wd_prefix}WD')
            ax.set_title(f"Top-{len(best_idx)} of {focal}")

        fig.suptitle(f'Top-N {wd_prefix}WD to All Conditions')
        fig.tight_layout()
        self._save_figure(fig, f'topn_wd_summary_bar{save_suffix}')

    def plot_topn_wd_to_all_conditions_raw(self, top_n=20):
        """Top-N WD to all conditions with raw (un-normalized) WD."""
        return self.plot_topn_wd_to_all_conditions(top_n=top_n, raw=True)

    def plot_metric_decomposition(self, top_n=10, raw=False):
        """Stacked bar: which metrics drive each condition's match."""
        comparison = self.results.get('Comparison', {})
        conds_with_data = [c for c in self.conditions if c in comparison]
        if len(conds_with_data) == 0:
            return

        # Collect per-metric mean distances for top-N of each condition
        metric_keys = self._get_active_metric_keys()
        # Build matrix: (n_conds, n_metrics)
        raw_mat = np.full((len(conds_with_data), len(metric_keys)), np.nan)
        for ci, cond in enumerate(conds_with_data):
            cond_data = comparison[cond]
            best_idx = self._get_best_idx(cond, raw=raw)[:top_n]
            indiv = cond_data.get('individual_dists', {})
            for mi, mk in enumerate(metric_keys):
                vals = indiv.get(mk, None)
                if vals is None:
                    continue
                vals = np.atleast_1d(np.asarray(vals, dtype=float))
                if len(best_idx) > 0 and len(vals) > 0:
                    valid = best_idx[best_idx < len(vals)]
                    raw_mat[ci, mi] = np.nanmean(vals[valid]) if len(valid) > 0 else np.nan

        if np.all(np.isnan(raw_mat)):
            return

        if raw:
            # Use raw values directly (no normalization)
            plot_mat = np.nan_to_num(raw_mat, nan=0.0)
        else:
            # Min-max normalize each metric column for comparability
            plot_mat = raw_mat.copy()
            for mi in range(plot_mat.shape[1]):
                col = plot_mat[:, mi]
                mn, mx = np.nanmin(col), np.nanmax(col)
                if mx > mn:
                    plot_mat[:, mi] = (col - mn) / (mx - mn)
                else:
                    plot_mat[:, mi] = 0.0
            plot_mat = np.nan_to_num(plot_mat, nan=0.0)

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_1_5, FIG_HEIGHT_UNIT))
        x = np.arange(len(conds_with_data))
        bottom = np.zeros(len(conds_with_data))
        for mi, mk in enumerate(metric_keys):
            color = SUBMETRIC_COLORS.get(mk, 'gray')
            label = SUBMETRIC_LABELS.get(mk, mk)
            ax.bar(x, plot_mat[:, mi], bottom=bottom, color=color, label=label, alpha=0.85)
            bottom += plot_mat[:, mi]

        ax.set_xticks(x)
        ax.set_xticklabels(conds_with_data, rotation=45, ha='right')
        ylabel = 'Raw WD contribution' if raw else 'Normalized WD contribution'
        ax.set_ylabel(ylabel)
        title_prefix = 'Raw ' if raw else ''
        ax.set_title(f'{title_prefix}Metric Decomposition (top {top_n})')
        ax.legend(fontsize=5, loc='upper right')
        fig.tight_layout()
        suffix = '_raw' if raw else ''
        self._save_figure(fig, f'metric_decomposition{suffix}')

    def plot_metric_decomposition_raw(self, top_n=10):
        """Metric decomposition with raw (un-normalized) WD."""
        return self.plot_metric_decomposition(top_n=top_n, raw=True)

    def plot_rank_concordance(self, raw=False):
        """Rank correlation heatmap + scatter grid between conditions."""
        comparison = self.results.get('Comparison', {})
        conds_with_data = [c for c in self.conditions if c in comparison
                           and len(self._get_rankings_for(comparison[c], raw)) > 0]
        if len(conds_with_data) < 2:
            return

        # Build rank vectors per condition
        rank_vectors = {}
        for cond in conds_with_data:
            rankings = self._get_rankings_for(comparison[cond], raw)
            n_sims = len(rankings)
            # Convert ranking (sorted indices) to rank array
            rank_arr = np.empty(n_sims, dtype=int)
            rank_arr[rankings] = np.arange(n_sims)
            rank_vectors[cond] = rank_arr

        n_conds = len(conds_with_data)

        # --- Heatmap ---
        rho_matrix = np.eye(n_conds)
        for i in range(n_conds):
            for j in range(i + 1, n_conds):
                r_i = rank_vectors[conds_with_data[i]]
                r_j = rank_vectors[conds_with_data[j]]
                n = min(len(r_i), len(r_j))
                rho, _ = scipy_stats.spearmanr(r_i[:n], r_j[:n])
                rho_matrix[i, j] = rho
                rho_matrix[j, i] = rho

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE * 1.5, FIG_WIDTH_SINGLE * 1.5))
        im = ax.imshow(rho_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n_conds))
        ax.set_xticklabels(conds_with_data, rotation=45, ha='right')
        ax.set_yticks(range(n_conds))
        ax.set_yticklabels(conds_with_data)
        for i in range(n_conds):
            for j in range(n_conds):
                ax.text(j, i, f'{rho_matrix[i, j]:.2f}', ha='center', va='center',
                        fontsize=6, color='black' if abs(rho_matrix[i, j]) < 0.5 else 'white')
        raw_prefix = 'Raw ' if raw else ''
        suffix = '_raw' if raw else ''
        ax.set_title(f'{raw_prefix}Rank Concordance (Spearman)')
        fig.tight_layout()
        self._save_figure(fig, f'rank_concordance_heatmap{suffix}')

        # --- Scatter grid (upper triangle) ---
        fig, axes = plt.subplots(n_conds, n_conds,
                                 figsize=(FIG_WIDTH_FULL, FIG_WIDTH_FULL))
        if n_conds == 1:
            axes = np.array([[axes]])

        for i in range(n_conds):
            for j in range(n_conds):
                ax = axes[i, j]
                if i >= j:
                    ax.set_visible(False)
                    continue

                r_i = rank_vectors[conds_with_data[i]]
                r_j = rank_vectors[conds_with_data[j]]
                n = min(len(r_i), len(r_j))

                if n > 2000:
                    ax.hexbin(r_i[:n], r_j[:n], gridsize=30, cmap='Blues', mincnt=1)
                else:
                    ax.scatter(r_i[:n], r_j[:n], s=1, alpha=0.15, color='gray', rasterized=True)
                    # Highlight top-50
                    top_mask = (r_i[:n] < 50) | (r_j[:n] < 50)
                    ax.scatter(r_i[:n][top_mask], r_j[:n][top_mask], s=4, alpha=0.6,
                               color='crimson', zorder=3)

                rho = rho_matrix[i, j]
                ax.set_title(f'{conds_with_data[i]} vs {conds_with_data[j]}\nrho={rho:.2f}',
                             fontsize=5)
                ax.tick_params(labelsize=4)

        fig.suptitle(f'{raw_prefix}Rank Concordance Scatter', fontsize=8)
        fig.tight_layout()
        self._save_figure(fig, f'rank_concordance_scatter{suffix}')

    def plot_rank_concordance_raw(self):
        """Rank concordance with raw (un-normalized) rankings."""
        return self.plot_rank_concordance(raw=True)

    def plot_leave_one_metric_out(self, top_n=10, raw=False):
        """Ranking robustness when each metric is dropped."""
        comparison = self.results.get('Comparison', {})
        metric_keys = self._get_active_metric_keys()

        conds_with_data = [c for c in self.conditions if c in comparison]
        if len(conds_with_data) == 0:
            return

        n_conditions = len(conds_with_data)
        fig, axes = plt.subplots(1, n_conditions, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT))
        if n_conditions == 1:
            axes = [axes]

        for ci, cond in enumerate(conds_with_data):
            ax = axes[ci]
            cond_data = comparison[cond]
            indiv = cond_data.get('individual_dists', {})
            best_idx = self._get_best_idx(cond, raw=raw)[:top_n]
            original_top = set(best_idx.tolist()) if len(best_idx) > 0 else set()

            if len(original_top) == 0 or len(indiv) == 0:
                ax.set_visible(False)
                continue

            jaccards = []
            drop_labels = []
            for drop_mk in metric_keys:
                remaining = [mk for mk in metric_keys if mk != drop_mk]
                remaining_with_data = [mk for mk in remaining if mk in indiv]
                if len(remaining_with_data) == 0:
                    jaccards.append(np.nan)
                    drop_labels.append(SUBMETRIC_LABELS.get(drop_mk, drop_mk))
                    continue

                # Recompute combined WD with remaining metrics
                # Equal weights among active metrics
                n_active = len(metric_keys)
                w = {mk: 1.0 / n_active for mk in remaining_with_data}
                w_sum = sum(w.values())
                if w_sum == 0:
                    w_sum = 1.0

                n_sims = None
                metric_arrays = {}
                for mk in remaining_with_data:
                    arr = np.atleast_1d(np.asarray(indiv[mk], dtype=float))
                    metric_arrays[mk] = arr
                    if n_sims is None:
                        n_sims = len(arr)

                if n_sims is None or n_sims == 0:
                    jaccards.append(np.nan)
                    drop_labels.append(SUBMETRIC_LABELS.get(drop_mk, drop_mk))
                    continue

                combined = np.zeros(n_sims)
                for mk in remaining_with_data:
                    arr = metric_arrays[mk]
                    if len(arr) == n_sims:
                        combined += (w[mk] / w_sum) * arr

                new_rankings = np.argsort(combined)
                new_top = set(new_rankings[:top_n].tolist())

                if len(original_top | new_top) > 0:
                    jaccard = len(original_top & new_top) / len(original_top | new_top)
                else:
                    jaccard = 0.0
                jaccards.append(jaccard)
                drop_labels.append(SUBMETRIC_LABELS.get(drop_mk, drop_mk))

            x = np.arange(len(drop_labels))
            colors = [SUBMETRIC_COLORS.get(mk, 'gray') for mk in metric_keys]
            ax.bar(x, jaccards, color=colors, alpha=0.8)
            ax.plot(x, jaccards, 'k-', linewidth=0.7, marker='o', markersize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(drop_labels, rotation=45, ha='right', fontsize=5)
            ax.set_ylabel('Jaccard overlap')
            ax.set_ylim(0, 1.05)
            ax.set_title(f'{cond}')

        raw_prefix = 'Raw ' if raw else ''
        suffix = '_raw' if raw else ''
        fig.suptitle(f'{raw_prefix}Leave-One-Metric-Out (top {top_n})')
        fig.tight_layout()
        self._save_figure(fig, f'leave_one_metric_out{suffix}')

    def plot_leave_one_metric_out_raw(self, top_n=10):
        """Leave-one-metric-out with raw (un-normalized) rankings."""
        return self.plot_leave_one_metric_out(top_n=top_n, raw=True)

    def plot_parameter_corner_plot(self, top_n=20, raw=False):
        """5x5 pairwise scatter of Ising parameters for each condition's best matches."""
        comparison = self.results.get('Comparison', {})
        ising_data = self.results.get('IsingData', {})
        params_data = ising_data.get('params', {})
        param_names = ['beta', 'c', 'decay_const', 'inhibition_range', 'bias']

        # Validate param data exists
        param_arrays = {}
        for pn in param_names:
            arr = params_data.get(pn, None)
            if arr is None:
                return
            param_arrays[pn] = np.atleast_1d(np.asarray(arr, dtype=float))

        conds_with_data = [c for c in self.conditions if c in comparison
                           and len(self._get_best_idx(c, raw=raw)) > 0]
        if len(conds_with_data) == 0:
            return

        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, n_params,
                                 figsize=(FIG_WIDTH_FULL, FIG_WIDTH_FULL))

        for row in range(n_params):
            for col in range(n_params):
                ax = axes[row, col]

                if col > row:
                    # Upper triangle: hide
                    ax.set_visible(False)
                    continue

                p_row = param_arrays[param_names[row]]
                p_col = param_arrays[param_names[col]]

                if row == col:
                    # Diagonal: KDE per condition
                    for cond in conds_with_data:
                        best_idx = self._get_best_idx(cond, raw=raw)[:top_n]
                        valid = best_idx[best_idx < len(p_row)]
                        if len(valid) < 2:
                            continue
                        vals = p_row[valid]
                        try:
                            kde = scipy_stats.gaussian_kde(vals)
                            x_grid = np.linspace(np.min(p_row), np.max(p_row), 100)
                            ax.plot(x_grid, kde(x_grid), color=self._get_color(cond),
                                    linewidth=1.0, label=cond)
                            ax.fill_between(x_grid, kde(x_grid), alpha=0.15,
                                            color=self._get_color(cond))
                        except Exception:
                            ax.hist(vals, bins=10, alpha=0.3, color=self._get_color(cond),
                                    density=True, label=cond)
                    if row == 0:
                        ax.legend(fontsize=4, loc='upper right')
                else:
                    # Lower triangle: scatter
                    # Background: all sims in light gray
                    ax.scatter(p_col, p_row, s=0.5, alpha=0.08, color='lightgray',
                               rasterized=True, zorder=1)

                    for cond in conds_with_data:
                        best_idx = self._get_best_idx(cond, raw=raw)[:top_n]
                        valid = best_idx[(best_idx < len(p_row)) & (best_idx < len(p_col))]
                        if len(valid) == 0:
                            continue
                        ax.scatter(p_col[valid], p_row[valid], s=8, alpha=0.7,
                                   color=self._get_color(cond), edgecolors='none',
                                   zorder=2, label=cond)

                # Labels
                if row == n_params - 1:
                    ax.set_xlabel(param_names[col], fontsize=5)
                else:
                    ax.set_xticklabels([])
                if col == 0:
                    ax.set_ylabel(param_names[row], fontsize=5)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=4)

        raw_prefix = 'Raw ' if raw else ''
        suffix = '_raw' if raw else ''
        fig.suptitle(f'{raw_prefix}Parameter Corner Plot (top {top_n})', fontsize=8)
        fig.tight_layout()
        self._save_figure(fig, f'parameter_corner_plot{suffix}')

    def plot_parameter_corner_plot_raw(self, top_n=20):
        """Parameter corner plot with raw (un-normalized) rankings."""
        return self.plot_parameter_corner_plot(top_n=top_n, raw=True)

    def plot_rank_gap_evolution(self, top_n=50, raw=False):
        """Condition specificity vs rank depth."""
        comparison = self.results.get('Comparison', {})
        conds_with_data = [c for c in self.conditions if c in comparison
                           and len(self._get_rankings_for(comparison[c], raw)) > 0]
        if len(conds_with_data) == 0:
            return

        n_conditions = len(conds_with_data)
        fig, axes = plt.subplots(1, n_conditions,
                                 figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_UNIT * 1.4))
        if n_conditions == 1:
            axes = [axes]

        for ci, focal in enumerate(conds_with_data):
            ax = axes[ci]
            focal_data = comparison[focal]
            rankings = self._get_rankings_for(focal_data, raw)
            if raw:
                focal_wd = self._compute_raw_combined_wd(focal_data)
                if focal_wd is None:
                    ax.set_visible(False)
                    continue
            else:
                focal_wd = np.atleast_1d(np.asarray(focal_data.get('wasserstein_dist', []), dtype=float))

            n_use = min(top_n, len(rankings))
            if n_use == 0:
                ax.set_visible(False)
                continue

            top_indices = rankings[:n_use]
            ranks = np.arange(1, n_use + 1)

            # Focal condition: solid line
            focal_vals = focal_wd[top_indices]
            ax.plot(ranks, focal_vals, color=self._get_color(focal), linewidth=1.5,
                    label=focal, zorder=3)

            # Other conditions: dashed lines
            for other in conds_with_data:
                if other == focal:
                    continue
                if raw:
                    other_wd = self._compute_raw_combined_wd(comparison[other])
                    if other_wd is None:
                        continue
                else:
                    other_wd = np.atleast_1d(np.asarray(
                        comparison[other].get('wasserstein_dist', []), dtype=float))
                if len(other_wd) == 0:
                    continue
                # Use same sim indices, lookup other condition's WD
                valid_mask = top_indices < len(other_wd)
                other_vals = np.full(n_use, np.nan)
                other_vals[valid_mask] = other_wd[top_indices[valid_mask]]
                ax.plot(ranks, other_vals, color=self._get_color(other), linewidth=0.8,
                        linestyle='--', alpha=0.7, label=other)

            ax.set_xlabel('Rank')
            ylabel = 'Raw WD' if raw else 'WD'
            ax.set_ylabel(ylabel)
            ax.set_title(f'{focal} top-{n_use}')
            if ci == 0:
                ax.legend(fontsize=5, loc='upper left')

        wd_label = 'Raw WD' if raw else 'WD'
        fig.suptitle(f'Rank Gap Evolution ({wd_label}) — Condition Specificity')
        fig.tight_layout()
        suffix = '_raw' if raw else ''
        self._save_figure(fig, f'rank_gap_evolution{suffix}')

    def plot_rank_gap_evolution_raw(self, top_n=50):
        """Rank gap evolution with raw (un-normalized) WD."""
        return self.plot_rank_gap_evolution(top_n=top_n, raw=True)

    def plot_rank_gap_evolution_top100(self):
        """Rank gap evolution zoomed to top 100."""
        return self.plot_rank_gap_evolution(top_n=100)

    def plot_rank_gap_evolution_top100_raw(self):
        """Rank gap evolution zoomed to top 100 with raw WD."""
        return self.plot_rank_gap_evolution(top_n=100, raw=True)

    # =========================================================================
    # MAIN PLOTTING FUNCTION
    # =========================================================================

    def plot_all(self):
        """Generate all visualization figures in categorical subfolders.

        Produces two variants for each category:
        - base/: 4 base conditions (Naive, Beginner, Expert, NoSpout)
        - with_hit_miss/: all 6 conditions (+ Expert_Hit, Expert_Miss)
          Only generated if hit/miss data is present.
        """
        print("\nGenerating visualizations...")

        all_conditions = list(self._all_conditions)
        base_conditions = [c for c in all_conditions if c in BASE_CONDITIONS]

        # Build variant list: always base, optionally with_hit_miss
        # When frame_label is set, prefix variant names (e.g. 'prestim_base')
        prefix = f"{self._frame_label}_" if self._frame_label else ""
        label_prefix = f"{self._frame_label} " if self._frame_label else ""
        variants = [(f'{prefix}base', base_conditions, f'{label_prefix}4 base conditions')]
        if self._has_hit_miss_data():
            variants.append((f'{prefix}with_hit_miss', all_conditions,
                             f'{label_prefix}all 6 conditions'))

        for category_name, methods in PLOT_CATEGORIES:
            for variant_suffix, cond_list, variant_label in variants:
                subfolder = os.path.join(category_name, variant_suffix)
                print(f"\n  {category_name} ({variant_label})...")
                self._run_category(category_name, methods, cond_list, subfolder)
            # Clear between categories so hit_miss variant doesn't reuse
            # base-variant embeddings (different condition sets)
            self._embedding_cache = {}
            gc.collect()

        # Restore original state
        self.conditions = list(self._all_conditions)
        self._current_subfolder = ''
        self._embedding_cache = {}

        print(f"\nVisualization complete. Figures saved to: {self.output_dir}")


# =============================================================================
# STANDALONE PLOTTING FUNCTIONS (for use without class)
# =============================================================================

def plot_comparison_results(results_file: str, output_dir: str = None,
                            ising_data_path: str = None, exp_data_path: str = None):
    """
    Load results file and generate all visualizations.

    Parameters
    ----------
    results_file : str
        Path to HDF5 results file from Figure5_IsingComparison_optimized.py
    output_dir : str, optional
        Output directory for figures
    ising_data_path : str, optional
        Path to Ising simulation data directory (for spatial snapshots)
    exp_data_path : str, optional
        Path to experimental data file (for spatial snapshots)
    """
    import h5py

    print(f"Loading results from: {results_file}")

    # Load results
    results = {}
    with h5py.File(results_file, 'r') as f:
        def load_group(group):
            d = {}
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Group):
                    d[key] = load_group(item)
                else:
                    d[key] = item[()]
            return d
        results = load_group(f)

    # MATLAB HDF5 files may have data nested under 'Results' key
    if 'Results' in results and isinstance(results['Results'], dict):
        results = results['Results']

    # Extract config
    config = results.get('config', {})
    if 'conditions' not in config:
        config['conditions'] = ['Naive', 'Beginner', 'Expert', 'NoSpout']

    # Determine frame_label from saved config
    frame_sel = config.get('exp_frame_selection', 'prestim')
    frame_label = 'prestim' if frame_sel == 'prestim' else 'full_trial'

    if output_dir is None:
        output_dir = os.path.dirname(results_file)

    # Create visualizer and generate plots
    viz = IsingVisualizer(results, config, output_dir,
                          ising_data_path=ising_data_path,
                          exp_data_path=exp_data_path,
                          frame_label=frame_label)
    viz.plot_all()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate Ising comparison visualizations')
    parser.add_argument('results_file', type=str, help='Path to HDF5 results file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for figures')
    args = parser.parse_args()

    plot_comparison_results(args.results_file, args.output_dir)
