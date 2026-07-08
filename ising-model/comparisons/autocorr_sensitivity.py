#!/usr/bin/env python3
"""
Autocorrelation Fit Range Sensitivity Analysis
===============================================

Port of MATLAB analyze_decay_fitRange_sensitivity.m to Python.
Analyzes how varying the decay curve fit range affects the extracted tau
(decay time constant) and conversion factors.

Usage:
    from autocorr_sensitivity import analyze_fit_range_sensitivity, plot_sensitivity_results

    results = analyze_fit_range_sensitivity(ising_data, exp_stats, comparison, config)
    plot_sensitivity_results(results, config, output_dir)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import warnings
import os


# =============================================================================
# CONDITION COLORS (matching MATLAB and ising_visualizations.py)
# =============================================================================

CONDITION_COLORS = {
    'Naive': (0.3373, 0.7059, 0.9137),       # Light blue
    'Beginner': (0.8431, 0.2549, 0.6078),    # Magenta/pink
    'Expert': (0.0, 0.6196, 0.4510),         # Teal/green
    'NoSpout': (0.8353, 0.3686, 0.0),        # Orange
}

EXP_COLOR = (0.2, 0.2, 0.2)  # Dark gray for experimental


# =============================================================================
# FIT FUNCTIONS
# =============================================================================

def fit_exponential_decay(acf: np.ndarray, lags: np.ndarray,
                          fit_range: Tuple[int, int]) -> Tuple[float, dict]:
    """
    Fit simple exponential decay to autocorrelation.

    Fits the model: acf = exp(-lag / tau) using log-linear regression.
    This is equivalent to fitting: log(acf) = -lag / tau

    Parameters
    ----------
    acf : np.ndarray
        Autocorrelation values
    lags : np.ndarray
        Lag values (same length as acf)
    fit_range : tuple
        (min_lag, max_lag) range to use for fitting

    Returns
    -------
    tau : float
        Decay time constant
    fit_result : dict
        Dict with fit statistics: R2, slope, intercept, method
    """
    acf = np.asarray(acf).ravel()
    lags = np.asarray(lags).ravel()

    # Select fit range
    mask = (lags >= fit_range[0]) & (lags <= fit_range[1])
    acf_fit = acf[mask]
    lags_fit = lags[mask]

    # Handle negative/zero values (need positive for log)
    valid = acf_fit > 0
    if np.sum(valid) < 3:
        return np.nan, {'R2': np.nan, 'method': 'insufficient_positive_values'}

    log_acf = np.log(acf_fit[valid])
    lags_valid = lags_fit[valid]

    # Linear regression: log(acf) = -lag/tau
    # Slope = -1/tau, so tau = -1/slope
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coeffs = np.polyfit(lags_valid, log_acf, 1)

    slope = coeffs[0]
    intercept = coeffs[1]

    if slope >= 0:
        return np.nan, {'R2': np.nan, 'method': 'negative_slope', 'slope': slope}

    tau = -1.0 / slope

    # Compute R-squared (coefficient of determination)
    predicted = coeffs[0] * lags_valid + coeffs[1]
    SS_res = np.sum((log_acf - predicted) ** 2)
    SS_tot = np.sum((log_acf - np.mean(log_acf)) ** 2)

    if SS_tot == 0:
        R2 = np.nan
    else:
        R2 = 1 - SS_res / SS_tot

    return tau, {'R2': R2, 'slope': slope, 'intercept': intercept, 'method': 'log-linear'}


def compute_trial_averaged_acf(activity_trials: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute trial-averaged autocorrelation function.

    Parameters
    ----------
    activity_trials : np.ndarray
        Activity data, shape (n_timepoints, n_trials). Each COLUMN is one trial.
    max_lag : int
        Maximum lag for autocorrelation

    Returns
    -------
    acf_mean : np.ndarray
        Trial-averaged ACF, shape (max_lag + 1,)
    """
    # Input is (n_timepoints, n_trials) - DO NOT transpose based on size heuristic
    # The heuristic fails when n_trials > n_timepoints (e.g., 80 prestim frames, 200 trials)
    if activity_trials.ndim == 1:
        activity_trials = activity_trials.reshape(-1, 1)

    n_timepoints, n_trials = activity_trials.shape
    acf_all = []

    for t in range(n_trials):
        trial_data = activity_trials[:, t]  # Extract COLUMN = one trial's timecourse
        trial_data = trial_data[~np.isnan(trial_data)]

        if len(trial_data) > max_lag:  # Only need > max_lag for valid ACF computation
            # Compute normalized ACF using FFT
            trial_centered = trial_data - np.mean(trial_data)
            n = len(trial_centered)
            fft_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
            fft_ts = np.fft.fft(trial_centered, fft_size)
            acf_full = np.fft.ifft(fft_ts * np.conj(fft_ts)).real
            acf_trial = acf_full[:max_lag + 1] / acf_full[0]
            acf_all.append(acf_trial)

    if len(acf_all) == 0:
        return np.full(max_lag + 1, np.nan)

    return np.mean(np.array(acf_all), axis=0)


# =============================================================================
# MAIN SENSITIVITY ANALYSIS FUNCTION
# =============================================================================

def analyze_fit_range_sensitivity(ising_data: List[dict], exp_stats: dict,
                                   comparison: dict, config: dict) -> dict:
    """
    Analyze how fit range affects tau extraction.

    Port of MATLAB analyze_decay_fitRange_sensitivity.m.

    Parameters
    ----------
    ising_data : list
        List of dicts containing Ising simulation results.
        Each dict should have 'Autocorr_acf' key.
    exp_stats : dict
        Dict with condition-keyed experimental statistics.
        Each condition should have 'Activity_trials' key.
    comparison : dict
        Dict with condition-keyed comparison results.
        Each condition should have 'bestMatch_idx' key.
    config : dict
        Configuration dict with 'conditions', 'autocorr' keys.

    Returns
    -------
    results : dict
        Sensitivity analysis results:
        - fitEndPoints: array of fit range endpoints tested
        - exp_tau: experimental tau for each fit range
        - exp_R2: experimental fit R-squared for each fit range
        - ising_tau: dict per condition, each (n_top x n_fit_ranges) array
        - ising_R2: dict per condition, each (n_top x n_fit_ranges) array
        - conversionFactor: dict per condition, each (n_top x n_fit_ranges) array
    """
    print('\n=== Decay Fit Range Sensitivity Analysis ===')

    # Configuration
    fit_end_points = np.array([5, 10, 15, 20, 30, 40, 50])
    n_fit_ranges = len(fit_end_points)

    max_lag = config.get('autocorr', {}).get('max_lag', 50)
    conditions = config.get('conditions', ['Naive', 'Beginner', 'Expert', 'NoSpout'])
    n_conditions = len(conditions)

    # Initialize results
    results = {
        'fitEndPoints': fit_end_points,
        'maxLag': max_lag,
        'exp_tau': np.zeros(n_fit_ranges),
        'exp_R2': np.zeros(n_fit_ranges),
        'ising_tau': {},
        'ising_R2': {},
        'conversionFactor': {},
    }

    # =========================================================================
    # Experimental Data: Compute trial-averaged ACF and refit for each range
    # =========================================================================

    print(f'Computing experimental tau for {n_fit_ranges} fit ranges...')

    # Collect all trial activity across conditions
    all_acf_trials = []
    valid_trials = 0

    for condition in conditions:
        if condition not in exp_stats:
            continue
        if 'Activity_trials' not in exp_stats[condition]:
            continue

        activity_trials = exp_stats[condition]['Activity_trials']

        # Activity_trials is (n_timepoints, n_trials) - DO NOT auto-transpose
        # The size heuristic fails when n_trials > n_timepoints (e.g., 80 prestim frames, 200 trials)
        if activity_trials.ndim == 1:
            activity_trials = activity_trials.reshape(-1, 1)

        n_timepoints, n_trials = activity_trials.shape

        for t in range(n_trials):
            trial_data = activity_trials[:, t]  # Extract COLUMN = one trial's timecourse
            trial_data = trial_data[~np.isnan(trial_data)]

            if len(trial_data) > max_lag:  # Only need > max_lag for valid ACF computation
                # Compute normalized ACF
                trial_centered = trial_data - np.mean(trial_data)
                n = len(trial_centered)
                fft_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
                fft_ts = np.fft.fft(trial_centered, fft_size)
                acf_full = np.fft.ifft(fft_ts * np.conj(fft_ts)).real
                acf_trial = acf_full[:max_lag + 1] / acf_full[0]
                all_acf_trials.append(acf_trial)
                valid_trials += 1

    if valid_trials == 0:
        print('  Warning: No valid trials found for experimental ACF computation.')
        exp_acf_mean = np.full(max_lag + 1, np.nan)
    else:
        exp_acf_mean = np.mean(np.array(all_acf_trials), axis=0)

    exp_lags = np.arange(max_lag + 1)

    print(f'  Computed ACF from {valid_trials} valid trials across {n_conditions} conditions')

    # Fit with each range
    for f, end_pt in enumerate(fit_end_points):
        fit_range = (1, end_pt)
        tau, fit_result = fit_exponential_decay(exp_acf_mean, exp_lags, fit_range)
        results['exp_tau'][f] = tau
        results['exp_R2'][f] = fit_result.get('R2', np.nan)

    print(f'  Experimental tau: range [{np.nanmin(results["exp_tau"]):.2f}, {np.nanmax(results["exp_tau"]):.2f}]')

    # =========================================================================
    # Ising Data: Refit tau for top 10 matches per condition
    # =========================================================================

    print('Computing Ising tau for top 10 matches per condition...')

    # Check if ACF data is available
    has_acf = False
    for sim in ising_data:
        if 'Autocorr_acf' in sim and sim['Autocorr_acf'] is not None:
            has_acf = True
            break

    if not has_acf:
        print('  Warning: Ising ACF data not available. Cannot compute Ising tau sensitivity.')
        return results

    ising_lags = np.arange(max_lag + 1)

    for condition in conditions:
        if condition not in comparison:
            print(f'  Skipping {condition} (not in Comparison)')
            continue

        # Get top 10 match indices
        best_idx = comparison[condition].get('bestMatch_idx', [])
        if len(best_idx) == 0:
            continue

        n_top = min(10, len(best_idx))
        top_idx = best_idx[:n_top]

        # Initialize matrices
        tau_matrix = np.zeros((n_top, n_fit_ranges))
        R2_matrix = np.zeros((n_top, n_fit_ranges))
        cf_matrix = np.zeros((n_top, n_fit_ranges))

        for m, sim_idx in enumerate(top_idx):
            if sim_idx >= len(ising_data):
                tau_matrix[m, :] = np.nan
                R2_matrix[m, :] = np.nan
                cf_matrix[m, :] = np.nan
                continue

            ising_acf = ising_data[sim_idx].get('Autocorr_acf', None)

            if ising_acf is None or len(ising_acf) == 0 or np.all(np.isnan(ising_acf)):
                tau_matrix[m, :] = np.nan
                R2_matrix[m, :] = np.nan
                cf_matrix[m, :] = np.nan
                continue

            ising_acf = np.asarray(ising_acf).ravel()

            # Ensure correct length
            if len(ising_acf) != max_lag + 1:
                print(f'    Warning: ACF length mismatch for sim {sim_idx}. '
                      f'Expected {max_lag + 1}, got {len(ising_acf)}.')
                # Pad or truncate
                if len(ising_acf) < max_lag + 1:
                    padded = np.full(max_lag + 1, np.nan)
                    padded[:len(ising_acf)] = ising_acf
                    ising_acf = padded
                else:
                    ising_acf = ising_acf[:max_lag + 1]

            # Refit with each range
            for f, end_pt in enumerate(fit_end_points):
                fit_range = (1, end_pt)
                tau, fit_result = fit_exponential_decay(ising_acf, ising_lags, fit_range)
                tau_matrix[m, f] = tau
                R2_matrix[m, f] = fit_result.get('R2', np.nan)

                # Conversion factor: tau_exp / tau_ising
                exp_tau_f = results['exp_tau'][f]
                if not np.isnan(tau) and tau > 0 and not np.isnan(exp_tau_f) and exp_tau_f > 0:
                    cf_matrix[m, f] = exp_tau_f / tau
                else:
                    cf_matrix[m, f] = np.nan

        results['ising_tau'][condition] = tau_matrix
        results['ising_R2'][condition] = R2_matrix
        results['conversionFactor'][condition] = cf_matrix

        print(f'  {condition}: processed {n_top} top matches')

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_sensitivity_results(results: dict, config: dict, output_dir: str,
                              save_format: str = 'png'):
    """
    Generate 2x2 sensitivity analysis figure.

    Port of MATLAB visualization from analyze_decay_fitRange_sensitivity.m.

    Parameters
    ----------
    results : dict
        Output from analyze_fit_range_sensitivity()
    config : dict
        Configuration dict with 'conditions', 'autocorr' keys
    output_dir : str
        Directory to save figures
    save_format : str
        Figure format ('png', 'pdf', etc.)
    """
    print('Creating sensitivity analysis figure...')

    fit_end_points = results['fitEndPoints']
    conditions = config.get('conditions', ['Naive', 'Beginner', 'Expert', 'NoSpout'])

    # Get colors
    colors = {}
    for cond in conditions:
        colors[cond] = CONDITION_COLORS.get(cond, (0.5, 0.5, 0.5))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # -------------------------------------------------------------------------
    # Subplot 1: Experimental tau vs fitRange
    # -------------------------------------------------------------------------
    ax = axes[0, 0]

    ax.plot(fit_end_points, results['exp_tau'], 'o-', color=EXP_COLOR,
            markerfacecolor=EXP_COLOR, linewidth=2, markersize=8)

    # Add R-squared annotation at each point
    for f, end_pt in enumerate(fit_end_points):
        r2 = results['exp_R2'][f]
        if not np.isnan(r2):
            ax.text(end_pt, results['exp_tau'][f] + 0.5, f'{r2:.2f}',
                    fontsize=7, ha='center', color=(0.5, 0.5, 0.5))

    ax.set_xlabel('Fit Range End Point (frames)')
    ax.set_ylabel('Experimental tau (frames)')
    ax.set_title('Experimental Tau vs Fit Range')
    ax.grid(True, alpha=0.3)

    # Add reference line at current fitRange endpoint
    current_fit_range = config.get('autocorr', {}).get('fit_range', (1, 10))
    if current_fit_range[1] <= 50:
        ax.axvline(current_fit_range[1], color='black', linestyle='--', linewidth=1)
        ax.text(current_fit_range[1] + 1, np.nanmax(results['exp_tau']) * 0.95,
                f'Current: {current_fit_range[1]}', fontsize=8, ha='left')

    # -------------------------------------------------------------------------
    # Subplot 2: Ising tau vs fitRange (mean +/- std per condition)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]

    for condition in conditions:
        if condition not in results['ising_tau']:
            continue

        tau_matrix = results['ising_tau'][condition]
        tau_mean = np.nanmean(tau_matrix, axis=0)
        tau_std = np.nanstd(tau_matrix, axis=0)

        cond_color = colors[condition]

        # Plot shaded error region
        ax.fill_between(fit_end_points, tau_mean - tau_std, tau_mean + tau_std,
                        color=cond_color, alpha=0.2)

        # Plot mean line
        ax.plot(fit_end_points, tau_mean, 'o-', color=cond_color,
                markerfacecolor=cond_color, linewidth=2, markersize=6,
                label=condition)

    ax.set_xlabel('Fit Range End Point (frames)')
    ax.set_ylabel('Ising tau (MC sweeps)')
    ax.set_title('Ising Tau vs Fit Range (Top 10 Matches)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Subplot 3: Conversion factor vs fitRange (mean +/- std per condition)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]

    for condition in conditions:
        if condition not in results['conversionFactor']:
            continue

        cf_matrix = results['conversionFactor'][condition]
        cf_mean = np.nanmean(cf_matrix, axis=0)
        cf_std = np.nanstd(cf_matrix, axis=0)

        cond_color = colors[condition]

        # Plot shaded error region
        ax.fill_between(fit_end_points, cf_mean - cf_std, cf_mean + cf_std,
                        color=cond_color, alpha=0.2)

        # Plot mean line
        ax.plot(fit_end_points, cf_mean, 'o-', color=cond_color,
                markerfacecolor=cond_color, linewidth=2, markersize=6,
                label=condition)

    ax.set_xlabel('Fit Range End Point (frames)')
    ax.set_ylabel('Conversion Factor (tau_exp / tau_ising)')
    ax.set_title('Temporal Conversion Factor vs Fit Range')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Subplot 4: R-squared (fit quality) vs fitRange
    # -------------------------------------------------------------------------
    ax = axes[1, 1]

    # Experimental R-squared
    ax.plot(fit_end_points, results['exp_R2'], 's-', color=EXP_COLOR,
            markerfacecolor=EXP_COLOR, linewidth=2, markersize=8,
            label='Experimental')

    # Ising R-squared per condition
    for condition in conditions:
        if condition not in results['ising_R2']:
            continue

        R2_matrix = results['ising_R2'][condition]
        R2_mean = np.nanmean(R2_matrix, axis=0)

        cond_color = colors[condition]
        ax.plot(fit_end_points, R2_mean, 'o-', color=cond_color,
                markerfacecolor=cond_color, linewidth=1.5, markersize=5,
                label=f'Ising ({condition})')

    ax.set_xlabel('Fit Range End Point (frames)')
    ax.set_ylabel('R-squared (Fit Quality)')
    ax.set_title('Fit Quality vs Fit Range')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.suptitle('Decay Fit Range Sensitivity Analysis', fontsize=14)
    fig.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f'decay_fitRange_sensitivity.{save_format}')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved to: {fig_path}')

    plt.close(fig)

    # Print summary statistics
    print('\n=== Summary Statistics ===')
    print(f'Fit Range Endpoints: {fit_end_points}')
    print(f'\nExperimental Tau:')
    print(f'  Range: [{np.nanmin(results["exp_tau"]):.2f}, {np.nanmax(results["exp_tau"]):.2f}] frames')
    print(f'  R-squared range: [{np.nanmin(results["exp_R2"]):.3f}, {np.nanmax(results["exp_R2"]):.3f}]')

    for condition in conditions:
        if condition in results['ising_tau']:
            tau_all = results['ising_tau'][condition].ravel()
            cf_all = results['conversionFactor'][condition].ravel()
            print(f'\n{condition} (top 10 matches):')
            print(f'  Tau range: [{np.nanmin(tau_all):.2f}, {np.nanmax(tau_all):.2f}] MC sweeps')
            print(f'  Conversion factor range: [{np.nanmin(cf_all):.3f}, {np.nanmax(cf_all):.3f}]')

    print('\n=== Sensitivity Analysis Complete ===')

    return fig_path


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """Run sensitivity analysis from command line."""
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description='ACF Fit Range Sensitivity Analysis')
    parser.add_argument('results_file', type=str,
                        help='Path to pickled results file or HDF5 mat file')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for figures')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Figure output format')
    args = parser.parse_args()

    # Load results
    print(f'Loading results from: {args.results_file}')

    if args.results_file.endswith('.pkl'):
        with open(args.results_file, 'rb') as f:
            data = pickle.load(f)
        ising_data = data.get('ising_data', [])
        exp_stats = data.get('ExpStats', {})
        comparison = data.get('Comparison', {})
        config = data.get('config', {})
    elif args.results_file.endswith('.mat'):
        import h5py
        # Load from HDF5 mat file
        print('Loading from HDF5 mat file...')
        with h5py.File(args.results_file, 'r') as f:
            # This requires custom loading logic based on file structure
            print('  Note: HDF5 mat file loading requires custom implementation')
            return
    else:
        print(f'Unknown file format: {args.results_file}')
        return

    # Run sensitivity analysis
    results = analyze_fit_range_sensitivity(ising_data, exp_stats, comparison, config)

    # Plot results
    plot_sensitivity_results(results, config, args.output_dir, args.format)


if __name__ == '__main__':
    main()
