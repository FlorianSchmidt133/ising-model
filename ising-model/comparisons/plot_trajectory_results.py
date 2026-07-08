#!/usr/bin/env python3
"""
Trajectory Comparison Visualizations
=====================================

Visualize results from trajectory_comparison.py: experimental vs Ising
model dynamics (Moran's I, activity, spatial correlation).

Produces 6 figures that expose the fundamental mismatch between Ising
equilibrium dynamics and experimental trajectories.

Usage:
    python plot_trajectory_results.py --results IsingTrajectories\\TrajectoryResults.h5
    python plot_trajectory_results.py -r /path/to/TrajectoryResults.h5 -o /path/to/output
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import h5py

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# =============================================================================
# COLOR SCHEMES (matching MATLAB exactly)
# =============================================================================

CONDITION_COLORS = {
    'Naive': (0.3373, 0.7059, 0.9137),
    'Beginner': (0.8431, 0.2549, 0.6078),
    'Expert': (0.0, 0.6196, 0.4510),
    'NoSpout': (0.8353, 0.3686, 0.0),
}

BIN_COLORS = {
    'low': (0.4, 0.6, 0.9),
    'mid': (0.6, 0.6, 0.6),
    'high': (0.9, 0.4, 0.4),
}

CONDITIONS = ['Naive', 'Beginner', 'Expert', 'NoSpout']

MODE_MARKERS = {
    'spontaneous': 'o',
    'stimulus_off': 's',
}


# =============================================================================
# HDF5 LOADER
# =============================================================================

def load_trajectory_results(filepath):
    """Load TrajectoryResults.h5 into a nested dict."""
    def read_group(group):
        result = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Group):
                result[key] = read_group(item)
            elif isinstance(item, h5py.Dataset):
                val = item[()]
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                result[key] = val
            else:
                result[key] = item
        return result

    with h5py.File(filepath, 'r') as f:
        data = read_group(f)
    print(f"Loaded: {filepath}")
    return data


# =============================================================================
# VISUALIZER CLASS
# =============================================================================

class TrajectoryVisualizer:
    """Visualization class for trajectory comparison results."""

    def __init__(self, data, output_dir=None):
        self.data = data
        self.conditions = [c for c in CONDITIONS if c in data]

        if output_dir is None:
            output_dir = '.'
        self.output_dir = os.path.join(output_dir, 'figures')
        os.makedirs(self.output_dir, exist_ok=True)

        # Style
        if HAS_SEABORN:
            sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10

    def _get_color(self, condition):
        return CONDITION_COLORS.get(condition, (0.2, 0.2, 0.2))

    def _save_figure(self, fig, name, formats=('png', 'pdf')):
        for fmt in formats:
            filepath = os.path.join(self.output_dir, f"{name}.{fmt}")
            fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"  Saved: {name}")

    # -----------------------------------------------------------------
    # Data helpers
    # -----------------------------------------------------------------

    def _collect_all_trajectories(self):
        """Walk HDF5 structure -> dict[condition] -> list of record dicts."""
        out = {}
        for condition in self.conditions:
            cond_data = self.data[condition]
            records = []
            for mode_key in ['spontaneous', 'stimulus_off']:
                if mode_key not in cond_data:
                    continue
                mode_data = cond_data[mode_key]
                if mode_key == 'spontaneous':
                    for bin_name in ['low', 'mid', 'high']:
                        if bin_name not in mode_data:
                            continue
                        bin_data = mode_data[bin_name]
                        for fkey in sorted(k for k in bin_data if k.startswith('frame_')):
                            rec = dict(bin_data[fkey])
                            rec['mode'] = mode_key
                            rec['bin'] = bin_name
                            rec['frame_key'] = fkey
                            records.append(rec)
                else:
                    for fkey in sorted(k for k in mode_data if k.startswith('frame_')):
                        rec = dict(mode_data[fkey])
                        rec['mode'] = mode_key
                        rec['bin'] = None
                        rec['frame_key'] = fkey
                        records.append(rec)
            out[condition] = records
        return out

    def _collect_metrics(self):
        """Walk HDF5 structure -> dict[condition] -> dict of metric arrays.

        Also stores mode and bin per record for filtering.
        """
        all_traj = self._collect_all_trajectories()
        out = {}
        for condition, records in all_traj.items():
            metrics_dict = {}
            modes_list = []
            bins_list = []
            for rec in records:
                m = rec.get('metrics', {})
                if not isinstance(m, dict):
                    continue
                has_values = False
                for k, v in m.items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        metrics_dict.setdefault(k, []).append(float(v))
                        has_values = True
                    elif isinstance(v, np.ndarray) and v.ndim == 0:
                        metrics_dict.setdefault(k, []).append(float(v))
                        has_values = True
                if has_values:
                    modes_list.append(rec['mode'])
                    bins_list.append(rec['bin'])
            # Convert to arrays
            for k in metrics_dict:
                metrics_dict[k] = np.array(metrics_dict[k])
            metrics_dict['_mode'] = modes_list
            metrics_dict['_bin'] = bins_list
            out[condition] = metrics_dict
        return out

    def _rescale_ising(self, ising_mi, scale_factor, n_exp):
        """Interpolate Ising MI onto experimental time using scale_factor."""
        n_sweeps = len(ising_mi)
        ising_time = np.arange(n_sweeps) * float(scale_factor)
        exp_time = np.arange(n_exp, dtype=np.float64)
        if ising_time[-1] < exp_time[-1]:
            n_valid = min(int(np.floor(ising_time[-1])) + 1, n_exp)
            if n_valid < 2:
                return np.full(n_exp, np.nan)
            interp_fn = interp1d(ising_time, ising_mi, kind='linear',
                                 fill_value='extrapolate')
            result = np.full(n_exp, np.nan)
            result[:n_valid] = interp_fn(exp_time[:n_valid])
            return result
        interp_fn = interp1d(ising_time, ising_mi, kind='linear',
                             fill_value='extrapolate')
        return interp_fn(exp_time)

    def _compute_ising_equilibrium(self, rec):
        """Compute mean MI over last sweeps of Ising trajectory (equilibrium).

        Returns mean MI over last 50 sweeps (or last 10 for short trajectories).
        """
        ising_mi_raw = np.asarray(rec.get('ising_mi', []), dtype=np.float64)
        if ising_mi_raw.size == 0:
            return np.nan
        if ising_mi_raw.ndim == 1:
            ising_mi_raw = ising_mi_raw.reshape(1, -1)
        # Mean across replicates first
        ising_mi_mean = np.nanmean(ising_mi_raw, axis=0)
        n_sweeps = len(ising_mi_mean)
        # Use last 50 sweeps, or last 10 for short trajectories (<=30)
        tail = 10 if n_sweeps <= 30 else 50
        tail = min(tail, n_sweeps)
        return float(np.nanmean(ising_mi_mean[-tail:]))

    def _compute_direction_agreement(self, rec):
        """Check whether exp and Ising MI move in the same direction.

        Returns True if net direction matches, False otherwise, or None if
        insufficient data.
        """
        exp_mi = np.asarray(rec.get('exp_mi', []), dtype=np.float64).ravel()
        ising_mi_raw = np.asarray(rec.get('ising_mi', []), dtype=np.float64)
        if exp_mi.size < 2 or ising_mi_raw.size == 0:
            return None
        if ising_mi_raw.ndim == 1:
            ising_mi_raw = ising_mi_raw.reshape(1, -1)
        ising_mean = np.nanmean(ising_mi_raw, axis=0)
        if len(ising_mean) < 2:
            return None

        # Use only valid (non-NaN) exp frames
        valid = ~np.isnan(exp_mi)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < 2:
            return None

        net_exp = exp_mi[valid_idx[-1]] - exp_mi[valid_idx[0]]
        net_ising = ising_mean[-1] - ising_mean[0]

        # Same sign = agreement (both zero counts as agreement)
        if net_exp == 0 or net_ising == 0:
            return True
        return (net_exp > 0) == (net_ising > 0)

    def _get_metric_value(self, rec, key):
        """Safely extract a scalar metric value from a record."""
        m = rec.get('metrics', {})
        if not isinstance(m, dict):
            return np.nan
        val = m.get(key, np.nan)
        if isinstance(val, np.ndarray):
            val = float(val) if val.ndim == 0 else float(val.ravel()[0])
        return float(val)

    def _select_median_rmse(self, records):
        """From a list of records, return the one with median raw_mi_rmse."""
        if not records:
            return None
        rmses = np.array([self._get_metric_value(r, 'raw_mi_rmse')
                          for r in records])
        valid = ~np.isnan(rmses)
        if not np.any(valid):
            return None
        valid_indices = np.where(valid)[0]
        sorted_valid = valid_indices[np.argsort(rmses[valid])]
        median_idx = sorted_valid[len(sorted_valid) // 2]
        return records[median_idx]

    # -----------------------------------------------------------------
    # 1. Trajectory Galleries (MI and Activity)
    # -----------------------------------------------------------------

    def _plot_gallery(self, signal, exp_key, ising_key, rmse_key, ylabel,
                      title_suffix, save_name):
        """Shared 2x3 gallery logic for MI or Activity trajectories."""
        all_traj = self._collect_all_trajectories()

        panel_specs = [
            ('Beginner', 'stimulus_off', None, 'Beginner stim-off\n(Best match)'),
            ('Naive', 'stimulus_off', None, 'Naive stim-off\n(Both decay)'),
            ('Expert', 'spontaneous', 'low', 'Expert spont/low\n(Both rise)'),
            ('Expert', 'stimulus_off', None, 'Expert stim-off\n(Direction mismatch)'),
            ('Expert', 'spontaneous', 'high', 'Expert spont/high\n(Ising rises, exp falls)'),
            ('NoSpout', 'spontaneous', 'high', 'NoSpout spont/high\n(Ising rises, exp falls)'),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(f'Trajectory Gallery: {title_suffix}',
                     fontsize=14, fontweight='bold')

        for idx, (cond, mode, bin_name, label) in enumerate(panel_specs):
            ax = axes[idx // 3, idx % 3]

            records = all_traj.get(cond, [])
            filtered = [r for r in records
                        if r['mode'] == mode and r['bin'] == bin_name]
            if not filtered:
                filtered = [r for r in records if r['mode'] == mode]

            rec = self._select_median_rmse(filtered)
            if rec is None:
                ax.text(0.5, 0.5, f'{label}\n(no data)', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')
                continue

            exp_vals = np.asarray(rec.get(exp_key, []), dtype=np.float64).ravel()
            ising_raw = np.asarray(rec.get(ising_key, []), dtype=np.float64)
            if ising_raw.ndim == 1:
                ising_raw = ising_raw.reshape(1, -1)

            n_sweeps = ising_raw.shape[1]
            n_exp = len(exp_vals)
            n_compare = min(n_sweeps, n_exp)

            ising_mean = np.nanmean(ising_raw[:, :n_compare], axis=0)
            ising_std = np.nanstd(ising_raw[:, :n_compare], axis=0)

            color = self._get_color(cond)
            t = np.arange(n_compare)

            ax.plot(t, exp_vals[:n_compare], color=color, linewidth=2.5,
                    label='Experimental')
            ax.fill_between(t, ising_mean - ising_std, ising_mean + ising_std,
                            color='gray', alpha=0.25, label='Ising +/- 1 SD')
            ax.plot(t, ising_mean, color='black', linewidth=1.5,
                    label='Ising mean')

            # RMSE annotation
            rmse_val = self._get_metric_value(rec, rmse_key)
            ax.text(0.97, 0.97, f'RMSE = {rmse_val:.4f}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.85, edgecolor='gray'))

            ax.set_title(label, fontsize=10)
            ax.set_xlabel('Frame / Sweep')
            ax.set_ylabel(ylabel)

            if idx == 0:
                ax.legend(fontsize=7, loc='lower right')

        fig.tight_layout()
        self._save_figure(fig, save_name)

    def plot_trajectory_gallery(self):
        """2x3 grid: Moran's I trajectories from good to bad."""
        self._plot_gallery(
            signal='mi',
            exp_key='exp_mi', ising_key='ising_mi',
            rmse_key='raw_mi_rmse',
            ylabel="Moran's I",
            title_suffix="Experimental vs Ising Moran's I",
            save_name='trajectory_gallery',
        )

    def plot_trajectory_gallery_activity(self):
        """2x3 grid: Activity trajectories from good to bad."""
        self._plot_gallery(
            signal='activity',
            exp_key='exp_activity', ising_key='ising_activity',
            rmse_key='raw_activity_rmse',
            ylabel='Mean Activity',
            title_suffix='Experimental vs Ising Activity',
            save_name='trajectory_gallery_activity',
        )

    # -----------------------------------------------------------------
    # 2. Equilibrium Mismatch
    # -----------------------------------------------------------------

    def plot_equilibrium_mismatch(self):
        """Show that the Ising model relaxes to its own equilibrium, not
        experimental dynamics."""
        all_traj = self._collect_all_trajectories()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('The Ising Model Relaxes to Its Own Equilibrium',
                     fontsize=13, fontweight='bold')

        # --- Left panel: Ising MI evolution (grand mean across all frames) ---
        ax_left = axes[0]
        ax_left.set_title('Ising MI Evolution (grand mean)', fontsize=11)

        for condition in self.conditions:
            records = all_traj.get(condition, [])
            if not records:
                continue

            # Collect all Ising MI trajectories
            all_ising = []
            for rec in records:
                ising_mi_raw = np.asarray(rec.get('ising_mi', []),
                                          dtype=np.float64)
                if ising_mi_raw.size == 0:
                    continue
                if ising_mi_raw.ndim == 1:
                    ising_mi_raw = ising_mi_raw.reshape(1, -1)
                # Mean across replicates for this frame
                all_ising.append(np.nanmean(ising_mi_raw, axis=0))

            if not all_ising:
                continue

            # Truncate to common length within condition
            min_len = min(len(s) for s in all_ising)
            stacked = np.array([s[:min_len] for s in all_ising])
            grand_mean = np.nanmean(stacked, axis=0)
            grand_sem = np.nanstd(stacked, axis=0) / np.sqrt(len(all_ising))

            color = self._get_color(condition)
            t = np.arange(min_len)
            ax_left.plot(t, grand_mean, color=color, linewidth=2,
                         label=condition)
            ax_left.fill_between(t, grand_mean - grand_sem,
                                 grand_mean + grand_sem,
                                 color=color, alpha=0.15)

            # Horizontal dashed line at equilibrium
            eq_mi = float(np.nanmean(grand_mean[-50:] if min_len > 50
                                     else grand_mean[-10:]))
            ax_left.axhline(eq_mi, color=color, linestyle='--', linewidth=1,
                            alpha=0.6)

        ax_left.set_xlabel('MC Sweep')
        ax_left.set_ylabel("Moran's I")
        ax_left.legend(fontsize=9)

        # --- Right panel: Equilibrium vs Reality (grouped bars) ---
        ax_right = axes[1]
        ax_right.set_title('Equilibrium MI vs Experimental Mean MI', fontsize=11)

        modes = ['spontaneous', 'stimulus_off']
        mode_labels = ['Spont', 'Stim-off']
        n_modes = len(modes)
        n_conds = len(self.conditions)
        bar_width = 0.35
        group_width = (n_modes * 2 + 1) * bar_width  # 2 bars per mode + gap

        for ci, condition in enumerate(self.conditions):
            records = all_traj.get(condition, [])
            color = self._get_color(condition)

            for mi, mode_key in enumerate(modes):
                mode_recs = [r for r in records if r['mode'] == mode_key]
                if not mode_recs:
                    continue

                # Ising equilibrium: mean across all frames
                eq_vals = [self._compute_ising_equilibrium(r)
                           for r in mode_recs]
                eq_vals = [v for v in eq_vals if not np.isnan(v)]

                # Experimental mean MI
                exp_means = []
                for r in mode_recs:
                    exp_mi = np.asarray(r.get('exp_mi', []),
                                        dtype=np.float64).ravel()
                    valid = exp_mi[~np.isnan(exp_mi)]
                    if len(valid) > 0:
                        exp_means.append(np.mean(valid))

                x_base = ci * (n_modes + 0.5) + mi
                x_ising = x_base - bar_width * 0.55
                x_exp = x_base + bar_width * 0.55

                if eq_vals:
                    ax_right.bar(x_ising, np.mean(eq_vals), bar_width * 0.9,
                                 color=color, edgecolor='black', linewidth=0.5,
                                 label='Ising eq.' if ci == 0 and mi == 0
                                 else None)
                if exp_means:
                    ax_right.bar(x_exp, np.mean(exp_means), bar_width * 0.9,
                                 color=color, edgecolor='black', linewidth=0.5,
                                 alpha=0.5, hatch='///',
                                 label='Exp mean' if ci == 0 and mi == 0
                                 else None)

        # X-axis labels
        tick_positions = []
        tick_labels = []
        for ci, condition in enumerate(self.conditions):
            for mi, ml in enumerate(mode_labels):
                tick_positions.append(ci * (n_modes + 0.5) + mi)
                tick_labels.append(f'{condition}\n{ml}')

        ax_right.set_xticks(tick_positions)
        ax_right.set_xticklabels(tick_labels, fontsize=7, rotation=0)
        ax_right.set_ylabel("Moran's I")
        ax_right.legend(fontsize=9)

        fig.tight_layout()
        self._save_figure(fig, 'equilibrium_mismatch')

    # -----------------------------------------------------------------
    # 3. Initial MI vs Equilibrium
    # -----------------------------------------------------------------

    def plot_initial_vs_equilibrium(self):
        """Match quality depends on proximity to Ising equilibrium."""
        all_traj = self._collect_all_trajectories()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Match Quality Depends on Proximity to Ising Equilibrium',
                     fontsize=13, fontweight='bold')

        # --- Left: Scatter (initial_mi - eq_mi) vs RMSE ---
        ax_left = axes[0]
        ax_left.set_title('Distance from Equilibrium vs RMSE', fontsize=11)

        for condition in self.conditions:
            records = all_traj.get(condition, [])
            color = self._get_color(condition)

            deltas = []
            rmses = []
            for rec in records:
                initial_mi = rec.get('initial_mi', np.nan)
                if isinstance(initial_mi, np.ndarray):
                    initial_mi = float(initial_mi)
                eq_mi = self._compute_ising_equilibrium(rec)
                rmse = self._get_metric_value(rec, 'raw_mi_rmse')

                if not (np.isnan(initial_mi) or np.isnan(eq_mi)
                        or np.isnan(rmse)):
                    deltas.append(initial_mi - eq_mi)
                    rmses.append(rmse)

            if deltas:
                marker = 'o'
                ax_left.scatter(deltas, rmses, s=25, color=color, alpha=0.6,
                                marker=marker, label=condition, edgecolors='none')

                # Annotate Beginner cluster
                if condition == 'Beginner':
                    mean_d = np.mean(deltas)
                    mean_r = np.mean(rmses)
                    ax_left.annotate('Beginner', xy=(mean_d, mean_r),
                                     xytext=(mean_d + 0.02, mean_r + 0.01),
                                     fontsize=8, color=color,
                                     arrowprops=dict(arrowstyle='->', color=color,
                                                     lw=0.8))

        ax_left.axvline(0, color='black', linestyle='--', linewidth=0.8,
                        alpha=0.5)
        ax_left.set_xlabel('Initial MI - Ising Equilibrium MI')
        ax_left.set_ylabel('Raw MI RMSE')
        ax_left.legend(fontsize=8)

        # --- Right: Direction agreement stacked bars ---
        ax_right = axes[1]
        ax_right.set_title('Direction Agreement', fontsize=11)

        x_positions = np.arange(len(self.conditions))
        match_counts = []
        mismatch_counts = []
        total_counts = []

        for condition in self.conditions:
            records = all_traj.get(condition, [])
            n_match = 0
            n_mismatch = 0
            for rec in records:
                agreement = self._compute_direction_agreement(rec)
                if agreement is True:
                    n_match += 1
                elif agreement is False:
                    n_mismatch += 1
            match_counts.append(n_match)
            mismatch_counts.append(n_mismatch)
            total_counts.append(n_match + n_mismatch)

        match_arr = np.array(match_counts, dtype=float)
        mismatch_arr = np.array(mismatch_counts, dtype=float)
        total_arr = np.array(total_counts, dtype=float)

        # Convert to percentages
        with np.errstate(divide='ignore', invalid='ignore'):
            match_pct = np.where(total_arr > 0, match_arr / total_arr * 100, 0)
            mismatch_pct = np.where(total_arr > 0,
                                    mismatch_arr / total_arr * 100, 0)

        colors_match = [(0.3, 0.7, 0.3)]  # green
        colors_mismatch = [(0.8, 0.3, 0.3)]  # red

        ax_right.bar(x_positions, match_pct, 0.6, color=colors_match[0],
                     edgecolor='black', linewidth=0.5, label='Same direction')
        ax_right.bar(x_positions, mismatch_pct, 0.6, bottom=match_pct,
                     color=colors_mismatch[0], edgecolor='black', linewidth=0.5,
                     label='Opposite direction')

        # Add count annotations
        for i, (m, mm, tot) in enumerate(zip(match_counts, mismatch_counts,
                                              total_counts)):
            if tot > 0:
                ax_right.text(i, 102, f'{m}/{tot}', ha='center', va='bottom',
                              fontsize=8)

        ax_right.set_xticks(x_positions)
        ax_right.set_xticklabels(self.conditions)
        ax_right.set_ylabel('Percentage of trajectories')
        ax_right.set_ylim(0, 115)
        ax_right.legend(fontsize=9, loc='upper right')

        fig.tight_layout()
        self._save_figure(fig, 'initial_vs_equilibrium')

    # -----------------------------------------------------------------
    # 4. Spatial Memory Decay
    # -----------------------------------------------------------------

    def plot_spatial_memory_decay(self):
        """The model forgets initial conditions: spatial correlation decay."""
        all_traj = self._collect_all_trajectories()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Ising Model Forgets Initial Spatial Pattern',
                     fontsize=13, fontweight='bold')

        modes = [('spontaneous', 'Spontaneous'),
                 ('stimulus_off', 'Stimulus-off')]

        for ax, (mode_key, mode_label) in zip(axes, modes):
            ax.set_title(mode_label, fontsize=11)

            for condition in self.conditions:
                records = all_traj.get(condition, [])
                mode_recs = [r for r in records if r['mode'] == mode_key]
                if not mode_recs:
                    continue

                # Collect spatial correlation arrays
                corr_arrays = []
                for rec in mode_recs:
                    sc = rec.get('ising_spatial_corr', None)
                    if sc is None:
                        continue
                    sc = np.asarray(sc, dtype=np.float64)
                    if sc.ndim == 2:
                        # (n_reps, n_sweeps) -> mean across reps
                        sc = np.nanmean(sc, axis=0)
                    sc = sc.ravel()
                    if len(sc) > 0:
                        corr_arrays.append(sc)

                if not corr_arrays:
                    continue

                # Truncate to min length
                min_len = min(len(c) for c in corr_arrays)
                stacked = np.array([c[:min_len] for c in corr_arrays])
                mean_corr = np.nanmean(stacked, axis=0)
                sem_corr = np.nanstd(stacked, axis=0) / np.sqrt(len(corr_arrays))

                color = self._get_color(condition)
                t = np.arange(min_len)
                ax.plot(t, mean_corr, color=color, linewidth=2, label=condition)
                ax.fill_between(t, mean_corr - sem_corr, mean_corr + sem_corr,
                                color=color, alpha=0.15)

                # Annotate half-life (first sweep where corr < 0.25)
                below = np.where(mean_corr < 0.25)[0]
                if len(below) > 0:
                    half_sweep = below[0]
                    ax.plot(half_sweep, mean_corr[half_sweep], 'v',
                            color=color, markersize=6)
                    ax.annotate(f't½={half_sweep}',
                                xy=(half_sweep, mean_corr[half_sweep]),
                                xytext=(half_sweep + min_len * 0.05,
                                        mean_corr[half_sweep] + 0.05),
                                fontsize=7, color=color,
                                arrowprops=dict(arrowstyle='->', color=color,
                                                lw=0.6))

            ax.axhline(0, color='black', linestyle='-', linewidth=0.5,
                       alpha=0.5)
            ax.set_xlabel('MC Sweep')
            ax.set_ylabel('Spatial Correlation with Initial Frame')
            ax.legend(fontsize=9)

        fig.tight_layout()
        self._save_figure(fig, 'spatial_memory_decay')

    # -----------------------------------------------------------------
    # 5. Temporal Scale Problem
    # -----------------------------------------------------------------

    def plot_temporal_scale_problem(self):
        """Why temporal rescaling fails for most conditions."""
        all_traj = self._collect_all_trajectories()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Temporal Rescaling Fails for Most Conditions',
                     fontsize=13, fontweight='bold')

        # --- Left: Coverage diagram ---
        ax_left = axes[0]
        ax_left.set_title('Temporal Coverage (n_sweeps x scale_factor)',
                          fontsize=11)

        y_positions = np.arange(len(self.conditions))
        coverages = []
        bar_colors = []

        for condition in self.conditions:
            records = all_traj.get(condition, [])
            if not records:
                coverages.append(0)
                bar_colors.append(self._get_color(condition))
                continue

            # Compute coverage: n_sweeps * scale_factor for each record
            frame_coverages = []
            for rec in records:
                sf = rec.get('scale_factor', 1.0)
                if isinstance(sf, np.ndarray):
                    sf = float(sf)
                ising_mi_raw = np.asarray(rec.get('ising_mi', []),
                                          dtype=np.float64)
                if ising_mi_raw.size == 0:
                    continue
                if ising_mi_raw.ndim == 1:
                    n_sw = len(ising_mi_raw)
                else:
                    n_sw = ising_mi_raw.shape[1]
                frame_coverages.append(n_sw * sf)

            if frame_coverages:
                coverages.append(np.mean(frame_coverages))
            else:
                coverages.append(0)
            bar_colors.append(self._get_color(condition))

        # Plot horizontal bars
        for i, (cov, color) in enumerate(zip(coverages, bar_colors)):
            ax_left.barh(i, cov, 0.6, color=color, edgecolor='black',
                         linewidth=0.5)
            ax_left.text(max(cov + 0.5, 1), i,
                         f'{cov:.1f} frames', va='center', fontsize=9)

        # Dashed line at 30 (full trajectory)
        ax_left.axvline(30, color='black', linestyle='--', linewidth=1.5,
                        alpha=0.7, label='Full trajectory (30 frames)')
        ax_left.set_yticks(y_positions)
        ax_left.set_yticklabels(self.conditions)
        ax_left.set_xlabel('Experimental frames covered')
        ax_left.legend(fontsize=8, loc='lower right')

        # Set x limit to show the 30-frame line clearly
        ax_left.set_xlim(0, max(max(coverages) * 1.15, 35))

        # --- Right: Raw vs Rescaled RMSE (paired scatter) ---
        ax_right = axes[1]
        ax_right.set_title('Raw vs Rescaled MI RMSE', fontsize=11)

        has_nan_rescaled = False

        for condition in self.conditions:
            records = all_traj.get(condition, [])
            color = self._get_color(condition)

            raw_vals = []
            resc_vals = []
            nan_raw_vals = []  # raw RMSE for records where rescaled is NaN

            for rec in records:
                raw = self._get_metric_value(rec, 'raw_mi_rmse')
                resc = self._get_metric_value(rec, 'rescaled_mi_rmse')
                if np.isnan(raw):
                    continue
                if np.isnan(resc):
                    nan_raw_vals.append(raw)
                    has_nan_rescaled = True
                else:
                    raw_vals.append(raw)
                    resc_vals.append(resc)

            # Plot paired points with connecting lines
            for rv, resv in zip(raw_vals, resc_vals):
                ax_right.plot([rv, rv], [rv, resv], color=color, linewidth=0.5,
                              alpha=0.4)

            if raw_vals:
                ax_right.scatter(raw_vals, resc_vals, s=20, color=color,
                                 alpha=0.6, label=condition, edgecolors='none')

            # NaN rescaled: mark at top of axis
            if nan_raw_vals:
                ax_right.scatter(nan_raw_vals,
                                 [ax_right.get_ylim()[1] if ax_right.get_ylim()[1] > 0
                                  else 0.5] * len(nan_raw_vals),
                                 s=20, color=color, alpha=0.6, marker='x',
                                 label=f'{condition} (NaN resc.)'
                                 if condition == self.conditions[0] else None)

        # Diagonal line
        all_lims = ax_right.get_xlim() + ax_right.get_ylim()
        lim_min = min(all_lims)
        lim_max = max(all_lims)
        ax_right.plot([lim_min, lim_max], [lim_min, lim_max], 'k--',
                      linewidth=1, alpha=0.5, label='No change')
        ax_right.set_xlabel('Raw MI RMSE')
        ax_right.set_ylabel('Rescaled MI RMSE')
        ax_right.legend(fontsize=7, loc='upper left')

        # Now handle NaN rescaled points properly (need to redo after axis
        # limits are set)
        if has_nan_rescaled:
            # Re-collect and plot at top of final axis
            ylim = ax_right.get_ylim()
            nan_y = ylim[1] * 0.95
            for condition in self.conditions:
                records = all_traj.get(condition, [])
                color = self._get_color(condition)
                nan_raw = [self._get_metric_value(rec, 'raw_mi_rmse')
                           for rec in records
                           if (not np.isnan(self._get_metric_value(
                               rec, 'raw_mi_rmse'))
                               and np.isnan(self._get_metric_value(
                                   rec, 'rescaled_mi_rmse')))]
                if nan_raw:
                    ax_right.scatter(nan_raw, [nan_y] * len(nan_raw),
                                     s=25, color=color, marker='x',
                                     alpha=0.7, zorder=10)
            ax_right.text(0.98, 0.95, 'x = rescaling\nnot computable',
                          transform=ax_right.transAxes, ha='right', va='top',
                          fontsize=7, color='gray',
                          bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='white', alpha=0.8))

        fig.tight_layout()
        self._save_figure(fig, 'temporal_scale_problem')

    # -----------------------------------------------------------------
    # 6. Quantitative Summary
    # -----------------------------------------------------------------

    def plot_quantitative_summary(self):
        """2x2 box/strip plots of fit quality metrics."""
        all_traj = self._collect_all_trajectories()
        metrics = self._collect_metrics()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Fit Quality Across Conditions and Modes',
                     fontsize=14, fontweight='bold')

        modes_list = ['spontaneous', 'stimulus_off']
        mode_short = {'spontaneous': 'Spont', 'stimulus_off': 'Stim-off'}

        # --- (0,0): MI RMSE by condition x mode ---
        ax = axes[0, 0]
        ax.set_title("Moran's I RMSE by Condition x Mode", fontsize=11)
        self._boxstrip_condition_mode(ax, all_traj, 'raw_mi_rmse',
                                      "Moran's I RMSE")

        # --- (0,1): Envelope fraction by condition x mode ---
        ax = axes[0, 1]
        ax.set_title('Envelope Fraction by Condition x Mode', fontsize=11)
        self._boxstrip_condition_mode(ax, all_traj, 'raw_mi_envelope_frac',
                                      'Envelope Fraction')
        ax.axhline(0.68, color='black', linestyle='--', linewidth=1, alpha=0.6,
                   label='Expected (1 SD)')
        ax.legend(fontsize=7, loc='lower right')

        # --- (1,0): MI RMSE by MI bin (spontaneous only) ---
        ax = axes[1, 0]
        ax.set_title("MI RMSE by MI Bin (Spontaneous)", fontsize=11)
        self._boxstrip_by_bin(ax, all_traj, 'raw_mi_rmse', "Moran's I RMSE")

        # --- (1,1): Activity RMSE by condition x mode ---
        ax = axes[1, 1]
        ax.set_title('Activity RMSE by Condition x Mode', fontsize=11)
        self._boxstrip_condition_mode(ax, all_traj, 'raw_activity_rmse',
                                      'Activity RMSE')

        fig.tight_layout()
        self._save_figure(fig, 'quantitative_summary')

    def _boxstrip_condition_mode(self, ax, all_traj, metric_key, ylabel):
        """Box/strip plot: x = condition, hue = mode."""
        modes_list = ['spontaneous', 'stimulus_off']
        mode_short = {'spontaneous': 'Spont', 'stimulus_off': 'Stim-off'}
        n_modes = len(modes_list)
        width = 0.35

        positions_all = []
        data_all = []
        colors_all = []
        labels_done = set()

        for ci, condition in enumerate(self.conditions):
            records = all_traj.get(condition, [])
            color = self._get_color(condition)

            for mi, mode_key in enumerate(modes_list):
                mode_recs = [r for r in records if r['mode'] == mode_key]
                vals = [self._get_metric_value(r, metric_key)
                        for r in mode_recs]
                vals = [v for v in vals if not np.isnan(v)]

                x_pos = ci * (n_modes + 0.5) + mi
                positions_all.append(x_pos)

                if vals:
                    bp = ax.boxplot([vals], positions=[x_pos], widths=width,
                                    patch_artist=True, showfliers=False,
                                    medianprops=dict(color='black', linewidth=1.5))
                    face_alpha = 1.0 if mode_key == 'spontaneous' else 0.5
                    bp['boxes'][0].set_facecolor((*color, face_alpha))
                    bp['boxes'][0].set_edgecolor('black')

                    # Strip (jittered points)
                    rng = np.random.default_rng(ci * 10 + mi)
                    jitter = rng.uniform(-width * 0.3, width * 0.3, len(vals))
                    ax.scatter(x_pos + jitter, vals, s=12, color='black',
                               alpha=0.4, zorder=5)

        # X labels
        tick_pos = []
        tick_lab = []
        for ci, condition in enumerate(self.conditions):
            for mi, mode_key in enumerate(modes_list):
                tick_pos.append(ci * (n_modes + 0.5) + mi)
                tick_lab.append(f'{condition}\n{mode_short[mode_key]}')
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, fontsize=7)
        ax.set_ylabel(ylabel)

    def _boxstrip_by_bin(self, ax, all_traj, metric_key, ylabel):
        """Box/strip plot: x = MI bin, hue = condition (spontaneous only)."""
        bins = ['low', 'mid', 'high']
        n_conds = len(self.conditions)
        width = 0.8 / max(n_conds, 1)

        for ci, condition in enumerate(self.conditions):
            records = all_traj.get(condition, [])
            color = self._get_color(condition)

            for bi, bin_name in enumerate(bins):
                bin_recs = [r for r in records
                            if r['mode'] == 'spontaneous' and r['bin'] == bin_name]
                vals = [self._get_metric_value(r, metric_key)
                        for r in bin_recs]
                vals = [v for v in vals if not np.isnan(v)]

                offset = (ci - n_conds / 2 + 0.5) * width
                x_pos = bi + offset

                if vals:
                    bp = ax.boxplot([vals], positions=[x_pos], widths=width * 0.8,
                                    patch_artist=True, showfliers=False,
                                    medianprops=dict(color='black', linewidth=1.5))
                    bp['boxes'][0].set_facecolor(color)
                    bp['boxes'][0].set_edgecolor('black')

                    rng = np.random.default_rng(ci * 10 + bi)
                    jitter = rng.uniform(-width * 0.2, width * 0.2, len(vals))
                    ax.scatter(x_pos + jitter, vals, s=12, color='black',
                               alpha=0.4, zorder=5)

        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([b.capitalize() for b in bins])
        ax.set_ylabel(ylabel)

        # Legend
        patches = [plt.Line2D([0], [0], marker='s', color='w',
                               markerfacecolor=self._get_color(c),
                               markersize=8, label=c)
                   for c in self.conditions]
        ax.legend(handles=patches, fontsize=8, loc='upper left')

    # -----------------------------------------------------------------
    # Run all
    # -----------------------------------------------------------------

    def plot_all(self):
        """Generate all 6 figures."""
        print(f"\nSaving figures to: {self.output_dir}\n")
        self.plot_trajectory_gallery()
        self.plot_trajectory_gallery_activity()
        self.plot_equilibrium_mismatch()
        self.plot_initial_vs_equilibrium()
        self.plot_spatial_memory_decay()
        self.plot_temporal_scale_problem()
        self.plot_quantitative_summary()
        print("\nDone! (7 figures)")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize trajectory comparison results (TrajectoryResults.h5)'
    )
    parser.add_argument(
        '--results', '-r', type=str,
        default=r'IsingTrajectories\TrajectoryResults.h5',
        help='Path to TrajectoryResults.h5'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output directory (defaults to dirname of results file)'
    )
    args = parser.parse_args()

    if not os.path.isfile(args.results):
        print(f"Error: results file not found: {args.results}")
        return

    output_dir = args.output if args.output else os.path.dirname(args.results)

    data = load_trajectory_results(args.results)
    viz = TrajectoryVisualizer(data, output_dir)
    viz.plot_all()


if __name__ == '__main__':
    main()
