#!/usr/bin/env python3
"""Rank free-dynamics window modes by Expert-vs-NoSpout difference.

Usage:
    python rank_expert_vs_nospout.py --heatmap-data /path/to/heatmap_data.npz --size 4 --duration 59 --stim-mode double_pulse10

Prints a table ranking all window_mode × metric combinations by absolute
mean difference between Expert and NoSpout.
"""

import argparse
import numpy as np
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from generate_perturbation_snapshots import (
    _get_free_dynamics_mask, _parse_clamp_width, FREE_DYNAMICS_MODES,
    PRE_STIM_FRAMES,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--heatmap-data', required=True, help='Path to heatmap_data.npz')
    parser.add_argument('--size', type=int, required=True, help='Stimulus size')
    parser.add_argument('--duration', type=int, required=True, help='Stimulus duration')
    parser.add_argument('--stim-mode', default='double_pulse', help='Stimulus mode label')
    parser.add_argument('--buffer', type=int, default=15)
    parser.add_argument('--post-stim-frames', type=int, default=25)
    args = parser.parse_args()

    data = dict(np.load(args.heatmap_data, allow_pickle=True))
    n_reps = data.get('__n_reps__', '?')

    spag_key = f'__spaghetti_indices_{args.duration}__'
    if spag_key not in data:
        print(f"ERROR: {spag_key} not found in data")
        return 1

    clamp_width = _parse_clamp_width(args.stim_mode)
    metrics = ['blob_area', 'blob_area_net', 'frac_crop']
    spaghetti_indices = np.asarray(data[spag_key])

    results = []

    def _add_result(category, label, expert_valid, nospout_valid, metric):
        expert_mean = np.mean(expert_valid)
        nospout_mean = np.mean(nospout_valid)
        diff = expert_mean - nospout_mean
        pooled_std = np.sqrt((np.var(expert_valid) + np.var(nospout_valid)) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0.0
        _, p_val = stats.mannwhitneyu(expert_valid, nospout_valid, alternative='two-sided')
        results.append({
            'category': category,
            'label': label,
            'metric': metric,
            'expert_mean': expert_mean,
            'nospout_mean': nospout_mean,
            'diff': diff,
            'cohens_d': cohens_d,
            'p_val': p_val,
            'n_expert': len(expert_valid),
            'n_nospout': len(nospout_valid),
        })

    # --- Free-dynamics window modes ---
    for wm in FREE_DYNAMICS_MODES:
        _, mode_tag = FREE_DYNAMICS_MODES[wm]
        mask, _ = _get_free_dynamics_mask(
            data[spag_key], args.duration, args.buffer, args.post_stim_frames,
            clamp_width=clamp_width, window_mode=wm)
        if mask.sum() == 0:
            continue

        for metric in metrics:
            expert_key = f'Expert_{args.size}_{args.duration}_{metric}'
            nospout_key = f'NoSpout_{args.size}_{args.duration}_{metric}'
            if expert_key not in data or nospout_key not in data:
                continue

            e_data, n_data = data[expert_key], data[nospout_key]
            m_e = mask[:e_data.shape[1]] if len(mask) > e_data.shape[1] else mask
            m_n = mask[:n_data.shape[1]] if len(mask) > n_data.shape[1] else mask

            e_vals = np.nanmean(e_data[:, m_e], axis=1)
            n_vals = np.nanmean(n_data[:, m_n], axis=1)
            e_valid = e_vals[~np.isnan(e_vals)]
            n_valid = n_vals[~np.isnan(n_vals)]
            if len(e_valid) == 0 or len(n_valid) == 0:
                continue
            _add_result('free_dynamics', mode_tag, e_valid, n_valid, metric)

    # --- Subwindow modes ---
    stim_mask = (spaghetti_indices >= PRE_STIM_FRAMES) & \
                (spaghetti_indices < PRE_STIM_FRAMES + args.duration)
    stim_col_indices = np.where(stim_mask)[0]
    n_stim = len(stim_col_indices)

    if n_stim > 0:
        window_cfg = {
            'first_10pct': ('First 10% peak',  0.10, 'peak'),
            'first_20pct': ('First 20% peak',  0.20, 'peak'),
            'last_10pct':  ('Last 10% peak',   0.10, 'peak'),
            'last_20pct':  ('Last 20% peak',   0.20, 'peak'),
            'peak_all':    ('Peak (full stim)', 1.00, 'peak'),
            'mean_all':    ('Mean (full stim)', 1.00, 'mean'),
        }
        for window, (title_label, frac, agg) in window_cfg.items():
            n_sel = max(1, round(frac * n_stim))
            if window.startswith('first'):
                sel_cols = stim_col_indices[:n_sel]
            elif window.startswith('last'):
                sel_cols = stim_col_indices[-n_sel:]
            else:
                sel_cols = stim_col_indices

            for metric in metrics:
                expert_key = f'Expert_{args.size}_{args.duration}_{metric}'
                nospout_key = f'NoSpout_{args.size}_{args.duration}_{metric}'
                if expert_key not in data or nospout_key not in data:
                    continue

                e_data, n_data = data[expert_key], data[nospout_key]
                e_cols = sel_cols[sel_cols < e_data.shape[1]]
                n_cols = sel_cols[sel_cols < n_data.shape[1]]
                if len(e_cols) == 0 or len(n_cols) == 0:
                    continue

                e_sub = e_data[:, e_cols]
                n_sub = n_data[:, n_cols]
                e_vals = np.nanmax(e_sub, axis=1) if agg == 'peak' else np.nanmean(e_sub, axis=1)
                n_vals = np.nanmax(n_sub, axis=1) if agg == 'peak' else np.nanmean(n_sub, axis=1)
                e_valid = e_vals[~np.isnan(e_vals)]
                n_valid = n_vals[~np.isnan(n_vals)]
                if len(e_valid) == 0 or len(n_valid) == 0:
                    continue
                _add_result('subwindow', title_label, e_valid, n_valid, metric)

    # Sort by Cohen's d (effect size, more meaningful than raw diff)
    results.sort(key=lambda r: abs(r['cohens_d']), reverse=True)

    print(f"\nExpert vs NoSpout ranking — size={args.size}, dur={args.duration}, "
          f"stim_mode={args.stim_mode}, n_reps={n_reps}")
    print(f"{'Rank':>4}  {'Category':<14} {'Window':<25} {'Metric':<16} {'Expert':>10} {'NoSpout':>10} "
          f"{'Diff':>10} {'Cohen d':>10} {'p-value':>12}  {'n_E':>4} {'n_N':>4}")
    print("-" * 140)
    for i, r in enumerate(results, 1):
        print(f"{i:4d}  {r['category']:<14} {r['label']:<25} {r['metric']:<16} {r['expert_mean']:10.3f} "
              f"{r['nospout_mean']:10.3f} {r['diff']:+10.3f} {r['cohens_d']:+10.3f} {r['p_val']:12.2e}  "
              f"{r['n_expert']:4d} {r['n_nospout']:4d}")

    # Highlight the best
    best = results[0]
    print(f"\n{'='*140}")
    print(f"BEST: [{best['category']}] {best['label']} / {best['metric']}")
    print(f"  Expert={best['expert_mean']:.3f}, NoSpout={best['nospout_mean']:.3f}, "
          f"Diff={best['diff']:+.3f}, Cohen's d={best['cohens_d']:+.3f}, "
          f"p={best['p_val']:.2e} (Mann-Whitney U)")

    return 0


if __name__ == '__main__':
    exit(main())
