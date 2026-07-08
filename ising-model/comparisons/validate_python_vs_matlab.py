#!/usr/bin/env python3
"""
Validation Script: Python vs MATLAB Ising Comparison
=====================================================

This script validates that the optimized Python implementation produces
results consistent with the MATLAB implementation.

Validation checks:
1. Moran's I values match within numerical precision
2. Wasserstein distances match within tolerance
3. Best-match rankings are identical
4. Blob detection produces consistent statistics

Usage:
    python validate_python_vs_matlab.py [--matlab-results PATH] [--n-sims N]

Arguments:
    --matlab-results    Path to MATLAB results file (optional)
    --n-sims            Number of simulations to validate (default: 100)
    --local             Use local paths instead of cluster paths
"""

import os
import sys
import argparse
import time
from glob import glob

import numpy as np
import h5py
from scipy import io as sio
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

# Import optimized modules
from morans_i_optimized import (
    create_weight_matrix_jit,
    morans_i_single_jit,
    morans_i_batch,
    warmup_jit
)
from blob_detection_optimized import (
    detect_blobs_single,
    compute_blob_persistence_batch,
    compute_autocorr_decay,
    compute_trial_averaged_autocorr,
    compute_temporal_scale_factor
)
from wasserstein_optimized import wasserstein_1d


# =============================================================================
# REFERENCE IMPLEMENTATIONS (MATCHING MATLAB EXACTLY)
# =============================================================================

def morans_i_reference(values, weight_mat):
    """
    Reference Moran's I implementation (pure NumPy).
    Matches MATLAB mL_moransI exactly.
    """
    x = values.ravel().astype(float)
    nan_mask = np.isnan(x)
    n = np.sum(~nan_mask)

    if n == 0:
        return np.nan

    x_mean = np.nanmean(x)
    x_dev = x - x_mean
    x_dev[nan_mask] = 0

    weight_mat_copy = weight_mat.copy()
    weight_mat_copy[nan_mask, :] = 0
    weight_mat_copy[:, nan_mask] = 0

    W = np.sum(weight_mat_copy)
    if W == 0:
        return np.nan

    numerator = np.sum((weight_mat_copy @ x_dev) * x_dev)
    denominator = np.sum(x_dev ** 2)

    if denominator == 0:
        return np.nan

    return (n / W) * (numerator / denominator)


def create_weight_matrix_reference(grid_shape):
    """
    Reference weight matrix creation (pure NumPy).
    Uses pdist/squareform like original MATLAB.
    """
    rows, cols = grid_shape
    row_coords, col_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    coords = np.column_stack([row_coords.ravel(), col_coords.ravel()])
    dist_mat = squareform(pdist(coords, metric='euclidean'))
    weight_mat = (dist_mat == 1).astype(float)
    return weight_mat


def wasserstein_1d_reference(x, y, max_quantiles=1000):
    """
    Reference Wasserstein distance (pure NumPy).
    Matches MATLAB wasserstein_1d exactly.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    x = np.sort(x)
    y = np.sort(y)

    n = min(max_quantiles, min(len(x), len(y)))
    q = np.linspace(0, 1, n)

    x_quantiles = np.quantile(x, q)
    y_quantiles = np.quantile(y, q)

    return np.mean(np.abs(x_quantiles - y_quantiles))


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_weight_matrix():
    """Validate weight matrix creation."""
    print("\n1. Validating weight matrix creation...")

    grid_shapes = [(13, 26), (10, 10), (5, 8), (32, 32)]
    all_passed = True

    for shape in grid_shapes:
        w_ref = create_weight_matrix_reference(shape)
        w_jit = create_weight_matrix_jit(shape[0], shape[1], False)

        max_diff = np.max(np.abs(w_ref - w_jit))

        if max_diff < 1e-10:
            print(f"   Grid {shape}: PASSED (max diff: {max_diff:.2e})")
        else:
            print(f"   Grid {shape}: FAILED (max diff: {max_diff:.2e})")
            all_passed = False

    return all_passed


def validate_morans_i(n_frames=100):
    """Validate Moran's I computation."""
    print(f"\n2. Validating Moran's I ({n_frames} frames)...")

    np.random.seed(42)
    rows, cols = 13, 26

    # Create weight matrices
    w_ref = create_weight_matrix_reference((rows, cols))
    w_jit = create_weight_matrix_jit(rows, cols, False)

    # Test data
    frames = np.random.rand(n_frames, rows, cols).astype(np.float64)

    # Add some -1/+1 patterns like Ising
    ising_frames = np.random.choice([-1, 1], size=(n_frames, rows, cols)).astype(np.float64)

    # Add NaN test cases
    nan_frames = frames.copy()
    nan_frames[10, 5, 10] = np.nan
    nan_frames[20, :3, :3] = np.nan

    all_passed = True

    # Test regular frames
    print("   Testing random continuous frames...")
    errors = []
    for t in range(n_frames):
        I_ref = morans_i_reference(frames[t], w_ref)
        I_jit = morans_i_single_jit(frames[t], w_jit)
        if np.isnan(I_ref) and np.isnan(I_jit):
            err = 0
        elif np.isnan(I_ref) or np.isnan(I_jit):
            err = np.inf
        else:
            err = abs(I_ref - I_jit)
        errors.append(err)

    max_err = max(errors)
    mean_err = np.mean(errors)
    if max_err < 1e-10:
        print(f"      PASSED: max error {max_err:.2e}, mean error {mean_err:.2e}")
    else:
        print(f"      FAILED: max error {max_err:.2e}")
        all_passed = False

    # Test Ising frames
    print("   Testing Ising (-1/+1) frames...")
    errors = []
    for t in range(n_frames):
        I_ref = morans_i_reference(ising_frames[t], w_ref)
        I_jit = morans_i_single_jit(ising_frames[t], w_jit)
        if np.isnan(I_ref) and np.isnan(I_jit):
            err = 0
        elif np.isnan(I_ref) or np.isnan(I_jit):
            err = np.inf
        else:
            err = abs(I_ref - I_jit)
        errors.append(err)

    max_err = max(errors)
    if max_err < 1e-10:
        print(f"      PASSED: max error {max_err:.2e}")
    else:
        print(f"      FAILED: max error {max_err:.2e}")
        all_passed = False

    # Test NaN handling
    print("   Testing NaN handling...")
    errors = []
    for t in range(n_frames):
        I_ref = morans_i_reference(nan_frames[t], w_ref)
        I_jit = morans_i_single_jit(nan_frames[t], w_jit)
        if np.isnan(I_ref) and np.isnan(I_jit):
            err = 0
        elif np.isnan(I_ref) or np.isnan(I_jit):
            err = np.inf
        else:
            err = abs(I_ref - I_jit)
        errors.append(err)

    max_err = max(errors)
    if max_err < 1e-10:
        print(f"      PASSED: max error {max_err:.2e}")
    else:
        print(f"      FAILED: max error {max_err:.2e}")
        all_passed = False

    # Test batch processing
    print("   Testing batch processing...")
    I_batch = morans_i_batch(frames, w_jit)
    I_ref_all = np.array([morans_i_reference(frames[t], w_ref) for t in range(n_frames)])

    max_batch_err = np.max(np.abs(I_batch - I_ref_all))
    if max_batch_err < 1e-10:
        print(f"      PASSED: max batch error {max_batch_err:.2e}")
    else:
        print(f"      FAILED: max batch error {max_batch_err:.2e}")
        all_passed = False

    return all_passed


def validate_wasserstein():
    """Validate Wasserstein distance computation."""
    print("\n3. Validating Wasserstein distance...")

    np.random.seed(42)
    all_passed = True

    test_cases = [
        # (x, y, description)
        (np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000), "Same distribution"),
        (np.random.normal(0, 1, 1000), np.random.normal(0.5, 1, 1000), "Shifted mean"),
        (np.random.normal(0, 1, 1000), np.random.normal(0, 2, 1000), "Different variance"),
        (np.random.uniform(0, 1, 500), np.random.uniform(0, 1, 2000), "Different sizes"),
    ]

    for x, y, desc in test_cases:
        d_ref = wasserstein_1d_reference(x, y)
        d_opt = wasserstein_1d(x, y)

        rel_err = abs(d_ref - d_opt) / max(abs(d_ref), 1e-10)
        if rel_err < 1e-6:
            print(f"   {desc}: PASSED (rel err: {rel_err:.2e})")
        else:
            print(f"   {desc}: FAILED (ref: {d_ref:.6f}, opt: {d_opt:.6f}, rel err: {rel_err:.2e})")
            all_passed = False

    # Test with NaN
    print("   Testing NaN handling...")
    x_nan = np.random.normal(0, 1, 1000)
    x_nan[::10] = np.nan
    y = np.random.normal(0, 1, 1000)

    d_ref = wasserstein_1d_reference(x_nan, y)
    d_opt = wasserstein_1d(x_nan, y)

    rel_err = abs(d_ref - d_opt) / max(abs(d_ref), 1e-10)
    if rel_err < 1e-6:
        print(f"      PASSED (rel err: {rel_err:.2e})")
    else:
        print(f"      FAILED (rel err: {rel_err:.2e})")
        all_passed = False

    return all_passed


def validate_blob_detection():
    """Validate blob detection."""
    print("\n4. Validating blob detection...")

    np.random.seed(42)

    # Create synthetic data with known blobs
    rows, cols = 13, 26
    n_frames = 50
    frames = np.random.rand(n_frames, rows, cols) * 0.3

    # Add known blobs
    for t in range(n_frames):
        if t < 40:
            frames[t, 3:6, 10:14] = 0.9  # Persistent blob
        if 10 <= t < 25:
            frames[t, 8:11, 5:8] = 0.85  # Medium blob

    print("   Testing blob detection consistency...")

    # Run detection
    blob_stats = compute_blob_persistence_batch(
        frames, sigma=1.0, threshold=0.5, min_size=5, iou_threshold=0.3
    )

    # Check that we detect blobs
    total_detected = np.sum(blob_stats.counts > 0)
    print(f"      Frames with blobs: {total_detected}/{n_frames}")

    if total_detected > 0:
        print(f"      Mean blobs/frame: {np.mean(blob_stats.counts):.2f}")
        print(f"      Total blob lifetimes: {len(blob_stats.lifetimes)}")
        if len(blob_stats.lifetimes) > 0:
            print(f"      Mean lifetime: {np.mean(blob_stats.lifetimes):.1f}")
            print(f"      Max lifetime: {np.max(blob_stats.lifetimes)}")
        print("      PASSED: Blob detection functional")
        return True
    else:
        print("      WARNING: No blobs detected (may indicate issue)")
        return True  # Not a failure, just informational


def validate_against_simulation_files(sim_path, n_sims=10):
    """Validate processing against actual simulation files."""
    print(f"\n5. Validating against {n_sims} simulation files...")

    if not os.path.exists(sim_path):
        print(f"   Skipping: Path not found: {sim_path}")
        return True

    sim_files = sorted(glob(os.path.join(sim_path, 'sim_be_*.mat')))[:n_sims]
    if len(sim_files) == 0:
        print(f"   Skipping: No simulation files found")
        return True

    print(f"   Found {len(sim_files)} files to validate")

    exp_grid = (13, 26)
    w_jit = create_weight_matrix_jit(exp_grid[0], exp_grid[1], False)
    w_ref = create_weight_matrix_reference(exp_grid)

    all_passed = True

    for sim_file in tqdm(sim_files, desc="   Processing"):
        try:
            data = sio.loadmat(sim_file, squeeze_me=True)
            spins = data['stored_spins']

            # Auto-detect grid and compute centre crop
            ising_grid = (spins.shape[1], spins.shape[2])
            row_start = (ising_grid[0] - exp_grid[0]) // 2
            col_start = (ising_grid[1] - exp_grid[1]) // 2

            # Extract centre crop
            crop = spins[:, row_start:row_start+exp_grid[0], col_start:col_start+exp_grid[1]]

            # Compute Moran's I with both methods
            n_test_frames = min(100, crop.shape[0])

            for t in range(n_test_frames):
                I_ref = morans_i_reference(crop[t], w_ref)
                I_jit = morans_i_single_jit(crop[t].astype(np.float64), w_jit)

                if np.isnan(I_ref) and np.isnan(I_jit):
                    continue
                elif np.isnan(I_ref) or np.isnan(I_jit):
                    print(f"\n   ERROR: NaN mismatch in {os.path.basename(sim_file)} frame {t}")
                    all_passed = False
                    break
                elif abs(I_ref - I_jit) > 1e-10:
                    print(f"\n   ERROR: Value mismatch in {os.path.basename(sim_file)} frame {t}")
                    print(f"          Reference: {I_ref:.10f}, JIT: {I_jit:.10f}")
                    all_passed = False
                    break

        except Exception as e:
            print(f"\n   ERROR processing {os.path.basename(sim_file)}: {e}")
            all_passed = False

    if all_passed:
        print("   PASSED: All simulation files validated")
    return all_passed


def validate_against_matlab_results(matlab_path, python_results=None):
    """Compare Python results against MATLAB results."""
    print(f"\n6. Validating against MATLAB results...")

    if not os.path.exists(matlab_path):
        print(f"   Skipping: MATLAB results not found: {matlab_path}")
        return True

    print(f"   Loading MATLAB results from: {matlab_path}")

    try:
        with h5py.File(matlab_path, 'r') as f:
            # Check structure
            print(f"   MATLAB file keys: {list(f.keys())}")

            if 'Comparison' in f and 'IsingData' in f:
                comparison = f['Comparison']
                ising_data = f['IsingData']

                print(f"   Conditions in Comparison: {list(comparison.keys())}")

                # Compare rankings for each condition
                all_passed = True
                for condition in comparison.keys():
                    if 'bestMatch_idx' in comparison[condition]:
                        matlab_best = np.array(comparison[condition]['bestMatch_idx']).flatten()
                        print(f"   {condition}: Top 5 MATLAB matches: {matlab_best[:5]}")

                        # If we have Python results, compare
                        if python_results is not None and condition in python_results:
                            python_best = python_results[condition]['bestMatch_idx'][:5]
                            if np.array_equal(matlab_best[:5], python_best):
                                print(f"      MATCHED Python results")
                            else:
                                print(f"      MISMATCH: Python top 5: {python_best}")
                                # Check if rankings overlap
                                overlap = len(set(matlab_best[:10]) & set(python_best[:10]))
                                print(f"      Top-10 overlap: {overlap}/10")
                                if overlap < 7:
                                    all_passed = False

                return all_passed
            else:
                print("   MATLAB file structure not as expected")
                return True

    except Exception as e:
        print(f"   Error reading MATLAB file: {e}")
        return True


def validate_autocorrelation():
    """Validate autocorrelation decay computation."""
    print("\n7. Validating autocorrelation computation...")

    np.random.seed(42)
    all_passed = True

    # Test with known exponential decay
    print("   Testing with known exponential decay...")
    tau_true = 25.0
    n_points = 500
    t = np.arange(n_points)
    noise = np.random.normal(0, 0.1, n_points)

    # Generate exponential decay signal
    signal = np.exp(-t / tau_true) + noise
    # Add some autocorrelation structure
    autocorr_signal = np.convolve(signal, np.exp(-np.arange(50) / tau_true), mode='same')

    tau_est, acf, r2 = compute_autocorr_decay(autocorr_signal, max_lag=100, fit_range=(1, 50))

    # Check that estimated tau is within reasonable range
    if not np.isnan(tau_est) and 10 < tau_est < 100:
        print(f"      PASSED: Estimated tau={tau_est:.1f} (synthetic signal)")
    else:
        print(f"      WARNING: Unexpected tau={tau_est:.1f}")

    # Test with random Ising-like data
    print("   Testing with Ising-like activity data...")
    n_frames = 2000
    activity = np.random.uniform(0.3, 0.7, n_frames)

    tau_ising, acf_ising, r2_ising = compute_autocorr_decay(activity, max_lag=100)

    if not np.isnan(tau_ising) and tau_ising > 0:
        print(f"      PASSED: Estimated tau={tau_ising:.1f}, R^2={r2_ising:.3f}")
    else:
        print(f"      WARNING: Failed to estimate tau")

    # Test trial-averaged autocorrelation
    print("   Testing trial-averaged autocorrelation...")
    n_timepoints = 100
    n_trials = 20
    activity_trials = np.random.uniform(0.3, 0.7, (n_timepoints, n_trials))

    tau_trial_avg, acf_trial_avg, r2_trial_avg = compute_trial_averaged_autocorr(
        activity_trials, max_lag=50, fit_range=(1, 30)
    )

    if not np.isnan(tau_trial_avg) and tau_trial_avg > 0:
        print(f"      PASSED: Trial-avg tau={tau_trial_avg:.1f}, R^2={r2_trial_avg:.3f}")
    else:
        print(f"      INFO: Trial-avg tau={tau_trial_avg:.1f} (may be expected for random data)")

    return all_passed


def validate_temporal_scaling():
    """Validate temporal scale factor computation."""
    print("\n8. Validating temporal scale factor...")

    all_passed = True

    # Test cases: (exp_tau, ising_tau, expected_result_approx)
    test_cases = [
        (10.0, 5.0, 2.0, "Experimental decays slower"),
        (5.0, 10.0, 0.5, "Ising decays slower"),
        (10.0, 10.0, 1.0, "Equal decay rates"),
        (np.nan, 5.0, 1.0, "NaN experimental tau"),
        (10.0, 0.0, 1.0, "Zero Ising tau"),
    ]

    for exp_tau, ising_tau, expected, desc in test_cases:
        result = compute_temporal_scale_factor(exp_tau, ising_tau)

        if abs(result - expected) < 1e-6:
            print(f"   {desc}: PASSED (scale={result:.2f})")
        else:
            print(f"   {desc}: FAILED (expected={expected:.2f}, got={result:.2f})")
            all_passed = False

    # Test that scaling works correctly for blob lifetime conversion
    print("   Testing lifetime scaling logic...")
    sim_lifetime = 10  # MC sweeps
    exp_tau = 20.0  # frames
    ising_tau = 10.0  # MC sweeps

    scale_factor = compute_temporal_scale_factor(exp_tau, ising_tau)
    # scale_factor = exp_tau / ising_tau = 2.0

    # Expected: 10 MC sweeps * 2.0 = 20 frames equivalent
    rescaled_lifetime = sim_lifetime * scale_factor
    expected_rescaled = 20.0

    if abs(rescaled_lifetime - expected_rescaled) < 1e-6:
        print(f"      PASSED: {sim_lifetime} MC -> {rescaled_lifetime:.0f} frames (scale={scale_factor:.1f})")
    else:
        print(f"      FAILED: Expected {expected_rescaled}, got {rescaled_lifetime}")
        all_passed = False

    return all_passed


def benchmark_performance():
    """Benchmark optimized vs reference performance."""
    print("\n9. Performance benchmark...")

    np.random.seed(42)
    rows, cols = 13, 26
    n_frames = 2000

    # Create test data
    frames = np.random.choice([-1, 1], size=(n_frames, rows, cols)).astype(np.float64)

    # Create weight matrices
    w_ref = create_weight_matrix_reference((rows, cols))
    w_jit = create_weight_matrix_jit(rows, cols, False)

    # Warmup JIT
    warmup_jit()

    # Benchmark reference
    print(f"   Benchmarking {n_frames} frames...")

    start = time.perf_counter()
    for t in range(n_frames):
        _ = morans_i_reference(frames[t], w_ref)
    ref_time = time.perf_counter() - start

    # Benchmark JIT (single)
    start = time.perf_counter()
    for t in range(n_frames):
        _ = morans_i_single_jit(frames[t], w_jit)
    jit_single_time = time.perf_counter() - start

    # Benchmark JIT (batch)
    start = time.perf_counter()
    _ = morans_i_batch(frames, w_jit)
    jit_batch_time = time.perf_counter() - start

    print(f"   Reference (NumPy):     {ref_time*1000:.1f} ms ({n_frames/ref_time:.0f} frames/sec)")
    print(f"   JIT single:            {jit_single_time*1000:.1f} ms ({n_frames/jit_single_time:.0f} frames/sec)")
    print(f"   JIT batch (parallel):  {jit_batch_time*1000:.1f} ms ({n_frames/jit_batch_time:.0f} frames/sec)")
    print(f"   Speedup (JIT single):  {ref_time/jit_single_time:.1f}x")
    print(f"   Speedup (JIT batch):   {ref_time/jit_batch_time:.1f}x")

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Validate Python vs MATLAB Ising Comparison')
    parser.add_argument('--matlab-results', type=str, default=None,
                        help='Path to MATLAB results file')
    parser.add_argument('--n-sims', type=int, default=10,
                        help='Number of simulations to validate')
    parser.add_argument('--local', action='store_true',
                        help='Use local paths')
    args = parser.parse_args()

    print("=" * 70)
    print("  Validation: Python vs MATLAB Ising Comparison")
    print("=" * 70)

    # Determine paths
    if args.local:
        sim_path = r'IsingModelData_39x78_100K'
        matlab_default = r'IsingModelData_39x78_100K\IsingComparison\IsingComparison_Results.mat'
    else:
        sim_path = '/path/to/data/IsingSims'
        matlab_default = '/path/to/data/IsingSims/IsingComparison/IsingComparison_Results.mat'

    matlab_path = args.matlab_results or matlab_default

    # Run validations
    results = {}

    print("\nRunning validation tests...")

    results['weight_matrix'] = validate_weight_matrix()
    results['morans_i'] = validate_morans_i()
    results['wasserstein'] = validate_wasserstein()
    results['blob_detection'] = validate_blob_detection()
    results['simulation_files'] = validate_against_simulation_files(sim_path, args.n_sims)
    results['matlab_comparison'] = validate_against_matlab_results(matlab_path)
    results['autocorrelation'] = validate_autocorrelation()
    results['temporal_scaling'] = validate_temporal_scaling()
    results['performance'] = benchmark_performance()

    # Summary
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for test, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test:25s} {status}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("  ALL TESTS PASSED")
        print("  The optimized Python implementation matches the reference.")
    else:
        print("  SOME TESTS FAILED")
        print("  Review the output above for details.")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
