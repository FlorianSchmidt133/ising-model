# -*- coding: utf-8 -*-
"""
Generate All Ising Model Simulations for Figure 5 Comparison

This script generates Ising model simulations across a 5D parameter grid:
- beta: [0.4, 0.45, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.65, 0.7, 0.8] - Inverse temperature (19 values)
- c: [1, 2, 3, 4, 5, 6, 7, 8, 9] - Coupling strength (9 values)
- decay_const: [2, 4, 5, 6, 7, 8, 9, 10, 11] - Decay constant (9 values)
- rad (inhibition_range): [2, 4, 9, 13] - Inhibition kernel radius (4 values)
- bias: [-1, -0.8, -0.6, -0.4] - Bias term (4 values)

Total: 19 x 9 x 9 x 4 x 4 = 24624 simulations

Smart Resume: The script automatically scans the output directory for existing
simulations by parsing filenames. Only missing parameter combinations are run.
For example, if beta=[0.5,0.6,0.7,0.8] already exists and you request beta=[0.4-0.8],
only beta=0.4 simulations will be generated.

Usage:
    # Run all simulations (automatically skips existing)
    python generate_all_ising_simulations.py --output IsingModelData_39x78 --workers 8

    # Scan directory and show coverage report
    python generate_all_ising_simulations.py --output IsingModelData_39x78 --scan

    # Run a single simulation by index (for SLURM)
    python generate_all_ising_simulations.py --output IsingModelData_39x78 --index 0

    # List all parameter combinations
    python generate_all_ising_simulations.py --list-params
"""

import numpy as np
from scipy.io import savemat
import os
import re
import argparse
import time
from itertools import product
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from numba import njit


# =============================================================================
# Simulation Parameters
# =============================================================================

# Grid dimensions (3x experimental 13x26)
L, M = 39, 78

# Simulation steps
BURN_IN_MIN = 2000       # Minimum burn-in (safety floor)
BURN_IN_TAU_MULT = 7     # Multiplier on tau (7 => 99.9% equilibrated)
N_STEPS = 100000    # Recording steps (100K)

# Parameter grid values
# Original 19 betas for rook; extended with 10 lower values (0.2-0.38) for queen
# Queen's 8-neighbor coupling needs lower beta to match experimental activity levels
BETA_VALUES = [0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.45, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.65, 0.7, 0.8]  # 29 values
C_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]            # 9 values
DECAY_CONST_VALUES = [2, 4, 5, 6, 7, 8, 9, 10, 11] # 9 values
RAD_VALUES = [2, 4, 9, 13]                         # 4 values
BIAS_VALUES = [-1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4]  # 7 values


# =============================================================================
# Core Simulation Functions (from updated_ising_code.py)
# =============================================================================

def build_diamond_kernel(rad, queen=False):
    """
    Construct the diamond-shaped inhibition kernel.

    EXCLUDES the center (0,0) and nearest neighbors so that NN contribute
    only to excitation (J=+1) and more distant neurons contribute only to
    inhibition.  Rook excludes 4 NN; queen excludes 8 NN.
    """
    kernel = np.zeros((2*rad+1, 2*rad+1), dtype=np.float64)
    center = rad  # center index
    for i in range(rad+1):
        kernel[i, rad-i:(rad+i+1)] = 1
        kernel[-i-1, rad-i:(rad+i+1)] = 1

    # Exclude center neuron
    kernel[center, center] = 0

    # Exclude 4 nearest neighbors (rook)
    kernel[center-1, center] = 0  # top
    kernel[center+1, center] = 0  # bottom
    kernel[center, center-1] = 0  # left
    kernel[center, center+1] = 0  # right

    if queen:
        kernel[center-1, center-1] = 0  # top-left
        kernel[center-1, center+1] = 0  # top-right
        kernel[center+1, center-1] = 0  # bottom-left
        kernel[center+1, center+1] = 0  # bottom-right

    return kernel


@njit(cache=True)
def heat_bath_numba(config, beta, c, decay_const, H, K, bias_val, K_sum, queen,
                    refractory, refractory_K):
    """
    Numba-optimized Monte Carlo sweep using heat-bath algorithm.

    Parameters:
    -----------
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
    refractory : ndarray (L x M), int32
        Per-site refractory countdown. Modified in place. A site with
        refractory > 0 is locked at -1 (spin and H updates skipped) and
        refractory is decremented.
    refractory_K : int
        Refractory period in MC attempts. After a +1 -> -1 transition,
        the site is locked at -1 for refractory_K subsequent attempts.
        0 = refractory disabled.

    Returns:
    --------
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

        # Refractory: site locked at -1, skip update + H update
        if refractory[i, j] > 0:
            refractory[i, j] -= 1
            continue

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

        # Spin update (track old spin for refractory arming)
        old_spin = config[i, j]
        if np.random.random() < pi_plus:
            config[i, j] = 1
        else:
            config[i, j] = -1

        # Arm refractory on +1 -> -1 transition
        if refractory_K > 0 and old_spin == 1 and config[i, j] == -1:
            refractory[i, j] = refractory_K

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


def monte_carlo(L, M, beta, c, decay_const, rad, bias_val, n_steps,
                queen=False, refractory_K=0):
    """
    Run Ising model Monte Carlo simulation (Numba-optimized).

    Parameters:
    -----------
    L, M : int
        Grid dimensions
    beta : float
        Inverse temperature
    c : float
        Coupling strength
    decay_const : float
        Decay constant
    rad : int
        Inhibition kernel radius
    bias_val : float
        Bias value
    n_steps : int
        Number of recording steps
    queen : bool
        If True, use 8-connectivity (queen); if False, 4-connectivity (rook)
    refractory_K : int
        Refractory period in MC attempts after a +1 -> -1 transition.
        0 = disabled. Typical biologically-motivated values: 5-20.

    Returns:
    --------
    store_spins : ndarray (n_steps x L x M)
        Recorded spin configurations
    wait_time : int
        Actual burn-in used
    """
    # Initialize with proper dtypes for Numba
    spin_configuration = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, M))
    H = np.zeros((L, M), dtype=np.float64)
    refractory = np.zeros((L, M), dtype=np.int32)
    kernel = build_diamond_kernel(rad, queen)
    K_sum = float(np.sum(kernel))

    store_spins = np.zeros((n_steps, L, M), dtype=np.int8)

    # Adaptive burn-in based on H-field timescale
    tau = (L * M) / decay_const
    wait_time = max(BURN_IN_MIN, int(BURN_IN_TAU_MULT * tau))

    # Burn-in (equilibration)
    for _ in range(wait_time):
        spin_configuration, H = heat_bath_numba(
            spin_configuration, beta, c, decay_const, H, kernel, bias_val, K_sum, queen,
            refractory, refractory_K
        )

    # Recording phase
    for t in range(n_steps):
        spin_configuration, H = heat_bath_numba(
            spin_configuration, beta, c, decay_const, H, kernel, bias_val, K_sum, queen,
            refractory, refractory_K
        )
        store_spins[t] = spin_configuration.copy()

    return store_spins, wait_time


# =============================================================================
# Parameter Grid Functions
# =============================================================================

def generate_parameter_grid():
    """Generate all 24624 parameter combinations."""
    params = list(product(
        BETA_VALUES,
        C_VALUES,
        DECAY_CONST_VALUES,
        RAD_VALUES,
        BIAS_VALUES
    ))
    return params


def index_to_params(index):
    """Convert linear index to parameter tuple."""
    params = generate_parameter_grid()
    if index < 0 or index >= len(params):
        raise ValueError(f"Index {index} out of range [0, {len(params)-1}]")
    return params[index]


def params_to_dict(beta, c, decay_const, rad, bias):
    """Convert parameter tuple to dictionary."""
    return {
        'beta': beta,
        'c': c,
        'decay_const': decay_const,
        'inhibition_range': rad,
        'bias': bias
    }


def params_to_filename(beta, c, decay_const, rad, bias, extension=None,
                        connectivity='rook', refractory_K=0):
    """
    Generate filename from parameters.

    Format: sim_be_{beta}_c_{c}_d_{decay}_r_{rad}_bi_{bias}[_queen][_rK{N}][.ext]
    Examples:
        sim_be_0.5_c_4_d_6_r_9_bi_-0.8.mat              # rook, K=0
        sim_be_0.5_c_4_d_6_r_9_bi_-0.8_queen.mat        # queen, K=0
        sim_be_0.5_c_4_d_6_r_9_bi_-0.8_rK10.mat         # rook, K=10
        sim_be_0.5_c_4_d_6_r_9_bi_-0.8_queen_rK10.mat   # queen, K=10

    Parameters:
    -----------
    beta, c, decay_const, rad, bias : parameter values
    extension : str, optional
        File extension (e.g., '.mat', '.npz'). If None, returns base filename.
    connectivity : str
        'rook' or 'queen'. Queen appends '_queen' to the base filename.
    refractory_K : int
        Refractory period. When > 0, appends '_rK{N}' to filename.

    Returns:
    --------
    str : Filename with or without extension
    """
    base = f"sim_be_{beta}_c_{int(c)}_d_{int(decay_const)}_r_{int(rad)}_bi_{bias}"
    if connectivity == 'queen':
        base += '_queen'
    if refractory_K > 0:
        base += f'_rK{int(refractory_K)}'
    if extension:
        return base + extension
    return base


def filename_to_params(filename):
    """
    Parse parameters from filename.

    Parameters:
    -----------
    filename : str
        Filename like 'sim_be_0.5_c_4_d_6_r_9_bi_-0.8.mat',
        'sim_be_0.5_c_4_d_6_r_9_bi_-0.8_queen.npz', or
        'sim_be_0.5_c_4_d_6_r_9_bi_-0.8_queen_rK10.mat'

    Returns:
    --------
    tuple or None
        (beta, c, decay_const, rad, bias) or None if parsing fails.
        Connectivity and refractory markers are stripped; use the helpers
        below to inspect them. Filtering by markers is handled by
        scan_existing_simulations.
    """
    pattern = (r'sim_be_([\d.]+)_c_(\d+)_d_(\d+)_r_(\d+)_bi_([-\d.]+?)'
               r'(_queen)?(_rK\d+)?\.(mat|npz)')
    match = re.match(pattern, os.path.basename(filename))
    if match:
        return (
            float(match.group(1)),  # beta
            int(match.group(2)),    # c
            int(match.group(3)),    # decay_const
            int(match.group(4)),    # rad
            float(match.group(5))   # bias
        )
    return None


def filename_has_queen(filename):
    """Return True if the filename encodes queen connectivity."""
    return bool(re.search(r'_queen(_rK\d+)?\.(mat|npz)$', os.path.basename(filename)))


def filename_get_refractory(filename):
    """Return the refractory K encoded in the filename (0 if absent)."""
    m = re.search(r'_rK(\d+)\.(mat|npz)$', os.path.basename(filename))
    return int(m.group(1)) if m else 0


def scan_existing_simulations(output_dir, format_type='both',
                               connectivity='rook', refractory_K=0):
    """
    Scan directory and return set of existing parameter tuples.

    Only files whose connectivity AND refractory_K marker match the
    requested values are counted.
    """
    existing = set()
    if not os.path.isdir(output_dir):
        return existing

    extensions = []
    if format_type in ('mat', 'both'):
        extensions.append('.mat')
    if format_type in ('npz', 'both'):
        extensions.append('.npz')

    want_queen = (connectivity == 'queen')
    for ext in extensions:
        pattern = os.path.join(output_dir, f'sim_be_*{ext}')
        for filepath in glob(pattern):
            if filename_has_queen(filepath) != want_queen:
                continue
            if filename_get_refractory(filepath) != refractory_K:
                continue
            params = filename_to_params(filepath)
            if params:
                existing.add(params)
    return existing


def get_missing_simulations(output_dir, format_type='both',
                             connectivity='rook', refractory_K=0):
    """
    Compare requested parameter grid against existing simulations.

    Returns (missing_sorted, existing_set) for the given connectivity +
    refractory_K combination.
    """
    requested = set(generate_parameter_grid())
    existing = scan_existing_simulations(output_dir, format_type,
                                         connectivity, refractory_K)
    missing = requested - existing
    return sorted(list(missing)), existing


# =============================================================================
# Simulation Runner
# =============================================================================

def run_single_simulation(args):
    """
    Run a single simulation and save to file.

    Parameters:
    -----------
    args : tuple
        Accepts 2-, 3-, 4- and 5-tuples for backward compatibility:
        - (params_tuple, output_dir)
        - (params_tuple, output_dir, format_type)
        - (params_tuple, output_dir, format_type, queen)
        - (params_tuple, output_dir, format_type, queen, refractory_K)

    Returns:
    --------
    dict : Result info including params, status, time
    """
    refractory_K = 0
    if len(args) == 2:
        params_tuple, output_dir = args
        format_type = 'both'
        queen = False
    elif len(args) == 3:
        params_tuple, output_dir, format_type = args
        queen = False
    elif len(args) == 4:
        params_tuple, output_dir, format_type, queen = args
    else:
        params_tuple, output_dir, format_type, queen, refractory_K = args

    beta, c, decay_const, rad, bias = params_tuple
    connectivity = 'queen' if queen else 'rook'

    # Generate base filename from parameters
    base_filename = params_to_filename(
        beta, c, decay_const, rad, bias,
        connectivity=connectivity, refractory_K=refractory_K
    )

    start_time = time.time()

    # Run simulation
    store_spins, wait_time = monte_carlo(
        L, M, beta, c, decay_const, rad, bias, N_STEPS,
        queen=queen, refractory_K=refractory_K
    )

    # Prepare data for saving
    params_dict = {
        'beta': beta,
        'c': c,
        'decay_const': decay_const,
        'inhibition_range': rad,
        'bias': bias,
        'refractory_K': refractory_K,
    }
    metadata_dict = {
        'L': L,
        'M': M,
        'n_steps': N_STEPS,
        'wait_time': wait_time
    }

    # Save to .mat file
    if format_type in ('mat', 'both'):
        output_file_mat = os.path.join(output_dir, base_filename + '.mat')
        data_to_save = {
            'stored_spins': store_spins,
            'params': params_dict,
            'metadata': metadata_dict
        }
        savemat(output_file_mat, data_to_save)

    # Save to .npz file
    if format_type in ('npz', 'both'):
        output_file_npz = os.path.join(output_dir, base_filename + '.npz')
        np.savez_compressed(
            output_file_npz,
            stored_spins=store_spins,
            params=params_dict,
            metadata=metadata_dict
        )

    elapsed = time.time() - start_time

    return {'params': params_tuple, 'status': 'completed', 'time': elapsed}


def run_all_simulations_parallel(output_dir, workers, format_type='both',
                                  queen=False, refractory_K=0):
    """
    Run simulations in parallel, automatically skipping existing ones.

    Scans the output directory for existing simulations and only runs
    parameter combinations that are missing.

    Parameters:
    -----------
    output_dir : str
        Output directory path
    workers : int
        Number of parallel workers
    format_type : str
        'mat', 'npz', or 'both' - output format(s) to save
    queen : bool
        If True, use queen (8-connectivity); if False, rook (4-connectivity)
    refractory_K : int
        Refractory period (0 = disabled).
    """
    os.makedirs(output_dir, exist_ok=True)
    connectivity = 'queen' if queen else 'rook'

    # Smart scan: find what's missing (matched on connectivity + refractory)
    missing, existing = get_missing_simulations(
        output_dir, format_type, connectivity, refractory_K
    )
    total_requested = len(generate_parameter_grid())

    print(f"Requested simulations: {total_requested}")
    print(f"Already exist: {len(existing)}")
    print(f"Need to run: {len(missing)}")
    print(f"Grid size: {L} x {M}")
    print(f"Steps: {N_STEPS} recording + adaptive burn-in (min {BURN_IN_MIN}, {BURN_IN_TAU_MULT}x tau)")
    print(f"Workers: {workers}")
    print(f"Output: {output_dir}")
    print(f"Format: {format_type}")
    print()

    if len(missing) == 0:
        print("All simulations already exist. Nothing to do.")
        return

    # Prepare arguments for missing simulations only
    args_list = [(params, output_dir, format_type, queen, refractory_K) for params in missing]

    start_time = time.time()

    with Pool(workers) as pool:
        results = list(tqdm(
            pool.imap(run_single_simulation, args_list),
            total=len(missing),
            desc="Simulations"
        ))

    completed = sum(1 for r in results if r['status'] == 'completed')
    total_time = time.time() - start_time

    print()
    print("=" * 50)
    print(f"Completed: {completed}")
    print(f"Total time: {total_time/3600:.2f} hours")
    if completed > 0:
        print(f"Avg time per sim: {total_time/completed:.1f} seconds")
    print("=" * 50)


def run_single_simulation_cli(index, output_dir, format_type='both',
                                queen=False, refractory_K=0):
    """Run a single simulation by index (for SLURM)."""
    os.makedirs(output_dir, exist_ok=True)
    connectivity = 'queen' if queen else 'rook'

    params_tuple = index_to_params(index)
    beta, c, decay_const, rad, bias = params_tuple
    base_filename = params_to_filename(
        beta, c, decay_const, rad, bias,
        connectivity=connectivity, refractory_K=refractory_K
    )

    # Check if already exists (for any requested format)
    mat_exists = os.path.exists(os.path.join(output_dir, base_filename + '.mat'))
    npz_exists = os.path.exists(os.path.join(output_dir, base_filename + '.npz'))

    skip = False
    if format_type == 'mat' and mat_exists:
        skip = True
    elif format_type == 'npz' and npz_exists:
        skip = True
    elif format_type == 'both' and mat_exists and npz_exists:
        skip = True

    if skip:
        print(f"Simulation {index} SKIPPED (already exists)")
        print(f"  beta={beta}, c={c}, decay={decay_const}, rad={rad}, bias={bias}")
        print(f"  File: {base_filename}")
        return

    print(f"Simulation {index}")
    print(f"  beta={beta}, c={c}, decay={decay_const}, rad={rad}, bias={bias}")
    print(f"  Refractory K: {refractory_K}")
    print(f"  Output: {base_filename}")
    print(f"  Grid: {L} x {M}")
    tau = (L * M) / decay_const
    burn_in = max(BURN_IN_MIN, int(BURN_IN_TAU_MULT * tau))
    print(f"  Steps: {N_STEPS} + {burn_in} burn-in (tau={tau:.0f})")
    print(f"  Format: {format_type}")

    result = run_single_simulation(
        (params_tuple, output_dir, format_type, queen, refractory_K)
    )

    print(f"  Status: {result['status']}")
    print(f"  Time: {result['time']:.1f} seconds")


def print_scan_report(output_dir, format_type='both', connectivity='rook',
                       refractory_K=0):
    """Print a coverage report showing existing vs requested simulations."""
    missing, existing = get_missing_simulations(
        output_dir, format_type, connectivity, refractory_K
    )
    total_requested = len(generate_parameter_grid())

    print("=" * 60)
    print("  Ising Simulation Coverage Report")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nRequested parameter grid: {total_requested} combinations")
    print(f"  beta: {BETA_VALUES}")
    print(f"  c: {C_VALUES}")
    print(f"  decay: {DECAY_CONST_VALUES}")
    print(f"  rad: {RAD_VALUES}")
    print(f"  bias: {BIAS_VALUES}")
    print()
    print(f"Existing simulations: {len(existing)}")
    print(f"Missing simulations: {len(missing)}")
    print()

    if len(existing) > 0:
        # Show coverage per parameter
        existing_betas = sorted(set(p[0] for p in existing))
        existing_c = sorted(set(p[1] for p in existing))
        existing_decay = sorted(set(p[2] for p in existing))
        existing_rad = sorted(set(p[3] for p in existing))
        existing_bias = sorted(set(p[4] for p in existing))

        print("Existing parameter coverage:")
        print(f"  beta: {existing_betas}")
        print(f"  c: {existing_c}")
        print(f"  decay: {existing_decay}")
        print(f"  rad: {existing_rad}")
        print(f"  bias: {existing_bias}")
        print()

    if len(missing) > 0 and len(missing) <= 20:
        print("Missing combinations:")
        for params in missing:
            print(f"  beta={params[0]}, c={params[1]}, decay={params[2]}, rad={params[3]}, bias={params[4]}")
    elif len(missing) > 20:
        print(f"Missing combinations: {len(missing)} (too many to list)")
        # Show sample
        print("First 5 missing:")
        for params in missing[:5]:
            print(f"  beta={params[0]}, c={params[1]}, decay={params[2]}, rad={params[3]}, bias={params[4]}")

    print()
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Ising model simulations for Figure 5 comparison'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=r'IsingModelData_39x78',
        help='Output directory for simulation files'
    )
    parser.add_argument(
        '--index', '-i',
        type=int,
        default=None,
        help='Run single simulation by index (0-24623). If not specified, runs all.'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=max(1, cpu_count() - 1),
        help='Number of parallel workers (default: CPU count - 1)'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['mat', 'npz', 'both'],
        default='both',
        help='Output format: mat, npz, or both (default: both)'
    )
    parser.add_argument(
        '--scan',
        action='store_true',
        help='Scan output directory and show coverage report (no simulations run)'
    )
    parser.add_argument(
        '--list-params',
        action='store_true',
        help='List all parameter combinations and exit'
    )
    parser.add_argument(
        '--connectivity',
        type=str,
        choices=['rook', 'queen'],
        default='rook',
        help='Neighbor connectivity: rook (4-connected) or queen (8-connected)'
    )
    parser.add_argument(
        '--refractory',
        type=int,
        default=0,
        help='Refractory period in MC attempts after a +1->-1 transition '
             '(default 0 = disabled; biologically motivated values: 5-20)'
    )

    args = parser.parse_args()
    queen = (args.connectivity == 'queen')
    refractory_K = int(args.refractory)

    # List parameters mode
    if args.list_params:
        params = generate_parameter_grid()
        print(f"Total combinations: {len(params)}")
        print()
        print("Index | beta | c | decay | rad | bias")
        print("-" * 45)
        for i, (beta, c, decay, rad, bias) in enumerate(params):
            print(f"{i:5d} | {beta:.1f} | {c} |   {decay}   | {rad:2d} | {bias:.1f}")
        return

    # Scan mode - show coverage report
    if args.scan:
        print_scan_report(args.output, args.format, args.connectivity, refractory_K)
        return

    # Single simulation mode (for SLURM)
    if args.index is not None:
        run_single_simulation_cli(args.index, args.output, args.format,
                                   queen, refractory_K)
        return

    # Parallel mode - automatically skips existing simulations
    run_all_simulations_parallel(args.output, args.workers, args.format,
                                  queen, refractory_K)


if __name__ == '__main__':
    main()
