# Demos

## `demo_ising.py`: the Ising model (no data needed)

Runs the **Expert-condition best-match model** using the actual model code
(`ising-model/generate_all_ising_simulations.py::monte_carlo`) on the paper's
39×78 grid (β=0.56, c=4, decay=4, rad=13, bias=−0.8; the top-1 Expert parameter
set), and plots spin snapshots + the fraction-active timecourse. (Figure 6c
additionally applies a double-pulse bias=1.75 perturbation during the stimulus;
this demo runs the unperturbed base model.)

```bash
pip install -r requirements.txt
python demos/demo_ising.py
```

Output: `results/demo_ising.png`. Takes ~1 minute (39×78 grid, radius-13 kernel,
long burn-in; first run also compiles the Numba kernels). Edit the parameter
block at the top of `demo_ising.py` for a smaller/faster run.

## `demo_figure6c_expert.m`: reproduce Figure 6c's Expert model trace exactly

Reads the exact per-replicate model activity that Figure 6c plots for the Expert
condition (sim 3, size 3, dur 51, mode `double_pulse_bias10`, matched local
bias 1.75) from the same HDF5 leaf as
`ising-model/comparisons/Figure6c_BiasMatched_RasterTraces.m`, and plots the
50-replicate raster + mean ± SEM fraction-active timecourse on the same seconds
axis (10 Hz).

The perturbation aggregate is large and lives with the data, so point the demo
at it:
```matlab
setenv('MBA_PERTURB_AGG', '<...>\PerturbationResults_double_pulse_bias10_*.mat');  % matched variant (bias grid incl. 1.75)
demo_figure6c_expert
```
Output: `results/demo_figure6c_expert.png`.
