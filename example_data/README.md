# Data location

Data and output locations resolve here by default. With no `MBA_DATA_ROOT` set,
`config.py` (Python) and `utils/mba_config.m` (MATLAB) look in this folder; set
`MBA_DATA_ROOT` to point at your local datasets instead.

- `demos/demo_ising.py` needs **no** data — it generates its own simulation.
- `demos/demo_figure6c_expert.m` needs the double-pulse perturbation aggregate
  (large; kept with the data). Point it there via `MBA_PERTURB_AGG`, e.g.
  `setenv('MBA_PERTURB_AGG', '<...>\PerturbationResults_double_pulse_bias10_*.mat')`.

The full simulation and perturbation datasets are deposited separately —
*deposit DOI to be added*.
