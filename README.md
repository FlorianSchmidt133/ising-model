# ising-model

Accompanying code for **Schmidt et al., 2026**:
the **Ising model** of intrinsic population dynamics in the mouse superficial
superior colliculus (sSC). Spin-lattice simulation, perturbation experiments,
and the model-to-data comparison.

> The experimental **analysis** code (population entropy, spatial organisation,
> performance modulation, etc.) will be released in a separate repository.

The model is a kinetic (heat-bath / Glauber) Ising model on a 2-D lattice with
short-range excitation and a slow, adaptive **center-surround inhibition** field.
That competition produces the sparse, drifting blobs of activity seen in the
2-photon recordings. Five parameters (β inverse temperature, c inhibition
strength, decay field relaxation, rad inhibition reach, bias spiking threshold)
are fit per training condition and then perturbed with a localized stimulus.

## Interactive explainer

`docs/index.html` is a self-contained interactive explainer: it runs a faithful
in-browser port of the model's update rule so you can drive the simulation live
(sliders for β, c, decay, rad, bias) and load the per-condition best-match
presets. Double-click the file to open it locally, or publish it with **GitHub
Pages** (Settings → Pages → Deploy from a branch → **main** / **/docs**); it then
goes live at `https://<user>.github.io/<repo-name>/`.

## Repository structure

```
ising-model/          the model
├── generate_all_ising_simulations.py   generate the spin-lattice simulations
├── comparisons/                         perturbations + model-to-data comparison
│   ├── run_ising_perturbations.py
│   ├── Figure5_IsingComparison.py       (Moran's I, blob stats, Wasserstein, …)
│   └── …
├── plots/ prep/ helpers/ sensitivity_analysis/ threshold_determination/
└── Figure6.m, Pertubation_Pipeline.md   perturbation (Figure 6) figures
docs/index.html       interactive explainer (GitHub Pages)
demos/                small runnable demos
utils/                helper functions the model scripts depend on
config.py             data/output locations for the Python code
init.m                adds the code folders to the MATLAB path
requirements.txt      Python dependencies
```

## Requirements

**Python 3.10+**: `pip install -r requirements.txt`
(numpy, scipy, matplotlib, h5py, numba, scikit-learn, …).

**MATLAB** (developed on R2025a) with the Statistics/ML, Image Processing and
Signal Processing toolboxes, for the comparison/figure scripts under
`ising-model/comparisons/`.

## Demos

```bash
python demos/demo_ising.py          # Expert-condition base model, from scratch -> results/demo_ising.png
```
```matlab
% reproduce Figure 6c's Expert model trace from the perturbation aggregate:
setenv('MBA_PERTURB_AGG', '<...>\PerturbationResults_double_pulse_bias10_*.mat')
demo_figure6c_expert
```
See `demos/README.md` for details. `demo_ising.py` needs no data;
`demo_figure6c_expert.m` reads the perturbation aggregate (large; kept with the
data, pointed at via `MBA_PERTURB_AGG`).

## Data and configuration

Data/output locations are configured in one place (no need to edit the code):
`config.py` (Python) and `utils/mba_config.m` (MATLAB), both overridable via the
`MBA_DATA_ROOT` / `MBA_*` environment variables. The large simulation and
perturbation datasets are deposited separately; the DOI will be added on publication.

> **Status / license:** A license has not been finalised yet; please contact the
> authors before reusing this code.
