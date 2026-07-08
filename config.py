"""Central data/output locations for the Ising-model code.

This is the single source of truth for where the Python pipeline reads input
data and writes results. Edit the defaults below, or override them without
touching code via environment variables:

    MBA_DATA_ROOT          root folder for all input/output data
    MBA_ISING_MODEL_DATA   precomputed Ising-model simulations
    MBA_ISING_PERTURBATIONS Ising perturbation outputs
    MBA_EXPERIMENTAL_DATA   experimental dataset (.mat)
    MBA_ISING_COMPARISON    Ising-vs-experiment comparison outputs
    MBA_FIGURE_ROOT         where figures are written

When unset, everything defaults under <repo>/example_data so the code runs
against the bundled example data out of the box.
"""

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val) if val else default


# Root for all data (inputs and outputs). Override with MBA_DATA_ROOT.
DATA_ROOT = _env_path("MBA_DATA_ROOT", _REPO / "example_data")

# Named locations (each individually overridable).
ISING_MODEL_DATA = _env_path("MBA_ISING_MODEL_DATA", DATA_ROOT / "IsingModelData")
ISING_PERTURBATIONS = _env_path("MBA_ISING_PERTURBATIONS", DATA_ROOT / "IsingPerturbations")
EXPERIMENTAL_DATA = _env_path("MBA_EXPERIMENTAL_DATA", DATA_ROOT / "ExperimentalData" / "ExperimentalData.mat")
ISING_COMPARISON = _env_path("MBA_ISING_COMPARISON", DATA_ROOT / "IsingComparison")
FIGURE_ROOT = _env_path("MBA_FIGURE_ROOT", _REPO / "results")


def data_path(*parts) -> str:
    """Resolve a path under DATA_ROOT, returned as a string."""
    return str(DATA_ROOT.joinpath(*parts))
