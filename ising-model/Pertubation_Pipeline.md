# Perturbation Pipeline (Figure 6)

Reference doc for the Ising perturbation analysis that produces Figure 6 of the paper.
Scope: `double_pulse10` (clamped) and `double_pulse_bias10` × 6 bias values.
Last updated: 2026-05-04.

---

## TL;DR

- **Goal**: drop a stimulus region into a parameter-matched Ising sim per behavioural condition (Naive/Beginner/Expert/NoSpout) and quantify how activity propagates.
- **Two stim mode flavours used in Fig. 6**: `double_pulse10` (region clamped to +1 at stim onset and offset) and `double_pulse_bias10` (additive local bias of 0.25 / 0.5 / 1.0 / 2.0 / 4.0 / 8.0 instead of a hard clamp).
- **Run on cluster, pull figures back to local**. Single orchestrator: `bash submit_dp10_pipeline.sh`.

---

## What the perturbation analysis does

For each behavioural condition, we previously parameter-matched an Ising simulation in the comparison stage (Figure 5). The perturbation stage takes those matched sims and applies a synthetic stimulus to a square region of the lattice for a fixed duration, then watches how spin activity spreads. The output of interest is the per-frame fraction of `+1` cells, plus blob-extent / Moran's I / propagation-velocity time series.

`double_pulse10` (clamped) is the strict version — the stim region is forced to +1 at stim onset and again at stim offset (10 MC sweeps apart). `double_pulse_bias10` is the soft version: instead of clamping, an additive bias is applied to the same region's local field. The 6 bias values let us trace a dose-response from "barely a perturbation" (0.25) to "effectively clamped" (8.0).

The downstream goal is to find which bias value best matches each experimental condition's measured fraction-active trajectory — see the **Matching process** section.

---

## Mode taxonomy

| Stim mode (CLI) | Flavour | What changes |
|---|---|---|
| `double_pulse10` | Clamped | Region forced to +1 at onset and offset (10 sweeps gap). One run per (cond, sim, size, dur). |
| `double_pulse_bias10_0p25` | Bias | Local field gets `+0.25` added inside the region during stim. Weakest perturbation. |
| `double_pulse_bias10_0p50` | Bias | `+0.50` |
| `double_pulse_bias10_1p00` | Bias | `+1.00` |
| `double_pulse_bias10_2p00` | Bias | `+2.00` |
| `double_pulse_bias10_4p00` | Bias | `+4.00` |
| `double_pulse_bias10_6p00` | Bias | `+6.00` (added 2026-05-04 for finer mid-range resolution) |
| `double_pulse_bias10_8p00` | Bias | `+8.00` |
| `double_pulse_bias10_10p00` | Bias | `+10.00` (added 2026-05-04 to bracket Beginner/Expert offset) |
| `double_pulse_bias10_12p00` | Bias | `+12.00` (added 2026-05-04). Strongest in current sweep; close to clamped behaviour. |

Underlying constants (`run_ising_perturbations.py:67-118`): `PRE_STIM_FRAMES=400`, `POST_STIM_FRAMES=300`, `SAMPLING_RATE=10 Hz`, `STIMULUS_SIZES=[1, 2, 3, 4, 6, 8, 10, 12]`, `N_TOP_MATCHES=10` (sims per condition), `STIMULUS_BIAS_VALUES=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]`.

The other bias-mode families (`bias`, `double_pulse_bias`, `_bias3`, `_bias5`) exist in `STIMULUS_MODES` but are not used in Figure 6.

---

## Pipeline stages

| # | Stage | Script | Output |
|---|---|---|---|
| 1 | Baseline Ising sims | `Figure5/generate_all_ising_simulations.py` via `Figure5/run_ising_slurm.sh` | `~/IsingSims/sim_be_*_c_*_d_*_r_*_bi_*.mat` |
| 2 | Parameter matching (Figure 5 work) | `Figure5/comparisons/Figure5_IsingComparison_optimized.py` | `~/IsingSims/IsingComparison/<metric>/IsingComparison_Results_*.mat` |
| 3 | Perturbations (per-combo runs) | `Figure5/comparisons/run_ising_perturbations.py` (orchestrated by `submit_dp10_pipeline.sh`) | `~/IsingPerturbations/<metric>/perturb_<cond>_sim<i>_size<s>_dur<d>_<mode>[_bias<v>].mat` |
| 4 | Aggregation | `run_ising_perturbations.py --combine --mode <mode>` (called via `run_ising_pipeline.sh combine`) | `~/IsingPerturbations/<metric>/PerturbationResults_<rawMode>_<timestamp>.mat` |
| 5 | MATLAB analysis (figures) | `Figure5/comparisons/Figure5_IsingPerturbationAnalysis.m` (called via `run_ising_pipeline.sh analyze`) | `~/IsingPerturbations/<metric>/<stim_mode>/Analysis/dur_<N>/...` |
| 6 | Matching ↔ experiment (local) | `Figure5/comparisons/match_experimental_to_isingBias.m` | `Paper/.../BiasMatchExperiment/BiasMatch_*.png` + `BiasMatch_Summary.csv` |

`<metric>` defaults to `moransI+activity` for Figure 6 work.

---

## Scripts overview

All paths are relative to `Neuron Activity Analysis/main_scripts/Figure5/comparisons/` unless noted.

| Script | Role | Key flags |
|---|---|---|
| `submit_dp10_pipeline.sh` | **Top-level orchestrator** for the dp10 + dp_bias10 chain. Submits perturb → combine → 7 parallel analyze sbatch jobs. | `--metric`, `--frame-label`, `--n-replicates`, `--force`, `--skip-perturb`, `--analyze-only`, `--dry-run` |
| `run_ising_pipeline.sh` | Stage dispatcher. Modes: `fullPipeline`, `comparison`, `perturbations`, `combine`, `analyze`, `retry`, `figures`, `snapshots`. | `--stim-mode <mode>` (analyze), `--modes <csv>` (perturbations), `--metric`, `--n-replicates` |
| `run_ising_perturbations.py` | Per-combo Ising perturbation runner. SLURM array task index → one (cond, sim, size, dur, mode, bias) job. | `--index`, `--batch-size`, `--modes`, `--n-replicates`, `--force`, `--combine --mode <m>`, `--count-jobs` |
| `Figure5_IsingPerturbationAnalysis.m` | MATLAB analysis. Accepts `config.stimMode = 'double_pulse10'` or bias-encoded `'double_pulse_bias10_2p00'` etc. — slices the 5-D bias data once, then everything downstream is 4-D. | `config.stimMode`, `config.perturbationResultsPath`, `config.outputPath`, `config.figuresOnly` |
| `match_experimental_to_isingBias.m` | Local matcher. Finds best-matching bias value per condition by RMSE over fraction-active. | `Mode`, `StimulusSize`, `MatchWindow`, `Conditions`, `ExperimentalDataFile`, `PerturbationResultsPath` |
| `generate_perturbation_snapshots.py` | Optional. Produces PNG snapshot grids and heatmap analyses from per-combo files. | `--stim-mode`, `--mode snapshots/heatmap/all`, `--n-reps` |

---

## File locations

### Cluster (`cluster`, user `your-user`)

| What | Path |
|---|---|
| Repo | `~/repo` |
| Baseline Ising sims | `~/IsingSims/sim_*.mat` |
| Comparison results | `~/IsingSims/IsingComparison/<metric>/IsingComparison_Results_*.mat` |
| Per-combo perturbations | `~/IsingPerturbations/<metric>/perturb_<cond>_*.mat` |
| Per-mode aggregates | `~/IsingPerturbations/<metric>/PerturbationResults_<rawMode>_*.mat` |
| Analysis figures | `~/IsingPerturbations/<metric>/<stim_mode>/Analysis/dur_<N>/...` |
| SLURM logs | `~/IsingPerturbations/<metric>/logs/dp10_*.{out,err}` |

### Local (Windows, ``)

| What | Path |
|---|---|
| Baseline Ising sims (mirror) | `IsingModelData_39x78_100K\` |
| Per-combo + aggregated perturbations | `IsingModelData_39x78_100K\IsingPerturbations\` |
| Analysis figures (after rsync from cluster) | `Paper\Fig. 5 Model\PerturbationAnalysis\<stim_mode>\dur_<N>\...` |
| Experimental binarised data (matcher input) | `Paper\Fig. 5 Ising Models\ExperimentalData.mat` |
| Matcher outputs | `Paper\Fig. 5 Model\PerturbationAnalysis\BiasMatchExperiment\` |

`<metric>` is `moransI+activity` for Figure 6.

---

## How to run end-to-end

All commands run on `the cluster` from `~/repo/ising-model/comparisons/`.

### Full chain (perturb → combine → 7-way analyze)

```bash
bash submit_dp10_pipeline.sh
```

Wall time: 3–8 h depending on cluster load. Submits:
1. Perturb array (~2k tasks at batch-size 10).
2. Two combine jobs (one for `double_pulse10`, one for `double_pulse_bias10`).
3. **7 independent analyze sbatch jobs**, one per stim mode: `double_pulse10` + `double_pulse_bias10_{0p25,0p50,1p00,2p00,4p00,8p00}`. Each gets its own 4 h slot.

### Skip generation, re-run analyze only

If aggregates already exist and you only want fresh figures:

```bash
bash submit_dp10_pipeline.sh --analyze-only
```

Submits the 7 analyze jobs immediately with no upstream dependencies. ~1 h wall (assumes scheduler gives all 7 a slot in parallel).

### Regenerate at 150 replicates per per-combo file

Existing aggregates were built from 20-replicate per-combo files. To regenerate at 150:

```bash
bash submit_dp10_pipeline.sh --force
```

`--force` causes `run_ising_perturbations.py` to overwrite existing outputs.

### Pull figures back to local

After analyze finishes:

```bash
# From Windows / WSL:
rsync -avz --include='*/' --include='*.png' --include='*.pdf' --exclude='*' \
    user@cluster:'~/IsingPerturbations/moransI+activity/double_pulse10/Analysis/' \
    'Paper/Fig. 5 Model/PerturbationAnalysis/double_pulse10/'

for b in 0p25 0p50 1p00 2p00 4p00 8p00; do
    rsync -avz --include='*/' --include='*.png' --include='*.pdf' --exclude='*' \
        user@cluster:"~/IsingPerturbations/moransI+activity/double_pulse_bias10_$b/Analysis/" \
        "Paper/Fig. 5 Model/PerturbationAnalysis/double_pulse_bias10_$b/"
done
```

The orchestrator script also prints these commands when it finishes submitting.

### Pull aggregated `.mat` files (needed for the matcher)

```bash
rsync -avz \
    user@cluster:'~/IsingPerturbations/moransI+activity/PerturbationResults_double_pulse_bias10_*.mat' \
    'IsingModelData_39x78_100K/IsingPerturbations/'
```

---

## The matching process

`match_experimental_to_isingBias.m` is the local-only matcher that finds the bias value within `double_pulse_bias10` whose simulated fraction-active trajectory best matches each experimental condition's measured trajectory.

### What it does

1. Loads `BinarisedData.<cond>` from `Paper\Fig. 5 Ising Models\ExperimentalData.mat`. For each condition, builds a per-frame fraction-active trace as `mean(active_pixel × any_trial)` — same masking convention as `analyze_ExperimentalData_FractionActive.m`.
2. Loads the most recent `PerturbationResults_double_pulse_bias10_*.mat` from `IsingModelData_39x78_100K\IsingPerturbations\`. Slices the 5-D `Data.<cond>.doublepulsebias10.activity` array per bias value and averages across (sims, replicates) to get one trace per bias value per condition.
3. Aligns both at stim onset (experimental frame 81 = `t=0`; Ising `pre_stim_frames+1 = 401`) and clips to `MatchWindow = [-1, 3]` s.
4. Scores each (condition, bias_value) pair via **RMSE** on the clipped trace. Lowest RMSE wins.
5. Writes per-condition overlay PNGs and `BiasMatch_Summary.csv`.

### Inputs (defaultable)

| Param | Default |
|---|---|
| `ExperimentalDataFile` | `Paper\Fig. 5 Ising Models\ExperimentalData.mat` |
| `PerturbationResultsPath` | `IsingModelData_39x78_100K\IsingPerturbations` |
| `Mode` | `double_pulse_bias10` |
| `StimulusSize` | `4` (px) |
| `MatchWindow` | `[-1, 3]` s |
| `SamplingRate` | `10` Hz |
| `Conditions` | `{'Naive', 'Beginner', 'Expert'}` |

### Outputs

```
Paper\Fig. 5 Model\PerturbationAnalysis\BiasMatchExperiment\
├── BiasMatch_Naive_size4.png        # overlay: experimental + 6 candidate sim traces
├── BiasMatch_Beginner_size4.png
├── BiasMatch_Expert_size4.png
├── BiasMatch_Summary.png            # bar of best bias per condition
└── BiasMatch_Summary.csv            # full RMSE matrix + IsBest flag
```

### Input file note

The matcher reads `BinarisedData.<cond>` from `Paper\Fig. 5 Ising Models\ExperimentalData.mat`, NOT directly from `Grid40.mat`. The two are equivalent for binarisation purposes — `Figure5_dataAggregation.m` builds `ExperimentalData.mat` *from* `Grid40.mat` using the same `dF/F = 2.0` threshold that `plot_FractionActive_BeforeDuring.m` applies on the fly. The differences:

- **Trial set**: `ExperimentalData.mat` includes only `.P1` (position-1 stimulus) trials. The figure plot uses `.All`. P1-only is the right scope for matching against a single-region Ising perturbation, since the simulation places its stim in one fixed grid region — including P2 trials would introduce a position mismatch.
- **RT filtering**: the figure plot applies `DisplayWindow=[-1,3]` to drop trials with RTs outside that window. `ExperimentalData.mat` does not. The matcher's per-condition trial-mean uses every P1 trial.
- **Pre-binarised vs raw**: `ExperimentalData.mat` is already 0/1; consuming it is trivial.

If at some point you want to reproduce the heatmap trace exactly, fork the matcher to read `Grid40.mat` and replicate the threshold + RT-filter steps. Otherwise stick with `ExperimentalData.mat`.

### Run locally (after the 9.2 GB aggregate is rsync'd to )

In MATLAB, from anywhere on `addpath`:

```matlab
out = match_experimental_to_isingBias();
% Or override defaults:
out = match_experimental_to_isingBias('StimulusSize', 6, 'MatchWindow', [-1.5, 4]);
```

`out.bestBiasPerCondition.<cond>.bias_value` is the answer; `out.scores` is the full RMSE matrix.

### Run on cluster (skip the 9.2 GB transfer)

`Figure5/comparisons/submit_match_dp10_bias.sh` runs the matcher inside a SLURM MATLAB job that reads the aggregate already on cluster. Prerequisites: upload `ExperimentalData.mat` once to `~/IsingPerturbations/moransI+activity/ExperimentalData.mat`. Submit:

```bash
sbatch ~/repo/ising-model/comparisons/submit_match_dp10_bias.sh
```

Outputs land at `~/IsingPerturbations/moransI+activity/BiasMatchExperiment/`. Pull back via:

```bash
rsync -avz user@cluster:'~/IsingPerturbations/moransI+activity/BiasMatchExperiment/' \
    'Paper/Fig. 5 Model/PerturbationAnalysis/BiasMatchExperiment/'
```

---

## Snapshot generation (optional)

Snapshots are an optional visualisation branch parallel to the analyze stage. They consume the same per-combo `perturb_*.mat` files (stage 3 output) but produce illustrative PNGs rather than time-course metrics. Useful when picking example panels for a figure or sanity-checking that a stim mode actually does what its name says.

Driven by `Figure5/comparisons/generate_perturbation_snapshots.py`.

### Two output flavours

The script's `--mode` flag controls what gets produced:

| `--mode` | What it does | Replicate count |
|---|---|---|
| `snapshots` (default) | Per-replicate grid visualisations: lattice state at pre-stim / stim onset / mid-stim / stim offset / post-stim. One PNG per (cond, size, dur, replicate). Plus detailed-timeline grids and per-condition example panels. | `DEFAULT_EXAMPLE_REPS = 15`, with overrides `REPS_PER_SIZE = {2: 20, 3: 20, 4: 20}` (sizes 2/3/4 get a few extra reps because the per-rep variability is higher at small stim sizes). |
| `heatmap` | `P(active)` probability heatmaps and wavefront-asymmetry metrics, averaged over many replicates. | `--n-reps` flag, default 100. Higher values give cleaner heatmaps. |
| `all` | Both. | as above |

### Mode coverage

`SNAPSHOT_STIM_MODES` (`generate_perturbation_snapshots.py:80-103`) was recently extended to enumerate all 6 bias values per family. Valid `--stim-mode` choices for Figure 6:

```
clamped
double_pulse10
double_pulse_bias10_0p25
double_pulse_bias10_0p50
double_pulse_bias10_1p00
double_pulse_bias10_2p00
double_pulse_bias10_4p00
double_pulse_bias10_8p00
```

(The other bias families — `bias`, `double_pulse_bias{,3,5}` — are also enumerated for completeness.)

### Run via the orchestrator (one-liner for all 7 dp10 modes)

```bash
bash submit_dp10_pipeline.sh --snapshots-only
```

Iterates dp10 + the 6 dp_bias10 bias variants, calling `run_ising_pipeline.sh snapshots` for each (login-node invocation; that script does its own SLURM submissions internally). Each mode produces a 6-job chain (snapshot workers + combine + figures, plus heatmap workers + combine + figures), so 7 modes × 6 = **42 SLURM jobs** total. Override the heatmap replicate count with `--snapshot-n-reps 200` (default 100).

### Run via SLURM (single mode)

```bash
# From the cluster, in the comparisons/ dir
bash run_ising_pipeline.sh snapshots --metric moransI+activity \
    --stim-mode double_pulse_bias10_2p00
```

Submits the same 6-job chain for one stim mode. See `run_ising_pipeline.sh`'s docstring for `--n-reps`, `--scan` (dry run), `--figures-only`, `--heatmap-only` variants.

`run_ising_pipeline.sh snapshots --stim-mode all` and `--stim-mode dp10_full` also exist as built-in fan-outs (the latter is identical to the orchestrator's `--snapshots-only`).

### Run directly (no SLURM, smaller datasets)

```bash
python generate_perturbation_snapshots.py \
    --output ~/IsingPerturbations/moransI+activity/snapshots \
    --comparison ~/IsingSims/IsingComparison/moransI+activity/IsingComparison_Results_full_trial_subselect_centre_vs_tiled_moransI+activity_optimized.mat \
    --stim-mode double_pulse_bias10_2p00 \
    --mode snapshots
```

Use `--scan` first to see how many output files would be produced.

### Output layout

Cluster:

```
~/IsingPerturbations/moransI+activity/snapshots/<stim_mode>/
├── snapshots/                         # --mode snapshots PNGs
│   ├── snapshot_size_<S>_<stim_mode>.png            # main grid
│   ├── snapshot_detail_size_<S>_<stim_mode>.png     # detailed timeline
│   ├── snapshot_expert_examples_size_<S>_<stim_mode>.png
│   ├── snapshot_expert_nospout_examples_<N>_size_<S>_<stim_mode>.png
│   └── snapshot_<cond>_rep<R>_<crop>_size_<S>_<stim_mode>.png
└── heatmap/                           # --mode heatmap outputs
    ├── heatmap_data.npz               # cached probability arrays
    └── *.png                          # rendered heatmaps + asymmetry
```

Pull back to local with the same rsync pattern as the analysis figures.

### When you'd run snapshots

- Picking a representative panel for a figure: run with `--mode snapshots --stim-mode <one mode>`, browse, copy the chosen PNG into `Paper\Fig. 6 ...`.
- Comparing two bias values visually before running the full analyze chain.
- Producing the `P(active)` heatmaps that show propagation asymmetry — `--mode heatmap --n-reps 200` for publication-quality.

Snapshots are **not** required for the matcher or for the dose-response figures; the analyze stage covers those independently.

---

## Progress so far (as of 2026-05-04, late afternoon)

### Code / orchestration — DONE

- `submit_dp10_pipeline.sh` orchestrator: full chain (perturb → combine → analyze) plus `--analyze-only` and `--snapshots-only` short-circuits. 7 independent sbatch analyze jobs (no srun-step contention).
- `run_ising_perturbations.py --modes` filter: restricts generation/aggregation to a comma-separated subset.
- `run_ising_pipeline.sh perturbations --modes` passthrough.
- `run_ising_pipeline.sh snapshots --stim-mode dp10_full` shortcut (and the legacy `all` enumeration was fixed to use the canonical `0p25..8p00` labels and all 6 bias values per family).
- `generate_perturbation_snapshots.py` `SNAPSHOT_STIM_MODES`: extended programmatically to `5 fully-clamped + 5 bias families × 6 bias values = 35 modes`. `REPS_PER_SIZE = {2: 20, 3: 20, 4: 20}` (previously only sizes 3/4).
- `Figure5_IsingPerturbationAnalysis.m`: accepts bias-encoded `config.stimMode` (e.g. `'double_pulse_bias10_2p00'`); per-invocation slicing produces 4-D data downstream without high/low collapse. Per-bias output paths auto-segregated.
- `match_experimental_to_isingBias.m`: written, parses, ready to run.

### Generation & aggregation — DONE on cluster (20 reps)

Per-mode aggregates exist for every dp10-relevant mode:

```
PerturbationResults_double_pulse10_20260429_145147.mat        (4.4 GB)
PerturbationResults_double_pulse_bias10_20260501_144027.mat   (9.2 GB)
```

Plus aggregates for `double_pulse{,3,5}` and `double_pulse_bias{,3,5}` (unused for Figure 6 but present). All built from per-combo files at **20 replicates each** — not the 150-rep target.

The most recent dp10-only perturb array (`58489437`, 2,016 tasks) ran today and exited 0:0, but every task `SKIP`-ed because the per-combo files were already on disk. So actual data is unchanged from the earlier April/May runs.

### Analyze — DONE today on cluster

Submitted via `bash submit_dp10_pipeline.sh --analyze-only`, which spawned **7 independent sbatch analyze jobs**:

| JobID | Mode | Elapsed | State |
|---|---|---|---|
| 58491812 | `double_pulse10` | 35:37 | COMPLETED 0:0 |
| 58491813 | `double_pulse_bias10_0p25` | 42:34 | COMPLETED 0:0 |
| 58491814 | `double_pulse_bias10_0p50` | 42:34 | COMPLETED 0:0 |
| 58491815 | `double_pulse_bias10_1p00` | 43:14 | COMPLETED 0:0 |
| 58491816 | `double_pulse_bias10_2p00` | 43:34 | COMPLETED 0:0 |
| 58491817 | `double_pulse_bias10_4p00` | 43:04 | COMPLETED 0:0 |
| 58491818 | `double_pulse_bias10_8p00` | 43:42 | COMPLETED 0:0 |

Wall-clock: ~45 min total (true parallelism). Figure counts on cluster:

```
double_pulse10/Analysis/                  : 12,102 PNGs   (cumulative across runs)
double_pulse_bias10_0p25/Analysis/        :  1,689 PNGs
double_pulse_bias10_0p50/Analysis/        :  1,689 PNGs
double_pulse_bias10_1p00/Analysis/        :  1,689 PNGs
double_pulse_bias10_2p00/Analysis/        :  1,689 PNGs
double_pulse_bias10_4p00/Analysis/        :  1,689 PNGs
double_pulse_bias10_8p00/Analysis/        :  1,689 PNGs
```

### Snapshots & heatmaps — first run DEGRADED (timeout); now patched

The initial `bash submit_dp10_pipeline.sh --snapshots-only` finished with substantial losses:

```
COMPLETED  4,341
FAILED     4,378
TIMEOUT      600
```

Root cause: snapshot worker `--time` was hardcoded to `00:20:00`. Cold-cache Numba JIT (~275 s on first task per node) plus matplotlib 300-DPI rendering blew past the cap on many tasks. The 600 explicit TIMEOUTs cascaded into the bulk FAILED count via `DependencyNeverSatisfied` on `snap_combine` / `snap_figures` / `heatmap_combine` / `heatmap_figures`.

**Patched**:

- `run_ising_pipeline.sh` (snapshots mode): default snapshot worker time bumped from `00:20:00` → `01:00:00`. Added `--snapshot-worker-time` CLI flag for further override (e.g. for `--n-reps 200` or larger stim sizes).
- `submit_dp10_pipeline.sh`: added orchestrator passthrough — `--snapshot-worker-time T`. Default `01:00:00`.

To re-submit with the patch:

```bash
# IMPORTANT: invoke under a LOGIN shell (`bash -l`). On the cluster today, lmod's
# `module load` doesn't fully wire PATH inside non-login non-interactive bash —
# python falls back to /usr/bin/python and numba/numpy/scipy import fails.
# `bash -l` ensures /etc/profile + lmod init source correctly.
bash -l submit_dp10_pipeline.sh --snapshots-only

# Override worker time if needed (default: 01:00:00):
bash -l submit_dp10_pipeline.sh --snapshots-only --snapshot-worker-time 02:00:00
```

Module-version fallback: `python/3.11.14` (the current `python/3.11` symlink) has a broken modulefile (`attempt to concatenate a nil value 'codename'`) that loads silently fails. The patched scripts try `python/3.11` first then fall back through `3.11.13 → 3.11.12 → 3.11.11 → 3.11.10 → 3.11.9` until one passes the import check. Today `python/3.11.13` is the one that wins.

Existing `COMPLETED` snapshot/heatmap PNGs from the first run will be overwritten; the re-submit redoes everything.

### Stale job to clean up

- `58491521 dp10_analyze_moransI+activity`: the original in-allocation 7-way fan-out, still RUNNING after 2:00+. Hit srun-step contention; superseded by jobs 58491812-58491818. **Action**: `scancel 58491521`.

### Matcher — DONE on cluster, results retrieved

Submitted `submit_match_dp10_bias.sh` (job `58499022`), completed in 1:22 wall on `eta406`. The first run failed because `load()` OOM'd on the 9.2 GB v7.3 .mat — patched the matcher to fall back to lazy HDF5 reads (`h5read` per leaf, only the chosen `(size, dur)` slice, ~240 leaves × ~50 reps × 759 frames each).

Best-matching bias value per condition (RMSE on trial-mean fraction-active in `[-1, 3]s` window, `StimulusSize=4`, `stimulus_duration=59` MC sweeps ≈ 2 s):

| Condition | Best bias | RMSE | Trend across bias |
|---|---|---|---|
| **Naive**    | **0.25** | 0.0180 | RMSE rises monotonically with bias (0.0180 → 0.0235) |
| **Beginner** | **8.00** | 0.0138 | RMSE falls monotonically with bias (0.0173 → 0.0138) |
| **Expert**   | **4.00** | 0.0097 | U-shaped; minimum in the middle |

Outputs landed at `Paper\Fig. 5 Model\PerturbationAnalysis\BiasMatchExperiment\`:
- `BiasMatch_<cond>_size4.png` — overlay per condition (experimental black, 6 sim traces + winner highlighted).
- `BiasMatch_Summary.png` — bar chart of best bias per condition.
- `BiasMatch_Summary.csv` — full RMSE matrix (3 conditions × 6 bias values).
- `matcher_output.mat` — `out` struct with full traces + scores for further plotting.

The 9.2 GB `PerturbationResults_double_pulse_bias10_20260501_144027.mat` aggregate is now also at `IsingModelData_39x78_100K\IsingPerturbations\` (rsync completed) — local re-runs will work without the cluster round-trip going forward.

### Pending

- Re-run snapshot/heatmap pipeline with longer worker timeout (see Snapshots section above).
- Decide whether 20-replicate aggregates are enough for the paper. To regenerate at 150 reps: delete the relevant per-combo files and run `bash submit_dp10_pipeline.sh --force` (~3–8 h wall).
- (Optional) Inspect the per-condition overlay PNGs and confirm the winner highlighting matches the visual best-fit.

---

## Troubleshooting

### `module: not found` in a SLURM job

`sbatch --wrap="..."` runs the body under `/bin/sh`, where `source` isn't a builtin and `module` isn't loaded. Fix: don't use `--wrap`; submit `run_ising_pipeline.sh perturbations` instead, which has `#!/bin/bash` and proper `lmod` initialization at lines ~210-224. The patched `submit_dp10_pipeline.sh` already does this.

### Combine/analyze stuck on `DependencyNeverSatisfied`

The upstream array job (perturb or combine) had ≥1 task end in `FAILED`/`TIMEOUT`/`CANCELLED`. Diagnose:

```bash
sacct -X -u $USER --format=JobID%25,JobName%30,State,ExitCode | grep dp10 | tail -30
```

Find the failed task IDs, then read its error log:

```bash
ls -t ~/IsingPerturbations/moransI+activity/logs/dp10_perturb_*.err | head -1 | xargs tail -60
```

If the failure is transient/nodes-related, `run_ising_pipeline.sh retry` resubmits failed indices.

### `srun: step creation still disabled, retrying`

Multiple `srun` calls competing for steps inside the same allocation. The original analyze fan-out submitted all 7 modes as `&` subprocesses inside one SLURM job, which serialised them. Fix already applied: `submit_dp10_pipeline.sh` now submits 7 *independent* sbatch jobs.

### Matcher complains about "No PerturbationResults file matching..."

The aggregated `.mat` for the requested mode isn't on local  yet. Either rsync it from cluster, or run the matcher with `'PerturbationResultsPath', '<some other dir>'` if it lives elsewhere.

### NumPy 2.x / scipy version mismatch on login node

System Python on `the cluster` has incompatible numpy/scipy. The orchestrator now `module load python/3.11` before any host-side python call — confirms `import numba, numpy, scipy` succeeds before submitting.

---

## Reference: stim duration sweep

`compute_stimulus_durations(global_mean_sf)` in `run_ising_perturbations.py:142-160` converts target durations in seconds to MC-sweep counts based on the matched `globalMeanSF`. Typical output for the dp10 pipeline: 9 durations covering ~0.5 s to ~10 s of equivalent real time. The MATLAB analysis writes one `dur_<N>/` subfolder per duration.
