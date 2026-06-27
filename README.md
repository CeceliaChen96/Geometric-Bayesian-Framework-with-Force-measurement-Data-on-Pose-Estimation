# Wrench-Based Bayesian Pose Estimation via Matrix-Fisher Gaussian Inference

This repository contains the reproducibility code and generated data for the
project:

> Wrench-Based Bayesian Pose Estimation via Matrix--Fisher Gaussian Inference

The code implements Matrix-Fisher Gaussian (MFG) posterior approximation and
residual-safeguarded refinement for wrench-based pose estimation on
`SO(3) x R^3`. The GitHub-facing entry points are Python scripts; the notebooks
are kept as experiment provenance and development records.

## Repository Scope

Included:

- Python package code in `src/mfg_pose_estimation/`.
- Lightweight runnable demos in `scripts/`.
- Python reproducibility entry points in `experiments/`.
- Development and paper-experiment notebooks in `notebooks/` for provenance.
- Generated synthetic-experiment data in `data/generated/`.
- Generated figures and notebook outputs kept in their original locations.
- Placeholders and links for physical robot experiment data and videos.

Not included:

- The manuscript LaTeX source and submission files. Those remain outside the
  public reproducibility package.
- Raw physical robot experiment code.

## Setup

Create a Python environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The code was organized for Python 3.10 or newer.

## Quick Checks

Create a manifest of the generated data:

```bash
python experiments/make_data_manifest.py
```

Create a compact paper-result summary from saved generated records:

```bash
python experiments/summarize_paper_results.py
```

Run a lightweight Algorithm 1 demo:

```bash
python scripts/run_alg1.py --skip-jacobian-check --noise-free
```

Run a lightweight safeguarded Algorithm 2 demo:

```bash
python scripts/run_alg2.py --Tmax 1 --noise-free
```

Run the Matrix-Fisher sampling visualization demo:

```bash
python scripts/run_sampling_demo.py
```

These commands write outputs under `results/demo_runs/`.

You can also run both lightweight package demos through the Python-first
experiment wrapper:

```bash
python experiments/run_smoke_demos.py
```

## Main Directories

```text
src/mfg_pose_estimation/
  Core geometry, MFG distribution utilities, wrench model, Jacobians,
  posterior updates, refinement algorithms, sampling, and plotting helpers.

scripts/
  Small package-based demos for Algorithm 1, Algorithm 2, and sampling.

experiments/
  Python-first reproducibility entry points for generated-data manifests,
  paper-result summaries, and smoke demos.

notebooks/Posterior Estimation/
  Main synthetic experiment notebooks and generated outputs for the paper.

notebooks/baseline_experiments/
  Baseline comparison scripts, generated records, summary tables, and figures.

notebooks/diagnostics/
  Diagnostic scripts and saved diagnostic records.

notebooks/sampling/
  Matrix-Fisher sampling and visualization notebook.

data/generated/
  Copied generated data records used by the paper experiments, organized for
  GitHub readers without changing the original notebook output paths.

data/real_experiments/
  Placeholder for physical robot experiment data. This directory is currently
  intentionally empty except for its README.

results/paper_figures/
  Generated paper figures from the package-level experiments.
```

## Generated Data

Generated data are available in two forms:

1. Original notebook/script output locations, which preserve the paths used
   during development.
2. A consolidated copy under `data/generated/`, with subfolders for posterior
   estimation, baseline experiments, and diagnostics.

The consolidated data folders contain JSON, CSV, NPZ, and NPY files only. Large
figures and plots are kept in their original `notebooks/` or `results/`
locations.

Python-generated summaries are written to `results/python_summaries/`.

## Physical Robot Experiment Videos

Two physical experiment videos are available externally:

- Probe experiment: <https://www.dropbox.com/scl/fi/gtrxmd8juf8imjyvv3u8k/probe.mp4?rlkey=90z5i6tph3a2qb54fm9tn9u8a&dl=0>
- 22-observation experiment: <https://www.dropbox.com/scl/fi/xom30hqhicx03cz5dz5yv/22obs.mp4?rlkey=cgk04zerw2qg3xo8e0ptwkcys&dl=0>

The corresponding numeric physical experiment data is added under
`data/real_experiments/`.

## Python Experiment Entry Points

For GitHub readers, start with the Python scripts in `experiments/`:

- `experiments/make_data_manifest.py`
- `experiments/summarize_paper_results.py`
- `experiments/run_smoke_demos.py`

These scripts read the generated records, create reproducibility summaries, or
run lightweight package demos without modifying the notebooks.

## Paper Notebooks

The most relevant notebooks for reproducing the synthetic paper experiments are:

- `notebooks/Posterior Estimation/New_3Dcase_Ex2A.ipynb`
- `notebooks/Posterior Estimation/New_3Dcase_Ex2B.ipynb`
- `notebooks/Posterior Estimation/New_3Dcase_Ex2C.ipynb`
- `notebooks/Posterior Estimation/New_3Dcase_Ex2C_b.ipynb`
- `notebooks/Posterior Estimation/Revised2_Multiseed_forEx2A.ipynb`
- `notebooks/Posterior Estimation/Revised3_Multiseed_forEx2B.ipynb`
- `notebooks/Posterior Estimation/Revised4_Multiseed_forEx2C.ipynb`
- `notebooks/Posterior Estimation/Revised5_Multiseed_forEx2C_b_main_plot.ipynb`

The baseline comparison scripts are in:

- `notebooks/baseline_experiments/code/run_baseline_experiments.py`
- `notebooks/baseline_experiments/code/run_protocol_accuracy_baselines.py`

## Notes for IEEE/TAC Submission

IEEE encourages authors to share code and data to improve reproducibility.
The IEEE Control Systems Society author page for IEEE Transactions on Automatic
Control also points authors to Code Ocean and IEEE DataPort. A GitHub link is
therefore appropriate as a reproducibility resource in the manuscript.


## Citation

If you use this code before the paper has a DOI, cite the repository metadata in
`CITATION.cff`. After publication, replace it with the final article citation.

## License

No open-source license has been selected yet. Until a license is added, the
repository is publicly readable but reuse rights are not explicitly granted.
Before final public release, choose an appropriate license such as MIT, BSD, or
Apache-2.0 if the authors intend to allow reuse.
