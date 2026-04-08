# MFG Pose Estimation from Force Measurements


## Current contents
- two main notebook projects:
  - `mfg_posterior_project`
  - `Sampling_project`
- generated paper figures (`.pdf`)


## Repository structure

```text
mfg-pose-estimation/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/
│   └── mfg_pose_estimation/
│       ├── __init__.py
│       ├── geometry.py
│       ├── distributions.py
│       ├── sdf.py
│       ├── wrench_model.py
│       ├── jacobians.py
│       ├── algorithms.py
│       ├── design.py
│       └── plotting.py
├── scripts/
│   ├── run_alg1.py
│   ├── run_alg2.py
│   └── run_sampling_demo.py
├── notebooks/
│   ├── posterior/
│   └── sampling/
├── results/
│   └── paper_figures/
└── docs/
    └── notes.md
```

## What stay in notebooks
Keep notebooks for:
- exploratory experiments
- ablation studies
- figure generation
- paper-ready visualizations
- one-off diagnostics


## Minimal setup

```bash
python -m venv .venv
source .venv/bin/activate   # on macOS / Linux
pip install -r requirements.txt
```

## Suggested workflow
1. Keep the current notebooks runnable.
3. Replace notebook-local definitions with imports from `src/`.
4. Add small scripts in `scripts/` for reproducible runs.
5. Clean outputs and publish.

