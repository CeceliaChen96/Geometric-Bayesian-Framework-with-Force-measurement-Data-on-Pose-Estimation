# Data Directory

This directory provides a GitHub-facing organization of the data records used by
the reproducibility package.

## `generated/`

Copied generated records from the synthetic and diagnostic experiments. These
files duplicate selected JSON, CSV, NPZ, and NPY outputs from the notebook and
script directories so readers can find the data without searching through
notebook outputs.

Subfolders:

- `generated/posterior_estimation/`: synthetic posterior-estimation records.
- `generated/baseline_experiments/`: baseline comparison records and summaries.
- `generated/diagnostics/`: chart-drift diagnostic records and summaries.

The original files remain in their development locations under `notebooks/`.

## `real_experiments/`

Reserved for physical robot experiment data. The numeric data are not included
yet and should be added when finalized.

