# Baseline Experiments

This directory is intentionally separate from `TAC_template/root.tex` and
`TAC_template/Supplementary_Material.tex`.

It implements the B0--B5 baseline plan:

- B0: initial / prior-mode pose.
- B1: Lie-Gauss--Newton / Levenberg--Marquardt MAP baseline.
- B2: tangent Gaussian / Laplace update.
- B3: decoupled rotation/translation local Gaussian baseline.
- B4: measurement-selection baselines.
- B5: local sampling reference around the MAP estimate.

Outputs are written under:

- `results/`: JSON/CSV records.

Generated plots and standalone TeX/PDF reports are intentionally not kept in
the GitHub-facing repository. They can be regenerated locally from the scripts
when needed.

The lightweight smoke-test script imports the existing local `posterior_core.py`
directly; the protocol script loads the saved Experiment 2C notebook
definitions.  Neither script touches the manuscript files.

The accuracy-centered protocol baseline can be regenerated with the two fairness
schemes requested for revision:

```bash
MPLCONFIGDIR=/private/tmp python baseline_experiments/code/run_protocol_accuracy_baselines.py \
  --seeds 42 43 44 45 46 \
  --max-iter 1 \
  --T-max 20
```

Scheme A compares the paper's Algorithm 1 with B1--B3 at the single-pass /
one-local-update level.  Scheme B gives B1--B3 up to 20 residual-safeguarded
outer re-centering steps and compares them with the saved 20-step Proposed
Algorithm 2 records.  All rows use the saved Experiment 2C false-confidence
protocol and the same initial pose.  Proposed Algorithm 2 accuracy and runtime
come from the saved manuscript records; Algorithm 1 and B1--B3 are recomputed
from the same protocol definitions.

The A/B protocol outputs are:

- `results/scheme_ab_accuracy_records.{json,csv}`: per-seed records.
- `results/scheme_ab_accuracy_summary.{json,csv}`: mean/std summary tables.
- `results/scheme_ab_accuracy_run_config.json`: command and source metadata.

The earlier lightweight B0--B5 smoke-test driver remains available as:

```bash
MPLCONFIGDIR=/private/tmp python baseline_experiments/code/run_baseline_experiments.py \
  --n-seeds 2 \
  --K 10 \
  --random-subsets 1 \
  --sampling-draws 20 \
  --selection-seeds 1 \
  --liegn-max-iter 5 \
  --proposed-tmax 2 \
  --skip-proposed-alg2
```

That lightweight driver is useful for quick code checks, but it should not be
used as the reviewer-facing accuracy comparison.
