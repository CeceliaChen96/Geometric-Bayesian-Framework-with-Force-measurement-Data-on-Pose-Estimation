# Python Reproducibility Entry Points

This directory provides the Python-first entry points for the public GitHub
repository. The original notebooks remain under `notebooks/` as provenance and
development records, but readers can use these scripts without opening a
notebook.

Run all commands from the repository root.

## Generated-Data Manifest

Create a machine-readable inventory of the generated data files:

```bash
python experiments/make_data_manifest.py
```

Outputs:

- `results/python_summaries/generated_data_manifest.csv`
- `results/python_summaries/generated_data_manifest.json`

## Paper-Result Summary

Create a compact Markdown/JSON summary from the saved generated records:

```bash
python experiments/summarize_paper_results.py
```

Outputs:

- `results/python_summaries/paper_results_summary.md`
- `results/python_summaries/paper_results_summary.json`

## Package Smoke Demos

Run the lightweight package-based Algorithm 1 and Algorithm 2 demos:

```bash
python experiments/run_smoke_demos.py
```

This script calls the package demos in `scripts/` and writes demo outputs under
`results/demo_runs/`. Those demo outputs are intentionally ignored by Git.

## Notes

- These scripts read generated records from `data/generated/`.


