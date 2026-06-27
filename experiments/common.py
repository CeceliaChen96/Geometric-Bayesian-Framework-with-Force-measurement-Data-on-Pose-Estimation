"""Shared helpers for Python-first reproducibility scripts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "generated"
SUMMARY_ROOT = REPO_ROOT / "results" / "python_summaries"


def ensure_summary_root() -> Path:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    return SUMMARY_ROOT


def load_json(relative_path: str | Path) -> Any:
    path = REPO_ROOT / relative_path
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def mean_std(row: dict[str, Any], mean_key: str, std_key: str, digits: int = 3) -> str:
    return f"{fmt(row.get(mean_key), digits)} +/- {fmt(row.get(std_key), digits)}"

