"""Create an inventory of generated data files.

This script is intentionally independent of the notebooks. It records file
paths, sizes, extensions, and SHA-256 checksums for the generated data included
under data/generated/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import DATA_ROOT, REPO_ROOT, ensure_summary_root, sha256_file, write_csv, write_json


DATA_EXTENSIONS = {".csv", ".json", ".npy", ".npz"}


def build_manifest(data_root: Path = DATA_ROOT) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in DATA_EXTENSIONS:
            continue
        stat = path.stat()
        rows.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "category": path.parent.name,
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a manifest for data/generated/.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ensure_summary_root(),
        help="Directory for generated manifest files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest()
    output_dir = args.output_dir
    write_json(output_dir / "generated_data_manifest.json", manifest)
    write_csv(
        output_dir / "generated_data_manifest.csv",
        manifest,
        ["path", "category", "extension", "size_bytes", "sha256"],
    )
    print(f"Wrote {len(manifest)} manifest rows to {output_dir}")


if __name__ == "__main__":
    main()

