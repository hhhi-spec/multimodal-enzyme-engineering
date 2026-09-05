from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .data import build_vocab_bundle, save_vocab_bundle
from .manifest import build_manifest
from .utils import load_jsonl


def validate_manifest(rows: list[dict]) -> None:
    required = [
        "parent_sequence",
        "parent_3d_path",
        "substrate_smiles",
        "variant_aa_sequence",
        "amino_acid_substitutions",
        "mutation_sites",
        "mutation_targets",
        "direction_text",
    ]
    missing = {key: 0 for key in required}
    for row in rows:
        for key in required:
            value = row.get(key)
            if value is None or value == "" or value == []:
                missing[key] += 1
    print("required-field-missing-counts")
    for key, value in missing.items():
        print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Build multimodal baseline artifacts.")
    parser.add_argument("--output-dir", type=Path, default=Path("multimodal_baseline_artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    paths = build_manifest(args.output_dir, seed=args.seed, test_ratio=args.test_ratio)
    manifest_rows = load_jsonl(paths["manifest"])
    validate_manifest(manifest_rows)

    train_rows = [row for row in manifest_rows if row.get("split") == "train"]
    vocab_bundle = build_vocab_bundle(train_rows)
    vocab_path = args.output_dir / "vocab.json"
    save_vocab_bundle(vocab_path, vocab_bundle)

    summary_path = args.output_dir / "manifest_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "count",
                "experiment_count",
                "ec_major",
            ],
        )
        writer.writeheader()
        for split in ("train", "test"):
            rows = [row for row in manifest_rows if row.get("split") == split]
            exp_ids = sorted({row.get("experiment_id") for row in rows})
            ec_ids = sorted({int(row.get("ec_major") or 0) for row in rows})
            for ec_major in ec_ids:
                ec_rows = [row for row in rows if int(row.get("ec_major") or 0) == ec_major]
                writer.writerow(
                    {
                        "split": split,
                        "count": len(ec_rows),
                        "experiment_count": len({row.get("experiment_id") for row in ec_rows}),
                        "ec_major": ec_major,
                    }
                )

    print(f"manifest={paths['manifest']}")
    print(f"splits={paths['splits']}")
    print(f"vocab={vocab_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()

