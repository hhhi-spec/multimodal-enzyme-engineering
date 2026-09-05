from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .ligand3d import build_ligand_3d
from .mmcif import parse_mmcif_backbone, project_coords_to_parent
from .utils import (
    ROOT,
    apply_substitutions_to_sequence,
    choose_primary_text,
    detect_metric_labels,
    derive_mutations_from_sequences,
    first_nonempty,
    infer_ec_major,
    is_positive_row,
    is_missing,
    normalize_text,
    parse_direction_texts,
    parse_substitutions,
    read_csv_rows,
    split_components,
    write_jsonl,
)


POSITIVE_DIR = ROOT / "direction_text_csvs_en_v3" / "positive_experiment_csvs_reduced_rewritten"
EC3_DIRS = [
    ROOT / "direction_text_csvs_en_v3" / "ec3-1_reduced",
    ROOT / "direction_text_csvs_en_v3" / "ec3-2_reduced",
    ROOT / "direction_text_csvs_en_v3" / "ec3-3_reduced",
    ROOT / "direction_text_csvs_en_v3" / "ec3-4_reduced",
    ROOT / "direction_text_csvs_en_v3" / "ec3-6_reduced",
]
RAW_DIR = ROOT / "experiments"
EC3_STRUCTURE_DIR = ROOT / "downloaded_structures" / "ec3"


def find_csv_for_dir(folder: Path) -> Path | None:
    csvs = sorted(folder.glob("*.csv"))
    if csvs:
        return csvs[0]
    return None


def find_structure_for_positive(experiment_id: str) -> Path | None:
    exp_dir = RAW_DIR / experiment_id
    if not exp_dir.exists():
        return None
    candidates = sorted(exp_dir.glob("*.cif"))
    return candidates[0] if candidates else None


def find_structure_for_ec3(experiment_id: str) -> Path | None:
    candidates = sorted(EC3_STRUCTURE_DIR.glob(f"{experiment_id}*.cif"))
    return candidates[0] if candidates else None


def load_parent_row(rows: List[dict]) -> dict | None:
    for row in rows:
        if (row.get("record_role") or "").strip().lower() == "parent":
            return row
    for row in rows:
        if (row.get("variant") or "").strip().upper() == "WT":
            return row
    return None


def extract_variant_rows(rows: List[dict]) -> List[dict]:
    filtered = []
    for row in rows:
        if not is_positive_row(row):
            continue
        substitutions = normalize_text(row.get("amino_acid_substitutions"))
        if substitutions in {"", "#parent#", "parent", "#low#", "low", "na", "n/a", "none", "nan"}:
            continue
        if "del" in substitutions or "ins" in substitutions or "indel" in substitutions:
            continue
        filtered.append(row)
    return filtered


def build_sample_record(
    *,
    dataset_group: str,
    experiment_id: str,
    source_csv: Path,
    parent_row: dict,
    variant_row: dict,
    structure_path: Path | None,
    structure_data: tuple[str, object, object] | None,
    ec_major: int | None,
    ligand_3d: dict | None,
) -> dict:
    parent_seq = first_nonempty(parent_row.get("parent_sequence"), variant_row.get("parent_sequence"))
    variant_seq = first_nonempty(
        variant_row.get("aa_sequence"),
        variant_row.get("variant_aa_sequence"),
        variant_row.get("sequence"),
    )
    substrate = first_nonempty(variant_row.get("substrate"), parent_row.get("substrate"))
    direction_texts = parse_direction_texts(variant_row)
    if not direction_texts:
        direction_texts = parse_direction_texts(parent_row)
    substitutions = first_nonempty(variant_row.get("amino_acid_substitutions"))
    sites, targets = parse_substitutions(substitutions)
    if not variant_seq and parent_seq and substitutions:
        variant_seq = apply_substitutions_to_sequence(parent_seq, substitutions)
    if (not sites or not targets) and parent_seq and variant_seq:
        derived_sites, derived_targets = derive_mutations_from_sequences(parent_seq, variant_seq)
        if derived_sites:
            sites = sites or derived_sites
            targets = targets or derived_targets
    if not sites or not targets:
        return None
    if not direction_texts:
        if substitutions:
            direction_texts = [f"Aim to improve catalytic performance; variant {substitutions}."]
        else:
            direction_texts = ["Aim to improve catalytic performance."]
    structure_seq = ""
    coords = []
    mask = []
    if structure_data is not None:
        structure_seq, raw_coords, _ = structure_data
        coords, mask = project_coords_to_parent(parent_seq, structure_seq, list(raw_coords))
    elif structure_path and structure_path.exists():
        structure_seq, raw_coords, raw_mask = parse_mmcif_backbone(structure_path)
        coords, mask = project_coords_to_parent(parent_seq, structure_seq, list(raw_coords))

    record = {
        "sample_id": first_nonempty(variant_row.get("id"), variant_row.get("variant"), f"{experiment_id}_row"),
        "dataset_group": dataset_group,
        "experiment_id": experiment_id,
        "ec_major": ec_major,
        "source_csv": str(source_csv.relative_to(ROOT)),
        "parent_row_id": first_nonempty(parent_row.get("id"), parent_row.get("variant"), "parent"),
        "variant_row_id": first_nonempty(variant_row.get("id"), variant_row.get("variant"), "variant"),
        "parent_sequence": parent_seq,
        "parent_3d_path": str(structure_path.relative_to(ROOT)) if structure_path else "",
        "parent_3d_seq": structure_seq,
        "parent_3d_coords": coords.tolist() if hasattr(coords, "tolist") else coords,
        "parent_3d_mask": mask.tolist() if hasattr(mask, "tolist") else mask,
        "substrate_smiles": substrate,
        "substrate_components": split_components(substrate),
        "substrate_atom_features": (ligand_3d or {}).get("substrate_atom_features", []),
        "substrate_atom_coords": (ligand_3d or {}).get("substrate_atom_coords", []),
        "substrate_atom_component_ids": (ligand_3d or {}).get("substrate_atom_component_ids", []),
        "substrate_component_sizes": (ligand_3d or {}).get("substrate_component_sizes", []),
        "substrate_3d_status": (ligand_3d or {}).get("substrate_3d_status", ""),
        "substrate_3d_error": (ligand_3d or {}).get("substrate_3d_error", ""),
        "variant_aa_sequence": variant_seq,
        "amino_acid_substitutions": substitutions,
        "mutation_sites": sites,
        "mutation_targets": targets,
        "direction_text": choose_primary_text(direction_texts),
        "direction_texts": direction_texts,
        "is_positive": 1,
    }
    return record


def build_manifest(
    output_dir: Path,
    *,
    seed: int = 42,
    test_ratio: float = 0.2,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: List[dict] = []
    split_rows: List[dict] = []
    split_by_exp: Dict[int, Dict[str, str]] = defaultdict(dict)
    split_summary: List[dict] = []

    sources = []
    if POSITIVE_DIR.exists():
        sources.append(("positive", POSITIVE_DIR))
    for ec_dir in EC3_DIRS:
        if ec_dir.exists():
            sources.append(("ec3", ec_dir))

    raw_records: List[dict] = []
    exp_to_class_and_count: Dict[Tuple[str, int], int] = defaultdict(int)
    exp_meta: Dict[Tuple[str, int], dict] = {}
    ligand_cache: Dict[str, dict] = {}

    for dataset_group, root_dir in sources:
        for csv_path in sorted(root_dir.glob("*.csv")):
            header, rows = read_csv_rows(csv_path)
            if not rows:
                continue
            parent_row = load_parent_row(rows)
            if parent_row is None:
                continue
            if dataset_group == "positive":
                experiment_id = csv_path.stem
                structure_path = find_structure_for_positive(experiment_id)
                ec_major = infer_ec_major(parent_row.get("ec_class"), fallback=None)
            else:
                experiment_id = csv_path.stem.replace("_filled", "")
                structure_path = find_structure_for_ec3(experiment_id)
                ec_major = 3
            variant_rows = extract_variant_rows(rows)
            if not variant_rows:
                continue
            structure_data = None
            if structure_path and structure_path.exists():
                structure_data = parse_mmcif_backbone(structure_path)
            for variant_row in variant_rows:
                substrate = first_nonempty(variant_row.get("substrate"), parent_row.get("substrate"))
                if substrate not in ligand_cache:
                    ligand_cache[substrate] = build_ligand_3d(substrate).to_dict()
                record = build_sample_record(
                    dataset_group=dataset_group,
                    experiment_id=experiment_id,
                    source_csv=csv_path,
                    parent_row=parent_row,
                    variant_row=variant_row,
                    structure_path=structure_path,
                    structure_data=structure_data,
                    ec_major=ec_major,
                    ligand_3d=ligand_cache.get(substrate, {}),
                )
                if record is None:
                    continue
                raw_records.append(record)
                exp_to_class_and_count[(experiment_id, ec_major or 0)] += 1
                exp_meta[(experiment_id, ec_major or 0)] = {
                    "dataset_group": dataset_group,
                    "experiment_id": experiment_id,
                    "ec_major": ec_major or 0,
                }

    # Split by experiment inside each EC class.
    rng = random.Random(seed)
    class_to_experiments: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for (experiment_id, ec_major), count in exp_to_class_and_count.items():
        class_to_experiments[ec_major].append((experiment_id, count))

    train_experiments = set()
    test_experiments = set()
    for ec_major, exp_list in sorted(class_to_experiments.items()):
        items = list(exp_list)
        rng.shuffle(items)
        total = sum(count for _, count in items)
        target_test = int(round(total * test_ratio))
        if total > 1:
            target_test = max(1, target_test)
        else:
            target_test = 0
        running = 0
        chosen_test: List[str] = []
        for experiment_id, count in items:
            if running < target_test:
                chosen_test.append(experiment_id)
                running += count
        if not chosen_test and items:
            chosen_test = [items[-1][0]]
        for experiment_id, _ in items:
            if experiment_id in chosen_test:
                test_experiments.add(experiment_id)
            else:
                train_experiments.add(experiment_id)

        split_summary.append(
            {
                "ec_major": ec_major,
                "experiment_count": len(items),
                "test_experiment_count": len(chosen_test),
                "train_experiment_count": len(items) - len(chosen_test),
                "sample_count": total,
                "target_test_sample_count": target_test,
            }
        )

    for record in raw_records:
        exp_id = record["experiment_id"]
        record["split"] = "test" if exp_id in test_experiments else "train"
        split_rows.append(record)

    manifest_path = output_dir / "manifest.jsonl"
    split_path = output_dir / "experiment_splits.json"
    summary_path = output_dir / "split_summary.csv"

    write_jsonl(manifest_path, split_rows)
    split_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "test_ratio": test_ratio,
                "train_experiments": sorted(train_experiments),
                "test_experiments": sorted(test_experiments),
                "by_ec": split_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ec_major",
                "experiment_count",
                "train_experiment_count",
                "test_experiment_count",
                "sample_count",
                "target_test_sample_count",
            ],
        )
        writer.writeheader()
        for row in split_summary:
            writer.writerow(row)

    return {
        "manifest": manifest_path,
        "splits": split_path,
        "split_summary": summary_path,
    }
