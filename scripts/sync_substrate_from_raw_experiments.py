import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "experiments"
TARGET_DIR = ROOT / "direction_text_csvs_en_v3" / "positive_experiment_csvs_reduced_rewritten"

REACTION_HINTS = (
    "reaction_smiles",
    "smiles_reaction",
    "reaction smiles",
    "reactionsmiles",
)


def read_csv(path: Path):
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "gbk", "latin1"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                return reader.fieldnames or [], list(reader)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Could not decode {path}")


def find_reaction_column(header):
    for col in header:
        lc = col.lower().strip()
        if any(h == lc or h in lc for h in REACTION_HINTS):
            return col
    return ""


def raw_csv_by_experiment():
    mapping = {}
    for exp_dir in RAW_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        csvs = list(exp_dir.glob("*.csv"))
        if csvs:
            mapping[exp_dir.name] = csvs[0]
    return mapping


def main():
    raw_map = raw_csv_by_experiment()
    updated_files = 0
    updated_rows = 0

    for target_path in sorted(TARGET_DIR.glob("*.csv")):
        experiment_id = target_path.stem
        raw_path = raw_map.get(experiment_id)
        if not raw_path:
            continue

        target_header, target_rows = read_csv(target_path)
        raw_header, raw_rows = read_csv(raw_path)
        if not target_rows or not raw_rows:
            continue

        raw_reaction_col = find_reaction_column(raw_header)
        if not raw_reaction_col:
            continue

        raw_by_id = {row.get("id", "").strip(): row for row in raw_rows if row.get("id")}

        changed = False
        for row in target_rows:
            row_id = (row.get("id") or "").strip()
            raw_row = raw_by_id.get(row_id)
            if not raw_row:
                continue
            raw_rxn = (raw_row.get(raw_reaction_col) or "").strip()
            if ">>" not in raw_rxn:
                continue
            left = raw_rxn.split(">>", 1)[0].strip()
            if left and row.get("substrate") != left:
                row["substrate"] = left
                changed = True
                updated_rows += 1

        if changed:
            with target_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=target_header, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(target_rows)
            updated_files += 1

    print(f"updated_files={updated_files}")
    print(f"updated_rows={updated_rows}")


if __name__ == "__main__":
    main()
