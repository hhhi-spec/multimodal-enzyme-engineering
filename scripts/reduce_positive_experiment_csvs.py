import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "positive_experiment_csvs"
OUTPUT_DIR = ROOT / "positive_experiment_csvs_reduced"
ADDEC_PATH = ROOT / "addec.csv"

BASE_KEEP = {
    "id",
    "amino_acid_substitutions",
    "aa_sequence",
    "smiles_string",
    "reaction_smiles",
    "record_role",
    "is_positive",
    "comparison_metric",
    "comparison_value",
}

METRIC_KEYWORDS = (
    "fitness",
    "ttn",
    "activity",
    "yield",
    "conversion",
    "selectivity",
    "ee",
    "kcat",
    "km",
    "turnover",
    "efficiency",
    "specificity",
    "ton",
    "tof",
)

ADDEC_FIELDS = ("substrate", "product", "parent_sequence", "ec_class")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def load_addec_lookup():
    header, rows = read_csv(ADDEC_PATH)
    if "experiment_id" not in header:
        raise ValueError("addec.csv is missing experiment_id")

    lookup = {}
    for row in rows:
        experiment_id = (row.get("experiment_id") or "").strip()
        if not experiment_id:
            continue
        lookup[experiment_id] = {field: (row.get(field) or "").strip() for field in ADDEC_FIELDS}
    return lookup


def is_metric_column(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in METRIC_KEYWORDS)


def choose_columns(header):
    columns = []
    for name in header:
        if name in BASE_KEEP and name not in columns:
            columns.append(name)
    for name in header:
        if name in BASE_KEEP:
            continue
        if is_metric_column(name) and name not in columns:
            columns.append(name)
    return columns


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    addec_lookup = load_addec_lookup()

    for src_path in sorted(INPUT_DIR.glob("*.csv")):
        header, rows = read_csv(src_path)
        if not header:
            continue

        experiment_id = src_path.stem
        meta = addec_lookup.get(experiment_id, {field: "" for field in ADDEC_FIELDS})
        selected_columns = choose_columns(header)

        out_header = ["experiment_id", *ADDEC_FIELDS, *selected_columns]
        seen = set()
        deduped_header = []
        for name in out_header:
            if name not in seen:
                seen.add(name)
                deduped_header.append(name)

        out_path = OUTPUT_DIR / src_path.name
        with out_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=deduped_header, extrasaction="ignore")
            writer.writeheader()

            for row in rows:
                out_row = {"experiment_id": experiment_id, **meta}
                for col in selected_columns:
                    out_row[col] = row.get(col, "")
                writer.writerow(out_row)


if __name__ == "__main__":
    main()
