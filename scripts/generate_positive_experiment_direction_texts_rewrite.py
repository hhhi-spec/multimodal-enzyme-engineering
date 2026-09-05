import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "direction_text_csvs_en_v3" / "positive_experiment_csvs_reduced"
OUTPUT_DIR = ROOT / "direction_text_csvs_en_v3" / "positive_experiment_csvs_reduced_rewritten"

TEXT_COLUMNS = [
    "direction_text",
    "direction_text_para_1",
    "direction_text_para_2",
    "direction_text_para_3",
]

SKIP_EXACT = {
    "experiment_id",
    "substrate",
    "product",
    "parent_sequence",
    "ec_class",
    "id",
    "amino_acid_substitutions",
    "aa_sequence",
    "smiles_string",
    "reaction_smiles",
    "record_role",
    "is_positive",
    "comparison_metric",
    "comparison_value",
    "improved_metrics",
    "fitness_type",
}

METRIC_KEYWORDS = (
    "fitness_value",
    "fitness",
    "ttn",
    "activity_for_reaction",
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
    "tm",
    "temperature",
    "stability",
)

REACTION_SMILES_HINTS = (
    "reaction_smiles",
    "smiles_reaction",
    "reaction smiles",
    "reactionsmiles",
)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def is_trace_value(value: str) -> bool:
    v = normalize(value)
    return bool(v) and ("trace" in v or "below loq" in v or "loq" in v)


def is_missing(value: str) -> bool:
    v = normalize(value)
    return v in {"", "?", "na", "n/a", "none", "nan"}


def parse_float(value: str):
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def row_kind(row):
    record_role = normalize(row.get("record_role"))
    if record_role == "parent":
        return "parent"
    aas = normalize(row.get("amino_acid_substitutions"))
    if aas in {"", "#parent#", "parent"}:
        return "parent"
    return "variant"


def metric_label(field: str) -> str:
    f = field.lower()
    if "fitness" in f:
        return "fitness value"
    if "activity_for_reaction" in f or f == "activity":
        return "reaction activity"
    if "ttn" in f:
        return "TTN"
    if "selectivity" in f or f == "ee":
        return "selectivity"
    if "yield" in f:
        return "yield"
    if "conversion" in f:
        return "conversion"
    if "ton" in f:
        return "TON"
    if "tof" in f:
        return "TOF"
    if "kcat" in f or "km" in f or "efficiency" in f:
        return "catalytic efficiency"
    if "tm" in f or "thermal" in f or "stability" in f:
        return "thermal stability"
    return ""


def clean_value(label: str, value: str) -> str:
    text = (value or "").strip()
    if is_trace_value(text):
        return "trace (below LOQ)"
    if label == "thermal stability":
        m = re.search(r"([-+]?\d*\.?\d+)\s*°?C", text, re.I)
        if m:
            return f"{m.group(1)} °C"
        m = re.search(r"([-+]?\d*\.?\d+)", text)
        return f"{m.group(1)} °C" if m else text
    if label == "selectivity":
        m = re.search(r"([-+]?\d*\.?\d+)\s*%", text)
        if m:
            return f"{m.group(1)}%"
    return text


def metric_columns(header):
    cols = []
    for col in header:
        lc = col.lower().strip()
        if lc in SKIP_EXACT:
            continue
        if lc.startswith("direction_text"):
            continue
        if lc.startswith("smiles_string"):
            continue
        if lc == "reaction_smiles":
            continue
        if any(key in lc for key in METRIC_KEYWORDS):
            cols.append(col)
    return cols


def reaction_smiles_column(header):
    for col in header:
        lc = col.lower().strip()
        if any(h == lc or h in lc for h in REACTION_SMILES_HINTS):
            return col
    return ""


def extract_substrate_smiles(row, reaction_col):
    if not reaction_col:
        return ""
    reaction = (row.get(reaction_col) or "").strip()
    if ">>" not in reaction:
        return ""
    left = reaction.split(">>", 1)[0].strip()
    return left


def build_clause(field: str, parent_value: str, variant_value: str):
    label = metric_label(field)
    if not label:
        return None

    parent_raw = (parent_value or "").strip()
    variant_raw = (variant_value or "").strip()

    if is_missing(parent_raw) or is_missing(variant_raw):
        return None
    if is_trace_value(variant_raw):
        return None

    parent_trace = is_trace_value(parent_raw)
    parent_num = parse_float(parent_raw)
    variant_num = parse_float(variant_raw)

    if parent_trace:
        return f"{label} highly increased from trace (below LOQ) to {clean_value(label, variant_raw)}"

    if parent_num is not None and variant_num is not None:
        tol = max(1e-9, 1e-6 * max(abs(parent_num), abs(variant_num), 1.0))
        if variant_num > parent_num + tol:
            if label == "thermal stability":
                return f"{label} increased from {clean_value(label, parent_raw)} to {clean_value(label, variant_raw)}"
            return f"{label} improved from {clean_value(label, parent_raw)} to {clean_value(label, variant_raw)}"
        return None

    if parent_raw and variant_raw:
        return f"{label} improved from {clean_value(label, parent_raw)} to {clean_value(label, variant_raw)}"

    if variant_raw:
        return f"{label} improved to {clean_value(label, variant_raw)}"
    return None


def build_target(labels):
    ordered = []
    seen = set()
    for label in labels:
        if label and label not in seen:
            seen.add(label)
            ordered.append(label)
    if not ordered:
        return "catalytic performance"
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) == 2:
        return " and ".join(ordered)
    return ", ".join(ordered[:-1]) + ", and " + ordered[-1]


def build_texts(labels, clauses):
    target = build_target(labels)
    detail = "; ".join(clauses)
    main = f"Aim to improve {target}; {detail}."
    para_1 = f"Improve {target}; {detail}."
    para_2 = f"{detail}; aim to improve {target}."
    para_3 = f"The variant shows {detail}, indicating a need to improve {target}."
    return main, para_1, para_2, para_3


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for src_path in sorted(INPUT_DIR.glob("*.csv")):
        header, rows = read_csv(src_path)
        if not header or not rows:
            continue

        parent_row = next((row for row in rows if row_kind(row) == "parent"), None)
        metric_cols = metric_columns(header)
        reaction_col = reaction_smiles_column(header)

        out_header = list(header)
        for extra in TEXT_COLUMNS:
            if extra not in out_header:
                out_header.append(extra)

        out_path = OUTPUT_DIR / src_path.name
        with out_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_header, extrasaction="ignore")
            writer.writeheader()

            for row in rows:
                if row_kind(row) == "parent":
                    for col in TEXT_COLUMNS:
                        row[col] = ""
                    if reaction_col:
                        row["substrate"] = extract_substrate_smiles(row, reaction_col) or row.get("substrate", "")
                    writer.writerow(row)
                    continue

                if reaction_col:
                    row["substrate"] = extract_substrate_smiles(row, reaction_col) or row.get("substrate", "")

                clauses = []
                labels = []
                for field in metric_cols:
                    clause = build_clause(
                        field,
                        parent_row.get(field, "") if parent_row else "",
                        row.get(field, ""),
                    )
                    if clause:
                        clauses.append(clause)
                        label = metric_label(field)
                        if label and label not in labels:
                            labels.append(label)

                if not clauses:
                    metrics = (row.get("comparison_metric") or "").replace(";", ", ").strip()
                    detail = "reported improvement" if not metrics else f"reported improvement in {metrics}"
                    target = metrics or "catalytic performance"
                    row["direction_text"] = f"Aim to improve {target}; {detail}."
                    row["direction_text_para_1"] = f"Improve {target}; {detail}."
                    row["direction_text_para_2"] = f"{detail}; aim to improve {target}."
                    row["direction_text_para_3"] = f"The variant shows {detail}, indicating a need to improve {target}."
                else:
                    main, p1, p2, p3 = build_texts(labels, clauses)
                    row["direction_text"] = main
                    row["direction_text_para_1"] = p1
                    row["direction_text_para_2"] = p2
                    row["direction_text_para_3"] = p3

                writer.writerow(row)


if __name__ == "__main__":
    main()
