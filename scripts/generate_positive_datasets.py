import csv
import io
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "experiments"
OUTPUT_DIR = ROOT / "positive_experiment_csvs"

ENCODINGS = ["utf-8-sig", "gb18030", "utf-16", "latin1"]
PARENT_MARKERS = {"#parent#", "parent"}
INVALID_AAS = {"", "#low#", "#n.a.#", "#n.a", "n.a.", "nan", "none", "na"}

RESULT_FIELD_CANDIDATES = [
    "yield",
    "conversion",
    "ttn",
    "TTN (if applicable)",
    "ton",
    "tof",
    "ee",
    "selectivity",
    "activity_for_reaction_% (if applicable)",
    "fitness_value",
]

SPECIAL_PARENT_BY_EXPERIMENT = {
    "ARNLD-0917-39903c09-5dc0-4cc8-b805-a5739a835e85": "nucleotide_mutation",
}

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin1", errors="replace")


def parse_csv(path: Path):
    text = read_text(path)
    reader = csv.DictReader(io.StringIO(text))
    return reader.fieldnames or [], list(reader)


def norm_text(value) -> str:
    return "" if value is None else str(value).strip().lower()


def to_float(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    sl = s.lower()
    if sl in {"?", "nan", "#n.a.#", "#n.a", "n.a.", "none", "false", "true"}:
        return None
    s = s.replace(",", "")
    if s.endswith("%"):
        s = s[:-1].strip()
    try:
        return float(s)
    except ValueError:
        m = NUM_RE.search(s)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def get_num(row, *names):
    for name in names:
        if name in row:
            value = to_float(row.get(name))
            if value is not None:
                return value
    return None


def row_kind(row, experiment_folder=None):
    special_field = SPECIAL_PARENT_BY_EXPERIMENT.get(experiment_folder or "")
    if special_field:
        special_value = norm_text(row.get(special_field) or "")
        if "parent" in special_value:
            return "parent"
        if special_value:
            return "variant"

    aas = norm_text(row.get("amino_acid_substitutions") or "")
    if aas in PARENT_MARKERS:
        return "parent"
    if aas in INVALID_AAS:
        return "other"
    return "variant"


def qc_pass(row):
    alignment_count = get_num(row, "alignment_count")
    if alignment_count is not None and alignment_count < 4:
        return False, "alignment_count"
    return True, None


def infer_result_fields(header):
    header_set = set(header)
    return [field for field in RESULT_FIELD_CANDIDATES if field in header_set]


def choose_context_fields(header, rows):
    header_set = set(header)
    for field in ["reaction_smiles", "smiles_reaction", "smiles_string", "compound_name"]:
        if field in header_set:
            vals = {
                str(r.get(field)).strip()
                for r in rows
                if str(r.get(field) or "").strip() and norm_text(r.get(field)) not in INVALID_AAS
            }
            if len(vals) > 1:
                return [field]
    return []


def context_key(row, fields):
    if not fields:
        return ("__file__",)
    return tuple((f, str(row.get(f) or "").strip()) for f in fields)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    manifest_rows = []

    for csv_path in sorted(INPUT_DIR.glob("*/*.csv")):
        header, rows = parse_csv(csv_path)
        if not rows:
            continue

        result_fields = infer_result_fields(header)
        context_fields = choose_context_fields(header, rows)

        parent_values = defaultdict(lambda: defaultdict(list))
        parent_lookup = {}
        for row in rows:
            if row_kind(row, csv_path.parent.name) != "parent":
                continue
            ok, _ = qc_pass(row)
            if not ok:
                continue
            key = context_key(row, context_fields)
            parent_lookup[key] = row
            for field in result_fields:
                value = get_num(row, field)
                if value is not None:
                    parent_values[key][field].append(value)

        parent_means = {
            key: {field: mean(values) for field, values in field_map.items() if values}
            for key, field_map in parent_values.items()
            if field_map
        }
        global_parent_means = {
            field: mean(
                value
                for field_map in parent_values.values()
                for value in field_map.get(field, [])
            )
            for field in result_fields
            if any(field in field_map for field_map in parent_values.values())
        }

        output_rows = []
        seen_parent_keys = set()
        stats = Counter()

        for row in rows:
            if row_kind(row, csv_path.parent.name) != "variant":
                continue

            stats["variant_rows_seen"] += 1
            ok, reason = qc_pass(row)
            if not ok:
                stats[f"excluded_{reason}"] += 1
                continue

            key = context_key(row, context_fields)
            parent_metric_map = parent_means.get(key) or global_parent_means
            comparable_fields = []
            positive = False
            negative = False

            for field in result_fields:
                variant_value = get_num(row, field)
                parent_value = parent_metric_map.get(field) if parent_metric_map else None
                if variant_value is None or parent_value is None:
                    continue
                comparable_fields.append(field)
                tol = max(1e-9, 1e-6 * max(abs(variant_value), abs(parent_value), 1.0))
                if variant_value > parent_value + tol:
                    positive = True
                elif variant_value < parent_value - tol:
                    negative = True

            if not comparable_fields:
                stats["excluded_no_metric"] += 1
                continue
            if not parent_metric_map:
                stats["excluded_no_parent"] += 1
                continue

            if positive:
                stats["positive_count"] += 1
                if key not in seen_parent_keys and key in parent_lookup:
                    parent_copy = dict(parent_lookup[key])
                    parent_copy["record_role"] = "parent"
                    parent_copy["is_positive"] = ""
                    parent_copy["comparison_metric"] = ";".join(comparable_fields)
                    parent_copy["comparison_value"] = ""
                    output_rows.append(parent_copy)
                    seen_parent_keys.add(key)

                variant_copy = dict(row)
                variant_copy["record_role"] = "variant_positive"
                variant_copy["is_positive"] = "1"
                variant_copy["comparison_metric"] = ";".join(comparable_fields)
                variant_copy["comparison_value"] = ""
                output_rows.append(variant_copy)
            elif negative:
                stats["negative_count"] += 1
            else:
                stats["tie_count"] += 1

        out_path = OUTPUT_DIR / f"{csv_path.parent.name}.csv"

        fieldnames = list(rows[0].keys())
        for extra in ["record_role", "is_positive", "comparison_metric", "comparison_value"]:
            if extra not in fieldnames:
                fieldnames.append(extra)

        with out_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            if output_rows:
                writer.writerows(output_rows)

        manifest_rows.append(
            {
                "experiment_folder": csv_path.parent.name,
                "output_csv": out_path.name,
                "metric_field": ";".join(result_fields),
                "metric_rule": "or_across_result_fields",
                "context_fields": ";".join(context_fields),
                "variant_rows_seen": stats["variant_rows_seen"],
                "positive_count": stats["positive_count"],
                "negative_count": stats["negative_count"],
                "tie_count": stats["tie_count"],
                "excluded_alignment_count": stats["excluded_alignment_count"],
                "excluded_no_metric": stats["excluded_no_metric"],
                "excluded_no_parent": stats["excluded_no_parent"],
            }
        )

    manifest_path = ROOT / "positive_dataset_manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
