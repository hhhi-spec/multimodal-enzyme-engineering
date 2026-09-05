from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_TO_IDX = {aa: i for i, aa in enumerate(AA20)}
IDX_TO_AA20 = {i: aa for aa, i in AA20_TO_IDX.items()}

SEQ_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
TEXT_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
SMILES_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

METRIC_HINTS = (
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
    "stability",
)


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def is_missing(text: str | None) -> bool:
    value = normalize_text(text)
    return value in {"", "?", "na", "n/a", "none", "nan"}


def is_trace_value(text: str | None) -> bool:
    value = normalize_text(text)
    return bool(value) and ("trace" in value or "below loq" in value or value == "loq")


def read_csv_rows(path: Path):
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "gbk", "latin1"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                return reader.fieldnames or [], list(reader)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Could not decode {path}")


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_nonempty(*values: str | None) -> str:
    for value in values:
        if value is not None and not is_missing(value):
            return str(value).strip()
    return ""


def split_components(smiles: str | None) -> List[str]:
    text = (smiles or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(".") if part.strip()]


def tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]", text.lower())


def tokenize_smiles(smiles: str) -> List[str]:
    if not smiles:
        return []
    return list(smiles.strip())


def infer_ec_major(text: str | None, fallback: int | None = None) -> int | None:
    match = re.search(r"EC\s*([1-6])", text or "", flags=re.I)
    if match:
        return int(match.group(1))
    return fallback


def is_positive_row(row: dict) -> bool:
    role = normalize_text(row.get("record_role"))
    if role == "parent":
        return False
    if "positive" in role:
        return True
    is_positive = normalize_text(row.get("is_positive"))
    return is_positive in {"1", "true", "yes"}


SUB_PATTERN = re.compile(r"([A-Z\*])(\d+)([A-Z\*])")


def parse_substitutions(substitutions: str | None):
    text = (substitutions or "").strip()
    if not text or normalize_text(text) in {"#parent#", "parent", "."}:
        return [], []
    sites: List[int] = []
    targets: List[str] = []
    for match in SUB_PATTERN.finditer(text):
        _, pos, target = match.groups()
        pos_i = int(pos)
        if pos_i not in sites:
            sites.append(pos_i)
            targets.append(target)
    return sites, targets


def apply_substitutions_to_sequence(parent_sequence: str, substitutions: str | None) -> str:
    sequence = list(parent_sequence or "")
    for match in SUB_PATTERN.finditer(substitutions or ""):
        wt, pos, target = match.groups()
        idx = int(pos) - 1
        if 0 <= idx < len(sequence):
            sequence[idx] = target
    return "".join(sequence)


def derive_mutations_from_sequences(parent_sequence: str, variant_sequence: str):
    sites: List[int] = []
    targets: List[str] = []
    if not parent_sequence or not variant_sequence:
        return sites, targets
    if len(parent_sequence) != len(variant_sequence):
        return sites, targets
    for idx, (parent_aa, variant_aa) in enumerate(zip(parent_sequence, variant_sequence), start=1):
        if parent_aa != variant_aa:
            sites.append(idx)
            targets.append(variant_aa)
    return sites, targets


def parse_direction_texts(row: dict) -> List[str]:
    texts: List[str] = []
    for key, value in row.items():
        if key and key.lower().startswith("direction_text"):
            if value and not is_missing(value):
                text = str(value).strip()
                if text not in texts:
                    texts.append(text)
    return texts


def choose_primary_text(texts: Sequence[str]) -> str:
    return texts[0] if texts else ""


def parse_float(text: str | None):
    if text is None:
        return None
    cleaned = str(text).strip().replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def detect_metric_labels(fields: Iterable[str]) -> List[str]:
    labels: List[str] = []
    for field in fields:
        lf = field.lower()
        if any(h in lf for h in METRIC_HINTS):
            if "fitness" in lf:
                label = "fitness value"
            elif "activity_for_reaction" in lf or lf == "activity":
                label = "reaction activity"
            elif "ttn" in lf:
                label = "TTN"
            elif "yield" in lf:
                label = "yield"
            elif "conversion" in lf:
                label = "conversion"
            elif "selectivity" in lf or lf == "ee":
                label = "selectivity"
            elif "tm" in lf or "thermal" in lf or "stability" in lf:
                label = "thermal stability"
            else:
                label = field
            if label not in labels:
                labels.append(label)
    return labels


def guess_dataset_kind(csv_path: Path) -> str:
    if "ec3" in csv_path.as_posix().lower():
        return "ec3"
    return "positive"


def safe_int(value: str | None, default: int | None = None) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return default
