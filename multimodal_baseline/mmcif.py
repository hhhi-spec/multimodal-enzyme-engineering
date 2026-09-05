from __future__ import annotations

import math
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "XLE": "J",
    "MSE": "M",
}


def _split_cif_line(line: str) -> List[str]:
    try:
        return shlex.split(line, posix=True)
    except ValueError:
        return line.strip().split()


def _load_atom_site_loop(path: Path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.lower() != "loop_":
            i += 1
            continue
        i += 1
        headers = []
        while i < len(lines):
            stripped = lines[i].lstrip()
            if not stripped.startswith("_"):
                break
            headers.append(stripped.split()[0])
            i += 1
        if not headers or not any(h.startswith("_atom_site.") for h in headers):
            continue

        rows = []
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                continue
            if stripped.startswith("#"):
                i += 1
                break
            if stripped.lower() == "loop_" or stripped.startswith("_"):
                break
            tokens = _split_cif_line(lines[i])
            if len(tokens) == len(headers):
                rows.append(tokens)
            i += 1
        return headers, rows
    return [], []


def _choose_chain(records: Dict[str, List[Tuple[Tuple[int, str], str, np.ndarray]]]) -> str | None:
    if not records:
        return None
    return max(records.items(), key=lambda kv: len(kv[1]))[0]


def _align_sequences(parent_seq: str, struct_seq: str):
    n = len(parent_seq)
    m = len(struct_seq)
    if n == 0 or m == 0:
        return "", ""
    match = 2
    mismatch = -1
    gap = -2

    score = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + gap
        trace[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + gap
        trace[0][j] = 2

    for i in range(1, n + 1):
        a = parent_seq[i - 1]
        for j in range(1, m + 1):
            b = struct_seq[j - 1]
            diag = score[i - 1][j - 1] + (match if a == b else mismatch)
            up = score[i - 1][j] + gap
            left = score[i][j - 1] + gap
            best = diag
            code = 0
            if up > best:
                best = up
                code = 1
            if left > best:
                best = left
                code = 2
            score[i][j] = best
            trace[i][j] = code

    aligned_parent = []
    aligned_struct = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 0:
            aligned_parent.append(parent_seq[i - 1])
            aligned_struct.append(struct_seq[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or trace[i][j] == 1):
            aligned_parent.append(parent_seq[i - 1])
            aligned_struct.append("-")
            i -= 1
        else:
            aligned_parent.append("-")
            aligned_struct.append(struct_seq[j - 1])
            j -= 1

    return "".join(reversed(aligned_parent)), "".join(reversed(aligned_struct))


def project_coords_to_parent(
    parent_seq: str,
    struct_seq: str,
    struct_coords: List[np.ndarray],
):
    if not parent_seq:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if not struct_seq or not struct_coords:
        return np.full((len(parent_seq), 3), np.nan, dtype=np.float32), np.zeros((len(parent_seq),), dtype=np.int64)
    if parent_seq == struct_seq and len(parent_seq) == len(struct_coords):
        return np.asarray(struct_coords, dtype=np.float32), np.ones((len(parent_seq),), dtype=np.int64)

    aligned_parent, aligned_struct = _align_sequences(parent_seq, struct_seq)
    coords = np.full((len(parent_seq), 3), np.nan, dtype=np.float32)
    mask = np.zeros((len(parent_seq),), dtype=np.int64)
    p_i = 0
    s_i = 0
    for a, b in zip(aligned_parent, aligned_struct):
        if a != "-" and b != "-":
            if s_i < len(struct_coords):
                coords[p_i] = np.asarray(struct_coords[s_i], dtype=np.float32)
                mask[p_i] = 1
        if a != "-":
            p_i += 1
        if b != "-":
            s_i += 1
    return coords, mask


def parse_mmcif_backbone(path: Path):
    headers, rows = _load_atom_site_loop(path)
    if not headers or not rows:
        return "", np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    col = {name: idx for idx, name in enumerate(headers)}
    chain_to_residues: Dict[str, Dict[Tuple[int, str], Tuple[str, np.ndarray]]] = defaultdict(dict)

    for rec in rows:
        if len(rec) != len(headers):
            continue
        atom_name = rec[col.get("_atom_site.label_atom_id", -1)] if "_atom_site.label_atom_id" in col else ""
        if atom_name != "CA":
            continue
        group = rec[col.get("_atom_site.group_PDB", -1)] if "_atom_site.group_PDB" in col else "ATOM"
        if group not in {"ATOM", "HETATM"}:
            continue
        model_num = rec[col.get("_atom_site.pdbx_PDB_model_num", -1)] if "_atom_site.pdbx_PDB_model_num" in col else "1"
        if model_num not in {"1", "?", "."}:
            continue
        resn = rec[col.get("_atom_site.label_comp_id", -1)] if "_atom_site.label_comp_id" in col else "UNK"
        chain = rec[col.get("_atom_site.label_asym_id", -1)] if "_atom_site.label_asym_id" in col else ""
        if not chain and "_atom_site.auth_asym_id" in col:
            chain = rec[col["_atom_site.auth_asym_id"]]
        seq_id = rec[col.get("_atom_site.label_seq_id", -1)] if "_atom_site.label_seq_id" in col else ""
        if seq_id in {"", "?", "."}:
            continue
        try:
            seq_id_i = int(float(seq_id))
        except Exception:
            continue
        ins_code = rec[col.get("_atom_site.pdbx_PDB_ins_code", -1)] if "_atom_site.pdbx_PDB_ins_code" in col else "."
        try:
            x = float(rec[col["_atom_site.Cartn_x"]])
            y = float(rec[col["_atom_site.Cartn_y"]])
            z = float(rec[col["_atom_site.Cartn_z"]])
        except Exception:
            continue
        aa = AA3_TO_AA1.get(resn.upper(), "X")
        chain_to_residues[chain][(seq_id_i, ins_code)] = (aa, np.asarray([x, y, z], dtype=np.float32))

    best_chain = _choose_chain(chain_to_residues)
    if best_chain is None:
        return "", np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    ordered = sorted(chain_to_residues[best_chain].items(), key=lambda kv: (kv[0][0], kv[0][1]))
    seq = "".join(aa for _, (aa, _) in ordered)
    coords = [coord for _, (_, coord) in ordered]
    mask = np.ones((len(coords),), dtype=np.int64)
    return seq, np.asarray(coords, dtype=np.float32), mask

