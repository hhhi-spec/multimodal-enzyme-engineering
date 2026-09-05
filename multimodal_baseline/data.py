from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    AA20,
    AA20_TO_IDX,
    IDX_TO_AA20,
    ROOT,
    SEQ_SPECIAL_TOKENS,
    SMILES_SPECIAL_TOKENS,
    TEXT_SPECIAL_TOKENS,
    load_jsonl,
    split_components,
    tokenize_smiles,
    tokenize_text,
)


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: Sequence[str], add_special: bool = True) -> List[int]:
        ids: List[int] = []
        if add_special:
            ids.append(self.bos_id)
        for token in tokens:
            ids.append(self.token_to_id.get(token, self.unk_id))
        if add_special:
            ids.append(self.eos_id)
        return ids

    def encode_text(self, text: str, max_len: int | None = None) -> List[int]:
        ids = self.encode(tokenize_text(text), add_special=True)
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def encode_smiles(self, smiles: str, max_len: int | None = None) -> List[int]:
        ids = self.encode(tokenize_smiles(smiles), add_special=True)
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    @classmethod
    def build(cls, tokens: Iterable[str], specials: Sequence[str]):
        vocab = {tok: idx for idx, tok in enumerate(specials)}
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        return cls(vocab)

    def to_dict(self) -> Dict[str, int]:
        return dict(self.token_to_id)

    @classmethod
    def from_dict(cls, d: Dict[str, int]):
        return cls(dict(d))


@dataclass
class VocabBundle:
    seq_vocab: Vocabulary
    text_vocab: Vocabulary
    smiles_vocab: Vocabulary

    def to_json(self) -> Dict[str, Dict[str, int]]:
        return {
            "seq": self.seq_vocab.to_dict(),
            "text": self.text_vocab.to_dict(),
            "smiles": self.smiles_vocab.to_dict(),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Dict[str, int]]):
        return cls(
            seq_vocab=Vocabulary.from_dict(data["seq"]),
            text_vocab=Vocabulary.from_dict(data["text"]),
            smiles_vocab=Vocabulary.from_dict(data["smiles"]),
        )


@dataclass
class HFTokenizerBundle:
    seq_tokenizer: object | None = None
    text_tokenizer: object | None = None


def build_vocab_bundle(manifest_rows: Sequence[dict]) -> VocabBundle:
    text_tokens: List[str] = []
    smiles_tokens: List[str] = []

    for row in manifest_rows:
        for text in row.get("direction_texts", []) or [row.get("direction_text", "")]:
            text_tokens.extend(tokenize_text(text or ""))
        for component in split_components(row.get("substrate_smiles", "")):
            smiles_tokens.extend(tokenize_smiles(component))

    seq_specials = ["<pad>", "<unk>", "<bos>", "<eos>"] + list(AA20) + ["X", "B", "Z", "J", "U", "O", "*"]
    text_vocab = Vocabulary.build(text_tokens, TEXT_SPECIAL_TOKENS)
    smiles_vocab = Vocabulary.build(smiles_tokens, SMILES_SPECIAL_TOKENS)
    seq_vocab = Vocabulary.build([], seq_specials)
    return VocabBundle(seq_vocab=seq_vocab, text_vocab=text_vocab, smiles_vocab=smiles_vocab)


def save_vocab_bundle(path: Path, bundle: VocabBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocab_bundle(path: Path) -> VocabBundle:
    data = json.loads(path.read_text(encoding="utf-8"))
    return VocabBundle.from_json(data)


def pad_1d(seqs: Sequence[Sequence[int]], pad_value: int) -> torch.Tensor:
    max_len = max((len(s) for s in seqs), default=0)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(seqs):
        if seq:
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def pad_2d(coords: Sequence[Sequence[Sequence[float]]], pad_value: float = float("nan")) -> torch.Tensor:
    max_len = max((len(s) for s in coords), default=0)
    out = torch.full((len(coords), max_len, 3), pad_value, dtype=torch.float32)
    for i, seq in enumerate(coords):
        if seq:
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.float32)
    return out


def pad_2d_features(
    rows: Sequence[Sequence[Sequence[float]]],
    feature_dim: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    max_len = max((len(s) for s in rows), default=0)
    out = torch.full((len(rows), max_len, feature_dim), pad_value, dtype=torch.float32)
    for i, seq in enumerate(rows):
        if seq:
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.float32)
    return out


def pad_mask(masks: Sequence[Sequence[int]], pad_value: int = 0) -> torch.Tensor:
    max_len = max((len(s) for s in masks), default=0)
    out = torch.full((len(masks), max_len), pad_value, dtype=torch.bool)
    for i, seq in enumerate(masks):
        if seq:
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.bool)
    return out


class MultimodalManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        vocab_path: Path,
        split: str | Sequence[str] | None,
        max_text_len: int = 256,
        max_smiles_len: int = 128,
        max_components: int = 6,
        train_text_augmentation: bool = False,
        rows: Sequence[dict] | None = None,
    ):
        all_rows = list(rows) if rows is not None else load_jsonl(manifest_path)
        if split is None:
            self.records = all_rows
        elif isinstance(split, str):
            self.records = [row for row in all_rows if row.get("split") == split]
        else:
            split_set = set(split)
            self.records = [row for row in all_rows if row.get("split") in split_set]
        self.vocab = load_vocab_bundle(vocab_path)
        self.split = split
        self.max_text_len = max_text_len
        self.max_smiles_len = max_smiles_len
        self.max_components = max_components
        self.train_text_augmentation = train_text_augmentation and split == "train"

    def __len__(self) -> int:
        return len(self.records)

    def _select_text(self, row: dict) -> str:
        texts = row.get("direction_texts") or [row.get("direction_text", "")]
        texts = [t for t in texts if t]
        if not texts:
            return ""
        if self.train_text_augmentation and len(texts) > 1:
            return random.choice(texts)
        return texts[0]

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        text = self._select_text(row)
        substrate_components = split_components(row.get("substrate_smiles", ""))
        substrate_components = substrate_components[: self.max_components]

        seq_ids = self.vocab.seq_vocab.encode(list(row.get("parent_sequence", "").strip()), add_special=False)
        text_ids = self.vocab.text_vocab.encode_text(text, max_len=self.max_text_len)
        component_ids = [
            self.vocab.smiles_vocab.encode_smiles(component, max_len=self.max_smiles_len)
            for component in substrate_components
        ]

        coords = row.get("parent_3d_coords") or []
        mask = row.get("parent_3d_mask") or []
        site_mask = [0] * len(row.get("parent_sequence", ""))
        aa_target = [-100] * len(row.get("parent_sequence", ""))
        mutation_sites = row.get("mutation_sites") or []
        mutation_targets = row.get("mutation_targets") or []
        for pos, target in zip(mutation_sites, mutation_targets):
            if pos is None:
                continue
            idx0 = int(pos) - 1
            if 0 <= idx0 < len(site_mask):
                site_mask[idx0] = 1
                aa_target[idx0] = AA20_TO_IDX.get(str(target).upper(), -100)

        return {
            "sample_id": row.get("sample_id", ""),
            "experiment_id": row.get("experiment_id", ""),
            "ec_major": int(row.get("ec_major") or 0),
            "parent_sequence_raw": row.get("parent_sequence", "").strip(),
            "direction_text_raw": text,
            "seq_ids": seq_ids,
            "text_ids": text_ids,
            "component_ids": component_ids,
            "ligand_atom_features": row.get("substrate_atom_features") or [],
            "ligand_atom_coords": row.get("substrate_atom_coords") or [],
            "ligand_atom_component_ids": row.get("substrate_atom_component_ids") or [],
            "coords": coords,
            "coord_mask": mask,
            "site_mask": site_mask,
            "aa_target": aa_target,
            "seq_len": len(row.get("parent_sequence", "")),
            "text_len": len(text_ids),
            "num_components": len(component_ids),
        }


def _tokenize_sequence_batch_with_hf(batch: Sequence[dict], tokenizer):
    encoded = tokenizer(
        [item["parent_sequence_raw"] for item in batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_special_tokens_mask=True,
    )
    seq_ids = encoded["input_ids"].long()
    attention_mask = encoded["attention_mask"].bool()
    special_tokens_mask = encoded.get("special_tokens_mask", torch.zeros_like(seq_ids)).bool()

    batch_size, seq_len = seq_ids.shape
    coords = torch.full((batch_size, seq_len, 3), float("nan"), dtype=torch.float32)
    coord_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    residue_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    site_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    aa_target = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        residue_positions = ((~special_tokens_mask[i]) & attention_mask[i]).nonzero(as_tuple=False).flatten().tolist()
        raw_coords = item["coords"]
        raw_coord_mask = item["coord_mask"]
        raw_site_mask = item["site_mask"]
        raw_aa_target = item["aa_target"]
        limit = min(len(residue_positions), len(raw_site_mask))
        for src_idx, token_pos in zip(range(limit), residue_positions[:limit]):
            residue_mask[i, token_pos] = True
            if src_idx < len(raw_coords):
                coords[i, token_pos] = torch.tensor(raw_coords[src_idx], dtype=torch.float32)
            if src_idx < len(raw_coord_mask):
                coord_mask[i, token_pos] = bool(raw_coord_mask[src_idx])
            if src_idx < len(raw_site_mask):
                site_mask[i, token_pos] = bool(raw_site_mask[src_idx])
            if src_idx < len(raw_aa_target):
                aa_target[i, token_pos] = int(raw_aa_target[src_idx])
    return seq_ids, coords, coord_mask, residue_mask, site_mask, aa_target


def _tokenize_text_batch_with_hf(batch: Sequence[dict], tokenizer):
    encoded = tokenizer(
        [item["direction_text_raw"] for item in batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    return encoded["input_ids"].long(), encoded["attention_mask"].bool()


def collate_multimodal(
    batch: Sequence[dict],
    seq_pad_id: int,
    text_pad_id: int,
    smiles_pad_id: int,
    tokenizer_bundle: HFTokenizerBundle | None = None,
):
    tokenizer_bundle = tokenizer_bundle or HFTokenizerBundle()
    seq_ids = pad_1d([item["seq_ids"] for item in batch], seq_pad_id)
    seq_attention_mask = seq_ids.ne(seq_pad_id)
    coords = pad_2d([item["coords"] for item in batch])
    coord_mask = pad_mask([item["coord_mask"] for item in batch])
    residue_mask = seq_attention_mask.clone()
    site_mask = pad_mask([item["site_mask"] for item in batch])
    aa_target = pad_1d([item["aa_target"] for item in batch], -100)

    if tokenizer_bundle.text_tokenizer is not None:
        text_ids, text_attention_mask = _tokenize_text_batch_with_hf(batch, tokenizer_bundle.text_tokenizer)
    else:
        text_ids = pad_1d([item["text_ids"] for item in batch], text_pad_id)
        text_attention_mask = text_ids.ne(text_pad_id)

    max_components = max((item["num_components"] for item in batch), default=0)
    max_comp_len = max(
        (len(comp) for item in batch for comp in item["component_ids"]),
        default=0,
    )
    component_ids = torch.full((len(batch), max_components, max_comp_len), smiles_pad_id, dtype=torch.long)
    component_mask = torch.zeros((len(batch), max_components), dtype=torch.bool)
    component_token_mask = torch.zeros((len(batch), max_components, max_comp_len), dtype=torch.bool)
    for b, item in enumerate(batch):
        for c, comp in enumerate(item["component_ids"]):
            component_mask[b, c] = True
            if comp:
                component_ids[b, c, : len(comp)] = torch.tensor(comp, dtype=torch.long)
                component_token_mask[b, c, : len(comp)] = True

    ligand_atom_features = pad_2d_features([item["ligand_atom_features"] for item in batch], feature_dim=5, pad_value=0.0)
    ligand_atom_coords = pad_2d([item["ligand_atom_coords"] for item in batch], pad_value=0.0)
    ligand_atom_mask = pad_mask([[1] * len(item["ligand_atom_features"]) for item in batch])
    max_atoms = ligand_atom_features.size(1)
    ligand_component_ids = torch.full((len(batch), max_atoms), -1, dtype=torch.long)
    for b, item in enumerate(batch):
        comp_ids = item["ligand_atom_component_ids"]
        if comp_ids:
            ligand_component_ids[b, : len(comp_ids)] = torch.tensor(comp_ids, dtype=torch.long)

    return {
        "sample_id": [item["sample_id"] for item in batch],
        "experiment_id": [item["experiment_id"] for item in batch],
        "ec_major": torch.tensor([item["ec_major"] for item in batch], dtype=torch.long),
        "parent_sequence_raw": [item["parent_sequence_raw"] for item in batch],
        "seq_ids": seq_ids,
        "seq_attention_mask": seq_attention_mask,
        "text_ids": text_ids,
        "text_attention_mask": text_attention_mask,
        "component_ids": component_ids,
        "component_mask": component_mask,
        "component_token_mask": component_token_mask,
        "ligand_atom_features": ligand_atom_features,
        "ligand_atom_coords": ligand_atom_coords,
        "ligand_atom_mask": ligand_atom_mask,
        "ligand_component_ids": ligand_component_ids,
        "coords": coords,
        "coord_mask": coord_mask,
        "residue_mask": residue_mask,
        "site_mask": site_mask,
        "aa_target": aa_target,
        "seq_pad_id": seq_pad_id,
        "text_pad_id": text_pad_id,
        "smiles_pad_id": smiles_pad_id,
    }
