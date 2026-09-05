from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertConfig, BertModel, EsmConfig, EsmModel

from .data import VocabBundle
from .enzygen2_egnn import EGNN, SubstrateEGNN, build_batch_knn_graph, build_grouped_batch_knn_graph


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    denom = mask_f.sum(dim=dim).clamp_min(eps)
    return (x * mask_f).sum(dim=dim) / denom


def split_sequence_windows(sequence: str, window_size: int, overlap: int) -> list[tuple[int, int]]:
    sequence = sequence or ""
    length = len(sequence)
    if length == 0:
        return []
    if length <= window_size:
        return [(0, length)]
    stride = max(1, window_size - overlap)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(length, start + window_size)
        spans.append((start, end))
        if end >= length:
            break
        start += stride
    return spans


class LocalTokenTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.hidden_size = d_model

    def forward(self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ids.numel() == 0:
            empty = ids.new_zeros((ids.size(0), 0, self.hidden_size), dtype=torch.float32)
            mask = ids.new_zeros((ids.size(0), 0), dtype=torch.bool)
            pooled = ids.new_zeros((ids.size(0), self.hidden_size), dtype=torch.float32)
            return empty, pooled, mask
        mask = attention_mask.bool() if attention_mask is not None else ids.ne(self.pad_id)
        seq_len = ids.size(1)
        pos = torch.arange(seq_len, device=ids.device).unsqueeze(0).expand(ids.size(0), -1)
        x = torch.zeros((ids.size(0), seq_len, self.hidden_size), device=ids.device, dtype=torch.float32)
        pooled = torch.zeros((ids.size(0), self.hidden_size), device=ids.device, dtype=torch.float32)
        valid_rows = mask.any(dim=1)
        if valid_rows.any():
            valid_ids = ids[valid_rows]
            valid_mask = mask[valid_rows]
            valid_pos = pos[valid_rows]
            valid_x = self.token_emb(valid_ids) + self.pos_emb(valid_pos)
            valid_x = self.encoder(valid_x, src_key_padding_mask=~valid_mask)
            valid_x = self.norm(valid_x)
            valid_x = valid_x * valid_mask.unsqueeze(-1).float()
            x[valid_rows] = valid_x
            pooled[valid_rows] = masked_mean(valid_x, valid_mask, dim=1)
        return x, pooled, mask


class HFSequenceEncoder(nn.Module):
    def __init__(
        self,
        seq_vocab,
        pad_token_id: int,
        mask_token_id: int,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_len: int = 2048,
        dropout: float = 0.1,
        pretrained_dir: str | None = None,
        window_size: int = 512,
        window_overlap: int = 256,
    ):
        super().__init__()
        self.seq_vocab = seq_vocab
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.pretrained_dir = pretrained_dir if pretrained_dir and Path(pretrained_dir).exists() else None
        self.tokenizer = None
        model = None
        if self.pretrained_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.pretrained_dir), local_files_only=True)
            model = EsmModel.from_pretrained(str(self.pretrained_dir), local_files_only=True)
        else:
            config = EsmConfig(
                vocab_size=max(seq_vocab.size, 33),
                pad_token_id=pad_token_id,
                mask_token_id=mask_token_id,
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                intermediate_size=hidden_size * 4,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                max_position_embeddings=max_len + 2,
                emb_layer_norm_before=True,
                token_dropout=False,
            )
            model = EsmModel(config)
        self.model = model
        self.hidden_size = model.config.hidden_size
        self.pad_token_id = int(
            self.tokenizer.pad_token_id if self.tokenizer is not None and self.tokenizer.pad_token_id is not None else pad_token_id
        )
        self.cls_token_id = int(
            self.tokenizer.cls_token_id if self.tokenizer is not None and self.tokenizer.cls_token_id is not None else seq_vocab.bos_id
        )
        self.eos_token_id = int(
            self.tokenizer.eos_token_id if self.tokenizer is not None and self.tokenizer.eos_token_id is not None else seq_vocab.eos_id
        )
        self.unk_token_id = int(
            self.tokenizer.unk_token_id if self.tokenizer is not None and self.tokenizer.unk_token_id is not None else seq_vocab.unk_id
        )

    def _encode_residue_ids(self, residues: str) -> list[int]:
        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(residues)
            return list(self.tokenizer.convert_tokens_to_ids(tokens))
        return [self.seq_vocab.token_to_id.get(token, self.unk_token_id) for token in list(residues)]

    def _build_window_inputs(self, sequence: str) -> list[tuple[list[int], int, int, bool, bool]]:
        spans = split_sequence_windows(sequence, self.window_size, self.window_overlap)
        windows: list[tuple[list[int], int, int, bool, bool]] = []
        for idx, (start, end) in enumerate(spans):
            token_ids = self._encode_residue_ids(sequence[start:end])
            add_cls = idx == 0
            add_eos = idx == len(spans) - 1
            if add_cls:
                token_ids = [self.cls_token_id] + token_ids
            if add_eos:
                token_ids = token_ids + [self.eos_token_id]
            windows.append((token_ids, start, end, add_cls, add_eos))
        return windows

    def _forward_tensor(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask.long())
        hidden = outputs.last_hidden_state
        mask = attention_mask.bool()
        hidden = hidden * mask.unsqueeze(-1).float()
        pooled = masked_mean(hidden, mask, dim=1)
        return hidden, pooled, mask

    def _forward_strings(self, sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        batch_size = len(sequences)
        lengths = [len(seq or "") for seq in sequences]
        max_len = max(lengths, default=0)
        if batch_size == 0 or max_len == 0:
            empty = torch.zeros((batch_size, 0, self.hidden_size), device=device)
            pooled = torch.zeros((batch_size, self.hidden_size), device=device)
            mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=device)
            return empty, pooled, mask

        window_inputs: list[torch.Tensor] = []
        window_meta: list[tuple[int, int, int, bool, bool]] = []
        for batch_idx, sequence in enumerate(sequences):
            for token_ids, start, end, add_cls, add_eos in self._build_window_inputs(sequence or ""):
                window_inputs.append(torch.tensor(token_ids, dtype=torch.long, device=device))
                window_meta.append((batch_idx, start, end, add_cls, add_eos))

        if not window_inputs:
            empty = torch.zeros((batch_size, max_len, self.hidden_size), device=device)
            pooled = torch.zeros((batch_size, self.hidden_size), device=device)
            mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
            return empty, pooled, mask

        padded = torch.nn.utils.rnn.pad_sequence(window_inputs, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = padded.ne(self.pad_token_id)
        outputs = self.model(input_ids=padded, attention_mask=attention_mask.long())
        hidden = outputs.last_hidden_state

        full_repr = torch.zeros((batch_size, max_len, self.hidden_size), device=device, dtype=hidden.dtype)
        coverage = torch.zeros((batch_size, max_len, 1), device=device, dtype=hidden.dtype)

        for window_idx, (batch_idx, start, end, add_cls, add_eos) in enumerate(window_meta):
            valid_len = int(attention_mask[window_idx].sum().item())
            window_hidden = hidden[window_idx, :valid_len]
            residue_hidden = window_hidden[1 if add_cls else 0 : valid_len - (1 if add_eos else 0)]
            if residue_hidden.numel() == 0:
                continue
            span_len = end - start
            take = min(span_len, residue_hidden.size(0))
            if take <= 0:
                continue
            full_repr[batch_idx, start : start + take] += residue_hidden[:take]
            coverage[batch_idx, start : start + take] += 1.0

        mask = coverage.squeeze(-1) > 0
        full_repr = full_repr / coverage.clamp_min(1.0)
        full_repr = full_repr * mask.unsqueeze(-1).float()
        pooled = masked_mean(full_repr, mask, dim=1)
        return full_repr, pooled, mask

    def forward(
        self,
        input_ids: torch.Tensor | Sequence[str],
        attention_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(input_ids, (list, tuple)) and (not input_ids or isinstance(input_ids[0], str)):
            return self._forward_strings(list(input_ids))
        if not torch.is_tensor(input_ids):
            raise TypeError(f"Unsupported input type for sequence encoder: {type(input_ids)!r}")
        return self._forward_tensor(input_ids, attention_mask)


class HFBioBERTStyleEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_len: int = 512,
        dropout: float = 0.1,
        pretrained_dir: str | None = None,
    ):
        super().__init__()
        model = None
        if pretrained_dir and Path(pretrained_dir).exists():
            model = BertModel.from_pretrained(str(pretrained_dir), local_files_only=True)
        else:
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                intermediate_size=hidden_size * 4,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                max_position_embeddings=max_len,
                pad_token_id=pad_token_id,
            )
            model = BertModel(config)
        self.model = model
        self.hidden_size = model.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask.long())
        hidden = outputs.last_hidden_state
        mask = attention_mask.bool()
        hidden = hidden * mask.unsqueeze(-1).float()
        pooled = masked_mean(hidden, mask, dim=1)
        return hidden, pooled, mask


class EnzyGen2StructureEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_layers: int = 3,
        knn_k: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.knn_k = knn_k
        self.egnn = EGNN(
            in_node_nf=hidden_size,
            hidden_nf=hidden_size,
            out_node_nf=hidden_size,
            n_layers=n_layers,
            attention=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, protein_repr: torch.Tensor, coords: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        flat_coords, edge_index, valid_indices_per_sample, _ = build_batch_knn_graph(coords.detach(), valid_mask, k=self.knn_k)
        if flat_coords.numel() == 0:
            return torch.zeros_like(protein_repr)

        flat_features = []
        for batch_idx, valid_indices in enumerate(valid_indices_per_sample):
            if valid_indices.numel() > 0:
                flat_features.append(protein_repr[batch_idx, valid_indices])
        flat_h = torch.cat(flat_features, dim=0)
        flat_h, _ = self.egnn(flat_h, flat_coords, edge_index, edge_attr=None)

        out = torch.zeros_like(protein_repr)
        cursor = 0
        for batch_idx, valid_indices in enumerate(valid_indices_per_sample):
            count = valid_indices.numel()
            if count == 0:
                continue
            out[batch_idx, valid_indices] = flat_h[cursor : cursor + count]
            cursor += count
        out = self.norm(self.dropout(out))
        return out


class SubstrateEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        token_layers: int = 2,
        token_heads: int = 8,
        component_layers: int = 2,
        component_heads: int = 8,
        max_token_len: int = 256,
        max_components: int = 8,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.token_encoder = LocalTokenTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=token_heads,
            n_layers=token_layers,
            max_len=max_token_len,
            dropout=dropout,
            pad_id=pad_id,
        )
        self.component_pos = nn.Embedding(max_components + 4, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=component_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.component_encoder = nn.TransformerEncoder(layer, num_layers=component_layers)
        self.component_norm = nn.LayerNorm(d_model)
        self.hidden_size = d_model

    def forward(self, component_ids: torch.Tensor, component_token_mask: torch.Tensor, component_mask: torch.Tensor):
        bsz, n_comp, comp_len = component_ids.shape
        if n_comp == 0:
            empty = component_ids.new_zeros((bsz, 0, self.hidden_size), dtype=torch.float32)
            pooled = component_ids.new_zeros((bsz, self.hidden_size), dtype=torch.float32)
            return empty, pooled, component_mask

        flat_ids = component_ids.view(bsz * n_comp, comp_len)
        flat_mask = component_token_mask.view(bsz * n_comp, comp_len)
        _, pooled, _ = self.token_encoder(flat_ids, flat_mask)
        comp_repr = pooled.view(bsz, n_comp, -1)
        pos = torch.arange(n_comp, device=component_ids.device).unsqueeze(0).expand(bsz, -1)
        comp_repr = comp_repr + self.component_pos(pos)
        comp_repr = self.component_encoder(comp_repr, src_key_padding_mask=~component_mask)
        comp_repr = self.component_norm(comp_repr)
        comp_repr = comp_repr * component_mask.unsqueeze(-1).float()
        substrate_vec = masked_mean(comp_repr, component_mask, dim=1)
        return comp_repr, substrate_vec, component_mask


class SubstrateGeometryEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_layers: int = 3,
        knn_k: int = 12,
        max_components: int = 8,
        component_layers: int = 2,
        component_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.knn_k = knn_k
        self.max_components = max_components
        self.substrate_egnn = SubstrateEGNN(
            in_node_nf=5,
            hidden_nf=hidden_size,
            out_node_nf=hidden_size,
            n_layers=n_layers,
            attention=True,
        )
        self.component_pos = nn.Embedding(max_components + 4, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=component_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.component_encoder = nn.TransformerEncoder(layer, num_layers=component_layers)
        self.component_norm = nn.LayerNorm(hidden_size)
        self.hidden_size = hidden_size

    def forward(
        self,
        atom_features: torch.Tensor,
        atom_coords: torch.Tensor,
        atom_mask: torch.Tensor,
        component_ids: torch.Tensor,
    ):
        bsz, max_atoms, _ = atom_features.shape
        if max_atoms == 0 or not atom_mask.any():
            empty_repr = atom_features.new_zeros((bsz, 0, self.hidden_size))
            empty_vec = atom_features.new_zeros((bsz, self.hidden_size))
            empty_mask = atom_mask.new_zeros((bsz, 0), dtype=torch.bool)
            return empty_repr, empty_vec, empty_mask

        atom_coords = torch.nan_to_num(atom_coords, nan=0.0, posinf=0.0, neginf=0.0)
        flat_coords, edge_index, valid_indices_per_sample, _ = build_grouped_batch_knn_graph(
            atom_coords.detach(),
            atom_mask,
            component_ids,
            k=self.knn_k,
        )
        if flat_coords.numel() == 0:
            empty_repr = atom_features.new_zeros((bsz, 0, self.hidden_size))
            empty_vec = atom_features.new_zeros((bsz, self.hidden_size))
            empty_mask = atom_mask.new_zeros((bsz, 0), dtype=torch.bool)
            return empty_repr, empty_vec, empty_mask

        flat_features = []
        for batch_idx, valid_indices in enumerate(valid_indices_per_sample):
            if valid_indices.numel() > 0:
                flat_features.append(atom_features[batch_idx, valid_indices])
        flat_h = torch.cat(flat_features, dim=0)
        flat_h, _ = self.substrate_egnn(flat_h, flat_coords, edge_index, edge_attr=None)

        atom_repr = atom_features.new_zeros((bsz, max_atoms, self.hidden_size))
        cursor = 0
        for batch_idx, valid_indices in enumerate(valid_indices_per_sample):
            count = valid_indices.numel()
            if count == 0:
                continue
            atom_repr[batch_idx, valid_indices] = flat_h[cursor : cursor + count]
            cursor += count

        max_components_in_batch = 0
        for batch_idx in range(bsz):
            valid_groups = component_ids[batch_idx][atom_mask[batch_idx]]
            if valid_groups.numel() > 0:
                max_components_in_batch = max(max_components_in_batch, int(valid_groups.max().item()) + 1)

        if max_components_in_batch == 0:
            empty_repr = atom_features.new_zeros((bsz, 0, self.hidden_size))
            empty_vec = atom_features.new_zeros((bsz, self.hidden_size))
            empty_mask = atom_mask.new_zeros((bsz, 0), dtype=torch.bool)
            return empty_repr, empty_vec, empty_mask

        comp_repr = atom_features.new_zeros((bsz, max_components_in_batch, self.hidden_size))
        comp_mask = atom_mask.new_zeros((bsz, max_components_in_batch), dtype=torch.bool)
        for batch_idx in range(bsz):
            valid_groups = component_ids[batch_idx][atom_mask[batch_idx]]
            if valid_groups.numel() == 0:
                continue
            for comp_idx in sorted(set(int(x) for x in valid_groups.tolist() if int(x) >= 0)):
                comp_atom_mask = atom_mask[batch_idx] & component_ids[batch_idx].eq(comp_idx)
                if not comp_atom_mask.any():
                    continue
                comp_repr[batch_idx, comp_idx] = atom_repr[batch_idx, comp_atom_mask].mean(dim=0)
                comp_mask[batch_idx, comp_idx] = True

        pos = torch.arange(max_components_in_batch, device=atom_features.device).unsqueeze(0).expand(bsz, -1)
        pos = pos.clamp_max(self.component_pos.num_embeddings - 1)
        comp_repr = comp_repr + self.component_pos(pos)
        comp_repr = self.component_encoder(comp_repr, src_key_padding_mask=~comp_mask)
        comp_repr = self.component_norm(comp_repr)
        comp_repr = comp_repr * comp_mask.unsqueeze(-1).float()
        substrate_vec = masked_mean(comp_repr, comp_mask, dim=1)
        return comp_repr, substrate_vec, comp_mask


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor):
        attn_out, _ = self.attn(query, context, context, key_padding_mask=~context_mask)
        x = self.norm1(query + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class MultimodalEnzymeBaseline(nn.Module):
    def __init__(
        self,
        vocab_bundle: VocabBundle,
        fusion_dim: int = 256,
        seq_hidden_size: int = 256,
        text_hidden_size: int = 256,
        seq_layers: int = 6,
        text_layers: int = 4,
        smiles_layers: int = 2,
        struct_layers: int = 3,
        fusion_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        max_text_len: int = 512,
        max_smiles_len: int = 256,
        max_components: int = 8,
        seq_pretrained_dir: str | None = None,
        text_pretrained_dir: str | None = None,
        seq_window_size: int = 512,
        seq_window_overlap: int = 256,
        freeze_seq_backbone: bool = False,
        freeze_text_backbone: bool = False,
    ):
        super().__init__()
        self.vocab_bundle = vocab_bundle
        self.seq_encoder = HFSequenceEncoder(
            seq_vocab=vocab_bundle.seq_vocab,
            pad_token_id=vocab_bundle.seq_vocab.pad_id,
            mask_token_id=vocab_bundle.seq_vocab.unk_id,
            hidden_size=seq_hidden_size,
            num_layers=seq_layers,
            num_heads=n_heads,
            max_len=max_seq_len,
            dropout=dropout,
            pretrained_dir=seq_pretrained_dir,
            window_size=seq_window_size,
            window_overlap=seq_window_overlap,
        )
        self.struct_encoder = EnzyGen2StructureEncoder(
            hidden_size=self.seq_encoder.hidden_size,
            n_layers=struct_layers,
            knn_k=30,
            dropout=dropout,
        )
        self.text_encoder = HFBioBERTStyleEncoder(
            vocab_size=vocab_bundle.text_vocab.size,
            pad_token_id=vocab_bundle.text_vocab.pad_id,
            hidden_size=text_hidden_size,
            num_layers=text_layers,
            num_heads=n_heads,
            max_len=max_text_len,
            dropout=dropout,
            pretrained_dir=text_pretrained_dir,
        )
        self.substrate_encoder = SubstrateGeometryEncoder(
            hidden_size=fusion_dim,
            n_layers=smiles_layers,
            knn_k=12,
            max_components=max_components,
            component_layers=2,
            component_heads=n_heads,
            dropout=dropout,
        )

        self.stage1_seq_proj = nn.Linear(self.seq_encoder.hidden_size, fusion_dim)
        self.stage1_struct_proj = nn.Linear(self.seq_encoder.hidden_size, fusion_dim)
        self.stage2_seq_proj = nn.Linear(self.seq_encoder.hidden_size, fusion_dim)
        self.text_proj = nn.Linear(self.text_encoder.hidden_size, fusion_dim)

        self.stage1_fusion_blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model=fusion_dim, n_heads=n_heads, dropout=dropout) for _ in range(fusion_layers)]
        )
        self.stage2_fusion_blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model=fusion_dim, n_heads=n_heads, dropout=dropout) for _ in range(fusion_layers)]
        )
        self.stage1_norm = nn.LayerNorm(fusion_dim)
        self.stage2_norm = nn.LayerNorm(fusion_dim)
        self.site_head = nn.Linear(fusion_dim, 1)
        self.aa_head = nn.Linear(fusion_dim, 20)

        if freeze_seq_backbone:
            for parameter in self.seq_encoder.model.parameters():
                parameter.requires_grad = False
        if freeze_text_backbone:
            for parameter in self.text_encoder.model.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        parent_sequences: Sequence[str] | torch.Tensor | None = None,
        seq_ids: torch.Tensor | None = None,
        seq_attention_mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        coord_mask: torch.Tensor | None = None,
        text_ids: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        ligand_atom_features: torch.Tensor | None = None,
        ligand_atom_coords: torch.Tensor | None = None,
        ligand_atom_mask: torch.Tensor | None = None,
        ligand_component_ids: torch.Tensor | None = None,
    ):
        seq_input = parent_sequences if parent_sequences is not None else seq_ids
        if seq_input is None:
            raise ValueError("Either parent_sequences or seq_ids must be provided.")
        if coords is None or coord_mask is None:
            raise ValueError("coords and coord_mask are required.")
        if text_ids is None or text_attention_mask is None:
            raise ValueError("text_ids and text_attention_mask are required.")
        if ligand_atom_features is None or ligand_atom_coords is None or ligand_atom_mask is None or ligand_component_ids is None:
            raise ValueError("Ligand inputs are required.")
        seq_repr, _, seq_mask = self.seq_encoder(seq_input, seq_attention_mask)

        stage1_seq = self.stage1_seq_proj(seq_repr)
        struct_repr = self.struct_encoder(seq_repr, coords, coord_mask)
        stage1_protein = self.stage1_norm(stage1_seq + self.stage1_struct_proj(struct_repr))

        text_repr, text_vec, text_mask = self.text_encoder(text_ids, text_attention_mask)
        text_repr = self.text_proj(text_repr)
        text_vec = self.text_proj(text_vec)

        comp_repr, substrate_vec, comp_mask = self.substrate_encoder(
            ligand_atom_features,
            ligand_atom_coords,
            ligand_atom_mask,
            ligand_component_ids,
        )
        substrate_tokens = torch.cat(
            [
                comp_repr,
                substrate_vec.unsqueeze(1),
            ],
            dim=1,
        )
        substrate_mask = torch.cat(
            [
                comp_mask,
                torch.ones((comp_mask.size(0), 1), dtype=torch.bool, device=comp_mask.device),
            ],
            dim=1,
        )

        for block in self.stage1_fusion_blocks:
            stage1_protein = block(stage1_protein, substrate_tokens, substrate_mask)

        stage1_protein = stage1_protein * seq_mask.unsqueeze(-1).float()

        stage2_protein = self.stage2_norm(self.stage2_seq_proj(seq_repr))
        text_tokens = torch.cat([text_repr, text_vec.unsqueeze(1)], dim=1)
        text_context_mask = torch.cat(
            [
                text_mask,
                torch.ones((text_mask.size(0), 1), dtype=torch.bool, device=text_mask.device),
            ],
            dim=1,
        )
        for block in self.stage2_fusion_blocks:
            stage2_protein = block(stage2_protein, text_tokens, text_context_mask)

        stage2_protein = stage2_protein * seq_mask.unsqueeze(-1).float()

        site_logits = self.site_head(stage1_protein).squeeze(-1)
        aa_logits = self.aa_head(stage2_protein)
        return {
            "site_logits": site_logits,
            "aa_logits": aa_logits,
            "stage1_protein_repr": stage1_protein,
            "stage2_protein_repr": stage2_protein,
            "substrate_context_repr": substrate_tokens,
            "text_context_repr": text_tokens,
            "seq_mask": seq_mask,
            "substrate_mask": substrate_mask,
            "text_mask": text_context_mask,
        }
