from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors


def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    result_shape = (num_segments, data.size(1))
    result = data.new_zeros(result_shape)
    expanded_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, expanded_ids, data)
    return result


class E_GCL(nn.Module):
    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        edges_in_d: int = 0,
        act_fn: nn.Module = nn.SiLU(),
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        coords_agg: str = "sum",
        tanh: bool = False,
    ):
        super().__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + 1 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1))

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def edge_model(self, source, target, radial, edge_attr=None):
        out = torch.cat([source, target, radial], dim=1)
        out = self.edge_mlp(out.float())
        if self.attention:
            out = out * torch.sigmoid(self.att_mlp(out))
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr=None):
        row, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, _ = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        else:
            raise ValueError(f"Unsupported coords_agg: {self.coords_agg}")
        coord = coord + agg
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None):
        if edge_index[0].numel() == 0:
            return h, coord, edge_attr
        radial, coord_diff = self.coord2radial(edge_index, coord)
        row, col = edge_index
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr=edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h = self.node_model(h, edge_index, edge_feat)
        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int,
        out_node_nf: int,
        in_edge_nf: int = 0,
        act_fn: nn.Module = nn.SiLU(),
        n_layers: int = 3,
        residual: bool = True,
        attention: bool = True,
        normalize: bool = False,
        tanh: bool = False,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [
                E_GCL(
                    hidden_nf,
                    hidden_nf,
                    hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                    coords_agg="sum",
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor, edges, edge_attr=None):
        for layer in self.layers:
            h, x, _ = layer(h, edges, x, edge_attr=edge_attr)
        return h, x


class SubstrateEGNN(nn.Module):
    def __init__(
        self,
        in_node_nf: int = 5,
        hidden_nf: int = 256,
        out_node_nf: int = 256,
        in_edge_nf: int = 0,
        act_fn: nn.Module = nn.SiLU(),
        n_layers: int = 3,
        residual: bool = True,
        attention: bool = True,
        normalize: bool = False,
        tanh: bool = False,
    ):
        super().__init__()
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        self.egnn = EGNN(
            in_node_nf=hidden_nf,
            hidden_nf=hidden_nf,
            out_node_nf=out_node_nf,
            in_edge_nf=in_edge_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            residual=residual,
            attention=attention,
            normalize=normalize,
            tanh=tanh,
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor, edges, edge_attr=None):
        h = self.embedding_in(h.float())
        return self.egnn(h, x, edges, edge_attr=edge_attr)


def build_batch_knn_graph(
    coords: torch.Tensor,
    mask: torch.Tensor,
    k: int = 30,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor], List[int]]:
    device = coords.device
    flat_coords: List[torch.Tensor] = []
    rows: List[int] = []
    cols: List[int] = []
    valid_indices_per_sample: List[torch.Tensor] = []
    offsets: List[int] = []
    offset = 0

    for batch_idx in range(coords.size(0)):
        valid_indices = mask[batch_idx].nonzero(as_tuple=False).flatten()
        valid_indices_per_sample.append(valid_indices)
        offsets.append(offset)
        if valid_indices.numel() == 0:
            continue
        sample_coords = coords[batch_idx, valid_indices]
        flat_coords.append(sample_coords)
        if valid_indices.numel() > 1:
            effective_k = min(k, valid_indices.numel() - 1)
            nbrs = NearestNeighbors(n_neighbors=effective_k + 1, algorithm="ball_tree").fit(sample_coords.detach().cpu().numpy())
            _, nn_indices = nbrs.kneighbors(sample_coords.detach().cpu().numpy())
            for src_local in range(valid_indices.numel()):
                for dst_local in nn_indices[src_local][1:]:
                    rows.append(offset + src_local)
                    cols.append(offset + int(dst_local))
        offset += valid_indices.numel()

    if flat_coords:
        flat_coord_tensor = torch.cat(flat_coords, dim=0).to(device)
    else:
        flat_coord_tensor = torch.zeros((0, 3), dtype=coords.dtype, device=device)
    edge_index = (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(cols, dtype=torch.long, device=device),
    )
    return flat_coord_tensor, edge_index, valid_indices_per_sample, offsets


def build_grouped_batch_knn_graph(
    coords: torch.Tensor,
    mask: torch.Tensor,
    group_ids: torch.Tensor,
    k: int = 16,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor], List[int]]:
    device = coords.device
    flat_coords: List[torch.Tensor] = []
    rows: List[int] = []
    cols: List[int] = []
    valid_indices_per_sample: List[torch.Tensor] = []
    offsets: List[int] = []
    offset = 0

    for batch_idx in range(coords.size(0)):
        valid_indices = mask[batch_idx].nonzero(as_tuple=False).flatten()
        valid_indices_per_sample.append(valid_indices)
        offsets.append(offset)
        if valid_indices.numel() == 0:
            continue

        sample_coords = coords[batch_idx, valid_indices]
        sample_groups = group_ids[batch_idx, valid_indices]
        flat_coords.append(sample_coords)

        unique_groups = [int(x) for x in sample_groups.unique(sorted=True).tolist() if int(x) >= 0]
        local_start = 0
        for group_id in unique_groups:
            group_local = (sample_groups == group_id).nonzero(as_tuple=False).flatten()
            if group_local.numel() <= 1:
                continue
            group_coords = sample_coords[group_local]
            effective_k = min(k, group_local.numel() - 1)
            nbrs = NearestNeighbors(n_neighbors=effective_k + 1, algorithm="ball_tree").fit(
                group_coords.detach().cpu().numpy()
            )
            _, nn_indices = nbrs.kneighbors(group_coords.detach().cpu().numpy())
            for src_idx, src_local in enumerate(group_local.tolist()):
                for dst_inner in nn_indices[src_idx][1:]:
                    dst_local = int(group_local[int(dst_inner)].item())
                    rows.append(offset + src_local)
                    cols.append(offset + dst_local)
            local_start += group_local.numel()
        offset += valid_indices.numel()

    if flat_coords:
        flat_coord_tensor = torch.cat(flat_coords, dim=0).to(device)
    else:
        flat_coord_tensor = torch.zeros((0, 3), dtype=coords.dtype, device=device)
    edge_index = (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(cols, dtype=torch.long, device=device),
    )
    return flat_coord_tensor, edge_index, valid_indices_per_sample, offsets
