from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .utils import split_components

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None
    AllChem = None
    RDLogger = None


@dataclass
class Ligand3DResult:
    atom_features: List[List[float]]
    atom_coords: List[List[float]]
    atom_component_ids: List[int]
    component_sizes: List[int]
    status: str
    error: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "substrate_atom_features": self.atom_features,
            "substrate_atom_coords": self.atom_coords,
            "substrate_atom_component_ids": self.atom_component_ids,
            "substrate_component_sizes": self.component_sizes,
            "substrate_3d_status": self.status,
            "substrate_3d_error": self.error,
        }


def rdkit_available() -> bool:
    return Chem is not None and AllChem is not None


if RDLogger is not None:  # pragma: no cover - cosmetic runtime behavior
    RDLogger.DisableLog("rdApp.*")


def atom_feature_vector(atom) -> List[float]:
    hybridization = int(atom.GetHybridization())
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs(includeNeighbors=True)),
        float(hybridization if hybridization >= 0 else 0),
    ]


def _embed_single_component(component_smiles: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(component_smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {component_smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = False
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise ValueError(f"RDKit failed to embed 3D conformer: {component_smiles}")

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    coords = np.asarray(conf.GetPositions(), dtype=np.float32)
    feats = np.asarray([atom_feature_vector(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    return feats, coords


def build_ligand_3d(substrate_smiles: str | None, seed: int = 42) -> Ligand3DResult:
    components = split_components(substrate_smiles)
    if not components:
        return Ligand3DResult([], [], [], [], "missing_smiles", "")
    if not rdkit_available():
        return Ligand3DResult([], [], [], [], "rdkit_unavailable", "RDKit is not installed.")

    all_features: List[np.ndarray] = []
    all_coords: List[np.ndarray] = []
    all_component_ids: List[int] = []
    component_sizes: List[int] = []

    try:
        for component_idx, component in enumerate(components):
            feats, coords = _embed_single_component(component, seed + component_idx)
            if coords.size == 0:
                continue
            all_features.append(feats)
            all_coords.append(coords)
            all_component_ids.extend([component_idx] * int(coords.shape[0]))
            component_sizes.append(int(coords.shape[0]))
    except Exception as exc:
        return Ligand3DResult([], [], [], [], "failed", str(exc))

    if not all_coords:
        return Ligand3DResult([], [], [], [], "empty_atoms", "No atoms retained after RDKit processing.")

    merged_features = np.concatenate(all_features, axis=0)
    merged_coords = np.concatenate(all_coords, axis=0)
    merged_coords = merged_coords - merged_coords.mean(axis=0, keepdims=True)

    return Ligand3DResult(
        atom_features=merged_features.astype(np.float32).tolist(),
        atom_coords=merged_coords.astype(np.float32).tolist(),
        atom_component_ids=all_component_ids,
        component_sizes=component_sizes,
        status="ok",
        error="",
    )
