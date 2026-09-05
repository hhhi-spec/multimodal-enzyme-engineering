# Multimodal Enzyme Baseline

## Build artifacts
```bash
python -m multimodal_baseline.prepare_data --output-dir multimodal_baseline_artifacts
```

This creates:
- `manifest.jsonl`
- `experiment_splits.json`
- `vocab.json`
- `manifest_summary.csv`

The preprocessing step now also:
- splits multi-component substrate SMILES on `.`
- uses `RDKit` to generate one 3D conformer per component
- exports ligand geometry into the manifest as:
  - `substrate_atom_features` with shape `[num_atoms, 5]`
  - `substrate_atom_coords` with shape `[num_atoms, 3]`
  - `substrate_atom_component_ids` for disjoint multi-component graphs

## Dry run
```bash
python -m multimodal_baseline.train --artifacts-dir multimodal_baseline_artifacts --dry-run
```

## Train
```bash
python -m multimodal_baseline.train --artifacts-dir multimodal_baseline_artifacts --epochs 10 --batch-size 2
```

## Server training template
For a Linux GPU server such as AutoDL, you can start from:

```bash
bash multimodal_baseline/train_server_template.sh
```

Equivalent direct command:

```bash
python -m multimodal_baseline.train \
  --artifacts-dir multimodal_baseline_artifacts \
  --device cuda \
  --epochs 20 \
  --batch-size 1 \
  --lr 3e-4 \
  --weight-decay 1e-2 \
  --fusion-dim 256 \
  --seq-hidden-size 1280 \
  --text-hidden-size 768 \
  --seq-pretrained-dir local_models/esm2_t33_650M \
  --text-pretrained-dir local_models/biobert-v1.1 \
  --site-label-smoothing 0.01 \
  --aa-label-smoothing 0.05 \
  --save-path multimodal_baseline_artifacts/model_server.pt
```

## Loss options
The trainer now supports optional label smoothing:

```bash
python -m multimodal_baseline.train \
  --artifacts-dir multimodal_baseline_artifacts \
  --site-label-smoothing 0.01 \
  --aa-label-smoothing 0.05
```

- `aa-label-smoothing` applies to the stage-2 amino-acid classification cross-entropy.
- `site-label-smoothing` applies to the stage-1 mutation-site BCE targets by softening `0/1` labels toward `0.5`.
- Set both to `0.0` for the most direct dry-run and debugging behavior.
- Recommended current setting:
  - `site-label-smoothing=0.01`
  - `aa-label-smoothing=0.05`
- Smoothing affects only the training loss targets.
- Inference should still use raw logits, and downstream metrics should be computed against the original hard labels.

## Optional local pretrained backbones
If you later download local HuggingFace-compatible checkpoints for ESM2 and BioBERT, you can point the trainer to them:

```bash
python -m multimodal_baseline.train \
  --artifacts-dir multimodal_baseline_artifacts \
  --seq-pretrained-dir path\\to\\local_esm2_dir \
  --text-pretrained-dir path\\to\\local_biobert_dir
```

Current code supports:
- `EsmModel` for the parent sequence encoder
- `BertModel` for the direction-text encoder
- EnzyGen2-style EGNN for parent C-alpha structure encoding
- EnzyGen2-style substrate EGNN over RDKit-derived ligand atom features and coordinates
