#!/usr/bin/env bash
set -euo pipefail

# Example server-side launch template for AutoDL / Linux GPU box.
# Adjust CUDA device, batch size, and save path to your environment.

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

/root/autodl-tmp/conda-envs/enzyme-mm-clone/bin/python -u -m multimodal_baseline.train \
  --artifacts-dir multimodal_baseline_artifacts \
  --device cuda \
  --epochs 15 \
  --batch-size 1 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --fusion-dim 256 \
  --fusion-layers 4 \
  --seq-hidden-size 1280 \
  --text-hidden-size 768 \
  --seq-pretrained-dir local_models/esm2_t33_650M \
  --text-pretrained-dir local_models/biobert-v1.1 \
  --freeze-seq-backbone \
  --freeze-text-backbone \
  --site-loss-type focal \
  --site-focal-gamma 2.0 \
  --site-label-smoothing 0.0 \
  --aa-label-smoothing 0.03 \
  --site-loss-weight 1.5 \
  --log-every 5 \
  --save-path multimodal_baseline_artifacts/model_server_frozen_focal.pt

# Second pass to try after the first run:
#   --site-loss-weight 1.5
