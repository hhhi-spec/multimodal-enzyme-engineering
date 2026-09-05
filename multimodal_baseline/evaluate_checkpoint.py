from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import HFTokenizerBundle, MultimodalManifestDataset, collate_multimodal, load_vocab_bundle
from .model import MultimodalEnzymeBaseline
from .train import (
    check_split_leakage,
    compute_site_pos_weight,
    evaluate,
    load_tokenizer_if_available,
    log_line,
    search_best_site_threshold,
    split_runtime_rows_by_experiment,
)
from .utils import load_jsonl


def resolve_vocab_path(args: argparse.Namespace, checkpoint: dict) -> Path:
    vocab_path = args.artifacts_dir / "vocab.json"
    if vocab_path.exists():
        return vocab_path
    vocab_data = checkpoint.get("vocab")
    if vocab_data is None:
        raise FileNotFoundError(f"Missing vocab.json in {args.artifacts_dir} and checkpoint does not contain vocab")
    temp_vocab_path = args.artifacts_dir / ".eval_checkpoint_vocab.json"
    temp_vocab_path.write_text(json.dumps(vocab_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return temp_vocab_path


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if "model_state" not in checkpoint and "state_dict" in checkpoint:
        checkpoint["model_state"] = checkpoint["state_dict"]
    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state or state_dict")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved multimodal enzyme checkpoint.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("multimodal_baseline_artifacts"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("multimodal_baseline_artifacts/model_server.pt"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-pretrained-dir", type=str, default="local_models/esm2_t33_650M")
    parser.add_argument("--text-pretrained-dir", type=str, default="local_models/biobert-v1.1")
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--seq-hidden-size", type=int, default=1280)
    parser.add_argument("--text-hidden-size", type=int, default=768)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--freeze-seq-backbone", action="store_true")
    parser.add_argument("--freeze-text-backbone", action="store_true")
    parser.add_argument("--site-loss-type", type=str, choices=["bce", "focal"], default="bce")
    parser.add_argument("--site-focal-gamma", type=float, default=2.0)
    parser.add_argument("--site-loss-weight", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint_path, device=device)

    manifest_path = args.artifacts_dir / "manifest.jsonl"
    rows = load_jsonl(manifest_path)
    check_split_leakage(rows)
    runtime_rows = split_runtime_rows_by_experiment(rows, val_ratio=0.1, seed=42)
    site_pos_weight = compute_site_pos_weight([row for row in runtime_rows if row.get("split") == "train"])

    vocab_path = resolve_vocab_path(args, checkpoint)
    vocab = load_vocab_bundle(vocab_path)
    model_config = dict(checkpoint.get("model_config") or {})
    fusion_dim = int(model_config.get("fusion_dim", args.fusion_dim))
    seq_hidden_size = int(model_config.get("seq_hidden_size", args.seq_hidden_size))
    text_hidden_size = int(model_config.get("text_hidden_size", args.text_hidden_size))
    fusion_layers = int(model_config.get("fusion_layers", args.fusion_layers))
    seq_pretrained_dir = model_config.get("seq_pretrained_dir", args.seq_pretrained_dir)
    text_pretrained_dir = model_config.get("text_pretrained_dir", args.text_pretrained_dir)
    freeze_seq_backbone = bool(model_config.get("freeze_seq_backbone", args.freeze_seq_backbone))
    freeze_text_backbone = bool(model_config.get("freeze_text_backbone", args.freeze_text_backbone))
    site_loss_type = str(model_config.get("site_loss_type", args.site_loss_type))
    site_focal_gamma = float(model_config.get("site_focal_gamma", args.site_focal_gamma))

    test_ds = MultimodalManifestDataset(
        manifest_path=manifest_path,
        vocab_path=vocab_path,
        split="test",
        rows=runtime_rows,
        train_text_augmentation=False,
    )
    val_ds = MultimodalManifestDataset(
        manifest_path=manifest_path,
        vocab_path=vocab_path,
        split="val",
        rows=runtime_rows,
        train_text_augmentation=False,
    )
    tokenizer_bundle = HFTokenizerBundle(
        seq_tokenizer=load_tokenizer_if_available(args.seq_pretrained_dir),
        text_tokenizer=load_tokenizer_if_available(args.text_pretrained_dir),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_multimodal(
            batch,
            vocab.seq_vocab.pad_id,
            vocab.text_vocab.pad_id,
            vocab.smiles_vocab.pad_id,
            tokenizer_bundle=tokenizer_bundle,
        ),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_multimodal(
            batch,
            vocab.seq_vocab.pad_id,
            vocab.text_vocab.pad_id,
            vocab.smiles_vocab.pad_id,
            tokenizer_bundle=tokenizer_bundle,
        ),
    )

    model = MultimodalEnzymeBaseline(
        vocab,
        fusion_dim=fusion_dim,
        seq_hidden_size=seq_hidden_size,
        text_hidden_size=text_hidden_size,
        fusion_layers=fusion_layers,
        seq_pretrained_dir=seq_pretrained_dir,
        text_pretrained_dir=text_pretrained_dir,
        freeze_seq_backbone=freeze_seq_backbone,
        freeze_text_backbone=freeze_text_backbone,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    site_threshold = float(checkpoint.get("best_site_threshold") or 0.5)
    if "best_site_threshold" not in checkpoint:
        val_raw = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            site_pos_weight=site_pos_weight,
            site_loss_weight=args.site_loss_weight,
            site_threshold=0.5,
            site_loss_type=site_loss_type,
            site_focal_gamma=site_focal_gamma,
            return_raw=True,
        )
        site_threshold, _ = search_best_site_threshold(val_raw["site_probs"], val_raw["site_true"])

    metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        site_pos_weight=site_pos_weight,
        site_loss_weight=args.site_loss_weight,
        site_threshold=site_threshold,
        site_loss_type=site_loss_type,
        site_focal_gamma=site_focal_gamma,
    )

    log_line(f"checkpoint={args.checkpoint_path}")
    log_line(f"test_samples={len(test_ds)} site_pos_weight={site_pos_weight:.3f}")
    log_line(f"site_threshold={site_threshold:.2f}")
    log_line(
        "metrics "
        f"loss={metrics['loss']:.4f} site_loss={metrics['site_loss']:.4f} aa_loss={metrics['aa_loss']:.4f} "
        f"site_f1={metrics['site_f1']:.4f} site_recall={metrics['site_recall']:.4f} site_auc={metrics['site_auc']:.4f} "
        f"aa_top1={metrics['aa_top1']:.4f} aa_top3={metrics['aa_top3']:.4f}"
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint_path),
                    "site_pos_weight": site_pos_weight,
                    "site_threshold": site_threshold,
                    "num_test_samples": len(test_ds),
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
