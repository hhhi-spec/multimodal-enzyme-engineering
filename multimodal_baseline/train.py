from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from .data import HFTokenizerBundle, MultimodalManifestDataset, collate_multimodal, load_vocab_bundle
from .model import MultimodalEnzymeBaseline
from .utils import load_jsonl


def log_line(message: str) -> None:
    print(message, flush=True)


def compute_site_pos_weight(rows: list[dict]) -> float:
    positives = 0
    total = 0
    for row in rows:
        seq_len = len(row.get("parent_sequence", ""))
        total += seq_len
        positives += len(row.get("mutation_sites") or [])
    if positives <= 0:
        return 1.0
    return float(max((total - positives) / positives, 1.0))


def check_split_leakage(rows: list[dict]) -> None:
    train_exps = {row["experiment_id"] for row in rows if row.get("split") == "train"}
    test_exps = {row["experiment_id"] for row in rows if row.get("split") == "test"}
    overlap = train_exps & test_exps
    if overlap:
        raise RuntimeError(f"Split leakage detected: {sorted(overlap)}")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def smooth_binary_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0.0:
        return targets
    smoothing = float(max(0.0, min(1.0, smoothing)))
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def focal_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal_factor = (1.0 - p_t).clamp_min(0.0).pow(gamma)
    return focal_factor * bce


def forward_loss(
    model,
    batch,
    site_pos_weight: float,
    site_loss_weight: float = 1.0,
    site_label_smoothing: float = 0.0,
    aa_label_smoothing: float = 0.0,
    site_loss_type: str = "bce",
    site_focal_gamma: float = 2.0,
):
    outputs = model(
        parent_sequences=batch["parent_sequence_raw"],
        coords=batch["coords"],
        coord_mask=batch["coord_mask"],
        text_ids=batch["text_ids"],
        text_attention_mask=batch["text_attention_mask"],
        ligand_atom_features=batch["ligand_atom_features"],
        ligand_atom_coords=batch["ligand_atom_coords"],
        ligand_atom_mask=batch["ligand_atom_mask"],
        ligand_component_ids=batch["ligand_component_ids"],
    )
    residue_mask = batch["residue_mask"]
    site_targets = batch["site_mask"].float()
    site_logits = outputs["site_logits"]
    pos_weight = torch.tensor(site_pos_weight, device=site_logits.device)
    if site_loss_type == "focal":
        site_loss = focal_binary_cross_entropy_with_logits(
            site_logits,
            site_targets,
            pos_weight=pos_weight,
            gamma=site_focal_gamma,
        )
    else:
        smoothed_site_targets = smooth_binary_targets(site_targets, site_label_smoothing)
        site_loss = F.binary_cross_entropy_with_logits(
            site_logits,
            smoothed_site_targets,
            pos_weight=pos_weight,
            reduction="none",
        )
    site_loss = (site_loss * residue_mask.float()).sum() / residue_mask.float().sum().clamp_min(1.0)

    aa_target = batch["aa_target"]
    aa_logits = outputs["aa_logits"]
    aa_valid = aa_target.ne(-100) & residue_mask
    if aa_valid.any():
        aa_loss = F.cross_entropy(
            aa_logits[aa_valid],
            aa_target[aa_valid],
            label_smoothing=float(max(0.0, min(1.0, aa_label_smoothing))),
        )
    else:
        aa_loss = aa_logits.sum() * 0.0

    loss = site_loss * float(site_loss_weight) + aa_loss
    return loss, site_loss.detach(), aa_loss.detach(), outputs


def compute_eval_metrics(
    site_probs: torch.Tensor,
    site_true: torch.Tensor,
    aa_pred: torch.Tensor,
    aa_true: torch.Tensor,
    aa_top3: torch.Tensor,
    site_threshold: float = 0.5,
) -> dict:
    site_pred = site_probs.ge(site_threshold).long()

    if site_true.numel() > 0:
        site_f1 = float(f1_score(site_true.numpy(), site_pred.numpy(), zero_division=0))
        site_recall = float(recall_score(site_true.numpy(), site_pred.numpy(), zero_division=0))
        if int(site_true.min().item()) != int(site_true.max().item()):
            site_auc = float(roc_auc_score(site_true.numpy(), site_probs.numpy()))
        else:
            site_auc = 0.0
    else:
        site_f1 = 0.0
        site_recall = 0.0
        site_auc = 0.0

    if aa_true.numel() > 0:
        aa_top1 = float(aa_pred.eq(aa_true).float().mean().item())
        aa_top3_value = float(aa_top3.float().mean().item())
    else:
        aa_top1 = 0.0
        aa_top3_value = 0.0

    return {
        "site_f1": site_f1,
        "site_recall": site_recall,
        "site_auc": site_auc,
        "aa_top1": aa_top1,
        "aa_top3": aa_top3_value,
    }


def search_best_site_threshold(
    site_probs: torch.Tensor,
    site_true: torch.Tensor,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    step: float = 0.01,
) -> tuple[float, dict]:
    if site_probs.numel() == 0 or site_true.numel() == 0:
        return 0.5, {"site_f1": 0.0, "site_recall": 0.0, "site_auc": 0.0}

    thresholds = torch.arange(min_threshold, max_threshold + 1e-9, step, device=site_probs.device)
    true = site_true.long().unsqueeze(0)
    pred = site_probs.unsqueeze(0) >= thresholds.unsqueeze(1)
    tp = (pred & true.bool()).sum(dim=1).float()
    fp = (pred & (~true.bool())).sum(dim=1).float()
    fn = ((~pred) & true.bool()).sum(dim=1).float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = torch.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall).clamp_min(1e-8),
        torch.zeros_like(precision),
    )
    best_idx = int(torch.argmax(f1).item())
    best_threshold = float(thresholds[best_idx].item())
    best_metrics = {
        "site_f1": float(f1[best_idx].item()),
        "site_recall": float(recall[best_idx].item()),
        "site_auc": 0.0,
    }
    return best_threshold, best_metrics


def evaluate(
    model,
    loader,
    device,
    site_pos_weight: float,
    site_loss_weight: float,
    site_threshold: float = 0.5,
    site_loss_type: str = "bce",
    site_focal_gamma: float = 2.0,
    return_raw: bool = False,
):
    model.eval()
    losses = []
    all_site_probs = []
    all_site_true = []
    all_aa_pred = []
    all_aa_true = []
    all_aa_top3 = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            loss, site_loss, aa_loss, outputs = forward_loss(
                model,
                batch,
                site_pos_weight,
                site_loss_weight=site_loss_weight,
                site_label_smoothing=0.0,
                aa_label_smoothing=0.0,
                site_loss_type=site_loss_type,
                site_focal_gamma=site_focal_gamma,
            )
            losses.append((loss.item(), site_loss.item(), aa_loss.item()))
            site_valid = batch["residue_mask"].bool()
            all_site_probs.append(torch.sigmoid(outputs["site_logits"][site_valid]).detach().cpu())
            all_site_true.append(batch["site_mask"][site_valid].detach().cpu().long())

            aa_valid = batch["aa_target"].ne(-100) & batch["residue_mask"]
            if aa_valid.any():
                aa_logits = outputs["aa_logits"][aa_valid].detach().cpu()
                aa_true_batch = batch["aa_target"][aa_valid].detach().cpu()
                all_aa_pred.append(aa_logits.argmax(dim=-1))
                all_aa_true.append(aa_true_batch)
                top3 = aa_logits.topk(k=min(3, aa_logits.size(-1)), dim=-1).indices
                all_aa_top3.append(top3.eq(aa_true_batch.unsqueeze(-1)).any(dim=-1))
    if not losses:
        return {
            "loss": 0.0,
            "site_loss": 0.0,
            "aa_loss": 0.0,
            "site_f1": 0.0,
            "site_recall": 0.0,
            "site_auc": 0.0,
            "aa_top1": 0.0,
            "aa_top3": 0.0,
        }
    arr = torch.tensor(losses)
    site_probs = torch.cat(all_site_probs, dim=0) if all_site_probs else torch.zeros((0,), dtype=torch.float32)
    site_true = torch.cat(all_site_true, dim=0) if all_site_true else torch.zeros((0,), dtype=torch.long)
    aa_pred = torch.cat(all_aa_pred, dim=0) if all_aa_pred else torch.zeros((0,), dtype=torch.long)
    aa_true = torch.cat(all_aa_true, dim=0) if all_aa_true else torch.zeros((0,), dtype=torch.long)
    aa_top3 = torch.cat(all_aa_top3, dim=0) if all_aa_top3 else torch.zeros((0,), dtype=torch.bool)
    metric_values = compute_eval_metrics(
        site_probs=site_probs,
        site_true=site_true,
        aa_pred=aa_pred,
        aa_true=aa_true,
        aa_top3=aa_top3,
        site_threshold=site_threshold,
    )
    metrics = {
        "loss": float(arr[:, 0].mean().item()),
        "site_loss": float(arr[:, 1].mean().item()),
        "aa_loss": float(arr[:, 2].mean().item()),
        "site_f1": metric_values["site_f1"],
        "site_recall": metric_values["site_recall"],
        "site_auc": metric_values["site_auc"],
        "aa_top1": metric_values["aa_top1"],
        "aa_top3": metric_values["aa_top3"],
    }
    if return_raw:
        metrics["site_probs"] = site_probs
        metrics["site_true"] = site_true
    return metrics


def split_runtime_rows_by_experiment(rows: list[dict], val_ratio: float, seed: int) -> list[dict]:
    if val_ratio <= 0:
        return [copy.deepcopy(row) for row in rows]
    rng = random.Random(seed)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "train":
            continue
        grouped[int(row.get("ec_major") or 0)].append(row)

    val_experiments: set[str] = set()
    for ec_major, ec_rows in grouped.items():
        exp_to_rows: dict[str, list[dict]] = defaultdict(list)
        for row in ec_rows:
            exp_to_rows[str(row.get("experiment_id"))].append(row)
        exp_items = list(exp_to_rows.items())
        rng.shuffle(exp_items)
        total = len(ec_rows)
        target_val = max(1, int(round(total * val_ratio))) if total > 1 else 0
        running = 0
        for experiment_id, exp_rows in exp_items:
            if running >= target_val:
                break
            val_experiments.add(experiment_id)
            running += len(exp_rows)

    runtime_rows = []
    for row in rows:
        new_row = copy.deepcopy(row)
        if new_row.get("split") == "train" and str(new_row.get("experiment_id")) in val_experiments:
            new_row["split"] = "val"
        runtime_rows.append(new_row)
    return runtime_rows


def load_tokenizer_if_available(path_str: str | None):
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        log_line(f"tokenizer_path_not_found={path}")
        return None
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def main():
    parser = argparse.ArgumentParser(description="Train the multimodal enzyme baseline.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("multimodal_baseline_artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overfit-n", type=int, default=0)
    parser.add_argument("--save-path", type=Path, default=Path("multimodal_baseline_artifacts/model.pt"))
    parser.add_argument("--seq-pretrained-dir", type=str, default=None)
    parser.add_argument("--text-pretrained-dir", type=str, default=None)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--seq-hidden-size", type=int, default=256)
    parser.add_argument("--text-hidden-size", type=int, default=256)
    parser.add_argument("--site-label-smoothing", type=float, default=0.0)
    parser.add_argument("--aa-label-smoothing", type=float, default=0.03)
    parser.add_argument("--site-loss-weight", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--freeze-seq-backbone", action="store_true")
    parser.add_argument("--freeze-text-backbone", action="store_true")
    parser.add_argument("--site-loss-type", type=str, choices=["bce", "focal"], default="bce")
    parser.add_argument("--site-focal-gamma", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    manifest_path = args.artifacts_dir / "manifest.jsonl"
    vocab_path = args.artifacts_dir / "vocab.json"
    rows = load_jsonl(manifest_path)
    check_split_leakage(rows)
    runtime_rows = split_runtime_rows_by_experiment(rows, val_ratio=args.val_ratio, seed=42)
    vocab = load_vocab_bundle(vocab_path)

    train_ds = MultimodalManifestDataset(
        manifest_path=manifest_path,
        vocab_path=vocab_path,
        split="train",
        rows=runtime_rows,
        train_text_augmentation=True,
    )
    val_ds = MultimodalManifestDataset(
        manifest_path=manifest_path,
        vocab_path=vocab_path,
        split="val",
        rows=runtime_rows,
        train_text_augmentation=False,
    )
    test_ds = MultimodalManifestDataset(
        manifest_path=manifest_path,
        vocab_path=vocab_path,
        split="test",
        rows=runtime_rows,
        train_text_augmentation=False,
    )
    if args.overfit_n > 0:
        keep = min(args.overfit_n, len(train_ds))
        train_ds = Subset(train_ds, list(range(keep)))

    site_pos_weight = compute_site_pos_weight([row for row in runtime_rows if row.get("split") == "train"])
    log_line(
        f"train_samples={len(train_ds)} val_samples={len(val_ds)} test_samples={len(test_ds)} "
        f"site_pos_weight={site_pos_weight:.3f}"
    )

    tokenizer_bundle = HFTokenizerBundle(
        seq_tokenizer=load_tokenizer_if_available(args.seq_pretrained_dir),
        text_tokenizer=load_tokenizer_if_available(args.text_pretrained_dir),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_multimodal(
            batch,
            vocab.seq_vocab.pad_id,
            vocab.text_vocab.pad_id,
            vocab.smiles_vocab.pad_id,
            tokenizer_bundle=tokenizer_bundle,
        ),
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

    device = torch.device(args.device)
    model = MultimodalEnzymeBaseline(
        vocab,
        fusion_dim=args.fusion_dim,
        seq_hidden_size=args.seq_hidden_size,
        text_hidden_size=args.text_hidden_size,
        fusion_layers=args.fusion_layers,
        seq_pretrained_dir=args.seq_pretrained_dir,
        text_pretrained_dir=args.text_pretrained_dir,
        freeze_seq_backbone=args.freeze_seq_backbone,
        freeze_text_backbone=args.freeze_text_backbone,
    ).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    first_batch = next(iter(train_loader), None)
    if first_batch is None:
        raise RuntimeError("No training data found.")
    first_batch = move_batch_to_device(first_batch, device)
    loss, site_loss, aa_loss, outputs = forward_loss(
        model,
        first_batch,
        site_pos_weight,
        site_loss_weight=args.site_loss_weight,
        site_label_smoothing=args.site_label_smoothing,
        aa_label_smoothing=args.aa_label_smoothing,
        site_loss_type=args.site_loss_type,
        site_focal_gamma=args.site_focal_gamma,
    )
    log_line("forward_shapes")
    log_line(f"parent_sequences={len(first_batch['parent_sequence_raw'])}")
    log_line(f"seq_ids={tuple(first_batch['seq_ids'].shape)}")
    log_line(f"coords={tuple(first_batch['coords'].shape)}")
    log_line(f"text_ids={tuple(first_batch['text_ids'].shape)}")
    log_line(f"ligand_atom_features={tuple(first_batch['ligand_atom_features'].shape)}")
    log_line(f"ligand_atom_coords={tuple(first_batch['ligand_atom_coords'].shape)}")
    log_line(f"site_logits={tuple(outputs['site_logits'].shape)}")
    log_line(f"aa_logits={tuple(outputs['aa_logits'].shape)}")
    log_line(f"dry_forward_loss={loss.item():.4f} site={site_loss.item():.4f} aa={aa_loss.item():.4f}")

    if args.dry_run:
        return

    best_val_site_f1 = None
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            loss, site_loss, aa_loss, _ = forward_loss(
                model,
                batch,
                site_pos_weight,
                site_loss_weight=args.site_loss_weight,
                site_label_smoothing=args.site_label_smoothing,
                aa_label_smoothing=args.aa_label_smoothing,
                site_loss_type=args.site_loss_type,
                site_focal_gamma=args.site_focal_gamma,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running.append((loss.item(), site_loss.item(), aa_loss.item()))
            if batch_idx == 1 or batch_idx % max(1, args.log_every) == 0:
                log_line(
                    f"epoch={epoch} step={batch_idx}/{len(train_loader)} "
                    f"batch_loss={loss.item():.4f} batch_site={site_loss.item():.4f} batch_aa={aa_loss.item():.4f}"
                )
        train_metrics = torch.tensor(running).mean(dim=0) if running else torch.zeros(3)
        val_metrics_raw = evaluate(
            model,
            val_loader,
            device,
            site_pos_weight,
            args.site_loss_weight,
            site_threshold=0.5,
            site_loss_type=args.site_loss_type,
            site_focal_gamma=args.site_focal_gamma,
            return_raw=True,
        )
        best_threshold, _ = search_best_site_threshold(
            val_metrics_raw["site_probs"],
            val_metrics_raw["site_true"],
        )
        val_metrics = {k: v for k, v in val_metrics_raw.items() if k not in {"site_probs", "site_true"}}
        site_only_metrics = compute_eval_metrics(
            site_probs=val_metrics_raw["site_probs"],
            site_true=val_metrics_raw["site_true"],
            aa_pred=torch.zeros((0,), dtype=torch.long),
            aa_true=torch.zeros((0,), dtype=torch.long),
            aa_top3=torch.zeros((0,), dtype=torch.bool),
            site_threshold=best_threshold,
        )
        val_metrics["site_f1"] = site_only_metrics["site_f1"]
        val_metrics["site_recall"] = site_only_metrics["site_recall"]
        val_metrics["site_auc"] = site_only_metrics["site_auc"]
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            site_pos_weight,
            args.site_loss_weight,
            site_threshold=best_threshold,
            site_loss_type=args.site_loss_type,
            site_focal_gamma=args.site_focal_gamma,
        )
        log_line(
            f"epoch={epoch} "
            f"train_loss={train_metrics[0]:.4f} train_site={train_metrics[1]:.4f} train_aa={train_metrics[2]:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_site={val_metrics['site_loss']:.4f} val_aa={val_metrics['aa_loss']:.4f} "
            f"best_site_th={best_threshold:.2f} "
            f"test_loss={test_metrics['loss']:.4f} test_site={test_metrics['site_loss']:.4f} test_aa={test_metrics['aa_loss']:.4f} "
            f"site_f1={test_metrics['site_f1']:.4f} site_recall={test_metrics['site_recall']:.4f} "
            f"site_auc={test_metrics['site_auc']:.4f} aa_top1={test_metrics['aa_top1']:.4f} aa_top3={test_metrics['aa_top3']:.4f}"
        )
        if best_val_site_f1 is None or val_metrics["site_f1"] > best_val_site_f1:
            best_val_site_f1 = val_metrics["site_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab.to_json(),
                    "epoch": epoch,
                    "best_val_site_f1": best_val_site_f1,
                    "val_loss": val_metrics["loss"],
                    "test_loss": test_metrics["loss"],
                    "best_site_threshold": best_threshold,
                    "model_config": {
                        "fusion_dim": args.fusion_dim,
                        "seq_hidden_size": args.seq_hidden_size,
                        "text_hidden_size": args.text_hidden_size,
                        "fusion_layers": args.fusion_layers,
                        "seq_pretrained_dir": args.seq_pretrained_dir,
                        "text_pretrained_dir": args.text_pretrained_dir,
                        "freeze_seq_backbone": args.freeze_seq_backbone,
                        "freeze_text_backbone": args.freeze_text_backbone,
                        "site_loss_type": args.site_loss_type,
                        "site_focal_gamma": args.site_focal_gamma,
                    },
                },
                args.save_path,
            )


if __name__ == "__main__":
    main()
