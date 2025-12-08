"""
Training and evaluation pipeline for the surrogate CNN.

Assumes dataset NPZ files under surrogate_model/data:
    train.npz / val.npz / test.npz
Fields per NPZ:
    - material_matrix: [N, H, W] float32 (0/1)
    - hzfield_state:   [N, M] or [N, 1, H, W] target (here M=num_flux_regions)
    - mode_transmission: [N, 2] float32
    - input_mode: [N] float32

TensorBoard logging is enabled via --log-dir (default surrogate_model/runs).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from surrogate_model.model import SurrogateCNN
from surrogate_model.config import config as surrogate_config
from config import config as main_config


def make_unique_run_dir(base_log_dir: str) -> Path:
    """Create a unique TensorBoard run directory like RUN_1, RUN_2, ..."""
    base = Path(base_log_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = [
        p for p in base.iterdir() if p.is_dir() and p.name.startswith("RUN_") and p.name[4:].isdigit()
    ]
    next_id = 1
    if existing:
        next_id = max(int(p.name[4:]) for p in existing) + 1
    run_dir = base / f"RUN_{next_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@dataclass
class TrainArgs:
    data_dir: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    device: str
    log_dir: str
    ckpt_dir: str
    num_workers: int
    eval_only: bool
    checkpoint: str | None
    hz_weight: float
    mode_weight: float
    input_weight: float


class SurrogateNPZDataset(Dataset):
    def __init__(self, npz_path: Path, stats: Dict[str, Tuple[float, float]] | None = None):
        data = np.load(npz_path)
        self.material = data["material_matrix"].astype(np.float32)  # [N,H,W]
        self.hz = data["hzfield_state"].astype(np.float32)          # [N,M] (num_flux_regions)
        self.mode = data["mode_transmission"].astype(np.float32)    # [N,2]
        self.input_mode = data["input_mode"].astype(np.float32)     # [N]

        # Expand channel dim for CNN input
        self.material = np.expand_dims(self.material, axis=1)       # [N,1,H,W]

        # Stats for normalization (mean, std) only for targets
        if stats is None:
            self.stats = self._compute_stats()
        else:
            self.stats = stats

        self._apply_norm()

    def _compute_stats(self) -> Dict[str, Tuple[float, float]]:
        def ms(x):
            return float(x.mean()), float(x.std() + 1e-8)

        return {
            "hz": ms(self.hz),
            "mode": ms(self.mode),
            "input_mode": ms(self.input_mode),
        }

    def _apply_norm(self) -> None:
        hz_mean, hz_std = self.stats["hz"]
        mode_mean, mode_std = self.stats["mode"]
        input_mean, input_std = self.stats["input_mode"]

        self.hz = (self.hz - hz_mean) / hz_std
        self.mode = (self.mode - mode_mean) / mode_std
        self.input_mode = (self.input_mode - input_mean) / input_std

    def __len__(self) -> int:
        return self.material.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.material[idx]),
            torch.from_numpy(self.hz[idx]),
            torch.from_numpy(self.mode[idx]),
            torch.tensor(self.input_mode[idx], dtype=torch.float32),
        )


def build_datasets(args: TrainArgs):
    data_dir = Path(args.data_dir)
    train_ds = SurrogateNPZDataset(data_dir / "train.npz")
    stats = train_ds.stats
    val_ds = SurrogateNPZDataset(data_dir / "val.npz", stats=stats)
    test_path = data_dir / "test.npz"
    test_ds = SurrogateNPZDataset(test_path, stats=stats) if test_path.exists() else None
    return train_ds, val_ds, test_ds, stats


def make_loaders(train_ds, val_ds, test_ds, args: TrainArgs):
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if test_ds is not None
        else None
    )
    return train_loader, val_loader, test_loader


def compute_losses(pred: Dict[str, torch.Tensor], hz, mode, input_mode, weights) -> Dict[str, torch.Tensor]:
    mse = nn.MSELoss()
    loss_hz = mse(pred["hzfield_state"], hz)
    loss_mode = mse(pred["mode_transmission"], mode)
    loss_input = mse(pred["input_mode"].squeeze(-1), input_mode)
    total = (
        weights["hz"] * loss_hz
        + weights["mode"] * loss_mode
        + weights["input"] * loss_input
    )
    return {
        "total": total,
        "hz": loss_hz,
        "mode": loss_mode,
        "input": loss_input,
    }


def train_one_epoch(model, loader, opt, device, weights):
    model.train()
    running = {"total": 0.0, "hz": 0.0, "mode": 0.0, "input": 0.0}
    for material, hz, mode, input_mode in loader:
        material = material.to(device)
        hz = hz.to(device)
        mode = mode.to(device)
        input_mode = input_mode.to(device)

        opt.zero_grad()
        pred = model(material)
        losses = compute_losses(pred, hz, mode, input_mode, weights)
        losses["total"].backward()
        opt.step()

        bsz = material.size(0)
        for k in running:
            running[k] += losses[k].item() * bsz

    n = len(loader.dataset)
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def evaluate(model, loader, device, weights, stats: Dict[str, Tuple[float, float]] | None = None, return_denorm: bool = False):
    """Evaluate loss; optionally compute denormalized MAE/RMSE/relative error."""
    model.eval()
    running = {"total": 0.0, "hz": 0.0, "mode": 0.0, "input": 0.0}

    denorm_running = None
    if return_denorm and stats is not None:
        denorm_running = {
            "hz": {"abs_sum": 0.0, "sq_sum": 0.0, "rel_sum": 0.0, "count": 0},
            "mode": {"abs_sum": 0.0, "sq_sum": 0.0, "rel_sum": 0.0, "count": 0},
            "input": {"abs_sum": 0.0, "sq_sum": 0.0, "rel_sum": 0.0, "count": 0},
        }

    eps = 1e-8

    def _accumulate_denorm(key: str, pred_t: torch.Tensor, target_t: torch.Tensor):
        stats_key = "input_mode" if key == "input" else key
        mean, std = stats[stats_key]
        pred_denorm = pred_t * std + mean
        target_denorm = target_t * std + mean
        diff = pred_denorm - target_denorm
        denorm_running[key]["abs_sum"] += diff.abs().sum().item()
        denorm_running[key]["sq_sum"] += (diff ** 2).sum().item()
        denorm_running[key]["rel_sum"] += (diff.abs() / (target_denorm.abs() + eps)).sum().item()
        denorm_running[key]["count"] += target_denorm.numel()

    for material, hz, mode, input_mode in loader:
        material = material.to(device)
        hz = hz.to(device)
        mode = mode.to(device)
        input_mode = input_mode.to(device)

        pred = model(material)
        losses = compute_losses(pred, hz, mode, input_mode, weights)

        if denorm_running is not None:
            _accumulate_denorm("hz", pred["hzfield_state"], hz)
            _accumulate_denorm("mode", pred["mode_transmission"], mode)
            _accumulate_denorm("input", pred["input_mode"].squeeze(-1), input_mode)

        bsz = material.size(0)
        for k in running:
            running[k] += losses[k].item() * bsz

    n = len(loader.dataset)
    losses_avg = {k: v / n for k, v in running.items()}

    if denorm_running is None:
        return losses_avg

    denorm_metrics = {}
    for k, vals in denorm_running.items():
        count = max(vals["count"], 1)
        mae = vals["abs_sum"] / count
        rmse = (vals["sq_sum"] / count) ** 0.5
        rel_mae = vals["rel_sum"] / count
        denorm_metrics[k] = {"mae": mae, "rmse": rmse, "rel_mae": rel_mae}

    return losses_avg, denorm_metrics


def save_checkpoint(model, opt, epoch, best_val, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(model, opt, path: Path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    return ckpt


def run(args: TrainArgs):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, stats = build_datasets(args)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, args)

    model = SurrogateCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_path = Path(args.ckpt_dir) / "best.pt"
    start_epoch = 0
    best_val = float("inf")

    if args.checkpoint:
        ckpt = load_checkpoint(model, opt if not args.eval_only else None, Path(args.checkpoint), device)
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_val", float("inf"))

    run_log_dir = make_unique_run_dir(args.log_dir)
    writer = SummaryWriter(log_dir=str(run_log_dir))
    print(f"Logging TensorBoard to {run_log_dir}")

    weights = {"hz": args.hz_weight, "mode": args.mode_weight, "input": args.input_weight}

    if args.eval_only:
        val_metrics, val_denorm = evaluate(model, val_loader, device, weights, stats=stats, return_denorm=True)
        print(f"[Eval-only] Val metrics: {val_metrics}")
        if val_denorm:
            print(f"[Eval-only] Val denorm metrics: {val_denorm}")
        if test_loader is not None:
            test_metrics, test_denorm = evaluate(model, test_loader, device, weights, stats=stats, return_denorm=True)
            print(f"[Eval-only] Test metrics: {test_metrics}")
            if test_denorm:
                print(f"[Eval-only] Test denorm metrics: {test_denorm}")
        return

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(model, train_loader, opt, device, weights)
        val_metrics = evaluate(model, val_loader, device, weights)

        writer.add_scalars("loss/train", train_metrics, epoch)
        writer.add_scalars("loss/val", val_metrics, epoch)

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"train_total={train_metrics['total']:.4f} val_total={val_metrics['total']:.4f}"
        )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(model, opt, epoch + 1, best_val, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")

    # Final eval on test split if present
    if test_loader is not None:
        ckpt = torch.load(ckpt_path, map_location=device) if ckpt_path.exists() else None
        if ckpt:
            model.load_state_dict(ckpt["model"])
        test_metrics = evaluate(model, test_loader, device, weights)
        print(f"Test metrics: {test_metrics}")

    writer.close()


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train surrogate CNN with TensorBoard logging.")

    dcfg = surrogate_config.dataset
    tcfg = surrogate_config.training

    parser.add_argument("--data-dir", type=str, default=dcfg.output_dir)
    parser.add_argument("--batch-size", type=int, default=tcfg.batch_size)
    parser.add_argument("--epochs", type=int, default=tcfg.epochs)
    parser.add_argument("--lr", type=float, default=tcfg.lr)
    parser.add_argument("--weight-decay", type=float, default=tcfg.weight_decay)
    parser.add_argument("--device", type=str, default=tcfg.device)
    parser.add_argument("--log-dir", type=str, default=tcfg.log_dir)
    parser.add_argument("--ckpt-dir", type=str, default=tcfg.ckpt_dir)
    parser.add_argument("--num-workers", type=int, default=tcfg.num_workers)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hz-weight", type=float, default=tcfg.hz_weight, help="Loss weight for hzfield_state")
    parser.add_argument("--mode-weight", type=float, default=tcfg.mode_weight, help="Loss weight for mode_transmission")
    parser.add_argument("--input-weight", type=float, default=tcfg.input_weight, help="Loss weight for input_mode")

    args = parser.parse_args()
    return TrainArgs(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        num_workers=args.num_workers,
        eval_only=args.eval_only,
        checkpoint=args.checkpoint,
        hz_weight=args.hz_weight,
        mode_weight=args.mode_weight,
        input_weight=args.input_weight,
    )


if __name__ == "__main__":
    run(parse_args())

