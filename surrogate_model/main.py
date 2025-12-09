import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from surrogate_model.config import config as surrogate_config
from surrogate_model.model import SurrogateCNN
from surrogate_model.train import SurrogateNPZDataset, SurrogateTrainer
from surrogate_model.data_generation import RLDataCollector

class RLSurrogateModel:
    """Lightweight wrapper for loading a trained surrogate and running inference."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        data_dir: Optional[str] = None,
        device: Optional[str] = None,
        checkpoint_num: int | None = None,
    ):
        cfg_train = surrogate_config.training
        cfg_data = surrogate_config.dataset

        self.data_dir = Path(data_dir) if data_dir is not None else Path(cfg_data.output_dir)
        self.checkpoint_dir = Path(cfg_train.ckpt_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_num = checkpoint_num if checkpoint_num is not None else 0
        self.checkpoint = (
            Path(checkpoint)
            if checkpoint is not None
            else (self.checkpoint_dir / "best.pt")
        )
        self.device = torch.device(
            device
            if device is not None
            else (cfg_train.device if torch.cuda.is_available() else "cpu")
        )

        self.stats = self._load_stats()
        self.model = self._load_model(self.checkpoint)
        self.RL_data_collector = RLDataCollector()

    def _checkpoint_path(self, num: int) -> Path:
        return self.checkpoint_dir / f"checkpoint_{num}.pt"

    def _load_stats(self) -> Dict[str, tuple]:
        """Load normalization stats from training data (mean/std per target)."""
        train_npz = self.data_dir / "train.npz"
        if not train_npz.exists():
            raise FileNotFoundError(f"train.npz not found at {train_npz}")
        train_ds = SurrogateNPZDataset(train_npz)
        return train_ds.stats

    def _load_model(self, ckpt_path: Path) -> SurrogateCNN:
        """Load model weights and switch to eval mode from given checkpoint path."""
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model = SurrogateCNN().to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model

    def predict(
        self,
        material_matrix: np.ndarray,
        use_latest_checkpoint: bool = True,
        checkpoint_num: int | None = None,
    ) -> Dict[str, object]:
        """
        Run inference.

        Args:
            material_matrix: 2D numpy array with shape [H, W], values 0/1.

        Args:
            use_latest_checkpoint: If True, use the latest finetune checkpoint_[num].pt (tracked in-memory).
            checkpoint_num: If provided, load checkpoint_{checkpoint_num}.pt instead.

        Returns:
            Dictionary with denormalized outputs: output_mode_1, output_mode_2, input_mode.
        """
        if material_matrix.ndim != 2:
            raise ValueError(f"material_matrix must be 2D [H, W]; got shape {material_matrix.shape}")

        ckpt_to_use = self.checkpoint
        if checkpoint_num is not None:
            ckpt_to_use = self._checkpoint_path(checkpoint_num)
        elif use_latest_checkpoint and self.checkpoint_num > 0:
            ckpt_to_use = self._checkpoint_path(self.checkpoint_num)

        # Reload model if different checkpoint is requested
        if ckpt_to_use != self.checkpoint:
            self.model = self._load_model(ckpt_to_use)
            self.checkpoint = ckpt_to_use

        material_t = (
            torch.from_numpy(material_matrix.astype(np.float32))
            .unsqueeze(0)  # batch
            .unsqueeze(0)  # channel
            .to(self.device)
        )

        with torch.no_grad():
            pred = self.model(material_t)

        def denorm(t: torch.Tensor, key: str) -> torch.Tensor:
            mean, std = self.stats[key]
            return t * std + mean

        hz = denorm(pred["hzfield_state"].cpu(), "hz").squeeze(0)  # [M]
        mode = denorm(pred["mode_transmission"].cpu(), "mode").squeeze(0)  # [2]
        input_mode = denorm(pred["input_mode"].cpu().squeeze(-1), "input_mode").squeeze(0)  # scalar

        return {
            "hzfield_state": hz.numpy().tolist(),
            "output_mode_1": float(mode[0].item()),
            "output_mode_2": float(mode[1].item()),
            "input_mode": float(input_mode.item()),
        }

    def finetune(
        self,
        epochs: int = 10,
        batch_size: int | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        num_workers: int | None = None,
    ) -> Path:
        """
        Finetune the surrogate model using merged_train/merged_val produced by RLDataCollector.build().

        Returns:
            Path to the saved checkpoint (checkpoint_[num].pt).
        """
        epochs = epochs or tcfg.finetune_epochs
        # Use config defaults when args are not provided
        tcfg = surrogate_config.training
        batch_size = batch_size or tcfg.batch_size
        lr = lr or tcfg.finetune_lr
        weight_decay = weight_decay or tcfg.weight_decay
        num_workers = num_workers if num_workers is not None else tcfg.num_workers

        # Ensure merged data exists
        self.RL_data_collector.build()
        dcfg = surrogate_config.dataset
        merge_dir = Path(dcfg.merge_dir)
        train_path = merge_dir / "merged_train.npz"
        val_path = merge_dir / "merged_val.npz"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Merged data not found. Expected {train_path} and {val_path}. "
                "Run RLDataCollector.build() first."
            )

        # Datasets share original stats to keep normalization consistent
        train_ds = SurrogateNPZDataset(train_path, stats=self.stats)
        val_ds = SurrogateNPZDataset(val_path, stats=self.stats)

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Reuse existing model; set to train mode
        model = self.model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        weights = {
            "hz": tcfg.hz_weight,
            "mode": tcfg.mode_weight,
            "input": tcfg.input_weight,
        }

        best_val = float("inf")
        next_num = self.checkpoint_num + 1
        ckpt_path = self._checkpoint_path(next_num)

        for epoch in range(epochs):
            # Train
            model.train()
            for material, hz, mode, input_mode in train_loader:
                material = material.to(self.device)
                hz = hz.to(self.device)
                mode = mode.to(self.device)
                input_mode = input_mode.to(self.device)

                pred = model(material)
                losses = SurrogateTrainer.compute_losses(pred, hz, mode, input_mode, weights)
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

            # Validate
            model.eval()
            val_loss = 0.0
            batches = 0
            with torch.no_grad():
                for material, hz, mode, input_mode in val_loader:
                    material = material.to(self.device)
                    hz = hz.to(self.device)
                    mode = mode.to(self.device)
                    input_mode = input_mode.to(self.device)

                    pred = model(material)
                    losses = SurrogateTrainer.compute_losses(pred, hz, mode, input_mode, weights)
                    val_loss += losses["total"].item()
                    batches += 1

            val_loss = val_loss / max(batches, 1)
            print(f"[Finetune] Epoch {epoch+1}/{epochs} val_total={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val": best_val,
                    "stats": self.stats,
                }
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(ckpt, ckpt_path)
                print(f"[Finetune] Saved checkpoint to {ckpt_path}")

        # Update internal tracker to the latest checkpoint number and path
        self.checkpoint_num = next_num
        self.checkpoint = ckpt_path
        return ckpt_path

def main():
    parser = argparse.ArgumentParser(description="Run surrogate model prediction on a material matrix.")
    # parser.add_argument("--matrix", type=str, required=True, help="Path to .npy/.npz file containing a 2D matrix.")
    parser.add_argument("--checkpoint", type=str, default="surrogate_model/checkpoints/best.pt", help="Path to model checkpoint (default: training ckpt_dir/best.pt).")
    parser.add_argument("--data-dir", type=str, default="surrogate_model/data", help="Data directory containing train.npz for stats (default: dataset.output_dir).")
    parser.add_argument("--device", type=str, default="cpu", help="Device string, e.g., cuda or cpu (default: auto).")
    args = parser.parse_args()

    # matrix_path = Path(args.matrix)
    # if not matrix_path.exists():
    #     raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    rng = np.random.default_rng(42)
    material_matrix = rng.integers(0, 2, size=(20, 20), dtype=np.int8)
    predictor = RLSurrogateModel(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device,
    )
    outputs = predictor.predict(material_matrix)

    print(
        f"output_mode_1={outputs['output_mode_1']:.6f}, "
        f"output_mode_2={outputs['output_mode_2']:.6f}, "
        f"input_mode={outputs['input_mode']:.6f}"
    )
    print(f"hzfield_state (denorm, list length {len(outputs['hzfield_state'])}):")
    print(outputs["hzfield_state"])


if __name__ == "__main__":
    main()
