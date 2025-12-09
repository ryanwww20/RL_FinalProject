import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from surrogate_model.config import config as surrogate_config
from surrogate_model.model import SurrogateCNN
from surrogate_model.train import SurrogateNPZDataset


class RLSurrogateModel:
    """Lightweight wrapper for loading a trained surrogate and running inference."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        data_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        cfg_train = surrogate_config.training
        cfg_data = surrogate_config.dataset

        self.data_dir = Path(data_dir) if data_dir is not None else Path(cfg_data.output_dir)
        self.checkpoint = Path(checkpoint) if checkpoint is not None else Path(cfg_train.ckpt_dir) / "best.pt"
        self.device = torch.device(
            device
            if device is not None
            else (cfg_train.device if torch.cuda.is_available() else "cpu")
        )

        self.stats = self._load_stats()
        self.model = self._load_model()

    def _load_stats(self) -> Dict[str, tuple]:
        """Load normalization stats from training data (mean/std per target)."""
        train_npz = self.data_dir / "train.npz"
        if not train_npz.exists():
            raise FileNotFoundError(f"train.npz not found at {train_npz}")
        train_ds = SurrogateNPZDataset(train_npz)
        return train_ds.stats

    def _load_model(self) -> SurrogateCNN:
        """Load model weights and switch to eval mode."""
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        model = SurrogateCNN().to(self.device)
        ckpt = torch.load(self.checkpoint, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model

    def predict(self, material_matrix: np.ndarray) -> Dict[str, float]:
        """
        Run inference.

        Args:
            material_matrix: 2D numpy array with shape [H, W], values 0/1.

        Returns:
            Dictionary with denormalized outputs: output_mode_1, output_mode_2, input_mode.
        """
        if material_matrix.ndim != 2:
            raise ValueError(f"material_matrix must be 2D [H, W]; got shape {material_matrix.shape}")

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

        mode = denorm(pred["mode_transmission"].cpu(), "mode").squeeze(0)  # [2]
        input_mode = denorm(pred["input_mode"].cpu().squeeze(-1), "input_mode").squeeze(0)  # scalar

        return {
            "output_mode_1": float(mode[0].item()),
            "output_mode_2": float(mode[1].item()),
            "input_mode": float(input_mode.item()),
        }


def _load_matrix(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        data = np.load(path)
        if len(data.files) == 0:
            raise ValueError("NPZ file is empty.")
        arr = data[data.files[0]]
    else:
        arr = np.load(path)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected matrix shape [H, W], got {arr.shape}")
    return arr


def main():
    parser = argparse.ArgumentParser(description="Run surrogate model prediction on a material matrix.")
    parser.add_argument("--matrix", type=str, required=True, help="Path to .npy/.npz file containing a 2D matrix.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (default: training ckpt_dir/best.pt).")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory containing train.npz for stats (default: dataset.output_dir).")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g., cuda or cpu (default: auto).")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    material_matrix = _load_matrix(str(matrix_path))
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


if __name__ == "__main__":
    main()
