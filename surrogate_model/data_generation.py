"""
Dataset builder for the surrogate CNN using config.yaml settings.

Outputs per sample (shapes driven by config.simulation.pixel_num_{x,y}):
- material_matrix: design binary grid
- hzfield_state: |Hz| magnitude map resampled to config grid
- mode_transmission_1, mode_transmission_2: scalar TE0 outputs
- input_mode: scalar TE0 power at input

Usage:
    python surrogate_model/data_generation.py --num-samples 10 --out surrogate_model/data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import random

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from envs.meep_simulation import WaveguideSimulation
from surrogate_model.config import config as surrogate_config
from config import config as main_config

PIXEL_SHAPE = (20, 20)
PIXEL_NUM_X = 20
PIXEL_NUM_Y = 20

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_split_npz(path: Path) -> Dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _concat_split(existing: Dict[str, np.ndarray] | None, new: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if existing is None:
        return new
    out = {}
    for k, v_new in new.items():
        v_exist = existing[k]
        out[k] = np.concatenate([v_exist, v_new], axis=0)
    return out
class SurrogateDatasetBuilder:
    """Generate supervised data by running Meep sweeps."""

    def __init__(self):
        self.config = surrogate_config.dataset
        self.rng = np.random.default_rng(self.config.seed)

    def _sample_matrix(self) -> np.ndarray:

        all_zero_matrix_prob = self.rng.random()
        if all_zero_matrix_prob < 0.01:
            return np.zeros((20, 20))

        layer = self.rng.integers(0, 21)
        built_matrix = self.rng.integers(0, 2, size=(layer, 20), dtype=np.int8)
        nubuilt_matrix = np.zeros((20-layer, 20))
        full_matrix = np.vstack((built_matrix, nubuilt_matrix))
        return full_matrix

    def _run_single(self, sample_idx: int) -> Dict[str, np.ndarray]:
        sim = WaveguideSimulation()
        matrix = self._sample_matrix()

        hzfield_state, _ = sim.calculate_flux(matrix)
        # Get flux-based quantities
        _, input_mode = sim.get_flux_input_mode(band_num=1)
        _, _, mode_transmission_1, mode_transmission_2, _ = sim.get_flux_output_mode(
            band_num=1
        )

        return {
            "material_matrix": matrix.astype(np.float32),
            "hzfield_state": hzfield_state.astype(np.float32),
            "mode_transmission_1": mode_transmission_1,
            "mode_transmission_2": mode_transmission_2,
            "input_mode": input_mode,
        }

    def _split_indices(self, n: int) -> Tuple[List[int], List[int], List[int]]:
        idxs = np.arange(n)
        self.rng.shuffle(idxs)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)
        train_idx = idxs[:n_train].tolist()
        val_idx = idxs[n_train : n_train + n_val].tolist()
        test_idx = idxs[n_train + n_val :].tolist()
        return train_idx, val_idx, test_idx

    def _extract_split(self, packed: Dict[str, np.ndarray], idx_list: List[int]) -> Dict[str, np.ndarray]:
        return {k: v[idx_list] for k, v in packed.items()}

    def _pack(self, samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            "material_matrix": np.stack([s["material_matrix"] for s in samples]),
            "hzfield_state": np.stack([s["hzfield_state"] for s in samples]),
            "mode_transmission": np.stack(
                [[s["mode_transmission_1"], s["mode_transmission_2"]] for s in samples]
            ).astype(np.float32),
            "input_mode": np.array(
                [s["input_mode"] for s in samples], dtype=np.float32
            ),
        }

    def build(self) -> None:
        _ensure_dir(self.config.output_dir)
        samples: List[Dict[str, np.ndarray]] = []

        start = time.time()
        for i in range(self.config.num_samples):
            sample = self._run_single(i)
            samples.append(sample)
            if (i + 1) % 5 == 0 or i == 0:
                elapsed = time.time() - start
                print(f"[{i+1}/{self.config.num_samples}] generated in {elapsed:.1f}s")

        train_idx, val_idx, test_idx = self._split_indices(len(samples))
        splits = {"train": train_idx, "val": val_idx, "test": test_idx}

        packed = self._pack(samples)

        # Load existing splits (if any) and append new data
        final_counts = {}
        for split_name, idx_list in splits.items():
            if not idx_list:
                continue
            subset_new = self._extract_split(packed, idx_list)
            split_path = Path(self.config.output_dir) / f"{split_name}.npz"
            existing = _load_split_npz(split_path)
            merged = _concat_split(existing, subset_new)
            np.savez_compressed(split_path, **merged)
            final_counts[split_name] = merged["material_matrix"].shape[0]
            prev = existing["material_matrix"].shape[0] if existing else 0
            print(f"Saved {split_name}: +{len(idx_list)} (prev {prev}) -> total {final_counts[split_name]}")

        meta = {
            "num_samples": self.config.num_samples,
            "train_ratio": self.config.train_ratio,
            "val_ratio": self.config.val_ratio,
            "seed": self.config.seed,
            "splits": final_counts,
            "pixel_num_x": PIXEL_SHAPE[0],
            "pixel_num_y": PIXEL_SHAPE[1],
        }
        with open(os.path.join(self.config.output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata saved to {os.path.join(self.config.output_dir, 'meta.json')}")


if __name__ == "__main__":
    builder = SurrogateDatasetBuilder()
    builder.build()

