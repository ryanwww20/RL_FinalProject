"""
CNN surrogate model for Meep outputs.

Inputs:
    - material_matrix: float tensor [B, 1, H, W] (H=W=20 from config)

Outputs:
    - hzfield_state: float tensor [B, 1, H, W] (regressed |Hz| map)
    - mode_transmission: float tensor [B, 2] (mode_transmission_1/2)
    - input_mode: float tensor [B, 1] (input mode power)

Shapes and defaults are driven by config.simulation.pixel_num_{x,y}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from config import config as main_config
from surrogate_model.config import config as surrogate_config


def _conv_block(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class SurrogateCNN(nn.Module):
    """Shared CNN backbone with multi-head decoders."""

    def __init__(self):
        super().__init__()
        self.config = surrogate_config.model
        c = self.config.base_channels
        hzfield_state_dim = main_config.num_monitors
        # Backbone keeps spatial dims (stride=1)
        self.backbone = nn.Sequential(
            _conv_block(self.config.input_channels, c),
            _conv_block(c, c),
            _conv_block(c, c * 2),
        )

        # Scalar head uses global pooling then MLP
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.scalar_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 2, c),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
        )
        self.hzfield_state_head = nn.Linear(c, hzfield_state_dim)
        self.transmission_head = nn.Linear(c, 2)
        self.input_mode_head = nn.Linear(c, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, 1, H, W] material matrix
        Returns:
            dict with keys: hzfield_state, mode_transmission, input_mode
        """
        feats = self.backbone(x)

        pooled = self.pool(feats)
        scalars = self.scalar_mlp(pooled)
        hzfield_state = self.hzfield_state_head(scalars)
        transmissions = self.transmission_head(scalars)
        input_mode = self.input_mode_head(scalars)

        return {
            "hzfield_state": hzfield_state,
            "mode_transmission": transmissions,
            "input_mode": input_mode,
        }


def build_model() -> SurrogateCNN:
    """Factory that builds the model with config-driven shapes."""
    return SurrogateCNN()


__all__ = ["SurrogateCNN", "SurrogateConfig", "build_model"]

