import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MatrixCombinedExtractor(BaseFeaturesExtractor):
    """
    Extract features from material matrix thru CNN. Then combine with monitor/idx/prev layer
    Observation layout assumption:
        [matrix_flat (pixel_num_x * pixel_num_y),
         monitors (num_monitors),
         idx (1),
         prev_layer (pixel_num_y)]
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        cnn_proj_dim: int = 128,
        pixel_num_x: int = 20,
        pixel_num_y: int = 20,
        num_monitors: int = 10,
    ):
        self.pixel_num_x = pixel_num_x
        self.pixel_num_y = pixel_num_y
        self.num_monitors = num_monitors

        matrix_len = pixel_num_x * pixel_num_y
        scalar_len = observation_space.shape[0] - matrix_len

        # CNN branch only operates on matrix
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 20x20 -> 10x10
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 10x10 -> 5x5
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, pixel_num_x, pixel_num_y)
            cnn_out_dim = self.cnn(dummy).shape[1]

        # Compress CNN output to cnn_proj_dim, then concat with scalar
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, cnn_proj_dim),
            nn.ReLU(),
        )

        # features_dim = CNN projection + scalar as is
        features_dim = cnn_proj_dim + scalar_len
        super().__init__(observation_space, features_dim)
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        matrix_len = self.pixel_num_x * self.pixel_num_y
        matrix_flat = observations[:, :matrix_len]
        scalars = observations[:, matrix_len:]

        matrix_2d = matrix_flat.view(-1, 1, self.pixel_num_x, self.pixel_num_y)
        cnn_feat = self.cnn(matrix_2d)
        cnn_feat = self.cnn_proj(cnn_feat)

        # Concat CNN features with raw scalar, pass to subsequent MLP (policy/value net_arch)
        return torch.cat([cnn_feat, scalars], dim=1)

