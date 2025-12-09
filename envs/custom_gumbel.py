# RL_FinalProject/custom_gumbel.py
import torch as th
import torch.nn as nn
from torch.distributions import RelaxedBernoulli
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.torch_layers import create_mlp


class GumbelBernoulliDistribution(Distribution):
    def __init__(self, action_dim: int, temperature: float = 0.5):
        super().__init__()
        self.action_dim = action_dim
        self.temperature = temperature
        self._dist = None

    def proba_distribution(self, action_logits: th.Tensor):
        self._dist = RelaxedBernoulli(self.temperature, logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self._dist.log_prob(actions).sum(dim=1, keepdim=True)

    def sample(self) -> th.Tensor:
        return self._dist.sample()

    def rsample(self) -> th.Tensor:
        return self._dist.rsample()

    def mode(self) -> th.Tensor:
        return th.sigmoid(self._dist.logits)


class GumbelActor(Actor):
    """
    Actor outputs logits, then Gumbel-Sigmoid sampling (0~1 continuous).
    """
    def __init__(self, *args, temperature: float = 0.5, net_arch=(512, 512), **kwargs):
        super().__init__(*args, **kwargs)
        act_dim = self.action_space.shape[0]
        self.action_dist = GumbelBernoulliDistribution(act_dim, temperature)
        self.action_net = nn.Sequential(
            *create_mlp(self.features_dim, act_dim, net_arch, nn.ReLU)
        )

    def get_action_dist_params(self, obs: th.Tensor):
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        logits = self.action_net(latent_pi)
        return logits, None, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        logits, _, _ = self.get_action_dist_params(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        actions = dist.mode() if deterministic else dist.rsample()  # soft action
        log_prob = dist.log_prob(actions)
        actions = th.clamp(actions, 0.0, 1.0)  # Just in case
        return actions, log_prob


class GumbelSACPolicy(SACPolicy):
    """
    Use GumbelActor instead of default Gaussian Actor.
    """
    def __init__(self, *args, temperature: float = 0.5, net_arch=(512, 512), **kwargs):
        self.gumbel_temperature = temperature
        self.gumbel_net_arch = net_arch
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        # 有些版本的 SB3 在 _build 之前尚未設置 features_dim，保險起見取用 extractor 的 features_dim
        extractor = features_extractor or self.features_extractor
        features_dim = getattr(self, "features_dim", None) or getattr(extractor, "features_dim")

        return GumbelActor(
            self.observation_space,
            self.action_space,
            extractor,
            features_dim,
            temperature=self.gumbel_temperature,
            net_arch=self.gumbel_net_arch,
            use_sde=False,
        )