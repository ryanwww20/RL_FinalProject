# RL_FinalProject/custom_gumbel.py
import torch as th
import torch.nn as nn
from torch.distributions import RelaxedBernoulli
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.torch_layers import create_mlp, MlpExtractor


class GumbelBernoulliDistribution(Distribution):
    def __init__(self, action_dim: int, temperature: float = 0.5):
        super().__init__()
        self.action_dim = action_dim
        self.temperature = temperature
        self._dist = None

    def proba_distribution(self, action_logits: th.Tensor):
        temperature = th.as_tensor(self.temperature, device=action_logits.device, dtype=action_logits.dtype)
        self._dist = RelaxedBernoulli(temperature, logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self._dist.log_prob(actions).sum(dim=1, keepdim=True)

    # === Abstracts from Distribution ===
    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        # For API compatibility; not used because Actor supplies logits directly
        return nn.Linear(latent_dim, self.action_dim)

    def log_prob_from_params(self, action_logits: th.Tensor, actions: th.Tensor) -> th.Tensor:
        dist = self.proba_distribution(action_logits=action_logits)
        return dist.log_prob(actions)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False):
        dist = self.proba_distribution(action_logits=action_logits)
        actions = dist.mode() if deterministic else dist.rsample()
        log_prob = dist.log_prob(actions)
        return actions, log_prob

    def entropy(self) -> th.Tensor:
        return self._dist.entropy().sum(dim=1, keepdim=True)

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
    def __init__(
        self,
        observation_space,
        action_space,
        features_extractor,
        features_dim,
        temperature: float = 0.5,
        net_arch=(512, 512),
        **kwargs,
    ):
        # 兼容不同版本 SB3：若上層以 feature_dim 傳入，轉為 features_dim
        if "feature_dim" in kwargs and "features_dim" not in kwargs:
            kwargs["features_dim"] = kwargs.pop("feature_dim")
        # Call base Actor to init common parts; then override action_net/action_dist
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=list(net_arch),
            activation_fn=nn.ReLU,
            log_std_init=-3.0,
            full_std=True,
            use_expln=False,
            **kwargs,
        )

        # SB3 的 Actor 通常會在 __init__ 建立 mlp_extractor
        # 若基類版本不同未設置，這裡補一個預設的 MlpExtractor
        if not hasattr(self, "mlp_extractor"):
            self.mlp_extractor = MlpExtractor(
                feature_dim=features_dim,
                net_arch=list(net_arch),
                activation_fn=nn.ReLU,
            )

        act_dim = self.action_space.shape[0]
        self.action_dist = GumbelBernoulliDistribution(act_dim, temperature)
        
        # Calculate latent_dim_pi based on net_arch
        # MlpExtractor outputs vectors of size net_arch[-1]
        latent_dim_pi = list(net_arch)[-1] if len(net_arch) > 0 else features_dim
        
        # Replace action_net to output logits only (no std head)
        # Input is latent_pi from mlp_extractor
        self.action_net = nn.Linear(latent_dim_pi, act_dim)

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

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Override to ensure we return only the action tensor, not (action, log_prob)
        # This fixes "tuple has no attribute cpu" error during evaluation
        actions, _ = self.actor(observation, deterministic=deterministic)
        return actions

    def make_actor(self, features_extractor=None):
        # Some versions of SB3 call make_actor before features_extractor is fully built
        extractor = features_extractor or getattr(self, "features_extractor", None)
        if extractor is None:
            # Build a fresh features_extractor when none is available
            extractor = self.features_extractor_class(
                self.observation_space, **self.features_extractor_kwargs
            )

        features_dim = getattr(self, "features_dim", None) or getattr(extractor, "features_dim", None)
        if features_dim is None:
            raise AttributeError("features_dim is not set on features_extractor")

        return GumbelActor(
            self.observation_space,
            self.action_space,
            extractor,
            features_dim,
            temperature=self.gumbel_temperature,
            net_arch=self.gumbel_net_arch,
            use_sde=False,
        )