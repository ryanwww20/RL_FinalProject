"""
Soft Actor-Critic (SAC) Training Script
Uses Stable-Baselines3 for SAC implementation
"""

import numpy as np
import os
from pathlib import Path
from datetime import datetime

import yaml
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from envs.Continuous_gym import MinimalEnv


class CustomMetricsCallback(BaseCallback):
    """
    Custom callback to log environment-specific metrics to WandB.
    Logs every step for detailed tracking.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metric_keys = [
            "total_transmission",
            "transmission_score",
            "balance_score",
            "current_score",
            "output_flux_1_ratio",
            "output_flux_2_ratio",
            "loss_ratio",
        ]

    def _on_step(self) -> bool:
        # Get info from the environment
        infos = self.locals.get("infos", [])
        for info in infos:
            metrics_to_log = {}
            for key in self.metric_keys:
                if key in info:
                    metrics_to_log[f"env/{key}"] = info[key]
            
            # Log immediately if we have metrics
            if metrics_to_log:
                wandb.log(metrics_to_log, step=self.num_timesteps)

        return True

CONFIG_ENV_VAR = "TRAINING_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
TRAIN_SAC_KWARGS = {
    "total_timesteps",
    "n_envs",
    "learning_rate",
    "buffer_size",
    "learning_starts",
    "batch_size",
    "tau",
    "gamma",
    "train_freq",
    "gradient_steps",
    "ent_coef",
    "target_update_interval",
    "target_entropy",
    "use_sde",
    "tensorboard_log",
    "save_path",
    "checkpoint_freq",
    "wandb_project",
    "wandb_entity",
    "wandb_run_name",
    "wandb_tags",
}


def train_sac(
    total_timesteps=100000,
    n_envs=4,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    target_update_interval=1,
    target_entropy="auto",
    use_sde=False,
    tensorboard_log="./sac_tensorboard/",
    save_path="./sac_model",
    checkpoint_freq=200,
    wandb_project=None,
    wandb_entity=None,
    wandb_run_name=None,
    wandb_tags=None,
):
    """
    Train a SAC agent on the MinimalEnv environment.

    Args:
        total_timesteps: Total number of timesteps to train
        n_envs: Number of parallel environments
        learning_rate: Learning rate for optimizer
        buffer_size: Replay buffer size
        learning_starts: Timesteps before training begins
        batch_size: Batch size sampled from replay buffer
        tau: Target smoothing coefficient
        gamma: Discount factor
        train_freq: How often to update (can be int or (int, "step"))
        gradient_steps: Gradient steps per update
        ent_coef: Entropy coefficient or "auto"
        target_update_interval: Frequency of target network updates
        target_entropy: Target entropy for entropy tuning
        use_sde: Whether to use State Dependent Exploration
        tensorboard_log: Directory for tensorboard logs
        save_path: Path to save the trained model
        checkpoint_freq: Frequency (in timesteps) to save checkpoints
        wandb_project: WandB project name (optional)
        wandb_entity: WandB entity/username (optional)
        wandb_run_name: WandB run name (optional)
        wandb_tags: List of tags for WandB run (optional)
    """

    # Initialize WandB if project name is provided
    callbacks = []
    
    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"{save_path}_checkpoints_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Add checkpoint callback to save model every checkpoint_freq steps
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="sac_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1,
    )
    callbacks.append(checkpoint_callback)
    print(f"Checkpoints will be saved every {checkpoint_freq} steps to {checkpoint_dir}")
    
    if wandb_project:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            tags=wandb_tags,
            sync_tensorboard=True,  # Sync TensorBoard logs
            monitor_gym=True,       # Monitor Gym environment
            save_code=True,         # Save code
        )
        callbacks.append(WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
        ))
        callbacks.append(CustomMetricsCallback(verbose=1))

    # Create vectorized environment (parallel environments)
    print("Creating environment...")
    env = make_vec_env(MinimalEnv, n_envs=n_envs,
                       env_kwargs={"render_mode": None},
                       vec_env_cls=SubprocVecEnv)

    # Create evaluation environment
    eval_env = MinimalEnv(render_mode=None)

    # Create SAC model
    print("Creating SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        ent_coef=ent_coef,
        target_update_interval=target_update_interval,
        target_entropy=target_entropy,
        use_sde=use_sde,
        tensorboard_log=tensorboard_log,
        verbose=1
    )

    # Train the model
    print(f"Training SAC agent for {total_timesteps} timesteps...")
    print("Press Ctrl+C to interrupt training and save current model...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=False  # Set to False to avoid tqdm/rich dependency
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Saving current model state...")
    except Exception as e:
        print(f"Error during training: {e}")

    # Save the final model (even if interrupted)
    # Use same timestamp as checkpoint directory
    save_path_with_timestamp = f"{save_path}_{timestamp}"

    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path_with_timestamp)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    model.save(save_path_with_timestamp)
    print(f"Model saved to {save_path_with_timestamp}")

    # Test the trained model
    print("\nTesting trained model...")
    test_model(model, eval_env, n_episodes=3)

    if wandb_project:
        wandb.finish()

    return model


def test_model(model, env, n_episodes=5):
    """
    Test a trained model on the environment.

    Args:
        model: Trained RL model
        env: Environment to test on
        n_episodes: Number of episodes to test
    """
    total_rewards = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1

        total_rewards.append(episode_reward)
        print(
            f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}")

    print(
        f"\nAverage reward over {n_episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Std deviation: {np.std(total_rewards):.2f}")


def load_training_config(config_path=None):
    """
    Load train_sac keyword arguments from YAML config.

    Args:
        config_path: Optional override path. Defaults to config.yaml next to this file.

    Returns:
        dict: Filtered kwargs to pass into train_sac.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        print(
            f"[config] Config file not found at {path}. Using train_sac defaults.")
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    training_cfg = data.get("training", {}).get("sac", {}) or {}
    filtered_cfg = {k: v for k, v in training_cfg.items()
                    if k in TRAIN_SAC_KWARGS}

    unknown_keys = sorted(set(training_cfg.keys()) - TRAIN_SAC_KWARGS)
    if unknown_keys:
        print(f"[config] Ignoring unsupported train_sac keys: {unknown_keys}")

    return filtered_cfg


if __name__ == "__main__":
    config_override_path = os.environ.get(CONFIG_ENV_VAR)
    train_kwargs = load_training_config(config_override_path)

    # Train SAC agent
    model = train_sac(**train_kwargs)

    print("\nTraining complete!")
