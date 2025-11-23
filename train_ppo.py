"""
PPO (Proximal Policy Optimization) Training Script
Uses Stable-Baselines3 for PPO implementation
"""

import numpy as np
import os
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.Discrete_gym import MinimalEnv

CONFIG_ENV_VAR = "TRAINING_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
TRAIN_PPO_KWARGS = {
    "total_timesteps",
    "n_envs",
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "tensorboard_log",
    "save_path",
}


def train_ppo(
    total_timesteps=100000,
    n_envs=4,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_tensorboard/",
    save_path="./ppo_model"
):
    """
    Train a PPO agent on the MinimalEnv environment.

    Args:
        total_timesteps: Total number of timesteps to train
        n_envs: Number of parallel environments
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps to collect per update
        batch_size: Batch size for training
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm for clipping
        tensorboard_log: Directory for tensorboard logs
        save_path: Path to save the trained model
    """

    # Create vectorized environment (parallel environments)
    print("Creating environment...")
    env = make_vec_env(MinimalEnv, n_envs=n_envs,
                       env_kwargs={"render_mode": None})

    # Create evaluation environment
    eval_env = MinimalEnv(render_mode=None)

    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=tensorboard_log,
        verbose=1
    )

    # Train the model
    print(f"Training PPO agent for {total_timesteps} timesteps...")
    print("Press Ctrl+C to interrupt training and save current model...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=False  # Set to False to avoid tqdm/rich dependency
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Saving current model state...")
    except Exception as e:
        print(f"Error during training: {e}")

    # Save the final model (even if interrupted)
    model.save(save_path)
    print(f"Model saved to {save_path}")

    # Test the trained model
    print("\nTesting trained model...")
    test_model(model, eval_env, n_episodes=5)

    return model


def test_model(model, env, n_episodes=5):
    """
    Test a trained model on the environment.

    Args:
        model: Trained PPO model
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
    Load train_ppo keyword arguments from YAML config.

    Args:
        config_path: Optional override path. Defaults to config.yaml next to this file.

    Returns:
        dict: Filtered kwargs to pass into train_ppo.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        print(
            f"[config] Config file not found at {path}. Using train_ppo defaults.")
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    training_cfg = data.get("training", {}).get("ppo", {}) or {}
    filtered_cfg = {k: v for k, v in training_cfg.items()
                    if k in TRAIN_PPO_KWARGS}

    unknown_keys = sorted(set(training_cfg.keys()) - TRAIN_PPO_KWARGS)
    if unknown_keys:
        print(f"[config] Ignoring unsupported train_ppo keys: {unknown_keys}")

    return filtered_cfg


if __name__ == "__main__":
    config_override_path = os.environ.get(CONFIG_ENV_VAR)
    train_kwargs = load_training_config(config_override_path)

    # Train PPO agent
    model = train_ppo(**train_kwargs)

    print("\nTraining complete!")
