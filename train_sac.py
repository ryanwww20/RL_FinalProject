"""
Soft Actor-Critic (SAC) Training Script
Uses Stable-Baselines3 for SAC implementation
"""

import numpy as np
import os
from pathlib import Path
from datetime import datetime

import yaml
from stable_baselines3 import SAC
from envs.Continuous_gym import MinimalEnv

CONFIG_ENV_VAR = "TRAINING_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
TRAIN_SAC_KWARGS = {
    "total_timesteps",
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
}


def train_sac(
    total_timesteps=100000,
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
    save_path="./sac_model"
):
    """
    Train a SAC agent on the MinimalEnv environment.

    Args:
        total_timesteps: Total number of timesteps to train
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
    """

    print("Creating environment...")
    env = MinimalEnv(render_mode=None)

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
            progress_bar=False  # Set to False to avoid tqdm/rich dependency
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Saving current model state...")
    except Exception as e:
        print(f"Error during training: {e}")

    # Save the final model (even if interrupted)
    # Add timestamp to model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path_with_timestamp = f"{save_path}_{timestamp}"

    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path_with_timestamp)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    model.save(save_path_with_timestamp)
    print(f"Model saved to {save_path_with_timestamp}")

    # Test the trained model
    print("\nTesting trained model...")
    test_model(model, eval_env, n_episodes=5)

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
    import sys
    
    # Check if user wants to test a saved model
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) < 3:
            print("Usage: python train_sac.py test <model_path> [n_episodes]")
            sys.exit(1)
        
        model_path = sys.argv[2]
        n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        
        print(f"Loading model from {model_path}...")
        model = SAC.load(model_path)
        
        print("Creating test environment...")
        test_env = MinimalEnv(render_mode=None)
        
        print(f"Testing model for {n_episodes} episodes...")
        test_model(model, test_env, n_episodes=n_episodes)
    else:
        # Normal training flow
        config_override_path = os.environ.get(CONFIG_ENV_VAR)
        train_kwargs = load_training_config(config_override_path)
        
        # Train SAC agent
        model = train_sac(**train_kwargs)
        
        print("\nTraining complete!")
