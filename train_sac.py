"""
Soft Actor-Critic (SAC) Training Script
Uses Stable-Baselines3 for SAC implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from envs.Continuous_gym import MinimalEnv, TARGET_FLUX

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


class RewardAndFluxCallback(BaseCallback):
    """
    Custom callback to save rewards to CSV and flux distribution images.
    """

    def __init__(self, save_dir="./training_logs/", save_freq=1000, verbose=1):
        """
        Initialize the callback.

        Args:
            save_dir: Directory to save CSV and images
            save_freq: Frequency (in steps) to save data
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        super(RewardAndFluxCallback, self).__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.step_count = 0
        self.num_timesteps = None  # Will be set when training starts

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "flux_images"), exist_ok=True)

        # CSV file path
        self.csv_path = os.path.join(save_dir, "rewards.csv")

        if self.verbose > 0:
            print(f"Callback initialized:")
            print(f"  Save directory: {save_dir}")
            print(f"  Save frequency: Every {save_freq} steps")
            print(f"  CSV file: {self.csv_path}")
            print(
                f"  Images directory: {os.path.join(save_dir, 'flux_images')}")

    def _on_training_start(self):
        """Called when training starts."""
        # Get total timesteps from the model
        if hasattr(self.model, 'n_steps') and hasattr(self.model, 'n_envs'):
            # Estimate total timesteps (may not be exact)
            pass
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Training Started!")
            print(f"{'='*60}")

    def _on_step(self) -> bool:
        """
        Called at each step of training.

        Returns:
            bool: True to continue training, False to stop
        """
        # Get info from the environment
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]

            # Track episode rewards
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.rewards.append({
                    "step": self.step_count,
                    "episode": len(self.episode_rewards),
                    "reward": episode_reward,
                    "length": episode_length
                })

                # Print progress for each completed episode
                if self.verbose > 0:
                    print(f"\nEpisode {len(self.episode_rewards)} | "
                          f"Reward: {episode_reward:.4f} | "
                          f"Length: {episode_length} | "
                          f"Step: {self.step_count}")

        self.step_count += 1

        # Save data periodically and show progress
        if self.step_count % self.save_freq == 0:
            # Get training progress info
            if self.num_timesteps is None:
                # Try to get from model or locals
                if hasattr(self, 'model') and hasattr(self.model, 'n_steps'):
                    # This is approximate
                    pass

            if self.verbose > 0:
                print(f"\n{'─'*60}")
                print(f"Progress Update at Step {self.step_count}")
                print(f"{'─'*60}")
                if len(self.episode_rewards) > 0:
                    recent_avg = np.mean(self.episode_rewards[-10:]) if len(
                        self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                    print(f"  Episodes completed: {len(self.episode_rewards)}")
                    print(f"  Average reward (last 10): {recent_avg:.4f}")
                    print(f"  Best reward: {np.max(self.episode_rewards):.4f}")
                    print(
                        f"  Average episode length: {np.mean(self.episode_lengths):.2f}")
                print(f"  Saving rewards and flux image...")

            self._save_rewards()
            self._save_flux_image()

        return True

    def _save_rewards(self):
        """Save rewards to CSV file."""
        if len(self.rewards) > 0:
            df = pd.DataFrame(self.rewards)
            df.to_csv(self.csv_path, index=False)
            if self.verbose > 0:
                print(f"Saved rewards to {self.csv_path}")

    def _save_flux_image(self):
        """Save flux distribution image from the environment."""
        try:
            # Get the environment from the model
            env = self.training_env.envs[0] if hasattr(
                self.training_env, 'envs') else None

            if env is not None and hasattr(env, 'flux_calculator'):
                # Get current material matrix
                if hasattr(env, 'material_matrix'):
                    material_matrix = env.material_matrix
                else:
                    # If material_matrix not directly accessible, create from unwrapped env
                    unwrapped = env.unwrapped if hasattr(
                        env, 'unwrapped') else env
                    material_matrix = unwrapped.material_matrix

                # Get flux calculator
                if hasattr(env, 'flux_calculator'):
                    flux_calc = env.flux_calculator
                else:
                    unwrapped = env.unwrapped if hasattr(
                        env, 'unwrapped') else env
                    flux_calc = unwrapped.flux_calculator

                # Calculate flux
                flux_array = flux_calc.calculate_flux(
                    material_matrix, x_position=2.0
                )

                # Create and save plot
                plt.figure(figsize=(10, 6))
                plt.plot(flux_array, 'b-', linewidth=2, label='Current Flux')

                # Plot target flux if available
                if TARGET_FLUX is not None:
                    plt.plot(TARGET_FLUX, 'r--', linewidth=2,
                             alpha=0.7, label='Target Flux')

                plt.xlabel('Detector Index')
                plt.ylabel('Flux')
                plt.title(f'Flux Distribution at Step {self.step_count}')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Save image
                image_path = os.path.join(
                    self.save_dir,
                    "flux_images",
                    f"flux_step_{self.step_count:06d}.png"
                )
                plt.savefig(image_path, dpi=150, bbox_inches='tight')
                plt.close()

                if self.verbose > 0:
                    print(f"Saved flux image to {image_path}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not save flux image: {e}")

    def _on_training_end(self):
        """Called when training ends."""
        # Final save
        self._save_rewards()
        self._save_flux_image()

        # Print summary
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"{'='*60}")
            print(f"Total timesteps: {self.step_count}")
            if len(self.episode_rewards) > 0:
                print(f"Total episodes: {len(self.episode_rewards)}")
                print(f"Average reward: {np.mean(self.episode_rewards):.4f}")
                print(f"Std deviation: {np.std(self.episode_rewards):.4f}")
                print(f"Best reward: {np.max(self.episode_rewards):.4f}")
                print(f"Worst reward: {np.min(self.episode_rewards):.4f}")
                print(
                    f"Average episode length: {np.mean(self.episode_lengths):.2f}")
                print(f"Rewards saved to: {self.csv_path}")
                print(
                    f"Flux images saved to: {os.path.join(self.save_dir, 'flux_images')}")
            print(f"{'='*60}\n")


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

    # Create custom callback for rewards and flux images
    reward_callback = RewardAndFluxCallback(
        save_dir=f"{save_path}_logs",
        save_freq=1000,  # Save every 1000 steps
        verbose=1
    )

    # Train the model
    print(f"Training SAC agent for {total_timesteps} timesteps...")
    print("Press Ctrl+C to interrupt training and save current model...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=reward_callback,
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
        print(f"[config] Config file not found at {path}. Using train_sac defaults.")
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    training_cfg = data.get("training", {}).get("sac", {}) or {}
    filtered_cfg = {k: v for k, v in training_cfg.items() if k in TRAIN_SAC_KWARGS}

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
