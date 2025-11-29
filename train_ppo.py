"""
PPO (Proximal Policy Optimization) Training Script
Uses Stable-Baselines3 for PPO implementation
"""

import numpy as np
import os
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from envs.Discrete_gym import MinimalEnv
from PIL import Image

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


class TrainingCallback(BaseCallback):
    """
    Callback to record metrics, plot designs, save to CSV, and create GIFs.
    Follows README structure: ppo_model_log_<start_time>/ with img/, plot/, result.csv
    """
    def __init__(self, save_dir, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_count = 0
        
        # Create directory structure according to README
        self.img_dir = self.save_dir / "img"
        self.plot_dir = self.save_dir / "plot"
        self.img_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)
        
        # Directories for temporary images (will be used for GIFs)
        self.design_dir = self.save_dir / "design_images"
        self.distribution_dir = self.save_dir / "distribution_images"
        self.design_dir.mkdir(exist_ok=True)
        self.distribution_dir.mkdir(exist_ok=True)
        
        # CSV file for metrics (result.csv as per README)
        self.csv_path = self.save_dir / "result.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, 'w') as f:
                f.write('timestamp,rollout_count,transmission,balance_score,score,reward\n')
        
        # Store image paths for GIF creation
        self.design_image_paths = []
        self.distribution_image_paths = []
    
    def _on_step(self) -> bool:
        """Called at each environment step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called when rollout collection ends."""
        self.rollout_count += 1
        
        # Get environment from training_env
        env = self.training_env
        
        # Get environment attributes (works with DummyVecEnv)
        try:
            if hasattr(env, 'envs'):
                # Vectorized environment - get first environment (DummyVecEnv)
                unwrapped_env = env.envs[0].unwrapped
                material_matrix = unwrapped_env.material_matrix.copy()
                simulation = unwrapped_env.simulation
            else:
                # Single environment
                unwrapped_env = env.unwrapped
                material_matrix = unwrapped_env.material_matrix.copy()
                simulation = unwrapped_env.simulation
        except Exception as e:
            print(f"Warning: Could not access environment attributes: {e}")
            return
        
        # Try to get metrics from rollout buffer or recalculate
        try:
            # Recalculate to get current state and metrics
            _, _, _, efield_state, _, _, _, _ = simulation.calculate_flux(material_matrix)
            transmission_1, transmission_2, total_transmission, _ = simulation.get_output_transmission(band_num=1)
            
            # Calculate balance score
            if total_transmission > 0:
                diff_ratio = abs(transmission_1 - transmission_2) / total_transmission
            else:
                diff_ratio = 1.0
            balance_score = max(1 - diff_ratio, 0)
            
            # Calculate score (same as in get_reward)
            transmission_score = min(max(total_transmission, 0), 1)
            current_score = transmission_score * balance_score
            
            # Get reward from rollout statistics if available
            reward = 0.0
            if hasattr(self, 'logger') and self.logger.name_to_value:
                # Try to get mean reward from logger
                if 'rollout/ep_rew_mean' in self.logger.name_to_value:
                    reward = self.logger.name_to_value['rollout/ep_rew_mean']
            
        except Exception as e:
            print(f"Warning: Could not get metrics: {e}")
            total_transmission = 0.0
            balance_score = 0.0
            current_score = 0.0
            reward = 0.0
            efield_state = None
        
        # Record metrics to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.csv_path, 'a') as f:
            f.write(f'{timestamp},{self.rollout_count},{total_transmission},{balance_score},{current_score},{reward}\n')
        
        # Plot and save design
        if material_matrix is not None:
            design_path = self.design_dir / f"design_rollout_{self.rollout_count:04d}.png"
            self.design_image_paths.append(str(design_path))
            simulation.plot_design(
                matrix=material_matrix,
                save_path=str(design_path),
                show_plot=False
            )
        
        # Plot and save distribution
        if efield_state is not None:
            distribution_path = self.distribution_dir / f"distribution_rollout_{self.rollout_count:04d}.png"
            self.distribution_image_paths.append(str(distribution_path))
            simulation.plot_distribution(
                efield_state=efield_state,
                save_path=str(distribution_path),
                show_plot=False
            )
        
        # Print rollout count and metrics
        print(f"Rollout {self.rollout_count}: Transmission={total_transmission:.4f}, "
              f"Balance={balance_score:.4f}, Score={current_score:.4f}, Reward={reward:.4f}")
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        print(f"\nTraining ended. Creating GIFs and plots...")
        
        # Create design GIF (save to img/design.gif as per README)
        if self.design_image_paths:
            gif_path = self.img_dir / "design.gif"
            self._create_gif(self.design_image_paths, str(gif_path))
            print(f"Design GIF saved to: {gif_path}")
        
        # Create distribution GIF (save to img/flux.gif as per README)
        if self.distribution_image_paths:
            gif_path = self.img_dir / "flux.gif"
            self._create_gif(self.distribution_image_paths, str(gif_path))
            print(f"Distribution GIF saved to: {gif_path}")
        
        # Clean up temporary image directories
        import shutil
        if self.design_dir.exists():
            shutil.rmtree(self.design_dir)
        if self.distribution_dir.exists():
            shutil.rmtree(self.distribution_dir)
        
        # Plot recorded metrics from CSV
        self._plot_metrics()
    
    def _plot_metrics(self):
        """Plot transmission, balance_score, and score from CSV."""
        if not self.csv_path.exists():
            print("Warning: CSV file not found, cannot plot metrics")
            return
        
        try:
            # Read CSV
            df = pd.read_csv(self.csv_path)
            
            if len(df) == 0:
                print("Warning: CSV file is empty, cannot plot metrics")
                return
            
            # Plot transmission
            plt.figure(figsize=(10, 6))
            plt.plot(df['rollout_count'], df['transmission'], 'b-', linewidth=2, marker='o', markersize=4)
            plt.xlabel('Rollout Count')
            plt.ylabel('Transmission')
            plt.title('Transmission Over Training')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            transmission_plot_path = self.plot_dir / "transmission.png"
            plt.savefig(transmission_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Transmission plot saved to: {transmission_plot_path}")
            
            # Plot balance score
            plt.figure(figsize=(10, 6))
            plt.plot(df['rollout_count'], df['balance_score'], 'g-', linewidth=2, marker='s', markersize=4)
            plt.xlabel('Rollout Count')
            plt.ylabel('Balance Score')
            plt.title('Balance Score Over Training')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            balance_plot_path = self.plot_dir / "balance.png"
            plt.savefig(balance_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Balance score plot saved to: {balance_plot_path}")
            
            # Plot score
            plt.figure(figsize=(10, 6))
            plt.plot(df['rollout_count'], df['score'], 'r-', linewidth=2, marker='^', markersize=4)
            plt.xlabel('Rollout Count')
            plt.ylabel('Score')
            plt.title('Score Over Training')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            score_plot_path = self.plot_dir / "score.png"
            plt.savefig(score_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Score plot saved to: {score_plot_path}")
            
        except Exception as e:
            print(f"Error plotting metrics: {e}")
    
    def _create_gif(self, image_paths, output_path, duration=500, loop=0):
        """Create a GIF from a list of image paths."""
        images = []
        for path in image_paths:
            if os.path.exists(path):
                img = Image.open(path)
                images.append(img)
        
        if images:
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=loop
            )
            print(f"GIF created: {output_path} ({len(images)} frames)")
        else:
            print(f"Warning: No images found to create GIF at {output_path}")


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
    save_path="./ppo_model",
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
    # Save starting timestamp for model saving
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory for callbacks (following README structure)
    callback_dir = f"ppo_model_log_{start_timestamp}"
    os.makedirs(callback_dir, exist_ok=True)

    # Create vectorized environment (parallel environments)
    # Using DummyVecEnv instead of SubprocVecEnv because Meep simulation objects
    # contain lambda functions that can't be pickled for multiprocessing
    print("Creating environment...")
    env = make_vec_env(MinimalEnv, n_envs=n_envs,
                       env_kwargs={"render_mode": None},
                       vec_env_cls=DummyVecEnv)

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

    # Create callback
    callback = TrainingCallback(save_dir=callback_dir, verbose=1)

    # Train the model
    print(f"Training PPO agent for {total_timesteps} timesteps...")
    print("Press Ctrl+C to interrupt training and save current model...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False  # Set to False to avoid tqdm/rich dependency
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Saving current model state...")
    except Exception as e:
        print(f"Error during training: {e}")

    # Save the final model (even if interrupted)
    # Use starting timestamp for consistent naming
    save_path_with_timestamp = f"models/ppo_model_{start_timestamp}.zip"

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    model.save(save_path_with_timestamp)
    print(f"Model saved to {save_path_with_timestamp}")

    # Test the trained model
    print("\nTesting trained model...")
    test_model(model, eval_env, n_episodes=1)

    return model


def test_model(model, env, n_episodes=1):
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
