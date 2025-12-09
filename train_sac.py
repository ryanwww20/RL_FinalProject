"""
SAC (Soft Actor-Critic) Training Script
Uses Stable-Baselines3 for SAC implementation
"""

import numpy as np
import os
from pathlib import Path
from datetime import datetime
import sys

print(f"DEBUG: Loading train_sac.py...", flush=True)

import yaml
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from envs.Continuous_gym import MinimalEnv
from PIL import Image
from eval import ModelEvaluator
from envs.custom_gumbel import GumbelSACPolicy

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
}


class TrainingCallback(BaseCallback):
    """
    Callback to record metrics, plot designs, save to CSV, and create GIFs.
    Follows README structure: sac_model_log_<start_time>/ with img/, plot/, result.csv
    """
    def __init__(self, save_dir, verbose=1, eval_env=None, model_save_path=None, eval_freq=5):
        super(TrainingCallback, self).__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_count = 0
        self.eval_env = eval_env
        self.model_save_path = model_save_path  # Path to save best model
        self.eval_freq = eval_freq  # Run full evaluation every N rollouts
        
        # Best model tracking (based on evaluation score)
        self.best_eval_score = -float('inf')
        self.best_eval_rollout = 0
        
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
        
        # CSV file for training metrics (every rollout)
        self.train_csv_path = self.save_dir / "train_metrics.csv"
        if not self.train_csv_path.exists():
            with open(self.train_csv_path, 'w') as f:
                f.write('timestamp,rollout_count,type,transmission,balance_score,score,reward,similarity_score\n')
        
        # CSV file for evaluation metrics (every eval_freq rollouts)
        self.eval_csv_path = self.save_dir / "eval_metrics.csv"
        if not self.eval_csv_path.exists():
            with open(self.eval_csv_path, 'w') as f:
                f.write('timestamp,rollout_count,transmission,balance_score,score,reward,similarity_score\n')
        
        # Store image paths for GIF creation
        self.design_image_paths = []
        self.distribution_image_paths = []
    
    def _on_step(self) -> bool:
        """Called at each environment step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called when rollout collection ends."""
        self.rollout_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ============================================================
        # 1. TRAINING METRICS: Get metrics from all parallel training envs
        # ============================================================
        env = self.training_env
        try:
            if hasattr(env, 'envs'):
                # DummyVecEnv - get metrics from all environments directly
                all_metrics = [e.unwrapped.get_current_metrics() for e in env.envs]
            else:
                # SubprocVecEnv - call on ALL environments
                all_metrics = env.env_method('get_current_metrics')
            
            # Calculate average across all environments
            n_envs = len(all_metrics)
            train_transmission = sum(m['total_transmission'] for m in all_metrics) / n_envs
            train_balance = sum(m['balance_score'] for m in all_metrics) / n_envs
            train_score = sum(m['current_score'] for m in all_metrics) / n_envs
            train_similarity = sum(m.get('similarity_score', 0.0) for m in all_metrics) / n_envs
            
            # Get episode reward for SAC
            # Method 1: Try to get from environment's last_episode_metrics (most accurate)
            train_reward = 0.0
            try:
                # Check if environment stores total_reward in metrics
                for m in all_metrics:
                    if 'total_reward' in m:
                        train_reward = m['total_reward']
                        break
                
                # Method 2: Fallback to replay_buffer if environment doesn't have it
                if train_reward == 0.0 and hasattr(self.model, 'replay_buffer') and self.model.replay_buffer is not None:
                    replay_buffer = self.model.replay_buffer
                    # Get recent rewards from buffer as approximation
                    if hasattr(replay_buffer, 'rewards') and len(replay_buffer.rewards) > 0:
                        recent_size = min(100, len(replay_buffer.rewards))
                        recent_rewards = replay_buffer.rewards[-recent_size:]
                        if len(recent_rewards) > 0:
                            train_reward = float(np.mean(recent_rewards))
                
                # Method 3: Fallback to logger if above methods don't work
                if train_reward == 0.0 and hasattr(self.model, 'logger') and self.model.logger is not None:
                    try:
                        if hasattr(self.model.logger, 'name_to_value'):
                            train_reward = (
                                self.model.logger.name_to_value.get('rollout/ep_rew_mean', 0.0) or
                                self.model.logger.name_to_value.get('train/episode_reward', 0.0)
                            )
                    except Exception:
                        pass
                            
            except Exception:
                # If all methods fail, keep reward as 0.0
                pass
        except Exception as e:
            print(f"Warning: Could not get training env metrics: {e}")
            train_transmission = 0.0
            train_balance = 0.0
            train_score = 0.0
            train_reward = 0.0
            train_similarity = 0.0
            n_envs = 1
        
        # Record training metrics to CSV
        with open(self.train_csv_path, 'a') as f:
            f.write(f'{timestamp},{self.rollout_count},train,{train_transmission},{train_balance},{train_score},{train_reward},{train_similarity}\n')
        
        # Print training metrics
        print(f"\n[Train] Rollout {self.rollout_count} (avg of {n_envs} envs): "
              f"Trans={train_transmission:.4f}, Bal={train_balance:.4f}, "
              f"Score={train_score:.4f}, Reward={train_reward:.4f}")
        
        # ============================================================
        # 2. EVALUATION: Run full eval every eval_freq rollouts
        # ============================================================
        run_eval = (self.rollout_count % self.eval_freq == 0) or (self.rollout_count == 1)
        
        if run_eval and self.eval_env is not None:
            print(f"[Eval]  Running full evaluation...")
            evaluator = ModelEvaluator(self.model, self.eval_env)
            results_df = evaluator.evaluate(n_episodes=1, deterministic=True)
            
            if len(results_df) > 0:
                metrics = results_df.iloc[0]
                eval_transmission = metrics.get('total_mode_transmission', metrics.get('total_transmission', 0.0))
                eval_balance = metrics.get('balance_score', 0.0)
                eval_score = metrics.get('current_score', 0.0)
                eval_reward = metrics.get('total_reward', 0.0)
                eval_similarity = metrics.get('similarity_score', 0.0)
            else:
                eval_transmission = 0.0
                eval_balance = 0.0
                eval_score = 0.0
                eval_reward = 0.0
                eval_similarity = 0.0
            
            # Record evaluation metrics to CSV
            with open(self.eval_csv_path, 'a') as f:
                f.write(f'{timestamp},{self.rollout_count},{eval_transmission},{eval_balance},{eval_score},{eval_reward},{eval_similarity}\n')
            
            # Also add to train_csv with 'eval' type for combined plotting
            with open(self.train_csv_path, 'a') as f:
                f.write(f'{timestamp},{self.rollout_count},eval,{eval_transmission},{eval_balance},{eval_score},{eval_reward},{eval_similarity}\n')
            
            # Print evaluation metrics
            print(f"[Eval]  Rollout {self.rollout_count} (deterministic): "
                  f"Trans={eval_transmission:.4f}, Bal={eval_balance:.4f}, "
                  f"Score={eval_score:.4f}, Reward={eval_reward:.4f}")
            
            # Check if this is the best evaluation score
            if eval_score > self.best_eval_score:
                self.best_eval_score = eval_score
                self.best_eval_rollout = self.rollout_count
                print(f"[Eval]  ★ New best eval score! Saving best model...")
                
                if self.model_save_path:
                    best_model_path = self.model_save_path.replace('.zip', '_best.zip')
                    self.model.save(best_model_path)
                    print(f"[Eval]  ★ Best model saved to: {best_model_path}")
                    
                    # Save best design and distribution
                    best_design_path = self.img_dir / "best_design.png"
                    best_distribution_path = self.img_dir / "best_distribution.png"
                    try:
                        title = f"Best (Rollout {self.rollout_count}, Score={eval_score:.4f})"
                        if hasattr(self.eval_env, 'unwrapped'):
                            self.eval_env.unwrapped.save_design_plot(str(best_design_path), title_suffix=title)
                            self.eval_env.unwrapped.save_distribution_plot(str(best_distribution_path), title_suffix=title)
                        else:
                            self.eval_env.save_design_plot(str(best_design_path), title_suffix=title)
                            self.eval_env.save_distribution_plot(str(best_distribution_path), title_suffix=title)
                    except Exception as e:
                        print(f"[Eval]  Warning: Could not save best plots: {e}")
            else:
                print(f"[Eval]  (Best eval score: {self.best_eval_score:.4f} at rollout {self.best_eval_rollout})")
            
            # Save design/distribution plots from evaluation
            design_path = self.design_dir / f"design_rollout_{self.rollout_count:04d}.png"
            distribution_path = self.distribution_dir / f"distribution_rollout_{self.rollout_count:04d}.png"
            self.design_image_paths.append(str(design_path))
            self.distribution_image_paths.append(str(distribution_path))
            
            title_suffix = f"Rollout {self.rollout_count} (Eval)"
            try:
                if hasattr(self.eval_env, 'unwrapped'):
                    self.eval_env.unwrapped.save_design_plot(str(design_path), title_suffix=title_suffix)
                    self.eval_env.unwrapped.save_distribution_plot(str(distribution_path), title_suffix=title_suffix)
                else:
                    self.eval_env.save_design_plot(str(design_path), title_suffix=title_suffix)
                    self.eval_env.save_distribution_plot(str(distribution_path), title_suffix=title_suffix)
            except Exception as e:
                print(f"Warning: Could not save plots: {e}")
        
        else:
            # No evaluation this rollout - save design from training env
            design_path = self.design_dir / f"design_rollout_{self.rollout_count:04d}.png"
            distribution_path = self.distribution_dir / f"distribution_rollout_{self.rollout_count:04d}.png"
            self.design_image_paths.append(str(design_path))
            self.distribution_image_paths.append(str(distribution_path))
            
            title_suffix = f"Rollout {self.rollout_count} (Train)"
            try:
                if hasattr(env, 'envs'):
                    env.envs[0].unwrapped.save_design_plot(str(design_path), title_suffix=title_suffix)
                    env.envs[0].unwrapped.save_distribution_plot(str(distribution_path), title_suffix=title_suffix)
                else:
                    env.env_method('save_design_plot', str(design_path), title_suffix, indices=[0])
                    env.env_method('save_distribution_plot', str(distribution_path), title_suffix, indices=[0])
            except Exception as e:
                print(f"Warning: Could not save plots: {e}")
        
        # Update GIFs and plots after each rollout
        self._update_gifs_and_plots()

    def _update_gifs_and_plots(self):
        """Update GIFs and metric plots with current data."""
        # Create/update design GIF
        if self.design_image_paths:
            gif_path = self.img_dir / "design.gif"
            self._create_gif(self.design_image_paths, str(gif_path))
        
        # Create/update distribution GIF
        if self.distribution_image_paths:
            gif_path = self.img_dir / "flux.gif"
            self._create_gif(self.distribution_image_paths, str(gif_path))
        
        # Update metric plots
        self._plot_metrics(verbose=False)

    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        print(f"\nTraining ended. Creating final GIFs and plots...")
        
        # Create final design GIF (save to img/design.gif as per README)
        if self.design_image_paths:
            gif_path = self.img_dir / "design.gif"
            self._create_gif(self.design_image_paths, str(gif_path))
            print(f"Design GIF saved to: {gif_path}")
        
        # Create final distribution GIF (save to img/flux.gif as per README)
        if self.distribution_image_paths:
            gif_path = self.img_dir / "flux.gif"
            self._create_gif(self.distribution_image_paths, str(gif_path))
            print(f"Distribution GIF saved to: {gif_path}")
        
        # Plot final metrics from CSV
        self._plot_metrics(verbose=True)
        
        # Note: We keep the image directories for reference
        # If you want to clean them up, uncomment below:
        # import shutil
        # if self.design_dir.exists():
        #     shutil.rmtree(self.design_dir)
        # if self.distribution_dir.exists():
        #     shutil.rmtree(self.distribution_dir)
    
    def _plot_metrics(self, verbose=True):
        """Plot transmission, balance_score, and score from CSV (both train and eval)."""
        if not self.train_csv_path.exists():
            if verbose:
                print("Warning: CSV file not found, cannot plot metrics")
            return
        
        try:
            # Read training CSV
            df = pd.read_csv(self.train_csv_path)
            
            if len(df) == 0:
                if verbose:
                    print("Warning: CSV file is empty, cannot plot metrics")
                return
            
            # Separate train and eval data
            train_df = df[df['type'] == 'train']
            eval_df = df[df['type'] == 'eval']
            
            # Plot transmission (train + eval)
            plt.figure(figsize=(12, 6))
            if len(train_df) > 0:
                plt.plot(train_df['rollout_count'], train_df['transmission'], 
                        'b-', linewidth=1.5, marker='o', markersize=3, alpha=0.7, label='Train (avg)')
            if len(eval_df) > 0:
                plt.plot(eval_df['rollout_count'], eval_df['transmission'], 
                        'r-', linewidth=2, marker='s', markersize=5, label='Eval (deterministic)')
            plt.xlabel('Rollout Count')
            plt.ylabel('Transmission')
            plt.title('Transmission Over Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plot_dir / "transmission.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot balance score (train + eval)
            plt.figure(figsize=(12, 6))
            if len(train_df) > 0:
                plt.plot(train_df['rollout_count'], train_df['balance_score'], 
                        'b-', linewidth=1.5, marker='o', markersize=3, alpha=0.7, label='Train (avg)')
            if len(eval_df) > 0:
                plt.plot(eval_df['rollout_count'], eval_df['balance_score'], 
                        'r-', linewidth=2, marker='s', markersize=5, label='Eval (deterministic)')
            plt.xlabel('Rollout Count')
            plt.ylabel('Balance Score')
            plt.title('Balance Score Over Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plot_dir / "balance.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot score (train + eval)
            plt.figure(figsize=(12, 6))
            if len(train_df) > 0:
                plt.plot(train_df['rollout_count'], train_df['score'], 
                        'b-', linewidth=1.5, marker='o', markersize=3, alpha=0.7, label='Train (avg)')
            if len(eval_df) > 0:
                plt.plot(eval_df['rollout_count'], eval_df['score'], 
                        'r-', linewidth=2, marker='s', markersize=5, label='Eval (deterministic)')
            plt.xlabel('Rollout Count')
            plt.ylabel('Score')
            plt.title('Score Over Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plot_dir / "score.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot reward (train + eval)
            plt.figure(figsize=(12, 6))
            if len(train_df) > 0:
                plt.plot(train_df['rollout_count'], train_df['reward'], 
                        'b-', linewidth=1.5, marker='o', markersize=3, alpha=0.7, label='Train (avg)')
            if len(eval_df) > 0:
                plt.plot(eval_df['rollout_count'], eval_df['reward'], 
                        'r-', linewidth=2, marker='s', markersize=5, label='Eval (deterministic)')
            plt.xlabel('Rollout Count')
            plt.ylabel('Reward')
            plt.title('Reward Over Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plot_dir / "reward.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            if verbose:
                print(f"Metric plots saved to: {self.plot_dir}")
            
        except Exception as e:
            if verbose:
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


def train_sac(
    total_timesteps=100000,
    n_envs=1,
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
):
    """
    Train a SAC agent on the MinimalEnv environment.

    Args:
        total_timesteps: Total number of timesteps to train
        n_envs: Number of parallel environments (usually 1 for SAC)
        learning_rate: Learning rate for optimizer
        buffer_size: Size of the replay buffer
        learning_starts: Number of steps before learning starts
        batch_size: Batch size for training
        tau: Soft update coefficient for target networks
        gamma: Discount factor
        train_freq: Update the model every train_freq steps
        gradient_steps: Number of gradient steps per update
        ent_coef: Entropy coefficient ('auto' for automatic tuning)
        target_update_interval: Update target network every N gradient steps
        target_entropy: Target entropy ('auto' for automatic)
        use_sde: Whether to use State-Dependent Exploration
        tensorboard_log: Directory for tensorboard logs
        save_path: Path to save the trained model
    """
    # Save starting timestamp for model saving
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory for callbacks (following README structure)
    callback_dir = f"sac_model_log_{start_timestamp}"
    os.makedirs(callback_dir, exist_ok=True)

    # Create vectorized environment
    # SAC typically uses a single environment (n_envs=1)
    # Use DummyVecEnv for single env, SubprocVecEnv for multiple envs
    print("Creating environment...")
    vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = make_vec_env(MinimalEnv, n_envs=n_envs,
                       env_kwargs={"render_mode": None},
                       vec_env_cls=vec_env_cls)

    # Create evaluation environment
    eval_env = MinimalEnv(render_mode=None)
    
    # Define model save path
    save_path_with_timestamp = f"models/sac_model_{start_timestamp}.zip"
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    policy_kwargs={
        "net_arch": dict(
            pi=[512, 512],
            qf=[512, 512]
        )
    }

    # Create SAC model
    print("Creating SAC model...")
    # model = SAC(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=learning_rate,
    #     policy_kwargs=policy_kwargs,
    #     buffer_size=buffer_size,
    #     learning_starts=learning_starts,
    #     batch_size=batch_size,
    #     tau=tau,
    #     gamma=gamma,
    #     train_freq=train_freq,
    #     gradient_steps=gradient_steps,
    #     ent_coef=ent_coef,
    #     target_update_interval=target_update_interval,
    #     target_entropy=target_entropy,
    #     use_sde=use_sde,
    #     tensorboard_log=tensorboard_log,
    #     verbose=1
    # )
    model = SAC(
        GumbelSACPolicy,
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        ent_coef=ent_coef,
        target_update_interval=target_update_interval,
        target_entropy=target_entropy,
        use_sde=False,
        tensorboard_log=tensorboard_log,
        verbose=1,
        policy_kwargs=dict(
            temperature=0.5,      # Gumbel 溫度，可調
            net_arch=(512, 512),  # Actor 網路結構
        ),
    )

    # Create callback with model save path for best model tracking
    callback = TrainingCallback(
        save_dir=callback_dir, 
        verbose=1, 
        eval_env=eval_env,
        model_save_path=save_path_with_timestamp,
        eval_freq=5  # Run full evaluation every 5 rollouts
    )

    # Train the model
    print(f"Training SAC agent for {total_timesteps} timesteps...")
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
    model.save(save_path_with_timestamp)
    print(f"Model saved to {save_path_with_timestamp}")
    
    # Print best model info
    if callback.best_eval_rollout > 0:
        print(f"\nBest model was at rollout {callback.best_eval_rollout} with eval score {callback.best_eval_score:.4f}")
        print(f"Best model saved to: {save_path_with_timestamp.replace('.zip', '_best.zip')}")
    else:
        print("\nNo evaluation was run during training.")

    # Final evaluation using ModelEvaluator (same as rollout end)
    print("\nRunning final evaluation...")
    final_eval(model, eval_env, save_dir=callback_dir)

    return model


def final_eval(model, env, save_dir=None):
    """
    Run final evaluation using ModelEvaluator (same logic as rollout end).
    
    Args:
        model: Trained model
        env: Evaluation environment
        save_dir: Directory to save final results (optional)
    """
    evaluator = ModelEvaluator(model, env)
    results_df = evaluator.evaluate(n_episodes=1, deterministic=True)
    
    if len(results_df) > 0:
        metrics = results_df.iloc[0]
        
        # Extract metrics with fallbacks
        transmission = metrics.get('total_mode_transmission', metrics.get('total_transmission', 0.0))
        balance = metrics.get('balance_score', 0.0)
        score = metrics.get('current_score', 0.0)
        total_reward = metrics.get('total_reward', 0.0)
        
        print(f"\n{'='*50}")
        print("Final Evaluation Results:")
        print(f"{'='*50}")
        print(f"  Transmission: {transmission:.4f}")
        print(f"  Balance Score: {balance:.4f}")
        print(f"  Score: {score:.4f}")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"{'='*50}")
        
        # Save final design and distribution plots if save_dir provided
        if save_dir:
            save_path = Path(save_dir)
            img_dir = save_path / "img"
            img_dir.mkdir(exist_ok=True)
            
            # Save final design
            final_design_path = img_dir / "final_design.png"
            try:
                if hasattr(env, 'unwrapped'):
                    env.unwrapped.save_design_plot(str(final_design_path), title_suffix="Final Evaluation")
                else:
                    env.save_design_plot(str(final_design_path), title_suffix="Final Evaluation")
                print(f"Final design saved to: {final_design_path}")
            except Exception as e:
                print(f"Warning: Could not save final design plot: {e}")
            
            # Save final distribution
            final_distribution_path = img_dir / "final_distribution.png"
            try:
                if hasattr(env, 'unwrapped'):
                    env.unwrapped.save_distribution_plot(str(final_distribution_path), title_suffix="Final Evaluation")
                else:
                    env.save_distribution_plot(str(final_distribution_path), title_suffix="Final Evaluation")
                print(f"Final distribution saved to: {final_distribution_path}")
            except Exception as e:
                print(f"Warning: Could not save final distribution plot: {e}")
    else:
        print("Warning: Evaluation returned no results.")


def load_training_config(config_path=None):
    """
    Load train_sac keyword arguments from YAML config.

    Args:
        config_path: Optional override path. Defaults to config.yaml next to this file.

    Returns:
        dict: Filtered kwargs to pass into train_sac.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    print(f"[config] Loading config from: {path.absolute()}", flush=True)
    
    if not path.exists():
        print(
            f"[config] Config file not found at {path}. Using train_sac defaults.")
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Print raw config for debugging
    print(f"[config] Raw YAML content (training section): {data.get('training', {})}")

    training_cfg = data.get("training", {}).get("sac", {}) or {}
    filtered_cfg = {k: v for k, v in training_cfg.items()
                    if k in TRAIN_SAC_KWARGS}

    unknown_keys = sorted(set(training_cfg.keys()) - TRAIN_SAC_KWARGS)
    if unknown_keys:
        print(f"[config] Ignoring unsupported train_sac keys: {unknown_keys}")

    print(f"[config] Parsed train_kwargs: {filtered_cfg}")
    return filtered_cfg


if __name__ == "__main__":
    print("DEBUG: Entering main block", flush=True)
    config_override_path = os.environ.get(CONFIG_ENV_VAR)
    train_kwargs = load_training_config(config_override_path)

    # Train SAC agent
    model = train_sac(**train_kwargs)

    print("\nTraining complete!")
