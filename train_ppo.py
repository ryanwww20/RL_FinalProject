"""
PPO (Proximal Policy Optimization) Training Script
Uses Stable-Baselines3 for PPO implementation
"""

import numpy as np
import os
from pathlib import Path
import time
from datetime import datetime
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from envs.Discrete_gym import MinimalEnv
from envs.custom_feature_extractor import MatrixCombinedExtractor
from PIL import Image
from eval import ModelEvaluator

from matplotlib import rcParams

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
    "use_cnn",
    "tensorboard_log",
    "save_path",
    "load_model_path",  # Path to existing model to resume training
}


class TrainingCallback(BaseCallback):
    """
    Callback to record metrics, plot designs, save to CSV, and create GIFs.
    Follows README structure: ppo_model_log_<start_time>/ with img/, plot/, result.csv
    """
    def __init__(self, save_dir, verbose=1, eval_env=None, model_save_path=None, eval_freq=5, 
                 initial_rollout_count=0, rolling_window=10, best_eval_score_resume=-float('inf'), 
                 best_eval_rollout_resume=0):
        super(TrainingCallback, self).__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_count = initial_rollout_count  # Start from previous count if resuming
        self.eval_env = eval_env
        self.model_save_path = model_save_path  # Path to save best model
        self.eval_freq = eval_freq  # Run full evaluation every N rollouts
        self.rolling_window = max(1, int(rolling_window))  # Window size for moving-average plots
        
        # Best model tracking (based on evaluation score)
        # Resume from previous best if resuming training
        self.best_eval_score = best_eval_score_resume
        self.best_eval_rollout = best_eval_rollout_resume
        
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
                f.write('timestamp,rollout_count,type,transmission_score,balance_score,score,reward,similarity_score\n')
        
        # CSV file for evaluation metrics (every eval_freq rollouts)
        self.eval_csv_path = self.save_dir / "eval_metrics.csv"
        if not self.eval_csv_path.exists():
            with open(self.eval_csv_path, 'w') as f:
                f.write('timestamp,rollout_count,transmission_score,balance_score,score,reward,similarity_score\n')
        
        # Store image paths for GIF creation
        # If resuming, try to load existing image paths
        self.design_image_paths = []
        self.distribution_image_paths = []
        
        # If resuming (check if CSV exists, not just rollout_count > 0)
        # This handles cases where CSV read failed but directory exists
        train_csv = self.save_dir / "train_metrics.csv"
        is_resuming = train_csv.exists() and initial_rollout_count > 0
        
        if is_resuming:
            design_dir = self.save_dir / "design_images"
            distribution_dir = self.save_dir / "distribution_images"
            
            if design_dir.exists():
                existing_designs = sorted(design_dir.glob("design_rollout_*.png"))
                self.design_image_paths = [str(p) for p in existing_designs]
                if self.design_image_paths:
                    print(f"Loaded {len(self.design_image_paths)} existing design images")
            
            if distribution_dir.exists():
                existing_distributions = sorted(distribution_dir.glob("distribution_rollout_*.png"))
                self.distribution_image_paths = [str(p) for p in existing_distributions]
                if self.distribution_image_paths:
                    print(f"Loaded {len(self.distribution_image_paths)} existing distribution images")

        # Time tracking
        self.rollout_start_time = time.time()
    
    def _on_step(self) -> bool:
        """Called at each environment step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called when rollout collection ends."""
        self.rollout_count += 1
        current_time = time.time()
        rollout_duration = current_time - self.rollout_start_time
        self.rollout_start_time = current_time  # Reset for next rollout

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
            # Use transmission_score (normalized efficiency) instead of raw total_transmission
            train_transmission_score = sum(m['transmission_score'] for m in all_metrics) / n_envs
            train_balance_score = sum(m['balance_score'] for m in all_metrics) / n_envs
            train_score = sum(m['current_score'] for m in all_metrics) / n_envs
            train_similarity = sum(m.get('similarity_score', 0.0) for m in all_metrics) / n_envs
            
            # Get episode reward from rollout buffer
            train_reward = 0.0
            if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
                rewards = self.model.rollout_buffer.rewards
                episode_rewards = np.sum(rewards, axis=0)
                train_reward = float(np.mean(episode_rewards))
                
        except Exception as e:
            print(f"Warning: Could not get training env metrics: {e}")
            train_transmission_score = 0.0
            train_balance_score = 0.0
            train_score = 0.0
            train_reward = 0.0
            train_similarity = 0.0
            n_envs = 1
        
        # Record training metrics to CSV
        with open(self.train_csv_path, 'a') as f:
            f.write(f'{timestamp},{self.rollout_count},train,{train_transmission_score},{train_balance_score},{train_score},{train_reward},{train_similarity}\n')
        
        # Print training metrics
        print(f"\n[Train] Rollout {self.rollout_count} (avg of {n_envs} envs): "
              f"Trans={train_transmission_score:.4f}, Bal={train_balance_score:.4f}, "
              f"Score={train_score:.4f}, Reward={train_reward:.4f}")
        print(f"        Rollout Duration: {rollout_duration:.2f}s")
        
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
                # Use transmission_score if available, fall back to normalized calculation
                eval_transmission_score = metrics.get('transmission_score', 
                                              metrics.get('total_mode_transmission', 0.0) / metrics.get('input_mode_flux', 1.0) 
                                              if metrics.get('input_mode_flux', 0) > 0 else 0.0)
                eval_balance_score = metrics.get('balance_score', 0.0)
                eval_score = metrics.get('current_score', 0.0)
                eval_total_reward = metrics.get('total_reward', 0.0)
                eval_similarity_score = metrics.get('similarity_score', 0.0)
            else:
                eval_transmission_score = 0.0
                eval_balance_score = 0.0
                eval_score = 0.0
                eval_total_reward = 0.0
                eval_similarity_score = 0.0
            
            # Record evaluation metrics to CSV
            with open(self.eval_csv_path, 'a') as f:
                f.write(f'{timestamp},{self.rollout_count},{eval_transmission_score},{eval_balance_score},{eval_score},{eval_total_reward},{eval_similarity_score}\n')
            
            # Also add to train_csv with 'eval' type for combined plotting
            with open(self.train_csv_path, 'a') as f:
                f.write(f'{timestamp},{self.rollout_count},eval,{eval_transmission_score},{eval_balance_score},{eval_score},{eval_total_reward},{eval_similarity_score}\n')
            
            # Print evaluation metrics
            print(f"[Eval]  Rollout {self.rollout_count} (deterministic): "
                  f"Trans={eval_transmission_score:.4f}, Bal={eval_balance_score:.4f}, "
                  f"Score={eval_score:.4f}, Reward={eval_total_reward:.4f}")
            
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
            
            def plot_metric(metric, title, ylabel, filename):
                with plt.style.context("seaborn-v0_8"), plt.rc_context({"figure.figsize": (12, 6), "font.size": 13}):
                    plt.figure()
                    plotted = False
                    
                    if len(train_df) > 0:
                        t = train_df.sort_values('rollout_count')
                        x = t['rollout_count']
                        y = t[metric]
                        plt.plot(x, y, 'b-o', linewidth=1.5, markersize=3, alpha=0.35, label='Train (raw)')
                        y_ma = y.rolling(window=self.rolling_window, min_periods=1).mean()
                        plt.plot(x, y_ma, 'b-', linewidth=2.2, label=f'Train MA (w={self.rolling_window})')
                        plotted = True
                    
                    if len(eval_df) > 0:
                        e = eval_df.sort_values('rollout_count')
                        x = e['rollout_count']
                        y = e[metric]
                        plt.plot(x, y, 'r-s', linewidth=1.5, markersize=4, alpha=0.35, label='Eval (raw)')
                        y_ma = y.rolling(window=self.rolling_window, min_periods=1).mean()
                        plt.plot(x, y_ma, 'r-', linewidth=2.2, label=f'Eval MA (w={self.rolling_window})')
                        plotted = True
                    
                    if not plotted:
                        plt.close()
                        return
                    
                    plt.xlabel('Rollout Count')
                    plt.ylabel(ylabel)
                    plt.title(title)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.plot_dir / filename, dpi=150, bbox_inches='tight')
                    plt.close()
            
            plot_metric('transmission_score', 'Transmission Score Over Training', 'Transmission Score', 'transmission.png')
            plot_metric('balance_score', 'Balance Score Over Training', 'Balance Score', 'balance.png')
            plot_metric('score', 'Score Over Training', 'Score', 'score.png')
            plot_metric('reward', 'Reward Over Training', 'Reward', 'reward.png')
            
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
    use_cnn=True,
    tensorboard_log="./ppo_tensorboard/",
    save_path="./ppo_model",
    load_model_path=None,  # Path to existing model to resume training
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
        use_cnn: Whether to use CNN feature extractor
        tensorboard_log: Directory for tensorboard logs
        save_path: Path to save the trained model
    """
    # Save starting timestamp for model saving
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine callback directory: use existing one if resuming, otherwise create new
    callback_dir = None
    initial_rollout_count = 0
    best_eval_score_resume = -float('inf')
    best_eval_rollout_resume = 0
    
    if load_model_path and os.path.exists(load_model_path):
        # Try to find existing callback directory from model path
        # Model path format: models/ppo_model_YYYYMMDD_HHMMSS.zip
        model_stem = Path(load_model_path).stem
        if 'ppo_model_' in model_stem:
            model_timestamp = model_stem.replace('ppo_model_', '')
            potential_callback_dir = f"ppo_model_log_{model_timestamp}"
            if os.path.exists(potential_callback_dir):
                callback_dir = potential_callback_dir
                print(f"Found existing callback directory: {callback_dir}")
                
                # Try to read last rollout_count from CSV
                train_csv = Path(callback_dir) / "train_metrics.csv"
                if train_csv.exists():
                    try:
                        df = pd.read_csv(train_csv)
                        if len(df) > 0:
                            # Get max rollout_count from CSV
                            csv_max_rollout = int(df['rollout_count'].max())
                            
                            # Also try to calculate from model timesteps (more accurate)
                            # This will be set after model is loaded
                            initial_rollout_count = csv_max_rollout
                            print(f"CSV shows last rollout: {initial_rollout_count}")
                            
                            # Try to restore best eval score from CSV
                            eval_df = df[df['type'] == 'eval']
                            if len(eval_df) > 0:
                                best_idx = eval_df['score'].idxmax()
                                best_eval_score_resume = float(eval_df.loc[best_idx, 'score'])
                                best_eval_rollout_resume = int(eval_df.loc[best_idx, 'rollout_count'])
                                print(f"Restored best eval: score={best_eval_score_resume:.4f} at rollout {best_eval_rollout_resume}")
                    except Exception as e:
                        print(f"Warning: Could not read rollout count from CSV: {e}")
                        import traceback
                        traceback.print_exc()
    
    # Create new callback directory if not resuming
    if callback_dir is None:
        callback_dir = f"ppo_model_log_{start_timestamp}"
        os.makedirs(callback_dir, exist_ok=True)
        print(f"Creating new callback directory: {callback_dir}")

    # Create vectorized environment (parallel environments)
    # Using DummyVecEnv instead of SubprocVecEnv because Meep simulation objects
    # contain lambda functions that can't be pickled for multiprocessing
    print("Creating environment...")
    
    env_kwargs = {
        "render_mode": None,
    }
    
    env = make_vec_env(
        MinimalEnv,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv
    )

    # Create evaluation environment
    eval_env = MinimalEnv(render_mode=None)
    
    # Define model save path
    save_path_with_timestamp = f"models/ppo_model_{start_timestamp}.zip"
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 128],
            vf=[256, 128],
        ),
    )

    if use_cnn:
        policy_kwargs.update(dict(
            features_extractor_class=MatrixCombinedExtractor,
            features_extractor_kwargs=dict(
                cnn_proj_dim=128,  # Dimension of CNN output after compression
                pixel_num_x=20,
                pixel_num_y=20,
                num_monitors=10,
            ),
        ))

    # Load existing model or create new one
    if load_model_path and os.path.exists(load_model_path):
        print(f"Loading existing model from {load_model_path}...")
        model = PPO.load(load_model_path, env=env)
        
        # Get the number of timesteps already trained
        trained_timesteps = model.num_timesteps
        remaining_timesteps = total_timesteps - trained_timesteps
        
        if remaining_timesteps <= 0:
            print(f"Model has already been trained for {trained_timesteps} timesteps, "
                  f"which exceeds the target {total_timesteps} timesteps.")
            print("Returning loaded model without additional training.")
            return model
        
        print(f"Model has been trained for {trained_timesteps} timesteps.")
        print(f"Continuing training for {remaining_timesteps} more timesteps...")
        
        # Calculate rollout count from timesteps (more accurate than CSV)
        # rollout_timesteps = n_envs * n_steps = 16 * 240 = 3840
        rollout_timesteps = n_envs * n_steps
        calculated_rollout_count = trained_timesteps // rollout_timesteps
        
        # Use the maximum of CSV and calculated rollout count (more accurate)
        if calculated_rollout_count > initial_rollout_count:
            print(f"Calculated rollout count from timesteps: {calculated_rollout_count} "
                  f"(CSV had {initial_rollout_count}, using calculated value)")
            initial_rollout_count = calculated_rollout_count
        else:
            print(f"Using rollout count from CSV: {initial_rollout_count} "
                  f"(calculated would be {calculated_rollout_count})")
        
        # Update total_timesteps to remaining amount
        total_timesteps = remaining_timesteps
    else:
        if load_model_path:
            print(f"Warning: Model path {load_model_path} not found. Creating new model...")
        
        # Create PPO model
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
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

    # Create callback with model save path for best model tracking
    callback = TrainingCallback(
        save_dir=callback_dir, 
        verbose=1, 
        eval_env=eval_env,
        model_save_path=save_path_with_timestamp,
        eval_freq=5,  # Run full evaluation every 5 rollouts
        initial_rollout_count=initial_rollout_count,  # Resume from previous rollout count
        best_eval_score_resume=best_eval_score_resume,  # Resume best eval score
        best_eval_rollout_resume=best_eval_rollout_resume  # Resume best eval rollout
    )

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
    # If resuming from a checkpoint, save to the same path or new timestamp
    if load_model_path and os.path.exists(load_model_path):
        # Option 1: Overwrite the loaded model
        model.save(load_model_path)
        print(f"Model saved to {load_model_path} (updated checkpoint)")
        # Option 2: Also save with new timestamp
        model.save(save_path_with_timestamp)
        print(f"Model also saved to {save_path_with_timestamp}")
    else:
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
