"""
SAC Training Script with Continuous Relaxation (SIMP)
"""

import numpy as np
import os
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from envs.Continuous_gym_simp import ContinuousSIMPEnv
from PIL import Image
from eval import ModelEvaluator

# Use configuration logic from train_sac.py (re-implemented here for standalone)
CONFIG_ENV_VAR = "TRAINING_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
TRAIN_SAC_KWARGS = {
    "total_timesteps", "n_envs", "learning_rate", "buffer_size", "learning_starts",
    "batch_size", "tau", "gamma", "train_freq", "gradient_steps", "ent_coef",
    "target_update_interval", "target_entropy", "use_sde", "tensorboard_log", "save_path",
}

class SIMPTrainingCallback(BaseCallback):
    """
    Callback for SIMP training:
    - Updates Beta (steepness) over time
    - Records metrics
    - Runs Double Evaluation (Soft/Projected vs Hard/Binary)
    """
    def __init__(self, save_dir, total_timesteps, beta_min=1.0, beta_max=100.0, 
                 verbose=1, eval_env=None, model_save_path=None, eval_freq=5):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_count = 0
        self.eval_env = eval_env
        self.model_save_path = model_save_path
        self.eval_freq = eval_freq
        
        # Beta Scheduling
        self.total_timesteps = total_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.current_beta = beta_min
        
        # Best model tracking
        self.best_eval_score = -float('inf')
        self.best_eval_rollout = 0
        
        # Directory structure
        self.img_dir = self.save_dir / "img"
        self.plot_dir = self.save_dir / "plot"
        self.img_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)
        
        self.design_dir = self.save_dir / "design_images"
        self.distribution_dir = self.save_dir / "distribution_images"
        self.design_dir.mkdir(exist_ok=True)
        self.distribution_dir.mkdir(exist_ok=True)
        
        # CSVs
        self.train_csv_path = self.save_dir / "train_metrics.csv"
        if not self.train_csv_path.exists():
            with open(self.train_csv_path, 'w') as f:
                f.write('timestamp,rollout_count,type,transmission,balance_score,score,reward,similarity_score,beta\n')
        
        self.eval_csv_path = self.save_dir / "eval_metrics.csv"
        if not self.eval_csv_path.exists():
            with open(self.eval_csv_path, 'w') as f:
                f.write('timestamp,rollout_count,type,transmission,balance_score,score,reward,similarity_score,beta\n')
        
        self.design_image_paths = []
        self.distribution_image_paths = []

    def _on_step(self) -> bool:
        """Called at each environment step. Updates Beta."""
        # Calculate progress [0, 1]
        progress = self.num_timesteps / self.total_timesteps
        progress = min(max(progress, 0), 1)
        
        # Exponential schedule: beta_min -> beta_max
        # 1 -> 10 (at 0.5) -> 100 (at 1.0) implies log scale
        self.current_beta = self.beta_min * (self.beta_max / self.beta_min) ** progress
        
        # Update training environment(s)
        if self.training_env is not None:
            self.training_env.env_method('set_beta', self.current_beta)
            
        return True

    def _on_rollout_end(self) -> None:
        """Called when rollout collection ends."""
        self.rollout_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # --- 1. TRAINING METRICS ---
        env = self.training_env
        try:
            # SubprocVecEnv or DummyVecEnv
            all_metrics = env.env_method('get_current_metrics')
            
            n_envs = len(all_metrics)
            train_trans = sum(m['total_transmission'] for m in all_metrics) / n_envs
            train_bal = sum(m['balance_score'] for m in all_metrics) / n_envs
            train_score = sum(m['current_score'] for m in all_metrics) / n_envs
            train_sim = sum(m.get('similarity_score', 0.0) for m in all_metrics) / n_envs
            train_reward = sum(m.get('total_reward', 0.0) for m in all_metrics) / n_envs
            
        except Exception as e:
            print(f"Warning: Could not get training env metrics: {e}")
            train_trans, train_bal, train_score, train_sim, train_reward = 0, 0, 0, 0, 0
            
        with open(self.train_csv_path, 'a') as f:
            f.write(f'{timestamp},{self.rollout_count},train,{train_trans},{train_bal},{train_score},{train_reward},{train_sim},{self.current_beta}\n')

        print(f"\n[Train] Rollout {self.rollout_count}: Score={train_score:.4f}, Beta={self.current_beta:.2f}")

        # --- 2. EVALUATION (Double Pass) ---
        run_eval = (self.rollout_count % self.eval_freq == 0) or (self.rollout_count == 1)
        
        if run_eval and self.eval_env is not None:
            print(f"[Eval]  Running evaluation (Soft & Hard)...")
            evaluator = ModelEvaluator(self.model, self.eval_env)
            
            # --- Pass A: Soft / Projected (Current Beta) ---
            # Sync eval env beta with current training beta
            if hasattr(self.eval_env, 'unwrapped'):
                self.eval_env.unwrapped.set_beta(self.current_beta)
            else:
                self.eval_env.set_beta(self.current_beta)
                
            res_soft = evaluator.evaluate(n_episodes=1, deterministic=True)
            self._log_eval_results(res_soft, timestamp, "eval_soft")
            
            # Extract score for best model tracking
            soft_score = 0
            if len(res_soft) > 0:
                soft_score = res_soft.iloc[0].get('current_score', 0.0)

            # --- Pass B: Hard Clip (Beta -> infinity) ---
            # Set a very high beta (e.g., 1000)
            hard_beta = 1000.0
            if hasattr(self.eval_env, 'unwrapped'):
                self.eval_env.unwrapped.set_beta(hard_beta)
            else:
                self.eval_env.set_beta(hard_beta)
                
            res_hard = evaluator.evaluate(n_episodes=1, deterministic=True)
            self._log_eval_results(res_hard, timestamp, "eval_hard")

            # Restore beta
            if hasattr(self.eval_env, 'unwrapped'):
                self.eval_env.unwrapped.set_beta(self.current_beta)
            else:
                self.eval_env.set_beta(self.current_beta)

            # Best Model Logic (based on Soft score or Hard score? Usually Soft during training)
            # But ultimately we want Hard. Let's track Soft for now as it's the optimization objective.
            if soft_score > self.best_eval_score:
                self.best_eval_score = soft_score
                self.best_eval_rollout = self.rollout_count
                print(f"[Eval]  ★ New best soft score! Saving model...")
                if self.model_save_path:
                    best_path = self.model_save_path.replace('.zip', '_best.zip')
                    self.model.save(best_path)
                    
                    # Save best plots (Soft)
                    self._save_plots(self.eval_env, self.img_dir / "best_design.png", 
                                     self.img_dir / "best_distribution.png", 
                                     f"Best (Rollout {self.rollout_count}, Soft)")

            # Save periodic plots (Soft)
            design_path = self.design_dir / f"design_rollout_{self.rollout_count:04d}.png"
            dist_path = self.distribution_dir / f"distribution_rollout_{self.rollout_count:04d}.png"
            self.design_image_paths.append(str(design_path))
            self.distribution_image_paths.append(str(dist_path))
            self._save_plots(self.eval_env, design_path, dist_path, f"Rollout {self.rollout_count} (Soft)")
            
        else:
            # Save plots from training env
            design_path = self.design_dir / f"design_rollout_{self.rollout_count:04d}.png"
            dist_path = self.distribution_dir / f"distribution_rollout_{self.rollout_count:04d}.png"
            self.design_image_paths.append(str(design_path))
            self.distribution_image_paths.append(str(dist_path))
            
            # Use training env to save
            try:
                env.env_method('save_design_plot', str(design_path), f"Rollout {self.rollout_count}", indices=[0])
                env.env_method('save_distribution_plot', str(dist_path), f"Rollout {self.rollout_count}", indices=[0])
            except Exception:
                pass
        
        self._update_gifs_and_plots()

    def _log_eval_results(self, df, timestamp, type_label):
        if len(df) == 0: return
        
        m = df.iloc[0]
        trans = m.get('total_mode_transmission', m.get('total_transmission', 0.0))
        bal = m.get('balance_score', 0.0)
        score = m.get('current_score', 0.0)
        rew = m.get('total_reward', 0.0)
        sim = m.get('similarity_score', 0.0)
        
        # Log to eval_metrics
        with open(self.eval_csv_path, 'a') as f:
            f.write(f'{timestamp},{self.rollout_count},{type_label},{trans},{bal},{score},{rew},{sim},{self.current_beta}\n')
            
        # Log to train_metrics (for combined plotting)
        with open(self.train_csv_path, 'a') as f:
            f.write(f'{timestamp},{self.rollout_count},{type_label},{trans},{bal},{score},{rew},{sim},{self.current_beta}\n')

        print(f"[Eval]  {type_label}: Trans={trans:.4f}, Score={score:.4f}")

    def _save_plots(self, env, d_path, f_path, title):
        try:
            if hasattr(env, 'unwrapped'):
                env.unwrapped.save_design_plot(str(d_path), title_suffix=title)
                env.unwrapped.save_distribution_plot(str(f_path), title_suffix=title)
            else:
                env.save_design_plot(str(d_path), title_suffix=title)
                env.save_distribution_plot(str(f_path), title_suffix=title)
        except Exception as e:
            print(f"Warning: Could not save plots: {e}")

    def _update_gifs_and_plots(self):
        # Create GIFs
        if self.design_image_paths:
            self._create_gif(self.design_image_paths, str(self.img_dir / "design.gif"))
        if self.distribution_image_paths:
            self._create_gif(self.distribution_image_paths, str(self.img_dir / "flux.gif"))
            
        # Plot Metrics
        self._plot_metrics()

    def _create_gif(self, paths, output_path):
        images = []
        for p in paths:
            if os.path.exists(p):
                images.append(Image.open(p))
        if images:
            images[0].save(output_path, save_all=True, append_images=images[1:], duration=500, loop=0)

    def _plot_metrics(self):
        if not self.train_csv_path.exists(): return
        try:
            df = pd.read_csv(self.train_csv_path)
            if len(df) == 0: return
            
            # Plot Score
            plt.figure(figsize=(10, 6))
            train_df = df[df['type'] == 'train']
            soft_df = df[df['type'] == 'eval_soft']
            hard_df = df[df['type'] == 'eval_hard']
            
            if len(train_df): plt.plot(train_df['rollout_count'], train_df['score'], 'b-', alpha=0.5, label='Train')
            if len(soft_df): plt.plot(soft_df['rollout_count'], soft_df['score'], 'g-s', label='Eval (Soft)')
            if len(hard_df): plt.plot(hard_df['rollout_count'], hard_df['score'], 'r-^', label='Eval (Hard)')
            
            # Plot Beta on secondary axis
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            if len(train_df):
                ax2.plot(train_df['rollout_count'], train_df['beta'], 'k--', alpha=0.3, label='Beta')
                ax2.set_ylabel('Beta')
            
            ax1.set_xlabel('Rollout')
            ax1.set_ylabel('Score')
            ax1.set_title('Score & Beta over Training')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(self.plot_dir / "score_beta.png")
            plt.close()
            
        except Exception:
            pass

def load_training_config(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists(): return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    training_cfg = data.get("training", {}).get("sac", {}) or {}
    return {k: v for k, v in training_cfg.items() if k in TRAIN_SAC_KWARGS}

def train_sac2(
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
    tensorboard_log="./sac_simp_tensorboard/",
    save_path="./sac_simp_model",
):
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callback_dir = f"sac_simp_log_{start_timestamp}"
    os.makedirs(callback_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create SIMP Envs
    print("Creating SIMP environment...")
    vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    # Initialize with beta=1.0
    env = make_vec_env(ContinuousSIMPEnv, n_envs=n_envs,
                       env_kwargs={"render_mode": None, "beta": 1.0},
                       vec_env_cls=vec_env_cls)
    
    # Eval Env
    eval_env = ContinuousSIMPEnv(render_mode=None, beta=1.0)
    
    save_path_with_timestamp = f"models/sac_simp_{start_timestamp}.zip"
    
    policy_kwargs = dict(net_arch=dict(pi=[512, 512], qf=[512, 512]))
    
    print("Creating SAC model...")
    model = SAC(
        "MlpPolicy", env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
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
        use_sde=use_sde,
        tensorboard_log=tensorboard_log,
        verbose=1
    )
    
    callback = SIMPTrainingCallback(
        save_dir=callback_dir,
        total_timesteps=total_timesteps,
        beta_min=1.0,
        beta_max=100.0, # Target 100 at end (approx 10 at 50%)
        eval_env=eval_env,
        model_save_path=save_path_with_timestamp,
        eval_freq=5
    )
    
    print(f"Training SAC (SIMP) for {total_timesteps} steps...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    except KeyboardInterrupt:
        print("Interrupted!")
        model.save(save_path_with_timestamp)
        
    model.save(save_path_with_timestamp)
    print(f"Saved to {save_path_with_timestamp}")
    
    return model

if __name__ == "__main__":
    config_override_path = os.environ.get(CONFIG_ENV_VAR)
    train_kwargs = load_training_config(config_override_path)
    train_sac2(**train_kwargs)

