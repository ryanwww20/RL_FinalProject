import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from typing import Dict, List, Any, Optional, Tuple, Union
import gymnasium as gym

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

class ModelEvaluator:
    """
    Class to evaluate trained RL models on the waveguide environment.
    """
    def __init__(self, model: Union[PPO, SAC], env: gym.Env, rolling_window: int = 5):
        """
        Initialize the evaluator.
        
        Args:
            model: The trained Stable-Baselines3 model (PPO or SAC)
            env: The evaluation environment
            rolling_window: Window size for moving-average smoothing in plots
        """
        self.model = model
        self.env = env
        self.rolling_window = max(1, int(rolling_window))
        self.results = []
        
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> pd.DataFrame:
        """
        Run evaluation episodes.
        
        Args:
            n_episodes: Number of episodes to run
            deterministic: Whether to use deterministic actions
            
        Returns:
            DataFrame containing metrics for each episode
        """
        print(f"Starting evaluation for {n_episodes} episodes...")
        
        episode_metrics = []
        
        for i in range(n_episodes):
            # Use different seed for each episode to ensure diversity
            # Start from seed=0 and increment for each episode
            obs, info = self.env.reset(seed=i)
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            # These will store the final step's info which usually contains the result metrics
            final_info = {}
            
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
                # Update final_info with the latest info
                # In MinimalEnv, the interesting metrics are updated in step()
                # and available in the info dict
                if info:
                    final_info.update(info)
            
            print(f"Episode {i+1}/{n_episodes} completed. Reward: {total_reward:.4f}")
            
            # Collect metrics for this episode
            metric = {
                "episode": i + 1,
                "total_reward": total_reward,
                "steps": steps,
                "timestamp": pd.Timestamp.now()
            }
            
            # Add specific environment metrics if available
            # Based on MinimalEnv, we expect:
            # total_mode_transmission, transmission_score, balance_score, current_score, etc.
            keys_to_extract = [
                "total_mode_transmission", 
                "transmission_score", 
                "balance_score", 
                "current_score",
                "output_mode_1_ratio",
                "output_mode_2_ratio",
                "mode_loss_ratio",
                # Discrete environment metrics
                "total_transmission",
                "transmission_1",
                "transmission_2",
                "diff_transmission",
                "similarity_score"
            ]
            
            for key in keys_to_extract:
                if key in final_info:
                    metric[key] = final_info[key]
                elif hasattr(self.env.unwrapped, 'last_score') and key == 'current_score':
                     metric[key] = self.env.unwrapped.last_score
            
            # Fallback for transmission/balance if not in info but calculable
            # (Note: MinimalEnv logic puts them in info['metrics'] or similar? 
            # Looking at MinimalEnv source, it puts them in self._step_metrics and returns as info
            # so they should be in final_info directly if step() returns them.)
            
            episode_metrics.append(metric)
            
        self.df = pd.DataFrame(episode_metrics)
        return self.df
    
    def save_results(self, output_dir: str):
        """
        Save evaluation results to CSV and generate plots.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_path / "evaluation_results.csv"
        self.df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Generate Summary Statistics
        summary = self.df.describe()
        summary_path = output_path / "evaluation_summary.csv"
        summary.to_csv(summary_path)
        print(f"Summary stats saved to {summary_path}")
        print("\nEvaluation Summary:")
        cols_to_show = ['total_reward', 'current_score', 'total_mode_transmission', 'balance_score']
        # Filter only existing columns
        existing_cols = [c for c in cols_to_show if c in summary.columns]
        if not existing_cols and 'total_transmission' in summary.columns:
             existing_cols = ['total_reward', 'current_score', 'total_transmission', 'balance_score']
             
        if existing_cols:
            print(summary[existing_cols])

        # Generate Plots
        self._plot_metric(output_path, 'current_score', 'Score per Episode')
        self._plot_metric(output_path, 'total_mode_transmission', 'Transmission per Episode')
        self._plot_metric(output_path, 'total_transmission', 'Transmission per Episode') # For discrete
        self._plot_metric(output_path, 'balance_score', 'Balance Score per Episode')
        self._plot_metric(output_path, 'total_reward', 'Total Reward per Episode')

        # Save design/distribution visuals if environment exposes helpers (parity with training logs)
        try:
            design_path = output_path / "design.png"
            distribution_path = output_path / "distribution.png"
            env_obj = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

            if hasattr(env_obj, "save_design_plot"):
                env_obj.save_design_plot(str(design_path), title_suffix="Evaluation")
                print(f"Design plot saved to {design_path}")

            if hasattr(env_obj, "save_distribution_plot"):
                env_obj.save_distribution_plot(str(distribution_path), title_suffix="Evaluation")
                print(f"Distribution plot saved to {distribution_path}")
        except Exception as e:
            print(f"Warning: could not save design/distribution plots: {e}")

    def _plot_metric(self, output_path: Path, metric: str, title: str):
        """Helper to plot a single metric."""
        if metric not in self.df.columns:
            return
        
        # Prepare x-axis
        x = self.df['episode'] if 'episode' in self.df.columns else pd.RangeIndex(1, len(self.df) + 1)

        # Moving-average smoothing (still show raw points with light alpha)
        series = self.df[metric]
        smooth = series.rolling(window=self.rolling_window, min_periods=1).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(x, series, 'o-', linewidth=1.5, alpha=0.35, label='raw')
        plt.plot(x, smooth, '-', linewidth=2.5, label=f'MA (window={self.rolling_window})')
        plt.xlabel('Episode')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / f"{metric}.png")
        plt.close()

def load_model(model_path: str, algo: str, env: gym.Env):
    """
    Load a trained model.
    
    Args:
        model_path: Path to the .zip file
        algo: Algorithm name ('ppo' or 'sac')
        env: Environment to attach to the model
        
    Returns:
        The loaded model
    """
    if algo.lower() == 'ppo':
        return PPO.load(model_path, env=env)
    elif algo.lower() == 'sac':
        return SAC.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Use 'ppo' or 'sac'.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model for waveguide design.")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model .zip file")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"], default="ppo", help="RL algorithm used (ppo or sac)")
    parser.add_argument("--env_type", type=str, choices=["continuous", "discrete"], default="discrete", 
                        help="Environment type (continuous or discrete). Default: continuous")
    parser.add_argument("--n_episodes", type=int, default=1, help="Number of evaluation episodes (default: 1)")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Verify model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
        
    print(f"Initializing {args.env_type.capitalize()} Environment...")
    
    # Import the correct environment based on argument
    if args.env_type == "continuous":
        from envs.Continuous_gym import MinimalEnv
    else:
        from envs.Discrete_gym import MinimalEnv
        
    # Initialize environment
    # Note: MinimalEnv writes logs to ppo_model_logs by default. 
    env = MinimalEnv(render_mode=None)
    
    print(f"Loading {args.algo.upper()} model from {args.model_path}...")
    try:
        model = load_model(args.model_path, args.algo, env)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try to hint user if they might have picked the wrong env/algo combo
        if "size mismatch" in str(e) or "Mismatch" in str(e):
            print("\nHint: This might be due to mismatched environment observation/action spaces.")
            print(f"You selected --env_type {args.env_type}. Ensure this matches the trained model.")
        sys.exit(1)
        
    print("Model loaded successfully.")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, env)
    
    if args.n_episodes > 1:
        print(f"Note: Running {args.n_episodes} episodes with deterministic actions and fixed start state.")
        print("      Expect identical results across episodes.")

    # Run evaluation
    evaluator.evaluate(n_episodes=args.n_episodes, deterministic=True)
    
    # Save results
    save_path = Path(args.output_dir) / f"{args.algo}_{Path(args.model_path).stem}"
    print(f"Saving results to {save_path}...")
    evaluator.save_results(str(save_path))
    
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
