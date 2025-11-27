"""
Minimal OpenAI Gymnasium Environment Template
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.meep_simulation import WaveguideSimulation
from datetime import datetime
from config import config
import os
import math

e = 1e-8

class MinimalEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        """
        Initialize the environment.

        Args:
            render_mode: "human" for GUI, "rgb_array" for image, None for no rendering
        """
        super().__init__()

        self.obs_size = config.environment.obs_size
        self.action_size = config.environment.action_size
        # Define observation and action spaces
        # State is an array of 100
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        index_diff = config.simulation.silicon_index - config.simulation.silica_index
        self.action_space = spaces.Box(
            low=config.simulation.silica_index - index_diff/2,
            high=config.simulation.silicon_index + index_diff/2,
            shape=(self.action_size,),
            dtype=np.float32
        )

        # Initialize state
        self.state = None
        self.render_mode = render_mode
        self.material_matrix = np.zeros(
            (config.simulation.pixel_num_x, config.simulation.pixel_num_y))
        self.material_matrix_idx = 0
        self.max_steps = config.environment.max_steps
        self.simulation = WaveguideSimulation()
        self.last_score = None

        # Determine project root and log paths
        # Assuming this file is in envs/ and project root is one level up
        current_file_path = os.path.abspath(__file__)
        self.project_root = os.path.dirname(os.path.dirname(current_file_path))
        self.log_dir = os.path.join(self.project_root, 'ppo_model_logs')

        self.reward_history = []
        self.current_score_history = []

        # Ensure base log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'flux_images'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'field_images'), exist_ok=True)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Reset material matrix and index
        self.material_matrix = np.zeros(
            (config.simulation.pixel_num_x, config.simulation.pixel_num_y))
        self.material_matrix_idx = 0
        self.last_score = None
        # Return initial observation (zeros since no material set yet)
        observation = np.zeros(self.obs_size, dtype=np.float32)
        info = {}

        return observation, info

    def step(self, action):
        print(
            f"Step {self.material_matrix_idx} with action: {action[:5]}", end="\r")
        """
        Execute one step in the environment.

        Args:
            action: Action to take (must be in action_space)

        Returns:
            observation: New observation after taking action
            reward: Reward for this step
            terminated: Whether episode has ended (goal reached)
            truncated: Whether episode was truncated (time limit)
            info: Additional information dictionary
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Action is a binary array of length 50
        # Action is a binary array of length 50
        for i in range(self.action_size):
            if action[i] > (config.simulation.silicon_index + config.simulation.silica_index) / 2:
                self.material_matrix[self.material_matrix_idx, i] = 1
            else:
                self.material_matrix[self.material_matrix_idx, i] = 0
        self.material_matrix_idx += 1

        input_flux, output_flux_1, output_flux_2, output_all_flux, ez_data = self.simulation.calculate_flux(
            self.material_matrix)
        print(f"Input flux: {input_flux:.4e}")
        print(f"Output flux 1: {output_flux_1:.4e}")
        print(f"Output flux 2: {output_flux_2:.4e}")

        current_score, reward = self.get_reward(
            input_flux, output_flux_1, output_flux_2)
        norm_current_score, norm_reward = self.normalize_reward(current_score, reward)
        # Save reward to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.log_dir, 'episode_rewards.csv')
        # add column names for the first time
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write(
                    'timestamp, current_score, reward, norm_current_score, norm_reward, output_flux_1_ratio, output_flux_2_ratio, loss_ratio\n')
        with open(csv_path, 'a') as f:
            f.write(f'{timestamp}, {current_score}, {reward}, {norm_current_score}, {norm_reward}, {output_flux_1/input_flux}, {output_flux_2/input_flux}, {(input_flux - (output_flux_1 + output_flux_2))/input_flux}\n')
        # Check if episode is done
        terminated = self.material_matrix_idx >= self.max_steps  # Goal reached
        if terminated:
            # Use simulation methods for plotting
            flux_img_path = os.path.join(
                self.log_dir, 'flux_images', f'flux_distribution_{timestamp}.png')
            self.simulation.plot_distribution(
                output_all_flux=output_all_flux,
                input_flux=input_flux,
                save_path=flux_img_path,
                show_plot=False
            )

            field_img_path = os.path.join(
                self.log_dir, 'field_images', f'field_distribution_{timestamp}.png')
            self.simulation.plot_design(
                material_matrix=self.material_matrix,
                save_path=field_img_path,
                show_plot=False
            )
            self.reward_history = []
            self.current_score_history = []

            print(
                f'Output Flux 1: {output_flux_1/input_flux:.2f}, Output Flux 2: {output_flux_2/input_flux:.2f}, Loss: {(input_flux - (output_flux_1 + output_flux_2))/input_flux:.2f}')

        truncated = False   # Time limit exceeded

        # Get observation - return the current flux distribution as observation
        # This gives the agent feedback about the current state
        if self.material_matrix_idx > 0:
            # Calculate current flux as observation
            observation = output_all_flux.copy()

        else:
            # Initial state: return zeros
            observation = np.zeros(self.obs_size, dtype=np.float32)

        # Info dictionary with custom metrics for WandB logging
        info = self._step_metrics if hasattr(self, '_step_metrics') else {}
        observation = np.append(observation, self.material_matrix_idx)

        return observation, norm_reward, terminated, truncated, info

    def get_reward(self, input_flux, output_flux_1, output_flux_2):
        current_score = -((output_flux_1 - input_flux*0.5)**2 +
                          (output_flux_2 - input_flux*0.5)**2)
        reward = current_score - self.last_score if self.last_score is not None else 0
        self.reward_history.append(reward)
        self.current_score_history.append(current_score)
        self.last_score = current_score

        # Calculate metrics for logging
        total_transmission = (output_flux_1 + output_flux_2) / input_flux
        transmission_score = min(max(total_transmission, 0), 1)
        diff_ratio = abs(output_flux_1 - output_flux_2) / (output_flux_1 + output_flux_2) if (output_flux_1 + output_flux_2) > 0 else 0
        balance_score = max(1 - diff_ratio, 0)
        output_flux_1_ratio = output_flux_1 / input_flux
        output_flux_2_ratio = output_flux_2 / input_flux
        loss_ratio = (input_flux - (output_flux_1 + output_flux_2)) / input_flux

        print(f"Total transmission: {total_transmission:.4e}, Transmission score: {transmission_score:.4e}, Balance score: {balance_score:.4e}, Current score: {current_score:.4e}, Reward: {reward:.4e}")

        # Store metrics for info dict
        self._step_metrics = {
            "total_transmission": total_transmission,
            "transmission_score": transmission_score,
            "balance_score": balance_score,
            "current_score": current_score,
            "output_flux_1_ratio": output_flux_1_ratio,
            "output_flux_2_ratio": output_flux_2_ratio,
            "loss_ratio": loss_ratio,
        }

        return current_score, reward

    def normalize_reward(self, current_score, reward):
        norm_reward = (reward - np.mean(self.reward_history)) / math.sqrt(np.std(self.reward_history) ** 2 + e)
        norm_current_score = (current_score - np.mean(self.current_score_history)) / math.sqrt(np.std(self.current_score_history) ** 2 + e)
        return norm_current_score, norm_reward