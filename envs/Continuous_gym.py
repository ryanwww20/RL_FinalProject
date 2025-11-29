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

e = 1e-6
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
        self.reward_history = []
        self.current_score_history = []

        # Determine project root and log paths
        # Assuming this file is in envs/ and project root is one level up
        current_file_path = os.path.abspath(__file__)
        self.project_root = os.path.dirname(os.path.dirname(current_file_path))
        self.log_dir = os.path.join(self.project_root, 'ppo_model_logs')

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
        # print(
        #     f"Step {self.material_matrix_idx} with action: {action[:5]}", end="\r")
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

        for i in range(self.action_size):
            if action[i] > (config.simulation.silicon_index + config.simulation.silica_index) / 2:
                self.material_matrix[self.material_matrix_idx, i] = 1
            else:
                self.material_matrix[self.material_matrix_idx, i] = 0
        self.material_matrix_idx += 1
        output_plane_x = -1 + (self.material_matrix_idx+0.1) * \
            config.simulation.pixel_size

        input_flux, output_flux_1, output_flux_2, output_all_flux, ez_data, input_mode, output_mode_1, output_mode_2 = self.simulation.calculate_flux(
            self.material_matrix)
        print('=============== Flux Results ===============')
        print(f'Input Flux: {input_flux:.4f}')
        print(f'Output Flux 1: {output_flux_1:.4f}')
        print(f'Output Flux 2: {output_flux_2:.4f}')
        print(f'Output Flux 1 ratio: {output_flux_1/input_flux*100:.2f}%\nOutput Flux 2 ratio: {output_flux_2/input_flux*100:.2f}%\nFlux Loss ratio: {(input_flux - (output_flux_1 + output_flux_2))/input_flux*100:.2f}%')
        print(f'Output_all_flux: {sum(output_all_flux)/input_flux*100:.2f}%')
        print('============================================')
        print('=============== Mode Results ===============')
        print(f'Input Mode (TE0): {input_mode:.4e}')
        print(f'Output Mode 1 (TE0): {output_mode_1:.4e}')
        print(f'Output Mode 2 (TE0): {output_mode_2:.4e}')
        print(f'Output Mode 1 ratio: {output_mode_1/input_mode*100:.2f}%')
        print(f'Output Mode 2 ratio: {output_mode_2/input_mode*100:.2f}%')
        print(f'Total Mode transmission: {(output_mode_1 + output_mode_2)/input_mode*100:.2f}%')
        print(f'Mode Loss ratio: {(input_mode - (output_mode_1 + output_mode_2))/input_mode*100:.2f}%')
        print('============================================')

        # Use MODE coefficients for reward calculation (instead of raw flux)
        current_score, reward = self.get_reward(
            input_mode, output_mode_1, output_mode_2)
        current_score, reward = self.normalize_reward(current_score, reward)

        # Save reward to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.log_dir, 'episode_rewards.csv')
        csv_path_terminated = os.path.join(self.log_dir, 'episode_rewards_terminated.csv')
        # add column names for the first time
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write(
                    'timestamp, current_score, reward, output_mode_1_ratio, output_mode_2_ratio, output_flux_1_ratio, output_flux_2_ratio, mode_loss_ratio, flux_loss_ratio\n')
        with open(csv_path, 'a') as f:
            f.write(f'{timestamp}, {current_score}, {reward}, {output_mode_1/input_mode}, {output_mode_2/input_mode}, {output_flux_1/input_flux}, {output_flux_2/input_flux}, {(input_mode - (output_mode_1 + output_mode_2))/input_mode}, {(input_flux - (output_flux_1 + output_flux_2))/input_flux}\n')
       
        terminated = self.material_matrix_idx >= self.max_steps  # Goal reached
        if terminated:
            self.reward_history = []
            self.current_score_history = []
            # Use simulation methods for plotting
            flux_img_path = os.path.join(
                self.log_dir, 'flux_images', f'flux_distribution_{timestamp}.png')
            self.simulation.plot_distribution(
                state_flux=output_all_flux,
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

            if not os.path.exists(csv_path_terminated):
                with open(csv_path_terminated, 'w') as f:
                    f.write('timestamp, current_score, reward, output_mode_1_ratio, output_mode_2_ratio, output_flux_1_ratio, output_flux_2_ratio, mode_loss_ratio, flux_loss_ratio\n')
            with open(csv_path_terminated, 'a') as f:
                f.write(f'{timestamp}, {current_score}, {reward}, {output_mode_1/input_mode}, {output_mode_2/input_mode}, {output_flux_1/input_flux}, {output_flux_2/input_flux}, {(input_mode - (output_mode_1 + output_mode_2))/input_mode}, {(input_flux - (output_flux_1 + output_flux_2))/input_flux}\n')
        truncated = False   # Time limit exceeded

        # Get observation - return the current flux distribution as observation
        # This gives the agent feedback about the current state
        if self.material_matrix_idx > 0:
            # Calculate current flux as observation
            observation = output_all_flux.copy()/input_flux

        else:
            # Initial state: return zeros
            observation = np.zeros(self.obs_size, dtype=np.float32)

        # Info dictionary with custom metrics for WandB logging
        info = self._step_metrics if hasattr(self, '_step_metrics') else {}
        observation = np.append(observation, self.material_matrix_idx)

        return observation, reward, terminated, truncated, info

    def get_reward(self, input_mode, output_mode_1, output_mode_2):
        """
        Calculate reward using MODE coefficients (fundamental mode transmission).
        This is more accurate than raw flux as it only counts light coupled into the waveguide mode.
        
        Args:
            input_mode: Input mode coefficient (|alpha_in|^2)
            output_mode_1: Output mode 1 coefficient (|alpha_out1|^2)
            output_mode_2: Output mode 2 coefficient (|alpha_out2|^2)
        """
        total_mode_transmission = (output_mode_1 + output_mode_2) / input_mode
        transmission_score = min(max(total_mode_transmission, 0), 1)
        
        # Avoid division by zero
        if (output_mode_1 + output_mode_2) > 0:
            diff_ratio = abs(output_mode_1 - output_mode_2) / (output_mode_1 + output_mode_2)
        else:
            diff_ratio = 1.0  # If no mode transmission, balance is worst
        balance_score = max(1 - diff_ratio, 0)

        current_score = transmission_score * balance_score
        reward = current_score - self.last_score if self.last_score is not None else 0
        self.reward_history.append(reward)
        self.current_score_history.append(current_score)
        self.last_score = current_score

        # Calculate ratios for logging (mode-to-mode transmission)
        output_mode_1_ratio = output_mode_1 / input_mode
        output_mode_2_ratio = output_mode_2 / input_mode
        mode_loss_ratio = (input_mode - (output_mode_1 + output_mode_2)) / input_mode

        print(f"[MODE] Total transmission: {total_mode_transmission:.4e}, Transmission score: {transmission_score:.4e}, Balance score: {balance_score:.4e}, Current score: {current_score:.4e}, Reward: {reward:.4e}")

        # Store metrics for info dict
        self._step_metrics = {
            "total_mode_transmission": total_mode_transmission,
            "transmission_score": transmission_score,
            "balance_score": balance_score,
            "current_score": current_score,
            "output_mode_1_ratio": output_mode_1_ratio,
            "output_mode_2_ratio": output_mode_2_ratio,
            "mode_loss_ratio": mode_loss_ratio,
        }

        return current_score, reward

    def normalize_reward(self, current_score, reward):
        norm_reward = (reward - np.mean(self.reward_history)) / math.sqrt(np.std(self.reward_history) ** 2 + e)
        norm_current_score = (current_score - np.mean(self.current_score_history)) / math.sqrt(np.std(self.current_score_history) ** 2 + e)
        return norm_current_score, norm_reward