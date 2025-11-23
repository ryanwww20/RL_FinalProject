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

TARGET_FLUX = np.zeros(100)
TARGET_FLUX[20:40] = 2.0
TARGET_FLUX[60:80] = 2.0
TARGET_FLUX -= 1.0


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

        # Action space: binary array of length 50 (0/1 values)
        self.action_space = spaces.MultiBinary(self.action_size)

        # Initialize state
        self.state = None
        self.render_mode = render_mode
        self.material_matrix = np.zeros(
            (config.simulation.pixel_num_x, config.simulation.pixel_num_y))
        self.material_matrix_idx = 0
        self.max_steps = config.environment.max_steps
        self.simulation = WaveguideSimulation()

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
        self.material_matrix[self.material_matrix_idx] = action
        self.material_matrix_idx += 1

        input_flux, output_flux_1, output_flux_2, output_all_flux, ez_data = self.simulation.calculate_flux(
            self.material_matrix)

        reward = self.get_reward(output_all_flux)

        # Check if episode is done
        terminated = self.material_matrix_idx >= self.max_steps  # Goal reached
        if terminated:
            # Save reward to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(self.log_dir, 'episode_rewards.csv')
            with open(csv_path, 'a') as f:
                f.write(f'{timestamp}, {reward}\n')

            # Use simulation methods for plotting
            flux_img_path = os.path.join(
                self.log_dir, 'flux_images', f'flux_distribution_{timestamp}.png')
            self.simulation.plot_distribution(
                output_all_flux=output_all_flux,
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

            print(
                f'Input Flux: {input_flux}, Output Flux 1: {output_flux_1}, Output Flux 2: {output_flux_2}')
        truncated = False   # Time limit exceeded

        # Get observation - return the current flux distribution as observation
        # This gives the agent feedback about the current state
        if self.material_matrix_idx > 0:
            # Calculate current flux as observation
            observation = output_all_flux.copy()

        else:
            # Initial state: return zeros
            observation = np.zeros(self.obs_size, dtype=np.float32)

        # Info dictionary (can contain debugging info)
        info = {}

        return observation, reward, terminated, truncated, info

    def get_reward(self, flux_data):
        return np.sum(flux_data * TARGET_FLUX)/np.sum(flux_data)
