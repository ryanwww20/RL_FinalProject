"""
Minimal OpenAI Gymnasium Environment Template
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.meep_simulation import WaveguideSimulation
import matplotlib.pyplot as plt
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
        self.material_matrix = np.zeros((config.simulation.pixel_num_x, config.simulation.pixel_num_y))
        self.material_matrix_idx = 0
        self.max_steps = config.environment.max_steps
        self.simulation = WaveguideSimulation()

        # Determine project root and log paths
        # Assuming this file is in envs/ and project root is one level up
        current_file_path = os.path.abspath(__file__)
        self.project_root = os.path.dirname(os.path.dirname(current_file_path))
        self.log_dir = os.path.join(self.project_root, 'sac_model_logs')
        
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
        self.material_matrix = np.zeros((config.simulation.pixel_num_x, config.simulation.pixel_num_y))
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
        for i in range(self.action_size):
            if action[i] > (config.simulation.silicon_index + config.simulation.silica_index) / 2:
                self.material_matrix[self.material_matrix_idx, i] = 1
            else:
                self.material_matrix[self.material_matrix_idx, i] = 0
        
        self.material_matrix_idx += 1

        output_flux, ez_data = self.simulation.calculate_flux(
            self.material_matrix)

        reward = self.get_reward(output_flux)

        # Check if episode is done
        terminated = self.material_matrix_idx >= self.max_steps # Goal reached
        if terminated:
            self.reward_plot(reward, output_flux)
            self.field_result_plot(ez_data)
        truncated = False   # Time limit exceeded

        # Get observation - return the current flux distribution as observation
        # This gives the agent feedback about the current state
        if self.material_matrix_idx > 0:
            # Calculate current flux as observation
            observation = output_flux.copy()

        else:
            # Initial state: return zeros
            observation = np.zeros(self.obs_size, dtype=np.float32)

        # Info dictionary (can contain debugging info)
        info = {}

        return observation, reward, terminated, truncated, info


    def get_reward(self, flux_data):
        return np.sum(flux_data * TARGET_FLUX)/np.sum(flux_data)

    def reward_plot(self, reward, flux_data):
        # save reward to csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_path = os.path.join(self.log_dir, 'episode_rewards.csv')
        with open(csv_path, 'a') as f:
            f.write(f'{timestamp}, {reward}\n')

        plt.figure(figsize=(10, 6))
        plt.plot(flux_data, 'b-', linewidth=2, label='Current Flux')
        # plt.plot(TARGET_FLUX, 'r--', linewidth=2,
        #          alpha=0.7, label='Target Flux')
        plt.xlabel('Detector Index')
        plt.ylabel('Flux')
        plt.title(f'Flux Distribution at Step {self.material_matrix_idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        img_path = os.path.join(self.log_dir, 'flux_images', f'flux_distribution_{timestamp}.png')
        plt.savefig(img_path)
        plt.close()

    def field_result_plot(self, ez_data):
        # plot field results with marked material matrix
        try:
            pixel_size = self.simulation.pixel_size
            extent = [-3, 3, -2, 2]
            plt.figure(figsize=(10, 6))
            plt.imshow(ez_data, interpolation='spline36', cmap='RdBu',
                       aspect='auto', extent=extent, origin='lower')
            plt.colorbar(label='Ez (electric field)')
            # mark material matrix
            for i in range(self.material_matrix.shape[0]):
                for j in range(self.material_matrix.shape[1]):
                    if self.material_matrix[i, j] == 1:
                        plt.plot(i*pixel_size-1, j*pixel_size-1, 'o', color='darkgrey',
                                 markersize=2, label='Silicon')
                    elif self.material_matrix[i, j] == 0:
                        plt.plot(i*pixel_size-1, j*pixel_size-1, 'o',
                                 color='black', markersize=2, label='Silica')
            plt.xlabel('x (microns) → right')
            plt.ylabel('y (microns) → top')
            plt.title(
                f'Field Distribution at Step {self.material_matrix_idx}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            img_path = os.path.join(self.log_dir, 'field_images', f'field_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(img_path)
            plt.close()
        except Exception as e:
            print(f'Error plotting field results: {e}')
