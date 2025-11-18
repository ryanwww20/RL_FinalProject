"""
Minimal OpenAI Gymnasium Environment Template
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from meep_simulation import WaveguideSimulation
import matplotlib.pyplot as plt
from datetime import datetime

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

        # Define observation and action spaces
        # State is an array of 100
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(100,),
            dtype=np.float32
        )

        # Action space: binary array of length 50 (0/1 values)
        self.action_space = spaces.MultiBinary(50)

        # Initialize state
        self.state = None
        self.render_mode = render_mode
        self.material_matrix = np.zeros((50, 50))
        self.material_matrix_idx = 0
        self.simulation = WaveguideSimulation()

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
        self.material_matrix = np.zeros((50, 50))
        self.material_matrix_idx = 0

        # Return initial observation (zeros since no material set yet)
        observation = np.zeros(100, dtype=np.float32)
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

        output_flux, ez_data = self.simulation.calculate_flux(
            self.material_matrix, x_position=2.0)

        reward = np.sum(output_flux * TARGET_FLUX)/np.sum(output_flux)

        # Check if episode is done
        terminated = self.material_matrix_idx >= 50  # Goal reached
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
            observation = np.zeros(100, dtype=np.float32)

        # Info dictionary (can contain debugging info)
        info = {}

        return observation, reward, terminated, truncated, info

    def reward_plot(self, reward, flux_data):
        # save reward to csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open('/Users/ryan/NTUEE_Local/114-1/RL_FinalPJ/RL_FinalProject/ppo_model_logs/episode_rewards.csv', 'a') as f:
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
        plt.savefig(
            f'/Users/ryan/NTUEE_Local/114-1/RL_FinalPJ/RL_FinalProject/ppo_model_logs/flux_images/flux_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()

    def field_result_plot(self, ez_data):
        # plot field results with marked material matrix
        try:
            extent = [-2, 2, -1, 1]
            plt.figure(figsize=(10, 6))
            plt.imshow(ez_data, interpolation='spline36', cmap='RdBu',
                       aspect='auto', extent=extent, origin='lower')
            plt.colorbar(label='Ez (electric field)')
            # mark material matrix
            for i in range(self.material_matrix.shape[0]):
                for j in range(self.material_matrix.shape[1]):
                    if self.material_matrix[i, j] == 1:
                        plt.plot(i*0.04, j*0.04-1, 'o', color='darkgrey',
                                 markersize=2)
                    # elif self.material_matrix[i, j] == 0:
                    #     plt.plot(i*0.04, j*0.04-1, 'o',
                    #              color='darkgrey', markersize=2)
            plt.xlabel('x (microns) → right')
            plt.ylabel('y (microns) → top')
            plt.title(
                f'Field Distribution at Step {self.material_matrix_idx}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(
                f'/Users/ryan/NTUEE_Local/114-1/RL_FinalPJ/RL_FinalProject/ppo_model_logs/field_images/field_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.close()
        except Exception as e:
            print(f'Error plotting field results: {e}')
