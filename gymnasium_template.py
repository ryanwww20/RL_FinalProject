"""
Minimal OpenAI Gymnasium Environment Template
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from meep_simulation import FluxCalculator
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
        self.flux_calculator = FluxCalculator()
        self.training_index = 0  # Track training episode/index
        self.save_visualizations = True  # Flag to enable/disable visualization saving

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
        # Note: training_index is incremented at the end of each episode, not at reset

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

        # Prepare visualization path if saving is enabled
        visualization_path = None
        if self.save_visualizations:
            visualization_path = f'img/train{self.training_index}/cell_visualization_step{self.material_matrix_idx}.png'

        output_flux = self.flux_calculator.calculate_flux(
            self.material_matrix, x_position=2.0, save_visualization_path=visualization_path)

        reward = np.sum(output_flux * TARGET_FLUX)/np.sum(output_flux)

        # Check if episode is done
        terminated = self.material_matrix_idx >= 50  # Goal reached
        if terminated:
            # Increment training index for next episode
            self.training_index += 1
            # save reward to csv
            with open('/Users/ryan/NTUEE_Local/114-1/RL_FinalPJ/RL_FinalProject/ppo_model_logs/episode_rewards.csv', 'a') as f:
                f.write(
                    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}, {reward}\n')
            # plot flux distribution of material matrix
            current_flux = self.flux_calculator.calculate_flux(
                self.material_matrix, x_position=2.0
            )
            plt.figure(figsize=(10, 6))
            plt.plot(current_flux, 'b-', linewidth=2, label='Current Flux')
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
        truncated = False   # Time limit exceeded

        # Get observation - return the current flux distribution as observation
        # This gives the agent feedback about the current state
        if self.material_matrix_idx > 0:
            # Calculate current flux as observation
            current_flux = self.flux_calculator.calculate_flux(
                self.material_matrix, x_position=2.0
            )
            observation = current_flux.copy()

        else:
            # Initial state: return zeros
            observation = np.zeros(100, dtype=np.float32)

        # Info dictionary (can contain debugging info)
        info = {}

        return observation, reward, terminated, truncated, info
