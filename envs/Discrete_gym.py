"""
Minimal OpenAI Gymnasium Environment Template
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.meep_simulation import WaveguideSimulation
from config import config
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
        self.last_score = None
        
        # Store the last completed episode's final metrics (for callback logging)
        self.last_episode_metrics = None

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

        # Use calculate_flux to get initial efield_state for empty matrix
        # This returns: input_mode_flux, output_mode_flux_1, output_mode_flux_2, efield_state, hz_data, input_mode, output_mode_1, output_mode_2
        _, _, _, efield_state, _, _, _, _ = self.simulation.calculate_flux(self.material_matrix)
        observation = efield_state.copy().astype(np.float32)
        # Append material_matrix_idx to match observation space shape (same as in step function)
        observation = np.append(observation, self.material_matrix_idx)
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

        # Action is a binary array of length 50
        self.material_matrix[self.material_matrix_idx] = action
        self.material_matrix_idx += 1

        # calculate_flux returns: input_mode_flux, output_mode_flux_1, output_mode_flux_2, efield_state, hz_data, input_mode, output_mode_1, output_mode_2
        _, _, _, efield_state, _, _, _, _ = self.simulation.calculate_flux(
            self.material_matrix)

        # Use MODE coefficients for reward calculation (instead of raw flux)
        current_score, reward = self.get_reward()
       
        terminated = self.material_matrix_idx >= self.max_steps  # Goal reached
        truncated = False   # Time limit exceeded
        
        # Save final metrics when episode ends (before reset happens)
        if terminated:
            self.last_episode_metrics = {
                'material_matrix': self.material_matrix.copy(),
                'efield_state': efield_state.copy(),
                'total_transmission': self._step_metrics['total_transmission'],
                'transmission_1': self._step_metrics['transmission_1'],
                'transmission_2': self._step_metrics['transmission_2'],
                'balance_score': self._step_metrics['balance_score'],
                'current_score': self._step_metrics['current_score'],
            }

        # Get observation - return the current efield_state as observation
        # This gives the agent feedback about the current state
        if self.material_matrix_idx > 0:
            # Use efield_state directly as observation
            observation = efield_state.copy().astype(np.float32)

        else:
            # Initial state: return zeros
            observation = np.zeros(self.obs_size, dtype=np.float32)

        # Info dictionary with custom metrics
        info = self._step_metrics if hasattr(self, '_step_metrics') else {}
        observation = np.append(observation, self.material_matrix_idx)

        return observation, reward, terminated, truncated, info

    def get_reward(self):
        """
        Calculate reward using transmission from meep_simulation.
        Uses get_output_transmission() method directly.
        """
        # Get transmission using the method from meep_simulation
        _, input_mode = self.simulation.get_flux_input_mode(band_num=1)
        transmission_1, transmission_2, total_transmission, diff_transmission = self.simulation.get_output_transmission(band_num=1)
        
        transmission_score = min(max(total_transmission/input_mode, 0), 1)

        # Calculate balance score (how evenly distributed between outputs)
        if total_transmission > 0:
            diff_ratio = diff_transmission / total_transmission
        else:
            diff_ratio = 1.0  # If no transmission, balance is worst
        balance_score = max(1 - diff_ratio, 0)

        current_score = transmission_score * balance_score
        reward = current_score - self.last_score if self.last_score is not None else 0
        self.last_score = current_score

        # Store metrics for info dict
        self._step_metrics = {
            "total_transmission": total_transmission,
            "diff_transmission": diff_transmission,
            "transmission_1": transmission_1,
            "transmission_2": transmission_2,
            "transmission_score": transmission_score,
            "balance_score": balance_score,
            "current_score": current_score,
        }

        return current_score, reward

    def get_current_metrics(self):
        """
        Get metrics for callback logging.
        Returns the LAST COMPLETED EPISODE's metrics if available,
        otherwise returns current state metrics.
        This ensures we log the final design quality, not a reset/partial state.
        """
        # Return last completed episode metrics if available
        if self.last_episode_metrics is not None:
            return self.last_episode_metrics
        
        # Fallback: return current state (for first rollout before any episode completes)
        _, _, _, efield_state, _, _, _, _ = self.simulation.calculate_flux(self.material_matrix)
        
        transmission_1, transmission_2, total_transmission, diff_transmission = \
            self.simulation.get_output_transmission(band_num=1)
        
        if total_transmission > 0:
            diff_ratio = diff_transmission / total_transmission
        else:
            diff_ratio = 1.0
        balance_score = max(1 - diff_ratio, 0)
        
        transmission_score = min(max(total_transmission, 0), 1)
        current_score = transmission_score * balance_score
        
        return {
            'material_matrix': self.material_matrix.copy(),
            'efield_state': efield_state,
            'total_transmission': total_transmission,
            'transmission_1': transmission_1,
            'transmission_2': transmission_2,
            'balance_score': balance_score,
            'current_score': current_score,
        }

    def save_design_plot(self, save_path):
        """Save design plot to file (called from subprocess).
        Uses last completed episode's design if available."""
        if self.last_episode_metrics is not None:
            matrix = self.last_episode_metrics['material_matrix']
        else:
            matrix = self.material_matrix
        self.simulation.plot_design(
            matrix=matrix,
            save_path=save_path,
            show_plot=False
        )

    def save_distribution_plot(self, save_path):
        """Save distribution plot to file (called from subprocess).
        Uses last completed episode's efield if available."""
        if self.last_episode_metrics is not None:
            efield_state = self.last_episode_metrics['efield_state']
        else:
            _, _, _, efield_state, _, _, _, _ = self.simulation.calculate_flux(self.material_matrix)
        self.simulation.plot_distribution(
            efield_state=efield_state,
            save_path=save_path,
            show_plot=False
        )