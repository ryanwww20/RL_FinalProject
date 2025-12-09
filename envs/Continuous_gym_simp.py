"""
Continuous Relaxation (SIMP) Environment
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.meep_simulation import WaveguideSimulation
from envs.Continuous_gym import MinimalEnv
from config import config

class ContinuousSIMPEnv(MinimalEnv):
    def __init__(self, render_mode=None, beta=1.0, eta=0.5):
        """
        Initialize the SIMP environment.
        
        Args:
            render_mode: Rendering mode
            beta: Steepness of the projection filter (1.0 = linear/smooth, >10 = step-like)
            eta: Threshold for the projection filter (usually 0.5)
        """
        super().__init__(render_mode)
        
        # Override action space to [-1, 1] as per instructions
        # This represents the raw output from SAC (before normalization to [0, 1])
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.action_size,),
            dtype=np.float32
        )
        
        # SIMP parameters
        self.beta = beta
        self.eta = eta
        
    def set_beta(self, beta):
        """Update the beta (steepness) parameter"""
        self.beta = beta
        
    def get_beta(self):
        """Get the current beta parameter"""
        return self.beta
        
    def _apply_projection(self, rho):
        """
        Apply Tanh Projection Filter.
        
        Args:
            rho: normalized density in [0, 1]
            
        Returns:
            rho_projected: projected density in [0, 1]
        """
        beta = self.beta
        eta = self.eta
        
        # Tanh Projection Formula
        num = np.tanh(beta * eta) + np.tanh(beta * (rho - eta))
        den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
        
        # Result should be in [0, 1], but numerical issues might occur
        return num / den

    def _calculate_similarity(self, current_layer, previous_layer):
        """
        Calculate similarity between current and previous layer.
        For continuous values, we use 1 - L1_distance.
        
        Args:
            current_layer: 1D array of current layer (length pixel_num_y)
            previous_layer: 1D array of previous layer (length pixel_num_y)
        
        Returns:
            similarity: Sum of (1 - abs(diff)), max is pixel_num_y
        """
        # Count pixels where current_layer is close to previous_layer
        # Using L1 distance metric: 1 - |a - b|
        # If a=b, score=1 per pixel. If |a-b|=1, score=0 per pixel.
        return np.sum(1.0 - np.abs(current_layer - previous_layer))

    def step(self, action):
        """
        Execute one step in the environment.
        1. Normalize action from [-1, 1] to [0, 1]
        2. Apply Projection Filter
        3. Update material matrix
        4. Run continuous simulation
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 1. Normalization: [-1, 1] -> [0, 1]
        rho = (action + 1) / 2.0
        # Clip to ensure [0, 1] (numerical stability)
        rho = np.clip(rho, 0, 1)
        
        # 2. Projection
        rho_projected = self._apply_projection(rho)
        # Clip again just in case
        rho_projected = np.clip(rho_projected, 0, 1)
        
        # Get previous layer before updating (for similarity calculation)
        previous_layer = self._get_previous_layer()
        
        # Update the material matrix: set the row at material_matrix_idx
        # Use projected values directly (float)
        self.material_matrix[self.material_matrix_idx, :] = rho_projected
        
        # Update history with projected action
        self.layer_history.append(rho_projected.copy())
        if len(self.layer_history) > self.num_previous_layers:
            self.layer_history.pop(0)
            
        self.material_matrix_idx += 1

        # Use calculate_flux_continuous for continuous material properties
        _, _, _, hzfield_state, _, _, _, _ = self.simulation.calculate_flux_continuous(
            self.material_matrix)

        # Calculate reward using projected layer
        current_score, reward = self.get_reward(current_layer=rho_projected, previous_layer=previous_layer)
        
        # Accumulate episode reward
        self.episode_reward += reward
       
        terminated = self.material_matrix_idx >= self.max_steps
        truncated = False
        
        # Save final metrics when episode ends
        if terminated:
            self.last_episode_metrics = {
                'material_matrix': self.material_matrix.copy(),
                'hzfield_state': hzfield_state.copy(),
                'total_transmission': self._step_metrics['total_transmission'],
                'transmission_1': self._step_metrics['transmission_1'],
                'transmission_2': self._step_metrics['transmission_2'],
                'balance_score': self._step_metrics['balance_score'],
                'current_score': self._step_metrics['current_score'],
                'similarity_score': self._step_metrics.get('similarity_score', 0.0),
                'total_reward': self.episode_reward,
                'beta': self.beta
            }

        # Get observation
        if self.material_matrix_idx > 0:
            hzfield_max = np.max(hzfield_state)
            if hzfield_max > 0:
                hzfield_state_normalized = hzfield_state / hzfield_max
            else:
                hzfield_state_normalized = hzfield_state
            
            observation = hzfield_state_normalized.copy().astype(np.float32)
            observation = np.append(observation, float(self.material_matrix_idx))
            previous_layer_obs = self._get_previous_layers_state()
            observation = np.append(observation, previous_layer_obs)
        else:
            observation = np.zeros(self.obs_size, dtype=np.float32)

        info = self._step_metrics if hasattr(self, '_step_metrics') else {}
        info['beta'] = self.beta

        return observation, reward, terminated, truncated, info

    def save_design_plot(self, save_path, title_suffix=None):
        """Save design plot, appending beta info to title"""
        if title_suffix:
            title_suffix += f" (Beta={self.beta:.1f})"
        else:
            title_suffix = f"Beta={self.beta:.1f}"
            
        super().save_design_plot(save_path, title_suffix)

