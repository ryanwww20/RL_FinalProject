"""
Minimal OpenAI Gymnasium Environment Template
"""

import gymnasium as gym
import time
from gymnasium import spaces
import numpy as np
from envs.meep_simulation import WaveguideSimulation
from config import config


class MinimalEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, use_cnn=True):
        """
        Initialize the environment.

        Args:
            render_mode: "human" for GUI, "rgb_array" for image, None for no rendering
            use_cnn: True -> new CNN-style observation (matrix flatten + monitors + idx + prev layer);
                     False -> legacy dense observation (monitors + idx + prev layer)
        """
        super().__init__()

        self.use_cnn = use_cnn
        self.action_size = config.environment.action_size
        self.pixel_num_x = config.simulation.pixel_num_x
        self.pixel_num_y = config.simulation.pixel_num_y
        
        # Validate that max_steps doesn't exceed material matrix size
        assert config.environment.max_steps <= self.pixel_num_x, \
            f"max_steps ({config.environment.max_steps}) must be <= pixel_num_x ({self.pixel_num_x})"
        
        # Observation
        num_monitors = config.simulation.num_flux_regions  # 10 monitors
        if self.use_cnn:
            # Flattened matrix + monitors + idx + previous layer
            self.obs_size = (
                self.pixel_num_x * self.pixel_num_y  # design matrix
                + num_monitors                       # monitor readings
                + 1                                  # layer index
                + self.pixel_num_y                   # previous layer
            )
        else:
            # Legacy dense obs (monitors + idx + previous layer), match old_discrete_gym
            self.obs_size = config.environment.obs_size
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        # Action space: binary array of length 20 (0/1 values)
        self.action_space = spaces.MultiBinary(self.action_size)

        # Initialize state
        self.state = None
        self.render_mode = render_mode
        # Material matrix is 2D: (pixel_num_x, pixel_num_y) representing the current design
        # Initialize with all 1's (silicon) instead of 0's (silica)
        self.material_matrix = np.ones(
            (config.simulation.pixel_num_x, config.simulation.pixel_num_y))
        self.material_matrix_idx = 0
        self.max_steps = config.environment.max_steps
        self.simulation = WaveguideSimulation()
        self.last_score = None
        
        # Store the last completed episode's final metrics (for callback logging)
        self.last_episode_metrics = None
        
        # Store previous layers for state space (history of layer matrices)
        self.num_previous_layers = config.environment.num_previous_layers
        self.layer_history = []  # List to store previous layer matrices
        self.waveguide_width = config.simulation.waveguide_width
        self.design_region_y_min = config.simulation.design_region_y_min
        self.design_region_y_max = config.simulation.design_region_y_max
        self.pixel_size = config.simulation.pixel_size

        # Timing statistics
        self.step_count = 0
        self.log_interval = 100  # Print timing every 100 steps
        self.total_step_time = 0.0
        self.total_sim_time = 0.0
        self.total_other_time = 0.0
        
        # Fixed input_mode: calculate once at initialization with all-silicon matrix
        all_silicon_matrix = np.ones((config.simulation.pixel_num_x, config.simulation.pixel_num_y))
        _, _, _ = self.simulation.calculate_flux(all_silicon_matrix)  # Run simulation to get flux
        _, self.fixed_input_mode = self.simulation.get_flux_input_mode(band_num=1)
        self.fixed_input_mode *= 2
        print(f"Fixed input_mode set to {self.fixed_input_mode:.6f} (all-silicon matrix)")

    def _get_default_waveguide_layer(self):
        """
        Create a default layer pattern: silica on sides, silicon in center (like input waveguide).
        A layer is one row (20 pixels in +y direction).
        This is used when there are fewer than 3 previous layers.
        
        Returns:
            1D array of length pixel_num_y (20 values) with 1=silicon, 0=silica
        """
        layer = np.zeros(self.pixel_num_y, dtype=np.float32)
        
        # Calculate which pixels are in the waveguide (centered at y=0)
        # Waveguide width is 0.4 um, centered at y=0
        # Design region spans from y=-1 to y=1 (2 um total)
        waveguide_y_min = -self.waveguide_width / 2
        waveguide_y_max = self.waveguide_width / 2
        
        # Convert y positions to pixel indices (for one row/layer)
        for j in range(self.pixel_num_y):
            # Calculate y center of this pixel
            y_center = self.design_region_y_min + (j + 0.5) * self.pixel_size
            
            # If pixel is within waveguide width, set to silicon (1)
            if waveguide_y_min <= y_center <= waveguide_y_max:
                layer[j] = 1.0
        
        return layer

    def _get_previous_layers_state(self):
        """
        Get the previous layer (the layer before the current one).
        A layer is one row (20 pixels in +y direction).
        For the first layer, returns the input waveguide pattern.
        
        Returns:
            1D array of length pixel_num_y (20 values: the previous layer)
        """
        if self.material_matrix_idx == 0:
            return self._get_default_waveguide_layer()
        return self.material_matrix[self.material_matrix_idx - 1].copy().astype(np.float32)

    def _get_previous_layer(self):
        """
        Return the physical previous layer.
        """
        return self._get_previous_layers_state()

    def _build_observation(self, hzfield_state_normalized: np.ndarray) -> np.ndarray:
        """
        Build observation.
        - use_cnn=True : [flattened matrix | monitors | index | previous_layer]
        - use_cnn=False: [monitors | index | previous_layer] (legacy dense)
        """
        monitors = hzfield_state_normalized.astype(np.float32)
        idx_arr = np.array([float(self.material_matrix_idx)], dtype=np.float32)
        previous_layer = self._get_previous_layers_state()

        if self.use_cnn:
            matrix_flat = self.material_matrix.flatten().astype(np.float32)
            return np.concatenate([matrix_flat, monitors, idx_arr, previous_layer])

        return np.concatenate([monitors, idx_arr, previous_layer]).astype(np.float32)

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
        # Initialize with all 0's (silica)
        self.material_matrix = np.zeros(
            (config.simulation.pixel_num_x, config.simulation.pixel_num_y))
        self.material_matrix_idx = 0
        self.layer_history = []  # Reset layer history
        self.last_score = None

        # Use calculate_flux to get initial hzfield_state for initial matrix
        # This returns: hzfield_state, hz_data
        # For initial state, use matrix with all 0's (silica)
        initial_matrix = np.zeros((config.simulation.pixel_num_x, config.simulation.pixel_num_y))
        hzfield_state, _, _ = self.simulation.calculate_flux(initial_matrix)
        
        # Normalize hzfield_state by dividing by maximum (bounded between 0 and 1)
        hzfield_max = np.max(hzfield_state)
        if hzfield_max > 0:
            hzfield_state_normalized = hzfield_state / hzfield_max
        else:
            hzfield_state_normalized = hzfield_state  # If all zeros, keep as is
        
        observation = self._build_observation(hzfield_state_normalized)
        
        info = {}

        return observation, info

    def step(self, action):
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
        t0 = time.time()

        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Action is a binary array representing one layer (row) of the design
        # Get previous layer before updating (for metrics/logging)
        previous_layer = self._get_previous_layer()
        
        # Update the material matrix: set the row at material_matrix_idx
        self.material_matrix[self.material_matrix_idx] = action
        
        # Store this layer (row) in history (keep only last num_previous_layers)
        # A layer is one row (20 pixels in +y direction)
        self.layer_history.append(action.copy())
        if len(self.layer_history) > self.num_previous_layers:
            self.layer_history.pop(0)  # Remove oldest layer
        
        self.material_matrix_idx += 1

        t_sim_start = time.time()
        
        # calculate_flux returns: input_mode_flux, output_mode_flux_1, output_mode_flux_2, hzfield_state, hz_data, input_mode, output_mode_1, output_mode_2
        hzfield_state, hz_data, hzfield_full_distribution = self.simulation.calculate_flux(
            self.material_matrix)
        t_sim_end = time.time()
        
        # Use MODE coefficients for reward calculation (instead of raw flux)
        # Pass current layer and previous layer for metrics/logging
        current_score, reward = self.get_reward(current_layer=action, previous_layer=previous_layer)
       
        terminated = self.material_matrix_idx >= self.max_steps  # Goal reached
        truncated = False   # Time limit exceeded
        
        # Save final metrics when episode ends (before reset happens)
        if terminated:
            self.last_episode_metrics = {
                'material_matrix': self.material_matrix.copy(),
                'hz_data': hz_data.copy(),
                'hzfield_state': hzfield_state.copy(),
                'hzfield_full_distribution': hzfield_full_distribution.copy(),
                'total_transmission': self._step_metrics['total_transmission'],
                'transmission_score': self._step_metrics.get('transmission_score', 0.0),  # Ensure this is included
                'transmission_1': self._step_metrics['transmission_1'],
                'transmission_2': self._step_metrics['transmission_2'],
                'balance_score': self._step_metrics['balance_score'],
                'current_score': self._step_metrics['current_score'],
            }

        if self.material_matrix_idx > 0:
            hzfield_max = np.max(hzfield_state)
            if hzfield_max > 0:
                hzfield_state_normalized = hzfield_state / hzfield_max
            else:
                hzfield_state_normalized = hzfield_state
            observation = self._build_observation(hzfield_state_normalized)
        else:
            observation = np.zeros(self.obs_size, dtype=np.float32)

        # Info dictionary with custom metrics
        info = self._step_metrics if hasattr(self, '_step_metrics') else {}

        # Timing analysis
        t_final = time.time()
        
        # Accumulate times
        step_time = t_final - t0
        sim_time = t_sim_end - t_sim_start
        other_time = step_time - sim_time
        
        self.total_step_time += step_time
        self.total_sim_time += sim_time
        self.total_other_time += other_time
        self.step_count += 1
        
        # Print stats every log_interval steps
        if self.step_count % self.log_interval == 0:
            avg_step = self.total_step_time / self.log_interval
            avg_sim = self.total_sim_time / self.log_interval
            avg_other = self.total_other_time / self.log_interval
            sim_ratio = avg_sim / avg_step if avg_step > 0 else 0
            
            print(f"[Stats {self.step_count} steps] Avg Step: {avg_step:.4f}s | Sim: {avg_sim:.4f}s ({sim_ratio*100:.1f}%) | Other: {avg_other:.4f}s")
            
            # Reset counters
            self.total_step_time = 0.0
            self.total_sim_time = 0.0
            self.total_other_time = 0.0

        return observation, reward, terminated, truncated, info

    def get_reward(self, current_layer=None, previous_layer=None):
        """
        Calculate reward using transmission from meep_simulation.
        Uses get_output_transmission() method directly.
        
        Args:
            current_layer: Current layer (1D array) for metrics/logging
            previous_layer: Previous layer (1D array) for metrics/logging
        """
        # Get transmission using the method from meep_simulation
        # Use fixed input_mode (calculated at initialization with all-silicon matrix)
        transmission_1, transmission_2, total_transmission, _ = self.simulation.get_output_transmission(band_num=1)
        transmission_score = total_transmission / self.fixed_input_mode

        # Calculate balance score based on how close to target ratio split
        if total_transmission > 0:
            actual_ratio = transmission_1 / total_transmission  # Actual % to output 1
            target_ratio = config.environment.target_ratio
            # balance_score = 1 when perfect, 0 when completely off
            balance_score = max(1 - abs(actual_ratio - target_ratio) / target_ratio, 0)
        else:
            balance_score = 0

        # Calculate current_score using transmission_score (no clipping)
        # transmission_score = total_transmission / fixed_input_mode
        current_score = transmission_score * 10 + balance_score * 10
        reward = current_score - self.last_score if self.last_score is not None else 0

        self.last_score = current_score

        # Store metrics for info dict
        self._step_metrics = {
            "total_transmission": total_transmission,
            "actual_ratio": actual_ratio if total_transmission > 0 else 0.0,  # Actual % to output 1
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
        # Use fixed input_mode (calculated at initialization with all-silicon matrix)
        hzfield_state, _, _ = self.simulation.calculate_flux(self.material_matrix)
        
        transmission_1, transmission_2, total_transmission, _ = \
            self.simulation.get_output_transmission(band_num=1)
        
        # Calculate balance score based on how close to target ratio split
        if total_transmission > 0:
            actual_ratio = transmission_1 / total_transmission  # Actual % to output 1
            target_ratio = config.environment.target_ratio
            balance_score = max(1 - abs(actual_ratio - target_ratio) / target_ratio, 0)
        else:
            balance_score = 0
        
        # Use same formula as get_reward() for consistency (no clipping)
        transmission_score = total_transmission / self.fixed_input_mode
        current_score = transmission_score * 10 + balance_score * 10
        
        return {
            'material_matrix': self.material_matrix.copy(),
            'hzfield_state': hzfield_state,
            'total_transmission': total_transmission,
            'transmission_score': transmission_score,  # Use normalized score for consistency
            'actual_ratio': actual_ratio if total_transmission > 0 else 0.0,  # Actual % to output 1
            'transmission_1': transmission_1,
            'transmission_2': transmission_2,
            'balance_score': balance_score,
            'current_score': current_score,
        }

    def save_design_plot(self, save_path, title_suffix=None):
        """Save design plot to file (called from subprocess).
        Uses last completed episode's design if available."""
        if self.last_episode_metrics is not None:
            matrix = self.last_episode_metrics['material_matrix']
            hz_data = self.last_episode_metrics['hz_data']
        else:
            matrix = self.material_matrix
            _, hz_data, _ = self.simulation.calculate_flux(self.material_matrix)
        self.simulation.plot_design(
            matrix=matrix,
            hz_data=hz_data,
            save_path=save_path,
            show_plot=False,
            title_suffix=title_suffix
        )

    def save_distribution_plot(self, save_path, title_suffix=None):
        """Save distribution plot to file (called from subprocess).
        Uses last completed episode's hzfield if available."""
        if self.last_episode_metrics is not None:
            hzfield_state = self.last_episode_metrics['hzfield_state']
            hzfield_full_distribution = self.last_episode_metrics['hzfield_full_distribution']
        else:
            hzfield_state, _, hzfield_full_distribution = self.simulation.calculate_flux(self.material_matrix)
        # self.simulation.plot_distribution(
        #     hzfield_state=hzfield_state,
        #     save_path=save_path,
        #     show_plot=False,
        #     title_suffix=title_suffix
        # )
        self.simulation.plot_full_distribution(
            hzfield_full_distribution=hzfield_full_distribution,
            save_path=save_path,
            show_plot=False,
            title_suffix=title_suffix
        )