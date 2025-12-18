"""
One-Shot Design Generation Environment

Action Space: MultiBinary(400) - Generate entire 20×20 pixel design in a single action
Observation Space: 410 values (400 matrix + 10 monitors)
Episode ends after 1 step (one action = one complete design)
Default state: All silicon (1's)
"""

import gymnasium as gym
import time
from gymnasium import spaces
import numpy as np
from envs.meep_simulation import WaveguideSimulation
from config import config


class OneShotEnv(gym.Env):
    """
    One-shot design generation environment.
    
    The agent generates the entire 20×20 pixel design in a single action.
    Episode flow:
        reset() → Initialize 20×20 matrix to all silicon, return observation
        step(action) → Set entire matrix from 400-bit action, run simulation, calculate reward, terminate
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, use_cnn=True):
        """
        Initialize the environment.

        Args:
            render_mode: "human" for GUI, "rgb_array" for image, None for no rendering
            use_cnn: True -> CNN-style observation with matrix;
                     False -> legacy dense observation (monitors only)
        """
        super().__init__()

        self.use_cnn = use_cnn
        self.pixel_num_x = config.simulation.pixel_num_x
        self.pixel_num_y = config.simulation.pixel_num_y
        self.matrix_size = self.pixel_num_x * self.pixel_num_y  # 400 pixels
        
        # Number of monitors
        self.num_monitors = config.simulation.num_flux_regions  # 10 monitors
        
        # Observation space: matrix (400) + monitors (10) = 410
        if self.use_cnn:
            self.obs_size = self.matrix_size + self.num_monitors  # 410
        else:
            # Legacy dense obs (monitors only)
            self.obs_size = self.num_monitors  # 10
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        # Action space: MultiBinary(400) - 400 binary values (0=silica, 1=silicon)
        self.action_space = spaces.MultiBinary(self.matrix_size)

        # Initialize state
        self.state = None
        self.render_mode = render_mode
        
        # Material matrix is 2D: (pixel_num_x, pixel_num_y) representing the current design
        # Initialize with all 1's (silicon)
        self.material_matrix = np.ones(
            (self.pixel_num_x, self.pixel_num_y), dtype=np.float32)
        
        # Episode tracking
        self.episode_step = 0
        self.max_steps = 1  # Episode ends after 1 step
        
        self.simulation = WaveguideSimulation()
        self.last_score = None
        
        # Store the last completed episode's final metrics (for callback logging)
        self.last_episode_metrics = None
        
        # Waveguide parameters (for potential use)
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
        all_silicon_matrix = np.ones((self.pixel_num_x, self.pixel_num_y))
        _, _, _ = self.simulation.calculate_flux(all_silicon_matrix)
        _, self.fixed_input_mode = self.simulation.get_flux_input_mode(band_num=1)
        self.fixed_input_mode *= 2
        print(f"Fixed input_mode set to {self.fixed_input_mode:.6f} (all-silicon matrix)")

    def _build_observation(self, hzfield_state_normalized: np.ndarray) -> np.ndarray:
        """
        Build observation.
        - use_cnn=True : [flattened matrix (400) | monitors (10)] = 410
        - use_cnn=False: [monitors (10)] only (legacy dense)
        """
        monitors = hzfield_state_normalized.astype(np.float32)

        if self.use_cnn:
            matrix_flat = self.material_matrix.flatten().astype(np.float32)
            return np.concatenate([matrix_flat, monitors])

        return monitors

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

        # Reset material matrix to all 1's (silicon)
        self.material_matrix = np.ones(
            (self.pixel_num_x, self.pixel_num_y), dtype=np.float32)
        self.episode_step = 0
        self.last_score = None

        # Run simulation to get initial hzfield_state for all-silicon matrix
        hzfield_state, _, _ = self.simulation.calculate_flux(self.material_matrix)
        
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
        
        Sets the entire 20×20 matrix from the 400-bit action, runs simulation,
        calculates reward, and terminates the episode.

        Args:
            action: Binary array of 400 values (0=silica, 1=silicon)

        Returns:
            observation: New observation after taking action
            reward: Reward for this step
            terminated: Whether episode has ended (always True after 1 step)
            truncated: Whether episode was truncated (always False)
            info: Additional information dictionary
        """
        t0 = time.time()

        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Set entire material matrix from 400-bit action
        # Reshape action from (400,) to (20, 20)
        self.material_matrix = action.reshape(
            (self.pixel_num_x, self.pixel_num_y)).astype(np.float32)
        
        self.episode_step += 1

        t_sim_start = time.time()
        
        # Run simulation
        hzfield_state, hz_data, hzfield_full_distribution = self.simulation.calculate_flux(
            self.material_matrix)
        
        t_sim_end = time.time()
        
        # Calculate reward
        current_score, reward = self.get_reward()
       
        # Episode ends after 1 step
        terminated = True  # Always terminate after one action
        truncated = False
        
        # Save final metrics (episode always ends)
        self.last_episode_metrics = {
            'material_matrix': self.material_matrix.copy(),
            'hz_data': hz_data.copy(),
            'hzfield_state': hzfield_state.copy(),
            'hzfield_full_distribution': hzfield_full_distribution.copy(),
            'total_transmission': self._step_metrics['total_transmission'],
            'transmission_score': self._step_metrics.get('transmission_score', 0.0),
            'transmission_1': self._step_metrics['transmission_1'],
            'transmission_2': self._step_metrics['transmission_2'],
            'balance_score': self._step_metrics['balance_score'],
            'current_score': self._step_metrics['current_score'],
        }

        # Normalize hzfield_state
        hzfield_max = np.max(hzfield_state)
        if hzfield_max > 0:
            hzfield_state_normalized = hzfield_state / hzfield_max
        else:
            hzfield_state_normalized = hzfield_state
        
        observation = self._build_observation(hzfield_state_normalized)

        # Info dictionary with custom metrics
        info = self._step_metrics if hasattr(self, '_step_metrics') else {}

        # Timing analysis
        t_final = time.time()
        
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

    def get_reward(self):
        """
        Calculate reward using transmission from meep_simulation.
        
        Reward = transmission_score * 10 + balance_score * 10
        
        For one-shot environment, we return the absolute score as reward
        (no delta calculation since there's only one step).
        """
        # Get transmission using the method from meep_simulation
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
            actual_ratio = 0.0

        # Calculate score: transmission_score * 10 + balance_score * 10
        current_score = transmission_score * 10 + balance_score * 10
        
        # For one-shot environment, reward is the absolute score
        # (no delta since there's only one step per episode)
        reward = current_score

        self.last_score = current_score

        # Store metrics for info dict
        self._step_metrics = {
            "total_transmission": total_transmission,
            "actual_ratio": actual_ratio,
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
        """
        # Return last completed episode metrics if available
        if self.last_episode_metrics is not None:
            return self.last_episode_metrics
        
        # Fallback: return current state (for first rollout before any episode completes)
        hzfield_state, _, _ = self.simulation.calculate_flux(self.material_matrix)
        
        transmission_1, transmission_2, total_transmission, _ = \
            self.simulation.get_output_transmission(band_num=1)
        
        # Calculate balance score
        if total_transmission > 0:
            actual_ratio = transmission_1 / total_transmission
            target_ratio = config.environment.target_ratio
            balance_score = max(1 - abs(actual_ratio - target_ratio) / target_ratio, 0)
        else:
            balance_score = 0
            actual_ratio = 0.0
        
        transmission_score = total_transmission / self.fixed_input_mode
        current_score = transmission_score * 10 + balance_score * 10
        
        return {
            'material_matrix': self.material_matrix.copy(),
            'hzfield_state': hzfield_state,
            'total_transmission': total_transmission,
            'transmission_score': transmission_score,
            'actual_ratio': actual_ratio,
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
        self.simulation.plot_full_distribution(
            hzfield_full_distribution=hzfield_full_distribution,
            save_path=save_path,
            show_plot=False,
            title_suffix=title_suffix
        )

