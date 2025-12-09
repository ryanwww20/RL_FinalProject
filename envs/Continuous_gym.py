"""
Minimal OpenAI Gymnasium Environment Template
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.meep_simulation import WaveguideSimulation
from config import config
from surrogate_model.main import RLSurrogateModel
class MinimalEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, is_eval=False):
        """
        Initialize the environment.

        Args:
            render_mode: "human" for GUI, "rgb_array" for image, None for no rendering
            is_eval: True when running in evaluation/inference mode
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
            low=0,
            high=1,
            shape=(self.action_size,),
            dtype=np.float32
        )
        # Initialize state
        self.state = None
        self.render_mode = render_mode
        # Material matrix is 2D: (pixel_num_x, pixel_num_y) representing the current design
        self.material_matrix = np.zeros(
            (config.simulation.pixel_num_x, config.simulation.pixel_num_y))
        self.material_matrix_idx = 0
        self.max_steps = config.environment.max_steps
        self.simulation = WaveguideSimulation()
        self.last_score = None
        
        # Store the last completed episode's final metrics (for callback logging)
        self.last_episode_metrics = None
        
        # Track episode reward for SAC callback
        self.episode_reward = 0.0
        
        # Store previous layers for state space (history of layer matrices)
        self.num_previous_layers = config.environment.num_previous_layers
        self.layer_history = []  # List to store previous layer matrices
        self.pixel_num_x = config.simulation.pixel_num_x
        self.pixel_num_y = config.simulation.pixel_num_y
        self.waveguide_width = config.simulation.waveguide_width
        self.design_region_y_min = config.simulation.design_region_y_min
        self.design_region_y_max = config.simulation.design_region_y_max
        self.pixel_size = config.simulation.pixel_size

        self.is_surrogate = False
        self.episode = 0
        self.surrogate_model = RLSurrogateModel()
        self.surrogate_step_metrics = None
        self.surrogate_train_data = None
        # Flag to let downstream code know whether this env is used for evaluation
        self.is_eval = is_eval

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
        # Get the previous layer
        # For first layer (material_matrix_idx == 0), use input waveguide pattern
        # For subsequent layers, get from material_matrix
        if self.material_matrix_idx == 0:
            # First layer: use input waveguide pattern as previous layer
            previous_layer = self._get_default_waveguide_layer()
        else:
            # Get the previous layer from material_matrix
            # material_matrix_idx has been incremented, so previous is at idx - 1
            previous_layer = self.material_matrix[self.material_matrix_idx - 1].copy()
        
        return previous_layer.astype(np.float32)

    def _get_previous_layer(self):
        """
        Get the previous layer (the layer before the current one).
        For the first layer (material_matrix_idx == 0), returns the input waveguide pattern.
        
        Returns:
            1D array of length pixel_num_y (20 values) with 1=silicon, 0=silica
        """
        if self.material_matrix_idx == 0:
            # First layer: use input waveguide pattern as previous layer
            return self._get_default_waveguide_layer()
        else:
            # Get the previous layer from material_matrix
            # material_matrix_idx has been incremented, so previous is at idx - 1
            return self.material_matrix[self.material_matrix_idx - 1].copy()

    def _calculate_similarity(self, current_layer, previous_layer):
        """
        Calculate similarity between current and previous layer.
        Similarity is the number of identical pixels (both 0 or both 1).
        
        Args:
            current_layer: 1D array of current layer (length pixel_num_y)
            previous_layer: 1D array of previous layer (length pixel_num_y)
        
        Returns:
            similarity: Number of identical pixels (0 to pixel_num_y)
        """
        # Count pixels where current_layer == previous_layer
        similarity = np.sum(current_layer == previous_layer)
        return float(similarity)

    def _is_surrogate_mode(self):
        """
        Determine whether to use surrogate model or meep simulation.
        """
        if self.is_eval:
            self.is_surrogate = False
            return
        elif self.episode < 10:
            self.is_surrogate = False
            return
        else:
            if self.episode % 5 != 1 and self.episode % 5 != 2: # use surrogate model 2 out of 5 episodes
                self.is_surrogate = True
            else:
                self.is_surrogate = False
                print("Finetuning surrogate model")
                print(f"Episode {self.episode}")
                self.surrogate_model.finetune()

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
        self.layer_history = []  # Reset layer history
        self.last_score = None
        self.episode_reward = 0.0  # Reset episode reward

        # Use calculate_flux to get initial hzfield_state for empty matrix
        # This returns: input_mode_flux, output_mode_flux_1, output_mode_flux_2, hzfield_state, hz_data, input_mode, output_mode_1, output_mode_2
        # For initial state, use empty matrix (all zeros)
        empty_matrix = np.zeros((config.simulation.pixel_num_x, config.simulation.pixel_num_y))
        hzfield_state, hz_data = self.simulation.calculate_flux(empty_matrix)
        
        # Normalize hzfield_state by dividing by maximum (bounded between 0 and 1)
        hzfield_max = np.max(hzfield_state)
        if hzfield_max > 0:
            hzfield_state_normalized = hzfield_state / hzfield_max
        else:
            hzfield_state_normalized = hzfield_state  # If all zeros, keep as is
        
        # Build observation: 10 monitors + matrix index + previous layer
        observation = hzfield_state_normalized.copy().astype(np.float32)  # 10 monitor values (normalized)
        observation = np.append(observation, float(self.material_matrix_idx))  # 1 matrix index
        previous_layer = self._get_previous_layers_state()  # 20 values (previous layer)
        observation = np.append(observation, previous_layer)
        
        info = {"is_eval": self.is_eval}

        self._is_surrogate_mode()

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

        # Action is a binary array representing one layer (row) of the design
        # Get previous layer before updating (for similarity calculation)
        previous_layer = self._get_previous_layer()
        
        # Update the material matrix: set the row at material_matrix_idx
        for i in range(self.action_size):
            if action[i] > 0.5:
                self.material_matrix[self.material_matrix_idx, i] = 1
            else:
                self.material_matrix[self.material_matrix_idx, i] = 0
        
        # Store this layer (row) in history (keep only last num_previous_layers)
        # A layer is one row (20 pixels in +y direction)
        self.layer_history.append(action.copy())
        if len(self.layer_history) > self.num_previous_layers:
            self.layer_history.pop(0)  # Remove oldest layer
        
        self.material_matrix_idx += 1

        if self.is_surrogate and not self.is_eval:
            print("Using surrogate model")
            pred = self.surrogate_model.predict(self.material_matrix)
            # surrogate_model.predict returns dict; normalize types for downstream use
            hzfield_state = np.array(pred.get("hzfield_state", pred.get("hz", [])), dtype=np.float32)
            mode_transmission_1 = float(pred.get("mode_transmission_1", pred.get("output_mode_1", 0.0)))
            mode_transmission_2 = float(pred.get("mode_transmission_2", pred.get("output_mode_2", 0.0)))
            input_mode = float(pred.get("input_mode", 1e-6))
            self.surrogate_step_metrics = {
                "material_matrix": self.material_matrix,
                "hzfield_state": hzfield_state,
                "mode_transmission_1": mode_transmission_1,
                "mode_transmission_2": mode_transmission_2,
                "input_mode": input_mode,
            }
            hz_data = None
            current_score, reward = self.surrogate_get_reward(current_layer=action, previous_layer=previous_layer)
        else:
            hzfield_state, hz_data= self.simulation.calculate_flux(self.material_matrix)
            current_score, reward = self.get_reward(current_layer=action, previous_layer=previous_layer)
            _, input_mode = self.simulation.get_flux_input_mode(band_num=1)
            mode_transmission_1, mode_transmission_2, _, _ = self.simulation.get_output_transmission(band_num=1)
            self.surrogate_train_data = {
                "material_matrix": self.material_matrix,
                "hzfield_state": hzfield_state,
                "mode_transmission_1": mode_transmission_1,
                "mode_transmission_2": mode_transmission_2,
                "input_mode": input_mode,
            }
            self.surrogate_model.RL_data_collector.add_sample(self.surrogate_train_data)
            current_score, reward = self.get_reward(current_layer=action, previous_layer=previous_layer)

        # Accumulate episode reward
        self.episode_reward += reward
       
        terminated = self.material_matrix_idx >= self.max_steps  # Goal reached
        truncated = False   # Time limit exceeded
        
        # Save final metrics when episode ends (before reset happens)
        if terminated:
            self.last_episode_metrics = {
                'material_matrix': self.material_matrix.copy(),
                'hz_data': hz_data.copy() if hz_data is not None else None,
                'hzfield_state': hzfield_state.copy() if hzfield_state is not None else None,
                'total_transmission': self._step_metrics['total_transmission'],
                'transmission_1': self._step_metrics['transmission_1'],
                'transmission_2': self._step_metrics['transmission_2'],
                'balance_score': self._step_metrics['balance_score'],
                'current_score': self._step_metrics['current_score'],
                'similarity_score': self._step_metrics.get('similarity_score', 0.0),
                'total_reward': self.episode_reward,  # Add total episode reward
            }
            # Count completed episodes (only when an episode ends)
            self.episode += 1

        # Get observation - return the current hzfield_state as observation
        # This gives the agent feedback about the current state
        if self.material_matrix_idx > 0:
            # Normalize hzfield_state by dividing by maximum (bounded between 0 and 1)
            hzfield_max = np.max(hzfield_state)
            if hzfield_max > 0:
                hzfield_state_normalized = hzfield_state / hzfield_max
            else:
                hzfield_state_normalized = hzfield_state  # If all zeros, keep as is
            
            # Build observation: 10 monitors + matrix index + previous layer
            observation = hzfield_state_normalized.copy().astype(np.float32)  # 10 monitor values (normalized)
            observation = np.append(observation, float(self.material_matrix_idx))  # 1 matrix index
            previous_layer = self._get_previous_layers_state()  # 20 values (previous layer)
            observation = np.append(observation, previous_layer)
        else:
            # Initial state: return zeros (shouldn't happen after reset, but just in case)
            observation = np.zeros(self.obs_size, dtype=np.float32)

        # Info dictionary with custom metrics
        info = self._step_metrics.copy() if hasattr(self, '_step_metrics') else {}
        info["is_eval"] = self.is_eval

        return observation, reward, terminated, truncated, info

    def get_reward(self, current_layer=None, previous_layer=None):
        """
        Calculate reward using transmission from meep_simulation.
        Uses get_output_transmission() method directly.
        
        Args:
            current_layer: Current layer (1D array) for similarity calculation
            previous_layer: Previous layer (1D array) for similarity calculation
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

        # Calculate similarity: number of identical pixels between current and previous layer
        if current_layer is not None and previous_layer is not None:
            similarity = self._calculate_similarity(current_layer, previous_layer)
            # Normalize similarity to [0, 1] by dividing by pixel_num_y
            similarity_score = similarity / self.pixel_num_y
        else:
            similarity = 0.0
            similarity_score = 0.0

        # Calculate current_score: use transmission_score which is already normalized to [0,1]
        # transmission_score = (total_transmission/input_mode) normalized to [0,1] with min/max clamping
        current_score = transmission_score * 10 + balance_score * 10
        reward = current_score - self.last_score if self.last_score is not None else 0
        # Add similarity_score directly to reward (not to current_score)
        reward += similarity_score/10

        self.last_score = current_score

        # Store metrics for info dict
        self._step_metrics = {
            "total_transmission": total_transmission,
            "diff_transmission": diff_transmission,
            "transmission_1": transmission_1,
            "transmission_2": transmission_2,
            "transmission_score": transmission_score,
            "balance_score": balance_score,
            "similarity": similarity,
            "similarity_score": similarity_score,
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
        _, input_mode = self.simulation.get_flux_input_mode(band_num=1)
        hzfield_state, hz_data = self.simulation.calculate_flux(self.material_matrix)
        
        transmission_1, transmission_2, total_transmission, diff_transmission = \
            self.simulation.get_output_transmission(band_num=1)
        
        if total_transmission > 0:
            diff_ratio = diff_transmission / total_transmission
        else:
            diff_ratio = 1.0
        balance_score = max(1 - diff_ratio, 0)
        
        # Keep transmission_score as-is (without dividing by input_mode) for logging
        transmission_score = min(max(total_transmission, 0), 1)
        # Use same formula as get_reward() for consistency
        # Calculate normalized transmission score for current_score (matching get_reward)
        normalized_transmission = min(max(total_transmission / input_mode, 0), 1)
        current_score = normalized_transmission * 10 + balance_score * 10
        
        return {
            'material_matrix': self.material_matrix.copy(),
            'hzfield_state': hzfield_state,
            'hz_data': hz_data,
            'total_transmission': total_transmission,
            'transmission_score': transmission_score,
            'diff_transmission': diff_transmission,
            'transmission_1': transmission_1,
            'transmission_2': transmission_2,
            'balance_score': balance_score,
            'current_score': current_score,
            'similarity_score': 0.0,  # Fallback: similarity not available in get_current_metrics
        }

    def save_design_plot(self, save_path, title_suffix=None):
        """Save design plot to file (called from subprocess).
        Uses last completed episode's design if available."""
        if self.last_episode_metrics is not None:
            matrix = self.last_episode_metrics['material_matrix']
            hz_data = self.last_episode_metrics['hz_data']
        else:
            matrix = self.material_matrix
            _, hz_data= self.simulation.calculate_flux(self.material_matrix)
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
        else:
            hzfield_state, _= self.simulation.calculate_flux(self.material_matrix)
        self.simulation.plot_distribution(
            hzfield_state=hzfield_state,
            save_path=save_path,
            show_plot=False,
            title_suffix=title_suffix
        )

    def surrogate_get_reward(self, current_layer=None, previous_layer=None):
        """
        Calculate reward using surrogate model.
        
        Args:
            current_layer: Current layer (1D array) for similarity calculation
            previous_layer: Previous layer (1D array) for similarity calculation
        """
        # Get transmission using the method from meep_simulation
        total_transmission = self.surrogate_step_metrics["mode_transmission_1"]+self.surrogate_step_metrics["mode_transmission_2"]
        diff_transmission = abs(self.surrogate_step_metrics["mode_transmission_1"]-self.surrogate_step_metrics["mode_transmission_2"])
        transmission_score = min(max((total_transmission)/self.surrogate_step_metrics["input_mode"], 0), 1)

        # Calculate balance score (how evenly distributed between outputs)
        if total_transmission > 0:
            diff_ratio = diff_transmission / total_transmission
        else:
            diff_ratio = 1.0  # If no transmission, balance is worst
        balance_score = max(1 - diff_ratio, 0)

        # Calculate similarity: number of identical pixels between current and previous layer
        if current_layer is not None and previous_layer is not None:
            similarity = self._calculate_similarity(current_layer, previous_layer)
            # Normalize similarity to [0, 1] by dividing by pixel_num_y
            similarity_score = similarity / self.pixel_num_y
        else:
            similarity = 0.0
            similarity_score = 0.0

        # Calculate current_score: use transmission_score which is already normalized to [0,1]
        # transmission_score = (total_transmission/input_mode) normalized to [0,1] with min/max clamping
        current_score = transmission_score * 10 + balance_score * 10
        reward = current_score - self.last_score if self.last_score is not None else 0
        # Add similarity_score directly to reward (not to current_score)
        reward += similarity_score/10

        self.last_score = current_score

        # Store metrics for info dict
        self._step_metrics = {
            "total_transmission": total_transmission,
            "diff_transmission": diff_transmission,
            "transmission_1": self.surrogate_step_metrics["mode_transmission_1"],
            "transmission_2": self.surrogate_step_metrics["mode_transmission_2"],
            "transmission_score": transmission_score,
            "balance_score": balance_score,
            "similarity": similarity,
            "similarity_score": similarity_score,
            "current_score": current_score,
        }

        return current_score, reward