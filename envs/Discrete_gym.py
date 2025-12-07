"""
Minimal OpenAI Gymnasium Environment Template
"""

import gymnasium as gym
import time
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from envs.meep_simulation import WaveguideSimulation
from config import config


class MaterialMatrixCNN(nn.Module):
    """
    CNN model for extracting features from material_matrix.
    Takes a 2D material matrix (pixel_num_x, pixel_num_y) and extracts features.
    """
    def __init__(self, input_height=20, input_width=20, feature_dim=64):
        super(MaterialMatrixCNN, self).__init__()
        self.feature_dim = feature_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Calculate the size after convolutions and pooling
        # After 3 conv layers with padding=1, size remains the same
        # After 3 maxpool layers (each /2), size becomes input_size / 8
        pooled_height = input_height // 8
        pooled_width = input_width // 8
        self.fc_input_size = 64 * pooled_height * pooled_width
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, feature_dim)
        
        # Activation and pooling
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch, 1, height, width)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class MinimalEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, use_cnn=False):
        """
        Initialize the environment.

        Args:
            render_mode: "human" for GUI, "rgb_array" for image, None for no rendering
            use_cnn: Whether to use CNN feature extraction
        """
        super().__init__()

        self.action_size = config.environment.action_size
        self.pixel_num_x = config.simulation.pixel_num_x
        self.pixel_num_y = config.simulation.pixel_num_y
        
        # Validate that max_steps doesn't exceed material matrix size
        assert config.environment.max_steps <= self.pixel_num_x, \
            f"max_steps ({config.environment.max_steps}) must be <= pixel_num_x ({self.pixel_num_x})"
        
        # Initialize CNN feature extractor (optional feature) - must be before obs_size calculation
        # Extract features from the entire material_matrix instead of just the previous layer
        # Note: Enabling this feature will change the observation space size (from pixel_num_y to cnn_feature_dim)
        self.use_cnn_features = use_cnn  # Set from argument
        self.cnn_feature_dim = 64  # Dimension of CNN-extracted features
        self.device = torch.device("cpu")  # Can be changed to "cuda" if GPU is available
        if self.use_cnn_features:
            self.cnn_model = MaterialMatrixCNN(
                input_height=self.pixel_num_x,
                input_width=self.pixel_num_y,
                feature_dim=self.cnn_feature_dim
            )
            self.cnn_model.eval()  # Set to evaluation mode
            self.cnn_model.to(self.device)
        else:
            self.cnn_model = None
        
        # Calculate observation size based on whether CNN is enabled
        # Observation = 10 monitors + 1 matrix index + (previous layer or CNN features)
        num_monitors = config.simulation.num_flux_regions  # 10 monitors
        self.use_hybrid_features = False  # If True, use both CNN features AND previous layer
        if self.use_cnn_features:
            if self.use_hybrid_features:
                # Hybrid: CNN features + previous layer
                self.obs_size = num_monitors + 1 + self.cnn_feature_dim + self.pixel_num_y
            else:
                # CNN only: replace previous layer with CNN features
                self.obs_size = num_monitors + 1 + self.cnn_feature_dim
        else:
            self.obs_size = config.environment.obs_size  # 10 + 1 + 20 = 31 (from config)
        
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
        self.material_matrix = np.zeros(
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
        # pixel_num_x and pixel_num_y are already defined at the beginning of __init__
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
            If use_cnn_features=False: 1D array of length pixel_num_y (20 values: the previous layer)
            If use_cnn_features=True: 1D array of CNN-extracted features (length = cnn_feature_dim)
        """
        # Use CNN feature extraction if enabled
        if hasattr(self, 'use_cnn_features') and self.use_cnn_features:
            return self._get_previous_layer(use_cnn=True)
        
        # Original behavior: return previous layer
        if self.material_matrix_idx == 0:
            # First layer: use input waveguide pattern as previous layer
            previous_layer = self._get_default_waveguide_layer()
        else:
            # Get the previous layer from material_matrix
            # material_matrix_idx has been incremented, so previous is at idx - 1
            previous_layer = self.material_matrix[self.material_matrix_idx - 1].copy()
        
        return previous_layer.astype(np.float32)

    def _get_previous_layer(self, use_cnn=None):
        """
        Get the previous layer (the layer before the current one).
        For the first layer (material_matrix_idx == 0), returns the input waveguide pattern.
        
        Args:
            use_cnn: If True, extract CNN features. If False, return physical layer.
                    If None (default), use self.use_cnn_features setting.
        
        Returns:
            If using CNN: 1D array of CNN-extracted features (length = cnn_feature_dim)
            If not using CNN: 1D array of length pixel_num_y (20 values) with 1=silicon, 0=silica
        """
        # Determine whether to use CNN based on argument priority
        # If use_cnn is explicitly provided (True/False), use it.
        # Otherwise, fall back to the environment configuration.
        should_use_cnn = self.use_cnn_features if use_cnn is None else use_cnn

        # Use CNN feature extraction if enabled
        if should_use_cnn:
            if not hasattr(self, 'cnn_model') or self.cnn_model is None:
                # Fallback to regular method if CNN not initialized
                if self.material_matrix_idx == 0:
                    return self._get_default_waveguide_layer()
                else:
                    return self.material_matrix[self.material_matrix_idx - 1].copy()
            
            # Prepare material_matrix for CNN input
            # Handle partial matrix (fill uninitialized rows with default waveguide pattern)
            matrix_for_cnn = self.material_matrix.copy()
            
            # Fill uninitialized rows (from material_matrix_idx onwards) with default pattern
            default_layer = self._get_default_waveguide_layer()
            for idx in range(self.material_matrix_idx, self.pixel_num_x):
                matrix_for_cnn[idx] = default_layer
            
            # Convert to tensor: (1, 1, height, width) for CNN input
            matrix_tensor = torch.FloatTensor(matrix_for_cnn).unsqueeze(0).unsqueeze(0)
            matrix_tensor = matrix_tensor.to(self.device)
            
            # Extract features using CNN
            with torch.no_grad():
                cnn_features = self.cnn_model(matrix_tensor)
            
            # Convert back to numpy array
            cnn_features_np = cnn_features.cpu().numpy().flatten().astype(np.float32)
            
            return cnn_features_np
        
        # Original behavior: return previous layer
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

        # Use calculate_flux to get initial hzfield_state for empty matrix
        # This returns: input_mode_flux, output_mode_flux_1, output_mode_flux_2, hzfield_state, hz_data, input_mode, output_mode_1, output_mode_2
        # For initial state, use empty matrix (all zeros)
        empty_matrix = np.zeros((config.simulation.pixel_num_x, config.simulation.pixel_num_y))
        hzfield_state, _ = self.simulation.calculate_flux(empty_matrix)
        
        # Normalize hzfield_state by dividing by maximum (bounded between 0 and 1)
        hzfield_max = np.max(hzfield_state)
        if hzfield_max > 0:
            hzfield_state_normalized = hzfield_state / hzfield_max
        else:
            hzfield_state_normalized = hzfield_state  # If all zeros, keep as is
        
        # Build observation: 10 monitors + matrix index + (previous layer or CNN features)
        observation = hzfield_state_normalized.copy().astype(np.float32)  # 10 monitor values (normalized)
        observation = np.append(observation, float(self.material_matrix_idx))  # 1 matrix index
        
        # Add CNN features and/or previous layer based on configuration
        if hasattr(self, 'use_cnn_features') and self.use_cnn_features:
            cnn_features = self._get_previous_layer(use_cnn=True)  # CNN features
            observation = np.append(observation, cnn_features)
            
            # If hybrid mode, also add previous layer
            if hasattr(self, 'use_hybrid_features') and self.use_hybrid_features:
                previous_layer = self._get_previous_layer(use_cnn=False)  # Original layer
                observation = np.append(observation, previous_layer)
        else:
            previous_layer = self._get_previous_layers_state()  # 20 values (previous layer)
            observation = np.append(observation, previous_layer)
        
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
        t0 = time.time()

        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Action is a binary array representing one layer (row) of the design
        # Get previous layer before updating (for similarity calculation)
        # FORCE use_cnn=False here because we need the physical layer for similarity calculation
        previous_layer = self._get_previous_layer(use_cnn=False)
        
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
        hzfield_state, hz_data= self.simulation.calculate_flux(
            self.material_matrix)

        t_sim_end = time.time()
        
        # Use MODE coefficients for reward calculation (instead of raw flux)
        # Pass current layer and previous layer for similarity calculation
        current_score, reward = self.get_reward(current_layer=action, previous_layer=previous_layer)
       
        terminated = self.material_matrix_idx >= self.max_steps  # Goal reached
        truncated = False   # Time limit exceeded
        
        # Save final metrics when episode ends (before reset happens)
        if terminated:
            self.last_episode_metrics = {
                'material_matrix': self.material_matrix.copy(),
                'hz_data': hz_data.copy(),
                'hzfield_state': hzfield_state.copy(),
                'total_transmission': self._step_metrics['total_transmission'],
                'transmission_score': self._step_metrics.get('transmission_score', 0.0),  # Ensure this is included
                'transmission_1': self._step_metrics['transmission_1'],
                'transmission_2': self._step_metrics['transmission_2'],
                'balance_score': self._step_metrics['balance_score'],
                'current_score': self._step_metrics['current_score'],
                'similarity_score': self._step_metrics.get('similarity_score', 0.0),
            }

        # Get observation - return the current hzfield_state as observation
        # This gives the agent feedback about the current state
        if self.material_matrix_idx > 0:
            # Normalize hzfield_state by dividing by maximum (bounded between 0 and 1)
            hzfield_max = np.max(hzfield_state)
            if hzfield_max > 0:
                hzfield_state_normalized = hzfield_state / hzfield_max
            else:
                hzfield_state_normalized = hzfield_state  # If all zeros, keep as is
            
            # Build observation: 10 monitors + matrix index + (previous layer or CNN features)
            observation = hzfield_state_normalized.copy().astype(np.float32)  # 10 monitor values (normalized)
            observation = np.append(observation, float(self.material_matrix_idx))  # 1 matrix index
            
            # Add CNN features and/or previous layer based on configuration
            if hasattr(self, 'use_cnn_features') and self.use_cnn_features:
                cnn_features = self._get_previous_layer(use_cnn=True)  # CNN features
                observation = np.append(observation, cnn_features)
                
                # If hybrid mode, also add previous layer
                if hasattr(self, 'use_hybrid_features') and self.use_hybrid_features:
                    # Explicitly get physical layer for observation
                    prev_phys_layer = self._get_previous_layer(use_cnn=False)  # Original layer
                    observation = np.append(observation, prev_phys_layer)
            else:
                # No CNN, use standard previous layer method
                previous_layer = self._get_previous_layers_state()  # 20 values (previous layer)
                observation = np.append(observation, previous_layer)
        else:
            # Initial state: return zeros (shouldn't happen after reset, but just in case)
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
        hzfield_state, _ = self.simulation.calculate_flux(self.material_matrix)
        
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
            'total_transmission': total_transmission,
            'transmission_score': normalized_transmission,  # Use normalized score for consistency
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
            print("self.last_episode_metrics is not None")
        else:
            matrix = self.material_matrix
            _, hz_data = self.simulation.calculate_flux(self.material_matrix)
            print("self.last_episode_metrics is None")
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
            hzfield_state, _ = self.simulation.calculate_flux(self.material_matrix)
        self.simulation.plot_distribution(
            hzfield_state=hzfield_state,
            save_path=save_path,
            show_plot=False,
            title_suffix=title_suffix
        )