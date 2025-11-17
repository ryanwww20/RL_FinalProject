import meep as mp

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

Cell_SX = 20
Cell_SY = 20
Design_Region_SX = 15
Design_Region_SY = 15
BLOCK_SIZE_X = 1
BLOCK_SIZE_Y = 1
BLOCK_NUM_X = 15
BLOCK_NUM_Y = 15
WAVELENGTH = 1.55  # μm
FREQUENCY = 1 / WAVELENGTH
RESOLUTION = 20
NUM_DETECTORS = 100
STATE_SIZE = NUM_DETECTORS
ACTION_SIZE = BLOCK_NUM_Y
OUTPUT_PLANE_X = Cell_SX/2 - 1


class MeepSimulation(gym.Env):
    metadata = {
        "render_modes": ['ansi', 'human', 'rgb_array'],
        "render_fps": 2,
    }

    def __init__(self):
        super().__init__()
        self.cell_sx = Cell_SX
        self.cell_sy = Cell_SY
        self.design_region_sx = Design_Region_SX
        self.design_region_sy = Design_Region_SY
        self.cell_size = mp.Vector3(self.cell_sx, self.cell_sy, 0)
        self.block_size_x = BLOCK_SIZE_X
        self.block_size_y = BLOCK_SIZE_Y
        self.block_num_x = BLOCK_NUM_X
        self.block_num_y = BLOCK_NUM_Y
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.frequency = FREQUENCY
        self.pml_layers = [mp.PML(1.0)]
        self.resolution = RESOLUTION
        self.pattern = np.array([])
        self.geometry = []  # Initialize geometry list
        self.num_detectors = NUM_DETECTORS
        self.output_plane_x = OUTPUT_PLANE_X
        self.layer_num = 0

        # Set up sources first
        self.set_sources()

        # Set up simulation (needs geometry and sources)
        self.set_simulation()

        # Set up flux monitors (needs sim to exist)
        self.flux_monitors = []  # Initialize flux monitors list
        self.set_flux_monitors()

        # Initialize target state
        self.target_state = np.zeros(STATE_SIZE, dtype=np.float32)
        self.set_target_state()

        # -----------------------gymnasium-----------------------------
        # Define action and observation spaces
        self.action_space = spaces.MultiBinary(n=ACTION_SIZE)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(STATE_SIZE,), dtype=np.float32)
        # -------------------------------------------------------------

    def set_simulation(self):
        self.sim = mp.Simulation(cell_size=self.cell_size, boundary_layers=self.pml_layers,
                                 geometry=self.geometry, sources=self.sources, resolution=self.resolution)

    def run_simulation(self, until=200):
        self.sim.run(until=until)

    def set_sources(self):
        """Set up sources for the simulation."""
        self.sources = [mp.Source(
            src=mp.GaussianSource(self.frequency, fwidth=0.2*self.frequency),
            component=mp.Ez,
            center=mp.Vector3(-self.cell_sx/2 + 1, 0),  # near left boundary
            size=mp.Vector3(0, self.cell_sy)
        )]

    def set_flux_monitors(self):
        detector_height = self.cell_sy / self.num_detectors
        for i in range(self.num_detectors):
            y_pos = -self.cell_sy/2 + (i + 0.5) * detector_height
            flux_region = mp.FluxRegion(
                center=mp.Vector3(self.output_plane_x, y_pos),
                size=mp.Vector3(0, detector_height)
            )
            self.flux_monitors.append(self.sim.add_flux(
                self.frequency, 0, 1, flux_region))

    def set_target_state(self):
        """Set up the target power distribution pattern.

        Creates a target distribution with two peaks:
        - First peak: detectors 12-37 (25% to 37.5% of range)
        - Second peak: detectors 62-87 (62.5% to 87.5% of range)
        """
        # Create binary pattern first
        for i in range(self.state_size):
            if i >= self.state_size / 8 and i <= self.state_size * 3 / 8:
                self.target_state[i] = 1
            elif i >= self.state_size * 5 / 8 and i <= self.state_size * 7 / 8:
                self.target_state[i] = 1
            else:
                self.target_state[i] = 0

    def add_layer(self, layer):
        if self.pattern.size == 0:
            self.pattern = layer
        else:
            self.pattern = np.hstack((self.pattern, layer))
        # print(f"pattern: {self.pattern}")
        # print(f"pattern shape: {self.pattern.shape}")
        ny, nx = self.pattern.shape
        for i in range(ny):  # add layer x = (-self.sx/2 + 1) + (nx+0.5)*self.block_size_x
            if self.pattern[i, nx-1] == 1:
                center_x = (-self.design_region_sx/2 + 1) + \
                    (nx+0.5)*self.block_size_x  # (-4 + 1.5)
                center_y = (self.design_region_sy/2 - 1) + \
                    (-i-0.5)*self.block_size_y  # (-2 + )
                # print(f"added block at ({center_x}, {center_y})")
                self.geometry.append(
                    mp.Block(
                        material=mp.Medium(index=3.45),  # Silicon
                        center=mp.Vector3(center_x, center_y),
                        size=mp.Vector3(self.block_size_x, self.block_size_y)
                    )
                )

    def cell_visualization(self):
        """Visualize the electric field distribution with coordinate markings."""
        # Get field data
        eps_data = self.sim.get_array(
            center=mp.Vector3(), size=self.cell_size, component=mp.Dielectric)
        ez_data = self.sim.get_array(
            center=mp.Vector3(), size=self.cell_size, component=mp.Ez)

        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary',
                   extent=[-self.cell_sx/2, self.cell_sx/2, -self.cell_sy/2, self.cell_sy/2], origin='lower')
        plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.8,
                   extent=[-self.cell_sx/2, self.cell_sx/2, -self.cell_sy/2, self.cell_sy/2], origin='lower')
        plt.xlabel('X Position (μm)', fontsize=12)
        plt.ylabel('Y Position (μm)', fontsize=12)
        plt.title('Electric Field Distribution', fontsize=12)
        plt.colorbar(label='Ez Field (au)', fraction=0.046, pad=0.04)
        plt.axhline(y=0, color='white', linestyle='--',
                    linewidth=0.5, alpha=0.5)
        plt.axvline(x=0, color='white', linestyle='--',
                    linewidth=0.5, alpha=0.5)
        # Mark the output plane
        plt.axvline(x=self.output_plane_x, color='yellow',
                    linestyle='-', linewidth=2, label='Output Plane')
        plt.legend(loc='upper left', fontsize=10)
        plt.tight_layout()
        return plt

    def power_distribution(self):
        """Extract and plot the power distribution along the output plane."""
        # Extract power distribution along y-axis at output plane
        power_distribution = []
        y_positions = []
        detector_height = self.cell_sy / self.num_detectors

        for i, monitor in enumerate(self.flux_monitors):
            power = mp.get_fluxes(monitor)[0]
            power_distribution.append(power)
            y_pos = -self.cell_sy/2 + (i + 0.5) * detector_height
            y_positions.append(y_pos)

        # Plot power distribution
        plt.figure(figsize=(8, 6))
        plt.plot(y_positions, power_distribution,
                 'o-', linewidth=2, markersize=8)
        plt.xlabel('Y Position (μm)', fontsize=12)
        plt.ylabel('Power (au)', fontsize=12)
        plt.title('Power Distribution at Output Plane', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Print results
        print("\n=== Power Distribution at Output Plane ===")
        print(f"Total transmitted power: {np.sum(power_distribution):.6f}")
        print("\nPower at each detector position:")
        for i, (y, p) in enumerate(zip(y_positions, power_distribution)):
            print(f"  Detector {i+1} (y={y:+.2f}): {p:.6f}")

        return plt, power_distribution, y_positions

    def get_power_distribution(self):
        """Extract the power distribution along the output plane."""
        power_distribution = []
        y_positions = []
        detector_height = self.cell_sy / self.num_detectors
        for i, monitor in enumerate(self.flux_monitors):
            power = mp.get_fluxes(monitor)[0]
            power_distribution.append(power)
            y_pos = -self.cell_sy/2 + (i + 0.5) * detector_height
            y_positions.append(y_pos)
        return power_distribution, y_positions

    def simulation_reset(self):
        self.set_simulation()
        self.flux_monitors = []
        self.set_flux_monitors()

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset simulation state
        self.pattern = np.array([])
        self.geometry = []
        self.layer_num = 0

        # Re-initialize simulation
        self.set_simulation()
        self.flux_monitors = []
        self.set_flux_monitors()

        # Return initial observation (zero power distribution)
        initial_observation = np.zeros(STATE_SIZE, dtype=np.float32)
        return initial_observation, {}

    def step(self, action):
        """Execute one step in the environment."""
        # Add the layer to the design
        self.add_layer(action.reshape(-1, 1))

        # Run electromagnetic simulation
        # self.simulation_reset()

        # print(f"pattern: {self.pattern}")
        self.run_simulation(until=200)

        # Get power distribution as observation
        # print(f"layer_num: {self.layer_num}", end='\r')
        power_dist, _ = self.get_power_distribution()
        observation = np.array(power_dist, dtype=np.float32)

        # Normalize observation to [0, 1] to match observation_space bounds
        power_min = np.min(observation)
        power_max = np.max(observation)
        if power_max > power_min:  # Avoid division by zero
            observation = (observation - power_min) / (power_max - power_min)
        else:
            # If all values are the same, set to 0
            observation = np.zeros_like(observation)

        # Ensure observation is within [0, 1] bounds and correct dtype
        observation = np.clip(observation, 0.0, 1.0).astype(np.float32)

        # Calculate reward (negative distance from target - minimize difference)
        # reward = float(-np.sum(np.abs(observation - self.target_state)))
        # use MSE
        reward = float(-np.mean(np.square(observation - self.target_state)))

        # Check if episode is done
        self.layer_num += 1
        if self.layer_num >= self.block_num_x:
            terminated = True
            self.layer_num = 0
        else:
            terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    # Create simulation
    simulation = MeepSimulation()

    # Add layers
    for i in range(10):
        layer = np.random.randint(0, 2, size=(15, 1))
        simulation.add_layer(layer)

    # Setup simulation
    simulation.set_sources()
    simulation.set_simulation()
    simulation.set_flux_monitors()

    # Run simulation
    simulation.run_simulation(until=200)

    # Visualize results
    simulation.cell_visualization()
    plt.show()

    simulation.power_distribution()
    plt.show()

    power_distribution, y_positions = simulation.get_power_distribution()
    print(f"power distribution: {power_distribution}")
    print(f"y positions: {y_positions}")

'''
target state:
0000000000000111111111111111111111111100000000000000000000000001111111111111111111111111000000000000
'''
