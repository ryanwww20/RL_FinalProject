"""
2D Meep simulation with waveguide and eigenmode source
- 2um x 2um square region
- Thin waveguide extending from left into the square
- EigenModeSource with continuous wave at 1550nm, entering from left
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
import sys
import os

# Add the parent directory to sys.path if running directly
if __name__ == "__main__":
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

from contextlib import redirect_stdout, redirect_stderr
from config import config


class WaveguideSimulation:
    """2D Meep simulation class for waveguide with eigenmode source"""

    def __init__(self):
        """
        Initialize simulation parameters

        Args:
            resolution: pixels per micron
            wavelength: wavelength in microns (1550nm = 1.55)
            cell_size: simulation cell size (Vector3)
            pml_thickness: PML layer thickness
            waveguide_width: waveguide width in microns
            waveguide_index: refractive index of waveguide
            waveguide_center_x: x-coordinate of waveguide center
            waveguide_length: length of waveguide in x-direction
            1 for silicon, 0 for silica
        """
        self.resolution = config.simulation.resolution
        self.wavelength = config.simulation.wavelength
        self.cell_size = config.simulation.cell_size
        self.pml_layers = [mp.PML(config.simulation.pml_thickness)]
        self.waveguide_width = config.simulation.waveguide_width
        self.waveguide_index = config.simulation.waveguide_index
        self.output_y_separation = config.simulation.output_y_separation
        # New coupling lengths
        self.input_coupler_length = config.simulation.input_coupler_length
        # Used for output 1 and 2
        self.output_coupler_length = config.simulation.output_coupler_length

        # Design region remains 2um x 2um centered at (0,0)
        self.design_region_x_min = -1.0
        self.design_region_x_max = 1.0
        self.design_region_y_min = -1.0
        self.design_region_y_max = 1.0

        # Initialize simulation components
        self.geometry = None
        self.sources = None
        self.sim = None
        self.ez_data = None
        self.hz_data = None
        self.flux = None  # Single flux monitor object
        self.flux_regions = []  # List of flux monitors for y-axis distribution
        self.input_flux_region = None  # Flux monitor at the input waveguide
        self.output_flux_region_1 = None  # Flux monitor at the output waveguide 1
        self.output_flux_region_2 = None  # Flux monitor at the output waveguide 2
        self.num_flux_regions = config.simulation.num_flux_regions
        self.simulation_time = config.simulation.simulation_time
        self.output_x = config.simulation.output_x
        self.design_region_x = config.simulation.design_region_x
        self.design_region_y = config.simulation.design_region_y
        self.pixel_size = config.simulation.pixel_size
        self.silicon_index = config.simulation.silicon_index
        self.silica_index = config.simulation.silica_index
        self.pixel_num_x = config.simulation.pixel_num_x
        self.pixel_num_y = config.simulation.pixel_num_y
        self.src_pos_shift_coeff = config.simulation.src_pos_shift_coeff
        self.input_flux_monitor_x = config.simulation.input_flux_monitor_x
        self.output_flux_monitor_x = config.simulation.output_flux_monitor_x

    def create_geometry(self, material_matrix=None):
        """
        Create waveguide geometry: 1 Input (left, connected to x=-1) 
        and 2 Outputs (right, connected to x=1) 
        with material distribution based on matrix in the 2um x 2um square region (-1 < x < 1).
        """
        geometry = []

        # --- 1. Input Waveguide (Left side) ---
        # Starts inside PML, ends exactly at the design region boundary (x=-1.0)
        input_start_x = self.design_region_x_min - \
            self.input_coupler_length  # e.g., -1.0 - 1.5 = -2.5
        input_end_x = self.design_region_x_min                              # x = -1.0
        input_length = self.input_coupler_length
        input_center_x = input_start_x + input_length / 2.0

        input_waveguide = mp.Block(
            center=mp.Vector3(input_center_x, 0, 0),
            size=mp.Vector3(input_length, self.waveguide_width, 0),
            material=mp.Medium(index=self.waveguide_index)
        )
        geometry.append(input_waveguide)

        # --- 2. Two Output Waveguides (Right side) ---
        # Starts at the design region boundary (x=1.0), extends right
        output_start_x = self.design_region_x_max                          # x = 1.0
        output_end_x = self.design_region_x_max + \
            self.output_coupler_length  # e.g., 1.0 + 1.5 = 2.5
        output_length = self.output_coupler_length
        output_center_x = output_start_x + output_length / 2.0

        # Symmetrical output positions (use the default 0.3 or a new class parameter) # You can make this a configurable class property later

        # Output Waveguide 1 (Top)
        output_waveguide_1 = mp.Block(
            center=mp.Vector3(output_center_x, self.output_y_separation, 0),
            size=mp.Vector3(output_length, self.waveguide_width, 0),
            material=mp.Medium(index=self.waveguide_index)
        )
        geometry.append(output_waveguide_1)

        # Output Waveguide 2 (Bottom)
        output_waveguide_2 = mp.Block(
            center=mp.Vector3(output_center_x, -self.output_y_separation, 0),
            size=mp.Vector3(output_length, self.waveguide_width, 0),
            material=mp.Medium(index=self.waveguide_index)
        )
        geometry.append(output_waveguide_2)
        # --- 3. Add material distribution from matrix (Design Region) ---
        if material_matrix is not None:
            material_matrix = np.array(material_matrix)

            if material_matrix.shape != (self.pixel_num_x, self.pixel_num_y):
                # Error handling remains the same
                raise ValueError(
                    f"material_matrix must be {self.pixel_num_x}x{self.pixel_num_y}, got shape {material_matrix.shape}")

            # Design region boundaries: x from -1 to 1, y from -1 to 1
            square_x_min = self.design_region_x_min  # -1.0
            square_y_min = self.design_region_y_min  # -1.0
            dx = self.pixel_size
            dy = self.pixel_size

            # Create pixel blocks, mapping 0-49 indices to -1 to +1 coordinates
            for i in range(self.pixel_num_x):
                for j in range(self.pixel_num_y):
                    # Calculate center within [-1, 1] range
                    x_center = square_x_min + (i + 0.5) * dx
                    y_center = square_y_min + (j + 0.5) * dy

                    if material_matrix[i, j] == 1:
                        silicon_pixel = mp.Block(
                            center=mp.Vector3(x_center, y_center, 0),
                            size=mp.Vector3(dx, dy, 0),
                            material=mp.Medium(index=self.silicon_index)
                        )
                        geometry.append(silicon_pixel)
                    else:
                        silica_pixel = mp.Block(
                            center=mp.Vector3(x_center, y_center, 0),
                            size=mp.Vector3(dx, dy, 0),
                            material=mp.Medium(index=self.silica_index)
                        )
                        geometry.append(silica_pixel)

        self.geometry = geometry

    def plot_geometry(self, save_path=None, show_plot=True, x_range=None, y_range=None):
        """
        Plot the generated geometry (waveguides and design region) based solely on 
        class parameters, ensuring visibility of all major components.
        """

        # CRITICAL: Ensure geometry is created with the current parameters
        if self.geometry is None:
            # Calling create_geometry populates the design region boundaries correctly
            self.create_geometry(material_matrix=None)

        plt.figure(figsize=(10, 5))
        ax = plt.gca()

        # --- 1. Determine Plotting Range ---
        # Default range covers the entire simulation cell
        x_min_default, x_max_default = -self.cell_size.x/2, self.cell_size.x/2
        y_min_default, y_max_default = -self.cell_size.y/2, self.cell_size.y/2
        x_min = x_range[0] if x_range is not None else x_min_default
        x_max = x_range[1] if x_range is not None else x_max_default
        y_min = y_range[0] if y_range is not None else y_min_default
        y_max = y_range[1] if y_range is not None else y_max_default

        # --- 2. Calculate Waveguide Positions ---
        waveguide_color = 'blue'  # Fixed separation, same as in create_geometry

        # 2a. Input Waveguide (Ends at x = -1.0)
        input_length = self.input_coupler_length
        input_x_start = self.design_region_x_min - input_length
        input_y_start = -self.waveguide_width / 2

        # 2b. Output Waveguides (Start at x = 1.0)
        output_length = self.output_coupler_length
        output_x_start = self.design_region_x_max

        output1_y_start = self.output_y_separation - self.waveguide_width / 2
        output2_y_start = -self.output_y_separation - self.waveguide_width / 2

        # --- 3. Plot Waveguides Explicitly ---

        # Input Waveguide
        ax.add_patch(Rectangle(
            (input_x_start, input_y_start), input_length, self.waveguide_width,
            linewidth=1, edgecolor='k', facecolor=waveguide_color, alpha=0.6, label='Input Waveguide'
        ))

        # Output Waveguide 1
        ax.add_patch(Rectangle(
            (output_x_start, output1_y_start), output_length, self.waveguide_width,
            linewidth=1, edgecolor='k', facecolor=waveguide_color, alpha=0.6, label='Output Waveguide'
        ))

        # Output Waveguide 2
        ax.add_patch(Rectangle(
            (output_x_start, output2_y_start), output_length, self.waveguide_width,
            linewidth=1, edgecolor='k', facecolor=waveguide_color, alpha=0.6
        ))
        print('===============================================')
        print(
            f"Output Waveguide 1: {output1_y_start}, Output Waveguide 2: {output2_y_start}")
        print('===============================================')

        # --- 4. Mark the Design Region ---
        design_region_rect = Rectangle(
            # Lower-left corner: (-1.0, -1.0)
            (self.design_region_x_min, self.design_region_y_min),
            2.0, 2.0,  # Width=2.0, Height=2.0
            linewidth=2, edgecolor='lime', facecolor='none',
            linestyle='--', label='Design Region (2x2um)')
        ax.add_patch(design_region_rect)

        # --- 5. Add Annotations for Verification (Fixing the visual length issue) ---

        # Input Waveguide Length Label
        input_center_x = self.design_region_x_min - self.input_coupler_length / 2
        ax.annotate(
            f"L={self.input_coupler_length}µm",
            xy=(input_center_x, self.waveguide_width * 1.5),
            ha='center', va='bottom', fontsize=9, color='darkred',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )

        # Output Waveguide Length Label
        output_center_x = self.design_region_x_max + self.output_coupler_length / 2
        ax.annotate(
            f"L={self.output_coupler_length}µm",
            xy=(output_center_x, self.waveguide_width * 1.5),
            ha='center', va='bottom', fontsize=9, color='darkred',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )

        # --- 6. Final Plot Setup ---
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # FIX: Ensure 1:1 scale for the axes
        ax.set_aspect('equal', adjustable='box')

        ax.set_aspect('equal')
        plt.xlabel('x (microns)')
        plt.ylabel('y (microns)')
        plt.title(f'Waveguide Geometry (Centered Design)')

        # Clean up legend handles
        handles, labels = ax.get_legend_handles_labels()
        unique_handles = dict(zip(labels, handles))

        # FIX: Move legend outside the plot area
        ax.legend(unique_handles.values(), unique_handles.keys(),
                  loc='upper left',        # Position within the bbox_to_anchor
                  # Place just outside the top-right corner of the axes
                  bbox_to_anchor=(1.02, 1),
                  borderaxespad=0.)        # No padding between the legend and the anchor point

        plt.grid(True, alpha=0.2)
        plt.tight_layout()  # IMPORTANT: Adjust plot to make room for the legend

        if show_plot:
            plt.show()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

    def create_sources(self):
        """Create eigenmode source at left edge of input coupler"""
        # Calculate the starting point of the input waveguide (cell's left edge)
        input_coupler_start_x = self.design_region_x_min - self.input_coupler_length

        sources = [mp.EigenModeSource(
            src=mp.ContinuousSource(
                wavelength=self.wavelength,
                width=20
            ),
            # Position source at the start of the input coupler
            center=mp.Vector3(input_coupler_start_x *
                              self.src_pos_shift_coeff, 0.0, 0),
            size=mp.Vector3(0, self.waveguide_width, 0),
            eig_band=1,
            direction=mp.NO_DIRECTION,
            eig_kpoint=mp.Vector3(1, 0, 0),  # +X direction (rightward)
            eig_match_freq=True
        )]
        self.sources = sources
        return self.sources

    def create_simulation(self, add_flux_at_x=None):
        """
        Create Meep simulation object

        Args:
            add_flux_at_x: if provided, add flux monitor at this x position
        """
        if self.geometry is None:
            self.create_geometry()
        if self.sources is None:
            self.create_sources()

        # Suppress Meep's verbose structure initialization output
        # Meep writes directly to file descriptors, so we need to redirect at OS level
        import subprocess
        import tempfile

        # Create a temporary file to capture output
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_file = f.name

        # Redirect stdout and stderr to the temp file
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            with open(temp_file, 'w') as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                self.sim = mp.Simulation(
                    cell_size=self.cell_size,
                    boundary_layers=self.pml_layers,
                    geometry=self.geometry,
                    sources=self.sources,
                    resolution=self.resolution,
                    dimensions=2
                )
        finally:
            # Restore stdout and stderr
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

        # Add flux monitor if requested
        if add_flux_at_x is not None:
            self.add_flux_monitor(add_flux_at_x)

        return self.sim

    def add_flux_monitor(self, height=None):
        """
        Add flux monitor at a specific x position

        Args:
            x_position: x coordinate in microns where to measure flux
            height: height of flux region in y-direction (default: full cell height)
        """
        if self.sim is None:
            raise ValueError(
                "Simulation must be created first. Call create_simulation() method.")

        if height is None:
            height = self.cell_size.y  # Full height of cell

        # Calculate frequency from wavelength
        frequency = 1.0 / self.wavelength

        # Create flux region at x_position
        flux_region = mp.FluxRegion(
            center=mp.Vector3(self.output_x, 0, 0),
            size=mp.Vector3(0, height, 0)  # Vertical line at x_position
        )

        # Add flux monitor to simulation
        self.flux = self.sim.add_flux(frequency, 0, 1, flux_region)

        return self.flux

    def add_flux_monitors_along_y(self, region_height=None):
        """
        Add multiple flux monitors along y-axis at a specific x position

        Args:
            x_position: x coordinate in microns where to measure flux
            num_regions: number of flux regions along y-axis
            region_height: height of each flux region (default: cell_height / num_regions)

        Returns:
            list of flux monitor objects
        """
        if self.sim is None:
            raise ValueError(
                "Simulation must be created first. Call create_simulation() method.")

        # Calculate frequency from wavelength
        frequency = 1.0 / self.wavelength

        # Determine region height
        if region_height is None:
            region_height = self.cell_size.y / self.num_flux_regions

        # Calculate y positions for each region
        # y spans from -cell_size.y/2 to +cell_size.y/2
        y_min = -self.cell_size.y / 2
        y_max = self.cell_size.y / 2
        y_positions = np.linspace(
            y_min + region_height/2, y_max - region_height/2, self.num_flux_regions)

        # Create flux monitors for each y position
        flux_monitors = []
        for y_pos in y_positions:
            flux_region = mp.FluxRegion(
                center=mp.Vector3(self.output_x, y_pos, 0),
                size=mp.Vector3(0, region_height, 0)  # Small vertical segment
            )
            flux_monitor = self.sim.add_flux(frequency, 0, 1, flux_region)
            flux_monitors.append(flux_monitor)

        self.flux_regions = flux_monitors
        return flux_monitors

    def add_input_flux_monitor(self):
        """
        Add flux monitor at the input waveguide
        """

        frequency = 1.0 / self.wavelength

        if self.sim is None:
            raise ValueError(
                "Simulation must be created first. Call create_simulation() method.")
        self.input_flux_region = mp.FluxRegion(
            center=mp.Vector3(self.input_flux_monitor_x, 0, 0),
            size=mp.Vector3(0, self.waveguide_width, 0)
        )
        self.input_flux_region = self.sim.add_flux(
            frequency, 0, 1, self.input_flux_region)
        return self.input_flux_region

    def add_output_flux_monitors(self):
        """
        Add flux monitors at the output waveguides
        """
        frequency = 1.0 / self.wavelength
        if self.sim is None:
            raise ValueError(
                "Simulation must be created first. Call create_simulation() method.")
        self.output_flux_region_1 = mp.FluxRegion(
            center=mp.Vector3(self.output_flux_monitor_x, self.output_y_separation, 0),
            size=mp.Vector3(0, self.waveguide_width, 0)
        )
        self.output_flux_region_1 = self.sim.add_flux(
            frequency, 0, 1, self.output_flux_region_1)

        self.output_flux_region_2 = mp.FluxRegion(
            center=mp.Vector3(self.output_flux_monitor_x, -self.output_y_separation, 0),
            size=mp.Vector3(0, self.waveguide_width, 0)
        )
        self.output_flux_region_2 = self.sim.add_flux(
            frequency, 0, 1, self.output_flux_region_2)
        return self.output_flux_region_1, self.output_flux_region_2
    
    def add_design_region_flux_monitor(self):
        """
        Add flux monitor at the design region
        """
        frequency = 1.0 / self.wavelength
        if self.sim is None:
            raise ValueError(
                "Simulation must be created first. Call create_simulation() method.")
        self.design_region_flux_region_up = mp.FluxRegion(
            center=mp.Vector3(0, 1, 0),
            size=mp.Vector3(self.design_region_x, 0, 0)
        )
        self.design_region_flux_region_up = self.sim.add_flux(
            frequency, 0, 1, self.design_region_flux_region_up)
        
        self.design_region_flux_region_down = mp.FluxRegion(
            center=mp.Vector3(0, -1, 0),
            size=mp.Vector3(self.design_region_x, 0, 0)
        )
        self.design_region_flux_region_down = self.sim.add_flux(
            frequency, 0, 1, self.design_region_flux_region_down)
        
        self.design_region_flux_region_left = mp.FluxRegion(
            center=mp.Vector3(-1, 0, 0),
            size=mp.Vector3(0, self.design_region_y, 0)
        )
        self.design_region_flux_region_left = self.sim.add_flux(
            frequency, 0, 1, self.design_region_flux_region_left)
        
        self.design_region_flux_region_right = mp.FluxRegion(
            center=mp.Vector3(1, 0, 0),
            size=mp.Vector3(0, self.design_region_y, 0)
        )
        self.design_region_flux_region_right = self.sim.add_flux(
            frequency, 0, 1, self.design_region_flux_region_right)
        return self.design_region_flux_region_up, self.design_region_flux_region_down, self.design_region_flux_region_left, self.design_region_flux_region_right

    def get_flux_distribution_along_y(self):
        """
        Get flux values for all y-axis flux monitors

        Returns:
            y_positions: array of y coordinates
            flux_values: array of flux values at each y position
        """
        if not self.flux_regions:
            raise ValueError(
                "No flux regions along y-axis. Call add_flux_monitors_along_y() first.")

        # Get flux values for all monitors
        # mp.get_fluxes returns a list/array, typically [flux_value] for single frequency
        flux_values = []
        for flux_monitor in self.flux_regions:
            fluxes = mp.get_fluxes(flux_monitor)
            # Extract the flux value (first element if it's a list/array)
            if isinstance(fluxes, (list, np.ndarray)):
                flux_value = fluxes[0] if len(fluxes) > 0 else 0.0
            else:
                flux_value = fluxes
            flux_values.append(flux_value)

        flux_values = np.array(flux_values)

        # Calculate y positions
        y_min = -self.cell_size.y / 2
        y_max = self.cell_size.y / 2
        num_regions = len(self.flux_regions)
        region_height = self.cell_size.y / num_regions
        y_positions = np.linspace(
            y_min + region_height/2, y_max - region_height/2, num_regions)

        return y_positions, flux_values

    def get_input_flux_value(self):
        """
        Get flux value at the input waveguide
        """
        if self.input_flux_region is None:
            raise ValueError(
                "No flux monitor added. Call add_flux_monitor() first.")
        return mp.get_fluxes(self.input_flux_region)[0]

    def get_output_flux_values_1(self):
        """
        Get flux values at the output waveguides
        """
        if self.output_flux_region_1 is None:
            raise ValueError(
                "No output flux monitors added. Call add_output_flux_monitors() first.")
        return mp.get_fluxes(self.output_flux_region_1)[0]

    def get_output_flux_values_2(self):
        """
        Get flux values at the output waveguides
        """
        if self.output_flux_region_2 is None:
            raise ValueError(
                "No output flux monitors added. Call add_output_flux_monitors() first.")
        return mp.get_fluxes(self.output_flux_region_2)[0]


    def get_design_region_flux_value(self):

        return mp.get_fluxes(self.design_region_flux_region_up)[0], mp.get_fluxes(self.design_region_flux_region_down)[0], mp.get_fluxes(self.design_region_flux_region_left)[0], mp.get_fluxes(self.design_region_flux_region_right)[0]

    def run(self):
        """Run the simulation"""
        if self.sim is None:
            self.create_simulation()
        # Suppress Meep's verbose output during simulation run
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                self.sim.run(until=self.simulation_time)
        # self.plot_design(save_path=None, show_plot=False)
        return self.sim

    def get_ezfield_data(self):
        """Get electric field data from simulation"""
        if self.sim is None:
            raise ValueError(
                "Simulation must be run first. Call run() method.")

        # Get field data
        ez_data = self.sim.get_array(
            center=mp.Vector3(0, 0, 0),
            size=self.cell_size,
            component=mp.Ez
        )

        # Transpose so indexing matches coordinate system: ez_data[x_idx, y_idx]
        self.ez_data = ez_data.T
        return self.ez_data

    def get_hzfield_data(self):
        """Get electric field data from simulation"""
        if self.sim is None:
            raise ValueError(
                "Simulation must be run first. Call run() method.")

        # Get field data
        hz_data = self.sim.get_array(
            center=mp.Vector3(0, 0, 0),
            size=self.cell_size,
            component=mp.Hz
        )

        # Transpose so indexing matches coordinate system: hz_data[x_idx, y_idx]
        self.hz_data = hz_data.T
        return self.hz_data

    def plot_design(self, material_matrix=None, save_path=None, show_plot=True):
        """
        Plot and visualize the simulation results (Ez field + geometry overlays).
        This version includes input/output waveguides and the output measurement plane.

        Args:
            material_matrix: 2D array (pixel_num_x x pixel_num_y) where 1=silicon, 0=silica.
                           If provided, will overlay material distribution as grey boxes.
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        if self.ez_data is None:
            self.get_ezfield_data()
        
        if self.hz_data is None:
            self.get_hzfield_data()

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'meep_2d_result_{timestamp}.png'

        plt.figure(figsize=(12, 6))
        ax = plt.gca()  # Get current axes for adding patches

        # Set extent dynamically based on cell_size
        extent = [-self.cell_size.x/2, self.cell_size.x/2,
                  -self.cell_size.y/2, self.cell_size.y/2]

        # Plot electric field
        # Transpose ez_data because imshow expects (rows, cols) where rows=y, cols=x
        # Meep's get_array gives (x, y), so transpose for correct orientation
        field_magnitude = np.abs(self.hz_data)
        plt.imshow(field_magnitude, interpolation='spline36', cmap='viridis',
                   aspect='auto', extent=extent, origin='lower')
        plt.colorbar(label='Hz (electric field)')
        plt.xlabel('x (microns)')
        plt.ylabel('y (microns)')
        plt.title(
            f'Waveguide Simulation - Hz Field (L_in={self.input_coupler_length}µm, L_out={self.output_coupler_length}µm)')

        # --- 0. Overlay Material Matrix (if provided) ---
        silicon_label_added = False
        silica_label_added = False
        if material_matrix is not None:
            material_matrix = np.array(material_matrix)
            if material_matrix.shape == (self.pixel_num_x, self.pixel_num_y):
                square_x_min = self.design_region_x_min  # -1.0
                square_y_min = self.design_region_y_min  # -1.0
                dx = self.pixel_size
                dy = self.pixel_size

                # Plot each pixel as a rectangle
                for i in range(self.pixel_num_x):
                    for j in range(self.pixel_num_y):
                        # Calculate lower-left corner position
                        x_left = square_x_min + i * dx
                        y_bottom = square_y_min + j * dy

                        if material_matrix[i, j] == 1:
                            # Silicon - darker grey
                            ax.add_patch(Rectangle(
                                (x_left, y_bottom), dx, dy,
                                facecolor='darkgrey', edgecolor='none', alpha=0.4,
                                label='Silicon' if not silicon_label_added else ''
                            ))
                            if not silicon_label_added:
                                silicon_label_added = True
                        else:
                            # Silica - lighter grey
                            ax.add_patch(Rectangle(
                                (x_left, y_bottom), dx, dy,
                                facecolor='lightgrey', edgecolor='none', alpha=0.4,
                                label='Silica' if not silica_label_added else ''
                            ))
                            if not silica_label_added:
                                silica_label_added = True

        # --- 1. Overlays: Input Waveguide ---
        input_waveguide_x_start = self.design_region_x_min - self.input_coupler_length
        input_waveguide_y_start = -self.waveguide_width / 2

        ax.add_patch(Rectangle(
            (input_waveguide_x_start, input_waveguide_y_start),
            self.input_coupler_length, self.waveguide_width,
            linewidth=2.5, edgecolor='yellow', facecolor='none', linestyle='-', alpha=0.9,
            label='Input Waveguide Outline'
        ))

        # --- 2. Overlays: Output Waveguides ---
        # Use the same fixed separation as in create_geometry
        output_y_separation = self.output_y_separation
        output_waveguide_x_start = self.design_region_x_max
        output_waveguide_length = self.output_coupler_length

        # Output Waveguide 1 (Top)
        output1_waveguide_y_start = output_y_separation - self.waveguide_width / 2
        ax.add_patch(Rectangle(
            (output_waveguide_x_start, output1_waveguide_y_start),
            output_waveguide_length, self.waveguide_width,
            linewidth=2.5, edgecolor='orange', facecolor='none', linestyle='-', alpha=0.9,
            label='Output Waveguide Outline'  # Only label once
        ))
        # Output Waveguide 2 (Bottom)
        output2_waveguide_y_start = -output_y_separation - self.waveguide_width / 2
        ax.add_patch(Rectangle(
            (output_waveguide_x_start, output2_waveguide_y_start),
            output_waveguide_length, self.waveguide_width,
            linewidth=2.5, edgecolor='orange', facecolor='none', linestyle='-', alpha=0.9
            # No label here to avoid duplicate legend entry
        ))
        # --- 3. Overlays: Design Region ---
        ax.add_patch(Rectangle(
            # Lower-left corner: (-1.0, -1.0)
            (self.design_region_x_min, self.design_region_y_min),
            2.0, 2.0,  # Width=2.0, Height=2.0
            linewidth=2.5, edgecolor='lime', facecolor='none',
            linestyle='--', alpha=0.9, label='Design Region (2x2um)')
        )

        # --- 4. Overlays: Output Measurement Plane ---
        # Draw a vertical dashed line at self.output_x
        plt.axvline(x=self.output_x, color='red', linestyle=':', linewidth=2,
                    label=f'Output Flux Plane (x={self.output_x}µm)')

        # --- 5. Legend and Final Setup ---
        # Use ax.legend() to collect labels from patches
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicate labels (e.g., if 'Output Waveguide Outline' appears twice)
        unique_handles = {}
        for h, l in zip(handles, labels):
            # Dict automatically handles duplicates, keeping last one
            unique_handles[l] = h

        # FIX: Move legend outside the plot area
        ax.legend(unique_handles.values(), unique_handles.keys(),
                  # Tells the legend to position its lower-right corner at the anchor point
                  loc='lower right',
                  # Sets the anchor point to the bottom-left corner of the axes (x=0, y=0)
                  bbox_to_anchor=(0.3, -0.4),
                  borderaxespad=0.)        # No padding between the legend and the anchor point

        # FIX: Ensure 1:1 scale for the axes
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()  # IMPORTANT: Adjust plot to make room for the legend

        # Save and show
        if save_path:
            try:
                # Create directory if it doesn't exist
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Simulation results saved to '{save_path}'")
            except (FileNotFoundError, OSError) as e:
                print(f"Warning: Could not save plot to '{save_path}': {e}")
            except Exception as e:
                print(f"Error saving plot: {e}")
        print(
            f"Waveguide: {self.waveguide_width}um wide, {self.waveguide_index} index")
        print(
            f"EigenModeSource: Continuous wave at {self.wavelength}um, from left")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_distribution(self, output_all_flux, input_flux, save_path=None, show_plot=True):
        """
        Plot the flux distribution along the output plane.

        Args:
            output_all_flux: 1D array of flux values at each detector position
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(output_all_flux/input_flux, 'b-',
                 linewidth=2, label='Flux Distribution')
        plt.xlabel('Detector Index')
        plt.ylabel('Flux Ratio (Output/Input)')
        plt.title('Flux Distribution Ratio at Output Plane')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            try:
                # Create directory if it doesn't exist
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Flux distribution plot saved to '{save_path}'")
            except (FileNotFoundError, OSError) as e:
                print(f"Warning: Could not save plot to '{save_path}': {e}")
            except Exception as e:
                print(f"Error saving plot: {e}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def calculate_flux(self, material_matrix):
        # Create simulation

        # Create geometry with material matrix
        self.create_geometry(material_matrix=material_matrix)

        # Create simulation and add flux monitors
        self.create_simulation()
        self.add_flux_monitors_along_y()
        self.add_input_flux_monitor()
        self.add_output_flux_monitors()

        # Run simulation
        self.run()

        # Get flux distribution
        _, output_all_flux = self.get_flux_distribution_along_y()
        input_flux_value = self.get_input_flux_value()
        output_flux_value_1 = self.get_output_flux_values_1()
        output_flux_value_2 = self.get_output_flux_values_2()

        # Get field data
        ez_data = self.get_field_data()

        return input_flux_value, output_flux_value_1, output_flux_value_2, output_all_flux, ez_data

import time

if __name__ == "__main__":
    start = time.time() 
    # Example 1: Standard centered setup
    calculator_A = WaveguideSimulation()

    # Create a simple test matrix
    material_matrix = np.ones((20, 20))
    # material_matrix[25, :] = 0

    # Apply the geometry
    calculator_A.create_geometry(material_matrix=material_matrix)

    # Plot to verify the new centering and lengths
    print("Plotting Centered Geometry (Design at x=[-1, 1])")
    calculator_A.plot_geometry(
        show_plot=False,
        save_path='sample_img/geometry_plot.png',  # Provide a file name here
        x_range=(-3.0, 3.0),
        y_range=(-2, 2)
    )

    # Example 2: Run a quick simulation test with the new geometry
    calculator_A.create_simulation()

    # Since the outputs are separated, define flux monitor location at x=2.5
    calculator_A.add_flux_monitors_along_y()  # Add monitors to measure flux split
    calculator_A.add_input_flux_monitor()
    calculator_A.add_output_flux_monitors()
    calculator_A.add_design_region_flux_monitor()

    print("\nRunning simulation with centered geometry...")
    calculator_A.run()
    print("output_x",calculator_A.output_x)
    calculator_A.plot_design(
        material_matrix=material_matrix,
        show_plot=False,
        save_path='sample_img/meep_simulation_hz_field.png'  # Provide a file name here
    )

    # Get total flux
    _, flux_values = calculator_A.get_flux_distribution_along_y()
    print(f"Total flux measured: {np.sum(flux_values):.4e}")
    # get input flux
    input_flux = calculator_A.get_input_flux_value()
    calculator_A.plot_distribution(
        output_all_flux=flux_values,
        input_flux=input_flux,
        show_plot=False,
        save_path='sample_img/flux_distribution.png'  # Provide a file name here
    )

    design_region_flux_value_up, design_region_flux_value_down, design_region_flux_value_left, design_region_flux_value_right = calculator_A.get_design_region_flux_value()
    print(f"Design region flux value up: {design_region_flux_value_up:.4e}")
    print(f"Design region flux value down: {design_region_flux_value_down:.4e}")
    print(f"Design region flux value left: {design_region_flux_value_left:.4e}")
    print(f"Design region flux value right: {design_region_flux_value_right:.4e}")

    output_flux_value_1 = calculator_A.get_output_flux_values_1()
    output_flux_value_2 = calculator_A.get_output_flux_values_2()
    print(f"Input flux: {input_flux:.4e}")
    print(f"Output flux value 1: {output_flux_value_1:.4e}")
    print(f"Output flux value 2: {output_flux_value_2:.4e}")

    end = time.time()       # Record end time
    print("Time elapsed:", end - start, "seconds")
