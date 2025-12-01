"""
2D Meep simulation with waveguide and eigenmode source
- 2um x 2um square region
- Thin waveguide extending from left into the square
- EigenModeSource with continuous wave at 1550nm, entering from left
"""

import time

from sympy.logic.boolalg import true
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
        self.hz_data = None
        
        # Electric field monitor for state
        self.efield_monitor = None  # Electric field monitor for state (along y-axis)
        self.hfield_monitor = None  # Magnetic field monitor for state (along y-axis)
        self.efield_region_y_positions = []  # Y-coordinates of state efield monitor
        self.hfield_region_y_positions = []  # Y-coordinates of state hfield monitor
        self.input_flux_region = None  # Input mode flux monitor
        self.output_flux_region_1 = None  # Output mode flux monitor 1
        self.output_flux_region_2 = None  # Output mode flux monitor 2
        self.design_region_flux_region_up = None
        self.design_region_flux_region_down = None
        self.design_region_flux_region_left = None
        self.design_region_flux_region_right = None
        self.num_flux_regions = config.simulation.num_flux_regions
        self.simulation_time = config.simulation.simulation_time
        self.state_output_x = config.simulation.state_output_x
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

    def create_geometry(self, matrix=None):
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
        if matrix is not None:
            matrix = np.array(matrix)

            if matrix.shape != (self.pixel_num_x, self.pixel_num_y):
                # Error handling remains the same
                raise ValueError(
                    f"matrix must be {self.pixel_num_x}x{self.pixel_num_y}, got shape {matrix.shape}")

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

                    if matrix[i, j] == 1:
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
            self.create_geometry(matrix=None)

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

    def create_source(self):
        """Create eigenmode source at left edge of input coupler"""
        # Calculate the starting point of the input waveguide (cell's left edge)
        input_coupler_start_x = self.design_region_x_min - self.input_coupler_length

        # Suppress MPB solver output when creating EigenModeSource
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                sources = [mp.EigenModeSource(
                    src=mp.ContinuousSource(
                        wavelength=self.wavelength,
                        width=20
                    ),
                    # Position source at the start of the input coupler
                    center=mp.Vector3(input_coupler_start_x *
                                      self.src_pos_shift_coeff, 0.0, 0),
                    size=mp.Vector3(0, self.waveguide_width, 0),
                    eig_band=2,
                    direction=mp.NO_DIRECTION,
                    eig_kpoint=mp.Vector3(1, 0, 0),  # +X direction (rightward)
                    eig_match_freq=True
                )]
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
        
        self.sources = sources
        return self.sources

    def create_simulation(self):
        """
        Create Meep simulation object

        Args:
            add_flux_at_x: if provided, add flux monitor at this x position
        """
        if self.geometry is None:
            self.create_geometry()
        if self.sources is None:
            self.create_source()

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


        return self.sim

    def add_efield_monitor_state(self):
        """
        Add electric field monitor along y-axis at a specific x position for state observation.
        Uses DFT (Discrete Fourier Transform) to monitor the electric field.

        Returns:
            DFT field monitor object
        """
        if self.sim is None:
            raise ValueError(
                "Simulation must be created first. Call create_simulation() method.")

        # Calculate frequency from wavelength
        frequency = 1.0 / self.wavelength

        # Calculate y positions for sampling
        # y spans from -cell_size.y/2 to +cell_size.y/2
        y_min = -self.cell_size.y / 2
        y_max = self.cell_size.y / 2
        
        # Create a vertical line at state_output_x spanning the full y range
        # We'll sample at num_flux_regions points along y
        y_positions = np.linspace(y_min, y_max, self.num_flux_regions)
        
        # Store y positions for plotting
        self.efield_region_y_positions = y_positions.copy()

        # Create DFT field monitor for electric field (Ez component for 2D TM mode)
        # The monitor is a vertical line at x = state_output_x
        efield_region = mp.Volume(
            center=mp.Vector3(self.state_output_x, 0, 0),
            size=mp.Vector3(0, self.cell_size.y, 0)  # Vertical line spanning full height
        )
        
        # Add DFT fields monitor for Ez component
        # For single frequency, use fcen, df, nfreq format (3 numbers)
        # fcen = center frequency, df = frequency width, nfreq = number of frequencies
        self.efield_monitor = self.sim.add_dft_fields(
            [mp.Ez],  # Monitor Ez component (for 2D TM mode)
            frequency, 0.0, 1,  # fcen, df, nfreq (single frequency: center freq, zero width, 1 frequency)
            center=efield_region.center,
            size=efield_region.size
        )
        
        return self.efield_monitor

    def add_hfield_monitor_state(self):
        """
        Add magnetic field monitor along y-axis at a specific x position for state observation.
        Uses DFT (Discrete Fourier Transform) to monitor the magnetic field.

        Returns:
            DFT field monitor object
        """
        if self.sim is None:
            raise ValueError(
                "Simulation must be created first. Call create_simulation() method.")

        # Calculate frequency from wavelength
        frequency = 1.0 / self.wavelength

        # Calculate y positions for sampling
        # y spans from -cell_size.y/2 to +cell_size.y/2
        y_min = -self.cell_size.y / 2
        y_max = self.cell_size.y / 2
        
        # Create a vertical line at state_output_x spanning the full y range
        # We'll sample at num_flux_regions points along y
        y_positions = np.linspace(y_min, y_max, self.num_flux_regions)
        
        # Store y positions for plotting
        self.hfield_region_y_positions = y_positions.copy()

        # Create DFT field monitor for electric field (Ez component for 2D TM mode)
        # The monitor is a vertical line at x = state_output_x
        hfield_region = mp.Volume(
            center=mp.Vector3(self.state_output_x, 0, 0),
            size=mp.Vector3(0, self.cell_size.y, 0)  # Vertical line spanning full height
        )
        
        # Add DFT fields monitor for Ez component
        # For single frequency, use fcen, df, nfreq format (3 numbers)
        # fcen = center frequency, df = frequency width, nfreq = number of frequencies
        self.hfield_monitor = self.sim.add_dft_fields(
            [mp.Hz],  # Monitor Hz component
            frequency, 0.0, 1,  # fcen, df, nfreq (single frequency: center freq, zero width, 1 frequency)
            center=hfield_region.center,
            size=hfield_region.size
        )
        
        return self.hfield_monitor

    def add_flux_monitor_input_mode(self):
        """
        Add flux monitor at the input waveguide for mode analysis
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

    def add_flux_monitor_output_mode(self):
        """
        Add flux monitors at the output waveguides for mode analysis
        """
        frequency = 1.0 / self.wavelength
        if self.sim is None:
            raise ValueError(
                "Simulation must be created first. Call create_simulation() method.")
        self.output_flux_region_1 = mp.FluxRegion(
            center=mp.Vector3(self.output_flux_monitor_x,
                              self.output_y_separation, 0),
            size=mp.Vector3(0, self.waveguide_width, 0)
        )
        self.output_flux_region_1 = self.sim.add_flux(
            frequency, 0, 1, self.output_flux_region_1)

        self.output_flux_region_2 = mp.FluxRegion(
            center=mp.Vector3(self.output_flux_monitor_x, -
                              self.output_y_separation, 0),
            size=mp.Vector3(0, self.waveguide_width, 0)
        )
        self.output_flux_region_2 = self.sim.add_flux(
            frequency, 0, 1, self.output_flux_region_2)
        return self.output_flux_region_1, self.output_flux_region_2

    def add_flux_monitor_rectangle(self):
        """
        Add flux monitors at the design region boundaries (rectangle)
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

    def get_efield_state(self):
        """
        Get electric field values for state monitor (along y-axis).
        Returns the magnitude squared of the electric field (|Ez|^2).

        Returns:
            efield_state: array of |Ez|^2 values at each y position
        """
        if self.efield_monitor is None:
            raise ValueError(
                "No electric field monitor. Call add_efield_monitor_state() first.")

        # Get electric field data from DFT monitor
        # get_dft_array returns the field data as a numpy array
        # For a vertical line monitor, this should be a 1D array along y
        efield_data = self.sim.get_dft_array(self.efield_monitor, mp.Ez, 0)
        
        # Extract values along the y-axis
        # For a vertical line monitor, Meep typically returns a 1D array
        if efield_data.ndim == 1:
            # Direct 1D array along y-axis
            efield_values = np.abs(efield_data) ** 2
        elif efield_data.ndim == 2:
            # If 2D, extract the column (for vertical line, should be single column)
            # Take the middle column or first column depending on shape
            if efield_data.shape[1] == 1:
                efield_values = np.abs(efield_data[:, 0]) ** 2
            else:
                # Multiple columns - take middle column
                mid_x = efield_data.shape[1] // 2
                efield_values = np.abs(efield_data[:, mid_x]) ** 2
        else:
            # Fallback: flatten and take first num_flux_regions elements
            efield_values = np.abs(efield_data.flatten()[:self.num_flux_regions]) ** 2
        
        # Ensure we have the right number of values
        if len(efield_values) != self.num_flux_regions:
            # Resample to match num_flux_regions
            if len(efield_values) > self.num_flux_regions:
                # Downsample by taking evenly spaced indices
                indices = np.linspace(0, len(efield_values) - 1, self.num_flux_regions, dtype=int)
                efield_values = efield_values[indices]
            else:
                # Upsample by linear interpolation using numpy
                old_indices = np.linspace(0, 1, len(efield_values))
                new_indices = np.linspace(0, 1, self.num_flux_regions)
                efield_values = np.interp(new_indices, old_indices, efield_values)

        return np.array(efield_values)

    def get_hfield_state(self):
        """
        Get magnetic field values for state monitor (along y-axis).
        Returns the magnitude squared of the magnetic field (|Hz|^2).

        Returns:
            hfield_state: array of |Hz|^2 values at each y position
        """
        if self.hfield_monitor is None:
            raise ValueError(
                "No magnetic field monitor. Call add_hfield_monitor_state() first.")

        # Get electric field data from DFT monitor
        # get_dft_array returns the field data as a numpy array
        # For a vertical line monitor, this should be a 1D array along y
        hfield_data = self.sim.get_dft_array(self.hfield_monitor, mp.Hz, 0)
        
        # Extract values along the y-axis
        # For a vertical line monitor, Meep typically returns a 1D array
        if hfield_data.ndim == 1:
            # Direct 1D array along y-axis
            hfield_values = np.abs(hfield_data) ** 2
        elif hfield_data.ndim == 2:
            # If 2D, extract the column (for vertical line, should be single column)
            # Take the middle column or first column depending on shape
            if hfield_data.shape[1] == 1:
                hfield_values = np.abs(hfield_data[:, 0]) ** 2
            else:
                # Multiple columns - take middle column
                mid_x = hfield_data.shape[1] // 2
                hfield_values = np.abs(hfield_data[:, mid_x]) ** 2
        else:
            # Fallback: flatten and take first num_flux_regions elements
            hfield_values = np.abs(hfield_data.flatten()[:self.num_flux_regions]) ** 2
        
        # Ensure we have the right number of values
        if len(hfield_values) != self.num_flux_regions:
            # Resample to match num_flux_regions
            if len(hfield_values) > self.num_flux_regions:
                # Downsample by taking evenly spaced indices
                indices = np.linspace(0, len(hfield_values) - 1, self.num_flux_regions, dtype=int)
                hfield_values = hfield_values[indices]
            else:
                # Upsample by linear interpolation using numpy
                old_indices = np.linspace(0, 1, len(hfield_values))
                new_indices = np.linspace(0, 1, self.num_flux_regions)
                hfield_values = np.interp(new_indices, old_indices, hfield_values)

        return np.array(hfield_values)

    def get_flux_input_mode(self, band_num=2):
        """
        Get input mode coefficient (power) using eigenmode expansion and raw flux value.
        
        Args:
            band_num: The mode band number (2 = second mode TE0)
            
        Returns:
            (raw_flux, mode_power): Tuple of raw flux value and mode power (|alpha|^2) for the specified mode at input
        """
        if self.sim is None:
            raise ValueError(
                "Simulation must be run first. Call run() method.")
        if self.input_flux_region is None:
            raise ValueError(
                "No input flux monitor added. Call add_flux_monitor_input_mode() first.")
        
        # Get raw flux value
        raw_flux = mp.get_fluxes(self.input_flux_region)[0]
        
        # Get eigenmode coefficients at input (suppress MPB output)
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                res = self.sim.get_eigenmode_coefficients(
                    self.input_flux_region, 
                    [band_num],
                    eig_parity=mp.NO_PARITY,
                    direction=mp.X
                )
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
        
        # alpha[band_idx, freq_idx, direction_idx]
        # direction_idx=0 for forward (+X direction)
        alpha_forward = res.alpha[0, 0, 0]
        
        # Mode power = |alpha|^2
        mode_power = np.abs(alpha_forward) ** 2
        
        return raw_flux, mode_power

    def get_flux_output_mode(self, band_num=2):
        """
        Get output mode coefficients (transmission) and raw flux values for both output waveguides.
        
        Args:
            band_num: The mode band number (2 = second mode TE0)
            
        Returns:
            (raw_flux_1, raw_flux_2, mode_1, mode_2): Tuple of raw flux values and mode transmission powers (|alpha|^2) for output 1 and 2
        """
        if self.sim is None:
            raise ValueError(
                "Simulation must be run first. Call run() method.")
        if self.output_flux_region_1 is None or self.output_flux_region_2 is None:
            raise ValueError(
                "No output flux monitors added. Call add_flux_monitor_output_mode() first.")
        
        # Get raw flux values
        raw_flux_1 = mp.get_fluxes(self.output_flux_region_1)[0]
        raw_flux_2 = mp.get_fluxes(self.output_flux_region_2)[0]
        
        # Get eigenmode coefficients for output 1 and 2 (suppress MPB output)
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                res1 = self.sim.get_eigenmode_coefficients(
                    self.output_flux_region_1, 
                    [band_num],
                    eig_parity=mp.NO_PARITY,
                    direction=mp.X
                )
                res2 = self.sim.get_eigenmode_coefficients(
                    self.output_flux_region_2, 
                    [band_num],
                    eig_parity=mp.NO_PARITY,
                    direction=mp.X
                )
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
        
        alpha_forward_1 = res1.alpha[0, 0, 0]
        mode_transmission_1 = np.abs(alpha_forward_1) ** 2
        alpha_forward_2 = res2.alpha[0, 0, 0]
        mode_transmission_2 = np.abs(alpha_forward_2) ** 2
        diff_transmission = abs(mode_transmission_1 - mode_transmission_2)
        return raw_flux_1, raw_flux_2, mode_transmission_1, mode_transmission_2, diff_transmission

    def get_flux_rectangle(self):
        """
        Get flux values at the design region rectangle boundaries.
        
        Returns:
            (up, down, left, right): Tuple of flux values at the four boundaries
        """
        if self.design_region_flux_region_up is None:
            raise ValueError(
                "No rectangle flux monitors added. Call add_flux_monitor_rectangle() first.")
        
        return (
            mp.get_fluxes(self.design_region_flux_region_up)[0],
            mp.get_fluxes(self.design_region_flux_region_down)[0],
            mp.get_fluxes(self.design_region_flux_region_left)[0],
            mp.get_fluxes(self.design_region_flux_region_right)[0]
        )

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

    def get_output_transmission(self, band_num=2):
        """
        Calculate transmission ratios for output modes.
        
        Args:
            band_num: The mode band number (2 = second mode TE0)
            
        Returns:
            (transmission_1, transmission_2, total_transmission): 
            Transmission ratios relative to input mode
        """
        _, input_mode = self.get_flux_input_mode(band_num)
        _, _, output_mode_1, output_mode_2, diff_transmission = self.get_flux_output_mode(band_num)
        
        transmission_1 = output_mode_1
        transmission_2 = output_mode_2
        total_transmission = transmission_1 + transmission_2
        
        return transmission_1, transmission_2, total_transmission, diff_transmission

    def plot_design(self, matrix=None, save_path=None, show_plot=True):
        """
        Plot and visualize the simulation results (Hz field + geometry overlays).
        This version includes input/output waveguides and the output measurement plane.

        Args:
            material_matrix: 2D array (pixel_num_x x pixel_num_y) where 1=silicon, 0=silica.
                           If provided, will overlay material distribution as grey boxes.
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
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

        # Plot magnetic field
        # Transpose hz_data because imshow expects (rows, cols) where rows=y, cols=x
        # Meep's get_array gives (x, y), so transpose for correct orientation
        field_magnitude = np.abs(self.hz_data)
        plt.imshow(field_magnitude, interpolation='spline36', cmap='viridis',
                   aspect='auto', extent=extent, origin='lower')
        plt.colorbar(label='Hz (magnetic field)')
        plt.xlabel('x (microns)')
        plt.ylabel('y (microns)')
        plt.title(
            f'Waveguide Simulation - Hz Field (L_in={self.input_coupler_length}µm, L_out={self.output_coupler_length}µm)')

        # --- 0. Overlay Material Matrix (if provided) ---
        silicon_label_added = False
        silica_label_added = False
        if matrix is not None:
            matrix = np.array(matrix)
            if matrix.shape == (self.pixel_num_x, self.pixel_num_y):
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

                        if matrix[i, j] == 1:
                            # Silicon - darker grey
                            ax.add_patch(Rectangle(
                                (x_left, y_bottom), dx, dy,
                                facecolor='black', edgecolor='none', alpha=0.4,
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
        # Draw a vertical dashed line at self.state_output_x
        plt.axvline(x=self.state_output_x, color='red', linestyle=':', linewidth=2,
                    label=f'Output Flux Plane (x={self.state_output_x}µm)')

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


    def plot_distribution(self, efield_state, save_path=None, show_plot=True):
        """
        Plot the electric field distribution along the output plane.

        Args:
            efield_state: 1D array of |Ez|^2 values at each detector position
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
        """
        # Use y-coordinates as x-axis if available
        if len(self.efield_region_y_positions) == len(efield_state):
            x_data = self.efield_region_y_positions
            x_label = 'Y Position (μm)'
        else:
            # Fallback to index if y positions not available
            x_data = np.arange(len(efield_state))
            x_label = 'Detector Index'
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, efield_state, 'b-',
                 linewidth=2, label='Electric Field |Ez|²')
        plt.xlabel(x_label)
        plt.ylabel('|Ez|²')
        plt.title('Electric Field Distribution at Output Plane')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            try:
                # Create directory if it doesn't exist
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Electric field distribution plot saved to '{save_path}'")
            except (FileNotFoundError, OSError) as e:
                print(f"Warning: Could not save plot to '{save_path}': {e}")
            except Exception as e:
                print(f"Error saving plot: {e}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_hfield_distribution(self, hfield_state, save_path=None, show_plot=True):
        """
        Plot the magnetic field distribution along the output plane.

        Args:
            hfield_state: 1D array of |Hz|^2 values at each detector position
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
        """
        # Use y-coordinates as x-axis if available
        if len(self.efield_region_y_positions) == len(hfield_state):
            x_data = self.hfield_region_y_positions
            x_label = 'Y Position (μm)'
        else:
            # Fallback to index if y positions not available
            x_data = np.arange(len(hfield_state))
            x_label = 'Detector Index'
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, hfield_state, 'b-',
                 linewidth=2, label='Magnetic Field |Hz|²')
        plt.xlabel(x_label)
        plt.ylabel('|Hz|²')
        plt.title('Magnetic Field Distribution at Output Plane')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            try:
                # Create directory if it doesn't exist
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Magnetic field distribution plot saved to '{save_path}'")
            except (FileNotFoundError, OSError) as e:
                print(f"Warning: Could not save plot to '{save_path}': {e}")
            except Exception as e:
                print(f"Error saving plot: {e}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def calculate_flux(self, matrix):
        """
        Complete simulation workflow: create geometry, run simulation, and get all results.
        
        Args:
            matrix: Material matrix (pixel_num_x x pixel_num_y), 1=silicon, 0=silica
            
        Returns:
            (input_mode_flux, output_mode_flux_1, output_mode_flux_2, efield_state, hz_data, 
             input_mode, output_mode_1, output_mode_2)
        """
        # Create geometry with material matrix
        self.create_geometry(matrix=matrix)

        # Create simulation and add monitors
        self.create_simulation()
        # self.add_efield_monitor_state()
        self.add_hfield_monitor_state()
        self.add_flux_monitor_input_mode()
        self.add_flux_monitor_output_mode()

        # Run simulation
        self.run()

        # Get electric field state (distribution along y-axis)
        hfield_state = self.get_hfield_state()  # Returns |Hz|^2 values
        
        # Get input and output flux values and mode coefficients (using existing functions)
        input_mode_flux, input_mode = self.get_flux_input_mode(band_num=2) 
        output_mode_flux_1, output_mode_flux_2, output_mode_1, output_mode_2, _ = self.get_flux_output_mode(band_num=2)

        return input_mode_flux, output_mode_flux_1, output_mode_flux_2, hfield_state, input_mode, output_mode_1, output_mode_2


if __name__ == "__main__":
    """
    Simplest usage example:
    1. Create simulation instance
    2. Define material matrix (design)
    3. Run calculate_flux() to get results
    """
    # 1. Create simulation instance
    sim = WaveguideSimulation()
    
    # 2. Define material matrix (pixel_num_x x pixel_num_y pixels)
    # 1 = silicon, 0 = silica
    matrix = np.ones((sim.pixel_num_x, sim.pixel_num_y))
    
    # Optional: Create a simple pattern (e.g., a vertical line of silica)
    # matrix[sim.pixel_num_x // 2, :] = 0
    
    # 3. Run simulation and get results
    results = sim.calculate_flux(matrix)
    input_mode_flux, output_mode_flux_1, output_mode_flux_2, hfield_state, input_mode, output_mode_1, output_mode_2 = results
    
    # 4. Display results
    trans_1, trans_2, total_trans, diff_trans = sim.get_output_transmission(band_num=2)
    print(f"Transmission: Output1={trans_1*100:.1f}%, Output2={trans_2*100:.1f}%, Total={total_trans*100:.1f}%, Diff={diff_trans:.6f}")
    
    # Optional: Plot results
    sim.plot_design(matrix=matrix, show_plot=true, 
                   save_path='sample_img/field_result.png')
    sim.plot_hfield_distribution(hfield_state=hfield_state,
                         save_path='sample_img/hfield_distribution.png', show_plot=False)
