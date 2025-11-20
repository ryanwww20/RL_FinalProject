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
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import redirect_stdout, redirect_stderr
from config import config

SAVE_FIG = False


class WaveguideSimulation:
    """2D Meep simulation class for waveguide with eigenmode source"""

    def __init__(self,
                 resolution=50,
                 wavelength=1.55,
                 cell_size=mp.Vector3(6, 4, 0), # Using 6x4 based on your previous example
                 pml_thickness=0.2,
                 waveguide_width=0.4, # Using 0.4 based on your previous example
                 waveguide_index=3.5,
                 # New parameters for coupling section lengths
                 input_coupler_length=1.5, # Length of waveguide before design area
                 output_coupler_length=1.5): # Length of waveguide after design area:
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
        """
        self.resolution = config.simulation.resolution
        self.wavelength = config.simulation.wavelength
        self.cell_size = config.simulation.cell_size
        self.pml_layers = [mp.PML(config.simulation.pml_thickness)]
        self.waveguide_width = config.simulation.waveguide_width
        self.waveguide_index = config.simulation.waveguide_index

        # New coupling lengths
        self.input_coupler_length = input_coupler_length
        self.output_coupler_length = output_coupler_length # Used for output 1 and 2
        
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
        self.flux = None  # Single flux monitor object
        self.flux_regions = []  # List of flux monitors for y-axis distribution
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

    def create_geometry(self, material_matrix=None):
        """
        Create waveguide geometry: 1 Input (left, connected to x=-1) 
        and 2 Outputs (right, connected to x=1) 
        with material distribution based on matrix in the 2um x 2um square region (-1 < x < 1).
        """
        geometry = []

        # --- 1. Input Waveguide (Left side) ---
        # Starts inside PML, ends exactly at the design region boundary (x=-1.0)
        input_start_x = self.design_region_x_min - self.input_coupler_length # e.g., -1.0 - 1.5 = -2.5
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
        output_end_x = self.design_region_x_max + self.output_coupler_length # e.g., 1.0 + 1.5 = 2.5
        output_length = self.output_coupler_length
        output_center_x = output_start_x + output_length / 2.0
        
        # Symmetrical output positions (use the default 0.3 or a new class parameter)
        output_y_separation = 0.6 # You can make this a configurable class property later

        # Output Waveguide 1 (Top)
        output_waveguide_1 = mp.Block(
            center=mp.Vector3(output_center_x, output_y_separation, 0),
            size=mp.Vector3(output_length, self.waveguide_width, 0),
            material=mp.Medium(index=self.waveguide_index)
        )
        geometry.append(output_waveguide_1)
        
        # Output Waveguide 2 (Bottom)
        output_waveguide_2 = mp.Block(
            center=mp.Vector3(output_center_x, -output_y_separation, 0),
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
            square_x_min = self.design_region_x_min # -1.0
            square_y_min = self.design_region_y_min # -1.0
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
        waveguide_color = 'blue'
        output_y_separation = 0.6 # Fixed separation, same as in create_geometry

        # 2a. Input Waveguide (Ends at x = -1.0)
        input_length = self.input_coupler_length
        input_x_start = self.design_region_x_min - input_length
        input_y_start = -self.waveguide_width / 2

        # 2b. Output Waveguides (Start at x = 1.0)
        output_length = self.output_coupler_length
        output_x_start = self.design_region_x_max

        output1_y_start = output_y_separation - self.waveguide_width / 2
        output2_y_start = -output_y_separation - self.waveguide_width / 2

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

        # --- 4. Mark the Design Region ---
        design_region_rect = Rectangle(
            (self.design_region_x_min, self.design_region_y_min), # Lower-left corner: (-1.0, -1.0)
            2.0, 2.0, # Width=2.0, Height=2.0
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
                bbox_to_anchor=(1.02, 1), # Place just outside the top-right corner of the axes
                borderaxespad=0.)        # No padding between the legend and the anchor point
        
        plt.grid(True, alpha=0.2)
        plt.tight_layout() # IMPORTANT: Adjust plot to make room for the legend

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
            center=mp.Vector3(input_coupler_start_x * self.src_pos_shift_coeff, 0, 0), 
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

    def plot_flux_distribution_y(self, save_path=None, show_plot=True):
        """
        Plot flux distribution along y-axis at a specific x position

        Args:
            x_position: x coordinate where flux was measured
            save_path: optional path to save the plot
            show_plot: whether to display the plot
        """
        y_positions, flux_values = self.get_flux_distribution_along_y()

        # Add timestamp to filename if not provided
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'flux_distribution_x{self.output_x}_{timestamp}.png'

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(y_positions, flux_values, 'b-', linewidth=2,
                 label=f'Flux at x = {self.output_x}')
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        plt.xlabel('y (microns)')
        plt.ylabel('Flux')
        plt.title(f'Flux Distribution along Y-axis at x = {self.output_x}μm')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add statistics
        total_flux = np.sum(
            flux_values) * (y_positions[1] - y_positions[0]) if len(y_positions) > 1 else flux_values[0]
        max_flux = np.max(flux_values)
        min_flux = np.min(flux_values)
        plt.text(0.02, 0.98,
                 f'Total flux: {total_flux:.6e}\n'
                 f'Max: {max_flux:.6e}\n'
                 f'Min: {min_flux:.6e}',
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if SAVE_FIG:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Flux distribution plot saved to '{save_path}'")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def get_flux_value(self):
        """
        Get the flux value from the flux monitor

        Returns:
            flux_value: total flux through the monitor
        """
        if self.flux is None:
            raise ValueError(
                "No flux monitor added. Call add_flux_monitor() first.")

        # Get flux value
        flux_value = mp.get_fluxes(self.flux)[0]

        return flux_value

    def run(self):
        """Run the simulation"""
        if self.sim is None:
            self.create_simulation()
        # Suppress Meep's verbose output during simulation run
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                self.sim.run(until=self.simulation_time)
        # self.plot_results(save_path=None, show_plot=False)
        return self.sim

    def get_field_data(self, component=mp.Ez):
        """Get electric field data from simulation"""
        if self.sim is None:
            raise ValueError(
                "Simulation must be run first. Call run() method.")

        # Get field data
        ez_data = self.sim.get_array(
            center=mp.Vector3(0, 0, 0),
            size=self.cell_size,
            component=component
        )

        # Transpose so indexing matches coordinate system: ez_data[x_idx, y_idx]
        self.ez_data = ez_data.T
        return self.ez_data

    def get_power_density_at_point(self, position, component='x'):
        """
        Get power density (Poynting vector) at a specific point.

        Args:
            position: mp.Vector3 or tuple (x, y) position where to measure
            component: 'x', 'y', or 'total' for power density direction

        Returns:
            power_density: Power density value at the point (real part)
        """
        if self.sim is None:
            raise ValueError(
                "Simulation must be run first. Call run() method.")

        # Convert tuple to Vector3 if needed
        if isinstance(position, (tuple, list)):
            position = mp.Vector3(position[0], position[1], 0)
        elif not isinstance(position, mp.Vector3):
            position = mp.Vector3(position.x, position.y, 0)

        # Method 1: Use Meep's built-in S-field (Poynting vector) components
        # Note: get_sfield_x() returns an array, so we need to get the array
        # and then interpolate to the point, OR use get_array with mp.Sx component

        # Method 2: Calculate from E and H fields (more direct for point values)
        # Get fields for 2D TM mode (Ez, Hx, Hy)
        Ez = self.sim.get_field_point(mp.Ez, position)
        Hx = self.sim.get_field_point(mp.Hx, position)
        Hy = self.sim.get_field_point(mp.Hy, position)

        # Calculate Poynting vector for 2D TM: Sx = -Ez * Hy, Sy = Ez * Hx
        # This is the standard way: S = (1/2) * Re(E × H*)
        # For real fields in 2D TM: Sx = -Ez * Hy, Sy = Ez * Hx
        Sx = -Ez * Hy  # Power flow in x-direction
        Sy = Ez * Hx   # Power flow in y-direction

        # Return real part (power density is typically real)
        if component == 'x':
            return np.real(Sx)
        elif component == 'y':
            return np.real(Sy)
        elif component == 'z':
            return 0.0  # No z-component in 2D
        elif component == 'total':
            return np.real(np.sqrt(Sx**2 + Sy**2))
        else:
            raise ValueError("component must be 'x', 'y', 'z', or 'total'")

    def get_field_at_point(self, position, field_component=mp.Ez):
        """
        Get field value at a specific point.

        Args:
            position: mp.Vector3 or tuple (x, y) position where to measure
            field_component: Field component (mp.Ez, mp.Hx, mp.Hy, etc.)

        Returns:
            field_value: Field value at the point
        """
        if self.sim is None:
            raise ValueError(
                "Simulation must be run first. Call run() method.")

        # Convert tuple to Vector3 if needed
        if isinstance(position, (tuple, list)):
            position = mp.Vector3(position[0], position[1], 0)
        elif not isinstance(position, mp.Vector3):
            position = mp.Vector3(position.x, position.y, 0)

        return self.sim.get_field_point(field_component, position)
    
    
    '''
    def plot_results(self, save_path=None, show_plot=True):
        """Plot and visualize the simulation results"""
        if self.ez_data is None:
            self.get_field_data()

        # Add timestamp to filename if not provided
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'meep_2d_result_{timestamp}.png'

        # Create figure
        plt.figure(figsize=(12, 6))

        # Set extent dynamically based on cell_size
        # cell spans from -cell_size.x/2 to +cell_size.x/2 in x, -cell_size.y/2 to +cell_size.y/2 in y
        extent = [-self.cell_size.x/2, self.cell_size.x/2,
                  -self.cell_size.y/2, self.cell_size.y/2]

        # Plot electric field
        plt.imshow(self.ez_data, interpolation='spline36', cmap='RdBu',
                   aspect='auto', extent=extent, origin='lower')
        plt.colorbar(label='Ez (electric field)')
        plt.xlabel('x (microns) → right')
        plt.ylabel('y (microns) → top')
        plt.title(f'Waveguide Simulation - 1550nm Continuous Wave from Left\n'
                  f'Waveguide width: {self.waveguide_width}um, Index: {self.waveguide_index}')

        # Mark waveguide outline
        waveguide_rect = Rectangle(
            (self.waveguide_x_min, -self.waveguide_width/2),
            self.waveguide_length,
            self.waveguide_width,
            linewidth=2.5,
            edgecolor='yellow',
            facecolor='none',
            linestyle='-',
            alpha=0.9,
            label='Waveguide outline'
        )
        plt.gca().add_patch(waveguide_rect)

        # Mark square region edges (2um x 2um, x from 0 to 2)
        plt.axvline(x=0, color='lime', linestyle='-', linewidth=2.5,
                    alpha=0.9, label='2um×2um square edges')
        plt.axvline(x=2, color='lime', linestyle='-', linewidth=2.5, alpha=0.9)
        plt.plot([0, 2], [-1, -1], color='lime',
                 linestyle='-', linewidth=2.5, alpha=0.9)
        plt.plot([0, 2], [1, 1], color='lime',
                 linestyle='-', linewidth=2.5, alpha=0.9)
        plt.legend(loc='upper right', fontsize=10)

        # Save and show
        if SAVE_FIG:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Simulation complete! Results saved to '{save_path}'")
        print(
            f"Waveguide: {self.waveguide_width}um wide, {self.waveguide_index} index")
        print(
            f"EigenModeSource: Continuous wave at {self.wavelength}um (1550nm), from left")

        if show_plot:
            plt.show()
        else:
            plt.close()
    '''
    def plot_results(self, save_path=None, show_plot=True):
        """
        Plot and visualize the simulation results (Ez field + geometry overlays).
        This version includes input/output waveguides and the output measurement plane.
        """
        if self.ez_data is None:
            self.get_field_data()

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'meep_2d_result_{timestamp}.png'
        
        plt.figure(figsize=(12, 6))
        ax = plt.gca() # Get current axes for adding patches

        # Set extent dynamically based on cell_size
        extent = [-self.cell_size.x/2, self.cell_size.x/2,
                -self.cell_size.y/2, self.cell_size.y/2]

        # Plot electric field
        # Transpose ez_data because imshow expects (rows, cols) where rows=y, cols=x
        # Meep's get_array gives (x, y), so transpose for correct orientation
        plt.imshow(self.ez_data, interpolation='spline36', cmap='RdBu',
                aspect='auto', extent=extent, origin='lower')
        plt.colorbar(label='Ez (electric field)')
        plt.xlabel('x (microns)')
        plt.ylabel('y (microns)')
        plt.title(f'Waveguide Simulation - Ez Field (L_in={self.input_coupler_length}µm, L_out={self.output_coupler_length}µm)')

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
        output_y_separation = 0.3 # Use the same fixed separation as in create_geometry
        output_waveguide_x_start = self.design_region_x_max
        output_waveguide_length = self.output_coupler_length

        # Output Waveguide 1 (Top)
        output1_waveguide_y_start = output_y_separation - self.waveguide_width / 2
        ax.add_patch(Rectangle(
            (output_waveguide_x_start, output1_waveguide_y_start),
            output_waveguide_length, self.waveguide_width,
            linewidth=2.5, edgecolor='orange', facecolor='none', linestyle='-', alpha=0.9,
            label='Output Waveguide Outline' # Only label once
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
            (self.design_region_x_min, self.design_region_y_min), # Lower-left corner: (-1.0, -1.0)
            2.0, 2.0, # Width=2.0, Height=2.0
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
            unique_handles[l] = h # Dict automatically handles duplicates, keeping last one
        
        # FIX: Move legend outside the plot area
        ax.legend(unique_handles.values(), unique_handles.keys(), 
              loc='lower right',       # Tells the legend to position its lower-right corner at the anchor point
              bbox_to_anchor=(0.3, -0.4),   # Sets the anchor point to the bottom-left corner of the axes (x=0, y=0)
              borderaxespad=0.)        # No padding between the legend and the anchor point
        
        # FIX: Ensure 1:1 scale for the axes
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout() # IMPORTANT: Adjust plot to make room for the legend

        # Save and show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Simulation results saved to '{save_path}'")
        print(f"Waveguide: {self.waveguide_width}um wide, {self.waveguide_index} index")
        print(f"EigenModeSource: Continuous wave at {self.wavelength}um, from left")

        if show_plot:
            plt.show()
        else:
            plt.close()
    def run_full_simulation(self, until=30, save_path=None, show_plot=False,
                            measure_flux_at_x=None, flux_along_y=False, num_flux_regions=50):
        """
        Run complete simulation workflow

        Args:
            until: simulation time
            save_path: path to save plot
            show_plot: whether to show plot
            measure_flux_at_x: if provided, measure flux at this x position
            flux_along_y: if True, measure flux distribution along y-axis
            num_flux_regions: number of flux regions for y-axis distribution
        """
        # Add flux monitors if requested
        if measure_flux_at_x is not None:
            if self.sim is None:
                self.create_simulation()

            if flux_along_y:
                # Add multiple flux monitors along y-axis
                self.add_flux_monitors_along_y(
                    measure_flux_at_x, num_regions=num_flux_regions)
            else:
                # Add single flux monitor
                self.add_flux_monitor(measure_flux_at_x)

        self.run(until=until)
        self.get_field_data()
        self.plot_results(save_path=save_path, show_plot=show_plot)

        # Calculate and print flux
        if self.flux_regions:
            # Multiple flux regions along y-axis
            y_positions, flux_values = self.get_flux_distribution_along_y()
            total_flux = np.sum(flux_values) * (y_positions[1] - y_positions[0]) if len(
                y_positions) > 1 else np.sum(flux_values)
            print(f"\nFlux distribution at x = {measure_flux_at_x}:")
            print(f"  Total flux: {total_flux:.6e}")
            print(f"  Max flux: {np.max(flux_values):.6e}")
            print(f"  Min flux: {np.min(flux_values):.6e}")

            # Plot flux distribution
            self.plot_flux_distribution_y(
                measure_flux_at_x, save_path=None, show_plot=False)

        elif self.flux is not None:
            # Single flux monitor
            flux_value = self.get_flux_value()
            print(f"\nFlux at x = {measure_flux_at_x}: {flux_value:.6e}")
    
    def calculate_flux(self, material_matrix):
        # Create simulation

        # Create geometry with material matrix
        self.create_geometry(material_matrix=material_matrix)

        # Create simulation and add flux monitors
        self.create_simulation()
        self.add_flux_monitors_along_y()

        # Run simulation
        self.run()

        # Get flux distribution
        _, flux_values = self.get_flux_distribution_along_y()

        # Get field data
        ez_data = self.get_field_data()

        return flux_values, ez_data
'''
if __name__ == "__main__":
    # Example usage
    material_matrix = np.zeros((50, 50))
    material_matrix[0:5, :] = 1  # Add silicon at x=0 to 0.2um

    calculator = WaveguideSimulation()
    flux_array = calculator.calculate_flux(material_matrix)

    print(f"Flux array shape: {flux_array.shape}")
    print(f"Total flux: {np.sum(flux_array):.6e}")
    print(f"Max flux: {np.max(flux_array):.6e}")
    print(f"Min flux: {np.min(flux_array):.6e}")
'''
if __name__ == "__main__":
    
    # Example 1: Standard centered setup
    calculator_A = WaveguideSimulation(
        cell_size=mp.Vector3(8, 4, 0), # Increase cell size to fit longer waveguides
        input_coupler_length=1.5,      # Input waveguide length is 2.0 µm
        output_coupler_length=1.5,     # Output waveguide length is 2.0 µm
        waveguide_width=0.4
    )
    
    # Create a simple test matrix
    material_matrix = np.ones((50, 50)) 

    # Apply the geometry
    calculator_A.create_geometry(material_matrix=material_matrix) 

    # Plot to verify the new centering and lengths
    print("Plotting Centered Geometry (Design at x=[-1, 1])")
    calculator_A.plot_geometry(
        show_plot=False, 
        save_path='img/geometry_plot.png', # Provide a file name here
        x_range=(-3.0, 3.0), 
        y_range=(-2, 2)
    )

    # Example 2: Run a quick simulation test with the new geometry
    calculator_A.create_simulation()
    
    # Since the outputs are separated, define flux monitor location at x=2.5
    calculator_A.output_x = 2.5 
    calculator_A.add_flux_monitors_along_y() # Add monitors to measure flux split
    
    print("\nRunning simulation with centered geometry...")
    calculator_A.run()
    calculator_A.plot_results(
        show_plot=False,
        save_path='img/simulation_ez_field.png' # Provide a file name here
    )
    
    # Get total flux
    _, flux_values = calculator_A.get_flux_distribution_along_y()
    print(f"Total flux measured: {np.sum(flux_values):.4e}")