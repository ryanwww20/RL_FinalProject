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
from contextlib import redirect_stdout, redirect_stderr

SAVE_FIG = False


class WaveguideSimulation:
    """2D Meep simulation class for waveguide with eigenmode source"""

    def __init__(self,
                 resolution=50,
                 wavelength=1.55,
                 cell_size=mp.Vector3(4, 2, 0),
                 pml_thickness=0.2,
                 waveguide_width=0.3,
                 waveguide_index=3.5,
                 waveguide_center_x=-0.9,
                 waveguide_length=1.8):
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
        self.resolution = resolution
        self.wavelength = wavelength
        self.cell_size = cell_size
        self.pml_layers = [mp.PML(pml_thickness)]
        self.waveguide_width = waveguide_width
        self.waveguide_index = waveguide_index
        self.waveguide_center_x = waveguide_center_x
        self.waveguide_length = waveguide_length

        # Calculate waveguide boundaries
        self.waveguide_x_min = waveguide_center_x - waveguide_length / 2
        self.waveguide_x_max = waveguide_center_x + waveguide_length / 2

        # Initialize simulation components
        self.geometry = None
        self.sources = None
        self.sim = None
        self.ez_data = None
        self.flux = None  # Single flux monitor object
        self.flux_regions = []  # List of flux monitors for y-axis distribution

    def create_geometry(self, material_matrix=None, silicon_index=3.5, silica_index=1.45):
        """
        Create waveguide geometry and add material distribution based on matrix

        Args:
            material_matrix: 50x50 numpy array where 0=silica, 1=silicon
                           If None, no additional materials are added
            silicon_index: refractive index of silicon (default: 3.5)
            silica_index: refractive index of silica/SiO2 (default: 1.45)
        """
        geometry = []

        # Create waveguide
        waveguide = mp.Block(
            center=mp.Vector3(self.waveguide_center_x, 0, 0),
            size=mp.Vector3(self.waveguide_length, self.waveguide_width, 0),
            material=mp.Medium(index=self.waveguide_index)
        )
        geometry.append(waveguide)

        # Add material distribution from matrix
        # Matrix applies only to square region: x from 0 to 2um, y from -1 to +1um
        if material_matrix is not None:
            # Convert to numpy array if needed
            material_matrix = np.array(material_matrix)

            # Check matrix dimensions
            if material_matrix.shape != (50, 50):
                raise ValueError(
                    f"material_matrix must be 50x50, got shape {material_matrix.shape}")

            # Square region boundaries
            square_x_min = 0.0  # Square starts at x = 0
            square_x_max = 2.0   # Square ends at x = 2um
            square_y_min = -1.0  # Square y from -1um
            square_y_max = 1.0  # Square y to +1um (symmetric)

            # Pixel size in square region
            dx = (square_x_max - square_x_min) / \
                50  # 2um / 50 = 0.04um per pixel
            dy = (square_y_max - square_y_min) / \
                50   # 2um / 50 = 0.04um per pixel

            # Create blocks for each pixel based on matrix value
            for i in range(50):
                for j in range(50):
                    # i corresponds to x (0 to 49 maps to x from 0 to 2um)
                    # j corresponds to y (0 to 49 maps to y from -1 to +1um, symmetric)
                    # Map matrix indices to physical coordinates in square region
                    x_center = square_x_min + (i + 0.5) * dx
                    y_center = square_y_min + (j + 0.5) * dy

                    if material_matrix[i, j] == 1:
                        # Silicon pixel
                        silicon_pixel = mp.Block(
                            center=mp.Vector3(x_center, y_center, 0),
                            size=mp.Vector3(dx, dy, 0),
                            material=mp.Medium(index=silicon_index)
                        )
                        geometry.append(silicon_pixel)
                    elif material_matrix[i, j] == 0:
                        # Silica pixel
                        silica_pixel = mp.Block(
                            center=mp.Vector3(x_center, y_center, 0),
                            size=mp.Vector3(dx, dy, 0),
                            material=mp.Medium(index=silica_index)
                        )
                        geometry.append(silica_pixel)

        self.geometry = geometry
        return geometry

    def create_sources(self):
        """Create eigenmode source at left edge of waveguide"""
        sources = [mp.EigenModeSource(
            src=mp.ContinuousSource(
                wavelength=self.wavelength,
                width=20
            ),
            center=mp.Vector3(self.waveguide_x_min, 0, 0),  # Left edge
            size=mp.Vector3(0, self.waveguide_width, 0),  # Vertical line
            eig_band=1,  # Fundamental mode
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

    def add_flux_monitor(self, x_position, height=None):
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
            center=mp.Vector3(x_position, 0, 0),
            size=mp.Vector3(0, height, 0)  # Vertical line at x_position
        )

        # Add flux monitor to simulation
        self.flux = self.sim.add_flux(frequency, 0, 1, flux_region)

        return self.flux

    def add_flux_monitors_along_y(self, x_position, num_regions=50, region_height=None):
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
            region_height = self.cell_size.y / num_regions

        # Calculate y positions for each region
        # y spans from -cell_size.y/2 to +cell_size.y/2
        y_min = -self.cell_size.y / 2
        y_max = self.cell_size.y / 2
        y_positions = np.linspace(
            y_min + region_height/2, y_max - region_height/2, num_regions)

        # Create flux monitors for each y position
        flux_monitors = []
        for y_pos in y_positions:
            flux_region = mp.FluxRegion(
                center=mp.Vector3(x_position, y_pos, 0),
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

    def plot_flux_distribution_y(self, x_position, save_path=None, show_plot=True):
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
            save_path = f'flux_distribution_x{x_position}_{timestamp}.png'

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(y_positions, flux_values, 'b-', linewidth=2,
                 label=f'Flux at x = {x_position}')
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        plt.xlabel('y (microns)')
        plt.ylabel('Flux')
        plt.title(f'Flux Distribution along Y-axis at x = {x_position}μm')
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

    def run(self, until=30):
        """Run the simulation"""
        if self.sim is None:
            self.create_simulation()
        # Suppress Meep's verbose output during simulation run
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                self.sim.run(until=until)
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


class FluxCalculator:
    """
    Simple interface for calculating flux from material matrix.

    Usage:
        from meep_simple_2d import FluxCalculator
        import numpy as np

        # Create 50x50 material matrix (0=silica, 1=silicon)
        material_matrix = np.zeros((50, 50))
        material_matrix[0:5, :] = 1  # Add some silicon

        # Calculate flux
        calculator = FluxCalculator()
        flux_array = calculator.calculate_flux(material_matrix, x_position=2.0)
    """

    def __init__(self,
                 resolution=50,
                 wavelength=1.55,
                 cell_size=None,
                 waveguide_width=0.3,
                 waveguide_index=3.5,
                 waveguide_center_x=-0.9,
                 waveguide_length=1.8,
                 silicon_index=3.5,
                 silica_index=1.45,
                 simulation_time=30,
                 num_flux_regions=100):
        """
        Initialize flux calculator with simulation parameters.

        Args:
            resolution: pixels per micron (default: 50)
            wavelength: wavelength in microns, 1550nm = 1.55 (default: 1.55)
            cell_size: simulation cell size (default: Vector3(6, 2, 0))
            waveguide_width: waveguide width in microns (default: 0.3)
            waveguide_index: refractive index of waveguide (default: 3.5)
            waveguide_center_x: x-coordinate of waveguide center (default: -0.9)
            waveguide_length: length of waveguide in x-direction (default: 1.8)
            silicon_index: refractive index of silicon (default: 3.5)
            silica_index: refractive index of silica/SiO2 (default: 1.45)
            simulation_time: simulation time (default: 30)
            num_flux_regions: number of flux regions along y-axis (default: 100)
        """
        if cell_size is None:
            cell_size = mp.Vector3(6, 2, 0)

        self.resolution = resolution
        self.wavelength = wavelength
        self.cell_size = cell_size
        self.waveguide_width = waveguide_width
        self.waveguide_index = waveguide_index
        self.waveguide_center_x = waveguide_center_x
        self.waveguide_length = waveguide_length
        self.silicon_index = silicon_index
        self.silica_index = silica_index
        self.simulation_time = simulation_time
        self.num_flux_regions = num_flux_regions

    def calculate_flux(self, material_matrix, x_position=2.0):
        """
        Calculate flux distribution along y-axis at specified x position.

        Args:
            material_matrix: 50x50 numpy array where 0=silica, 1=silicon
                           Matrix applies to square region: x from 0 to 2um, y from -1 to +1um
            x_position: x coordinate where to measure flux (default: 2.0)

        Returns:
            flux_array: numpy array of flux values along y-axis
                       Shape: (num_flux_regions,)
        """
        # Create simulation
        sim = WaveguideSimulation(
            resolution=self.resolution,
            wavelength=self.wavelength,
            cell_size=self.cell_size,
            waveguide_width=self.waveguide_width,
            waveguide_index=self.waveguide_index,
            waveguide_center_x=self.waveguide_center_x,
            waveguide_length=self.waveguide_length
        )

        # Create geometry with material matrix
        sim.create_geometry(
            material_matrix=material_matrix,
            silicon_index=self.silicon_index,
            silica_index=self.silica_index
        )

        # Create simulation and add flux monitors
        sim.create_simulation()
        sim.add_flux_monitors_along_y(
            x_position, num_regions=self.num_flux_regions)

        # Run simulation
        sim.run(until=self.simulation_time)

        # Get flux distribution
        y_positions, flux_values = sim.get_flux_distribution_along_y()

        # Get field data
        ez_data = sim.get_field_data()

        return flux_values, ez_data


if __name__ == "__main__":
    # Example usage
    material_matrix = np.zeros((50, 50))
    material_matrix[0:5, :] = 1  # Add silicon at x=0 to 0.2um

    calculator = FluxCalculator()
    flux_array = calculator.calculate_flux(material_matrix, x_position=2.0)

    print(f"Flux array shape: {flux_array.shape}")
    print(f"Total flux: {np.sum(flux_array):.6e}")
    print(f"Max flux: {np.max(flux_array):.6e}")
    print(f"Min flux: {np.min(flux_array):.6e}")
