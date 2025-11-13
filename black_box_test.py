import meep as mp

import numpy as np
import matplotlib.pyplot as plt

# Cell size
sx = 10  # propagation direction (x)
sy = 6   # height (y)
cell = mp.Vector3(sx, sy, 0)

resolution = 20  # pixels/μm (depends on how fine you want)

# Boundary conditions
pml_layers = [mp.PML(1.0)]

# Define pattern (1 = Silicon, 0 = Air)
pattern = np.array([
    [1,0,0,0,1],
    [0,0,1,1,0],
    [1,0,1,0,1],
    [0,1,1,0,0],
    [1,0,0,0,1],
])

ny, nx = pattern.shape
block_size_x = sx / nx
block_size_y = sy / ny

geometry = []
for i in range(ny):
    for j in range(nx):
        if pattern[i, j] == 1:
            center_x = -sx/2 + (j+0.5)*block_size_x
            center_y = -sy/2 + (ny-i-0.5)*block_size_y
            geometry.append(
                mp.Block(
                    material=mp.Medium(index=3.45),  # Silicon
                    center=mp.Vector3(center_x, center_y),
                    size=mp.Vector3(block_size_x, block_size_y)
                )
            )

# Source wavelength and frequency
wavelength = 1.55  # μm
frequency = 1 / wavelength

sources = [mp.Source(
    src=mp.GaussianSource(frequency, fwidth=0.2*frequency),
    component=mp.Ez,
    center=mp.Vector3(-sx/2 + 1, 0),  # near left boundary
    size=mp.Vector3(0, sy)
)]


sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    boundary_layers=pml_layers,
                    sources=sources,
                    resolution=resolution)

# Define multiple flux monitors along the output plane to measure power distribution
num_detectors = 10  # number of detectors along y-axis
detector_height = sy / num_detectors
output_x = sx/2 - 1  # x position of output plane

flux_monitors = []
for i in range(num_detectors):
    y_pos = -sy/2 + (i + 0.5) * detector_height
    flux_region = mp.FluxRegion(
        center=mp.Vector3(output_x, y_pos),
        size=mp.Vector3(0, detector_height)
    )
    flux_monitors.append(sim.add_flux(frequency, 0, 1, flux_region))

sim.run(until=200)  # time steps

# Get field data (for visualization)
eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

# Visualize the simulation
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary',
           extent=[-sx/2, sx/2, -sy/2, sy/2], origin='lower')
plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.8,
           extent=[-sx/2, sx/2, -sy/2, sy/2], origin='lower')
plt.xlabel('X Position (μm)', fontsize=12)
plt.ylabel('Y Position (μm)', fontsize=12)
plt.title('Electric Field Distribution', fontsize=12)
plt.colorbar(label='Ez Field (au)', fraction=0.046, pad=0.04)
plt.axhline(y=0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
plt.axvline(x=0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
# Mark the output plane
plt.axvline(x=output_x, color='yellow', linestyle='-', linewidth=2, label='Output Plane')
plt.legend(loc='upper left', fontsize=10)

# Extract power distribution along y-axis at output plane
power_distribution = []
y_positions = []
for i, monitor in enumerate(flux_monitors):
    power = mp.get_fluxes(monitor)[0]
    power_distribution.append(power)
    y_pos = -sy/2 + (i + 0.5) * detector_height
    y_positions.append(y_pos)

# Plot power distribution
plt.subplot(1, 2, 2)
# plt.plot(power_distribution, y_positions, 'o-', linewidth=2, markersize=8)
# plt.xlabel('Power (au)', fontsize=12)
# plt.ylabel('Y Position (μm)', fontsize=12)
plt.plot(y_positions, power_distribution, 'o-', linewidth=2, markersize=8)
plt.xlabel('Y Position (μm)', fontsize=12)
plt.ylabel('Power (au)', fontsize=12)
plt.title('Power Distribution at Output Plane', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print results
print("\n=== Power Distribution at Output Plane ===")
print(f"Total transmitted power: {np.sum(power_distribution):.6f}")
print("\nPower at each detector position:")
for i, (y, p) in enumerate(zip(y_positions, power_distribution)):
    print(f"  Detector {i+1} (y={y:+.2f}): {p:.6f}")