"""
Meep FDTD Simulation Test Script

This script demonstrates a basic 2D electromagnetic simulation using Meep (MIT
Electromagnetic Equation Propagation). It simulates a dielectric block in a 
simulation cell with a continuous wave source and visualizes both the dielectric
structure and the resulting electric field distribution.
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# Simulation cell dimensions (x, y, z) - z=0 for 2D simulation
cell = mp.Vector3(16, 8, 0)
'''
the cell looks like this:
            |---|
            |   |
            |   |
            |---|
            |   |
            |   |
            |---|
            |   |
            |   |
A           |---|
|
x
   y ->            

'''


# Define geometry: a dielectric block with epsilon=12
# The block extends infinitely in x and z directions, 1 unit thick in y
geometry = [mp.Block(mp.Vector3(mp.inf, 1, mp.inf),
                     center=mp.Vector3(),
                     material=mp.Medium(epsilon=12))]

# Define source: continuous wave source at frequency 0.15
# Ez component (out-of-plane electric field for 2D simulation)
# Positioned at x=-7, y=0 (left side of the cell)
sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                     component=mp.Ez,
                     center=mp.Vector3(-7, 0))]

# Perfectly Matched Layer (PML) for absorbing boundaries
# 1.0 unit thick PML layers on all boundaries

'''
PML (Perfectly Matched Layer) is a boundary treatment, not a physical object. It’s an absorbing layer that prevents reflections at the edges of the simulation cell.
What PML does:
Absorbs outgoing waves — when waves hit the boundary, they’re absorbed instead of reflecting back
Simulates open boundaries — mimics an infinite space, not a box with walls
Prevents unwanted reflections — without PML, waves would bounce off the edges and interfere with the simulation

┌─────────────────────────────────┐
│  PML (1.0 unit thick)          │ ← Absorbing layer
│  ┌───────────────────────────┐  │
│  │                           │  │
│  │   Simulation Cell         │  │
│  │   (16 x 8)                │  │
│  │                           │  │
│  │   [Geometry objects]      │  │ ← Your actual structures
│  │   [Sources]               │  │
│  │                           │  │
│  └───────────────────────────┘  │
│  PML (1.0 unit thick)          │ ← Absorbing layer
└─────────────────────────────────┘
'''

pml_layers = [mp.PML(1.0)]

# Resolution: number of pixels per unit distance
resolution = 10

# Create and configure the simulation
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

# Run simulation until time=200
sim.run(until=200)

# Extract and visualize the dielectric structure
eps_data = sim.get_array(
    center=mp.Vector3(), size=cell, component=mp.Dielectric)
plt.figure()
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.axis('off')
plt.show()

# Extract and visualize the electric field (Ez component)
# Overlay the field on top of the dielectric structure
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
plt.figure()
# First plot the dielectric structure as background
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
# Then overlay the electric field with transparency
plt.imshow(ez_data.transpose(), interpolation='spline36',
           cmap='RdBu', alpha=0.9)
plt.axis('off')
plt.show()
