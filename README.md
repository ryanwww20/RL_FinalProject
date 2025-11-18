# Meep 2D Light Simulation

This project contains a simple 2D Meep simulation in a 2um × 2um space.

## About 2D vs 3D in Meep

**For 2D simulations, the third dimension (z) is NOT necessary!**

- In 2D mode, Meep automatically treats the z-dimension as infinite/extended
- This means the simulation is effectively a slice through an infinite structure
- Set `dimensions=2` and use `cell_size = mp.Vector3(x, y, 0)` where z=0 indicates 2D
- This is much faster and uses less memory than 3D simulations

## Requirements

```bash
pip install meep matplotlib numpy
```

## Running the Simulation

```bash
python meep_simple_2d.py
```

## What the Simulation Does

1. Creates a 2um × 2um 2D simulation space
2. Adds a continuous light source at the center (0.5um wavelength)
3. Runs the simulation and visualizes the electric field distribution
4. Saves the result as `meep_2d_result.png`

## Customization

You can modify:
- `cell_size`: Change the simulation domain size
- `resolution`: Higher resolution = more accurate but slower
- `sources`: Change the light source properties (wavelength, position, type)
- `geometry`: Add objects (dielectrics, metals, etc.) to the simulation


