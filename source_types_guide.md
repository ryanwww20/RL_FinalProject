# Meep Source Types Guide

## Overview of Source Types

Meep offers several source types for different simulation needs:

### 1. **GaussianSource** (Most Common)

Gaussian temporal profile - creates a realistic pulsed source with a frequency spectrum.

```python
mp.Source(
    mp.GaussianSource(
        wavelength=0.5,    # Wavelength in microns
        width=20,          # Temporal width (larger = narrower bandwidth)
        fcen=1/0.5,        # Center frequency (1/wavelength)
        cutoff=3           # Cutoff (3 = 3 standard deviations)
    ),
    component=mp.Ez,
    center=mp.Vector3(0, 0, 0),
    size=mp.Vector3(0.5, 0.5, 0)  # Can be point or extended
)
```

**Use when:**

- You want a realistic laser pulse
- You need a finite bandwidth
- Most general-purpose simulations

---

### 2. **ContinuousSource**

Continuous wave (monochromatic) - infinite duration sine wave.

```python
mp.Source(
    mp.ContinuousSource(
        wavelength=0.5,
        width=20
    ),
    component=mp.Ez,
    center=mp.Vector3(0, 0, 0),
    size=mp.Vector3(0, 0, 0)  # Point source
)
```

**Use when:**

- Steady-state simulations
- Simple plane wave excitation
- You need a single frequency

---

### 3. **EigenModeSource** (Waveguide Source)

Excites a specific eigenmode of a waveguide structure.

```python
# First create waveguide geometry
waveguide = mp.Block(
    center=mp.Vector3(0, 0, 0),
    size=mp.Vector3(2, 0.5, 0),
    material=mp.Medium(index=3.5)
)

# Then use EigenModeSource
mp.EigenModeSource(
    src=mp.GaussianSource(wavelength=0.5, width=20),
    center=mp.Vector3(-0.8, 0, 0),  # At waveguide input
    size=mp.Vector3(0, 0.5, 0),     # Matches waveguide cross-section
    eig_band=1,                      # Mode number (1=fundamental, 2=first higher order, etc.)
    direction=mp.X                   # Propagation direction
)
```

**Use when:**

- Coupling light into waveguides
- Exciting specific waveguide modes
- Photonic integrated circuits
- Mode analysis

**Key parameters:**

- `eig_band`: Mode number (1 = fundamental, 2 = first higher order mode, etc.)
- `direction`: Propagation direction (mp.X, mp.Y, mp.Z, or mp.NO_DIRECTION)
- `size`: Must match the waveguide cross-section

---

### 4. **Plane Wave Source**

Extended source that creates a plane wave.

```python
mp.Source(
    mp.GaussianSource(wavelength=0.5, width=20),
    component=mp.Ez,
    center=mp.Vector3(-0.8, 0, 0),  # At edge of simulation
    size=mp.Vector3(0, 2, 0)        # Full height (or width) for plane wave
)
```

**Use when:**

- Illuminating large areas uniformly
- Scattering problems
- Periodic structures

---

## Source Components

For 2D simulations, you typically use:

- `mp.Ez` - Electric field in z-direction (TM mode)
- `mp.Hz` - Magnetic field in z-direction (TE mode)

For 3D simulations, you can use:

- `mp.Ex`, `mp.Ey`, `mp.Ez` - Electric field components
- `mp.Hx`, `mp.Hy`, `mp.Hz` - Magnetic field components

---

## Source Size

- **Point source**: `size=mp.Vector3(0, 0, 0)` - Single point
- **Extended source**: `size=mp.Vector3(w, h, 0)` - Extended area
- **Line source**: `size=mp.Vector3(0, h, 0)` - Line in one dimension

---

## Tips

1. **GaussianSource** is usually the best default choice
2. **EigenModeSource** is essential for waveguide simulations
3. Place sources away from PML boundaries (at least 0.5-1 wavelength)
4. For waveguide sources, the `size` must match the waveguide cross-section
5. Use `eig_band=1` for fundamental mode, higher numbers for higher-order modes
