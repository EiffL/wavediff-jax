# Simulation

WaveDiff-JAX includes a ground-truth PSF simulator for generating synthetic training data. The simulator uses numpy-based FFT (not JAX) and is not differentiable — it serves as an independent reference implementation.

## PSFSimulator

The main simulation class:

```python
from wavediff_jax.sims.psf_simulator import PSFSimulator

sim = PSFSimulator(
    max_order=45,          # number of Zernike modes
    max_wfe_rms=0.1,       # maximum wavefront error RMS
    output_dim=64,         # PSF image size
    rand_seed=42,
    pupil_diam=256,        # pupil grid size
    output_Q=1,            # oversampling factor
    euclid_obsc=True,      # use Euclid pupil geometry
)
```

### Generating PSFs

```python
# Random Zernike coefficients
sim.gen_random_Z_coeffs(max_order=45, rand_seed=42)

# Monochromatic PSF at a single wavelength
sim.gen_mono_psf(lambda_obs=0.8)
psf = sim.get_psf()  # (64, 64)

# Polychromatic PSF with SED weighting
sim.gen_poly_psf(sed=my_sed)
poly_psf = sim.get_psf()  # (64, 64)
```

### OPD and Phase

```python
# Generate OPD from Zernike coefficients
sim.gen_opd()

# Generate complex phase
sim.gen_phase()
```

### SED Processing

```python
# Discretise SED into wavelength bins
wave_values = sim.calc_SED_wave_values(SED, n_bins=20)

# Get feasible FFT size for a wavelength
N = sim.feasible_N(wavelength=0.7)  # int
```

### Wavefront Error

```python
# Compute WFE RMS
wfe_rms = sim.get_wfe_rms()

# With normalisation
wfe_rms_norm = sim.get_wfe_rms(normalise=True)
```

## Spatial Variation

The `spatial_varying_psf` module provides tools for simulating field-varying PSFs:

```python
from wavediff_jax.sims.spatial_varying_psf import SpatialVaryingPSF, MeshHelper

# Create a grid of positions
xv, yv = MeshHelper.build_mesh(
    x_lims=[0, 1000],
    y_lims=[0, 1000],
    grid_points=[10, 10],
)
```

### Helper Classes

| Class | Purpose |
|---|---|
| `MeshHelper` | Build 2D position grids |
| `CoordinateHelper` | Scale positions, add random shifts, check limits |
| `PolynomialMatrixHelper` | Generate polynomial basis matrices |
| `ZernikeHelper` | Zernike coefficient field generation |
| `SpatialVaryingPSF` | Full spatially-varying PSF field simulation |

## Euclid Pupil

The simulator uses the Euclid telescope pupil geometry:

```python
from wavediff_jax.optics.obscurations import generate_euclid_pupil_obscurations

pupil = generate_euclid_pupil_obscurations(
    N_pix=256,
    N_filter=2,           # low-pass filter iterations
    rotation_angle=0.0,   # pupil rotation in degrees
)
```

The pupil includes:
- 1200mm diameter primary mirror
- 395mm central obscuration (M2)
- Three spider arms at specific angles
- Optional low-pass filtering for smoother edges
