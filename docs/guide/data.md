# Data Loading

WaveDiff-JAX loads astronomical PSF data from `.npz` files and converts them to JAX arrays for training.

## DataHandler

The primary interface for loading and preprocessing training/test data:

```python
from wavediff_jax.data.preprocessing import DataHandler

handler = DataHandler(
    dataset_type="training",    # or "test"
    data_params=data_config,    # RecursiveNamespace with paths
    simPSF=psf_simulator,       # PSFSimulator instance
    n_bins_lambda=20,           # wavelength bins for SED discretisation
)
```

`DataHandler` loads:

- **Star images:** PSF observations, shape `(n_stars, dim, dim)`
- **Positions:** Field coordinates, shape `(n_stars, 2)`
- **SEDs:** Spectral energy distributions
- **Masks:** Pixel masks (0 = include, 1 = exclude)
- **Zernike priors:** Optional prior coefficients from ray-tracing

## SED Processing

Spectral Energy Distributions are discretised into wavelength bins:

```python
from wavediff_jax.data.preprocessing import (
    generate_SED_elems,
    generate_SED_elems_jax,
    generate_packed_elems,
)

# Numpy output
feasible_N, SED_data = generate_SED_elems(SED, psf_simulator, n_bins=20)

# JAX float64 output
jax_elems = generate_SED_elems_jax(SED, psf_simulator, n_bins=20)

# Packed format: (n_stars, n_wavelengths, 3)
# Each entry: [phase_N, lambda_obs, sed_weight]
packed = generate_packed_elems(SED, psf_simulator, n_bins=20)
```

## Position Extraction

```python
from wavediff_jax.data.preprocessing import get_obs_positions, get_np_obs_positions

# JAX array
positions = get_obs_positions(dataset)  # jnp.ndarray, shape (n_stars, 2)

# NumPy array
np_positions = get_np_obs_positions(dataset)  # np.ndarray
```

## Zernike Priors

For the physical polychromatic model:

```python
from wavediff_jax.data.preprocessing import get_zernike_prior, get_np_zernike_prior

# JAX array
zk_prior = get_zernike_prior(dataset)  # jnp.ndarray

# NumPy array
np_zk_prior = get_np_zernike_prior(dataset)  # np.ndarray
```

## Data Format

WaveDiff-JAX expects `.npz` files with the following keys:

| Key | Shape | Description |
|---|---|---|
| `stars` | `(n_stars, dim, dim)` | Observed PSF images |
| `positions` | `(n_stars, 2)` | Field positions (x, y) |
| `SEDs` | `(n_stars, n_wavelengths)` | Spectral energy distributions |
| `masks` | `(n_stars, dim, dim)` | Pixel masks (0=include, 1=exclude) |
| `zernike_prior` | `(n_stars, n_zernikes)` | Optional Zernike coefficient priors |

## Mask Convention

!!! warning "Mask convention: 0 = include, 1 = exclude"
    This follows the original TensorFlow implementation's convention. Pixels with mask value 0 are included in loss computation; pixels with mask value 1 are excluded.
