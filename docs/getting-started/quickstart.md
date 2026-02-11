# Quick Start

This guide walks through creating a PSF model, running a forward pass, and performing a training step.

## Creating a Model

Every model needs two optical primitives: a Zernike polynomial basis and a telescope pupil obscuration mask.

```python
import jax
import jax.numpy as jnp
from wavediff_jax.utils.math_utils import generate_zernike_maps_3d, obscurations_from_params

# Generate a Zernike basis (15 polynomials on a 256x256 grid)
zernike_maps = generate_zernike_maps_3d(n_zernikes=15, pupil_diam=256)
# shape: (15, 256, 256)

# Generate Euclid pupil obscurations
obscurations = obscurations_from_params(pupil_diam=256)
# shape: (256, 256), dtype: complex64
```

Now create a parametric PSF model:

```python
from wavediff_jax.models.parametric import ParametricPSFFieldModel

model = ParametricPSFFieldModel(
    zernike_maps=zernike_maps,
    obscurations=obscurations,
    output_Q=1,           # oversampling factor
    output_dim=64,        # PSF image size
    n_zernikes=15,        # number of Zernike modes
    d_max=2,              # polynomial degree for field variation
    x_lims=[0, 1e3],     # field-of-view limits
    y_lims=[0, 1e3],
    key=jax.random.PRNGKey(42),
)
```

## Forward Pass

The model takes a list `[positions, packed_seds]` and returns PSF images and OPD maps:

```python
# Star positions in field coordinates
positions = jnp.array([[500.0, 500.0], [200.0, 800.0]])

# Packed SED data: (n_stars, n_wavelengths, 3)
# Each wavelength entry is [phase_N, lambda_obs, sed_weight]
packed_seds = jnp.zeros((2, 10, 3))

# Forward pass
psf_batch, opd_maps = model([positions, packed_seds])
# psf_batch: (2, 64, 64) — polychromatic PSF images
# opd_maps:  (2, 256, 256) — optical path difference maps
```

## Predicting OPD and Monochromatic PSFs

```python
# OPD maps at given positions
opd = model.predict_opd(positions)  # (2, 256, 256)

# Monochromatic PSFs at a single wavelength
mono_psfs = model.predict_mono_psfs(
    positions, lambda_obs=0.8, phase_N=914
)  # (2, 64, 64)
```

## Training Step

WaveDiff-JAX uses a functional training loop with Equinox and Optax:

```python
import optax
import equinox as eqx
from wavediff_jax.training.losses import total_loss
from wavediff_jax.training.trainer import make_step, param_filter

# Create optimizer
optimizer = optax.adam(1e-3)

# Create loss function
def loss_fn(model, batch):
    positions, packed_seds, targets, masks = batch
    return total_loss(
        model, positions, packed_seds, targets, masks,
        sample_weight=None, l2_param=0.0, l1_rate=0.0,
    )

# Get filter spec (which parameters to train)
filter_spec = param_filter(model)

# Initialize optimizer state
opt_state = optimizer.init(eqx.filter(model, filter_spec))

# Single training step
batch = (positions, packed_seds, target_stars, masks)
model, opt_state, loss_val = make_step(
    model, opt_state, optimizer, batch, loss_fn, filter_spec
)
print(f"Loss: {loss_val:.6f}")
```

## Saving and Loading

```python
from wavediff_jax.training.callbacks import save_checkpoint, load_checkpoint

# Save
save_checkpoint(model, "my_model.eqx")

# Load (needs a model template with correct structure)
loaded_model = load_checkpoint(model, "my_model.eqx")
```

## Using the Factory

Models can also be created via the registry:

```python
import wavediff_jax as wdj

# All registered models
print(wdj.PSF_FACTORY.keys())
# dict_keys(['poly', 'semi-param', 'physical-poly', ...])

# Create via factory (requires config namespace objects)
model = wdj.get_psf_model(model_params, training_params, data)
```

## Next Steps

- [Concepts](concepts.md) — understand Zernike polynomials, BCD training, and model architectures
- [PSF Models](../guide/models.md) — detailed guide to all three model types
- [Training](../guide/training.md) — full training loop with BCD cycles
