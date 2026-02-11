# WaveDiff-JAX

> [!WARNING]
> **This is an experimental JAX port and is not validated for scientific use.**
> If you are looking for the production-ready, peer-reviewed PSF modelling framework, please use the original **[WaveDiff (wf-psf)](https://github.com/CosmoStat/wf-psf)** by CosmoStat.
> This repository is a work-in-progress reimplementation for development and experimentation purposes only. Numerical equivalence with the original has **not** been fully verified. Do not use this for published science results.

**Differentiable PSF modelling for astronomical imaging, powered by JAX.**

WaveDiff-JAX is an experimental reimplementation of the [WaveDiff](https://github.com/CosmoStat/wf-psf) framework in [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox). It models spatially-varying, polychromatic Point Spread Functions (PSFs) using Zernike polynomial wavefronts and FFT-based Fourier optics — fully differentiable and JIT-compiled for high-performance training on GPU/TPU.

---

## Highlights

- **Pure JAX** — all differentiable operations in JAX; `jax.vmap` replaces nested loops over stars and wavelengths
- **Equinox modules** — clean PyTree-based model definitions with functional parameter updates
- **Three PSF architectures** — parametric, semi-parametric, and physics-informed polychromatic models
- **Block Coordinate Descent** — alternating optimisation of parametric and non-parametric components
- **Euclid-ready** — models the Euclid space telescope pupil geometry including spider arms and central obscuration
- **217 tests** covering optics primitives, model layers, training loops, data loading, simulation, and metrics

## Physics Pipeline

```
Field positions ──► Polynomial Zernike field ──► Zernike coefficients
                                                        │
                                                        ▼
                                                   Zernike basis
                                                        │
                                                        ▼
            Obscurations ──► build_phase ◄── OPD (Optical Path Difference)
                                │
                                ▼
                           FFT diffract
                                │
                                ▼
            SED weighting ──► Polychromatic PSF
```

## Installation

```bash
# Clone with reference submodule
git clone --recurse-submodules https://github.com/your-org/wavediff-jax.git
cd wavediff-jax

# Create environment and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10, JAX >= 0.4.20, Equinox >= 0.11.0, Optax >= 0.1.7

For shape metrics (galsim): `pip install -e ".[metrics]"`

## Quick Start

```python
import jax
import jax.numpy as jnp
import wavediff_jax as wdj
from wavediff_jax.utils.math_utils import generate_zernike_maps_3d, obscurations_from_params

# Generate optical elements
zernike_maps = generate_zernike_maps_3d(n_zernikes=15, pupil_diam=256)
obscurations = obscurations_from_params(pupil_diam=256)

# Create a parametric PSF model
key = jax.random.PRNGKey(42)
model = wdj.models.parametric.ParametricPSFFieldModel(
    zernike_maps=zernike_maps,
    obscurations=obscurations,
    output_Q=1,
    output_dim=64,
    n_zernikes=15,
    d_max=2,
    x_lims=[0, 1e3],
    y_lims=[0, 1e3],
    key=key,
)

# Forward pass: positions + packed SEDs → polychromatic PSFs
positions = jnp.array([[500.0, 500.0], [200.0, 800.0]])
packed_seds = jnp.zeros((2, 10, 3))  # (n_stars, n_wavelengths, 3)
psf_batch, opd_maps = model([positions, packed_seds])

# Predict OPD maps
opd = model.predict_opd(positions)  # shape: (2, 256, 256)
```

## PSF Model Architectures

| Model | ID | Description |
|---|---|---|
| **Parametric** | `"poly"` | Trainable polynomial Zernike field maps positions to wavefront coefficients |
| **Semi-Parametric** | `"semi-param"` | Adds a non-parametric OPD component (polynomial, MCCD, or graph-based) to capture residuals beyond the parametric model |
| **Physical Polychromatic** | `"physical-poly"` | Combines a physics-based Zernike prior (e.g. from ray-tracing) with learned parametric corrections and non-parametric refinement |

Models are registered via a factory pattern and can be instantiated by name:

```python
model = wdj.get_psf_model(model_params, training_params, data)
```

## Training

Training uses a functional Equinox + Optax loop with Block Coordinate Descent (BCD):

```python
from wavediff_jax.training.trainer import general_train_cycle

# training_hparams is a namespace with lr, batch_size, n_epochs, cycle_def, etc.
results = general_train_cycle(
    model=model,
    train_data=train_data,
    val_data=val_data,
    training_hparams=training_hparams,
    key=jax.random.PRNGKey(0),
)

trained_model = results["model"]
```

**BCD cycle types:**
- `"parametric"` — optimise polynomial Zernike coefficients only
- `"non-parametric"` — optimise non-parametric OPD dictionaries only
- `"complete"` — optimise all parameters jointly

**Checkpointing:**
```python
wdj.save_checkpoint(model, "model.eqx")
model = wdj.load_checkpoint(model_template, "model.eqx")
```

## Evaluation

```python
from wavediff_jax.metrics.metrics import (
    compute_poly_metric,
    compute_opd_metrics,
    compute_shape_metrics,
)

# Polychromatic pixel RMSE
rmse, rel_rmse, std, rel_std = compute_poly_metric(
    model, gt_model, positions, packed_seds
)

# OPD RMSE
opd_rmse, opd_rel, opd_std, opd_rel_std = compute_opd_metrics(
    model, gt_model, positions
)

# Shape metrics (requires galsim)
shape_dict = compute_shape_metrics(predicted_psfs, target_psfs)
# → {'rmse_e1': ..., 'rmse_e2': ..., 'rmse_R2_meanR2': ..., ...}
```

## Project Structure

```
src/wavediff_jax/
├── optics/                     # Pure JAX optical computation
│   ├── zernike.py              # Noll-indexed Zernike polynomials
│   ├── diffraction.py          # FFT diffraction, phase construction
│   ├── obscurations.py         # Euclid pupil geometry
│   └── interpolation.py        # Thin-plate RBF spline (replaces tfa)
├── models/                     # Equinox PSF field models
│   ├── registry.py             # Factory pattern (@register_psfclass)
│   ├── layers.py               # All layer modules (8 classes)
│   ├── parametric.py           # ParametricPSFFieldModel
│   ├── semiparametric.py       # SemiParametricField
│   ├── physical_polychromatic.py  # PhysicalPolychromaticField
│   └── ground_truth.py         # Ground-truth wrappers
├── training/                   # Functional training
│   ├── trainer.py              # BCD loop, make_step, train_epoch
│   ├── losses.py               # MSE, masked MSE, regularisation
│   ├── callbacks.py            # Checkpoint I/O, L1 scheduling
│   └── train_utils.py          # Sample weights, optimizer config
├── data/                       # Data loading
│   └── preprocessing.py        # DataHandler, SED processing
├── sims/                       # Ground-truth simulation (numpy)
│   ├── psf_simulator.py        # PSFSimulator
│   └── spatial_varying_psf.py  # Spatially-varying PSF synthesis
├── metrics/                    # Evaluation
│   ├── metrics.py              # Pixel, shape, OPD metrics
│   └── metrics_interface.py    # Evaluation orchestration
├── utils/                      # Utilities
│   ├── config.py               # YAML config → RecursiveNamespace
│   ├── io.py                   # FileIOHandler
│   ├── math_utils.py           # Polynomial basis, Zernike maps
│   ├── centroids.py            # CentroidEstimator
│   ├── ccd_misalignments.py    # CCD geometry
│   └── preprocessing.py        # Coordinate transforms
├── plotting/                   # Visualisation
│   └── plots_interface.py      # Metrics plotting
└── tests/                      # 217 tests
    ├── test_optics/            # 38 tests
    ├── test_models/            # 34 tests
    ├── test_training/          # 35 tests
    ├── test_data/              # 17 tests
    ├── test_sims/              # 19 tests
    ├── test_metrics/           # 29 tests
    └── test_utils/             # 45 tests
```

## Testing

```bash
# Full suite
python -m pytest src/wavediff_jax/tests/ -v

# Single subpackage
python -m pytest src/wavediff_jax/tests/test_optics/ -v

# Single test
python -m pytest src/wavediff_jax/tests/test_models/test_models.py::TestParametricPSFFieldModel::test_forward_pass_shapes -v

# With parallelism
python -m pytest src/wavediff_jax/tests/ -n auto
```

## Key Design Decisions

| TensorFlow Pattern | JAX/Equinox Replacement |
|---|---|
| `tf.keras.layers.Layer` | `eqx.Module` |
| `self.add_loss()` | Auxiliary return `(output, aux_dict)` |
| `layer.trainable = False` | `eqx.partition` with filter functions |
| `tf.Variable.assign()` | `eqx.tree_at` (returns new model) |
| `tf.map_fn` | `jax.vmap` |
| `model.compile() + model.fit()` | Explicit `make_step` + `train_epoch` |
| `model.save_weights()` | `eqx.tree_serialise_leaves` |
| `tfa.image.interpolate_spline` | Custom `thin_plate_spline_interpolate` |

## Documentation

```bash
pip install mkdocs mkdocs-material "mkdocstrings[python]"
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## References

- **Original WaveDiff:** [CosmoStat/wf-psf](https://github.com/CosmoStat/wf-psf)
- **Liaudat et al. (2023):** *Multi-CCD modelling of the point spread function* ([arXiv:2306.07996](https://arxiv.org/abs/2306.07996))
- **JAX:** [github.com/jax-ml/jax](https://github.com/jax-ml/jax)
- **Equinox:** [github.com/patrick-kidger/equinox](https://github.com/patrick-kidger/equinox)

## License

MIT
