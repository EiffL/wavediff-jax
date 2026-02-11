# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WaveDiff-JAX is a JAX/Equinox reimplementation of the WaveDiff differentiable PSF (Point Spread Function) modelling framework for astronomical imaging (originally TensorFlow/Keras). It models spatially-varying, polychromatic PSFs using Zernike polynomial wavefronts and FFT-based diffraction, with applications to Euclid-class space telescope data.

The original TF implementation lives at `extern/wf-psf/` (read-only Git submodule). **Do NOT install or import TensorFlow** — the submodule is reference-only.

## Build & Development

```bash
# Setup (use the existing .venv)
source .venv/bin/activate
pip install -e ".[dev]"

# Run full test suite (217 tests)
python -m pytest src/wavediff_jax/tests/ -v

# Run a single test file
python -m pytest src/wavediff_jax/tests/test_optics/test_zernike.py -v

# Run a single test class or method
python -m pytest src/wavediff_jax/tests/test_models/test_models.py::TestParametricPSFFieldModel -v
python -m pytest src/wavediff_jax/tests/test_training/test_training.py::TestMakeStep::test_loss_decreases -v

# Build docs
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve        # local preview at http://127.0.0.1:8000
mkdocs build        # static site in site/
```

## Architecture

### Package Layout

```
src/wavediff_jax/
├── optics/          # Pure JAX optical computation kernels
├── models/          # Equinox PSF field models + registry
├── training/        # Functional training loop (BCD)
├── data/            # Data loading (numpy → JAX arrays)
├── sims/            # Ground-truth PSF simulation (numpy)
├── metrics/         # Evaluation (pixel, shape, OPD metrics)
├── utils/           # Config, I/O, math utilities
├── plotting/        # Matplotlib/seaborn visualisation
└── tests/           # Full test suite (217 tests)
```

### Physics Pipeline

The forward model flows: **Positions → Zernike coefficients → OPD → Phase → FFT → PSF**

1. `PolynomialZernikeField` maps field positions to Zernike coefficient vectors via a trainable polynomial matrix
2. `ZernikeOPD` converts coefficients to Optical Path Difference maps using a precomputed Zernike basis
3. `build_phase` converts OPD to complex pupil phase, applies obscurations, zero-pads
4. `fft_diffract` performs FFT, computes intensity, crops, and downsamples to output resolution
5. For polychromatic PSFs, `jax.vmap` over wavelength bins with SED weighting

### Three PSF Model Architectures

All registered in `PSF_FACTORY` via `@register_psfclass`:

| Model ID | Class | Description |
|---|---|---|
| `"poly"` | `ParametricPSFFieldModel` | Purely parametric (polynomial Zernike field) |
| `"semi-param"` | `SemiParametricField` | Parametric + non-parametric OPD refinement |
| `"physical-poly"` | `PhysicalPolychromaticField` | Physical prior + learned correction + non-parametric |

Ground-truth variants: `"ground-truth-semi-param"`, `"ground-truth-physical-poly"`

### Key Equinox Patterns

- **Immutable models**: All parameter mutations use `eqx.tree_at` returning new model instances. No in-place assignment.
- **Trainability control**: `eqx.partition`/`eqx.combine` with boolean filter pytrees (`param_filter`, `nonparam_filter`, `complete_filter`) for Block Coordinate Descent.
- **Auxiliary returns**: Forward pass returns `(psf_batch, opd_maps)` or `(psf_batch, opd_maps, aux_dict)`. Regularization terms (Lp on alpha_graph) are in `aux_dict`, not `add_loss()`.
- **Batch via vmap**: `jax.vmap` replaces `tf.map_fn` for parallelism over stars and wavelengths.
- **Static shapes under JIT**: All `phase_N` values pre-padded to `max_phase_N` to enable full vmap.
- **Checkpoints**: `eqx.tree_serialise_leaves` / `eqx.tree_deserialise_leaves` for model persistence.

### Training: Block Coordinate Descent (BCD)

The `general_train_cycle` function supports 5 cycle types:
- `"parametric"` — train only `PolynomialZernikeField.coeff_mat`
- `"non-parametric"` — train only `S_mat`/`alpha_mat` (or graph equivalents)
- `"complete"` — train all parameters
- `"only-parametric"` / `"only-non-parametric"` — single-phase training

Each cycle: `eqx.filter_value_and_grad` → `optax.update` → `eqx.apply_updates`. L1 rate halves every 10 epochs.

### Data Flow

- `.npz` files → `DataHandler` → JAX arrays (positions, stars, SEDs)
- SED processing: `generate_packed_elems` → packed SED arrays `(n_stars, n_wavelengths, 3)` with `[phase_N, lambda, weight]`
- Masks follow TF convention: **0 = include, 1 = exclude**

## Key Dependencies

| Package | Role |
|---|---|
| `jax`/`jaxlib` | Autodiff, JIT, vmap |
| `equinox` | PyTree neural network modules |
| `optax` | Optimizers (Adam) |
| `numpy`/`scipy` | Init-time computation, simulation |
| `galsim` | Shape metrics only (optional, `pip install ".[metrics]"`) |

## Conventions

- Zernike indexing: **Noll convention** (1-based), implemented in `optics/zernike.py`
- Wavelengths: in **micrometres (um)**
- OPD: in **micrometres (um)**
- Positions: field coordinates, scaled to `[-1, 1]` internally via `x_lims`/`y_lims`
- Default optimizer: `optax.adam(lr, b1=0.9, b2=0.999, eps=1e-7)` matching TF defaults
- Obscurations: complex-valued masks (Euclid pupil geometry)

## Submodule Management

```bash
git submodule update --init --recursive
```
