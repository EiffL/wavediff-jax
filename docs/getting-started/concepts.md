# Key Concepts

## Zernike Polynomials

Zernike polynomials form an orthonormal basis on the unit disk, making them ideal for representing optical wavefront aberrations. WaveDiff-JAX uses the **Noll indexing convention** (1-based), where:

| Noll index | Name | Formula |
|---|---|---|
| 1 | Piston | 1 |
| 2 | Tip (x-tilt) | \\(2\rho\cos\theta\\) |
| 3 | Tilt (y-tilt) | \\(2\rho\sin\theta\\) |
| 4 | Defocus | \\(\sqrt{3}(2\rho^2 - 1)\\) |
| 5 | Oblique astigmatism | \\(\sqrt{6}\rho^2\sin 2\theta\\) |
| 6 | Vertical astigmatism | \\(\sqrt{6}\rho^2\cos 2\theta\\) |

The wavefront at any field position is described as a weighted sum of Zernike polynomials:

\\[
W(\rho, \theta) = \sum_{j=1}^{N} a_j \, Z_j(\rho, \theta)
\\]

where \\(a_j\\) are the Zernike coefficients and \\(Z_j\\) are the basis functions.

## Optical Path Difference (OPD)

The OPD is the 2D map of wavefront errors across the telescope pupil. It represents how much the actual wavefront deviates from a perfect sphere:

- **Units:** micrometres (um)
- **Shape:** `(pupil_diam, pupil_diam)` — typically 256x256
- Computed via `ZernikeOPD`: weighted sum of Zernike basis maps

## PSF Formation (Fourier Optics)

The PSF is formed via **Fraunhofer diffraction**:

1. Convert OPD to complex phase: \\(\phi = \exp\left(i \frac{2\pi}{\lambda} \cdot \text{OPD}\right)\\)
2. Multiply by pupil obscuration mask (spider arms, central obscuration)
3. Zero-pad to `phase_N` pixels for desired sampling
4. Compute \\(|\text{FFT}(\phi)|^2\\) — the PSF intensity
5. Crop and downsample to output resolution

## Polychromatic PSFs and SEDs

Real astronomical observations are polychromatic. The polychromatic PSF is a weighted sum of monochromatic PSFs:

\\[
\text{PSF}_{\text{poly}} = \sum_{k=1}^{N_\lambda} w_k \cdot \text{PSF}_{\text{mono}}(\lambda_k)
\\]

where \\(w_k\\) are SED (Spectral Energy Distribution) weights. In WaveDiff-JAX, this sum is computed via `jax.vmap` over wavelength bins for maximum parallelism.

**Packed SED format:** Each star's SED is encoded as `(n_wavelengths, 3)` where columns are `[phase_N, lambda_obs, sed_weight]`.

## Field Variation

PSFs vary across the telescope's field of view. WaveDiff-JAX models this variation by making the Zernike coefficients position-dependent:

\\[
a_j(x, y) = \sum_{k=0}^{N_{\text{poly}}} c_{jk} \, P_k(x, y)
\\]

where \\(P_k\\) are 2D polynomials up to degree `d_max` and \\(c_{jk}\\) is the trainable coefficient matrix. Field positions are normalised to \\([-1, 1]\\) via `x_lims`/`y_lims`.

## Equinox Modules

All models are [Equinox](https://github.com/patrick-kidger/equinox) modules — JAX-compatible PyTrees with:

- **Immutability:** Parameters are never modified in-place. Updates create new model instances via `eqx.tree_at`.
- **JIT compatibility:** Models can be JIT-compiled for performance.
- **Trainability control:** `eqx.partition`/`eqx.combine` split models into trainable and frozen parts.

```python
import equinox as eqx

# Functional parameter update
new_model = eqx.tree_at(
    lambda m: m.poly_field.coeff_mat,
    model,
    new_coeff_mat,
)
```

## Block Coordinate Descent (BCD)

Semi-parametric and physical models have two groups of parameters:

1. **Parametric:** The polynomial Zernike field coefficients (`coeff_mat`)
2. **Non-parametric:** Spatial OPD dictionaries (`S_mat`, `alpha_mat`, or graph features)

BCD alternates between optimising these groups:

```
Cycle 1: Train parametric only    → freeze non-parametric
Cycle 2: Train non-parametric only → freeze parametric
Cycle 3: Train all jointly        → (optional)
```

This is implemented via filter functions that create boolean pytrees:

```python
from wavediff_jax.training.trainer import param_filter, nonparam_filter

# True for parametric params, False for everything else
param_spec = param_filter(model)

# True for non-parametric params only
nonparam_spec = nonparam_filter(model)
```

## Telescope Obscurations

WaveDiff-JAX models the Euclid space telescope pupil, which includes:

- **Primary mirror:** 1.2m diameter aperture
- **Secondary mirror:** Central obscuration (395mm)
- **Spider arms:** Three support struts at specific angles
- **Low-pass filtering:** Optional smoothing to reduce diffraction artefacts

The obscuration mask is complex-valued and multiplied into the pupil phase before FFT.

## Mask Convention

Masks follow the TensorFlow convention: **0 = include, 1 = exclude**. This is used consistently throughout data loading, training losses, and metric computation.
