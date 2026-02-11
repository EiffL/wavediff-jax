# Design Decisions

This page documents the key architectural decisions made during the JAX reimplementation.

## 1. Pre-pad to `max_phase_N` for vmap

**Problem:** The TF code creates a new `TFMonochromaticPSF` object per wavelength with different `phase_N` (FFT grid sizes). Under JAX JIT, array shapes must be static.

**Solution:** Pre-pad all OPDs to `max(phase_N)` across the passband, enabling full `jax.vmap` parallelism over wavelengths. The `phase_N` values vary by ~20% across a typical passband, so the overhead from padding is modest compared to the parallelism gain.

```python
# All phase_N values extracted as concrete int before vmap
max_phase_N = int(jnp.max(packed_sed[:, 0]))
```

## 2. Auxiliary Returns Instead of `add_loss`

**Problem:** TF models use `self.add_loss()` to accumulate regularisation terms during the forward pass. This relies on mutable object state.

**Solution:** Equinox models return `(predictions, aux_dict)` and the loss function computes all terms explicitly:

```python
# Model forward pass
psf_batch, opd_maps, aux = model([positions, packed_seds])

# Loss function uses aux for regularisation
lp_reg = lp_regularization(aux['alpha_graph'], p=1.1)
total = data_loss + l2_param * l2_reg + l1_rate * lp_reg
```

This keeps the model purely functional and makes the loss computation transparent.

## 3. Functional Parameter Mutations

**Problem:** TF uses `tf.Variable.assign()` for in-place weight updates (e.g., zeroing alpha, projecting features).

**Solution:** All mutations use `eqx.tree_at` returning new model instances:

```python
def set_alpha_zero(model):
    return eqx.tree_at(
        lambda m: m.np_opd.alpha_mat,
        model,
        jnp.zeros_like(model.np_opd.alpha_mat),
    )
```

## 4. Custom Zernike Implementation

**Problem:** The TF code depends on the external `zernike` library (`jacopoantonello/zernike`) for `RZern` objects.

**Solution:** Reimplemented Zernike polynomial generation from the standard analytical formulae using pure numpy. Validated against:

- Known analytical values (piston, tip, tilt, defocus)
- Orthonormality over the unit disk
- Noll indexing convention

This removes a dependency and gives us full control over the implementation.

## 5. Thin-Plate Spline Interpolation in JAX

**Problem:** The TF code uses `tfa.image.interpolate_spline` (TensorFlow Addons), which has no JAX equivalent.

**Solution:** Implemented a pure JAX thin-plate RBF spline interpolator:

1. Build augmented system: `[K+λI, P; P^T, 0] [w; v] = [f; 0]`
2. Solve via `jnp.linalg.solve`
3. Evaluate at query points

JIT-compatible and validated against `scipy.interpolate.RBFInterpolator` to tolerance 1e-4.

## 6. Adam Optimizer (not RAdam)

**Problem:** Which optimizer to use? The TF code uses `tf.keras.optimizers.Adam`.

**Decision:** Use `optax.adam` with TF-matching defaults:

```python
optax.adam(lr, b1=0.9, b2=0.999, eps=1e-7)
```

Not RAdam (which was never used in the original) or any other variant.

## 7. Keep GalSim for Shape Metrics

**Problem:** `galsim.hsm.FindAdaptiveMom` has no JAX equivalent for adaptive moment measurement.

**Decision:** Keep GalSim as an optional dependency for shape metrics only. All other functionality works without it. Guarded with `try/except ImportError`.

## 8. Trainability via Filter Functions

**Problem:** TF uses `layer.trainable = False` for BCD. Equinox modules don't have a `trainable` flag.

**Solution:** Boolean filter pytrees + `eqx.partition`/`eqx.combine`:

```python
def param_filter(model):
    """True for PolynomialZernikeField.coeff_mat, False elsewhere."""
    filter_spec = jax.tree.map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda m: m.poly_field.coeff_mat, filter_spec, True
    )
    return filter_spec

# In training step:
trainable, frozen = eqx.partition(model, filter_spec)
grad_fn = eqx.filter_value_and_grad(loss_fn)
loss, grads = grad_fn(trainable, frozen, batch)
```

## 9. Mask Convention: 0 = Include

Preserved the TF convention where mask value 0 means "include this pixel" and 1 means "exclude". This ensures compatibility with existing datasets and avoids subtle bugs from convention mismatch.

## 10. Static Fields for Large Arrays

**Problem:** JAX traces treat array-valued fields as tracers during JIT, but large static arrays (Zernike maps, obscurations, observation positions) should not be traced.

**Solution:** Convert to numpy arrays before storing in Equinox modules:

```python
def _to_numpy(x):
    """Convert to numpy for safe static storage."""
    return np.asarray(x)

class ZernikeOPD(eqx.Module):
    zernike_maps: np.ndarray  # stored as numpy, not traced
```

This avoids "JAX array set as static" warnings and prevents unnecessary recompilation.
