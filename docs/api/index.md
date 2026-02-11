# API Reference

Complete reference for all public modules in WaveDiff-JAX.

## Package Structure

| Module | Description |
|---|---|
| [`wavediff_jax.optics`](optics.md) | Pure JAX optical computation: Zernike polynomials, FFT diffraction, obscurations, interpolation |
| [`wavediff_jax.models`](models.md) | Equinox PSF field models: layers, parametric, semi-parametric, physical, ground-truth, registry |
| [`wavediff_jax.training`](training.md) | Functional training: BCD loop, losses, callbacks, utilities |
| [`wavediff_jax.data`](data.md) | Data loading and SED preprocessing |
| [`wavediff_jax.metrics`](metrics.md) | Evaluation: pixel RMSE, shape metrics, OPD metrics |
| [`wavediff_jax.sims`](sims.md) | Ground-truth PSF simulation |
| [`wavediff_jax.utils`](utils.md) | Configuration, I/O, math utilities, centroids |
| [`wavediff_jax.plotting`](plotting.md) | Matplotlib/seaborn visualisation |

## Top-Level Exports

The `wavediff_jax` package exports the most commonly used functions:

```python
import wavediff_jax as wdj

# Model creation
wdj.get_psf_model(...)
wdj.PSF_FACTORY          # dict of registered models
wdj.register_psfclass     # decorator

# Training
wdj.general_train_cycle(...)
wdj.train_epoch(...)
wdj.make_step(...)

# Losses
wdj.total_loss(...)
wdj.mse_loss(...)
wdj.masked_mse_loss(...)

# Checkpoints
wdj.save_checkpoint(...)
wdj.load_checkpoint(...)

# Data
wdj.DataHandler(...)

# Metrics
wdj.compute_poly_metric(...)
wdj.compute_mono_metric(...)
wdj.compute_opd_metrics(...)
wdj.compute_shape_metrics(...)
```
