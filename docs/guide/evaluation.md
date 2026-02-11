# Evaluation

WaveDiff-JAX provides comprehensive metrics for evaluating PSF model quality at the pixel, wavefront, and shape levels.

## Pixel Metrics

### Polychromatic RMSE

Compare predicted polychromatic PSFs against ground truth:

```python
from wavediff_jax.metrics.metrics import compute_poly_metric

rmse, rel_rmse, std_rmse, std_rel_rmse = compute_poly_metric(
    model=trained_model,
    gt_model=ground_truth_model,
    positions=test_positions,
    packed_seds=test_packed_seds,
    batch_size=16,
)

print(f"RMSE: {rmse:.4e} +/- {std_rmse:.4e}")
print(f"Relative RMSE: {rel_rmse:.2f}% +/- {std_rel_rmse:.2f}%")
```

You can also provide precomputed ground-truth stars:

```python
rmse, rel_rmse, std_rmse, std_rel_rmse = compute_poly_metric(
    model=trained_model,
    gt_model=None,
    positions=test_positions,
    packed_seds=test_packed_seds,
    gt_stars=precomputed_gt_psfs,  # skip gt_model forward pass
)
```

### Monochromatic RMSE

Per-wavelength evaluation:

```python
from wavediff_jax.metrics.metrics import compute_mono_metric

lambda_list = [0.55, 0.65, 0.75, 0.85, 0.95]

rmse_lda, rel_rmse_lda, std_lda, std_rel_lda = compute_mono_metric(
    model=trained_model,
    gt_model=ground_truth_model,
    positions=test_positions,
    lambda_list=lambda_list,
    batch_size=32,
)

for lda, r, rr in zip(lambda_list, rmse_lda, rel_rmse_lda):
    print(f"λ={lda:.2f}um: RMSE={r:.4e}, Rel={rr:.2f}%")
```

## OPD Metrics

Compare predicted OPD (wavefront) maps:

```python
from wavediff_jax.metrics.metrics import compute_opd_metrics

rmse, rel_rmse, std, rel_std = compute_opd_metrics(
    model=trained_model,
    gt_model=ground_truth_model,
    positions=test_positions,
)
```

OPD metrics apply:

- Mean removal (each OPD map is zero-meaned before comparison)
- Obscuration masking (only pupil-interior pixels are compared)

## Shape Metrics

PSF shape metrics using GalSim's adaptive moments:

```python
from wavediff_jax.metrics.metrics import compute_shape_metrics

# Requires galsim: pip install ".[metrics]"
result = compute_shape_metrics(predicted_psfs, target_psfs)

print(f"σ(e1) RMSE: {result['rmse_e1']:.4e}")
print(f"σ(e2) RMSE: {result['rmse_e2']:.4e}")
print(f"σ(R2)/<R2>: {result['rmse_R2_meanR2']:.4e}")
print(f"Pixel RMSE: {result['pix_rmse']:.4e}")
```

### Available Shape Metric Keys

| Key | Description |
|---|---|
| `pred_e1_HSM`, `pred_e2_HSM` | Predicted ellipticity components |
| `pred_R2_HSM` | Predicted size (2σ²) |
| `gt_pred_e1_HSM`, `gt_pred_e2_HSM` | Ground-truth ellipticity |
| `gt_pred_R2_HSM` | Ground-truth size |
| `rmse_e1`, `std_rmse_e1` | Ellipticity e1 RMSE and std |
| `rmse_e2`, `std_rmse_e2` | Ellipticity e2 RMSE and std |
| `rel_rmse_e1`, `rel_rmse_e2` | Relative ellipticity RMSE (%) |
| `rmse_R2_meanR2`, `std_rmse_R2_meanR2` | Normalised size RMSE |
| `pix_rmse`, `pix_rmse_std` | Pixel-level RMSE |
| `rel_pix_rmse`, `rel_pix_rmse_std` | Relative pixel RMSE (%) |

## Helper Functions

Low-level utilities:

```python
from wavediff_jax.metrics.metrics import compute_residuals, compute_rmse, compute_psf_images

# Per-star RMSE
residuals = compute_residuals(predictions, targets)  # (n_stars,)

# Overall RMSE
rmse = compute_rmse(predictions, targets)  # scalar

# Batched PSF generation from model
psf_images = compute_psf_images(model, positions, packed_seds, batch_size=16)
```

## Metrics Interface

For automated evaluation across datasets:

```python
from wavediff_jax.metrics.metrics_interface import evaluate_model

results = evaluate_model(
    model=trained_model,
    gt_model=ground_truth_model,
    train_data=train_dataset,
    test_data=test_dataset,
    metrics_params=metrics_config,
    compute_poly=True,
    compute_opd=True,
    compute_mono=True,
    compute_shape=True,
)
```
