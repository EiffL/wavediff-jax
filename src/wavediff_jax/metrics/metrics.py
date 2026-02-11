"""Metrics for PSF model evaluation.

Provides functions for computing pixel-level, shape, and OPD metrics
for polychromatic and monochromatic PSF reconstructions.

Port of ``extern/wf-psf/src/wf_psf/metrics/metrics.py`` to JAX.

:Authors: WaveDiff-JAX contributors
"""

import numpy as np
import jax.numpy as jnp
import logging

logger = logging.getLogger(__name__)

try:
    import galsim as gs

    HAS_GALSIM = True
except ImportError:
    HAS_GALSIM = False


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def compute_psf_images(model, positions, packed_seds, batch_size=None):
    """Compute PSF images from a model.

    Parameters
    ----------
    model : eqx.Module
        PSF model (Equinox module, callable with ``[positions, packed_seds]``).
    positions : jnp.ndarray, shape (n_samples, 2)
        Star positions.
    packed_seds : jnp.ndarray, shape (n_samples, n_wavelengths, 3)
        Packed SED data.
    batch_size : int or None
        If not None, process in batches of this size to save memory.

    Returns
    -------
    predictions : np.ndarray, shape (n_samples, output_dim, output_dim)
        Predicted PSF images.
    """
    n_samples = positions.shape[0]

    if batch_size is None or batch_size >= n_samples:
        outputs = model([positions, packed_seds])
        psf_batch = outputs[0]
        return np.asarray(psf_batch)

    predictions = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_pos = positions[start:end]
        batch_seds = packed_seds[start:end]
        outputs = model([batch_pos, batch_seds])
        predictions.append(np.asarray(outputs[0]))

    return np.concatenate(predictions, axis=0)


def compute_residuals(predictions, targets):
    """Compute per-star pixel residuals (RMSE per star).

    Parameters
    ----------
    predictions : np.ndarray, shape (n_samples, H, W)
        Predicted images.
    targets : np.ndarray, shape (n_samples, H, W)
        Target images.

    Returns
    -------
    residuals : np.ndarray, shape (n_samples,)
        Per-star RMSE values.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    return np.sqrt(np.mean((targets - predictions) ** 2, axis=(1, 2)))


def compute_rmse(predictions, targets):
    """Compute overall RMSE between predictions and targets.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted values.
    targets : np.ndarray
        Target values.

    Returns
    -------
    rmse : float
        Scalar RMSE value.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    return float(np.sqrt(np.mean((targets - predictions) ** 2)))


# ---------------------------------------------------------------------------
# Polychromatic PSF metric
# ---------------------------------------------------------------------------


def compute_poly_metric(
    model,
    gt_model,
    positions,
    packed_seds,
    gt_packed_seds=None,
    gt_stars=None,
    batch_size=16,
    masks=None,
):
    """Calculate metrics for polychromatic PSF reconstructions.

    Computes absolute and relative RMSE (with standard deviations) comparing
    model predictions to ground-truth PSFs.

    Parameters
    ----------
    model : eqx.Module
        Trained PSF model to evaluate.
    gt_model : eqx.Module or None
        Ground-truth model. Used to generate target PSFs if ``gt_stars`` is
        not provided.
    positions : array-like, shape (n_samples, 2)
        Positions at which to evaluate the model.
    packed_seds : array-like, shape (n_samples, n_wavelengths, 3)
        Packed SED data for the trained model.
    gt_packed_seds : array-like or None
        Packed SED data for the ground-truth model. If None, ``packed_seds``
        is used.
    gt_stars : array-like or None
        Precomputed ground-truth PSF images. If provided, ``gt_model`` is
        not called.
    batch_size : int
        Batch size for inference.
    masks : array-like or None
        Masks for the predictions. Convention: 0 = include, 1 = exclude.
        If provided, only unmasked pixels are used for metric computation.

    Returns
    -------
    rmse : float
        Mean absolute RMSE.
    rel_rmse : float
        Mean relative RMSE in percent.
    std_rmse : float
        Standard deviation of absolute RMSE.
    std_rel_rmse : float
        Standard deviation of relative RMSE in percent.
    """
    positions = jnp.asarray(positions)
    packed_seds = jnp.asarray(packed_seds)

    # Model predictions
    preds = compute_psf_images(model, positions, packed_seds, batch_size=batch_size)

    # Ground truth
    if gt_stars is not None:
        gt_preds = np.asarray(gt_stars)
    else:
        if gt_packed_seds is None:
            gt_packed_seds = packed_seds
        else:
            gt_packed_seds = jnp.asarray(gt_packed_seds)
        gt_preds = compute_psf_images(
            gt_model, positions, gt_packed_seds, batch_size=batch_size
        )

    # Masking
    if masks is not None:
        masks = np.asarray(masks)
        keep = 1.0 - masks.astype(preds.dtype)
        weights = np.maximum(np.sum(keep, axis=(1, 2)), 1e-7)
        preds = preds * keep
        gt_preds = gt_preds * keep
    else:
        weights = np.ones(gt_preds.shape[0]) * gt_preds.shape[1] * gt_preds.shape[2]

    # Per-star RMSE
    residuals = np.sqrt(np.sum((gt_preds - preds) ** 2, axis=(1, 2)) / weights)
    gt_star_mean = np.sqrt(np.sum(gt_preds ** 2, axis=(1, 2)) / weights)

    rmse = float(np.mean(residuals))
    rel_rmse = float(100.0 * np.mean(residuals / gt_star_mean))
    std_rmse = float(np.std(residuals))
    std_rel_rmse = float(100.0 * np.std(residuals / gt_star_mean))

    logger.info("Absolute RMSE:\t %.4e \t +/- %.4e", rmse, std_rmse)
    logger.info("Relative RMSE:\t %.4e %% \t +/- %.4e %%", rel_rmse, std_rel_rmse)

    return rmse, rel_rmse, std_rmse, std_rel_rmse


# ---------------------------------------------------------------------------
# Monochromatic PSF metric
# ---------------------------------------------------------------------------


def compute_mono_metric(
    model,
    gt_model,
    positions,
    lambda_list,
    phase_N_fn=None,
    batch_size=32,
):
    """Calculate metrics for monochromatic PSF reconstructions.

    Evaluates a model at a list of wavelengths and computes per-wavelength
    RMSE statistics.

    Parameters
    ----------
    model : eqx.Module
        Trained model with ``predict_mono_psfs`` method.
    gt_model : eqx.Module
        Ground-truth model with ``predict_mono_psfs`` method.
    positions : array-like, shape (n_samples, 2)
        Positions to evaluate.
    lambda_list : array-like
        List of wavelength values (in um).
    phase_N_fn : callable or None
        Function mapping wavelength -> phase_N integer. If None, a fixed
        default of 914 is used.
    batch_size : int
        Batch size for monochromatic PSF calculations.

    Returns
    -------
    rmse_lda : list of float
        Per-wavelength absolute RMSE.
    rel_rmse_lda : list of float
        Per-wavelength relative RMSE in percent.
    std_rmse_lda : list of float
        Per-wavelength standard deviation of RMSE.
    std_rel_rmse_lda : list of float
        Per-wavelength standard deviation of relative RMSE in percent.
    """
    positions = jnp.asarray(positions)
    total_samples = positions.shape[0]

    rmse_lda = []
    rel_rmse_lda = []
    std_rmse_lda = []
    std_rel_rmse_lda = []

    for lambda_obs in lambda_list:
        lambda_obs = float(lambda_obs)
        phase_N = int(phase_N_fn(lambda_obs)) if phase_N_fn is not None else 914

        residuals = np.zeros(total_samples)
        gt_star_mean = np.zeros(total_samples)

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_pos = positions[start:end]

            gt_mono = np.asarray(
                gt_model.predict_mono_psfs(
                    batch_pos, lambda_obs=lambda_obs, phase_N=phase_N
                )
            )
            model_mono = np.asarray(
                model.predict_mono_psfs(
                    batch_pos, lambda_obs=lambda_obs, phase_N=phase_N
                )
            )

            num_pixels = gt_mono.shape[1] * gt_mono.shape[2]
            residuals[start:end] = (
                np.sum((gt_mono - model_mono) ** 2, axis=(1, 2)) / num_pixels
            )
            gt_star_mean[start:end] = (
                np.sum(gt_mono ** 2, axis=(1, 2)) / num_pixels
            )

        residuals = np.sqrt(residuals)
        gt_star_mean = np.sqrt(gt_star_mean)

        rmse_lda.append(float(np.mean(residuals)))
        rel_rmse_lda.append(float(100.0 * np.mean(residuals / gt_star_mean)))
        std_rmse_lda.append(float(np.std(residuals)))
        std_rel_rmse_lda.append(float(100.0 * np.std(residuals / gt_star_mean)))

    return rmse_lda, rel_rmse_lda, std_rmse_lda, std_rel_rmse_lda


# ---------------------------------------------------------------------------
# OPD metric
# ---------------------------------------------------------------------------


def compute_opd_metrics(
    model,
    gt_model,
    positions,
    obscurations=None,
    batch_size=16,
):
    """Compute OPD RMSE metrics.

    Computes RMSE between predicted OPD maps (mean-removed, obscured) and
    ground-truth OPD maps.

    Parameters
    ----------
    model : eqx.Module
        Trained model with ``predict_opd`` method.
    gt_model : eqx.Module
        Ground-truth model with ``predict_opd`` method.
    positions : array-like, shape (n_samples, 2)
        Positions at which to predict OPD maps.
    obscurations : array-like or None
        Obscuration mask. If None, tries ``gt_model.batch_poly_psf.obscurations``.
    batch_size : int
        Batch size for OPD calculation.

    Returns
    -------
    rmse : float
        Absolute RMSE.
    rel_rmse : float
        Relative RMSE in percent.
    rmse_std : float
        Standard deviation of absolute RMSE.
    rel_rmse_std : float
        Standard deviation of relative RMSE in percent.
    """
    positions = jnp.asarray(positions)
    n_samples = positions.shape[0]

    # Get obscurations
    if obscurations is None:
        obscurations = np.real(np.asarray(gt_model.batch_poly_psf.obscurations))
    else:
        obscurations = np.real(np.asarray(obscurations))

    obsc_mask = obscurations > 0
    nb_mask_elems = np.sum(obsc_mask)

    rmse_vals = np.zeros(n_samples)
    rel_rmse_vals = np.zeros(n_samples)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_pos = positions[start:end]

        opd_batch = np.array(model.predict_opd(batch_pos), copy=True)
        gt_opd_batch = np.array(gt_model.predict_opd(batch_pos), copy=True)

        # Remove mean
        opd_batch -= np.mean(opd_batch, axis=(1, 2)).reshape(-1, 1, 1)
        gt_opd_batch -= np.mean(gt_opd_batch, axis=(1, 2)).reshape(-1, 1, 1)

        # Apply obscuration mask
        opd_batch *= obscurations
        gt_opd_batch *= obscurations

        # Per-sample RMSE on obscured elements
        res_opd = np.sqrt(
            np.array([
                np.sum((im1[obsc_mask] - im2[obsc_mask]) ** 2) / nb_mask_elems
                for im1, im2 in zip(opd_batch, gt_opd_batch)
            ])
        )
        gt_opd_mean = np.sqrt(
            np.array([
                np.sum(im2[obsc_mask] ** 2) / nb_mask_elems
                for im2 in gt_opd_batch
            ])
        )

        rmse_vals[start:end] = res_opd
        rel_rmse_vals[start:end] = 100.0 * (res_opd / gt_opd_mean)

    rmse = float(np.mean(rmse_vals))
    rel_rmse = float(np.mean(rel_rmse_vals))
    rmse_std = float(np.std(rmse_vals))
    rel_rmse_std = float(np.std(rel_rmse_vals))

    logger.info("OPD Absolute RMSE:\t %.4e \t +/- %.4e", rmse, rmse_std)
    logger.info("OPD Relative RMSE:\t %.4e %% \t +/- %.4e %%", rel_rmse, rel_rmse_std)

    return rmse, rel_rmse, rmse_std, rel_rmse_std


# ---------------------------------------------------------------------------
# Shape metric (requires galsim)
# ---------------------------------------------------------------------------


def compute_shape_metrics(predicted_psfs, target_psfs):
    """Compute e1, e2, R2 shape metrics using galsim adaptive moments.

    Measures the shape and size of predicted and target PSF images using
    ``galsim.hsm.FindAdaptiveMom``, then computes RMSE statistics for
    ellipticity components (e1, e2) and size (R2).

    Parameters
    ----------
    predicted_psfs : array-like, shape (n_samples, H, W)
        Predicted PSF images.
    target_psfs : array-like, shape (n_samples, H, W)
        Target (ground-truth) PSF images.

    Returns
    -------
    result_dict : dict
        Dictionary with keys:

        - ``pred_e1_HSM``, ``pred_e2_HSM``, ``pred_R2_HSM``
        - ``gt_pred_e1_HSM``, ``gt_pred_e2_HSM``, ``gt_pred_R2_HSM``
        - ``rmse_e1``, ``std_rmse_e1``, ``rel_rmse_e1``, ``std_rel_rmse_e1``
        - ``rmse_e2``, ``std_rmse_e2``, ``rel_rmse_e2``, ``std_rel_rmse_e2``
        - ``rmse_R2_meanR2``, ``std_rmse_R2_meanR2``
        - ``pix_rmse``, ``pix_rmse_std``, ``rel_pix_rmse``, ``rel_pix_rmse_std``

    Raises
    ------
    ImportError
        If galsim is not installed.
    """
    if not HAS_GALSIM:
        raise ImportError(
            "galsim is required for shape metrics. "
            "Install it with: pip install galsim"
        )

    predicted_psfs = np.asarray(predicted_psfs)
    target_psfs = np.asarray(target_psfs)

    # Pixel RMSE
    residuals = np.sqrt(np.mean((target_psfs - predicted_psfs) ** 2, axis=(1, 2)))
    gt_star_mean = np.sqrt(np.mean(target_psfs ** 2, axis=(1, 2)))

    pix_rmse = float(np.mean(residuals))
    rel_pix_rmse = float(100.0 * np.mean(residuals / gt_star_mean))
    pix_rmse_std = float(np.std(residuals))
    rel_pix_rmse_std = float(100.0 * np.std(residuals / gt_star_mean))

    logger.info("Pixel star absolute RMSE:\t %.4e \t +/- %.4e", pix_rmse, pix_rmse_std)
    logger.info(
        "Pixel star relative RMSE:\t %.4e %% \t +/- %.4e %%",
        rel_pix_rmse,
        rel_pix_rmse_std,
    )

    # Measure shapes
    pred_moments = [
        gs.hsm.FindAdaptiveMom(gs.Image(pred), strict=False) for pred in predicted_psfs
    ]
    gt_moments = [
        gs.hsm.FindAdaptiveMom(gs.Image(gt), strict=False) for gt in target_psfs
    ]

    pred_e1, pred_e2, pred_R2 = [], [], []
    gt_e1, gt_e2, gt_R2 = [], [], []

    for i in range(len(gt_moments)):
        if pred_moments[i].moments_status == 0 and gt_moments[i].moments_status == 0:
            pred_e1.append(pred_moments[i].observed_shape.g1)
            pred_e2.append(pred_moments[i].observed_shape.g2)
            pred_R2.append(2 * (pred_moments[i].moments_sigma ** 2))

            gt_e1.append(gt_moments[i].observed_shape.g1)
            gt_e2.append(gt_moments[i].observed_shape.g2)
            gt_R2.append(2 * (gt_moments[i].moments_sigma ** 2))

    pred_e1 = np.array(pred_e1)
    pred_e2 = np.array(pred_e2)
    pred_R2 = np.array(pred_R2)
    gt_e1 = np.array(gt_e1)
    gt_e2 = np.array(gt_e2)
    gt_R2 = np.array(gt_R2)

    # e1 metrics
    e1_res = gt_e1 - pred_e1
    e1_res_rel = e1_res / np.where(np.abs(gt_e1) > 1e-12, gt_e1, 1e-12)
    rmse_e1 = float(np.sqrt(np.mean(e1_res ** 2)))
    rel_rmse_e1 = float(100.0 * np.sqrt(np.mean(e1_res_rel ** 2)))
    std_rmse_e1 = float(np.std(e1_res))
    std_rel_rmse_e1 = float(100.0 * np.std(e1_res_rel))

    # e2 metrics
    e2_res = gt_e2 - pred_e2
    e2_res_rel = e2_res / np.where(np.abs(gt_e2) > 1e-12, gt_e2, 1e-12)
    rmse_e2 = float(np.sqrt(np.mean(e2_res ** 2)))
    rel_rmse_e2 = float(100.0 * np.sqrt(np.mean(e2_res_rel ** 2)))
    std_rmse_e2 = float(np.std(e2_res))
    std_rel_rmse_e2 = float(100.0 * np.std(e2_res_rel))

    # R2 metrics
    R2_res = gt_R2 - pred_R2
    mean_gt_R2 = np.mean(gt_R2) if len(gt_R2) > 0 else 1.0
    rmse_R2_meanR2 = float(np.sqrt(np.mean(R2_res ** 2)) / mean_gt_R2)
    std_rmse_R2_meanR2 = float(np.std(R2_res / gt_R2))

    logger.info("sigma(e1) RMSE =\t %.4e \t +/- %.4e", rmse_e1, std_rmse_e1)
    logger.info("sigma(e2) RMSE =\t %.4e \t +/- %.4e", rmse_e2, std_rmse_e2)
    logger.info("sigma(R2)/<R2> =\t %.4e \t +/- %.4e", rmse_R2_meanR2, std_rmse_R2_meanR2)

    n_total = len(gt_moments)
    n_good = len(gt_e1)
    logger.info("Total stars: %d, Problematic: %d", n_total, n_total - n_good)

    return {
        "pred_e1_HSM": pred_e1,
        "pred_e2_HSM": pred_e2,
        "pred_R2_HSM": pred_R2,
        "gt_pred_e1_HSM": gt_e1,
        "gt_pred_e2_HSM": gt_e2,
        "gt_pred_R2_HSM": gt_R2,
        "rmse_e1": rmse_e1,
        "std_rmse_e1": std_rmse_e1,
        "rel_rmse_e1": rel_rmse_e1,
        "std_rel_rmse_e1": std_rel_rmse_e1,
        "rmse_e2": rmse_e2,
        "std_rmse_e2": std_rmse_e2,
        "rel_rmse_e2": rel_rmse_e2,
        "std_rel_rmse_e2": std_rel_rmse_e2,
        "rmse_R2_meanR2": rmse_R2_meanR2,
        "std_rmse_R2_meanR2": std_rmse_R2_meanR2,
        "pix_rmse": pix_rmse,
        "pix_rmse_std": pix_rmse_std,
        "rel_pix_rmse": rel_pix_rmse,
        "rel_pix_rmse_std": rel_pix_rmse_std,
    }
