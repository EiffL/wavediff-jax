"""Loss functions for PSF model training.

Port of loss functions from ``extern/wf-psf/src/wf_psf/training/train_utils.py``
to JAX. Implements masked MSE, OPD regularization, and Lp sparsity penalties.

:Authors: WaveDiff-JAX contributors
"""

import jax.numpy as jnp


def mse_loss(predictions, targets):
    """Mean squared error loss.

    Parameters
    ----------
    predictions : jnp.ndarray
        Predicted values.
    targets : jnp.ndarray
        Target values (same shape as predictions).

    Returns
    -------
    jnp.ndarray
        Scalar MSE value.
    """
    return jnp.mean((predictions - targets) ** 2)


def masked_mse_loss(predictions, targets, masks):
    """Masked mean squared error loss.

    Follows the TF convention where mask values of 0 mean *include* the pixel
    and 1 means *exclude* the pixel. Float values in (0, 1) provide partial
    weighting.

    The per-sample loss is normalized by the number of included pixels in each
    sample, then averaged over the batch.

    Parameters
    ----------
    predictions : jnp.ndarray, shape (batch, height, width)
        Predicted values.
    targets : jnp.ndarray, shape (batch, height, width)
        Target values.
    masks : jnp.ndarray, shape (batch, height, width)
        Mask array in [0, 1] where 0 = include, 1 = exclude.

    Returns
    -------
    jnp.ndarray
        Scalar masked MSE value.
    """
    keep = 1.0 - masks
    diff_sq = (predictions - targets) ** 2
    error = diff_sq * keep  # (batch, H, W)

    # Sum of keep per sample for normalization
    mask_sum = jnp.sum(keep, axis=(1, 2))  # (batch,)
    # Normalize each sample by its mask_sum, then average over batch
    per_sample = jnp.sum(error, axis=(1, 2)) / mask_sum  # (batch,)
    return jnp.mean(per_sample)


def weighted_masked_mse_loss(predictions, targets, masks, sample_weight=None):
    """Weighted masked mean squared error loss.

    Extends :func:`masked_mse_loss` with optional per-sample weighting.

    The implementation matches the TF reference: the squared error is multiplied
    by the inverted mask and (optionally) sample weights, then normalized by the
    mask sum per sample and averaged over the batch.

    Parameters
    ----------
    predictions : jnp.ndarray, shape (batch, height, width)
    targets : jnp.ndarray, shape (batch, height, width)
    masks : jnp.ndarray, shape (batch, height, width)
        Mask in [0, 1]. 0 = include, 1 = exclude.
    sample_weight : jnp.ndarray or None, shape (batch,)
        Per-sample weights. If ``None``, all samples are weighted equally.

    Returns
    -------
    jnp.ndarray
        Scalar loss.
    """
    keep = 1.0 - masks
    diff_sq = (predictions - targets) ** 2
    error = diff_sq * keep  # (batch, H, W)

    if sample_weight is not None:
        # Broadcast (batch,) -> (batch, 1, 1)
        error = error * sample_weight[:, None, None]

    # Normalize per sample by mask_sum, then average over batch
    mask_sum = jnp.sum(keep, axis=(1, 2))  # (batch,)
    per_sample = jnp.sum(error, axis=(1, 2)) / mask_sum  # (batch,)
    return jnp.mean(per_sample)


def l2_opd_regularization(opd_maps, l2_param):
    """L2 regularization on OPD maps.

    Parameters
    ----------
    opd_maps : jnp.ndarray
        OPD maps of arbitrary shape.
    l2_param : float
        Regularization strength.

    Returns
    -------
    jnp.ndarray
        Scalar L2 penalty.
    """
    return l2_param * jnp.sum(opd_maps ** 2)


def lp_regularization(alpha, p=1.1):
    """Lp regularization for sparsity (typically on alpha_graph).

    Parameters
    ----------
    alpha : jnp.ndarray
        Array to regularize.
    p : float
        Exponent for the Lp norm (default 1.1).

    Returns
    -------
    jnp.ndarray
        Scalar Lp penalty.
    """
    return jnp.sum(jnp.abs(alpha) ** p)


def total_loss(
    model,
    positions,
    packed_seds,
    targets,
    masks,
    sample_weight,
    l2_param=0.0,
    l1_rate=0.0,
):
    """Full training loss: data loss + OPD regularization + optional Lp.

    Runs a forward pass through the model and computes the combined loss.

    Parameters
    ----------
    model : eqx.Module
        PSF model that returns ``(psf_batch, opd_maps)`` or
        ``(psf_batch, opd_maps, aux)`` from ``model([positions, packed_seds])``.
    positions : jnp.ndarray, shape (batch, 2)
    packed_seds : jnp.ndarray, shape (batch, n_wavelengths, 3)
    targets : jnp.ndarray, shape (batch, output_dim, output_dim)
    masks : jnp.ndarray, shape (batch, output_dim, output_dim)
    sample_weight : jnp.ndarray or None, shape (batch,)
    l2_param : float
        L2 regularization strength on OPD maps.
    l1_rate : float
        Lp regularization rate on alpha_graph (if present).

    Returns
    -------
    jnp.ndarray
        Scalar total loss.
    """
    outputs = model([positions, packed_seds])
    if len(outputs) == 3:
        psf_batch, opd_maps, aux = outputs
    else:
        psf_batch, opd_maps = outputs
        aux = None

    # Data loss
    data_loss = weighted_masked_mse_loss(psf_batch, targets, masks, sample_weight)

    # L2 regularization on OPD
    reg_loss = l2_opd_regularization(opd_maps, l2_param)

    # Lp on alpha_graph if present
    if aux is not None and "alpha_graph" in aux and l1_rate > 0:
        reg_loss = reg_loss + l1_rate * lp_regularization(aux["alpha_graph"])

    return data_loss + reg_loss
