"""Training utility functions.

Provides sample weight calculation and optimizer configuration for the
WaveDiff-JAX training loop.

:Authors: WaveDiff-JAX contributors
"""

import numpy as np
import optax
import logging

from wavediff_jax.utils.math_utils import NoiseEstimator, generalised_sigmoid

logger = logging.getLogger(__name__)


def calculate_sample_weights(
    outputs,
    use_sample_weights=True,
    masked=False,
    apply_sigmoid=False,
    sigmoid_max_val=5.0,
    sigmoid_power_k=1.0,
):
    """Calculate per-sample weights based on noise estimation.

    Weights are computed as inverse variance (estimated via MAD) of each image,
    normalized by the median weight.

    Parameters
    ----------
    outputs : np.ndarray
        Image array. If ``masked`` is True, expected shape is
        ``(batch, H, W, 2)`` where ``[..., 0]`` are images and ``[..., 1]``
        are masks. Otherwise ``(batch, H, W)``.
    use_sample_weights : bool
        If False, return None immediately.
    masked : bool
        Whether the outputs contain a mask channel (axis -1).
    apply_sigmoid : bool
        Whether to apply a generalized sigmoid to the weights.
    sigmoid_max_val : float
        Maximum value for the sigmoid function.
    sigmoid_power_k : float
        Power parameter for the sigmoid function.

    Returns
    -------
    np.ndarray or None
        Sample weights of shape ``(batch,)`` or None.
    """
    if not use_sample_weights:
        return None

    if masked:
        images = outputs[..., 0]
        masks_bool = np.array(1 - outputs[..., 1], dtype=bool)
    else:
        images = outputs
        masks_bool = None

    img_dim = (images.shape[1], images.shape[2])
    win_rad = np.ceil(images.shape[1] / 3.33)
    std_est = NoiseEstimator(img_dim=img_dim, win_rad=win_rad)

    if masks_bool is not None:
        logger.info("Estimating noise standard deviation for masked images..")
        imgs_std = np.array(
            [
                std_est.estimate_noise(_im, _win)
                for _im, _win in zip(images, masks_bool)
            ]
        )
    else:
        logger.info("Estimating noise standard deviation for images..")
        imgs_std = np.array([std_est.estimate_noise(_im) for _im in images])

    # Inverse variance weighting, normalized by median
    variances = imgs_std ** 2
    sample_weight = 1.0 / variances
    sample_weight /= np.median(sample_weight)

    if apply_sigmoid:
        sample_weight = generalised_sigmoid(
            sample_weight, max_val=sigmoid_max_val, power_k=sigmoid_power_k
        )

    return sample_weight


def configure_optimizer(lr, b1=0.9, b2=0.999, eps=1e-7):
    """Return an optax Adam optimizer matching the TF reference defaults.

    Parameters
    ----------
    lr : float
        Learning rate.
    b1 : float
        Beta1 parameter.
    b2 : float
        Beta2 parameter.
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    optax.GradientTransformation
        Configured Adam optimizer.
    """
    return optax.adam(lr, b1=b1, b2=b2, eps=eps)
