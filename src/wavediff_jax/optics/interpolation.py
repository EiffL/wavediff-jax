"""Thin-plate RBF spline interpolation in pure JAX.

Replaces ``tfa.image.interpolate_spline`` with a JIT-compatible JAX
implementation of polyharmonic spline interpolation.

:Authors: WaveDiff-JAX contributors
"""

import jax.numpy as jnp


def _polyharmonic_kernel(r, order, d):
    """Evaluate the polyharmonic kernel for the given order and dimension *d*.

    Parameters
    ----------
    r : jnp.ndarray
        Pairwise distance matrix (non-negative).
    order : int
        Polyharmonic order.
    d : int
        Spatial dimension of the data points.

    Returns
    -------
    jnp.ndarray
        Kernel values.
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    r_safe = r + eps

    if order == 1:
        return r_safe
    elif order == 2:
        return r_safe ** 2 * jnp.log(r_safe)
    else:
        k = 2 * order - d
        if k > 0 and k % 2 != 0:
            return r_safe ** k
        else:
            return r_safe ** (2 * order) * jnp.log(r_safe)


def thin_plate_spline_interpolate(
    train_points,
    train_values,
    query_points,
    order=2,
    regularization_weight=0.0,
):
    """Polyharmonic spline interpolation (thin-plate spline).

    Solves the classic augmented linear system to determine the RBF weights
    and polynomial coefficients, then evaluates at the query locations.

    Parameters
    ----------
    train_points : jnp.ndarray
        Training point coordinates, shape ``(n_train, d)``.
    train_values : jnp.ndarray
        Training values, shape ``(n_train, n_outputs)``.
    query_points : jnp.ndarray
        Query point coordinates, shape ``(n_query, d)``.
    order : int, optional
        Polyharmonic order.  ``2`` = thin-plate spline (``r^2 log r``).
    regularization_weight : float, optional
        Regularisation added to the diagonal of the kernel block.

    Returns
    -------
    jnp.ndarray
        Interpolated values at query points, shape ``(n_query, n_outputs)``.
    """
    n_train = train_points.shape[0]
    d = train_points.shape[1]

    # ------------------------------------------------------------------
    # Kernel matrix for training points
    # ------------------------------------------------------------------
    diff = train_points[:, None, :] - train_points[None, :, :]
    r = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-20)
    K = _polyharmonic_kernel(r, order, d)

    # Regularisation
    K = K + regularization_weight * jnp.eye(n_train)

    # ------------------------------------------------------------------
    # Polynomial (affine) terms: [1, x1, x2, ...]
    # ------------------------------------------------------------------
    P = jnp.concatenate([jnp.ones((n_train, 1)), train_points], axis=1)
    n_poly = P.shape[1]

    # ------------------------------------------------------------------
    # Augmented system
    # ------------------------------------------------------------------
    top = jnp.concatenate([K, P], axis=1)
    bottom = jnp.concatenate([P.T, jnp.zeros((n_poly, n_poly))], axis=1)
    A = jnp.concatenate([top, bottom], axis=0)

    # Right-hand side
    rhs = jnp.concatenate(
        [train_values, jnp.zeros((n_poly, train_values.shape[1]))], axis=0
    )

    # Solve
    coeffs = jnp.linalg.solve(A, rhs)

    w = coeffs[:n_train]   # RBF weights
    v = coeffs[n_train:]   # Polynomial weights

    # ------------------------------------------------------------------
    # Evaluate at query points
    # ------------------------------------------------------------------
    diff_q = query_points[:, None, :] - train_points[None, :, :]
    r_q = jnp.sqrt(jnp.sum(diff_q ** 2, axis=-1) + 1e-20)
    K_q = _polyharmonic_kernel(r_q, order, d)

    P_q = jnp.concatenate(
        [jnp.ones((query_points.shape[0], 1)), query_points], axis=1
    )

    return K_q @ w + P_q @ v
