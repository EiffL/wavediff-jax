"""Tests for thin-plate spline interpolation."""

import numpy as np
import numpy.testing as npt
import jax.numpy as jnp
import pytest

from wavediff_jax.optics.interpolation import thin_plate_spline_interpolate


# ------------------------------------------------------------------ #
# Exact recovery of linear (affine) functions
# ------------------------------------------------------------------ #

class TestLinearRecovery:
    """A polyharmonic spline should recover affine functions exactly."""

    def test_1d_linear(self):
        """f(x) = 3x + 2  should be recovered exactly."""
        train_x = jnp.array([[0.0], [1.0], [2.0], [3.0]])
        train_y = 3.0 * train_x + 2.0

        query_x = jnp.array([[0.5], [1.5], [2.5]])
        expected = 3.0 * query_x + 2.0

        result = thin_plate_spline_interpolate(
            train_x, train_y, query_x, order=2,
            regularization_weight=0.0,
        )
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-4)

    def test_2d_affine(self):
        """f(x, y) = 2x - y + 1 should be recovered exactly."""
        rng = np.random.RandomState(42)
        n_train = 20
        pts = rng.uniform(-1, 1, (n_train, 2))
        train_points = jnp.array(pts)
        train_values = jnp.array(2.0 * pts[:, 0:1] - pts[:, 1:2] + 1.0)

        n_query = 10
        qpts = rng.uniform(-1, 1, (n_query, 2))
        query_points = jnp.array(qpts)
        expected = 2.0 * qpts[:, 0:1] - qpts[:, 1:2] + 1.0

        result = thin_plate_spline_interpolate(
            train_points, train_values, query_points, order=2,
            regularization_weight=0.0,
        )
        npt.assert_allclose(np.array(result), expected, atol=1e-3)


# ------------------------------------------------------------------ #
# Known 2-D function
# ------------------------------------------------------------------ #

class TestKnownFunction:
    """Interpolation of a smooth 2-D function should be accurate."""

    def test_smooth_2d(self):
        """Interpolate f(x,y) = sin(x) * cos(y) and check accuracy."""
        rng = np.random.RandomState(123)
        n_train = 50
        pts = rng.uniform(-2, 2, (n_train, 2))

        def f(xy):
            return np.sin(xy[:, 0:1]) * np.cos(xy[:, 1:2])

        train_points = jnp.array(pts)
        train_values = jnp.array(f(pts))

        n_query = 15
        qpts = rng.uniform(-1.5, 1.5, (n_query, 2))
        query_points = jnp.array(qpts)
        expected = f(qpts)

        result = thin_plate_spline_interpolate(
            train_points, train_values, query_points, order=2,
            regularization_weight=1e-4,
        )
        # Tolerance is looser for a non-linear function
        npt.assert_allclose(np.array(result), expected, atol=0.15)


# ------------------------------------------------------------------ #
# Comparison with scipy.interpolate.RBFInterpolator
# ------------------------------------------------------------------ #

class TestAgainstScipy:
    """Validate against scipy.interpolate.RBFInterpolator."""

    def test_vs_scipy(self):
        """Results should match scipy's RBFInterpolator within tolerance."""
        from scipy.interpolate import RBFInterpolator

        rng = np.random.RandomState(7)
        n_train = 30
        pts = rng.uniform(-1, 1, (n_train, 2))
        vals = np.sin(pts[:, 0:1]) + np.cos(pts[:, 1:2])

        n_query = 12
        qpts = rng.uniform(-0.8, 0.8, (n_query, 2))

        # scipy reference
        rbf = RBFInterpolator(pts, vals, kernel='thin_plate_spline', degree=1)
        scipy_result = rbf(qpts)

        # our implementation
        jax_result = thin_plate_spline_interpolate(
            jnp.array(pts), jnp.array(vals), jnp.array(qpts),
            order=2, regularization_weight=0.0,
        )

        npt.assert_allclose(np.array(jax_result), scipy_result, atol=1e-4)
