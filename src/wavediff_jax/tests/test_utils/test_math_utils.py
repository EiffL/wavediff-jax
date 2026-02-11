"""Tests for utils/math_utils.py"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from wavediff_jax.utils.math_utils import (
    calc_poly_position_mat,
    generate_zernike_maps_3d,
    obscurations_from_params,
    NoiseEstimator,
    generalised_sigmoid,
    decompose_obscured_opd_basis,
)


class TestCalcPolyPositionMat:
    def test_shape_d1(self):
        pos = jnp.array([[0.5, 0.5]])
        result = calc_poly_position_mat(pos, [0, 1], [0, 1], d_max=1)
        # d_max=1: n_poly = 3 (1, x, y)
        assert result.shape == (3, 1)

    def test_shape_d3(self):
        pos = jnp.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        result = calc_poly_position_mat(pos, [0, 1], [0, 1], d_max=3)
        # d_max=3: n_poly = (3+1)*(3+2)/2 = 10
        assert result.shape == (10, 3)

    def test_constant_term_is_one(self):
        """The d=0 polynomial term should be x^0 * y^0 = 1 for all positions."""
        pos = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = calc_poly_position_mat(pos, [0, 1], [0, 1], d_max=2)
        assert jnp.allclose(result[0, :], 1.0)

    def test_center_gives_zero_linear_terms(self):
        """Center of the field should produce zero for all linear terms."""
        pos = jnp.array([[500.0, 500.0]])
        result = calc_poly_position_mat(pos, [0, 1000], [0, 1000], d_max=2)
        # Terms at d=1: x, y -- should both be 0 at center
        assert jnp.allclose(result[1, 0], 0.0, atol=1e-6)
        assert jnp.allclose(result[2, 0], 0.0, atol=1e-6)

    def test_corner_positions(self):
        """Corners should map to (-1,-1), (-1,1), (1,-1), (1,1)."""
        pos = jnp.array([
            [0.0, 0.0],
            [0.0, 1000.0],
            [1000.0, 0.0],
            [1000.0, 1000.0],
        ])
        result = calc_poly_position_mat(pos, [0, 1000], [0, 1000], d_max=1)
        # Index 1 is x term, index 2 is y term
        assert jnp.allclose(result[1, 0], -1.0, atol=1e-5)  # x at (0, *)
        assert jnp.allclose(result[1, 3], 1.0, atol=1e-5)   # x at (1000, *)
        assert jnp.allclose(result[2, 0], -1.0, atol=1e-5)   # y at (*, 0)
        assert jnp.allclose(result[2, 1], 1.0, atol=1e-5)    # y at (*, 1000)

    def test_jit_compatible(self):
        pos = jnp.array([[500.0, 500.0]])
        jitted_fn = jax.jit(lambda p: calc_poly_position_mat(p, [0, 1e3], [0, 1e3], d_max=2))
        result = jitted_fn(pos)
        assert result.shape == (6, 1)


class TestGenerateZernikeMaps3D:
    def test_shape(self):
        result = generate_zernike_maps_3d(n_zernikes=5, pupil_diam=32)
        assert result.shape == (5, 32, 32)

    def test_no_nans(self):
        result = generate_zernike_maps_3d(n_zernikes=5, pupil_diam=32)
        assert not jnp.any(jnp.isnan(result))

    def test_dtype(self):
        result = generate_zernike_maps_3d(n_zernikes=3, pupil_diam=16)
        assert result.dtype == jnp.float32


class TestObscurationsFromParams:
    def test_shape(self):
        result = obscurations_from_params(pupil_diam=64, N_filter=2)
        assert result.shape == (64, 64)

    def test_dtype(self):
        result = obscurations_from_params(pupil_diam=64)
        assert result.dtype == jnp.complex64


class TestNoiseEstimator:
    def test_init(self):
        ne = NoiseEstimator(img_dim=(64, 64), win_rad=10)
        assert ne.window.shape == (64, 64)

    def test_sigma_mad_constant(self):
        """MAD of a constant array should be 0."""
        x = np.ones(100)
        assert NoiseEstimator.sigma_mad(x) == 0.0

    def test_estimate_noise(self):
        ne = NoiseEstimator(img_dim=(64, 64), win_rad=10)
        rng = np.random.default_rng(42)
        image = rng.normal(0, 1, (64, 64))
        sigma = ne.estimate_noise(image)
        # MAD estimate of sigma for Gaussian should be close to 1
        assert 0.5 < sigma < 1.5


class TestGeneralisedSigmoid:
    def test_zero_input(self):
        assert generalised_sigmoid(0.0) == 0.0

    def test_positive_input(self):
        result = generalised_sigmoid(1.0, max_val=1, power_k=1)
        assert 0.0 < result <= 1.0


class TestDecomposeObscuredOpdBasis:
    def test_raises_on_too_many_zernikes(self):
        opd = jnp.zeros((32, 32))
        obsc = jnp.ones((32, 32), dtype=jnp.complex64)
        zk_basis = jnp.ones((5, 32, 32))
        with pytest.raises(ValueError, match="n_zernike exceeds"):
            decompose_obscured_opd_basis(opd, obsc, zk_basis, n_zernike=10)

    def test_zero_opd(self):
        """Zero OPD should decompose to zero coefficients."""
        opd = jnp.zeros((32, 32))
        obsc = jnp.ones((32, 32), dtype=jnp.complex64)
        zk_basis = generate_zernike_maps_3d(5, 32)
        coeffs = decompose_obscured_opd_basis(opd, obsc, zk_basis, n_zernike=5, iters=5)
        assert np.allclose(coeffs, 0.0, atol=1e-6)
