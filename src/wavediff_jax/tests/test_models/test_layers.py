"""Tests for models/layers.py"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest
from wavediff_jax.models.layers import (
    PolynomialZernikeField,
    ZernikeOPD,
    BatchMonochromaticPSF,
    NonParametricPolynomialOPD,
    PhysicalLayer,
)
from wavediff_jax.utils.math_utils import calc_poly_position_mat, generate_zernike_maps_3d


class TestPolynomialZernikeField:
    def test_output_shape(self):
        key = jax.random.PRNGKey(42)
        field = PolynomialZernikeField(
            x_lims=[0, 1e3], y_lims=[0, 1e3], n_zernikes=15, d_max=2, key=key
        )
        positions = jnp.array([[500.0, 500.0], [100.0, 200.0]])
        result = field(positions)
        assert result.shape == (2, 15, 1, 1)

    def test_jit_compatible(self):
        key = jax.random.PRNGKey(42)
        field = PolynomialZernikeField(
            x_lims=[0, 1e3], y_lims=[0, 1e3], n_zernikes=15, d_max=2, key=key
        )
        jitted = eqx.filter_jit(field)
        positions = jnp.array([[500.0, 500.0]])
        result = jitted(positions)
        assert result.shape == (1, 15, 1, 1)


class TestZernikeOPD:
    def test_output_shape(self):
        n_zernikes, wfe_dim = 10, 64
        zernike_maps = generate_zernike_maps_3d(n_zernikes, wfe_dim)
        opd_layer = ZernikeOPD(zernike_maps)
        z_coeffs = jnp.ones((3, n_zernikes, 1, 1))
        result = opd_layer(z_coeffs)
        assert result.shape == (3, wfe_dim, wfe_dim)


class TestNonParametricPolynomialOPD:
    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        layer = NonParametricPolynomialOPD(
            x_lims=[0, 1e3], y_lims=[0, 1e3], d_max=2, opd_dim=32, key=key
        )
        positions = jnp.array([[500.0, 500.0], [100.0, 200.0]])
        result = layer(positions)
        assert result.shape == (2, 32, 32)


class TestPhysicalLayer:
    @pytest.mark.filterwarnings("ignore:A JAX array is being set as static")
    def test_call_output_shape(self):
        obs_pos = np.array([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]])
        zks_prior = np.ones((3, 15))
        layer = PhysicalLayer(obs_pos, zks_prior)
        positions = jnp.array([[100.0, 200.0], [300.0, 400.0]])
        result = layer(positions)
        assert result.shape == (2, 15, 1, 1)


class TestCalcPolyPositionMat:
    def test_shape(self):
        pos = jnp.array([[500.0, 500.0], [100.0, 200.0]])
        result = calc_poly_position_mat(pos, [0, 1e3], [0, 1e3], d_max=2)
        # n_poly for d_max=2 is 6
        assert result.shape == (6, 2)

    def test_center_position(self):
        pos = jnp.array([[500.0, 500.0]])
        result = calc_poly_position_mat(pos, [0, 1e3], [0, 1e3], d_max=1)
        # For center position, scaled coords are (0, 0)
        # d=0: 1; d=1: x=0, y=0
        assert jnp.allclose(result[0, 0], 1.0)  # constant term
        assert jnp.allclose(result[1, 0], 0.0, atol=1e-6)  # x term
        assert jnp.allclose(result[2, 0], 0.0, atol=1e-6)  # y term
