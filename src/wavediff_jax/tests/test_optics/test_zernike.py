"""Tests for Zernike polynomial generation (Noll indexing)."""

import numpy as np
import numpy.testing as npt
import pytest

from wavediff_jax.optics.zernike import noll_to_nm, zernike_generator


# ------------------------------------------------------------------ #
# noll_to_nm mapping
# ------------------------------------------------------------------ #

# Expected (n, m) for the first 15 Noll indices
NOLL_TABLE = {
    1:  (0,  0),
    2:  (1,  1),
    3:  (1, -1),
    4:  (2,  0),
    5:  (2, -2),
    6:  (2,  2),
    7:  (3, -1),
    8:  (3,  1),
    9:  (3, -3),
    10: (3,  3),
    11: (4,  0),
    12: (4,  2),
    13: (4, -2),
    14: (4,  4),
    15: (4, -4),
}


@pytest.mark.parametrize("j, expected", list(NOLL_TABLE.items()))
def test_noll_to_nm(j, expected):
    """Noll index j should map to the expected (n, m)."""
    assert noll_to_nm(j) == expected, (
        f"noll_to_nm({j}) = {noll_to_nm(j)}, expected {expected}"
    )


# ------------------------------------------------------------------ #
# Output shape and NaN masking
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("wfe_dim", [64, 128])
def test_output_shape(wfe_dim):
    """Each Zernike map should have shape (wfe_dim, wfe_dim)."""
    n_zk = 5
    maps = zernike_generator(n_zk, wfe_dim)
    assert len(maps) == n_zk
    for Z in maps:
        assert Z.shape == (wfe_dim, wfe_dim)


def test_nan_outside_unit_circle():
    """Values outside the unit disk should be NaN."""
    wfe_dim = 128
    maps = zernike_generator(3, wfe_dim)
    x = np.linspace(-1.0, 1.0, wfe_dim)
    xv, yv = np.meshgrid(x, x)
    rho = np.sqrt(xv ** 2 + yv ** 2)

    for Z in maps:
        # Pixels strictly outside the unit circle must be NaN
        assert np.all(np.isnan(Z[rho > 1.0]))
        # Pixels strictly inside should be finite
        assert np.all(np.isfinite(Z[rho < 0.99]))


# ------------------------------------------------------------------ #
# Analytical verification of specific modes
# ------------------------------------------------------------------ #

def _polar_grid(wfe_dim):
    """Return (rho, theta, inside_mask) on a [-1,1] grid."""
    x = np.linspace(-1.0, 1.0, wfe_dim)
    xv, yv = np.meshgrid(x, x)
    rho = np.sqrt(xv ** 2 + yv ** 2)
    theta = np.arctan2(yv, xv)
    inside = rho <= 1.0
    return rho, theta, inside


def test_z1_piston():
    """Z1 (piston) should be constant over the unit disk."""
    wfe_dim = 256
    maps = zernike_generator(1, wfe_dim)
    rho, _, inside = _polar_grid(wfe_dim)
    vals = maps[0][inside]
    # All values should be equal (constant)
    npt.assert_allclose(vals, vals[0], atol=1e-12)
    # Z1 = sqrt(1) * 1 = 1  (N = sqrt(n+1) for m=0, n=0)
    npt.assert_allclose(vals[0], 1.0, atol=1e-12)


def test_z2_tilt():
    """Z2 (tip) = 2 * rho * cos(theta)."""
    wfe_dim = 256
    maps = zernike_generator(2, wfe_dim)
    rho, theta, inside = _polar_grid(wfe_dim)
    expected = 2.0 * rho * np.cos(theta)
    npt.assert_allclose(maps[1][inside], expected[inside], atol=1e-10)


def test_z3_tilt():
    """Z3 (tilt) = 2 * rho * sin(theta)."""
    wfe_dim = 256
    maps = zernike_generator(3, wfe_dim)
    rho, theta, inside = _polar_grid(wfe_dim)
    expected = 2.0 * rho * np.sin(theta)
    npt.assert_allclose(maps[2][inside], expected[inside], atol=1e-10)


def test_z4_defocus():
    """Z4 (defocus) = sqrt(3) * (2*rho^2 - 1)."""
    wfe_dim = 256
    maps = zernike_generator(4, wfe_dim)
    rho, _, inside = _polar_grid(wfe_dim)
    expected = np.sqrt(3.0) * (2.0 * rho ** 2 - 1.0)
    npt.assert_allclose(maps[3][inside], expected[inside], atol=1e-10)


# ------------------------------------------------------------------ #
# Orthonormality
# ------------------------------------------------------------------ #

def test_orthonormality():
    r"""Integral of Z_i * Z_j over unit disk should be pi * delta_{ij}.

    We approximate the integral with a discrete sum: each pixel on the
    [-1,1]^2 grid with rho<=1 covers an area dA = (2/N)^2.
    """
    wfe_dim = 512  # Use a finer grid for accuracy
    n_zk = 10
    maps = zernike_generator(n_zk, wfe_dim)

    rho, _, inside = _polar_grid(wfe_dim)
    dA = (2.0 / wfe_dim) ** 2

    for i in range(n_zk):
        for j in range(n_zk):
            Zi = maps[i].copy()
            Zj = maps[j].copy()
            Zi[~inside] = 0.0
            Zj[~inside] = 0.0

            integral = np.nansum(Zi * Zj) * dA

            if i == j:
                npt.assert_allclose(integral, np.pi, atol=0.05,
                                    err_msg=f"Self-integral Z{i+1} failed")
            else:
                npt.assert_allclose(integral, 0.0, atol=0.05,
                                    err_msg=f"Cross-integral Z{i+1}*Z{j+1} failed")
