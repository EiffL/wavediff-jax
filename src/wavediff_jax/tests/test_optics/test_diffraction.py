"""Tests for the FFT diffraction pipeline."""

import numpy as np
import numpy.testing as npt
import jax.numpy as jnp
import pytest

from wavediff_jax.optics.diffraction import (
    fft_diffract,
    build_phase,
    zernike_to_opd,
    monochromatic_psf,
)
from wavediff_jax.optics.zernike import zernike_generator


# ------------------------------------------------------------------ #
# fft_diffract
# ------------------------------------------------------------------ #

class TestFFTDiffract:
    """Tests for fft_diffract."""

    def test_output_shape(self):
        """Output should match (output_dim, output_dim)."""
        phase_N = 256
        output_dim = 64
        output_Q = 1
        phase = jnp.ones((phase_N, phase_N), dtype=jnp.complex128)
        psf = fft_diffract(phase, output_dim, output_Q)
        assert psf.shape == (output_dim, output_dim)

    def test_output_shape_with_Q(self):
        """Output shape should still be (output_dim, output_dim) when Q > 1."""
        phase_N = 512
        output_dim = 64
        output_Q = 2
        phase = jnp.ones((phase_N, phase_N), dtype=jnp.complex128)
        psf = fft_diffract(phase, output_dim, output_Q)
        assert psf.shape == (output_dim, output_dim)

    def test_energy_conservation(self):
        """PSF should sum to 1 (normalised)."""
        phase_N = 256
        output_dim = 64
        output_Q = 1
        phase = jnp.ones((phase_N, phase_N), dtype=jnp.complex128)
        psf = fft_diffract(phase, output_dim, output_Q)
        npt.assert_allclose(float(jnp.sum(psf)), 1.0, atol=1e-6)

    def test_symmetric_psf_from_circular_pupil(self):
        """A centred circular pupil should give an approximately centrosymmetric PSF."""
        # Use an odd phase_N so there is a true centre pixel
        phase_N = 255
        output_dim = 63
        output_Q = 1

        # Create a centred circular aperture on an odd grid (has a true centre)
        x = np.linspace(-1.0, 1.0, phase_N)
        xv, yv = np.meshgrid(x, x)
        rho = np.sqrt(xv ** 2 + yv ** 2)
        pupil = np.where(rho <= 0.8, 1.0, 0.0)
        phase = jnp.array(pupil, dtype=jnp.complex128)

        psf = fft_diffract(phase, output_dim, output_Q)
        psf_np = np.array(psf)

        # Should be symmetric under 180-degree rotation (centrosymmetric)
        npt.assert_allclose(psf_np, psf_np[::-1, ::-1], atol=1e-8)


# ------------------------------------------------------------------ #
# build_phase
# ------------------------------------------------------------------ #

class TestBuildPhase:
    """Tests for build_phase."""

    def test_output_shape(self):
        """Padded output should be (phase_N, phase_N)."""
        wfe_dim = 64
        phase_N = 256
        opd = jnp.zeros((wfe_dim, wfe_dim))
        obsc = jnp.ones((wfe_dim, wfe_dim))
        phase = build_phase(opd, 0.8, obsc, phase_N)
        assert phase.shape == (phase_N, phase_N)

    def test_zero_opd_gives_real_phase(self):
        """Zero OPD should give exp(0) = 1 inside the pupil."""
        wfe_dim = 64
        phase_N = 128
        opd = jnp.zeros((wfe_dim, wfe_dim))
        obsc = jnp.ones((wfe_dim, wfe_dim))
        phase = build_phase(opd, 0.8, obsc, phase_N)

        # Centre region should be all ones
        pad = (phase_N - wfe_dim) // 2
        centre = phase[pad:pad + wfe_dim, pad:pad + wfe_dim]
        npt.assert_allclose(np.abs(np.array(centre)), 1.0, atol=1e-12)

    def test_padding_is_zero(self):
        """Padded region should be zero."""
        wfe_dim = 64
        phase_N = 128
        opd = jnp.ones((wfe_dim, wfe_dim))
        obsc = jnp.ones((wfe_dim, wfe_dim))
        phase = build_phase(opd, 0.8, obsc, phase_N)

        pad = (phase_N - wfe_dim) // 2
        # Top padding strip
        npt.assert_allclose(
            np.array(phase[:pad, :]), 0.0, atol=1e-15
        )


# ------------------------------------------------------------------ #
# zernike_to_opd
# ------------------------------------------------------------------ #

class TestZernikeToOPD:
    """Tests for zernike_to_opd."""

    def test_basic(self):
        """Single coefficient should reproduce that Zernike map."""
        n_zk = 4
        wfe_dim = 64
        maps_np = zernike_generator(n_zk, wfe_dim)
        # Replace NaN with 0 for JAX
        maps_jnp = jnp.array(
            [np.nan_to_num(m, nan=0.0) for m in maps_np]
        )
        coeffs = jnp.array([0.0, 0.0, 1.0, 0.0])
        opd = zernike_to_opd(coeffs, maps_jnp)
        npt.assert_allclose(
            np.array(opd), np.nan_to_num(maps_np[2], nan=0.0), atol=1e-10
        )

    def test_shape_3d_coeffs(self):
        """Coefficients with shape (n_zk, 1, 1) should also work."""
        n_zk = 3
        wfe_dim = 32
        maps_np = zernike_generator(n_zk, wfe_dim)
        maps_jnp = jnp.array(
            [np.nan_to_num(m, nan=0.0) for m in maps_np]
        )
        coeffs = jnp.array([1.0, 2.0, 3.0])[:, None, None]
        opd = zernike_to_opd(coeffs, maps_jnp)
        assert opd.shape == (wfe_dim, wfe_dim)


# ------------------------------------------------------------------ #
# monochromatic_psf end-to-end
# ------------------------------------------------------------------ #

class TestMonochromaticPSF:
    """End-to-end test: OPD -> PSF."""

    def test_end_to_end(self):
        """The full pipeline should produce a normalised PSF."""
        wfe_dim = 64
        phase_N = 256
        output_dim = 32
        output_Q = 1
        lambda_obs = 0.8

        opd = jnp.zeros((wfe_dim, wfe_dim))
        obsc = jnp.ones((wfe_dim, wfe_dim))

        psf = monochromatic_psf(opd, lambda_obs, phase_N, obsc, output_dim, output_Q)
        assert psf.shape == (output_dim, output_dim)
        npt.assert_allclose(float(jnp.sum(psf)), 1.0, atol=1e-6)

    def test_nonzero_opd(self):
        """A non-zero OPD should still produce a valid normalised PSF."""
        wfe_dim = 64
        phase_N = 256
        output_dim = 32
        output_Q = 1
        lambda_obs = 0.8

        # Use defocus-like OPD
        x = np.linspace(-1.0, 1.0, wfe_dim)
        xv, yv = np.meshgrid(x, x)
        rho = np.sqrt(xv ** 2 + yv ** 2)
        opd = jnp.array(0.01 * (2 * rho ** 2 - 1))
        obsc = jnp.ones((wfe_dim, wfe_dim))

        psf = monochromatic_psf(opd, lambda_obs, phase_N, obsc, output_dim, output_Q)
        assert psf.shape == (output_dim, output_dim)
        npt.assert_allclose(float(jnp.sum(psf)), 1.0, atol=1e-6)
        assert float(jnp.max(psf)) > 0
