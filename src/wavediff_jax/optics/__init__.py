"""Optics primitives for WaveDiff-JAX."""

from wavediff_jax.optics.zernike import noll_to_nm, zernike_radial, zernike_generator
from wavediff_jax.optics.diffraction import (
    fft_diffract,
    build_phase,
    zernike_to_opd,
    monochromatic_psf,
    polychromatic_psf,
)
from wavediff_jax.optics.obscurations import generate_euclid_pupil_obscurations
from wavediff_jax.optics.interpolation import thin_plate_spline_interpolate
