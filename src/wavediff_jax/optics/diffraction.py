"""FFT diffraction pipeline in pure JAX.

Port of the diffraction operations from
``extern/wf-psf/src/wf_psf/psf_models/tf_modules.py`` into JAX, with all
functions designed to be JIT-compatible.

:Authors: WaveDiff-JAX contributors
"""

import jax
import jax.numpy as jnp


def fft_diffract(input_phase, output_dim, output_Q):
    """Diffract a wavefront to a PSF via FFT.

    Parameters
    ----------
    input_phase : jnp.ndarray
        Complex phase array of shape ``(phase_N, phase_N)``.  Single image,
        **not** batched.
    output_dim : int
        Output PSF dimension.
    output_Q : int
        Oversampling / downsampling factor.

    Returns
    -------
    jnp.ndarray
        Normalised PSF of shape ``(output_dim, output_dim)``.
    """
    # FFT and shift
    fft_phase = jnp.fft.fftshift(jnp.fft.fft2(input_phase))
    psf = jnp.abs(fft_phase) ** 2

    # Crop to output_dim * output_Q centred
    crop_dim = output_dim * output_Q
    N = psf.shape[0]
    start = N // 2 - crop_dim // 2
    cropped = jax.lax.dynamic_slice(psf, (start, start), (crop_dim, crop_dim))

    # Down-sample via reshape + mean (average pooling)
    if output_Q > 1:
        cropped = (cropped
                   .reshape(output_dim, output_Q, output_dim, output_Q)
                   .mean(axis=(1, 3)))

    # Normalise so the PSF sums to 1
    return cropped / jnp.sum(cropped)


def build_phase(opd, lambda_obs, obscurations, phase_N):
    """Build a complex phase screen from an OPD map.

    Parameters
    ----------
    opd : jnp.ndarray
        OPD map of shape ``(wfe_dim, wfe_dim)``.
    lambda_obs : float
        Observation wavelength (same units as *opd*).
    obscurations : jnp.ndarray
        Binary (or soft) mask of shape ``(wfe_dim, wfe_dim)``.
    phase_N : int
        Output padded phase dimension.

    Returns
    -------
    jnp.ndarray
        Complex phase of shape ``(phase_N, phase_N)``.
    """
    pre_phase = (2.0 * jnp.pi / lambda_obs) * opd
    phase = jnp.exp(1j * pre_phase)
    phase = phase * obscurations

    # Zero-pad to (phase_N, phase_N)
    wfe_dim = opd.shape[0]
    pad = (phase_N - wfe_dim) // 2
    padded = jnp.pad(phase, pad, mode='constant')
    return padded


def zernike_to_opd(z_coeffs, zernike_maps):
    """Convert Zernike coefficients to an OPD map.

    Parameters
    ----------
    z_coeffs : jnp.ndarray
        Coefficients of shape ``(n_zernikes,)`` or ``(n_zernikes, 1, 1)``.
    zernike_maps : jnp.ndarray
        Zernike maps of shape ``(n_zernikes, wfe_dim, wfe_dim)``.

    Returns
    -------
    jnp.ndarray
        OPD map of shape ``(wfe_dim, wfe_dim)``.
    """
    if z_coeffs.ndim == 1:
        z_coeffs = z_coeffs[:, None, None]
    return jnp.nansum(zernike_maps * z_coeffs, axis=0)


def monochromatic_psf(opd, lambda_obs, phase_N, obscurations, output_dim, output_Q):
    """Full pipeline: OPD -> phase -> PSF for a single wavelength.

    Parameters
    ----------
    opd : jnp.ndarray
        OPD map ``(wfe_dim, wfe_dim)``.
    lambda_obs : float
        Wavelength.
    phase_N : int
        Padded phase dimension.
    obscurations : jnp.ndarray
        Pupil mask.
    output_dim : int
        Output PSF size.
    output_Q : int
        Over-/down-sampling factor.

    Returns
    -------
    jnp.ndarray
        Normalised PSF of shape ``(output_dim, output_dim)``.
    """
    phase = build_phase(opd, lambda_obs, obscurations, phase_N)
    return fft_diffract(phase, output_dim, output_Q)


def polychromatic_psf(opd, packed_sed, obscurations, output_dim, output_Q):
    """Compute a polychromatic PSF by summing monochromatic PSFs weighted by
    the SED.

    Parameters
    ----------
    opd : jnp.ndarray
        OPD map ``(wfe_dim, wfe_dim)``.
    packed_sed : jnp.ndarray
        Array of shape ``(n_wavelengths, 3)`` where each row is
        ``[lambda_obs, phase_N, sed_weight]``.
        **Note:** all ``phase_N`` values must be equal (padded to
        ``max_phase_N``) for static-shape ``vmap`` compatibility.
    obscurations : jnp.ndarray
        Obscuration mask.
    output_dim : int
        Output PSF size.
    output_Q : int
        Over-/down-sampling factor.

    Returns
    -------
    jnp.ndarray
        Polychromatic PSF of shape ``(output_dim, output_dim)``.
    """
    lambdas = packed_sed[:, 0]
    phase_Ns = packed_sed[:, 1]
    weights = packed_sed[:, 2]

    max_phase_N = int(phase_Ns[0])  # Static -- all padded to same value

    def single_mono(lam, weight):
        phase = build_phase(opd, lam, obscurations, max_phase_N)
        psf = fft_diffract(phase, output_dim, output_Q)
        return psf * weight

    mono_psfs = jax.vmap(single_mono)(lambdas, weights)
    return jnp.sum(mono_psfs, axis=0)
