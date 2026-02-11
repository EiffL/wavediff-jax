"""Euclid-like pupil obscuration mask generation.

Ported from
``extern/wf-psf/src/wf_psf/sims/psf_simulator.py``
(``PSFSimulator.generate_euclid_pupil_obscurations``).

:Authors: WaveDiff-JAX contributors
"""

import numpy as np
from scipy.signal import convolve2d


def generate_euclid_pupil_obscurations(N_pix=1024, N_filter=None, rotation_angle=0.0):
    """Generate a Euclid-like pupil obscuration mask.

    The mask models the aperture stop, the M1/M2 central obscuration, and
    three spider arms of the Euclid telescope.  An optional low-pass
    (top-hat) filter can be applied to smooth out aliasing artefacts.

    Parameters
    ----------
    N_pix : int
        Pixel dimension of the output square mask.
    N_filter : int or None, optional
        Size of the low-pass filter kernel.  If ``None``, no filtering is
        applied.
    rotation_angle : float, optional
        Spider rotation angle in degrees.  Only multiples of 90 are
        supported (0, 90, 180, 270).

    Returns
    -------
    np.ndarray
        Pupil mask of shape ``(N_pix, N_pix)`` with values in ``[0, 1]``.
    """
    # -----------------------------------------------------------------
    # Telescope parameters (all lengths in mm)
    # -----------------------------------------------------------------
    AS_diam = 1200        # Aperture stop diameter
    M1_diam = 395         # Mirror-1 cap stopper diameter

    sp_length = 700       # Spider arm length
    sp_width = 12         # Spider arm width

    AS_centre = [0, 0]
    M1_centre = [0, 51]

    sp1_angle = 106.78 - 90  # degrees
    sp2_angle = 50.11 - 90
    sp3_angle = -10.76 - 90

    sp1_x_pos, sp1_y_pos = 260, 240
    sp2_x_pos, sp2_y_pos = -330, 130
    sp3_x_pos, sp3_y_pos = 70, -330

    # -----------------------------------------------------------------
    # Build pupil plane
    # -----------------------------------------------------------------
    pupil_plane = np.ones((N_pix, N_pix))

    # Coordinate grid in mm
    W, H = np.meshgrid(
        np.linspace(-AS_diam // 2, AS_diam // 2, N_pix),
        np.linspace(-AS_diam // 2, AS_diam // 2, N_pix),
    )

    # --- Aperture stop ---
    aperture_stop_mask = np.sqrt(
        (W - AS_centre[0]) ** 2 + (H - AS_centre[1]) ** 2
    ) <= (AS_diam / 2)
    pupil_plane[~aperture_stop_mask] = 0

    # --- M1/M2 central obscuration ---
    M1_mask = np.sqrt(
        (W - M1_centre[0]) ** 2 + (H - M1_centre[1]) ** 2
    ) <= (M1_diam / 2)
    pupil_plane[M1_mask] = 0

    # --- Spider arms ---
    def _spider_mask(angle_deg, x_pos, y_pos):
        a = np.tan(np.deg2rad(angle_deg))
        b = y_pos - a * x_pos
        half_w = sp_width / 2 * np.sqrt(1 + a ** 2)

        mask_lo = (a * W + b - half_w) < H
        mask_hi = (a * W + b + half_w) > H
        strip = np.logical_and(mask_lo, mask_hi)

        length_mask = np.sqrt((W - x_pos) ** 2 + (H - y_pos) ** 2) <= (sp_length / 2)
        return np.logical_and(strip, length_mask)

    pupil_plane[_spider_mask(sp1_angle, sp1_x_pos, sp1_y_pos)] = 0
    pupil_plane[_spider_mask(sp2_angle, sp2_x_pos, sp2_y_pos)] = 0
    pupil_plane[_spider_mask(sp3_angle, sp3_x_pos, sp3_y_pos)] = 0

    # --- Low-pass filter ---
    if N_filter is not None and N_filter > 0:
        top_hat = np.ones((N_filter, N_filter))
        pupil_plane = convolve2d(
            pupil_plane, top_hat, boundary='fill', mode='same', fillvalue=0
        )
        pupil_plane /= np.sum(top_hat)

    # --- Rotation (multiples of 90 degrees only) ---
    k = int((rotation_angle // 90) % 4)
    pupil_plane = np.rot90(pupil_plane, k=k)

    return pupil_plane
