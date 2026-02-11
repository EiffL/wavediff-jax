"""Math utilities for WaveDiff-JAX."""

import jax.numpy as jnp
import numpy as np


def calc_poly_position_mat(pos, x_lims, y_lims, d_max):
    """Calculate matrix of position polynomials.

    Scale positions from [x_lims[0], x_lims[1]] x [y_lims[0], y_lims[1]] to [-1,1] x [-1,1],
    then compute polynomial basis up to degree d_max.

    Parameters
    ----------
    pos : jnp.ndarray
        Positions of shape (batch, 2). First column is x, second is y.
    x_lims : list or array
        [x_min, x_max] limits.
    y_lims : list or array
        [y_min, y_max] limits.
    d_max : int
        Maximum polynomial degree.

    Returns
    -------
    jnp.ndarray
        Polynomial matrix of shape (n_poly, batch), where n_poly = (d_max+1)*(d_max+2)/2.
    """
    scaled_pos_x = (pos[:, 0] - x_lims[0]) / (x_lims[1] - x_lims[0])
    scaled_pos_x = (scaled_pos_x - 0.5) * 2
    scaled_pos_y = (pos[:, 1] - y_lims[0]) / (y_lims[1] - y_lims[0])
    scaled_pos_y = (scaled_pos_y - 0.5) * 2

    poly_list = []
    for d in range(d_max + 1):
        for p in range(d + 1):
            poly_list.append(scaled_pos_x ** (d - p) * scaled_pos_y ** p)

    return jnp.stack(poly_list, axis=0)


def generate_zernike_maps_3d(n_zernikes, pupil_diam):
    """Generate 3D Zernike maps as a JAX array.

    Parameters
    ----------
    n_zernikes : int
    pupil_diam : int

    Returns
    -------
    jnp.ndarray
        Shape (n_zernikes, pupil_diam, pupil_diam), NaN replaced with 0.
    """
    from wavediff_jax.optics.zernike import zernike_generator

    zernikes = zernike_generator(n_zernikes=n_zernikes, wfe_dim=pupil_diam)
    np_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))
    for i in range(len(zernikes)):
        np_cube[i, :, :] = zernikes[i]
    np_cube[np.isnan(np_cube)] = 0
    return jnp.array(np_cube, dtype=jnp.float32)


def obscurations_from_params(pupil_diam, N_filter=2, rotation_angle=0):
    """Generate obscurations as a JAX complex array.

    Parameters
    ----------
    pupil_diam : int
    N_filter : int
    rotation_angle : float

    Returns
    -------
    jnp.ndarray
        Complex obscuration mask of shape (pupil_diam, pupil_diam).
    """
    from wavediff_jax.optics.obscurations import generate_euclid_pupil_obscurations

    obsc = generate_euclid_pupil_obscurations(
        N_pix=pupil_diam, N_filter=N_filter, rotation_angle=rotation_angle
    )
    return jnp.array(obsc, dtype=jnp.complex64)


class NoiseEstimator:
    """Estimate noise levels in images using MAD estimator.

    Port from wf_psf.utils.utils.NoiseEstimator (numpy-only).
    """

    def __init__(self, img_dim, win_rad):
        self.img_dim = img_dim
        self.win_rad = win_rad
        self._init_window()

    def _init_window(self):
        self.window = np.ones(self.img_dim, dtype=bool)
        mid_x = self.img_dim[0] / 2
        mid_y = self.img_dim[1] / 2
        for _x in range(self.img_dim[0]):
            for _y in range(self.img_dim[1]):
                if np.sqrt((_x - mid_x) ** 2 + (_y - mid_y) ** 2) <= self.win_rad:
                    self.window[_x, _y] = False

    def apply_mask(self, mask=None):
        if mask is None:
            return self.window
        return self.window & mask

    @staticmethod
    def sigma_mad(x):
        return 1.4826 * np.median(np.abs(x - np.median(x)))

    def estimate_noise(self, image, mask=None):
        if mask is not None:
            return self.sigma_mad(image[self.apply_mask(mask)])
        return self.sigma_mad(image[self.window])


def generalised_sigmoid(x, max_val=1, power_k=1):
    """Apply a generalized sigmoid function."""
    return max_val * x / np.power(1 + np.power(np.abs(x), power_k), 1 / power_k)


def decompose_obscured_opd_basis(opd, obscurations, zk_basis, n_zernike, iters=20):
    """Decompose obscured OPD into Zernike basis using iterative projection.

    Pure JAX implementation (replaces TF version).

    Parameters
    ----------
    opd : jnp.ndarray, shape (opd_dim, opd_dim)
    obscurations : jnp.ndarray, shape (opd_dim, opd_dim)
    zk_basis : jnp.ndarray, shape (n_basis, opd_dim, opd_dim)
    n_zernike : int
    iters : int

    Returns
    -------
    np.ndarray, shape (n_zernike,)
    """
    if n_zernike > zk_basis.shape[0]:
        raise ValueError("n_zernike exceeds available basis elements.")

    input_opd = jnp.array(opd)
    obsc_real = jnp.real(obscurations)
    ngood = float(jnp.sum(obsc_real))

    obsc_coeffs = np.zeros(n_zernike)

    for _ in range(iters):
        new_coeffs = np.zeros(n_zernike)
        for i in range(n_zernike):
            new_coeffs[i] = float(jnp.sum(input_opd * zk_basis[i])) / ngood

        for i in range(n_zernike):
            input_opd = input_opd - new_coeffs[i] * zk_basis[i] * obsc_real

        obsc_coeffs += new_coeffs

    return obsc_coeffs
