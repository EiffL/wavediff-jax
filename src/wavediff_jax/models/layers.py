"""Equinox layers for PSF field modelling."""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from wavediff_jax.utils.math_utils import calc_poly_position_mat
from wavediff_jax.optics.diffraction import build_phase, fft_diffract, monochromatic_psf
from wavediff_jax.optics.interpolation import thin_plate_spline_interpolate


def _to_numpy(arr):
    """Convert to numpy array for safe use in static fields."""
    if isinstance(arr, jnp.ndarray):
        return np.asarray(arr)
    return np.asarray(arr)


class PolynomialZernikeField(eqx.Module):
    """Calculate Zernike coefficients for given positions using polynomial field model.

    Trainable parameter: coeff_mat of shape (n_zernikes, n_poly).
    """

    coeff_mat: jnp.ndarray  # trainable
    x_lims: list = eqx.field(static=True)
    y_lims: list = eqx.field(static=True)
    n_zernikes: int = eqx.field(static=True)
    d_max: int = eqx.field(static=True)

    def __init__(self, x_lims, y_lims, n_zernikes=45, d_max=2, *, key=None):
        self.x_lims = list(x_lims)
        self.y_lims = list(y_lims)
        self.n_zernikes = n_zernikes
        self.d_max = d_max
        n_poly = int((d_max + 1) * (d_max + 2) / 2)
        if key is None:
            key = jax.random.PRNGKey(0)
        self.coeff_mat = jax.random.uniform(key, (n_zernikes, n_poly), minval=-0.01, maxval=0.01)

    def __call__(self, positions):
        """
        Parameters
        ----------
        positions : jnp.ndarray, shape (batch, 2)

        Returns
        -------
        zernike_coeffs : jnp.ndarray, shape (batch, n_zernikes, 1, 1)
        """
        poly_mat = calc_poly_position_mat(positions, self.x_lims, self.y_lims, self.d_max)
        # poly_mat shape: (n_poly, batch)
        # coeff_mat shape: (n_zernikes, n_poly)
        zernike_coeffs = (self.coeff_mat @ poly_mat).T  # (batch, n_zernikes)
        return zernike_coeffs[:, :, None, None]


class ZernikeOPD(eqx.Module):
    """Convert Zernike coefficients to OPD maps."""

    zernike_maps: jnp.ndarray  # static (n_zernikes, wfe_dim, wfe_dim)

    def __init__(self, zernike_maps):
        self.zernike_maps = zernike_maps

    def __call__(self, z_coeffs):
        """
        Parameters
        ----------
        z_coeffs : jnp.ndarray, shape (batch, n_zernikes, 1, 1)

        Returns
        -------
        opd : jnp.ndarray, shape (batch, wfe_dim, wfe_dim)
        """
        return jnp.sum(self.zernike_maps[None, :, :, :] * z_coeffs, axis=1)


class BatchPolychromaticPSF(eqx.Module):
    """Compute polychromatic PSFs from OPD batch and SED data.

    Uses vmap over wavelengths and batch for parallelism.
    """

    obscurations: jnp.ndarray  # static
    output_Q: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    def __init__(self, obscurations, output_Q, output_dim=64):
        self.obscurations = obscurations
        self.output_Q = int(output_Q)
        self.output_dim = output_dim

    def _single_mono_psf(self, opd, lambda_obs, phase_N, weight):
        """Single monochromatic PSF for one star, one wavelength."""
        phase = build_phase(opd, lambda_obs, self.obscurations, phase_N)
        psf = fft_diffract(phase, self.output_dim, self.output_Q)
        return psf * weight

    def _single_poly_psf(self, opd, packed_sed, max_phase_N):
        """Single polychromatic PSF for one star.

        Parameters
        ----------
        opd : jnp.ndarray, shape (wfe_dim, wfe_dim)
        packed_sed : jnp.ndarray, shape (n_wavelengths, 3) -- [phase_N, lambda_obs, sed_weight]
        max_phase_N : int
            Pre-extracted concrete phase_N (must be extracted outside vmap).
        """
        def mono_fn(packed_elem):
            lambda_obs = packed_elem[1]
            weight = packed_elem[2]
            return self._single_mono_psf(opd, lambda_obs, max_phase_N, weight)

        mono_psfs = jax.vmap(mono_fn)(packed_sed)
        return jnp.sum(mono_psfs, axis=0)

    def __call__(self, inputs):
        """
        Parameters
        ----------
        inputs : list of [opd_batch, packed_SED_data]
            opd_batch: (batch, wfe_dim, wfe_dim)
            packed_SED_data: (batch, n_wavelengths, 3)

        Returns
        -------
        psf_batch : jnp.ndarray, shape (batch, output_dim, output_dim)
        """
        opd_batch, packed_SED_data = inputs
        # Extract max phase_N as a concrete integer before entering vmap.
        # All phase_N values are expected to be the same (padded to max).
        max_phase_N = int(packed_SED_data[0, 0, 0])

        def single_star(opd, packed_sed):
            return self._single_poly_psf(opd, packed_sed, max_phase_N)

        return jax.vmap(single_star)(opd_batch, packed_SED_data)


class BatchMonochromaticPSF(eqx.Module):
    """Compute monochromatic PSFs from a batch of OPDs."""

    obscurations: jnp.ndarray
    output_Q: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    phase_N: int = eqx.field(static=True)
    lambda_obs: float = eqx.field(static=True)

    def __init__(self, obscurations, output_Q, output_dim=64, phase_N=914, lambda_obs=0.7):
        self.obscurations = obscurations
        self.output_Q = int(output_Q)
        self.output_dim = output_dim
        self.phase_N = phase_N
        self.lambda_obs = lambda_obs

    def _single_mono(self, opd):
        return monochromatic_psf(
            opd, self.lambda_obs, self.phase_N, self.obscurations, self.output_dim, self.output_Q
        )

    def __call__(self, opd_batch):
        """
        Parameters
        ----------
        opd_batch : jnp.ndarray, shape (batch, wfe_dim, wfe_dim)

        Returns
        -------
        psf_batch : jnp.ndarray, shape (batch, output_dim, output_dim)
        """
        return jax.vmap(self._single_mono)(opd_batch)


class NonParametricPolynomialOPD(eqx.Module):
    """Non-parametric OPD generation with polynomial variations.

    Trainable parameters: S_mat, alpha_mat.
    """

    S_mat: jnp.ndarray  # trainable (n_poly, opd_dim, opd_dim)
    alpha_mat: jnp.ndarray  # trainable (n_poly, n_poly)
    x_lims: list = eqx.field(static=True)
    y_lims: list = eqx.field(static=True)
    d_max: int = eqx.field(static=True)
    opd_dim: int = eqx.field(static=True)

    def __init__(self, x_lims, y_lims, d_max=3, opd_dim=256, *, key=None):
        self.x_lims = list(x_lims)
        self.y_lims = list(y_lims)
        self.d_max = d_max
        self.opd_dim = opd_dim
        n_poly = int((d_max + 1) * (d_max + 2) / 2)
        if key is None:
            key = jax.random.PRNGKey(0)
        self.S_mat = jax.random.uniform(key, (n_poly, opd_dim, opd_dim), minval=-0.001, maxval=0.001)
        self.alpha_mat = jnp.eye(n_poly)

    def __call__(self, positions):
        """
        Parameters
        ----------
        positions : jnp.ndarray, shape (batch, 2)

        Returns
        -------
        opd_maps : jnp.ndarray, shape (batch, opd_dim, opd_dim)
        """
        poly_mat = calc_poly_position_mat(positions, self.x_lims, self.y_lims, self.d_max)
        # poly_mat: (n_poly, batch)
        poly_mat_T = poly_mat.T  # (batch, n_poly)
        inter_res = poly_mat_T @ self.alpha_mat  # (batch, n_poly)
        return jnp.tensordot(inter_res, self.S_mat, axes=1)  # (batch, opd_dim, opd_dim)


class NonParametricMCCDOPD(eqx.Module):
    """Non-parametric OPD with hybrid MCCD variations.

    Trainable: S_poly, S_graph, alpha_poly, alpha_graph.
    Returns (opd_maps, aux_dict) where aux_dict contains alpha_graph for Lp regularization.
    """

    S_poly: jnp.ndarray
    S_graph: jnp.ndarray
    alpha_poly: jnp.ndarray
    alpha_graph: jnp.ndarray
    obs_pos: np.ndarray = eqx.field(static=True)
    poly_dic: np.ndarray = eqx.field(static=True)
    graph_dic: np.ndarray = eqx.field(static=True)
    x_lims: list = eqx.field(static=True)
    y_lims: list = eqx.field(static=True)
    d_max: int = eqx.field(static=True)
    opd_dim: int = eqx.field(static=True)
    poly_features: int = eqx.field(static=True)
    graph_features: int = eqx.field(static=True)

    def __init__(self, obs_pos, spatial_dic, x_lims, y_lims, d_max=2, graph_features=6, opd_dim=256, *, key=None):
        self.x_lims = list(x_lims)
        self.y_lims = list(y_lims)
        self.d_max = d_max
        self.opd_dim = opd_dim
        self.obs_pos = _to_numpy(obs_pos)
        self.poly_dic = _to_numpy(spatial_dic[0])
        self.graph_dic = _to_numpy(spatial_dic[1])
        self.poly_features = int((d_max + 1) * (d_max + 2) / 2)
        self.graph_features = graph_features
        n_graph_elems = self.graph_dic.shape[1]

        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        self.S_poly = jax.random.uniform(
            k1, (self.poly_features, opd_dim, opd_dim), minval=-0.001, maxval=0.001
        )
        self.S_graph = jax.random.uniform(
            k2, (graph_features, opd_dim, opd_dim), minval=-0.001, maxval=0.001
        )
        self.alpha_poly = jnp.eye(self.poly_features)
        self.alpha_graph = jnp.eye(n_graph_elems, graph_features)

    def predict(self, positions):
        """Prediction with RBF interpolation (for inference)."""
        # Polynomial part
        poly_mat = calc_poly_position_mat(positions, self.x_lims, self.y_lims, self.d_max)
        A_poly = poly_mat.T @ self.alpha_poly
        interp_poly_opd = jnp.tensordot(A_poly, self.S_poly, axes=1)

        # Graph part with RBF interpolation
        A_graph_train = self.graph_dic @ self.alpha_graph
        A_interp_graph = thin_plate_spline_interpolate(
            train_points=self.obs_pos,
            train_values=A_graph_train,
            query_points=positions,
            order=2,
            regularization_weight=0.0,
        )
        interp_graph_opd = jnp.tensordot(A_interp_graph, self.S_graph, axes=1)

        return interp_poly_opd + interp_graph_opd

    def __call__(self, positions):
        """Training forward pass. Uses index lookup instead of interpolation.

        Returns (opd_maps, aux_dict) where aux_dict has 'alpha_graph' for Lp reg.
        """
        # Find indices of positions in obs_pos
        def find_index(pos):
            dists = jnp.sum((self.obs_pos - pos) ** 2, axis=1)
            return jnp.argmin(dists)

        indices = jax.vmap(find_index)(positions)

        # Poly contribution
        batch_poly_dict = self.poly_dic[indices]
        intermediate_poly = batch_poly_dict @ self.alpha_poly
        contribution_poly = jnp.tensordot(intermediate_poly, self.S_poly, axes=1)

        # Graph contribution
        batch_graph_dict = self.graph_dic[indices]
        intermediate_graph = batch_graph_dict @ self.alpha_graph
        contribution_graph = jnp.tensordot(intermediate_graph, self.S_graph, axes=1)

        opd_maps = contribution_poly + contribution_graph
        aux = {"alpha_graph": self.alpha_graph}
        return opd_maps, aux


class NonParametricGraphOPD(eqx.Module):
    """Non-parametric OPD with graph-constraint only.

    Similar to MCCD but without polynomial part.
    """

    S_graph: jnp.ndarray
    alpha_graph: jnp.ndarray
    obs_pos: np.ndarray = eqx.field(static=True)
    poly_dic: np.ndarray = eqx.field(static=True)
    graph_dic: np.ndarray = eqx.field(static=True)
    opd_dim: int = eqx.field(static=True)
    graph_features: int = eqx.field(static=True)

    def __init__(self, obs_pos, spatial_dic, graph_features=6, opd_dim=256, *, key=None):
        self.opd_dim = opd_dim
        self.obs_pos = _to_numpy(obs_pos)
        self.poly_dic = _to_numpy(spatial_dic[0])
        self.graph_dic = _to_numpy(spatial_dic[1])
        self.graph_features = graph_features
        n_graph_elems = self.graph_dic.shape[1]

        if key is None:
            key = jax.random.PRNGKey(0)
        self.S_graph = jax.random.uniform(
            key, (graph_features, opd_dim, opd_dim), minval=-0.001, maxval=0.001
        )
        self.alpha_graph = jnp.eye(n_graph_elems, graph_features)

    def predict(self, positions):
        A_graph_train = self.graph_dic @ self.alpha_graph
        A_interp_graph = thin_plate_spline_interpolate(
            train_points=self.obs_pos,
            train_values=A_graph_train,
            query_points=positions,
            order=2,
            regularization_weight=0.0,
        )
        return jnp.tensordot(A_interp_graph, self.S_graph, axes=1)

    def __call__(self, positions):
        def find_index(pos):
            dists = jnp.sum((self.obs_pos - pos) ** 2, axis=1)
            return jnp.argmin(dists)

        indices = jax.vmap(find_index)(positions)
        batch_graph_dict = self.graph_dic[indices]
        intermediate_graph = batch_graph_dict @ self.alpha_graph
        contribution_graph = jnp.tensordot(intermediate_graph, self.S_graph, axes=1)

        aux = {"alpha_graph": self.alpha_graph}
        return contribution_graph, aux


class PhysicalLayer(eqx.Module):
    """Physical Zernike layer (not trainable).

    Provides prior Zernike coefficients at input positions.
    """

    obs_pos: np.ndarray = eqx.field(static=True)
    zks_prior: np.ndarray = eqx.field(static=True)
    interpolation_type: str = eqx.field(static=True)
    interpolation_args: dict = eqx.field(static=True)

    def __init__(self, obs_pos, zks_prior, interpolation_type=None, interpolation_args=None):
        self.obs_pos = _to_numpy(obs_pos)
        self.zks_prior = _to_numpy(zks_prior)
        self.interpolation_type = interpolation_type or "none"
        self.interpolation_args = interpolation_args or {"order": 2, "K": 50}

    def interpolate_all(self, positions):
        """RBF interpolation using all training points."""
        interp_zks = thin_plate_spline_interpolate(
            train_points=self.obs_pos,
            train_values=self.zks_prior,
            query_points=positions,
            order=self.interpolation_args["order"],
            regularization_weight=0.0,
        )
        return interp_zks[:, :, None, None]

    def __call__(self, positions):
        """Index-based lookup (training mode)."""

        def find_index(pos):
            dists = jnp.sum((self.obs_pos - pos) ** 2, axis=1)
            return jnp.argmin(dists)

        indices = jax.vmap(find_index)(positions)
        batch_zks = self.zks_prior[indices]
        return batch_zks[:, :, None, None]

    def predict(self, positions):
        """Prediction with interpolation."""
        if self.interpolation_type == "all":
            return self.interpolate_all(positions)
        return self.__call__(positions)
