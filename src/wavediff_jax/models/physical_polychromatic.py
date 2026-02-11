"""Physical polychromatic PSF field model.

Combines a physical Zernike prior (from observations) with learned parametric
correction and optional non-parametric OPD component.
Port of ``extern/wf-psf/src/wf_psf/psf_models/psf_model_physical_polychromatic.py``
to JAX/Equinox.

:Authors: WaveDiff-JAX contributors
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import logging

from wavediff_jax.models.registry import register_psfclass, PSFModelBaseFactory
from wavediff_jax.models.layers import (
    PolynomialZernikeField,
    ZernikeOPD,
    BatchPolychromaticPSF,
    BatchMonochromaticPSF,
    NonParametricPolynomialOPD,
    PhysicalLayer,
)

logger = logging.getLogger(__name__)


class PhysicalPolychromaticField(eqx.Module):
    """Physical polychromatic PSF field model.

    Combines physical prior Zernike coefficients with a learned polynomial
    correction and an optional non-parametric OPD component.

    Pipeline:
        positions -> PhysicalLayer           -> z_physical  (prior)
        positions -> PolynomialZernikeField  -> z_correction (learned)
        z_total = pad_and_sum(z_physical, z_correction)
        z_total -> ZernikeOPD -> param_opd
        positions -> NonParametricOPD -> nonparam_opd
        opd_total = param_opd + nonparam_opd
        [opd_total, packed_SED_data] -> BatchPolychromaticPSF -> psf_batch

    Parameters
    ----------
    zernike_maps : jnp.ndarray, shape (n_zks_total, wfe_dim, wfe_dim)
        Precomputed Zernike maps covering the union of prior and parametric sizes.
    obscurations : jnp.ndarray, shape (wfe_dim, wfe_dim)
    obs_pos : array, shape (n_obs, 2)
        Observation positions for the physical layer.
    zks_prior : array, shape (n_obs, n_zks_prior)
        Prior Zernike coefficients at observation positions.
    output_Q : int
    output_dim : int
    n_zernikes_param : int
        Number of Zernike modes for the parametric (learned) part.
    n_zks_total : int
        Total number of Zernike modes (max of prior and parametric).
    d_max : int
    d_max_nonparam : int
    x_lims : list
    y_lims : list
    interpolation_type : str or None
    interpolation_args : dict or None
    key : jax.random.PRNGKey
    """

    physical_layer: PhysicalLayer
    poly_field: PolynomialZernikeField
    zernike_opd: ZernikeOPD
    np_opd: NonParametricPolynomialOPD
    batch_poly_psf: BatchPolychromaticPSF
    n_zernikes_param: int = eqx.field(static=True)
    n_zks_total: int = eqx.field(static=True)
    output_Q: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    def __init__(
        self,
        zernike_maps,
        obscurations,
        obs_pos,
        zks_prior,
        output_Q,
        output_dim=64,
        n_zernikes_param=45,
        n_zks_total=45,
        d_max=2,
        d_max_nonparam=3,
        x_lims=None,
        y_lims=None,
        interpolation_type=None,
        interpolation_args=None,
        *,
        key=None,
    ):
        if x_lims is None:
            x_lims = [0, 1e3]
        if y_lims is None:
            y_lims = [0, 1e3]
        if key is None:
            key = jax.random.PRNGKey(0)

        self.n_zernikes_param = n_zernikes_param
        self.n_zks_total = n_zks_total
        self.output_Q = int(output_Q)
        self.output_dim = output_dim

        k1, k2 = jax.random.split(key)

        # Physical prior layer
        self.physical_layer = PhysicalLayer(
            obs_pos=obs_pos,
            zks_prior=zks_prior,
            interpolation_type=interpolation_type,
            interpolation_args=interpolation_args,
        )

        # Parametric correction
        self.poly_field = PolynomialZernikeField(
            x_lims=x_lims,
            y_lims=y_lims,
            n_zernikes=n_zernikes_param,
            d_max=d_max,
            key=k1,
        )

        # Zernike to OPD (uses the total zernike maps)
        self.zernike_opd = ZernikeOPD(zernike_maps=zernike_maps)

        # Non-parametric OPD
        opd_dim = zernike_maps.shape[1]
        self.np_opd = NonParametricPolynomialOPD(
            x_lims=x_lims,
            y_lims=y_lims,
            d_max=d_max_nonparam,
            opd_dim=opd_dim,
            key=k2,
        )

        # Polychromatic PSF generation
        self.batch_poly_psf = BatchPolychromaticPSF(
            obscurations=obscurations,
            output_Q=output_Q,
            output_dim=output_dim,
        )

    def _pad_zernikes(self, zk_param, zk_prior):
        """Pad Zernike coefficient tensors to match n_zks_total.

        Parameters
        ----------
        zk_param : jnp.ndarray, shape (batch, n_zernikes_param, 1, 1)
        zk_prior : jnp.ndarray, shape (batch, n_zks_prior, 1, 1)

        Returns
        -------
        padded_param : jnp.ndarray, shape (batch, n_zks_total, 1, 1)
        padded_prior : jnp.ndarray, shape (batch, n_zks_total, 1, 1)
        """
        pad_param = self.n_zks_total - zk_param.shape[1]
        pad_prior = self.n_zks_total - zk_prior.shape[1]

        padded_param = zk_param
        if pad_param > 0:
            padded_param = jnp.pad(
                zk_param, ((0, 0), (0, pad_param), (0, 0), (0, 0))
            )

        padded_prior = zk_prior
        if pad_prior > 0:
            padded_prior = jnp.pad(
                zk_prior, ((0, 0), (0, pad_prior), (0, 0), (0, 0))
            )

        return padded_param, padded_prior

    def compute_zernikes(self, positions):
        """Compute combined Zernike coefficients (training: index lookup).

        Parameters
        ----------
        positions : jnp.ndarray, shape (batch, 2)

        Returns
        -------
        zks_coeffs : jnp.ndarray, shape (batch, n_zks_total, 1, 1)
        """
        zk_param = self.poly_field(positions)
        zk_prior = self.physical_layer(positions)
        padded_param, padded_prior = self._pad_zernikes(zk_param, zk_prior)
        return padded_param + padded_prior

    def predict_zernikes(self, positions):
        """Predict combined Zernike coefficients (inference: interpolation).

        Parameters
        ----------
        positions : jnp.ndarray, shape (batch, 2)

        Returns
        -------
        zks_coeffs : jnp.ndarray, shape (batch, n_zks_total, 1, 1)
        """
        zk_param = self.poly_field(positions)
        zk_prior = self.physical_layer.predict(positions)
        padded_param, padded_prior = self._pad_zernikes(zk_param, zk_prior)
        return padded_param + padded_prior

    def __call__(self, inputs, training=True):
        """Forward model.

        Parameters
        ----------
        inputs : list
            [positions, packed_SED_data]
        training : bool
            If True, uses index-based lookup for physical layer.
            If False, uses interpolation.

        Returns
        -------
        tuple
            (psf_batch, opd_total)
        """
        positions = inputs[0]
        packed_SED_data = inputs[1]

        # Compute Zernike coefficients
        if training:
            zks_coeffs = self.compute_zernikes(positions)
        else:
            zks_coeffs = self.predict_zernikes(positions)

        # OPD from parametric (Zernike) path
        param_opd = self.zernike_opd(zks_coeffs)

        # Non-parametric contribution
        nonparam_opd = self.np_opd(positions)

        # Total OPD
        opd_total = param_opd + nonparam_opd

        # Generate PSFs
        psf_batch = self.batch_poly_psf([opd_total, packed_SED_data])

        return psf_batch, opd_total

    def predict_opd(self, positions):
        """Predict OPD maps at given positions (using interpolation).

        Parameters
        ----------
        positions : jnp.ndarray, shape (batch, 2)

        Returns
        -------
        opd_maps : jnp.ndarray, shape (batch, wfe_dim, wfe_dim)
        """
        zks_coeffs = self.predict_zernikes(positions)
        param_opd = self.zernike_opd(zks_coeffs)
        nonparam_opd = self.np_opd(positions)
        return param_opd + nonparam_opd

    def predict_mono_psfs(self, positions, lambda_obs, phase_N=914):
        """Predict monochromatic PSFs at given positions.

        Parameters
        ----------
        positions : jnp.ndarray, shape (batch, 2)
        lambda_obs : float
        phase_N : int

        Returns
        -------
        mono_psf_batch : jnp.ndarray, shape (batch, output_dim, output_dim)
        """
        zks_coeffs = self.predict_zernikes(positions)
        param_opd = self.zernike_opd(zks_coeffs)
        nonparam_opd = self.np_opd(positions)
        opd_maps = param_opd + nonparam_opd

        batch_mono_psf = BatchMonochromaticPSF(
            obscurations=self.batch_poly_psf.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
            phase_N=phase_N,
            lambda_obs=lambda_obs,
        )
        return batch_mono_psf(opd_maps)


@register_psfclass
class PhysicalPolychromaticFieldFactory(PSFModelBaseFactory):
    """Factory for the physical polychromatic PSF field model.

    Registered with id ``"physical-poly"``.
    """

    ids = ("physical-poly",)

    def get_model_instance(self, model_params, training_params, data=None, coeff_mat=None):
        """Create a PhysicalPolychromaticField from config namespaces.

        Parameters
        ----------
        model_params : object
        training_params : object
        data : object
            Must provide obs_pos and zks_prior.
        coeff_mat : array or None

        Returns
        -------
        PhysicalPolychromaticField
        """
        from wavediff_jax.utils.math_utils import (
            generate_zernike_maps_3d,
            obscurations_from_params,
        )
        import numpy as np

        n_zernikes_param = model_params.param_hparams.n_zernikes
        zks_prior = data.zks_prior
        n_zks_prior = zks_prior.shape[1]
        n_zks_total = max(n_zernikes_param, n_zks_prior)

        zernike_maps = generate_zernike_maps_3d(
            n_zks_total, model_params.pupil_diameter
        )
        obscurations = obscurations_from_params(
            pupil_diam=model_params.pupil_diameter,
            N_filter=model_params.LP_filter_length,
            rotation_angle=model_params.obscuration_rotation_angle,
        )

        key = jax.random.PRNGKey(
            getattr(model_params.param_hparams, "random_seed", 0)
        )

        model = PhysicalPolychromaticField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            obs_pos=data.obs_pos,
            zks_prior=np.array(zks_prior),
            output_Q=model_params.output_Q,
            output_dim=model_params.output_dim,
            n_zernikes_param=n_zernikes_param,
            n_zks_total=n_zks_total,
            d_max=model_params.param_hparams.d_max,
            d_max_nonparam=model_params.nonparam_hparams.d_max_nonparam,
            x_lims=model_params.x_lims,
            y_lims=model_params.y_lims,
            interpolation_type=getattr(model_params, "interpolation_type", None),
            interpolation_args=getattr(model_params, "interpolation_args", None),
            key=key,
        )

        if coeff_mat is not None:
            model = eqx.tree_at(
                lambda m: m.poly_field.coeff_mat,
                model,
                jnp.array(coeff_mat),
            )

        return model
