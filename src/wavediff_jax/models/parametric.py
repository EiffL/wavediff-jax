"""Parametric PSF field model.

Fully parametric model based on the Zernike polynomial basis.
Port of ``extern/wf-psf/src/wf_psf/psf_models/psf_model_parametric.py`` to JAX/Equinox.

:Authors: WaveDiff-JAX contributors
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from wavediff_jax.models.registry import register_psfclass, PSFModelBaseFactory
from wavediff_jax.models.layers import (
    PolynomialZernikeField,
    ZernikeOPD,
    BatchPolychromaticPSF,
    BatchMonochromaticPSF,
)


class ParametricPSFFieldModel(eqx.Module):
    """Parametric PSF field model.

    Fully parametric model based on the Zernike polynomial basis.

    Pipeline:
        [positions] -> PolynomialZernikeField -> ZernikeOPD -> BatchPolychromaticPSF

    Parameters
    ----------
    zernike_maps : jnp.ndarray, shape (n_zernikes, wfe_dim, wfe_dim)
        Precomputed Zernike polynomial maps.
    obscurations : jnp.ndarray, shape (wfe_dim, wfe_dim)
        Pupil obscuration mask (complex).
    output_Q : int
        Oversampling factor for PSF generation.
    output_dim : int
        Output PSF stamp dimension.
    n_zernikes : int
        Number of Zernike polynomials.
    d_max : int
        Maximum polynomial degree for field variations.
    x_lims : list
        [x_min, x_max] field-of-view limits.
    y_lims : list
        [y_min, y_max] field-of-view limits.
    key : jax.random.PRNGKey
        Random key for parameter initialization.
    """

    poly_field: PolynomialZernikeField
    zernike_opd: ZernikeOPD
    batch_poly_psf: BatchPolychromaticPSF
    n_zernikes: int = eqx.field(static=True)
    output_Q: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    def __init__(
        self,
        zernike_maps,
        obscurations,
        output_Q,
        output_dim=64,
        n_zernikes=45,
        d_max=2,
        x_lims=None,
        y_lims=None,
        *,
        key=None,
    ):
        if x_lims is None:
            x_lims = [0, 1e3]
        if y_lims is None:
            y_lims = [0, 1e3]
        if key is None:
            key = jax.random.PRNGKey(0)

        self.n_zernikes = n_zernikes
        self.output_Q = int(output_Q)
        self.output_dim = output_dim

        self.poly_field = PolynomialZernikeField(
            x_lims=x_lims,
            y_lims=y_lims,
            n_zernikes=n_zernikes,
            d_max=d_max,
            key=key,
        )
        self.zernike_opd = ZernikeOPD(zernike_maps=zernike_maps)
        self.batch_poly_psf = BatchPolychromaticPSF(
            obscurations=obscurations,
            output_Q=output_Q,
            output_dim=output_dim,
        )

    def __call__(self, inputs):
        """Forward model.

        Parameters
        ----------
        inputs : list
            [positions, packed_SED_data] where
            positions: (batch, 2), packed_SED_data: (batch, n_wavelengths, 3)

        Returns
        -------
        tuple
            (psf_batch, opd_maps) where
            psf_batch: (batch, output_dim, output_dim),
            opd_maps: (batch, wfe_dim, wfe_dim)
        """
        positions = inputs[0]
        packed_SED_data = inputs[1]

        z_coeffs = self.poly_field(positions)
        opd_maps = self.zernike_opd(z_coeffs)
        psf_batch = self.batch_poly_psf([opd_maps, packed_SED_data])

        return psf_batch, opd_maps

    def predict_opd(self, positions):
        """Predict OPD maps at given positions.

        Parameters
        ----------
        positions : jnp.ndarray, shape (batch, 2)

        Returns
        -------
        opd_maps : jnp.ndarray, shape (batch, wfe_dim, wfe_dim)
        """
        z_coeffs = self.poly_field(positions)
        return self.zernike_opd(z_coeffs)

    def predict_mono_psfs(self, positions, lambda_obs, phase_N=914):
        """Predict monochromatic PSFs at given positions.

        Parameters
        ----------
        positions : jnp.ndarray, shape (batch, 2)
        lambda_obs : float
            Observation wavelength.
        phase_N : int
            Wavefront dimension for padding.

        Returns
        -------
        mono_psf_batch : jnp.ndarray, shape (batch, output_dim, output_dim)
        """
        z_coeffs = self.poly_field(positions)
        opd_maps = self.zernike_opd(z_coeffs)

        batch_mono_psf = BatchMonochromaticPSF(
            obscurations=self.batch_poly_psf.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
            phase_N=phase_N,
            lambda_obs=lambda_obs,
        )
        return batch_mono_psf(opd_maps)


@register_psfclass
class ParametricPSFFieldFactory(PSFModelBaseFactory):
    """Factory for the parametric PSF field model.

    Registered with id ``"poly"``.
    """

    ids = ("poly",)

    def get_model_instance(self, model_params, training_params, data=None, coeff_mat=None):
        """Create a ParametricPSFFieldModel from config namespaces.

        Parameters
        ----------
        model_params : object
            Model configuration with attributes: output_Q, output_dim,
            param_hparams.n_zernikes, param_hparams.d_max, x_lims, y_lims,
            pupil_diameter, LP_filter_length, obscuration_rotation_angle.
        training_params : object
            Training configuration (batch_size, etc.).
        data : object or None
            Optional data handler.
        coeff_mat : array or None
            Optional coefficient matrix initialization.

        Returns
        -------
        ParametricPSFFieldModel
        """
        from wavediff_jax.utils.math_utils import (
            generate_zernike_maps_3d,
            obscurations_from_params,
        )

        n_zernikes = model_params.param_hparams.n_zernikes
        zernike_maps = generate_zernike_maps_3d(
            n_zernikes, model_params.pupil_diameter
        )
        obscurations = obscurations_from_params(
            pupil_diam=model_params.pupil_diameter,
            N_filter=model_params.LP_filter_length,
            rotation_angle=model_params.obscuration_rotation_angle,
        )

        key = jax.random.PRNGKey(
            getattr(model_params.param_hparams, "random_seed", 0)
        )

        model = ParametricPSFFieldModel(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=model_params.output_Q,
            output_dim=model_params.output_dim,
            n_zernikes=n_zernikes,
            d_max=model_params.param_hparams.d_max,
            x_lims=model_params.x_lims,
            y_lims=model_params.y_lims,
            key=key,
        )

        if coeff_mat is not None:
            model = eqx.tree_at(
                lambda m: m.poly_field.coeff_mat,
                model,
                jnp.array(coeff_mat),
            )

        return model
