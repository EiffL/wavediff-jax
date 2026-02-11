"""Ground-truth PSF field model wrappers.

Convenience constructors that create pre-configured models for ground-truth
evaluation. Port of ``extern/wf-psf/src/wf_psf/psf_models/tf_psf_field.py``.

:Authors: WaveDiff-JAX contributors
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import logging

from wavediff_jax.models.registry import register_psfclass, PSFModelBaseFactory
from wavediff_jax.models.semiparametric import SemiParametricField, set_alpha_zero
from wavediff_jax.models.physical_polychromatic import PhysicalPolychromaticField
from wavediff_jax.models.layers import (
    ZernikeOPD,
    BatchPolychromaticPSF,
    BatchMonochromaticPSF,
    PhysicalLayer,
)

logger = logging.getLogger(__name__)


def create_ground_truth_semi_parametric(
    zernike_maps,
    obscurations,
    output_Q,
    output_dim=64,
    n_zernikes=45,
    d_max=2,
    d_max_nonparam=3,
    x_lims=None,
    y_lims=None,
    coeff_mat=None,
    *,
    key=None,
):
    """Create a SemiParametricField with non-parametric part zeroed out.

    This is the ground-truth model: the non-parametric alpha_mat is set to
    zero so only the parametric Zernike polynomial field contributes.

    Parameters
    ----------
    zernike_maps : jnp.ndarray
    obscurations : jnp.ndarray
    output_Q : int
    output_dim : int
    n_zernikes : int
    d_max : int
    d_max_nonparam : int
    x_lims : list or None
    y_lims : list or None
    coeff_mat : array or None
        If provided, sets the parametric coefficient matrix.
    key : jax.random.PRNGKey or None

    Returns
    -------
    SemiParametricField
        Model with non-parametric part zeroed out.
    """
    model = SemiParametricField(
        zernike_maps=zernike_maps,
        obscurations=obscurations,
        output_Q=output_Q,
        output_dim=output_dim,
        n_zernikes=n_zernikes,
        d_max=d_max,
        d_max_nonparam=d_max_nonparam,
        x_lims=x_lims,
        y_lims=y_lims,
        key=key,
    )

    # Set the non-parametric part to zero
    model = set_alpha_zero(model)

    # Optionally set the coefficient matrix
    if coeff_mat is not None:
        model = eqx.tree_at(
            lambda m: m.poly_field.coeff_mat,
            model,
            jnp.array(coeff_mat),
        )

    return model


class GroundTruthPhysicalField(eqx.Module):
    """Ground-truth PSF field model with physical layer only.

    Uses only the physical Zernike prior (no learned correction, no
    non-parametric component). This serves as the reference model for
    evaluating the physical polychromatic field.

    Parameters
    ----------
    zernike_maps : jnp.ndarray, shape (n_zks_total, wfe_dim, wfe_dim)
    obscurations : jnp.ndarray, shape (wfe_dim, wfe_dim)
    obs_pos : array, shape (n_obs, 2)
    zks_prior : array, shape (n_obs, n_zks_prior)
    output_Q : int
    output_dim : int
    interpolation_type : str or None
        For the physical layer. Use None for exact lookup.
    """

    physical_layer: PhysicalLayer
    zernike_opd: ZernikeOPD
    batch_poly_psf: BatchPolychromaticPSF
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
        interpolation_type=None,
    ):
        self.output_Q = int(output_Q)
        self.output_dim = output_dim

        self.physical_layer = PhysicalLayer(
            obs_pos=obs_pos,
            zks_prior=zks_prior,
            interpolation_type=interpolation_type,
        )
        self.zernike_opd = ZernikeOPD(zernike_maps=zernike_maps)
        self.batch_poly_psf = BatchPolychromaticPSF(
            obscurations=obscurations,
            output_Q=output_Q,
            output_dim=output_dim,
        )

    def compute_zernikes(self, positions):
        """Compute Zernike coefficients via physical layer (index lookup)."""
        return self.physical_layer(positions)

    def predict_zernikes(self, positions):
        """Predict Zernike coefficients via physical layer (interpolation)."""
        return self.physical_layer.predict(positions)

    def __call__(self, inputs, training=True):
        """Forward model.

        Parameters
        ----------
        inputs : list
            [positions, packed_SED_data]
        training : bool
            If True uses index lookup; if False uses interpolation.

        Returns
        -------
        tuple
            (psf_batch, opd_maps)
        """
        positions = inputs[0]
        packed_SED_data = inputs[1]

        if training:
            zks = self.compute_zernikes(positions)
        else:
            zks = self.predict_zernikes(positions)

        opd_maps = self.zernike_opd(zks)
        psf_batch = self.batch_poly_psf([opd_maps, packed_SED_data])

        return psf_batch, opd_maps

    def predict_opd(self, positions):
        """Predict OPD maps at given positions."""
        zks = self.predict_zernikes(positions)
        return self.zernike_opd(zks)

    def predict_mono_psfs(self, positions, lambda_obs, phase_N=914):
        """Predict monochromatic PSFs at given positions."""
        zks = self.predict_zernikes(positions)
        opd_maps = self.zernike_opd(zks)
        batch_mono_psf = BatchMonochromaticPSF(
            obscurations=self.batch_poly_psf.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
            phase_N=phase_N,
            lambda_obs=lambda_obs,
        )
        return batch_mono_psf(opd_maps)


@register_psfclass
class GroundTruthSemiParametricFieldFactory(PSFModelBaseFactory):
    """Factory for the ground-truth semi-parametric PSF field model.

    Registered with id ``"ground-truth-semi-param"``.
    """

    ids = ("ground-truth-semi-param",)

    def get_model_instance(self, model_params, training_params, data=None, coeff_mat=None):
        from wavediff_jax.utils.math_utils import (
            generate_zernike_maps_3d,
            obscurations_from_params,
        )

        n_zernikes = model_params.param_hparams.n_zernikes
        zernike_maps = generate_zernike_maps_3d(n_zernikes, model_params.pupil_diameter)
        obscurations = obscurations_from_params(
            pupil_diam=model_params.pupil_diameter,
            N_filter=model_params.LP_filter_length,
            rotation_angle=model_params.obscuration_rotation_angle,
        )

        key = jax.random.PRNGKey(getattr(model_params.param_hparams, "random_seed", 0))

        return create_ground_truth_semi_parametric(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=model_params.output_Q,
            output_dim=model_params.output_dim,
            n_zernikes=n_zernikes,
            d_max=model_params.param_hparams.d_max,
            d_max_nonparam=model_params.nonparam_hparams.d_max_nonparam,
            x_lims=model_params.x_lims,
            y_lims=model_params.y_lims,
            coeff_mat=coeff_mat,
            key=key,
        )


@register_psfclass
class GroundTruthPhysicalFieldFactory(PSFModelBaseFactory):
    """Factory for the ground-truth physical PSF field model.

    Registered with id ``"ground-truth-physical-poly"``.
    """

    ids = ("ground-truth-physical-poly",)

    def get_model_instance(self, model_params, training_params, data=None, coeff_mat=None):
        from wavediff_jax.utils.math_utils import (
            generate_zernike_maps_3d,
            obscurations_from_params,
        )

        zks_prior = data.zks_prior
        n_zks_total = max(
            model_params.param_hparams.n_zernikes,
            zks_prior.shape[1],
        )
        zernike_maps = generate_zernike_maps_3d(n_zks_total, model_params.pupil_diameter)
        obscurations = obscurations_from_params(
            pupil_diam=model_params.pupil_diameter,
            N_filter=model_params.LP_filter_length,
            rotation_angle=model_params.obscuration_rotation_angle,
        )

        return GroundTruthPhysicalField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            obs_pos=data.obs_pos,
            zks_prior=np.array(zks_prior),
            output_Q=model_params.output_Q,
            output_dim=model_params.output_dim,
            interpolation_type=None,
        )
