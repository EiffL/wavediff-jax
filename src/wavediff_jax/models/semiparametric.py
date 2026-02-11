"""Semi-parametric PSF field model.

Combines a parametric Zernike polynomial field with a non-parametric OPD component.
Port of ``extern/wf-psf/src/wf_psf/psf_models/psf_model_semiparametric.py`` to JAX/Equinox.

:Authors: WaveDiff-JAX contributors
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import logging

from wavediff_jax.models.registry import register_psfclass, PSFModelBaseFactory
from wavediff_jax.models.layers import (
    PolynomialZernikeField,
    ZernikeOPD,
    BatchPolychromaticPSF,
    BatchMonochromaticPSF,
    NonParametricPolynomialOPD,
    NonParametricMCCDOPD,
    NonParametricGraphOPD,
)

logger = logging.getLogger(__name__)

# Mapping from string identifier to the layer class
_NONPARAM_CLASSES = {
    "poly": NonParametricPolynomialOPD,
    "mccd": NonParametricMCCDOPD,
    "graph": NonParametricGraphOPD,
}


class SemiParametricField(eqx.Module):
    """Semi-parametric PSF field model.

    Combines a parametric Zernike field with a non-parametric OPD component.

    Pipeline:
        positions -> PolynomialZernikeField -> ZernikeOPD -> param_opd
        positions -> NonParametricOPD                     -> nonparam_opd
        opd_total = param_opd + nonparam_opd
        [opd_total, packed_SED_data] -> BatchPolychromaticPSF -> psf_batch

    Parameters
    ----------
    zernike_maps : jnp.ndarray, shape (n_zernikes, wfe_dim, wfe_dim)
    obscurations : jnp.ndarray, shape (wfe_dim, wfe_dim)
    output_Q : int
    output_dim : int
    n_zernikes : int
    d_max : int
    d_max_nonparam : int
    x_lims : list
    y_lims : list
    nonparam_model_class : str
        One of "poly", "mccd", "graph".
    nonparam_kwargs : dict or None
        Extra keyword arguments for the non-parametric layer constructor.
    key : jax.random.PRNGKey
    """

    poly_field: PolynomialZernikeField
    zernike_opd: ZernikeOPD
    np_opd: eqx.Module  # one of NonParametricPolynomialOPD / MCCD / Graph
    batch_poly_psf: BatchPolychromaticPSF
    n_zernikes: int = eqx.field(static=True)
    output_Q: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    nonparam_type: str = eqx.field(static=True)

    def __init__(
        self,
        zernike_maps,
        obscurations,
        output_Q,
        output_dim=64,
        n_zernikes=45,
        d_max=2,
        d_max_nonparam=3,
        x_lims=None,
        y_lims=None,
        nonparam_model_class="poly",
        nonparam_kwargs=None,
        *,
        key=None,
    ):
        if x_lims is None:
            x_lims = [0, 1e3]
        if y_lims is None:
            y_lims = [0, 1e3]
        if key is None:
            key = jax.random.PRNGKey(0)
        if nonparam_kwargs is None:
            nonparam_kwargs = {}

        self.n_zernikes = n_zernikes
        self.output_Q = int(output_Q)
        self.output_dim = output_dim
        self.nonparam_type = nonparam_model_class

        k1, k2 = jax.random.split(key)

        # Parametric layers
        self.poly_field = PolynomialZernikeField(
            x_lims=x_lims,
            y_lims=y_lims,
            n_zernikes=n_zernikes,
            d_max=d_max,
            key=k1,
        )
        self.zernike_opd = ZernikeOPD(zernike_maps=zernike_maps)

        # Non-parametric layer
        opd_dim = zernike_maps.shape[1]
        np_cls = _NONPARAM_CLASSES[nonparam_model_class]
        if nonparam_model_class == "poly":
            self.np_opd = np_cls(
                x_lims=x_lims,
                y_lims=y_lims,
                d_max=d_max_nonparam,
                opd_dim=opd_dim,
                key=k2,
            )
        else:
            # mccd and graph require obs_pos, spatial_dic, etc.
            self.np_opd = np_cls(
                opd_dim=opd_dim,
                key=k2,
                **nonparam_kwargs,
            )

        # PSF layer
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
            [positions, packed_SED_data]

        Returns
        -------
        tuple
            For "poly" variant: (psf_batch, opd_total)
            For "mccd"/"graph" variant: (psf_batch, opd_total, aux_dict)
        """
        positions = inputs[0]
        packed_SED_data = inputs[1]

        # Parametric part
        z_coeffs = self.poly_field(positions)
        opd_param = self.zernike_opd(z_coeffs)

        # Non-parametric part
        if self.nonparam_type in ("mccd", "graph"):
            opd_nonparam, aux = self.np_opd(positions)
        else:
            opd_nonparam = self.np_opd(positions)
            aux = None

        opd_total = opd_param + opd_nonparam
        psf_batch = self.batch_poly_psf([opd_total, packed_SED_data])

        if aux is not None:
            return psf_batch, opd_total, aux
        return psf_batch, opd_total

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
        opd_param = self.zernike_opd(z_coeffs)

        if self.nonparam_type in ("mccd", "graph"):
            opd_nonparam, _ = self.np_opd(positions)
        else:
            opd_nonparam = self.np_opd(positions)

        return opd_param + opd_nonparam

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
        opd_maps = self.predict_opd(positions)
        batch_mono_psf = BatchMonochromaticPSF(
            obscurations=self.batch_poly_psf.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
            phase_N=phase_N,
            lambda_obs=lambda_obs,
        )
        return batch_mono_psf(opd_maps)


# ---------------------------------------------------------------------------
# Functional mutations (pure -- return new model copies)
# ---------------------------------------------------------------------------


def set_alpha_zero(model):
    """Return a new model with alpha_mat set to zero.

    Parameters
    ----------
    model : SemiParametricField

    Returns
    -------
    SemiParametricField
        New model with zeroed alpha_mat.
    """
    zero_alpha = jnp.zeros_like(model.np_opd.alpha_mat)
    return eqx.tree_at(lambda m: m.np_opd.alpha_mat, model, zero_alpha)


def set_alpha_identity(model):
    """Return a new model with alpha_mat reset to identity.

    Parameters
    ----------
    model : SemiParametricField

    Returns
    -------
    SemiParametricField
        New model with identity alpha_mat.
    """
    n = model.np_opd.alpha_mat.shape[0]
    m = model.np_opd.alpha_mat.shape[1]
    identity_alpha = jnp.eye(n, m)
    return eqx.tree_at(lambda m: m.np_opd.alpha_mat, model, identity_alpha)


def project_DD_features(model, zernike_cube=None, obscurations=None, n_zernikes=None, iters=40):
    """Project data-driven features onto the parametric Zernike basis.

    This is the functional equivalent of TFSemiParametricField.project_DD_features.
    It extracts Zernike content from the non-parametric S features, transfers it
    to the parametric coefficient matrix, and returns a new model with updated
    coeff_mat and S_mat.

    Parameters
    ----------
    model : SemiParametricField
        Must use the "poly" non-parametric variant.
    zernike_cube : jnp.ndarray or None
        Zernike maps for projection. If None, uses model.zernike_opd.zernike_maps.
    obscurations : jnp.ndarray or None
        Obscuration mask. If None, uses model.batch_poly_psf.obscurations.
    n_zernikes : int or None
        Number of Zernikes to project onto. If None, uses model.n_zernikes.
    iters : int
        Number of iterations for the decomposition.

    Returns
    -------
    SemiParametricField
        New model with updated coeff_mat and S_mat.
    """
    from wavediff_jax.utils.math_utils import decompose_obscured_opd_basis

    if zernike_cube is None:
        zernike_cube = model.zernike_opd.zernike_maps
    if obscurations is None:
        obscurations = model.batch_poly_psf.obscurations
    if n_zernikes is None:
        n_zernikes = model.n_zernikes

    n_poly_param = model.poly_field.coeff_mat.shape[1]

    # Compute alpha * S for the first n_poly_param terms
    inter_res = model.np_opd.alpha_mat[:n_poly_param] @ model.np_opd.S_mat.reshape(
        model.np_opd.S_mat.shape[0], -1
    )
    inter_res = inter_res.reshape(n_poly_param, model.np_opd.opd_dim, model.np_opd.opd_dim)

    # Project over first n_zernikes
    delta_C_poly = np.array([
        decompose_obscured_opd_basis(
            opd=inter_res[j],
            obscurations=obscurations,
            zk_basis=zernike_cube,
            n_zernike=n_zernikes,
            iters=iters,
        )
        for j in range(n_poly_param)
    ]).T  # (n_zernikes, n_poly_param)

    new_coeff_mat = model.poly_field.coeff_mat + jnp.array(delta_C_poly, dtype=jnp.float32)

    # Compute s_tilde = alpha @ S
    s_tilde = jnp.tensordot(model.np_opd.alpha_mat, model.np_opd.S_mat, axes=1)

    # Project first n_poly_param slices
    beta_tilde_inner = np.array([
        decompose_obscured_opd_basis(
            opd=s_tilde[j],
            obscurations=obscurations,
            zk_basis=zernike_cube,
            n_zernike=n_zernikes,
            iters=iters,
        )
        for j in range(n_poly_param)
    ])

    # Pad to full alpha dimension
    beta_tilde = np.pad(
        beta_tilde_inner,
        [(0, s_tilde.shape[0] - beta_tilde_inner.shape[0]), (0, 0)],
        mode="constant",
    )

    # Unmix
    beta = jnp.array(
        np.linalg.inv(np.array(model.np_opd.alpha_mat)) @ beta_tilde,
        dtype=jnp.float32,
    )
    s_mat_projected = jnp.tensordot(beta, zernike_cube, axes=[1, 0])
    new_S_mat = model.np_opd.S_mat - s_mat_projected

    # Build new model
    model = eqx.tree_at(lambda m: m.poly_field.coeff_mat, model, new_coeff_mat)
    model = eqx.tree_at(lambda m: m.np_opd.S_mat, model, new_S_mat)
    return model


@register_psfclass
class SemiParametricFieldFactory(PSFModelBaseFactory):
    """Factory for the semi-parametric PSF field model.

    Registered with id ``"semi-param"``.
    """

    ids = ("semi-param",)

    def get_model_instance(self, model_params, training_params, data=None, coeff_mat=None):
        """Create a SemiParametricField from config namespaces.

        Parameters
        ----------
        model_params : object
        training_params : object
        data : object or None
        coeff_mat : array or None

        Returns
        -------
        SemiParametricField
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

        model = SemiParametricField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=model_params.output_Q,
            output_dim=model_params.output_dim,
            n_zernikes=n_zernikes,
            d_max=model_params.param_hparams.d_max,
            d_max_nonparam=model_params.nonparam_hparams.d_max_nonparam,
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
