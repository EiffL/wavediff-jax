"""Tests for PSF field models: parametric, semiparametric, physical polychromatic, ground truth."""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from wavediff_jax.utils.math_utils import generate_zernike_maps_3d
from wavediff_jax.models.parametric import ParametricPSFFieldModel
from wavediff_jax.models.semiparametric import (
    SemiParametricField,
    set_alpha_zero,
    set_alpha_identity,
)
from wavediff_jax.models.physical_polychromatic import PhysicalPolychromaticField
from wavediff_jax.models.ground_truth import (
    create_ground_truth_semi_parametric,
    GroundTruthPhysicalField,
)

# ---- Shared fixtures ----

N_ZERNIKES = 15
WFE_DIM = 32
OUTPUT_DIM = 8
OUTPUT_Q = 1
BATCH_SIZE = 2
N_WAVELENGTHS = 3
X_LIMS = [0.0, 1e3]
Y_LIMS = [0.0, 1e3]
D_MAX = 2
D_MAX_NONPARAM = 2


@pytest.fixture
def zernike_maps():
    return generate_zernike_maps_3d(N_ZERNIKES, WFE_DIM)


@pytest.fixture
def obscurations():
    """Simple circular pupil (complex) for testing."""
    y, x = np.mgrid[-1:1:complex(WFE_DIM), -1:1:complex(WFE_DIM)]
    r = np.sqrt(x ** 2 + y ** 2)
    mask = (r <= 1.0).astype(np.float32)
    return jnp.array(mask + 0j, dtype=jnp.complex64)


@pytest.fixture
def positions():
    return jnp.array([[500.0, 500.0], [200.0, 800.0]], dtype=jnp.float32)


@pytest.fixture
def packed_seds():
    """Fake packed SED data: (batch, n_wavelengths, 3) -- [phase_N, lambda_obs, weight]."""
    phase_N = float(WFE_DIM * 2)
    seds = np.zeros((BATCH_SIZE, N_WAVELENGTHS, 3), dtype=np.float32)
    for i in range(N_WAVELENGTHS):
        seds[:, i, 0] = phase_N  # phase_N
        seds[:, i, 1] = 0.5 + 0.1 * i  # lambda_obs
        seds[:, i, 2] = 1.0 / N_WAVELENGTHS  # weight
    return jnp.array(seds)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ---- Parametric model tests ----


class TestParametricPSFFieldModel:
    def test_forward_pass_shapes(self, zernike_maps, obscurations, positions, packed_seds, key):
        model = ParametricPSFFieldModel(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        psf_batch, opd_maps = model([positions, packed_seds])
        assert psf_batch.shape == (BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM)
        assert opd_maps.shape == (BATCH_SIZE, WFE_DIM, WFE_DIM)

    def test_psf_non_negative(self, zernike_maps, obscurations, positions, packed_seds, key):
        model = ParametricPSFFieldModel(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        psf_batch, _ = model([positions, packed_seds])
        assert jnp.all(psf_batch >= 0), "PSF values should be non-negative"

    def test_predict_opd(self, zernike_maps, obscurations, positions, key):
        model = ParametricPSFFieldModel(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        opd = model.predict_opd(positions)
        assert opd.shape == (BATCH_SIZE, WFE_DIM, WFE_DIM)

    def test_predict_mono_psfs(self, zernike_maps, obscurations, positions, key):
        model = ParametricPSFFieldModel(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        mono_psfs = model.predict_mono_psfs(positions, lambda_obs=0.7, phase_N=WFE_DIM * 2)
        assert mono_psfs.shape == (BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM)
        assert jnp.all(mono_psfs >= 0)

    def test_coeff_mat_update(self, zernike_maps, obscurations, key):
        model = ParametricPSFFieldModel(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            key=key,
        )
        new_coeff = jnp.ones_like(model.poly_field.coeff_mat) * 0.5
        model2 = eqx.tree_at(lambda m: m.poly_field.coeff_mat, model, new_coeff)
        assert jnp.allclose(model2.poly_field.coeff_mat, 0.5)
        # Original is unchanged
        assert not jnp.allclose(model.poly_field.coeff_mat, 0.5)


# ---- Semi-parametric model tests ----


class TestSemiParametricField:
    def test_forward_pass_poly(self, zernike_maps, obscurations, positions, packed_seds, key):
        model = SemiParametricField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            nonparam_model_class="poly",
            key=key,
        )
        result = model([positions, packed_seds])
        assert len(result) == 2
        psf_batch, opd_total = result
        assert psf_batch.shape == (BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM)
        assert opd_total.shape == (BATCH_SIZE, WFE_DIM, WFE_DIM)

    def test_psf_non_negative(self, zernike_maps, obscurations, positions, packed_seds, key):
        model = SemiParametricField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        psf_batch, _ = model([positions, packed_seds])
        assert jnp.all(psf_batch >= 0)

    def test_set_alpha_zero(self, zernike_maps, obscurations, positions, packed_seds, key):
        model = SemiParametricField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        model_zeroed = set_alpha_zero(model)
        # alpha_mat should be all zeros
        assert jnp.allclose(model_zeroed.np_opd.alpha_mat, 0.0)
        # Original unchanged
        assert not jnp.allclose(model.np_opd.alpha_mat, 0.0)
        # Model should still produce valid output
        psf_batch, opd = model_zeroed([positions, packed_seds])
        assert psf_batch.shape == (BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM)

    def test_set_alpha_identity(self, zernike_maps, obscurations, key):
        model = SemiParametricField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        # Zero first, then reset to identity
        model_zeroed = set_alpha_zero(model)
        model_identity = set_alpha_identity(model_zeroed)
        n = model_identity.np_opd.alpha_mat.shape[0]
        assert jnp.allclose(model_identity.np_opd.alpha_mat, jnp.eye(n))

    def test_predict_opd(self, zernike_maps, obscurations, positions, key):
        model = SemiParametricField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        opd = model.predict_opd(positions)
        assert opd.shape == (BATCH_SIZE, WFE_DIM, WFE_DIM)

    def test_zeroed_nonparam_matches_parametric_only(
        self, zernike_maps, obscurations, positions, packed_seds, key
    ):
        """When nonparam alpha is zeroed, OPD should equal parametric-only OPD."""
        model = SemiParametricField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        model_zeroed = set_alpha_zero(model)

        # Compute parametric-only OPD
        z_coeffs = model_zeroed.poly_field(positions)
        param_opd = model_zeroed.zernike_opd(z_coeffs)

        # Compute total OPD via predict_opd (which includes zeroed nonparam)
        total_opd = model_zeroed.predict_opd(positions)

        assert jnp.allclose(param_opd, total_opd, atol=1e-6)


# ---- Physical polychromatic model tests ----


class TestPhysicalPolychromaticField:
    @pytest.fixture
    def obs_pos(self):
        """Mock observation positions."""
        return np.array(
            [[500.0, 500.0], [200.0, 800.0], [800.0, 200.0], [100.0, 100.0]],
            dtype=np.float32,
        )

    @pytest.fixture
    def zks_prior(self):
        """Mock prior Zernike coefficients."""
        return np.random.RandomState(0).randn(4, N_ZERNIKES).astype(np.float32) * 0.01

    @pytest.mark.filterwarnings("ignore:A JAX array is being set as static")
    def test_forward_pass_training(
        self, zernike_maps, obscurations, positions, packed_seds, obs_pos, zks_prior, key
    ):
        model = PhysicalPolychromaticField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            obs_pos=obs_pos,
            zks_prior=zks_prior,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes_param=N_ZERNIKES,
            n_zks_total=N_ZERNIKES,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        psf_batch, opd_total = model([positions, packed_seds], training=True)
        assert psf_batch.shape == (BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM)
        assert opd_total.shape == (BATCH_SIZE, WFE_DIM, WFE_DIM)
        assert jnp.all(psf_batch >= 0)

    @pytest.mark.filterwarnings("ignore:A JAX array is being set as static")
    def test_forward_pass_inference(
        self, zernike_maps, obscurations, positions, packed_seds, obs_pos, zks_prior, key
    ):
        model = PhysicalPolychromaticField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            obs_pos=obs_pos,
            zks_prior=zks_prior,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes_param=N_ZERNIKES,
            n_zks_total=N_ZERNIKES,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        psf_batch, opd_total = model([positions, packed_seds], training=False)
        assert psf_batch.shape == (BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM)
        assert opd_total.shape == (BATCH_SIZE, WFE_DIM, WFE_DIM)

    @pytest.mark.filterwarnings("ignore:A JAX array is being set as static")
    def test_combines_physical_and_learned(
        self, zernike_maps, obscurations, positions, obs_pos, zks_prior, key
    ):
        """Physical + learned correction should differ from either alone."""
        model = PhysicalPolychromaticField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            obs_pos=obs_pos,
            zks_prior=zks_prior,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes_param=N_ZERNIKES,
            n_zks_total=N_ZERNIKES,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        # Combined Zernike coefficients
        combined = model.compute_zernikes(positions)

        # Physical only
        physical_only = model.physical_layer(positions)

        # Learned only
        learned_only = model.poly_field(positions)

        # The combined should be sum of padded versions
        padded_learned, padded_physical = model._pad_zernikes(learned_only, physical_only)
        expected = padded_learned + padded_physical
        assert jnp.allclose(combined, expected, atol=1e-6)

    @pytest.mark.filterwarnings("ignore:A JAX array is being set as static")
    def test_predict_opd(self, zernike_maps, obscurations, positions, obs_pos, zks_prior, key):
        model = PhysicalPolychromaticField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            obs_pos=obs_pos,
            zks_prior=zks_prior,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes_param=N_ZERNIKES,
            n_zks_total=N_ZERNIKES,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        opd = model.predict_opd(positions)
        assert opd.shape == (BATCH_SIZE, WFE_DIM, WFE_DIM)

    @pytest.mark.filterwarnings("ignore:A JAX array is being set as static")
    def test_padding_with_different_zernike_counts(self, obscurations, key):
        """Test that models work when prior has different n_zernikes than parametric."""
        n_zks_param = 10
        n_zks_prior = 15
        n_zks_total = max(n_zks_param, n_zks_prior)
        zernike_maps = generate_zernike_maps_3d(n_zks_total, WFE_DIM)
        obs_pos = np.array([[500.0, 500.0], [200.0, 800.0]], dtype=np.float32)
        zks_prior = np.random.RandomState(0).randn(2, n_zks_prior).astype(np.float32) * 0.01
        positions = jnp.array([[500.0, 500.0]], dtype=jnp.float32)

        model = PhysicalPolychromaticField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            obs_pos=obs_pos,
            zks_prior=zks_prior,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes_param=n_zks_param,
            n_zks_total=n_zks_total,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        zks = model.compute_zernikes(positions)
        assert zks.shape == (1, n_zks_total, 1, 1)


# ---- Ground truth model tests ----


class TestGroundTruthSemiParametric:
    def test_nonparam_is_zeroed(self, zernike_maps, obscurations, key):
        model = create_ground_truth_semi_parametric(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        assert jnp.allclose(model.np_opd.alpha_mat, 0.0)

    def test_forward_pass(self, zernike_maps, obscurations, positions, packed_seds, key):
        model = create_ground_truth_semi_parametric(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            key=key,
        )
        psf_batch, opd = model([positions, packed_seds])
        assert psf_batch.shape == (BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM)
        assert opd.shape == (BATCH_SIZE, WFE_DIM, WFE_DIM)

    def test_coeff_mat_initialization(self, zernike_maps, obscurations, key):
        n_poly = int((D_MAX + 1) * (D_MAX + 2) / 2)
        custom_coeff = np.ones((N_ZERNIKES, n_poly), dtype=np.float32) * 0.42
        model = create_ground_truth_semi_parametric(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
            n_zernikes=N_ZERNIKES,
            d_max=D_MAX,
            d_max_nonparam=D_MAX_NONPARAM,
            x_lims=X_LIMS,
            y_lims=Y_LIMS,
            coeff_mat=custom_coeff,
            key=key,
        )
        assert jnp.allclose(model.poly_field.coeff_mat, 0.42, atol=1e-6)


class TestGroundTruthPhysicalField:
    @pytest.fixture
    def obs_pos(self):
        return np.array(
            [[500.0, 500.0], [200.0, 800.0], [800.0, 200.0], [100.0, 100.0]],
            dtype=np.float32,
        )

    @pytest.fixture
    def zks_prior(self):
        return np.random.RandomState(0).randn(4, N_ZERNIKES).astype(np.float32) * 0.01

    @pytest.mark.filterwarnings("ignore:A JAX array is being set as static")
    def test_forward_pass(
        self, zernike_maps, obscurations, positions, packed_seds, obs_pos, zks_prior
    ):
        model = GroundTruthPhysicalField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            obs_pos=obs_pos,
            zks_prior=zks_prior,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
        )
        psf_batch, opd = model([positions, packed_seds])
        assert psf_batch.shape == (BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM)
        assert opd.shape == (BATCH_SIZE, WFE_DIM, WFE_DIM)
        assert jnp.all(psf_batch >= 0)

    @pytest.mark.filterwarnings("ignore:A JAX array is being set as static")
    def test_predict_opd(self, zernike_maps, obscurations, positions, obs_pos, zks_prior):
        model = GroundTruthPhysicalField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            obs_pos=obs_pos,
            zks_prior=zks_prior,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
        )
        opd = model.predict_opd(positions)
        assert opd.shape == (BATCH_SIZE, WFE_DIM, WFE_DIM)

    @pytest.mark.filterwarnings("ignore:A JAX array is being set as static")
    def test_only_physical_prior(
        self, zernike_maps, obscurations, positions, obs_pos, zks_prior
    ):
        """GroundTruth physical field should only use the physical prior."""
        model = GroundTruthPhysicalField(
            zernike_maps=zernike_maps,
            obscurations=obscurations,
            obs_pos=obs_pos,
            zks_prior=zks_prior,
            output_Q=OUTPUT_Q,
            output_dim=OUTPUT_DIM,
        )
        # The Zernike coefficients should come purely from the physical layer
        zks_computed = model.compute_zernikes(positions)
        zks_from_layer = model.physical_layer(positions)
        assert jnp.allclose(zks_computed, zks_from_layer, atol=1e-6)


# ---- Registry tests ----


class TestModelRegistry:
    def test_parametric_registered(self):
        from wavediff_jax.models.registry import PSF_FACTORY
        assert "poly" in PSF_FACTORY

    def test_semi_parametric_registered(self):
        from wavediff_jax.models.registry import PSF_FACTORY
        assert "semi-param" in PSF_FACTORY

    def test_physical_poly_registered(self):
        from wavediff_jax.models.registry import PSF_FACTORY
        assert "physical-poly" in PSF_FACTORY

    def test_ground_truth_semi_param_registered(self):
        from wavediff_jax.models.registry import PSF_FACTORY
        assert "ground-truth-semi-param" in PSF_FACTORY

    def test_ground_truth_physical_poly_registered(self):
        from wavediff_jax.models.registry import PSF_FACTORY
        assert "ground-truth-physical-poly" in PSF_FACTORY
