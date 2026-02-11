"""Tests for metrics module: residuals, RMSE, OPD, polychromatic, shape, and interface."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wavediff_jax.utils.math_utils import generate_zernike_maps_3d
from wavediff_jax.models.parametric import ParametricPSFFieldModel
from wavediff_jax.models.semiparametric import SemiParametricField, set_alpha_zero
from wavediff_jax.metrics.metrics import (
    compute_residuals,
    compute_rmse,
    compute_psf_images,
    compute_poly_metric,
    compute_mono_metric,
    compute_opd_metrics,
    compute_shape_metrics,
    HAS_GALSIM,
)
from wavediff_jax.metrics.metrics_interface import (
    MetricsParamsHandler,
    evaluate_model,
)


# ---- Shared constants ----

N_ZERNIKES = 15
WFE_DIM = 32
OUTPUT_DIM = 8
OUTPUT_Q = 1
BATCH_SIZE = 4
N_WAVELENGTHS = 3
X_LIMS = [0.0, 1e3]
Y_LIMS = [0.0, 1e3]
D_MAX = 2
D_MAX_NONPARAM = 2


# ---- Fixtures ----


@pytest.fixture
def zernike_maps():
    return generate_zernike_maps_3d(N_ZERNIKES, WFE_DIM)


@pytest.fixture
def obscurations():
    y, x = np.mgrid[-1:1:complex(WFE_DIM), -1:1:complex(WFE_DIM)]
    r = np.sqrt(x ** 2 + y ** 2)
    mask = (r <= 1.0).astype(np.float32)
    return jnp.array(mask + 0j, dtype=jnp.complex64)


@pytest.fixture
def positions():
    return jnp.array(
        [[500.0, 500.0], [200.0, 800.0], [800.0, 200.0], [100.0, 900.0]],
        dtype=jnp.float32,
    )


@pytest.fixture
def packed_seds():
    phase_N = float(WFE_DIM * 2)
    seds = np.zeros((BATCH_SIZE, N_WAVELENGTHS, 3), dtype=np.float32)
    for i in range(N_WAVELENGTHS):
        seds[:, i, 0] = phase_N
        seds[:, i, 1] = 0.5 + 0.1 * i
        seds[:, i, 2] = 1.0 / N_WAVELENGTHS
    return jnp.array(seds)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def parametric_model(zernike_maps, obscurations, key):
    return ParametricPSFFieldModel(
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


@pytest.fixture
def gt_model(zernike_maps, obscurations, key):
    """Ground truth model: a semi-parametric model with alpha zeroed."""
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
    return set_alpha_zero(model)


# ---- Tests for low-level helpers ----


class TestComputeResiduals:
    def test_shape(self):
        preds = np.random.randn(5, 8, 8).astype(np.float32)
        targets = np.random.randn(5, 8, 8).astype(np.float32)
        residuals = compute_residuals(preds, targets)
        assert residuals.shape == (5,)

    def test_zero_residuals_for_identical(self):
        data = np.random.randn(3, 8, 8).astype(np.float32)
        residuals = compute_residuals(data, data)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-7)

    def test_positive_residuals(self):
        preds = np.zeros((3, 4, 4), dtype=np.float32)
        targets = np.ones((3, 4, 4), dtype=np.float32)
        residuals = compute_residuals(preds, targets)
        assert np.all(residuals > 0)

    def test_jax_array_input(self):
        preds = jnp.ones((2, 4, 4))
        targets = jnp.zeros((2, 4, 4))
        residuals = compute_residuals(preds, targets)
        assert residuals.shape == (2,)


class TestComputeRMSE:
    def test_scalar_output(self):
        preds = np.random.randn(5, 8, 8)
        targets = np.random.randn(5, 8, 8)
        rmse = compute_rmse(preds, targets)
        assert isinstance(rmse, float)

    def test_zero_for_identical(self):
        data = np.random.randn(3, 8, 8)
        rmse = compute_rmse(data, data)
        assert rmse == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        preds = np.zeros((1, 2, 2))
        targets = np.ones((1, 2, 2))
        rmse = compute_rmse(preds, targets)
        assert rmse == pytest.approx(1.0, abs=1e-10)

    def test_jax_array_input(self):
        rmse = compute_rmse(jnp.zeros((3,)), jnp.ones((3,)))
        assert isinstance(rmse, float)
        assert rmse > 0


# ---- Tests for compute_psf_images ----


class TestComputePSFImages:
    def test_unbatched(self, parametric_model, positions, packed_seds):
        images = compute_psf_images(parametric_model, positions, packed_seds)
        assert images.shape == (BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM)

    def test_batched(self, parametric_model, positions, packed_seds):
        images = compute_psf_images(
            parametric_model, positions, packed_seds, batch_size=2
        )
        assert images.shape == (BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM)

    def test_batched_matches_unbatched(self, parametric_model, positions, packed_seds):
        images_full = compute_psf_images(parametric_model, positions, packed_seds)
        images_batched = compute_psf_images(
            parametric_model, positions, packed_seds, batch_size=2
        )
        np.testing.assert_allclose(images_full, images_batched, atol=1e-5)


# ---- Tests for polychromatic metric ----


class TestComputePolyMetric:
    def test_returns_four_values(self, parametric_model, gt_model, positions, packed_seds):
        result = compute_poly_metric(
            model=parametric_model,
            gt_model=gt_model,
            positions=positions,
            packed_seds=packed_seds,
            batch_size=BATCH_SIZE,
        )
        assert len(result) == 4
        rmse, rel_rmse, std_rmse, std_rel_rmse = result
        assert isinstance(rmse, float)
        assert isinstance(rel_rmse, float)
        assert isinstance(std_rmse, float)
        assert isinstance(std_rel_rmse, float)

    def test_self_comparison_zero_rmse(self, parametric_model, positions, packed_seds):
        """Comparing a model to itself should give zero RMSE."""
        rmse, rel_rmse, std_rmse, std_rel_rmse = compute_poly_metric(
            model=parametric_model,
            gt_model=parametric_model,
            positions=positions,
            packed_seds=packed_seds,
            batch_size=BATCH_SIZE,
        )
        assert rmse == pytest.approx(0.0, abs=1e-6)

    def test_with_precomputed_stars(self, parametric_model, gt_model, positions, packed_seds):
        """Test using precomputed ground-truth stars."""
        gt_stars = compute_psf_images(gt_model, positions, packed_seds)
        rmse, rel_rmse, std_rmse, std_rel_rmse = compute_poly_metric(
            model=parametric_model,
            gt_model=None,
            positions=positions,
            packed_seds=packed_seds,
            gt_stars=gt_stars,
            batch_size=BATCH_SIZE,
        )
        assert rmse >= 0.0
        assert rel_rmse >= 0.0


# ---- Tests for OPD metrics ----


class TestComputeOPDMetrics:
    def test_returns_four_values(self, parametric_model, gt_model, positions, obscurations):
        result = compute_opd_metrics(
            model=parametric_model,
            gt_model=gt_model,
            positions=positions,
            obscurations=obscurations,
        )
        assert len(result) == 4
        rmse, rel_rmse, rmse_std, rel_rmse_std = result
        assert isinstance(rmse, float)
        assert isinstance(rel_rmse, float)

    def test_self_comparison(self, parametric_model, positions, obscurations):
        """Comparing a model to itself should give zero OPD RMSE."""
        rmse, rel_rmse, rmse_std, rel_rmse_std = compute_opd_metrics(
            model=parametric_model,
            gt_model=parametric_model,
            positions=positions,
            obscurations=obscurations,
        )
        assert rmse == pytest.approx(0.0, abs=1e-6)

    def test_known_opd_difference(self, zernike_maps, obscurations, positions, key):
        """Two models with different coefficients should have nonzero OPD RMSE."""
        import equinox as eqx

        model1 = ParametricPSFFieldModel(
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
        # Create a different model by changing coefficients
        new_coeff = model1.poly_field.coeff_mat + 0.1
        model2 = eqx.tree_at(lambda m: m.poly_field.coeff_mat, model1, new_coeff)

        rmse, rel_rmse, _, _ = compute_opd_metrics(
            model=model1,
            gt_model=model2,
            positions=positions,
            obscurations=obscurations,
        )
        assert rmse > 0.0
        assert rel_rmse > 0.0

    def test_batched_opd(self, parametric_model, gt_model, positions, obscurations):
        """Test that batching gives consistent results."""
        r1 = compute_opd_metrics(
            model=parametric_model,
            gt_model=gt_model,
            positions=positions,
            obscurations=obscurations,
            batch_size=2,
        )
        r2 = compute_opd_metrics(
            model=parametric_model,
            gt_model=gt_model,
            positions=positions,
            obscurations=obscurations,
            batch_size=BATCH_SIZE,
        )
        assert r1[0] == pytest.approx(r2[0], rel=1e-5)


# ---- Tests for monochromatic metric ----


class TestComputeMonoMetric:
    def test_returns_lists(self, parametric_model, gt_model, positions):
        lambda_list = [0.6, 0.7]
        result = compute_mono_metric(
            model=parametric_model,
            gt_model=gt_model,
            positions=positions,
            lambda_list=lambda_list,
            batch_size=BATCH_SIZE,
        )
        assert len(result) == 4
        rmse_lda, rel_rmse_lda, std_rmse_lda, std_rel_rmse_lda = result
        assert len(rmse_lda) == 2
        assert len(rel_rmse_lda) == 2

    def test_self_comparison(self, parametric_model, positions):
        lambda_list = [0.65]
        rmse_lda, _, _, _ = compute_mono_metric(
            model=parametric_model,
            gt_model=parametric_model,
            positions=positions,
            lambda_list=lambda_list,
            batch_size=BATCH_SIZE,
        )
        assert rmse_lda[0] == pytest.approx(0.0, abs=1e-6)


# ---- Tests for shape metrics (galsim-dependent) ----


class TestComputeShapeMetrics:
    @pytest.fixture
    def gaussian_psfs(self):
        """Create synthetic Gaussian PSFs for shape testing."""
        n_samples = 8
        dim = 64
        y, x = np.mgrid[:dim, :dim] - dim / 2.0
        sigma = 3.0
        psf = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        psf = psf / psf.sum()
        return np.stack([psf] * n_samples).astype(np.float64)

    @pytest.mark.skipif(not HAS_GALSIM, reason="galsim not available")
    def test_self_comparison(self, gaussian_psfs):
        result = compute_shape_metrics(gaussian_psfs, gaussian_psfs)
        assert result["pix_rmse"] == pytest.approx(0.0, abs=1e-10)
        assert result["rmse_e1"] == pytest.approx(0.0, abs=1e-10)
        assert result["rmse_e2"] == pytest.approx(0.0, abs=1e-10)
        assert result["rmse_R2_meanR2"] == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.skipif(not HAS_GALSIM, reason="galsim not available")
    def test_result_dict_keys(self, gaussian_psfs):
        result = compute_shape_metrics(gaussian_psfs, gaussian_psfs)
        expected_keys = {
            "pred_e1_HSM", "pred_e2_HSM", "pred_R2_HSM",
            "gt_pred_e1_HSM", "gt_pred_e2_HSM", "gt_pred_R2_HSM",
            "rmse_e1", "std_rmse_e1", "rel_rmse_e1", "std_rel_rmse_e1",
            "rmse_e2", "std_rmse_e2", "rel_rmse_e2", "std_rel_rmse_e2",
            "rmse_R2_meanR2", "std_rmse_R2_meanR2",
            "pix_rmse", "pix_rmse_std", "rel_pix_rmse", "rel_pix_rmse_std",
        }
        assert set(result.keys()) == expected_keys

    @pytest.mark.skipif(not HAS_GALSIM, reason="galsim not available")
    def test_nonzero_shape_difference(self, gaussian_psfs):
        """Slightly shifted Gaussians should show nonzero e1/e2 differences."""
        dim = gaussian_psfs.shape[1]
        y, x = np.mgrid[:dim, :dim] - dim / 2.0

        # Elliptical PSFs (stretched in x)
        sigma_x, sigma_y = 3.5, 2.5
        elliptical = np.exp(-(x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)))
        elliptical = elliptical / elliptical.sum()
        elliptical_psfs = np.stack([elliptical] * gaussian_psfs.shape[0]).astype(np.float64)

        result = compute_shape_metrics(elliptical_psfs, gaussian_psfs)
        # There should be a detectable shape difference
        assert result["rmse_e1"] > 0 or result["rmse_e2"] > 0

    @pytest.mark.skipif(not HAS_GALSIM, reason="galsim not available")
    def test_pixel_rmse_nonzero_for_different(self, gaussian_psfs):
        noisy = gaussian_psfs + np.random.RandomState(0).randn(*gaussian_psfs.shape) * 1e-4
        result = compute_shape_metrics(noisy, gaussian_psfs)
        assert result["pix_rmse"] > 0

    def test_import_error_without_galsim(self):
        """compute_shape_metrics should raise ImportError if galsim is missing."""
        import wavediff_jax.metrics.metrics as mm

        original = mm.HAS_GALSIM
        try:
            mm.HAS_GALSIM = False
            with pytest.raises(ImportError, match="galsim"):
                compute_shape_metrics(np.zeros((1, 8, 8)), np.zeros((1, 8, 8)))
        finally:
            mm.HAS_GALSIM = original


# ---- Tests for metrics interface ----


class TestMetricsInterface:
    def test_evaluate_poly_metric(
        self, parametric_model, gt_model, positions, packed_seds
    ):
        handler = MetricsParamsHandler(
            metrics_params=None,
            trained_model=None,
        )
        dataset = {
            "positions": positions,
            "packed_seds": packed_seds,
        }
        result = handler.evaluate_metrics_polychromatic_lowres(
            psf_model=parametric_model,
            gt_model=gt_model,
            dataset=dataset,
        )
        assert "rmse" in result
        assert "rel_rmse" in result
        assert isinstance(result["rmse"], float)

    def test_evaluate_opd_metric(
        self, parametric_model, gt_model, positions
    ):
        handler = MetricsParamsHandler(
            metrics_params=None,
            trained_model=None,
        )
        dataset = {
            "positions": positions,
        }
        result = handler.evaluate_metrics_opd(
            psf_model=parametric_model,
            gt_model=gt_model,
            dataset=dataset,
        )
        assert "rmse_opd" in result
        assert isinstance(result["rmse_opd"], float)

    def test_evaluate_model_orchestration(
        self, parametric_model, gt_model, positions, packed_seds
    ):
        dataset = {
            "positions": positions,
            "packed_seds": packed_seds,
        }
        all_metrics = evaluate_model(
            psf_model=parametric_model,
            gt_model=gt_model,
            train_dataset=dataset,
            test_dataset=dataset,
        )
        assert "train_metrics" in all_metrics
        assert "test_metrics" in all_metrics
        assert all_metrics["test_metrics"]["poly_metric"] is not None
        assert all_metrics["test_metrics"]["mono_metric"] is None
        assert all_metrics["test_metrics"]["opd_metric"] is None

    def test_evaluate_model_with_flags(
        self, parametric_model, gt_model, positions, packed_seds
    ):
        dataset = {
            "positions": positions,
            "packed_seds": packed_seds,
        }
        all_metrics = evaluate_model(
            psf_model=parametric_model,
            gt_model=gt_model,
            train_dataset=dataset,
            test_dataset=dataset,
            eval_flags={
                "poly_metric": True,
                "mono_metric": False,
                "opd_metric": True,
                "shape_results_dict": False,
            },
        )
        assert all_metrics["test_metrics"]["poly_metric"] is not None
        assert all_metrics["test_metrics"]["opd_metric"] is not None
        assert all_metrics["test_metrics"]["mono_metric"] is None
