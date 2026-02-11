"""Tests for the WaveDiff-JAX training module.

Covers losses, trainer mechanics, callbacks, and BCD training cycles.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import pytest
import tempfile
import os

from wavediff_jax.utils.math_utils import generate_zernike_maps_3d
from wavediff_jax.models.parametric import ParametricPSFFieldModel
from wavediff_jax.models.semiparametric import (
    SemiParametricField,
    set_alpha_zero,
    set_alpha_identity,
)
from wavediff_jax.training.losses import (
    mse_loss,
    masked_mse_loss,
    weighted_masked_mse_loss,
    l2_opd_regularization,
    lp_regularization,
    total_loss,
)
from wavediff_jax.training.trainer import (
    make_step,
    train_epoch,
    general_train_cycle,
    param_filter,
    nonparam_filter,
    complete_filter,
)
from wavediff_jax.training.callbacks import (
    save_checkpoint,
    load_checkpoint,
    l1_schedule_rule,
)
from wavediff_jax.training.train_utils import (
    configure_optimizer,
)

# ---- Test dimensions (small for speed) ----

N_ZERNIKES = 10
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
    """Simple circular pupil (complex) for testing."""
    y, x = np.mgrid[-1:1:complex(WFE_DIM), -1:1:complex(WFE_DIM)]
    r = np.sqrt(x ** 2 + y ** 2)
    mask = (r <= 1.0).astype(np.float32)
    return jnp.array(mask + 0j, dtype=jnp.complex64)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def positions():
    return jnp.array(
        [[500.0, 500.0], [200.0, 800.0], [800.0, 200.0], [100.0, 900.0]],
        dtype=jnp.float32,
    )


@pytest.fixture
def packed_seds():
    """Fake packed SED data: (batch, n_wavelengths, 3)."""
    phase_N = float(WFE_DIM * 2)
    seds = np.zeros((BATCH_SIZE, N_WAVELENGTHS, 3), dtype=np.float32)
    for i in range(N_WAVELENGTHS):
        seds[:, i, 0] = phase_N
        seds[:, i, 1] = 0.5 + 0.1 * i
        seds[:, i, 2] = 1.0 / N_WAVELENGTHS
    return jnp.array(seds)


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
def semiparam_model(zernike_maps, obscurations, key):
    return SemiParametricField(
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


@pytest.fixture
def train_data(parametric_model, positions, packed_seds, key):
    """Generate synthetic training data from the model itself (+ noise)."""
    psf_batch, opd_maps = parametric_model([positions, packed_seds])
    # Add a bit of noise as targets
    noise_key = jax.random.PRNGKey(99)
    noise = 0.01 * jax.random.normal(noise_key, psf_batch.shape)
    targets = psf_batch + noise
    masks = jnp.zeros_like(targets)  # no masking
    return {
        "positions": positions,
        "packed_seds": packed_seds,
        "targets": targets,
        "masks": masks,
    }


@pytest.fixture
def train_data_semiparam(semiparam_model, positions, packed_seds):
    """Training data for semi-parametric model."""
    psf_batch, opd_maps = semiparam_model([positions, packed_seds])
    noise_key = jax.random.PRNGKey(99)
    noise = 0.01 * jax.random.normal(noise_key, psf_batch.shape)
    targets = psf_batch + noise
    masks = jnp.zeros_like(targets)
    return {
        "positions": positions,
        "packed_seds": packed_seds,
        "targets": targets,
        "masks": masks,
    }


# ---- Loss function tests ----


class TestMSELoss:
    def test_mse_loss_basic(self):
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.0, 2.0, 3.0])
        assert float(mse_loss(pred, target)) == pytest.approx(0.0)

    def test_mse_loss_nonzero(self):
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([2.0, 3.0, 4.0])
        # Each diff is 1.0, squared is 1.0, mean is 1.0
        assert float(mse_loss(pred, target)) == pytest.approx(1.0)

    def test_mse_loss_batch(self):
        pred = jnp.zeros((4, 8, 8))
        target = jnp.ones((4, 8, 8))
        assert float(mse_loss(pred, target)) == pytest.approx(1.0)


class TestMaskedMSELoss:
    def test_mask_convention_zero_include(self):
        """Mask=0 means include; mask=1 means exclude."""
        pred = jnp.ones((2, 4, 4))
        target = jnp.zeros((2, 4, 4))
        # All included
        mask_all_include = jnp.zeros((2, 4, 4))
        loss_all = float(masked_mse_loss(pred, target, mask_all_include))
        assert loss_all == pytest.approx(1.0)

    def test_mask_excludes_pixels(self):
        """Masked pixels should not contribute to the loss."""
        pred = jnp.ones((1, 4, 4))
        target = jnp.zeros((1, 4, 4))
        # Exclude all -> loss should be well-defined via included
        # Exclude half: top 2 rows excluded, bottom 2 included
        mask = jnp.zeros((1, 4, 4))
        mask = mask.at[:, :2, :].set(1.0)  # exclude top 2 rows
        loss = float(masked_mse_loss(pred, target, mask))
        # Included pixels have error 1.0, so loss should be 1.0
        assert loss == pytest.approx(1.0)

    def test_partial_mask(self):
        """Float mask values should act as weights."""
        pred = jnp.ones((1, 2, 2))
        target = jnp.zeros((1, 2, 2))
        # All pixels at 0.5 mask -> keep = 0.5
        mask = 0.5 * jnp.ones((1, 2, 2))
        loss = float(masked_mse_loss(pred, target, mask))
        # error * keep = 1.0 * 0.5 = 0.5 per pixel
        # mask_sum = 4 * 0.5 = 2.0
        # per_sample = (4 * 0.5) / 2.0 = 1.0
        assert loss == pytest.approx(1.0)


class TestWeightedMaskedMSELoss:
    def test_no_weight_matches_masked(self):
        pred = jnp.ones((2, 4, 4))
        target = jnp.zeros((2, 4, 4))
        mask = jnp.zeros((2, 4, 4))
        loss_masked = float(masked_mse_loss(pred, target, mask))
        loss_weighted = float(weighted_masked_mse_loss(pred, target, mask, None))
        assert loss_masked == pytest.approx(loss_weighted)

    def test_sample_weights_affect_loss(self):
        pred = jnp.ones((2, 4, 4))
        target = jnp.zeros((2, 4, 4))
        mask = jnp.zeros((2, 4, 4))
        # Weight first sample 2x, second 0x
        sw = jnp.array([2.0, 0.0])
        loss = float(weighted_masked_mse_loss(pred, target, mask, sw))
        # First sample: 1.0 * 2.0 = 2.0, second: 1.0 * 0.0 = 0.0
        # Mean: (2.0 + 0.0) / 2 = 1.0
        assert loss == pytest.approx(1.0)


class TestL2OPDRegularization:
    def test_zero_opd(self):
        opd = jnp.zeros((2, 32, 32))
        assert float(l2_opd_regularization(opd, 1.0)) == pytest.approx(0.0)

    def test_nonzero_opd(self):
        opd = jnp.ones((1, 2, 2))  # 4 elements, sum of squares = 4
        result = float(l2_opd_regularization(opd, 0.5))
        assert result == pytest.approx(2.0)

    def test_l2_param_scales(self):
        opd = jnp.ones((1, 2, 2))
        r1 = float(l2_opd_regularization(opd, 1.0))
        r2 = float(l2_opd_regularization(opd, 2.0))
        assert r2 == pytest.approx(2 * r1)


class TestLpRegularization:
    def test_zero_alpha(self):
        alpha = jnp.zeros((5, 5))
        assert float(lp_regularization(alpha)) == pytest.approx(0.0)

    def test_identity_alpha(self):
        alpha = jnp.eye(3)
        # |1|^1.1 = 1 for each of 3 diagonal elements, rest zero
        result = float(lp_regularization(alpha, p=1.1))
        assert result == pytest.approx(3.0)

    def test_p1_equals_l1(self):
        alpha = jnp.array([[-1.0, 2.0], [3.0, -4.0]])
        result = float(lp_regularization(alpha, p=1.0))
        assert result == pytest.approx(10.0)


# ---- Trainer tests ----


class TestMakeStep:
    def test_make_step_reduces_loss(self, parametric_model, train_data):
        """A single gradient step should reduce the loss."""
        model = parametric_model
        filter_spec = param_filter(model)

        optimizer = configure_optimizer(1e-3)
        diff_model, _ = eqx.partition(model, filter_spec)
        opt_state = optimizer.init(diff_model)

        def loss_fn(m, pos, seds, tgt, msk, sw):
            return total_loss(m, pos, seds, tgt, msk, sw)

        batch = (
            train_data["positions"],
            train_data["packed_seds"],
            train_data["targets"],
            train_data["masks"],
            None,
        )

        # Compute initial loss
        loss_before = float(loss_fn(model, *batch))

        # Take a step
        new_model, new_opt_state, step_loss = make_step(
            model, opt_state, optimizer, batch, loss_fn, filter_spec
        )

        # Compute loss after step
        loss_after = float(loss_fn(new_model, *batch))

        assert jnp.isfinite(step_loss)
        assert loss_after < loss_before, (
            f"Loss should decrease: {loss_before:.6e} -> {loss_after:.6e}"
        )

    def test_make_step_returns_finite(self, parametric_model, train_data):
        model = parametric_model
        filter_spec = param_filter(model)
        optimizer = configure_optimizer(1e-3)
        diff_model, _ = eqx.partition(model, filter_spec)
        opt_state = optimizer.init(diff_model)

        def loss_fn(m, pos, seds, tgt, msk, sw):
            return total_loss(m, pos, seds, tgt, msk, sw)

        batch = (
            train_data["positions"],
            train_data["packed_seds"],
            train_data["targets"],
            train_data["masks"],
            None,
        )

        _, _, loss = make_step(model, opt_state, optimizer, batch, loss_fn, filter_spec)
        assert jnp.isfinite(loss)


class TestTrainEpoch:
    def test_train_epoch_finite_loss(self, parametric_model, train_data, key):
        model = parametric_model
        filter_spec = param_filter(model)
        optimizer = configure_optimizer(1e-3)
        diff_model, _ = eqx.partition(model, filter_spec)
        opt_state = optimizer.init(diff_model)

        def loss_fn(m, pos, seds, tgt, msk, sw):
            return total_loss(m, pos, seds, tgt, msk, sw)

        model_out, opt_state_out, epoch_loss, key_out = train_epoch(
            model, opt_state, optimizer, train_data, batch_size=2,
            loss_fn=loss_fn, filter_spec=filter_spec, key=key,
        )
        assert np.isfinite(epoch_loss)
        assert epoch_loss > 0

    def test_train_epoch_changes_model(self, parametric_model, train_data, key):
        model = parametric_model
        filter_spec = param_filter(model)
        optimizer = configure_optimizer(1e-3)
        diff_model, _ = eqx.partition(model, filter_spec)
        opt_state = optimizer.init(diff_model)

        def loss_fn(m, pos, seds, tgt, msk, sw):
            return total_loss(m, pos, seds, tgt, msk, sw)

        coeff_before = model.poly_field.coeff_mat.copy()

        model_out, _, _, _ = train_epoch(
            model, opt_state, optimizer, train_data, batch_size=2,
            loss_fn=loss_fn, filter_spec=filter_spec, key=key,
        )

        coeff_after = model_out.poly_field.coeff_mat
        assert not jnp.allclose(coeff_before, coeff_after), (
            "Coefficients should change after a training epoch"
        )


# ---- Filter function tests ----


class TestParamFilter:
    def test_param_filter_selects_coeff_mat(self, parametric_model):
        fspec = param_filter(parametric_model)
        # The coeff_mat leaf should be True
        leaves_spec = jax.tree.leaves(fspec)
        leaves_model = jax.tree.leaves(parametric_model)
        # Count True leaves
        n_true = sum(1 for v in leaves_spec if v is True)
        # Only coeff_mat should be trainable (1 leaf)
        assert n_true == 1

    def test_nonparam_filter_selects_S_and_alpha(self, semiparam_model):
        fspec = nonparam_filter(semiparam_model)
        leaves_spec = jax.tree.leaves(fspec)
        n_true = sum(1 for v in leaves_spec if v is True)
        # S_mat and alpha_mat -> 2 leaves
        assert n_true == 2

    def test_complete_filter_is_union(self, semiparam_model):
        p = param_filter(semiparam_model)
        np_ = nonparam_filter(semiparam_model)
        c = complete_filter(semiparam_model)
        p_true = sum(1 for v in jax.tree.leaves(p) if v is True)
        np_true = sum(1 for v in jax.tree.leaves(np_) if v is True)
        c_true = sum(1 for v in jax.tree.leaves(c) if v is True)
        assert c_true == p_true + np_true


# ---- Checkpoint tests ----


class TestCheckpoint:
    def test_checkpoint_roundtrip(self, parametric_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.eqx")
            save_checkpoint(parametric_model, path)

            # Create a fresh template model
            loaded = load_checkpoint(parametric_model, path)

            # All leaves should match
            leaves_orig = jax.tree.leaves(parametric_model)
            leaves_loaded = jax.tree.leaves(loaded)
            for lo, ll in zip(leaves_orig, leaves_loaded):
                if isinstance(lo, jnp.ndarray):
                    assert jnp.allclose(lo, ll), "Weights should match after roundtrip"

    def test_checkpoint_roundtrip_semiparam(self, semiparam_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "semiparam.eqx")
            save_checkpoint(semiparam_model, path)
            loaded = load_checkpoint(semiparam_model, path)

            assert jnp.allclose(
                loaded.poly_field.coeff_mat,
                semiparam_model.poly_field.coeff_mat,
            )
            assert jnp.allclose(
                loaded.np_opd.S_mat,
                semiparam_model.np_opd.S_mat,
            )
            assert jnp.allclose(
                loaded.np_opd.alpha_mat,
                semiparam_model.np_opd.alpha_mat,
            )


# ---- L1 schedule rule tests ----


class TestL1ScheduleRule:
    def test_epoch_0_no_change(self):
        assert l1_schedule_rule(0, 1.0) == 1.0

    def test_epoch_10_halves(self):
        assert l1_schedule_rule(10, 1.0) == pytest.approx(0.5)

    def test_epoch_20_halves(self):
        assert l1_schedule_rule(20, 0.5) == pytest.approx(0.25)

    def test_non_multiple_no_change(self):
        assert l1_schedule_rule(5, 1.0) == 1.0
        assert l1_schedule_rule(13, 0.5) == 0.5
        assert l1_schedule_rule(99, 0.25) == 0.25

    def test_cumulative_halving(self):
        """Simulate cumulative halving across multiple epochs."""
        rate = 1.0
        for epoch in range(31):
            rate = l1_schedule_rule(epoch, rate)
        # Should have halved at epochs 10, 20, 30 -> rate = 1/8
        assert rate == pytest.approx(0.125)


# ---- Optimizer config test ----


class TestConfigureOptimizer:
    def test_returns_optimizer(self):
        opt = configure_optimizer(1e-3)
        # optax optimizers are GradientTransformation named tuples
        assert hasattr(opt, "init")
        assert hasattr(opt, "update")


# ---- BCD cycle integration test ----


class TestBCDCycle:
    def test_bcd_cycle_parametric_only(self, parametric_model, train_data, key):
        """Run parametric-only training for a few epochs."""
        hparams = {
            "n_epochs_param": 3,
            "n_epochs_nonparam": 0,
            "lr_param": 1e-3,
            "lr_nonparam": 1.0,
            "batch_size": 2,
            "l2_param": 0.0,
            "l1_rate": 0.0,
            "cycle_def": "only-parametric",
            "first_run": False,
        }

        model_out, history, key_out = general_train_cycle(
            parametric_model, train_data, None, hparams, key
        )
        assert len(history["param_losses"]) == 3
        assert all(np.isfinite(l) for l in history["param_losses"])
        # Loss should generally decrease (or at least be finite)
        assert history["param_losses"][-1] <= history["param_losses"][0] * 1.5

    def test_bcd_cycle_complete_semiparam(
        self, semiparam_model, train_data_semiparam, key
    ):
        """Run 2 iterations of BCD (param + nonparam) on a semi-parametric model."""
        # First run
        hparams = {
            "n_epochs_param": 2,
            "n_epochs_nonparam": 2,
            "lr_param": 1e-3,
            "lr_nonparam": 1e-2,
            "batch_size": 2,
            "l2_param": 0.0,
            "l1_rate": 0.0,
            "cycle_def": "complete",
            "first_run": True,
        }

        model_out, history1, key = general_train_cycle(
            semiparam_model, train_data_semiparam, None, hparams, key
        )
        assert len(history1["param_losses"]) == 2
        assert len(history1["nonparam_losses"]) == 2
        assert all(np.isfinite(l) for l in history1["param_losses"])
        assert all(np.isfinite(l) for l in history1["nonparam_losses"])

        # Second run (not first_run)
        hparams["first_run"] = False
        model_out2, history2, key = general_train_cycle(
            model_out, train_data_semiparam, None, hparams, key
        )
        assert len(history2["param_losses"]) == 2
        assert len(history2["nonparam_losses"]) == 2

    def test_bcd_nonparam_only(self, semiparam_model, train_data_semiparam, key):
        """Run non-parametric only training."""
        hparams = {
            "n_epochs_param": 0,
            "n_epochs_nonparam": 3,
            "lr_param": 1e-3,
            "lr_nonparam": 1e-2,
            "batch_size": 2,
            "l2_param": 0.0,
            "l1_rate": 0.0,
            "cycle_def": "non-parametric",
            "first_run": True,
        }

        model_out, history, key_out = general_train_cycle(
            semiparam_model, train_data_semiparam, None, hparams, key
        )
        assert len(history["param_losses"]) == 0
        assert len(history["nonparam_losses"]) == 3
        assert all(np.isfinite(l) for l in history["nonparam_losses"])


# ---- Total loss integration test ----


class TestTotalLoss:
    def test_total_loss_parametric(self, parametric_model, positions, packed_seds):
        """Total loss should be finite and differentiable for parametric model."""
        targets = jnp.ones((BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM)) * 0.01
        masks = jnp.zeros((BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM))
        loss_val = total_loss(
            parametric_model, positions, packed_seds, targets, masks, None
        )
        assert jnp.isfinite(loss_val)
        assert float(loss_val) > 0

    def test_total_loss_with_l2(self, parametric_model, positions, packed_seds):
        targets = jnp.ones((BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM)) * 0.01
        masks = jnp.zeros((BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM))
        loss_no_reg = float(
            total_loss(parametric_model, positions, packed_seds, targets, masks, None, l2_param=0.0)
        )
        loss_with_reg = float(
            total_loss(parametric_model, positions, packed_seds, targets, masks, None, l2_param=1e-3)
        )
        assert loss_with_reg > loss_no_reg

    def test_total_loss_differentiable(self, parametric_model, positions, packed_seds):
        """Ensure we can compute gradients through total_loss."""
        targets = jnp.ones((BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM)) * 0.01
        masks = jnp.zeros((BATCH_SIZE, OUTPUT_DIM, OUTPUT_DIM))

        filter_spec = param_filter(parametric_model)
        diff_model, static_model = eqx.partition(parametric_model, filter_spec)

        @eqx.filter_value_and_grad
        def compute_loss(diff_m):
            m = eqx.combine(diff_m, static_model)
            return total_loss(m, positions, packed_seds, targets, masks, None)

        loss_val, grads = compute_loss(diff_model)
        assert jnp.isfinite(loss_val)
        # Check that gradient of coeff_mat is not all zeros
        grad_leaves = jax.tree.leaves(grads)
        has_nonzero_grad = any(
            isinstance(g, jnp.ndarray) and jnp.any(g != 0)
            for g in grad_leaves
        )
        assert has_nonzero_grad, "Gradients should be non-zero"
