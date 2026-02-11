"""Functional training loop for PSF models using Equinox + Optax.

Replaces the Keras ``model.compile()``/``model.fit()`` approach with explicit
functional gradient descent. The core abstraction uses ``eqx.partition`` /
``eqx.combine`` for trainability control, enabling Block Coordinate Descent
(BCD) training cycles.

Port of ``extern/wf-psf/src/wf_psf/training/train_utils.py`` lines 578-807.

:Authors: WaveDiff-JAX contributors
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import logging

from wavediff_jax.training.losses import total_loss as _total_loss
from wavediff_jax.training.callbacks import l1_schedule_rule
from wavediff_jax.training.train_utils import configure_optimizer
from wavediff_jax.models.layers import (
    PolynomialZernikeField,
    NonParametricPolynomialOPD,
    NonParametricMCCDOPD,
    NonParametricGraphOPD,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter functions for BCD trainability control
# ---------------------------------------------------------------------------


def param_filter(model):
    """Return a boolean pytree where only parametric leaves are ``True``.

    Marks the ``coeff_mat`` of :class:`PolynomialZernikeField` as trainable.
    Everything else is frozen.

    Parameters
    ----------
    model : eqx.Module

    Returns
    -------
    pytree of bool
    """
    filter_spec = jax.tree.map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda m: m.poly_field.coeff_mat, filter_spec, True
    )
    return filter_spec


def nonparam_filter(model):
    """Return a boolean pytree where only non-parametric leaves are ``True``.

    Marks ``S_mat`` and ``alpha_mat`` (for the "poly" variant) or
    ``S_poly``, ``S_graph``, ``alpha_poly``, ``alpha_graph`` (for "mccd"/"graph")
    as trainable. Everything else is frozen.

    Parameters
    ----------
    model : eqx.Module

    Returns
    -------
    pytree of bool
    """
    filter_spec = jax.tree.map(lambda _: False, model)
    np_opd = model.np_opd

    if isinstance(np_opd, NonParametricPolynomialOPD):
        filter_spec = eqx.tree_at(
            lambda m: m.np_opd.S_mat, filter_spec, True
        )
        filter_spec = eqx.tree_at(
            lambda m: m.np_opd.alpha_mat, filter_spec, True
        )
    elif isinstance(np_opd, (NonParametricMCCDOPD, NonParametricGraphOPD)):
        if hasattr(np_opd, "S_poly"):
            filter_spec = eqx.tree_at(
                lambda m: m.np_opd.S_poly, filter_spec, True
            )
        if hasattr(np_opd, "S_graph"):
            filter_spec = eqx.tree_at(
                lambda m: m.np_opd.S_graph, filter_spec, True
            )
        if hasattr(np_opd, "alpha_poly"):
            filter_spec = eqx.tree_at(
                lambda m: m.np_opd.alpha_poly, filter_spec, True
            )
        if hasattr(np_opd, "alpha_graph"):
            filter_spec = eqx.tree_at(
                lambda m: m.np_opd.alpha_graph, filter_spec, True
            )
    return filter_spec


def complete_filter(model):
    """Return a boolean pytree where both parametric and non-parametric leaves
    are ``True``.

    Parameters
    ----------
    model : eqx.Module

    Returns
    -------
    pytree of bool
    """
    p = param_filter(model)
    np_ = nonparam_filter(model)
    return jax.tree.map(lambda a, b: a or b, p, np_)


# ---------------------------------------------------------------------------
# Single gradient step
# ---------------------------------------------------------------------------


def make_step(model, opt_state, optimizer, batch, loss_fn, filter_spec):
    """Perform a single gradient step with trainability filtering.

    Parameters
    ----------
    model : eqx.Module
        Current model.
    opt_state : optax state
        Optimizer state.
    optimizer : optax.GradientTransformation
        Optimizer.
    batch : tuple
        ``(positions, packed_seds, targets, masks, sample_weight)``
    loss_fn : callable
        Loss function with signature ``loss_fn(model, positions, packed_seds,
        targets, masks, sample_weight) -> scalar``.
    filter_spec : pytree of bool
        Boolean pytree from a filter function (True = trainable).

    Returns
    -------
    new_model : eqx.Module
    new_opt_state : optax state
    loss : jnp.ndarray (scalar)
    """
    diff_model, static_model = eqx.partition(model, filter_spec)

    @eqx.filter_value_and_grad
    def compute_loss(diff_model):
        full_model = eqx.combine(diff_model, static_model)
        positions, packed_seds, targets, masks, sample_weight = batch
        return loss_fn(full_model, positions, packed_seds, targets, masks, sample_weight)

    loss, grads = compute_loss(diff_model)
    updates, new_opt_state = optimizer.update(grads, opt_state, diff_model)
    new_diff_model = eqx.apply_updates(diff_model, updates)
    new_model = eqx.combine(new_diff_model, static_model)
    return new_model, new_opt_state, loss


# ---------------------------------------------------------------------------
# Epoch-level training
# ---------------------------------------------------------------------------


def train_epoch(model, opt_state, optimizer, data, batch_size, loss_fn, filter_spec, key):
    """Train for one epoch: shuffle data and iterate over mini-batches.

    Parameters
    ----------
    model : eqx.Module
    opt_state : optax state
    optimizer : optax.GradientTransformation
    data : dict
        Must contain keys ``'positions'``, ``'packed_seds'``, ``'targets'``,
        ``'masks'``. Optionally ``'sample_weight'``.
    batch_size : int
    loss_fn : callable
    filter_spec : pytree of bool
    key : jax.random.PRNGKey

    Returns
    -------
    model : eqx.Module
    opt_state : optax state
    epoch_loss : float
    key : jax.random.PRNGKey
    """
    n_samples = data["positions"].shape[0]
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, n_samples)

    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, n_samples, batch_size):
        idx = perm[i : i + batch_size]
        batch = (
            data["positions"][idx],
            data["packed_seds"][idx],
            data["targets"][idx],
            data["masks"][idx],
            data["sample_weight"][idx] if "sample_weight" in data else None,
        )
        model, opt_state, loss = make_step(
            model, opt_state, optimizer, batch, loss_fn, filter_spec
        )
        epoch_loss += float(loss)
        n_batches += 1

    return model, opt_state, epoch_loss / max(n_batches, 1), key


# ---------------------------------------------------------------------------
# Multi-cycle BCD training
# ---------------------------------------------------------------------------


def _has_nonparam(model):
    """Check whether the model has a non-parametric OPD component."""
    return hasattr(model, "np_opd")


def _get_filter_for_cycle(cycle_def, model):
    """Return the appropriate filter spec for a given cycle definition.

    Parameters
    ----------
    cycle_def : str
        One of ``'parametric'``, ``'non-parametric'``, ``'complete'``,
        ``'only-parametric'``, ``'only-non-parametric'``.
    model : eqx.Module

    Returns
    -------
    tuple of (filter_spec_param, filter_spec_nonparam)
        Either may be None if the cycle does not include that phase.
    """
    has_np = _has_nonparam(model)

    if cycle_def == "parametric":
        return param_filter(model), nonparam_filter(model) if has_np else None
    elif cycle_def == "non-parametric":
        if not has_np:
            raise ValueError("Cannot run non-parametric cycle on a model without np_opd")
        return None, nonparam_filter(model)
    elif cycle_def == "complete":
        return param_filter(model), nonparam_filter(model) if has_np else None
    elif cycle_def == "only-parametric":
        return param_filter(model), None
    elif cycle_def == "only-non-parametric":
        if not has_np:
            raise ValueError("Cannot run only-non-parametric cycle on a model without np_opd")
        return None, nonparam_filter(model)
    else:
        raise ValueError(f"Unknown cycle_def: {cycle_def}")


def general_train_cycle(
    model,
    train_data,
    val_data,
    training_hparams,
    key,
):
    """Multi-cycle Block Coordinate Descent (BCD) training loop.

    Alternates between training parametric and non-parametric model parts.

    Parameters
    ----------
    model : eqx.Module
        The PSF model (e.g., ``SemiParametricField``).
    train_data : dict
        Training data dict with keys: ``'positions'``, ``'packed_seds'``,
        ``'targets'``, ``'masks'``, and optionally ``'sample_weight'``.
    val_data : dict or None
        Validation data dict (same keys). Currently used only for logging.
    training_hparams : dict
        Training hyperparameters. Expected keys:

        - ``'n_epochs_param'`` : int -- epochs for parametric phase
        - ``'n_epochs_nonparam'`` : int -- epochs for non-parametric phase
        - ``'lr_param'`` : float -- learning rate for parametric phase
        - ``'lr_nonparam'`` : float -- learning rate for non-parametric phase
        - ``'batch_size'`` : int
        - ``'l2_param'`` : float -- OPD L2 regularization
        - ``'l1_rate'`` : float -- Lp sparsity rate (0 to disable)
        - ``'cycle_def'`` : str -- one of 'parametric', 'non-parametric',
          'complete', 'only-parametric', 'only-non-parametric'
        - ``'first_run'`` : bool -- whether this is the first BCD iteration
    key : jax.random.PRNGKey

    Returns
    -------
    model : eqx.Module
        Trained model.
    history : dict
        Training history with keys ``'param_losses'`` and
        ``'nonparam_losses'`` (lists of per-epoch losses).
    key : jax.random.PRNGKey
    """
    n_epochs_param = training_hparams.get("n_epochs_param", 20)
    n_epochs_nonparam = training_hparams.get("n_epochs_nonparam", 100)
    lr_param = training_hparams.get("lr_param", 1e-2)
    lr_nonparam = training_hparams.get("lr_nonparam", 1.0)
    batch_size = training_hparams.get("batch_size", 32)
    l2_param = training_hparams.get("l2_param", 0.0)
    l1_rate = training_hparams.get("l1_rate", 0.0)
    cycle_def = training_hparams.get("cycle_def", "complete")
    first_run = training_hparams.get("first_run", False)

    history = {"param_losses": [], "nonparam_losses": []}

    # Build loss function (closed over regularization params; l1_rate may change)
    current_l1_rate = l1_rate

    def make_loss_fn(l1_r):
        def loss_fn(m, pos, seds, tgt, msk, sw):
            return _total_loss(m, pos, seds, tgt, msk, sw, l2_param=l2_param, l1_rate=l1_r)
        return loss_fn

    filter_param, filter_nonparam = _get_filter_for_cycle(cycle_def, model)

    # ---- First-run logic for non-parametric zeroing ----
    has_np = _has_nonparam(model)

    if first_run and filter_param is not None and has_np:
        # Zero out non-parametric contribution (alpha -> 0)
        from wavediff_jax.models.semiparametric import set_alpha_zero
        model = set_alpha_zero(model)

    if cycle_def == "only-parametric" and not first_run and has_np:
        from wavediff_jax.models.semiparametric import set_alpha_zero
        model = set_alpha_zero(model)

    # ---- Parametric phase ----
    if filter_param is not None:
        logger.info("Starting parametric training phase (%d epochs)...", n_epochs_param)
        optimizer_param = configure_optimizer(lr_param)
        opt_state_param = optimizer_param.init(
            eqx.partition(model, filter_param)[0]
        )

        loss_fn = make_loss_fn(current_l1_rate)
        for epoch in range(n_epochs_param):
            model, opt_state_param, epoch_loss, key = train_epoch(
                model, opt_state_param, optimizer_param, train_data,
                batch_size, loss_fn, filter_param, key,
            )
            history["param_losses"].append(epoch_loss)
            if epoch % 10 == 0 or epoch == n_epochs_param - 1:
                logger.info("  Param epoch %d/%d  loss=%.6e", epoch + 1, n_epochs_param, epoch_loss)

    # ---- Non-parametric phase ----
    if filter_nonparam is not None:
        # first_run: restore non-param to identity
        if first_run:
            from wavediff_jax.models.semiparametric import set_alpha_identity
            model = set_alpha_identity(model)

        if cycle_def == "only-non-parametric":
            # Zero out parametric part
            model = eqx.tree_at(
                lambda m: m.poly_field.coeff_mat,
                model,
                jnp.zeros_like(model.poly_field.coeff_mat),
            )

        logger.info("Starting non-parametric training phase (%d epochs)...", n_epochs_nonparam)
        # Recompute filter_nonparam in case model structure changed
        filter_nonparam = nonparam_filter(model)
        optimizer_nonparam = configure_optimizer(lr_nonparam)
        opt_state_nonparam = optimizer_nonparam.init(
            eqx.partition(model, filter_nonparam)[0]
        )

        for epoch in range(n_epochs_nonparam):
            # L1 schedule
            current_l1_rate = l1_schedule_rule(epoch, current_l1_rate)
            loss_fn = make_loss_fn(current_l1_rate)

            model, opt_state_nonparam, epoch_loss, key = train_epoch(
                model, opt_state_nonparam, optimizer_nonparam, train_data,
                batch_size, loss_fn, filter_nonparam, key,
            )
            history["nonparam_losses"].append(epoch_loss)
            if epoch % 10 == 0 or epoch == n_epochs_nonparam - 1:
                logger.info(
                    "  Non-param epoch %d/%d  loss=%.6e",
                    epoch + 1, n_epochs_nonparam, epoch_loss,
                )

    return model, history, key
