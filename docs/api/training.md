# wavediff_jax.training

Functional training loop using Equinox and Optax.

## Trainer

### Filter Functions

::: wavediff_jax.training.trainer.param_filter

::: wavediff_jax.training.trainer.nonparam_filter

::: wavediff_jax.training.trainer.complete_filter

### Training Functions

::: wavediff_jax.training.trainer.make_step

::: wavediff_jax.training.trainer.train_epoch

::: wavediff_jax.training.trainer.general_train_cycle

## Loss Functions

::: wavediff_jax.training.losses.mse_loss

::: wavediff_jax.training.losses.masked_mse_loss

::: wavediff_jax.training.losses.weighted_masked_mse_loss

::: wavediff_jax.training.losses.l2_opd_regularization

::: wavediff_jax.training.losses.lp_regularization

::: wavediff_jax.training.losses.total_loss

## Callbacks

::: wavediff_jax.training.callbacks.save_checkpoint

::: wavediff_jax.training.callbacks.load_checkpoint

::: wavediff_jax.training.callbacks.l1_schedule_rule

## Utilities

::: wavediff_jax.training.train_utils.calculate_sample_weights

::: wavediff_jax.training.train_utils.configure_optimizer
