# Training

WaveDiff-JAX uses a functional training loop built on [Equinox](https://github.com/patrick-kidger/equinox) and [Optax](https://github.com/google-deepmind/optax), replacing Keras's `model.compile()`/`model.fit()` with explicit gradient steps and parameter management.

## Core Concepts

### Functional Training Loop

Unlike Keras, there is no hidden mutable state. Each training step is a pure function:

```python
model, opt_state, loss = make_step(model, opt_state, optimizer, batch, loss_fn, filter_spec)
```

This returns a **new** model and optimizer state — the originals are unchanged.

### Trainability Filtering

Which parameters get updated is controlled by **filter specifications** — boolean PyTrees that mirror the model structure:

```python
from wavediff_jax.training.trainer import param_filter, nonparam_filter, complete_filter

# Only parametric parameters (coeff_mat)
spec = param_filter(model)

# Only non-parametric parameters (S_mat, alpha_mat, etc.)
spec = nonparam_filter(model)

# All trainable parameters
spec = complete_filter(model)
```

## Single Training Step

```python
import optax
import equinox as eqx
from wavediff_jax.training.trainer import make_step, param_filter
from wavediff_jax.training.losses import total_loss

# Setup
optimizer = optax.adam(1e-3)
filter_spec = param_filter(model)
opt_state = optimizer.init(eqx.filter(model, filter_spec))

# Loss function
def loss_fn(model, batch):
    positions, packed_seds, targets, masks = batch
    return total_loss(
        model, positions, packed_seds, targets, masks,
        sample_weight=None, l2_param=1e-4, l1_rate=0.0,
    )

# Step
model, opt_state, loss = make_step(
    model, opt_state, optimizer, batch, loss_fn, filter_spec
)
```

## Training an Epoch

`train_epoch` shuffles data into mini-batches and runs `make_step` on each:

```python
from wavediff_jax.training.trainer import train_epoch

model, opt_state, epoch_loss = train_epoch(
    model, opt_state, optimizer,
    data=(positions, packed_seds, targets, masks),
    batch_size=32,
    loss_fn=loss_fn,
    filter_spec=filter_spec,
    key=jax.random.PRNGKey(epoch),
)
```

## Block Coordinate Descent (BCD)

For semi-parametric and physical models, `general_train_cycle` runs multi-phase BCD:

```python
from wavediff_jax.training.trainer import general_train_cycle

# training_hparams should contain:
#   n_epochs_param, n_epochs_nonparam, n_epochs_complete (int)
#   lr_param, lr_nonparam (float)
#   batch_size (int)
#   l2_param (float)
#   l1_rate (float)
#   cycle_def (str): "parametric", "non-parametric", "complete",
#                    "only-parametric", "only-non-parametric"
#   first_run (bool): if True, zeros out alpha before parametric phase

results = general_train_cycle(
    model=model,
    train_data=(positions, packed_seds, targets, masks),
    val_data=(val_pos, val_seds, val_targets, val_masks),
    training_hparams=hparams,
    key=jax.random.PRNGKey(0),
)

trained_model = results["model"]
train_losses = results["train_losses"]
val_losses = results["val_losses"]
```

### BCD Cycle Types

| Cycle | Phases | Use Case |
|---|---|---|
| `"parametric"` | 1. Train parametric → 2. Train non-parametric | Standard BCD |
| `"non-parametric"` | 1. Train non-parametric → 2. Train parametric | Reversed BCD |
| `"complete"` | 1. Train parametric → 2. Train non-parametric → 3. Train all | Full cycle |
| `"only-parametric"` | 1. Train parametric only | Parametric-only models |
| `"only-non-parametric"` | 1. Train non-parametric only | Fine-tuning |

### First-Run Behaviour

When `first_run=True`, the parametric phase starts with the non-parametric component zeroed out (via `set_alpha_zero`). After the parametric phase completes, `set_alpha_identity` restores the non-parametric mixing matrix before training it.

## Loss Functions

### Available Losses

```python
from wavediff_jax.training.losses import (
    mse_loss,              # basic MSE
    masked_mse_loss,       # MSE with mask (0=include, 1=exclude)
    weighted_masked_mse_loss,  # masked MSE with per-sample weights
    l2_opd_regularization, # L2 penalty on OPD maps
    lp_regularization,     # Lp sparsity on alpha_graph (p=1.1)
    total_loss,            # combined: data + L2 + Lp
)
```

### Total Loss Composition

`total_loss` combines:

1. **Data term:** Weighted masked MSE between predicted and target PSFs
2. **L2 regularisation:** \\(\lambda_2 \sum \text{OPD}^2\\) to penalise large wavefront errors
3. **Lp regularisation:** \\(\lambda_1 \sum |\\alpha|^p\\) (p=1.1) for sparsity in graph-based non-parametric models

The L1/Lp rate is scheduled to decay during training:

```python
from wavediff_jax.training.callbacks import l1_schedule_rule

# Halves every 10 epochs
l1 = l1_schedule_rule(epoch=20, l1_rate=1e-3)
# → 2.5e-4 (halved twice: at epoch 10 and 20)
```

## Checkpointing

```python
from wavediff_jax.training.callbacks import save_checkpoint, load_checkpoint

# Save model parameters
save_checkpoint(model, "checkpoints/epoch_100.eqx")

# Load (requires a template model with correct structure)
model = load_checkpoint(model_template, "checkpoints/epoch_100.eqx")
```

Uses Equinox's `tree_serialise_leaves` / `tree_deserialise_leaves` for exact parameter preservation.

## Sample Weights

Per-sample weights based on noise estimation:

```python
from wavediff_jax.training.train_utils import calculate_sample_weights

weights = calculate_sample_weights(
    target_stars,
    use_sample_weights=True,
    masked=False,
)
# Returns inverse-variance weights normalised by median
```

## Optimizer Configuration

```python
from wavediff_jax.training.train_utils import configure_optimizer

optimizer = configure_optimizer(lr=1e-3, b1=0.9, b2=0.999, eps=1e-7)
# Returns optax.adam with TF-matching defaults
```
