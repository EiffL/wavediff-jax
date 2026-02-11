# TensorFlow to JAX Migration Guide

This page documents how TensorFlow/Keras patterns were translated to JAX/Equinox equivalents during the WaveDiff reimplementation.

## Pattern Mapping

| TensorFlow | JAX/Equinox | Notes |
|---|---|---|
| `tf.keras.layers.Layer` | `eqx.Module` | Immutable PyTree |
| `self.add_weight(...)` | Class field + `jax.random` init | Declared as typed annotation |
| `layer.trainable = False` | `eqx.partition(model, filter_spec)` | Boolean pytree controls trainability |
| `tf.Variable.assign(...)` | `eqx.tree_at(where, model, value)` | Returns new model |
| `self.add_loss(...)` | Return `(output, aux_dict)` | Explicit auxiliary data |
| `model.compile() + model.fit()` | `make_step()` + `train_epoch()` | Explicit functional loop |
| `model.predict(x)` | `model(x)` | Direct call |
| `model.save_weights(path)` | `eqx.tree_serialise_leaves(path, model)` | Exact parameter preservation |
| `tf.map_fn(fn, elems)` | `jax.vmap(fn)(elems)` | Parallel, not sequential |
| `tf.convert_to_tensor(x)` | `jnp.asarray(x)` | Same semantics |
| `tfa.image.interpolate_spline` | `thin_plate_spline_interpolate` | Custom JAX implementation |

## Detailed Examples

### Layer Definition

=== "TensorFlow"

    ```python
    class TFPolynomialZernikeField(tf.keras.layers.Layer):
        def __init__(self, n_zernikes, d_max, **kwargs):
            super().__init__(**kwargs)
            self.coeff_mat = self.add_weight(
                shape=(n_zernikes, n_poly),
                initializer="random_normal",
                trainable=True,
            )

        def call(self, positions):
            poly_mat = calc_poly_position_mat(positions)
            return tf.linalg.matmul(self.coeff_mat, poly_mat)
    ```

=== "JAX/Equinox"

    ```python
    class PolynomialZernikeField(eqx.Module):
        coeff_mat: jnp.ndarray  # trainable

        def __init__(self, n_zernikes, d_max, *, key):
            self.coeff_mat = jax.random.normal(key, (n_zernikes, n_poly)) * 0.01

        def __call__(self, positions):
            poly_mat = calc_poly_position_mat(positions)
            return jnp.matmul(self.coeff_mat, poly_mat)
    ```

### Trainability Control

=== "TensorFlow"

    ```python
    # Freeze non-parametric parameters
    model.np_opd.trainable = False
    model.poly_field.trainable = True
    model.compile(optimizer=Adam(lr=1e-3), loss='mse')
    model.fit(x_train, y_train, epochs=100)
    ```

=== "JAX/Equinox"

    ```python
    # Create filter spec
    filter_spec = param_filter(model)  # True for poly_field.coeff_mat only

    # Training step
    trainable, frozen = eqx.partition(model, filter_spec)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(
        eqx.filter(grads, filter_spec), opt_state
    )
    model = eqx.apply_updates(model, updates)
    ```

### Functional Mutation

=== "TensorFlow"

    ```python
    # Zero out alpha matrix in-place
    model.np_opd.alpha_mat.assign(tf.zeros_like(model.np_opd.alpha_mat))
    ```

=== "JAX/Equinox"

    ```python
    # Return new model with zeroed alpha
    model = eqx.tree_at(
        lambda m: m.np_opd.alpha_mat,
        model,
        jnp.zeros_like(model.np_opd.alpha_mat),
    )
    ```

### Batch Processing

=== "TensorFlow"

    ```python
    # Sequential map over batch
    psf_batch = tf.map_fn(
        lambda args: single_star_psf(*args),
        (opd_batch, sed_batch),
        dtype=tf.float32,
    )
    ```

=== "JAX/Equinox"

    ```python
    # Parallel vmap over batch
    psf_batch = jax.vmap(single_star_psf)(opd_batch, sed_batch)
    ```

### Regularisation

=== "TensorFlow"

    ```python
    class NonParametricLayer(tf.keras.layers.Layer):
        def call(self, x):
            output = self._compute(x)
            self.add_loss(tf.reduce_sum(tf.abs(self.alpha) ** 1.1))
            return output
    ```

=== "JAX/Equinox"

    ```python
    class NonParametricLayer(eqx.Module):
        def __call__(self, x):
            output = self._compute(x)
            aux = {'alpha_graph': self.alpha_graph}
            return output, aux
        # Regularisation computed in loss function:
        # lp_reg = jnp.sum(jnp.abs(aux['alpha_graph']) ** 1.1)
    ```

## Module Mapping

| TF File | JAX File | Key Changes |
|---|---|---|
| `tf_layers.py` (1010 lines) | `models/layers.py` (405 lines) | 8 Keras layers → 8 `eqx.Module` classes |
| `tf_modules.py` (545 lines) | `optics/diffraction.py` (166 lines) | 6 Keras classes → 5 pure functions |
| `psf_model_parametric.py` | `models/parametric.py` | `model.predict()` → `model()` |
| `psf_model_semiparametric.py` | `models/semiparametric.py` | `.trainable` → filter functions |
| `train_utils.py` (807 lines) | `training/trainer.py` (398 lines) | `model.fit()` → explicit loop |

## Performance Considerations

| Aspect | TensorFlow | JAX |
|---|---|---|
| Batch processing | Sequential `tf.map_fn` | Parallel `jax.vmap` |
| JIT compilation | `@tf.function` (graph tracing) | `jax.jit` / `eqx.filter_jit` |
| Gradient computation | `tf.GradientTape` | `jax.grad` / `eqx.filter_value_and_grad` |
| Memory management | Automatic (eager mode) | Explicit donation via `eqx.partition` |
| Multi-device | `tf.distribute.Strategy` | `jax.pmap` / `jax.sharding` |
