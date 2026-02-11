# Configuration

WaveDiff-JAX uses YAML configuration files parsed into `RecursiveNamespace` objects. This is a direct port of the original WaveDiff config system.

## Reading Configs

```python
from wavediff_jax.utils.config import read_conf, read_yaml, read_stream

# Parse YAML into RecursiveNamespace (dot-access)
config = read_conf("config.yaml")
print(config.model.n_zernikes)  # 45

# Parse as raw dict
config_dict = read_yaml("config.yaml")

# Multi-document YAML
for doc in read_stream("multi_config.yaml"):
    print(doc)
```

## RecursiveNamespace

Nested dictionaries are automatically converted to dot-accessible namespaces:

```python
from wavediff_jax.utils.config import RecursiveNamespace

config = RecursiveNamespace(**{
    "model": {"n_zernikes": 45, "d_max": 2},
    "training": {"lr": 1e-3, "batch_size": 32},
})

print(config.model.n_zernikes)  # 45
print(config.training.lr)       # 0.001
```

## Example Configuration

```yaml
# model_config.yaml
model_name: poly
model:
  n_zernikes: 45
  d_max: 2
  output_dim: 64
  output_Q: 1
  pupil_diam: 256
  x_lims: [0, 1000]
  y_lims: [0, 1000]

training:
  n_epochs_param: 100
  n_epochs_nonparam: 50
  lr_param: 1.0e-3
  lr_nonparam: 1.0e-4
  batch_size: 32
  l2_param: 1.0e-4
  l1_rate: 1.0e-5
  cycle_def: complete

data:
  train_path: data/train.npz
  test_path: data/test.npz
  n_bins_lambda: 20

metrics:
  compute_poly: true
  compute_opd: true
  compute_mono: true
  compute_shape: true
```

## File I/O

The `FileIOHandler` manages output directory structure:

```python
from wavediff_jax.utils.io import FileIOHandler

io = FileIOHandler(output_path="/results", config_path="/configs")
io.setup_outputs()

# Creates:
# /results/wf-outputs/wf-outputs-YYYYMMDDHHMMSS/
#   ├── config/
#   ├── checkpoint/
#   ├── log-files/
#   ├── metrics/
#   ├── optim-hist/
#   ├── plots/
#   └── psf_model/
```
