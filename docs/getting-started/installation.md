# Installation

## Requirements

- Python >= 3.10
- JAX >= 0.4.20 with jaxlib
- Equinox >= 0.11.0
- Optax >= 0.1.7

## From Source

```bash
# Clone with reference submodule
git clone --recurse-submodules https://github.com/your-org/wavediff-jax.git
cd wavediff-jax

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

## Optional Dependencies

### Development (testing)

```bash
pip install -e ".[dev]"
```

This installs `pytest`, `pytest-xdist`, and `galsim`.

### Shape Metrics (galsim)

```bash
pip install -e ".[metrics]"
```

[GalSim](https://github.com/GalSim-developers/GalSim) is required only for PSF shape metrics (`compute_shape_metrics`). All other functionality works without it.

### Documentation

```bash
pip install mkdocs mkdocs-material "mkdocstrings[python]"
```

## GPU Support

JAX automatically uses GPU if available. For explicit GPU installation:

```bash
pip install --upgrade "jax[cuda12]"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for platform-specific instructions.

## Verifying the Installation

```bash
# Run the test suite
python -m pytest src/wavediff_jax/tests/ -v --tb=short

# Quick smoke test
python -c "import wavediff_jax; print(wavediff_jax.PSF_FACTORY.keys())"
# → dict_keys(['poly', 'semi-param', 'physical-poly', 'ground-truth-semi-param', 'ground-truth-physical-poly'])
```

## Submodule (Reference Implementation)

The original TensorFlow-based WaveDiff is included as a Git submodule at `extern/wf-psf/` for reference. It is **not imported or executed** by WaveDiff-JAX.

```bash
# Initialize submodule (if not cloned with --recurse-submodules)
git submodule update --init --recursive
```

!!! warning
    Do **not** install TensorFlow or attempt to import `wf_psf`. The submodule is read-only reference code.
