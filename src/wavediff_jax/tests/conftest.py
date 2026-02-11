"""Conftest.

Common test fixtures for wavediff-jax test suite.

"""

import pytest
import numpy as np


@pytest.fixture
def rng():
    """Provide a seeded numpy random generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_yaml_content():
    """Provide sample YAML content for config tests."""
    return (
        "model:\n"
        "  name: test_model\n"
        "  n_layers: 3\n"
        "  params:\n"
        "    learning_rate: 0.001\n"
        "    batch_size: 32\n"
        "training:\n"
        "  epochs: 100\n"
        "  optimizer: adam\n"
        "  callbacks:\n"
        "    - early_stopping\n"
        "    - checkpoint\n"
    )


@pytest.fixture
def sample_multi_doc_yaml_content():
    """Provide sample multi-document YAML content for stream tests."""
    return (
        "doc1_key: doc1_value\n"
        "---\n"
        "doc2_key: doc2_value\n"
        "---\n"
        "doc3_key: doc3_value\n"
    )
