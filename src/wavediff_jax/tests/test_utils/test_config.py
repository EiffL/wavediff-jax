"""Test Config.

Tests for the wavediff_jax.utils.config module.

"""

import pytest
import os
from wavediff_jax.utils.config import RecursiveNamespace, read_yaml, read_conf, read_stream


class TestRecursiveNamespace:
    """Tests for RecursiveNamespace class."""

    def test_simple_attributes(self):
        """Test that simple key-value pairs become attributes."""
        ns = RecursiveNamespace(name="test", value=42)
        assert ns.name == "test"
        assert ns.value == 42

    def test_nested_dict(self):
        """Test that nested dicts are converted to RecursiveNamespace."""
        ns = RecursiveNamespace(
            model={"name": "psf_model", "params": {"lr": 0.001, "layers": 3}}
        )
        assert isinstance(ns.model, RecursiveNamespace)
        assert ns.model.name == "psf_model"
        assert isinstance(ns.model.params, RecursiveNamespace)
        assert ns.model.params.lr == 0.001
        assert ns.model.params.layers == 3

    def test_list_with_dicts(self):
        """Test that lists containing dicts are properly converted."""
        ns = RecursiveNamespace(
            items=[{"a": 1, "b": 2}, {"c": 3, "d": 4}]
        )
        assert isinstance(ns.items, list)
        assert len(ns.items) == 2
        assert isinstance(ns.items[0], RecursiveNamespace)
        assert ns.items[0].a == 1
        assert ns.items[1].c == 3

    def test_list_with_non_dicts(self):
        """Test that lists of non-dict items are preserved as-is."""
        ns = RecursiveNamespace(tags=["alpha", "beta", "gamma"])
        assert ns.tags == ["alpha", "beta", "gamma"]

    def test_mixed_list(self):
        """Test list with both dict and non-dict entries."""
        ns = RecursiveNamespace(
            items=[{"key": "val"}, "plain_string", 42]
        )
        assert isinstance(ns.items[0], RecursiveNamespace)
        assert ns.items[0].key == "val"
        assert ns.items[1] == "plain_string"
        assert ns.items[2] == 42

    def test_map_entry_with_dict(self):
        """Test map_entry static method with dict input."""
        result = RecursiveNamespace.map_entry({"x": 10})
        assert isinstance(result, RecursiveNamespace)
        assert result.x == 10

    def test_map_entry_with_non_dict(self):
        """Test map_entry static method with non-dict input."""
        assert RecursiveNamespace.map_entry("hello") == "hello"
        assert RecursiveNamespace.map_entry(42) == 42


class TestReadYaml:
    """Tests for read_yaml function."""

    def test_read_yaml(self, tmp_path, sample_yaml_content):
        """Test reading a YAML file returns a dict."""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(sample_yaml_content)

        config = read_yaml(str(yaml_file))

        assert isinstance(config, dict)
        assert config["model"]["name"] == "test_model"
        assert config["model"]["n_layers"] == 3
        assert config["model"]["params"]["learning_rate"] == 0.001
        assert config["training"]["epochs"] == 100
        assert "early_stopping" in config["training"]["callbacks"]


class TestReadConf:
    """Tests for read_conf function."""

    def test_read_conf_returns_namespace(self, tmp_path, sample_yaml_content):
        """Test that read_conf returns a RecursiveNamespace."""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(sample_yaml_content)

        conf = read_conf(str(yaml_file))

        assert isinstance(conf, RecursiveNamespace)
        assert conf.model.name == "test_model"
        assert conf.model.n_layers == 3
        assert conf.model.params.learning_rate == 0.001
        assert conf.training.epochs == 100
        assert conf.training.optimizer == "adam"

    def test_read_conf_empty_file_raises(self, tmp_path):
        """Test that an empty config file raises TypeError."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        with pytest.raises(TypeError, match="empty"):
            read_conf(str(yaml_file))


class TestReadStream:
    """Tests for read_stream function."""

    def test_read_stream_multi_doc(self, tmp_path, sample_multi_doc_yaml_content):
        """Test reading a multi-document YAML file."""
        yaml_file = tmp_path / "multi_doc.yaml"
        yaml_file.write_text(sample_multi_doc_yaml_content)

        docs = list(read_stream(str(yaml_file)))

        assert len(docs) == 3
        assert docs[0]["doc1_key"] == "doc1_value"
        assert docs[1]["doc2_key"] == "doc2_value"
        assert docs[2]["doc3_key"] == "doc3_value"

    def test_read_stream_single_doc(self, tmp_path):
        """Test reading a single-document YAML via stream."""
        yaml_file = tmp_path / "single.yaml"
        yaml_file.write_text("key: value\n")

        docs = list(read_stream(str(yaml_file)))

        assert len(docs) == 1
        assert docs[0]["key"] == "value"
