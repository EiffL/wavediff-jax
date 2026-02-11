"""Test IO.

Tests for the wavediff_jax.utils.io module.

"""

import os
import pytest
from wavediff_jax.utils.io import FileIOHandler


class TestFileIOHandler:
    """Tests for FileIOHandler class."""

    def test_get_timestamp_returns_string(self, tmp_path):
        """Test that get_timestamp returns a non-empty string."""
        handler = FileIOHandler(
            output_path=str(tmp_path),
            config_path=str(tmp_path),
        )
        timestamp = handler.get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0

    def test_setup_outputs_creates_directory_structure(self, tmp_path):
        """Test that setup_outputs creates the expected directory tree."""
        handler = FileIOHandler(
            output_path=str(tmp_path),
            config_path=str(tmp_path),
        )
        handler.setup_outputs()

        # Check that the parent output directory exists
        parent_dir = os.path.join(str(tmp_path), handler.parent_output_dir)
        assert os.path.isdir(parent_dir)

        # Check that the run directory exists
        run_dir = handler._run_output_dir
        assert os.path.isdir(run_dir)

        # Check that all subdirectories exist
        expected_subdirs = [
            "config",
            "checkpoint",
            "log-files",
            "metrics",
            "optim-hist",
            "plots",
            "psf_model",
        ]
        for subdir in expected_subdirs:
            subdir_path = os.path.join(run_dir, subdir)
            assert os.path.isdir(subdir_path), f"Missing subdirectory: {subdir}"

    def test_copy_conffile_to_output_dir(self, tmp_path):
        """Test that copy_conffile_to_output_dir copies the file correctly."""
        # Create a source config file
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        source_file = config_dir / "test_config.yaml"
        source_file.write_text("key: value\n")

        handler = FileIOHandler(
            output_path=str(tmp_path),
            config_path=str(config_dir),
        )
        handler.setup_outputs()

        handler.copy_conffile_to_output_dir("test_config.yaml")

        # Check the file was copied
        destination = os.path.join(
            handler.get_config_dir(handler._run_output_dir),
            "test_config.yaml",
        )
        assert os.path.isfile(destination)

        with open(destination) as f:
            content = f.read()
        assert content == "key: value\n"

    def test_get_config_dir(self, tmp_path):
        """Test that get_config_dir returns correct path."""
        handler = FileIOHandler(
            output_path=str(tmp_path),
            config_path=str(tmp_path),
        )
        config_dir = handler.get_config_dir("/some/run/dir")
        assert config_dir == "/some/run/dir/config"

    def test_get_checkpoint_dir(self, tmp_path):
        """Test that get_checkpoint_dir returns correct path."""
        handler = FileIOHandler(
            output_path=str(tmp_path),
            config_path=str(tmp_path),
        )
        checkpoint_dir = handler.get_checkpoint_dir("/some/run/dir")
        assert checkpoint_dir == "/some/run/dir/checkpoint"

    def test_get_metrics_dir(self, tmp_path):
        """Test that get_metrics_dir returns correct path."""
        handler = FileIOHandler(
            output_path=str(tmp_path),
            config_path=str(tmp_path),
        )
        metrics_dir = handler.get_metrics_dir("/some/run/dir")
        assert metrics_dir == "/some/run/dir/metrics"

    def test_get_plots_dir(self, tmp_path):
        """Test that get_plots_dir returns correct path."""
        handler = FileIOHandler(
            output_path=str(tmp_path),
            config_path=str(tmp_path),
        )
        plots_dir = handler.get_plots_dir("/some/run/dir")
        assert plots_dir == "/some/run/dir/plots"

    def test_get_optimizer_dir(self, tmp_path):
        """Test that get_optimizer_dir returns correct path."""
        handler = FileIOHandler(
            output_path=str(tmp_path),
            config_path=str(tmp_path),
        )
        optim_dir = handler.get_optimizer_dir("/some/run/dir")
        assert optim_dir == "/some/run/dir/optim-hist"

    def test_get_psf_model_dir(self, tmp_path):
        """Test that get_psf_model_dir returns correct path."""
        handler = FileIOHandler(
            output_path=str(tmp_path),
            config_path=str(tmp_path),
        )
        psf_dir = handler.get_psf_model_dir("/some/run/dir")
        assert psf_dir == "/some/run/dir/psf_model"
