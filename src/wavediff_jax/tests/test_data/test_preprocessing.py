"""Tests for the data preprocessing module.

Tests SED processing functions and DataHandler initialization.
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest
import os
import tempfile
from types import SimpleNamespace

from wavediff_jax.data.preprocessing import (
    generate_SED_elems,
    generate_SED_elems_jax,
    generate_packed_elems,
    DataHandler,
    get_np_obs_positions,
    get_obs_positions,
    extract_star_data,
    get_np_zernike_prior,
)
from wavediff_jax.sims.psf_simulator import PSFSimulator


@pytest.fixture(scope="module")
def small_simulator():
    """Create a small PSFSimulator for testing."""
    return PSFSimulator(
        max_order=15,
        max_wfe_rms=0.1,
        output_dim=32,
        rand_seed=42,
        plot_opt=False,
        oversampling_rate=3.0,
        output_Q=1,
        pix_sampling=12,
        tel_diameter=1.2,
        tel_focal_length=24.5,
        pupil_diameter=64,
        euclid_obsc=False,
        LP_filter_length=3,
        verbose=0,
        SED_sigma=0,
        SED_interp_pts_per_bin=0,
        SED_extrapolate=True,
        SED_interp_kind="linear",
    )


@pytest.fixture
def synthetic_sed():
    """Create a synthetic SED for testing."""
    n_points = 100
    wavelengths = np.linspace(550, 900, n_points)
    sed_values = np.ones(n_points)
    return np.column_stack([wavelengths, sed_values])


class TestGenerateSEDElems:
    """Test generate_SED_elems function."""

    def test_basic_output(self, small_simulator, synthetic_sed):
        """Test that generate_SED_elems returns correct types and shapes."""
        n_bins = 5
        feasible_N, feasible_wv, SED_norm = generate_SED_elems(
            synthetic_sed, small_simulator, n_bins=n_bins
        )

        assert isinstance(feasible_N, np.ndarray)
        assert isinstance(feasible_wv, np.ndarray)
        assert len(feasible_N) == n_bins
        assert len(feasible_wv) == n_bins

    def test_feasible_N_values(self, small_simulator, synthetic_sed):
        """Test that feasible_N contains positive even integers."""
        feasible_N, _, _ = generate_SED_elems(
            synthetic_sed, small_simulator, n_bins=5
        )

        for N in feasible_N:
            assert N > 0
            assert int(N) % 2 == 0

    def test_feasible_wv_range(self, small_simulator, synthetic_sed):
        """Test that feasible wavelengths are in the VIS passband."""
        _, feasible_wv, _ = generate_SED_elems(
            synthetic_sed, small_simulator, n_bins=5
        )

        assert np.all(feasible_wv > 0.5)  # > 500nm in um
        assert np.all(feasible_wv < 1.0)  # < 1000nm in um

    def test_sed_norm_normalized(self, small_simulator, synthetic_sed):
        """Test that SED_norm is normalized to sum to 1."""
        _, _, SED_norm = generate_SED_elems(
            synthetic_sed, small_simulator, n_bins=5
        )

        assert abs(np.sum(SED_norm) - 1.0) < 1e-10


class TestGenerateSEDElemsJAX:
    """Test generate_SED_elems_jax function."""

    def test_output_types(self, small_simulator, synthetic_sed):
        """Test that outputs are JAX arrays."""
        result = generate_SED_elems_jax(
            synthetic_sed, small_simulator, n_bins=5
        )

        assert len(result) == 3
        for arr in result:
            assert isinstance(arr, jnp.ndarray)
            assert arr.dtype == jnp.float64

    def test_output_values(self, small_simulator, synthetic_sed):
        """Test that JAX outputs match numpy outputs."""
        n_bins = 5
        np_result = generate_SED_elems(
            synthetic_sed, small_simulator, n_bins=n_bins
        )
        jax_result = generate_SED_elems_jax(
            synthetic_sed, small_simulator, n_bins=n_bins
        )

        for np_val, jax_val in zip(np_result, jax_result):
            np.testing.assert_allclose(
                np.asarray(jax_val), np_val, rtol=1e-5
            )


class TestGeneratePackedElems:
    """Test generate_packed_elems function."""

    def test_output_types(self, small_simulator, synthetic_sed):
        """Test that outputs are JAX float64 arrays."""
        result = generate_packed_elems(
            synthetic_sed, small_simulator, n_bins=5
        )

        assert len(result) == 3
        for arr in result:
            assert isinstance(arr, jnp.ndarray)
            assert arr.dtype == jnp.float64

    def test_consistency_with_generate_sed_elems(self, small_simulator, synthetic_sed):
        """Test consistency between packed and unpacked versions."""
        n_bins = 5
        np_N, np_wv, np_norm = generate_SED_elems(
            synthetic_sed, small_simulator, n_bins=n_bins
        )
        packed = generate_packed_elems(
            synthetic_sed, small_simulator, n_bins=n_bins
        )

        np.testing.assert_allclose(np.asarray(packed[0]), np_N, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(packed[1]), np_wv, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(packed[2]), np_norm, rtol=1e-5)


class TestDataHandler:
    """Test DataHandler initialization with mock data."""

    @pytest.fixture
    def mock_dataset_dir(self, small_simulator, synthetic_sed):
        """Create a temporary directory with mock dataset files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock training dataset
            n_train = 5
            n_test = 3
            stamp_size = 32

            # Generate synthetic SEDs for training
            train_seds = np.array([synthetic_sed for _ in range(n_train)])
            test_seds = np.array([synthetic_sed for _ in range(n_test)])

            train_dataset = {
                "positions": np.random.rand(n_train, 2).astype(np.float32) * 1000,
                "noisy_stars": np.random.rand(n_train, stamp_size, stamp_size).astype(
                    np.float32
                ),
                "SEDs": train_seds,
            }

            test_dataset = {
                "positions": np.random.rand(n_test, 2).astype(np.float32) * 1000,
                "stars": np.random.rand(n_test, stamp_size, stamp_size).astype(
                    np.float32
                ),
                "SEDs": test_seds,
            }

            # Save as .npz files
            train_path = os.path.join(tmpdir, "train_data.npy")
            test_path = os.path.join(tmpdir, "test_data.npy")
            np.save(train_path, train_dataset)
            np.save(test_path, test_dataset)

            yield tmpdir, n_train, n_test, stamp_size

    def test_data_handler_training_init(
        self, mock_dataset_dir, small_simulator
    ):
        """Test DataHandler initialization for training dataset."""
        tmpdir, n_train, n_test, stamp_size = mock_dataset_dir

        data_params = SimpleNamespace(
            training=SimpleNamespace(
                data_dir=tmpdir,
                file="train_data.npy",
            ),
            test=SimpleNamespace(
                data_dir=tmpdir,
                file="test_data.npy",
            ),
        )

        handler = DataHandler(
            dataset_type="training",
            data_params=data_params,
            simPSF=small_simulator,
            n_bins_lambda=5,
            load_data=True,
        )

        assert handler.dataset is not None
        assert handler.dataset_type == "training"
        assert isinstance(handler.dataset["positions"], jnp.ndarray)
        assert handler.dataset["positions"].shape == (n_train, 2)
        assert isinstance(handler.dataset["noisy_stars"], jnp.ndarray)
        assert handler.sed_data is not None

    def test_data_handler_test_init(
        self, mock_dataset_dir, small_simulator
    ):
        """Test DataHandler initialization for test dataset."""
        tmpdir, n_train, n_test, stamp_size = mock_dataset_dir

        data_params = SimpleNamespace(
            training=SimpleNamespace(
                data_dir=tmpdir,
                file="train_data.npy",
            ),
            test=SimpleNamespace(
                data_dir=tmpdir,
                file="test_data.npy",
            ),
        )

        handler = DataHandler(
            dataset_type="test",
            data_params=data_params,
            simPSF=small_simulator,
            n_bins_lambda=5,
            load_data=True,
        )

        assert handler.dataset is not None
        assert handler.dataset_type == "test"
        assert isinstance(handler.dataset["positions"], jnp.ndarray)
        assert handler.dataset["positions"].shape == (n_test, 2)
        assert isinstance(handler.dataset["stars"], jnp.ndarray)

    def test_data_handler_deferred_load(
        self, mock_dataset_dir, small_simulator
    ):
        """Test DataHandler with deferred loading."""
        tmpdir, _, _, _ = mock_dataset_dir

        data_params = SimpleNamespace(
            training=SimpleNamespace(
                data_dir=tmpdir,
                file="train_data.npy",
            ),
            test=SimpleNamespace(
                data_dir=tmpdir,
                file="test_data.npy",
            ),
        )

        handler = DataHandler(
            dataset_type="training",
            data_params=data_params,
            simPSF=small_simulator,
            n_bins_lambda=5,
            load_data=False,
        )

        assert handler.dataset is None
        assert handler.sed_data is None

    def test_sed_data_shape(self, mock_dataset_dir, small_simulator):
        """Test that SED data has the correct shape after processing."""
        tmpdir, n_train, _, _ = mock_dataset_dir

        data_params = SimpleNamespace(
            training=SimpleNamespace(
                data_dir=tmpdir,
                file="train_data.npy",
            ),
            test=SimpleNamespace(
                data_dir=tmpdir,
                file="test_data.npy",
            ),
        )

        handler = DataHandler(
            dataset_type="training",
            data_params=data_params,
            simPSF=small_simulator,
            n_bins_lambda=5,
            load_data=True,
        )

        assert handler.sed_data is not None
        assert isinstance(handler.sed_data, jnp.ndarray)
        # Shape should be (n_train, n_bins, 3) after transpose
        assert handler.sed_data.shape[0] == n_train
        assert handler.sed_data.dtype == jnp.float32


class TestUtilityFunctions:
    """Test utility functions for data extraction."""

    @pytest.fixture
    def mock_data_config(self):
        """Create mock DataConfigHandler-like object."""
        n_train = 5
        n_test = 3

        train_positions = np.random.rand(n_train, 2).astype(np.float32) * 1000
        test_positions = np.random.rand(n_test, 2).astype(np.float32) * 1000

        train_stars = np.random.rand(n_train, 32, 32).astype(np.float32)
        test_stars = np.random.rand(n_test, 32, 32).astype(np.float32)

        train_zk_prior = np.random.rand(n_train, 15).astype(np.float32)
        test_zk_prior = np.random.rand(n_test, 15).astype(np.float32)

        data = SimpleNamespace(
            training_data=SimpleNamespace(
                dataset={
                    "positions": jnp.array(train_positions),
                    "noisy_stars": jnp.array(train_stars),
                    "zernike_prior": train_zk_prior,
                }
            ),
            test_data=SimpleNamespace(
                dataset={
                    "positions": jnp.array(test_positions),
                    "stars": jnp.array(test_stars),
                    "zernike_prior": test_zk_prior,
                }
            ),
        )

        return data, n_train, n_test

    def test_get_np_obs_positions(self, mock_data_config):
        """Test get_np_obs_positions returns concatenated positions."""
        data, n_train, n_test = mock_data_config
        positions = get_np_obs_positions(data)

        assert isinstance(positions, np.ndarray)
        assert positions.shape == (n_train + n_test, 2)

    def test_get_obs_positions(self, mock_data_config):
        """Test get_obs_positions returns JAX array."""
        data, n_train, n_test = mock_data_config
        positions = get_obs_positions(data)

        assert isinstance(positions, jnp.ndarray)
        assert positions.shape == (n_train + n_test, 2)
        assert positions.dtype == jnp.float32

    def test_extract_star_data(self, mock_data_config):
        """Test extract_star_data concatenates correctly."""
        data, n_train, n_test = mock_data_config
        stars = extract_star_data(data, train_key="noisy_stars", test_key="stars")

        assert isinstance(stars, np.ndarray)
        assert stars.shape == (n_train + n_test, 32, 32)

    def test_extract_star_data_missing_key(self, mock_data_config):
        """Test that extract_star_data raises KeyError for missing keys."""
        data, _, _ = mock_data_config
        with pytest.raises(KeyError):
            extract_star_data(data, train_key="nonexistent", test_key="stars")

    def test_get_np_zernike_prior(self, mock_data_config):
        """Test get_np_zernike_prior concatenates correctly."""
        data, n_train, n_test = mock_data_config
        zk_prior = get_np_zernike_prior(data)

        assert isinstance(zk_prior, np.ndarray)
        assert zk_prior.shape == (n_train + n_test, 15)
