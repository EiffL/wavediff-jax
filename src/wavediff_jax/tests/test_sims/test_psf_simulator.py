"""Tests for the PSF simulator module.

Tests PSFSimulator construction, monochromatic and polychromatic PSF generation.
"""

import numpy as np
import pytest
from wavediff_jax.sims.psf_simulator import PSFSimulator


@pytest.fixture(scope="module")
def small_simulator():
    """Create a small PSFSimulator for testing.

    Uses reduced parameters to speed up tests:
    - Small pupil diameter (64 pixels)
    - Small output dimension (32 pixels)
    - Few Zernike orders (15)
    - No Euclid obscurations (for speed)
    """
    sim = PSFSimulator(
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
    return sim


@pytest.fixture(scope="module")
def simulator_with_obsc():
    """Create a PSFSimulator with Euclid obscurations."""
    sim = PSFSimulator(
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
        euclid_obsc=True,
        LP_filter_length=3,
        verbose=0,
    )
    return sim


class TestPSFSimulatorConstruction:
    """Test PSFSimulator initialization."""

    def test_basic_construction(self, small_simulator):
        """Test that the simulator is created with correct attributes."""
        sim = small_simulator
        assert sim.max_order == 15
        assert sim.output_dim == 32
        assert sim.pupil_diameter == 64
        assert sim.oversampling_rate == 3.0
        assert sim.pix_sampling == 12
        assert sim.tel_diameter == 1.2
        assert sim.tel_focal_length == 24.5

    def test_zernike_maps_generated(self, small_simulator):
        """Test that zernike maps are generated correctly."""
        sim = small_simulator
        assert len(sim.zernike_maps) == sim.max_order
        for zmap in sim.zernike_maps:
            assert zmap.shape == (sim.pupil_diameter, sim.pupil_diameter)

    def test_pupil_mask(self, small_simulator):
        """Test that the pupil mask has the correct shape and is boolean."""
        sim = small_simulator
        assert sim.pupil_mask.shape == (sim.pupil_diameter, sim.pupil_diameter)
        assert sim.pupil_mask.dtype == bool
        # There should be some True values (inside the unit circle)
        assert np.any(sim.pupil_mask)

    def test_obscurations_no_euclid(self, small_simulator):
        """Test that obscurations are all ones when euclid_obsc=False."""
        sim = small_simulator
        assert sim.obscurations.shape == (sim.pupil_diameter, sim.pupil_diameter)
        np.testing.assert_array_equal(
            sim.obscurations, np.ones((sim.pupil_diameter, sim.pupil_diameter))
        )

    def test_obscurations_euclid(self, simulator_with_obsc):
        """Test that Euclid obscurations are generated when euclid_obsc=True."""
        sim = simulator_with_obsc
        assert sim.obscurations.shape == (sim.pupil_diameter, sim.pupil_diameter)
        # Euclid obscurations should have values between 0 and 1
        assert np.all(sim.obscurations >= 0)
        assert np.all(sim.obscurations <= 1)
        # There should be some obscured regions (value < 1)
        assert np.any(sim.obscurations < 1)


class TestRandomZernikeCoefficients:
    """Test random Zernike coefficient generation."""

    def test_gen_random_z_coeffs(self, small_simulator):
        """Test random Zernike coefficient generation."""
        sim = small_simulator
        sim.gen_random_Z_coeffs(max_order=15, rand_seed=42)
        assert sim.z_coeffs is not None
        assert len(sim.z_coeffs) == 15

    def test_set_z_coeffs(self, small_simulator):
        """Test setting Zernike coefficients."""
        sim = small_simulator
        coeffs = [0.01 * i for i in range(sim.max_order)]
        sim.set_z_coeffs(coeffs)
        assert sim.z_coeffs == coeffs

    def test_set_z_coeffs_wrong_length(self, small_simulator):
        """Test that setting wrong-length coefficients is handled."""
        sim = small_simulator
        old_coeffs = sim.z_coeffs
        sim.set_z_coeffs([0.1, 0.2])  # Wrong length
        # Coefficients should not have changed
        assert sim.z_coeffs == old_coeffs


class TestFeasibleWavelength:
    """Test feasible wavelength calculations."""

    def test_feasible_N(self, small_simulator):
        """Test that feasible_N returns an even integer."""
        sim = small_simulator
        N = sim.feasible_N(0.725)
        assert isinstance(N, int)
        assert N % 2 == 0
        assert N > 0

    def test_feasible_wavelength(self, small_simulator):
        """Test that feasible_wavelength returns a valid wavelength."""
        sim = small_simulator
        wv = sim.feasible_wavelength(0.725)
        assert isinstance(wv, float)
        # Should be close to the requested wavelength
        assert abs(wv - 0.725) < 0.05


class TestMonochromaticPSF:
    """Test monochromatic PSF generation."""

    def test_generate_mono_psf(self, small_simulator):
        """Test monochromatic PSF generation produces valid output."""
        sim = small_simulator
        # Set known coefficients
        coeffs = [0.01 / (i + 1) for i in range(sim.max_order)]
        sim.set_z_coeffs(coeffs)

        psf = sim.generate_mono_PSF(lambda_obs=0.725, get_psf=True)

        assert psf is not None
        assert psf.shape == (sim.output_dim, sim.output_dim)
        # PSF should be non-negative
        assert np.all(psf >= 0)
        # PSF should be normalized to sum approximately 1
        assert abs(np.sum(psf) - 1.0) < 1e-3

    def test_generate_mono_psf_different_wavelengths(self, small_simulator):
        """Test that different wavelengths produce different PSFs."""
        sim = small_simulator
        coeffs = [0.01 / (i + 1) for i in range(sim.max_order)]
        sim.set_z_coeffs(coeffs)

        psf1 = sim.generate_mono_PSF(lambda_obs=0.6, get_psf=True)
        sim.set_z_coeffs(coeffs)  # Reset coefficients
        psf2 = sim.generate_mono_PSF(lambda_obs=0.8, get_psf=True)

        assert psf1.shape == psf2.shape
        # PSFs at different wavelengths should differ
        assert not np.allclose(psf1, psf2)

    def test_generate_mono_psf_with_regen(self, small_simulator):
        """Test monochromatic PSF generation with regenerated sample."""
        sim = small_simulator
        psf = sim.generate_mono_PSF(
            lambda_obs=0.725, regen_sample=True, get_psf=True
        )
        assert psf is not None
        assert psf.shape == (sim.output_dim, sim.output_dim)
        assert np.all(psf >= 0)


class TestPolychromaticPSF:
    """Test polychromatic PSF generation."""

    def test_generate_poly_psf(self, small_simulator):
        """Test polychromatic PSF generation with a synthetic SED."""
        sim = small_simulator
        coeffs = [0.01 / (i + 1) for i in range(sim.max_order)]
        sim.set_z_coeffs(coeffs)

        # Create a simple synthetic SED: uniform distribution
        n_sed_points = 100
        wavelengths = np.linspace(550, 900, n_sed_points)
        sed_values = np.ones(n_sed_points) / n_sed_points
        SED = np.column_stack([wavelengths, sed_values])

        poly_psf = sim.generate_poly_PSF(SED, n_bins=5)

        assert poly_psf is not None
        assert poly_psf.shape == (sim.output_dim, sim.output_dim)
        # Polychromatic PSF should be non-negative
        assert np.all(poly_psf >= 0)
        # Should have meaningful values (not all zeros)
        assert np.sum(poly_psf) > 0


class TestWFERMS:
    """Test WFE RMS calculations."""

    def test_calculate_wfe_rms(self, small_simulator):
        """Test WFE RMS calculation."""
        sim = small_simulator
        coeffs = [0.01 / (i + 1) for i in range(sim.max_order)]
        sim.set_z_coeffs(coeffs)

        wfe_rms = sim.calculate_wfe_rms(z_coeffs=coeffs)
        assert isinstance(wfe_rms, float)
        assert wfe_rms > 0

    def test_normalize_zernikes(self, small_simulator):
        """Test Zernike normalization."""
        sim = small_simulator
        coeffs = [0.05 / (i + 1) for i in range(sim.max_order)]

        normalized = sim.normalize_zernikes(z_coeffs=coeffs, max_wfe_rms=0.1)
        assert len(normalized) == sim.max_order

        # After normalization, WFE RMS should be close to max_wfe_rms
        wfe_rms = sim.calculate_wfe_rms(z_coeffs=normalized)
        assert abs(wfe_rms - 0.1) < 0.02  # Allow some tolerance


class TestRadialIdx:
    """Test radial index generation."""

    def test_get_radial_idx(self):
        """Test radial index generation."""
        radial_idxs = PSFSimulator.get_radial_idx(max_order=10)
        assert len(radial_idxs) > 10
        # First element should be 0 (piston)
        assert radial_idxs[0] == 0


class TestSEDOperations:
    """Test SED-related operations."""

    def test_filter_sed(self, small_simulator):
        """Test SED filtering."""
        n_sed_points = 100
        wavelengths = np.linspace(550, 900, n_sed_points)
        sed_values = np.ones(n_sed_points)
        SED = np.column_stack([wavelengths, sed_values])

        n_bins = 10
        SED_filt = PSFSimulator.filter_SED(SED, n_bins)

        assert SED_filt.shape == (n_bins, 2)
        # Filtered SED should be normalized
        assert abs(np.sum(SED_filt[:, 1]) - 1.0) < 1e-10
        # Wavelengths should be in increasing order
        assert np.all(np.diff(SED_filt[:, 0]) > 0)

    def test_calc_sed_wave_values(self, small_simulator):
        """Test SED wave value calculation."""
        sim = small_simulator
        n_sed_points = 100
        wavelengths = np.linspace(550, 900, n_sed_points)
        sed_values = np.ones(n_sed_points)
        SED = np.column_stack([wavelengths, sed_values])

        feasible_wv, SED_norm = sim.calc_SED_wave_values(SED, n_bins=5)

        assert len(feasible_wv) == 5
        assert len(SED_norm) == 5
        # Feasible wavelengths should be in the VIS band (in um)
        assert np.all(feasible_wv > 0.5)
        assert np.all(feasible_wv < 1.0)
        # SED should be normalized
        assert abs(np.sum(SED_norm) - 1.0) < 1e-10
