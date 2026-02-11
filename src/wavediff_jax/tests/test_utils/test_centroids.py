"""Test Centroids.

Tests for the wavediff_jax.utils.centroids module.

"""

import numpy as np
import pytest
from wavediff_jax.utils.centroids import CentroidEstimator


def _make_gaussian_psf(size, center_x, center_y, sigma=3.0, amplitude=1.0):
    """Create a synthetic 2D Gaussian PSF image.

    Parameters
    ----------
    size : int
        Size of the square image stamp.
    center_x : float
        x-coordinate of the Gaussian center.
    center_y : float
        y-coordinate of the Gaussian center.
    sigma : float
        Standard deviation of the Gaussian.
    amplitude : float
        Peak amplitude.

    Returns
    -------
    np.ndarray
        2D array of shape (size, size).
    """
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    gaussian = amplitude * np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2)
    )
    return gaussian


class TestCentroidEstimator:
    """Tests for CentroidEstimator class."""

    def test_centered_gaussian_centroid(self):
        """Test that centroid of a centered Gaussian is found near center."""
        size = 51
        center = size / 2.0
        n_images = 3

        images = np.stack(
            [_make_gaussian_psf(size, center, center, sigma=5.0) for _ in range(n_images)]
        )

        estimator = CentroidEstimator(
            im=images, sigma_init=5.0, n_iter=20, auto_run=True
        )

        centroids = estimator.get_centroids()
        # centroids shape is (2, n_images)
        assert centroids.shape == (2, n_images)

        # Each centroid should be very close to the true center
        for i in range(n_images):
            np.testing.assert_allclose(centroids[0, i], center, atol=0.1)
            np.testing.assert_allclose(centroids[1, i], center, atol=0.1)

    def test_off_center_gaussian_centroid(self):
        """Test that centroid of an off-center Gaussian is found near its true location."""
        size = 51
        true_cx = 27.3
        true_cy = 23.7
        n_images = 2

        images = np.stack(
            [_make_gaussian_psf(size, true_cx, true_cy, sigma=5.0) for _ in range(n_images)]
        )

        estimator = CentroidEstimator(
            im=images, sigma_init=5.0, n_iter=20, auto_run=True
        )

        centroids = estimator.get_centroids()

        for i in range(n_images):
            np.testing.assert_allclose(centroids[0, i], true_cx, atol=0.5)
            np.testing.assert_allclose(centroids[1, i], true_cy, atol=0.5)

    def test_get_intra_pixel_shifts_shape(self):
        """Test that get_intra_pixel_shifts returns the correct shape."""
        size = 31
        center = size / 2.0
        n_images = 5

        images = np.stack(
            [_make_gaussian_psf(size, center, center, sigma=4.0) for _ in range(n_images)]
        )

        estimator = CentroidEstimator(
            im=images, sigma_init=4.0, n_iter=10, auto_run=True
        )

        shifts = estimator.get_intra_pixel_shifts()
        assert shifts.shape == (n_images, 2)

    def test_centered_gaussian_shifts_near_zero(self):
        """Test that intra-pixel shifts for centered Gaussians are near zero."""
        size = 51
        center = size / 2.0
        n_images = 4

        images = np.stack(
            [_make_gaussian_psf(size, center, center, sigma=5.0) for _ in range(n_images)]
        )

        estimator = CentroidEstimator(
            im=images, sigma_init=5.0, n_iter=20, auto_run=True
        )

        shifts = estimator.get_intra_pixel_shifts()

        # Shifts should be near zero for centered PSFs
        np.testing.assert_allclose(shifts, 0.0, atol=0.1)

    def test_with_mask(self):
        """Test centroid estimation with a mask applied."""
        size = 51
        center = size / 2.0
        n_images = 2

        images = np.stack(
            [_make_gaussian_psf(size, center, center, sigma=5.0) for _ in range(n_images)]
        )

        # Create a mask that zeros out the edges
        mask = np.zeros_like(images)
        mask[:, :5, :] = 1
        mask[:, -5:, :] = 1
        mask[:, :, :5] = 1
        mask[:, :, -5:] = 1

        estimator = CentroidEstimator(
            im=images, mask=mask, sigma_init=5.0, n_iter=20, auto_run=True
        )

        centroids = estimator.get_centroids()

        # Centroid should still be near the center
        for i in range(n_images):
            np.testing.assert_allclose(centroids[0, i], center, atol=0.5)
            np.testing.assert_allclose(centroids[1, i], center, atol=0.5)

    def test_auto_run_false(self):
        """Test that auto_run=False does not run estimation."""
        size = 31
        center = size / 2.0
        n_images = 2

        images = np.stack(
            [_make_gaussian_psf(size, center, center, sigma=4.0) for _ in range(n_images)]
        )

        estimator = CentroidEstimator(
            im=images, sigma_init=4.0, n_iter=10, auto_run=False
        )

        # xc and yc should still be at the initial center guess
        np.testing.assert_allclose(estimator.xc, center, atol=1e-10)
        np.testing.assert_allclose(estimator.yc, center, atol=1e-10)
