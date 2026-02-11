"""Spatially Varying PSF Simulator.

This module contains classes and methods for simulating spatially varying
Point Spread Functions (PSFs) using polynomial variations of Zernike coefficients.

Ported from wf-psf (CosmoStat/wf-psf) to wavediff-jax.
Original authors: Tobias Liaudat <tobias.liaudat@cea.fr> & Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class MeshHelper:
    """Mesh Helper.

    A utility class for generating mesh grids.
    """

    @staticmethod
    def build_mesh(x_lims, y_lims, grid_points=None, grid_size=None, endpoint=True):
        """Build Mesh.

        A method to build a mesh.

        Parameters
        ----------
        x_lims : list
            A list representing the lower and upper limits along the x-axis.
        y_lims : list
            A list representing the lower and upper limits along the y-axis.
        grid_points : list or None, optional
            List defining the size of each axis grid. (default is None)
        grid_size : int or None, optional
            Number of points to generate for each axis. (default is None)
        endpoint : bool, optional
            If True, stop is the last sample. (default is True).

        Returns
        -------
        tuple
            A tuple containing two 2-dimensional arrays for x- and y-coordinate axes.
        """
        if grid_size is None:
            if grid_points is None:
                raise ValueError(
                    "At least one of 'grid_points' or 'grid_size' must be provided."
                )
            num_x, num_y = grid_points
        else:
            num_x = grid_size
            num_y = grid_size

        x = np.linspace(x_lims[0], x_lims[1], num=num_x, endpoint=endpoint)
        y = np.linspace(y_lims[0], y_lims[1], num=num_y, endpoint=endpoint)

        return np.meshgrid(x, y)


class CoordinateHelper:
    """Coordinate Helper.

    A utility class for handling coordinate operations.
    """

    @staticmethod
    def scale_positions(x, y, x_lims, y_lims):
        """Scale Positions.

        A method to scale x- and y-positions.

        Parameters
        ----------
        x : numpy.ndarray
            A 1-dimensional numpy ndarray denoting a vector of x positions.
        y : numpy.ndarray
            A 1-dimensional numpy ndarray denoting a vector of y positions.
        x_lims : list
            Lower and upper limits along the x-axis.
        y_lims : list
            Lower and upper limits along the y-axis.

        Returns
        -------
        scaled_x : numpy.ndarray
            Scaled x positions.
        scaled_y : numpy.ndarray
            Scaled y positions.
        """
        scaled_x = (x - x_lims[0]) / (x_lims[1] - x_lims[0])
        scaled_x = (scaled_x - 0.5) * 2
        scaled_y = (y - y_lims[0]) / (y_lims[1] - y_lims[0])
        scaled_y = (scaled_y - 0.5) * 2

        return scaled_x, scaled_y

    @staticmethod
    def calculate_shift(x_lims, y_lims, grid_points):
        """Calculate Shift.

        Parameters
        ----------
        x_lims : list
            Lower and upper limits along the x-axis.
        y_lims : list
            Lower and upper limits along the y-axis.
        grid_points : list
            List defining the size of each axis grid.

        Returns
        -------
        x_step : float
        y_step : float
        """
        x_step = (x_lims[1] - x_lims[0]) / grid_points[0]
        y_step = (y_lims[1] - y_lims[0]) / grid_points[1]

        return x_step, y_step

    @staticmethod
    def add_random_shift_to_positions(
        xv_grid, yv_grid, grid_points, x_lims, y_lims, seed=None
    ):
        """Add Random Shift to Positions.

        Parameters
        ----------
        xv_grid : numpy.ndarray
            Grid of x-coordinates.
        yv_grid : numpy.ndarray
            Grid of y-coordinates.
        grid_points : list
            A list defining the size of each axis grid.
        x_lims : list
            Lower and upper limits along the x-axis.
        y_lims : list
            Lower and upper limits along the y-axis.
        seed : int
            Seed for random number generation.

        Returns
        -------
        xv_s : numpy.ndarray
            Positions with added random shifts along the x-axis.
        yv_s : numpy.ndarray
            Positions with added random shifts along the y-axis.
        """
        np.random.seed(seed)
        xv_rand = np.random.rand(grid_points[0], grid_points[1])
        yv_rand = np.random.rand(grid_points[0], grid_points[1])
        x_step, y_step = CoordinateHelper.calculate_shift(x_lims, y_lims, grid_points)
        xv_rand = (xv_rand - 0.5) * x_step
        yv_rand = (yv_rand - 0.5) * y_step
        xv = xv_grid + xv_rand.T
        yv = yv_grid + yv_rand.T

        xv_s, yv_s = CoordinateHelper.check_and_adjust_coordinate_limits(
            xv.flatten(), yv.flatten(), x_lims, y_lims
        )

        return xv_s, yv_s

    @staticmethod
    def check_and_adjust_coordinate_limits(x, y, x_lims, y_lims):
        """Check and adjust coordinate limits.

        Parameters
        ----------
        x : numpy.ndarray
            Positions along x-axis.
        y : numpy.ndarray
            Positions along y-axis.
        x_lims : list
            Lower and upper limits along the x-axis.
        y_lims : list
            Lower and upper limits along the y-axis.

        Returns
        -------
        x : numpy.ndarray
            Adjusted positions along the x-axis.
        y : numpy.ndarray
            Adjusted positions along the y-axis.
        """
        x[x > x_lims[1]] = x_lims[1]
        x[x < x_lims[0]] = x_lims[0]
        y[y > y_lims[1]] = y_lims[1]
        y[y < y_lims[0]] = y_lims[0]

        return x, y

    @staticmethod
    def check_position_coordinate_limits(xv, yv, x_lims, y_lims, verbose):
        """Check Position Coordinate Limits.

        Parameters
        ----------
        xv : numpy.ndarray
            The x coordinates to be checked.
        yv : numpy.ndarray
            The y coordinates to be checked.
        x_lims : tuple
            (min, max) for x coordinates.
        y_lims : tuple
            (min, max) for y coordinates.
        verbose : bool
            If True, print warning messages.
        """
        x_check = np.sum(xv >= x_lims[1] * 1.1) + np.sum(xv <= x_lims[0] * 1.1)
        y_check = np.sum(yv >= y_lims[1] * 1.1) + np.sum(yv <= y_lims[0] * 1.1)

        if verbose and x_check > 0:
            logger.warning(
                f"WARNING! x value is outside the limits [{x_lims[0]}, {x_lims[1]}]"
            )

        if verbose and y_check > 0:
            logger.warning(
                f"WARNING! y value is outside the limits [{y_lims[0]}, {y_lims[0]}]"
            )


class PolynomialMatrixHelper:
    """PolynomialMatrixHelper.

    Helper class with methods for generating polynomial matrices of positions.
    """

    @staticmethod
    def generate_polynomial_matrix(x, y, x_lims, y_lims, d_max):
        """Generate polynomial matrix of positions.

        Parameters
        ----------
        x : numpy.ndarray
            x positions.
        y : numpy.ndarray
            y positions.
        x_lims : list
            Lower and upper limits along the x-axis.
        y_lims : list
            Lower and upper limits along the y-axis.
        d_max : int
            Maximum polynomial degree.

        Returns
        -------
        Pi : numpy.ndarray
            A 2-dimensional polynomial matrix.
        """
        n_mono = (d_max + 1) * (d_max + 2) // 2
        if np.isscalar(x):
            Pi = np.zeros((n_mono, 1))
        else:
            Pi = np.zeros((n_mono, x.shape[0]))

        scaled_x, scaled_y = CoordinateHelper.scale_positions(x, y, x_lims, y_lims)

        for d in range(d_max + 1):
            row_idx = d * (d + 1) // 2
            for p in range(d + 1):
                Pi[row_idx + p, :] = scaled_x ** (d - p) * scaled_y ** p

        return Pi


class ZernikeHelper:
    """ZernikeHelper.

    Helper class for generating Zernike Polynomials.
    """

    @staticmethod
    def initialize_Z_matrix(max_order, size, seed=None):
        """Initialize Zernike Matrix.

        Parameters
        ----------
        max_order : int
            Maximum order of Zernike polynomials.
        size : int
            Size of the position vector.
        seed : int
            Seed for random number generation.

        Returns
        -------
        numpy.ndarray
            Array of shape (max_order, size).
        """
        np.random.seed(seed)
        return np.random.randn(max_order, size)

    @staticmethod
    def normalize_Z_matrix(Z, lim_max_wfe_rms):
        """Normalize Zernike Matrix.

        Parameters
        ----------
        Z : numpy.ndarray
            The Zernike matrix.
        lim_max_wfe_rms : int
            Upper limit for WFE RMS.

        Returns
        -------
        Z : numpy.ndarray
            Normalized Zernike matrix.
        """
        norm_weights = np.sqrt(np.sum(Z ** 2, axis=1))
        Z /= norm_weights.reshape((-1, 1)) / lim_max_wfe_rms
        return Z

    @staticmethod
    def initialize_normalized_zernike_matrix(
        max_order, size, lim_max_wfe_rms, seed=None
    ):
        """Initialize Normalized Zernike Matrix.

        Parameters
        ----------
        max_order : int
            Maximum order of Zernike polynomials.
        size : int
            Size of the position vector.
        lim_max_wfe_rms : int
            Upper limit for WFE RMS.
        seed : int
            Seed for random number generation.

        Returns
        -------
        numpy.ndarray
            A normalized Zernike matrix.
        """
        return ZernikeHelper.normalize_Z_matrix(
            ZernikeHelper.initialize_Z_matrix(max_order, size, seed), lim_max_wfe_rms
        )

    @staticmethod
    def generate_zernike_polynomials(xv, yv, x_lims, y_lims, d_max, polynomial_coeffs):
        """Generate Zernike Polynomials.

        Parameters
        ----------
        xv : np.ndarray (dim,)
            x positions.
        yv : np.ndarray (dim,)
            y positions.
        x_lims : list
            Lower and upper limits along the x-axis.
        y_lims : list
            Lower and upper limits along the y-axis.
        d_max : int
            Maximum polynomial degree.
        polynomial_coeffs : numpy.ndarray
            Polynomial coefficients.

        Returns
        -------
        numpy.ndarray
            Spatial polynomials generated from the given positions and coefficients.
        """
        Pi_samples = PolynomialMatrixHelper.generate_polynomial_matrix(
            xv, yv, x_lims, y_lims, d_max
        )

        return polynomial_coeffs @ Pi_samples

    @staticmethod
    def calculate_zernike(
        xv, yv, x_lims, y_lims, d_max, polynomial_coeffs, verbose=False
    ):
        """Calculate Zernikes for a specific position.

        Parameters
        ----------
        xv : numpy.ndarray
            x positions.
        yv : numpy.ndarray
            y positions.
        x_lims : list
            Lower and upper limits along the x-axis.
        y_lims : list
            Lower and upper limits along the y-axis.
        d_max : int
            Maximum polynomial degree.
        polynomial_coeffs : numpy.ndarray
            Polynomial coefficients.
        verbose : bool
            Flag for warning messages.

        Returns
        -------
        numpy.ndarray
            Computed Zernike polynomials.
        """
        CoordinateHelper.check_position_coordinate_limits(
            xv, yv, x_lims, y_lims, verbose
        )

        return ZernikeHelper.generate_zernike_polynomials(
            xv, yv, x_lims, y_lims, d_max, polynomial_coeffs
        )


class SpatialVaryingPSF:
    """Spatial Varying PSF.

    Generate PSF field with polynomial variations of Zernike coefficients.

    Parameters
    ----------
    psf_simulator : PSFSimulator object
        Class instance of the PSFSimulator.
    d_max : int
        Maximum polynomial degree for spatial variation.
    grid_points : list
        List defining grid size for each axis.
    grid_size : int or None, optional
        Number of points for the grid. (default is None)
    max_order : int
        Maximum order of Zernike polynomials.
    x_lims : list
        Lower and upper limits along the x-axis.
    y_lims : list
        Lower and upper limits along the y-axis.
    n_bins : int
        Number of bins for polychromatic PSFs.
    lim_max_wfe_rms : float
        Upper limit for WFE RMS.
    verbose : bool
        Verbosity flag.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        psf_simulator,
        d_max=2,
        grid_points=[4, 4],
        grid_size=None,
        max_order=45,
        x_lims=[0, 1e3],
        y_lims=[0, 1e3],
        n_bins=35,
        lim_max_wfe_rms=None,
        verbose=False,
        seed=None,
    ):
        self.psf_simulator = psf_simulator
        self.max_order = max_order
        self.d_max = d_max
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.grid_points = grid_points
        self.grid_size = grid_size
        self.n_bins = n_bins
        self.verbose = verbose
        self.seed = seed
        self._lim_max_wfe_rms = lim_max_wfe_rms

        self.polynomial_coeffs = None
        self.WFE_RMS = None

        self.build_polynomial_coeffs()

    @property
    def lim_max_wfe_rms(self):
        """Get the upper limit for WFE RMS."""
        if self._lim_max_wfe_rms is None:
            return self.psf_simulator.max_wfe_rms
        else:
            return self._lim_max_wfe_rms

    @lim_max_wfe_rms.setter
    def lim_max_wfe_rms(self, value):
        """Set the upper limit for WFE RMS."""
        self._lim_max_wfe_rms = value

    def estimate_polynomial_coeffs(self, xv, yv, Z):
        """Estimate polynomial coefficients using least squares.

        Parameters
        ----------
        xv : numpy.ndarray
            x positions.
        yv : numpy.ndarray
            y positions.
        Z : numpy.ndarray
            Zernike coefficients.

        Returns
        -------
        numpy.ndarray
            Estimated polynomial coefficients.
        """
        Pi = PolynomialMatrixHelper.generate_polynomial_matrix(
            xv, yv, self.x_lims, self.y_lims, self.d_max
        )

        return Z @ np.linalg.pinv(Pi)

    def calculate_wfe_rms(self, xv, yv, polynomial_coeffs):
        """Calculate the WFE RMS.

        Parameters
        ----------
        xv : numpy.ndarray
            x positions.
        yv : numpy.ndarray
            y positions.
        polynomial_coeffs : numpy.ndarray
            Polynomial coefficients.

        Returns
        -------
        numpy.ndarray
            WFE RMS values.
        """
        Z = ZernikeHelper.generate_zernike_polynomials(
            xv, yv, self.x_lims, self.y_lims, self.d_max, polynomial_coeffs
        )
        return np.sqrt(np.sum(Z ** 2, axis=0))

    def build_polynomial_coeffs(self):
        """Build polynomial coefficients for spatial variation."""
        xv_grid, yv_grid = MeshHelper.build_mesh(
            self.x_lims, self.y_lims, self.grid_points
        )

        xv, yv = CoordinateHelper.add_random_shift_to_positions(
            xv_grid, yv_grid, self.grid_points, self.x_lims, self.y_lims, self.seed
        )

        Z = ZernikeHelper.initialize_normalized_zernike_matrix(
            self.max_order, len(xv), self.lim_max_wfe_rms, self.seed
        )

        self.polynomial_coeffs = self.estimate_polynomial_coeffs(xv, yv, Z)

        xv_grid, yv_grid = MeshHelper.build_mesh(
            self.x_lims, self.y_lims, self.grid_points, self.grid_size, endpoint=True
        )

        xv = xv_grid.flatten()
        yv = yv_grid.flatten()

        calc_wfe = self.calculate_wfe_rms(xv, yv, self.polynomial_coeffs)

        scale_factor = (0.8 * self.lim_max_wfe_rms) / np.mean(calc_wfe)
        self.polynomial_coeffs *= scale_factor

        ZernikeHelper.generate_zernike_polynomials(
            xv, yv, self.x_lims, self.y_lims, self.d_max, self.polynomial_coeffs
        )

        self.WFE_RMS = self.calculate_wfe_rms(xv, yv, self.polynomial_coeffs).reshape(
            xv_grid.shape
        )

    def get_monochromatic_PSF(self, xv, yv, lambda_obs=0.725):
        """Calculate the monochromatic PSF at a specific position and wavelength.

        Parameters
        ----------
        xv : numpy.ndarray
            x positions.
        yv : numpy.ndarray
            y positions.
        lambda_obs : float, optional
            Wavelength of observation. Default is 0.725 um.

        Returns
        -------
        numpy.ndarray
            The generated monochromatic PSF.
        """
        zernikes = ZernikeHelper.calculate_zernike(
            xv, yv, self.x_lims, self.y_lims, self.d_max, self.polynomial_coeffs
        )

        self.psf_simulator.set_z_coeffs(zernikes)
        self.psf_simulator.generate_mono_PSF(lambda_obs=lambda_obs, regen_sample=False)
        return self.psf_simulator.get_psf()

    def get_polychromatic_PSF(self, xv, yv, SED):
        """Calculate the polychromatic PSF for a specific position and SED.

        Parameters
        ----------
        xv : numpy.ndarray
            x positions.
        yv : numpy.ndarray
            y positions.
        SED : array_like
            Spectral Energy Distribution.

        Returns
        -------
        tuple
            (polychromatic_psf, zernikes, opd)
        """
        zernikes = ZernikeHelper.calculate_zernike(
            xv, yv, self.x_lims, self.y_lims, self.d_max, self.polynomial_coeffs
        )

        self.psf_simulator.set_z_coeffs(zernikes)
        polychromatic_psf = self.psf_simulator.generate_poly_PSF(
            SED, n_bins=self.n_bins
        )
        opd = self.psf_simulator.opd

        return polychromatic_psf, zernikes, opd
