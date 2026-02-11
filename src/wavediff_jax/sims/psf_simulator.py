"""PSF simulator.

A module to simulate PSFs from Zernike coefficients.

Ported from wf-psf (CosmoStat/wf-psf) to wavediff-jax.
Original authors: Tobias Liaudat <tobias.liaudat@cea.fr> & Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import scipy.signal as spsig
import scipy.interpolate as sinterp
from PIL import Image as PILImage
from wavediff_jax.optics.zernike import zernike_generator
from wavediff_jax.optics.obscurations import generate_euclid_pupil_obscurations
import logging

logger = logging.getLogger(__name__)


def downsample_im(input_im, output_dim):
    """Downsample image to (output_dim, output_dim).

    Uses PIL/Pillow LANCZOS resampling.

    Parameters
    ----------
    input_im : np.ndarray
        Input 2D image to be downsampled.
    output_dim : int
        Desired output dimension (both height and width).

    Returns
    -------
    np.ndarray
        Downsampled 2D image of shape (output_dim, output_dim).
    """
    output_dim = int(output_dim)
    pil_im = PILImage.fromarray(input_im)
    im_resized = pil_im.resize(
        (output_dim, output_dim), resample=PILImage.LANCZOS
    )
    return np.array(im_resized)


class PSFSimulator:
    """Simulate PSFs.

    In the future the zernike maps could be created with galsim or some other
    alternative.

    Parameters
    ----------
    max_order : int
        Maximum Zernike polynomial order. Default is 45.
    max_wfe_rms : float
        Maximum allowed WFE in RMS. Used for normalization. Units in [um].
        Default is 0.1.
    output_dim : int
        Output dimension of the square PSF stamp. Default is 64.
    rand_seed : int
        Random seed to be used to generate random zernike values.
        Default is None.
    plot_opt : bool
        Option to plot some characteristics of the PSF generated.
        Default is False.
    oversampling_rate : float
        Oversampling rate for the wavefront PSF simulation. Default is 3.0.
    output_Q : float
        Downsampling rate to match the specified telescope's sampling.
        Default is 1.
    pix_sampling : float
        Pixel sampling in [um]. Default is 12 [um] (Euclid-like).
    tel_diameter : float
        Telescope's main mirror diameter in [m]. Default is 1.2 [m].
    tel_focal_length : float
        Telescope's focal length in [m]. Default is 24.5 [m].
    pupil_diameter : int
        Pupil diameter in pixels. Default is 1024 [pix].
    euclid_obsc : bool
        Whether to use Euclid-like obscurations. Default is True.
    LP_filter_length : int
        Length of the Low-Pass filter. Default is 3.
    verbose : int
        Verbosity level. Default is 0.
    SED_sigma : float
        Standard deviation of the multiplicative SED Gaussian noise.
    SED_interp_pts_per_bin : int
        Number of points to interpolate in between SED values.
    SED_extrapolate : bool
        SED interpolation mode. Default uses extrapolation.
    SED_interp_kind : str
        SED interpolation kind. Options are 'cubic' or 'linear'.
    """

    def __init__(
        self,
        max_order=45,
        max_wfe_rms=0.1,
        output_dim=64,
        rand_seed=None,
        plot_opt=False,
        oversampling_rate=3.0,
        output_Q=1,
        pix_sampling=12,
        tel_diameter=1.2,
        tel_focal_length=24.5,
        pupil_diameter=1024,
        euclid_obsc=True,
        LP_filter_length=3,
        verbose=0,
        SED_sigma=0,
        SED_interp_pts_per_bin=0,
        SED_extrapolate=True,
        SED_interp_kind="linear",
    ):
        # Telescope characteristics
        self.oversampling_rate = oversampling_rate  # dimensionless
        self.output_Q = output_Q  # dimensionless
        self.pix_sampling = pix_sampling  # In [um]
        self.tel_diameter = tel_diameter  # In [m]
        self.tel_focal_length = tel_focal_length  # In [m]
        self.pupil_diameter = pupil_diameter  # In [pix]

        # Input attributes
        self.max_order = max_order
        self.rand_seed = rand_seed
        self.plot_opt = plot_opt
        self.zernike_maps = zernike_generator(self.max_order, self.pupil_diameter)
        self.max_wfe_rms = max_wfe_rms  # In [um]
        self.output_dim = output_dim  # In pixels per dimension
        self.verbose = verbose

        self.SED_sigma = SED_sigma
        self.SED_interp_pts_per_bin = SED_interp_pts_per_bin
        self.SED_extrapolate = SED_extrapolate
        self.SED_interp_kind = SED_interp_kind

        # Class attributes
        self.z_coeffs = None
        self.psf = None
        self.opd = None
        self.phase = None
        self.lambda_obs = None
        self.poly_psf = None

        # Generate pupil mask
        self.pupil_mask = ~np.isnan(self.zernike_maps[0])

        # Generate obscurations
        if euclid_obsc:
            self.obscurations = generate_euclid_pupil_obscurations(
                N_pix=pupil_diameter, N_filter=LP_filter_length
            )
        else:
            self.obscurations = np.ones((pupil_diameter, pupil_diameter))

    @staticmethod
    def fft_diffract(wf, output_Q, output_dim=64):
        """Perform a fft-based diffraction.

        Parameters
        ----------
        wf : np.ndarray
            A complex 2D array that corresponds to the wavefront function.
        output_Q : float
            Downsampling rate to match the specified telescope's sampling.
        output_dim : int
            Output dimension of the square PSF stamp.

        Returns
        -------
        psf : np.ndarray
            A real 2D array corresponding to the PSF.
        """
        # Perform the FFT-based diffraction operation
        fft_wf = np.fft.fftshift(np.fft.fft2(wf))
        psf = np.abs(fft_wf) ** 2

        # Calculate crop dimensions
        if output_dim * output_Q < psf.shape[0]:
            start = int(psf.shape[0] // 2 - (output_dim * output_Q) // 2)
            stop = int(psf.shape[0] // 2 + (output_dim * output_Q) // 2)
        else:
            start = 0
            stop = psf.shape[0]

        # Crop & Downsample PSF to match telescope sampling
        return downsample_im(psf[start:stop, start:stop], output_dim)

    @staticmethod
    def crop_img(to_crop_img, ref_im):
        """Crop image.

        Parameters
        ----------
        to_crop_img : np.ndarray
            Image to be cropped.
        ref_im : np.ndarray
            Reference image to match the size.

        Returns
        -------
        cropped_img : np.ndarray
            Cropped image.
        """
        cent_x = int(to_crop_img.shape[0] // 2)
        cent_y = int(to_crop_img.shape[1] // 2)

        delta_x = int(ref_im.shape[0] // 2)
        delta_y = int(ref_im.shape[1] // 2)

        return to_crop_img[
            cent_x - delta_x : cent_x + delta_x, cent_y - delta_y : cent_y + delta_y
        ]

    @staticmethod
    def decimate_im(input_im, decim_f):
        """Decimate image.

        Decimated by a factor of decim_f.
        Based on the PIL library using the default interpolator.
        """
        pil_im = PILImage.fromarray(input_im)
        (width, height) = (pil_im.width // decim_f, pil_im.height // decim_f)
        im_resized = pil_im.resize((width, height))

        return np.array(im_resized)

    @staticmethod
    def get_radial_idx(max_order=45):
        """Get radial zernike indices.

        Parameters
        ----------
        max_order : int
            Maximum Zernike polynomial order.

        Returns
        -------
        radial_idxs : np.ndarray
            Array of radial indices up to the specified maximum order.
        """
        it = 1
        radial_idxs = []

        while len(radial_idxs) <= max_order:
            for _it in range(it):
                radial_idxs.append(it - 1)

            it += 1

        return np.array(radial_idxs)

    def get_psf(self):
        """Get computed PSF.

        Returns
        -------
        psf : np.ndarray
            A real 2D array corresponding to the PSF.
        """
        if self.psf is not None:
            return self.psf
        else:
            logger.warning("No PSF has been computed yet.")

    # This method is a setter
    def gen_random_Z_coeffs(self, max_order=45, rand_seed=None):
        """Generate a random set of Zernike coefficients.

        The coefficients are generated following a uniform law U~[-1,1]
        divided by their radial zernike index.

        Parameters
        ----------
        max_order : int
            Maximum order of Zernike polynomials.
        rand_seed : int
            Seed for the random initialization.

        Returns
        -------
        rand_coeffs : list of floats
            List containing the random coefficients.
        """
        if rand_seed is not None:
            np.random.seed(rand_seed)

        rad_idx = self.get_radial_idx(max_order)
        rad_idx[0] = 1

        z_coeffs = []

        for it in range(max_order):
            z_coeffs.append((np.random.rand() - 0.5) * 2.0 / rad_idx[it])

        self.z_coeffs = z_coeffs

    def get_z_coeffs(self):
        """Get random coefficients.

        Returns
        -------
        z_coeffs : list of floats
            List containing the random coefficients.
        """
        if self.z_coeffs is not None:
            return self.z_coeffs
        else:
            logger.warning("Random coeffs not generated.")

    def set_z_coeffs(self, z_coeffs):
        """Set zernike coefficients."""
        if len(z_coeffs) == self.max_order:
            self.z_coeffs = z_coeffs
        else:
            logger.warning(
                f"Zernike coefficients should be of length {self.max_order}"
            )

    def normalize_zernikes(self, z_coeffs=None, max_wfe_rms=None):
        """Normalize zernike coefficients.

        Returns a set of zernike coefficients normalized to the maximum
        allowed WFE rms.

        Parameters
        ----------
        z_coeffs : list of floats
            List containing the zernike coefficients to be normalized.
        max_wfe_rms : float
            Maximum allowed WFE in RMS. Units in [um].

        Returns
        -------
        z_coeffs : list of floats
            List containing the normalized zernike coefficients.
        """
        if max_wfe_rms is None:
            max_wfe_rms = self.max_wfe_rms

        # Calculate normalization factor
        wfe_rms = self.calculate_wfe_rms(z_coeffs=z_coeffs)
        mult_factor = max_wfe_rms / wfe_rms

        # Normalize Zernike coefficients and return them
        z_coeffs = [_z * mult_factor for _z in z_coeffs]

        return z_coeffs

    def calculate_wfe_rms(self, z_coeffs=None):
        """Calculate WFE rms from a set of zernike coefficients.

        Parameters
        ----------
        z_coeffs : list of floats
            List containing the zernike coefficients to be used.

        Returns
        -------
        wfe_rms : float
            Wavefront error in RMS [um].
        """
        if z_coeffs is None:
            if self.z_coeffs is None:
                self.gen_random_Z_coeffs(self.max_order, self.rand_seed)
                z_coeffs = self.get_z_coeffs()
            else:
                z_coeffs = self.get_z_coeffs()

        # Create the phase with the Zernike basis
        opd = 0
        for it in range(self.max_order):
            opd += self.zernike_maps[it] * z_coeffs[it]

        # Project obscurations on to the OPD
        opd *= self.obscurations

        # Calculate normalization factor
        wfe_rms = np.sqrt(
            np.mean((opd[self.pupil_mask] - np.mean(opd[self.pupil_mask])) ** 2)
        )

        return wfe_rms

    def check_wfe_rms(self, z_coeffs=None, max_wfe_rms=None):
        """Check if Zernike coefficients are within the maximum admitted error."""
        if max_wfe_rms is None:
            max_wfe_rms = self.max_wfe_rms

        wfe_rms = self.calculate_wfe_rms(z_coeffs=z_coeffs)

        return max_wfe_rms - wfe_rms

    def generate_mono_PSF(self, lambda_obs=0.725, regen_sample=False, get_psf=False):
        """Generate monochromatic PSF.

        Parameters
        ----------
        lambda_obs : float
            Wavelength of observation in [um]. Default is 0.725.
        regen_sample : bool
            Whether to regenerate a new random sample. Default is False.
        get_psf : bool
            If True, return the PSF array. Default is False.

        Returns
        -------
        psf : np.ndarray or None
            The PSF if get_psf is True, else None.
        """
        if lambda_obs < 0.55 * 0.9 or lambda_obs > 0.9 * 1.1:
            logger.warning(
                f"Requested wavelength {lambda_obs:.4f} um is not in VIS passband [0.55,0.9]um"
            )
        self.lambda_obs = lambda_obs

        # Calculate the OPD from the Zernike coefficients
        self.calculate_opd(regen_sample)

        # Apply the diffraction operator using the opd (optical path differences)
        self.diffract_phase()

        if get_psf is True:
            return np.copy(self.psf)

    def calculate_opd(self, regen_sample=False):
        """Calculate the OPD from the Zernike coefficients.

        Parameters
        ----------
        regen_sample : bool
            Whether to regenerate a new random sample of Zernike coefficients.
        """
        if self.z_coeffs is None or regen_sample is True:
            # Generate a random sample of coefficients
            self.gen_random_Z_coeffs(self.max_order, self.rand_seed)
            # Normalize coefficients
            z_coeffs = self.normalize_zernikes(self.get_z_coeffs(), self.max_wfe_rms)
            # Save coefficients
            self.set_z_coeffs(z_coeffs)
        else:
            z_coeffs = self.get_z_coeffs()

        # Create the phase with the Zernike basis
        opd = 0
        for it in range(self.max_order):
            opd += self.zernike_maps[it] * z_coeffs[it]

        # Save the wavefront
        self.opd = opd

    def diffract_phase(self, lambda_obs=None):
        """Diffract the phase map.

        Parameters
        ----------
        lambda_obs : float
            Wavelength of observation in [um]. If None, use the stored wavelength.
        """
        if lambda_obs is None:
            if self.lambda_obs is None:
                logger.warning(
                    "No wavelength is defined. Using default value 0.8um."
                )
                lambda_obs = 0.8
            else:
                lambda_obs = self.lambda_obs
        elif lambda_obs < 0.55 * 0.99 or lambda_obs > 0.9 * 1.01:
            logger.warning(
                f"Wavelength {lambda_obs:.4f} is not in VIS passband [0.55,0.9]um"
            )

        # Calculate the feasible lambda closest to lambda_obs
        possible_lambda = self.feasible_wavelength(lambda_obs)

        # Save wavelength
        self.lambda_obs = possible_lambda

        # Calculate the required N for the input lambda_obs
        possible_N = self.feasible_N(self.lambda_obs)

        # Generate the full phase and
        # Add zeros to the phase to have the correct fourier sampling
        start = possible_N // 2 - self.opd.shape[0] // 2
        stop = possible_N // 2 + self.opd.shape[0] // 2

        self.phase = np.zeros((possible_N, possible_N), dtype=np.complex128)
        self.phase[start:stop, start:stop][self.pupil_mask] = np.exp(
            2j * np.pi * self.opd[self.pupil_mask] / self.lambda_obs
        )

        # Project obscurations to the phase
        self.phase[start:stop, start:stop] *= self.obscurations

        # FFT-diffract the phase (wavefront) and then crop to desired dimension
        self.psf = self.fft_diffract(
            wf=self.phase, output_Q=self.output_Q, output_dim=self.output_dim
        )

        # Normalize psf
        self.psf /= np.sum(self.psf)

    def feasible_N(self, lambda_obs):
        """Calculate the feasible N for a lambda_obs diffraction.

        Input wavelength must be in [um].

        Parameters
        ----------
        lambda_obs : float
            Target wavelength in [um].

        Returns
        -------
        possible_N : int
            Feasible N value (an even integer).
        """
        req_N = (
            self.oversampling_rate
            * self.pupil_diameter
            * lambda_obs
            * self.tel_focal_length
        ) / (self.tel_diameter * self.pix_sampling)
        possible_N = int((req_N // 2) * 2)

        return possible_N

    def feasible_wavelength(self, lambda_obs):
        """Calculate closest feasible wavelength to target wavelength.

        Input wavelength must be in [um].

        Parameters
        ----------
        lambda_obs : float
            Target wavelength in [um].

        Returns
        -------
        possible_lambda : float
            Feasible wavelength in [um].
        """
        possible_N = self.feasible_N(lambda_obs)

        possible_lambda = (possible_N * self.tel_diameter * self.pix_sampling) / (
            self.pupil_diameter * self.oversampling_rate * self.tel_focal_length
        )

        if self.verbose > 0:
            logger.info(
                f"Possible wavelength: {possible_lambda:.5f} Possible N: {possible_N:.2f}"
            )

        return possible_lambda

    @staticmethod
    def gen_SED_interp(SED, n_bins=35, interp_kind="cubic"):
        """Generate SED interpolator.

        Returns the interpolator and the wavelengths in [nm].

        Parameters
        ----------
        SED : np.ndarray
            The SED. First column is wavelength positions, second column is SED value.
        n_bins : int
            Number of desired bins for the SED interpolation.
        interp_kind : str
            Type of interpolation to be used.

        Returns
        -------
        wvlength : np.ndarray
            Wavelength positions in [nm].
        SED_interp : callable
            SED interpolator function.
        """
        wv_max = 900
        wv_min = 550
        wvlength = np.linspace(wv_min, wv_max, num=n_bins, endpoint=True)

        SED_interp = sinterp.interp1d(
            SED[:, 0],
            SED[:, 1],
            kind=interp_kind,
            bounds_error=False,
            fill_value="extrapolate",
        )

        return wvlength, SED_interp

    @staticmethod
    def filter_SED(SED, n_bins, filter_lims=None):
        """Generate filtered SED.

        Returns a 'n_bins' point SED and wvlength vector.

        Parameters
        ----------
        SED : np.ndarray
            The unfiltered SED.
        n_bins : int
            Number of desired bins.
        filter_lims : list of np.ndarray or None
            Integration limits for each bin.

        Returns
        -------
        SED_filt : np.ndarray
            Filtered SED with shape (n_bins, 2).
        """
        wv_step = SED[1, 0] - SED[0, 0]
        wv_max = SED[-1, 0] + wv_step / 2
        wv_min = SED[0, 0] - wv_step / 2

        if filter_lims is None:
            wvlength = np.linspace(wv_min, wv_max, num=n_bins + 1, endpoint=True)
            filter_lims = [wvlength[it : it + 2] for it in range(n_bins)]

        SED_filt = np.zeros((n_bins, 2))

        for idx, lims in enumerate(filter_lims):
            lim_low = np.abs(SED[:, 0] - lims[0]).argmin()
            lim_hi = np.abs(SED[:, 0] - lims[1]).argmin()
            SED_filt[idx, 1] = np.sum(SED[(lim_low + 1) : lim_hi, 1])
            SED_filt[idx, 1] = (
                SED_filt[idx, 1]
                + SED[lim_low, 1]
                * (SED[lim_low, 0] - lims[0] + wv_step / 2)
                / wv_step
            )
            if lim_hi != lim_low:
                SED_filt[idx, 1] = (
                    SED_filt[idx, 1]
                    + SED[lim_hi, 1]
                    * (lims[1] - SED[lim_hi, 0] + wv_step / 2)
                    / wv_step
                )
            SED_filt[idx, 1] = SED_filt[idx, 1] * (lims[1] - lims[0])

        SED_filt[:, 1] = SED_filt[:, 1] / np.sum(SED_filt[:, 1])
        SED_filt[:, 0] = np.sum(np.array(filter_lims), axis=1) / 2

        return SED_filt

    @staticmethod
    def SED_gen_noise(n_bins, SED_sigma):
        """Generate random normal errors for the binned SED.

        Parameters
        ----------
        n_bins : int
            Number of bins.
        SED_sigma : positive float
            Standard deviation of the Gaussian noise.

        Returns
        -------
        np.ndarray
            Noise vector.
        """
        return np.random.normal(0, SED_sigma, n_bins)

    def interp_SED(self, SED_filt, n_points=0, n_bins=35, interp_kind="cubic"):
        """Interpolate the binned SED.

        Parameters
        ----------
        SED_filt : np.ndarray
            The filtered SED.
        n_points : int
            Number of points to add in each bin.
        n_bins : int
            Number of bins.
        interp_kind : str
            Interpolation kind.

        Returns
        -------
        SED : np.ndarray
            Interpolated SED.
        """
        _, SED_interpolator = self.gen_SED_interp(
            SED_filt, n_bins, interp_kind=interp_kind
        )
        wv_step = SED_filt[1, 0] - SED_filt[0, 0]

        if n_points == 1:
            if self.SED_extrapolate:
                SED = np.zeros((n_bins * 2 + 1, 3))
                SED[1::2, 0] = SED_filt[:, 0]
                SED[2::2, 0] = SED_filt[:, 0] + wv_step / 2
                SED[0, 0] = SED_filt[0, 0] - wv_step / 2
                SED[:, 1] = SED_interpolator(SED[:, 0])
                SED[:, 2] = np.ones(n_bins * 2 + 1)
                SED[0, 2], SED[-1, 2] = 0.5, 0.5
                SED[:, 1] *= SED[:, 2]
            else:
                SED = np.zeros((n_bins * 2 - 1, 3))
                SED[::2, 0] = SED_filt[:, 0]
                SED[1::2, 0] = SED_filt[1:, 0] - wv_step / 2
                SED[:, 1] = SED_interpolator(SED[:, 0])
                SED[:, 2] = np.ones(n_bins * 2 - 1)
                SED[0, 2], SED[-1, 2] = 1.5, 1.5
                SED[:, 1] *= SED[:, 2]
        elif n_points == 2:
            if self.SED_extrapolate:
                SED = np.zeros((n_bins * 3, 3))
                SED[1::3, 0] = SED_filt[:, 0]
                SED[::3, 0] = SED_filt[:, 0] - wv_step / 3
                SED[2::3, 0] = SED_filt[:, 0] + wv_step / 3
                SED[:, 1] = SED_interpolator(SED[:, 0])
                SED[:, 2] = np.ones(n_bins * 3)
                SED[:, 1] *= SED[:, 2]
            else:
                SED = np.zeros((n_bins * 3 - 2, 3))
                SED[::3, 0] = SED_filt[:, 0]
                SED[1::3, 0] = SED_filt[1:, 0] - 2 * wv_step / 3
                SED[2::3, 0] = SED_filt[1:, 0] - wv_step / 3
                SED[:, 1] = SED_interpolator(SED[:, 0])
                SED[:, 2] = np.ones(n_bins * 3 - 2)
                SED[0, 2], SED[-1, 2] = 2, 2
                SED[:, 1] *= SED[:, 2]
        elif n_points == 3:
            if self.SED_extrapolate:
                SED = np.zeros((n_bins * 4 + 1, 3))
                SED[4::4, 0] = SED_filt[:, 0] + wv_step / 2
                SED[0, 0] = SED_filt[0, 0] - wv_step / 2
                SED[1::4, 0] = SED_filt[:, 0] - wv_step / 4
                SED[2::4, 0] = SED_filt[:, 0]
                SED[3::4, 0] = SED_filt[:, 0] + wv_step / 4
                SED[:, 1] = SED_interpolator(SED[:, 0])
                SED[:, 2] = np.ones(n_bins * 4 + 1)
                SED[0, 2], SED[-1, 2] = 0.5, 0.5
                SED[:, 1] *= SED[:, 2]
        else:
            SED = SED_filt

        # Normalize SED
        SED[:, 1] = SED[:, 1] / np.sum(SED[:, 1])

        return SED

    def gen_SED_sampler(self, SED, n_bins, interp_kind="cubic"):
        """Generate SED sampler.

        Parameters
        ----------
        SED : np.ndarray
            Spectral energy distribution for a star.
        n_bins : int
            Number of bins.
        interp_kind : str
            Interpolation kind.

        Returns
        -------
        wvlength : np.ndarray
            Wavelength positions in [nm].
        SED_sampler : callable
            Interpolator function.
        weights : np.ndarray
            Weights for each wavelength bin.
        """
        SED_filt = self.filter_SED(SED, n_bins)

        SED_filt[:, 1] += np.multiply(
            SED_filt[:, 1], self.SED_gen_noise(len(SED_filt), self.SED_sigma)
        )
        SED_filt[:, 1] = SED_filt[:, 1] / np.sum(SED_filt[:, 1])

        SED_filt = self.interp_SED(
            SED_filt, self.SED_interp_pts_per_bin, n_bins, self.SED_interp_kind
        )

        if SED_filt.shape[1] == 2:
            weights = np.ones((SED_filt.shape[0], 1))
            SED_filt = np.hstack((SED_filt, weights))

        SED_sampler = sinterp.interp1d(
            SED_filt[:, 0],
            SED_filt[:, 1] / SED_filt[:, 2],
            kind=interp_kind,
            bounds_error=False,
            fill_value="extrapolate",
        )

        return SED_filt[:, 0], SED_sampler, SED_filt[:, 2]

    def calc_SED_wave_values(self, SED, n_bins=35):
        """Calculate feasible wavelength and SED values.

        Parameters
        ----------
        SED : np.ndarray
            Spectral energy distribution for a star.
        n_bins : int
            Number of bins.

        Returns
        -------
        feasible_wv : np.ndarray
            Feasible wavelength positions in [um].
        SED_norm : np.ndarray
            Normalized SED values.
        """
        wvlength, SED_interp, weights = self.gen_SED_sampler(SED, n_bins)

        # Convert wavelength from [nm] to [um]
        wvlength_um = wvlength / 1e3

        # Calculate feasible wavelengths (in [um])
        verbose = self.verbose
        self.verbose = 0
        feasible_wv = np.array(
            [self.feasible_wavelength(_wv) for _wv in wvlength_um]
        )
        self.verbose = verbose

        # Interpolate and normalize SED
        SED_norm = SED_interp(feasible_wv * 1e3)
        SED_norm *= weights
        SED_norm /= np.sum(SED_norm)

        return feasible_wv, SED_norm

    def generate_poly_PSF(self, SED, n_bins=35):
        """Generate polychromatic PSF with a specific SED.

        The wavelength space will be the Euclid VIS instrument band:
        [550,900]nm and will be sampled in ``n_bins``.

        Parameters
        ----------
        SED : np.ndarray
            Spectral energy distribution.
        n_bins : int
            Number of wavelength bins.

        Returns
        -------
        stacked_psf : np.ndarray
            Polychromatic PSF.
        """
        feasible_wv, SED_norm = self.calc_SED_wave_values(SED, n_bins)

        stacked_psf = 0

        for it in range(feasible_wv.shape[0]):
            self.generate_mono_PSF(lambda_obs=feasible_wv[it])
            stacked_psf += self.get_psf() * SED_norm[it]

        self.poly_psf = stacked_psf

        return stacked_psf
