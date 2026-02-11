"""Training Data Processing.

A module to load and preprocess training and validation test data.

Ported from wf-psf (CosmoStat/wf-psf) to wavediff-jax.
Original authors: Jennifer Pollack <jennifer.pollack@cea.fr> and Tobias Liaudat <tobiasliaudat@gmail.com>

"""

import os
import numpy as np
import jax.numpy as jnp
from wavediff_jax.utils.centroids import compute_zernike_tip_tilt
from wavediff_jax.utils.math_utils import calc_poly_position_mat
from fractions import Fraction
import logging

logger = logging.getLogger(__name__)


def generate_SED_elems(SED, psf_simulator, n_bins=20):
    """Generate SED elements for PSF modeling.

    Computes feasible Zernike mode numbers, wavelength values, and normalized
    SED for a given spectral energy distribution (SED) sampled across specified
    wavelength bins.

    Parameters
    ----------
    SED : np.ndarray
        The unfiltered SED with shape (n_wavelengths, 2). The first column contains
        wavelength positions (in wavelength units), and the second column contains
        the corresponding SED flux values.
    psf_simulator : PSFSimulator
        An instance of the PSFSimulator class initialized with the correct
        optical and instrumental parameters.
    n_bins : int, optional
        Number of wavelength bins to sample the SED. Default is 20.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray or float)
        - feasible_N : np.ndarray, shape (n_bins,)
            Feasible Zernike mode numbers at each wavelength bin.
        - feasible_wv : np.ndarray, shape (n_bins,)
            Sampled wavelength values across the SED.
        - SED_norm : np.ndarray or float
            Normalized SED values corresponding to feasible wavelengths.
    """
    feasible_wv, SED_norm = psf_simulator.calc_SED_wave_values(SED, n_bins)
    feasible_N = np.array([psf_simulator.feasible_N(_wv) for _wv in feasible_wv])

    return feasible_N, feasible_wv, SED_norm


def generate_SED_elems_jax(SED, psf_simulator, n_bins=20):
    """Generate SED elements as JAX arrays.

    A function to generate the SED elements needed for the JAX PSF models.

    Parameters
    ----------
    SED : np.ndarray
        The unfiltered SED. The first column contains the wavelength positions.
        The second column contains the SED value at each wavelength.
    psf_simulator : PSFSimulator object
        An instance of the PSFSimulator class with the correct initialization values.
    n_bins : int
        Number of wavelength bins.

    Returns
    -------
    list of jnp.ndarray
        [feasible_N, feasible_wv, SED_norm] as JAX float64 arrays.
    """
    feasible_wv, SED_norm = psf_simulator.calc_SED_wave_values(SED, n_bins)
    feasible_N = np.array([psf_simulator.feasible_N(_wv) for _wv in feasible_wv])

    return [
        jnp.array(feasible_N, dtype=jnp.float64),
        jnp.array(feasible_wv, dtype=jnp.float64),
        jnp.array(SED_norm, dtype=jnp.float64),
    ]


def generate_packed_elems(SED, psf_simulator, n_bins=20):
    """Generate packed SED elements as JAX arrays.

    Wrapper around generate_SED_elems that converts the returned NumPy
    arrays into JAX arrays with dtype=float64.

    Parameters
    ----------
    SED : numpy.ndarray
        The unfiltered SED with shape (n_wavelengths, 2). The first column contains
        the wavelength positions, and the second column contains the corresponding
        SED flux values.
    psf_simulator : PSFSimulator object
        An instance of the PSF simulator providing calc_SED_wave_values and feasible_N.
    n_bins : int, optional
        Number of wavelength bins used to sample the SED (default 20).

    Returns
    -------
    list of jnp.ndarray
        [feasible_N, feasible_wv, SED_norm] as JAX float64 arrays.
    """
    feasible_N, feasible_wv, SED_norm = generate_SED_elems(
        SED, psf_simulator, n_bins=n_bins
    )

    feasible_N = jnp.array(feasible_N, dtype=jnp.float64)
    feasible_wv = jnp.array(feasible_wv, dtype=jnp.float64)
    SED_norm = jnp.array(SED_norm, dtype=jnp.float64)

    return [feasible_N, feasible_wv, SED_norm]


class DataHandler:
    """Data Handler.

    This class manages loading and processing of training and testing data for
    use during PSF model training and validation. It provides methods to access
    and preprocess the data.

    Parameters
    ----------
    dataset_type : str
        Type of dataset ("training" or "test").
    data_params : RecursiveNamespace
        Recursive Namespace object containing parameters for both 'training'
        and 'test' datasets.
    simPSF : PSFSimulator
        Instance of the PSFSimulator class for simulating PSF models.
    n_bins_lambda : int
        Number of wavelength bins for SED processing.
    load_data : bool, optional
        If True, data is loaded and processed during initialization. If False,
        data loading is deferred until explicitly called. Default is True.

    Attributes
    ----------
    dataset_type : str
        Type of dataset ("training" or "test").
    data_params : RecursiveNamespace
        Parameters for the current dataset type.
    dataset : dict or None
        Dictionary containing the loaded dataset.
    simPSF : PSFSimulator
        Instance of the PSFSimulator class.
    n_bins_lambda : int
        Number of wavelength bins.
    sed_data : jnp.ndarray or None
        JAX array containing processed SED data.
    load_data_on_init : bool
        Flag controlling whether data is loaded during initialization.
    """

    def __init__(
        self, dataset_type, data_params, simPSF, n_bins_lambda, load_data: bool = True
    ):
        self.dataset_type = dataset_type
        self.data_params = data_params.__dict__[dataset_type]
        self.simPSF = simPSF
        self.n_bins_lambda = n_bins_lambda
        self.dataset = None
        self.sed_data = None
        self.load_data_on_init = load_data
        if self.load_data_on_init:
            self.load_dataset()
            self.process_sed_data()

    def load_dataset(self):
        """Load dataset.

        Load the dataset based on the specified dataset type.
        """
        self.dataset = np.load(
            os.path.join(self.data_params.data_dir, self.data_params.file),
            allow_pickle=True,
        )[()]
        self.dataset["positions"] = jnp.array(
            self.dataset["positions"], dtype=jnp.float32
        )
        if self.dataset_type == "training":
            if "noisy_stars" in self.dataset:
                self.dataset["noisy_stars"] = jnp.array(
                    self.dataset["noisy_stars"], dtype=jnp.float32
                )
            else:
                logger.warning(f"Missing 'noisy_stars' in {self.dataset_type} dataset.")
        elif self.dataset_type == "test":
            if "stars" in self.dataset:
                self.dataset["stars"] = jnp.array(
                    self.dataset["stars"], dtype=jnp.float32
                )
            else:
                logger.warning(f"Missing 'stars' in {self.dataset_type} dataset.")

    def process_sed_data(self):
        """Process SED Data.

        Generate and process SED data using JAX arrays.
        """
        self.sed_data = [
            generate_SED_elems_jax(
                _sed, self.simPSF, n_bins=self.n_bins_lambda
            )
            for _sed in self.dataset["SEDs"]
        ]
        self.sed_data = jnp.array(self.sed_data, dtype=jnp.float32)
        self.sed_data = jnp.transpose(self.sed_data, axes=(0, 2, 1))


def get_np_obs_positions(data):
    """Get observed positions in numpy from the provided dataset.

    This method concatenates the positions of the stars from both the training
    and test datasets to obtain the observed positions.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    np.ndarray
        Numpy array containing the observed positions of the stars.
    """
    obs_positions = np.concatenate(
        (
            np.asarray(data.training_data.dataset["positions"]),
            np.asarray(data.test_data.dataset["positions"]),
        ),
        axis=0,
    )

    return obs_positions


def get_obs_positions(data):
    """Get observed positions from the provided dataset.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    jnp.ndarray
        JAX array containing the observed positions of the stars.
    """
    obs_positions = get_np_obs_positions(data)

    return jnp.array(obs_positions, dtype=jnp.float32)


def extract_star_data(data, train_key: str, test_key: str) -> np.ndarray:
    """Extract specific star-related data from training and test datasets.

    This function retrieves and concatenates specific star-related data from
    the training and test datasets based on the provided keys.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.
    train_key : str
        The key to retrieve data from the training dataset.
    test_key : str
        The key to retrieve data from the test dataset.

    Returns
    -------
    np.ndarray
        A NumPy array containing the concatenated data for the given keys.

    Raises
    ------
    KeyError
        If the specified keys do not exist in the training or test datasets.
    """
    missing_keys = [
        key
        for key, dataset in [
            (train_key, data.training_data.dataset),
            (test_key, data.test_data.dataset),
        ]
        if key not in dataset
    ]

    if missing_keys:
        raise KeyError(f"Missing keys in dataset: {missing_keys}")

    train_data = data.training_data.dataset[train_key]
    test_data = data.test_data.dataset[test_key]

    # Convert to NumPy if necessary (handles JAX arrays)
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)

    return np.concatenate((train_data, test_data), axis=0)


def get_np_zernike_prior(data):
    """Get the zernike prior from the provided dataset.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    zernike_prior : np.ndarray
        Numpy array containing the full prior.
    """
    zernike_prior = np.concatenate(
        (
            np.asarray(data.training_data.dataset["zernike_prior"]),
            np.asarray(data.test_data.dataset["zernike_prior"]),
        ),
        axis=0,
    )

    return zernike_prior


def compute_centroid_correction(model_params, data, batch_size: int = 1) -> np.ndarray:
    """Compute centroid corrections using Zernike polynomials.

    Parameters
    ----------
    model_params : RecursiveNamespace
        Object containing parameters for the PSF model.
    data : DataConfigHandler
        Object containing training and test datasets.
    batch_size : int, optional
        The batch size to use when processing the stars. Default is 1.

    Returns
    -------
    zernike_centroid_array : np.ndarray
        A 2D NumPy array of shape (n_stars, 3).
    """
    star_postage_stamps = extract_star_data(
        data=data, train_key="noisy_stars", test_key="stars"
    )

    # Get star mask catalogue only if "masks" exist in both datasets
    star_masks = None
    train_masks = data.training_data.dataset.get("masks")
    test_masks = data.test_data.dataset.get("masks")
    if (
        train_masks is not None
        and test_masks is not None
        and np.asarray(train_masks).size > 0
        and np.asarray(test_masks).size > 0
    ):
        star_masks = extract_star_data(data=data, train_key="masks", test_key="masks")

    pix_sampling = model_params.pix_sampling * 1e-6  # Change units from [um] to [m]

    reference_shifts = [
        float(Fraction(value)) for value in model_params.reference_shifts
    ]

    n_stars = len(star_postage_stamps)
    zernike_centroid_array = []

    for i in range(0, n_stars, batch_size):
        batch_postage_stamps = star_postage_stamps[i : i + batch_size]
        batch_masks = star_masks[i : i + batch_size] if star_masks is not None else None

        zk1_2_batch = -1.0 * compute_zernike_tip_tilt(
            batch_postage_stamps, batch_masks, pix_sampling, reference_shifts
        )

        zernike_centroid_array.append(
            np.pad(
                zk1_2_batch,
                pad_width=[(0, 0), (1, 0)],
                mode="constant",
                constant_values=0,
            )
        )

    return np.concatenate(zernike_centroid_array, axis=0)


def compute_ccd_misalignment(model_params, data):
    """Compute CCD misalignment.

    Parameters
    ----------
    model_params : RecursiveNamespace
        Object containing parameters for this PSF model class.
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    zernike_ccd_misalignment_array : np.ndarray
        Numpy array containing the Zernike contributions to model the CCD
        chip misalignments.
    """
    from wavediff_jax.utils.ccd_misalignments import CCDMisalignmentCalculator

    obs_positions = get_np_obs_positions(data)

    ccd_misalignment_calculator = CCDMisalignmentCalculator(
        tiles_path=model_params.ccd_misalignments_input_path,
        x_lims=model_params.x_lims,
        y_lims=model_params.y_lims,
        tel_focal_length=model_params.tel_focal_length,
        tel_diameter=model_params.tel_diameter,
    )
    zk4_values = np.array(
        [
            ccd_misalignment_calculator.get_zk4_from_position(single_pos)
            for single_pos in obs_positions
        ]
    ).reshape(-1, 1)

    zernike_ccd_misalignment_array = np.pad(
        zk4_values, pad_width=[(0, 0), (3, 0)], mode="constant", constant_values=0
    )

    return zernike_ccd_misalignment_array


def get_zernike_prior(model_params, data, batch_size: int = 16):
    """Get Zernike priors from the provided dataset.

    Parameters
    ----------
    model_params : RecursiveNamespace
        Object containing parameters for this PSF model class.
    data : DataConfigHandler
        Object containing training and test datasets.
    batch_size : int, optional
        The batch size to use when processing the stars. Default is 16.

    Returns
    -------
    jnp.ndarray
        JAX array containing the Zernike prior contribution.
    """
    zernike_contribution_list = []

    if model_params.use_prior:
        logger.info("Reading in Zernike prior into Zernike contribution list...")
        zernike_contribution_list.append(get_np_zernike_prior(data))

    if model_params.correct_centroids:
        logger.info("Adding centroid correction to Zernike contribution list...")
        zernike_contribution_list.append(
            compute_centroid_correction(model_params, data, batch_size)
        )

    if model_params.add_ccd_misalignments:
        logger.info("Adding CCD mis-alignments to Zernike contribution list...")
        zernike_contribution_list.append(compute_ccd_misalignment(model_params, data))

    if len(zernike_contribution_list) == 1:
        zernike_contribution = zernike_contribution_list[0]
    else:
        max_zk_order = np.max(
            np.array(
                [
                    zk_contribution.shape[1]
                    for zk_contribution in zernike_contribution_list
                ]
            )
        )

        zernike_contribution = np.zeros(
            (zernike_contribution_list[0].shape[0], max_zk_order)
        )

        for it in range(len(zernike_contribution_list)):
            current_zk_order = zernike_contribution_list[it].shape[1]
            current_zernike_contribution = np.pad(
                zernike_contribution_list[it],
                pad_width=[(0, 0), (0, int(max_zk_order - current_zk_order))],
                mode="constant",
                constant_values=0,
            )

            zernike_contribution += current_zernike_contribution

    return jnp.array(zernike_contribution, dtype=jnp.float32)
