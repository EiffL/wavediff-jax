"""Metrics Interface.

Orchestrates evaluation of trained PSF models using various metrics
(polychromatic, monochromatic, OPD, shape).

Port of ``extern/wf-psf/src/wf_psf/metrics/metrics_interface.py`` to JAX.

:Authors: WaveDiff-JAX contributors
"""

import numpy as np
import time
import logging

from wavediff_jax.metrics import metrics as wf_metrics

logger = logging.getLogger(__name__)


class MetricsParamsHandler:
    """Metrics Parameters Handler.

    Orchestrates metric evaluations for a trained PSF model against a
    ground-truth model.

    Parameters
    ----------
    metrics_params : object
        Configuration namespace containing metrics hyperparameters and
        ground-truth model parameters.
    trained_model : object
        Configuration namespace containing trained model parameters.
    """

    def __init__(self, metrics_params, trained_model):
        self.metrics_params = metrics_params
        self.trained_model = trained_model

    def evaluate_metrics_polychromatic_lowres(
        self, psf_model, gt_model, dataset
    ):
        """Evaluate RMSE metrics for low-resolution polychromatic PSFs.

        Parameters
        ----------
        psf_model : eqx.Module
            The trained PSF model to evaluate.
        gt_model : eqx.Module
            The ground-truth PSF model.
        dataset : dict
            Dictionary with keys ``'positions'``, ``'packed_seds'``, and
            optionally ``'stars'``, ``'masks'``.

        Returns
        -------
        dict
            Dictionary with ``rmse``, ``rel_rmse``, ``std_rmse``,
            ``std_rel_rmse``.
        """
        logger.info("Computing polychromatic metrics at low resolution.")

        gt_stars = dataset.get("stars", None)
        masks = dataset.get("masks", None)

        rmse, rel_rmse, std_rmse, std_rel_rmse = wf_metrics.compute_poly_metric(
            model=psf_model,
            gt_model=gt_model,
            positions=dataset["positions"],
            packed_seds=dataset["packed_seds"],
            gt_packed_seds=dataset.get("gt_packed_seds", None),
            gt_stars=gt_stars,
            batch_size=getattr(
                self.metrics_params, "batch_size",
                getattr(self.metrics_params, "metrics_hparams", self.metrics_params),
            )
            if not hasattr(self.metrics_params, "metrics_hparams")
            else self.metrics_params.metrics_hparams.batch_size,
            masks=masks,
        )

        return {
            "rmse": rmse,
            "rel_rmse": rel_rmse,
            "std_rmse": std_rmse,
            "std_rel_rmse": std_rel_rmse,
        }

    def evaluate_metrics_mono_rmse(
        self, psf_model, gt_model, dataset, lambda_list=None, phase_N_fn=None
    ):
        """Evaluate RMSE metrics for monochromatic PSFs.

        Parameters
        ----------
        psf_model : eqx.Module
            The trained PSF model.
        gt_model : eqx.Module
            The ground-truth PSF model.
        dataset : dict
            Dictionary with ``'positions'``.
        lambda_list : array-like or None
            Wavelengths to evaluate. Defaults to 0.55-0.90 um in 10 nm steps.
        phase_N_fn : callable or None
            Maps wavelength to phase_N integer.

        Returns
        -------
        dict
            Dictionary with ``rmse_lda``, ``rel_rmse_lda``, ``std_rmse_lda``,
            ``std_rel_rmse_lda``.
        """
        logger.info("Computing monochromatic metrics.")

        if lambda_list is None:
            lambda_list = np.arange(0.55, 0.9, 0.01)

        rmse_lda, rel_rmse_lda, std_rmse_lda, std_rel_rmse_lda = (
            wf_metrics.compute_mono_metric(
                model=psf_model,
                gt_model=gt_model,
                positions=dataset["positions"],
                lambda_list=lambda_list,
                phase_N_fn=phase_N_fn,
            )
        )

        return {
            "rmse_lda": rmse_lda,
            "rel_rmse_lda": rel_rmse_lda,
            "std_rmse_lda": std_rmse_lda,
            "std_rel_rmse_lda": std_rel_rmse_lda,
        }

    def evaluate_metrics_opd(self, psf_model, gt_model, dataset):
        """Evaluate OPD metrics.

        Parameters
        ----------
        psf_model : eqx.Module
            The trained PSF model.
        gt_model : eqx.Module
            The ground-truth PSF model.
        dataset : dict
            Dictionary with ``'positions'``.

        Returns
        -------
        dict
            Dictionary with ``rmse_opd``, ``rel_rmse_opd``, ``rmse_std_opd``,
            ``rel_rmse_std_opd``.
        """
        logger.info("Computing OPD metrics.")

        rmse_opd, rel_rmse_opd, rmse_std_opd, rel_rmse_std_opd = (
            wf_metrics.compute_opd_metrics(
                model=psf_model,
                gt_model=gt_model,
                positions=dataset["positions"],
            )
        )

        return {
            "rmse_opd": rmse_opd,
            "rel_rmse_opd": rel_rmse_opd,
            "rmse_std_opd": rmse_std_opd,
            "rel_rmse_std_opd": rel_rmse_std_opd,
        }

    def evaluate_metrics_shape(
        self, psf_model, gt_model, dataset
    ):
        """Evaluate PSF shape metrics.

        Parameters
        ----------
        psf_model : eqx.Module
            The trained PSF model.
        gt_model : eqx.Module
            The ground-truth PSF model.
        dataset : dict
            Dictionary with ``'positions'``, ``'packed_seds'``, and optionally
            ``'super_res_stars'`` or ``'SR_stars'``.

        Returns
        -------
        dict
            Shape metrics dictionary from :func:`compute_shape_metrics`.
        """
        logger.info("Computing shape metrics.")

        positions = dataset["positions"]
        packed_seds = dataset["packed_seds"]

        # Get predictions
        predictions = wf_metrics.compute_psf_images(
            psf_model, positions, packed_seds
        )

        # Get ground truth
        gt_key = None
        for key in ("super_res_stars", "SR_stars"):
            if key in dataset:
                gt_key = key
                break

        if gt_key is not None:
            gt_predictions = np.asarray(dataset[gt_key])
        else:
            gt_packed_seds = dataset.get("gt_packed_seds", packed_seds)
            gt_predictions = wf_metrics.compute_psf_images(
                gt_model, positions, gt_packed_seds
            )

        return wf_metrics.compute_shape_metrics(predictions, gt_predictions)


def evaluate_model(
    psf_model,
    gt_model,
    train_dataset,
    test_dataset,
    metrics_params=None,
    trained_model_params=None,
    output_path=None,
    eval_flags=None,
):
    """Evaluate a trained model on training and test datasets.

    Parameters
    ----------
    psf_model : eqx.Module
        The trained PSF model.
    gt_model : eqx.Module
        The ground-truth PSF model.
    train_dataset : dict
        Training dataset dictionary.
    test_dataset : dict
        Test dataset dictionary.
    metrics_params : object or None
        Metrics configuration namespace.
    trained_model_params : object or None
        Trained model configuration namespace.
    output_path : str or None
        Path to save metrics results. If None, results are not saved.
    eval_flags : dict or None
        Dictionary specifying which metrics to evaluate. Keys should be
        ``'poly_metric'``, ``'mono_metric'``, ``'opd_metric'``,
        ``'shape_results_dict'``. Values should be booleans. Defaults to
        only ``'poly_metric': True``.

    Returns
    -------
    all_metrics : dict
        Nested dictionary with keys ``'train_metrics'`` and
        ``'test_metrics'``.
    """
    starting_time = time.time()

    if eval_flags is None:
        eval_flags = {
            "poly_metric": True,
            "mono_metric": False,
            "opd_metric": False,
            "shape_results_dict": False,
        }

    handler = MetricsParamsHandler(
        metrics_params=metrics_params,
        trained_model=trained_model_params,
    )

    datasets = {"test": test_dataset, "train": train_dataset}

    metric_functions = {
        "poly_metric": handler.evaluate_metrics_polychromatic_lowres,
        "mono_metric": handler.evaluate_metrics_mono_rmse,
        "opd_metric": handler.evaluate_metrics_opd,
        "shape_results_dict": handler.evaluate_metrics_shape,
    }

    all_metrics = {}

    for dataset_type, dataset in datasets.items():
        logger.info("Metric evaluation on the %s dataset", dataset_type)
        dataset_metrics = {}

        for metric_name, metric_fn in metric_functions.items():
            if eval_flags.get(metric_name, False):
                dataset_metrics[metric_name] = metric_fn(
                    psf_model, gt_model, dataset
                )
            else:
                dataset_metrics[metric_name] = None

        all_metrics[f"{dataset_type}_metrics"] = dataset_metrics

    if output_path is not None:
        np.save(output_path, all_metrics, allow_pickle=True)

    elapsed = time.time() - starting_time
    logger.info("Total elapsed time: %.2f s", elapsed)

    return all_metrics
