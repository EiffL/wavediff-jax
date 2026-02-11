"""PSF model factory and registry."""

import logging

logger = logging.getLogger(__name__)

PSF_FACTORY = {}


class PSFModelError(Exception):
    def __init__(self, message="An error with your PSF model parameter settings occurred."):
        self.message = message
        super().__init__(self.message)


def register_psfclass(psf_factory_class):
    """Register a PSF factory class."""
    for id in psf_factory_class.ids:
        PSF_FACTORY[id] = psf_factory_class
        logger.info(f"Registered PSF model: {id}")
    return psf_factory_class


class PSFModelBaseFactory:
    """Base factory for PSF models."""

    def get_model_instance(self, model_params, training_params, data=None, coeff_matrix=None):
        raise NotImplementedError


def set_psf_model(model_name):
    try:
        return PSF_FACTORY[model_name]
    except KeyError:
        raise PSFModelError(f"PSF model '{model_name}' is invalid.")


def get_psf_model(*psf_model_params):
    model_name = psf_model_params[0].model_name
    factory = set_psf_model(model_name)
    return factory().get_model_instance(*psf_model_params)
