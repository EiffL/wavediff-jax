"""WaveDiff-JAX: JAX reimplementation of the WaveDiff PSF modelling framework."""

# Trigger model registration via @register_psfclass decorators
import wavediff_jax.models.parametric  # noqa: F401
import wavediff_jax.models.semiparametric  # noqa: F401
import wavediff_jax.models.physical_polychromatic  # noqa: F401
import wavediff_jax.models.ground_truth  # noqa: F401

# Public API
from wavediff_jax.models.registry import (  # noqa: F401
    get_psf_model,
    register_psfclass,
    PSF_FACTORY,
)
from wavediff_jax.training.trainer import (  # noqa: F401
    general_train_cycle,
    train_epoch,
    make_step,
)
from wavediff_jax.training.losses import (  # noqa: F401
    total_loss,
    mse_loss,
    masked_mse_loss,
)
from wavediff_jax.training.callbacks import (  # noqa: F401
    save_checkpoint,
    load_checkpoint,
)
from wavediff_jax.data.preprocessing import DataHandler  # noqa: F401
from wavediff_jax.metrics.metrics import (  # noqa: F401
    compute_poly_metric,
    compute_mono_metric,
    compute_opd_metrics,
    compute_shape_metrics,
)
