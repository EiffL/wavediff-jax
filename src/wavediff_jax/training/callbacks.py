"""Training callbacks and scheduling utilities.

Provides checkpoint save/load via Equinox serialization and L1 rate scheduling.

:Authors: WaveDiff-JAX contributors
"""

import equinox as eqx
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(model, path):
    """Save model checkpoint via Equinox leaf serialization.

    Parameters
    ----------
    model : eqx.Module
        The model to serialize.
    path : str or pathlib.Path
        Destination file path.
    """
    eqx.tree_serialise_leaves(path, model)


def load_checkpoint(model_template, path):
    """Load a model checkpoint.

    Parameters
    ----------
    model_template : eqx.Module
        A model instance with the same structure as the saved model.
        The structure is used for deserialization; leaf values are replaced.
    path : str or pathlib.Path
        Source file path.

    Returns
    -------
    eqx.Module
        The deserialized model.
    """
    return eqx.tree_deserialise_leaves(path, model_template)


def l1_schedule_rule(epoch, l1_rate):
    """Schedule L1 rate: halve every 10 epochs (except epoch 0).

    Matches the TF reference ``l1_schedule_rule`` in ``train_utils.py``.

    Parameters
    ----------
    epoch : int
        Current epoch index (0-based).
    l1_rate : float
        Current L1 regularization rate.

    Returns
    -------
    float
        Updated L1 rate.
    """
    if epoch != 0 and epoch % 10 == 0:
        scheduled = l1_rate / 2.0
        logger.info("Epoch %05d: L1 rate is %.4e.", epoch, scheduled)
        return scheduled
    return l1_rate
