"""scvi-tools."""

# Set default logging handler to avoid logging with logging.lastResort logger.
import logging
import warnings

from ._constants import REGISTRY_KEYS
from ._settings import settings

# this import needs to come after prior imports to prevent circular import
from . import data, model, external, utils

from importlib.metadata import version

try:
    from importlib.metadata import version
    __version__ = version("scvi-tools")
except Exception:
    __version__ = "custom"

settings.verbosity = logging.INFO

# Jax sets the root logger, this prevents double output.
scvi_logger = logging.getLogger("scvi")
scvi_logger.propagate = False


__all__ = [
    "settings",
    "REGISTRY_KEYS",
    "data",
    "model",
    "external",
    "utils",
    "criticism",
]
