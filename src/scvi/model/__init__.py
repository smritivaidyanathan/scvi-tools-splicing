from . import utils
from ._amortizedlda import AmortizedLDA
from ._autozi import AUTOZI
from ._condscvi import CondSCVI
from ._destvi import DestVI
from ._jaxscvi import JaxSCVI
from ._linear_scvi import LinearSCVI
from ._multivi import MULTIVI
from ._scvi_lin import SCVI_Linear
from ._multivi_splice import MULTIVISPLICE
from ._peakvi import PEAKVI
from ._scanvi import SCANVI
from ._scvi import SCVI
from ._totalvi import TOTALVI
from ._splicevi import SPLICEVI
from ._utils import get_max_epochs_heuristic

__all__ = [
    "SCVI",
    "TOTALVI",
    "LinearSCVI",
    "AUTOZI",
    "SCANVI",
    "PEAKVI",
    "CondSCVI",
    "DestVI",
    "MULTIVI",
    "MULTIVISPLICE",
    "AmortizedLDA",
    "SPLICEVI",
    "SCVI_Linear",
    "utils",
    "JaxSCVI",
]
