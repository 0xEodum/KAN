"""NumPy interface to the optional KAN C++ extension."""
import os as _os
import sys as _sys


def _add_dll_directories():
    """Windows: Python does not search PATH for extension dependencies. A CUDA
    build links the cuBLAS runtime DLL; register the toolkit directories the
    build recorded (kan/_dll_paths.py) and those of CUDA_PATH."""
    if _sys.platform != "win32":
        return []
    try:
        from ._dll_paths import DIRECTORIES
    except ImportError:
        DIRECTORIES = ()
    candidates = list(DIRECTORIES)
    if _os.environ.get("CUDA_PATH"):
        candidates += [_os.path.join(_os.environ["CUDA_PATH"], "bin", "x64"),
                       _os.path.join(_os.environ["CUDA_PATH"], "bin")]
    # Keep the handles: closing one removes its directory from the search.
    return [_os.add_dll_directory(d) for d in dict.fromkeys(candidates) if _os.path.isdir(d)]


_dll_directories = _add_dll_directories()

from _kan import (ChebyshevConfig, LegendreConfig, JacobiConfig, HermiteConfig, FourierConfig,
                  GaussianRbfConfig, TrainableRbfConfig, BSplineConfig, MexicanHatConfig, basis_size)
from _kan import DenominatorPolicy, RationalConfig, evaluate_basis, evaluate_rational, Layer, Network
from _kan import BasisEdges, TrainableRbfEdges, RationalEdges
from _kan import insert_knot, adapt_grid, set_rbf_parameters, set_rational_parameters
from _kan import AffineMap, TanhMap, LayerNormMap, InputMap, affine_from_range, affine_from_moments
from _kan import LayerGradients, InputMapGradients, NetworkGradients, cuda_enabled, cuda_available, Precision, Loss

if cuda_enabled:
    from _kan import ResidentNetwork
