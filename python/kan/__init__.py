"""NumPy interface to the optional KAN C++ extension."""
from _kan import (ChebyshevConfig, LegendreConfig, JacobiConfig, HermiteConfig, FourierConfig,
                  GaussianRbfConfig, TrainableRbfConfig, BSplineConfig, MexicanHatConfig, basis_size)
from _kan import RationalConfig, evaluate_basis, evaluate_rational, Layer, Network
from _kan import BasisEdges, TrainableRbfEdges, RationalEdges
from _kan import insert_knot, adapt_grid, set_rbf_parameters, set_rational_parameters
from _kan import AffineMap, TanhMap, LayerNormMap, InputMap, affine_from_range, affine_from_moments
from _kan import LayerGradients, InputMapGradients, NetworkGradients, cuda_enabled, cuda_available

if cuda_enabled:
    from _kan import ResidentNetwork
