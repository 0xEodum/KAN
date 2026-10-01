"""NumPy interface to the optional KAN C++ extension."""
from _kan import BasisKind, BasisConfig, RationalConfig, evaluate_basis, evaluate_rational, Layer, Network
from _kan import LayerGradients, NetworkGradients, cuda_enabled, cuda_available

if cuda_enabled:
    from _kan import ResidentNetwork
