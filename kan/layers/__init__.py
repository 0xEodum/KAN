"""
KAN layers for building Kolmogorov-Arnold Networks.

This module provides various layer implementations for KAN architectures,
each using different basis functions to represent univariate components
of the Kolmogorov-Arnold decomposition.
"""

from .base import KANLayer, KANSequential
from .chebykan import ChebyKANLayer
from .jacobikan import JacobiKANLayer

# Dictionary mapping layer names to their classes
LAYER_REGISTRY = {
    'chebykan': ChebyKANLayer,
    'jacobikan': JacobiKANLayer,
}


def get_layer(name: str, **kwargs):
    """
    Get a KAN layer by name.
    
    Args:
        name: Name of the layer
        **kwargs: Additional arguments to pass to the layer constructor
        
    Returns:
        Instance of the requested KAN layer
    """
    if name not in LAYER_REGISTRY:
        raise ValueError(f"Unknown layer: {name}. Available options: "
                       f"{', '.join(LAYER_REGISTRY.keys())}")
    
    return LAYER_REGISTRY[name](**kwargs)