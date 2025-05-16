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
    # Convenience aliases for specific polynomial configurations
    'legendrekan': lambda input_dim, output_dim, degree, **kwargs: JacobiKANLayer(
        input_dim=input_dim, output_dim=output_dim, degree=degree, 
        alpha=0.0, beta=0.0, **kwargs
    ),
    'gegenbauerkan': lambda input_dim, output_dim, degree, lambda_param=1.0, **kwargs: JacobiKANLayer(
        input_dim=input_dim, output_dim=output_dim, degree=degree,
        alpha=lambda_param-0.5, beta=lambda_param-0.5, **kwargs
    ),
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