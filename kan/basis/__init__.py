"""
Basis functions for Kolmogorov-Arnold Networks.

This module provides various basis functions that can be used in KAN layers
for representing univariate functions according to the Kolmogorov-Arnold theorem.
"""

from .base import BasisFunction
from .chebyshev import ChebyshevBasis
from .jacobi import JacobiBasis

# Dictionary mapping basis names to their classes
BASIS_REGISTRY = {
    'chebyshev': ChebyshevBasis,
    'jacobi': JacobiBasis,
    'legendre': lambda **kwargs: JacobiBasis(alpha=0.0, beta=0.0, **kwargs),
    'chebyshev1': lambda **kwargs: JacobiBasis(alpha=-0.5, beta=-0.5, **kwargs),
    'chebyshev2': lambda **kwargs: JacobiBasis(alpha=0.5, beta=0.5, **kwargs),
    'gegenbauer': lambda lambda_param=1.0, **kwargs: JacobiBasis(
        alpha=lambda_param-0.5, beta=lambda_param-0.5, **kwargs
    ),
}


def get_basis(name: str, **kwargs):
    """
    Get a basis function by name.
    
    Args:
        name: Name of the basis function
        **kwargs: Additional arguments to pass to the basis constructor
        
    Returns:
        Instance of the requested basis function
    """
    if name not in BASIS_REGISTRY:
        raise ValueError(f"Unknown basis function: {name}. Available options: "
                       f"{', '.join(BASIS_REGISTRY.keys())}")
    
    return BASIS_REGISTRY[name](**kwargs)