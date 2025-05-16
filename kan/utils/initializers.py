import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple, Callable


def init_chebyshev_uniform(tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Initialize Chebyshev coefficients with uniform distribution.
    
    This initialization is based on the idea that the magnitude of
    Chebyshev coefficients typically decreases with increasing degree
    for smooth functions.
    
    Args:
        tensor: Tensor to initialize (shape: input_dim, output_dim, degree+1)
        scale: Scaling factor for initialization
        
    Returns:
        Initialized tensor
    """
    input_dim, output_dim, degree_plus_one = tensor.shape
    degree = degree_plus_one - 1
    
    # Initialize with uniform distribution scaled by degree
    bound = scale / (input_dim * math.sqrt(degree + 1))
    nn.init.uniform_(tensor, -bound, bound)
    
    # Apply degree-based scaling to favour lower degrees
    degree_factors = torch.linspace(1.0, 0.1, degree + 1)
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_chebyshev_normal(tensor: torch.Tensor, 
                         mean: float = 0.0, 
                         std: Optional[float] = None) -> torch.Tensor:
    """
    Initialize Chebyshev coefficients with normal distribution.
    
    Args:
        tensor: Tensor to initialize (shape: input_dim, output_dim, degree+1)
        mean: Mean of the normal distribution
        std: Standard deviation (if None, uses 1/sqrt(input_dim * (degree+1)))
        
    Returns:
        Initialized tensor
    """
    input_dim, output_dim, degree_plus_one = tensor.shape
    degree = degree_plus_one - 1
    
    # Set default std if not provided
    if std is None:
        std = 1.0 / math.sqrt(input_dim * (degree + 1))
    
    # Initialize with normal distribution
    nn.init.normal_(tensor, mean=mean, std=std)
    
    # Apply degree-based scaling to favour lower degrees
    degree_factors = torch.linspace(1.0, 0.1, degree + 1)
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_chebyshev_orthogonal(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Initialize Chebyshev coefficients with orthogonal initialization.
    
    This is adapted from nn.init.orthogonal_ but modified to work with
    the 3D tensor structure of KAN coefficients.
    
    Args:
        tensor: Tensor to initialize (shape: input_dim, output_dim, degree+1)
        gain: Scaling factor
        
    Returns:
        Initialized tensor
    """
    input_dim, output_dim, degree_plus_one = tensor.shape
    degree = degree_plus_one - 1
    
    # Reshape for orthogonal initialization
    flat_shape = (input_dim, output_dim * degree_plus_one)
    reshaped = tensor.reshape(flat_shape)
    
    # Apply orthogonal initialization
    nn.init.orthogonal_(reshaped, gain=gain)
    
    # Reshape back
    tensor.copy_(reshaped.reshape(input_dim, output_dim, degree_plus_one))
    
    # Apply degree-based scaling
    degree_factors = torch.linspace(1.0, 0.1, degree + 1)
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_chebyshev_zeros(tensor: torch.Tensor) -> torch.Tensor:
    """
    Initialize Chebyshev coefficients with zeros except for the constant term.
    
    This initialization sets all coefficients to zero except for the
    constant term (degree 0), which is initialized to small random values.
    This is useful for starting with an almost-identity function.
    
    Args:
        tensor: Tensor to initialize (shape: input_dim, output_dim, degree+1)
        
    Returns:
        Initialized tensor
    """
    # Set all coefficients to zero
    nn.init.zeros_(tensor)
    
    # Initialize only the constant term (degree 0)
    input_dim, output_dim, _ = tensor.shape
    const_term = tensor[:, :, 0]
    nn.init.normal_(const_term, mean=0.0, std=0.01)
    
    # For input dimension i, set the linear term (degree 1) for output i to near 1
    # This helps create an approximately identity initial function
    for i in range(min(input_dim, output_dim)):
        tensor[i, i, 1] = 1.0 + torch.randn(1).item() * 0.01
    
    return tensor


def init_chebyshev_identity(tensor: torch.Tensor, 
                           exact: bool = False,
                           noise_scale: float = 0.01) -> torch.Tensor:
    """
    Initialize Chebyshev coefficients to approximate the identity function.
    
    For the identity function f(x) = x, the Chebyshev expansion has
    T_1(x) = x as the only non-zero term. This initialization sets
    coefficients to create an identity-like mapping with optional noise.
    
    Args:
        tensor: Tensor to initialize (shape: input_dim, output_dim, degree+1)
        exact: Whether to use exact identity or add noise
        noise_scale: Scale of the noise to add if not exact
        
    Returns:
        Initialized tensor
    """
    # Set all coefficients to zero
    nn.init.zeros_(tensor)
    
    input_dim, output_dim, degree_plus_one = tensor.shape
    
    # Check if degree is at least 1 (we need T_1(x) = x for identity)
    if degree_plus_one < 2:
        raise ValueError("Degree must be at least 1 for identity initialization")
    
    # Identity mapping: set coefficient for T_1(x) = x
    min_dim = min(input_dim, output_dim)
    for i in range(min_dim):
        tensor[i, i, 1] = 1.0
    
    # Add noise if not exact
    if not exact:
        noise = torch.randn_like(tensor) * noise_scale
        tensor += noise
    
    return tensor


def get_initializer(name: str) -> Callable:
    """
    Get initializer function by name.
    
    Args:
        name: Name of the initializer
        
    Returns:
        Initializer function
    """
    initializers = {
        'normal': init_chebyshev_normal,
        'uniform': init_chebyshev_uniform,
        'orthogonal': init_chebyshev_orthogonal,
        'zeros': init_chebyshev_zeros,
        'identity': init_chebyshev_identity
    }
    
    if name not in initializers:
        raise ValueError(f"Unknown initializer: {name}. Available initializers: "
                       f"{', '.join(initializers.keys())}")
    
    return initializers[name]