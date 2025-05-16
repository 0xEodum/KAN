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


def init_jacobi_uniform(tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Initialize Jacobi coefficients with uniform distribution.
    
    This initialization is similar to the Chebyshev case but adapted for Jacobi polynomials.
    
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
    # For Jacobi polynomials, we decrease weights more rapidly with degree
    # since higher degree polynomials can have larger magnitudes
    degree_factors = torch.exp(-torch.linspace(0.0, 2.0, degree + 1))
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_jacobi_normal(tensor: torch.Tensor, 
                      mean: float = 0.0, 
                      std: Optional[float] = None) -> torch.Tensor:
    """
    Initialize Jacobi coefficients with normal distribution.
    
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
    
    # Apply degree-based scaling
    degree_factors = torch.exp(-torch.linspace(0.0, 2.0, degree + 1))
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_jacobi_orthogonal(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Initialize Jacobi coefficients with orthogonal initialization.
    
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
    degree_factors = torch.exp(-torch.linspace(0.0, 2.0, degree + 1))
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_jacobi_identity(tensor: torch.Tensor, 
                        alpha: float = 0.0, 
                        beta: float = 0.0,
                        exact: bool = False,
                        noise_scale: float = 0.01) -> torch.Tensor:
    """
    Initialize Jacobi coefficients to approximate the identity function.
    
    For the identity function f(x) = x, we need to determine the coefficients
    in the Jacobi expansion. For α = β = 0 (Legendre), the coefficient of P_1 is 1,
    while for other values of α, β, we need specific values.
    
    Args:
        tensor: Tensor to initialize (shape: input_dim, output_dim, degree+1)
        alpha: α parameter of Jacobi polynomials
        beta: β parameter of Jacobi polynomials
        exact: Whether to use exact identity or add noise
        noise_scale: Scale of the noise to add if not exact
        
    Returns:
        Initialized tensor
    """
    # Set all coefficients to zero
    nn.init.zeros_(tensor)
    
    input_dim, output_dim, degree_plus_one = tensor.shape
    
    # Check if degree is at least 1 (we need at least P_1 for identity)
    if degree_plus_one < 2:
        raise ValueError("Degree must be at least 1 for identity initialization")
    
    # For Jacobi polynomials P_1^(α,β)(x) = ((α + β + 2)x + (α - β))/2
    # To represent identity f(x) = x, we need to adjust the coefficient
    identity_coef = 2.0 / (alpha + beta + 2)  # To make the x coefficient = 1
    
    # Identity mapping: set coefficient for P_1
    min_dim = min(input_dim, output_dim)
    for i in range(min_dim):
        tensor[i, i, 1] = identity_coef
    
    # If α ≠ β, we need a constant term to cancel the (α - β) term in P_1
    if alpha != beta and degree_plus_one > 0:
        const_coef = -(alpha - beta) * identity_coef / 2
        for i in range(min_dim):
            tensor[i, i, 0] = const_coef
    
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
        'chebyshev_normal': init_chebyshev_normal,
        'chebyshev_uniform': init_chebyshev_uniform,
        'chebyshev_orthogonal': init_chebyshev_orthogonal,
        'chebyshev_zeros': init_chebyshev_zeros,
        'chebyshev_identity': init_chebyshev_identity,
        'jacobi_normal': init_jacobi_normal,
        'jacobi_uniform': init_jacobi_uniform,
        'jacobi_orthogonal': init_jacobi_orthogonal,
        'jacobi_identity': init_jacobi_identity,
        # For backward compatibility, keep the original names
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