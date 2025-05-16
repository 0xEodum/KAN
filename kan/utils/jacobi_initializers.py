import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple, Callable


def init_jacobi_uniform(tensor: torch.Tensor, alpha: float = 0.0, beta: float = 0.0, 
                        scale: float = 1.0) -> torch.Tensor:
    """
    Initialize Jacobi coefficients with uniform distribution.
    
    This initialization is based on the idea that the magnitude of
    Jacobi coefficients typically decreases with increasing degree
    for smooth functions.
    
    Args:
        tensor: Tensor to initialize (shape: input_dim, output_dim, degree+1)
        alpha: First Jacobi parameter (α > -1)
        beta: Second Jacobi parameter (β > -1)
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
    # The scaling is adjusted based on α and β parameters
    degree_factors = torch.linspace(1.0, 0.1, degree + 1)
    
    # For Chebyshev polynomials (α = β = -0.5), the degree factors
    # should decay more slowly since higher degrees are more important
    # for approximation in the max norm
    if abs(alpha + 0.5) < 1e-10 and abs(beta + 0.5) < 1e-10:
        degree_factors = torch.linspace(1.0, 0.3, degree + 1)
    
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_jacobi_normal(tensor: torch.Tensor, 
                      alpha: float = 0.0, 
                      beta: float = 0.0,
                      mean: float = 0.0, 
                      std: Optional[float] = None) -> torch.Tensor:
    """
    Initialize Jacobi coefficients with normal distribution.
    
    Args:
        tensor: Tensor to initialize (shape: input_dim, output_dim, degree+1)
        alpha: First Jacobi parameter (α > -1)
        beta: Second Jacobi parameter (β > -1)
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
    
    # Apply degree-based scaling adjusted by α and β
    degree_factors = torch.linspace(1.0, 0.1, degree + 1)
    
    # Adjust for specific polynomial types
    if abs(alpha + 0.5) < 1e-10 and abs(beta + 0.5) < 1e-10:  # Chebyshev first kind
        degree_factors = torch.linspace(1.0, 0.3, degree + 1)
    elif abs(alpha - 0.5) < 1e-10 and abs(beta - 0.5) < 1e-10:  # Chebyshev second kind
        degree_factors = torch.linspace(1.0, 0.2, degree + 1)
    
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_jacobi_orthogonal(tensor: torch.Tensor, 
                          alpha: float = 0.0, 
                          beta: float = 0.0,
                          gain: float = 1.0) -> torch.Tensor:
    """
    Initialize Jacobi coefficients with orthogonal initialization.
    
    This is adapted from nn.init.orthogonal_ but modified to work with
    the 3D tensor structure of KAN coefficients.
    
    Args:
        tensor: Tensor to initialize (shape: input_dim, output_dim, degree+1)
        alpha: First Jacobi parameter (α > -1)
        beta: Second Jacobi parameter (β > -1)
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


def init_jacobi_zeros(tensor: torch.Tensor, 
                     alpha: float = 0.0, 
                     beta: float = 0.0) -> torch.Tensor:
    """
    Initialize Jacobi coefficients with zeros except for the constant and linear terms.
    
    This initialization sets all coefficients to zero except for the
    constant term (degree 0) and linear term (degree 1) for diagonal elements,
    which helps create an approximately identity initial function.
    
    Args:
        tensor: Tensor to initialize (shape: input_dim, output_dim, degree+1)
        alpha: First Jacobi parameter (α > -1)
        beta: Second Jacobi parameter (β > -1)
        
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
    min_dim = min(input_dim, output_dim)
    for i in range(min_dim):
        # For Jacobi polynomials, linear component depends on α and β
        # P_1^(α,β)(x) = (α + β + 2)x/2 + (α - β)/2
        # We want P_1(0) ≈ 0 and P_1(1) ≈ 1, so we scale appropriately
        scale = 2.0 / (alpha + beta + 2) if alpha + beta + 2 != 0 else 1.0
        tensor[i, i, 1] = scale * (1.0 + torch.randn(1).item() * 0.01)
    
    return tensor


def init_jacobi_identity(tensor: torch.Tensor, 
                        alpha: float = 0.0, 
                        beta: float = 0.0,
                        exact: bool = False,
                        noise_scale: float = 0.01) -> torch.Tensor:
    """
    Initialize Jacobi coefficients to approximate the identity function.
    
    For Jacobi polynomials, the identity function f(x) = x requires a linear
    combination of basis functions. This initialization sets coefficients
    to create an identity-like mapping with optional noise.
    
    Args:
        tensor: Tensor to initialize (shape: input_dim, output_dim, degree+1)
        alpha: First Jacobi parameter (α > -1)
        beta: Second Jacobi parameter (β > -1)
        exact: Whether to use exact identity or add noise
        noise_scale: Scale of the noise to add if not exact
        
    Returns:
        Initialized tensor
    """
    # Set all coefficients to zero
    nn.init.zeros_(tensor)
    
    input_dim, output_dim, degree_plus_one = tensor.shape
    
    # Check if degree is at least 1 (we need P_1 for identity)
    if degree_plus_one < 2:
        raise ValueError("Degree must be at least 1 for identity initialization")
    
    # For identity function with Jacobi polynomials:
    # For P_1^(α,β)(x) = (α+β+2)x/2 + (α-β)/2
    # We need to set the coefficient of P_1 to handle the linear term
    # and possibly adjust P_0 to handle the constant term
    
    # Compute scaling factor for the linear term
    p1_scale = 2.0 / (alpha + beta + 2) if alpha + beta + 2 != 0 else 1.0
    
    # Compute offset for the constant term if needed
    p0_offset = -(alpha - beta) / 2 * p1_scale if alpha != beta else 0.0
    
    # Set coefficients for identity mapping
    min_dim = min(input_dim, output_dim)
    for i in range(min_dim):
        # Set coefficient for P_1 (linear term)
        tensor[i, i, 1] = p1_scale
        # Set coefficient for P_0 (constant term) if needed to offset
        if abs(p0_offset) > 1e-10:
            tensor[i, i, 0] = p0_offset
    
    # Add noise if not exact
    if not exact:
        noise = torch.randn_like(tensor) * noise_scale
        tensor += noise
    
    return tensor


# Update the initializer registry to include Jacobi initializers
def update_initializer_registry(registry: dict) -> dict:
    """
    Update the initializer registry to include Jacobi initializers.
    
    Args:
        registry: Existing initializer registry
        
    Returns:
        Updated registry
    """
    jacobi_initializers = {
        'jacobi_normal': init_jacobi_normal,
        'jacobi_uniform': init_jacobi_uniform,
        'jacobi_orthogonal': init_jacobi_orthogonal,
        'jacobi_zeros': init_jacobi_zeros,
        'jacobi_identity': init_jacobi_identity
    }
    
    registry.update(jacobi_initializers)
    return registry