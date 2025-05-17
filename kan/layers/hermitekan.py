import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union

from ..basis.hermite import HermiteBasis
from .base import KANLayer


class HermiteKANLayer(KANLayer):
    """
    KAN layer using Hermite polynomial basis functions.
    
    This layer is based on the representation of each univariate component
    function as a linear combination of Hermite polynomials.
    
    Hermite polynomials are particularly well-suited for approximating 
    functions on the entire real line with normal/Gaussian distribution 
    characteristics, making them valuable for many machine learning applications.
    """
    
    def __init__(self, input_dim: int, output_dim: int, degree: int,
                 scaling: str = 'physicist', domain: Optional[Tuple[float, float]] = None,
                 normalize_domain: bool = True, init_scale: float = None):
        """
        Initialize the HermiteKAN layer.
        
        Args:
            input_dim: Number of input dimensions
            output_dim: Number of output dimensions
            degree: Maximum degree of Hermite polynomials
            scaling: Type of Hermite polynomials:
                     'physicist' (default) - uses H_n with H_{n+1} = 2x·H_n - 2n·H_{n-1}
                     'probabilist' - uses He_n with He_{n+1} = x·He_n - n·He_{n-1}
            domain: Optional domain specification.
                    If None, uses the natural domain of Hermite polynomials (-∞, +∞)
            normalize_domain: Whether to normalize input to a standard domain
            init_scale: Scale for coefficient initialization (if None, uses 1/(input_dim * (degree+1)))
        """
        # Create the Hermite basis
        basis_function = HermiteBasis(
            degree=degree,
            scaling=scaling,
            domain=domain,
            normalize_domain=normalize_domain
        )
        
        # Initialize the base class
        super(HermiteKANLayer, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            basis_function=basis_function
        )
        
        # Save parameters for later use
        self.degree = degree
        self.scaling = scaling
        self.domain = domain
        self.normalize_domain = normalize_domain
        
        # Initialize the coefficients
        init_scale = init_scale or 1.0 / (input_dim * (degree + 1))
        self.hermite_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        nn.init.normal_(self.hermite_coeffs, mean=0.0, std=init_scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Ensure x has the right shape
        x = x.view(-1, self.input_dim)
        
        # Delegate evaluation to the basis function
        return self.basis_function.forward(x, self.hermite_coeffs)
    
    def get_analytical_form(self) -> Dict[str, Any]:
        """
        Return the analytical form of the layer function.
        
        Returns:
            Dictionary containing information about the analytical form
        """
        return {
            'type': 'HermiteKANLayer',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'degree': self.degree,
            'scaling': self.scaling,
            'domain': self.domain,
            'normalize_domain': self.normalize_domain,
            'basis': self.basis_function.name,
            'coefficients': self.hermite_coeffs.detach().cpu().numpy()
        }
    
    def get_coefficients(self) -> torch.Tensor:
        """
        Get the coefficients used by this layer.
        
        Returns:
            Tensor of shape (input_dim, output_dim, degree+1) containing the coefficients
        """
        return self.hermite_coeffs
    
    def extra_repr(self) -> str:
        """
        Return a string representation of the layer.
        
        Returns:
            String representation
        """
        domain_str = f", domain={self.domain}" if self.domain else ""
        normalize_str = f", normalize_domain={self.normalize_domain}"
        return (f'input_dim={self.input_dim}, output_dim={self.output_dim}, '
                f'degree={self.degree}, scaling={self.scaling}{domain_str}{normalize_str}')