import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union

from ..basis.chebyshev import ChebyshevBasis
from .base import KANLayer


class ChebyKANLayer(KANLayer):
    """
    KAN layer using Chebyshev polynomial basis functions.
    
    This layer is based on the representation of each univariate component
    function as a linear combination of Chebyshev polynomials.
    
    Chebyshev polynomials provide optimal approximation properties in the max norm,
    making them suitable for function approximation tasks.
    """
    
    def __init__(self, input_dim: int, output_dim: int, degree: int,
                 use_recurrence: bool = False, init_scale: float = None):
        """
        Initialize the ChebyKAN layer.
        
        Args:
            input_dim: Number of input dimensions
            output_dim: Number of output dimensions
            degree: Maximum degree of Chebyshev polynomials
            use_recurrence: Whether to use recurrence relation instead of the closed form
            init_scale: Scale for coefficient initialization (if None, uses 1/(input_dim * (degree+1)))
        """
        # Create the Chebyshev basis
        basis_function = ChebyshevBasis(degree=degree, use_recurrence=use_recurrence)
        
        # Initialize the base class
        super(ChebyKANLayer, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            basis_function=basis_function
        )
        
        # Save the degree for later use
        self.degree = degree
        
        # Initialize the coefficients
        init_scale = init_scale or 1.0 / (input_dim * (degree + 1))
        self.cheby_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=init_scale)
    
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
        return self.basis_function.forward(x, self.cheby_coeffs)
    
    def get_analytical_form(self) -> Dict[str, Any]:
        """
        Return the analytical form of the layer function.
        
        Returns:
            Dictionary containing information about the analytical form
        """
        return {
            'type': 'ChebyKANLayer',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'degree': self.degree,
            'basis': self.basis_function.name,
            'coefficients': self.cheby_coeffs.detach().cpu().numpy()
        }
    
    def get_coefficients(self) -> torch.Tensor:
        """
        Get the coefficients used by this layer.
        
        Returns:
            Tensor of shape (input_dim, output_dim, degree+1) containing the coefficients
        """
        return self.cheby_coeffs
    
    def extra_repr(self) -> str:
        """
        Return a string representation of the layer.
        
        Returns:
            String representation
        """
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}, degree={self.degree}'