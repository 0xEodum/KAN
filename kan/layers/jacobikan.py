import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union

from ..basis.jacobi import JacobiBasis
from .base import KANLayer


class JacobiKANLayer(KANLayer):
    """
    KAN layer using Jacobi polynomial basis functions.
    
    This layer is based on the representation of each univariate component
    function as a linear combination of Jacobi polynomials.
    
    Jacobi polynomials P_n^(α,β)(x) generalize many classical orthogonal polynomials
    and provide flexibility through the α and β parameters that affect the
    weight distribution over the interval [-1, 1].
    
    Special cases:
    - α = β = 0: Legendre polynomials (uniform weight)
    - α = β = -1/2: Chebyshev polynomials of the first kind (good for approximation in max norm)
    - α = β = 1/2: Chebyshev polynomials of the second kind
    """
    
    def __init__(self, input_dim: int, output_dim: int, degree: int,
                 alpha: float = 0.0, beta: float = 0.0, 
                 init_scale: float = None):
        """
        Initialize the JacobiKAN layer.
        
        Args:
            input_dim: Number of input dimensions
            output_dim: Number of output dimensions
            degree: Maximum degree of Jacobi polynomials
            alpha: First Jacobi parameter (α > -1)
            beta: Second Jacobi parameter (β > -1)
            init_scale: Scale for coefficient initialization (if None, uses 1/(input_dim * (degree+1)))
        """
        # Validate parameters
        if alpha <= -1 or beta <= -1:
            raise ValueError("Parameters must satisfy α > -1 and β > -1")
            
        # Create the Jacobi basis
        basis_function = JacobiBasis(degree=degree, alpha=alpha, beta=beta)
        
        # Initialize the base class
        super(JacobiKANLayer, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            basis_function=basis_function
        )
        
        # Save parameters for later use
        self.degree = degree
        self.alpha = alpha
        self.beta = beta
        
        # Initialize the coefficients
        init_scale = init_scale or 1.0 / (input_dim * (degree + 1))
        self.jacobi_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        self._initialize_coefficients(init_scale)
    
    def _initialize_coefficients(self, scale: float):
        """
        Initialize the coefficients based on the properties of Jacobi polynomials.
        
        Args:
            scale: Scaling factor for initialization
        """
        # Basic initialization with scaled normal distribution
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=scale)
        
        # Apply degree-based scaling to favor lower degrees
        # Lower degree terms tend to be more important in smooth functions
        degree_factors = torch.linspace(1.0, 0.1, self.degree + 1)
        self.jacobi_coeffs.data *= degree_factors.reshape(1, 1, -1)
        
        # Specialized initialization for specific polynomial types
        polynomial_type = ""
        if abs(self.alpha - 0.0) < 1e-10 and abs(self.beta - 0.0) < 1e-10:
            polynomial_type = "legendre"
        elif abs(self.alpha + 0.5) < 1e-10 and abs(self.beta + 0.5) < 1e-10:
            polynomial_type = "chebyshev1"
        elif abs(self.alpha - 0.5) < 1e-10 and abs(self.beta - 0.5) < 1e-10:
            polynomial_type = "chebyshev2"
            
        # For identity-like initialization for appropriate dimensions
        if polynomial_type in ["legendre", "chebyshev1", "chebyshev2"]:
            # Initialize near-identity mapping for matching dimensions
            min_dim = min(self.input_dim, self.output_dim)
            for i in range(min_dim):
                # For Chebyshev, the linear term is at index 1
                # For Legendre, the linear term is also at index 1
                # Use .data для безопасного in-place операций с параметрами требующими градиентов
                self.jacobi_coeffs.data[i, i, 1] = 1.0 + torch.randn(1).item() * 0.01
    
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
        return self.basis_function.forward(x, self.jacobi_coeffs)
    
    def get_analytical_form(self) -> Dict[str, Any]:
        """
        Return the analytical form of the layer function.
        
        Returns:
            Dictionary containing information about the analytical form
        """
        return {
            'type': 'JacobiKANLayer',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'degree': self.degree,
            'alpha': self.alpha,
            'beta': self.beta,
            'basis': self.basis_function.name,
            'coefficients': self.jacobi_coeffs.detach().cpu().numpy()
        }
    
    def get_coefficients(self) -> torch.Tensor:
        """
        Get the coefficients used by this layer.
        
        Returns:
            Tensor of shape (input_dim, output_dim, degree+1) containing the coefficients
        """
        return self.jacobi_coeffs
    
    def extra_repr(self) -> str:
        """
        Return a string representation of the layer.
        
        Returns:
            String representation
        """
        return (f'input_dim={self.input_dim}, output_dim={self.output_dim}, '
                f'degree={self.degree}, alpha={self.alpha:.1f}, beta={self.beta:.1f}')