import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union

from ..basis.relu import ReLUBasis
from .base import KANLayer


class ReLUKANLayer(KANLayer):
    """
    KAN layer using parameterized ReLU-like basis functions.
    
    This layer represents each univariate component function as a linear 
    combination of parameterized ReLU-like activation functions. These
    functions can adapt to approximate a wide variety of shapes, from
    standard ReLU to Leaky ReLU, PReLU, and smooth variants like SiLU/Swish.
    
    The flexibility of parameterized ReLU functions allows the layer to learn
    activation patterns that are optimal for a given task, which can be particularly
    beneficial for non-smooth function approximation tasks.
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_basis: int,
                 domain: Tuple[float, float] = (-10, 10),
                 init_alphas: Optional[torch.Tensor] = None,
                 init_betas: Optional[torch.Tensor] = None,
                 init_gammas: Optional[torch.Tensor] = None,
                 init_deltas: Optional[torch.Tensor] = None,
                 init_centers: Optional[torch.Tensor] = None,
                 init_scales: Optional[torch.Tensor] = None,
                 learn_basis_parameters: bool = True,
                 init_scale: float = None):
        """
        Initialize the ReLUKAN layer.
        
        Args:
            input_dim: Number of input dimensions
            output_dim: Number of output dimensions
            num_basis: Number of ReLU basis functions to use
            domain: Domain for the basis functions
            init_alphas: Initial values for alpha parameters (negative slope)
            init_betas: Initial values for beta parameters (sigmoid component weight)
            init_gammas: Initial values for gamma parameters (positive slope)
            init_deltas: Initial values for delta parameters (sigmoid steepness)
            init_centers: Initial values for centers of shifted basis functions
            init_scales: Initial values for scales of shifted basis functions
            learn_basis_parameters: Whether to make the basis parameters learnable
            init_scale: Scale for coefficient initialization (if None, uses 1/(input_dim * num_basis))
        """
        # Create the ReLU basis
        basis_function = ReLUBasis(
            num_basis=num_basis,
            domain=domain,
            init_alphas=init_alphas,
            init_betas=init_betas,
            init_gammas=init_gammas,
            init_deltas=init_deltas,
            init_centers=init_centers,
            init_scales=init_scales,
            learn_parameters=learn_basis_parameters
        )
        
        # Initialize the base class
        super(ReLUKANLayer, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            basis_function=basis_function
        )
        
        # Save parameters for later use
        self.num_basis = num_basis
        self.domain = domain
        self.learn_basis_parameters = learn_basis_parameters
        
        # Initialize the coefficients
        init_scale = init_scale or 1.0 / (input_dim * num_basis)
        self.relu_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, num_basis)
        )
        nn.init.normal_(self.relu_coeffs, mean=0.0, std=init_scale)
    
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
        return self.basis_function.forward(x, self.relu_coeffs)
    
    def get_analytical_form(self) -> Dict[str, Any]:
        """
        Return the analytical form of the layer function.
        
        Returns:
            Dictionary containing information about the analytical form
        """
        # Extract basis parameters
        basis = self.basis_function
        
        return {
            'type': 'ReLUKANLayer',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_basis': self.num_basis,
            'domain': self.domain,
            'basis': basis.name,
            'alphas': basis.alphas.detach().cpu().numpy(),
            'betas': basis.betas.detach().cpu().numpy(),
            'gammas': basis.gammas.detach().cpu().numpy(),
            'deltas': basis.deltas.detach().cpu().numpy(),
            'centers': basis.centers.detach().cpu().numpy(),
            'scales': basis.scales.detach().cpu().numpy(),
            'coefficients': self.relu_coeffs.detach().cpu().numpy()
        }
    
    def get_coefficients(self) -> torch.Tensor:
        """
        Get the coefficients used by this layer.
        
        Returns:
            Tensor of shape (input_dim, output_dim, num_basis) containing the coefficients
        """
        return self.relu_coeffs
    
    def extra_repr(self) -> str:
        """
        Return a string representation of the layer.
        
        Returns:
            String representation
        """
        return (f'input_dim={self.input_dim}, output_dim={self.output_dim}, '
                f'num_basis={self.num_basis}, learn_basis_parameters={self.learn_basis_parameters}')