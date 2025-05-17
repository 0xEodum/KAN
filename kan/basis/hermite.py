import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Union
from .base import BasisFunction


class HermiteBasis(BasisFunction):
    """
    Hermite polynomial basis functions for KAN.
    
    Hermite polynomials H_n(x) are a family of orthogonal polynomials
    defined on the entire real line (-∞, +∞) with respect to the weight 
    function e^(-x²).
    
    They can be computed using the recurrence relation:
    H_0(x) = 1
    H_1(x) = 2x
    H_{n+1}(x) = 2x·H_n(x) - 2n·H_{n-1}(x)
    
    Hermite polynomials have several important properties:
    - They are orthogonal with respect to the Gaussian weight function
    - They appear in the solutions of the quantum harmonic oscillator
    - They are related to the derivatives of the Gaussian function: 
      H_n(x) = (-1)^n·e^(x²)·(d^n/dx^n)(e^(-x²))
    - Their derivatives satisfy: H_n'(x) = 2n·H_{n-1}(x)
    """
    
    def __init__(self, degree: int, scaling: str = 'physicist', 
                 domain: Optional[Tuple[float, float]] = None,
                 normalize_domain: bool = True):
        """
        Initialize the Hermite basis.
        
        Args:
            degree: Maximum degree of Hermite polynomials
            scaling: Type of Hermite polynomials to use:
                     'physicist' (default) - uses the recurrence relation H_{n+1} = 2x·H_n - 2n·H_{n-1}
                     'probabilist' - uses He_n where He_{n+1} = x·He_n - n·He_{n-1}
            domain: Optional domain specification.
                    If None, uses the natural domain of Hermite polynomials (-∞, +∞)
            normalize_domain: Whether to normalize input to a standard domain 
                            (only relevant if domain is specified)
        """
        super().__init__(degree)
        
        if scaling not in ['physicist', 'probabilist']:
            raise ValueError(f"Invalid scaling: {scaling}. "
                           f"Must be one of ['physicist', 'probabilist']")
        
        self.scaling = scaling
        self._normalize_domain = normalize_domain
        self._domain = domain or (-float('inf'), float('inf'))
        
        # Pre-compute the arange for faster evaluation
        self.register_buffer("arange", torch.arange(0, degree + 1, dtype=torch.float32))
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """
        Register a buffer for the basis function.
        This is a utility method to mimic nn.Module's register_buffer
        without inheriting from nn.Module.
        
        Args:
            name: Name of the buffer
            tensor: Tensor to register
        """
        if not hasattr(self, name):
            setattr(self, name, tensor)
        else:
            # Update the existing buffer
            existing_buffer = getattr(self, name)
            existing_buffer.data = tensor.data
    
    @property
    def name(self) -> str:
        """Return the name of the basis function."""
        return f"Hermite-{self.degree}-{self.scaling}"
    
    @property
    def domain(self) -> Tuple[float, float]:
        """Return the domain of Hermite polynomials."""
        return self._domain
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input for Hermite polynomials.
        
        For Hermite polynomials, the natural domain is the entire real line,
        but it's often useful to scale inputs to have appropriate magnitude
        since very large values can lead to numerical overflow. Here we use
        a simple normalization approach that preserves the sign but scales
        magnitude.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized input tensor
        """
        # If a custom domain is specified and normalization is enabled
        if self._normalize_domain and self._domain != (-float('inf'), float('inf')):
            # Map from the specified domain to a standard domain for Hermite polynomials
            # We use [-3, 3] as a practical approximation of the unbounded domain
            # since Hermite polynomials behave well in this range
            domain_min, domain_max = self._domain
            
            # Check if domain bounds are finite
            if np.isfinite(domain_min) and np.isfinite(domain_max):
                # Linear mapping from specified domain to [-3, 3]
                scale = 6.0 / (domain_max - domain_min)
                return scale * (x - domain_min) - 3.0
            
        # For unbounded domain or when normalization is disabled, 
        # we use a softer approach to avoid extreme values
        return torch.tanh(x) * 3.0
    
    def forward(self, x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Hermite expansion at points x with given coefficients.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            coefficients: Coefficients tensor of shape (input_dim, output_dim, degree+1)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Ensure x has the right shape (batch_size, input_dim)
        x = x.view(-1, coefficients.shape[0])
        
        # Normalize input
        x = self.normalize_input(x)
        
        # Compute the Hermite basis functions
        basis_values = self.basis_functions(x)
        
        # Apply coefficients using einsum
        return torch.einsum('bid,iod->bo', basis_values, coefficients)
    
    def basis_functions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the values of all Hermite polynomials at points x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, input_dim, degree+1) containing 
            the values of Hermite polynomials H_0 to H_degree (or He_0 to He_degree) at each point
        """
        # Use appropriate recurrence relation based on the scaling
        if self.scaling == 'physicist':
            return self._basis_physicist(x)
        else:  # self.scaling == 'probabilist'
            return self._basis_probabilist(x)
    
    def _basis_physicist(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute physicist's Hermite polynomials using the recurrence relation:
        H_0(x) = 1
        H_1(x) = 2x
        H_{n+1}(x) = 2x·H_n(x) - 2n·H_{n-1}(x)
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, input_dim, degree+1)
        """
        # Ensure x has shape (batch_size, input_dim)
        batch_size, input_dim = x.shape
        
        # Initialize the result tensor
        result = torch.ones(batch_size, input_dim, self.degree + 1, 
                          device=x.device, dtype=x.dtype)
        
        if self.degree > 0:
            # H_1(x) = 2x
            result[:, :, 1] = 2 * x
            
            # Recurrence relation for n ≥ 2
            for n in range(2, self.degree + 1):
                # H_{n}(x) = 2x·H_{n-1}(x) - 2(n-1)·H_{n-2}(x)
                result[:, :, n] = 2 * x * result[:, :, n-1] - 2 * (n-1) * result[:, :, n-2]
        
        return result
    
    def _basis_probabilist(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilist's Hermite polynomials using the recurrence relation:
        He_0(x) = 1
        He_1(x) = x
        He_{n+1}(x) = x·He_n(x) - n·He_{n-1}(x)
        
        These are related to the physicist's Hermite polynomials by:
        He_n(x) = 2^(-n/2)·H_n(x/√2)
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, input_dim, degree+1)
        """
        # Ensure x has shape (batch_size, input_dim)
        batch_size, input_dim = x.shape
        
        # Initialize the result tensor
        result = torch.ones(batch_size, input_dim, self.degree + 1, 
                          device=x.device, dtype=x.dtype)
        
        if self.degree > 0:
            # He_1(x) = x
            result[:, :, 1] = x
            
            # Recurrence relation for n ≥ 2
            for n in range(2, self.degree + 1):
                # He_{n}(x) = x·He_{n-1}(x) - (n-1)·He_{n-2}(x)
                result[:, :, n] = x * result[:, :, n-1] - (n-1) * result[:, :, n-2]
        
        return result
    
    def derivative(self, x: torch.Tensor, coefficients: torch.Tensor, 
                  order: int = 1) -> torch.Tensor:
        """
        Compute the derivative of the Hermite expansion.
        
        The derivative of Hermite polynomials follows specific rules:
        For physicist's scaling:
        H'_n(x) = 2n·H_{n-1}(x)
        
        For probabilist's scaling:
        He'_n(x) = n·He_{n-1}(x)
        
        Args:
            x: Input tensor
            coefficients: Coefficients tensor
            order: Order of the derivative
            
        Returns:
            Derivative tensor of shape (batch_size, output_dim)
        """
        if order == 0:
            return self.forward(x, coefficients)
        
        # For first order derivative
        derivative_coeffs = self._derivative_coefficients(coefficients)
        
        if self.degree > 0:
            # Create a basis of one lower degree for the derivative
            derivative_basis = HermiteBasis(self.degree - 1, 
                                          scaling=self.scaling,
                                          domain=self.domain,
                                          normalize_domain=self._normalize_domain)
            
            # Ensure it's on the same device as x
            derivative_basis.arange = derivative_basis.arange.to(device=x.device)
            
            if order > 1:
                # For higher order derivatives, apply recursively
                return derivative_basis.derivative(x, derivative_coeffs, order - 1)
            else:
                # First order derivative
                return derivative_basis.forward(x, derivative_coeffs)
        else:
            # If degree is 0, derivative is 0
            return torch.zeros_like(x[:, :coefficients.shape[1]], device=x.device)
    
    def _derivative_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Compute the coefficients for the derivative of the Hermite expansion.
        
        Args:
            coefficients: Coefficients tensor of shape (input_dim, output_dim, degree+1)
            
        Returns:
            Derivative coefficients of shape (input_dim, output_dim, degree)
        """
        input_dim, output_dim, _ = coefficients.shape
        device = coefficients.device
        
        if self.degree == 0:
            # Derivative of constant is 0
            return torch.zeros(input_dim, output_dim, 0, 
                             device=device, dtype=coefficients.dtype)
        
        # Initialize the derivative coefficients (one less degree)
        derivative_coeffs = torch.zeros(input_dim, output_dim, self.degree, 
                                     device=device, 
                                     dtype=coefficients.dtype)
        
        # For each degree n ≥ 1, apply the derivative rule
        if self.scaling == 'physicist':
            # H'_n(x) = 2n·H_{n-1}(x)
            for n in range(1, self.degree + 1):
                derivative_coeffs[:, :, n-1] = 2 * n * coefficients[:, :, n]
        else:  # self.scaling == 'probabilist'
            # He'_n(x) = n·He_{n-1}(x)
            for n in range(1, self.degree + 1):
                derivative_coeffs[:, :, n-1] = n * coefficients[:, :, n]
        
        return derivative_coeffs