import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Union
from .base import BasisFunction


class JacobiBasis(BasisFunction):
    """
    Jacobi polynomial basis functions for KAN.
    
    Jacobi polynomials P_n^(α,β)(x) are a family of orthogonal polynomials
    defined on the interval [-1, 1] with respect to the weight function
    (1-x)^α * (1+x)^β for α, β > -1.
    
    They can be computed using the recurrence relation:
    P_0^(α,β)(x) = 1
    P_1^(α,β)(x) = ((α + β + 2)x + (α - β))/2
    
    For n ≥ 2:
    P_n^(α,β)(x) = ((2n + α + β - 1)((2n + α + β)(2n + α + β - 2)x + α² - β²)P_{n-1}^(α,β)(x) - 
                   2(n + α - 1)(n + β - 1)(2n + α + β)P_{n-2}^(α,β)(x)) / 
                   (2n(n + α + β)(2n + α + β - 2))
    
    Jacobi polynomials generalize several well-known polynomial families:
    - α = β = 0: Legendre polynomials
    - α = β = -1/2: Chebyshev polynomials of the first kind
    - α = β = 1/2: Chebyshev polynomials of the second kind
    - α = 0, β = -1/2: Legendre polynomials of the second kind
    """
    
    def __init__(self, degree: int, alpha: float = 0.0, beta: float = 0.0,
                 domain: Tuple[float, float] = (-1, 1)):
        """
        Initialize the Jacobi basis.
        
        Args:
            degree: Maximum degree of Jacobi polynomials
            alpha: First parameter for Jacobi polynomials (α > -1)
            beta: Second parameter for Jacobi polynomials (β > -1)
            domain: Natural domain for the basis functions
        """
        super().__init__(degree)
        
        # Validate parameters
        if alpha <= -1 or beta <= -1:
            raise ValueError(f"Parameters must satisfy α, β > -1, got α={alpha}, β={beta}")
        
        self.alpha = alpha
        self.beta = beta
        self._domain = domain
        
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
        return f"Jacobi-{self.degree}-{self.alpha}-{self.beta}"
    
    @property
    def domain(self) -> Tuple[float, float]:
        """Return the natural domain of Jacobi polynomials."""
        return self._domain
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input to the domain [-1, 1] using tanh.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized input tensor
        """
        return torch.tanh(x)
    
    def forward(self, x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Jacobi expansion at points x with given coefficients.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            coefficients: Coefficients tensor of shape (input_dim, output_dim, degree+1)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Ensure x has the right shape (batch_size, input_dim)
        x = x.view(-1, coefficients.shape[0])
        
        # Normalize input to [-1, 1]
        x = self.normalize_input(x)
        
        # Compute the Jacobi basis functions
        basis_values = self.basis_functions(x)
        
        # Apply coefficients using einsum
        return torch.einsum('bid,iod->bo', basis_values, coefficients)
    
    def basis_functions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the values of all Jacobi polynomials at points x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, input_dim, degree+1) containing 
            the values of Jacobi polynomials P_0^(α,β) to P_degree^(α,β) at each point
        """
        return self._basis_recurrence(x)
    
    def _basis_recurrence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobi polynomials using the recurrence relation.
        
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
        
        alpha, beta = self.alpha, self.beta
        
        if self.degree > 0:
            # P_1^(α,β)(x) = ((α + β + 2)x + (α - β))/2
            # Avoid inplace operation - create a new tensor
            p1 = ((alpha + beta + 2) * x + (alpha - beta)) / 2
            result[:, :, 1] = p1
            
            # Recurrence relation for n ≥ 2
            for n in range(2, self.degree + 1):
                # Get previous polynomials P_{n-1} and P_{n-2}
                p_n_minus_1 = result[:, :, n-1].clone()  # Clone to prevent modification
                p_n_minus_2 = result[:, :, n-2].clone()  # Clone to prevent modification
                
                # Temporary variables to make the formula more readable
                n_float = float(n)
                ab_sum = alpha + beta
                ab_diff = alpha**2 - beta**2
                
                # Common term in numerator: (2n + α + β - 1)
                common_term = 2 * n_float + ab_sum - 1
                
                # Term for multiplication with P_{n-1}
                term1 = (2 * n_float + ab_sum) * (2 * n_float + ab_sum - 2) * x + ab_diff
                
                # Coefficient for P_{n-1}
                coef1 = common_term * term1
                
                # Coefficient for P_{n-2}
                coef2 = -2 * (n_float + alpha - 1) * (n_float + beta - 1) * (2 * n_float + ab_sum)
                
                # Denominator
                denom = 2 * n_float * (n_float + ab_sum) * (2 * n_float + ab_sum - 2)
                
                # Compute P_n without inplace operations
                p_n = (coef1 * p_n_minus_1 + coef2 * p_n_minus_2) / denom
                
                # Store the result
                result[:, :, n] = p_n
        
        return result
    
    def derivative(self, x: torch.Tensor, coefficients: torch.Tensor, 
                   order: int = 1) -> torch.Tensor:
        """
        Compute the derivative of the Jacobi expansion.
        
        The derivative of a Jacobi polynomial P_n^(α,β)(x) is related to another Jacobi polynomial:
        d/dx P_n^(α,β)(x) = (n + α + β + 1) * P_{n-1}^(α+1,β+1)(x) / 2
        
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
            # Note: derivative of P_n^(α,β)(x) is related to P_{n-1}^(α+1,β+1)(x)
            derivative_basis = JacobiBasis(self.degree - 1, 
                                          alpha=self.alpha + 1, 
                                          beta=self.beta + 1,
                                          domain=self.domain)
            
            # Ensure it's on the same device as x
            if hasattr(self, 'arange'):
                derivative_basis.arange = self.arange.to(device=x.device)
            
            if order > 1:
                # For higher order derivatives, apply recursively
                return derivative_basis.derivative(x, derivative_coeffs, order - 1)
            else:
                # First order derivative
                return derivative_basis.forward(x, derivative_coeffs)
        else:
            # If degree is 0, derivative is 0
            return torch.zeros(x.shape[0], coefficients.shape[1], device=x.device, dtype=x.dtype)
    
    def _derivative_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Compute the coefficients for the derivative of the Jacobi expansion.
        
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
        
        # For each degree n ≥ 1, the derivative of P_n^(α,β)(x) is related to P_{n-1}^(α+1,β+1)(x)
        for n in range(1, self.degree + 1):
            # Scale factor for the derivative
            scale = (n + self.alpha + self.beta + 1) / 2
            derivative_coeffs[:, :, n-1] = scale * coefficients[:, :, n]
        
        return derivative_coeffs