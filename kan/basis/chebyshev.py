import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Union
from .base import BasisFunction


class ChebyshevBasis(BasisFunction):
    """
    Chebyshev polynomial basis functions for KAN.
    
    Chebyshev polynomials of the first kind are defined as:
    T_n(x) = cos(n * arccos(x)) for x ∈ [-1, 1]
    
    Alternatively, they can be computed using the recurrence relation:
    T_0(x) = 1
    T_1(x) = x
    T_{n+1}(x) = 2x * T_n(x) - T_{n-1}(x)
    
    Chebyshev polynomials have optimal approximation properties in the max norm,
    and they form an orthogonal basis with respect to the weight function 1/sqrt(1-x²).
    """
    
    def __init__(self, degree: int, domain: Tuple[float, float] = (-1, 1), 
                 use_recurrence: bool = False):
        """
        Initialize the Chebyshev basis.
        
        Args:
            degree: Maximum degree of Chebyshev polynomials
            domain: Natural domain for the basis functions
            use_recurrence: Whether to use recurrence relation instead of the closed form
        """
        super().__init__(degree)
        self._domain = domain
        self.use_recurrence = use_recurrence
        
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
        return f"Chebyshev-{self.degree}"
    
    @property
    def domain(self) -> Tuple[float, float]:
        """Return the natural domain of Chebyshev polynomials."""
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
        Evaluate the Chebyshev expansion at points x with given coefficients.
        
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
        
        # Compute the Chebyshev basis functions
        basis_values = self.basis_functions(x)
        
        # Apply coefficients using einsum
        return torch.einsum('bid,iod->bo', basis_values, coefficients)
    
    def basis_functions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the values of all Chebyshev polynomials at points x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, input_dim, degree+1) containing 
            the values of Chebyshev polynomials T_0 to T_degree at each point
        """
        if not self.use_recurrence:
            return self._basis_closed_form(x)
        else:
            return self._basis_recurrence(x)
    
    def _basis_closed_form(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Chebyshev polynomials using the closed form T_n(x) = cos(n * arccos(x)).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, input_dim, degree+1)
        """
        # Ensure x has shape (batch_size, input_dim)
        batch_size, input_dim = x.shape
        
        # Expand x to (batch_size, input_dim, 1) for broadcasting
        x = x.view(batch_size, input_dim, 1)
        
        # Compute arccos(x) and broadcast
        arccos_x = torch.acos(torch.clamp(x, -1.0, 1.0))
        
        # Ensure arange is on the same device as the input
        arange = self.arange.to(device=x.device)
        
        # Multiply by arange [0, 1, ..., degree]
        angles = arccos_x * arange
        
        # Compute cos(n * arccos(x)) for each n
        return torch.cos(angles)
    
    def _basis_recurrence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Chebyshev polynomials using the recurrence relation.
        
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
            # T_1(x) = x
            result[:, :, 1] = x
            
            # Recurrence relation: T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
            for n in range(2, self.degree + 1):
                result[:, :, n] = 2 * x * result[:, :, n-1] - result[:, :, n-2]
        
        return result
    
    def derivative(self, x: torch.Tensor, coefficients: torch.Tensor, 
                  order: int = 1) -> torch.Tensor:
        """
        Compute the derivative of the Chebyshev expansion.
        
        The derivative of Chebyshev polynomials follows specific rules:
        T_0'(x) = 0
        T_1'(x) = 1
        T_n'(x) = n * U_{n-1}(x)  where U_n is the Chebyshev polynomial of the second kind
        
        Since we're working with linear combinations, we can compute derivatives
        by adjusting the coefficients and then evaluating with lower degree polynomials.
        
        Args:
            x: Input tensor
            coefficients: Coefficients tensor
            order: Order of the derivative
            
        Returns:
            Derivative tensor of shape (batch_size, output_dim)
        """
        if order == 0:
            return self.forward(x, coefficients)
        
        if order > 1:
            # For higher order derivatives, we apply the method recursively
            # First compute the first derivative's coefficients
            first_deriv_coeffs = self._derivative_coefficients(coefficients)
            # Create a new basis with degree-1
            if self.degree > 0:
                derivative_basis = ChebyshevBasis(self.degree - 1, 
                                               self.domain, 
                                               self.use_recurrence)
                # Ensure it's on the same device as x
                derivative_basis.arange = derivative_basis.arange.to(device=x.device)
                # Compute the (order-1)th derivative of the first derivative
                return derivative_basis.derivative(x, first_deriv_coeffs, order - 1)
            else:
                # If degree is 0, all higher derivatives are 0
                return torch.zeros_like(x[:, :coefficients.shape[1]], device=x.device)
        
        # First order derivative
        derivative_coeffs = self._derivative_coefficients(coefficients)
        
        if self.degree > 0:
            # Create a basis of one lower degree for the derivative
            derivative_basis = ChebyshevBasis(self.degree - 1, 
                                           self.domain, 
                                           self.use_recurrence)
            # Ensure it's on the same device as x
            derivative_basis.arange = derivative_basis.arange.to(device=x.device)
            return derivative_basis.forward(x, derivative_coeffs)
        else:
            # If degree is 0, derivative is 0
            return torch.zeros_like(x[:, :coefficients.shape[1]], device=x.device)
    
    def _derivative_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Compute the coefficients for the derivative of the Chebyshev expansion.
        
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
        
        # Compute derivative coefficients using the formula:
        # For n > 0, the coefficient of T_{n-1} in the derivative is 2n times
        # the coefficient of T_n in the original expansion
        # (except for the highest degree term which has a special case)
        
        # For n=1 to degree
        for n in range(1, self.degree + 1):
            derivative_coeffs[:, :, n-1] = 2 * n * coefficients[:, :, n]
        
        # Adjustments for the recurrence relation of derivatives
        for n in range(3, self.degree + 1, 2):
            for k in range(1, (n+1)//2):
                derivative_coeffs[:, :, n-2*k-1] -= n * coefficients[:, :, n]
        
        return derivative_coeffs