import torch
import numpy as np
from typing import Tuple, Optional, List, Union
from .base import BasisFunction


class JacobiBasis(BasisFunction):
    """
    Jacobi polynomial basis functions for KAN.
    
    Jacobi polynomials P_n^(α,β)(x) are a class of orthogonal polynomials
    defined on the interval [-1, 1] with respect to the weight function
    (1-x)^α (1+x)^β where α, β > -1.
    
    They generalize many classical orthogonal polynomials:
    - Legendre polynomials: α = β = 0
    - Chebyshev polynomials (first kind): α = β = -1/2
    - Chebyshev polynomials (second kind): α = β = 1/2
    - Gegenbauer polynomials: α = β
    
    The Jacobi polynomials can be computed using the recurrence relation:
    P_0^(α,β)(x) = 1
    P_1^(α,β)(x) = (α + β + 2)x/2 + (α - β)/2
    For n ≥ 1:
    P_{n+1}^(α,β)(x) = ((2n + α + β + 1)((2n + α + β + 2)x + α^2 - β^2)P_n^(α,β)(x)
                        - 2(n + α)(n + β)(2n + α + β + 2)P_{n-1}^(α,β)(x))
                        / (2(n + 1)(n + α + β + 1)(2n + α + β))
    """
    
    def __init__(self, degree: int, alpha: float = 0.0, beta: float = 0.0, 
                 domain: Tuple[float, float] = (-1, 1)):
        """
        Initialize the Jacobi polynomial basis.
        
        Args:
            degree: Maximum degree of Jacobi polynomials
            alpha: First parameter (α > -1)
            beta: Second parameter (β > -1)
            domain: Natural domain for the basis functions
        """
        super().__init__(degree)
        
        # Validate parameters
        if alpha <= -1 or beta <= -1:
            raise ValueError("Parameters must satisfy α > -1 and β > -1")
        
        self.alpha = alpha
        self.beta = beta
        self._domain = domain
        
        # Special case detection for known polynomial families
        self.polynomial_type = self._detect_polynomial_type()
        
        # Pre-compute the arange for faster evaluation
        self.register_buffer("arange", torch.arange(0, degree + 1, dtype=torch.float32))
        
        # Pre-compute constants used in recurrence relation
        self._precompute_recurrence_constants()
    
    def _detect_polynomial_type(self) -> str:
        """
        Detect if these Jacobi polynomials correspond to a classical family.
        
        Returns:
            String identifying the polynomial type
        """
        if abs(self.alpha - 0.0) < 1e-10 and abs(self.beta - 0.0) < 1e-10:
            return "legendre"
        elif abs(self.alpha + 0.5) < 1e-10 and abs(self.beta + 0.5) < 1e-10:
            return "chebyshev1"
        elif abs(self.alpha - 0.5) < 1e-10 and abs(self.beta - 0.5) < 1e-10:
            return "chebyshev2"
        elif abs(self.alpha - self.beta) < 1e-10:
            return "gegenbauer"
        else:
            return "jacobi"
    
    def _precompute_recurrence_constants(self):
        """Precompute constants used in the recurrence relation."""
        # Constants for P_1 computation
        self.register_buffer("p1_const1", torch.tensor(0.5 * (self.alpha + self.beta + 2), dtype=torch.float32))
        self.register_buffer("p1_const2", torch.tensor(0.5 * (self.alpha - self.beta), dtype=torch.float32))
        
        # Precompute constants for recurrence relation if degree > 1
        if self.degree > 1:
            # For n >= 1: an, bn, cn are coefficients in the recurrence relation
            n_values = torch.arange(1, self.degree, dtype=torch.float32)
            
            # Compute coefficient terms
            two_n_ab = 2 * n_values + self.alpha + self.beta
            
            # an = (2n + α + β + 1)((2n + α + β + 2))/(2(n + 1)(n + α + β + 1)(2n + α + β))
            an_numerator = (two_n_ab + 1) * (two_n_ab + 2)
            an_denominator = 2 * (n_values + 1) * (n_values + self.alpha + self.beta + 1) * (two_n_ab)
            self.register_buffer("an", an_numerator / an_denominator)
            
            # bn = (2n + α + β + 1)(α^2 - β^2)/(2(n + 1)(n + α + β + 1)(2n + α + β))
            bn_numerator = (two_n_ab + 1) * (self.alpha**2 - self.beta**2)
            bn_denominator = 2 * (n_values + 1) * (n_values + self.alpha + self.beta + 1) * (two_n_ab)
            self.register_buffer("bn", bn_numerator / bn_denominator)
            
            # cn = -2(n + α)(n + β)(2n + α + β + 2)/(2(n + 1)(n + α + β + 1)(2n + α + β))
            cn_numerator = -2 * (n_values + self.alpha) * (n_values + self.beta) * (two_n_ab + 2)
            cn_denominator = 2 * (n_values + 1) * (n_values + self.alpha + self.beta + 1) * (two_n_ab)
            self.register_buffer("cn", cn_numerator / cn_denominator)
    
    @property
    def name(self) -> str:
        """Return the name of the basis function."""
        if self.polynomial_type != "jacobi":
            return f"{self.polynomial_type.capitalize()}-{self.degree}"
        else:
            return f"Jacobi-{self.degree}(α={self.alpha:.1f},β={self.beta:.1f})"
    
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
        
        # Apply coefficients using einsum for efficient batch multiplication
        return torch.einsum('bid,iod->bo', basis_values, coefficients)
    
    def basis_functions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the values of all Jacobi polynomials at points x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, input_dim, degree+1) containing 
            the values of Jacobi polynomials P_0 to P_degree at each point
        """
        # Ensure buffers are on the same device as input
        device = x.device
        
        # Optimized path for special cases
        if self.polynomial_type == "chebyshev1":
            return self._basis_chebyshev1(x)
        elif self.polynomial_type == "legendre":
            return self._basis_legendre(x)
        else:
            # Use general recurrence for Jacobi polynomials
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
        device = x.device
        
        # Ensure constants are on the same device as input
        p1_const1 = self.p1_const1.to(device)
        p1_const2 = self.p1_const2.to(device)
        
        # Initialize the result tensor
        result = torch.ones(batch_size, input_dim, self.degree + 1, 
                          device=device, dtype=x.dtype)
        
        if self.degree > 0:
            # P_1(x) = (α + β + 2)x/2 + (α - β)/2
            result[:, :, 1] = p1_const1 * x + p1_const2
            
            if self.degree > 1:
                # Ensure recurrence coefficients are on the same device
                an = self.an.to(device)
                bn = self.bn.to(device)
                cn = self.cn.to(device)
                
                # Recurrence relation for higher degrees
                for n in range(1, self.degree):
                    # P_{n+1} = a_n * x * P_n + b_n * P_n + c_n * P_{n-1}
                    result[:, :, n+1] = (an[n-1] * x * result[:, :, n] + 
                                       bn[n-1] * result[:, :, n] + 
                                       cn[n-1] * result[:, :, n-1])
        
        return result
    
    def _basis_chebyshev1(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Chebyshev polynomials of the first kind using the closed form.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, input_dim, degree+1)
        """
        # Ensure x has shape (batch_size, input_dim)
        batch_size, input_dim = x.shape
        device = x.device
        
        # Expand x to (batch_size, input_dim, 1) for broadcasting
        x = x.view(batch_size, input_dim, 1)
        
        # Compute arccos(x) and broadcast
        arccos_x = torch.acos(torch.clamp(x, -1.0, 1.0))
        
        # Ensure arange is on the same device as the input
        arange = self.arange.to(device=device)
        
        # Multiply by arange [0, 1, ..., degree]
        angles = arccos_x * arange
        
        # Compute cos(n * arccos(x)) for each n
        return torch.cos(angles)
    
    def _basis_legendre(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Legendre polynomials using the recurrence relation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, input_dim, degree+1)
        """
        # Ensure x has shape (batch_size, input_dim)
        batch_size, input_dim = x.shape
        device = x.device
        
        # Initialize the result tensor
        result = torch.ones(batch_size, input_dim, self.degree + 1, 
                          device=device, dtype=x.dtype)
        
        if self.degree > 0:
            # P_1(x) = x
            result[:, :, 1] = x
            
            # Use simplified recurrence for Legendre polynomials
            # P_{n+1}(x) = ((2n+1)x P_n(x) - n P_{n-1}(x)) / (n+1)
            for n in range(1, self.degree):
                result[:, :, n+1] = ((2*n + 1) * x * result[:, :, n] - n * result[:, :, n-1]) / (n + 1)
        
        return result
    
    def derivative(self, x: torch.Tensor, coefficients: torch.Tensor, 
                  order: int = 1) -> torch.Tensor:
        """
        Compute the derivative of the Jacobi polynomial expansion.
        
        The derivative of Jacobi polynomials follows the relation:
        d/dx[P_n^(α,β)(x)] = ((n + α + β + 1)/2) * P_{n-1}^(α+1,β+1)(x)
        
        Args:
            x: Input tensor
            coefficients: Coefficients tensor
            order: Order of the derivative
            
        Returns:
            Derivative tensor of shape (batch_size, output_dim)
        """
        if order == 0:
            return self.forward(x, coefficients)
        
        if self.degree < order:
            # All derivatives of order higher than the degree are zero
            batch_size = x.shape[0]
            output_dim = coefficients.shape[1]
            return torch.zeros(batch_size, output_dim, device=x.device, dtype=x.dtype)
        
        # For first order derivative, use the relation between Jacobi polynomials
        derivative_coeffs = self._compute_derivative_coefficients(coefficients)
        
        # Create a new basis with increased α and β and decreased degree
        derivative_basis = JacobiBasis(
            degree=self.degree - 1,
            alpha=self.alpha + 1,
            beta=self.beta + 1,
            domain=self.domain
        )
        
        # Ensure it's on the same device as x
        for attr_name in dir(derivative_basis):
            attr = getattr(derivative_basis, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(derivative_basis, attr_name, attr.to(device=x.device))
        
        if order == 1:
            # Compute the first derivative
            return derivative_basis.forward(x, derivative_coeffs)
        else:
            # For higher order derivatives, recursively apply the method
            return derivative_basis.derivative(x, derivative_coeffs, order - 1)
    
    def _compute_derivative_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Compute the coefficients for the derivative of Jacobi polynomial expansion.
        
        Args:
            coefficients: Coefficients tensor of shape (input_dim, output_dim, degree+1)
            
        Returns:
            Derivative coefficients of shape (input_dim, output_dim, degree)
        """
        input_dim, output_dim, _ = coefficients.shape
        device = coefficients.device
        
        # Initialize the derivative coefficients (one less degree)
        derivative_coeffs = torch.zeros(input_dim, output_dim, self.degree, 
                                     device=device, 
                                     dtype=coefficients.dtype)
        
        # Compute derivative coefficients using the relation:
        # d/dx[P_n^(α,β)(x)] = ((n + α + β + 1)/2) * P_{n-1}^(α+1,β+1)(x)
        for n in range(1, self.degree + 1):
            scale = (n + self.alpha + self.beta + 1) / 2
            derivative_coeffs[:, :, n-1] = scale * coefficients[:, :, n]
        
        return derivative_coeffs
    
    def weight_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the weight function (1-x)^α (1+x)^β for Jacobi polynomials.
        
        Args:
            x: Input tensor
            
        Returns:
            Weight function values
        """
        x = torch.clamp(x, -1.0 + 1e-10, 1.0 - 1e-10)  # Avoid numerical issues
        return (1 - x) ** self.alpha * (1 + x) ** self.beta
    
    def norm_squared(self, n: int) -> float:
        """
        Compute the squared norm of the n-th Jacobi polynomial.
        
        The squared norm is defined as:
        ∫_{-1}^{1} (P_n^(α,β)(x))^2 (1-x)^α (1+x)^β dx
        
        Args:
            n: Degree of the polynomial
            
        Returns:
            Squared norm value
        """
        if n == 0:
            return 2 ** (self.alpha + self.beta + 1) * torch.exp(
                torch.lgamma(torch.tensor(self.alpha + 1)) + 
                torch.lgamma(torch.tensor(self.beta + 1)) - 
                torch.lgamma(torch.tensor(self.alpha + self.beta + 2))
            ).item()
        else:
            return (2 ** (self.alpha + self.beta + 1) / (2 * n + self.alpha + self.beta + 1) * 
                   torch.exp(
                       torch.lgamma(torch.tensor(n + self.alpha + 1)) + 
                       torch.lgamma(torch.tensor(n + self.beta + 1)) - 
                       torch.lgamma(torch.tensor(n + 1)) - 
                       torch.lgamma(torch.tensor(n + self.alpha + self.beta + 1))
                   ).item())