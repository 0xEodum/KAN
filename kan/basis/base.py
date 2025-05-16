from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Tuple, Optional, List, Union, Callable


class BasisFunction(ABC):
    """
    Abstract base class for all basis functions used in KAN.
    
    This class defines the interface that all basis functions must implement.
    Based on the Kolmogorov-Arnold representation theorem, different basis
    functions can be used to approximate continuous multivariate functions.
    """
    
    @abstractmethod
    def __init__(self, degree: int, **kwargs):
        """
        Initialize the basis function.
        
        Args:
            degree: Maximum degree of the basis functions
            **kwargs: Additional parameters specific to the basis function
        """
        self.degree = degree
    
    @abstractmethod
    def forward(self, x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the basis function expansion at points x with given coefficients.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, input_dim, 1)
            coefficients: Coefficients tensor of shape (input_dim, output_dim, degree+1)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        pass
    
    @abstractmethod
    def basis_functions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the values of all basis functions at points x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, input_dim, 1)
            
        Returns:
            Tensor of shape (batch_size, input_dim, degree+1) containing 
            the values of all basis functions at each point
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the basis function."""
        pass
    
    @property
    @abstractmethod
    def domain(self) -> Tuple[float, float]:
        """Return the natural domain of the basis function."""
        pass
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input to the natural domain of the basis function.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized input tensor
        """
        # Default implementation uses tanh to map to (-1, 1)
        return torch.tanh(x)
    
    @abstractmethod
    def derivative(self, x: torch.Tensor, coefficients: torch.Tensor, 
                   order: int = 1) -> torch.Tensor:
        """
        Compute the derivative of the basis function expansion.
        
        Args:
            x: Input tensor
            coefficients: Coefficients tensor
            order: Order of the derivative
            
        Returns:
            Derivative tensor of shape (batch_size, output_dim)
        """
        pass
    
    def numerical_derivative(self, x: torch.Tensor, coefficients: torch.Tensor, 
                            order: int = 1, eps: float = 1e-6) -> torch.Tensor:
        """
        Compute numerical derivative using finite difference method.
        This is a fallback implementation for basis functions without 
        analytical derivatives.
        
        Args:
            x: Input tensor
            coefficients: Coefficients tensor
            order: Order of the derivative
            eps: Step size for finite difference
            
        Returns:
            Numerical derivative tensor
        """
        # Implementation of numerical derivative using finite difference
        with torch.no_grad():
            if order == 1:
                x_plus = x + eps
                x_minus = x - eps
                f_plus = self.forward(x_plus, coefficients)
                f_minus = self.forward(x_minus, coefficients)
                return (f_plus - f_minus) / (2 * eps)
            elif order > 1:
                # Recursive application for higher order derivatives
                first_deriv = self.numerical_derivative(x, coefficients, 1, eps)
                return self.numerical_derivative(x, first_deriv, order - 1, eps)
            else:
                return self.forward(x, coefficients)
    
    def to_numpy(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Convert torch tensor to numpy array if needed.
        
        Args:
            x: Input tensor or array
            
        Returns:
            Numpy array
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    def visualize(self, coefficients: torch.Tensor, input_idx: int = 0, 
                 output_idx: int = 0, num_points: int = 200,
                 domain: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize the basis function expansion for a specific input-output pair.
        
        Args:
            coefficients: Coefficients tensor
            input_idx: Input dimension index to visualize
            output_idx: Output dimension index to visualize
            num_points: Number of points for visualization
            domain: Domain for visualization, defaults to self.domain
            
        Returns:
            Tuple of (x_values, y_values) as numpy arrays for plotting
        """
        domain = domain or self.domain
        x_values = np.linspace(domain[0], domain[1], num_points)
        
        # Convert to tensor for evaluation
        x_tensor = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
        
        # Normalize input if the domain is not the natural domain
        if domain != self.domain:
            x_tensor = self.normalize_domain(x_tensor, domain, self.domain)
        
        # Get the basis functions
        basis_values = self.basis_functions(x_tensor)
        
        # Select the coefficients for the specified input and output
        selected_coeffs = coefficients[input_idx, output_idx]
        
        # Compute the output values by applying coefficients
        y_values = torch.sum(basis_values * selected_coeffs, dim=1)
        
        return x_values, self.to_numpy(y_values)
    
    @staticmethod
    def normalize_domain(x: torch.Tensor, src_domain: Tuple[float, float], 
                        dst_domain: Tuple[float, float]) -> torch.Tensor:
        """
        Map values from source domain to destination domain linearly.
        
        Args:
            x: Input tensor
            src_domain: Source domain (min, max)
            dst_domain: Destination domain (min, max)
            
        Returns:
            Mapped tensor
        """
        src_min, src_max = src_domain
        dst_min, dst_max = dst_domain
        
        # Linear mapping from src_domain to dst_domain
        return ((x - src_min) / (src_max - src_min)) * (dst_max - dst_min) + dst_min