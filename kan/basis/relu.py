import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Union
from .base import BasisFunction


class ReLUBasis(BasisFunction):
    """
    Parameterized ReLU-like basis functions for KAN.
    
    This class implements a family of ReLU-like activation functions that 
    can generalize many popular activation functions through learned parameters.
    The general form is:
    
    f(x; α, β, γ, δ) = γ⋅max(0, x) + α⋅max(0, -x) + β⋅sigmoid(δx)⋅x
    
    This formulation can represent:
    - ReLU: α=0, β=0, γ=1
    - Leaky ReLU: α=small negative value, β=0, γ=1
    - Parametric ReLU (PReLU): α=learned negative slope, β=0, γ=1
    - SiLU/Swish-like: α=0, β≈1, γ≈0, δ≈1
    
    The basis supports creating multiple "shifted" versions of these activation
    functions to provide a rich basis for function approximation.
    """
    
    def __init__(self, num_basis: int, domain: Tuple[float, float] = (-10, 10),
                 init_alphas: Optional[torch.Tensor] = None,
                 init_betas: Optional[torch.Tensor] = None,
                 init_gammas: Optional[torch.Tensor] = None,
                 init_deltas: Optional[torch.Tensor] = None,
                 init_centers: Optional[torch.Tensor] = None,
                 init_scales: Optional[torch.Tensor] = None,
                 learn_parameters: bool = True):
        """
        Initialize the ReLU basis function.
        
        Args:
            num_basis: Number of basis functions to use
            domain: Domain for the basis functions
            init_alphas: Initial values for alpha parameters (negative slope)
            init_betas: Initial values for beta parameters (sigmoid component weight)
            init_gammas: Initial values for gamma parameters (positive slope)
            init_deltas: Initial values for delta parameters (sigmoid steepness)
            init_centers: Initial values for centers of shifted basis functions
            init_scales: Initial values for scales of shifted basis functions
            learn_parameters: Whether to make the parameters learnable
        """
        # The degree parameter in BasisFunction doesn't directly apply to ReLU basis,
        # but we set it to num_basis-1 to maintain compatibility
        super().__init__(degree=num_basis-1)
        
        self.num_basis = num_basis
        self._domain = domain
        self.learn_parameters = learn_parameters
        
        # Initialize parameters
        if init_alphas is None:
            # Default to small negative slopes [-0.1, -0.01, ...]
            alphas = torch.linspace(0.01, 0.1, num_basis)
        else:
            alphas = init_alphas
            
        if init_betas is None:
            # Default to values between 0 and 1
            betas = torch.linspace(0.0, 1.0, num_basis)
        else:
            betas = init_betas
            
        if init_gammas is None:
            # Default to values around 1
            gammas = torch.ones(num_basis)
        else:
            gammas = init_gammas
            
        if init_deltas is None:
            # Default to values between 0.5 and 2
            deltas = torch.linspace(0.5, 2.0, num_basis)
        else:
            deltas = init_deltas
            
        if init_centers is None:
            # Distribute centers across the domain
            centers = torch.linspace(domain[0], domain[1], num_basis)
        else:
            centers = init_centers
            
        if init_scales is None:
            # Default scale based on domain size
            domain_size = domain[1] - domain[0]
            scales = torch.ones(num_basis) * (domain_size / (2 * num_basis))
        else:
            scales = init_scales
        
        # Register parameters or buffers based on whether they should be learned
        if learn_parameters:
            self.alphas = nn.Parameter(alphas)
            self.betas = nn.Parameter(betas)
            self.gammas = nn.Parameter(gammas)
            self.deltas = nn.Parameter(deltas)
            self.centers = nn.Parameter(centers)
            self.scales = nn.Parameter(scales)
        else:
            self.register_buffer("alphas", alphas)
            self.register_buffer("betas", betas)
            self.register_buffer("gammas", gammas)
            self.register_buffer("deltas", deltas)
            self.register_buffer("centers", centers)
            self.register_buffer("scales", scales)
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """
        Register a buffer for the basis function.
        
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
        return f"ParametricReLU-{self.num_basis}"
    
    @property
    def domain(self) -> Tuple[float, float]:
        """Return the domain of the basis functions."""
        return self._domain
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input to the domain.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized input tensor
        """
        # For ReLU basis, tanh normalization is usually sufficient
        return torch.tanh(x) * ((self._domain[1] - self._domain[0]) / 2)
    
    def forward(self, x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the ReLU expansion at points x with given coefficients.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            coefficients: Coefficients tensor of shape (input_dim, output_dim, num_basis)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Ensure x has the right shape (batch_size, input_dim)
        x = x.view(-1, coefficients.shape[0])
        
        # Normalize input
        x = self.normalize_input(x)
        
        # Compute the basis functions
        basis_values = self.basis_functions(x)
        
        # Apply coefficients using einsum
        return torch.einsum('bid,iod->bo', basis_values, coefficients)
    
    def parameterized_relu(self, x: torch.Tensor, alpha: torch.Tensor, 
                          beta: torch.Tensor, gamma: torch.Tensor, 
                          delta: torch.Tensor, center: torch.Tensor, 
                          scale: torch.Tensor) -> torch.Tensor:
        """
        Compute parameterized ReLU-like activation.
        
        Args:
            x: Input tensor
            alpha: Negative slope parameter
            beta: Sigmoid component weight
            gamma: Positive slope parameter
            delta: Sigmoid steepness parameter
            center: Center for the shifted function
            scale: Scale factor for the shift
            
        Returns:
            Activation values
        """
        # Shift and scale the input
        x_shifted = (x - center) / scale
        
        # Compute the three components
        relu_pos = gamma * torch.relu(x_shifted)
        relu_neg = alpha * torch.relu(-x_shifted)
        sigmoid_part = beta * torch.sigmoid(delta * x_shifted) * x_shifted
        
        # Sum the components
        result = relu_pos + relu_neg + sigmoid_part
        return result
    
    def basis_functions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the values of all basis functions at points x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, input_dim, num_basis) containing
            the values of all basis functions at each point
        """
        batch_size, input_dim = x.shape
        
        # Initialize output tensor
        result = torch.zeros(batch_size, input_dim, self.num_basis,
                           device=x.device, dtype=x.dtype)
        
        # For each basis function, compute its value at each point
        for i in range(self.num_basis):
            alpha = self.alphas[i]
            beta = self.betas[i]
            gamma = self.gammas[i]
            delta = self.deltas[i]
            center = self.centers[i]
            scale = self.scales[i]
            
            # Reshape x for broadcasting
            x_reshaped = x.view(batch_size, input_dim)
            
            # Compute the parameterized ReLU for this basis function
            values = self.parameterized_relu(x_reshaped, alpha, beta, gamma, 
                                           delta, center, scale)
            
            # Store the values
            result[:, :, i] = values
        
        return result
    
    def derivative(self, x: torch.Tensor, coefficients: torch.Tensor, 
                  order: int = 1) -> torch.Tensor:
        """
        Compute the derivative of the ReLU expansion.
        
        Args:
            x: Input tensor
            coefficients: Coefficients tensor
            order: Order of the derivative
            
        Returns:
            Derivative tensor of shape (batch_size, output_dim)
        """
        # For higher order derivatives, the behavior depends on the specific parameters
        # For standard ReLU, the second derivative is 0 everywhere except at x=0
        # For smooth approximations involving sigmoid, derivatives exist at all orders
        
        # If order is 0, just return the original function value
        if order == 0:
            return self.forward(x, coefficients)
            
        # For first order derivative, we compute it analytically
        if order == 1:
            # Normalize input
            x = self.normalize_input(x)
            
            batch_size, input_dim = x.shape
            output_dim = coefficients.shape[1]
            
            # Initialize output tensor
            result = torch.zeros(batch_size, output_dim, device=x.device, dtype=x.dtype)
            
            # For each input dimension
            for dim_idx in range(input_dim):
                x_dim = x[:, dim_idx].view(-1, 1)  # Shape: (batch_size, 1)
                
                # Initialize the derivative for this dimension
                dim_deriv = torch.zeros(batch_size, self.num_basis, device=x.device, dtype=x.dtype)
                
                # Compute the derivative for each basis function
                for i in range(self.num_basis):
                    alpha = self.alphas[i]
                    beta = self.betas[i]
                    gamma = self.gammas[i]
                    delta = self.deltas[i]
                    center = self.centers[i]
                    scale = self.scales[i]
                    
                    # Shift and scale the input
                    x_shifted = (x_dim - center) / scale
                    
                    # Compute derivatives of each component
                    # relu_pos derivative: gamma * (x > 0)
                    drelu_pos = gamma * (x_shifted > 0).float() / scale
                    
                    # relu_neg derivative: -alpha * (x < 0)
                    drelu_neg = -alpha * (x_shifted < 0).float() / scale
                    
                    # sigmoid_part derivative: beta * (sigmoid(delta*x) + delta*x*sigmoid(delta*x)*(1-sigmoid(delta*x)))
                    sigmoid_val = torch.sigmoid(delta * x_shifted)
                    dsigmoid_part = beta * (sigmoid_val + delta * x_shifted * sigmoid_val * (1 - sigmoid_val)) / scale
                    
                    # Sum the derivatives
                    dim_deriv[:, i] = drelu_pos.squeeze() + drelu_neg.squeeze() + dsigmoid_part.squeeze()
                
                # Apply coefficients for this dimension
                for out_idx in range(output_dim):
                    coefs = coefficients[dim_idx, out_idx, :]  # Shape: (num_basis,)
                    result[:, out_idx] += torch.sum(dim_deriv * coefs, dim=1)
            
            return result
        
        # For higher order derivatives, we use numerical approximation
        return self.numerical_derivative(x, coefficients, order)