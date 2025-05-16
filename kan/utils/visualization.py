import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Union
from ..layers.base import KANLayer
from ..basis.base import BasisFunction


def plot_basis_functions(basis: BasisFunction, num_points: int = 200, 
                        max_degree: Optional[int] = None,
                        figsize: Tuple[int, int] = (10, 6),
                        domain: Optional[Tuple[float, float]] = None) -> plt.Figure:
    """
    Plot the basis functions used in a KAN layer.
    
    Args:
        basis: Basis function object
        num_points: Number of points for visualization
        max_degree: Maximum degree to plot (defaults to basis.degree)
        figsize: Figure size
        domain: Domain for plotting (defaults to basis.domain)
        
    Returns:
        Matplotlib figure
    """
    max_degree = max_degree or basis.degree
    max_degree = min(max_degree, basis.degree)
    domain = domain or basis.domain
    
    # Create x values
    x_values = np.linspace(domain[0], domain[1], num_points)
    x_tensor = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
    
    # Normalize if needed
    if domain != basis.domain:
        x_tensor = basis.normalize_domain(x_tensor, domain, basis.domain)
    
    # Get all basis functions
    all_basis = basis.basis_functions(x_tensor)
    all_basis = all_basis.squeeze(1).detach().cpu().numpy()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for i in range(max_degree + 1):
        ax.plot(x_values, all_basis[:, i], label=f'Basis {i}')
    
    ax.set_title(f'{basis.name} Basis Functions')
    ax.set_xlabel('x')
    ax.set_ylabel('Value')
    ax.grid(True)
    ax.legend()
    
    return fig


def plot_layer_functions(layer: KANLayer, input_indices: List[int] = None,
                       output_indices: List[int] = None, 
                       num_points: int = 200,
                       figsize: Tuple[int, int] = (12, 8),
                       domain: Optional[Tuple[float, float]] = None) -> plt.Figure:
    """
    Plot the functions represented by a KAN layer.
    
    Args:
        layer: KAN layer
        input_indices: List of input indices to visualize (defaults to all)
        output_indices: List of output indices to visualize (defaults to all)
        num_points: Number of points for visualization
        figsize: Figure size
        domain: Domain for plotting (defaults to basis.domain)
        
    Returns:
        Matplotlib figure
    """
    # Set default indices if not provided
    input_indices = input_indices or list(range(min(5, layer.input_dim)))
    output_indices = output_indices or list(range(min(5, layer.output_dim)))
    
    # Get coefficients
    coefficients = layer.get_coefficients()
    basis = layer.basis
    domain = domain or basis.domain
    
    # Create the plot
    fig, axes = plt.subplots(len(input_indices), len(output_indices), 
                           figsize=figsize, squeeze=False)
    
    for i, input_idx in enumerate(input_indices):
        for j, output_idx in enumerate(output_indices):
            # Get x and y values
            x_values, y_values = basis.visualize(
                coefficients=coefficients,
                input_idx=input_idx,
                output_idx=output_idx,
                num_points=num_points,
                domain=domain
            )
            
            # Plot
            axes[i, j].plot(x_values, y_values)
            axes[i, j].set_title(f'Input {input_idx} → Output {output_idx}')
            axes[i, j].grid(True)
    
    plt.tight_layout()
    return fig


def plot_network_functions(model, input_tensor: torch.Tensor,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Visualize the transformation of input through the network layers.
    
    Args:
        model: KAN model (sequential or single layer)
        input_tensor: Input tensor to trace through the network
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Check if model is sequential
    if hasattr(model, 'children'):
        layers = [m for m in model.children() if isinstance(m, KANLayer)]
    else:
        layers = [model] if isinstance(model, KANLayer) else []
    
    if not layers:
        raise ValueError("No KAN layers found in the model")
    
    # Track the input through each layer
    layer_inputs = [input_tensor]
    layer_outputs = []
    
    with torch.no_grad():
        current_input = input_tensor
        for layer in layers:
            output = layer(current_input)
            layer_outputs.append(output)
            if len(layers) > 1:  # Don't add the last output as input to a non-existent next layer
                layer_inputs.append(output)
            current_input = output
    
    # Create plot
    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 1, figsize=figsize)
    if n_layers == 1:
        axes = [axes]
    
    for i, layer in enumerate(layers):
        layer_input = layer_inputs[i]
        layer_output = layer_outputs[i]
        
        # Get a subset of dimensions to visualize
        input_dims = min(5, layer.input_dim)
        output_dims = min(5, layer.output_dim)
        
        # Plot inputs
        for j in range(input_dims):
            axes[i].plot(layer_input[:, j].detach().cpu().numpy(), 
                      alpha=0.5, linestyle='--', label=f'Input {j}')
        
        # Plot outputs
        for j in range(output_dims):
            axes[i].plot(layer_output[:, j].detach().cpu().numpy(), 
                      label=f'Output {j}')
        
        axes[i].set_title(f'Layer {i+1}')
        axes[i].grid(True)
        axes[i].legend()
    
    plt.tight_layout()
    return fig


def plot_function_approximation(model, x_true: np.ndarray, y_true: np.ndarray,
                              num_points: int = 200,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the true function versus the approximation by the KAN model.
    
    Args:
        model: KAN model
        x_true: True x values
        y_true: True y values
        num_points: Number of points for visualization
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Prepare input tensor
    x_tensor = torch.tensor(x_true, dtype=torch.float32)
    if len(x_tensor.shape) == 1:
        x_tensor = x_tensor.reshape(-1, 1)
    
    # Get model prediction
    with torch.no_grad():
        y_pred = model(x_tensor).detach().cpu().numpy()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot true function
    ax.plot(x_true, y_true, 'o', markersize=3, alpha=0.7, label='True Function')
    
    # Plot approximation
    ax.plot(x_true, y_pred, '-', linewidth=2, label='KAN Approximation')
    
    ax.set_title('Function Approximation')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()
    
    return fig