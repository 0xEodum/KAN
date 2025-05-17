import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import torch
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from ..layers.base import KANLayer
from ..basis.chebyshev import ChebyshevBasis
from ..basis.jacobi import JacobiBasis
from ..basis.hermite import HermiteBasis


def require_sympy():
    """Check if sympy is available, raise error if not."""
    if not SYMPY_AVAILABLE:
        raise ImportError(
            "Symbolic operations require sympy. "
            "Please install it with 'pip install sympy'."
        )


def get_chebyshev_symbolic(degree: int) -> List[sp.Expr]:
    """
    Get symbolic expressions for Chebyshev polynomials up to the given degree.
    
    Args:
        degree: Maximum degree of polynomials
        
    Returns:
        List of symbolic expressions for T_0(x) to T_degree(x)
    """
    require_sympy()
    
    x = sp.Symbol('x')
    cheby_polys = [None] * (degree + 1)
    
    # Initial values
    cheby_polys[0] = 1
    if degree > 0:
        cheby_polys[1] = x
    
    # Recurrence relation
    for n in range(2, degree + 1):
        cheby_polys[n] = 2 * x * cheby_polys[n-1] - cheby_polys[n-2]
    
    return cheby_polys


def get_jacobi_symbolic(degree: int, alpha: float, beta: float) -> List[sp.Expr]:
    """
    Get symbolic expressions for Jacobi polynomials up to the given degree.
    
    Args:
        degree: Maximum degree of polynomials
        alpha: First parameter for Jacobi polynomials (α > -1)
        beta: Second parameter for Jacobi polynomials (β > -1)
        
    Returns:
        List of symbolic expressions for P_0^(α,β)(x) to P_degree^(α,β)(x)
    """
    require_sympy()
    
    x = sp.Symbol('x')
    jacobi_polys = [None] * (degree + 1)
    
    # Initial values
    jacobi_polys[0] = 1
    if degree > 0:
        jacobi_polys[1] = ((alpha + beta + 2) * x + (alpha - beta)) / 2
    
    # Recurrence relation for n ≥ 2
    for n in range(2, degree + 1):
        n_float = float(n)
        ab_sum = alpha + beta
        ab_diff = alpha**2 - beta**2
        
        # Common term in numerator: (2n + α + β - 1)
        common_term = 2 * n_float + ab_sum - 1
        
        # Coefficient for P_{n-1}
        coef1 = common_term * ((2 * n_float + ab_sum) * (2 * n_float + ab_sum - 2) * x + ab_diff)
        
        # Coefficient for P_{n-2}
        coef2 = -2 * (n_float + alpha - 1) * (n_float + beta - 1) * (2 * n_float + ab_sum)
        
        # Denominator
        denom = 2 * n_float * (n_float + ab_sum) * (2 * n_float + ab_sum - 2)
        
        # Compute P_n
        jacobi_polys[n] = (coef1 * jacobi_polys[n-1] + coef2 * jacobi_polys[n-2]) / denom
    
    return jacobi_polys


def get_hermite_symbolic(degree: int, scaling: str = 'physicist') -> List[sp.Expr]:
    """
    Get symbolic expressions for Hermite polynomials up to the given degree.
    
    Args:
        degree: Maximum degree of polynomials
        scaling: Type of Hermite polynomials:
                 'physicist' (default) - H_n(x) with recurrence H_{n+1} = 2x·H_n - 2n·H_{n-1}
                 'probabilist' - He_n(x) with recurrence He_{n+1} = x·He_n - n·He_{n-1}
        
    Returns:
        List of symbolic expressions for H_0(x) to H_degree(x) or He_0(x) to He_degree(x)
    """
    require_sympy()
    
    x = sp.Symbol('x')
    hermite_polys = [None] * (degree + 1)
    
    # Initial values
    hermite_polys[0] = 1
    
    if degree > 0:
        if scaling == 'physicist':
            hermite_polys[1] = 2 * x
        else:  # scaling == 'probabilist'
            hermite_polys[1] = x
    
    # Recurrence relation
    for n in range(2, degree + 1):
        if scaling == 'physicist':
            # H_{n}(x) = 2x·H_{n-1}(x) - 2(n-1)·H_{n-2}(x)
            hermite_polys[n] = 2 * x * hermite_polys[n-1] - 2 * (n-1) * hermite_polys[n-2]
        else:  # scaling == 'probabilist'
            # He_{n}(x) = x·He_{n-1}(x) - (n-1)·He_{n-2}(x)
            hermite_polys[n] = x * hermite_polys[n-1] - (n-1) * hermite_polys[n-2]
    
    return hermite_polys


def get_layer_symbolic_expr(layer: KANLayer, input_var_names: Optional[List[str]] = None) -> Dict[int, Union[sp.Expr, str]]:
    """
    Get symbolic expressions for each output of a KAN layer.
    
    Args:
        layer: KAN layer
        input_var_names: Names for input variables (defaults to 'x_0', 'x_1', ...)
        
    Returns:
        Dictionary mapping output indices to symbolic expressions
    """
    require_sympy()
    
    # Check if this is a type of layer we can handle
    if not hasattr(layer, 'basis_function'):
        return {0: "Cannot generate symbolic expression for this layer type"}
    
    # Get layer parameters
    basis = layer.basis_function
    coeffs = layer.get_coefficients().detach().cpu().numpy()
    input_dim, output_dim, degree_plus_one = coeffs.shape
    
    # Set default input variable names if not provided
    if input_var_names is None:
        input_var_names = [f'x_{i}' for i in range(input_dim)]
    
    # Create symbolic variables for inputs
    input_vars = [sp.Symbol(name) for name in input_var_names]
    
    # Different handling based on basis type
    result = {}
    
    if isinstance(basis, ChebyshevBasis):
        # Get symbolic Chebyshev polynomials
        cheby_polys = get_chebyshev_symbolic(basis.degree)
        
        # For each output dimension
        for o in range(output_dim):
            # Initialize expression for this output
            expr = 0
            
            # For each input dimension
            for i in range(input_dim):
                # Get the transformed input (applying tanh for normalization)
                x_transformed = sp.tanh(input_vars[i])
                
                # Sum up the contribution from each basis function
                input_expr = 0
                for d in range(basis.degree + 1):
                    # Substitute the transformed input into the Chebyshev polynomial
                    poly_expr = cheby_polys[d].subs(sp.Symbol('x'), x_transformed)
                    # Multiply by coefficient and add to the sum
                    input_expr += coeffs[i, o, d] * poly_expr
                
                # Add this input's contribution to the output
                expr += input_expr
            
            # Store the expression for this output
            result[o] = expr
    elif isinstance(basis, JacobiBasis):
        # Get symbolic Jacobi polynomials
        jacobi_polys = get_jacobi_symbolic(basis.degree, basis.alpha, basis.beta)
        
        # For each output dimension
        for o in range(output_dim):
            # Initialize expression for this output
            expr = 0
            
            # For each input dimension
            for i in range(input_dim):
                # Get the transformed input (applying tanh for normalization)
                x_transformed = sp.tanh(input_vars[i])
                
                # Sum up the contribution from each basis function
                input_expr = 0
                for d in range(basis.degree + 1):
                    # Substitute the transformed input into the Jacobi polynomial
                    poly_expr = jacobi_polys[d].subs(sp.Symbol('x'), x_transformed)
                    # Multiply by coefficient and add to the sum
                    input_expr += coeffs[i, o, d] * poly_expr
                
                # Add this input's contribution to the output
                expr += input_expr
            
            # Store the expression for this output
            result[o] = expr
    elif isinstance(basis, HermiteBasis):
        # Get symbolic Hermite polynomials
        hermite_polys = get_hermite_symbolic(basis.degree, basis.scaling)
        
        # For each output dimension
        for o in range(output_dim):
            # Initialize expression for this output
            expr = 0
            
            # For each input dimension
            for i in range(input_dim):
                # Get the transformed input using the basis's normalization function
                # For Hermite, we use a scaled tanh to handle the unbounded domain
                if hasattr(basis, '_normalize_domain') and basis._normalize_domain:
                    x_transformed = sp.tanh(input_vars[i]) * 3.0
                else:
                    x_transformed = input_vars[i]
                
                # Sum up the contribution from each basis function
                input_expr = 0
                for d in range(basis.degree + 1):
                    # Substitute the transformed input into the Hermite polynomial
                    poly_expr = hermite_polys[d].subs(sp.Symbol('x'), x_transformed)
                    # Multiply by coefficient and add to the sum
                    input_expr += coeffs[i, o, d] * poly_expr
                
                # Add this input's contribution to the output
                expr += input_expr
            
            # Store the expression for this output
            result[o] = expr
    else:
        # For unsupported basis types
        for o in range(output_dim):
            result[o] = f"Symbolic expression not implemented for {type(basis).__name__}"
    
    return result


def get_network_symbolic_expr(model, input_var_names: Optional[List[str]] = None) -> Dict[int, Union[sp.Expr, str]]:
    """
    Get symbolic expressions for the outputs of a KAN network.
    
    Args:
        model: KAN model (sequential or single layer)
        input_var_names: Names for input variables (defaults to 'x_0', 'x_1', ...)
        
    Returns:
        Dictionary mapping output indices to symbolic expressions
    """
    require_sympy()
    
    # Check if model is sequential
    if hasattr(model, 'children'):
        layers = [m for m in model.children() if isinstance(m, KANLayer)]
    else:
        layers = [model] if isinstance(m, KANLayer) else []
    
    if not layers:
        return {0: "No KAN layers found in the model"}
    
    # If it's a single layer, just get its expressions
    if len(layers) == 1:
        return get_layer_symbolic_expr(layers[0], input_var_names)
    
    # For multi-layer networks, we need to compose the expressions
    current_exprs = get_layer_symbolic_expr(layers[0], input_var_names)
    
    # Process each subsequent layer
    for layer_idx in range(1, len(layers)):
        layer = layers[layer_idx]
        
        # Get symbolic expressions for this layer
        # We need to create temporary variable names for the outputs of the previous layer
        temp_var_names = [f'temp_{i}' for i in range(len(current_exprs))]
        layer_exprs = get_layer_symbolic_expr(layer, temp_var_names)
        
        # Create substitution map
        subs_map = {sp.Symbol(temp_var_names[i]): expr 
                   for i, expr in current_exprs.items()
                   if isinstance(expr, sp.Expr)}
        
        # Apply substitutions to compose the expressions
        new_exprs = {}
        for out_idx, expr in layer_exprs.items():
            if isinstance(expr, sp.Expr):
                new_exprs[out_idx] = expr.subs(subs_map)
            else:
                new_exprs[out_idx] = expr
        
        current_exprs = new_exprs
    
    return current_exprs


def simplify_expressions(exprs: Dict[int, sp.Expr], 
                        simplify_method: str = 'basic') -> Dict[int, sp.Expr]:
    """
    Simplify symbolic expressions.
    
    Args:
        exprs: Dictionary mapping indices to expressions
        simplify_method: Simplification method ('basic', 'full', or 'rational')
        
    Returns:
        Dictionary mapping indices to simplified expressions
    """
    require_sympy()
    
    result = {}
    for idx, expr in exprs.items():
        if isinstance(expr, sp.Expr):
            if simplify_method == 'basic':
                result[idx] = sp.expand(expr)
            elif simplify_method == 'full':
                result[idx] = sp.simplify(expr)
            elif simplify_method == 'rational':
                result[idx] = sp.cancel(expr)
            else:
                result[idx] = expr
        else:
            result[idx] = expr
    
    return result


def export_to_latex(exprs: Dict[int, Union[sp.Expr, str]]) -> Dict[int, str]:
    """
    Convert symbolic expressions to LaTeX strings.
    
    Args:
        exprs: Dictionary mapping indices to expressions
        
    Returns:
        Dictionary mapping indices to LaTeX strings
    """
    require_sympy()
    
    result = {}
    for idx, expr in exprs.items():
        if isinstance(expr, sp.Expr):
            result[idx] = sp.latex(expr)
        else:
            result[idx] = str(expr)
    
    return result


def export_to_function(exprs: Dict[int, Union[sp.Expr, str]], 
                     function_name: str = 'kan_function',
                     library: str = 'numpy') -> str:
    """
    Convert symbolic expressions to a Python function.
    
    Args:
        exprs: Dictionary mapping indices to expressions
        function_name: Name for the generated function
        library: Numerical library to use ('numpy', 'torch', or 'jax')
        
    Returns:
        String containing the Python function definition
    """
    require_sympy()
    
    # Get all variables used in the expressions
    all_vars = set()
    for expr in exprs.values():
        if isinstance(expr, sp.Expr):
            all_vars.update(expr.free_symbols)
    
    # Sort variables by name
    sorted_vars = sorted(all_vars, key=lambda s: s.name)
    var_names = [var.name for var in sorted_vars]
    
    # Convert expressions to the appropriate library
    if library == 'numpy':
        import_line = "import numpy as np"
        module = "np"
    elif library == 'torch':
        import_line = "import torch"
        module = "torch"
    elif library == 'jax':
        import_line = "import jax.numpy as jnp"
        module = "jnp"
    else:
        raise ValueError(f"Unsupported library: {library}")
    
    # Create function definition
    function_lines = [
        import_line,
        "",
        f"def {function_name}({', '.join(var_names)}):",
        "    # Generated KAN function",
    ]
    
    # Add computation for each output
    for idx, expr in exprs.items():
        if isinstance(expr, sp.Expr):
            # Convert to string with the appropriate library
            expr_str = str(expr)
            # Replace mathematical functions with library equivalents
            replacements = {
                'sin': f'{module}.sin',
                'cos': f'{module}.cos',
                'tan': f'{module}.tan',
                'exp': f'{module}.exp',
                'log': f'{module}.log',
                'sqrt': f'{module}.sqrt',
                'tanh': f'{module}.tanh',
                'acos': f'{module}.arccos',
                'asin': f'{module}.arcsin',
                'atan': f'{module}.arctan'
            }
            for old, new in replacements.items():
                expr_str = expr_str.replace(old + '(', new + '(')
            
            function_lines.append(f"    y_{idx} = {expr_str}")
        else:
            function_lines.append(f"    # Output {idx}: {expr}")
            function_lines.append(f"    y_{idx} = None")
    
    # Return all outputs
    output_vars = [f"y_{idx}" for idx in sorted(exprs.keys())]
    return_line = f"    return {', '.join(output_vars)}"
    if len(output_vars) > 1:
        return_line = f"    return ({', '.join(output_vars)})"
    function_lines.append(return_line)
    
    return "\n".join(function_lines)