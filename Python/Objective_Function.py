import numpy as np

def objective_function(x):
    """
    Objective function: f(x) = x₁² + x₂² - 6x₁ - 8x₂ + 10
    
    Args:
        x: Input vector [x1, x2]
        
    Returns:
        Function value at point x
    """
    return x[0]**2 + x[1]**2 - 6*x[0] - 8*x[1] + 10