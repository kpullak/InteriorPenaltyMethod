import numpy as np

def constraint_functions(x):
    """
    Constraint functions in the form g(x) <= 0
    
    Args:
        x: Input vector [x1, x2]
        
    Returns:
        Array of constraint values
    """
    return np.array([4*x[0]**2 + x[1]**2 - 16, 
                     3*x[0] + 5*x[1] - 4, 
                     -x[0], 
                     -x[1]])