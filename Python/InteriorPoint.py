import numpy as np
import pandas as pd
from typing import Callable, Tuple, List, Union
import warnings
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

class BarrierMethodOptimizer:
    """
    Implementation of the Barrier Method (Interior Point Method) for constrained optimization.
    
    Solves problems of the form:
        minimize    f(x)
        subject to  g_i(x) ≤ 0, i = 1, 2, ..., m
    
    The method transforms the constrained problem into a sequence of unconstrained problems
    using logarithmic barrier functions.
    """
    
    def __init__(self, 
                 objective_function: Callable[[np.ndarray], float],
                 constraint_functions: Callable[[np.ndarray], np.ndarray],
                 objective_gradient: Callable[[np.ndarray], np.ndarray] = None,
                 constraint_gradients: Callable[[np.ndarray], np.ndarray] = None):
        """
        Initialize the Barrier Method Optimizer.
        
        Args:
            objective_function: Function f(x) to minimize
            constraint_functions: Function returning array of constraints g(x) ≤ 0
            objective_gradient: Gradient of objective function (optional)
            constraint_gradients: Gradients of constraints (optional)
        """
        self.objective_function = objective_function
        self.constraint_functions = constraint_functions
        self.objective_gradient = objective_gradient
        self.constraint_gradients = constraint_gradients
        
        # Algorithm parameters
        self.epsilon_relative_change = 1e-8
        self.initial_mu = 0.1
        self.mu_multiplier = 10.0
        self.max_mu = 1e8
        self.initial_step_size = 0.005
        self.step_reduction_factor = 0.2
        self.wolfe_c1 = 1e-4  # Armijo condition parameter
        self.wolfe_c2 = 0.9   # Curvature condition parameter
        
        # Results storage
        self.iteration_history = []
        self.convergence_data = []
    
    def barrier_function(self, x: np.ndarray, mu: float) -> float:
        """
        Compute the barrier function value.
        
        Barrier function: f(x) - μ * Σ log(-g_i(x))
        
        Args:
            x: Current point
            mu: Barrier parameter
            
        Returns:
            Barrier function value
        """
        try:
            objective_value = self.objective_function(x)
            constraints = self.constraint_functions(x)
            
            # Check feasibility
            if np.any(constraints >= 0):
                return np.inf  # Point is infeasible
            
            # Compute barrier term: -μ * Σ log(-g_i(x))
            barrier_term = -mu * np.sum(np.log(-constraints))
            
            return objective_value + barrier_term
            
        except (ValueError, RuntimeWarning):
            return np.inf
    
    def barrier_gradient(self, x: np.ndarray, mu: float) -> np.ndarray:
        """
        Compute the gradient of the barrier function.
        
        ∇f_barrier(x) = ∇f(x) - μ * Σ (∇g_i(x) / g_i(x))
        
        Args:
            x: Current point
            mu: Barrier parameter
            
        Returns:
            Gradient of barrier function
        """
        # Use numerical gradient if analytical gradient not provided
        if self.objective_gradient is None:
            obj_grad = self._numerical_gradient(self.objective_function, x)
        else:
            obj_grad = self.objective_gradient(x)
        
        constraints = self.constraint_functions(x)
        
        # Check feasibility
        if np.any(constraints >= 0):
            return np.full_like(x, np.inf)
        
        # Compute constraint gradient contribution
        if self.constraint_gradients is None:
            constraint_grad_contribution = np.zeros_like(x)
            for i in range(len(constraints)):
                const_grad = self._numerical_gradient(
                    lambda y: self.constraint_functions(y)[i], x
                )
                constraint_grad_contribution -= mu * const_grad / constraints[i]
        else:
            constraint_grads = self.constraint_gradients(x)
            constraint_grad_contribution = -mu * np.sum(
                constraint_grads / constraints[:, np.newaxis], axis=0
            )
        
        return obj_grad + constraint_grad_contribution
    
    def _numerical_gradient(self, func: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """Compute numerical gradient using finite differences."""
        gradient = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return gradient
    
    def is_feasible(self, x: np.ndarray) -> bool:
        """
        Check if a point is feasible (all constraints satisfied).
        
        Args:
            x: Point to check
            
        Returns:
            True if feasible, False otherwise
        """
        constraints = self.constraint_functions(x)
        return np.all(constraints < 0)
    
    def is_decreasing_function(self, x_new: np.ndarray, x_old: np.ndarray, mu: float) -> bool:
        """
        Check if the relative change in barrier function is small enough for convergence.
        
        Args:
            x_new: New point
            x_old: Old point  
            mu: Current barrier parameter
            
        Returns:
            True if converged, False otherwise
        """
        f_new = self.barrier_function(x_new, mu)
        f_old = self.barrier_function(x_old, mu)
        
        if f_old == 0:
            return abs(f_new) < self.epsilon_relative_change
        
        relative_change = abs((f_new - f_old) / f_old)
        return relative_change < self.epsilon_relative_change
    
    def line_search_armijo(self, x: np.ndarray, search_direction: np.ndarray, 
                          mu: float, initial_step: float = 1.0) -> float:
        """
        Perform line search using Armijo condition to ensure sufficient decrease.
        
        Args:
            x: Current point
            search_direction: Search direction
            mu: Barrier parameter
            initial_step: Initial step size
            
        Returns:
            Optimal step size
        """
        alpha = initial_step
        gradient = self.barrier_gradient(x, mu)
        directional_derivative = np.dot(gradient, search_direction)
        
        # If direction is not a descent direction, return small step
        if directional_derivative >= 0:
            return 1e-6
        
        f_current = self.barrier_function(x, mu)
        
        # Backtracking line search
        max_iterations = 50
        for _ in range(max_iterations):
            x_new = x + alpha * search_direction
            
            # Check feasibility
            if not self.is_feasible(x_new):
                alpha *= self.step_reduction_factor
                continue
            
            f_new = self.barrier_function(x_new, mu)
            
            # Armijo condition
            if f_new <= f_current + self.wolfe_c1 * alpha * directional_derivative:
                return alpha
            
            alpha *= self.step_reduction_factor
        
        return alpha
    
    def optimize_for_fixed_mu(self, x_start: np.ndarray, mu: float, 
                             max_iterations: int = 100) -> Tuple[np.ndarray, bool]:
        """
        Optimize the barrier function for a fixed value of mu.
        
        Args:
            x_start: Starting point
            mu: Barrier parameter
            max_iterations: Maximum iterations for this subproblem
            
        Returns:
            Optimal point and convergence flag
        """
        x_current = x_start.copy()
        
        for iteration in range(max_iterations):
            # Compute search direction (negative gradient)
            gradient = self.barrier_gradient(x_current, mu)
            
            if np.any(np.isinf(gradient)) or np.any(np.isnan(gradient)):
                warnings.warn(f"Invalid gradient at iteration {iteration}")
                return x_current, False
            
            search_direction = -gradient
            
            # Check for convergence (gradient norm)
            if np.linalg.norm(gradient) < 1e-6:
                return x_current, True
            
            # Perform line search
            step_size = self.line_search_armijo(x_current, search_direction, mu)
            
            # Update point
            x_new = x_current + step_size * search_direction
            
            # Check feasibility
            if not self.is_feasible(x_new):
                step_size *= 0.1
                x_new = x_current + step_size * search_direction
                
                if not self.is_feasible(x_new):
                    warnings.warn(f"Cannot maintain feasibility at iteration {iteration}")
                    return x_current, False
            
            # Check for convergence
            if self.is_decreasing_function(x_new, x_current, mu):
                return x_new, True
            
            x_current = x_new
        
        return x_current, False
    
    def optimize(self, x0: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, float, dict]:
        """
        Main optimization routine using the barrier method.
        
        Args:
            x0: Initial feasible point
            verbose: Print progress information
            
        Returns:
            Optimal point, optimal value, and optimization info
        """
        # Verify initial point is feasible
        if not self.is_feasible(x0):
            raise ValueError("Initial point must be feasible (all constraints < 0)")
        
        x_current = x0.copy()
        mu_current = self.initial_mu
        
        if verbose:
            print("="*80)
            print("BARRIER METHOD OPTIMIZATION")
            print("="*80)
            print(f"Initial point: {x0}")
            print(f"Initial barrier parameter: {mu_current}")
            print("-"*80)
        
        outer_iteration = 0
        
        while mu_current < self.max_mu:
            outer_iteration += 1
            
            if verbose:
                print(f"\nOuter Iteration {outer_iteration}: μ = {mu_current:.2e}")
            
            # Solve subproblem for current mu
            x_optimal, converged = self.optimize_for_fixed_mu(x_current, mu_current)
            
            if not converged:
                warnings.warn(f"Subproblem did not converge for μ = {mu_current}")
            
            # Store iteration data
            barrier_value = self.barrier_function(x_optimal, mu_current)
            objective_value = self.objective_function(x_optimal)
            
            self.iteration_history.append({
                'outer_iteration': outer_iteration,
                'mu': mu_current,
                'x': x_optimal.copy(),
                'objective': objective_value,
                'barrier_value': barrier_value,
                'feasible': self.is_feasible(x_optimal)
            })
            
            if verbose:
                print(f"  Solution: {x_optimal}")
                print(f"  Objective: {objective_value:.6f}")
                print(f"  Feasible: {self.is_feasible(x_optimal)}")
            
            # Check overall convergence
            if outer_iteration > 1:
                prev_x = self.iteration_history[-2]['x']
                if np.linalg.norm(x_optimal - prev_x) < 1e-8:
                    if verbose:
                        print(f"\nConverged: Solution change < tolerance")
                    break
            
            # Update for next iteration
            x_current = x_optimal
            mu_current *= self.mu_multiplier
        
        # Final results
        final_objective = self.objective_function(x_current)
        final_constraints = self.constraint_functions(x_current)
        
        optimization_info = {
            'outer_iterations': outer_iteration,
            'final_mu': mu_current / self.mu_multiplier,
            'final_constraints': final_constraints,
            'constraint_violations': np.maximum(0, final_constraints),
            'iteration_history': self.iteration_history
        }
        
        if verbose:
            print("="*80)
            print("OPTIMIZATION COMPLETE")
            print("="*80)
            print(f"Final solution: {x_current}")
            print(f"Final objective: {final_objective:.8f}")
            print(f"Constraint values: {final_constraints}")
            print(f"Max constraint violation: {np.max(final_constraints):.2e}")
            print(f"Total outer iterations: {outer_iteration}")
        
        return x_current, final_objective, optimization_info
    
    def plot_convergence(self):
        """Plot convergence history."""
        if not self.iteration_history:
            print("No optimization history to plot")
            return
        
        iterations = [data['outer_iteration'] for data in self.iteration_history]
        objectives = [data['objective'] for data in self.iteration_history]
        mus = [data['mu'] for data in self.iteration_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot objective function
        ax1.plot(iterations, objectives, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Outer Iteration')
        ax1.set_ylabel('Objective Function Value')
        ax1.set_title('Convergence of Objective Function')
        ax1.grid(True, alpha=0.3)
        
        # Plot barrier parameter
        ax2.semilogy(iterations, mus, 'r-s', linewidth=2, markersize=6)
        ax2.set_xlabel('Outer Iteration')
        ax2.set_ylabel('Barrier Parameter μ')
        ax2.set_title('Barrier Parameter Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage and test functions
def example_quadratic_problem():
    """
    Example: Minimize (x1-2)² + (x2-1)² 
    Subject to: x1 + x2 - 3 ≤ 0
                x1 - 1 ≤ 0  
                x2 - 1 ≤ 0
                -x1 ≤ 0
                -x2 ≤ 0
    """
    
    def objective_function(x):
        return (x[0] - 2)**2 + (x[1] - 1)**2
    
    def constraint_functions(x):
        return np.array([
            x[0] + x[1] - 3,  # x1 + x2 ≤ 3
            x[0] - 1,         # x1 ≤ 1
            x[1] - 1,         # x2 ≤ 1  
            -x[0],            # x1 ≥ 0
            -x[1]             # x2 ≥ 0
        ])
    
    def objective_gradient(x):
        return np.array([2*(x[0] - 2), 2*(x[1] - 1)])
    
    def constraint_gradients(x):
        return np.array([
            [1, 1],   # ∇(x1 + x2 - 3)
            [1, 0],   # ∇(x1 - 1)  
            [0, 1],   # ∇(x2 - 1)
            [-1, 0],  # ∇(-x1)
            [0, -1]   # ∇(-x2)
        ])
    
    return objective_function, constraint_functions, objective_gradient, constraint_gradients


def example_rosenbrock_problem():
    """
    Example: Minimize Rosenbrock function with circular constraint
    f(x) = 100*(x2 - x1²)² + (1 - x1)²
    Subject to: x1² + x2² - 1 ≤ 0
    """
    
    def objective_function(x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    def constraint_functions(x):
        return np.array([x[0]**2 + x[1]**2 - 1])
    
    def objective_gradient(x):
        df_dx1 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        df_dx2 = 200 * (x[1] - x[0]**2)
        return np.array([df_dx1, df_dx2])
    
    def constraint_gradients(x):
        return np.array([[2*x[0], 2*x[1]]])
    
    return objective_function, constraint_functions, objective_gradient, constraint_gradients


if __name__ == "__main__":
    print("BARRIER METHOD OPTIMIZATION EXAMPLES")
    print("="*60)
    
    # Example 1: Quadratic Problem
    print("\nExample 1: Quadratic Problem with Linear Constraints")
    obj_func, const_func, obj_grad, const_grad = example_quadratic_problem()
    
    optimizer = BarrierMethodOptimizer(
        objective_function=obj_func,
        constraint_functions=const_func,
        objective_gradient=obj_grad,
        constraint_gradients=const_grad
    )
    
    # Start from feasible point
    x0 = np.array([0.4, 0.4])
    x_opt, f_opt, info = optimizer.optimize(x0)
    
    print(f"\nOptimal solution: {x_opt}")
    print(f"Optimal value: {f_opt:.8f}")
    
    # Example 2: Rosenbrock Problem  
    print("\n" + "="*60)
    print("\nExample 2: Constrained Rosenbrock Problem")
    obj_func2, const_func2, obj_grad2, const_grad2 = example_rosenbrock_problem()
    
    optimizer2 = BarrierMethodOptimizer(
        objective_function=obj_func2,
        constraint_functions=const_func2,
        objective_gradient=obj_grad2,
        constraint_gradients=const_grad2
    )
    
    # Start from feasible point inside unit circle
    x0_2 = np.array([0.5, 0.5])
    x_opt2, f_opt2, info2 = optimizer2.optimize(x0_2)
    
    print(f"\nOptimal solution: {x_opt2}")
    print(f"Optimal value: {f_opt2:.8f}")
    
    # Plot convergence
    try:
        optimizer.plot_convergence()
    except ImportError:
        print("Matplotlib not available for plotting")