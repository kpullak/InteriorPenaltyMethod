## Interior Penalty Method Overview

The **Interior Penalty Method** (also called the **Barrier Method**) is a technique for solving constrained optimization problems by transforming them into a series of unconstrained problems.

### How It Works

1. **Original Problem:**
   ```
   minimize f(x)
   subject to g(x) ≤ 0
   ```

2. **Transformed Problem:**
   ```
   minimize f(x) - μ × Σ log(-g_i(x))
   ```
   where μ > 0 is the barrier parameter.

### Key Ideas

- **Barrier Function:** The term `-μ × log(-g_i(x))` creates an invisible "barrier" at the constraint boundary
- **Interior Points Only:** The logarithm is only defined when g_i(x) < 0 (inside feasible region)
- **Automatic Penalty:** As you approach a constraint boundary (g_i(x) → 0), the barrier term → +∞
- **Parameter Sequence:** Start with small μ, gradually increase it. As μ → ∞, the solution approaches the true optimum

### Simple Example
If you have constraint `x ≥ 0`, this becomes `-x ≤ 0`. The barrier term `-μ log(x)` will:
- Be finite when x > 0 (feasible)
- Go to +∞ as x → 0 (approaching boundary)
- Be undefined when x < 0 (infeasible)

This naturally keeps the optimization algorithm inside the feasible region!

## Functions

### Objective Function
```python
def objective_function(x):
    return x[0]**2 + x[1]**2 - 6*x[0] - 8*x[1] + 10
```

**Mathematical form:** f(x) = x₁² + x₂² - 6x₁ - 8x₂ + 10

### Constraint Functions
```python
def constraint_functions(x):
    return np.array([4*x[0]**2 + x[1]**2 - 16, 
                     3*x[0] + 5*x[1] - 4, 
                     -x[0], 
                     -x[1]])
```

**Mathematical form:** All constraints in g(x) ≤ 0 format
1. 4x₁² + x₂² - 16 ≤ 0 (ellipse constraint)
2. 3x₁ + 5x₂ - 4 ≤ 0 (linear constraint)
3. x₁ ≥ 0 (non-negativity)
4. x₂ ≥ 0 (non-negativity)

## Usage

```python
import numpy as np

# Example point
x = [1.0, 2.0]

# Evaluate objective function
obj_value = objective_function(x)
print(f"Objective value: {obj_value}")

# Evaluate constraints
constraints = constraint_functions(x)
print(f"Constraint values: {constraints}")

# Check feasibility (all constraints should be ≤ 0)
feasible = np.all(constraints <= 0)
print(f"Point is feasible: {feasible}")
```

## Requirements

- Python 3.6+
- NumPy

## Installation

```bash
pip install numpy
```

## Problem Description

This defines a constrained optimization problem:

**Minimize:** f(x₁, x₂) = x₁² + x₂² - 6x₁ - 8x₂ + 10

**Subject to:**
- 4x₁² + x₂² ≤ 16
- 3x₁ + 5x₂ ≤ 4  
- x₁ ≥ 0
- x₂ ≥ 0