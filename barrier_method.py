import itertools

import numpy as np


BARRIER_METHOD_T0 = 1
BARRIER_METHOD_MU_FACTOR = 4
LINE_SEARCH_ALPHA = 0.1
LINE_SEARCH_BETA = 0.1



def objective(X):
    return -np.log(np.linalg.det(X))

def objective_gradient(X):
    return -np.linalg.inv(X)

def constraint(X, ai):
    return (ai.T @ X @ ai) - 1.0

def constraint_gradient(ai):
    return np.outer(ai, ai)

def constraints_barrier(X, a):
    return -sum(np.log(-constraint(X, ai)) for ai in a)

def constraints_barrier_gradient(X, a):
    return np.sum(np.array([-1/constraint(X, ai) * constraint_gradient(ai) for ai in a]), axis=0)

def project_to_definite_positive(X, minimal_eigenvalue):
    eigenvals, eigenvects = np.linalg.eigh(X)
    print(eigenvals)
    eigenvals[eigenvals < 0] = minimal_eigenvalue
    print(eigenvals)
    projected_X = eigenvects @ np.diag(eigenvals) @ eigenvects.T
    return projected_X

# def project_to_constraint(X, a):
#     val = a.T @ X @ a
#     if val <= 1:
#         return X

#     norm = np.linalg.norm(a)
#     k = (val - 1) / (norm ** 4)
#     P = X - k * np.outer(a, a)
#     # print(a.T@P@a)
#     return P

def backtrack_line_search(X, move_direction, objective_function, objective_gradient, alpha = LINE_SEARCH_ALPHA, beta = LINE_SEARCH_BETA):
    t = 1
    while np.isnan(objective_function(X + t * move_direction)) or \
          objective_function(X + t * move_direction) > objective_function(X) - alpha * t * (np.linalg.norm(move_direction) ** 2):
        t = beta * t

    return t

def gradient_descent(objective_function, objective_gradient, initial_guess, max_iterations=20000, tolerance=0.005):

    X = initial_guess

    it = range(max_iterations) if max_iterations else itertools.count()

    for i in it:

        grad = objective_gradient(X)

        if np.linalg.norm(grad) < tolerance:
            break

        step_size = backtrack_line_search(X, -grad, objective_function, objective_gradient)
        assert objective_function(X) - objective_function(X - step_size * grad) >= 0
        X -= step_size * grad

        # print(np.linalg.norm(step_size * (grad)))
        # print("\tbefore:", [v.T @ X @ v for v in ai])
        # X = project_to_definite_positive(X)


        if not i % 2000:
            print(f"round {i}::")
            print("\tafter:", [v.T @ X @ v for v in ai])
            print("\tnorm:", np.linalg.norm(grad))
            print("\tobjective: ", objective_function(X))
            print("\tstep size:", step_size)



    return X


def solve(ai, tolerance = 0.001):
    m, dim = ai.shape # Dimension of the matrix X
    max_norm = np.max(np.linalg.norm(ai, axis=1))

    X = np.eye(dim) / (max_norm**2.01)  # Initial guess for X

    t = BARRIER_METHOD_T0

    mu_factor = BARRIER_METHOD_MU_FACTOR

    while m / t > tolerance:
        current_objective_function = lambda X: t * objective(X) + constraints_barrier(X, ai)
        current_objective_gradient = lambda X: t * objective_gradient(X) + constraints_barrier_gradient(X, ai)
        X =  gradient_descent(objective_function=current_objective_function,
                              objective_gradient=current_objective_gradient,
                              initial_guess=X,
                              max_iterations=None)
        t = t * mu_factor

    return X

# ]]Example usage
ai = 0.1 * np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])  # Replace with your own set of vectors

X_optimized = solve(ai)
print("Optimized matrix X:")
print(X_optimized)
print(objective(X_optimized))
eigenvals, eigenvects = np.linalg.eigh(X_optimized)
print(eigenvals)


for vector in ai:
    print(vector.T@X_optimized@vector)
