import itertools

import numpy as np

BARRIER_METHOD_T0 = 1
BARRIER_METHOD_MU_FACTOR = 4
LINE_SEARCH_ALPHA = 0.4
LINE_SEARCH_BETA = 0.5
GRADIENT_DECENT_MOMENTUM = 0.9
SOLVER_TOLERANCE = 1e-4
GRADIENT_DECENT_CHANGE_TOLERANCE = 1e-12
EPSILON = 1e-4


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

def is_matrix_positive(X):
    # return np.all(np.linalg.eigvals(X) > 0)
    try:
        np.linalg.cholesky(X)
        return True
    except np.linalg.LinAlgError:
        return False

def project_to_definite_positive(X, minimal_eigenvalue):
    eigenvals, eigenvects = np.linalg.eigh(X)
    if not np.any(eigenvals <= 0):
        return X, False
    eigenvals[eigenvals <= 0] = minimal_eigenvalue
    projected_X = eigenvects @ np.diag(eigenvals) @ eigenvects.T

    return projected_X, True

def backtrack_line_search(X, move_direction, objective_function, alpha = LINE_SEARCH_ALPHA, beta = LINE_SEARCH_BETA):
    t = 1
    while np.isnan(objective_function(X + t * move_direction)) or \
          not is_matrix_positive(X + t * move_direction) or \
          objective_function(X + t * move_direction) > objective_function(X) - alpha * t * (np.linalg.norm(move_direction) ** 2):
        t = beta * t

    return t

def gradient_descent(objective_function, objective_gradient, initial_guess, ai, min_iterations = None, tolerance=GRADIENT_DECENT_CHANGE_TOLERANCE, momentum = GRADIENT_DECENT_MOMENTUM):

    X = initial_guess

    change = np.zeros(X.shape)

    old_objective = np.inf

    for i in itertools.count():
        grad = objective_gradient(X)

        step_size = backtrack_line_search(X, -grad, objective_function)

        new_change = step_size * grad + momentum * change

        if np.isnan(objective_function(X - new_change)) or not is_matrix_positive(X - new_change):
            new_change = step_size * grad
            change = np.zeros(X.shape)
        else:
            change = new_change

        X -= new_change

        X, changed = project_to_definite_positive(X, minimal_eigenvalue=EPSILON)
        if changed:
            m = max([constraint(X, v) + 1 for v in ai])
            if m > 1:
                X = X / (m + EPSILON)

        new_objective = objective_function(X)
        if old_objective - new_objective < tolerance and (not min_iterations or i > min_iterations):
            break
        old_objective = new_objective

    return X


def solve(A: np.ndarray) -> np.ndarray:
    m, dim = A.shape # Dimension of the matrix X
    max_norm = np.max(np.linalg.norm(A, axis=1))

    X = np.eye(dim) / (max_norm**2.01)  # Initial guess for X

    t = BARRIER_METHOD_T0
    mu_factor = BARRIER_METHOD_MU_FACTOR

    while m / t > SOLVER_TOLERANCE:
        print(t)
        current_objective_function = lambda X: t * objective(X) + constraints_barrier(X, ai)
        current_objective_gradient = lambda X: t * objective_gradient(X) + constraints_barrier_gradient(X, ai)
        X =  gradient_descent(objective_function=current_objective_function,
                              objective_gradient=current_objective_gradient,
                              initial_guess=X,
                              ai=A,
                              min_iterations=20)
        t = t * mu_factor


    # we care that the last iteration of the barrier will more precise than the rest of the iterations
    X =  gradient_descent(objective_function=current_objective_function,
                        objective_gradient=current_objective_gradient,
                        initial_guess=X,
                        ai=A,
                        min_iterations=50)


    return np.linalg.inv(X)
