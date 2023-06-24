import itertools

import numpy as np


BARRIER_METHOD_T0 = 1
BARRIER_METHOD_MU_FACTOR = 4
LINE_SEARCH_ALPHA = 0.4
LINE_SEARCH_BETA = 0.5



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
    if not np.any(eigenvals < 0):
        return X, False
    print('!!!!')
    eigenvals[eigenvals < 0] = minimal_eigenvalue
    projected_X = eigenvects @ np.diag(eigenvals) @ eigenvects.T

    return projected_X, True


# def project_to_constraint(X, a):
#     val = a.T @ X @ a
#     if val <= 1:
#         return X

#     norm = np.linalg.norm(a)
#     k = (val - 1) / (norm ** 4)
#     P = X - k * np.outer(a, a)
#     # print(a.T@P@a)
#     return P

def backtrack_line_search(X, move_direction, objective_function, alpha = LINE_SEARCH_ALPHA, beta = LINE_SEARCH_BETA):
    t = 1
    while np.isnan(objective_function(X + t * move_direction)) or \
          not is_matrix_positive(X + t * move_direction) or \
          objective_function(X + t * move_direction) > objective_function(X) - alpha * t * (np.linalg.norm(move_direction) ** 2):
        t = beta * t

    return t

def gradient_descent(objective_function, objective_gradient, initial_guess, ai, max_iterations=20000, tolerance=1e-6, momentum = 0.90):

    X = initial_guess

    it = range(max_iterations) if max_iterations else itertools.count()

    change = np.zeros(X.shape)

    for i in it:
        grad = objective_gradient(X)

        step_size = backtrack_line_search(X, -grad, objective_function)

        if np.linalg.norm(grad) < tolerance:
            break

        new_change = step_size * grad + momentum * change


        if np.isnan(objective_function(X - new_change)) or not is_matrix_positive(X - new_change):
            new_change = step_size * grad
            change = 0
        else:
            change = new_change

        X -= new_change
        # print(objective_function(X))

        X, changed = project_to_definite_positive(X, minimal_eigenvalue=0.0001)
        if changed:
            m = max([constraint(X, v) + 1 for v in ai])
            if m > 1:
                X = X / (m + 0.001)
        # print(objective_function(X))

        if not i % 2000:
            print(f"round {i}::")
            print("\tafter:", ' '.join([f"{v.T @ X @ v:.3}" for v in ai]))
            print("\tgradient norm:", np.linalg.norm(grad))
            print("\tdeterminant: ", np.linalg.det(X))
            print("\tobjective: ", objective_function(X))
            print("\tstep size:", step_size)



    return X


def solve(ai, tolerance = 0.001):
    m, dim = ai.shape # Dimension of the matrix X
    max_norm = np.max(np.linalg.norm(ai, axis=1))

    X = np.eye(dim) / (max_norm**2.01)  # Initial guess for X

    t = 1
    mu_factor = BARRIER_METHOD_MU_FACTOR

    while m / t > tolerance:
        print(t)
        current_objective_function = lambda X: t * objective(X) + constraints_barrier(X, ai)
        current_objective_gradient = lambda X: t * objective_gradient(X) + constraints_barrier_gradient(X, ai)
        X =  gradient_descent(objective_function=current_objective_function,
                              objective_gradient=current_objective_gradient,
                              initial_guess=X,
                              ai=ai,
                              max_iterations=None,
                              tolerance=0.001)
        t = t * mu_factor

    return X

# ]]Example usage
np.random.seed(2)
ai = np.random.normal(0, 10, 800).reshape((100, 8))


X_optimized = solve(ai)
print("Optimized matrix X:")
print(X_optimized)
print(objective(X_optimized))
eigenvals, eigenvects = np.linalg.eigh(X_optimized)
print(eigenvals)


for vector in ai:
    print(vector.T@X_optimized@vector)
