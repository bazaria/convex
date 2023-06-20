import numpy as np



def objective(X):
    return -np.log(np.linalg.det(X))

def gradient(X):
    return -np.linalg.inv(X).T

def constraint(X, ai):
    return np.dot(ai.T, np.dot(X, ai)) - 1.0

project_epsilon = 0.01
def project_to_definite_positive(X):
    eigenvals, eigenvects = np.linalg.eigh(X)
    eigenvals = np.maximum(project_epsilon, eigenvals)
    projected_X = np.dot(np.dot(eigenvects, np.diag(eigenvals)), eigenvects.T)
    return projected_X

def project_to_constraint(X, a):
    val = a.T @ X @ a
    if val <= 1:
        return X

    norm = np.linalg.norm(a)
    k = (val - 1) / (norm ** 4)
    P = X - k * np.outer(a, a)
    print(a.T@P@a)
    return P

def gradient_descent(ai, learning_rate=0.1, max_iterations=1000, tolerance=1e-6):
    dim = ai.shape[1]  # Dimension of the matrix X
    X = np.eye(dim)  # Initial guess for X

    X = np.random.uniform(-1, 1 , dim **2).reshape(dim, dim)
    
    X = np.linalg.svd(X)[0]
    X = X @ np.diag([1, 2, 3]) @ X.T

    for i in range(max_iterations):
        grad = gradient(X)
        X -= learning_rate * grad


        for vector in ai:
            X = project_to_constraint(X, vector)

        X = project_to_definite_positive(X)

        if np.linalg.norm(grad) < tolerance:
            break

        # print(np.linalg.det(X))
    return X

# ]]Example usage
ai = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10], [-2 ,-5 ,6]])  # Replace with your own set of vectors
X_optimized = gradient_descent(ai*0.1)
print("Optimized matrix X:")
print(X_optimized)
print(np.linalg.det(X_optimized))
print(ai.shape)
# print(np.linalg.eigh(X_optimized)[1])
for vector in ai:
    print(vector.T@X_optimized@vector)