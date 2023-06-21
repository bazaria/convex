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
    # print(a.T@P@a)
    return P

def constraint_objective(X,points):
    outer_products = np.array([np.outer(point,point) for point in points])
    # return np.sum(outer_products,axis=0)
    dim = ai.shape[1]
    val = np.array([a.T@X@a -1 for a in points])
    gradients = outer_products * np.broadcast_to(val[:,np.newaxis,np.newaxis],(points.shape[0],dim,dim))
    return np.sum(gradients,axis=0)


def generate_barrier_gradient(X, a):
    val = a.T @ X @ a
    if val < 1:
        return (1/ (1-a.T @ X @ a)) * np.outer(a, a)

    else:
        return np.outer(a, a)
    

def gradient_descent(ai, learning_rate=0.0001, max_iterations=6000, tolerance=1e-6):
    dim = ai.shape[1]  # Dimension of the matrix X
    m = np.max(np.linalg.norm(ai, axis=1))
    learning_rate = (m ** -4)

    X = np.eye(dim) / (m**2)  # Initial guess for X
    
    # X = np.random.uniform(-1, 1 , dim **2).reshape(dim, dim)
    
    # X = np.linalg.svd(X)[0]
    # X = X @ np.diag([1, 2, 3]) @ X.T
    print("initial:", [v.T @ X @ v for v in ai])
    print(gradient(X))
    for i in range(max_iterations):
        grad = gradient(X)

        # for a in ai:
        #     grad += generate_barrier_gradient(X, a)

        X -= learning_rate * (grad)#  + 2*constraint_objective(X,ai))

        print(f"round {i}::")

        print("\tbefore:", [v.T @ X @ v for v in ai])
        X = project_to_definite_positive(X)

        # for vector in ai:
        #     X = project_to_constraint(X, vector)

        print("\tafter:", [v.T @ X @ v for v in ai])
        
        if np.linalg.norm(grad) < tolerance:
            break

        print("\tobjective: ", objective(X))
    return X

# ]]Example usage
ai = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10], [-2 ,-5 ,6]])  # Replace with your own set of vectors
for v in ai * 0.1:
    print(np.linalg.norm(v))
X_optimized = gradient_descent(ai)
print("Optimized matrix X:")
print(X_optimized)
print(objective(X_optimized))
print(ai.shape)
# print(np.linalg.eigh(X_optimized)[1])
for vector in ai:
    print(vector.T@X_optimized@vector)
