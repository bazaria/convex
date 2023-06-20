import numpy as np

def solve(A: np.array, order = '+'):

    print(np.linalg.matrix_rank(A))
    orderer = (np.argmax, max) if order == '+' else (np.argmin, min)

    base = []
    eigenvalues = []

    vector_index = np.argmax(np.linalg.norm(A, axis=1))
    vector = A[vector_index]
    eigenvalues.append(1 / np.linalg.norm(vector))
    base.append(vector * eigenvalues[-1])
    A = np.delete(A, vector_index, 0)

    while True:
        new_norms = {}
        print(base)
        for i, row in enumerate(A):
            gram_schmidt = row - sum(np.dot(b, row) * b for b in base)
            if np.linalg.norm(gram_schmidt) < 0.0001:
                A = np.delete(A, i, 0)
                break
            normalized_gram_schmidt = gram_schmidt / np.linalg.norm(gram_schmidt)
            new_vector = sum(eigen * np.dot(b, row) * b for eigen, b in zip(eigenvalues, base))
            print(new_vector, np.linalg.norm(new_vector))
            size_of_other_coordinates = 1 - np.linalg.norm(new_vector) ** 2
            gram_schmidt_coordinate = np.dot(row, normalized_gram_schmidt)
            eigen_value = np.sqrt(size_of_other_coordinates / (gram_schmidt_coordinate **2))

            new_norms[i] = (eigen_value, normalized_gram_schmidt)

        if not A.size:
            break

        vector_index = max(new_norms, key=lambda x: new_norms[x][0])
        eigenvalues.append(new_norms[vector_index][0])
        base.append(new_norms[vector_index][1])
        A = np.delete(A, vector_index, 0)


    print(base)
    print(eigenvalues)

if __name__ == '__main__':
    solve(np.array([[4, 5],[1,2]]))