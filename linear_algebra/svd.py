import numpy as np
from tqdm import tqdm


def power_iteration(A, num_iterations):
    """
    power iterations to find out 1 eigenvector of given matrix
    """
    b = np.random.rand(A.shape[1])

    for _ in range(num_iterations):
        Ab = np.dot(A, b)
        norm = np.linalg.norm(Ab)
        b = Ab / norm

    return b


def svd(A, num_iterations=1000, tol=1e-8):
    """
    return SVD of matrix A, S will be of (r,r) dimensions where r is rank of matrix A
    equivalent to numpy's np.linalg.svd(A, full_natrices=False) with tolerance for singular values
    """
    ATA = np.dot(A.T, A)

    V = []
    S = []

    # for reproducibility of code results
    np.random.seed(42)

    # finding right singular vectors
    for _ in tqdm(range(min(ATA.shape)), desc='Performing decomposition'):
        eigenvector = power_iteration(ATA, num_iterations)
        eigenvalue = np.dot(np.dot(eigenvector, ATA), eigenvector)

        if eigenvalue < tol:
            break

        eigenvector /= np.linalg.norm(eigenvector)

        V.append(eigenvector)
        S.append(eigenvalue)

        # reduce lambda * vT v from the eigendecomposition of AtA
        ATA = ATA - eigenvalue * np.outer(eigenvector, eigenvector)

    if not S:
        raise ValueError('Matrix has rank 0')

    S = np.sqrt(S)

    # find the left singular vectors
    V = np.column_stack(V)
    U = np.multiply(np.dot(A, V), 1/S)
    return U, S, V.T


def pinv(A, num_iterations=1000, tol=1e-8):
    """
    return pseudo inverse of the matrix A
    """
    U, S, Vt = svd(A, num_iterations, tol)
    return np.linalg.multi_dot([Vt.T, np.diag(1/S), U.T])
