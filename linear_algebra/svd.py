import numpy as np
from tqdm import tqdm


def power_iteration(A, num_iterations=1000, tol=1e-8):
    """
    power iterations to find out 1 eigenvector of given matrix
    """
    np.random.seed(42)
    b = np.random.rand(A.shape[1])

    for _ in range(num_iterations):
        Ab = np.dot(A, b)

        # early stopping if we know Ab ~ 0, ie. eigv is close to 0
        # if np.allclose(Ab, 0, atol=tol):
        #     return 0

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

    # finding right singular vectors
    for _ in tqdm(range(min(ATA.shape)), desc='Performing decomposition'):
        eigenvector = power_iteration(ATA, num_iterations, tol).reshape(-1, 1)
        eigenvalue = eigenvector.T @ A @ eigenvector

        if eigenvalue == np.nan:
            raise ValueError('Null encountered!')

        if abs(eigenvalue) < tol:
            break

        eigenvector = eigenvector / np.linalg.norm(eigenvector)

        V.append(eigenvector)
        S.append(eigenvalue)

        # reduce lambda * vT v from the eigendecomposition of AtA
        ATA = ATA - eigenvalue * np.outer(eigenvector, eigenvector)

    S = np.sqrt(S)

    # find the left singular vectors
    U = []
    for i in range(len(S)):
        sigma_i = S[i]
        vi = V[i]
        ui = np.dot(A, vi) / sigma_i
        U.append(ui)

    U = np.column_stack(U)
    Vt = np.column_stack(V).T

    return U, S, Vt


def pinv(A, num_iterations=1000, tol=1e-8):
    """
    return pseudo inverse of the matrix A
    """
    U, S, Vt = svd(A, num_iterations, tol)
    return np.linalg.multi_dot([Vt.T, np.diag(1/S), U.T])