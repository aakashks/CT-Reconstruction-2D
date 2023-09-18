import numpy as np


class Solution:
    """
    object having all properties of the solved equation
    """
    def __init__(self, A, b, aug_matrix, pivot_list, rank, x_p, X_n, k):
        self.A = A
        self.m, self.n = A.shape
        self.b = b
        self.aug_matrix = aug_matrix
        self.R = aug_matrix[:, :-1]
        self.d = aug_matrix[:, -1]
        self.pivot_list = pivot_list
        self.rank = rank
        self.x_particular = x_p
        self.X_nullspace = X_n
        self.k = k


def general_soln(A, b, tol=1e-6):
    """
    reduce A | b to R | d to
    get a general solution of Ax = b where A rk(A) = rk(A|b) < min(m, n) in terms of x particular and x nullspace
    A may be rectangular matrix
    in case of full rank square (invertible) matrix x particular is the exact solution

    for free variables stored in free_variables = [x2, x4.. ] like this
    solution space will be x_partcular + x_nullspace @ free_variables

    Returns
    -------
    Solution

    Raises
    ------
    ValueError
        when system is inconsistent

    References
    ----------
    G. Strang 3.4-The complete solution to Ax = b p.g. 155.
    """
    b = b.reshape(-1, 1)
    # Augmenting both matrices
    aug_matrix = np.hstack([A, b]).astype('float64')
    m, n = A.shape
    if m != b.shape[0]:
        raise ValueError('wrong shapes of A and b')

    # x_particular must have all 0s in free variables
    x_p = np.zeros([n, 1])
    # x nullspace (matrix)
    X_n = np.zeros([n, 0])

    # free columns (non-pivot columns)
    free_cols = np.empty([m, 0])

    pivot_list = []

    ## make RREF
    # k will track column of pivot
    k = -1
    for i in range(m):
        k += 1
        # partial pivoting
        max_row = i + np.argmax(np.abs(aug_matrix[i:, k]))
        if max_row != i:
            aug_matrix[[i, max_row]] = aug_matrix[[max_row, i]]

        pivot = aug_matrix[i, k]

        while abs(pivot) < tol and k < n:
            # corresopnding free variable be 1 rest 0
            id_col = np.zeros([n, 1])
            id_col[k, 0] = 1
            X_n = np.hstack([X_n, id_col])

            # store free columns
            free_cols = np.hstack([free_cols, -aug_matrix[:, k].reshape(-1, 1)])
            k += 1

            # if this row has no pivot (all 0 elements)
            if k == n:
                break

            # partial pivoting
            max_row = i + np.argmax(np.abs(aug_matrix[i:, k]))
            if max_row != i:
                aug_matrix[[i, max_row]] = aug_matrix[[max_row, i]]

            pivot = aug_matrix[i, k]

        if k < n:
            # make piot 1
            aug_matrix[i, :] = aug_matrix[i, :] / pivot
            # make all elements 0 in pivot column (RREF)
            piv_row = aug_matrix[i, :].reshape(-1, 1)
            factors = aug_matrix[:, k].reshape(-1, 1)
            aug_matrix = aug_matrix - factors @ piv_row.T
            aug_matrix[i, :] = piv_row.reshape(-1)
            pivot_list.append([i, k])

        else:
            rank = i
            break
    else:
        # full row rank
        rank = m

    # add remaining cols (in r=m < n case)
    for j in range(k + 1, n):
        free_cols = np.hstack([free_cols, -aug_matrix[:, j].reshape(-1, 1)])
        id_col = np.zeros([n, 1])
        id_col[j, 0] = 1
        X_n = np.hstack([X_n, id_col])

    # check if rk(A|b) > rk(A)
    if (np.abs(aug_matrix[rank:, n]) >= tol).any():
        raise ValueError('inconsistent system!!')

    ctr = 0
    for i, j in pivot_list:
        # element in d corresponding to pivot element
        x_p[j, 0] = aug_matrix[i, n]
        # fill value in pivot variables
        X_n[j, :] = free_cols[ctr, :]
        ctr += 1

    soln = Solution(A, b, aug_matrix, pivot_list, rank, x_p, X_n, k)

    return soln
