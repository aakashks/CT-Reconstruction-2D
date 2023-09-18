import numpy as np


class Solution:
    """
    object having all details of the equation
    """
    def __init__(self, A, b, aug_matrix, pivot_list, rank, x_p, X_n):
        self.A = A
        self.b = b
        self.aug_matrix = aug_matrix
        self.R = aug_matrix[:, :-1]
        self.d = aug_matrix[:, -1]
        self.pivot_list = pivot_list
        self.rank = rank
        self.x_particular = x_p
        self.X_nullspace = X_n


def general_soln(A, b):
    """
    reduce A | b to R | d to
    get a general solution of Ax = b where A rk(A) = rk(A|b) < min(m, n) in terms of x particular and x nullspace
    A may be rectangular matrix
    in case of full rank square (invertible) matrix x particular is the exact solution

    for free variables stored in free_variables = [x2, x4.. ] like this
    solution space will be x_partcular + x_nullspace @ free_variables

    refer Gilbert-Strang 3.4 The complete solution to Ax = b Pg 155

    Returns
    -------
    Solution

    Raises
    ------
    ValueError
        when system is inconsistent
    """
    b = b.reshape(-1, 1)
    # Augmenting both matrices
    aug_matrix = np.hstack([A, b])
    m, n = A.shape
    if m != b.shape[1]:
        print('please check shapes of A and b')
        raise ValueError

    # x_particular must have all 0s in free variables
    x_p = np.zeros([n, 1])
    # x nullspace (matrix)
    X_n = np.zeros([n, 0])

    # free columns (non-pivot columns)
    free_col = np.empty([m, 0])

    pivot_list = []

    ## make RREF
    # k will track column of pivot
    k = -1
    for i in range(m):
        # partial pivoting
        max_row = i + np.argmax(np.abs(aug_matrix[i:, i]))
        aug_matrix[[i, max_row]] = aug_matrix[[max_row, i]]

        k += 1
        pivot = aug_matrix[i, k]

        while pivot == 0 and k < n:
            # corresopnding free variable be 1 rest 0
            id_col = np.zeros([n, 1])
            id_col[k, 0] = 1
            X_n = np.hstack([X_n, id_col])

            # store free columns
            free_col = np.hstack([free_col, -aug_matrix[:, k].reshape(-1, 1)])
            k += 1

            # if this row has no pivot (all 0 elements)
            if k == n:
                break
            pivot = aug_matrix[i, k]

        if k < n:
            # make piot 1
            aug_matrix[i, :] /= pivot

            for j in range(0, m):
                # make all elements 0 in pivot row (RREF)
                if j == i:
                    continue
                factor = aug_matrix[j, k]
                aug_matrix[j, :] -= factor * aug_matrix[i, :]

            pivot_list.append([i, k])

        else:
            rank = i
            break
    else:
        # full row rank
        rank = m

    # check if rk(A|b) > rk(A)
    if aug_matrix[rank:, n].any():
        print('inconsistent system!!')
        raise ValueError

    ctr = 0
    for i, k in pivot_list:
        # element in d corresponding to pivot element
        x_p[k, 0] = aug_matrix[i, n]
        # fill value in pivot variables
        X_n[k, :] = free_col[ctr, :]
        ctr += 1

    soln = Solution(A, b, aug_matrix, pivot_list, rank, x_p, X_n)

    return soln
