import numpy as np


def diagonalize(matrix):
    """
    Diagonalize a square matrix A such that A = P @ D @ P_inv,
    where D is diagonal and P is the matrix of eigenvectors.

    Parameters
    ----------
    matrix : array-like
        A square (n x n) matrix.

    Returns
    -------
    P : numpy.ndarray
        Matrix whose columns are the eigenvectors of A.
    D : numpy.ndarray
        Diagonal matrix of eigenvalues.
    P_inv : numpy.ndarray
        Inverse of P.

    Raises
    ------
    ValueError
        If the matrix is not square or is not diagonalizable.
    """
    A = np.array(matrix, dtype=complex)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")

    eigenvalues, eigenvectors = np.linalg.eig(A)

    if np.linalg.matrix_rank(eigenvectors) < A.shape[0]:
        raise ValueError("Matrix is not diagonalizable (eigenvectors are linearly dependent).")

    P = eigenvectors
    D = np.diag(eigenvalues)
    P_inv = np.linalg.inv(P)

    return P, D, P_inv


if __name__ == "__main__":
    A = np.array([[4, 1], [2, 3]])
    P, D, P_inv = diagonalize(A)

    print("Original matrix A:")
    print(A)
    print("\nEigenvector matrix P:")
    print(P)
    print("\nDiagonal matrix D:")
    print(D)
    print("\nVerification A == P @ D @ P_inv:")
    print(np.allclose(A, P @ D @ P_inv))
