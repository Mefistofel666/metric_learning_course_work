import numpy as np


def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


def test_pca():
    n = 4
    d = 2
    # prepare test matrix X
    X = np.random.random([n, n])
    X = symmetrize(X)

    # calc eigenvaleus and eigenvectors with norm 1
    w, v = np.linalg.eigh(X)
    print(f"Eigenvalues: {w}")
    print(f"Eigenvectors: {v}")

    # best d eigenvalues and eigenvectors
    best_w = w[-d:]
    print(f"Best eigenvalues: {best_w}")
    L = v[:, -d:]
    print(f"Best eigenvectors: {L}")
    M = L.T @ L
    print(f"Mahalanobis Matrix: {M}")
    assert L.shape == (n, d)
    print(M.shape)
    print(L.shape)
