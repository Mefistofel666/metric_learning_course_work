import warnings

import numpy as np


def SDProject(M):
    eigvals, eigvecs = np.linalg.eig(M)
    eigvals = eigvals.astype(float)
    eigvecs = eigvecs.astype(float)
    eigvals[eigvals < 0.0] = 0.0
    diag_sdp = np.diag(eigvals)
    return eigvecs.dot(diag_sdp).dot(eigvecs.T)


def calc_outers(X, Y=None):
    n, d = X.shape
    if Y is None:
        Y = X
    m, e = Y.shape
    if n * m * d * e > 600_000_000:
        return None
    try:
        diff = X[:, None] - Y[None]
        return np.einsum("...i,...j->...ij", diff, diff)

    except:
        warnings.warn(
            "Memory is not enough to calculate all outer products at once. "
            "Algorithm will be slower."
        )
        outers = None

    return outers


def calc_outers_i(X, outers, i, Y=None):
    if outers is not None:
        return outers[i, :]
    else:
        n, d = X.shape
        if Y is None:
            Y = X
        m, e = Y.shape
        try:
            outers_i = np.empty([n, d, d], dtype=float)

            for j in range(m):
                outers_i[j] = np.outer(X[i, :] - Y[j, :], X[i, :] - Y[j, :])
            return outers_i
        except Exception as e:
            pass

        return None


def calc_outers_ij(X, outers_i, i, j, Y=None):
    if outers_i is not None:
        return outers_i[j]
    else:
        if Y is None:
            Y = X

        return np.outer(X[i, :] - Y[j, :], (X[i, :] - Y[j, :]).T)


def metric_sq_distance(M, x, y):
    d = (x - y).reshape(1, -1)
    return d.dot(M).dot(d.T)


def metric_to_linear(M):
    eigvals, eigvecs = np.linalg.eig(M)
    eigvals = eigvals.astype(float)
    eigvecs = eigvecs.astype(float)
    eigvals[eigvals < 0.0] = 0.0
    sqrt_diag = np.sqrt(eigvals)
    return eigvecs.dot(np.diag(sqrt_diag)).T
