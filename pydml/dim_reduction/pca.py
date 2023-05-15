import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MyPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.transformer = None

    def fit(self, X: np.ndarray):
        X_ = X - np.mean(X, axis=0)
        sigma = np.zeros((X_.shape[1], X_.shape[1]))
        for i in range(X_.shape[0]):
            a = X_[i].reshape((X_[i].shape[0], 1))
            sigma += a @ a.T

        w, v = np.linalg.eigh(sigma)
        L = v[: self.n_components].T[::-1]
        self.transformer = L
        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None):
        if self.transformer is None:
            self.fit(X)
        X_new = X.dot(self.transformer)
        return X_new


def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    wine = load_wine()
    X, y = wine.data, wine.target

    pca = MyPCA(n_components=2)

    pipe = Pipeline([("scaler", StandardScaler()), ("pca", pca)])
    Xt = pipe.fit_transform(X)
    plot = plt.scatter(Xt[:, 0], Xt[:, 1], c=y)
    plt.show()

    sk_pca = PCA(n_components=2)
    pipe = Pipeline([("scaler", StandardScaler()), ("pca", sk_pca)])
    Xt = pipe.fit_transform(X)
    plot = plt.scatter(Xt[:, 0], Xt[:, 1], c=y)
    plt.show()
