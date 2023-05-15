from numpy.linalg import cholesky
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from pydml.utils import metric_to_linear


class DML_Algorithm(BaseEstimator, TransformerMixin):
    def __init__(self):
        raise NotImplementedError(
            "Class DML_Algorithm is abstract and cannot be instantiated."
        )

    def metric(self):
        if hasattr(self, "M_"):
            return self.M_
        else:
            if hasattr(self, "L_"):
                L = self.transformer()
                M = L.T.dot(L)
                return M
            else:
                raise NameError("Metric was not defined. Algorithm was not fitted.")

    def transformer(self):
        if hasattr(self, "L_"):
            return self.L_
        else:
            if hasattr(self, "M_"):
                try:
                    L = cholesky(self.metric()).T
                    return L
                except:
                    L = metric_to_linear(self.metric())
                    return L
                # self.L_ = L
                return L
            else:
                raise NameError(
                    "Transformer was not defined. Algorithm was not fitted."
                )

    def transform(self, X=None):
        if X is None:
            X = self.X_
        else:
            X = check_array(X, accept_sparse=True)
        L = self.transformer()
        return X.dot(L.T)

    def metadata(self):
        return {}
