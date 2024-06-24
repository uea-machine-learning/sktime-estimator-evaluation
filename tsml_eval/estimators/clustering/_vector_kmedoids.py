import pandas as pd
from aeon.utils.validation._dependencies import _check_soft_dependencies
import numpy as np

_check_soft_dependencies("kmedoids")
from kmedoids import KMedoids


class VectorKmedoids(KMedoids):

    def __init__(
            self,
            n_clusters,
            *,
            metric="precomputed",
            metric_params=None,
            method="fasterpam",
            init="random",
            max_iter=300,
            random_state=None,
    ):
        if "kmedoids" in method:
            method = method.replace("kmedoids", "alternate")
        super().__init__(
            n_clusters=n_clusters,
            metric=metric,
            metric_params=metric_params,
            method=method,
            init=init,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        super().fit(X)

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return super().predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 2D array of shape [n_cases, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        # Preds for kmedoids are the index value (e.g. [22, 38, 22, 38,..]) rather than
        # a cluster index (e.g. [0, 1, 0, 1,..]). We need to convert the former to the
        # latter to be consistent with the rest of the clusterers.
        preds = self.predict(X)
        unique = np.unique(preds)
        for i, u in enumerate(unique):
            preds[preds == u] = i
        n_cases = len(preds)
        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = int(max(preds)) + 1
        dists = np.zeros((X.shape[0], n_clusters))
        for i in range(n_cases):
            dists[i, preds[i]] = 1
        return dists
