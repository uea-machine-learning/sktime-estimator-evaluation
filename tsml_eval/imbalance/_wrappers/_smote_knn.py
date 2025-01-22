import numpy as np
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.distances import get_distance_function, pairwise_distance
from aeon.utils.validation import check_n_jobs


class _SmoteKNN(KNeighborsTimeSeriesClassifier):
    """K-Nearest Neighbors Time Series Classifier for SMOTE.

    This is different from the KNeighborsTimeSeriesClassifier in that it doesn't
    require y to be passed in the fit method. This is because the class is used
    in the SMOTE class which doesn't require y to be passed in the fit method.

    See the docs for the aeon KNN classifier for more information on the parameters.
    """

    def fit(self, X, y=None):
        X = self._preprocess_collection(X)
        self._fit(X, y)

    def _fit(self, X, y):
        self.metric_ = get_distance_function(method=self.distance)
        self.X_ = X
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        if self.classes_ is None:
            raise ValueError(
                "Classes have not been set. Please fit the model first "
                "with labels before trying to predict"
            )
        return super().predict(X)

    def kneighbors(self, X, return_distance=True):
        distance_matrix = pairwise_distance(X, method=self.distance)
        indices = np.argsort(distance_matrix, axis=1)[:, : self.n_neighbors]
        if return_distance:
            row_ids = np.arange(distance_matrix.shape[0])[:, None]
            dist_k = distance_matrix[row_ids, indices]
            return dist_k, indices
        else:
            return indices

    def _preprocess_collection(self, X, store_metadata=True):
        if isinstance(X, list) and isinstance(X[0], np.ndarray):
            X = self._reshape_np_list(X)
        meta = self._check_X(X)
        if len(self.metadata_) == 0 and store_metadata:
            self.metadata_ = meta

        X = self._convert_X(X)
        multithread = self.get_tag("capability:multithreading")
        if multithread:
            if hasattr(self, "n_jobs"):
                self._n_jobs = check_n_jobs(self.n_jobs)
            else:
                raise AttributeError(
                    "self.n_jobs must be set if capability:multithreading is True"
                )
        return X
