import numpy as np
from aeon.clustering import BaseClusterer
from sklearn.cluster import Birch

# COME BACK TO THIS


class TimeSeriesBirch(BaseClusterer):

    def __init__(
        self,
        n_clusters: int = 3,
        threshold: float = 0.5,
        branching_factor: int = 50,
    ):
        raise NotImplementedError("Birch is not implemented yet.")
        self.threshold = threshold
        self.branching_factor = branching_factor

        self._model = Birch(
            n_clusters=n_clusters,
            threshold=threshold,
            branching_factor=branching_factor,
        )

        self.root_ = None
        self.dummy_leaf_ = None
        self.subcluster_centers_ = None
        self.subcluster_labels_ = None
        self.labels_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

        super().__init__(n_clusters=n_clusters)

    def _fit(self, X, y=None):
        if self._model.n_clusters != self.n_clusters:
            self._model.set_params(n_clusters=self.n_clusters)

        self._model.fit(X)

        self.root_ = self._model.root_
        self.dummy_leaf_ = self._model.dummy_leaf_
        self.subcluster_centers_ = self._model.subcluster_centers_
        self.subcluster_labels_ = self._model.subcluster_labels_
        self.labels_ = self._model.labels_
        if hasattr(self._model, "feature_names_in_"):
            self.feature_names_in_ = self._model.feature_names_in_
        if hasattr(self._model, "n_features_in_"):
            self.n_features_in_ = self._model.n_features_in_

    def fit_predict(self, X, y=None) -> np.ndarray:
        self.fit(X, y)
        return self.labels_

    def _predict(self, X, y=None) -> np.ndarray:
        return self._model.predict(X)

    def _score(self, X, y=None):
        return 0.0
