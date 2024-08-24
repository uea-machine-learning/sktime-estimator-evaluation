from typing import Dict, Optional

import numpy as np
from aeon.clustering.base import BaseClusterer
from aeon.distances import pairwise_distance
from sklearn.cluster import DBSCAN


class TimeSeriesDBScan(BaseClusterer):
    """Perform TimeSeriesDBSCAN clustering from vector array or distance matrix.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    The worst case memory complexity of DBSCAN is :math:`O({n}^2)`, which can
    occur when the `eps` param is large and `min_samples` is low.

    Read more in the :ref:`User Guide <dbscan>`.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.
    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point to
        be considered as a core point. This includes the point itself. If
        `min_samples` is set to a higher value, DBSCAN will find denser clusters,
        whereas if it is set to a lower value, the found clusters will be more
        sparse.
    distance : str or Callable, default='euclidean'
        Distance metric to compute similarity between time series. A list of valid
        strings for metrics can be found in the documentation for
        :func:`aeon.distances.get_distance_function`. If a callable is passed it must be
        a function that takes two 2d numpy arrays as input and returns a float.
        If you are using precomputed distance, you should still specify the distance
        so that it is written to the results file
    distance_params : dict, default=None
        Dictionary containing kwargs for the distance being used. For example if you
        wanted to specify a window for DTW you would pass
        distance_params={"window": 0.2}. See documentation of aeon.distances for more
        details.
    precomputed_distances : ndarray of shape (n_samples, n_samples), default=None
        A pairwise distance matrix where `precomputed_distances[i, j]` is the distance
        between samples `X[i]` and `X[j]`. If None, the distance matrix is computed
        using the distance metric specified by `distance`.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.
    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.
    p : float, default=None
        The power of the Minkowski metric to be used to calculate distance
        between points. If None, then ``p=2`` (equivalent to the Euclidean
        distance).
    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.
    components_ : ndarray of shape (n_core_samples, n_features)
        Copy of each core sample found by training.
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        distance: str = "euclidean",
        distance_params: Optional[Dict] = None,
        precomputed_distances: Optional[np.ndarray] = None,
        algorithm: str = "auto",
        leaf_size: int = 30,
        n_jobs: Optional[int] = None,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.distance = distance
        self.distance_params = distance_params

        # Set to None so the values dont get written to a file
        self.precomputed_distances = None
        self._precomputed_distances = precomputed_distances

        self._model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="precomputed",
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )

        self.core_sample_indices_ = None
        self.components_ = None
        self.labels_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

        super().__init__()

    def _fit(self, X, y=None):
        distance_params = {}
        if self.distance_params is not None:
            distance_params = self.distance_params

        if self._precomputed_distances is None:
            self._precomputed_distances = pairwise_distance(
                X, metric=self.distance, **distance_params
            )

        self._model.fit(self._precomputed_distances)

        self.core_sample_indices_ = self._model.core_sample_indices_
        self.components_ = self._model.components_
        self.labels_ = self._model.labels_
        if hasattr(self._model, "n_features_in_"):
            self.n_features_in_ = self._model.n_features_in_
        if hasattr(self._model, "feature_names_in_"):
            self.feature_names_in_ = self._model.feature_names_in_

    def fit_predict(self, X, y=None) -> np.ndarray:
        self.fit(X, y)
        return self.labels_

    def _predict(self, X, y=None) -> np.ndarray:
        raise RuntimeError("DBSCAN does not support predict method")

    def _score(self, X, y=None):
        return 0.0
