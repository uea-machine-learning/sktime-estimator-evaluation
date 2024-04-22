from typing import Dict, Optional, Union
import numpy as np
from sklearn.cluster import HDBSCAN
from aeon.clustering.base import BaseClusterer
from aeon.distances import pairwise_distance


class TimeSeriesHDBScan(BaseClusterer):
    """Cluster data using hierarchical density-based clustering.

    HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications
    with Noise. Performs :class:`~sklearn.cluster.DBSCAN` over varying epsilon
    values and integrates the result to find a clustering that gives the best
    stability over epsilon.
    This allows HDBSCAN to find clusters of varying densities (unlike
    :class:`~sklearn.cluster.DBSCAN`), and be more robust to parameter selection.
    Read more in the :ref:`User Guide <hdbscan>`.

    Parameters
    ----------
    min_cluster_size : int, default=5
        The minimum number of samples in a group for that group to be
        considered a cluster; groupings smaller than this size will be left
        as noise.
    min_samples : int, default=None
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        When `None`, defaults to `min_cluster_size`.
    cluster_selection_epsilon : float, default=0.0
        A distance threshold. Clusters below this value will be merged.
        See [5]_ for more information.
    max_cluster_size : int, default=None
        A limit to the size of clusters returned by the `"eom"` cluster
        selection algorithm. There is no limit when `max_cluster_size=None`.
        Has no effect if `cluster_selection_method="leaf"`.
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
    alpha : float, default=1.0
        A distance scaling parameter as used in robust single linkage.
        See [3]_ for more information.
    algorithm : {"auto", "brute", "kd_tree", "ball_tree"}, default="auto"
        Exactly which algorithm to use for computing core distances; By default
        this is set to `"auto"` which attempts to use a
        :class:`~sklearn.neighbors.KDTree` tree if possible, otherwise it uses
        a :class:`~sklearn.neighbors.BallTree` tree. Both `"kd_tree"` and
        `"ball_tree"` algorithms use the
        :class:`~sklearn.neighbors.NearestNeighbors` estimator.
        If the `X` passed during `fit` is sparse or `metric` is invalid for
        both :class:`~sklearn.neighbors.KDTree` and
        :class:`~sklearn.neighbors.BallTree`, then it resolves to use the
        `"brute"` algorithm.
    leaf_size : int, default=40
        Leaf size for trees responsible for fast nearest neighbour queries when
        a KDTree or a BallTree are used as core-distance algorithms. A large
        dataset size and small `leaf_size` may induce excessive memory usage.
        If you are running out of memory consider increasing the `leaf_size`
        parameter. Ignored for `algorithm="brute"`.
    n_jobs : int, default=None
        Number of jobs to run in parallel to calculate distances.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    cluster_selection_method : {"eom", "leaf"}, default="eom"
        The method used to select clusters from the condensed tree. The
        standard approach for HDBSCAN* is to use an Excess of Mass (`"eom"`)
        algorithm to find the most persistent clusters. Alternatively you can
        instead select the clusters at the leaves of the tree -- this provides
        the most fine grained and homogeneous clusters.
    allow_single_cluster : bool, default=False
        By default HDBSCAN* will not produce a single cluster, setting this
        to True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.
    store_centers : str, default=None
        Which, if any, cluster centers to compute and store. The options are:
        - `None` which does not compute nor store any centers.
        - `"centroid"` which calculates the center by taking the weighted
          average of their positions. Note that the algorithm uses the
          euclidean metric and does not guarantee that the output will be
          an observed data point.
        - `"medoid"` which calculates the center by taking the point in the
          fitted data which minimizes the distance to all other points in
          the cluster. This is slower than "centroid" since it requires
          computing additional pairwise distances between points of the
          same cluster but guarantees the output is an observed data point.
          The medoid is also well-defined for arbitrary metrics, and does not
          depend on a euclidean metric.
        - `"both"` which computes and stores both forms of centers.
    copy : bool, default=False
        If `copy=True` then any time an in-place modifications would be made
        that would overwrite data passed to :term:`fit`, a copy will first be
        made, guaranteeing that the original data will be unchanged.
        Currently, it only applies when `metric="precomputed"`, when passing
        a dense array or a CSR sparse matrix and when `algorithm="brute"`.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to :term:`fit`.
        Outliers are labeled as follows:
        - Noisy samples are given the label -1.
        - Samples with infinite elements (+/- np.inf) are given the label -2.
        - Samples with missing data are given the label -3, even if they
          also have infinite elements.
    probabilities_ : ndarray of shape (n_samples,)
        The strength with which each sample is a member of its assigned
        cluster.
        - Clustered samples have probabilities proportional to the degree that
          they persist as part of the cluster.
        - Noisy samples have probability zero.
        - Samples with infinite elements (+/- np.inf) have probability 0.
        - Samples with missing data have probability `np.nan`.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    centroids_ : ndarray of shape (n_clusters, n_features)
        A collection containing the centroid of each cluster calculated under
        the standard euclidean metric. The centroids may fall "outside" their
        respective clusters if the clusters themselves are non-convex.
        Note that `n_clusters` only counts non-outlier clusters. That is to
        say, the `-1, -2, -3` labels for the outlier clusters are excluded.
    medoids_ : ndarray of shape (n_clusters, n_features)
        A collection containing the medoid of each cluster calculated under
        the whichever metric was passed to the `metric` parameter. The
        medoids are points in the original cluster which minimize the average
        distance to all other points in that cluster under the chosen metric.
        These can be thought of as the result of projecting the `metric`-based
        centroid back onto the cluster.
        Note that `n_clusters` only counts non-outlier clusters. That is to
        say, the `-1, -2, -3` labels for the outlier clusters are excluded.
    """

    def __init__(
            self,
            min_cluster_size: int = 5,
            min_samples: Optional[int] = None,
            cluster_selection_epsilon: float = 0.0,
            max_cluster_size: Optional[int] = None,
            distance: str = 'euclidean',
            distance_params: Dict = None,
            precomputed_distances: Optional[np.ndarray] = None,
            alpha: float = 1.0,
            algorithm: str = 'auto',
            leaf_size: int = 40,
            n_jobs: Optional[int] = None,
            cluster_selection_method: str = "eom",
            allow_single_cluster: bool = False,
            store_centers: Optional[str] = None,
            copy: bool = False
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size
        self.distance = distance
        self.distance_params = distance_params
        self.alpha = alpha
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.store_centers = store_centers
        self.copy = copy

        # Set to None so the values dont get written to a file
        self.precomputed_distances = None
        self._precomputed_distances = precomputed_distances

        self._model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            max_cluster_size=self.max_cluster_size,
            metric="precomputed",
            alpha=self.alpha,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
            store_centers=store_centers,
            copy=self.copy
        )

        self.labels_ = None
        self.probabilities_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.centroids_ = None
        self.medoids_ = None

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

        if hasattr(self._model, 'labels_'):
            self.labels_ = self._model.labels_
        if hasattr(self._model, 'probabilities_'):
            self.probabilities_ = self._model.probabilities_
        if hasattr(self._model, 'n_features_in_'):
            self.n_features_in_ = self._model.n_features_in_
        if hasattr(self._model, 'feature_names_in_'):
            self.feature_names_in_ = self._model.feature_names_in_
        if hasattr(self._model, 'cluster_centroids_'):
            self.centroids_ = self._model.cluster_centroids_
        if hasattr(self._model, 'cluster_medoids_'):
            self.medoids_ = self._model.cluster_medoids_

    def fit_predict(self, X, y=None) -> np.ndarray:
        self.fit(X, y)
        return self.labels_

    def _predict(self, X, y=None) -> np.ndarray:
        raise RuntimeError("HDBSCAN does not support predict method")

    def _score(self, X, y=None):
        return 0.
