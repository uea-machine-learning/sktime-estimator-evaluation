from typing import Dict, Optional, Union

import numpy as np
from aeon.clustering.base import BaseClusterer
from aeon.distances import pairwise_distance
from sklearn.cluster import OPTICS


class TimeSeriesOPTICS(BaseClusterer):
    """Estimate clustering structure from vector array.

    OPTICS (Ordering Points To Identify the Clustering Structure), closely
    related to DBSCAN, finds core sample of high density and expands clusters
    from them [1]_. Unlike DBSCAN, keeps cluster hierarchy for a variable
    neighborhood radius. Better suited for usage on large datasets than the
    current sklearn implementation of DBSCAN.

    Clusters are then extracted using a DBSCAN-like method
    (cluster_method = 'dbscan') or an automatic
    technique proposed in [1]_ (cluster_method = 'xi').

    This implementation deviates from the original OPTICS by first performing
    k-nearest-neighborhood searches on all points to identify core sizes, then
    computing only the distances to unprocessed points when constructing the
    cluster order. Note that we do not employ a heap to manage the expansion
    candidates, so the time complexity will be O(n^2).

    Read more in the :ref:`User Guide <optics>`.

    Parameters
    ----------
    min_samples : int > 1 or float between 0 and 1, default=5
        The number of samples in a neighborhood for a point to be considered as
        a core point. Also, up and down steep regions can't have more than
        ``min_samples`` consecutive non-steep points. Expressed as an absolute
        number or a fraction of the number of samples (rounded to be at least
        2).
    max_eps : float, default=np.inf
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Default value of ``np.inf`` will
        identify clusters across all scales; reducing ``max_eps`` will result
        in shorter run times.
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
    cluster_method : str, default='xi'
        The extraction method used to extract clusters using the calculated
        reachability and ordering. Possible values are "xi" and "dbscan".
    eps : float, default=None
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. By default it assumes the same value
        as ``max_eps``.
        Used only when ``cluster_method='dbscan'``.
    xi : float between 0 and 1, default=0.05
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.
        Used only when ``cluster_method='xi'``.
    predecessor_correction : bool, default=True
        Correct clusters according to the predecessors calculated by OPTICS
        [2]_. This parameter has minimal effect on most datasets.
        Used only when ``cluster_method='xi'``.
    min_cluster_size : int > 1 or float between 0 and 1, default=None
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded to be
        at least 2). If ``None``, the value of ``min_samples`` is used instead.
        Used only when ``cluster_method='xi'``.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use :class:`~sklearn.neighbors.BallTree`.
        - 'kd_tree' will use :class:`~sklearn.neighbors.KDTree`.
        - 'brute' will use a brute-force search.
        - 'auto' (default) will attempt to decide the most appropriate
          algorithm based on the values passed to :meth:`fit` method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    leaf_size : int, default=30
        Leaf size passed to :class:`~sklearn.neighbors.BallTree` or
        :class:`~sklearn.neighbors.KDTree`. This can affect the speed of the
        construction and query, as well as the memory required to store the
        tree. The optimal value depends on the nature of the problem.
    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples and points which are not included in a leaf cluster
        of ``cluster_hierarchy_`` are labeled as -1.
    reachability_ : ndarray of shape (n_samples,)
        Reachability distances per sample, indexed by object order. Use
        ``clust.reachability_[clust.ordering_]`` to access in cluster order.
    ordering_ : ndarray of shape (n_samples,)
        The cluster ordered list of sample indices.
    core_distances_ : ndarray of shape (n_samples,)
        Distance at which each sample becomes a core point, indexed by object
        order. Points which will never be core have a distance of inf. Use
        ``clust.core_distances_[clust.ordering_]`` to access in cluster order.
    predecessor_ : ndarray of shape (n_samples,)
        Point that a sample was reached from, indexed by object order.
        Seed points have a predecessor of -1.
    cluster_hierarchy_ : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of ``[start, end]`` in each row, with
        all indices inclusive. The clusters are ordered according to
        ``(end, -start)`` (ascending) so that larger clusters encompassing
        smaller clusters come after those smaller ones. Since ``labels_`` does
        not reflect the hierarchy, usually
        ``len(cluster_hierarchy_) > np.unique(optics.labels_)``. Please also
        note that these indices are of the ``ordering_``, i.e.
        ``X[ordering_][start:end + 1]`` form a cluster.
        Only available when ``cluster_method='xi'``.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    """

    def __init__(
        self,
        min_samples: Union[int, float] = 5,
        max_eps: float = np.inf,
        distance: str = "euclidean",
        distance_params: Optional[Dict] = None,
        precomputed_distances: Optional[np.ndarray] = None,
        cluster_method: str = "xi",
        eps: Optional[float] = None,
        xi: float = 0.05,
        predecessor_correction: bool = True,
        min_cluster_size: Optional[Union[int, float]] = None,
        algorithm: str = "auto",
        leaf_size: int = 30,
        memory: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ):
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.distance = distance
        self.distance_params = distance_params
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.memory = memory
        self.n_jobs = n_jobs

        # Set to None so the values dont get written to a file
        self.precomputed_distances = None
        self._precomputed_distances = precomputed_distances

        self._model = OPTICS(
            min_samples=self.min_samples,
            max_eps=self.max_eps,
            metric="precomputed",
            cluster_method=self.cluster_method,
            eps=self.eps,
            xi=self.xi,
            predecessor_correction=self.predecessor_correction,
            min_cluster_size=self.min_cluster_size,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            memory=self.memory,
            n_jobs=self.n_jobs,
        )

        self.labels_ = None
        self.reachability_ = None
        self.ordering_ = None
        self.core_distances_ = None
        self.predecessor_ = None
        self.cluster_hierarchy_ = None
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

        if hasattr(self._model, "labels_"):
            self.labels_ = self._model.labels_
        if hasattr(self._model, "reachability_"):
            self.reachability_ = self._model.reachability_
        if hasattr(self._model, "ordering_"):
            self.ordering_ = self._model.ordering_
        if hasattr(self._model, "core_distances_"):
            self.core_distances_ = self._model.core_distances_
        if hasattr(self._model, "predecessor_"):
            self.predecessor_ = self._model.predecessor_
        if hasattr(self._model, "cluster_hierarchy_"):
            self.cluster_hierarchy_ = self._model.cluster_hierarchy_
        if hasattr(self._model, "n_features_in_"):
            self.n_features_in_ = self._model.n_features_in_
        if hasattr(self._model, "feature_names_in_"):
            self.feature_names_in_ = self._model.feature_names_in_

    def fit_predict(self, X, y=None) -> np.ndarray:
        self.fit(X, y)
        return self.labels_

    def _predict(self, X, y=None) -> np.ndarray:
        raise RuntimeError("OPTICS does not support predict method")

    def _score(self, X, y=None):
        return 0.0
