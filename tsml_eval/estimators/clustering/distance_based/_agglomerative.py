import warnings
from typing import Dict, Optional, Union

import numpy as np
from aeon.clustering.base import BaseClusterer
from aeon.distances import pairwise_distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster._agglomerative import _TREE_BUILDERS, _hc_cut, _hierarchical
from sklearn.utils import check_array
from sklearn.utils.validation import check_memory


class TimeSeriesAgglomerative(BaseClusterer):
    """Time series Agglomerative Clustering.

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.
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
    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.
    connectivity : array-like or callable, default=None
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        `kneighbors_graph`. Default is ``None``, i.e, the
        hierarchical clustering algorithm is unstructured.
    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at ``n_clusters``. This is
        useful to decrease computation time if the number of clusters is not
        small compared to the number of samples. This option is useful only
        when specifying a connectivity matrix. Note also that when varying the
        number of clusters and using caching, it may be advantageous to compute
        the full tree. It must be ``True`` if ``distance_threshold`` is not
        ``None``. By default `compute_full_tree` is "auto", which is equivalent
        to `True` when `distance_threshold` is not `None` or that `n_clusters`
        is inferior to the maximum between 100 or `0.02 * n_samples`.
        Otherwise, "auto" is equivalent to `False`.
    linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - 'ward' minimizes the variance of the clusters being merged.
        - 'average' uses the average of the distances of each observation of
          the two sets.
        - 'complete' or 'maximum' linkage uses the maximum distances between
          all observations of the two sets.
        - 'single' uses the minimum of the distances between all observations
          of the two sets.
    distance_threshold : float, default=None
        The linkage distance threshold at or above which clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.
    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.
    n_leaves_ : int
        Number of leaves in the hierarchical tree.
    n_connected_components_ : int
        The estimated number of connected components in the graph.
    children_ : array-like of shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        distance: str = "euclidean",
        distance_params: Optional[Dict] = None,
        precomputed_distances: Optional[np.ndarray] = None,
        memory: Optional[str] = None,
        connectivity: Optional[np.ndarray] = None,
        compute_full_tree: Union[bool, str] = "auto",
        linkage: str = "ward",
        distance_threshold: Optional[float] = None,
        compute_distances: bool = False,
    ):
        self.distance = distance
        self.distance_params = distance_params
        self.precomputed_distances = precomputed_distances
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances

        # Set to None so the values dont get written to a file
        self.precomputed_distances = None
        self._precomputed_distances = precomputed_distances

        self._model = PatchedTimeSeriesAgglomerative(
            n_clusters=n_clusters,
            metric="precomputed",
            memory=self.memory,
            connectivity=self.connectivity,
            compute_full_tree=self.compute_full_tree,
            linkage=self.linkage,
            distance_threshold=self.distance_threshold,
            compute_distances=self.compute_distances,
        )

        self.n_clusters_ = None
        self.labels_ = None
        self.n_leaves_ = None
        self.n_connected_components_ = None
        self.children_ = None
        self.distances_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

        super().__init__(n_clusters=n_clusters)

    def _fit(self, X, y=None):
        distance_params = {}
        if self.distance_params is not None:
            distance_params = self.distance_params

        if self._precomputed_distances is None:
            self._precomputed_distances = pairwise_distance(
                X, metric=self.distance, **distance_params
            )

        # This is here for experiments as they set this value
        if self._model.n_clusters != self.n_clusters:
            self._model.set_params(n_clusters=self.n_clusters)

        self._model.fit(self._precomputed_distances)

        self.n_clusters_ = self._model.n_clusters_
        self.labels_ = self._model.labels_
        self.n_leaves_ = self._model.n_leaves_
        self.n_connected_components_ = self._model.n_connected_components_
        self.children_ = self._model.children_
        if hasattr(self._model, "distances_"):
            self.distances_ = self._model.distances_
        if hasattr(self._model, "feature_names_in_"):
            self.feature_names_in_ = self._model.feature_names_in_
        if hasattr(self._model, "n_features_in_"):
            self.n_features_in_ = self._model.n_features_in_

    def fit_predict(self, X, y=None) -> np.ndarray:
        self.fit(X, y)
        return self.labels_

    def _predict(self, X, y=None) -> np.ndarray:
        raise RuntimeError("Agglomerative does not support predict method")

    def _score(self, X, y=None):
        return 0.0


class PatchedTimeSeriesAgglomerative(AgglomerativeClustering):
    """Patched Agglomerative Clustering.

    The default AgglomerativeClustering implementation in scikit-learn does not
    allow using 'ward' with precomputed distances. So this just overwrites the _fit
    and simply deletes where the error is being thrown.
    """

    def _fit(self, X):
        """Fit without validation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.

        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        memory = check_memory(self.memory)

        # TODO(1.6): remove in 1.6
        if self.metric is None:
            warnings.warn(
                (
                    "`metric=None` is deprecated in version 1.4 and will be removed in "
                    "version 1.6. Let `metric` be the default value "
                    "(i.e. `'euclidean'`) instead."
                ),
                FutureWarning,
                stacklevel=2,
            )
            self._metric = "euclidean"
        else:
            self._metric = self.metric

        if not ((self.n_clusters is None) ^ (self.distance_threshold is None)):
            raise ValueError(
                "Exactly one of n_clusters and "
                "distance_threshold has to be set, and the other "
                "needs to be None."
            )

        if self.distance_threshold is not None and not self.compute_full_tree:
            raise ValueError(
                "compute_full_tree must be True if distance_threshold is set."
            )

        tree_builder = _TREE_BUILDERS[self.linkage]

        connectivity = self.connectivity
        if self.connectivity is not None:
            if callable(self.connectivity):
                connectivity = self.connectivity(X)
            connectivity = check_array(
                connectivity, accept_sparse=["csr", "coo", "lil"]
            )

        n_samples = len(X)
        compute_full_tree = self.compute_full_tree
        if self.connectivity is None:
            compute_full_tree = True
        if compute_full_tree == "auto":
            if self.distance_threshold is not None:
                compute_full_tree = True
            else:
                # Early stopping is likely to give a speed up only for
                # a large number of clusters. The actual threshold
                # implemented here is heuristic
                compute_full_tree = self.n_clusters < max(100, 0.02 * n_samples)
        n_clusters = self.n_clusters
        if compute_full_tree:
            n_clusters = None

        # Construct the tree
        kwargs = {}
        if self.linkage != "ward":
            kwargs["linkage"] = self.linkage
            kwargs["affinity"] = self._metric

        distance_threshold = self.distance_threshold

        return_distance = (distance_threshold is not None) or self.compute_distances

        out = memory.cache(tree_builder)(
            X,
            connectivity=connectivity,
            n_clusters=n_clusters,
            return_distance=return_distance,
            **kwargs,
        )
        (self.children_, self.n_connected_components_, self.n_leaves_, parents) = out[
            :4
        ]

        if return_distance:
            self.distances_ = out[-1]

        if self.distance_threshold is not None:  # distance_threshold is used
            self.n_clusters_ = (
                np.count_nonzero(self.distances_ >= distance_threshold) + 1
            )
        else:  # n_clusters is used
            self.n_clusters_ = self.n_clusters

        # Cut the tree
        if compute_full_tree:
            self.labels_ = _hc_cut(self.n_clusters_, self.children_, self.n_leaves_)
        else:
            labels = _hierarchical.hc_get_heads(parents, copy=False)
            # copy to avoid holding a reference on the original array
            labels = np.copy(labels[:n_samples])
            # Reassign cluster numbers
            self.labels_ = np.searchsorted(np.unique(labels), labels)
        return self
