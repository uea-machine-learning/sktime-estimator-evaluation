"""Time series kshapes."""

from typing import Union

import numpy as np
from aeon.clustering.base import BaseClusterer
from aeon.distances import pairwise_distance
from aeon.utils.validation._dependencies import _check_soft_dependencies
from numpy.random import RandomState
from sklearn.utils import check_random_state


class TimeSeriesKShapesExtended(BaseClusterer):
    _tags = {
        "capability:multivariate": True,
        "python_dependencies": "tslearn",
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Union[str, np.ndarray] = "random",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: Union[int, RandomState] = None,
    ):
        self.init_algorithm = init_algorithm
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._tslearn_k_shapes = None
        self._random_state = None

        super().__init__(n_clusters=n_clusters)

    def _fit(self, X, y=None):
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
                (n_cases, n_timepoints)
            A collection of time series instances.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self:
            Fitted estimator.
        """
        _check_soft_dependencies("tslearn", severity="error")
        from tslearn.clustering import KShape

        self._random_state = check_random_state(self.random_state)

        init_algorithm = self.init_algorithm
        if self.init_algorithm == "kmeans++":
            init_algorithm = self._kmeans_plus_plus_center_initializer(X)
            init_algorithm = init_algorithm.swapaxes(1, 2)

        self._tslearn_k_shapes = KShape(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            n_init=self.n_init,
            verbose=self.verbose,
            init=init_algorithm,
        )

        _X = X.swapaxes(1, 2)

        self._tslearn_k_shapes.fit(_X)
        self._cluster_centers = self._tslearn_k_shapes.cluster_centers_
        self.labels_ = self._tslearn_k_shapes.labels_
        self.inertia_ = self._tslearn_k_shapes.inertia_
        self.n_iter_ = self._tslearn_k_shapes.n_iter_

    def _kmeans_plus_plus_center_initializer(self, X: np.ndarray):
        initial_center_idx = self._random_state.randint(X.shape[0])
        indexes = [initial_center_idx]

        min_distances = pairwise_distance(
            X,
            X[initial_center_idx].reshape(1, -1),
            metric="sbd",
        ).flatten()

        for _ in range(1, self.n_clusters):
            probabilities = min_distances / min_distances.sum()
            next_center_idx = self._random_state.choice(X.shape[0], p=probabilities)
            indexes.append(next_center_idx)
            distances_new_center = pairwise_distance(
                X,
                X[next_center_idx].reshape(1, -1),
                metric="sbd",
            ).flatten()
            min_distances = np.minimum(min_distances, distances_new_center)

        centers = X[indexes]
        return centers

    def _predict(self, X, y=None) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
                (n_cases, n_timepoints)
            A collection of time series instances.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_cases,))
            Index of the cluster each time series in X belongs to.
        """
        _X = X.swapaxes(1, 2)
        return self._tslearn_k_shapes.predict(_X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {
            "n_clusters": 2,
            "init_algorithm": "random",
            "n_init": 1,
            "max_iter": 1,
            "tol": 1e-4,
            "verbose": False,
            "random_state": 1,
        }

    def _score(self, X, y=None):
        return np.abs(self.inertia_)
