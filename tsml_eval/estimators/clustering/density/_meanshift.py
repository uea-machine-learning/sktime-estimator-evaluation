import warnings
from collections import defaultdict

import numpy as np
from aeon.clustering import BaseClusterer
from aeon.clustering.averaging import VALID_BA_METRICS
from aeon.clustering.averaging._averaging import _resolve_average_callable
from aeon.distances import get_distance_function, pairwise_distance
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state, gen_batches
from sklearn.utils.parallel import Parallel, delayed


class TimeSeriesMeanShift(BaseClusterer):

    def _score(self, X, y=None):
        return 0.0

    def __init__(
        self,
        *,
        distance="euclidean",
        distance_params=None,
        bandwidth=None,
        seeds=None,
        cluster_all=True,
        n_jobs=None,
        max_iter=300,
        averaging_method="mean",
        average_params=None,
        random_state=None,
    ):
        self.distance = distance
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.cluster_all = cluster_all
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.random_state = random_state
        self.distance_params = distance_params
        self.averaging_method = averaging_method
        self.average_params = average_params

        self._distance = None
        self._random_state = None
        self._distance_params = {}
        self._average_params = {}
        self._averaging_method = None

        super().__init__()

    def _fit(self, X, y=None):
        """Perform clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to cluster.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
               Fitted instance.
        """
        self._check_params(X)

        bandwidth = self.bandwidth
        if bandwidth is None:
            bandwidth = self._estimate_bandwidth(X, n_jobs=self.n_jobs)

        seeds = self.seeds
        if seeds is None:
            seeds = X
        n_samples, n_channels, n_timepoints = X.shape
        center_intensity_list = []

        # We use n_jobs=1 because this will be used in nested calls under
        # parallel calls to _mean_shift_single_seed so there is no need for
        # further parallelism.
        nbrs = NearestNeighbors(
            radius=bandwidth,
            n_jobs=1,
            algorithm="brute",
            metric=self._distance,
            metric_params=self._distance_params,
        ).fit(X)

        # execute iterations on all seeds in parallel
        all_res = Parallel(n_jobs=self.n_jobs)(
            delayed(self._mean_shift_single_seed)(seed, X, nbrs, self.max_iter)
            for seed in seeds
        )
        # copy results in a list
        for i in range(len(seeds)):
            if all_res[i][1]:  # i.e. len(points_within) > 0
                center = all_res[i][0]
                intensity = all_res[i][1]
                center_intensity_list.append((center, intensity))

        # Sort list by intensity
        center_intensity_list.sort(key=lambda tup: tup[1], reverse=True)

        sorted_centers = np.array([tup[0] for tup in center_intensity_list])

        unique = np.ones(len(sorted_centers), dtype=bool)
        nbrs = NearestNeighbors(
            radius=bandwidth,
            n_jobs=self.n_jobs,
            algorithm="brute",
            metric=self._distance,
            metric_params=self._distance_params,
        ).fit(sorted_centers)
        for i, center in enumerate(sorted_centers):
            if unique[i]:
                neighbor_idxs = nbrs.radius_neighbors([center], return_distance=False)[
                    0
                ]
                unique[neighbor_idxs] = 0
                unique[i] = 1  # leave the current point as unique
        cluster_centers = sorted_centers[unique]

        # ASSIGN LABELS: a point belongs to the cluster that it is closest to
        nbrs = NearestNeighbors(
            n_neighbors=1,
            n_jobs=self.n_jobs,
            metric=self._distance,
            metric_params=self._distance_params,
            algorithm="brute",
        ).fit(cluster_centers)
        labels = np.zeros(n_samples, dtype=int)
        distances, idxs = nbrs.kneighbors(X)
        if self.cluster_all:
            labels = idxs.flatten()
        else:
            labels.fill(-1)
            bool_selector = distances.flatten() <= bandwidth
            labels[bool_selector] = idxs.flatten()[bool_selector]

        self.cluster_centers_, self.labels_ = cluster_centers, labels
        return self

    def _estimate_bandwidth(self, X, quantile=0.3, n_samples=None, n_jobs=None):
        if n_samples is not None:
            idx = self._random_state.permutation(X.shape[0])[:n_samples]
            X = X[idx]
        n_neighbors = int(X.shape[0] * quantile)
        if n_neighbors < 1:  # cannot fit NearestNeighbors with n_neighbors = 0
            n_neighbors = 1
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
            metric=self._distance,
            metric_params=self._distance_params,
            algorithm="brute",
        )
        nbrs.fit(X)

        bandwidth = 0.0
        for batch in gen_batches(len(X), 500):
            d, _ = nbrs.kneighbors(X[batch, :], return_distance=True)
            bandwidth += np.max(d, axis=1).sum()

        return bandwidth / X.shape[0]

    # separate function for each seed's iterative loop
    def _mean_shift_single_seed(self, my_mean, X, nbrs, max_iter):
        # For each seed, climb gradient until convergence or max_iter
        bandwidth = nbrs.get_params()["radius"]
        stop_thresh = 1e-3 * bandwidth  # when mean has converged
        completed_iterations = 0
        my_mean = my_mean
        while True:
            # Find mean of points within bandwidth
            i_nbrs = nbrs.radius_neighbors([my_mean], bandwidth, return_distance=False)[
                0
            ]

            temp = nbrs.radius_neighbors([my_mean], bandwidth, return_distance=False)
            points_within = X[i_nbrs]
            if len(points_within) == 0:
                break  # Depending on seeding strategy this condition may occur
            my_old_mean = my_mean  # save the old mean
            my_mean = self._averaging_method(points_within, **self._average_params)
            # If converged or at max_iter, adds the cluster
            if (
                self._distance(my_mean, my_old_mean) <= stop_thresh
                or completed_iterations == max_iter
            ):
                break
            completed_iterations += 1
        return my_mean, len(points_within), completed_iterations

    def _predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return pairwise_distance(
            X, self.cluster_centers_, metric=self._distance, **self._distance_params
        ).argmin(axis=1)

    def _check_params(self, X: np.ndarray) -> None:
        self._random_state = check_random_state(self.random_state)
        self._distance = get_distance_function(self.distance)

        if self.distance_params is None:
            self._distance_params = {}
        else:
            self._distance_params = self.distance_params
        if self.average_params is None:
            self._average_params = {}
        else:
            self._average_params = self.average_params

        # Add the distance to average params
        if "distance" not in self._average_params:
            # Must be a str and a valid distance for ba averaging
            if isinstance(self.distance, str) and self.distance in VALID_BA_METRICS:
                self._average_params["distance"] = self.distance
            else:
                # Invalid distance passed for ba so default to dba
                self._average_params["distance"] = "dtw"

        if "random_state" not in self._average_params:
            self._average_params["random_state"] = self._random_state

        self._averaging_method = _resolve_average_callable(self.averaging_method)
