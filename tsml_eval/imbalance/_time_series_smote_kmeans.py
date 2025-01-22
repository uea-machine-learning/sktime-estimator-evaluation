import math
from typing import Optional, Union

import numpy as np
from aeon.clustering import TimeSeriesKMeans
from aeon.distances import pairwise_distance
from imblearn.over_sampling import KMeansSMOTE
from imblearn.utils import check_sampling_strategy
from sklearn.utils import check_random_state

from tsml_eval.imbalance._base import BaseTimeSeriesImbalance
from tsml_eval.imbalance._utils import _make_samples
from tsml_eval.imbalance._wrappers import _SmoteKNN


class TimeSeriesSMOTEKmeans(BaseTimeSeriesImbalance, KMeansSMOTE):
    """SMOTE with K-means clustering for time series data.

    Combines the K-means clustering-based SMOTE algorithm with support for time
    series data through elastic distance measures and averaging methods.

    Parameters
    ----------
    sampling_strategy : str or dict, default='auto'
        The sampling strategy to use. When a string, it specifies the class
        sampling strategy:
        - 'minority': resample the minority class.
        - 'not minority': resample all classes but the minority class.
        - 'not majority': resample all classes but the majority class.
        - 'all': resample all classes.
        When a dictionary, the keys are the target classes, and the values are the
        desired number of samples after resampling.
    random_state : int, np.random.RandomState, or None, default=None
        Controls the random number generator for reproducibility.
    distance : str or callable, default='dtw'
        Distance metric for time series data. If a string, it must be a valid
        distance metric name available in `aeon.distances`. If a callable, it must
        accept two 2D numpy arrays of shape `(n_channels, n_timepoints)` and return
        a float.
    distance_params : dict, default=None
        Parameters for the distance metric, if `distance` is specified as a string.
    averaging_method : str, default='ba'
        Averaging method used for clustering, such as dynamic barycenter averaging
        (ba). Supports custom averaging methods through a callable.
    average_params : dict, default=None
        Dictionary of parameters for the averaging method. Custom parameters for
        distance and averaging can be specified here.
    n_neighbors : int, default=1
        Number of neighbors to consider in the KNN model used within SMOTE.
    weights : str or callable, default='uniform'
        Weighting mechanism for KNN voting. Options are:
        - 'uniform': All neighbors contribute equally.
        - 'distance': Neighbors contribute inversely proportional to their distance.
        - A callable function that computes custom weights.
    n_jobs : int, default=1
        Number of parallel jobs to use for neighbor searches.
        - `None`: Use a single process.
        - `-1`: Use all available processors.
    n_clusters : int, default=2
        Number of clusters to form for the K-means clustering step.
    cluster_balance_threshold : "auto" or float, default="auto"
        Threshold to determine whether a cluster is balanced. If "auto", it is
        calculated based on the ratio of each class.
    density_exponent : "auto" or float, default="auto"
        Exponent used to calculate the density of a cluster. "Auto" uses a
        feature-length-based exponent.
    n_init : int, default=1
        Number of runs of the K-means algorithm with different centroid seeds.
        The result with the lowest inertia is chosen.
    init : str, default="kmeans++"
        Initialisation method for centroids. Options are "kmeans++", "random", or
        a numpy array with predefined centroids.

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary with the class labels as keys and the number of samples to
        generate as values.
    kmeans_estimator_ : TimeSeriesKMeans
        The fitted K-means clustering method used for grouping prior to SMOTE.
    nn_k_ : _SmoteKNN
        The fitted KNN estimator used in SMOTE.
    cluster_balance_threshold_ : float
        The threshold used during `fit` to determine cluster balance.
    """

    def __init__(
        self,
        *,
        sampling_strategy: Union[str, dict] = "auto",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        distance: Union[str, callable] = "dtw",
        distance_params: Optional[dict] = None,
        averaging_method: str = "ba",
        average_params: Optional[dict] = None,
        n_neighbors: int = 1,
        weights: Union[str, callable] = "uniform",
        n_jobs: int = 1,
        n_clusters: int = 2,
        cluster_balance_threshold: Union[float, str] = "auto",
        density_exponent: Union[float, str] = "auto",
        n_init: int = 1,
        init: str = "kmeans++",
    ):
        self.distance = distance
        self.distance_params = distance_params
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.n_jobs = n_jobs
        self._random_state = None
        self.average_params = average_params
        self.n_clusters = n_clusters
        self.cluster_balance_threshold = cluster_balance_threshold
        self.density_exponent = density_exponent
        self.averaging_method = averaging_method
        self.n_init = n_init
        self.init = init

        self.sampling_strategy_ = None

        self._distance_params = distance_params or {}
        self._average_params = {
            "distance": distance,
            **self._distance_params,
            **(average_params or {}),
        }

        BaseTimeSeriesImbalance.__init__(self)
        KMeansSMOTE.__init__(
            self,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=_SmoteKNN(
                distance=distance,
                distance_params=distance_params,
                n_neighbors=n_neighbors,
                weights=weights,
                n_jobs=n_jobs,
            ),
            kmeans_estimator=TimeSeriesKMeans(
                distance=distance,
                distance_params=distance_params,
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
                init=init,
                averaging_method=averaging_method,
                average_params=self._average_params,
            ),
            cluster_balance_threshold=cluster_balance_threshold,
            density_exponent=density_exponent,
        )

    def _fit_resample(self, X, y, **kwargs):
        self._random_state = check_random_state(self.random_state)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )
        return KMeansSMOTE._fit_resample(self, X, y, **kwargs)

    def _fit(self, X, y):
        return KMeansSMOTE.fit(self, X, y)

    def _predict(self, X) -> np.ndarray:
        return KMeansSMOTE.predict(self, X)

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0, y=None
    ):
        return _make_samples(
            X,
            self._random_state,
            y_dtype,
            y_type,
            nn_data,
            nn_num,
            n_samples,
            step_size,
            y,
        )

    def _find_cluster_sparsity(self, X):
        """Compute the cluster sparsity."""
        distance_matrix = pairwise_distance(
            X, method=self.distance, **self._distance_params
        )
        for ind in range(X.shape[0]):
            distance_matrix[ind, ind] = 0

        non_diag_elements = (X.shape[0] ** 2) - X.shape[0]
        mean_distance = distance_matrix.sum() / non_diag_elements
        exponent = (
            math.log(X.shape[0], 1.6) ** 1.8 * 0.16
            if self.density_exponent == "auto"
            else self.density_exponent
        )
        return (mean_distance**exponent) / X.shape[0]


if __name__ == "__main__":
    from collections import Counter

    from aeon.datasets import load_gunpoint

    from tsml_eval.imbalance._utils import unbalance_data

    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")

    target_class = "1"
    n_samples = 20

    X_unbalanced, y_unbalanced = unbalance_data(
        X_train, y_train, target_class, n_samples
    )
    smote = TimeSeriesSMOTEKmeans(
        n_neighbors=10,
        distance="euclidean",
        averaging_method="mean",
        cluster_balance_threshold=0.1,
    )
    X_resampled, y_resampled = smote.fit_resample(X_unbalanced, y_unbalanced)

    print("Resampled dataset shape %s" % Counter(y_resampled))  # noqa: T201
    print(X_resampled.shape)  # noqa: T201
