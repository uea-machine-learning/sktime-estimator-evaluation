from typing import Optional, Union

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.utils import check_sampling_strategy
from sklearn.utils import check_random_state

from tsml_eval.imbalance._base import BaseTimeSeriesImbalance
from tsml_eval.imbalance._wrappers import _SmoteKNN


class TimeSeriesSMOTEKnn(BaseTimeSeriesImbalance, SMOTE):

    def __init__(
        self,
        *,
        sampling_strategy: Union[str, dict] = "auto",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        distance: Union[str, callable] = "dtw",
        distance_params: Optional[dict] = None,
        n_neighbors: int = 1,
        weights: Union[str, callable] = "uniform",
        n_jobs: int = 1,
    ):
        self.distance = distance
        self.distance_params = distance_params
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.n_jobs = n_jobs
        self._random_state = None

        self.sampling_strategy_ = None

        BaseTimeSeriesImbalance.__init__(self)
        SMOTE.__init__(
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
        )

    def _fit_resample(self, X, y, **kwargs):
        self._random_state = check_random_state(self.random_state)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )
        return SMOTE._fit_resample(self, X, y, **kwargs)

    def _fit(self, X, y):
        return SMOTE.fit(self, X, y)

    def _predict(self, X) -> np.ndarray:
        return SMOTE.predict(self, X)

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


if __name__ == "__main__":
    from collections import Counter

    from aeon.datasets import load_gunpoint

    from tsml_eval.imbalance._utils import _make_samples, unbalance_data

    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")

    target_class = "1"
    n_samples = 20

    X_unbalanced, y_unbalanced = unbalance_data(
        X_train, y_train, target_class, n_samples
    )
    smote = TimeSeriesSMOTEKnn(n_neighbors=10)
    X_resampled, y_resampled = smote.fit_resample(X_unbalanced, y_unbalanced)

    print("Resampled dataset shape %s" % Counter(y_resampled))  # noqa: T201
    print(X_resampled.shape)  # noqa: T201
