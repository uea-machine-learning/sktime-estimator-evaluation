from collections import Counter
from typing import Optional, Union

import numpy as np
from aeon.classification.base import BaseClassifier
from imblearn.utils._validation import check_sampling_strategy
from sklearn.utils import check_random_state

from tsml_eval.imbalance._utils import _make_samples, unbalance_data
from tsml_eval.imbalance._wrappers import _ADASYN, _SmoteKNN


class TimeSeriesADASYN(_ADASYN, BaseClassifier):
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

        BaseClassifier.__init__(self)
        _ADASYN.__init__(
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

    def _fit(self, X, y):
        return _ADASYN.fit(self, X, y)

    def _predict(self, X) -> np.ndarray:
        return _ADASYN.predict(self, X)

    def fit_resample(self, X, y, **kwargs):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        X, y, single_class = self._fit_setup(X, y)
        self._random_state = check_random_state(self.random_state)

        if not single_class:

            self.sampling_strategy_ = check_sampling_strategy(
                self.sampling_strategy, y, self._sampling_type
            )

            return self._fit_resample(X, y, **kwargs)
        return X

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
    from aeon.datasets import load_gunpoint

    # Load dataset
    X_train, y_train = load_gunpoint(split="train")

    X_test, y_test = load_gunpoint(split="test")

    X_train = np.concatenate((X_train, X_test))
    y_train = np.concatenate((y_train, y_test))

    target_class = y_train[0]
    n_samples = 20

    X_unbalanced, y_unbalanced = unbalance_data(
        X_train, y_train, target_class, n_samples
    )
    smote = TimeSeriesADASYN(n_neighbors=31)
    X_resampled, y_resampled = smote.fit_resample(X_unbalanced, y_unbalanced)

    print("Resampled dataset shape %s" % Counter(y_resampled))  # noqa: T201
    print(X_resampled.shape)  # noqa: T201
