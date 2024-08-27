"""Test MeanShift estimator."""

import numpy as np
import pytest
from aeon.clustering import TimeSeriesKMeans
from aeon.testing.data_generation import make_example_3d_numpy

from tsml_eval.estimators.clustering.density._meanshift import TimeSeriesMeanShift


def test_mean_shift():
    """Test MeanShift estimator."""
    # X_train = make_example_3d_numpy(10, 1, 10, return_y=False, random_state=1)
    # X_test = make_example_3d_numpy(10, 1, 10, return_y=False, random_state=2)
    from aeon.datasets import load_gunpoint as load_data
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
        rand_score,
    )

    X_train, y_train = load_data(split="train")
    X_test, y_test = load_data(split="test")

    X_train = np.concatenate((X_train, X_test))
    y_train = np.concatenate((y_train, y_test))
    # estimator = TimeSeriesMeanShift(distance="dtw", averaging_method="mean")
    meanshift = TimeSeriesMeanShift(
        distance="msm", averaging_method="mean", bandwidth=50
    )
    label = meanshift.fit(X_train)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"\nAMI {adjusted_mutual_info_score(label, y_train)}")
    print(f"NMI {normalized_mutual_info_score(label, y_train)}")
    print(f"ARI {adjusted_rand_score(label, y_train)}")
    print(f"Rand {rand_score(label, y_train)}")

    # test_labels = estimator.predict(X_test)
    stop = ""


def test_original():
    """Test MeanShift estimator."""
    import numpy as np
    from sklearn.cluster import MeanShift

    X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
    clustering = MeanShift(bandwidth=2).fit(X)
    clustering.labels_
    clustering.predict([[0, 0], [5, 5]])
