import numpy as np
import pytest
from aeon.testing.utils.data_gen import make_example_3d_numpy
from aeon.distances import pairwise_distance

from tsml_eval.estimators.clustering import (
    TimeSeriesDBScan,
    TimeSeriesHDBScan,
    TimeSeriesAgglomerative
)

n_cases = 20
n_channels = 1
n_timepoints = 10


def _run_distance_test(clusterer, distance, **kwargs):
    X = make_example_3d_numpy(n_cases, n_channels, n_timepoints, return_y=False)
    model = clusterer(**kwargs, distance=distance).fit(X)
    assert model.labels_.shape[0] == n_cases
    assert model.precomputed_distances is None

    precomputed = pairwise_distance(X, metric=distance)

    precomputed_model = clusterer(
        **kwargs,
        distance=distance,
        precomputed_distances=precomputed
    ).fit(X)
    assert precomputed_model.labels_.shape[0] == n_cases
    assert np.array_equal(model.labels_, precomputed_model.labels_)
    assert precomputed_model.precomputed_distances is None

    fit_predict_model = clusterer(**kwargs, distance=distance).fit_predict(X)

    assert np.array_equal(model.labels_, fit_predict_model)


@pytest.mark.parametrize(
    "distance",
    [
        "euclidean",
        "adtw",
        "dtw",
        "ddtw",
        "wdtw",
        "wddtw",
        "erp",
        "edr",
        "lcss",
        "twe",
        "msm",
        "shape_dtw",
    ],
)
def test_dbscan(distance):
    _run_distance_test(TimeSeriesDBScan, distance)

@pytest.mark.parametrize(
    "distance",
    [
        "euclidean",
        "adtw",
        "dtw",
        "ddtw",
        "wdtw",
        "wddtw",
        "erp",
        "edr",
        "lcss",
        "twe",
        "msm",
        "shape_dtw",
    ],
)
def test_hdbscan(distance):
    _run_distance_test(TimeSeriesHDBScan, distance)


@pytest.mark.parametrize(
    "distance",
    [
        "euclidean",
        "adtw",
        "dtw",
        "ddtw",
        "wdtw",
        "wddtw",
        "erp",
        "edr",
        "lcss",
        "twe",
        "msm",
        "shape_dtw",
    ],
)
def test_agglomerative(distance):
    _run_distance_test(TimeSeriesAgglomerative, distance)
