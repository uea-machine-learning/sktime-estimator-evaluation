import numpy as np

# import pytest
from aeon.datasets import load_arrow_head
from sklearn.metrics import rand_score

# from tsml_eval.estimators.clustering.consensus.ivc import IterativeVotingClustering
from tsml_eval.estimators.clustering.consensus.elastic_ensemble import (
    ElasticEnsembleClustererFromFile,
)
from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH


def test_from_file_iterative_voting_clustering():
    """Test SimpleVote from file with ArrowHead results."""
    X_train, y_train = load_arrow_head(split="train")
    X_test, y_test = load_arrow_head(split="test")

    file_paths = [
        _TEST_RESULTS_PATH + "/clustering/PAM-DTW/Predictions/ArrowHead/",
        _TEST_RESULTS_PATH + "/clustering/PAM-ERP/Predictions/ArrowHead/",
        _TEST_RESULTS_PATH + "/clustering/PAM-MSM/Predictions/ArrowHead/",
    ]

    ee = ElasticEnsembleClustererFromFile(
        clusterers=file_paths, n_clusters=3, random_state=0
    )
    ee.fit(X_train, y_train)
    preds = ee.predict(X_test)

    assert ee.labels_.shape == (len(X_train),)
    assert isinstance(ee.labels_, np.ndarray)
    assert rand_score(y_train, ee.labels_) >= 0.6
    assert preds.shape == (len(X_test),)
    assert isinstance(preds, np.ndarray)
    assert rand_score(y_test, preds) >= 0.6
