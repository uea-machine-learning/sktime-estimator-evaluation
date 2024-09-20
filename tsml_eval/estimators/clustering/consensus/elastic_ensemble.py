import numpy as np

from tsml_eval.estimators.clustering.consensus.base_from_file_consensus import (
    BaseFromFileConsensus,
)
from tsml_eval.estimators.clustering.consensus.utils.unsupervised_evaluation import (
    calinski_harabasz_score_time_series,
    davies_bouldin_score_time_series,
)


class ElasticEnsembleClustererFromFile(BaseFromFileConsensus):
    def __init__(
        self,
        clusterers: list[str],
        n_clusters=1,
        evaluation_metric: str = "davies_bouldin_score",
        distances_to_average_over: list[str] = ["twe"],
        distances_to_average_over_params: dict[str, dict] = None,
        random_state=None,
    ):
        self.evaluation_metric = evaluation_metric
        self.train_accs_by_classifier = []
        self.distances_to_average_over = distances_to_average_over
        self.distances_to_average_over_params = (
            distances_to_average_over_params
            if distances_to_average_over_params is not None
            else {}
        )
        super().__init__(
            clusterers=clusterers, n_clusters=n_clusters, random_state=random_state
        )

    def _learn_accs(self, X):
        pass

    def fit(self, X, y=None):
        """Fit model to X using IVC."""
        X = self._check_x(X)

        if self.random_state is not None:
            file_name = f"trainResample{self.random_state}.csv"
        else:
            file_name = "trainResample.csv"

        cluster_assignments = self._load_results_from_file(X, file_name, y)

        evaluation_method = (
            davies_bouldin_score_time_series
            if self.evaluation_metric == "davies_bouldin_score"
            else calinski_harabasz_score_time_series
        )
        scores = []
        for assignments in cluster_assignments:
            curr = 0
            for distance in self.distances_to_average_over:
                curr += evaluation_method(
                    X,
                    assignments,
                    distance=distance,
                    distance_params=self.distances_to_average_over_params,
                )
            avg_score = curr / len(self.distances_to_average_over)
            scores.append(avg_score)

        # Adjust weights based on the evaluation metric
        if self.evaluation_metric == "davies_bouldin_score":
            # Invert DB scores and normalize
            epsilon = 1e-10  # Small value to prevent division by zero
            inverted_scores = [1 / (score + epsilon) for score in scores]
            total_inverted_score = sum(inverted_scores)
            self.train_accs_by_classifier = [
                w / total_inverted_score for w in inverted_scores
            ]
        else:  # calinski_harabasz_score
            # Use CH scores directly and normalize
            total_score = sum(scores)
            self.train_accs_by_classifier = [score / total_score for score in scores]

        probas = self._predict_probs_from_labels(cluster_assignments)

        preds = np.argmax(probas, axis=1)
        n_cases = len(preds)
        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = int(max(preds)) + 1
        dists = np.zeros((X.shape[0], n_clusters))
        for i in range(n_cases):
            dists[i, preds[i]] = 1
        self.labels_ = np.argmax(dists, axis=1)
        return self

    def predict_proba(self, X):
        """Predict cluster probabilities for X."""
        X = self._check_x(X)

        if self.random_state is not None:
            file_name = f"testResample{self.random_state}.csv"
        else:
            file_name = "testResample.csv"

        cluster_assignments = self._load_results_from_file(X, file_name)
        probas = self._predict_probs_from_labels(cluster_assignments)
        preds = np.argmax(probas, axis=1)
        n_cases = len(preds)
        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = int(max(preds)) + 1
        dists = np.zeros((X.shape[0], n_clusters))
        for i in range(n_cases):
            dists[i, preds[i]] = 1
        return dists

    def predict(self, X):
        """Predict cluster labels for X."""
        probas = self.predict_proba(X)
        idx = np.argmax(probas, axis=1)
        return idx

    def _format_cluster_assignments(self, cluster_assignments):
        new_cluster_assignments = []

        for assignments in cluster_assignments:
            curr = np.zeros((len(assignments), self.n_clusters))
            for i in range(len(assignments)):
                curr[i, assignments[i]] = 1
            new_cluster_assignments.append(curr)
        return new_cluster_assignments

    def _predict_probs_from_labels(self, cluster_assignments) -> np.ndarray:
        output_probas = []
        if len(cluster_assignments.shape) == 2:
            cluster_assignments = self._format_cluster_assignments(cluster_assignments)

        for c in range(len(self.train_accs_by_classifier)):
            this_train_acc = self.train_accs_by_classifier[c]
            this_probas = np.multiply(cluster_assignments[c], this_train_acc)
            output_probas.append(this_probas)

        output_probas = np.sum(output_probas, axis=0)
        return output_probas

    # Remove this is just kept as it is part of the ABC interface
    def _build_ensemble(self, cluster_assignments) -> np.ndarray:
        pass


if __name__ == "__main__":
    import numpy as np

    # import pytest
    from aeon.datasets import load_arrow_head

    # from tsml_eval.estimators.clustering.consensus.ivc import IterativeVotingClustering
    from tsml_eval.estimators.clustering.consensus.elastic_ensemble import (
        ElasticEnsembleClustererFromFile,
    )
    from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH

    """Test SimpleVote from file with ArrowHead results."""
    X_train, y_train = load_arrow_head(split="train")
    X_test, y_test = load_arrow_head(split="test")

    file_paths = [
        _TEST_RESULTS_PATH + "/clustering/PAM-DTW/Predictions/ArrowHead/",
        _TEST_RESULTS_PATH + "/clustering/PAM-ERP/Predictions/ArrowHead/",
        _TEST_RESULTS_PATH + "/clustering/PAM-MSM/Predictions/ArrowHead/",
    ]

    ee = ElasticEnsembleClustererFromFile(
        clusterers=file_paths,
        n_clusters=3,
        random_state=0,
        distances_to_average_over_params={"lmbda": 20},
    )
    ee.fit(X_train, y_train)
    print(ee.labels_)
    preds = ee.predict(X_test)

    print(preds)
    # print(ee.predict_proba(X_test))
