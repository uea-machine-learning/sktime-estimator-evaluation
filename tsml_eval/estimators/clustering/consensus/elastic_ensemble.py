import numpy as np
from scipy.optimize import linear_sum_assignment

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
        distances_to_average_over: str = "twe",
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

    def _compute_scores(self, X, cluster_assignments, evaluation_metric):
        scores = []
        for assignments in cluster_assignments:
            if evaluation_metric == "davies_bouldin_score":
                score = davies_bouldin_score_time_series(
                    X,
                    assignments,
                    distance=self.distances_to_average_over,
                    distance_params=self.distances_to_average_over_params,
                )
            else:
                score = calinski_harabasz_score_time_series(
                    X,
                    assignments,
                    distance=self.distances_to_average_over,
                    distance_params=self.distances_to_average_over_params,
                )
            scores.append(score)

        # Adjust weights based on the evaluation metric
        if evaluation_metric == "davies_bouldin_score":
            # Invert DB scores and normalize
            epsilon = 1e-10  # Small value to prevent division by zero
            inverted_scores = [1 / (score + epsilon) for score in scores]
            total_inverted_score = sum(inverted_scores)
            return [w / total_inverted_score for w in inverted_scores]
        else:  # calinski_harabasz_score
            # Use CH scores directly and normalize
            total_score = sum(scores)
            return [score / total_score for score in scores]

    def fit(self, X, y=None):
        """Fit model to X using IVC."""
        X = self._check_x(X)

        if self.random_state is not None:
            file_name = f"trainResample{self.random_state}.csv"
        else:
            file_name = "trainResample.csv"

        cluster_assignments = self._load_results_from_file(X, file_name, y)

        # Ensure cluster_assignments is a numpy array
        cluster_assignments = np.array(cluster_assignments)

        # Align cluster labels
        cluster_assignments = self._align_cluster_labels(cluster_assignments)

        self.train_accs_by_classifier = self._compute_scores(
            X, cluster_assignments, self.evaluation_metric
        )

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

    def _align_cluster_labels(self, cluster_assignments):
        """Align cluster labels using the cost matrix and Hungarian algorithm."""
        self._label_mappings = []
        n_clusterers = cluster_assignments.shape[0]
        n_samples = cluster_assignments.shape[1]
        new_assignments = np.zeros((n_clusterers, n_samples), dtype=np.int32)
        new_assignments[0] = cluster_assignments[0]
        reference_labels = cluster_assignments[0]
        for i in range(1, n_clusterers):
            # Compute cost matrix
            cost_matrix = -np.histogram2d(
                cluster_assignments[i], reference_labels, bins=self.n_clusters
            )[0]
            _, col_indices = linear_sum_assignment(cost_matrix)
            self._label_mappings.append(col_indices)
            # Map the cluster labels
            new_assignments[i] = col_indices[cluster_assignments[i]]
        return new_assignments

    def predict_proba(self, X):
        """Predict cluster probabilities for X."""
        X = self._check_x(X)

        if self.random_state is not None:
            file_name = f"testResample{self.random_state}.csv"
        else:
            file_name = "testResample.csv"

        cluster_assignments = self._load_results_from_file(X, file_name)
        cluster_assignments = np.array(cluster_assignments)

        # Align cluster labels using the stored label mappings
        cluster_assignments = self._align_predict_cluster_labels(cluster_assignments)

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

    def _align_predict_cluster_labels(self, cluster_assignments):
        """Align cluster labels during prediction using stored label mappings."""
        n_clusterers = cluster_assignments.shape[0]
        n_samples = cluster_assignments.shape[1]
        new_assignments = np.zeros((n_clusterers, n_samples), dtype=np.int32)
        new_assignments[0] = cluster_assignments[0]
        for i in range(1, n_clusterers):
            mapping = self._label_mappings[i - 1]
            new_assignments[i] = mapping[cluster_assignments[i]]
        return new_assignments

    def predict(self, X):
        """Predict cluster labels for X."""
        probas = self.predict_proba(X)
        idx = np.argmax(probas, axis=1)
        return idx

    def _predict_probs_from_labels(self, cluster_assignments) -> np.ndarray:
        """Compute weighted probabilities from cluster assignments."""
        output_probas = []
        for c in range(len(self.train_accs_by_classifier)):
            this_train_acc = self.train_accs_by_classifier[c]
            # Convert labels to one-hot encoding
            labels_one_hot = np.eye(self.n_clusters)[cluster_assignments[c]]
            this_probas = labels_one_hot * this_train_acc
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
