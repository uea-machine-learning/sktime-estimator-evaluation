from tsml_eval.estimators.clustering.consensus.base_from_file_consensus import \
    BaseFromFileConsensus


class ElasticEnsembleClustererFromFile(BaseFromFileConsensus):
    def __init__(self, clusterers: list[str], n_clusters=8, random_state=None):
        super().__init__(clusterers=clusterers, n_clusters=n_clusters,
                         random_state=random_state)

    def _build_ensemble(self, cluster_assignments):
        pass