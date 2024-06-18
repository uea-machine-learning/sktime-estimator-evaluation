from abc import ABC
from tsml_eval.estimators.clustering.consensus.base_from_file_consensus import \
    BaseFromFileConsensus
from tsml_eval.estimators.clustering.consensus.utils.cluster_ensemble_package import \
    cluster_ensembles_package_run


class _BaseClusterEnsemblePackage(BaseFromFileConsensus, ABC):
    def __init__(self, clusterers: list[str], solver: str, n_clusters=8,
                 random_state=None):
        self.solver = solver
        super().__init__(clusterers=clusterers, n_clusters=n_clusters,
                         random_state=random_state)

    def _build_ensemble(self, cluster_assignments):
        return cluster_ensembles_package_run(cluster_assignments,
                                             nclass=self.n_clusters, solver=self.solver)


class CSPAFromFile(_BaseClusterEnsemblePackage):

    def __init__(self, clusterers: list[str], n_clusters=8, random_state=None):
        super().__init__(clusterers=clusterers, solver="cspa", n_clusters=n_clusters,
                         random_state=random_state)


class HGPAFromFile(_BaseClusterEnsemblePackage):

    def __init__(self, clusterers: list[str], n_clusters=8, random_state=None):
        super().__init__(clusterers=clusterers, solver="hgpa", n_clusters=n_clusters,
                         random_state=random_state)


class MCLAFromFile(_BaseClusterEnsemblePackage):

    def __init__(self, clusterers: list[str], n_clusters=8, random_state=None):
        super().__init__(clusterers=clusterers, solver="mcla", n_clusters=n_clusters,
                         random_state=random_state)


class HBGFFromFile(_BaseClusterEnsemblePackage):

    def __init__(self, clusterers: list[str], n_clusters=8, random_state=None):
        super().__init__(clusterers=clusterers, solver="hbgf", n_clusters=n_clusters,
                         random_state=random_state)


class NMFFromFile(_BaseClusterEnsemblePackage):

    def __init__(self, clusterers: list[str], n_clusters=8, random_state=None):
        super().__init__(clusterers=clusterers, solver="nmf", n_clusters=n_clusters,
                         random_state=random_state)
