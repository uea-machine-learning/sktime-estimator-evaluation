from abc import ABC, abstractmethod

from aeon.classification.base import BaseClassifier


class BaseTimeSeriesImbalance(BaseClassifier, ABC):

    @abstractmethod
    def _fit_resample(self, X, y, **kwargs): ...

    def fit_resample(self, X, y, **kwargs):
        X, y, single_class = self._fit_setup(X, y)

        if not single_class:
            return self._fit_resample(X, y, **kwargs)
        return X
