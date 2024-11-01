"""DummyForecaster always predicts the last value seen in training."""

from aeon.forecasting.base import BaseForecaster


class ETS(BaseForecaster):

    def __init__(self):
        super().__init__()

    def _fit(self, y, exog=None):
        """Fit dummy forecaster."""
        return self

    def _predict(self, y=None, exog=None):
        """Predict using dummy forecaster."""
        return 0

    def _forecast(self, y, exog=None):
        """Forecast using dummy forecaster."""
        return 0
