"""Implementation of Hyndman C functions for exponential smoothing in numba.

Three functions from here
https://github.com/robjhyndman/forecast/blob/master/src/etscalc.c

// Functions called by R
void etscalc(double *, int *, double *, int *, int *, int *, int *,
    double *, double *, double *, double *, double *, double *, double *, int*);
void etssimulate(double *, int *, int *, int *, int *,
    double *, double *, double *, double *, int *, double *, double *);
void etsforecast(double *, int *, int *, int *, double *, int *, double *);


Nixtla version is generated by a notebook using  something called mbdev.

https://github.com/Nixtla/statsforecast/blob/main/statsforecast/ets.py

completely undocumented. We need to verify what each of the parameters mean,
and check translation.
"""
from enum import Enum, auto
import math

from numba import njit
from numba.experimental import jitclass
import numpy as np

from tsml_eval._wip.forecasting.base import BaseForecaster

NA = -99999.0
MAX_NMSE = 30
MAX_SEASONAL_PERIOD = 24

NONE = 0
ADDITIVE = 1
MULTIPLICATIVE = 2

@jitclass
class ModelType:
    """
    Class describing the error, trend and seasonality model of an ETS forecaster

    Attributes
    ----------
    error_type : ComponentType
        The type of error model; either Additive or Multiplicative
    trend_type : ComponentType
        The type of trend model; one of None, additive, or multiplicative.
    seasonality_type : ComponentType
        The type of seasonality model; one of None, additive, multiplicative.
    seasonal_period : int
        The period of the seasonality (m) (e.g., for quaterly data seasonal_period = 4).
    """
    error_type : int
    trend_type : int
    seasonality_type : int
    seasonal_period : int
    horizon : int

    def __init__ (self,
                  error_type = ADDITIVE,
                  trend_type = NONE,
                  seasonality_type = NONE,
                  seasonal_period=1,
                  horizon=1
                  ):
        assert (seasonal_period <= MAX_SEASONAL_PERIOD) or (seasonality_type == NONE), "Seasonal period must be <= 24 if seasonality is enabled"
        assert (error_type != NONE) , "Error must be either additive or multiplicative"
        if seasonal_period < 1:
            seasonal_period = 1
        self.error_type = error_type
        self.trend_type = trend_type
        self.seasonality_type = seasonality_type
        self.seasonal_period = seasonal_period
        self.horizon = horizon

@jitclass
class ModelParameters:
    """
    Class describing the error, trend and seasonality model of an ETS forecaster

    Attributes
    ----------
    alpha : float
        Smoothing parameter for the level.
    beta : float
        Smoothing parameter for the trend.
    gamma : float
        Smoothing parameter for the seasonality.
    phi : float
        Damping parameter.
    """
    alpha : float
    beta : float
    gamma : float
    phi : float

    def __init__ (self,
                  alpha,
                  beta,
                  gamma,
                  phi
                  ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi

class ExponentialSmoothingForecaster(BaseForecaster):
    """
    Forecaster that implements the exponential smoothing model.

    Attributes
    ----------
    model_type : ModelType
        Collection of parameters decribing the (Error, Trend, Seasonality) 
        types of the model, as well as the seasonal period if applicable.
    model_parameters : ModelParameters
        Collection of parameters describing the level, trend and 
        seasonality smoothing parameters, as well as the damping parameter.
    _level : float
        Inital value for the level parameter l_0
    _trend : float
        Initial value for the trend parameter b_0
    _seasonality : np.ndarray
        Intial values for the seasonality parameter s_0 ... s_m
    _residuals : np.ndarray
        Residuals of the fitted model.
    _avg_mean_sq_err : float
        Average MSE of the 1-step forecasts
    _liklihood : float
        Liklihood of the model
    """

    def __init__(self,
                 window : int,
                 model_parameters : ModelParameters,
                 _level : float,
                 _trend : float,
                 _seasonality : float,
                 model_type = ModelType()):
        super().__init__(model_type.horizon, window)
        self.model_parameters = model_parameters
        self._level = _level
        self._trend = _trend
        self._seasonality = _seasonality
        self.model_type = model_type
        self._residuals = np.zeros(0)
        self._avg_mean_sq_err = 0
        self._liklihood = 0

    def fit(self, y, X=None):
        if self.model_type.trend_type == NONE:
            self._trend = 0.0
        if self.model_type.seasonality_type == NONE:
            for j in range(self.model_type.seasonal_period):
                self._seasonality[j] = 0
        self._level, self._trend, self._seasonality, self._residuals, self._liklihood, self._avg_mean_sq_err = fit(y, self.model_type, self.model_parameters, self._level, self._trend, self._seasonality, X)
        return self

    def predict(self, y=None, X=None):
        return predict(self.model_type, self.model_parameters, self._level, self._trend, self._seasonality, y, X)

    def forecast(self, y, X=None):
        """

        basically fit_predict.

        Returns
        -------
        np.ndarray
            single prediction directly after the last point in X.
        """
        self.fit(y,X)
        return self.predict()
    
    def __str__ (self):
        return f"ExponentialSmoothingForecaster\n\
        (error_type={self.model_type.error_type},\n\
        trend_type={self.model_type.trend_type},\n\
        seasonality_type={self.model_type.seasonality_type},\n\
        seasonal_period={self.model_type.seasonal_period},\n\
        horizon={self.model_type.horizon},\n\
        alpha={self.model_parameters.alpha},\n\
        beta={self.model_parameters.beta},\n\
        gamma={self.model_parameters.gamma},\n\
        phi={self.model_parameters.phi},\n\
        _level={self._level},\n\
        _trend={self._trend},\n\
        _seasonality={self._seasonality},\n\
        self._residuals={self._residuals},\n\
        self._avg_mean_sq_err={self._avg_mean_sq_err},\n\
        self._liklihood={self._liklihood},\n\
        )"

@njit(fastmath=True, cache=True)
def fit(y, model_type, model_parameters, level, trend, seasonality, X=None):
    """
    Fit forecaster to y, optionally using exogenous data X.

    Split y into windows of length window and train the forecaster on each window
    Exponential smooting (fit?)

    Do parameters map to Hyndman?? Why 14 not 15?

    Parameters
    ----------
    X : Time series on which to learn a forecaster
    y : np.ndarray
        Time series data.

    Returns
    -------
    self
        Fitted forecaster
    """
    _residuals = np.zeros(len(y))
    _liklihood = 0
    _avg_mean_sq_err = 0
    olds = np.zeros(MAX_SEASONAL_PERIOD)
    lik2 = 0.0
    for i, y_i in enumerate(y):
        # Copy previous state.
        oldl = level
        oldb = 0
        if model_type.trend_type != NONE:
            oldb = trend
        if model_type.seasonality_type != NONE:
            for j in range(model_type.seasonal_period):
                olds[j] = seasonality[j]

        # One step forecast.
        forecast_value = predict(model_type, model_parameters, level, trend, seasonality)
        if(math.fabs(forecast_value - NA) < 1.0e-10):  # TOL
            _liklihood = NA
            return
        if model_type.error_type == ADDITIVE:
            _residuals[i] = y_i - forecast_value
        else:
            _residuals[i] = (y_i - forecast_value) / forecast_value
        _avg_mean_sq_err += math.pow(y_i - forecast_value, 2)
        # Update state.
        level, trend, seasonality = update(model_type, model_parameters, oldl, level, oldb, trend, olds, seasonality, y[i])
        _liklihood += _residuals[i]*_residuals[i]
        lik2 += np.log(math.fabs(forecast_value))
    _avg_mean_sq_err /= len(y)
    _liklihood = len(y) * np.log(_liklihood)
    if model_type.error_type == MULTIPLICATIVE:
        _liklihood += 2*lik2
    return level, trend, seasonality, _residuals, _liklihood, _avg_mean_sq_err


@njit(fastmath=True, cache=True)
def predict(model_type, model_parameters, level, trend, seasonality, y=None, X=None):
    """Predict values for time series X. Performs forcasting.

    Helper function for fit_ets.

    Parameters
    ----------
    model_type : ModelType
        Parameters describing the (E,T,S) model type, 
        seasonal period and forecasting horizon.
    model_parameters: ModelParameters
        Collection of the smoothing parameters 
        alpha, beta, gamma and phi of the model.
    l : float
        Current level.
    b : float
        Current trend.
    s : float
        Current seasonal components.
    """
    phistar = model_parameters.phi

    # Forecasts
    if model_type.trend_type == NONE:  # No trend component.
        forecast_value = level
    elif model_type.trend_type == ADDITIVE:  # Additive trend component.
        forecast_value = level + phistar*trend
    elif trend < 0:
        forecast_value = NA
    else:
        forecast_value = level * trend**phistar

    j = model_type.seasonal_period - 1 - model_type.horizon
    while j < 0:
        j += model_type.seasonal_period

    if model_type.seasonality_type == ADDITIVE:
        forecast_value += seasonality[j]
    elif model_type.seasonality_type == ADDITIVE:
        forecast_value *= seasonality[j]
    # if i < (h-1): #Shouldn't be needed, only relates to forecast values before the forecast horizon
    #     if math.fabs(self.phi-1) < 1.0e-10:  # TOL
    #         phistar = phistar + 1
    #     else:
    #         phistar = phistar + self.phi**(i+1)
    return forecast_value

@njit(fastmath=True, cache=True)
def update(model_type, model_parameters, oldl, l, oldb, b, olds, s, y):
    """Updates states.

    Helper function for fit_ets

    Parameters
    ----------
    model_type : ModelType
        Parameters describing the (E,T,S) model type, 
        seasonal period and forecasting horizon.
    model_parameters: ModelParameters
        Collection of the smoothing parameters 
        alpha, beta, gamma and phi of the model.
    oldl : float
        Previous level.
    l : float
        Current level.
    oldb : float
        Previous trend.
    b : float
        Current trend.
    olds : np.ndarray
        Previous seasonal components.
    s : np.ndarray
        Current seasonal components.
    y : np.ndarray
        Time series data.

    Returns
    ----------
    l : float
        Updated level.
    b : float
        Updated trend.
    s : float
        Updated seasonal components.
    """
    # New level.
    if model_type.trend_type == NONE:
        phib = 0
        q = oldl   # l(t-1)
    elif model_type.trend_type == ADDITIVE:
        phib = model_parameters.phi*(oldb)
        q = oldl + phib   # l(t-1) + phi*b(t-1)
    elif math.fabs(model_parameters.phi-1) < 1.0e-10:  # TOL
        phib = oldb
        q = oldl * oldb   # l(t-1) * b(t-1)
    else:
        phib = oldb**model_parameters.phi
        q = oldl * phib   # l(t-1) * b(t-1)^phi
    if model_type.seasonality_type == NONE:
        p = y
    elif model_type.seasonality_type == ADDITIVE:
        p = y - olds[model_type.seasonal_period-1]   # y[t] - s[t-m]
    else:
        if math.fabs(olds[model_type.seasonal_period-1]) < 1.0e-10:  # TOL
            p = 1.0e10  # HUGEN
        else:
            p = y / olds[model_type.seasonal_period-1]   # y[t] / s[t-m]
    l = q + model_parameters.alpha*(p-q)
    # New growth.
    if model_type.trend_type != NONE:
        if model_type.trend_type == ADDITIVE:  # Additive trend component.
            r = l - oldl   # l[t] - l[t-1]
        else:  # Multiplicative trend component.
            if math.fabs(oldl) < 1.0e-10:  # TOL
                r = 1.0e10  # HUGEN
            else:
                r = l / oldl   # l[t] / l[t-1]
        b = phib + (model_parameters.beta / model_parameters.alpha)*(r - phib)   # b[t] = phi*b[t-1] + beta*(r - phi*b[t-1])
                                            # b[t] = b[t-1]^phi + beta*(r - b[t-1]^phi)
    # New season.
    if model_type.seasonality_type != NONE:
        if model_type.seasonality_type == ADDITIVE:  # Additive seasonal component.
            t = y - q
        else:  # Multiplicative seasonal compoenent.
            if math.fabs(q) < 1.0e-10:
                t = 1.0e10
            else:
                t = y / q
        s[0] = olds[model_type.seasonal_period-1] + model_parameters.gamma*(t - olds[model_type.seasonal_period-1])  # s[t] = s[t-m] + gamma*(t - s[t-m])
        for j in range(model_type.seasonal_period):
            s[j] = olds[j-1]   # s[t] = s[t]
    return l, b, s
