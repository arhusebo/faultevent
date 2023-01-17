from typing import TypeVar

import numpy as np
from . import Signal, SignalModel


def _X(x, p):
    """Construct the order-p design matrix from a process x"""
    return np.array([np.flip(x[i-p:i]) for i in range(p, len(x))])


def _y(x, p):
    """Construct the order-p target vector from a process x"""
    return x[p:]


def _fit(x, p):
    """Fits the p-order model on data x using least squares"""
    X = _X(x, p)
    y = _y(x, p)
    c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    e = X@c-y
    return c, e


def _predict(c, x):
    """Predict process x given AR coefficients c"""
    p = len(c)
    X = _X(x, p)
    y = _y(x, p)
    ypred = X@c
    err = y-ypred
    return ypred, err


def process(c, s, x0=None):
    """Continue process x0 over inputs s"""
    p = len(c)
    if x0 is None: x = np.zeros((p,), dtype=float)
    else:
        if len(x0)>=p:
            raise ValueError("x0 must be longer than c")
        x = np.flip(x0[-p:]) # last p entries of x0 reversed

    for sn in s:
        xn = np.dot(c, x) + sn
        yield xn
        x = np.roll(x, 1)
        x[0] = xn


SelfARModel = TypeVar("SelfARModel", bound="ARModel")

class ARModel(SignalModel):
    """Autoregressive signal model
    
    Attributes:
    c -- model coefficients
    var -- noise variance
    p -- model order
    
    Methods:
    predict -- predict next entry
    process -- process over given inputs
    from_data -- creates an AR model fit on data"""
    def __init__(self, c, var):
        self.c = np.array(c)
        self.var = var
        self.p = len(c)

    def __len__(self):
        """Returns the order of the AR model"""
        return len(self.c)

    def residuals(self, signal: Signal) -> Signal:
        """Prediction residuals of signal"""
        _, e = _predict(self.c, signal.y)
        x = signal.x[self.p:]
        return Signal(e, x, uniform_samples=True)

    def process(self, data: Signal, x0=None):
        """Response to input with initial state x0"""
        y = np.fromiter(process(self.c, data.y, x0), dtype=float, count=len(data))
        return Signal(y, data.x)

    @classmethod
    def from_signal(cls, signal: Signal, p) -> SelfARModel:
        """Creates an AR model fit on data"""
        c, e = _fit(signal.y, p)
        var = np.var(e)
        return cls(c, var)