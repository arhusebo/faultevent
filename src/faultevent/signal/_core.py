from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np
import numpy.typing as npt

SelfSignal = TypeVar("SelfSignal", bound="Signal")

class Signal:
    """This class serves to relate signal y to an independent variable x.
    Useful if dimesionality-reducing operations, such as filtering,
    are performed and it is desired to keep track of the time of every sample.
    
    Attributes:
    y -- signal samples
    x -- independent variable

    Methods:
    idx_closest(x) -- returns the signal index closest to the specified x value.
    """
    def __init__(self, y: npt.ArrayLike, x: npt.ArrayLike, uniform_samples=False):
        """Signal constructor.
        
        Arguments:
        y -- the signal samples

        Keyword arguments:
        x -- the independent variable. Must be the same length as y.
        If None, the sample indices are used.
        """
        self.y = y
        if len(y) != len(x):
            raise ValueError(f"y ({len(y)}) and x ({len(x)}) must be of same length")
        self.x = x
        self.uniform_samples = uniform_samples

    @classmethod
    def from_uniform_samples(cls, y: npt.ArrayLike, dx: float = 1) -> SelfSignal:
        """Returns a Signal instance where the independent variable is
        inferred from the sample period dx."""
        n = len(y)
        x = np.linspace(0, dx*(n-1), n)
        return cls(y, x, True)

    def __len__(self) -> int:
        """Returns the signal length"""
        return len(self.y)

    def __getitem__(self, n) -> SelfSignal:
        """Returns a new signal instance"""
        return type(self)(self.y[n], self.x[n], self.uniform_samples)
    
    def idx_closest(self, x: npt.ArrayLike) -> np.ndarray:
        """Returns the signal index closest to the specified x value.
        Much more efficient if signal samples are uniformly spaces."""
        if self.uniform_samples:
            dx = self.x[1]-self.x[0]
            idx = np.array((x-self.x[0])/dx + .5, dtype=int)
            return idx
        else:
            return np.argmin(abs(self.x-x), axis=-1)
    
    @property
    def dx(self) -> float:
        if not self.uniform_samples:
            raise ValueError("Signal samples are not uniformly spaced")
        else:
            return self.x[1] - self.x[0]
        
    @property
    def fs(self) -> float:
        return 1/self.dx


class SignalModel(ABC):
    """Abstract class used to implement signal models for prediction
    or dynamic signal process modelling"""
    
    @abstractmethod
    def residuals(self, signal: Signal) -> Signal:
        """Residuals of signal prediction"""
        pass

    @abstractmethod
    def process(self, signal: Signal) -> Signal:
        """Signal process as response to input data"""
        pass