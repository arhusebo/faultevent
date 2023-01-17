from abc import ABC, abstractmethod

import numpy as np

from . import Signal

class Detector(ABC):
    """Abstract class used to implement general detector statistics.
    
    Methods:
    moments() -- returns the mean and standard deviation of the detector
    statistic(signal) -- returns the detector test statistic evaluated
    over signal
    """

    @abstractmethod
    def moments(self):
        """Returns the first two moments of the detector statistic"""
        pass

    @abstractmethod
    def statistic(self, signal):
        """Returns the statistic evaluated over given signal"""
        pass

class EnergyDetector(Detector):
    """Implements the energy detector. Useful when the signature is not
    known."""
    def __init__(self, m):
        self.m = m

    def moments(self, var):
        """ Returns the mean and standard deviation of the test statistic under H0 """
        mu = self.m*var
        std = var*np.sqrt(2*self.m)
        return mu, std

    def statistic(self, data: Signal) -> Signal:
        s = np.correlate(data.y**2, np.ones((self.m,)), mode="valid")
        return Signal(s, data.x[self.m-1:], data.uniform_samples)


class MatchedFilterDetector(Detector):
    """Implements the matched filter detector also known as replica correlator"""
    def __init__(self, h):
        self.h = h

    def moments(self, var):
        """Returns the standard deviation of the test statistic under H0"""
        std = np.sqrt(var)*np.linalg.norm(self.h)
        return 0, std

    def statistic(self, data: Signal) -> Signal:
        s = np.correlate(data.y, self.h, mode="valid")
        return Signal(s, data.x[:-len(self.h)+1], data.uniform_samples)


class MatchedFilterEnvelopeDetector(Detector):
    """Implements a matched filter envelope detector. Useful when the
    signature contains multiple periods of its components."""
    def __init__(self, h):
        self.h = h
    
    def moments(self, var):
        """NOT IMPLEMENTED"""
        # TODO: implement
        raise NotImplementedError

    def statistic(self, data: Signal) -> Signal:
        s = np.correlate(data.y, self.h, mode="valid")
        dft = np.fft.rfft(s)
        hsdft = np.zeros_like(s, dtype=complex)
        hsdft[-len(dft):] = 2*dft
        anl = np.fft.ifft(hsdft)
        return Signal(abs(anl), data.x[:len(s)], data.uniform_samples)
