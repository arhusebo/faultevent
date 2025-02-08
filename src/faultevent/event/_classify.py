from . import map_circle, event_spectrum, weighted_event_spectrum, find_order
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable
import numpy as np
import numpy.typing as npt
from scipy.stats import vonmises, uniform

"""Module for performing classification of events.

TODO: Develop a strategy for incorporating classification of events with
similar frequency. Currently only supports classification of events of
different frequencies.

This module requires Scipy."""

dx_default = np.ones_like
kappa_approx = lambda r: (1.28-0.53*abs(r)**2)*np.tan(np.pi/2*abs(r))


class EventModel(ABC):
    
    @abstractmethod
    def pdf(self, x: npt.ArrayLike) -> npt.ArrayLike:
        ...

    @abstractmethod
    def refit(self, x: npt.ArrayLike, w: npt.ArrayLike = 1.0):
        ...


@dataclass
class FaultEvent(EventModel):
    xmin: float
    xmax: float
    mu: float
    kappa: float
    f: float

    def pdf(self, x: npt.ArrayLike) -> npt.ArrayLike:
        zf = map_circle(self.f, x)
        return vonmises.pdf(zf, self.kappa, self.mu)/(self.xmax-self.xmin)

    def refit(self, x: npt.ArrayLike, w: npt.ArrayLike = 1.0):
        p = weighted_event_spectrum(self.f, x, w)
        M = np.sum(w)
        r = np.conj(p) / M
        mu = np.angle(r)
        kappa = kappa_approx(r)
        return type(self)(self.xmin, self.xmax, mu, kappa, self.f)


@dataclass
class AnomalousEvent(EventModel):
    xmin: float
    xmax: float
    dx: Callable[[npt.ArrayLike], npt.ArrayLike] = dx_default

    def pdf(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return uniform.pdf(x, self.xmin, self.xmax) / np.abs(self.dx(x))

    def refit(self, x: npt.ArrayLike, w: npt.ArrayLike = 1.0):
        return type(self)(self.xmin, self.xmax, self.dx)

    def from_data(self, x: npt.ArrayLike, w: npt.ArrayLike = 1.0,):
        return self


class MixtureModel:

    def __init__(self, components: Iterable[EventModel], weights: Iterable[float]):
        self.components = components
        self.weights = weights

    def event_probabilities(self, x: npt.ArrayLike):
        """For each each event whose location is given by x, returns
        their membership probabilities of each mixture component.
        
        Output shape is (n_events, n_components)
        """
        shape_events = np.shape(x)
        n_components = len(self.components)
        uw = np.zeros((*shape_events, n_components))
        for i, (component, weights) in enumerate(zip(self.components, self.weights)):
            uw[...,i] = weights*component.pdf(x)
        w = uw/np.sum(uw, axis=-1)[...,np.newaxis]
        return w
    
    def classify_hard(self, x: npt.ArrayLike):
        """Performs a hard classification of events whose locations are
        given by x. The returned values correspond to the predicted
        event labels."""
        w = self.event_probabilities(x)
        y = np.argmax(w, axis=-1)
        return y

    def entropy(self, x: npt.ArrayLike):
        """Returns the entropy between mixture components at locations
        x, in nats."""
        w = self.event_probabilities(x)
        return -np.sum(w*np.log(w), axis=-1)


def em_step(model: MixtureModel, x: npt.ArrayLike):
    """Perform an expectation maximization step using current mixture
    and return an updated mixture"""
    n_events = len(x)
    new_components = []
    new_tau = []
    w = model.event_probabilities(x)
    for i, component in enumerate(model.components):
        if type(component) == AnomalousEvent:
            new_tau.append(model.weights[i])
        else:
            new_tau.append(sum(w[:,i]) / n_events)
        new_components.append(component.refit(x, w[:,i]))
    
    return MixtureModel(new_components, new_tau)


def expectation_maximization(initial_model: MixtureModel,
                             x: npt.ArrayLike,
                             maxiter: int = 10,) -> MixtureModel:
    """Performs iterations of the expectation maximization algorithm."""
    # TODO: Implement convergence criterion
    new_model = initial_model
    iter = 0
    while iter < maxiter:
        new_model = em_step(new_model, x)
        iter += 1
    
    return new_model


def initial_conditions(f: npt.ArrayLike,
                       x: npt.ArrayLike,
                       xmin: float,
                       xmax: float,
                       anomaly_rate: int,
                       default_kappa: float = 10.0,
                       dx: Callable[[npt.ArrayLike], npt.ArrayLike] = dx_default,
                       approximate_f: bool = False):
    """Estimates a set of initial condition components for every
    frequency f and anomalies."""
    # TODO: Automatically find anomaly rate
    xspan = xmax - xmin
    n_events = sum((ff*xspan for ff in f)) + anomaly_rate
    components = []
    weights = []
    components.append(AnomalousEvent(xmin, xmax, dx))
    weights.append(anomaly_rate/n_events)
    for f_ in f:
        ff = f_
        if approximate_f: ff, _ = find_order(x, f_-0.1, f_+0.1)
        pf = event_spectrum(ff, x)
        k = ff*xspan
        r = np.conj(pf)/k
        mu = np.angle(r)
        kappa = default_kappa
        weights.append(ff*xspan/n_events)
        components.append(FaultEvent(xmin, xmax, mu, kappa, ff))
    return MixtureModel(components, weights)


def classify_hard(w: npt.ArrayLike):
    """Performs a hard classification of events whose marginal mixture
    component probabilities are given by w. The returned values
    correspond to the predicted event labels."""
    y = np.argmax(w, axis=-1)
    return y


def entropy(w: npt.ArrayLike):
    """Returns the entropy of a probabilities w."""
    return -np.sum(w*np.log(w), axis=-1)