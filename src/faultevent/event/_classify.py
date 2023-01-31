from . import map_circle, event_spectrum, weighted_event_spectrum
from collections.abc import Iterable
from dataclasses import dataclass
import copy
import numpy as np
import numpy.typing as npt
from scipy.stats import vonmises

"""Module for performing classification of events.

TODO: Develop a strategy for incorporating classification of events with
similar frequency. Currently only supports classification of events of
different frequencies.

This module requires Scipy."""

@dataclass
class EMComponent:
    mu: float
    kappa: float
    tau: float
    ord: float
    fixed_mu: bool = False
    fixed_kappa: bool = False

kappa_approx = lambda r: (1.28-0.53*abs(r)**2)*np.tan(np.pi/2*abs(r))

def event_probabilities(components: Iterable[EMComponent], x: npt.ArrayLike):
    """Returns the membership probabilities of each component for every
    event whose location is given by x"""
    F = len(components)
    M = np.shape(x)
    uw = np.zeros((F, *M))
    for f in range(F):
        cmpf = components[f]
        zf = map_circle(cmpf.ord, x)
        uw[f] = cmpf.tau * vonmises.pdf(zf, cmpf.kappa, cmpf.mu)
    w = uw/np.sum(uw, axis=0)
    return w

def em(components: Iterable[EMComponent], x: npt.ArrayLike, maxiter: int = 10):
    """Performs expectation maximisation on event locations x, given a
    set of partially fit components."""
    F = len(components)
    M = len(x)

    new_components = copy.deepcopy(components)
    iter = 0
    converged = False
    while not converged:
        # Expectation step
        w = event_probabilities(new_components, x)
        # Maximization step
        for f in range(F):
            cmpf = new_components[f]
            Mf = sum(w[f])
            tauf = Mf / M
            pf = weighted_event_spectrum(cmpf.ord, x, w[f])
            rf = np.conj(pf)/Mf
            muf = np.angle(rf) if not cmpf.fixed_mu else cmpf.mu
            kappaf = kappa_approx(rf) if not cmpf.fixed_kappa else cmpf.kappa

            new_class = EMComponent(muf, kappaf, tauf,
                                    cmpf.ord, cmpf.fixed_mu, cmpf.fixed_kappa)
            new_components[f] = new_class

        iter += 1
        if iter>maxiter: converged = True
    return w, new_components

def initial_conditions(f: npt.ArrayLike,
                       x: npt.ArrayLike,
                       x_inf: float,
                       anomaly_rate: int):
    """Estimates a set of initial condition components for every
    frequency f and anomalies."""
    # TODO: Automatically find anomaly rate
    n_events = sum((ff*x_inf for ff in f)) + anomaly_rate
    components = []
    for ff in f:
        pf = event_spectrum(ff, x)
        k = ff*x_inf
        r = np.conj(pf)/k
        mu = np.angle(r)
        kappa = 10
        tau = ff*x_inf/n_events
        components.append(EMComponent(mu, kappa, tau, ff))
    components.append(EMComponent(0, 1.e-5, anomaly_rate/n_events, 1/x_inf, True, True))
    return components

def expectation_maximisation(f: npt.ArrayLike,
                             x: npt.ArrayLike,
                             x_inf: float = None,
                             anomaly_rate: int = 0,
                             maxiter: int = 10,):
    """Perform expectation maximisation on events given their locations
    and their associated frequencies. Returns the membership
    probabilities of each component for every event.
    
    Arguments:
    f -- the frequencies associated with the events
    x -- the event locations
    
    Keyword arguments:
    x_inf -- length of the measurement period
    anomaly_rate -- number of anomalous events
    maxiter -- maximum number of EM iterations
    """
    if x_inf is None: x_inf = x[-1]
    cmp0 = initial_conditions(f, x, x_inf, anomaly_rate)
    w, _ = em(cmp0, x, maxiter=maxiter)
    return w