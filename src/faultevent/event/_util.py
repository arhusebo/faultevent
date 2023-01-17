import numpy as np
import numpy.typing as npt

from . import event_spectrum

def find_order(x: npt.ArrayLike, ordmin: float, ordmax: float,
               density: float = 10000.0) -> tuple[float, float]:
    """Returns the accurate fault order by evaluating the event spectrum
    over the range (ordmin, ordmax). Density specifies the number of
    spectrum evaluations per order.
    Also returns the spectrum magnitude at this order."""
    ords = np.arange(ordmin, ordmax, 1/density)
    evsp = event_spectrum(ords, x)
    ordf_idx = np.argmax(abs(evsp))
    ordf = ords[ordf_idx]
    return ordf, abs(evsp[ordf_idx])

def fit_vonmises(f: float, x: npt.ArrayLike) -> tuple[float, float]:
    """Fit von Mises distribution to mapped locations x of frequency f.
    Kappa parameter uses approximation from
    'Directional Statistics', Kanti V. Mardia & Peter E. Jupp (2000)"""
    r = np.conj(event_spectrum(f, x))/len(x)
    ar = np.abs(r)
    mu = np.angle(r)
    kappa = (1.28-0.53*ar**2)*np.tan(np.pi/2*ar)
    return mu, kappa
