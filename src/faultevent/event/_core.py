import numpy as np
import numpy.typing as npt


def fourier_sequence(
        f: npt.ArrayLike,
        x: npt.ArrayLike,
        w: npt.ArrayLike) -> np.ndarray:
    return w*np.exp(-2j*np.pi*np.outer(f, x))


def weighted_event_spectrum(
        f: npt.ArrayLike,
        x: npt.ArrayLike,
        w: npt.ArrayLike) -> np.ndarray:
    """Evaluates the weighted event spectrum at f given locations x and
    weights w"""
    return np.sum(fourier_sequence(f, x, w), axis=-1)


def event_spectrum(
        f: npt.ArrayLike,
        x: npt.ArrayLike) -> np.ndarray:
    """Evaluates the event spectrum at f given locations x """
    return weighted_event_spectrum(f, x, 1.0)


def cumulative_event_spectrum(
        f: npt.ArrayLike,
        x: npt.ArrayLike) -> np.ndarray:
    """Evaluates the event spectrum at f given locations x """
    return np.cumsum(fourier_sequence(f, x, 1.0))


def map_circle(f: float, x: npt.ArrayLike) -> np.ndarray:
    """Returns locations x mapped to a 'circle' at frequency f"""
    return 2*np.pi*f*(np.mod(x, 1/f))


def period_number(f: float, x: npt.ArrayLike) -> np.ndarray:
    """Returns the number of the period in which x falls, given
    frequency f"""
    return np.asarray(x//(1/f))
