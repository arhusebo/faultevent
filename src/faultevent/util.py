import copy
from typing import Literal
import numpy as np
import numpy.typing as npt

from .signal import Signal, resample, Comparison,\
    matched_filter_location_estimates, energy_detector_location_estimates
from .event import find_order

def resampling_shaft_positions(time_series: Signal,
                               shaft_pos: Signal) -> np.ndarray:
    """Returns shaft positions of samples to use for resampling from
    time domain to shaft position domain. Calculated from the average
    samples per shaft revolution.
    
    Arguments:
    time_series -- signal to resample. Samples must be uniformly spaced
    in time as sample frequency is inferred from sample times.
    shaft_pos -- recorded shaft positions. Shaft position sample
    times should overlap with sample times of time_series.
    """
    fs = 1/(time_series.x[1] - time_series.x[0])
    startpos = shaft_pos.y[shaft_pos.idx_closest(time_series.x[0])]
    endpos = shaft_pos.y[shaft_pos.idx_closest(time_series.x[-1])]
    nrevs = endpos-startpos
    tottime = time_series.x[-1] - time_series.x[0]
    revs_per_sample = nrevs / (tottime * fs)
    pos_to_eval = np.arange(startpos, endpos, revs_per_sample)
    return pos_to_eval

def time_to_shaft(to_eval, time_series: Signal, shaft_pos: Signal) -> np.ndarray:
    """Interpolates signal from time to shaft position domain given
    which shaft positions to evaluate"""
    times_to_eval = np.interp(to_eval, shaft_pos.y, shaft_pos.x)
    return resample(times_to_eval, time_series, m=1000, batch_size=10000)

def order_track_time_series(time_series: Signal,
                            shaft_position: Signal) -> Signal:
    """ Returns the order tracked version of a time series given
    a record of shaft positions"""
    to_eval = resampling_shaft_positions(time_series, shaft_position)
    resampled = time_to_shaft(to_eval, time_series, shaft_position)
    return Signal(resampled, to_eval, uniform_samples=True)


def best_threshold(data: Signal,
                   search_intervals: list[tuple[float, float]],
                   thresholds: npt.ArrayLike | None = None,
                   n=10,
                   hysteresis: int | None = None,
                   dettype: Literal["mf", "ed"] = "mf",
                   order_search_density = 1000) -> tuple[float, float]:
    """Evaluates a metric over multiple thresholds and returns the
    best threshold and the score metric"""
    if thresholds is None: thresholds = np.linspace(0, 5*np.std(data.y), n)
    scores = np.zeros_like(thresholds, dtype=float)

    for i, thr in enumerate(thresholds):
        hys = None if hysteresis is None else hysteresis*thr
        cmp = Comparison.from_comparator(data, thr, hys)
        match dettype:
            case "mf": spos, _ = np.asarray(matched_filter_location_estimates(cmp))
            case "ed": spos = np.asarray(energy_detector_location_estimates(cmp))
            case _: raise ValueError
        magsum = 0.0
        for interval in search_intervals:
            _, mag = find_order(spos, *interval, order_search_density)
            magsum += mag
        scores[i] = magsum/np.sqrt(len(spos)) if len(spos) > 0 else 0.0
    
    i_best_score = np.argmax(scores)

    return thresholds[i_best_score], scores[i_best_score]


def estimate_signature(data: Signal,
                       m: int,
                       x: npt.ArrayLike = None,
                       idx: npt.ArrayLike = None,
                       weights: npt.ArrayLike = None,
                       max_error: int = 0) -> npt.ArrayLike:

    """Estimates the fault signature given a set of
    (possibly inaccurate) event locations x and their weights."""
    data = copy.deepcopy(data)

    if idx is None:
        if x is None:
            raise ValueError("Either indices idx or locations x must be specified.")
        else:
            sampind = np.array(data.idx_closest(x))
    else:
        sampind = np.array(idx)

    # remove the signature windows that fall partially outside the signal
    idx_keep = np.where((sampind >= 0) & (sampind + m < len(data)))
    sampind = sampind[idx_keep]
    weights = weights[idx_keep]

    totweight = sum(weights)

    if max_error == 0:
        #slices = (data.y[n:n+m] for n in sampind)
        #h = np.sum(x*w for x, w in zip(slices, weights))/totweight
        
        running_sum = data.y[sampind[0]: sampind[0] + m]
        for i in range(1, len(sampind)):
            idx = sampind[i]
            idx0 = max(0, idx)
            idx1 = min(idx+m, len(data.y))
            sigwin_new = data.y[idx0: idx1]
            if len(sigwin_new) == m:
                running_sum += sigwin_new * weights[i]

        h = running_sum/totweight

    else:
        running_sum = data.y[sampind[0]: sampind[0] + m]

        for i in range(1, len(sampind)):
            idx = sampind[i]
            idx0 = max(0, idx-max_error)
            idx1 = min(idx+m+max_error, len(data.y))
            sigwin = data.y[idx0: idx1]
            corr = np.correlate(sigwin, running_sum)
            shift = np.argmax(corr) - max_error
            sigwin_new = data.y[idx+shift: idx+shift+m]
            if len(sigwin_new) == m:
                running_sum += sigwin_new * weights[i]

        h = running_sum/totweight
    return h
