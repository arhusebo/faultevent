import copy
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

def best_threshold(data: Signal, orda, ordb,
                   thrs: npt.ArrayLike = None,
                   n=10, hys=.2, dettype="mf") -> float:
    """Evaluates a metric over multiple thresholds and returns the
    best threshold"""
    if thrs==None: thrs = np.linspace(0, 5*np.std(data.y), n)
    scores = np.zeros_like(thrs, dtype=float)
    for i, thr in enumerate(thrs):
        cmp = Comparison.from_comparator(data, thr, hys*thr)
        if dettype=="mf": spos, _ = np.asarray(matched_filter_location_estimates(cmp))
        elif dettype=="ed": spos = np.asarray(energy_detector_location_estimates(cmp))
        else: raise ValueError
        _, mag = find_order(spos, orda, ordb)
        scores[i] = mag/np.sqrt(len(spos)) if len(spos) > 0 else 0.0

    return thrs[np.argmax(scores)]


def estimate_signature(data: Signal,
                       m: int,
                       weights: npt.ArrayLike | None = None,
                       idx: npt.ArrayLike | None = None,
                       x: npt.ArrayLike | None = None,
                       max_error: int = 0,
                       n0: int = 0) -> npt.ArrayLike:

    """Estimates the fault signature given a set of
    (possibly inaccurate) event locations x and their weights."""

    # take index into account for weights as well
    data = copy.deepcopy(data)
    if idx is None:
        if x is None:
            raise ValueError("Either indices idx or locations x must be specified.")
        idx = np.array(data.idx_closest(x)) + n0
    idx = idx[np.where((idx >= 0) & (idx + m < len(data)))]
    #idx = idx[np.where(idx >= 0)]
    #idx = idx[np.where(idx + m < len(data))]

    if weights is None:
        weights = np.ones_like(idx, dtype=float)
    else:
        weights = weights[idx]
    slices = np.array([np.arange(n, n+m) for n in idx])

    totweight = sum(weights)

    if max_error == 0:
        h = np.sum(np.asarray(data.y)[slices].T*weights, axis=1)/totweight
        #h = np.sum(np.transpose([data.y[i: i + m] for i in idx])*weights, axis=1)/totweight
    else:
        shifts = np.zeros_like(idx)
        running_sum = data.y[idx[0]: idx[0] + m]
        for i in range(1, len(idx)):
            signat = data.y[max(0, idx[i] - max_error): min(idx[i] + m + max_error, len(data.y))]
            corr = np.correlate(signat, running_sum)
            shift = np.argmax(corr) - max_error
            shifts[i] = shift
            signat_new = data.y[idx[i]+shift: idx[i]+shift+m]
            running_sum += signat_new * weights[i]

        h = running_sum/totweight
    return h
