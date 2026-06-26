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
                   hysteresis: float | None = None,
                   dettype: Literal["mf", "ed"] = "mf",
                   order_search_density = 1000) -> tuple[float, float]:
    """Evaluates a metric over multiple thresholds and returns the
    best threshold and the score metric"""
    if thresholds is None: thresholds = np.linspace(0, 5*np.std(data.y), n)

    def score(thr):
        hys = None if hysteresis is None else hysteresis*thr
        match dettype:
            case "mf":
                data_env = Signal(abs(data.y), data.x, uniform_samples=data.uniform_samples)
                data_real = Signal(data.y.real, data.x, uniform_samples=data.uniform_samples)
                cmp = Comparison.from_comparator(data_env, thr, hys)
                cmp_ = Comparison(
                        data=data_real,
                        state=cmp.state,
                        regions=cmp.regions,
                        threshold=cmp.threshold,
                        hysteresis=cmp.hysteresis,
                        empty=cmp.empty,)
                eoi = matched_filter_location_estimates(cmp_)
            case "ed":
                cmp = Comparison.from_comparator(data, thr, hys)
                eoi = energy_detector_location_estimates(cmp)
            case _: raise ValueError
        if len(eoi)==0:
            return 0.0
        return sum((find_order(data.x[eoi], *interval, order_search_density)[1]
                      for interval in search_intervals))/np.sqrt(len(eoi))

    scores = [score(thr) for thr in thresholds]
    idx = np.argmax(scores)

    return thresholds[idx], scores[idx]


def estimate_signature_old(data: Signal,
                           length: int,
                           x: npt.ArrayLike | None = None,
                           indices: npt.ArrayLike | None = None,
                           weights: npt.ArrayLike | None = None,
                           max_error: int = 0) -> npt.ArrayLike:

    """Estimates the fault signature given a set of
    (possibly inaccurate) event locations x and their weights."""

    data = copy.deepcopy(data)

    has_weights = weights is not None
    has_x = x is not None
    has_indices = indices is not None

    if (has_indices and has_x) or not(has_indices or has_x):
        raise ValueError("Either indices idx or locations x must be specified.")

    if has_indices:
        sampind = np.array(indices)
    elif has_x:
        sampind = np.array(data.idx_closest(x))

    if (has_indices and has_weights) and len(indices)!=len(weights):
        raise ValueError("indices and weights must be of same length")
    else:
        weights = np.ones_like(sampind, dtype=float)

    # remove the signature windows that fall partially outside the signal
    idx_keep = np.where((sampind >= 0) & (sampind + length < len(data)))
    sampind = sampind[idx_keep]

    if (has_x and has_weights) and len(x)!=len(weights):
        weights = weights[idx_keep]
        raise ValueError("x and weights must be of same length")
    else:
        weights = np.ones_like(sampind, dtype=float)
        idx_sorted = np.argsort(weights) # ascending weights
        sampind[::-1] = sampind[idx_sorted]
        weights[::-1] = weights[idx_sorted]

    totweight = sum(weights)

    if max_error == 0:
        #slices = (data.y[n:n+length] for n in sampind)
        #h = np.sum(x*w for x, w in zip(slices, weights))/totweight
        
        running_sum = data.y[sampind[0]: sampind[0] + length]
        for i in range(1, len(sampind)):
            idx = sampind[i]
            idx0 = max(0, idx)
            idx1 = min(idx+length, len(data.y))
            sigwin_new = data.y[idx0: idx1]
            if len(sigwin_new) == length:
                running_sum += sigwin_new * weights[i]

        h = running_sum/totweight

    else:
        running_sum = data.y[sampind[0]: sampind[0] + length]

        for i in range(1, len(sampind)):
            idx = sampind[i]
            idx0 = max(0, idx-max_error)
            idx1 = min(idx+length+max_error, len(data.y))
            sigwin = data.y[idx0: idx1]
            corr = np.correlate(sigwin, running_sum)
            shift = np.argmax(corr) - max_error
            sigwin_new = data.y[idx+shift: idx+shift+length]
            if len(sigwin_new) == length:
                running_sum += sigwin_new * weights[i]

        h = running_sum/totweight
    return h


def estimate_signature(signal: Signal,
                       length: int,
                       indices: npt.ArrayLike,
                       weights: npt.ArrayLike = None,) -> npt.ArrayLike:
    if weights is None:
        weights = np.ones_like(indices, dtype=float)
    if len(indices)!=len(weights):
        raise ValueError("indices and weights must be of the same length")
    sum_ = np.zeros((length,), dtype=float)
    for w, i in zip(weights, indices):
        if i<0 or i+length>len(signal):
            continue
        sum_ += w*signal.y[i:i+length]
    return sum_/sum(weights)


def scm(signal: npt.ArrayLike, length: int, maxerror: int,
        eoi: npt.ArrayLike, weights: npt.ArrayLike,):
    """Sequential cross-correlation maximization.

    From a set of innaccurate signature occurrence indices (EOIs),
    estimate the signature.
    """

    psignal = np.pad(signal, maxerror+length)
    eoi_remain = eoi+maxerror+length # shift indices to account for padding
    
    # Initial signature estimate
    # Find the two "best" signature occurences, i.e. the two that maximizes
    # their cross-correlation maximum
    result_best = {"score": 0.0}
    for i, eoi_ in enumerate(eoi_remain):
        template = psignal[eoi_:eoi_+length]
        result = best_match(template, psignal, maxerror, np.delete(eoi_remain, i))
        if result["score"]>result_best["score"]:
            result_best = result
            i_best = i
            sigest = weights[i]*template+weights[result["idx"]]*result["signature"]
    
    # remove indices of the two signature occurences that went into the initial estimate
    eoi_remain = np.delete(eoi_remain, [i_best, result_best["idx"]])

    # Subsequent signature estimates
    # Find the next "best" signature occurence, i.e. the one that maximizes
    # its cross-correlation maximum with the current estimate.
    # Updates the signature estimate.
    # Do this until no signature occurences remain
    while len(eoi_remain)>0:
        result = best_match(sigest, psignal, maxerror, eoi_remain)
        sigest += weights[result["idx"]]*result["signature"]
        eoi_remain = np.delete(eoi_remain, result["idx"])

    return sigest/sum(weights)


def best_match(template, signal, maxerror, eoi):
    """Return, from an array of EOIs, the signature occurrence that best
    matches the given template"""
    corr_best = -np.inf
    for i, eoi_ in enumerate(eoi):
        padded = signal[eoi_-maxerror:eoi_+len(template)+maxerror]
        corr = np.correlate(a=padded,
                            v=template,
                            mode="valid")
        eoi_shift_max = np.argmax(corr)
        corr_max = corr[eoi_shift_max]
        if corr_max > corr_best:
            corr_best = corr_max
            i_best = i
            eoi_best = eoi_-maxerror+eoi_shift_max
    return {
        "idx": i_best,
        "signature": signal[eoi_best:eoi_best+len(template)],
        "score": corr_best,}
