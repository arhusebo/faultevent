from dataclasses import dataclass
from collections.abc import Sequence
import itertools
from typing import Iterator, TypeVar, Optional
import numpy as np
import numpy.typing as npt

from . import Signal

def detect_func(thr, hys):

    def _detect(d_prev: bool, x_new: bool) -> bool:
        return (hys if d_prev else thr) < x_new

    return _detect


def detect(y, y1, y0):
    """Performs detection with hysteresis.
    """
    if y0==None: y0 = y1
    d_iter = itertools.accumulate(y, detect_func(y1, y0), initial=y[0]>y1)
    return list(d_iter)[1:]


# Alternative detection using numpy ufunc:
#def detect(y, y1, y0):
#        dfun = detect_func(y1, y0)
#        ufunc = np.frompyfunc(dfun, 2, 1)
#        d = ufunc.accumulate(y, dtype=np.dtype(bool))
#        d[0] = y[0]<=y1
#        return d


def regions_from_sequence(seq: Sequence[bool]):
    """ Returns pairs (start, stop) of detected regions """
    dd = np.diff(np.array(seq, dtype=int))
    on = np.argwhere(dd>0)+1
    off = np.argwhere(dd<0)+1
    if len(on) <= 1 or len(off) <= 1:
        return []
    if off[0] <= on[0]: off = off[1:]
    if off[-1] <= on[-1]: on = on[:-1]
    reg = np.hstack((on, off))
    return reg

SelfComparison = TypeVar("SelfComparison", bound="Comparison")

@dataclass(frozen=True)
class Comparison:
    """Class for comparing a signal to a threshold
    
    Attributes:
    data: the data to compare
    state: the state indicating where the signal is above threshold
    regions: the regions where the signal is above the threshold
    threshold: the threshold
    hysteresis: optional hysteresis threshold
    """
    data: Signal
    state: Signal
    regions: Sequence[tuple[float, float]]
    threshold: float
    hysteresis: Optional[float]
    empty: bool

    @classmethod
    def from_comparator(cls, data: Signal, threshold: float,
                hysteresis: float = None) -> SelfComparison:
        det = detect(data.y, threshold, hysteresis)
        state = Signal(det, data.x)
        reg = regions_from_sequence(det)
        empty = len(reg)==0
        return cls(data, state, reg, threshold, hysteresis, empty)

    def signal_segments(self) -> Iterator[Signal]:
        """Returns signal segments above the threshold"""
        for ia, ib in self.regions:
            yield self.data[ia:ib]


def energy_detector_location_estimates(comparison: Comparison):
    """Given a comparison, returns locations of events as detected using
    an energy detector"""
    return [sigseg.x[0] for sigseg in comparison.signal_segments()]

def matched_filter_location_estimates(comparison: Comparison)\
        -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Given a comparison, returns the locations and test statistic
    magnitudes of events as detected using a matched filter detector"""
    loclist = []
    maglist = []
    for sigseg in comparison.signal_segments():
        idx = np.argmax(sigseg.y)
        loc = sigseg.x[idx]
        mag = sigseg.y[idx]
        loclist.append(loc)
        maglist.append(mag)
    return np.asarray(loclist), np.asarray(maglist)
