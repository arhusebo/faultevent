from dataclasses import dataclass
from collections.abc import Sequence
from typing import Iterator, TypeVar, Optional
import numpy as np
import numpy.typing as npt

from . import Signal

def detect_gen(y: npt.ArrayLike,
               y1: float, y0: float=None) -> Iterator[bool]:
    """Generator yielding detections on given sequence
    
    Arguments:
    y -- sequence to evaluate
    y1 -- detection threshold

    Keyword arguments:
    y0 -- Optional hysteresis threshold. y0 < y1.
    """
    if y0:
        if y0>=y1: raise ValueError("y0 must be lower than y1")
    dn = False
    for yn in y:
        if not dn and yn >= y1: dn = True
        elif dn and yn < (y0 if y0 else y1): dn = False
        yield dn

def detect(y: npt.ArrayLike, y1, y0=None):
    """ Simple detection with optional hysteresis """
    gen = detect_gen(y, y1, y0)
    det = np.fromiter(gen, dtype=bool, count=len(y))
    return det

def regions_from_generator(gen: Iterator[bool]):
    """ Returns the pairs (start, stop) for regions detected by
    specified detection generator"""
    reg = []
    dp = next(gen)
    on = False
    for n, dn in enumerate(gen):
        diff = int(dn) - int(dp)
        if on:
            if diff == -1:
                b = n+1
                on = False
                reg.append((a, b))
        else:
            if diff == 1:
                a = n+1
                on = True
    return reg


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

def matched_filter_location_estimates(comparison: Comparison):
    """Given a comparison, returns the locations of events as detected
    using a matched filter detector"""
    loclist = []
    for sigseg in comparison.signal_segments():
        idx = np.argmax(sigseg.y)
        loc = sigseg.x[idx]
        loclist.append(loc)
    return loclist
