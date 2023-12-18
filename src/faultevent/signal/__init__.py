from ._core import Signal, SignalModel
from ._autoreg import ARModel
from ._util import spectrum, resample
from ._detector import Detector, EnergyDetector, MatchedFilterDetector,\
    MatchedFilterEnvelopeDetector
from ._comparator import Comparison,\
    energy_detector_location_estimates, matched_filter_location_estimates