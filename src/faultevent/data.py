from abc import ABC, abstractmethod
from dataclasses import dataclass

from .signal import Signal

@dataclass
class Measurement:
    """Dataclass containing vibration and shaft position measurements"""
    vib: Signal
    pos: Signal


class DataLoader(ABC):
    """Abstract data loader class meant to be subclassed"""
    @abstractmethod
    def __getitem__(self, idx) -> Measurement:
        pass