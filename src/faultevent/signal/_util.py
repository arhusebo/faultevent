import numpy as np
import numpy.typing as npt

from . import Signal


def spectrum(signal: Signal):
    """Returns a signal object representing the DFT spectrum of input
    signal. If input signal is real, only the zero and real frequencies
    are computed."""
    if not signal.uniform_samples:
        raise ValueError("Signal must have evenly spaced samples")
    dx = signal.x[1]-signal.x[0]
    if np.iscomplexobj(signal.y):
        spec = np.fft.fft(signal.y)
        freq = np.fft.fftfreq(len(signal.y), dx)
    else:
        spec = np.fft.rfft(signal.y)
        freq = np.fft.rfftfreq(len(signal.y), dx)
    return Signal(spec, freq, uniform_samples=True)


def resample(x_to_eval: npt.ArrayLike,
             data: Signal,
             m: int = 100,
             batch_size: int = 10000) -> np.ndarray:
    """Resamples given signal at specified sequence of new sample x
    values using sinc resampling. Returns new y values.
    
    Arguments:
    x_to_eval -- sequence of resample x values
    data -- signal to resample
    
    Keyword arguments:
    m -- resampling filter window size
    batch_size -- size of resampling batches. Useful for long signals.
    """
    dx = data.x[1] - data.x[0]
    idx = data.idx_closest(x_to_eval)
    n_new_samples = len(x_to_eval)
    ymat = np.zeros((n_new_samples, m), dtype=float)
    sincwin = np.zeros((n_new_samples, m), dtype=float)
    ypad = np.pad(data.y, m//2)
    for n_ in range(m):
        n = idx + n_
        ymat[:,n_] = ypad[n]
        sincwin[:,n_] = np.sinc(x_to_eval/dx - n + m//2)
    
    n_batches = len(x_to_eval)//batch_size+1
    ynewbatches = []
    for i in range(n_batches):
        slc = slice(i*batch_size, min(n_new_samples, (i+1)*batch_size))
        ymatb = ymat[slc]
        sincwinb = sincwin[slc]
        ynewb = np.sum(ymatb*sincwinb, axis=-1)
        ynewbatches.append(ynewb)
    ynew = np.concatenate(ynewbatches)
    return ynew
