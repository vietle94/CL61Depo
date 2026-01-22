import numpy as np


def mid_range(x):
    bin = np.median(np.diff(x.values))
    return x + bin


def smooth(data, window_size):
    """Simple moving average smoothing."""
    if window_size < 3:
        return data
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, mode="same")
