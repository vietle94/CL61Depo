import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import iqr
from scipy.signal import ShortTimeFFT, get_window


def analyze_noise(noise):
    fft = ShortTimeFFT(
        win=get_window("hamm", 256),
        hop=128,
        fs=10,
    )

    f_noise = fft.stft(noise.values, axis=0)
    mag_noise = np.abs(f_noise)
    phase_noise = np.angle(f_noise)

    mag_noise_median = median_filter(mag_noise, size=(21, 1, 1))
    residual = mag_noise - mag_noise_median
    iqr_value = iqr(residual, axis=0)
    # Mark as outliers using IQR rule (e.g., 1.5x IQR)
    outlier_mask = residual > (1.5 * iqr_value)
    outlier_mask = np.max(outlier_mask, axis=2)
    outlier_mask_noise = np.repeat(
        outlier_mask[:, :, np.newaxis], f_noise.shape[2], axis=2
    )

    mag_noise_filtered = np.where(outlier_mask_noise, mag_noise_median, mag_noise)
    mag_noise_filtered[:5, :, :] = mag_noise[
        :5, :, :
    ]  # keep the first frequency unchanged
    f_noise_filtered = mag_noise_filtered * np.exp(1j * phase_noise)
    noise_filtered = np.real(
        fft.istft(f_noise_filtered, k1=noise.shape[0], f_axis=0, t_axis=2)
    )
    return noise_filtered, outlier_mask


def denoise_fft(ppol, outlier_mask):
    fft = ShortTimeFFT(
        win=get_window("hamm", 256),
        hop=128,
        fs=10,
    )
    f_signal = fft.stft(ppol.values, axis=0)
    mag_signal = np.abs(f_signal)
    phase_signal = np.angle(f_signal)

    mag_signal_median = median_filter(mag_signal, size=(21, 1, 1))
    outlier_mask_signal = np.repeat(
        outlier_mask[:, :, np.newaxis], f_signal.shape[2], axis=2
    )
    mag_signal_filtered = np.where(outlier_mask_signal, mag_signal_median, mag_signal)
    mag_signal_filtered[:5, :, :] = mag_signal[
        :5, :, :
    ]  # keep the first frequency unchanged
    f_signal_filtered = mag_signal_filtered * np.exp(1j * phase_signal)
    signal_filtered = np.real(
        fft.istft(f_signal_filtered, k1=ppol.shape[0], f_axis=0, t_axis=2)
    )
    return signal_filtered
