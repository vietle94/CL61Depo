import numpy as np
import pywt


def noise_detection(df, wavelet="bior1.1"):
    # pre processing
    df = df.isel(range=slice(1, None))
    co = df["p_pol"] / (df["range"] ** 2)
    profile = co.mean(dim="time")

    # wavelet transform
    n_pad = (len(profile) // 2**7 + 3) * 2**7 - len(profile)
    coeff = pywt.swt(
        np.pad(
            profile,
            (n_pad - n_pad // 2, n_pad // 2),
            "constant",
            constant_values=(0, 0),
        ),
        wavelet,
        trim_approx=True,
        level=7,
    )
    minimax_thresh = (
        np.median(np.abs(coeff[1]))
        / 0.6745
        * (0.3936 + 0.1829 * np.log2(len(coeff[1])))
    )
    coeff[1:] = (
        pywt.threshold(i, value=minimax_thresh, mode="hard") for i in coeff[1:]
    )
    filtered = pywt.iswt(coeff, wavelet)
    filtered = filtered[(n_pad - n_pad // 2) : len(profile) + (n_pad - n_pad // 2)]

    # half the minimax threshold
    df["noise"] = (["range"], filtered < 0.5 * minimax_thresh)
    df["p_pol_smooth"] = (["range"], filtered)
    df.attrs["minimax_thresh"] = minimax_thresh
    return df


def noise_filter(profile, wavelet="bior1.5"):
    # wavelet transform
    n_pad = (len(profile) // 2**7 + 3) * 2**7 - len(profile)
    coeff = pywt.swt(
        np.pad(
            profile,
            (n_pad - n_pad // 2, n_pad // 2),
            "constant",
            constant_values=(0, 0),
        ),
        wavelet,
        trim_approx=True,
        # level=7,
        level=8,
    )
    minimax_thresh = (
        np.median(np.abs(coeff[1]))
        / 0.6745
        * (0.3936 + 0.1829 * np.log2(len(coeff[1])))
    )
    coeff[1:] = (
        pywt.threshold(i, value=minimax_thresh, mode="hard") for i in coeff[1:]
    )
    filtered = pywt.iswt(coeff, wavelet)
    filtered = filtered[(n_pad - n_pad // 2) : len(profile) + (n_pad - n_pad // 2)]
    return filtered
