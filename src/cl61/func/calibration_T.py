import numpy as np
import xarray as xr
import pywt
import glob


def noise_filter(profile, wavelet="bior1.5"):
    # wavelet transform
    level = 8
    n_pad = (len(profile) // 2**level + 4) * 2**level - len(profile)
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
        level=level,
    )

    uthresh = np.median(np.abs(coeff[1])) / 0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    filtered = pywt.iswt(coeff, wavelet)
    filtered = filtered[(n_pad - n_pad // 2) : len(profile) + (n_pad - n_pad // 2)]
    return filtered


def temperature_ref(df):
    # create temerature bins refernce for calibration file
    temp_range = np.arange(-10, 40)
    df["ppol_r"] = df.p_pol / (df.range**2)
    df["xpol_r"] = df.x_pol / (df.range**2)

    df_gr = df.groupby_bins("internal_temperature", temp_range, labels=temp_range[:-1])
    df_mean = df_gr.mean(dim="time", skipna=True)
    df_std = df_gr.std(dim="time", skipna=True)
    df_count = df_gr.count(dim="time")

    df_mean = df_mean.dropna(
        dim="internal_temperature_bins", how="all", subset=["ppol_r", "xpol_r"]
    )
    df_std = df_std.dropna(
        dim="internal_temperature_bins", how="all", subset=["ppol_r", "xpol_r"]
    )

    ppol = []
    xpol = []
    ppol_std = []
    xpol_std = []
    for t in df_mean.internal_temperature_bins.values:
        filtered = noise_filter_std(df_mean.sel(internal_temperature_bins=t)["ppol_r"])
        filtered[:50] = df_mean.sel(internal_temperature_bins=t)["ppol_r"][:50].values
        ppol.append(filtered)

        filtered = noise_filter_std(df_mean.sel(internal_temperature_bins=t)["xpol_r"])
        filtered[:50] = df_mean.sel(internal_temperature_bins=t)["xpol_r"][:50].values
        xpol.append(filtered)

        filtered = noise_filter_std(df_std.sel(internal_temperature_bins=t)["ppol_r"])
        # filtered[:50] = df_std.sel(internal_temperature_bins=t)["ppol_r"][:50].values
        ppol_std.append(filtered)

        filtered = noise_filter_std(df_std.sel(internal_temperature_bins=t)["xpol_r"])
        # filtered[:50] = df_std.sel(internal_temperature_bins=t)["xpol_r"][:50].values
        xpol_std.append(filtered)

    df_mean["ppol_ref"] = xr.DataArray(
        ppol, dims=["internal_temperature_bins", "range"]
    )
    df_mean["xpol_ref"] = xr.DataArray(
        xpol, dims=["internal_temperature_bins", "range"]
    )
    df_std["ppol_ref"] = xr.DataArray(
        ppol_std, dims=["internal_temperature_bins", "range"]
    )
    df_std["xpol_ref"] = xr.DataArray(
        xpol_std, dims=["internal_temperature_bins", "range"]
    )

    return df_mean, df_std, df_count


def noise_filter_std(profile, wavelet="bior2.2"):
    # wavelet transform
    level = 8
    n_pad = (len(profile) // 2**level + 4) * 2**level - len(profile)
    coeff = pywt.swt(
        np.pad(
            profile,
            (n_pad - n_pad // 2, n_pad // 2),
            "symmetric",
            # constant_values=(0, 0),
        ),
        wavelet,
        trim_approx=True,
        # level=7,
        level=level,
    )

    uthresh = np.median(np.abs(coeff[1])) / 0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    filtered = pywt.iswt(coeff, wavelet)
    filtered = filtered[(n_pad - n_pad // 2) : len(profile) + (n_pad - n_pad // 2)]
    return filtered


def temperature_calibration(path_in, path_out):
    files = glob.glob(path_in + "/*.nc")
    df = xr.open_mfdataset(files)
    df_mean, df_std, df_count = temperature_ref(df)
    df_mean.drop_vars(["p_pol", "x_pol"]).to_netcdf(path_out + "/calibration_mean.nc")
    df_std.drop_vars(["p_pol", "x_pol"]).to_netcdf(path_out + "/calibration_std.nc")
    df_count.drop_vars(["p_pol", "x_pol"]).to_netcdf(path_out + "/calibration_count.nc")
