import pandas as pd
import requests
from cl61.fetch.utils import process_metadata, response, process_metadata_child
import xarray as xr
import io
import numpy as np
import os
import importlib.resources

with importlib.resources.files("cl61.fetch").joinpath("cal_ref.npy").open("rb") as f:
    ref = np.load(f)


def fetch_processing(func, site, start_date, end_date, save_path):
    url = "https://cloudnet.fmi.fi/api/raw-files"
    pr = pd.period_range(start=start_date, end=end_date, freq="D")

    for i in pr:
        if os.path.exists(save_path + i.strftime("%Y%m%d") + ".csv"):
            continue
        idate = i.strftime("%Y-%m-%d")
        print(idate)
        params = {
            "dateFrom": idate,
            "dateTo": idate,
            "site": site,
            "instrument": "cl61d",
        }
        metadata = requests.get(url, params).json()
        if not metadata:
            continue
        result = process_metadata(metadata, func)
        print("saving")
        result.to_csv(save_path + i.strftime("%Y%m%d") + ".csv", index=False)


def noise(res):
    from cl61.func.noise import noise_detection

    df, _ = response(res)
    df = noise_detection(df)
    df_noise = df.where(df["noise"])
    df_noise["p_pol"] = df_noise["p_pol"] / (df["range"] ** 2)
    df_noise["x_pol"] = df_noise["x_pol"] / (df["range"] ** 2)
    grp_range = df_noise[["p_pol", "x_pol"]].groupby_bins(
        "range", [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000]
    )

    grp_mean = grp_range.mean(dim=["range", "time"])
    grp_std = grp_range.std(dim=["range", "time"])

    result_ = pd.DataFrame(
        {
            "datetime": df.time[0].values,
            "co_mean": grp_mean["p_pol"],
            "co_std": grp_std["p_pol"],
            "cross_mean": grp_mean["x_pol"],
            "cross_std": grp_std["x_pol"],
            "range": grp_mean.range_bins.values.astype(str),
        }
    )
    return result_


def housekeeping(res):
    _, diag_ = response(res)
    return diag_


def fetch_attrs(func, site, start_date, end_date, save_path):
    url = "https://cloudnet.fmi.fi/api/raw-files"
    pr = pd.period_range(start=start_date, end=end_date, freq="D")
    for i in pr:
        idate = i.strftime("%Y-%m-%d")
        print(idate)
        params = {
            "dateFrom": idate,
            "dateTo": idate,
            "site": site,
            "instrument": "cl61d",
        }
        metadata = requests.get(url, params).json()
        if not metadata:
            continue
        for row in metadata:
            if "live" in row["filename"]:
                result = process_metadata_child(row, func)
                if result is not False:
                    print("saving")
                    result.to_csv(
                        save_path + i.strftime("%Y%m%d") + ".csv", index=False
                    )
                break


def integration(res):
    df = xr.open_dataset(io.BytesIO(res.content))
    integration_name = [
        "time between consecutive profiles in seconds",
        "profile_interval_in_seconds",
    ]
    result = pd.DataFrame(
        {
            "datetime": [df.time[0].values],
            "integration_t": [
                np.median(
                    np.diff(df["time"].values).astype("timedelta64[ns]")
                    / np.timedelta64(1, "s")
                )
            ],
        }
    )
    for attr in integration_name:
        if attr in df.attrs:
            result["integration"] = df.attrs[attr]
    return result


def sw_version(res):
    df = xr.open_dataset(io.BytesIO(res.content))
    if "sw_version" in df.attrs:
        return pd.DataFrame(
            {"datetime": [df.time[0].values], "sw_version": [df.attrs["sw_version"]]}
        )
    return False


def convolve_1d(arr, kernel):
    return np.convolve(arr, kernel, mode="same")


def cloud_calibration(res):
    df, _ = response(res)
    df = df.sel(range=slice(100, 5000))
    result = xr.apply_ufunc(
        convolve_1d,
        df.beta_att,
        input_core_dims=[["range"]],  # Apply along 'lon'
        kwargs={"kernel": ref},
        output_core_dims=[["range"]],
        vectorize=True,
    )

    result2 = xr.apply_ufunc(  # calculate sum beta in-cloud
        convolve_1d,
        df.beta_att,
        input_core_dims=[["range"]],  # Apply along 'lon'
        kwargs={"kernel": ref > 1e-5},
        output_core_dims=[["range"]],
        vectorize=True,
    )
    range_max = result.idxmax(dim="range")
    cross_correlation = result.max(dim="range").values

    sum_beta = df.beta_att.sum(dim="range")
    cloud_beta_sum = result2.sel(range=xr.DataArray(range_max.values, dims="time"))
    cloud_beta_percent = cloud_beta_sum / sum_beta

    cloud_mask = (cloud_beta_percent > 0.9) & (cross_correlation > 3e-7)
    df = df.isel(time=cloud_mask.values)
    range_max = range_max.isel(time=cloud_mask)

    df_cloud = df.where(df.range < range_max + 76.8)
    df_cloud = df_cloud.where(
        df_cloud.beta_att.T / df_cloud.beta_att.max(dim="range") > 0.05
    )
    df_cloud["depo"] = df_cloud["x_pol"] / df_cloud["p_pol"]
    cloud_base = df_cloud["depo"].idxmin(dim="range")

    df_incloud = df_cloud.where(df_cloud.range > cloud_base)

    n_time, n_range = df_incloud.depo.shape

    # -----------------------------
    # Step 1: Find first non-NaN index per time
    # -----------------------------
    mask = ~np.isnan(df_incloud.depo.values)
    first_idx = mask.argmax(axis=1)  # shape (time,)

    # -----------------------------
    # Step 2: Create shifted array
    # -----------------------------

    # Compute indices for broadcasting
    range_idx = np.arange(100)  # 0..max_len-1
    src_idx = first_idx[:, None] + range_idx

    # Assign values using broadcasting
    shifted = df_incloud.depo.values[np.arange(n_time)[:, None], src_idx]

    return pd.DataFrame(
        {
            "datetime": result.time.values,
            "cross_correlation": result.max(dim="range").values,
            "etaS": 1 / (2 * sum_beta * 4.8),
            "range": range_max,
            "cloud_beta_percent": cloud_beta_sum / sum_beta,
        }
    )


def cloud_detection(res):
    df = cloud_calibration(res)
    df["range"] = df["range"] + 76.8
    df = df[df["cloud_beta_percent"] > 0.9]

    return pd.DataFrame(
        {
            "datetime": df["datetime"],
            "range": df["range"],
        }
    )


# def cloud_detection(res):
#     """
#     Convolve x with ref, find max index per time,
#     then compute convolution with ref > 1e-5 at that index.
#     Returns a summary DataArray with time, value at max index in result1, and result2.

#     Parameters
#     ----------
#     x : xr.DataArray, shape (T, R)
#         Input signal with dims ('time', 'range')

#     Returns
#     -------
#     summary : xr.DataArray, shape (T,), with fields 'time', 'result1_max', 'result2'
#     """
#     with (
#         importlib.resources.files("cl61.fetch").joinpath("cal_ref.npy").open("rb") as f
#     ):
#         ref = np.load(f)
#     x, _ = response(res)
#     x = x.sel(range=slice(100, 5000))
#     x_vals = x.beta_att.values
#     T, R = x_vals.shape

#     # Step 1: Convolve with ref using real FFT
#     N1 = R + ref.size - 1
#     X = np.fft.rfft(x_vals, n=N1, axis=1)
#     ref = np.fft.rfft(ref, n=N1)
#     conv1 = np.fft.irfft(X * ref, n=N1, axis=1)

#     # 'same' length along range
#     start1 = (ref.size - 1) // 2
#     end1 = start1 + R
#     result1 = conv1[:, start1:end1]

#     # Max index along range and value at max
#     idx = np.argmax(result1, axis=1)
#     result1_max = result1[np.arange(T), idx]

#     # Map max index to range coordinate
#     range_max = x["range"].values[idx]

#     # Step 2: Convolve with ref > 1e-5 only at max index
#     half = (ref > 1e-5).size // 2
#     result2 = np.zeros(T)

#     for t in range(T):
#         k = idx[t]
#         r_start = max(0, k - half)
#         r_end = min(R, k + half + 1)
#         y_start = half - (k - r_start)
#         y_end = y_start + (r_end - r_start)
#         result2[t] = np.dot(x_vals[t, r_start:r_end], (ref > 1e-5)[y_start:y_end])

#     # Build summary xarray DataArray
#     # summary = xr.Dataset(
#     #     {
#     #         "result1_max": ("time", result1_max),
#     #         "result2": ("time", result2),
#     #         "range_max": ("time", range_max),
#     #     },
#     #     coords={"time": x["time"]},
#     # )

#     # return summary
#     return result1
