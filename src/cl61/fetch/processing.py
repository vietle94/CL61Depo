import pandas as pd
import requests
from cl61.fetch.utils import process_metadata
import xarray as xr
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
        if not result:
            print("no cloud day")
            return None
        print("saving")
        result.to_netcdf(save_path + i.strftime("%Y%m%d") + ".nc")


def convolve_1d(arr, kernel):
    return np.convolve(arr, kernel, mode="same")


def liquid_cloud_detection(res):
    df = xr.open_dataset(res.content)
    df = df.sel(range=slice(100, 5000))
    df["depo"] = df["x_pol"] / df["p_pol"]

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
    if not cloud_mask.values.any():
        print("no cloud")
        return None
    df = df.isel(time=cloud_mask.values)
    range_max = range_max.isel(time=cloud_mask) + 76.8

    df_cloud = df.where(df.range < range_max + 400)
    df_cloud = df_cloud.where(
        df_cloud.beta_att.T / df_cloud.beta_att.max(dim="range") > 0.05
    )
    cloud_base = df_cloud["depo"].idxmin(dim="range")

    df_incloud = df.where(df.range > cloud_base)

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
    depo_save = df_incloud.depo.values[np.arange(n_time)[:, None], src_idx]
    beta_save = df_incloud.beta_att.values[np.arange(n_time)[:, None], src_idx]
    summary = xr.Dataset(
        {
            "depo": (("time", "range"), depo_save),
            "beta": (("time", "range"), beta_save),
            "cloud_base": ("time", cloud_base.data),
        },
        coords={"time": df_incloud.time.values, "range": np.arange(100)},
    )
    return summary
