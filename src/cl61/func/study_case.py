import numpy as np
import glob
import pandas as pd
import xarray as xr


def process_raw(file_dir, t1, t2):
    files = glob.glob(file_dir + "*.nc")
    df_sample_signal = xr.open_mfdataset(files)
    df_sample_diag = xr.open_mfdataset(files, group="monitoring")
    df_sample_signal = df_sample_signal.sel(
        time=slice(pd.to_datetime(t1), pd.to_datetime(t2))
    )
    df_sample_diag = df_sample_diag.sel(
        time=slice(pd.to_datetime(t1), pd.to_datetime(t2))
    )

    df_sample_diag = df_sample_diag.reindex(
        time=df_sample_signal.time.values, method="nearest", tolerance="8s"
    )
    df_sample_diag = df_sample_diag.dropna(dim="time")
    df_sample = df_sample_diag.merge(df_sample_signal, join="inner")

    df_sample["ppol_r"] = df_sample.p_pol / (df_sample.range**2)
    df_sample["xpol_r"] = df_sample.x_pol / (df_sample.range**2)
    df_sample["internal_temperature_bins"] = np.floor(df_sample.internal_temperature)
    df_sample = df_sample.isel(range=slice(1, None))
    return df_sample


def background_noise(site, date):
    file_dir = f"/media/viet/CL61/{site}/Noise/*.csv"
    files = glob.glob(file_dir)
    file = [x for x in files if date in x][0]
    df = pd.read_csv(file)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"] > "2000-01-01"]
    df = df.groupby("range").get_group("(10000, 12000]")
    return df
