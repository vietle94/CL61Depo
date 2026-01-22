import pandas as pd
import glob
import xarray as xr
import os.path

calibration = pd.read_csv("/media/viet/CL61/calibration/calibration.txt")
calibration = calibration.astype(str)
calibration["t1"] = pd.to_datetime(calibration["date1"] + " " + calibration["time1"])
calibration["t2"] = pd.to_datetime(calibration["date2"] + " " + calibration["time2"])


def merge_calibration(site):
    cal = calibration[calibration["site"] == site]
    for i, row in cal.iterrows():
        t1 = row["t1"]
        t2 = row["t2"]
        save_date = t1.strftime("%Y%m%d")
        save_name = f"/media/viet/CL61/calibration/{site}/merged/{save_date}.nc"
        if os.path.isfile(save_name):
            continue
        print(save_name)
        file_path = glob.glob(
            f"/media/viet/CL61/calibration/{site}/{row['file']}"
        )  # Select data in a whole day

        df_path = pd.DataFrame({"path": file_path})
        df_path["date"] = df_path["path"].str.split("_").str[1]
        df_path["time"] = df_path["path"].str.split("_").str[2].str.split(".").str[0]
        df_path["datetime"] = pd.to_datetime(df_path["date"] + " " + df_path["time"])

        path_cal = df_path.loc[
            (df_path["datetime"] > t1) & (df_path["datetime"] < t2), "path"
        ].values

        df = xr.open_mfdataset(path_cal)
        df = df.isel(range=slice(1, None))
        df_cal = df.sel(time=slice(t1, t2))
        df_cal = df_cal[["p_pol", "x_pol"]]

        if (site == "hyytiala") & (t1 < pd.to_datetime("2024-01-01")):
            df_diag = xr.open_mfdataset(
                path_cal, group="monitoring", preprocess=hyytiala_preprocess
            )
        elif site == "vehmasmaki":
            df_diag = xr.open_mfdataset(
                path_cal, group="monitoring", preprocess=vehmasmaki_preprocess
            )
        else:
            df_diag = xr.open_mfdataset(path_cal, group="monitoring")
        df_diag_cal = df_diag.sel(time=slice(t1, t2))
        df_diag_cal = df_diag_cal[["laser_temperature", "internal_temperature"]]

        df_diag_cal = df_diag_cal.reindex(
            time=df_cal.time.values, method="nearest", tolerance="8s"
        )
        df_diag_cal = df_diag_cal.dropna(dim="time")
        df = df_diag_cal.merge(df_cal, join="inner")
        df.to_netcdf(save_name)
        print("Saved")


def hyytiala_preprocess(x):
    x["internal_temperature"] = (["time"], [x.attrs["internal_temperature"]])
    x["laser_temperature"] = (["time"], [x.attrs["laser_temperature"]])
    return x.assign_coords(time=[pd.to_datetime(float(x.attrs["Timestamp"]), unit="s")])


def vehmasmaki_preprocess(x):
    return x.sortby("time")


if __name__ == "__main__":
    for site in calibration["site"].unique():
        merge_calibration(site)
