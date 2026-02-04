import requests
import xarray as xr
import glob
import numpy as np


def fetch_model(site, start_date, end_date, save_path):
    """Download model data"""
    url = "https://cloudnet.fmi.fi/api/model-files"
    params = {
        "dateFrom": start_date,
        "dateTo": end_date,
        "site": site,
    }
    print(params)
    metadata = requests.get(url, params).json()
    print(metadata)
    for row in metadata:
        res = requests.get(row["downloadUrl"])
        df = xr.open_dataset(res.content)
        return df


def fetch_model_cloud(path):
    files = glob.glob(path + "/*.nc")
    for file in files:
        file_date = file.split("/")[-1].split(".")[0]
        idate = file_date[:4] + "-" + file_date[4:6] + "-" + file_date[6:]
        file_site = files[0].split("/")[-2].lower()
        model = fetch_model(
            file_site,
            idate,
            idate,
            path,
        )
        df = xr.open_dataset(file)
        model = model[["temperature", "q", "height"]]
        model = model.reindex(
            time=df.time, method="nearest", tolerance=np.timedelta64(30, "m")
        )
        height_delta = np.abs(model["height"] - df.cloud_base)
        height_delta = height_delta.where(height_delta < 300)
        closest_idx = height_delta.argmin(dim="level")
        model.temperature.isel(level=closest_idx)
        result = xr.Dataset(
            {
                "T": (("time"), model.temperature.isel(level=closest_idx).data),
                "q": (("time"), model.q.isel(level=closest_idx).data),
            },
            coords={"time": model.time.values},
        )
        result.to_netcdf(path + f"/model/{file_date}_model.nc")
