import pandas as pd
import requests
from cl61.fetch.utils import process_metadata
import xarray as xr
import numpy as np
import os
import importlib.resources
import glob

with importlib.resources.files("cl61.fetch").joinpath("cal_ref.npy").open("rb") as f:
    ref = np.load(f)


def fetch_categorize(site, start_date, product="categorize"):
    url = "https://cloudnet.fmi.fi/api/files"
    params = {
        "dateFrom": start_date,
        "dateTo": start_date,
        "site": site,
        "product": product,
    }
    metadata = requests.get(url, params).json()
    for row in metadata:
        print(row["filename"])
        res = requests.get(row["downloadUrl"])
        df = xr.open_dataset(res.content)
        return df


def fetch_lwc_cloud(path):
    files = glob.glob(path + "/*.nc")
    for file in files:
        file_date = file.split("/")[-1].split(".")[0]
        idate = file_date[:4] + "-" + file_date[4:6] + "-" + file_date[6:]
        file_site = files[0].split("/")[-2].lower()
        lwc = fetch_categorize(file_site, idate, "lwc")
        if lwc is None:
            continue
        df = xr.open_dataset(file)
        lwc = lwc.where(lwc.lwc_retrieval_status == 1)
        lwc = lwc.reindex(
            time=df.time, method="nearest", tolerance=np.timedelta64(30, "m")
        )
        lwc_adiabatic_full = lwc.lwc.differentiate("height")
        mask = lwc_adiabatic_full.notnull()
        if (~mask).all():
            continue

        first_valid_idx = mask.argmax("height")
        has_valid = mask.any("height")

        lwc_adiabatic = lwc_adiabatic_full.isel(height=first_valid_idx).where(has_valid)
        result = xr.Dataset(
            {
                "lwc_adiabatic": (("time"), lwc_adiabatic.data),
            },
            coords={"time": lwc_adiabatic.time.values},
        )
        result.to_netcdf(path + f"/lwc/{file_date}_lwc.nc")
