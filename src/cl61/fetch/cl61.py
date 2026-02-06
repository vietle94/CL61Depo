import pandas as pd
import requests
from cl61.fetch.utils import process_metadata
import xarray as xr
import numpy as np
import os
import importlib.resources

with importlib.resources.files("cl61.fetch").joinpath("cal_ref.npy").open("rb") as f:
    ref = np.load(f)


def fetch_cl61(site, start_date):
    url = "https://cloudnet.fmi.fi/api/files"
    params = {
        "dateFrom": start_date,
        "dateTo": start_date,
        "site": site,
        "instrument": "cl61d",
    }
    metadata = requests.get(url, params).json()
    for row in metadata:
        print(row["filename"])
        res = requests.get(row["downloadUrl"])
        df = xr.open_dataset(res.content)
        return df
