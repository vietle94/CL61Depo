import requests
import xarray as xr
import numpy as np


def fetch_halo(site, start_date):
    url = "https://cloudnet.fmi.fi/api/files"
    params = {
        "dateFrom": start_date,
        "dateTo": start_date,
        "site": site,
        "product": "doppler-lidar",
    }
    metadata = requests.get(url, params).json()
    for row in metadata:
        print(row["filename"])
        res = requests.get(row["downloadUrl"])
        df = xr.open_dataset(res.content)
        return df
