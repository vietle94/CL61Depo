import requests
import xarray as xr
import numpy as np
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
        result = process_lwc(lwc, df)
        if result is None:
            continue
        else:
            result.to_netcdf(path + f"/lwc/{file_date}_lwc.nc")


def process_lwc(lwc, df):
    lwc = lwc.where(lwc.lwc_retrieval_status == 1)
    lwc = lwc.reindex(time=df.time, method="nearest", tolerance=np.timedelta64(30, "m"))
    lwc = lwc.where(lwc.lwc_error.mean("height") < 0.75)
    lwc_adiabatic_full = lwc.lwc.differentiate("height")
    mask = lwc_adiabatic_full.notnull()
    if (~mask).all():
        return None

    first_valid_idx = mask.argmax("height")
    has_valid = mask.any("height")

    lwc_adiabatic = lwc_adiabatic_full.isel(height=first_valid_idx).where(has_valid)
    result = xr.Dataset(
        {
            "lwc_adiabatic": (("time"), lwc_adiabatic.data),
        },
        coords={"time": lwc_adiabatic.time.values},
    )
    return result


def fetch_cloud_time(path):
    files = glob.glob(path + "/*.nc")
    for file in files:
        file_date = file.split("/")[-1].split(".")[0]
        idate = file_date[:4] + "-" + file_date[4:6] + "-" + file_date[6:]
        file_site = files[0].split("/")[-2].lower()
        classi = fetch_categorize(file_site, idate, "classification")
        if classi is None:
            continue
        df = xr.open_dataset(file)
        classi = classi.reindex(
            time=df.time, method="nearest", tolerance=np.timedelta64(30, "m")
        )
        h = xr.DataArray(np.arange(classi.sizes["height"]), dims="height")
        bad_vals = [2, 3, 4, 5, 6, 7]
        valid_time = (
            ~(
                classi.target_classification.isin(bad_vals)
                & (h < (classi.target_classification == 1).argmax("height"))
            ).any("height")
        ) & (classi.target_classification == 1).any("height")
        result = xr.Dataset(
            {
                "valid_time": (("time"), valid_time.data),
            },
            coords={"time": valid_time.time.values},
        )
        if result is None:
            continue
        else:
            result.to_netcdf(path + f"/time/{file_date}_valid_time.nc")
