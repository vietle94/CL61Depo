import xarray as xr
import io
import pandas as pd
import requests
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def process_metadata_child(row, func):
    if "live" in row["filename"]:
        i = 0
        if int(row["size"]) < 100000:
            return None
        while True:
            try:
                print(row["filename"])
                bad_file = False
                res = requests.get(row["downloadUrl"])
                result_ = func(res)
                return result_
            except ValueError as error:
                i += 1
                print(i)
                if i > 50:
                    print("skip")
                    break
                print(error)
                time.sleep(1)
                continue
            except (OSError, KeyError):
                bad_file = True
                print("Bad file")
                break
            break
        if bad_file:
            return None


def process_metadata(metadata, func):
    result = []
    with ProcessPoolExecutor() as exe:
        result = exe.map(process_metadata_child, metadata, repeat(func))
    df = pd.concat(result, ignore_index=True)
    return df


def response(res):
    """return data in netcdf and diagnostics in csv"""
    df = xr.open_groups(io.BytesIO(res.content))
    if "/diagnostics" in df.keys():
        df["/"] = df["/"].swap_dims({"profile": "time"})
        diag = pd.DataFrame([df["/diagnostics"].attrs])
        diag["datetime"] = df["/"].time[0].values

    elif "Timestamp" in df["/monitoring"].attrs:
        monitoring = pd.DataFrame([df["/monitoring"].attrs])
        monitoring = monitoring.rename({"Timestamp": "datetime"}, axis=1)
        monitoring.datetime = monitoring.datetime.astype(float)
        monitoring["datetime"] = pd.to_datetime(monitoring["datetime"], unit="s")

        status = pd.DataFrame([df["/status"].attrs])
        status = status.rename({"Timestamp": "datetime"}, axis=1)
        status.datetime = status.datetime.astype(float)
        status["datetime"] = pd.to_datetime(status["datetime"], unit="s")
        diag = monitoring.merge(status)
    else:
        monitoring = df["/monitoring"].to_dataframe().reset_index()
        monitoring = monitoring.rename({"time": "datetime"}, axis=1)

        status = df["/status"].to_dataframe().reset_index()
        status = status.rename({"time": "datetime"}, axis=1)
        diag = monitoring.merge(status)

    return df["/"], diag
