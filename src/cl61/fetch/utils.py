import xarray as xr
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
                df = xr.open_dataset(res.content)
                result_ = func(df)
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


def process_metadata(metadata, func) -> xr.Dataset | None:
    result = []
    with ProcessPoolExecutor() as exe:
        result = exe.map(process_metadata_child, metadata, repeat(func))
        result = [ds for ds in result if ds is not None]
    if not result:
        return None
    df = xr.concat(result, dim="time")
    df = df.sortby("time")
    return df
