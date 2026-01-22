import requests
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import time


def fetch_raw(site, start_date, end_date, save_path):
    """Download just raw data"""
    url = "https://cloudnet.fmi.fi/api/raw-files"
    params = {
        "dateFrom": start_date,
        "dateTo": end_date,
        "site": site,
        "instrument": "cl61d",
    }
    print(params)
    metadata = requests.get(url, params).json()
    with ThreadPoolExecutor() as exe:
        exe.map(raw, metadata, repeat(save_path))


def raw(row, save_path):
    if "live" in row["filename"]:
        i = 0
        if int(row["size"]) < 100000:
            return None
        while True:
            try:
                print(row["filename"])
                bad_file = False
                res = requests.get(row["downloadUrl"])
                file_name = save_path + "/" + row["filename"]
                with open(file_name, "wb") as f:
                    f.write(res.content)
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
    with ThreadPoolExecutor() as exe:
        exe.map(raw_model, metadata, repeat(save_path))


def raw_model(row, save_path):
    res = requests.get(row["downloadUrl"])
    file_name = save_path + "/" + row["filename"]
    with open(file_name, "wb") as f:
        f.write(res.content)


def fetch_daily(site, start_date, end_date, save_path):
    """Download model data"""
    url = "https://cloudnet.fmi.fi/api/files"
    params = {
        "dateFrom": start_date,
        "dateTo": end_date,
        "site": site,
        "instrument": "cl61d",
    }
    print(params)
    metadata = requests.get(url, params).json()
    with ThreadPoolExecutor() as exe:
        exe.map(raw_model, metadata, repeat(save_path))
