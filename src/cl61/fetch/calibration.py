import pandas as pd
from cl61.fetch.raw import fetch_raw


file_dir = "/media/viet/CL61/calibration/"
df = pd.read_csv(file_dir + "calibration.txt")

for _, row in df.iterrows():
    save_path = "/media/viet/CL61/calibration/" + row["site"] + "/"
    datefrom = pd.to_datetime(str(row["date1"])).strftime("%Y-%m-%d")
    dateto = pd.to_datetime(str(row["date2"])).strftime("%Y-%m-%d")
    print(str(row["date1"]))
    fetch_raw(row["site"], datefrom, dateto, save_path)
