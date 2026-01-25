import requests
import xarray as xr
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import importlib.resources

with importlib.resources.files("cl61.fetch").joinpath("cal_ref.npy").open("rb") as f:
    ref = np.load(f)

# %%
start_date = "2025-10-22"
end_date = "2025-10-22"
site = "hyytiala"

url = "https://cloudnet.fmi.fi/api/raw-files"

params = {
    "dateFrom": start_date,
    "dateTo": end_date,
    "site": site,
    "instrument": "cl61d",
}
metadata = requests.get(url, params).json()
res = requests.get(metadata[3]["downloadUrl"])
df = xr.open_dataset(io.BytesIO(res.content))
df["depo"] = df["x_pol"] / df["p_pol"]


# %%
def convolve_1d(arr, kernel):
    return np.convolve(arr, kernel, mode="same")


# %%
df = df.sel(range=slice(100, 5000))
result = xr.apply_ufunc(
    convolve_1d,
    df.beta_att,
    input_core_dims=[["range"]],  # Apply along 'lon'
    kwargs={"kernel": ref},
    output_core_dims=[["range"]],
    vectorize=True,
)
range_max = result.idxmax(dim="range").values
# %%
df_plot = df.where(df.range < result.idxmax(dim="range") + 76.8)
df_plot = df_plot.where(df_plot.beta_att.T / df_plot.beta_att.max(dim="range") > 0.05)
cloud_base = df_plot["depo"].idxmin(dim="range")

# %%
fig, ax = plt.subplots()
p = ax.pcolormesh(
    df_plot.time, df_plot.range, df_plot.beta_att.T, norm=LogNorm(vmin=1e-7, vmax=1e-4)
)
ax.plot(df.time, range_max + 76.8, ".")
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
ax.plot(df.time, cloud_base, ".")
ax.set_ylim(0, 600)
fig.colorbar(p, ax=ax)

# %%
fig, ax = plt.subplots()
ax.pcolormesh(df.time, df.range, df.beta_att.T, norm=LogNorm(vmin=1e-7, vmax=1e-4))
ax.set_ylim(0, 600)

# %%
fig, ax = plt.subplots()
ax.plot(df.depo.isel(time=5), df.range)
ax.set_ylim(0, 600)
ax.set_xlim(-0.001, 0.05)
# %%
fig, ax = plt.subplots()
p = ax.pcolormesh(
    df2.time, df2.range, df2.beta_att.T, norm=LogNorm(vmin=1e-7, vmax=1e-4)
)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
ax.set_ylim(0, 600)
fig.colorbar(p, ax=ax)
# %%
df2 = df.where(df.range > cloud_base)

n_time, n_range = df2.depo.shape

# -----------------------------
# Step 1: Find first non-NaN index per time
# -----------------------------
mask = ~np.isnan(df2.depo.values)
first_idx = mask.argmax(axis=1)  # shape (time,)

# -----------------------------
# Step 2: Create shifted array
# -----------------------------
shifted = np.full((n_time, 100), np.nan)

# Compute indices for broadcasting
range_idx = np.arange(100)  # 0..max_len-1
src_idx = first_idx[:, None] + range_idx

# Assign values using broadcasting
shifted = df2.depo.values[np.arange(n_time)[:, None], src_idx]
# %%
fig, ax = plt.subplots()
p = ax.pcolormesh(
    df2.time,
    df2.isel(range=slice(0, 100)).range - df2.isel(range=0).range,
    shifted.T,
    vmin=0,
    vmax=0.1,
)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
# ax.set_ylim(0, 600)
fig.colorbar(p, ax=ax)
# %%

lapse_lwc = 2 * lwp / (cloud_thickness**2)
# use categorize to get the radar data for Reff
# and get lwp
# %%
fig, ax = plt.subplots()
p = ax.pcolormesh(
    df.time,
    df.isel(range=slice(0, 100)).range - df.isel(range=0).range,
    shifted.T,
    vmin=0,
    vmax=0.1,
)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
# ax.set_ylim(0, 600)
fig.colorbar(p, ax=ax)
# %%
dd
