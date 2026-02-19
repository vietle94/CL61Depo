import xarray as xr
import numpy as np


def depo_fit(df, gates=5):
    imax = df.beta.argmax("range")
    irange = xr.DataArray(
        np.arange(df.beta.sizes["range"]),
        dims="range",
        coords={"range": df.beta["range"]},
    )

    mask_below = (irange <= imax) & (irange >= imax - gates)
    depo = df["depo"].where(mask_below)
    # depo = df["depo"].isel(range=slice(0, gates))
    fit = depo.polyfit("range", deg=1)
    slope = fit.polyfit_coefficients.sel(degree=1)
    intercept = fit.polyfit_coefficients.sel(degree=0)

    # Calculate R2
    y_pred = slope * depo.range + intercept
    ss_res = ((depo - y_pred) ** 2).sum(dim="range", skipna=True)
    ss_tot = ((depo - depo.mean(dim="range")) ** 2).sum(dim="range", skipna=True)
    r2 = 1 - ss_res / ss_tot
    summary = xr.Dataset(
        {
            "slope": ("time", slope.data),
            "intercept": ("time", intercept.data),
            "r2": ("time", r2.data),
        },
        coords={"time": depo.time.values},
    )
    return summary
