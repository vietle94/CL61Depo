import numpy as np
from scipy.integrate import cumulative_trapezoid

"""
Bucholtz, A. Rayleigh-scattering calculations for the terrestrial atmosphere.
Applied Optics, Vol. 34, No. 15, 2766-2773 (1995)

https://doi.org/10.1364/AO.34.002765

"""


def molecular_backscatter(angle, temperature, pressure):
    """
    Calculate molecular backscatter coefficient at 910.55nm.

    Parameters:
        temperature (np.ndarray): Temperature in Kelvin [K]
        pressure (np.ndarray): Pressure in hPa [hPa]

    Returns:
        molBsc (np.ndarray): Molecular backscatter coefficient [km^-1 sr^-1]
    """
    y = 1.384e-2
    P_ray = 3 / (4 * (1 + 2 * y)) * ((1 + 3 * y) + (1 - y) * np.cos(angle) ** 2)
    molBsc = (
        1.5e-3 * (pressure / 1013.25) * (288.15 / temperature) / (4 * np.pi) * P_ray
    )
    return molBsc


""" Calculate atmospheric molecular depolarization ratio
Tomasi, C., Vitale, V., Petkov, B., Lupi, A. & Cacciari, A. Improved
algorithm for calculations of Rayleigh-scattering optical depth in standard
atmospheres. Applied Optics 44, 3320 (2005).

https://doi.org/10.1364/AO.44.003320

"""


def f1(wavelength):
    return 1.034 + 3.17 * 1e-4 * wavelength ** (-2)


def f2(wavelength):
    return 1.096 + 1.385 * 1e-3 * wavelength ** (-2) + 1.448 * 1e-4 * wavelength ** (-4)


def f(wavelength, C, water_over_air):
    """
    Calculate King's factor.

    Parameters:
        wavelength : wavelength [um]
        C: CO2 concentration [ppmv]
        water_over_air: Water vapor over air pressure ratio

    Returns:
        king_factor : King's factor
    """
    numerator = (
        0.78084 * f1(wavelength)
        + 0.20946 * f2(wavelength)
        + 0.00934 * 1
        + 1e-6 * C * 1.15
        + water_over_air * 1.001
    )
    denominator = 0.999640 + 1e-6 * C + water_over_air
    return numerator / denominator


def depo(king_factor):
    return (6 * king_factor - 6) / (7 * king_factor + 3)


def humidity_conversion(specific_humidity):
    """
    Calculate water over air pressure ratio from specific humidity.
    """
    return 1 / (18 / 29 * (1 / specific_humidity - 1) + 1)


def rayleigh_fitting(beta_profile, beta_mol, zmin, zmax):
    """
    Fit the backscatter profile to the molecular profile in a given height range.

    Parameters:
    - beta_profile: array-like
        Measured backscatter profile [m^-1 sr^-1].
    - beta_mol: array-like
        Molecular backscatter profile [m^-1 sr^-1].
    - zmin: float
        Minimum height for fitting [m].
    - zmax: float
        Maximum height for fitting [m].

    Returns:
    - popt: tuple
        Optimal parameters for the fitting.
    - pcov: 2D array
        Covariance of the optimal parameters.
    """
    # Select the fitting range
    beta_profile = beta_profile.sel(range=slice(zmin, zmax))
    beta_mol = beta_mol.sel(range=slice(zmin, zmax))

    # attenuated mol beta
    #

    # Calibration factor
    # c = att_beta_mol.sum() / beta_profile.sum()
    c = beta_mol.sum() / beta_profile.sum()

    return c.values


def backward(beta, beta_mol, Sa, zref, z):
    beta = beta.sel(range=slice(None, zref))
    beta_mol = beta_mol.sel(range=slice(None, zref))
    z = z.sel(range=slice(None, zref))
    beta = beta[::-1]
    beta_mol = beta_mol[::-1]
    z = z[::-1]
    beta[0] = beta_mol[0]  # boundary condition after normalization
    Zb = beta * np.exp(
        2 * cumulative_trapezoid((Sa - 8 / 3 * np.pi) * beta_mol, z, initial=0)
    )
    Nb = (beta[0] / beta_mol[0]).values + 2 * cumulative_trapezoid(
        Sa * Zb, z, initial=0
    )

    beta_a = Zb / Nb - beta_mol
    return beta_a[::-1]


def forward(beta, beta_mol, Sa, c, z):
    Zb = beta * np.exp(
        -2 * cumulative_trapezoid((Sa - 8 / 3 * np.pi) * beta_mol, z, initial=0)
    )
    Nb = c - 2 * cumulative_trapezoid(Sa * Zb, z, initial=0)

    beta_a = Zb / Nb - beta_mol
    return beta_a


def forward_sigma(beta, beta_mol, Sa, c, z, sigma_beta):
    term1 = (
        forward(beta + sigma_beta, beta_mol, Sa, c, z)
        - forward(beta - sigma_beta, beta_mol, Sa, c, z)
    ) / 2
    term2 = (
        forward(beta, beta_mol, Sa, c * 0.9, z)
        - forward(beta, beta_mol, Sa, c * 1.1, z)
    ) / 2
    term3 = (
        forward(beta, beta_mol, Sa + 10, c, z) - forward(beta, beta_mol, Sa - 10, c, z)
    ) / 2
    result = np.sqrt(term1**2 + term2**2 + term3**2)
    return result


def backscatter_ratio(beta_aerosol, beta_mol):
    """
    Calculate backscatter ratio.

    Parameters:
        beta_total (np.ndarray): Total backscatter coefficient [km^-1 sr^-1]
        beta_mol (np.ndarray): Molecular backscatter coefficient [km^-1 sr^-1]

    Returns:
        bsr (np.ndarray): Backscatter ratio
    """
    return (beta_aerosol + beta_mol) / beta_mol


def depo_aerosol(depo_volume, depo_mol, beta_ratio):
    """
    Calculate aerosol depolarization ratio.

    Parameters:
        depo_volume (np.ndarray): Volume depolarization ratio
        depo_mol (np.ndarray): Molecular depolarization ratio
        beta_ratio (np.ndarray): Backscatter ratio

    Returns:
        depo_aer (np.ndarray): Aerosol depolarization ratio
    """
    term1 = (1 + depo_mol) * depo_volume * beta_ratio - (1 + depo_volume) * depo_mol
    term2 = (1 + depo_mol) * beta_ratio - (1 + depo_volume)
    return term1 / term2


def depo_aersosol_sigma(
    depo_volume, depo_mol, beta_ratio, sigma_depo_volume, sigma_beta_ratio
):
    """
    Calculate aerosol depolarization ratio using extinction ratio.

    Parameters:
        depo_volume (np.ndarray): Volume depolarization ratio
        depo_mol (np.ndarray): Molecular depolarization ratio
        sigma_ratio (np.ndarray): Extinction ratio

    Returns:
        depo_aer (np.ndarray): Aerosol depolarization ratio
    """
    term1_nominator = (1 + depo_mol) ** 2 * beta_ratio * (beta_ratio - 1)
    term1_denominator = ((1 + depo_mol) * beta_ratio - (1 + depo_volume)) ** 2
    term2_nominator = (1 + depo_mol) * (1 + depo_volume) * (depo_mol - depo_volume)
    term2_denominator = ((1 + depo_mol) * beta_ratio - (1 + depo_volume)) ** 2
    result = np.sqrt(
        (term1_nominator / term1_denominator) * sigma_depo_volume**2
        + (term2_nominator / term2_denominator) * sigma_beta_ratio**2
    )
    return result
