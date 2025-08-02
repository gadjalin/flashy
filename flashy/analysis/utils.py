import numpy as np
from itertools import groupby


def get_bounce_time(log_file: str) -> float:
    """
    Finds the exact bounce time from the log file.

    Parameters
    ----------
    log_file : str
        Path to the log file of the simulation.

    Returns
    -------
    float or None
        The bounce time in seconds, or None if not found.
    """
    with open(log_file, 'r') as f:
        for line in f:
            if 'Bounce!' in line:
                line = line.strip()
                return float(line.split()[1])
    return None


def get_midcell_dr(r):
    # Assume cell-centred coordinates in model
    faces = np.zeros(len(r) + 1)
    faces[1:-1] = 0.5 * (r[1:] + r[:-1])
    faces[0] = r[0] - 0.5 * (r[1] - r[0])
    faces[-1] = r[-1] + 0.5 * (r[-1] - r[-2])
    dr = faces[1:] - faces[:-1]
    return dr


def calculate_shell_mass(r, dr, dens):
    """
    Calculates the mass of the shells.

    Parameters
    ----------
    r : array-like
        The mid-cell radial coordinates of the shells.
    dr : array-like
        The size of the shells.
    dens : array-like
        The average density in the shells.

    Returns
    -------
    array-like
        The masses of individual mass shells.
    """
    return (4./3.) * np.pi * ((r + dr*0.5)**3 - (r - dr*0.5)**3) * dens


# TODO give m in grams and do conversion here
def calculate_compactness(at_mass: float, m: list[float], r: list[float]) -> float:
    """
    Calculate compactness given mass and radius coordinates.

    Parameters
    ----------
    at_mass : float
        Compactness mass (in solar mass).
    m : array-like
        Mass coordinates at bounce (in solar mass).
    r : array-like
        Radius coordinates at bounce (in cm).

    Returns
    -------
    float
        Compactness at mass coordinate `at_mass`.

    Raises
    ------
    ValueError
        If `at_mass` is not found in `m`.
    """
    if at_mass > np.max(m) or at_mass < np.min(m):
        raise ValueError(f'Given domain does not contain requested mass coordinate: {at_mass}')
    return at_mass/(np.interp(at_mass, m, r)*1e-8)


# TODO Improve shock velocity calculation
def calculate_shock(time, shock_radius):
    """
    Process shock radius from dat file.

    Parameters
    ----------
    time : array-like
        The list of times at which the shock radius is evaluated.
    shock_rad : array-like
        The shock radius at different times.
        Typically the min|max|mean shock radius column from the dat file.

    Returns
    -------
    Lists of floats, containing the processed shock times,
    radii and velocity.
    """

    shock_times_smooth = [0]
    shock_rad_smooth = [0]
    offset = 0
    # The dat file usually contains duplicates of the same time.
    # Removes the duplicates for a smoother result.
    for k, g in groupby(shock_radius):
        offset += len(list(g))
        shock_times_smooth.append(time[offset - 1])
        shock_rad_smooth.append(k)

    #t_bounce = time[np.min(np.nonzero(max_shock_rad)) - 2]
    #shock_times = np.logspace(np.log10(t_bounce), np.log10(time[-1]), 1000)
    shock_times = np.linspace(time[0], time[-1], 1000)
    shock_rad = np.interp(shock_times, shock_times_smooth, shock_rad_smooth)
    shock_vel = np.gradient(shock_rad, shock_times)
    return shock_times, shock_rad, shock_vel

