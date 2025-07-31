from enum import IntEnum
import numpy as np
from itertools import groupby


def get_bounce_time(logfile: str) -> float:
    """
    Finds the exact bounce time from the log file.

    Arguments
    ---
    logfile : str
        Path to the log file of the simulation.

    Returns
    ---
        The bounce time in seconds, or None if not found.
    """

    with open(logfile, 'r') as f:
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

    Arguments
    ---
    r : list[float]
        The mid-cell radial coordinates of the shells.
    dr : list[float]
        The width of the shells.
    dens : list[float]
        The average density in the shells.
    """

    return (4./3.) * np.pi * ((r + dr*0.5)**3 - (r - dr*0.5)**3) * dens

def calculate_shock(time, shock_radius):
    """
    Calculates shock velocity.

    Arguments
    ---
    time : list[float]
        The list of times at which the shock radius is evaluated.
    shock_rad : list[float]
        The shock radius at different times.
        The min|max|mean_shock_radius column from the dat file.

    Returns
    ---
    A tuple of lists of floats, containing the processed shock times,
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

