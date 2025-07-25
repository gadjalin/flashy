import re
from enum import IntEnum
import numpy as np
from itertools import groupby
from collections.abc import Mapping, Sequence
from abc import ABC, abstractmethod


class _LabelMap(IntEnum):
    LABEL = 0
    UNITS = 1
    LOG = 2


_PLOT_LABELS = { \
#   VARNAME            : ( LABEL,                    UNITS,                            LOG PREFERENCES )
    't'                : ( r'$t$',                   r'$\mathrm{s}$',                  False ), \
    'time'             : ( r'$t$',                   r'$\mathrm{s}$',                  False ), \
    'r'                : ( r'$r$',                   r'$\mathrm{cm}$',                 True  ), \
    'dr'               : ( r'$\mathrm{d}r$',         r'$\mathrm{cm}$',                 True  ), \
    'radius'           : ( r'$r$',                   r'$\mathrm{cm}$',                 True  ), \
    'm'                : ( r'$M$',                   r'$M_\odot$',                     False ), \
    'mass'             : ( r'$M$',                   r'$M_\odot$',                     False ), \
    'dens'             : ( r'$\rho$',                r'$\mathrm{g\,cm^{-3}}$',         True  ), \
    'temp'             : ( r'$T$',                   r'$\mathrm{K}$',                  True  ), \
    'eint'             : ( r'$e_\mathrm{int}$',      r'$\mathrm{erg\,g^{-1}}$',        True  ), \
    'ekin'             : ( r'$e_\mathrm{kin}$',      r'$\mathrm{erg\,g^{-1}}$',        True  ), \
    'ener'             : ( r'$e_\mathrm{tot}$',      r'$\mathrm{erg\,g^{-1}}$',        True  ), \
    'etot'             : ( r'$e_\mathrm{tot}$',      r'$\mathrm{erg\,g^{-1}}$',        True  ), \
    'velx'             : ( r'$v$',                   r'$\mathrm{cm\,s^{-1}}$',         False ), \
    'vrad'             : ( r'$v_\mathrm{rad}$',      r'$\mathrm{cm\,s^{-1}}$',         False ), \
    'eexp'             : ( r'$E_\mathrm{exp}$',      r'$\mathrm{erg}$',                False ), \
    'shock_radius'     : ( r'$r_\mathrm{sh}$',       r'$\mathrm{cm}$',                 True  ), \
    'min_shock_radius' : ( r'$r_\mathrm{sh}$',       r'$\mathrm{cm}$',                 True  ), \
    'mean_shock_radius': ( r'$r_\mathrm{sh}$',       r'$\mathrm{cm}$',                 True  ), \
    'max_shock_radius' : ( r'$r_\mathrm{sh}$',       r'$\mathrm{cm}$',                 True  ), \
    'shock_vel'        : ( r'$v_\mathrm{sh}$',       r'$\mathrm{cm\,s^{-1}}$',         False ), \
    'explosion_energy' : ( r'$E_\mathrm{exp}$',      r'$\mathrm{erg}$',                False ), \
    'point_mass'       : ( r'$m_\mathrm{point}$',    r'$\mathrm{g}$',                  False ), \
    'neutron_star_mass': ( r'$M_\mathrm{NS}$',       r'$M_\odot$',                     False ), \
    'central_density'  : ( r'$\rho_\mathrm{c}$',     r'$\mathrm{g\,cm^{-3}}$',         True  ), \
    'pres'             : ( r'$P$',                   r'$\mathrm{g\,cm^{-1}\,s^{-2}}$', True  ), \
    'entr'             : ( r'$s$',                   r'$k_B\,\mathrm{baryon^{-1}}$',   False ), \
    'ye'               : ( r'$Y_e$',                 None,                             False ), \
    'sumy'             : ( r'SumY',                  None,                             False ), \
    'abar'             : ( r'$\mathcal{\bar{A}}$',   None,                             False ), \
    'zbar'             : ( r'$\mathcal{\bar{Z}}$',   None,                             False ), \
}


def _fmt_magnitude(scale_factor, log):
    if log:
        return f'$\\times 10^{{{int(np.log10(scale_factor))}}}$'
    else:
        return f'$\\times {int(scale_factor)}$'


def get_plot_label(var: str, scale_factor: int = None, log: bool = None) -> str:
    """
    Returns an appropriate LaTeX label for use in a matplotlib plot.

    Arguments
    ---
    var : str
        Common variable name from a FLASH profile or plot file.
    scale_factor : int
        When the data are normalised to a different scale.
        Use in combination with log. If log is True and this is
        different than 0, a 10^{scale_factor} is prepended to the units in
        the label.
        If log is False and scale_factor is different than 1, the value of
        the scale_factor variable is prepended to the units in the label.
        If None, use unity scaling depending on the value of log.
    log : bool
        Use in combination with scale_factor.
        If log is True, the data are normalised on a log scale, and on
        a linear scale otherwise.
        If None, use default for the var quantity.

    TODO
    ---
    Automatically figure out simpler units e.g. print "GK"
    instead of "10^9 K".
    """

    label = ''
    
    if var in _PLOT_LABELS:
        label += _PLOT_LABELS[var][_LabelMap.LABEL]
        
        if _PLOT_LABELS[var][_LabelMap.UNITS] is not None:
            label += ' ['

            if log is None:
                log = _PLOT_LABELS[var][_LabelMap.LOG]

            if scale_factor is None:
                scale_factor = 0 if log else 1

            if (scale_factor != 0 and log) or (scale_factor != 1 and not log):
                label += _fmt_magnitude(scale_factor, log) + ' '

            label += _PLOT_LABELS[var][_LabelMap.UNITS]

            label += ']'
        
    elif len(var) > 0:
        label = var

    return label


def get_log(var: str) -> bool:
    """
    Determines if a variable should preferably be plotted on
    a log scale.

    Arguments
    ---
    var : str
        Common variable name from a FLASH profile or plot file.

    Returns
    ---
    True if var should be on a log scale, False if linear or
    variable is unknown.
    """

    if var in _PLOT_LABELS:
        return _PLOT_LABELS[var][_LabelMap.LOG]
    else:
        return False


def get_logstr(var: str) -> str:
    if get_log(var):
        return 'log'
    else:
        return 'linear'


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

