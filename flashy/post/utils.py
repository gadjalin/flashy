import numpy as np
from itertools import groupby
from scipy.ndimage import gaussian_filter1d, median_filter

from ..eos import EosTable, EOS_MODE
from ..constants import PI


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
            if 'Bounce time' in line:
                line = line.strip()
                return float(line.split()[-1])
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
    return (4./3.) * PI * ((r + dr*0.5)**3 - (r - dr*0.5)**3) * dens


# TODO give m in grams and do conversion here
def calculate_compactness(at_mass, m, r) -> float:
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
        Compactness at mass coordinate at_mass.

    Raises
    ------
    ValueError
        If at_mass is not found in m.
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
    # The dat file usually contains consecutive duplicate values of shock radius
    # I think this is the shock detection that just gets stuck
    # to one cell for a few steps before moving to the next. So of course it is
    # not smooth. It also sometimes oscillates between two cells for a few time
    # steps
    # Removes the duplicates for a smoother result.
    for k, g in groupby(shock_radius):
        offset += len(list(g))
        shock_times_smooth.append(time[offset - 1])
        shock_rad_smooth.append(k)

    # Then interpolate on a (fine) regular time grid for filtering
    shock_times = np.linspace(time[0], time[-1], int(1000*(time[-1] - time[0])))
    shock_rad = np.interp(shock_times, shock_times_smooth, shock_rad_smooth)

    # Then derive velocity
    shock_vel = np.gradient(shock_rad, shock_times)

    # Filter out sharp outliers
    shock_vel = median_filter(shock_vel, size=15)
    # Then smooth back to legitimate trend
    shock_vel = gaussian_filter1d(shock_vel, sigma=5)
    return shock_times, shock_rad, shock_vel


def _diagnostic_energy(cells, eos):
    dens = np.asarray(cells['dens'], dtype=np.float64)
    temp = np.full_like(dens, eos.minimum_temperature)
    ye = np.asarray(cells['ye'], dtype=np.float64)

    coldenergydensity = eos.nuc_eos_zone(EOS_MODE.DENS_TEMP, dens, temp, ye)['logenergy']

    ener = np.asarray(cells['ener'], dtype=np.float64)
    gpot = np.asarray(cells['gpot'], dtype=np.float64)
    velr = np.asarray(cells['velx'], dtype=np.float64)

    scoldenergy = (10**(coldenergydensity) - eos.energy_shift)
    sdener = (ener - eos.energy_shift)
    sdgrav = gpot
    return sdener, sdgrav, scoldenergy, velr


def diagnostic_mask(cells, eos: EosTable) -> np.ndarray:
    sdener, sdgrav, scoldenergy, velr = _diagnostic_energy(cells, eos)
    sdetot = sdener + sdgrav - scoldenergy
    return (sdetot > 0.0) & (velr > 0.0)


def diagnostic_mass_1d(cells, eos: EosTable) -> float:
    sdener, sdgrav, scoldenergy, velr = _diagnostic_energy(cells, eos)

    dens = np.asarray(cells['dens'], dtype=np.float64)
    r = np.asarray(cells['r'], dtype=np.float64)
    dr = np.asarray(cells['dr'], dtype=np.float64)
    vol = (4./3.) * PI * ((r + dr*0.5)**3 - (r - dr*0.5)**3) 
    mass = dens*vol

    sdetot = sdener + sdgrav - scoldenergy
    return np.sum(mass[(sdetot > 0.0) & (velr > 0.0)])


def diagnostic_explosion_energy_1d(cells, eos) -> float:
    sdener, sdgrav, scoldenergy, velr = _diagnostic_energy(cells, eos)

    dens = np.asarray(cells['dens'], dtype=np.float64)
    r = np.asarray(cells['r'], dtype=np.float64)
    dr = np.asarray(cells['dr'], dtype=np.float64)
    vol = (4./3.) * PI * ((r + dr*0.5)**3 - (r - dr*0.5)**3) 
    mass = dens*vol

    detot = (sdener + sdgrav - scoldenergy)*mass
    return np.sum(detot[(detot > 0.0) & (velr > 0.0)])


def diagnostic_mass_2d(cells, eos: EosTable) -> float:
    sdener, sdgrav, scoldenergy, velr = _diagnostic_energy(cells, eos)

    dens = np.asarray(cells['dens'], dtype=np.float64)
    r = np.asarray(cells['r'], dtype=np.float64)
    dr = np.asarray(cells['dr'], dtype=np.float64)
    vol = 2.*PI*r*dr*dr
    mass = dens*vol

    sdetot = sdener + sdgrav - scoldenergy
    return np.sum(mass[(sdetot > 0.0) & (velr > 0.0)])


def diagnostic_explosion_energy_2d(cells, eos) -> float:
    sdener, sdgrav, scoldenergy, velr = _diagnostic_energy(cells, eos)

    dens = np.asarray(cells['dens'], dtype=np.float64)
    r = np.asarray(cells['r'], dtype=np.float64)
    dr = np.asarray(cells['dr'], dtype=np.float64)
    vol = 2.*PI*r*dr*dr
    mass = dens*vol

    detot = (sdener + sdgrav - scoldenergy)*mass
    return np.sum(detot[(detot > 0.0) & (velr > 0.0)])


