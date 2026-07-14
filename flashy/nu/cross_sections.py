import numpy as np
from .fermi import fermidirac_logtermonly_log10ed
from ..constants import M_E, AMU

# Neutrino cross sections from Burrows 2006
# FIXME: Ignore Fermi blocking and induced absorption corrections

_SIGMA0 = 1.76e-44   # reference cross section [cm2]
_GA = -1.25          # coupling constant
_DELTA_NP = 1.293332 # neutron - proton [MeV]
_M_REF = AMU         # Reference mass is amu for SFHo


# electron neutrino - neutron cross section
def sigma_e(e_nu: float) -> float:
    return _SIGMA0 * (1.0 + 3.0*_GA**2)/4.0 \
         * ((e_nu + _DELTA_NP) / M_E)**2 \
         * (1.0 - (M_E / (e_nu + _DELTA_NP)**2))**0.5 \
         * (1.0 + 1.1*e_nu/_M_REF) # Weak magnetism

# electron antineutrino - proton cross section
def sigma_a(e_nu: float) -> float:
    return _SIGMA0 * (1.0 + 3.0*_GA**2)/4.0 \
         * ((e_nu - _DELTA_NP) / M_E)**2 \
         * (1.0 - (M_E / (e_nu - _DELTA_NP)**2))**0.5 \
         * (1.0 - 7.1*e_nu/_M_REF) # Weak magnetism

# electron neutrino - heavy nucleus cross section
def sigma_A(
    e_nu: float,
    T: float,
    Abar: float,
    Zbar: float,
    mu_e: float,
    muhat: float
) -> float:
    Nbar = Abar - Zbar
    Z = 0 if Zbar < 20 else (Zbar - 20 if Zbar < 28 else 8)
    N = 6 if Nbar < 34 else (40 - Nbar if Nbar < 40 else 0)
    Q = muhat + 3
    f = fermidirac_logtermonly_log10ed(e_nu + Q, mu_e, T)
    SA = fermidirac_logtermonly_log10ed(e_nu, mu_e - muhat, T)

    return _SIGMA0/14.0 * _GA**2 * Z * N \
         * ((e_nu + Q) / M_E)**2 \
         * (1.0 - (M_E / (e_nu + Q)**2))**0.5 \
         * 10**(np.log10(1.0 + 10**SA) - np.log10(1.0 + 10**f))
         #* np.exp((mu_n - mu_p - Q)/(T*KELVIN_TO_MEV))

