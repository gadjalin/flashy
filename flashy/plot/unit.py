from __future__ import annotations
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from ..constants import YEAR, AMU_G, M_SUN, R_SUN, EV_TO_ERG, BETHE


@dataclass(frozen=True)
class Unit:
    """
    A unit for plot labels and unit conversion.

    Units can be combined into compound units.
    Powers of the same base unit are reduced.
    """
    latex: str
    conversion_factor: Optional[float]
    _parts: Tuple[Tuple[str, float], ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self):
        if not self._parts:
            object.__setattr__(self, '_parts', ((self.latex, 1),))

    @staticmethod
    def _from_parts(parts, conversion_factor: Optional[float]) -> Unit:
        # Reduce powers of identical base units
        powers: Dict[str, float] = {}
        for latex,power in parts:
            powers[latex] = powers.get(latex, 0) + power
        parts = tuple((latex, power) for latex,power in powers.items() if power != 0)

        def fmt(latex, power):
            if power == 1:
                return latex
            # Group bases that are themselves compound to keep the power unambiguous
            if '^' in latex or r'\,' in latex:
                latex = f'{{{latex}}}'
            return f'{latex}^{{{power:.7g}}}'

        return Unit(r'\,'.join(fmt(*p) for p in parts), conversion_factor, parts)

    @staticmethod
    def _combine(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return None if a is None or b is None else a*b

    def __mul__(self, other: Unit) -> Unit:
        return Unit._from_parts(self._parts + other._parts,
                                Unit._combine(self.conversion_factor, other.conversion_factor))

    def __truediv__(self, other: Unit) -> Unit:
        return self * other**(-1)

    def __pow__(self, n: float) -> Unit:
        factor = None if self.conversion_factor is None else self.conversion_factor**n
        return Unit._from_parts(tuple((latex, power*n) for latex,power in self._parts), factor)


# Time
UNIT_MS  = Unit(r'\mathrm{ms}', 1e-3)
UNIT_S   = Unit(r'\mathrm{s}', 1.0)
UNIT_YR  = Unit(r'\mathrm{yr}', YEAR)
UNIT_MYR = Unit(r'\mathrm{Myr}', YEAR*1e6)
UNIT_GYR = Unit(r'\mathrm{Gyr}', YEAR*1e9)

TIME_UNITS = {
    'cgs': UNIT_S, 'si': UNIT_S,
    's': UNIT_S, 'ms': UNIT_MS,
    'yr': UNIT_YR, 'Myr': UNIT_MYR, 'Gyr': UNIT_GYR
}

# Length
UNIT_MM    = Unit(r'\mathrm{mm}', 1e-1)
UNIT_CM    = Unit(r'\mathrm{cm}', 1.0)
UNIT_M     = Unit(r'\mathrm{m}', 1e2)
UNIT_KM    = Unit(r'\mathrm{km}', 1e5)
UNIT_FERMI = Unit(r'\mathrm{fm}', 1e-13)
UNIT_RSUN  = Unit(r'R_\odot', R_SUN)

LENGTH_UNITS = {
    'cgs': UNIT_CM, 'si': UNIT_M, 'sun': UNIT_RSUN,
    'mm': UNIT_MM, 'cm': UNIT_CM, 'm': UNIT_M, 'km': UNIT_KM,
    'fm': UNIT_FERMI, 'fermi': UNIT_FERMI
}

# Mass
UNIT_MG   = Unit(r'\mathrm{mg}', 1e-3)
UNIT_G    = Unit(r'\mathrm{g}', 1.0)
UNIT_KG   = Unit(r'\mathrm{kg}', 1e3)
UNIT_AMU  = Unit(r'\mathrm{amu}', AMU_G)
UNIT_MSUN = Unit(r'M_\odot', M_SUN)

MASS_UNITS = {
    'cgs': UNIT_G, 'si': UNIT_KG, 'sun': UNIT_MSUN,
    'mg': UNIT_MG, 'g': UNIT_G, 'kg': UNIT_KG,
    'amu': UNIT_AMU
}

# Temperature
UNIT_K    = Unit(r'\mathrm{K}', 1.0)
UNIT_MK   = Unit(r'\mathrm{MK}', 1e6)
UNIT_GK   = Unit(r'\mathrm{GK}', 1e9)

TEMPERATURE_UNITS = {
    'cgs': UNIT_K, 'si': UNIT_K,
    'K': UNIT_K, 'MK': UNIT_MK, 'GK': UNIT_GK
}

# Energy
UNIT_ERG   = Unit(r'\mathrm{erg}', 1.0)
UNIT_JOULE = Unit(r'\mathrm{J}', 1e7)
UNIT_BETHE = Unit(r'\mathrm{B}', BETHE)
UNIT_FOE   = Unit(r'\mathrm{foe}', BETHE)
UNIT_EV    = Unit(r'\mathrm{eV}', EV_TO_ERG)
UNIT_MEV   = Unit(r'\mathrm{MeV}', 1e6*EV_TO_ERG)

ENERGY_UNITS = {
    'cgs': UNIT_ERG, 'si': UNIT_JOULE,
    'erg': UNIT_ERG, 'J': UNIT_JOULE,
    'B': UNIT_BETHE, 'Bethe': UNIT_BETHE, 'foe': UNIT_FOE,
    'eV': UNIT_EV, 'MeV': UNIT_MEV
}

# Entropy
UNIT_ENTROPY = Unit(r'k_B\,\mathrm{baryon^{-1}}', 1.0)

ENTROPY_UNITS = {'cgs': UNIT_ENTROPY, 'si': UNIT_ENTROPY}

# Derived quantities
def _per(units: Dict[str, Unit], other: Unit) -> Dict[str, Unit]:
    return {k: u/other for k,u in units.items()}

VELOCITY_UNITS = {
    'cgs': UNIT_CM/UNIT_S, 'si': UNIT_M/UNIT_S,
    'km': UNIT_KM/UNIT_S,
}

MASS_RATE_UNITS = {
    'cgs': UNIT_G/UNIT_S, 'si': UNIT_KG/UNIT_S,
    'sun': UNIT_MSUN/UNIT_S, 'yr': UNIT_MSUN/UNIT_YR,
}

DENSITY_UNITS = {
    'cgs': UNIT_G/UNIT_CM**3, 'si': UNIT_KG/UNIT_M**3,
    'fermi': Unit(r'\mathrm{fm}^{-3}', AMU_G/UNIT_FERMI.conversion_factor**3),
}

PRESSURE_UNITS = {
    'cgs': UNIT_G/UNIT_CM/UNIT_S**2, 'si': UNIT_KG/UNIT_M/UNIT_S**2,
}

POWER_UNITS = {
    'cgs': UNIT_ERG/UNIT_S, 'si': UNIT_JOULE/UNIT_S,
    'B': UNIT_BETHE/UNIT_S, 'Bethe': UNIT_BETHE/UNIT_S,
    'W': Unit(r'\mathrm{W}', UNIT_JOULE.conversion_factor/UNIT_S.conversion_factor)
}

SPECIFIC_ENERGY_UNITS = {
    'cgs': UNIT_ERG/UNIT_G, 'si': UNIT_JOULE/UNIT_KG,
}

SPECIFIC_POWER_UNITS = _per(SPECIFIC_ENERGY_UNITS, UNIT_S)

