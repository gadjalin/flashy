# For type hints
from __future__ import annotations
from typing import Union

from dataclasses import dataclass, field
import numpy as np
import re

# Dictionary mapping element symbols and names to atomic numbers and mass
# number of most stable isotope
_SYMBOL_TO_ELEMENT = {
   "N0": (0,  1  ), "H": (1,  1  ), "He": (2,  4  ), "Li": (3,  7  ), "Be": (4,  9  ), "B": (5,  11 ), "C": (6,  12 ),
    "N": (7,  14 ), "O": (8,  16 ), "F": (9,  19 ), "Ne": (10, 20 ), "Na": (11, 23 ), "Mg": (12, 24 ), "Al": (13, 27 ),
   "Si": (14, 28 ), "P": (15, 31 ), "S": (16, 32 ), "Cl": (17, 35 ), "Ar": (18, 40 ), "K": (19, 39 ), "Ca": (20, 40 ),
   "Sc": (21, 45 ), "Ti": (22, 48 ), "V": (23, 51 ), "Cr": (24, 52 ), "Mn": (25, 55 ), "Fe": (26, 56 ), "Co": (27, 59 ),
   "Ni": (28, 58 ), "Cu": (29, 63 ), "Zn": (30, 64 ), "Ga": (31, 69 ), "Ge": (32, 74 ), "As": (33, 75 ), "Se": (34, 80 ),
   "Br": (35, 79 ), "Kr": (36, 84 ), "Rb": (37, 85 ), "Sr": (38, 88 ), "Y": (39, 89 ), "Zr": (40, 90 ), "Nb": (41, 93 ),
   "Mo": (42, 98 ), "Tc": (43, 97 ), "Ru": (44, 102), "Rh": (45, 103), "Pd": (46, 106), "Ag": (47, 107), "Cd": (48, 114),
   "In": (49, 115), "Sn": (50, 120), "Sb": (51, 121), "Te": (52, 130), "I": (53, 127), "Xe": (54, 132), "Cs": (55, 133),
   "Ba": (56, 138), "La": (57, 139), "Ce": (58, 140), "Pr": (59, 141), "Nd": (60, 142), "Pm": (61, 145), "Sm": (62, 152),
   "Eu": (63, 153), "Gd": (64, 158), "Tb": (65, 159), "Dy": (66, 164), "Ho": (67, 165), "Er": (68, 166), "Tm": (69, 169),
   "Yb": (70, 174), "Lu": (71, 175), "Hf": (72, 180), "Ta": (73, 181), "W": (74, 184), "Re": (75, 187), "Os": (76, 192),
   "Ir": (77, 193), "Pt": (78, 195), "Au": (79, 197), "Hg": (80, 202), "Tl": (81, 205), "Pb": (82, 208), "Bi": (83, 209),
   "Po": (84, 209), "At": (85, 210), "Rn": (86, 222), "Fr": (87, 223), "Ra": (88, 226), "Ac": (89, 227), "Th": (90, 232),
   "Pa": (91, 231), "U": (92, 238), "Np": (93, 237), "Pu": (94, 244), "Am": (95, 243), "Cm": (96, 247), "Bk": (97, 247),
   "Cf": (98, 251), "Es": (99, 252), "Fm": (100,257), "Md": (101,258), "No": (102,259), "Lr": (103,262), "Rf": (104,267),
   "Db": (105,262), "Sg": (106,269), "Bh": (107,264), "Hs": (108,269), "Mt": (109,278), "Ds": (110,281), "Rg": (111,282),
   "Cn": (112,285), "Nh": (113,286), "Fl": (114,289), "Mc": (115,289), "Lv": (116,293), "Ts": (117,294), "Og": (118,294)
}

_NAME_TO_ELEMENT = {
    "Neutron": (0, 1), "Hydrogen": (1, 1), "Helium": (2, 4), "Lithium": (3, 7), "Beryllium": (4, 9),
    "Boron": (5, 11), "Carbon": (6, 12), "Nitrogen": (7, 14), "Oxygen": (8, 16), "Fluorine": (9, 19),
    "Neon": (10, 20), "Sodium": (11, 23), "Magnesium": (12, 24), "Aluminium": (13, 27), "Silicon": (14, 28),
    "Phosphorus": (15, 31), "Sulfur": (16, 32), "Chlorine": (17, 35), "Argon": (18, 40), "Potassium": (19, 39),
    "Calcium": (20, 40), "Scandium": (21, 45), "Titanium": (22, 48), "Vanadium": (23, 51), "Chromium": (24, 52),
    "Manganese": (25, 55), "Iron": (26, 56), "Cobalt": (27, 59), "Nickel": (28, 58), "Copper": (29, 63),
    "Zinc": (30, 64), "Gallium": (31, 69), "Germanium": (32, 74), "Arsenic": (33, 75), "Selenium": (34, 80),
    "Bromine": (35, 79), "Krypton": (36, 84), "Rubidium": (37, 85), "Strontium": (38, 88), "Yttrium": (39, 89),
    "Zirconium": (40, 90), "Niobium": (41, 93), "Molybdenum": (42, 98), "Technetium": (43, 97), "Ruthenium": (44, 102),
    "Rhodium": (45, 103), "Palladium": (46, 106), "Silver": (47, 107), "Cadmium": (48, 114), "Indium": (49, 115),
    "Tin": (50, 120), "Antimony": (51, 121), "Tellurium": (52, 130), "Iodine": (53, 127), "Xenon": (54, 132),
    "Caesium": (55, 133), "Barium": (56, 138), "Lanthanum": (57, 139), "Cerium": (58, 140), "Praseodymium": (59, 141),
    "Neodymium": (60, 142), "Promethium": (61, 145), "Samarium": (62, 152), "Europium": (63, 153), "Gadolinium": (64, 158),
    "Terbium": (65, 159), "Dysprosium": (66, 164), "Holmium": (67, 165), "Erbium": (68, 166), "Thulium": (69, 169),
    "Ytterbium": (70, 174), "Lutetium": (71, 175), "Hafnium": (72, 180), "Tantalum": (73, 181), "Tungsten": (74, 184),
    "Rhenium": (75, 187), "Osmium": (76, 192), "Iridium": (77, 193), "Platinum": (78, 195), "Gold": (79, 197), "Mercury": (80, 202),
    "Thallium": (81, 205), "Lead": (82, 208), "Bismuth": (83, 209), "Polonium": (84, 209), "Astatine": (85, 210),
    "Radon": (86, 222), "Francium": (87, 223), "Radium": (88, 226), "Actinium": (89, 227), "Thorium": (90, 232),
    "Protactinium": (91, 231), "Uranium": (92, 238), "Neptunium": (93, 237), "Plutonium": (94, 244), "Americium": (95, 243),
    "Curium": (96, 247), "Berkelium": (97, 247), "Californium": (98, 251), "Einsteinium": (99, 252), "Fermium": (100,257),
    "Mendelevium": (101,258), "Nobelium": (102,259), "Lawrencium": (103,262), "Rutherfordium": (104,267), "Dubnium": (105,262),
    "Seaborgium": (106,269), "Bohrium": (107,264), "Hassium": (108,269), "Meitnerium": (109,278), "Darmstadtium": (110,281),
    "Roentgenium": (111,282), "Copernicium": (112,285), "Nihonium": (113,286), "Flerovium": (114,289), "Moscovium": (115,289),
    "Livermorium": (116,293), "Tennessine": (117,294), "Oganesson": (118,294)
}

_Z_TO_SYMBOL = {v[0]: k for k,v in _SYMBOL_TO_ELEMENT.items()}
_Z_TO_STABLE = {v[0]: v[1] for k,v in _SYMBOL_TO_ELEMENT.items()}
_Z_TO_NAME = {v[0]: k for k,v in _NAME_TO_ELEMENT.items()}

@dataclass(frozen=True)
class Nucleus(object):
    A: int
    Z: int
    symbol: str = field(init=False)
    name: str = field(init=False)

    def __post_init__(self):
        if self.Z not in _Z_TO_SYMBOL:
            raise ValueError(f'No known element for atomic number {self.Z}')
        symbol = _Z_TO_SYMBOL[self.Z]
        name = _Z_TO_NAME[self.Z]
        object.__setattr__(self, 'symbol', symbol)
        object.__setattr__(self, 'name', name)

    def __str__(self) -> str:
        sym = self.symbol.lower()
        if sym == 'n0':
            return 'neut'
        else:
            return f'{sym}{self.A}'

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def latex_repr(self) -> str:
        return fr'$^{{{self.A}}}$' + self.symbol.capitalize()


def find_isotope(iid: str) -> Nucleus:
    """
    Return a nucleus object corresponding to the given isotope id.

    Arguments
    ---
    iid : str
        The isotope id in the form e.g. Ni56.
        The string must start with the symbol or name of the element, and the full or shorten atomic weight of the isotope.
        For example, Pb07 must be parsed as Lead 207 (82, 207).

    Returns
    ---
    Nucleus
        A nucleus object

    Raises
    ---
    ValueError
        If the isotope id cannot be parsed;
    """

    # Special cases
    if iid.lower() in ['n', 'nt0', 'nt1', 'neut', 'neutron', 'neutrons']:
        return Nucleus(A=1, Z=0)
    if iid.lower() in ['p', 'prot', 'h', 'proton', 'protons']:
        return Nucleus(A=1, Z=1)
    if iid.lower() in ['d', 'deut', 'deuterium']:
        return Nucleus(A=2, Z=1)
    if iid.lower() in ['t', 'trit', 'tritium']:
        return Nucleus(A=3, Z=1)

    match = re.match(r'^([^\d]*)(.*)$', iid)
    if match is None:
        raise ValueError(f'Unrecognised isotope id: {iid}')

    symbol = match[1]
    weight = match[2]
    try:
        Z = _SYMBOL_TO_ELEMENT[symbol.strip().title()][0]
    except KeyError:
        raise ValueError(f'Not a valid isotope id: {iid}')

    A_parsed = int(weight)
    if (A_parsed < 100):
        if (Z < 35): # Hydrogen to Bromine, A < 100
            A = A_parsed
        elif (Z >= 35 and Z < 51): # Bromine (max(A) = 101) to Tin (min(A) = 99)
            if (A_parsed > 50):
                A = A_parsed
            else:
                A = 100 + A_parsed
        elif (Z >= 51 and Z < 76): # Antimony to Rhenium, 100 < A < 200
            A = 100 + A_parsed
        elif (Z >= 76 and Z < 89): # Osmium (max(A) = 203) to Francium (min(A) = 197)
            if (A_parsed > 50):
                A = 100 + A_parsed
            else:
                A = 200 + A_parsed
        elif (Z >= 89): # Over Radium, A >= 200
            A = 200 + A_parsed
    else:
        A = A_parsed

    if A < Z:
        raise ValueError(f'Invalid mass number: A={A} < Z={Z}')

    return Nucleus(A=A, Z=Z)


def get_most_stable_isotope(val: Union[int, str]) -> Nucleus:
    if isinstance(val, (int, np.int32, np.int64)):
        return Nucleus(A=_Z_TO_STABLE[val], Z=val)
    elif isinstance(val, str):
        if val in _NAME_TO_ELEMENT:
            return Nucleus(A=_NAME_TO_ELEMENT[val][1], Z=_NAME_TO_ELEMENT[val][0])
        elif val in _SYMBOL_TO_ELEMENT:
            return Nucleus(A=_SYMBOL_TO_ELEMENT[val][1], Z=_SYMBOL_TO_ELEMENT[val][0])
        else:
            raise ValueError(f'Unknown element: {val}')
    else:
        raise TypeError(f'Unhandled input type: {type(val)})')


def sort_isotope_id(iid: str):
    n = find_isotope(iid)
    return (n.Z, n.A)


def sort_nucleus(n: Nucleus):
    return (n.Z, n.A)

