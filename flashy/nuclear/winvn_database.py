import numpy as np
try:
    from importlib import resources
except ModuleNotFoundError:
    import importlib_resources as resources
from .nucleus import sort_isotope_id


# TODO Move constants someplace generic
MP = 938.272046 # Proton mass MeV/c2
ME = 0.5110     # Electron mass MeV/c2
MN = 939.565379 # Neutron mass MeV/c2
AMU = 931.494061# Atomic mass unit MeV/c2
PEXC = (MP + ME - AMU) # Proton mass excess
NEXC = (MN - AMU)      # Neutron mass excess
CLIGHT = 2.99792458e10 # Speed of light cm/s
AVO = 6.0221367e23 # Avogadro number
H = 6.6260755e-27 # Planck constant erg.s
KERG = 1.380658e-16 # Boltzmann constant cgs

# Temperature grid in GK for the partition functions
_WINVN_TEMP_GRID = np.array( \
    [1e-1, 1.5e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, \
     1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] \
)


class Nuclide(object):
    _name: str
    _A: float
    _Z: float
    _N: float
    _spin: float
    _mass_excess: float
    _pf: np.ndarray
    
    def __init__(self, name, A, Z, N, spin, mass_excess, pf):
        self._name = name
        self._A = A
        self._Z = Z
        self._N = N
        self._spin = spin
        self._mass_excess = mass_excess
        self._pf = np.asarray(pf)

    def get_A(self):
        return self._A

    def get_Z(self):
        return self._Z

    def get_N(self):
        return self._N

    def get_spin(self):
        return self._spin

    def get_mass_excess(self):
        return self._mass_excess

    def get_binding_energy(self):
        return self._Z*PEXC + self._N*NEXC - self._mass_excess

    def get_pf(self):
        return self._pf.copy()
    
    def eval_pf(self, t9):
        # Simple linear interpolation
        #if interpolation == 'linear':
        return np.interp(t9, _WINVN_TEMP_GRID, self._pf)
        #elif interpolation == 'linlog':
            #return np.exp(np.interp(t9, _WINVN_TEMP_GRID, np.log10(self._pf)))
        #else:
            #raise RuntimeError(f'Invalid interpolation scheme {interpolation}')


class WinvnDatabase(object):
    _initialised: bool
    _names: list
    _nuclides: dict

    def __init__(self):
        self._read_winvn()
        self._names.sort(key=sort_isotope_id)

    def _read_winvn(self):
        self._initialised = False
        winvn_path = resources.files("flashy.nuclear").joinpath('data/winvne_v2.0.dat')
        with winvn_path.open('r') as winvn:
            # skip first 2 lines
            winvn.readline()
            winvn.readline()

            # skip nuclides name. Last is repeated twice
            count = 0
            previous_name = ""
            name = winvn.readline()
            while (name != previous_name):
                previous_name = name
                name = winvn.readline()
                count += 1

            # Read database
            self._names = []
            self._nuclides = {}
            while True:
                line = winvn.readline().strip()
                if not line:
                    break

                # Read nuclide definition
                header = line.split()
                name        = header[0]
                A           = float(header[1])
                Z           = float(header[2])
                N           = float(header[3])
                spin        = float(header[4])
                mass_excess = float(header[5])
                # Read partition function coefficients
                pf = np.loadtxt([winvn.readline() for i in range(3)])
                pf = np.concatenate(pf)

                self._names.append(name)
                self._nuclides[name] = Nuclide(name, A, Z, N, spin, mass_excess, pf)
        
        self._initialised = True

    def __getitem__(self, index):
        return self.get_nuclide(index)

    def get_nuclide(self, index):
        if isinstance(index, int):
            return self._nuclides[self._names[i]]
        elif isinstance(index, str):
            return self._nuclides[index]
        else:
            raise RuntimeError(f'Invalid index type {type(index)}')

    def size(self):
        return len(self._nuclides)

    def names(self):
        return self._names

    def contains(self, name):
        return name in self._names

    def contains_nuclide(self, Z, A):
        if A < Z:
            raise RuntimeError(f'Invalid atomic configuration Z={Z}, A={A}')
        pred = lambda item: (round(item.get_A()) == round(A)) and (round(item.get_Z()) == round(Z))
        return next(filter(pred, self._nuclides.values()), None) is not None
