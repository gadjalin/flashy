import numpy as np
from .winvn_database import WinvnDatabase, _WINVN_TEMP_GRID
from .nucleus import sort_isotope_id


_winvn_database = None

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
H_EV = 4.135667696e-15 # Planck constant eV.s
K_ERG = 1.380658e-16 # Boltzmann constant erg/K
K_EV = 8.6173e-5 # Boltzmann constant eV/K
K_MEV = K_EV*1e-6 # Boltzmann constant MeV/K
SQRT2 = np.sqrt(2.0)

_SAHA_CONST = 3.2047203586e3 # This is what you get after reducing all constants in saha

_NSE_MAX_NEWTON = 10000
_NSE_TOL = 1e-5
_NSE_SMALL = 1e-10


class NSE(object):
    _network_names: list
    _network: list

    _Yi: np.ndarray
    _Yp: float
    _Yn: float

    _As: np.ndarray
    _Zs: np.ndarray
    _Ns: np.ndarray
    _Bs: np.ndarray
    _PFs: np.ndarray
    _ln_consts: np.ndarray
    _ln_dists: np.ndarray
    _ln_saha_factors: np.ndarray
    
    def __init__(self, nuclide_names=None):
        global _winvn_database

        if _winvn_database is None:
            _winvn_database = WinvnDatabase()

        # Setup network
        if nuclide_names is not None:
            self._network_names = self._sanitiseLabels(nuclide_names)
        else:
            self._network_names = _winvn_database.names()

        self._network = [_winvn_database.get_nuclide(name) for name in self._network_names]

        self._As    = np.atleast_1d([nuc.get_A()              for nuc in self._network])
        self._Zs    = np.atleast_1d([nuc.get_Z()              for nuc in self._network])
        self._Ns    = np.atleast_1d([nuc.get_N()              for nuc in self._network])
        self._Bs    = np.atleast_1d([nuc.get_binding_energy() for nuc in self._network])
        self._spins = np.atleast_1d([(2.*nuc.get_spin()+1)    for nuc in self._network])
        self._PFs   = np.atleast_1d([nuc.get_pf()             for nuc in self._network])
        self._Gs    = np.ones(len(self._network))

        self._Yn = 0.5
        self._Yp = 0.5

    def solve(self, dens, temp, Ye, reset_solver=False):
        # Set initial conditions
        self._Yi = np.zeros(len(self._network))
        if reset_solver or self._Yn == 0.5:
            self._Yn = 1.0 - Ye
            self._Yp = Ye

        # Initialise constants of saha equation
        self._init_solver(dens, temp)

        # Start Newton-Raphson
        self._solve_nse(dens, temp, Ye)

        # Calculate final abundances
        self._saha()

    def get_Yi(self):
        return self._Yi.copy()

    def get_Yn(self):
        return self._Yn

    def get_Yp(self):
        return self._Yp

    def get_Yalpha(self):
        return self._Yi[self._network_names.index('he4')]

    def get_Yheavy(self):
        heavyStart = self._network_names.index('he4') + 1
        if heavyStart >= len(self._network_names):
            return 0.0
        else:
            return np.sum(self._Yi[heavyStart:])

    def get_Xi(self):
        return self.get_Yi()*self._As

    def get_Xn(self):
        return self.get_Yn()

    def get_Xp(self):
        return self.get_Yp()

    def get_Xalpha(self):
        return self.get_Yalpha()*4.0

    def get_Xheavy(self):
        heavyStart = self._network_names.index('he4') + 1
        if heavyStart >= len(self._network_names):
            return 0.0
        else:
            return np.sum(self._Yi[heavyStart:]*self._As[heavyStart:])

    def get_Abar(self):
        return 1./np.sum(self._Yi)

    def get_Zbar(self):
        return np.sum(self._Yi*self._Zs)/np.sum(self._Yi)

    def get_AbarHeavy(self):
        heavyStart = self._network_names.index('he4') + 1
        if heavyStart >= len(self._network_names):
            return 0.0
        else:
            return np.sum(self._Yi[heavyStart:]*self._As[heavyStart:])/np.sum(self._Yi[heavyStart:])

    def get_ZbarHeavy(self):
        heavyStart = self._network_names.index('he4') + 1
        if heavyStart >= len(self._network_names):
            return 0.0
        else:
            return np.sum(self._Yi[heavyStart:]*self._Zs[heavyStart:])/np.sum(self._Yi[heavyStart:])

    def _init_solver(self, dens, temp):
        t9 = temp*1e-9

        if (t9 <= _WINVN_TEMP_GRID[0]):
            self._Gs = self._spins * self._PFs[:,0]
        elif (t9 >= _WINVN_TEMP_GRID[-1]):
            self._Gs = self._spins * self._PFs[:,-1]
        else:
            t9i = np.searchsorted(_WINVN_TEMP_GRID, t9)
        
            t9left, t9right = _WINVN_TEMP_GRID[t9i-1], _WINVN_TEMP_GRID[t9i]
            pfleft, pfright = self._PFs[:,t9i-1], self._PFs[:,t9i]
        
            slope = (pfright - pfleft) / (t9right - t9left)
            self._Gs = self._spins * (slope*t9 - slope*t9right + pfright)

        self._ln_consts = (self._As - 1)*np.log(_SAHA_CONST * dens * temp**(-3./2.)) \
                         + 1.5*np.log(self._As) \
                         - self._As*np.log(2)
        self._ln_dists  = self._Bs/(K_MEV*temp)

        self._ln_saha_factors = np.log(self._Gs) + self._ln_dists + self._ln_consts

    def _saha(self):
        # Calculate abundances
        self._Yi[0] = self._Yn
        self._Yi[1] = self._Yp
        # Prevent large numbers
        self._Yi[2:] = np.exp(self._ln_saha_factors[2:] + self._Ns[2:]*np.log(self._Yn) + self._Zs[2:]*np.log(self._Yp))
        # This slows down A LOT
        #for i in range(2, len(self._Yi)):
        #    if ln_saha[i] < -80.0:
        #        self._Yi[i] = 0.0
        #    elif ln_saha[i] > 80.0:
        #        self._Yi[i] = 100.0
        #    else:
        #        self._Yi[i] = np.exp(ln_saha[i])

    def _solve_nse(self, dens, temp, Ye):
        for i in range(_NSE_MAX_NEWTON):
            # Calculate abundances
            self._saha()
            
            # Mass conservation
            m = 1.0 - np.sum(self._Yi*self._As)
            # Prevent insane values
            if (np.log2(abs(m)) > 33.0):
                if (Ye > 0.4):
                    self._Yn /= SQRT2
                    continue
                else:
                    self._Yp /= SQRT2
                    continue

            # Charge conservation
            c = Ye - np.sum(self._Yi*self._Zs)

            # Calculate derivatives and solve Jacobian
            dmdYn = np.sum(self._Yi[2:]*self._As[2:]*self._Ns[2:])/self._Yn + 1.0
            dmdYp = np.sum(self._Yi[2:]*self._As[2:]*self._Zs[2:])/self._Yp + 1.0
            dcdYn = np.sum(self._Yi[2:]*self._Zs[2:]*self._Ns[2:])/self._Yn
            dcdYp = np.sum(self._Yi[2:]*self._Zs[2:]*self._Zs[2:])/self._Yp + 1.0

            detJ = dmdYn*dcdYp - dcdYn*dmdYp
            delYn = 1./detJ * (dcdYp*m - dmdYp*c)
            delYp = 1./detJ * (dmdYn*c - dcdYn*m)

            # Convergence
            if (max(abs(m), abs(c)) < _NSE_TOL and max(abs(delYn), abs(delYp)) < _NSE_TOL):
                break

            # Constrain abundances to positive values
            if (delYn > 0.0 or self._Yn > abs(delYn)):
                self._Yn += delYn
            if (delYp > 0.0 or self._Yp > abs(delYp)):
                self._Yp += delYp
        else:
            raise RuntimeError(f'NSE did not converge rho={dens} temp={temp} ye={Ye}')

        #print(i)

    def _sanitiseLabels(self, labels):
        if 'neut' in labels:
            labels[labels.index('neut')] = 'n'
        if 'h1' in labels:
            labels[labels.index('h1')] = 'p'
        if 'h2' in labels:
            labels[labels.index('h2')] = 'd'
        if 'h3' in labels:
            labels[labels.index('h3')] = 't'

        if 'n' in labels:
            labels.remove('n')
        if 'p' in labels:
            labels.remove('p')

        labels.sort(key=sort_isotope_id)

        for label in labels:
            if not _winvn_database.contains(label):
                raise RuntimeError(f'Invalid nuclide {label}')
        
        return ['n', 'p'] + labels
