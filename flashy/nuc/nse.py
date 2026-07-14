# For type hints
from __future__ import annotations
from typing import List, Dict, Optional, Union, Tuple

import numpy as np

from .winvn_database import WinvnDatabase, WinvnNucleus, _WINVN_TEMP_GRID
from ..constants import K_MEV


_winvn_database = None

SQRT2 = np.sqrt(2.0)
_SAHA_CONST = 3.2047203586e3 # This is what you get after reducing all constants in saha

_NSE_MAX_NEWTON = 10000
_NSE_TOL = 1e-5
_NSE_SMALL = 1e-10


class NSESolver(object):
    _nuclei_names: List[str]
    _nuclei: List[WinvnNucleus]

    _Yi: np.ndarray
    _Yp: float
    _Yn: float

    _Ai: np.ndarray
    _Zi: np.ndarray
    _Ni: np.ndarray
    _Bi: np.ndarray
    _spini: np.ndarray
    _PFi: np.ndarray
    _Gi: np.ndarray

    _ln_consts: np.ndarray
    _ln_dists: np.ndarray
    _ln_saha_factors: np.ndarray

    def __init__(self, network: Optional[List[str]] = None):
        global _winvn_database

        if _winvn_database is None:
            _winvn_database = WinvnDatabase()

        # Setup network
        if network is not None:
            if len(network) == 0:
                raise ValueError('Network must not be empty')

            network = self._sortNetwork(network)
        else:
            network = _winvn_database.species

        self._nuclei = [_winvn_database[sp] for sp in network]

        self._Ai    = np.atleast_1d([nuc.A           for nuc in self._nuclei])
        self._Zi    = np.atleast_1d([nuc.Z           for nuc in self._nuclei])
        self._Ni    = np.atleast_1d([nuc.N           for nuc in self._nuclei])
        self._Bi    = np.atleast_1d([nuc.B           for nuc in self._nuclei])
        self._spini = np.atleast_1d([(2.*nuc.spin+1) for nuc in self._nuclei])
        self._PFi   = np.atleast_1d([nuc.pf          for nuc in self._nuclei])
        self._Gi    = np.ones(len(self._nuclei))

        self._Yn = 0.5
        self._Yp = 0.5

    def solve(self, dens, temp, Ye, reset_solver=False):
        # Set initial conditions
        self._Yi = np.zeros(len(self._nuclei))
        if reset_solver or self._Yn == 0.5:
            self._Yn = 1.0 - Ye
            self._Yp = Ye

        # Initialise constants of saha equation
        self._init_solver(dens, temp)

        # Start Newton-Raphson
        self._solve_nse(dens, temp, Ye)

        # Calculate final abundances
        self._saha()

    @property
    def Ai(self) -> np.ndarray:
        return self._Ai

    @property
    def Zi(self) -> np.ndarray:
        return self._Zi

    @property
    def Ni(self) -> np.ndarray:
        return self._Ni

    @property
    def Yi(self) -> np.ndarray:
        return self._Yi

    @property
    def Yn(self) -> float:
        return self._Yn

    @property
    def Yp(self) -> float:
        return self._Yp

    @property
    def Yalpha(self) -> float:
        return self._Yi[self._nuclei_names.index('he4')]

    @property
    def Yheavy(self) -> float:
        heavyi = self._nuclei_names.index('he4') + 1
        return np.sum(self._Yi[heavyi:])

    @property
    def Xi(self) -> np.ndarray:
        return self.Yi*self.Ai

    @property
    def Xn(self) -> float:
        return self.Yn

    @property
    def Xp(self) -> float:
        return self.Yp

    @property
    def Xalpha(self) -> float:
        return self.Yalpha*4.0

    @property
    def Xheavy(self) -> float:
        heavyi = self._nuclei_names.index('he4') + 1
        return np.sum(self._Yi[heavyi:]*self._Ai[heavyi:])

    @property
    def Abar(self) -> float:
        return 1./np.sum(self._Yi)

    @property
    def Zbar(self) -> float:
        return np.sum(self._Yi*self._Zi)/np.sum(self._Yi)

    @property
    def Aheavy(self) -> float:
        heavyi = self._nuclei_names.index('he4') + 1
        return np.sum(self._Yi[heavyi:]*self._Ai[heavyi:])/np.sum(self._Yi[heavyi:])

    @property
    def Zheavy(self) -> float:
        heavyi = self._nuclei_names.index('he4') + 1
        return np.sum(self._Yi[heavyi:]*self._Zi[heavyi:])/np.sum(self._Yi[heavyi:])

    @property
    def finab(self) -> np.ndarray:
        return np.array([(A, Z, X) for A,Z,X in zip(self.Ai, self.Zi, self.Xi)], dtype=[(x, float) for x in ['A', 'Z', 'X']])

    @property
    def finabA(self) -> np.ndarray:
        As, iAs = np.unique(self._Ai, return_inverse=True)
        Xs = np.bincount(iAs, weights=self._Yi*self._Ai)
        return np.array([(A, X) for A,X in zip(As, Xs)], dtype=[(x, float) for x in ['A', 'X']])

    @property
    def finabZ(self) -> np.ndarray:
        Zs, iZs = np.unique(self._Zi, return_inverse=True)
        Xs = np.bincount(iZs, weights=self._Yi*self._Ai)
        return np.array([(Z, X) for Z,X in zip(Zs, Xs)], dtype=[(x, float) for x in ['Z', 'X']])

    def _init_solver(self, dens, temp):
        t9 = temp*1e-9

        if (t9 <= _WINVN_TEMP_GRID[0]):
            self._Gi = self._spini * self._PFi[:,0]
        elif (t9 >= _WINVN_TEMP_GRID[-1]):
            self._Gi = self._spini * self._PFi[:,-1]
        else:
            t9i = np.searchsorted(_WINVN_TEMP_GRID, t9)

            t9left, t9right = _WINVN_TEMP_GRID[t9i-1], _WINVN_TEMP_GRID[t9i]
            pfleft, pfright = self._PFi[:,t9i-1], self._PFi[:,t9i]

            slope = (pfright - pfleft) / (t9right - t9left)
            self._Gi = self._spini * (slope*t9 - slope*t9right + pfright)

        self._ln_consts = (self._Ai - 1)*np.log(_SAHA_CONST * dens * temp**(-3./2.)) \
                         + 1.5*np.log(self._Ai) \
                         - self._Ai*np.log(2)
        self._ln_dists  = self._Bi/(K_MEV*temp)

        self._ln_saha_factors = np.log(self._Gi) + self._ln_dists + self._ln_consts

    def _saha(self):
        # Calculate abundances
        self._Yi[0] = self._Yn
        self._Yi[1] = self._Yp
        # Prevent large numbers
        self._Yi[2:] = np.exp(self._ln_saha_factors[2:] + self._Ni[2:]*np.log(self._Yn) + self._Zi[2:]*np.log(self._Yp))
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
            m = 1.0 - np.sum(self._Yi*self._Ai)
            # Prevent insane values
            if (np.log2(abs(m)) > 33.0):
                if (Ye > 0.4):
                    self._Yn /= SQRT2
                    continue
                else:
                    self._Yp /= SQRT2
                    continue

            # Charge conservation
            c = Ye - np.sum(self._Yi*self._Zi)

            # Calculate derivatives and solve Jacobian
            dmdYn = np.sum(self._Yi[2:]*self._Ai[2:]*self._Ni[2:])/self._Yn + 1.0
            dmdYp = np.sum(self._Yi[2:]*self._Ai[2:]*self._Zi[2:])/self._Yp + 1.0
            dcdYn = np.sum(self._Yi[2:]*self._Zi[2:]*self._Ni[2:])/self._Yn
            dcdYp = np.sum(self._Yi[2:]*self._Zi[2:]*self._Zi[2:])/self._Yp + 1.0

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

    def _sortNetwork(self, network: List[str]):
        if 'neut' in network:
            network[network.index('neut')] = 'n'
        if 'prot' in network:
            network[network.index('prot')] = 'p'
        if 'h1' in network:
            network[network.index('h1')] = 'p'
        if 'h2' in network:
            network[network.index('h2')] = 'd'
        if 'h3' in network:
            network[network.index('h3')] = 't'

        if 'p' not in network:
            network = ['p'] + network
        if 'n' not in network:
            network = ['n'] + network

        network.sort(key=lambda x: (self._winvn_database[x].Z, self._winvn_database[x].A))

        for sp in network:
            if not _winvn_database.contains(sp):
                raise RuntimeError(f'Invalid species: {sp}')

        return network

