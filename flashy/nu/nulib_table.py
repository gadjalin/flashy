from __future__ import annotations
from typing import Tuple, List, Dict
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from ..constants import KELVIN_TO_MEV


class NulibTable(h5py.File):
    # Metadata
    _n_species: int
    _n_groups: int
    _n_rho: int
    _n_temp: int
    _n_ye: int
    _n_Itemp: int
    _n_Ieta: int
    _eas_shape: Tuple[int, ...]
    _ies_shape: Tuple[int, ...]
    _eas_field_list: List[str]
    _ies_field_list: List[str]

    # Grid
    _logrho: np.ndarray
    _logtemp: np.ndarray
    _ye: np.ndarray
    # Inelastic electron scattering
    _logItemp: np.ndarray
    _logIeta: np.ndarray

    # Table bounds
    _min_rho: float
    _max_rho: float
    _min_temp: float
    _max_temp: float
    _min_ye: float
    _max_ye: float
    _min_Itemp: float
    _max_Itemp: float
    _min_Ieta: float
    _max_Ieta: float

    # Bins
    _bin_energies: np.ndarray
    _bin_widths: np.ndarray
    _bin_bottom: np.ndarray
    _bin_top: np.ndarray

    # Tables
    _eas_table: np.ndarray
    _ies_table: np.ndarray
    _eas_interpolator: RegularGridInterpolator
    _ies_interpolator: RegularGridInterpolator

    def __init__(self, file: str, interp_method: str = 'linear'):
        super(NulibTable, self).__init__(file, 'r')
        self._read_table(interp_method)

    @staticmethod
    def load(file: str, interp_method: str = 'linear') -> NulibTable:
        return NulibTable.load_file(file, interp_method)

    @classmethod
    def load_file(cls, file: str, interp_method: str = 'linear') -> NulibTable:
        obj = cls(file, interp_method)
        return obj

    def _read_table(self, interp_method: str) -> None:
        # Read metadata
        self._n_species = self['number_species'][()][0]
        self._n_groups = self['number_groups'][()][0]
        self._n_rho = self['nrho'][()][0]
        self._n_temp = self['ntemp'][()][0]
        self._n_ye = self['nye'][()][0]
        self._n_Itemp = self['Itemp'][()][0]
        self._n_Ieta = self['Ieta'][()][0]
        self._eas_shape = (self._n_groups, self._n_species, self._n_ye, self._n_temp, self._n_rho)
        self._ies_shape = (self._n_groups, self._n_species, self._n_groups, self._n_Ieta, self._n_Itemp)
        
        # Read grid
        self._logrho = np.log10(self['rho_points'][()])
        self._logtemp = np.log10(self['temp_points'][()])
        self._ye = self['ye_points'][()]
        self._logItemp = np.log10(self['temp_Ipoints'][()])
        self._logIeta = np.log10(self['eta_Ipoints'][()])

        # Grid bounds
        self._min_rho = np.min(self._logrho)
        self._max_rho = np.max(self._logrho)
        self._min_temp = np.min(self._logtemp)
        self._max_temp = np.max(self._logtemp)
        self._min_ye = np.min(self._ye)
        self._max_ye = np.max(self._ye)
        self._min_Itemp = np.min(self._logItemp)
        self._max_Itemp = np.max(self._logItemp)
        self._min_Ieta = np.min(self._logIeta)
        self._max_Ieta = np.max(self._logIeta)

        # Read bins metadata
        self._bin_energies = self['neutrino_energies'][()]
        self._bin_widths = self['bin_widths'][()]
        self._bin_bottom = self['bin_bottom'][()]
        self._bin_top = self['bin_top'][()]

        # Get available tabulated quantities
        self._eas_field_list = ['emissivities', 'absorption_opacity', 'scattering_opacity']
        self._ies_field_list = ['inelastic_phi0', 'inelastic_phi1']

        # Read tables and construct interpolators
        self._eas_table = np.array([self[k] for k in self._eas_field_list])
        self._ies_table = np.zeros((2, int(self._n_species*self._n_groups*(self._n_groups+1)/2),self._n_Ieta, self._n_Itemp))
        index = 0
        for i in range(self._n_species):
            for j in range(self._n_groups):
                for k in range(j):
                    self._ies_table[0,index] = self['inelastic_phi0'][j,i,k]
                    self._ies_table[1,index] = self['inelastic_phi1'][j,i,k]
                    index += 1

        # Emissivities
        for i in range(self._n_groups):
            self._eas_table[0,i] *= self._bin_widths[i]
        self._eas_table[0][self._eas_table[0] < 1e-30] = 1e-30
        self._eas_table[0] = np.log10(self._eas_table[0])
        # Absorption
        self._eas_table[1] = np.log10(self._eas_table[1])
        # Scattering
        self._eas_table[2] = np.log10(self._eas_table[2])
        # Inelastic electron scattering
        self._ies_table[0][self._ies_table[0] < 1e-200] = 1e-200
        self._ies_table[0] = np.log10(self._ies_table[0])
        self._ies_table[1] = self._ies_table[1]/(10**self._ies_table[0])

        self._eas_table = np.moveaxis(self._eas_table, [-3, -2, -1], [0, 1, 2])
        self._ies_table = np.moveaxis(self._ies_table, [-2, -1], [0, 1])

        self._eas_interpolator = RegularGridInterpolator((self._ye, self._logtemp, self._logrho), self._eas_table, \
                                                         method=interp_method, bounds_error=False, fill_value=None)
        self._ies_interpolator = RegularGridInterpolator((self._logIeta, self._logItemp), self._ies_table, \
                                                         method=interp_method, bounds_error=False, fill_value=None)

    @property
    def number_species(self) -> int:
        return self._n_species

    @property
    def number_groups(self) -> int:
        return self._n_groups

    @property
    def EAS_field_list(self) -> List[str]:
        return self._eas_field_list

    def EAS(self, xrho: np.ndarray, xtemp: np.ndarray, xye: np.ndarray) -> Dict[str, np.ndarray]:
        xlrho = np.log10(np.atleast_1d(xrho))
        xltemp = np.log10(np.atleast_1d(xtemp) * KELVIN_TO_MEV)
        xye = np.atleast_1d(xye)
        assert(len(xlrho) == len(xltemp) and len(xlrho) == len(xye))

        data_points = np.transpose([xye, xltemp, xlrho])
        data = self._eas_interpolator(data_points)
        data = np.moveaxis(data, 0, -1)
        
        return {field: 10**data[i] for i,field in enumerate(self._eas_field_list)}

    def IES(self, xrho: np.ndarray, xtemp: np.ndarray, xye: np.ndarray, eos: EosTable) -> Dict[str, np.ndarray]:
        xrho = np.atleast_1d(xrho)
        xtemp = np.atleast_1d(xtemp)
        xye = np.atleast_1d(xye)
        assert(len(xrho) == len(xtemp) and len(xrho) == len(xye))

        raise NotImplementedError('Inelastic electron scattering')

