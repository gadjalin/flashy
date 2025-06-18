import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from ._eos_var import EOS_VAR
from ._eos_mode import EOS_MODE


# TODO Move constants to generic place
_KTOMEV = 8.617385687984878e-11
_MEVTOERG = 1.60217733e-6


class EosTable(h5py.File):
    # Zero-point shift
    _energyShift: float

    # Table fields list and indexing
    _vars: list
    _tableData: np.ndarray

    # Grid info
    _logRho: np.ndarray
    _logTemp: np.ndarray
    _Ye: np.ndarray
    _dataShape: tuple

    # Internal eos data rows
    xDens: np.ndarray
    xTemp: np.ndarray
    xYe: np.ndarray
    xAbar: np.ndarray
    xZbar: np.ndarray
    xPres: np.ndarray
    xEner: np.ndarray
    xEntr: np.ndarray
    xCs2: np.ndarray

    # Table bounds
    _minRho: float
    _maxRho: float
    _minTemp: float
    _maxTemp: float
    _minYe: float
    _maxYe: float

    def __init__(self, filename, interp_method='linear'):
        super(EosTable, self).__init__(filename, 'r')
        self._init_table()

    @classmethod
    def from_file(cls, filename, interp_method='linear'):
        """
        Initialise EoS table from hdf5 file
        """
        obj = cls(filename)
        return obj

    def _init_table(self, interp_method='linear'):
        # Get the table grid
        self._logRho = self['/logrho'][()]
        self._logTemp = self['/logtemp'][()]
        self._Ye = self['/ye'][()]
        self._dataShape = (len(self._Ye), len(self._logTemp), len(self._logRho))

        # Table bounds
        self._minRho = 10**np.min(self._logRho)
        self._maxRho = 10**np.max(self._logRho)
        self._minTemp = 10**np.min(self._logTemp)/_KTOMEV
        self._maxTemp = 10**np.max(self._logTemp)/_KTOMEV
        self._minYe = np.min(self._Ye)
        self._maxYe = np.max(self._Ye)

        # Get the actual tabulated quantities available in the table
        self._vars = [k for k in list(self.keys()) if self[k].shape == self._dataShape]
        self._vars.sort()

        # Get EoS energy shift
        self._energyShift = self['/energy_shift'][()][0]

        # load EoS data
        varData = np.array([self['/' + var][()] for var in self._vars])
        self._tableData = np.moveaxis(varData, 0, -1)

        # initalize interpolator
        self._tableInterpolator = RegularGridInterpolator((self._Ye, self._logTemp, self._logRho), self._tableData, \
                                                          method=interp_method, bounds_error=False, fill_value=None)

    def energy_shift(self) -> float:
        return self._energyShift

    def field_list(self) -> list:
        return self._vars

    def __call__(self, mode: EOS_MODE, eosData: list | np.ndarray):
        return self.eos_nuclear(mode, eosData)

    def call(self, mode: EOS_MODE, eosData: list | np.ndarray):
        """
        Call the EoS for every zone in eosData
        It interpolates the tabulated EoS table according to the
        interpolation method requested at initialisation (default is linear).

        Parameters
        ----------
        mode : EOS_MODE
            The mode in which to calculate the EOS

        eosData : list
            An array containing the eosData,
            as layed out by the EOS_VAR enumeration.
            The data of multiple cells can be passed at once,
            in which case the array must be of size vecLen*EOS_VAR.NUM,
            where each quantity is contained in continuous blocks,
            e.g. the densities for each cell is stored in the slice
            (EOS_VAR.DENS*vecLen : (EOS_VAR.DENS + 1)*vecLen).
        """
        if (eosData is None) or (len(eosData) % EOS_VAR.NUM != 0): 
            raise RuntimeError("eosData must be an array of size N*EOS_VAR.NUM")

        vecLen = int(len(eosData) / EOS_VAR.NUM)
        # Make sure this is a numpy array, for convenience
        # and copy to preserve input data on the callers side
        eosData = np.asarray(eosData, dtype=np.float64).copy()

        # Convenience constants for indexing eosData
        presStart = EOS_VAR.PRES*vecLen
        densStart = EOS_VAR.DENS*vecLen
        tempStart = EOS_VAR.TEMP*vecLen
        eintStart = EOS_VAR.EINT*vecLen
        gamcStart = EOS_VAR.GAMC*vecLen
        abarStart = EOS_VAR.ABAR*vecLen
        zbarStart = EOS_VAR.ZBAR*vecLen
        entrStart = EOS_VAR.ENTR*vecLen

        self.xDens = eosData[densStart:densStart+vecLen]
        self.xTemp = eosData[tempStart:tempStart+vecLen]
        self.xYe   = eosData[zbarStart:zbarStart+vecLen] / eosData[abarStart:abarStart+vecLen]
        self.xEner = eosData[eintStart:eintStart+vecLen]
        self.xPres = eosData[presStart:presStart+vecLen]

        self._check_bounds()

        # Call eos table for each zone
        zones = self.nuc_eos_zone(mode, self.xDens, self.xTemp, self.xYe, self.xEner, self.xPres)

        # Set appropriate variables
        self.xTemp = 10**zones['logtemp']/_KTOMEV
        self.xPres = 10**zones['logpress']
        self.xEner = 10**zones['logenergy'] - self._energyShift
        self.xEntr = zones['entropy']
        self.xCs2  = zones['cs2']
        self.xAbar = zones['Abar']
        self.xZbar = zones['Zbar']

        eosData[densStart:densStart+vecLen] = self.xDens
        eosData[tempStart:tempStart+vecLen] = self.xTemp
        eosData[presStart:presStart+vecLen] = self.xPres
        eosData[eintStart:eintStart+vecLen] = self.xEner + self._energyShift
        eosData[entrStart:entrStart+vecLen] = self.xEntr
        eosData[gamcStart:gamcStart+vecLen] = self.xCs2 * self.xDens/self.xPres
        eosData[abarStart:abarStart+vecLen] = self.xAbar
        eosData[zbarStart:zbarStart+vecLen] = self.xZbar

        return eosData

    def nuc_eos_zone(self, mode: EOS_MODE, xrho, xtemp, xye, xenr = None, xprs = None):
        if mode == EOS_MODE.DENS_EI:
            raise RuntimeError('Unsupported EOS mode')
        elif mode == EOS_MODE.DENS_PRES:
            raise RuntimeError('Unsupported EOS mode')
        elif mode != EOS_MODE.DENS_TEMP:
            raise RuntimeError('Unknown EOS mode')

        # Have rho, temp, ye
        logrho = np.log10(xrho)
        logtemp = np.log10(np.asarray(xtemp) * _KTOMEV)
        ye = np.asarray(xye)

        # table interpolation
        zone_points = np.asarray([ye, logtemp, logrho]).T
        data = self._tableInterpolator(zone_points)

        zones = {}
        zones['logrho'] = logrho[()]
        zones['logtemp'] = logtemp[()]
        zones['ye'] = ye[()]
        for i in range(len(self._vars)):
            zones[self._vars[i]] = data[:,i]

        return zones

    def nuc_eos_grid(self, xrho, xtemp, xye):
        logrho = np.log10(xrho)
        logtemp = np.log10(np.asarray(xtemp) * _KTOMEV)
        ye = np.asarray(xye)

        YE, LOGTEMP, LOGRHO = np.meshgrid(ye, logtemp, logrho, indexing='ij')
        grid_points = np.stack([YE.ravel(), LOGTEMP.ravel(), LOGRHO.ravel()], axis=-1)
        data = self._tableInterpolator(grid_points)
        data = data.reshape(self._tableData.shape)

        grid = {}
        for i in range(len(self._vars)):
            grid[self._vars[i]] = data[:,:,:,i]

        return grid

    def _check_bounds(self):
        if np.any(self.xDens < self._minRho) or np.any(self.xDens > self._maxRho):
            raise RuntimeError('Density out of bounds')
        if np.any(self.xTemp < self._minTemp) or np.any(self.xTemp > self._maxTemp):
            raise RuntimeError('Temperature out of bounds')
        if np.any(self.xYe < self._minYe) or np.any(self.xYe > self._maxYe):
            raise RuntimeError('Electron fraction out of bounds')

