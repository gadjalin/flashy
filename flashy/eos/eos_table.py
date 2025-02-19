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
    _zeroPoint: float

    # Table fields list and indexing
    _vars: list
    _tableData: np.ndarray
    _varToIndex: dict
    _cs2Index: int

    # Table grid
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

    def __init__(self, filename):
        super(EosTable, self).__init__(filename, 'r')
        self._init_table()

    @classmethod
    def from_file(cls, filename):
        """
        Initialise EoS table from hdf5 file
        """
        obj = cls(filename)
        return obj

    def _init_table(self):
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

        # Get EoS energy shift
        self._zeroPoint = self['/energy_shift'][()][0]

        # Get the actual quantities available in the table
        self._vars = [k for k in list(self.keys()) if self[k].shape == self._dataShape]
        self._vars.sort()

        # Create EOS_VAR index map
        indices = (EOS_VAR.PRES, EOS_VAR.EINT, EOS_VAR.GAMC, EOS_VAR.ABAR, EOS_VAR.ZBAR, EOS_VAR.ENTR)
        fields = ('logpress', 'logenergy', 'gamma', 'Abar', 'Zbar', 'entropy')
        self._varToIndex = {}
        for index,var in zip(indices, fields):
            self._varToIndex[index] = self._vars.index(var)

        # Sound speed index
        self._cs2Index = self._vars.index('cs2')
        
        # load EoS data
        varData = np.array([self['/' + var][()] for var in self._vars])
        # change the shape for the interpolator
        self._tableData = np.rollaxis(varData, 0, varData.ndim)

        # initalize interpolator
        #TDOO: Add the kwargs as input for the interpolator
        self._tableInterpolator = RegularGridInterpolator((self._Ye, self._logTemp, self._logRho), self._tableData)

    def energy_shift(self) -> float:
        return self._zeroPoint

    def fields(self):
        return self._vars

    def __call__(self, mode: EOS_MODE, eosData: list | np.ndarray):
        return self.eos_nuclear(mode, eosData)

    def eos_nuclear(self, mode: EOS_MODE, eosData: list | np.ndarray):
        """
        Call the EoS table
        It linearly interpolates the tabulated EoS table.

        TODO Implement different eos modes
        
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
        assert (eosData is not None) and (len(eosData) % EOS_VAR.NUM == 0), "eosData must be an array of size N*EOS_VAR.NUM"

        vecLen = int(len(eosData) / EOS_VAR.NUM)
        # Make sure this is a numpy array, for convenience
        # and copy to preserve input data on the callers side
        eosData = np.asarray(eosData, dtype=np.float64).copy()

        # Convenience constants to index eosData
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
        #self.xAbar = eosData[abarStart:abarStart+vecLen]
        #self.xZbar = eosData[zbarStart:zbarStart+vecLen]
        #self.xEner = eosData[eintStart:eintStart+vecLen]
        #self.xEntr = eosData[entrStart:entrStart+vecLen]
        #self.xPres = eosData[presStart:presStart+vecLen]

        self._check_bounds()

        self.eos_internal(0, vecLen)

        eosData[presStart:presStart+vecLen] = self.xPres
        eosData[eintStart:eintStart+vecLen] = self.xEner + self._zeroPoint
        eosData[entrStart:entrStart+vecLen] = self.xEntr
        eosData[gamcStart:gamcStart+vecLen] = self.xCs2 * self.xDens/self.xPres
        eosData[abarStart:abarStart+vecLen] = self.xAbar
        eosData[zbarStart:zbarStart+vecLen] = self.xZbar

        return eosData

    def eos_internal(self, first:int, vecLen: int):
        # Interpolate quantities from table
        result = self.eos_table(self.xDens[first:first+vecLen], self.xTemp[first:first+vecLen], self.xYe[first:first+vecLen])

        # Set appropriate variables
        self.xPres = result[:,self._varToIndex[EOS_VAR.PRES]]
        self.xEner = result[:,self._varToIndex[EOS_VAR.EINT]]
        self.xEntr = result[:,self._varToIndex[EOS_VAR.ENTR]]
        self.xCs2  = result[:,self._cs2Index]
        self.xAbar = result[:,self._varToIndex[EOS_VAR.ABAR]]
        self.xZbar = result[:,self._varToIndex[EOS_VAR.ZBAR]]

    # TODO Rework this
    def eos_table(self, xDens, xTemp, xYe):
        dens = np.array(xDens)
        temp = np.array(xTemp)
        ye = np.array(xYe)
        
        temp *= _KTOMEV

        dens = np.log10(dens)
        temp = np.log10(temp)

        # setup interpolation
        coords = np.array([ye,temp,dens]).T

        result = self._tableInterpolator(coords)

        # Apply energy shift
        result[:,self._varToIndex[EOS_VAR.EINT]] = 10.**result[:,self._varToIndex[EOS_VAR.EINT]] - self._zeroPoint
        result[:,self._varToIndex[EOS_VAR.PRES]] = 10.**result[:,self._varToIndex[EOS_VAR.PRES]]

        return result

    def _check_bounds(self):
        if np.any(self.xDens < self._minRho) or np.any(self.xDens > self._maxRho):
            raise RuntimeError('Density out of bounds')
        if np.any(self.xTemp < self._minTemp) or np.any(self.xTemp > self._maxTemp):
            raise RuntimeError('Temperature out of bounds')
        if np.any(self.xYe < self._minYe) or np.any(self.xYe > self._maxYe):
            print(self._minYe, self._maxYe)
            print(self._minTemp, self._maxTemp)
            print(self._minRho, self._maxRho)
            raise RuntimeError('Electron fraction out of bounds')
