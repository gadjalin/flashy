import numpy as np
from pathlib import Path
import os
try:
    from importlib import resources
except ModuleNotFoundError:
    import importlib_resources as resources
from ._eos_var import EOS_VAR
from ._eos_mode import EOS_MODE


# TODO Move constants to generic place
CLIGHT = 2.99792458e10 # Speed of light (cm/s)
H = 6.6260755e-27 # Planck constant (erg s)
AVO = 6.0221367e23 # Avogadro number
AMU = 1.6605402e-24 # Atomic mass unit (g)
QE = 4.8032068e-10 # Elementary charge (statC)

KERG = 1.380658e-16
KERGAVO = KERG * AVO

SSOL = 5.67051e-5
ASOL = 4.0 * SSOL / CLIGHT
ASOLI3 = ASOL / 3.0
SIONCON = (2.0 * np.pi * AMU * KERG) / H**2

# Table dimensions constants
_EOSIMAX = 261 # This is the number of steps in density (I axis/rows)
_EOSJMAX = 101 # This is the number of steps in temperature (J axis/columns)
_LOGT_LO = 3.0 # Minimum temperature in the table in log
_LOGT_HI = 13.0 # Maximum temperature in the table in log
_LOGD_LO = -12.0 # Minimum density in the table in log
_LOGD_HI = 14.0 # Maximum density in the table in log
_T_STEP = (_LOGT_HI - _LOGT_LO) / (_EOSJMAX - 1)
_D_STEP = (_LOGD_HI - _LOGD_LO) / (_EOSIMAX - 1)

# Free energy tables
_eos_f     = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_fd    = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_ft    = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_fdd   = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_ftt   = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_fdt   = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_fddt  = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_fdtt  = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_fddtt = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)

# Pressure derivatives tables
_eos_dpdf   = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_dpdfd  = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_dpdft  = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_dpdfdt = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)

# Checmical potential tables
_eos_ef   = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_efd  = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_eft  = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_efdt = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)

# Number densitites tables
_eos_xf   = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_xfd  = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_xft  = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)
_eos_xfdt = np.zeros((_EOSIMAX, _EOSJMAX), dtype=np.float64)


# Table temperature and density grid
_EOS_T = 10.0**(_LOGT_LO + np.arange(_EOSJMAX) * _T_STEP)
_EOS_D = 10.0**(_LOGD_LO + np.arange(_EOSIMAX) * _D_STEP)

# Table temperature and density resolution
_EOS_DT = np.diff(_EOS_T)
_EOS_DD = np.diff(_EOS_D)

# Some constants
_EOS_SMALLT = 1e-10
_EOS_TOL = 1e-8
_EOS_MAXNEWTON = 50

# For the uniform backgound coulomb correction
a1, b1, c1, d1cc, e1cc, a2, b2, c2 = np.array([
    -0.898004, 0.96786, 0.220703, -0.86097,
    2.5269, 0.29561, 1.9885, 0.288675
], dtype=np.float64)

# Has the helmholtz been initialised (has helm_table.dat been read)
_initialised = False


class Helmholtz(object):
    # Input quantities
    _densRow: np.ndarray
    _tempRow: np.ndarray
    _abarRow: np.ndarray
    _zbarRow: np.ndarray

    # Derived quantities
    _ptotRow: np.ndarray
    _etotRow: np.ndarray
    _stotRow: np.ndarray
    _dpdRow : np.ndarray
    _dptRow : np.ndarray
    _dedRow : np.ndarray
    _detRow : np.ndarray
    _dsdRow : np.ndarray
    _dstRow : np.ndarray
    _pelRow : np.ndarray
    _neRow  : np.ndarray
    _etaRow : np.ndarray
    _gamcRow: np.ndarray
    _cvRow  : np.ndarray
    _cpRow  : np.ndarray

    def __init__(self):
        if not _initialised:
            self.init()

    def _check_init(self) -> None:
        if not _initialised:
            raise RuntimeError("Helmholtz EOS not initialised!")

    def init(self) -> None:
        global _eos_f, _eos_fd, _eos_ft, _eos_fdd, _eos_ftt, _eos_fdt, _eos_fddt, _eos_fdtt, _eos_fddtt
        global _eos_dpdf, _eos_dpdfd, _eos_dpdft, _eos_dpdfdt
        global _eos_ef, _eos_efd, _eos_eft, _eos_efdt
        global _eos_xf, _eos_xfd, _eos_xft, _eos_xfdt
        global _initialised

        _initialised = False

        # Find path to package helm_table.dat
        helm_table_path = resources.files("flashy.eos").joinpath('data/helm_table.dat')

        # Read the helmholtz table
        # Fastest way to read the table according to ChatGPT
        with helm_table_path.open('r') as helm_table:
            # Inner helper function to read table's tables
            def read_block(width: int):
                blck = [np.empty((_EOSIMAX,_EOSJMAX), dtype=np.float64) for _ in range(width)]
                for j in range(_EOSJMAX):
                    for i in range(_EOSIMAX):
                        row = np.fromstring(helm_table.readline(), dtype=np.float64, sep=' ')
                        for k in range(width):
                            blck[k][i,j] = row[k]
                return blck

            # Read the free energy table
            _eos_f, _eos_fd, _eos_ft, _eos_fdd, _eos_ftt, _eos_fdt, _eos_fddt, _eos_fdtt, _eos_fddtt = read_block(width=9)
        
            # Read the pressure derivative table
            _eos_dpdf, _eos_dpdfd, _eos_dpdft, _eos_dpdfdt = read_block(width=4)
        
            # Read the chemical potential table
            _eos_ef, _eos_efd, _eos_eft, _eos_efdt = read_block(width=4)
        
            # Read the number density table
            _eos_xf, _eos_xfd, _eos_xft, _eos_xfdt = read_block(width=4)
        
        _initialised = True

    def __call__(self, mode: EOS_MODE, eosData: list | np.ndarray, extra_vars: bool = False,
                 force_constant_input: bool = False, coulomb_multiplier: float = 1.0):
        return self.eos_helmholtz(mode, eosData, extra_vars, force_constant_input, coulomb_multiplier)

    def eos_helmholtz(self, mode: EOS_MODE, eosData: list | np.ndarray, extra_vars: bool = False,
                      force_constant_input: bool = False, coulomb_multiplier: float = 1.0):
        """
        Call the Helmholtz EOS

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

        extra_vars : bool
            Should the extra variables listed in EOS_VAR be returned in
            a seperate list as well. This is mainly for analysing separately
            each contributions.
        """
        assert (eosData is not None) and (len(eosData) % EOS_VAR.NUM == 0), "eosData must be an array of size N*EOS_VAR.NUM"

        vecLen = int(len(eosData) / EOS_VAR.NUM)
        # Make sure this is a numpy array, for convenience
        eosData = np.asarray(eosData, dtype=np.float64)

        # Convenience constants to index eosData
        presStart = EOS_VAR.PRES*vecLen
        densStart = EOS_VAR.DENS*vecLen
        tempStart = EOS_VAR.TEMP*vecLen
        eintStart = EOS_VAR.EINT*vecLen
        gamcStart = EOS_VAR.GAMC*vecLen
        abarStart = EOS_VAR.ABAR*vecLen
        zbarStart = EOS_VAR.ZBAR*vecLen
        entrStart = EOS_VAR.ENTR*vecLen

        # Main quantities
        self._densRow = np.array(eosData[densStart:densStart+vecLen], dtype=np.float64)
        self._tempRow = np.array(eosData[tempStart:tempStart+vecLen], dtype=np.float64)
        self._abarRow = np.array(eosData[abarStart:abarStart+vecLen], dtype=np.float64)
        self._zbarRow = np.array(eosData[zbarStart:zbarStart+vecLen], dtype=np.float64)

        self._ptotRow = np.zeros(vecLen, dtype=np.float64)
        self._etotRow = np.zeros(vecLen, dtype=np.float64)
        self._stotRow = np.zeros(vecLen, dtype=np.float64)
        self._dpdRow  = np.zeros(vecLen, dtype=np.float64)
        self._dptRow  = np.zeros(vecLen, dtype=np.float64)
        self._dedRow  = np.zeros(vecLen, dtype=np.float64)
        self._detRow  = np.zeros(vecLen, dtype=np.float64)
        self._dsdRow  = np.zeros(vecLen, dtype=np.float64)
        self._dstRow  = np.zeros(vecLen, dtype=np.float64)
        self._pelRow  = np.zeros(vecLen, dtype=np.float64)
        self._neRow   = np.zeros(vecLen, dtype=np.float64)
        self._etaRow  = np.zeros(vecLen, dtype=np.float64)
        self._gamcRow = np.zeros(vecLen, dtype=np.float64)
        self._cvRow   = np.zeros(vecLen, dtype=np.float64)
        self._cpRow   = np.zeros(vecLen, dtype=np.float64)

        if extra_vars:
            self._pradRow  = np.zeros(vecLen, dtype=np.float64)
            self._pionRow  = np.zeros(vecLen, dtype=np.float64)
            self._peleRow  = np.zeros(vecLen, dtype=np.float64)
            self._pcoulRow = np.zeros(vecLen, dtype=np.float64)

            self._eradRow  = np.zeros(vecLen, dtype=np.float64)
            self._eionRow  = np.zeros(vecLen, dtype=np.float64)
            self._eeleRow  = np.zeros(vecLen, dtype=np.float64)
            self._ecoulRow = np.zeros(vecLen, dtype=np.float64)

            self._sradRow  = np.zeros(vecLen, dtype=np.float64)
            self._sionRow  = np.zeros(vecLen, dtype=np.float64)
            self._seleRow  = np.zeros(vecLen, dtype=np.float64)
            self._scoulRow = np.zeros(vecLen, dtype=np.float64)

        if mode is EOS_MODE.DENS_TEMP:
            # Call helmholtz
            self.eos_helm(0, vecLen, extra_vars, coulomb_multiplier)

            # Set output
            eosData[presStart:presStart+vecLen] = self._ptotRow
            eosData[eintStart:eintStart+vecLen] = self._etotRow
            eosData[gamcStart:gamcStart+vecLen] = self._gamcRow
            eosData[entrStart:entrStart+vecLen] = self._stotRow
        elif mode is EOS_MODE.DENS_EI:
            # Desired EI
            ewant = eosData[eintStart:eintStart+vecLen]
            # Initialise errors
            tnew = np.zeros(vecLen, dtype=np.float64)
            error = np.zeros(vecLen, dtype=np.float64)

            # Call helmholtz
            self.eos_helm(0, vecLen, extra_vars, coulomb_multiplier)

            # Create initial condition
            tnew = self._tempRow - (self._etotRow - ewant) / self._detRow
            # Don't allow the temperature to change by more than an order of magnitude
            tnew = np.clip(tnew, 0.1 * self._tempRow, 10.0 * self._tempRow)
            # Compute the error
            error = np.abs((tnew - self._tempRow) / self._tempRow)
            # Store the new temperature
            # Make sure to copy, so that this does not just reference the same object
            self._tempRow = tnew[:]
            # Check if we are freezing, if so set the temperature to smallt
            # and adjust the error so we don't wait for this one
            self._tempRow[tnew < _EOS_SMALLT] = _EOS_SMALLT
            error[tnew < _EOS_SMALLT] = 0.1 * _EOS_TOL

            # Loop
            for k in range(vecLen):
                for i in range(2, _EOS_MAXNEWTON+1):
                    if error[k] < _EOS_TOL:
                        break

                    # Call helmholtz
                    self.eos_helm(k, 1, extra_vars, coulomb_multiplier)

                    tnew[k] = self._tempRow[k] - (self._etotRow[k] - ewant[k]) / self._detRow[k]

                    # Don't allow the temperature to change by more than an order of magnitude
                    tnew[k] = np.clip(tnew[k], 0.1 * self._tempRow[k], 10.0 * self._tempRow[k])
                    # Compute the error
                    error[k] = np.abs((tnew[k] - self._tempRow[k]) / self._tempRow[k])

                    # Store the new temperature
                    self._tempRow[k] = tnew[k]

                    # Check if we are freezing, if so set the temperature to smallt
                    # and adjust the error so we don't wait for this one
                    if self._tempRow[k] < _EOS_SMALLT:
                        self._tempRow[k] = _EOS_SMALLT
                        error[k] = 0.1 * _EOS_TOL
                else:  # If the Newton loop fails to converge
                    print("Newton-Raphson failed in subroutine eos_helmholtz")
                    print("(e and rho as input):")
                    print(f"Too many iterations: {_EOS_MAXNEWTON}")
                    print(f"k    = {k}, ({vecLen})")
                    print(f"Temp = {self._tempRow[k]}")
                    print(f"Dens = {self._densRow[k]}")
                    print(f"Pres = {self._ptotRow[k]}")

                    raise RuntimeError("too many iterations in Helmholtz Eos")

            # Crank through the entire eos one last time
            self.eos_helm(0, vecLen, extra_vars, coulomb_multiplier)

            eosData[tempStart:tempStart+vecLen] = self._tempRow
            eosData[presStart:presStart+vecLen] = self._ptotRow
            eosData[gamcStart:gamcStart+vecLen] = self._gamcRow
            eosData[entrStart:entrStart+vecLen] = self._stotRow

            if force_constant_input:
                eosData[eintStart:eintStart+vecLen] = ewant
            else:
                eosData[eintStart:eintStart+vecLen] = self._etotRow
        elif mode is EOS_MODE.DENS_PRES:
            # Desired PRES
            pwant = eosData[presStart:presStart+vecLen]
            # Initialise errors
            tnew = np.zeros(vecLen, dtype=np.float64)
            error = np.zeros(vecLen, dtype=np.float64)

            # Call helmholtz
            self.eos_helm(0, vecLen, extra_vars, coulomb_multiplier)

            # Create initial condition
            tnew = self._tempRow - (self._ptotRow - pwant) / self._dptRow
            # Don't allow the temperature to change by more than an order of magnitude
            tnew = np.clip(tnew, 0.1 * self._tempRow, 10.0 * self._tempRow)
            # Compute the error
            error = np.abs((tnew - self._tempRow) / self._tempRow)
            # Store the new temperature
            # Make sure to copy, so that this does not just reference the same object
            self._tempRow = tnew[:]
            # Check if we are freezing, if so set the temperature to smallt
            # and adjust the error so we don't wait for this one
            self._tempRow[tnew < _EOS_SMALLT] = _EOS_SMALLT
            error[tnew < _EOS_SMALLT] = 0.1 * _EOS_TOL

            # Loop
            for k in range(vecLen):
                for i in range(2, _EOS_MAXNEWTON+1):
                    if error[k] < _EOS_TOL:
                        break

                    # Call helmholtz
                    eos_helm(k, 1, extra_vars, coulomb_multiplier)

                    tnew[k] = self._tempRow[k] - (self._ptotRow[k] - pwant[k]) / self._dptRow[k]

                    # Don't allow the temperature to change by more than an order of magnitude
                    tnew[k] = np.clip(tnew[k], 0.1 * self._tempRow[k], 10.0 * self._tempRow[k])

                    # Compute the error
                    error[k] = np.abs((tnew[k] - self._tempRow[k]) / self._tempRow[k])

                    # Store the new temperature
                    self._tempRow[k] = tnew[k]

                    # Check if we are freezing, if so set the temperature to smallt
                    # and adjust the error so we don't wait for this one
                    if self._tempRow[k] < _EOS_SMALLT:
                        self._tempRow[k] = _EOS_SMALLT
                        error[k] = 0.1 * _EOS_TOL
                else:  # If the Newton loop fails to converge
                    print("Newton-Raphson failed in subroutine eos_helmholtz")
                    print("(e and rho as input):")
                    print(f"Too many iterations: {_EOS_MAXNEWTON}")
                    print(f"k    = {k}, ({vecLen})")
                    print(f"Temp = {self._tempRow[k]}")
                    print(f"Dens = {self._densRow[k]}")
                    print(f"Pres = {self._ptotRow[k]}")

                    raise RuntimeError("too many iterations in Helmholtz Eos")

            # Crank through the entire eos one last time
            self.eos_helm(0, vecLen, extra_vars, coulomb_multiplier)
    
            eosData[tempStart:tempStart+vecLen] = self._tempRow
            eosData[gamcStart:gamcStart+vecLen] = self._gamcRow
            eosData[eintStart:eintStart+vecLen] = self._etotRow
            eosData[entrStart:entrStart+vecLen] = self._stotRow

            if force_constant_input:
                eosData[presStart:presStart+vecLen] = pwant
            else:
                eosData[presStart:presStart+vecLen] = self._ptotRow
        else:
            raise RuntimeError('Unknown EOS mode')

        # Get the derivatives
        dstStart = EOS_VAR.DST * vecLen
        dsdStart = EOS_VAR.DSD * vecLen
        dptStart = EOS_VAR.DPT * vecLen
        dpdStart = EOS_VAR.DPD * vecLen
        detStart = EOS_VAR.DET * vecLen
        dedStart = EOS_VAR.DED * vecLen
        deaStart = EOS_VAR.DEA * vecLen
        dezStart = EOS_VAR.DEZ * vecLen
        pelStart = EOS_VAR.PEL * vecLen
        neStart  = EOS_VAR.NE  * vecLen
        etaStart = EOS_VAR.ETA * vecLen
        cvStart  = EOS_VAR.CV  * vecLen
        cpStart  = EOS_VAR.CP  * vecLen

        eosData[dstStart:dstStart + vecLen] = self._dstRow
        eosData[dsdStart:dsdStart + vecLen] = self._dsdRow
        eosData[dptStart:dptStart + vecLen] = self._dptRow
        eosData[dpdStart:dpdStart + vecLen] = self._dpdRow
        eosData[detStart:detStart + vecLen] = self._detRow
        eosData[dedStart:dedStart + vecLen] = self._dedRow
        eosData[deaStart:deaStart + vecLen] = 0.0 # deaRow
        eosData[dezStart:dezStart + vecLen] = 0.0 # dezRow
        eosData[pelStart:pelStart + vecLen] = self._pelRow
        eosData[neStart :neStart  + vecLen] = self._neRow
        eosData[etaStart:etaStart + vecLen] = self._etaRow
        eosData[cvStart :cvStart  + vecLen] = self._cvRow
        eosData[cpStart :cpStart  + vecLen] = self._cpRow

        if extra_vars:
            extraData = np.zeros(EOS_VAR.NUM_EXTRA*vecLen, dtype=np.float64)

            pradStart  = EOS_VAR.PRAD  * vecLen
            pionStart  = EOS_VAR.PION  * vecLen
            peleStart  = EOS_VAR.PELE  * vecLen
            pcoulStart = EOS_VAR.PCOUL * vecLen

            eradStart  = EOS_VAR.ERAD  * vecLen
            eionStart  = EOS_VAR.EION  * vecLen
            eeleStart  = EOS_VAR.EELE  * vecLen
            ecoulStart = EOS_VAR.ECOUL * vecLen

            sradStart  = EOS_VAR.SRAD  * vecLen
            sionStart  = EOS_VAR.SION  * vecLen
            seleStart  = EOS_VAR.SELE  * vecLen
            scoulStart = EOS_VAR.SCOUL * vecLen

            extraData[pradStart :pradStart  + vecLen] = self._pradRow
            extraData[pionStart :pionStart  + vecLen] = self._pionRow
            extraData[peleStart :peleStart  + vecLen] = self._peleRow
            extraData[pcoulStart:pcoulStart + vecLen] = self._pcoulRow
            
            extraData[eradStart :eradStart  + vecLen] = self._eradRow
            extraData[eionStart :eionStart  + vecLen] = self._eionRow
            extraData[eeleStart :eeleStart  + vecLen] = self._eeleRow
            extraData[ecoulStart:ecoulStart + vecLen] = self._ecoulRow
            
            extraData[sradStart :sradStart  + vecLen] = self._sradRow
            extraData[sionStart :sionStart  + vecLen] = self._sionRow
            extraData[seleStart :seleStart  + vecLen] = self._seleRow
            extraData[scoulStart:scoulStart + vecLen] = self._scoulRow

        if extra_vars:
            return eosData, extraData
        else:
            return eosData

    def eos_helm(self, first: int, vecLen: int, extra_vars: bool = False, coulomb_multiplier: float = 1.0):
        self._check_init()

        from ._herm import (
            psi0, dpsi0, ddpsi0, psi1, dpsi1, ddpsi1, psi2, dpsi2, ddpsi2,
            herm5,
            xpsi0, xpsi1,
            herm3dpd, herm3e, herm3x
        )

        for i in range(first, first+vecLen):
            # Main quantities
            dens  = self._densRow[i]
            temp  = self._tempRow[i]
            abar  = self._abarRow[i]
            zbar  = self._zbarRow[i]
            Ye    = zbar/abar

            # frequent combinations
            kT      = KERG * temp
            kAvoY   = KERGAVO / abar

            # radiation section:
            prad    = ASOLI3 * temp**4
            dpraddt = 4.0 * prad / temp
            dpraddd = 0.0

            x1      =  prad / dens
            erad    =  3.0  * x1
            deraddd = -erad / dens
            deraddt =  4.0  * erad / temp

            srad    = (x1 + erad)/temp
            dsraddd = (dpraddd/dens - x1/dens + deraddd)/temp
            dsraddt = (dpraddt/dens + deraddt - srad)/temp

            # ion section:
            dxnidd  =  AVO / abar
            xni     =  dxnidd * dens
            pion    =  xni * kT
            dpiondd =  AVO * kT / abar
            dpiondt =  xni * KERG
            eion    =  1.5 * pion / dens
            deiondd = (1.5 * dpiondd - eion) / dens
            deiondt =  1.5 * xni * KERG / dens

            # sackur-tetrode equation for the ion entropy of 
            # a single ideal gas characterized by abar
            x2      = abar**2 * np.sqrt(abar) / (dens*AVO)
            y0      = SIONCON * temp
            z0      = x2 * y0 * np.sqrt(y0)
            sion    = (pion/dens + eion)/temp + kAvoY*np.log(z0)
            dsiondd = (dpiondd/dens - pion/(dens**2) + deiondd)/temp - kAvoY / dens
            dsiondt = (dpiondt/dens + deiondt)/temp - (pion/dens + eion) / (temp**2) + 1.5 * kAvoY / temp

            # electron-positron section:
            # enter the table with ye*den, no checks of the input
            dIn = Ye*dens
            # hash locate this temperature and density
            jat = int((np.log10(temp) - _LOGT_LO)/_T_STEP)
            jat = max(0, min(jat, _EOSJMAX-2))
            iat = int((np.log10(dIn) - _LOGD_LO)/_D_STEP)
            iat = max(0, min(iat, _EOSIMAX-2))
            # access the table locations only once
            fi = np.zeros(36, dtype=np.float64)
            fi[0]  = _eos_f[iat,jat]
            fi[1]  = _eos_f[iat+1,jat]
            fi[2]  = _eos_f[iat,jat+1]
            fi[3]  = _eos_f[iat+1,jat+1]
            fi[4]  = _eos_ft[iat,jat]
            fi[5]  = _eos_ft[iat+1,jat]
            fi[6]  = _eos_ft[iat,jat+1]
            fi[7]  = _eos_ft[iat+1,jat+1]
            fi[8]  = _eos_ftt[iat,jat]
            fi[9]  = _eos_ftt[iat+1,jat]
            fi[10] = _eos_ftt[iat,jat+1]
            fi[11] = _eos_ftt[iat+1,jat+1]
            fi[12] = _eos_fd[iat,jat]
            fi[13] = _eos_fd[iat+1,jat]
            fi[14] = _eos_fd[iat,jat+1]
            fi[15] = _eos_fd[iat+1,jat+1]
            fi[16] = _eos_fdd[iat,jat]
            fi[17] = _eos_fdd[iat+1,jat]
            fi[18] = _eos_fdd[iat,jat+1]
            fi[19] = _eos_fdd[iat+1,jat+1]
            fi[20] = _eos_fdt[iat,jat]
            fi[21] = _eos_fdt[iat+1,jat]
            fi[22] = _eos_fdt[iat,jat+1]
            fi[23] = _eos_fdt[iat+1,jat+1]
            fi[24] = _eos_fddt[iat,jat]
            fi[25] = _eos_fddt[iat+1,jat]
            fi[26] = _eos_fddt[iat,jat+1]
            fi[27] = _eos_fddt[iat+1,jat+1]
            fi[28] = _eos_fdtt[iat,jat]
            fi[29] = _eos_fdtt[iat+1,jat]
            fi[30] = _eos_fdtt[iat,jat+1]
            fi[31] = _eos_fdtt[iat+1,jat+1]
            fi[32] = _eos_fddtt[iat,jat]
            fi[33] = _eos_fddtt[iat+1,jat]
            fi[34] = _eos_fddtt[iat,jat+1]
            fi[35] = _eos_fddtt[iat+1,jat+1]

            # various differences
            xt  = max( (temp - _EOS_T[jat])/_EOS_DT[jat], 0.0)
            xd  = max( (dIn  - _EOS_D[iat])/_EOS_DD[iat], 0.0)
            mxt = 1.0 - xt
            mxd = 1.0 - xd

            # the density and temperature basis functions
            si0t  =  psi0(xt)
            si1t  =  psi1(xt)  * _EOS_DT[jat]
            si2t  =  psi2(xt)  * _EOS_DT[jat]**2
            si0mt =  psi0(mxt)
            si1mt = -psi1(mxt) * _EOS_DT[jat]
            si2mt =  psi2(mxt) * _EOS_DT[jat]**2
            si0d  =  psi0(xd)
            si1d  =  psi1(xd)  * _EOS_DD[iat]
            si2d  =  psi2(xd)  * _EOS_DD[iat]**2
            si0md =  psi0(mxd)
            si1md = -psi1(mxd) * _EOS_DD[iat]
            si2md =  psi2(mxd) * _EOS_DD[iat]**2

            # the first derivatives of the basis functions
            dsi0t  =  dpsi0(xt)  / _EOS_DT[jat]
            dsi1t  =  dpsi1(xt)
            dsi2t  =  dpsi2(xt)  * _EOS_DT[jat]
            dsi0mt = -dpsi0(mxt) / _EOS_DT[jat]
            dsi1mt =  dpsi1(mxt)
            dsi2mt = -dpsi2(mxt) * _EOS_DT[jat]

            dsi0d  =  dpsi0(xd)  / _EOS_DD[iat]
            dsi1d  =  dpsi1(xd)
            dsi2d  =  dpsi2(xd)  * _EOS_DD[iat]
            dsi0md = -dpsi0(mxd) / _EOS_DD[iat]
            dsi1md =  dpsi1(mxd)
            dsi2md = -dpsi2(mxd) * _EOS_DD[iat]

            # the second derivatives of the basis functions
            ddsi0t  =  ddpsi0(xt)  / _EOS_DT[jat]**2
            ddsi1t  =  ddpsi1(xt)  / _EOS_DT[jat]
            ddsi2t  =  ddpsi2(xt)
            ddsi0mt =  ddpsi0(mxt) / _EOS_DT[jat]**2
            ddsi1mt = -ddpsi1(mxt) / _EOS_DT[jat]
            ddsi2mt =  ddpsi2(mxt)

            # the free energy
            free  = herm5(
                si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
                si0d,   si1d,   si2d,   si0md,   si1md,   si2md,
                fi
            )
            # derivative with respect to density
            df_d  = herm5(
                si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
                dsi0d,  dsi1d,  dsi2d,  dsi0md,  dsi1md,  dsi2md,
                fi
            )
            # derivative with respect to temperature
            df_t = herm5(
                dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt,
                si0d,   si1d,   si2d,   si0md,   si1md,   si2md,
                fi
            )
            # second derivative with respect to temperature
            df_tt = herm5(
                ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt,
                si0d,   si1d,   si2d,   si0md,   si1md,   si2md,
                fi
            )
            # second derivative with respect to temperature and density
            df_dt = herm5(
                dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt,
                dsi0d,  dsi1d,  dsi2d,  dsi0md,  dsi1md,  dsi2md,
                fi
            )

            # now get the pressure derivative with density, chemical potential, and 
            # electron positron number densities
            # get the interpolation weight functions
            si0t   =   xpsi0(xt)
            si1t   =   xpsi1(xt)  * _EOS_DT[jat]
            si0mt  =   xpsi0(mxt)
            si1mt  =  -xpsi1(mxt) * _EOS_DT[jat]
            si0d   =   xpsi0(xd)
            si1d   =   xpsi1(xd)  * _EOS_DD[iat]
            si0md  =   xpsi0(mxd)
            si1md  =  -xpsi1(mxd) * _EOS_DD[iat]
            # pressure derivative with density
            fi = np.zeros(16, dtype=np.float64)
            fi[0]  = _eos_dpdf[iat, jat]
            fi[1]  = _eos_dpdf[iat+1, jat]
            fi[2]  = _eos_dpdf[iat, jat+1]
            fi[3]  = _eos_dpdf[iat+1, jat+1]
            fi[4]  = _eos_dpdft[iat, jat]
            fi[5]  = _eos_dpdft[iat+1, jat]
            fi[6]  = _eos_dpdft[iat, jat+1]
            fi[7]  = _eos_dpdft[iat+1, jat+1]
            fi[8]  = _eos_dpdfd[iat, jat]
            fi[9]  = _eos_dpdfd[iat+1, jat]
            fi[10] = _eos_dpdfd[iat, jat+1]
            fi[11] = _eos_dpdfd[iat+1, jat+1]
            fi[12] = _eos_dpdfdt[iat, jat]
            fi[13] = _eos_dpdfdt[iat+1, jat]
            fi[14] = _eos_dpdfdt[iat, jat+1]
            fi[15] = _eos_dpdfdt[iat+1, jat+1]

            dpepdd  = herm3dpd(
                si0t,   si1t,   si0mt,   si1mt,
                si0d,   si1d,   si0md,   si1md,
                fi
            )
            dpepdd  = max(Ye * dpepdd, 0.0)

            # electron chemical potential etaele
            fi[0]  = _eos_ef[iat, jat]
            fi[1]  = _eos_ef[iat+1, jat]
            fi[2]  = _eos_ef[iat, jat+1]
            fi[3]  = _eos_ef[iat+1, jat+1]
            fi[4]  = _eos_eft[iat, jat]
            fi[5]  = _eos_eft[iat+1, jat]
            fi[6]  = _eos_eft[iat, jat+1]
            fi[7]  = _eos_eft[iat+1, jat+1]
            fi[8]  = _eos_efd[iat, jat]
            fi[9]  = _eos_efd[iat+1, jat]
            fi[10] = _eos_efd[iat, jat+1]
            fi[11] = _eos_efd[iat+1, jat+1]
            fi[12] = _eos_efdt[iat, jat]
            fi[13] = _eos_efdt[iat+1, jat]
            fi[14] = _eos_efdt[iat, jat+1]
            fi[15] = _eos_efdt[iat+1, jat+1]

            etaele  = herm3e(
                si0t,   si1t,   si0mt,   si1mt,
                si0d,   si1d,   si0md,   si1md,
                fi
            )

            # electron + positron number densities
            fi[0]  = _eos_xf[iat, jat]
            fi[1]  = _eos_xf[iat+1, jat]
            fi[2]  = _eos_xf[iat, jat+1]
            fi[3]  = _eos_xf[iat+1, jat+1]
            fi[4]  = _eos_xft[iat, jat]
            fi[5]  = _eos_xft[iat+1, jat]
            fi[6]  = _eos_xft[iat, jat+1]
            fi[7]  = _eos_xft[iat+1, jat+1]
            fi[8]  = _eos_xfd[iat, jat]
            fi[9]  = _eos_xfd[iat+1, jat]
            fi[10] = _eos_xfd[iat, jat+1]
            fi[11] = _eos_xfd[iat+1, jat+1]
            fi[12] = _eos_xfdt[iat, jat]
            fi[13] = _eos_xfdt[iat+1, jat]
            fi[14] = _eos_xfdt[iat, jat+1]
            fi[15] = _eos_xfdt[iat+1, jat+1]

            xnefer  = herm3x(
                si0t,   si1t,   si0mt,   si1mt,
                si0d,   si1d,   si0md,   si1md,
                fi
            )

            # the desired electron-positron thermodynamic quantities
            x3      = dIn**2
            pele    = x3 * df_d
            dpepdt  = x3 * df_dt
            sele    = -df_t  * Ye
            dsepdt  = -df_tt * Ye
            dsepdd  = -df_dt * Ye**2

            eele    = Ye * free + temp * sele
            deepdt  = temp * dsepdt
            deepdd  = Ye**2 * df_d + temp * dsepdd

            # coulomb section:
            # initialize
            pcoul    = 0.0e0
            dpcouldd = 0.0e0
            dpcouldt = 0.0e0
            ecoul    = 0.0e0
            decouldd = 0.0e0
            decouldt = 0.0e0
            scoul    = 0.0e0
            dscouldd = 0.0e0
            dscouldt = 0.0e0

            # Set the coulomb multiplier to a local value -- we might change it only within this call
            local_coulombMult = coulomb_multiplier

            #  uniform background corrections & only the needed parts for speed
            #  plasg is the plasma coupling parameter
            #  split up calculations below -- they all used to depend upon a redefined z
            z1        = (4.0/3.0) * np.pi
            s1        = z1 * xni
            dsdd      = z1 * dxnidd
            lami      = 1.0/s1**(1.0/3.0)
            z2        = -(1.0/3.0) * lami/s1
            lamidd    = z2 * dsdd
            plasg     = zbar**2 * QE**2/(kT*lami)
            z3        = -plasg / lami
            plasgdd   = z3 * lamidd
            plasgdt   = -plasg / kT * KERG
            #  yakovlev & shalybkov 1989 equations 82, 85, 86, 87
            if (plasg >= 1.0):
                x4       = plasg**(1.0/4.0)
                z4       = c1/x4
                ecoul    = dxnidd * kT * (a1*plasg + b1*x4 + z4 + d1cc)
                pcoul    = (1.0/3.0) * dens * ecoul
                scoul    = -kAvoY*(3.0*b1*x4 - 5.0*z4 + d1cc*(np.log(plasg) - 1.0) - e1cc)
                y1       = dxnidd * kT * (a1 + 0.25e0/plasg*(b1*x4 - z4))
                decouldd = y1 * plasgdd 
                decouldt = y1 * plasgdt + ecoul / temp
                dpcouldd = (1.0/3.0) * (ecoul + dens * decouldd)
                dpcouldt = (1.0/3.0) * dens  * decouldt
                y2       = -kAvoY/plasg*(0.75*b1*x4 + 1.25*z4 + d1cc)
                dscouldd = y2 * plasgdd
                dscouldt = y2 * plasgdt
            # yakovlev & shalybkov 1989 equations 102, 103, 104
            elif (plasg < 1.0):
                x5       = plasg * np.sqrt(plasg)
                y3       = plasg**b2
                z5       = c2 * x5 - (1.0/3.0) * a2 * y3
                pcoul    = -pion * z5
                ecoul    = 3.0e0 * pcoul / dens
                scoul    = -kAvoY*(c2*x5 - a2*(b2 - 1.0)/b2*y3)
                s2       = (1.5*c2*x5 - (1.0/3.0)*a2*b2*y3)/plasg
                dpcouldd = -dpiondd*z5 - pion*s2*plasgdd
                dpcouldt = -dpiondt*z5 - pion*s2*plasgdt
                decouldd = 3.0*dpcouldd/dens - ecoul/dens
                decouldt = 3.0*dpcouldt/dens
                s3       = -kAvoY/plasg * (1.5*c2*x5 - a2*(b2 - 1.0)*y3)
                dscouldd = s3 * plasgdd
                dscouldt = s3 * plasgdt


            s4 = prad + pion + pele
            x6 = s4 + pcoul * local_coulombMult

            # TODO More standard python error handling
            if (x6 < 0.0 or np.isnan(x6)):
                print('Negative total pressure.')
                print(' values: dens,temp: ', dens, temp)
                print(' values: abar,zbar: ', abar, zbar)
                print(' coulomb coupling parameter Gamma: ', plasg)

                if (abar < 0.0 or np.isnan(abar)):
                    print('However, abar is negative, abar=', abar)
                    print('It is possible that the mesh is of low quality.')
                    raise RuntimeError('abar is negative.')

                if (s4 > 0.0):
                    print('nonpositive P caused by coulomb correction: Pnocoul,Pwithcoul: ', s4, x6)

                    if (coulomb_multiplier > 0.0):
                       print('set coulombCorrection to 0.0 if plasma Coulomb corrections are not important')

                    #if (eos_coulombAbort):
                    #    raise RuntimeError('coulomb correction causing negative total pressure.')
                    #else:
                    print('Setting coulombMult to zero for this call')
                    local_coulombMult = 0.0
                else:
                    print(' Prad  ', prad)
                    print(' Pion  ', pion)
                    print(' Pele  ', pele)
                    print(' Pcoul ', pcoul*coulomb_multiplier)
                    print(' Ptot  ', x6)
                    print(' df_d  ', df_d)

                    raise RuntimeError('negative total pressure.')

            pcoul    = pcoul    * local_coulombMult
            dpcouldd = dpcouldd * local_coulombMult
            dpcouldt = dpcouldt * local_coulombMult
            ecoul    = ecoul    * local_coulombMult
            decouldd = decouldd * local_coulombMult
            decouldt = decouldt * local_coulombMult
            scoul    = scoul    * local_coulombMult
            dscouldd = dscouldd * local_coulombMult 
            dscouldt = dscouldt * local_coulombMult

            # sum all the components
            pres    = prad    + pion    + pele   + pcoul
            ener    = erad    + eion    + eele   + ecoul
            entr    = srad    + sion    + sele   + scoul
            dpresdd = dpraddd + dpiondd + dpepdd + dpcouldd
            dpresdt = dpraddt + dpiondt + dpepdt + dpcouldt
            denerdd = deraddd + deiondd + deepdd + decouldd
            denerdt = deraddt + deiondt + deepdt + decouldt
            dentrdd = dsraddd + dsiondd + dsepdd + dscouldd
            dentrdt = dsraddt + dsiondt + dsepdt + dscouldt

            # form gamma_1
            chit  = temp/pres * dpresdt
            chid  = dpresdd * dens/pres
            x7    = pres / dens * chit/(temp * denerdt)
            gamc  = chit*x7 + chid
            cv    = denerdt
            cp    = cv*gamc/chid

            # store the output
            self._ptotRow[i] = pres         # used by Eos as EOS_PRES = PRES_VAR
            self._etotRow[i] = ener         # used by Eos as EOS_EINT = EINT_VAR
            self._stotRow[i] = entr/KERGAVO # this is entropy, used by Eos as EOS_ENTR (masked)
            self._dpdRow[i]  = dpresdd  # used as EOS_DPD
            self._dptRow[i]  = dpresdt  # used as EOS_DPT  ALWAYS used by MODE_DENS_PRES in Eos.F90
            self._dedRow[i]  = denerdd  # used as EOS_DED
            self._detRow[i]  = denerdt  # used as EOS_DET  ALWAYS used by MODE_DENS_EI in Eos.F90
            self._dsdRow[i]  = dentrdd  # used as EOS_DSD
            self._dstRow[i]  = dentrdt  # used as EOS_DST
            self._pelRow[i]  = pele     # used as EOS_PEL
            self._neRow[i]   = xnefer   # used as EOS_NE
            self._etaRow[i]  = etaele   # used as EOS_ETA 
            self._gamcRow[i] = gamc     # used as EOS_GAMC = GAMC_VAR
            self._cvRow[i]   = cv       # EOS_CV
            self._cpRow[i]   = cp       # EOS_CP

            if extra_vars:
                self._pradRow[i]  = prad
                self._pionRow[i]  = pion
                self._peleRow[i]  = pele
                self._pcoulRow[i] = pcoul

                self._eradRow[i]  = erad
                self._eionRow[i]  = eion
                self._eeleRow[i]  = eele
                self._ecoulRow[i] = ecoul

                self._sradRow[i]  = srad
                self._sionRow[i]  = sion
                self._seleRow[i]  = sele
                self._scoulRow[i] = scoul
