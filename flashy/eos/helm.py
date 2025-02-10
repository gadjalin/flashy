from enum import Enum
import numpy as np
from pathlib import Path
import os
try:
    from importlib import resources
except ModuleNotFoundError:
    import importlib_resources as resources
# This simply follows closely eos_helm.F90

# Private module fields

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
_eos_f     = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_fd    = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_ft    = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_fdd   = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_ftt   = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_fdt   = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_fddt  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_fdtt  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_fddtt = np.zeros((_EOSIMAX, _EOSJMAX))

# Pressure derivatives tables
_eos_dpdf   = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_dpdfd  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_dpdft  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_dpdfdt = np.zeros((_EOSIMAX, _EOSJMAX))

# Checmical potential tables
_eos_ef   = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_efd  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_eft  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_efdt = np.zeros((_EOSIMAX, _EOSJMAX))

# Number densitites tables
_eos_xf   = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_xfd  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_xft  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_xfdt = np.zeros((_EOSIMAX, _EOSJMAX))

_EOS_T = 10.0**(_LOGT_LO + np.arange(_EOSJMAX) * _T_STEP)
_EOS_D = 10.0**(_LOGD_LO + np.arange(_EOSIMAX) * _D_STEP)

_EOS_DT = np.diff(_EOS_T)
_EOS_DD = np.diff(_EOS_D)

_initialised = False

# EOS parameters
coulombCorrection = True
coulombMultiplier = 1.0

# eosData indices
class EOS(Enum):
    PRES = 0
    DENS = 1
    EINT = 2
    TEMP = 3
    GAMC = 4
    ABAR = 5
    ZBAR = 6
    ENTR = 7
    EKIN = 8
    
    DPT = 9
    DPD = 10
    DET = 11
    DED = 12
    DEA = 13
    DEZ = 14
    DST = 15
    DSD = 16
    CV  = 17
    CP  = 18
    PEL = 19
    NE  = 20
    ETA = 21
    
    NUM_VARS = 22

    MODE_DENS_TEMP = 100
    MODE_DENS_EI   = 101
    MODE_DENS_PRES = 102


# Check helmholtz has been initialised for convenience
def _check_init() -> None:
    if not _initialised:
        raise RuntimeError("Helmholtz EOS not initialised!")


# Init helmholtz EOS. Read helmholtz table.
def init(helm_table_file = None) -> None:
    global _initialised
    global _eos_f, _eos_fd, _eos_ft, _eos_fdd, _eos_ftt, _eos_fdt, _eos_fddt, _eos_fdtt, _eos_fddtt
    global _eos_dpdf, _eos_dpdfd, _eos_dpdft, _eos_dpdfdt
    global _eos_ef, _eos_efd, _eos_eft, _eos_efdt
    global _eos_xf, _eos_xfd, _eos_xft, _eos_xfdt

    _initialised = False

    # Utilise user-provided file path if one is given. Otherwise, use package default.
    if helm_table_file is None:
        helm_table_path = resources.files("flashy.eos").joinpath('data/helm_table.dat')
    else:
        helm_table_path = Path(helm_table_file)
    
    # Read the helmholtz table
    # Fastest way to read the table according to ChatGPT
    with helm_table_path.open('r') as helm_table:
        # Read the free energy table
        blck = [np.empty((_EOSIMAX,_EOSJMAX), dtype=np.float64) for _ in range(9)]
        for j in range(_EOSJMAX):
            for i in range(_EOSIMAX):
                row = np.fromstring(helm_table.readline(), dtype=np.float64, sep=' ')
                for k in range(9):
                    blck[k][i,j] = row[k]
        _eos_f, _eos_fd, _eos_ft, _eos_fdd, _eos_ftt, _eos_fdt, _eos_fddt, _eos_fdtt, _eos_fddtt = blck
    
        # Read the pressure derivative table
        blck = [np.empty((_EOSIMAX,_EOSJMAX), dtype=np.float64) for _ in range(4)]
        for j in range(_EOSJMAX):
            for i in range(_EOSIMAX):
                row = np.fromstring(helm_table.readline(), dtype=np.float64, sep=' ')
                for k in range(4):
                    blck[k][i,j] = row[k]
        _eos_dpdf, _eos_dpdfd, _eos_dpdft, _eos_dpdfdt = blck
    
        # Read the chemical potential table
        blck = [np.empty((_EOSIMAX,_EOSJMAX), dtype=np.float64) for _ in range(4)]
        for j in range(_EOSJMAX):
            for i in range(_EOSIMAX):
                row = np.fromstring(helm_table.readline(), dtype=np.float64, sep=' ')
                for k in range(4):
                    blck[k][i,j] = row[k]
        _eos_ef, _eos_efd, _eos_eft, _eos_efdt = blck
    
        # Read the number density table
        blck = [np.empty((_EOSIMAX,_EOSJMAX), dtype=np.float64) for _ in range(4)]
        for j in range(_EOSJMAX):
            for i in range(_EOSIMAX):
                row = np.fromstring(helm_table.readline(), dtype=np.float64, sep=' ')
                for k in range(4):
                    blck[k][i,j] = row[k]
        _eos_xf, _eos_xfd, _eos_xft, _eos_xfdt = blck
    
    _initialised = True


def eos(eosData: list | np.ndarray):
    """
    Call the Helmholtz EOS

    
    """
    assert (eosData is not None) and (len(eosData) % EOS.NUM_VARS == 0), "eosData must be an array of size N*EOS.NUM_VARS"


def helm(densRow, tempRow, abarRow, zbarRow):
    _check_init()
    assert (len(densRow) == len(tempRow) and
            len(densRow) == len(abarRow) and
            len(densRow) == len(zbarRow))

    from ._herm import (
        psi0, dpsi0, ddpsi0, psi1, dpsi1, ddpsi1, psi2, dpsi2, ddpsi2,
        herm5, 
        xpsi0, xpsi1,
        herm3dpd, herm3e, herm3x
    )

    # Move constants to generic place
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

    # For the uniform backgound coulomb correction
    a1, b1, c1, d1cc, e1cc, a2, b2, c2 = np.array([
        -0.898004, 0.96786, 0.220703, -0.86097,
        2.5269, 0.29561, 1.9885, 0.288675
    ])

    vecLen = len(densRow)

    ptotRow = np.zeros(vecLen)
    etotRow = np.zeros(vecLen)
    stotRow = np.zeros(vecLen)
    dpdRow  = np.zeros(vecLen)
    dptRow  = np.zeros(vecLen)
    dedRow  = np.zeros(vecLen)
    detRow  = np.zeros(vecLen)
    dsdRow  = np.zeros(vecLen)
    dstRow  = np.zeros(vecLen)
    pelRow  = np.zeros(vecLen)
    neRow   = np.zeros(vecLen)
    etaRow  = np.zeros(vecLen)
    gamcRow = np.zeros(vecLen)
    cvRow   = np.zeros(vecLen)
    cpRow   = np.zeros(vecLen)
    
    for i in range(vecLen):
        # Main quantities
        dens  = densRow[i]
        temp  = tempRow[i]
        abar  = abarRow[i]
        zbar  = zbarRow[i]
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
        jat = max(0, min(jat, _EOSJMAX-1))
        iat = int((np.log10(dIn) - _LOGD_LO)/_D_STEP)
        iat = max(0, min(iat, _EOSIMAX-1))
        # access the table locations only once
        fi = np.zeros(36)
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
        fi = np.zeros(16)
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
        if coulombCorrection:
            local_coulombMult = coulombMultiplier
        else:
            local_coulombMult = 0.0

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

                if (coulombCorrection):
                   print('set coulombCorrection to False if plasma Coulomb corrections are not important')
                  
                #if (eos_coulombAbort):
                #    raise RuntimeError('coulomb correction causing negative total pressure.')
                #else:
                print('Setting coulombMult to zero for this call')
                local_coulombMult = 0.0
            else:
                print(' Prad  ', prad)
                print(' Pion  ', pion)
                print(' Pele  ', pele)
                print(' Pcoul ', pcoul*eos_coulombMult)
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
        ptotRow[i] = pres         # used by Eos as EOS_PRES = PRES_VAR
        etotRow[i] = ener         # used by Eos as EOS_EINT = EINT_VAR
        stotRow[i] = entr/KERGAVO # this is entropy, used by Eos as EOS_ENTR (masked)
        dpdRow[i]  = dpresdd  # used as EOS_DPD
        dptRow[i]  = dpresdt  # used as EOS_DPT  ALWAYS used by MODE_DENS_PRES in Eos.F90
        dedRow[i]  = denerdd  # used as EOS_DED
        detRow[i]  = denerdt  # used as EOS_DET  ALWAYS used by MODE_DENS_EI in Eos.F90
        dsdRow[i]  = dentrdd  # used as EOS_DSD
        dstRow[i]  = dentrdt  # used as EOS_DST
        pelRow[i]  = pele     # used as EOS_PEL
        neRow[i]   = xnefer   # used as EOS_NE
        etaRow[i]  = etaele   # used as EOS_ETA 
        gamcRow[i] = gamc     # used as EOS_GAMC = GAMC_VAR
        cvRow[i]   = cv       # EOS_CV
        cpRow[i]   = cp       # EOS_CP


    return ptotRow, etotRow, stotRow, dpdRow, dptRow, dedRow, detRow, dsdRow, dstRow, pelRow, neRow, etaRow, gamcRow, cvRow, cpRow
