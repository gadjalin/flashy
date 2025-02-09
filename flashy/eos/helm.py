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
_T_STEP = (_LOGT_HI - _LOGT_LO)/float(_EOSJMAX-1)
_D_STEP = (_LOGD_HI - _LOGD_LO)/float(_EOSIMAX-1)

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

_initialised = False


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


def call():
    _check_init()
    pass
