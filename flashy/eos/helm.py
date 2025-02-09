import numpy as np
import importlib.resources
from pathlib import Path
import os
# This simply follows closely eos_helm.F90

# Private module fields

# Table dimensions
_EOSIMAX = 261 # This is the number of steps in density (I axis/rows)
_EOSJMAX = 101 # This is the number of steps in temperature (J axis/columns)
_logT_lo = 3.0 # Minimum temperature in the table in log
_logT_hi = 13.0 # Maximum temperature in the table in log
_logd_lo = -12.0 # Minimum density in the table in log
_logd_hi = 14.0 # Maximum density in the table in log
_T_step = (logT_hi - logT_lo)/float(EOSJMAX-1)
_d_step = (logd_hi - logd_lo)/float(EOSIMAX-1)

# Free energy
_eos_f     = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_fd    = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_ft    = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_fdd   = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_ftt   = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_fdt   = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_fddt  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_fdtt  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_fddtt = np.zeros((_EOSIMAX, _EOSJMAX))

# Pressure derivatives
_eos_dpdf   = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_dpdfd  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_dpdft  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_dpdfdt = np.zeros((_EOSIMAX, _EOSJMAX))

# Checmical potential
_eos_ef   = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_efd  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_eft  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_efdt = np.zeros((_EOSIMAX, _EOSJMAX))

# Number densitites
_eos_xf   = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_xfd  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_xft  = np.zeros((_EOSIMAX, _EOSJMAX))
_eos_xfdt = np.zeros((_EOSIMAX, _EOSJMAX))

_initialised = False


# Public module fields
npoints_dens = _EOSIMAX
npoints_temp = _EOSJMAX
min_logdens = _logd_lo
max_logdens = _logd_hi
min_logtemp = _logT_lo
max_logtemp = _logT_hi


# Check helmholtz has been initialised for convenience
def _check_init() -> None:
    if not _initialised:
        raise RuntimeError("Helmholtz EOS not initialised!")


# Init helmholtz EOS. Read helmholtz table.
def init(helm_table_file = None) -> None:
    _initialised = False

    # Utilise user-provided file path if one is given. Otherwise, use package default.
    if helm_table_file is None:
        helm_table_path = importlib.resources.files(__name__).joinpath('data/helm_table.dat')
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
    pass


init()
