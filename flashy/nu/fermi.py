import numpy as np
from ..constants import KELVIN_TO_MEV, AVO


LOG10EXP = 0.43429448190325182765 # log10(e)


# Borrowed from NuLib
def fermidirac_logtermonly_log10ed(e: float, mu: float, T: float) -> float:
    return (e - mu)/(T*KELVIN_TO_MEV)*LOG10EXP

