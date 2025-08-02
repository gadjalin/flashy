"""
Flashy!
---

It's all very extravagant for something you call 'state-of-the-art'...

---

A python package for performing various FLASH related tasks.

"""
from . import eos
from . import io
from . import nuc
from . import plot
from . import analysis

from .plot import get_label, get_plot_scale, register_label
from .analysis import get_bounce_time, get_midcell_dr, calculate_shell_mass, calculate_shock, \
                      calculate_compactness

