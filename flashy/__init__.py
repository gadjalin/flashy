"""
Flashy!
---

It's all very extravagant for something you call 'state-of-the-art'...

---

A python package for doing core-collapse supernova
physics with the FLASH code.

"""
from . import eos
from . import io
from . import nu
from . import nuc
from . import plot
from . import post
from . import constants

from .plot import get_label, make_label, get_plot_scale, register_label, get_unit_factor, convert_units, \
                  get_isotope_label, get_plot_absolute
from .post import get_bounce_time, get_midcell_dr, calculate_shell_mass, calculate_shock, \
                  calculate_compactness, diagnostic_mask, \
                  diagnostic_explosion_energy_1d, diagnostic_explosion_energy_2d, diagnostic_mass_1d, diagnostic_mass_2d

