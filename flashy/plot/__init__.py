from typing import Optional, Union

from .label_descriptor import LabelDescriptor
from .label_registry import LabelRegistry
from .label_defaults import _DEFAULT_DESCRIPTORS

_REGISTRY = LabelRegistry()
for desc in _DEFAULT_DESCRIPTORS:
    _REGISTRY.register(desc)

def get_label(
    key: str,
    units: Optional[str] = None,
    scale: Optional[Union[int, float, str]] = None,
    scale_style: str = 'auto',
    unit_style: str = 'brackets',
    name_style: str = 'symbol'
) -> str:
    """
    Get an appropriate plotting label for a quantity.
    
    Parameters
    ----------
    key : str
        A key to a quantity. This includes common quantity names from
        plot files, dat files, and progenitor files.
    units : str, optional
        Unit system to use. Common systems include 'cgs', 'si', and 'sun',
        where appropriate, but specific ones exist for certain quantities.
    scale : int, float, or str, optional
        A scaling factor to show in the label.
        This can be a user defined string which will be
        printed as-is.
    scale_style : {'auto', 'linear', 'log'}
        Formatting style for the scaling factor.
    unit_style : {'brackets', 'parens', 'open', 'none'}
        Determines which character to surround the units with.
        Units are either shown between square brackets (default),
        parentheses, or nothing. Using 'none' completely deactivates
        units and scaling factor from the generated label.
    name_style : {'symbol', 'short', 'full'}
        Which name style to use.

    Returns
    -------
    label : str
        A fully formatted, LaTeX, label to be used
        as matplotlib axis label.
    """
    return _REGISTRY.get_label(key, units, scale, scale_style, unit_style, name_style)


def get_plot_scale(key: str) -> str:
    """
    Get the preferred plot scale for a quantity.

    Parameters
    ----------
    key : str
        A key to a quantity. This includes common quantity names from
        plot files, dat files, and progenitor files.

    Returns
    -------
    scale : str
        Either 'log' or 'linear', to be used directly with matplotlib
        set_Xscale functions.
    """
    return _REGISTRY.get_plot_scale(key)

def register_label(descriptor: LabelDescriptor) -> None:
    _REGISTRY.register(desc)

