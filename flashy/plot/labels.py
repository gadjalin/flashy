from typing import Optional, Union

from .label_descriptor import LabelDescriptor, _DEFAULT_DESCRIPTORS
from .label_registry import LabelRegistry


_REGISTRY = LabelRegistry()
for descriptor in _DEFAULT_DESCRIPTORS:
    _REGISTRY.register(descriptor)


def get_label(
    key: str,
    units: Optional[str] = None,
    scale: Optional[Union[int, float, str]] = None,
    scale_style: str = 'auto',
    unit_style: str = 'brackets',
    name_style: str = 'symbol',
    default: Optional[str] = None,
    absolute: bool = False
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
    default : str, optional
        Name to use if key is not in the registry.
        It is formatted like make_label does, so the scaling
        factor is still shown (there are no units to show).
        If None (default), an unknown key raises a KeyError.
    absolute : bool
        If True, the name is shown as an absolute value, e.g. $|\\phi|$.
        This only changes the label, see `get_plot_absolute`.

    Returns
    -------
    label : str
        A fully formatted, LaTeX, label to be used
        as matplotlib axis label.

    Raises
    ------
    KeyError
        If key is not in the registry and no default is given.
    ValueError
        If the unit system or the name, unit or scale style is invalid.
        These are not covered by default.
    """
    return _REGISTRY.get_label(key, units, scale, scale_style, unit_style, name_style, default, absolute)


def make_label(
    name: str,
    unit_str: Optional[str] = None,
    scale: Optional[Union[int, float, str]] = None,
    scale_style: str = 'auto',
    unit_style: str = 'brackets',
) -> str:
    return _REGISTRY.make_label(name, unit_str, scale, scale_style, unit_style)


def get_plot_scale(key: str, default: Optional[str] = None) -> str:
    """
    Get the preferred plot scale for a quantity.

    Parameters
    ----------
    key : str
        A key to a quantity. This includes common quantity names from
        plot files, dat files, and progenitor files.
    default : str, optional
        Scale returned as-is if key is not in the registry,
        e.g. 'linear'. If None (default), an unknown key raises a KeyError.

    Returns
    -------
    scale : str
        Either 'log' or 'linear' (or default), to be used directly
        with matplotlib set_Xscale functions.
    """
    return _REGISTRY.get_plot_scale(key, default)


def get_plot_absolute(key: str, default: Optional[bool] = None) -> bool:
    """
    Whether a quantity is preferably plotted as its absolute value.

    This is the case for quantities of constant negative sign that are
    plotted in log scale, such as the gravitational potential.

    Parameters
    ----------
    key : str
        A key to a quantity.
    default : bool, optional
        Returned as-is if `key` is not in the registry, e.g. False.
        If None (default), an unknown key raises a KeyError.

    Returns
    -------
    absolute : bool
        True if the absolute value should be plotted (or `default`).
    """
    return _REGISTRY.get_plot_absolute(key, default)

def get_unit_factor(
    key: str,
    units: Optional[str] = None,
    from_units: Optional[str] = None,
    default: Optional[float] = None
) -> float:
    """
    Get the factor conversion factor to the given unit system.

    Parameters
    ----------
    key : str
        A key to a quantity.
    units : str, optional
        Unit system to convert to, or None (default)
        for default units.
    from_units : str, optional
        Unit system the data is in, or None (default)
        for default code units (from FLASH output).
    default : float, optional
        If key is not in the registry, returns this default value.
        If None (default), an unknown key raises a KeyError.

    Returns
    -------
    factor : float
        Corresponding conversion factor.

    Raises
    ------
    KeyError
        If key is not in the registry and no default is given.
    ValueError
        If a unit system does not exist for this quantity,
        or has no conversion factor.
    """
    return _REGISTRY.get_unit_factor(key, units, from_units, default)


def convert_units(
    key: str,
    value,
    units: Optional[str] = None,
    from_units: Optional[str] = None,
    default_factor: Optional[float] = None
):
    """
    Convert data to the specified unit system.

    Parameters
    ----------
    key : str
        A key to a quantity.
    value : float or array-like
        The data to convert.
    units : str, optional
        Unit system to convert to, or None (default)
        for the default units.
    from_units : str, optional
        Unit system the data is in, or None (default) to
        use the default code (FLASH) units.
    default_factor : float, optional
        Factor applied if key is not in the registry.

    Returns
    -------
    value : float or np.ndarray
        The converted data.
    """
    return _REGISTRY.convert(key, value, units, from_units, default_factor)


def get_isotope_label(
    species: str,
    quantity: str = 'mass fraction',
    scale: Optional[Union[int, float, str]] = None,
    scale_style: str = 'auto',
    unit_style: str = 'brackets',
    name_style: str = 'symbol'
) -> str:
    """
    Get a plotting label for the mass fraction (or abundance) of an isotope.

    Isotope keys (e.g. 'ni56', 'He4', 'neut') are also resolved directly by
    get_label, get_plot_scale and convert_units when they are not otherwise
    registered, so this function is only needed to choose another quantity.

    Parameters
    ----------
    species : str or Nucleus
        Isotope (e.g. 'ni56', 'Ni56', 'he4', 'neut', 'p').
    quantity : str
        'mass fraction' (default) or 'abundance'.
    scale, scale_style, unit_style, name_style
        Same as in get_label.

    Returns
    -------
    label : str
        Generated label.

    Raises
    ------
    ValueError
        If species is not a valid isotope, or quantity has units.
    KeyError
        If quantity is not registered.
    """
    descriptor = _REGISTRY.get_isotope_descriptor(species, quantity)
    return _REGISTRY.format_label(descriptor, None, scale, scale_style, unit_style, name_style)


def register_label(descriptor: LabelDescriptor) -> None:
    _REGISTRY.register(descriptor)

