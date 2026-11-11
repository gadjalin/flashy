from typing import Optional, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .labels import convert_units, get_label, get_plot_scale, get_plot_absolute
from ..io import Progenitor, DatFile, DatRun


def prepare_data(
    key: str,
    data,
    units: Optional[str] = None,
    scale_factor: float = 1.0,
    absolute: Optional[bool] = None
) -> np.ndarray:
    """
    Prepare data for plotting.

    Parameters
    ----------
    key : str
        Key to a quantity
    data : float or array-like
        Data to be prepared
    units : str, optional
        Unit system to convert to, or None for the default units.
    scale_factor : float
        Scale the data by this factor
    absolute : bool, optional
        Whether to take the absolute value of the data.

    Returns
    -------
    data : np.ndarray
        Prepared data
    """
    if absolute is None:
        absolute = get_plot_absolute(key, default=False)

    data = convert_units(key, data, units=units, default_factor=1.0)
    if absolute:
        data = np.abs(data)
    return np.asarray(data)/scale_factor


def decorate(
    ax,
    xkey: Optional[str] = None,
    ykey: Optional[str] = None,
    xunits: Optional[str] = None,
    yunits: Optional[str] = None,
    xscale_factor: float = 1.0,
    yscale_factor: float = 1.0,
    xabsolute: Optional[bool] = None,
    yabsolute: Optional[bool] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    name_style: str = 'symbol',
):
    """
    Helper function to decorate the axis (set labels, scales, limits).

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axis to decorate. If None, use plt.gca().
    xkey, ykey : str, optional
        Key to quantities on each axis, or None to leave untouched.
    xunits, yunits : str, optional
        Unit system of each axis, or None for the default units.
    xscale_factor, yscale_factor : float
        Scaling factor shown in each label.
    xabsolute, yabsolute : bool, optional
        Whether the axis shows an absolute value, or None for
        default preference for the quantity.
    xscale, yscale : str, optional
        Axis scale, or None to use the default preference
        for the quantity.
    xlim, ylim : tuple of float, optional
        Axis limits.
    name_style : {'symbol', 'short', 'full'}
        Name style of the labels.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The decorated axis.
    """
    if ax is None:
        ax = plt.gca()

    for key,units,scale_factor,absolute,scale,lim,set_label,set_scale,set_lim in (
        (xkey, xunits, xscale_factor, xabsolute, xscale, xlim, ax.set_xlabel, ax.set_xscale, ax.set_xlim),
        (ykey, yunits, yscale_factor, yabsolute, yscale, ylim, ax.set_ylabel, ax.set_yscale, ax.set_ylim),
    ):
        if key is not None:
            if absolute is None:
                absolute = get_plot_absolute(key, default=False)
            set_label(
                get_label(key, units=units, scale=scale_factor, name_style=name_style,
                          default=key, absolute=absolute)
            )
            if scale is None:
                scale = get_plot_scale(key, default='linear')

        if scale is not None:
            set_scale(scale)
        if lim is not None:
            set_lim(*lim)

    return ax


def plot_dat(
    dat: Union[DatFile, DatRun],
    var: str,
    ax = None,
    tbounce: float = 0.0,
    scale_factor: float = 1.0,
    name_style: str = 'symbol',
    xunits: Optional[str] = None,
    yunits: Optional[str] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    absolute: Optional[bool] = None,
    **kwargs
):
    if isinstance(dat, DatFile):
        dat = dat[-1]

    if ax is None:
        ax = plt.gca()

    xkey = 'time' if tbounce == 0.0 else 'time post-bounce'
    x = prepare_data(xkey, dat['t'] - tbounce, units=xunits, absolute=False)
    y = prepare_data(var, dat[var], units=yunits, scale_factor=scale_factor, absolute=absolute)
    res = ax.plot(x, y, **kwargs)

    decorate(ax, xkey, var, xunits=xunits, yunits=yunits, yscale_factor=scale_factor,
             xabsolute=False, yabsolute=absolute, xscale=xscale, yscale=yscale,
             xlim=xlim, ylim=ylim, name_style=name_style)

    return res


def plot_profile(
    r: np.ndarray,
    f: np.ndarray,
    var: str,
    ax = None,
    scale_factor: float = 1.0,
    name_style: str = 'symbol',
    xunits: Optional[str] = None,
    yunits: Optional[str] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    absolute: Optional[bool] = None,
    **kwargs
):
    if ax is None:
        ax = plt.gca()

    x = prepare_data('radius', r, units=xunits, absolute=False)
    y = prepare_data(var, f, units=yunits, scale_factor=scale_factor, absolute=absolute)
    res = ax.plot(x, y, **kwargs)

    decorate(ax, 'radius', var, xunits=xunits, yunits=yunits, yscale_factor=scale_factor,
             xabsolute=False, yabsolute=absolute, xscale=xscale, yscale=yscale,
             xlim=xlim, ylim=ylim, name_style=name_style)

    return res


def plot_progenitor(
    model: Progenitor,
    var: str,
    ax = None,
    scale_factor: float = 1.0,
    name_style: str = 'symbol',
    xunits: Optional[str] = None,
    yunits: Optional[str] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    absolute: Optional[bool] = None,
    **kwargs
):
    return plot_profile(
        model['r'], model[var], var, ax=ax,
        scale_factor=scale_factor, name_style=name_style,
        xunits=xunits, xscale=xscale, xlim=xlim,
        yunits=yunits, yscale=yscale, ylim=ylim,
        absolute=absolute, **kwargs
    )

