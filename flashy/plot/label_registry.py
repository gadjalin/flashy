from typing import Dict, Optional, Union
import math
import re

import numpy as np

from .label_descriptor import LabelDescriptor
from ..nuc.nucleus import Nucleus, find_isotope


_UNIT_SURROUND = {
    'brackets': ('[', ']'),
    'parens': ('(', ')'),
    'open': ('', '')
}

# Only resolve isotope keys e.g. h1, ni56, not single letters e.g. p, n
_ISOTOPE_KEY = re.compile(r'^[A-Za-z]{1,2}\d{1,3}$')
_ISOTOPE_NAMES = {'neut', 'neutron', 'neutrons', 'prot', 'proton', 'protons'}


def _parse_isotope(key: str) -> Optional[Nucleus]:
    if key.lower() not in _ISOTOPE_NAMES and not _ISOTOPE_KEY.match(key):
        return None
    try:
        return find_isotope(key)
    except ValueError:
        return None


def _absolute_name(name: str) -> str:
    # Wrap a name in absolute value bars: $\phi$ -> $|\phi|$, Name -> $|$Name$|$
    if len(name) > 1 and name.startswith('$') and name.endswith('$') and name.count('$') == 2:
        return f'$|{name[1:-1]}|$'
    return f'$|${name}$|$'


def _isotope_tex(nucleus: Nucleus) -> str:
    if nucleus.Z == 0:
        return r'\mathrm{n}'
    return fr'^{{{nucleus.A}}}\mathrm{{{nucleus.symbol}}}'


class LabelRegistry(object):
    _descriptors: Dict[str, LabelDescriptor]
    _alias_map: Dict[str, str]

    def __init__(self):
        self._descriptors = {}
        self._alias_map = {}

    def __contains__(self, key: str) -> bool:
        return key in self._alias_map or _parse_isotope(key) is not None

    def __str__(self) -> str:
        return f'LabelRegistry; {len(self._descriptors)} entries'

    def __len__(self) -> int:
        return len(self._descriptors)

    def __getitem__(self, key: str) -> str:
        return self.get_label(key)

    def __iter__(self):
        return iter(list(self._descriptors.values()))

    def __reversed__(self):
        return reversed(list(self._descriptors.values()))

    @property
    def field_list(self) -> list[str]:
        return list(self._alias_map.keys())

    def register(self, descriptor: LabelDescriptor) -> None:
        key = descriptor.key

        # Check no key or alias clash
        for alias in [key, *descriptor.aliases]:
            if alias in self._alias_map:
                existing = self._alias_map[alias]
                raise ValueError(f'Alias already registered for key {existing}: {alias})')

        # Register new descriptor
        self._descriptors[key] = descriptor

        # Register new aliases
        for alias in [key, *descriptor.aliases]:
            self._alias_map[alias] = key

    def get_label(
        self,
        key: str,
        units: Optional[str] = None,
        scale: Optional[Union[int, float, str]] = None,
        scale_style: str = 'auto',
        unit_style: str = 'brackets',
        name_style: str = 'symbol',
        default: Optional[str] = None,
        absolute: bool = False
    ) -> str:
        # Get descriptor, or format the default as the name if the key is unknown
        if key not in self:
            if default is None:
                raise KeyError(f'Unresolved key: {key}')
            name = _absolute_name(default) if absolute else default
            return self.make_label(name, None, scale, scale_style, unit_style)

        return self.format_label(self.get_descriptor(key), units, scale, scale_style, unit_style, name_style, absolute)

    def format_label(
        self,
        descriptor: LabelDescriptor,
        units: Optional[str] = None,
        scale: Optional[Union[int, float, str]] = None,
        scale_style: str = 'auto',
        unit_style: str = 'brackets',
        name_style: str = 'symbol',
        absolute: bool = False
    ) -> str:
        # Resolve name style
        name = getattr(descriptor, name_style, None)
        if name is None:
            raise ValueError(f'Invalid name style: {name_style}')
        if absolute:
            name = _absolute_name(name)

        # Check unit style
        if unit_style != 'none' and unit_style not in _UNIT_SURROUND:
            raise ValueError(f'Invalid unit style: {unit_style}')

        if unit_style != 'none' and unit_style is not None:
            # Resolve unit system
            unit_str = self._resolve_unit_format(descriptor, units)

            # Format scale factor
            scale_str = self._resolve_scale_format(scale, scale_style)

            # Format units with scale factor
            unit_full = ''
            if scale_str:
                unit_full = fr'{scale_str}\ {unit_str}' if unit_str else scale_str
            else:
                unit_full = unit_str
        else:
            unit_full = ''

        if unit_full:
            unit_open = _UNIT_SURROUND[unit_style][0]
            unit_close = _UNIT_SURROUND[unit_style][1]
            label = fr'{name} {unit_open}${unit_full}${unit_close}'
        else:
            label = name

        return label

    def make_label(
        self,
        name: str,
        unit_str: Optional[str] = None,
        scale: Optional[Union[int, float, str]] = None,
        scale_style: str = 'auto',
        unit_style: str = 'brackets',
    ) -> str:

        # Check unit style
        if unit_style != 'none' and unit_style not in _UNIT_SURROUND:
            raise ValueError(f'Invalid unit style: {unit_style}')

        if unit_style != 'none' and unit_style is not None:
            # Format scale factor
            scale_str = self._resolve_scale_format(scale, scale_style)

            # Format units with scale factor
            unit_full = ''
            if scale_str:
                unit_full = fr'{scale_str}\ {unit_str}' if unit_str else scale_str
            else:
                unit_full = unit_str
        else:
            unit_full = ''

        if unit_full:
            unit_open = _UNIT_SURROUND[unit_style][0]
            unit_close = _UNIT_SURROUND[unit_style][1]
            label = fr'{name} {unit_open}${unit_full}${unit_close}'
        else:
            label = name

        return label

    def get_plot_scale(self, key: str, default: Optional[str] = None) -> str:
        if key not in self and default is not None:
            return default
        return 'log' if self.get_descriptor(key).log else 'linear'

    def get_plot_absolute(self, key: str, default: Optional[bool] = None) -> bool:
        if key not in self and default is not None:
            return default
        return self.get_descriptor(key).absolute

    def get_unit_factor(
        self,
        key: str,
        units: Optional[str] = None,
        from_units: Optional[str] = None,
        default: Optional[float] = None
    ) -> float:
        if key not in self and default is not None:
            return float(default)

        desc = self.get_descriptor(key)
        target = units or desc.default_units
        source = from_units or desc.data_units

        # Quantities without units, or no conversion requested
        if target == source:
            if target is not None and target not in desc.units:
                raise ValueError(f'Unit system not found in descriptor {desc.key}: {target}')
            return 1.0

        if not desc.units:
            raise ValueError(f'Quantity {desc.key} has no units, cannot convert to {target or source}')
        factors = []
        for system in (target, source):
            if system is None or system not in desc.units:
                raise ValueError(f'Unit system not found in descriptor {desc.key}: {system}')
            factor = desc.units[system].conversion_factor
            if factor is None:
                raise ValueError(f'No conversion factor for unit system {system} in descriptor {desc.key}')
            factors.append(factor)

        target_factor, source_factor = factors
        return source_factor / target_factor

    def convert(
        self,
        key: str,
        value,
        units: Optional[str] = None,
        from_units: Optional[str] = None,
        default_factor: Optional[float] = None
    ):
        factor = self.get_unit_factor(key, units, from_units, default_factor)
        if np.isscalar(value):
            return value * factor
        return np.asarray(value) * factor

    def get_descriptor(self, key: str, default: Optional[LabelDescriptor] = None) -> LabelDescriptor:
        if key not in self and default is not None:
            return default
        if key not in self._alias_map:
            nucleus = _parse_isotope(key)
            if nucleus is not None:
                return self.get_isotope_descriptor(nucleus)
        return self._descriptors[self._resolve_canonical_key(key)]

    def get_isotope_descriptor(
        self,
        species: Union[str, Nucleus],
        quantity: str = 'mass fraction'
    ) -> LabelDescriptor:
        nucleus = species if isinstance(species, Nucleus) else find_isotope(species)
        desc = self._descriptors[self._resolve_canonical_key(quantity)]
        if desc.units:
            raise ValueError(f'Quantity {desc.key} has units, cannot build an isotope label from it')

        tex = _isotope_tex(nucleus)
        symbol = desc.symbol.strip('$') if desc.symbol else desc.key
        return LabelDescriptor(
            key=f'{desc.key} {nucleus}',
            aliases=[],
            symbol=f'${symbol}({tex})$',
            short=f'${tex}$ {desc.short}' if desc.short else None,
            full=f'${tex}$ {desc.full}' if desc.full else None,
            log=desc.log,
        )

    def _resolve_canonical_key(self, key: str) -> str:
        canonical_key = self._alias_map.get(key)
        if canonical_key is None:
            raise KeyError(f'Unresolved key: {key}')
        else:
            return canonical_key

    def _resolve_unit_format(self, desc: LabelDescriptor, units: Optional[str]) -> str:
        if units and units not in desc.units:
            raise ValueError(f'Unit system not found in descriptor {desc.key}: {units}')

        unit_key = units or desc.default_units
        return desc.units[unit_key].latex if unit_key else ''

    def _resolve_scale_format(self, scale: Union[int, float, str], style: str) -> str:
        precision = 3
        if scale is None:
            return ''
        elif isinstance(scale, str):
            return scale
        elif isinstance(scale, (int, float)):
            scale = float(scale)
            if math.isclose(scale, 1.0):
                return ''

            try:
                log10 = math.log10(abs(scale))
            except (ValueError, ZeroDivisionError):
                return fr'\times{scale}'

            if style == 'log':
                exponent = round(log10, 2)
                return f'10^{{{exponent:.{precision}g}}}' if not exponent.is_integer() else f'10^{int(exponent)}'
            elif style == 'linear':
                return fr'\times{scale}' if not scale.is_integer() else fr'\times{int(scale)}'
            elif style == 'auto':
                is_exact_log10 = math.isclose(log10, round(log10), rel_tol=1e-4)
                if is_exact_log10 and not math.isclose(scale, 10.0):
                    res = f'10^{{{int(round(log10))}}}'
                elif abs(scale) > 1e3 or abs(scale) < 1e-3:
                    res = f'10^{{{log10:.{precision}g}}}'
                else:
                    return fr'\times{scale}' if not scale.is_integer() else fr'\times{int(scale)}'

                return fr'\times{res}' if scale > 0.0 else fr'\times -{res}'
            else:
                raise ValueError(f'Invalid scale style: {style}')
        else:
            raise TypeError(f'Invalid scale type: {type(scale)})')

