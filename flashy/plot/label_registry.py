from typing import Dict, Optional, Union
from .label_descriptor import LabelDescriptor

import math


_UNIT_SURROUND = {
    'brackets': ('[', ']'),
    'parens': ('(', ')'),
    'open': ('', '')
}


class LabelRegistry(object):
    _descriptors: Dict[str, LabelDescriptor]
    _alias_map: Dict[str, str]

    def __init__(self):
        self._descriptors = {}
        self._alias_map = {}

    def __contains__(self, key: str) -> bool:
        return key in self._alias_map

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
        for alias in ([key] + descriptor.aliases):
            if alias in self._alias_map:
                existing = self._alias_map[alias]
                raise ValueError(f'Alias already registered for key {existing}: {alias})')

        # Register new descriptor
        self._descriptors[key] = descriptor

        # Register new aliases
        for alias in ([key] + descriptor.aliases):
            self._alias_map[alias] = key

    def get_label(
        self,
        key: str,
        units: Optional[str] = None,
        scale: Optional[Union[int, float, str]] = None,
        scale_style: str = 'auto',
        unit_style: str = 'brackets',
        name_style: str = 'symbol'
    ) -> str:
        # Get descriptor
        descriptor = self.get_descriptor(key)

        # Resolve name style
        name = getattr(descriptor, name_style, None)
        if name is None:
            raise ValueError(f'Invalid name style: {name_style}')

        # Check unit style
        if unit_style != 'none' and unit_style not in _UNIT_SURROUND:
            raise ValueError(f'Invalid unit style: {unit_style}')

        if unit_style != 'none':
            # Resolve unit system
            unit_str = self._resolve_unit_format(descriptor, units)

            # Format scale factor
            scale_str = self._resolve_scale_format(scale, scale_style)

            # Format units with scale factor
            unit_full = ''
            if scale_str:
                unit_full = f'{scale_str}\ {unit_str}' if unit_str else scale_str
            else:
                unit_full = unit_str
        else:
            unit_full = ''

        if unit_full:
            unit_open = _UNIT_SURROUND[unit_style][0]
            unit_close = _UNIT_SURROUND[unit_style][1]
            label = f'{name} {unit_open}${unit_full}${unit_close}'
        else:
            label = name

        return label

    def get_plot_scale(self, key: str) -> str:
        return 'log' if self._descriptors[self._resolve_canonical_key(key)].log else 'linear'

    def get_descriptor(self, key: str) -> LabelDescriptor:
        return self._descriptors[self._resolve_canonical_key(key)]

    def _resolve_canonical_key(self, key: str) -> str:
        canonical_key = self._alias_map.get(key)
        if canonical_key is None:
            raise KeyError(f'Unresolved key: {key}')
        else:
            return canonical_key

    def _resolve_unit_format(self, desc: LabelDescriptor, units: Optional[str]) -> str:
        if units and units not in desc.units:
            raise ValueError(f'Unit system not found in decriptor {desc.key}: {units}')

        unit_key = units or desc.default_units
        return desc.units.get(unit_key) if unit_key else ''

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
                is_exact_log10 = math.isclose(log10, round(log10), rel_tol=1e-9)
                if is_exact_log10 and not math.isclose(scale, 10.0):
                    return f'10^{{{int(round(log10))}}}'
                elif abs(scale) > 1e3 or abs(scale) < 1e-3:
                    return f'10^{{{log10:.{precision}g}}}'
                else:
                    return fr'\times{scale}' if not scale.is_integer() else fr'\times{int(scale)}'
            else:
                raise ValueError(f'Invalid scale style: {style}')
        else:
            raise TypeError(f'Invalid scale type: {type(scale)})')

