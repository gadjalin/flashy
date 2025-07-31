from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LabelDescriptor(object):
    # Identification
    key: str            # Canonical identifier
    aliases: List[str]  # Aliases

    # Labels
    symbol: Optional[str] = None  # Symbolic representation
    short: Optional[str] = None   # Short representation
    full: Optional[str] = None    # Full representation

    # Units
    units: Dict[str, str] = field(default_factory=dict)  # Unit representations in different systems
    log: bool = field(default=False)                     # Is it preferred to plot quantity in log scale?
    default_units: Optional[str] = None                  # Default unit representation key in the `units` dictionary

    def __post_init__(self):
        if self.default_units and self.default_units not in self.units:
            raise ValueError(f'Default units not found in unit systems: {default_units}')

