from typing import List, Mapping, Optional, Union
from dataclasses import dataclass, field
from types import MappingProxyType

from .unit import Unit, TIME_UNITS, LENGTH_UNITS, MASS_UNITS, TEMPERATURE_UNITS, ENERGY_UNITS, \
                  ENTROPY_UNITS, VELOCITY_UNITS, MASS_RATE_UNITS, DENSITY_UNITS, PRESSURE_UNITS, \
                  POWER_UNITS, SPECIFIC_ENERGY_UNITS, SPECIFIC_POWER_UNITS


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
    # Unit systems available for this quantity. Plain LaTeX strings are accepted
    # for convenience, they give labels but no unit conversion (except identity).
    units: Mapping[str, Union[Unit, str]] = field(default_factory=dict)
    log: bool = field(default=False)          # Is it preferred to plot quantity in log scale?
    default_units: Optional[str] = None       # Default unit system key in the units dictionary
    data_units: Optional[str] = None          # Unit system of the data stored in the FLASH output

    def __post_init__(self):
        object.__setattr__(self, 'aliases', tuple(self.aliases))
        units = {k: (u if isinstance(u, Unit) else Unit(str(u), None)) for k,u in self.units.items()}
        object.__setattr__(self, 'units', MappingProxyType(units))
        if self.data_units is None:
            object.__setattr__(self, 'data_units', self.default_units)

        if self.default_units and self.default_units not in self.units:
            raise ValueError(f'Default units not found in unit systems: {self.default_units}')
        if self.data_units and self.data_units not in self.units:
            raise ValueError(f'Data units not found in unit systems: {self.data_units}')


_DEFAULT_DESCRIPTORS = [
    LabelDescriptor(
        key='time',
        aliases=['t'],
        symbol=r'$t$',
        short='t',
        full='Time',
        units=TIME_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='time post-bounce',
        aliases=['time post bounce', 'tbounce', 'tpb'],
        symbol=r'$t - t_\mathrm{bounce}$',
        short=r'$t_\mathrm{pb}$',
        full='Time Post Bounce',
        units=TIME_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='radius',
        aliases=['r'],
        symbol=r'$r$',
        short='r',
        full='Radius',
        units=LENGTH_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='cell size',
        aliases=['dr'],
        symbol=r'$\mathrm{d}r$',
        short='dr',
        full='Cell size',
        units=LENGTH_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='shock radius',
        aliases=['rsh', 'rshock', 'max shock radius', 'min shock radius', 'mean shock radius'],
        symbol=r'$r_\mathrm{sh}$',
        short='Shock Rad.',
        full='Shock Radius',
        units=LENGTH_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='neutron star radius',
        aliases=['ns radius'],
        symbol=r'$R_\mathrm{NS}$',
        short=r'$r_\mathrm{NS}$',
        full='Neutron Star Radius',
        units=LENGTH_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='proto-neutron star radius',
        aliases=['correct PNS radius', 'pns radius'],
        symbol=r'$R_\mathrm{PNS}$',
        short=r'$r_\mathrm{PNS}$',
        full='Proto-Neutron Star Radius',
        units=LENGTH_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='electron neutrino neutrinosphere radius',
        aliases=['R_nue'],
        symbol=r'$R_{\nu_e}$',
        short=r'$\nu_e$-sphere Rad.',
        full=r'Electron Neutrino Neutrinosphere Radius',
        units=LENGTH_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='electron antineutrino neutrinosphere radius',
        aliases=['R_anue'],
        symbol=r'$R_{\bar{\nu}_e}$',
        short=r'$\bar{\nu}_e$-sphere Rad.',
        full=r'Electron Antineutrino Neutrinosphere Radius',
        units=LENGTH_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='heavy lepton neutrino neutrinosphere radius',
        aliases=['R_nux'],
        symbol=r'$R_{\nu_x}$',
        short=r'$\nu_x$-sphere Rad.',
        full=r'Heavy Lepton Neutrino Neutrinosphere Radius',
        units=LENGTH_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='mass',
        aliases=['m'],
        symbol=r'$M$',
        short=r'$m$',
        full='Mass',
        units=MASS_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='point mass',
        aliases=[],
        symbol=r'$M_\mathrm{point}$',
        short=r'$m_\mathrm{point}$',
        full='Point Mass',
        units=MASS_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='neutron star mass',
        aliases=['ns mass'],
        symbol=r'$M_\mathrm{NS}$',
        short=r'$m_\mathrm{NS}$',
        full='Neutron Star Mass',
        units=MASS_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='proto-neutron star mass',
        aliases=['pns mass'],
        symbol=r'$M_\mathrm{PNS}$',
        short=r'$m_\mathrm{PNS}$',
        full='Proto-Neutron Star Mass',
        units=MASS_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='mass accretion rate',
        aliases=['mass accretion', 'mdot', 'accretion rate'],
        symbol=r'$\dot{M}$',
        short=r'$\dot{m}$',
        full='Mass Accretion Rate',
        units=MASS_RATE_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='density',
        aliases=['rho', 'dens'],
        symbol=r'$\rho$',
        short='Dens.',
        full='Density',
        units=DENSITY_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='central density',
        aliases=['central rho', 'rhoc', 'central dens', 'densc'],
        symbol=r'$\rho_\mathrm{c}$',
        short='Cent. Dens.',
        full='Central Density',
        units=DENSITY_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='scaled density',
        aliases=['rhor3'],
        symbol=r'$\rho r^3$',
        short='Scaled Dens.',
        full='Scaled Density',
        units=MASS_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='temperature',
        aliases=['T', 'temp'],
        symbol=r'$T$',
        short='Temp.',
        full='Temperature',
        units=TEMPERATURE_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='central temperature',
        aliases=['Tc', 'tempc', 'central temp'],
        symbol=r'$T_\mathrm{c}$',
        short='Cent. Temp.',
        full='Central Temperature',
        units=TEMPERATURE_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='pressure',
        aliases=['pres', 'press', 'P'],
        symbol=r'$P$',
        short='Pres.',
        full='Pressure',
        units=PRESSURE_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='central pressure',
        aliases=['presc', 'Pc'],
        symbol=r'$P_\mathrm{c}$',
        short='Cent. Pres.',
        full='Central Pressure',
        units=PRESSURE_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='internal energy',
        aliases=['eint'],
        symbol=r'$e_\mathrm{int}$',
        short='Int. Ener.',
        full='Internal Energy',
        units=SPECIFIC_ENERGY_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='kinetic energy',
        aliases=['ekin'],
        symbol=r'$e_\mathrm{kin}$',
        short='Kin. Ener.',
        full='Kinetic Energy',
        units=SPECIFIC_ENERGY_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='total energy',
        aliases=['ener', 'etot', 'energy'],
        symbol=r'$e_\mathrm{tot}$',
        short='Ener.',
        full='Energy',
        units=SPECIFIC_ENERGY_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='explosion energy',
        aliases=['eexp'],
        symbol=r'$E_\mathrm{exp}$',
        short='Exp. Ener.',
        full='Diagnostic Explosion Energy',
        units=ENERGY_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='gravitational potential energy',
        aliases=['gravitational potential', 'gravitational energy', 'gpot'],
        symbol=r'$\phi$',
        short='Grav. Pot. Ener.',
        full='Gravitational Potential Energy',
        units=SPECIFIC_ENERGY_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='neutrino heating rate',
        aliases=['deps'],
        symbol=r'$\dot{Q}_\nu$',
        short=r'$\nu$ Heating Rate',
        full='Neutrino Heating Rate',
        units=SPECIFIC_POWER_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='nuclear energy generation rate',
        aliases=['enuc'],
        symbol=r'$\dot{Q}_\mathrm{nuc}$',
        short=r'Nuc. Ener. Gen.',
        full='Nuclear Energy Generation Rate',
        units=SPECIFIC_POWER_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='integrated total energy',
        aliases=['E_total'],
        symbol=r'$E_\mathrm{tot}$',
        short=r'Tot. Ener.',
        full=r'Total Energy',
        units=ENERGY_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='integrated kinetic energy',
        aliases=['E_kinetic'],
        symbol=r'$E_\mathrm{kin}$',
        short=r'Kin. Ener.',
        full=r'Kinetic Energy',
        units=ENERGY_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='integrated internal energy',
        aliases=['E_internal'],
        symbol=r'$E_\mathrm{int}$',
        short=r'Int. Ener.',
        full=r'Internal Energy',
        units=ENERGY_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='integrated gravitational energy',
        aliases=['E_grav', 'gravitational binding energy'],
        symbol=r'$E_\mathrm{grav}$',
        short=r'Grav. Ener.',
        full=r'Gravitational Binding Energy',
        units=ENERGY_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='gain region binding energy',
        aliases=['E_bind,gain', 'ebind gain'],
        symbol=r'$E_\mathrm{bind,gain}$',
        short=r'Gain Bind. Ener.',
        full=r'Binding Energy in the Gain Region',
        units=ENERGY_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='gain region heating rate',
        aliases=['net heating rate, gain', 'qgain'],
        symbol=r'$\dot{Q}_\mathrm{gain}$',
        short=r'Gain Heating Rate',
        full=r'Net Heating Rate in the Gain Region',
        units=POWER_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='gain region mass',
        aliases=['mass in gain region', 'mgain'],
        symbol=r'$M_\mathrm{gain}$',
        short=r'Gain Mass',
        full=r'Mass in the Gain Region',
        units=MASS_UNITS,
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='gain region entropy',
        aliases=['avg entropy in gain', 'sgain'],
        symbol=r'$\langle s \rangle_\mathrm{gain}$',
        short=r'Gain Entr.',
        full=r'Average Entropy in the Gain Region',
        units=ENTROPY_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='electron neutrino luminosity',
        aliases=['lum nue', 'M1 lum tot nue'],
        symbol=r'$L_{\nu_e}$',
        short=r'Electron Neutrino Lum.',
        full='Electron Neutrino Luminosity',
        units=POWER_UNITS,
        data_units='B',
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='electron antineutrino luminosity',
        aliases=['lum anue', 'lum nua', 'M1 lum tot anue'],
        symbol=r'$L_{\bar{\nu}_e}$',
        short=r'Electron Antineutrino Lum.',
        full='Electron Antineutrino Luminosity',
        units=POWER_UNITS,
        data_units='B',
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='heavy lepton neutrino luminosity',
        aliases=['lum nux', 'M1 lum tot nux'],
        symbol=r'$L_{\nu_x}$',
        short=r'Heavy Lepton Neutrino Lum.',
        full='Heavy Lepton Neutrino Luminosity',
        units=POWER_UNITS,
        data_units='B',
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='neutrino luminosity',
        aliases=['lum nu', 'M1 lum tot nu'],
        symbol=r'$L_\nu$',
        short=r'Neutrino Lum.',
        full='Neutrino Luminosity',
        units=POWER_UNITS,
        data_units='B',
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='mean electron neutrino energy',
        aliases=['electron neutrino energy', 'enue', 'M1 aveE nue'],
        symbol=r'$\langle E_{\nu_e} \rangle$',
        short=r'Mean $\nu_e$ Energy',
        full='Mean Electron Neutrino Energy',
        units=ENERGY_UNITS,
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='mean electron antineutrino energy',
        aliases=['electron antineutrino energy', 'enua', 'M1 aveE anue'],
        symbol=r'$\langle E_{\bar{\nu}_e} \rangle$',
        short=r'Mean $\bar{\nu}_e$ Energy',
        full='Mean Electron Antineutrino Energy',
        units=ENERGY_UNITS,
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='mean heavy lepton neutrino energy',
        aliases=['heavy lepton neutrino energy', 'enux', 'M1 aveE nux'],
        symbol=r'$\langle E_{\nu_x} \rangle$',
        short=r'Mean $\nu_x$ Energy',
        full='Mean Heavy Lepton Neutrino Energy',
        units=ENERGY_UNITS,
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='mean neutrino energy',
        aliases=['neutrino energy', 'enu'],
        symbol=r'$\langle E_{\nu} \rangle$',
        short=r'Mean $\nu$ Energy',
        full='Mean Neutrino Energy',
        units=ENERGY_UNITS,
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='root-mean-square electron neutrino energy',
        aliases=['rms electron neutrino energy', 'rnue', 'M1 rmsE nue'],
        symbol=r'$\sqrt{\langle E^2_{\nu_e} \rangle}$',
        short=r'RMS $\nu_e$ Energy',
        full='Root-Mean-Square Electron Neutrino Energy',
        units=ENERGY_UNITS,
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='root-mean-square electron antineutrino energy',
        aliases=['rms electron antineutrino energy', 'rnua', 'M1 rmsE anue'],
        symbol=r'$\sqrt{\langle E^2_{\bar{\nu}_e} \rangle}$',
        short=r'RMS $\bar{\nu}_e$ Energy',
        full='Root-Mean-Square Electron Antineutrino Energy',
        units=ENERGY_UNITS,
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='root-mean-square heavy lepton neutrino energy',
        aliases=['rms heavy lepton neutrino energy', 'rnux', 'M1 rmsE nux'],
        symbol=r'$\sqrt{\langle E^2_{\nu_x} \rangle}$',
        short=r'RMS $\nu_x$ Energy',
        full='Root-Mean-Square Heavy Lepton Neutrino Energy',
        units=ENERGY_UNITS,
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='root-mean-square neutrino energy',
        aliases=['rms neutrino energy', 'rnu'],
        symbol=r'$\sqrt{\langle E^2_{\nu} \rangle}$',
        short=r'RMS $\nu$ Energy',
        full='Root-Mean-Square Neutrino Energy',
        units=ENERGY_UNITS,
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='velocity',
        aliases=['velx', 'vely', 'velz'],
        symbol=r'$v$',
        short='Vel.',
        full='Velocity',
        units=VELOCITY_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='radial velocity',
        aliases=['vrad'],
        symbol=r'$v_\mathrm{rad}$',
        short='Rad. Vel.',
        full='Radial Velocity',
        units=VELOCITY_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='shock velocity',
        aliases=['vsh', 'vshock'],
        symbol=r'$v_\mathrm{sh}$',
        short='Sh. Vel.',
        full='Shock Velocity',
        units=VELOCITY_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='entropy per baryon',
        aliases=['entropy', 'entr'],
        symbol=r'$s$',
        short='Entr.',
        full='Entropy',
        units=ENTROPY_UNITS,
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='electron fraction',
        aliases=['ye'],
        symbol=r'$Y_e$',
        short='Elec. Frac.',
        full='Electron Fraction',
        log=False,
    ),
    LabelDescriptor(
        key='sumy',
        aliases=[],
        symbol=r'$SumY$',
        short='SumY',
        full='SumY',
        log=False,
    ),
    LabelDescriptor(
        key='mean atomic mass',
        aliases=['mean mass number', 'average mass number', 'average atomic mass', 'abar'],
        symbol=r'$\bar{A}$',
        short=r'<A>',
        full='Mean Atomic Mass',
        log=False,
    ),
    LabelDescriptor(
        key='mean atomic number',
        aliases=['average atomic number', 'zbar'],
        symbol=r'$\bar{Z}$',
        short='<Z>',
        full='Mean Atomic Number',
        log=False,
    ),
    LabelDescriptor(
        key='atomic mass number',
        aliases=['mass number'],
        symbol=r'$A$',
        short='Mass Number',
        full='Mass Number (A)',
        log=False,
    ),
    LabelDescriptor(
        key='atomic proton number',
        aliases=['proton number', 'atomic number'],
        symbol=r'$Z$',
        short='Proton Number',
        full='Proton Number (Z)',
        log=False,
    ),
    LabelDescriptor(
        key='atomic neutron number',
        aliases=['neutron number'],
        symbol=r'$N$',
        short='Neutron Number',
        full='Neutron Number (N)',
        log=False,
    ),
    LabelDescriptor(
        key='compactness',
        aliases=['compactness M=2.5', 'xi', 'xi2.5'],
        symbol=r'$\xi_{2.5}$',
        short='Compactness',
        full=r'Compactness at 2.5$M_\odot$',
        log=False,
    ),
    LabelDescriptor(
        key='compactness M=2.0',
        aliases=['compactness M=2', 'xi2.0', 'xi2'],
        symbol=r'$\xi_{2.0}$',
        short='Compactness',
        full=r'Compactness at 2.0$M_\odot$',
        log=False,
    ),
    LabelDescriptor(
        key='compactness M=1.75',
        aliases=['xi1.75'],
        symbol=r'$\xi_{1.75}$',
        short='Compactness',
        full=r'Compactness at 1.75$M_\odot$',
        log=False,
    ),
    LabelDescriptor(
        key='mass fraction',
        aliases=[],
        symbol=r'$X$',
        short='Mass Frac.',
        full='Mass Fraction',
        log=True,
    ),
    LabelDescriptor(
        key='abundance',
        aliases=[],
        symbol=r'$Y$',
        short='Abundance',
        full='Abundance',
        log=True,
    ),
]

