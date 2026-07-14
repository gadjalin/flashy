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


_DEFAULT_DESCRIPTORS = [
    LabelDescriptor(
        key='time',
        aliases=['t'],
        symbol=r'$t$',
        short='t',
        full='Time',
        units={
            'cgs': r'\mathrm{s}',
            'si': r'\mathrm{s}',
            'yr': r'\mathrm{yr}',
            'Myr': r'\mathrm{Myr}',
            'Gyr': r'\mathrm{Gyr}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='time post bounce',
        aliases=['tbounce', 'tpb'],
        symbol=r'$t - t_\mathrm{bounce}$',
        short=r'$t_\mathrm{pb}$',
        full='Time Post Bounce',
        units={
            'cgs': r'\mathrm{s}',
            'si': r'\mathrm{s}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='radius',
        aliases=['r'],
        symbol=r'$r$',
        short='r',
        full='Radius',
        units={
            'cgs': r'\mathrm{cm}',
            'si': r'\mathrm{m}',
            'km': r'\mathrm{km}',
            'sun': r'R_\odot',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='cell size',
        aliases=['dr'],
        symbol=r'$\mathrm{d}r$',
        short='dr',
        full='Cell size',
        units={
            'cgs': r'\mathrm{cm}',
            'si': r'\mathrm{m}',
            'km': r'\mathrm{km}',
            'sun': r'R_\odot',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='shock radius',
        aliases=['rsh', 'rshock', 'max shock radius', 'min shock radius', 'mean shock radius'],
        symbol=r'$r_\mathrm{sh}$',
        short='Shock Rad.',
        full='Shock Radius',
        units={
            'cgs': r'\mathrm{cm}',
            'si': r'\mathrm{m}',
            'km': r'\mathrm{km}',
            'sun': r'R_\odot',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='neutron star radius',
        aliases=['ns radius'],
        symbol=r'$R_\mathrm{NS}$',
        short=r'$r_\mathrm{NS}$',
        full='Neutron Star Radius',
        units={
            'cgs': r'\mathrm{cm}',
            'si': r'\mathrm{m}',
            'km': r'\mathrm{km}',
            'sun': r'R_\odot',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='proto-neutron star radius',
        aliases=['correct PNS radius', 'pns radius'],
        symbol=r'$R_\mathrm{PNS}$',
        short=r'$r_\mathrm{PNS}$',
        full='Proto-Neutron Star Radius',
        units={
            'cgs': r'\mathrm{cm}',
            'si': r'\mathrm{m}',
            'km': r'\mathrm{km}',
            'sun': r'R_\odot',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='mass',
        aliases=['m'],
        symbol=r'$M$',
        short=r'$m$',
        full='Mass',
        units={
            'cgs': r'\mathrm{g}',
            'si': r'\mathrm{kg}',
            'amu': r'\mathrm{amu}',
            'sun': r'M_\odot',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='point mass',
        aliases=[],
        symbol=r'$M_\mathrm{point}$',
        short=r'$m_\mathrm{point}$',
        full='Point Mass',
        units={
            'cgs': r'\mathrm{g}',
            'si': r'\mathrm{kg}',
            'sun': r'M_\odot',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='neutron star mass',
        aliases=['ns mass'],
        symbol=r'$M_\mathrm{NS}$',
        short=r'$m_\mathrm{NS}$',
        full='Neutron Star Mass',
        units={
            'cgs': r'\mathrm{g}',
            'si': r'\mathrm{kg}',
            'sun': r'M_\odot',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='proto-neutron star mass',
        aliases=['pns mass'],
        symbol=r'$M_\mathrm{PNS}$',
        short=r'$m_\mathrm{PNS}$',
        full='Proto-Neutron Star Mass',
        units={
            'cgs': r'\mathrm{g}',
            'si': r'\mathrm{kg}',
            'sun': r'M_\odot',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='mass accretion rate',
        aliases=['mass accretion', 'mdot', 'accretion rate'],
        symbol=r'$\dot{M}$',
        short=r'$\dot{m}$',
        full='Mass Accretion Rate',
        units={
            'cgs': r'\mathrm{g\,s^{-1}}',
            'si': r'\mathrm{kg\,s^{-1}}',
            'sun': r'M_\odot\,\mathrm{s^{-1}}',
            'yr': r'M_\odot\,\mathrm{yr^{-1}}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='density',
        aliases=['rho', 'dens'],
        symbol=r'$\rho$',
        short='Dens.',
        full='Density',
        units={
            'cgs': r'\mathrm{g\,cm^{-3}}',
            'si': r'\mathrm{kg\,m^{-3}}',
            'fermi': r'\mathrm{fm^{-3}}',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='central density',
        aliases=['central rho', 'rhoc', 'central dens', 'densc'],
        symbol=r'$\rho_\mathrm{c}$',
        short='Cent. Dens.',
        full='Cenral Density',
        units={
            'cgs': r'\mathrm{g\,cm^{-3}}',
            'si': r'\mathrm{kg\,m^{-3}}',
            'fermi': r'\mathrm{fm^{-3}}',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='scaled density',
        aliases=['rhor3'],
        symbol=r'$\rho r^3$',
        short='Scaled Dens.',
        full='Scaled Density',
        units={
            'cgs': r'\mathrm{g}',
            'si': r'\mathrm{kg}',
            'sun': r'M_\odot',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='temperature',
        aliases=['T', 'temp'],
        symbol=r'$T$',
        short='Temp.',
        full='Temperature',
        units={
            'cgs': r'\mathrm{K}',
            'si': r'\mathrm{K}',
            'GK': r'\mathrm{GK}',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='central temperature',
        aliases=['Tc', 'tempc', 'central temp'],
        symbol=r'$T_\mathrm{c}$',
        short='Cent. Temp.',
        full='Central Temperature',
        units={
            'cgs': r'\mathrm{K}',
            'si': r'\mathrm{K}',
            'GK': r'\mathrm{GK}',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='pressure',
        aliases=['pres', 'press', 'P'],
        symbol=r'$P$',
        short='Pres.',
        full='Pressure',
        units={
            'cgs': r'\mathrm{g\,cm^{-1}\,s^{-2}}',
            'si': r'\mathrm{kg\,m^{-1}\,s^{-2}}',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='central pressure',
        aliases=['presc', 'Pc'],
        symbol=r'$P_\mathrm{c}$',
        short='Cent. Pres.',
        full='Central Pressure',
        units={
            'cgs': r'\mathrm{g\,cm^{-1}\,s^{-2}}',
            'si': r'\mathrm{kg\,m^{-1}\,s^{-2}}',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='internal energy',
        aliases=['eint'],
        symbol=r'$e_\mathrm{int}$',
        short='Int. Ener.',
        full='Internal Energy',
        units={
            'cgs': r'\mathrm{erg\,g^{-1}}',
            'si': r'\mathrm{J\,kg^{-1}}',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='kinetic energy',
        aliases=['ekin'],
        symbol=r'$e_\mathrm{kin}$',
        short='Kin. Ener.',
        full='Kinetic Energy',
        units={
            'cgs': r'\mathrm{erg\,g^{-1}}',
            'si': r'\mathrm{J\,kg^{-1}}',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='total energy',
        aliases=['ener', 'etot', 'energy'],
        symbol=r'$e_\mathrm{tot}$',
        short='Ener.',
        full='Energy',
        units={
            'cgs': r'\mathrm{erg\,g^{-1}}',
            'si': r'\mathrm{J\,kg^{-1}}',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='explosion energy',
        aliases=['eexp'],
        symbol=r'$E_\mathrm{exp}$',
        short='Exp. Ener.',
        full='Diagnostic Explosion Energy',
        units={
            'cgs': r'\mathrm{erg}',
            'si': r'\mathrm{J}',
            'Bethe': r'\mathrm{B}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='gravitational potential energy',
        aliases=['gravitational potential', 'gravitational energy', 'gpot'],
        symbol=r'$\phi$',
        short='Grav. Pot. Ener.',
        full='Gravitational Potential Energy',
        units={
            'cgs': r'\mathrm{erg\,g^{-1}}',
            'si': r'\mathrm{J\,kg^{-1}}',
        },
        log=True,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='neutrino heating rate',
        aliases=['deps'],
        symbol=r'$\dot{Q}_\nu$',
        short=r'$\nu$ Heating Rate',
        full='Neutrino Heating Rate',
        units={
            'cgs': r'\mathrm{erg\,g^{-1}\,s^{-1}}',
            'si': r'\mathrm{J\,kg^{-1}\,s^{-1}}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='nuclear energy generation rate',
        aliases=['enuc'],
        symbol=r'$\dot{Q}_\mathrm{nuc}$',
        short=r'Nuc. Ener. Gen.',
        full='Nuclear Energy Generation Rate',
        units={
            'cgs': r'\mathrm{erg\,g^{-1}\,s^{-1}}',
            'si': r'\mathrm{J\,kg^{-1}\,s^{-1}}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='electron neutrino luminosity',
        aliases=['lum nue', 'M1 lum tot nue'],
        symbol=r'$L_{\nu_e}$',
        short=r'Electron Neutrino Lum.',
        full='Electron Neutrino Luminosity',
        units={
            'cgs': r'\mathrm{erg\,s^{-1}}',
            'si': r'\mathrm{J\,s^{-1}}',
            'Bethe': r'\mathrm{B\,s^{-1}}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='electron antineutrino luminosity',
        aliases=['lum anue', 'M1 lum tot anue'],
        symbol=r'$L_{\bar{\nu}_e}$',
        short=r'Electron Antineutrino Lum.',
        full='Electron Antineutrino Luminosity',
        units={
            'cgs': r'\mathrm{erg\,s^{-1}}',
            'si': r'\mathrm{J\,s^{-1}}',
            'Bethe': r'\mathrm{B\,s^{-1}}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='heavy lepton neutrino luminosity',
        aliases=['lum nux', 'M1 lum tot nux'],
        symbol=r'$L_{\bar{\nu}_e}$',
        short=r'Heavy Lepton Neutrino Lum.',
        full='Heavy Lepton Neutrino Luminosity',
        units={
            'cgs': r'\mathrm{erg\,s^{-1}}',
            'si': r'\mathrm{J\,s^{-1}}',
            'Bethe': r'\mathrm{B\,s^{-1}}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='neutrino luminosity',
        aliases=['lum nu', 'M1 lum tot nu'],
        symbol=r'$L_\nu$',
        short=r'Neutrino Lum.',
        full='Neutrino Luminosity',
        units={
            'cgs': r'\mathrm{erg\,s^{-1}}',
            'si': r'\mathrm{J\,s^{-1}}',
            'Bethe': r'\mathrm{B\,s^{-1}}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='mean electron neutrino energy',
        aliases=['electron neutrino energy', 'enue'],
        symbol=r'$\langle E_{\nu_e} \rangle$',
        short=r'Mean $\nu_e$ Energy',
        full='Mean Electron Neutrino Energy',
        units={
            'cgs': r'\mathrm{erg}',
            'si': r'\mathrm{J}',
            'MeV': r'\mathrm{MeV}',
        },
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='mean electron antineutrino energy',
        aliases=['electron antineutrino energy', 'enua'],
        symbol=r'$\langle E_{\bar{\nu}_e} \rangle$',
        short=r'Mean $\bar{\nu}_e$ Energy',
        full='Mean Electron Antineutrino Energy',
        units={
            'cgs': r'\mathrm{erg}',
            'si': r'\mathrm{J}',
            'MeV': r'\mathrm{MeV}',
        },
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='mean heavy lepton neutrino energy',
        aliases=['heavy lepton neutrino energy', 'enux'],
        symbol=r'$\langle E_{\nu_x} \rangle$',
        short=r'Mean $\nu_x$ Energy',
        full='Mean Heavy Lepton Neutrino Energy',
        units={
            'cgs': r'\mathrm{erg}',
            'si': r'\mathrm{J}',
            'MeV': r'\mathrm{MeV}',
        },
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='mean neutrino energy',
        aliases=['neutrino energy', 'enu'],
        symbol=r'$\langle E_{\nu} \rangle$',
        short=r'Mean $\nu$ Energy',
        full='Mean Neutrino Energy',
        units={
            'cgs': r'\mathrm{erg}',
            'si': r'\mathrm{J}',
            'MeV': r'\mathrm{MeV}',
        },
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='root-mean-square electron neutrino energy',
        aliases=['rms electron neutrino energy', 'rnue'],
        symbol=r'$\sqrt{\langle E^2_{\nu_e} \rangle}$',
        short=r'RMS $\nu_e$ Energy',
        full='Root-Mean-Square Electron Neutrino Energy',
        units={
            'cgs': r'\mathrm{erg}',
            'si': r'\mathrm{J}',
            'MeV': r'\mathrm{MeV}',
        },
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='root-mean-square electron antineutrino energy',
        aliases=['rms electron antineutrino energy', 'rnua'],
        symbol=r'$\sqrt{\langle E^2_{\bar{\nu}_e} \rangle}$',
        short=r'RMS $\bar{\nu}_e$ Energy',
        full='Root-Mean-Square Electron Antineutrino Energy',
        units={
            'cgs': r'\mathrm{erg}',
            'si': r'\mathrm{J}',
            'MeV': r'\mathrm{MeV}',
        },
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='root-mean-square heavy lepton neutrino energy',
        aliases=['rms heavy lepton neutrino energy', 'rnux'],
        symbol=r'$\sqrt{\langle E^2_{\nu_x} \rangle}$',
        short=r'RMS $\nu_x$ Energy',
        full='Root-Mean-Square Heavy Lepton Neutrino Energy',
        units={
            'cgs': r'\mathrm{erg}',
            'si': r'\mathrm{J}',
            'MeV': r'\mathrm{MeV}',
        },
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='root-mean-square neutrino energy',
        aliases=['rms neutrino energy', 'rnu'],
        symbol=r'$\sqrt{\langle E^2_{\nu} \rangle}$',
        short=r'RMS $\nu$ Energy',
        full='Root-Mean-Square Neutrino Energy',
        units={
            'cgs': r'\mathrm{erg}',
            'si': r'\mathrm{J}',
            'MeV': r'\mathrm{MeV}',
        },
        log=False,
        default_units='MeV'
    ),
    LabelDescriptor(
        key='velocity',
        aliases=['velx', 'vely', 'velz'],
        symbol=r'$v$',
        short='Vel.',
        full='Velocity',
        units={
            'cgs': r'\mathrm{cm\,s^{-1}}',
            'si': r'\mathrm{m\,s^{-1}}',
            'km': r'\mathrm{km\,s^{-1}}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='radial velocity',
        aliases=['vrad'],
        symbol=r'$v_\mathrm{rad}$',
        short='Rad. Vel.',
        full='Radial Velocity',
        units={
            'cgs': r'\mathrm{cm\,s^{-1}}',
            'si': r'\mathrm{m\,s^{-1}}',
            'km': r'\mathrm{km\,s^{-1}}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='shock velocity',
        aliases=['vsh', 'vshock'],
        symbol=r'$v_\mathrm{sh}$',
        short='Sh. Vel.',
        full='Shock Velocity',
        units={
            'cgs': r'\mathrm{cm\,s^{-1}}',
            'si': r'\mathrm{m\,s^{-1}}',
            'km': r'\mathrm{km\,s^{-1}}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='entropy per baryon',
        aliases=['entropy', 'entr'],
        symbol=r'$s$',
        short='Entr.',
        full='Entropy',
        units={
            'cgs': r'k_B\,\mathrm{baryon^{-1}}',
            'si': r'k_B\,\mathrm{baryon^{-1}}',
        },
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

