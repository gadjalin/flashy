from .label_descriptor import LabelDescriptor


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
        aliases=['pns radius'],
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
        aliases=['pns mass', 'ns mass'],
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
        key='mass accretion rate',
        aliases=['mass accretion', 'mdot'],
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
            'B': r'\mathrm{B}',
        },
        log=False,
        default_units='cgs'
    ),
    LabelDescriptor(
        key='neutrino heating rate',
        aliases=['deps'],
        symbol=r'$\dot{Q_\nu}$',
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
        symbol=r'$\dot{Q_\mathrm{nuc}}$',
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
        aliases=['abar'],
        symbol=r'$\mathcal{\bar{A}}$',
        short='<A>',
        full='Mean Atomic Mass',
        log=False,
    ),
    LabelDescriptor(
        key='mean atomic charge',
        aliases=['zbar'],
        symbol=r'$\mathcal{\bar{Z}}$',
        short='<Z>',
        full='Mean Atomic Charge',
        log=False,
    ),
    LabelDescriptor(
        key='compactness 2.5',
        aliases=['compactness', 'xi', 'xi2.5'],
        symbol=r'$\xi_{2.5}}$',
        short='Compactness',
        full=r'Compactness at 2.5$M_\odot$',
        log=False,
    ),
    LabelDescriptor(
        key='compactness 1.75',
        aliases=['xi1.75'],
        symbol=r'$\xi_{1.75}}$',
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

