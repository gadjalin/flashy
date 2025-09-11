# For type hints
from __future__ import annotations
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import xarray as xr
from glob import glob
from tqdm import tqdm
from pathlib import Path
import h5py

from ..io import Progenitor, DatFile, TimeSeries1D
from ..analysis import get_bounce_time
from ..nuc import find_isotope, Nucleus


_REQUIRED_FIELDS = ['dens', 'temp', 'entr', 'ye', 'fnue', 'fnua', 'fnux', 'enue', 'enua', 'enux']
_TRAJ_FIELDS = ['temp', 'dens', 'r', 'ye', 'fnue', 'fnua', 'fnux', 'enue', 'enua', 'enux', 'entr']
_TQDM_FORMAT = '{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'


def _pair_times(paths):
    time_pairs = []
    for file_path in tqdm(paths, bar_format='Analysing plot files: ' + _TQDM_FORMAT):
        with h5py.File(file_path, 'r') as f:
            for real_scalar in f['real scalars']:
                key = real_scalar[0].decode('ascii').strip()
                if key == 'time':
                    time_pairs.append((real_scalar[1], file_path))
                    break
    return sorted(time_pairs, key=lambda pair: pair[0])


def _filter_pairs(pairs, step_size):
    filtered = [pairs[0][1]]
    last_time = pairs[0][0]
    previous_pair = pairs[0]
    previous_exists = False
    for pair in pairs[1:]:
        if abs(pair[0] - last_time - float(step_size)*1e-3) < 1e-4:
            last_time = pair[0]
            filtered.append(pair[1])
            previous_exists = False
        elif (pair[0] - last_time - float(step_size)*1e-3) > 1e-4:
            if previous_exists:
                last_time = previous_pair[0]
                filtered.append(previous_pair[1])
                previous_exists = False
            else:
                actual_step_size = previous_pair[0] - last_time
                raise RuntimeError(f'No plot file found within step size of {step_size}ms: closest is {actual_step_size:.1f}ms ({last_time:.1f}-{previous_pair[0]:.1f}ms)')
        else:
            previous_exists = True
        previous_pair = pair
    return filtered


def _filter_times(times, step_size):
    filtered = [times[0]]
    previous_time = times[0]
    previous_exists = False
    for time in times[1:]:
        if abs(time - filtered[-1] - float(step_size)*1e-3) < 1e-4:
            filtered.append(time)
            previous_exists = False
        elif (time - filtered[-1] - float(step_size)*1e-3) > 1e-4:
            if previous_exists:
                filtered.append(previous_time)
                previous_exists = False
            else:
                actual_step_size = (previous_time - filtered[-1])*1e3
                raise RuntimeError(f'No plot file found within step size of {step_size}ms: closest is {actual_step_size:.1f}ms ({round(filtered[-1]*1e3)}-{round(previous_time*1e3)}ms)')
        else:
            previous_exists = True
        previous_time = time
    return filtered


class Tracers1D(object):
    def __init__(self):
        pass

    @staticmethod
    def read_simulation(
        path: Union[str, List],
        step_size: int = 5,
        basename: Optional[str] = None
    ) -> TimeSeries1D:
        if isinstance(path, str) and Path(path).is_dir():
            return Tracers1D.read_simulation_dir(path, step_size, basename)
        elif isinstance(path, str) and h5py.is_hdf5(path):
            return Tracers1D.read_simulation_series(path, step_size)
        elif isinstance(path, (list, tuple, np.ndarray)) and all(isinstance(file, str) for file in path):
            return Tracers1D.read_simulation_files(path, step_size)
        else:
            raise TypeError(f'Unrecognised simulation path: {path}')

    @staticmethod
    def read_simulation_dir(
        path: str,
        step_size: int = 5,
        basename: Optional[str] = None
    ) -> TimeSeries1D:
        Tracers1D._check_stepsize(step_size)

        if basename is not None:
            print(f'Reading directory {path} for simulation {basename} with step size {step_size}ms')
            files = glob(path + '/*' + basename + '*plt_cnt*')
        else:
            print(f'Reading simulation directory {path} with step size {step_size}ms')
            files = glob(path + '/*plt_cnt*')
        # Exclude 'forced' plot files from restarts
        # This means you better not have 'forced' in your basename
        files = [f for f in files if 'forced_hdf5' not in f]
        return Tracers1D.read_simulation_files(files, step_size)

    @staticmethod
    def read_simulation_files(
        paths: List[str],
        step_size: int = 5
    ) -> TimeSeries1D:
        Tracers1D._check_stepsize(step_size)

        pairs = _pair_times(paths)
        filtered_paths = _filter_pairs(pairs, step_size)
        return TimeSeries1D.load(filtered_paths, field_list=_REQUIRED_FIELDS)


    @staticmethod
    def read_simulation_series(
        path: str,
        step_size: int = 5
    ) -> TimeSeries1D:
        Tracers1D._check_stepsize(step_size)

        print(f'Reading time series {path} with step size {step_size}ms')
        series = TimeSeries1D.load(path)
        missing_fields = [field for field in _REQUIRED_FIELDS if field not in series]
        if missing_fields:
            raise RuntimeError(f'Provided time series is missing some fields: {missing_fields}')
        filtered_times = _filter_times(series.times, step_size)
        filtered_series = series[filtered_times]
        return filtered_series

    @staticmethod
    def calculate_mass_shells(
        series: TimeSeries1D,
        dat: Union[DatFile, str],
        n_shells: int,
        r_start: Optional[float] = None,
        t_start: Optional[float] = None
    ) -> np.ndarray:
        if isinstance(dat, str):
            dat = DatFile.load(dat)
        if t_start is None:
            # Find shock revival time
            t_start = float(dat[0]['time'][dat[0]['explosion energy'] > 1e48][0])
        if r_start is None:
            # Shock radius at shock revival, or given t_start
            r_start = float(dat[0][t_start]['max shock radius'])

        # Times where to find mass shells crossing r_start
        ts = np.linspace(t_start, series.times[-1], int(np.ceil(n_shells*3/4)))
        mass_shells = []

        # Look at every time which mass shell crosses radius at r_start
        for t in tqdm(ts, bar_format=_TQDM_FORMAT):
            data = series[t]
            mass_shells.append(float(data._data[0].interp(r=r_start)['mass'].to_numpy()))

        shells_const_radius = np.sort(mass_shells)

        # Radii where to find mass shells crossing shock
        data = series._data[-1]
        r_min = np.interp(np.max(shells_const_radius), data['mass'], data['r'])
        r_max = dat[0]['max shock radius'][-1]
        rs = np.linspace(r_min, r_max, int(np.floor(n_shells*1/4)) + 2)
        mass_shells = []

        # Look at last time step which mass shells are behind the shock and
        # above shells calculated just before
        for r in tqdm(rs[1:-1], bar_format=_TQDM_FORMAT):
            mass_shells.append(float(data.interp(r=r)['mass'].to_numpy()))

        shells_const_time = np.sort(mass_shells)
        return np.concatenate((shells_const_radius, shells_const_time))

    @staticmethod
    def calculate_seeds(
        model: Union[Progenitor, str],
        masses: np.ndarray
    ) -> Tuple[np.ndarray, List[Nucleus]]:
        if isinstance(model, str):
            model = Progenitor.load(model)

        # Find available species in progenitor
        model_net = []
        aion = []
        zion = []
        for field in model.field_list:
            try:
                species = find_isotope(field)
                model_net.append(species)
                aion.append(species.A)
                zion.append(species.Z)
            except ValueError:
                continue

        aion = np.array(aion)
        zion = np.array(zion)

        # Find radii of mass shells in progenitor
        rs = np.interp(masses, model['mass'], model['r'])
        # Find initial composition at radii
        dtypes = [(str(species), np.float64) for species in model_net]
        Xs = np.zeros(len(masses), dtype=dtypes)
        for species in model_net:
            Xs[str(species)] = np.interp(rs, model['r'], model[str(species)])

        return Xs, model_net

    @staticmethod
    def save_seeds(
        network: List[Nucleus],
        Xs: np.ndarray,
        seed_dir: str
    ) -> None:
        aion = np.array([species.A for species in network])
        zion = np.array([species.Z for species in network])

        for row,i in tqdm(zip(Xs,range(len(Xs))), total=len(Xs), bar_format=_TQDM_FORMAT):
            X = np.array(row.tolist(), dtype=np.float64)
            data = np.column_stack((aion, zion, X))
            header = 'A     Z     X \n ---------------'
            np.savetxt(f'{seed_dir}/seed_{i:>04}.dat', data, fmt=('%d', '%d', '%.6e'), header=header, delimiter='\t')

    @staticmethod
    def calculate_trajectories(
        series: TimeSeries1D,
        masses: np.ndarray
    ) -> xr.Dataset:
        # We start from the end of the simulation
        # And save data for every tracers
        L_from_f = lambda r, f: 4*np.pi*r**2*f
        dtypes = [(field, np.float64) for field in _TRAJ_FIELDS]

        times = series.times
        data_list = []

        for t,data in tqdm(series, total=len(series), bar_format=_TQDM_FORMAT):
            # Get tracers locations
            rs = np.interp(masses, data['mass'], data['r'])
            all_data = data.interp(r=rs)[_TRAJ_FIELDS + ['mass']]

            # Swap r coordinate for tracer mass
            all_data = all_data.swap_dims({'r':'mass'})
            all_data = all_data.reset_coords('r')
            all_data = all_data.rename({'mass':'tracer'})
            all_data.coords['tracer'] = masses

            # Append to data list
            all_data = all_data.assign_coords(time=t)
            data_list.append(all_data)

        # Concatenate tracer data
        trajectories = xr.concat(data_list, dim='time')

        # Reprocess neutrino data
        trajectories = trajectories.assign(lnue=L_from_f(trajectories.r, trajectories.fnue),
                                           lnua=L_from_f(trajectories.r, trajectories.fnua),
                                           lnux=L_from_f(trajectories.r, trajectories.fnux))
        trajectories = trajectories.assign(tnue=trajectories.enue/3.1513,
                                           tnua=trajectories.enua/3.1513,
                                           tnux=trajectories.enux/3.1513)

        return trajectories

    @staticmethod
    def save_trajectories(
        trajectories: xr.Dataset,
        tracer_dir: str,
        Tmax: float = 1e10
    ) -> None:
        # Save data for every tracer until that tracer crosses
        # the threshold temperature Tmax
        header = 't [s]           T [GK]           rho[g/cm^3]      r [km]           Ye               lnue [erg/s]     lanue [erg/s]    lnux [erg/s]     lanux [erg/s]    tnue [MeV]      tanue [MeV]      tnux [Mev]        tanux [Mev]     entr [k_B/baryon]\n----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

        for i in tqdm(range(len(trajectories.coords['tracer'].to_numpy())), bar_format=_TQDM_FORMAT):
            # Get tracer data
            data = trajectories.isel(tracer=i)

            # Cut where tracer stays below Tmax
            mask = data['temp'] < Tmax
            last_mask = mask[::-1].cumprod(dim='time')[::-1].astype(bool)
            cut_time = float(data.time.where(last_mask).dropna('time')[0])
            cut_data = data.sel(time=slice(cut_time, None))

            # Format data to save
            save_data = np.column_stack((cut_data.time, cut_data.temp/1e9, cut_data.dens, cut_data.r/1e5, cut_data.ye,
                                         cut_data.lnue*1e51, cut_data.lnua*1e51, cut_data.lnux*0.5e51, cut_data.lnux*0.5e51,
                                         cut_data.tnue, cut_data.tnua, cut_data.tnux, cut_data.tnux, cut_data.entr))
            save_file = tracer_dir + f'/tracer_{i:>04}.dat'
            np.savetxt(save_file, save_data, fmt=' %.6e   ', header=header)

    @staticmethod
    def _check_stepsize(step_size: int) -> None:
        # Sanity check
        if not isinstance(step_size, int) or step_size < 1:
            raise ValueError(f'Step size is required to be a positive int in milliseconds: {step_size}')

