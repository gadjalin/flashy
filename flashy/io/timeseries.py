# For type hints
from __future__ import annotations
# Abstract Base Class
from abc import ABC, abstractmethod

import numpy as np
import xarray as xr
# TODO yt is heavy to import
import yt
import h5py
from glob import glob
from tqdm import tqdm


_TQDM_FORMAT = '{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'
_DEFAULT_FIELDS = ['dens', 'temp', 'pres', 'velx', 'entr', 'eint', 'ener', 'ye', 'sumy', 'gamc', 'gpot', 'deps']
_DEFAULT_REAL_SCALARS = ['bouncetime', 'hyb_trans', 'hyb_translow', 'hyb_offsetshift']


class TimeSeries1D(ABC):
    _source_file: str
    _times: list[float]
    _field_list: list[str]
    _real_scalars: dict[str, float]
    _str_scalars: dict[str, str]
    _loaded: bool

    def __init__(self):
        self._source_file = ''
        self._times = None
        self._field_list = None
        self._real_scalars = None
        self._str_scalars = None
        self._loaded = False

    # Loading and saving
    # TODO interpolate in situ to return a UniformTimeSeries1D
    @staticmethod
    def from_dir(path, basename=None, field_list=None) -> AMRTimeSeries1D:
        obj = AMRTimeSeries1D()
        obj.read_dir(path, basename, field_list)
        return obj

    @staticmethod
    def from_files(files, field_list=None) -> AMRTimeSeries1D:
        obj = AMRTimeSeries1D()
        obj.read_files(files, field_list)
        return obj

    @staticmethod
    def from_hdf5(file) -> AMRTimeSeries1D | UniformTimeSeries1D:
        obj = None
        with h5py.File(file, 'r') as f:
            file_type = f.attrs.get('type', '').strip()
            if file_type == 'amr':
                obj = AMRTimeSeries1D()
            elif file_type == 'uniform':
                obj = UniformTimeSeries1D()
            else:
                raise ValueError(f'Unknown time series type {file_type}')

        obj.read_hdf5(file)
        return obj

    def _checkloaded(self) -> None:
        if not self._loaded:
            raise RuntimeError('No simulation has been loaded yet!')

    # Accessors
    def times(self) -> list[float]:
        return self._times

    def field_list(self) -> list[str]:
        return self._field_list

    def real_scalars(self) -> dict[str, float]:
        return self._real_scalars

    def string_scalars(self) -> dict[str, float]:
        return self._str_scalars

    def size(self) -> int:
        return len(self._times)

    def __len__(self) -> int:
        return len(self._times)

    def __contains__(self, key: str) -> bool:
        return (key in ['t', 'r', 'dr']) or (key in self._field_list) or \
               (key in self._real_scalars) or (key in self._str_scalars)

    def __str__(self) -> str:
        return f'{self.__class__.__name__} @ {self._source_file}; {len(self._times)} times ({self._times[0]:.4f}-{self._times[-1]:.4f} [s])'

    # Abstract methods
    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def __reversed__(self):
        pass

    @abstractmethod
    def interp(self, r: np.ndarray, fields=None) -> UniformTimeSeries1D:
        pass

    @abstractmethod
    def save_hdf5(self, file: str) -> None:
        pass

    @abstractmethod
    def read_hdf5(self, file: str) -> None:
        pass

    # Utility methods
    @staticmethod
    def _sanitise_field_list(field_list):
        """
        Check that all fields appear only once, and ensure
        'r' and 'dr' are at the beginning of the list
        """
        if field_list is None:
            field_list = _DEFAULT_FIELDS
        else:
            field_list = list(set(field_list)) # Get unique fields

        if 'r' in field_list:
            field_list.remove('r')
        if 'dr' in field_list:
            field_list.remove('dr')

        return field_list

    # TODO Put this in a more general "grid" module or something
    # I also don't think this is 100% accurate
    @staticmethod
    def _calc_dr(r: np.ndarray):
        # Assume uniform, cell-centred radii
        edges = np.zeros(len(r) + 1, dtype=np.float32)
        edges[1:-1] = 0.5 * (r[1:] + r[:-1])
        edges[0] = r[0] - 0.5 * (r[1] - r[0])
        edges[-1] = r[-1] + 0.5 * (r[-1] - r[-2])
        dr = edges[1:] - edges[:-1]
        return dr


class AMRTimeSeries1D(TimeSeries1D):
    _data: list[xr.Dataset]

    def __init__(self):
        self._data = None
        super().__init__()

    def __getitem__(self, key):
        # One string, identify one field
        if isinstance(key, str):
            if key in self._real_scalars:
                return self._real_scalars[key]
            elif key in self._str_scalars:
                return self._str_scalars[key]
            elif key in self._field_list:
                return AMRSeriesView1D(self, key)
            else:
                raise IndexError(f'Invalid key {key}')
        # A list of strings, identify multiple fields at once
        elif isinstance(key, (list, tuple, np.ndarray)) and all(isinstance(k, str) for k in key):
            if all(k in self._field_list for k in key):
                return AMRSeriesView1D(self, key)
            else:
                raise IndexError(f'Invalid keys {[k for k in key if k not in self._field_list]}')
        # A list of floats (hopefully, because ints are not always detected with numpy), identify specific timesteps
        elif isinstance(key, (list, tuple, np.ndarray)) and all(isinstance(k, float) for k in key):
            pass
        # A slice, identify a slice of time steps
        elif isinstance(key, slice):
            pass
        else:
            raise IndexError(f'Invalid index type: {type(key)}')

    def __iter__(self):
        raise NotImplementedError()

    def __reversed__(self):
        raise NotImplementedError()

    def interp(self, r, field_list: list[str] = None) -> UniformTimeSeries1D:
        return UniformTimeSeries1D.interp_amr_series(self, r, field_list)

    def read_dir(self, path: str, basename: str = None, field_list: str | list[str] = None) -> None:
        if basename is not None:
            files = glob(path + '/*' + basename + '*plt_cnt*')
        else:
            files = glob(path + '/*plt_cnt*')
        # Exclude 'forced' plot files from restarts
        # This means you better not have 'forced' in your basename
        files = [f for f in files if 'forced_hdf5' not in f]
        self.read_files(files, field_list)

    def read_files(self, files: list[str], field_list: list[str] = None) -> None:
        self.clear()
        yt.set_log_level('error') # Shut up yt while reading many files

        # Sanitise field list
        if field_list == 'all':
            ds = yt.load(files[0])
            field_list = [field[1] for field in ds.field_list]
        field_list = self._sanitise_field_list(field_list)

        self._times = []
        self._data = []
        self._real_scalars = {}
        self._str_scalars = {}

        files.sort()
        # We get latest plot file to ensure 'bouncetime' in the real scalars is correct
        last_file = self._read_plt_files(files, field_list)
        self._sort_series()
        self._read_real_scalars(last_file)

        self._source_file = files[0]
        self._field_list = field_list
        self._loaded = True
        yt.set_log_level('info') # Reset yt logging

    # Private helper methods
    def _read_plt_files(self, files: list[str], field_list: list[str]) -> str:
        # TODO Parallelism
        last_time = 0
        last_file = files[-1]
        for i,file in tqdm(zip(range(len(files)), files), total=len(files), bar_format=_TQDM_FORMAT):
            time, data = self._read_plt_file(file, field_list)
            xds = xr.Dataset(
                    data_vars={k: ('r', v) for k,v in data.items() if k not in ['r']},
                    coords={'r': data['r']},
                    attrs={'time': time}
            )

            if time > last_time:
                last_time = time
                last_file = file

            self._times.append(time)
            self._data.append(xds)

        return last_file

    def _read_plt_file(self, file, field_list):
        ds = yt.load(file)
        ad = ds.all_data()

        # TODO spherical averages for higher dimensions
        if ds.parameters['dimensionality'] > 1:
            raise RuntimeError('Expected 1D simulation but got multi-D')

        # Plot files are already 32bits floats, so we save memory
        time = np.float32(ds.current_time.v)
        data = {}
        for field in field_list:
            data[field] = ad[field].v.astype(np.float32)

        data['r'] = ad['r'].v.astype(np.float32)
        data['dr'] = ad['dr'].v.astype(np.float32)

        return time, data

    def _read_real_scalars(self, last_file: str):
        # Read real scalars
        # Using last plot file to make sure 'bouncetime' is correct
        # use h5py directly because yt sometimes has trouble decoding these fields
        with h5py.File(last_file, 'r') as f:
            # In fortran, keys are saved as byte fields, so we need to decode the ASCII
            raw_data = f['real scalars'][()]
            decoded_data = {k.decode('ascii').strip(): v for k,v in raw_data}
            for real_scalar in _DEFAULT_REAL_SCALARS:
                if real_scalar == 'hyb_trans':
                    if 'hyb_transdens' in decoded_data:
                        self._real_scalars[real_scalar] = decoded_data['hyb_transdens']
                        self._str_scalars['hyb_transtype'] = 'dens'
                    elif 'hyb_transtemp' in decoded_data:
                        self._real_scalars[real_scalar] = decoded_data['hyb_transtemp']
                        self._str_scalars['hyb_transtype'] = 'temp'
                    else:
                        print(f'Warning: real scalar {real_scalar} not found in simulation data {last_file}!')
                        self._real_scalars[real_scalar] = 0.0
                        self._str_scalars['hyb_transtype'] = 'none'
                else:
                    if real_scalar in decoded_data:
                        self._real_scalars[real_scalar] = decoded_data[real_scalar]
                    else:
                        print(f'Warning: real scalar {real_scalar} not found in simulation data {last_file}!')
                        self._real_scalars[real_scalar] = 0.0

    def save_hdf5(self, file: str) -> None:
        with h5py.File(file, 'w') as f:
            # Save real scalars
            dtype = np.dtype([('key', 'S80'), ('value', np.float64)])
            data = np.array([(k.ljust(80).encode('ascii'), v) for k,v in self._real_scalars.items()], dtype=dtype)
            f.create_dataset('real scalars', data=data, compression=None)
            # Save string scalars
            dtype = np.dtype([('key', 'S80'), ('value', 'S80')])
            data = np.array([(k.ljust(80).encode('ascii'), v.ljust(80).encode('ascii')) for k,v in self._str_scalars.items()], dtype=dtype)
            f.create_dataset('string scalars', data=data, compression=None)
            # Save time series grid type
            f.attrs['type'] = 'amr'

            f.create_dataset('t', data=self._times, compression=None)

            # Save time series
            for xds,i in zip(self._data, range(len(self._times))):
                grp = f.create_group(f'step_{i:>04}')

                for attr in xds.attrs:
                    grp.attrs[attr] = xds.attrs[attr]

                for field in xds.data_vars:
                    grp.create_dataset(field, data=xds[field].values, compression=None)

                grp.create_dataset('r', data=xds['r'].values, compression=None)

    def read_hdf5(self, file: str) -> None:
        self.clear()

        self._times = []
        self._data = []
        self._real_scalars = {}
        self._str_scalars = {}

        with h5py.File(file, 'r') as f:
            # Sanity check
            file_type = f.attrs.get('type', '')
            if file_type != 'amr':
                raise RuntimeError(f'Incorrect time series file type "{file_type}" (expected "amr")')

            # Read field list
            self._field_list = [field for field in f['step_0000'].keys() if field not in ['r', 'dr']]
            # Read real scalars
            raw_data = f['real scalars'][()]
            self._real_scalars = {k.decode('ascii').strip(): v for k,v in raw_data}
            # Read string scalars
            raw_data = f['string scalars'][()]
            self._str_scalars = {k.decode('ascii').strip(): v.decode('ascii').strip() for k,v in raw_data}

            self._times = f['t'][()]

            # Read time series
            for i in range(len(self._times)):
                grp = f[f'step_{i:>04}']
                data_vars = {}
                coords = {}
                attrs = {k: v for k,v in grp.attrs.items()}

                # Sanity check
                if abs(self._times[i] - grp.attrs.get('time')) > 1e-6:
                    print(f'Warning: step {i} in file {file} may not be incorrectly ordered')

                # Separate quantities from radius
                for key in grp.keys():
                    data = grp[key][()]
                    if key == 'r':
                        coords[key] = data
                    else:
                        data_vars[key] = ('r', data)

                # Save each timestep in a list of xarrays
                xds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
                self._data.append(xds)

        self._sort_series()

        self._source_file = file
        self._loaded = True

    def clear(self) -> None:
        self._source_file = ''
        self._times = None
        self._field_list = None
        self._real_scalars = None
        self._str_scalars = None
        self._data = None
        self._loaded = False

    def _sort_series(self) -> None:
        sort_idx = np.argsort(self._times)
        self._times = [self._times[i] for i in sort_idx]
        self._data = [self._data[i] for i in sort_idx]


class UniformTimeSeries1D(TimeSeries1D):
    _data: xr.Dataset

    def __init__(self):
        self._data = None
        super().__init__()

    @classmethod
    def interp_amr_series(cls, series: AMRTimeSeries1D, r, field_list: list[str] = None):
        obj = cls()
        r = np.asarray(r, dtype=np.float32)

        # Sanitise field list
        if field_list is None or field_list == 'all':
            field_list = series._field_list.copy()
        else:
            # Make sure we actually have the requested fields in this dataset
            field_list = [field for field in list(set(field_list)) if field in series._field_list]

        times = series._times.copy()
        # Reserve storage
        data = {field: np.zeros((len(times), len(r)), dtype=np.float32) for field in field_list}
        data['dr'] = obj._calc_dr(r)
        for i in range(len(series)):
            xds = series._data[i]
            r_amr = xds.coords['r'].values
            for field in field_list:
                data[field][i,:] = np.interp(r, r_amr, xds[field].values)

        data_vars = {k: (('t', 'r'), v) for k,v in data.items()}
        data_vars['dr'] = ('r', data['dr'])
        obj._data = xr.Dataset(
                data_vars=data_vars,
                coords={'t': times, 'r': r},
                attrs={'time': times}
        )
        obj._source_file = series._source_file
        obj._times = times
        obj._field_list = field_list
        obj._real_scalars = series._real_scalars.copy()
        obj._str_scalars = series._str_scalars.copy()
        obj._loaded = True
        return obj

    @classmethod
    def interp_uniform_series(cls, series: UniformTimeSeries1D, r, field_list: list[str] = None):
        obj = cls()
        r = np.asarray(r, dtype=np.float32)

        # Sanitise field list
        if field_list is None or field_list == 'all':
            field_list = series._field_list.copy()
        else:
            # Make sure we actually have the requested fields in this dataset
            field_list = [field for field in list(set(field_list)) if field in series._field_list]

        obj._data = series._data.interp(r=r, method='linear', assume_sorted=True)[field_list]
        obj._data['dr'] = ('r', obj._calc_dr(r))

        obj._source_file = series._source_file
        obj._times = series._times.copy()
        obj._field_list = field_list
        obj._real_scalars = series._real_scalars.copy()
        obj._str_scalars = series._str_scalars.copy()
        obj._loaded = True
        return obj

    @classmethod
    def select_times(cls, series: UniformTimeSeries1D, t: float | list[float] | slice, field_list: list[str] = None):
        obj = cls()

        # Sanitise field list
        if field_list is None or field_list == 'all':
            field_list = series._field_list.copy()
        else:
            # Make sure we actually have the requested fields in this dataset
            field_list = [field for field in list(set(field_list)) if field in series._field_list]

        if isinstance(t, slice):
            obj._data = series._data.sel(t=t)[['dr'] + field_list]
        else:
            obj._data = series._data.sel(t=list(t), method='nearest')[['dr'] + field_list]

        obj._source_file = series._source_file
        obj._times = obj._data.coords['t'].values
        obj._field_list = field_list
        obj._real_scalars = series._real_scalars.copy()
        obj._str_scalars = series._str_scalars.copy()
        obj._loaded = True
        return obj

    def __getitem__(self, key):
        # One string, identify one field
        if isinstance(key, str):
            if key in self._real_scalars:
                return self._real_scalars[key]
            elif key in self._str_scalars:
                return self._str_scalars[key]
            elif key in self._field_list:
                return TimeSeriesView1D(self, key)
            elif key in ['t', 'r', 'dr']:
                return self._data[key].to_numpy()
            else:
                raise IndexError(f'Invalid key {key}')
        # A list of strings, identify multiple fields at once
        elif isinstance(key, (list, tuple, np.ndarray)) and all(isinstance(k, str) for k in key):
            if all(k in self._field_list for k in key):
                return TimeSeriesView1D(self, key)
            else:
                raise IndexError(f'Invalid keys in list {[k for k in key if k not in self._field_list]}')
        # A float, identify the nearest time
        elif isinstance(key, float):
            return UniformTimeSeries1D.select_times(self, key)
        # A list of floats (hopefully, because ints are not always detected with numpy), identify specific timesteps
        elif isinstance(key, (list, tuple, np.ndarray)) and all(isinstance(k, float) for k in key):
            return UniformTimeSeries1D.select_times(self, key)
        # A slice, identify a slice of time steps
        elif isinstance(key, slice):
            return UniformTimeSeries1D.select_times(self, key)
        else:
            raise IndexError(f'Invalid index type: {type(key)}')

    def __iter__(self):
        raise NotImplementedError()

    def __reversed__(self):
        raise NotImplementedError()

    def interp(self, r: np.ndarray, field_list: list[str] = None):
        return UniformTimeSeries1D.interp_uniform_series(self, r, field_list)

    def save_hdf5(self, file: str) -> None:
        with h5py.File(file, 'w') as f:
            # Save real scalars
            dtype = np.dtype([('key', 'S80'), ('value', np.float64)])
            data = np.array([(k.ljust(80).encode('ascii'), v) for k,v in self._real_scalars.items()], dtype=dtype)
            f.create_dataset('real scalars', data=data, compression=None)
            # Save string scalars
            dtype = np.dtype([('key', 'S80'), ('value', 'S80')])
            data = np.array([(k.ljust(80).encode('ascii'), v.ljust(80).encode('ascii')) for k,v in self._str_scalars.items()], dtype=dtype)
            f.create_dataset('string scalars', data=data, compression=None)
            # Save time series grid type
            f.attrs['type'] = 'uniform'

            f.create_dataset('t', data=self._data.coords['t'].values, compression=None)
            f.create_dataset('r', data=self._data.coords['r'].values, compression=None)
            f.create_dataset('dr', data=self._data['dr'].values, compression=None)

            for i in range(len(self._data.coords['t'])):
                grp = f.create_group(f'step_{i:>04}')

                for attr in self._data.attrs:
                    grp.attrs[attr] = self._data.attrs[attr][i]

                for field in self._data.data_vars:
                    if field not in ['dr']:
                        grp.create_dataset(field, data=self._data[field][i,:].values, compression=None)

    def read_hdf5(self, file: str) -> None:
        self.clear()

        self._times = []
        self._data = []
        self._real_scalars = {}
        self._str_scalars = {}

        with h5py.File(file, 'r') as f:
            # Sanity check
            file_type = f.attrs.get('type', '')
            if file_type != 'uniform':
                raise RuntimeError(f'Incorrect time series file type "{file_type}" (expected "uniform")')

            # Read field list
            self._field_list = [field for field in f['step_0000'].keys() if field not in ['r', 'dr']]
            # Read real scalars
            raw_data = f['real scalars'][()]
            self._real_scalars = {k.decode('ascii').strip(): v for k,v in raw_data}
            # Read string scalars
            raw_data = f['string scalars'][()]
            self._str_scalars = {k.decode('ascii').strip(): v.decode('ascii').strip() for k,v in raw_data}

            self._times = f['t'][()]
            r = f['r'][()]
            dr = f['dr'][()]

            # Read time series
            data = {field: np.zeros((len(self._times), len(r)), dtype=np.float32) for field in self._field_list}
            coords = {'t': self._times, 'r': r}
            for i in range(len(self._times)):
                grp = f[f'step_{i:>04}']
                # Sanity check
                if abs(self._times[i] - grp.attrs.get('time')) > 1e-6:
                    print(f'Warning: step {i} in file {file} may not be incorrectly ordered')

                for field in data:
                    data[field][i,:] = grp[field]

        data_vars = {k: (('t', 'r'), v) for k,v in data.items()}
        data_vars['dr'] = ('r', dr)
        self._data = xr.Dataset(
                data_vars=data_vars,
                coords={'t': self._times, 'r': r},
                attrs={'time': self._times}
        )
        self._source_file = file
        self._loaded = True

    def clear(self) -> None:
        self._source_file = ''
        self._times = None
        self._field_list = None
        self._real_scalars = None
        self._str_scalars = None
        self._data = None
        self._loaded = False


class TimeSeriesView1D(object):
    _series: TimeSeries1D
    _field_list: list[str]

    def __init__(self, series: TimeSeries1D, field_list: str | list[str]):
        if isinstance(series, AMRTimeSeries1D):
            raise NotImplementedError()

        self._series = series
        self._field_list = [field_list] if isinstance(field_list, str) else list(field_list)

    def __contains__(self, key: str) -> bool:
        return (key in self._field_list) or (key in ['t', 'r'])

    def __getitem__(self, key: str | list[str]):
        if key == 't' or key == 'r':
            return self._series[key]
        elif all(key in self._field_list for key in list(key)):
            return UniformTimeSeriesView1D(self._series, key)
        else:
            raise IndexError(f'Invalid keys in list {[k for k in list(key) if k not in self._field_list]}')

    def at_time(self, t: float) -> np.ndarray | dict[str, np.ndarray]:
        if len(self._field_list) == 1:
            return self._series._data[self._field_list[0]].sel(t=t, method='nearest').values
        else:
            return {field: self._series._data[field].sel(t=t, method='nearest').values for field in self._field_list}

    def as_array(self) -> np.ndarray | dict[str, np.ndarray]:
        if len(self._field_list) == 1:
            return self._field_as_array(self._field_list[0])
        else:
            return {field: self._field_as_array(field) for field in self._field_list}

    def _field_as_array(self, field: str):
        if len(self._series) > 1:
            return np.swapaxes(self._series._data[field].values, 0, 1)
        else:
            return self._series._data[field].values

