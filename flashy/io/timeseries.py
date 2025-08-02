# For type hints
from __future__ import annotations
# Abstract Base Class
from abc import ABC, abstractmethod

import numpy as np
import xarray as xr
from glob import glob
from tqdm import tqdm
from pathlib import Path
# TODO custom reader cause yt is heavy to import
import yt
import h5py


_TQDM_FORMAT = '{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'
_DEFAULT_FIELDS = ['dens', 'temp', 'pres', 'velx', 'entr', 'eint', 'ener', 'ye', 'sumy', 'gamc', 'gpot', 'deps']
_DEFAULT_REAL_SCALARS = ['bouncetime', 'hyb_trans', 'hyb_translow', 'hyb_offsetshift']


class TimeSeries1D(ABC):
    """
    Load, store, and save FLASH plot files as a time series.

    This class is the base class of AMRTimeSeries1D and
    UniformTimeSeries1D.
    AMRTimeSeries1D is used to directly store the AMR data
    from FLASH plot files. The data can be interpolated
    to a uniform grid, giving a UniformTimeSeries1D,
    which is more convenient for plotting.

    See Also
    --------
    AMRTimeSeries1D
    UniformTimeSeries1D
    """
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
    @staticmethod
    def load(path, field_list=None, basename=None) -> AMRTimeSeries1D | UniformTimeSeries1D:
        """
        Load a time series.

        Depending on the type of `path`, this can either load
        all plot files from a directory, a given list of plot files,
        or an existing time series saved as an hdf5 file.

        Parameters
        ----------
        path : str or list
            Either a path to a directory, a list of paths to plot files,
            or a path to an existing time series hdf5 save.
        field_list : list
            Used to specify which quantities to save from plot files, when
            reading plot files.
        basename : str
            Used to specify the basename of the plot files to read,
            when reading a directory

        Returns
        -------
        series : AMRTimeSeries1D or UniformTimeSeries1D

        See Also
        --------
        load_dir
        load_plots
        load_series
        """
        if isinstance(path, str) and Path(path).is_dir():
            return TimeSeries1D.load_dir(path, field_list, basename)
        elif isinstance(path, str) and h5py.is_hdf5(path):
            return TimeSeries1D.load_series(path)
        elif isinstance(path, (list, tuple, np.ndarray)) and all(isinstance(file, str) for file in path):
            return TimeSeries1D.load_plots(path, field_list)
        else:
            raise RuntimeError(f'Unrecognised file type: {path}')

    @staticmethod
    def load_dir(path, field_list=None, basename=None) -> AMRTimeSeries1D:
        """
        Load all plot files in a directory.

        Parameters
        ----------
        path : str
            Path to a directory containing FLASH plot files.
        field_list : None, 'all', or list, optional
            A list of quantities to read and save
            from the plot files into the time series.
            If None (default), reads most common quantities.
            If 'all', read all available data.
        basename : None or str, optional
            The plot files' basename for disambiguation if multiple
            simulations reside in the same directory.

        Returns
        -------
        series : AMRTimeSeries1D
        """
        obj = AMRTimeSeries1D()
        obj.read_dir(path, field_list, basename)
        return obj

    @staticmethod
    def load_plots(files, field_list=None) -> AMRTimeSeries1D:
        """
        Load a list of plot files.

        Parameters
        ----------
        files : list
            A list of paths to plot files.
        field_list : None, 'all', or list, optional
            A list of quantities to read and save
            from the plot files into the time series.
            If None (default), reads most common quantities.
            If 'all', read all available data.

        Returns
        -------
        series : AMRTimeSeries1D
        """
        obj = AMRTimeSeries1D()
        obj.read_files(files, field_list)
        return obj

    @staticmethod
    def load_series(file) -> AMRTimeSeries1D | UniformTimeSeries1D:
        """
        Load an existing time series save file.

        Parameters
        ----------
        file : str
            Path to an existing AMR or Uniform time
            series save file. Time series are saved in
            HDF5 format.

        Returns
        -------
        series : AMRTimeSeries1D or UniformTimeSeries1D
        """
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
    @property
    def times(self) -> np.ndarray:
        """
        Return a sorted list of times in the series.

        Returns
        -------
        times : list
            The list of times stored in the series
        """
        return np.asarray(self._times)

    @property
    def field_list(self) -> list[str]:
        """
        Return a list of available quantities stored in the series.

        The 'r' (radius) and 'dr' (cell size) quantities are implicitly available.
        For UniformTimeSeries1D, the 't' (time) is also available.

        Returns
        -------
        field_list : list
            The list of quantities stored in the series.
        """
        return self._field_list

    @property
    def real_scalars(self) -> dict[str, float]:
        """
        A dictionary of scalar quantities from the simulation.

        This holds useful scalar quantities from the simulation
        such as 'hyb_trans' and 'hyb_translow', the upper and lower
        boundaries of the transition region of the hybrid EoS.
        Other quantities include 'bouncetime', which is determined
        using the last available time in the series when reading
        plot files, and 'hyb_offsetshift', the progenitor dependent
        constant shift of the hybrid EoS.

        Returns
        -------
        real_scalars : dict
        """
        return self._real_scalars

    @property
    def string_scalars(self) -> dict[str, float]:
        """
        A dictionary of scalar string quantities.

        This dictionary is used to store the
        hybrid EoS transition type. The value 'hyb_transtype' is
        either 'dens' or 'temp' for a density or temperature
        based transition.

        Returns
        -------
        str_scalars : dict
        """
        return self._str_scalars

    @property
    def size(self) -> int:
        """
        The size of the time series.

        Effectively the number of times stored.
        Equivalent to 'len(series.times)'.
        """
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
    def _select_times(self, t):
        pass 

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def __reversed__(self):
        pass

    @abstractmethod
    def interp(self, r: np.ndarray, field_list=None) -> UniformTimeSeries1D:
        """
        Interpolates the data on a new, uniform grid.

        Parameters
        ----------
        r : array-like
            The new grid on which to interpolate the data.
            It must be uniform (linear or logarithmic),
            typically using `numpy.linspace` and `numpy.logspace`.
        field_list : None, 'all' or list, optional
            A list of quantities to keep in the new interpolated
            series. If None (default) or 'all', interpolate all the
            quantities in the current series.

        Returns
        -------
        series : UniformTimeSeries1D
            An new time series interpolated on the new
            grid given by the `r` parameter.
        """
        pass

    @abstractmethod
    def save_hdf5(self, file: str) -> None:
        """
        Save the time series as a HDF5 file.

        Parameters
        ----------
        file : str
            Path to the file where to save
            the time series.
        """
        pass

    @abstractmethod
    def read_hdf5(self, file: str) -> None:
        """
        Restore a time series saved as a HDF5 file.

        Parameters
        ----------
        file : str
            Path to the HDF5 file where the series
            is saved.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    # Utility methods
    def _resolve_key(self, key):
        # Resolve a scalar or an individual field
        if isinstance(key, str):
            if key in self._real_scalars:
                return self._real_scalars[key]
            elif key in self._str_scalars:
                return self._str_scalars[key]
            elif key in self._field_list:
                if len(self) == 1 and isinstance(self, AMRTimeSeries1D):
                    return self._data[0][key].to_numpy()
                elif len(self) == 1 and isinstance(self, UniformTimeSeries1D):
                    return self._data[key].to_numpy()
                else:
                    return TimeSeriesView1D(self, key)
            elif key in ['t', 'r', 'dr'] and isinstance(self, UniformTimeSeries1D):
                return self._data[key].to_numpy()
            elif key in ['t'] and isinstance(self, AMRTimeSeries1D):
                return self.times
            elif key in ['r', 'dr'] and isinstance(self, AMRTimeSeries1D):
                if len(self) == 1:
                    return self._data[0][key].to_numpy()
                else:
                    return TimeSeriesView1D(self, key)
            else:
                raise IndexError(f'Invalid key {key}')
        # Resolve multiple fields
        elif isinstance(key, (list, tuple, np.ndarray)) and all(isinstance(k, str) for k in key):
            if all(k in ['r', 'dr'] + self._field_list for k in key):
                return TimeSeriesView1D(self, key)
            else:
                raise IndexError(f'Invalid keys: {[k for k in key if k not in self._field_list]}')
        # Resolve physical time
        elif isinstance(key, float) or \
             (isinstance(key, (list, tuple, np.ndarray)) and all(isinstance(k, float) for k in key)) or \
             isinstance(key, slice):
            return self._select_times(key)
        elif isinstance(key, int):
            raise IndexError('Time selection must use physical time as float')
        else:
            raise IndexError(f'Invalid key type: {type(key)}')

    @staticmethod
    def _sanitise_field_list(field_list):
        """
        Check that all fields appear only once, and ensure
        'r' and 'dr' are not in the list, as they are implied.
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
    """
    Store a time series based on AMR data.

    When reading plot files from a FLASH simulation,
    an AMRTimeSeries1D is built using the AMR data.
    Use `interp` to construct a UniformTimeSeries1D
    using a uniform grid for plotting.

    See Also
    --------
    TimeSeries1D.load_dir
    TimeSeries1D.load_plots
    interp
    """
    _data: list[xr.Dataset]

    def __init__(self):
        self._data = None
        super().__init__()

    @classmethod
    def select_times(cls, series: AMRTimeSeries1D, t: float | list[float] | slice, field_list: list[str] = None) -> AMRTimeSeries1D:
        """
        Select specific times in the series.

        Parameters
        ----------
        series : AMRTimeSeries1D
            The series from which to select times.
        t : float, list or slice
            A specific time, a list of times, or a slice.
            The times closest to the requested ones are
            selected and used to build a new time series.
            It also ensures that there are no duplicates,
            therefore, it is not guaranteed that the resulting
            series has the same length as `t`, if `t` is a list
            of times.
        field_list : None, 'all' or list, optional
            A list of quantities to keep in the new series.
            If None (default) or 'all', keep everything.

        Returns
        -------
        series : AMRTimeSeries1D
            A new time series built from the `series` parameter
            using the times given by the `t` parameter.
        """
        obj = cls()

        # Sanitise field list
        if field_list is None or field_list == 'all':
            field_list = series._field_list.copy()
        else:
            # Make sure we actually have the requested fields in this dataset
            field_list = [field for field in list(set(field_list)) if field in series._field_list]

        # Find datasets nearest to requested times
        times = np.asarray(series._times)
        if isinstance(t, slice):
            start = t.start if t.start is not None else times[0]
            stop = t.stop if t.stop is not None else times[-1]
            step = t.step

            if step is None:
                begin = int(np.argmin(np.abs(times - start)))
                end = int(np.argmin(np.abs(times - stop)))
                ind = list(range(begin, end))
            else:
                ind = [int(np.argmin(np.abs(times - time))) for time in np.arange(start, stop, step)]
        elif isinstance(t, float):
            ind = [int(np.argmin(np.abs(times - t)))]
        else:
            ind = [int(np.argmin(np.abs(times - time))) for time in list(t)]

        # Unique entries only
        ind = sorted(set(ind))

        obj._data = [series._data[i][['dr'] + field_list] for i in ind]
        obj._times = [series._times[i] for i in ind]
        obj._source_file = series._source_file
        obj._field_list = field_list
        obj._real_scalars = series._real_scalars
        obj._str_scalars = series._str_scalars
        obj._loaded = True
        return obj

    def __getitem__(self, key):
        return self._resolve_key(key)

    def __iter__(self):
        return TimeSeriesIterator1D(self)

    def __reversed__(self):
        return TimeSeriesIterator1D(self, reverse=True)

    def interp(self, r, field_list: list[str] = None) -> UniformTimeSeries1D:
        return UniformTimeSeries1D.to_uniform(self, r, field_list)

    def read_dir(self, path: str, field_list: str | list[str] = None, basename: str = None) -> None:
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
                    grp.create_dataset(field, data=xds[field].to_numpy(), compression=None)

                grp.create_dataset('r', data=xds['r'].to_numpy(), compression=None)

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

    def _select_times(self, t):
        return AMRTimeSeries1D.select_times(self, t)

    interp.__doc__ = TimeSeries1D.interp.__doc__
    save_hdf5.__doc__ = TimeSeries1D.save_hdf5.__doc__
    read_hdf5.__doc__ = TimeSeries1D.read_hdf5.__doc__


class UniformTimeSeries1D(TimeSeries1D):
    """
    Store a time series on a uniform grid.

    A UniformTimeSeries1D can be constructed using
    the `interp` method. This is more convenient for
    plotting.

    See Also
    --------
    interp
    """
    _data: xr.Dataset

    def __init__(self):
        self._data = None
        super().__init__()

    @staticmethod
    def to_uniform(series: TimeSeries1D, r, field_list: list[str] = None) -> UniformTimeSeries1D:
        """
        Interpolate a TimeSeries1D on a uniform grid.

        Parameters
        ----------
        series : TimeSeries1D
            An AMRTimeSeries1D or UniformTimeSeries1D to interpolate.
        r : array-like
            A uniform grid of radii coordinates on which
            to interpolate the `series`.
            It must be uniform (linear or logarithmic).
        field_list : None, 'all' or list, optional
            A list of quantities to keep in the new series.
            If None (default) or 'all', keep everything.

        Returns
        -------
        series : UniformTimeSeries1D
            A new interpolated time series.
        """
        if isinstance(series, AMRTimeSeries1D):
            return UniformTimeSeries1D.interp_amr_series(series, r, field_list)
        elif isinstance(series, UniformTimeSeries1D):
            return UniformTimeSeries1D.interp_uniform_series(series, r, field_list)
        else:
            raise NotImplementedError()

    @classmethod
    def interp_amr_series(cls, series: AMRTimeSeries1D, r, field_list: list[str] = None) -> UniformTimeSeries1D:
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
            r_amr = xds.coords['r'].to_numpy()
            for field in field_list:
                data[field][i,:] = np.interp(r, r_amr, xds[field].to_numpy())

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
    def interp_uniform_series(cls, series: UniformTimeSeries1D, r, field_list: list[str] = None) -> UniformTimeSeries1D:
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
    def select_times(cls, series: UniformTimeSeries1D, t: float | list[float] | slice, field_list: list[str] = None) -> UniformTimeSeries1D:
        obj = cls()

        # Sanitise field list
        if field_list is None or field_list == 'all':
            field_list = series._field_list.copy()
        else:
            # Make sure we actually have the requested fields in this dataset
            field_list = [field for field in list(set(field_list)) if field in series._field_list]

        if isinstance(t, slice):
            obj._data = series._data.sel(t=t)[['dr'] + field_list]
        elif isinstance(t, float):
            obj._data = series._data.sel(t=[t], method='nearest')[['dr'] + field_list]
        else:
            obj._data = series._data.sel(t=list(t), method='nearest')[['dr'] + field_list]

        obj._source_file = series._source_file
        obj._times = list(obj._data.coords['t'].to_numpy())
        obj._field_list = field_list
        obj._real_scalars = series._real_scalars.copy()
        obj._str_scalars = series._str_scalars.copy()
        obj._loaded = True
        return obj

    def __getitem__(self, key):
        return self._resolve_key(key)

    def __iter__(self):
        return TimeSeriesIterator1D(self)

    def __reversed__(self):
        return TimeSeriesIterator1D(self, reverse=True)

    def interp(self, r: np.ndarray, field_list: list[str] = None):
        return UniformTimeSeries1D.to_uniform(self, r, field_list)

    @property
    def radii(self) -> np.ndarray:
        return self._data.coords['r'].to_numpy()

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

            f.create_dataset('t', data=self._data.coords['t'].to_numpy(), compression=None)
            f.create_dataset('r', data=self._data.coords['r'].to_numpy(), compression=None)
            f.create_dataset('dr', data=self._data['dr'].to_numpy(), compression=None)

            for i in range(len(self._data.coords['t'])):
                grp = f.create_group(f'step_{i:>04}')

                for attr in self._data.attrs:
                    grp.attrs[attr] = self._data.attrs[attr][i]

                for field in self._data.data_vars:
                    if field not in ['dr']:
                        grp.create_dataset(field, data=self._data[field][i,:].to_numpy(), compression=None)

    def read_hdf5(self, file: str) -> None:
        self.clear()

        self._times = []
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
                    print(f'Warning: step {i} in file {file} may be incorrectly ordered')

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

    def _select_times(self, t):
        return UniformTimeSeries1D.select_times(self, t)

    select_times.__doc__ = AMRTimeSeries1D.select_times.__doc__
    interp.__doc__ = TimeSeries1D.interp.__doc__
    save_hdf5.__doc__ = TimeSeries1D.save_hdf5.__doc__
    read_hdf5.__doc__ = TimeSeries1D.read_hdf5.__doc__


class TimeSeriesView1D(object):
    """
    A view to a time series object.

    This is typically obtained by indexing a
    time series using the square-brackets (__getitem__) operator.
    """
    _series: TimeSeries1D
    _field_list: list[str]

    def __init__(self, series: TimeSeries1D, field_list: str | list[str] = None):
        if field_list is None:
            field_list = ['r', 'dr'] + series.field_list
        else:
            field_list = [field_list] if isinstance(field_list, str) else list(set(field_list))
            if any(field not in ['r', 'dr'] + series.field_list for field in field_list):
                missing_fields = [field for field in field_list if field not in series.field_list]
                raise RuntimeError(f'Field not in series: {missing_fields}')

        self._series = series
        self._field_list = field_list

    def __contains__(self, key: str) -> bool:
        return (key in self._field_list) or (key in ['t'])

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self._field_list:
                if len(self._series) == 1:
                    if isinstance(self._series, AMRTimeSeries1D):
                        return self._series._data[0][key].to_numpy()
                    elif isinstance(self._series, UniformTimeSeries1D):
                        return self._series._data[key].to_numpy()
                else:
                    return TimeSeriesView1D(self._series, key)
            elif key == 't':
                return self._series.times
            else:
                raise IndexError(f'Key not in dataset: {key}')
        elif isinstance(key, (list, tuple, np.ndarray)):
            if all(key in self._field_list for key in list(key)):
                return TimeSeriesView1D(self._series, key)
            else:
                raise IndexError(f'Key not in dataset: {[k for k in list(key) if k not in self._field_list]}')
        else:
            raise IndexError(f'Invalid key type: {type(key)}')

    def __iter__(self):
        return TimeSeriesIterator1D(self._series, self._field_list)

    def at_time(self, t: float) -> np.ndarray | dict[str, np.ndarray]:
        """
        Get arrays of quantities at time `t`.

        Parameters
        ----------
        t : float
            A specific time.
            The time closest to it available in the series
            is used.

        Returns
        -------
        data : array-like or dict
            Returns a radial profile of the quantities
            indexed in the view at time `t`.
            It is a dictionary of arrays if multiple quantities are indexed
            in the view.
        """
        if isinstance(self._series, AMRTimeSeries1D):
            times = self._series.times
            ind = int(np.argmin(np.abs(times - t)))
            if len(self._field_list) == 1:
                return self._series._data[ind][self._field_list[0]].to_numpy()
            else:
                return {field: self._series._data[ind][field].to_numpy() for field in self._field_list}
        elif isinstance(self._series, UniformTimeSeries1D):
            if len(self._field_list) == 1:
                return self._series._data[self._field_list[0]].sel(t=t, method='nearest').to_numpy()
            else:
                return {field: self._series._data[field].sel(t=t, method='nearest').to_numpy() for field in self._field_list}

    def as_array(self) -> np.ndarray | dict[str, np.ndarray]:
        """
        Return 2D time series arrays of indexed quantities.

        Returns
        -------
        data : array-like or dict
            2D time series numpy arrays of indexed quantities,
            directly usable in e.g. matplotlib's `contour` and `pcolormesh`.
            It is a dictionary of arrays if multiple quantities are indexed
            in the view.
        """
        if isinstance(self._series, AMRTimeSeries1D):
            raise RuntimeError('Cannot construct array from AMR time series')
        elif isinstance(self._series, UniformTimeSeries1D):
            if len(self._field_list) == 1:
                return self._field_as_array(self._field_list[0])
            else:
                return {field: self._field_as_array(field) for field in self._field_list if field not in ['r', 'dr']}

    def _field_as_array(self, field: str):
        if isinstance(self._series, AMRTimeSeries1D):
            raise RuntimeError('Cannot construct array from AMR time series')
        elif isinstance(self._series, UniformTimeSeries1D):
            if len(self._series) > 1:
                return np.swapaxes(self._series._data[field].to_numpy(), 0, 1)
            else:
                return self._series._data[field].to_numpy()


class TimeSeriesIterator1D(object):
    _series: TimeSeries1D
    _field_list: list[str]
    _ind: int
    _reverse: bool

    def __init__(self, series: TimeSeries1D, field_list: list[str] = None, reverse: bool = False):
        self._series = series
        self._field_list = field_list or ['r', 'dr'] + series.field_list
        self._reverse = reverse
        self._ind = len(series) - 1 if reverse else 0

    def __iter__(self):
        return self

    def __next__(self):
        if (self._reverse and self._ind < 0) or (self._ind >= len(self._series)):
            raise StopIteration

        time = self._series._times[self._ind]

        if isinstance(self._series, AMRTimeSeries1D):
            if len(self._field_list) > 1:
                data = {field: self._series._data[self._ind][field].to_numpy()
                        for field in self._field_list}
            else:
                data = self._series._data[self._ind][self._field_list[0]].to_numpy()
        elif isinstance(self._series, UniformTimeSeries1D):
            if len(self._field_list) > 1:
                data = {field: self._series._data.isel(t=self._ind)[field].to_numpy()
                        for field in self._field_list}
            else:
                data = self._series._data.isel(t=self._ind)[self._field_list[0]].to_numpy()

        self._ind = self._ind - 1 if self._reverse else self._ind + 1
        return time, data

