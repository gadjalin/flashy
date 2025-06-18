import numpy as np
import xarray as xr
from glob import glob
from tqdm import tqdm
import yt
import h5py

_TQDM_FORMAT = '{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'
_DEFAULT_FIELDS = ['dens', 'temp', 'entr', 'eint']

class TimeSeries1D(object):
    __source_file: str
    __times: list
    __all_data: list
    __field_list: list
    __attr_list: list
    __loaded: bool

    def __init__(self):
        self.__source_file = ''
        self.__times     = []
        self.__all_data  = []
        self.__field_list= []
        self.__attr_list = []
        self.__loaded    = False

    @classmethod
    def from_directory(cls, path, basename = None, field_list = None, attr_list = None):
        obj = cls()
        if basename is not None:
            files = glob(path + '/*' + basename + '*plt_cnt*')
        else:
            files = glob(path + '/*plt_cnt*')
        files = [f for f in files if 'forced' not in f] # Exclude 'forced' plot files from restarts
        files.sort()
        obj.read_files(files, field_list, attr_list)
        return obj

    @classmethod
    def from_files(cls, files, field_list = None, attr_list = None):
        obj = cls()
        obj.read_files(files, field_list, attr_list)
        return obj

    @classmethod
    def from_save(cls, save_file, field_list = None, attr_list = None):
        obj = cls()
        obj.read_save(save_file, field_list, attr_list)
        return obj

    def __checkloaded(self) -> bool:
        if not self.__loaded:
            raise RuntimeError('No simulation has been loaded yet!')

    def __getitem__(self, index):
        return self.__all_data[index]

    def __str__(self):
        return f'1D Time Series @ {self.__source_file}; {len(self.__times)} times ({self.__times[0]:.4f}-{self.__times[-1]:.4f} [s])'

    def field_list(self):
        return self.__field_list

    def attr_list(self):
        return self.__attr_list

    def size(self):
        return len(self.__times)

    def times(self):
        return np.array(self.__times)

    def read_files(self, files, field_list = None, attr_list = None):
        yt.set_log_level('error')

        # Sanitise field list
        if field_list == 'all':
            ds = yt.load(files[0])
            field_list = [field[1] for field in ds.field_list]

        field_list = self._sanitise_field_list(field_list)
        attr_list = self._sanitise_attr_list(attr_list)

        self.__times = []
        self.__all_data = []

        # TODO Parallelism
        # Read simulation data
        for i,file in tqdm(zip(range(len(files)), files), total=len(files), bar_format=_TQDM_FORMAT):
#        for file in files:
            time, data, attrs = self._read_file(file, field_list, attr_list)
            self.__times.append(time)
            xds = xr.Dataset(
                    data_vars={k: ('r', v) for k,v in data.items() if k not in ['r']},
                    coords={'r': data['r']},
                    attrs={'time': time, 'source_file': file}
            )
            for k,v in attrs.items():
                xds.attrs[k] = v

            self.__all_data.append(xds)

        self._sort_series()

        self.__source_file = files[0]
        self.__field_list = field_list
        self.__attr_list = attr_list
        self.__loaded = True
        yt.set_log_level('info')

    def _read_file(self, file, field_list, attr_list):
        ds = yt.load(file)
        ad = ds.all_data()

        if ds.parameters['dimensionality'] > 1:
            raise RuntimeError('Expected 1D simulation but got multi-D')

        time = float(ds.current_time.v)
        data = {}
        attrs = {}
        for field in field_list:
            data[field] = ad[field].v

        for attr in attr_list:
            attrs[attr] = ds.parameters[attr]

        return time, data, attrs

    def read_save(self, save_file, field_list = None, attr_list = None):
        self.__times = []
        self.__all_data = []

        with h5py.File(save_file, 'r') as f:
            for group_name in f.keys():
                grp = f[group_name]
                data_vars = {}
                coords = {}
                for key in grp.keys():
                    data = grp[key][()]
                    if key == 'r':
                        coords[key] = data
                    else:
                        data_vars[key] = ('r', data)

                xds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=grp.attrs)
                self.__times.append(xds.attrs['time'])
                self.__all_data.append(xds)

            self.__field_list = list(f[list(f.keys())[0]].keys())
            self.__attr_list  = list(f[list(f.keys())[0]].attrs.keys())

        self._sort_series()

        self.__source_file = save_file
        self.__loaded = True

    def write_save(self, save_file):
        with h5py.File(save_file, 'w') as f:
            for xds in self.__all_data:
                source_index = xds.attrs['source_file'][-4:]
                grp = f.create_group(f"{source_index}")

                for attr in xds.attrs:
                    grp.attrs[attr] = xds.attrs[attr]

                for field in xds.data_vars:
                    data = xds[field].values
                    grp.create_dataset(field, data=data, compression=None)

                for coord in xds.coords:
                    if coord not in xds.data_vars:
                        data = xds[coord].values
                        grp.create_dataset(coord, data=data, compression=None)

    def clear(self) -> None:
        self.__source_file = ''
        self.__times     = []
        self.__all_data  = []
        self.__field_list= []
        self.__attr_list = []
        self.__loaded    = False

    def _sort_series(self):
        sort_idx = np.argsort(self.__times)
        self.__times = [self.__times[i] for i in sort_idx]
        self.__all_data = [self.__all_data[i] for i in sort_idx]

    def _sanitise_field_list(self, field_list):
        if field_list is None or field_list == 'default':
            field_list = _DEFAULT_FIELDS
        else:
            field_list = list(set(field_list)) # Get unique fields

        if 'r' in field_list:
            field_list.remove('r')
        if 'dr' in field_list:
            field_list.remove('dr')

        field_list = ['r', 'dr'] + field_list

        return field_list

    def _sanitise_attr_list(self, attr_list):
        if attr_list is None:
            attr_list = []
        if 'time' in attr_list:
            attr_list.remove('time')

        return attr_list

