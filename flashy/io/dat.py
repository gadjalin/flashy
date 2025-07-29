# For type hints
from __future__ import annotations

import numpy as np
import xarray as xr
import re


class DatRun(object):
    _source_file: str
    _field_list: list[str]
    _data: xr.DataArray

    def __init__(self, data: np.ndarray, source: str):
        """
        Create a new DatRun.

        Parameters
        ----------
        data : np.ndarray
            Must be a structured numpy array with the first column
            being the 'time' column.
        source : str
            Path to the dat file where this run comes from.
        """
        self._source_file = source
        if isinstance(data, np.ndarray):
            self._field_list = list(data.dtype.names)
            # Omit time column
            values = np.vstack([[row[field] for field in self._field_list[1:]] for row in data])
            t = data['time']
            self._data = xr.DataArray(
                values,
                coords={
                    'time': t,
                    'field': self._field_list[1:]
                },
                dims=['time', 'field']
            )
        elif isinstance(data, xr.DataArray):
            self._field_list = ['time'] + list(data.coords['field'].values)
            self._data = data
        else:
            raise RuntimeError('Invalid input data')

    def __contains__(self, key: str | int) -> bool:
        return (key in (['time'] + list(self._data.coords['field'].values))) or \
               (key in np.arange(len(self._data.coords['field'].values) + 1))

    def __getitem__(self, key):
        if isinstance(key, int):
            if key == 0:
                return self._data.coords['time'].to_numpy()
            else:
                return self._data.isel(field=key-1).to_numpy()
        elif isinstance(key, str):
            if key == 'time':
                return self._data.coords['time'].to_numpy()
            else:
                return self._data.sel(field=key).to_numpy()
        else:
            raise IndexError(f'Invalid index type: {type(key)}')

    def select_times(self, t: float | list[float] | slice) -> DatRun:
        """
        Select specific times in the run.

        Parameters
        ----------
        t : float, list or slice
            The times to select. This can be a single time,
            a list of times, or a whole slice.
            This only selects the times nearest to the request ones.

        Returns
        -------
        dat : DatRun
            A new DatRun object containing only entries for the selected times.
            This can be useful for plotting.
        """
        if isinstance(t, slice):
            data = self._data.sel(time=t)
        elif isinstance(t, float):
            data = self._data.sel(time=[t], method='nearest')
        elif isinstance(t, (list, tuple, np.ndarray)) and all(isinstance(v, float) for v in t):
            data = self._data.sel(time=list(t), method='nearest')
        else:
            raise RuntimeError(f'Invalid type: {type(t)}')

        return DatRun(data, self._source_file)

    def __str__(self):
        source = self._source_file
        n_columns = len(self._data.coords['field'].values) + 1 # Include 'time' column
        n_rows = len(self._data.coords['time'].values)
        return f'DatRun @ {source}; {n_columns} columns; {n_rows} rows'

    @property
    def field_list(self) -> list[str]:
        """
        A list of available fields (columns) in the dat file.
        """
        return self._field_list


class DatFile(object):
    """
    Store a FLASH dat file.

    This class reads and stores the simulation output
    from a FLASH dat file.
    If the file contains multiple runs,
    the data from each run are split into different lists.
    Different runs can only be detected if they are
    preceded by a new header, or if there is a jump in time.
    Thus, contiguous output resulting from from multiple restarts
    is considered as one run.
    """
    _source_file: str
    _field_list: list[str]
    _runs: list[DatRun]
    _loaded: bool

    def __init__(self):
        self._source_file = ''
        self._field_list = None
        self._runs = None
        self._loaded = False

    # TODO This may be redundant
    @staticmethod
    def load(file) -> DatFile:
        return DatFile.load_file(file)

    @classmethod
    def load_file(cls, file: str) -> DatFile:
        """
        Load a dat file.

        Parameters
        ----------
        file : str
            Path to a dat file.

        Returns
        -------
        dat : DatFile
        """
        obj = cls()
        obj.read_file(file)
        return obj

    def __checkloaded(self) -> bool:
        if not self._loaded:
            raise RuntimeError('No dat file has been loaded yet!')

    def __contains__(self, key: str | int) -> bool:
        return key in self._field_list or \
               key in np.arange(len(self._field_list) + 1)

    def __getitem__(self, key):
        return self.get(key)

    def __str__(self):
        if self._loaded:
            source = self._source_file
            n_run = len(self._runs)
            n_fields = len(self._field_list)
            return f'DatFile @ {source}; {n_run} runs; {n_fields} fields;'
        else:
            return 'No Dat file loaded'

    def __len__(self) -> int:
        return len(self._runs)

    @property
    def field_list(self) -> list[str]:
        """
        A list of available fields (columns) in the dat file.
        """
        return self._field_list

    @property
    def size(self) -> int:
        """
        How many runs this dat file contains.
        """
        return len(self._runs)

    def get(self, key) -> DatRun:
        """
        Get specific runs.

        Parameters
        ----------
        key : int, array-like or slice
            Runs to collect.

        Returns
        -------
        dat : DatRun
            A new DatRun object containing all the data
            collected from the specified runs.
            The new object is kept monotonic, any overlapping
            data points will be discarded in favour of the most recent
            ones.

        Raises
        ------
        RuntimeError
            If no file has been loaded in memory yet.
        IndexError
            If the key type is invalid.
        """
        self.__checkloaded()

        if isinstance(key, int):
            runs = [self._runs[key]]
        elif isinstance(key, (list, tuple, np.ndarray)) and all(isinstance(i, int) for i in key):
            runs = [self._runs[i] for i in key]
        elif isinstance(runs, slice):
            runs = self._runs[key]
        elif key is None:
            runs = self._runs[:]
        else:
            raise IndexError(f'Invalid index type: {type(key)}')

        new_xda = None
        for run in runs:
            xda = run._data
            if new_xda is not None:
                tmin = xda.coords['time'].values[0]
                new_xda = new_xda.sel(time=new_xda['time'] < tmin)
                xda = xr.concat([new_xda, xda], dim='time')
            new_xda = xda

        new_run = DatRun(new_xda, self._source_file)
        return new_run

    def read_file(self, file: str) -> None:
        """
        Read a dat file.

        Parameters
        ----------
        file : str
            The path to the dat file.
        """
        self.clear()

        # Assuming that wherever a header is found, every run in the file has the same columns
        with open(file, 'r') as f:
            for line in f:
                if line.strip().startswith('#'):
                    field_list = self._parse_header_line(line)
                    break
            else:
                n = len(line.split())
                field_list = [f'{i+1}' for i in range(n)]

        # Parse runs in the file
        self._runs = self._parse_dat_runs(file, field_list)

        if self._runs:
            self._source_file = file
            self._field_list = field_list
            self._loaded = True
        else:
            # Nothing was found in the dat file
            print(f'Empty dat file @ {file}')
            self._loaded = False

    def clear(self) -> None:
        self._source_file = ''
        self._field_list = None
        self._runs = None
        self._loaded = False

    def _parse_header_line(self, header: str) -> list[str]:
        field_list = []

        field_width = 26
        start = 3
        end = field_width+1

        # Read column headers. Strip the number at the beginning and save the name
        while end < len(header):
            field = header[start:end].strip()
            field = re.sub(r'^\d+\s*', '', field).strip()
            field_list.append(field)
            start = end
            end = min(start + field_width, len(header))

        return field_list

    def _parse_dat_runs(self, file: str, field_list: list[str]) -> list[DatRun]:
        raw_data = []
        runs = []
        last_time = None
        line_number = 0
        with open(file, 'r') as f:
            for line in f:
                line_number += 1
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue

                # Reached a header, start new run
                if line.startswith('#'):
                    if raw_data:
                        runs.append(self._make_run(raw_data, field_list, file))
                        raw_data = []
                    last_time = None
                    continue

                # Parse numeric values from line
                values = self._parse_dat_line(line)
                # Sanity check
                if len(values) != len(field_list):
                    raise RuntimeError(f'{file}:{line_number}: values do not match column count ({len(values)} vs {len(field_list)})')

                # Discontinuous time means restart from earlier checkpoint.
                # TODO no-overlap restarts: if no header is printed but time is
                # discontinuous, stitch runs together anyway using the most recent data
                current_time = values[0]
                if last_time is not None and current_time < last_time:
                    runs.append(self._make_run(raw_data, field_list, file))
                    raw_data = []

                raw_data.append(values)
                last_time = current_time

            # Save last run in file
            if raw_data:
                runs.append(self._make_run(raw_data, field_list, file))

        return runs

    def _parse_dat_line(self, line: str) -> list[np.float64]:
        strs = line.split()
        values = []
        for s in strs:
            try:
                v = np.float64(s)
            except ValueError:
                v = 0.0
            values.append(v)

        return values

    def _make_run(self, raw_data: list, field_list: list[str], source: str) -> DatRun:
        # Make a structured numpy array and pass to DatRun constructor
        dtype = [(field, np.float64) for field in field_list]
        data = np.array([tuple(row) for row in raw_data], dtype=dtype)
        run = DatRun(data, source)
        return run

