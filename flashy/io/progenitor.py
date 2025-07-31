# For type hints
from __future__ import annotations
from typing import Callable

import numpy as np
import xarray as xr
import re


class Progenitor(object):
    """
    Store progenitor data.

    This class can read progenitor profiles
    from a FLASH progenitor file, take data from memory,
    or parse progenitor files from other codes (mesa, kepler, ...).
    Radial profiles for each variable are made available
    and can be saved to a FLASH file.
    """

    # Private fields
    _source_file: str
    _comment: str
    _field_list: list[str]
    _data: xr.Dataset
    _loaded: bool

    def __init__(self):
        self._source_file = ''
        self._comment = ''
        self._field_list = None
        self._data = None
        self._loaded = False

    @staticmethod
    def load(file: str, parser: str | Callable = 'flash', include_fields: list[str] = None, exclude_fields: list[str] = None) -> Progenitor:
        return Progenitor.load_file(file, parser, include_fields, exclude_fields)

    @classmethod
    def load_file(cls, file: str, parser: str | Callable = 'flash', include_fields: list[str] = None, exclude_fields: list[str] = None) -> Progenitor:
        """
        Load progenitor from a file.

        Parameters
        ----------
        file : str
            Path to the progenitor file.
        parser : str or Callable, optional
            By default, the FLASH format for progenitor files
            is assumed. This variable can be use to force the use of
            another parser, by passing a Callable
            object which signature must be equivalent to
            - custom_parser(file: str) -> (str, np.ndarray)
            and return a numpy structured array.
            Alternatively, the following string values can be used
            - 'flash' or 'default': default flash parser
            - 'mesa': parser for MESA log profiles
            - 'kepler': parser for KEPLER profiles
            Caution: mesa format is roughly the same for any version,
            but kepler files tend to be formatted differently.
        include_fields : list[str], optional
            A list of required variable names.
            All other variables are discarded.
            If both `include_fields` and `exclude_fields` are None (default),
            all variables are stored.
            The `include_fields` and `exclude_fields` parameters
            are mutually exclusive.
        exclude_fields : list[str], optional
            A list of variable names to discard.
            If both `include_fields` and `exclude_fields` are None (default),
            all variables are stored.
            The `include_fields` and `exclude_fields` parameters
            are mutually exclusive.
        """
        obj = cls()
        obj.read(file, parser, include_fields, exclude_fields)
        return obj

    @classmethod
    def load_data(cls, r: np.ndarray, data: dict[str, np.ndarray], comment: str = '') -> Progenitor:
        """
        Load progenitor from data stored in memory.

        Parameters
        ----------
        r : array-like
            Mid-cell radii of the progenitor profile in increasing order.
        data : dict[str, array-like]
            Dictionary with the name of the relevant variables as keys
            and their values at each given cell radius.
        comment : str, optional
            An additional comment to be used when saving to a file.
        """
        obj = cls()
        obj.set_data(r, data, comment)
        return obj

    def is_loaded(self) -> bool:
        return self._loaded

    def __checkloaded(self) -> None:
        if not self._loaded:
            raise RuntimeError('No progenitor has been loaded yet!')

    def __str__(self) -> str:
        source = self._source_file
        n_fields = len(self._field_list)
        n_cells = self._data.sizes['r']
        if self._comment:
            return f'Progenitor @ {source}; {self._comment}; {n_fields} fields, {n_cells} cells'
        else:
            return f'Progenitor @ {source}; {n_fields} fields, {n_cells} cells'

    def __len__(self) -> int:
        return self.size

    def __contains__(self, key: str) -> bool:
        self.__checkloaded()
        return key in self._data.coords['field'].values

    def __bool__(self) -> bool:
        return self.is_loaded()

    @property
    def field_list(self) -> list[str]:
        """
        Get the list of available fields.
        """
        self.__checkloaded()
        return self._field_list

    @property
    def size(self) -> int:
        """
        Get the number of cells in the progenitor.
        """
        self.__checkloaded()
        return self._data.sizes['r']

    @property
    def comment(self) -> str:
        """
        Get the comment associated with this progenitor.
        """
        self.__checkloaded()
        return self._comment

    @comment.setter
    def comment(self, comment: str):
        """
        Set the comment associated with this progenitor.
        """
        self.__checkloaded()
        self._comment = comment

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key):
        """
        Get the data in the specified column.

        Parameters
        ----------
        key : str
            Name of the column.

        Returns
        -------
        The data from the specified column.

        Raises
        ------
        RuntimeError
            If no data has been loaded in memory yet.
        IndexError
            If the index type is invalid.
        """
        self.__checkloaded()
        
        if isinstance(key, str):
            if key == 'r':
                return self._data.coords['r'].to_numpy()
            else:
                return self._data.sel(field=key).to_numpy()
        else:
            raise IndexError(f'Invalid key type: {type(key)}')

    def read(self, file: str, parser: str | Callable = 'flash', include_fields: list[str] = None, exclude_fields: list[str] = None) -> None:
        file_parser = self._find_parser(parser)

        # Check optional parameters
        if include_fields is not None and exclude_fields is not None:
            raise RuntimeError('include_fields and exclude_fields are mutually exclusive')
        if not isinstance(include_fields, list):
            include_fields = []
        if not isinstance(exclude_fields, list):
            exclude_fields = []

        # Reset state and clear data
        self.clear()

        # Call parser and get data
        comment, raw_data = file_parser(file)

        # Check no missing included variables from progenitor file
        if any(field not in raw_data.dtype.names for field in include_fields):
            missing_fields = [field for field in include_fields if field not in raw_data.dtype.names]
            raise RuntimeError(f'Requested field not in progenitor file: {missing_fields}')

        field_list = list(raw_data.dtype.names)
        r = raw_data['r']
        values = np.vstack([[row[field] for field in field_list[1:]] for row in raw_data])
        data = xr.DataArray(
            values,
            coords={
                'r': r,
                'field': field_list[1:]
            },
            dims=['r', 'field']
        )

        # Remove unwanted columns
        for field in exclude_fields:
            if field in data.coords['field'].values:
                data = data.drop_sel(field=field)

        if include_fields:
            for field in data.coords['field'].values:
                if field not in include_fields:
                    data = data.drop_sel(field=field)

        self._source_file = file
        self._comment = comment.strip()
        self._data = data
        self._field_list = ['r'] + list(data.coords['field'].values)
        self._loaded = True

    def set_data(self, data: np.ndarray, comment: str = '') -> None:
        """
        Load progenitor from data stored in memory.

        Parameters
        ----------
        data : np.ndarray
            Structured numpy array where each row is a cell
            and the first column is the cell-centred radius 'r'.
        comment : str, optional
            An additional comment to be used when saving to a file.
        """
        self.clear()

        field_list = list(data.dtype.names)
        r = data['r']
        values = np.vstack([[row[field] for field in field_list[1:]] for row in data])
        xda = xr.DataArray(
            values,
            coords={
                'r': r,
                'field': field_list[1:]
            },
            dims=['r', 'field']
        )

        self._source_file = ''
        self._comment = comment.strip()
        self._data = xda
        self._field_list = field_list
        self._loaded = True

    def save(self, file: str):
        """
        Save the progenitor profile to a new file.

        Parameters
        ----------
        file : str
            The name of the file where to save the progenitor.

        Raises
        ------
        RuntimeError
            If no data has been loaded in memory yet.
        """
        self.__checkloaded()

        with open(file, 'w') as f:
            print("#", self._comment, file=f)
            print("number of variables =", len(self._field_list) - 1, file=f)

            # Don't print the 'r' var name
            for field in self._field_list:
                if field != 'r':
                    print(field, file=f)

            for i in range(self._data.sizes['r']):
                print(*([self._data.coords['r'].values[i]] + list(self._data.values[i])), file=f)

        self._source_file = file

    def clear(self):
        """
        Restore the object to its default, empty state.
        
        Clear all data from memory.
        """
        self._source_file = ''
        self._comment = ''
        self._field_list = None
        self._data = None
        self._loaded = False

    def _find_parser(self, parser):
        # Check progenitor file parser
        if isinstance(parser, str):
            parser = parser.lower()
            if parser in ['default', 'flash']:
                return flash_parser
            elif parser == 'mesa':
                return mesa_parser
            elif parser == 'kepler':
                return kepler_parser
            else:
                raise RuntimeError(f'Unknown parser: {parser}')
        elif isinstance(parser, Callable):
            return parser
        else:
            raise RuntimeError(f'Invalid parser type: {type(parser)}')

    read.__doc__ = load_file.__doc__


def flash_parser(file):
    comment = ''

    data_start = 0
    with open(file, 'r') as f:
        line = f.readline().strip()
        # Read comment on first line if any
        if line.startswith('#'):
            data_start += 1
            comment = line[1:] # Exclude the leading #
            line = f.readline().strip()
        # Read "number of variables" line
        num_vars = int(line.split()[-1])
        data_start += num_vars + 1

        # Read variables names
        var_names = ['r']
        var_names += [f.readline().split()[0].strip() for i in range(num_vars)]

    # Read columns
    data = np.genfromtxt(
        fname=file,
        skip_header=data_start,
        names=var_names,
        dtype=None,
        encoding='ascii'
    )

    return comment, data


def mesa_parser(file):
    comment = ''

    # Read data after header and column numbers (5 lines)
    data = np.genfromtxt(
        fname=file,
        skip_header=5,
        names=True,
        dtype=None,
        encoding='ascii'
    )

    # Reorder cells from core to surface
    for name in data.dtype.names:
        data[name] = np.flip(data[name])

    return comment, data

# TODO Update to new interface
def kepler_parser(file):
    """
    Tested with kepler progenitor files format from Woosley & Heger 2002 and Sukhbold 2018
    But they are not always the same
    """
    raise NotImplementedError()
    comment = ''

    skip = 2
    with open(file) as f:
        line = f.readline()
        while line.strip() == '':
            line = f.readline()
            skip += 1
        comment = f.readline()
        line = f.readline()
        while line.strip() == '':
            line = f.readline()
            skip += 1
        kepler_header = re.split(r'\s{2,}', line)
        if kepler_header[0] == '':
            kepler_header = kepler_header[1:]
        if kepler_header[-1] == '':
            kepler_header = kepler_header[:-1]
        while f.readline().strip() == '':
            skip += 1

    kepler_header = [name.lower() for name in kepler_header]

    raw_data = np.genfromtxt(
            fname=file,
            skip_header=skip,
            dtype=None,
            missing_values='---',
            filling_values=0.0,
            names=kepler_header,
            usecols=list(range(1, len(kepler_header))),
            skip_footer=2,
            encoding='ascii'
    )

    data['r'] = raw_data['radius']
    for field in raw_data.dtype.names:
        if field != 'radius':
            data[field] = raw_data[field]
    # mass is cell mass
    data['mass'] = np.cumsum(data['mass'])

    return comment, data

