from typing import Callable
import numpy as np
import re


class Progenitor(object):
    """
    Store progenitor data.

    This class can read progenitor profiles
    from a FLASH progenitor file, take data from memory,
    or parse progenitor files from other codes (mesa, kepler, ...).
    The profiles of each variable can then be used,
    and the progenitor can be saved to a FLASH file.
    """

    # Private fields
    __source_file: str
    __comment: str
    __field_list: list[str]
    __data: np.ndarray
    __loaded: bool

    # Constructor
    def __init__(self):
        self.__comment = ''
        self.__field_list = None
        self.__data = None
        self.__loaded = False

    # Alternative constructors
    @classmethod
    def from_file(cls, file: str, parser: str | Callable = 'flash', include_fields: list[str] = None, exclude_fields: list[str] = None):
        """
        Initialise progenitor object from a progenitor profile

        Parameters
        ----------
        file : str
            Path to the progenitor file.
        parser : str | Callable
            By default, the FLASH format for progenitor files
            is assumed. This variable can be use to force the use of
            another parser, by passing a Callable
            object which signature must be equivalent to
            - custom_parser(file: str) -> (str, dict[str, list[float]])
            Alternatively, the following string values can be used
            - 'flash' or 'default': default flash parser
            - 'mesa': parser for MESA log profiles
            - 'kepler': parser for KEPLER profiles
            Caution: mesa format is roughly the same for any version,
            but kepler files tend to be formatted differently.
        include_fields : list[str]
            A list of required variable names.
            All other variables are discarded.
        exclude_fields : list[str]
            A list of variable names from which the associated
            data must not be stored.
        """
        obj = cls()
        obj.read_file(file, parser, include_fields, exclude_fields)
        return obj

    @classmethod
    def from_data(cls, r: np.ndarray, data: dict[str, np.ndarray], comment: str = ""):
        """
        Initialise progenitor object from memory

        Parameters
        ----------
        r :
            Mid-cell radii of the progenitor profile in increasing order.
        data :
            Dictionary with the name of the relevant variables as keys
            and their values at each given cell radius.
        comment : str
            An additional comment to be used when saving to a file.
        """
        obj = cls()
        obj.set_data(r, data, comment)
        return obj

    def is_loaded(self) -> bool:
        return self.__loaded

    def __checkloaded(self) -> None:
        if not self.__loaded:
            raise RuntimeError('No progenitor has been loaded yet!')

    def __str__(self) -> str:
        return f'Progenitor @ {self.__source_file}; {self.__comment}; {len(self.__field_list)} fields, {self.size()} cells'

    def __len__(self) -> int:
        return len(self.__data['r'])

    def __contains__(self, key: str):
        self.__checkloaded()
        return key in self.__data.dtype.names

    def __bool__(self) -> bool:
        return self.is_loaded()

    @property
    def field_list(self):
        """
        Get the list of available fields
        """
        self.__checkloaded()
        return self.__field_list

    def size(self):
        """
        Get the number of cells in the progenitor.
        """
        self.__checkloaded()
        return len(self.__data['r'])

    @property
    def comment(self):
        """
        Get the comment associated with this progenitor
        """
        self.__checkloaded()
        return self.__comment

    @comment.setter
    def comment(self, comment: str):
        """
        Set the comment associated with this progenitor
        """
        self.__checkloaded()
        self.__comment = comment

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index):
        """
        Get the data in the specified column.

        Parameters
        ----------
        index : int | str | tuple | list | np.ndarray
            Index of the column, either by its name, or numerical index.
            Can be a list of indices or names, or a slice.
            If a list of indices is passed, they must all be ints
            or all strings.

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
        
        if isinstance(index, int):
            return self.__data[self.__data.dtype.names[index]]
        elif isinstance(index, str):
            return self.__data[index]
        elif isinstance(index, (tuple, list, np.ndarray)):
            if np.all([isinstance(i, int) for i in index]):
                return self.__data[[self.__data.dtype.names[i] for i in index]]
            elif np.all([isinstance(i, str) for i in index]):
                return self.__data[list(index)]
            else:
                raise IndexError(f'Elements of list indexing should all be of the same type: {index}')
        elif isinstance(index, slice):
            return self.__data[list(self.__data.dtype.names[index])]
        else:
            raise IndexError(f'Invalid index type: {type(index)}')

    def read_file(self, file: str, parser: str | Callable = 'flash', include_fields: list[str] = None, exclude_fields: list[str] = None) -> None:
        """
        Read a progenitor file and loads the data in memory.

        Parameters
        ----------
        file : str
            Path to the progenitor file.
        parser : str | Callable
            By default, the FLASH format for progenitor files
            is assumed. This variable can be use to force the use of
            another parser, by passing a Callable
            object which signature must be equivalent to
            - custom_parser(file: str) -> (str, dict[str, list[float]])
            Alternatively, the following string values can be used
            - 'flash' or 'default': default flash parser
            - 'mesa': parser for MESA log profiles
            - 'kepler': parser for KEPLER profiles
            Caution: mesa format is roughly the same for any version,
            but kepler files tend to be formatted differently.
        include_fields : list[str]
            A list of required variable names.
            All other variables are discarded.
        exclude_fields : list[str]
            A list of variable names from which the associated
            data must not be stored.

        Raises
        ------
        RuntimeError
            If parser has invalid or unknown type.
            If a variable in include_vars is missing from the data.
        """
        # Check progenitor file parser
        if isinstance(parser, str):
            parser = parser.lower()
            if parser in ['default', 'flash']:
                file_parser = flash_parser
            elif parser == 'mesa':
                file_parser = mesa_parser
            elif parser == 'kepler':
                file_parser = kepler_parser
            else:
                raise RuntimeError(f'Unknown parser: {parser}')
        elif isinstance(parser, Callable):
            file_parser = parser
        else:
            raise RuntimeError(f'Invalid parser type: {type(parser)}')

        if not isinstance(exclude_fields, list):
            exclude_fields = []
        if not isinstance(include_fields, list):
            include_fields = []

        # Check if a variable is in both include_fields and exclude_fields
        for field in include_fields:
            if field in exclude_fields:
                raise RuntimeError(f'Ambiguous field {field} found in both included and excluded fields')

        # Reset state and clear data
        self.clear()

        # Call parser and get data
        comment, data = file_parser(file)
        self.__comment = comment.strip()

        # Remove unwanted columns
        for field in exclude_fields:
            data.pop(field, None)

        # Check no missing included variables from progenitor file
        for field in include_fields:
            if field not in data:
                raise RuntimeError(f'Field {field} requested but not in progenitor file')

        if len(include_fields) > 0:
            if 'r' not in include_fields:
                include_fields = ['r'] + include_fields
            for field in list(data.keys()):
                if field not in include_fields:
                    data.pop(field, None)

        # Save data
        self.__field_list = list(data.keys())
        dtype = [(key, float) for key in data]
        self.__data = np.array(list(zip(*data.values())), dtype=dtype)

        self.__source_file = file
        self.__loaded = True

    def set_data(self, r: np.ndarray, data: dict[str, np.ndarray], comment: str = '') -> None:
        """
        Set the data describing the progenitor.

        Set the data for the progenitor from a list of mid-cell radii
        and a dictionary mapping the name of variables to their values
        for each given radius.

        Parameters
        ----------
        r :
            Mid-cell radii of the progenitor profile in increasing order.
        data :
            Dictionary with the name of the relevant variables as keys
            and their values at each given cell radius.
        comment : str
            An additional comment to be used when saving to a file.
        """
        self.clear()

        self.__comment = comment.strip()
        self.__field_list = ['r'] + list(data.keys())
        dtype = [(key, float) for key in self.__field_list]
        self.__data = np.array(list(zip(r, *data.values())), dtype=dtype)

        self.__source_file = ''
        self.__loaded = True

    def save_file(self, file: str):
        """
        Save the progenitor profile to a new file.

        Parameters
        ----------
        file
            The name of the file where to save the progenitor.

        Raises
        ------
        RuntimeError
            If no data has been loaded in memory yet.
        """
        self.__checkloaded()
        assert 'r' in self.__data.dtype.names, 'Progenitor must have a radial profile!'

        with open(file, 'w') as f:
            print("#", self.__comment, file=f)
            print("number of variables =", len(self.__data.dtype.names) - 1, file=f)

            # Don't print the 'r' var name
            for field in self.__field_list:
                if field != 'r':
                    print(field, file=f)
    
            for i in range(len(self.__data['r'])):
                print(*[self.__data[field][i] for field in self.__field_list], file=f)

        self.__source_file = file

    def clear(self):
        """
        Restore the object to its default, empty state.
        
        Clear all data from memory.
        """
        self.__source_file = ''
        self.__comment = ''
        self.__field_list = None
        self.__data = None
        self.__loaded = False


def flash_parser(file):
    comment = ''
    data = {}

    data_start = 0
    with open(file, 'r') as f:
        line = f.readline()
        # Read comment on first line if any
        if (line.startswith('#')):
            data_start += 1
            comment = line[1:]
            line = f.readline()
        # Read "number of variables" line
        num_vars = int(line.split()[-1])
        data_start += num_vars + 1

        # Read variables names
        var_names = ['r']
        var_names += [f.readline().split()[0] for i in range(num_vars)]

    # Read columns
    array_data = np.genfromtxt(
        fname=file,
        skip_header=data_start,
        names=var_names,
        dtype=None,
        encoding='ascii'
    )

    data = {var: array_data[var] for var in array_data.dtype.names}

    return comment, data


def mesa_parser(file):
    comment = ''
    data = {}

    # Read data after header and column numbers (5 lines)
    array_data = np.genfromtxt(
        fname=file,
        skip_header=5,
        names=True,
        dtype=None,
        encoding='ascii'
    )

    # Reorder cells from core to surface
    for name in array_data.dtype.names:
        array_data[name] = np.flip(array_data[name])

    data = {var: array_data[var] for var in array_data.dtype.names}

    return comment, data

def kepler_parser(file):
    """
    Tested with kepler progenitor files format from Woosley & Heger 2002 and Sukhbold 2018
    But they are not always the same
    """
    comment = ''
    data = {}

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

