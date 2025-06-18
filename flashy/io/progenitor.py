import numpy as np


class Progenitor(object):
    """
    Store FLASH progenitor data.

    This class can read progenitor profiles
    from a FLASH progenitor file, or take data from memory.
    The profiles of each variable can then be used,
    and the progenitor can be saved to a file.
    """

    # Private fields
    __source_file: str
    __comment: str
    __columns: np.ndarray
    __data: np.ndarray
    __loaded: bool

    # Constructor
    def __init__(self, filename = None, parser = 'flash', include_vars = list(), exclude_vars = list()):
        self.__comment = ''
        self.__columns = np.empty(0)
        self.__data = np.empty(0)
        self.__loaded = False

        if filename is not None:
            self.read_file(filename, parser, include_vars, exclude_vars)

    # Alternative constructors
    @classmethod
    def from_file(cls, filename, parser = 'flash', include_vars = list(), exclude_vars = list()):
        """
        Initialise progenitor object from a progenitor profile
        """
        prog = cls()
        prog.read_file(filename, parser, include_vars, exclude_vars)
        return prog

    @classmethod
    def from_data(cls, r, data, comment: str = ""):
        """
        Initialise progenitor object from memory
        """
        prog = cls()
        prog.set_data(r, data, comment)
        return prog

    def is_loaded(self) -> bool:
        return self.__loaded

    def __checkloaded(self) -> None:
        if not self.__loaded:
            raise RuntimeError('No progenitor has been stored yet!')

    def __str__(self) -> str:
        return f'Progenitor @ {self.__source_file}; {self.__comment}; {len(self.__columns)} vars, {self.cells()} cells'

    def columns(self):
        """
        Get the name of the columns.
        """
        self.__checkloaded()
        return self.__columns.copy()

    def cells(self):
        """
        Get the number of cells in the progenitor.
        """
        self.__checkloaded()
        return len(self.__data['r'])

    def comment(self):
        """
        Get the comment associated with this progenitor
        """
        self.__checkloaded()
        return self.__comment

    def set_comment(self, comment: str):
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

    def read_file(self, filename, parser = 'flash', include_vars = list(), exclude_vars = list()) -> None:
        """
        Read a progenitor file and loads the data in memory.

        Parameters
        ----------
        filename : str
            The path to the progenitor file.
        parser : str | Callable
            Allow the use of custom file parser.
            By default, the FLASH format for progenitor files
            is assumed. This variable can be use to force the use of
            another parser, by passing a Callable
            object which signature must be equivalent to
            - custom_parser(filename: str) -> (str, dict[str, list[float]])
            Alternatively, the following string values can be used
            - 'flash' or 'default': default flash parser
            - 'mesa': parser for MESA log profiles
            - 'kepler': parser for KEPLER profiles
        include_vars : list[str]
            A list of required variable names.
            All other variables are discarded.
        exclude_vars : list[str]
            A list of variable names from which the associated
            data must not be stored.

        Raises
        ------
        RuntimeError
            If parser has invalid or unknown type.
            If a variable in include_vars is missing from the data.
        """
        from typing import Callable

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

        # Check if a variable is in both include_vars and exclude_vars
        for invar in include_vars:
            if invar in exclude_vars:
                raise RuntimeError(f'Ambiguous variable {invar} found in both included and excluded variables')

        # Reset state and clear data
        self.clear()

        # Call parser and get data
        self.__comment, data = file_parser(filename)
        self.__comment = self.__comment.strip()

        # Remove unwanted columns
        for exvar in exclude_vars:
            data.pop(exvar, None)

        # Check no missing included variables from progenitor file
        for invar in include_vars:
            if invar not in data:
                raise RuntimeError(f'Variable {invar} requested but absent from progenitor file')

        if len(include_vars) > 0:
            include_vars = list(include_vars)
            if 'r' not in include_vars:
                include_vars.append('r')
            for var in list(data.keys()):
                if var not in include_vars:
                    data.pop(var, None)

        # Save data
        self.__columns = np.atleast_1d(list(data.keys()))
        dtype = [(key, float) for key in data.keys()]
        self.__data = np.array(list(zip(*data.values())), dtype=dtype)
        
        self.__source_file = filename
        self.__loaded = True

    def set_data(self, r, data, comment: str = "") -> None:
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

        self.__comment = comment
        self.__columns = np.concatenate((np.atleast_1d('r'), np.atleast_1d(list(data.keys()))))
        dtype = [(key, float) for key in self.__columns]
        self.__data = np.array(list(zip(r, *data.values())), dtype=dtype)

        self.__source_file = 'UNSAVED'
        self.__loaded = True

    def save_file(self, filename):
        """
        Save the progenitor profile to a new file.
    
        Parameters
        ----------
        filename
            The name of the file where to save the progenitor.

        Raises
        ------
        RuntimeError
            If no data has been loaded in memory yet.
        """
        self.__checkloaded()
        assert 'r' in self.__data.dtype.names, 'Progenitor must have a radial profile!'
        
        with open(filename, 'w') as f:
            print("#", self.__comment, file=f)
            print("number of variables =", len(self.__data.dtype.names) - 1, file=f)

            # Don't print the 'r' var name
            for var_name in self.__columns[1:]:
                print(var_name, file=f)
    
            for i in range(len(self.__data['r'])):
                print(*[self.__data[var_name][i] for var_name in self.__columns], file=f)

        self.__source_file = filename

    def clear(self):
        """
        Restore the object to its default, empty state.
        
        Clear all data from memory.
        """
        self.__source_file = ''
        self.__comment = ''
        self.__columns = np.empty(0)
        self.__data = np.empty(0)
        self.__loaded = False


def flash_parser(filename):
    comment = ''
    data = {}

    data_start = 0
    with open(filename, 'r') as f:
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
        fname=filename,
        skip_header=data_start,
        names=var_names,
        dtype=None,
        encoding='ascii'
    )

    data = {var: array_data[var] for var in array_data.dtype.names}

    return comment, data


def mesa_parser(filename):
    comment = ''
    data = {}

    # Read data after header and column numbers (5 lines)
    array_data = np.genfromtxt(
        fname=filename,
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

def kepler_parser(filename):
    raise NotImplementedError('KEPLER parser')
