import numpy as np
import re


class DatRun(object):
    __source_file: str
    __data: np.ndarray

    def __init__(self, data, source):
        self.__source_file = source
        self.__data = np.atleast_1d(data.copy())

    def __getitem__(self, index):
        return self.get(index)

    def __str__(self):
        return f'DatRun @ {self.__source_file}; {len(self.__data.dtype.names)} columns; {len(self.__data)} rows;'

    def get(self, index):
        if isinstance(index, (int, slice)):
            return self.__data[self.__data.dtype.names[index]]
        elif isinstance(index, str):
            return self.__data[index]
        elif isinstance(index, (list, tuple, np.ndarray)):
            if np.all([isinstance(i, int) for i in index]):
                return self.__data[[self.__data.dtype.names[i] for i in index]]
            elif np.all([isinstance(i, str) for i in index]):
                return self.__data[list(index)]
            else:
                raise IndexError(f'Indices must all have same type: {index}')
        else:
            raise IndexError(f'Invalid index type: {type(index)}')

    def columns(self):
        return np.asarray(self.__data.dtype.names)
    
    def data(self):
        return self.__data.copy()


class DatFile(object):
    """
    Open a FLASH dat file

    This class reads and stores the simulation output
    from a FLASH dat file.
    If the file contains multiple runs (usually starting with
    a new header, containing the columns' names),
    the data from each run are split into different lists.
    """

    __source_file: str
    __columns: np.ndarray
    __runs: list
    __loaded: bool

    def __init__(self, filename = None):
        self.__columns = np.empty(0)
        self.__runs = []
        self.__loaded = False

        if filename is not None:
            self.read_file(filename)

    @classmethod
    def from_file(cls, filename):
        obj = cls()
        obj.read_file(filename)
        return obj

    def __checkloaded(self) -> bool:
        if not self.__loaded:
            raise RuntimeError('No dat file has been loaded yet!')

    def __getitem__(self, index):
        return self.get_run(index)

    def __str__(self):
        return f'DatFile @ {self.__source_file}; {len(self.__runs)} runs; {len(self.__columns)} columns;'

    def columns(self):
        """
        Return the name of the columns.
        """
        return self.__columns.copy()

    def run_count(self):
        """
        How many runs this dat file contains.
        """
        return len(self.__runs)

    def all_data(self):
        """
        Return all columns for all, or a specific run(s).

        Returns
        -------
        An array containing the data of all runs.
        Can be indexed using the appropriate column name.

        Raises
        ------
        RuntimeError
            If no file has been loaded in memory yet, either via the
            constructor or the readfile method.
        """
        self.__checkloaded()
        return DatRun(np.concatenate([datrun.data() for datrun in self.__runs]), self.__source_file)

    def get_run(self, runs, no_overlap: bool = False):
        """
        Return the data from specific runs.

        Parameters
        ----------
        runs : int | list | tuple | slice | np.ndarray
            Index of the runs.

        no_overlap : bool
            If True, assumes a succession of restarts and remove overlapping
            data points. The most recent run overwrites the overlapping data
            from the previous ones.
            Be careful, this means that the data will only go as far as
            the latest run goes. If a more recent run went less far
            than previous ones, the longer ones will not show up.

        Returns
        -------
        The data from the specified runs.

        Raises
        ------
        RuntimeError
            If no file has been loaded in memory yet, either via the
            constructor or the readfile method.
        IndexError
            If the index type is invalid.
        """
        self.__checkloaded()

        if isinstance(runs, int):
            runlist = [self.__runs[runs]]
        elif isinstance(runs, (list, tuple, np.ndarray)) and np.all([isinstance(index, int) for index in runs]):
            runlist = [self.__runs[index] for index in runs]
        elif isinstance(runs, slice):
            runlist = self.__runs[runs]
        elif runs is None:
            runlist = self.__runs[:]
        else:
            raise RuntimeError(f'Invalid index type: {type(runs)}')

        if no_overlap:
            cutoffs = np.array([run['time'][0] for run in runlist])[1:]
            cutoffs.append(runlist[-1]['time'][-1] + 1e-6)
            return DatRun(np.concatenate([run.data()[run['time'] < cutoff] for run,cutoff in zip(runlist, cutoffs)]), self.__source_file)
        else:
            return DatRun(np.concatenate([run.data() for run in runlist], axis=0), self.__source_file)

    def read_file(self, filename) -> None:
        """
        Reads the specified dat file and loads the data in memory.

        Parameters
        ----------
        filename
            The path to the dat file.
        """
        self.clear()

        with open(filename, 'r') as f:
            l = f.readline().strip()

        no_header = False
        # First line is header
        if l.startswith('#'):
            # Read column headers. Strip the number at the beginning and save the name
            offset = 26
            width = 25
            # First column begins with a #
            columns = []
            columns.append(re.sub(r'^\d+\s*', '', l[1:width]).strip())
    
            # Read the other columns
            while (offset + width < len(l)):
                column = re.sub(r'^\d+\s*', '', l[offset:offset+width].strip()).strip()
                columns.append(column)
                offset += width+1
    
            # Read last one to avoid indexing line out of bounds
            column = re.sub(r'^\d+\s*', '', l[offset:].strip()).strip()
            if len(column) > 0:
                columns.append(column)
        else:
            # File is missing an header. Just parse the number of available columns.
            n_column = len(l.split())
            columns = [f'{i+1}' for i in range(n_column)]
            no_header = True

        with open(filename, 'r') as f:
            nline = 0
            run_starts = 1 if no_header else None
            for line in f:
                nline += 1
                
                if (line.strip().startswith('#')):
                    if run_starts is not None:
                        # Parse file between two headers (one FLASH run)
                        if (nline-run_starts-1) > 0:
                            data = np.genfromtxt(
                                fname=filename,
                                dtype=None,
                                names=columns,
                                encoding='ascii',
                                skip_header=run_starts,
                                max_rows=nline-run_starts-1
                            )
                            self.__runs.append(DatRun(data, filename))

                        run_starts = nline
                    else:
                        run_starts = nline

            # Parse last run, from header to EOF
            if run_starts is not None:
                if nline > run_starts:
                    data = np.genfromtxt(
                        fname=filename,
                        names=columns,
                        dtype=None,
                        encoding='ascii',
                        skip_header=run_starts
                    )
                    self.__runs.append(DatRun(data, filename))

            # Update column names to match python compliants one (_ instead of spaces and - etc)
            self.__columns = self.__runs[0].columns()

            if len(self.__runs) > 0:
                self.__source_file = filename
                self.__loaded = True
            else:
                # Nothing was found in the dat file
                # TODO Print some warning
                self.__loaded = False

    def clear(self) -> None:
        self.__columns = np.empty(0)
        self.__runs = []
        self.__loaded = False

