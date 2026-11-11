import numpy as np


def readonly(a) -> np.ndarray:
    """
    Return a read-only view of an array.

    No data is copied. Writing into the returned array raises
    a ValueError, protecting the internal state of the object
    it was obtained from. Arithmetic operations still work
    and produce new, writeable arrays.
    """
    v = np.asarray(a).view()
    v.flags.writeable = False
    return v
