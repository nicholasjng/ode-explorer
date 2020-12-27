
import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def euler_scalar(f, double t, double y, double h, **kwargs):
    return y + h * f(t, y, **kwargs)

@cython.boundscheck(False)
@cython.wraparound(False)
def euler_ndim(f, double t, double[:] y, double h, **kwargs):
    return y + h * f(t, y, **kwargs)