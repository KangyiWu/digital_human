import numpy
import timeit
import numba
from numba import njit
numba.config.NUMBA_DEFAULT_NUM_THREADS=4
# def do_trig(x, y):
#     z = numpy.empty_like(x)
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             z[i, j] = numpy.sin(x[i, j]**2) + numpy.cos(y[i, j])
#     return z

@njit
def do_trig(x, y):
    z = numpy.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = numpy.sin(x[i, j]**2) + numpy.cos(y[i, j])
    return z

def main():

    x = numpy.random.random((1000, 1000))
    y = numpy.random.random((1000, 1000))
    do_trig(x, y)

timeit.timeit(stmt="main()", setup="import numpy; from numba import njit", number = 10)