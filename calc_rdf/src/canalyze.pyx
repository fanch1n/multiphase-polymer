# canalyze.pyx:

cimport cython
from libc.math cimport exp, log, sqrt, fabs, pi, ceil, floor
cimport numpy as np

import numpy as np

@cython.boundscheck(False)
def calc_com_rdf_fast(np.ndarray[np.double_t, ndim=1] L, double ddr, double maxdr, molcom, moltypes):
    # Calculate radial distribution function for polymer COMs.
    # L: box dimensions
    # ddr: bin width for rdf
    # maxdf: cutoff for rdf (must be less than min(L)/2)
    # molcom: (N_molecules x 3) matrix of COM positions #FIXME currently using dictionary
    # moltypes: N_molecules-length dictionary assigning each molecule id 1,...,N_molecules to a molecule type
    cdef int i, j, d
    cdef double r, r_d
    cdef double maxdr2 = maxdr*maxdr
    cdef int nbins = int(maxdr / ddr) + 1
    cdef double V = L[0] * L[1] * L[2]
    cdef int N = len(molcom)
    cdef np.ndarray[np.double_t, ndim=2] molcom_pbc = np.zeros((N, 3), dtype=np.double)
    for i in molcom: # Put the COMs back in the box.
        for d in range(3):
            molcom_pbc[i-1,d] = molcom[i][d] % L[d]
    #        #molcom_pbc[i-1,d] = molcom[i-1][d] % L[d]
    #        #molcom_pbc[i-1,d] = molcom[i-1][d] - np.floor(molcom[i-1][d] / L[d]) * L[d]


    types = set(moltypes.values())
    rdf = {(ti, tj) : np.zeros(nbins, dtype=np.double) for ti in types for tj in types}
    n = {k : 0 for k in rdf}
    for i in range(N):
        for j in range(i + 1, N):
        #for j in range(N):
            r = 0
            for d in range(3):
                r_d = molcom_pbc[j,d] - molcom_pbc[i,d]
                if fabs(r_d) < L[d] / 2.:
                    r += r_d**2
                elif r_d > L[d] / 2.:
                    r += (r_d - L[d])**2
                else:
                    r += (r_d + L[d])**2
            k = (moltypes[i+1], moltypes[j+1])
            n[k] += 1
            if r < maxdr2:
                r = sqrt(r)
                rdf[k][int(r / ddr)] += 1.
    dr = np.array([ddr * (i + 0.5) for i in range(nbins)])
    return dr, rdf, n

# MAKEFILE:

# # Python include path
# PYTHONINC = -I/usr/include/python3.7m
# NUMPYINC = -I/usr/lib/python3.7/site-packages/numpy/core/include

# CC = gcc
# CYTHON = cython
# CPPFLAGS = $(PYTHONINC) $(NUMPYINC)
# CFLAGS = -fPIC -O3 -g -Wall
# LDFLAGS = -L`pwd` -Wl,-rpath=`pwd`
# LIBS = -lm

# all:	canalyze.so

# canalyze.so:	canalyze.o
# 	$(CC) -shared -o canalyze.so canalyze.o $(LDFLAGS)

# canalyze.c:	canalyze.pyx
# 	$(CYTHON) canalyze.pyx -X language_level=3

# -include $(OBJS:.o=.d)
# -include canalyze.d

# %.o: %.c
# 	$(CC) $(CPPFLAGS) -c $(CFLAGS) $*.c -o $*.o
# 	$(CC) $(CPPFLAGS) -MM $*.c > $*.d

# .PHONY : clean
# clean:
# 	-rm *.so *.o *.d canalyze.c
