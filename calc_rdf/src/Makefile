# MAKEFILE
include Makefile.inc

#PYTHONINC = -I/Users/fanc/miniconda3/include/python3.9
#NUMPYINC = -I/Users/fanc/miniconda3/lib/python3.9/site-packages/numpy/core/include

CC = gcc
CYTHON = cython
CPPFLAGS = $(PYTHONINC) $(NUMPYINC)
CFLAGS = -fPIC -O3 -g -Wall
LDFLAGS = -L`pwd` -Wl,-rpath=`pwd`
#LDFLAGS = -L$(PWD) -Wl, "-rpath" "$(PWD)"
LIBS = -lm

all:	canalyze.so

canalyze.so:	canalyze.o
	$(CC) -shared -o canalyze.so canalyze.o $(LDFLAGS)

canalyze.c:	canalyze.pyx
	$(CYTHON) canalyze.pyx -X language_level=3

-include $(OBJS:.o=.d)
-include canalyze.d

%.o: %.c
	$(CC) $(CPPFLAGS) -c $(CFLAGS) $*.c -o $*.o
	$(CC) $(CPPFLAGS) -MM $*.c > $*.d

.PHONY : clean
clean:
	-rm *.so *.o *.d canalyze.c
