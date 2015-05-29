CC=g++ 
ARCH=-msse2
OPT=-O3
OPENMP=-fopenmp
CXXFLAGS+=-g -std=c++11 -I.. $(OPENMP) $(OPT) $(ARCH)
LDFLAGS+=$(OPENMP) 
LDLIBS+=-lflann_cpp_s -lflann_s -lboost_timer -lboost_chrono -lboost_system -lboost_program_options -lm -lrt

.PHONY:	all clean

PROGS = flann_index flann_search split lshkit2fvec fvec2lshkit

all:	$(PROGS)

clean:
	rm -f $(PROGS)

