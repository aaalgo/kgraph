CC=g++ 
ARCH = -msse2
OPT = -O3
OPENMP = -fopenmp
CXXFLAGS += -g -std=c++11 -I.. $(OPENMP) $(OPT) $(ARCH)
LDFLAGS += $(OPENMP) 
LDLIBS += -lflann_cpp_s -lflann_s -lboost_timer -lboost_chrono -lboost_system -lboost_program_options -lm

.PHONY:	all clean

PROGS = flann_index flann_search split lshkit2fvec fvec2lshkit

all:	$(PROGS)


#flann_index:	kgraph.cpp index.cpp

#flann_search:	kgraph.cpp search.cpp

clean:
	rm -f $(PROGS)

