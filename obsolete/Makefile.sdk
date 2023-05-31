CC=g++ 

OPT = -O3 
OPENMP = -fopenmp
CXXFLAGS += -Wall -g -std=c++11 -I. $(OPENMP) $(OPT) $(ARCH)
LDFLAGS += $(OPENMP) 
LDLIBS += -lboost_timer -lboost_chrono -lboost_system -lboost_program_options -lm -lrt
FLANN_LIBS+=-lflann_cpp_s -lflann_s

.PHONY:	all clean

PROGS=index search split prune fvec2lshkit
FLANN_PROGS=flann_index flann_search


all:	$(PROGS) $(FLANN_PROGS)

$(PROGS):	%:	%.cpp 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $*.cpp bin/libkgraph.so $(LDLIBS)

$(FLANN_PROGS):	%:	%.cpp 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $*.cpp bin/libkgraph.so $(FLANN_LIBS) $(LDLIBS)

%.o:	%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $*.cpp 

clean:
	rm -f $(PROGS) *.o

