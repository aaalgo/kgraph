CC=g++ 

ARCH = -msse2
#ARCH = -march=corei7-avx
#OPT = -O3 -fprofile-arcs
OPT = -O3 
OPENMP = -fopenmp
CXXFLAGS += -g -std=c++11 -I. $(OPENMP) $(OPT) $(ARCH)
LDFLAGS += $(OPENMP) 
#CXXFLAGS += -std=c++11 -g  -Wall -static -I. -msse2
#LDLIBS += -lopencv_flann -lopencv_core -lboost_timer -lboost_chrono -lboost_system -lboost_program_options  -lpthread -lm -lz
LDLIBS += -lboost_timer -lboost_chrono -lboost_system -lboost_program_options -lm

.PHONY:	benchmark all clean

COMMON = kgraph.o
HEADERS = kgraph.h kgraph-matrix.h kgraph-util.h
PROGS = index search prune stat 

all:	$(PROGS)

benchmark:
	make -C benchmark

$(PROGS):	%:	%.cpp $(HEADERS) $(COMMON)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $*.cpp $(COMMON) $(LDLIBS)

%.o:	%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $*.cpp 

clean:
	rm -f $(PROGS)

