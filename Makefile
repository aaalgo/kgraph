CC=g++ 

#ARCH = -msse2
ARCH = -march=corei7-avx
#OPT = -O3 -fprofile-arcs
OPT = -O3 
OPENMP = -fopenmp
CXXFLAGS += -g -std=c++11 -I. $(OPENMP) $(OPT) $(ARCH)
LDFLAGS += $(OPENMP) 
#CXXFLAGS += -std=c++11 -g  -Wall -static -I. -msse2
#LDLIBS += -lopencv_flann -lopencv_core -lboost_timer -lboost_chrono -lboost_system -lboost_program_options  -lpthread -lm -lz
LDLIBS += -lboost_timer -lboost_chrono -lboost_system -lboost_program_options -lm

.PHONY:	benchmark all clean

PROGS = index search 

all:	$(PROGS)

benchmark:
	make -C benchmark

index:	kgraph.cpp index.cpp

search:	kgraph.cpp search.cpp

clean:
	rm -f $(PROGS)

