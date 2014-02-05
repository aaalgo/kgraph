CC=g++ 

#CXXFLAGS += -std=c++11  -g -O3 -Wall -static -I. -msse2
ARCH = -msse2 #-march=corei7-avx
OPENMP = -fopenmp
LDFLAGS += $(OPENMP) -L./opencv/lib
CXXFLAGS += -O3 -I. -I./opencv/include -std=c++11 $(OPENMP) -g -Wall -static -I. $(ARCH) -I../nndes
#CXXFLAGS += -std=c++11 -g  -Wall -static -I. -msse2
#LDLIBS += -lopencv_flann -lopencv_core -lboost_timer -lboost_chrono -lboost_system -lboost_program_options  -lpthread -lm -lz
LDLIBS += -lflann_cpp_s -lflann_s -lboost_timer -lboost_chrono -lboost_system -lboost_program_options  -lpthread -lm -lz

.PHONY:	all clean

PROGS= index search flann_search flann_index # opencv_index opencv_search split

all:	$(PROGS)


index:	kgraph.cpp index.cpp

search:	kgraph.cpp search.cpp

flann_search:	kgraph.cpp flann_search.cpp

clean:
	rm -f $(PROGS)

