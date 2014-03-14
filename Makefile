CC=g++ 

ARCH = -msse2
ARCH = #-march=corei7-avx
#OPT = -O3 -fprofile-arcs
OPT = -O3 
OPENMP = -fopenmp
CXXFLAGS += -fPIC -Wall -g -std=c++11 -I. $(OPENMP) $(OPT) $(ARCH) 
LDFLAGS += $(OPENMP) 
#CXXFLAGS += -std=c++11 -g  -Wall -static -I. -msse2
#LDLIBS += -lopencv_flann -lopencv_core -lboost_timer -lboost_chrono -lboost_system -lboost_program_options  -lpthread -lm -lz
LDLIBS += -lboost_timer -lboost_chrono -lboost_system -lboost_program_options -lm -lrt

.PHONY:	benchmark all clean release

COMMON = kgraph.o metric.o
HEADERS = kgraph.h kgraph-data.h 
PROGS = index search #prune #stat 

all:	libkgraph.so $(PROGS)

RELEASE=kgraph-1.0-x86_64

release:	libkgraph.so
	rm -rf $(RELEASE)
	mkdir $(RELEASE)
	cp kgraph.h kgraph-data.h libkgraph.so index.cpp search.cpp $(RELEASE)
	cp Makefile.sdk $(RELEASE)/Makefile
	mkdir $(RELEASE)/benchmark
	cp benchmark/flann_index.cpp benchmark/flann_search.cpp benchmark/split.cpp benchmark/lshkit2fvec.cpp benchmark/fvec2lshkit.cpp benchmark/Makefile $(RELEASE)/benchmark
	tar zcf $(RELEASE).tar.gz $(RELEASE)

benchmark:
	make -C benchmark

$(PROGS):	%:	%.cpp $(HEADERS) $(COMMON)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $*.cpp $(COMMON) $(LDLIBS)

libkgraph.so:	$(COMMON)
	$(CXX) -shared -o $@ $^ $(LDLIBS)

%.o:	%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $*.cpp 

clean:
	rm -f $(PROGS) *.o

