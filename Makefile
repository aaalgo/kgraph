CC=g++ 
ARCH=-msse2
OPT=-O3
OPENMP=-fopenmp
VERSION=$(shell git describe --always)
BUILD_INFO=-DKGRAPH_VERSION=\"$(VERSION)\" -DKGRAPH_BUILD_ID=\"$(BUILD_ID)\" -DKGRAPH_BUILD_NUMBER=\"$(BUILD_NUMBER)\"
CXXFLAGS+=$(BUILD_INFO) -fPIC -Wall -g -std=c++11 -I. $(OPENMP) $(OPT) $(ARCH) 
LDFLAGS+=-static $(OPENMP)
LDLIBS+=-lboost_timer -lboost_chrono -lboost_system -lboost_program_options -lgomp -lm -lrt
FLANN_LIBS+=-lflann_cpp_s -lflann_s
NABO_LIBS+=-lnabo

.PHONY:	all python clean release deps-ubuntu flann

COMMON=kgraph.o metric.o
HEADERS=kgraph.h kgraph-data.h 
PROGS=index search prune split fvec2lshkit 
EXTRA_PROGS=test 
FLANN_PROGS=flann_index flann_search
NABO_PROGS=nabo_search

all:	libkgraph.a libkgraph.so $(PROGS) python $(EXTRA_PROGS)
	echo $(BUILD_INFO)

flann:	$(FANN_PROGS)

deps-ubuntu:
	apt-get install -y libboost-timer-dev libboost-chrono-dev libboost-program-options-dev libboost-system-dev libboost-python-dev libflann-dev python-numpy

RELEASE=kgraph-release
RELEASE_SRC=Makefile LICENSE kgraph.h kgraph-data.h index.cpp prune.cpp search.cpp flann_index.cpp flann_search.cpp split.cpp fvec2lshkit.cpp
RELEASE_BIN=libkgraph.a libkgraph.so $(PROGS) #$(FLANN_PROGS)

python:
	make -C python

install:	
	mkdir -p /usr/local/bin /usr/local/lib /usr/local/include
	cp libkgraph.so /usr/local/lib
	cp $(HEADERS) /usr/local/include
	make -C python install
	ldconfig

release:	all
	rm -rf $(RELEASE)
	mkdir $(RELEASE)
	cp $(RELEASE_SRC) $(RELEASE)
	cp Makefile.sdk $(RELEASE)/Makefile
	mkdir $(RELEASE)/bin
	cp $(RELEASE_BIN) $(RELEASE)/bin
	cp -r python $(RELEASE)/
	#tar zcf $(RELEASE).tar.gz $(RELEASE)

$(PROGS) $(EXTRA_PROGS):	%:	%.cpp $(HEADERS) $(COMMON)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $*.cpp $(COMMON) $(LDLIBS)

$(FLANN_PROGS):	%:	%.cpp $(HEADERS) $(COMMON)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $*.cpp $(COMMON) $(FLANN_LIBS) $(LDLIBS)

$(NABO_PROGS):	%:	%.cpp $(HEADERS) $(COMMON)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $*.cpp $(COMMON) $(NABO_LIBS) $(LDLIBS)

libkgraph.so:	$(COMMON)
	$(CXX) -shared -o $@ $^ $(LDLIBS)

libkgraph.a:	$(COMMON)
	ar rvs $@ $^

%.o:	%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $*.cpp 

clean:
	rm -f $(PROGS) *.o

