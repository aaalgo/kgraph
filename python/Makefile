CC = g++
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
CXXFLAGS = -fopenmp -g -fPIC -std=c++11 -I$(PYTHON_INCLUDE) -I..
LDFLAGS = -L../bin 
LDLIBS = ../libkgraph.a -lboost_python -lboost_timer -lboost_chrono -lpython$(PYTHON_VERSION) -lgomp

ifeq (${BLAS}, false)
else
	CXXFLAGS += -DUSE_BLAS=1
	LDLIBS += /usr/lib/openblas-base/libblas.a
endif

PYTHON_DIR = /usr/local/lib/python$(PYTHON_VERSION)/dist-packages/

.PHONY:	clean install

pykgraph.so:	pykgraph.cpp
	$(CC) -shared -Wl,--export-dynamic -o pykgraph.so $(CXXFLAGS) pykgraph.cpp $(LDFLAGS) $(LDLIBS)

install:	pykgraph.so
	mkdir -p $(PYTHON_DIR)
	cp pykgraph.so $(PYTHON_DIR)

clean:
	rm pykgraph.so
