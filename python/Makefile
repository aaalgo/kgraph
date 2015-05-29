PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
CXXFLAGS = -g -fPIC -std=c++11 -I$(PYTHON_INCLUDE) -I..
LDFLAGS = -L../bin
LDLIBS = -lkgraph -lboost_python -lpython$(PYTHON_VERSION)

.PHONY:	clean

pykgraph.so:	pykgraph.cpp
	g++ -shared -Wl,--export-dynamic -o pykgraph.so $(CXXFLAGS) pykgraph.cpp $(LDFLAGS) $(LDLIBS)

clean:
	rm pykgraph.so
