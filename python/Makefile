PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
CXXFLAGS = -fPIC -std=c++11 -I$(PYTHON_INCLUDE) -I..
LDFLAGS = -L..
LDLIBS = -lkgraph -lboost_python -lpython$(PYTHON_VERSION)

.PHONY:	clean

pykgraph.so:	pykgraph.cpp
	g++ -shared -Wl,--export-dynamic -o pykgraph.so $(CXXFLAGS) pykgraph.cpp $(LDFLAGS) $(LDLIBS)

clean:
	rm pykgraph.so
