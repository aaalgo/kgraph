PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
CXXFLAGS += -fPIC -std=c++11 -I$(PYTHON_INCLUDE) -I..
LDLIBS += -lboost_python -lpython$(PYTHON_VERSION)

pykgraph.so:	pykgraph.cpp
	g++ -shared -Wl,--export-dynamic -o pykgraph.so $(CXXFLAGS) pykgraph.cpp $(LDFLAGS) $(LDLIBS) -L/usr/lib/python$(PYTHON_VERSION)/config  
