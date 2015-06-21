CC = g++
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
CXXFLAGS = -g -fPIC -std=c++11 -I$(PYTHON_INCLUDE) -I..
LDFLAGS = -L../bin -L..
LDLIBS = -lkgraph -lboost_python -lpython$(PYTHON_VERSION)

PYTHON_DIR = /usr/local/lib/python$(PYTHON_VERSION)/dist-packages/

.PHONY:	clean install

pykgraph.so:	pykgraph.cpp
	$(CC) -shared -Wl,--export-dynamic -o pykgraph.so $(CXXFLAGS) pykgraph.cpp $(LDFLAGS) $(LDLIBS)

install:	pykgraph.so
	mkdir -p $(PYTHON_DIR)
	cp pykgraph.so $(PYTHON_DIR)

clean:
	rm pykgraph.so
