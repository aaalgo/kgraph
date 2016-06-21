KGraph: A Library for Approximate Nearest Neighbor Search
=========================================================

# Introduction

KGraph is a library for k-nearest neighbor (k-NN) graph construction and
online k-NN search using a k-NN Graph as index.  KGraph implements 
heuristic algorithms that are extremely generic and fast:
* KGraph works on abstract objects.  The only assumption it makes is
that a similarity score can be computed on any pair of objects, with
a user-provided function.
* KGraph is among the fastest of libraries for k-NN search.

For best generality, the C++ API should be used.  A python wrapper
is provided under the module name pykgraph, which supports Euclidean
and Angular distances on top of NumPy matrices.

# Building and Installation

KGraph depends on a recent version of GCC with C++11 support, and the 
Boost library.  A Makefile is provided which produces libkgraph.a,
libkgraph.so and a few utility functions.  To install, copy *.h to
files to /usr/local/include and libkgraph.* to /usr/local/lib.

The Python API can be installed with
```
python setup.py install
```

# Python Quick Start

```python
>>> from numpy import random
>>> import pykgraph
>>> dataset = random.rand(1000000, 16)
>>> query = random.rand(1000, 16)
>>> index = pykgraph.KGraph(dataset, 'euclidean')
>>> index.build()
>>> index.save("index_file");
>>> # load with index.load("index_file");
......
>>> result = index.search(query, K=10)                        # this uses all CPU threads, set prune=1 to make index smaller (no accuracy loss)
>>> result = index.search(query, K=10, threads=1)             # one thread, slower
>>> result.shape             # the k-nn IDs.
>>> result = index.search(query, K=1000, P=100)                # search for 1000-nn, no need to recompute index.
(1000, 10) 
```
# C++ Quick Start




http://www.kgraph.org/
