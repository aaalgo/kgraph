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
and Angular distances on rows of NumPy matrices.



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
from numpy import random
import pykgraph

dataset = random.rand(1000000, 16)
query = random.rand(1000, 16)

index = pykgraph.KGraph(dataset, 'euclidean')
index.build(reverse=-1)                        #
index.save("index_file");
# load with index.load("index_file");

knn = index.search(query, K=10)                       # this uses all CPU threads
knn = index.search(query, K=10, threads=1)            # one thread, slower
knn = index.search(query, K=1000, P=100)              # search for 1000-nn, no need to recompute index.
```

Both index.build and index.search supports a number of optional keywords
arguments to fine tune the performance.  The default values should work
reasonably well for many datasets.  One exception is that reverse=-1 should be
added if the purpose of building index is to speedup search, which is the
typical case, rather than to obtain the k-NN graph itself.

Two precautions should be taken:
* Although matrices of both float32 and float64 are supported, the latter is not optimized.  It is recommened that
matrices be converted to float32 before being passed into kgraph.
* The dimension (columns of matrices) should be a multiple of 4.  If not, zeros must be padded.

For performance considerations, the Python API does not support user-defined similarity function,
as the callback function is invoked in such a high frequency that, if written in Python, speedup will
inevitably be brought down.  For the full generality, the C++ API should be used.

# C++ Quick Start

The KGraph C++ API is based on two central concepts: the index oracle and the search oracle.
(Oracle is a fancy way of calling a user-defined callback function that behaves like a black box.)
KGraph works solely with object IDs from 0 to N-1, and relies on the oracles to map the IDs to
actual data objects and computes the similarity.  To use KGraph, the user has to extend the following
two abstract classes

```cpp
    class IndexOracle {
    public:
        // returns size of dataset
        virtual unsigned size () const = 0;
        // computes similarity of object i and j
        virtual float operator () (unsigned i, unsigned j) const = 0;
    };

    class SearchOracle {
    public:
        /// Returns the size of the dataset.
        virtual unsigned size () const = 0;
	/// Computes similarity of query and object i.
        virtual float operator () (unsigned i) const = 0;
    };
```



http://www.kgraph.org/

