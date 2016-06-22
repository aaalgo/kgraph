Oracles for Common Tasks
========================

KGraph provides the following efficient oracle implementation for
common tasks.  The user is encouraged to use the provided oracles
when they meet the demand.

# Rows of Matrices
In a very common scenario, objects are stored as rows of a matrix
in the row-major order, and L2 distance is used as the similarity
function.  KGraph provides the class Matrix for user to create
such a dataset, and the MatrixProxy class for user to directly use
data managed by a OpenCV, Numpy or a FLANN matrix.

Note that the matrix proxy does not copy memory, so the original
OpenCV or NumPy matrix must live throughput the life time of
the matrix proxy.

## OpenCV Matrix
```cpp
#include <kgraph.h>
#include <opencv2/opencv.hpp>
#include <kgraph-data.h> // must follow opencv

cv::Mat data(rows, cols, CV_32FC1);

typedef kgraph::MatrixOracle<float, kgraph::metric::l2sqr
        > MyOracle;

kgraph::MatrixProxy<float> proxy(data);

MyOracle oracle(proxy);	// defined as above
index->build(oracle, ...);

// MyOracle::query(float const *) returns a search oracle
index->search(oracle.query(data.ptr<float const>(i)), ...);
```

## NumPy Matrix
```cpp
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <kgraph.h>
#include <kgraph-data.h>

npy_intp dims[] = {rows, cols};
PyArrayObject *data = PyArray_SimpleNew(2, dims, NPY_FLOAT);
kgraph::MatrixProxy<float> proxy(data);

MyOracle oracle(proxy);	// defined as above 

index->build(oracle, ...);
float const *query;
index->search(oracle.query(query), ...);
```

## KGraph's Native Matrix
```cpp
#include <kgraph.h>
#include <kgraph-data.h>

kgraph::Matrix<float> data(rows, cols);	// kgraph native matrix

data.size();   // returns # rows
data.dim();    // returns # cols
float *row = data[i];       // pointer to i-th row

// save and load matrix in LSHKIT format.
data.save_lshkit("path");
data.load_lshkit("path");

MyOracle oracle(data);	    // MyOracle defined as above

index->build(oracle, ...);

float const *query;	// pointer to query vector

// oracle.query(query) returns a search oracle.
index->search(oracle.query(query), ...)
```


# Entries of Vectors

If data objects are stored in std::vector, then VectorOracle can come in handy.
```cpp
#include <kgraph.h>

vector<MyType> data;
auto fn = [](MyType const &a, MyType const &b) {
           compute and return similarity between a and b;
          };

typedef kgraph::VectorOracle<vector<MyType>, MyType> MyOracle;
MyOracle oracle(data, fn);

index->build(oracle, ...);

MyType query;

index->search(oracle.query(query), ...);

```
