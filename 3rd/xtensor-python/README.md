# ![xtensor-python](docs/source/xtensor-python.svg)

[![Azure Pipelines](https://dev.azure.com/xtensor-stack/xtensor-stack/_apis/build/status/xtensor-stack.xtensor-python?branchName=master)](https://dev.azure.com/xtensor-stack/xtensor-stack/_build/latest?definitionId=7&branchName=master)
[![Appveyor](https://ci.appveyor.com/api/projects/status/4j2yd6k8o5xbimqf?svg=true)](https://ci.appveyor.com/project/xtensor-stack/xtensor-python)
[![Documentation](http://readthedocs.org/projects/xtensor-python/badge/?version=latest)](https://xtensor-python.readthedocs.io/en/latest/?badge=latest)
[![Join the Gitter Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/QuantStack/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Python bindings for the [xtensor](https://github.com/xtensor-stack/xtensor) C++ multi-dimensional array library.

 - `xtensor` is a C++ library for multi-dimensional arrays enabling numpy-style broadcasting and lazy computing.
 - `xtensor-python` enables inplace use of numpy arrays in C++ with all the benefits from `xtensor`

     - C++ universal function and broadcasting
     - STL - compliant APIs.
     - A broad coverage of numpy APIs (see [the numpy to xtensor cheat sheet](http://xtensor.readthedocs.io/en/latest/numpy.html)).

The Python bindings for `xtensor` are based on the [pybind11](https://github.com/pybind/pybind11/) C++ library, which enables seamless interoperability between C++ and Python.

## Installation

`xtensor-python` is a header-only library. We provide a package for the mamba (or conda) package manager.

```bash
mamba install -c conda-forge xtensor-python
```

## Documentation

To get started with using `xtensor-python`, check out the full documentation

http://xtensor-python.readthedocs.io/

## Usage

xtensor-python offers two container types wrapping numpy arrays inplace to provide an xtensor semantics

 - `pytensor`
 - `pyarray`.

Both containers enable the numpy-style APIs of xtensor (see [the numpy to xtensor cheat sheet](http://xtensor.readthedocs.io/en/latest/numpy.html)).

 - On the one hand, `pyarray` has a dynamic number of dimensions. Just like numpy arrays, it can be reshaped with a shape of a different length (and the new shape is reflected on the python side).

 - On the other hand `pytensor` has a compile time number of dimensions, specified with a template parameter. Shapes of `pytensor` instances are stack allocated, making `pytensor` a significantly faster expression than `pyarray`.

### Example 1: Use an algorithm of the C++ standard library on a numpy array inplace.

**C++ code**

```cpp
#include <numeric>                        // Standard library import for std::accumulate
#include <pybind11/pybind11.h>            // Pybind11 import to define Python bindings
#include <xtensor/xmath.hpp>              // xtensor import for the C++ universal functions
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>     // Numpy bindings

double sum_of_sines(xt::pyarray<double>& m)
{
    auto sines = xt::sin(m);  // sines does not actually hold values.
    return std::accumulate(sines.begin(), sines.end(), 0.0);
}

PYBIND11_MODULE(xtensor_python_test, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";

    m.def("sum_of_sines", sum_of_sines, "Sum the sines of the input values");
}
```

**Python Code**

```python
import numpy as np
import xtensor_python_test as xt

v = np.arange(15).reshape(3, 5)
s = xt.sum_of_sines(v)
print(s)
```

**Outputs**

```
1.2853996391883833
```

**Working example**

Get the working example here:

*   [`CMakeLists.txt`](docs/source/examples/readme_example_1/CMakeLists.txt)
*   [`main.cpp`](docs/source/examples/readme_example_1/main.cpp)
*   [`example.py`](docs/source/examples/readme_example_1/example.py)

### Example 2: Create a universal function from a C++ scalar function

**C++ code**

```cpp
#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyvectorize.hpp>
#include <numeric>
#include <cmath>

namespace py = pybind11;

double scalar_func(double i, double j)
{
    return std::sin(i) - std::cos(j);
}

PYBIND11_MODULE(xtensor_python_test, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";

    m.def("vectorized_func", xt::pyvectorize(scalar_func), "");
}
```

**Python Code**

```python
import numpy as np
import xtensor_python_test as xt

x = np.arange(15).reshape(3, 5)
y = [1, 2, 3, 4, 5]
z = xt.vectorized_func(x, y)
print(z)
```

**Outputs**

```
[[-0.540302,  1.257618,  1.89929 ,  0.794764, -1.040465],
 [-1.499227,  0.136731,  1.646979,  1.643002,  0.128456],
 [-1.084323, -0.583843,  0.45342 ,  1.073811,  0.706945]]
```

## Installation

We provide a package for the conda package manager.

```bash
conda install -c conda-forge xtensor-python
```

This will pull the dependencies to xtensor-python, that is `pybind11` and `xtensor`.

## Project cookiecutter

A template for a project making use of `xtensor-python` is available in the form of a cookiecutter [here](https://github.com/xtensor-stack/xtensor-python-cookiecutter).

This project is meant to help library authors get started with the xtensor python bindings.

It produces a project following the best practices for the packaging and distribution of Python extensions based on `xtensor-python`, including a `setup.py` file and a conda recipe.

## Building and Running the Tests

Testing `xtensor-python` requires `pytest`

  ``` bash
  py.test .
  ```

To pick up changes in `xtensor-python` while rebuilding, delete the `build/` directory.

## Building the HTML Documentation

`xtensor-python`'s documentation is built with three tools

 - [doxygen](http://www.doxygen.org)
 - [sphinx](http://www.sphinx-doc.org)
 - [breathe](https://breathe.readthedocs.io)

While doxygen must be installed separately, you can install breathe by typing

```bash
pip install breathe
```

Breathe can also be installed with `conda`

```bash
conda install -c conda-forge breathe
```

Finally, build the documentation with

```bash
make html
```

from the `docs` subdirectory.

## Dependencies on `xtensor` and `pybind11`

`xtensor-python` depends on the `xtensor` and `pybind11` libraries

| `xtensor-python` | `xtensor` |  `pybind11`      |
|------------------|-----------|------------------|
| master           |  ^0.24.0  | ~2.4.3           |
| 0.26.1           |  ^0.24.0  | ~2.4.3           |
| 0.26.0           |  ^0.24.0  | ~2.4.3           |
| 0.25.3           |  ^0.23.0  | ~2.4.3           |
| 0.25.2           |  ^0.23.0  | ~2.4.3           |
| 0.25.1           |  ^0.23.0  | ~2.4.3           |
| 0.25.0           |  ^0.23.0  | ~2.4.3           |
| 0.24.1           |  ^0.21.2  | ~2.4.3           |
| 0.24.0           |  ^0.21.1  | ~2.4.3           |

These dependencies are automatically resolved when using the conda package manager.

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

This software is licensed under the BSD-3-Clause license. See the [LICENSE](LICENSE) file for details.
