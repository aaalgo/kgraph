#ifdef _OPENMP
#include <omp.h>
#endif
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <iostream>
#include <boost/assert.hpp>
#include <boost/python.hpp>
#include <kgraph.h>
#include <kgraph-data.h>

using namespace std;
namespace python = boost::python;

class KGraph {
    kgraph::KGraph *index;
    bool hasIndex;

    template <typename TYPE>
    void checkArray (python::object const &data) {
        PyArrayObject *array = reinterpret_cast<PyArrayObject *>(data.ptr());
        BOOST_VERIFY(array->nd == 2);
        //cerr << "dims: " << array->dimensions[0] << ' ' << array->dimensions[1] << endl;
        //cerr << "stride: " << array->strides[0] << ' ' << array->strides[1] << endl;
        PyArray_Descr *descr = array->descr;
        /*
        BOOST_VERIFY(descr->type_num == NPY_FLOAT);
        */
        BOOST_VERIFY(descr->elsize == sizeof(TYPE));
        //cerr << "type: " << descr->type_num << endl;
        //cerr << "size: " << descr->elsize << endl;
        //cerr << "alignment: " << descr->alignment << endl;
    }

    template <typename TYPE>
    void buildImpl (python::object const &data,
                               kgraph::KGraph::IndexParams params) {
        checkArray<TYPE>(data);
        kgraph::MatrixProxy<TYPE> dmatrix(reinterpret_cast<PyArrayObject *>(data.ptr()));
        kgraph::MatrixOracle<TYPE, kgraph::metric::l2sqr> oracle(dmatrix);
        index->build(oracle, params, NULL);
        hasIndex = true;
    }

    template <typename TYPE>
    python::object searchImpl (python::object const &data,
                               python::object const &query,
                               kgraph::KGraph::SearchParams params,
                               unsigned threads) {
        checkArray<TYPE>(data);
        checkArray<TYPE>(query);
        kgraph::MatrixProxy<TYPE> dmatrix(reinterpret_cast<PyArrayObject *>(data.ptr()));
        kgraph::MatrixProxy<TYPE> qmatrix(reinterpret_cast<PyArrayObject *>(query.ptr()));
        kgraph::MatrixOracle<TYPE, kgraph::metric::l2sqr> oracle(dmatrix);
        npy_intp dims[] = {qmatrix.size(), params.K};
        PyObject *result =  PyArray_SimpleNew(2, dims, NPY_UINT32);
        kgraph::MatrixProxy<unsigned, 1> rmatrix(reinterpret_cast<PyArrayObject *>(result));
#ifdef _OPENMP
        if (threads) ::omp_set_max_threads(threads);
#endif
        if (hasIndex) {
#pragma omp parallel for reduction(+:cost)
            for (unsigned i = 0; i < qmatrix.size(); ++i) {
                index->search(oracle.query(qmatrix[i]), params, const_cast<unsigned *>(rmatrix[i]), NULL);
            }
        }
        else {
#pragma omp parallel for reduction(+:cost)
            for (unsigned i = 0; i < qmatrix.size(); ++i) {
                oracle.query(qmatrix[i]).search(params.K, params.epsilon, const_cast<unsigned *>(rmatrix[i]));
            }
        }
        return python::object(python::handle<>(result));
    }

public:
    KGraph (): index(kgraph::KGraph::create()), hasIndex(false) {
        if (!index) throw runtime_error("error creating kgraph instance");
    }
    ~KGraph () {
        if (index) delete index;
    }
    void load (char const *path) {
        index->load(path);
        hasIndex = true;
    }
    void save (char const *path) const {
        index->save(path);
    }
    void build (python::object const &data,
               unsigned iterations,
               unsigned L,
               unsigned K,
               unsigned S,
               unsigned controls,
               float delta,
               float recall,
               unsigned prune) {
        kgraph::KGraph::IndexParams params;
        params.iterations = iterations;
        params.L = L;
        params.K = K;
        params.S = S;
        params.controls = controls;
        params.delta = delta;
        params.recall = recall;
        params.prune = prune;
        PyArrayObject *pd = reinterpret_cast<PyArrayObject *>(data.ptr());
        switch (pd->descr->type_num) {
            case NPY_FLOAT: buildImpl<float>(data, params); return;
            case NPY_DOUBLE: buildImpl<double>(data, params); return;
        }
        throw runtime_error("data type not supported.");
    }
    python::object search (python::object const &data,
                           python::object const &query,
                unsigned K,
                unsigned P,
                unsigned M,
                unsigned T,
                unsigned threads) {
        kgraph::KGraph::SearchParams params;
        params.K = K;
        params.P = P;
        params.M = M;
        params.T = T;
        PyArrayObject *pd = reinterpret_cast<PyArrayObject *>(data.ptr());
        PyArrayObject *pq = reinterpret_cast<PyArrayObject *>(data.ptr());
        if (pd->descr->type_num != pq->descr->type_num) throw runtime_error("data and query have different types");
        switch (pd->descr->type_num) {
            case NPY_FLOAT: return searchImpl<float>(data, query, params, threads);
            case NPY_DOUBLE: return searchImpl<double>(data, query, params, threads);
        }
        throw runtime_error("data type not supported.");
        return python::object();
    }
};

BOOST_PYTHON_MODULE(pykgraph)
{
    import_array();
    python::numeric::array::set_module_and_type("numpy", "ndarray");
    python::class_<KGraph>("KGraph")
        .def("load", &KGraph::load)
        .def("save", &KGraph::save)
        .def("build", &KGraph::build,
             (python::arg("data"),
              python::arg("iterations") = kgraph::default_iterations,
              python::arg("L") = kgraph::default_L,
              python::arg("K") = kgraph::default_K,
              python::arg("S") = kgraph::default_S,
              python::arg("controls") = kgraph::default_controls,
              python::arg("delta") = kgraph::default_delta,
              python::arg("recall") = kgraph::default_recall,
              python::arg("prune") = kgraph::default_prune))
        .def("search", &KGraph::search,
            (python::arg("data"),
             python::arg("query"),
             python::arg("K") = kgraph::default_K,
             python::arg("P") = kgraph::default_P,
             python::arg("M") = kgraph::default_M,
             python::arg("T") = kgraph::default_T,
             python::arg("threads") = 0))
        ;
}

