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

#ifdef USE_BLAS
extern "C" {
    enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
    enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
    void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A, 
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);
    void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A, 
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);
}
#endif

using namespace std;
namespace python = boost::python;

/// Oracle for matrix data.
/** DATA_TYPE can be Matrix or MatrixProxy,
* DIST_TYPE should be one class within the namespace kgraph.metric.
*/
namespace kgraph {

#ifdef USE_BLAS
    void blas_dot (MatrixProxy<float> const &p1, MatrixProxy<float> const &p2, Matrix<float> *r) {
        r->zero();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, p2.size(), p1.size(), p1.dim(),
                    -2.0, p1[0], p1.dim(), p2[0], p2.dim(), 0, (*r)[0], p1.dim());
    }

    void blas_dot (MatrixProxy<double> const &p1, MatrixProxy<double> const &p2, Matrix<double> *r) {
        r->zero();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, p2.size(), p1.size(), p1.dim(),
                    -2.0, p1[0], p1.dim(), p2[0], p2.dim(), 0, (*r)[0], p1.dim());
    }
#endif

template <typename DATA_TYPE>
class MatrixOracleL2SQR: public kgraph::IndexOracle {
    MatrixProxy<DATA_TYPE> proxy;
    vector<DATA_TYPE> n2;
public:
    class SearchOracle: public kgraph::SearchOracle {
        MatrixProxy<DATA_TYPE> proxy;
        DATA_TYPE const *proxy_n2;
        DATA_TYPE const *query;
        DATA_TYPE n2;
    public:
        SearchOracle (MatrixProxy<DATA_TYPE> const &p, DATA_TYPE const *p_n2, DATA_TYPE const *q): proxy(p), proxy_n2(p_n2), query(q) {
            n2 = kgraph::metric::l2sqr::norm2(q, proxy.dim());
        }
        virtual unsigned size () const {
            return proxy.size();
        }
        virtual float operator () (unsigned i) const {
            return proxy_n2[i] + n2 - 2 * kgraph::metric::l2sqr::dot(proxy[i], query, proxy.dim());
        }
    };
    template <typename MATRIX_TYPE>
    MatrixOracleL2SQR (MATRIX_TYPE const &m): proxy(m), n2(proxy.size()) {
        for (unsigned i = 0; i < proxy.size(); ++i) {
            n2[i] = kgraph::metric::l2sqr::norm2(proxy[i], proxy.dim());
        }
    }
    virtual unsigned size () const {
        return proxy.size();
    }
    virtual float operator () (unsigned i, unsigned j) const {
        return n2[i] + n2[j] - 2 * kgraph::metric::l2sqr::dot(proxy[i], proxy[j], proxy.dim());
    }
    SearchOracle query (DATA_TYPE const *query) const {
        return SearchOracle(proxy, &n2[0], query);
    }

#ifdef USE_BLAS
    template <typename MATRIX_TYPE>
    void blasQuery (MATRIX_TYPE const &q, unsigned K, float epsilon,
            kgraph::MatrixProxy<unsigned, 1> *ids, kgraph::MatrixProxy<float, 1> *dists) {

        MatrixProxy<DATA_TYPE> q_proxy(q);
        vector<DATA_TYPE> q_n2(q_proxy.size());
        for (unsigned i = 0; i < q_proxy.size(); ++i) {
            q_n2[i] = kgraph::metric::l2sqr::norm2(q_proxy[i], proxy.dim());
        }
        Matrix<DATA_TYPE> dot(q_proxy.size(), proxy.size());
        blas_dot(q_proxy, proxy, &dot);
        vector<float> dummy(K);
#pragma omp parallel
        {
            vector<pair<DATA_TYPE, unsigned>> rank(proxy.size());
#pragma omp for
            for (unsigned i = 0; i < q_proxy.size(); ++i) {
                DATA_TYPE *row = dot[i];
                DATA_TYPE qn2 = q_n2[i];
                for (unsigned j = 0; j < proxy.size(); ++j) {
                    rank[j] = std::make_pair(qn2 + n2[j] + row[j], j);
                }
                std::sort(rank.begin(), rank.end());
                unsigned j = 0;

                unsigned *pid = (*ids)[i];
                float *pdist = dists ? (*dists)[i] : &dummy[0];

                while ((j < K) && (rank[j].second <= epsilon)) {
                    pid[j] = rank[j].first;
                    pdist[j] = rank[j].second;
                    ++j;
                }
            }
        }
    }
#endif
};

}

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
        kgraph::MatrixOracleL2SQR<TYPE> oracle(dmatrix);
        index->build(oracle, params, NULL);
        hasIndex = true;
    }

    template <typename TYPE>
    python::object searchImpl (python::object const &data,
                               python::object const &query,
                               kgraph::KGraph::SearchParams params,
                               unsigned threads,
                               bool withDistance,
                               bool blas) {
        checkArray<TYPE>(data);
        checkArray<TYPE>(query);
        kgraph::MatrixProxy<TYPE> dmatrix(reinterpret_cast<PyArrayObject *>(data.ptr()));
        kgraph::MatrixProxy<TYPE> qmatrix(reinterpret_cast<PyArrayObject *>(query.ptr()));
        kgraph::MatrixOracleL2SQR<TYPE> oracle(dmatrix);
        npy_intp dims[] = {qmatrix.size(), params.K};
        PyObject *result =  PyArray_SimpleNew(2, dims, NPY_UINT32);
        PyObject *distance =  PyArray_SimpleNew(2, dims, NPY_FLOAT);
        kgraph::MatrixProxy<unsigned, 1> rmatrix(reinterpret_cast<PyArrayObject *>(result));
        kgraph::MatrixProxy<float, 1> distmatrix(reinterpret_cast<PyArrayObject *>(distance));
#ifdef _OPENMP
        if (threads) ::omp_set_num_threads(threads);
#endif
#ifdef USE_BLAS
        if (blas) {
            if (withDistance) {
                oracle.blasQuery(qmatrix, params.K, params.epsilon, &rmatrix, &distmatrix);
            }
            else {
                oracle.blasQuery(qmatrix, params.K, params.epsilon, &rmatrix, NULL);
            }
        }
        else
#endif
            if (hasIndex) {
#pragma omp parallel for 
            for (unsigned i = 0; i < qmatrix.size(); ++i) {
                if (withDistance) {
                    index->search(oracle.query(qmatrix[i]), params, const_cast<unsigned *>(rmatrix[i]), 
                                  const_cast<float *>(distmatrix[i]),NULL);
                }
                else {
                    index->search(oracle.query(qmatrix[i]), params, const_cast<unsigned *>(rmatrix[i]), NULL);
                }
            }
        }
        else {
#pragma omp parallel for 
            for (unsigned i = 0; i < qmatrix.size(); ++i) {
                if (withDistance) {
                    oracle.query(qmatrix[i]).search(params.K, params.epsilon, const_cast<unsigned *>(rmatrix[i]), 
                                 const_cast<float *>(distmatrix[i]));
                }
                else {
                    oracle.query(qmatrix[i]).search(params.K, params.epsilon, const_cast<unsigned *>(rmatrix[i]), NULL);
                }
            }
        }

        if (withDistance) {
            PyObject* tup = PyTuple_New(2);
            PyTuple_SetItem(tup,0,result);
            PyTuple_SetItem(tup,1,distance);
            return python::object(python::handle<>(tup));
        }
        else {
            return python::object(python::handle<>(result));
        }
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
                unsigned threads,
                bool withDistance,
                bool blas) {
        kgraph::KGraph::SearchParams params;
        params.K = K;
        params.P = P;
        params.M = M;
        params.T = T;
        PyArrayObject *pd = reinterpret_cast<PyArrayObject *>(data.ptr());
        PyArrayObject *pq = reinterpret_cast<PyArrayObject *>(query.ptr());
        if (pd->descr->type_num != pq->descr->type_num) throw runtime_error("data and query have different types");
        switch (pd->descr->type_num) {
            case NPY_FLOAT: return searchImpl<float>(data, query, params, threads, withDistance, blas);
            case NPY_DOUBLE: return searchImpl<double>(data, query, params, threads, withDistance, blas);
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
             python::arg("threads") = 0,
             python::arg("withDistance") = false,
             python::arg("blas") = false))
        ;
}

