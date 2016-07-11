#ifdef _OPENMP
#include <omp.h>
#endif
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <limits>
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

static void blas_prod (kgraph::MatrixProxy<float> const &p1, float const *p2, int n2, int l2, kgraph::Matrix<float> *r) {
    r->zero();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, p1.size(), n2, p1.dim(),
                1.0, p1[0], p1[1]-p1[0], p2, l2, 0, (*r)[0], (*r)[1]-(*r)[0]);
}

static void blas_prod (kgraph::MatrixProxy<double> const &p1, double const *p2, int n2, int l2, kgraph::Matrix<double> *r) {
    r->zero();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, p1.size(), n2, p1.dim(),
                1.0, p1[0], p1[1]-p1[0], p2, l2, 0, (*r)[0], (*r)[1]-(*r)[0]);
}

static unsigned constexpr MIN_BLAS_BATCH_SIZE = 16;
#endif

using namespace std;
namespace python = boost::python;

/// Oracle for matrix data.
/** DATA_TYPE can be Matrix or MatrixProxy,
* DIST_TYPE should be one class within the namespace kgraph.metric.
*/
namespace {

    PyObject *load_lshkit (string const &path, PyObject *type) {
        static const unsigned LSHKIT_HEADER = 3;
        std::ifstream is(path.c_str(), std::ios::binary);
        unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
        is.read((char *)header, sizeof header);
        if (!is) throw kgraph::io_error(path);
        unsigned elsize = header[0];
        unsigned N = header[1];
        unsigned D = header[2];
        npy_intp dims[2] = {N, D};
        PyArray_Descr *descr = PyArray_DescrFromTypeObject(type);
        if (!(descr->elsize == elsize)) throw kgraph::io_error(path);
        PyObject *nd = PyArray_SimpleNewFromDescr(2, &dims[0], descr);
        if ((!nd) || (PyArray_ITEMSIZE(nd) != elsize)) throw kgraph::runtime_error("error creating numpy matrix");
        float *buf = reinterpret_cast<float*>(PyArray_DATA(nd));
        is.read((char *)&buf[0], 1LL * N * D * elsize);
        if (!is) throw kgraph::io_error(path);
        return nd;
    }

    struct EuclideanLike {
        template <typename T>
        static float norm (T const *t1, unsigned dim) {
            return kgraph::metric::l2sqr::norm2(t1, dim)/2;
        }
        static float dist (float dot, float n1, float n2) {
            return n1 + n2 - dot;
        }
    };
    
    struct AngularLike {
        template <typename T>
        static float norm (T const *t1, unsigned dim) {
            return std::sqrt(kgraph::metric::l2sqr::norm2(t1, dim));
        }
        static float dist (float dot, float n1, float n2) {
            return -dot/(n1 * n2 + std::numeric_limits<float>::epsilon());
        }
    };

    template <typename DATA_TYPE, typename METRIC>
    class NDArrayOracle: public kgraph::IndexOracle {
        PyArrayObject *handle; // we need to keep reference
        kgraph::MatrixProxy<DATA_TYPE> proxy;
        vector<float> n2;
    public:
        class SearchOracle: public kgraph::SearchOracle {
            kgraph::MatrixProxy<DATA_TYPE> proxy;
            float const *proxy_n2;
            DATA_TYPE const *query;
            float n2;
        public:
            SearchOracle (kgraph::MatrixProxy<DATA_TYPE> const &p, float const *p_n2, DATA_TYPE const *q)
                : proxy(p), proxy_n2(p_n2), query(q) {
                n2 = METRIC::norm(q, proxy.dim());
            }
            virtual unsigned size () const {
                return proxy.size();
            }
            virtual float operator () (unsigned i) const {
                return METRIC::dist(kgraph::metric::l2sqr::dot(proxy[i], query, proxy.dim()), proxy_n2[i], n2);
            }
        };

        NDArrayOracle (PyArrayObject *data): handle(data), proxy(data), n2(proxy.size()) {
            Py_INCREF(handle);
            for (unsigned i = 0; i < proxy.size(); ++i) {
                n2[i] = METRIC::norm(proxy[i], proxy.dim());
            }
        }

        ~NDArrayOracle () {
            Py_DECREF(handle);
        }

        virtual unsigned size () const {
            return proxy.size();
        }

        virtual float operator () (unsigned i, unsigned j) const {
            return METRIC::dist(kgraph::metric::l2sqr::dot(proxy[i], proxy[j], proxy.dim()), n2[i], n2[j]);
        }

        SearchOracle query (DATA_TYPE const *query) const {
            return SearchOracle(proxy, &n2[0], query);
        }

#ifdef USE_BLAS
        template <typename MATRIX_TYPE>
        void blasQuery (MATRIX_TYPE const &q_proxy,
                unsigned K, float epsilon,
                kgraph::MatrixProxy<unsigned, 1> *ids, kgraph::MatrixProxy<float, 1> *dists) const {
            BOOST_VERIFY(dists);
            static unsigned constexpr BLOCK_SIZE = 1024;
            unsigned block = BLOCK_SIZE;
            if (block < K) {
                block = K;
            }
            if (block >= proxy.size()) {
                block = proxy.size();
            }
            BOOST_VERIFY(block >= K);
            kgraph::Matrix<DATA_TYPE> dot(q_proxy.size(), block);

            vector<float> qn2s(q_proxy.size());
            for (unsigned i = 0; i < q_proxy.size(); ++i) {
                qn2s[i] = METRIC::norm(q_proxy[i], q_proxy.dim());
            }

            unsigned begin = 0; // divide all data into blocks
            while (begin < proxy.size()) {
                unsigned end = begin + block;
                if (end > proxy.size()) {
                    end = proxy.size();
                }
                blas_prod(q_proxy, proxy[begin], end-begin, proxy[1]-proxy[0], &dot);
                // do one block
                if (begin == 0) {
                    // first block
#pragma omp parallel for
                    for (unsigned i = 0; i < q_proxy.size(); ++i) {
                        DATA_TYPE *row = dot[i];
                        DATA_TYPE qn2 = qn2s[i];
                        vector<pair<DATA_TYPE, unsigned>> rank(end-begin);
                        for (unsigned j = 0; j < rank.size(); ++j) {
                            rank[j] = std::make_pair(METRIC::dist(qn2,n2[j],row[j]), j);
                        }
                        std::sort(rank.begin(), rank.end());
                        unsigned *pid = (*ids)[i];
                        float *pdist = (*dists)[i];
                        for (unsigned j = 0; j < K; ++j) {
                            pid[j] = rank[j].second;
                            pdist[j] = rank[j].first;
                        }
                    }
                }
                else { // subsequent blocks, using inserting instead of sort
#pragma omp parallel for
                    for (unsigned i = 0; i < q_proxy.size(); ++i) {
                        DATA_TYPE *row = dot[i];
                        DATA_TYPE qn2 = qn2s[i];
                        unsigned *pid = (*ids)[i];
                        float *pdist = (*dists)[i];
                        for (unsigned j = 0; j < end-begin; ++j) {
                            // insert
                            unsigned id = begin + j;
                            float d = METRIC::dist(qn2, n2[id], row[j]);
                            unsigned c = K-1;
                            if (d >= pdist[c]) continue;
                            while ((c > 0) && (d < pdist[c-1])) {
                                pid[c] = pid[c-1];
                                pdist[c] = pdist[c-1];
                                --c;
                            }
                            pid[c] = id;
                            pdist[c] = d;
                        }
                    }
                }
                begin = end;
            }
        }
#endif
    };

    struct IndexParams: public kgraph::KGraph::IndexParams {
    };

    struct SearchParams: public kgraph::KGraph::SearchParams {
        unsigned threads;
        bool withDistance;
        bool blas;
        SearchParams (): threads(0), withDistance(false), blas(false)  {
        }
    };

    void check_array (PyArrayObject *array, float) {
        do {
            if (array->nd != 2) break;
            if (array->dimensions[1] % 4) break;
            if (array->descr->type_num != NPY_FLOAT) break;
            return;
        } while (false);
        throw kgraph::invalid_argument("bad array type");
    }

    void check_array (PyArrayObject *array, double) {
        do {
            if (array->nd != 2) break;
            if (array->dimensions[1] % 4) break;
            if (array->descr->type_num != NPY_DOUBLE) break;
            return;
        } while (false);
        throw kgraph::invalid_argument("bad array type");
    }

    class ImplBase {
    protected:
        kgraph::KGraph *index;
        bool hasIndex;
    public:
        ImplBase (): index(kgraph::KGraph::create()), hasIndex(false) {
            if (!index) throw runtime_error("error creating kgraph instance");
        }

        virtual ~ImplBase () {
            delete index;
        }

        virtual void build (IndexParams params) = 0;
        virtual python::object search (PyArrayObject *, SearchParams) const = 0;

        void load (char const *path) {
            index->load(path);
            hasIndex = true;
        }
        void save (char const *path) const {
            index->save(path);
        }
    };

    template <typename DATA_TYPE, typename METRIC_TYPE>
    class Impl: public ImplBase {
        NDArrayOracle<DATA_TYPE, METRIC_TYPE> oracle; 
    public:
        Impl (PyArrayObject *data): oracle(data) {
            check_array(data, DATA_TYPE());
        }

        void build (IndexParams params) {
            index->build(oracle, params, NULL);
            hasIndex = true;
        }

        python::object search (PyArrayObject *query, SearchParams params) const {
            check_array(query, DATA_TYPE());
            kgraph::MatrixProxy<DATA_TYPE> qmatrix(query);
            npy_intp dims[] = {qmatrix.size(), params.K};
            PyObject *result =  PyArray_SimpleNew(2, dims, NPY_UINT32);
            PyObject *distance =  PyArray_SimpleNew(2, dims, NPY_FLOAT);
            Py_BEGIN_ALLOW_THREADS
            kgraph::MatrixProxy<unsigned, 1> rmatrix(reinterpret_cast<PyArrayObject *>(result));
            kgraph::MatrixProxy<float, 1> distmatrix(reinterpret_cast<PyArrayObject *>(distance));
#ifdef _OPENMP
            if (params.threads) {
                params.threads = ::omp_get_num_threads();
                ::omp_set_num_threads(params.threads);
            }
#endif
#ifdef USE_BLAS
            if (params.blas && (qmatrix.size() >= MIN_BLAS_BATCH_SIZE)) {
                oracle.blasQuery(qmatrix, params.K, params.epsilon, &rmatrix, &distmatrix);
            }
            else
#endif
                if (hasIndex) {
#pragma omp parallel for 
                for (unsigned i = 0; i < qmatrix.size(); ++i) {
                    if (params.withDistance) {
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
                    if (params.withDistance) {
                        oracle.query(qmatrix[i]).search(params.K, params.epsilon, const_cast<unsigned *>(rmatrix[i]), 
                                     const_cast<float *>(distmatrix[i]));
                    }
                    else {
                        oracle.query(qmatrix[i]).search(params.K, params.epsilon, const_cast<unsigned *>(rmatrix[i]), NULL);
                    }
                }
            }
            Py_END_ALLOW_THREADS

            if (params.withDistance) {
                PyObject* tup = PyTuple_New(2);
                PyTuple_SetItem(tup,0,result);
                PyTuple_SetItem(tup,1,distance);
                return python::object(python::handle<>(tup));
            }
            else {
                Py_DECREF(distance);
                return python::object(python::handle<>(result));
            }
#ifdef _OPENMP
            if (params.threads) {
                ::omp_set_num_threads(params.threads);
            }
#endif
        }
    };
}

class KGraph {
    ImplBase *impl;
public:
    KGraph () {
        cerr << "!!!!!!!!!!" << endl;
        cerr << "pykgraph API has been changed" << endl;
        cerr << "Old:" << endl;
        cerr << "   index = pykgraph.KGraph()" << endl;
        cerr << "   index.build(dataset, ...)" << endl;
        cerr << "   index.search(dataset, query, ...)" << endl;
        cerr << "New (dataset passed in constructor):" << endl;
        cerr << "   index = pykgraph.KGraph(dataset, metric)" << endl;
        cerr << "   # metric: 'euclidean' or 'angular'" << endl;
        cerr << "   index.build(...)" << endl;
        cerr << "   index.search(query, ...)" << endl;
        cerr << "!!!!!!!!!!" << endl;
        throw kgraph::invalid_argument("obsolete constructor");
    }

    KGraph (PyObject *data, string const &metric): impl(nullptr) {
        PyArrayObject *pd = reinterpret_cast<PyArrayObject *>(data);
        if (!pd) throw kgraph::invalid_argument("bad array");
        if (metric == "euclidean") {
            switch (pd->descr->type_num) {
                case NPY_FLOAT: impl = new Impl<float, EuclideanLike>(pd); break;
                case NPY_DOUBLE: impl = new Impl<double, EuclideanLike>(pd); break;
            }
        }
        else if (metric == "angular") {
            switch (pd->descr->type_num) {
                case NPY_FLOAT: impl = new Impl<float, AngularLike>(pd); break;
                case NPY_DOUBLE: impl = new Impl<double, AngularLike>(pd); break;
            }
        }
        else throw runtime_error("metric not supported");
        if (!impl) throw runtime_error("data type not supported.");
    }

    ~KGraph () {
        delete impl;
    }

    void load (char const *path) {
        impl->load(path);
    }

    void save (char const *path) const {
        impl->save(path);
    }

    void build (unsigned iterations,
               unsigned L,
               unsigned K,
               unsigned S,
               unsigned controls,
               float delta,
               float recall,
               unsigned prune,
               int reverse) {
        IndexParams params;
        params.iterations = iterations;
        params.L = L;
        params.K = K;
        params.S = S;
        params.controls = controls;
        params.delta = delta;
        params.recall = recall;
        params.prune = prune;
        params.reverse = reverse;
        impl->build(params);
        /*
        PyArrayObject *pd = reinterpret_cast<PyArrayObject *>(data.ptr());
        switch (pd->descr->type_num) {
            case NPY_FLOAT: buildImpl<float>(params); return;
            case NPY_DOUBLE: buildImpl<double>(params); return;
        }
        throw runtime_error("data type not supported.");
        */
    }
    python::object search (PyObject *query,
                unsigned K,
                unsigned P,
                unsigned M,
                unsigned S,
                unsigned T,
                unsigned threads,
                bool withDistance,
                bool blas) {
        SearchParams params;
        params.K = K;
        params.P = P;
        params.M = M;
        params.S = S;
        params.T = T;
        params.threads = threads;
        params.withDistance = withDistance;
        params.blas = blas;
        return impl->search(reinterpret_cast<PyArrayObject *>(query), params);
    }
};

BOOST_PYTHON_MODULE(pykgraph)
{
    import_array();
    python::numeric::array::set_module_and_type("numpy", "ndarray");
    python::class_<KGraph>("KGraph", python::init<PyObject *, string>())
        .def(python::init<>())
        //.def(python::init<PyObject *, string>())
            //  (python::arg("data"), python::arg("metric") = "euclidean")))
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
              python::arg("prune") = kgraph::default_prune,
              python::arg("reverse") = kgraph::default_reverse))
        .def("search", &KGraph::search,
            (python::arg("data"),
             python::arg("query"),
             python::arg("K") = kgraph::default_K,
             python::arg("P") = kgraph::default_P,
             python::arg("M") = kgraph::default_M,
             python::arg("S") = kgraph::default_S,
             python::arg("T") = kgraph::default_T,
             python::arg("threads") = 0,
             python::arg("withDistance") = false,
             python::arg("blas") = false))
        ;
    python::def("load_lshkit", ::load_lshkit);
    python::def("version", kgraph::KGraph::version);
}


