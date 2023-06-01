#define FORCE_IMPORT_ARRAY
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor-python/pytensor.hpp>
//#include <numpy/ndarrayobject.h>
#include "kgraph.h"
#include "kgraph-data.h"

namespace py = pybind11;
using std::string;
using std::vector;
using std::runtime_error;

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


template <typename T>
class AlignedMatrix: public kgraph::Matrix<T> {
public:
    AlignedMatrix (xt::pytensor<T, 2> const &data) {
        unsigned row = data.shape(0);
        unsigned col = data.shape(1);
        this->reset(row, col);
        for (unsigned i = 0; i < row; ++i) {
            T const *from = &data(i, 0);
            T *to = this->operator[](i);
            std::copy(from, from + col, to);
        }
    }
};

template <typename DATA_TYPE, typename METRIC>
class NDArrayOracle: public kgraph::IndexOracle {
    kgraph::MatrixProxy<DATA_TYPE> proxy;
    vector<DATA_TYPE> n2;
public:
    class SearchOracle: public kgraph::SearchOracle {
        kgraph::MatrixProxy<DATA_TYPE> proxy;
        DATA_TYPE const *proxy_n2;
        DATA_TYPE const *query;
        float n2;
    public:
        SearchOracle (kgraph::MatrixProxy<DATA_TYPE> const &p, DATA_TYPE const *p_n2, DATA_TYPE const *q)
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

    NDArrayOracle (AlignedMatrix<DATA_TYPE> const &data): proxy(data), n2(proxy.size()) {
        for (unsigned i = 0; i < proxy.size(); ++i) {
            n2[i] = METRIC::norm(proxy[i], proxy.dim());
        }
    }

    ~NDArrayOracle () {
    }

    virtual unsigned size () const {
        return proxy.size();
    }

    virtual float operator () (unsigned i, unsigned j) const {
        return METRIC::dist(kgraph::metric::l2sqr::dot(proxy[i], proxy[j], proxy.dim()), n2[i], n2[j]);
    }

    SearchOracle query (DATA_TYPE const *q) const {
        return SearchOracle(proxy, &n2[0], q);
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
                    vector<std::pair<DATA_TYPE, unsigned>> rank(end-begin);
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
    virtual py::object search (py::object, SearchParams) const = 0;

    void load (char const *path) {
        index->load(path);
        hasIndex = true;
    }
    void save (char const *path) const {
        index->save(path);
    }
    void save_text (char const *path) const {
        index->save(path, kgraph::KGraph::FORMAT_TEXT);
    }
};


template <typename DATA_TYPE, typename METRIC_TYPE>
class Impl: public ImplBase {
    AlignedMatrix<DATA_TYPE> data;
    NDArrayOracle<DATA_TYPE, METRIC_TYPE> oracle; 
public:
    Impl (py::object obj): data(py::cast<xt::pytensor<DATA_TYPE, 2>>(obj)), oracle(data) {
    }

    ~Impl () {
    }

    void build (IndexParams params) {
        index->build(oracle, params, NULL);
        hasIndex = true;
    }

    py::object search (py::object query, SearchParams params) const {
        AlignedMatrix<DATA_TYPE> data(py::cast<xt::pytensor<DATA_TYPE, 2>>(query));
        kgraph::MatrixProxy<DATA_TYPE> qmatrix(data);
        //npy_intp dims[] = {qmatrix.size(), params.K};
        xt::pytensor<uint32_t, 2> result;
        result.resize({size_t(qmatrix.size()), size_t(params.K)});
        xt::pytensor<float, 2> distance;
        distance.resize({size_t(qmatrix.size()), size_t(params.K)});

        kgraph::MatrixProxy<uint32_t, 1> rmatrix(result);
        kgraph::MatrixProxy<float, 1> distmatrix(distance);

        Py_BEGIN_ALLOW_THREADS
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
#ifdef _OPENMP
        if (params.threads) {
            ::omp_set_num_threads(params.threads);
        }
#endif

        if (params.withDistance) {
            return py::make_tuple(result, distance);
        }
        else {
            return result;
        }
    }
};

class KGraph {
    ImplBase *impl;
public:
    KGraph (py::object data, string const &metric): impl(nullptr) {
        PyArrayObject *pd = reinterpret_cast<PyArrayObject *>(data.ptr());
        if (!pd) throw kgraph::invalid_argument("bad array");
        if (metric == "euclidean") {
            switch (PyArray_DESCR(pd)->type_num) {
                case NPY_FLOAT: impl = new Impl<float, EuclideanLike>(data); break;
                case NPY_DOUBLE: impl = new Impl<double, EuclideanLike>(data); break;
            }
        }
        else if (metric == "angular") {
            switch (PyArray_DESCR(pd)->type_num) {
                case NPY_FLOAT: impl = new Impl<float, AngularLike>(data); break;
                case NPY_DOUBLE: impl = new Impl<double, AngularLike>(data); break;
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

    void save_text (char const *path) const {
        impl->save_text(path);
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
    }
    py::object search (py::object query,
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
        return impl->search(query, params);
    }
};

PYBIND11_MODULE(pykgraph, module)
{
    xt::import_numpy();
    module.doc() = "";
    module.def("arch", []() {
                py::dict dict;
                dict["name"] = kgraph::xsimd_arch::name();
                dict["alignment"] = kgraph::xsimd_arch::alignment();
                return dict;
            });
    py::class_<KGraph>(module, "KGraph") 
        .def(py::init<py::object, string const &>())
        .def("load", &KGraph::load, "load")
        .def("save", &KGraph::save, "save")
        .def("save_text", &KGraph::save_text, "save_text")
        .def("build", &KGraph::build, "build",
              py::arg("iterations") = kgraph::default_iterations,
              py::arg("L") = kgraph::default_L,
              py::arg("K") = kgraph::default_K,
              py::arg("S") = kgraph::default_S,
              py::arg("controls") = kgraph::default_controls,
              py::arg("delta") = kgraph::default_delta,
              py::arg("recall") = kgraph::default_recall,
              py::arg("prune") = kgraph::default_prune,
              py::arg("reverse") = kgraph::default_reverse)
        .def("search", &KGraph::search, "search",
             py::arg("query"),
             py::arg("K") = kgraph::default_K,
             py::arg("P") = kgraph::default_P,
             py::arg("M") = kgraph::default_M,
             py::arg("S") = kgraph::default_S,
             py::arg("T") = kgraph::default_T,
             py::arg("threads") = 0,
             py::arg("withDistance") = false,
             py::arg("blas") = false)
    ;
}


