#ifndef WDONG_KGRAPH_DATA
#define WDONG_KGRAPH_DATA

#include <cstring>
#include <malloc.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <boost/assert.hpp>

#ifdef __GNUC__
#ifdef __AVX__
#define KGRAPH_MATRIX_ALIGN 32
#else
#ifdef __SSE2__
#define KGRAPH_MATRIX_ALIGN 16
#else
#define KGRAPH_MATRIX_ALIGN 4
#endif
#endif
#endif

namespace kgraph {

    /// L2 square distance with AVX instructions.
    /** AVX instructions have strong alignment requirement for t1 and t2.
     */
    extern float float_l2sqr_avx (float const *t1, float const *t2, unsigned dim);
    /// L2 square distance with SSE2 instructions.
    extern float float_l2sqr_sse2 (float const *t1, float const *t2, unsigned dim);
    extern float float_l2sqr_sse2 (float const *, unsigned dim);
    extern float float_dot_sse2 (float const *, float const *, unsigned dim);
    /// L2 square distance for uint8_t with SSE2 instructions (for SIFT).
    extern float uint8_l2sqr_sse2 (uint8_t const *t1, uint8_t const *t2, unsigned dim);

    extern float float_l2sqr (float const *, float const *, unsigned dim);
    extern float float_l2sqr (float const *, unsigned dim);
    extern float float_dot (float const *, float const *, unsigned dim);


    using std::vector;
    using std::runtime_error;

    /// namespace for various distance metrics.
    namespace metric {
        /// L2 square distance.
        struct l2sqr {
            template <typename T>
            static float apply (T const *t1, T const *t2, unsigned dim) {
                float r = 0;
                for (unsigned i = 0; i < dim; ++i) {
                    float v = float(t1[i]) - float(t2[i]);
                    v *= v;
                    r += v;
                }
                return r;
            }

            template <typename T>
            static float dot (T const *t1, T const *t2, unsigned dim) {
                float r = 0;
                for (unsigned i = 0; i < dim; ++i) {
                    r += float(t1[i]) *float(t2[i]);
                }
                return r;
            }

            template <typename T>
            static float norm2 (T const *t1, unsigned dim) {
                float r = 0;
                for (unsigned i = 0; i < dim; ++i) {
                    float v = float(t1[i]);
                    v *= v;
                    r += v;
                }
                return r;
            }
        };
        /// L2 distance.
        struct l2 {
            template <typename T>
            static float apply (T const *t1, T const *t2, unsigned dim) {
                return sqrt(l2sqr::apply<T>(t1, t2, dim));
            }
        };
    }

    /// Matrix data.
    template <typename T, unsigned A = KGRAPH_MATRIX_ALIGN>
    class Matrix {
        unsigned col;
        unsigned row;
        size_t stride;
        char *data;

        void reset (unsigned r, unsigned c) {
            row = r;
            col = c;
            stride = (sizeof(T) * c + A - 1) / A * A;
            /*
            data.resize(row * stride);
            */
            if (data) free(data);
            data = (char *)memalign(A, row * stride); // SSE instruction needs data to be aligned
            if (!data) throw runtime_error("memalign");
        }
    public:
        Matrix (): col(0), row(0), stride(0), data(0) {}
        Matrix (unsigned r, unsigned c): data(0) {
            reset(r, c);
        }
        ~Matrix () {
            if (data) free(data);
        }
        unsigned size () const {
            return row;
        }
        unsigned dim () const {
            return col;
        }
        size_t step () const {
            return stride;
        }
        void resize (unsigned r, unsigned c) {
            reset(r, c);
        }
        T const *operator [] (unsigned i) const {
            return reinterpret_cast<T const *>(&data[stride * i]);
        }
        T *operator [] (unsigned i) {
            return reinterpret_cast<T *>(&data[stride * i]);
        }
        void zero () {
            memset(data, 0, row * stride);
        }
        void load (const std::string &path, unsigned dim, unsigned skip = 0, unsigned gap = 0) {
            std::ifstream is(path.c_str(), std::ios::binary);
            BOOST_VERIFY(is);
            is.seekg(0, std::ios::end);
            size_t size = is.tellg();
            size -= skip;
            unsigned line = sizeof(T) * dim + gap;
            unsigned N =  size / line;
            reset(N, dim);
            zero();
            is.seekg(skip, std::ios::beg);
            for (unsigned i = 0; i < N; ++i) {
                is.read(&data[stride * i], sizeof(T) * dim);
                is.seekg(gap, std::ios::cur);
            }
            BOOST_VERIFY(is);
        }

        void load_lshkit (std::string const &path) {
            static const unsigned LSHKIT_HEADER = 3;
            std::ifstream is(path.c_str(), std::ios::binary);
            unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
            is.read((char *)header, sizeof header);
            BOOST_VERIFY(is);
            BOOST_VERIFY(header[0] == sizeof(T));
            is.close();
            unsigned D = header[2];
            unsigned skip = LSHKIT_HEADER * sizeof(unsigned);
            unsigned gap = 0;
            load(path, D, skip, gap);
        }

        void save_lshkit (std::string const &path) {
            std::ofstream os(path.c_str(), std::ios::binary);
            unsigned header[3];
            assert(sizeof header == 3*4);
            header[0] = sizeof(T);
            header[1] = row;
            header[2] = col;
            os.write((const char *)header, sizeof(header));
            for (unsigned i = 0; i < row; ++i) {
                os.write(&data[stride * i], sizeof(T) * col);
            }
        }
    };

    /// Matrix proxy to interface with 3rd party libraries (FLANN, OpenCV, NumPy).
    template <typename DATA_TYPE, unsigned A = KGRAPH_MATRIX_ALIGN>
    class MatrixProxy {
        unsigned rows;
        unsigned cols;      // # elements, not bytes, in a row, 
        size_t stride;    // # bytes in a row, >= cols * sizeof(element)
        uint8_t const *data;
    public:
        MatrixProxy (Matrix<DATA_TYPE> const &m)
            : rows(m.size()), cols(m.dim()), stride(m.step()), data(reinterpret_cast<uint8_t const *>(m[0])) {
        }

#ifndef __AVX__
#ifdef FLANN_DATASET_H_
        /// Construct from FLANN matrix.
        MatrixProxy (flann::Matrix<DATA_TYPE> const &m)
            : rows(m.rows), cols(m.cols), stride(m.stride), data(m.data) {
            BOOST_VERIFY(stride % A == 0);
        }
#endif
#ifdef __OPENCV_CORE_HPP__
        /// Construct from OpenCV matrix.
        MatrixProxy (cv::Mat const &m)
            : rows(m.rows), cols(m.cols), stride(m.step), data(m.data) {
            BOOST_VERIFY(stride % A == 0);
        }
#endif
#ifdef NPY_NDARRAYOBJECT_H
        /// Construct from NumPy matrix.
        MatrixProxy (PyArrayObject *obj) {
            BOOST_VERIFY(obj->nd == 2);
            rows = obj->dimensions[0];
            cols = obj->dimensions[1];
            stride = obj->strides[0];
            data = reinterpret_cast<uint8_t const *>(obj->data);
            BOOST_VERIFY(obj->descr->elsize == sizeof(DATA_TYPE));
            BOOST_VERIFY(stride % A == 0);
            BOOST_VERIFY(stride >= cols * sizeof(DATA_TYPE));
        }
#endif
#endif
        unsigned size () const {
            return rows;
        }
        unsigned dim () const {
            return cols;
        }
        DATA_TYPE const *operator [] (unsigned i) const {
            return reinterpret_cast<DATA_TYPE const *>(data + stride * i);
        }
        DATA_TYPE *operator [] (unsigned i) {
            return const_cast<DATA_TYPE *>(reinterpret_cast<DATA_TYPE const *>(data + stride * i));
        }
    };

    /// Oracle for matrix data.
    /** DATA_TYPE can be Matrix or MatrixProxy,
    * DIST_TYPE should be one class within the namespace kgraph.metric.
    */
    template <typename DATA_TYPE, typename DIST_TYPE>
    class MatrixOracle: public kgraph::IndexOracle {
        MatrixProxy<DATA_TYPE> proxy;
    public:
        class SearchOracle: public kgraph::SearchOracle {
            MatrixProxy<DATA_TYPE> proxy;
            DATA_TYPE const *query;
        public:
            SearchOracle (MatrixProxy<DATA_TYPE> const &p, DATA_TYPE const *q): proxy(p), query(q) {
            }
            virtual unsigned size () const {
                return proxy.size();
            }
            virtual float operator () (unsigned i) const {
                return DIST_TYPE::apply(proxy[i], query, proxy.dim());
            }
        };
        template <typename MATRIX_TYPE>
        MatrixOracle (MATRIX_TYPE const &m): proxy(m) {
        }
        virtual unsigned size () const {
            return proxy.size();
        }
        virtual float operator () (unsigned i, unsigned j) const {
            return DIST_TYPE::apply(proxy[i], proxy[j], proxy.dim());
        }
        SearchOracle query (DATA_TYPE const *query) const {
            return SearchOracle(proxy, query);
        }
    };

    inline float AverageRecall (Matrix<float> const &gs, Matrix<float> const &result, unsigned K = 0) {
        if (K == 0) {
            K = result.dim();
        }
        BOOST_VERIFY(gs.dim() >= K);
        BOOST_VERIFY(result.dim() >= K);
        BOOST_VERIFY(gs.size() >= result.size());
        float sum = 0;
        for (unsigned i = 0; i < result.size(); ++i) {
            float const *gs_row = gs[i];
            float const *re_row = result[i];
            // compare
            unsigned found = 0;
            unsigned gs_n = 0;
            unsigned re_n = 0;
            while ((gs_n < K) && (re_n < K)) {
                if (gs_row[gs_n] < re_row[re_n]) {
                    ++gs_n;
                }
                else if (gs_row[gs_n] == re_row[re_n]) {
                    ++found;
                    ++gs_n;
                    ++re_n;
                }
                else {
                    throw runtime_error("distance is unstable");
                }
            }
            sum += float(found) / K;
        }
        return sum / result.size();
    }


}

#ifndef KGRAPH_NO_VECTORIZE
#ifdef __GNUC__
#ifdef __AVX__
#if 0
namespace kgraph { namespace metric {
        template <>
        inline float l2sqr::apply<float> (float const *t1, float const *t2, unsigned dim) {
            return float_l2sqr_avx(t1, t2, dim);
        }
}}
#endif
#else
#ifdef __SSE2__
namespace kgraph { namespace metric {
        template <>
        inline float l2sqr::apply<float> (float const *t1, float const *t2, unsigned dim) {
            return float_l2sqr_sse2(t1, t2, dim);
        }
        template <>
        inline float l2sqr::dot<float> (float const *t1, float const *t2, unsigned dim) {
            return float_dot_sse2(t1, t2, dim);
        }
        template <>
        inline float l2sqr::norm2<float> (float const *t1, unsigned dim) {
            return float_l2sqr_sse2(t1, dim);
        }
        template <>
        inline float l2sqr::apply<uint8_t> (uint8_t const *t1, uint8_t const *t2, unsigned dim) {
            return uint8_l2sqr_sse2(t1, t2, dim);
        }
}}
#endif
#endif
#endif
#endif



#endif

