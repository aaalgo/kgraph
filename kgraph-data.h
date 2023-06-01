#ifndef WDONG_KGRAPH_DATA
#define WDONG_KGRAPH_DATA

#include <cmath>
#include <cstring>
#include <malloc.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <boost/assert.hpp>
#include <xsimd/xsimd.hpp>


namespace kgraph {

typedef xsimd::best_arch xsimd_arch;

static constexpr size_t KGRAPH_MATRIX_ALIGN = xsimd_arch::alignment();

template <typename T>
T simd_l2sqr (T const *t1, unsigned dim) {
}

template <typename T>
T simd_dot (T const *t1, T const *t2, unsigned dim) {
}

}


namespace kgraph {

    using std::vector;

    /// namespace for various distance metrics.
    namespace metric {
        /// L2 square distance.
        struct l2sqr {
            template <typename T>
            /// L2 square distance.
            static T apply (T const *t1, T const *t2, unsigned size) {
                using b_type = xsimd::batch<T, xsimd_arch>;
                unsigned constexpr inc = b_type::size;
                unsigned vec_size = size - size % inc;
                unsigned i = 0;
                b_type c = 0;
                for (; i < vec_size; i += inc) {
                    b_type a = b_type::load_aligned(t1 + i);
                    b_type b = b_type::load_aligned(t2 + i);
                    a -= b;
                    c += a * a;
                }
                T acc = xsimd::reduce_add(c);
                for (; i < size;  ++i) {
                    T a = t1[i];
                    T b = t2[i];
                    a -= b;
                    acc += a * a;
                }
                return acc;
            }

            /// inner product.
            template <typename T>
            static T dot (T const *t1, T const *t2, unsigned size) {
                using b_type = xsimd::batch<T, xsimd_arch>;
                unsigned constexpr inc = b_type::size;
                unsigned vec_size = size - size % inc;
                unsigned i = 0;
                b_type c = 0;
                for (; i < vec_size; i += inc) {
                    b_type a = b_type::load_aligned(t1 + i);
                    b_type b = b_type::load_aligned(t2 + i);
                    c += a * b;
                }
                T acc = xsimd::reduce_add(c);
                for (; i < size;  ++i) {
                    T a = t1[i];
                    T b = t2[i];
                    acc += a * b;
                }
                return acc;
            }

            /// L2 norm.
            template <typename T>
            static float norm2 (T const *t1, unsigned size) {
                using b_type = xsimd::batch<T, xsimd_arch>;
                unsigned constexpr inc = b_type::size;
                unsigned vec_size = size - size % inc;
                unsigned i = 0;
                b_type c = 0;
                for (; i < vec_size; i += inc) {
                    b_type a = b_type::load_aligned(t1 + i);
                    c += a * a;
                }
                T acc = xsimd::reduce_add(c);
                for (; i < size;  ++i) {
                    T a = t1[i];
                    acc += a * a;
                }
                return acc;
            }
        };

        struct l2 {
            template <typename T>
            static T apply (T const *t1, T const *t2, unsigned dim) {
                return std::sqrt(l2sqr::apply<T>(t1, t2, dim));
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
    protected:
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

        void normalize2 () {
#pragma omp parallel for
            for (unsigned i = 0; i < row; ++i) {
                T *p = operator[](i);
                double sum = metric::l2sqr::norm2(p, col);
                sum = std::sqrt(sum);
                for (unsigned j = 0; j < col; ++j) {
                    p[j] /= sum;
                }
            }
        }
        
        void load (const std::string &path, unsigned dim, unsigned skip = 0, unsigned gap = 0) {
            std::ifstream is(path.c_str(), std::ios::binary);
            if (!is) throw io_error(path);
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
            if (!is) throw io_error(path);
        }

        void load_lshkit (std::string const &path) {
            static const unsigned LSHKIT_HEADER = 3;
            std::ifstream is(path.c_str(), std::ios::binary);
            unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
            is.read((char *)header, sizeof header);
            if (!is) throw io_error(path);
            if (header[0] != sizeof(T)) throw io_error(path);
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

        ~MatrixProxy () {
        }

//#ifndef __AVX__
#ifdef FLANN_DATASET_H_
        /// Construct from FLANN matrix.
        MatrixProxy (flann::Matrix<DATA_TYPE> const &m)
            : rows(m.rows), cols(m.cols), stride(m.stride), data(m.data) {
            if (stride % A) throw invalid_argument("bad alignment");
        }
#endif
#ifdef CV_MAJOR_VERSION
        /// Construct from OpenCV matrix.
        MatrixProxy (cv::Mat const &m)
            : rows(m.rows), cols(m.cols), stride(m.step), data(m.data) {
            if (stride % A) throw invalid_argument("bad alignment");
        }
#endif
#ifdef XTENSOR_ARRAY_HPP
        /// Construct from NumPy matrix.
        MatrixProxy (xt::xtensor<DATA_TYPE, 2> const &obj) {
            rows = obj.shape(0);
            cols = obj.shape(1);
            if (rows <= 1) {
                stride = (cols * sizeof(DATA_TYPE) + A -1) / A * A;
            }
            else {
                stride = reinterpret_cast<char const *>(&obj(1,0))
                   - reinterpret_cast<char const *>(&obj(0,0));
            }
            data = reinterpret_cast<uint8_t const *>(&obj(0,0));
            if (stride % A) throw invalid_argument("bad alignment");
            if (!(stride >= cols * sizeof(DATA_TYPE))) throw invalid_argument("bad stride");
        }

        MatrixProxy (xt::pytensor<DATA_TYPE, 2> const &obj) {
            rows = obj.shape(0);
            cols = obj.shape(1);
            if (rows <= 1) {
                stride = (cols * sizeof(DATA_TYPE) + A -1) / A * A;
            }
            else {
                stride = reinterpret_cast<char const *>(&obj(1,0))
                   - reinterpret_cast<char const *>(&obj(0,0));
            }
            data = reinterpret_cast<uint8_t const *>(&obj(0,0));
            if (stride % A) throw invalid_argument("bad alignment");
            if (!(stride >= cols * sizeof(DATA_TYPE))) throw invalid_argument("bad stride");
        }
#endif
//#endif
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

    /// Oracle for Matrix or MatrixProxy.
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
        if (!(gs.dim() >= K)) throw invalid_argument("gs.dim() >= K");
        if (!(result.dim() >= K)) throw invalid_argument("result.dim() >= K");
        if (!(gs.size() >= result.size())) throw invalid_argument("gs.size() > result.size()");
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

#endif

