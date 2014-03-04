#ifndef WDONG_KGRAPH_MATRIX
#define WDONG_KGRAPH_MATRIX

#include <malloc.h>
#include <vector>
#include <fstream>
#include <stdexcept>

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
    using std::vector;
    using std::runtime_error;

    namespace metric {
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
        };
        struct l2 {
            template <typename T>
            static float apply (T const *t1, T const *t2, unsigned dim) {
                return sqrt(l2sqr::apply<T>(t1, t2, dim));
            }
        };
    }

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

    template <typename DATA_TYPE>
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
        MatrixProxy (flann::Matrix<DATA_TYPE> const &m)
            : rows(m.rows), cols(m.cols), stride(m.stride), data(m.data) {
            BOOST_VERIFY(stride % KGRAPH_MATRIX_ALIGN == 0);
        }
#endif
#ifdef __OPENCV_CORE_HPP__
        MatrixProxy (cv::Mat const &m)
            : rows(m.rows), cols(m.cols), stride(m.step), data(m.data) {
            BOOST_VERIFY(stride % KGRAPH_MATRIX_ALIGN == 0);
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
    };

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
}  // namespace 

#ifdef __GNUC__
#ifdef __AVX__
#include <immintrin.h>
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_load_ps(addr1);\
    tmp2 = _mm256_load_ps(addr2);\
    tmp1 = _mm256_sub_ps(tmp1, tmp2); \
    tmp1 = _mm256_mul_ps(tmp1, tmp1); \
    dest = _mm256_add_ps(dest, tmp1); 
namespace kgraph { namespace metric {
        template <>
        float l2sqr::apply<float> (float const *t1, float const *t2, unsigned dim) {
            __m256 sum;
            __m256 l0, l1, l2, l3;
            __m256 r0, r1, r2, r3;
            unsigned D = (dim + 7) & ~7U; // # dim aligned up to 256 bits, or 8 floats
            unsigned DR = D % 32;
            unsigned DD = D - DR;
            const float *l = t1;
            const float *r = t2;
            const float *e_l = l + DD;
            const float *e_r = r + DD;
            float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};
            float ret = 0.0;
            sum = _mm256_load_ps(unpack);
            switch (DR) {
                case 24:
                    AVX_L2SQR(e_l+16, e_r+16, sum, l2, r2);
                case 16:
                    AVX_L2SQR(e_l+8, e_r+8, sum, l1, r1);
                case 8:
                    AVX_L2SQR(e_l, e_r, sum, l0, r0);
            }
            for (unsigned i = 0; i < DD; i += 32, l += 32, r += 32) {
                AVX_L2SQR(l, r, sum, l0, r0);
                AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
                AVX_L2SQR(l + 16, r + 16, sum, l2, r2);
                AVX_L2SQR(l + 24, r + 24, sum, l3, r3);
            }
            _mm256_storeu_ps(unpack, sum);
            ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
                + unpack[4] + unpack[5] + unpack[6] + unpack[7];
            return ret;//sqrt(ret);
        }
}}
#else
#ifdef __SSE2__
#include <xmmintrin.h>
#define SSE_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm_load_ps(addr1);\
    tmp2 = _mm_load_ps(addr2);\
    tmp1 = _mm_sub_ps(tmp1, tmp2); \
    tmp1 = _mm_mul_ps(tmp1, tmp1); \
    dest = _mm_add_ps(dest, tmp1); 
namespace kgraph { namespace metric {
    template <>
    float l2sqr::apply<float> (float const *t1, float const *t2, unsigned dim) {
        __m128 sum;
        __m128 l0, l1, l2, l3;
        __m128 r0, r1, r2, r3;
        unsigned D = (dim + 3) & ~3U;
        unsigned DR = D % 16;
        unsigned DD = D - DR;
        const float *l = t1;
        const float *r = t2;
        const float *e_l = l + DD;
        const float *e_r = r + DD;
        float unpack[4] __attribute__ ((aligned (16))) = {0, 0, 0, 0};
        float ret = 0.0;
        sum = _mm_load_ps(unpack);
        switch (DR) {
            case 12:
                SSE_L2SQR(e_l+8, e_r+8, sum, l2, r2);
            case 8:
                SSE_L2SQR(e_l+4, e_r+4, sum, l1, r1);
            case 4:
                SSE_L2SQR(e_l, e_r, sum, l0, r0);
        }
        for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
            SSE_L2SQR(l, r, sum, l0, r0);
            SSE_L2SQR(l + 4, r + 4, sum, l1, r1);
            SSE_L2SQR(l + 8, r + 8, sum, l2, r2);
            SSE_L2SQR(l + 12, r + 12, sum, l3, r3);
        }
        _mm_storeu_ps(unpack, sum);
        ret = unpack[0] + unpack[1] + unpack[2] + unpack[3];
        return ret;//sqrt(ret);
    }
}}

/*
template <typename T>
void print_128 (__m128i v) {
    static unsigned constexpr L = 16 / sizeof(T);
    T unpack[L] __attribute__ ((aligned (16)));
    _mm_store_si128((__m128i *)unpack, v);
    cout << '(' << int(unpack[0]);
    for (unsigned i = 1; i < L; ++i) {
        cout << ',' << int(unpack[i]);
    }
    cout << ')';
}
*/


#define SSE_L2SQR_BYTE(addr1, addr2, sum, z) \
    do { \
        const __m128i o = _mm_load_si128((__m128i const *)(addr1));\
        const __m128i p = _mm_load_si128((__m128i const *)(addr2));\
        __m128i o1 = _mm_unpackhi_epi8(o,z); \
        __m128i p1 = _mm_unpackhi_epi8(p,z); \
        __m128i d = _mm_sub_epi16(o1, p1); \
        sum = _mm_add_epi32(sum, _mm_madd_epi16(d, d)); \
        o1 = _mm_unpacklo_epi8(o,z); \
        p1 = _mm_unpacklo_epi8(p,z); \
        d = _mm_sub_epi16(o1, p1); \
        sum = _mm_add_epi32(sum, _mm_madd_epi16(d, d)); \
    } while (false)
namespace kgraph { namespace metric {
    template <>
    float l2sqr::apply<uint8_t> (uint8_t const *t1, uint8_t const *t2, unsigned dim) {
        unsigned D = (dim + 0xFU) & ~0xFU;   // actual dimension used in calculation, 0-padded
        unsigned DR = D % 64;           // process 32 dims per iteration
        unsigned DD = D - DR;
        const uint8_t *l = t1;
        const uint8_t *r = t2;
        const uint8_t *e_l = l + DD;
        const uint8_t *e_r = r + DD;
        int32_t unpack[4] __attribute__ ((aligned (16))) = {0, 0, 0, 0};
        __m128i sum = _mm_load_si128((__m128i *)unpack);
        const __m128i z = sum;
        switch (DR) {
            case 48:
                SSE_L2SQR_BYTE(e_l+32, e_r+32, sum, z);
            case 32:
                SSE_L2SQR_BYTE(e_l+16, e_r+16, sum, z);
            case 16:
                SSE_L2SQR_BYTE(e_l, e_r, sum, z);
        }
        for (unsigned i = 0; i < DD; i += 64, l += 64, r += 64) {
            SSE_L2SQR_BYTE(l, r, sum, z);
            SSE_L2SQR_BYTE(l + 16, r + 16, sum, z);
            SSE_L2SQR_BYTE(l + 32, r + 32, sum, z);
            SSE_L2SQR_BYTE(l + 48, r + 48, sum, z);
        }
        _mm_store_si128((__m128i *)unpack, sum);
        int32_t ret = unpack[0] + unpack[1] + unpack[2] + unpack[3];
        return float(ret);//sqrt(ret);
    }
}}

#endif
#endif
#endif



#endif

