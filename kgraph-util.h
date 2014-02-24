#ifndef WDONG_KGRAPH_UTIL
#define WDONG_KGRAPH_UTIL

#include <kgraph-matrix.h>

namespace kgraph {

    typedef Matrix<float> FloatDataMatrix;
    typedef Matrix<unsigned, 1> IndexMatrix;

    // both groundtruth and results should be sorted
    float AverageRecall (IndexMatrix const &eval, IndexMatrix const &result, unsigned K = 0) {
        if (K == 0) {
            K = result.dim();
        }
        BOOST_VERIFY(eval.dim() >= K);
        BOOST_VERIFY(result.dim() >= K);
        BOOST_VERIFY(eval.size() >= result.size());
        float sum = 0;
        for (unsigned i = 0; i < result.size(); ++i) {
            unsigned const *gs = eval[i];
            unsigned const *re = result[i];
            // compare
            unsigned found = 0;
            for (unsigned j = 0; j < K; ++j) {
                for (unsigned k = 0; k < K; ++k) {
                    if (gs[j] == re[k]) {
                        ++found;
                        break;
                    }
                }
            }
            sum += float(found) / K;
        }
        return sum / result.size();
    }

}

#endif
