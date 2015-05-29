// Copyright (C) 2013, 2014 Wei Dong <wdong@wdong.org>. All Rights Reserved.

#ifndef WDONG_KGRAPH
#define WDONG_KGRAPH

namespace kgraph {
    static unsigned const default_iterations =  100;
    static unsigned const default_L = 50;
    static unsigned const default_K = 10;
    static unsigned const default_P = 100;
    static unsigned const default_M = 0;
    static unsigned const default_T = 1;
    static unsigned const default_S = 10;
    static unsigned const default_R = 100;
    static unsigned const default_controls = 100;
    static unsigned const default_seed = 1998;
    static float const default_delta = 0.005;
    static float const default_recall = 0.98;
    static float const default_epsilon = 1e30;
    static unsigned const default_verbosity = 1;
    enum {
        PRUNE_LEVEL_1 = 1,
        PRUNE_LEVEL_2 = 2
    };
    static unsigned const default_prune = 0;

    extern unsigned verbosity;

    class IndexOracle {
    public:
        virtual unsigned size () const = 0;
        virtual float operator () (unsigned i, unsigned j) const = 0;
    };

    class SearchOracle {
    public:
        virtual unsigned size () const = 0;
        virtual float operator () (unsigned i) const = 0;
        unsigned search (unsigned K, float epsilon, unsigned *ids, float *dists = nullptr) const;
    };

    class KGraph {
    public:
        struct IndexParams {
            unsigned iterations; 
            unsigned L;
            unsigned K;
            unsigned S;
            unsigned R;
            unsigned controls;
            unsigned seed;
            float delta;
            float recall;
            unsigned prune;

            IndexParams (): iterations(default_iterations), L(default_L), K(default_K), S(default_S), R(default_R), controls(default_controls), seed(default_seed), delta(default_delta), recall(default_recall), prune(default_prune) {
            }
        };

        struct SearchParams {
            unsigned K;
            unsigned M;
            unsigned P;
            unsigned T;
            float epsilon;
            unsigned seed;
            unsigned init;

            SearchParams (): K(default_K), M(default_M), P(default_P), T(default_T), epsilon(default_epsilon), seed(1998), init(0) {
            }
        };

        struct IndexInfo {
            enum StopCondition {
                ITERATION = 0,
                DELTA,
                RECALL
            } stop_condition;
            unsigned iterations;
            float cost;
            float recall;
            float accuracy;
            float delta;
            float M;
        };

        struct SearchInfo {
            float cost;
            unsigned updates;
        };

        virtual ~KGraph () {
        }
        virtual void load (char const *path) = 0;
        virtual void save (char const *path) const = 0; // save to file
        virtual void build (IndexOracle const &oracle, IndexParams const &params, IndexInfo *info) = 0;
        virtual void prune (IndexOracle const &oracle, unsigned level) = 0;
        unsigned search (SearchOracle const &oracle, SearchParams const &params, unsigned *ids, SearchInfo *info) const {
            return search(oracle, params, ids, nullptr, info);
        }
        virtual unsigned search (SearchOracle const &oracle, SearchParams const &params, unsigned *ids, float *dists, SearchInfo *info) const = 0;
        static KGraph *create ();
        static char const* version ();

        virtual void get_nn (unsigned id, unsigned *nns, unsigned *M, unsigned *L) const {
            get_nn(id, nns, nullptr, M, L);
        }
        virtual void get_nn (unsigned id, unsigned *nns, float *dist, unsigned *M, unsigned *L) const = 0;
    };
}

#if __cplusplus > 199711L
#include <functional>
namespace kgraph {
    template <typename CONTAINER_TYPE, typename OBJECT_TYPE>
    class VectorOracle: public IndexOracle {
    public:
        typedef std::function<float(OBJECT_TYPE const &, OBJECT_TYPE const &)> METRIC_TYPE;
    private:
        CONTAINER_TYPE const &data;
        METRIC_TYPE dist;
    public:
        class VectorSearchOracle: public SearchOracle {
            CONTAINER_TYPE const &data;
            OBJECT_TYPE const query;
            METRIC_TYPE dist;
        public:
            VectorSearchOracle (CONTAINER_TYPE const &p, OBJECT_TYPE const &q, METRIC_TYPE m): data(p), query(q), dist(m) {
            }
            virtual unsigned size () const {
                return data.size();
            }
            virtual float operator () (unsigned i) const {
                return dist(data[i], query);
            }
        };
        VectorOracle (CONTAINER_TYPE const &d, METRIC_TYPE m): data(d), dist(m) {
        }
        virtual unsigned size () const {
            return data.size();
        }
        virtual float operator () (unsigned i, unsigned j) const {
            return dist(data[i], data[j]);
        }
        VectorSearchOracle query (OBJECT_TYPE const &q) const {
            return VectorSearchOracle(data, q, dist);
        }
    };
}
#endif

#endif

