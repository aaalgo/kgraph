// Copyright (C) 2013, 2014 Wei Dong <wdong@wdong.org>. All Rights Reserved.

#ifndef WDONG_KGRAPH
#define WDONG_KGRAPH

namespace kgraph {
    static unsigned const default_iterations =  100;
    static unsigned const default_L = 50;
    static unsigned const default_K = 10;
    static unsigned const default_M = 100;
    static unsigned const default_S = 10;
    static unsigned const default_R = 100;
    static unsigned const default_controls = 100;
    static unsigned const default_seed = 1998;
    static float const default_delta = 0.005;
    static float const default_recall = 0.98;
    static float const default_epsilon = 1e30;
    static unsigned const default_verbosity = 1;

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
        unsigned search (unsigned K, float epsilon, unsigned *ids) const;
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

            IndexParams (): iterations(default_iterations), L(default_L), K(default_K), S(default_S), R(default_R), controls(default_controls), seed(default_seed), delta(default_delta), recall(default_recall) {
            }
        };

        struct SearchParams {
            unsigned K;
            unsigned M;
            float epsilon;
            unsigned seed;
            unsigned init;

            SearchParams (): K(default_K), M(default_M), epsilon(default_epsilon), seed(1998), init(0) {
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
        virtual void save (char const *path) = 0; // save to file
        virtual void build (IndexOracle const &oracle, IndexParams const &params, IndexInfo *info) = 0;
        virtual void prune (unsigned K) = 0;
        virtual unsigned search (SearchOracle const &oracle, SearchParams const &params, unsigned *ids, SearchInfo *info) = 0;
        static KGraph *create ();
    };
}

#endif

