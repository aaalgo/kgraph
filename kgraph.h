#ifndef WDONG_KGRAPH
#define WDONG_KGRAPH

#include <string>
#include <vector>
#include <boost/timer/timer.hpp>

namespace kgraph {

    using std::string;
    using std::vector;
    using boost::timer::cpu_times;

    // we use unsigned for size because it will be impossible for the library
    // to index > 4 billion points in the near future.
    class IndexOracle {
    public:
        virtual unsigned size () const = 0;
        virtual float operator () (unsigned i, unsigned j) const = 0;
    };

    class SearchOracle {
    public:
        virtual unsigned size () const = 0;
        virtual float operator () (unsigned i) const = 0;
        // brutal force K-NN search
        void search (unsigned K, unsigned *ids) const;
    };

    class KGraph {
    protected:
        vector<vector<unsigned>> graph;
    public:
        struct IndexParams {
            unsigned iterations; // # iteration
            unsigned L;
            unsigned K;
            unsigned S;
            unsigned R;
            unsigned controls; // control
            unsigned seed;
            float delta;
            float recall; // target recall

            IndexParams (): iterations(100), L(50), K(10), S(5), R(20), controls(100), seed(1998), delta(0.005), recall(0.98) {
            }

            void check () {
                //BOOST_VERIFY(S <= K);
                BOOST_VERIFY(K > 0);
            }
        };

        struct IndexInfo {
            enum StopCondition {
                ITERATION,
                DELTA,
                RECALL
            } stop_condition;
            unsigned iterations;
            float cost;
            float recall;
            float accuracy;
            float delta;
            cpu_times times;
        };

        struct SearchParams {
            unsigned K;
            unsigned seed;
            bool init;
            SearchParams (): K(20), seed(1998), init(false) {
            }
        };

        struct SearchInfo {
            float cost;
            unsigned updates;
            cpu_times times;
        };

        void load (string const &path);
        void save (string const &path); // save to file
        void build (IndexOracle const &oracle, IndexParams const &params, IndexInfo *info = nullptr);
        void search (SearchOracle const &oracle, SearchParams const &params, unsigned *ids, SearchInfo *info = nullptr);

        KGraph () {}
        KGraph (IndexOracle const &oracle, IndexParams const &params, IndexInfo *info = nullptr) {
            build(oracle, params, info);
        }
        KGraph (string const &path) {
            load(path);
        }
    };
}

#endif

