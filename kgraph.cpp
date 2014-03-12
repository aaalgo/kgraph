#include <omp.h>
#include <mutex>
#include <set>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#define timer timer_for_boost_progress_t
#include <boost/progress.hpp>
#undef timer
#include <boost/timer/timer.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include "boost/smart_ptr/detail/spinlock.hpp"
#include "kgraph.h"

namespace kgraph {

    using namespace std;
    using namespace boost;
    using namespace boost::accumulators;

    typedef boost::detail::spinlock Lock;
    typedef std::lock_guard<Lock> LockGuard;

    // generate size distinct random numbers < N
    template <typename RNG>
    void GenRandom (RNG &rng, unsigned *addr, unsigned size, unsigned N) {
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }
        sort(addr, addr + size);
        for (unsigned i = 1; i < size; ++i) {
            if (addr[i] <= addr[i-1]) {
                addr[i] = addr[i-1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }

    struct Neighbor {
        unsigned id;
        float dist;
        bool flag;  // whether this entry is a newly found one
        Neighbor () {}
        Neighbor (unsigned i, float d, bool f = true): id(i), dist(d), flag(f) {
        }
    };

    static inline bool operator < (Neighbor const &n1, Neighbor const &n2) {
        return n1.dist < n2.dist;
    }

    typedef vector<Neighbor> Neighbors;

    // both pool and knn should be sorted
    static float EvaluateRecall (Neighbors const &pool, Neighbors const &knn) {
        if (knn.empty()) return 1.0;
        unsigned found = 0;
        unsigned n_p = 0;
        unsigned n_k = 0;
        for (;;) {
            if (n_p >= pool.size()) break;
            while ((n_k < knn.size()) && (knn[n_k].dist < pool[n_p].dist)) ++n_k;
            if (n_k >= knn.size()) break;
            unsigned n = n_k;
            while ((n < knn.size()) && (knn[n].dist == pool[n_p].dist)) {
                if (knn[n].id == pool[n_p].id) {
                    ++found;
                    break;
                }
                ++n;
            }
            ++n_p;
        }
        return float(found) / knn.size();
    }

    static float EvaluateAccuracy (Neighbors const &pool, Neighbors const &knn) {
        unsigned m = std::min(pool.size(), knn.size());
        float sum = 0;
        unsigned cnt = 0;
        for (unsigned i = 0; i < m; ++i) {
            if (knn[i].dist >  0) {
                sum += abs(pool[i].dist - knn[i].dist) / knn[i].dist;
                ++cnt;
            }
        }
        return cnt > 0 ? sum / cnt: 0;
    }

    static float EvaluateOneRate (Neighbors const &pool, Neighbors const &knn) {
        if (pool[0].id == knn[0].id) return 1.0;
        if (pool[0].dist == knn[0].dist) return 1.0;
        return 0;
    }

    static float EvaluateOneRatio (Neighbors const &pool, Neighbors const &knn) {
        return pool[0].dist / knn[0].dist;
    }

    static float EvaluateDelta (Neighbors const &pool, unsigned K) {
        unsigned c = 0;
        if (pool.size() < K) K = pool.size();
        for (unsigned i = 0; i < K; ++i) {
            if (pool[i].flag) ++c;
        }
        return float(c) / K;
    }


    struct Control {
        unsigned id;
        Neighbors neighbors;
    };

    // try insert nn into the list
    // the array addr must contain at least K+1 entries:
    //      addr[0..K-1] is a sorted list
    //      addr[K] is as output parameter
    // * if nn is already in addr[0..K-1], return K+1
    // * Otherwise, do the equivalent of the following
    //      put nn into addr[K]
    //      make addr[0..K] sorted
    //      return the offset of nn's index in addr (could be K)
    //
    // Special case:  K == 0
    //      addr[0] <- nn
    //      return 0
    static inline unsigned  __attribute__ ((noinline)) UpdateKnnList (Neighbor *addr, unsigned K, Neighbor nn) {
        // find the location to insert
        unsigned j;
        unsigned i = K;
        while (i > 0) {
            j = i - 1;
            if (addr[j].dist < nn.dist) break;
            if (addr[j].id == nn.id) return K + 1;
            i = j;
        }
        // i <= K-1
        j = K;
        while (j > i) {
            addr[j] = addr[j-1];
            --j;
        }
        addr[i] = nn;
        return i;
    }

    /*
    static inline unsigned UpdateKnnListNoCheckId (Neighbor *addr, unsigned K, Neighbor nn) {
        // find the location to insert
        unsigned K_1 = K - 1;
        if (nn.dist > addr[K_1].dist) return K;
        unsigned i = K_1;
        while (i > 0) {
            unsigned j = i - 1;
            if (addr[j].dist < nn.dist) break;
            i = j;
        }
        // i <= K-1
        unsigned j = K_1;
        while (j > i) {
            addr[j] = addr[j-1];
            --j;
        }
        addr[i] = nn;
        return i;
    }
    */

    void LinearSearch (IndexOracle const &oracle, unsigned i, unsigned K, vector<Neighbor> *pnns) {
        vector<Neighbor> nns(K+1);
        unsigned N = oracle.size();
        BOOST_VERIFY(N >= K + 1);
        Neighbor nn;
        nn.id = 0;
        nn.flag = true; // we don't really use this
        unsigned k = 0;
        while (nn.id < N) {
            if (nn.id != i) {
                nn.dist = oracle(i, nn.id);
                int r = UpdateKnnList(&nns[0], k, nn);
                if (k < K) ++k;
            }
            ++nn.id;
        }
        nns.resize(K);
        pnns->swap(nns);
    }

    void SearchOracle::search (unsigned K, unsigned *ids) const {
        vector<Neighbor> nns(K+1);
        unsigned N = size();
        BOOST_VERIFY(N >= K);
        for (unsigned k = 0; k < N; ++k) {
            UpdateKnnList(&nns[0], std::min(k, K), Neighbor(k, operator()(k)));
        }
        if (ids) {
            for (unsigned k = 0; k < K; ++k) {
                ids[k] = nns[k].id;
            }
        }
    }

    void GenerateControl (IndexOracle const &oracle, unsigned C, unsigned K, vector<Control> *pcontrols) {
        vector<Control> controls(C);
        {
            vector<unsigned> index(oracle.size());
            int i = 0;
            for (unsigned &v: index) {
                v = i++;
            }
            random_shuffle(index.begin(), index.end());
#pragma omp parallel for
            for (unsigned i = 0; i < C; ++i) {
                controls[i].id = index[i];
                LinearSearch(oracle, index[i], K, &controls[i].neighbors);
            }
        }
        pcontrols->swap(controls);
    }

    class KGraphConstructor: public KGraph {
        // The neighborhood structure maintains a pool of near neighbors of an object.
        // The neighbors are stored in the pool.  "n" (<="params.L") is the number of valid entries
        // in the pool, with the beginning "k" (<="n") entries sorted.
        struct Nhood { // neighborhood
            Lock lock;
            float radius;   // distance of interesting range
            float radiusM;
            Neighbors pool;
            unsigned L;     // # valid items in the pool,  L + 1 <= pool.size()
            unsigned M;     // we only join items in pool[0..M)
            bool found;     // helped found new NN in this round
            vector<unsigned> nn_old;
            vector<unsigned> nn_new;
            vector<unsigned> rnn_old;
            vector<unsigned> rnn_new;

            // only non-readonly method which is supposed to be called in parallel
            unsigned parallel_try_insert (unsigned id, float dist) {
                if (dist > radius) return pool.size();
                LockGuard guard(lock);
                unsigned l = UpdateKnnList(&pool[0], L, Neighbor(id, dist, true));
                if (l <= L) { // inserted
                    if (L + 1 < pool.size()) { // if l == L + 1, there's a duplicate
                        ++L;
                    }
                    else {
                        radius = pool[L-1].dist;
                    }
                }
                return l;
            }

            // join should not be conflict with insert
            template <typename C>
            void join (C callback) const {
                for (unsigned const i: nn_new) {
                    for (unsigned const j: nn_new) {
                        if (i < j) {
                            callback(i, j);
                        }
                    }
                    for (unsigned j: nn_old) {
                        callback(i, j);
                    }
                }
            }
        };

        IndexOracle const &oracle;
        IndexParams params;
        IndexInfo *pinfo;
        vector<Nhood> nhoods;
        size_t n_comps;

        void init ();
        void join ();
        void update ();
public:
        KGraphConstructor (IndexOracle const &o, IndexParams const &p, IndexInfo *r);
    };

    void KGraphConstructor::init () {
        unsigned N = oracle.size();
        unsigned seed = params.seed;
        mt19937 rng(seed);
        for (auto &nhood: nhoods) {
            nhood.nn_new.resize(params.S * 2);
            nhood.pool.resize(params.L+1);
            nhood.radius = numeric_limits<float>::max();
        }
#pragma omp parallel
        {
#ifdef _OPENMP
            mt19937 rng(seed ^ omp_get_thread_num());
#else
            mt19937 rng(seed);
#endif
            vector<unsigned> random(params.S + 1);
#pragma omp for
            for (unsigned n = 0; n < N; ++n) {
                auto &nhood = nhoods[n];
                Neighbors &pool = nhood.pool;
                GenRandom(rng, &nhood.nn_new[0], nhood.nn_new.size(), N);
                GenRandom(rng, &random[0], random.size(), N);
                nhood.L = params.S;
                nhood.M = params.S;
                unsigned i = 0;
                for (unsigned l = 0; l < nhood.L; ++l) {
                    if (random[i] == n) ++i;
                    auto &nn = nhood.pool[l];
                    nn.id = random[i++];
                    nn.dist = oracle(nn.id, n);
                    nn.flag = true;
                }
                sort(pool.begin(), pool.begin() + nhood.L);
            }
        }
    }

    void KGraphConstructor::join () {
        size_t cc = 0;
#pragma omp parallel for default(shared) schedule(dynamic, 100) reduction(+:cc)
        for (unsigned n = 0; n < oracle.size(); ++n) {
            size_t uu = 0;
            nhoods[n].found = false;
            nhoods[n].join([&](unsigned i, unsigned j) {
                    float dist = oracle(i, j);
                    ++cc;
                    unsigned r;
                    r = nhoods[i].parallel_try_insert(j, dist);
                    if (r < params.K) ++uu;
                    nhoods[j].parallel_try_insert(i, dist);
                    if (r < params.K) ++uu;
            });
            nhoods[n].found = uu > 0;
        }
        n_comps += cc;
    }

    void KGraphConstructor::update () {
        unsigned N = oracle.size();
        for (auto &nhood: nhoods) {
            nhood.nn_new.clear();
            nhood.nn_old.clear();
            nhood.rnn_new.clear();
            nhood.rnn_old.clear();
            nhood.radius = nhood.pool.back().dist;
        }
        //!!! compute radius2
#pragma omp parallel for
        for (unsigned n = 0; n < N; ++n) {
            auto &nhood = nhoods[n];
            if (nhood.found) {
                unsigned maxl = std::min(nhood.M + params.S, nhood.L);
                unsigned c = 0;
                unsigned l = 0;
                while ((l < maxl) && (c < params.S)) {
                    if (nhood.pool[l].flag) ++c;
                    ++l;
                }
                nhood.M = l;
            }
            BOOST_VERIFY(nhood.M > 0);
            nhood.radiusM = nhood.pool[nhood.M-1].dist;
        }
#pragma omp parallel for
        for (unsigned n = 0; n < N; ++n) {
            auto &nhood = nhoods[n];
            auto &nn_new = nhood.nn_new;
            auto &nn_old = nhood.nn_old;
            unsigned l = 0;
            for (unsigned l = 0; l < nhood.M; ++l) {
                auto &nn = nhood.pool[l];
                auto &nhood_o = nhoods[nn.id];  // nhood on the other side of the edge
                if (nn.flag) {
                    nn_new.push_back(nn.id);
                    if (nn.dist > nhood_o.radiusM) {
                        LockGuard guard(nhood_o.lock);
                        nhood_o.rnn_new.push_back(n);
                    }
                    nn.flag = false;
                }
                else {
                    nn_old.push_back(nn.id);
                    if (nn.dist > nhood_o.radiusM) {
                        LockGuard guard(nhood_o.lock);
                        nhood_o.rnn_old.push_back(n);
                    }
                }
            }
        }
        for (unsigned i = 0; i < N; ++i) {
            auto &nn_new = nhoods[i].nn_new;
            auto &nn_old = nhoods[i].nn_old;
            auto &rnn_new = nhoods[i].rnn_new;
            auto &rnn_old = nhoods[i].rnn_old;
            if (params.R && (rnn_new.size() > params.R)) {
                random_shuffle(rnn_new.begin(), rnn_new.end());
                rnn_new.resize(params.R);
            }
            nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
            if (params.R && (rnn_old.size() > params.R)) {
                random_shuffle(rnn_old.begin(), rnn_old.end());
                rnn_old.resize(params.R);
            }
            nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
        }
    }

    KGraphConstructor::KGraphConstructor (IndexOracle const &o, IndexParams const &p, IndexInfo *r)
            : oracle(o), params(p), pinfo(r), nhoods(o.size()), n_comps(0)
    {
        boost::timer::cpu_timer timer;
        params.check();
        unsigned N = oracle.size();
        BOOST_VERIFY(N > params.K); // has to be > because an object cannot be it's own neighbor.

        vector<Control> controls;
        cerr << "Generating control..." << endl;
        GenerateControl(oracle, params.controls, params.K, &controls);

        cerr << "Contructing graph..." << endl;
        cerr << "Initializing..." << endl;
        // initialize nhoods
        init();

        // iterate until converge
        float total = N * float(N - 1) / 2;
        IndexInfo info;
        info.stop_condition = IndexInfo::ITERATION;
        info.recall = 0;
        info.accuracy = numeric_limits<float>::max();
        info.cost = 0;
        info.iterations = 0;
        info.delta = 1.0;

        float update_time = 0;

        for (unsigned it = 0; (params.iterations <= 0) || (it < params.iterations); ++it) {
            ++info.iterations;
            join();
            {
                info.cost = n_comps / total;
                accumulator_set<float, stats<tag::mean>> one_exact;
                accumulator_set<float, stats<tag::mean>> one_approx;
                accumulator_set<float, stats<tag::mean>> one_recall;
                accumulator_set<float, stats<tag::mean>> recall;
                accumulator_set<float, stats<tag::mean>> accuracy;
                accumulator_set<float, stats<tag::mean>> M;
                accumulator_set<float, stats<tag::mean>> delta;
                for (auto const &nhood: nhoods) {
                    M(nhood.M);
                    delta(EvaluateDelta(nhood.pool, params.K));
                }
                for (auto const &c: controls) {
                    one_approx(nhoods[c.id].pool[0].dist);
                    one_exact(c.neighbors[0].dist);
                    one_recall(EvaluateOneRate(nhoods[c.id].pool, c.neighbors));
                    recall(EvaluateRecall(nhoods[c.id].pool, c.neighbors));
                    accuracy(EvaluateAccuracy(nhoods[c.id].pool, c.neighbors));
                }
                info.delta = mean(delta);
                info.recall = mean(recall);
                info.accuracy = mean(accuracy);
                info.times = timer.elapsed();
                cout << "iteration: " << info.iterations
                     << " recall: " << info.recall
                     << " accuracy: " << info.accuracy
                     << " cost: " << info.cost
                     << " M: " << mean(M)
                     << " delta: " << info.delta
                     << " time: " << info.times.wall / 1e9
                     << " one-recall: " << mean(one_recall)
                     << " one-ratio: " << mean(one_approx) / mean(one_exact)
                     << endl;
            }
            if (info.delta <= params.delta) {
                info.stop_condition = IndexInfo::DELTA;
                break;
            }
            if (info.recall >= params.recall) {
                info.stop_condition = IndexInfo::RECALL;
                break;
            }
            boost::timer::cpu_timer timer2;
            update();
            update_time += timer2.elapsed().wall / 1e9;
        }
        graph.resize(N);
        for (unsigned n = 0; n < N; ++n) {
            auto &knn = graph[n];
            auto const &pool = nhoods[n].pool;
            knn.resize(params.K);
            for (unsigned k = 0; k < params.K; ++k) {
                knn[k] = pool[k].id;
            }
        }
        info.times = timer.elapsed();
        if (pinfo) {
            *pinfo = info;
        }
        cerr << "UPDATE: " << update_time << endl;
    }

    void KGraph::build (IndexOracle const &oracle, IndexParams const &param, IndexInfo *info) {
        KGraphConstructor con(oracle, param, info);
        graph.swap(con.graph);
    }

    static char const *KGRAPH_MAGIC = "KNNGRAPH";
    static unsigned const KGRAPH_MAGIC_SIZE = 8;
    static uint32_t const VERSION_MAJOR = 1;
    static uint32_t const VERSION_MINOR = 0;

    void KGraph::load (string const &path) {
        BOOST_VERIFY(sizeof(unsigned) == sizeof(uint32_t));
        ifstream is(path.c_str(), ios::binary);
        char magic[KGRAPH_MAGIC_SIZE];
        uint32_t major;
        uint32_t minor;
        uint32_t N;
        is.read(magic, sizeof(magic));
        is.read(reinterpret_cast<char *>(&major), sizeof(major));
        is.read(reinterpret_cast<char *>(&minor), sizeof(minor));
        is.read(reinterpret_cast<char *>(&N), sizeof(N));
        BOOST_VERIFY(is);
        for (unsigned i = 0; i < KGRAPH_MAGIC_SIZE; ++i) {
            BOOST_VERIFY(KGRAPH_MAGIC[i] == magic[i]);
        }
        BOOST_VERIFY(major == VERSION_MAJOR && minor == VERSION_MINOR);
        graph.resize(N);
        for (auto &knn: graph) {
            unsigned K;
            is.read(reinterpret_cast<char *>(&K), sizeof(K));
            BOOST_VERIFY(is);
            knn.resize(K);
            is.read(reinterpret_cast<char *>(&knn[0]), K * sizeof(knn[0]));
        }
    }

    void KGraph::save (string const &path) {
        uint32_t N = graph.size();
        ofstream os(path.c_str(), ios::binary);
        os.write(KGRAPH_MAGIC, KGRAPH_MAGIC_SIZE);
        os.write(reinterpret_cast<char const *>(&VERSION_MAJOR), sizeof(VERSION_MAJOR));
        os.write(reinterpret_cast<char const *>(&VERSION_MINOR), sizeof(VERSION_MINOR));
        os.write(reinterpret_cast<char const *>(&N), sizeof(N));
        for (auto const &knn: graph) {
            uint32_t K = knn.size();
            os.write(reinterpret_cast<char const *>(&K), sizeof(K));
            os.write(reinterpret_cast<char const *>(&knn[0]), K * sizeof(knn[0]));
        }
    }

    void KGraph::search (SearchOracle const &oracle, SearchParams const &params, unsigned *ids, SearchInfo *pinfo) {
        /*
        BOOST_VERIFY(graph.size() <= oracle.size());
        boost::timer::cpu_timer timer;
        vector<Neighbor> knn(params.K);
        boost::dynamic_bitset<> flags(graph.size(), false);

        if (params.init) {
            for (unsigned k = 0; k < params.K; ++k) {
                knn[k].id = ids[k];
            }
        }
        else {
            unsigned seed = params.seed;
            if (seed == 0) seed = time(NULL);
            mt19937 rng(seed);
            vector<unsigned> random(params.K);
            GenRandom(rng, &random[0], params.K, graph.size());
            for (unsigned k = 0; k < params.K; ++k) {
                knn[k].id = random[k];
            }
        }
        for (unsigned k = 0; k < params.K; ++k) {
            flags[knn[k].id] = false;
            knn[k].flag = true;
            knn[k].dist = oracle(knn[k].id);
        }
        sort(knn.begin(), knn.end());

        unsigned updates = 0;
        unsigned n_comps = 0;
        unsigned k =  0;
        while (k < params.K) {
            unsigned nk = params.K;
            if (knn[k].flag) {
                knn[k].flag = false;
                unsigned cur = knn[k].id;
                for (unsigned id: graph[cur]) {
                    if (flags[id]) continue;
                    flags[id] = true;
                    ++n_comps;
                    Neighbor nn(id, oracle(id));
                    if (nn.dist < knn.back().dist) {
                        unsigned r = UpdateKnnList(&knn[0], params.K, nn);
                        if (r < nk) {
                            nk = r;
                            ++updates;
                        }
                    }
                }
            }
            if (nk <= k) {
                k = nk;
            }
            else {
                ++k;
            }
        }
        if (ids) {
            for (unsigned k = 0; k < params.K; ++k) {
                ids[k] = knn[k].id;
            }
        }
        if (pinfo) {
            pinfo->updates = updates;
            pinfo->cost = float(n_comps) / graph.size();
            pinfo->times = timer.elapsed();
        }
        */
    }
}

