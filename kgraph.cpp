static char const *kgraph_version = KGRAPH_VERSION "-" KGRAPH_BUILD_NUMBER "," KGRAPH_BUILD_ID;

#include <omp.h>
#include <unordered_set>
#include <mutex>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <boost/timer/timer.hpp>
#define timer timer_for_boost_progress_t
#include <boost/progress.hpp>
#undef timer
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

    unsigned verbosity = default_verbosity;

    typedef boost::detail::spinlock Lock;
    typedef std::lock_guard<Lock> LockGuard;

    // generate size distinct random numbers < N
    template <typename RNG>
    static void GenRandom (RNG &rng, unsigned *addr, unsigned size, unsigned N) {
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

    // both pool and knn should be sorted in ascending order
    static float EvaluateRecall (Neighbors const &pool, Neighbors const &knn) {
        if (knn.empty()) return 1.0;
        unsigned found = 0;
        unsigned n_p = 0;
        unsigned n_k = 0;
        for (;;) {
            if (n_p >= pool.size()) break;
            if (n_k >= knn.size()) break;
            if (knn[n_k].dist < pool[n_p].dist) {
                ++n_k;
            }
            else if (knn[n_k].dist == pool[n_p].dist) {
                ++found;
                ++n_k;
                ++n_p;
            }
            else {
                cerr << "Distance is unstable." << endl;
                cerr << "Exact";
                for (auto const &p: knn) {
                    cerr << ' ' << p.id << ':' << p.dist;
                }
                cerr << endl;
                cerr << "Approx";
                for (auto const &p: pool) {
                    cerr << ' ' << p.id << ':' << p.dist;
                }
                cerr << endl;
                throw runtime_error("distance is unstable");
            }
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

    static float EvaluateOneRecall (Neighbors const &pool, Neighbors const &knn) {
        if (pool[0].dist == knn[0].dist) return 1.0;
        return 0;
    }

    static float EvaluateDelta (Neighbors const &pool, unsigned K) {
        unsigned c = 0;
        unsigned N = K;
        if (pool.size() < N) N = pool.size();
        for (unsigned i = 0; i < N; ++i) {
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
    static inline unsigned UpdateKnnList (Neighbor *addr, unsigned K, Neighbor nn) {
        // find the location to insert
        unsigned j;
        unsigned i = K;
        while (i > 0) {
            j = i - 1;
            if (addr[j].dist <= nn.dist) break;
            i = j;
        }
        // check for equal ID
        unsigned l = i;
        while (l > 0) {
            j = l - 1;
            if (addr[j].dist < nn.dist) break;
            if (addr[j].id == nn.id) return K + 1;
            l = j;
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

    void LinearSearch (IndexOracle const &oracle, unsigned i, unsigned K, vector<Neighbor> *pnns) {
        vector<Neighbor> nns(K+1);
        unsigned N = oracle.size();
        Neighbor nn;
        nn.id = 0;
        nn.flag = true; // we don't really use this
        unsigned k = 0;
        while (nn.id < N) {
            if (nn.id != i) {
                nn.dist = oracle(i, nn.id);
                UpdateKnnList(&nns[0], k, nn);
                if (k < K) ++k;
            }
            ++nn.id;
        }
        nns.resize(K);
        pnns->swap(nns);
    }

    unsigned SearchOracle::search (unsigned K, float epsilon, unsigned *ids, float *dists) const {
        vector<Neighbor> nns(K+1);
        unsigned N = size();
        unsigned L = 0;
        for (unsigned k = 0; k < N; ++k) {
            float k_dist = operator () (k);
            if (k_dist > epsilon) continue;
            UpdateKnnList(&nns[0], L, Neighbor(k, k_dist));
            if (L < K) ++L;
        }
        if (ids) {
            for (unsigned k = 0; k < L; ++k) {
                ids[k] = nns[k].id;
            }
        }
        if (dists) {
            for (unsigned k = 0; k < L; ++k) {
                dists[k] = nns[k].dist;
            }
        }
        return L;
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

    static char const *KGRAPH_MAGIC = "KNNGRAPH";
    static unsigned constexpr KGRAPH_MAGIC_SIZE = 8;
    static uint32_t constexpr VERSION_MAJOR = 2;
    static uint32_t constexpr VERSION_MINOR = 0;

    class KGraphImpl: public KGraph {
    protected:
        vector<unsigned> M;
        vector<vector<Neighbor>> graph;
    public:
        virtual ~KGraphImpl () {
        }
        virtual void load (char const *path) {
            BOOST_VERIFY(sizeof(unsigned) == sizeof(uint32_t));
            ifstream is(path, ios::binary);
            char magic[KGRAPH_MAGIC_SIZE];
            uint32_t major;
            uint32_t minor;
            uint32_t N;
            is.read(magic, sizeof(magic));
            is.read(reinterpret_cast<char *>(&major), sizeof(major));
            is.read(reinterpret_cast<char *>(&minor), sizeof(minor));
            if (major != VERSION_MAJOR) throw runtime_error("data version not supported.");
            is.read(reinterpret_cast<char *>(&N), sizeof(N));
            if (!is) runtime_error("error reading index file.");
            for (unsigned i = 0; i < KGRAPH_MAGIC_SIZE; ++i) {
                if (KGRAPH_MAGIC[i] != magic[i]) runtime_error("index corrupted.");
            }
            graph.resize(N);
            M.resize(N);
            for (unsigned i = 0; i < graph.size(); ++i) {
                auto &knn = graph[i];
                unsigned K;
                is.read(reinterpret_cast<char *>(&M[i]), sizeof(M[i]));
                is.read(reinterpret_cast<char *>(&K), sizeof(K));
                if (!is) runtime_error("error reading index file.");
                knn.resize(K);
                is.read(reinterpret_cast<char *>(&knn[0]), K * sizeof(knn[0]));
            }
        }

        virtual void save (char const *path) const {
            uint32_t N = graph.size();
            ofstream os(path, ios::binary);
            os.write(KGRAPH_MAGIC, KGRAPH_MAGIC_SIZE);
            os.write(reinterpret_cast<char const *>(&VERSION_MAJOR), sizeof(VERSION_MAJOR));
            os.write(reinterpret_cast<char const *>(&VERSION_MINOR), sizeof(VERSION_MINOR));
            os.write(reinterpret_cast<char const *>(&N), sizeof(N));
            for (unsigned i = 0; i < graph.size(); ++i) {
                auto const &knn = graph[i];
                uint32_t K = knn.size();
                os.write(reinterpret_cast<char const *>(&M[i]), sizeof(M[i]));
                os.write(reinterpret_cast<char const *>(&K), sizeof(K));
                os.write(reinterpret_cast<char const *>(&knn[0]), K * sizeof(knn[0]));
            }
        }

        virtual void build (IndexOracle const &oracle, IndexParams const &param, IndexInfo *info);

        /*
        virtual void prune (unsigned K) {
            for (auto &v: graph) {
                if (v.size() > K) {
                    v.resize(K);
                }
            }
            return;
            vector<vector<unsigned>> pruned(graph.size());
            vector<set<unsigned>> reachable(graph.size());
            vector<bool> added(graph.size());
            for (unsigned k = 0; k < K; ++k) {
#pragma omp parallel for
                for (unsigned n = 0; n < graph.size(); ++n) {
                    vector<unsigned> const &from = graph[n];
                    if (from.size() <= k) continue;
                    unsigned e = from[k];
                    if (reachable[n].count(e)) {
                        added[n] = false;
                    }
                    else {
                        pruned[n].push_back(e);
                        added[n] = true;
                    }
                }
                // expand reachable
#pragma omp parallel for
                for (unsigned n = 0; n < graph.size(); ++n) {
                    vector<unsigned> const &to = pruned[n];
                    set<unsigned> &nn = reachable[n];
                    if (added[n]) {
                        for (unsigned v: pruned[to.back()]) {
                            nn.insert(v);
                        }
                    }
                    for (unsigned v: to) {
                        if (added[v]) {
                            nn.insert(pruned[v].back());
                        }
                    }
                }
            }
            graph.swap(pruned);
        }
        */

        virtual unsigned search (SearchOracle const &oracle, SearchParams const &params, unsigned *ids, float *dists, SearchInfo *pinfo) const {
            if (graph.size() > oracle.size()) {
                throw runtime_error("dataset larger than index");
            }
            if (params.P >= graph.size()) {
                if (pinfo) {
                    pinfo->updates = 0;
                    pinfo->cost = 1.0;
                }
                return oracle.search(params.K, params.epsilon, ids, dists);
            }
            vector<Neighbor> knn(params.K + params.P +1);
            vector<Neighbor> results;
            boost::dynamic_bitset<> flags(graph.size(), false);

            if (params.init && params.T > 1) {
                throw runtime_error("when init > 0, T must be 1.");
            }

            unsigned seed = params.seed;
            unsigned updates = 0;
            if (seed == 0) seed = time(NULL);
            mt19937 rng(seed);
            unsigned n_comps = 0;
            for (unsigned trial = 0; trial < params.T; ++trial) {
                unsigned L = params.init;
                if (L == 0) {   // generate random starting points
                    vector<unsigned> random(params.P);
                    GenRandom(rng, &random[0], random.size(), graph.size());
                    for (unsigned s: random) {
                        if (!flags[s]) {
                            knn[L++].id = s;
                            //flags[s] = true;
                        }
                    }
                }
                else {          // user-provided starting points.
                    BOOST_VERIFY(ids);
                    BOOST_VERIFY(L < params.K);
                    for (unsigned l = 0; l < L; ++l) {
                        knn[l].id = ids[l];
                    }
                }
                for (unsigned k = 0; k < L; ++k) {
                    flags[knn[k].id] = true;
                    knn[k].flag = true;
                    knn[k].dist = oracle(knn[k].id);
                }
                sort(knn.begin(), knn.begin() + L);

                unsigned k =  0;
                while (k < L) {
                    unsigned nk = L;
                    if (knn[k].flag) {
                        knn[k].flag = false;
                        unsigned cur = knn[k].id;
                        //BOOST_VERIFY(cur < graph.size());
                        unsigned maxM = M[cur];
                        if (params.M > maxM) maxM = params.M;
                        auto const &neighbors = graph[cur];
                        if (maxM > neighbors.size()) {
                            maxM = neighbors.size();
                        }
                        for (unsigned m = 0; m < maxM; ++m) {
                            unsigned id = neighbors[m].id;
                            //BOOST_VERIFY(id < graph.size());
                            if (flags[id]) continue;
                            flags[id] = true;
                            ++n_comps;
                            float dist = oracle(id);
                            Neighbor nn(id, dist);
                            unsigned r = UpdateKnnList(&knn[0], L, nn);
                            BOOST_VERIFY(r <= L);
                            //if (r > L) continue;
                            if (L + 1 < knn.size()) ++L;
                            if (r < nk) {
                                nk = r;
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
                if (L > params.K) L = params.K;
                if (results.empty()) {
                    results.reserve(params.K + 1);
                    results.resize(L + 1);
                    copy(knn.begin(), knn.begin() + L, results.begin());
                }
                else {
                    // update results
                    for (unsigned l = 0; l < L; ++l) {
                        unsigned r = UpdateKnnList(&results[0], results.size() - 1, knn[l]);
                        if (r < results.size() /* inserted */ && results.size() < (params.K + 1)) {
                            results.resize(results.size() + 1);
                        }
                    }
                }
            }
            // check epsilon
            {
                for (unsigned l = 0; l < results.size(); ++l) {
                    if (results[l].dist > params.epsilon) {
                        results.resize(l);
                        break;
                    }
                }
            }
            unsigned L = results.size() - 1;
            BOOST_VERIFY(L <= params.K);
            // check epsilon
            if (ids) {
                for (unsigned k = 0; k < L; ++k) {
                    ids[k] = results[k].id;
                }
            }
            if (dists) {
                for (unsigned k = 0; k < L; ++k) {
                    dists[k] = results[k].dist;
                }
            }
            if (pinfo) {
                pinfo->updates = updates;
                pinfo->cost = float(n_comps) / graph.size();
            }
            return L;
        }

        virtual void get_nn (unsigned id, unsigned *nns, float *dist, unsigned *pM, unsigned *pL) const {
            BOOST_VERIFY(id < graph.size());
            auto const &v = graph[id];
            *pM = M[id];
            *pL = v.size();
            if (nns) {
                for (unsigned i = 0; i < v.size(); ++i) {
                    nns[i] = v[i].id;
                }
            }
            if (dist) {
                for (unsigned i = 0; i < v.size(); ++i) {
                    dist[i] = v[i].dist;
                }
            }
        }

        void prune1 () {
            for (unsigned i = 0; i < graph.size(); ++i) {
                if (graph[i].size() > M[i]) {
                    graph[i].resize(M[i]);
                }
            }
        }

        void prune2 () {
#if 0
            vector<vector<unsigned>> new_graph(graph.size());
            vector<unsigned> new_M(graph.size());
            vector<vector<unsigned>> reverse(graph.size());
            vector<unordered_set<unsigned>> todo(graph.size());
            unsigned L = 0;
            {
                cerr << "Level 2 Prune, stage 1/2 ..." << endl;
                progress_display progress(graph.size(), cerr);
                for (unsigned i = 0; i < graph.size(); ++i) {
                    if (graph[i].size() > L) L = graph[i].size();
                    todo[i] = unordered_set<unsigned>(graph[i].begin(), graph[i].end());
                    ++progress;
                }
            }
            {
                cerr << "Level 2 Prune, stage 2/2 ..." << endl;
                progress_display progress(L, cerr);
                for (unsigned l = 0; l < L; ++l) {
                    for (unsigned i = 0; i < graph.size(); ++i) {
                        if (todo[i].empty()) continue;
                        BOOST_VERIFY(l < graph[i].size());
                        unsigned T = graph[i][l];
                        if (todo[i].erase(T)) { // still there, need to be added
                            new_graph[i].push_back(T);
                            reverse[T].push_back(i);
                            // mark newly reachable nodes
                            for (auto n2: new_graph[T]) {
                                todo[i].erase(n2);
                            }
                            for (auto r: reverse[i]) {
                                todo[r].erase(T);
                            }
                        }
                        if (l + 1 == M[i]) {
                            new_M[i] = new_graph[i].size();
                        }
                    }
                    ++progress;
                }
            }
            graph.swap(new_graph);
            M.swap(new_M);
#endif
        }

        virtual void prune (IndexOracle const &oracle, unsigned level) {
            if (level & PRUNE_LEVEL_1) {
                prune1();
            }
            if (level & PRUNE_LEVEL_2) {
                prune2();
            }
        }
    };

    class KGraphConstructor: public KGraphImpl {
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

        void init () {
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
        void join () {
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
        void update () {
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

public:
        KGraphConstructor (IndexOracle const &o, IndexParams const &p, IndexInfo *r)
            : oracle(o), params(p), pinfo(r), nhoods(o.size()), n_comps(0)
        {
            boost::timer::cpu_timer timer;
            //params.check();
            unsigned N = oracle.size();
            if (N <= params.K) throw runtime_error("K larger than dataset size");

            vector<Control> controls;
            if (verbosity > 0) cerr << "Generating control..." << endl;
            GenerateControl(oracle, params.controls, params.K, &controls);
            if (verbosity > 0) cerr << "Initializing..." << endl;
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
                        one_recall(EvaluateOneRecall(nhoods[c.id].pool, c.neighbors));
                        recall(EvaluateRecall(nhoods[c.id].pool, c.neighbors));
                        accuracy(EvaluateAccuracy(nhoods[c.id].pool, c.neighbors));
                    }
                    info.delta = mean(delta);
                    info.recall = mean(recall);
                    info.accuracy = mean(accuracy);
                    info.M = mean(M);
                    auto times = timer.elapsed();
                    if (verbosity > 0) {
                        cerr << "iteration: " << info.iterations
                             << " recall: " << info.recall
                             << " accuracy: " << info.accuracy
                             << " cost: " << info.cost
                             << " M: " << info.M
                             << " delta: " << info.delta
                             << " time: " << times.wall / 1e9
                             << " one-recall: " << mean(one_recall)
                             << " one-ratio: " << mean(one_approx) / mean(one_exact)
                             << endl;
                    }
                }
                if (info.delta <= params.delta) {
                    info.stop_condition = IndexInfo::DELTA;
                    break;
                }
                if (info.recall >= params.recall) {
                    info.stop_condition = IndexInfo::RECALL;
                    break;
                }
                update();
            }
            M.resize(N);
            graph.resize(N);
            if (params.prune > 2) throw runtime_error("prune level not supported.");
            for (unsigned n = 0; n < N; ++n) {
                auto &knn = graph[n];
                M[n] = nhoods[n].M;
                auto const &pool = nhoods[n].pool;
                unsigned K = params.L;
                knn.resize(K);
                for (unsigned k = 0; k < K; ++k) {
                    knn[k].id = pool[k].id;
                    knn[k].dist = pool[k].dist;
                }
            }
            if (params.prune) {
                prune(o, params.prune);
            }
            if (pinfo) {
                *pinfo = info;
            }
        }

    };

    void KGraphImpl::build (IndexOracle const &oracle, IndexParams const &param, IndexInfo *info) {
        KGraphConstructor con(oracle, param, info);
        M.swap(con.M);
        graph.swap(con.graph);
    }

    KGraph *KGraph::create () {
        return new KGraphImpl;
    }

    char const* KGraph::version () {
        return kgraph_version;
    }
}

