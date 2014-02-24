#include <random>
#include <iomanip>
#include <boost/tr1/random.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
#include "kgraph.h"
#include "kgraph-matrix.h"

using namespace std;
using namespace boost;
using namespace kgraph;

namespace po = boost::program_options; 

class Prune: public KGraph {
    struct Entry {
        unsigned id;
        float dist;
        bool keep;
    };
    vector<vector<Entry>> all;
    /*
     *   We should only keep an edge from  A---B
     *   only if not exist C s.t.
     *         |AC| < |AB|
     *   and   |BC| < |AB|
     */
    void prune () {
        cerr << "Pruning..." << endl;
        boost::timer::auto_cpu_timer timer;
        for (unsigned i = 0; i < all.size(); ++i) {
            auto &v = all[i];
            for (unsigned j = 0; j < v.size(); ++j) {
                // A: point i
                // C: v[j].id
                // B: some v[k].id for k > j
                // if there such B, then mark B false
                if (!v[j].keep) continue;
                unsigned A = i;
                unsigned C = v[j].id;
                float AC = v[j].dist;
                for (unsigned k = j + 1; k < v.size(); ++k) {
                    unsigned B = v[k].id;
                    unsigned AB = v[k].dist;
                    if (!v[k].keep)continue;
                    // check BC
                    for (auto const &e: all[B]) {
                        if (e.keep && e.id == C && e.dist < AB) {
                            v[k].keep = false;
                            break;
                        }
                        if (e.dist > AB) break;
                    }
                }
            }
        }
        for (unsigned i = 0; i < all.size(); ++i) {
            auto const &v = all[i];
            auto &u = graph[i];
            u.clear();
            for (auto const &e: v) {
                if (e.keep) {
                    u.push_back(e.id);
                }
            }
        }
    }
public:
    Prune (IndexOracle const &oracle, string const &path) {
        load(path);
        {
            cerr << "Recomputing distances..." << endl;
            boost::timer::auto_cpu_timer timer;
            all.resize(graph.size());
            for (unsigned i = 0; i < graph.size(); ++i) {
                auto const &ll = graph[i];
                auto &oo = all[i];
                oo.resize(ll.size());
                for (unsigned j = 0; j < ll.size(); ++j) {
                    oo[j].id = ll[j];
                    oo[j].dist = oracle(i, ll[j]);
                    oo[j].keep = true;
                }
            }
        }
        prune();
    }
};

int main (int argc, char *argv[]) {
    string data_path;
    string index_path;
    string output_path;
    unsigned D;
    unsigned skip, gap;
    bool lshkit = true;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("data", po::value(&data_path), "input path")
    ("index", po::value(&index_path), "index path")
    ("output", po::value(&output_path), "output path")
    ("dim,D", po::value(&D), "dimension, see format")
    ("skip", po::value(&skip)->default_value(0), "see format")
    ("gap", po::value(&gap)->default_value(0), "see format")
    ("raw", "read raw binary file, need to specify D.")
    ;
    po::options_description desc("Allowed options");
    desc.add(desc_visible);

    po::positional_options_description p;
    p.add("data", 1);
    p.add("index", 1);
    p.add("output", 1);

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("raw") == 1) {
        lshkit = false;
    }

    if (vm.count("help") || (vm.count("data") == 0) || (vm.count("index") == 0)) {
        cout << desc_visible << endl;
        return 0;
    }

    if (lshkit) {
        static const unsigned LSHKIT_HEADER = 3;
        ifstream is(data_path.c_str(), ios::binary);
        unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
        is.read((char *)header, sizeof header);
        BOOST_VERIFY(is);
        BOOST_VERIFY(header[0] == sizeof(float));
        is.close();
        D = header[2];
        skip = LSHKIT_HEADER * sizeof(unsigned);
        gap = 0;
    }

    Matrix<float> data;
    data.load(data_path, D, skip, gap);
    MatrixOracle<float, metric::l2sqr> oracle(data);

    Prune prune(oracle, index_path);
    if (vm.count("output")) {
        prune.save(output_path);
    }

    return 0;
}

