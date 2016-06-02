/* 
    Copyright (C) 2013,2014 Wei Dong <wdong@wdong.org>. All Rights Reserved.
*/

#include <sys/time.h>
#include <cctype>
#include <random>
#include <iomanip>
#include <type_traits>
#include <boost/timer/timer.hpp>
#include <boost/tr1/random.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include "kgraph.h"
#include "kgraph-data.h"

using namespace std;
using namespace boost;
using namespace boost::timer;
using namespace kgraph;

namespace po = boost::program_options; 

class DummyIndexOracle: public IndexOracle {
public:
    /// Returns the size of the dataset.
    virtual unsigned size () const {
        throw 0;
        return 0;
    }
    /// Computes similarity
    /**
     * 0 <= i, j < size() are the index of two objects in the dataset.
     * This method return the distance between objects i and j.
     */
    virtual float operator () (unsigned i, unsigned j) const {
        throw 0;
        return 0;
    }
};

int main (int argc, char *argv[]) {
    string input_path;
    string output_path;
    int format = 0;
    int prune = 0;

    po::options_description desc("General options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("input", po::value(&input_path), "")
    ("output", po::value(&output_path), "")
    ("prune", po::value(&prune), "")
    ("no-dist", "")
    ;

    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || vm.count("input") == 0) {
        cout << desc << endl;
        return 0;
    }

    if (vm.count("no-dist")) format |= kgraph::KGraph::FORMAT_NO_DIST;

    KGraph *kgraph = kgraph::KGraph::create();
    kgraph->load(input_path.c_str());
    if (vm.count("prune")) {
        DummyIndexOracle o;
        kgraph->prune(o, prune);
    }

    if (vm.count("output")) {
        kgraph->save(output_path.c_str(), format);
    }

    delete kgraph;

    return 0;
}

