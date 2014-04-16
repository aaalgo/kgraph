/* 
    Copyright (C) 2013,2014 Wei Dong <wdong@wdong.org>. All Rights Reserved.
*/

#ifndef KGRAPH_VALUE_TYPE
#define KGRAPH_VALUE_TYPE float
#endif

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

typedef KGRAPH_VALUE_TYPE value_type;

int main (int argc, char *argv[]) {
    string data_path;
    string input_path;
    string output_path;
    unsigned prune;
    unsigned D;
    unsigned skip;
    unsigned gap;

    bool lshkit = true;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("data", po::value(&data_path), "input path")
    ("input", po::value(&input_path), "input index path")
    ("output", po::value(&output_path), "output index path")
    ("prune", po::value(&prune)->default_value(PRUNE_LEVEL_1 & PRUNE_LEVEL_2), "prune level")
    ;

    po::options_description desc_hidden("Expert options");
    desc_hidden.add_options()
    ("dim,D", po::value(&D), "dimension, see format")
    ("skip", po::value(&skip)->default_value(0), "see format")
    ("gap", po::value(&gap)->default_value(0), "see format")
    ("raw", "read raw binary file, need to specify D.")
    ;

    po::options_description desc("Allowed options");
    desc.add(desc_visible).add(desc_hidden);

    po::positional_options_description p;
    p.add("data", 1);
    p.add("input", 1);
    p.add("output", 1);

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("raw") == 1) {
        lshkit = false;
    }

    if (vm.count("help")
            || vm.count("input") == 0) {
        cout << "Usage: prune [OTHER OPTIONS]... DATA INPUT [OUTPUT]" << endl;
        cout << desc_visible << endl;
        cout << desc_hidden << endl;
        return 0;
    }

    if (lshkit) {
        static const unsigned LSHKIT_HEADER = 3;
        ifstream is(data_path.c_str(), ios::binary);
        unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
        is.read((char *)header, sizeof header);
        BOOST_VERIFY(is);
        BOOST_VERIFY(header[0] == sizeof(value_type));
        is.close();
        D = header[2];
        skip = LSHKIT_HEADER * sizeof(unsigned);
        gap = 0;
    }

    Matrix<value_type> data;
    data.load(data_path, D, skip, gap);

    MatrixOracle<value_type, metric::l2sqr> oracle(data);
    KGraph *kgraph = KGraph::create(); //(oracle, params, &info);
    kgraph->load(input_path.c_str());
    {
        auto_cpu_timer timer;
        kgraph->prune(oracle, prune);
    }
    if (output_path.size()) {
        kgraph->save(output_path.c_str());
    }
    delete kgraph;

    return 0;
}

