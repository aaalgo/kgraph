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
    string output_path;
    KGraph::IndexParams params;
    unsigned D;
    unsigned skip;
    unsigned gap;
    unsigned synthetic;
    float noise;

    bool lshkit = true;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("version,v", "print version information.")
    ("data", po::value(&data_path), "input path")
    ("output", po::value(&output_path), "output path")
    (",K", po::value(&params.K)->default_value(default_K), "number of nearest neighbor")
    ("controls,C", po::value(&params.controls)->default_value(default_controls), "number of control pounsigneds")
    ;

    po::options_description desc_hidden("Expert options");
    desc_hidden.add_options()
    ("iterations,I", po::value(&params.iterations)->default_value(default_iterations), "")
    (",S", po::value(&params.S)->default_value(default_S), "")
    (",R", po::value(&params.R)->default_value(default_R), "")
    (",L", po::value(&params.L)->default_value(default_L), "")
    ("delta", po::value(&params.delta)->default_value(default_delta), "")
    ("recall", po::value(&params.recall)->default_value(default_recall), "")
    ("prune", po::value(&params.prune)->default_value(default_prune), "")
    ("noise", po::value(&noise)->default_value(0), "noise")
    ("seed", po::value(&params.seed)->default_value(default_seed), "")
    ("dim,D", po::value(&D), "dimension, see format")
    ("skip", po::value(&skip)->default_value(0), "see format")
    ("gap", po::value(&gap)->default_value(0), "see format")
    ("raw", "read raw binary file, need to specify D.")
    ("synthetic", po::value(&synthetic)->default_value(0), "generate synthetic data, for performance evaluation only, specify number of points")
    ;

    po::options_description desc("Allowed options");
    desc.add(desc_visible).add(desc_hidden);

    po::positional_options_description p;
    p.add("data", 1);
    p.add("output", 1);

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("raw") == 1) {
        lshkit = false;
    }

    if (vm.count("version")) {
        cout << "KGraph version " << KGraph::version() << endl;
        return 0;
    }

    if (vm.count("help")
            || (synthetic && (vm.count("dim") == 0 || vm.count("data")))
            || (!synthetic && (vm.count("data") == 0 || (vm.count("dim") == 0 && !lshkit)))) {
        cout << "Usage: index [OTHER OPTIONS]... INPUT [OUTPUT]" << endl;
        cout << desc_visible << endl;
        cout << desc_hidden << endl;
        return 0;
    }

    if (params.S == 0) {
        params.S = params.K;
    }

    if (lshkit && (synthetic == 0)) {   // read dimension information from the data file
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
    if (synthetic) {
        if (!std::is_floating_point<value_type>::value) {
            throw runtime_error("synthetic data not implemented for non-floating-point values.");
        }
        data.resize(synthetic, D);
        cerr << "Generating synthetic data..." << endl;
        default_random_engine rng(params.seed);
        uniform_real_distribution<double> distribution(-1.0, 1.0);
        data.zero(); // important to do that
        for (unsigned i = 0; i < synthetic; ++i) {
            value_type *row = data[i];
            for (unsigned j = 0; j < D; ++j) {
                row[j] = distribution(rng);
            }
        }
    }
    else {
        data.load(data_path, D, skip, gap);
    }
    if (noise != 0) {
        if (!std::is_floating_point<value_type>::value) {
            throw runtime_error("noise injection not implemented for non-floating-point value.");
        }
        tr1::ranlux64_base_01 rng;
        double sum = 0, sum2 = 0;
        for (unsigned i = 0; i < data.size(); ++i) {
            for (unsigned j = 0; j < data.dim(); ++j) {
                value_type v = data[i][j];
                sum += v;
                sum2 += v * v;
            }
        }
        double total = double(data.size()) * data.dim();
        double avg2 = sum2 / total, avg = sum / total;
        double dev = sqrt(avg2 - avg * avg);
        cerr << "Adding Gaussian noise w/ " << noise << "x sigma(" << dev << ")..." << endl;
        boost::normal_distribution<double> gaussian(0, noise * dev);
        for (unsigned i = 0; i < data.size(); ++i) {
            for (unsigned j = 0; j < data.dim(); ++j) {
                data[i][j] += gaussian(rng);
            }
        }
    }

    MatrixOracle<value_type, metric::l2sqr> oracle(data);
    KGraph::IndexInfo info;
    KGraph *kgraph = KGraph::create(); //(oracle, params, &info);
    {
        auto_cpu_timer timer;
        kgraph->build(oracle, params, &info);
        cerr << info.stop_condition << endl;
    }
    if (output_path.size()) {
        kgraph->save(output_path.c_str());
    }
    delete kgraph;

    return 0;
}

