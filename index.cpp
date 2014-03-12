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
#include "kgraph-matrix.h"

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
    ("data", po::value(&data_path), "input path")
    ("output", po::value(&output_path), "output path")
    (",K", po::value(&params.K)->default_value(default_K), "number of nearest neighbor")
    ("controls,C", po::value(&params.controls)->default_value(default_controls), "number of control pounsigneds")
    ;

    po::options_description desc_hidden("Expert options");
    desc_hidden.add_options()
    ("iterations,I", po::value(&params.iterations)->default_value(100), "expert")
    (",S", po::value(&params.S)->default_value(5), "expert, if S = 0 (default) then K will be used")
    (",R", po::value(&params.R)->default_value(0), "expert")
    (",L", po::value(&params.L)->default_value(50), "expert")
    ("delta", po::value(&params.delta)->default_value(0.005), "expert")
    ("noise", po::value(&noise)->default_value(0), "expert")
    ("recall", po::value(&params.recall)->default_value(0.98), "expert")
    ("seed", po::value(&params.seed)->default_value(1998), "")
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

    if (vm.count("help")
            || synthetic && (vm.count("dim") == 0 || vm.count("data"))
            || !synthetic && (vm.count("data") == 0 || (vm.count("dim") == 0 && !lshkit))) {
        cout << "Usage: index [OTHER OPTIONS]... INPUT [OUTPUT]" << endl;
        cout << "Construct k-nearest neighbor graph for Euclidean spaces using L2 distance as similarity measure..\n" << endl;
        cout << desc_visible << endl;
        cout << "Input Format:" << endl;
        cout << "  The INPUT file is parsed as a architecture-dependent binary file.  The initial <skip> bytes are skipped.  After that, every <D * sizeof(float)> bytes are read as a D-dimensional vector.  There could be an optional <gap>-byte gap between each vectors.  Therefore, the program expect the file to contain [size(INPUT)-skip]/[D*sizeof(float)+gap] vectors.\n"
                "  If the option \"--lshkit\" is specified, the initial 3*sizeof(unsigned) bytes are unsignederpreted as three 32-bit unsignedegers: sizeof(float), number of vectors in the file and the dimension.  The program then sets D = dimension, skip = 3 * sizeof(unsigned) and gap = 0.\n"  << endl;
        cout << "Output Format:" << endl;
        cout << "  Each input vector is assigned an serial ID (0, 1, ...) according to the order they appear in the input.  Each output line contains the ID of a pounsigned followed by the K IDs of its nearest neighbor.\n" << endl;
        cout << "Control:" << endl;
        cout << "  To measure the accuracy of the algorithm, <control> pounsigneds are randomly sampled, and their k-nearest neighbors are found with brute-force search.  The control is then used to measure the recall of the main algorithm.\n" << endl;
        cout << "Progress Report:" << endl;
        cout << "  The following parameters are reported after each iteration:\n"
                "  update: update rate of the K * N result entries.\n"
                "  recall: estimated recall, or 0 if no control is specified.\n"
                "  cost: number of similarity evaluate / [N*(N-1)/2, the brute force cost].\n";
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
    KGraph *kgraph = KGraph::make(); //(oracle, params, &info);
    {
        auto_cpu_timer timer;
        kgraph->build(oracle, params, &info);
    }
    if (output_path.size()) {
        kgraph->save(output_path.c_str());
    }
    delete kgraph;

    return 0;
}

