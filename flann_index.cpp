/* 
    Copyright (C) 2010,2011 Wei Dong <wdong.pku@gmail.com>. All Rights Reserved.

    DISTRIBUTION OF THIS PROGRAM IN EITHER BINARY OR SOURCE CODE FORM MUST BE
    PERMITTED BY THE AUTHOR.
*/

#include <boost/timer/timer.hpp>
#include <boost/program_options.hpp>
#include <flann/flann.h>
#include <kgraph.h>
#include <kgraph-data.h>

using namespace std;
using namespace boost;
using namespace flann;
namespace po = boost::program_options; 

int main (int argc, char *argv[]) {
    vector<string> pass;
    string input_path;
    string output_path;
    string algo;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("data", po::value(&input_path), "input path")
    ("output", po::value(&output_path), "output path")
    ("algorithm,A", po::value(&algo)->default_value("linear"), "linear, kdtree, kmeans, hier, comp, auto, lsh")
    (",P", po::value(&pass), "parameters passed to flann")
    ("verbose,v", "")
    ;

    po::options_description desc("Allowed options");
    desc.add(desc_visible);

    po::positional_options_description p;
    p.add("data", 1);
    p.add("output", 1);

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || vm.count("data") == 0) {
        cout << "flann_index [-A algo] [-P key=value ...] <data> [output]" << endl;
        cout << desc_visible << endl;
        cout << "Algorithm specific parameters are passed as '-P name%=value' (integer) or '-P name!=value' (float)." << endl;
        cout << "Available parameters and their default values:" << endl;
        cout << "linear" << endl;
        cout << "kdtree" << endl;
        cout << "           trees=4" << endl;
        cout << "kmeans" << endl;
        cout << "           branching=32" << endl;
        cout << "           iterations=11" << endl;
        cout << "           center_init=0 (RANDOM 0, GONZALES 1, KEMANSPP 2)" << endl;
        cout << "           cb_index=0.2" << endl;
        cout << "hier[archical]" << endl;
        cout << "           branching=32" << endl;
        cout << "           center_init=0 (RANDOM 0, GONZALES 1, KEMANSPP 2)" << endl;
        cout << "           trees=4" << endl;
        cout << "           leaf_max_size=100" << endl;
        cout << "comp[osite]" << endl;
        cout << "           trees=4" << endl;
        cout << "           branching=32" << endl;
        cout << "           iterations=11" << endl;
        cout << "           center_init=0 (RANDOM 0, GONZALES 1, KEMANSPP 2)" << endl;
        cout << "           cb_index=0.2" << endl;
        cout << "auto[tuned]" << endl;
        cout << "           target_precision=0.8" << endl;
        cout << "           build_weight=0.01" << endl;
        cout << "           memory_weight=0" << endl;
        cout << "           sample_fraction=0.1" << endl;
        cout << "lsh" << endl;
        cout << "           table_number=12" << endl;
        cout << "           key_size=20" << endl;
        cout << "           multi_probe_level=2" << endl;
        return 0;
    }

    kgraph::Matrix<float, 1> data;
    data.load_lshkit(input_path);

    IndexParams params;
    if (algo == "linear") {
        params = LinearIndexParams();
    }
    else if (algo == "kdtree") {
        params = KDTreeIndexParams();
    }
    else if (algo == "kmeans") {
        params = KMeansIndexParams();
    }
    else if (algo == "hier") {
        params = HierarchicalClusteringIndexParams();
    }
    else if (algo == "comp") {
        params = CompositeIndexParams();
    }
    else if (algo == "auto") {
        params = AutotunedIndexParams();
    }
    else if (algo == "lsh") {
        cerr << "LSH doesn't work with L2." << endl;
        BOOST_VERIFY(0);
        params = LshIndexParams();
    }
    else BOOST_VERIFY(0);

    for (string const &p: pass) {
        size_t off = p.find("=");
        string key, value;
        BOOST_VERIFY(off != p.npos);
        char type = p[off-1];
        key = p.substr(0, off-1);
        value = p.substr(off + 1);
        if (type == '%') {
            params[key] = lexical_cast<int>(value);
        }
        else if (type == '!') {
            params[key] = lexical_cast<float>(value);
        }
    }
    if (vm.count("verbose")) {
        for (auto const &p: params) {
            cerr << p.first << ": " << p.second << endl;
        }
    }

    flann::Matrix<float> fdata(data[0], data.size(), data.dim());

    boost::timer::auto_cpu_timer timer;
    flann::Index<flann::L2<float>> index(fdata, params);
    index.buildIndex();
    if (output_path.size()) {
        index.save(output_path);
    }

    return 0;
}

