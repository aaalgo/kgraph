#include <random>
#include <iomanip>
#include <boost/tr1/random.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
#include "kgraph.h"
#include "kgraph-data.h"

using namespace std;
using namespace boost;
using namespace kgraph;

namespace po = boost::program_options; 

int main (int argc, char *argv[]) {
    string input_path;
    string output_path;
    unsigned K;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("input", po::value(&input_path), "input index path")
    ("output", po::value(&output_path), "output index path")
    (",K", po::value(&K)->default_value(20), "")
    ;
    po::options_description desc("Allowed options");
    desc.add(desc_visible);

    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || (vm.count("input") == 0) || (vm.count("output") == 0)) {
        cout << desc_visible << endl;
        return 0;
    }

    KGraph *index = KGraph::create();
    index->load(input_path.c_str());
    index->prune(K);
    index->save(output_path.c_str());
    delete index;
    return 0;
}

