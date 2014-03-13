#include <iostream>
#include <fstream>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/assert.hpp>
#include <kgraph.h>
#include <kgraph-data.h>

using namespace std;

int main (int argc, char *argv[]) {
    string input_path;
    string output_path;

    namespace po = boost::program_options; 
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("input,I", po::value(&input_path), "")
        ("output,O", po::value(&output_path), "")
        ;
    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || (vm.count("input") < 1) || (vm.count("output") < 1))
    {
        cout << "lshkit2fvec <input> <output>" << endl;
        return 0;
    }
    kgraph::Matrix<float> data;
    data.load_lshkit(input_path);

    ofstream os(output_path.c_str(), ios::binary);
    int d = data.dim();

    for (unsigned i = 0; i < data.size(); ++i) {
        os.write((char const *)&d, sizeof(d));
        os.write((char const *)data[i], sizeof(float) * d);
    }

    os.close();
    return 0;
}
