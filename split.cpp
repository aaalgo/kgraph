#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <boost/program_options.hpp>
#include <kgraph.h>
#include <kgraph-data.h>

using namespace std;

int main (int argc, char *argv[]) {
    string data_file;
    string postfix;
    unsigned Q;

    namespace po = boost::program_options; 
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("data,D", po::value(&data_file), "data file")
        (",Q", po::value(&Q)->default_value(1000), "")
        (",O", po::value(&postfix), "")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm); 

    if (vm.count("help") || (vm.count("data") < 1))
    {
        cout << desc;
        return 0;
    }

    kgraph::Matrix<float> data;
    data.load_lshkit(data_file);
    kgraph::Matrix<float> train(data.size() - Q, data.dim());
    kgraph::Matrix<float> test(Q, data.dim());

    vector<unsigned> idx(data.size());
    for (unsigned i = 0; i < idx.size(); ++i) idx[i] = i;
    random_shuffle(idx.begin(), idx.end());

    for (unsigned i = 0; i < test.size(); ++i) {
        copy(data[idx[i]], data[idx[i]] + data.dim(), test[i]);
    }
    for (unsigned i = 0; i < train.size(); ++i) {
        copy(data[idx[i + Q]], data[idx[i + Q]] + data.dim(), train[i]);
    }

    train.save_lshkit(postfix + ".data");
    test.save_lshkit(postfix + ".query");

    return 0;
}
