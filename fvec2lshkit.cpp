#include <iostream>
#include <fstream>
#include <vector>
#include <boost/assert.hpp>
#include <boost/program_options.hpp>

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
        cout << "fvec2lshkit <input> <output>" << endl;
        return 0;
    }
    ifstream is(input_path.c_str(), ios::binary);
    ofstream os(output_path.c_str(), ios::binary);

    int d = 4;

    os.write((char const *)&d, sizeof(d));

    is.read((char *)&d, sizeof(d));

    is.seekg(0, ios::end);

    int n = is.tellg() / (4 + d * 4);

    os.write((char const *)&n, sizeof(n));
    os.write((char const *)&d, sizeof(d));

    is.seekg(0, ios::beg);

    vector<float> vec(d + 1);
    for (int i = 0; i < n; ++i) {
        is.read((char *)&vec[0], sizeof(float) * vec.size());
        if (i == 0) {
            for (int j = 0; j < d; ++j) {
                cout << vec[j+1] << ' ';
            }
            cout << endl;
        }
        BOOST_VERIFY(*(int *)&vec[0] == d);
        os.write((char const *)&vec[1], sizeof(float) * d);
    }

    os.close();
    return 0;
}
