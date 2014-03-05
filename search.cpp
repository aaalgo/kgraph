/* 
    Copyright (C) 2010,2011 Wei Dong <wdong.pku@gmail.com>. All Rights Reserved.

    DISTRIBUTION OF THIS PROGRAM IN EITHER BINARY OR SOURCE CODE FORM MUST BE
    PERMITTED BY THE AUTHOR.
*/
#ifndef KGRAPH_VALUE_TYPE
#define KGRAPH_VALUE_TYPE float
#endif


#include <cctype>
#include <type_traits>
#include <iostream>
#include <boost/timer/timer.hpp>
#include <boost/program_options.hpp>
#include <kgraph.h>
#include <kgraph-matrix.h>
#include <kgraph-util.h>

using namespace std;
using namespace boost;
using namespace kgraph;
namespace po = boost::program_options; 

typedef KGRAPH_VALUE_TYPE value_type;

int main (int argc, char *argv[]) {
    vector<string> params;
    string input_path;
    string index_path;
    string query_path;
    string output_path;
    string init_path;
    string eval_path;
    unsigned K, U;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("data", po::value(&input_path), "input path")
    ("index", po::value(&index_path), "index path")
    ("query", po::value(&query_path), "query path")
    ("output", po::value(&output_path), "output path")
    ("init", po::value(&init_path), "init path")
    ("eval", po::value(&eval_path), "eval path")
    (",K", po::value(&K)->default_value(20), "")
    (",U", po::value(&U)->default_value(100), "")
    ("linear", "")
    ;

    po::options_description desc("Allowed options");
    desc.add(desc_visible);

    po::positional_options_description p;
    p.add("data", 1);
    p.add("index", 1);
    p.add("query", 1);
    p.add("output", 1);

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || vm.count("data") == 0 || vm.count("index") == 0 || vm.count("query") == 0) {
        cout << "search <data> <index> <query> [output]" << endl;
        cout << desc_visible << endl;
        return 0;
    }

    if (U < K) {
        U = K;
    }

    Matrix<value_type> data;
    Matrix<value_type> query;
    IndexMatrix result; //(query.size(), U);

    data.load_lshkit(input_path);
    query.load_lshkit(query_path);
    if (init_path.size()) {
        result.load_lshkit(init_path);
        BOOST_VERIFY(result.size() == query.size());
        BOOST_VERIFY(result.dim() == U);
    }
    MatrixOracle<value_type, metric::l2sqr> oracle(data);
    if (vm.count("linear")) {
        boost::timer::auto_cpu_timer timer;
        result.resize(query.size(), K);
#pragma omp parallel for
        for (unsigned i = 0; i < query.size(); ++i) {
            oracle.query(query[i]).search(K, result[i]);
        }
    }
    else {
        result.resize(query.size(), U);
        KGraph::SearchParams params;
        params.K = U;
        if (init_path.size()) {
            params.init = true;
        }
        KGraph kgraph(index_path);
        boost::timer::auto_cpu_timer timer;
        cerr << "Searching..." << endl;

        float cost = 0;
#pragma omp parallel for reduction(+:cost)
        for (unsigned i = 0; i < query.size(); ++i) {
            KGraph::SearchInfo info;
            kgraph.search(oracle.query(query[i]), params, result[i], &info);
            cost += info.cost;
        }
        cost /= query.size();
        cerr << "Cost: " << cost << endl;
    }
    if (output_path.size()) {
        result.save_lshkit(output_path);
    }
    if (eval_path.size()) {
        IndexMatrix gs;
        gs.load_lshkit(eval_path);
        cerr << "Recall: " << AverageRecall(gs, result, K) << endl;
    }

    return 0;
}

