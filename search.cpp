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
#include <kgraph-data.h>

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
    unsigned K, P;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("data", po::value(&input_path), "input path")
    ("index", po::value(&index_path), "index path")
    ("query", po::value(&query_path), "query path")
    ("output", po::value(&output_path), "output path")
    ("init", po::value(&init_path), "init path")
    ("eval", po::value(&eval_path), "eval path")
    (",K", po::value(&K)->default_value(default_K), "")
    (",P", po::value(&P)->default_value(default_P), "")
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

    if (P < K) {
        P = K;
    }

    Matrix<value_type> data;
    Matrix<value_type> query;
    IndexMatrix result; //(query.size(), U);
    unsigned init = 0;

    data.load_lshkit(input_path);
    query.load_lshkit(query_path);
    if (init_path.size()) {
        result.load_lshkit(init_path);
        BOOST_VERIFY(result.size() == query.size());
        init = result.dim();
        BOOST_VERIFY(init >= K);
    }
    MatrixOracle<value_type, metric::l2sqr> oracle(data);
    float recall = 0;
    float cost = 0;
    float time = 0;
    if (vm.count("linear")) {
        boost::timer::auto_cpu_timer timer;
        result.resize(query.size(), K);
#pragma omp parallel for
        for (unsigned i = 0; i < query.size(); ++i) {
            oracle.query(query[i]).search(K, default_epsilon, result[i]);
        }
        cost = 1.0;
        time = timer.elapsed().wall / 1e9;
    }
    else {
        result.resize(query.size(), K);
        KGraph::SearchParams params;
        params.K = K;
        params.P = P;
        params.init = init;
        KGraph *kgraph = KGraph::create();
        kgraph->load(index_path.c_str());
        boost::timer::auto_cpu_timer timer;
        cerr << "Searching..." << endl;

#pragma omp parallel for reduction(+:cost)
        for (unsigned i = 0; i < query.size(); ++i) {
            KGraph::SearchInfo info;
            kgraph->search(oracle.query(query[i]), params, result[i], &info);
            cost += info.cost;
        }
        cost /= query.size();
        time = timer.elapsed().wall / 1e9;
        //cerr << "Cost: " << cost << endl;
        delete kgraph;
    }
    if (output_path.size()) {
        result.save_lshkit(output_path);
    }
    if (eval_path.size()) {
        IndexMatrix gs;
        gs.load_lshkit(eval_path);
        recall = AverageRecall(gs, result, K);
    }
    cout << "Time: " << time << " Recall: " << recall << " Cost: " << cost << endl;

    return 0;
}

