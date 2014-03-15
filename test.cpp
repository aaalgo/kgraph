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
    string input_path;
    string query_path;
    string output_path;
    string eval_path;
    unsigned K, P;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("data", po::value(&input_path), "input path")
    ("query", po::value(&query_path), "query path")
    ("eval", po::value(&eval_path), "eval path")
    (",K", po::value(&K)->default_value(default_K), "")
    (",P", po::value(&P)->default_value(default_P), "")
    ;

    po::options_description desc("Allowed options");
    desc.add(desc_visible);

    po::positional_options_description p;
    p.add("data", 1);
    p.add("query", 1);

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || vm.count("data") == 0 || vm.count("query") == 0) {
        cout << "search <data> <index> <query> [output]" << endl;
        cout << desc_visible << endl;
        return 0;
    }

    if (P < K) {
        P = K;
    }

    Matrix<value_type> data;
    Matrix<value_type> query;
    Matrix<unsigned> result;

    data.load_lshkit(input_path);
    query.load_lshkit(query_path);
    unsigned dim = data.dim();
    VectorOracle<Matrix<value_type>, value_type const*> oracle(data,
            [dim](value_type const *a, value_type const *b)
            {
                float r = 0;
                for (unsigned i = 0; i < dim; ++i) {
                    float v = float(a[i]) - (b[i]);
                    r += v * v;
                }
                return r;
            });
    float recall = 0;
    float cost = 0;
    float time = 0;

    result.resize(query.size(), K);
    KGraph::SearchParams params;
    params.K = K;
    params.P = P;
    KGraph *kgraph = KGraph::create();
    {
        KGraph::IndexParams params;
        kgraph->build(oracle, params, NULL);
    }

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
    delete kgraph;
    if (eval_path.size()) {
        Matrix<unsigned> gs;
        gs.load_lshkit(eval_path);
        BOOST_VERIFY(gs.dim() >= K);
        BOOST_VERIFY(gs.size() >= query.size());
        kgraph::Matrix<float> gs_dist(query.size(), K);
        kgraph::Matrix<float> result_dist(query.size(), K);
#pragma omp parallel for
        for (unsigned i = 0; i < query.size(); ++i) {
            auto const Q = oracle.query(query[i]);
            float *gs_dist_row = gs_dist[i];
            float *result_dist_row = result_dist[i];
            unsigned const *gs_row = gs[i];
            unsigned const *result_row = result[i];
            for (unsigned k = 0; k < K; ++k) {
                gs_dist_row[k] = Q(gs_row[k]); 
                result_dist_row[k] = Q(result_row[k]); 
            }
            sort(gs_dist_row, gs_dist_row + K);
            sort(result_dist_row, result_dist_row + K);
        }
        recall = AverageRecall(gs_dist, result_dist, K);
    }
    cout << "Time: " << time << " Recall: " << recall << " Cost: " << cost << endl;
    return 0;
}

