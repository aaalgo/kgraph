/* 
    Copyright (C) 2010,2011 Wei Dong <wdong.pku@gmail.com>. All Rights Reserved.

    DISTRIBUTION OF THIS PROGRAM IN EITHER BINARY OR SOURCE CODE FORM MUST BE
    PERMITTED BY THE AUTHOR.
*/

#include <iostream>
#include <boost/timer/timer.hpp>
#include <boost/program_options.hpp>
#include <flann/flann.h>
#include <kgraph.h>
#include <kgraph-data.h>

using namespace std;
using namespace boost;
namespace po = boost::program_options; 

int main (int argc, char *argv[]) {
    flann::SearchParams params;
    params.cores = 0;
    string data_path;
    string index_path;
    string query_path;
    string output_path;
    string eval_path;
    unsigned K;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("data", po::value(&data_path), "data path")
    ("index", po::value(&index_path), "index path")
    ("query", po::value(&query_path), "query path")
    ("output", po::value(&output_path), "output_path")
    ("eval", po::value(&eval_path), "eval path")
    (",K", po::value(&K)->default_value(10), "")
    ("checks,C", po::value(&params.checks), "")
    ("eps", po::value(&params.eps), "")
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
        cout << "flann_search [...] <data> <index> <query> [output]" << endl;
        cout << desc_visible << endl;
        return 0;
    }


    kgraph::Matrix<float, 1> data;
    kgraph::Matrix<float, 1> query;
    data.load_lshkit(data_path);
    query.load_lshkit(query_path);

    kgraph::Matrix<unsigned,1> result(query.size(), K);
    kgraph::Matrix<float,1> dists(query.size(), K);

    flann::Matrix<float> fdata(data[0], data.size(), data.dim());
    flann::Matrix<float> fquery(query[0], query.size(), query.dim());
    flann::Matrix<int> fresult(reinterpret_cast<int *>(result[0]), query.size(), K);
    flann::Matrix<float> fdists(dists[0], query.size(), K);

    boost::timer::auto_cpu_timer timer;
    cerr << "Loading..." << endl;
    flann::Index<flann::L2<float>> index(fdata, flann::SavedIndexParams(index_path));
    timer.stop();
    timer.report();
    timer.start();
    cerr << "Searching..." << endl;
    index.knnSearch(fquery, fresult, fdists, K, params);
    timer.stop();
    timer.report();
    float time = timer.elapsed().wall / 1e9;
    if (output_path.size()) {
        result.save_lshkit(output_path);
    }
    float recall = 0;
    if (eval_path.size()) {
        flann::L2<float> l2;
        kgraph::Matrix<unsigned> gs; // gold standard
        gs.load_lshkit(eval_path);
        BOOST_VERIFY(gs.dim() >= K);
        BOOST_VERIFY(gs.size() >= query.size());
        kgraph::Matrix<float> gs_dist(query.size(), K);
        kgraph::Matrix<float> result_dist(query.size(), K);
#pragma omp parallel for
        for (unsigned i = 0; i < query.size(); ++i) {
            float *gs_dist_row = gs_dist[i];
            float *result_dist_row = result_dist[i];
            unsigned const *gs_row = gs[i];
            unsigned const *result_row = result[i];
            for (unsigned k = 0; k < K; ++k) {
                gs_dist_row[k] = l2(query[i], data[gs_row[k]], data.dim());
                result_dist_row[k] = l2(query[i], data[result_row[k]], data.dim());
            }
            sort(gs_dist_row, gs_dist_row + K);
            sort(result_dist_row, result_dist_row + K);
        }
        /*
        for (unsigned i = 0; i < 10; ++i) {
            for (unsigned j = 0;j < 10; ++j) {
                cout << ' ' << gs_dist[i][j] << ':' << result_dist[i][j];
            }
            cout << endl;
        }
        */
        recall = kgraph::AverageRecall(gs_dist, result_dist, K);
    }
    cout << "Time: " << time << " Recall: " << recall << endl;
    return 0;
}

