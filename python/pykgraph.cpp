#include <boost/assert.hpp>
#include <boost/python.hpp>
#include <kgraph.h>
#include <kgraph-data.h>

using namespace kgraph;
using namespace boost::python;

class PyKGraph {
    KGraph *index;
public:
    PyKGraph (): index(KGraph::create()) {
        BOOST_VERIFY(index);
    }
    ~PyKGraph () {
        if (index) delete index;
    }
    void load (char const *path) {
        index->load(path);
    }
    void save (char const *path) const {
        index->save(path);
    }
    /*
    void build (IndexOracle const &oracle, IndexParams const &params, IndexInfo *info) = 0;
    unsigned search (SearchParams const &params, unsigned *ids) {
    }
    */
};

BOOST_PYTHON_MODULE(pykgraph)
{
    class_<PyKGraph>("PyKGraph")
        .def("load", &PyKGraph::load)
        .def("save", &PyKGraph::save);
}
