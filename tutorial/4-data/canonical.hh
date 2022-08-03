#ifndef TUTORIAL_4_DATA_CANONICAL_HH
#define TUTORIAL_4_DATA_CANONICAL_HH

#include <flecsi/flog.hh>
#include <flecsi/topo/unstructured/interface.hh>

using namespace flecsi;

struct canon : topo::specialization<topo::unstructured, canon> {
  enum index_space { vertices, cells };
  using index_spaces = has<cells, vertices>;
  using connectivities = list<>;
  enum entity_list {};
  using entity_lists = list<>;

  static coloring color(std::string const &) {
    flog(info) << "invoking coloring" << std::endl;
    return {MPI_COMM_WORLD,
      1,
      {4, 2},
      {{{0, 1}, {0, 1}, {}, {}}, {{0, 1, 2, 3}, {0, 1, 2, 3}, {}, {}}},
      {{}},
      {{}}};
  }
};

#endif
