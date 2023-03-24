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

  template<class B>
  struct interface : B {

    auto cells() {
      return B::template entities<index_space::cells>();
    }

    template<index_space From>
    auto cells(topo::id<From> from) {
      return B::template entities<index_space::cells>(from);
    }

    auto vertices() {
      return B::template entities<index_space::vertices>();
    }

  }; // struct interface

  static coloring color(std::string const &) {
    flog(info) << "invoking coloring" << std::endl;

    // clang-format off
    return {
      /* number of colors */
      1,
      { /* over index spaces */
        {
          /* peers */
          {{}},
          { /* cells over global number of colors */
            4 /* partition size */
          },
          4,
          { /* cells over process colors */
            {
              4, /* entities */
              {}, /* shared */
              {}, /* ghost */
              {} /* cnx_allocs */
            }
          },
          {1}
        },
        {
          /* peers */
          {{}},
          { /* vertices over global number of colors */
            2 /* partition size */
          },
          2,
          { /* vertices over process colors */
            {
              2, /* entities */
              {}, /* shared */
              {}, /* ghost */
              {} /* cnx_allocs */
            }
          },
          {1}
        }
      },
      /* number of peers per color over all index spaces */
      {0}
    };
    // clang-format on
  } // color
};

#endif
