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
      MPI_COMM_WORLD,
      1, /* colors */
      /* process_colors */
      { /* over global processes */
        std::vector<Color>{ 0 }
      },
      /* color_peers */
      {
        0
      },
      /* peers */
      {
        {{}},
        {{}}
      },
      { /* over index spaces */
        { /* cells over global number of colors */
          4 /* partition size */
        },
        { /* vertices over global number of colors */
          2 /* partition size */
        }
      },
      { /* over index spaces */
        { /* cells over process colors */
          base::process_coloring{
            0,
            4,
            { /* index_coloring */
              {0, 1, 2, 3}, /* all */
              {0, 1, 2, 3}, /* owned */
              {0, 1, 2, 3}, /* exclusive */
              {}, /* shared */
              {} /* ghost */
            },
            {}, /* peers */
            {}, /* cnx_allocs */
            {} /* cnx_colorings */
          }
        },
        { /* vertices over process colors */
          base::process_coloring{
            0,
            2,
            { /* index_coloring */
              {0, 1}, /* all */
              {0, 1}, /* owned */
              {0, 1}, /* exclusive */
              {}, /* shared */
              {} /* ghost */
            },
            {}, /* peers */
            {}, /* cnx_allocs */
            {} /* cnx_colorings */
          }
        }
      }
    };
    // clang-format on
    return {};
  } // color
};

#endif
