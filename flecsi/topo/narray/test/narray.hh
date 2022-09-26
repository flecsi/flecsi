#ifndef FLECSI_TOPO_NARRAY_TEST_NARRAY_HH
#define FLECSI_TOPO_NARRAY_TEST_NARRAY_HH

#include "flecsi/data.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/narray/coloring_utils.hh"
#include "flecsi/topo/narray/interface.hh"

using namespace flecsi;

template<std::size_t D>
struct mesh {
  static_assert((D >= 1 && D <= 3), "Invalid dimension for testing !");

  using range = topo::narray_base::range;
  enum axis { x_axis, y_axis, z_axis };

  struct meta_data {
    double delta;
  };

  static constexpr Dimension dimension = D;

  /*--------------------------------------------------------------------------*
    Interface
   *--------------------------------------------------------------------------*/

  template<class B>
  struct interface : B {

    template<axis A, range SE = range::logical>
    std::size_t size() const {
      return B::template size<topo::elements, A, SE>();
    }

    template<axis A, range SE = range::logical>
    auto extents() const {
      return B::template extents<topo::elements, A, SE>();
    }

    template<axis A, range SE = range::logical>
    auto offset() const {
      return B::template offset<topo::elements, A, SE>();
    }
  };
}; // mesh

struct mesh1d : topo::specialization<topo::narray, mesh1d> {
  using meshbase = mesh<1>;

  using range = meshbase::range;
  using axis = meshbase::axis;
  using meta_data = meshbase::meta_data;

  static constexpr Dimension dimension = meshbase::dimension;

  template<auto>
  static constexpr PrivilegeCount privilege_count = 2;

  template<class B>
  using interface = meshbase::interface<B>;

  using axes = has<axis::x_axis>;
  using coord = base::coord;
  using coloring_definition = base::coloring_definition;

  static coloring color(std::vector<coloring_definition> index_definitions) {
    auto [colors, index_colorings] =
      topo::narray_utils::color(index_definitions, MPI_COMM_WORLD);

    flog_assert(colors == processes(),
      "current implementation is restricted to 1-to-1 mapping");

    coloring c;
    c.comm = MPI_COMM_WORLD;
    c.colors = colors;
    for(auto idx : index_colorings) {
      for(auto ic : idx) {
        c.idx_colorings.emplace_back(ic.second);
      }
    }
    return c;
  } // color

}; // mesh1d

struct mesh2d : topo::specialization<topo::narray, mesh2d> {
  using meshbase = mesh<2>;

  using range = meshbase::range;
  using axis = meshbase::axis;
  using meta_data = meshbase::meta_data;

  static constexpr Dimension dimension = meshbase::dimension;

  template<auto>
  static constexpr PrivilegeCount privilege_count = 2;

  template<class B>
  using interface = meshbase::interface<B>;

  using axes = has<axis::x_axis, axis::y_axis>;
  using coord = base::coord;
  using coloring_definition = base::coloring_definition;

  static coloring color(std::vector<coloring_definition> index_definitions) {
    auto [colors, index_colorings] =
      topo::narray_utils::color(index_definitions, MPI_COMM_WORLD);

    flog_assert(colors == processes(),
      "current implementation is restricted to 1-to-1 mapping");

    coloring c;
    c.comm = MPI_COMM_WORLD;
    c.colors = colors;
    for(auto idx : index_colorings) {
      for(auto ic : idx) {
        c.idx_colorings.emplace_back(ic.second);
      }
    }
    return c;
  } // color
}; // mesh2d

struct mesh3d : topo::specialization<topo::narray, mesh3d> {
  using meshbase = mesh<3>;

  using range = meshbase::range;
  using axis = meshbase::axis;
  using meta_data = meshbase::meta_data;

  static constexpr Dimension dimension = meshbase::dimension;

  template<auto>
  static constexpr PrivilegeCount privilege_count = 2;

  template<class B>
  using interface = meshbase::interface<B>;

  using axes = has<axis::x_axis, axis::y_axis, axis::z_axis>;
  using coord = base::coord;
  using coloring_definition = base::coloring_definition;

  static coloring color(std::vector<coloring_definition> index_definitions) {
    auto [colors, index_colorings] =
      topo::narray_utils::color(index_definitions, MPI_COMM_WORLD);

    flog_assert(colors == processes(),
      "current implementation is restricted to 1-to-1 mapping");

    coloring c;
    c.comm = MPI_COMM_WORLD;
    c.colors = colors;
    for(auto idx : index_colorings) {
      for(auto ic : idx) {
        c.idx_colorings.emplace_back(ic.second);
      }
    }
    return c;
  } // color
}; // mesh3d

#endif
