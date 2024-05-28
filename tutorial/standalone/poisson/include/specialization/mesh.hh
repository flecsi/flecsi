#ifndef POISSON_SPECIALIZATION_MESH_HH
#define POISSON_SPECIALIZATION_MESH_HH

#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "types.hh"

namespace poisson {

struct mesh : flecsi::topo::specialization<flecsi::topo::narray, mesh> {

  /*--------------------------------------------------------------------------*
    Policy Information.
   *--------------------------------------------------------------------------*/

  enum index_space { vertices };
  using index_spaces = has<vertices>;
  enum axis { x_axis, y_axis };
  using axes = has<x_axis, y_axis>;
  enum boundary { low, high };

  using coord = base::coord;
  using gcoord = base::gcoord;
  using colors = base::colors;
  using hypercube = base::hypercube;
  using axis_definition = base::axis_definition;
  using index_definition = base::index_definition;

  struct meta_data {
    double xdelta;
    double ydelta;
  };

  static constexpr std::size_t dimension = 2;

  template<auto>
  static constexpr std::size_t privilege_count = 2;

  /*--------------------------------------------------------------------------*
    Interface.
   *--------------------------------------------------------------------------*/

  template<class B>
  struct interface : B {
    template<mesh::axis A>
    FLECSI_INLINE_TARGET base::axis_info axis() const {
      return B::template axis<mesh::vertices, A>();
    }

    template<mesh::axis A>
    FLECSI_INLINE_TARGET auto vertices() const {
      // The outermost layer is either ghosts or fixed boundaries:
      return flecsi::topo::make_ids<mesh::vertices>(
        flecsi::util::iota_view<flecsi::util::id>(
          1, axis<A>().layout.extent() - 1));
    }

    template<mesh::axis A>
    FLECSI_INLINE_TARGET auto red(std::size_t row) const {
      // The checkerboard extends across colors.  The (boundary) point with
      // global ID (0,0) is red; row is local, and 0 in the space of the
      // stride_view is the first interior vertex.
      return flecsi::util::stride_view(vertices<A>(),
        2,
        (axis<(A == x_axis ? y_axis : x_axis)>().global_id(row) +
          axis<A>().global_id(1)) %
          2);
    }

    template<mesh::axis A>
    FLECSI_INLINE_TARGET auto black(std::size_t row) const {
      return red<A>(row + 1);
    }

    void set_geometry(double x, double y) { // available if writable
      this->policy_meta() = {x, y};
    }

    FLECSI_INLINE_TARGET double xdelta() const {
      return this->policy_meta().xdelta;
    }

    FLECSI_INLINE_TARGET double ydelta() const {
      return this->policy_meta().ydelta;
    }

    FLECSI_INLINE_TARGET double dxdy() const {
      return xdelta() * ydelta();
    }

    template<mesh::axis A>
    FLECSI_INLINE_TARGET double value(std::size_t i) const {
      return (A == x_axis ? xdelta() : ydelta()) * axis<A>().global_id(i);
    }
  }; // struct interface

  /*--------------------------------------------------------------------------*
    Color Method.
   *--------------------------------------------------------------------------*/

  static coloring color(std::size_t num_colors, gcoord axis_extents) {
    index_definition idef;
    idef.axes = base::make_axes(num_colors, axis_extents);
    for(auto & a : idef.axes) {
      a.hdepth = 1;
    }

    return {{idef}};
  } // color

  /*--------------------------------------------------------------------------*
    Initialization.
   *--------------------------------------------------------------------------*/

  using grect = std::array<std::array<double, 2>, 2>;

  static void set_geometry(mesh::accessor<flecsi::rw> sm, grect const & g) {
    sm.set_geometry(
      std::abs(g[0][1] - g[0][0]) / (sm.axis<x_axis>().axis.extent - 1),
      std::abs(g[1][1] - g[1][0]) / (sm.axis<y_axis>().axis.extent - 1));
  }

  static void initialize(flecsi::data::topology_slot<mesh> & s,
    coloring const &,
    grect const & geometry) {
    flecsi::execute<set_geometry>(s, geometry);
  } // initialize

}; // struct mesh

} // namespace poisson

#endif
