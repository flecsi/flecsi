#ifndef POISSON_SPECIALIZATION_MESH_HH
#define POISSON_SPECIALIZATION_MESH_HH

#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>
#include <flecsi/topo/narray/coloring_utils.hh>
#include <flecsi/topo/narray/interface.hh>

#include "types.hh"

namespace poisson {

struct mesh : flecsi::topo::specialization<flecsi::topo::narray, mesh> {

  /*--------------------------------------------------------------------------*
    Policy Information.
   *--------------------------------------------------------------------------*/

  enum index_space { vertices };
  using index_spaces = has<vertices>;
  enum domain { interior, logical, all, global };
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

    template<axis A, domain DM = interior>
    std::size_t size() {
      if constexpr(DM == interior) {
        const bool low = B::template is_low<mesh::vertices, A>();
        const bool high = B::template is_high<mesh::vertices, A>();

        if(low && high) { /* degenerate */
          return size<A, logical>() - 2;
        }
        else if(low || high) {
          return size<A, logical>() - 1;
        }
        else { /* interior */
          return size<A, logical>();
        }
      }
      else if constexpr(DM == logical) {
        return B::template size<mesh::vertices, A, base::domain::logical>();
      }
      else if(DM == all) {
        return B::template size<mesh::vertices, A, base::domain::all>();
      }
      else if(DM == global) {
        return B::template size<mesh::vertices, A, base::domain::global>();
      }
    }

    template<axis A>
    FLECSI_INLINE_TARGET std::size_t global_id(std::size_t i) const {
      return B::template global_id<mesh::vertices, A>(i);
    }

    template<axis A, domain DM = interior>
    FLECSI_INLINE_TARGET auto vertices() const {
      if constexpr(DM == interior) {
        // The outermost layer is either ghosts or fixed boundaries:
        return flecsi::topo::make_ids<mesh::vertices>(
          flecsi::util::iota_view<flecsi::util::id>(
            1, B::template size<mesh::vertices, A, base::domain::all>() - 1));
      }
      else if constexpr(DM == logical) {
        return B::template range<mesh::vertices, A, base::domain::logical>();
      }
      else if(DM == all) {
        return B::template range<mesh::vertices, A, base::domain::all>();
      }
    }

    template<axis A>
    FLECSI_INLINE_TARGET auto red(std::size_t row) const {
      // The checkerboard extends across colors.  The (boundary) point with
      // global ID (0,0) is red; row is local, and 0 in the space of the
      // stride_view is the first interior vertex.
      return flecsi::util::stride_view(vertices<A>(),
        2,
        (global_id<(A == x_axis ? y_axis : x_axis)>(row) + global_id<A>(1)) %
          2);
    }

    template<axis A>
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

    template<axis A>
    FLECSI_INLINE_TARGET double value(std::size_t i) const {
      return (A == x_axis ? xdelta() : ydelta()) * global_id<A>(i);
    }

    template<axis A, boundary BD>
    bool is_boundary(std::size_t i) {

      auto const loff =
        B::template offset<mesh::vertices, A, base::domain::logical>();
      auto const lsize =
        B::template size<mesh::vertices, A, base::domain::logical>();
      const bool l = B::template is_low<mesh::vertices, A>();
      const bool h = B::template is_high<mesh::vertices, A>();

      if(l && h) { /* degenerate */
        if constexpr(BD == boundary::low) {
          return i == loff;
        }
        else {
          return i == (lsize + loff - 1);
        }
      }
      else if(l) {
        if constexpr(BD == boundary::low) {
          return i == loff;
        }
        else {
          return false;
        }
      }
      else if(h) {
        if constexpr(BD == boundary::low) {
          return false;
        }
        else {
          return i == (lsize + loff - 1);
        }
      }
      else { /* interior */
        return false;
      }
    } // is_boundary
  }; // struct interface

  /*--------------------------------------------------------------------------*
    Color Method.
   *--------------------------------------------------------------------------*/

  static coloring color(std::size_t num_colors, gcoord axis_extents) {
    index_definition idef;
    idef.axes = flecsi::topo::narray_utils::make_axes(num_colors, axis_extents);
    for(auto & a : idef.axes) {
      a.hdepth = 1;
    }

    flog_assert(idef.colors() == flecsi::processes(),
      "current implementation is restricted to 1-to-1 mapping");

    return {{idef}};
  } // color

  /*--------------------------------------------------------------------------*
    Initialization.
   *--------------------------------------------------------------------------*/

  using grect = std::array<std::array<double, 2>, 2>;

  static void set_geometry(mesh::accessor<flecsi::rw> sm, grect const & g) {
    sm.set_geometry(
      std::abs(g[0][1] - g[0][0]) / (sm.size<x_axis, global>() - 1),
      std::abs(g[1][1] - g[1][0]) / (sm.size<y_axis, global>() - 1));
  }

  static void initialize(flecsi::data::topology_slot<mesh> & s,
    coloring const &,
    grect const & geometry) {
    flecsi::execute<set_geometry, flecsi::mpi>(s, geometry);
  } // initialize

}; // struct mesh

} // namespace poisson

#endif
