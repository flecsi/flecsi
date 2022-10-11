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
  using coloring_definition = base::coloring_definition;
  using color_map = base::color_map;

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
    auto vertices() {
      if constexpr(DM == interior) {
        const bool low = B::template is_low<mesh::vertices, A>();
        const bool high = B::template is_high<mesh::vertices, A>();
        const std::size_t start = B::template logical<mesh::vertices, A, 0>();
        const std::size_t end = B::template logical<mesh::vertices, A, 1>();

        return flecsi::topo::make_ids<mesh::vertices>(
          flecsi::util::iota_view<flecsi::util::id>(start + low, end - high));
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
      const bool low = B::template is_low<mesh::vertices, A>();
      const bool high = B::template is_high<mesh::vertices, A>();
      const std::size_t start = B::template logical<mesh::vertices, A, 0>();
      const std::size_t end = B::template logical<mesh::vertices, A, 1>();
      const std::size_t rng = (end - high) - (start + low);
      const bool parity = (0 == global_id<A>(start + low) % 2);

      // clang-format off
      const std::size_t pts =
        (0 == rng % 2) ? // even number of red and black points
          rng / 2 :
          parity == (0 == row % 2) ? // more red than black
            (rng + 1) / 2 :
            (rng - 1) / 2; // fewer red than black
      // clang-format on

      return flecsi::topo::make_stride_ids<mesh::vertices, 2>(
        flecsi::util::iota_view<flecsi::util::id>(0, pts),
        start + low,
        (row + parity) % 2);
    }

    template<axis A>
    FLECSI_INLINE_TARGET auto black(std::size_t row) const {
      return red<A>(row + 1);
    }

    double xdelta() {
      return (*(this->policy_meta_)).xdelta;
    }

    double ydelta() {
      return (*(this->policy_meta_)).ydelta;
    }

    double dxdy() {
      return xdelta() * ydelta();
    }

    template<axis A>
    double value(std::size_t i) {
      return (A == x_axis ? xdelta() : ydelta()) *
             (B::template offset<mesh::vertices, A, base::domain::global>() +
               i);
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
    coord hdepths{1, 1};
    coord bdepths{0, 0};
    std::vector<bool> periodic{false, false};

    auto axcm =
      flecsi::topo::narray_utils::make_color_maps(num_colors, axis_extents);
    coloring_definition cd{axcm, hdepths, bdepths, periodic};

    auto [nc, ne, pcs, partitions] =
      flecsi::topo::narray_utils::color(cd, MPI_COMM_WORLD);

    flog_assert(nc == flecsi::processes(),
      "current implementation is restricted to 1-to-1 mapping");

    coloring c;
    c.comm = MPI_COMM_WORLD;
    c.colors = nc;
    c.idx_colorings.emplace_back(std::move(pcs));
    c.partitions.emplace_back(std::move(partitions));
    return c;
  } // color

  /*--------------------------------------------------------------------------*
    Initialization.
   *--------------------------------------------------------------------------*/

  using grect = std::array<std::array<double, 2>, 2>;

  static void set_geometry(mesh::accessor<flecsi::rw> sm, grect const & g) {
    meta_data & md = sm.policy_meta_;
    double xdelta =
      std::abs(g[0][1] - g[0][0]) / (sm.size<x_axis, global>() - 1);
    double ydelta =
      std::abs(g[1][1] - g[1][0]) / (sm.size<y_axis, global>() - 1);

    md.xdelta = xdelta;
    md.ydelta = ydelta;
  }

  static void initialize(flecsi::data::topology_slot<mesh> & s,
    coloring const &,
    grect const & geometry) {
    flecsi::execute<set_geometry, flecsi::mpi>(s, geometry);
  } // initialize

}; // struct mesh

} // namespace poisson

#endif
