#ifndef FLECSI_TOPO_NARRAY_TEST_NARRAY_HH
#define FLECSI_TOPO_NARRAY_TEST_NARRAY_HH

#include "flecsi/data.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/narray/interface.hh"

template<std::size_t D>
struct axes_helper {};

template<>
struct axes_helper<1> {
  enum axis { x_axis };
  using axes = flecsi::topo::help::has<x_axis>;
};

template<>
struct axes_helper<2> {
  enum axis { x_axis, y_axis };
  using axes = flecsi::topo::help::has<x_axis, y_axis>;
};

template<>
struct axes_helper<3> {
  enum axis { x_axis, y_axis, z_axis };
  using axes = flecsi::topo::help::has<x_axis, y_axis, z_axis>;
};

template<>
struct axes_helper<4> {
  enum axis { x_axis, y_axis, z_axis, t_axis };
  using axes = flecsi::topo::help::has<x_axis, y_axis, z_axis, t_axis>;
};

template<std::size_t D>
struct mesh : flecsi::topo::specialization<flecsi::topo::narray, mesh<D>>,
              axes_helper<D> {
  static_assert((D >= 1 && D <= 4), "Invalid dimension for testing !");

  using domain = typename mesh::base::domain;

  using axis = typename axes_helper<D>::axis;
  using axes = typename axes_helper<D>::axes;

  struct meta_data {
    double delta;
  };

  static constexpr flecsi::Dimension dimension = D;

  template<auto>
  static constexpr flecsi::PrivilegeCount privilege_count = 2;

  using coord = typename mesh::base::coord;
  using gcoord = typename mesh::base::gcoord;
  using axis_definition = typename mesh::base::axis_definition;
  using index_definition = typename mesh::base::index_definition;
  using coloring = typename mesh::base::coloring;
  static coloring color(index_definition const & idef) {
    return {{idef}};
  } // color

  /*--------------------------------------------------------------------------*
    Interface
   *--------------------------------------------------------------------------*/

  template<class B>
  struct interface : B {
    using M = std::array<flecsi::util::id, dimension>;

    template<axis A, domain DM = domain::logical>
    auto size() const {
      return B::template size<flecsi::topo::elements, A, DM>();
    }

    template<axis A, domain DM = domain::logical>
    auto range() const {
      return B::template range<flecsi::topo::elements, A, DM>();
    }

    template<axis A, domain DM = domain::logical>
    auto offset() const {
      return B::template offset<flecsi::topo::elements, A, DM>();
    }

    template<axis A>
    flecsi::util::gid global_id(flecsi::util::id lid) const {
      return B::template global_id<flecsi::topo::elements, A>(lid);
    }

    auto global_ids(M lid) const {
      std::array<flecsi::util::gid, dimension> val;
      if constexpr(D == 1) {
        val[0] = global_id<axis::x_axis>(lid[0]);
      }
      else if constexpr(D == 2) {
        val[0] = global_id<axis::x_axis>(lid[0]);
        val[1] = global_id<axis::y_axis>(lid[1]);
      }
      else {
        val[0] = global_id<axis::x_axis>(lid[0]);
        val[1] = global_id<axis::y_axis>(lid[1]);
        val[2] = global_id<axis::z_axis>(lid[2]);
      }
      return val;
    }

    template<typename S = flecsi::util::id, domain DM = domain::all>
    auto strides() {
      std::array<S, dimension> val;
      if constexpr(D == 1) {
        val[0] = size<axis::x_axis, DM>();
      }
      else if constexpr(D == 2) {
        val[0] = size<axis::x_axis, DM>();
        val[1] = size<axis::y_axis, DM>();
      }
      else {
        val[0] = size<axis::x_axis, DM>();
        val[1] = size<axis::y_axis, DM>();
        val[2] = size<axis::z_axis, DM>();
      }
      return val;
    }

    template<axis A, domain DM = domain::logical>
    std::array<flecsi::util::id, 2> axis_bounds() {
      flecsi::util::id lo = (DM == domain::logical || DM == domain::ghost_high)
                              ? offset<A, DM>()
                              : 0;
      return {lo, lo + size<A, DM>()};
    }

    template<domain DM = domain::logical>
    void bounds(M & lbnds, M & ubnds) {
      if constexpr(D == 1) {
        auto b = axis_bounds<axis::x_axis, DM>();
        lbnds[0] = b[0];
        ubnds[0] = b[1];
      }
      else if constexpr(D == 2) {
        auto bx = axis_bounds<axis::x_axis, DM>();
        auto by = axis_bounds<axis::y_axis, DM>();
        lbnds = {bx[0], by[0]};
        ubnds = {bx[1], by[1]};
      }
      else {
        auto bx = axis_bounds<axis::x_axis, DM>();
        auto by = axis_bounds<axis::y_axis, DM>();
        auto bz = axis_bounds<axis::z_axis, DM>();
        lbnds = {bx[0], by[0], bz[0]};
        ubnds = {bx[1], by[1], bz[1]};
      }
    }

    bool check_diag_bounds(M bnds) {
      return check_diag_bounds(bnds, axes());
    }

  private:
    template<auto... Axes>
    bool check_diag_bounds(M bnds, flecsi::util::constants<Axes...>) {
      return std::apply(
               [this](auto... bb) {
                 return ((bb >= this->offset<Axes, domain::ghost_high>() ||
                           bb < this->offset<Axes, domain::logical>()) +
                         ... + 0);
               },
               bnds) > 1;
    }
  };
}; // mesh

#endif
