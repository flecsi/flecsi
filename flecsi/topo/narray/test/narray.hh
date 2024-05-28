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
    template<axis A>
    auto axis() const {
      return B::template axis<flecsi::topo::elements, A>();
    }

  private:
    template<class F, auto... AA>
    auto map(F && f, flecsi::util::constants<AA...>) const {
      return std::array{f(axis<AA>())...};
    }
    template<class F>
    auto map(F && f) const {
      return map(f, axes());
    }
    using axis_info = flecsi::topo::narray_base::axis_info;

  public:
    using M = std::array<flecsi::util::id, dimension>;

    auto global_ids(M lid) const {
      return map(
        [p = lid.begin()](axis_info a) mutable { return a.global_id(*p++); });
    }

    flecsi::topo::narray_impl::linearize<D> linear() const {
      return {map([](axis_info a) { return a.layout.extent(); })};
    }
    flecsi::topo::narray_impl::linearize<D, flecsi::util::gid> glinear() const {
      return {map([](axis_info a) { return a.axis.extent; })};
    }
    flecsi::topo::narray_impl::traverse<D> range(bool g) const {
      if(g)
        return {ghost<0>(), ghost<1>()};
      return {logical<0>(), logical<1>()};
    }

    bool check_diag_bounds(M bnds) {
      return check_diag_bounds(bnds, axes());
    }

  private:
    template<auto... Axes>
    bool check_diag_bounds(M bnds, flecsi::util::constants<Axes...>) {
      return std::apply(
               [&](auto... bb) { // Clang 17.0.6 deems 'this' unused
                 return ([bb](axis_info a) {
                   return bb >= a.layout.logical<1>() ||
                          bb < a.layout.logical<0>();
                 }(axis<Axes>()) +
                         ... + 0);
               },
               bnds) > 1;
    }
    template<short E>
    M logical() const {
      return map([](axis_info a) { return a.layout.logical<E>(); });
    }
    template<short E>
    M ghost() const {
      return map([](axis_info a) { return a.layout.ghost<E>(); });
    }
  };
}; // mesh

#endif
