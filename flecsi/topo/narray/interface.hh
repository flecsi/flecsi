/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

/*! @file */

#include "flecsi/data/accessor.hh"
#include "flecsi/data/copy_plan.hh"
#include "flecsi/data/layout.hh"
#include "flecsi/data/privilege.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/core.hh"
#include "flecsi/topo/index.hh"
#include "flecsi/topo/narray/coloring_utils.hh"
#include "flecsi/topo/narray/types.hh"
#include "flecsi/topo/utility_types.hh"
#include "flecsi/util/array_ref.hh"

#include <memory>
#include <utility>

namespace flecsi {
namespace topo {

/*----------------------------------------------------------------------------*
  Narray Topology.
 *----------------------------------------------------------------------------*/

template<typename Policy>
struct narray : narray_base, with_ragged<Policy>, with_meta<Policy> {

  using index_space = typename Policy::index_space;
  using index_spaces = typename Policy::index_spaces;
  using axis = typename Policy::axis;
  using axes = typename Policy::axes;
  using id = util::id;

  static constexpr Dimension dimension = Policy::dimension;

  template<Privileges>
  struct access;

  narray(coloring const & c)
    : with_ragged<Policy>(c.colors), with_meta<Policy>(c.colors),
      part_(make_partitions(c,
        index_spaces(),
        std::make_index_sequence<index_spaces::size>())),
      plans_(make_plans(c,
        index_spaces(),
        std::make_index_sequence<index_spaces::size>())) {
    init_meta(c);
    init_policy_meta(c);
  }

  struct meta_data {
    std::uint32_t orientation;

    using scoord = util::key_array<std::size_t, axes>;
    using shypercube = std::array<scoord, 2>;

    scoord global;
    scoord offset;
    scoord extents;
    shypercube logical;
    shypercube extended;
  };

  static inline const typename field<util::key_array<meta_data, index_spaces>,
    data::single>::template definition<meta<Policy>>
    meta_field;

  static inline const typename field<typename Policy::meta_data,
    data::single>::template definition<meta<Policy>>
    policy_meta_field;

  util::key_array<repartitioned, index_spaces> part_;
  util::key_array<std::vector<std::unique_ptr<data::copy_plan>>, index_spaces>
    plans_;

  Color colors() const {
    return part_.front().colors();
  }

  template<index_space S>
  data::region & get_region() {
    return part_.template get<S>();
  }

  template<index_space S>
  const data::partition & get_partition(field_id_t) const {
    return part_.template get<S>();
  }

  template<typename Type,
    data::layout Layout,
    typename Topo,
    typename Topo::index_space Space>
  void ghost_copy(data::field_reference<Type, Layout, Topo, Space> const & f) {
    for(auto const & p : plans_.template get<Space>()) {
      p->issue_copy(f.fid());
    }
  }

private:
  template<auto... Value, std::size_t... Index>
  util::key_array<repartitioned, util::constants<Value...>> make_partitions(
    narray_base::coloring const & c,
    util::constants<Value...> /* index spaces to deduce pack */,
    std::index_sequence<Index...>) {
    flog_assert(c.idx_colorings.size() == sizeof...(Value),
      c.idx_colorings.size()
        << " sizes for " << sizeof...(Value) << " index spaces");
    return {{make_repartitioned<Policy, Value>(
      c.colors, make_partial<idx_size>(c.partitions[Index]))...}};
  }

  template<index_space S>
  std::vector<std::unique_ptr<data::copy_plan>> make_plan(
    std::vector<process_color> const & vpc,
    repartitioned & p,
    MPI_Comm const & comm) {
    std::vector<std::unique_ptr<data::copy_plan>> plans;

    for(auto pc : vpc) {
      std::vector<std::size_t> num_intervals;
      execute<idx_itvls, mpi>(pc, num_intervals, comm);

      // clang-format off
      auto dest_task = [&pc, &comm](auto f) {
        execute<set_dests, mpi>(f, pc.intervals, comm);
      };

      auto ptrs_task = [&pc, &comm](auto f) {
        execute<set_ptrs<Policy::template privilege_count<S>>, mpi>(
          f, pc.points, comm);
      };
      // clang-format on

      plans.emplace_back(std::make_unique<data::copy_plan>(
        *this, p, num_intervals, dest_task, ptrs_task, util::constant<S>()));
    } // for

    return plans;
  }

  template<auto... Value, std::size_t... Index>
  util::key_array<std::vector<std::unique_ptr<data::copy_plan>>,
    util::constants<Value...>>
  make_plans(narray_base::coloring const & c,
    util::constants<Value...> /* index spaces to deduce pack */,
    std::index_sequence<Index...>) {
    flog_assert(c.idx_colorings.size() == sizeof...(Value),
      c.idx_colorings.size()
        << " sizes for " << sizeof...(Value) << " index spaces");
    return {
      {make_plan<Value>(c.idx_colorings[Index], part_[Index], c.comm)...}};
  }

  static void set_meta_idx(meta_data & md,
    std::vector<process_color> const & vpc) {
    // clang-format off
    static constexpr auto copy = [](const coord & c,
      typename meta_data::scoord & s) {
      const auto n = s.size();
      flog_assert(
        c.size() == n, "invalid #axes(" << c.size() << ") must be: " << n);
      std::copy_n(c.begin(), n, s.begin());
    };

    static constexpr auto copy2 = [](const hypercube & h,
      typename meta_data::shypercube & s) {
      for(auto i = h.size(); i--;)
        copy(h[i], s[i]);
    };
    // clang-format on

    md.orientation = vpc[0].orientation;
    copy(vpc[0].global, md.global);
    copy(vpc[0].offset, md.offset);
    copy(vpc[0].extents, md.extents);
    copy2(vpc[0].logical, md.logical);
    copy2(vpc[0].extended, md.extended);
  }

  template<auto... Value>
  static void visit_meta_is(util::key_array<meta_data, index_spaces> & m,
    narray_base::coloring const & c,
    util::constants<Value...> /* index spaces to deduce pack */) {
    std::size_t index{0};
    (set_meta_idx(m.template get<Value>(), c.idx_colorings[index++]), ...);
  }

  static void set_meta(typename field<util::key_array<meta_data, index_spaces>,
                         data::single>::template accessor<wo> m,
    narray_base::coloring const & c) {
    visit_meta_is(m, c, index_spaces());
  }

  void init_meta(narray_base::coloring const & c) {
    execute<set_meta, mpi>(meta_field(this->meta), c);
  }

  // Default initilaization for user policy meta data
  static void set_policy_meta(typename field<typename Policy::meta_data,
    data::single>::template accessor<wo>) {}

  void init_policy_meta(narray_base::coloring const &) {
    execute<set_policy_meta, mpi>(policy_meta_field(this->meta));
  }
}; // struct narray

/*----------------------------------------------------------------------------*
  Narray Access.
 *----------------------------------------------------------------------------*/

template<typename Policy>
template<Privileges Priv>
struct narray<Policy>::access {
  util::key_array<data::scalar_access<topo::resize::field, Priv>, index_spaces>
    size_;

  data::scalar_access<narray::meta_field, Priv> meta_;
  data::scalar_access<narray::policy_meta_field, Priv> policy_meta_;

  access() {}

  enum class range : std::size_t {
    logical,
    extended,
    all,
    boundary_low,
    boundary_high,
    ghost_low,
    ghost_high,
    global
  };

  using hypercubes = index::has<range::logical,
    range::extended,
    range::all,
    range::boundary_low,
    range::boundary_high,
    range::ghost_low,
    range::ghost_high,
    range::global>;

  template<index_space S>
  std::uint32_t orientation() {
    return meta_->template get<S>().orientation;
  }

  template<index_space S, axis A>
  std::size_t global() {
    return meta_->template get<S>().global.template get<A>();
  }

  template<index_space S, axis A>
  std::size_t offset() {
    return meta_->template get<S>().offset.template get<A>();
  }

  template<index_space S, axis A>
  std::size_t extents() {
    return meta_->template get<S>().extents.template get<A>();
  }

  template<index_space S>
  auto extents() {
    return meta_->template get<S>().extents;
  }

  template<index_space S, std::size_t P, axis A>
  std::size_t logical() {
    return meta_->template get<S>().logical[P].template get<A>();
  }

  template<index_space S, std::size_t P, axis A>
  std::size_t extended() {
    return meta_->template get<S>().extended[P].template get<A>();
  }

  template<index_space S, axis A>
  bool is_low() {
    return (orientation<S>() >> to_idx<A>() * 2) & narray_impl::low;
  }

  template<index_space S, axis A>
  bool is_high() {
    return (orientation<S>() >> to_idx<A>() * 2) & narray_impl::high;
  }

  template<axis A>
  bool is_interior() {
    return !is_low<A>() && !is_high<A>();
  }

  template<index_space S, axis A, range SE>
  std::size_t size() {
    static_assert(
      std::size_t(SE) < hypercubes::size, "invalid size identifier");
    if constexpr(SE == range::logical) {
      return logical<S, 1, A>() - logical<S, 0, A>();
    }
    else if constexpr(SE == range::extended) {
      return extended<S, 1, A>() - extended<S, 0, A>();
    }
    else if constexpr(SE == range::all) {
      return extents<S, A>();
    }
    else if constexpr(SE == range::boundary_low) {
      return logical<S, 0, A>() - extended<S, 0, A>();
    }
    else if constexpr(SE == range::boundary_high) {
      return extended<S, 1, A>() - logical<S, 1, A>();
    }
    else if constexpr(SE == range::ghost_low) {
      if(!is_low<S, A>())
        return logical<S, 0, A>();
      else
        return 0;
    }
    else if constexpr(SE == range::ghost_high) {
      if(!is_high<S, A>())
        return extents<S, A>() - logical<S, 1, A>();
      else
        return 0;
    }
    else if constexpr(SE == range::global) {
      return global<S, A>();
    }
  }

  template<index_space S, axis A, range SE>
  auto extents() {
    static_assert(
      std::size_t(SE) < hypercubes::size, "invalid extents identifier");

    if constexpr(SE == range::logical) {
      return make_ids<S>(
        util::iota_view<util::id>(logical<S, 0, A>(), logical<S, 1, A>()));
    }
    else if constexpr(SE == range::extended) {
      return make_ids<S>(
        util::iota_view<util::id>(extended<S, 0, A>(), extended<S, 1, A>()));
    }
    else if constexpr(SE == range::all) {
      return make_ids<S>(util::iota_view<util::id>(0, extents<S, A>()));
    }
    else if constexpr(SE == range::boundary_low) {
      return make_ids<S>(util::iota_view<util::id>(0, size<S, A, SE>()));
    }
    else if constexpr(SE == range::boundary_high) {
      return make_ids<S>(util::iota_view<util::id>(
        logical<S, 1, A>(), logical<S, 1, A>() + size<S, A, SE>()));
    }
    else if constexpr(SE == range::ghost_low) {
      return make_ids<S>(util::iota_view<util::id>(0, size<S, A, SE>()));
    }
    else if constexpr(SE == range::ghost_high) {
      return make_ids<S>(util::iota_view<util::id>(
        logical<S, 1, A>(), logical<S, 1, A>() + size<S, A, SE>()));
    }
    else {
      flog_error("invalid range");
    }
  }

  template<index_space S, axis A, range SE>
  std::size_t offset() {
    static_assert(
      std::size_t(SE) < hypercubes::size, "invalid offset identifier");
    if constexpr(SE == range::logical) {
      return logical<S, 0, A>();
    }
    else if constexpr(SE == range::extended) {
      return extended<S, 0, A>();
    }
    else if constexpr(SE == range::all) {
      return 0;
    }
    else if constexpr(SE == range::boundary_low) {
      return extended<S, 0, A>();
    }
    else if constexpr(SE == range::boundary_high) {
      return logical<S, 1, A>();
    }
    else if constexpr(SE == range::ghost_low) {
      return 0;
    }
    else if constexpr(SE == range::ghost_high) {
      return logical<S, 1, A>();
    }
    else if constexpr(SE == range::global) {
      return offset<S, A>();
    }
  }

  template<index_space S, typename T, Privileges P>
  auto mdspan(data::accessor<data::dense, T, P> const & a) {
    auto const s = a.span();
    return util::mdspan<typename decltype(s)::element_type, dimension>(
      s.data(), extents<S>());
  }
  template<index_space S, typename T, Privileges P>
  auto mdcolex(data::accessor<data::dense, T, P> const & a) {
    return util::mdcolex<
      typename std::remove_reference_t<decltype(a)>::element_type,
      dimension>(a.span().data(), extents<S>());
  }

  template<class F>
  void send(F && f) {
    std::size_t i{0};
    for(auto & a : size_)
      a.topology_send(
        f, [&i](narray & n) -> auto & { return n.part_[i++].sz; });

    meta_.topology_send(f, &narray::meta);
    policy_meta_.topology_send(f, &narray::meta);
  }

private:
  template<axis A>
  static constexpr std::uint32_t to_idx() {
    using axis_t = typename std::underlying_type_t<axis>;
    static_assert(std::is_convertible_v<axis_t, std::uint32_t>,
      "invalid axis type: cannot be converted to std::uint32_t");
    return static_cast<std::uint32_t>(A);
  }
}; // struct narray<Policy>::access

/*----------------------------------------------------------------------------*
  Define Base.
 *----------------------------------------------------------------------------*/

template<>
struct detail::base<narray> {
  using type = narray_base;
}; // struct detail::base<narray>

} // namespace topo
} // namespace flecsi
