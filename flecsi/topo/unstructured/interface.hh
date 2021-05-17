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
#include "flecsi/data/topology.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/core.hh"
#include "flecsi/topo/unstructured/coloring_utils.hh"
#include "flecsi/topo/unstructured/types.hh"
#include "flecsi/topo/utility_types.hh"
#include "flecsi/util/color_map.hh"
#include "flecsi/util/dcrs.hh"
#include "flecsi/util/set_utils.hh"
#include "flecsi/util/tuple_visitor.hh"

#include <map>
#include <utility>

namespace flecsi {
namespace topo {

/*----------------------------------------------------------------------------*
  Unstructured Topology.
 *----------------------------------------------------------------------------*/

template<typename Policy>
struct unstructured : unstructured_base,
                      with_ragged<Policy>,
                      with_meta<Policy> {
  using index_space = typename Policy::index_space;
  using index_spaces = typename Policy::index_spaces;

  template<Privileges>
  struct access;

  unstructured(coloring const & c)
    : with_ragged<Policy>(c[0].colors), with_meta<Policy>(c[0].colors),
      part_(make_partitions(c,
        index_spaces(),
        std::make_index_sequence<index_spaces::size>())),
      plan_(make_plans(c,
        index_spaces(),
        std::make_index_sequence<index_spaces::size>())),
      special_(c[0].colors) {
    allocate_connectivities(c, connect_);
#if 0
    make_subspaces(c, std::make_index_sequence<index_spaces::size>());
#endif
  }

  static inline const connect_t<Policy> connect_;
  static inline const field<util::id>::definition<array<Policy>> special_field;

  template<typename, typename>
  struct key_define;

  template<typename T, auto... SS>
  struct key_define<T, util::constants<SS...>> {
    using type = util::key_tuple<util::key_type<SS,
      typename field<T>::template definition<Policy, SS>>...>;
  };

  static inline const typename key_define<util::id, index_spaces>::type
    forward_map_;
  //  util::key_array<repartition, index_spaces> map_;

  util::key_array<repartitioned, index_spaces> part_;
  util::key_array<data::copy_plan, index_spaces> plan_;
  lists<Policy> special_;

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

  template<index_space S>
  auto const & forward_map() {
    return forward_map_.template get<S>();
  }

  template<index_space S>
  auto reverse_map() && {
    return std::move(reverse_map_.template get<S>());
  }

  template<typename Type,
    data::layout Layout,
    typename Topo,
    typename Topo::index_space Space>
  void ghost_copy(data::field_reference<Type, Layout, Topo, Space> const & f) {
    if constexpr(Layout == data::ragged)
      ; // TODO
    else
      plan_.template get<Space>().issue_copy(f.fid());
  }

private:
  void print(std::size_t v, std::size_t i) {
    flog(info) << "value: " << v << " index: " << i << std::endl;
  }

  template<auto... Value, std::size_t... Index>
  util::key_array<repartitioned, util::constants<Value...>> make_partitions(
    unstructured_base::coloring const & c,
    util::constants<Value...> /* index spaces to deduce pack */,
    std::index_sequence<Index...>) {
    (print(Value, Index), ...);
    flog_assert(c[0].idx_colorings.size() == sizeof...(Value),
      c[0].idx_colorings.size()
        << " sizes for " << sizeof...(Value) << " index spaces");
    return {{make_repartitioned<Policy, Value>(
      c[0].colors, make_partial<idx_size>(c[0].idx_colorings[Index]))...}};
  }

  template<index_space S>
  data::copy_plan make_plan(index_coloring const & ic, MPI_Comm const & comm) {
    constexpr PrivilegeCount NP = Policy::template privilege_count<S>;

    std::vector<std::size_t> num_intervals;
    std::vector<std::pair<std::size_t, std::size_t>> intervals;
    std::map<Color, std::vector<std::pair<std::size_t, std::size_t>>> points;

    auto const & fmd = forward_map_.template get<S>();
    execute<idx_itvls<NP>, mpi>(ic,
      num_intervals,
      intervals,
      points,
      fmd(*this),
      reverse_map_.template get<S>(),
      comm);

    // clang-format off
    auto dest_task = [&intervals, &comm](auto f) {
      execute<set_dests, mpi>(f, intervals, comm);
    };

    auto ptrs_task = [&points, &comm](auto f) {
      execute<set_ptrs<NP>, mpi>(
        f, points, comm);
    };
    // clang-format on

    return {*this, num_intervals, dest_task, ptrs_task, util::constant<S>()};
  }

  template<auto... Value, std::size_t... Index>
  util::key_array<data::copy_plan, util::constants<Value...>> make_plans(
    unstructured_base::coloring const & c,
    util::constants<Value...> /* index spaces to deduce pack */,
    std::index_sequence<Index...>) {
    flog_assert(c[0].idx_colorings.size() == sizeof...(Value),
      c[0].idx_colorings.size()
        << " sizes for " << sizeof...(Value) << " index spaces");
    return {{make_plan<Value>(c[0].idx_colorings[Index], c[0].comm)...}};
  }

  /*
    Allocate space for connectivity information for the given connectivity
    type. This method has a double pack expansion because each entity type may
    have connections to multiple other entity types, e.g., if there are
    entity types a, b, and c, a could connect to b and c, etc.

    @param VV Entity constants from the user's policy.
   */

  template<auto... VV, typename... TT>
  void allocate_connectivities(const unstructured_base::coloring & c,
    util::key_tuple<util::key_type<VV, TT>...> const & /* deduce pack */) {
    std::size_t entity = 0;
    (
      [&](TT const & row) { // invoked for each from-entity
        auto & cc = c[0].cnx_allocs[entity++];
        std::size_t is{0};
        for(auto & fd : row) { // invoked for each to-entity
          auto & p = this->ragged.template get_partition<VV>(fd.fid);
          execute<cnx_size>(cc[is++], p.sizes());
        }
      }(connect_.template get<VV>()),
      ...);
  }

#if 0
  template<std::size_t... Index>
  void make_subspaces(unstructured_base::coloring const & c,
    std::index_sequence<Index...>) {
   // auto & owned = owned_.get<>().get<>();
  //  execute<idx_subspaces>(c[Index], owned_.get<Index>
  }
#endif

  util::key_array<std::map<std::size_t, std::size_t>, index_spaces>
    reverse_map_;
}; // struct unstructured

/*----------------------------------------------------------------------------*
  Unstructured Access.
 *----------------------------------------------------------------------------*/

template<typename Policy>
template<Privileges Privileges>
struct unstructured<Policy>::access {
private:
  using entity_list = typename Policy::entity_list;
  template<const auto & Field>
  using accessor =
    data::accessor_member<Field, privilege_pack<privilege_merge(Privileges)>>;
  util::key_array<data::scalar_access<topo::resize::field>, index_spaces> size_;
  connect_access<Policy, Privileges> connect_;
  lists_t<accessor<special_field>, Policy> special_;

  template<index_space From, index_space To>
  auto & connectivity() {
    return connect_.template get<From>().template get<To>();
  }

  template<index_space From, index_space To>
  auto const & connectivity() const {
    return connect_.template get<From>().template get<To>();
  }

public:
  using subspace_list = std::size_t;
  // using subspace_list = typename unstructured::subspace_list;
  access() : connect_(unstructured::connect_) {}

  /*!
    Return an iterator to the parameterized index space.

    @tparam IndexSpace The index space identifier.
   */

  template<index_space IndexSpace>
  auto entities() {
    return make_ids<IndexSpace>(
      util::iota_view<util::id>(0, *size_.template get<IndexSpace>()));
  }

  /*!
    Return an iterator to the connectivity information for the parameterized
    index spaces.

    @tparam To   The connected index space.
    @tparam From The index space with connections.
   */

  template<index_space To, index_space From>
  auto entities(id<From> from) const {
    return make_ids<To>(connectivity<From, To>()[from]);
  }

  template<index_space I, entity_list L>
  auto special_entities() const {
    return make_ids<I>(special_.template get<I>().template get<L>().span());
  }

  template<index_space I, subspace_list L>
  auto subspace_entities() const {
#if 0
    if constexpr(L == owned) {
      return make_ids<I>(owned_.template get<I>().template get<L>()[0]);
    }
    else if(L == exclusive) {
      return make_ids<I>(exclusive_.template get<I>().template get<L>()[0]);
    }
    else if(L == shared) {
      return make_ids<I>(shared_.template get<I>().template get<L>()[0]);
    }
    else {
      return make_ids<I>(ghost_.template get<I>().template get<L>()[0]);
    }
#endif
  }

  template<class F>
  void send(F && f) {
    std::size_t i = 0;
    for(auto & a : size_)
      a.topology_send(
        f, [&i](unstructured & u) -> auto & { return u.part_[i++].sz; });

    connect_send(f, connect_, unstructured::connect_);
    lists_send(f, special_, special_field, &unstructured::special_);
  }
}; // struct unstructured<Policy>::access

template<>
struct detail::base<unstructured> {
  using type = unstructured_base;
}; // struct detail::base<unstructured>

} // namespace topo
} // namespace flecsi
