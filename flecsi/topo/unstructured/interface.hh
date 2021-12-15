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
#include "flecsi/util/crs.hh"
#include "flecsi/util/set_utils.hh"

#include <map>
#include <memory>
#include <utility>

namespace flecsi {
namespace topo {
/// \defgroup unstructured Unstructured Mesh
/// Configurable unstructured mesh interface.
/// \ingroup topology
/// \{

/*----------------------------------------------------------------------------*
  Unstructured Topology.
 *----------------------------------------------------------------------------*/

template<typename Policy>
struct unstructured : unstructured_base,
                      with_ragged<Policy>,
                      with_meta<Policy> {

  friend Policy;

  /*--------------------------------------------------------------------------*
    Public types.
   *--------------------------------------------------------------------------*/

  using index_space = typename Policy::index_space;
  using index_spaces = typename Policy::index_spaces;

  template<Privileges>
  struct access;

  /*--------------------------------------------------------------------------*
    Constructor.
   *--------------------------------------------------------------------------*/

  unstructured(coloring const & c)
    : unstructured(
        [&c]() -> auto & {
          flog_assert(c.idx_spaces.size() == index_spaces::size,
            c.idx_spaces.size()
              << " sizes for " << index_spaces::size << " index spaces");
          return c;
        }(),
        index_spaces()) {}

  Color colors() const {
    return part_.front().colors();
  }

  template<index_space S>
  static constexpr std::size_t index = index_spaces::template index<S>;

  template<index_space S>
  data::region & get_region() {
    return part_.template get<S>();
  }

  template<index_space S>
  repartition & get_partition(field_id_t) {
    return part_.template get<S>();
  }

  template<typename Type, data::layout Layout, typename Policy::index_space S>
  void ghost_copy(data::field_reference<Type, Layout, Policy, S> const & f) {
    if constexpr(Layout == data::ragged)
      ; // TODO
    /*
      Use ghost information (ids?) to fill buffers to send to receiving colors.

      vector of vector of color
      vector over source color
        vector over destination colors

     */
    else
      plan_.template get<S>().issue_copy(f.fid());
  }

  template<index_space F, index_space T>
  auto & get_connectivity() {
    return connect_.template get<F>().template get<T>();
  }

  /*
    Return the forward map (local-to-global) for the given index space.
   */

  template<index_space S>
  auto const & forward_map() {
    return forward_maps_.template get<S>();
  }

private:
  /*
    Communication graph topology for ragged ghost updates.
   */

  struct ctopo : specialization<user, ctopo> {};

  /*
    Constant reference version:
    Return the reverse maps (global-to-local) for the given index space.
    The vector is over local process colors.

    Use like:
      auto const & maps = slot->reverse_map();
   */

  template<index_space S>
  auto reverse_map() & {
    return reverse_maps_.template get<S>();
  }

  /*
    Move version:
    Return the reverse maps (global-to-local) for the given index space.
    The vector is over local process colors.

    Use like:
      auto maps = std::move(slot)->reverse_map();
   */

  template<index_space S>
  auto reverse_map() && {
    return std::move(reverse_maps_.template get<index<S>>());
  }

  /*
    Define the partial closures that will be used to construct partitions
    when an instance of this topology is allocated.
   */

  // clang-format off
  template<auto... VV>
  unstructured(unstructured_base::coloring const & c,
    util::constants<VV...> /* index spaces to deduce pack */)
    : with_ragged<Policy>(c.colors),
      with_meta<Policy>(c.colors),
      ctopo_(c.color_peers),
      part_{
        {
        make_repartitioned<Policy, VV>(
          c.colors,
          make_partial<idx_size>(c.partitions[index<VV>]))...
        }
      },
      special_(c.colors),
      /* all data members need to be initialized before make_copy_plan */
      plan_{
        {
        make_copy_plan<VV>(c, c.comm)...
        }
      }
  {
    allocate_connectivities(c, connect_);
  }
  // clang-format on

  /*
    Construct copy plan for the given index space S.
   */

  template<index_space S>
  data::copy_plan make_copy_plan(unstructured_base::coloring const & c,
    MPI_Comm const & comm) {
    constexpr PrivilegeCount NP = Policy::template privilege_count<S>;

    std::vector<std::size_t> num_intervals(c.colors, 0);
    destination_intervals intervals;
    source_pointers pointers;

    // auto const & cg = cgraph_.template get<S>();
    auto const & fmd = forward_maps_.template get<S>();

    execute<idx_itvls<NP>, mpi>(c.idx_spaces[index<S>],
      c.process_colors,
      num_intervals,
      intervals,
      pointers,
      // cg(ctopo_),
      fmd(*this),
      reverse_maps_.template get<S>(),
      comm);

    // clang-format off
    auto dest_task = [&intervals, &comm](auto f) {
      execute<set_dests, mpi>(f, intervals, comm);
    };

    auto ptrs_task = [&](auto f) {
      execute<set_ptrs<NP>, mpi>(f, pointers, comm);
    };
    // clang-format on

    return {*this,
      part_.template get<S>(),
      num_intervals,
      dest_task,
      ptrs_task,
      util::constant<S>()};
  }

  /*
    Allocate space for connectivity information for the given connectivity
    type. This method has a double pack expansion because each entity type may
    have connections to multiple other entity types, e.g., if there are
    entity types a, b, and c, a could connect to b and c, etc.

    @param VV Entity constants from the user's policy.
    @param TT Connectivity targets.
   */

  template<auto... VV, typename... TT>
  void allocate_connectivities(const unstructured_base::coloring & c,
    util::key_tuple<util::key_type<VV, TT>...> const & /* deduce pack */) {
    (
      [&](TT const & row) { // invoked for each from-entity
        const std::vector<process_coloring> & pc = c.idx_spaces[index<VV>];
        for_each(
          [&](auto v) { // invoked for each to-entity
            auto & p = this->ragged.template get_partition<VV>(
              row.template get<v.value>().fid);
            execute<cnx_size, mpi>(pc, index<v.value>, p.sizes());
          },
          typename TT::keys());
      }(connect_.template get<VV>()),
      ...);
  }

  template<class F, auto... VV>
  static void for_each(F && f, util::constants<VV...> /* deduce pack */) {
    (f(util::constant<VV>()), ...);
  }

  /*--------------------------------------------------------------------------*
    Private data members.
   *--------------------------------------------------------------------------*/

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
    forward_maps_;

  typename ctopo::core ctopo_;
  static inline const util::key_array<
    typename field<unstructured_impl::cmap,
      data::ragged>::template definition<ctopo>,
    index_spaces>
    cgraph_;

  util::key_array<repartitioned, index_spaces> part_;
  lists<Policy> special_;
  util::key_array<std::vector<std::map<std::size_t, std::size_t>>, index_spaces>
    reverse_maps_;
  // Initializing this depends on the above:
  util::key_array<data::copy_plan, index_spaces> plan_;

}; // struct unstructured

/*----------------------------------------------------------------------------*
  Unstructured Access.
 *----------------------------------------------------------------------------*/

template<typename Policy>
template<Privileges Privileges>
struct unstructured<Policy>::access {

  /*
    FIXME: This should be private or protected.
   */

  template<class F>
  void send(F && f) {
    std::size_t i = 0;
    for(auto & a : size_)
      a.topology_send(
        f, [&i](unstructured & u) -> auto & { return u.part_[i++].sz; });

    connect_send(f, connect_, unstructured::connect_);
    lists_send(f, special_, special_field, &unstructured::special_);
  }

protected:
  using subspace_list = std::size_t;
  using entity_list = typename Policy::entity_list;
  access() : connect_(unstructured::connect_) {}

  /*!
    Return an index space as a range.

    @tparam IndexSpace The index space identifier.
   */

  template<index_space S>
  auto entities() const {
    return make_ids<S>(util::iota_view<util::id>(0, *size_.template get<S>()));
  }

  /*!
    Return a range of connectivity information for the parameterized
    index spaces.

    @tparam T The connected index space.
    @tparam F The index space with connections.
   */

  template<index_space T, index_space F>
  auto entities(id<F> from) const {
    return make_ids<T>(connectivity<F, T>()[from]);
  }

  template<index_space I, entity_list L>
  auto special_entities() const {
    return make_ids<I>(special_.template get<I>().template get<L>().span());
  }

private:
  template<index_space F, index_space T>
  auto & connectivity() {
    return connect_.template get<F>().template get<T>();
  }

  template<index_space F, index_space T>
  auto const & connectivity() const {
    return connect_.template get<F>().template get<T>();
  }

  /*--------------------------------------------------------------------------*
    Private data members.
   *--------------------------------------------------------------------------*/

  template<const auto & Field>
  using accessor =
    data::accessor_member<Field, privilege_pack<privilege_merge(Privileges)>>;
  util::key_array<data::scalar_access<topo::resize::field, Privileges>,
    index_spaces>
    size_;
  connect_access<Policy, Privileges> connect_;
  lists_t<accessor<special_field>, Policy> special_;

}; // struct unstructured<Policy>::access

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/

template<>
struct detail::base<unstructured> {
  using type = unstructured_base;
}; // struct detail::base<unstructured>

/// \}
} // namespace topo
} // namespace flecsi
