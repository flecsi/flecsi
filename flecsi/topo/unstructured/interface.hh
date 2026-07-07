// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_UNSTRUCTURED_INTERFACE_HH
#define FLECSI_TOPO_UNSTRUCTURED_INTERFACE_HH

#include "flecsi/data/map.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/core.hh"
#include "flecsi/topo/types.hh"
#include "flecsi/topo/unstructured/types.hh"
#include "flecsi/util/color_map.hh"
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

/// Topology category.
template<class P>
using unstructured = topology<P, unstructured_base>;

/// Topology type.
/// \tparam Policy the specialization, following unstructured_specialization
template<typename Policy>
struct topology<Policy, unstructured_base> : unstructured_base,
                                             with_ragged<Policy> {

  /*--------------------------------------------------------------------------*
    Public types.
   *--------------------------------------------------------------------------*/

  using index_space = typename Policy::index_space;
  using index_spaces = typename Policy::index_spaces;
  using copy_spaces = util::to_copy_spaces<Policy>;

  template<Privileges>
  struct access;

  /*--------------------------------------------------------------------------*
    Constructor.
   *--------------------------------------------------------------------------*/

  topology(scheduler & s, coloring const & c)
    : topology(
        s,
        [&c]() -> auto & {
          flog_assert(c.idx_spaces.size() == index_spaces::size,
            c.idx_spaces.size()
              << " sizes for " << index_spaces::size << " index spaces");
          return c;
        }(),
        index_spaces(),
        copy_spaces()) {}

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
  repartition & get_partition() {
    return part_.template get<S>();
  }

  template<typename Type, data::layout Layout, typename Policy::index_space S>
  [[nodiscard]] const data::copy_plan * ghost_copy(scheduler & s,
    data::field_reference<Type, Layout, Policy, S> const & f) {
    if constexpr(Layout == data::ragged) {
      auto const & cg = cgraph_.template get<S>();
      auto const & cg_sh = cgraph_shared_.template get<S>();
      constexpr PrivilegeCount N = Policy::template privilege_count<S>;
      ragged_buffers_.template get<S>()
        .template xfer<ragged_impl<Type, N>::start, ragged_impl<Type, N>::xfer>(
          s, f, cg(ctopo_), cg_sh(ctopo_));
      return nullptr;
    }
    else
      return &plan_.template get<S>();
  }

  /*!
    Return the connectivity between two index spaces.
    \tparam F The "from" index space.
    \tparam T The "to" index space.
    \return a \c ragged field (of type flecsi::field::definition) that
            maps each element of \c F to zero or more elements of \c T.
   */
  template<index_space F, index_space T>
  auto & get_connectivity() {
    return connect_.template get<F>().template get<T>();
  }

  /*!
    Get the special entities for a given index space.
    \return topology instance on which \c special_field is registered
   */
  template<index_space S, typename Policy::entity_list E>
  auto & get_special_entities() {
    return special_.template get<S>().template get<E>();
  }

  /// Field that holds each list of special entity IDs.
  static inline const field<util::id>::definition<array<Policy>> special_field;

private:
  /*
    Communication graph topology for ragged ghost updates.
   */

  struct ctopo : specialization<user, ctopo> {};

  /*
    Define the partial closures that will be used to construct partitions
    when an instance of this topology is allocated.
   */

  // clang-format off
  template<auto... VV, auto... CI>
  topology(scheduler & s, unstructured_base::coloring const & c,
    util::constants<VV...>, util::constants<CI...> /* deduce pack */)
    : with_ragged<Policy>(s, c.colors),
      ctopo_(s, c.color_peers),
      part_{
        {
        make_repartitioned<Policy, VV>(
          c.colors,
          s,
          [p = c.idx_spaces[index<VV>].partitions](
            std::size_t i) { return p[i]; })...
        }
      },
      special_(s, c.colors),
      /* all data members need to be initialized before make_copy_plan */
      plan_{
        { 
        // make a copy plan only for index spaces != 1
         make_copy_plan<CI>(s, c)...
        }
      },
      ragged_buffers_{
        {data::buffers::topology(s, c.idx_spaces[index<CI>].peers)...}
      }
  {
    allocate_connectivities(s, c, connect_);
    // Sanity checks for indexes spaces for which privilege count is 1
    ([&]{ 
      if(Policy::template privilege_count<VV> == 1) {
        // check that each element of coloring::index_space::peers is empty
        for(auto & comm_peers : c.idx_spaces[index<VV>].peers)
          if(!comm_peers.empty())
            throw std::invalid_argument(
              "Privilege count is 1 but `coloring::index_space::peers` "
              "are not empty");
        // check that each element of num_intervals is 0
        for(std::size_t nb_of_ghost_intervals : c.idx_spaces[index<VV>].num_intervals)
          if(nb_of_ghost_intervals != 0)
            throw std::invalid_argument("Privilege count is 1 but the elements "
                                        "of `num_intervals` are non-zero");
        // check that index_color::peers is empty
        for(auto & index_color : c.idx_spaces[index<VV>].colors)
          if(!index_color.peers.empty())
            throw std::invalid_argument(
              "Privilege count is 1 but `index_color::peers` are not empty");
      }
    }(), ...);
  }
  // clang-format on

  /*
    Construct copy plan for the given index space S.
   */

  template<index_space S>
  data::copy_plan make_copy_plan(scheduler & s,
    unstructured_base::coloring const & c) {
    constexpr PrivilegeCount NP = Policy::template privilege_count<S>;

    auto & c1 = c.idx_spaces[index<S>].colors;
    destination_intervals intervals(c1.size());
    source_pointers pointers(c1.size());

    const auto grp = exec::group::world();
    const auto blk = exec::mapping::block();

    const auto cg = cgraph_.template get<S>()(ctopo_);
    auto & cgp = cg.get_elements();
    s.execute<cgraph_size>(&c1, cgp.sizes(), grp, blk);
    cgp.resize();

    const auto sh = cgraph_shared_.template get<S>()(ctopo_);
    auto & shp = sh.get_elements();
    s.execute<cgraph_shared_size>(&c1, shp.sizes(), grp, blk);
    shp.resize();

    // Synchronous, so guarantees the completion of the above tasks:
    s.execute<idx_itvls>(c1, intervals, pointers, cg, sh, grp, blk);

    return {s,
      *this,
      c.idx_spaces[index<S>].num_intervals,
      [&](auto f) { s.execute<set_dests>(exec::on, f, &intervals, grp, blk); },
      // This wait covers set_dests as well.
      [&](auto f) { s.execute<set_ptrs<NP>>(f, &pointers, grp, blk).wait(); },
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
  void allocate_connectivities(scheduler & s,
    const unstructured_base::coloring & c,
    util::key_tuple<util::key_type<VV, TT>...> const & /* deduce pack */) {
    (
      [&](TT const & row) { // invoked for each from-entity
        const std::vector<index_color> & ic = c.idx_spaces[index<VV>].colors;
        for_each(
          [&](auto v) { // invoked for each to-entity
            auto & p = row.template get<v.value>()(*this).get_elements();
            s.execute<cnx_size>(&ic,
               index<v.value>,
               p.sizes(),
               exec::group::world(),
               exec::mapping::block())
              .wait();
            p.resize();
          },
          typename TT::keys());
      }(connect_.template get<VV>()),
      ...);
  }

  template<class F, auto... VV>
  static void for_each(F && f, util::constants<VV...> /* deduce pack */) {
    (f(util::constant<VV>()), ...);
  }

  auto & get_sizes(std::size_t i) {
    return part_[i].sz;
  }

  friend borrow_extra<topology>;

  static inline const connect_t<Policy> connect_;

  typename ctopo::topology ctopo_;

  static inline const util::key_array<
    typename field<util::id, data::ragged>::template definition<ctopo>,
    index_spaces>
    cgraph_, cgraph_shared_;

  util::key_array<repartitioned, index_spaces> part_;
  lists<Policy> special_;
  // Initializing this depends on the above:
  util::key_array<data::copy_plan, copy_spaces> plan_;
  util::key_array<data::buffers::topology, copy_spaces> ragged_buffers_;
};

template<class P>
struct borrow_extra<unstructured<P>> : borrow_sizes<P> {
  borrow_extra(unstructured<P> & u, const data::borrow & b, bool f)
    : borrow_extra(u, b, f, typename P::entity_lists()) {}

private:
  friend unstructured<P>; // for access::send

  typename decltype(unstructured<P>::special_)::Base::template map_type<
    unstructured_base::borrow_array>
    special_;

  template<typename P::index_space... VV, class... TT>
  borrow_extra(unstructured<P> & u,
    const data::borrow & b,
    bool f,
    util::types<util::key_type<VV, TT>...> /* deduce pack */)
    : borrow_extra::borrow_sizes(u, b, f),
      special_(u.special_.template get<VV>().map([&](auto & t) {
        return borrow_base::wrap<std::remove_cvref_t<decltype(t)>>(t, b, f);
      })...) {}
};

/// Topology interface base.
/// \gpu.
/// \see specialization_base::interface
template<typename Policy>
template<Privileges Privileges>
struct topology<Policy, topo::unstructured_base>::access {
  // For unstructured_base::bounding_box, which can't use the specialization's
  // interface.
  friend unstructured_base;
  template<class F>
  void send(F && f) {
    std::size_t i = 0;
    for(auto & a : size_)
      f(a, [&i](auto & u) { return topo::resize::field(u.get_sizes(i++)); });

    connect_send(f, connect_, topology::connect_);
    lists_send(f, special_, special_field, [](auto & u) -> auto & {
      return u.special_;
    });
  }

protected:
  using entity_list = typename Policy::entity_list;

  /*!
    Return an index space as a range.
    \host.

    \return range of \c id\<IndexSpace\> values
   */

  template<index_space S>
  FLECSI_INLINE_TARGET auto entities() const {
    return make_ids<S>(util::iota_view<util::id>(0, *size_.template get<S>()));
  }

  /*!
    Return a range of connectivity information for the parameterized
    index spaces.
    \tparam T The connected index space.
    \tparam F The index space with connections.
    \param from query entity
    \return range of \c id\<To\> values
   */

  template<index_space To, index_space From>
  FLECSI_INLINE_TARGET auto entities(id<From> from) const {
    return make_ids<To>(connectivity<From, To>()[from]);
  }

  /// Get a special-entities list.
  /// \return range of \c id\<I\> values
  template<index_space I, entity_list L>
  FLECSI_INLINE_TARGET auto special_entities() const {
    return make_ids<I>(special_.template get<I>().template get<L>().span());
  }

private:
  template<index_space F, index_space T>
  auto & connectivity() {
    return connect_.template get<F>().template get<T>();
  }

  template<index_space F, index_space T>
  FLECSI_INLINE_TARGET auto const & connectivity() const {
    return connect_.template get<F>().template get<T>();
  }

  /*--------------------------------------------------------------------------*
    Private data members.
   *--------------------------------------------------------------------------*/

  util::key_array<data::scalar_access<topo::resize::Field::value_type,
                    privilege_merge(Privileges)>,
    index_spaces>
    size_;
  connect_access<Policy, Privileges> connect_;
  lists_t<field<util::id>::accessor<privilege_merge(Privileges)>, Policy>
    special_;

}; // struct unstructured<Policy>::access

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/

template<>
struct detail::base<unstructured> {
  using type = unstructured_base;
}; // struct detail::base<unstructured>

#ifdef DOXYGEN
/// Example specialization which is not really implemented.
struct unstructured_specialization
  : specialization<unstructured, unstructured_specialization> {
  /// Connectivity information to store.
  /// The format is\code
  /// list<from<cells, to<vertices, edges>>>,
  ///      from<...>, ...>
  /// \endcode
  /// where each leaf is an \c index_space.
  using connectivities = list<>;

  /// Enumeration of special entity lists.
  enum entity_list {};
  /// Special entity lists to store.
  /// The format is\code
  /// list<entity<edges, has<dirichlet, neumann>>,
  ///      entity<...>, ...>
  /// \endcode
  /// where each \c has associates \c entity_list values with an
  /// \c index_space.
  using entity_lists = list<>;
};
#endif

/// \}
} // namespace topo
} // namespace flecsi

#endif
