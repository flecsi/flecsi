// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_UNSTRUCTURED_INTERFACE_HH
#define FLECSI_TOPO_UNSTRUCTURED_INTERFACE_HH

#include "flecsi/data/layout.hh"
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

/// Topology type.
/// \tparam Policy the specialization, following unstructured_specialization
template<typename Policy>
struct unstructured : unstructured_base,
                      with_ragged<Policy>,
                      with_meta<Policy> {

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

  unstructured(coloring const & c)
    : unstructured(
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
  [[nodiscard]] const data::copy_plan * ghost_copy(
    data::field_reference<Type, Layout, Policy, S> const & f) {
    if constexpr(Layout == data::ragged) {
      auto const & cg = cgraph_.template get<S>();
      auto const & cg_sh = cgraph_shared_.template get<S>();
      constexpr PrivilegeCount N = Policy::template privilege_count<S>;
      ragged_buffers_.template get<S>()
        .template xfer<ragged_impl<Type, N>::start, ragged_impl<Type, N>::xfer>(
          f, cg(ctopo_), cg_sh(ctopo_));
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
  unstructured(unstructured_base::coloring const & c,
    util::constants<VV...>, util::constants<CI...> /* deduce pack */)
    : with_ragged<Policy>(c.colors),
      with_meta<Policy>(c.colors),
      ctopo_(c.color_peers),
      part_{
        {
        make_repartitioned<Policy, VV>(
          c.colors,
          make_partial<idx_size>(c.idx_spaces[index<VV>].partitions))...
        }
      },
      special_(c.colors),
      /* all data members need to be initialized before make_copy_plan */
      plan_{
        { 
        // make a copy plan only for index spaces != 1
        make_copy_plan<CI>(c)...
        }
      },
      ragged_buffers_{{data::buffers::core(c.idx_spaces[index<CI>].peers)...}}

  {
    allocate_connectivities(c, connect_);
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
  data::copy_plan make_copy_plan(unstructured_base::coloring const & c) {
    constexpr PrivilegeCount NP = Policy::template privilege_count<S>;

    destination_intervals intervals;
    source_pointers pointers;

    // set the sizes of the cgraph accessor.
    auto const & cg = cgraph_.template get<S>();
    auto & cgp = ctopo_.ragged.template get<elements>()[cg.fid];

    // creating a launch map for the underlying ragged partition
    auto cgplm = data::launch::make(cgp);
    execute<cgraph_size, mpi>(c.idx_spaces[index<S>].colors, cgplm);

    // the actual resize of the underlying fields
    cgp.resize();

    // set up cgraph_shared_
    auto const & sh = cgraph_shared_.template get<S>();
    auto & shp = ctopo_.ragged.template get<elements>()[sh.fid];
    auto shplm = data::launch::make(shp);
    execute<cgraph_shared_size, mpi>(c.idx_spaces[index<S>].colors, shplm);
    shp.resize();

    // compute the launch maps for the fields
    auto clm = data::launch::make(ctopo_);

    auto lm = data::launch::make(*this);

    execute<idx_itvls, mpi>(
      c.idx_spaces[index<S>].colors, intervals, pointers, cg(clm), sh(clm));

    // clang-format off
    auto dest_task = [&intervals](auto f) {
      // TODO: make this just once for all index spaces
      auto lm = data::launch::make(f.topology());
      execute<set_dests, mpi>(lm(f), intervals);
    };

    auto ptrs_task = [&](auto f) {
      auto lm = data::launch::make(f.topology());
      execute<set_ptrs<NP>, mpi>(lm(f), pointers);
    };
    // clang-format on

    return {*this,
      c.idx_spaces[index<S>].num_intervals,
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
    auto lm = data::launch::make(this->meta);
    (
      [&](TT const & row) { // invoked for each from-entity
        const std::vector<index_color> & ic = c.idx_spaces[index<VV>].colors;
        for_each(
          [&](auto v) { // invoked for each to-entity
            execute<cnx_size, mpi>(ic, index<v.value>, temp_size(lm));
            auto & p = row.template get<v.value>()(*this).get_elements();
            execute<copy_sizes>(temp_size(this->meta), p.sizes());
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

  /*--------------------------------------------------------------------------*
    Private data members.
   *--------------------------------------------------------------------------*/
  friend borrow_extra<unstructured>;

  static inline const connect_t<Policy> connect_;

  template<typename, typename>
  struct key_define;

  template<typename T, auto... SS>
  struct key_define<T, util::constants<SS...>> {
    using type = util::key_tuple<util::key_type<SS,
      typename field<T>::template definition<Policy, SS>>...>;
  };

  typename ctopo::core ctopo_;

  static inline const util::key_array<
    typename field<util::id, data::ragged>::template definition<ctopo>,
    index_spaces>
    cgraph_, cgraph_shared_;

  // static inline const resize::Field::definition<ctopo> temp_cgsize;

  static inline const resize::Field::definition<meta<Policy>> temp_size;

  util::key_array<repartitioned, index_spaces> part_;
  lists<Policy> special_;
  // Initializing this depends on the above:
  util::key_array<data::copy_plan, copy_spaces> plan_;
  util::key_array<data::buffers::core, copy_spaces> ragged_buffers_;

}; // struct unstructured

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
      special_(u.special_.template get<VV>().map([&b, f](auto & t) {
        return borrow_base::wrap<std::decay_t<decltype(t)>>(t, b, f);
      })...) {}
};

/// Topology interface base.
/// \gpu.
/// \see specialization_base::interface
template<typename Policy>
template<Privileges Privileges>
struct unstructured<Policy>::access {
  // For unstructured_base::bounding_box, which can't use the specialization's
  // interface.
  friend unstructured_base;
  template<class F>
  void send(F && f) {
    std::size_t i = 0;
    for(auto & a : size_)
      a.topology_send(
        f, [&i](auto & u) -> auto & { return u.get_sizes(i++); });

    connect_send(f, connect_, unstructured::connect_);
    lists_send(
      f, special_, special_field, [
      ](auto & u) -> auto & { return u.special_; });
  }

protected:
  using subspace_list = std::size_t;
  using entity_list = typename Policy::entity_list;
  access() : connect_(unstructured::connect_) {}

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
