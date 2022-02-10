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
#include "flecsi/data/privilege.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/core.hh"
#include "flecsi/topo/index.hh"
#include "flecsi/topo/narray/coloring_utils.hh"
#include "flecsi/topo/narray/types.hh"
#include "flecsi/topo/utility_types.hh"
#include "flecsi/util/array_ref.hh"

#include <utility>

namespace flecsi {
namespace topo {
/// \defgroup narray Multi-dimensional Array
/// Configurable multi-dimensional array topology.
/// Can be used for structured meshes.
/// \ingroup topology
/// \{

/*!----------------------------------------------------------------------------*
  Narray Topology.

  \tparam Policy Policy is the CRTP derived type (narray specialization)

   The policy is required to provide the following information:
   index_space : enum listing the entities in the specialization
   index_spaces : created using the utility type "has" to convert the
 index_space enum to indexed constants axis : enum listing the axes in the
 specialization, they should be consistent with the dimension of mesh. axes :
 created using the utility type "has" to convert the axis enum to indexed
 constants dimension: mesh dimension

  interface : template<class B> struct interface : B {}. This type is the
 specialization  interface build using the base interface provided by the core
 topology and should have specialization specific methods.

  color : static method returning the narray coloring type, there is no
 requirement for the argument list.
 *----------------------------------------------------------------------------*/
template<typename Policy>
struct narray : narray_base, with_ragged<Policy>, with_meta<Policy> {

  using index_space = typename Policy::index_space;
  using index_spaces = typename Policy::index_spaces;
  using axis = typename Policy::axis;
  using axes = typename Policy::axes;
  using id = util::id;

  static constexpr Dimension dimension = Policy::dimension;

  /// This type is the topology accessor base type "B"
  /// from which specialization interface type is derived.
  /// \sa core
  template<Privileges>
  struct access;

  narray(coloring const & c)
    : narray(
        [&c]() -> auto & {
          flog_assert(c.idx_colorings.size() == index_spaces::size,
            c.idx_colorings.size()
              << " sizes for " << index_spaces::size << " index spaces");
          return c;
        }(),
        index_spaces(),
        std::make_index_sequence<index_spaces::size>()) {}

  /// Structural information about one color.
  /// \image html narray-layout.png "Layouts for each possible orientation."
  struct meta_data {
    using scoord = std::array<std::size_t, dimension>;
    using shypercube = std::array<scoord, 2>;
    std::array<std::uint32_t, index_spaces::size> faces;
    /// Global extents per index space.
    /// These are necessarily the same on every color.
    std::array<scoord, index_spaces::size> global,
      /// The global offsets to the beginning of the color's region per index
      /// space, excluding any non-physical boundary padding.
      /// Use to map from local to global ids.
      offset,
      /// The size of the color's region per index space, including ghosts and
      /// boundaries.
      extents;
    /// The range of the color's elements that logically exist in each index
    /// space.  Ghosts and boundaries are not included.
    std::array<shypercube, index_spaces::size> logical,
      /// The range of the color's elements in each index space, including
      /// boundaries but not ghosts.
      extended;
  };

  // fields for storing topology meta data per index-space
  static inline const typename field<meta_data,
    data::single>::template definition<meta<Policy>>
    meta_field;

  // field for storing user-defined meta data
  static inline const typename field<typename Policy::meta_data,
    data::single>::template definition<meta<Policy>>
    policy_meta_field;

  // index-space specific parts
  util::key_array<repartitioned, index_spaces> part_;

  // index-space specific copy plans
  util::key_array<data::copy_plan, index_spaces> plan_;

  Color colors() const {
    return part_.front().colors();
  }

  template<index_space S>
  data::region & get_region() {
    return part_.template get<S>();
  }

  template<index_space S>
  const data::partition & get_partition() const {
    return part_.template get<S>();
  }

  template<typename Type,
    data::layout Layout,
    typename Policy::index_space Space>
  void ghost_copy(
    data::field_reference<Type, Layout, Policy, Space> const & f) {
    plan_.template get<Space>().issue_copy(f.fid());
  }

private:
  template<auto... Value, std::size_t... Index>
  narray(const coloring & c,
    util::constants<Value...> /* index spaces to deduce pack */,
    std::index_sequence<Index...>)
    : with_ragged<Policy>(c.colors), with_meta<Policy>(c.colors),
      part_{{make_repartitioned<Policy, Value>(c.colors,
        make_partial<idx_size>(c.idx_colorings[Index]))...}},
      plan_{
        {make_plan<Value>(c.idx_colorings[Index], part_[Index], c.comm)...}} {
    init_meta(c);
    init_policy_meta(c);
  }

  /*!
   Method to create copy plans for entities of an index-sapce.
   @param colors  The number of colors
   @param vpc  Vector of process_colors, where an index into vpc provides
   coloring information corresponding to a particular color.
  */
  template<index_space S>
  data::copy_plan make_plan(index_coloring const & ic,
    repartitioned & p,
    MPI_Comm const & comm) {
    std::vector<std::size_t> num_intervals;

    // In this method, a mpi task "idx_itvls" is invoked, which computes couple
    // of information: intervals and points. The intervals encode local ghost
    // intervals, whereas points capture the  local offset and corresponding
    // remote/shared offset on remote/shared color. The intervals and points are
    // used to create function objects "dest_tasks" and "ptrs_tasks" that is
    // subsequently used by copy plan to perform the data communication. Note
    // that after this call the copy plan objects have been created. The actual
    // communication is invoked as part of task execution depending upon the
    // privilege requirements of the task.

    execute<idx_itvls, mpi>(ic, num_intervals, comm);

    // clang-format off
    auto dest_task = [&ic, &comm](auto f) {
      execute<set_dests, mpi>(f, ic.intervals, comm);
    };

    auto ptrs_task = [&ic, &comm](auto f) {
      execute<set_ptrs<Policy::template privilege_count<S>>, mpi>(
        f, ic.points, comm);
    };

    return {*this, p,  num_intervals, dest_task, ptrs_task, util::constant<S>()};
    // clang-format on
  }

  static void set_meta(
    typename field<meta_data, data::single>::template accessor<wo> m,
    narray_base::coloring const & c) {
    meta_data & md = m;

    for(std::size_t i{0}; i < index_spaces::size; ++i) {
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

      const auto & ci = c.idx_colorings[i];
      md.faces[i] = ci.faces;
      copy(ci.global, md.global[i]);
      copy(ci.offset, md.offset[i]);
      copy(ci.extents, md.extents[i]);
      copy2(ci.logical, md.logical[i]);
      copy2(ci.extended, md.extended[i]);
    } // for
  } // set_meta

  /// Initialization for topology meta_data.
  /// Executes a mpi task "set_meta" which essentially
  /// copies the relevant color info into internal meta fields.
  void init_meta(narray_base::coloring const & c) {
    execute<set_meta, mpi>(meta_field(this->meta), c);
  }

  static void set_policy_meta(typename field<typename Policy::meta_data,
    data::single>::template accessor<wo>) {}

  /// Initialization for user policy meta_data.
  /// Executes a mpi task "set_policy_meta" which
  /// copies the relevant user-define meta data as part of the topology.
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

  /*!
   This range enumeration provides a classification of the various
   types of partition entities that can be requested out of a topology
   specialization created using this type. The following describes what each
   of the range enumeration means in a mesh part returned by the coloring
   algorithm. For the structured mesh partitioning, the partition info is
   specified per axis.

   These ranges are used in many of the interface methods to provide
   information such as size, extents, offsets about them.
  */
  enum class range : std::size_t {
    logical, ///<  the logical, i.e., the owned part of the axis
    extended, ///< the boundary padding along with the logical part
    all, ///< the ghost padding along with the logical part
    boundary_low, ///< the boundary padding on the lower bound of the axis
    boundary_high, ///< the boundary padding on the upper bound of the axis
    ghost_low, ///< the ghost padding on the lower bound of the axis
    ghost_high, ///< the ghost padding on the upper bound of the axis
    global ///< global info about the mesh, the meaning depends on what is being
           ///< queried
  };

  using hypercubes = index::has<range::logical,
    range::extended,
    range::all,
    range::boundary_low,
    range::boundary_high,
    range::ghost_low,
    range::ghost_high,
    range::global>;

  /*!
   Method to check if an axis of the local mesh is incident on the lower
   bound of the corresponding axis of the global mesh
   \sa meta_data, process_color
  */
  template<index_space S, axis A>
  bool is_low() const {
    return (meta_->faces[S] >> A * 2) & narray_impl::low;
  }

  /*!
   Method to check if an axis of the local mesh is incident on the upper
   bound of the corresponding axis of the global mesh
   \sa meta_data, process_color
  */
  template<index_space S, axis A>
  bool is_high() const {
    return (meta_->faces[S] >> A * 2) & narray_impl::high;
  }

  /*!
   Method to check if axis A of index-space S is in between the lower and upper
   bound along axis A of the global domain. \sa meta_data, process_color
  */
  template<axis A>
  bool is_interior() const {
    return !is_low<A>() && !is_high<A>();
  }

  /*!
    Method to return size of the index-space S along axis A for range SE.
    \sa enum range
  */
  template<index_space S, axis A, range SE>
  std::size_t size() const {
    static_assert(
      std::size_t(SE) < hypercubes::size, "invalid size identifier");
    if constexpr(SE == range::logical) {
      return meta_->logical[S][1][A] - meta_->logical[S][0][A];
    }
    else if constexpr(SE == range::extended) {
      return meta_->extended[S][1][A] - meta_->extended[S][0][A];
    }
    else if constexpr(SE == range::all) {
      return meta_->extents[S][A];
    }
    else if constexpr(SE == range::boundary_low) {
      return meta_->logical[S][0][A] - meta_->extended[S][0][A];
    }
    else if constexpr(SE == range::boundary_high) {
      return meta_->extended[S][1][A] - meta_->logical[S][1][A];
    }
    else if constexpr(SE == range::ghost_low) {
      if(!is_low<S, A>())
        return meta_->logical[S][0][A];
      else
        return 0;
    }
    else if constexpr(SE == range::ghost_high) {
      if(!is_high<S, A>())
        return meta_->extents[S][A] - meta_->logical[S][1][A];
      else
        return 0;
    }
    else if constexpr(SE == range::global) {
      return meta_->global[S][A];
    }
  }

  /*!
    Method to return an iterator over the extents of the index-space S along
    axis A for range SE. \sa enum range, note that this method is not applicable
    to range::global.
  */
  template<index_space S, axis A, range SE>
  auto extents() const {
    static_assert(
      std::size_t(SE) < hypercubes::size, "invalid extents identifier");
    if constexpr(SE == range::logical) {
      return make_ids<S>(util::iota_view<util::id>(
        meta_->logical[S][0][A], meta_->logical[S][1][A]));
    }
    else if constexpr(SE == range::extended) {
      return make_ids<S>(util::iota_view<util::id>(
        meta_->extended[S][0][A], meta_->extended[S][1][A]));
    }
    else if constexpr(SE == range::all) {
      return make_ids<S>(util::iota_view<util::id>(0, meta_->extents[S][A]));
    }
    else if constexpr(SE == range::boundary_low) {
      return make_ids<S>(util::iota_view<util::id>(0, size<S, A, SE>()));
    }
    else if constexpr(SE == range::boundary_high) {
      return make_ids<S>(util::iota_view<util::id>(
        meta_->logical[S][1][A], meta_->logical[S][1][A] + size<S, A, SE>()));
    }
    else if constexpr(SE == range::ghost_low) {
      return make_ids<S>(util::iota_view<util::id>(0, size<S, A, SE>()));
    }
    else if constexpr(SE == range::ghost_high) {
      return make_ids<S>(util::iota_view<util::id>(
        meta_->logical[S][1][A], meta_->logical[S][1][A] + size<S, A, SE>()));
    }
    else {
      flog_error("invalid range");
    }
  }

  /*!
    Method to return an offset of the index-space S along axis A for range SE.
    \sa enum range
  */
  template<index_space S, axis A, range SE>
  std::size_t offset() const {
    static_assert(
      std::size_t(SE) < hypercubes::size, "invalid offset identifier");
    if constexpr(SE == range::logical) {
      return meta_->logical[S][0][A];
    }
    else if constexpr(SE == range::extended) {
      return meta_->extended[S][0][A];
    }
    else if constexpr(SE == range::all) {
      return 0;
    }
    else if constexpr(SE == range::boundary_low) {
      return meta_->extended[S][0][A];
    }
    else if constexpr(SE == range::boundary_high) {
      return meta_->logical[S][1][A];
    }
    else if constexpr(SE == range::ghost_low) {
      return 0;
    }
    else if constexpr(SE == range::ghost_high) {
      return meta_->logical[S][1][A];
    }
    else if constexpr(SE == range::global) {
      return meta_->offset[S][A];
    }
  }

  ///  This method provides a mdspan of the field underlying data.
  ///  It can be used to create data views with the shape appropriate to S.
  template<index_space S, typename T, Privileges P>
  auto mdspan(data::accessor<data::dense, T, P> const & a) const {
    auto const s = a.span();
    return util::mdspan<typename decltype(s)::element_type, dimension>(
      s.data(), meta_->extents[S]);
  }
  template<index_space S, typename T, Privileges P>
  auto mdcolex(data::accessor<data::dense, T, P> const & a) const {
    return util::mdcolex<
      typename std::remove_reference_t<decltype(a)>::element_type,
      dimension>(a.span().data(), meta_->extents[S]);
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
}; // struct narray<Policy>::access

/*----------------------------------------------------------------------------*
  Define Base.
 *----------------------------------------------------------------------------*/

template<>
struct detail::base<narray> {
  using type = narray_base;
}; // struct detail::base<narray>

/// \}
} // namespace topo
} // namespace flecsi
