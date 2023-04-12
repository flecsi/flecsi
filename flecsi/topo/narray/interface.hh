// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_NARRAY_INTERFACE_HH
#define FLECSI_TOPO_NARRAY_INTERFACE_HH

#include "flecsi/data/accessor.hh"
#include "flecsi/data/copy_plan.hh"
#include "flecsi/data/layout.hh"
#include "flecsi/data/map.hh"
#include "flecsi/data/privilege.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/core.hh"
#include "flecsi/topo/index.hh"
#include "flecsi/topo/narray/coloring_utils.hh"
#include "flecsi/topo/narray/types.hh"
#include "flecsi/topo/types.hh"
#include "flecsi/util/array_ref.hh"

#include <memory>
#include <utility>

namespace flecsi {
namespace topo {
/// \defgroup narray Multi-dimensional Array
/// Configurable multi-dimensional array topology.
/// Can be used for structured meshes.
/// \ingroup topology
/// \{

/*!
  Narray Topology.
  \tparam Policy the specialization, following
   \ref narray_specialization.

  \sa topo::specialization, topo::topology
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

  Color colors() const {
    return part_.front().colors();
  }

  template<index_space S>
  data::region & get_region() {
    return part_.template get<S>();
  }

  template<index_space S>
  repartition & get_partition() {
    return part_.template get<S>();
  }

  template<typename Type,
    data::layout Layout,
    typename Policy::index_space Space>
  void ghost_copy(
    data::field_reference<Type, Layout, Policy, Space> const & f) {
    if constexpr(Layout == data::ragged) {
      using Impl =
        ragged_impl<Space, Type, Policy::template privilege_count<Space>>;
      buffers_.template get<Space>().template xfer<Impl::start, Impl::xfer>(
        f, meta_field(this->meta));
    }
    else
      plan_.template get<Space>().issue_copy(f.fid());
  }

private:
  /// Structural information about one color.
  /// \image html narray-layout.svg "Layouts for each possible orientation." width=100%

  using meta_data = util::key_array<axis_color, axes>;

  template<auto... Value, std::size_t... Index>
  narray(const coloring & c,
    util::constants<Value...> /* index spaces to deduce pack */,
    std::index_sequence<Index...>)
    : with_ragged<Policy>(c.colors()), with_meta<Policy>(c.colors()),
      part_{{make_repartitioned<Policy, Value>(c.colors(),
        make_partial<idx_size>([&]() {
          std::vector<std::size_t> partitions;
          for(const auto & idxco :
            c.idx_colorings[Index].process_coloring(c.comm)) {
            partitions.push_back(idxco.extents());
          }
          concatenate(partitions, c.colors(), c.comm);
          return partitions;
        }()))...}},
      plan_{{make_copy_plan<Value>(c.colors(),
        c.idx_colorings[Index],
        part_[Index],
        c.comm)...}},
      buffers_{{data::buffers::core(
        narray_impl::peers<dimension>(c.idx_colorings[Index]))...}} {
    auto lm = data::launch::make(this->meta);
    execute<set_meta<Value...>, mpi>(meta_field(lm), c);
    init_policy_meta(c);
  }

  /*!
   Method to create copy plans for entities of an index-space.
   @param colors  The number of colors
   @param idef index definition
   @param p partition
   @param comm MPI communicator
  */
  template<index_space S>
  data::copy_plan make_copy_plan(Color colors,
    index_definition const & idef,
    repartitioned & p,
    MPI_Comm const & comm) {

    std::vector<std::size_t> num_intervals(colors, 0);
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> intervals;
    std::vector<
      std::map<Color, std::vector<std::pair<std::size_t, std::size_t>>>>
      points;

    // In this method, a mpi task "idx_itvls" is invoked, which computes couple
    // of information: intervals and points. The intervals encode local ghost
    // intervals, whereas points capture the  local offset and corresponding
    // remote/shared offset on remote/shared color. The intervals and points are
    // used to create function objects "dest_tasks" and "ptrs_tasks" that is
    // subsequently used by copy plan to perform the data communication. Note
    // that after this call the copy plan objects have been created. The actual
    // communication is invoked as part of task execution depending upon the
    // privilege requirements of the task.

    execute<idx_itvls, mpi>(idef, num_intervals, intervals, points, comm);

    // clang-format off
    auto dest_task = [&intervals](auto f) {
      auto lm = data::launch::make(f.topology());
      execute<set_dests, mpi>(lm(f), intervals);
    };

    auto ptrs_task = [&points](auto f) {
      auto lm = data::launch::make(f.topology());
      execute<set_ptrs<Policy::template privilege_count<S>>, mpi>(
        lm(f), points);
    };
    // clang-format on

    return {*this, p, num_intervals, dest_task, ptrs_task, util::constant<S>()};
  }

  template<auto... Value> // index_spaces
  static void set_meta(
    data::multi<typename field<util::key_array<meta_data, index_spaces>,
      data::single>::template accessor<wo>> mm,
    narray_base::coloring const & c) {
    const auto ma = mm.accessors();
    auto copy_meta_data = [](meta_data & md, const index_color & idxco) {
      std::copy(idxco.axis_colors.begin(), idxco.axis_colors.end(), md.begin());
    };
    for(auto i = ma.size(); i--;) {
      std::size_t index{0};
      (copy_meta_data(ma[i]->template get<Value>(),
         c.idx_colorings[index++].process_coloring(c.comm)[i]),
        ...);
    }
  }

  static void set_policy_meta(typename field<typename Policy::meta_data,
    data::single>::template accessor<wo>) {}

  /// Initialization for user policy meta_data.
  /// Executes a mpi task "set_policy_meta" which
  /// copies the relevant user-define meta data as part of the topology.
  void init_policy_meta(narray_base::coloring const &) {
    execute<set_policy_meta>(policy_meta_field(this->meta));
  }

  auto & get_sizes(std::size_t i) {
    return part_[i].sz;
  }

  /// Ragged communication routines
  template<typename Policy::index_space Space, typename T, PrivilegeCount N>
  struct ragged_impl {
    using A = std::array<int, dimension>;
    using A2 = std::array<A, 3>;

    using fa = typename field<T,
      data::ragged>::template accessor1<privilege_ghost_repeat<ro, na, N>>;

    using fm_rw = typename field<T,
      data::ragged>::template mutator1<privilege_ghost_repeat<ro, rw, N>>;

    using mfa = typename field<util::key_array<meta_data, index_spaces>,
      data::single>::template accessor<ro>;

    static void start(fa v, mfa mf, data::buffers::Start mv) {
      send(v, mf, true, mv);
    } // start

    static int xfer(fm_rw g, mfa mf, data::buffers::Transfer mv) {
      // get the meta data for the index space
      meta_data md = mf->template get<Space>();
      std::map<Color, std::vector<int>> color_bounds;
      get_ngb_color_bounds(false, md, color_bounds);

      A lbnds, ubnds, strs;
      for(Dimension i = 0; i < dimension; i++) {
        strs[i] = md[i].extent();
      }

      // Loop over the receiving colors and receive the data for all boxes
      int bufid = color_bounds.size();
      for(auto & [c, v] : color_bounds) {
        std::vector<int> lids;
        auto get_lids = [&](int lid) { lids.push_back(lid); };

        int nboxes = v.size() / (2 * dimension);
        // get lids of all boxes
        for(int k = nboxes - 1; k >= 0; --k) {
          for(Dimension d = 0; d < dimension; ++d) {
            lbnds[d] = v[2 * dimension * k + d];
            ubnds[d] = v[2 * dimension * k + dimension + d];
          }
          traverse(lbnds, ubnds, strs, get_lids);
        }

        // receive
        data::buffers::ragged::read(g, mv[bufid], lids);
        ++bufid;
      }

      // resume transfer if data was not fully packed during start
      return send(g, mf, false, mv);
    } // xfer

  private:
    template<typename F, typename B>
    static bool send(F f, mfa mf, bool first, B mv) {
      // get the meta data for the index space
      meta_data md = mf->template get<Space>();
      bool sent = false;

      A lbnds, ubnds, strs;
      for(Dimension i = 0; i < dimension; i++) {
        strs[i] = md[i].extent();
      }

      std::map<Color, std::vector<int>> color_bounds;
      get_ngb_color_bounds(true, md, color_bounds);

      int p = 0;
      for(auto & [c, v] : color_bounds) {
        auto b = data::buffers::ragged{mv[p++], first};
        auto send_data = [&](int lid) {
          if(!b(f, lid, sent))
            return;
        };

        int nboxes = v.size() / (2 * dimension);
        // send each box
        for(int k = 0; k < nboxes; ++k) {
          for(Dimension d = 0; d < dimension; ++d) {
            lbnds[d] = v[2 * dimension * k + d];
            ubnds[d] = v[2 * dimension * k + dimension + d];
          }
          traverse(lbnds, ubnds, strs, send_data);
        }
      }

      return sent;
    } // send

    static void get_ngb_color_bounds(bool send,
      meta_data & md,
      std::map<Color, std::vector<int>> & color_bounds) {

      // For each axis, store the neigh color ids, and the
      // lower and upper bounds for sending/receiving data
      A2 ngb_lbnds, ngb_ubnds;
      axes_bounds(send, md, ngb_lbnds, ngb_ubnds);

      // Obtain the set of boxes (lower, upper bounds) that will
      // be sent to a particular color in a map
      A color_strs, center_color, bdepth;
      std::array<bool, dimension> periodic;
      for(Dimension i = 0; i < dimension; i++) {
        color_strs[i] = md[i].colors;
        center_color[i] = md[i].color_index;
        bdepth[i] = md[i].bdepth;
        periodic[i] = md[i].periodic;
      }

      ngb_color_boxes(periodic,
        bdepth,
        color_strs,
        center_color,
        ngb_lbnds,
        ngb_ubnds,
        color_bounds);
    }

    template<typename F>
    static void traverse(A & lbnds, A & ubnds, A & strs, F f) {
      using tb = narray_impl::traverse<dimension>;
      narray_impl::linearize<dimension> ln{strs};
      for(auto && v : tb(lbnds, ubnds)) {
        auto lid = ln(v);
        f(lid);
      }
    }; // traverse

    static void ngb_color_boxes(std::array<bool, dimension> & periodic,
      A & bdepth,
      A & color_strs,
      A & center_color,
      A2 & ngb_lbnds,
      A2 & ngb_ubnds,
      std::map<Color, std::vector<int>> & color_bounds) {

      using nview = flecsi::topo::narray_impl::neighbors_view<dimension>;
      narray_impl::linearize<dimension> ln{color_strs};
      for(auto && v : nview()) {
        A color_indices;
        for(Dimension k = 0; k < dimension; ++k) {
          color_indices[k] = center_color[k] + v[k];
          // if boundary depth, add the correct ngb color
          if(periodic[k]) {
            if((color_indices[k] == -1) && (center_color[k] == 0) && bdepth[k])
              color_indices[k] = color_strs[k] - 1;
            if((color_indices[k] == color_strs[k]) &&
               (center_color[k] == color_strs[k] - 1) && bdepth[k])
              color_indices[k] = 0;
          }
        }

        bool valid_ngb = true;
        for(Dimension k = 0; k < dimension; ++k) {
          if((color_indices[k] == -1) || (color_indices[k] == color_strs[k]))
            valid_ngb = false;
        }

        if(valid_ngb) {
          // get color id
          auto lid = ln(color_indices);

          for(Dimension k = 0; k < dimension; ++k)
            color_bounds[lid].push_back(ngb_lbnds[v[k] + 1][k]);
          for(Dimension k = 0; k < dimension; ++k)
            color_bounds[lid].push_back(ngb_ubnds[v[k] + 1][k]);
        }
      }
    }; // color_boxes

    /*
     *
     * In this method, we figure out:
     * 1) Left side color, lower and upper bounds along the axis which will
     * either be sent or received 2) Center color which is itself, logical lower
     * and upper bounds of this color to send or receive data from lower/upper
     * colors abutting this color 3) Right side color, lower and upper bounds
     * along the axis
     *
     * For example, for a color in the middle of the partition with halo layers
     * on both sides along an axis A:
     *
     *                      Left color      center         Right color
     *   Axis            |-------------|--------------|--------------|
     *
     *  Sending bounds to left         |----|
     *                                   h
     *  Sending bounds to lower/upper  |-------------|
     *                                  logical
     *  Sending bounds to right                 |----|
     *                                            h
     *  Receiving bounds         |----|--------------|----|
     *                             h      logical      h
     *
     * */
    static void
    axes_bounds(bool send, meta_data & md, A2 & ngb_lbnds, A2 & ngb_ubnds) {

      // fill out the indices of the ngb colors
      int p = 0;
      for(auto & ax : md) {
        auto ci = ax.color_index;

        if(send) {
          ngb_lbnds[0][p] = ax.template logical<0>();
          ngb_lbnds[1][p] = ax.template logical<0>();
          ngb_lbnds[2][p] = ax.template logical<1>() - ax.hdepth;

          ngb_ubnds[0][p] = ngb_lbnds[0][p] + ax.hdepth;
          ngb_ubnds[1][p] = ax.template logical<1>();
          ngb_ubnds[2][p] = ax.template logical<1>();

          // bdepths
          if((ci == 0) && ax.bdepth) {
            ngb_ubnds[0][p] = ax.template logical<0>() + ax.bdepth;
          }

          if((ci == ax.colors - 1) && ax.bdepth) {
            ngb_lbnds[2][p] = ax.template logical<1>() - ax.bdepth;
            ngb_ubnds[2][p] = ax.template logical<1>();
          }
        }
        else {
          ngb_lbnds[0][p] = 0;
          ngb_lbnds[1][p] = ax.template logical<0>();
          ngb_lbnds[2][p] = ax.template logical<1>();

          ngb_ubnds[0][p] = ax.hdepth;
          ngb_ubnds[1][p] = ax.template logical<1>();
          ngb_ubnds[2][p] = ax.template logical<1>() + ax.hdepth;

          // bdepths
          if((ci == 0) && ax.bdepth) {
            ngb_lbnds[0][p] = 0;
            ngb_ubnds[0][p] = ax.bdepth;
          }

          if((ci == ax.colors - 1) && ax.bdepth) {
            ngb_lbnds[2][p] = ax.template logical<1>();
            ngb_ubnds[2][p] = ax.template logical<1>() + ax.bdepth;
          }
        }
        ++p;
      }
    }

  }; // struct ragged_impl

  /*--------------------------------------------------------------------------*
    Private data members.
   *--------------------------------------------------------------------------*/
  friend borrow_extra<narray>;

  // fields for storing topology meta data per index-space
  static inline const typename field<util::key_array<meta_data, index_spaces>,
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

  // This key_array of buffers core objects are needed to transfer
  // ragged data. We have a key array over index_spaces because
  // each index_space possibly may have a different communication graph.
  util::key_array<data::buffers::core, index_spaces> buffers_;
}; // struct narray

template<class P>
struct borrow_extra<narray<P>> : borrow_sizes<P> {
  using borrow_extra::borrow_sizes::borrow_sizes;
};

/*----------------------------------------------------------------------------*
  Narray Access.
 *----------------------------------------------------------------------------*/
/// This class is supported for GPU execution.
template<typename Policy>
template<Privileges Priv>
struct narray<Policy>::access {
  ///  This method provides a mdspan of the field underlying data.
  ///  It can be used to create data views with the shape appropriate to S.
  /// This function is \ref topology "host-accessible", although the values in
  /// \a a are typically not.
  template<index_space S, typename T, Privileges P>
  FLECSI_INLINE_TARGET auto mdspan(
    data::accessor<data::dense, T, P> const & a) const {
    auto const s = a.span();
    return util::mdspan(s.data(), extents<S>());
  }
  /// Create a Fortran-like view of a field.
  /// This function is \ref topology "host-accessible", although the values in
  /// \a a are typically not.
  /// \return \c\ref mdcolex
  template<index_space S, typename T, Privileges P>
  FLECSI_INLINE_TARGET auto mdcolex(
    data::accessor<data::dense, T, P> const & a) const {
    return util::mdcolex(a.span().data(), extents<S>());
  }

  template<class F>
  void send(F && f) {
    std::size_t i{0};
    for(auto & a : size_)
      a.topology_send(
        f, [&i](auto & n) -> auto & { return n.get_sizes(i++); });
    const auto meta = [](auto & n) -> auto & {
      return n.meta;
    };
    meta_.topology_send(f, meta);
    policy_meta_.topology_send(f, meta);
  }

private:
  data::scalar_access<narray::policy_meta_field, Priv> policy_meta_;
  util::key_array<data::scalar_access<topo::resize::field, Priv>, index_spaces>
    size_;

  data::scalar_access<narray::meta_field, Priv> meta_;

  /*!
   Method to access global extents of index space S along
   axis A. This function is \ref topology "host-accessible".
    \sa meta_data, axis_color
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET util::gid global() const {
    return get_axis<S, A>().global();
  }

  /*!
   Method to access global offset of the local mesh i.e., the global
   coordinate offset of the local mesh w.r.t the global mesh of index
   space S along axis A.
   This function is \ref topology "host-accessible".
    \sa meta_data, axis_color
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET util::gid offset() const {
    return get_axis<S, A>().offset();
  }

  template<index_space S, axis A>
  FLECSI_INLINE_TARGET util::id extent() const {
    return get_axis<S, A>().extent();
  }

  template<index_space S, auto... A>
  FLECSI_INLINE_TARGET auto extents(util::constants<A...>) const {
    util::key_array<util::gid, axes> ext{{extent<S, A>()...}};
    return ext;
  }

  /*!
    Method to access local extents of all axes of index space S.
    This function is \ref topology "host-accessible".
    \sa meta_data, axis_color
   */
  template<index_space S>
  FLECSI_INLINE_TARGET auto extents() const {
    return extents<S>(axes());
  }

  /*!
     Method to access logical lower/upper bounds of index space S
     along axis A.
     @tparam P Value 0 denotes lower bound, and value 1 denotes upper
               bound.
     This function is \ref topology "host-accessible".
     \sa meta_data, axis_color
    */
  template<index_space S, axis A, std::size_t P>
  FLECSI_INLINE_TARGET util::id logical() const {
    return get_axis<S, A>().template logical<P>();
  }

  /*!
    Method to access extended lower/upper bounds of index space
    S along axis A.
    @tparam P Value 0 denotes lower bound, and value 1 denotes upper
              bound.
    This function is \ref topology "host-accessible".
    \sa meta_data, axis_color
   */
  template<index_space S, axis A, std::size_t P>
  FLECSI_INLINE_TARGET util::id extended() const {
    return get_axis<S, A>().template extended<P>();
  }

protected:
  /// Get the specialization's metadata.
  FLECSI_INLINE_TARGET auto & policy_meta() const {
    return *policy_meta_;
  }

  /*!
   Method to check if an axis of the local mesh is incident on the lower
   bound of the corresponding axis of the global mesh.
   This function is \ref topology "host-accessible".
   \sa meta_data, axis_color
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET bool is_low() const {
    return get_axis<S, A>().is_low();
  }

  /*!
   Method to check if an axis of the local mesh is incident on the upper
   bound of the corresponding axis of the global mesh.
   This function is \ref topology "host-accessible".
   \sa meta_data, axis_color
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET bool is_high() const {
    return get_axis<S, A>().is_high();
  }

  /*!
   Method to check if axis A of index-space S is in between the lower and upper
   bound along axis A of the global domain.
   This function is \ref topology "host-accessible".
   \sa meta_data, axis_color
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET bool is_interior() const {
    return !is_low<S, A>() && !is_high<S, A>();
  }

  /*!
     Method to check if the partition returned by the coloring is degenerate.
     This method checks if the axis A is incident on both the lower and upper
     bound of the global domain. This function is \ref topology
     "host-accessible". \sa meta_data, axis_color
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET bool is_degenerate() const {
    return is_low<S, A>() && is_high<S, A>();
  }

  /*!
     Method returning the global id of a logical index of an index space
     S along axis A. This function is \ref topology
     "host-accessible". \sa meta_data, axis_color
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET util::gid global_id(util::id logical_id) const {
    return get_axis<S, A>().global_id(logical_id);
  }

  /*!
    Method to return size of \c S along \c A for \a DM.
    This function is \ref topology "host-accessible".
    \sa enum domain
  */
  template<index_space S, axis A, domain DM>
  FLECSI_INLINE_TARGET auto size() const {
    if constexpr(DM == domain::logical) {
      return logical<S, A, 1>() - logical<S, A, 0>();
    }
    else if constexpr(DM == domain::extended) {
      return extended<S, A, 1>() - extended<S, A, 0>();
    }
    else if constexpr(DM == domain::all) {
      return extent<S, A>();
    }
    else if constexpr(DM == domain::boundary_low) {
      return logical<S, A, 0>() - extended<S, A, 0>();
    }
    else if constexpr(DM == domain::boundary_high) {
      return extended<S, A, 1>() - logical<S, A, 1>();
    }
    else if constexpr(DM == domain::ghost_low) {
      if(!is_low<S, A>())
        return logical<S, A, 0>();
      else
        return util::id();
    }
    else if constexpr(DM == domain::ghost_high) {
      if(!is_high<S, A>())
        return extent<S, A>() - logical<S, A, 1>();
      else
        return util::id();
    }
    else {
      static_assert(DM == domain::global, "invalid domain identifier");
      return global<S, A>();
    }
  }

  /*!
     Method to return an iterator over the extents of the index-space S along
     axis A for domain DM.
     \tparam DM not \c domain::global
     This function is \ref topology "host-accessible".
   */
  template<index_space S, axis A, domain DM>
  FLECSI_INLINE_TARGET auto range() const {
    if constexpr(DM == domain::logical) {
      return make_ids<S>(
        util::iota_view<util::id>(logical<S, A, 0>(), logical<S, A, 1>()));
    }
    else if constexpr(DM == domain::extended) {
      return make_ids<S>(
        util::iota_view<util::id>(extended<S, A, 0>(), extended<S, A, 1>()));
    }
    else if constexpr(DM == domain::all) {
      return make_ids<S>(util::iota_view<util::id>(0, extent<S, A>()));
    }
    else if constexpr(DM == domain::boundary_low) {
      return make_ids<S>(util::iota_view<util::id>(0, size<S, A, DM>()));
    }
    else if constexpr(DM == domain::boundary_high) {
      return make_ids<S>(util::iota_view<util::id>(
        logical<S, A, 1>(), logical<S, A, 1>() + size<S, A, DM>()));
    }
    else if constexpr(DM == domain::ghost_low) {
      return make_ids<S>(util::iota_view<util::id>(0, size<S, A, DM>()));
    }
    else {
      static_assert(DM == domain::ghost_high, "invalid domain identifier");
      return make_ids<S>(util::iota_view<util::id>(
        logical<S, A, 1>(), logical<S, A, 1>() + size<S, A, DM>()));
    }
  }

  /*!
    Method to return an offset of \c S along \c A for \a DM.
    This function is \ref topology "host-accessible".
    \sa enum domain
  */
  template<index_space S, axis A, domain DM>
  FLECSI_INLINE_TARGET util::gid offset() const {
    if constexpr(DM == domain::logical) {
      return logical<S, A, 0>();
    }
    else if constexpr(DM == domain::extended) {
      return extended<S, A, 0>();
    }
    else if constexpr(DM == domain::all) {
      return 0;
    }
    else if constexpr(DM == domain::boundary_low) {
      return extended<S, A, 0>();
    }
    else if constexpr(DM == domain::boundary_high) {
      return logical<S, A, 1>();
    }
    else if constexpr(DM == domain::ghost_low) {
      return 0;
    }
    else if constexpr(DM == domain::ghost_high) {
      return logical<S, A, 1>();
    }
    else {
      static_assert(DM == domain::global, "invalid domain identifier");
      return offset<S, A>();
    }
  }

private:
  template<axis A>
  FLECSI_TARGET static constexpr std::uint32_t to_idx() {
    return axes::template index<A>;
  }

  template<index_space S, axis A>
  FLECSI_INLINE_TARGET const axis_color & get_axis() const {
    return meta_->template get<S>().template get<A>();
  }
}; // struct narray<Policy>::access

/*----------------------------------------------------------------------------*
  Define Base.
 *----------------------------------------------------------------------------*/

template<>
struct detail::base<narray> {
  using type = narray_base;
}; // struct detail::base<narray>

#ifdef DOXYGEN
/// Example specialization which is not really implemented.
struct narray_specialization : specialization<narray, narray_specialization> {

  /// Enumeration of the axes, they should be
  /// consistent with the dimension of mesh.
  enum axis { x, y };
  /// Axes to store.
  /// The format is\code
  /// has<x, y, ..>
  /// \endcode
  using axes = has<x, y>;

  /// mesh dimension
  static constexpr Dimension dimension = 2;
}
#endif

/// \}
} // namespace topo
} // namespace flecsi

#endif
