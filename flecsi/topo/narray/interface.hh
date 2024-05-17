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
  Colors are assigned lexicographically; the first dimension varies fastest.
  \tparam Policy the specialization, following
   \ref narray_specialization.
  */
template<typename Policy>
struct narray : narray_base, with_ragged<Policy>, with_meta<Policy> {

  using index_space = typename Policy::index_space;
  using index_spaces = typename Policy::index_spaces;
  using copy_spaces = util::to_copy_spaces<Policy>;

  using axis = typename Policy::axis;
  using axes = typename Policy::axes;
  using id = util::id;
  static_assert(index_spaces::size, "no index spaces");

  static constexpr Dimension dimension = Policy::dimension;
  static_assert(dimension == axes::size);

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

  template<typename Type,
    data::layout Layout,
    typename Policy::index_space Space>
  void ghost_copy(
    data::field_reference<Type, Layout, Policy, Space> const & f) {
    if constexpr(Layout == data::ragged) {
      using Impl = ragged_impl<Space, Type>;
      ragged_buffers_.template get<Space>()
        .template xfer<Impl::start, Impl::xfer>(f, meta_field(this->meta));
    }
    else
      plan_.template get<Space>().issue_copy(f.fid());
  }

private:
  using coord = std::array<util::id, dimension>;
  using hypercube = std::pair<coord, coord>;
  using nview = narray_impl::neighbors_view<dimension>;
  // Structural information about one color.
  struct meta_data {
    using Colors = std::array<Color, dimension>;

    util::key_array<axis_color, axes> axcol;
    bool diagonals;

    // Dynamically-sized parameter type for client convenience.
    static meta_data make(const index_definition & idef,
      const narray_impl::colors & ci) {
      if(ci.size() != dimension)
        flog_fatal("need " << dimension << " axes, not " << ci.size());
      meta_data ret;
      for(Dimension d = 0; d < dimension; ++d)
        ret.axcol[d] = idef.make_axis(d, ci[d]);
      ret.diagonals = idef.diagonals;
      return ret;
    }

    coord extent() const {
      coord ret;
      for(Dimension d = 0; d < dimension; ++d)
        ret[d] = axcol[d]().extent();
      return ret;
    }
    Colors colors() const {
      Colors ret;
      for(Dimension d = 0; d < dimension; ++d)
        ret[d] = axcol[d].ax.colors;
      return ret;
    }

    struct message {
      typename nview::value_type neighbor;
      hypercube region; // ghost or shared
    };
    // NB: periodicity can give the same neighbor more than once.
    std::vector<message> traffic(bool send = true) const {
      std::vector<message> ret;
      for(const auto off : nview()) {
        bool skin[dimension]{};
        if(!diagonals) {
          Dimension taxi = 0;
          for(Dimension d = 0; d < dimension; ++d)
            if(off[d])
              ++taxi;
          if(taxi > 1) {
            // We can still need to communicate with some diagonal neighbors
            // those auxiliaries incident on another color's primaries.
            for(Dimension d = 0; d < dimension; ++d) {
              auto & ax = axcol[d].ax;
              if((skin[d] = ax.auxiliary && ax.full_ghosts &&
                            (send ? -1 : 1) * off[d] > 0))
                --taxi;
            }
            if(taxi > 1)
              continue;
          }
        }
        auto & msg = ret.emplace_back();
        msg.neighbor = off;
        auto lo = msg.region.first.begin(), hi = msg.region.second.begin();
        for(Dimension d = 0; d < dimension; ++d) {
          const short o = off[d];
          const axis_layout al = axcol[d](skin[d]);
          const auto log0 = al.logical<0>(), log1 = al.logical<1>();
          // Each bound of the communication region has a case each for lower
          // neighbors, upper neighbors, and peer neighbors.
          *lo = o < 0 ? send ? log0 : al.ghost<0>()
                : o ? send ? al.exclusive<1>() : log1
                      : log0;
          *hi = o < 0 ? send ? al.exclusive<0>() : log0
                : o   ? send ? log1 : al.ghost<1>()
                      : log1;
          if(*lo++ == *hi++) {
            ret.pop_back(); // region empty
            break;
          }
        }
      }
      return ret;
    }

    Colors neighbor(typename nview::value_type off) const {
      Colors ret;
      for(Dimension d = 0; d < dimension; ++d)
        ret[d] = axcol[d].color_step(off[d]);
      return ret;
    }

    using points = std::map<Color,
      std::vector<std::pair</* local ghost offset, remote shared offset */
        std::size_t,
        std::size_t>>>;

    using intervals = std::vector<std::pair<std::size_t, std::size_t>>;

    // The index_definition provides the layout of other colors to compute
    // shared offsets.
    static std::pair<points, intervals> ghosts(const index_definition & idef,
      const narray_impl::colors & ci) {
      using narray_impl::linearize;
      const auto md = make(idef, ci);

      const linearize<dimension> local{md.extent()};
      const linearize<dimension, Color> global{md.colors()};

      points points;
      std::vector<std::size_t> ghost;
      for(const auto & [ngh, reg] : md.traffic(false)) {
        const auto src = md.neighbor(ngh);
        linearize<dimension> remote;
        coord roff;
        for(Dimension d = 0; d < dimension; ++d) {
          const short c = ngh[d];
          const axis_layout ax = md.axcol[d](), r = idef.make_axis(d, src[d])();
          remote.strs[d] = r.extent();
          // Choose a corresponding pair of local indices to compute delta:
          roff[d] = c < 0 ? r.logical<1>() - ax.logical<0>()
                    : c   ? r.logical<0>() - ax.logical<1>() // "negative"
                          : 0;
        }
        auto & pts = points[global(src)];
        for(coord g = reg.first;;) {
          coord s = g;
          for(Dimension d = 0; d < dimension; ++d)
            s[d] += roff[d];
          pts.emplace_back(ghost.emplace_back(local(g)), remote(s));
          // Advance odometer:
          auto it = g.begin(), e = g.end();
          auto b = reg.first.begin(), u = reg.second.begin();
          for(; it != e && ++*it == *u; *it++ = *b++, ++u)
            ;
          if(it == e)
            break;
        }
      }

      std::sort(ghost.begin(), ghost.end());
      return {std::move(points), rle(ghost)};
    }

    static std::vector<std::vector<Color>> peers( // send graph
      const index_definition & idef) {
      narray_impl::linearize<dimension, Color> global;
      for(Dimension k = 0; k < dimension; ++k) {
        global.strs[k] = idef.axes[k].colormap.size();
      }

      std::vector<std::vector<Color>> peer;
      peer.reserve(idef.colors());

      // Loop over all colors
      for(auto && v : narray_impl::traverse<dimension>({}, global.strs)) {
        auto & cc = peer.emplace_back();
        const auto md = make(idef, {v.begin(), v.end()});
        for(const auto & msg : md.traffic())
          cc.push_back(global(md.neighbor(msg.neighbor)));
        // Periodicity denies both ordering and uniqueness:
        std::sort(cc.begin(), cc.end());
        cc.erase(std::unique(cc.begin(), cc.end()), cc.end());
      }

      return peer;
    }
  };

  template<auto... Value, auto... CI>
  narray(const coloring & c,
    util::constants<Value...>,
    util::constants<CI...> /* deduce pack */)
    : with_ragged<Policy>(c.colors()), with_meta<Policy>(c.colors()),
      part_{{make_repartitioned<Policy, Value>(c.colors(),
        make_partial<idx_size>([&]() {
          auto & idef = c.idx_colorings[index<Value>];
          std::vector<std::size_t> partitions;
          for(const auto & ci : idef.process_colors()) {
            auto & total = partitions.emplace_back(1);
            Dimension d = 0;
            for(const auto i : ci)
              total *= idef.make_axis(d++, i)().extent();
          }
          concatenate(partitions, c.colors(), MPI_COMM_WORLD);
          return partitions;
        }()))...}},
      plan_{{make_copy_plan<CI>(c.colors(), c.idx_colorings[index<CI>])...}},
      ragged_buffers_{{data::buffers::core(
        meta_data::peers(c.idx_colorings[index<CI>]))...}} {
    auto lm = data::launch::make(this->meta);
    execute<set_meta<Value...>, mpi>(meta_field(lm), c);
    init_policy_meta(c);
    (
      [&] { // Sanity checks for indexes spaces for which privilege count is 1
        if(Policy::template privilege_count<Value> == 1) {
          if(c.idx_colorings[index<Value>].full_ghosts)
            throw std::invalid_argument(
              "Privilege count is 1 but `full_ghosts` is set to `true`");
          for(auto & axis_def : c.idx_colorings[index<Value>].axes)
            if(axis_def.hdepth != 0)
              throw std::invalid_argument(
                "Privilege count is 1 but `axis_definition::hdepth` are "
                "non-zero");
        }
      }(),
      ...);
  }

  /*!
   Method to compute the local ghost "intervals" and "points" which store map of
   local ghost offset to remote/shared offset. This method is called by the
   "make_copy_plan" method in the derived topology to create the copy plan
   objects.

   @param idef index definition
   @param[out] num_intervals vector of number of ghost intervals, over all
   colors, this vector is assumed to be sized correctly (all colors)
   @param[out] intervals  vector of local ghost intervals, over process colors
   @param[out] points vector of maps storing (local ghost offset, remote shared
   offset) for a shared color, over process colors
  */
  static void idx_itvls(index_definition const & idef,
    std::vector<std::size_t> & num_intervals,
    std::vector<typename meta_data::intervals> & intervals,
    std::vector<typename meta_data::points> & points,
    MPI_Comm const & comm) {
    std::vector<std::size_t> local_itvls;
    for(const auto & c : idef.process_colors(comm)) {
      auto [pts, itvls] = meta_data::ghosts(idef, c);
      local_itvls.emplace_back(itvls.size());
      intervals.emplace_back(std::move(itvls));
      points.emplace_back(std::move(pts));
    }

    /*
      Gather global interval sizes.
     */

    auto global_itvls = util::mpi::all_gatherv(local_itvls, comm);

    auto it = num_intervals.begin();
    for(const auto & pv : global_itvls) {
      for(auto i : pv) {
        *it++ = i;
      }
    }
  } // idx_itvls

  /*!
   Method to create copy plans for entities of an index-space.
   @param colors  The number of colors
   @param idef index definition
   @param p partition
  */
  template<index_space S>
  data::copy_plan make_copy_plan(Color colors, index_definition const & idef) {

    std::vector<std::size_t> num_intervals(colors, 0);
    std::vector<typename meta_data::intervals> intervals;
    std::vector<typename meta_data::points> points;

    // The intervals encode local ghost
    // intervals, whereas points capture the  local offset and corresponding
    // remote/shared offset on remote/shared color. The intervals and points are
    // used to create function objects "dest_tasks" and "ptrs_tasks" that is
    // subsequently used by copy plan to perform the data communication. Note
    // that after this call the copy plan objects have been created. The actual
    // communication is invoked as part of task execution depending upon the
    // privilege requirements of the task.

    idx_itvls(idef, num_intervals, intervals, points, MPI_COMM_WORLD);

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

    return {*this, num_intervals, dest_task, ptrs_task, util::constant<S>()};
  }

  template<auto... Value> // index_spaces
  static void set_meta(
    data::multi<typename field<util::key_array<meta_data, index_spaces>,
      data::single>::template accessor<wo>> mm,
    narray_base::coloring const & c) {
    std::size_t index{0};
    ((
       [&] {
         const auto ma = mm.accessors();
         const auto & idef = c.idx_colorings[index];
         auto it = ma.begin();
         for(const auto & ci : idef.process_colors())
           (*it++)->template get<Value>() = meta_data::make(idef, ci);
       }(),
       ++index),
      ...);
  }

  static void set_policy_meta(typename field<typename Policy::meta_data,
    data::single>::template accessor<wo>) {}
  void init_policy_meta(narray_base::coloring const &) {
    execute<set_policy_meta>(policy_meta_field(this->meta));
  }

  auto & get_sizes(std::size_t i) {
    return part_[i].sz;
  }

  /// Ragged communication routines
  template<typename Policy::index_space Space, typename T>
  struct ragged_impl {
    using A = coord;

    // Sorted to match the order created by meta_data::peers.
    using Bounds = std::map<Color, std::vector<hypercube>>;

    static constexpr PrivilegeCount N = Policy::template privilege_count<Space>;
    using fa = typename field<T,
      data::ragged>::template accessor1<privilege_ghost_repeat<ro, na, N>>;

    using fm_rw = typename field<T,
      data::ragged>::template mutator1<privilege_ghost_repeat<ro, rw, N>>;

    using mfa = typename field<util::key_array<meta_data, index_spaces>,
      data::single>::template accessor<ro>;

    static void start(fa v, mfa mf, data::buffers::Start mv) {
      send(
        v, mf, true, mv, get_ngb_color_bounds(true, mf->template get<Space>()));
    } // start

    static int xfer(fm_rw g, mfa mf, data::buffers::Transfer mv) {
      // get the meta data for the index space
      meta_data md = mf->template get<Space>();

      // The start of receive buffer id depends on the number
      // of sent buffers.
      const Bounds color_bounds_send = get_ngb_color_bounds(true, md);

      const auto strs = md.extent();

      // Loop over the receiving colors and receive the data for all boxes
      int bufid = color_bounds_send.size();
      for(auto & [c, v] : get_ngb_color_bounds(false, md)) {
        std::vector<int> lids;
        auto get_lids = [&](int lid) {
          lids.push_back(lid);
          return false;
        };

        // get lids of all boxes
        for(auto i = v.rbegin(), e = v.rend(); i != e; ++i)
          traverse(i->first, i->second, strs, get_lids);

        // receive
        data::buffers::ragged::read(g, mv[bufid], lids);
        ++bufid;
      }

      // resume transfer if data was not fully packed during start
      return send(g, mf, false, mv, color_bounds_send);
    } // xfer

  private:
    // color_bounds could be computed, but xfer already needs it.
    template<typename F, typename B>
    static bool
    send(F f, mfa mf, bool first, B mv, const Bounds & color_bounds) {
      // get the meta data for the index space
      meta_data md = mf->template get<Space>();
      bool sent = false;

      const auto strs = md.extent();

      int p = 0;
      for(auto & [c, v] : color_bounds) {
        auto b = data::buffers::ragged{mv[p++], first};
        auto send_data = [&](int lid) { return !b(f, lid, sent); };

        for(auto & h : v)
          traverse(h.first, h.second, strs, send_data);
      }

      return sent;
    } // send

    static Bounds get_ngb_color_bounds(bool send, const meta_data & md) {
      const narray_impl::linearize<dimension, Color> global{md.colors()};

      Bounds ret;
      for(const auto & [ngh, reg] : md.traffic(send))
        ret[global(md.neighbor(ngh))].push_back(reg);
      return ret;
    }

    template<typename F>
    static void
    traverse(const A & lbnds, const A & ubnds, const A & strs, F f) {
      using tb = narray_impl::traverse<dimension>;
      const narray_impl::linearize<dimension> ln{strs};
      for(auto && v : tb(lbnds, ubnds)) {
        auto lid = ln(v);
        if(f(lid))
          break;
      }
    } // traverse
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
  util::key_array<data::copy_plan, copy_spaces> plan_;
  util::key_array<data::buffers::core, copy_spaces> ragged_buffers_;
}; // struct narray

template<class P>
struct borrow_extra<narray<P>> : borrow_sizes<P> {
  using borrow_extra::borrow_sizes::borrow_sizes;
};

/// Topology interface base.
/// \gpu.
/// \see specialization_base::interface
template<typename Policy>
template<Privileges Priv>
struct narray<Policy>::access {
  ///  This method provides a mdspan of the field underlying data.
  ///  It can be used to create data views with the shape appropriate to S.
  /// \host, although the values in
  /// \a a are typically not.
  template<index_space S, typename T, Privileges P>
  FLECSI_INLINE_TARGET auto mdspan(
    data::accessor<data::dense, T, P> const & a) const {
    auto const s = a.span();
    return util::mdspan(s.data(), check_extents<S>(s));
  }
  /// Create a Fortran-like view of a field.
  /// \host, although the values in
  /// \a a are typically not.
  /// \return \c util::mdcolex
  template<index_space S, typename T, Privileges P>
  FLECSI_INLINE_TARGET auto mdcolex(
    data::accessor<data::dense, T, P> const & a) const {
    const auto s = a.span();
    return util::mdcolex(s.data(), check_extents<S>(s));
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

  template<index_space S, class C>
  auto check_extents(const C & c) const {
    const auto e = extents<S>();
    const auto sz = std::apply([](auto... ii) { return (util::gid(ii) * ...); },
      static_cast<const typename decltype(e)::array &>(e));
    flog_assert(
      sz == c.size(), "field has size " << c.size() << ", not " << sz);
    return e;
  }

  /*!
   Method to access global extents of index space S along
   axis A.  \host.
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET util::gid global() const {
    return get_axis<S, A>().ax.extent;
  }

  /*!
   Method to access global offset of the local mesh i.e., the global
   coordinate offset of the local mesh w.r.t the global mesh of index
   space S along axis A.
   \host.
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET util::gid offset() const {
    return get_axis<S, A>().offset;
  }

  template<index_space S, axis A>
  FLECSI_INLINE_TARGET util::id extent() const {
    return get_axis<S, A>()().extent();
  }

  template<index_space S, auto... A>
  FLECSI_INLINE_TARGET auto extents(util::constants<A...>) const {
    util::key_array<util::gid, axes> ext{{extent<S, A>()...}};
    return ext;
  }

  /*!
    Method to access local extents of all axes of index space S.
    \host.
   */
  template<index_space S>
  FLECSI_INLINE_TARGET auto extents() const {
    return extents<S>(axes());
  }

  /*!
     Method to access logical lower/upper bounds of index space S
     along axis A.
     \host.
     @tparam P Value 0 denotes lower bound, and value 1 denotes upper
               bound.
    */
  template<index_space S, axis A, std::size_t P>
  FLECSI_INLINE_TARGET util::id logical() const {
    return get_axis<S, A>()().template logical<P>();
  }

  /*!
    Method to access extended lower/upper bounds of index space
    S along axis A.
    \host.
    @tparam P Value 0 denotes lower bound, and value 1 denotes upper
              bound.
   */
  template<index_space S, axis A, std::size_t P>
  FLECSI_INLINE_TARGET util::id extended() const {
    const axis_color & a = get_axis<S, A>();
    if constexpr(P == 0) {
      return a.low() ? 0 : a().logical<P>();
    }
    return a.high() ? a().extent() : a().logical<P>();
  }

protected:
  /// Get the specialization's metadata.
  /// \host.
  FLECSI_INLINE_TARGET auto & policy_meta() const {
    return *policy_meta_;
  }

  /*!
   Method to check if an axis of the local mesh is incident on the lower
   bound of the corresponding axis of the global mesh.
   \host.
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET bool is_low() const {
    return get_axis<S, A>().low();
  }

  /*!
   Method to check if an axis of the local mesh is incident on the upper
   bound of the corresponding axis of the global mesh.
   \host.
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET bool is_high() const {
    return get_axis<S, A>().high();
  }

  /*!
   Method to check if axis A of index-space S is in between the lower and upper
   bound along axis A of the global domain.
   \host.
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET bool is_interior() const {
    return !is_low<S, A>() && !is_high<S, A>();
  }

  /*!
     Method to check if the partition returned by the coloring is degenerate.
     This method checks if the axis A is incident on both the lower and upper
     bound of the global domain.  \host.
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET bool is_degenerate() const {
    return is_low<S, A>() && is_high<S, A>();
  }

  /*!
     Method returning the global id of a logical index of an index space
     \a S along axis \a A.  If \a logical_id refers to a boundary point, it is
     treated as periodic.
     \host.
  */
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET util::gid global_id(util::id logical_id) const {
    return get_axis<S, A>().global_id(logical_id);
  }

  /*!
    Method to return size of \c S along \c A for \a DM.
    \host.
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
     \host.
     \tparam DM not \c domain::global
   */
  template<index_space S, axis A, domain DM>
  FLECSI_INLINE_TARGET auto range() const {
    static_assert(DM != domain::global, "no global range");
    const auto o = offset<S, A, DM>();
    return make_ids<S>(util::iota_view<util::id>(o, o + size<S, A, DM>()));
  }

  /*!
    Method to return an offset of \c S along \c A for \a DM.
    \host.
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
  template<index_space S, axis A>
  FLECSI_INLINE_TARGET const axis_color & get_axis() const {
    return meta_->template get<S>().axcol.template get<A>();
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

  /// Axis enumeration.
  enum axis { x, y };
  /// Axes to store.
  /// Must have as many elements as \c dimension.
  /// The format is\code
  /// has<x, y, ..>
  /// \endcode
  using axes = has<x, y>;

  /// mesh dimension
  static constexpr Dimension dimension = 2;

  /// Specialization-specific data to store once per color.
  struct meta_data {};
};
#endif

/// \}
} // namespace topo
} // namespace flecsi

#endif
