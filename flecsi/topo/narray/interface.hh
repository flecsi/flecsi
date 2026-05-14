// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_NARRAY_INTERFACE_HH
#define FLECSI_TOPO_NARRAY_INTERFACE_HH

#include "flecsi/data/accessor.hh"
#include "flecsi/data/copy_plan.hh"
#include "flecsi/data/map.hh"
#include "flecsi/data/privilege.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/core.hh"
#include "flecsi/topo/narray/types.hh"
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

/// Topology category.
template<class P>
using narray = topology<P, narray_base>;

/*!
  Narray Topology.
  Colors are assigned lexicographically; the first dimension varies fastest.
  \tparam Policy the specialization, following
   \ref narray_specialization.
  */
template<typename Policy>
struct topology<Policy, narray_base>
  : narray_base, with_ragged<Policy>, with_meta<Policy> {

  using index_space = typename Policy::index_space;
  using index_spaces = typename Policy::index_spaces;
  using copy_spaces = util::to_copy_spaces<Policy>;

  using Axis = typename Policy::axis;
  using axes = typename Policy::axes;
  using id = util::id;
  static_assert(index_spaces::size, "no index spaces");

  static constexpr Dimension dimension = axes::size;

  template<Privileges>
  struct access;

  topology(scheduler & s, coloring const & c)
    : topology(
        s,
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
  static constexpr IndexSpace index = index_spaces::template index<S>;

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
  [[nodiscard]] const data::copy_plan * ghost_copy(scheduler & s,
    data::field_reference<Type, Layout, Policy, Space> const & f) {
    if constexpr(Layout == data::ragged) {
      using Impl = ragged_impl<Space, Type>;
      ragged_buffers_.template get<Space>()
        .template xfer<Impl::start, Impl::xfer>(s, f, meta_field(this->meta));
      return nullptr;
    }
    else
      return &plan_.template get<Space>();
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
        ret[d] = axcol[d].axis.colors;
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
              auto & ax = axcol[d].axis;
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
      std::vector<
        std::pair<util::id /* ghost */, util::id /* remote shared */>>>;

    using intervals = std::vector<data::subrow>;

    // The index_definition provides the layout of other colors to compute
    // shared offsets.
    static std::pair<points, intervals> ghosts(const index_definition & idef,
      Color i) {
      using narray_impl::linearize;
      const auto md = make(idef, idef.color_indices(i));

      const linearize<dimension> local{md.extent()};
      const linearize<dimension, Color> global{md.colors()};

      points points;
      std::vector<util::id> ghost;
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
  struct policy_meta {
    using Field = field<policy_meta, data::single>;

    util::key_array<meta_data, index_spaces> index;
    typename Policy::meta_data policy;
  };

  template<auto... Value, auto... CI>
  topology(scheduler & s,
    const coloring & c,
    util::constants<Value...>,
    util::constants<CI...> /* deduce pack */)
    : with_ragged<Policy>(s, c.colors()), with_meta<Policy>(s, c.colors()),
      part_{{make_repartitioned<Policy, Value>(c.colors(),
        s,
        [p =
            [&] {
              auto & idef = c.idx_colorings[index<Value>];
              const Color nc = idef.colors();
              std::vector<std::size_t> partitions;
              partitions.reserve(nc);
              for(Color i = 0; i < nc; ++i) {
                auto & total = partitions.emplace_back(1);
                Dimension d = 0;
                for(const auto i : idef.color_indices(i))
                  if(util::ckd_mul(
                       &total, total, idef.make_axis(d++, i)().extent()))
                    flog_fatal("overflow: total number of index points exceed "
                               "std::size_t limits");
              }
              return partitions;
            }()](Color i) { return p[i]; })...}},
      plan_{{make_copy_plan<CI>(s, c.idx_colorings[index<CI>])...}},
      ragged_buffers_{{data::buffers::topology(s,
        meta_data::peers(c.idx_colorings[index<CI>]))...}} {
    s.execute<set_meta<Value...>>(exec::on, meta_field(this->meta), &c).wait();
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

  static void set_dests(exec::cpu s,
    field<data::intervals::Value>::accessor<wo> a,
    const index_definition * idef) noexcept {
    const auto c = s.launch().index;
    util::id i = 0;
    for(auto & it : meta_data::ghosts(*idef, c).second)
      a[i++] = data::intervals::make({it.first, it.second}, c);
  }

  template<index_space S>
  static void set_ptrs(exec::cpu s,
    field<data::copy_engine::Point>::accessor1<
      privilege_repeat<wo, Policy::template privilege_count<S>>> a,
    const index_definition * idef) noexcept {
    for(const auto & [own, gg] :
      meta_data::ghosts(*idef, s.launch().index).first)
      for(const auto & [l, r] : gg)
        a[l] = data::copy_engine::point(own, r);
  }

  template<index_space S>
  data::copy_plan make_copy_plan(scheduler & s, index_definition const & idef) {
    for(auto & ax : idef.axes)
      ax.check_halo();
    // Call meta_data::ghosts several times to avoid communication:
    return {s,
      *this,
      [&idef] {
        const Color nc = idef.colors();
        data::copy_plan::Sizes ret;
        ret.reserve(nc);
        for(Color i = 0; i < nc; ++i)
          ret.push_back(meta_data::ghosts(idef, i).second.size());
        return ret;
      }(),
      [&](auto f) { s.execute<set_dests>(exec::on, f, &idef).wait(); },
      [&](auto f) { s.execute<set_ptrs<S>>(exec::on, f, &idef).wait(); },
      util::constant<S>()};
  }

  template<auto... Value> // index_spaces
  static void set_meta(exec::cpu s,
    typename policy_meta::Field::template accessor<wo> m,
    const coloring * c) noexcept {
    IndexSpace index{0};
    (
      [&] {
        const auto & idef = c->idx_colorings[index++];
        m->index.template get<Value>() =
          meta_data::make(idef, idef.color_indices(s.launch().index));
      }(),
      ...);
  }

  auto & get_sizes(IndexSpace i) {
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

    using mfa = typename policy_meta::Field::template accessor<ro>;

    static void start(fa v, mfa mf, data::buffers::Start mv) noexcept {
      send(v,
        mf,
        true,
        mv,
        get_ngb_color_bounds(true, mf->index.template get<Space>()));
    } // start

    static int xfer(fm_rw g, mfa mf, data::buffers::Transfer mv) noexcept {
      // get the meta data for the index space
      meta_data md = mf->index.template get<Space>();

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
      meta_data md = mf->index.template get<Space>();
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
  friend borrow_extra<topology>;

  // fields for storing topology meta data per index-space
  static inline const typename policy_meta::Field::template definition<
    meta<Policy>>
    meta_field;

  // index-space specific parts
  util::key_array<repartitioned, index_spaces> part_;
  util::key_array<data::copy_plan, copy_spaces> plan_;
  util::key_array<data::buffers::topology, copy_spaces> ragged_buffers_;
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
struct topology<Policy, narray_base>::access {
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
    IndexSpace i{0};
    for(auto & a : size_)
      f(a, [&i](auto & n) { return topo::resize::field(n.get_sizes(i++)); });
    std::forward<F>(f)(meta_, [](auto & n) { return meta_field(n.meta); });
  }

private:
  util::key_array<
    data::scalar_access<topo::resize::Field::value_type, privilege_merge(Priv)>,
    index_spaces>
    size_;

  data::scalar_access<topology::policy_meta, privilege_merge(Priv)> meta_;

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
  template<index_space S, Axis A>
  FLECSI_INLINE_TARGET util::gid global() const {
    return get_axis<S, A>().axis.extent;
  }

  /*!
   Method to access global offset of the local mesh i.e., the global
   coordinate offset of the local mesh w.r.t the global mesh of index
   space S along axis A.
   \host.
  */
  template<index_space S, Axis A>
  FLECSI_INLINE_TARGET util::gid offset() const {
    return get_axis<S, A>().offset;
  }

  template<index_space S, Axis A>
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
  template<index_space S, Axis A, axis_layout::End P>
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
  template<index_space S, Axis A, axis_layout::End P>
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
    return meta_->policy;
  }

  /// Get axis information.
  /// \host.
  template<index_space S, Axis A>
  FLECSI_INLINE_TARGET axis_info axis() const {
    return get_axis<S, A>();
  }

  /*!
   Method to check if an axis of the local mesh is incident on the lower
   bound of the corresponding axis of the global mesh.
   \host.
   \deprecated Use \c axis_color::low.
  */
  template<index_space S, Axis A>
  [[deprecated("use axis_color::low")]] FLECSI_INLINE_TARGET bool
  is_low() const {
    return get_axis<S, A>().low();
  }

  /*!
   Method to check if an axis of the local mesh is incident on the upper
   bound of the corresponding axis of the global mesh.
   \host.
   \deprecated Use \c axis_color::high.
  */
  template<index_space S, Axis A>
  [[deprecated("use axis_color::high")]] FLECSI_INLINE_TARGET bool
  is_high() const {
    return get_axis<S, A>().high();
  }

  /*!
   Method to check if axis A of index-space S is in between the lower and upper
   bound along axis A of the global domain.
   \host.
   \deprecated Use \c axis_color::low and \c axis_color::high.
  */
  template<index_space S, Axis A>
  [[deprecated("use axis_color::low and axis_color::high")]]
  FLECSI_INLINE_TARGET bool is_interior() const {
    return !is_low<S, A>() && !is_high<S, A>();
  }

  /*!
     Method to check if the partition returned by the coloring is degenerate.
     This method checks if the axis A is incident on both the lower and upper
     bound of the global domain.  \host.
     \deprecated Use \c axis_color::low and \c axis_color::high.
  */
  template<index_space S, Axis A>
  [[deprecated("use axis_color::low and axis_color::high")]]
  FLECSI_INLINE_TARGET bool is_degenerate() const {
    return is_low<S, A>() && is_high<S, A>();
  }

  /*!
     Method returning the global id of a logical index of an index space
     \a S along axis \a A.  If \a logical_id refers to a boundary point, it is
     treated as periodic.
     \host.
     \deprecated Use \c axis_color::global_id.
  */
  template<index_space S, Axis A>
  [[deprecated("use axis_color::global_id")]] FLECSI_INLINE_TARGET util::gid
  global_id(util::id logical_id) const {
    return get_axis<S, A>().global_id(logical_id);
  }

  /*!
    Method to return size of \c S along \c A for \a DM.
    \host.
    \deprecated Use \c axis_layout.
  */
  template<index_space S, Axis A, domain DM>
  [[deprecated("use axis_layout")]] FLECSI_INLINE_TARGET auto size() const {
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
     \deprecated Use \c axis_layout.
   */
  template<index_space S, Axis A, domain DM>
  [[deprecated("use axis_layout")]] FLECSI_INLINE_TARGET auto range() const {
    static_assert(DM != domain::global, "no global range");
    const auto o = offset<S, A, DM>();
    return make_ids<S>(util::iota_view<util::id>(o, o + size<S, A, DM>()));
  }

  /*!
    Method to return an offset of \c S along \c A for \a DM.
    \host.
    \deprecated Use \c axis_layout.
  */
  template<index_space S, Axis A, domain DM>
  [[deprecated("use axis_layout")]] FLECSI_INLINE_TARGET util::gid
  offset() const {
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
  template<index_space S, Axis A>
  FLECSI_INLINE_TARGET const axis_color & get_axis() const {
    return meta_->index.template get<S>().axcol.template get<A>();
  }
}; // struct narray<Policy>::access

template<>
struct detail::base<narray> {
  using type = narray_base;
}; // struct detail::base<narray>

#ifdef DOXYGEN
/// Example specialization which is not really implemented.
/// \remark Previously, a constant \c dimension was required, but the size of
///   \c axes already provides that information.
struct narray_specialization : specialization<narray, narray_specialization> {

  /// Axis enumeration.
  enum axis { x, y };
  /// Axes to store.
  /// The format is\code
  /// has<x, y, ..>
  /// \endcode
  using axes = has<x, y>;

  /// Specialization-specific data to store once per color.
  struct meta_data {};
};
#endif

/// \}
} // namespace topo
} // namespace flecsi

#endif
