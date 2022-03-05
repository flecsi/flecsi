// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_UNSTRUCTURED_TYPES_HH
#define FLECSI_TOPO_UNSTRUCTURED_TYPES_HH

#include "flecsi/data/field_info.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/index.hh"
#include "flecsi/util/common.hh"
#include "flecsi/util/crs.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/serialize.hh"

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <vector>

namespace flecsi {
namespace topo {
/// \addtogroup unstructured
/// \{
namespace unstructured_impl {

/// Information about an entity that is shared with other colors.
struct shared_entity {
  /// Global id.
  std::size_t id;
  /// The \e colors with which this entity is shared.
  std::vector<std::size_t> dependents;

  /*
    These operators are designed to be used with util::force_unique, which
    first applies std::sort (using operator<), and then applies std::unique
    (using operator==). The implementation of the operators is designed to be
    greedy, i.e., it favors shared entity variants that have more dependents by
    placing them first in the sorted order. When std::unique is applied, the
    variant with the most dependents will be kept.
   */

  bool operator<(const shared_entity & s) const {
    return id == s.id ? dependents.size() > s.dependents.size()
                      : /* greedy dependendents */
             id < s.id;
  }

  bool operator==(const shared_entity & s) const {
    return id == s.id; // ignores dependendents
  }
};

inline std::ostream &
operator<<(std::ostream & stream, shared_entity const & s) {
  stream << "<" << s.id << ": ";
  bool first = true;
  for(auto d : s.dependents) {
    if(first)
      first = false;
    else
      stream << ", ";
    stream << d;
  } // for
  stream << ">";
  return stream;
}

/// Information about an entity owned by another color.
struct ghost_entity {
  /// Global id.
  std::size_t id;
  /// Owning process.
  Color process;
  /// Color within those of \e process.
  Color local;
  /// Owning color.
  Color global;

  bool operator<(const ghost_entity & g) const {
    return id < g.id;
  }

  bool operator==(const ghost_entity & g) const {
    return id == g.id && process == g.process && local == g.local &&
           global == g.global;
  }
};

inline std::ostream &
operator<<(std::ostream & stream, ghost_entity const & g) {
  stream << "<" << g.id << ":" << g.process << ":" << g.local << ":" << g.global
         << ">";
  return stream;
}

/// Information for one index space and one color.
struct index_coloring {
  /// Global ids used by this color.
  std::vector<std::size_t> all;
  /// Global ids owned by this color.
  std::vector<std::size_t> owned;
  /// The subset of \c owned that are not ghosts on any other color.
  std::vector<std::size_t> exclusive;
  /// Entities that are ghosts on another color.
  std::vector<shared_entity> shared;
  /// Entities that are owned by another color.
  std::vector<ghost_entity> ghost;
};

inline std::ostream &
operator<<(std::ostream & stream, index_coloring const & ic) {
  stream << "all\n" << log::container{ic.all};
  stream << "owned\n" << log::container{ic.owned};
  stream << "exclusive\n" << log::container{ic.exclusive};
  stream << "shared\n" << log::container{ic.shared};
  stream << "ghost\n" << log::container{ic.ghost};
  return stream;
}

inline void
transpose(field<util::id, data::ragged>::accessor<ro, ro, na> input,
  field<util::id, data::ragged>::mutator<wo, wo, na> output) {
  std::size_t e = 0;
  for(auto && i : input) {
    for(auto v : i)
      output[v].push_back(e);
    ++e;
  }
}

/// Strategy for contructing colorings.
struct coloring_definition {
  /// Total number of colors.
  Color colors;
  /// Index of primary entity in \c index_spaces.
  /// \warning Not an \c index_space enumerator \b value.
  std::size_t idx;
  /// Dimensionality of the mesh.
  std::size_t dim;
  /// Number of layers of ghosts needed.
  std::size_t depth;

  /// Index of vertices in \c index_spaces.
  std::size_t vidx;

  struct auxiliary {
    std::size_t idx;
    std::size_t dim;
    bool cnx;
  };

  /// Information for auxiliary entities.
  std::vector<auxiliary> aux;
};

/// Information specific to a local color.
struct process_coloring {

  /*!
    Global color.
   */

  Color color;

  /*!
    The global number of entities in this index space.
   */

  std::size_t entities;

  /*!
    The local coloring information for this index space.

    The coloring information is expressed in the mesh index space,
    i.e., the ids are global.
   */

  index_coloring coloring;

  /*!
    Communication peers (needed for ragged/sparse buffer creation).
   */

  std::vector<Color> peers;

  /*!
   The local allocation size for each connectivity. The vector is over
   connectivities between this entity type and another entity type. The
   ordering follows that given in the specialization policy.
   */

  std::vector<std::size_t> cnx_allocs;

  /*!
    The local graph for each connectivity.

    The graph information is expressed in the mesh index space,
    i.e., the ids are global. The vector is like cnx_alloc.
   */

  std::vector<util::crs> cnx_colorings;
}; // struct process_coloring

inline std::ostream &
operator<<(std::ostream & stream, process_coloring const & pc) {
  stream << "color: " << pc.color << std::endl;
  stream << "entities: " << pc.entities << std::endl;
  stream << "coloring:\n" << pc.coloring << std::endl;
  stream << "cnx_allocs:\n" << log::container{pc.cnx_allocs} << std::endl;
  stream << "cnx_colorings:\n" << log::container{pc.cnx_colorings} << std::endl;
  return stream;
}

/*
  Type for mapping ragged ghosts into buffers.
 */

struct cmap {
  Color c;
  std::size_t lid;
  std::size_t rid;
};

} // namespace unstructured_impl

struct unstructured_base {

  using index_coloring = unstructured_impl::index_coloring;
  using process_coloring = unstructured_impl::process_coloring;
  using ghost_entity = unstructured_impl::ghost_entity;
  using crs = util::crs;
  using cmap = unstructured_impl::cmap;

  using source_pointers = std::vector</* over local colors */
    std::map</* over global source colors */
      Color,
      std::vector</* over color source pointers */
        std::pair<std::size_t /* global id */, std::size_t /* offset */>>>>;

  using destination_intervals = std::vector</* over local colors */
    std::vector</* over contiguous intervals */
      data::subrow>>;

  /// Coloring type.
  /// \ingroup unstructured
  struct coloring {
    /*!
      Communicator over which the coloring is distributed.
     */
    MPI_Comm comm;

    /// Global number of colors.
    Color colors;

    /// List of colors assigned for each process.
    std::vector<std::vector<Color>> process_colors;

    /// Count of communication peers for each global color.
    std::vector<std::size_t> color_peers;

    /// Communication peers for each global color for each index space.
    std::vector<std::vector<std::vector<Color>>> peers;

    /// Number of index points for each global color for each index space.
    std::vector<std::vector<std::size_t>> partitions;

    /// Detailed information for each local color for each index space.
    std::vector<std::vector<process_coloring>> idx_spaces;
  }; // struct coloring

  template<class A>
  using borrow_array = typename A::template map_type<borrow_base::wrap>;

  static std::size_t idx_size(std::vector<std::size_t> vs, std::size_t c) {
    return vs[c];
  }

  /*
    Using the Mesh Index Space (MIS) ordering, compute intervals,
    and the number of intervals for each color. Also compute the
    point offsets. The references num_intervals, intervals,
    and pointers, respectively, are filled with this information.
   */

  template<PrivilegeCount N>
  static void idx_itvls(std::vector<process_coloring> const & vpc,
    std::vector<std::vector<Color>> const & pcs,
    std::vector<std::size_t> & nis,
    destination_intervals & intervals,
    source_pointers & pointers,
    // field<cmap, data::ragged>::mutator<wo> cgraph,
    data::multi<field<util::id>::accessor1<
      privilege_cat<privilege_repeat<wo, N - (N > 1)>,
        privilege_repeat<na, (N > 1)>>>> fmap,
    std::vector<std::map<std::size_t, std::size_t>> & rmaps,
    MPI_Comm const & comm) {
    flog_assert(vpc.size() == fmap.depth(),
      vpc.size() << " colorings for " << fmap.depth() << " colors");

    // FIXME: This is a place holder for Navamita
    //(void)cgraph;

    auto [rank, size] = util::mpi::info(comm);

    std::vector<std::map<std::size_t, std::size_t>> shared_offsets(vpc.size()),
      ghost_offsets(vpc.size());

    std::vector</* over processes */
      std::vector</* over local colors */
        std::vector<std::tuple<std::size_t /* id */,
          std::size_t /* local color */,
          std::size_t /* global color */>>>>
      sources(size);

    pointers.resize(vpc.size());
    rmaps.resize(vpc.size());
    std::size_t lco{0};
    auto vi = vpc.begin();
    for(auto & fa : fmap.accessors()) {
      auto & pc = *vi++;
      std::vector<std::size_t> entities;
      auto const & ic = pc.coloring;

      /*
        Define the entity ordering from coloring. This version uses the
        mesh ordering, i.e., the entities are sorted by ascending mesh id.
       */

      for(auto const & e : ic.owned) {
        entities.push_back(e);
      } // for

      for(auto const & e : ic.ghost) {
        entities.push_back(e.id);

        if(sources[e.process].size() == 0) {
          sources[e.process].resize(vpc.size());
        } // if

        auto const s = std::make_tuple(e.id, e.local, e.global);
        sources[e.process][lco].emplace_back(s);
      } // for

      /*
        This call is what actually establishes the entity ordering by
        sorting the mesh entity ids.
       */

      util::force_unique(entities);

      /*
        Initialize the forward and reverse maps.
       */

      flog_assert(entities.size() == (ic.owned.size() + ic.ghost.size()),
        "entities size(" << entities.size() << ") doesn't match sum of owned("
                         << ic.owned.size() << ") and ghost(" << ic.ghost.size()
                         << ")");

      std::size_t off{0};
      auto & rmap = rmaps[lco];
      rmap.clear();
      for(auto e : entities) {
        fa[off] = e;
        rmap.try_emplace(e, off++);
      } // for

      /*
        After the entity order has been established, we need to create a
        lookup table for local ghost offsets.
       */

      for(auto e : ic.ghost) {
        auto it = std::find(entities.begin(), entities.end(), e.id);
        flog_assert(it != entities.end(), "ghost entity doesn't exist");
        ghost_offsets[lco][e.id] = std::distance(entities.begin(), it);
      } // for

      /*
        We also need to add shared offsets to the lookup table so that we can
        provide local shared offset information to other processes that
        request it.
       */

      for(auto & e : ic.shared) {
        auto it = std::find(entities.begin(), entities.end(), e.id);
        flog_assert(it != entities.end(), "shared entity doesn't exist");
        shared_offsets[lco][e.id] = std::distance(entities.begin(), it);
      } // for

      ++lco;
    } // for

    /*
      Send/Receive requests for shared offsets with other processes.
     */

    auto requested = util::mpi::all_to_allv(
      [&sources](int r, int) -> auto & { return sources[r]; }, comm);

    /*
      Fulfill the requests that we received from other processes, i.e.,
      provide the locaL offset for the requested shared mesh ids.
     */

    std::vector<std::vector<std::size_t>> fulfills(size);
    {
      int r = 0;
      for(const auto & rv : requested) {
        for(auto const pc : rv) {
          for(auto [id, lc, dmmy] : pc) {
            fulfills[r].emplace_back(shared_offsets[lc][id]);
          } // for
        } // for

        ++r;
      } // for
    } // scope

    /*
      Send/Receive the local offset information with other processes.
     */

    auto fulfilled = util::mpi::all_to_allv(
      [f = std::move(fulfills)](int r, int) { return std::move(f[r]); }, comm);

    /*
      Setup source pointers.
     */

    int r{0};
    for(auto const & rv : sources) {
      (void)rv;
      std::size_t pc{0};
      std::size_t cnt{0};
      for(auto const & cv : sources[r]) {
        auto & pts = pointers[pc];

        for(auto [id, dmmy, gco] : cv) {
          pts[gco].emplace_back(
            std::make_pair(ghost_offsets[pc][id], fulfilled[r][cnt++]));
        } // for

        ++pc;
      } // for

      ++r;
    } // for

    /*
      Compute local intervals.
     */

    std::vector<std::size_t> local_itvls(vpc.size());
    intervals.resize(vpc.size());
    for(std::size_t lc{0}; lc < vpc.size(); ++lc) {
      auto g = ghost_offsets[lc].begin();
      std::size_t begin = g == ghost_offsets[lc].end() ? 0 : g->second, run = 0;
      for(; g != ghost_offsets[lc].end(); ++g) {
        if(!run || g->second != begin + run) {
          if(run) {
            intervals[lc].emplace_back(std::make_pair(begin, begin + run));
            begin = g->second;
          }
          run = 1;
        }
        else {
          ++run;
        }
      } // for

      intervals[lc].emplace_back(std::make_pair(begin, begin + run));
      local_itvls[lc] = intervals[lc].size();
    } // for

    /*
      Gather global interval sizes.
     */

    auto global_itvls = util::mpi::all_gatherv(local_itvls, comm);

    std::size_t p{0};
    for(auto const & pv : pcs) {
      std::size_t c{0};
      for(auto const & pc : pv) {
        nis[pc] = global_itvls[p][c++];
      } // for

      ++p;
    } // for
  } // idx_itvls

  static void set_dests(
    data::multi<field<data::intervals::Value>::accessor<wo>> aa,
    std::vector<std::vector<data::subrow>> const & intervals,
    MPI_Comm const &) {
    std::size_t ci = 0;
    for(auto [c, a] : aa.components()) {
      auto & iv = intervals[ci++];
      flog_assert(a.span().size() == iv.size(),
        "interval size mismatch a.span ("
          << a.span().size() << ") != intervals (" << iv.size() << ")");
      std::size_t i{0};
      for(auto & it : iv) {
        a[i++] = data::intervals::make(it, c);
      } // for
    } // for
  }

  template<PrivilegeCount N>
  static void set_ptrs(
    data::multi<field<data::points::Value>::accessor1<privilege_repeat<wo, N>>>
      aa,
    std::vector<std::map<Color,
      std::vector<std::pair<std::size_t, std::size_t>>>> const & points,
    MPI_Comm const &) {
    std::size_t ci = 0;
    for(auto & a : aa.accessors()) {
      for(auto const & si : points[ci++]) {
        for(auto p : si.second) {
          // si.first: owner
          // p.first: local ghost offset
          // p.second: remote shared offset
          a[p.first] = data::points::make(si.first, p.second);
        } // for
      } // for
    } // for
  }

  template<std::size_t S>
  static void idx_subspaces(index_coloring const & ic,
    field<util::id, data::ragged>::mutator<rw> owned,
    field<util::id, data::ragged>::mutator<rw> exclusive,
    field<util::id, data::ragged>::mutator<rw> shared,
    field<util::id, data::ragged>::mutator<rw> ghost) {
    const auto cp = [](auto r, const std::vector<util::id> & v) {
      r.assign(v.begin(), v.end());
    };

    cp(owned[S], ic.owned);
    cp(exclusive[S], ic.exclusive);
    cp(shared[S], ic.shared);
    cp(ghost[S], ic.ghost);
  }

  static void cnx_size(std::vector<process_coloring> const & vpc,
    std::size_t is,
    data::multi<resize::Field::accessor<wo>> aa) {
    auto it = vpc.begin();
    for(auto & a : aa.accessors()) {
      a = it++->cnx_allocs[is];
    }
  }

  static void copy_sizes(resize::Field::accessor<ro> src,
    resize::Field::accessor<wo> dest) {
    dest = src.get();
  }
}; // struct unstructured_base

inline std::ostream &
operator<<(std::ostream & stream,
  typename unstructured_base::coloring const & c) {
  stream << "colors: " << c.colors << std::endl;
  stream << "process_colors\n" << log::container{c.process_colors} << std::endl;
  stream << "color_peers\n" << log::container{c.color_peers} << std::endl;
  stream << "peers\n" << log::container{c.peers} << std::endl;
  stream << "partitions\n" << log::container{c.partitions} << std::endl;
  stream << "idx_spaces\n" << log::container{c.idx_spaces} << std::endl;
  return stream;
}
/// \}
} // namespace topo

/*----------------------------------------------------------------------------*
  Serialization Rules
 *----------------------------------------------------------------------------*/

template<>
struct util::serial::traits<topo::unstructured_impl::shared_entity> {
  using type = topo::unstructured_impl::shared_entity;
  template<class P>
  static void put(P & p, const type & s) {
    serial::put(p, s.id, s.dependents);
  }
  static type get(const std::byte *& p) {
    const cast r{p};
    return type{r, r};
  }
};

template<>
struct util::serial::traits<topo::unstructured_impl::index_coloring> {
  using type = topo::unstructured_impl::index_coloring;
  template<class P>
  static void put(P & p, const type & c) {
    serial::put(p, c.all, c.owned, c.exclusive, c.shared, c.ghost);
  }
  static type get(const std::byte *& p) {
    const cast r{p};
    return type{r, r, r, r, r};
  }
};

template<>
struct util::serial::traits<topo::unstructured_impl::process_coloring> {
  using type = topo::unstructured_impl::process_coloring;
  template<class P>
  static void put(P & p, const type & c) {
    serial::put(p,
      c.color,
      c.entities,
      c.coloring,
      c.peers,
      c.cnx_allocs,
      c.cnx_colorings);
  }
  static type get(const std::byte *& p) {
    const cast r{p};
    return type{r, r, r, r, r, r};
  }
};

} // namespace flecsi

#endif
