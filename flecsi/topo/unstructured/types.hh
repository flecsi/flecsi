// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_UNSTRUCTURED_TYPES_HH
#define FLECSI_TOPO_UNSTRUCTURED_TYPES_HH

#include "flecsi/data/copy_plan.hh"
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
#include <tuple>
#include <vector>

namespace flecsi {
namespace topo {

/// \addtogroup unstructured
/// \{
namespace unstructured_impl {

using entity_kind = std::size_t;
using entity_index_space = std::size_t;

/// Information about an entity that is shared with other colors.
struct shared_entity {
  /// Global id.
  util::gid id;
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
  util::gid id;
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
  std::vector<util::gid> all;
  /// Global ids owned by this color.
  std::vector<util::gid> owned;
  /// The subset of \c owned that are not ghosts on any other color.
  std::vector<util::gid> exclusive;
  /// Entities that are ghosts on another color.
  std::vector<shared_entity> shared;
  /// Entities that are owned by another color.
  std::vector<ghost_entity> ghost;
};

inline std::ostream &
operator<<(std::ostream & stream, index_coloring const & ic) {
  stream << "all\n" << flog::container{ic.all} << std::endl;
  stream << "owned\n" << flog::container{ic.owned} << std::endl;
  stream << "exclusive\n" << flog::container{ic.exclusive} << std::endl;
  stream << "shared\n" << flog::container{ic.shared} << std::endl;
  stream << "ghost\n" << flog::container{ic.ghost} << std::endl;
  return stream;
}

/*!
  Initialize a connectivity using its transpose connectivity, e.g.,
  initializing vertex-to-cell connectivity using cell-to-vertex.

  @tparam NI Number of privileges for input connectivity.
  @tparam NO Number of privileges for output connectivity.
  @param input Conectivity used to compute the tranpose.
  @param output Connectivity to initialize from transposition.
  */
template<PrivilegeCount NI, PrivilegeCount NO>
void
transpose(
  field<util::id, data::ragged>::accessor1<privilege_repeat<ro, NI>> input,
  field<util::id, data::ragged>::mutator1<privilege_repeat<wo, NO>> output) {
  std::size_t e = 0;
  for(auto && i : input) {
    for(auto v : i)
      output[v].push_back(e);
    ++e;
  }
}

/// Strategy for contructing colorings.
struct coloring_definition {
  /// Instances of this type are used to map from a mesh definition
  /// to an index space. In many cases, it is not possible to simply
  /// use the topological dimension of the entity type, e.g., edges will
  /// collide with corners. This problem is solved by using explicit
  /// identifiers for each entity kind in the mesh definition and
  /// associating it with the entity index space defined by the
  /// specialization.
  struct index_map {
    /// mesh definition entity kind
    entity_kind kind;
    /// entity index space id
    entity_index_space idx;
  };

  /// Total number of colors.
  Color colors;
  /// Index of primary entity in \c index_spaces.
  /// \warning Not an \c index_space enumerator \b value.
  index_map cid;
  /// Number of layers of ghosts needed.
  std::size_t depth;
  /// Index of vertices in \c index_spaces.
  index_map vid;
  /// Indices of auxiliary entities in \c index_spaces.
  std::vector<index_map> aidxs;
};

/// Information specific to a local color.
/// \ingroup unstructured
struct process_coloring {

  /// The global color that this object defines.
  Color color;

  /// The global number of entities in this index space.
  util::gid entities;

  /// The local coloring information for this index space.
  /// The coloring information is expressed in the mesh index space,
  /// i.e., the ids are global.
  index_coloring coloring;

  /// Communication peers (needed for ragged/sparse buffer creation).
  std::vector<Color> peers;

  /// The local allocation size for each connectivity. The vector is over
  /// connectivities between this entity type and another entity type. The
  /// ordering follows that given in the specialization policy.
  std::vector<std::size_t> cnx_allocs;

  /// The local graph for each connectivity.
  /// The graph information is expressed in the mesh index space,
  /// i.e., the ids are global. The vector is like cnx_alloc.
  std::vector<util::crs> cnx_colorings;
}; // struct process_coloring

inline std::ostream &
operator<<(std::ostream & stream, process_coloring const & pc) {
  stream << "color: " << pc.color << std::endl;
  stream << "entities: " << pc.entities << std::endl;
  stream << "coloring:\n" << pc.coloring << std::endl;
  stream << "peers:\n" << flog::container{pc.peers} << std::endl;
  stream << "cnx_allocs:\n" << flog::container{pc.cnx_allocs} << std::endl;
  stream << "cnx_colorings:\n"
         << flog::container{pc.cnx_colorings} << std::endl;
  return stream;
}

/*
  Type for mapping ragged ghosts into buffers.
 */

struct cmap {
  std::size_t lid;
  std::size_t rid;
};

} // namespace unstructured_impl

struct unstructured_base {

  using index_coloring = unstructured_impl::index_coloring;
  using process_coloring = unstructured_impl::process_coloring;
  using ghost_entity = unstructured_impl::ghost_entity;
  using cmap = unstructured_impl::cmap;
  using reverse_maps_t = std::vector<std::map<util::gid, util::id>>;

  using source_pointers = std::vector</* over local colors */
    std::map</* over global source colors */
      Color,
      std::vector</* over color source pointers */
        std::pair<util::id /* local offset */, util::id /* remote offset */>>>>;

  using destination_intervals = std::vector</* over local colors */
    std::vector</* over contiguous intervals */
      data::subrow>>;

  /// The coloring data strcuture is how information is passed to the FleCSI
  /// runtime to construct one or more unstructured mesh specialization types.
  /// The coloring object is returned by the specailization's `color` method.
  /// \ingroup unstructured
  struct coloring {
    /// An MPI communicator that can be used to specify subsets of COMM_WORLD
    /// ranks on which the coloring should be computed. This variable is used
    /// internaly in several of the FleCSI coloring utilities methods.
    MPI_Comm comm;

    /// The global number of colors, i.e., the number of partitions into which
    /// this coloring instance will divide the input mesh.
    Color colors;

    /// The local colors that belong to a given process. This varaible stores
    /// the local color information over all global processes.
    std::vector</* over global processes */
      std::vector</* over local process colors */
        Color>>
      process_colors;

    /// The superset of communication peers over the global colors, i.e., for
    /// each color, this stores the number of communication peers over all
    /// index spaces.
    std::vector</* over global colors */
      std::size_t>
      color_peers;

    /// The communication peers over each index space over all colors, i.e.,
    /// for each index space and for each color, the communication peers
    /// (color ids) are stored.
    std::vector</* over index spaces */
      std::vector</* over global colors */
        std::vector</* over peers */
          Color>>>
      peers;

    /// The partition sizes over each index space and over all colors.
    std::vector</* over index spaces */
      std::vector</* over global colors */
        std::size_t>>
      partitions;

    /// The index space coloring over each index space and local color.
    std::vector</* over index spaces */
      std::vector</* over process colors */
        process_coloring>>
      idx_spaces;
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

  // privilege_cat<privilege_repeat<wo, N - (N > 1)>,
  //   privilege_repeat<na, (N > 1)>>>> fmap,
  template<PrivilegeCount N>
  static void idx_itvls(std::vector<process_coloring> const & vpc,
    std::vector<std::vector<Color>> const & pcs,
    std::vector<std::size_t> & nis,
    destination_intervals & intervals,
    source_pointers & pointers,
    data::multi<field<cmap, data::ragged>::mutator<wo>> cgraph,
    data::multi<field<cmap, data::ragged>::mutator<wo>> cgraph_shared,
    data::multi<field<util::gid>::accessor1<privilege_ghost_repeat<wo, na, N>>>
      fmap,
    reverse_maps_t & rmaps,
    MPI_Comm const & comm) {
    flog_assert(vpc.size() == fmap.depth(),
      vpc.size() << " colorings for " << fmap.depth() << " colors");

    auto [rank, size] = util::mpi::info(comm);

    std::vector<std::map<util::gid, util::id>> shared_offsets(vpc.size()),
      ghost_offsets(vpc.size());

    std::vector</* over processes */
      std::vector</* over local colors */
        std::vector<std::tuple<util::gid,
          std::size_t /* local color */,
          std::size_t /* global color */>>>>
      sources(size);

    std::vector<std::vector<std::size_t>> sources_lco(size);

    std::vector</* over local colors */
      std::map<std::size_t /* shared color */,
        std::vector<std::size_t /* local shared offsets */>>>
      shared_cg(vpc.size());

    pointers.resize(vpc.size());
    rmaps.resize(vpc.size());
    std::size_t lco{0};
    auto vi = vpc.begin();
    for(auto & fa : fmap.accessors()) {
      auto & pc = *vi++;
      std::vector<util::gid> entities;
      auto const & ic = pc.coloring;

      /*
        Define the entity ordering from the coloring. The ordering is
        defined by "all".
       */

      for(auto const & e : ic.all) {
        entities.push_back(e);
      } // for

      /*
        Add ghosts to sources.
       */

      for(auto const & e : ic.ghost) {
        if(sources[e.process].size() == 0) {
          sources[e.process].resize(vpc.size());
        } // if

        auto const s = std::make_tuple(e.id, e.local, e.global);
        sources[e.process][lco].emplace_back(s);
        sources_lco[e.process].emplace_back(lco);
      } // for

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
        Create a lookup table for local ghost offsets.
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
        auto offset = std::distance(entities.begin(), it);
        shared_offsets[lco][e.id] = offset;
        // shared_offsets[lco][e.id] = std::distance(entities.begin(), it);

        // fill out the map of dependent color to local offset
        for(auto & dep : e.dependents) {
          shared_cg[lco][dep].push_back(offset);
        }
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
      provide the local offset for the requested shared mesh ids.
     */

    std::vector<std::vector<std::size_t>> fulfills(size);
    {
      int r = 0;
      for(const auto & rv : requested) {
        for(const auto & pc : rv) {
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

    /*
      Setup cgraph data
     */

    using ghost_info = std::tuple<util::gid, /*ghost id*/
      std::size_t, /*local color*/
      std::size_t, /*local offset*/
      std::size_t /*remote offset*/
      >;

    std::vector</*local colors*/
      std::map<std::size_t, /*global color*/
        std::vector<ghost_info>>>
      peer_ghosts;

    // lambda
    auto find_in_source = [&sources, &sources_lco, &ghost_offsets, &fulfilled](
                            util::gid const & in_id,
                            std::size_t const & in_lco,
                            std::size_t & gcolor,
                            ghost_info & ginfo) {
      int k{0};
      for(auto const & rv : sources) { // process
        std::size_t pc{0}, cnt{0};
        for(auto const & cv : rv) { // local colors
          for(auto [id, dmmy, gco] : cv) {
            if((in_id == id) && (in_lco == sources_lco[k][cnt])) {
              gcolor = gco;
              ginfo = std::make_tuple(
                in_id, in_lco, ghost_offsets[pc][id], fulfilled[k][cnt]);
              return true;
            }
            ++cnt;
          } // for
          ++pc;
        } // for
        ++k;
      } // for
      return false;
    };

    for(std::size_t i = 0; i < vpc.size(); ++i) {
      std::map<std::size_t, std::vector<ghost_info>> temp;
      for(auto const & g : ghost_offsets[i]) {
        std::size_t gcolor;
        ghost_info ginfo;
        if(find_in_source(g.first, i, gcolor, ginfo)) {
          temp[gcolor].emplace_back(ginfo);
        }
      }

      // sort the vectors for each color by the remote id of the ghosts
      for(auto & c : temp) {
        std::sort(std::begin(c.second),
          std::end(c.second),
          [](const ghost_info & t1, const ghost_info & t2) {
            return std::get<3>(t1) < std::get<3>(t2);
          });
      }

      peer_ghosts.push_back(temp);
    }

    int k = 0;
    for(auto & a : cgraph.accessors()) {
      // peers for local color k
      std::size_t p{0};
      for(auto const & pg : peer_ghosts[k]) {

        // resize field
        a[p].resize(pg.second.size());

        // loop over entries
        std::size_t cnt{0};
        for(auto [id, dmmy, lid, rid] : pg.second) {
          a[p][cnt].lid = lid;
          a[p][cnt].rid = rid;
          ++cnt;
        }
        ++p;
      }
      ++k;
    }

    // fill out cgraph_shared
    k = 0;
    for(auto & a : cgraph_shared.accessors()) {
      // peers for local color k
      std::size_t p{0};
      for(auto const & sh : shared_cg[k]) {

        // resize field
        a[p].resize(sh.second.size());

        // loop over entries
        std::size_t cnt{0};
        for(auto offset : sh.second) {
          a[p][cnt].lid = offset;
          a[p][cnt].rid = offset;
          ++cnt;
        }
        ++p;
      }
      ++k;
    }

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
      std::vector<std::pair<util::id, util::id>>>> const & points,
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

  // resize ragged fields storing communication graph for ghosts
  static void cgraph_size(std::vector<process_coloring> const & vpc,
    data::multi<ragged_partition<1>::accessor<wo>> aa) {
    auto it = vpc.begin();
    for(auto & a : aa.accessors()) {
      auto & ic = it->coloring;
      a.size() = ic.ghost.size();
      ++it;
    }
  } // cgraph_size

  // resize ragged fields storing communication graph for shared
  static void cgraph_shared_size(std::vector<process_coloring> const & vpc,
    data::multi<ragged_partition<1>::accessor<wo>> aa) {
    auto it = vpc.begin();

    for(auto & a : aa.accessors()) {
      std::size_t count = 0;
      for(auto & e : it++->coloring.shared) {
        count += e.dependents.size();
      }
      a.size() = count;
    }

  } // cgraph_shared_size

  template<typename T, PrivilegeCount N>
  struct ragged_impl {

    using fa = typename field<T,
      data::ragged>::template accessor1<privilege_ghost_repeat<ro, na, N>>;

    using fm_rw = typename field<T,
      data::ragged>::template mutator1<privilege_ghost_repeat<ro, rw, N>>;

    using cga = field<cmap, data::ragged>::accessor<ro>;

    static void start(fa v, cga, cga cgraph_shared, data::buffers::Start mv) {
      send(v, cgraph_shared, true, mv);
    } // start

    static int
    xfer(fm_rw g, cga cgraph, cga cgraph_shared, data::buffers::Transfer mv) {
      int p = cgraph_shared.size(); // number of send buffers
      for(auto pg : cgraph) { // over peers
        data::buffers::ragged::read(g, mv[p], [&pg](std::size_t i) {
          return std::partition_point(
            pg.begin(), pg.end(), [i](const cmap & c) { return (i > c.rid); })
            ->lid;
        });
        ++p;
      }

      // resume transfer if data was not fully packed during start
      return send(g, cgraph_shared, false, mv);
    } // xfer

  private:
    template<typename F, typename B>
    static bool send(F f, cga cgraph_shared, bool first, B mv) {
      int p = 0;
      bool sent = false;
      for(auto pg : cgraph_shared) { // over peers
        auto b = data::buffers::ragged{mv[p++], first};
        for(auto & ent : pg) { // send data on shared entities
          if(!b(f, ent.lid, sent))
            return sent; // if no more data can be packed, stop sending, will be
                         // packed by xfer
        }
      }
      return sent;
    } // send

  }; // struct ragged_impl

}; // struct unstructured_base

inline std::ostream &
operator<<(std::ostream & stream,
  typename unstructured_base::coloring const & c) {
  stream << "colors: " << c.colors << std::endl;
  stream << "process_colors\n"
         << flog::container{c.process_colors} << std::endl;
  stream << "color_peers\n" << flog::container{c.color_peers} << std::endl;
  stream << "peers\n" << flog::container{c.peers} << std::endl;
  stream << "partitions\n" << flog::container{c.partitions} << std::endl;
  stream << "idx_spaces\n" << flog::container{c.idx_spaces} << std::endl;
  return stream;
}

#if 1
namespace unstructured_impl {
/*!
  Initialize a connectivity using the coloring. This method uses
  from-to nomenclature, e.g., 'from' cells to 'vertices' initializes the
  cell index space connectivity to the vertices index space.

  @tparam NF    Number of privileges for connectivity field.
  @param  mconn A multi-accessor to the connectivity field.
  @param  c     The coloring.
  @param  map   The global-to-local id map for the to entities.
 */
template<PrivilegeCount NF>
void
init_connectivity(entity_index_space from,
  entity_index_space to,
  data::multi<field<util::id, data::ragged>::mutator1<privilege_repeat<wo, NF>>>
    mconn,
  unstructured_base::coloring const & c,
  unstructured_base::reverse_maps_t const & maps) {

  auto pcs = c.idx_spaces[from].begin();
  auto mp = maps.begin();
  for(auto & x2y : mconn.accessors()) {
    auto const & pc = *pcs++;
    auto const & vm = *mp++;
    util::id off{0};

    auto const & cnx = pc.cnx_colorings[to];
    for(const util::crs::span r : cnx) {
      auto v = util::transform_view(r, [&vm](util::gid i) { return vm.at(i); });
      x2y[off++].assign(v.begin(), v.end());
    } // for
  } // for
}
} // namespace unstructured_impl
#endif

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
