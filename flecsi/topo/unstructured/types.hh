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

using entity_index_space = std::size_t;

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

/// Communication pattern for a single pair of colors.
struct peer_entities {
  /// Local entities sent to the peer.
  std::set<util::id> shared;
  /// Local ID for (the remote ID) each entity received from the peer.
  std::map<util::id, util::id> ghost;
};

/// Information specific to a single index space and color.
/// \ingroup unstructured
struct index_color {
  /// Total number of entities stored by this color, including ghosts.
  util::id entities = 0;

  /// Entities sent to and received from other colors.
  std::map<Color, peer_entities> peers;

  /// The local allocation size for each connectivity. The vector is over
  /// all connectivities in the order given in the specialization policy.
  std::vector<std::size_t> cnx_allocs;

  /// \cond core

  /// Entities received from other colors
  auto ghosts() const {
    std::set<util::id> ghst;
    for(auto & p : peers) {
      for(auto [rid, lid] : p.second.ghost)
        ghst.insert(lid);
    }
    return ghst;
  }

  /// Entities owned by this color
  auto owned() const {
    std::vector<util::id> ownd;
    std::set<util::id> ghst = ghosts();

    for(util::id e = 0; e < entities; ++e) {
      if(!ghst.count(e)) {
        ownd.push_back(e);
      }
    }
    return ownd;
  }

  /// Entities that are ghosts on another color.
  auto shared() const {
    std::set<util::id> shr;
    for(auto & p : peers) {
      shr.insert(p.second.shared.begin(), p.second.shared.end());
    }
    return shr;
  }

  /// The subset of \c owned that are not ghosts on any other color.
  auto exclusive() const {
    const auto ss = shared();
    std::vector<util::id> ex;
    for(auto o : owned())
      if(!ss.count(o))
        ex.push_back(o);
    return ex;
  }

  auto ghost_intervals() const {
    std::vector<data::subrow> intervals;
    auto gs = ghosts();
    auto g = gs.begin();
    std::size_t begin = g == gs.end() ? 0 : *g, run = 0;
    for(; g != gs.end(); ++g) {
      if(!run || *g != begin + run) {
        if(run) {
          intervals.emplace_back(std::make_pair(begin, begin + run));
          begin = *g;
        }
        run = 1;
      }
      else {
        ++run;
      }
    } // for

    intervals.emplace_back(std::make_pair(begin, begin + run));
    return intervals;
  }

  /// \endcond
}; // struct index_color

inline std::ostream &
operator<<(std::ostream & stream, index_color const & ic) {
  stream << "owned\n" << flog::container{ic.owned()} << "\n";
  stream << "exclusive\n" << flog::container{ic.exclusive()} << "\n";
  stream << "shared\n" << flog::container{ic.shared()} << "\n";
  stream << "ghosts\n" << flog::container{ic.ghosts()} << "\n";
  stream << "cnx_allocs:\n" << flog::container{ic.cnx_allocs} << "\n";
  return stream;
}

} // namespace unstructured_impl

struct unstructured_base {

  using index_color = unstructured_impl::index_color;

  using source_pointers = std::vector</* over local colors */
    std::map</* over global source colors */
      Color,
      std::vector</* over color source pointers */
        std::pair<util::id /* local offset */, util::id /* remote offset */>>>>;

  using destination_intervals = std::vector</* over local colors */
    std::vector</* over contiguous intervals */
      data::subrow>>;

  /// The coloring data structure is how information is passed to the FleCSI
  /// runtime to construct one or more unstructured mesh specialization types.
  /// The coloring object is returned by the specialization's `color` method.
  /// \ingroup unstructured
  struct coloring {
    struct index_space {
      /// The communication peers over all colors, i.e.,
      /// for each color, the communication peers
      /// (color ids) are stored.
      std::vector</* over global colors */
        std::vector</* over peers */
          Color>>
        peers;

      /// The partition sizes over all colors.
      std::vector</* over global colors */
        std::size_t>
        partitions;

      /// The global number of entities in this index space.
      util::gid entities;

      /// Information specific to local colors.
      std::vector</* over process colors */
        index_color>
        colors;

      // number of ghost intervals over all colors
      std::vector<std::size_t> num_intervals;
    };

    /// The global number of colors, i.e., the number of partitions into which
    /// this coloring instance will divide the input mesh.
    Color colors;

    std::vector<index_space> idx_spaces;

    /// The superset of communication peers over the global colors, i.e., for
    /// each color, this stores the maximum number of sending or receiving
    /// peers over all index spaces.
    std::vector</* over global colors */
      std::size_t>
      color_peers;

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

  static void idx_itvls(std::vector<index_color> const & vic,
    destination_intervals & intervals,
    source_pointers & pointers,
    data::multi<field<util::id, data::ragged>::mutator<wo>> cgraph,
    data::multi<field<util::id, data::ragged>::mutator<wo>> cgraph_shared) {

    const auto prep = [n = vic.size()](auto & v) {
      v.clear();
      v.reserve(n);
    };
    prep(intervals);
    prep(pointers);
    const auto ca = cgraph.accessors(), sa = cgraph_shared.accessors();
    auto ci = ca.begin(), si = sa.begin();
    for(auto & ic : vic) {
      intervals.push_back(ic.ghost_intervals());
      auto & pts = pointers.emplace_back();

      auto ci1 = (*ci++).begin(), si1 = (*si++).begin();
      for(auto const & [global, pe] : ic.peers) {
        auto c = *ci1++, s = *si1++;
        auto & p = pts[global];
        c.reserve(pe.ghost.size());
        p.reserve(pe.ghost.size());
        for(auto [rid, lid] : pe.ghost) {
          p.emplace_back(lid, rid);
          c.push_back(lid);
        }
        s.assign(pe.shared.begin(), pe.shared.end());
      } // for
    } // for
  } // idx_itvls

  static void set_dests(
    data::multi<field<data::intervals::Value>::accessor<wo>> aa,
    std::vector<std::vector<data::subrow>> const & intervals) {
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
      std::vector<std::pair<util::id, util::id>>>> const & points) {
    std::size_t ci = 0;
    for(auto & a : aa.accessors()) {
      for(auto const & [owner, ghosts] : points[ci++]) {
        for(auto const & [local_offset, remote_offset] : ghosts) {
          a[local_offset] = data::points::make(owner, remote_offset);
        } // for
      } // for
    } // for
  }

  static void cnx_size(std::vector<index_color> const & vic,
    std::size_t is,
    data::multi<resize::Field::accessor<wo>> aa) {
    auto it = vic.begin();
    for(auto & a : aa.accessors()) {
      a = it++->cnx_allocs[is];
    }
  }

  static void copy_sizes(resize::Field::accessor<ro> src,
    resize::Field::accessor<wo> dest) {
    dest = src.get();
  }

  // resize ragged fields storing communication graph for ghosts
  static void cgraph_size(std::vector<index_color> const & vic,
    data::multi<ragged_partition<1>::accessor<wo>> aa) {
    auto it = vic.begin();
    for(auto & a : aa.accessors()) {
      a.size() = it->ghosts().size();
      ++it;
    }
  } // cgraph_size

  // resize ragged fields storing communication graph for shared
  static void cgraph_shared_size(std::vector<index_color> const & vic,
    data::multi<ragged_partition<1>::accessor<wo>> aa) {
    auto it = vic.begin();

    for(auto & a : aa.accessors()) {
      std::size_t count = 0;
      for(auto & p : it++->peers) {
        count += p.second.shared.size();
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

    using cga = field<util::id, data::ragged>::accessor<ro>;

    static void start(fa v, cga, cga cgraph_shared, data::buffers::Start mv) {
      send(v, cgraph_shared, true, mv);
    } // start

    static int
    xfer(fm_rw g, cga cgraph, cga cgraph_shared, data::buffers::Transfer mv) {
      // find the number of send buffers
      int p = 0;
      for(auto ps : cgraph_shared) { // over peers
        if(!ps.empty())
          ++p;
      }
      for(auto pg : cgraph) { // over peers
        if(!pg.empty())
          data::buffers::ragged::read(g, mv[p++], pg);
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
        if(!pg.empty()) {
          auto b = data::buffers::ragged{mv[p++], first};
          for(auto & ent : pg) { // send data on shared entities
            if(!b(f, ent, sent))
              return sent; // if no more data can be packed, stop sending, will
                           // be packed by xfer
          }
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
  stream << "color_peers\n" << flog::container{c.color_peers} << std::endl;
  stream << "idx_spaces\n" << flog::container{c.idx_spaces} << std::endl;
  return stream;
}

inline std::ostream &
operator<<(std::ostream & stream,
  typename unstructured_base::coloring::index_space const & idx) {
  stream << "peers: " << flog::container(idx.peers) << std::endl;
  stream << "partitions\n" << flog::container{idx.partitions} << std::endl;
  stream << "colors\n" << flog::container{idx.colors} << std::endl;
  return stream;
}

namespace unstructured_impl {
/// \cond core
/*!
  Initialize a connectivity using the coloring. This method uses
  from-to nomenclature, e.g., 'from' cells to 'vertices' initializes the
  cell index space connectivity to the vertices index space.

  @tparam NF    Number of privileges for connectivity field.
  @param  mconn A multi-accessor to the connectivity field.
  @param  connectivities
 */
template<PrivilegeCount NF>
void
init_connectivity(
  data::multi<field<util::id, data::ragged>::mutator1<privilege_repeat<wo, NF>>>
    mconn,
  std::vector<util::crs> const & connectivities) {

  auto cnxs = connectivities.begin();
  for(auto & x2y : mconn.accessors()) {
    auto const & cnx = *cnxs++;
    util::id off{0};

    for(const auto r : cnx) {
      x2y[off++].assign(r.begin(), r.end());
    } // for
  } // for
}
/// \endcond
} // namespace unstructured_impl

/// \}
} // namespace topo

} // namespace flecsi

#endif
