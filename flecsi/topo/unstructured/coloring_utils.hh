// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_UNSTRUCTURED_COLORING_UTILS_HH
#define FLECSI_TOPO_UNSTRUCTURED_COLORING_UTILS_HH

#include "flecsi/flog.hh"
#include "flecsi/topo/unstructured/coloring_functors.hh"
#include "flecsi/topo/unstructured/types.hh"
#include "flecsi/util/color_map.hh"
#include "flecsi/util/common.hh"
#include "flecsi/util/dcrs.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/serialize.hh"
#include "flecsi/util/set_utils.hh"

#include <algorithm>
#include <iterator>
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

/// \cond core
namespace flecsi {
namespace topo {
namespace unstructured_impl {
/// \addtogroup unstructured
/// \{

#ifdef DOXYGEN
/// An example mesh definition that is not really implemented.
struct mesh_definition {
  /// Get the dimensionality of the mesh.
  static constexpr Dimension dimension();

  /// Get the global number of entities of a kind.
  std::size_t num_entities(Dimension) const;

  /// Get the entities connected to or associated with an entity.
  /// \param id of entity of dimension \a from
  /// \return ids of entities of dimension \a to
  std::vector<std::size_t>
  entities(Dimension from, Dimension to, std::size_t id) const;

  /// Return the vertex with the given id.
  point vertex(std::size_t) const;

  /// Vertex information type, perhaps spatial coordinates.
  using point = decltype(std::declval<mesh_definition>().vertex(0));
};
#endif

/*!
  Create a distributed graph representation of the highest-dimensional
  entity type in the given mesh definition.
  \tparam Definition like \c mesh_definition
  \param md off the \a comm root, only \c num_entities used
  \param through_dimension minimum dimensionality of common entity to
    consider their primary entities to be neighbors
  \return a \c tuple of
    - a na&iuml;vely distributed graph of owned "cells" (top-level entities)
    - a pair of the total numbers of cells and vertices
    - a \c vector of \c vector objects holding the vertices for each owned
      cell
    - a \c map of \c vector objects holding the cells for each owned vertex
    - a \c map of \c vector objects holding the neighbors for each owned cell
 */

template<typename Definition>
inline auto
make_dcrs(Definition const & md,
  Dimension through_dimension,
  MPI_Comm comm = MPI_COMM_WORLD) {
  auto [rank, size] = util::mpi::info(comm);

  std::size_t nc = md.num_entities(Definition::dimension());
  std::size_t nv = md.num_entities(0);

  util::color_map cm(size, size, nc);

  /*
    Get the initial cells for this rank. The cells will be read by
    the root process and sent to each initially "owning" rank using
    a naive distribution.
   */

  auto c2v = util::mpi::one_to_allv<pack_cells<Definition>>(
    {md, cm.distribution()}, comm);

  /*
    Create a map of vertex-to-cell connectivity information from
    the initial cell distribution.
   */

  // Populate local vertex connectivity information
  std::size_t offset{cm.distribution()[rank]};
  std::size_t indices{cm.indices(rank, 0)};
  std::map<std::size_t, std::vector<std::size_t>> v2c;

  std::size_t i{0};
  for(auto c : c2v) {
    for(auto v : c) {
      v2c[v].emplace_back(offset + i);
    } // for
    ++i;
  } // for

  // Request all referencers of our connected vertices
  util::color_map vm(size, size, nv);
  auto referencers = util::mpi::all_to_allv<vertex_referencers>(
    {v2c, vm.distribution(), rank}, comm);

  /*
    Update our local connectivity information. We now have all
    vertex-to-cell connectivity informaiton for the naive distribution
    of cells that we own.
   */

  i = 0;
  for(auto & r : referencers) {
    for(auto v : r) {
      for(auto c : v.second) {
        v2c[v.first].emplace_back(c);
      } // for
    } // for
    ++i;
  } // for

  // Remove duplicate referencers
  util::unique_each(v2c);

  std::vector<std::vector<std::size_t>> referencer_inverse(size);

  for(auto const & v : v2c) {
    for(auto c : v.second) {
      auto r = util::distribution_offset(cm.distribution(), c);
      if(r != rank) {
        referencer_inverse[r].emplace_back(v.first);
      } // if
    } // for
  } // for

  // Remove duplicate inverses
  util::unique_each(referencer_inverse);

  // Request vertex-to-cell connectivity for the cells that are
  // on other ranks in the naive cell distribution.
  auto connectivity = util::mpi::all_to_allv<cell_connectivity>(
    {referencer_inverse, v2c, vm.distribution(), rank}, comm);

  for(auto & r : connectivity) {
    for(auto & v : r) {
      for(auto c : v.second) {
        v2c[v.first].emplace_back(c);
      } // for
    } // for
  } // for

  // Remove duplicate referencers
  util::unique_each(v2c);

  std::map<std::size_t, std::vector<std::size_t>> c2c;
  std::size_t c{offset};
  for(auto & cd /* cell definition */ : c2v) {
    std::map<std::size_t, std::size_t> through;

    for(auto v : cd) {
      auto it = v2c.find(v);
      if(it != v2c.end()) {
        for(auto rc : v2c.at(v)) {
          if(rc != c)
            ++through[rc];
        } // for
      } // if
    } // for

    for(auto tc : through) {
      if(tc.second > through_dimension) {
        c2c[c].emplace_back(tc.first);
        c2c[tc.first].emplace_back(c);
      } // if
    } // for

    ++c;
  } // for

  // Remove duplicate connections
  util::unique_each(c2c);

  util::dcrs dcrs;
  dcrs.distribution = cm.distribution();

  dcrs.offsets.emplace_back(0);
  for(std::size_t c{0}; c < indices; ++c) {
    for(auto cr : c2c[offset + c]) {
      dcrs.indices.emplace_back(cr);
    } // for

    dcrs.offsets.emplace_back(dcrs.offsets[c] + c2c[offset + c].size());
  } // for

  return std::make_tuple(dcrs, std::make_pair(nc, nv), c2v, v2c, c2c);
} // make_dcrs

/// Redistribute ownership information.
/// \param naive graph from \c make_dcrs
/// \param index_colors owning color for each local entity
/// \return the owned ids for each color owned by this rank
inline std::vector<std::vector<std::size_t>>
distribute(util::dcrs const & naive,
  Color colors,
  std::vector<std::size_t> const & index_colors,
  MPI_Comm comm = MPI_COMM_WORLD) {
  auto [rank, size] = util::mpi::info(comm);

  auto color_primaries = util::mpi::all_to_allv<distribute_cells>(
    {naive, colors, index_colors, rank}, comm);

  util::color_map cm(size, colors, naive.distribution.back());
  const std::size_t offset = cm.color_offset(rank);
  std::vector<std::vector<std::size_t>> primaries(cm.colors(rank));

  for(auto cp : color_primaries) {
    for(auto c : cp) {
      primaries[std::get<0>(c) - offset].emplace_back(std::get<1>(c));
    } // for
  } // for

  return primaries;
} // distribute

/// Redistribute connectivity information.
/// \param naive graph from \c make_dcrs
/// \param index_colors owning color for each local primary entity
/// \param c2v "cell" (primary entity) to vertex connectivity
/// \param v2c vertex to cell connectivity
/// \param c2c cell to cell connectivity
/// \return a \c tuple of
///   - a \c map from each owned \c Color to a \c vector of global cell ids
///   - a \c vector of global ids for each local primary entity
///   - a \c map from global to local ids
inline auto
migrate(util::dcrs const & naive,
  Color colors,
  std::vector<std::size_t> const & index_colors,
  std::vector<std::vector<std::size_t>> & c2v,
  std::map<std::size_t, std::vector<std::size_t>> & v2c,
  std::map<std::size_t, std::vector<std::size_t>> & c2c,
  MPI_Comm comm = MPI_COMM_WORLD) {
  auto [rank, size] = util::mpi::info(comm);

  auto migrated = util::mpi::all_to_allv<migrate_cells>(
    {naive, colors, index_colors, c2v, v2c, c2c, rank}, comm);

  std::map<Color, std::vector<std::size_t>> primaries;
  std::vector<std::size_t> p2m; /* process to mesh map */
  std::map<std::size_t, std::size_t> m2p;

  for(auto const & r : migrated) {
    auto const & cell_pack = std::get<0>(r);
    for(auto const & c : cell_pack) { /* std::vector over cells */
      auto const & info = std::get<0>(c); /* std::array<color, mesh id> */
      c2v.emplace_back(std::get<1>(c)); /* cell definition (vertex mesh ids) */
      m2p[std::get<1>(info)] = c2v.size() - 1; /* offset map */
      p2m.emplace_back(std::get<1>(info)); /* cell mesh id */
      primaries[std::get<0>(info)].emplace_back(std::get<1>(info));
    } // for

    // vertex-to-cell connectivity
    auto v2c_pack = std::get<1>(r);
    for(auto const & v : v2c_pack) {
      v2c.try_emplace(v.first, v.second);
    } // for

    // cell-to-cell connectivity
    auto c2c_pack = std::get<2>(r);
    for(auto const & c : c2c_pack) {
      c2c.try_emplace(c.first, c.second);
    } // for
  } // for

  return std::make_tuple(primaries, p2m, m2p);
} // migrate

/// Communicate connectivity information.
/// \tparam Policy class with
///   - \c primary: a specialization of \c primary_independent
///   - \c auxiliary_colorings: number of additional \c index_coloring objects
/// \tparam Definition like \c mesh_definition
/// \param md only \c num_entities used
/// \param raw owning color for each local primary entity in na&iuml;ve
///   distribution
/// \param primaries global entity ids per owned color
/// \param e2v entity to vertex connectivity (augmented)
/// \param v2e vertex to entity connectivity (augmented)
/// \param e2e entity to entity connectivity (augmented)
/// \param m2p global &rarr; local id (augmented)
/// \param p2m global ids for each local entity (augmented)
/// \return a \c map from each owned \c Color to a \c coloring
template<typename Policy, typename Definition>
inline auto
closure(Definition const & md,
  Color colors,
  std::vector<std::size_t> const & raw,
  std::map<Color, std::vector<std::size_t>> const & primaries,
  std::vector<std::vector<std::size_t>> & e2v,
  std::map<std::size_t, std::vector<std::size_t>> & v2e,
  std::map<std::size_t, std::vector<std::size_t>> & e2e,
  std::map<std::size_t, std::size_t> & m2p,
  std::vector<std::size_t> & p2m,
  MPI_Comm comm = MPI_COMM_WORLD) {
  auto [rank, size] = util::mpi::info(comm);

  std::size_t ne = md.num_entities(Policy::primary::dimension);

  /*
    Save color info for all of our initial local entities. The variable
    'e2co' will get updated as we build out our dependencies.
   */

  std::unordered_map<std::size_t, Color> e2co;
  std::map<std::size_t, std::vector<std::size_t>> wkset;
  for(auto const & p : primaries) {
    auto & wk = wkset[p.first];
    wk.reserve(p.second.size());
    for(auto e : p.second) {
      e2co.try_emplace(e, p.first);
      wk.emplace_back(e);
    } // for
  } // for

  std::unordered_map<std::size_t, std::set<Color>> dependents, dependencies;
  std::unordered_map<Color, std::set<std::size_t>> shared, ghosts;

  constexpr std::size_t depth = Policy::primary::depth;
  for(std::size_t d{0}; d < depth + 1; ++d) {
    std::vector<std::size_t> layer;

    /*
      Create request layer, and add local information.
     */

    for(auto const & p : primaries) {
      auto & wk = wkset.at(p.first);
      for(auto e : wk) {
        for(auto v : e2v[m2p.at(e)]) {
          for(auto en : v2e.at(v)) {
            if(e2e.find(en) == e2e.end()) {
              // If we don't have the entity, we need to request it.
              layer.emplace_back(en);

              // Collect the dependent color for request.
              dependencies[en].insert(p.first);

              // If we're within the requested depth, this is also
              // a ghost entity.
              if(d < depth) {
                ghosts[p.first].insert(en);
              }
            }
            else if(d < depth && e2co.at(en) != p.first) {
              // This entity is on the local process, but not
              // owned by the current color.

              // Add this entity to the shared of the owning color.
              shared[e2co.at(en)].insert(en);

              // Add the current color as a dependent.
              dependents[en].insert(p.first);

              // This entity is a ghost for the current color.
              ghosts[p.first].insert(en);
            } // if
          } // for
        } // for
      } // for

      wk.clear();
    } // for

    util::force_unique(layer);

    /*
      Request entity owners from naive-owners.
     */

    std::vector<std::vector<std::size_t>> requests(size);
    util::color_map nm(size, size, ne);
    for(auto e : layer) {
      requests[nm.process(nm.index_color(e))].emplace_back(e);
    } // for

    std::vector<std::vector<std::pair<std::size_t, std::set<Color>>>> reqs(
      size);
    {
      auto requested = util::mpi::all_to_allv(
        [&requests](int r, int) -> auto & { return requests[r]; }, comm);

      /*
        Fulfill naive-owner requests with migrated owners.
       */

      std::vector<std::vector<std::size_t>> fulfills(size);
      {
        Color r = 0;
        util::color_map cm(size, colors, ne);
        for(auto rv : requested) {
          for(auto e : rv) {
            const std::size_t start = nm.index_offset(rank, 0);
            fulfills[r].emplace_back(cm.process(raw[e - start]));
          } // for
          ++r;
        } // for
      } // scope

      auto fulfilled = util::mpi::all_to_allv(
        [&fulfills](int r, int) -> auto & { return fulfills[r]; }, comm);

      /*
        Request entity information from migrated owners.
       */

      std::vector<std::size_t> offs(size, 0ul);
      for(auto e : layer) {
        auto p = nm.process(nm.index_color(e));
        reqs[fulfilled[p][offs[p]++]].emplace_back(
          std::make_pair(e, dependencies.at(e)));
      } // for
    } // scope

    auto requested = util::mpi::all_to_allv(
      [&reqs](int r, int) -> auto & { return reqs[r]; }, comm);

    /*
      Keep track of dependent colors for requested entities.
     */

    requests.clear();
    requests.resize(size);
    Color r = 0;
    for(auto rv : requested) {
      for(auto e : rv) {
        requests[r].emplace_back(e.first);
        if(d < depth) {
          dependents[e.first].insert(e.second.begin(), e.second.end());
          shared[e2co.at(e.first)].insert(e.first);
        }
      } // for
      ++r;
    } // for

    auto fulfilled = util::mpi::all_to_allv<communicate_entities>(
      {requests, e2co, e2v, v2e, e2e, m2p}, comm);

    /*
      Update local information.
     */

    for(auto const & r : fulfilled) {
      auto const & entity_pack = std::get<0>(r);
      for(auto const & e : entity_pack) {
        auto const & info = std::get<0>(e);
        e2v.emplace_back(std::get<1>(e));
        m2p[std::get<1>(info)] = e2v.size() - 1;
        p2m.emplace_back(std::get<1>(info));
        e2co.try_emplace(std::get<1>(info), std::get<0>(info));

        for(auto co : dependencies.at(std::get<1>(info))) {
          wkset.at(co).emplace_back(std::get<1>(info));
        } // for
      } // for

      // vertex-to-cell connectivity
      auto v2e_pack = std::get<1>(r);
      for(auto const & v : v2e_pack) {
        v2e.try_emplace(v.first, v.second);
      } // for

      // cell-to-cell connectivity
      auto e2e_pack = std::get<2>(r);
      for(auto const & e : e2e_pack) {
        e2e.try_emplace(e.first, e.second);
      } // for
    } // for
  } // for

  std::map<Color, unstructured_base::coloring> colorings;

  for(auto p : primaries) {
    colorings[p.first].idx_colorings.resize(1 + Policy::auxiliary_colorings);

    auto & primary =
      colorings.at(p.first).idx_colorings[Policy::primary::index_space];
    // primary.owned = p.second;
    primary.owned.reserve(p.second.size());
    primary.owned.insert(
      primary.owned.begin(), p.second.begin(), p.second.end());

    for(auto e : p.second) {
      if(shared.at(p.first).count(e)) {
        auto d = dependents.at(e);
        primary.shared.emplace_back(
          shared_entity{e, {dependents.at(e).begin(), dependents.at(e).end()}});
      }
      else {
        primary.exclusive.emplace_back(e);
      } // if
    } // for

    for(auto e : ghosts.at(p.first)) {
      primary.ghosts.emplace_back(ghost_entity{e, e2co.at(e)});
    } // for

    util::force_unique(primary.owned);
    util::force_unique(primary.exclusive);
    util::force_unique(primary.shared);
    util::force_unique(primary.ghosts);

    std::stringstream ss;
    ss << "color " << p.first << std::endl;
    ss << log::container{primary.owned} << std::endl;

    ss << "shared:" << std::endl;
    for(auto e : primary.shared) {
      ss << "  " << e.id << ": { ";
      for(auto d : e.dependents) {
        ss << d << " ";
      }
      ss << "}" << std::endl;
    } // for
    ss << std::endl;

    ss << "ghosts: ";
    for(auto e : primary.ghosts) {
      ss << "(" << e.id << ", " << e.color << ") ";
    } // for
    ss << std::endl;

    flog(warn) << ss.str() << std::endl;
  } // for

#if 0
  for(auto p : primaries) {
    auto const & primary =
      colorings.at(p.first).idx_colorings[Policy::primary::index_space];

    for(auto e : primary.owned) {
    } // for
  } // for
#endif

  return colorings;
} // closure

/// \}
} // namespace unstructured_impl
} // namespace topo
} // namespace flecsi
/// \endcond

#endif
