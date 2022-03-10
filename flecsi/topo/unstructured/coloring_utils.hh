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

#include "flecsi/flog.hh"
#include "flecsi/topo/unstructured/coloring_functors.hh"
#include "flecsi/topo/unstructured/types.hh"
#include "flecsi/util/color_map.hh"
#include "flecsi/util/common.hh"
#include "flecsi/util/crs.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/serialize.hh"
#include "flecsi/util/set_utils.hh"

#include <algorithm>
#include <iterator>
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace flecsi {
namespace topo {
namespace unstructured_impl {

/*!
  Create a distributed graph representation of the highest-dimensional
  entity type in the given mesh definition. This is a cell-to-cell graph
  through the \emph{through_dimension} provided by the user.

  @tparam MD The mesh definition type.

  @param md                An instance of the mesh definition type.
  @param through_dimension The dimension through which connections should
                           be computed, e.g., cell connections through faces.
  @param comm              The MPI communicator to use for communication.

  @return A std::tuple containing the distributed graph, and the work products
          of the function, e.g.,  the cell-to-vertex connectivity, the
          vertex-to-cell connectivity, and the cell-to-cell connectivity.
 */

template<typename MD>
auto
make_dcrs(MD const & md,
  Dimension through_dimension,
  MPI_Comm comm = MPI_COMM_WORLD) {
  auto [rank, size] = util::mpi::info(comm);

  std::size_t ne = md.num_entities(MD::dimension());
  std::size_t nv = md.num_entities(0);

  util::color_map ecm(size, size, ne);

  /*
    Get the initial cells for this rank. The cells will be read by
    the root process and sent to each initially "owning" rank using
    a naive distribution.
   */

  auto e2v =
    util::mpi::one_to_allv<pack_cells<MD>>({md, ecm.distribution()}, comm);

  /*
    Create a map of vertex-to-cell connectivity information from
    the initial cell distribution.
   */

  // Populate local vertex connectivity information
  std::size_t offset{ecm.distribution()[rank]};
  std::size_t indices{ecm.indices(rank, 0)};
  std::map<std::size_t, std::vector<std::size_t>> v2c;

  std::size_t i{0};
  for(auto c : e2v) {
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

  /*
    Invert the vertex referencer information, i.e., find the cells
    that are referenced by our connected vertices.
   */

  std::vector<std::vector<std::size_t>> referencer_inverse(size);

  for(auto const & v : v2c) {
    for(auto c : v.second) {
      auto r = util::distribution_offset(ecm.distribution(), c);
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

  /*
    Fill in the cell-to-cell connectivity through the through_dimension
    argument.
   */

  std::map<std::size_t, std::vector<std::size_t>> c2c;
  std::size_t c{offset};
  for(auto const & cd : e2v) {
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

  /*
    Populate the actual distributed crs data structure.
   */

  util::dcrs dcrs;
  dcrs.distribution = ecm.distribution();

  dcrs.offsets.emplace_back(0);
  for(std::size_t c{0}; c < indices; ++c) {
    for(auto cr : c2c[offset + c]) {
      dcrs.indices.emplace_back(cr);
    } // for

    dcrs.offsets.emplace_back(dcrs.offsets[c] + c2c[offset + c].size());
  } // for

  return std::make_tuple(dcrs, e2v, v2c, c2c);
} // make_dcrs

inline auto
migrate(util::dcrs const & naive,
  Color colors,
  std::vector<Color> const & index_colors,
  util::crs & e2v,
  std::map<std::size_t, std::vector<std::size_t>> & v2c,
  std::map<std::size_t, std::vector<std::size_t>> & c2c,
  MPI_Comm comm = MPI_COMM_WORLD) {
  auto [rank, size] = util::mpi::info(comm);

  auto migrated = util::mpi::all_to_allv<migrate_cells>(
    {naive, colors, index_colors, e2v, v2c, c2c, rank}, comm);

  std::map<Color, std::vector<std::size_t>> primaries;
  std::vector<std::size_t> p2m; /* process to mesh map */
  std::map<std::size_t, std::size_t> m2p; /* mesh to process map */

  for(auto const & r : migrated) {
    auto const & cell_pack = std::get<0>(r);
    for(auto const & c : cell_pack) { /* std::vector over cells */
      auto const & info = std::get<0>(c); /* std::array<color, mesh id> */
      e2v.add_row(std::get<1>(c)); /* cell definition (vertex mesh ids) */
      m2p[std::get<1>(info)] = e2v.size() - 1; /* offset map */
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

/*!
  Request owner information about the given entity list from the naive owners.

  @param request  The global ids of the desired entities.

  @param ne       The global number of entities.

  @param colors   The number of colors.

  @param idx_cos  The naive coloring of the entities.

  @param comm     The MPI communicator to use for communication.
 */

inline auto
request_owners(std::vector<std::size_t> const & request,
  std::size_t ne,
  Color colors,
  std::vector<Color> const & idx_cos,
  MPI_Comm comm = MPI_COMM_WORLD) {
  auto [rank, size] = util::mpi::info(comm);

  std::vector<std::vector<std::size_t>> requests(size);
  util::color_map pm(size, size, ne);
  for(auto e : request) {
    requests[pm.process(pm.index_color(e))].emplace_back(e);
  } // for

  auto requested = util::mpi::all_to_allv(
    [&requests](int r, int) -> auto & { return requests[r]; }, comm);

  /*
    Fulfill naive-owner requests with migrated owners.
   */

  std::vector<std::vector<std::size_t>> fulfills(size);
  {
    Color r = 0;
    util::color_map ecm(size, colors, ne);
    for(auto rv : requested) {
      for(auto e : rv) {
        const std::size_t start = pm.index_offset(rank, 0);
        fulfills[r].emplace_back(ecm.process(idx_cos[e - start]));
      } // for
      ++r;
    } // for
  } // scope

  auto fulfilled = util::mpi::all_to_allv(
    [&fulfills](int r, int) -> auto & { return fulfills[r]; }, comm);

  std::vector<std::size_t> offs(size, 0ul);
  std::vector<Color> owners;
  for(auto e : request) {
    auto p = pm.process(pm.index_color(e));
    owners.emplace_back(fulfilled[p][offs[p]++]);
  } // for

  return owners;
} // request_owners

/*!
  FIXME: should this be called closure?

  Form the dependency closure for the given coloring definition and independent
  coloring (primaries).

  @tparam MD The mesh definition type.

  @param md A mesh definition instance.

  @param cd        The coloring definition that defines how the primary and
                   auxiliary entities should be colored, e.g., primary entity
                   dimension, halo depth, and auxiliary types.

  @param idx_cos   The naive coloring of the primary entities.

  @param primaries The owned entities for each color that lives on this
                   process.

  @param e2v       Existing entity-to-vertex connectivity information. This
                   will be added to as the function collects information from
                   other processes.

  @param v2e       Existing vertex-to-entity connectivity information. This
                   will be added to as the function collects information from
                   other processes.

  @param e2e       Existing entity-to-entity connectivity information. This
                   will be added to as the function collects information from
                   other processes.

  @param m2p       Mesh-to-process map for the primary entities.

  @param p2m       Process-to-mesh map for the primary entities.

  @param comm      The MPI communicator to use for communication.
 */

template<typename MD>
auto
color(MD const & md,
  coloring_definition const & cd,
  std::vector<Color> const & idx_cos,
  std::map<Color, std::vector<std::size_t>> const & primaries,
  util::crs & e2v,
  std::map<std::size_t, std::vector<std::size_t>> & v2e,
  std::map<std::size_t, std::vector<std::size_t>> & e2e,
  std::map<std::size_t, std::size_t> & m2p,
  std::vector<std::size_t> & p2m,
  MPI_Comm comm = MPI_COMM_WORLD) {
  auto [rank, size] = util::mpi::info(comm);

  std::size_t ne = md.num_entities(cd.dim);

  /*
    Save color info for all of our initial local entities. The variable
    'e2co' will get updated as we build out our dependencies.
   */

  std::unordered_map<std::size_t, Color> e2co;
  std::map<std::size_t, std::vector<std::size_t>> wkset;
  for(auto const & p : primaries) {
    wkset[p.first].reserve(p.second.size());
    for(auto e : p.second) {
      e2co.try_emplace(e, p.first);
      wkset[p.first].emplace_back(e);
    } // for
  } // for

  std::unordered_map<std::size_t, std::set<Color>> dependents, dependencies;
  std::unordered_map<Color, std::set<std::size_t>> shared, ghost;

  /*
    The gist of this loop is to add layers entities out to the depth specified
    by the input arguments. Additional information about vertex connectivity is
    also collected.

    Each iteration of the loop creates a "layer" of entities that need to be
    requested for that depth of the halo. The working set "wkset" is the
    current collection of entities including layers that were added during
    previous iterations.
   */

  const std::size_t depth = cd.depth;
  for(std::size_t d{0}; d < depth + 1; ++d) {
    std::vector<std::size_t> layer;

    /*
      Create request layer, and add local information.
     */

    for(auto const & p : primaries) {
      for(auto const & e : wkset.at(p.first)) {
        for(auto const & v : e2v[m2p.at(e)]) {
          for(auto const & en : v2e.at(v)) {
            if(e2e.find(en) == e2e.end()) {
              // If we don't have the entity, we need to request it.
              layer.emplace_back(en);

              // Collect the dependent color for request.
              dependencies[en].insert(p.first);

              // If we're within the requested depth, this is also
              // a ghost entity.
              if(d < depth) {
                ghost[p.first].insert(en);
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
              ghost[p.first].insert(en);
            } // if
          } // for
        } // for
      } // for

      wkset.at(p.first).clear();
    } // for

    util::force_unique(layer);

    std::vector<std::vector<std::pair<std::size_t, std::set<Color>>>> request(
      size);
    {
      /*
        Request entity owners from naive-owners.
       */

      auto owners = request_owners(layer, ne, cd.colors, idx_cos, comm);

      /*
        Request entity information from migrated owners.
       */

      std::size_t ei{0};
      for(auto e : layer) {
        request[owners[ei++]].emplace_back(
          std::make_pair(e, dependencies.at(e)));
      } // for
    } // scope

    auto requested = util::mpi::all_to_allv(
      [&request](int r, int) -> auto & { return request[r]; }, comm);

    /*
      Keep track of dependent colors for requested entities.
     */

    std::vector<std::vector<std::size_t>> fulfills(size);
    Color r{0};
    for(auto rv : requested) {
      for(auto e : rv) {
        fulfills[r].emplace_back(e.first);
        if(d < depth) {
          dependents[e.first].insert(e.second.begin(), e.second.end());
          shared[e2co.at(e.first)].insert(e.first);
        }
      } // for
      ++r;
    } // for

    auto fulfilled = util::mpi::all_to_allv<communicate_entities>(
      {fulfills, e2co, e2v, v2e, e2e, m2p}, comm);

    /*
      Update local information.
     */

    for(auto const & r : fulfilled) {
      auto const & entity_pack = std::get<0>(r);
      for(auto const & e : entity_pack) {
        auto const & info = std::get<0>(e);
        e2v.add_row(std::get<1>(e));
        m2p[std::get<1>(info)] = e2v.size() - 1;
        p2m.emplace_back(std::get<1>(info));
        e2co.try_emplace(std::get<1>(info), std::get<0>(info));

        for(auto co : dependencies.at(std::get<1>(info))) {
          wkset.at(co).emplace_back(std::get<1>(info));
        } // for
      } // for

      // vertex-to-entity connectivity
      auto v2e_pack = std::get<1>(r);
      for(auto const & v : v2e_pack) {
        v2e.try_emplace(v.first, v.second);
      } // for

      // entity-to-entity connectivity
      auto e2e_pack = std::get<2>(r);
      for(auto const & e : e2e_pack) {
        e2e.try_emplace(e.first, e.second);
      } // for
    } // for
  } // for

  unstructured_base::coloring coloring;

  /*
    Set meta data for what we can at this point.
   */

  coloring.comm = comm;
  coloring.colors = cd.colors; /* global colors */
  coloring.peers.resize(2 + cd.aux.size());
  coloring.partitions.resize(2 + cd.aux.size());
  coloring.idx_spaces.resize(2 + cd.aux.size());

  /*
    Primary entities.
   */

  coloring.idx_spaces[cd.idx].resize(primaries.size());
  coloring.idx_spaces[cd.vidx].resize(primaries.size());

  std::size_t c{0};
  std::vector<std::size_t> partitions;
  std::vector<Color> process_colors;
  std::vector<std::set<Color>> color_peers(primaries.size());
  std::vector<std::vector<Color>> is_peers(primaries.size());
  util::color_map em(size, cd.colors, ne);
  for(auto p : primaries) {
    process_colors.emplace_back(p.first);
    auto & pc = coloring.idx_spaces[cd.idx][c];
    pc.color = p.first;
    pc.entities = ne;

    pc.coloring.all.reserve(p.second.size());
    pc.coloring.all.insert(
      pc.coloring.all.begin(), p.second.begin(), p.second.end());
    pc.coloring.owned.reserve(p.second.size());
    pc.coloring.owned.insert(
      pc.coloring.owned.begin(), p.second.begin(), p.second.end());

    const auto & sh = shared[p.first];
    for(auto e : p.second) {
      if(sh.count(e)) {
        pc.coloring.shared.emplace_back(
          shared_entity{e, {dependents.at(e).begin(), dependents.at(e).end()}});
      }
      else {
        pc.coloring.exclusive.emplace_back(e);
      } // if
    } // for

    std::set<Color> peers;
    for(auto e : ghost[p.first]) {
      const auto gco = e2co.at(e);
      const auto pr = em.process(gco);
      pc.coloring.ghost.emplace_back(
        ghost_entity{e, pr, em.local_id(pr, gco), gco});
      pc.coloring.all.emplace_back(e);
      peers.insert(gco);
    } // for
    color_peers[c].insert(peers.begin(), peers.end());
    is_peers[c].resize(peers.size());
    std::copy(peers.begin(), peers.end(), is_peers[c].begin());

    util::force_unique(pc.coloring.all);
    util::force_unique(pc.coloring.owned);
    util::force_unique(pc.coloring.exclusive);
    util::force_unique(pc.coloring.shared);
    util::force_unique(pc.coloring.ghost);

    partitions.emplace_back(pc.coloring.all.size());

    // These may not all be used, but we allocate and populate them anyway.
    pc.cnx_allocs.resize(2 + cd.aux.size());
    pc.cnx_colorings.resize(2 + cd.aux.size());
    coloring.idx_spaces[cd.vidx][c].cnx_allocs.resize(2 + cd.aux.size());
    coloring.idx_spaces[cd.vidx][c].cnx_colorings.resize(2 + cd.aux.size());

    /*
      Populate the entity-to-vertex connectivity.
     */

    auto & crs = pc.cnx_colorings[cd.vidx];
    std::size_t o{0};
    crs.offsets.emplace_back(o);
    for(auto e : pc.coloring.all) {
      auto const & vertices = e2v[m2p[e]];
      crs.indices.insert(crs.indices.end(), vertices.begin(), vertices.end());
      crs.offsets.emplace_back(crs.offsets[++o - 1] + vertices.size());
    } // for

    /*
      Set the allocation size for entities-to-vertices, and
      vertices-to-entities (transpose).
     */

    pc.cnx_allocs[cd.vidx] = crs.indices.size();
    coloring.idx_spaces[cd.vidx][c].cnx_allocs[cd.idx] = crs.indices.size();

    ++c;
  } // for

  /*
    Gather the tight peer information for the primary entity type.
   */

  {
    auto pgthr = util::mpi::all_gatherv(is_peers, comm);
    coloring.peers[cd.idx].resize(cd.colors);

    std::size_t p{0};
    for(auto vp : pgthr) { /* over processes */
      std::size_t c{0};
      for(auto pc : vp) { /* over process colors */
        for(auto pe : pc) { /* over peers */
          coloring.peers[cd.idx][em.color_id(p, c)].emplace_back(pe);
        } // for
        ++c;
      } // for
      ++p;
    } // for
  }

  /*
    Gather the process-to-color mapping.
   */

  coloring.process_colors = util::mpi::all_gatherv(process_colors, comm);

  /*
    Gather partition sizes for entities.
   */

  {
    auto pgthr = util::mpi::all_gatherv(partitions, comm);

    coloring.partitions[cd.idx].resize(cd.colors);
    std::size_t p{0};
    for(auto vp : pgthr) {
      std::size_t c{0};
      for(auto v : vp) {
        coloring.partitions[cd.idx][em.color_id(p, c)] = v;
        ++c;
      } // for
      ++p;
    } // for
  }

  /*
    Assign vertex colors.
   */

  std::unordered_map<std::size_t, Color> v2co;
  c = 0;
  for(auto p : primaries) {
    auto const & primary = coloring.idx_spaces[cd.idx][c].coloring;

    for(auto e : primary.owned) {
      for(auto v : e2v[m2p.at(e)]) {
        Color co = std::numeric_limits<Color>::max();
        for(auto ev : v2e.at(v)) {
          co = std::min(e2co[ev], co);
        } // for

        v2co[v] = co;
      } // for
    } // for

    ++c;
  } // for

  /*
    The first of these color maps, "vpm", is the process map, which we need to
    construct the correct naive partitioning of the vertices. The second map,
    "vcm", is the actual color map, which we need to correctly color the
    vertices into the naive partitioning.
   */

  const std::size_t nv = md.num_entities(0);
  util::color_map vpm(size, size, nv);
  util::color_map vcm(size, cd.colors, nv);

  /*
    The following several steps create a coloring of the naive partitioning of
    the vertices (similar to what parmetis would do) using the vertex colors
    computed above. This strategy is employed to enable weak scalability.
   */

  auto rank_colors =
    util::mpi::all_to_allv<vertex_coloring>({vpm.distribution(), v2co}, comm);

  const std::size_t voff = vpm.distribution()[rank];
  std::vector<Color> vtx_idx_cos(vpm.distribution()[rank + 1] - voff);
  for(auto r : rank_colors) {
    for(auto v : r) {
      vtx_idx_cos[std::get<0>(v) - voff] = std::get<1>(v);
    } // for
  } // for

  /*
    Read in the vertex coordinates. These will need to be migrated to the
    owning processes as defined by the coloring that we just populated in
    "vtx_idx_cos".
   */

  auto vertices =
    util::mpi::one_to_allv<pack_vertices<MD>>({md, vpm.distribution()}, comm);

  /*
    Migrate the vertices to their actual owners.
   */

  auto migrated = util::mpi::all_to_allv<migrate_vertices<MD>>(
    {vpm.distribution(), cd.colors, vtx_idx_cos, vertices, rank}, comm);
  std::unordered_map<std::size_t, std::tuple<Color, typename MD::point>> v2info;

  /*
    Update local information.
   */

  for(auto const & r : migrated) {
    for(auto const & v : r) {
      auto const & info = std::get<0>(v);
      v2info.try_emplace(
        std::get<1>(info), std::make_tuple(std::get<0>(info), std::get<1>(v)));
    } // for
  } // for

  /*
    Populate the vertex index coloring. This goes through the current local
    vertex information, populating the index coloring where possible, and
    adding remote requests.
   */

  ghost.clear();
  std::vector<std::size_t> remote;
  c = 0;
  for(auto p : primaries) {
    auto const & primary = coloring.idx_spaces[cd.idx][c].coloring;
    coloring.idx_spaces[cd.vidx][c].entities = nv;
    auto & vaux = coloring.idx_spaces[cd.vidx][c].coloring;

    for(auto e : primary.exclusive) {
      for(auto v : e2v[m2p.at(e)]) {
        vaux.exclusive.emplace_back(v);
        vaux.owned.emplace_back(v);
      } // for
    } // for

    if(cd.colors > 1) {
      for(auto e : primary.shared) {
        for(auto v : e2v[m2p.at(e.id)]) {
          auto vi = v2co.find(v);
          flog_assert(vi != v2co.end(), "invalid vertex id");

          if(vi->second == p.first) {
            // This vertex is owned by the current color
            vaux.shared.emplace_back(shared_entity{v, e.dependents});
            vaux.owned.emplace_back(v);
          }
          else {
            if(vcm.has_color(rank, vi->second)) {
              // This process owns the current color.
              vaux.ghost.emplace_back(ghost_entity{
                v, Color(rank), vcm.local_id(rank, vi->second), vi->second});
            }
            else {
              // The ghost is remote: add to remote requests.
              remote.emplace_back(v);
              ghost[p.first].insert(v);
            } // if
          } // if
        } // for
      } // for

      for(auto e : primary.ghost) {
        for(auto v : e2v[m2p.at(e.id)]) {
          auto vi = v2co.find(v);

          if(vi != v2co.end()) {
            if(vi->second != p.first) {
              const auto pr = vcm.process(vi->second);
              vaux.ghost.emplace_back(
                ghost_entity{v, pr, vcm.local_id(pr, vi->second), vi->second});
            } // if
          }
          else {
            // The ghost is remote: add to remote requests.
            remote.emplace_back(v);
            ghost[p.first].insert(v);
          } // if
        } // for
      } // for
    } // if

    util::force_unique(vaux.owned);
    util::force_unique(vaux.exclusive);
    util::force_unique(vaux.shared);

    ++c;
  } // for

  util::force_unique(remote);

  /*
    Request the migrated owners for remote vertex requests.
   */

  auto owners = request_owners(remote, nv, cd.colors, vtx_idx_cos, comm);

  std::vector<std::vector<std::size_t>> request(size);
  std::size_t vi{0};
  for(auto v : remote) {
    request[owners[vi++]].emplace_back(v);
  } // for

  auto requested = util::mpi::all_to_allv(
    [&request](int r, int) -> auto & { return request[r]; }, comm);

  /*
    Fulfill requests from other ranks for our vertex information.
   */

  std::vector<std::vector<std::tuple<Color, typename MD::point>>> fulfills(
    size);
  Color r{0};
  for(auto rv : requested) {
    for(auto const & v : rv) {
      auto const & vi = v2info.at(v);
      fulfills[r].emplace_back(
        std::make_tuple(std::get<0>(vi), std::get<1>(vi)));
    } // for
    ++r;
  } // for

  auto fulfilled = util::mpi::all_to_allv(
    [&fulfills](int r, int) -> auto & { return fulfills[r]; }, comm);

  /*
    Update our local information.
   */

  std::vector<typename MD::point> v2cd;
  std::map<std::size_t, std::size_t> m2pv;
  std::vector<std::size_t> p2mv;

  std::size_t ri{0};
  for(auto rv : request) {
    std::size_t vi{0};
    for(auto v : rv) {
      v2co.try_emplace(v, std::get<0>(fulfilled[ri][vi]));
      v2cd.emplace_back(std::get<1>(fulfilled[ri][vi]));
      m2pv[v] = vi;
      p2mv.emplace_back(v);
      ++vi;
    } // for
    ++ri;
  } // for

  /*
    Finish populating the vertex index coloring.
   */

  c = 0;
  partitions.clear();
  for(auto p : primaries) {
    auto & vaux = coloring.idx_spaces[cd.vidx][c].coloring;

    vaux.all.reserve(vaux.owned.size() + vaux.ghost.size());
    vaux.all.insert(vaux.all.begin(), vaux.owned.begin(), vaux.owned.end());

    if(cd.colors > 1) {
      // Add ghosts that were available from the local process.
      for(auto v : vaux.ghost) {
        vaux.all.emplace_back(v.id);
      } // for

      if(size > 1) {
        // Add requested ghosts.
        std::set<Color> peers;
        for(auto v : ghost.at(p.first)) {
          const auto gco = v2co.at(v);
          peers.insert(gco);
          const auto pr = vcm.process(gco);
          vaux.ghost.emplace_back(
            ghost_entity{v, pr, vcm.local_id(pr, gco), gco});
          vaux.all.emplace_back(v);
        } // for
        color_peers[c].insert(peers.begin(), peers.end());
        is_peers[c].resize(peers.size());
        std::copy(peers.begin(), peers.end(), is_peers[c].begin());
      } // if
    } // if

    util::force_unique(vaux.all);
    util::force_unique(vaux.ghost);

    partitions.emplace_back(vaux.all.size());

    ++c;
  } // for

  /*
    Gather the tight peer information for the primary entity type.
   */

  {
    auto pgthr = util::mpi::all_gatherv(is_peers, comm);
    coloring.peers[cd.vidx].resize(cd.colors);

    std::size_t p{0};
    for(auto vp : pgthr) { /* over processes */
      std::size_t c{0};
      for(auto pc : vp) { /* over process colors */
        for(auto pe : pc) { /* over peers */
          coloring.peers[cd.vidx][vcm.color_id(p, c)].emplace_back(pe);
        } // for
        ++c;
      } // for
      ++p;
    } // for
  }

  /*
    Gather partition sizes for vertices.
   */

  {
    auto pgthr = util::mpi::all_gatherv(partitions, comm);

    coloring.partitions[cd.vidx].resize(cd.colors);
    std::size_t p{0};
    for(auto vp : pgthr) {
      std::size_t c{0};
      for(auto v : vp) {
        coloring.partitions[cd.vidx][vcm.color_id(p, c)] = v;
        ++c;
      } // for
      ++p;
    } // for
  }

  /*
    Auxiliary entities.
   */

  /*
    Gather color peers (superset of color peers over all index spaces).
   */

  {
    auto pgthr = util::mpi::all_gatherv(color_peers, comm);

    coloring.color_peers.resize(cd.colors);
    std::size_t p{0};
    for(auto vp : pgthr) {
      std::size_t c{0};
      for(auto pc : vp) {
        coloring.color_peers[em.color_id(p, c)] = pc.size();
        ++c;
      } // for
      ++p;
    } // for
  }

  return coloring;
} // color

/*!
  Build connectivity through connectivity intersection.  Given X-to-Y
  and Y-to-Z connectivities, build X-to-Z connectivity.

  \param c2f X-to-Y connectivity
  \param f2e Y-to-Z connectivity
  \return X-to-Z connectivity (c2e)
*/
inline util::crs
intersect_connectivity(const util::crs & c2f, const util::crs & f2e) {
  util::crs c2e;
  c2e.offsets.reserve(c2f.offsets.size());
  // Note: this is a rough estimate.
  c2e.indices.reserve(c2f.indices.size() + f2e.indices.size());

  for(std::size_t cell = 0; cell < c2f.offsets.size() - 1; cell++) {
    std::vector<std::size_t> edges;

    // accumulate edges in cell
    for(std::size_t fi = c2f.offsets[cell]; fi < c2f.offsets[cell + 1]; fi++) {
      auto face = c2f.indices[fi];
      for(std::size_t ei = f2e.offsets[face]; ei < f2e.offsets[face + 1];
          ei++) {
        auto it = std::find(edges.begin(), edges.end(), f2e.indices[ei]);
        if(it == edges.end()) {
          edges.push_back(f2e.indices[ei]);
        }
      }
    }
    c2e.add_row(edges.begin(), edges.end());
  }

  return c2e;
}

/*!
  Build intermediary entities locally from cell to vertex graph.

  @tparam from_dim index of the dimension for rows indices in e2v.
  @param dim index of dimension for intermediary.
  @param md mesh definition (provides function to build intermediary from list
  of vertices).
  @param e2v entity to vertex graph.
  @param p2m Process-to-mesh map for the primary entities.
*/
template<Dimension from_dim, class MD>
auto
build_intermediary(Dimension dim,
  const MD & md,
  const std::vector<std::vector<std::size_t>> & e2v,
  const std::vector<std::size_t> & p2m) {
  flog_assert((dim > 0 and dim < MD::dimension()),
    "Invalid dimension for intermediary entity: " << dim);

  util::crs c2e, i2v;
  std::map<std::vector<std::size_t>, std::size_t> v2e;

  // temporary storage
  util::crs edges;
  std::vector<typename MD::index> sorted_vs;
  std::vector<typename MD::index> these_edges;

  // iterate over cells, adding all of their edges to the table
  for(std::size_t cell{0}; cell < e2v.size(); cell++) {
    const auto & these_verts = e2v[cell];
    // clear this cells edges
    these_edges.clear();

    // build the edges for the cell
    edges.offsets.clear();
    edges.indices.clear();
    if constexpr(MD::dimension() == from_dim)
      md.build_intermediary_from_vertices(dim, p2m[cell], these_verts, edges);
    else
      md.build_intermediary_from_vertices(dim, 0, these_verts, edges);

    /*
      look for existing vertex pairs in the edge-to-vertex master list
      or add a new edge to the list.  Add the matched edge id to the
      cell-to-edge list
    */
    for(std::size_t row = 0; row < edges.offsets.size() - 1; row++) {
      // sort the vertices
      auto beg = edges.indices.begin() + edges.offsets[row];
      auto end = edges.indices.begin() + edges.offsets[row + 1];
      sorted_vs.assign(beg, end);
      std::sort(sorted_vs.begin(), sorted_vs.end());
      // if we don't find the edge
      auto it = v2e.find(sorted_vs);
      if(it == v2e.end()) {
        // add to the local reverse map
        auto edgeid = v2e.size();
        v2e.emplace(sorted_vs, edgeid);
        // add to the original sorted and unsorted mpas
        i2v.add_row(beg, end);
        // add to the list of edges
        these_edges.push_back(edgeid);
      }
      else {
        // if we find the edge, add the id to the list of edges
        these_edges.push_back(it->second);
      }
    }
    // add this cells edges
    c2e.add_row(these_edges.begin(), these_edges.end());
  }

  return std::make_tuple(c2e, i2v, v2e);
} // build_intermediary

} // namespace unstructured_impl
} // namespace topo
} // namespace flecsi
