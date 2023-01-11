// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_UNSTRUCTURED_COLORING_UTILS_HH
#define FLECSI_TOPO_UNSTRUCTURED_COLORING_UTILS_HH

#include "flecsi/flog.hh"
#include "flecsi/topo/types.hh"
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

/// \cond core

namespace flecsi {
namespace topo {
namespace unstructured_impl {
/// \addtogroup unstructured
/// \{

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

#ifdef DOXYGEN
/// An example mesh definition that is not really implemented.
struct mesh_definition {
  /// Get the dimensionality of the mesh.
  static constexpr Dimension dimension();

  /// Get the global number of entities of a kind.
  util::gid num_entities(Dimension) const;

  /// Get the entities connected to or associated with an entity.
  /// \param id of entity of dimension \a from
  /// \return ids of entities of dimension \a to
  std::vector<std::size_t>
  entities(Dimension from, Dimension to, util::gid id) const;

  /// Get entities of a given dimension associated with a set of vertices.
  /// \param k Entity kind
  /// \param g global ID of primary, or 0 if \a v pertains to a
  ///   lower-dimensionality entity
  /// \param v vertex IDs
  /// \param[out] e entities, initially empty
  void make_entity(entity_kind k,
    util::gid g,
    const std::vector<util::gid> & v,
    util::crs & e);
}; // struct mesh_definition
#endif

/// The coloring_utils interface provides utility methods for
/// generating colorings of unstructured input meshes. The member functions
/// in this interface modify the internal state of the object to add or create
/// coloring information. Once the requisite methods have been invoked, the
/// coloring object can be accessed via the \ref generate method.
/// \ingroup unstructured
/// \tparam MD mesh definition, following mesh_definition; usually deduced
///   from constructor argument
template<typename MD>
struct coloring_utils {
  /// connectivity allocation: from, to, transpose
  using connectivity_allocation_t = std::tuple<std::size_t, std::size_t, bool>;

  /// Offset to auxiliary information.
  static constexpr std::size_t aoff = 2;

  /// Construct a coloring_utils object with mesh definition type MD.
  /// \param md   The mesh definition.
  /// \param cd   The coloring definition.
  /// \param ca   Information on what connectivities to initialize.
  /// \param comm An MPI communicator.
  coloring_utils(MD const * md,
    unstructured_impl::coloring_definition const & cd,
    std::vector<connectivity_allocation_t> const & ca,
    MPI_Comm comm = MPI_COMM_WORLD)
    : md_(*md), cd_(cd), ca_(ca), comm_(comm) {
    std::tie(rank_, size_) = util::mpi::info(comm_);

    // Create a map from mesh definition entity kinds to local offsets.
    idmap_[cd.cid.kind] = 0;
    idmap_[cd.vid.kind] = 1;

    for(std::size_t i{0}; i < cd.aidxs.size(); ++i) {
      idmap_[cd.aidxs[i].kind] = i + aoff;
    } // for

    conns_.resize(idmap_.size());
    auxs_.resize(cd.aidxs.size());

    coloring_.comm = comm_;
    coloring_.colors = cd_.colors; /* global colors */
    coloring_.peers.resize(2 + cd_.aidxs.size());
    coloring_.partitions.resize(2 + cd_.aidxs.size());
    coloring_.idx_spaces.resize(2 + cd_.aidxs.size());
  } // coloring_utils

  /// Create connectivity graph for the given entity type.
  ///
  /// \note This method marshalls the naive partition information on the
  /// root process and sends the respective data to each initially-owning
  /// process. The current implementation is designed to avoid having many
  /// processes hit the file system at once. This may not be optimal on all
  /// parallel file systems.
  ///
  /// \param  kind   Entity kind for which to build connectivity.
  void create_graph(entity_kind kind);

  /// Color the primary entity type using the provided coloring function.
  /// \param shared maximum number of shared vertices to disregard
  /// \param c coloring function object with the signature of
  ///          flecsi::util::parmetis::color
  /// \warning This method will fail on certain non-convex mesh entities. In
  /// particular, if a non-convex cell abuts a cell that is inscribed in the
  /// convex hull of the non-convex entity, it is possible that the non-convex
  /// entity will share \em shared vertices with an entity that abuts the
  /// convex hull without actually being connected. Other similar cases of
  /// this nature are also possible.
  template<typename C>
  void color_primaries(util::id shared, C && c);

  /// Redistribute the primary entities. This method moves the primary
  /// entities to their owning colors.
  void migrate_primaries();

  /// Close the local primary distributions with respect to off-color
  /// dependencies, i.e., assemble the local shared and ghost information.
  void close_primaries();

  /// Create the vertex coloring induced by the primary coloring using the
  /// heuristic that each vertex is assigned the color of the lowest color
  /// of a primary that references it.
  void color_vertices();

  /// Close the local vertex distributions with respect to off-color
  /// dependencies, i.e., assemble the local shared and ghost information.
  void close_vertices();

  /// Build auxiliary entities for the given entity kind.
  /// \param kind The mesh definition entity kind of the auxiliary to build.
  /// \param h    The heuristic to use to assign colors to the newly-created
  ///             auxiliaries. Specifying \em vertex will assign each new
  ///             auxiliary the color of the lowest-ranked vertex that is
  ///             connected to the auxiliary. Specifying \em cell will do the
  ///             same, but with the lowest-ranked cell.
  enum heuristic { vertex, cell };
  void build_auxiliary(entity_kind kind, heuristic h = vertex);

  /// Create the auxiliary coloring induced by the primary coloring using the
  /// heuristic that each auxiliary is assigned the color of the lowest color
  /// of a primary that references it.
  /// \param kind       The mesh definition entity kind of the auxiliary to
  ///                   color.
  void color_auxiliary(entity_kind kind);

  /// Close the local auxiliary distributions with respect to off-color
  /// dependencies, i.e., assemble the local shared and ghost information.
  /// \param kind The mesh definition entity kind of the auxiliary to close.
  /// \param idx  The entity index space of the auxiliary to close.
  void close_auxiliary(entity_kind kind, std::size_t idx);

  /// Return a reference to the coloring object. Note that this method
  /// has side effects: it gathers global peer information for the superset
  /// of communication pairs over the collection of index spaces in the
  /// coloring, and it computes the connectivity allocation sizes for the
  /// connectivities specified in the coloring_definition.
  auto & generate();

  /// Distribute a field based on the existing \p kind repartition.
  /// Only primaries and vertices entity_kind are supported.
  /// The input, \p elems, must be empty everywhere except on rank 0.
  /// This function returns a vector of vectors of the field distributed in a
  /// fashion similar to ParMETIS. Each vector correspond to the colors of the
  /// process, in order.
  template<class T>
  std::vector<std::vector<T>> send_field(entity_kind kind,
    const std::vector<T> & elems);

  //////////////////////////////////////////////////////////////////////////////
  // Internal: These are used for testing.
  //////////////////////////////////////////////////////////////////////////////

  util::equal_map pmap() const {
    return {cd_.colors, Color(size_)};
  }
  auto ours() const {
    return pmap()[rank_];
  }
  bool ours(Color g) const {
    const auto r = ours();
    return g >= r.front() && g < *r.end(); // well defined even if empty
  }
  Color lc(Color g) const {
    return g - ours().front();
  }

  // Return the intermediate connectivity information for the
  // primary entity type.
  auto & primary_connectivity_state() {
    return conns_[idmap_.at(cd_.cid.kind)];
  }

  // Return the intermediate connectivity information for the given
  // entity kind.
  auto & connectivity_state(entity_kind kind) {
    flog_assert(kind != cd_.vid.kind,
      "invalid kind: state does not store vertex connectivity");
    return conns_[idmap_.at(kind)];
  }

  // Return the color assignments for the naive partitioning.
  auto & primary_raw() {
    return primary_raw_;
  }

  // Return a vector relating local colors to primary indices. These are the
  // owned entities for each color (Ghost entities are excluded.).
  auto & primaries() {
    return primaries_;
  }

  auto & get_naive() const {
    return naive;
  }

private:
  bool shared_has(Color c, util::gid i) const {
    const auto it = shared_.find(c);
    return it != shared_.end() && it->second.count(i);
  }

  void build_intermediary(std::size_t kind,
    const util::crs & e2v,
    std::vector<util::gid> const & p2m);

  util::gid num_primaries() {
    return md_.num_entities(cd_.cid.kind);
  }

  util::gid num_vertices() {
    return md_.num_entities(cd_.vid.kind);
  }

  util::gid num_entities(entity_kind kind) {
    return built_.count(kind) ? auxiliary_state(kind).entities
                              : md_.num_entities(kind);
  }

  auto & primary_coloring() {
    return coloring_.idx_spaces[cd_.cid.idx];
  }

  auto & vertex_coloring() {
    return coloring_.idx_spaces[cd_.vid.idx];
  }

  auto & auxiliary_coloring(std::size_t idx) {
    return coloring_.idx_spaces[idx];
  }

  auto & primary_peers() {
    return coloring_.peers[cd_.cid.idx];
  }

  auto & vertex_peers() {
    return coloring_.peers[cd_.vid.idx];
  }

  auto & primary_partitions() {
    return coloring_.partitions[cd_.cid.idx];
  }

  auto & vertex_partitions() {
    return coloring_.partitions[cd_.vid.idx];
  }

  auto & auxiliary_state(entity_kind kind) {
    return auxs_[idmap_.at(kind) - aoff];
  }

  struct connectivity_state_t {
    util::crs e2v;
    std::map<util::gid, std::vector<util::gid>> v2e, e2e;
    std::vector<util::gid> p2m;
    std::map<util::gid, util::id> m2p;
  };

  struct auxiliary_state_t {
    util::gid entities;
    util::crs e2i, i2e, i2v;
    std::map<std::vector<util::gid>, util::gid> v2i;
    std::map<Color, std::map<util::gid, std::set<Color>>> ldependents,
      dependents;
    std::map<Color, std::map<util::gid, Color>> ldependencies, dependencies;
    std::map<util::gid, std::pair<Color, bool>> a2co;
    std::map<util::gid, std::pair<Color, std::vector<util::gid>>> ghost;
    std::vector<
      std::map<std::vector<util::gid>, std::pair<util::id, util::gid>>>
      shared;
    std::map<util::id, util::gid> l2g;
  };

  MD const & md_;
  unstructured_impl::coloring_definition cd_;
  std::vector<connectivity_allocation_t> ca_;
  MPI_Comm comm_;
  int size_, rank_;
  std::map<std::size_t, std::size_t> idmap_;

  // Connectivity state is associated with an entity's kind (as opposed to
  // its index space).
  std::vector<connectivity_state_t> conns_;
  util::crs naive;

  std::vector<Color> primary_raw_;
  std::vector<Color> vertex_raw_;
  std::vector<std::vector<util::gid>> primaries_;
  std::vector<std::map<entity_index_space, std::set<util::gid>>> all;
  std::unordered_map<util::gid, Color> p2co_;
  std::unordered_map<util::gid, Color> v2co_;
  std::unordered_map<Color, std::set<util::gid>> shared_;
  std::vector<std::set<util::gid>> ghost;
  std::unordered_map<std::size_t, std::set<Color>> vdeps_;
  std::vector<std::size_t> rghost_;
  unstructured_base::coloring coloring_;
  std::vector<std::set<Color>> color_peers_; // both sending and receiving

  // Auxiliary state is associated with an entity's kind (as opposed to
  // its index space).
  std::vector<auxiliary_state_t> auxs_;
  std::set<std::size_t> built_;
}; // struct coloring_utils

// New mechanisms for color initialization that do not involve a shared file
// being opened by all processes may require changes or alternatives to this
// function.
template<typename MD>
void
coloring_utils<MD>::create_graph(entity_kind kind) {

  const util::equal_map ecm(num_entities(kind), size_);

  /*
    Get the initial entities for this rank. The entities will be read by
    the root process and sent to each initially "owning" rank using
    a naive distribution.
   */

  auto & cnns = connectivity_state(kind);
  cnns.e2v = util::mpi::one_to_allv(pack_definitions(md_, kind, ecm), comm_);

  /*
    Create a map of vertex-to-entity connectivity information from
    the initial entity distribution.
   */

  // Populate local vertex connectivity information
  const std::size_t offset = ecm(rank_);

  std::size_t i{0};
  for(auto c : cnns.e2v) {
    for(auto v : c) {
      cnns.v2e[v].emplace_back(offset + i);
    } // for
    ++i;
  } // for

  // Request all referencers of our connected vertices
  const util::equal_map vm(num_vertices(), size_);
  auto referencers =
    util::mpi::all_to_allv(vertex_referencers(cnns.v2e, vm, rank_), comm_);

  /*
    Update our local connectivity information. We now have all
    vertex-to-entity connectivity informaiton for the naive distribution
    of entities that we own.
   */

  i = 0;
  for(auto & r : referencers) {
    for(auto v : r) {
      for(auto c : v.second) {
        cnns.v2e[v.first].emplace_back(c);
      } // for
    } // for
    ++i;
  } // for

  // Remove duplicate referencers
  util::unique_each(cnns.v2e);

  /*
    Invert the vertex referencer information, i.e., find the entities
    that are referenced by our connected vertices.
   */

  std::vector<std::vector<util::gid>> referencer_inverse(size_);

  for(auto const & v : cnns.v2e) {
    for(auto c : v.second) {
      const int r = ecm.bin(c);
      if(r != rank_) {
        referencer_inverse[r].emplace_back(v.first);
      } // if
    } // for
  } // for

  // Remove duplicate inverses
  util::unique_each(referencer_inverse);

  // Request vertex-to-entity connectivity for the entities that are
  // on other ranks in the naive entity distribution.
  auto connectivity = util::mpi::all_to_allv(
    entity_connectivity(referencer_inverse, cnns.v2e, vm, rank_), comm_);

  for(auto & r : connectivity) {
    for(auto & v : r) {
      for(auto c : v.second) {
        cnns.v2e[v.first].emplace_back(c);
      } // for
    } // for
  } // for

  // Remove duplicate referencers
  util::unique_each(cnns.v2e);
}

template<typename MD>
template<typename C>
void
coloring_utils<MD>::color_primaries(util::id shared, C && f) {
  auto & cnns = primary_connectivity_state();
  const util::equal_map ecm(num_primaries(), size_);

  /*
    Fill in the entity-to-entity connectivity that have "shared" vertices
    in common.
   */

  std::size_t c = ecm(rank_);
  for(auto const & cdef : cnns.e2v) {
    std::map<util::gid, util::id> shr;

    for(auto v : cdef) {
      auto it = cnns.v2e.find(v);
      if(it != cnns.v2e.end()) {
        for(auto rc : cnns.v2e.at(v)) {
          if(rc != c)
            ++shr[rc];
        } // for
      } // if
    } // for

    for(auto tc : shr) {
      if(tc.second > shared) {
        cnns.e2e[c].emplace_back(tc.first);
        cnns.e2e[tc.first].emplace_back(c);
      } // if
    } // for

    ++c;
  } // for

  // Remove duplicate connections
  util::unique_each(cnns.e2e);

  /*
    Populate the actual distributed crs data structure.
   */

  for(const std::size_t c : ecm[rank_]) {
    naive.add_row(cnns.e2e[c]);
  } // for

  primary_raw_ = std::forward<C>(f)(ecm, naive, cd_.colors, comm_);
} // color_primaries

template<typename MD>
void
coloring_utils<MD>::migrate_primaries() {

  auto & cnns = primary_connectivity_state();

  auto migrated = util::mpi::all_to_allv(
    move_primaries(util::equal_map(num_primaries(), size_),
      cd_.colors,
      primary_raw_,
      cnns.e2v,
      cnns.v2e,
      cnns.e2e,
      rank_),
    comm_);

  primaries_.resize(ours().size());
  ghost.resize(primaries_.size());

  for(auto const & r : migrated) {
    auto const & cell_pack = std::get<0>(r);
    for(auto const & c : cell_pack) { /* std::vector over cells */
      auto const & info = std::get<0>(c); /* std::array<color, mesh id> */
      cnns.e2v.add_row(std::get<1>(c)); /* cell definition (vertex mesh ids) */
      cnns.m2p[std::get<1>(info)] = cnns.e2v.size() - 1; /* offset map */
      cnns.p2m.emplace_back(std::get<1>(info)); /* cell mesh id */
      primaries()[lc(std::get<0>(info))].emplace_back(std::get<1>(info));
    } // for

    // vertex-to-cell connectivity
    auto v2c_pack = std::get<1>(r);
    for(auto const & v : v2c_pack) {
      cnns.v2e.try_emplace(v.first, v.second);
    } // for

    // cell-to-cell connectivity
    auto c2c_pack = std::get<2>(r);
    for(auto const & c : c2c_pack) {
      cnns.e2e.try_emplace(c.first, c.second);
    } // for
  } // for

  color_peers_.resize(primaries().size());
  coloring_.idx_spaces[cd_.cid.idx].resize(primaries().size());
  coloring_.idx_spaces[cd_.vid.idx].resize(primaries().size());
  for(auto const [kind, idx] : cd_.aidxs) {
    coloring_.idx_spaces[idx].resize(primaries().size());
  } // for

  // Set the vector sizes for connectivity information.
  for(auto [from, to, transpose] : ca_) {
    for(auto & pc : coloring_.idx_spaces[from]) {
      pc.cnx_allocs.resize(2 + cd_.aidxs.size());
      pc.cnx_colorings.resize(2 + cd_.aidxs.size());
    } // for
  } // for
} // migrate_primaries

/// Request owner information about the given entity list from the naive
/// owners.
/// @param request  The global ids of the desired entities.
/// @param ne       The global number of entities.
/// @param colors   The number of colors.
/// @param idx_cos  The naive coloring of the entities.
/// @param comm     The MPI communicator to use for communication.
/// \return the owning process for each entity
inline std::vector<Color>
request_owners(std::vector<util::gid> const & request,
  util::gid ne,
  Color colors,
  std::vector<Color> const & idx_cos,
  MPI_Comm comm = MPI_COMM_WORLD) {
  auto [rank, size] = util::mpi::info(comm);

  std::vector<std::vector<util::gid>> requests(size);
  const util::equal_map pm(ne, size);
  for(auto e : request) {
    requests[pm.bin(e)].emplace_back(e);
  } // for

  auto requested = util::mpi::all_to_allv(
    [&requests](int r, int) -> auto & { return requests[r]; }, comm);

  /*
    Fulfill naive-owner requests with migrated owners.
   */

  std::vector<std::vector<util::gid>> fulfill(size);
  {
    const std::size_t start = pm(rank);
    Color r = 0;
    const util::equal_map ecm(colors, size);
    for(auto rv : requested) {
      for(auto e : rv) {
        fulfill[r].emplace_back(ecm.bin(idx_cos[e - start]));
      } // for
      ++r;
    } // for
  } // scope

  auto fulfilled = util::mpi::all_to_allv(
    [&fulfill](int r, int) -> auto & { return fulfill[r]; }, comm);

  std::vector<std::size_t> offs(size, 0ul);
  std::vector<Color> owners;
  for(auto e : request) {
    auto p = pm.bin(e);
    owners.emplace_back(fulfilled[p][offs[p]++]);
  } // for

  return owners;
} // request_owners

template<typename MD>
void
coloring_utils<MD>::close_primaries() {

  /*
    Save color info for all of our initial local entities. The variable
    'p2co' will get updated as we build out our dependencies.
   */

  std::vector<std::vector<util::gid>> wkset;
  {
    auto co = ours().begin();
    for(auto const & p : primaries()) {
      auto & v = wkset.emplace_back();
      v.reserve(p.size());
      for(auto e : p) {
        p2co_.try_emplace(e, *co);
        v.emplace_back(e);
      } // for
      ++co;
    } // for
  }

  auto & cnns = primary_connectivity_state();
  std::unordered_map<util::gid, std::set<Color>> dependents, dependencies;

  /*
    The gist of this loop is to add layers of entities out to the depth
    specified by the input arguments. Additional information about vertex
    connectivity is also collected.

    Each iteration of the loop creates a "layer" of entities that need to be
    requested for that depth of the halo. The working set "wkset" is the
    current collection of entities including layers that were added during
    previous iterations.
   */

  for(std::size_t d{0}; d < cd_.depth + 1; ++d) {
    std::vector<util::gid> layer;

    /*
      Create request layer, and add local information.
     */

    for(const Color co : ours()) {
      const Color lco = lc(co);
      auto & g = ghost[lco];
      for(auto const & e : std::exchange(wkset[lco], {})) {
        for(auto const & v : cnns.e2v[cnns.m2p.at(e)]) {
          for(auto const & en : cnns.v2e.at(v)) {
            if(cnns.e2e.find(en) == cnns.e2e.end()) {
              // If we don't have the entity, we need to request it.
              layer.emplace_back(en);

              // This primary depends on the current color.
              dependencies[en].insert(co);

              // If we're within the requested depth, this is also
              // a ghost entity.
              if(d < cd_.depth) {
                g.insert(en);
                rghost_.emplace_back(en);
              }
            }
            else if(d < cd_.depth && p2co_.at(en) != co) {
              // This entity is on the local process, but not
              // owned by the current color.

              // Add this entity to the shared of the owning color.
              shared_[p2co_.at(en)].insert(en);

              // Add the current color as a dependent of this primary.
              vdeps_[en].insert(co);
              dependents[en].insert(co);

              // This entity is a ghost for the current color.
              g.insert(en);
            } // if
          } // for
        } // for
      } // for
    } // for

    util::force_unique(layer);

    std::vector<std::vector<std::pair<util::gid, std::set<Color>>>> request(
      size_);
    {
      /*
        Request entity owners from naive-owners.
       */

      auto owners =
        request_owners(layer, num_primaries(), cd_.colors, primary_raw_, comm_);

      /*
        Request entity information from migrated owners.
       */

      std::size_t eid{0};
      for(auto e : layer) {
        request[owners[eid++]].emplace_back(
          std::make_pair(e, dependencies.at(e)));
      } // for
    } // scope

    auto requested = util::mpi::all_to_allv(
      [&request](int r, int) -> auto & { return request[r]; }, comm_);

    /*
      Keep track of dependent colors for requested entities.
     */

    std::vector<std::vector<util::gid>> fulfill(size_);
    Color r{0};
    for(const auto & rv : requested) {
      for(auto e : rv) {
        // Add this entity to the list of entities that we need to fulfill
        // in `communicate_entities` below.
        fulfill[r].emplace_back(e.first);

        // Add this primary to shared and add the colors that depend on it
        if(d < cd_.depth) {
          dependents[e.first].insert(e.second.begin(), e.second.end());
          shared_[p2co_.at(e.first)].insert(e.first);
        }
      } // for
      ++r;
    } // for

    auto fulfilled = util::mpi::all_to_allv(
      communicate_entities(
        fulfill, dependents, p2co_, cnns.e2v, cnns.v2e, cnns.e2e, cnns.m2p),
      comm_);

    /*
      Update local information.
     */

    for(auto const & r : fulfilled) {
      auto const & entity_pack = std::get<0>(r);
      for(auto const & e : entity_pack) {
        auto const & info = std::get<0>(e);
        auto const id = std::get<1>(info);
        cnns.e2v.add_row(std::get<1>(e));
        cnns.m2p[id] = cnns.e2v.size() - 1;
        p2co_.try_emplace(id, std::get<0>(info));
        auto const & deps = std::get<2>(e);

        if(d < cd_.depth) {
          vdeps_[id].insert(deps.begin(), deps.end());
        } // if

        for(auto co : dependencies.at(id)) {
          wkset.at(lc(co)).emplace_back(id);
        } // for
      } // for

      // vertex-to-entity connectivity
      auto v2e_pack = std::get<1>(r);
      for(auto const & v : v2e_pack) {
        cnns.v2e.try_emplace(v.first, v.second);
      } // for

      // entity-to-entity connectivity
      auto e2e_pack = std::get<2>(r);
      for(auto const & e : e2e_pack) {
        cnns.e2e.try_emplace(e.first, e.second);
      } // for
    } // for
  } // for

  util::force_unique(rghost_);

  /*
    Primary entities.
   */

  auto & partitions = coloring_.partitions[cd_.cid.idx];
  std::vector<std::vector<Color>> is_peers(primaries().size());
  {
    auto co = ours().begin();
    for(const auto & p : primaries()) {
      const Color lco = lc(*co);
      auto & pc = coloring_.idx_spaces[cd_.cid.idx][lco];
      pc.color = *co;
      pc.entities = num_primaries();

      pc.coloring.all.reserve(p.size());
      pc.coloring.all.insert(pc.coloring.all.begin(), p.begin(), p.end());
      pc.coloring.owned.reserve(p.size());
      pc.coloring.owned.insert(pc.coloring.owned.begin(), p.begin(), p.end());

      for(auto e : p) {
        if(shared_has(*co, e)) {
          pc.coloring.shared.emplace_back(shared_entity{
            e, {dependents.at(e).begin(), dependents.at(e).end()}});
        }
        else {
          pc.coloring.exclusive.emplace_back(e);
        } // if
      } // for

      auto & cp = color_peers_[lco];
      for(auto e : ghost[lco]) {
        const auto gco = p2co_.at(e);
        const auto [pr, lco] = pmap().invert(gco);
        pc.coloring.ghost.emplace_back(ghost_entity{e, pr, Color(lco), gco});
        pc.coloring.all.emplace_back(e);
        cp.insert(gco);
      } // for
      std::set<Color> peers;
      for(auto e : shared_[*co])
        peers.insert(dependents.at(e).begin(), dependents.at(e).end());
      cp.insert(peers.begin(), peers.end());
      pc.peers.assign(peers.begin(), peers.end());
      is_peers[lco] = pc.peers;

      flog_assert(pc.coloring.owned.size() ==
                    pc.coloring.exclusive.size() + pc.coloring.shared.size(),
        "exclusive and shared primaries != owned primaries");

      util::force_unique(pc.coloring.all);
      util::force_unique(pc.coloring.owned);
      util::force_unique(pc.coloring.exclusive);
      util::force_unique(pc.coloring.shared);
      util::force_unique(pc.coloring.ghost);

      if(auxs_.size()) {
        all.emplace_back()[cd_.cid.idx].insert(
          pc.coloring.all.begin(), pc.coloring.all.end());
      }

      partitions.emplace_back(pc.coloring.all.size());

      /*
        Populate the entity-to-vertex connectivity.
       */

      auto & crs = pc.cnx_colorings[cd_.vid.idx];
      for(auto e : pc.coloring.all) {
        crs.add_row(cnns.e2v[cnns.m2p[e]]);
      } // for

      ++co;
    } // for
  } // scope

  /*
    Gather the tight peer information for the primary entity type.
   */

  {
    auto & peers = coloring_.peers[cd_.cid.idx];
    peers.reserve(cd_.colors);

    for(auto vp : util::mpi::all_gatherv(is_peers, comm_)) {
      for(auto & pc : vp) { /* over process colors */
        peers.push_back(std::move(pc));
      } // for
    } // for
  } // scope

  for(const auto r : pmap())
    coloring_.process_colors.emplace_back(r.begin(), r.end());

  /*
    Gather partition sizes for entities.
   */
  concatenate(partitions, cd_.colors, comm_);
} // close_primaries

template<typename MD>
void
coloring_utils<MD>::color_vertices() {
  auto & cnns = primary_connectivity_state();

  for(std::size_t lco{0}; lco < primaries().size(); ++lco) {
    auto const & primary = primary_coloring()[lco].coloring;

    for(auto e : primary.owned) {
      for(auto v : cnns.e2v[cnns.m2p.at(e)]) {
        Color co = std::numeric_limits<Color>::max();
        for(auto ev : cnns.v2e.at(v)) {
          co = std::min(p2co_.at(ev), co);
        } // for

        v2co_[v] = co;
      } // for
    } // for
  } // for

  /*
    The following several steps create a coloring of the naive partitioning of
    the vertices (similar to what parmetis would do) using the vertex colors
    computed above. This strategy is employed to enable weak scalability.
   */

  const util::equal_map vpm(num_vertices(), size_);
  auto rank_colors = util::mpi::all_to_allv(rank_coloring(vpm, v2co_), comm_);

  const auto vr = vpm[rank_];
  vertex_raw_.resize(vr.size());
  for(auto r : rank_colors) {
    for(auto v : r) {
      vertex_raw_[std::get<0>(v) - vr.front()] = std::get<1>(v);
    } // for
  } // for
} // color_vertices

template<typename MD>
void
coloring_utils<MD>::close_vertices() {
  auto const & cnns = primary_connectivity_state();

  /*
    Populate the vertex index coloring. This goes through the current local
    vertex information, populating the index coloring where possible, and
    adding remote requests.
   */

  vertex_coloring().resize(primaries().size());
  std::vector<util::gid> remote;
  std::unordered_map<Color, std::set<util::gid>> vg;
  {
    for(const Color co : ours()) {
      const Color lco = lc(co);
      auto primary = primary_coloring()[lco].coloring;
      vertex_coloring()[lco].entities = num_vertices();
      auto & pc = vertex_coloring()[lco];
      pc.color = co;
      pc.entities = num_vertices();

      std::unordered_map<std::size_t, std::set<std::size_t>> dependents;
      std::set<std::size_t> shared;
      if(cd_.colors > 1) {
        // Go through the shared primaries and look for ghosts. Some of these
        // may be on the local processor, i.e., we don't need to request
        // remote information about them.
        for(auto e : primary.shared) {
          for(auto v : cnns.e2v[cnns.m2p.at(e.id)]) {
            auto vit = v2co_.find(v);
            flog_assert(vit != v2co_.end(), "invalid vertex id");

            if(vit->second == co) {
              shared.insert(v);
              dependents[v].insert(e.dependents.begin(), e.dependents.end());
              pc.coloring.owned.emplace_back(v);
            }
            else {
              if(const auto [pr, li] = pmap().invert(vit->second);
                 pr == Color(rank_)) {
                // This process owns the current color.
                pc.coloring.ghost.emplace_back(
                  ghost_entity{v, pr, Color(li), vit->second});
              }
              else {
                // The ghost is remote: add to remote requests.
                remote.emplace_back(v);
                vg[co].insert(v);
              } // if
            } // if
          } // for
        } // for

        // Go through the ghost primaries and look for ghosts. Some of these
        // may also be on the local processor.
        for(auto e : primary.ghost) {
          for(auto v : cnns.e2v[cnns.m2p.at(e.id)]) {
            auto vit = v2co_.find(v);

            // Add dependents through ghosts.
            if(shared.count(v) && vdeps_.count(e.id)) {
              auto deps = vdeps_.at(e.id);
              deps.erase(co);
              dependents[v].insert(deps.begin(), deps.end());
            } // if

            if(vit != v2co_.end()) {
              if(vit->second != co) {
                const auto [pr, li] = pmap().invert(vit->second);
                pc.coloring.ghost.emplace_back(
                  ghost_entity{v, pr, Color(li), vit->second});
              } // if
            }
            else {
              // The ghost is remote: add to remote requests.
              remote.emplace_back(v);
              vg[co].insert(v);
            } // if
          } // for
        } // for
      } // if

      for(auto s : shared) {
        pc.coloring.shared.push_back(
          {s, {dependents.at(s).begin(), dependents.at(s).end()}});
      }

      for(auto e : primary.exclusive) {
        for(auto v : cnns.e2v[cnns.m2p.at(e)]) {
          if(!shared.count(v)) {
            pc.coloring.exclusive.emplace_back(v);
          }
          pc.coloring.owned.emplace_back(v);
        } // for
      } // for

      util::force_unique(pc.coloring.owned);
      util::force_unique(pc.coloring.exclusive);
      util::force_unique(pc.coloring.shared);
    } // for
  } // scope

  util::force_unique(remote);

  /*
    Request the migrated owners for remote vertex requests.
   */

  auto owners =
    request_owners(remote, num_vertices(), cd_.colors, vertex_raw_, comm_);

  std::vector<std::vector<util::gid>> request(size_);
  {
    std::size_t vid{0};
    for(auto v : remote) {
      request[owners[vid++]].emplace_back(v);
    } // for
  }

  auto requested = util::mpi::all_to_allv(
    [&request](int r, int) -> auto & { return request[r]; }, comm_);

  /*
    Fulfill requests from other ranks for our vertex information.
   */

  std::vector<std::vector<Color>> fulfill;
  for(auto rv : requested) {
    auto & f = fulfill.emplace_back();
    f.reserve(rv.size());
    for(auto const & v : rv) {
      f.push_back(v2co_.at(v));
    } // for
  } // for

  auto fulfilled = util::mpi::all_to_allv(
    [&fulfill](int r, int) -> auto & { return fulfill[r]; }, comm_);

  /*
    Update our local information.
   */

  std::map<std::size_t, std::size_t> m2pv;
  std::vector<std::size_t> p2mv;

  std::size_t ri{0};
  for(auto rv : request) {
    std::size_t vid{0};
    for(auto v : rv) {
      v2co_.try_emplace(v, fulfilled[ri][vid]);
      m2pv[v] = vid;
      p2mv.emplace_back(v);
      ++vid;
    } // for
    ++ri;
  } // for

  /*
    Finish populating the vertex index coloring.
   */

  std::vector<std::vector<Color>> is_peers(primaries().size());
  {
    for(const Color co : ours()) {
      const Color lco = lc(co);
      auto & pc = vertex_coloring()[lco];

      pc.coloring.all.reserve(
        pc.coloring.owned.size() + pc.coloring.ghost.size());
      pc.coloring.all.insert(pc.coloring.all.begin(),
        pc.coloring.owned.begin(),
        pc.coloring.owned.end());

      if(cd_.colors > 1) {
        auto & cp = color_peers_[lco];
        std::set<Color> peers;

        // Add ghosts that were available from the local process.
        for(auto v : pc.coloring.ghost) {
          pc.coloring.all.emplace_back(v.id);
          cp.insert(v.global);
        } // for
        for(const auto & v : pc.coloring.shared)
          peers.insert(v.dependents.begin(), v.dependents.end());

        if(size_ > 1) { // only necessary if there are multiple processes.
          // Add requested ghosts.
          auto it = vg.find(co);
          if(it != vg.end()) {
            for(auto v : it->second) {
              const auto gco = v2co_.at(v);
              cp.insert(gco);
              const auto [pr, li] = pmap().invert(gco);
              pc.coloring.ghost.emplace_back(
                ghost_entity{v, pr, Color(li), gco});
              pc.coloring.all.emplace_back(v);
            } // for
          } // if
        } // if

        cp.insert(peers.begin(), peers.end());
        is_peers[lco].resize(peers.size());
        std::copy(peers.begin(), peers.end(), is_peers[lco].begin());
        pc.peers.resize(peers.size());
        std::copy(peers.begin(), peers.end(), pc.peers.begin());
      } // if

      util::force_unique(pc.coloring.all);
      util::force_unique(pc.coloring.ghost);

      if(auxs_.size()) {
        all[lco][cd_.vid.idx].insert(
          pc.coloring.all.begin(), pc.coloring.all.end());
      }

      vertex_partitions().emplace_back(pc.coloring.all.size());
    } // for
  } // scope

  /*
    Gather the tight peer information for the vertices.
   */

  vertex_peers().reserve(cd_.colors);

  for(auto vp : util::mpi::all_gatherv(is_peers, comm_)) {
    for(auto & pc : vp) { /* over process colors */
      vertex_peers().push_back(std::move(pc));
    } // for
  } // for

  /*
    Gather partition sizes for vertices.
   */
  concatenate(vertex_partitions(), cd_.colors, comm_);
} // close_vertices

template<typename MD>
void
coloring_utils<MD>::build_auxiliary(entity_kind kind, heuristic h) {
  auto const & cnns = primary_connectivity_state();

  /*
    Add primaries that will be used to build the intermediaries and populate
    the forward map.
   */

  util::crs i_p2v; /* intermediary primary-to-vertex */

  // Start with the remote ghosts from the primaries (input).
  std::vector<util::gid> i_p2m(rghost_.begin(), rghost_.end());

  // Add owned primaries for all local colors.
  for(const auto & lco : primary_coloring()) {
    for(auto & e : lco.coloring.owned) {
      i_p2m.emplace_back(e);
    } // for
  } // for

  util::force_unique(i_p2m);

  /*
    Populate the reverse map.
   */

  std::map<util::gid, util::id> i_m2p;
  {
    util::id eid{0};
    for(auto const & e : i_p2m) {
      i_p2v.add_row(cnns.e2v[cnns.m2p.at(e)]);
      i_m2p[e] = eid++;
    } // for
  } // scope

  /*
    Build the local (to this process) intermediaries and populate a
    reverse map.
   */

  build_intermediary(kind, i_p2v, i_p2m);

  auto & aux = auxiliary_state(kind);
  {
    std::vector<std::vector<util::gid>> i2e(aux.i2v.size());
    std::size_t eid{0};
    for(auto e : aux.e2i) {
      for(auto in : e) {
        i2e[in].emplace_back(i_p2m.at(eid));
      } // for
      ++eid;
    } // for

    // Save intermediary to primary connections for building connectivity.
    for(std::size_t in{0}; in < aux.i2v.size(); ++in) {
      aux.i2e.add_row(i2e.at(in));
    } // for
  } // scope

  /*
    This section does the color assignment and counts how many auxiliary
    entities we have on the local process. We then do an all_gatherv
    to figure out the starting global id.
   */

  std::size_t pcnt{0};
  for(const auto gco : ours()) {
    for(auto e : primary_coloring()[lc(gco)].coloring.all) {
      bool halo{false};

      // Entity to connected intermediaries.
      for(auto in : aux.e2i[i_m2p.at(e)] /* util::crs */) {
        Color co = std::numeric_limits<Color>::max();

        switch(h) {
          case vertex:
            for(auto iv : aux.i2v[in]) {
              co = std::min(v2co_.at(iv), co);
            }
            break;
          case cell:
            for(auto ie : aux.i2e[in]) {
              co = std::min(p2co_.at(ie), co);
            } // for
            break;
        } // switch

        for(auto ie : aux.i2e[in]) {
          halo = /* greedy, but just used to construct a lookup table */
            halo || shared_has(gco, ie) || ghost[lc(gco)].count(ie);
        } // for

        auto const [it, add] =
          aux.a2co.try_emplace(in, std::make_pair(co, halo));

        if(add) {
          if(ours(co)) {
            // Add this auxiliary to the count if we haven't already seen it.
            ++pcnt;
          }
          else {
            // Keep track of remote ghost auxiliaries so that we can request
            // them (by vertices).
            auto v = util::to_vector(aux.i2v[in]);
            aux.ghost.try_emplace(in, gco, std::move(v));
          } // if
        } // if

        // Keep track of ghost and shared between local colors (on process).
        if(ours(co) && co != gco) {
          // co -> owning color
          // in -> auxiliary
          // gco -> dependent color

          aux.ldependents[co][in].insert(gco);
          aux.ldependencies[gco].try_emplace(in, co);
        } // if
      } // for
    } // for
  } // for

  // Figure out our starting offset.
  // Offsets are per-process, not per-color.
  std::size_t offset{0};
  {
    std::size_t r{0};
    std::size_t entities{0};
    for(auto c : util::mpi::all_gatherv(pcnt, comm_)) {
      entities += c;
      if(r++ < std::size_t(rank_)) {
        offset += c;
      }
    } // for

    aux.entities = entities;
  } // scope

  /*
    Assign global ids to the auxiliaries that belong to this process.
   */

  aux.shared.resize(primaries().size());
  for(auto && [lid, info] /* local id, color info */ : aux.a2co) {
    auto && [co, halo] = info; /* global color, boolean: primary is halo */

    if(ours(co)) {
      if(halo) { /* potentially shared, save for fulfill */
        // These are sorted for matching with off-process requesters.
        auto v = util::to_vector(aux.i2v[lid]);
        util::force_unique(v);
        aux.shared[lc(co)].try_emplace(std::move(v), lid, offset);
      } // if

      aux.l2g[lid] = offset++;
    } // if
  } // for

  // Add this id to the set of built auxiliaries
  built_.insert(kind);
} // build_auxiliary

template<typename MD>
void
coloring_utils<MD>::color_auxiliary(entity_kind kind) {
  auto & aux = auxiliary_state(kind);

  /*
    These next two loops add color-to-color dependencies for this process.
    The local intermediary ids (local offsets) are mapped to global ids,
    so the aux.{dependents/dependencies} are with respect to global ids.
   */

  for(auto const & [oco, deps] : aux.ldependents) {
    for(auto const & [in, dco] : deps) {
      aux.dependents[oco][aux.l2g.at(in)].insert(dco.begin(), dco.end());
    } // for
  } // for
  aux.ldependents.clear(); // done with this

  for(auto const & [lco, m] : aux.ldependencies) {
    for(auto const & [in, oco] : m) {
      aux.dependencies[oco].try_emplace(aux.l2g.at(in), lco);
    } // for
  } // for
  aux.ldependencies.clear(); // done with this

  /*
    This section exchanges global id information for the newly created
    auxiliaries.
   */

  std::vector<std::vector<util::id>> lids(size_);
  std::vector<std::vector<std::tuple<Color /* owning color */,
    Color /* requesting color */,
    std::vector<util::gid>>>>
    request(size_);
  for(auto const & [in, info] : aux.ghost) {
    auto const & [rco, def] = info;

    // Get process that owns this ghost and add to requests.
    auto const pr{pmap().bin(aux.a2co.at(in).first)};
    request[pr].emplace_back(std::make_tuple(aux.a2co.at(in).first, rco, def));

    // Keep track of local ids on each process.
    lids[pr].emplace_back(in);
  } // for

  auto requested = util::mpi::all_to_allv(
    [&request](int r, int) -> auto & { return request[r]; }, comm_);

  // Fulfill requested information.
  std::vector<std::vector<
    std::tuple<util::gid, std::vector<util::gid>, std::vector<util::gid>>>>
    fulfill(size_);
  std::size_t pr{0};
  for(auto & rv : requested) {
    for(auto & [oco, rco, def] : rv) {
      // Sort the vertices for match (aux.shared is already sorted.)
      util::force_unique(def);
      auto it = aux.shared[lc(oco)].find(def);
      flog_assert(
        it != aux.shared[lc(oco)].end(), "invalid auxiliary definition");

      // FIXME: Remove unused information, i.e., aux.i2e

      // Fulfillment sends the unsorted order to the requester.
      fulfill[pr].emplace_back(std::make_tuple(it->second.second,
        util::to_vector(aux.i2v[it->second.first]),
        util::to_vector(aux.i2e[it->second.first])));

      // Add the requesting color and auxiliary to things that depend on us.
      aux.dependents[oco][it->second.second].insert(rco);
    } // for

    ++pr;
  } // for

  auto fulfilled = util::mpi::all_to_allv(
    [&fulfill](int r, int) -> auto & { return fulfill[r]; }, comm_);

  // This really just updates our local-to-global id map with the remote
  // entities that we just requested.
  pr = 0;
  for(auto & fv : fulfilled) {
    std::size_t off{0};
    for(auto const & ff : fv) {
      aux.l2g.try_emplace(lids[pr][off], std::get<0>(ff));
      ++off;
    } // for

    ++pr;
  } // for

  // Add the remote dependency information that we got from our requests.
  // We now have complete information for each local color on what global
  // auxiliary entities they depend on and what color owns that auxliary.
  for(auto const & [in, info] : aux.ghost) {
    auto const & [rco, def] = info;
    aux.dependencies[aux.a2co.at(in).first].try_emplace(aux.l2g.at(in), rco);
  } // for
} // color_auxiliary

template<typename MD>
void
coloring_utils<MD>::close_auxiliary(entity_kind kind, std::size_t idx) {
  auto & aux = auxiliary_state(kind);
  auto & ai = auxiliary_coloring(idx);

  for(const Color c : ours()) {
    auto & a1 = ai[lc(c)];
    a1.color = c;
    a1.entities = num_entities(kind);
  } // for

  /*
    Populate the coloring information.
   */

  std::map<Color, std::set<Color>> peers;
  for(auto [lid, gid] : aux.l2g) {
    for(const Color gco : ours()) {
      const Color lco = lc(gco);
      auto const co = aux.a2co.at(lid).first;

      bool const ownd = (gco == co);
      bool const ghst = !ownd && aux.dependencies[co].count(gid) &&
                        aux.dependencies[co][gid] == gco;

      auto & pc = ai[lco];

      if(ownd || ghst) {
        pc.coloring.all.emplace_back(gid);
      } // if

      if(ownd) {
        pc.coloring.owned.emplace_back(gid);
        pc.cnx_colorings[cd_.cid.idx].add_row(aux.i2e[lid]);
        pc.cnx_colorings[cd_.vid.idx].add_row(aux.i2v[lid]);

        if(aux.dependents[co].count(gid)) {
          auto const & deps = aux.dependents.at(co).at(gid);
          pc.coloring.shared.push_back({gid, {deps.begin(), deps.end()}});
          peers[lco].insert(deps.begin(), deps.end());
        }
        else {
          pc.coloring.exclusive.emplace_back(gid);
        }
      } // if

      if(ghst) {
        const auto [pr, li] = pmap().invert(co);
        pc.coloring.ghost.push_back({gid, pr, Color(li), co});

        // Only add auxiliary connectivity that is covered by
        // the primary closure.
        std::vector<util::gid> pall;
        auto && pclo = all[lco].at(cd_.cid.idx);
        for(auto e : aux.i2e[lid]) {
          if(pclo.count(e)) {
            pall.push_back(e);
          } // if
        } // for

        if(pall.size()) {
          pc.cnx_colorings[cd_.cid.idx].add_row(pall);
        }

        pc.cnx_colorings[cd_.vid.idx].add_row(aux.i2v[lid]);
        color_peers_[lco].insert(co);
      } // if
    } // for
  } // for

  /*
    Populate peer information.
   */

  std::vector<std::size_t> & partitions = coloring_.partitions[idx];
  std::vector<std::vector<Color>> is_peers(primaries().size());
  for(std::size_t lco{0}; lco < primaries().size(); ++lco) {
    auto & pc = ai[lco];
    partitions.emplace_back(pc.coloring.all.size());
    if(peers.count(lco)) {
      color_peers_[lco].insert(peers.at(lco).begin(), peers.at(lco).end());
      is_peers[lco].resize(peers.at(lco).size());
      std::copy(
        peers.at(lco).begin(), peers.at(lco).end(), is_peers[lco].begin());
      pc.peers.resize(peers.at(lco).size());
      std::copy(peers.at(lco).begin(), peers.at(lco).end(), pc.peers.begin());
    } // if
  } // for

  /*
    Gather the tight peer information for this auxiliary entity type.
   */

  {
    auto & peers = coloring_.peers[idx];
    peers.reserve(cd_.colors);

    for(auto vp : util::mpi::all_gatherv(is_peers, comm_)) {
      for(auto & pc : vp) { /* over process colors */
        peers.push_back(std::move(pc));
      } // for
    } // for
  } // scope

  /*
    Gather partition sizes for entities.
   */

  concatenate(partitions, cd_.colors, comm_);
} // close_auxiliary

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
  c2e.values.reserve(c2f.values.size() + f2e.values.size());

  for(const util::crs::span cell : c2f) {
    std::vector<util::gid> edges;

    // accumulate edges in cell
    for(const std::size_t face : cell) {
      for(const std::size_t ei : f2e.offsets[face]) {
        auto it = std::find(edges.begin(), edges.end(), ei);
        if(it == edges.end()) {
          edges.push_back(ei);
        }
      }
    }
    c2e.add_row(edges);
  }

  return c2e;
} // intersect_connectivity

/*!
  Build intermediary entities locally from cell to vertex graph.

  \param kind The mesh definition entity kind.
  @param e2v entity to vertex graph.
  @param p2m Process-to-mesh map for the primary entities.
*/

template<class MD>
void
coloring_utils<MD>::build_intermediary(entity_kind kind,
  const util::crs & e2v,
  std::vector<util::gid> const & p2m) {
  auto & aux = auxiliary_state(kind);

  // temporary storage
  util::crs edges;
  std::vector<util::gid> sorted;
  std::vector<util::gid> these_edges;

  // iterate over primaries, adding all of their edges to the table
  std::size_t entity{0};
  for(auto const & e : e2v) {
    auto const & these_verts = util::to_vector(e);
    these_edges.clear();

    // build the edges for the cell
    edges.offsets.clear();
    edges.values.clear();
    if(MD::dimension() == cd_.cid.kind)
      md_.make_entity(kind, p2m[entity++], these_verts, edges);
    else
      md_.make_entity(kind, 0, these_verts, edges);

    for(const util::crs::span row : edges) {
      sorted.assign(row.begin(), row.end());
      std::sort(sorted.begin(), sorted.end());

      const auto eid = aux.v2i.size();
      if(const auto vit = aux.v2i.try_emplace(sorted, eid); vit.second) {
        aux.i2v.add_row(row);
        these_edges.push_back(eid);
      }
      else {
        these_edges.push_back(vit.first->second);
      } // if
    } // for

    aux.e2i.add_row(these_edges);
  } // for
} // build_intermediary

template<typename MD>
auto &
coloring_utils<MD>::generate() {
  // Gather peer information;
  auto & cp = coloring_.color_peers;
  cp.reserve(color_peers_.size());
  for(const auto & s : color_peers_)
    cp.push_back(s.size());
  concatenate(cp, cd_.colors, comm_);

  // Set the connectivity allocation sizes.
  for(auto [from, to, transpose] : ca_) {
    for(std::size_t lco{0}; lco < primaries().size(); ++lco) {
      coloring_.idx_spaces[from][lco].cnx_allocs[to] =
        coloring_.idx_spaces[transpose ? to : from][lco]
          .cnx_colorings[transpose ? from : to]
          .values.size();
    } // for
  } // for

  return coloring_;
} // generate

template<typename M, typename T>
inline std::vector<T>
find_field_color(const M & map,
  const std::vector<std::vector<std::pair<std::size_t, T>>> & l,
  Color c) {
  std::vector<T> res;
  for(const auto & lc : l) {
    for(const auto & v : lc) {
      if(map.at(v.first) == c)
        res.push_back(v.second);
    } // for
  } // for
  return res;
} // find_field_color

template<class MD>
template<class T>
std::vector<std::vector<T>>
coloring_utils<MD>::send_field(entity_kind k, const std::vector<T> & f) {
  flog_assert(idmap_.find(k) != idmap_.end(), "Invalid kind");
  flog_assert(
    k < 2, "Invalid kind, only primaries (0) and vertices (1) supported ");

  // local id and type pairing
  using l2t = std::pair<std::size_t, T>;

  const util::equal_map em(num_entities(k), size_);
  auto entities = util::mpi::one_to_allv(pack_field(em, f), comm_);

  std::vector<std::vector<l2t>> locals = util::mpi::all_to_allv(
    move_field(
      em.size(), cd_.colors, k == 0 ? vertex_raw_ : primary_raw_, entities),
    comm_);

  std::vector<std::vector<T>> res;

  for(auto i : ours())
    res.emplace_back(find_field_color(k == 0 ? v2co_ : p2co_, locals, i));

  return res;
} // send_field

/// \}
} // namespace unstructured_impl
} // namespace topo
} // namespace flecsi

/// \endcond

#endif
