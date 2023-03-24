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
  /// Mapping of primary entity kind to its entity index in \c index_spaces.
  /// \warning Not an \c index_space enumerator \b value.
  index_map cid;
  /// Number of layers of ghosts needed.
  std::size_t depth;
  /// Mapping of vertices entity kind to its entity index in \c index_spaces.
  index_map vid;
  /// Mapping of auxiliary entity kinds to their indices in \c index_spaces.
  std::vector<index_map> aidxs;
};

#ifdef DOXYGEN
/// An example mesh definition that is not really implemented.
struct mesh_definition {
  /// Get the dimensionality of the mesh.
  static constexpr Dimension dimension();

  /// Get the global number of entities of a kind.
  util::gid num_entities(entity_kind) const;

  /// Get the entities connected to or associated with an entity.
  /// \param id of entity kind \a from
  /// \return ids of entity kind \a to
  std::vector<std::size_t>
  entities(entity_kind from, entity_kind to, util::gid id) const;

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
    : md_(*md), cd_(cd), ca_(ca), comm_(comm), rank_(util::mpi::rank(comm_)),
      size_(util::mpi::size(comm_)) {

    // Create a map from mesh definition entity kinds to local offsets.
    idmap_[cd.cid.kind] = 0;
    idmap_[cd.vid.kind] = 1;

    for(std::size_t i{0}; i < cd.aidxs.size(); ++i) {
      idmap_[cd.aidxs[i].kind] = i + aoff;
    } // for

    auxs_.resize(cd.aidxs.size());

    coloring_.idx_spaces.resize(idmap_.size());
    connectivity_.resize(idmap_.size());
  } // coloring_utils

  /// Color the primary entity type using the provided coloring function.
  /// \param shared maximum number of shared vertices to disregard
  /// \param c coloring function object with the signature of
  ///          flecsi::util::parmetis::color
  //
  /// \note This method first marshalls the naive partition information on the
  /// root process and sends the respective data to each initially-owning
  /// process. The current implementation is designed to avoid having many
  /// processes hit the file system at once. This may not be optimal on all
  /// parallel file systems.
  //
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

  std::vector<util::crs> get_connectivity(entity_kind from, entity_kind to);

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
    return primary_conns_;
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
    return idmap_.at(kind) >= aoff ? auxiliary_state(kind).entities
                                   : md_.num_entities(kind);
  }

  auto & coloring(std::size_t idx) {
    return coloring_.idx_spaces[idx].colors;
  }

  auto & connectivity(std::size_t idx) {
    return connectivity_[idx];
  }

  auto cell_index() const {
    return cd_.cid.idx;
  }

  auto vertex_index() const {
    return cd_.vid.idx;
  }

  auto & entities(std::size_t idx) {
    return coloring_.idx_spaces[idx].entities;
  }

  auto & peers(std::size_t idx) {
    return coloring_.idx_spaces[idx].peers;
  }

  auto & partitions(std::size_t idx) {
    return coloring_.idx_spaces[idx].partitions;
  }

  auto & auxiliary_state(entity_kind kind) {
    return auxs_[idmap_.at(kind) - aoff];
  }

  std::vector<Color> request_owners(const std::vector<util::gid> &,
    util::gid n,
    const std::vector<Color> &) const;

  void compute_interval_sizes(std::size_t idx,
    std::vector<std::map<util::gid, util::id>> & ghost_offsets) {
    /*
     * Compute local intervals.
     */

    std::vector<std::size_t> local_itvls(ours().size());
    for(std::size_t lc{0}; lc < ours().size(); ++lc) {
      std::size_t nlocal = 0;
      auto g = ghost_offsets[lc].begin();

      std::size_t begin = g == ghost_offsets[lc].end() ? 0 : g->second, run = 0;
      for(; g != ghost_offsets[lc].end(); ++g) {
        if(!run || g->second != begin + run) {
          if(run) {
            begin = g->second;
            nlocal++;
          }
          run = 1;
        }
        else {
          ++run;
        }
      } // for

      local_itvls[lc] = ++nlocal;
    } // for

    /*
     * Gather global interval sizes.
     */

    auto global_itvls = util::mpi::all_gatherv(local_itvls, comm_);
    auto & nis = coloring_.idx_spaces[idx].num_intervals;
    nis.reserve(coloring_.colors);

    for(const auto & p : global_itvls) {
      nis.insert(nis.end(), p.begin(), p.end());
    }
  }

  struct connectivity_state_t {
    util::crs e2v;
    std::map<util::gid, std::vector<util::gid>> v2e;
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
  int rank_, size_;
  std::map<std::size_t, std::size_t> idmap_;

  struct process_primary_color_data {
    std::vector<util::gid> all;
  };

  struct process_color_data : process_primary_color_data {
    std::vector<util::gid> owned;
    std::vector<util::gid> ghost;
    std::set<util::gid> shared;
    std::unordered_map<std::size_t, std::set<std::size_t>> dependents;
  };

  std::vector</* over index spaces */
    std::vector</* over local colors */
      std::vector<util::crs>>>
    connectivity_;

  std::vector<process_primary_color_data> primary_pcdata;
  std::vector<process_color_data> vertex_pcdata;
  std::map<entity_kind, std::vector<process_color_data>> aux_pcdata;

  // Connectivity state is associated with primary entity's kind (as opposed to
  // its index space).
  connectivity_state_t primary_conns_;
  util::crs naive;

  std::vector<Color> primary_raw_;
  std::vector<Color> vertex_raw_;
  std::vector<std::vector<util::gid>> primaries_;
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
}; // struct coloring_utils

template<typename MD>
template<typename C>
void
coloring_utils<MD>::color_primaries(util::id shared, C && f) {
  auto & cnns = primary_connectivity_state();

  /*
    Get the initial entities for this rank. The entities will be read by
    the root process and sent to each initially "owning" rank using
    a naive distribution.
   */
  const util::equal_map ecm(num_primaries(), size_);
  cnns.e2v =
    util::mpi::one_to_allv(pack_definitions(md_, cd_.cid.kind, ecm), comm_);

  /*
    Create a map of vertex-to-entity connectivity information from
    the initial entity distribution.
   */

  // Populate local vertex connectivity information
  std::size_t offset = ecm(rank_);

  for(const auto & c : cnns.e2v) {
    for(auto v : c) {
      cnns.v2e[v].emplace_back(offset);
    } // for
    ++offset;
  } // for

  // Request all referencers of our connected vertices
  const util::equal_map vm(num_vertices(), size_);
  auto referencers =
    util::mpi::all_to_allv(vertex_referencers(cnns.v2e, vm, rank_), comm_);

  /*
    Update our local connectivity information. We now have all
    vertex-to-entity connectivity information for the naive distribution
    of entities that we own.
   */

  for(const auto & r : referencers) {
    for(const auto & v : r) {
      for(auto c : v.second) {
        cnns.v2e[v.first].emplace_back(c);
      } // for
    } // for
  } // for

  // Remove duplicate referencers
  util::unique_each(cnns.v2e);

  /*
    Invert the vertex referencer information, i.e., find the entities
    that are referenced by our connected vertices.
   */

  std::vector<std::vector<util::gid>> referencer_inverse(size_);

  for(const auto & v : cnns.v2e) {
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

  for(const auto & r : connectivity) {
    for(const auto & v : r) {
      for(auto c : v.second) {
        cnns.v2e[v.first].emplace_back(c);
      } // for
    } // for
  } // for

  // Remove duplicate referencers
  util::unique_each(cnns.v2e);

  /*
    Fill in the entity-to-entity connectivity that have "shared" vertices
    in common.
   */
  std::map<util::gid, std::vector<util::gid>> e2e;

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
        e2e[c].emplace_back(tc.first);
        e2e[tc.first].emplace_back(c);
      } // if
    } // for

    ++c;
  } // for

  // Remove duplicate connections
  util::unique_each(e2e);

  /*
    Populate the actual distributed crs data structure.
   */

  for(const std::size_t c : ecm[rank_]) {
    naive.add_row(e2e[c]);
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
      rank_),
    comm_);

  primaries_.resize(ours().size());
  ghost.resize(ours().size());

  for(auto const & [entity_pack, v2e_pack] : migrated) {
    for(auto const & [info, vertices] : entity_pack) {
      auto const [co, eid] = info;
      cnns.e2v.add_row(vertices);
      cnns.m2p[eid] = cnns.e2v.size() - 1; /* offset map */
      cnns.p2m.emplace_back(eid);
      primaries()[lc(co)].emplace_back(eid);
    } // for

    // vertex-to-entity connectivity
    for(auto const & [v, entities] : v2e_pack) {
      cnns.v2e.try_emplace(v, entities);
    } // for
  } // for

  color_peers_.resize(ours().size());
  coloring(cell_index()).resize(ours().size());
  coloring(vertex_index()).resize(ours().size());
  connectivity(cell_index()).resize(ours().size());
  connectivity(vertex_index()).resize(ours().size());
  for(auto const [kind, idx] : cd_.aidxs) {
    coloring(idx).resize(ours().size());
    connectivity(idx).resize(ours().size());
  } // for

  // Set the vector sizes for connectivity information.
  for(auto [from, to, transpose] : ca_) {
    for(auto & ic : coloring(from)) {
      ic.cnx_allocs.resize(idmap_.size());
    }

    for(auto & cnx : connectivity(from)) {
      cnx.resize(idmap_.size());
    }
  } // for
} // migrate_primaries

/// Request owner information about the given entity list from the naive
/// owners.
/// @param request  The global ids of the desired entities.
/// @param ne       The global number of entities.
/// @param idx_cos  The naive coloring of the entities.
/// \return the owning process for each entity
template<typename MD>
std::vector<Color>
coloring_utils<MD>::request_owners(std::vector<util::gid> const & request,
  util::gid ne,
  std::vector<Color> const & idx_cos) const {
  std::vector<std::vector<util::gid>> requests(size_);
  const util::equal_map pm(ne, size_);
  for(auto e : request) {
    requests[pm.bin(e)].emplace_back(e);
  } // for

  auto requested = util::mpi::all_to_allv(
    [&requests](int r, int) -> auto & { return requests[r]; }, comm_);

  /*
    Fulfill naive-owner requests with migrated owners.
   */

  std::vector<std::vector<util::gid>> fulfill(size_);
  {
    const std::size_t start = pm(rank_);
    Color r = 0;
    for(auto rv : requested) {
      for(auto e : rv) {
        fulfill[r].emplace_back(pmap().bin(idx_cos[e - start]));
      } // for
      ++r;
    } // for
  } // scope

  auto fulfilled = util::mpi::all_to_allv(
    [&fulfill](int r, int) -> auto & { return fulfill[r]; }, comm_);

  std::vector<std::size_t> offs(size_);
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
    for(auto const & entities : primaries()) {
      auto & v = wkset.emplace_back();
      v.reserve(entities.size());
      for(auto e : entities) {
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
            if(!cnns.m2p.count(en)) {
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

      auto owners = request_owners(layer, num_primaries(), primary_raw_);

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
        fulfill, dependents, p2co_, cnns.e2v, cnns.v2e, cnns.m2p),
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
        cnns.p2m.emplace_back(id);
        flog_assert(cnns.p2m.size() == cnns.e2v.size(), "local info corrupted");
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
    } // for
  } // for

  util::force_unique(rghost_);

  /*
    Primary entities.
   */

  auto & primary_partitions = partitions(cell_index());
  std::vector<std::vector<Color>> is_peers(ours().size());

  std::vector<std::map<util::gid, util::id>> shared_offsets(ours().size()),
    ghost_offsets(ours().size());

  std::vector</* over processes */
    std::vector</* over local colors */
      std::vector<std::tuple<util::gid,
        std::size_t /* local color */,
        std::size_t /* global color */>>>>
    sources(size_);

  std::vector<std::vector<std::size_t>> sources_lco(size_);
  primary_pcdata.resize(ours().size());

  {
    auto co = ours().begin();
    for(const auto & entities : primaries()) {
      const Color lco = lc(*co);
      auto & gall = primary_pcdata[lco].all;
      gall.reserve(entities.size());

      for(auto e : entities) {
        gall.push_back(e);
      }

      for(auto e : ghost[lco]) {
        const auto gco = p2co_.at(e);
        const auto [pr, li] = pmap().invert(gco);
        gall.emplace_back(e);

        if(sources[pr].size() == 0) {
          sources[pr].resize(ours().size());
        }

        auto const s = std::make_tuple(e, Color(li), gco);
        sources[pr][lco].emplace_back(s);
        sources_lco[pr].emplace_back(lco);
      } // for

      util::force_unique(gall);

      /*
        Create a lookup table for local ghost offsets.
       */

      for(auto e : ghost[lco]) {
        auto it = std::find(gall.begin(), gall.end(), e);
        flog_assert(it != gall.end(), "ghost entity doesn't exist");
        ghost_offsets[lco][e] = std::distance(gall.begin(), it);
      } // for

      /*
        We also need to add shared offsets to the lookup table so that we can
        provide local shared offset information to other processes that
        request it.
       */

      for(auto e : entities) {
        if(shared_has(*co, e)) {
          auto it = std::find(gall.begin(), gall.end(), e);
          flog_assert(it != gall.end(), "shared entity doesn't exist");
          auto offset = std::distance(gall.begin(), it);
          shared_offsets[lco][e] = offset;
        }
      } // for

      primary_partitions.emplace_back(gall.size());

      ++co;
    } // for

    /*
      Send/Receive requests for shared offsets with other processes.
     */

    auto requested = util::mpi::all_to_allv(
      [&sources](int r, int) -> auto & { return sources[r]; }, comm_);

    /*
      Fulfill the requests that we received from other processes, i.e.,
      provide the local offset for the requested shared mesh ids.
     */

    std::vector<std::vector<util::id>> fulfills(size_);
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
      [f = std::move(fulfills)](int r, int) { return std::move(f[r]); }, comm_);

    struct ghost_info {
      util::gid gid;
      std::size_t lco;
      util::id lid;
      util::id rid;
    };

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
              ginfo = {in_id, in_lco, ghost_offsets[pc][id], fulfilled[k][cnt]};
              return;
            }
            ++cnt;
          } // for
          ++pc;
        } // for
        ++k;
      } // for
      flog_fatal("could not find ghost info");
    };

    entities(cell_index()) = num_primaries();

    co = ours().begin();
    for(const auto & entities : primaries()) {
      const Color lco = lc(*co);
      auto & ic = coloring(cell_index())[lco];
      auto & gall = primary_pcdata[lco].all;
      ic.entities = gall.size();

      for(auto e : entities) {
        if(shared_has(*co, e)) {
          auto it = std::find(gall.begin(), gall.end(), e);
          const auto lid = std::distance(gall.begin(), it);

          for(auto d : dependents.at(e)) {
            ic.peer_shared[d].insert(lid);
          }
        }
      } // for

      auto & cp = color_peers_[lco];
      for(auto e : ghost[lco]) {
        const auto gco = p2co_.at(e);
        std::size_t gcolor;
        ghost_info ginfo;
        find_in_source(e, lco, gcolor, ginfo);
        flog_assert(gcolor == gco, "global color mismatch");
        ic.ghost.emplace_back(ghost_entity{ginfo.lid, ginfo.rid, gco});
        cp.insert(gco);
      } // for
      std::set<Color> peers;
      for(auto e : shared_[*co])
        peers.insert(dependents.at(e).begin(), dependents.at(e).end());
      cp.insert(peers.begin(), peers.end());
      is_peers[lco].assign(peers.begin(), peers.end());

      ++co;
    } // for

  } // scope

  /*
    Gather the tight peer information for the primary entity type.
   */

  {
    auto & primary_peers = peers(cell_index());
    primary_peers.reserve(cd_.colors);

    for(auto vp : util::mpi::all_gatherv(is_peers, comm_)) {
      for(auto & pc : vp) { /* over process colors */
        primary_peers.push_back(std::move(pc));
      } // for
    } // for
  } // scope

  coloring_.colors = cd_.colors;

  /*
    Gather partition sizes for entities.
   */
  concatenate(primary_partitions, cd_.colors, comm_);

  /*
    Compute primary ghost interval sizes
   */
  compute_interval_sizes(cell_index(), ghost_offsets);

} // close_primaries

template<typename MD>
void
coloring_utils<MD>::color_vertices() {
  auto & cnns = primary_connectivity_state();

  for(std::size_t lco{0}; lco < ours().size(); ++lco) {
    auto const & primary = coloring(cell_index())[lco];
    auto const & primary_pcd = primary_pcdata[lco];

    for(auto lid : primary.owned()) {
      auto egid = primary_pcd.all[lid];
      for(auto v : cnns.e2v[cnns.m2p.at(egid)]) {
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

  coloring(vertex_index()).resize(ours().size());
  std::vector<std::vector<Color>> is_peers(ours().size());
  std::vector<util::gid> remote;

  std::vector<std::map<util::gid, util::id>> shared_offsets(ours().size()),
    ghost_offsets(ours().size());

  std::vector</* over processes */
    std::vector</* over local colors */
      std::vector<std::tuple<util::gid,
        std::size_t /* local color */,
        std::size_t /* global color */>>>>
    sources(size_);

  std::vector<std::vector<std::size_t>> sources_lco(size_);

  vertex_pcdata.resize(ours().size());

  {
    for(const Color co : ours()) {
      const Color lco = lc(co);
      auto primary = coloring(cell_index())[lco];
      auto & vertex_pcd = vertex_pcdata[lco];
      auto & primary_pcd = primary_pcdata[lco];

      if(cd_.colors > 1) {
        // Go through the shared primaries and look for ghosts. Some of these
        // may be on the local processor, i.e., we don't need to request
        // remote information about them.
        for(const auto & [d, shared_entities] : primary.peer_shared) {
          for(auto lid : shared_entities) {
            auto sgid = primary_pcd.all[lid];
            for(auto v : cnns.e2v[cnns.m2p.at(sgid)]) {
              auto vit = v2co_.find(v);
              flog_assert(vit != v2co_.end(), "invalid vertex id");
              auto gco = vit->second;

              if(gco == co) {
                vertex_pcd.shared.insert(v);
                vertex_pcd.dependents[v].insert(d);
                vertex_pcd.owned.emplace_back(v);
              }
              else {
                vertex_pcd.ghost.emplace_back(v);
                if(!ours(gco)) {
                  // The ghost is remote: add to remote requests.
                  remote.emplace_back(v);
                } // if
              } // if
            } // for
          } // for
        } // for

        // Go through the ghost primaries and look for ghosts. Some of these
        // may also be on the local processor.
        for(auto e : primary.ghost) {
          auto egid = primary_pcd.all[e.lid];

          for(auto v : cnns.e2v[cnns.m2p.at(egid)]) {
            auto vit = v2co_.find(v);

            // Add dependents through ghosts.
            if(vertex_pcd.shared.count(v) && vdeps_.count(egid)) {
              auto deps = vdeps_.at(egid);
              deps.erase(co);
              vertex_pcd.dependents[v].insert(deps.begin(), deps.end());
            } // if

            if(vit != v2co_.end()) {
              auto gco = vit->second;
              if(gco != co) {
                vertex_pcd.ghost.emplace_back(v);
              } // if
            }
            else {
              // The ghost is remote: add to remote requests.
              vertex_pcd.ghost.emplace_back(v);
              remote.emplace_back(v);
            } // if
          } // for
        } // for
      } // if

      for(auto lid : primary.exclusive()) {
        auto egid = primary_pcd.all[lid];

        for(auto v : cnns.e2v[cnns.m2p.at(egid)]) {
          vertex_pcd.owned.emplace_back(v);
        } // for
      } // for

      for(auto v : vertex_pcd.owned) {
        vertex_pcd.all.emplace_back(v);
      }

      for(auto v : vertex_pcd.ghost) {
        vertex_pcd.all.emplace_back(v);
      }

      util::force_unique(vertex_pcd.all);
      util::force_unique(vertex_pcd.owned);
      util::force_unique(vertex_pcd.ghost);
    } // for
  } // scope

  util::force_unique(remote);

  {
    /*
      Request the migrated owners for remote vertex requests.
     */

    auto owners = request_owners(remote, num_vertices(), vertex_raw_);

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

    std::size_t ri{0};
    for(auto rv : request) {
      std::size_t vid{0};
      for(auto v : rv) {
        v2co_.try_emplace(v, fulfilled[ri][vid]);
        ++vid;
      } // for
      ++ri;
    } // for
  }

  {
    const std::size_t num_lco = ours().size();

    for(std::size_t lco = 0; lco < num_lco; ++lco) {
      auto & vertex_pcd = vertex_pcdata[lco];

      if(vertex_pcd.ghost.size()) {
        for(auto v : vertex_pcd.ghost) {
          const auto gco = v2co_.at(v);
          const auto [pr, li] = pmap().invert(gco);

          if(sources[pr].size() == 0) {
            sources[pr].resize(ours().size());
          }

          auto const s = std::make_tuple(v, Color(li), gco);

          sources[pr][lco].emplace_back(s);
          sources_lco[pr].emplace_back(lco);
        } // for
      } // if

      /*
        Create a lookup table for local ghost offsets.
       */

      for(auto v : vertex_pcd.ghost) {
        auto it = std::find(vertex_pcd.all.begin(), vertex_pcd.all.end(), v);
        flog_assert(it != vertex_pcd.all.end(), "ghost entity doesn't exist");
        ghost_offsets[lco][v] = std::distance(vertex_pcd.all.begin(), it);
      } // for

      /*
        We also need to add shared offsets to the lookup table so that we can
        provide local shared offset information to other processes that
        request it.
       */

      for(auto v : vertex_pcd.shared) {
        auto it = std::find(vertex_pcd.all.begin(), vertex_pcd.all.end(), v);
        flog_assert(it != vertex_pcd.all.end(), "shared entity doesn't exist");
        auto offset = std::distance(vertex_pcd.all.begin(), it);
        shared_offsets[lco][v] = offset;
      } // for

    } // for
  } // scope

  /*
    Send/Receive requests for shared offsets with other processes.
   */

  auto requested = util::mpi::all_to_allv(
    [&sources](int r, int) -> auto & { return sources[r]; }, comm_);

  /*
    Fulfill the requests that we received from other processes, i.e.,
    provide the local offset for the requested shared mesh ids.
   */

  std::vector<std::vector<util::id>> fulfills(size_);
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
    [f = std::move(fulfills)](int r, int) { return std::move(f[r]); }, comm_);

  struct ghost_info {
    util::gid gid;
    std::size_t lco;
    util::id lid;
    util::id rid;
  };

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
            ginfo = {in_id, in_lco, ghost_offsets[pc][id], fulfilled[k][cnt]};
            return;
          }
          ++cnt;
        } // for
        ++pc;
      } // for
      ++k;
    } // for
    flog_fatal("could not find ghost info");
  };

  /*
    Finish populating the vertex index coloring.
   */
  entities(vertex_index()) = num_vertices();

  {
    for(const Color co : ours()) {
      const Color lco = lc(co);
      auto & ic = coloring(vertex_index())[lco];
      auto & vertex_pcd = vertex_pcdata[lco];

      ic.entities = vertex_pcd.all.size();

      auto & cp = color_peers_[lco];
      std::set<Color> peers;

      for(auto v : vertex_pcd.owned) {
        auto it = std::find(vertex_pcd.all.begin(), vertex_pcd.all.end(), v);
        const auto lid = std::distance(vertex_pcd.all.begin(), it);

        if(vertex_pcd.shared.size() && vertex_pcd.shared.count(v)) {
          for(auto d : vertex_pcd.dependents.at(v)) {
            ic.peer_shared[d].insert(lid);
            peers.insert(d);
          }
        }
      } // for

      if(vertex_pcd.ghost.size()) {
        for(auto v : vertex_pcd.ghost) {
          const auto gco = v2co_.at(v);
          cp.insert(gco);
          std::size_t gcolor = -1;
          ghost_info ginfo;
          find_in_source(v, lco, gcolor, ginfo);
          flog_assert(gcolor == gco, "global color mismatch");
          ic.ghost.emplace_back(ghost_entity{ginfo.lid, ginfo.rid, gco});
        } // for
      } // if

      cp.insert(peers.begin(), peers.end());
      is_peers[lco].assign(peers.begin(), peers.end());

      partitions(vertex_index()).emplace_back(vertex_pcd.all.size());
    } // for
  } // scope

  /*
    Gather the tight peer information for the vertices.
   */

  auto & vertex_peers = peers(vertex_index());
  vertex_peers.reserve(cd_.colors);

  for(auto vp : util::mpi::all_gatherv(is_peers, comm_)) {
    for(auto & pc : vp) { /* over process colors */
      vertex_peers.push_back(std::move(pc));
    } // for
  } // for

  /*
    Gather partition sizes or vertices.
   */
  concatenate(partitions(vertex_index()), cd_.colors, comm_);

  /*
   * Compute vertex ghost interval sizes
   */

  compute_interval_sizes(vertex_index(), ghost_offsets);

  /*
   * Build up connectivity
   */

  for(std::size_t lco = 0; lco < ours().size(); ++lco) {
    auto & ic = coloring(cell_index())[lco];
    auto & vertex_pcd = vertex_pcdata[lco];
    auto & primary_pcd = primary_pcdata[lco];
    auto & crs = connectivity(cell_index())[lco][vertex_index()];

    auto g2l = [&all = vertex_pcd.all](util::gid gid) -> util::id {
      auto it = std::find(all.begin(), all.end(), gid);
      flog_assert(it != all.end(), "unknown vertex");
      return std::distance(all.begin(), it);
    };

    for(auto lid : ic.owned()) {
      auto egid = primary_pcd.all[lid];
      crs.add_row(util::transform_view(cnns.e2v[cnns.m2p.at(egid)], g2l));
    } // for

    for(auto g : ic.ghost) {
      auto egid = primary_pcd.all[g.lid];
      crs.add_row(util::transform_view(cnns.e2v[cnns.m2p.at(egid)], g2l));
    } // for
  }

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
  for(std::size_t lco = 0; lco < ours().size(); ++lco) {
    auto & ic = coloring(cell_index())[lco];
    auto & primary_pcd = primary_pcdata[lco];
    for(auto & plid : ic.owned()) {
      i_p2m.emplace_back(primary_pcd.all[plid]);
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
    const auto lco = lc(gco);
    auto & primary_all = primary_pcdata[lco].all;

    for(auto e : primary_all) {
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

  aux.shared.resize(ours().size());
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
      // local, global
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
  auto & ai = coloring(idx);

  ai.resize(ours().size());

  std::vector<std::map<util::gid, util::id>> shared_offsets(ours().size()),
    ghost_offsets(ours().size());

  std::vector</* over processes */
    std::vector</* over local colors */
      std::vector<std::tuple<util::gid,
        std::size_t /* local color */,
        std::size_t /* global color */>>>>
    sources(size_);

  std::vector<std::vector<std::size_t>> sources_lco(size_);
  std::vector<std::set<util::gid>> pclo;
  std::map<Color, std::set<Color>> peers;

  pclo.resize(ours().size());

  if(auxs_.size()) {
    for(std::size_t lco = 0; lco < ours().size(); ++lco) {
      pclo[lco].insert(
        primary_pcdata[lco].all.begin(), primary_pcdata[lco].all.end());
    }
  }

  aux_pcdata[kind].resize(ours().size());

  for(auto gco : ours()) {
    auto const lco = lc(gco);
    auto & aux_pcd = aux_pcdata[kind][lco];
    auto & primary_pcd = primary_pcdata[lco];
    auto & vertex_pcd = vertex_pcdata[lco];
    auto & cnx = connectivity(idx)[lco];

    auto entity_g2l = [&all = primary_pcd.all](util::gid gid) -> util::id {
      auto it = std::find(all.begin(), all.end(), gid);
      return std::distance(all.begin(), it);
    };

    auto vertex_g2l = [&all = vertex_pcd.all](util::gid gid) -> util::id {
      auto it = std::find(all.begin(), all.end(), gid);
      flog_assert(it != all.end(), "unknown vertex");
      return std::distance(all.begin(), it);
    };

    for(auto [lid, gid] : aux.l2g) {
      auto const co = aux.a2co.at(lid).first;

      if(gco == co) {
        aux_pcd.owned.emplace_back(gid);

        cnx[cell_index()].add_row(
          util::transform_view(aux.i2e[lid], entity_g2l));
        cnx[vertex_index()].add_row(
          util::transform_view(aux.i2v[lid], vertex_g2l));

        if(aux.dependents[co].count(gid)) {
          aux_pcd.shared.insert(gid);
          auto const & deps = aux.dependents.at(co).at(gid);

          for(auto d : deps) {
            peers[lco].insert(d);
            aux_pcd.dependents[gid].insert(d);
          }
        }
      }
      else if(aux.dependencies[co].count(gid) &&
              aux.dependencies[co][gid] == gco) {
        aux_pcd.ghost.push_back(gid);
        const auto [pr, li] = pmap().invert(co);
        auto const s = std::make_tuple(gid, Color(li), co);

        if(sources[pr].size() == 0) {
          sources[pr].resize(ours().size());
        }

        sources[pr][lco].emplace_back(s);
        sources_lco[pr].emplace_back(lco);
        color_peers_[lco].insert(co);

        // Only add auxiliary connectivity that is covered by
        // the primary closure.
        std::vector<util::gid> pall;
        for(auto e : aux.i2e[lid]) {
          if(pclo[lco].count(e)) {
            pall.push_back(e);
          } // if
        } // for

        if(pall.size()) {
          cnx[cell_index()].add_row(util::transform_view(pall, entity_g2l));
        }

        cnx[vertex_index()].add_row(
          util::transform_view(aux.i2v[lid], vertex_g2l));
      }
    }

    for(auto gid : aux_pcd.owned) {
      aux_pcd.all.emplace_back(gid);
    }

    for(auto gid : aux_pcd.ghost) {
      aux_pcd.all.emplace_back(gid);
    }

    util::force_unique(aux_pcd.all);
    util::force_unique(aux_pcd.owned);
    util::force_unique(aux_pcd.ghost);

    /*
      Create a lookup table for local ghost offsets.
     */

    for(auto gid : aux_pcd.ghost) {
      auto it = std::find(aux_pcd.all.begin(), aux_pcd.all.end(), gid);
      ghost_offsets[lco][gid] = std::distance(aux_pcd.all.begin(), it);
    } // for

    /*
      We also need to add shared offsets to the lookup table so that we can
      provide local shared offset information to other processes that
      request it.
     */

    for(auto gid : aux_pcd.shared) {
      auto it = std::find(aux_pcd.all.begin(), aux_pcd.all.end(), gid);
      shared_offsets[lco][gid] = std::distance(aux_pcd.all.begin(), it);
    } // for

  } // for

  /*
    Send/Receive requests for shared offsets with other processes.
   */

  auto requested = util::mpi::all_to_allv(
    [&sources](int r, int) -> auto & { return sources[r]; }, comm_);

  /*
    Fulfill the requests that we received from other processes, i.e.,
    provide the local offset for the requested shared mesh ids.
   */

  std::vector<std::vector<util::id>> fulfills(size_);
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
    [f = std::move(fulfills)](int r, int) { return std::move(f[r]); }, comm_);

  struct ghost_info {
    util::gid gid;
    std::size_t lco;
    util::id lid;
    util::id rid;
  };

  // lambda
  auto find_in_source = [&sources, &sources_lco, &ghost_offsets, &fulfilled](
                          util::gid const & in_id,
                          std::size_t const & in_lco,
                          Color & gcolor,
                          ghost_info & ginfo) {
    int k{0};
    for(auto const & rv : sources) { // process
      std::size_t pc{0}, cnt{0};
      for(auto const & cv : rv) { // local colors
        for(auto [id, dmmy, gco] : cv) {
          if((in_id == id) && (in_lco == sources_lco[k][cnt])) {
            gcolor = gco;
            ginfo = {in_id, in_lco, ghost_offsets[pc][id], fulfilled[k][cnt]};
            return;
          }
          ++cnt;
        } // for
        ++pc;
      } // for
      ++k;
    } // for
    flog_fatal("could not find ghost info");
  };

  /*
    Populate the coloring information.
   */
  entities(idx) = num_entities(kind);

  for(auto gco : ours()) {
    auto lco = lc(gco);
    auto & ic = ai[lco];
    auto & aux_pcd = aux_pcdata[kind][lco];

    ic.entities = aux_pcd.all.size();

    for(auto gid : aux_pcd.owned) {
      auto it = std::find(aux_pcd.all.begin(), aux_pcd.all.end(), gid);
      util::id real_lid = (util::id)std::distance(aux_pcd.all.begin(), it);

      if(aux_pcd.shared.count(gid)) {
        for(auto d : aux_pcd.dependents[gid]) {
          ic.peer_shared[d].insert(real_lid);
        }
      }
    }

    // order ghosts by gid
    std::map<util::gid, ghost_entity> ghosts;

    for(auto gid : aux_pcd.ghost) {
      Color gcolor;
      ghost_info ginfo;
      find_in_source(gid, lco, gcolor, ginfo);
      ghosts[gid] = ghost_entity{ginfo.lid, ginfo.rid, gcolor};
    } // for

    for(auto & g : ghosts) {
      ic.ghost.emplace_back(std::move(g.second));
    }
  } // for

  /*
    Populate peer information.
   */

  auto & parts = partitions(idx);
  std::vector<std::vector<Color>> is_peers(ours().size());
  for(std::size_t lco{0}; lco < ours().size(); ++lco) {
    auto & ic = ai[lco];
    parts.emplace_back(ic.entities);
    if(peers.count(lco)) {
      color_peers_[lco].insert(peers.at(lco).begin(), peers.at(lco).end());
      is_peers[lco].assign(peers.at(lco).begin(), peers.at(lco).end());
    } // if
  } // for

  /*
    Gather the tight peer information for this auxiliary entity type.
   */

  {
    auto & apeers = coloring_utils::peers(idx);
    apeers.reserve(cd_.colors);

    for(auto vp : util::mpi::all_gatherv(is_peers, comm_)) {
      for(auto & pc : vp) { /* over process colors */
        apeers.push_back(std::move(pc));
      } // for
    } // for
  } // scope

  /*
    Gather partition sizes for entities.
   */

  concatenate(parts, cd_.colors, comm_);

  /*
   * Compute aux ghost interval sizes
   */

  compute_interval_sizes(idx, ghost_offsets);

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

  for(const auto cell : c2f) {
    std::vector<util::gid> edges;

    // accumulate edges in cell
    for(const std::size_t face : cell) {
      for(const std::size_t ei : f2e[face]) {
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
  Build intermediary entities locally from entity to vertex graph.

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

    // build the edges for the entity
    edges.offsets.clear();
    edges.values.clear();
    if(MD::dimension() == cd_.cid.kind)
      md_.make_entity(kind, p2m[entity++], these_verts, edges);
    else
      md_.make_entity(kind, 0, these_verts, edges);

    for(const auto row : edges) {
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

  for(auto [from, to, transpose] : ca_) {
    for(std::size_t lco{0}; lco < ours().size(); ++lco) {
      coloring(from)[lco].cnx_allocs[to] =
        connectivity(transpose ? to : from)[lco][transpose ? from : to]
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
std::vector<util::crs>
coloring_utils<MD>::get_connectivity(entity_kind from, entity_kind to) {
  auto from_idx = idmap_.at(from);
  auto to_idx = idmap_.at(to);
  std::vector<util::crs> cnxs;
  for(auto & cnx : connectivity(from_idx)) {
    cnxs.push_back(cnx[to_idx]);
  }
  return cnxs;
}

template<class MD>
template<class T>
std::vector<std::vector<T>>
coloring_utils<MD>::send_field(entity_kind k, const std::vector<T> & f) {
  flog_assert(idmap_.find(k) != idmap_.end(), "Invalid kind");
  flog_assert(k == cd_.cid.kind || k == cd_.vid.kind,
    "Invalid kind, only primaries and vertices supported ");

  // local id and type pairing
  using l2t = std::pair<std::size_t, T>;

  const util::equal_map em(num_entities(k), size_);
  auto entities = util::mpi::one_to_allv(pack_field(em, f), comm_);

  std::vector<std::vector<l2t>> locals =
    util::mpi::all_to_allv(move_field(em.size(),
                             cd_.colors,
                             k == cd_.vid.kind ? vertex_raw_ : primary_raw_,
                             entities),
      comm_);

  std::vector<std::vector<T>> res;

  for(auto i : ours())
    res.emplace_back(
      find_field_color(k == cd_.vid.kind ? v2co_ : p2co_, locals, i));

  return res;
} // send_field

/// \}
} // namespace unstructured_impl
} // namespace topo
} // namespace flecsi

/// \endcond

#endif
