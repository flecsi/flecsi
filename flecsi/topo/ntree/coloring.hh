// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_NTREE_COLORING_HH
#define FLECSI_TOPO_NTREE_COLORING_HH

#include <map>
#include <vector>

namespace flecsi {
namespace topo {
/// \addtogroup ntree
/// \{

/// Ntree topology base
struct ntree_base : base {

  /// Index spaces used for the ntree topology
  enum index_space {
    entities, /// Index space for entities related fields
    nodes, /// Index space for nodes related fields
    hashmap,
    tree_data,
    comms,
    share_ghosts_comms,
    // Buffer for the top tree entities used during the make_tree
    // phase. This is used to perform an AllGather via multi
    // accessors.
    top_tree_ents,
    // Buffer for the top tree nodes used during the make_tree
    // phase. This is used to perform an AllGather via multi
    // accessors.
    top_tree_nodes,
    // Buffer for the color/id. It contains the local
    // entities used elsewhere to find
    // neighbors during the share_ghosts phase. This is
    // used to perform an AllToAllv. We are using the
    // entities index space as the receiving buffer of
    // the buffer copy in this case.
    share_ghosts_cid_comm,
    // Buffer for the neighbors entities. This
    // temporary buffer stores the information for the
    // AllToAllv communication in the share_ghosts
    // phase.
    share_ghosts_buffer_comm,
    // Buffer containing the neighbor
    // entities, result of the AllToAllv
    // communication from
    // share_ghosts_buffer_comm. This is used
    // in the share_ghosts phase.
    share_ghosts_distant_buffer_comm
  };

  /// \hideinitializer The specialization developer is required to use the index
  /// spaces provided by the N-Tree.
  using index_spaces = util::constants<entities,
    nodes,
    hashmap,
    tree_data,
    comms,
    share_ghosts_comms,
    top_tree_ents,
    top_tree_nodes,
    share_ghosts_cid_comm,
    share_ghosts_buffer_comm,
    share_ghosts_distant_buffer_comm>;
  /// Parallel types for nodes and entities.
  enum ptype_t {
    exclusive, ///< Owned data.
    ghost, ///< Remote data.
    all ///< Both kinds.
  };
  /// Traversal types for DFS
  enum ttype_t {
    preorder, ///< Pre-ordered DFS traversal
    postorder, ///< Post-ordered DFS traversal
    reverse_preorder, ///< Reverse pre-ordered DFS traversal
    reverse_postorder ///< Reverse post-ordered DFS traversal
  };

  /// Ntree coloring
  struct coloring {

    /// Build a coloring based on the number of colors \p nparts, and the size
    /// of the hashtable.
    /// \param nparts Number of colors
    /// \param hmap_size Number of entries in the hashtable
    coloring(Color nparts, util::id hmap_size)
      : nparts_(nparts), local_hmap_(hmap_size) {}

    /// Number of colors
    Color nparts_;
    /// Entities distribution: number of entities per color
    std::vector<util::id> entities_sizes_;
    /// Nodes distribution: number of nodes per color
    std::vector<util::id> nodes_sizes_;

    /// Size of the hashtable on each color
    util::id local_hmap_;
  }; // struct coloring

protected:
  using ent_id = topo::id<entities>;
  using node_id = topo::id<nodes>;

  struct ent_node {
    util::id ents;
    util::id nodes;
  };

  struct meta_type {
    // #local entities and nodes
    ent_node local;
    // #ghost entities
    util::id ghosts;
    // #ghosts nodes in the top_tree
    util::id top_tree;
    // #entities received during the first and second AllToAllv buffer copy in
    // share_ghosts phase
    util::id nents_recv, nents_recv_2;
    // Total #entities and #nodes received from make_tree_distributed_task to
    // create top tree copy plans
    util::id cp_nents_tt, cp_nnodes_tt;
  };

  struct color_id {
    std::size_t color;
    ent_id id;
    std::size_t from_color;
  };

  static void set_dests(
    field<data::intervals::Value>::accessor<wo> a) noexcept {
    assert(a.span().size() == 1);
    a[0] = data::intervals::make({1, 3});
  }
  static void set_ptrs(
    field<data::copy_engine::Point>::accessor<wo, wo> a) noexcept {
    const auto & c = run::context::instance();
    const auto i = c.color(), n = c.colors();
    assert(a.span().size() == 3);
    a[1] = data::copy_engine::point(i == 0 ? i : i - 1, 0);
    a[2] = data::copy_engine::point(i == n - 1 ? i : i + 1, 0);
  }

  static void set_dests_share_ghosts_comms(
    field<data::intervals::Value>::accessor<wo> a) noexcept {
    const auto & c = run::context::instance().colors();
    a[0] = data::intervals::make({c, 2 * c - 1});
  }
  static void set_ptrs_share_ghosts_comms(
    field<data::copy_engine::Point>::accessor<wo, wo> a) noexcept {
    const auto & c = run::context::instance();
    assert(a.span().size() == 2 * c.colors() - 1);
    for(Color i = 0; i < c.colors() - 1; ++i)
      a[c.colors() + i] =
        data::copy_engine::point(i + (i >= c.color()), c.color());
  }

  template<auto * F> // work around Clang 10.0.1 bug with auto&
  static auto task(scheduler & s) {
    return [&](auto f) { s.execute<*F>(f); };
  }
}; // struct ntree_base

/// \}

} // namespace topo
} // namespace flecsi

#endif
