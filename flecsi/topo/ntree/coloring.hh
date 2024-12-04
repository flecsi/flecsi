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
struct ntree_base {

  /// Index spaces used for the ntree topology
  enum index_space { entities, nodes, hashmap, tree_data, comms };
  using index_spaces =
    util::constants<entities, nodes, hashmap, tree_data, comms>;
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

    ///  Build a coloring based on the number of colors \p nparts
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
    std::size_t max_depth;
    ent_node local;
    util::id ghosts;
    util::id top_tree;
    util::id nents_recv;
  };

  struct en_size {
    std::vector<util::id> ent, node;
  };

  struct color_id {
    std::size_t color;
    ent_id id;
    std::size_t from_color;
  };

  static std::size_t allocate(const std::vector<util::id> & arr,
    const std::size_t & i) {
    return arr[i];
  }

  static void set_dests(field<data::intervals::Value>::accessor<wo> a) {
    assert(a.span().size() == 1);
    a[0] = data::intervals::make({1, 3});
  }
  static void set_ptrs(field<data::copy_engine::Point>::accessor<wo, na> a) {
    const auto & c = run::context::instance();
    const auto i = c.color(), n = c.colors();
    assert(a.span().size() == 3);
    a[1] = data::copy_engine::point(i == 0 ? i : i - 1, 0);
    a[2] = data::copy_engine::point(i == n - 1 ? i : i + 1, 0);
  }
  template<auto * F> // work around Clang 10.0.1 bug with auto&
  static constexpr auto task = [](auto f) { execute<*F>(f); };
}; // struct ntree_base

/// \}

} // namespace topo
} // namespace flecsi

#endif
