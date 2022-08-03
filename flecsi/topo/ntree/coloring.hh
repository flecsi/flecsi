// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_NTREE_COLORING_HH
#define FLECSI_TOPO_NTREE_COLORING_HH

#include <map>
#include <vector>

/// \cond core
namespace flecsi {
namespace topo {
/// \addtogroup ntree
/// \{

/// Ntree topology base
struct ntree_base {

  /// Index spaces used for the ntree topology
  enum index_space { entities, nodes, hashmap, tree_data };
  using index_spaces = util::constants<entities, nodes, hashmap, tree_data>;

  /// Ntree coloring
  struct coloring {

    ///  Build a coloring based on the number of colors \p nparts
    coloring(Color nparts)
      : nparts_(nparts), global_hmap_(nparts * local_hmap_),
        hmap_offset_(nparts, local_hmap_), tdata_offset_(nparts, 3) {}

    /// Number of colors
    Color nparts_;

    /// Number of local entities
    size_t local_entities_;
    /// Number of global entities
    size_t global_entities_;
    /// Entities distribution: number of entities per color
    std::vector<std::size_t> entities_distribution_;
    std::vector<std::size_t> entities_offset_;

    /// Number of local nodes
    size_t local_nodes_;
    /// Number of global nodes
    size_t global_nodes_;
    std::vector<std::size_t> nodes_offset_;

    static constexpr size_t local_hmap_ = 1 << 15;
    size_t global_hmap_;
    std::vector<std::size_t> hmap_offset_;
    std::vector<std::size_t> tdata_offset_;
    std::vector<std::size_t> global_sizes_;
  }; // struct coloring

  static std::size_t allocate(const std::vector<std::size_t> & arr,
    const std::size_t & i) {
    return arr[i];
  }

  static void set_dests(field<data::intervals::Value>::accessor<wo> a) {
    assert(a.span().size() == 1);
    a[0] = data::intervals::make({1, 3});
  }
  static void set_ptrs(field<data::points::Value>::accessor<wo> a) {
    const auto & c = run::context::instance();
    const auto i = c.color(), n = c.colors();
    assert(a.span().size() == 3);
    a[1] = data::points::make(i == 0 ? i : i - 1, 0);
    a[2] = data::points::make(i == n - 1 ? i : i + 1, 0);
  }
  template<auto * F> // work around Clang 10.0.1 bug with auto&
  static constexpr auto task = [](auto f) { execute<*F>(f); };
}; // struct ntree_base

/// \}

} // namespace topo
} // namespace flecsi
/// \endcond

#endif
