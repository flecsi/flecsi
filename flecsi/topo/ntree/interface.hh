// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_NTREE_INTERFACE_HH
#define FLECSI_TOPO_NTREE_INTERFACE_HH

#include "flecsi/config.hh"
#include "flecsi/data/accessor.hh"
#include "flecsi/data/copy_plan.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/core.hh" // base
#include "flecsi/topo/ntree/coloring.hh"
#include "flecsi/topo/ntree/types.hh"
#include "flecsi/util/gpu_containers.hh"
#include "flecsi/util/hashtable.hh"
#include "flecsi/util/sort.hh"

#if defined(FLECSI_ENABLE_GRAPHVIZ)
#include "flecsi/util/graphviz.hh"
#endif

#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <stack>
#include <type_traits>
#include <unordered_map>

namespace flecsi {
namespace topo {

//---------------------------------------------------------------------------//
// NTree topology.
//---------------------------------------------------------------------------//

/// \defgroup ntree N-dimensional Tree
/// Binary, Quad and Oct Tree topology.
/// The ntree topology is using a hashing table to store and access the entities
/// and nodes of the tree.
/// \warning Only the Legion backend is supported for the N-Tree topology
/// \warning N-Tree topology does not have support for Ragged or Sparse fields
/// \ingroup topology
/// \{

/// The ntree topology represents a binary, quad or octree stored/accessed
/// using a hashtable. The creation of the N-Tree requires three steps, after
/// filling the appropriate index spaces data:
///   - call make_tree function
///   - Compute the local information for the interation. This information is
///   used for the next step to compute the ghosts.
///   - call share_ghosts function
/// After these calls the N-Tree is ready to be used and the neighbors are
/// computed/available.
/// \tparam Policy the specialization, following \ref ntree_specialization
/// \see [The N-Tree tutorial](../../tutorial/ntree.html)
template<typename Policy>
struct topology<Policy, ntree_base> : ntree_base, with_meta<Policy> {

private:
  constexpr static Dimension dimension = Policy::dimension;
  constexpr static util::id max_neighbors = Policy::max_neighbors;
  using key_t = typename Policy::key_t;

  using type_t = double;
  /// Type store in the hastable. It can represent both node or entity
  using hcell_t = hcell_base_t<dimension, type_t, key_t>;

  using entity_data = typename Policy::entity_data;
  using node_data = typename Policy::node_data;

  struct ntree_data {
    key_t hibound = key_t::root(), lobound = key_t::root();
  };

  constexpr static std::size_t nchildren_ = 1 << dimension;

public:
  template<Privileges>
  struct access;

  // Create the ntree data structure based on a coloring.
  // This allocates the different index space and create the copy_plan for the
  // meta data. This copy_plan never changes throughout the lifetime of the
  // tree.
  topology(scheduler & s, const coloring & c)
    : with_meta<Policy>(s, c.nparts_),
      part{{rep<entities>(s, c, c.entities_sizes_),
        rep<nodes>(s, c, c.nodes_sizes_),
        rep<hashmap>(s, c, c.local_hmap_),
        rep<tree_data>(s, c, 3),
        rep<comms>(s, c, c.nparts_),
        rep<share_ghosts_comms>(s, c, c.nparts_ * 2 - 1),
        rep<top_tree_ents>(s, c, 1),
        rep<top_tree_nodes>(s, c, 1),
        rep<share_ghosts_cid_comm>(s, c, 1),
        rep<share_ghosts_buffer_comm>(s, c, 1),
        rep<share_ghosts_distant_buffer_comm>(s, c, 1)}},
      cp_data_tree(s,
        *this,
        // Avoid initializer-list constructor:
        data::copy_plan::Sizes(c.nparts_, 1),
        task<set_dests>(s),
        task<set_ptrs>(s),
        util::constant<tree_data>()),
      cp_share_ghosts_comms(s,
        *this,
        // Avoid initializer-list constructor:
        data::copy_plan::Sizes(c.nparts_, 1),
        task<set_dests_share_ghosts_comms>(s),
        task<set_ptrs_share_ghosts_comms>(s),
        util::constant<share_ghosts_comms>()),
      buf(s, [&c] {
        data::buffers::coloring ret(c.nparts_);
        for(std::size_t i_r = 0; i_r < ret.size(); ++i_r) {
          for(std::size_t i = 0; i < c.nparts_; ++i) {
            if(i != i_r) {
              ret[i_r].push_back(i);
            }
          }
        }
        return ret;
      }()) {
    // Initialize the meta_field
    s.execute<init_meta_field>(meta_field(this->meta), c.entities_sizes_);
  }

private:
  template<index_space idx>
  auto
  rep(scheduler & s, const coloring & c, const std::vector<util::id> & size) {
    return make_repartitioned<Policy, idx>(
      c.nparts_, s, [size](std::size_t i) { return size[i]; });
  }

  template<index_space idx>
  auto rep(scheduler & s, const coloring & c, util::id size) {
    return make_repartitioned<Policy, idx>(
      c.nparts_, s, [size](std::size_t) { return size; });
  }

  // Ntree mandatory fields ---------------------------------------------------
public:
  /// Entities keys field
  static inline const typename field<key_t>::template definition<Policy,
    entities>
    e_keys;
  /// Entities ids field. This field can be used to identify entities with the
  /// same key.
  static inline const field<util::id>::definition<Policy, entities> e_ids;
  /// Entities color field. This represent which color owns an entity.
  static inline const typename field<Color>::template definition<Policy,
    entities>
    e_colors;
  /// Field containing the structure for entities interation from the
  /// specialization
  static inline const typename field<entity_data>::template definition<Policy,
    entities>
    e_i;

  /// Node keys field
  static inline const typename field<key_t>::template definition<Policy, nodes>
    n_keys;
  /// Field containing the structure for nodes interation from the
  /// specialization
  static inline const typename field<node_data>::template definition<Policy,
    nodes>
    n_i;

private:
  using h_s_t = std::pair<hcell_t, std::size_t>;
  using hmap_pair_t = std::pair<key_t, hcell_t>;
  // Hmap fields, the hashing table are reconstructed based on this field
  static inline const typename field<hmap_pair_t>::template definition<Policy,
    hashmap>
    hcells;
  // Meta data fields for ntree, genetic data and communications
  static inline const typename field<ntree_data>::template definition<Policy,
    tree_data>
    data_field;
  static inline const typename field<meta_type,
    data::single>::template definition<meta<Policy>>
    meta_field;

  // Field for copies
  static inline const field<util::id>::definition<Policy, comms> comms_field;
  static inline const field<util::id>::definition<Policy, share_ghosts_comms>
    share_ghosts_comms_field;
  static inline const typename field<hcell_t>::template definition<Policy,
    top_tree_ents>
    top_tree_ents_field;
  static inline const typename field<hcell_t>::template definition<Policy,
    top_tree_nodes>
    top_tree_nodes_field;
  static inline const typename field<color_id>::template definition<Policy,
    share_ghosts_cid_comm>
    share_ghosts_cid_comm_field;
  static inline const typename field<h_s_t>::template definition<Policy,
    share_ghosts_buffer_comm>
    share_ghosts_buffer_comm_field;
  static inline const typename field<h_s_t>::template definition<Policy,
    share_ghosts_distant_buffer_comm>
    share_ghosts_distant_buffer_comm_field;

  // --------------------------------------------------------------------------

  // Index space index
  util::key_array<repartitioned, index_spaces> part;

  // Copy plan for the tree data field
  data::copy_plan cp_data_tree, cp_share_ghosts_comms;
  std::optional<data::copy_plan> cp_top_tree_nodes, cp_entities;

  // Buffer for ghosts shared
  data::buffers::topology buf;

  /// Hashing table type
  using hmap_t = util::hashtable<key_t, hcell_t, Policy>;

  FLECSI_INLINE_TARGET static hmap_t map(
    typename field<hmap_pair_t>::template accessor<rw, na> hcells) {
    return hcells.span();
  }

  static void init_meta_field(
    typename field<meta_type, data::single>::template accessor<wo> mf,
    const std::vector<util::id> & size) noexcept {
    mf->local.ents = size[run::context::instance().color()];
  }

  // ----------------------- Top Tree Construction Tasks -----------------------
  // Build the local tree.
  // Add the entities in the hashmap and create needed nodes.
  // After this first step, the top tree entities and nodes are returned. These
  // will build the top tree, shared by all colors.
  static std::array<std::size_t, 2> make_tree_local_task(
    typename field<key_t>::template accessor<rw, na> e_keys,
    typename field<key_t>::template accessor<rw, na> n_keys,
    typename field<ntree_data>::template accessor<ro, ro> data_field,
    typename field<hmap_pair_t>::template accessor<rw, na> hcells,
    typename field<meta_type, data::single>::template accessor<rw>
      mf) noexcept {
    // Cstr htable
    auto hmap = map(hcells);

    // Create the tree
    const Color size = run::context::instance().colors(),
                color = run::context::instance().color();

    flog_assert(e_keys.span().end() ==
                  std::unique(e_keys.span().begin(), e_keys.span().end()),
      "The keys are not unique");

    /* Exchange high and low bound */
    const auto hibound =
      color == size - 1 ? key_t::max() : data_field(2).lobound;
    const auto lobound = color == 0 ? key_t::min() : data_field(1).hibound;
    // Check sort and data_field communication
    flog_assert(lobound <= e_keys(0), "The keys are not globally sorted");
    flog_assert(hibound >= e_keys(mf->local.ents - 1),
      "The keys are not globally sorted");

    // Add the root
    hmap.insert(key_t::root(), key_t::root());
    auto root_ = hmap.find(key_t::root());
    root_->second.set_color(color);
    {
      const std::size_t cnode = mf->local.nodes++;
      root_->second.set_node_idx(cnode);
      n_keys(cnode) = root_->second.key();
    }
    std::size_t current_depth = key_t::max_depth();
    // Entity keys, last and current
    key_t lastekey = key_t(0);
    if(color != 0)
      lastekey = lobound;
    // Node keys, last and Current
    key_t lastnkey = key_t::root();
    key_t nkey, loboundnode, hiboundnode;
    // Current parent and value
    hcell_t * parent = nullptr;
    bool old_is_ent = false;

    const bool iam0 = color == 0;
    const bool iamlast = color == size - 1;

    // The extra turn in the loop is to finish the missing
    // parent of the last entity
    for(std::size_t i = 0; i <= e_keys.span().size(); ++i) {
      const key_t ekey = i < e_keys.span().size() ? e_keys(i) : hibound;
      nkey = ekey;
      nkey.pop(current_depth);
      bool loopagain = false;
      // Loop while there is a difference in the current keys
      while(nkey != lastnkey || (iamlast && i == e_keys.span().size())) {
        loboundnode = lobound;
        loboundnode.pop(current_depth);
        hiboundnode = hibound;
        hiboundnode.pop(current_depth);
        if(loopagain && (iam0 || lastnkey > loboundnode) &&
           (iamlast || lastnkey < hiboundnode)) {
          hcell_t & n = hmap.at(lastnkey);
          if(!n.is_ent()) {
            n.set_complete();
          }
        }
        if(iamlast && lastnkey == key_t::root())
          break;
        loopagain = true;
        current_depth++;
        nkey = ekey;
        nkey.pop(current_depth);
        lastnkey = lastekey;
        lastnkey.pop(current_depth);
      } // while

      if(iamlast && i == e_keys.span().size())
        break;

      parent = &(hmap.at(lastnkey));
      old_is_ent = parent->is_ent();
      // Insert the eventual missing parents in the tree
      // Find the current parent of the two entities
      while(1) {
        current_depth--;
        lastnkey = lastekey;
        lastnkey.pop(current_depth);
        nkey = ekey;
        nkey.pop(current_depth);
        if(nkey != lastnkey)
          break;
        // Add a children
        parent->add_child(nkey.last_value());
        parent->set_node();
        parent = &(hmap.insert(nkey, nkey)->second);
        parent->set_color(color);
      } // while

      // Recover deleted entity
      if(old_is_ent) {
        parent->add_child(lastnkey.last_value());
        parent->set_node();
        auto it = hmap.insert(lastnkey, lastnkey);
        it->second.set_ent_idx(i - 1);
        it->second.set_color(color);
      } // if

      if(i < e_keys.span().size()) {
        // Insert the new entity
        parent->add_child(nkey.last_value());
        auto it = hmap.insert(nkey, nkey);
        it->second.set_ent_idx(i);
        it->second.set_color(color);
      } // if

      // Prepare next loop
      lastekey = ekey;
      lastnkey = nkey;
    } // for

    // Generate the indices of the local nodes
    std::queue<hcell_t *> tqueue;
    tqueue.push(&hmap.at(key_t::root()));
    while(!tqueue.empty()) {
      hcell_t * cur = tqueue.front();
      tqueue.pop();
      assert(cur->is_node());
      auto nkey = cur->key();
      if(cur->key() != key_t::root()) {
        assert(cur->idx() == 0);
        std::size_t cnode = mf->local.nodes++;
        cur->set_node_idx(cnode);
        n_keys(cnode) = cur->key();
      }
      for(std::size_t j = 0; j < nchildren_; ++j) {
        if(cur->has_child(j)) {
          auto it = hmap.find(nkey.push(j));
          if(it->second.is_node())
            tqueue.push(&it->second);
        }
      } // for
    } // while
    std::size_t count_ents = 0, count_nodes = 0;
    top_tree_boundaries<true>(hmap, count_ents, count_nodes);
    return {count_ents, count_nodes};
  } // make_tree

  static void fill_top_tree_task(
    typename field<hmap_pair_t>::template accessor<rw, na> hcells,
    typename field<hcell_t>::template accessor<wo, wo> top_tree_ents_field,
    typename field<hcell_t>::template accessor<wo, wo>
      top_tree_nodes_field) noexcept {
    auto hmap = map(hcells);
    top_tree_boundaries<false>(hmap, top_tree_ents_field, top_tree_nodes_field);
  }

  // Template to return if we count or fill the index space
  template<bool C, typename T>
  static void top_tree_boundaries(hmap_t & hmap,
    T & count_accessor_ents,
    T & count_accessor_nodes) {

    [[maybe_unused]] std::size_t count_ents = 0, count_nodes = 0;

    auto color = run::context::instance().color();
    std::vector<hcell_t *> queue;
    std::vector<hcell_t *> nqueue;
    queue.push_back(&hmap.find(key_t::root())->second);
    while(!queue.empty()) {
      for(hcell_t * cur : queue) {
        cur->set_color(color);
        key_t nkey = cur->key();
        if(cur->is_node() && cur->is_incomplete()) {
          assert(cur->type() != 0);
          for(std::size_t j = 0; j < nchildren_; ++j) {
            if(cur->has_child(j)) {
              nqueue.push_back(&hmap.at(nkey.push(j)));
            }
          } // for
        }
        else {
          if(cur->is_node()) {
            if constexpr(C)
              ++count_accessor_nodes;
            else
              count_accessor_nodes[count_nodes++] = *cur;
          }
          else {
            if constexpr(C)
              ++count_accessor_ents;
            else
              count_accessor_ents[count_ents++] = *cur;
          }
        } // else
      } // for
      queue = std::move(nqueue);
      nqueue.clear();
    } // while
  } // top_tree_boundaries

  // Add missing parent from distant node/entity
  // This version does add the parents and add an idx
  // This can only be done before any ghosts are received
  static void add_parent(key_t key,
    typename key_t::int_t child,
    const int & color,
    hmap_t & hmap,
    typename field<meta_type, data::single>::template accessor<rw> mf,
    typename field<key_t>::template accessor<rw, na> n_keys) {
    auto parent = hmap.end();
    while((parent = hmap.find(key)) == hmap.end()) {
      parent = hmap.insert(key, key);
      const std::size_t cnode = mf->local.nodes++;
      parent->second.set_node_idx(cnode);
      n_keys(cnode) = key;
      parent->second.add_child(child);
      parent->second.set_color(color);
      child = key.pop();

    } // while
    assert(parent->second.is_incomplete());
    parent->second.add_child(child);
  }

  static void load_shared_entity(const std::size_t & c,
    const key_t & k,
    hmap_t & hmap,
    typename field<meta_type, data::single>::template accessor<rw> mf,
    typename field<key_t>::template accessor<rw, na> n_keys) {
    auto key = k;
    auto f = hmap.find(key);
    if(f == hmap.end()) {
      auto & cur = hmap.insert(key, key)->second;
      cur.set_nonlocal();
      cur.set_color(c);
      auto eid = mf->local.ents + mf->ghosts++;
      cur.set_ent_idx(eid);
      // Add missing parent(s)
      auto lastbit = key.pop();
      add_parent(key, lastbit, c, hmap, mf, n_keys);
    }
    else {
      assert(false);
    }
  }

  static void load_shared_node(const std::size_t & c,
    const key_t & k,
    hmap_t & hmap,
    typename field<meta_type, data::single>::template accessor<rw> mf,
    typename field<key_t>::template accessor<rw, na> n_keys) {
    key_t key = k;
    // Node doesnt exists already
    auto cur = hmap.find(key);
    if(cur == hmap.end()) {
      auto & cur = hmap.insert(key, key)->second;
      cur.set_nonlocal();
      cur.set_color(c);
      // Add missing parent(s)
      auto lastbit = key.pop();
      add_parent(key, lastbit, c, hmap, mf, n_keys);
    }
    else {
      assert(false);
    }
  }

  // Add the top tree to the local tree.
  // First step is to load the top tree entities and nodes.
  // The new sizes are then returned to create the copy plan for the top tree.
  static void make_tree_distributed_task(
    typename field<key_t>::template accessor<rw, na> n_keys,
    typename field<meta_type, data::single>::template accessor<rw> mf,
    typename field<ntree_data>::template accessor<rw, na> data_field,
    typename field<hmap_pair_t>::template accessor<rw, na> hcell,
    data::multi<typename field<hcell_t>::template accessor<ro, na>>
      ents_cells_multi,
    data::multi<typename field<hcell_t>::template accessor<ro, na>>
      nodes_cells_multi) noexcept {

    auto hmap = map(hcell);
    util::id total_ents = 0, total_nodes = 0;
    const auto color = run::context::instance().color();
    for(auto & m : ents_cells_multi.accessors()) {
      for(auto & c : m.span()) {
        if(c.color() == color)
          continue;
        ++total_ents;
        load_shared_entity(c.color(), c.key(), hmap, mf, n_keys);
      }
    }
    for(auto & m : nodes_cells_multi.accessors()) {
      for(auto & c : m.span()) {
        if(c.color() == color)
          continue;
        ++total_nodes;
        load_shared_node(c.color(), c.key(), hmap, mf, n_keys);
      }
    }
    // Update the lo and hi bounds
    data_field(0).lobound = key_t::min();
    data_field(0).hibound = key_t::max();

    // Add the distant nodes, at the end
    for(auto & m : nodes_cells_multi.accessors()) {
      for(auto & c : m.span()) {
        if(c.color() == color)
          continue;
        assert(n_keys.span().size() > mf->local.nodes + mf->top_tree + 1);
        auto cur = hmap.find(c.key());
        assert(cur != hmap.end());
        const std::size_t cnode = mf->local.nodes + mf->top_tree++;
        cur->second.set_node_idx(cnode);
        n_keys(cnode) = cur->second.key();
      }
    }

    // Update meta for copy plans
    mf->cp_nents_tt = total_ents;
    mf->cp_nnodes_tt = total_nodes;
  }

  static void increase_size_task(topo::resize::Field::accessor<rw> a,
    util::id v) {
    a = a.get() + v;
  }

  static void resize_entities_update_meta_task(
    topo::resize::Field::accessor<rw> a,
    typename field<meta_type, data::single>::template accessor<rw>
      mf) noexcept {
    a = mf->local.ents + mf->nents_recv_2;
  }

  static void copy_sizes_meta_top_tree_task(topo::resize::Field::accessor<wo> a,
    typename field<meta_type, data::single>::template accessor<ro>
      mf) noexcept {
    a = mf->local.ents + mf->cp_nents_tt;
  }

  static void copy_sizes_meta_task(topo::resize::Field::accessor<wo> a,
    typename field<meta_type, data::single>::template accessor<ro>
      mf) noexcept {
    a = mf->local.ents;
  }

  static void copy_sizes_task(topo::resize::Field::accessor<wo> a,
    field<util::id>::accessor<ro, ro> b) noexcept {
    a = std::accumulate(b.span().begin(),
      b.span().begin() + run::context::instance().colors(),
      0);
  } // copy_sizes_task

  template<bool C>
  static void copy_sizes_resize_task(topo::resize::Field::accessor<rw> a,
    typename field<meta_type, data::single>::template accessor<ro> mf,
    field<util::id>::accessor<ro, na> b) noexcept {
    const auto color = run::context::instance().color();
    a = (C ? mf->local.ents : 0) + b[color] +
        std::accumulate(b.span().begin() + run::context::instance().colors(),
          b.span().end(),
          0);
  }

  template<index_space E = entities>
  static void copy_sizes_top_tree_task(topo::resize::Field::accessor<wo> a,
    future<std::array<std::size_t, 2>> b) noexcept {
    a = b.get()[E != entities];
  }

  // Return the number of entities/nodes for local, top tree and ghosts
  // stores in the respective index spaces.
  static std::array<util::id, 4> sizes_task(
    typename field<meta_type, data::single>::template accessor<ro> mf) {
    return {{mf->local.ents, mf->local.nodes, mf->top_tree, mf->ghosts}};
  }

  static void reset_ghosts(
    typename field<meta_type, data::single>::template accessor<rw> mf,
    typename field<hmap_pair_t>::template accessor<rw, na> hcell) noexcept {
    auto hmap = map(hcell);
    // Remove ghosts from hmap
    for(auto & i : hmap)
      if(i.second.is_ent() && i.second.is_nonlocal()) {
        auto c = i.first.pop();
        hmap.find(i.first)->second.remove_child(c);
        i = {};
      }
    mf->ghosts = 0;
  }

  // Copy plan: set destination sizes
  template<index_space IS = entities>
  static void set_destination_meta_top_tree(
    field<data::intervals::Value>::accessor<wo> a,
    typename field<meta_type, data::single>::template accessor<ro>
      mf) noexcept {
    const auto n = IS == entities ? mf->local.ents : mf->local.nodes;
    a(0) = data::intervals::make(
      {n, n + (IS == entities ? mf->cp_nents_tt : mf->cp_nnodes_tt)},
      run::context::instance().color());
  }

  // Copy plan: set destination sizes
  static void set_destination_meta(
    field<data::intervals::Value>::accessor<wo> a,
    typename field<meta_type, data::single>::template accessor<ro>
      mf) noexcept {
    a(0) =
      data::intervals::make({mf->local.ents, mf->local.ents + mf->nents_recv_2},
        run::context::instance().color());
  }

  // Copy plan: set pointers for top tree
  template<index_space IS = entities>
  static void set_top_tree_ptrs(
    field<data::copy_engine::Point>::accessor<wo, wo> a,
    typename field<meta_type, data::single>::template accessor<ro> mf,
    data::multi<typename field<hcell_t>::template accessor<ro, na>>
      hcells_multi) noexcept {
    const auto i = run::context::instance().color();
    util::id idx = IS == entities ? mf->local.ents : mf->local.nodes;
    for(auto & m : hcells_multi.accessors())
      for(auto & c : m.span())
        if((IS == entities ? c.is_ent() : c.is_node()) && c.color() != i)
          a(idx++) = data::copy_engine::point(c.color(), c.idx());
  }

  // Copy plan: set pointers to entities
  static void set_entities_ptrs(
    field<data::copy_engine::Point>::accessor<wo, wo> a,
    typename field<meta_type, data::single>::template accessor<ro> mf,
    typename field<h_s_t>::template accessor<ro, na> ids) noexcept {
    util::id idx = mf->local.ents;
    for(std::size_t j = 0; j < mf->nents_recv_2; ++j) {
      assert(ids[j].first.color() != run::context::instance().color());
      a(idx++) =
        data::copy_engine::point(ids[j].first.color(), ids[j].first.idx());
    }
  }

  // Exchange the boundaries from the local tree to other colors.
  // This shares part of the top tree information.
  static void exchange_boundaries_task(
    typename field<key_t>::template accessor<ro, na> e_keys,
    typename field<meta_type, data::single>::template accessor<rw> mf,
    typename field<ntree_data>::template accessor<rw, na> df) noexcept {
    auto c = run::context::instance().color();
    mf->local.ents = e_keys.span().size();
    df(0).lobound = c == 0 ? key_t::min() : e_keys(0);
    df(0).hibound = c == run::context::instance().colors() - 1
                      ? key_t::max()
                      : e_keys(mf->local.ents - 1);
  }

  // Recolor the entities, some might have moved after the sort.
  static void recolor_task(
    typename Policy::template accessor<rw, wo> t) noexcept {
    for(auto & v : t.e_colors.span())
      v = run::context::instance().color();
  }

  // ---------------------------- Top tree construction -----------------------
public:
  /// Build the local tree and share the tree boundaries
  void make_tree(scheduler & s) {
    //  Sort entities
    util::sort(s, e_keys(*this))();
    s.execute<recolor_task>(*this);
    s.execute<exchange_boundaries_task>(
      e_keys(*this), meta_field(this->meta), data_field(*this));

    // Create the local tree
    // Return the list of nodes to share (top of the tree)
    auto fm_top_tree = s.execute<make_tree_local_task>(e_keys(*this),
      n_keys(*this),
      data_field(*this),
      hcells(*this),
      meta_field(this->meta));

    {
      auto & p = get_partition<top_tree_ents>();
      s.execute<copy_sizes_top_tree_task<entities>>(p.sizes(), fm_top_tree);
      p.resize();
    }

    {
      auto & p = get_partition<top_tree_nodes>();
      s.execute<copy_sizes_top_tree_task<nodes>>(p.sizes(), fm_top_tree);
      p.resize();
    }

    s.execute<fill_top_tree_task>(
      hcells(*this), top_tree_ents_field(*this), top_tree_nodes_field(*this));

    auto lm_top_tree =
      data::launch::make(s, *this, data::launch::gather(colors(), colors()));

    // Add the new hcells to the local tree + return new sizes for allocation
    s.execute<make_tree_distributed_task>(n_keys(*this),
      meta_field(this->meta),
      data_field(*this),
      hcells(*this),
      top_tree_ents_field(lm_top_tree),
      top_tree_nodes_field(lm_top_tree));

    {
      auto & p = get_partition<entities>();
      s.execute<copy_sizes_meta_top_tree_task>(
        p.sizes(), meta_field(this->meta));
      p.resize();
    }

    // Fake initialization for the new ghosts
    for(auto & f : run::context::field_info_store<Policy, entities>()) {
      auto fr = data::field_reference<std::byte, data::raw, Policy, entities>(
        f->fid, *this);
      s.execute<fake_initialize>(fr);
    }

    cp_entities.emplace(
      s,
      *this,
      data::copy_plan::Sizes(colors(), 1),
      [&](auto f) {
        s.execute<set_destination_meta_top_tree<entities>>(
          f, meta_field(this->meta));
      },
      [&](auto f) {
        s.execute<set_top_tree_ptrs<entities>>(
          f, meta_field(this->meta), top_tree_ents_field(lm_top_tree));
      },
      util::constant<entities>());

    cp_top_tree_nodes.emplace(
      s,
      *this,
      data::copy_plan::Sizes(colors(), 1),
      [&](auto f) {
        s.execute<set_destination_meta_top_tree<nodes>>(
          f, meta_field(this->meta));
      },
      [&](auto f) {
        s.execute<set_top_tree_ptrs<nodes>>(
          f, meta_field(this->meta), top_tree_nodes_field(lm_top_tree));
      },
      util::constant<nodes>());
  }
  /// \deprecated Pass a \c scheduler.
  [[deprecated("pass a scheduler")]] static void make_tree(
    typename Policy::slot & ts) {
    ts->make_tree(*scheduler::instance);
  }

  // ---------------------------- Ghosts exchange tasks -----------------------
private:
  static void xfer_entities_req_start(
    typename field<entity_data>::template accessor<rw, na> a,
    field<util::id>::accessor<wo, na> restart,
    data::buffers::Start mv,
    typename field<color_id>::accessor<ro, na> f) noexcept {
    std::fill(restart.span().begin(), restart.span().end(), 0);
    util::id cur = 0;
    const auto color = run::context::instance().color();
    for(Color c = 0; c < run::context::instance().colors(); ++c) {
      if(c != color) {
        auto w = mv[cur].write();
        for(std::size_t i = 0; i < f.span().size(); ++i) {
          if(f[i].color == c && !w(a(f[i].id))) {
            restart(cur) = i;
            break;
          } // if
        } // for
        ++cur;
      } // if
    } // for
  } // xfer_nodes_req_start

  static void xfer_entities_cp_start(field<util::id>::accessor<wo, na> restart,
    data::buffers::Start mv,
    typename field<h_s_t>::template accessor<ro, na> f) noexcept {
    std::fill(restart.span().begin(), restart.span().end(), 0);
    util::id cur = 0;
    const auto color = run::context::instance().color();
    for(Color c = 0; c < run::context::instance().colors(); ++c) {
      if(c != color) {
        auto w = mv[cur].write();
        for(std::size_t i = 0; i < f.span().size(); ++i) {
          if(f[i].second == c && !w(f[i])) {
            restart(cur) = i;
            break;
          } // if
        } // for
        ++cur;
      } // if
    } // for
  } // xfer_nodes_cp_start

  static auto xfer_entities_req(
    typename field<entity_data>::template accessor<rw, na> a,
    typename field<meta_type, data::single>::template accessor<rw> m,
    field<util::id>::accessor<rw, na> restart,
    data::buffers::Transfer mv,
    typename field<color_id>::template accessor<ro, na> f,
    typename field<Color>::template accessor<rw, na> e_c) noexcept {

    // Read
    std::size_t cs = run::context::instance().colors();
    int cur = 0;
    util::id idx = m->local.ents;
    const auto color = run::context::instance().color();
    for(Color c = 0; c < cs; ++c) {
      if(c != color) {
        auto r = mv[cur + cs - 1].read();
        while(r) {
          e_c[idx + m->nents_recv] = c;
          a(idx + m->nents_recv++) = r();
        } // while
        ++cur;
      } // if
    } // for

    // Keep copying if needed
    bool more_to_copy = false;
    cur = 0;
    for(Color c = 0; c < cs; ++c) {
      if(c != color) {
        bool done = true;
        if(restart(cur) != 0) {
          auto w = mv[cur].write();
          for(std::size_t i = restart(c); i < f.span().size(); ++i) {
            if(f[i].color == c && !w(a(f[i].id))) {
              restart(cur) = i;
              done = false;
              more_to_copy = true;
              break;
            } // if
          } // for
          if(done)
            restart(cur) = 0;
          ++cur;
        } // if
      } // if
    } // for
    return more_to_copy;
  } // xfer_entities_req

  static auto xfer_entities_cp(
    typename field<h_s_t>::template accessor<rw, na> a,
    typename field<meta_type, data::single>::template accessor<rw> m,
    field<util::id>::accessor<rw, na> restart,
    data::buffers::Transfer mv,
    typename field<h_s_t>::template accessor<ro, na> f) noexcept {
    // Read
    std::size_t cs = run::context::instance().colors();
    int cur = 0;
    const auto color = run::context::instance().color();
    for(Color c = 0; c < cs; ++c) {
      if(c != color) {
        auto r = mv[cur + cs - 1].read();
        while(r) {
          a(m->nents_recv_2++) = r();
        } // while
        ++cur;
      } // if
    } // for

    // Keep copying if needed
    bool more_to_copy = false;
    cur = 0;
    for(Color c = 0; c < cs; ++c) {
      if(c != color) {
        bool done = true;
        if(restart(cur) != 0) {
          auto w = mv[cur].write();
          for(std::size_t i = restart(c); i < f.span().size(); ++i) {
            if(f[i].second == c && !w(f[i])) {
              restart(cur) = i;
              done = false;
              more_to_copy = true;
              break;
            } // if
          } // for
          if(done)
            restart(cur) = 0;
          ++cur;
        } // if
      } // if
    } // for
    return more_to_copy;
  } // xfer_entities_cp

  template<bool C>
  static void find_local_task(typename Policy::template accessor<rw, na> t,
    typename field<std::conditional_t<C, util::id, color_id>>::
      template accessor<wo, na> a) noexcept {
    t.template find_send_entities<C>(a);
  }

  template<bool C>
  static void find_distant_task(typename Policy::template accessor<rw, na> t,
    typename field<std::conditional_t<C, util::id, h_s_t>>::
      template accessor<wo, na> a) noexcept {
    t.template find_intersect_entities<C>(a);
  }

  // Add missing parent from distant node/entity
  // This version does add the parent but does not provides an idx for it.
  // This is used when the local tree already received distant entities/nodes.
  static void add_parent_distant(key_t key,
    typename key_t::int_t child,
    const int & color,
    hmap_t & hmap) {
    auto parent = hmap.end();
    while((parent = hmap.find(key)) == hmap.end()) {
      parent = hmap.insert(key, key);
      parent->second.add_child(child);
      parent->second.set_color(color);
      child = key.pop();

    } // while
    assert(parent->second.is_incomplete());
    parent->second.add_child(child);
  }

  static void load_entities_task(
    typename field<hmap_pair_t>::template accessor<rw, na> hcells,
    typename field<meta_type, data::single>::template accessor<rw> mf,
    typename field<h_s_t>::template accessor<ro, na> recv) noexcept {
    auto hmap = map(hcells);
    auto c = run::context::instance().color();
    for(std::size_t i = 0; i < mf->nents_recv_2; ++i) {
      auto key = recv[i].first.key();
      auto f = hmap.find(key);
      if(f == hmap.end()) {
        auto & cur = hmap.insert(key, key)->second;
        cur.set_nonlocal();
        cur.set_color(c);
        auto eid = mf->local.ents + mf->ghosts++;
        cur.set_ent_idx(eid);
        // Add missing parent(s)
        auto lastbit = key.pop();
        add_parent_distant(key, lastbit, c, hmap);
      }
      else {
        auto & cur = hmap.insert(key, key)->second;
        auto eid = mf->local.ents + mf->ghosts++;
        cur.set_nonlocal();
        cur.set_color(c);
        cur.set_ent_idx(eid);
      }
    }
  }

  // ----------------------------------- Share ghosts -------------------------
public:
  /// Search entities' neighbors and complete the hmap, create copy plans (or
  /// buffer)
  void share_ghosts(scheduler & s) {
    // Remove copy plan
    cp_entities.reset();

    // Find entities that will be used
    s.execute<find_local_task<true>>(*this, share_ghosts_comms_field(*this));

    {
      auto & p = get_partition<share_ghosts_cid_comm>();
      s.execute<copy_sizes_task>(p.sizes(), share_ghosts_comms_field(*this));
      p.resize();
    }

    s.execute<find_local_task<false>>(
      *this, share_ghosts_cid_comm_field(*this));

    s.execute<reset_ghosts>(meta_field(this->meta), hcells(*this));

    {
      auto & p = get_partition<entities>();
      s.execute<copy_sizes_resize_task<true>>(
        p.sizes(), meta_field(this->meta), share_ghosts_comms_field(*this));
      p.resize();
    }

    s.execute<xfer_entities_req_start>(e_i(*this),
      comms_field(*this),
      *(buf),
      share_ghosts_cid_comm_field(*this));
    while(
      s.reduce<xfer_entities_req, exec::fold::sum>(e_i(*this),
         meta_field(this->meta),
         comms_field(*this),
         *(buf),
         share_ghosts_cid_comm_field(*this),
         e_colors(*this))
        .get()) {
    } // while

    // Count all sizes for each color, use special field
    s.execute<find_distant_task<true>>(*this, share_ghosts_comms_field(*this));

    // Resize share_ghosts_buffer_comm_field
    {
      auto & p = get_partition<share_ghosts_buffer_comm>();
      s.execute<copy_sizes_task>(p.sizes(), share_ghosts_comms_field(*this));
      p.resize();
    }
    // Now fill info
    s.execute<find_distant_task<false>>(
      *this, share_ghosts_buffer_comm_field(*this));

    // Resize
    {
      auto & p = get_partition<share_ghosts_distant_buffer_comm>();
      s.execute<copy_sizes_resize_task<false>>(
        p.sizes(), meta_field(this->meta), share_ghosts_comms_field(*this));
      p.resize();
    }

    // Perform buffered copy
    s.execute<xfer_entities_cp_start>(
      comms_field(*this), *(buf), share_ghosts_buffer_comm_field(*this));
    while(
      s.reduce<xfer_entities_cp, exec::fold::sum>(
         share_ghosts_distant_buffer_comm_field(*this),
         meta_field(this->meta),
         comms_field(*this),
         *(buf),
         share_ghosts_buffer_comm_field(*this))
        .get())
      ;

    // Load entities sent here:
    s.execute<load_entities_task>(hcells(*this),
      meta_field(this->meta),
      share_ghosts_distant_buffer_comm_field(*this));

    {
      auto & p = get_partition<entities>();
      s.execute<resize_entities_update_meta_task>(
        p.sizes(), meta_field(this->meta));
      p.resize();
    }

    // create copy plan for ghosts entities
    auto entities_dests_task = [&](auto f) {
      s.execute<set_destination_meta>(f, meta_field(this->meta));
    };
    auto entities_ptrs_task = [&](auto f) {
      s.execute<set_entities_ptrs>(f,
        meta_field(this->meta),
        share_ghosts_distant_buffer_comm_field(*this));
    };

    // Merge the cp_top_tree_entities into the cp_entities to avoid copy plan on
    // the same index space
    cp_entities.emplace(s,
      *this,
      data::copy_plan::Sizes(colors(), 1),
      entities_dests_task,
      entities_ptrs_task,
      util::constant<entities>());

    // Fake initialization for the new ghosts
    for(auto & f : run::context::field_info_store<Policy, entities>()) {
      auto fr = data::field_reference<std::byte, data::raw, Policy, entities>(
        f->fid, *this);
      s.execute<fake_initialize>(fr);
    }
  }
  /// \deprecated Pass a \c scheduler.
  [[deprecated("pass a scheduler")]] static void share_ghosts(
    typename Policy::slot & ts) {
    ts->share_ghosts(*scheduler::instance);
  }

  static void fake_initialize(
    field<std::byte, data::raw>::accessor<rw, na>) noexcept {}

  //------------------------------ reset tree ---------------------------------
private:
  static void reset_task(
    typename field<meta_type, data::single>::template accessor<rw> mf,
    typename field<hmap_pair_t>::template accessor<wo, na> hm,
    typename field<node_data>::template accessor<wo, na> ni) noexcept {
    hmap_t hmap(hm.span());
    hmap.clear();
    mf->local.nodes = 0;
    mf->top_tree = 0;
    mf->ghosts = 0;
    mf->nents_recv = 0;
    std::fill(ni.span().begin(), ni.span().end(), node_data{});
  }

public:
  /// Reset the ntree topology. After this call the ntree can be re-build with
  /// make_tree and share_ghosts.
  void reset(scheduler & s) {
    s.execute<reset_task>(meta_field(this->meta), hcells(*this), n_i(*this));
    cp_top_tree_nodes.reset();
    cp_entities.reset();

    // Resize the entities
    auto & p = get_partition<entities>();
    std::vector<util::id> nents_rz(colors());
    s.execute<copy_sizes_meta_task>(p.sizes(), meta_field(this->meta));
    p.resize();
  }
  /// \deprecated Pass a \c scheduler.
  [[deprecated("pass a scheduler")]] static void reset(
    typename Policy::slot & ts) {
    ts->reset(*scheduler::instance);
  }

  //---------------------------------------------------------------------------
  template<typename Type,
    data::layout Layout,
    typename Topo,
    typename Topo::index_space Space>
  [[nodiscard]] const data::copy_plan * ghost_copy(scheduler &,
    data::field_reference<Type, Layout, Topo, Space> const &) {
    static_assert(Layout != data::ragged,
      "N-Tree does not support ragged or sparse fields");

    if constexpr(Space == entities) {
      // Need to check that the copy plan exists for the time it is being
      // re-created in share_ghosts
      if(cp_entities.has_value())
        return &*cp_entities;
    }
    else if constexpr(Space == nodes)
      return &*cp_top_tree_nodes;
    else if constexpr(Space == tree_data)
      return &cp_data_tree;
    else if constexpr(Space == share_ghosts_comms)
      return &cp_share_ghosts_comms;
    return nullptr;
  }

  // Get the number of colors
  Color colors() const {
    return part.front().colors();
  }

  template<index_space S>
  data::region & get_region() {
    return part.template get<S>();
  }

  template<index_space S>
  repartition & get_partition() {
    return part.template get<S>();
  }
};

/// See \ref specialization_base::interface
template<class Policy>
template<Privileges Priv>
struct topology<Policy, ntree_base>::access {
  template<class T>
  using accessor = typename field<T>::template accessor1<Priv>;
  accessor<key_t> e_keys, ///< Entities keys
    n_keys; ///< Nodes keys
  /// Entities Color
  accessor<Color> e_colors;
  /// Entities Id (for key collisions)
  accessor<util::id> e_ids;
  // Entities interaction fields
  accessor<entity_data> e_i;
  /// Nodes interaction fields
  accessor<node_data> n_i;

private:
  accessor<ntree_data> data_field;
  accessor<hmap_pair_t> hcells;
  data::scalar_access<meta_type, ro> mf;

public:
  template<class F>
  void send(F && f) {
    f(e_keys, topology::e_keys);
    f(n_keys, topology::n_keys);
    f(e_colors, topology::e_colors);
    f(e_ids, topology::e_ids);
    f(data_field, topology::data_field);
    f(hcells, topology::hcells);
    f(e_i, topology::e_i);
    f(n_i, topology::n_i);
    std::forward<F>(f)(mf, [](auto & n) { return meta_field(n.meta); });
  }

  /// Hashing table type
  using hmap_t = util::hashtable<key_t, hcell_t, Policy>;

#ifdef FLECSI_DEVICE_CODE
  using vector_type =
    util::inplace_vector<id<index_space::entities>, Policy::max_neighbors>;
  using queue_type = util::queue<hcell_t *, 1000>;
#else
  using vector_type = std::vector<id<index_space::entities>>;
  using queue_type = std::queue<hcell_t *>;
#endif

  // In order to avoid complexifying the hashtable class and since this usage is
  // strictly internal, we are using a const_cast to get an unprotected access
  // to the field.
  FLECSI_INLINE_TARGET hmap_t map() const {
    return util::span<hmap_pair_t>(
      const_cast<hmap_pair_t *>(hcells.span().data()), hcells.span().size());
  }

  // Standard traversal function
  template<typename F, typename HT>
  FLECSI_INLINE_TARGET void
  traversal(hcell_t * hcell, F && f, HT && hmap) const {
    queue_type tqueue;
    tqueue.push(hcell);
    while(!tqueue.empty()) {
      hcell_t * cur = tqueue.front();
      tqueue.pop();
      // Intersection
      if(f(cur) && cur->has_child()) {
        auto nkey = cur->key();
        for(std::size_t j = 0; j < nchildren_; ++j) {
          if(cur->has_child(j)) {
            tqueue.push(&hmap.at(nkey.push(j)));
          } // if
        } // for
      } // if
    } // while
  }

  template<bool C, typename T>
  void find_intersect_entities(T count_accessor) const {
    auto hmap = map();
    const auto cs = run::context::instance().colors();
    [[maybe_unused]] std::size_t count = 0;
    // Make a tree traversal per last elements in the intersection field.
    // Caution entities can be detected several time for the same neighbor.
    std::vector<std::set<hcell_t>> send_ids(cs);
    std::size_t start = mf->local.ents;
    std::size_t stop = start + mf->nents_recv;
    for(std::size_t i = start; i < stop; ++i) {
      ent_id id(i);
      auto tcolor = e_colors[i];
      assert(tcolor != run::context::instance().color());

      traversal(
        &hmap.at(key_t::root()),
        [&](hcell_t * cur) {
          if(cur->is_node()) {
            return Policy::intersect(
              e_i(id), n_i(topo::id<ntree_base::nodes>(cur->node_idx())));
          }
          else {
            // \todo add check here to see if the entities interact
            // For now, send a maximum of 8 entities
            if(cur->is_local())
              send_ids[tcolor].insert(*cur);
          }
          return false;
        },
        hmap);
    } // for
    // Add all the std::sets to the end vector to create the copy plan
    for(std::size_t i = 0; i < cs; ++i) {
      if constexpr(C)
        count_accessor[i] = send_ids[i].size();
      else {
        for(auto a : send_ids[i]) {
          assert(count < count_accessor.span().size());
          count_accessor[count++] = {a, i};
        }
      }
    }
  }

  template<bool C, typename T>
  void find_send_entities(T && count_accessor) const {
    std::optional<util::id> count;
    if constexpr(C)
      std::fill(count_accessor.span().begin(), count_accessor.span().end(), 0);
    else
      count = 0;

    auto hmap = map();
    for(std::size_t i = 0; i < mf->local.ents; ++i) {
      std::set<std::size_t> send_colors;
      ent_id id(i);
      traversal(
        &hmap.at(key_t::root()),
        [&](hcell_t * cur) {
          if(cur->is_node()) {
            bool intersect = Policy::intersect(
              e_i(id), n_i(topo::id<ntree_base::nodes>(cur->node_idx())));
            if(intersect) {
              if(cur->is_local()) {
                return true;
              }
              else {
                send_colors.insert(cur->color());
              }
            }
          } // if
          else {
            // \todo add check here to see if the entities interact
            // For now, send a maximum of 8 entities
            if(!cur->is_local() &&
               Policy::intersect(e_i(id), e_i(cur->ent_idx())))
              send_colors.insert(cur->color());
          }
          return false;
        },
        hmap);
      if constexpr(C)
        for(auto v : send_colors)
          count_accessor[v]++;
      else
        for(auto v : send_colors)
          count_accessor[(*count)++] =
            color_id{v, id, run::context::instance().color()};
    } // for
  }

  // --------------------------------------------------------------------------//
  //                                 ACCESSORS //
  // --------------------------------------------------------------------------//

  /// Return a range of all entities of a \c ntree_base::ptype_t
  template<ptype_t PT = ptype_t::exclusive>
  FLECSI_INLINE_TARGET auto entities() const {
    if constexpr(PT == ptype_t::exclusive) {
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(0, mf->local.ents));
    }
    else if constexpr(PT == ptype_t::ghost) {
      // Ghosts starts from local to end
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(mf->local.ents, mf->local.ents + mf->ghosts));
    }
    else {
      // Iterate on all
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(0, e_keys.span().size()));
    }
  }

  /// Get entities of a specific type under a node.
  template<ptype_t PT = ptype_t::exclusive>
  std::vector<id<index_space::entities>> entities(
    const id<index_space::nodes> & node_id) const {

    std::vector<id<index_space::entities>> ids;
    // Get the node and find its sub-entities
    auto nkey = n_keys[node_id];
    auto hmap = map();
    auto cur = &(hmap.find(nkey)->second);
    for(std::size_t j = 0; j < nchildren_; ++j) {
      if(cur->has_child(j)) {
        auto it = hmap.find(nkey.push(j));
        if(it->second.is_ent()) {
          ids.push_back(id<index_space::entities>(it->second.ent_idx()));
        }
      } // if
    } // for
    return ids;
  }

  /// Get entities interacting with an entity.
  /// This function uses the interaction functions featured in the policy.
  FLECSI_INLINE_TARGET auto neighbors(
    const id<index_space::entities> & ent_id) const {
    auto hmap = map();
    vector_type ids;
    // Perform tree traversal to find neighbors
    traversal(
      &hmap.at(key_t::root()),
      [&](hcell_t * cur) {
        if(cur->is_node())
          return Policy::intersect(
            e_i(ent_id), n_i(topo::id<ntree_base::nodes>(cur->node_idx())));
        else if(Policy::intersect(e_i(ent_id),
                  e_i(topo::id<ntree_base::entities>(cur->ent_idx()))))
          ids.push_back(topo::id<ntree_base::entities>(cur->ent_idx()));
        return false;
      },
      hmap);
    return ids;
  }

  /// Return a range of all nodes of a \c ntree_base::ptype_t
  template<ptype_t PT = ptype_t::exclusive>
  auto nodes() const {
    if constexpr(PT == ptype_t::exclusive)
      return make_ids<index_space::nodes>(
        util::iota_view<util::id>(0, mf->local.nodes));
    else if constexpr(PT == ptype_t::ghost)
      // Ghosts starts from local to end
      return make_ids<index_space::entities>(util::iota_view<util::id>(
        mf->local.nodes, mf->local.nodes + mf->top_tree));
    else
      // Iterate on all
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(0, mf->local.nodes + mf->top_tree));
  }

  /// Get nodes belonging to a node.
  std::vector<id<index_space::nodes>> nodes(
    const id<index_space::nodes> & node_id) const {
    std::vector<id<index_space::nodes>> ids;
    // Get the node and find its sub-entities
    auto nkey = n_keys[node_id];
    auto hmap = map();
    auto cur = &(hmap.find(nkey)->second);
    for(std::size_t j = 0; j < nchildren_; ++j) {
      if(cur->has_child(j)) {
        auto it = hmap.find(nkey.push(j));
        if(it->second.is_node()) {
          ids.push_back(id<index_space::nodes>(it->second.node_idx()));
        }
      } // if
    } // for
    return ids;
  }

  /// BFS traversal, return vector of ids in Breadth First Search order
  auto bfs() const {
    auto hmap = map();

    std::vector<id<index_space::nodes>> ids;
    std::queue<hcell_t *> tqueue;
    tqueue.push(&hmap.at(key_t::root()));
    ids.push_back(id<index_space::nodes>(0));

    while(!tqueue.empty()) {
      hcell_t * cur = tqueue.front();
      tqueue.pop();
      assert(cur->is_node());
      auto nkey = cur->key();
      for(std::size_t j = 0; j < nchildren_; ++j) {
        if(cur->has_child(j)) {
          auto it = hmap.find(nkey.push(j));
          if(it->second.is_node()) {
            ids.push_back(id<index_space::nodes>(it->second.idx()));
            tqueue.push(&it->second);
          }
        } // if
      } // for
    } // while
    return ids;
  } // bfs

  /// DFS traversal, return vector of ids in Depth First Search order
  /// \tparam complete Retrieve all completed nodes only: ignore non-local node.
  /// This is only valid while building the ntree.
  template<ttype_t TT = ttype_t::preorder, bool complete = false>
  auto dfs() const {

    auto hmap = map();
    std::vector<id<index_space::nodes>> ids;

    // Postorder and reverse postorder
    if constexpr(TT == ttype_t::postorder || TT == ttype_t::reverse_postorder) {

      std::stack<hcell_t *> stk;
      stk.push(&hmap.find(key_t::root())->second);

      while(!stk.empty()) {
        hcell_t * cur = stk.top();
        stk.pop();
        auto nkey = cur->key();
        for(std::size_t j = 0; j < nchildren_; ++j) {
          if(cur->has_child(j)) {
            auto it = hmap.find(nkey.push(j));
            if(it->second.is_node()) {
              if constexpr(complete) {
                if(it->second.is_complete()) {
                  ids.push_back(id<index_space::nodes>(cur->idx()));
                  stk.push(&it->second);
                }
              }
              else {
                ids.push_back(id<index_space::nodes>(cur->idx()));
                stk.push(&it->second);
              }
            }
          } // if
        } // for
      } // while
      if constexpr(TT == ttype_t::reverse_postorder) {
        std::reverse(ids.begin(), ids.end());
      }
    }
    // Preorder and reverse preorder
    else if constexpr(TT == ttype_t::preorder ||
                      TT == ttype_t::reverse_preorder) {
      std::stack<hcell_t *> stk;
      stk.push(&hmap.find(key_t::root())->second);

      while(!stk.empty()) {
        hcell_t * cur = stk.top();
        stk.pop();
        auto nkey = cur->key();
        if constexpr(complete) {
          if(cur->is_complete()) {
            ids.push_back(id<index_space::nodes>(cur->idx()));
          }
        }
        else {
          ids.push_back(id<index_space::nodes>(cur->idx()));
        }
        for(std::size_t j = 0; j < nchildren_; ++j) {
          const std::size_t child =
            nchildren_ - 1 - j; // Take children in reverse order
          if(cur->has_child(child)) {
            auto it = hmap.find(nkey.push(child));
            if(it->second.is_node())
              stk.push(&it->second);
          } // if
        } // for
      } // while
      if constexpr(TT == ttype_t::reverse_preorder) {
        std::reverse(ids.begin(), ids.end());
      }
    } // if
    return ids;
  } // dfs

#if defined(FLECSI_ENABLE_GRAPHVIZ)
  /// Output a representation of the ntree using graphviz.
  /// The output files are formatted as: color()_tag.gv
  /// \param tag Tag for these files names
  void graphviz_draw(const std::string & tag) const {
    util::graphviz gv("G");
    static constexpr std::pair<const char *, const char *> completeness[] = {
      {"complete", "octagon"}, {"incomplete", "doubleoctagon"}};
    static constexpr std::pair<const char *, const char *> locality[] = {
      {"non_local", "red"}, {"local", "blue"}};

    // Print a legend
    for(bool i : {true, false}) {
      for(bool j : {true, false}) {
        std::string name = std::string(completeness[j].first) + '_' +
                           locality[i].first + "_node";
        auto node = gv.add_node(name.c_str(),
          gv.html_label(
            (name + "<br/><FONT POINT-SIZE='10'>c(color)</FONT>").c_str()));
        gv.set_node_attribute(node, "xlabel", "#child");
        gv.set_node_attribute(node, "shape", completeness[j].second);
        gv.set_node_attribute(node, "color", locality[i].second);
      }
      std::string name = std::string(locality[i].first) + "_entity";
      auto node = gv.add_node(name.c_str(),
        gv.html_label(
          (name + "<br/><FONT POINT-SIZE=\"10\">c(color)</FONT>").c_str()));
      gv.set_node_attribute(node, "xlabel", "index");
      gv.set_node_attribute(node, "shape", "circle");
      gv.set_node_attribute(node, "color", locality[i].second);
    }

    std::queue<std::pair<hcell_t *, Agnode_t *>> stk;
    auto hmap = map();
    stk.push(std::pair(&(hmap.find(key_t::root())->second), nullptr));

    while(!stk.empty()) {
      auto [cur, parent] = stk.front();
      stk.pop();
      std::stringstream ss_name, ss_label, ss_xlabel;
      ss_name << cur->key();
      Agnode_t * node;
      if(cur->is_node()) {
        ss_label << cur->key() << "<br/><FONT POINT-SIZE=\"10\"> c("
                 << cur->color() << ")</FONT>";
        ss_xlabel << cur->nchildren();
        node = gv.add_node(
          ss_name.str().c_str(), gv.html_label(ss_label.str().c_str()));
        gv.set_node_attribute(node, "xlabel", ss_xlabel.str().c_str());
        gv.set_node_attribute(
          node, "shape", completeness[cur->is_complete()].second);
        gv.set_node_attribute(node, "color", locality[cur->is_local()].second);
        // Add the child to the stack and add for display
        for(std::size_t i = 0; i < nchildren_; ++i) {
          auto it = hmap.find(cur->key().push(i));
          if(it != hmap.end()) {
            stk.push(std::pair(&it->second, node));
          }
        }
      }
      else {
        ss_label << cur->key() << "<br/><FONT POINT-SIZE=\"10\"> c("
                 << cur->color() << ")</FONT>";
        ss_xlabel << cur->idx();
        node = gv.add_node(
          ss_name.str().c_str(), gv.html_label(ss_label.str().c_str()));
        gv.set_node_attribute(node, "xlabel", ss_xlabel.str().c_str());
        gv.set_node_attribute(node, "shape", "circle");
        gv.set_node_attribute(node, "color", locality[cur->is_local()].second);
      }
      if(parent != nullptr)
        gv.add_edge(parent, node);
    } // while
    std::ostringstream fname;
    fname << std::setfill('0') << std::setw(3)
          << run::context::instance().color() << "_" << tag << ".gv";
    gv.write(std::move(fname).str());
  }
#endif
};

/// Topology category.
template<class P>
using ntree = topology<P, ntree_base>;
template<>
struct detail::base<ntree> {
  using type = ntree_base;
};

#ifdef DOXYGEN
/// Example specialization which is not really implemented.
struct ntree_specialization : specialization<ntree, ntree_specialization> {
  /// Dimension of the N-Tree. It can be 1, 2 or 3 for a binary tree, quadtree
  /// or octree respectively.
  static constexpr unsigned int dimension = 1;
  /// Specify the space filling curves to use in the domain and the N-Tree
  /// construction. The provided class must derive from the CRTP
  /// filling_curve_key.
  using key_t = flecsi::morton_curve<dimension, std::size_t>;
  /// Hashing function used in the hashtable to distribute the keys.
  static std::size_t hash(key_t k) {}
  /// Struct containing data for entities to compute interactions
  struct entity_data {};
  /// Struct containing data for nodes to compute interactions
  struct node_data {};
  /// Maximum number of neighbors per entities. This is used to compute
  /// neighbors list on GPU architectures.
  constexpr static unsigned int max_neighbors = 42;
  /// Mandatory index spaces featured by the N-Tree interface.
  /// This allows the access to the two index spaces entities and nodes.
  using index_spaces = base::index_spaces;

  /// \name Intersection Functions
  /// Function computing interation between entity-entity, entity-node and
  /// node-node. Returns true if there is an interaction. A possible
  /// implementation is to use a templated function.
  /// \{

  /// <a></a>
  static bool intersect(entity_data, entity_data) {}
  static bool intersect(entity_data, node_data) {}
  static bool intersect(node_data, node_data) {}
  /// \}
};
#endif

/// \}
} // namespace topo
} // namespace flecsi

#endif
