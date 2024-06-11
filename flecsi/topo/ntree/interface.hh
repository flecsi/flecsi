// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_NTREE_INTERFACE_HH
#define FLECSI_TOPO_NTREE_INTERFACE_HH

#include "flecsi/data/accessor.hh"
#include "flecsi/data/copy_plan.hh"
#include "flecsi/execution.hh"
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

template<typename T>
class serdez_vector : public std::vector<T>
{
public:
  inline std::size_t legion_buffer_size(void) const {
    return util::serial::size(static_cast<const std::vector<T> &>(*this));
  }

  inline void legion_serialize(void * buffer) const {
    std::byte * p = static_cast<std::byte *>(buffer);
    util::serial::put(p, static_cast<const std::vector<T> &>(*this));
  }

  inline void legion_deserialize(const void * buffer) {
    const std::byte * p = static_cast<const std::byte *>(buffer);
    auto v = util::serial::get1<std::vector<T>>(p);
    this->swap(v);
  }
};

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
/// \see [The N-Tree tutorial](../../src/tutorial/ntree.html)
template<typename Policy>
struct ntree : ntree_base, with_meta<Policy> {

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
  ntree(const coloring & c)
    : with_meta<Policy>(c.nparts_),
      part{{make_repartitioned<Policy, entities>(c.nparts_,
              make_partial<allocate>(c.entities_sizes_)),
        make_repartitioned<Policy, nodes>(c.nparts_,
          make_partial<allocate>(c.nodes_sizes_)),
        make_repartitioned<Policy, hashmap>(c.nparts_,
          make_partial<allocate>(
            std::vector<util::id>(c.nparts_, c.local_hmap_))),
        make_repartitioned<Policy, tree_data>(c.nparts_,
          make_partial<allocate>(std::vector<util::id>(c.nparts_, 3))),
        make_repartitioned<Policy, comms>(c.nparts_,
          make_partial<allocate>(
            std::vector<util::id>(c.nparts_, c.nparts_)))}},
      cp_data_tree(*this,
        // Avoid initializer-list constructor:
        data::copy_plan::Sizes(c.nparts_, 1),
        task<set_dests>,
        task<set_ptrs>,
        util::constant<tree_data>()),
      buf([] {
        const auto p = processes();
        data::buffers::coloring ret(p);
        for(std::size_t i_r = 0; i_r < ret.size(); ++i_r) {
          for(std::size_t i = 0; i < processes(); ++i) {
            if(i != i_r) {
              ret[i_r].push_back(i);
            }
          }
        }
        return ret;
      }()) {
    // Initialize the meta_field
    flecsi::execute<init_meta_field>(meta_field(this->meta));
  }

  // Ntree mandatory fields ---------------------------------------------------
public:
  /// Entities keys field
  static inline const typename field<key_t>::template definition<Policy,
    entities>
    e_keys;
  /// Entities ids field. This field can be used to identify entities with the
  /// same key.
  static inline const typename field<util::id>::template definition<Policy,
    entities>
    e_ids;
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
  static inline const typename field<util::id>::template definition<Policy,
    comms>
    comms_field;

  // --------------------------------------------------------------------------

  // Index space index
  util::key_array<repartitioned, index_spaces> part;

  // Copy plan for the tree data field
  data::copy_plan cp_data_tree;
  std::optional<data::copy_plan> cp_top_tree_nodes, cp_entities;

  // Buffer for ghosts shared
  data::buffers::core buf;
  static const util::id buffer_size =
    (data::buffers::Buffer::size / sizeof(entity_data)) * 2;

  ntree_base::en_size rz, sz;

  // Save top tree information to merge copy plans
  static inline std::vector<hcell_t> top_tree = {};

  /// Hashing table type
  using hmap_t = util::hashtable<ntree::key_t, ntree::hcell_t, Policy>;

  FLECSI_INLINE_TARGET static hmap_t map(
    typename field<hmap_pair_t>::template accessor<rw, na> hcells) {
    return hcells.span();
  }

  static void init_meta_field(
    typename field<meta_type, data::single>::template accessor<wo> meta_field) {
    meta_field = {};
  }

  // ----------------------- Top Tree Construction Tasks -----------------------
  // Build the local tree.
  // Add the entities in the hashmap and create needed nodes.
  // After this first step, the top tree entities and nodes are returned. These
  // will build the top tree, shared by all colors.
  static auto make_tree_local_task(
    typename field<key_t>::template accessor<rw, na> e_keys,
    typename field<key_t>::template accessor<rw, na> n_keys,
    typename field<ntree_data>::template accessor<ro, na> data_field,
    typename field<hmap_pair_t>::template accessor<rw, na> hcells,
    typename field<meta_type, data::single>::template accessor<rw> meta_field) {
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
    flog_assert(hibound >= e_keys(meta_field->local.ents - 1),
      "The keys are not globally sorted");

    // Add the root
    hmap.insert(key_t::root(), key_t::root());
    auto root_ = hmap.find(key_t::root());
    root_->second.set_color(color);
    {
      const std::size_t cnode = meta_field->local.nodes++;
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
      meta_field->max_depth = std::max(meta_field->max_depth, current_depth);

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
        std::size_t cnode = meta_field->local.nodes++;
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
    return top_tree_boundaries(hmap);
  } // make_tree

  // Count number of entities to send to each other color
  static auto top_tree_boundaries(hmap_t & hmap) {
    serdez_vector<hcell_t> sdata;
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
          sdata.emplace_back(*cur);
        } // else
      } // for
      queue = std::move(nqueue);
      nqueue.clear();
    } // while
    return sdata;
  } // top_tree_boundaries

  // Add missing parent from distant node/entity
  // This version does add the parents and add an idx
  // This can only be done before any ghosts are received
  static void add_parent(key_t key,
    typename key_t::int_t child,
    const int & color,
    hmap_t & hmap,
    typename field<meta_type, data::single>::template accessor<rw> meta_field,
    typename field<key_t>::template accessor<rw, na> n_keys) {
    auto parent = hmap.end();
    while((parent = hmap.find(key)) == hmap.end()) {
      parent = hmap.insert(key, key);
      const std::size_t cnode = meta_field->local.nodes++;
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
    typename field<meta_type, data::single>::template accessor<rw> meta_field,
    typename field<key_t>::template accessor<rw, na> n_keys) {
    auto key = k;
    auto f = hmap.find(key);
    if(f == hmap.end()) {
      auto & cur = hmap.insert(key, key)->second;
      cur.set_nonlocal();
      cur.set_color(c);
      auto eid = meta_field->local.ents + meta_field->ghosts++;
      cur.set_ent_idx(eid);
      // Add missing parent(s)
      auto lastbit = key.pop();
      add_parent(key, lastbit, c, hmap, meta_field, n_keys);
    }
    else {
      assert(false);
    }
  }

  static void load_shared_node(const std::size_t & c,
    const key_t & k,
    hmap_t & hmap,
    typename field<meta_type, data::single>::template accessor<rw> meta_field,
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
      add_parent(key, lastbit, c, hmap, meta_field, n_keys);
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
    typename field<meta_type, data::single>::template accessor<rw> meta_field,
    typename field<ntree_data>::template accessor<rw, na> data_field,
    typename field<hmap_pair_t>::template accessor<rw, na> hcell,
    const std::vector<hcell_t> & cells) {

    auto hmap = map(hcell);
    const auto color = run::context::instance().color();
    for(auto c : cells) {
      if(c.color() == color)
        continue;
      if(c.is_ent()) {
        load_shared_entity(c.color(), c.key(), hmap, meta_field, n_keys);
      }
      else {
        load_shared_node(c.color(), c.key(), hmap, meta_field, n_keys);
      }
    }
    // Update the lo and hi bounds
    data_field(0).lobound = key_t::min();
    data_field(0).hibound = key_t::max();

    // Add the distant nodes, at the end
    for(auto c : cells) {
      if(c.color() == color)
        continue;
      if(c.is_node()) {
        assert(n_keys.span().size() >
               meta_field->local.nodes + meta_field->top_tree + 1);
        auto cur = hmap.find(c.key());
        assert(cur != hmap.end());
        const std::size_t cnode =
          meta_field->local.nodes + meta_field->top_tree++;
        cur->second.set_node_idx(cnode);
        n_keys(cnode) = cur->second.key();
      }
    }
  }

  // Return the number of entities/nodes for local, top tree and ghosts
  // stores in the respective index spaces.
  static std::array<util::id, 4> sizes_task(
    typename field<meta_type, data::single>::template accessor<ro> m) {
    return {{m->local.ents, m->local.nodes, m->top_tree, m->ghosts}};
  }

  static void reset_ghosts(
    typename field<meta_type, data::single>::template accessor<rw> m) {
    m->ghosts = 0;
  }

  // Copy plan: set destination sizes
  static void set_destination(field<data::intervals::Value>::accessor<wo> a,
    const std::vector<util::id> & base,
    const std::vector<util::id> & total) {
    auto i = process();
    a(0) = data::intervals::make({base[i], base[i] + total[i]}, i);
  }

  // Copy plan: set pointers for top tree
  template<index_space IS = entities>
  static void set_top_tree_ptrs(field<data::points::Value>::accessor<wo, wo> a,
    const std::vector<util::id> & base,
    const std::vector<hcell_t> & hcells) {
    auto i = process();
    util::id idx = base[i];
    for(std::size_t j = 0; j < hcells.size(); ++j) {
      auto & h = hcells[j];
      if((IS == entities ? h.is_ent() : h.is_node()) && h.color() != i)
        a(idx++) = data::points::make(h.color(), h.idx());
    }
  }

  // Copy plan: set pointers to entities
  static void set_entities_ptrs(field<data::points::Value>::accessor<wo, wo> a,
    const std::vector<util::id> & nents_base,
    const std::vector<std::pair<hcell_t, std::size_t>> & ids) {
    auto i = process();
    util::id idx = nents_base[i];
    for(std::size_t j = 0; j < ids.size(); ++j) {
      assert(ids[j].first.color() != i);
      a(idx++) = data::points::make(ids[j].first.color(), ids[j].first.idx());
    }
  }

  // Exchange the boundaries from the local tree to other colors.
  // This shares part of the top tree information.
  static void exchange_boundaries(
    typename field<key_t>::template accessor<ro, na> e_keys,
    typename field<meta_type, data::single>::template accessor<rw> m,
    typename field<ntree_data>::template accessor<rw, na> df) {
    auto c = run::context::instance().color();
    m->max_depth = 0;
    m->local.ents = e_keys.span().size();
    df(0).lobound = c == 0 ? key_t::min() : e_keys(0);
    df(0).hibound = c == run::context::instance().colors() - 1
                      ? key_t::max()
                      : e_keys(m->local.ents - 1);
  }

  // Recolor the entities, some might have moved after the sort.
  static void recolor_task(typename Policy::template accessor<rw, wo> t) {
    for(auto & v : t.e_colors.span()) {
      v = run::context::instance().color();
    }
  }

  // ---------------------------- Top tree construction -----------------------
public:
  /// Build the local tree and share the tree boundaries
  static void make_tree(typename Policy::slot & ts) {
    //  Sort entities
    util::sort(e_keys(ts))();
    flecsi::execute<recolor_task>(ts);
    flecsi::execute<exchange_boundaries>(
      e_keys(ts), meta_field(ts->meta), data_field(ts));

    const auto cs = ts->colors();
    ts->cp_data_tree.issue_copy(data_field.fid);

    // Create the local tree
    // Return the list of nodes to share (top of the tree)
    auto fm_top_tree = flecsi::execute<make_tree_local_task>(
      e_keys(ts), n_keys(ts), data_field(ts), hcells(ts), meta_field(ts->meta));

    // Merge all the hcells informations for the top tree
    top_tree.clear();
    std::vector<util::id> top_tree_nnodes(cs, 0);
    std::vector<util::id> top_tree_nents(cs, 0);
    fm_top_tree.wait();

    for(Color i = 0; i < cs; ++i) {
      auto f = fm_top_tree.get(i);
      top_tree.insert(top_tree.end(), f.begin(), f.end());
    }

    for(Color c = 0; c < cs; ++c) {
      for(std::size_t j = 0; j < top_tree.size(); ++j) {
        if(top_tree[j].color() != c) {
          if(top_tree[j].is_ent())
            ++top_tree_nents[c];
          else
            ++top_tree_nnodes[c];
        }
      }
    }

    // Add the new hcells to the local tree + return new sizes for allocation
    flecsi::execute<make_tree_distributed_task>(
      n_keys(ts), meta_field(ts->meta), data_field(ts), hcells(ts), top_tree);
    auto fm_sizes = flecsi::execute<sizes_task>(meta_field(ts->meta));

    ts->sz.ent.resize(cs);
    ts->sz.node.resize(cs);
    ts->rz.node.resize(cs);
    ts->rz.ent.resize(cs);

    for(std::size_t i = 0; i < fm_sizes.size(); ++i) {
      auto f = fm_sizes.get(i);
      ts->sz.ent[i] = f[0];
      ts->sz.node[i] = f[1];
    }

    for(std::size_t i = 0; i < cs; ++i) {
      ts->rz.ent[i] = ts->sz.ent[i] + top_tree_nents[i];
      ts->rz.node[i] = ts->sz.node[i] + top_tree_nnodes[i];
    }

    // Properly resize the partitions for the new number of ents + ghosts
    ts->part.template get<entities>().resize(
      make_partial<allocate>(ts->rz.ent));

    // Fake initialization for the new ghosts
    for(auto & f :
      run::context::instance().field_info_store<Policy, entities>()) {
      auto fr = data::field_reference<std::byte, data::raw, Policy, entities>(
        f->fid, ts.get());
      execute<fake_initialize>(fr);
    }

    ts->cp_entities.emplace(
      ts.get(),
      data::copy_plan::Sizes(processes(), 1),
      [&](auto f) { execute<set_destination>(f, ts->sz.ent, top_tree_nents); },
      [&](auto f) {
        execute<set_top_tree_ptrs<entities>>(f, ts->sz.ent, top_tree);
      },
      util::constant<entities>());

    ts->cp_top_tree_nodes.emplace(
      ts.get(),
      data::copy_plan::Sizes(processes(), 1),
      [&](
        auto f) { execute<set_destination>(f, ts->sz.node, top_tree_nnodes); },
      [&](auto f) {
        execute<set_top_tree_ptrs<nodes>>(f, ts->sz.node, top_tree);
      },
      util::constant<nodes>());
  }

  // ---------------------------- Ghosts exchange tasks -----------------------
private:
  static void xfer_entities_req_start(
    typename field<entity_data>::template accessor<rw, na> a,
    typename field<util::id>::template accessor<wo, na> restart,
    data::buffers::Start mv,
    const std::vector<color_id> & f) {
    std::fill(restart.span().begin(), restart.span().end(), 0);
    util::id cur = 0;
    const auto color = run::context::instance().color();
    for(Color c = 0; c < run::context::instance().colors(); ++c) {
      if(c != color) {
        auto w = mv[cur].write();
        for(std::size_t i = 0; i < f.size(); ++i) {
          if(f[i].color == c) {
            if(!w(a(f[i].id))) {
              restart(cur) = i;
              break;
            } // if
          } // if
        } // for
        ++cur;
      } // if
    } // for
  } // xfer_nodes_req_start

  static auto xfer_entities_req(
    typename field<entity_data>::template accessor<rw, na> a,
    typename field<meta_type, data::single>::template accessor<rw> m,
    typename field<util::id>::template accessor<rw, na> restart,
    data::buffers::Transfer mv,
    const std::vector<color_id> & f,
    typename field<Color>::template accessor<rw, na> e_c) {

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

    // Keep copy if needed
    bool more_to_copy = false;
    cur = 0;
    for(Color c = 0; c < cs; ++c) {
      if(c != color) {
        bool done = true;
        if(restart(cur) != 0) {
          auto w = mv[cur].write();
          for(std::size_t i = restart(c); i < f.size(); ++i) {
            if(f[i].color == c) {
              if(!w(a(f[i].id))) {
                restart(cur) = i;
                done = false;
                more_to_copy = true;
                break;
              }
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

  static serdez_vector<color_id> find_task(
    typename Policy::template accessor<rw, na> t) {
    return t.find_send_entities();
  }

  static serdez_vector<std::pair<hcell_t, std::size_t>> find_distant_task(
    typename Policy::template accessor<rw, na> t) {
    return t.find_intersect_entities();
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
    typename field<meta_type, data::single>::template accessor<rw> meta_field,
    const std::vector<std::pair<hcell_t, std::size_t>> & recv) {
    auto hmap = map(hcells);
    auto c = run::context::instance().color();
    for(std::size_t i = 0; i < recv.size(); ++i) {
      auto key = recv[i].first.key();
      auto f = hmap.find(key);
      if(f == hmap.end()) {
        auto & cur = hmap.insert(key, key)->second;
        cur.set_nonlocal();
        cur.set_color(c);
        auto eid = meta_field->local.ents + meta_field->ghosts++;
        cur.set_ent_idx(eid);
        // Add missing parent(s)
        auto lastbit = key.pop();
        add_parent_distant(key, lastbit, c, hmap);
      }
      else {
        auto & cur = hmap.insert(key, key)->second;
        auto eid = meta_field->local.ents + meta_field->ghosts++;
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
  static void share_ghosts(typename Policy::slot & ts) {

    // Remove copy plan
    ts->cp_entities.reset();

    // Find entities that will be used
    auto to_send = flecsi::execute<find_task>(ts);
    // Reset ghosts and get current sizes
    flecsi::execute<reset_ghosts>(meta_field(ts->meta));
    auto fm_sizes = flecsi::execute<sizes_task>(meta_field(ts->meta));

    std::vector<util::id> ents_sizes_rz(ts->colors());
    // Resize, add by the max size capability of the buffer
    for(Color c = 0; c < ts->colors(); ++c) {
      ents_sizes_rz[c] = fm_sizes.get(c)[0] + buffer_size;
    }

    ts->part.template get<entities>().resize(
      make_partial<allocate>(ents_sizes_rz));

    // Perform buffered copy
    auto full = 1;
    execute<xfer_entities_req_start>(
      e_i(ts.get()), comms_field(ts.get()), *(ts->buf), to_send.get(process()));
    while(reduce<xfer_entities_req, exec::fold::sum>(e_i(ts.get()),
      meta_field(ts->meta),
      comms_field(ts.get()),
      *(ts->buf),
      to_send.get(process()),
      e_colors(ts.get()))
            .get()) {
      // Size of IS is twice buffer size
      if(!(++full % 2)) {
        // Resize the partition if the maximum size is reached
        for(Color c = 0; c < ts->colors(); ++c) {
          ents_sizes_rz[c] += buffer_size;
        }

        ts->part.template get<entities>().resize(
          make_partial<allocate>(ents_sizes_rz));
      } // if
    } // while

    // Find the interactions with the extra particles locally
    auto to_reply = flecsi::execute<find_distant_task>(ts);

    // Compute which entities are destinated for this process
    // This is not efficient since it is performed on the top level task and is
    // not an all to all operation recv contains the hcell but also from which
    // color it is received from
    std::vector<std::pair<hcell_t, std::size_t>> recv;
    for(Color c = 0; c < ts->colors(); ++c) {
      auto tr = to_reply.get(c);
      for(auto t : tr) {
        if(t.second == process()) {
          recv.push_back(t);
        }
      }
    }

    // Load entities destinated for this rank
    flecsi::execute<load_entities_task>(hcells(ts), meta_field(ts->meta), recv);

    // Find the current sizes and how much needs to be resized
    std::vector<util::id> nents_base(processes());
    std::vector<util::id> nents_tt(processes());
    std::vector<util::id> nents_rz(processes());

    for(Color c = 0; c < ts->colors(); ++c) {
      auto f = fm_sizes.get(c);
      nents_base[c] = f[0];
      if(c == process()) {
        nents_tt[c] = recv.size();
        nents_rz[c] = nents_base[c] + recv.size();
      }
    }

    // Resize partitions
    ts->part.template get<entities>().resize(make_partial<allocate>(nents_rz));

    // create copy plan for ghosts entities
    auto entities_dests_task = [&nents_base, &nents_tt](auto f) {
      execute<set_destination>(f, nents_base, nents_tt);
    };
    auto entities_ptrs_task = [&nents_base, &recv](auto f) {
      execute<set_entities_ptrs>(f, nents_base, recv);
    };

    // Merge the cp_top_tree_entities into the cp_entities to avoid copy plan on
    // the same index space
    ts->cp_entities.emplace(ts.get(),
      data::copy_plan::Sizes(processes(), 1),
      entities_dests_task,
      entities_ptrs_task,
      util::constant<entities>());

    // Fake initialization for the new ghosts
    for(auto & f :
      run::context::instance().field_info_store<Policy, entities>()) {
      auto fr = data::field_reference<std::byte, data::raw, Policy, entities>(
        f->fid, ts.get());
      execute<fake_initialize>(fr);
    }
  }

  static void fake_initialize(field<std::byte, data::raw>::accessor<rw, na>) {}

  //------------------------------ reset tree ---------------------------------
private:
  static void reset_task(
    typename field<meta_type, data::single>::template accessor<rw> mf,
    typename field<hmap_pair_t>::template accessor<wo, na> hm,
    typename field<node_data>::template accessor<wo, na> ni) {
    hmap_t hmap(hm.span());
    hmap.clear();
    mf->max_depth = 0;
    mf->local.nodes = 0;
    mf->top_tree = 0;
    mf->ghosts = 0;
    mf->nents_recv = 0;
    std::fill(ni.span().begin(), ni.span().end(), node_data{});
  }

public:
  /// Reset the ntree topology. After this call the ntree can be re-build with
  /// make_tree and share_ghosts.
  static void reset(typename Policy::slot & ts) {
    flecsi::execute<reset_task>(meta_field(ts->meta), hcells(ts), n_i(ts));
    ts->cp_top_tree_nodes.reset();
    ts->cp_entities.reset();

    // Resize the entities
    std::vector<util::id> nents_rz(ts->colors());
    auto fm_sizes = flecsi::execute<sizes_task>(meta_field(ts->meta));
    for(Color c = 0; c < ts->colors(); ++c) {
      auto f = fm_sizes.get(c);
      nents_rz[c] = f[0];
    }
    ts->part.template get<entities>().resize(make_partial<allocate>(nents_rz));
  }

  //---------------------------------------------------------------------------
  template<typename Type,
    data::layout Layout,
    typename Topo,
    typename Topo::index_space Space>
  void ghost_copy(data::field_reference<Type, Layout, Topo, Space> const & f) {
    static_assert(Layout != data::ragged,
      "N-Tree does not support ragged or sparse fields");

    if constexpr(Space == entities) {
      // Need to check that the copy plan exists for the time it is being
      // re-created in share_ghosts
      if(cp_entities.has_value())
        cp_entities->issue_copy(f.fid());
    }
    else if constexpr(Space == nodes) {
      cp_top_tree_nodes->issue_copy(f.fid());
    }
    else if constexpr(Space == tree_data) {
      cp_data_tree.issue_copy(f.fid());
    }
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
struct ntree<Policy>::access {
  template<const auto & F>
  using accessor = data::accessor_member<F, Priv>;
  /// Entities keys
  accessor<ntree::e_keys> e_keys;
  /// Entities Color
  accessor<ntree::e_colors> e_colors;
  /// Entities Id (for key collisions)
  accessor<ntree::e_ids> e_ids;
  /// Nodes keys
  accessor<ntree::n_keys> n_keys;
  // Entities interaction fields
  accessor<ntree::e_i> e_i;
  /// Nodes interaction fields
  accessor<ntree::n_i> n_i;

private:
  accessor<ntree::data_field> data_field;
  accessor<ntree::hcells> hcells;
  data::scalar_access<ntree::meta_field, privilege_pack<ro>> meta_field;

public:
  template<class F>
  void send(F && f) {
    e_keys.topology_send(f);
    n_keys.topology_send(f);
    e_colors.topology_send(f);
    e_ids.topology_send(f);
    data_field.topology_send(f);
    hcells.topology_send(f);
    e_i.topology_send(f);
    n_i.topology_send(f);
    meta_field.topology_send(
      std::forward<F>(f), [](auto & n) -> auto & { return n.meta; });
  }

  /// Hashing table type
  using hmap_t = util::hashtable<ntree::key_t, ntree::hcell_t, Policy>;

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
    return util::span<ntree::hmap_pair_t>(
      const_cast<ntree::hmap_pair_t *>(hcells.span().data()),
      hcells.span().size());
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
      if(f(cur)) {
        if(cur->has_child()) {
          auto nkey = cur->key();
          for(std::size_t j = 0; j < nchildren_; ++j) {
            if(cur->has_child(j)) {
              tqueue.push(&hmap.at(nkey.push(j)));
            } // if
          } // for
        }
      }
    } // while
  }

  auto find_intersect_entities() const {
    auto hmap = map();
    const auto cs = run::context::instance().colors();
    serdez_vector<std::pair<hcell_t, std::size_t>> entities;

    // Make a tree traversal per last elements in the intersection field.
    // Caution entities can be detected several time for the same neighbor.
    std::vector<std::set<hcell_t>> send_ids(cs);
    std::size_t start = meta_field->local.ents;
    std::size_t stop = start + meta_field->nents_recv;
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
            if(cur->is_local()) {
              send_ids[tcolor].insert(*cur);
            }
          }
          return false;
        },
        hmap);
    } // for
    // Add all the std::sets to the end vector to create the copy plan
    for(std::size_t i = 0; i < cs; ++i) {
      for(auto a : send_ids[i]) {
        entities.push_back({a, i});
      }
    }
    return entities;
  }

  auto find_send_entities() const {
    // The ranks to send and the id of the entity
    serdez_vector<color_id> entities;
    auto hmap = map();
    // 1. for all local entities
    for(std::size_t i = 0; i < meta_field->local.ents; ++i) {
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
      for(auto v : send_colors) {
        entities.push_back(color_id{v, id, run::context::instance().color()});
      } // for
    } // for
    return entities;
  }

  // --------------------------------------------------------------------------//
  //                                 ACCESSORS //
  // --------------------------------------------------------------------------//

  /// Return a range of all entities of a \c ntree_base::ptype_t
  template<ptype_t PT = ptype_t::exclusive>
  FLECSI_INLINE_TARGET auto entities() const {
    if constexpr(PT == ptype_t::exclusive) {
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(0, meta_field->local.ents));
    }
    else if constexpr(PT == ptype_t::ghost) {
      // Ghosts starts from local to end
      return make_ids<index_space::entities>(util::iota_view<util::id>(
        meta_field->local.ents, e_keys.span().size()));
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
  FLECSI_INLINE_TARGET
  auto neighbors(const id<index_space::entities> & ent_id) const {
    auto hmap = map();
    vector_type ids;
    // Perform tree traversal to find neighbors
    traversal(
      &hmap.at(key_t::root()),
      [&](hcell_t * cur) {
        if(cur->is_node()) {
          return Policy::intersect(
            e_i(ent_id), n_i(topo::id<ntree_base::nodes>(cur->node_idx())));
        }
        else {
          if(Policy::intersect(e_i(ent_id),
               e_i(topo::id<ntree_base::entities>(cur->ent_idx())))) {
            ids.push_back(topo::id<ntree_base::entities>(cur->ent_idx()));
          }
        }
        return false;
      },
      hmap);
    return ids;
  }

  /// Return a range of all nodes of a \c ntree_base::ptype_t
  template<ptype_t PT = ptype_t::exclusive>
  auto nodes() const {
    if constexpr(PT == ptype_t::exclusive) {
      return make_ids<index_space::nodes>(
        util::iota_view<util::id>(0, meta_field->local.nodes));
    }
    else if constexpr(PT == ptype_t::ghost) {
      // Ghosts starts from local to end
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(meta_field->local.nodes,
          meta_field->local.nodes + meta_field->top_tree));
    }
    else {
      // Iterate on all
      return make_ids<index_space::entities>(util::iota_view<util::id>(
        0, meta_field->local.nodes + meta_field->top_tree));
    }
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
  /// The output files are formatted as: process()_tag.gv
  /// \param tag Tag for these files names
  void graphviz_draw(const std::string & tag) const {
    util::graphviz gv("G");
    static constexpr std::pair<const char *, const char *> completeness[] = {
      {"incomplete", "doubleoctagon"},
      { "complete",
        "octagon" }};
    static constexpr std::pair<const char *, const char *> locality[] = {
      {"local", "blue"},
      { "non_local",
        "red" }};

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
    fname << std::setfill('0') << std::setw(3) << process() << "_" << tag
          << ".gv";
    gv.write(std::move(fname).str());
  }
#endif

}; // namespace topo

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

  /// \name Intersection Functions
  /// Function computing interation between entity-entity, entity-node and
  /// node-node. Returns true if there is an interaction. A possible
  /// implementation is to use a templated function.
  /// \{
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
