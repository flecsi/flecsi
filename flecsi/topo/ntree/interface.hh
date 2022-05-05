// Copyright (c) 2016, Triad National Security, LLC
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
#include "flecsi/util/hashtable.hh"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <stack>
#include <type_traits>
#include <unordered_map>

/// \cond core
namespace flecsi {
namespace topo {

template<typename T>
class serdez_vector : public std::vector<T>
{
public:
  inline size_t legion_buffer_size(void) const {
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
/// \ingroup topology
/// \{

/// The ntree topology represents a binary, quad or octree stored/accessed using
/// a hashtable
/// \tparam Policy the specialization, which must define the following:
/// information such as:
///         - \c dimension : the Dimension of the ntree: 1, 2 or 3 for Binary,
///         Quad or Oct tree
///         - \c key_int_t :  Integer type used for the hashing table key
///         - \c key_t : Key type for the hashing table
///         - \c node_t : Node type for the ntree
///         - \c ent_t : Entity type for the ntree
///         - \c hash_f : Hashing function used in the hashing table
///         - \c interaction_entities : Function defining if two entities
///         interact
///         - \c interaction_nodes : Function defining if two nodes interact
template<typename Policy>
struct ntree : ntree_base, with_meta<Policy> {

private:
  constexpr static unsigned int dimension = Policy::dimension;
  using key_int_t = typename Policy::key_int_t;
  using key_t = typename Policy::key_t;
  using node_t = typename Policy::node_t;
  using ent_t = typename Policy::ent_t;
  using hash_f = typename Policy::hash_f;

  using type_t = double;
  using hcell_t = hcell_base_t<dimension, type_t, key_t>;

  using interaction_entities = typename Policy::interaction_entities;
  using interaction_nodes = typename Policy::interaction_nodes;

  struct ntree_data {
    key_t hibound, lobound;
  };

public:
  template<Privileges>
  struct access;

  ntree(const coloring & c)
    : with_meta<Policy>(c.nparts_),
      part{{make_repartitioned<Policy, entities>(c.nparts_,
              make_partial<allocate>(c.entities_offset_)),
        make_repartitioned<Policy, nodes>(c.nparts_,
          make_partial<allocate>(c.nodes_offset_)),
        make_repartitioned<Policy, hashmap>(c.nparts_,
          make_partial<allocate>(c.hmap_offset_)),
        make_repartitioned<Policy, tree_data>(c.nparts_,
          make_partial<allocate>(c.tdata_offset_)),
        make_repartitioned<Policy, meta>(c.nparts_,
          make_partial<allocate>(c.tdata_offset_)),
        make_repartitioned<Policy, comms>(c.nparts_,
          make_partial<allocate>(c.comms_offset_))}},
      cp_data_tree(*this,
        part.template get<tree_data>(),
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
      }()) {}

  // Ntree mandatory fields ---------------------------------------------------
public:
  // Entities keys field
  static inline const typename field<key_t>::template definition<Policy,
    entities>
    e_keys;
  // Node keys field
  static inline const typename field<key_t>::template definition<Policy, nodes>
    n_keys;

  static inline const typename field<
    interaction_entities>::template definition<Policy, entities>
    e_i;
  static inline const typename field<
    interaction_nodes>::template definition<Policy, nodes>
    n_i;

private:
  // Hmap fields,  the hashing table are reconstructed based on this field
  static inline const typename field<
    std::pair<key_t, hcell_t>>::template definition<Policy, hashmap>
    hcells;
  // Tree data field
  static inline const typename field<ntree_data>::template definition<Policy,
    tree_data>
    data_field;

  static inline const typename field<meta_type>::template definition<Policy,
    meta>
    meta_field;

  static inline const typename field<std::size_t>::template definition<Policy,
    comms>
    comms_field;

  // --------------------------------------------------------------------------

  // Index space index
  util::key_array<repartitioned, index_spaces> part;

  // Copy plan for the tree data field
  data::copy_plan cp_data_tree;
  std::optional<data::copy_plan> cp_top_tree_entities, cp_top_tree_nodes,
    cp_entities;

  // Buffer for ghosts shared
  data::buffers::core buf;
  static const std::size_t buffer_size =
    (data::buffers::Buffer::size / sizeof(interaction_entities)) * 2;

  ntree_base::en_size rz, sz;

private:
  // ----------------------- Top Tree Construction Tasks -----------------------
  // Build the local tree.
  // Add the entities in the hashmap and create needed nodes.
  // After this first step, the top tree entities and nodes are returned. These
  // will build the top tree, shared by all colors.
  static auto make_tree_local_task(
    typename Policy::template accessor<rw, na> t) {
    t.make_tree();
    return t.top_tree_boundaries();
  } // make_tree

  // Add the top tree to the local tree.
  // First step is to load the top tree entities and nodes.
  // The new sizes are then returned to create the copy plan for the top tree.
  static auto make_tree_distributed_task(
    typename Policy::template accessor<rw, na> t,
    const std::vector<hcell_t> & v) {
    t.add_boundaries(v);
  }

  // Return the number of entities/nodes for local, top tree and ghosts
  // stores in the respective index spaces.
  static std::array<ent_node, 3> sizes_task(
    typename field<meta_type>::template accessor<ro, na> m) {
    return {{m(0).local, m(0).top_tree, m(0).ghosts}};
  }

  // Copy plan: set destination sizes
  static void set_destination(field<data::intervals::Value>::accessor<wo> a,
    const std::vector<std::size_t> & base,
    const std::vector<std::size_t> & total) {
    auto i = process();
    a(0) = data::intervals::make({base[i], base[i] + total[i]}, i);
  }

  // Copy plan: set pointers for top tree
  template<index_space IS = entities>
  static void set_top_tree_ptrs(field<data::points::Value>::accessor<wo, na> a,
    const std::vector<std::size_t> & base,
    const std::vector<hcell_t> & hcells) {
    auto i = process();
    std::size_t idx = base[i];
    for(std::size_t j = 0; j < hcells.size(); ++j) {
      auto & h = hcells[j];
      if((IS == entities ? h.is_ent() : h.is_node()) && h.color() != i)
        a(idx++) = data::points::make(h.color(), h.idx());
    }
  }

  // Copy plan: set pointers to entities
  static void set_entities_ptrs(field<data::points::Value>::accessor<wo, na> a,
    const std::vector<std::size_t> & nents_base,
    const std::vector<std::pair<hcell_t, std::size_t>> & ids) {
    auto i = process();
    std::size_t idx = nents_base[i];
    for(std::size_t j = 0; j < ids.size(); ++j) {
      assert(ids[j].first.color() != i);
      a(idx++) = data::points::make(ids[j].first.color(), ids[j].first.idx());
    }
  }

  // ---------------------------- Top tree construction -----------------------
public:
  /// Build the local tree and share the top part of the tree
  static void make_tree(typename Policy::slot & ts) {

    auto cs = ts->colors();
    ts->cp_data_tree.issue_copy(data_field.fid);

    // Create the local tree
    // Return the list of nodes to share (top of the tree)
    auto fm_top_tree = flecsi::execute<make_tree_local_task>(ts);

    // Merge all the hcells informations for the top tree
    std::vector<hcell_t> top_tree;
    std::vector<std::size_t> top_tree_nnodes(cs, 0);
    std::vector<std::size_t> top_tree_nents(cs, 0);
    fm_top_tree.wait();

    for(std::size_t i = 0; i < cs; ++i) {
      auto f = fm_top_tree.get(i);
      top_tree.insert(top_tree.end(), f.begin(), f.end());
    }

    for(std::size_t c = 0; c < cs; ++c) {
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
    flecsi::execute<make_tree_distributed_task>(ts, top_tree);
    auto fm_sizes = flecsi::execute<sizes_task>(meta_field(ts));

    ts->sz.ent.resize(cs);
    ts->sz.node.resize(cs);
    ts->rz.node.resize(cs);
    ts->rz.ent.resize(cs);

    for(std::size_t i = 0; i < fm_sizes.size(); ++i) {
      auto f = fm_sizes.get(i);
      ts->sz.ent[i] = f[0].ents;
      ts->sz.node[i] = f[0].nodes;
    }

    for(std::size_t i = 0; i < cs; ++i) {
      ts->rz.ent[i] = ts->sz.ent[i] + top_tree_nents[i];
      ts->rz.node[i] = ts->sz.node[i] + top_tree_nnodes[i];
    }

    // Properly resize the partitions for the new number of ents + ghosts
    ts->part.template get<entities>().resize(
      make_partial<allocate>(ts->rz.ent));
    ts->part.template get<nodes>().resize(make_partial<allocate>(ts->rz.node));

    ts->cp_top_tree_entities.emplace(
      ts.get(),
      ts->part.template get<entities>(),
      data::copy_plan::Sizes(processes(), 1),
      [&](auto f) { execute<set_destination>(f, ts->sz.ent, top_tree_nents); },
      [&](auto f) {
        execute<set_top_tree_ptrs<entities>>(f, ts->sz.ent, top_tree);
      },
      util::constant<entities>());

    ts->cp_top_tree_nodes.emplace(
      ts.get(),
      ts->part.template get<nodes>(),
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
    typename field<interaction_entities>::template accessor<rw, na> a,
    typename field<std::size_t>::template accessor<wo, na> restart,
    data::buffers::Start mv,
    const std::vector<color_id> & f) {
    std::size_t cur = 0;
    std::size_t cs = run::context::instance().colors();
    auto color = run::context::instance().color();
    for(std::size_t c = 0; c < cs; ++c) {
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
    typename field<interaction_entities>::template accessor<rw, na> a,
    typename field<meta_type>::template accessor<rw, na> m,
    typename field<std::size_t>::template accessor<rw, na> restart,
    data::buffers::Transfer mv,
    const std::vector<color_id> & f) {

    // Read
    std::size_t cs = run::context::instance().colors();
    int stop = 0;
    int cur = 0;
    std::size_t idx = m(0).local.ents + m(0).top_tree.ents + m(0).ghosts.ents;
    auto color = run::context::instance().color();
    for(std::size_t c = 0; c < cs; ++c) {
      if(c != color) {
        auto r = mv[cur + cs - 1].read();
        while(r) {
          a(idx + m(0).nents_recv++) = r();
        } // while
        ++cur;
      } // if
    } // for

    // Keep copy if needed
    cur = 0;
    for(std::size_t c = 0; c < cs; ++c) {
      if(c != color) {
        if(restart(cur) != 0) {
          auto w = mv[cur].write();
          for(std::size_t i = restart(c); i < f.size(); ++i) {
            if(f[i].color == c) {
              if(!w(a(f[i].id))) {
                restart(cur) = i;
                stop = 1;
                break;
              }
            } // if
          } // for
          ++cur;
        } // if
      } // if
    } // for
    return stop;
  } // xfer_entities_req

  static serdez_vector<color_id> find_task(
    typename Policy::template accessor<rw, na> t) {
    return t.find_send_entities();
  }

  static serdez_vector<std::pair<hcell_t, std::size_t>> find_distant_task(
    typename Policy::template accessor<rw, na> t) {
    return t.find_intersect_entities();
  }

  static void load_entities_task(typename Policy::template accessor<rw, na> t,
    const std::vector<std::pair<hcell_t, std::size_t>> & recv) {
    auto hmap = t.map();
    for(std::size_t i = 0; i < recv.size(); ++i) {
      t.load_shared_entity_distant(0, recv[i].first.key(), hmap);
    }
  }

  // ----------------------------------- Share ghosts -------------------------
public:
  /// Search neighbors and complete the hmap, create copy plans (or buffer)
  static void share_ghosts(typename Policy::slot & ts) {
    // Find entities that will be used
    auto to_send = flecsi::execute<find_task>(ts);
    // Get current sizes
    auto fm_sizes = flecsi::execute<sizes_task>(meta_field(ts));

    std::vector<std::size_t> ents_sizes_rz(ts->colors());
    // Resize, add by the max size capability of the buffer
    for(std::size_t c = 0; c < ts->colors(); ++c) {
      auto f = fm_sizes.get(c);
      ents_sizes_rz[c] = f[0].ents + f[1].ents + f[2].ents + buffer_size;
    }

    ts->part.template get<entities>().resize(
      make_partial<allocate>(ents_sizes_rz));

    // Perform buffered copy
    auto full = 1;
    execute<xfer_entities_req_start>(
      e_i(ts.get()), comms_field(ts.get()), *(ts->buf), to_send.get(process()));
    while(reduce<xfer_entities_req, exec::fold::sum>(e_i(ts.get()),
      meta_field(ts),
      comms_field(ts.get()),
      *(ts->buf),
      to_send.get(process()))
            .get()) {
      // Size of IS is twice buffer size
      if(!(++full % 2)) {
        // Resize the partition if the maximum size is reached
        for(std::size_t c = 0; c < ts->colors(); ++c) {
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
    for(std::size_t c = 0; c < ts->colors(); ++c) {
      auto tr = to_reply.get(c);
      for(auto t : tr) {
        if(t.second == process()) {
          recv.push_back(t);
        }
      }
    }

    // Load entities destinated for this rank
    flecsi::execute<load_entities_task>(ts, recv);

    // Find the current sizes and how much needs to be resized
    std::vector<std::size_t> nents_base(processes());
    std::vector<std::size_t> nents_tt(processes());
    std::vector<std::size_t> nents_rz(processes());

    for(std::size_t c = 0; c < ts->colors(); ++c) {
      auto f = fm_sizes.get(c);
      nents_base[c] = f[0].ents + f[1].ents + f[2].ents;
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

    ts->cp_entities.emplace(data::copy_plan(ts.get(),
      ts->part.template get<entities>(),
      data::copy_plan::Sizes(processes(), 1),
      entities_dests_task,
      entities_ptrs_task,
      util::constant<entities>()));

    ts->cp_entities->issue_copy(e_keys.fid);
    ts->cp_entities->issue_copy(e_i.fid);
  }

  //------------------------------ reset tree ---------------------------------
private:
  static void reset_task(typename Policy::template accessor<rw> t) {
    t.reset();
  }

public:
  /// Reset the ntree topology, recompute the local tree and share new and share
  /// new neighbors
  static void reset(typename Policy::slot & ts) {
    flecsi::execute<reset_task>(ts);
    ts->cp_top_tree_entities.reset();
    ts->cp_top_tree_nodes.reset();
    ts->cp_entities.reset();
  }

  //---------------------------------------------------------------------------
  template<typename Type,
    data::layout Layout,
    typename Topo,
    typename Topo::index_space Space>
  void ghost_copy(data::field_reference<Type, Layout, Topo, Space> const & f) {
    if constexpr(Space == entities) {
      cp_top_tree_entities->issue_copy(f.fid());
      if(cp_entities.has_value()) {
        cp_entities->issue_copy(f.fid());
      }
    }
    else if constexpr(Space == nodes) {
      cp_top_tree_nodes->issue_copy(f.fid());
    }
    else if constexpr(Space == tree_data) {
      cp_data_tree.issue_copy(f.fid());
    }
  }

public:
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

private:
  const static size_t nchildren_ = 1 << dimension;
};

/// See \ref specialization_base::interface
template<class Policy>
template<Privileges Priv>
struct ntree<Policy>::access {
  template<const auto & F>
  using accessor = data::accessor_member<F, Priv>;
  /// Entities keys
  accessor<ntree::e_keys> e_keys;
  /// Nodes keys
  accessor<ntree::n_keys> n_keys;
  // Entities interaction fields
  accessor<ntree::e_i> e_i;
  /// Nodes interaction fields
  accessor<ntree::n_i> n_i;

private:
  accessor<ntree::data_field> data_field;
  accessor<ntree::hcells> hcells;
  accessor<ntree::meta_field> mf;

public:
  template<class F>
  void send(F && f) {
    e_keys.topology_send(f);
    n_keys.topology_send(f);
    data_field.topology_send(f);
    hcells.topology_send(f);
    e_i.topology_send(f);
    n_i.topology_send(f);
    mf.topology_send(std::forward<F>(f));
  }

  /// Hashing table type
  using hmap_t = util::hashtable<ntree::key_t, ntree::hcell_t, ntree::hash_f>;

  hmap_t map() const {
    return hmap_t(hcells.span());
  }

  void reset() {
    hmap_t hmap = map();
    hmap.clear();
    mf(0).max_depth = 0;
    mf(0).top_tree.ents = 0;
    mf(0).ghosts.ents = 0;
    mf(0).nents_recv = 0;
    mf(0).local.nodes = 0;
    mf(0).top_tree.nodes = 0;
    mf(0).ghosts.nodes = 0;
  }

  // Standard traversal function
  template<typename F>
  void traversal(hcell_t * hcell, F && f) const {
    auto hmap = map();
    std::queue<hcell_t *> tqueue;
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
              key_t ckey = nkey;
              ckey.push(j);
              tqueue.push(&hmap.at(ckey));
            } // if
          } // for
        }
      }
    } // while
  }

  auto find_intersect_entities() {
    auto hmap = map();
    auto cs = run::context::instance().colors();
    serdez_vector<std::pair<hcell_t, std::size_t>> entities;

    // Make a tree traversal per last elements in the intersection field.
    // Caution entities can be detected several time for the same neighbor.
    std::vector<std::set<hcell_t>> send_ids(cs);
    std::size_t start =
      mf(0).local.ents + mf(0).top_tree.ents + mf(0).ghosts.ents;
    std::size_t stop = start + mf(0).nents_recv;
    for(std::size_t i = start; i < stop; ++i) {
      ent_id id(i);

      std::size_t tcolor = e_i(i).color;
      assert(tcolor != run::context::instance().color());

      traversal(&hmap.at(key_t::root()), [&](hcell_t * cur) {
        if(cur->is_node()) {
          if(Policy::intersect_entity_node(
               e_i(id), n_i(topo::id<ntree_base::nodes>(cur->node_idx())))) {
            return true;
          } // if
        }
        else {
          // \todo add check here to see if the entities interact
          // For now, send a maximum of 8 entities
          if(cur->is_local()) {
            send_ids[tcolor].insert(*cur);
          }
        }
        return false;
      });
    } // for
    // Add all the std::sets to the end vector to create the copy plan
    for(std::size_t i = 0; i < cs; ++i) {
      for(auto a : send_ids[i]) {
        entities.push_back({a, i});
      }
    }
    return entities;
  }

  auto find_send_entities() {
    // The ranks to send and the id of the entity
    serdez_vector<color_id> entities;
    auto color = run::context::instance().color();
    auto hmap = map();
    // 1. for all local entities
    for(std::size_t i = 0; i < mf(0).local.ents; ++i) {
      ent_id id(i);
      std::set<std::size_t> send_colors;
      traversal(&hmap.at(key_t::root()), [&](hcell_t * cur) {
        if(cur->is_node()) {
          if(Policy::intersect_entity_node(
               e_i(id), n_i(topo::id<ntree_base::nodes>(cur->node_idx())))) {
            return true;
          } // if
        }
        else {
          // \todo add check here to see if the entities interact
          // For now, send a maximum of 8 entities
          if(cur->is_local()) {
            send_colors.insert(cur->color());
          }
        }
        return false;
      });
      // If color present in set, send this entity
      for(auto v : send_colors) {
        entities.push_back(color_id{v, id, color});
      } // for
    } // for
    return entities;
  }

  // --------------------------------------------------------------------------//
  //                                 ACCESSORS //
  // --------------------------------------------------------------------------//

  /// Return a range of all entities of a \c ntree_base::ptype_t
  template<ptype_t PT = ptype_t::exclusive>
  auto entities() {
    if constexpr(PT == ptype_t::exclusive) {
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(0, mf(0).local.ents));
    }
    else if constexpr(PT == ptype_t::ghost) {
      // Ghosts starts from local to end
      return make_ids<index_space::entities>(util::iota_view<util::id>(
        mf(0).local.ents + mf(0).top_tree.ents, e_keys.span().size()));
    }
    else {
      // Iterate on all
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(0, e_keys.span().size()));
    }
  }

  /// Get entities of a specific type from a node.
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
        key_t ckey = nkey;
        ckey.push(j);
        auto it = hmap.find(ckey);
        if(it->second.is_ent()) {
          ids.push_back(id<index_space::entities>(it->second.ent_idx()));
        }
      } // if
    } // for
    return ids;
  }

  /// Get entities adjacent to an entity.
  std::vector<id<index_space::entities>> neighbors(
    const id<index_space::entities> & ent_id) const {
    auto hmap = map();
    std::vector<id<index_space::entities>> ids;
    // Perform tree traversal to find neighbors
    traversal(&hmap.at(key_t::root()), [&](hcell_t * cur) {
      if(cur->is_node()) {
        if(Policy::intersect_entity_node(
             e_i(ent_id), n_i(topo::id<ntree_base::nodes>(cur->node_idx())))) {
          return true;
        } // if
      }
      else {
        // \todo add check here to see if the entities interact
        // For now, send a maximum of 8 entities
        if(Policy::intersect_entity_entity(e_i(ent_id),
             e_i(topo::id<ntree_base::entities>(cur->ent_idx())))) {
          ids.push_back(topo::id<ntree_base::entities>(cur->ent_idx()));
        }
      }
      return false;
    });
    return ids;
  }

  /// Return a range of all nodes of a \c ntree_base::ptype_t
  template<ptype_t PT = ptype_t::exclusive>
  auto nodes() {
    if constexpr(PT == ptype_t::exclusive) {
      return make_ids<index_space::nodes>(
        util::iota_view<util::id>(0, mf(0).local.nodes));
    }
    else if constexpr(PT == ptype_t::ghost) {
      // Ghosts starts from local to end
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(mf(0).local.nodes, n_keys.span().size()));
    }
    else {
      // Iterate on all
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(0, n_keys.span().size()));
    }
  }

  /// Get nodes belonging to a node.
  std::vector<id<index_space::nodes>> nodes(
    const id<index_space::nodes> & node_id) {
    std::vector<id<index_space::nodes>> ids;
    // Get the node and find its sub-entities
    auto nkey = n_keys[node_id];
    auto hmap = map();
    auto cur = &(hmap.find(nkey)->second);
    for(std::size_t j = 0; j < nchildren_; ++j) {
      if(cur->has_child(j)) {
        key_t ckey = nkey;
        ckey.push(j);
        auto it = hmap.find(ckey);
        if(it->second.is_node()) {
          ids.push_back(id<index_space::nodes>(it->second.node_idx()));
        }
      } // if
    } // for
    return ids;
  }

  /// BFS traversal, return vector of ids in Breadth First Search order
  auto bfs() {
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
          key_t ckey = nkey;
          ckey.push(j);
          auto it = hmap.find(ckey);
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
  auto dfs() {

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
            key_t ckey = nkey;
            ckey.push(j);
            auto it = hmap.find(ckey);
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
          std::size_t child =
            nchildren_ - 1 - j; // Take children in reverse order
          if(cur->has_child(child)) {
            key_t ckey = nkey;
            ckey.push(child);
            auto it = hmap.find(ckey);
            if(it->second.is_node()) {
              stk.push(&it->second);
            }
          } // if
        } // for
      } // while
      if constexpr(TT == ttype_t::reverse_preorder) {
        std::reverse(ids.begin(), ids.end());
      }
    } // if
    return ids;
  } // dfs

  //---------------------------------------------------------------------------//
  //                              MAKE TREE //
  //---------------------------------------------------------------------------//

  /// Compute local information about entities' keys boundaries
  void exchange_boundaries() {
    mf(0).max_depth = 0;
    mf(0).local.ents = e_keys.span().size();
    data_field(0).lobound = process() == 0 ? key_t::min() : e_keys(0);
    data_field(0).hibound = process() == processes() - 1
                              ? key_t::max()
                              : e_keys(mf(0).local.ents - 1);
  }

  void add_boundaries(const std::vector<hcell_t> & cells) {
    auto hmap = map();
    auto color = run::context::instance().color();
    for(auto c : cells) {
      if(c.color() == color)
        continue;
      if(c.is_ent()) {
        load_shared_entity(c.color(), c.key(), hmap);
      }
      else {
        load_shared_node(c.color(), c.key(), hmap);
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
        auto cur = hmap.find(c.key());
        assert(cur != hmap.end());
        std::size_t cnode = mf(0).local.nodes + mf(0).top_tree.nodes++;
        cur->second.set_node_idx(cnode);
        n_keys(cnode) = cur->second.key();
      }
    }
  }

  /// Construct the ntree based on the list of entities provided by the user
  /// This fills the hashing table with the entities and creates the
  /// intermediates nodes.
  void make_tree() {
    // Cstr htable
    auto hmap = map();

    // Create the tree
    const Color size = run::context::instance().colors(),
                rank = run::context::instance().color();

    /* Exchange high and low bound */
    const auto hibound =
      rank == size - 1 ? key_t::max() : data_field(2).lobound;
    const auto lobound = rank == 0 ? key_t::min() : data_field(1).hibound;
    assert(lobound <= e_keys(0));
    assert(hibound >= e_keys(mf(0).local.ents - 1));

    // Add the root
    hmap.insert(key_t::root(), key_t::root());
    auto root_ = hmap.find(key_t::root());
    root_->second.set_color(rank);
    {
      std::size_t cnode = mf(0).local.nodes++;
      root_->second.set_node_idx(cnode);
      n_keys(cnode) = root_->second.key();
    }
    size_t current_depth = key_t::max_depth();
    // Entity keys, last and current
    key_t lastekey = key_t(0);
    if(rank != 0)
      lastekey = lobound;
    // Node keys, last and Current
    key_t lastnkey = key_t::root();
    key_t nkey, loboundnode, hiboundnode;
    // Current parent and value
    hcell_t * parent = nullptr;
    bool old_is_ent = false;

    bool iam0 = rank == 0;
    bool iamlast = rank == size - 1;

    // The extra turn in the loop is to finish the missing
    // parent of the last entity
    for(size_t i = 0; i <= e_keys.span().size(); ++i) {
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
        int bit = nkey.last_value();
        parent->add_child(bit);
        parent->set_node();
        parent = &(hmap.insert(nkey, nkey)->second);
        parent->set_color(rank);
      } // while

      // Recover deleted entity
      if(old_is_ent) {
        int bit = lastnkey.last_value();
        parent->add_child(bit);
        parent->set_node();
        auto it = hmap.insert(lastnkey, lastnkey);
        it->second.set_ent_idx(i - 1);
        it->second.set_color(rank);
      } // if

      if(i < e_keys.span().size()) {
        // Insert the new entity
        int bit = nkey.last_value();
        parent->add_child(bit);
        auto it = hmap.insert(nkey, nkey);
        it->second.set_ent_idx(i);
        it->second.set_color(rank);
      } // if

      // Prepare next loop
      lastekey = ekey;
      lastnkey = nkey;
      mf(0).max_depth = std::max(mf(0).max_depth, current_depth);

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
        std::size_t cnode = mf(0).local.nodes++;
        cur->set_node_idx(cnode);
        n_keys(cnode) = cur->key();
      }
      for(std::size_t j = 0; j < nchildren_; ++j) {
        if(cur->has_child(j)) {
          key_t ckey = nkey;
          ckey.push(j);
          auto it = hmap.find(ckey);
          if(it->second.is_node()) {
            tqueue.push(&it->second);
          }
        } // if
      } // for
    } // while

  } // make_tree

  // Count number of entities to send to each other color
  auto top_tree_boundaries() {
    serdez_vector<hcell_t> sdata;
    auto hmap = map();
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
              key_t ckey = nkey;
              ckey.push(j);
              nqueue.push_back(&hmap.at(ckey));
            } // if
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
  }

  // Output a representation of the ntree using graphviz.
  // This will output a gv formatted as: output_graphviz_XXX_YYY.gv
  // With XXX = rank (color) and YYY = num (parameter).
  void graphviz_draw(int num) {
    int rank = process();
    std::ostringstream fname;
    fname << "output_graphviz_" << std::setfill('0') << std::setw(3) << rank
          << "_" << std::setfill('0') << std::setw(3) << num << ".gv";
    std::ofstream output;
    output.open(fname.str().c_str());
    output << "digraph G {\nforcelabels=true;\n";

    std::stack<hcell_t *> stk;
    auto hmap = map();

    stk.push(&(hmap.find(key_t::root())->second));

    while(!stk.empty()) {
      hcell_t * cur = stk.top();
      stk.pop();
      if(cur->is_node()) {
        if(cur->is_complete()) {
          output << std::oct << cur->key() << std::dec << " [label=<"
                 << std::oct << cur->key() << std::dec
                 << "<br/><FONT POINT-SIZE=\"10\"> c(" << cur->color()
                 << ")</FONT>>,xlabel=\"" << cur->nchildren() << "\"]\n";
          if(cur->is_nonlocal()) {
            output << std::oct << cur->key() << std::dec
                   << " [shape=doublecircle,color=grey]" << std::endl;
          }
          else {
            output << std::oct << cur->key() << std::dec
                   << " [shape=doublecircle,color=blue]\n";
          }
        }
        else {
          output << std::oct << cur->key() << std::dec << " [label=<"
                 << std::oct << cur->key() << std::dec
                 << "<br/><FONT POINT-SIZE=\"10\"> c(" << cur->color()
                 << ")</FONT>>,xlabel=\"" << cur->nchildren() << "\"]\n";
          if(cur->is_nonlocal()) {
            output << std::oct << cur->key() << std::dec
                   << " [shape=rect,color=black]" << std::endl;
          }
          else {
            output << std::oct << cur->key() << std::dec
                   << " [shape=rect,color=blue]\n";
          }
        }
        // Add the child to the stack and add for display
        for(std::size_t i = 0; i < nchildren_; ++i) {
          key_t ckey = cur->key();
          ckey.push(i);
          auto it = hmap.find(ckey);
          if(it != hmap.end()) {
            stk.push(&it->second);
            output << std::oct << cur->key() << "->" << it->second.key()
                   << std::dec << std::endl;
          }
        }
      }
      else if(cur->is_ent()) {
        output << std::oct << cur->key() << std::dec << " [label=<" << std::oct
               << cur->key() << std::dec << "<br/><FONT POINT-SIZE=\"10\"> c("
               << cur->color() << ")</FONT>>,xlabel=\"1\"]\n";
        if(cur->is_nonlocal()) {
          output << std::oct << cur->key() << std::dec
                 << " [shape=circle,color=grey]" << std::endl;
        }
        else {
          output << std::oct << cur->key() << std::dec
                 << " [shape=circle,color=blue]\n";
        }
      }
      else {
        // assert(false && "not type for node");
      }
    } // while
    output << "}\n";
    output.close();
  }

  void load_shared_entity_distant(const std::size_t & c,
    const key_t & k,
    hmap_t & hmap) {
    auto key = k;
    auto f = hmap.find(key);
    // assert(hmap.find(key) == hmap.end());
    if(f == hmap.end()) {
      auto & cur = hmap.insert(key, key)->second;
      cur.set_nonlocal();
      cur.set_color(c);
      auto eid = mf(0).local.ents + mf(0).top_tree.ents + mf(0).ghosts.ents++;
      cur.set_ent_idx(eid);
      // Add missing parent(s)
      int lastbit = key.pop_value();
      add_parent_distant(key, lastbit, c, hmap);
    }
    else {
      // todo Correct this. Do not transfer these entities
      auto & cur = hmap.insert(key, key)->second;
      auto eid = mf(0).local.ents + mf(0).top_tree.ents + mf(0).ghosts.ents++;
      cur.set_nonlocal();
      cur.set_color(c);
      cur.set_ent_idx(eid);
    }
  }

private:
  void
  load_shared_entity(const std::size_t & c, const key_t & k, hmap_t & hmap) {
    auto key = k;
    auto f = hmap.find(key);
    // assert(hmap.find(key) == hmap.end());
    if(f == hmap.end()) {
      auto & cur = hmap.insert(key, key)->second;
      cur.set_nonlocal();
      cur.set_color(c);
      auto eid = mf(0).local.ents + mf(0).top_tree.ents++;
      cur.set_ent_idx(eid);
      // Add missing parent(s)
      int lastbit = key.pop_value();
      add_parent(key, lastbit, c, hmap);
    }
    else {
      assert(false);
    }
  }

  void load_shared_node(const std::size_t & c, const key_t & k, hmap_t & hmap) {
    key_t key = k;
    // Node doesnt exists already
    auto cur = hmap.find(key);
    if(cur == hmap.end()) {
      auto & cur = hmap.insert(key, key)->second;
      cur.set_nonlocal();
      cur.set_color(c);
      // Add missing parent(s)
      int lastbit = key.pop_value();
      add_parent(key, lastbit, c, hmap);
    }
    else {
      assert(false);
    }
  }

  // Add missing parent from distant node/entity
  // This version does add the parents and add an idx
  // This can only be done before any ghosts are received
  void add_parent(key_t key, int child, const int & color, hmap_t & hmap) {
    auto parent = hmap.end();
    while((parent = hmap.find(key)) == hmap.end()) {
      parent = hmap.insert(key, key);
      std::size_t cnode = mf(0).local.nodes++;
      parent->second.set_node_idx(cnode);
      n_keys(cnode) = key;
      // parent->second.set_nonlocal();
      parent->second.add_child(child);
      parent->second.set_color(color);
      child = key.pop_value();

    } // while
    assert(parent->second.is_incomplete());
    parent->second.add_child(child);
  }

  // Add missing parent from distant node/entity
  // This version does add the parent but does not provides an idx for it.
  // This is used when the local tree already received distant entities/nodes.
  void
  add_parent_distant(key_t key, int child, const int & color, hmap_t & hmap) {
    auto parent = hmap.end();
    while((parent = hmap.find(key)) == hmap.end()) {
      parent = hmap.insert(key, key);
      parent->second.add_child(child);
      parent->second.set_color(color);
      child = key.pop_value();

    } // while
    assert(parent->second.is_incomplete());
    parent->second.add_child(child);
  }

  std::pair<key_t, key_t> key_boundary(const key_t & key) {
    key_t stop = key_t::min();
    std::pair<key_t, key_t> min_max = {key, key};
    while(min_max.first < stop) {
      min_max.second.push((1 << dimension) - 1);
      min_max.first.push(0);
    } // while
    return min_max;
  } // key_boundary

}; // namespace topo

template<>
struct detail::base<ntree> {
  using type = ntree_base;
};

/// \}

} // namespace topo
} // namespace flecsi
/// \endcond

#endif
