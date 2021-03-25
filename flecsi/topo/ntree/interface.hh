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

/*! @file */

#if !defined(__FLECSI_PRIVATE__)
#error Do not include this file directly!
#endif

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

namespace flecsi {
namespace topo {

template<typename T>
class serdez_vector : public std::vector<T>
{
public:
  inline size_t legion_buffer_size(void) const {
    return (sizeof(size_t) + (sizeof(T) * this->size()));
  }

  inline void legion_serialize(void * buffer) const {
    char * ptr = (char *)buffer;
    std::size_t size = this->size();
    memcpy(ptr, &size, sizeof(std::size_t));
    ptr += sizeof(std::size_t);
    for(std::size_t i = 0; i < this->size(); ++i, ptr += sizeof(T)) {
      memcpy(ptr, this->data() + i, sizeof(T));
    }
  }

  inline void legion_deserialize(const void * buffer) {
    const char * ptr = (const char *)buffer;
    size_t elements = *(std::size_t *)ptr;
    ptr += sizeof(std::size_t);
    this->resize(elements);
    for(std::size_t i = 0; i < elements; ++i, ptr += sizeof(T)) {
      memcpy(this->data() + i, ptr, sizeof(T));
    }
  }
};

//---------------------------------------------------------------------------//
// NTree topology.
//---------------------------------------------------------------------------//

/*!
  @ingroup topology
 */

//-----------------------------------------------------------------//
//! The tree topology is parameterized on a policy P which defines its nodes
//! and entity types.
//-----------------------------------------------------------------//
template<typename Policy>
struct ntree : ntree_base, with_meta<Policy> {
  // Get types from Policy
  constexpr static unsigned int dimension = Policy::dimension;
  using key_int_t = typename Policy::key_int_t;
  using key_t = typename Policy::key_t;
  using node_t = typename Policy::node_t;
  using ent_t = typename Policy::ent_t;
  using hash_f = typename Policy::hash_f;

  using index_space = typename Policy::index_space;
  using index_spaces = typename Policy::index_spaces;

  using type_t = double;
  using hcell_t = hcell_base_t<dimension, type_t, key_t>;

  using interaction_entities = typename Policy::interaction_entities;
  using interaction_nodes = typename Policy::interaction_nodes;

  using ent_id = topo::id<entities>;
  using node_id = topo::id<nodes>;
  
  template<auto>  
  static constexpr std::size_t privilege_count = 2;

  struct ent_node {
    std::size_t ents;
    std::size_t nodes;
  };

  struct ntree_data {
    key_t hibound, lobound;
  };

  struct meta_type {
    std::size_t max_depth;
    ent_node local, ghosts, top_tree;
    std::size_t nents_recv;
  };

  struct {
    std::vector<std::size_t> ent, node;
  } sz, rz;

  struct color_id {
    std::size_t color;
    ent_id id;
    std::size_t from_color;
  };

  template<std::size_t>
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
          make_partial<allocate>(c.tdata_offset_))}},
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
    execute<set_meta>(meta_field(this->meta));
  }

  // Ntree mandatory fields ---------------------------------------------------

  static inline const typename field<key_t>::template definition<Policy,
    entities>
    e_keys;
  static inline const typename field<key_t>::template definition<Policy, nodes>
    n_keys;

  // Hmap fields
  static inline const typename field<
    std::pair<key_t, hcell_t>>::template definition<Policy, hashmap>
    hcells;

  // Tdata field
  static inline const typename field<ntree_data>::template definition<Policy,
    tree_data>
    data_field;

  static inline const typename field<
    interaction_entities>::template definition<Policy, entities>
    e_i;
  static inline const typename field<
    interaction_nodes>::template definition<Policy, nodes>
    n_i;

  static inline const typename field<meta_type,
    data::single>::template definition<meta<Policy>>
    meta_field;

  // --------------------------------------------------------------------------

  // Use to reference the index spaces by id
  util::key_array<repartitioned, index_spaces> part;

  data::copy_plan cp_data_tree;
  std::optional<data::copy_plan> cp_top_tree_entities, cp_top_tree_nodes,
    cp_entities;

  // Buffer for ghosts shared
  data::buffers::core buf;
  static const std::size_t buffer_size =
    (data::buffers::Buffer::size / sizeof(interaction_entities)) * 2;


  // ----------------------- Top Tree Construction Tasks -----------------------
  // This is decomposed in two steps since
  // we need to share information using a
  // future_map to continue the construction
  static serdez_vector<hcell_t> make_tree_local_task(
    typename Policy::template accessor<rw,na> t) {
    t.make_tree();
    return t.top_tree_boundaries();
  } // make_tree

  static auto make_tree_distributed_task(
    typename Policy::template accessor<rw,na> t,
    typename field<meta_type, data::single>::template accessor<ro> m,
    const std::vector<hcell_t> & v) {
    t.add_boundaries(v);
    return sizes_task(m);
  }

  static std::array<ent_node, 3> sizes_task(
    typename field<meta_type, data::single>::template accessor<ro> m) {
    return {{m.get().local, m.get().top_tree, m.get().ghosts}};
  }

  // Copy plan creation tasks
  static void set_destination(field<data::intervals::Value>::accessor<wo> a,
    const std::vector<std::size_t> & base,
    const std::vector<std::size_t> & total) {
    auto i = process();
    a(0) = data::intervals::make({base[i], base[i] + total[i]}, i);
  }

  template<index_space IS = entities>
  static void set_top_tree_ptrs(field<data::points::Value>::accessor<wo,na> a,
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

  static void set_entities_ptrs(field<data::points::Value>::accessor<wo,na> a,
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

  /**
   * @brief  Build the local tree and share the top part of the tree
   */
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
    auto fm_sizes = flecsi::execute<make_tree_distributed_task>(
      ts, meta_field(ts->meta), top_tree);

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
    ts->part.template get<entities>().resize(make_partial<allocate>(ts->rz.ent));
    ts->part.template get<nodes>().resize(make_partial<allocate>(ts->rz.node));

    ts->cp_top_tree_entities.emplace(
      ts.get(),
      data::copy_plan::Sizes(processes(), 1),
      [&](auto f) { execute<set_destination>(f, ts->sz.ent, top_tree_nents); },
      [&](
        auto f) { execute<set_top_tree_ptrs<entities>>(f, ts->sz.ent, top_tree); },
      util::constant<entities>());

    ts->cp_top_tree_nodes.emplace(
      ts.get(),
      data::copy_plan::Sizes(processes(), 1),
      [&](auto f) { execute<set_destination>(f, ts->sz.node, top_tree_nnodes); },
      [&](auto f) { execute<set_top_tree_ptrs<nodes>>(f, ts->sz.node, top_tree); },
      util::constant<nodes>());

    ts->cp_top_tree_entities->issue_copy(e_keys.fid);
    ts->cp_top_tree_nodes->issue_copy(n_keys.fid);

  }

  // ---------------------------- Ghosts exchange tasks -----------------------

  static auto xfer_entities_req_start(
    typename field<interaction_entities>::template accessor<rw,na> a,
    data::buffers::Start mv,
    const std::vector<color_id> & f) {
    std::size_t cur = 0;
    std::size_t cs = run::context::instance().colors();
    serdez_vector<std::size_t> restart; // Use last value to store total
    restart.resize(cs); 
    auto color = run::context::instance().color();
    for(std::size_t c = 0; c < cs; ++c) {
      if(c != color) {
        auto w = mv[cur].write();
        for(std::size_t i = 0; i < f.size(); ++i) {
          if(f[i].color == c) {
            if(!w(a(f[i].id))) {
              restart[cur] = i;
              restart[cs - 1] += i;
              break;
            } // if
          } // if
        } // for
        ++cur;
      } // if
    } // for
    return restart;
  } // xfer_nodes_req_start

  static auto xfer_entities_req(
    typename field<interaction_entities>::template accessor<rw,na> a,
    typename field<meta_type, data::single>::template accessor<rw> m,
    data::buffers::Transfer mv,
    const std::vector<color_id> & f,
    const std::vector<std::size_t> & v) {

    std::size_t cs = run::context::instance().colors();
    serdez_vector<std::size_t> restart;
    restart.resize(cs); 
    int cur = 0;
    std::size_t idx =
      m.get().local.ents + m.get().top_tree.ents + m.get().ghosts.ents;
    auto color = run::context::instance().color();
    for(std::size_t c = 0; c < cs; ++c) {
      if(c != color) {
        auto r = mv[cur + cs - 1].read();
        while(r) {
          a(idx + m.get().nents_recv++) = r();
        } // while
        ++cur;
      } // if
    } // for

    // Keep copy if needed
    cur = 0;
    for(std::size_t c = 0; c < cs; ++c) {
      if(c != color) {
        if(v[cur] != 0) {
          auto w = mv[cur].write();
          for(std::size_t i = v[cur]; i < f.size(); ++i) {
            if(f[i].color == c) {
              if(!w(a(f[i].id))) {
                restart[cur] = i;
                restart[cs - 1] += i;
                break;
              }
            } // if
          } // for
          ++cur;
        } // if
      } // if
    } // for

    return restart;
  } // xfer_entities_req

  static serdez_vector<color_id> find_task(
    typename Policy::template accessor<rw,na> t) {
    return t.find_send_entities();
  }

  static serdez_vector<std::pair<hcell_t, std::size_t>> find_distant_task(
    typename Policy::template accessor<rw,na> t) {
    return t.find_intersect_entities();
  }

  static void load_entities_task(typename Policy::template accessor<rw,na> t,
    const std::vector<std::pair<hcell_t, std::size_t>> & recv) {
    auto hmap = t.map();
    for(std::size_t i = 0; i < recv.size(); ++i) {
      t.load_shared_entity_distant(0, recv[i].first.key(), hmap);
    }
  }

  // ----------------------------------- Share ghosts -------------------------

  // Search neighbors and complete the hmap, create copy plans (or buffer)
  static void share_ghosts(typename Policy::slot & ts) {
    // Find entities that will be used
    auto to_send = flecsi::execute<find_task>(ts);
    // Get current sizes
    auto fm_sizes = flecsi::execute<sizes_task>(meta_field(ts->meta));

    std::vector<std::size_t> ents_sizes_rz(ts->colors());
    // Resize, add by the max size capability of the buffer
    for(std::size_t c = 0; c < ts->colors(); ++c) {
      auto f = fm_sizes.get(c);
      ents_sizes_rz[c] = f[0].ents + f[1].ents + f[2].ents + buffer_size;
    }

    ts->part.template get<entities>().resize(make_partial<allocate>(ents_sizes_rz));

    // Perform buffered copy
    auto full = 0;
    auto vi = execute<xfer_entities_req_start>(
      e_i(ts.get()), *(ts->buf), to_send.get(process()));
    while((vi = execute<xfer_entities_req>(e_i(ts.get()),
             meta_field(ts->meta),
             *(ts->buf),
             to_send.get(process()),
             vi.get(process())))
            .get(process())[ts->colors() - 1] != 0) {

      // Size of IS is twice buffer size
      if(!(full % 2)) {
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
    std::vector<hcell_t> hcells;

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
      data::copy_plan::Sizes(processes(), 1),
      entities_dests_task,
      entities_ptrs_task,
      util::constant<entities>()));

    ts->cp_entities->issue_copy(e_keys.fid);
    ts->cp_entities->issue_copy(e_i.fid);
  }

  //------------------------------ reset tree ---------------------------------

  static void reset_task(typename Policy::template accessor<rw> t) {
    t.reset();
  }

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
  }

  std::size_t colors() const {
    return part.front().colors();
  }

  template<index_space S>
  data::region & get_region() {
    return part.template get<S>();
  }

  template<index_space S>
  const data::partition & get_partition(field_id_t) const {
    return part.template get<S>();
  }

private:

  static void set_meta(
    typename field<meta_type, data::single>::template accessor<wo> m) {
    m.get() = {};
  } // set_meta

  const static size_t nchildren_ = 1 << dimension;
};

/**
 * Ntree access
 */
template<class Policy>
template<std::size_t Priv>
struct ntree<Policy>::access {
  template<const auto & F>
  using accessor = data::accessor_member<F, Priv>;
  accessor<ntree::e_keys> e_keys;
  accessor<ntree::n_keys> n_keys;
  accessor<ntree::data_field> data_field;
  accessor<ntree::hcells> hcells;
  accessor<ntree::e_i> e_i;
  accessor<ntree::n_i> n_i;

  data::scalar_access<ntree::meta_field> mf;

  template<class F>
  void send(F && f) {
    e_keys.topology_send(f);
    n_keys.topology_send(f);
    data_field.topology_send(f);
    hcells.topology_send(f);
    e_i.topology_send(f);
    n_i.topology_send(f);
    mf.topology_send(f, &ntree::meta);
  }

  using hmap_t = util::hashtable<ntree::key_t, ntree::hcell_t, ntree::hash_f>;

  hmap_t map() const {
    return hmap_t(hcells.span());
  }

  void reset() {
    hmap_t hmap = map();
    hmap.clear();
    mf->max_depth = 0;
    mf->top_tree.ents = 0;
    mf->ghosts.ents = 0;
    mf->nents_recv = 0;
    mf->local.nodes = 0;
    mf->top_tree.nodes = 0;
    mf->ghosts.nodes = 0;
  }

  // Standard traversal function
  template<typename FUNC>
  void traversal(hcell_t * hcell, FUNC && func) const {
    auto hmap = map();
    std::queue<hcell_t *> tqueue;
    tqueue.push(hcell);
    while(!tqueue.empty()) {
      hcell_t * cur = tqueue.front();
      tqueue.pop();
      // Intersection
      if(func(cur)) {
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
    std::size_t start = mf->local.ents + mf->top_tree.ents + mf->ghosts.ents;
    std::size_t stop = start + mf->nents_recv;
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
    for(std::size_t i = 0; i < mf->local.ents; ++i) {
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
        assert(v != color);
        entities.push_back(color_id{v, id, color});
      } // for
    } // for
    return entities;
  }

  // --------------------------------------------------------------------------//
  //                                 ACCESSORS //
  // --------------------------------------------------------------------------//

  // Iterate on all exclusive or ghost entities
  template<ptype_t PT = ptype_t::exclusive>
  auto entities() {
    if constexpr(PT == ptype_t::exclusive) {
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(0, mf->local.ents));
    }
    else if constexpr(PT == ptype_t::ghost) {
      // Ghosts starts from local to end
      return make_ids<index_space::entities>(util::iota_view<util::id>(
        mf->local.ents + mf->top_tree.ents, e_keys.span().size()));
    }
    else {
      // Iterate on all
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(0, e_keys.span().size()));
    }
  }

  // Return entities, exclusive/ghosts from a specific node
  template<ptype_t PT = ptype_t::exclusive>
  auto entities(const id<index_space::nodes> & node_id) const {

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

  auto neighbors(const id<index_space::entities> & ent_id) const {
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

  // Iterate on all exclusive or ghost nodes
  template<ptype_t PT = ptype_t::exclusive>
  auto nodes() {
    if constexpr(PT == ptype_t::exclusive) {
      return make_ids<index_space::nodes>(
        util::iota_view<util::id>(0, mf->local.nodes));
    }
    else if constexpr(PT == ptype_t::ghost) {
      // Ghosts starts from local to end
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(mf->local.nodes, n_keys.span().size()));
    }
    else {
      // Iterate on all
      return make_ids<index_space::entities>(
        util::iota_view<util::id>(0, n_keys.span().size()));
    }
  }

  // Return entities, exclusive/ghosts from a specific node
  auto nodes(const id<index_space::nodes> & node_id) {
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

  // BFS traversal, return vector of ids
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
              if constexpr(complete == true) {
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
        if(!complete || cur->is_complete()) {
          ids.push_back(id<index_space::nodes>(cur->idx()));
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

  void exchange_boundaries() {
    mf->max_depth = 0;
    mf->local.ents = e_keys.span().size();
    data_field(0).lobound = process() == 0 ? key_t::min() : e_keys(0);
    data_field(0).hibound =
      process() == processes() - 1 ? key_t::max() : e_keys(mf->local.ents - 1);
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
        std::size_t cnode = mf->local.nodes + mf->top_tree.nodes++;
        cur->second.set_node_idx(cnode);
        n_keys(cnode) = cur->second.key();
      }
    }
  }

  void make_tree() {
    // Cstr htable
    auto hmap = map();

    // Create the tree
    size_t size = run::context::instance().colors(); // colors();
    size_t rank = run::context::instance().color(); // color();

    /* Exchange high and low bound */
    const auto hibound =
      rank == size - 1 ? key_t::max() : data_field(2).lobound;
    const auto lobound = rank == 0 ? key_t::min() : data_field(1).hibound;
    assert(lobound <= e_keys(0));
    assert(hibound >= e_keys(mf->local.ents - 1));

    // Add the root
    hmap.insert(key_t::root(), key_t::root());
    auto root_ = hmap.find(key_t::root());
    root_->second.set_color(rank);
    {
      std::size_t cnode = mf->local.nodes++;
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
      mf->max_depth = std::max(mf->max_depth, current_depth);

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
  serdez_vector<hcell_t> top_tree_boundaries() {
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
              auto it = hmap.find(ckey);
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

  // Draw the tree
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
      auto eid = mf->local.ents + mf->top_tree.ents + mf->ghosts.ents++;
      cur.set_ent_idx(eid);
      // Add missing parent(s)
      int lastbit = key.pop_value();
      add_parent_distant(key, lastbit, c, hmap);
    }
    else {
      // todo Correct this. Do not transfer these entities
      auto & cur = hmap.insert(key, key)->second;
      auto eid = mf->local.ents + mf->top_tree.ents + mf->ghosts.ents++;
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
      auto eid = mf->local.ents + mf->top_tree.ents++;
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

  /**
   * @brief Add missing parent from distant node/entity
   * This version does add the parents and add an idx
   * This can only be done before any ghosts are received
   */
  void add_parent(key_t key, int child, const int & color, hmap_t & hmap) {
    auto parent = hmap.end();
    while((parent = hmap.find(key)) == hmap.end()) {
      parent = hmap.insert(key, key);
      std::size_t cnode = mf->local.nodes++;
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

  /**
   * @brief Add missing parent from distant node/entity
   * This version does add the parent but does not provides an idx for it.
   * This is used when the local tree already received distant entities/nodes.
   */
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

} // namespace topo
} // namespace flecsi
