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
#include "flecsi/data/topology.hh"
#include "flecsi/execution.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/core.hh" // base
#include "flecsi/topo/ntree/coloring.hh"
#include "flecsi/topo/ntree/types.hh"
#include "flecsi/util/hashtable.hh"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stack>
#include <type_traits>
#include <unordered_map>

namespace flecsi {
namespace topo {

// Special vector type
// Cannot contain other std containers
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

//----------------------------------------------------------------------------//
// NTree topology.
//----------------------------------------------------------------------------//

/*!
  @ingroup topology
 */

//-----------------------------------------------------------------//
//! The tree topology is parameterized on a policy P which defines its nodes
//! and entity types.
//-----------------------------------------------------------------//
template<typename Policy>
struct ntree : ntree_base {
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

  struct ntree_data {
    key_t hibound, lobound;
    std::size_t max_depth = 0;
    std::size_t nents = 0;
    std::size_t nents_ghosts = 0; 
    std::size_t nnodes = 0;
    std::size_t nnodes_ghosts = 0; 
  };

  template<std::size_t>
  struct access;

  ntree(const coloring & c)
    : part{{make_repartitioned<Policy, entities>(c.nparts_,
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
        util::constant<tree_data>()) {}
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

  // --------------------------------------------------------------------------

  // Use to reference the index spaces by id
  util::key_array<repartitioned, index_spaces> part;

  data::copy_plan cp_data_tree;
  std::optional<data::copy_plan> cp_top_tree_entities;
  std::optional<data::copy_plan> cp_top_tree_nodes;

  // Construction of the tree -------------------------------------------------
  // This is decomposed in two steps since
  // we need to share information using a
  // future_map to continue the construction
  static serdez_vector<hcell_t> make_local_tree(
    typename Policy::template accessor<rw> t) {
    t.make_tree();
    t.graphviz_draw(0);
    return t.top_tree_boundaries();
  } // make_tree

  static std::pair<std::size_t, std::size_t> make_distributed_tree(
    typename Policy::template accessor<rw> t,
    const std::vector<hcell_t> & v) {
    std::pair<std::size_t, std::size_t> sizes = t.get_sizes();
    t.add_boundaries(v);
    t.graphviz_draw(1);
    return sizes;
  }

  static void display_keys(typename Policy::template accessor<ro> t) {
    std::cout << color() << ": E: ";
    for(auto a : t.e_keys.span()) {
      auto b = a;
      b.truncate(5);
      std::cout << b << " - ";
    }
    std::cout << std::endl;
    std::cout << color() << ": N: ";
    for(auto a : t.n_keys.span()) {
      auto b = a;
      b.truncate(5);
      std::cout << b << " - ";
    }
    std::cout << std::endl;
  }

#if 0
  static void
  xfer_start(typename field<key_t>::template accessor<ro> a, 
  data::buffers::Start mv) { 
    auto w1 = mv[0].write(); 
    w1(a(0)); 
    auto w2 = mv[1].write(); 
    w2(a(0));
    auto w3 = mv[2].write(); 
    w3(a(0)); 
  }

  static int
  xfer(typename field<key_t>::template accessor<rw> a, 
    data::buffers::Transfer mv) {
    auto r1 = mv[3].read(); 
    a(1) = r1(); 
    auto r2 = mv[4].read(); 
    a(2) = r2();
    auto r3 = mv[5].read(); 
    a(3) = r3(); 
    return 0; 
  }
#endif 

  template<class T>
  void make_tree(T && ts) {
    cp_data_tree.issue_copy(data_field.fid);

    // Create the local tree
    // Return the list of nodes to share (top of the tree)
    auto fm_top_tree = flecsi::execute<make_local_tree>(ts);

    // Merge all the hcells informations 
    std::vector<hcell_t> top_tree;
    std::vector<std::size_t> top_tree_nnodes(colors(), 0);
    std::vector<std::size_t> top_tree_nents(colors(), 0);
    fm_top_tree.wait();

    for(std::size_t i = 0; i < colors(); ++i) {
      auto f = fm_top_tree.get(i);
      top_tree.insert(top_tree.end(), f.begin(), f.end());
    }

    for(std::size_t c = 0; c < colors(); ++c) {
      for(std::size_t j = 0; j < top_tree.size(); ++j) {
        if(top_tree[j].color() != c) {
          if(top_tree[j].is_ent())
            ++top_tree_nents[c];
          else
            ++top_tree_nnodes[c];
        }
      }
    }

    // Add the new hcells to the local tree 
    auto fm_sizes = flecsi::execute<make_distributed_tree>(ts, top_tree);

    std::vector<std::size_t> ents_sizes(colors());
    std::vector<std::size_t> nodes_sizes(colors());
    std::vector<std::size_t> ents_sizes_rz(colors());
    std::vector<std::size_t> nodes_sizes_rz(colors());

    for(std::size_t i = 0; i < fm_sizes.size(); ++i) {
      auto f = fm_sizes.get(i);
      ents_sizes[i] = f.first;
      nodes_sizes[i] = f.second;
    }

    flecsi::execute<display_keys>(ts);

    // Add the total to rebuild the partitions
    for(std::size_t i = 0; i < colors(); ++i) {
      ents_sizes_rz[i] = ents_sizes[i] + top_tree_nents[i];
      nodes_sizes_rz[i] = nodes_sizes[i] + top_tree_nnodes[i];
    }

    resize_repartitioned(
      part.template get<entities>(), make_partial<allocate>(ents_sizes_rz));
    resize_repartitioned(
      part.template get<nodes>(), make_partial<allocate>(nodes_sizes_rz));

    // Create this new copy plan
    top_tree_cp(
      top_tree, top_tree_nents, top_tree_nnodes, ents_sizes, nodes_sizes);

    // Retrieve the information needed by the specialization
    cp_top_tree_entities->issue_copy(e_keys.fid);
    cp_top_tree_nodes->issue_copy(n_keys.fid);

    flecsi::execute<display_keys>(ts);

    // Several rounds of communications to find all the neighbors 
    share_ghosts();

#if 0 
    // Share the keys 
    data::buffers::core buf([] {
      const auto p = processes();
      data::buffers::coloring ret(p);
      for(std::size_t i_r = 0 ; i_r < ret.size(); ++i_r){
        for(std::size_t i = 0 ; i < processes(); ++i){
          if(i!=i_r){
            ret[i_r].push_back(i); 
          }
        }
      }
      return ret;
    }());

    flecsi::execute<display_keys>(ts);

    execute<xfer_start>(e_keys(*this), *buf);
    while(reduce<xfer, exec::fold::max>(e_keys(*this), *buf).get());

    flecsi::execute<display_keys>(ts);
#endif 

  }

  static void set_top_tree_entities_dests(
     field<data::intervals::Value>::accessor<wo> a, 
    const std::vector<std::size_t>& nents_base, 
    const std::vector<std::size_t>& nents_tt){
      auto i = process(); 
      a(0) = data::intervals::make({nents_base[i], nents_base[i] + nents_tt[i]},i);
    } 

  static void set_top_tree_entities_ptrs(
    field<data::points::Value>::accessor<wo> a,
    const std::vector<std::size_t>& nents_base, 
    const std::vector<hcell_t>& hcells){
      auto i = process();
      std::size_t idx = nents_base[i]; 
      for(std::size_t j = 0; j < hcells.size(); ++j) {
        if(hcells[j].is_ent() && hcells[j].color() != i) {
          a(idx++) = data::points::make(hcells[j].color(), hcells[j].idx());
        }
      }
    }

  static void set_top_tree_nodes_dests(
     field<data::intervals::Value>::accessor<wo> a, 
    const std::vector<std::size_t>& nnodes_base, 
    const std::vector<std::size_t>& nnodes_tt){
      auto i = process(); 
      a(0) = data::intervals::make({nnodes_base[i], nnodes_base[i] + nnodes_tt[i]},i);
    }

  static void set_top_tree_nodes_ptrs(
    field<data::points::Value>::accessor<wo> a,
    const std::vector<std::size_t>& nnodes_base, 
    const std::vector<hcell_t>& hcells){
      auto i = process(); 
      std::size_t idx = nnodes_base[i]; 
      for(std::size_t j = 0; j < hcells.size(); ++j) {
        if(hcells[j].is_node() && hcells[j].color() != i) {
          a(idx++) = data::points::make(hcells[j].color(), hcells[j].idx());
        }
      }
    }
 
  void top_tree_cp(const std::vector<hcell_t> & hcells,
    const std::vector<std::size_t> & nents_tt,
    const std::vector<std::size_t> & nnodes_tt,
    const std::vector<std::size_t> & nents_base,
    const std::vector<std::size_t> & nnodes_base) {
    // create the vectors of: destination size (based on total + current
    // position)
    auto entities_dests_task = [&nents_base, &nents_tt](auto f) {
      execute<set_top_tree_entities_dests>(f, nents_base, nents_tt);
    };
    auto entities_ptrs_task = [&nents_base, &hcells](auto f){
      execute<set_top_tree_entities_ptrs>(f, nents_base, hcells); 
    };

    auto nodes_dests_task = [&nnodes_base, &nnodes_tt](auto f) {
      execute<set_top_tree_nodes_dests>(f, nnodes_base, nnodes_tt);
    };
    auto nodes_ptrs_task = [&nnodes_base, &hcells](auto f){
      execute<set_top_tree_nodes_ptrs>(f, nnodes_base, hcells); 
    };

    cp_top_tree_entities = data::copy_plan(
      *this,
      data::copy_plan::Sizes(processes(), 1), 
      entities_dests_task, 
      entities_ptrs_task,  
      util::constant<entities>()
    ); 

    cp_top_tree_nodes = data::copy_plan(
      *this,
      data::copy_plan::Sizes(processes(), 1), 
      nodes_dests_task, 
      nodes_ptrs_task, 
      util::constant<nodes>()); 
  }

  // Search neighbors and complete the hmap, create copy plans (or buffer)
  void share_ghosts(){
    bool done = false; 

    // Create buffer communicator
    // For now, complete all to all   

    //do{

      // 1. Send all 
    //  execute<xfer_start>(e_keys(*this), *buf);
    //  while(reduce<xfer, exec::fold::max>(e_keys(*this), *buf).get());


    //  execute<xfer_start>(e_keys(*this), *buf);
    //  while(reduce<xfer, exec::fold::max>(e_keys(*this), *buf).get());
    
    //}while(!done); 
  }

  //---------------------------------------------------------------------------

  template<typename Type,
    data::layout Layout,
    typename Topo,
    typename Topo::index_space Space>
  void ghost_copy(data::field_reference<Type, Layout, Topo, Space> const & f) {
    if constexpr (Space == entities){
      cp_top_tree_entities->issue_copy(f.fid());
    } else if constexpr(Space == nodes){
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
  const static size_t nchildren_ = 1 << dimension;
};

template<class Policy>
template<std::size_t Priv>
struct ntree<Policy>::access {
  template<const auto & F>
  using accessor = data::accessor_member<F, Priv>;
  accessor<ntree::e_keys> e_keys;
  accessor<ntree::n_keys> n_keys;
  accessor<ntree::data_field> data_field;
  accessor<ntree::hcells> hcells;

  template<class F>
  void send(F && f) {
    e_keys.topology_send(f);
    n_keys.topology_send(f);
    data_field.topology_send(f);
    hcells.topology_send(f);
  }

  // Accessors for exclusive and ghost entities 
  //template<index_space IndexSpace>
  template<ptype_t PT = ptype_t::exclusive>
  auto entities() {
    if constexpr (PT == ptype_t::exclusive){
      return make_ids<index_space::entities>(util::iota_view<util::id>(
       0, data_field(0).nents));
    } else if constexpr (PT == ptype_t::ghost){
      // Ghosts starts from local to end 
      return make_ids<index_space::entities>(util::iota_view<util::id>(
       data_field(0).nents, e_keys.span().size())); 
    }else{
      // Iterate on all 
      return make_ids<index_space::entities>(util::iota_view<util::id>(
       0, e_keys.span().size()));
    }
  }

  using hmap_t = util::hashtable<ntree::key_t, ntree::hcell_t, ntree::hash_f>;

  std::pair<std::size_t, std::size_t> get_sizes() {
    return {data_field(0).nents, data_field(0).nnodes};
  }

  void exchange_boundaries() {
    data_field(0).max_depth = 0;
    data_field(0).nents = e_keys.span().size();
    data_field(0).lobound = process() == 0 ? key_t::min() : e_keys(0);
    data_field(0).hibound = process() == processes() - 1
                              ? key_t::max()
                              : e_keys(data_field(0).nents - 1);
  }

  void add_boundaries(const std::vector<hcell_t> & cells) {
    hmap_t hmap(hcells.span());
    for(auto c : cells) {
      if(c.color() == color())
        continue;
      if(c.is_ent()) {
        load_shared_entity(c, hmap);
      }
      else {
        load_shared_node(c, hmap);
      }
    }
    // Update the lo and hi bounds
    data_field(0).lobound = key_t::min();
    data_field(0).hibound = key_t::max();
    hcell_t * root_ = &(hmap.find(key_t::root())->second);
    centroid_update(root_, hmap);
  }

  void make_tree() {
    // Cstr htable
    hmap_t hmap(hcells.span());

    // Create the tree
    size_t size = run::context::instance().colors(); // colors();
    size_t rank = run::context::instance().color(); // color();

    /* Exchange high and low bound */
    key_t lokey = e_keys(0);
    key_t hikey = e_keys(data_field(0).nents - 1);

    const auto hibound =
      rank == size - 1 ? key_t::max() : data_field(2).lobound;
    const auto lobound = rank == 0 ? key_t::min() : data_field(1).hibound;
    assert(lobound <= lokey);
    assert(hibound >= hikey);

    // Add the root
    hmap.insert(key_t::root(), key_t::root());
    auto root_ = hmap.find(key_t::root());
    root_->second.set_color(rank);
    assert(root_ != hmap.end());

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
          finish_subtree(lastnkey, hmap);
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

      parent = &(hmap.find(lastnkey)->second);
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
        parent->unset();
        parent = &(hmap.insert(nkey, nkey)->second);
        parent->set_color(rank);
      } // while

      // Recover deleted entity
      if(old_is_ent) {
        int bit = lastnkey.last_value();
        parent->add_child(bit);
        parent->unset();
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
      data_field(0).max_depth =
        std::max(data_field(0).max_depth, current_depth);

    } // for
    // Add the newly created nodes keys
    // Local traversal to discover all nodes generated 
    std::vector<hcell_t *> queue;
    std::vector<hcell_t *> nqueue;
    queue.push_back(&hmap.find(key_t::root())->second);
    while(!queue.empty()) {
      for(hcell_t * cur : queue) {
        key_t nkey = cur->key();
        if(cur->is_unset() || cur->is_node()) {
          if(cur->is_unset()){ 
            std::size_t cnode = data_field(0).nnodes++;
            n_keys(cnode) = cur->key();
          }
          for(std::size_t j = 0; j < nchildren_; ++j) {
            if(cur->has_child(j)) {
              key_t ckey = nkey;
              ckey.push(j);
              auto it = hmap.find(ckey);
              nqueue.push_back(&it->second);
            } // if
          } // for
        } // if 
      } // for
      queue = std::move(nqueue);
      nqueue.clear();
    } // while  
  } // make_tree

  // Count number of entities to send to each other color
  serdez_vector<hcell_t> top_tree_boundaries() {
    serdez_vector<hcell_t> sdata;
    hmap_t hmap(hcells.span());

    std::vector<hcell_t *> queue;
    std::vector<hcell_t *> nqueue;
    queue.push_back(&hmap.find(key_t::root())->second);
    while(!queue.empty()) {
      for(hcell_t * cur : queue) {
        cur->set_color(color());
        key_t nkey = cur->key();
        if(cur->is_unset()) {
          assert(cur->type() != 0);
          for(std::size_t j = 0; j < nchildren_; ++j) {
            if(cur->has_child(j)) {
              key_t ckey = nkey;
              ckey.push(j);
              auto it = hmap.find(ckey);
              nqueue.push_back(&it->second);
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
    hmap_t hmap(hcells.span());

    stk.push(&(hmap.find(key_t::root())->second));

    while(!stk.empty()) {
      hcell_t * cur = stk.top();
      stk.pop();
      if(cur->is_node()) {
        output << std::oct << cur->key() << std::dec << " [label=<" << std::oct
               << cur->key() << std::dec << "<br/><FONT POINT-SIZE=\"10\"> c("
               << cur->color() << ")</FONT>>,xlabel=\"" << cur->nchildren()
               << "\"]\n";
        if(cur->is_nonlocal()) {
          output << std::oct << cur->key() << std::dec
                 << " [shape=doublecircle,color=grey]" << std::endl;
        }
        else {
          output << std::oct << cur->key() << std::dec
                 << " [shape=doublecircle,color=blue]\n";
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
      else if(cur->is_unset()) {
        output << std::oct << cur->key() << std::dec << " [label=<" << std::oct
               << cur->key() << std::dec << "<br/><FONT POINT-SIZE=\"10\"> c("
               << cur->color() << ")</FONT>>,xlabel=\"" << cur->nchildren()
               << "\"]\n";
        if(cur->is_nonlocal()) {
          output << std::oct << cur->key() << std::dec
                 << " [shape=rect,color=black]" << std::endl;
        }
        else {
          output << std::oct << cur->key() << std::dec
                 << " [shape=rect,color=blue]\n";
        }
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
      } // if
    } // while
    output << "}\n";
    output.close();
  }

private:
  void finish_subtree(const key_t & key, hmap_t & hmap) {
    auto it = hmap.find(key);
    assert(it != hmap.end());
    hcell_t * n = &(it->second);
    if(n->is_ent())
      return;
    std::size_t cnode = data_field(0).nnodes++;
    n_keys(cnode) = key; 
    n->set_node_idx(cnode);
    std::vector<std::size_t> entities_ids;
    std::vector<std::size_t> nodes_ids;
    for(std::size_t i = 0; i < nchildren_; ++i) {
      if(n->has_child(i)) {
        key_t ckey = key;
        ckey.push(i);
        auto it = hmap.find(ckey);
        assert(it != hmap.end());
        if(it->second.is_ent()) {
          entities_ids.push_back(it->second.ent_idx());
        }
        else {
          assert(it->second.is_node());
          nodes_ids.push_back(it->second.node_idx());
        }
      } // if
    } // for
  }

  void load_shared_entity(hcell_t & c, hmap_t & hmap) {
    key_t key = c.key();
    assert(hmap.find(key) == hmap.end());
    auto & cur = hmap.insert(key, key)->second;
    cur.set_nonlocal();
    cur.set_color(c.color());
    cur.set_ent_idx(c.ent_idx());
    // Add missing parent(s)
    int lastbit = key.pop_value();
    add_parent(key, lastbit, c.color(), hmap);
  }

  void load_shared_node(hcell_t & c, hmap_t & hmap) {
    key_t key = c.key();
    assert(hmap.find(key) == hmap.end());
    auto & cur = hmap.insert(key, key)->second;
    // cur = c;
    // cur.set_type(c.type());
    cur.set_nonlocal();
    std::size_t cnode = data_field(0).nnodes++;
    cur.set_node_idx(cnode);
    cur.set_color(c.color());
    // Add missing parent(s)
    int lastbit = key.pop_value();
    add_parent(key, lastbit, c.color(), hmap);
  }

  void add_parent(key_t key, int child, const int & color, hmap_t & hmap) {
    auto parent = hmap.end();
    while((parent = hmap.find(key)) == hmap.end()) {
      parent = hmap.insert(key, key);
      parent->second.set_nonlocal();
      parent->second.add_child(child);
      parent->second.set_color(color);
      child = key.pop_value();
    } // while
    assert(!parent->second.is_node());
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

  void centroid_update(hcell_t * current, hmap_t & hmap) {
    key_t nkey = current->key();
    std::pair<key_t, key_t> min_max = key_boundary(nkey);
    if(min_max.first >= data_field(0).lobound &&
       min_max.second <= data_field(0).hibound) {
      if(current->is_unset()) {
        std::vector<std::size_t> entities_ids;
        std::vector<std::size_t> nodes_ids;
        for(std::size_t i = 0; i < nchildren_; ++i) {
          if(current->has_child(i)) {
            key_t ckey = nkey;
            ckey.push(i);
            auto it = hmap.find(ckey);
            assert(it != hmap.end());
            centroid_update(&(it->second), hmap);
            if(it->second.is_ent()) {
              entities_ids.push_back(it->second.ent_idx());
            }
            else {
              assert(it->second.is_node());
              nodes_ids.push_back(it->second.node_idx());
            }
          } // if
        } // for
        current->set_nonlocal();
        std::size_t cnode = data_field(0).nnodes++;
        current->set_node_idx(cnode);
        // Policy::compute_centroid(current->node_idx(),
        //  entities_ids,
        //  nodes_ids,
        //  std::forward<Ts>(args)...);
      } // if
    }
  }
};

template<>
struct detail::base<ntree> {
  using type = ntree_base;
};

} // namespace topo
} // namespace flecsi
