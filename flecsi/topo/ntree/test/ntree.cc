/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Los Alamos National Security, LLC
   All rights reserved.
                                                                              */
#define __FLECSI_PRIVATE__
#include <flecsi/data.hh>

#include "flecsi/util/geometry/filling_curve.hh"
#include "flecsi/util/geometry/point.hh"
#include "flecsi/util/unit.hh"

#include "flecsi/topo/ntree/interface.hh"
#include "flecsi/topo/ntree/types.hh"

#include "txt_definition.hh"

using namespace flecsi;

struct sph_ntree_t : topo::specialization<topo::ntree, sph_ntree_t> {
  static constexpr unsigned int dimension = 3;
  using key_int_t = uint64_t;
  using key_t = morton_curve<dimension, key_int_t>;

  using index_space = flecsi::topo::ntree_base::index_space;
  using index_spaces = flecsi::topo::ntree_base::index_spaces;
  using ttype_t = flecsi::topo::ntree_base::ttype_t;

  struct key_t_hasher {
    std::size_t operator()(const key_t & k) const noexcept {
      return static_cast<std::size_t>(k.value() & ((1 << 22) - 1));
    }
  };

  using hash_f = key_t_hasher;

  using ent_t = flecsi::topo::sort_entity<dimension, double, key_t>;
  using node_t = flecsi::topo::node<dimension, double, key_t>;

  using point_t = util::point<double, dimension>;

  struct interaction_nodes {
    point_t coordinates;
    double mass;
    double radius;
  };

  // Problem: this contains the color which should not 
  // depend on the topology user 
  struct interaction_entities {
    std::size_t color; 
    point_t coordinates;
    double mass;
    double radius;
  };

  static inline const field<interaction_entities>::definition<sph_ntree_t,
    sph_ntree_t::base::entities>
    e_i;

  static inline const field<interaction_nodes>::definition<sph_ntree_t,
    sph_ntree_t::base::nodes>
    n_i;

  static void init_fields(sph_ntree_t::accessor<wo> t,
    field<interaction_entities>::accessor<wo> e_i,
    const std::vector<sph_ntree_t::ent_t> & ents) {
    auto c = process();
    for(size_t i = 0; i < ents.size(); ++i) {
      e_i(i).coordinates = ents[i].coordinates();
      e_i(i).radius = ents[i].radius();
      e_i(i).color = c; 
      t.e_keys(i) = ents[i].key();
      e_i(i).mass = ents[i].mass();
    }
    t.exchange_boundaries();
  } // init_task

  static void initialize(data::topology_slot<sph_ntree_t> & ts,
    coloring,
    std::vector<ent_t> & ents) {

    flecsi::execute<init_fields, flecsi::mpi>(ts, e_i(ts), ents);

    ts->make_tree(ts);

    flecsi::execute<compute_centroid_local>(ts,
      topo::ntree<sph_ntree_t>::n_keys(ts),
      n_i(ts),
      topo::ntree<sph_ntree_t>::e_keys(ts),
      e_i(ts));
    flecsi::execute<compute_centroid>(ts,
      topo::ntree<sph_ntree_t>::n_keys(ts),
      n_i(ts),
      topo::ntree<sph_ntree_t>::e_keys(ts),
      e_i(ts));

    ts->share_ghosts(ts, e_i, n_i);

    // Recompute center of masses to include new entities/nodes
    // See if it is mandatory 
  }

  static coloring color(const std::string & name, std::vector<ent_t> & ents) {
    txt_definition<key_t, dimension> hd(name);
    int size, rank;
    rank = process();
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    coloring c(size);

    c.global_entities_ = hd.global_num_entities();
    c.entities_distribution_.resize(size);
    for(int i = 0; i < size; ++i)
      c.entities_distribution_[i] = hd.distribution(i);
    c.entities_offset_.resize(size);

    ents = hd.entities();

    std::ostringstream oss;
    if(rank == 0)
      oss << "Ents Offset: ";

    for(int i = 0; i < size; ++i) {
      c.entities_offset_[i] = c.entities_distribution_[i];
      if(rank == 0)
        oss << c.entities_offset_[i] << " ; ";
    }

    if(rank == 0)
      flog(info) << oss.str() << std::endl;

    c.local_nodes_ = c.local_entities_ + 10;
    c.global_nodes_ = c.global_entities_;
    c.nodes_offset_ = c.entities_offset_;
    std::for_each(c.nodes_offset_.begin(),
      c.nodes_offset_.end(),
      [](std::size_t & d) { d += 10; });

    c.global_sizes_.resize(4);
    c.global_sizes_[0] = c.global_entities_;
    c.global_sizes_[1] = c.global_nodes_;
    c.global_sizes_[2] = c.global_hmap_;
    c.global_sizes_[3] = c.nparts_;

    return c;
  } // color

  // Compute local center of masses
  // They will then be sent to other ranks to compute
  // the whole tree information
  static void compute_centroid_local(sph_ntree_t::accessor<rw, na> t,
    field<key_t>::accessor<rw, na> n_k,
    field<interaction_nodes>::accessor<rw, na> n_i,
    field<key_t>::accessor<rw, na> e_k,
    field<interaction_entities>::accessor<rw, na> e_i) {

    // DFS traversal, reverse preorder, access the lowest nodes first
    for(auto n_idx : t.dfs_complete<ttype_t::reverse_preorder>()) {
      // Get entities and nodes under this node
      point_t coordinates = 0.;
      double radius = 0;
      double mass = 0;
      // Get entities child of this node
      for(auto e_idx : t.entities(n_idx)) {
        coordinates += e_i[e_idx].mass * e_i[e_idx].coordinates;
        mass += e_i[e_idx].mass;
      }
      // Get nodes child of this node
      for(auto nc_idx : t.nodes(n_idx)) {
        coordinates += n_i[nc_idx].mass * n_i[nc_idx].coordinates;
        mass += n_i[nc_idx].mass;
      }
      assert(mass != 0.);
      coordinates /= mass;
      for(auto e_idx : t.entities(n_idx)) {
        double dist = distance(coordinates, e_i[e_idx].coordinates);
        radius = std::max(radius, dist + e_i[e_idx].radius);
      }
      for(auto nc_idx : t.nodes(n_idx)) {
        double dist = distance(coordinates, n_i[nc_idx].coordinates);
        radius = std::max(radius, dist + n_i[nc_idx].radius);
      }
      n_i[n_idx].coordinates = coordinates;
      n_i[n_idx].radius = radius;
      n_i[n_idx].mass = mass;
    }
  } // compute_centroid_local

  // Compute local center of masses
  // They will then be sent to other ranks to compute
  // the whole tree information
  static void compute_centroid(sph_ntree_t::accessor<rw, ro> t,
    field<key_t>::accessor<rw, ro> n_k,
    field<interaction_nodes>::accessor<rw, ro> n_i,
    field<key_t>::accessor<ro, ro> e_k,
    field<interaction_entities>::accessor<ro, ro> e_i) {

    // DFS traversal, reverse preorder, access the lowest nodes first
    for(auto n_idx : t.dfs<ttype_t::reverse_preorder>()) {
      if(n_i[n_idx].mass == 0) {
        // Get entities and nodes under this node
        point_t coordinates = 0.;
        double radius = 0;
        double mass = 0;
        // Get entities child of this node
        for(auto e_idx : t.entities(n_idx)) {
          coordinates += e_i[e_idx].mass * e_i[e_idx].coordinates;
          mass += e_i[e_idx].mass;
        }
        // Get nodes child of this node
        for(auto nc_idx : t.nodes(n_idx)) {
          coordinates += n_i[nc_idx].mass * n_i[nc_idx].coordinates;
          mass += n_i[nc_idx].mass;
        }
        assert(mass != 0.);
        coordinates /= mass;
        for(auto e_idx : t.entities(n_idx)) {
          double dist = distance(coordinates, e_i[e_idx].coordinates);
          radius = std::max(radius, dist + e_i[e_idx].radius);
        }
        for(auto nc_idx : t.nodes(n_idx)) {
          double dist = distance(coordinates, n_i[nc_idx].coordinates);
          radius = std::max(radius, dist + n_i[nc_idx].radius);
        }
        n_i[n_idx].coordinates = coordinates;
        n_i[n_idx].radius = radius;
        n_i[n_idx].mass = mass;
      }
      if(n_idx == 0) {
        std::cout << "Total mass: " << n_i[0].mass << std::endl;
      }
    }
  } // compute_centroid

  static bool intersect_entity_node(topo::id<topo::ntree_base::entities> e_idx,
    topo::id<topo::ntree_base::nodes> n_idx,
    field<interaction_entities>::accessor<ro, ro> e_i,
    field<interaction_nodes>::accessor<ro, ro> n_i) {
    double dist = distance(n_i[n_idx].coordinates, e_i[e_idx].coordinates);
    if(dist <= n_i[n_idx].radius + e_i[e_idx].radius)
      return true;
    return false;
  }

  static bool intersect_node_node(topo::id<topo::ntree_base::nodes> node_1,
    topo::id<topo::ntree_base::nodes> node_2,
    field<interaction_nodes>::accessor<ro, ro> n_i) {
    double dist = distance(n_i[node_1].coordinates, n_i[node_2].coordinates);
    if(dist <= n_i[node_1].radius + n_i[node_2].radius)
      return true;
    return false;
  }

  static bool intersect_entity_entity(
    topo::id<topo::ntree_base::entities> ent_1,
    topo::id<topo::ntree_base::entities> ent_2,
    field<interaction_entities>::accessor<ro, ro> e_i) {
    double dist = distance(e_i[ent_1].coordinates, e_i[ent_2].coordinates);
    if(dist <= e_i[ent_1].radius + e_i[ent_2].radius)
      return true;
    return false;
  }

}; // sph_ntree_t

using ntree_t = topo::ntree<sph_ntree_t>;

sph_ntree_t::slot sph_ntree;
sph_ntree_t::cslot coloring;

const field<double>::definition<sph_ntree_t, sph_ntree_t::base::entities>
  density;
const field<double>::definition<sph_ntree_t, sph_ntree_t::base::entities>
  pressure;

void
init_density(sph_ntree_t::accessor<ro> t,
  field<double>::accessor<wo, na> p,
  field<sph_ntree_t::interaction_entities>::accessor<wo, na> e_i) {
  for(auto a : t.entities()) {
    p[a] = e_i[a].mass * e_i[a].radius;
  }
}

void
print_density(sph_ntree_t::accessor<ro> t, field<double>::accessor<ro, ro> p) {
  std::cout << color() << " Print density exclusive: ";
  for(auto a : t.entities()) {
    std::cout << p[a] << " - ";
  }
  std::cout << std::endl;
  std::cout << color() << " Print density ghosts: ";
  for(auto a : t.entities<sph_ntree_t::base::ptype_t::ghost>()) {
    std::cout << p[a] << " - ";
  }
  std::cout << std::endl;
}

int
ntree_driver() {

  std::vector<sph_ntree_t::ent_t> ents;
  coloring.allocate("coordinates.blessed", ents);
  sph_ntree.allocate(coloring.get(), ents);

  auto d = density(sph_ntree);
  auto e_i = sph_ntree_t::e_i(sph_ntree);
  flecsi::execute<init_density>(sph_ntree, d, e_i);
  flecsi::execute<print_density>(sph_ntree, d);

  return 0;
} // ntree_driver
flecsi::unit::driver<ntree_driver> driver;
