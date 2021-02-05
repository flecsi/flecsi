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

  static inline const field<point_t>::definition<sph_ntree_t,
    sph_ntree_t::base::entities>
    e_coordinates;
  static inline const field<double>::definition<sph_ntree_t,
    sph_ntree_t::base::entities>
    e_radius, e_masses;

  static inline const field<double>::definition<sph_ntree_t,
    sph_ntree_t::base::nodes>
    n_radius, n_masses;
  static inline const field<point_t>::definition<sph_ntree_t,
    sph_ntree_t::base::nodes>
    n_coordinates;

  static void init_fields(sph_ntree_t::accessor<wo> t,
    field<point_t>::accessor<wo> coordinates,
    field<double>::accessor<wo> radius,
    field<double>::accessor<wo> masses,
    const std::vector<sph_ntree_t::ent_t> & ents) {
    for(size_t i = 0; i < ents.size(); ++i) {
      coordinates(i) = ents[i].coordinates();
      radius(i) = ents[i].radius();
      t.e_keys(i) = ents[i].key();
      masses(i) = 1;
    }
    t.exchange_boundaries();
  } // init_task

  static void initialize(data::topology_slot<sph_ntree_t> & ts,
    coloring,
    std::vector<ent_t> & ents) {

    flecsi::execute<init_fields, flecsi::mpi>(
      ts, e_coordinates(ts), e_radius(ts), e_masses(ts), ents);

    ts->make_tree(ts);
    flecsi::execute<compute_centroid_local>(ts,
      topo::ntree<sph_ntree_t>::n_keys(ts),
      n_masses(ts),
      n_coordinates(ts),
      n_radius(ts),
      topo::ntree<sph_ntree_t>::e_keys(ts),
      e_masses(ts),
      e_coordinates(ts),
      e_radius(ts));
    flecsi::execute<compute_centroid>(ts,
      topo::ntree<sph_ntree_t>::n_keys(ts),
      n_masses(ts),
      n_coordinates(ts),
      n_radius(ts),
      topo::ntree<sph_ntree_t>::e_keys(ts),
      e_masses(ts),
      e_coordinates(ts),
      e_radius(ts));
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
    field<double>::accessor<rw, na> n_m,
    field<point_t>::accessor<rw, na> n_p,
    field<double>::accessor<rw, na> n_r,
    field<key_t>::accessor<rw, na> e_k,
    field<double>::accessor<rw, na> e_m,
    field<point_t>::accessor<rw, na> e_p,
    field<double>::accessor<rw, na> e_r) {

    int c = run::context::instance().color();

    // DFS traversal, reverse preorder, access the lowest nodes first
    for(auto n_idx : t.dfs_complete<ttype_t::reverse_preorder>()) {
      // Get entities and nodes under this node
      point_t coordinates = point_t{};
      double radius = 0;
      double mass = 0;
      // Get entities child of this node
      for(auto e_idx : t.entities(n_idx)) {
        coordinates += e_m[e_idx] * e_p[e_idx];
        mass += e_m[e_idx];
      }
      // Get nodes child of this node
      for(auto nc_idx : t.nodes(n_idx)) {
        coordinates += n_m[nc_idx] * n_p[nc_idx];
        mass += n_m[nc_idx];
      }
      assert(mass != 0.);
      coordinates /= mass;
      for(auto e_idx : t.entities(n_idx)) {
        double dist = distance(coordinates, e_p[e_idx]);
        radius = std::max(radius, dist);
      }
      for(auto nc_idx : t.nodes(n_idx)) {
        double dist = distance(coordinates, n_p[nc_idx]);
        radius = std::max(radius, dist + n_r[nc_idx]);
      }
      n_p[n_idx] = coordinates;
      n_r[n_idx] = radius;
      n_m[n_idx] = mass;
    }
  } // compute_centroid_local

  // Compute local center of masses
  // They will then be sent to other ranks to compute
  // the whole tree information
  static void compute_centroid(sph_ntree_t::accessor<rw, ro> t,
    field<key_t>::accessor<rw, ro> n_k,
    field<double>::accessor<rw, ro> n_m,
    field<point_t>::accessor<rw, ro> n_p,
    field<double>::accessor<rw, ro> n_r,
    field<key_t>::accessor<ro, ro> e_k,
    field<double>::accessor<ro, ro> e_m,
    field<point_t>::accessor<ro, ro> e_p,
    field<double>::accessor<ro, ro> e_r) {

    int c = run::context::instance().color();

    // DFS traversal, reverse preorder, access the lowest nodes first
    for(auto n_idx : t.dfs<ttype_t::reverse_preorder>()) {
      if(n_m[n_idx] == 0) {
        // Get entities and nodes under this node
        point_t coordinates = point_t{};
        double radius = 0;
        double mass = 0;
        // Get entities child of this node
        for(auto e_idx : t.entities(n_idx)) {
          coordinates += e_m[e_idx] * e_p[e_idx];
          mass += e_m[e_idx];
        }
        // Get nodes child of this node
        for(auto nc_idx : t.nodes(n_idx)) {
          coordinates += n_m[nc_idx] * n_p[nc_idx];
          mass += n_m[nc_idx];
        }
        // assert(mass != 0.);
        coordinates /= mass;
        for(auto e_idx : t.entities(n_idx)) {
          double dist = distance(coordinates, e_p[e_idx]);
          radius = std::max(radius, dist);
        }
        for(auto nc_idx : t.nodes(n_idx)) {
          double dist = distance(coordinates, n_p[nc_idx]);
          radius = std::max(radius, dist + n_r[nc_idx]);
        }
        n_p[n_idx] = coordinates;
        n_r[n_idx] = radius;
        n_m[n_idx] = mass;
      }
    }
  } // compute_centroid

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
  field<double>::accessor<wo, na> e_m,
  field<double>::accessor<wo, na> e_r) {
  double v = static_cast<double>(color());
  double tmp = 0;
  for(auto a : t.entities()) {
    p[a] = e_m[a] * e_r[a];
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
  auto e_m = sph_ntree_t::e_masses(sph_ntree);
  auto e_r = sph_ntree_t::e_radius(sph_ntree);
  flecsi::execute<init_density>(sph_ntree, d, e_m, e_r);
  flecsi::execute<print_density>(sph_ntree, d);

  return 0;
} // ntree_driver
flecsi::unit::driver<ntree_driver> driver;
