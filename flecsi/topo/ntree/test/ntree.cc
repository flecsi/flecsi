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
    sph_ntree_t::base::entities>
    n_radius, n_masses, n_lap;
  static inline const field<point_t>::definition<sph_ntree_t,
    sph_ntree_t::base::entities>
    n_coordinates, n_bmin, n_bmax;

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

    c.local_nodes_ = c.local_entities_;
    c.global_nodes_ = c.global_entities_;
    c.nodes_offset_ = c.entities_offset_;

    c.global_sizes_.resize(4);
    c.global_sizes_[0] = c.global_entities_;
    c.global_sizes_[1] = c.global_nodes_;
    c.global_sizes_[2] = c.global_hmap_;
    c.global_sizes_[3] = c.nparts_;

    return c;
  } // color

  static void compute_centroid(const std::size_t & cur_node_idx,
    const std::vector<std::size_t> & ent_idx,
    const std::vector<std::size_t> & node_idx,
    field<point_t>::accessor<ro> e_c,
    field<double>::accessor<ro> e_r,
    field<double>::accessor<ro> e_m,
    field<point_t>::accessor<rw> n_c,
    field<double>::accessor<rw> n_r,
    field<double>::accessor<rw> n_m,
    field<point_t>::accessor<rw> n_bi,
    field<point_t>::accessor<rw> n_ba,
    field<double>::accessor<rw> n_l) {
    point_t coordinates = point_t{};
    double radius = 0; // bmax
    double mass = 0;
    point_t bmin, bmax;
    double lap = 0;
    for(std::size_t i = 0; i < dimension; ++i) {
      bmax[i] = -DBL_MAX;
      bmin[i] = DBL_MAX;
    }
    // Compute the center of mass and mass
    for(auto idx : ent_idx) {
      coordinates += e_m[idx] * e_c[idx];
      mass += e_m[idx];
      for(std::size_t d = 0; d < dimension; ++d) {
        bmin[d] = std::min(bmin[d], e_c[idx][d] - e_r[idx] / 2.);
        bmax[d] = std::max(bmax[d], e_c[idx][d] + e_r[idx] / 2.);
      } // for
    }
    for(auto idx : node_idx) {
      coordinates += n_m[idx] * n_c[idx];
      mass += n_m[idx];
      for(std::size_t d = 0; d < dimension; ++d) {
        bmin[d] = std::min(bmin[d], n_bi[idx][d]);
        bmax[d] = std::max(bmax[d], n_ba[idx][d]);
      } // for
    } // for
    assert(mass != 0.);
    // Compute the radius
    coordinates /= mass;
    for(auto idx : ent_idx) {
      double dist = distance(coordinates, e_c[idx]);
      radius = std::max(radius, dist);
      lap = std::max(lap, dist + e_r[idx]);
    }
    for(auto idx : node_idx) {
      double dist = distance(coordinates, n_c[idx]);
      radius = std::max(radius, dist + n_r[idx]);
      lap = std::max(lap, dist + n_r[idx] + n_l[idx]);
    } // for
    n_c[cur_node_idx] = coordinates;
    n_r[cur_node_idx] = radius;
    n_m[cur_node_idx] = mass;
    n_bi[cur_node_idx] = bmin;
    n_ba[cur_node_idx] = bmax;
    n_l[cur_node_idx] = lap;
  } // compute_centroid

}; // sph_ntree_t

using ntree_t = topo::ntree<sph_ntree_t>;

sph_ntree_t::slot sph_ntree;
sph_ntree_t::cslot coloring;

const field<double>::definition<sph_ntree_t, sph_ntree_t::base::entities>
  density;
const field<double>::definition<sph_ntree_t, sph_ntree_t::base::entities>
  pressure;

void init_density(sph_ntree_t::accessor<ro> t, field<double>::accessor<wo, na> p){
  std::cout<<color()<<" Init density ("<<p.span().size()<<")"<<std::endl;
  double v = static_cast<double>(color());
  double tmp = 0; 
  for(auto a: t.entities()){
    p[a] = v*5+tmp++; 
  }
}

void update_density(sph_ntree_t::accessor<ro> t, field<double>::accessor<rw, ro> p){
  std::cout<<color()<<" Update density ("<<p.span().size()<<")"<<std::endl;
  for(auto a: t.entities()){
    p[a] = p[a]*10; 
  }
}

void print_density(sph_ntree_t::accessor<ro> t, field<double>::accessor<ro, ro> p){
  std::cout<<color()<<" Print density exclusive: ";
  for(auto a: t.entities()){
    std::cout<<p[a]<<" - "; 
  }
  std::cout<<std::endl;
  std::cout<<color()<<" Print density ghosts: ";
  for(auto a: t.entities<sph_ntree_t::base::ptype_t::ghost>()){
    std::cout<<p[a]<<" - "; 
  }
  std::cout<<std::endl;
}


int
ntree_driver() {

  std::vector<sph_ntree_t::ent_t> ents;
  coloring.allocate("coordinates.blessed", ents);
  sph_ntree.allocate(coloring.get(), ents);

  auto d = density(sph_ntree); 
  flecsi::execute<init_density>(sph_ntree,d);
  flecsi::execute<update_density>(sph_ntree,d); 
  flecsi::execute<print_density>(sph_ntree,d);  

  return 0;
} // ntree_driver
flecsi::unit::driver<ntree_driver> driver;
