// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef TUTORIAL_6_TOPOLOGY_NTREE_SPH_HH
#define TUTORIAL_6_TOPOLOGY_NTREE_SPH_HH

#include "flecsi/data.hh"
#include "flecsi/topo/ntree/interface.hh"
#include "flecsi/topo/ntree/types.hh"
#include "flecsi/util/geometry/filling_curve_key.hh"
#include "flecsi/util/geometry/point.hh"
#include "sph_physics.hh"

// Sort entities used to create the initial ntree
template<flecsi::Dimension DIM, typename T, class KEY>
struct sort_entity {
  using point_t = flecsi::util::point<T, DIM>;
  using key_t = KEY;
  using type_t = T;

  // Compare entities using key and id
  bool operator<(const sort_entity & s) const {
    return std::tie(key_, id_) < std::tie(s.key_, s.id_);
  }

  key_t key_;
  int64_t id_;
  point_t coordinates_;
  type_t mass_;
  type_t radius_;
}; // class sort_entity

struct sph_ntree_t
  : flecsi::topo::specialization<flecsi::topo::ntree, sph_ntree_t> {

  //-------------------- Base policy inputs --------------------- //
  static constexpr unsigned int dimension = 1;
  using key_int_t = uint64_t;
  using key_t = flecsi::util::morton_key<dimension, key_int_t>;
  // In this hashing function we use the low bits (less than 22) to scatter the
  // leaves but gather the roots. This is particularly efficient since the roots
  // are accessed more often during the neighbor search.
  static std::size_t hash(const key_t & k) {
    return static_cast<std::size_t>(k.value() & ((1 << 22) - 1));
  }
  template<auto>
  static constexpr std::size_t privilege_count = 2;

  using index_space = base::index_space;
  using index_spaces = base::index_spaces;

  //-------------- User description of interactions ------------- //

  using point_t = flecsi::util::point<double, dimension>;

  struct node_data {
    point_t coordinates;
    double mass;
    double radius;
  }; // struct node_data

  struct entity_data {
    point_t coordinates;
    double mass;
    double radius;
  }; // struct entity_data

  // In this implementation the interactions are computed using spheres.
  // If the two spheres are in contact, there is an interaction.
  template<typename T1, typename T2>
  static bool intersect(const T1 & in1, const T2 & in2) {
    return distance(in1.coordinates, in2.coordinates) <=
           in1.radius + in2.radius;
  } // intersect

  // --------- Reduction to compute range and then keys ---------- //

  using range_t = std::array<point_t, 2>;

  // Reduction functor to reduce both min and max coordinate at once
  struct minmax {
    static range_t combine(range_t a, range_t b) {
      range_t res;
      for(unsigned int d = 0; d < dimension; ++d) {
        res[0][d] = std::min(a[0][d], b[0][d]);
        res[1][d] = std::max(a[1][d], b[1][d]);
      }
      return res;
    }
    // This is rejected by GCC without the template (#97340)
    template<typename>
    static constexpr range_t identity = {std::numeric_limits<double>::max(),
      std::numeric_limits<double>::lowest()};
  };

  // Compute the size of the domain: reduction on maximum and minimum entities
  // coordinates
  static void range_task(sph_ntree_t::accessor<flecsi::ro, flecsi::na> ts,
    flecsi::field<range_t>::reduction<minmax> red) {
    range_t r = {ts.e_i[0].coordinates, ts.e_i[0].coordinates};
    for(auto & e : ts.e_i.span()) {
      for(unsigned int d = 0; d < dimension; ++d) {
        r[d][0] = std::min(r[0][d], e.coordinates[d] - e.radius);
        r[d][1] = std::max(r[1][d], e.coordinates[d] + e.radius);
      }
    }
    red[0](r);
  }

  // Compute the keys (aka filling curve) for each entities.
  // The keys are cnmputed using the domain (min/max coordinates) and the
  // coordinates of the entity.
  static void keys_task(sph_ntree_t::accessor<flecsi::rw, flecsi::wo> ts,
    flecsi::field<range_t>::accessor<flecsi::ro> r) {
    for(std::size_t e = 0; e < ts.e_i.span().size(); ++e) {
      ts.e_keys[e] = key_t(r[0], ts.e_i[e].coordinates);
    }
  }

  static void init_reduction(flecsi::field<range_t>::accessor<flecsi::wo> r) {
    r[0] = minmax::identity<range_t>;
  }

  const static inline flecsi::field<range_t>::definition<flecsi::topo::global>
    range_reduction_f;

  // --------------- Initialize fields and build N-tree ----------------- //

  using ent_t = sort_entity<dimension, double, key_t>;

  // Feed the index space / fields with initial information for the entities
  static void init_fields(sph_ntree_t::accessor<flecsi::wo, flecsi::wo> t,
    const std::vector<sph_ntree_t::ent_t> & ents) {
    auto c = flecsi::process();
    for(std::size_t i = 0; i < ents.size(); ++i) {
      t.e_i(i).coordinates = ents[i].coordinates_;
      t.e_i(i).radius = ents[i].radius_;
      t.e_colors(i) = c;
      t.e_i(i).mass = ents[i].mass_;
    }
  } // init_fields

  // Compute the range of the domain, the keys for each entities and generate
  // the N-Tree data structure
  static void generate_ntree(flecsi::data::topology_slot<sph_ntree_t> & ts) {
    // Initialize key values: compute range and keys
    {
      flecsi::topo::global::slot red;
      red.allocate(1);
      auto r = range_reduction_f(red);
      flecsi::execute<init_reduction>(r);
      flecsi::execute<range_task>(ts, r);
      flecsi::execute<keys_task>(ts, r);
    }

    ts->make_tree(ts);
    flecsi::execute<compute_centroid<true>>(ts);
    flecsi::execute<compute_centroid<false>>(ts);
    ts->share_ghosts(ts);
  } // generate_ntree

  static void initialize(flecsi::data::topology_slot<sph_ntree_t> & ts,
    coloring,
    std::vector<ent_t> & ents) {
    flecsi::execute<init_fields>(ts, ents);
    generate_ntree(ts);
  }

  // Reset and recreate the N-Tree data structure following a change in position
  // of the entities
  static void sph_reset(flecsi::data::topology_slot<sph_ntree_t> & ts) {
    // Reset the data structure
    core::reset(ts);
    generate_ntree(ts);
  }

  // N-Tree coloring
  static coloring color(flecsi::util::id nents, std::vector<ent_t> & ents) {
    const int size = flecsi::processes(), rank = flecsi::process();
    const flecsi::util::id hmap_size = 1 << 20;
    coloring c(size, hmap_size);
    c.entities_sizes_.resize(size);
    std::vector<flecsi::util::id> offset(size);

    int lm = nents % size;
    for(int i = 0; i < size; ++i) {
      c.entities_sizes_[i] = nents / size;
      if(i < lm)
        ++c.entities_sizes_[i];
      if(i > 0)
        offset[i] += offset[i - 1] + c.entities_sizes_[i];
    }

    // Feed default values
    ents.resize(c.entities_sizes_[rank]);
    sph::init_base(ents, nents, offset[rank]);

    c.nodes_sizes_ = c.entities_sizes_;
    for(flecsi::util::id & d : c.nodes_sizes_)
      d += 100;

    return c;
  } // color

  using ttype_t = flecsi::topo::ntree_base::ttype_t; // Tree traversal types

  // Compute local center of masses
  // They will then be sent to other ranks to compute
  // the whole tree information
  template<bool local = false>
  static void compute_centroid(
    sph_ntree_t::accessor<flecsi::rw, flecsi::ro> t) {
    // DFS traversal, reverse preorder, access the lowest nodes first
    for(auto n_idx : t.dfs<ttype_t::reverse_preorder, local>()) {
      // Get entities and nodes under this node
      if(local || t.n_i[n_idx].mass == 0) {
        point_t coordinates = 0.;
        double radius = 0;
        double mass = 0;
        // Get entities child of this node
        for(auto e_idx : t.entities(n_idx)) {
          coordinates += t.e_i[e_idx].mass * t.e_i[e_idx].coordinates;
          mass += t.e_i[e_idx].mass;
        }
        // Get nodes child of this node
        for(auto nc_idx : t.nodes(n_idx)) {
          coordinates += t.n_i[nc_idx].mass * t.n_i[nc_idx].coordinates;
          mass += t.n_i[nc_idx].mass;
        }
        assert(mass != 0.);
        coordinates /= mass;
        for(auto e_idx : t.entities(n_idx)) {
          double dist = distance(coordinates, t.e_i[e_idx].coordinates);
          radius = std::max(radius, dist + t.e_i[e_idx].radius);
        }
        for(auto nc_idx : t.nodes(n_idx)) {
          double dist = distance(coordinates, t.n_i[nc_idx].coordinates);
          radius = std::max(radius, dist + t.n_i[nc_idx].radius);
        }
        t.n_i[n_idx].coordinates = coordinates;
        t.n_i[n_idx].radius = radius;
        t.n_i[n_idx].mass = mass;
      }
    }
  } // compute_centroid

  // Override default behavior for the default index space
  // The fields added by the user will be stored in the entities index space by
  // default.
  static constexpr auto default_space() {
    return base::entities;
  }

}; // sph_ntree_t

#endif
