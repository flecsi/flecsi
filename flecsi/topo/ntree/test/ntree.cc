#include "flecsi/data.hh"
#include "flecsi/topo/ntree/interface.hh"
#include "flecsi/topo/ntree/types.hh"
#include "flecsi/util/geometry/filling_curve.hh"
#include "flecsi/util/geometry/point.hh"
#include "flecsi/util/unit.hh"

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

  template<auto>
  static constexpr std::size_t privilege_count = 2;

  using hash_f = key_t_hasher;

  using ent_t = flecsi::topo::sort_entity<dimension, double, key_t>;
  using node_t = flecsi::topo::node<dimension, double, key_t>;

  using point_t = util::point<double, dimension>;

  struct interaction_nodes {
    int id;
    point_t coordinates;
    double mass;
    double radius;
  };

  // Problem: this contains the color which should not
  // depend on the topology user
  struct interaction_entities {
    int id;
    std::size_t color;
    point_t coordinates;
    double mass;
    double radius;
  };

  static void init_fields(sph_ntree_t::accessor<wo, na> t,
    const std::vector<sph_ntree_t::ent_t> & ents) {
    auto c = process();
    for(std::size_t i = 0; i < ents.size(); ++i) {
      t.e_i(i).coordinates = ents[i].coordinates();
      t.e_i(i).radius = ents[i].radius();
      t.e_i(i).color = c;
      t.e_i(i).id = ents[i].id();
      t.e_keys(i) = ents[i].key();
      t.e_i(i).mass = ents[i].mass();
    }
    t.exchange_boundaries();
  } // init_fields

  static void initialize(data::topology_slot<sph_ntree_t> & ts,
    coloring,
    std::vector<ent_t> & ents) {

    flecsi::execute<init_fields>(ts, ents);

    ts->make_tree(ts);

    flecsi::execute<compute_centroid<true>>(ts);
    flecsi::execute<compute_centroid<false>>(ts);

    ts->share_ghosts(ts);
  }

  static coloring color(const std::string & name, std::vector<ent_t> & ents) {
    txt_definition<key_t, dimension> hd(name);
    const int size = processes(), rank = process();

    coloring c(size);

    c.global_entities_ = hd.global_num_entities();
    c.entities_distribution_.resize(size);
    for(int i = 0; i < size; ++i)
      c.entities_distribution_[i] = hd.distribution();
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

    c.local_nodes_ = c.local_entities_ + 20;
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
  template<bool local = false>
  static void compute_centroid(sph_ntree_t::accessor<rw, ro> t) {

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

  static bool intersect_entity_node(const interaction_entities & ie,
    const interaction_nodes & in) {
    double dist = distance(in.coordinates, ie.coordinates);
    if(dist <= in.radius + ie.radius)
      return true;
    return false;
  }

  static bool intersect_node_node(const interaction_nodes & in1,
    const interaction_nodes & in2) {
    double dist = distance(in1.coordinates, in2.coordinates);
    if(dist <= in1.radius + in2.radius)
      return true;
    return false;
  }

  static bool intersect_entity_entity(const interaction_entities & ie1,
    const interaction_entities & ie2) {
    double dist = distance(ie1.coordinates, ie2.coordinates);
    if(dist <= ie1.radius + ie2.radius)
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
check_neighbors(sph_ntree_t::accessor<rw, ro> t) {
  std::vector<std::pair<int, int>> stencil = {{2, 0},
    {1, 0},
    {0, 0},
    {-1, 0},
    {-2, 0},
    {0, 2},
    {0, 1},
    {0, -1},
    {0, -2},
    {1, 1},
    {1, -1},
    {-1, 1},
    {-1, -1}};
  // Check neighbors of entities
  for(auto e : t.entities()) {
    std::vector<std::pair<std::size_t, bool>> s_id; // stencil ids
    std::size_t eid = t.e_i(e).id;
    // Compute stencil based on id
    int line = eid / 7;
    int col = eid % 7;
    for(int i = 0; i < static_cast<int>(stencil.size()); ++i) {
      int l = line + stencil[i].first;
      int c = col + stencil[i].second;
      if(l >= 0 && l < 7)
        if(c >= 0 && c < 7)
          s_id.push_back(std::make_pair(l * 7 + c, false));
    }
    for(auto n : t.neighbors(e)) {
      std::size_t n_id = t.e_i[n].id;
      auto f = std::find(s_id.begin(), s_id.end(), std::make_pair(n_id, false));
      assert(f != s_id.end());
      f->second = true;
    }
#ifdef DEBUG
    for(auto a : s_id)
      assert(a.second == true);
#endif
  }
}

void
init_density(sph_ntree_t::accessor<ro, na> t,
  field<double>::accessor<wo, na> p) {
  for(auto a : t.entities()) {
    p[a] = t.e_i[a].mass * t.e_i[a].radius;
  }
}

void
print_density(sph_ntree_t::accessor<ro, na> t,
  field<double>::accessor<ro, ro>) {
  std::cout << color() << " Print id exclusive: ";
  for(auto a : t.entities()) {
    std::cout << t.e_i[a].id << " - ";
  }
  std::cout << std::endl;
  std::cout << color() << " Print id ghosts : ";
  for(auto a : t.entities<sph_ntree_t::base::ptype_t::ghost>()) {
    std::cout << t.e_i[a].id << " - ";
  }
  std::cout << std::endl;
  std::cout << color() << " Print id all : ";
  for(auto a : t.e_i.span()) {
    std::cout << a.id << " - ";
  }
  std::cout << std::endl;
}

void
move_entities(sph_ntree_t::accessor<rw, na> t) {
  for(auto a : t.entities()) {
    // Add 1 on z coordinate
    t.e_i[a].coordinates[2] += 1;
  }
}

int
ntree_driver() {

  std::vector<sph_ntree_t::ent_t> ents;
  coloring.allocate("coordinates.blessed", ents);
  sph_ntree.allocate(coloring.get(), ents);

  auto d = density(sph_ntree);

  flecsi::execute<init_density>(sph_ntree, d);
  flecsi::execute<print_density>(sph_ntree, d);
  flecsi::execute<check_neighbors>(sph_ntree);

  flecsi::execute<move_entities>(sph_ntree);
  // sph_ntree_t::update(sph_ntree);

  return 0;
} // ntree_driver
util::unit::driver<ntree_driver> driver;
