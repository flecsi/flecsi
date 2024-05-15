#include "flecsi/data.hh"
#include "flecsi/topo/ntree/interface.hh"
#include "flecsi/topo/ntree/types.hh"
#include "flecsi/util/geometry/filling_curve_key.hh"
#include "flecsi/util/geometry/point.hh"
#include "flecsi/util/unit.hh"

#include "txt_definition.hh"

using namespace flecsi;

using ngb_array_t = std::array<std::pair<util::id, bool>, 13>;
constexpr std::array<std::pair<int, int>, 13> stencil = {{{2, 0},
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
  {-1, -1}}};

using arr = topo::array<void>;
const field<int>::definition<arr> arr_f;

struct sph_ntree_t : topo::specialization<topo::ntree, sph_ntree_t> {
  static constexpr flecsi::Dimension dimension = 3;
  using key_int_t = uint64_t;
  using key_t = util::morton_key<dimension, key_int_t>;

  using index_space = flecsi::topo::ntree_base::index_space;
  using index_spaces = flecsi::topo::ntree_base::index_spaces;
  using ttype_t = flecsi::topo::ntree_base::ttype_t;

  FLECSI_INLINE_TARGET static std::size_t hash(const key_t & k) {
    return static_cast<std::size_t>(k.value() & ((1 << 22) - 1));
  }

  // In this test we only have one iteration and one stencil of 13 neighbors
  // exactly.
  static constexpr util::id max_neighbors = 13;

  template<auto>
  static constexpr std::size_t privilege_count = 2;

  using ent_t = sort_entity<dimension, double, key_t>;
  using node_t = flecsi::topo::node<dimension, double, key_t>;

  using point_t = util::point<double, dimension>;

  struct node_data {
    point_t coordinates;
    double mass;
    double radius;
  };

  // Problem: this contains the color which should not
  // depend on the topology user
  struct entity_data {
    point_t coordinates;
    double mass;
    double radius;
  };

  static void init_fields(sph_ntree_t::accessor<wo, wo> t,
    const std::vector<sph_ntree_t::ent_t> & ents) {
    auto c = process();
    for(std::size_t i = 0; i < ents.size(); ++i) {
      t.e_i(i).coordinates = ents[i].coordinates_;
      t.e_i(i).radius = ents[i].radius_;
      t.e_colors(i) = c;
      t.e_ids(i) = ents[i].id_;
      t.e_keys(i) = ents[i].key_;
      t.e_i(i).mass = ents[i].mass_;
    }
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
    const int size = processes();
    util::id hmap_size = 1 << 20;
    coloring c(size, hmap_size);

    c.entities_sizes_.resize(size);
    for(int i = 0; i < size; ++i)
      c.entities_sizes_[i] = hd.distribution();

    ents = hd.entities();

    c.nodes_sizes_ = c.entities_sizes_;
    std::for_each(c.nodes_sizes_.begin(),
      c.nodes_sizes_.end(),
      [](util::id & d) { d += 10; });

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

  template<typename T1, typename T2>
  FLECSI_INLINE_TARGET static bool intersect(const T1 & in1, const T2 & in2) {
    double dist = distance(in1.coordinates, in2.coordinates);
    return dist <= in1.radius + in2.radius;
  }

}; // sph_ntree_t

using ntree_t = topo::ntree<sph_ntree_t>;

const field<flecsi::util::id>::definition<sph_ntree_t,
  sph_ntree_t::base::entities>
  id_check;

FLECSI_INLINE_TARGET
ngb_array_t::iterator
find(ngb_array_t::iterator it,
  ngb_array_t::iterator end,
  const std::pair<util::id, bool> & v) {
  for(; it != end; ++it) {
    if(*it == v)
      break;
  }
  return it;
}

FLECSI_INLINE_TARGET
bool
verify_neighbors(flecsi::topo::id<flecsi::topo::ntree_base::entities> e,
  sph_ntree_t::accessor<rw, ro> t) {

  ngb_array_t s_id;
  for(auto & v : s_id) {
    v.first = 0;
    v.second = true;
  }
  int idx = 0;

  for(auto s : stencil) {
    int l = t.e_ids(e) / 7 + s.first;
    int c = t.e_ids(e) % 7 + s.second;
    if(l >= 0 && l < 7 && c >= 0 && c < 7) {
      s_id[idx].first = l * 7 + c;
      s_id[idx++].second = false;
    }
  }

  for(auto n : t.neighbors(e)) {
    auto f = find(
      s_id.begin(), s_id.end(), std::pair<util::id, bool>(t.e_ids[n], false));
    assert(f != s_id.end());
    f->second = true;
  }
  for(auto a : s_id)
    if(!a.second)
      return false;
  return true;
}

int
check_neighbors(sph_ntree_t::accessor<rw, ro> t) {
  UNIT("CHECK_NEIGHBORS") {
    // Check neighbors of entities
    for(auto e : t.entities())
      EXPECT_TRUE(verify_neighbors(e, t));
  };
}

void
check_neighbors_accelerator(sph_ntree_t::accessor<rw, ro> t) {
  forall(e, t.entities(), "test_gpu") {
    [[maybe_unused]] auto v = verify_neighbors(e, t);
    assert(v);
  };
}

void
init_ids(sph_ntree_t::accessor<ro, na> t,
  field<flecsi::util::id>::accessor<wo, na> p) {
  forall(a, t.entities(), "Initialize") { p[a] = t.e_ids(a); };
}

void
print_ids(sph_ntree_t::accessor<ro, ro> t,
  field<flecsi::util::id>::accessor<ro, ro> d) {
  std::cout << color() << " Print id exclusive: ";
  for(auto a : t.entities()) {
    std::cout << t.e_ids[a] << "=" << d(a) << " - ";
    assert(t.e_ids[a] == d(a));
  }
  std::cout << std::endl;
  std::cout << color() << " Print id ghosts : ";
  for(auto a : t.entities<sph_ntree_t::base::ptype_t::ghost>()) {
    std::cout << t.e_ids[a] << "=" << d(a) << " - ";
    assert(t.e_ids[a] == d(a));
  }
  std::cout << std::endl;
  std::cout << color() << " Print id all : ";
  for(auto id : t.e_ids.span()) {
    std::cout << id << " - ";
  }
  std::cout << std::endl;
}

void
move_entities(sph_ntree_t::accessor<rw, na> t) {
  forall(a, t.entities(), "Move entities") {
    // Add 1 on z coordinate
    t.e_i[a].coordinates[2] += 1;
  };
}

// Sort testing tasks
void
init_array_task(field<int>::accessor<wo> v) {
  std::iota(v.span().begin(), v.span().end(), 0);
  auto rng = std::default_random_engine{};
  std::shuffle(v.span().begin(), v.span().end(), rng);
}

auto
check_sort_task(typename field<int>::accessor<ro> v) {
  bool sorted = std::is_sorted(v.span().begin(), v.span().end());
  std::tuple<int, int, bool> rt = {v[0], v.span().back(), sorted};
  return rt;
}

int
ntree_driver() {
  UNIT("NTREE") {
    sph_ntree_t::slot sph_ntree;

    {
      std::vector<sph_ntree_t::ent_t> ents;
      sph_ntree_t::mpi_coloring coloring("coordinates.blessed", ents);
      sph_ntree.allocate(coloring, ents);
    }

    auto d = id_check(sph_ntree);

    flecsi::execute<init_ids, default_accelerator>(sph_ntree, d);
    flecsi::execute<print_ids>(sph_ntree, d);
    EXPECT_EQ(test<check_neighbors>(sph_ntree), 0);
    flecsi::execute<check_neighbors_accelerator, default_accelerator>(
      sph_ntree);

    flecsi::execute<move_entities, default_accelerator>(sph_ntree);

    // Sort utility testing
    // Sort/shuffle an array multiple times
    arr::slot arr_s;
    arr_s.allocate(arr::coloring(4, 100));

    util::sort s(arr_f(arr_s));

    for(int i = 0; i < 10; ++i) {
      flecsi::execute<init_array_task>(arr_f(arr_s));
      s();
      auto fm = flecsi::execute<check_sort_task>(arr_f(arr_s));
      EXPECT_TRUE(std::get<2>(fm.get(process())));
      for(unsigned int p = 0; p < processes() - 1; ++p) {
        EXPECT_LE(std::get<0>(fm.get(p)), std::get<1>(fm.get(p + 1)));
      }
    }
  };
} // ntree_driver
util::unit::driver<ntree_driver> driver;
