#include "flecsi/data.hh"
#include "flecsi/exec/kernel.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/topo/unstructured/types.hh"
#include "flecsi/util/color_map.hh"
#include "flecsi/util/common.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/unit.hh"

#include <algorithm>
#include <vector>

using namespace flecsi;

namespace ftui = topo::unstructured_impl;

// clang-format off
namespace {

constexpr Color ncolors = 4;
constexpr std::size_t num_cells = 4 * 4, num_vertices = 5 * 5;

const std::vector<std::vector<util::gid>> cell_l2g {
  {2, 3, 6, 7, 1, 5, 9, 10, 11},
  {0, 1, 4, 5, 2, 6, 8,  9, 10},
  {10, 11, 14, 15, 5, 6, 7, 9, 13},
  {8, 9, 12, 13, 4, 5, 6, 10, 14}
};

const std::vector<std::vector<util::gid>> vertex_l2g {
  { 2,  3,  4,  7,  8,  9, 12, 13, 14,  1,  6, 11, 16, 17, 18, 19},
  { 0,  1,  5,  6, 10, 11,  2,  3,  7,  8, 12, 13, 15, 16, 17, 18},
  {17, 18, 19, 22, 23, 24,  6,  7,  8,  9, 11, 12, 13, 14, 16, 21},
  {15, 16, 20, 21,  5,  6,  7,  8, 10, 11, 12, 13, 17, 18, 22, 23},
};

const std::vector<util::crs> local_connectivity {
  {{util::equal_map(9 * 4, 9)}, {
    // owned
    0, 1, 4, 3,
    1, 2, 5, 4,
    3, 4, 7, 6,
    4, 5, 8, 7,
    // ghost
    9, 0, 3, 10,
    10, 3, 6, 11,
    11, 6, 13, 12,
    6, 7, 14, 13,
    7, 8, 15, 14
  }},
  {{util::equal_map(9 * 4, 9)}, {
    // owned
    0, 1, 3, 2,
    1, 6, 8, 3,
    2, 3, 5, 4,
    3, 8, 10, 5,
    // ghost
    6, 7, 9, 8,
    8, 9, 11, 10,
    4, 5, 13, 12,
    5, 10, 14, 13,
    10, 11, 15, 14
  }},
  {{util::equal_map(9 * 4, 9)}, {
    // owned
    11, 12, 1, 0,
    12, 13, 2, 1,
    0, 1, 4, 3,
    1, 2, 5, 4,
    // ghost
    6, 7, 11, 10,
    7, 8, 12, 11,
    8, 9, 13, 12,
    10, 11, 0, 14,
    14, 0, 3, 15
  }},
  {{util::equal_map(9 * 4, 9)}, {
    // owned
    8, 9, 1, 0,
    9, 10, 12, 1,
    0, 1, 3, 2,
    1, 12, 14, 3,
    // ghost
    4, 5, 9, 8,
    5, 6, 10, 9,
    6, 7, 11, 10,
    10, 11, 13, 12,
    12, 13, 15, 14
  }}
};


const std::vector<std::size_t> num_intervals {1, 1, 1, 1};

const util::id local_owned_cells = 4;

const std::vector<std::map<Color, std::set<util::id>>> local_shared_cells {
  {
    {1, {0, 2}},
    {2, {2, 3}},
    {3, {2}}
  },
  {
    {0, {1, 3}},
    {2, {3}},
    {3, {2, 3}}
  },
  {
    {0, {0, 1}},
    {1, {0}},
    {3, {0, 2}}},
  {
    {0, {1}},
    {1, {0, 1}},
    {2, {1, 3}}
  }
};

const std::vector<std::vector<ftui::ghost_entity>> local_ghost_cells {
  {
    {4, 1, 1},
    {5, 3, 1},
    {6, 1, 3},
    {7, 0, 2},
    {8, 1, 2},
  },
  {
    {4, 0, 0},
    {5, 2, 0},
    {6, 0, 3},
    {7, 1, 3},
    {8, 0, 2}},
  {
    {4, 3, 1},
    {5, 2, 0},
    {6, 3, 0},
    {7, 1, 3},
    {8, 3, 3}},
  {
    {4, 2, 1},
    {5, 3, 1},
    {6, 2, 0},
    {7, 0, 2},
    {8, 2, 2}
  }
};

const std::vector<util::id> local_owned_vertices {9, 6, 6, 4};

const std::vector<std::map<Color, std::set<util::id>>> local_shared_vertices {
  {
    {1, {0, 1, 3, 4, 6, 7}},
    {2, {3, 4, 5, 6, 7, 8}},
    {3, {3, 4, 6, 7}}
  },
  {
    {0, {1, 3, 5}},
    {2, {3, 5}},
    {3, {2, 3, 4, 5}}
  },
  {
    {0, {0, 1, 2}},
    {1, {0, 1}},
    {3, {0, 1, 3, 4}}
  },
  {
    {0, {1}},
    {1, {0, 1}},
    {2, {1, 3}}
  }
};

const std::vector<std::vector<ftui::ghost_entity>> local_ghost_vertices {
  {
    { 9, 1, 1},
    {10, 3, 1},
    {11, 5, 1},
    {12, 1, 3},
    {13, 0, 2},
    {14, 1, 2},
    {15, 2, 2}
  },
  {
    { 6, 0, 0},
    { 7, 1, 0},
    { 8, 3, 0},
    { 9, 4, 0},
    {10, 6, 0},
    {11, 7, 0},
    {12, 0, 3},
    {13, 1, 3},
    {14, 0, 2},
    {15, 1, 2}
  },
  {
    { 6, 3, 1},
    { 7, 3, 0},
    { 8, 4, 0},
    { 9, 5, 0},
    {10, 5, 1},
    {11, 6, 0},
    {12, 7, 0},
    {13, 8, 0},
    {14, 1, 3},
    {15, 3, 3}
  },
  {
    { 4, 2, 1},
    { 5, 3, 1},
    { 6, 3, 0},
    { 7, 4, 0},
    { 8, 4, 1},
    { 9, 5, 1},
    {10, 6, 0},
    {11, 7, 0},
    {12, 0, 2},
    {13, 1, 2},
    {14, 3, 2},
    {15, 4, 2}
  }
};

}

// clang-format on

struct fixed_mesh : topo::specialization<topo::unstructured, fixed_mesh> {

  /*--------------------------------------------------------------------------*
    Structure
   *--------------------------------------------------------------------------*/

  enum index_space { cells, vertices };
  using index_spaces = has<cells, vertices>;
  using connectivities =
    list<from<cells, to<vertices>>, from<vertices, to<cells>>>;

  enum entity_list { owned, exclusive, shared, ghost, special_vertices };
  using entity_lists = list<entity<cells, has<owned, exclusive, shared, ghost>>,
    entity<vertices, has<special_vertices>>>;

  template<auto>
  static constexpr PrivilegeCount privilege_count = 3;

  static const inline flecsi::field<std::size_t>::definition<fixed_mesh, cells>
    cid;
  static const inline flecsi::field<std::size_t>::definition<fixed_mesh,
    vertices>
    vid;

  /*--------------------------------------------------------------------------*
    Interface
   *--------------------------------------------------------------------------*/

  template<class B>
  struct interface : B {

    auto cells() const {
      return B::template entities<index_space::cells>();
    }

    template<typename B::subspace_list L>
    auto cells() const {
      return B::template subspace_entities<index_space::cells, L>();
    }

    template<index_space From>
    auto cells(topo::id<From> from) const {
      return B::template entities<index_space::cells>(from);
    }

    auto vertices() const {
      return B::template entities<index_space::vertices>();
    }

    template<typename B::subspace_list L>
    auto vertices() const {
      return B::template subspace_entities<index_space::vertices, L>();
    }

    template<index_space From>
    auto vertices(topo::id<From> from) const {
      return B::template entities<index_space::vertices>(from);
    }

#if 0
    auto edges() const {
      return B::template entities<index_space::edges>();
    }

    template<entity_list List>
    auto edges() const {
      return B::template special_entities<fixed_mesh::edges, List>();
    }
#endif

  }; // struct interface

  /*--------------------------------------------------------------------------*
    Coloring
   *--------------------------------------------------------------------------*/

  static coloring color() {
    flog_assert(processes() == ncolors, "color to process mismatch");

    // clang-format off
    return {
      /* number of global colors */
      ncolors,
      { /* over index spaces */
        {
          { /* cell peers */
            { 1, 2, 3 }, { 0, 2, 3}, { 0, 1, 3}, { 0, 1, 2}
          },
          { /* cell partitions */
            local_owned_cells + local_ghost_cells[0].size(),
            local_owned_cells + local_ghost_cells[1].size(),
            local_owned_cells + local_ghost_cells[2].size(),
            local_owned_cells + local_ghost_cells[3].size()
          },
          num_cells,
          { /* over process colors */
            {

              local_owned_cells + static_cast<util::id>(local_ghost_cells[process()].size()),
              local_shared_cells[process()],
              local_ghost_cells[process()],

              /* cnx_allocs */
              {
                0,
                local_connectivity[process()].values.size()
              }
            }
          },
          num_intervals
        },
        {
          { /* vertex peers */
            { 1, 2, 3 }, { 0, 2, 3}, { 0, 1, 3}, { 0, 1, 2}
          },
          { /* vertex partitions */
            local_owned_vertices[0] + local_ghost_vertices[0].size(),
            local_owned_vertices[1] + local_ghost_vertices[1].size(),
            local_owned_vertices[2] + local_ghost_vertices[2].size(),
            local_owned_vertices[3] + local_ghost_vertices[3].size()
          },
          num_vertices,
          { /* over process colors */
            {

              local_owned_vertices[process()] + static_cast<util::id>(local_ghost_vertices[process()].size()),
              local_shared_vertices[process()],
              local_ghost_vertices[process()],

              /* cnx_allocs */
              {
                /* for allocation only, replaced by using cell connectivity transpose */
                local_connectivity[process()].values.size(),
                0
              }
            }
          },
          num_intervals
        }
      },
      {3, 3, 3, 3} 
    };
    // clang-format on
  } // color

  /*--------------------------------------------------------------------------*
    Initialization
   *--------------------------------------------------------------------------*/

  static void init_cnx(field<util::id, data::ragged>::mutator<wo, wo, na>) {}

  static void initialize(data::topology_slot<fixed_mesh> & s,
    coloring const &) {
    auto & c2v = s->get_connectivity<fixed_mesh::cells, fixed_mesh::vertices>();
    auto & v2c = s->get_connectivity<fixed_mesh::vertices, fixed_mesh::cells>();

    execute<init_cnx>(c2v(s));
    execute<init_cnx>(v2c(s));

    auto lm = data::launch::make(s);
    execute<topo::unstructured_impl::init_connectivity<privilege_count<cells>>,
      mpi>(
      c2v(lm), std::vector<flecsi::util::crs>{local_connectivity[process()]});

    constexpr PrivilegeCount NPC = privilege_count<index_space::cells>;
    constexpr PrivilegeCount NPV = privilege_count<index_space::vertices>;
    execute<topo::unstructured_impl::transpose<NPC, NPV>>(c2v(s), v2c(s));
  } // initialize

}; // struct fixed_mesh

fixed_mesh::slot mesh;
fixed_mesh::cslot coloring;

const field<int>::definition<fixed_mesh, fixed_mesh::cells> pressure;
const field<double>::definition<fixed_mesh, fixed_mesh::vertices> density;

// Exercise the std::vector-like interface:
int
permute(topo::connect_field::mutator<rw, rw, wo> m) {
  UNIT("TASK") {
    return;
    const auto && r = m[0];
    const auto n = r.size();
    const auto p = &r.front();
    ASSERT_GT(n, 1u);
    EXPECT_EQ(p + 1, &r[1]);
    r.push_back(101);
    r.pop_back();
    EXPECT_NE(&r.end()[-2] + 1, &r.back()); // the latter is in the overflow
    EXPECT_GT(r.size(), r.capacity());

    // Intermediate sizes can exceed the capacity of the underlying raw field:
    r.insert(r.begin(), 100, 3);
    EXPECT_EQ(r.end()[-1], 101u);
    EXPECT_EQ(r[0], r[99]);

    r.erase(r.begin(), r.begin() + 100);
    r.pop_back();
    EXPECT_EQ(r.size(), n);
    EXPECT_NE(&r.front(), p);
    // TODO: test shrink_to_fit

    // BUG: remove
    r.clear();
  };
}

void
init_mesh_ids(fixed_mesh::accessor<ro, ro, ro> m,
  field<util::gid>::accessor<wo, wo, wo> cids,
  field<util::gid>::accessor<wo, wo, wo> vids) {
  flog(warn) << __func__ << std::endl;

  for(auto c : m.cells()) {
    cids[c] = cell_l2g[color()][c];
  }
  for(auto v : m.vertices()) {
    vids[v] = vertex_l2g[color()][v];
  }
}

void
init_pressure(fixed_mesh::accessor<ro, ro, ro> m,
  field<int>::accessor<wo, wo, wo> p) {
  flog(warn) << __func__ << std::endl;
  for(auto c : m.cells()) {
    static_assert(std::is_same_v<decltype(c), topo::id<fixed_mesh::cells>>);
    p[c] = -1;
  }
}

void
update_pressure(fixed_mesh::accessor<ro, ro, ro> m,
  field<int>::accessor<rw, rw, rw> p) {
  flog(warn) << __func__ << std::endl;
  int clr = color();
  forall(c, m.cells(), "pressure_c") { p[c] = clr; };
}

void
check_pressure(data::multi<fixed_mesh::accessor<ro, ro, ro>> mm,
  data::multi<field<int>::accessor<ro, ro, ro>> pp) {
  flog(warn) << __func__ << std::endl;
  const auto pc = pp.components();
  auto i = pc.begin();
  for(auto [clr, m] : mm.components()) {
    auto [clr2, p] = *i++;
    flog_assert(clr == clr2, "color mismatch");
    for(auto c : m.cells()) {
      unsigned int v = p[c];
      flog_assert(v == clr, "invalid pressure");
    }
  }
}

void
init_density(fixed_mesh::accessor<ro, ro, ro> m,
  field<double>::accessor<wo, wo, wo> d) {
  flog(warn) << __func__ << std::endl;
  for(auto c : m.vertices()) {
    d[c] = -1;
  }
}

void
update_density(fixed_mesh::accessor<ro, ro, ro> m,
  field<double>::accessor<rw, rw, rw> d) {
  flog(warn) << __func__ << std::endl;
  auto clr = color();
  forall(v, m.vertices(), "density_c") { d[v] = clr; };
}

void
check_density(fixed_mesh::accessor<ro, ro, ro> m,
  field<double>::accessor<ro, ro, ro> d) {
  flog(warn) << __func__ << std::endl;
  auto clr = color();
  for(auto c : m.vertices()) {
    unsigned int v = d[c];
    flog_assert(v == clr, "invalid pressure");
  }
}

int
verify_mesh(fixed_mesh::accessor<ro, ro, ro> m,
  field<util::gid>::accessor<ro, ro, ro> cids,
  field<util::gid>::accessor<ro, ro, ro> vids) {
  UNIT("TASK") {
    auto & out = UNIT_CAPTURE();

    for(auto c : m.cells()) {
      out << "cell(" << cids[c] << "," << c << "):";
      for(auto v : m.vertices(c)) {
        out << " " << vids[v];
      }
      out << "\n";
    }
    out << "\n";

    for(auto v : m.vertices()) {
      out << "vertex(" << vids[v] << "," << v << "):";
      for(auto c : m.cells(v)) {
        out << " " << cids[c];
      }
      out << "\n";
    }
    out << "\n";
    EXPECT_TRUE(
      UNIT_EQUAL_BLESSED("fixed_" + std::to_string(color()) + ".blessed"));
  };
}

static data::launch::Claims
rotate(Color n) {
  data::launch::Claims ret(n);
  for(Color i = 0; i < n; ++i)
    ret[(i + n - (FLECSI_BACKEND != FLECSI_BACKEND_mpi)) % n].push_back(i);
  return ret;
}

int
fixed_driver() {
  UNIT() {
    coloring.allocate();
    mesh.allocate(coloring.get());

    execute<init_mesh_ids>(mesh, fixed_mesh::cid(mesh), fixed_mesh::vid(mesh));

    EXPECT_EQ(
      test<permute>(
        mesh->get_connectivity<fixed_mesh::vertices, fixed_mesh::cells>()(
          mesh)),
      0);

    EXPECT_EQ(
      test<verify_mesh>(mesh, fixed_mesh::cid(mesh), fixed_mesh::vid(mesh)), 0);

    execute<init_pressure>(mesh, pressure(mesh));
    execute<update_pressure, default_accelerator>(mesh, pressure(mesh));
    {
      auto lm = data::launch::make(mesh, rotate(mesh.colors()));
      execute<check_pressure>(lm, pressure(lm));
    }

    execute<init_density>(mesh, density(mesh));
    execute<update_density, default_accelerator>(mesh, density(mesh));
    execute<check_density>(mesh, density(mesh));
  };
} // unstructured_driver

util::unit::driver<fixed_driver> driver;
