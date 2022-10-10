#include "fixed.hh"

#include "flecsi/data.hh"
#include "flecsi/exec/kernel.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/unit.hh"

#include <algorithm>

using namespace flecsi;

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
    flog_assert(processes() == fixed::colors, "color to process mismatch");

    // clang-format off
    return {
      MPI_COMM_WORLD,
      fixed::colors,
      /* process_colors */
      { /* over global processes */
        std::vector<Color>{ 0 },
        std::vector<Color>{ 1 },
        std::vector<Color>{ 2 },
        std::vector<Color>{ 3 }
      },
      /* num_peers */
      { /* over global colors */
        3, 3, 3, 3
      },
      /* peers */
      { /* over index spaces */
        { /* cell peers */
          { 1, 2, 3 }, { 0, 2, 3}, { 0, 1, 3}, { 0, 1, 2}
        },
        { /* vertex peers */
          { 1, 2, 3 }, { 0, 2, 3}, { 0, 1, 3}, { 0, 1, 2}
        }
      },
      {
        {
          fixed::cells[0].all.size(),
          fixed::cells[1].all.size(),
          fixed::cells[2].all.size(),
          fixed::cells[3].all.size()
        },
        {
          fixed::vertices[0].all.size(),
          fixed::vertices[1].all.size(),
          fixed::vertices[2].all.size(),
          fixed::vertices[3].all.size()
        }
      },
      { /* over index spaces */
        { /* over process colors */
          base::process_coloring{

            /* color */
            process(),

            /* entities */
            fixed::num_cells,

            /* coloring */
            fixed::cells[process()],

            /* peers */
            fixed::peers[process()],

            /* cnx_allocs */
            {
              0,
              fixed::connectivity[process()][0].indices.size()
            },

            /* cnx_colorings */
            {
              {},
              fixed::connectivity[process()][0]
            }
          }
        },
        {
          base::process_coloring{

            /* color */
            process(),

            /* entities */
            fixed::num_vertices,

            /* coloring */
            fixed::vertices[process()],

            /* peers */
            fixed::peers[process()],

            /* cnx_allocs */
            {
              fixed::connectivity[process()][0].indices.size(),
              0
            },

            /* cnx_colorings */
            {
              {} /* use cell connectivity transpose */
            }
          }
        }
      }
    };
    // clang-format on
  } // color

  /*--------------------------------------------------------------------------*
    Initialization
   *--------------------------------------------------------------------------*/

  static void init_cnx(field<util::id, data::ragged>::mutator<wo, wo, na>) {}

  static void initialize(data::topology_slot<fixed_mesh> & s,
    coloring const & c) {
    auto & c2v = s->get_connectivity<fixed_mesh::cells, fixed_mesh::vertices>();
    auto & v2c = s->get_connectivity<fixed_mesh::vertices, fixed_mesh::cells>();
    auto const & vmaps = s->reverse_map<fixed_mesh::vertices>();

    execute<init_cnx>(c2v(s));
    execute<init_cnx>(v2c(s));

    auto lm = data::launch::make(s);
    execute<
      topo::unstructured_impl::init_connectivity<core::index<fixed_mesh::cells>,
        core::index<fixed_mesh::vertices>,
        privilege_count<cells>>,
      mpi>(c2v(lm), c, vmaps);

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
  std::stringstream ss;
  for(auto c : m.vertices()) {
    ss << d[c] << " ";
  }
  flog(info) << ss.str() << std::endl;
}

void
print(fixed_mesh::accessor<ro, ro, ro> m,
  field<util::gid>::accessor<ro, ro, ro> cids,
  field<util::gid>::accessor<ro, ro, ro> vids) {

  std::stringstream ss;
  for(auto c : m.cells()) {
    ss << "cell(" << cids[c] << "," << c << "): ";
    for(auto v : m.vertices(c)) {
      ss << vids[v] << " ";
    }
    ss << std::endl;
  }
  flog(info) << ss.str() << std::endl;

  ss.str("");
  for(auto v : m.vertices()) {
    ss << "vertex(" << vids[v] << "," << v << "): ";
    for(auto c : m.cells(v)) {
      ss << cids[c] << " ";
    }
    ss << std::endl;
  }
  flog(info) << ss.str() << std::endl;
}

static bool
rotate(topo::claims::Field::accessor<wo> a, Color, Color n) {
  a = topo::claims::row((color() + (FLECSI_BACKEND != FLECSI_BACKEND_mpi)) % n);
  return false;
}

int
fixed_driver() {
  UNIT() {
    coloring.allocate();
    mesh.allocate(coloring.get());

    EXPECT_EQ(
      test<permute>(
        mesh->get_connectivity<fixed_mesh::vertices, fixed_mesh::cells>()(
          mesh)),
      0);

    auto const & cids = mesh->forward_map<fixed_mesh::cells>();
    auto const & vids = mesh->forward_map<fixed_mesh::vertices>();
    execute<print>(mesh, cids(mesh), vids(mesh));

    execute<init_pressure>(mesh, pressure(mesh));
    execute<update_pressure, default_accelerator>(mesh, pressure(mesh));
    {
      auto lm = data::launch::make<rotate>(mesh, mesh->colors());
      execute<check_pressure>(lm, pressure(lm));
    }

#if 1
    execute<init_density>(mesh, density(mesh));
    execute<update_density, default_accelerator>(mesh, density(mesh));
    execute<check_density>(mesh, density(mesh));
#endif
  };
} // unstructured_driver

flecsi::unit::driver<fixed_driver> driver;
