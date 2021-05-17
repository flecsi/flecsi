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

#include "fixed.hh"

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/unit.hh"

#include <algorithm>

using namespace flecsi;

namespace global {
std::vector<std::size_t> cells;
std::vector<std::size_t> vertices;
} // namespace global

struct fixed_mesh : topo::specialization<topo::unstructured, fixed_mesh> {

  /*--------------------------------------------------------------------------*
    Structure
   *--------------------------------------------------------------------------*/

  enum index_space { vertices, cells };
  using index_spaces = has<cells, vertices>;
  using connectivities =
    list<from<cells, to<vertices>>, from<vertices, to<cells>>>;

  enum entity_list { owned, exclusive, shared, ghost, special_vertices };
  using entity_lists = list<entity<cells, has<owned, exclusive, shared, ghost>>,
    entity<vertices, has<special_vertices>>>;

  template<auto>
  static constexpr PrivilegeCount privilege_count = 2;

  /*--------------------------------------------------------------------------*
    Interface
   *--------------------------------------------------------------------------*/

  template<class B>
  struct interface : B {

    auto cells() {
      return B::template entities<index_space::cells>();
    }

    template<typename B::subspace_list L>
    auto cells() {
      return B::template subspace_entities<index_space::cells, L>();
    }

    template<index_space From>
    auto cells(topo::id<From> from) {
      return B::template entities<index_space::cells>(from);
    }

    auto vertices() {
      return B::template entities<index_space::vertices>();
    }

    template<typename B::subspace_list L>
    auto vertices() {
      return B::template subspace_entities<index_space::vertices, L>();
    }

    template<index_space From>
    auto vertices(topo::id<From> from) {
      return B::template entities<index_space::vertices>(from);
    }

#if 0
    auto edges() {
      return B::template entities<index_space::edges>();
    }

    template<entity_list List>
    auto edges() {
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
    return
      {
        {
          MPI_COMM_WORLD,
          process(),
          fixed::colors,
          {
            fixed::num_cells, fixed::num_vertices
          },
          fixed::idx_colorings[process()],
          fixed::cnx_allocs[process()],
          fixed::cnx_colorings[process()]
        }
      };
    // clang-format on
  } // color

  /*--------------------------------------------------------------------------*
    Initialization
   *--------------------------------------------------------------------------*/

  static void init_c2v(field<util::id, data::ragged>::mutator<rw, na> c2v,
    topo::unstructured_impl::crs const & cnx,
    std::map<std::size_t, std::size_t> & map) {
    std::size_t off{0};

    for(std::size_t c{0}; c < cnx.offsets.size() - 1; ++c) {
      const std::size_t start = cnx.offsets[off];
      const std::size_t size = cnx.offsets[off + 1] - start;

      c2v[c].resize(size);

      for(std::size_t i{0}; i < size; ++i) {
        c2v[c][i] = map[cnx.indices[start + i]];
      }
      ++off;
    }
  }

  static void init_v2c(topo::connect_field::mutator<rw, na> v2c,
    topo::connect_field::accessor<ro, na> c2v) {
    for(std::size_t c{0}; c < c2v.size(); ++c) {
      for(std::size_t v{0}; v < c2v[c].size(); ++v) {
        flog(trace) << "v: " << c2v[c][v] << " c: " << c << std::endl;
        v2c[c2v[c][v]].push_back(c);
      }
    }
  }

  static void initialize(data::topology_slot<fixed_mesh> & s,
    coloring const & c) {

    /*
      Define the cell ordering from coloring. This version uses the
      mesh ordering, i.e., the cells are sorted by ascending mesh id.
     */

    auto cell_coloring = c.at(0).idx_colorings[0];
    global::cells.clear();
    for(auto e : cell_coloring.owned) {
      global::cells.push_back(e);
    }

    for(auto e : cell_coloring.ghost) {
      global::cells.push_back(e.id);
    }

    util::force_unique(global::cells);

    /*
      Define the vertex ordering from coloring. This version uses the
      mesh ordering, i.e., the vertices are sorted by ascending mesh id.
     */

    auto vertex_coloring = c.at(0).idx_colorings[1];
    global::vertices.clear();
    for(auto e : vertex_coloring.owned) {
      global::vertices.push_back(e);
    }

    for(auto e : vertex_coloring.ghost) {
      global::vertices.push_back(e.id);
    }

    util::force_unique(global::vertices);

    std::map<std::size_t, std::size_t> vertex_map;
    std::size_t off{0};
    for(auto e : global::vertices) {
      vertex_map[e] = off++;
    }

    auto & c2v =
      s->connect_.get<fixed_mesh::cells>().get<fixed_mesh::vertices>();
    execute<init_c2v, mpi>(c2v(s), c.at(0).cnx_colorings[0][0], vertex_map);

    auto & v2c =
      s->connect_.get<fixed_mesh::vertices>().get<fixed_mesh::cells>();
#if 0
    execute<init_v2c, mpi>(v2c(s), c2v(s));
#endif
    execute<topo::unstructured_impl::transpose, mpi>(c2v(s), v2c(s));
  } // initialize

}; // struct fixed_mesh

fixed_mesh::slot mesh;
fixed_mesh::cslot coloring;

const field<double>::definition<fixed_mesh, fixed_mesh::cells> pressure;
const field<double>::definition<fixed_mesh, fixed_mesh::vertices> density;
const field<std::size_t>::definition<fixed_mesh, fixed_mesh::cells> cids;
const field<std::size_t>::definition<fixed_mesh, fixed_mesh::vertices> vids;

void
init_ids(fixed_mesh::accessor<ro, ro> m,
  field<std::size_t>::accessor<wo, wo> cids,
  field<std::size_t>::accessor<wo, wo> vids) {
  for(auto c : m.cells()) {
    cids[c] = global::cells[c];
  }
  for(auto v : m.vertices()) {
    vids[v] = global::vertices[v];
  }
}

// Exercise the std::vector-like interface:
int
permute(topo::connect_field::mutator<rw, na> m) {
  UNIT {
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
init_pressure(fixed_mesh::accessor<ro, ro> m,
  field<double>::accessor<wo, na> p) {
  flog(warn) << __func__ << std::endl;
  for(auto c : m.cells()) {
    static_assert(std::is_same_v<decltype(c), topo::id<fixed_mesh::cells>>);
    p[c] = -1.0;
  }
}

void
update_pressure(fixed_mesh::accessor<ro, ro> m,
  field<double>::accessor<rw, ro> p) {
  flog(warn) << __func__ << std::endl;
  for(auto c : m.cells()) {
    p[c] = color();
  }
}

#if defined(FLECSI_ENABLE_KOKKOS)
void
parallel_update_pressure(fixed_mesh::accessor<ro, ro> m,
  field<double>::accessor<rw, ro> p) {
  forall(c, m.cells(), "pressure_c") {
    p[c] += 0.0;
  };
}
#endif

void
print_pressure(fixed_mesh::accessor<ro, ro> m,
  field<double>::accessor<ro, ro> p) {
  flog(warn) << __func__ << std::endl;
  std::stringstream ss;
  for(auto c : m.cells()) {
    ss << p[c] << " ";
  }
  flog(info) << ss.str() << std::endl;
}

void
init_density(fixed_mesh::accessor<ro, ro> m,
  field<double>::accessor<wo, na> d) {
  flog(warn) << __func__ << std::endl;
  for(auto c : m.vertices()) {
    d[c] = -1.0;
  }
}

void
update_density(fixed_mesh::accessor<ro, ro> m,
  field<double>::accessor<rw, ro> d) {
  flog(warn) << __func__ << std::endl;
  for(auto c : m.vertices()) {
    d[c] = color();
  }
}

void
print_density(fixed_mesh::accessor<ro, ro> m,
  field<double>::accessor<ro, ro> d) {
  flog(warn) << __func__ << std::endl;
  std::stringstream ss;
  for(auto c : m.vertices()) {
    ss << d[c] << " ";
  }
  flog(info) << ss.str() << std::endl;
}

void
print(fixed_mesh::accessor<ro, ro> m,
  field<std::size_t>::accessor<ro, ro> cids,
  field<std::size_t>::accessor<ro, ro> vids) {
  (void)cids;
#if 1
  for(auto c : m.cells()) {
    std::stringstream ss;
    ss << "cell(" << cids[c] << "," << c << "): ";
    for(auto v : m.vertices(c)) {
      ss << vids[v] << " ";
    }
    flog(info) << ss.str() << std::endl;
  }
#endif

  for(auto v : m.vertices()) {
    std::stringstream ss;
    ss << "vertex(" << vids[v] << "," << v << "): ";
    ss << m.cells(v).size();
#if 0
    for(auto c : m.cells(v)) {
      ss << cids[c] << " ";
    }
#endif
    flog(info) << ss.str() << std::endl;
  }
}

int
fixed_driver() {
  UNIT {
    coloring.allocate();
    mesh.allocate(coloring.get());
    execute<init_ids>(mesh, cids(mesh), vids(mesh));
    EXPECT_EQ(
      test<permute>(
        mesh->connect_.get<fixed_mesh::vertices>().get<fixed_mesh::cells>()(
          mesh)),
      0);
    execute<print>(mesh, cids(mesh), vids(mesh));
#if 0
    execute<init_pressure>(mesh, pressure(mesh));
    execute<update_pressure>(mesh, pressure(mesh));
#if defined(FLECSI_ENABLE_KOKKOS)
    execute<parallel_update_pressure, default_accelerator>(
      mesh, pressure(mesh));
#endif
    execute<print_pressure>(mesh, pressure(mesh));
    execute<init_density>(mesh, density(mesh));
    execute<update_density>(mesh, density(mesh));
    execute<print_density>(mesh, density(mesh));
#endif
  };
} // unstructured_driver

flecsi::unit::driver<fixed_driver> driver;
