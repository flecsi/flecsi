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

  enum index_space { cells, vertices };
  using index_spaces = has<cells, vertices>;
  using connectivities =
    list<from<cells, to<vertices>>, from<vertices, to<cells>>>;

  enum entity_list { owned, exclusive, shared, ghosts, special_vertices };
  using entity_lists =
    list<entity<cells, has<owned, exclusive, shared, ghosts>>,
      entity<vertices, has<special_vertices>>>;

  template<auto>
  static constexpr PrivilegeCount privilege_count = 2;

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
    return {MPI_COMM_WORLD,
      fixed::colors,
      {fixed::num_cells, fixed::num_vertices},
      fixed::idx_colorings[process()],
      fixed::cnx_allocs[process()],
      fixed::cnx_colorings[process()]};
  } // color

  /*--------------------------------------------------------------------------*
    Initialization
   *--------------------------------------------------------------------------*/

  static void set_dests(field<data::intervals::Value>::accessor<wo> a,
    std::map<std::size_t, topo::unstructured_impl::crs> const & ghost_itvls) {
    std::size_t i{0}, ioff{0};
    for(auto const & gi : ghost_itvls) {
      for(std::size_t o{0}; o < gi.second.offsets.size() - 1; ++o) {
        const std::size_t start = ioff + gi.second.offsets[o] + 1;
        const std::size_t end = ioff + gi.second.offsets[o + 1] + 1;
        a[i++] = data::intervals::make({start, end}, process());
      } // for
      ioff = gi.second.offsets.back();
    } // for
  }

  static void set_ptrs(field<data::points::Value>::accessor<wo> a,
    std::map<std::size_t, topo::unstructured_impl::crs> const & ghost_itvls) {
    std::size_t i{0};
    for(auto const & gi : ghost_itvls) {
      for(auto idx : gi.second.indices) {
        a[i + 1] = data::points::make(idx, gi.first);
        ++i;
      } // for
    } // for
  }

  static void init_c2v(topo::connect_field::mutator<wo, na> c2v,
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

  static void init_v2c(topo::connect_field::mutator<wo, na> v2c,
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

    auto cell_coloring = c.idx_colorings[0];
    global::cells.clear();
    for(auto e : cell_coloring.owned) {
      global::cells.push_back(e);
    }

    for(auto e : cell_coloring.ghosts) {
      global::cells.push_back(e.id);
    }

    /*
      Define the vertex ordering from coloring. This version uses the
      mesh ordering, i.e., the vertices are sorted by ascending mesh id.
     */

    auto vertex_coloring = c.idx_colorings[1];
    global::vertices.clear();
    for(auto e : vertex_coloring.owned) {
      global::vertices.push_back(e);
    }

    for(auto e : vertex_coloring.ghosts) {
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
    execute<init_c2v, mpi>(c2v(s), c.cnx_colorings[0][0], vertex_map);

    auto & v2c =
      s->connect_.get<fixed_mesh::vertices>().get<fixed_mesh::cells>();
    execute<init_v2c, mpi>(v2c(s), c2v(s));
    // execute<topo::unstructured_impl::transpose, mpi>(v2c(s), c2v(s));
  } // initialize

}; // struct fixed_mesh

fixed_mesh::slot mesh;
fixed_mesh::cslot coloring;

const field<int>::definition<fixed_mesh, fixed_mesh::cells> pressure;
const field<int>::definition<fixed_mesh, fixed_mesh::vertices> density;
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
init_pressure(fixed_mesh::accessor<ro, ro> m, field<int>::accessor<wo, wo> p) {
  flog(warn) << __func__ << std::endl;
  for(auto c : m.cells()) {
    static_assert(std::is_same_v<decltype(c), topo::id<fixed_mesh::cells>>);
    p[c] = -1;
  }
}

void
update_pressure(fixed_mesh::accessor<ro, ro> m,
  field<int>::accessor<rw, rw> p) {
  flog(warn) << __func__ << std::endl;
  int clr = color();
  forall(c, m.cells(), "pressure_c") { p[c] = clr; };
}

void
check_pressure(fixed_mesh::accessor<ro, ro> m, field<int>::accessor<ro, ro> p) {
  flog(warn) << __func__ << std::endl;
  unsigned int clr = color();
  for(auto c : m.cells()) {
    unsigned int v = p[c];
    flog_assert(v == clr, "invalid pressure");
  }
}

void
init_density(fixed_mesh::accessor<ro, ro> m,
  field<double>::accessor<wo, wo> d) {
  flog(warn) << __func__ << std::endl;
  for(auto c : m.vertices()) {
    d[c] = -1;
  }
}

void
update_density(fixed_mesh::accessor<ro, ro> m,
  field<double>::accessor<rw, rw> d) {
  flog(warn) << __func__ << std::endl;
  auto clr = color();
  forall(v, m.vertices(), "density_c") { d[v] = clr; };
}

void
check_density(fixed_mesh::accessor<ro, ro> m,
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

    execute<init_pressure>(mesh, pressure(mesh));
    execute<update_pressure, default_accelerator>(mesh, pressure(mesh));
    execute<check_pressure>(mesh, pressure(mesh));

#if 0
    execute<init_density>(mesh, density(mesh));
    execute<update_density, default_accelerator>(mesh, density(mesh));
    execute<check_density>(mesh, density(mesh));
#endif
  };
} // unstructured_driver

flecsi::unit::driver<fixed_driver> driver;
