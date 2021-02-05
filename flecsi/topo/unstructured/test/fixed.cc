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

#define __FLECSI_PRIVATE__
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
  static constexpr std::size_t privilege_count = 2;

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
    return {fixed::colors,
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

#if 0
  static void init_csub(field<util::id, data::ragged>::mutator<rw> own,
    field<util::id, data::ragged>::mutator<rw> exc,
    field<util::id, data::ragged>::mutator<rw> shr,
    field<util::id, data::ragged>::mutator<rw> ghs,
    topo::unstructured_base::index_coloring const & cell_coloring) {
    (void)own;
    exc[flecsi::color()].resize(cell_coloring.exclusive.size());
    shr[flecsi::color()].resize(cell_coloring.shared.size());
    ghs[flecsi::color()].resize(cell_coloring.ghosts.size());
  }
#endif

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

  static void init_v2c(field<util::id, data::ragged>::mutator<rw, na> v2c,
    field<util::id, data::ragged>::accessor<ro, na> c2v) {
    for(std::size_t c{0}; c < c2v.size(); ++c) {
      for(std::size_t v{0}; v < c2v[c].size(); ++v) {
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

void
init_pressure(fixed_mesh::accessor<ro, ro> m,
  field<double>::accessor<wo, na> p) {
  flog(warn) << __func__ << std::endl;
  for(auto c : m.cells()) {
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
  field<std::size_t>::accessor<ro, wo> cids,
  field<std::size_t>::accessor<ro, wo> vids) {
  for(auto c : m.cells()) {
    std::stringstream ss;
    ss << "cell(" << cids[c] << "): ";
    for(auto v : m.vertices(c)) {
      ss << vids[v] << " ";
    }
    flog(info) << ss.str() << std::endl;
  }

  for(auto v : m.vertices()) {
    std::stringstream ss;
    ss << "vertex(" << vids[v] << "): ";
    for(auto c : m.cells(v)) {
      ss << cids[c] << " ";
    }
    flog(info) << ss.str() << std::endl;
  }
}

int
fixed_driver() {
  UNIT {
    coloring.allocate();
    mesh.allocate(coloring.get());
    //    execute<init_ids>(mesh, cids(mesh), vids(mesh));
    //    execute<print>(mesh, cids(mesh), vids(mesh));
    execute<init_pressure>(mesh, pressure(mesh));
    execute<update_pressure>(mesh, pressure(mesh));
    execute<print_pressure>(mesh, pressure(mesh));
    execute<init_density>(mesh, density(mesh));
    execute<update_density>(mesh, density(mesh));
    execute<print_density>(mesh, density(mesh));
  };
} // unstructured_driver

flecsi::unit::driver<fixed_driver> driver;
