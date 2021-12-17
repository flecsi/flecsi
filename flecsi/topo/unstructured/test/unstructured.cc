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

#include "simple_definition.hh"

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/unstructured/coloring_utils.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/util/parmetis.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

struct unstructured : topo::specialization<topo::unstructured, unstructured> {

  /*--------------------------------------------------------------------------*
    Structure
   *--------------------------------------------------------------------------*/

  enum class index_space { vertices = 160, cells = 2718 };
  static constexpr auto cells = index_space::cells;
  static constexpr auto vertices = index_space::vertices;
  using index_spaces = has<cells, vertices>;
  using connectivities =
    list<from<cells, to<vertices>>, from<vertices, to<cells>>>;

  enum entity_list { special };
  using entity_lists = list<entity<vertices, has<special>>>;

  template<auto>
  static constexpr PrivilegeCount privilege_count = 3;

  /*--------------------------------------------------------------------------*
    Interface
   *--------------------------------------------------------------------------*/

  template<class B>
  struct interface : B {

    auto cells() {
      return B::template entities<index_space::cells>();
    }

    template<index_space F>
    auto cells(topo::id<F> from) {
      return B::template entities<index_space::cells>(from);
    }

    auto vertices() {
      return B::template entities<index_space::vertices>();
    }

    template<index_space F>
    auto vertices(topo::id<F> from) const {
      return B::template entities<index_space::vertices>(from);
    }

  }; // struct interface

  /*--------------------------------------------------------------------------*
    Coloring
   *--------------------------------------------------------------------------*/

  static coloring color(std::string const & filename) {

    topo::unstructured_impl::simple_definition sd(filename.c_str());

    const Color colors =
      processes() * (1 + (FLECSI_BACKEND != FLECSI_BACKEND_mpi));
    // const Color colors = 4;

    auto [naive, c2v, v2c, c2c] = topo::unstructured_impl::make_dcrs(sd, 1);
    auto raw = util::parmetis::color(naive, colors);

    auto [primaries, p2m, m2p] =
      topo::unstructured_impl::migrate(naive, colors, raw, c2v, v2c, c2c);

    // clang-format off
    topo::unstructured_impl::coloring_definition cd{
      colors /* global colors */,
      core::index<index_space::cells> /* primary index */,
      2 /* primary dimension */,
      1 /* halo depth */,
      core::index<index_space::vertices> /* vertex index */,
      {/* auxiliary entities */}
    };
    // clang-format on

    return topo::unstructured_impl::color(
      sd, cd, raw, primaries, c2v, v2c, c2c, m2p, p2m);
  } // color

  /*--------------------------------------------------------------------------*
    Initialization
   *--------------------------------------------------------------------------*/

  static void init_cnx(field<util::id, data::ragged>::mutator<wo, wo, na>) {}

  static void init_c2v(field<util::id, data::ragged>::mutator<wo, wo, na> c2v,
    std::vector<base::process_coloring> const & prc_clrngs,
    std::vector<std::map<std::size_t, std::size_t>> const & vmaps) {

    auto pcs = prc_clrngs.begin();
    auto vms = vmaps.begin();
    auto const & pc = *pcs++;
    auto const & vm = *vms++;
    std::size_t off{0};

    auto const & cnx = pc.cnx_colorings[core::index<vertices>];
    for(std::size_t c{0}; c < cnx.offsets.size() - 1; ++c) {
      const std::size_t start = cnx.offsets[off];
      const std::size_t size = cnx.offsets[off + 1] - start;
      c2v[c].resize(size);

      for(std::size_t i{0}; i < size; ++i) {
        c2v[c][i] = vm.at(cnx.indices[start + i]);
      } // for

      ++off;
    } // for
  } // init_c2v

  static void initialize(data::topology_slot<unstructured> & s,
    coloring const & c) {
    auto & c2v =
      s->get_connectivity<index_space::cells, index_space::vertices>();
    auto & v2c =
      s->get_connectivity<index_space::vertices, index_space::cells>();
    auto const & vmaps = s->reverse_map<index_space::vertices>();

    execute<init_cnx>(c2v(s));
    execute<init_cnx>(v2c(s));

    execute<init_c2v, mpi>(c2v(s), c.idx_spaces[core::index<cells>], vmaps);
    execute<topo::unstructured_impl::transpose>(c2v(s), v2c(s));
  } // initialize
}; // struct unstructured

void
print(unstructured::accessor<ro, ro, ro> m,
  field<util::id>::accessor<ro, ro, ro> cids,
  field<util::id>::accessor<ro, ro, ro> vids) {

  std::stringstream ss;
  ss << "color(" << color() << ")" << std::endl;
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

unstructured::slot mesh;
unstructured::cslot coloring;

int
unstructured_driver() {
  UNIT {
    coloring.allocate("simple2d-16x16.msh");
    mesh.allocate(coloring.get());

    auto const & cids = mesh->forward_map<unstructured::cells>();
    auto const & vids = mesh->forward_map<unstructured::vertices>();
    execute<print>(mesh, cids(mesh), vids(mesh));
#if 0
    auto & neuf =
      mesh->special_.get<unstructured::edges>().get<unstructured::neumann>();
    execute<allocate>(neuf.sizes());
    neuf.resize();
    execute<init>(mesh->special_field(neuf));
    EXPECT_EQ(test<check>(mesh), 0);
#endif
  };
} // unstructured_driver

flecsi::unit::driver<unstructured_driver> driver;
