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

  enum index_space { vertices, cells };
  using index_spaces = has<cells, vertices>;
  using connectivities = list<from<cells, to<vertices>>>;

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

    template<index_space From>
    auto cells(topo::id<From> from) {
      return B::template entities<index_space::cells>(from);
    }

    auto vertices() {
      return B::template entities<index_space::vertices>();
    }

  }; // struct interface

  /*--------------------------------------------------------------------------*
    Coloring
   *--------------------------------------------------------------------------*/

  static coloring color(std::string const & filename) {

    topo::unstructured_impl::simple_definition sd(filename.c_str());

    const Color colors = processes();
    // const Color colors = 4;

    auto [naive, c2v, v2c, c2c] = topo::unstructured_impl::make_dcrs(sd, 1);
    auto raw = util::parmetis::color(naive, colors);

    auto [primaries, p2m, m2p] =
      topo::unstructured_impl::migrate(naive, colors, raw, c2v, v2c, c2c);

    // clang-format off
    topo::unstructured_impl::coloring_definition cd{
      colors /* global colors */,
      cells /* primary index */,
      2 /* primary dimension */,
      1 /* halo depth */,
      vertices /* vertex index */,
      {/* auxiliary entities */}
    };
    // clang-format on

    return topo::unstructured_impl::color(
      sd, cd, raw, primaries, c2v, v2c, c2c, m2p, p2m);
  } // color

  /*--------------------------------------------------------------------------*
    Initialization
   *--------------------------------------------------------------------------*/

  template<std::size_t VI>
  static void init_c2v(field<util::id, data::ragged>::mutator<rw, rw, na> c2v,
    std::vector<topo::unstructured_impl::process_color> const & pc,
    std::vector<std::map<std::size_t, std::size_t>> const & maps) {
    (void)c2v;
    (void)pc;
    (void)maps;
  }

  static void initialize(data::topology_slot<unstructured> & s,
    coloring const & c) {
    flog(warn) << log::container{c.partitions} << std::endl;

    // auto & c2v =
    // s->connect_.get<unstructured::cells>().get<unstructured::vertices>();
    auto & c2v =
      s->get_connectivity<unstructured::cells, unstructured::vertices>();
    auto maps = std::move(s)->reverse_maps<unstructured::cells>();
    execute<init_c2v<unstructured::vertices>, mpi>(
      c2v(s), c.idx_spaces[unstructured::cells], maps);
  } // initialize

}; // struct unstructured

unstructured::slot mesh;
unstructured::cslot coloring;

int
unstructured_driver() {
  UNIT {
    coloring.allocate("simple2d-8x8.msh");
    mesh.allocate(coloring.get());

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
