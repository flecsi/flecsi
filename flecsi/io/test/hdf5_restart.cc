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

#define __FLECSI_PRIVATE__

#include <vector>

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/io.hh"
#include "flecsi/topo/narray/test/narray.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

mesh::slot m;
mesh::cslot mc;

const field<double>::definition<mesh, mesh::entities> m_field_1, m_field_2;

void
init(mesh::accessor<ro> m,
  field<double>::accessor<wo, na> mf1,
  field<double>::accessor<wo, na> mf2) {
  auto ms1 = m.mdspan<mesh::entities>(mf1);
  auto ms2 = m.mdspan<mesh::entities>(mf2);
  for(auto i : m.extents<mesh::x_axis, mesh::all>()) {
    for(auto j : m.extents<mesh::y_axis, mesh::all>()) {
      double val = 16. * color() + 8. * (int)i + (int)j;
      ms1(j, i) = val;
      ms2(j, i) = val + 1000.;
    } // for
  } // for
} // init

void
clear(mesh::accessor<ro> m,
  field<double>::accessor<wo, na> mf1,
  field<double>::accessor<wo, na> mf2) {
  auto ms1 = m.mdspan<mesh::entities>(mf1);
  auto ms2 = m.mdspan<mesh::entities>(mf2);
  for(auto i : m.extents<mesh::x_axis, mesh::all>()) {
    for(auto j : m.extents<mesh::y_axis, mesh::all>()) {
      ms1(j, i) = 0.;
      ms2(j, i) = 0.;
    } // for
  } // for
} // clear

int
check(mesh::accessor<ro> m,
  field<double>::accessor<ro, na> mf1,
  field<double>::accessor<ro, na> mf2) {
  UNIT {
    auto ms1 = m.mdspan<mesh::entities>(mf1);
    auto ms2 = m.mdspan<mesh::entities>(mf2);
    for(auto i : m.extents<mesh::x_axis, mesh::all>()) {
      for(auto j : m.extents<mesh::y_axis, mesh::all>()) {
        double val = 16. * color() + 8. * (int)i + (int)j;
        auto s1exp = val;
        auto s2exp = val + 1000.;
        ASSERT_EQ(ms1(j, i), s1exp);
        ASSERT_EQ(ms2(j, i), s2exp);
      } // for
    } // for
  };
} // check

int
restart_driver() {
  UNIT {
    mesh::coord indices{8, 8};
    mesh::coord colors{4, 1};
    mesh::coord hdepths{0, 0};
    mesh::coord bdepths{0, 0};
    std::vector<bool> periodic{false, false};
    std::vector<mesh::coloring_definition> index_definitions = {
      {colors, indices, hdepths, bdepths, periodic, false}};
    mc.allocate(index_definitions);
    m.allocate(mc.get());

    auto mf1 = m_field_1(m);
    auto mf2 = m_field_2(m);
    execute<init>(m, mf1, mf2);

    int num_files = 4;
    io::io_interface iif{num_files};
    // TODO:  make this registration automatic, not manual
    iif.add_region<mesh, mesh::entities>(m);
    iif.checkpoint_all_fields("hdf5_restart.dat");

    execute<clear>(m, mf1, mf2);
    iif.recover_all_fields("hdf5_restart.dat");

    EXPECT_EQ(test<check>(m, mf1, mf2), 0);
  };

  return 0;
}

flecsi::unit::driver<restart_driver> driver;
