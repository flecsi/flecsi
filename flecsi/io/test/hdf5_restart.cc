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

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/io.hh"
#include "flecsi/topo/narray/test/narray.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/unit.hh"

#include <vector>

using namespace flecsi;

using is = mesh2d::index_space;
using rg = mesh2d::range;
using ax = mesh2d::axis;

mesh2d::slot m;
mesh2d::cslot mc;

const field<double>::definition<mesh2d, mesh2d::index_space::entities>
  m_field_1, m_field_2;

void
init(mesh2d::accessor<ro> m,
  field<double>::accessor<wo, na> mf1,
  field<double>::accessor<wo, na> mf2) {
  auto ms1 = m.mdspan<is::entities>(mf1);
  auto ms2 = m.mdspan<is::entities>(mf2);
  for(auto i : m.extents<ax::x_axis, rg::all>()) {
    for(auto j : m.extents<ax::y_axis, rg::all>()) {
      double val = 16. * color() + 8. * (int)i + (int)j;
      ms1[j][i] = val;
      ms2[j][i] = val + 1000.;
    } // for
  } // for
} // init

void
clear(mesh2d::accessor<ro> m,
  field<double>::accessor<wo, na> mf1,
  field<double>::accessor<wo, na> mf2) {
  auto ms1 = m.mdspan<is::entities>(mf1);
  auto ms2 = m.mdspan<is::entities>(mf2);
  for(auto i : m.extents<ax::x_axis, rg::all>()) {
    for(auto j : m.extents<ax::y_axis, rg::all>()) {
      ms1[j][i] = 0.;
      ms2[j][i] = 0.;
    } // for
  } // for
} // clear

int
check(mesh2d::accessor<ro> m,
  field<double>::accessor<ro, na> mf1,
  field<double>::accessor<ro, na> mf2) {
  UNIT {
    auto ms1 = m.mdspan<is::entities>(mf1);
    auto ms2 = m.mdspan<is::entities>(mf2);
    for(auto i : m.extents<ax::x_axis, rg::all>()) {
      for(auto j : m.extents<ax::y_axis, rg::all>()) {
        double val = 16. * color() + 8. * (int)i + (int)j;
        auto s1exp = val;
        auto s2exp = val + 1000.;
        ASSERT_EQ(ms1[j][i], s1exp);
        ASSERT_EQ(ms2[j][i], s2exp);
      } // for
    } // for
  };
} // check

int
restart_driver() {
  UNIT {
    mesh2d::coord indices{8, 8};
    mesh2d::base::colors colors{4, 1};
    mesh2d::coord hdepths{0, 0};
    mesh2d::coord bdepths{0, 0};
    std::vector<bool> periodic{false, false};
    std::vector<mesh2d::coloring_definition> index_definitions = {
      {colors, indices, hdepths, bdepths, periodic, false}};
    mc.allocate(index_definitions);
    m.allocate(mc.get());

    auto mf1 = m_field_1(m);
    auto mf2 = m_field_2(m);
    execute<init>(m, mf1, mf2);

    int num_files = 4;
    io::io_interface iif{num_files};
    // TODO:  make this registration automatic, not manual
    iif.add_region<mesh2d, mesh2d::index_space::entities>(m);
    iif.checkpoint_all_fields("hdf5_restart.dat");

    execute<clear>(m, mf1, mf2);
    iif.recover_all_fields("hdf5_restart.dat");

    EXPECT_EQ(test<check>(m, mf1, mf2), 0);
  };

  return 0;
}

flecsi::unit::driver<restart_driver> driver;
