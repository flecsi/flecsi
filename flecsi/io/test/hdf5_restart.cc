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
#include "flecsi/run/context.hh"
#include "flecsi/topo/narray/test/narray.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/unit.hh"

#include <vector>

using namespace flecsi;

using mesh2d = mesh<2>;
using is = mesh2d::index_space;
using rg = mesh2d::range;
using ax = mesh2d::axis;

mesh2d::slot m;
mesh2d::cslot mc;

const field<double>::definition<mesh2d, mesh2d::index_space::entities>
  m_field_1, m_field_2;
const field<int>::definition<mesh2d, mesh2d::index_space::entities> m_field_i;
const field<std::size_t>::definition<mesh2d, mesh2d::index_space::entities>
  m_field_s;

void
init(mesh2d::accessor<ro> m,
  field<double>::accessor<wo, na> mf1,
  field<double>::accessor<wo, na> mf2,
  field<int>::accessor<wo, na> mfi,
  field<std::size_t>::accessor<wo, na> mfs) {
  auto ms1 = m.mdspan<is::entities>(mf1);
  auto ms2 = m.mdspan<is::entities>(mf2);
  auto msi = m.mdspan<is::entities>(mfi);
  auto mss = m.mdspan<is::entities>(mfs);
  for(auto i : m.extents<ax::x_axis, rg::all>()) {
    for(auto j : m.extents<ax::y_axis, rg::all>()) {
      double val = 16. * color() + 8. * (int)i + (int)j;
      ms1[j][i] = val;
      ms2[j][i] = val + 1000.;
      msi[j][i] = val + 2000;
      mss[j][i] = val + 3000;
    } // for
  } // for
} // init

void
clear(mesh2d::accessor<ro> m,
  field<double>::accessor<wo, na> mf1,
  field<double>::accessor<wo, na> mf2,
  field<int>::accessor<wo, na> mfi,
  field<std::size_t>::accessor<wo, na> mfs) {
  auto ms1 = m.mdspan<is::entities>(mf1);
  auto ms2 = m.mdspan<is::entities>(mf2);
  auto msi = m.mdspan<is::entities>(mfi);
  auto mss = m.mdspan<is::entities>(mfs);
  for(auto i : m.extents<ax::x_axis, rg::all>()) {
    for(auto j : m.extents<ax::y_axis, rg::all>()) {
      ms1[j][i] = 0.;
      ms2[j][i] = 0.;
      msi[j][i] = 0;
      mss[j][i] = 0;
    } // for
  } // for
} // clear

int
check(mesh2d::accessor<ro> m,
  field<double>::accessor<ro, na> mf1,
  field<double>::accessor<ro, na> mf2,
  field<int>::accessor<ro, na> mfi,
  field<std::size_t>::accessor<ro, na> mfs) {
  UNIT {
    auto ms1 = m.mdspan<is::entities>(mf1);
    auto ms2 = m.mdspan<is::entities>(mf2);
    auto msi = m.mdspan<is::entities>(mfi);
    auto mss = m.mdspan<is::entities>(mfs);
    for(auto i : m.extents<ax::x_axis, rg::all>()) {
      for(auto j : m.extents<ax::y_axis, rg::all>()) {
        double val = 16. * color() + 8. * (int)i + (int)j;
        ASSERT_EQ(ms1[j][i], val);
        ASSERT_EQ(ms2[j][i], val + 1000.);
        ASSERT_EQ(msi[j][i], val + 2000);
        ASSERT_EQ(mss[j][i], val + 3000);
      } // for
    } // for
  };
} // check

static bool added_topology = false;

template<bool Attach>
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

    // add topology only on first call
    if(!added_topology) {
      run::context::instance().add_topology<mesh2d>(m);
      added_topology = true;
    }

    auto mf1 = m_field_1(m);
    auto mf2 = m_field_2(m);
    auto mfi = m_field_i(m);
    auto mfs = m_field_s(m);

    execute<init>(m, mf1, mf2, mfi, mfs);

    int num_files = 4;
    io::io_interface iif{num_files};
    auto filename =
      std::string{"hdf5_restart"} + (Attach ? "_w" : "_wo") + ".dat";
    iif.checkpoint_all_fields(filename, Attach);

    execute<clear>(m, mf1, mf2, mfi, mfs);
    iif.recover_all_fields(filename, Attach);

    EXPECT_EQ(test<check>(m, mf1, mf2, mfi, mfs), 0);
  };

  return 0;
}

// Run test twice, once with and once without attach.
flecsi::unit::driver<restart_driver<true>> driver_w;
flecsi::unit::driver<restart_driver<false>> driver_wo;
