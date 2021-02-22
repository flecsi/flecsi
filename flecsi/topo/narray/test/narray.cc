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

#include "narray.hh"

#define __FLECSI_PRIVATE__
#include "flecsi/execution.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

void
extents(mesh::accessor<ro> m, field<std::size_t>::accessor<wo, na> ca) {
  auto c = m.mdspan<mesh::entities>(ca);
  for(auto j : m.extents<mesh::y_axis>()) {
    for(auto i : m.extents<mesh::x_axis>()) {
      c[j][i] = color();
    } // for
  } // for
}

void
print(mesh::accessor<ro> m, field<std::size_t>::accessor<ro, ro> ca) {
  auto c = m.mdspan<mesh::entities>(ca);
  std::stringstream ss;
  for(int j{int(m.size<mesh::y_axis, mesh::all>() - 1)}; j >= 0; --j) {
    for(auto i : m.extents<mesh::x_axis, mesh::all>()) {
      ss << c[j][i] << " ";
    } // for
    ss << std::endl;
  } // for
  ss << std::endl;
  flog(warn) << ss.str() << std::endl;
}

mesh::slot m;
mesh::cslot coloring;

const field<std::size_t>::definition<mesh, mesh::entities> cs;

int
narray_driver() {
  UNIT {
    {
      using topo::narray_utils::factor;
      using V = std::vector<std::size_t>;
      EXPECT_EQ(factor(2 * 5 * 11 * 13 * 29), (V{29, 13, 11, 5, 2}));
      EXPECT_EQ(factor(2 * 2 * 23 * 23), (V{23, 23, 2, 2}));
    }

    {
      mesh::coord indices{8, 8};
      // mesh::coord indices{16, 16};
      // mesh::coord indices{25, 10};
      // mesh::coord indices{10, 25};
      auto colors = topo::narray_utils::distribute(processes(), indices);
      flog(warn) << log::container{colors} << std::endl;
      // mesh::coord colors{2, 2};

      mesh::coord hdepths{1, 1};
      mesh::coord bdepths{0, 0};
      std::vector<bool> periodic{false, false};
      std::vector<mesh::coloring_definition> index_definitions = {
        {colors, indices, hdepths, bdepths, periodic, true}};
      coloring.allocate(index_definitions);
      m.allocate(coloring.get());
      execute<extents>(m, cs(m));
      execute<print>(m, cs(m));
    } // scope

    {
      mesh::coord indices{8, 8, 8};
      auto colors = topo::narray_utils::distribute(processes(), indices);
      mesh::coord hdepths{1, 1, 1};
      mesh::coord bdepths{0, 0, 0};
      std::vector<bool> periodic{false, false, false};
      std::vector<mesh::coloring_definition> index_definitions = {
        {colors, indices, hdepths, bdepths, periodic, true}};
      auto [clrs, idx_clrngs] = topo::narray_utils::color(index_definitions);
    } // scope
  };
} // coloring_driver

flecsi::unit::driver<narray_driver> driver;
