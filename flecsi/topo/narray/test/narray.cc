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

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/narray/test/narray_tasks.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

int
coloring_driver() {
  UNIT() {
#if 1
    mesh3d::coord indices{8, 8, 8};
    std::vector<Color> colors = {2, 2, 1};
#else
    mesh3d::coord indices{9, 9, 9};
    std::vector<Color> colors = {3, 3, 1};
#endif

    flog(warn) << "colors: " << std::endl
               << flog::container{colors} << std::endl;

    mesh3d::coord hdepths{1, 1, 1};
    mesh3d::coord bdepths{0, 0, 0};
    std::vector<bool> periodic{false, false, false};
    std::vector<bool> extend{true, false, false};

    mesh3d::coloring_definition cd = {
      colors, indices, hdepths, bdepths, periodic, false};
    auto [nc, ne, coloring, partitions] =
      topo::narray_utils::color(cd, MPI_COMM_WORLD);

    auto [avpc, aprts] =
      topo::narray_utils::color_auxiliary(ne, nc, coloring, extend);

    std::stringstream ss;
    ss << "primary" << std::endl;
    for(auto p : coloring) {
      ss << p << std::endl;
    } // for
    ss << "auxiliary" << std::endl;
    for(auto p : avpc) {
      ss << p << std::endl;
    } // for
    flog(warn) << ss.str() << std::endl;

    flog(warn) << flog::container{partitions} << std::endl;
    flog(warn) << flog::container{aprts} << std::endl;
  };
}

flecsi::unit::driver<coloring_driver> cd;

#if 1
// 1D Mesh
mesh1d::slot m1;
mesh1d::cslot coloring1;
const field<std::size_t>::definition<mesh1d, mesh1d::index_space::entities> f1;

// 2D Mesh
mesh2d::slot m2;
mesh2d::cslot coloring2;
const field<std::size_t>::definition<mesh2d, mesh2d::index_space::entities> f2;

// 3D Mesh
mesh3d::slot m3;
mesh3d::cslot coloring3;
const field<std::size_t>::definition<mesh3d, mesh3d::index_space::entities> f3;

int
narray_driver() {
  UNIT() {
    {
      using topo::narray_utils::factor;
      using V = std::vector<std::size_t>;
      EXPECT_EQ(factor(2 * 5 * 11 * 13 * 29), (V{29, 13, 11, 5, 2}));
      EXPECT_EQ(factor(2 * 2 * 23 * 23), (V{23, 23, 2, 2}));
    } // scope

    {
      // 1D Mesh
      mesh1d::coord indices{9};
      auto colors = topo::narray_utils::distribute(processes(), indices);
      flog(warn) << flog::container{colors} << std::endl;

      mesh1d::coord hdepths{1};
      mesh1d::coord bdepths{2};
      std::vector<bool> periodic{false};
      bool diagonals = true;

      mesh1d::coloring_definition cd = {
        colors, indices, hdepths, bdepths, periodic, diagonals};

      // coloring1.allocate(index_definitions);
      coloring1.allocate(cd);
      m1.allocate(coloring1.get());
      execute<init_field<1>, default_accelerator>(m1, f1(m1));
      execute<print_field<1>>(m1, f1(m1));
      execute<update_field<1>, default_accelerator>(m1, f1(m1));
      execute<print_field<1>>(m1, f1(m1));
      execute<check_mesh_field<1>>(m1, f1(m1));
    } // scope

    {
      // 2D Mesh
      mesh2d::coord indices{8, 8};
      auto colors = topo::narray_utils::distribute(processes(), indices);
      flog(warn) << flog::container{colors} << std::endl;

      mesh2d::coord hdepths{1, 2};
      mesh2d::coord bdepths{2, 1};
      std::vector<bool> periodic{false, false};
      bool diagonals = true;

      mesh1d::coloring_definition cd = {
        colors, indices, hdepths, bdepths, periodic, diagonals};

      coloring2.allocate(cd);
      m2.allocate(coloring2.get());
      execute<init_field<2>, default_accelerator>(m2, f2(m2));
      execute<print_field<2>>(m2, f2(m2));
      execute<update_field<2>, default_accelerator>(m2, f2(m2));
      execute<print_field<2>>(m2, f2(m2));
      execute<check_mesh_field<2>>(m2, f2(m2));

    } // scope

    {
      // 3D Mesh
      mesh3d::coord indices{4, 4, 4};
      auto colors = topo::narray_utils::distribute(processes(), indices);
      flog(warn) << flog::container{colors} << std::endl;

      mesh3d::coord hdepths{1, 1, 1};
      mesh3d::coord bdepths{1, 1, 1};
      std::vector<bool> periodic{false, false, false};
      bool diagonals = true;
      mesh3d::coloring_definition cd = {
        colors, indices, hdepths, bdepths, periodic, diagonals};

      coloring3.allocate(cd);
      m3.allocate(coloring3.get());
      execute<init_field<3>, default_accelerator>(m3, f3(m3));
      execute<print_field<3>>(m3, f3(m3));
      execute<update_field<3>, default_accelerator>(m3, f3(m3));
      execute<print_field<3>>(m3, f3(m3));
      execute<check_mesh_field<3>>(m3, f3(m3));
    } // scope
  };
} // narray_driver

flecsi::unit::driver<narray_driver> nd;
#endif
