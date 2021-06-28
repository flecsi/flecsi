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

/*
//4D Mesh
mesh4d::slot m4;
mesh4d::cslot coloring4;
*/

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
      // 1D Mesh
      mesh1d::coord indices{9};
      auto colors = topo::narray_utils::distribute(processes(), indices);
      flog(warn) << log::container{colors} << std::endl;

      mesh1d::coord hdepths{1};
      mesh1d::coord bdepths{2};
      std::vector<bool> periodic{false};
      bool diagonals = true;
      std::vector<mesh1d::coloring_definition> index_definitions = {
        {colors, indices, hdepths, bdepths, periodic, diagonals}};

      coloring1.allocate(index_definitions);
      m1.allocate(coloring1.get());
      execute<init_field<1>>(m1, f1(m1));
      execute<print_field<1>>(m1, f1(m1));
      execute<update_field<1>>(m1, f1(m1));
      execute<print_field<1>>(m1, f1(m1));
      execute<check_mesh_field<1>>(m1, f1(m1));
    } // scope

    {
      // 2D Mesh
      mesh2d::coord indices{8, 8};
      auto colors = topo::narray_utils::distribute(processes(), indices);
      flog(warn) << log::container{colors} << std::endl;

      mesh2d::coord hdepths{1, 2};
      mesh2d::coord bdepths{2, 1};
      std::vector<bool> periodic{false, false};
      bool diagonals = true;
      std::vector<mesh2d::coloring_definition> index_definitions = {
        {colors, indices, hdepths, bdepths, periodic, diagonals}};
      coloring2.allocate(index_definitions);
      m2.allocate(coloring2.get());
      execute<init_field<2>>(m2, f2(m2));
      execute<print_field<2>>(m2, f2(m2));
      execute<update_field<2>>(m2, f2(m2));
      execute<print_field<2>>(m2, f2(m2));
      execute<check_mesh_field<2>>(m2, f2(m2));

    } // scope

    {
      // 3D Mesh
      mesh3d::coord indices{4, 4, 4};
      auto colors = topo::narray_utils::distribute(processes(), indices);
      flog(warn) << log::container{colors} << std::endl;

      mesh3d::coord hdepths{1, 1, 1};
      mesh3d::coord bdepths{1, 1, 1};
      std::vector<bool> periodic{false, false, false};
      bool diagonals = true;
      std::vector<mesh3d::coloring_definition> index_definitions = {
        {colors, indices, hdepths, bdepths, periodic, diagonals}};
      coloring3.allocate(index_definitions);
      m3.allocate(coloring3.get());
      execute<init_field<3>>(m3, f3(m3));
      execute<print_field<3>>(m3, f3(m3));
      execute<update_field<3>>(m3, f3(m3));
      execute<print_field<3>>(m3, f3(m3));
      execute<check_mesh_field<3>>(m3, f3(m3));
    } // scope

    /*
     {
       mesh4d::coord indices{4, 4, 4, 4};
       auto colors = topo::narray_utils::distribute(processes(), indices);
       flog(warn) << log::container{colors} << std::endl;

       mesh4d::coord hdepths{0, 0, 0, 0};
       mesh4d::coord bdepths{0, 0, 0, 0};
       std::vector<bool> periodic{false, false, false, false};
       std::vector<mesh4d::coloring_definition> index_definitions = {
         {colors, indices, hdepths, bdepths, periodic, true}};
       coloring4.allocate(index_definitions);

       auto [colors_out, index_colorings] =
     topo::narray_utils::color(index_definitions, MPI_COMM_WORLD);

       mesh4d::coloring c;
       c.comm = MPI_COMM_WORLD;
       c.colors = colors_out;
       for(auto idx : index_colorings) {
         for(auto ic : idx) {
           c.idx_colorings.emplace_back(ic.second);
         }
       }

      auto idc = c.idx_colorings[0];

      int dim = indices.size();
      int owner = process();
      std::string fname1 = "narray_" + std::to_string(dim) + "d_" +
     std::to_string(owner) + ".current";

       UNIT_CAPTURE() << "index_coloring ---->\n";
       UNIT_CAPTURE() << " extents = [ ";
       for (auto i = 0; i < dim; i++) {
         UNIT_CAPTURE()<<idc.extents[i]<<"   ";
       }
       UNIT_CAPTURE()<<" ]\n";

       UNIT_CAPTURE() << " logical_min = [ ";
       for (auto i = 0; i < dim; i++) {
         UNIT_CAPTURE()<<idc.logical[0][i]<<"   ";
       }
       UNIT_CAPTURE()<<" ]\n";

       UNIT_CAPTURE() << " logical_max = [ ";
       for (auto i = 0; i < dim; i++) {
         UNIT_CAPTURE()<<idc.logical[1][i]<<"   ";
       }
       UNIT_CAPTURE()<<" ]\n";

       UNIT_CAPTURE() << " extended_min = [ ";
       for (auto i = 0; i < dim; i++) {
         UNIT_CAPTURE()<<idc.extended[0][i]<<"   ";
       }
       UNIT_CAPTURE()<<" ]\n";

       UNIT_CAPTURE() << " extended_max = [ ";
       for (auto i = 0; i < dim; i++) {
         UNIT_CAPTURE()<<idc.extended[1][i]<<"   ";
       }
       UNIT_CAPTURE()<<" ]\n";

       UNIT_EQUAL_BLESSED(fname1.c_str());
     }
   */

  }; // UNIT
} // coloring_driver

flecsi::unit::driver<narray_driver> driver;
