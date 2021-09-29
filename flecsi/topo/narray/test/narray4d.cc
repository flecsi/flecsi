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

// !!!!!FIX
// This test should be merged into narray.cc
// once the multi-accessor functionality is available.

// 4D Mesh
mesh4d::slot m4;
mesh4d::cslot coloring4;

int
narray_driver() {
  UNIT {

    {
      // 4D Mesh
      mesh4d::coord indices{4, 4, 4, 4};
      auto colors = topo::narray_utils::distribute(processes(), indices);
      flog(warn) << log::container{colors} << std::endl;

      mesh4d::coord hdepths{1, 1, 1, 1};
      mesh4d::coord bdepths{1, 1, 1, 1};
      std::vector<bool> periodic{false, false, false, false};
      bool diagonals = true;
      mesh4d::coloring_definition cd = {
        colors, indices, hdepths, bdepths, periodic, diagonals};
      coloring4.allocate(cd);
      m4.allocate(coloring4.get());
      execute<check_4dmesh>(m4);
    } // scope

  }; // UNIT
} // narray_driver

flecsi::unit::driver<narray_driver> nd;
