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
#include <flecsi/data.hh>
#include <flecsi/utils/ftest.hh>

using namespace flecsi;

int
identifiers(int argc, char ** argv) {

  FTEST();

  flog(info) << "global_topology_t: "
             << topology::id<topology::global_topology_t>() << std::endl;
  flog(info) << "index_topology_t: "
             << topology::id<topology::index_topology_t>() << std::endl;

  flog(info) << "global topology handle " << flecsi_global_topology.identifier()
             << std::endl;
  flog(info) << "index topology handle " << flecsi_index_topology.identifier()
             << std::endl;

  return FTEST_RESULT();
}

ftest_register_driver(identifiers);