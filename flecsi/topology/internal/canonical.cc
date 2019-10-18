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

/*! @file */

#include <flecsi/execution.hh>
#include <flecsi/topology/internal/canonical.hh>

#if !defined(FLECSI_ENABLE_MPI)
#error FLECSI_ENABLE_MPI not defined! This file depends on MPI!
#endif

#include <mpi.h>

namespace flecsi {
namespace topology {

void
canonical_topology_base::coloring::color(coloring & coloring_info,
  std::string const & filename) {

  std::cout << "process " << process() << " of " << processes() << " with "
            << threads_per_process() << " (tpp) and input " << filename
            << std::endl;
} // canonical_topology_base::coloring::color

} // namespace topology
} // namespace flecsi