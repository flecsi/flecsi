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
#ifndef FLECSI_IO_BACKEND_HH
#define FLECSI_IO_BACKEND_HH

#include <flecsi-config.h>

#include "flecsi/data/field.hh"
#include "flecsi/io/hdf5.hh"
#include "flecsi/topo/index.hh"

/*----------------------------------------------------------------------------*
  This section works with the build system to select the correct backend
  implemenation for the io model.
 *----------------------------------------------------------------------------*/

#if FLECSI_BACKEND == FLECSI_BACKEND_legion

#include "flecsi/io/leg/policy.hh"

#elif FLECSI_BACKEND == FLECSI_BACKEND_mpi

#include <flecsi/io/mpi/policy.hh>

#elif FLECSI_BACKEND == FLECSI_BACKEND_hpx

#include <flecsi/io/hpx/policy.hh>

#endif // FLECSI_BACKEND

#endif
