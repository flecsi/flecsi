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

/*!
  @file

  User interface to the FleCSI
  data model.
 */

#include "flecsi/data.hh"

namespace flecsi {
/*
  Default global topology instance definition.
 */
topo::global::slot global_topology;

/*
  Per-process topology instance definition.
 */
topo::index::slot process_topology;
} // namespace flecsi
