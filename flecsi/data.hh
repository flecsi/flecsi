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
#pragma once

#include "flecsi/data/map.hh"
#include "flecsi/data/topology_accessor.hh"
#include "flecsi/topo/global.hh"
#include "flecsi/topo/index.hh"
#include "flecsi/topo/narray/interface.hh"
#include "flecsi/topo/ntree/interface.hh"
#include "flecsi/topo/set/interface.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include <flecsi/data/accessor.hh>
#include <flecsi/data/coloring.hh>
#include <flecsi/data/privilege.hh>

namespace flecsi {
/// \defgroup data Data Model
/// Defining topology instances and fields.
/// \{

/*!
  Default global topology instance with size 1.
  \deprecated Create instances as with any other topology.
 */
inline topo::global::slot global_topology;

/*!
  Topology instance with one color per process.
  \warning The values are not bound to processes except with MPI tasks.
  \deprecated Create instances as with any other topology.
 */
inline topo::index::slot process_topology;

/// \cond core
namespace detail {
/// An RAII type to manage the global coloring and topologies.
struct data_guard {
  struct global_guard {
    global_guard() {
      global_topology.allocate({});
    }
    global_guard(global_guard &&) = delete;
    ~global_guard() {
      global_topology.deallocate();
    }
  } g;
  struct process_guard {
    process_guard() {
      process_topology.allocate(run::context::instance().processes());
    }
    process_guard(process_guard &&) = delete;
    ~process_guard() {
      process_topology.deallocate();
    }
  } p;
};
} // namespace detail
/// \endcond
/// \}
} // namespace flecsi
