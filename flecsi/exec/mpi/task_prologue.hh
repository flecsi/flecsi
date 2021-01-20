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

/*! @file */

#include <flecsi-config.h>

#if !defined(__FLECSI_PRIVATE__)
#error Do not include this file directly!
#endif

#include "flecsi/data/privilege.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/exec/mpi/future.hh"
#include "flecsi/util/demangle.hh"

#if !defined(FLECSI_ENABLE_MPI)
#error FLECSI_ENABLE_MPI not defined! This file depends on MPI!
#endif

#include <mpi.h>

#include <memory>

namespace flecsi {
namespace topo {
struct global_base;
}

namespace exec {

struct task_prologue {
protected:
  // Those methods are "protected" because they are *only* called by
  // flecsi::exec::prolog() which inherits from task_prologue.

  // Patch up the "un-initialized" type conversion from future<R, index>
  // to future<R, single> in the generic code.
  template<typename R>
  static void visit(future<R, exec::launch_type_t::single> & single,
    future<R, exec::launch_type_t::index> & index) {
    single.result_ = index.result;
  }

  template<typename T, size_t P, class Topo, typename Topo::index_space Space>
  static void visit(data::accessor<data::raw, T, P> & accessor,
    const data::field_reference<T, data::raw, Topo, Space> & ref) {
    const field_id_t f = ref.fid();
    auto & t = ref.topology();
    data::region & reg = t.template get_region<Space>();
    constexpr bool glob =
      std::is_same_v<typename Topo::base, topo::global_base>;

    const auto storage = [&]() -> auto & {
      if constexpr(glob)
        return *t;
      else
        // The partition controls how much memory is allocated.
        return t.template get_partition<Space>(f);
    }
    ().template get_storage<T>(f);
    if constexpr(glob) {
      if(reg.ghost<privilege_pack<get_privilege(0, P), ro>>(f))
        MPI_Bcast(storage.data(),
          storage.size(),
          flecsi::util::mpi::type<T>(),
          0,
          MPI_COMM_WORLD);
    }
    else
      reg.ghost_copy<P>(ref);
    accessor.bind(storage);
  } // visit generic topology
}; // struct task_prologue_t
} // namespace exec
} // namespace flecsi
