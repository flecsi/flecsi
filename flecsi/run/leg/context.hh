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

#include <flecsi-config.h>

#include "flecsi/run/context.hh"

#if !defined(FLECSI_ENABLE_LEGION)
#error FLECSI_ENABLE_LEGION not defined! This file depends on Legion!
#endif

#include <legion.h>

#if !defined(FLECSI_ENABLE_MPI)
#error FLECSI_ENABLE_MPI not defined! This file depends on MPI!
#endif

#include <mpi.h>

#include <functional>
#include <map>
#include <string_view>
#include <unordered_map>

namespace flecsi::run {

inline constexpr Legion::TaskID FLECSI_TOP_LEVEL_TASK_ID = 0;
namespace mapper {
// force_rank_match: mapper tag that will force 1-to-1 MPI-to-Legion mapping;
// compacted_storage:  mapper tag that is used for creating compacted Regions;
// subrank_launch: mapper tag that is used for nested tasks, not supported in
// FleCSI yet;
// exclusive_lr: mapper tag that is used to mark first region in compacted
//              instance;
// prefer_gpu, prefer_omp:  tags, that are used to specify what is apreferred
//              processor type;
constexpr Legion::MappingTagID force_rank_match = 0x00001000,
#if 0
                               compacted_storage = 0x00002000,
                               subrank_launch = 0x00003000,
                               exclusive_lr = 0x00004000,
#endif
                               prefer_gpu = 0x11000001, prefer_omp = 0x11000002;
} // namespace mapper

namespace leg {
template<class R = void>
using task = R(const Legion::Task *,
  const std::vector<Legion::PhysicalRegion> &,
  Legion::Context,
  Legion::Runtime *);
}

struct context_t : context {

  /*
    Friend declarations. Some parts of this interface are intentionally private
    to avoid inadvertent corruption of initialization logic.
   */

  friend leg::task<> top_level_task;

  //--------------------------------------------------------------------------//
  //  Runtime.
  //--------------------------------------------------------------------------//

  /*
    Documentation for this interface is in the top-level context type.
   */

  int initialize(int argc, char ** argv, bool dependent);

  /*
    Documentation for this interface is in the top-level context type.
   */

  void finalize();

  /*
    Documentation for this interface is in the top-level context type.
   */

  int start(const std::function<int()> &);

  /*
    Documentation for this interface is in the top-level context type.
   */

  void clear() {} // clear

  /*
    Documentation for this interface is in the top-level context type.
   */

  Color process() const {
    return context::process_;
  } // process

  /*
    Documentation for this interface is in the top-level context type.
   */

  Color processes() const {
    return context::processes_;
  } // processes

  /*
    Documentation for this interface is in the top-level context type.
   */

  Color threads_per_process() const {
    return context::threads_per_process_;
  } // threads_per_process

  /*
    Documentation for this interface is in the top-level context type.
   */

  Color threads() const {
    return context::threads_;
  } // threads

  /*
    Documentation for this interface is in the top-level context type.
   */

  static int task_depth() {
    return Legion::Runtime::get_runtime()
      ->get_current_task(Legion::Runtime::get_context())
      ->get_depth();
  } // task_depth

  /*
    Documentation for this interface is in the top-level context type.
   */

  static Color color() {
    flog_assert(
      task_depth() > 0, "this method can only be called from within a task");
    return Legion::Runtime::get_runtime()
      ->get_current_task(Legion::Runtime::get_context())
      ->index_point.point_data[0];
  } // color

  /*
    Documentation for this interface is in the top-level context type.
   */

  static Color colors() {
    flog_assert(
      task_depth() > 0, "this method can only be called from within a task");
    return Legion::Runtime::get_runtime()
      ->get_current_task(Legion::Runtime::get_context())
      ->index_domain.get_volume();
  } // colors

  //--------------------------------------------------------------------------//
  //  MPI interoperability.
  //--------------------------------------------------------------------------//

  void * mpi_params;

  /*!
    Set the MPI user task. When control is given to the MPI runtime
    it will execute whichever function is currently set.
   */

  void mpi_call(std::function<void()> mpi_task) {
    {
      log::devel_guard guard(context_tag);
      flog_devel(info) << "In mpi_call" << std::endl;
    }

    mpi_task_ = std::move(mpi_task);
    mpi_handoff();
    mpi_wait();
  }

  /*!
    Set the distributed-memory domain.
   */

  void set_all_processes(const LegionRuntime::Arrays::Rect<1> & all_processes) {
    all_processes_ = all_processes;
  } // all_processes

  /*!
     Return the distributed-memory domain.
   */

  const LegionRuntime::Arrays::Rect<1> & all_processes() const {
    return all_processes_;
  } // all_processes

  /*!
    Connect with the MPI runtime.

    @param ctx The Legion runtime context.
    @param runtime The Legion task runtime pointer.
   */

  void connect_with_mpi(Legion::Context & ctx, Legion::Runtime * runtime);

  /*!
    Handoff to MPI from Legion.
   */
  void mpi_handoff() {
    handshake_.legion_handoff_to_mpi();
  }

  /*!
    Wait for MPI runtime to complete task execution.
   */

  void mpi_wait() {
    handshake_.legion_wait_on_mpi();
  }

private:
  /*--------------------------------------------------------------------------*
    Runtime data.
   *--------------------------------------------------------------------------*/

  const std::function<int()> * top_level_action_ = nullptr;

  /*--------------------------------------------------------------------------*
    Interoperability data members.
   *--------------------------------------------------------------------------*/

  std::function<void()> mpi_task_;
  Legion::MPILegionHandshake handshake_;
  LegionRuntime::Arrays::Rect<1> all_processes_;
};

} // namespace flecsi::run
