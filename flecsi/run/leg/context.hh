// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_LEG_CONTEXT_HH
#define FLECSI_RUN_LEG_CONTEXT_HH

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
/// \defgroup legion-runtime Legion Runtime
/// State for and control of the Legion runtime.
/// \ingroup runtime
/// \{

namespace mapper {
/// \addtogroup legion-runtime
/// \{

/// \name Mapper tags
/// Flags used to request custom mapper features.
/// \{

inline constexpr Legion::MappingTagID
  force_rank_match = 0x00001000, ///< Put colors on corresponding MPI ranks.
#if 0
  compacted_storage = 0x00002000, ///< Combine exclusive, shared, and ghosts.
  subrank_launch = 0x00003000, ///< For nested tasks.
  exclusive_lr = 0x00004000, ///< Indicate first region in compacted set.
#endif
  prefer_gpu = 0x11000001, ///< Request GPU execution.
  prefer_omp = 0x11000002; ///< Request OpenMP execution.
/// \}
/// \}
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

/// \}
} // namespace flecsi::run

#endif
