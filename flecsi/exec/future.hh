// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_FUTURE_HH
#define FLECSI_EXEC_FUTURE_HH

#if FLECSI_BACKEND == FLECSI_BACKEND_legion

#include "flecsi/exec/leg/future.hh"

#elif FLECSI_BACKEND == FLECSI_BACKEND_mpi

#include "flecsi/exec/mpi/future.hh"

#endif // FLECSI_BACKEND

#ifdef DOXYGEN // implemented per-backend
/// Single-valued future.
template<typename Return>
struct future<Return> {
  /// Wait on the task to finish.
  void wait();
  /// Get the task's result.
  [[nodiscard]] Return get(bool silence_warnings = false);
};

/// Multi-valued future from an index launch.
template<typename Return>
struct future<Return, exec::launch_type_t::index> {
  /// Wait on all the tasks to finish.
  void wait(bool silence_warnings = false);
  /// Get the result of one of the tasks.
  /// Note that all processes must select the same \a index.
  Return get(Color index = 0, bool silence_warnings = false);
  /// Get the number of tasks.
  Color size() const;
};

/// \cond core
/// Generate a new future from a value
template<class Return>
future<Return> make_future(Return);
/// \endcond
#endif // DOXYGEN
#endif
