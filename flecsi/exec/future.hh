// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_FUTURE_HH
#define FLECSI_EXEC_FUTURE_HH

#include "flecsi/config.hh"
#include "flecsi/data/tags.hh"
#include "flecsi/util/types.hh" // Color

#include <type_traits>
#include <vector>

namespace flecsi {
/// \addtogroup execution
/// \{

namespace exec {
/// Selects a type of \c future.
/// That it is a type is \b deprecated; it will become a namespace.
enum class launch_type_t : size_t {
  /// A future from a reduction or single task.
  single,
  /// A future from a non-reduction index task.
  index
};

} // namespace exec

/// The type of the second template parameter for \c future.
/// That it identifies the particular type shown is \b deprecated.
/// \ns.
/// \showinitializer
using future_kind = exec::launch_type_t;

/*!
  \link future<Return> Single\endlink or \link
  future<Return,exec::launch_type_t::index> multiple\endlink future.

  A single future can be a task argument and parameter; the task runs only
  when the value is ready.
  A multi-valued future may be passed to a task expecting a single one
  (which is then executed once with each value).

  @tparam Return The return type of the task.
  @tparam Launch FleCSI launch type: single/index.
  \ns.
*/
template<typename Return, future_kind Launch = exec::launch_type_t::single>
struct future;

/// \}
} // namespace flecsi

#if FLECSI_BACKEND == FLECSI_BACKEND_legion

#include "flecsi/exec/leg/future.hh"

#elif FLECSI_BACKEND == FLECSI_BACKEND_mpi

#include "flecsi/exec/mpi/future.hh"

#elif FLECSI_BACKEND == FLECSI_BACKEND_hpx

#include "flecsi/exec/hpx/future.hh"

#endif // FLECSI_BACKEND

#ifdef DOXYGEN // implemented per-backend
namespace flecsi {
/// \addtogroup execution
/// \{

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
  /// \deprecated Use \c all or pass to a task to process values in parallel.
  Return get(Color index = 0, bool silence_warnings = false);
  /// Get the results of all tasks.
  /// \note This member does not exist if \a Return is \c void.
  std::vector<Return> all();
  /// Get the number of tasks.
  Color size() const;
};

/// \cond core
/// Generate a new future from a value.  \ns.
template<class Return>
future<std::decay_t<Return>> make_future(Return &&);
/// \endcond

/// \}
} // namespace flecsi
#endif // DOXYGEN
#endif
