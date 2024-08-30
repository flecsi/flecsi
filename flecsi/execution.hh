// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXECUTION_HH
#define FLECSI_EXECUTION_HH

#include "flecsi/exec/backend.hh"
#include "flecsi/exec/fold.hh"
#include "flecsi/exec/kernel.hh"

#include "flecsi/flog.hh"
#include "flecsi/runtime.hh" // for compatibility

namespace flecsi {

namespace flog {

/*!
  Explicitly flush buffered flog output.
  \code#include "flecsi/execution.hh"\endcode

  @ingroup flog
 */

inline void
flush() {
#if defined(FLECSI_ENABLE_FLOG) && defined(FLOG_ENABLE_MPI)
  auto & s = flog::state::instance();
  if(s.source_process() == 0 || (s.active_process() && s.processes() == 1)) {
    flog::state::gather(s); // no MPI communication needed
  }
  else {
    flecsi::exec::reduce_internal<flog::state::gather, void, flecsi::mpi>(s);
  }
  flecsi::run::context::instance().flog_task_count() = 0;
#endif
} // flush

inline void
maybe_flush() {
#if defined(FLECSI_ENABLE_FLOG) && defined(FLOG_ENABLE_MPI)
  auto & flecsi_context = run::context::instance();
  unsigned & flog_task_count = flecsi_context.flog_task_count();
  if(flog_task_count >= flog::state::instance().serialization_interval())
    flush();
#endif
} // maybe_flush

} // namespace flog

/// \defgroup execution Execution Model
/// Launching tasks and kernels.  Tasks are coarse-grained and use
/// distributed-memory with restricted side effects; kernels are fine-grained
/// and data-parallel, possibly using an accelerator.
/// \code#include "flecsi/execution.hh"\endcode
/// \{

/// A global variable with a task-specific value.
/// Must be constructed before calling \c start.
/// The value for a task has the lifetime of that task; the value outside of
/// any task has the lifetime of \c start.  Each is value-initialized.
/// \note Thread-local variables do not function correctly in all backends.
template<class T>
struct task_local
#ifdef DOXYGEN // implemented per-backend
{
  /// Create a task-local variable.
  task_local();
  /// It would not be clear whether moving a \c task_local applied to the
  /// (current) value or the identity of the variable.
  task_local(task_local &&) = delete;

  /// Get the current task's value.
  T & operator*() & noexcept;
  /// Access a member of the current task's value.
  T * operator->() noexcept;
}
#endif
;

/*!
  Execute a reduction task.

  @tparam Task       The user task.
  @tparam Reduction  The reduction operation type.
  @tparam Attributes The task attributes mask.
  @tparam Args       The user-specified task arguments.
  \return a \ref future providing the reduced return value

  \see \c execute about parameter and argument types.
 */

// To avoid compile- and runtime recursion, only user tasks trigger logging.
template<auto & Task,
  class Reduction,
  TaskAttributes Attributes = flecsi::loc | flecsi::leaf,
  typename... Args>
[[nodiscard]] auto
reduce(Args &&... args) {
  using namespace exec;

  ++run::context::instance().flog_task_count();
  flog::maybe_flush();

  return reduce_internal<Task, Reduction, Attributes, Args...>(
    std::forward<Args>(args)...);
} // reduce

template<auto & TASK, TaskAttributes ATTRIBUTES, typename... ARGS>
auto
execute(ARGS &&... args) {
  return reduce<TASK, void, ATTRIBUTES>(std::forward<ARGS>(args)...);
} // execute

/*!
  Execute a test task. This interface is provided for FleCSI's unit testing
  framework. Test tasks must return an integer that is non-zero on failure,
  and zero otherwise.

  @tparam TASK       The user task. Its parameters may be of any
                     default-constructible, trivially-move-assignable,
                     non-pointer type, any type that supports the Legion
                     return-value serialization interface, or any of several
                     standard containers of such types. If \a ATTRIBUTES
                     specifies an MPI task, parameters need merely be movable.
  @tparam ATTRIBUTES The task attributes mask.
  @tparam ARGS       The user-specified task arguments, implicitly converted to
                     the parameter types for \a TASK.

  @return zero on success, non-zero on failure.
 */

template<auto & TASK,
  TaskAttributes ATTRIBUTES = flecsi::loc | flecsi::leaf,
  typename... ARGS>
[[nodiscard]] int
test(ARGS &&... args) {
  return reduce<TASK, exec::fold::sum, ATTRIBUTES>(std::forward<ARGS>(args)...)
    .get();
} // test

/// \}

namespace exec {
/// \addtogroup execution
/// \{

#ifdef DOXYGEN // implemented per-backend
/// Records execution of a loop whose iterations all execute the same sequence
/// of tasks.  With the Legion backend, subsequent iterations run faster if
/// traced.  Some \c data::mutator specializations cannot be traced.  The
/// first iteration should be ignored if it might perform different
/// ghost copies.
struct trace {

  using id_t = int;

  /// Construct a trace.
  trace();
  /// Construct a trace with user defined id
  /// \param id User defined id for the trace
  /// \deprecated Use the default constructor.
  explicit trace(id_t id);

  /// Traces are movable.  Those that have been moved from must not be used.
  trace(trace &&) noexcept;
  /// Traces can be (move-)assigned.
  trace & operator=(trace) & noexcept;

  struct guard;

  /// Create a <code>\ref guard</code> for this \c trace.
  inline guard make_guard();

  /// Skip the next call to the tracer
  void skip();

  /// \if core
  /// Check whether a trace is supported, active, and not skipped.
  /// \endif
  static bool is_tracing();

private:
  /// Non-RAII interface.  Does nothing if \c skip flag is set.
  void start();
  /// Non-RAII interface.  Merely clears \c skip flag if set.
  void stop();
};
#endif

/// RAII guard for executing a trace.
/// Flog output is deferred to the end of the trace as needed.
struct trace::guard {
  /// Immovable.
  guard(guard &&) = delete;

  /// Start a trace.  Required in certain contexts like use of \c
  /// std::optional; otherwise prefer \c trace::make_guard.
  explicit guard(trace & t_) : t(t_) {
    current_flog_task_count =
      std::exchange(flecsi::run::context::instance().flog_task_count(), 0);
    t.start();
  }

  // Destroy a guard by stopping the tracing.
  // The flog count is merged and triggered if needed.
  ~guard() {
    t.stop();
    flecsi::run::context::instance().flog_task_count() +=
      current_flog_task_count;
    flog::maybe_flush();
  }

private:
  trace & t;
  unsigned current_flog_task_count;

}; // struct trace::guard

/// \}

trace::guard
trace::make_guard() {
  return guard(*this);
}
} // namespace exec

} // namespace flecsi

#endif
