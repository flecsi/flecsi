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
/// \defgroup execution Execution Model
/// Launching tasks and kernels.  Tasks are coarse-grained and use
/// distributed-memory with restricted side effects; kernels are fine-grained
/// and data-parallel, possibly using an accelerator.
///
/// \ns{exec}.
/// \code#include "flecsi/execution.hh"\endcode
///
/// The inclusion of \ref runtime "flecsi/runtime.hh" by this header is
/// \b deprecated.
///
/// \{

/// A global variable with a task-specific value.
/// Must be constructed before running the control model (or \c start).
/// The value for a task has the lifetime of that task; the value outside of
/// any task has the lifetime of the control model execution.
/// Each is value-initialized.
/// \note Thread-local variables do not function correctly in all backends.
/// \ns.
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

template<class V, class R, class... AA>
auto
scheduler::reduce(AA &&... aa) {
  if constexpr(exec::has_variant_v<V, void>)
    return reduce<V::task, R>(std::forward<AA>(aa)...);
  else {
    static_assert(exec::consistent_task<V, exec::cpu, exec::gpu, exec::omp>,
      "inconsistent parameter types for variants");
    using space = exec::task_variant<V>;
    static constexpr auto & f = V::template task<space>;
    static_assert(util::function_t<f>::nonthrowing, "tasks must be noexcept");
    return flecsi::reduce<f, R, as_mask(space::proc)>(std::forward<AA>(aa)...);
  }
}
template<auto & F, class R, class... AA>
auto
scheduler::reduce(AA &&... aa) {
  using ft = util::function_t<F>;
  static_assert(ft::nonthrowing, "tasks must be noexcept");
  using ps = typename exec::param_space<typename ft::arguments_type>::type;
  return flecsi::reduce<F,
    R,
    as_mask(std::conditional_t<std::is_void_v<ps>, exec::cpu, ps>::proc)>(
    std::forward<AA>(aa)...);
}
template<auto & F, class... AA>
int
scheduler::test(AA &&... aa) {
  return reduce<F, exec::fold::sum>(std::forward<AA>(aa)...).get();
}
template<class V, class... AA>
int
scheduler::test(AA &&... aa) {
  return reduce<V, exec::fold::sum>(std::forward<AA>(aa)...).get();
}

template<class T, class... AA>
auto &
scheduler::allocate(std::unique_ptr<topology<T>> & p,
  const typename T::coloring & c,
  AA &&... aa) {
  p = std::make_unique<topology<T>>(*this, c);
  T::initialize(*this, *p, c, std::forward<AA>(aa)...);
  return *p;
}

// To avoid compile- and runtime recursion, only user tasks trigger logging.
template<auto & Task,
  class Reduction,
  TaskAttributes Attributes,
  typename... Args>
auto
reduce(Args &&... args) {
  using namespace exec;

  flog::maybe_flush();
  return reduce_internal<Task, Reduction, Attributes, Args...>(
    std::forward<Args>(args)...);
} // reduce

/*!
  Execute a test task. This interface is provided for FleCSI's unit testing
  framework. Test tasks must return an integer that is non-zero on failure,
  and zero otherwise.

  \see \c execute about parameter and argument types.

  @return zero on success, non-zero on failure.

  \ns.
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
  explicit guard(trace & t) : t(t), current_flog_task_count(flog::unflush()) {
    t.start();
  }

  // Destroy a guard by stopping the tracing.
  // The Flog count is merged and triggered if needed.
  ~guard() {
    t.stop();
    flog::maybe_flush(current_flog_task_count);
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

namespace topo {
void
repartition::reduce_rsz_required() {
  // now that reduce has been defined
  rsz_required = reduce<resize_required, exec::fold::max>(sizes());
}
} // namespace topo

} // namespace flecsi

#endif
