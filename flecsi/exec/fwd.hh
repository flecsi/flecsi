// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

// Forward declarations for task execution, for use by templated code on which
// the definition of reduce depends.  Other clients must include execution.hh.

#ifndef FLECSI_EXEC_FWD_HH
#define FLECSI_EXEC_FWD_HH

#include "flecsi/exec/task_attributes.hh"

#include <optional>
#include <utility>

namespace flecsi {
/// \addtogroup execution
/// \{

/*!
  Execute a reduction task.

  @tparam Reduction  The reduction operation type.
  \return a \ref future providing the reduced return value

  \see \c execute about parameter and argument types.

  \ns.
 */
template<auto & Task,
  class Reduction,
  TaskAttributes Attributes = flecsi::loc | flecsi::leaf,
  typename... Args>
[[nodiscard]] auto reduce(Args &&...);

/*!
  Execute a task.

  @tparam TASK          The user task.
    Its parameters must be copyable or a reference to a const, movable type.
    Any that is a pointer must be to a const type or to a function.
    If \a ATTRIBUTES specifies an MPI task, parameters need merely be movable.
  @tparam ATTRIBUTES    The task attributes mask.
  @tparam ARGS The user-specified task arguments, implicitly converted to the
    parameter types for \a TASK.
    Certain FleCSI-defined parameter types accept particular, different
    argument types that serve as selectors for information stored by the
    backend; each type involved documents the correspondence.
    Additionally, a parameter may be a \c std::vector of such a type or a \c
    std::tuple that includes such a type; it accepts a \c std::vector of or a
    \c std::tuple including the corresponding argument type.
  \return a \ref future providing the value(s) returned from the task

  \ns.
 */

template<auto & TASK,
  TaskAttributes ATTRIBUTES = flecsi::loc | flecsi::leaf,
  typename... ARGS>
auto
execute(ARGS &&... args) {
  return reduce<TASK, void, ATTRIBUTES>(std::forward<ARGS>(args)...);
} // execute

struct runtime;

/// Launches tasks according to their execution-space (template) parameters.
/// An instance is passed to control-model actions that accept it.
/// \note MPI tasks cannot use this interface.
struct scheduler {
  explicit scheduler(flecsi::runtime & r) : r(r) {}
  /// Immovable.
  scheduler(scheduler &&) = delete;

  /// Launch a variant of a reduction task.
  template<class, class R, class... AA>
  auto reduce(AA &&...);
  /// Launch a variant of a task.
  /// \tparam V class with a static member \c task that is either a function
  ///   or a function template with a single type template parameter \p S (the
  ///   execution \ref space).  In the latter case, the signatures of the
  ///   specializations for \c exec::cpu, \c exec::gpu, and \c exec::omp may
  ///   vary only in that the type of a parameter may be \a S; an unspecified
  ///   such specialization that is not deleted is launched.
  template<class V, class... AA>
  auto execute(AA &&... aa) {
    return reduce<V, void>(std::forward<AA>(aa)...);
  }
  /// Execute a variant of a test task.
  template<class V, class... AA>
  [[nodiscard]] int test(AA &&... aa);

  /// Launch a reduction task.
  /// \tparam R reduction operation
  /// \return a \ref future providing the reduced return value
  /// \see \c execute about parameter and argument types.
  template<auto &, class R, class... AA>
  auto reduce(AA &&...);
  /// Launch a task.
  template<auto & F, class... AA>
  auto execute(AA &&... aa) {
    return reduce<F, void>(std::forward<AA>(aa)...);
  }
  /// Execute a test task.
  template<auto &, class... AA>
  [[nodiscard]] int test(AA &&...);

  // Will become a non-static member of runtime in 3.
  static std::optional<scheduler> instance;

  /// Get the runtime (which created this scheduler).
  const flecsi::runtime & runtime() const {
    return r;
  }

private:
  flecsi::runtime & r;
};
inline std::optional<scheduler> scheduler::instance;

namespace exec {
// Learn from backend whether a trace is supported, active, and not skipped.
inline bool is_tracing();
} // namespace exec

/// \}
} // namespace flecsi

#endif
