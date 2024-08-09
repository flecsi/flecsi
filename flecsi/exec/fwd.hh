// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

// Forward declarations for task execution, for use by templated code on which
// the definition of reduce depends.  Other clients must include execution.hh.

#ifndef FLECSI_EXEC_FWD_HH
#define FLECSI_EXEC_FWD_HH

#include "flecsi/exec/task_attributes.hh"

namespace flecsi {
/// \addtogroup execution
/// \{

/*!
  Execute a reduction task.

  @tparam Reduction  The reduction operation type.
  \return a \ref future providing the reduced return value

  \see \c execute about parameter and argument types.
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
 */

template<auto & TASK,
  TaskAttributes ATTRIBUTES = flecsi::loc | flecsi::leaf,
  typename... ARGS>
auto
execute(ARGS &&... args) {
  return reduce<TASK, void, ATTRIBUTES>(std::forward<ARGS>(args)...);
} // execute

namespace exec {
// Learn from backend whether a trace is supported, active, and not skipped.
inline bool is_tracing();
} // namespace exec

/// \}
} // namespace flecsi

#endif
