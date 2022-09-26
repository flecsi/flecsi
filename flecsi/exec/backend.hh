// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_BACKEND_HH
#define FLECSI_EXEC_BACKEND_HH

#include "flecsi/exec/task_attributes.hh"
#include <flecsi-config.h>

#include <cstddef>

namespace flecsi {
/// \addtogroup execution
/// \{

/*!
  Execute a task.

  @tparam TASK          The user task.
    Its parameters must support \ref serial.
    If \a ATTRIBUTES specifies an MPI task, parameters need merely be movable.
  @tparam ATTRIBUTES    The task attributes mask.
  @tparam ARGS The user-specified task arguments, implicitly converted to the
    parameter types for \a TASK.
    Certain FleCSI-defined parameter types accept particular, different
    argument types that serve as selectors for information stored by the
    backend; each type involved documents the correspondence.
  \return a \ref future providing the value(s) returned from the task

  \note
    Avoid
    passing large objects to tasks repeatedly; use global variables (and,
    perhaps, pass keys to select from them) or fields.
 */

template<auto & TASK,
  TaskAttributes ATTRIBUTES = flecsi::loc | flecsi::leaf,
  typename... ARGS>
auto execute(ARGS &&...);
/// \}
} // namespace flecsi

//----------------------------------------------------------------------------//
// This section works with the build system to select the correct backend
// implementation for the task model.
//----------------------------------------------------------------------------//

#if FLECSI_BACKEND == FLECSI_BACKEND_legion

#include "flecsi/exec/leg/policy.hh"

#elif FLECSI_BACKEND == FLECSI_BACKEND_mpi

#include "flecsi/exec/mpi/policy.hh"

#elif FLECSI_BACKEND == FLECSI_BACKEND_hpx

#include "flecsi/exec/hpx/policy.hh"

#endif // FLECSI_BACKEND

#endif
