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

#include "flecsi/exec/task_attributes.hh"
#include <flecsi-config.h>

#include <cstddef>

namespace flecsi {
/// \addtogroup execution
/// \{

/*!
  Execute a task.

  @tparam TASK          The user task.
    Its parameters may be of any default-constructible,
    trivially-move-assignable, non-pointer type, any type that supports the
    Legion return-value serialization interface, or any of several standard
    containers of such types.
    If \a ATTRIBUTES specifies an MPI task, parameters need merely be movable.
  @tparam ATTRIBUTES    The task attributes mask.
  @tparam ARGS The user-specified task arguments, implicitly converted to the
    parameter types for \a TASK.
    Certain FleCSI-defined parameter types accept particular, different
    argument types that serve as selectors for information stored by the
    backend; each type involved documents the correspondence.

  \note Additional types may be supported by defining appropriate
    specializations of \c util::serial::traits or \c util::serial::convert.
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
