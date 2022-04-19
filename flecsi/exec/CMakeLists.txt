# Copyright (c) 2016, Triad National Security, LLC
# All rights reserved

#------------------------------------------------------------------------------#
# Add header files. Note that these will be "exported" to the parent
# scope below.
#------------------------------------------------------------------------------#

set(exec_HEADERS
  backend.hh
  buffers.hh
  fold.hh
  kernel.hh
  launch.hh
  prolog.hh
  task_attributes.hh
)

if(FLECSI_BACKEND STREQUAL "legion")

  set(exec_HEADERS
    ${exec_HEADERS}
    leg/bind_accessors.hh
    leg/task_wrapper.hh
    leg/policy.hh
    leg/future.hh
    leg/task_prologue.hh
    leg/reduction_wrapper.hh
  )

elseif(FLECSI_BACKEND STREQUAL "mpi")

  set(exec_HEADERS
    ${exec_HEADERS}
    mpi/future.hh
    mpi/policy.hh
    mpi/reduction_wrapper.hh
    mpi/task_prologue.hh
  )

elseif(FLECSI_BACKEND STREQUAL "hpx")

endif()

#------------------------------------------------------------------------------#
# Export header list to parent scope.
#------------------------------------------------------------------------------#

set(exec_HEADERS
  ${exec_HEADERS}
  PARENT_SCOPE
)

#------------------------------------------------------------------------------#
# Unit tests.
#------------------------------------------------------------------------------#

add_unit(kernel
    SOURCES
      test/kernel.cc
)

add_unit(task
  SOURCES
    test/task.cc
  THREADS
    4 # Warning: This number is hard-coded into the test.
)

add_unit(future
  SOURCES
    test/future.cc
  THREADS 2
)
