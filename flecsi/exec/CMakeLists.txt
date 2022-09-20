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

flecsi_add_test(kernel
    SOURCES
      test/kernel.cc
)

flecsi_add_test(task
  SOURCES
    test/task.cc
  PROCS
    4 # Warning: This number is hard-coded into the test.
)

flecsi_add_test(future
  SOURCES
    test/future.cc
  PROCS 2
)
