flecsi_headers(
  backend.hh
  bind_parameters.hh
  buffers.hh
  fold.hh
  fwd.hh
  kernel.hh
  launch.hh
  prolog.hh
  task_attributes.hh
  future.hh
)

if(FLECSI_BACKEND STREQUAL "legion")
  flecsi_headers(
    leg/bind_accessors.hh
    leg/task_wrapper.hh
    leg/policy.hh
    leg/future.hh
    leg/task_prologue.hh
    leg/reduction_wrapper.hh
    leg/tracer.hh
    bind_parameters.hh
  )
elseif(FLECSI_BACKEND STREQUAL "mpi")
  flecsi_headers(
    local/params.hh
    mpi/future.hh
    mpi/params.hh
    mpi/policy.hh
    mpi/reduction_wrapper.hh
    tracer.hh
  )
elseif(FLECSI_BACKEND STREQUAL "hpx")
  flecsi_headers(
    local/params.hh
    hpx/bind_accessors.hh
    hpx/future.hh
    hpx/policy.hh
    hpx/reduction_wrapper.hh
    hpx/task_prologue.hh
    bind_parameters.hh
    tracer.hh
  )
endif()

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
