#------------------------------------------------------------------------------#
#  @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@ # /@@/////  /@@          @@////@@ @@////// /@@
# /@@       /@@  @@@@@  @@    // /@@       /@@
# /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
# /@@////   /@@/@@@@@@@/@@       ////////@@/@@
# /@@       /@@/@@//// //@@    @@       /@@/@@
# /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
# //       ///  //////   //////  ////////  //
#
# Copyright (c) 2016, Triad National Security, LLC
# All rights reserved
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Add header files. Note that these will be "exported" to the parent
# scope below.
#------------------------------------------------------------------------------#

set(exec_HEADERS
  backend.hh
  buffers.hh
  fold.hh
  kernel_interface.hh
  launch.hh
  prolog.hh
  task_attributes.hh
)

if(FLECSI_RUNTIME_MODEL STREQUAL "legion")

  set(exec_HEADERS
    ${exec_HEADERS}
    leg/bind_accessors.hh
    leg/task_wrapper.hh
    leg/policy.hh
    leg/future.hh
    leg/task_prologue.hh
    leg/reduction_wrapper.hh
  )

elseif(FLECSI_RUNTIME_MODEL STREQUAL "mpi")

  set(exec_HEADERS
    ${exec_HEADERS}
    mpi/future.hh
    mpi/policy.hh
    mpi/reduction_wrapper.hh
    mpi/task_prologue.hh
  )

elseif(FLECSI_RUNTIME_MODEL STREQUAL "hpx")

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

if(NOT FLECSI_RUNTIME_MODEL STREQUAL "mpi") # this test uses oversubscription

add_unit(interface
  SOURCES
    test/interface.cc
  THREADS
    2
)

endif()

if(FLECSI_RUNTIME_MODEL STREQUAL "legion")
if(ENABLE_KOKKOS)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  add_unit(kokkos
    SOURCES
      test/kokkos.cc
    THREADS
     2 #warning 2 is required when running with CUDA since color topology has 2 colors
  )
endif()
endif()
endif()

add_unit(task
  SOURCES
    test/task.cc
  THREADS
    4 # Warning: This number is hard-coded into the the test.
)

add_unit(future
  SOURCES
    test/future.cc
  THREADS 2
  )
