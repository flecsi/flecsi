#------------------------------------------------------------------------------#
#  @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
# /@@/////  /@@          @@////@@ @@////// /@@
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

set(run_HEADERS
  backend.hh
  context.hh
  control.hh
  point_walker.hh
)

set(run_SOURCES
)

if(FLECSI_RUNTIME_MODEL STREQUAL "legion")

  set(run_HEADERS
    ${run_HEADERS}
    leg/context.hh
    leg/mapper.hh
  )

  set(run_SOURCES
    ${run_SOURCES}
    leg/context.cc
  )

elseif(FLECSI_RUNTIME_MODEL STREQUAL "mpi")

  set(run_HEADERS
    ${run_HEADERS}
    mpi/context.hh
  )

  set(run_SOURCES
    ${run_SOURCES}
    mpi/context.cc
  )

elseif(FLECSI_RUNTIME_MODEL STREQUAL "hpx")

  set(run_HEADERS
    ${run_HEADERS}
    hpx/context.hh
  )

  set(run_SOURCES
    ${run_SOURCES}
    hpx/context.cc
  )

endif()

#------------------------------------------------------------------------------#
# Export header list to parent scope.
#------------------------------------------------------------------------------#

set(run_HEADERS
  ${run_HEADERS}
  PARENT_SCOPE
)

#------------------------------------------------------------------------------#
# Export source list to parent scope.
#------------------------------------------------------------------------------#

set(run_SOURCES
  ${run_SOURCES}
  PARENT_SCOPE
)

#------------------------------------------------------------------------------#
# Unit tests.
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Note: The cycle test does not use the 'add_unit' interface because it is
# meant to test different low-level control model features.
#------------------------------------------------------------------------------#

add_executable(cycle
  test/cycle.cc
  test/cycle_control.hh
  test/package_a.hh
  test/package_b.hh
  test/package_c.hh)
target_link_libraries(cycle FleCSI)
target_include_directories(cycle PRIVATE ${CMAKE_BINARY_DIR})

if(MSVC)
  target_compile_definitions(cycle PRIVATE
    _SCL_SECURE_NO_WARNINGS
    _CRT_SECURE_NO_WARNINGS
    _SCL_SECURE_NO_DEPRECATE
    _CRT_SECURE_NO_DEPRECATE
    _CRT_NONSTDC_NO_WARNINGS
    _HAS_AUTO_PTR_ETC=1
    _SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING
    _SILENCE_CXX17_ALLOCATOR_VOID_DEPRECATION_WARNING
    NOMINMAX)
endif()

add_unit(flog
  SOURCES
    test/flog.cc
)

add_unit(program-options
  SOURCES
    test/program-options.cc
)
