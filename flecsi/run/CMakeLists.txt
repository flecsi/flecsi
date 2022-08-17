set(run_HEADERS
  backend.hh
  context.hh
  control.hh
  point_walker.hh
)

set(run_SOURCES
)

if(FLECSI_BACKEND STREQUAL "legion")

  set(run_HEADERS
    ${run_HEADERS}
    leg/context.hh
    leg/mapper.hh
  )

  set(run_SOURCES
    ${run_SOURCES}
    leg/context.cc
  )

elseif(FLECSI_BACKEND STREQUAL "mpi")

  set(run_HEADERS
    ${run_HEADERS}
    mpi/context.hh
  )

  set(run_SOURCES
    ${run_SOURCES}
    mpi/context.cc
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

add_unit(flog
  SOURCES
    test/flog.cc
)

add_unit(program-options
  SOURCES
    test/program-options.cc
)
