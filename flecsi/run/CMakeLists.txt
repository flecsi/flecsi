set(run_HEADERS
  backend.hh
  context.hh
  control.hh
  types.hh
)

set(run_SOURCES
  context.cc
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
# Note: The cycle test does not use the 'flecsi_add_test' interface because it 
# is meant to test different low-level control model features.
#------------------------------------------------------------------------------#

if(ENABLE_UNIT_TESTS)
  add_executable(cycle
    test/cycle.cc
    test/cycle.hh
    test/package_a.hh
    test/package_b.hh
    test/package_c.hh)
  target_link_libraries(cycle FleCSI)
  target_include_directories(cycle PRIVATE ${CMAKE_BINARY_DIR})
  add_dependencies(${FLECSI_UNIT_TESTS_TARGET} cycle)
endif()

flecsi_add_test(flog
  SOURCES
    test/flog.cc
)

flecsi_add_test(program-options
  SOURCES
    test/program-options.cc
)
