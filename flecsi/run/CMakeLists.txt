flecsi_headers(
  backend.hh
  context.hh
  control.hh
  options.hh
  types.hh
)
flecsi_sources(context.cc)

if(FLECSI_BACKEND STREQUAL "legion")
  flecsi_headers(
    leg/context.hh
    leg/mapper.hh
  )
  flecsi_sources(leg/context.cc)
elseif(FLECSI_BACKEND STREQUAL "mpi")
  flecsi_headers(
    mpi/context.hh
  )
  flecsi_sources(mpi/context.cc)
endif()

if(ENABLE_UNIT_TESTS)
  add_executable(cycle
    test/cycle.cc
    test/cycle.hh)
  target_link_libraries(cycle FleCSI)
  add_dependencies(${FLECSI_UNIT_TESTS_TARGET} cycle)
  _flecsi_get_unit_test_backend_flags(CYCLE_FLAGS)
  _flecsi_add_mpi_test(cycle TARGET cycle ARGUMENTS ${CYCLE_FLAGS})
endif()

flecsi_add_test(program-options
  SOURCES
    test/program-options.cc
)
