flecsi_headers(
  backend.hh
  context.hh
  control.hh
  init.hh
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
  # This file exercises the old interface for main.
  set_source_files_properties(test/cycle.cc
    PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  target_link_libraries(cycle FleCSI)
  add_dependencies(${FLECSI_UNIT_TESTS_TARGET} cycle)
  flecsi_add_target_test(cycle)
endif()

flecsi_add_test(program-options
  SOURCES
    test/program-options.cc
)
