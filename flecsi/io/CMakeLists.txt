flecsi_headers(
  backend.hh
  hdf5.hh
)

if(FLECSI_BACKEND STREQUAL "legion")
  flecsi_headers(leg/policy.hh)
elseif(FLECSI_BACKEND STREQUAL "mpi")
  flecsi_headers(mpi/policy.hh)
elseif(FLECSI_BACKEND STREQUAL "hpx")
endif()

flecsi_add_test(hdf5_restart
  SOURCES
    test/hdf5_restart.cc
  PROCS 4
)

if(FLECSI_BACKEND STREQUAL "legion")
  flecsi_add_test(io_metadata
    SOURCES
      test/io_metadata.cc
    PROCS 2
  )
endif()
