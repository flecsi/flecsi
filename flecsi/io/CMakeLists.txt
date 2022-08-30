if(ENABLE_HDF5)

set(io_HEADERS
  backend.hh
  hdf5.hh
)

set(io_SOURCES
)

if(FLECSI_BACKEND STREQUAL "legion")

  set(io_HEADERS
    leg/policy.hh
    ${io_HEADERS}
  )

elseif(FLECSI_BACKEND STREQUAL "mpi")

  set(io_HEADERS
    ${io_HEADERS}
  )

  set(io_SOURCES
    ${io_SOURCES}
  )

elseif(FLECSI_BACKEND STREQUAL "hpx")

  set(io_HEADERS ${io_HEADERS}
  )

  set(io_SOURCES
    ${io_SOURCES}
  )

endif()

#------------------------------------------------------------------------------#
# Export header list to parent scope.
#------------------------------------------------------------------------------#

set(io_HEADERS
  ${io_HEADERS}
  PARENT_SCOPE
)

#------------------------------------------------------------------------------#
# Export source list to parent scope.
#------------------------------------------------------------------------------#

set(io_SOURCES
  ${io_SOURCES}
  PARENT_SCOPE
)

#------------------------------------------------------------------------------#
# Unit tests.
#------------------------------------------------------------------------------#

if(FLECSI_BACKEND STREQUAL "legion")

flecsi_add_test(hdf5_restart
  SOURCES
    test/hdf5_restart.cc
  PROCS 4
)

flecsi_add_test(io_metadata
  SOURCES
    test/io_metadata.cc
  PROCS 2
)
endif()
                
endif()
