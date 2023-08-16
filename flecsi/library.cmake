set(flecsi_HEADERS
  data.hh
  execution.hh
  flog.hh
  runtime.hh
  topology.hh
  utilities.hh
)

set(flecsi_SOURCES
)

if(ENABLE_HDF5)
  list(APPEND flecsi_HEADERS io.hh)
endif()
