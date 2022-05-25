# Copyright (c) 2016, Triad National Security, LLC
# All rights reserved

set(flecsi_HEADERS
  data.hh
  execution.hh
  flog.hh
)

set(flecsi_SOURCES
)

if(ENABLE_HDF5)
  list(APPEND flecsi_HEADERS io.hh)
endif()
