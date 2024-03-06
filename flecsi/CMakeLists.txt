function(flecsi_headers)
  target_sources(FleCSI PUBLIC FILE_SET public_headers TYPE HEADERS FILES ${ARGN})
endfunction()

function(flecsi_sources)
  target_sources(FleCSI PRIVATE ${ARGN})
endfunction()

flecsi_headers(
  data.hh
  execution.hh
  flog.hh
)

add_subdirectory(data)
add_subdirectory(flog)
add_subdirectory(exec)
add_subdirectory(run)
add_subdirectory(topo)
add_subdirectory(util)

if(ENABLE_HDF5)
  flecsi_headers(io.hh)
  add_subdirectory(io)
endif()
