flecsi_headers(
  accessor.hh
  backend.hh
  coloring.hh
  copy.hh
  field.hh
  field_info.hh
  layout.hh
  map.hh
  privilege.hh
  topology_accessor.hh
  topology_slot.hh
  topology.hh
  copy_plan.hh
)

if(FLECSI_BACKEND STREQUAL "legion")
  flecsi_headers(
    leg/copy.hh
    leg/policy.hh
  )
elseif(FLECSI_BACKEND STREQUAL "mpi")
  flecsi_headers(
    mpi/copy.hh
    mpi/policy.hh
  )
elseif(FLECSI_BACKEND STREQUAL "hpx")
endif()

flecsi_add_test(data
  SOURCES
    test/data.cc
  PROCS 2
)
