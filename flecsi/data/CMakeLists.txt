#------------------------------------------------------------------------------#
# Add header files. Note that these will be "exported" to the parent
# scope below.
#------------------------------------------------------------------------------#

set(data_HEADERS
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

set(data_SOURCES
)

if(FLECSI_BACKEND STREQUAL "legion")

  set(data_HEADERS
    leg/copy.hh
    leg/policy.hh
    ${data_HEADERS}
  )

  set(data_SOURCES
    ${data_SOURCES}
  )

elseif(FLECSI_BACKEND STREQUAL "mpi")

  set(data_HEADERS
    local/copy.hh
    local/policy.hh
    local/storage.hh
    mpi/copy.hh
    mpi/policy.hh
    ${data_HEADERS}
  )

  set(data_SOURCES
    local/copy.cc
    ${data_SOURCES}
  )

elseif(FLECSI_BACKEND STREQUAL "hpx")

  set(data_HEADERS ${data_HEADERS}
    local/copy.hh
    local/policy.hh
    local/storage.hh
    hpx/copy.hh
    hpx/policy.hh
  )

  set(data_SOURCES
    local/copy.cc
    ${data_SOURCES}
  )

endif()

#------------------------------------------------------------------------------#
# Export header list to parent scope.
#------------------------------------------------------------------------------#

set(data_HEADERS
  ${data_HEADERS}
  PARENT_SCOPE
)

#------------------------------------------------------------------------------#
# Export source list to parent scope.
#------------------------------------------------------------------------------#

set(data_SOURCES
  ${data_SOURCES}
  PARENT_SCOPE
)
