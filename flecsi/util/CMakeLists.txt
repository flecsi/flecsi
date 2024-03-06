flecsi_headers(
  annotation.hh
  array_ref.hh
  bitutils.hh
  color_map.hh
  common.hh
  constant.hh
  dag.hh
  demangle.hh
  dimensioned_array.hh
  function_traits.hh
  geometry/filling_curve.hh
  geometry/point.hh
  geometry/kdtree.hh
  crs.hh
  parmetis.hh
  hashtable.hh
  mpi.hh
  reorder.hh
  serialize.hh
  set_intersection.hh
  set_utils.hh
  target.hh
  type_traits.hh
  types.hh
)
flecsi_sources(demangle.cc)

if(ENABLE_FLOG)
  flecsi_headers(
    unit.hh
    unit/output.hh
    unit/types.hh
  )
endif()

if(ENABLE_GRAPHVIZ)
  flecsi_headers(graphviz.hh)
endif()

################################################################################
# Unit tests.
################################################################################

#------------------------------------------------------------------------------#
# array_ref
#------------------------------------------------------------------------------#

set(array_ref_blessed_input test/array_ref.blessed.gnug)

if(MSVC)
  set(array_ref_blessed_input test/array_ref.blessed.msvc)
endif()

flecsi_add_test(array_ref
  SOURCES
    test/array_ref.cc
  INPUTS
    test/array_ref.blessed
    ${array_ref_blessed_input}
)

#------------------------------------------------------------------------------#
# common
#------------------------------------------------------------------------------#

set(common_blessed_input test/common.blessed.gnug)

if(MSVC)
  set(common_blessed_input test/common.blessed.msvc)
endif()

flecsi_add_test(common
  SOURCES
    test/common.cc
  INPUTS
    test/common.blessed.ppc
    test/common.blessed
    ${common_blessed_input}
)

#------------------------------------------------------------------------------#
# color_map
#------------------------------------------------------------------------------#

flecsi_add_test(color_map
  SOURCES
    test/color_map.cc
  PROCS 3
)

#------------------------------------------------------------------------------#
# unit
#------------------------------------------------------------------------------#

flecsi_add_test(unit
  SOURCES
    test/unit.cc
  PROCS
    1 2 4
  ARGUMENTS "--flog-process=-1"
)

#------------------------------------------------------------------------------#
# serialize
#------------------------------------------------------------------------------#

flecsi_add_test(serialize
  SOURCES
    test/serialize.cc
)

#------------------------------------------------------------------------------#
# geometry
#------------------------------------------------------------------------------#

flecsi_add_test(geometry
  SOURCES test/geometry.cc
)

#------------------------------------------------------------------------------#
# hashtable
#------------------------------------------------------------------------------#

flecsi_add_test(hashtable
  SOURCES
    test/hashtable.cc
)

#------------------------------------------------------------------------------#
# annotation
#------------------------------------------------------------------------------#
# This test is currently unreliable.
if(NOT CALIPER_DETAIL STREQUAL "none")
flecsi_add_test(annotation
  SOURCES
  test/annotation.cc
  PROCS
  3 4
)
endif()
