#------------------------------------------------------------------------------#
# Add header files. Note that these will be "exported" to the parent
# scope below.
#------------------------------------------------------------------------------#

set(topo_HEADERS
  color.hh
  core.hh
  global.hh
  index.hh
  size.hh
  narray/coloring_utils.hh
  narray/interface.hh
  narray/types.hh
  ntree/coloring.hh
  ntree/geometry.hh
  ntree/interface.hh
  ntree/types.hh
  set/interface.hh
  unstructured/coloring_functors.hh
  unstructured/coloring_utils.hh
  unstructured/interface.hh
  unstructured/types.hh
  types.hh
)

set(topo_SOURCES
)

#------------------------------------------------------------------------------#
# Export header list to parent scope.
#------------------------------------------------------------------------------#

set(topo_HEADERS
  ${topo_HEADERS}
  PARENT_SCOPE
)

#------------------------------------------------------------------------------#
# Export source list to parent scope.
#------------------------------------------------------------------------------#

set(topo_SOURCES
  ${topo_SOURCES}
  PARENT_SCOPE
)

#------------------------------------------------------------------------------#
# Unit tests.
#------------------------------------------------------------------------------#

# index -----------------------------------------------------------------------#

flecsi_add_test(index
  SOURCES
    test/index.cc
  PROCS 2
  )

# unstructured ---------------------------------------------------------------#

flecsi_add_test(coloring
  SOURCES
    unstructured/test/coloring.cc
    unstructured/test/simple_definition.hh
  INPUTS
    unstructured/test/simple2d-16x16.msh
    unstructured/test/coloring_5.0.blessed
    unstructured/test/coloring_5.1.blessed
    unstructured/test/coloring_5.2.blessed
    unstructured/test/coloring_5.3.blessed
    unstructured/test/coloring_5.4.blessed
    unstructured/test/coloring_2.0.blessed
    unstructured/test/coloring_2.1.blessed
  PROCS 5
)

set(UPROCS 4)
if(FLECSI_BACKEND STREQUAL "legion")
  set(UPROCS 1 2 4 6)
endif()

flecsi_add_test(unstructured
  SOURCES
    unstructured/test/simple_definition.hh
    unstructured/test/unstructured.hh
    unstructured/test/unstructured.cc
  INPUTS
    unstructured/test/simple2d-3x3.msh
    unstructured/test/simple2d-8x8.msh
    unstructured/test/single-cell.msh
    unstructured/test/two-cell.msh
    unstructured/test/disconnected.msh
    unstructured/test/unstructured_disconnected_entities_1_0.blessed
    unstructured/test/unstructured_disconnected_entities_1_1.blessed
    unstructured/test/unstructured_disconnected_entities_1_2.blessed
    unstructured/test/unstructured_disconnected_entities_1_3.blessed
    unstructured/test/unstructured_disconnected_entities_2_0.blessed
    unstructured/test/unstructured_disconnected_entities_2_1.blessed
    unstructured/test/unstructured_disconnected_entities_2_2.blessed
    unstructured/test/unstructured_disconnected_entities_2_3.blessed
    unstructured/test/unstructured_disconnected_entities_4_0.blessed
    unstructured/test/unstructured_disconnected_entities_4_1.blessed
    unstructured/test/unstructured_disconnected_entities_4_2.blessed
    unstructured/test/unstructured_disconnected_entities_4_3.blessed
    unstructured/test/unstructured_disconnected_entities_6_0.blessed
    unstructured/test/unstructured_disconnected_entities_6_1.blessed
    unstructured/test/unstructured_disconnected_entities_6_2.blessed
    unstructured/test/unstructured_disconnected_entities_6_3.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_1_0.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_1_1.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_1_2.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_1_3.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_2_0.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_2_1.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_2_2.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_2_3.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_4_0.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_4_1.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_4_2.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_4_3.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_6_0.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_6_1.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_6_2.blessed
    unstructured/test/unstructured_simple2d-8x8_entities_6_3.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_1_0.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_1_1.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_1_2.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_1_3.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_2_0.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_2_1.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_2_2.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_2_3.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_4_0.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_4_1.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_4_2.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_4_3.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_6_0.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_6_1.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_6_2.blessed
    unstructured/test/unstructured_simple2d-3x3_entities_6_3.blessed
    unstructured/test/unstructured_single-cell_entities_1_0.blessed
    unstructured/test/unstructured_single-cell_entities_1_1.blessed
    unstructured/test/unstructured_single-cell_entities_1_2.blessed
    unstructured/test/unstructured_single-cell_entities_1_3.blessed
    unstructured/test/unstructured_single-cell_entities_2_0.blessed
    unstructured/test/unstructured_single-cell_entities_2_1.blessed
    unstructured/test/unstructured_single-cell_entities_2_2.blessed
    unstructured/test/unstructured_single-cell_entities_2_3.blessed
    unstructured/test/unstructured_single-cell_entities_4_0.blessed
    unstructured/test/unstructured_single-cell_entities_4_1.blessed
    unstructured/test/unstructured_single-cell_entities_4_2.blessed
    unstructured/test/unstructured_single-cell_entities_4_3.blessed
    unstructured/test/unstructured_single-cell_entities_6_0.blessed
    unstructured/test/unstructured_single-cell_entities_6_1.blessed
    unstructured/test/unstructured_single-cell_entities_6_2.blessed
    unstructured/test/unstructured_single-cell_entities_6_3.blessed
    unstructured/test/unstructured_two-cell_entities_1_0.blessed
    unstructured/test/unstructured_two-cell_entities_1_1.blessed
    unstructured/test/unstructured_two-cell_entities_1_2.blessed
    unstructured/test/unstructured_two-cell_entities_1_3.blessed
    unstructured/test/unstructured_two-cell_entities_2_0.blessed
    unstructured/test/unstructured_two-cell_entities_2_1.blessed
    unstructured/test/unstructured_two-cell_entities_2_2.blessed
    unstructured/test/unstructured_two-cell_entities_2_3.blessed
    unstructured/test/unstructured_two-cell_entities_4_0.blessed
    unstructured/test/unstructured_two-cell_entities_4_1.blessed
    unstructured/test/unstructured_two-cell_entities_4_2.blessed
    unstructured/test/unstructured_two-cell_entities_4_3.blessed
    unstructured/test/unstructured_two-cell_entities_6_0.blessed
    unstructured/test/unstructured_two-cell_entities_6_1.blessed
    unstructured/test/unstructured_two-cell_entities_6_2.blessed
    unstructured/test/unstructured_two-cell_entities_6_3.blessed
  PROCS ${UPROCS}
)

flecsi_add_test(fixed
  SOURCES
    unstructured/test/fixed.cc
  INPUTS
    unstructured/test/fixed_0.blessed
    unstructured/test/fixed_1.blessed
    unstructured/test/fixed_2.blessed
    unstructured/test/fixed_3.blessed
  PROCS 4
)

flecsi_add_test(set
  SOURCES
    unstructured/test/unstructured.hh
    set/test/mpm_set.cc
  INPUTS
    unstructured/test/simple2d-16x16.msh
  PROCS 4
)

# narray ----------------------------------------------------------------------#

flecsi_add_test(narray
  SOURCES
    narray/test/narray.cc
    narray/test/narray.hh
  PROCS 4
)

# ntree -----------------------------------------------------------------------#

if(FLECSI_BACKEND STREQUAL "legion")
  flecsi_add_test(ntree_policy
    SOURCES ntree/test/ntree.cc
    INPUTS ntree/test/coordinates.blessed
    PROCS 4
  )
endif()

flecsi_add_test(ntree_geometry
  SOURCES ntree/test/geometry.cc
)
