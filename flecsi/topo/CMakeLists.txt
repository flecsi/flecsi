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
  set/types.hh
  unstructured/coloring_functors.hh
  unstructured/coloring_utils.hh
  unstructured/interface.hh
  unstructured/types.hh
  utility_types.hh
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
    unstructured/test/simple2d-8x8.msh
  PROCS 5
)

flecsi_add_test(closure
  SOURCES
    unstructured/test/closure.cc
    unstructured/test/simple_definition.hh
    unstructured/test/ugm_definition.hh
  INPUTS
    unstructured/test/simple2d-16x16.msh
    unstructured/test/simple2d-8x8.msh
    unstructured/test/bunny.ugm
  PROCS 2
)

#flecsi_add_test(closure
#  SOURCES
#    unstructured/test/closure.cc
#    unstructured/test/test_definition.hh
#)

flecsi_add_test(unstructured
  SOURCES
    unstructured/test/simple_definition.hh
    unstructured/test/unstructured.cc
  INPUTS
    unstructured/test/simple2d-8x8.msh
  PROCS 2
)

flecsi_add_test(fixed
  SOURCES
    unstructured/test/fixed.cc
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

flecsi_add_test(ntree_policy
  SOURCES ntree/test/ntree.cc
  INPUTS ntree/test/coordinates.blessed
  PROCS 4
)

flecsi_add_test(ntree_geometry
  SOURCES ntree/test/geometry.cc
)
