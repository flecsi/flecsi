#------------------------------------------------------------------------------#
#  @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
# /@@/////  /@@          @@////@@ @@////// /@@
# /@@       /@@  @@@@@  @@    // /@@       /@@
# /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
# /@@////   /@@/@@@@@@@/@@       ////////@@/@@
# /@@       /@@/@@//// //@@    @@       /@@/@@
# /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
# //       ///  //////   //////  ////////  //
#
# Copyright (c) 2016, Triad National Security, LLC
# All rights reserved
#------------------------------------------------------------------------------#

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

add_unit(index
  SOURCES
    test/index.cc
  THREADS 2
  )

# unstructured ---------------------------------------------------------------#

add_unit(coloring
  SOURCES
    unstructured/test/coloring.cc
    unstructured/test/simple_definition.hh
  INPUTS
    unstructured/test/simple2d-16x16.msh
    unstructured/test/simple2d-8x8.msh
  THREADS 5
)

add_unit(closure
  SOURCES
    unstructured/test/closure.cc
    unstructured/test/simple_definition.hh
    unstructured/test/ugm_definition.hh
  INPUTS
    unstructured/test/simple2d-16x16.msh
    unstructured/test/simple2d-8x8.msh
    unstructured/test/bunny.ugm
  THREADS 2
)

set(UTHREADS 4)
if(FLECSI_BACKEND STREQUAL "legion")
  set(UTHREADS 1 2 4 6)
endif()

add_unit(unstructured
  SOURCES
    unstructured/test/simple_definition.hh
    unstructured/test/unstructured.cc
  INPUTS
    unstructured/test/simple2d-8x8.msh
  THREADS ${UTHREADS}
)

add_unit(fixed
  SOURCES
    unstructured/test/fixed.cc
  THREADS 4
)

# narray ----------------------------------------------------------------------#

add_unit(narray
  SOURCES
    narray/test/narray.cc
    narray/test/narray.hh
    narray/test/narray_tasks.hh
  THREADS 4
)

# ntree -----------------------------------------------------------------------#

if(FLECSI_BACKEND STREQUAL "legion")
  add_unit(ntree_policy
    SOURCES ntree/test/ntree.cc
    INPUTS ntree/test/coordinates.blessed
    THREADS 4
  )
endif()

add_unit(ntree_geometry
  SOURCES ntree/test/geometry.cc
)
