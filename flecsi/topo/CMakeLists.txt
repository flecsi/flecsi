flecsi_headers(
  color.hh
  core.hh
  global.hh
  index.hh
  size.hh
  narray/interface.hh
  narray/types.hh
  set/interface.hh
  unstructured/interface.hh
  unstructured/types.hh
  types.hh
)

if(FLECSI_BACKEND STREQUAL "legion")
  flecsi_headers(
    ntree/coloring.hh
    ntree/interface.hh
    ntree/types.hh
  )
endif()  

# index -----------------------------------------------------------------------#

flecsi_add_test(index
  SOURCES
    test/index.cc
  PROCS 2
)

# unstructured ---------------------------------------------------------------#

flecsi_add_test(fixed
  SOURCES
    unstructured/test/fixed.cc
  INPUTS
    unstructured/test/fixed_0.blessed
    unstructured/test/fixed_1.blessed
    unstructured/test/fixed_2.blessed
    unstructured/test/fixed_3.blessed
    unstructured/test/simple-4x4.fixed.0
    unstructured/test/simple-4x4.fixed.1
    unstructured/test/simple-4x4.fixed.2
    unstructured/test/simple-4x4.fixed.3
  PROCS 4
)

flecsi_add_test(set
  SOURCES
    set/test/mpm_set.cc
  INPUTS
    unstructured/test/simple-4x4.fixed.0
    unstructured/test/simple-4x4.fixed.1
    unstructured/test/simple-4x4.fixed.2
    unstructured/test/simple-4x4.fixed.3
  PROCS 4
)

# narray ----------------------------------------------------------------------#

flecsi_add_test(narray
  SOURCES
    narray/test/narray.cc
  INPUTS
    narray/test/mesh_1d_4_0.blessed
    narray/test/mesh_1d_4_1.blessed
    narray/test/mesh_1d_4_2.blessed
    narray/test/mesh_1d_4_3.blessed
    narray/test/mesh_2d_4_0.blessed
    narray/test/mesh_2d_4_1.blessed
    narray/test/mesh_2d_4_2.blessed
    narray/test/mesh_2d_4_3.blessed
    narray/test/mesh_diag_4_0.blessed
    narray/test/mesh_diag_4_1.blessed
    narray/test/mesh_diag_4_2.blessed
    narray/test/mesh_diag_4_3.blessed
    narray/test/mesh_3d_4_0.blessed
    narray/test/mesh_3d_4_1.blessed
    narray/test/mesh_3d_4_2.blessed
    narray/test/mesh_3d_4_3.blessed
    narray/test/coloring_aux_xy_9x9x9_3x3x1_4_0.blessed
    narray/test/coloring_aux_xy_9x9x9_3x3x1_4_1.blessed
    narray/test/coloring_aux_xy_9x9x9_3x3x1_4_2.blessed
    narray/test/coloring_aux_xy_9x9x9_3x3x1_4_3.blessed
    narray/test/coloring_aux_x_9x9x9_3x3x1_fg_4_0.blessed
    narray/test/coloring_aux_x_9x9x9_3x3x1_fg_4_1.blessed
    narray/test/coloring_aux_x_9x9x9_3x3x1_fg_4_2.blessed
    narray/test/coloring_aux_x_9x9x9_3x3x1_fg_4_3.blessed
    narray/test/coloring_primary_9x9x9_3x3x1_4_0.blessed
    narray/test/coloring_primary_9x9x9_3x3x1_4_1.blessed
    narray/test/coloring_primary_9x9x9_3x3x1_4_2.blessed
    narray/test/coloring_primary_9x9x9_3x3x1_4_3.blessed
  PROCS 4
)

# ntree -----------------------------------------------------------------------#

if(FLECSI_BACKEND STREQUAL "legion")
  flecsi_add_test(ntree
    SOURCES ntree/test/ntree.cc
    INPUTS ntree/test/coordinates.blessed
    PROCS 4
  )
endif()

