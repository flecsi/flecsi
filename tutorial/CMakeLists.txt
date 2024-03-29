cmake_minimum_required(VERSION 3.12...3.19)

#------------------------------------------------------------------------------#
# Project.
#------------------------------------------------------------------------------#

project(FleCSI-Tutorial LANGUAGES CXX C)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

#------------------------------------------------------------------------------#
# FleCSI
#------------------------------------------------------------------------------#

find_package(FleCSI REQUIRED)

#------------------------------------------------------------------------------#
# Auxiliary Tools
#------------------------------------------------------------------------------#

option(ENABLE_DEVELOPER_MODE "Enable devel targets" OFF)

if(ENABLE_DEVELOPER_MODE)
  find_package(Graphviz REQUIRED)
  find_package(ImageMagick COMPONENTS convert REQUIRED)
endif()

#------------------------------------------------------------------------------#
# Unit tests
#------------------------------------------------------------------------------#

option(ENABLE_UNIT_TESTS "Enable unit tests" OFF)
mark_as_advanced(ENABLE_UNIT_TESTS)
include(FleCSI/unit)

if(ENABLE_UNIT_TESTS)
  flecsi_enable_testing()
endif()

function(add_unit name)
  target_link_libraries(${name} PRIVATE FleCSI::FleCSI)
  flecsi_add_target_test(${name} ${ARGN})
endfunction()

#------------------------------------------------------------------------------#
# 1-runtime
#------------------------------------------------------------------------------#

add_executable(runtime-minimal
  1-runtime/1-minimal.cc
)
add_unit(runtime-minimal)

add_executable(runtime-program_options
  1-runtime/2-program_options.cc
)
file(TOUCH ${CMAKE_BINARY_DIR}/input.txt)
add_unit(runtime-program_options
  ARGUMENTS
    input.txt
)

add_executable(runtime-flog
  1-runtime/3-flog.cc
)
add_unit(runtime-flog)

add_executable(runtime-caliper
  1-runtime/4-caliper.cc)
add_unit(runtime-caliper)

#------------------------------------------------------------------------------#
# 2-control
#------------------------------------------------------------------------------#

add_executable(control-simple
  2-control/1-simple.cc
  2-control/1-simple.hh
)
add_unit(control-simple)

add_executable(control-cycle
  2-control/2-cycle.cc
  2-control/2-cycle.hh
)
add_unit(control-cycle)

add_executable(control-subcycle
  2-control/2-cycle.cc
  2-control/2-subcycle.hh
)
target_compile_definitions(control-subcycle PUBLIC SUBCYCLE)
add_unit(control-subcycle)

add_executable(control-dependencies
  2-control/3-dependencies.cc
  2-control/3-actions.hh
  2-control/3-dependencies.hh
)
add_unit(control-dependencies)

add_executable(control-state
  2-control/4-state.cc
  2-control/4-state.hh
)
add_unit(control-state)

if(ENABLE_DEVELOPER_MODE)
  add_custom_target(pngs
    $<TARGET_FILE:control-simple> --control-model &&
    $<TARGET_FILE:Graphviz::dot> -Teps control-simple-control-model.dot > simple.eps &&
    ${ImageMagick_convert_EXECUTABLE} -density 600 simple.eps simple.png &&
    $<TARGET_FILE:control-cycle> --control-model &&
    $<TARGET_FILE:Graphviz::dot> -Teps control-cycle-control-model.dot > cycle.eps &&
    ${ImageMagick_convert_EXECUTABLE} -density 600 cycle.eps cycle.png &&
    $<TARGET_FILE:control-subcycle> --control-model &&
    $<TARGET_FILE:Graphviz::dot> -Teps control-subcycle-control-model.dot > subcycle.eps &&
    ${ImageMagick_convert_EXECUTABLE} -density 600 subcycle.eps subcycle.png &&
    $<TARGET_FILE:control-dependencies> --control-model &&
    $<TARGET_FILE:Graphviz::dot> -Teps control-dependencies-control-model.dot > dependencies.eps &&
    ${ImageMagick_convert_EXECUTABLE} -density 600 dependencies.eps dependencies.png
    DEPENDS
      control-simple
      control-cycle
      control-subcycle
      control-dependencies
  )
endif()

#------------------------------------------------------------------------------#
# 3-execution
#------------------------------------------------------------------------------#

add_executable(execution-single-task
  3-execution/1-single-task.cc
  3-execution/control.hh
  3-execution/main.cc
)
add_unit(execution-single-task)

add_executable(execution-index-task
  3-execution/2-index-task.cc
  3-execution/control.hh
  3-execution/main.cc
)
add_unit(execution-index-task)

add_executable(execution-mpi-task
  3-execution/3-mpi-task.cc
  3-execution/control.hh
  3-execution/main.cc
)
add_unit(execution-mpi-task)

#------------------------------------------------------------------------------#
# 4-data
#------------------------------------------------------------------------------#

add_executable(data-global
  4-data/1-global.cc
  3-execution/main.cc
  3-execution/control.hh
)
add_unit(data-global)

add_executable(data-index
  4-data/2-index.cc
  3-execution/main.cc
  3-execution/control.hh
)
add_unit(data-index)

add_executable(data-dense
  4-data/3-dense.cc
  3-execution/main.cc
  4-data/canonical.hh
  3-execution/control.hh
)
add_unit(data-dense)

#------------------------------------------------------------------------------#
# 5-parallel
#------------------------------------------------------------------------------#

add_executable(parallel-forall
  5-parallel/1-forall.cc
  3-execution/main.cc
  3-execution/control.hh
  4-data/canonical.hh
)
add_unit(parallel-forall)

add_executable(parallel-reduceall
  5-parallel/2-reduceall.cc
  3-execution/main.cc
  3-execution/control.hh
  4-data/canonical.hh
)
add_unit(parallel-reduceall)

#------------------------------------------------------------------------------#
# 6-ntree
#------------------------------------------------------------------------------#

add_executable(ntree
  6-topology/ntree.cc
  6-topology/sph_physics.cc
  6-topology/sph_physics.hh
  6-topology/ntree_sph.hh
)
add_unit(ntree)

