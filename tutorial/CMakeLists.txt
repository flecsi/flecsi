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

cmake_minimum_required(VERSION 3.12...3.19)

#------------------------------------------------------------------------------#
# Project.
#------------------------------------------------------------------------------#

project(FleCSI-Tutorial LANGUAGES CXX C)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

#------------------------------------------------------------------------------#
# Compiler version requirements.
#------------------------------------------------------------------------------#

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "Version 9.0 of gnu compilers required!")
  endif()
endif()

#------------------------------------------------------------------------------#
# Auxiliary Tools
#------------------------------------------------------------------------------#

option(ENABLE_DEVELOPER_MODE "Enable devel targets" OFF)

if(ENABLE_DEVELOPER_MODE)
  find_program(DOT_EXECUTABLE dot)
  if(DOT_EXECUTABLE-NOTFOUND)
    message(FATAL_ERROR "Failed to find dot executable")
  endif()

  find_package(ImageMagick COMPONENTS convert REQUIRED)
endif()

#------------------------------------------------------------------------------#
# Unit tests
#------------------------------------------------------------------------------#

option(ENABLE_UNIT_TESTS "Enable unit tests" OFF)
mark_as_advanced(ENABLE_UNIT_TESTS)

if(ENABLE_UNIT_TESTS)
  enable_testing()
endif()

function(add_unit name)
  set(options)
  set(one_value_args)
  set(multi_value_args ARGUMENTS)

  cmake_parse_arguments(unit "${options}" "${one_value_args}"
    "${multi_value_args}" ${ARGN})

  add_test(NAME ${name}
    COMMAND
      $<TARGET_FILE:${name}> ${unit_ARGUMENTS}
    WORKING_DIRECTORY
      ${CMAKE_BINARY_DIR}
  )
endfunction()

#------------------------------------------------------------------------------#
# FleCSI
#------------------------------------------------------------------------------#

find_package(FleCSI REQUIRED)

#------------------------------------------------------------------------------#
# FleCSI-Tutorial library.
#------------------------------------------------------------------------------#

add_library(FleCSI-Tutorial
  ${CMAKE_SOURCE_DIR}/specialization/dummy/dummy.cc
)

include_directories(${PROJECT_SOURCE_DIR})

#------------------------------------------------------------------------------#
# 1-runtime
#------------------------------------------------------------------------------#

add_executable(runtime-minimal
  1-runtime/1-minimal.cc
)
target_link_libraries(runtime-minimal FleCSI::FleCSI)
add_unit(runtime-minimal)

add_executable(runtime-program_options
  1-runtime/2-program_options.cc
)
target_link_libraries(runtime-program_options FleCSI::FleCSI)
file(TOUCH ${CMAKE_BINARY_DIR}/input.txt)
add_unit(runtime-program_options
  ARGUMENTS
    input.txt
)

add_executable(runtime-flog
  1-runtime/3-flog.cc
)
target_link_libraries(runtime-flog FleCSI::FleCSI)
add_unit(runtime-flog)

add_executable(runtime-caliper
  1-runtime/4-caliper.cc)
target_link_libraries(runtime-caliper FleCSI::FleCSI)
add_unit(runtime-caliper)

#------------------------------------------------------------------------------#
# 2-control
#------------------------------------------------------------------------------#

add_executable(control-simple
  2-control/1-simple.cc
  2-control/1-simple.hh
)
target_link_libraries(control-simple FleCSI::FleCSI)
add_unit(control-simple)

add_executable(control-cycle
  2-control/2-cycle.cc
  2-control/2-cycle.hh
)
target_link_libraries(control-cycle FleCSI::FleCSI)
add_unit(control-cycle)

add_executable(control-subcycle
  2-control/2-cycle.cc
  2-control/2-subcycle.hh
)
target_compile_definitions(control-subcycle PUBLIC SUBCYCLE)
target_link_libraries(control-subcycle FleCSI::FleCSI)
add_unit(control-subcycle)

add_executable(control-dependencies
  2-control/3-dependencies.cc
  2-control/3-actions.hh
  2-control/3-dependencies.hh
)
target_link_libraries(control-dependencies FleCSI::FleCSI)
add_unit(control-dependencies)

add_executable(control-state
  2-control/4-state.cc
  2-control/4-state.hh
)
target_link_libraries(control-state FleCSI::FleCSI)
add_unit(control-state)

if(ENABLE_DEVELOPER_MODE)
  add_custom_target(pngs
    ${CMAKE_BINARY_DIR}/control-simple --control-model &&
    ${DOT_EXECUTABLE} -Teps control-simple-control-model.dot > simple.eps &&
    ${IMAGEMAGICK_CONVERT_EXECUTABLE} -density 600 simple.eps simple.png &&
    ${CMAKE_BINARY_DIR}/control-cycle --control-model &&
    ${DOT_EXECUTABLE} -Teps control-cycle-control-model.dot > cycle.eps &&
    ${IMAGEMAGICK_CONVERT_EXECUTABLE} -density 600 cycle.eps cycle.png &&
    ${CMAKE_BINARY_DIR}/control-subcycle --control-model &&
    ${DOT_EXECUTABLE} -Teps control-subcycle-control-model.dot > subcycle.eps &&
    ${IMAGEMAGICK_CONVERT_EXECUTABLE} -density 600 subcycle.eps subcycle.png &&
    ${CMAKE_BINARY_DIR}/control-dependencies --control-model &&
    ${DOT_EXECUTABLE} -Teps control-dependencies-control-model.dot > dependencies.eps &&
    ${IMAGEMAGICK_CONVERT_EXECUTABLE} -density 600 dependencies.eps dependencies.png
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
target_link_libraries(execution-single-task FleCSI::FleCSI)
add_unit(execution-single-task)

add_executable(execution-index-task
  3-execution/2-index-task.cc
  3-execution/control.hh
  3-execution/main.cc
)
target_link_libraries(execution-index-task FleCSI::FleCSI)
add_unit(execution-index-task)

add_executable(execution-mpi-task
  3-execution/3-mpi-task.cc
  3-execution/control.hh
  3-execution/main.cc
)
target_link_libraries(execution-mpi-task FleCSI::FleCSI)
add_unit(execution-mpi-task)

#------------------------------------------------------------------------------#
# 4-data
#------------------------------------------------------------------------------#

add_executable(data-global
  4-data/1-global.cc
  4-data/main.cc
  4-data/control.hh
)
target_link_libraries(data-global FleCSI::FleCSI)
add_unit(data-global)

add_executable(data-index
  4-data/2-index.cc
  4-data/main.cc
  4-data/control.hh
)
target_link_libraries(data-index FleCSI::FleCSI)
add_unit(data-index)

add_executable(data-dense
  4-data/3-dense.cc
  4-data/main.cc
  4-data/canonical.hh
  4-data/control.hh
)
target_link_libraries(data-dense FleCSI::FleCSI)
add_unit(data-dense)

#------------------------------------------------------------------------------#
# 5-parallel
#------------------------------------------------------------------------------#

add_executable(parallel-forall
  5-parallel/1-forall.cc
  5-parallel/main.cc
  5-parallel/control.hh
  4-data/canonical.hh
)
target_link_libraries(parallel-forall ${FleCSI_LIBRARIES}
  ${FleCSI_LIBRARY_DEPENDENCIES})
add_unit(parallel-forall)
