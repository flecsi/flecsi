#------------------------------------------------------------------------------#
# Copyright (c) 2016, Triad National Security, LLC
# All rights reserved
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.13...3.20)

#------------------------------------------------------------------------------#
# Set the project name.
#------------------------------------------------------------------------------#

project(poisson LANGUAGES CXX C)

#------------------------------------------------------------------------------#
# FleCSI currently depends on C++17.
#------------------------------------------------------------------------------#

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

#------------------------------------------------------------------------------#
# Find the FleCSI installation.
#
# The easiest way to help CMake find FleCSI is to install it,
# and add the location to the CMAKE_PREFIX_PATH environment variable.
#------------------------------------------------------------------------------#

find_package(FleCSI REQUIRED)

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

 if(FleCSI_ENABLE_KOKKOS AND FleCSI_ENABLE_LEGION AND 
    (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP))
    list(APPEND UNIT_FLAGS "--backend-args=-ll:gpu 1")
 endif()
 
  add_test(NAME ${name}
  COMMAND
     ${MPIEXEC_EXECUTABLE} -np 1 $<TARGET_FILE:${name}> ${unit_ARGUMENTS} ${UNIT_FLAGS}
     WORKING_DIRECTORY
      ${CMAKE_BINARY_DIR}
  )

endfunction()

#------------------------------------------------------------------------------#
# App
#------------------------------------------------------------------------------#

add_subdirectory(app)
