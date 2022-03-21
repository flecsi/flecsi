#------------------------------------------------------------------------------#
# Copyright (c) 2016, Triad National Security, LLC
# All rights reserved
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.12)
project(Flaxpy LANGUAGES CXX C)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

find_package(FleCSI REQUIRED)

add_executable(flaxpy flaxpy.cc)
target_link_libraries(flaxpy FleCSI::FleCSI)


# ----------------------------------------------------------------------

# The rest of this file instructs "make test" to confirm that flaxpy
# can run without error.

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

add_unit(flaxpy)

# ----------------------------------------------------------------------
