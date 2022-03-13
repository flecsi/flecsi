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

execute_process(
  COMMAND ${CMAKE_SOURCE_DIR}/config/version.sh
  OUTPUT_VARIABLE version
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX REPLACE "\n$" "" version "${version}")

# debug
# message(STATUS "VERSION: ${version}")

string(REPLACE " " ";" fields ${version})
list(LENGTH fields size)

list(GET fields 0 branch)
list(GET fields 1 version)

if(size GREATER 2)
  list(SUBLIST fields 2 -1 rest)
endif()

# debug
#message(STATUS "branch: ${branch}")
#message(STATUS "version: ${version}")
#message(STATUS "fields: ${fields}")
#message(STATUS "rest: ${rest}")

set(${PROJECT_NAME}_COMMITS)
if(rest)
  string(REPLACE ";" "\ " commits "${rest}")
  string(REGEX REPLACE "([()])" "" commits "${commits}")
  string(STRIP commits "${commits}")
  set(${PROJECT_NAME}_COMMITS ${commits})
endif()

set(${PROJECT_NAME}_VERSION ${version})

if(branch STREQUAL "devel")
  string(REPLACE "." ";" vfields ${version})
  list(GET vfields 0 major)
  math(EXPR next "${major}+1")
  set(${PROJECT_NAME}_DOCUMENTATION_VERSION "${next}.-1 (devel)")
else()
  set(${PROJECT_NAME}_DOCUMENTATION_VERSION "${version}")
endif()
