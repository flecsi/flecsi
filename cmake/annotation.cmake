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

set(CALIPER_DETAILS none low medium high)

if(NOT CALIPER_DETAIL)
  list(GET CALIPER_DETAILS 0 CALIPER_DETAIL)
endif()

set(CALIPER_DETAIL "${CALIPER_DETAIL}" CACHE STRING
  "Select the Caliper annotation detail (none,low,medium,high)")

set_property(CACHE CALIPER_DETAIL
  PROPERTY STRINGS ${CALIPER_DETAILS})

if (NOT CALIPER_DETAIL STREQUAL "none")
  find_package(caliper REQUIRED)

  message(STATUS "Found Caliper")

  list(APPEND TPL_LIBRARIES caliper)
endif()
