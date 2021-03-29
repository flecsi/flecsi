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

option(ENABLE_LEGION "Enable Legion" OFF)

if(ENABLE_LEGION)

  find_package(Legion REQUIRED)

  if(NOT Legion_FOUND)
    message(FATAL_ERROR "Legion is required for this build configuration")
  endif(NOT Legion_FOUND)

  list(APPEND TPL_DEFINES -DLEGION_USE_CMAKE -DREALM_USE_CMAKE)
  list(APPEND TPL_INCLUDES ${Legion_INCLUDE_DIRS}) 
  list(APPEND TPL_LIBRARIES ${Legion_LIBRARIES})

endif(ENABLE_LEGION)
