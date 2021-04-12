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

set(Boost_NO_BOOST_CMAKE ON)
set(ENABLE_BOOST ON CACHE BOOL "Enable Boost" FORCE)
mark_as_advanced(ENABLE_BOOST)

if(ENABLE_BOOST)

  #----------------------------------------------------------------------------#
  # Set BOOST_COMPONENTS to the desired components, e.g., program_options,
  # regex, etc.
  #----------------------------------------------------------------------------#

  find_package(Boost REQUIRED ${BOOST_COMPONENTS})

  list(APPEND TPL_INCLUDES ${Boost_INCLUDE_DIRS})
  list(APPEND TPL_LIBRARIES Boost::boost ${Boost_LIBRARIES})

endif()
