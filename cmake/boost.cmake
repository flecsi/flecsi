# Copyright (c) 2016, Triad National Security, LLC
# All rights reserved

set(Boost_NO_BOOST_CMAKE ON)
set(ENABLE_BOOST ON CACHE BOOL "Enable Boost" FORCE)
mark_as_advanced(ENABLE_BOOST)

if(ENABLE_BOOST)

  #----------------------------------------------------------------------------#
  # Set BOOST_COMPONENTS to the desired components, e.g., program_options,
  # regex, etc.
  #----------------------------------------------------------------------------#

  # Disable warnings about new versions.
  set(Boost_NO_WARN_NEW_VERSIONS ON)

  find_package(Boost REQUIRED ${BOOST_COMPONENTS})

  list(APPEND TPL_LIBRARIES Boost::boost)

  foreach(_COMP IN LISTS BOOST_COMPONENTS)
    list(APPEND TPL_LIBRARIES Boost::${_COMP})
  endforeach()

endif()
