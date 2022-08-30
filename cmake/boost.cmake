macro(flecsi_enable_boost)
  set(Boost_NO_BOOST_CMAKE ON)
  set(BOOST_COMPONENTS "${ARGN}")

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

  set(FLECSI_CMAKE_ENABLE_BOOST ON)
endmacro()
