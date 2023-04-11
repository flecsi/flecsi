macro(flecsi_enable_boost target)
  set(Boost_NO_BOOST_CMAKE ON)
  cmake_parse_arguments(BOOST "" "" "COMPONENTS" ${ARGN})

  #----------------------------------------------------------------------------#
  # Set BOOST_COMPONENTS to the desired components, e.g., program_options,
  # regex, etc.
  #----------------------------------------------------------------------------#

  # Disable warnings about new versions.
  set(Boost_NO_WARN_NEW_VERSIONS ON)

  find_package(Boost REQUIRED ${BOOST_COMPONENTS})

  target_link_libraries(${target} PUBLIC Boost::boost)

  foreach(_COMP IN LISTS BOOST_COMPONENTS)
    target_link_libraries(${target} PUBLIC Boost::${_COMP})
  endforeach()

  # workaround for boost::stacktrace on MacOS
  if(APPLE)
    target_compile_definitions(${target} PUBLIC BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED)
  endif()

  set(FLECSI_CMAKE_ENABLE_BOOST ON)
endmacro()
