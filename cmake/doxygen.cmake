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

include(CMakeDependentOption)
include(colors)

cmake_dependent_option(ENABLE_DOXYGEN "Enable Doxygen documentation"
  ON "ENABLE_DOCUMENTATION" OFF)
mark_as_advanced(ENABLE_DOXYGEN)

cmake_dependent_option(ENABLE_DOXYGEN_WARN "Enable Doxygen warnings"
  OFF "ENABLE_DOCUMENTATION" OFF)
mark_as_advanced(ENABLE_DOXYGEN_WARN)

#------------------------------------------------------------------------------#
# This creates a `doxygen` target that can be used to build all of the
# doxygen targets added with `add_doxygen_target`.
#------------------------------------------------------------------------------#

if(ENABLE_DOXYGEN)
  add_custom_target(doxygen
    ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/.dox-dummy
  )
endif()

#------------------------------------------------------------------------------#
# Add a doxygen target
#
# CONFIGS The input configuration files. The input files will be modified and
#         copied to temporaries using `configure_file`.
#------------------------------------------------------------------------------#

function(add_doxygen_target name)

  set(options)
  set(multi_value_args CONFIGS)

  cmake_parse_arguments(dox "${options}" "${one_value_args}"
    "${multi_value_args}" ${ARGN})

  if(ENABLE_DOXYGEN)
    find_package(Doxygen REQUIRED)

    foreach(conf ${dox_CONFIGS})
      get_filename_component(output ${conf} NAME_WLE)
      configure_file(${conf} ${CMAKE_BINARY_DIR}/.doxygen/${name}/${output})
    endforeach()

    add_custom_target(${name}
      ${DOXYGEN} ${CMAKE_BINARY_DIR}/.doxygen/${name}/doxygen.conf
      DEPENDS ${dox_CONFIGS})

    add_dependencies(doxygen ${name})
    add_dependencies(doc doxygen)
  endif()
endfunction()
