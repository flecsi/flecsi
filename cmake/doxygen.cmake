#------------------------------------------------------------------------------#
# This creates a `doxygen` target that can be used to build all of the
# doxygen targets added with `add_doxygen_target`.
#------------------------------------------------------------------------------#

macro(flecsi_enable_doxygen)
  add_custom_target(doxygen
    ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/.dox-dummy
  )
  set(FLECSI_CMAKE_ENABLE_DOXYGEN ON)
endmacro()

#------------------------------------------------------------------------------#
# Add a doxygen target
#
# CONFIGS The input configuration files. The input files will be modified and
#         copied to temporaries using `configure_file`.
#------------------------------------------------------------------------------#

function(flecsi_add_doxygen_target name)
  set(options)
  set(multi_value_args CONFIGS)

  cmake_parse_arguments(dox "${options}" "${one_value_args}"
    "${multi_value_args}" ${ARGN})

  if(FLECSI_CMAKE_ENABLE_DOXYGEN)
    find_package(Doxygen REQUIRED)

    foreach(conf ${dox_CONFIGS})
      get_filename_component(output ${conf} NAME_WLE)
      configure_file(${conf} ${CMAKE_BINARY_DIR}/.doxygen/${name}/${output})
    endforeach()

    add_custom_target(doxygen-${name}
      ${CMAKE_COMMAND} -E remove_directory '${CMAKE_BINARY_DIR}/doc/api/${name}' &&
      $<TARGET_FILE:Doxygen::doxygen> ${CMAKE_BINARY_DIR}/.doxygen/${name}/conf
      DEPENDS ${dox_CONFIGS})

    add_dependencies(doxygen doxygen-${name})
    add_dependencies(doc doxygen)
  endif()
endfunction()
