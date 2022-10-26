#------------------------------------------------------------------------------#
# Set a custom doxygen group target name
#------------------------------------------------------------------------------#

macro(flecsi_set_doxygen_target_name target)
  set(FLECSI_DOXYGEN_TARGET ${target})
endmacro()

#------------------------------------------------------------------------------#
# create group target ${FLECSI_DOXYGEN_TARGET}
# if no name is set, use default name "doxygen"
# ${FLECSI_DOXYGEN_TARGET} is also added as dependency of the
# ${FLECSI_DOC_TARGET} target defined via documentation.cmake
#------------------------------------------------------------------------------#

macro(_flecsi_define_doxygen_group_target)
  if(NOT DEFINED FLECSI_DOXYGEN_TARGET)
    # set default target name if not set
    set(FLECSI_DOXYGEN_TARGET doxygen PARENT_SCOPE)
    set(FLECSI_DOXYGEN_TARGET doxygen)
  endif()

  if(NOT DEFINED FLECSI_DOC_TARGET OR NOT TARGET ${FLECSI_DOC_TARGET})
    include(FleCSI/documentation)
    _flecsi_define_doc_group_target()
  endif()

  if(NOT TARGET ${FLECSI_DOXYGEN_TARGET})
    add_custom_target(${FLECSI_DOXYGEN_TARGET}
      ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/.dox-dummy
    )
    add_dependencies(${FLECSI_DOC_TARGET} ${FLECSI_DOXYGEN_TARGET})
  endif()
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

  _flecsi_define_doxygen_group_target()

  find_package(Doxygen REQUIRED)

  foreach(conf ${dox_CONFIGS})
    get_filename_component(output ${conf} NAME_WLE)
    configure_file(${conf} ${CMAKE_BINARY_DIR}/.doxygen/${name}/${output})
  endforeach()

  add_custom_target(${FLECSI_DOXYGEN_TARGET}-${name}
    ${CMAKE_COMMAND} -E remove_directory '${CMAKE_BINARY_DIR}/doc/api/${name}' &&
    $<TARGET_FILE:Doxygen::doxygen> ${CMAKE_BINARY_DIR}/.doxygen/${name}/conf
    DEPENDS ${dox_CONFIGS})

  add_dependencies(${FLECSI_DOXYGEN_TARGET} ${FLECSI_DOXYGEN_TARGET}-${name})
endfunction()
