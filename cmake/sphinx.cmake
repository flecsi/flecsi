#------------------------------------------------------------------------------#
# Set a custom sphinx group target name
#------------------------------------------------------------------------------#

macro(flecsi_set_sphinx_target_name target)
  set(FLECSI_SPHINX_TARGET ${target})
endmacro()

#------------------------------------------------------------------------------#
# create group target ${FLECSI_SPHINX_TARGET}
# if no name is set, use default name "sphinx"
# ${FLECSI_SPHINX_TARGET} is also added as dependency of the
# ${FLECSI_DOC_TARGET} target defined via documentation.cmake
#------------------------------------------------------------------------------#

macro(_flecsi_define_sphinx_group_target)
  if(NOT DEFINED FLECSI_SPHINX_TARGET)
    set(FLECSI_SPHINX_TARGET sphinx PARENT_SCOPE)
    set(FLECSI_SPHINX_TARGET sphinx)
  endif()

  if(NOT DEFINED FLECSI_DOC_TARGET OR NOT TARGET ${FLECSI_DOC_TARGET})
    include(documentation)
    _flecsi_define_doc_group_target()
  endif()

  if(NOT TARGET ${FLECSI_SPHINX_TARGET})
    add_custom_target(${FLECSI_SPHINX_TARGET}
      ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/.sphx-dummy
    )
    add_dependencies(${FLECSI_DOC_TARGET} ${FLECSI_SPHINX_TARGET})
  endif()
endmacro()

#------------------------------------------------------------------------------#
# Add a sphinx target
#
# CONFIG The sphinx configuration directory:
#          - conf.py.in will be processed with configure_file and copied to the
#            temporaries directory.
#          - _static and _templates will be copied to the OUTPUT directory.
# OUTPUT The output directory for documentation and temporaries
#        (.sphinx subdir).
#------------------------------------------------------------------------------#

function(flecsi_add_sphinx_target name)

  set(options)
  set(one_value_args CONFIG OUTPUT INSTALL)
  set(multi_value_args)

  cmake_parse_arguments(sphx "${options}" "${one_value_args}"
    "${multi_value_args}" ${ARGN})

  _flecsi_define_sphinx_group_target()

  find_package(Sphinx REQUIRED)

  file(COPY ${sphx_CONFIG}/_static DESTINATION ${sphx_OUTPUT}/.sphinx/)
  file(COPY ${sphx_CONFIG}/_templates DESTINATION ${sphx_OUTPUT}/.sphinx/)

  # Generate SVG graphics from GraphViz Dot sources.
  file(
    GLOB_RECURSE source_dot_files
    LIST_DIRECTORIES false
    "${sphx_CONFIG}/*.dot"
  )

  if(source_dot_files)
    find_package(Graphviz REQUIRED)
    foreach(image_dot IN LISTS source_dot_files)
      string(REPLACE .dot .svg image_svg "${image_dot}")
      add_custom_command(OUTPUT ${image_svg}
        MAIN_DEPENDENCY ${image_dot}
        COMMAND Graphviz::dot -Tsvg -o ${image_svg} ${image_dot})
    endforeach()
    string(REPLACE .dot .svg generated_svg_files "${source_dot_files}")
  endif()

  add_custom_target(${name}
    COMMAND ${SPHINX_EXECUTABLE} -nqW -c
      ${sphx_OUTPUT}/.sphinx
      ${sphx_CONFIG}
      ${sphx_OUTPUT}
  )

  configure_file(${sphx_CONFIG}/conf.py.in
    ${sphx_OUTPUT}/.sphinx/conf.py)

  add_custom_target(${FLECSI_SPHINX_TARGET}-${name}
    COMMAND Sphinx::Sphinx -nqW -c
      ${sphx_OUTPUT}/.sphinx
      ${sphx_CONFIG}
      ${sphx_OUTPUT}
    DEPENDS "${generated_svg_files}"
  )

  add_dependencies(${FLECSI_SPHINX_TARGET} ${FLECSI_SPHINX_TARGET}-${name})

endfunction()
