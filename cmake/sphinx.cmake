#------------------------------------------------------------------------------#
# This creates a `sphinx` target that can be used to build all of the
# sphinx targets added with `flecsi_add_sphinx_target`.
#------------------------------------------------------------------------------#

macro(flecsi_enable_sphinx)
  add_custom_target(sphinx
    ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/.sphx-dummy
  )
  set(FLECSI_CMAKE_ENABLE_SPHINX ON)
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

  if(FLECSI_CMAKE_ENABLE_SPHINX)
    find_package(Sphinx REQUIRED)

    file(COPY ${sphx_CONFIG}/_static DESTINATION ${sphx_OUTPUT}/.sphinx/)
    file(COPY ${sphx_CONFIG}/_templates DESTINATION ${sphx_OUTPUT}/.sphinx/)

    configure_file(${sphx_CONFIG}/conf.py.in
      ${sphx_OUTPUT}/.sphinx/conf.py)

    add_custom_target(${name}
      COMMAND ${SPHINX_EXECUTABLE} -nqW -c
        ${sphx_OUTPUT}/.sphinx
        ${sphx_CONFIG}
        ${sphx_OUTPUT}
    )

    add_dependencies(sphinx ${name})
    add_dependencies(doc sphinx)
  endif()

endfunction()
