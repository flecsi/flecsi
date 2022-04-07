# Copyright (c) 2016, Triad National Security, LLC
# All rights reserved

include(CMakeDependentOption)
include(colors)

cmake_dependent_option(ENABLE_SPHINX "Enable Sphinx documentation"
  ON "ENABLE_DOCUMENTATION" OFF)
mark_as_advanced(ENABLE_SPHINX)

#------------------------------------------------------------------------------#
# This creates a `sphinx` target that can be used to build all of the
# sphinx targets added with `add_sphinx_target`.
#------------------------------------------------------------------------------#

if(ENABLE_SPHINX)
  add_custom_target(sphinx
    ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/.sphx-dummy
  )
endif()

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

function(add_sphinx_target name)

  set(options)
  set(one_value_args CONFIG OUTPUT INSTALL)
  set(multi_value_args)

  cmake_parse_arguments(sphx "${options}" "${one_value_args}"
    "${multi_value_args}" ${ARGN})

  if(ENABLE_SPHINX)
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
