#------------------------------------------------------------------------------#
# Set a custom format group target name
#------------------------------------------------------------------------------#

macro(flecsi_set_format_target_name name)
  set(FLECSI_FORMAT_TARGET ${name})
endmacro()

#------------------------------------------------------------------------------#
# create format group target ${FLECSI_FORMAT_TARGET}
# this collects all format targets as dependencies
# if no name is set, use default name "format"
#------------------------------------------------------------------------------#

macro(_flecsi_define_format_group_target)
  if(NOT DEFINED FLECSI_FORMAT_TARGET)
    set(FLECSI_FORMAT_TARGET format PARENT_SCOPE)
    set(FLECSI_FORMAT_TARGET format)
  endif()

  if(NOT TARGET ${FLECSI_FORMAT_TARGET})
    add_custom_target(${FLECSI_FORMAT_TARGET})
  endif()
endmacro()

function(flecsi_add_format_target target source_dir version)
  _flecsi_define_format_group_target()

  find_package(ClangFormat "${version}" REQUIRED)
  find_package(Git REQUIRED)

  if(EXISTS ${source_dir}/.git)
    execute_process(COMMAND ${GIT_EXECUTABLE} -C ${source_dir}
      ls-files OUTPUT_VARIABLE _FILES OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)

    string(REGEX REPLACE "\n" ";" _FILES "${_FILES}")

    set(FORMAT_SOURCES)

    foreach(_FILE ${_FILES})
      if(_FILE MATCHES "\\.(hh|cc)$")
        list(APPEND FORMAT_SOURCES "${source_dir}/${_FILE}")
      endif()
    endforeach()

    add_custom_target(${FLECSI_FORMAT_TARGET}-${target}
      COMMAND ClangFormat::ClangFormat -style=file -i ${FORMAT_SOURCES})
    add_dependencies(${FLECSI_FORMAT_TARGET} ${FLECSI_FORMAT_TARGET}-${target})
  else()
    message(WARNING "format target requested, but not a Git checkout")
  endif()
endfunction()
