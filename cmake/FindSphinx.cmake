find_program(Sphinx_EXECUTABLE
  NAMES sphinx-build
  PATH_SUFFIXES bin
  DOC "Sphinx documenation build executable"
)
mark_as_advanced(Sphinx_EXECUTABLE)

if(Sphinx_EXECUTABLE)
  execute_process(COMMAND ${Sphinx_EXECUTABLE} --version
                  OUTPUT_VARIABLE sphinx_version
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  RESULT_VARIABLE _sphinx_version_result)

  if(_sphinx_version_result)
    message(WARNING
      "Unable to determine sphinx-build verison: ${_sphinx_version_result}")
  else()
    string(REGEX REPLACE "sphinx-build.* ([0-9.]+).*"
                         "\\1"
                         Sphinx_VERSION
                         "${sphinx_version}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Sphinx
  REQUIRED_VARS
    Sphinx_EXECUTABLE
  VERSION_VAR
    Sphinx_VERSION
)

if(Sphinx_FOUND AND NOT TARGET Sphinx::Sphinx)
  add_executable(Sphinx::Sphinx IMPORTED GLOBAL)
  set_target_properties(Sphinx::Sphinx
    PROPERTIES IMPORTED_LOCATION "${Sphinx_EXECUTABLE}")
endif()
