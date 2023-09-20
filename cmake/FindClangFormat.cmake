#
# FindClangFormat
# ---------------
#
# The module defines the following variables
#
# ClangFormat_EXECUTABLE   - Path to clang-format executable
# ClangFormat_FOUND        - True if the clang-format executable was found.
# ClangFormat_VERSION      - The version of clang-format found
#
find_program(ClangFormat_EXECUTABLE
  NAMES
    clang-format-13
    clang-format-12
    clang-format-11
    clang-format-10
    clang-format-9
    clang-format-8
    clang-format-7
    clang-format-6.0
    clang-format-5.0
    clang-format-4.0
    clang-format-3.9
    clang-format-3.8
    clang-format-3.7
    clang-format-3.6
    clang-format-3.5
    clang-format-3.4
    clang-format-3.3
    clang-format
  DOC
    "clang-format executable"
)
mark_as_advanced(ClangFormat_EXECUTABLE)

# Extract version from command "clang-format -version"
if(ClangFormat_EXECUTABLE)
  execute_process(COMMAND ${ClangFormat_EXECUTABLE} -version
    OUTPUT_VARIABLE clang_format_version
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(clang_format_version MATCHES "clang-format version ([.0-9]+)")
    # clang_format_version sample: "clang-format version 3.9.1-4ubuntu3~16.04.1
    # (tags/RELEASE_391/rc2)"
    set(ClangFormat_VERSION ${CMAKE_MATCH_1})
  else()
    set(ClangFormat_VERSION 0.0)
  endif()
else()
  set(ClangFormat_VERSION 0.0)
endif()

include(FindPackageHandleStandardArgs)

# handle the QUIETLY and REQUIRED arguments and set ClangFormat_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ClangFormat
  REQUIRED_VARS
    ClangFormat_EXECUTABLE
  VERSION_VAR
    ClangFormat_VERSION
  HANDLE_VERSION_RANGE
)

if(ClangFormat_FOUND AND NOT TARGET ClangFormat::ClangFormat)
  add_executable(ClangFormat::ClangFormat IMPORTED GLOBAL)
  set_target_properties(ClangFormat::ClangFormat
    PROPERTIES IMPORTED_LOCATION "${ClangFormat_EXECUTABLE}")
endif()
