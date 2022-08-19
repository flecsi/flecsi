option(ENABLE_FORMAT "Enable format target" OFF)
mark_as_advanced(ENABLE_FORMAT)

if(ENABLE_FORMAT)
  set(ClangFormat_VERSION "13.0.0" CACHE STRING
    "Set the required version (major[.minor[.patch]]) of clang-format")
  mark_as_advanced(ClangFormat_VERSION)

  if(NOT TARGET format)
    add_custom_target(format)
  endif()

  find_package(ClangFormat ${ClangFormat_VERSION} EXACT REQUIRED)
  find_package(Git REQUIRED)

  if(EXISTS ${PROJECT_SOURCE_DIR}/.git)
    execute_process(COMMAND ${GIT_EXECUTABLE} -C ${PROJECT_SOURCE_DIR}
      ls-files OUTPUT_VARIABLE _FILES OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)

    string(REGEX REPLACE "\n" ";" _FILES "${_FILES}")

    set(FORMAT_SOURCES)

    foreach(_FILE ${_FILES})
      if(_FILE MATCHES "\\.(hh|cc)$")
        list(APPEND FORMAT_SOURCES "${PROJECT_SOURCE_DIR}/${_FILE}")
      endif()
    endforeach()

    add_custom_target(format-${PROJECT_NAME}
      COMMAND ${ClangFormat_EXECUTABLE} -style=file -i ${FORMAT_SOURCES})
    add_dependencies(format format-${PROJECT_NAME})
  else()
    message(WARNING "format target requested, but not a Git checkout")
  endif()
endif()
