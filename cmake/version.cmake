file(STRINGS ${CMAKE_SOURCE_DIR}/.version version
     REGEX "^(develop|f[0-9]+|r[0-9]+.[0-9]+|v[0-9]+.[0-9]+.[0-9]+)$"
     LIMIT_COUNT 1)

if(version STREQUAL "")
  message(FATAL_ERROR "Invalid version string in .version")
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/.git)
  find_package(Git)
endif()

if(version STREQUAL "develop")
  set(FleCSI_VERSION "255.0.0")
  set(FleCSI_BRANCH "develop")
  set(FleCSI_VERSION_BRIEF "develop")
  set(last_release_regex "^v")
else()
  string(SUBSTRING "${version}" 0 1 branch)
  string(SUBSTRING "${version}" 1 -1 version)
  set(last_release_regex "^v${version}")

  if(branch STREQUAL "f")
    set(FleCSI_VERSION "${version}.255.0")
    set(FleCSI_VERSION_BRIEF "${version} (feature branch)")
  elseif(branch STREQUAL "r")
    set(FleCSI_VERSION "${version}.255")
    set(FleCSI_VERSION_BRIEF "${version} (release branch)")
  elseif(branch STREQUAL "v")
    set(FleCSI_VERSION "${version}")
    set(FleCSI_VERSION_BRIEF "${version} (release)")
    set(last_release_regex "")
  endif()
endif()

string(REPLACE "." ";" ver_comp "${FleCSI_VERSION}")
list(GET ver_comp 0 FleCSI_VERSION_MAJOR)
list(GET ver_comp 1 FleCSI_VERSION_MINOR)
list(GET ver_comp 2 FleCSI_VERSION_PATCH)

math(EXPR FleCSI_VERSION_HEX
    "${FleCSI_VERSION_MAJOR} << 16 | ${FleCSI_VERSION_MINOR} << 8 | ${FleCSI_VERSION_PATCH}"
    OUTPUT_FORMAT HEXADECIMAL)

if(Git_FOUND AND last_release_regex)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} tag
    OUTPUT_VARIABLE output
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  string(REPLACE "\n" ";" tags "${output}")
  list(FILTER tags INCLUDE REGEX "${last_release_regex}")

  if(tags)
    list(GET tags -1 last_release)

    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse ${last_release}
      OUTPUT_VARIABLE last_release_sha
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-list --count ${last_release}..HEAD
      OUTPUT_VARIABLE commits
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set(FleCSI_REPO_STATE "${commits} commits since ${last_release} (${last_release_sha})")
  else()
    set(FleCSI_REPO_STATE "")
  endif()
endif()
