#------------------------------------------------------------------------------#
#  @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
# /@@/////  /@@          @@////@@ @@////// /@@
# /@@       /@@  @@@@@  @@    // /@@       /@@
# /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
# /@@////   /@@/@@@@@@@/@@       ////////@@/@@
# /@@       /@@/@@//// //@@    @@       /@@/@@
# /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
# //       ///  //////   //////  ////////  //
#
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.23)

#------------------------------------------------------------------------------#
# Add local module path.
#------------------------------------------------------------------------------#

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

#------------------------------------------------------------------------------#
# Project.
#------------------------------------------------------------------------------#

project(FleCSI LANGUAGES NONE)

#------------------------------------------------------------------------------#
# FleCSI Version
#------------------------------------------------------------------------------#
include(version)

#------------------------------------------------------------------------------#
# Formatting
#------------------------------------------------------------------------------#
option(ENABLE_FORMAT "Enable format target" OFF)
mark_as_advanced(ENABLE_FORMAT)

if(ENABLE_FORMAT)
  include(FleCSI/format)

  set(CLANG_FORMAT_VERSION "13...<14" CACHE STRING
      "Set the required version (major[.minor[.patch]]) of clang-format")
  mark_as_advanced(CLANG_FORMAT_VERSION)

  flecsi_add_format_target(${PROJECT_NAME} ${PROJECT_SOURCE_DIR} "${CLANG_FORMAT_VERSION}")
endif()

#------------------------------------------------------------------------------#
# Documentation
#------------------------------------------------------------------------------#
option(ENABLE_DOCUMENTATION "Enable documentation" OFF)
set(GITHUB_PAGES_REPO "git@github.com:flecsi/flecsi.git"
    CACHE STRING "GitHub Pages repository for gh-pages branch")
mark_as_advanced(ENABLE_DOCUMENTATION)
mark_as_advanced(GITHUB_PAGES_REPO)

if(ENABLE_DOCUMENTATION)
  include(FleCSI/documentation)

  option(ENABLE_DOXYGEN "Enable Doxygen documentation" ON)
  option(ENABLE_SPHINX  "Enable Sphinx documentation"  ON)
  mark_as_advanced(ENABLE_DOXYGEN)
  mark_as_advanced(ENABLE_SPHINX)


  if(ENABLE_DOXYGEN)
    include(FleCSI/doxygen)

    # Create Doxygen output directory eagerly to avoid race condition.
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/doc/api)

    flecsi_add_doxygen_target(core
      CONFIGS
        ${CMAKE_SOURCE_DIR}/doc/doxygen/core.conf.in
        ${CMAKE_SOURCE_DIR}/doc/doxygen/conf.in
    )
    flecsi_add_doxygen_target(user
      CONFIGS
        ${CMAKE_SOURCE_DIR}/doc/doxygen/user.conf.in
        ${CMAKE_SOURCE_DIR}/doc/doxygen/conf.in
    )
  endif()

  if(ENABLE_SPHINX)
    include(FleCSI/sphinx)

    flecsi_add_sphinx_target(main
      CONFIG ${CMAKE_SOURCE_DIR}/doc/sphinx
      OUTPUT ${CMAKE_BINARY_DIR}/doc
    )
  endif()

  # This is for local install of the documentation. To deploy the
  # documentation to the flecsi website, use the "deploy-documentation" target.
  install(
    DIRECTORY
      ${CMAKE_BINARY_DIR}/doc/
    DESTINATION
      ${CMAKE_INSTALL_PREFIX}/share/doc/FleCSI
    OPTIONAL
  )

  if((NOT GITHUB_PAGES_REPO STREQUAL "") AND ENABLE_SPHINX AND ENABLE_DOXYGEN)
    flecsi_add_doc_deployment(deploy-docs GITHUB_PAGES_REPO ${GITHUB_PAGES_REPO})
  endif()
endif()


#------------------------------------------------------------------------------#
# FleCSI Library
#------------------------------------------------------------------------------#
option(ENABLE_LIBRARY "Enable build of FleCSI library" ON)
mark_as_advanced(ENABLE_LIBRARY)

if(ENABLE_LIBRARY)
  #----------------------------------------------------------------------------#
  # Build shared libraries by default
  #----------------------------------------------------------------------------#
  option(BUILD_SHARED_LIBS "Build shared libs" ON)
  mark_as_advanced(BUILD_SHARED_LIBS)

  enable_language(CXX)
  enable_language(C)

  set(CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_EXTENSIONS OFF)

  include(GNUInstallDirs)

  include(FleCSI/coverage)
  include(FleCSI/legion)
  include(FleCSI/mpi)
  include(FleCSI/parmetis)
  include(FleCSI/hpx)
  include(FleCSI/openmp)
  include(FleCSI/hdf5)
  include(FleCSI/graphviz)
  include(FleCSI/kokkos)
  include(FleCSI/boost)
  include(FleCSI/caliper)

  add_library(FleCSI)
  add_library(FleCSI::FleCSI ALIAS FleCSI)

  target_include_directories(FleCSI
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  #----------------------------------------------------------------------------#
  # Add option for setting entity id type
  #----------------------------------------------------------------------------#

  set(FLECSI_ID_TYPE "std::uint32_t" CACHE STRING
	 "Select the type that will be used for entity ids by topologies")
  mark_as_advanced(FLECSI_ID_TYPE)

  set(FLECSI_GID_TYPE "std::uint64_t" CACHE STRING
    "Select the type that will be used for global ids by topologies")
  mark_as_advanced(FLECSI_GID_TYPE)

  #----------------------------------------------------------------------------#
  # Coverage.
  #----------------------------------------------------------------------------#
  option(ENABLE_COVERAGE_BUILD "Do a coverage build" OFF)
  mark_as_advanced(ENABLE_COVERAGE_BUILD)

  if(ENABLE_COVERAGE_BUILD)
    flecsi_enable_coverage()
  endif()

  #----------------------------------------------------------------------------#
  # Add options for backend selection
  #----------------------------------------------------------------------------#

  set(FLECSI_BACKENDS legion mpi hpx)

  if(NOT FLECSI_BACKEND)
    list(GET FLECSI_BACKENDS 0 FLECSI_BACKEND)
  endif()

  set(FLECSI_BACKEND "${FLECSI_BACKEND}" CACHE STRING
    "Select the backend")
  set_property(CACHE FLECSI_BACKEND
    PROPERTY STRINGS ${FLECSI_BACKENDS})
  set(FleCSI_BACKEND "${FLECSI_BACKEND}")

  #----------------------------------------------------------------------------#
  # Backend defaults.
  #----------------------------------------------------------------------------#

  if(FLECSI_BACKEND STREQUAL "mpi")
    set(ENABLE_LEGION OFF CACHE BOOL "Enable Legion" FORCE)
  elseif(FLECSI_BACKEND STREQUAL "legion")
    set(ENABLE_LEGION ON CACHE BOOL "Enable Legion" FORCE)
  elseif(FLECSI_BACKEND STREQUAL "hpx")
    set(ENABLE_HPX ON CACHE BOOL "Enable HPX" FORCE)
  endif()

  mark_as_advanced(ENABLE_LEGION ENABLE_HPX)

  option(ENABLE_BOOST "Enable Boost" ON)
  option(ENABLE_MPI_CXX_BINDINGS "Enable MPI C++ Bindings" OFF)
  option(ENABLE_HDF5 "Enable HDF5" OFF)
  option(ENABLE_OPENMP "Enable OpenMP Task Support" OFF)
  option(ENABLE_GRAPHVIZ "Enable Graphviz" OFF)
  option(ENABLE_HPX "Enable HPX" OFF)
  option(ENABLE_KOKKOS "Enable Kokkos" OFF)

  mark_as_advanced(ENABLE_MPI_CXX_BINDINGS)
  mark_as_advanced(ENABLE_BOOST)
  mark_as_advanced(ENABLE_HPX)

  #----------------------------------------------------------------------------#
  # Caliper
  #----------------------------------------------------------------------------#
  set(CALIPER_DETAILS none low medium high)

  if(NOT CALIPER_DETAIL)
    list(GET CALIPER_DETAILS 0 CALIPER_DETAIL)
  endif()

  set(CALIPER_DETAIL "${CALIPER_DETAIL}" CACHE STRING
    "Select the Caliper annotation detail (none,low,medium,high)")

  set_property(CACHE CALIPER_DETAIL PROPERTY STRINGS ${CALIPER_DETAILS})

  set(FLECSI_CALIPER_DETAIL ${CALIPER_DETAIL})

  #----------------------------------------------------------------------------#
  # Apply options
  #----------------------------------------------------------------------------#
  if(ENABLE_LEGION)
    flecsi_enable_legion(FleCSI)
  endif()

  flecsi_enable_mpi(FleCSI ${ENABLE_MPI_CXX_BINDINGS})

  flecsi_enable_parmetis(FleCSI)

  if(ENABLE_HPX)
    flecsi_enable_hpx(FleCSI)
  endif()

  if(ENABLE_OPENMP)
    flecsi_enable_openmp(FleCSI)
  endif()

  if(ENABLE_HDF5)
    flecsi_enable_hdf5(FleCSI)
  endif()

  if(ENABLE_GRAPHVIZ)
    flecsi_enable_graphviz(FleCSI)
  endif()

  if(ENABLE_KOKKOS)
    flecsi_enable_kokkos(FleCSI)
  endif()

  if(ENABLE_BOOST)
    flecsi_enable_boost(FleCSI
    COMPONENTS
      program_options
    )
  endif()

  if (NOT CALIPER_DETAIL STREQUAL "none")
    flecsi_enable_caliper(FleCSI)
  endif()

  #----------------------------------------------------------------------------#
  # Dynamic libraries.
  #----------------------------------------------------------------------------#
  target_link_libraries(FleCSI PUBLIC ${CMAKE_DL_LIBS})

  #----------------------------------------------------------------------------#
  # FLOG
  #----------------------------------------------------------------------------#
  include(flog)

  #----------------------------------------------------------------------------#
  # Capture enable settings.
  #----------------------------------------------------------------------------#

  get_cmake_property(_variableNames VARIABLES)
  string (REGEX MATCHALL "(^|;)ENABLE_[A-Za-z0-9_]*"
    _matchedVars "${_variableNames}")

  foreach(_variableName ${_matchedVars})
    set(FLECSI_${_variableName} ${${_variableName}})
    set(FleCSI_${_variableName} ${${_variableName}})
  endforeach()


  #----------------------------------------------------------------------------#
  # Configure header.
  #----------------------------------------------------------------------------#

  configure_file(${PROJECT_SOURCE_DIR}/flecsi/config.hh.in
    ${CMAKE_BINARY_DIR}/flecsi/config.hh @ONLY)

  install(
    FILES ${CMAKE_BINARY_DIR}/flecsi/config.hh
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flecsi
  )

  #----------------------------------------------------------------------------#
  # Extract all project options so they can be exported to the
  # FleCSIConfig.cmake file.
  #----------------------------------------------------------------------------#

  get_cmake_property(_variableNames VARIABLES)
  string (REGEX MATCHALL "(^|;)FLECSI_[A-Za-z0-9_]*"
    _matchedVars "${_variableNames}")

  foreach(_variableName ${_matchedVars})
    string(REGEX REPLACE "FLECSI" "FleCSI" _tmpvariableName ${_variableName})
    set(FLECSI_CONFIG_CODE
      "${FLECSI_CONFIG_CODE}\nset(${_tmpvariableName} \"${${_variableName}}\")")
  endforeach()

  string(REGEX MATCHALL "(^|;)FLOG_[A-Za-z0-9_]*"
    _matchedVars "${_variableNames}")

  foreach(_variableName ${_matchedVars})
    set(FLOG_CONFIG_CODE
      "${FLOG_CONFIG_CODE}\nset(${_variableName} \"${${_variableName}}\")")
  endforeach()

  configure_file(${PROJECT_SOURCE_DIR}/cmake/FleCSI/FleCSIConfig.cmake.in
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FleCSIConfig.cmake @ONLY)

  install(
    FILES ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FleCSIConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FleCSI
  )

  #----------------------------------------------------------------------------#
  # Unit Tests
  #----------------------------------------------------------------------------#
  set(FLECSI_UNIT_MAIN ${CMAKE_SOURCE_DIR}/flecsi/util/unit/main.cc)
  include(FleCSI/unit)

  option(ENABLE_UNIT_TESTS "Enable unit testing" OFF)

  if(ENABLE_UNIT_TESTS)
    flecsi_enable_testing()
  endif()

  #----------------------------------------------------------------------------#
  # FleCSI library.
  #----------------------------------------------------------------------------#
  add_subdirectory(flecsi)

  #----------------------------------------------------------------------------#
  # Install
  #----------------------------------------------------------------------------#
  install(TARGETS FleCSI
    EXPORT
      FleCSITargets
    DESTINATION
      ${CMAKE_INSTALL_LIBDIR}
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flecsi
  )

  install(EXPORT FleCSITargets
    FILE
      FleCSITargets.cmake
    NAMESPACE
      FleCSI::
    DESTINATION
      ${CMAKE_INSTALL_LIBDIR}/cmake/FleCSI
  )

  #----------------------------------------------------------------------------#
  # Version
  #----------------------------------------------------------------------------#
  set_target_properties(FleCSI
    PROPERTIES
      VERSION ${FleCSI_VERSION}
  )

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/FleCSIConfigVersion.cmake
    VERSION ${FleCSI_VERSION}
    COMPATIBILITY AnyNewerVersion
  )
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/FleCSIConfigVersion.cmake
    DESTINATION
      ${CMAKE_INSTALL_LIBDIR}/cmake/${target}
  )

  #----------------------------------------------------------------------------#
  # Local export
  #----------------------------------------------------------------------------#
  export(EXPORT FleCSITargets
    FILE
      ${CMAKE_BINARY_DIR}/FleCSITargets.cmake
    NAMESPACE
      FleCSI::
  )

  export(PACKAGE FleCSI)

  #----------------------------------------------------------------------------#
  # Tools
  #----------------------------------------------------------------------------#
  add_subdirectory(tools)

  #----------------------------------------------------------------------------#
  # CMake utilities.
  #----------------------------------------------------------------------------#

  install(
    DIRECTORY
     ${PROJECT_SOURCE_DIR}/cmake/FleCSI
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
    FILES_MATCHING
    PATTERN "*"
    PATTERN "*.cmake.in" EXCLUDE
  )

  install(
    FILES
      ${PROJECT_SOURCE_DIR}/cmake/FindParMETIS.cmake
      ${PROJECT_SOURCE_DIR}/cmake/FindMETIS.cmake
      ${PROJECT_SOURCE_DIR}/cmake/FindClangFormat.cmake
      ${PROJECT_SOURCE_DIR}/cmake/FindSphinx.cmake
      ${PROJECT_SOURCE_DIR}/cmake/FindGraphviz.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
  )

  if(ENABLE_FLOG)
    install(
      FILES ${FLECSI_UNIT_MAIN}
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FleCSI
      RENAME unit-main.cc
    )
  endif()

  #----------------------------------------------------------------------------#
  # HPX extra.
  #----------------------------------------------------------------------------#

  if(FLECSI_BACKEND STREQUAL "hpx")
    hpx_setup_target(FleCSI NONAMEPREFIX)
  endif()
endif()

#------------------------------------------------------------------------------#
# Output configuration summary.
#------------------------------------------------------------------------------#
include(FleCSI/summary)

flecsi_summary_header()
flecsi_summary_info("FleCSI Version" "${FleCSI_VERSION_BRIEF}" FALSE)

string(APPEND _summary "\n")

if(FleCSI_REPO_STATE)
  flecsi_summary_info("Repository state" "${FleCSI_REPO_STATE}" FALSE)
endif()

string(APPEND _summary "\n")

if(ENABLE_LIBRARY)
  flecsi_summary_info("CMAKE_BUILD_TYPE" "${CMAKE_BUILD_TYPE}" TRUE)
  flecsi_summary_info("CMAKE_INSTALL_PREFIX" "${CMAKE_INSTALL_PREFIX}" TRUE)
  string(APPEND _summary "\n")
  flecsi_summary_info("CMAKE_CXX_COMPILER" "${CMAKE_CXX_COMPILER}" TRUE)
  flecsi_summary_info("CMAKE_CXX_COMPILER_VERSION"
    "${CMAKE_CXX_COMPILER_VERSION}" TRUE)
  flecsi_summary_info("CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}" TRUE)
  flecsi_summary_info("CMAKE_C_COMPILER" "${CMAKE_C_COMPILER}" TRUE)
  flecsi_summary_info("CMAKE_C_COMPILER_VERSION" "${CMAKE_C_COMPILER_VERSION}" TRUE)
  flecsi_summary_info("CMAKE_C_FLAGS" "${CMAKE_C_FLAGS}" TRUE)

  string(APPEND _summary "\n")
  flecsi_summary_info("FLECSI_BACKEND" "${FLECSI_BACKEND}" TRUE)
  flecsi_summary_option("ENABLE_FLOG" ${ENABLE_FLOG} "")
  flecsi_summary_option("ENABLE_UNIT_TESTS" ${ENABLE_UNIT_TESTS} "")
  string(APPEND _summary "\n")
  flecsi_summary_option("ENABLE_KOKKOS" ${ENABLE_KOKKOS}
    " (${Kokkos_DEVICES})")
  flecsi_summary_option("ENABLE_GRAPHVIZ" ${ENABLE_GRAPHVIZ} "")
  flecsi_summary_option("ENABLE_OPENMP" ${ENABLE_OPENMP} "")
  flecsi_summary_option("ENABLE_HDF5" ${ENABLE_HDF5} "")
  flecsi_summary_info("CALIPER_DETAIL" "${CALIPER_DETAIL}" TRUE)
endif()

if(ENABLE_DOCUMENTATION)
  flecsi_summary_option("ENABLE_SPHINX" ${ENABLE_SPHINX} " (ENABLE_DOCUMENTATION)")
  flecsi_summary_option("ENABLE_DOXYGEN" ${ENABLE_DOXYGEN} " (ENABLE_DOCUMENTATION)")
endif()

if(ENABLE_FORMAT AND TARGET ClangFormat::ClangFormat)
  flecsi_summary_info("ClangFormat_EXECUTABLE" "${ClangFormat_EXECUTABLE}" TRUE)
endif()

message(STATUS ${_summary})
