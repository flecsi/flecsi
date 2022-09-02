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

cmake_minimum_required(VERSION 3.12)

#------------------------------------------------------------------------------#
# Add local module path.
#------------------------------------------------------------------------------#

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

#------------------------------------------------------------------------------#
# Project.
#------------------------------------------------------------------------------#

project(FleCSI LANGUAGES CXX C)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------------------------#
# Compiler version requirements.
#------------------------------------------------------------------------------#

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "Version 9.0 of gnu compilers required!")
  endif()
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wno-parentheses NO_PARENTHESES)
check_cxx_compiler_flag(-Wno-braced-scalar-init NO_BRACED_SCALAR_INIT)

if(NO_PARENTHESES)
  add_compile_options(-Wno-parentheses)
endif()
if(NO_BRACED_SCALAR_INIT)
  add_compile_options(-Wno-braced-scalar-init)
endif()

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
  include(format)

  set(ClangFormat_VERSION "13.0.1" CACHE STRING
      "Set the required version (major[.minor[.patch]]) of clang-format")
  mark_as_advanced(ClangFormat_VERSION)

  flecsi_add_format_target(${PROJECT_NAME} ${PROJECT_SOURCE_DIR} "${ClangFormat_VERSION}")
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
  include(documentation)

  option(ENABLE_DOXYGEN "Enable Doxygen documentation" ON)
  option(ENABLE_SPHINX  "Enable Sphinx documentation"  ON)
  mark_as_advanced(ENABLE_DOXYGEN)
  mark_as_advanced(ENABLE_SPHINX)


  if(ENABLE_DOXYGEN)
    include(doxygen)

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
    include(sphinx)

    flecsi_add_sphinx_target(main
      CONFIG ${CMAKE_SOURCE_DIR}/doc/sphinx
      OUTPUT ${CMAKE_BINARY_DIR}/doc
    )

    # This is for local install of the documentation. To deploy the
    # documentation to the flecsi website, use the "deploy-documentation" target.
    install(
      DIRECTORY
        ${CMAKE_BINARY_DIR}/doc/
      DESTINATION
        ${CMAKE_INSTALL_PREFIX}/share/doc/FleCSI
      OPTIONAL
    )
  endif()

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
  include(coverage)
  include(legion)
  include(mpi)
  include(parmetis)
  include(hpx)
  include(openmp)
  include(hdf5)
  include(graphviz)
  include(kokkos)
  include(boost)
  include(caliper)
  include(library)

  add_library(FleCSI)
  add_library(FleCSI::FleCSI ALIAS FleCSI)

  target_include_directories(FleCSI PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)

  #----------------------------------------------------------------------------#
  # Add option for setting entity id type
  #----------------------------------------------------------------------------#

  set(FLECSI_ID_TYPE "std::uint32_t" CACHE STRING
	 "Select the type that will be used for entity ids by topologies")
  mark_as_advanced(FLECSI_ID_TYPE)

  #----------------------------------------------------------------------------#
  # Add option for counter size
  #----------------------------------------------------------------------------#

  set(FLECSI_COUNTER_TYPE "std::int32_t" CACHE STRING
	 "Select the type that will be used for loop and iterator values"
	)
  mark_as_advanced(FLECSI_COUNTER_TYPE)

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
    set(ENABLE_MPI ON CACHE BOOL "Enable MPI" FORCE)
    set(ENABLE_PARMETIS ON CACHE BOOL "Enable ParMETIS" FORCE)
    set(ENABLE_LEGION OFF CACHE BOOL "Enable Legion" FORCE)
  elseif(FLECSI_BACKEND STREQUAL "legion")
    set(ENABLE_MPI ON CACHE BOOL "Enable MPI" FORCE)
    set(ENABLE_PARMETIS ON CACHE BOOL "Enable ParMETIS" FORCE)
    set(ENABLE_LEGION ON CACHE BOOL "Enable Legion" FORCE)
  elseif(FLECSI_BACKEND STREQUAL "hpx")
    set(ENABLE_MPI ON CACHE BOOL "Enable MPI" FORCE)
    set(ENABLE_PARMETIS ON CACHE BOOL "Enable ParMETIS" FORCE)
    set(ENABLE_HPX ON CACHE BOOL "Enable HPX" FORCE)
  endif()

  mark_as_advanced(ENABLE_MPI ENABLE_LEGION ENABLE_HPX)

  option(ENABLE_BOOST "Enable Boost" ON)
  option(ENABLE_MPI "Enable MPI" OFF)
  option(ENABLE_MPI_CXX_BINDINGS "Enable MPI C++ Bindings" OFF)
  option(ENABLE_PARMETIS "Enable ParMETIS" OFF)
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

  if(ENABLE_MPI)
    flecsi_enable_mpi(FleCSI ${ENABLE_MPI_CXX_BINDINGS})
  endif()

  if(ENABLE_PARMETIS)
    flecsi_enable_parmetis(FleCSI)
  endif()

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
      atomic
      filesystem
      regex
      system
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

  configure_file(${PROJECT_SOURCE_DIR}/config/flecsi-config.h.in
    ${CMAKE_BINARY_DIR}/flecsi-config.h @ONLY)

  install(
    FILES ${CMAKE_BINARY_DIR}/flecsi-config.h
    DESTINATION include
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

  configure_file(${PROJECT_SOURCE_DIR}/config/FleCSIConfig.cmake.in
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FleCSIConfig.cmake @ONLY)

  install(
    FILES ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FleCSIConfig.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${LIBDIR}/cmake/FleCSI
  )

  #----------------------------------------------------------------------------#
  # Unit Tests
  #----------------------------------------------------------------------------#
  set(FLECSI_UNIT_MAIN ${CMAKE_SOURCE_DIR}/flecsi/util/unit/main.cc)
  include(unit)

  option(ENABLE_UNIT_TESTS "Enable unit testing" OFF)

  if(ENABLE_UNIT_TESTS)
    flecsi_enable_testing()
  endif()

  #----------------------------------------------------------------------------#
  # FleCSI library.
  #----------------------------------------------------------------------------#

  add_library_target(FleCSI flecsi
    EXPORT_TARGET
      FleCSITargets
    VERSION
      ${FleCSI_VERSION}
    NAMESPACE
      FleCSI
  )

  #----------------------------------------------------------------------------#
  # Tools
  #----------------------------------------------------------------------------#
  add_subdirectory(tools)

  #----------------------------------------------------------------------------#
  # CMake utilities.
  #----------------------------------------------------------------------------#

  install(
    FILES
     ${PROJECT_SOURCE_DIR}/cmake/boost.cmake
     ${PROJECT_SOURCE_DIR}/cmake/caliper.cmake
     ${PROJECT_SOURCE_DIR}/cmake/colors.cmake
     ${PROJECT_SOURCE_DIR}/cmake/documentation.cmake
     ${PROJECT_SOURCE_DIR}/cmake/format.cmake
     ${PROJECT_SOURCE_DIR}/cmake/graphviz.cmake
     ${PROJECT_SOURCE_DIR}/cmake/hdf5.cmake
     ${PROJECT_SOURCE_DIR}/cmake/hpx.cmake
     ${PROJECT_SOURCE_DIR}/cmake/kokkos.cmake
     ${PROJECT_SOURCE_DIR}/cmake/legion.cmake
     ${PROJECT_SOURCE_DIR}/cmake/library.cmake
     ${PROJECT_SOURCE_DIR}/cmake/mpi.cmake
     ${PROJECT_SOURCE_DIR}/cmake/parmetis.cmake
     ${PROJECT_SOURCE_DIR}/cmake/sphinx.cmake
     ${PROJECT_SOURCE_DIR}/cmake/subdirlist.cmake
     ${PROJECT_SOURCE_DIR}/cmake/summary.cmake
     ${PROJECT_SOURCE_DIR}/cmake/unit.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${LIBDIR}/cmake/FleCSI
  )

  install(
    FILES
      ${PROJECT_SOURCE_DIR}/cmake/FindParMETIS.cmake
      ${PROJECT_SOURCE_DIR}/cmake/FindMETIS.cmake
      ${PROJECT_SOURCE_DIR}/cmake/FindClangFormat.cmake
      ${PROJECT_SOURCE_DIR}/cmake/FindSphinx.cmake
      ${PROJECT_SOURCE_DIR}/cmake/FindGraphviz.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${LIBDIR}/cmake
  )

  install(
    FILES ${FLECSI_UNIT_MAIN}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${LIBDIR}/cmake/FleCSI
    RENAME unit-main.cc
  )

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
include(summary)

flecsi_summary_header()
flecsi_summary_info("FleCSI Version" "${FleCSI_VERSION_BRIEF}" FALSE)

if(ENABLE_DOCUMENTATION)
  flecsi_summary_info("FleCSI_DOCUMENTATION_VERSION"
    "${FleCSI_DOCUMENTATION_VERSION}" FALSE)
endif()

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
  flecsi_summary_option("ENABLE_FLOG" ${ENABLE_FLOG}
    " (FLOG_STRIP_LEVEL ${FLOG_STRIP_LEVEL})")
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

if(ENABLE_FORMAT AND ClangFormat_FOUND)
  flecsi_summary_info("ClangFormat_EXECUTABLE" "${ClangFormat_EXECUTABLE}" TRUE)
endif()

message(STATUS ${_summary})
