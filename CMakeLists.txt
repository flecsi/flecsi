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
# Copyright (c) 2016, Triad National Security, LLC
# All rights reserved
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.12...3.19)

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

#------------------------------------------------------------------------------#
# Compiler version requirements.
#------------------------------------------------------------------------------#

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "Version 9.0 of gnu compilers required!")
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "10.0")
    message(FATAL_ERROR "Cannot use this GCC version due to a regression <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=83258>")
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
# Add extra cmake modules.
#------------------------------------------------------------------------------#

find_package(LANLCMakeModules)

#------------------------------------------------------------------------------#
# Version.
#------------------------------------------------------------------------------#

include(version)

set(FleCSI_VERSION ${${PROJECT_NAME}_VERSION})

#------------------------------------------------------------------------------#
# Add summary macros.
#------------------------------------------------------------------------------#

include(summary)

#------------------------------------------------------------------------------#
# Add format target.
#------------------------------------------------------------------------------#

include(format)

if(NOT FORMAT_ONLY)

  #----------------------------------------------------------------------------#
  # Add option for setting entity id type 
  #----------------------------------------------------------------------------#

  set(FLECSI_ID_TYPE "std::uint32_t" CACHE STRING
    "Select the type that will be used for entity ids by topologies")
  mark_as_advanced(FLECSI_ID_TYPE)

  #----------------------------------------------------------------------------#
  # Add option for counter size
  #----------------------------------------------------------------------------#

  set(FLECSI_COUNTER_TYPE "int32_t" CACHE STRING
    "Select the type that will be used for loop and iterator values")
  mark_as_advanced(FLECSI_COUNTER_TYPE)

  #----------------------------------------------------------------------------#
  # These variables are used to collect library and include dependencies
  # for the FleCSIConfig file below.
  #----------------------------------------------------------------------------#

  set(FLECSI_INCLUDE_DEPENDENCIES)
  set(FLECSI_LIBRARY_DEPENDENCIES)

  #----------------------------------------------------------------------------#
  # Coverage.
  #----------------------------------------------------------------------------#

  include(coverage)

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

  #----------------------------------------------------------------------------#
  # Includes
  #----------------------------------------------------------------------------#

  include(legion)
  include(mpi)
  include(parmetis)
  include(hpx)
  include(openmp)
  include(hdf5)
  include(doxygen)
  include(sphinx)
  include(documentation)
  include(graphviz)
  include(kokkos)

  # These must be set before "include(boost)" below.
  list(APPEND
    BOOST_COMPONENTS
      program_options
      atomic
      filesystem
      regex
      system
  )
  include(boost)
  include(annotation)

  #----------------------------------------------------------------------------#
  # Documentation
  #----------------------------------------------------------------------------#


  add_sphinx_target(main
    CONFIG ${CMAKE_SOURCE_DIR}/doc/sphinx
    OUTPUT ${CMAKE_BINARY_DIR}/doc
  )

  # Create Doxygen output directory eagerly to avoid race condition.
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/doc/api)

  add_doxygen_target(core
    CONFIGS
      ${CMAKE_SOURCE_DIR}/doc/doxygen/core.conf.in
      ${CMAKE_SOURCE_DIR}/doc/doxygen/conf.in
  )
  add_doxygen_target(user
    CONFIGS
      ${CMAKE_SOURCE_DIR}/doc/doxygen/user.conf.in
      ${CMAKE_SOURCE_DIR}/doc/doxygen/conf.in
  )

  # This is for local install of the documentation. To deploy the
  # documentation to the flecsi website, use the "deploy-documentation" target.

  if(ENABLE_SPHINX)
    install(
      DIRECTORY
        ${CMAKE_BINARY_DIR}/doc/
      DESTINATION
        ${CMAKE_INSTALL_PREFIX}/share/doc/FleCSI
      OPTIONAL
    )
  endif()

  #----------------------------------------------------------------------------#
  # Caliper.
  #----------------------------------------------------------------------------#

  set(FLECSI_CALIPER_DETAIL ${CALIPER_DETAIL})

  #----------------------------------------------------------------------------#
  # Dynamic libraries.
  #----------------------------------------------------------------------------#

  foreach(_DL_LIB ${CMAKE_DL_LIBS})
    find_library(DL_LIB ${_DL_LIB})

    if(DL_LIB STREQUAL "NOTFOUND")
      message(FATAL_ERROR "Dynamic library not found")
    endif()

    list(APPEND TPL_LIBRARIES ${DL_LIB})
  endforeach()
  mark_as_advanced(DL_LIB)

  #  list(APPEND FLECSI_LIBRARY_DEPENDENCIES ${FLECSI_BACKEND_LIBRARIES})
  #set(FLECSI_BACKEND_LIBRARIES)

  #----------------------------------------------------------------------------#
  # FLOG and FTEST
  #----------------------------------------------------------------------------#

  include(flog)

  set(FLECSI_UNIT_MAIN ${CMAKE_SOURCE_DIR}/flecsi/util/unit/main.cc)
  include(unit)

  #----------------------------------------------------------------------------#
  # Capture enable settings.
  #----------------------------------------------------------------------------#

  get_cmake_property(_variableNames VARIABLES)
  string (REGEX MATCHALL "(^|;)ENABLE_[A-Za-z0-9_]*"
    _matchedVars "${_variableNames}")

  foreach(_variableName ${_matchedVars})
    set(FLECSI_${_variableName} ${${_variableName}})
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
  # mesh-gen target
  #----------------------------------------------------------------------------#

  add_executable(mesh-gen tools/mesh-gen.cc)

  #----------------------------------------------------------------------------#
  # FleCSI library.
  #----------------------------------------------------------------------------#

  include(library)

  add_library_target(FleCSI flecsi
    EXPORT_TARGET
      FleCSITargets
    VERSION
      ${FleCSI_VERSION}
    NAMESPACE
      FleCSI
    DEFINE_PRIVATE
      ${TPL_DEFINES}
    INCLUDE_PUBLIC
      ${TPL_INCLUDES}
    INCLUDE_PRIVATE
      ${CMAKE_BINARY_DIR}
    LINK_PUBLIC
      ${TPL_LIBRARIES}
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
  # CMake utilities.
  #----------------------------------------------------------------------------#

  configure_file(${PROJECT_SOURCE_DIR}/config/FleCSICMakeConfig.cmake.in
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FleCSICMakeConfig.cmake @ONLY)

  install(
    FILES ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FleCSICMakeConfig.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${LIBDIR}/cmake/FleCSI
  )

  install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/annotation.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/boost.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/colors.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/documentation.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/flog.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/format.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/graphviz.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/hdf5.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/hpx.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/kokkos.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/legion.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/library.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/mpi.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/parmetis.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/subdirlist.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/summary.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/unit.cmake
    FILES ${PROJECT_SOURCE_DIR}/cmake/version.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${LIBDIR}/cmake/FleCSI
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

  #----------------------------------------------------------------------------#
  # Output configuration summary.
  #----------------------------------------------------------------------------#

  summary_header()
  summary_info("CMAKE_BUILD_TYPE" "${CMAKE_BUILD_TYPE}" TRUE)
  summary_info("CMAKE_INSTALL_PREFIX" "${CMAKE_INSTALL_PREFIX}" TRUE)
  string(APPEND _summary "\n")
  summary_info("CMAKE_CXX_COMPILER" "${CMAKE_CXX_COMPILER}" TRUE)
  summary_info("CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}" TRUE)
  summary_info("CMAKE_C_COMPILER" "${CMAKE_C_COMPILER}" TRUE)
  summary_info("CMAKE_C_FLAGS" "${CMAKE_C_FLAGS}" TRUE)
  if(ClangFormat_FOUND)
    summary_info("ClangFormat_EXECUTABLE" "${ClangFormat_EXECUTABLE}" TRUE)
  endif()
  string(APPEND _summary "\n")
  summary_info("FleCSI_VERSION" "${FleCSI_VERSION}" TRUE)
  summary_info("FleCSI_DOCUMENTATION_VERSION"
    "${FleCSI_DOCUMENTATION_VERSION}" FALSE)
  if(${PROJECT_NAME}_COMMITS)
    summary_info("Repository state" "${${PROJECT_NAME}_COMMITS}" FALSE)
  endif()
  summary_info("FLECSI_BACKEND" "${FLECSI_BACKEND}" TRUE)
  summary_option("ENABLE_FLOG" ${ENABLE_FLOG}
    " (FLOG_STRIP_LEVEL ${FLOG_STRIP_LEVEL})")
  summary_option("ENABLE_UNIT_TESTS" ${ENABLE_UNIT_TESTS} "")
  string(APPEND _summary "\n")
  summary_option("ENABLE_KOKKOS" ${ENABLE_KOKKOS}
    " (${Kokkos_DEVICES})")
  if(ENABLE_DOCUMENTATION)
    summary_option("ENABLE_SPHINX" ${ENABLE_SPHINX} " (ENABLE_DOCUMENTATION)")
    summary_option("ENABLE_DOXYGEN" ${ENABLE_DOXYGEN} " (ENABLE_DOCUMENTATION)")
  endif()
  summary_option("ENABLE_GRAPHVIZ" ${ENABLE_GRAPHVIZ} "")
  summary_option("ENABLE_OPENMP" ${ENABLE_OPENMP} "")
  summary_option("ENABLE_HDF5" ${ENABLE_HDF5} "")
  summary_info("CALIPER_DETAIL" "${CALIPER_DETAIL}" TRUE)

  message(STATUS ${_summary})

else()

  summary_header()
  summary_option("FORMAT_ONLY" ${FORMAT_ONLY} "")

  message(STATUS ${_summary})

endif()
