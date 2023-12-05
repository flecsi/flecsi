#------------------------------------------------------------------------------#
# Set a custom unit_tests target name
#------------------------------------------------------------------------------#

macro(flecsi_set_unit_tests_target_name name)
  set(FLECSI_UNIT_TESTS_TARGET ${name})
endmacro()

#------------------------------------------------------------------------------#
# create unit target ${FLECSI_UNIT_TESTS_TARGET}
# this collects all unit tests added by flecsi_add_test
# if no name is set, use default name "unit_tests"
#------------------------------------------------------------------------------#

macro(_flecsi_define_unit_tests_target)
  if(NOT DEFINED FLECSI_UNIT_TESTS_TARGET)
    set(FLECSI_UNIT_TESTS_TARGET unit_tests)
  endif()

  if(NOT TARGET ${FLECSI_UNIT_TESTS_TARGET})
    add_custom_target(${FLECSI_UNIT_TESTS_TARGET} ALL)
  endif()
endmacro()

macro(_flecsi_get_unit_test_backend_flags flag_variable)
  if (FleCSI_ENABLE_KOKKOS AND FleCSI_ENABLE_LEGION AND
     (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP))
   list(APPEND ${flag_variable} "--backend-args=-ll:gpu 1")
  endif()

  if (FleCSI_ENABLE_KOKKOS AND FleCSI_ENABLE_LEGION AND Kokkos_ENABLE_OPENMP AND Legion_USE_OpenMP)
    list(APPEND ${flag_variable} "--backend-args=-ll:ocpu 1 -ll:onuma 0")
  endif()
endmacro()

macro(flecsi_enable_testing)
  if(NOT FleCSI_ENABLE_FLOG)
    message(FATAL_ERROR "Unit tests require FleCSI with FLOG enabled")
  endif()

  enable_testing()
  _flecsi_define_unit_tests_target()
  set(FLECSI_ENABLE_TESTING ON)
endmacro()

function(_flecsi_define_unit_main_target)
  if(NOT TARGET flecsi-unit-main)
    add_library(flecsi-unit-main OBJECT ${FLECSI_UNIT_MAIN})
    target_link_libraries(flecsi-unit-main PUBLIC FleCSI::FleCSI)
  endif()
endfunction()

function(flecsi_test_link_libraries)
  _flecsi_define_unit_main_target()
  target_link_libraries(flecsi-unit-main PUBLIC ${ARGN})
endfunction()

function(_flecsi_add_mpi_test name)
  set(one_value_args TARGET PROCS WORKING_DIRECTORY)
  set(multi_value_args ARGUMENTS)
  cmake_parse_arguments(test "" "${one_value_args}"
    "${multi_value_args}" ${ARGN})

  if(NOT test_TARGET)
    message(FATAL_ERROR "You must specify a test target via TARGET")
  endif()

  if(NOT test_PROCS)
    set(test_PROCS 1)
  endif()

  if(NOT test_WORKING_DIRECTORY)
    get_target_property(test_WORKING_DIRECTORY ${test_TARGET} BINARY_DIR)
  endif()

  add_test(
    NAME "${name}"
    COMMAND
      ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_PREFLAGS}
      ${MPIEXEC_NUMPROC_FLAG} ${test_PROCS}
      $<TARGET_FILE:${test_TARGET}>
      ${MPIEXEC_POSTFLAGS}
      ${test_ARGUMENTS}
    WORKING_DIRECTORY "${test_WORKING_DIRECTORY}"
  )
endfunction()

function(flecsi_add_test name)

  if(NOT FLECSI_ENABLE_TESTING)
    return()
  endif()

  _flecsi_define_unit_main_target()

  #----------------------------------------------------------------------------#
  # Setup argument options.
  #----------------------------------------------------------------------------#

  set(options)
  set(one_value_args)
  set(multi_value_args
    SOURCES INPUTS PROCS LIBRARIES DEFINES ARGUMENTS TESTLABELS
  )
  cmake_parse_arguments(unit "${options}" "${one_value_args}"
    "${multi_value_args}" ${ARGN})

  #----------------------------------------------------------------------------#
  # Set output directory
  #----------------------------------------------------------------------------#

  get_filename_component(_SOURCE_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    set(_OUTPUT_DIR "${CMAKE_BINARY_DIR}/test/${_SOURCE_DIR_NAME}")
  else()
    set(_OUTPUT_DIR
      "${CMAKE_BINARY_DIR}/test/${PROJECT_NAME}/${_SOURCE_DIR_NAME}")
  endif()

  #----------------------------------------------------------------------------#
  # Set output directory information.
  #----------------------------------------------------------------------------#

  if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
      set(_TEST_PREFIX)
  else()
      set(_TEST_PREFIX "${PROJECT_NAME}:")
  endif()

  #----------------------------------------------------------------------------#
  # Make sure that the user specified sources.
  #----------------------------------------------------------------------------#

  if(NOT unit_SOURCES)
    message(FATAL_ERROR
      "You must specify unit test source files using SOURCES")
  endif()

  add_executable(${name} ${unit_SOURCES})
  add_dependencies(${FLECSI_UNIT_TESTS_TARGET} ${name})

  set_target_properties(${name}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${_OUTPUT_DIR})

  #----------------------------------------------------------------------------#
  # Set the folder property for VS and XCode
  #----------------------------------------------------------------------------#

  get_filename_component(_leafdir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  string(SUBSTRING ${_leafdir} 0 1 _first)
  string(TOUPPER ${_first} _first)
  string(REGEX REPLACE "^.(.*)" "${_first}\\1" _leafdir "${_leafdir}")
  string(CONCAT _folder "Tests/" ${_leafdir})
  set_target_properties(${name} PROPERTIES FOLDER "${_folder}")

  #----------------------------------------------------------------------------#
  # Check for defines.
  #----------------------------------------------------------------------------#

  if(unit_DEFINES)
    target_compile_definitions(${name} PRIVATE ${unit_DEFINES})
  endif()

  #----------------------------------------------------------------------------#
  # Check for input files.
  #----------------------------------------------------------------------------#

  if(unit_INPUTS)
    set(_OUTPUT_FILES)
    foreach(input ${unit_INPUTS})
      get_filename_component(_OUTPUT_NAME ${input} NAME)
      get_filename_component(_PATH ${input} ABSOLUTE)
      configure_file(${_PATH} ${_OUTPUT_DIR}/${_OUTPUT_NAME} COPYONLY)
      list(APPEND _OUTPUT_FILES ${_OUTPUT_DIR}/${_OUTPUT_NAME})
    endforeach()
    add_custom_target(${name}_inputs
      DEPENDS ${_OUTPUT_FILES})
    set_target_properties(${name}_inputs
      PROPERTIES FOLDER "${_folder}/Inputs")
    add_dependencies(${name} ${name}_inputs)
  endif()

  #----------------------------------------------------------------------------#
  # Check for library dependencies.
  #----------------------------------------------------------------------------#

  if(unit_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${unit_LIBRARIES})
  endif()

  target_link_libraries(${name} PRIVATE flecsi-unit-main)

  #----------------------------------------------------------------------------#
  # Check for procs
  #----------------------------------------------------------------------------#

  if(NOT unit_PROCS)
    set(unit_PROCS 1)
  endif()

  #----------------------------------------------------------------------------#
  # Add the test target to CTest
  #----------------------------------------------------------------------------#
  list(LENGTH unit_PROCS proc_instances)

  _flecsi_get_unit_test_backend_flags(UNIT_FLAGS)

  if(${proc_instances} EQUAL 1)
    set(unit_NAMES "${_TEST_PREFIX}${name}")
  else()
    set(unit_NAMES)
    foreach(instance ${unit_PROCS})
      list(APPEND unit_NAMES "${_TEST_PREFIX}${name}_${instance}")
    endforeach()
  endif()

  foreach(test procs IN ZIP_LISTS unit_NAMES unit_PROCS)
    _flecsi_add_mpi_test("${test}" TARGET ${name} PROCS ${procs}
      ARGUMENTS ${unit_ARGUMENTS} ${UNIT_FLAGS}
      WORKING_DIRECTORY ${_OUTPUT_DIR}
    )
    set_tests_properties("${test}" PROPERTIES LABELS "${unit_TESTLABELS}")
  endforeach()
endfunction()
