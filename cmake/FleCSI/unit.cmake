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

macro(flecsi_enable_testing)
  enable_testing()
  _flecsi_define_unit_tests_target()
  set(FLECSI_ENABLE_TESTING ON)
endmacro()

function(_flecsi_define_unit_main_target)
  if(NOT FleCSI_ENABLE_FLOG)
    message(FATAL_ERROR "Unit tests require FleCSI with FLOG enabled")
  endif()

  if(NOT TARGET flecsi-unit-main)
    add_library(flecsi-unit-main OBJECT ${FLECSI_UNIT_MAIN})
    target_link_libraries(flecsi-unit-main PUBLIC FleCSI::FleCSI)
  endif()
endfunction()

function(flecsi_test_link_libraries)
  _flecsi_define_unit_main_target()
  target_link_libraries(flecsi-unit-main PUBLIC ${ARGN})
endfunction()

function(_flecsi_get_test_output_dir OUTPUT_DIR)
  get_filename_component(_SOURCE_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    set(_OUTPUT_DIR "${CMAKE_BINARY_DIR}/test/${_SOURCE_DIR_NAME}")
  else()
    set(_OUTPUT_DIR
      "${CMAKE_BINARY_DIR}/test/${PROJECT_NAME}/${_SOURCE_DIR_NAME}")
  endif()
  set(${OUTPUT_DIR} "${_OUTPUT_DIR}" PARENT_SCOPE)
endfunction()

function(flecsi_add_test name)
  if(NOT FLECSI_ENABLE_TESTING)
    return()
  endif()

  set(one_value_args LAUNCHER)
  set(multi_value_args
    SOURCES LIBRARIES DEFINES INPUTS PROCS ARGUMENTS TESTLABELS LAUNCHER_ARGUMENTS
  )
  cmake_parse_arguments(unit "" "${one_value_args}"
    "${multi_value_args}" ${ARGN})

  if(NOT unit_SOURCES)
    message(FATAL_ERROR "You must specify unit test source files using SOURCES")
  endif()

  _flecsi_define_unit_main_target()

  add_executable(${name} ${unit_SOURCES})
  target_link_libraries(${name} PRIVATE flecsi-unit-main)

  add_dependencies(${FLECSI_UNIT_TESTS_TARGET} ${name})

  if(unit_DEFINES)
    target_compile_definitions(${name} PRIVATE ${unit_DEFINES})
  endif()

  if(unit_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${unit_LIBRARIES})
  endif()

  _flecsi_get_test_output_dir(_OUTPUT_DIR)

  # Set the folder property for VS and XCode
  get_filename_component(_leafdir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  string(SUBSTRING ${_leafdir} 0 1 _first)
  string(TOUPPER ${_first} _first)
  string(REGEX REPLACE "^.(.*)" "${_first}\\1" _leafdir "${_leafdir}")
  string(CONCAT _folder "Tests/" ${_leafdir})
  set_target_properties(${name} PROPERTIES FOLDER "${_folder}")
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${_OUTPUT_DIR})

  set(_TEST_PREFIX)
  if(NOT "${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(_TEST_PREFIX "${PROJECT_NAME}:")
  endif()

  flecsi_add_target_test("${_TEST_PREFIX}${name}"
    TARGET "${name}"
    INPUTS "${unit_INPUTS}"
    PROCS "${unit_PROCS}"
    ARGUMENTS "${unit_ARGUMENTS}"
    TESTLABELS "${unit_TESTLABELS}"
    LAUNCHER "${unit_LAUNCHER}" LAUNCHER_ARGUMENTS "${unit_LAUNCHER_ARGUMENTS}"
  )

endfunction()

function(flecsi_add_target_test name)
  if(NOT FLECSI_ENABLE_TESTING)
    return()
  endif()

  set(one_value_args TARGET LAUNCHER)
  set(multi_value_args
    INPUTS PROCS ARGUMENTS TESTLABELS LAUNCHER_ARGUMENTS
  )
  cmake_parse_arguments(unit "" "${one_value_args}"
    "${multi_value_args}" ${ARGN})

  _flecsi_get_test_output_dir(_OUTPUT_DIR)
  file(MAKE_DIRECTORY "${_OUTPUT_DIR}")

  if(NOT unit_TARGET)
      if(TARGET ${name})
        set(unit_TARGET "${name}")
      else()
        message(FATAL_ERROR "Target '${name}' doesn't exist and no TARGET defined.")
      endif()
  endif()

  if(unit_INPUTS)
    set(_OUTPUT_FILES)
    foreach(input ${unit_INPUTS})
      get_filename_component(_OUTPUT_NAME ${input} NAME)
      get_filename_component(_PATH ${input} ABSOLUTE)
      configure_file(${_PATH} ${_OUTPUT_DIR}/${_OUTPUT_NAME} COPYONLY)
      list(APPEND _OUTPUT_FILES ${_OUTPUT_DIR}/${_OUTPUT_NAME})
    endforeach()
    add_custom_target(${name}_inputs DEPENDS ${_OUTPUT_FILES})
    set_target_properties(${name}_inputs PROPERTIES FOLDER "${_folder}/Inputs")
    add_dependencies(${unit_TARGET} ${name}_inputs)
  endif()

  if(NOT unit_PROCS)
    set(unit_PROCS 1)
  endif()

  list(LENGTH unit_PROCS proc_instances)

  if(${proc_instances} EQUAL 1)
    set(unit_NAMES "${name}")
  else()
    set(unit_NAMES)
    foreach(instance ${unit_PROCS})
      list(APPEND unit_NAMES "${name}_${instance}")
    endforeach()
  endif()

  foreach(test procs IN ZIP_LISTS unit_NAMES unit_PROCS)
    add_test(
      NAME "${test}"
      COMMAND
        ${unit_LAUNCHER}
        ${unit_LAUNCHER_ARGUMENTS}
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_PREFLAGS}
        ${MPIEXEC_NUMPROC_FLAG} ${procs}
        $<TARGET_FILE:${unit_TARGET}>
        ${MPIEXEC_POSTFLAGS}
        ${unit_ARGUMENTS}
      WORKING_DIRECTORY "${_OUTPUT_DIR}"
    )
    set_tests_properties("${test}" PROPERTIES LABELS "${unit_TESTLABELS}")
  endforeach()
endfunction()
