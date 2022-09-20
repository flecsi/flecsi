macro(flecsi_enable_testing)
  if(NOT FleCSI_ENABLE_FLOG)
    message(FATAL_ERROR "Unit tests require FleCSI with FLOG enabled")
  endif()

  enable_testing()
  set(FLECSI_ENABLE_TESTING ON)
endmacro()

function(_flecsi_define_unit_main_target)
  if(NOT TARGET flecsi-unit-main)
    add_library(flecsi-unit-main OBJECT ${FLECSI_UNIT_MAIN})
    target_link_libraries(flecsi-unit-main PUBLIC FleCSI::FleCSI)
  endif()
endfunction()

function(flecsi_add_test name)

  #----------------------------------------------------------------------------#
  # Enable new behavior for in-list if statements.
  #----------------------------------------------------------------------------#

  cmake_policy(SET CMP0057 NEW)

  if(NOT FLECSI_ENABLE_TESTING)
    return()
  endif()

  _flecsi_define_unit_main_target()

  #----------------------------------------------------------------------------#
  # Setup argument options.
  #----------------------------------------------------------------------------#

  set(options)
  set(one_value_args POLICY)
  set(multi_value_args
    SOURCES INPUTS PROCS LIBRARIES DEFINES DRIVER ARGUMENTS TESTLABELS
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
  # Make sure that MPI_LANGUAGE is set.
  # This is not a standard variable set by FindMPI, but we might set it.
  #
  # Right now, the MPI policy only works with C/C++.
  #----------------------------------------------------------------------------#

  if(NOT MPI_LANGUAGE)
    set(MPI_LANGUAGE C)
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
  # Check to see if the user has specified a backend and process it.
  #----------------------------------------------------------------------------#

  if(FleCSI_BACKEND STREQUAL "mpi")

    set(unit_policy_flags ${MPI_${MPI_LANGUAGE}_COMPILE_FLAGS})
    set(unit_policy_includes ${MPI_${MPI_LANGUAGE}_INCLUDE_PATH})
    set(unit_policy_libraries ${MPI_${MPI_LANGUAGE}_LIBRARIES})
    set(unit_policy_exec ${MPIEXEC})
    set(unit_policy_exec_procs ${MPIEXEC_NUMPROC_FLAG})
    set(unit_policy_exec_preflags ${MPIEXEC_PREFLAGS})
    set(unit_policy_exec_postflags ${MPIEXEC_POSTFLAGS})

  elseif(FleCSI_BACKEND STREQUAL "legion")

    set(unit_policy_flags ${Legion_CXX_FLAGS}
      ${MPI_${MPI_LANGUAGE}_COMPILE_FLAGS})
    set(unit_policy_includes ${Legion_INCLUDE_DIRS}
      ${MPI_${MPI_LANGUAGE}_INCLUDE_PATH})
    set(unit_policy_libraries ${Legion_LIBRARIES} ${Legion_LIB_FLAGS}
      ${MPI_${MPI_LANGUAGE}_LIBRARIES})
    set(unit_policy_exec ${MPIEXEC})
    set(unit_policy_exec_procs ${MPIEXEC_NUMPROC_FLAG})
    set(unit_policy_exec_preflags ${MPIEXEC_PREFLAGS})
    set(unit_policy_exec_postflags ${MPIEXEC_POSTFLAGS})

  else()

    message(WARNING "invalid backend")
    return()

  endif()

  #----------------------------------------------------------------------------#
  # Make sure that the user specified sources.
  #----------------------------------------------------------------------------#

  if(NOT unit_SOURCES)
    message(FATAL_ERROR
      "You must specify unit test source files using SOURCES")
  endif()

  #----------------------------------------------------------------------------#
  # Set file properties for FleCSI language files.
  #----------------------------------------------------------------------------#

  foreach(source ${unit_SOURCES})
    # Identify FleCSI language source files and add the appropriate
    # language and compiler flags to properties.

    get_filename_component(_EXT ${source} EXT)

    if("${_EXT}" STREQUAL ".fcc")
      set_source_files_properties(${source} PROPERTIES LANGUAGE CXX
    )
    endif()
  endforeach()

  add_executable(${name}
    ${unit_SOURCES}
  )
  
  set_target_properties(${name}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${_OUTPUT_DIR})

  if(unit_policy_flags)
    set( unit_policy_list ${unit_policy_flags} )
    separate_arguments(unit_policy_list)

    target_compile_options(${name} PRIVATE ${unit_policy_list})
  endif()

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

  if(unit_policy_defines)
    target_compile_definitions(${name} PRIVATE ${unit_policy_defines})
  endif()

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
  target_link_libraries(${name} PRIVATE ${CMAKE_THREAD_LIBS_INIT})

  if(unit_policy_libraries)
    target_link_libraries(${name} PRIVATE ${unit_policy_libraries})
  endif()


  target_include_directories(${name} PRIVATE ${CMAKE_BINARY_DIR})

  if(unit_policy_includes)
      target_include_directories(${name}
          PRIVATE ${unit_policy_includes})
  endif()

  #----------------------------------------------------------------------------#
  # Check for procs
  #
  # If found, replace the semi-colons with pipes to avoid list
  # interpretation.
  #----------------------------------------------------------------------------#

  if(NOT unit_PROCS)
    set(unit_PROCS 1)
  endif()

  #----------------------------------------------------------------------------#
  # Add the test target to CTest
  #----------------------------------------------------------------------------#
  list(LENGTH unit_PROCS proc_instances)

  #we need to add -ll:gpu 1 to arguments if CUDA is enabled 
  if (FleCSI_ENABLE_KOKKOS AND FleCSI_ENABLE_LEGION AND Kokkos_ENABLE_CUDA)
   list(APPEND  UNIT_FLAGS "--backend-args=-ll:gpu 1") 
  endif()

  if(${proc_instances} GREATER 1)
    foreach(instance ${unit_PROCS})
      add_test(
        NAME
          "${_TEST_PREFIX}${name}_${instance}"
        COMMAND
          ${unit_policy_exec}
          ${unit_policy_exec_preflags}
          ${unit_policy_exec_procs} ${instance}
          $<TARGET_FILE:${name}>
          ${unit_ARGUMENTS}
          ${unit_policy_exec_postflags}
          ${UNIT_FLAGS}
        WORKING_DIRECTORY ${_OUTPUT_DIR}
      )

      set_tests_properties("${_TEST_PREFIX}${name}_${instance}"
        PROPERTIES LABELS "${unit_TESTLABELS}"
      )
    endforeach()
  else()
    if(unit_policy_exec)
      add_test(
        NAME
          "${_TEST_PREFIX}${name}"
        COMMAND
          ${unit_policy_exec}
          ${unit_policy_exec_procs}
          ${unit_PROCS}
          ${unit_policy_exec_preflags}
          $<TARGET_FILE:${name}>
          ${unit_ARGUMENTS}
          ${unit_policy_exec_postflags}
          ${UNIT_FLAGS}
        WORKING_DIRECTORY ${_OUTPUT_DIR}
      )
  else()
      add_test(
        NAME
          "${_TEST_PREFIX}${name}"
        COMMAND
          $<TARGET_FILE:${name}>
          ${UNIT_FLAGS}
        WORKING_DIRECTORY ${_OUTPUT_DIR}
      )
    endif()

    set_tests_properties("${_TEST_PREFIX}${name}" PROPERTIES
      LABELS "${unit_TESTLABELS}")

  endif()

endfunction()
