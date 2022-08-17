option(ENABLE_PARMETIS "Enable ParMETIS" OFF)

if(ENABLE_PARMETIS)
  find_package(METIS REQUIRED)

  if(NOT METIS_FOUND)
    message(FATAL_ERROR "METIS is required for this configuration")
  endif()

  set(PARMETIS_TEST_RUNS TRUE)
  find_package(ParMETIS REQUIRED)

  if(NOT ParMETIS_FOUND)
    message(FATAL_ERROR "ParMETIS is required for this configuration")
  endif()

  list(APPEND TPL_INCLUDES ${PARMETIS_INCLUDE_DIRS} ${METIS_INCLUDE_DIRS})
  list(APPEND TPL_LIBRARIES ${METIS_LIBRARIES} ${PARMETIS_LIBRARIES})
endif()
