macro(flecsi_enable_parmetis target)
  find_package(METIS REQUIRED)

  set(PARMETIS_TEST_RUNS TRUE)
  find_package(ParMETIS REQUIRED)

  target_include_directories(${target} SYSTEM PUBLIC ${PARMETIS_INCLUDE_DIRS} ${METIS_INCLUDE_DIRS})
  target_link_libraries(${target} PUBLIC ${METIS_LIBRARIES} ${PARMETIS_LIBRARIES})
endmacro()
