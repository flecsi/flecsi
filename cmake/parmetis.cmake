macro(flecsi_enable_parmetis target)
  set(PARMETIS_TEST_RUNS TRUE)
  find_package(ParMETIS REQUIRED)

  target_link_libraries(${target} PUBLIC ParMetis::ParMetis)
endmacro()
