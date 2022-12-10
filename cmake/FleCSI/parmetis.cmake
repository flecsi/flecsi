macro(flecsi_enable_parmetis target)
  set(PARMETIS_TEST_RUNS TRUE)
  find_package(ParMETIS REQUIRED)
  target_link_libraries(${target} PUBLIC ParMetis::ParMetis)

  # Some(?) versions of ParMETIS rely on GKlib
  find_package(GKlib QUIET)
  if(GKlib_FOUND)
    target_link_libraries(${target} PUBLIC GKlib)
  endif()
endmacro()
