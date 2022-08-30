macro(flecsi_enable_caliper)
  find_package(caliper REQUIRED)

  message(STATUS "Found Caliper: ${caliper_INCLUDE_DIR}")

  list(APPEND TPL_INCLUDES  ${caliper_INCLUDE_DIR})
  list(APPEND TPL_LIBRARIES caliper)
endmacro()
