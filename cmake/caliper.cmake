macro(flecsi_enable_caliper target)
  find_package(caliper REQUIRED)

  message(STATUS "Found Caliper: ${caliper_INCLUDE_DIR}")

  target_include_directories(${target} SYSTEM PUBLIC ${caliper_INCLUDE_DIR})
  target_link_libraries(${target} PUBLIC caliper)
endmacro()
