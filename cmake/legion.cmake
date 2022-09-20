macro(flecsi_enable_legion target)
  find_package(Legion REQUIRED)
  target_link_libraries(${target} PUBLIC Legion::Legion)
endmacro()
