macro(flecsi_enable_hpx target)

  find_package(HPX 1.10.0 REQUIRED NO_CMAKE_PACKAGE_REGISTRY)

  target_link_libraries(${target} PUBLIC HPX::hpx)

endmacro()
