macro(flecsi_enable_hpx target)

  find_package(HPX 1.8.1 REQUIRED NO_CMAKE_PACKAGE_REGISTRY)

  target_compile_definitions(${target} PUBLIC ENABLE_HPX)
  target_include_directories(${target} SYSTEM PUBLIC ${HPX_INCLUDE_DIRS})
  target_link_libraries(${target} PUBLIC HPX::hpx)

  if(MSVC)
    target_compile_definitions(${target} PUBLIC
      _SCL_SECURE_NO_WARNINGS
      NOMINMAX
    )
  endif()

endmacro()
